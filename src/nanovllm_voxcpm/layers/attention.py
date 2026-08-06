import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache, flash_attn_func
from src.nanovllm_voxcpm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(  # pragma: no cover – GPU Triton kernel, not executable on CPU
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )  # pragma: no cover – dispatches GPU Triton kernel


@triton.jit
def paged_attention_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    output_ptr,
    scale,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    block_table_stride_batch: tl.constexpr,
    block_table_stride_block: tl.constexpr,
    output_stride_batch: tl.constexpr,
    output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    TOKENS_PER_ITERATION: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // (NUM_Q_HEADS // NUM_KV_HEADS)
    dimensions = tl.arange(0, HEAD_DIM)
    query = tl.load(
        q_ptr
        + batch_index * q_stride_batch
        + query_head * q_stride_head
        + dimensions * q_stride_dim
    ).to(tl.float32)

    context_length = tl.load(context_lens_ptr + batch_index)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for token_start in tl.range(
        0,
        context_length,
        TOKENS_PER_ITERATION,
        num_stages=1,
    ):
        token_offsets = token_start + tl.arange(0, TOKENS_PER_ITERATION)
        token_mask = token_offsets < context_length
        logical_blocks = token_offsets // CACHE_BLOCK_SIZE
        block_offsets = token_offsets % CACHE_BLOCK_SIZE
        physical_blocks = tl.load(
            block_tables_ptr
            + batch_index * block_table_stride_batch
            + logical_blocks * block_table_stride_block,
            mask=token_mask,
            other=0,
        )
        cache_offsets = (
            physical_blocks[:, None] * cache_stride_block
            + block_offsets[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + dimensions[None, :] * cache_stride_dim
        )
        keys = tl.load(
            k_cache_ptr + cache_offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        values = tl.load(
            v_cache_ptr + cache_offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = (
            accumulator * old_scale
            + tl.sum(probabilities[:, None] * values, axis=0)
        )
        running_sum = (
            running_sum * old_scale + tl.sum(probabilities, axis=0)
        )
        running_max = new_max

    output = accumulator / tl.maximum(running_sum, 1.0e-6)
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + dimensions * output_stride_dim,
        output,
    )


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Graph-safe paged decode attention for pre-Ampere CUDA devices."""
    batch_size, num_query_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    if head_dim != k_cache.shape[3] or head_dim != v_cache.shape[3]:
        raise ValueError("Query and KV-cache head dimensions must match")
    if num_query_heads % num_kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    output = torch.empty_like(q)
    paged_attention_decode_kernel[(batch_size, num_query_heads)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        output,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_Q_HEADS=num_query_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        CACHE_BLOCK_SIZE=k_cache.shape[1],
        TOKENS_PER_ITERATION=32,
    )
    return output


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        is_causal: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.is_causal = is_causal

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size(1) != k.size(1):
            repeats = q.size(1) // k.size(1)
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            is_causal=is_causal,
        ).transpose(1, 2).contiguous()

    def _sdpa_causal(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                raise RuntimeError(
                    "V100 SDPA fallback does not support prefix-cache prefill"
                )
            q_offsets = context.cu_seqlens_q.tolist()
            k_offsets = context.cu_seqlens_k.tolist()
            outputs = []
            for index in range(len(q_offsets) - 1):
                q_item = q[q_offsets[index] : q_offsets[index + 1]].unsqueeze(0)
                k_item = k[k_offsets[index] : k_offsets[index + 1]].unsqueeze(0)
                v_item = v[k_offsets[index] : k_offsets[index + 1]].unsqueeze(0)
                outputs.append(
                    self._sdpa(q_item, k_item, v_item, is_causal=True).squeeze(0)
                )
            return torch.cat(outputs, dim=0)

        return paged_attention_decode(
            q,
            k_cache,
            v_cache,
            context.block_tables,
            context.context_lens,
            self.scale,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        if torch.cuda.get_device_capability(q.device)[0] < 8:
            if self.is_causal:
                return self._sdpa_causal(q, k, v)
            return self._sdpa(q, k, v, is_causal=False)

        if self.is_causal:
            context = get_context()
            k_cache, v_cache = self.k_cache, self.v_cache
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if context.is_prefill:
                if context.block_tables is not None:  # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:  # decode
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
        else:
            # non causal attention, no kvcache required
            o = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=False)
        return o
