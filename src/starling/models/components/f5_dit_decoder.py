"""F5-style DiT estimator behind the native Matcha CFM interface.

This module intentionally does not copy F5-TTS's text/audio prompt input
interface or classifier-free guidance path. Matcha already expands text into a
frame-level prior through MAS; this estimator only replaces the internal
velocity-prediction network used by CFM.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        if x.ndim == 0:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1).to(dtype=x.dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.time_embed(timestep))


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConvPositionEmbedding requires an odd kernel size")
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.masked_layer_indices = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.masked_layer_indices:
                x = x.masked_fill(~mask, 0.0)
        return x.transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(dtype=self.weight.dtype) * self.weight


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def rotary_cos_sin(seq_len: int, dim_head: int, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head, 2, device=device).float() / dim_head))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("n,d->nd", positions, inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    cos = emb.cos()[None, None, :, :].to(dtype=dtype)
    sin = emb.sin()[None, None, :, :].to(dtype=dtype)
    return cos, sin


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.dropout = dropout
        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)
        else:
            raise ValueError(f"Unsupported qk_norm={qk_norm}")
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.to_q(x).view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        cos, sin = rotary_cos_sin(seq_len, self.dim_head, x.device, q.dtype)
        if self.pe_attn_head is None:
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        elif self.pe_attn_head > 0:
            q[:, : self.pe_attn_head] = apply_rotary_pos_emb(q[:, : self.pe_attn_head], cos, sin)
            k[:, : self.pe_attn_head] = apply_rotary_pos_emb(k[:, : self.pe_attn_head], cos, sin)

        attn_mask = None
        if self.attn_mask_enabled and mask is not None:
            attn_mask = mask[:, None, None, :].expand(batch_size, self.heads, seq_len, seq_len)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)
        x = self.to_out(x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float = 4,
        dropout: float = 0.1,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = False,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_mask_enabled=attn_mask_enabled,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm, mask=mask)

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(norm)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class MatchaF5DiTDecoder(nn.Module):
    """F5-style DiT decoder preserving Matcha's estimator interface."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 512,
        depth: int = 16,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.05,
        ff_mult: float = 2,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = False,
        conv_pos_embed: bool = True,
        conv_pos_kernel_size: int = 31,
        long_skip_connection: bool = True,
        checkpoint_activations: bool = False,
        global_cond_dim: int | None = None,
        global_cond_scale: float = 1.0,
        prompt_mel_condition: bool = False,
    ):
        super().__init__()
        if dim != heads * dim_head:
            raise ValueError(f"dim must equal heads * dim_head, got dim={dim}, heads={heads}, dim_head={dim_head}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.depth = depth
        self.checkpoint_activations = checkpoint_activations
        self.global_cond_scale = float(global_cond_scale)
        self.prompt_mel_condition = bool(prompt_mel_condition)

        self.input_proj = nn.Linear(in_channels, dim)
        self.conv_pos_embed = (
            ConvPositionEmbedding(dim=dim, kernel_size=conv_pos_kernel_size) if conv_pos_embed else None
        )
        self.time_embed = TimestepEmbedding(dim)
        self.global_cond_proj = nn.Linear(global_cond_dim, dim, bias=False) if global_cond_dim else None
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNormFinal(dim)
        self.proj_out = nn.Linear(dim, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def _checkpoint_block(self, block, x, t, mask):
        return torch.utils.checkpoint.checkpoint(block, x, t, mask, use_reentrant=False)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        batch_size = x.shape[0]
        if t.ndim == 0:
            t = t.repeat(batch_size)
        elif t.ndim > 1:
            t = t.reshape(batch_size)

        inputs = [x, mu]
        if self.prompt_mel_condition:
            prompt_x = cond.get("prompt_x") if isinstance(cond, dict) else None
            if prompt_x is None:
                prompt_x = torch.zeros_like(x)
            if prompt_x.shape != x.shape:
                raise ValueError(f"Expected prompt_x shape {tuple(x.shape)}, got {tuple(prompt_x.shape)}")
            inputs.append(prompt_x.to(device=x.device, dtype=x.dtype))

        h = torch.cat(inputs, dim=1)
        global_cond = spks
        if spks is not None:
            spks = spks.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = torch.cat((h, spks), dim=1)
        if h.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {h.shape[1]}")

        h = (h * mask).transpose(1, 2)
        mask_bool = mask.squeeze(1).bool()

        h = self.input_proj(h)
        if self.conv_pos_embed is not None:
            h = h + self.conv_pos_embed(h, mask=mask_bool)
        h = h.masked_fill(~mask_bool.unsqueeze(-1), 0.0)

        time = self.time_embed(t)
        if self.global_cond_proj is not None:
            if global_cond is None:
                raise ValueError("global_cond_dim is configured but no speaker condition was provided")
            time = time + self.global_cond_proj(global_cond.to(device=time.device, dtype=time.dtype)) * self.global_cond_scale
        residual = h
        for block in self.transformer_blocks:
            if self.checkpoint_activations and self.training:
                h = self._checkpoint_block(block, h, time, mask_bool)
            else:
                h = block(h, time, mask=mask_bool)

        if self.long_skip_connection is not None:
            h = self.long_skip_connection(torch.cat((h, residual), dim=-1))
            h = h.masked_fill(~mask_bool.unsqueeze(-1), 0.0)

        h = self.norm_out(h, time)
        out = self.proj_out(h).transpose(1, 2)
        return out * mask
