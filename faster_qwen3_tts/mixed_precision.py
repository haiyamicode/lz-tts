"""Selective mixed-precision helpers for inference."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


class InnerDtypeLinear(nn.Module):
    """Run a Linear layer in inner_dtype and cast the result back.

    This is useful on V100-class GPUs: keeping residual activations and caches in
    fp32 preserves generation stability, while selected large GEMMs can use fp16
    Tensor Cores.
    """

    def __init__(self, linear: nn.Linear, inner_dtype: torch.dtype):
        super().__init__()
        self.inner_dtype = inner_dtype
        self.weight = nn.Parameter(
            linear.weight.detach().to(inner_dtype),
            requires_grad=False,
        )
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(
                linear.bias.detach().to(inner_dtype),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x.to(self.inner_dtype), self.weight, self.bias)
        return output.to(x.dtype)


class FP16LayerIsland(nn.Module):
    """Run a whole decoder layer in fp16 and cast hidden states back.

    This keeps the residual stream/caller-visible activations in fp32 while using
    fp16 for the expensive layer internals. The caller must also set that layer's
    KV cache dtype to fp16.
    """

    cache_dtype = torch.float16

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer.to(dtype=torch.float16)

    @property
    def attention_type(self):
        return self.layer.attention_type

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        input_dtype = hidden_states.dtype
        kwargs = dict(kwargs)
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            kwargs["position_embeddings"] = tuple(
                item.to(torch.float16) for item in position_embeddings
            )
        output = self.layer(hidden_states.to(torch.float16), *args, **kwargs)
        if isinstance(output, tuple):
            return (output[0].to(input_dtype),) + output[1:]
        return output.to(input_dtype)


class TensorFP16Island(nn.Module):
    """Run a single-input tensor module in fp16 and cast outputs back."""

    def __init__(self, module: nn.Module, output_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module.to(dtype=torch.float16)
        self.output_dtype = output_dtype

    def forward(self, *args, **kwargs):
        input_dtype = self.output_dtype

        def cast_input(value):
            nonlocal input_dtype
            if torch.is_tensor(value) and value.is_floating_point():
                if input_dtype is None:
                    input_dtype = value.dtype
                return value.to(torch.float16)
            return value

        output = self.module(
            *[cast_input(arg) for arg in args],
            **{key: cast_input(value) for key, value in kwargs.items()},
        )
        output_dtype = input_dtype or torch.float32
        return _cast_floating_output(output, output_dtype)


def _cast_floating_output(value, dtype: torch.dtype):
    if torch.is_tensor(value) and value.is_floating_point():
        return value.to(dtype)
    if isinstance(value, tuple):
        return tuple(_cast_floating_output(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_floating_output(item, dtype) for item in value]
    return value


def replace_linear_modules(
    module: nn.Module,
    predicate: Callable[[str, nn.Linear], bool],
    *,
    inner_dtype: torch.dtype,
) -> int:
    """Replace matching nn.Linear children recursively.

    Returns the number of replaced modules.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and predicate(name, child):
            setattr(module, name, InnerDtypeLinear(child, inner_dtype))
            count += 1
        else:
            count += replace_linear_modules(
                child,
                predicate,
                inner_dtype=inner_dtype,
            )
    return count


def _parse_layer_selector(spec: str, n_layers: int) -> Callable[[int], bool]:
    spec = spec.strip().lower()
    if spec in ("none", "off", "disabled"):
        return lambda _idx: False
    if spec == "all":
        return lambda _idx: True
    if spec == "even":
        return lambda idx: idx % 2 == 0
    if spec == "odd":
        return lambda idx: idx % 2 == 1
    if spec.startswith("last"):
        count = int(spec[4:])
        return lambda idx: idx >= n_layers - count
    if spec.startswith("first"):
        count = int(spec[5:])
        return lambda idx: idx < count
    if "-" in spec:
        start, end = [int(part) for part in spec.split("-", 1)]
        return lambda idx: start <= idx <= end
    selected = {int(part) for part in spec.split(",") if part}
    return lambda idx: idx in selected


def wrap_decoder_layers(module: nn.Module, spec: str) -> list[int]:
    """Wrap selected decoder layers as fp16 islands.

    `module` is expected to expose a `layers` ModuleList.
    Returns the selected layer indices.
    """
    layers = module.layers
    selector = _parse_layer_selector(spec, len(layers))
    wrapped: list[int] = []
    for idx, layer in enumerate(layers):
        if selector(idx):
            layers[idx] = FP16LayerIsland(layer)
            wrapped.append(idx)
    return wrapped


def wrap_tokenizer_decoder_synthesis(decoder: nn.Module, output_dtype: torch.dtype = torch.float32) -> int:
    """Wrap the tokenizer decoder's post-transformer synthesis blocks.

    This intentionally leaves the RVQ quantizer in the caller's dtype. The tested
    V100-safe path is fp32 quantizer output, fp16 conv/upsample/decoder internals,
    and fp32 tensors between synthesis blocks.

    Returns the number of wrapped modules.
    """
    count = 0
    decoder.pre_conv = TensorFP16Island(decoder.pre_conv, output_dtype=output_dtype)
    count += 1

    for blocks in decoder.upsample:
        for idx, block in enumerate(blocks):
            blocks[idx] = TensorFP16Island(block, output_dtype=output_dtype)
            count += 1

    for idx, block in enumerate(decoder.decoder):
        decoder.decoder[idx] = TensorFP16Island(block, output_dtype=output_dtype)
        count += 1

    return count
