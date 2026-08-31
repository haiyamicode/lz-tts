"""Low-rank convolution adapters for Sparrow voice specialization."""

from __future__ import annotations

import fnmatch
import math
from typing import Iterable, Sequence

import torch
from torch import nn


DEFAULT_VOICE_ADAPTER_TARGETS = (
    "enc_p.encoder.spk_emb_linear",
    "dec.cond",
    "flow.flows.0.enc.cond_layer",
    "flow.flows.2.enc.cond_layer",
    "flow.flows.4.enc.cond_layer",
    "flow.flows.6.enc.cond_layer",
    "sdp.cond",
    "dp.cond",
)


class LoRAConv1d(nn.Module):
    """Add a trainable low-rank residual to a frozen Conv1d."""

    def __init__(
        self,
        base: nn.Conv1d,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Conv1d):
            raise TypeError(f"Expected Conv1d, got {type(base).__name__}")
        if base.groups != 1:
            raise ValueError("Grouped Conv1d targets are not supported")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Conv1d(
            base.in_channels,
            self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.lora_b = nn.Conv1d(self.rank, base.out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_b(self.lora_a(self.dropout(inputs)))
        return self.base(inputs) + residual * self.scale


class LoRAConvTranspose1d(nn.Module):
    """Add a trainable low-rank residual to a frozen ConvTranspose1d."""

    def __init__(
        self,
        base: nn.ConvTranspose1d,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.ConvTranspose1d):
            raise TypeError(f"Expected ConvTranspose1d, got {type(base).__name__}")
        if base.groups != 1:
            raise ValueError("Grouped ConvTranspose1d targets are not supported")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.ConvTranspose1d(
            base.in_channels,
            self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            output_padding=base.output_padding,
            groups=1,
            bias=False,
            dilation=base.dilation,
            padding_mode=base.padding_mode,
        )
        self.lora_b = nn.Conv1d(self.rank, base.out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        inputs: torch.Tensor,
        output_size: list[int] | None = None,
    ) -> torch.Tensor:
        base_output = self.base(inputs, output_size=output_size)
        dropped = self.dropout(inputs)
        if output_size is None:
            projected = self.lora_a(dropped)
        else:
            projected = self.lora_a(
                dropped,
                output_size=[inputs.size(0), self.rank, base_output.size(-1)],
            )
        residual = self.lora_b(projected)
        return base_output + residual * self.scale


_SUPPORTED_CONVOLUTIONS = (nn.Conv1d, nn.ConvTranspose1d)
_LORA_CONVOLUTIONS = (LoRAConv1d, LoRAConvTranspose1d)


def unwrap_lora_base(module: nn.Module) -> nn.Module:
    """Return the frozen convolution wrapped by a voice LoRA, if present."""

    if isinstance(module, _LORA_CONVOLUTIONS):
        return module.base
    return module


def _expand_target_modules(
    model: nn.Module, target_modules: Sequence[str]
) -> tuple[str, ...]:
    named_modules = dict(model.named_modules())
    supported_names = tuple(
        name
        for name, module in named_modules.items()
        if name and isinstance(module, _SUPPORTED_CONVOLUTIONS)
    )

    expanded: list[str] = []
    installed: set[str] = set()
    seen_specs: set[str] = set()
    for raw_target in target_modules:
        target = str(raw_target).strip()
        if not target or target in seen_specs:
            raise ValueError(f"Invalid or duplicate voice adapter target: {target!r}")
        seen_specs.add(target)

        if any(character in target for character in "*?["):
            matches = [
                name for name in supported_names if fnmatch.fnmatchcase(name, target)
            ]
            if not matches:
                raise ValueError(
                    f"Voice adapter target pattern matched no convolutions: {target}"
                )
        else:
            module = named_modules.get(target)
            if module is None:
                raise ValueError(f"Voice adapter target does not exist: {target}")
            if isinstance(module, _LORA_CONVOLUTIONS):
                raise ValueError(f"Voice adapter target is already wrapped: {target}")
            if not isinstance(module, _SUPPORTED_CONVOLUTIONS):
                raise TypeError(
                    f"Voice adapter target {target} must be Conv1d or "
                    f"ConvTranspose1d, got {type(module).__name__}"
                )
            matches = [target]

        for match in matches:
            if match not in installed:
                installed.add(match)
                expanded.append(match)

    return tuple(expanded)


def install_conditioning_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[str, ...]:
    """Replace exact or glob-selected convolution paths with LoRA wrappers."""

    installed = _expand_target_modules(model, target_modules)
    for target in installed:
        parent_path, separator, child_name = target.rpartition(".")
        if not separator:
            raise ValueError(f"Voice adapter target must be a dotted path: {target}")
        parent = model.get_submodule(parent_path)
        base = getattr(parent, child_name)
        wrapper_type = (
            LoRAConvTranspose1d
            if isinstance(base, nn.ConvTranspose1d)
            else LoRAConv1d
        )
        setattr(parent, child_name, wrapper_type(base, rank, alpha, dropout))

    return installed


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA tensors and the dedicated adapter speaker vector."""

    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if name == "voice_adapter_embedding" or any(
            marker in name for marker in (".lora_a.", ".lora_b.")
        ):
            state[name] = value.detach().cpu().contiguous()
    return state


def adapter_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def source_key_for_wrapped_target(
    target_key: str, target_modules: Iterable[str]
) -> str:
    """Map a wrapped target state key back to its unwrapped base checkpoint key."""

    for target in target_modules:
        wrapped_prefix = f"{target}.base."
        if target_key.startswith(wrapped_prefix):
            return f"{target}.{target_key.removeprefix(wrapped_prefix)}"
    return target_key
