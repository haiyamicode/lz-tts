"""Reusable conditioning layers for Starling model components."""

from __future__ import annotations

import torch
from torch import nn


class FiLM1d(nn.Module):
    """Feature-wise linear modulation for tensors shaped ``[batch, channels, time]``.

    This follows the same scale/shift conditioning form used by the F5 DiT
    AdaLayerNorm blocks, but is kept as a small plug-in that can be attached to
    encoder and duration-predictor layers without changing their channel width.
    """

    def __init__(self, condition_dim: int, channels: int):
        super().__init__()
        if condition_dim <= 0:
            raise ValueError(f"condition_dim must be positive, got {condition_dim}")
        self.condition_dim = int(condition_dim)
        self.channels = int(channels)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(self.condition_dim, self.channels * 2)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is None:
            return x if mask is None else x * mask
        if condition.dim() != 2:
            raise ValueError(f"FiLM condition must be [batch, dim], got {tuple(condition.shape)}")
        if condition.shape[0] != x.shape[0] or condition.shape[1] != self.condition_dim:
            raise ValueError(
                "FiLM condition shape mismatch: "
                f"expected ({x.shape[0]}, {self.condition_dim}), got {tuple(condition.shape)}"
            )

        condition = condition.to(device=x.device, dtype=x.dtype)
        shift, scale = self.linear(self.silu(condition)).chunk(2, dim=1)
        x = x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return x if mask is None else x * mask
