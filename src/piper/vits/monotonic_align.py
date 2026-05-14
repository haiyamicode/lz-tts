"""Adapter for the monotonic-align package used by VITS MAS."""

from __future__ import annotations

import numpy as np
import torch
from monotonic_align.core import maximum_path_c


def maximum_path(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MAS path for tensors shaped [batch, text_time, audio_time]."""

    value = value * mask
    device = value.device
    dtype = value.dtype
    values = value.detach().cpu().numpy().astype(np.float32)
    paths = np.zeros_like(values, dtype=np.int32)
    mask_np = mask.detach().cpu().numpy()

    t_xs = mask_np.sum(1)[:, 0].astype(np.int32)
    t_ys = mask_np.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(paths, values, t_xs, t_ys)
    return torch.from_numpy(paths).to(device=device, dtype=dtype)
