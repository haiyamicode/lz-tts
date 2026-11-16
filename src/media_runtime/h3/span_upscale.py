"""H3 ``upscale`` (vigen 43c32e7): SPAN 2x super-resolution, then lanczos to the exact size.

The frames go through a 2x SPAN model ("2x NomosUni span multijpg ldl" by Helaman (Philip Hofmann), CC-BY-4.0,
https://openmodeldb.info/models/2x-NomosUni-span-multijpg-ldl; ``data/span-upscale``, converted to safetensors, see
its NOTICE.txt) and ffmpeg's lanczos then resamples them to the size asked for: the size the job delivers without
``upscale`` times its factor (:func:`target_size`).

The model is small (2.2M parameters) and only a job that asks for ``upscale`` needs it, so it is never kept:
:func:`loaded_span` loads it for one job's frames and frees it, and its GPU memory, as soon as they are done.
"""
from __future__ import annotations

import gc
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .compat import SPAN_UPSCALE_MODEL_PATH, log_event

#: SPAN's own factor; anything else is lanczos.
SPAN_SCALE = 2
#: ``upscale: true``; a requested factor is above 1 and at most SPAN's own 2.
DEFAULT_UPSCALE = 1.5
MAX_UPSCALE = 2.0
#: Longest side delivered, whatever the factor: 4K (2160x3840 for a 1080x1920 portrait at 2x).
MAX_LONG_SIDE = 3840
#: The model's name in results.
SPAN_MODEL_NAME = "2xNomosUni_span_multijpg_ldl"
#: A frame goes through in horizontal strips of at most this many rows, each with ``STRIP_OVERLAP`` rows of
#: context above and below that are cut off again: a 768x1344 frame peaks at 0.5 GiB instead of 1.9 GiB whole
#: and matches the whole frame to within one 8-bit level (vigen, RTX PRO 6000).
STRIP_ROWS = 384
STRIP_OVERLAP = 24


def requested_scale(payload: dict[str, Any]) -> float | None:
    """The factor a job's ``upscale`` asks for (``true`` or a number), or ``None``."""
    value = payload.get("upscale")
    if value is None or value is False:
        return None
    scale = DEFAULT_UPSCALE if value is True else float(value)
    if not 1.0 < scale <= MAX_UPSCALE:
        raise ValueError(f"upscale must be above 1 and at most {MAX_UPSCALE:g}, got {scale:g}")
    return scale


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def target_size(width: int, height: int, scale: float, *,
                bounding_box: tuple[int, int] | None = None) -> tuple[int, int]:
    """``width`` x ``height`` times ``scale``, fitted inside ``bounding_box`` (``output_dimensions``) and
    :data:`MAX_LONG_SIDE`, aspect kept, on even numbers."""
    w, h = width * float(scale), height * float(scale)
    ratio = min(1.0, MAX_LONG_SIDE / max(w, h))
    if bounding_box is not None:
        ratio = min(ratio, bounding_box[0] / w, bounding_box[1] / h)
    return _even(w * ratio), _even(h * ratio)


class SpanUpscaler:
    """One loaded SPAN model: RGB uint8 frames in, frames twice the size out, one frame at a time in strips."""

    #: Frames handed over per call (they still go through one at a time).
    batch = 4

    def __init__(self, descriptor: Any, device: Any, dtype: Any) -> None:
        self.descriptor = descriptor
        self.device = device
        self.dtype = dtype
        self.frames = 0
        self.seconds = 0.0

    def _frame(self, frame: Any) -> Any:
        import numpy as np
        import torch

        x = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).to(self.device).to(self.dtype)
        x = x.unsqueeze(0) / 255.0
        height = x.shape[2]
        strips = -(-height // STRIP_ROWS)
        step = -(-height // strips)
        parts = []
        for index in range(strips):
            top, bottom = index * step, min(height, (index + 1) * step)
            low, high = max(0, top - STRIP_OVERLAP), min(height, bottom + STRIP_OVERLAP)
            y = self.descriptor(x[:, :, low:high])
            parts.append(y[:, :, SPAN_SCALE * (top - low): SPAN_SCALE * (bottom - low)])
        out = torch.cat(parts, 2).clamp_(0, 1).mul_(255).round_().to(torch.uint8)
        return out[0].permute(1, 2, 0).cpu().numpy()

    def __call__(self, frames: list[Any]) -> list[Any]:
        import torch

        began = time.perf_counter()
        with torch.inference_mode():
            out = [self._frame(frame) for frame in frames]
        self.frames += len(frames)
        self.seconds += time.perf_counter() - began
        return out


@contextmanager
def loaded_span(*, job_id: str = "") -> Iterator[SpanUpscaler]:
    """The SPAN model, loaded for the block and freed when it ends (fp16 on CUDA, fp32 on the CPU). On the GPU it
    first makes room beside H3's weight pages (:func:`h3.minimax_h3_runtime.make_room_beside_h3`)."""
    import spandrel
    import torch

    path = SPAN_UPSCALE_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"the SPAN upscale model is missing: {path}")
    began = time.perf_counter()
    cuda = torch.cuda.is_available()
    if cuda:
        from .minimax_h3_runtime import make_room_beside_h3

        make_room_beside_h3()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float16 if cuda else torch.float32
    descriptor = spandrel.ModelLoader().load_from_file(str(path))
    if descriptor.scale != SPAN_SCALE:
        raise ValueError(f"{path.name} upscales {descriptor.scale}x, not {SPAN_SCALE}x")
    descriptor.to(device).eval()
    if cuda:
        descriptor.half()
    upscaler = SpanUpscaler(descriptor, device, dtype)
    log_event("span_upscale_loaded", job_id=job_id, model=SPAN_MODEL_NAME, device=str(device),
              seconds=round(time.perf_counter() - began, 3))
    try:
        yield upscaler
    finally:
        upscaler.descriptor = None
        del descriptor
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        log_event("span_upscale_unloaded", job_id=job_id, frames=upscaler.frames, seconds=round(upscaler.seconds, 3))
