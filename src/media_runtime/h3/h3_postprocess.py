"""An H3 render's frames on their way to the delivered video (vigen 43c32e7): RIFE up to 24 fps
(``use_interpolation``, :mod:`h3.h3_interpolation`) and SPAN 2x (``upscale``, :mod:`h3.span_upscale`), streamed
between two ffmpeg processes so every frame is encoded once.

:func:`stream_video` reads the frames ffmpeg makes of the source, makes each output frame - on a rendered instant
that frame untouched, between two one RIFE call at its exact timestep, past the last the last held - runs them
through SPAN when asked, and hands them to the encoding ffmpeg, whose ``output_args`` do the rest (the lanczos
resize to the exact size, the audio, the codec).
"""
from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Iterator
from contextlib import nullcontext
from fractions import Fraction
from typing import Any

import cancel

from .h3_interpolation import OUTPUT_FPS


def _read_frames(process: subprocess.Popen, width: int, height: int) -> Iterator[Any]:
    import numpy as np

    size = width * height * 3
    assert process.stdout is not None
    while True:
        data = process.stdout.read(size)
        if len(data) < size:
            return
        yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def stream_video(source_args: list[str], output_args: list[str], *, width: int, height: int, source_fps: int,
                 frames: int, output_fps: int = OUTPUT_FPS, upscale: bool = False, job_id: str = "") -> dict[str, Any]:
    """``frames`` frames at ``output_fps`` from the ``source_fps`` video ffmpeg makes of ``source_args`` (its inputs,
    filters and maps; RGB ``width``x``height`` frames). Output frame ``n`` sits at ``n * source_fps / output_fps``
    source frames: on a whole one it is that frame untouched, between two it is one RIFE call at that exact
    timestep (RIFE is loaded only when a frame falls between two), past the last it holds the last. ``upscale``
    sends every output frame through SPAN 2x, loaded for this call only. The frames go to ffmpeg as its first input
    (``-i`` raw video, ``2*width``x``2*height`` when upscaled, then ``output_args``: more inputs, maps, filters - the
    lanczos resize - codecs and the output path), the only encode."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from .span_upscale import SPAN_MODEL_NAME, SPAN_SCALE, loaded_span

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_h, pad_w = math.ceil(height / 128) * 128, math.ceil(width / 128) * 128
    rife: list[Any] = []

    def interpolate(first: Any, second: Any, timestep: float) -> Any:
        if not rife:
            from .h3_interpolation import rife_model
            from .minimax_h3_runtime import make_room_beside_h3

            make_room_beside_h3()
            rife.append(rife_model())
        mid = rife[0].inference(first, second, timestep=timestep, scale=1.0)
        return (mid[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0)[:height, :width].cpu().numpy()

    def tensor(frame: Any) -> Any:
        t = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).to(device).unsqueeze(0).float() / 255.0
        return F.pad(t, (0, pad_w - width, 0, pad_h - height))

    out_w, out_h = (width * SPAN_SCALE, height * SPAN_SCALE) if upscale else (width, height)
    # Every source frame exactly once (vigen 8e559ef): left to its default, ffmpeg resamples raw video to the
    # stream's declared rate (an fps join of 24 fps renders still declares 24), duplicating frames and drifting
    # off the audio.
    reader = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *source_args, "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-r", str(output_fps), "-i", "-", *output_args],
        stdin=subprocess.PIPE,
    )
    assert writer.stdin is not None
    source = _read_frames(reader, width, height)
    # The two source frames around the current output frame, by index, and their tensors.
    window: dict[int, Any] = {}
    tensors: dict[int, Any] = {}
    loaded = -1
    exhausted = False
    rife_calls = 0
    span_metrics: dict[str, Any] | None = None
    try:
        with loaded_span(job_id=job_id) if upscale else nullcontext() as span, torch.inference_mode():
            pending: list[Any] = []

            def flush() -> None:
                for frame in (span(pending) if span is not None else pending):
                    writer.stdin.write(np.ascontiguousarray(frame).tobytes())
                pending.clear()

            for n in range(int(frames)):
                # The writer and reader are killed below, so a cancel stops the encode too.
                cancel.check()
                position = Fraction(n * int(source_fps), int(output_fps))
                index, offset = math.floor(position), position - math.floor(position)
                while not exhausted and loaded < index + (1 if offset else 0):
                    frame = next(source, None)
                    if frame is None:
                        exhausted = True
                        break
                    loaded += 1
                    window[loaded] = frame
                for stale in [key for key in window if key < min(index, loaded)]:
                    window.pop(stale)
                    tensors.pop(stale, None)
                if loaded < 0:
                    raise ValueError("the video to process has no frames")
                if index >= loaded or not offset:
                    pending.append(window[min(index, loaded)])
                else:
                    for key in (index, index + 1):
                        if key not in tensors:
                            tensors[key] = tensor(window[key])
                    pending.append(interpolate(tensors[index], tensors[index + 1], float(offset)))
                    rife_calls += 1
                if span is None or len(pending) >= span.batch:
                    flush()
            flush()
            if span is not None:
                span_metrics = {"model": SPAN_MODEL_NAME, "frames": span.frames, "seconds": round(span.seconds, 3)}
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("ffmpeg could not encode the processed video")
    finally:
        if writer.poll() is None:
            writer.kill()
        if reader.stdout is not None:
            reader.stdout.close()
        reader.kill()
        reader.wait()
        tensors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "source_fps": int(source_fps),
        "output_fps": int(output_fps),
        "source_frames": loaded + 1,
        "output_frames": int(frames),
        "rife_frames": rife_calls,
        **({"span": span_metrics} if span_metrics else {}),
        "seconds": round(time.perf_counter() - started, 3),
    }
