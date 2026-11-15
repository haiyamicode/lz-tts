"""H3 at fewer frames a second, RIFE back up to its native 24 (``use_interpolation`` / ``interpolation_fps``).

H3 has no frame-rate input. Its RoPE time axis runs 40 units a second: an audio latent frame is 1 unit and a pixel
frame ``FRAME_RESCALE = 5/3`` (24 fps). Rendering ``F`` frames for every second of the output costs about ``F/24``
of the compute; RIFE v4.26 (``rife``, weights ``data/rife-v4.26``) then makes the frames in between
(vigen 7280e2f, local/experiments/h3-lowfps-20261114/REPORT.md there, RTX 5090):

- **With a voice** (lipsync, videogen with ``pinned_audio_url``) the voice is sped up ``24/F`` (``atempo``, pitch kept,
  :func:`speed_up`) and rendered at H3's own 24 fps; the frames are then played at ``F``. Positions stay where the
  model was trained and the lips follow the speech as at 24 fps: at 16 fps SyncNet 7.2 against 7.3-8.5 native and
  the mouth as sharp, 1.5x faster; at 12 fps sync holds (5.0-5.3) with a softer mouth, 2x faster.
- **Without one** (videogen making its own soundtrack) nothing can be sped up, so the time axis itself is set to
  ``F`` frames a second (:func:`h3_time_axis`): the clip and the audio it generates last ``frames / F`` seconds.

:func:`upsample_video` streams the low-rate frames through RIFE: an output frame on a rendered instant is that
frame untouched, any other is one RIFE call at its exact timestep, and everything is encoded once.
"""
from __future__ import annotations

import math
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

#: H3's own frame rate, the rate every interpolated job is delivered at.
OUTPUT_FPS = 24
DEFAULT_INTERPOLATION_FPS = 16
MIN_INTERPOLATION_FPS = 12
MAX_INTERPOLATION_FPS = 23
#: H3's audio latent frames a second, the unit of its time axis.
AUDIO_LATENT_FPS = 40


_RIFE: Any = None


def rife_model() -> Any:
    """RIFE v4.26, loaded once (~25 MB of weights) and kept for the life of the media runtime."""
    global _RIFE
    if _RIFE is None:
        import rife

        _RIFE = rife.load_model()
    return _RIFE


def interpolated_frame(frame: int, fps: int) -> int:
    """A 24 fps frame index (a pin) as the nearest frame of an ``fps`` render; never frame 0, the job's own image."""
    return max(1, round(int(frame) * int(fps) / OUTPUT_FPS))


def interpolated_render_frames(frames: int, fps: int) -> int:
    """The frames an ``fps`` render needs to last as long as ``frames`` at 24 fps."""
    return -(-int(frames) * int(fps) // OUTPUT_FPS)


def requested_fps(payload: dict[str, Any]) -> int | None:
    """The frame rate a job asks H3 to render at (``use_interpolation``, ``interpolation_fps``), or ``None``."""
    fps = payload.get("interpolation_fps")
    if not payload.get("use_interpolation"):
        if fps is not None:
            raise ValueError("interpolation_fps needs use_interpolation")
        return None
    fps = int(fps) if fps is not None else DEFAULT_INTERPOLATION_FPS
    if not MIN_INTERPOLATION_FPS <= fps <= MAX_INTERPOLATION_FPS:
        raise ValueError(f"interpolation_fps must be {MIN_INTERPOLATION_FPS} to {MAX_INTERPOLATION_FPS}, got {fps}")
    return fps


def speed_up(audio_path: Path, output_path: Path, *, fps: int, duration_seconds: float) -> Path:
    """``audio_path`` sped up ``24/fps`` with its pitch kept, padded or cut to ``duration_seconds``: the voice an
    ``fps`` render hears on H3's own 24 fps time axis."""
    factor = OUTPUT_FPS / int(fps)
    if not 1.0 <= factor <= 2.0:  # one atempo stage, the range it keeps the pitch cleanly
        raise ValueError(f"cannot speed audio up {factor:g}x for a {fps} fps render")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(audio_path), "-vn",
            "-af", f"atempo={factor:.6f},apad=whole_dur={duration_seconds:.6f}",
            "-t", f"{duration_seconds:.6f}", "-c:a", "pcm_s16le", str(output_path),
        ],
        check=True,
    )
    return output_path


@contextmanager
def h3_time_axis(fps: int | float) -> Iterator[None]:
    """H3 renders ``fps`` frames a second inside this block: ``FRAME_RESCALE = 40 / fps`` in the model and the H3
    node (which imports its own copy), and the node's ``FPS``, which sets how long the generated audio lasts. The
    media runtime renders one job at a time."""
    if int(fps) == OUTPUT_FPS:
        yield
        return
    import comfy.ldm.minimax.model as h3_model  # type: ignore

    # ComfyUI registers its extra nodes under their file path, not ``comfy_extras.nodes_minimax_h3``: an import by
    # package name is a second copy the graph never runs, so patch every loaded copy of the node module.
    nodes = [module for name, module in list(sys.modules.items())
             if name.rsplit("/", 1)[-1].rsplit(".", 1)[-1] == "nodes_minimax_h3" and hasattr(module, "FPS")]
    if not nodes:
        raise RuntimeError("the H3 node module is not loaded")
    saved = [(module, module.FRAME_RESCALE, module.FPS) for module in nodes]
    rescale = h3_model.FRAME_RESCALE
    h3_model.FRAME_RESCALE = AUDIO_LATENT_FPS / int(fps)
    for module in nodes:
        module.FRAME_RESCALE, module.FPS = AUDIO_LATENT_FPS / int(fps), int(fps)
    try:
        yield
    finally:
        h3_model.FRAME_RESCALE = rescale
        for module, frame_rescale, node_fps in saved:
            module.FRAME_RESCALE, module.FPS = frame_rescale, node_fps


def output_frames_ending_on_last(frames: int, fps: int) -> int:
    """24 fps frames for ``frames`` rendered at ``fps`` whose last one is the last rendered frame (a clip pinned on
    its last frame, or continued from it, ends on exactly that frame)."""
    return -(-(int(frames) - 1) * OUTPUT_FPS // int(fps)) + 1


def _read_frames(process: subprocess.Popen, width: int, height: int) -> Iterator[Any]:
    import numpy as np

    size = width * height * 3
    assert process.stdout is not None
    while True:
        data = process.stdout.read(size)
        if len(data) < size:
            return
        yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def upsample_video(source_args: list[str], output_args: list[str], *, width: int, height: int, source_fps: int,
                   frames: int, output_fps: int = OUTPUT_FPS) -> dict[str, Any]:
    """``frames`` frames at ``output_fps`` from the ``source_fps`` video ffmpeg makes of ``source_args`` (its inputs,
    filters and maps; RGB ``width``x``height`` frames). Output frame ``n`` sits at ``n * source_fps / output_fps``
    source frames: on a whole one it is that frame untouched, between two it is one RIFE call at that exact
    timestep, past the last it holds the last. The frames go to ffmpeg as its first input (``-i`` raw video, then
    ``output_args``: more inputs, maps, filters, codecs and the output path), the only encode."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from .minimax_h3_runtime import make_room_beside_h3

    started = time.perf_counter()
    make_room_beside_h3()
    model = rife_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_h, pad_w = math.ceil(height / 128) * 128, math.ceil(width / 128) * 128

    def tensor(frame: Any) -> Any:
        t = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).to(device).unsqueeze(0).float() / 255.0
        return F.pad(t, (0, pad_w - width, 0, pad_h - height))

    reader = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *source_args, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(output_fps), "-i", "-", *output_args],
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
    try:
        with torch.inference_mode():
            for n in range(int(frames)):
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
                    raise ValueError("the video to interpolate has no frames")
                if index >= loaded or not offset:
                    out = window[min(index, loaded)]
                else:
                    for key in (index, index + 1):
                        if key not in tensors:
                            tensors[key] = tensor(window[key])
                    mid = model.inference(tensors[index], tensors[index + 1], timestep=float(offset), scale=1.0)
                    out = (mid[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0)[:height, :width].cpu().numpy()
                    rife_calls += 1
                writer.stdin.write(np.ascontiguousarray(out).tobytes())
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("ffmpeg could not encode the interpolated video")
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
        "seconds": round(time.perf_counter() - started, 3),
    }
