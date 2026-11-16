"""H3 ``upscale`` (vigen 43c32e7): the target size, the request, SPAN's strips against the whole frame (the real
model, skipped without data/span-upscale), and a clip through :func:`h3.h3_postprocess.stream_video`."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch

from h3 import span_upscale
from h3.compat import SPAN_UPSCALE_MODEL_PATH
from h3.h3_postprocess import stream_video

needs_model = pytest.mark.skipif(not SPAN_UPSCALE_MODEL_PATH.exists(), reason="data/span-upscale is not on disk")


def test_the_target_is_the_delivered_size_times_the_factor_within_4k_and_the_box():
    assert span_upscale.target_size(1088, 1920, 1.5) == (1632, 2880)
    assert span_upscale.target_size(1088, 1920, 2.0) == (2176, 3840)
    assert span_upscale.target_size(2160, 3840, 1.5) == (2160, 3840)  # 4K long side at most
    assert span_upscale.target_size(640, 608, 1.5, bounding_box=(800, 800)) == (800, 760)


def test_upscale_is_true_or_a_factor_up_to_2():
    assert span_upscale.requested_scale({"upscale": True}) == 1.5
    assert span_upscale.requested_scale({"upscale": 2}) == 2.0
    assert span_upscale.requested_scale({"upscale": False}) is None
    assert span_upscale.requested_scale({}) is None
    for value in (3.0, 1.0, 0.5):
        with pytest.raises(ValueError, match="upscale"):
            span_upscale.requested_scale({"upscale": value})


@needs_model
def test_strips_match_the_whole_frame(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # fp32 on the CPU: a deterministic reference
    frame = np.random.default_rng(0).integers(0, 256, (900, 48, 3), dtype=np.uint8)
    with span_upscale.loaded_span() as span:
        stripped = span([frame])[0]  # three strips of 300 rows
        with torch.inference_mode():
            whole = span.descriptor(torch.from_numpy(frame.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255.0)
        whole = whole.clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).numpy()
    assert stripped.shape == (1800, 96, 3)
    assert int(np.abs(stripped.astype(int) - whole.astype(int)).max()) <= 1
    assert span.descriptor is None  # freed with the block


@needs_model
def test_a_clip_goes_through_span_then_lanczos_to_the_target(tmp_path):
    width, height, frames = 160, 96, 6
    source = tmp_path / "src.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=24",
                    "-frames:v", str(frames), "-c:v", "ffv1", str(source)], check=True)
    target = span_upscale.target_size(width, height, 1.5)
    out = tmp_path / "out.mkv"
    metrics = stream_video(["-i", str(source)],
                           ["-vf", f"scale={target[0]}:{target[1]}:flags=lanczos", "-c:v", "ffv1", str(out)],
                           width=width, height=height, source_fps=24, frames=frames, upscale=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                            "stream=width,height,nb_read_frames", "-of", "csv=p=0", str(out)],
                           check=True, capture_output=True, text=True).stdout.strip().split(",")
    assert (int(probe[0]), int(probe[1]), int(probe[2])) == (*target, frames) == (240, 144, 6)
    assert metrics["rife_frames"] == 0 and metrics["span"]["frames"] == frames
