"""Interpolated H3 renders (vigen 7280e2f): the frame maths, the lipsync plan on the model's clock, and RIFE's
pass-through of rendered frames (GPU)."""

from __future__ import annotations

import subprocess
from fractions import Fraction

import numpy as np
import pytest

from h3.h3_interpolation import (
    interpolated_frame,
    interpolated_render_frames,
    output_frames_ending_on_last,
    requested_fps,
    upsample_video,
)
from h3.lipsync import chunk_instructions, freestyle_plan, plan_anchors, plan_chunks
from h3.speech_chunks import MAX_CHUNK_SECONDS
from h3.test_h3_restore import _speech


def test_the_24_fps_clip_ends_on_the_last_rendered_frame():
    for frames in (5, 22, 73, 158, 226):
        for fps in (12, 16, 20, 23):
            out = output_frames_ending_on_last(frames, fps)
            # The last output frame is the last rendered frame (on it, or just past it and holding it); the one
            # before is still an earlier instant, so the clip is no longer than it needs to be.
            assert Fraction((out - 1) * fps, 24) >= frames - 1 > Fraction((out - 2) * fps, 24)


def test_render_lengths_and_pins_at_the_render_rate():
    assert interpolated_render_frames(72, 16) == 48
    assert interpolated_render_frames(73, 16) == 49  # never shorter than the 24 fps clip
    assert interpolated_render_frames(72, 12) == 36
    assert interpolated_frame(48, 16) == 32
    assert interpolated_frame(1, 12) == 1  # a pin never lands on frame 0, the job's own image


def test_requested_fps():
    assert requested_fps({}) is None
    assert requested_fps({"use_interpolation": True}) == 16
    assert requested_fps({"use_interpolation": True, "interpolation_fps": 12}) == 12
    with pytest.raises(ValueError):
        requested_fps({"interpolation_fps": 16})
    with pytest.raises(ValueError):
        requested_fps({"use_interpolation": True, "interpolation_fps": 24})


def test_interpolated_chunks_cover_more_audio_but_render_no_more_frames():
    seconds, cap = 60, 226
    at24 = plan_chunks(24 * seconds, max_frames=cap, anchors=plan_anchors(
        freestyle_plan(_speech(seconds), total_frames=24 * seconds, fps=24.0, chunk_cap=cap)[1], []),
        cut_frames=freestyle_plan(_speech(seconds), total_frames=24 * seconds, fps=24.0, chunk_cap=cap)[0])
    total = 16 * seconds
    cuts, pins = freestyle_plan(_speech(seconds), total_frames=total, fps=16.0, chunk_cap=cap, model_fps=24.0)
    at16 = plan_chunks(total, max_frames=cap, anchors=plan_anchors(pins, []), cut_frames=cuts)
    # The cap is in the model's frames: a 16 fps chunk may run 10 s of the model's clock (240 frames, here the
    # 225-frame render cap), i.e. up to 14 s of the audio, so the job needs fewer chunks.
    longest = max(chunk.frames for chunk in at16)
    assert longest > MAX_CHUNK_SECONDS * 16
    assert all(chunk.render_frames <= cap for chunk in at16)
    assert len(at16) < len(at24)


def test_timed_instructions_are_on_the_models_clock():
    chunks = plan_chunks(16 * 6, max_frames=226, anchors=plan_anchors([0, 16 * 6 - 1], []), cut_frames=[])
    items = [{"timestamp": 3.0, "instruction": "smile"}]
    plain = chunk_instructions(items, chunks, fps=16.0, total_frames=16 * 6)
    scaled = chunk_instructions(items, chunks, fps=16.0, total_frames=16 * 6, model_fps=24.0)
    # 3 s into the audio is 2 s into the sped-up voice the model hears.
    assert "3 seconds" in plain[0][0] and "2 seconds" in scaled[0][0]


def _frames(path, width, height):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="RIFE runs on the GPU")
def test_rendered_frames_pass_through_untouched(tmp_path):
    width, height, rendered = 160, 96, 9
    source = tmp_path / "src.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=16",
                    "-frames:v", str(rendered), "-c:v", "ffv1", str(source)], check=True)
    out = tmp_path / "out.mkv"
    frames = output_frames_ending_on_last(rendered, 16)
    metrics = upsample_video(["-i", str(source)], ["-c:v", "ffv1", str(out)], width=width, height=height,
                             source_fps=16, frames=frames)
    src, got = _frames(source, width, height), _frames(out, width, height)
    assert len(got) == frames == 13
    # Output frame n sits at n*2/3 source frames: every third one is a rendered frame, the rest are RIFE's.
    assert metrics["rife_frames"] == 8 and metrics["source_frames"] == rendered
    for n in range(0, frames, 3):
        assert np.array_equal(got[n], src[n * 2 // 3])
    assert not np.array_equal(got[1], src[0]) and not np.array_equal(got[1], src[1])
