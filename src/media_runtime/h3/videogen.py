"""H3 image-to-video (``videogen``): one 4-step Turbo render with its own audio.

A job is ``{"id": ..., "payload": {...}}``; the payload takes:

- ``prompt`` (required): what happens in the clip, H3's FL2VA prompt;
- ``image_url`` / ``image_path`` (required): the first frame;
- ``last_frame_url`` / ``last_frame_path``: an optional last frame (first-last
  frame to video);
- ``duration_seconds`` (default 3): clip length at 24 fps, snapped up by the
  node to H3's 17k+5 frame grid; capped by ``VIDEOGEN_MAX_SECONDS``;
- ``width`` / ``height``: a delivered frame size (both or neither). The render
  sits on H3's 32-px lattice with the image's aspect, then is scaled and
  centre-cropped to that size. Omitted, the clip keeps the image's shape;
- ``seed`` (random when omitted), ``generate_audio`` (default true);
- ``pinned_frames``: ``[{"image_url" | "image_path", "seconds"}]`` guide images
  pinned at those times (``MiniMaxH3AddGuide`` anchors, not overwrites), cover-fit
  to the first frame's size like lipsync's pins;
- ``pinned_audio_url`` / ``pinned_audio_path``: audio pinned in the latent (held
  out of denoising, like lipsync), so the video moves to that voice or track
  instead of inventing its own. ``duration_seconds`` then defaults to the audio's
  length (an audio longer than ``VIDEOGEN_MAX_SECONDS`` needs an explicit
  duration); the audio is silence-padded to the clip and that same audio is muxed
  onto the output (none with ``generate_audio: false``).

The result is ``{"path", "item", "metrics", "seed"}`` like :mod:`h3.lipsync`.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

from .compat import job_dir, log_event, project_path
from .lipsync import _download_url, _probe, _run, download_job_input, extract_audio_slice, prepare_pin_image

DEFAULT_DURATION_SECONDS = 3.0
DEFAULT_MAX_SECONDS = 10.0
MAX_PINNED_FRAMES = 16


def max_seconds() -> float:
    return float(os.environ.get("VIDEOGEN_MAX_SECONDS") or DEFAULT_MAX_SECONDS)


def _pinned_frames(job: dict[str, Any], payload: dict[str, Any], scratch_dir, image_path, length: int) -> list[tuple[int, Any]]:
    """``(frame index, prepared image)`` for each ``pinned_frames`` entry, in time order."""
    from PIL import Image, ImageOps  # type: ignore

    from .minimax_h3_runtime import H3_FPS

    entries = list(payload.get("pinned_frames") or [])
    if len(entries) > MAX_PINNED_FRAMES:
        raise ValueError(f"at most {MAX_PINNED_FRAMES} pinned_frames")
    if not entries:
        return []
    with Image.open(image_path) as first:
        size = ImageOps.exif_transpose(first).size
    guides = []
    for index, entry in enumerate(entries):
        seconds = float(entry["seconds"])
        frame = round(seconds * H3_FPS)
        if not 0 <= frame < length:
            raise ValueError(f"pinned_frames[{index}].seconds {seconds:g} is outside the clip")
        if entry.get("image_url"):
            raw = _download_url(str(job["id"]), str(entry["image_url"]), scratch_dir / "inputs" / f"pin{index}_raw",
                                input_name=f"pinned_frames[{index}].image_url", kind="videogen")
        elif entry.get("image_path"):
            raw = project_path(str(entry["image_path"]))
            if not raw.is_file():
                raise FileNotFoundError(f"pinned_frames[{index}].image_path does not exist: {raw}")
        else:
            raise ValueError(f"pinned_frames[{index}] needs image_url or image_path")
        guides.append((frame, prepare_pin_image(raw, scratch_dir / "inputs" / f"pin{index}.png", size=size)))
    return sorted(guides, key=lambda guide: guide[0])


def run_videogen(job: dict[str, Any]) -> dict[str, Any]:
    from .minimax_h3_runtime import (
        H3_FPS,
        cleanup_h3_job_files,
        VIDEO_PRESETS,
        delivery_dimensions,
        env_recipe,
        h3_preset,
        probe_image_dimensions,
        resolve_canvas,
        run_minimax_h3_inference,
    )

    payload = job["payload"]
    job_id = str(job["id"])
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("videogen requires a prompt")
    width, height = payload.get("width"), payload.get("height")
    if (width is None) != (height is None):
        raise ValueError("width and height go together")
    width = int(width) if width is not None else None
    height = int(height) if height is not None else None
    if width is not None and (width < 64 or height < 64 or width > 4096 or height > 4096):
        raise ValueError("width/height must be within 64..4096")
    seed = int(payload["seed"]) if payload.get("seed") is not None else random.randint(0, 2**31 - 1)
    generate_audio = payload.get("generate_audio") is not False
    preset = payload.get("preset") or None
    recipe = h3_preset(str(preset), VIDEO_PRESETS) if preset else env_recipe()

    scratch_dir = job_dir(job_id)
    image_path = download_job_input(job, payload, "image", scratch_dir, ".png", kind="videogen")
    pinned_audio = None
    if payload.get("pinned_audio_url") or payload.get("pinned_audio_path"):
        pinned_audio = download_job_input(job, payload, "pinned_audio", scratch_dir, ".wav", kind="videogen")
    if payload.get("duration_seconds"):
        duration = float(payload["duration_seconds"])
    elif pinned_audio is not None:
        duration = float(_probe(pinned_audio)["format"]["duration"])
        if duration > max_seconds():
            raise ValueError(
                f"the pinned audio lasts {duration:.2f} s, over videogen's {max_seconds():g} s; "
                "pass duration_seconds to render its start"
            )
    else:
        duration = DEFAULT_DURATION_SECONDS
    if not 0 < duration <= max_seconds():
        raise ValueError(f"duration_seconds must be in (0, {max_seconds():g}]")
    last_frame_path = None
    if payload.get("last_frame_url") or payload.get("last_frame_path"):
        last_frame_path = download_job_input(job, payload, "last_frame", scratch_dir, ".png", kind="videogen")

    canvas = resolve_canvas(image_path=image_path, requested_width=width, requested_height=height)
    delivery = delivery_dimensions(canvas, width, height, source_aspect=probe_image_dimensions(image_path))
    length = max(1, round(duration * H3_FPS))
    # The node snaps the clip up to the 17k+5 grid; pins and the pinned audio cover that clip.
    grid_frames = 5 + 17 * -(-max(0, length - 5) // 17)
    guides = _pinned_frames(job, payload, scratch_dir, image_path, length)
    if pinned_audio is not None:
        pinned_audio = extract_audio_slice(pinned_audio, start_seconds=0.0, duration_seconds=grid_frames / H3_FPS,
                                           output_path=scratch_dir / "inputs" / "pinned_audio.wav")
    output_path = scratch_dir / "outputs" / "output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_event("videogen_started", job_id=job_id, seed=seed, length=length, canvas=[canvas.width, canvas.height],
              delivered=list(delivery) if delivery else None, generate_audio=generate_audio,
              pinned_frames=[frame for frame, _ in guides], pinned_audio=pinned_audio is not None)
    started = time.perf_counter()
    try:
        metrics = run_minimax_h3_inference(
            job_id=job_id,
            prompt=prompt,
            image_path=image_path,
            last_frame_path=last_frame_path,
            output_path=output_path,
            seed=seed,
            width=width,
            height=height,
            length=length,
            preset=preset,
            generate_audio=generate_audio,
            output_width=delivery[0] if delivery else None,
            output_height=delivery[1] if delivery else None,
            guides=guides,
            pinned_audio_path=pinned_audio,
        )
    finally:
        cleanup_h3_job_files(job_id)
    if not output_path.is_file():
        raise RuntimeError(f"H3 did not write {output_path}")
    if pinned_audio is not None and generate_audio:
        muxed = output_path.with_name("output.audio.mp4")
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(output_path), "-i", str(pinned_audio),
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
              "-movflags", "+faststart", str(muxed)])
        os.replace(muxed, output_path)
    wall_seconds = time.perf_counter() - started
    video = next(s for s in _probe(output_path)["streams"] if s.get("codec_type") == "video")
    item = {
        "seed": seed,
        "fps": H3_FPS,
        "width": int(video["width"]),
        "height": int(video["height"]),
        "canvas_width": canvas.width,
        "canvas_height": canvas.height,
        "frames": int(video.get("nb_frames") or 0) or length,
        "duration_seconds": round(float(video.get("duration") or 0.0), 3),
        "preset": preset,
        "steps": recipe.steps,
        "generate_audio": generate_audio,
        "last_frame": last_frame_path is not None,
        "pinned_frames": [frame for frame, _ in guides],
        "pinned_audio": pinned_audio is not None,
    }
    log_event("videogen_complete", job_id=job_id, wall_seconds=round(wall_seconds, 3))
    return {
        "path": str(output_path),
        "item": item,
        "metrics": {"engine": "comfyui_h3", "wall_seconds": round(wall_seconds, 3), **metrics},
        "seed": seed,
    }
