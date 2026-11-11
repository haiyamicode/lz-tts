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
- ``seed`` (random when omitted), ``generate_audio`` (default true).

The result is ``{"path", "item", "metrics", "seed"}`` like :mod:`h3.lipsync`.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

from .compat import job_dir, log_event
from .lipsync import _probe, download_job_input

DEFAULT_DURATION_SECONDS = 3.0
DEFAULT_MAX_SECONDS = 10.0


def max_seconds() -> float:
    return float(os.environ.get("VIDEOGEN_MAX_SECONDS") or DEFAULT_MAX_SECONDS)


def run_videogen(job: dict[str, Any]) -> dict[str, Any]:
    from .minimax_h3_runtime import (
        H3_FPS,
        cleanup_h3_job_files,
        delivery_dimensions,
        h3_steps,
        probe_image_dimensions,
        resolve_canvas,
        run_minimax_h3_inference,
    )

    payload = job["payload"]
    job_id = str(job["id"])
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("videogen requires a prompt")
    duration = float(payload.get("duration_seconds") or DEFAULT_DURATION_SECONDS)
    if not 0 < duration <= max_seconds():
        raise ValueError(f"duration_seconds must be in (0, {max_seconds():g}]")
    width, height = payload.get("width"), payload.get("height")
    if (width is None) != (height is None):
        raise ValueError("width and height go together")
    width = int(width) if width is not None else None
    height = int(height) if height is not None else None
    if width is not None and (width < 64 or height < 64 or width > 4096 or height > 4096):
        raise ValueError("width/height must be within 64..4096")
    seed = int(payload["seed"]) if payload.get("seed") is not None else random.randint(0, 2**31 - 1)
    generate_audio = payload.get("generate_audio") is not False

    scratch_dir = job_dir(job_id)
    image_path = download_job_input(job, payload, "image", scratch_dir, ".png", kind="videogen")
    last_frame_path = None
    if payload.get("last_frame_url") or payload.get("last_frame_path"):
        last_frame_path = download_job_input(job, payload, "last_frame", scratch_dir, ".png", kind="videogen")

    canvas = resolve_canvas(image_path=image_path, requested_width=width, requested_height=height)
    delivery = delivery_dimensions(canvas, width, height, source_aspect=probe_image_dimensions(image_path))
    length = max(1, round(duration * H3_FPS))
    output_path = scratch_dir / "outputs" / "output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_event("videogen_started", job_id=job_id, seed=seed, length=length, canvas=[canvas.width, canvas.height],
              delivered=list(delivery) if delivery else None, generate_audio=generate_audio)
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
            generate_audio=generate_audio,
            output_width=delivery[0] if delivery else None,
            output_height=delivery[1] if delivery else None,
        )
    finally:
        cleanup_h3_job_files(job_id)
    if not output_path.is_file():
        raise RuntimeError(f"H3 did not write {output_path}")
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
        "steps": h3_steps(),
        "generate_audio": generate_audio,
        "last_frame": last_frame_path is not None,
    }
    log_event("videogen_complete", job_id=job_id, wall_seconds=round(wall_seconds, 3))
    return {
        "path": str(output_path),
        "item": item,
        "metrics": {"engine": "comfyui_h3", "wall_seconds": round(wall_seconds, 3), **metrics},
        "seed": seed,
    }
