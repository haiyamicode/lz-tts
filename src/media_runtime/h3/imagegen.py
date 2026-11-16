"""H3 still images (``imagegen``): text-to-image and multi-reference editing.

Ported from vigen's ``run_h3_image_generation``. The recipe is a
:data:`h3.h3_still.STILL_PRESETS` entry on the FL2VA INT8 base, no CFG. The
still renders on H3's native ~1 MP canvas with the delivered aspect and is then
Lanczos-resized to the delivered size.

A job is ``{"id": ..., "payload": {...}}``; the payload takes:

- ``prompt`` (required). ``image 1`` / ``picture 2`` style mentions become H3's
  ``<Picture N>`` tags;
- ``image_urls`` / ``image_paths``: 0..9 reference images, ``<Picture 1>`` first;
- ``width`` / ``height``: the delivered size (both or neither, 64..4096).
  Omitted, an edit follows ``<Picture 1>``'s shape at the native area and a
  text-only image is 768x1024;
- ``seed`` (random when omitted);
- ``preset``: ``standard`` (default; plain base, 50 steps), ``turbo-8`` (8-step
  Turbo LoRA) or ``turbo-4`` (videogen's 4-step Turbo LoRA);
- ``rest_frame`` (``k`` >= 0): rest still ``k`` of the person in the one
  reference image, for lipsync frame pins (:mod:`h3.rest_stills`). No
  ``prompt``: the still's prompt is written here. Rendered on ``seed + k``
  at the native area, an open mouth retaken; delivered at the image's size
  (long side <= 2048) unless ``width``/``height`` are given.

The result is ``{"path", "item", "metrics", "seed"}`` like :mod:`h3.videogen`.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .compat import job_dir, log_event, project_path
from .lipsync import _download_url


def _references(job: dict[str, Any], payload: dict[str, Any], scratch_dir: Path) -> list[Path]:
    from .h3_still import MAX_REFERENCES

    urls = [str(url) for url in payload.get("image_urls") or []]
    paths = [str(path) for path in payload.get("image_paths") or []]
    if urls and paths:
        raise ValueError("give image_urls or image_paths, not both")
    if len(urls) + len(paths) > MAX_REFERENCES:
        raise ValueError(f"H3 conditions on at most {MAX_REFERENCES} reference images")
    resolved = []
    for index, url in enumerate(urls, start=1):
        suffix = Path(urlparse(url).path).suffix or ".png"
        target = scratch_dir / "inputs" / f"picture_{index}{suffix}"
        resolved.append(_download_url(str(job["id"]), url, target, input_name=f"image_urls[{index - 1}]", kind="imagegen"))
    for value in paths:
        path = project_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"image_paths entry does not exist: {path}")
        resolved.append(path)
    return resolved


def run_imagegen(job: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageOps  # type: ignore

    from . import h3_still, rest_stills
    from .lipsync import prepare_input_image
    from .minimax_h3_runtime import run_minimax_h3_still

    payload = job["payload"]
    job_id = str(job["id"])
    rest_frame = payload.get("rest_frame")
    prompt = str(payload.get("prompt") or "").strip()
    if rest_frame is not None:
        rest_frame = int(rest_frame)
        if rest_frame < 0:
            raise ValueError("rest_frame must be >= 0")
        if prompt:
            raise ValueError("rest_frame writes its own prompt; send no prompt")
        if len(payload.get("image_urls") or payload.get("image_paths") or []) != 1:
            raise ValueError("rest_frame takes exactly one image (the person)")
        prompt = rest_stills.rest_prompt(rest_frame)
    if not prompt:
        raise ValueError("imagegen requires a prompt")
    width, height = payload.get("width"), payload.get("height")
    if (width is None) != (height is None):
        raise ValueError("width and height go together")
    if width is not None:
        width, height = int(width), int(height)
        if not (64 <= width <= 4096 and 64 <= height <= 4096):
            raise ValueError("width/height must be within 64..4096")
    preset = str(payload.get("preset") or h3_still.DEFAULT_PRESET)
    spec = h3_still.still_preset(preset)
    seed = int(payload["seed"]) if payload.get("seed") is not None else random.randint(0, 2**31 - 1)

    scratch_dir = job_dir(job_id)
    started = time.perf_counter()
    # Upright RGB PNGs: EXIF rotation applied and alpha dropped before H3 reads them.
    references = [
        prepare_input_image(path, scratch_dir / "inputs" / f"picture_{index}.png")
        for index, path in enumerate(_references(job, payload, scratch_dir), start=1)
    ]
    input_seconds = time.perf_counter() - started

    if rest_frame is not None:
        with Image.open(references[0]) as first:
            source_size = first.size
        if width is None:
            width, height = rest_stills.delivered_size(*source_size)
        render_width, render_height = h3_still.fit_canvas(*source_size)
    elif width is None:
        if references:
            with Image.open(references[0]) as first:
                aspect_w, aspect_h = first.size
            width, height = h3_still.fit_canvas(aspect_w, aspect_h)
        else:
            width, height = h3_still.DEFAULT_WIDTH, h3_still.DEFAULT_HEIGHT
    if rest_frame is None:
        # Small requests still render at the native area, then scale down.
        render_width, render_height = h3_still.fit_canvas(
            width, height, pixels=max(width * height, h3_still.DEFAULT_CANVAS_PIXELS)
        )
    output_path = scratch_dir / "outputs" / "output.png"
    log_event("imagegen_started", job_id=job_id, preset=preset, seed=seed, references=len(references),
              render=[render_width, render_height], delivered=[width, height])
    inference_started = time.perf_counter()
    takes: list[dict[str, Any]] = []
    for attempt in range(rest_stills.MAX_ATTEMPTS if rest_frame is not None else 1):
        take_seed = seed if rest_frame is None else rest_stills.attempt_seed(seed, rest_frame, attempt)
        render_path = scratch_dir / "outputs" / f"h3_still_{attempt}.png"
        render_metrics = run_minimax_h3_still(
            job_id=job_id,
            prompt=h3_still.picture_tags(prompt),
            reference_paths=references,
            output_path=render_path,
            width=render_width,
            height=render_height,
            seed=take_seed,
            preset=preset,
        )
        take: dict[str, Any] = {"seed": take_seed, "path": render_path}
        takes.append(take)
        if rest_frame is None:
            break
        take["mouth"] = rest_stills.mouth_opening(render_path)
        log_event("imagegen_rest_take", job_id=job_id, rest_frame=rest_frame, attempt=attempt, seed=take_seed,
                  mouth=take["mouth"])
        if take["mouth"] is None or take["mouth"] <= rest_stills.MOUTH_CLOSED:
            break
    kept = min(takes, key=lambda t: t.get("mouth") or 0.0)
    with Image.open(kept["path"]) as rendered:
        image = rendered.convert("RGB")
        if image.size != (width, height):
            image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        image.save(output_path)
    inference_seconds = time.perf_counter() - inference_started
    wall_seconds = time.perf_counter() - started
    log_event("imagegen_complete", job_id=job_id, wall_seconds=round(wall_seconds, 3))
    return {
        "path": str(output_path),
        "item": {
            "seed": seed,
            "width": width,
            "height": height,
            "render_width": render_width,
            "render_height": render_height,
            "references": len(references),
            "preset": preset,
            "steps": spec["steps"],
            **({"rest_frame": rest_frame, "still_seed": kept["seed"],
                "variation": rest_stills.REST_VARIATIONS[rest_frame % len(rest_stills.REST_VARIATIONS)],
                "mouth": None if kept["mouth"] is None else round(kept["mouth"], 4),
                "takes": [{"seed": t["seed"], "mouth": None if t["mouth"] is None else round(t["mouth"], 4)}
                          for t in takes]} if rest_frame is not None else {}),
        },
        "metrics": {
            "wall_seconds": round(wall_seconds, 3),
            "input_seconds": round(input_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            **render_metrics,
        },
        "seed": seed,
    }
