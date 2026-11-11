"""H3 still images (``imagegen``): text-to-image and multi-reference editing.

Ported from vigen's ``run_h3_image_generation``. The recipe is :mod:`h3.h3_still`'s
(plain FL2VA INT8 base, no LoRA, ``er_sde``/``simple``, 50 steps, no CFG). The
still renders on H3's native ~1 MP canvas with the delivered aspect and is then
Lanczos-resized to the delivered size.

A job is ``{"id": ..., "payload": {...}}``; the payload takes:

- ``prompt`` (required). ``image 1`` / ``picture 2`` style mentions become H3's
  ``<Picture N>`` tags;
- ``image_urls`` / ``image_paths``: 0..9 reference images, ``<Picture 1>`` first;
- ``width`` / ``height``: the delivered size (both or neither, 64..4096).
  Omitted, an edit follows ``<Picture 1>``'s shape at the native area and a
  text-only image is 768x1024;
- ``seed`` (random when omitted).

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
    from PIL import Image  # type: ignore

    from . import h3_still
    from .lipsync import prepare_input_image
    from .minimax_h3_runtime import run_minimax_h3_still

    payload = job["payload"]
    job_id = str(job["id"])
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("imagegen requires a prompt")
    width, height = payload.get("width"), payload.get("height")
    if (width is None) != (height is None):
        raise ValueError("width and height go together")
    if width is not None:
        width, height = int(width), int(height)
        if not (64 <= width <= 4096 and 64 <= height <= 4096):
            raise ValueError("width/height must be within 64..4096")
    seed = int(payload["seed"]) if payload.get("seed") is not None else random.randint(0, 2**31 - 1)

    scratch_dir = job_dir(job_id)
    started = time.perf_counter()
    # Upright RGB PNGs: EXIF rotation applied and alpha dropped before H3 reads them.
    references = [
        prepare_input_image(path, scratch_dir / "inputs" / f"picture_{index}.png")
        for index, path in enumerate(_references(job, payload, scratch_dir), start=1)
    ]
    input_seconds = time.perf_counter() - started

    if width is None:
        if references:
            with Image.open(references[0]) as first:
                aspect_w, aspect_h = first.size
            width, height = h3_still.fit_canvas(aspect_w, aspect_h)
        else:
            width, height = h3_still.DEFAULT_WIDTH, h3_still.DEFAULT_HEIGHT
    # Small requests still render at the native area, then scale down.
    render_width, render_height = h3_still.fit_canvas(
        width, height, pixels=max(width * height, h3_still.DEFAULT_CANVAS_PIXELS)
    )
    render_path = scratch_dir / "outputs" / "h3_still.png"
    output_path = scratch_dir / "outputs" / "output.png"
    log_event("imagegen_started", job_id=job_id, seed=seed, references=len(references),
              render=[render_width, render_height], delivered=[width, height])
    inference_started = time.perf_counter()
    render_metrics = run_minimax_h3_still(
        job_id=job_id,
        prompt=h3_still.picture_tags(prompt),
        reference_paths=references,
        output_path=render_path,
        width=render_width,
        height=render_height,
        seed=seed,
        steps=h3_still.DEFAULT_STEPS,
    )
    with Image.open(render_path) as rendered:
        image = rendered.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
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
            "steps": h3_still.DEFAULT_STEPS,
        },
        "metrics": {
            "wall_seconds": round(wall_seconds, 3),
            "input_seconds": round(input_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            **render_metrics,
        },
        "seed": seed,
    }
