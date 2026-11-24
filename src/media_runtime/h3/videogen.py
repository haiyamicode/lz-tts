"""H3 video (``videogen``): one 4-step Turbo render with its own audio.

The clip is made from a prompt, optionally opening on a frame the caller gives
(a ``MiniMaxH3AddGuide`` pin at frame 0 on the Ref2VA base).

A job is ``{"id": ..., "payload": {...}}``; the payload takes:

- ``prompt`` (required): what happens in the clip, H3's prompt;
- ``image_url`` / ``image_path``: the first frame. Without one the clip is made
  from the prompt alone, and ``width``/``height`` then say what shape it is;
- ``last_frame_url`` / ``last_frame_path``: an optional last frame (first-last
  frame to video);
- ``duration_seconds`` (default 3): clip length at 24 fps, snapped up by the
  node to H3's 17k+5 frame grid; capped by ``VIDEOGEN_MAX_SECONDS``;
- ``width`` / ``height``: a delivered frame size (both or neither). The render
  sits on H3's 32-px lattice with the image's aspect, then is scaled and
  centre-cropped to that size. Omitted, the clip keeps the image's shape;
- ``seed`` (random when omitted), ``generate_audio`` (default true);
- ``sparse_attention`` (default true): lipsync's attention
  (:func:`h3.minimax_h3_runtime.add_sparse_attention`: Comfy Kitchen INT8 + SLA, 85% of
  key blocks skipped, step 2 dense), ~2x faster diffusion; ``false`` renders dense. A
  worker without the kernel renders dense and the item says so;
- ``addons`` (default none): community LoRAs stacked on the preset at strength 1.0
  (:data:`h3.minimax_h3_runtime.VIDEO_ADDON_LORAS`): ``motion`` (Better Human Motion H3),
  ``dance`` (civitai 3056222);
- ``pinned_frames``: ``[{"image_url" | "image_path", "seconds"}]`` guide images
  pinned at those times (``MiniMaxH3AddGuide`` anchors, not overwrites), cover-fit
  to the first frame's size like lipsync's pins;
- ``pinned_audio_url`` / ``pinned_audio_path``: audio pinned in the latent (held
  out of denoising, like lipsync), so the video moves to that voice or track
  instead of inventing its own. ``duration_seconds`` then defaults to the audio's
  length (an audio longer than ``VIDEOGEN_MAX_SECONDS`` needs an explicit
  duration); the audio is silence-padded to the clip and that same audio is muxed
  onto the output (none with ``generate_audio: false``).
- ``continuable`` (default false): the clip ends on a clean frame another job can
  open on (:mod:`h3.h3_restore`, vigen e9791cc). Its last frame is previewed (the
  recipe's preview schedule, that frame alone decoded), restored toward the
  chain's anchor (``anchor_url`` / ``anchor_path``, else this job's own image, else, for a
  prompt-only first clip, its own previewed last frame: cleaned, not pulled anywhere) and
  pinned as the last frame, so chained clips do not pile up each other's drift
  (speckle, colour creep). Not with ``last_frame``. The continued job passes the
  returned last frame as its ``image_url`` and the same ``anchor_url`` (a prompt-only
  chain's anchor is its first clip's frame 0).
- ``use_interpolation`` / ``interpolation_fps`` (12-23, default 16; vigen 7280e2f): H3
  renders that many frames for every second of the clip and RIFE makes the rest of
  the 24 (:mod:`h3.h3_interpolation`). A pinned audio is heard sped up ``24/fps`` on
  H3's own 24 fps clock and the output carries it in real time; with none, H3's time
  axis itself runs at ``fps`` (the generated audio lasts the clip). Pins land on the
  nearest rendered frame; the clip still ends on its last rendered frame.
- ``upscale`` (``true`` = 1.5x, or a factor above 1 and at most 2; vigen 43c32e7): the
  delivered size (``width``/``height``, else the canvas) times the factor, longest
  side at most 3840, fitted inside ``output_dimensions`` (``{width, height}``, with upscale
  or ``output_short_side`` only, aspect kept; as lipsync's). The canvas render goes through SPAN 2x
  (loaded for this job only, after RIFE when both are asked for) and lanczos lands on that size
  (:mod:`h3.span_upscale`).
- ``output_short_side`` (64..2160; not with ``width``/``height`` or ``upscale``): the clip is delivered in
  the image's aspect with exactly that short side (480p, 720p, 1080p): above the canvas's short side
  through SPAN 2x then lanczos (:func:`h3.span_upscale.short_side_scale`), else lanczos from the render
  (:func:`h3.span_upscale.short_side_size`).
- ``native`` (default false; with ``output_short_side`` only): render at the delivered size itself
  instead of ~1 MP + SPAN, past the model's 1 MP cap up to 1920x1088
  (:func:`h3.minimax_h3_runtime.native_canvas`): 1080p of 16:9 renders 1920x1088 and the centre crop takes
  8 rows, no rescale, no SPAN. About 3x the cost of the SPAN path (~80x RTF turbo-8 on the 3090).

The result is ``{"path", "item", "metrics", "seed"}`` like :mod:`h3.lipsync`, plus
``last_frame_path`` (a JPEG, quality 95) for a continuable clip, at the clip's delivered size: the next
clip opens on it and so takes exactly this clip's shape (the canvas is off the delivered aspect by up to
a few rows, which ``output_short_side`` would carry into the next clip's size).
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import progress

from .compat import job_dir, log_event, project_path
from .lipsync import _download_url, _probe, _run, download_job_input, extract_audio_slice, prepare_pin_image

#: A continuable clip's restored last frame as delivered (the PNG is what the render pins).
LAST_FRAME_JPEG_QUALITY = 95
DEFAULT_DURATION_SECONDS = 3.0
DEFAULT_MAX_SECONDS = 10.0
MAX_PINNED_FRAMES = 16


def max_seconds() -> float:
    return float(os.environ.get("VIDEOGEN_MAX_SECONDS") or DEFAULT_MAX_SECONDS)


def _pinned_frames(job: dict[str, Any], payload: dict[str, Any], scratch_dir, image_path, length: int,
                   fps: float, *, canvas_size: tuple[int, int]) -> list[tuple[int, Any]]:
    """``(frame index, prepared image)`` for each ``pinned_frames`` entry, in time order, on the ``fps`` render's
    frames (an interpolated job's pins land on the nearest frame it renders).

    Pins are cover-fit to the first frame's size, or to the canvas when the clip has no first frame."""
    from PIL import Image, ImageOps  # type: ignore

    entries = list(payload.get("pinned_frames") or [])
    if len(entries) > MAX_PINNED_FRAMES:
        raise ValueError(f"at most {MAX_PINNED_FRAMES} pinned_frames")
    if not entries:
        return []
    if image_path is not None:
        with Image.open(image_path) as first:
            size = ImageOps.exif_transpose(first).size
    else:
        size = canvas_size
    guides = []
    for index, entry in enumerate(entries):
        seconds = float(entry["seconds"])
        frame = round(seconds * fps)
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
    frames = [frame for frame, _ in guides]
    if len(set(frames)) < len(frames):
        raise ValueError(f"pinned_frames: two pins fall on the same frame of the {fps:g} fps render")
    return sorted(guides, key=lambda guide: guide[0])


def run_videogen(job: dict[str, Any]) -> dict[str, Any]:
    from .h3_interpolation import interpolated_render_frames, requested_fps, speed_up
    from .span_upscale import requested_scale, requested_short_side, short_side_scale, short_side_size, target_size
    from .minimax_h3_runtime import (
        H3_FPS,
        H3_NATIVE_MAX_PIXEL_FRAMES,
        cleanup_h3_job_files,
        VIDEO_ADDON_LORAS,
        VIDEO_PRESETS,
        delivery_dimensions,
        env_recipe,
        h3_preset,
        probe_image_dimensions,
        native_canvas,
        resolve_canvas,
        short_side_canvas,
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
    # Lipsync's Kitchen INT8 + SLA 85% attention unless the caller asks for dense.
    sparse_attention = payload.get("sparse_attention") is not False
    addons = tuple(dict.fromkeys(str(name) for name in payload.get("addons") or ()))
    if unknown := [name for name in addons if name not in VIDEO_ADDON_LORAS]:
        raise ValueError(f"unknown addons {unknown}; available: {sorted(VIDEO_ADDON_LORAS)}")
    # use_interpolation: H3 renders interpolation_fps frames for every second of the clip and RIFE makes the
    # rest of the 24; lengths and pins below are in frames of the render.
    interpolation_fps = requested_fps(payload)
    render_fps = interpolation_fps or H3_FPS
    # upscale: the delivered size times the factor, through SPAN 2x (loaded for this job only) then lanczos.
    upscale = requested_scale(payload)
    output_dimensions = payload.get("output_dimensions")
    short_side = requested_short_side(payload)
    native = payload.get("native") is True
    if native and short_side is None:
        raise ValueError("native renders at output_short_side; send it")
    if output_dimensions is not None and not (upscale or short_side):
        raise ValueError("output_dimensions goes with upscale or output_short_side; else width/height are the delivered size")
    bounding_box = ((int(output_dimensions["width"]), int(output_dimensions["height"]))
                    if output_dimensions is not None else None)
    # The first frame is the caller's option: with one the clip opens on it and takes its shape from
    # it, without one the prompt alone makes it and width/height say what shape it comes out, since
    # nothing else states that. Alongside output_short_side they say only the shape — it says the
    # delivered size — and on their own they are the delivered size, as they are with a frame.
    has_first_frame = bool(payload.get("image_url") or payload.get("image_path"))
    aspect: tuple[int, int] | None = None
    if has_first_frame:
        if short_side is not None and width is not None:
            raise ValueError("output_short_side sizes the clip from the image's aspect; send it or width/height")
    else:
        if width is None:
            raise ValueError("a clip made from the prompt alone needs width and height")
        aspect = (width, height)
        if short_side is not None:
            width = height = None

    scratch_dir = job_dir(job_id)
    progress.report("preparing")
    image_path = (download_job_input(job, payload, "image", scratch_dir, ".png", kind="videogen")
                  if has_first_frame else None)
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
    # A continuable clip ends on a restored frame the next job opens on; its look is restored toward the
    # chain's anchor: the anchor a continued job passes on, else this job's own image.
    continuable = payload.get("continuable") is True
    anchor_path = None
    if continuable:
        if last_frame_path is not None:
            raise ValueError("continuable clips end on their own restored frame; drop last_frame")
        if payload.get("anchor_url") or payload.get("anchor_path"):
            anchor_path = download_job_input(job, payload, "anchor", scratch_dir, ".png", kind="videogen")
        elif image_path is not None:
            anchor_path = image_path
        # else a prompt-only first clip: nothing has drifted yet, so its end is restored toward itself.

    # The shape everything is sized from: the first frame's, or the one a prompt-only clip stated.
    source_aspect = probe_image_dimensions(image_path) if image_path is not None else aspect
    # An output_short_side below the usual canvas renders on the target's own canvas, not ~1 MP fitted down.
    canvas = (short_side_canvas(image_path, short_side, aspect=aspect) if short_side is not None
              else resolve_canvas(image_path=image_path, requested_width=width, requested_height=height))
    delivery = delivery_dimensions(canvas, width, height, source_aspect=source_aspect)
    # The shape the render sits on. With a first frame the render takes it from the frame, as it always
    # has; with none, and the delivered size left to output_short_side, the canvas has to say it — it is
    # on the lattice already, so naming it leaves the render exactly where it would have been.
    render_width, render_height = width, height
    if image_path is None and render_width is None:
        render_width, render_height = canvas.width, canvas.height
    # The size the clip leaves at: H3 resizes to delivery itself; an upscaled clip goes to final_size after SPAN.
    final_size = None
    max_pixels = None
    if short_side is not None:
        final_size = short_side_size(source_aspect, short_side, bounding_box=bounding_box)
        if native:
            # The canvas is the delivered size on the lattice, past the 1 MP cap: no SPAN, H3 crops the sliver.
            canvas = native_canvas(image_path, final_size)
            render_width, render_height = canvas.width, canvas.height
            max_pixels = canvas.width * canvas.height
        upscale = None if native else short_side_scale(final_size, (canvas.width, canvas.height))
        if not upscale:
            delivery = final_size
    elif upscale:
        final_size = target_size(*(delivery or (canvas.width, canvas.height)), upscale, bounding_box=bounding_box)
    length = max(1, round(duration * H3_FPS))
    if interpolation_fps:
        length = interpolated_render_frames(length, interpolation_fps)
    # The node snaps the clip up to the 17k+5 grid; pins and the pinned audio cover that clip.
    grid_frames = 5 + 17 * -(-max(0, length - 5) // 17)
    if native and canvas.width * canvas.height * grid_frames > H3_NATIVE_MAX_PIXEL_FRAMES:
        frames = H3_NATIVE_MAX_PIXEL_FRAMES // (canvas.width * canvas.height)
        limit = (5 + 17 * ((frames - 5) // 17)) / render_fps
        raise ValueError(f"a native {canvas.width}x{canvas.height} clip fits the GPU up to {limit:g} s; send a shorter "
                         "duration_seconds (or pinned audio) or drop native")
    guides = _pinned_frames(job, payload, scratch_dir, image_path, length, render_fps,
                            canvas_size=(canvas.width, canvas.height))
    if continuable and guides and guides[-1][0] >= grid_frames - 1:
        raise ValueError("a continuable clip's last frame is its restored frame; pin frames before it")
    soundtrack = None
    if pinned_audio is not None:
        pinned_audio = extract_audio_slice(pinned_audio, start_seconds=0.0, duration_seconds=grid_frames / render_fps,
                                           output_path=scratch_dir / "inputs" / "pinned_audio.wav")
        if interpolation_fps:
            # The render's frames run on H3's own 24 fps clock, so it hears the audio that much faster; the
            # output carries it in real time.
            soundtrack = pinned_audio
            pinned_audio = speed_up(pinned_audio, scratch_dir / "inputs" / "pinned_audio_sped_up.wav",
                                    fps=interpolation_fps, duration_seconds=grid_frames / H3_FPS)
    # No pinned audio to speed up: H3's time axis itself runs at the render's rate.
    time_axis_fps = interpolation_fps if interpolation_fps and pinned_audio is None else None
    output_path = scratch_dir / "outputs" / "output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_event("videogen_started", job_id=job_id, seed=seed, length=length, canvas=[canvas.width, canvas.height],
              delivered=list(delivery) if delivery else None, generate_audio=generate_audio,
              pinned_frames=[frame for frame, _ in guides], pinned_audio=pinned_audio is not None,
              interpolation_fps=interpolation_fps, upscale=upscale)
    started = time.perf_counter()
    continuation: dict[str, Any] | None = None
    try:
        if continuable:
            last_frame_path, continuation = _restore_end(
                job_id=job_id, prompt=prompt, image_path=image_path, seed=seed, width=render_width,
                height=render_height,
                length=length, grid_frames=grid_frames, preset=preset, recipe=recipe, generate_audio=generate_audio,
                guides=guides, pinned_audio=pinned_audio, anchor_path=anchor_path,
                work_dir=scratch_dir / "continuation", fps=time_axis_fps, sparse_attention=sparse_attention,
                addons=addons, max_pixels=max_pixels,
            )
        metrics = run_minimax_h3_inference(
            job_id=job_id,
            prompt=prompt,
            image_path=image_path,
            last_frame_path=last_frame_path,
            output_path=output_path,
            seed=seed,
            width=render_width,
            height=render_height,
            length=length,
            preset=preset,
            generate_audio=generate_audio,
            # An upscaled clip keeps the canvas render: SPAN works on it, lanczos lands on the delivered size.
            output_width=delivery[0] if delivery and not upscale else None,
            output_height=delivery[1] if delivery and not upscale else None,
            guides=guides,
            pinned_audio_path=pinned_audio,
            fps=time_axis_fps,
            sparse_attention=sparse_attention,
            addons=addons,
            max_pixels=max_pixels,
        )
    finally:
        cleanup_h3_job_files(job_id)
    if not output_path.is_file():
        raise RuntimeError(f"H3 did not write {output_path}")
    postprocessed: dict[str, Any] = {}
    if interpolation_fps or upscale:
        postprocessed = _postprocess(
            output_path, fps=interpolation_fps, grid_frames=grid_frames, upscale=upscale,
            size=final_size if upscale else None, bounding_box=bounding_box,
            audio=(soundtrack or pinned_audio) if pinned_audio is not None and generate_audio else None,
            voice=pinned_audio is not None, job_id=job_id,
        )
    elif pinned_audio is not None and generate_audio:
        muxed = output_path.with_name("output.audio.mp4")
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(output_path), "-i", str(pinned_audio),
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
              "-movflags", "+faststart", str(muxed)])
        os.replace(muxed, output_path)
    wall_seconds = time.perf_counter() - started
    video = next(s for s in _probe(output_path)["streams"] if s.get("codec_type") == "video")
    item = {
        "seed": seed,
        "fps": float(H3_FPS),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "canvas_width": canvas.width,
        "canvas_height": canvas.height,
        "frames": int(video.get("nb_frames") or 0) or length,
        "duration_seconds": round(float(video.get("duration") or 0.0), 3),
        "preset": preset,
        "steps": recipe.steps,
        "sparse_attention": bool(metrics.get("sparse_attention", sparse_attention)),
        "addons": list(addons),
        "native": native,
        "generate_audio": generate_audio,
        "last_frame": last_frame_path is not None and not continuable,
        "pinned_frames": [frame for frame, _ in guides],
        "pinned_audio": pinned_audio is not None,
        "continuable": continuable,
        **({"continuation": continuation} if continuation else {}),
        **postprocessed,
    }
    log_event("videogen_complete", job_id=job_id, wall_seconds=round(wall_seconds, 3))
    result = {
        "path": str(output_path),
        "item": item,
        "metrics": {"engine": "comfyui_h3", "wall_seconds": round(wall_seconds, 3), **metrics},
        "seed": seed,
    }
    if continuable:
        from PIL import Image, ImageOps  # type: ignore

        jpeg = scratch_dir / "outputs" / "last_frame.jpg"
        with Image.open(last_frame_path) as restored:
            # The canvas frame fitted to the delivered size the way the video was (cover scale, centre crop).
            ImageOps.fit(restored.convert("RGB"), (item["width"], item["height"]),
                         method=Image.Resampling.LANCZOS).save(jpeg, format="JPEG", quality=LAST_FRAME_JPEG_QUALITY)
        result["last_frame_path"] = str(jpeg)
    return result


def _postprocess(output_path, *, fps: int | None, grid_frames: int, upscale: float | None,
                 size: tuple[int, int] | None, bounding_box: tuple[int, int] | None = None, audio, voice: bool,
                 job_id: str) -> dict[str, Any]:
    """The render at ``output_path`` finished in place (:func:`h3.h3_postprocess.stream_video`): an ``fps`` render
    brought up to 24 fps by RIFE, still ending on its last rendered frame (a pinned last frame, a continuable
    clip's restored one); an ``upscale`` clip through SPAN 2x, then lanczos to ``size``. It carries ``audio`` (the caller's pinned audio in real time) or
    else whatever audio H3 generated. Returns the item's ``interpolation`` / ``upscale`` entries."""
    from .h3_interpolation import OUTPUT_FPS, output_frames_ending_on_last
    from .h3_postprocess import stream_video
    from .lipsync import image_size
    from .span_upscale import SPAN_MODEL_NAME

    rendered = output_path.with_name(f"{output_path.stem}.render.mp4")
    os.replace(output_path, rendered)
    if fps:
        source_fps, frames = fps, output_frames_ending_on_last(grid_frames, fps)
    else:
        source_fps, frames = OUTPUT_FPS, grid_frames
    width, height = image_size(rendered)
    resize = (["-vf", f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase:flags=lanczos,"
                      f"crop={size[0]}:{size[1]},setsar=1"] if size else [])
    metrics = stream_video(
        ["-i", str(rendered)],
        ["-i", str(audio or rendered), "-map", "0:v", "-map", "1:a:0?", *resize, "-c:v", "libx264", "-preset", "fast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", f"{frames / OUTPUT_FPS:.6f}",
         "-movflags", "+faststart", str(output_path)],
        width=width, height=height, source_fps=source_fps, frames=frames, upscale=bool(upscale), job_id=job_id,
    )
    rendered.unlink(missing_ok=True)
    mode = "voice" if voice else "time_axis"
    log_event("videogen_postprocessed", job_id=job_id, interpolation_fps=fps, mode=mode if fps else None,
              upscale=upscale, **metrics)
    span_seconds = float((metrics.get("span") or {}).get("seconds", 0.0))
    out: dict[str, Any] = {}
    if fps:
        out["interpolation"] = {"fps": fps, "mode": mode, "rendered_frames": metrics["source_frames"],
                                "rife_frames": metrics["rife_frames"],
                                "seconds": round(metrics["seconds"] - span_seconds, 3)}
    if size:
        out["upscale"] = {"scale": upscale, "method": "span2x_lanczos", "model": SPAN_MODEL_NAME,
                          **({"output_dimensions": {"width": bounding_box[0], "height": bounding_box[1]}}
                             if bounding_box else {}),
                          "span_seconds": span_seconds}
    return out


def _restore_end(*, job_id: str, prompt: str, image_path, seed: int, width, height, length: int, grid_frames: int,
                 preset, recipe, generate_audio: bool, guides, pinned_audio, anchor_path, work_dir, fps=None,
                 sparse_attention: bool = True, addons: tuple[str, ...] = (), max_pixels: int | None = None):
    """A continuable clip's last frame: this clip's own last frame previewed (the recipe's preview
    schedule, that frame alone decoded), then restored toward ``anchor_path``. The render that follows
    pins it as its last frame, so the clip ends on it and the next job opens on it."""
    from .h3_restore import FramePreview, restore_frame
    from .minimax_h3_runtime import run_minimax_h3_inference

    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    preview = run_minimax_h3_inference(
        job_id=job_id,
        prompt=prompt,
        image_path=image_path,
        last_frame_path=None,
        output_path=work_dir / "unused.mp4",
        seed=seed,
        width=width,
        height=height,
        length=length,
        preset=preset,
        generate_audio=generate_audio,
        guides=guides,
        pinned_audio_path=pinned_audio,
        preview=FramePreview(frame=grid_frames - 1, output_path=work_dir / "preview.png", sigmas=recipe.preview_sigmas),
        fps=fps,
        sparse_attention=sparse_attention,
        addons=addons,
        max_pixels=max_pixels,
    )
    preview_seconds = round(time.perf_counter() - started, 3)
    started = time.perf_counter()
    restored = work_dir / "last_frame.png"
    restore_frame(job_id=job_id, frame=work_dir / "preview.png", anchor=anchor_path or work_dir / "preview.png",
                  output_path=restored,
                  width=int(preview["width"]), height=int(preview["height"]), seed=seed)
    restore_seconds = round(time.perf_counter() - started, 3)
    log_event("videogen_continuable_end_restored", job_id=job_id, frame=grid_frames - 1,
              preview_sigmas=recipe.preview_sigmas, preview_seconds=preview_seconds, restore_seconds=restore_seconds)
    return restored, {"preview_seconds": preview_seconds, "restore_seconds": restore_seconds,
                      "preview_sigmas": recipe.preview_sigmas}
