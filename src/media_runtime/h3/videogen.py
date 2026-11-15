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
- ``continuable`` (default false): the clip ends on a clean frame another job can
  open on (:mod:`h3.h3_restore`, vigen e9791cc). Its last frame is previewed (the
  recipe's preview schedule, that frame alone decoded), restored toward the
  chain's anchor (``anchor_url`` / ``anchor_path``, else this job's own image) and
  pinned as the last frame, so chained clips do not pile up each other's drift
  (speckle, colour creep). Not with ``last_frame``. The continued job passes the
  returned last frame as its ``image_url`` and the same ``anchor_url``.
- ``use_interpolation`` / ``interpolation_fps`` (12-23, default 16; vigen 7280e2f): H3
  renders that many frames for every second of the clip and RIFE makes the rest of
  the 24 (:mod:`h3.h3_interpolation`). A pinned audio is heard sped up ``24/fps`` on
  H3's own 24 fps clock and the output carries it in real time; with none, H3's time
  axis itself runs at ``fps`` (the generated audio lasts the clip). Pins land on the
  nearest rendered frame; the clip still ends on its last rendered frame.

The result is ``{"path", "item", "metrics", "seed"}`` like :mod:`h3.lipsync`, plus
``last_frame_path`` (a JPEG, quality 95) for a continuable clip.
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
                   fps: float) -> list[tuple[int, Any]]:
    """``(frame index, prepared image)`` for each ``pinned_frames`` entry, in time order, on the ``fps`` render's
    frames (an interpolated job's pins land on the nearest frame it renders)."""
    from PIL import Image, ImageOps  # type: ignore

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
    # use_interpolation: H3 renders interpolation_fps frames for every second of the clip and RIFE makes the
    # rest of the 24; lengths and pins below are in frames of the render.
    interpolation_fps = requested_fps(payload)
    render_fps = interpolation_fps or H3_FPS

    scratch_dir = job_dir(job_id)
    progress.report("preparing")
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
    # A continuable clip ends on a restored frame the next job opens on; its look is restored toward the
    # chain's anchor: the anchor a continued job passes on, else this job's own image.
    continuable = payload.get("continuable") is True
    anchor_path = None
    if continuable:
        if last_frame_path is not None:
            raise ValueError("continuable clips end on their own restored frame; drop last_frame")
        if payload.get("anchor_url") or payload.get("anchor_path"):
            anchor_path = download_job_input(job, payload, "anchor", scratch_dir, ".png", kind="videogen")
        else:
            anchor_path = image_path

    canvas = resolve_canvas(image_path=image_path, requested_width=width, requested_height=height)
    delivery = delivery_dimensions(canvas, width, height, source_aspect=probe_image_dimensions(image_path))
    length = max(1, round(duration * H3_FPS))
    if interpolation_fps:
        length = interpolated_render_frames(length, interpolation_fps)
    # The node snaps the clip up to the 17k+5 grid; pins and the pinned audio cover that clip.
    grid_frames = 5 + 17 * -(-max(0, length - 5) // 17)
    guides = _pinned_frames(job, payload, scratch_dir, image_path, length, render_fps)
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
              interpolation_fps=interpolation_fps)
    started = time.perf_counter()
    continuation: dict[str, Any] | None = None
    try:
        if continuable:
            last_frame_path, continuation = _restore_end(
                job_id=job_id, prompt=prompt, image_path=image_path, seed=seed, width=width, height=height,
                length=length, grid_frames=grid_frames, preset=preset, recipe=recipe, generate_audio=generate_audio,
                guides=guides, pinned_audio=pinned_audio, anchor_path=anchor_path,
                work_dir=scratch_dir / "continuation", fps=time_axis_fps,
            )
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
            fps=time_axis_fps,
        )
    finally:
        cleanup_h3_job_files(job_id)
    if not output_path.is_file():
        raise RuntimeError(f"H3 did not write {output_path}")
    interpolation: dict[str, Any] | None = None
    if interpolation_fps:
        interpolation = _interpolate(output_path, fps=interpolation_fps, grid_frames=grid_frames,
                                     audio=soundtrack if generate_audio else None, voice=soundtrack is not None,
                                     job_id=job_id)
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
        "generate_audio": generate_audio,
        "last_frame": last_frame_path is not None and not continuable,
        "pinned_frames": [frame for frame, _ in guides],
        "pinned_audio": pinned_audio is not None,
        "continuable": continuable,
        **({"continuation": continuation} if continuation else {}),
        **({"interpolation": interpolation} if interpolation else {}),
    }
    log_event("videogen_complete", job_id=job_id, wall_seconds=round(wall_seconds, 3))
    result = {
        "path": str(output_path),
        "item": item,
        "metrics": {"engine": "comfyui_h3", "wall_seconds": round(wall_seconds, 3), **metrics},
        "seed": seed,
    }
    if continuable:
        import cv2

        jpeg = scratch_dir / "outputs" / "last_frame.jpg"
        cv2.imwrite(str(jpeg), cv2.imread(str(last_frame_path)), [cv2.IMWRITE_JPEG_QUALITY, LAST_FRAME_JPEG_QUALITY])
        result["last_frame_path"] = str(jpeg)
    return result


def _interpolate(output_path, *, fps: int, grid_frames: int, audio, voice: bool, job_id: str) -> dict[str, Any]:
    """The ``fps`` render at ``output_path`` brought up to 24 fps in place: RIFE makes the frames in between
    (:func:`h3.h3_interpolation.upsample_video`), the clip still ends on its last rendered frame (a pinned last
    frame, a continuable clip's restored one), and it carries ``audio`` (the caller's pinned audio in real time)
    or else whatever audio H3 generated on its ``fps`` time axis."""
    from .h3_interpolation import OUTPUT_FPS, output_frames_ending_on_last, upsample_video
    from .lipsync import image_size

    rendered = output_path.with_name(f"{output_path.stem}.{fps}fps.mp4")
    os.replace(output_path, rendered)
    frames = output_frames_ending_on_last(grid_frames, fps)
    width, height = image_size(rendered)
    metrics = upsample_video(
        ["-i", str(rendered)],
        ["-i", str(audio or rendered), "-map", "0:v", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", f"{frames / OUTPUT_FPS:.6f}",
         "-movflags", "+faststart", str(output_path)],
        width=width, height=height, source_fps=fps, frames=frames,
    )
    rendered.unlink(missing_ok=True)
    mode = "voice" if voice else "time_axis"
    log_event("videogen_interpolated", job_id=job_id, mode=mode, **metrics)
    return {"fps": fps, "mode": mode, "rendered_frames": metrics["source_frames"],
            "rife_frames": metrics["rife_frames"], "seconds": metrics["seconds"]}


def _restore_end(*, job_id: str, prompt: str, image_path, seed: int, width, height, length: int, grid_frames: int,
                 preset, recipe, generate_audio: bool, guides, pinned_audio, anchor_path, work_dir, fps=None):
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
    )
    preview_seconds = round(time.perf_counter() - started, 3)
    started = time.perf_counter()
    restored = work_dir / "last_frame.png"
    restore_frame(job_id=job_id, frame=work_dir / "preview.png", anchor=anchor_path, output_path=restored,
                  width=int(preview["width"]), height=int(preview["height"]), seed=seed)
    restore_seconds = round(time.perf_counter() - started, 3)
    log_event("videogen_continuable_end_restored", job_id=job_id, frame=grid_frames - 1,
              preview_sigmas=recipe.preview_sigmas, preview_seconds=preview_seconds, restore_seconds=restore_seconds)
    return restored, {"preview_seconds": preview_seconds, "restore_seconds": restore_seconds,
                      "preview_sigmas": recipe.preview_sigmas}
