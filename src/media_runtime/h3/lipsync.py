"""Lipsync: one image of a person plus audio in, a talking video out.

The video is generated with the lipsync student - our 3-step
first-fold avatar LoRA (:data:`ai_worker.minimax_h3_runtime.LIPSYNC_LORA_NAME`)
- while the job's audio is pinned in the latent
(:func:`ai_worker.minimax_h3_runtime.build_lipsync_graph`), so face, head,
hands and body all move with the speech and nothing is composited:

- the output lasts as long as the audio;
- the output has the image's size, or fits inside ``output_dimensions`` with
  its aspect kept;
- the result carries ``outputs`` / ``item`` / ``metrics`` / ``seed`` / ``model``.

**Freestyle.** The job's image (``image_url`` / ``image_s3_key`` /
``image_path``) is the only image the render sees. The chunks are cut
at the quietest frames of the voice (:func:`ai_worker.speech_chunks.quiet_cuts`),
and the input image is pinned on every chunk's first and last frame and inside
it every ``speech_chunks.PIN_GAP_SECONDS`` (2.5-5 s) at the quietest frames in
reach (:func:`ai_worker.speech_chunks.pause_pins`) - the middle of a pause where
there is one, else the lowest speech energy. Motion and gestures between the
pins come from the prompt and the pinned audio, and a pose or framing the model
invents never outlasts the next pin.

**Voice.** The render never hears the raw audio: music, room tone and noise make
H3 invent things (overlay graphics, punch-ins) and blur the pauses the pins go
in. :func:`ai_worker.speech_cleanup.clean_voice` separates the vocals (Demucs)
and denoises them (ZipEnhancer) into a 16 kHz voice, and that voice is what
Whisper transcribes, what places the cuts and pins, and what is pinned into
every chunk's latent. The caller's audio is what the job delivers.

**Speech.** Each chunk's words come from a timed transcript: the caller's
``transcript`` (text items with start/end seconds), or a Whisper transcription
when there is none. Unless the caller overrides ``prompt``, every chunk is
prompted with :data:`ai_worker.speech_chunks.PROMPT_TEMPLATE` (general for any
person on camera) and the words spoken in it; a caller ``prompt`` is a template
of the same kind (:mod:`apiserver.lipsync_prompt`: ``{{text}}`` required,
``{{language}}`` optional) rendered per chunk the same way. The caller's
``custom_instructions`` (a ``timestamp`` or a ``start``-``end`` range in
seconds of the whole video, ``instruction``, optional ``text_hint``) are added
on their own lines at the end of the prompt of the chunk playing at their
timestamp, or of every chunk their range overlaps, with the time inside that
chunk (``At 3.25 seconds, when speaking "...": ...``, ``During 1.5 to 6
seconds: ...``, :func:`chunk_instructions`), so each one steers those chunks only.
A range with ``pin_chunk`` is its own chunk instead, and its instruction is that
whole chunk's, without a time (:func:`pinned_instructions`): in the built-in
template it takes the place of the hand choreography
(:data:`ai_worker.speech_chunks.GESTURE_LINE` and ``SETTLE_LINE``), which
otherwise wins over it; a caller template gets it as an added line.

**Chunks.** H3 renders one window per pass, so the timeline is cut into chunks
that share their boundary frame and fit one render (at most
``MAX_CHUNK_SECONDS`` and the render cap): the fewest chunks that fit, and one
more when that makes its loudest cut quieter. Each chunk renders on the model's
``17k+5`` frame grid and keeps only its own span. A ``pin_chunk`` range is cut on
exactly its ``start`` and ``end`` frames and pinned on those two frames only, to
its ``pin_image_url`` (fitted to the input image's frame) or else the input
image; the quiet cuts and pause pins fill the stretches between pinned chunks
(:func:`pinned_spans`, :func:`freestyle_plan`).

**Validators.** ``detect_flicker`` (lighting jumps, punch-ins, cuts),
``detect_desync`` (SyncNet) and ``detect_glitches`` (a CNN for invented
graphics, sparkles and botched hands) are on unless the payload sets them
``false``. Each rendered chunk is checked (:mod:`ai_worker.lipsync_checks`) and one that fails is rendered
again on a new seed, up to ``DEFAULT_CHECK_RETRIES`` times; the best take is
kept and every take's scores are in the chunk's metrics. A custom instruction
with ``detect_flicker`` / ``detect_desync`` / ``detect_glitches`` set ``false``
turns that validator off for the chunks it goes to (:func:`checks_off`): a
stretch where the person does not speak the audio, or a generated effect, would
fail it on purpose.

**Attention.** By default every chunk samples with Comfy Kitchen INT8 attention
under the SLA block-sparse node
(:func:`ai_worker.minimax_h3_runtime.add_sparse_attention`), the attention the
student was trained with, where the worker supports it, and dense otherwise;
``sparse_attention: false`` asks for dense. The result reports what every chunk
ran.

Output frames are H3's native 24 fps; the caller's audio is muxed back
unmodified (the model never decodes audio in this mode).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .lipsync_prompt import render_prompt_template
from .compat import log_event
from .compat import job_dir
from .speech_cleanup import clean_voice
from .speech_chunks import (
    LANGUAGE_NAMES,
    MAX_CHUNK_SECONDS,
    Transcript,
    MIN_CHUNK_SECONDS,
    PIN_GAP_SECONDS,
    frame_loudness,
    load_audio_16k,
    pause_pins,
    quiet_cuts,
    render_prompt,
    transcribe,
    words_between,
)

#: H3 FL2VA's frame grid: the node snaps every length up to ``17k+5``.
H3_FRAME_GRID = 17
H3_FRAME_GRID_OFFSET = 5
#: Longest chunk, in frames: 226 frames (9.4 s) is the clip length the pinned
#: audio recipe was validated on.
DEFAULT_MAX_CHUNK_FRAMES = 226
#: Render budget in canvas pixels x frames. Activations grow with both, so a
#: large canvas gets shorter chunks. The default is the largest render measured
#: inside a 32 GB envelope: 226 frames at 736x1280 (12.3 GiB peak reserved, the
#: low_vram graph streaming the rest).
DEFAULT_MAX_PIXEL_FRAMES = 226 * 736 * 1280
DEFAULT_SEED = 42
#: ``detect_flicker`` / ``detect_desync`` / ``detect_glitches``: rerenders of a chunk that fails them
#: (``LIPSYNC_CHECK_RETRIES``), each on its own seed this far from the last.
DEFAULT_CHECK_RETRIES = 2
RESEED_STRIDE = 1000

# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


#: Pins of the job's own image; a pinned chunk's ``pin_image_url`` names its image instead.
INPUT_IMAGE = "input"


@dataclass(frozen=True)
class Anchor:
    """A pin of an image: the job's input image unless a pinned chunk names its own."""

    #: Output-timeline frame index.
    frame: int
    #: ``start`` (frame 0), ``pin`` (an end of a pinned chunk) or ``pause`` (every other pin).
    label: str
    #: :data:`INPUT_IMAGE` or the ``pin_image_url`` of the pinned chunk this pin ends.
    image: str = INPUT_IMAGE


@dataclass(frozen=True)
class PinnedSpan:
    """A caller's ``pin_chunk`` range: its own chunk, cut at exactly these frames,
    pinned to ``image`` on both ends and nowhere inside."""

    first: int
    last: int
    image: str = INPUT_IMAGE


def longest_chunk(*, fps: float, chunk_cap: int) -> int:
    """Most frames a chunk runs past its first: ``MAX_CHUNK_SECONDS``, within the render cap."""
    return min(int(MAX_CHUNK_SECONDS * fps), chunk_cap - 1)


def pinned_spans(items: list[dict[str, Any]], *, fps: float, total_frames: int, longest: int) -> list[PinnedSpan]:
    """The ``custom_instructions`` with ``pin_chunk`` as frame spans in time order: ``start`` and
    ``end`` on the nearest frames, a span running past the audio stopped at its last frame. Spans
    may touch (share a cut) when they pin the same image, never overlap, and each fits one render
    (``longest`` frames past its first)."""
    spans = []
    for item in items:
        if not item.get("pin_chunk"):
            continue
        start, end = float(item["start"]), float(item["end"])
        # Rounded so a time on a frame boundary is that frame despite float error.
        first, last = round(start * fps), min(round(end * fps), total_frames - 1)
        if last <= first:
            raise ValueError(f"pinned chunk {start}-{end} s has no frames inside the "
                             f"{total_frames / fps:.2f} s audio")
        if last - first > longest:
            raise ValueError(f"pinned chunk {start}-{end} s is longer than one render "
                             f"({longest / fps:.2f} s at this image size)")
        spans.append(PinnedSpan(first=first, last=last, image=str(item.get("pin_image_url") or INPUT_IMAGE)))
    spans.sort(key=lambda span: span.first)
    for previous, current in zip(spans, spans[1:]):
        if current.first < previous.last:
            raise ValueError(f"pinned chunks at {previous.first / fps:.2f} s and {current.first / fps:.2f} s overlap")
        if current.first == previous.last and current.image != previous.image:
            raise ValueError(f"pinned chunks meeting at {current.first / fps:.2f} s pin different images")
    return spans


def freestyle_plan(samples, *, total_frames: int, fps: float, chunk_cap: int,
                   pinned: list[PinnedSpan] = ()) -> tuple[list[int], list[int]]:
    """Freestyle's chunk cuts and pins from the cleaned 16 kHz voice: the cuts at
    the quietest frames (:func:`ai_worker.speech_chunks.quiet_cuts`, chunks of at
    most ``MAX_CHUNK_SECONDS`` and ``chunk_cap``), and the pins - every chunk's
    ends plus its inner pins every ``PIN_GAP_SECONDS`` at the quietest frames
    (:func:`ai_worker.speech_chunks.pause_pins`). Each ``pinned`` span is its own
    chunk with no inner pins, and the quiet cuts fill the stretches between them.
    Returns ``(cut_frames, pins)``, the pins sorted and including frame 0 and the
    last frame."""
    loudness = frame_loudness(samples, fps=fps, frames=total_frames)
    shortest, longest = math.ceil(MIN_CHUNK_SECONDS * fps), longest_chunk(fps=fps, chunk_cap=chunk_cap)
    last = total_frames - 1
    cuts: set[int] = set()
    free_start = 0
    for span in [*pinned, PinnedSpan(first=last, last=last)]:
        if span.first > free_start:
            cuts.update(free_start + cut for cut in quiet_cuts(loudness[free_start:span.first + 1],
                                                               shortest=shortest, longest=longest))
        cuts.update((span.first, span.last))
        free_start = span.last
    cut_frames = sorted(cut for cut in cuts if 0 < cut < last)
    bounds = [0, *cut_frames, last]
    inside_pinned = {(span.first, span.last) for span in pinned}
    pins = {0}
    for start, end in zip(bounds, bounds[1:]):
        if (start, end) in inside_pinned:
            pins.update((start, end))
            continue
        pins.update(start + pin for pin in pause_pins(loudness[start:end + 1], shortest=int(round(PIN_GAP_SECONDS[0] * fps)),
                                                      longest=int(round(PIN_GAP_SECONDS[1] * fps))))
    return cut_frames, sorted(pins)


def plan_anchors(pins: list[int], pinned: list[PinnedSpan]) -> list[Anchor]:
    """``pins`` as anchors: a pinned span's ends pin its image, every other pin the input image."""
    ends = {frame: span.image for span in pinned for frame in (span.first, span.last)}
    return [Anchor(frame=pin, label="start" if pin == 0 else "pin" if pin in ends else "pause",
                   image=ends.get(pin, INPUT_IMAGE)) for pin in pins]


# ---------------------------------------------------------------------------
# chunk planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LipsyncChunk:
    index: int
    #: First output frame this chunk renders (shared with the previous chunk).
    start_frame: int
    #: Output frames this chunk keeps (its last one is shared with the next chunk).
    frames: int
    #: ``(index inside the chunk, anchor)`` guides; the chunk's first frame is
    #: the timeline start (chunk 0) or the previous chunk's last frame.
    guides: tuple[tuple[int, Anchor], ...]

    @property
    def last_frame(self) -> int:
        return self.start_frame + self.frames - 1

    @property
    def render_frames(self) -> int:
        """Frames rendered: ``frames`` rounded up to the model's grid; the
        surplus past ``frames`` is dropped."""
        return grid_frames_ceil(self.frames)


def _pinned_chunk(item: dict[str, Any], chunks: list[LipsyncChunk], *, fps: float) -> LipsyncChunk:
    """A ``pin_chunk`` range's own chunk, cut on the frame nearest its start (:func:`pinned_spans`)."""
    start = float(item["start"])
    chunk = next((chunk for chunk in chunks if chunk.start_frame == round(start * fps)), None)
    if chunk is None:
        raise ValueError(f"no chunk starts at the pinned chunk's {start} s")
    return chunk


def _own_end(chunk: LipsyncChunk, chunks: list[LipsyncChunk], total_frames: int) -> int:
    """The end of the chunk's own frames: the cut where the next chunk starts, or the end of the audio."""
    return chunk.last_frame if chunk.index < len(chunks) - 1 else total_frames


def instruction_chunks(item: dict[str, Any], chunks: list[LipsyncChunk], *, fps: float,
                       total_frames: int) -> list[LipsyncChunk]:
    """The chunks a custom instruction goes to: a ``pin_chunk`` range's own chunk, the chunk playing at
    a ``timestamp`` (a cut frame belongs to the chunk it starts), or every chunk a ``start``-``end``
    range overlaps (one ending on a cut stops before the next chunk)."""
    point = item.get("timestamp") is not None
    start = float(item["timestamp"] if point else item["start"])
    end = start if point else float(item["end"])
    # In frames, rounded so a time on a frame boundary is that frame despite float error.
    first, last = round(start * fps, 6), round(end * fps, 6)
    if first >= total_frames:
        raise ValueError(f"custom instruction at {start} s is past the end of the {total_frames / fps:.2f} s audio")
    if item.get("pin_chunk"):
        return [_pinned_chunk(item, chunks, fps=fps)]
    return [chunk for chunk in chunks
            if (chunk.start_frame <= first < _own_end(chunk, chunks, total_frames) if point else
                first < _own_end(chunk, chunks, total_frames) and last > chunk.start_frame)]


def chunk_instructions(items: list[dict[str, Any]], chunks: list[LipsyncChunk], *, fps: float,
                       total_frames: int) -> list[list[str]]:
    """The caller's ``custom_instructions`` as prompt lines per chunk (:func:`instruction_chunks`), in
    time order. Times are seconds into the chunk, a range clipped to it, and a ``text_hint`` names the
    words: ``At 3.25 seconds, when speaking "<text_hint>": <instruction>``, ``During 1.5 to 6 seconds:
    ...``. ``pin_chunk`` ranges are left to :func:`pinned_instructions`."""
    def clock(seconds: float) -> str:
        return f"{max(0.0, seconds):.2f}".rstrip("0").rstrip(".")

    def begins(item: dict[str, Any]) -> float:
        return float(item["timestamp"] if item.get("timestamp") is not None else item["start"])

    per_chunk: list[list[str]] = [[] for _ in chunks]
    for item in sorted(items, key=begins):
        targets = instruction_chunks(item, chunks, fps=fps, total_frames=total_frames)
        if item.get("pin_chunk"):
            continue
        point = item.get("timestamp") is not None
        start = begins(item)
        end = start if point else float(item["end"])
        hint = " ".join(str(item.get("text_hint") or "").split())
        instruction = " ".join(str(item["instruction"]).split())
        for chunk in targets:
            offset = chunk.start_frame / fps
            own_end = _own_end(chunk, chunks, total_frames)
            when = (f"At {clock(start - offset)} seconds" if point else
                    f"During {clock(max(start, offset) - offset)} to {clock(min(end, own_end / fps) - offset)} seconds")
            if hint:
                when += f', when speaking "{hint}"'
            per_chunk[chunk.index].append(f"{when}: {instruction}")
    return per_chunk


def pinned_instructions(items: list[dict[str, Any]], chunks: list[LipsyncChunk], *, fps: float) -> dict[int, str]:
    """Each ``pin_chunk`` range's instruction by the index of its chunk. It is that whole chunk's, so it
    has no time: ``<instruction>``, or ``When speaking "<text_hint>": <instruction>``."""
    lines: dict[int, str] = {}
    for item in items:
        if not item.get("pin_chunk"):
            continue
        hint = " ".join(str(item.get("text_hint") or "").split())
        instruction = " ".join(str(item["instruction"]).split())
        lines[_pinned_chunk(item, chunks, fps=fps).index] = f'When speaking "{hint}": {instruction}' if hint else instruction
    return lines


#: The validators, by their payload names; a custom instruction setting one ``false`` turns it off for its chunks.
CHECKS = ("detect_flicker", "detect_desync", "detect_glitches")


def checks_off(items: list[dict[str, Any]], chunks: list[LipsyncChunk], *, fps: float,
               total_frames: int) -> list[set[str]]:
    """Per chunk, the validators (:data:`CHECKS`) that a custom instruction going to it
    (:func:`instruction_chunks`) turns off: a stretch where the person does not speak the audio, or a
    generated effect, light change or graphic the image does not have, would fail them on purpose."""
    off: list[set[str]] = [set() for _ in chunks]
    for item in items:
        names = {name for name in CHECKS if item.get(name) is False}
        if names:
            for chunk in instruction_chunks(item, chunks, fps=fps, total_frames=total_frames):
                off[chunk.index] |= names
    return off


def grid_frames_floor(frames: int) -> int:
    """Largest ``17k+5`` frame count that is ``<= frames`` (at least 5)."""
    return max(H3_FRAME_GRID_OFFSET, H3_FRAME_GRID * ((int(frames) - H3_FRAME_GRID_OFFSET) // H3_FRAME_GRID) + H3_FRAME_GRID_OFFSET)


def grid_frames_ceil(frames: int) -> int:
    """Smallest ``17k+5`` frame count that is ``>= frames``."""
    k = max(0, math.ceil((int(frames) - H3_FRAME_GRID_OFFSET) / H3_FRAME_GRID))
    return H3_FRAME_GRID * k + H3_FRAME_GRID_OFFSET


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def max_chunk_frames(width: int, height: int) -> int:
    """Longest chunk (on the frame grid) one pass may render at this canvas."""
    cap = _env_int("LIPSYNC_MAX_CHUNK_FRAMES", DEFAULT_MAX_CHUNK_FRAMES)
    budget = _env_int("LIPSYNC_MAX_PIXEL_FRAMES", DEFAULT_MAX_PIXEL_FRAMES)
    by_memory = budget // max(1, int(width) * int(height))
    return grid_frames_floor(max(H3_FRAME_GRID_OFFSET, min(cap, by_memory)))


def plan_chunks(total_frames: int, *, max_frames: int, anchors: list[Anchor], cut_frames: list[int]) -> list[LipsyncChunk]:
    """Chunks between ``cut_frames`` (each cut is the last frame of one chunk
    and the first of the next); every anchor but frame 0 is a guide exactly once."""
    if total_frames < 1:
        raise ValueError("lipsync needs at least one output frame")
    last = total_frames - 1
    bounds = [0, *sorted({int(c) for c in cut_frames if 0 < int(c) < last}), max(last, 0)]
    chunks: list[LipsyncChunk] = []
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        frames = end - start + 1
        if grid_frames_ceil(frames) > max_frames:
            raise ValueError(f"chunk {index} spans {frames} frames, over the {max_frames}-frame render cap")
        guides = tuple((anchor.frame - start, anchor) for anchor in anchors if start < anchor.frame <= end)
        chunks.append(LipsyncChunk(index=index, start_frame=start, frames=frames, guides=guides))
    return chunks


# ---------------------------------------------------------------------------
# media helpers
# ---------------------------------------------------------------------------


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def media_duration(path: Path) -> float:
    duration = float(_probe(path).get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise ValueError(f"Media has no positive duration: {path}")
    return duration


def prepare_input_image(image_path: Path, output_path: Path) -> Path:
    """The job's image as an upright RGB PNG: EXIF rotation applied, alpha dropped."""
    from PIL import Image, ImageOps  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        ImageOps.exif_transpose(image).convert("RGB").save(output_path, format="PNG")
    return output_path


def prepare_pin_image(image_path: Path, output_path: Path, *, size: tuple[int, int]) -> Path:
    """A pinned chunk's image as an upright RGB PNG of the input image's ``size`` (cover scale and
    centre crop), so the render fits it to the canvas exactly as it fits the input image."""
    from PIL import Image, ImageOps  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        upright = ImageOps.exif_transpose(image).convert("RGB")
        ImageOps.fit(upright, size, method=Image.Resampling.LANCZOS).save(output_path, format="PNG")
    return output_path


def extract_audio_slice(audio_path: Path, *, start_seconds: float, duration_seconds: float, output_path: Path) -> Path:
    """``duration_seconds`` of audio from ``start_seconds``, silence-padded past the end."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_seconds:.6f}", "-i", str(audio_path),
            "-vn", "-af", f"apad=whole_dur={duration_seconds:.6f}",
            "-t", f"{duration_seconds:.6f}", "-c:a", "pcm_s16le", str(output_path),
        ]
    )
    return output_path


def image_size(path: Path) -> tuple[int, int]:
    stream = next(s for s in _probe(path)["streams"] if s.get("codec_type") == "video")
    return int(stream["width"]), int(stream["height"])


def assemble_output(
    chunks: list[tuple[Path, int]],
    *,
    audio_path: Path,
    total_frames: int,
    fps: float,
    duration_seconds: float,
    output_path: Path,
    width: int | None,
    height: int | None,
    bounding_box: tuple[int, int] | None,
) -> None:
    """Join the chunks - ``(render, frames kept)``, dropping each shared boundary
    frame and every chunk's grid surplus - trim to the audio's length, scale,
    and mux the caller's audio.

    ``width``/``height`` deliver an exact frame size (cover scale + centre crop,
    never a squeeze); ``bounding_box`` instead fits the frame inside it.
    """
    inputs: list[str] = []
    parts: list[str] = []
    for index, (path, frames) in enumerate(chunks):
        inputs += ["-i", str(path)]
        parts.append(f"[{index}:v]select=between(n\\,{1 if index else 0}\\,{int(frames) - 1}),setpts=N/({fps}*TB)[v{index}]")
    joined = "".join(f"[v{index}]" for index in range(len(chunks)))
    if bounding_box is not None:
        scale = (
            f"scale={bounding_box[0]}:{bounding_box[1]}:force_original_aspect_ratio=decrease"
            ":force_divisible_by=2,setsar=1"
        )
    elif width and height:
        scale = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    else:
        scale = "setsar=1"
    graph = ";".join(parts) + f";{joined}concat=n={len(chunks)}:v=1:a=0,{scale}[out]"
    audio_index = len(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs, "-i", str(audio_path),
            "-filter_complex", graph,
            "-map", "[out]", "-map", f"{audio_index}:a:0",
            "-frames:v", str(int(total_frames)), "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-t", f"{duration_seconds:.6f}",
            "-movflags", "+faststart", str(output_path),
        ]
    )


# ---------------------------------------------------------------------------
# job I/O
# ---------------------------------------------------------------------------


def _download_url(job_id: str, url: str, path: Path, *, input_name: str, kind: str = "lipsync") -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{input_name} must use http or https")
    path.parent.mkdir(parents=True, exist_ok=True)
    log_event("job_input_download_started", job_id=job_id, kind=kind, input=input_name, url_host=parsed.netloc)
    with urlopen(Request(url, headers={"User-Agent": "vigen-api/0.1"}), timeout=120) as response:
        with path.open("wb") as file:
            shutil.copyfileobj(response, file, 1024 * 1024)
    log_event("job_input_download_complete", job_id=job_id, kind=kind, input=input_name, bytes=path.stat().st_size)
    return path


def download_job_input(
    job: dict[str, Any],
    payload: dict[str, Any],
    name: str,
    scratch_dir: Path,
    fallback_suffix: str,
    *,
    kind: str = "lipsync",
) -> Path:
    """Fetch a ``<name>_url`` / ``<name>_s3_key`` / ``<name>_path`` job input."""
    job_id = str(job["id"])
    url = payload.get(f"{name}_url")
    s3_key = payload.get(f"{name}_s3_key")
    if url:
        suffix = Path(urlparse(str(url)).path).suffix or fallback_suffix
        return _download_url(job_id, str(url), scratch_dir / "inputs" / f"{name}_url{suffix}", input_name=f"{name}_url", kind=kind)
    if s3_key:
        raise ValueError(f"{name}_s3_key is not supported here: pass {name}_url or {name}_path")
    if payload.get(f"{name}_path"):
        from .compat import project_path

        path = project_path(str(payload[f"{name}_path"]))
        if not path.exists():
            raise FileNotFoundError(f"{name}_path does not exist: {path}")
        return path
    raise ValueError(f"{kind} job payload requires {name}_url, {name}_s3_key or {name}_path")


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> Path:
    """Frame ``frame_index`` (0-based, decode order) of ``video_path`` as a PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vf", f"select=eq(n\\,{frame_index})",
          "-frames:v", "1", str(output_path)])
    if not output_path.is_file():
        raise ValueError(f"video has no frame {frame_index}: {video_path.name}")
    return output_path


def _check(render: Path, audio: Path, chunk: LipsyncChunk, *, flicker: bool, desync: bool, glitches: bool, image: Path,
           job_id: str, judge_drift: bool) -> dict[str, Any]:
    """The chunk's validator verdict; a validator that cannot run keeps the take and says why."""
    from .lipsync_checks import check_chunk

    try:
        return check_chunk(render, audio=audio, kept_frames=chunk.frames, flicker=flicker, desync=desync,
                           glitches=glitches, image=image, job_id=job_id, chunk_index=chunk.index,
                           judge_drift=judge_drift)
    except Exception as exc:  # a broken validator must not fail or reroll the render
        log_event("lipsync_chunk_check_failed", job_id=job_id, chunk=chunk.index, error=f"{type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}", "flagged": False, "badness": 0.0}


#: Layout of a paused job's ``result.progress`` (:mod:`ai_worker.job_checkpoint`); progress saved under
#: another version is ignored and the job starts over.
PROGRESS_VERSION = 1


def _new_plan(payload: dict[str, Any], *, voice_path: Path, audio_path: Path, input_image: Path,
              pin_images: int, fps: float, job_id: str, finish: Any) -> dict[str, Any]:
    """The job's plan - chunks, pins, prompts and per-chunk validators - as JSON, so a paused job
    resumes on exactly this plan wherever it is claimed next."""
    from .minimax_h3_runtime import resolve_canvas

    # Timing: the caller's timed transcript, else Whisper's word timestamps.
    began = time.perf_counter()
    caller_prompt = str(payload.get("prompt") or "").strip()
    caller_items = payload.get("transcript") or []
    custom_instructions = payload.get("custom_instructions") or []
    samples = load_audio_16k(voice_path)
    transcript = Transcript.from_items(list(caller_items)) if caller_items else transcribe(samples)
    transcript_source = "caller" if caller_items else "whisper"
    finish("speech_seconds", began)

    began = time.perf_counter()
    audio_duration = media_duration(audio_path)
    total_frames = max(1, int(round(audio_duration * fps)))
    source_width, source_height = image_size(input_image)
    canvas = resolve_canvas(image_path=input_image)
    chunk_cap = max_chunk_frames(canvas.width, canvas.height)
    # The input image on both ends of every chunk and at the pauses between; a pinned
    # chunk is cut on its own times and pinned on its ends only, to its own image if it has one.
    pinned = pinned_spans(custom_instructions, fps=fps, total_frames=total_frames,
                          longest=longest_chunk(fps=fps, chunk_cap=chunk_cap))
    cut_frames, pins = freestyle_plan(samples, total_frames=total_frames, fps=fps, chunk_cap=chunk_cap, pinned=pinned)
    anchors = plan_anchors(pins, pinned)
    chunks = plan_chunks(total_frames, max_frames=chunk_cap, anchors=anchors, cut_frames=cut_frames)
    pinned_chunks = sorted(chunk.index for chunk in chunks
                           if any((chunk.start_frame, chunk.last_frame) == (span.first, span.last) for span in pinned))
    chunk_texts = [
        words_between(transcript.words, chunk.start_frame / fps, chunk.last_frame / fps, last=chunk.index == len(chunks) - 1)
        for chunk in chunks
    ]
    # A pinned chunk's instruction is that chunk's: in the built-in template it replaces the hand
    # choreography that otherwise wins over it; a caller template gets it as its first added line.
    pinned_lines = pinned_instructions(custom_instructions, chunks, fps=fps)
    if caller_prompt:
        language_name = LANGUAGE_NAMES.get(transcript.language or "")
        prompts = [render_prompt_template(caller_prompt, text=text, language=language_name) for text in chunk_texts]
        prompts = ["\n".join([prompt, pinned_lines[i]]) if i in pinned_lines else prompt for i, prompt in enumerate(prompts)]
    else:
        prompts = [render_prompt(text, transcript.language, pinned_instruction=pinned_lines.get(i))
                   for i, text in enumerate(chunk_texts)]
    # The caller's timed extra instructions, each on its own line at the end of its chunks' prompts.
    timed_lines = chunk_instructions(custom_instructions, chunks, fps=fps, total_frames=total_frames)
    prompts = ["\n".join([prompt, *extra]) for prompt, extra in zip(prompts, timed_lines)]
    instructions = [([pinned_lines[i]] if i in pinned_lines else []) + extra for i, extra in enumerate(timed_lines)]
    finish("plan_seconds", began)
    log_event(
        "lipsync_planned",
        job_id=job_id,
        audio_seconds=round(audio_duration, 3),
        output_frames=total_frames,
        anchors=[anchor.frame for anchor in anchors],
        source_width=source_width,
        source_height=source_height,
        canvas_width=canvas.width,
        canvas_height=canvas.height,
        chunk_cap=chunk_cap,
        chunks=[(chunk.start_frame, chunk.frames) for chunk in chunks],
        chunk_seconds=[round(chunk.frames / fps, 3) for chunk in chunks],
        pinned_chunks=pinned_chunks,
        pin_images=pin_images,
        transcript_source=transcript_source,
        language=transcript.language,
    )
    return {
        "audio_duration": audio_duration,
        "total_frames": total_frames,
        "source_size": [source_width, source_height],
        "canvas": [canvas.width, canvas.height],
        "chunk_cap": chunk_cap,
        "cut_frames": list(cut_frames),
        "anchors": [[anchor.frame, anchor.label, anchor.image] for anchor in anchors],
        "pinned_chunks": pinned_chunks,
        "chunk_texts": chunk_texts,
        "prompts": prompts,
        "instructions": instructions,
        # Validators a custom instruction turns off for its chunks.
        "checks_off": [sorted(off) for off in checks_off(custom_instructions, chunks, fps=fps, total_frames=total_frames)],
        "transcript": {"text": transcript.text, "language": transcript.language, "source": transcript_source},
    }


def run_lipsync_inference(job: dict[str, Any]) -> dict[str, Any]:
    """Run one ``lipsync`` job; returns the result.
    """
    from .minimax_h3_runtime import (
        H3_FPS,
        LIPSYNC_LORA_NAME,
        LIPSYNC_SIGMAS,
        LIPSYNC_STEPS,
        cleanup_h3_job_files,
        run_lipsync_chunk,
    )

    job_started = time.perf_counter()
    payload = job["payload"]
    job_id = str(job["id"])
    seed = int(payload["seed"]) if payload.get("seed") is not None else DEFAULT_SEED
    sparse_attention = payload.get("sparse_attention") is not False
    detect_flicker = payload.get("detect_flicker") is not False
    detect_desync = payload.get("detect_desync") is not False
    detect_glitches = payload.get("detect_glitches") is not False
    checking = detect_flicker or detect_desync or detect_glitches
    check_retries = _env_int("LIPSYNC_CHECK_RETRIES", DEFAULT_CHECK_RETRIES)
    fps = float(H3_FPS)
    # A job paused at a checkpoint carries on from its saved progress.
    progress = None  # no pause/resume here: Taskflow schedules whole jobs
    stage_timings: dict[str, float] = dict(progress["stage_timings"]) if progress else {}

    def finish(name: str, began: float) -> None:
        stage_timings[name] = round(time.perf_counter() - began, 3)

    scratch_dir = job_dir(job_id)
    work_dir = scratch_dir / "lipsync"
    began = time.perf_counter()
    image_path = download_job_input(job, payload, "image", scratch_dir, ".png")
    audio_path = download_job_input(job, payload, "audio", scratch_dir, ".wav")
    custom_instructions = payload.get("custom_instructions") or []
    # The pinned chunks' own images, each URL fetched once.
    pin_downloads = {
        url: _download_url(job_id, url, scratch_dir / "inputs" / f"pin_image_{index}{Path(urlparse(url).path).suffix or '.png'}",
                           input_name="pin_image_url")
        for index, url in enumerate(dict.fromkeys(
            str(item["pin_image_url"]) for item in custom_instructions if item.get("pin_chunk") and item.get("pin_image_url")
        ))
    }
    if progress:
        voice_path = _download_url(job_id, progress["voice"]["download_url"], work_dir / "voice.wav",
                                   input_name="progress_voice")
    finish("input_download_seconds", began)

    # The job's image, the only image the render sees.
    input_image = prepare_input_image(image_path, work_dir / "anchors" / "input.png")
    source_width, source_height = image_size(input_image)
    images = {INPUT_IMAGE: input_image, **{
        url: prepare_pin_image(path, work_dir / "anchors" / f"pin{index}.png", size=(source_width, source_height))
        for index, (url, path) in enumerate(pin_downloads.items())
    }}
    if progress:
        plan = progress["plan"]
        log_event("lipsync_resumed", job_id=job_id, chunks_done=len(progress["chunks"]),
                  chunks=len(plan["chunk_texts"]), claims=progress["claims"])
    else:
        # The voice alone - no music, no noise - is what the rest of the job hears.
        began = time.perf_counter()
        voice_path = clean_voice(audio_path, work_dir / "voice.wav", job_id=job_id)
        finish("voice_cleanup_seconds", began)
        plan = _new_plan(payload, voice_path=voice_path, audio_path=audio_path, input_image=input_image,
                         pin_images=len(pin_downloads), fps=fps, job_id=job_id, finish=finish)
    audio_duration = float(plan["audio_duration"])
    total_frames = int(plan["total_frames"])
    canvas_width, canvas_height = plan["canvas"]
    anchors = [Anchor(frame=int(frame), label=str(label), image=str(image)) for frame, label, image in plan["anchors"]]
    anchor_at = {anchor.frame: anchor for anchor in anchors}
    chunks = plan_chunks(total_frames, max_frames=int(plan["chunk_cap"]), anchors=anchors, cut_frames=plan["cut_frames"])
    pinned_chunks = set(plan["pinned_chunks"])
    chunk_texts, prompts, instructions = plan["chunk_texts"], plan["prompts"], plan["instructions"]
    claims = (progress["claims"] if progress else 0) + 1

    # The chunks rendered in earlier claims.
    done = list(progress["chunks"]) if progress else []
    chunk_paths: list[tuple[Path, int]] = []
    chunk_metrics: list[dict[str, Any]] = []
    for entry in done:
        index = int(entry["index"])
        path = _download_url(job_id, entry["download_url"], work_dir / f"chunk{index:03d}" / "kept.mp4",
                             input_name="progress_chunk")
        chunk_paths.append((path, chunks[index].frames))
        chunk_metrics.append(entry["metrics"])

    claim_output_seconds = 0.0
    try:
        for chunk in chunks[len(done):]:
            began = time.perf_counter()
            chunk_dir = work_dir / f"chunk{chunk.index:03d}"
            audio_slice = extract_audio_slice(
                voice_path,
                start_seconds=chunk.start_frame / fps,
                duration_seconds=chunk.render_frames / fps,
                output_path=chunk_dir / "audio.wav",
            )
            # Every chunk opens on a pin and ends on one at its last kept frame (not the
            # grid surplus past it): the input image, or a pinned chunk's own image.
            first_image = images[anchor_at[chunk.start_frame].image]
            guides = [(position, images[anchor.image]) for position, anchor in chunk.guides]
            guide_labels = [f"{position}:{anchor.label}" for position, anchor in chunk.guides]
            # With validators on, a chunk that fails them is rendered again on a new
            # seed; the pins make any take fit between its neighbours. A custom
            # instruction may turn some off for its chunks.
            off = set(plan["checks_off"][chunk.index])
            flicker = detect_flicker and "detect_flicker" not in off
            desync = detect_desync and "detect_desync" not in off
            glitches = detect_glitches and "detect_glitches" not in off
            chunk_checking = flicker or desync or glitches
            attempts: list[tuple[Path, dict[str, Any], dict[str, Any] | None]] = []
            for attempt in range(1 + (check_retries if chunk_checking else 0)):
                output = chunk_dir / ("render.mp4" if attempt == 0 else f"render_retry{attempt}.mp4")
                metrics = run_lipsync_chunk(
                    job_id=job_id,
                    chunk_index=chunk.index,
                    prompt=prompts[chunk.index],
                    seed=seed + chunk.index + attempt * RESEED_STRIDE,
                    width=canvas_width,
                    height=canvas_height,
                    length=chunk.render_frames,
                    first_frame_path=first_image,
                    last_frame_path=None,
                    guides=guides,
                    audio_path=audio_slice,
                    output_path=output,
                    sparse_attention=sparse_attention,
                )
                # Glitches are judged against the image the chunk opens on; a chunk moving into another
                # pinned image drifts from it on purpose, so its flicker check judges sudden steps only.
                verdict = _check(output, audio_slice, chunk, flicker=flicker, desync=desync,
                                 glitches=glitches, image=first_image, job_id=job_id,
                                 judge_drift={first_image, *(path for _, path in guides)} == {first_image},
                                 ) if chunk_checking else None
                attempts.append((output, metrics, verdict))
                if verdict is None or not verdict["flagged"]:
                    break
            kept = min(range(len(attempts)), key=lambda i: attempts[i][2]["badness"] if attempts[i][2] else 0.0)
            output, metrics, verdict = attempts[kept]
            stage_timings[f"chunk_{chunk.index}_seconds"] = round(time.perf_counter() - began, 3)
            chunk_paths.append((output, chunk.frames))
            chunk_metrics.append({
                **metrics,
                "start_frame": chunk.start_frame,
                "kept_frames": chunk.frames,
                "text": chunk_texts[chunk.index],
                **({"pinned": True} if chunk.index in pinned_chunks else {}),
                **({"instructions": instructions[chunk.index]} if instructions[chunk.index] else {}),
                "anchors": guide_labels,
                **({"checks": [attempt_verdict for _, _, attempt_verdict in attempts if attempt_verdict is not None],
                    "kept_attempt": kept, "flagged": bool(verdict and verdict["flagged"])} if checking else {}),
                **({"checks_off": sorted(off)} if checking and off else {}),
                # The claim that rendered it: 1, or later when the job paused for other work.
                "claim": claims,
            })
            claim_output_seconds += chunk.frames / fps
    finally:
        cleanup_h3_job_files(job_id)

    output_dimensions = payload.get("output_dimensions")
    bounding_box = (
        (int(output_dimensions["width"]), int(output_dimensions["height"]))
        if isinstance(output_dimensions, dict)
        else None
    )
    delivered_width = source_width - source_width % 2
    delivered_height = source_height - source_height % 2
    began = time.perf_counter()
    output_name = "output.mp4"
    output_path = scratch_dir / "outputs" / output_name
    assemble_output(
        chunk_paths,
        audio_path=audio_path,
        total_frames=total_frames,
        fps=fps,
        duration_seconds=audio_duration,
        output_path=output_path,
        width=None if bounding_box else delivered_width,
        height=None if bounding_box else delivered_height,
        bounding_box=bounding_box,
    )
    finish("assemble_seconds", began)
    if bounding_box is not None:
        delivered_width, delivered_height = image_size(output_path)

    began = time.perf_counter()
    uploaded = {"download_url": None, "size": output_path.stat().st_size, "etag": None}
    finish("upload_seconds", began)

    total_seconds = (progress["wall_seconds"] if progress else 0.0) + time.perf_counter() - job_started
    render_seconds = sum(float(m["pipeline_seconds"]) for m in chunk_metrics)
    transcript = plan["transcript"]
    item = {
        "seed": seed,
        "fps": fps,
        "width": delivered_width,
        "height": delivered_height,
        "frames": total_frames,
        "duration_seconds": round(audio_duration, 3),
        "source": "freestyle",
        "anchors": len(anchors),
        "pinned_chunks": len(pinned_chunks),
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "chunks": len(chunks),
        "steps": LIPSYNC_STEPS,
        "sigmas": LIPSYNC_SIGMAS,
        "turbo_lora": LIPSYNC_LORA_NAME,
        "sparse_attention": sparse_attention and all(metric["sparse_attention"] for metric in chunk_metrics),
        "checks": {
            "detect_flicker": detect_flicker,
            "detect_desync": detect_desync,
            "detect_glitches": detect_glitches,
            "rerendered_chunks": sum(len(metric["checks"]) > 1 for metric in chunk_metrics),
            "flagged_chunks": sum(metric["flagged"] for metric in chunk_metrics),
        } if checking else None,
        "prompt_source": "payload" if str(payload.get("prompt") or "").strip() else "template",
        "transcript_source": transcript["source"],
        "language": transcript["language"],
        "transcript": transcript["text"],
    }
    if bounding_box is not None:
        item["output_dimensions"] = {"width": bounding_box[0], "height": bounding_box[1]}
    log_event("lipsync_complete", job_id=job_id, wall_seconds=round(total_seconds, 3), claims=claims, **{
        key: item[key] for key in ("frames", "chunks", "anchors", "width", "height")
    })
    return {
        "outputs": [
            {
                "name": output_name,
                "path": str(output_path),
                "size": uploaded["size"],
                "etag": uploaded["etag"],
            }
        ],
        "item": item,
        "metrics": {
            "engine": "comfyui_lipsync",
            "wall_seconds": round(total_seconds, 3),
            "render_seconds": round(render_seconds, 3),
            "wall_rtf": round(total_seconds / audio_duration, 3) if audio_duration else None,
            "render_rtf": round(render_seconds / audio_duration, 3) if audio_duration else None,
            # Claims the job was rendered over: more than one when it paused for other work.
            "claims": claims,
            "stage_timings": stage_timings,
            "chunks": chunk_metrics,
        },
        "seed": seed,
    }
