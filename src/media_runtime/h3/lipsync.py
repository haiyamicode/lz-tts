"""Lipsync: one image of a person plus audio in, a talking video out.

The video is generated with the lipsync student - our 3-step
first-fold avatar LoRA (:data:`h3.minimax_h3_runtime.LIPSYNC_LORA_NAME`)
- while the job's audio is pinned in the latent
(:func:`h3.minimax_h3_runtime.build_lipsync_graph`), so face, head,
hands and body all move with the speech and nothing is composited:

- the output lasts as long as the audio;
- the output has the image's size, or fits inside ``output_dimensions`` with
  its aspect kept;
- the result carries ``path`` / ``item`` / ``metrics`` / ``seed``.

**Auto** (``auto``, the default; :func:`auto_payload`): the avatar render. The caller
sends the person's canonical image, the audio and the avatar's rest ``stills`` (its
imagegen ``rest_frame`` stills, any number; none = the image itself), and the job
runs on our setup, the same for every length and every audio: the default preset
(``turbo-3``) with sparse attention, every validator on, the built-in prompt, and
H3's native 24 fps (no interpolation: an interpolated lipsync renders its frames
from a sped-up voice, so every motion the voice does not drive - blinks, head
turns, hands - plays back ``24/interpolation_fps`` times slow). Pins: a clip of
``AUTO_SHORT_SECONDS`` (3 s) or more pins the stills as ``multi_image`` ``random``
(one still: ``single_image``); a shorter clip is too short to travel between two
pins (an end pin froze it) and pins frame 0 alone (``none``). ``seed``,
``transcript``, ``custom_instructions`` (``pin_chunk`` spans and per-instruction
validator switches included), ``output_dimensions`` and ``upscale`` work as
always; the knobs auto chooses (:data:`AUTO_CHOSEN`) are an error with it -
``auto: false`` (manual) takes them, and every mode below, instead of ``stills``.
Addons: a plain auto job (no ``addons`` key, no ``custom_instructions``) of a photo
avatar (:func:`h3.rest_stills.clip_addons`, stylized < ``MAX_STYLIZED``) renders with
``realism``; a stylized one, or any custom job, without. ``generate`` rest stills never
take addons (``stills_addons: []``). Caller ``addons`` (``[]`` too)
decide; ``auto_choices`` reports ``addons`` (+ ``stylized`` when checked).

**Freestyle.** The job's image (``image_url`` / ``image_path``) is the first
frame. The chunks are cut at the quietest frames of the voice
(:func:`h3.speech_chunks.quiet_cuts`), and ``pinned_frames`` (:data:`PIN_MODES`,
:class:`PinSettings`) says what else is pinned:

- ``none`` (default): nothing. Every chunk boundary is pinned to the previous
  chunk's own last frame, restored toward the image last pinned before it
  (:mod:`h3.h3_restore`, vigen e9791cc), so the motion carries on; the last frame
  is free.
- ``single_image``: one image (the job's, or ``image_url``) on every chunk's
  first and last frame and inside it every ``speech_chunks.PIN_GAP_SECONDS``
  (2.5-5 s) at the quietest frames in reach (:func:`h3.speech_chunks.pause_pins`)
  - the middle of a pause where there is one, else the lowest speech energy - so a
  pose or framing the model invents never outlasts the next pin.
- ``multi_image``: the same pins, their images taken in turn from ``image_urls``:
  ``order`` ``round_robin`` (1, 2, 3, 1, 2, ...; the default) or ``random``
  (seeded by the job's seed, never the same image twice running).
- ``timed_images``: the caller's ``frames`` (``image_url`` at ``seconds``) and
  nothing else; boundaries as with ``none``, except that a cut with a pin in reach lands
  on it (it then joins on the caller's image, no restored frame needed).
- ``generate`` (vigen 9ff3052): the job first renders ``count`` (2-12, default 6) rest
  stills of the person from its own image (:func:`h3.rest_stills.render_still`, the
  imagegen ``rest_frame`` stills: still k on ``seed + k``, retaken at most ``max_retries``
  times on a lost face or open mouth, the best take kept either way) and pins them like
  ``multi_image`` (``order``); they are delivered with the video as the result's
  ``still<k>`` files.

Every pinned image is fitted to the job's image's frame (cover scale, centre
crop). Motion and gestures between the pins come from the prompt and the pinned
audio.

**Voice.** The render never hears the raw audio: room tone and noise make H3
invent things (overlay graphics, punch-ins) and blur the pauses the pins go in.
:func:`h3.speech_cleanup.clean_voice` denoises it (ZipEnhancer; no music
separation) into a 16 kHz voice, and that voice is what is transcribed, what
places the cuts and pins, and what is pinned into every chunk's latent. The
caller's audio is what the job delivers.

**Speech.** Each chunk's words come from a timed transcript: the caller's
``transcript`` (text items with start/end seconds), or a greedy CTC
transcription (:func:`h3.speech_chunks.transcribe`, romanized words) when there
is none. Unless the caller overrides ``prompt``, every chunk is
prompted with :data:`h3.speech_chunks.PROMPT_TEMPLATE` (general for any
person on camera) and the words spoken in it; a caller ``prompt`` is a template
of the same kind (:mod:`h3.lipsync_prompt`: ``{{text}}`` required,
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
(:data:`h3.speech_chunks.GESTURE_LINE` and ``SETTLE_LINE``), which
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

**Staged jobs** (``stage``; :func:`run_lipsync_inference`). A whole job holds the GPU
for every chunk of up to 300 s of audio; staged, each step is a task of its own, so
other jobs' chunks take turns between them. ``plan`` (the request plus ``stage``) does
everything before the first chunk and returns the state tar (payload after auto, the
plan, the cleaned voice, the pin images) with the job's audio as the ``audio`` file
and ``chunk_plan``. ``chunk`` (``chunk_index``, the state, and the previous chunk's
``boundary`` file when ``needs_boundary``) renders that chunk exactly as the whole job
does - same seeds, validators, retries - and returns its kept take plus, when it ends on
a restored boundary, that frame as ``boundary``; chunks run in order, since chunk k
opens on chunk k-1's boundary. ``assemble`` (the state, every chunk's render in order,
the plan's ``audio``, optionally the chunks' ``report`` as ``chunk_reports``) joins,
sizes and muxes them into the whole job's result. ``generate`` is not staged: render
its stills as imagegen ``rest_frame`` tasks and pin them.

**Validators.** ``detect_flicker`` (lighting jumps, punch-ins, cuts),
``detect_desync`` (SyncNet) and ``detect_glitches`` (a CNN for invented
graphics, sparkles and botched hands) are on unless the payload sets them
``false``. Each rendered chunk is checked (:mod:`h3.lipsync_checks`) and one that fails is rendered
again on a new seed, up to ``DEFAULT_CHECK_RETRIES`` times; the best take is
kept and every take's scores are in the chunk's metrics. A custom instruction
with ``detect_flicker`` / ``detect_desync`` / ``detect_glitches`` set ``false``
turns that validator off for the chunks it goes to (:func:`checks_off`): a
stretch where the person does not speak the audio, or a generated effect, would
fail it on purpose.

**Attention.** By default every chunk samples with Comfy Kitchen INT8 attention
under the SLA block-sparse node
(:func:`h3.minimax_h3_runtime.add_sparse_attention`), the attention the
student was trained with, where the worker supports it, and dense otherwise;
``sparse_attention: false`` asks for dense. The result reports what every chunk
ran.

**Interpolation** (vigen 7280e2f). ``use_interpolation`` renders ``interpolation_fps``
(default 16) frames for every second of the audio: the plan's frames run at that
rate, each chunk renders its frames on the model's own 24 fps clock from its voice
sped up ``24/interpolation_fps`` (so a chunk covers that much more of the audio),
its validators judge that render, and RIFE brings the joined video up to 24 fps
(:mod:`h3.h3_interpolation`).

**Upscale** (vigen 43c32e7). ``upscale`` (``true`` = 1.5x, or a factor above 1 and
at most 2) delivers the image's size times the factor (longest side at most 3840,
``output_dimensions`` still bounds it): the joined frames go through a 2x SPAN
model, loaded for that job only, and lanczos resamples them to that size
(:mod:`h3.span_upscale`, after RIFE when both are asked for).

Output frames are H3's native 24 fps; the caller's audio is muxed back
unmodified (the model never decodes audio in this mode).
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

import cancel
import progress

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

#: H3's frame grid: the node snaps every length up to ``17k+5``.
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


#: Longest audio a job may lipsync (``LIPSYNC_MAX_SECONDS`` overrides).
DEFAULT_MAX_SECONDS = 300.0
#: Largest image / audio a job may fetch.
MAX_INPUT_BYTES = 200 * 1024 * 1024

#: Pins of the job's own image; every other pinned image is keyed by its URL.
INPUT_IMAGE = "input"
#: ``pinned_frames.mode`` (the module docstring): ``none`` (default), ``single_image``, ``multi_image``,
#: ``timed_images``, ``generate``.
PIN_MODES = ("none", "single_image", "multi_image", "timed_images", "generate")
#: ``pinned_frames.order`` of a ``multi_image`` / ``generate`` rotation.
PIN_ORDERS = ("round_robin", "random")
MAX_PIN_IMAGES = 16
MAX_FRAME_PINS = 100
#: ``generate``: rest stills the job renders, by default and at most (vigen's ``pinned_frames`` ``generate``).
DEFAULT_GENERATED_STILLS = 6
MAX_GENERATED_STILLS = 12
#: Image key of a ``generate`` rest still: ``generated:<k>``.
GENERATED_IMAGE = "generated:"
#: ``auto``: the request fields auto mode sets itself; a caller who sets one turns ``auto`` off.
AUTO_CHOSEN = ("pinned_frames", "preset", "use_interpolation", "interpolation_fps", "sparse_attention",
               "detect_flicker", "detect_desync", "detect_glitches", "prompt")
#: ``auto``: a clip shorter than this pins frame 0 alone (too short to travel between two pins).
AUTO_SHORT_SECONDS = 3.0
#: Image key of a continuous boundary's restored frame: ``restored:<frame>``, filled in when rendered.
RESTORED_IMAGE = "restored:"


@dataclass(frozen=True)
class PinSettings:
    """A job's ``pinned_frames``: its ``mode``, the ``images`` its boundary and pause pins take
    (``single_image``: one, :data:`INPUT_IMAGE` unless ``image_url``; ``multi_image``: the rotation, in
    ``order``; ``generate``: its ``count`` rest stills, :data:`GENERATED_IMAGE` keys, rendered with at most
    ``max_retries`` retakes each) and a ``timed_images`` job's ``frames`` as ``(seconds, image URL)``."""

    mode: str = "none"
    images: tuple[str, ...] = ()
    order: str = PIN_ORDERS[0]
    frames: tuple[tuple[float, str], ...] = ()
    max_retries: int = 0

    @property
    def continuous(self) -> bool:
        """No boundary or pause pins of the caller's: each boundary is the previous chunk's restored frame."""
        return self.mode in ("none", "timed_images")

    @property
    def urls(self) -> list[str]:
        """Every image URL these pins need, once each."""
        return list(dict.fromkeys(image for image in (*self.images, *(url for _, url in self.frames))
                                  if image != INPUT_IMAGE and not image.startswith(GENERATED_IMAGE)))

    @classmethod
    def from_payload(cls, value: dict[str, Any] | None) -> PinSettings:
        value = dict(value or {})
        mode = str(value.get("mode") or "none")
        if mode not in PIN_MODES:
            raise ValueError(f"pinned_frames.mode must be one of {', '.join(PIN_MODES)}, got {mode!r}")
        given = {key: value.get(key) for key in ("image_url", "image_urls", "order", "frames", "count", "max_retries")}
        allowed = {"none": (), "single_image": ("image_url",), "multi_image": ("image_urls", "order"),
                   "timed_images": ("frames",), "generate": ("count", "order", "max_retries")}[mode]
        extra = [key for key, item in given.items() if item not in (None, "", []) and key not in allowed]
        if extra:
            raise ValueError(f"pinned_frames mode {mode} takes no {', '.join(extra)}")
        if mode == "none":
            return cls()
        if mode == "single_image":
            return cls(mode=mode, images=(str(given["image_url"] or INPUT_IMAGE),))
        if mode in ("multi_image", "generate"):
            order = str(given["order"] or PIN_ORDERS[0])
            if order not in PIN_ORDERS:
                raise ValueError(f"pinned_frames.order must be one of {', '.join(PIN_ORDERS)}, got {order!r}")
        if mode == "multi_image":
            images = tuple(str(url) for url in given["image_urls"] or [])
            if not 2 <= len(images) <= MAX_PIN_IMAGES:
                raise ValueError(f"multi_image pinned_frames takes 2 to {MAX_PIN_IMAGES} image_urls, got {len(images)}")
            return cls(mode=mode, images=images, order=order)
        if mode == "generate":
            from .rest_stills import DEFAULT_RETRIES, MAX_RETRIES

            count = DEFAULT_GENERATED_STILLS if given["count"] is None else int(given["count"])
            if not 2 <= count <= MAX_GENERATED_STILLS:
                raise ValueError(f"generate pinned_frames takes a count of 2 to {MAX_GENERATED_STILLS}, got {count}")
            retries = DEFAULT_RETRIES if given["max_retries"] is None else int(given["max_retries"])
            if not 0 <= retries <= MAX_RETRIES:
                raise ValueError(f"pinned_frames.max_retries must be within 0..{MAX_RETRIES}, got {retries}")
            return cls(mode=mode, images=tuple(f"{GENERATED_IMAGE}{k}" for k in range(count)), order=order,
                       max_retries=retries)
        frames = list(given["frames"] or [])
        if not 1 <= len(frames) <= MAX_FRAME_PINS:
            raise ValueError(f"timed_images pinned_frames takes 1 to {MAX_FRAME_PINS} frames, got {len(frames)}")
        return cls(mode=mode, frames=tuple((float(frame["seconds"]), str(frame["image_url"])) for frame in frames))


@dataclass(frozen=True)
class Anchor:
    """A pin of an image: the job's input image unless a pinned chunk names its own."""

    #: Output-timeline frame index.
    frame: int
    #: ``start`` (frame 0), ``pin`` (an end of a pinned chunk), ``frame`` (a ``timed_images`` pin), ``boundary`` (a
    #: continuous chunk boundary) or ``pause`` (a ``single_image`` / ``multi_image`` boundary or pause pin).
    label: str
    #: :data:`INPUT_IMAGE`, an image URL, or a continuous boundary's ``restored:<frame>``.
    image: str = INPUT_IMAGE


@dataclass(frozen=True)
class FramePin:
    """A ``timed_images`` pin: ``image`` (its URL) on output frame ``frame``."""

    frame: int
    image: str


@dataclass(frozen=True)
class PinnedSpan:
    """A caller's ``pin_chunk`` range: its own chunk, cut at exactly these frames,
    pinned to ``image`` on both ends and with no boundary or pause pins inside."""

    first: int
    last: int
    image: str = INPUT_IMAGE


def longest_chunk(*, fps: float, chunk_cap: int) -> int:
    """Most frames a chunk runs past its first: ``MAX_CHUNK_SECONDS`` at ``fps`` - the model's 24 fps, whatever
    rate the timeline runs at (an interpolated job's chunks render as many frames and last longer) - within the
    render cap."""
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


def frame_pins(settings: PinSettings, *, fps: float, total_frames: int, pinned: list[PinnedSpan] = ()) -> list[FramePin]:
    """A ``timed_images`` job's ``frames`` on the nearest frames of the ``fps`` timeline, in time order. Frame 0 is
    the job's own image and no two pins share a frame; a pin on a pinned chunk's end must pin that chunk's
    image (it is then that end)."""
    pins: dict[int, str] = {}
    for index, (seconds, image) in enumerate(settings.frames):
        # Rounded so a time on a frame boundary is that frame despite float error.
        frame = round(seconds * fps)
        where = f"pinned_frames.frames[{index}] at {seconds:g} s"
        if frame < 1:
            raise ValueError(f"{where} falls on frame 0, which is the job's image_url")
        if frame > total_frames - 1:
            raise ValueError(f"{where} is past the end of the {total_frames / fps:.2f} s audio")
        if frame in pins:
            raise ValueError(f"{where} falls on the same frame as another pin")
        pins[frame] = image
    ends = {frame: span.image for span in pinned for frame in (span.first, span.last)}
    for frame, image in pins.items():
        if frame in ends and ends[frame] != image:
            raise ValueError(f"the pin at {frame / fps:.2f} s falls on a pinned chunk's end, which pins another image")
    return [FramePin(frame=frame, image=image) for frame, image in sorted(pins.items()) if frame not in ends]


def auto_payload(payload: dict[str, Any], *, seconds: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The manual payload an ``auto`` job (the default) runs as, for audio ``seconds`` long, and what auto
    chose (``None`` for a manual job, which is returned as it is). Raises ``ValueError`` for an auto job
    that sets a field auto chooses (:data:`AUTO_CHOSEN`), or a manual one that sends ``stills``."""
    if payload.get("auto", True) is False:
        if payload.get("stills"):
            raise ValueError("stills go with auto; with auto false, pin them with pinned_frames")
        return {key: value for key, value in payload.items() if key != "auto"}, None
    chosen = [key for key in AUTO_CHOSEN if key in payload]
    if chosen:
        raise ValueError(f"auto chooses {', '.join(chosen)}; send auto: false to set them")
    stills = list(dict.fromkeys(str(url) for url in payload.get("stills") or []))
    if len(stills) > MAX_PIN_IMAGES:
        raise ValueError(f"auto takes at most {MAX_PIN_IMAGES} stills, got {len(stills)}")
    resolved = {key: value for key, value in payload.items() if key not in ("auto", "stills")}
    if seconds < AUTO_SHORT_SECONDS:
        pinned = {"mode": "none"}
    elif len(stills) > 1:
        pinned = {"mode": "multi_image", "image_urls": stills, "order": "random"}
    else:
        pinned = {"mode": "single_image", **({"image_url": stills[0]} if stills else {})}
    resolved["pinned_frames"] = pinned
    return resolved, {"stills": len(stills), "pin_mode": pinned["mode"], "short": seconds < AUTO_SHORT_SECONDS}


def freestyle_plan(samples, *, total_frames: int, fps: float, chunk_cap: int,
                   pinned: list[PinnedSpan] = (), continuous: bool = False, fixed: list[FramePin] = (),
                   model_fps: float | None = None) -> tuple[list[int], list[int]]:
    """Freestyle's chunk cuts and pins from the cleaned 16 kHz voice: the cuts at
    the quietest frames (:func:`h3.speech_chunks.quiet_cuts`, chunks of at
    most ``MAX_CHUNK_SECONDS`` of the model's ``model_fps`` frames (default ``fps``)
    and ``chunk_cap``), and the pins - every chunk's
    ends plus its inner pins every ``PIN_GAP_SECONDS`` at the quietest frames
    (:func:`h3.speech_chunks.pause_pins`). Each ``pinned`` span is its own
    chunk with no inner pins, and the quiet cuts fill the stretches between them.
    Returns ``(cut_frames, pins)``, the pins sorted and including frame 0 and the
    last frame. ``continuous`` boundaries pin only frame 0, the cuts and the pinned
    spans' ends: no pauses, and the last frame is free. The ``fixed`` pins (a
    ``timed_images`` job's) are pins too, and a cut with one in reach lands on it."""
    loudness = frame_loudness(samples, fps=fps, frames=total_frames)
    shortest, longest = math.ceil(MIN_CHUNK_SECONDS * fps), longest_chunk(fps=model_fps or fps, chunk_cap=chunk_cap)
    last = total_frames - 1
    fixed_frames = [pin.frame for pin in fixed]
    cuts: set[int] = set()
    free_start = 0
    for span in [*pinned, PinnedSpan(first=last, last=last)]:
        if span.first > free_start:
            preferred = [frame - free_start for frame in fixed_frames if free_start < frame < span.first]
            cuts.update(free_start + cut for cut in quiet_cuts(loudness[free_start:span.first + 1],
                                                               shortest=shortest, longest=longest, preferred=preferred))
        cuts.update((span.first, span.last))
        free_start = span.last
    cut_frames = sorted(cut for cut in cuts if 0 < cut < last)
    if continuous:
        ends = {frame for span in pinned for frame in (span.first, span.last)}
        return cut_frames, sorted({0, *cut_frames, *ends, *fixed_frames})
    bounds = [0, *cut_frames, last]
    inside_pinned = {(span.first, span.last) for span in pinned}
    pins = {0}
    for start, end in zip(bounds, bounds[1:]):
        if (start, end) in inside_pinned:
            pins.update((start, end))
            continue
        pins.update(start + pin for pin in pause_pins(loudness[start:end + 1], shortest=int(round(PIN_GAP_SECONDS[0] * fps)),
                                                      longest=int(round(PIN_GAP_SECONDS[1] * fps))))
    return cut_frames, sorted({*pins, *fixed_frames})


def plan_anchors(pins: list[int], pinned: list[PinnedSpan], *, continuous: bool = False, fixed: list[FramePin] = (),
                 images: tuple[str, ...] = (INPUT_IMAGE,), order: str = PIN_ORDERS[0], seed: int = 0) -> list[Anchor]:
    """``pins`` as anchors: frame 0 the input image, a pinned span's ends its image, a ``fixed`` pin its own,
    and every other pin - with ``continuous`` boundaries the frame restored there when the chunk ending on it
    renders - else the next of ``images`` in ``order``: ``round_robin``, or ``random`` (``seed``; never the
    image the pin before took, when there is another)."""
    ends = {frame: span.image for span in pinned for frame in (span.first, span.last)}
    own = {pin.frame: pin.image for pin in fixed}
    rng = random.Random(seed)
    turn, previous = 0, None
    anchors = []
    for pin in pins:
        if pin == 0 or pin in ends:
            anchors.append(Anchor(frame=pin, label="start" if pin == 0 else "pin", image=ends.get(pin, INPUT_IMAGE)))
        elif pin in own:
            anchors.append(Anchor(frame=pin, label="frame", image=own[pin]))
        elif continuous:
            anchors.append(Anchor(frame=pin, label="boundary", image=f"{RESTORED_IMAGE}{pin}"))
        else:
            if order == "random":
                image = rng.choice([candidate for candidate in images if candidate != previous] or list(images))
            else:
                image = images[turn % len(images)]
                turn += 1
            previous = image
            anchors.append(Anchor(frame=pin, label="pause", image=image))
    return anchors


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
                       total_frames: int, model_fps: float | None = None) -> list[list[str]]:
    """The caller's ``custom_instructions`` as prompt lines per chunk (:func:`instruction_chunks`), in
    time order. Times are seconds into the chunk, a range clipped to it, and a ``text_hint`` names the
    words: ``At 3.25 seconds, when speaking "<text_hint>": <instruction>``, ``During 1.5 to 6 seconds:
    ...``. ``pin_chunk`` ranges are left to :func:`pinned_instructions`. The prompt's seconds are the
    model's: an interpolated job's ``fps`` timeline renders on the model's ``model_fps`` clock, so its
    times are scaled by ``fps / model_fps`` (the voice the model hears is that much faster)."""
    scale = fps / (model_fps or fps)

    def clock(seconds: float) -> str:
        return f"{max(0.0, seconds * scale):.2f}".rstrip("0").rstrip(".")

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
    """Runs ffmpeg (an assembly encode takes tens of seconds); a cancelled job kills it."""
    process = subprocess.Popen(command)
    try:
        while True:
            try:
                returncode = process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                cancel.check()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


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
    output_fps: int | None = None,
    upscale: bool = False,
    job_id: str = "",
) -> dict[str, Any] | None:
    """Join the chunks - ``(render, frames kept)``, dropping each shared boundary
    frame and every chunk's grid surplus - trim to the audio's length, scale,
    and mux the caller's audio.

    ``width``/``height`` deliver an exact frame size (cover scale + centre crop,
    never a squeeze); ``bounding_box`` instead fits the frame inside it.

    ``output_fps`` above ``fps`` (an interpolated job): the joined ``fps`` frames
    stream through RIFE up to it, and ``upscale`` sends them through SPAN 2x
    (:func:`h3.h3_postprocess.stream_video`) before the scale, then lanczos, and
    the one encode; returns that stream's metrics.
    """
    inputs: list[str] = []
    parts: list[str] = []
    for index, (path, frames) in enumerate(chunks):
        inputs += ["-i", str(path)]
        parts.append(f"[{index}:v]select=between(n\\,{1 if index else 0}\\,{int(frames) - 1}),setpts=N/({fps}*TB)[v{index}]")
    joined = "".join(f"[v{index}]" for index in range(len(chunks)))
    flags = ":flags=lanczos" if upscale else ""
    if bounding_box is not None:
        scale = (
            f"scale={bounding_box[0]}:{bounding_box[1]}:force_original_aspect_ratio=decrease"
            f":force_divisible_by=2{flags},setsar=1"
        )
    elif width and height:
        scale = f"scale={width}:{height}:force_original_aspect_ratio=increase{flags},crop={width}:{height},setsar=1"
    else:
        scale = "setsar=1"
    audio_index = len(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encode = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-b:a", "192k", "-t", f"{duration_seconds:.6f}", "-movflags", "+faststart", str(output_path)]
    if upscale or (output_fps and output_fps != fps):
        from .h3_postprocess import stream_video

        output_fps = int(output_fps or fps)
        width_in, height_in = image_size(chunks[0][0])
        join = ";".join(parts) + f";{joined}concat=n={len(chunks)}:v=1:a=0[out]"
        return stream_video(
            [*inputs, "-filter_complex", join, "-map", "[out]", "-frames:v", str(int(total_frames))],
            ["-i", str(audio_path), "-map", "0:v", "-map", "1:a:0", "-vf", scale, *encode],
            width=width_in, height=height_in, source_fps=int(fps), output_fps=output_fps,
            frames=max(1, round(duration_seconds * output_fps)), upscale=upscale, job_id=job_id,
        )
    graph = ";".join(parts) + f";{joined}concat=n={len(chunks)}:v=1:a=0,{scale}[out]"
    _run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs, "-i", str(audio_path),
            "-filter_complex", graph,
            "-map", "[out]", "-map", f"{audio_index}:a:0",
            "-frames:v", str(int(total_frames)), "-r", str(fps), *encode,
        ]
    )
    return None


# ---------------------------------------------------------------------------
# job I/O
# ---------------------------------------------------------------------------


def _download_url(job_id: str, url: str, path: Path, *, input_name: str, kind: str = "lipsync") -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{input_name} must use http or https")
    path.parent.mkdir(parents=True, exist_ok=True)
    log_event("job_input_download_started", job_id=job_id, kind=kind, input=input_name, url_host=parsed.netloc)
    written = 0
    try:
        with urlopen(Request(url, headers={"User-Agent": "lzai-media/1.0"}), timeout=120) as response:
            with path.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_INPUT_BYTES:
                        raise ValueError(f"{input_name} is larger than {MAX_INPUT_BYTES // 2**20} MB")
                    file.write(chunk)
    except (URLError, TimeoutError) as exc:
        # A dead link or an unreachable host is the caller's input, not a worker failure.
        raise ValueError(f"{input_name} could not be fetched: {exc}") from exc
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
    """Fetch a ``<name>_url`` / ``<name>_path`` job input."""
    job_id = str(job["id"])
    url = payload.get(f"{name}_url")
    if url:
        suffix = Path(urlparse(str(url)).path).suffix or fallback_suffix
        return _download_url(job_id, str(url), scratch_dir / "inputs" / f"{name}_url{suffix}", input_name=f"{name}_url", kind=kind)
    if payload.get(f"{name}_path"):
        from .compat import project_path

        path = project_path(str(payload[f"{name}_path"]))
        if not path.exists():
            raise FileNotFoundError(f"{name}_path does not exist: {path}")
        return path
    raise ValueError(f"{kind} job payload requires {name}_url or {name}_path")


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


def _new_plan(payload: dict[str, Any], *, voice_path: Path, audio_path: Path, input_image: Path, pins: PinSettings,
              pin_images: int, fps: float, seed: int, job_id: str, finish: Any) -> dict[str, Any]:
    """The job's plan: chunks, pins, prompts and per-chunk validators. Its frames run at ``fps``: the
    model's 24, or an interpolated job's ``interpolation_fps``, whose every chunk still renders on the
    model's own clock (:mod:`h3.h3_interpolation`)."""
    from .minimax_h3_runtime import H3_FPS, short_side_canvas
    from .span_upscale import requested_short_side

    # Timing: the caller's timed transcript, else the CTC words.
    began = time.perf_counter()
    caller_prompt = str(payload.get("prompt") or "").strip()
    caller_items = payload.get("transcript") or []
    custom_instructions = payload.get("custom_instructions") or []
    samples = load_audio_16k(voice_path)
    transcript = Transcript.from_items(list(caller_items)) if caller_items else transcribe(samples)
    transcript_source = "caller" if caller_items else "ctc"
    finish("speech_seconds", began)

    began = time.perf_counter()
    audio_duration = media_duration(audio_path)
    max_seconds = float(os.environ.get("LIPSYNC_MAX_SECONDS") or DEFAULT_MAX_SECONDS)
    if audio_duration > max_seconds:
        raise ValueError(f"audio is {audio_duration:.1f} s; lipsync takes at most {max_seconds:g} s")
    total_frames = max(1, int(round(audio_duration * fps)))
    source_width, source_height = image_size(input_image)
    # An output_short_side below the image's canvas renders on the target's own canvas (480p at ~854x480).
    canvas = short_side_canvas(input_image, requested_short_side(payload))
    chunk_cap = max_chunk_frames(canvas.width, canvas.height)
    # A pinned chunk is cut on its own times and pinned on its ends only, to its own image if it has
    # one; the rest is pinned_frames' (the module docstring).
    pinned = pinned_spans(custom_instructions, fps=fps, total_frames=total_frames,
                          longest=longest_chunk(fps=H3_FPS, chunk_cap=chunk_cap))
    fixed = frame_pins(pins, fps=fps, total_frames=total_frames, pinned=pinned)
    cut_frames, pin_frames = freestyle_plan(samples, total_frames=total_frames, fps=fps, chunk_cap=chunk_cap,
                                            pinned=pinned, continuous=pins.continuous, fixed=fixed, model_fps=H3_FPS)
    anchors = plan_anchors(pin_frames, pinned, continuous=pins.continuous, fixed=fixed,
                           images=pins.images or (INPUT_IMAGE,), order=pins.order, seed=seed)
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
    timed_lines = chunk_instructions(custom_instructions, chunks, fps=fps, total_frames=total_frames,
                                     model_fps=H3_FPS)
    prompts = ["\n".join([prompt, *extra]) for prompt, extra in zip(prompts, timed_lines)]
    instructions = [([pinned_lines[i]] if i in pinned_lines else []) + extra for i, extra in enumerate(timed_lines)]
    finish("plan_seconds", began)
    log_event(
        "lipsync_planned",
        job_id=job_id,
        audio_seconds=round(audio_duration, 3),
        output_frames=total_frames,
        pin_mode=pins.mode,
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
        "pin_mode": pins.mode,
        "anchors": [[anchor.frame, anchor.label, anchor.image] for anchor in anchors],
        "pinned_chunks": pinned_chunks,
        "chunk_texts": chunk_texts,
        "prompts": prompts,
        "instructions": instructions,
        # Validators a custom instruction turns off for its chunks.
        "checks_off": [sorted(off) for off in checks_off(custom_instructions, chunks, fps=fps, total_frames=total_frames)],
        "transcript": {"text": transcript.text, "language": transcript.language, "source": transcript_source},
    }


def _restore_boundary(chunk: LipsyncChunk, end_anchor: Anchor, *, images: dict[str, Path], anchor_at: dict[int, Anchor],
                      look: Path, job_id: str, prompt: str, seed: int, width: int, height: int, audio_slice: Path,
                      work_dir: Path, recipe: Any, sparse_attention: bool, addons: tuple[str, ...] = ()) -> dict[str, Any]:
    """Fill ``images[end_anchor.image]``: this chunk's last kept frame, previewed on its own pins (the
    recipe's preview schedule, that frame alone decoded), then restored toward ``look``, the image last
    pinned before it."""
    from .h3_restore import FramePreview, restore_frame
    from .minimax_h3_runtime import run_lipsync_chunk

    began = time.perf_counter()
    preview_path = work_dir / "boundary_preview.png"
    run_lipsync_chunk(
        job_id=job_id,
        chunk_index=chunk.index,
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        length=chunk.render_frames,
        first_frame_path=images[anchor_at[chunk.start_frame].image],
        last_frame_path=None,
        guides=[(position, images[anchor.image]) for position, anchor in chunk.guides if anchor != end_anchor],
        audio_path=audio_slice,
        output_path=work_dir / "unused.mp4",
        recipe=recipe,
        sparse_attention=sparse_attention,
        preview=FramePreview(frame=chunk.frames - 1, output_path=preview_path, sigmas=recipe.preview_sigmas),
        addons=addons,
    )
    preview_seconds = time.perf_counter() - began
    restored = work_dir.parent / "anchors" / f"{end_anchor.image.replace(':', '_')}.png"
    restore_frame(job_id=job_id, frame=preview_path, anchor=look, output_path=restored,
                  width=width, height=height, seed=seed)
    images[end_anchor.image] = restored
    metrics = {"frame": end_anchor.frame, "preview_seconds": round(preview_seconds, 3),
               "seconds": round(time.perf_counter() - began, 3)}
    log_event("lipsync_boundary_restored", job_id=job_id, chunk=chunk.index, **metrics)
    return metrics


STATE_VERSION = 1
STAGES = ("plan", "chunk", "assemble")


@dataclass
class _Job:
    """One lipsync job's settings, images, voice and plan: everything a chunk renders from. The whole job
    builds it once (:func:`_prepare`); a staged job's plan stage saves it (:func:`_save_state`) and every
    chunk / assemble stage loads it back (:func:`_load_state`)."""

    job_id: str
    scratch_dir: Path
    payload: dict[str, Any]
    auto: dict[str, Any] | None
    voice_path: Path
    images: dict[str, Path]
    plan: dict[str, Any]
    stage_timings: dict[str, float]
    started: float
    audio_path: Path | None = None
    stills_item: dict[str, Any] | None = None
    still_files: dict[str, str] | None = None

    def __post_init__(self) -> None:
        from .h3_interpolation import requested_fps
        from .minimax_h3_runtime import DEFAULT_LIPSYNC_PRESET, H3_FPS, LIPSYNC_PRESETS, VIDEO_ADDON_LORAS, h3_preset

        payload = self.payload
        self.work_dir = self.scratch_dir / "lipsync"
        self.seed = int(payload["seed"]) if payload.get("seed") is not None else DEFAULT_SEED
        self.preset = str(payload.get("preset") or DEFAULT_LIPSYNC_PRESET)
        self.recipe = h3_preset(self.preset, LIPSYNC_PRESETS)
        self.sparse_attention = payload.get("sparse_attention") is not False
        self.addons = tuple(dict.fromkeys(str(name) for name in payload.get("addons") or ()))
        if unknown := [name for name in self.addons if name not in VIDEO_ADDON_LORAS]:
            raise ValueError(f"unknown addons {unknown}; available: {sorted(VIDEO_ADDON_LORAS)}")
        self.detect_flicker = payload.get("detect_flicker") is not False
        self.detect_desync = payload.get("detect_desync") is not False
        self.detect_glitches = payload.get("detect_glitches") is not False
        self.checking = self.detect_flicker or self.detect_desync or self.detect_glitches
        self.check_retries = _env_int("LIPSYNC_CHECK_RETRIES", DEFAULT_CHECK_RETRIES)
        # use_interpolation: the timeline runs at interpolation_fps, every chunk renders its frames on the model's
        # 24 fps clock from its voice sped up to match, and RIFE brings the joined video up to 24 fps.
        self.interpolation_fps = requested_fps(payload)
        self.fps = float(self.interpolation_fps or H3_FPS)
        self.anchors = [Anchor(frame=int(frame), label=str(label), image=str(image))
                        for frame, label, image in self.plan["anchors"]]
        self.anchor_at = {anchor.frame: anchor for anchor in self.anchors}
        self.chunks = plan_chunks(int(self.plan["total_frames"]), max_frames=int(self.plan["chunk_cap"]),
                                  anchors=self.anchors, cut_frames=self.plan["cut_frames"])

    def finish(self, name: str, began: float) -> None:
        self.stage_timings[name] = round(time.perf_counter() - began, 3)

    def opens_on_boundary(self, chunk: LipsyncChunk) -> bool:
        """The chunk opens on the previous chunk's restored last frame (its ``boundary`` file)."""
        return self.anchor_at[chunk.start_frame].image.startswith(RESTORED_IMAGE)


def _prepare(job: dict[str, Any], *, staged: bool = False) -> _Job:
    """Inputs fetched, auto applied, pin images and rest stills ready, the voice cleaned, the job planned."""
    job_started = time.perf_counter()
    job_id = str(job["id"])
    stage_timings: dict[str, float] = {}

    def finish(name: str, began: float) -> None:
        stage_timings[name] = round(time.perf_counter() - began, 3)

    scratch_dir = job_dir(job_id)
    work_dir = scratch_dir / "lipsync"
    began = time.perf_counter()
    progress.report("preparing")
    image_path = download_job_input(job, job["payload"], "image", scratch_dir, ".png")
    audio_path = download_job_input(job, job["payload"], "audio", scratch_dir, ".wav")
    # auto (the default): the avatar render's own setup, and its pins for this audio's length.
    payload, auto = auto_payload(job["payload"], seconds=media_duration(audio_path))
    payload.pop("stage", None)
    log_event("lipsync_mode", job_id=job_id, auto=auto is not None, **(auto or {}))
    seed = int(payload["seed"]) if payload.get("seed") is not None else DEFAULT_SEED
    from .h3_interpolation import requested_fps
    from .minimax_h3_runtime import H3_FPS

    fps = float(requested_fps(payload) or H3_FPS)
    custom_instructions = payload.get("custom_instructions") or []
    pins = PinSettings.from_payload(payload.get("pinned_frames"))
    if staged and pins.mode == "generate":
        raise ValueError("a staged lipsync takes no pinned_frames generate: render the stills as imagegen "
                         "rest_frame tasks and pin them (multi_image, or auto stills)")
    # Every pinned image - the pinned chunks' own, pinned_frames' - each URL fetched once.
    pin_urls = {url: "pin_image_url" for item in custom_instructions
                if item.get("pin_chunk") and (url := str(item.get("pin_image_url") or ""))}
    pin_urls.update({url: "pinned_frames image" for url in pins.urls if url not in pin_urls})
    pin_downloads = {
        url: _download_url(job_id, url, scratch_dir / "inputs" / f"pin_image_{index}{Path(urlparse(url).path).suffix or '.png'}",
                           input_name=name)
        for index, (url, name) in enumerate(pin_urls.items())
    }
    finish("input_download_seconds", began)

    # The job's image, the only image the render sees.
    input_image = prepare_input_image(image_path, work_dir / "anchors" / "input.png")
    source_width, source_height = image_size(input_image)
    images = {INPUT_IMAGE: input_image, **{
        url: prepare_pin_image(path, work_dir / "anchors" / f"pin{index}.png", size=(source_width, source_height))
        for index, (url, path) in enumerate(pin_downloads.items())
    }}
    # auto: a photo avatar renders with the realism addon (rest_stills.clip_addons: CLIP stylized score below
    # MAX_STYLIZED), unless the caller sent anything custom: addons (even []) or
    # custom_instructions (prompts or pins). Manual jobs only get the addons they send.
    if auto is not None:
        if payload.get("addons") is None and not custom_instructions:
            from . import rest_stills

            stylized = rest_stills.stylized_score(input_image)
            payload["addons"] = list(rest_stills.clip_addons(stylized))
            auto["stylized"] = round(stylized, 3)
        auto["addons"] = list(payload.get("addons") or [])
        log_event("lipsync_auto_addons", job_id=job_id, addons=auto["addons"], stylized=auto.get("stylized"))
    # generate: rest stills of the person, rendered from the job's image and pinned like multi_image's
    # images; a still whose every take lost the face is left out, and with none left the job's image is pinned.
    # A non-realistic character (cartoon, anime, 3D animation, painting) or a faceless image is not checked.
    generated: list[dict[str, Any]] = []
    still_files: dict[str, str] = {}
    stills_item: dict[str, Any] = {}
    if pins.mode == "generate":
        from dataclasses import replace

        from . import rest_stills

        began = time.perf_counter()
        reference, stylized = rest_stills.reference_face(input_image)
        delivered = rest_stills.delivered_size(source_width, source_height)
        kept_keys = []
        for key in pins.images:
            index = int(key[len(GENERATED_IMAGE):])
            progress.report("preparing", index, len(pins.images))
            still = rest_stills.render_still(job_id=job_id, image=input_image, index=index, seed=seed,
                                             retries=pins.max_retries, out_dir=work_dir / "stills",
                                             reference=reference)
            generated.append(still.report())
            path = rest_stills.save_fitted(still.kept.path, scratch_dir / "outputs" / f"still_{index}.png", delivered)
            images[key] = prepare_pin_image(path, work_dir / "anchors" / f"generated_{index}.png",
                                            size=(source_width, source_height))
            still_files[f"still{index}"] = str(path)
            kept_keys.append(key)
        pins = replace(pins, images=tuple(kept_keys))
        finish("generate_stills_seconds", began)
        stills_item = {"generated_stills": generated, "stills_face_checked": reference is not None,
                       "stylized": round(stylized, 3), "stills_addons": []}
        log_event("lipsync_stills", job_id=job_id, kept=len(kept_keys), **stills_item)
    # The voice alone - no music, no noise - is what the rest of the job hears.
    began = time.perf_counter()
    voice_path = clean_voice(audio_path, work_dir / "voice.wav", job_id=job_id)
    finish("voice_cleanup_seconds", began)
    plan = _new_plan(payload, voice_path=voice_path, audio_path=audio_path, input_image=input_image, pins=pins,
                     pin_images=len(pin_downloads), fps=fps, seed=seed, job_id=job_id, finish=finish)
    return _Job(job_id=job_id, scratch_dir=scratch_dir, payload=payload, auto=auto, voice_path=voice_path,
                images=images, plan=plan, stage_timings=stage_timings, started=job_started, audio_path=audio_path,
                stills_item=stills_item, still_files=still_files)


def _render_chunk(job: _Job, chunk: LipsyncChunk) -> tuple[Path, dict[str, Any]]:
    """One chunk rendered (its continuous end restored first, validators and reseeded retries): the kept
    take and its report. Fills ``job.images`` with this chunk's restored end frame, if it ends on one."""
    from .h3_interpolation import speed_up
    from .minimax_h3_runtime import H3_FPS, run_lipsync_chunk

    began = time.perf_counter()
    plan, images, anchor_at = job.plan, job.images, job.anchor_at
    canvas_width, canvas_height = plan["canvas"]
    prompts = plan["prompts"]
    # The chunks are what a long clip is made of, and the measure of how far along it is; inside
    # one, its render's sampler steps move it too (`progress.chunk_steps`), so a one-chunk clip
    # is not stuck at 0% for minutes.
    progress.report("rendering", chunk.index, len(job.chunks))
    chunk_dir = job.work_dir / f"chunk{chunk.index:03d}"
    audio_slice = extract_audio_slice(
        job.voice_path,
        start_seconds=chunk.start_frame / job.fps,
        duration_seconds=chunk.render_frames / job.fps,
        output_path=chunk_dir / ("voice.wav" if job.interpolation_fps else "audio.wav"),
    )
    if job.interpolation_fps:
        # The chunk's frames render on the model's 24 fps clock: it hears its voice that much faster.
        audio_slice = speed_up(audio_slice, chunk_dir / "audio.wav", fps=job.interpolation_fps,
                               duration_seconds=chunk.render_frames / H3_FPS)
    # A continuous boundary at this chunk's end: its own last frame, previewed and restored toward
    # the image last pinned before it, pinned there; the next chunk opens on it.
    end_anchor = anchor_at.get(chunk.last_frame)
    restore_metrics: dict[str, Any] | None = None
    if end_anchor is not None and end_anchor.image.startswith(RESTORED_IMAGE) and end_anchor.image not in images:
        look = max((anchor for anchor in job.anchors if anchor.frame < end_anchor.frame
                    and not anchor.image.startswith(RESTORED_IMAGE)), key=lambda anchor: anchor.frame)
        restore_metrics = _restore_boundary(
            chunk, end_anchor, images=images, anchor_at=anchor_at, look=images[look.image], job_id=job.job_id,
            prompt=prompts[chunk.index],
            seed=job.seed + chunk.index, width=canvas_width, height=canvas_height, audio_slice=audio_slice,
            work_dir=chunk_dir, recipe=job.recipe, sparse_attention=job.sparse_attention, addons=job.addons)
        job.stage_timings[f"chunk_{chunk.index}_boundary_seconds"] = restore_metrics["seconds"]
    # Every chunk opens on a pin: the input image, a pinned image, or a continuous
    # boundary's restored frame (it ends on one too, unless it is the last and free).
    first_image = images[anchor_at[chunk.start_frame].image]
    guides = [(position, images[anchor.image]) for position, anchor in chunk.guides]
    guide_labels = [f"{position}:{anchor.label}" for position, anchor in chunk.guides]
    # With validators on, a chunk that fails them is rendered again on a new
    # seed; the pins make any take fit between its neighbours. A custom
    # instruction may turn some off for its chunks.
    off = set(plan["checks_off"][chunk.index])
    flicker = job.detect_flicker and "detect_flicker" not in off
    desync = job.detect_desync and "detect_desync" not in off
    glitches = job.detect_glitches and "detect_glitches" not in off
    chunk_checking = flicker or desync or glitches
    attempts: list[tuple[Path, dict[str, Any], dict[str, Any] | None]] = []
    for attempt in range(1 + (job.check_retries if chunk_checking else 0)):
        cancel.check()
        output = chunk_dir / ("render.mp4" if attempt == 0 else f"render_retry{attempt}.mp4")
        with progress.chunk_steps():
            metrics = run_lipsync_chunk(
                job_id=job.job_id,
                chunk_index=chunk.index,
                prompt=prompts[chunk.index],
                seed=job.seed + chunk.index + attempt * RESEED_STRIDE,
                width=canvas_width,
                height=canvas_height,
                length=chunk.render_frames,
                first_frame_path=first_image,
                last_frame_path=None,
                guides=guides,
                audio_path=audio_slice,
                output_path=output,
                recipe=job.recipe,
                sparse_attention=job.sparse_attention,
                addons=job.addons,
            )
        # Glitches are judged against the image the chunk opens on; a chunk moving into another
        # pinned image drifts from it on purpose, so its flicker check judges sudden steps only.
        verdict = _check(output, audio_slice, chunk, flicker=flicker, desync=desync,
                         glitches=glitches, image=first_image, job_id=job.job_id,
                         judge_drift={first_image, *(path for _, path in guides)} == {first_image},
                         ) if chunk_checking else None
        attempts.append((output, metrics, verdict))
        if verdict is None or not verdict["flagged"]:
            break
    kept = min(range(len(attempts)), key=lambda i: attempts[i][2]["badness"] if attempts[i][2] else 0.0)
    output, metrics, verdict = attempts[kept]
    job.finish(f"chunk_{chunk.index}_seconds", began)
    return output, {
        **metrics,
        "start_frame": chunk.start_frame,
        "kept_frames": chunk.frames,
        "text": plan["chunk_texts"][chunk.index],
        **({"pinned": True} if chunk.index in set(plan["pinned_chunks"]) else {}),
        **({"instructions": plan["instructions"][chunk.index]} if plan["instructions"][chunk.index] else {}),
        "anchors": guide_labels,
        **({"checks": [attempt_verdict for _, _, attempt_verdict in attempts if attempt_verdict is not None],
            "kept_attempt": kept, "flagged": bool(verdict and verdict["flagged"])} if job.checking else {}),
        **({"checks_off": sorted(off)} if job.checking and off else {}),
        **({"boundary_restore": restore_metrics} if restore_metrics else {}),
    }


def _assemble(job: _Job, chunk_paths: list[tuple[Path, int]], chunk_metrics: list[dict[str, Any]] | None,
              *, audio_path: Path) -> dict[str, Any]:
    """The rendered chunks joined, sized and muxed with the job's audio: the job's result. ``chunk_metrics``
    None (a staged job whose caller sent no chunk reports) leaves the per-chunk summary out."""
    from .minimax_h3_runtime import H3_FPS
    from .span_upscale import (SPAN_MODEL_NAME, requested_scale, requested_short_side, short_side_scale,
                               short_side_size, target_size)

    payload, plan = job.payload, job.plan
    audio_duration = float(plan["audio_duration"])
    total_frames = int(plan["total_frames"])
    source_width, source_height = plan["source_size"]
    canvas_width, canvas_height = plan["canvas"]
    # upscale: the frames go through SPAN 2x, then lanczos to the delivered size times the factor.
    upscale = requested_scale(payload)
    # Every chunk is rendered; joining them, interpolating and muxing is the rest.
    progress.report("finishing", len(job.chunks), len(job.chunks))
    output_dimensions = payload.get("output_dimensions")
    bounding_box = (
        (int(output_dimensions["width"]), int(output_dimensions["height"]))
        if isinstance(output_dimensions, dict)
        else None
    )
    delivered_width = source_width - source_width % 2
    delivered_height = source_height - source_height % 2
    short_side = requested_short_side(payload)
    if short_side is not None:
        # Exactly that short side in the image's aspect; SPAN first when it is above the render (H3's canvas).
        delivered_width, delivered_height = short_side_size((source_width, source_height), short_side,
                                                            bounding_box=bounding_box)
        upscale = short_side_scale((delivered_width, delivered_height), (canvas_width, canvas_height))
        bounding_box = None
    elif upscale:
        # The image's size times the factor, then fitted inside output_dimensions (and 4K).
        delivered_width, delivered_height = target_size(delivered_width, delivered_height, upscale,
                                                        bounding_box=bounding_box)
        bounding_box = None
    began = time.perf_counter()
    output_path = job.scratch_dir / "outputs" / "output.mp4"
    streamed = assemble_output(
        chunk_paths,
        audio_path=audio_path,
        total_frames=total_frames,
        fps=job.fps,
        duration_seconds=audio_duration,
        output_path=output_path,
        width=None if bounding_box else delivered_width,
        height=None if bounding_box else delivered_height,
        bounding_box=bounding_box,
        output_fps=int(H3_FPS) if job.interpolation_fps else None,
        upscale=bool(upscale),
        job_id=job.job_id,
    )
    job.finish("assemble_seconds", began)
    if bounding_box is not None:
        delivered_width, delivered_height = image_size(output_path)

    total_seconds = time.perf_counter() - job.started
    reported = chunk_metrics is not None
    render_seconds = sum(float(m["pipeline_seconds"]) for m in chunk_metrics) if reported else None
    transcript = plan["transcript"]
    recipe = job.recipe
    item = {
        "seed": job.seed,
        "fps": float(H3_FPS),
        "width": delivered_width,
        "height": delivered_height,
        "frames": streamed["output_frames"] if streamed else total_frames,
        "duration_seconds": round(audio_duration, 3),
        "source": "freestyle",
        "auto": job.auto is not None,
        **({"auto_choices": job.auto} if job.auto is not None else {}),
        "pin_mode": plan["pin_mode"],
        # generate: each rest still (its seed, variation, identity, mouth and takes); the kept ones are
        # the result's still<k> files.
        **(job.stills_item or {}),
        "anchors": len(job.anchors),
        "pinned_chunks": len(plan["pinned_chunks"]),
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "chunks": len(job.chunks),
        "preset": job.preset,
        "steps": recipe.steps,
        "sampler": recipe.sampler_label,
        "sigmas": recipe.sigmas,
        "lora": recipe.lora_name,
        "addons": list(job.addons),
        "sparse_attention": job.sparse_attention and (not reported or all(m["sparse_attention"] for m in chunk_metrics)),
        "checks": {
            "detect_flicker": job.detect_flicker,
            "detect_desync": job.detect_desync,
            "detect_glitches": job.detect_glitches,
            "rerendered_chunks": sum(len(m.get("checks") or ()) > 1 for m in chunk_metrics) if reported else None,
            "flagged_chunks": sum(bool(m.get("flagged")) for m in chunk_metrics) if reported else None,
        } if job.checking else None,
        "prompt_source": "payload" if str(payload.get("prompt") or "").strip() else "template",
        "transcript_source": transcript["source"],
        "language": transcript["language"],
        "transcript": transcript["text"],
    }
    if bounding_box is not None:
        item["output_dimensions"] = {"width": bounding_box[0], "height": bounding_box[1]}
    span_seconds = float(((streamed or {}).get("span") or {}).get("seconds", 0.0))
    if streamed and job.interpolation_fps:
        # Rendered at interpolation_fps; RIFE made the rest of the 24 fps frames.
        rife_seconds = round(streamed["seconds"] - span_seconds, 3)
        item["interpolation"] = {"fps": job.interpolation_fps, "rendered_frames": total_frames,
                                 "rife_frames": streamed["rife_frames"], "seconds": rife_seconds}
        job.stage_timings["interpolation_seconds"] = rife_seconds
    if streamed and upscale:
        # SPAN 2x on the joined render, then lanczos to width x height.
        item["upscale"] = {"scale": upscale, "method": "span2x_lanczos", "model": SPAN_MODEL_NAME,
                           "span_seconds": span_seconds}
        job.stage_timings["upscale_span_seconds"] = span_seconds
    log_event("lipsync_complete", job_id=job.job_id, wall_seconds=round(total_seconds, 3), **{
        key: item[key] for key in ("frames", "chunks", "anchors", "width", "height")
    })
    return {
        "path": str(output_path),
        "item": item,
        "metrics": {
            "engine": "comfyui_lipsync",
            "wall_seconds": round(total_seconds, 3),
            "render_seconds": round(render_seconds, 3) if reported else None,
            "wall_rtf": round(total_seconds / audio_duration, 3) if audio_duration else None,
            "render_rtf": round(render_seconds / audio_duration, 3) if reported and audio_duration else None,
            "stage_timings": job.stage_timings,
            **({"chunks": chunk_metrics} if reported else {}),
        },
        "seed": job.seed,
        # generate: the kept rest stills, by result file name.
        **({"files": job.still_files} if job.still_files else {}),
    }


# ---------------------------------------------------------------------------
# Staged jobs: plan -> chunk 0..n-1 -> assemble, one Taskflow task each
# ---------------------------------------------------------------------------


def _save_state(job: _Job, path: Path) -> Path:
    """The plan stage's state: ``state.json`` (payload after auto, plan, image files, timings), the cleaned
    voice and the pin images, one tar every later stage starts from."""
    keys = sorted(job.images)
    state = {
        "version": STATE_VERSION,
        "payload": job.payload,
        "auto": job.auto,
        "plan": job.plan,
        "images": {key: f"images/{index}{job.images[key].suffix}" for index, key in enumerate(keys)},
        "stage_timings": job.stage_timings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        data = json.dumps(state).encode()
        info = tarfile.TarInfo("state.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        tar.add(job.voice_path, arcname="voice.wav")
        for key in keys:
            tar.add(job.images[key], arcname=state["images"][key])
    return path


def _load_state(job: dict[str, Any], *, began: float) -> _Job:
    """A staged job's state, from the plan stage's tar (``state_url`` / ``state_path``)."""
    job_id = str(job["id"])
    scratch_dir = job_dir(job_id)
    tar_path = download_job_input(job, job["payload"], "state", scratch_dir, ".tar")
    state_dir = scratch_dir / "lipsync" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(state_dir, filter="data")
        state = json.loads((state_dir / "state.json").read_text())
    except (tarfile.TarError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"state is not a lipsync plan's state: {exc}") from exc
    if state.get("version") != STATE_VERSION:
        raise ValueError(f"state version {state.get('version')} is not {STATE_VERSION}; plan the job again")
    return _Job(job_id=job_id, scratch_dir=scratch_dir, payload=state["payload"], auto=state["auto"],
                voice_path=state_dir / "voice.wav",
                images={key: state_dir / name for key, name in state["images"].items()},
                plan=state["plan"], stage_timings=dict(state["stage_timings"]), started=began)


def _snake_keys(value: Any) -> Any:
    """A chunk report as the API returned it (camelCase keys) back in the runtime's snake_case."""
    if isinstance(value, dict):
        return {re.sub(r"(?<!^)(?=[A-Z])", "_", str(key)).lower(): _snake_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snake_keys(item) for item in value]
    return value


def _stage_plan(job: dict[str, Any]) -> dict[str, Any]:
    """``stage: "plan"``: everything before the first chunk. The result is the state tar (later stages'
    ``state``) plus the job's audio as the ``audio`` file (assemble muxes it), and the chunk list."""
    ctx = _prepare(job, staged=True)
    outputs = ctx.scratch_dir / "outputs"
    state_path = _save_state(ctx, outputs / "state.tar")
    audio_copy = outputs / f"audio{ctx.audio_path.suffix or '.wav'}"
    shutil.copyfile(ctx.audio_path, audio_copy)
    plan = ctx.plan
    item = {
        "stage": "plan",
        "seed": ctx.seed,
        "chunks": len(ctx.chunks),
        # Chunk k > 0 that opens on a boundary needs chunk k-1's boundary file: the chunks run in order.
        "chunk_plan": [{"index": chunk.index, "start_frame": chunk.start_frame, "frames": chunk.frames,
                        "seconds": round(chunk.frames / ctx.fps, 3), "needs_boundary": ctx.opens_on_boundary(chunk)}
                       for chunk in ctx.chunks],
        "frames": int(plan["total_frames"]),
        "duration_seconds": round(float(plan["audio_duration"]), 3),
        "canvas_width": plan["canvas"][0],
        "canvas_height": plan["canvas"][1],
        "pin_mode": plan["pin_mode"],
        "auto": ctx.auto is not None,
        **({"auto_choices": ctx.auto} if ctx.auto is not None else {}),
        "transcript_source": plan["transcript"]["source"],
        "language": plan["transcript"]["language"],
        "transcript": plan["transcript"]["text"],
    }
    seconds = round(time.perf_counter() - ctx.started, 3)
    log_event("lipsync_stage_planned", job_id=ctx.job_id, chunks=len(ctx.chunks), wall_seconds=seconds)
    return {"path": str(state_path), "item": item, "seed": ctx.seed, "files": {"audio": str(audio_copy)},
            "metrics": {"engine": "comfyui_lipsync", "wall_seconds": seconds, "stage_timings": ctx.stage_timings}}


def _stage_chunk(job: dict[str, Any], began: float) -> dict[str, Any]:
    """``stage: "chunk"``: chunk ``chunk_index`` of the planned job, rendered exactly as the whole job
    renders it (same seeds, validators, retries). Opening on a boundary, it needs the previous chunk's
    ``boundary`` file (``boundary_url`` / ``boundary_path``). Its result is the kept take (more frames
    than it keeps: assemble trims) and, when it ends on a restored boundary, that frame as ``boundary``."""
    from .minimax_h3_runtime import cleanup_h3_job_files

    payload = job["payload"]
    ctx = _load_state(job, began=began)
    ctx.stage_timings = {}
    index = int(payload.get("chunk_index", -1))
    if not 0 <= index < len(ctx.chunks):
        raise ValueError(f"chunk_index {index} is not one of the plan's {len(ctx.chunks)} chunks")
    chunk = ctx.chunks[index]
    start_key = ctx.anchor_at[chunk.start_frame].image
    if ctx.opens_on_boundary(chunk):
        if not (payload.get("boundary_url") or payload.get("boundary_path")):
            raise ValueError(f"chunk {index} opens on chunk {index - 1}'s boundary: send its boundary file")
        # The restored frame as the previous chunk made it (canvas size), as the whole job pins it.
        ctx.images[start_key] = download_job_input(job, payload, "boundary", ctx.scratch_dir, ".png")
    try:
        render, report = _render_chunk(ctx, chunk)
    finally:
        cleanup_h3_job_files(ctx.job_id)
    end_anchor = ctx.anchor_at.get(chunk.last_frame)
    files = ({"boundary": str(ctx.images[end_anchor.image])}
             if end_anchor is not None and end_anchor.image.startswith(RESTORED_IMAGE) else {})
    seconds = round(time.perf_counter() - began, 3)
    report["seconds"] = ctx.stage_timings[f"chunk_{index}_seconds"]
    item = {"stage": "chunk", "seed": ctx.seed, "chunk_index": index, "chunks": len(ctx.chunks),
            "start_frame": chunk.start_frame, "frames": chunk.frames, "render_frames": chunk.render_frames,
            "boundary": bool(files), "report": report}
    log_event("lipsync_stage_chunk", job_id=ctx.job_id, chunk=index, wall_seconds=seconds)
    return {"path": str(render), "item": item, "seed": ctx.seed, **({"files": files} if files else {}),
            "metrics": {"engine": "comfyui_lipsync", "wall_seconds": seconds, "stage_timings": ctx.stage_timings}}


def _stage_assemble(job: dict[str, Any], began: float) -> dict[str, Any]:
    """``stage: "assemble"``: the chunks' renders (``chunk_urls`` / ``chunk_paths``, in order) joined, sized
    and muxed with the job's audio (``audio_url`` / ``audio_path``: the plan's ``audio`` file); the
    whole job's result. ``chunk_reports`` (each chunk result's ``report``) fill the per-chunk summary."""
    payload = job["payload"]
    ctx = _load_state(job, began=began)
    audio_path = download_job_input(job, payload, "audio", ctx.scratch_dir, ".wav")
    urls, paths = payload.get("chunk_urls"), payload.get("chunk_paths")
    sources = list(urls or paths or [])
    if len(sources) != len(ctx.chunks):
        raise ValueError(f"the plan has {len(ctx.chunks)} chunks; got {len(sources)} chunk renders")
    renders = []
    for chunk, source in zip(ctx.chunks, sources):
        name = f"chunk{chunk.index:03d}"
        if urls:
            renders.append(_download_url(ctx.job_id, str(source), ctx.scratch_dir / "inputs" / f"{name}.mp4",
                                         input_name=f"chunk_urls[{chunk.index}]"))
        else:
            from .compat import project_path

            path = project_path(str(source))
            if not path.exists():
                raise FileNotFoundError(f"chunk_paths[{chunk.index}] does not exist: {path}")
            renders.append(path)
    reports = payload.get("chunk_reports")
    if reports is not None and len(reports) != len(ctx.chunks):
        raise ValueError(f"the plan has {len(ctx.chunks)} chunks; got {len(reports)} chunk reports")
    return _assemble(ctx, [(path, chunk.frames) for path, chunk in zip(renders, ctx.chunks)],
                     _snake_keys(reports) if reports is not None else None, audio_path=audio_path)


def run_lipsync_inference(job: dict[str, Any]) -> dict[str, Any]:
    """Run one ``lipsync`` job; returns the result.

    ``stage`` absent: the whole job in one go. A long job runs staged instead, one Taskflow task per
    stage so other jobs' chunks take turns on the GPU between them: ``plan``, then ``chunk`` 0..n-1 in
    order, then ``assemble`` (:func:`_stage_plan`, :func:`_stage_chunk`, :func:`_stage_assemble`). The
    staged result is the whole job's, chunk for chunk."""
    from .minimax_h3_runtime import cleanup_h3_job_files

    began = time.perf_counter()
    stage = job["payload"].get("stage")
    if stage is not None and stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    if stage == "plan":
        return _stage_plan(job)
    if stage == "chunk":
        return _stage_chunk(job, began)
    if stage == "assemble":
        return _stage_assemble(job, began)
    ctx = _prepare(job)
    chunk_paths: list[tuple[Path, int]] = []
    chunk_metrics: list[dict[str, Any]] = []
    try:
        for chunk in ctx.chunks:
            render, report = _render_chunk(ctx, chunk)
            chunk_paths.append((render, chunk.frames))
            chunk_metrics.append(report)
    finally:
        cleanup_h3_job_files(ctx.job_id)
    return _assemble(ctx, chunk_paths, chunk_metrics, audio_path=ctx.audio_path)
