"""Client for the media runtime process (src/media_runtime): generate-music, videogen, lipsync, imagegen.

YuE2, YingMusic-SVC and ComfyUI/H3 run from their own venv (Python 3.11, torch 2.14, cu130), so
the runtime cannot be a multiprocessing child like the speech and transcribe workers. The API
spawns it with two pipes it opens first — one line of JSON per message — and talks to it over
those: one request per task (``generate``, ``videogen``, ``lipsync``, ``imagegen``), its reply,
the state it reports once its models are loaded, and the progress of the job it is rendering.
Nothing listens on a port; the channel ends when either process does
(``src/media_runtime/channel.py`` is the other end).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..process_guard import exit_with_parent
from .downloads import MAX_REFERENCE_BYTES, DownloadError, fetch_to_file
from .job_progress import report as report_progress

_LOGGER = logging.getLogger(__name__)

REFERENCE_TIMEOUT_SECONDS = 60.0
#: The inherited pipe ends, as the child reads them (src/media_runtime/channel.py).
REQUEST_FD_VARIABLE = "LZAI_REQUEST_FD"
REPLY_FD_VARIABLE = "LZAI_REPLY_FD"


class MediaInputError(ValueError):
    """The request is invalid for the media runtime (HTTP 400)."""


class MediaRuntimeError(RuntimeError):
    """The media runtime failed to start, load, or render."""


class MediaCancelled(MediaRuntimeError):
    """The job was cancelled (its lease released) and the runtime stopped rendering it."""


#: How long a cancelled job may take to stop: the runtime checks between model ops and sampler steps.
CANCEL_GRACE_SECONDS = 60.0


class SamplingOverrides(BaseModel):
    """Token sampling for one YuE2 pass; every field is optional and falls back to the model default."""

    model_config = {"extra": "forbid"}

    temperature: float | None = Field(default=None, ge=0, le=5)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    repetition_penalty: float | None = Field(default=None, gt=0, le=3)
    penalty_window: int | None = Field(default=None, ge=1, le=100)
    min_tokens: int | None = Field(default=None, ge=0, le=9000)
    max_tokens: int | None = Field(default=None, ge=1, le=9000)


class MusicRequest(BaseModel):
    """The ``generate-music`` request.

    YuE2 writes the song from ``style`` and ``lyrics``; the length follows the lyrics sections (there is
    no duration, BPM, key or instrumental parameter: put tempo and mood in ``style``, or give ``abc``).
    Without ``reference_url`` the song is YuE2's render; with it, the lead vocal is re-voiced by YingMusic-SVC.
    """

    model_config = {"extra": "forbid"}

    style: str = Field(min_length=1, max_length=2000, description="Comma-separated genre, vocal, instruments, mood")
    lyrics: str = Field(min_length=1, max_length=8000, description="Plain text in [verse]/[chorus]... sections")
    cot: Literal["full", "melody", "off"] = Field(
        default="full", description="Planning pass: melody and chords, melody only, or none (covers: melody)"
    )
    cfg_scale: float | None = Field(default=None, ge=0, le=20, description="Text guidance; null = model default")
    seed: int | None = Field(default=None, ge=0, le=2**31 - 2)
    abc: str | None = Field(
        default=None, max_length=20000, description="Your own ABC melody score (cot melody or full); skips planning"
    )
    ode_steps: int | None = Field(default=None, ge=1, le=100, description="Audio-stage flow steps; model default 32")
    abc_sampling: SamplingOverrides | None = None
    semantic_sampling: SamplingOverrides | None = Field(
        default=None, description="max_tokens caps the song length at about 25 tokens per second"
    )
    reference_url: str | None = None

    @model_validator(mode="after")
    def _abc_needs_plan(self) -> "MusicRequest":
        if self.abc is not None and self.cot == "off":
            raise ValueError("abc needs cot 'melody' or 'full'")
        return self


class VideogenPinnedFrame(BaseModel):
    """A guide image pinned ``seconds`` into the clip (an anchor the render passes through)."""

    model_config = {"extra": "forbid"}

    image_url: str = Field(min_length=1)
    seconds: float = Field(ge=0, le=10)


H3VideoPreset = Literal["standard", "turbo-8", "turbo-6", "turbo-5", "turbo-4", "dmd-3"]
H3StillPreset = Literal["standard", "turbo-8", "turbo-6", "turbo-4"]  # h3_still.STILL_PRESETS


#: ``use_interpolation`` is refused (400) while this is off; the runtime code stays for a fixed method.
INTERPOLATION_ENABLED = False


class H3Interpolation(BaseModel):
    """Postprocessing of an H3 clip. ``use_interpolation`` (vigen 7280e2f, ``h3/h3_interpolation.py``): H3
    renders ``interpolation_fps`` frames a second and RIFE v4.26 makes the rest of the 24 fps output.
    Disabled for now (:data:`INTERPOLATION_ENABLED`): with a voice, motion the voice does not drive
    (blinks, head, hands) plays back ``24/fps`` times slow, and the RIFE frames' softer mouths pulse at
    8 Hz; a 19.9 s avatar clip ran blinks up to 2x long for a 1.5x shorter render.
    ``upscale`` (vigen 43c32e7, ``h3/span_upscale.py``): SPAN 2x, then lanczos to the delivered size times the
    factor (longest side at most 3840). ``output_short_side`` (480, 720, 1080; not with ``upscale``): the
    delivered frame in the image's aspect with exactly that short side; above the render's (H3's canvas, at most
    the picture's) it goes through SPAN 2x then lanczos, the result's ``upscale`` entry saying so;
    ``output_dimensions`` then bounds it."""

    use_interpolation: bool = Field(
        default=False,
        description="Render at interpolation_fps and RIFE-interpolate up to 24 fps (disabled for now: slow motion)",
    )
    interpolation_fps: int | None = Field(
        default=None, ge=12, le=23, description="Frames H3 renders a second under use_interpolation (default 16)"
    )

    upscale: bool | float = Field(
        default=False,
        description="SPAN 2x then lanczos to the delivered size times this factor: true = 1.5, or above 1 and at most 2",
    )

    output_short_side: int | None = Field(
        default=None, ge=64, le=2160,
        description="Delivered short side (480, 720, 1080), image aspect kept, exactly; SPAN-upscaled when above the render",
    )

    @field_validator("upscale")
    @classmethod
    def _upscale_factor(cls, value: bool | float) -> bool | float:
        if not isinstance(value, bool) and not 1.0 < value <= 2.0:
            raise ValueError("upscale must be true or a factor above 1 and at most 2")
        return value

    @model_validator(mode="after")
    def _fps_needs_flag(self):
        if self.use_interpolation and not INTERPOLATION_ENABLED:
            raise ValueError("use_interpolation is disabled for now: 16 fps renders play motion the voice does not drive slow")
        if self.output_short_side is not None and self.upscale:
            raise ValueError("output_short_side picks the upscale itself (SPAN when it is above the render); drop upscale")
        if self.interpolation_fps is not None and not self.use_interpolation:
            raise ValueError("interpolation_fps needs use_interpolation")
        return self


class VideogenRequest(H3Interpolation):
    """The ``videogen`` request: an H3 clip from a first frame or from the prompt alone, with H3's
    own audio track or a pinned one.

    ``preset`` (all on the Ref2VA base): ``standard`` (plain base, res_multistep 10 steps, slowest),
    ``turbo-8`` (the Ref2V 8-step Turbo LoRA), ``turbo-6``/``turbo-5`` (the ema850 4-step LoRA at 6 or 5
    steps), ``turbo-4`` (the Ref2V 4-step Turbo LoRA), ``dmd-3`` (vigen's 3-step DMD3 avatar student,
    fastest); omitted, the worker's default recipe (ema850 at 4 steps).

    ``width``/``height`` (both or neither) are the delivered frame size; omitted, the clip keeps
    the image's aspect on H3's 32-px lattice (about 1 MP) — a clip made from the prompt alone has
    no image to take its shape from, so it states them. ``upscale`` multiplies that size;
    ``output_dimensions`` (with ``upscale`` only) bounds the result, aspect kept, as on lipsync:
    ``width`` 1920, ``height`` 1080, ``upscale`` and ``output_dimensions`` 1920x1080 is a SPAN-detailed
    1080p clip, not a 2880x1620 one.

    Long clips (``h3/videogen.py`` docstring, "Long clips"): ``duration_seconds`` above 10 (up to 60) renders
    parts of at most ~9 s, each continuing the previous one's latent (its last 22 frames handed over) and
    refined toward the first part's colour, decodes them once and returns one mp4; later parts follow
    ``continuation_prompt`` (default ``prompt``). The same chain one part per request: ``latent_only`` returns
    the part's latent (safetensors) instead of an mp4, ``continue_latent_url`` (+ ``chain_reference_latent_url``,
    the first part's latent) continues one with ``duration_seconds`` of new frames, on its canvas, and
    ``videogen-chain-decode`` (:class:`VideogenChainDecodeRequest`) decodes the parts into the mp4. On Taskflow
    the worker fills the ``*_path`` fields from ``chain`` (lease artifacts), never a caller.
    """

    model_config = {"extra": "forbid"}

    prompt: str = Field(min_length=1, max_length=4000, description="What happens in the clip")
    image_url: str | None = Field(
        default=None, min_length=1,
        description="First frame; omitted, the clip is made from the prompt alone and states width/height",
    )
    last_frame_url: str | None = Field(default=None, description="Optional last frame")
    duration_seconds: float | None = Field(
        default=None, gt=0, le=60,
        description="Clip length at 24 fps; omitted: the pinned audio's length, else 3 s. Above 10 s (no pins, no "
        "last frame, not continuable): rendered in parts and decoded once; a continued part: its new seconds",
    )
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    generate_audio: bool = Field(
        default=True, description="Audio track on the output (the pinned audio when one is given)"
    )
    pinned_frames: list[VideogenPinnedFrame] = Field(
        default_factory=list, max_length=16, description="Guide images pinned at times in the clip"
    )
    pinned_audio_url: str | None = Field(
        default=None, min_length=1,
        description="Audio pinned in the latent: the video moves to it; it is the output's audio track",
    )
    preset: H3VideoPreset | None = None
    sparse_attention: bool = Field(
        default=True,
        description="Lipsync's attention (Comfy Kitchen INT8 + SLA, 85% of key blocks skipped, step 2 dense): "
        "~2x faster diffusion; false renders dense",
    )
    addons: list[Literal["motion", "dance"]] = Field(
        default_factory=list, max_length=2,
        description="Community LoRAs stacked on the preset at strength 1.0: motion (Better Human Motion H3, "
        "civitai 2734359), dance (civitai 3056222)",
    )
    continuable: bool = Field(
        default=False,
        description="End on a clean frame another clip can open on: the last frame is previewed, restored "
        "toward anchor_url (else image_url; a prompt-only first clip, toward itself) and pinned; the result "
        "returns it as its lastFrame file (JPEG)",
    )
    anchor_url: str | None = Field(
        default=None, min_length=1,
        description="The chain's look (its first clip's image): a continued clip passes it with continuable",
    )
    output_dimensions: OutputDimensions | None = Field(
        default=None, description="With upscale: the box the upscaled clip fits inside, aspect kept"
    )
    native: bool = Field(
        default=False,
        description="With output_short_side: render at that size itself, past H3's 1 MP canvas up to 1920x1088 "
        "(native 1080p), instead of ~1 MP + SPAN; about 3x the render time, and at 1080p at most 8 s (the 3090's "
        "VRAM; a longer duration is a 400)",
    )
    continuation_prompt: str | None = Field(
        default=None, min_length=1, max_length=4000,
        description="A long clip's later parts follow this prompt (default prompt)",
    )
    latent_only: bool = Field(
        default=False, description="Return the part's latent (safetensors) instead of an mp4: a chain part"
    )
    return_latent: bool = Field(default=False, description="Also return the clip's latent as the latent file")
    continue_latent_url: str | None = Field(
        default=None, min_length=1,
        description="Continue this part's latent: no image, the canvas is the latent's, duration_seconds new",
    )
    chain_reference_latent_url: str | None = Field(
        default=None, min_length=1, description="The chain's first part's latent: later parts are refined toward it"
    )
    continue_latent_path: str | None = Field(default=None, min_length=1, description="Worker-staged (chain)")
    chain_reference_latent_path: str | None = Field(default=None, min_length=1, description="Worker-staged (chain)")
    chain: VideogenChain | None = Field(
        default=None, description="Taskflow chain part: the worker reads its latents from the lease's artifacts"
    )

    @model_validator(mode="after")
    def _size_pair(self) -> "VideogenRequest":
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height go together")
        if self.native and self.output_short_side is None:
            raise ValueError("native renders at output_short_side; send it")
        if self.output_dimensions is not None and not (self.upscale or self.output_short_side):
            raise ValueError("output_dimensions goes with upscale or output_short_side; else width/height are the delivered size")
        continued = bool(
            self.continue_latent_url or self.continue_latent_path or (self.chain is not None and self.chain.index > 0)
        )
        if self.chain_reference_latent_url or self.chain_reference_latent_path:
            if not continued:
                raise ValueError("chain_reference_latent_url goes with continue_latent_url")
        if continued:
            opened = [name for name in ("image_url", "last_frame_url", "pinned_audio_url", "anchor_url")
                      if getattr(self, name)] + (["pinned_frames"] if self.pinned_frames else []) + (
                ["continuable"] if self.continuable else [])
            if opened:
                raise ValueError(f"a continued part opens on the previous part's latent; drop {opened}")
            if self.duration_seconds is not None and self.duration_seconds > 10:
                raise ValueError("a continued part adds at most ~9 s (7 s native 1080p); send a shorter duration_seconds")
            # The canvas is the previous latent's; sizing fields are the decode's.
            return self
        if self.duration_seconds is not None and self.duration_seconds > 10:
            parts = [name for name in ("last_frame_url", "pinned_audio_url", "chain") if getattr(self, name)] + [
                name for name in ("pinned_frames", "continuable", "latent_only", "return_latent", "use_interpolation")
                if getattr(self, name)]
            if parts:
                raise ValueError(f"a clip over 10 s renders in parts; drop {parts} or send at most 10 s")
        if self.image_url is not None:
            if self.output_short_side is not None and self.width is not None:
                raise ValueError("output_short_side sizes the clip from the image's aspect; send it or width/height")
        elif self.width is None:
            # Nothing to take a shape from: a prompt-only clip says what it is.
            raise ValueError("a clip made from the prompt alone needs width and height")
        if self.continuable and self.last_frame_url:
            raise ValueError("continuable clips end on their own restored frame; drop last_frame_url")
        if self.anchor_url and not self.continuable:
            raise ValueError("anchor_url needs continuable")
        if self.duration_seconds is not None:
            late = [pin.seconds for pin in self.pinned_frames if pin.seconds >= self.duration_seconds]
            if late:
                raise ValueError(f"pinned_frames at {late} s are past duration_seconds")
        return self


class VideogenChain(BaseModel):
    """A Taskflow chain part (``videogen`` ``request.chain``): the part uploads only its latent, to
    ``latent_artifact_id`` (the payload's ``artifactId``). A later part (``index`` > 0) continues
    ``previous_latent_artifact_id`` and is refined toward ``anchor_latent_artifact_id`` (part 0's latent);
    it sends no image and no width/height."""

    model_config = {"extra": "forbid"}

    index: int = Field(ge=0, le=64)
    latent_artifact_id: str = Field(min_length=1)
    previous_latent_artifact_id: str | None = Field(default=None, min_length=1)
    anchor_latent_artifact_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _later_parts(self) -> "VideogenChain":
        if self.index > 0 and self.previous_latent_artifact_id is None:
            raise ValueError("a chain part after the first needs previous_latent_artifact_id")
        if self.index == 0 and (self.previous_latent_artifact_id or self.anchor_latent_artifact_id):
            raise ValueError("chain part 0 opens the chain; it has no previous or anchor latent")
        return self


class LipsyncTranscriptItem(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class LipsyncInstruction(BaseModel):
    """A caller instruction at a ``timestamp`` or over ``start``-``end`` (seconds of the whole video)."""

    model_config = {"extra": "forbid"}

    instruction: str = Field(min_length=1, max_length=2000)
    timestamp: float | None = Field(default=None, ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    text_hint: str | None = Field(default=None, max_length=500)
    pin_chunk: bool = False
    pin_image_url: str | None = None
    detect_flicker: bool | None = None
    detect_desync: bool | None = None
    detect_glitches: bool | None = None

    @model_validator(mode="after")
    def _when(self) -> "LipsyncInstruction":
        point = self.timestamp is not None
        span = self.start is not None and self.end is not None
        if point == span or (self.start is None) != (self.end is None):
            raise ValueError("an instruction takes either timestamp or start and end")
        if span and self.end <= self.start:
            raise ValueError("instruction end must be after start")
        if self.pin_chunk and not span:
            raise ValueError("pin_chunk needs start and end")
        if self.pin_image_url and not self.pin_chunk:
            raise ValueError("pin_image_url needs pin_chunk")
        return self


class LipsyncPinnedFrame(BaseModel):
    """An image pinned ``seconds`` into the video (``timed_images``)."""

    model_config = {"extra": "forbid"}

    image_url: str = Field(min_length=1)
    seconds: float = Field(gt=0)


class LipsyncPinnedFrames(BaseModel):
    """What lipsync pins besides frame 0 (the job's image) and ``pin_chunk`` spans.

    ``none``: nothing; every chunk boundary is pinned to the previous chunk's own last frame
    restored toward the image last pinned, so the motion carries on. ``single_image``: one image
    (``image_url``, else the job's image) on every chunk boundary and every 2.5-5 s at the pauses.
    ``multi_image``: those pins, their images taken in turn from ``image_urls`` (``order``:
    ``round_robin`` 1, 2, 3, 1, ..., or ``random``, seeded by the job's seed, never the same image
    twice running). ``timed_images``: each of ``frames`` at its ``seconds`` and nothing else;
    boundaries as with ``none``, but a cut with a pin in reach lands on it. ``generate``: the job
    first renders ``count`` rest stills of the person from its image (imagegen ``rest_frame``
    stills, each retaken at most ``max_retries`` times on a lost face or open mouth, the best take
    kept either way) and pins them like ``multi_image``; the stills come back as the result's ``still<k>`` files (PNG), so
    the lease payload needs ``artifactIds`` for ``still0`` .. ``still<count-1>``.
    """

    model_config = {"extra": "forbid"}

    mode: Literal["none", "single_image", "multi_image", "timed_images", "generate"] = "none"
    image_url: str | None = Field(default=None, min_length=1)
    image_urls: list[str] = Field(default_factory=list, max_length=16)
    order: Literal["round_robin", "random"] | None = None
    frames: list[LipsyncPinnedFrame] = Field(default_factory=list, max_length=100)
    count: int | None = Field(default=None, ge=2, le=12)
    max_retries: int | None = Field(default=None, ge=0, le=5)

    @model_validator(mode="after")
    def _fields_of_mode(self) -> "LipsyncPinnedFrames":
        given = {"image_url": self.image_url, "image_urls": self.image_urls, "order": self.order,
                 "frames": self.frames, "count": self.count, "max_retries": self.max_retries}
        allowed = {"none": (), "single_image": ("image_url",), "multi_image": ("image_urls", "order"),
                   "timed_images": ("frames",), "generate": ("count", "order", "max_retries")}[self.mode]
        extra = [key for key, value in given.items() if value not in (None, "", []) and key not in allowed]
        if extra:
            raise ValueError(f"pinned_frames mode {self.mode} takes no {', '.join(extra)}")
        if self.mode == "multi_image" and len(self.image_urls) < 2:
            raise ValueError("multi_image takes at least two image_urls (one is single_image)")
        if self.mode == "timed_images" and not self.frames:
            raise ValueError("timed_images needs frames")
        return self


class OutputDimensions(BaseModel):
    model_config = {"extra": "forbid"}

    width: int = Field(ge=64, le=4096)
    height: int = Field(ge=64, le=4096)


class VideogenChainDecodeRequest(BaseModel):
    """The ``videogen-chain-decode`` request: the latents of a videogen chain (``latent_only`` parts, in
    order) decoded once into one mp4, sized like a videogen clip: ``width``/``height`` (the delivered
    size), ``output_short_side`` (+ ``native`` when the parts rendered natively, ``output_dimensions``) or
    ``upscale``; none of them = the parts' canvas. Taskflow sends ``latent_artifact_ids`` (the worker reads
    them from the lease and fills ``latent_paths``); /task/sync sends ``latent_urls``."""

    model_config = {"extra": "forbid"}

    latent_artifact_ids: list[str] | None = Field(default=None, min_length=1, max_length=64)
    latent_urls: list[str] | None = Field(default=None, min_length=1, max_length=64)
    latent_paths: list[str] | None = Field(default=None, min_length=1, max_length=64, description="Worker-staged")
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    upscale: bool | float = False
    output_short_side: int | None = Field(default=None, ge=64, le=2160)
    native: bool = False
    output_dimensions: OutputDimensions | None = None
    generate_audio: bool = True
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1, description="Reported back only")

    @model_validator(mode="after")
    def _one_source(self) -> "VideogenChainDecodeRequest":
        if sum(x is not None for x in (self.latent_urls, self.latent_paths)) != 1:
            raise ValueError("send latent_urls (or, on Taskflow, latent_artifact_ids)")
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height go together")
        if self.native and self.output_short_side is None:
            raise ValueError("native goes with output_short_side")
        if not isinstance(self.upscale, bool) and not 1.0 < self.upscale <= 2.0:
            raise ValueError("upscale must be true or a factor above 1 and at most 2")
        return self

    def runtime_payload(self) -> dict[str, Any]:
        """The media runtime's ``videogen`` payload: a chain decode."""
        payload = self.model_dump(exclude_none=True, exclude={"latent_artifact_ids", "latent_urls", "latent_paths"})
        if self.latent_urls is not None:
            payload["decode_chain_latent_urls"] = self.latent_urls
        else:
            payload["decode_chain_latent_paths"] = self.latent_paths
        return payload


class ImagegenRequest(BaseModel):
    """The ``imagegen`` request: an H3 still, text-only or edited from references.

    ``preset``: ``standard`` (plain base, 50 steps, best), ``turbo-8`` (8-step Turbo LoRA, close on
    edits, illustrated look text-only, ~2.5x faster), ``turbo-6`` (the 4-step Turbo LoRA at 6 steps) or
    ``turbo-4`` (4-step Turbo LoRA, fastest, roughest). No ``turbo-5`` for stills.

    ``image_urls`` are ``<Picture 1>``..``<Picture N>`` (the prompt may say "image 1" / "picture 2").
    ``width``/``height`` (both or neither) are the delivered size; the render is H3's ~1 MP canvas
    with that aspect, resized. Omitted: an edit follows picture 1's shape, text-only is 768x1024.

    ``rest_frame`` ``k`` (no ``prompt``, exactly one image, the person): rest still ``k`` of them for
    lipsync ``pinned_frames`` ``multi_image`` (vigen's avatar stills, ``h3/rest_stills.py``): same shot a
    few seconds later in a pause, hands at rest, mouth closed, eyes on the lens, head/posture variation
    ``k % 6``; rendered on ``seed + k``. A take that lost the person's face (ArcFace) or opened the mouth is
    retaken on ``+100``, ``+200``, ... at most ``max_retries`` (0..5, default 1) times; out of retakes the
    best take is delivered, never a failure (``likeness {checked, takes, chosen, score, passed, problem}`` in
    the result). Delivered at the image's size, long side <= 2048, unless ``width``/``height`` are
    given. Fan out ``k = 0..n-1`` on one seed.
    """

    model_config = {"extra": "forbid"}

    prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    image_urls: list[str] = Field(default_factory=list, max_length=9)
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    preset: H3StillPreset = "standard"
    rest_frame: int | None = Field(default=None, ge=0, le=1000)
    max_retries: int | None = Field(default=None, ge=0, le=5)

    @model_validator(mode="after")
    def _size_pair(self) -> "ImagegenRequest":
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height go together")
        if self.rest_frame is None and self.prompt is None:
            raise ValueError("prompt is required (only rest_frame writes its own)")
        if self.rest_frame is not None and self.prompt is not None:
            raise ValueError("rest_frame writes its own prompt; send no prompt")
        if self.max_retries is not None and self.rest_frame is None:
            raise ValueError("max_retries goes with rest_frame")
        if self.rest_frame is not None and len(self.image_urls) != 1:
            raise ValueError("rest_frame takes exactly one image_urls entry (the person)")
        return self


#: The lipsync fields ``auto`` sets itself (``h3/lipsync.py`` ``AUTO_CHOSEN``).
LIPSYNC_AUTO_CHOSEN = (
    "pinned_frames",
    "preset",
    "use_interpolation",
    "interpolation_fps",
    "sparse_attention",
    "detect_flicker",
    "detect_desync",
    "detect_glitches",
    "prompt",
)


class LipsyncRequest(H3Interpolation):
    """The ``lipsync`` request: the person in ``image_url`` speaks ``audio_url`` (H3 lipsync LoRA).

    The output lasts as long as the audio and carries it unmodified; it has the image's size, or
    fits inside ``output_dimensions`` with its aspect kept.

    ``auto`` (default true): the avatar render (``h3/lipsync.py`` docstring, **Auto**). Send the
    person's canonical image, the audio and the avatar's rest ``stills`` (none: the image is
    pinned); the preset, attention, validators, prompt, frame rate and pins are ours: 24 fps native,
    the stills pinned at random at the pauses, and frame 0 alone for a clip under 3 s. ``seed``,
    ``transcript``, ``custom_instructions``, ``output_dimensions`` and ``upscale`` still apply. The
    fields auto chooses (:data:`LIPSYNC_AUTO_CHOSEN`) need ``auto: false`` (manual).

    Manual only: ``preset`` ``turbo-3`` (default: the 3-step lipsync student, trained for this mode
    with sparse attention) or one of the videogen presets (``standard``, ``turbo-8``, ``turbo-6``,
    ``turbo-5``, ``turbo-4``, ``dmd-3``) on the same pinned-audio graph; ``pinned_frames``; ``prompt``; the validator switches;
    ``use_interpolation`` (disabled for now, see :class:`H3Interpolation`).
    """

    model_config = {"extra": "forbid"}

    image_url: str = Field(min_length=1)
    audio_url: str = Field(min_length=1)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    prompt: str | None = Field(
        default=None, max_length=8000, description="Prompt template with {{text}} (and optional {{language}})"
    )
    transcript: list[LipsyncTranscriptItem] | None = Field(
        default=None, description="Timed words; omitted, Whisper transcribes the audio"
    )
    custom_instructions: list[LipsyncInstruction] = Field(default_factory=list, max_length=100)
    preset: Literal["turbo-3", "standard", "turbo-8", "turbo-6", "turbo-5", "turbo-4", "dmd-3"] = "turbo-3"
    sparse_attention: bool = True
    detect_flicker: bool = True
    detect_desync: bool = True
    detect_glitches: bool = True
    output_dimensions: OutputDimensions | None = None
    pinned_frames: LipsyncPinnedFrames = Field(
        default_factory=LipsyncPinnedFrames,
        description="What is pinned besides frame 0: none (default), single_image, multi_image or timed_images",
    )
    auto: bool = Field(default=True, description="The avatar render's own setup; false to set the manual fields")
    stills: list[str] = Field(
        default_factory=list, max_length=16, description="auto: the avatar's rest stills, pinned at the pauses"
    )
    stage: Literal["plan"] | None = Field(
        default=None, description="plan: the first task of a staged job (LipsyncStageRequest has the rest)"
    )

    @model_validator(mode="after")
    def _auto_or_manual(self) -> "LipsyncRequest":
        if self.stage == "plan" and self.pinned_frames.mode == "generate":
            raise ValueError(
                "a staged lipsync takes no pinned_frames generate: render the stills as imagegen rest_frame "
                "tasks and pin them (multi_image, or auto stills)"
            )
        if not self.auto:
            if self.stills:
                raise ValueError("stills go with auto; with auto false, pin them with pinned_frames")
            return self
        chosen = [name for name in LIPSYNC_AUTO_CHOSEN if name in self.model_fields_set]
        if chosen:
            raise ValueError(f"auto chooses {', '.join(chosen)}; send auto: false to set them")
        return self

    def runtime_payload(self) -> dict[str, Any]:
        """The payload the media runtime gets: an auto job leaves out the fields auto chooses."""
        return self.model_dump(exclude_none=True, exclude=set(LIPSYNC_AUTO_CHOSEN) if self.auto else None)


class LipsyncStageRequest(BaseModel):
    """A staged ``lipsync`` job's later tasks (``h3/lipsync.py`` :func:`run_lipsync_inference`). The job's
    first task is its :class:`LipsyncRequest` with ``stage: "plan"``: its result is the state tar (the
    main artifact) plus the job's audio (the ``audio`` file) and ``chunkPlan``. Then:

    - ``stage: "chunk"``, ``chunk_index`` k, the state; a chunk whose ``chunkPlan`` entry
      ``needsBoundary`` also chunk k-1's ``boundary`` file. Chunks run in order. Result: the chunk's
      render (mp4, the main artifact), ``report``, and its ``boundary`` file when it ends on one (every
      chunk whose successor needs it).
    - ``stage: "assemble"``: the state, every chunk's render in order, the plan's ``audio`` and,
      optionally, the chunks' ``report`` data as ``chunk_reports``. Result: the job's mp4 and data, as
      the whole job returns them.

    Taskflow sends ``*_artifact_id(s)`` (the worker reads them from the lease into ``*_path(s)``);
    /task/sync sends ``*_url(s)``."""

    model_config = {"extra": "forbid"}

    stage: Literal["chunk", "assemble"]
    chunk_index: int | None = Field(default=None, ge=0, le=999)
    state_artifact_id: str | None = None
    state_url: str | None = None
    state_path: str | None = Field(default=None, description="Worker-staged")
    boundary_artifact_id: str | None = None
    boundary_url: str | None = None
    boundary_path: str | None = Field(default=None, description="Worker-staged")
    chunk_artifact_ids: list[str] | None = Field(default=None, min_length=1, max_length=1000)
    chunk_urls: list[str] | None = Field(default=None, min_length=1, max_length=1000)
    chunk_paths: list[str] | None = Field(default=None, min_length=1, max_length=1000, description="Worker-staged")
    audio_artifact_id: str | None = None
    audio_url: str | None = None
    audio_path: str | None = Field(default=None, description="Worker-staged")
    chunk_reports: list[dict[str, Any]] | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _stage_inputs(self) -> "LipsyncStageRequest":
        def given(name: str) -> int:
            return sum(getattr(self, f"{name}_{kind}", None) is not None for kind in ("url", "path"))

        def sent(name: str, plural: bool = False) -> bool:
            kinds = ("artifact_ids", "urls", "paths") if plural else ("artifact_id", "url", "path")
            return any(getattr(self, f"{name}_{kind}", None) is not None for kind in kinds)

        if given("state") != 1:
            raise ValueError("send the plan's state: state_url (or, on Taskflow, state_artifact_id)")
        if given("boundary") > 1:
            raise ValueError("send one of boundary_url and boundary_path")
        if self.stage == "chunk":
            if self.chunk_index is None:
                raise ValueError("a chunk stage needs chunk_index")
            if sent("chunk", plural=True) or sent("audio") or self.chunk_reports is not None:
                raise ValueError("chunk renders, audio and chunk_reports go with stage assemble")
        else:
            if self.chunk_index is not None or sent("boundary"):
                raise ValueError("chunk_index and boundary go with stage chunk")
            if sum(x is not None for x in (self.chunk_urls, self.chunk_paths)) != 1:
                raise ValueError("send the chunk renders: chunk_urls (or, on Taskflow, chunk_artifact_ids)")
            if given("audio") != 1:
                raise ValueError("send the plan's audio: audio_url (or, on Taskflow, audio_artifact_id)")
        return self

    def runtime_payload(self) -> dict[str, Any]:
        return self.model_dump(
            exclude_none=True,
            exclude={"state_artifact_id", "boundary_artifact_id", "chunk_artifact_ids", "audio_artifact_id"},
        )


async def fetch_reference(url: str, dest: Path) -> Path:
    """Download the clone reference to ``dest``; a fetch failure is the caller's bad input."""
    try:
        return await fetch_to_file(
            url,
            dest,
            MAX_REFERENCE_BYTES,
            url_name="reference_url",
            timeout_seconds=REFERENCE_TIMEOUT_SECONDS,
        )
    except DownloadError as exc:
        raise MediaInputError(str(exc)) from exc


class MediaRuntime:
    """The media runtime child process and the channel it was spawned with; one per API process.

    A request goes down one pipe and its reply comes back up the other, matched by id. The child
    answers each on a thread of its own, so a reply never waits behind the render in front of it,
    and the two messages that answer nothing — the state of its models, the progress of the job
    it is rendering — arrive on the same pipe.
    """

    def __init__(
        self,
        python: str,
        runtime_dir: str,
        gpu: str | None,
        ready_timeout_seconds: float,
        features: list[str],
    ):
        # Relative paths are the repo's (server config), not the runtime dir the child starts in.
        repo_dir = Path(__file__).resolve().parents[2]
        self.python = str(repo_dir / python) if not Path(python).is_absolute() and "/" in python else python
        self.runtime_dir = repo_dir / runtime_dir
        self.gpu = gpu
        self.ready_timeout_seconds = ready_timeout_seconds
        self.features = list(features)
        self._process: subprocess.Popen | None = None
        self._start_lock = asyncio.Lock()
        #: The loop the requests and the progress sinks belong to; replies are read on a thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        #: The write end of the request pipe, and the lock that keeps two lines from interleaving.
        self._request_fd: int | None = None
        self._write_lock = threading.Lock()
        #: The requests waiting for a reply, by id.
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)
        #: What the child last said about its models, and the wait for it to say anything at all.
        self._state: dict[str, Any] = {"status": "starting", "reason": "spawning"}
        self._settled: asyncio.Future | None = None

    def _start(self) -> None:
        self.stop()
        env = dict(os.environ)
        if self.gpu is not None:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        _LOGGER.info("Starting media runtime (gpu=%s, features=%s)", self.gpu, self.features)
        request_read, request_write = os.pipe()
        reply_read, reply_write = os.pipe()
        env[REQUEST_FD_VARIABLE] = str(request_read)
        env[REPLY_FD_VARIABLE] = str(reply_write)
        try:
            self._process = subprocess.Popen(
                [
                    self.python,
                    "server.py",
                    "--device",
                    "cuda:0",
                    "--features",
                    ",".join(self.features),
                ],
                cwd=self.runtime_dir,
                env=env,
                start_new_session=True,
                preexec_fn=exit_with_parent,
                pass_fds=(request_read, reply_write),
            )
        except BaseException:
            os.close(request_write)
            os.close(reply_read)
            raise
        finally:
            # The child holds the ends it was given now: each pipe ends when it does.
            os.close(request_read)
            os.close(reply_write)
        self._request_fd = request_write
        self._read_replies(reply_read)

    def _read_replies(self, reply_fd: int) -> None:
        """Hands each line the child sends to the loop: a reply, its state, or a job's progress.

        Read on a thread of its own, because the pipe blocks between messages. The thread ends
        with the pipe, which ends with the child.
        """
        loop = self._loop
        if loop is None:
            os.close(reply_fd)
            return

        def pump() -> None:
            try:
                with os.fdopen(reply_fd, "r", buffering=1) as replies:
                    for line in replies:
                        try:
                            message = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(message, dict):
                            continue
                        try:
                            loop.call_soon_threadsafe(self._deliver, message)
                        except RuntimeError:
                            # The loop is gone, so nobody is waiting on any of this.
                            return
            finally:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._channel_closed)

        threading.Thread(target=pump, name="media-runtime-replies", daemon=True).start()

    def _deliver(self, message: dict[str, Any]) -> None:
        """One message from the child, on the loop."""
        if "progress" in message:
            body = message["progress"]
            if isinstance(body, dict) and "jobId" in body and "stage" in body:
                report_progress(body["jobId"], body["stage"], body.get("done"), body.get("total"))
            return
        if "state" in message:
            if isinstance(message["state"], dict):
                self._state = message["state"]
            self._settle()
            return
        request = self._pending.pop(message.get("id"), None)
        if request is None or request.done():
            # Nobody is waiting: the request timed out, or this is a reply to a dead generation.
            return
        error = message.get("error")
        if error:
            kind = error.get("kind") if isinstance(error, dict) else None
            text = (error.get("message") if isinstance(error, dict) else str(error)) or "media runtime failed"
            failure = (
                MediaInputError(text)
                if kind == "input"
                else MediaCancelled(text) if kind == "cancelled" else MediaRuntimeError(text)
            )
            request.set_exception(failure)
            return
        result = message.get("result")
        request.set_result(result if isinstance(result, dict) else {})

    def _channel_closed(self) -> None:
        """The child's pipe ended: it has exited, and nothing it was asked is coming back."""
        if self._state.get("status") != "error":
            self._state = {"status": "error", "reason": "media runtime exited"}
        self._settle()
        pending, self._pending = self._pending, {}
        for request in pending.values():
            if not request.done():
                request.set_exception(MediaRuntimeError("media runtime exited"))

    def _settle(self) -> None:
        """Wakes whoever is waiting for the child to have loaded, or failed to."""
        if self._settled is not None and not self._settled.done():
            self._settled.set_result(None)

    async def ensure_ready(self) -> None:
        """Start the runtime if it is not running, then wait until its models are loaded."""
        async with self._start_lock:
            if self._process is None or self._process.poll() is not None:
                self._loop = asyncio.get_running_loop()
                self._state = {"status": "starting", "reason": "loading models"}
                self._settled = self._loop.create_future()
                await asyncio.to_thread(self._start)
            settled = self._settled
            if settled is not None and not settled.done():
                try:
                    await asyncio.wait_for(asyncio.shield(settled), self.ready_timeout_seconds)
                except asyncio.TimeoutError:
                    raise MediaRuntimeError(
                        f"media runtime not ready after {self.ready_timeout_seconds:.0f} s"
                    ) from None
            if self._state.get("status") != "ok":
                raise MediaRuntimeError(f"media runtime failed: {self._state.get('reason')}")

    async def request(
        self,
        op: str,
        body: dict[str, Any],
        timeout_seconds: float,
        cancelled: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        """Run one task on the runtime and wait for its reply; an ``input`` error is the caller's.

        A job nobody waits for any more must not keep the GPU: when ``cancelled`` is set (its lease
        was released), the request times out, or the caller is cancelled, the runtime is told to stop
        the job (``body["job_id"]``). It answers with a ``cancelled`` error once it has, raised as
        :class:`MediaCancelled`.
        """
        await self.ensure_ready()
        loop = asyncio.get_running_loop()
        request_id = next(self._ids)
        reply: asyncio.Future = loop.create_future()
        self._pending[request_id] = reply
        job_id = body.get("job_id")
        try:
            await asyncio.to_thread(self._send, {"id": request_id, "op": op, "body": body})
            try:
                if await self._reply_or_cancel(reply, cancelled, timeout_seconds):
                    return reply.result()
            except asyncio.CancelledError:
                self._cancel_job(job_id)
                raise
            if cancelled is not None and cancelled.is_set():
                self._cancel_job(job_id)
                await self._stopped(reply, job_id)
                raise MediaCancelled(f"media job {job_id} was cancelled")
            self._cancel_job(job_id)
            await self._stopped(reply, job_id)
            raise MediaRuntimeError(f"media runtime {op} timed out after {timeout_seconds:.0f} s")
        finally:
            self._pending.pop(request_id, None)

    @staticmethod
    async def _reply_or_cancel(
        reply: asyncio.Future, cancelled: asyncio.Event | None, timeout_seconds: float
    ) -> bool:
        """Waits for the reply; False when ``cancelled`` was set or the time ran out first."""
        waiter = asyncio.ensure_future(cancelled.wait()) if cancelled is not None else None
        try:
            await asyncio.wait(
                [reply] if waiter is None else [reply, waiter],
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if waiter is not None:
                waiter.cancel()
        return reply.done()

    def _cancel_job(self, job_id: Any) -> None:
        """Tells the runtime to stop ``job_id``; its own reply is not waited for (nobody needs it)."""
        if not isinstance(job_id, str):
            return
        _LOGGER.info("Cancelling media job %s", job_id)
        with contextlib.suppress(MediaRuntimeError):
            self._send({"id": next(self._ids), "op": "cancel", "body": {"job_id": job_id}})

    @staticmethod
    async def _stopped(reply: asyncio.Future, job_id: Any) -> None:
        """Waits for the cancelled job's reply: until it comes, the runtime still holds the GPU."""
        try:
            await asyncio.wait_for(asyncio.shield(reply), CANCEL_GRACE_SECONDS)
        except asyncio.TimeoutError:
            _LOGGER.warning("Media job %s not stopped %.0f s after its cancel", job_id, CANCEL_GRACE_SECONDS)
        except Exception:  # pylint: disable=broad-exception-caught
            # The cancelled error, or the job failing on its own: either way it has stopped.
            pass

    def _send(self, message: dict[str, Any]) -> None:
        """Writes one request down the pipe, under the lock so two lines never interleave.

        A write to a pipe is whole only up to ``PIPE_BUF``, and a lipsync request carries its
        transcript and its instructions, so the rest of a long line is written in the loop.
        """
        fd = self._request_fd
        if fd is None:
            raise MediaRuntimeError("media runtime is not running")
        line = memoryview(f"{json.dumps(message)}\n".encode())
        with self._write_lock:
            try:
                while line:
                    line = line[os.write(fd, line) :]
            except OSError as exc:
                raise MediaRuntimeError(f"media runtime request could not be sent: {exc}") from exc

    def stop(self) -> None:
        process = self._process
        self._process = None
        fd, self._request_fd = self._request_fd, None
        if fd is not None:
            # Closing the request pipe is how the child is told there is no more work.
            with contextlib.suppress(OSError):
                os.close(fd)
        if process is None or process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
