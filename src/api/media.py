"""Client for the media runtime process (src/media_runtime): generate-music, videogen, lipsync, imagegen.

YuE2, YingMusic-SVC and ComfyUI/H3 run from their own venv (Python 3.11, torch 2.14, cu130), so
the runtime cannot be a multiprocessing child like the speech and transcribe workers. The API
starts it as ``server.py --port N --features ...``, polls GET /health until it reports ``ok``, and
sends each task as one POST on 127.0.0.1 (/generate, /videogen, /lipsync, /imagegen).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, model_validator

from ..process_guard import exit_with_parent

_LOGGER = logging.getLogger(__name__)

MAX_REFERENCE_BYTES = 30 * 1024 * 1024
REFERENCE_TIMEOUT_SECONDS = 60.0


class MediaInputError(ValueError):
    """The request is invalid for the media runtime (HTTP 400)."""


class MediaRuntimeError(RuntimeError):
    """The media runtime failed to start, load, or render."""


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


class VideogenRequest(BaseModel):
    """The ``videogen`` request: H3 image-to-video (4-step Turbo), with H3's own audio track.

    ``width``/``height`` (both or neither) are the delivered frame size; omitted, the clip keeps
    the image's aspect on H3's 32-px lattice (about 1 MP).
    """

    model_config = {"extra": "forbid"}

    prompt: str = Field(min_length=1, max_length=4000, description="What happens in the clip")
    image_url: str = Field(min_length=1, description="First frame")
    last_frame_url: str | None = Field(default=None, description="Optional last frame")
    duration_seconds: float = Field(default=3.0, gt=0, le=10, description="Clip length at 24 fps")
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    generate_audio: bool = True

    @model_validator(mode="after")
    def _size_pair(self) -> "VideogenRequest":
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height go together")
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


class OutputDimensions(BaseModel):
    model_config = {"extra": "forbid"}

    width: int = Field(ge=64, le=4096)
    height: int = Field(ge=64, le=4096)


class ImagegenRequest(BaseModel):
    """The ``imagegen`` request: an H3 still (50 steps, plain base), text-only or edited from references.

    ``image_urls`` are ``<Picture 1>``..``<Picture N>`` (the prompt may say "image 1" / "picture 2").
    ``width``/``height`` (both or neither) are the delivered size; the render is H3's ~1 MP canvas
    with that aspect, resized. Omitted: an edit follows picture 1's shape, text-only is 768x1024.
    """

    model_config = {"extra": "forbid"}

    prompt: str = Field(min_length=1, max_length=4000)
    image_urls: list[str] = Field(default_factory=list, max_length=9)
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _size_pair(self) -> "ImagegenRequest":
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height go together")
        return self


class LipsyncRequest(BaseModel):
    """The ``lipsync`` request: the person in ``image_url`` speaks ``audio_url`` (H3 lipsync LoRA).

    The output lasts as long as the audio and carries it unmodified; it has the image's size, or
    fits inside ``output_dimensions`` with its aspect kept.
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
    sparse_attention: bool = True
    detect_flicker: bool = True
    detect_desync: bool = True
    detect_glitches: bool = True
    output_dimensions: OutputDimensions | None = None


async def fetch_reference(url: str, dest: Path) -> Path:
    """Download the clone reference to ``dest``; a fetch failure is the caller's bad input."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=REFERENCE_TIMEOUT_SECONDS) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                written = 0
                with dest.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        written += len(chunk)
                        if written > MAX_REFERENCE_BYTES:
                            raise MediaInputError("reference_url is larger than 30 MB")
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        raise MediaInputError(f"reference_url could not be fetched: {exc}") from exc
    return dest


class MediaRuntime:
    """The media runtime child process and its HTTP client; one instance per API process."""

    def __init__(
        self,
        python: str,
        runtime_dir: str,
        port: int,
        gpu: str | None,
        ready_timeout_seconds: float,
        features: list[str],
    ):
        # Relative paths are the repo's (server.json), not the runtime dir the child starts in.
        repo_dir = Path(__file__).resolve().parents[2]
        self.python = str(repo_dir / python) if not Path(python).is_absolute() and "/" in python else python
        self.runtime_dir = repo_dir / runtime_dir
        self.port = port
        self.gpu = gpu
        self.ready_timeout_seconds = ready_timeout_seconds
        self.features = list(features)
        self._process: subprocess.Popen | None = None
        self._start_lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _start(self) -> None:
        self.stop()
        env = dict(os.environ)
        if self.gpu is not None:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        _LOGGER.info("Starting media runtime on port %s (gpu=%s, features=%s)", self.port, self.gpu, self.features)
        self._process = subprocess.Popen(
            [
                self.python,
                "server.py",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--device",
                "cuda:0",
                "--features",
                ",".join(self.features),
            ],
            cwd=self.runtime_dir,
            env=env,
            start_new_session=True,
            preexec_fn=exit_with_parent,
        )

    async def ensure_ready(self) -> None:
        """Start the runtime if it is not running, then wait until its models are loaded."""
        async with self._start_lock:
            if self._process is None or self._process.poll() is not None:
                await asyncio.to_thread(self._start)
            await self._wait_ready()

    async def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_seconds
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
            while True:
                if self._process is None or self._process.poll() is not None:
                    raise MediaRuntimeError("media runtime exited before it was ready")
                try:
                    health = (await client.get("/health")).json()
                except (httpx.HTTPError, ValueError):
                    health = {"status": "starting", "reason": "not listening yet"}
                status = health.get("status")
                if status == "ok":
                    return
                if status == "error":
                    raise MediaRuntimeError(f"media runtime failed: {health.get('reason')}")
                if time.monotonic() >= deadline:
                    raise MediaRuntimeError(f"media runtime not ready after {self.ready_timeout_seconds:.0f} s")
                await asyncio.sleep(2.0)

    async def post(self, path: str, body: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        """POST one task to the runtime; a 400 is the caller's bad input, anything else a runtime failure."""
        await self.ensure_ready()
        timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=60.0, pool=10.0)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
            try:
                response = await client.post(path, json=body)
            except httpx.HTTPError as exc:
                raise MediaRuntimeError(f"media runtime request failed: {exc}") from exc
        if response.status_code == 400:
            raise MediaInputError(str(response.json().get("detail", "invalid media request")))
        if response.status_code >= 400:
            raise MediaRuntimeError(f"media runtime returned {response.status_code}: {response.text[:500]}")
        return response.json()

    def stop(self) -> None:
        process = self._process
        self._process = None
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
