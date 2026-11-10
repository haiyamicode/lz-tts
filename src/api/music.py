"""generate-music: client for the music runtime process (src/music_runtime).

YuE2 and YingMusic-SVC run from their own venv (Python 3.11, torch 2.10), so
the runtime cannot be a multiprocessing child like the speech and transcribe
workers. The API starts it as ``server.py --port N``, polls GET /health until it
reports ``ok``, and sends each task as POST /generate on 127.0.0.1.
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


class MusicInputError(ValueError):
    """The request is invalid for the music runtime (HTTP 400)."""


class MusicRuntimeError(RuntimeError):
    """The music runtime failed to start, load, or generate."""


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
                            raise MusicInputError("reference_url is larger than 30 MB")
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        raise MusicInputError(f"reference_url could not be fetched: {exc}") from exc
    return dest


class MusicRuntime:
    """The music runtime child process and its HTTP client; one instance per API process."""

    def __init__(
        self,
        python: str,
        runtime_dir: str,
        port: int,
        gpu: str | None,
        ready_timeout_seconds: float,
    ):
        self.python = python
        self.runtime_dir = Path(runtime_dir)
        self.port = port
        self.gpu = gpu
        self.ready_timeout_seconds = ready_timeout_seconds
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
        _LOGGER.info("Starting music runtime on port %s (gpu=%s)", self.port, self.gpu)
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
                    raise MusicRuntimeError("music runtime exited before it was ready")
                try:
                    health = (await client.get("/health")).json()
                except (httpx.HTTPError, ValueError):
                    health = {"status": "starting", "reason": "not listening yet"}
                status = health.get("status")
                if status == "ok":
                    return
                if status == "error":
                    raise MusicRuntimeError(f"music runtime failed: {health.get('reason')}")
                if time.monotonic() >= deadline:
                    raise MusicRuntimeError(f"music runtime not ready after {self.ready_timeout_seconds:.0f} s")
                await asyncio.sleep(2.0)

    async def generate(self, body: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        await self.ensure_ready()
        timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=60.0, pool=10.0)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
            try:
                response = await client.post("/generate", json=body)
            except httpx.HTTPError as exc:
                raise MusicRuntimeError(f"music runtime request failed: {exc}") from exc
        if response.status_code == 400:
            raise MusicInputError(str(response.json().get("detail", "invalid music request")))
        if response.status_code >= 400:
            raise MusicRuntimeError(f"music runtime returned {response.status_code}: {response.text[:500]}")
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
