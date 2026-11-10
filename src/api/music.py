"""generate-music: client for the music runtime process (src/music_runtime).

ACE-Step and YingMusic-SVC run from their own venv (Python 3.11, torch 2.10), so
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
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..process_guard import exit_with_parent

_LOGGER = logging.getLogger(__name__)

MAX_REFERENCE_BYTES = 30 * 1024 * 1024
REFERENCE_TIMEOUT_SECONDS = 60.0


class MusicInputError(ValueError):
    """The request is invalid for the music runtime (HTTP 400)."""


class MusicRuntimeError(RuntimeError):
    """The music runtime failed to start, load, or generate."""


class MusicRequest(BaseModel):
    """The ``generate-music`` request. Without ``reference_url`` the song is ACE-Step's render; with it, the lead vocal is re-voiced by YingMusic-SVC."""

    caption: str = Field(min_length=1, max_length=2000)
    duration: float = Field(default=30.0, gt=0, le=240.0)
    lyrics: str = Field(default="", max_length=8000)
    instrumental: bool = False
    vocal_language: str = Field(default="en", max_length=16)
    bpm: int | None = Field(default=None, ge=30, le=300)
    keyscale: str | None = Field(default=None, max_length=40)
    timesignature: str = Field(default="4/4", max_length=8)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 2)
    inference_steps: int = Field(default=8, ge=1, le=100)
    reference_url: str | None = None


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
