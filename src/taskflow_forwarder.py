"""Forwarding Taskflow worker for LZ-TTS — dev helper.

Protocol-identical to :mod:`src.taskflow_worker` (join / heartbeat / pull /
lease fencing / artifact upload / synthesis-event acks are all reused
verbatim), but instead of a local model runtime each task operation is
forwarded over HTTP to a real LZ-TTS service's ``/task/sync`` endpoint (the
in-process HTTP adapter that every lz-tts worker exposes on ``PORT``).

This lets a small dev machine serve a Lazybird dev API's ``tts-synthesis``
and ``voice-enhance`` queues without loading the multi-GB model runtime:
the forwards execute on the production worker box over the LAN/Tailscale.

Env:
  LZB_API                    taskflow base url (default http://localhost:4001)
  TASKFLOW_WORKER_TOKEN      worker auth token (required)
  TASKFLOW_WORKER_ID         worker id (default lz-tts-forwarder-<host>)
  TASKFLOW_WORKER_CONCURRENCY  lease batch size (default 4)
  LZTTS_FORWARD_URL          base url of the remote lz-tts HTTP adapter
                             (default http://home.vnet.local:8000)
  API_KEY                    api key for the remote adapter (required)
  PORT                       this worker's own HTTP adapter port (default 8010)

Run: python3 -m src.taskflow_forwarder  (or via console script once added)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import logging
import os
import signal
import socket
import threading
from typing import Any

import httpx
from dotenv import load_dotenv

from .api.server import (
    InferenceOperationError,
    InferenceResult,
    LzTtsInferenceSession,
    _env_bool,
    create_app,
    get_health_status,
    set_status,
)
from .process_guard import hard_exit
from .taskflow_worker import (
    _REQUEST_TIMEOUT,
    SynthesisAckBatcher,
    TaskflowWorker,
    _start_http_server,
    _serve_taskflow,
)

_LOGGER = logging.getLogger(__name__)

_FORWARD_TIMEOUT = httpx.Timeout(connect=30.0, read=1800.0, write=1800.0, pool=60.0)


class ForwardingInferenceSession:
    """Duck-typed stand-in for ``LzTtsInferenceSession``.

    Same four-method surface the Taskflow worker expects (start / close /
    synthesis_capabilities / execute_many), but every operation executes on a
    remote lz-tts HTTP adapter and comes back as an ``InferenceResult``.
    Error mapping mirrors the real worker contract: HTTP >= 500 and 408/429
    are retriable (``InferenceOperationError`` with that status), other 4xx
    are terminal.
    """

    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key.strip()
        # Config-only instance used to advertise the same synthesis
        # capabilities a real worker of this checkout would report on join.
        # Model runtimes are NOT started (config read is cheap; start() is a
        # no-op here).
        self._capabilities_source = LzTtsInferenceSession()
        self._client = httpx.AsyncClient(timeout=_FORWARD_TIMEOUT)
        self._started = False

    @property
    def config(self):
        # create_app (the in-process HTTP adapter) reads session.config.
        return self._capabilities_source.config

    async def start(self) -> None:
        if not self._api_key:
            raise RuntimeError("API_KEY is required to forward to the remote lz-tts service")
        if not self._base_url:
            raise RuntimeError("LZTTS_FORWARD_URL is required for the forwarding worker")
        self._started = True

    async def close(self) -> None:
        await self._client.aclose()
        self._started = False

    def synthesis_capabilities(self) -> dict[str, Any]:
        return self._capabilities_source.synthesis_capabilities()

    async def execute_many(
        self,
        operations: list[tuple[str, dict[str, Any]]],
    ) -> list[Any]:
        if not self._started:
            raise RuntimeError("Forwarding inference session has not been started")
        outcomes = await asyncio.gather(
            *(self._forward(operation, request) for operation, request in operations),
            return_exceptions=True,
        )
        return list(outcomes)

    async def _forward(self, operation: str, request: dict[str, Any]) -> Any:
        url = f"{self._base_url}/task/sync"
        try:
            response = await self._client.post(
                url,
                json={"input": {"operation": operation, "request": request or {}}},
                headers={"X-Api-Key": self._api_key},
            )
        except httpx.HTTPError as error:
            _LOGGER.error("Forward to remote lz-tts failed (transport) url=%s: %s", url, error)
            return InferenceOperationError(503, f"forward transport failure: {error}")

        try:
            body = response.json()
        except ValueError:
            body = {}
        output = body.get("output") if isinstance(body, dict) else None
        failed = body.get("status") in ("FAILED", "REJECTED", None) or output is None

        if failed:
            status = response.status_code
            detail = body.get("error", f"remote returned HTTP {status}")
            if status >= 500 or status in (408, 429):
                return InferenceOperationError(503, f"remote error: {detail}")
            if status < 400:
                # Malformed 2xx response — treat as an upstream failure.
                return InferenceOperationError(502, f"malformed remote response: {str(body)[:400]}")
            return InferenceOperationError(status, detail)

        if output.get("kind") == "json":
            return InferenceResult(kind="json", data=output.get("data"))

        audio_b64 = output.get("audioBase64")
        if not audio_b64:
            return InferenceOperationError(502, "remote completed without audio base64")
        try:
            audio = base64.b64decode(audio_b64, validate=False)
        except (ValueError, TypeError) as error:
            return InferenceOperationError(502, f"remote audio base64 invalid: {error}")
        return InferenceResult(
            kind="audio",
            content_type=output.get("contentType") or "audio/mpeg",
            audio=audio,
        )


async def run_forwarder_worker() -> None:
    load_dotenv()
    worker_token = os.environ.get("TASKFLOW_WORKER_TOKEN", "").strip()
    if not worker_token:
        raise RuntimeError("TASKFLOW_WORKER_TOKEN is required to authenticate with Lazybird Taskflow")
    forward_url = os.environ.get("LZTTS_FORWARD_URL", "http://home.vnet.local:8000").strip()
    api_key = os.environ.get("API_KEY", "").strip()

    taskflow = TaskflowWorker(
        base_url=f"{os.environ.get('LZB_API', 'http://localhost:4001').rstrip('/')}/internal/taskflow/v1",
        worker_token=worker_token,
        worker_id=os.environ.get("TASKFLOW_WORKER_ID", f"lz-tts-forwarder-{socket.gethostname()}"),
        concurrency=max(1, int(os.environ.get("TASKFLOW_WORKER_CONCURRENCY", "8"))),
        persistent=_env_bool("TASKFLOW_WORKER_PERSISTENT", False),
    )
    inference = ForwardingInferenceSession(forward_url, api_key)
    try:
        await inference.start()
    except BaseException:  # pylint: disable=broad-exception-caught
        _LOGGER.exception("Forwarding inference session startup failed")
        hard_exit("inference runtime startup failed")
    http_server, http_task = _start_http_server(inference)
    await asyncio.sleep(0)
    main_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    shutdown_grace = max(1.0, float(os.environ.get("LZ_TTS_SHUTDOWN_GRACE_SECONDS", "20")))
    shutdown_watchdog = threading.Timer(
        shutdown_grace,
        hard_exit,
        args=("graceful shutdown exceeded its grace period",),
    )
    shutdown_watchdog.daemon = True
    shutdown_requested = False

    def _request_shutdown() -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        shutdown_watchdog.start()
        main_task.cancel()

    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(shutdown_signal, _request_shutdown)
    acks = SynthesisAckBatcher(
        taskflow._client,
        f"{os.environ.get('LZB_API', 'http://localhost:4001').rstrip('/')}/internal/synthesis-events/v1/batch",
    )
    acks.start()
    try:
        synthesis_capabilities = inference.synthesis_capabilities()
        await _serve_taskflow(taskflow, inference, synthesis_capabilities, acks)
    except asyncio.CancelledError:
        _LOGGER.info("LZ-TTS forwarding worker shutdown requested")
    finally:
        shutdown_watchdog.cancel()
        http_server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await http_task
        set_status("starting")
        await acks.stop()
        await taskflow.close()
        await inference.close()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run_forwarder_worker())
    except BaseException:  # pylint: disable=broad-exception-caught
        _LOGGER.exception("LZ-TTS forwarding worker failed; exiting")
        raise


if __name__ == "__main__":
    run()