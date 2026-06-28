"""Small reusable process-worker transport for model-serving engines."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
    stream=sys.stdout,
)
_LOGGER = logging.getLogger(__name__)


def error_response(exc: Exception) -> dict[str, Any]:
    """Serialize an exception across a multiprocessing queue."""
    if isinstance(exc, HTTPException):
        return {
            "ok": False,
            "status_code": exc.status_code,
            "detail": exc.detail,
        }
    return {
        "ok": False,
        "status_code": 500,
        "error": type(exc).__name__,
        "detail": str(exc),
    }


def run_worker_loop(engine_name: str, handler: Callable[[str, Any], dict[str, Any]], request_queue: Any, response_queue: Any) -> None:
    """Run a simple action/payload worker loop."""
    _LOGGER.info("%s worker ready pid=%s", engine_name, os.getpid())
    while True:
        message = request_queue.get()
        if not isinstance(message, dict):
            response_queue.put({"ok": False, "status_code": 400, "detail": "invalid worker message", "request_id": None})
            continue

        request_id = message.get("request_id")
        action = message.get("action")
        payload = message.get("payload")
        if action == "shutdown":
            _LOGGER.info("%s worker shutting down pid=%s", engine_name, os.getpid())
            return

        try:
            response = handler(str(action), payload)
            if isinstance(response, dict):
                response_queue.put({**response, "request_id": request_id})
            else:
                response_queue.put({"ok": False, "status_code": 502, "detail": "invalid worker handler response", "request_id": request_id})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.exception("%s worker action failed: %s", engine_name, action)
            response_queue.put({**error_response(exc), "request_id": request_id})


class WorkerProcessClient:
    """Parent-process helper for a single long-lived model worker."""

    def __init__(
        self,
        *,
        name: str,
        target: Callable[..., None],
        args: tuple[Any, ...] = (),
    ):
        self.name = name
        self.target = target
        self.args = args
        self.process: mp.Process | None = None
        self.requests: Any | None = None
        self.responses: Any | None = None
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()

    def start(self) -> None:
        with self.start_lock:
            if self.process is not None and self.process.is_alive():
                return
            ctx = mp.get_context("spawn")
            self.requests = ctx.Queue()
            self.responses = ctx.Queue()
            self.process = ctx.Process(
                target=self.target,
                args=(*self.args, self.requests, self.responses),
                name=f"lz-tts-{self.name}-worker",
                daemon=False,
            )
            self.process.start()

    def stop(self) -> None:
        with self.start_lock:
            process = self.process
            requests = self.requests
            if process is not None and process.is_alive() and requests is not None:
                try:
                    requests.put({"action": "shutdown", "payload": None})
                    process.join(timeout=10)
                except Exception:  # pylint: disable=broad-exception-caught
                    _LOGGER.exception("Failed graceful %s worker shutdown", self.name)
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=10)
            self.process = None
            self.requests = None
            self.responses = None

    def call(self, action: str, payload: Any | None = None) -> dict[str, Any]:
        self.start()
        if (
            self.process is None
            or self.requests is None
            or self.responses is None
            or not self.process.is_alive()
        ):
            raise HTTPException(status_code=503, detail=f"{self.name} worker is not running")

        request_id = uuid.uuid4().hex
        self.lock.acquire()
        started = time.perf_counter()
        _LOGGER.info(
            "%s worker request start action=%s request_id=%s pid=%s",
            self.name,
            action,
            request_id,
            self.process.pid if self.process is not None else None,
        )
        try:
            self.requests.put({"request_id": request_id, "action": action, "payload": payload})
            while True:
                response = self.responses.get()
                if not isinstance(response, dict):
                    raise HTTPException(status_code=502, detail=f"{self.name} worker returned invalid response")
                if response.get("request_id") == request_id:
                    break
                _LOGGER.warning(
                    "Discarding stale %s worker response action=%s expected_request_id=%s got_request_id=%s",
                    self.name,
                    action,
                    request_id,
                    response.get("request_id"),
                )
        finally:
            self.lock.release()

        elapsed = time.perf_counter() - started
        if response.get("ok"):
            _LOGGER.info(
                "%s worker request done action=%s request_id=%s elapsed=%.2fs",
                self.name,
                action,
                request_id,
                elapsed,
            )
            return response

        status_code = int(response.get("status_code") or 500)
        detail = response.get("detail") or response.get("error") or f"{self.name} worker failed"
        _LOGGER.warning(
            "%s worker request failed action=%s request_id=%s status_code=%s elapsed=%.2fs detail=%s",
            self.name,
            action,
            request_id,
            status_code,
            elapsed,
            detail,
        )
        raise HTTPException(status_code=status_code, detail=detail)
