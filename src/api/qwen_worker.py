"""Isolated multiprocessing worker for Qwen3 TTS serving."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from fastapi import HTTPException


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
_LOGGER = logging.getLogger(__name__)


def _error_response(exc: Exception) -> dict[str, Any]:
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


def qwen_worker_main(
    settings_data: dict[str, Any],
    request_queue: Any,
    response_queue: Any,
) -> None:
    """Run the Qwen model in a dedicated OS process.

    The parent process owns HTTP. This worker owns Qwen model loading, CUDA graph
    state, and Qwen inference.
    """
    from . import qwen3

    _LOGGER.info("Qwen worker starting pid=%s", os.getpid())
    settings = qwen3.QwenSettings(**settings_data)
    qwen3.configure(settings)
    qwen3.preload_model(
        background=False,
        include_dp_budget=settings.dp_budget.enabled,
    )
    _LOGGER.info("Qwen worker ready pid=%s", os.getpid())

    while True:
        message = request_queue.get()
        if not isinstance(message, dict):
            response_queue.put(
                {
                    "ok": False,
                    "status_code": 400,
                    "detail": "invalid worker message",
                }
            )
            continue

        action = message.get("action")
        payload = message.get("payload")
        if action == "shutdown":
            _LOGGER.info("Qwen worker shutting down pid=%s", os.getpid())
            return

        try:
            if action == "health":
                response_queue.put(
                    {
                        "ok": True,
                        "data": {
                            "worker": "ok",
                            "worker_pid": os.getpid(),
                            **qwen3.model_status(),
                        },
                    }
                )
            elif action == "synthesize":
                req = qwen3.SynthesizeRequest(**(payload or {}))
                response = qwen3.synthesize(req)
                response_queue.put(
                    {
                        "ok": True,
                        "media_type": response.media_type,
                        "content": response.body,
                    }
                )
            elif action == "synthesize_batch":
                req = qwen3.BatchSynthesizeRequest(**(payload or {}))
                response = qwen3.synthesize_batch(req)
                response_queue.put(
                    {
                        "ok": True,
                        "data": response.model_dump(mode="json"),
                    }
                )
            else:
                response_queue.put(
                    {
                        "ok": False,
                        "status_code": 400,
                        "detail": f"unknown worker action: {action}",
                    }
                )
        except Exception as exc:
            _LOGGER.exception("Qwen worker action failed: %s", action)
            response_queue.put(_error_response(exc))
