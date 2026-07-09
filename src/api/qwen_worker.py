"""Isolated multiprocessing worker for Qwen3 TTS serving."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
_LOGGER = logging.getLogger(__name__)


def _select_cuda_device(device: str) -> None:
    normalized = device.strip().lower()
    if normalized == "cuda":
        normalized = "cuda:0"
    if not normalized.startswith("cuda:"):
        return

    try:
        index = int(normalized.split(":", 1)[1])
    except ValueError:
        raise RuntimeError(f"Invalid Qwen worker CUDA device: {device!r}") from None

    import torch

    torch.cuda.set_device(index)
    _LOGGER.info("Qwen worker selected CUDA device=%s", normalized)


def qwen_worker_main(
    settings_data: dict[str, Any],
    request_queue: Any,
    response_queue: Any,
    worker_name: str = "primary",
) -> None:
    """Run the Qwen model in a dedicated OS process.

    The parent process owns HTTP. This worker owns Qwen model loading, CUDA graph
    state, and Qwen inference.
    """
    _select_cuda_device(str(settings_data.get("device") or ""))

    from . import qwen3

    _LOGGER.info("Qwen worker starting name=%s pid=%s", worker_name, os.getpid())
    settings = qwen3.QwenSettings(**settings_data)
    qwen3.configure(settings)
    qwen3.preload_model(
        background=False,
        include_dp_budget=settings.dp_budget.enabled,
    )
    _LOGGER.info("Qwen worker ready name=%s pid=%s", worker_name, os.getpid())

    while True:
        message = request_queue.get()
        if not isinstance(message, dict):
            response_queue.put(
                {
                    "ok": False,
                    "status_code": 400,
                    "detail": "invalid worker message",
                    "request_id": None,
                }
            )
            continue

        request_id = message.get("request_id")
        action = message.get("action")
        payload = message.get("payload")
        if action == "shutdown":
            _LOGGER.info("Qwen worker shutting down name=%s pid=%s", worker_name, os.getpid())
            return

        try:
            if action == "health":
                response_queue.put(
                    {
                        "ok": True,
                        "request_id": request_id,
                        "data": {
                            "worker": "ok",
                            "worker_name": worker_name,
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
                        "request_id": request_id,
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
                        "request_id": request_id,
                        "data": response.model_dump(mode="json"),
                    }
                )
            else:
                response_queue.put(
                    {
                        "ok": False,
                        "request_id": request_id,
                        "status_code": 400,
                        "detail": f"unknown worker action: {action}",
                    }
                )
        except Exception:
            _LOGGER.exception("Qwen worker action failed: %s", action)
            raise
