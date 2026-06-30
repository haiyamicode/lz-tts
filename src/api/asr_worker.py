"""Qwen3 ASR backend and isolated worker for reference transcription."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from .worker_common import run_worker_loop


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
    stream=sys.stdout,
)
_LOGGER = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class QwenASRSettings(BaseModel):
    """Qwen3 ASR configuration for reference voice transcription."""

    enabled: bool = Field(default_factory=lambda: _env_bool("QWEN_ASR_ENABLED", True))
    isolated: bool = Field(default_factory=lambda: _env_bool("QWEN_ASR_ISOLATED", True))
    preload: bool = Field(default_factory=lambda: _env_bool("QWEN_ASR_PRELOAD", False))
    model: str = Field(
        default_factory=lambda: os.environ.get(
            "QWEN_ASR_MODEL",
            os.environ.get("QWEN_TTS_REFERENCE_TRANSCRIPTION_MODEL", "Qwen/Qwen3-ASR-0.6B"),
        )
    )
    device: str = Field(
        default_factory=lambda: os.environ.get(
            "QWEN_ASR_DEVICE",
            os.environ.get("QWEN_TTS_REFERENCE_TRANSCRIPTION_DEVICE", "cuda"),
        )
    )
    dtype: str = Field(
        default_factory=lambda: os.environ.get(
            "QWEN_ASR_DTYPE",
            os.environ.get("QWEN_TTS_REFERENCE_TRANSCRIPTION_DTYPE", "bfloat16"),
        )
    )
    attn: str = Field(default_factory=lambda: os.environ.get("QWEN_ASR_ATTN", "sdpa"))
    max_new_tokens: int = Field(default_factory=lambda: int(os.environ.get("QWEN_ASR_MAX_NEW_TOKENS", "256")), ge=1)
    max_inference_batch_size: int = Field(
        default_factory=lambda: int(os.environ.get("QWEN_ASR_MAX_INFERENCE_BATCH_SIZE", "1")),
        ge=-1,
    )


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise RuntimeError(f"Unsupported Qwen ASR dtype: {dtype_name!r}")


class QwenASRBackend:
    """Lazy Qwen3 ASR model wrapper."""

    def __init__(self, settings: QwenASRSettings):
        self.settings = settings
        self.model: Any | None = None

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        if not self.settings.enabled:
            raise RuntimeError("Qwen ASR is disabled")

        from qwen_asr import Qwen3ASRModel

        dtype = _resolve_dtype(self.settings.dtype)
        device = self.settings.device.strip() or "cuda"
        if device == "cuda":
            device = "cuda:0"
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device,
            "max_inference_batch_size": self.settings.max_inference_batch_size,
            "max_new_tokens": self.settings.max_new_tokens,
        }
        attn = self.settings.attn.strip()
        if attn and attn.lower() != "auto":
            kwargs["attn_implementation"] = attn

        _LOGGER.info(
            "Loading Qwen ASR model: model=%s device=%s dtype=%s attn=%s max_new_tokens=%s",
            self.settings.model,
            device,
            dtype,
            attn or "auto",
            self.settings.max_new_tokens,
        )
        self.model = Qwen3ASRModel.from_pretrained(self.settings.model, **kwargs)
        _LOGGER.info("Qwen ASR model ready.")
        return self.model

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "isolated": self.settings.isolated,
            "model_loaded": self.model is not None,
            "model": self.settings.model,
            "device": self.settings.device,
            "dtype": self.settings.dtype,
            "attn": self.settings.attn,
            "max_new_tokens": self.settings.max_new_tokens,
            "max_inference_batch_size": self.settings.max_inference_batch_size,
        }

    def transcribe(self, audio_file: str | Path, language: str | None = None) -> dict[str, Any]:
        model = self.load()
        with torch.inference_mode():
            results = model.transcribe(
                audio=str(audio_file),
                language=language if language and language.strip() else None,
            )
        if not results:
            return {"text": "", "language": ""}
        result = results[0]
        return {
            "text": str(getattr(result, "text", "") or "").strip(),
            "language": str(getattr(result, "language", "") or "").strip(),
        }


def qwen_asr_worker_main(settings_data: dict[str, Any], request_queue: Any, response_queue: Any) -> None:
    """Run Qwen3 ASR in a dedicated OS process."""
    settings = QwenASRSettings(**settings_data)
    backend = QwenASRBackend(settings)
    _LOGGER.info("Qwen ASR worker starting pid=%s", os.getpid())
    if settings.preload:
        backend.load()

    def handle(action: str, payload: Any) -> dict[str, Any]:
        if action == "health":
            return {"ok": True, "data": {"worker": "ok", "worker_pid": os.getpid(), **backend.status()}}
        if action == "preload":
            backend.load()
            return {"ok": True, "data": backend.status()}
        if action == "transcribe":
            payload = payload or {}
            data = backend.transcribe(payload["audio_file"], language=payload.get("language"))
            return {"ok": True, "data": data}
        return {"ok": False, "status_code": 400, "detail": f"unknown worker action: {action}"}

    run_worker_loop("Qwen ASR", handle, request_queue, response_queue)
