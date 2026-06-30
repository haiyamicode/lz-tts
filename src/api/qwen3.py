import base64
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .asr_worker import QwenASRBackend, QwenASRSettings, qwen_asr_worker_main
from .worker_common import WorkerProcessClient

_LOGGER = logging.getLogger(__name__)

CACHE_DIR = Path("cache/voice_samples")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BINARY_RESPONSE_HEADERS = {"Content-Encoding": "identity"}


def _binary_response(content: bytes, media_type: str) -> Response:
    return Response(content=content, media_type=media_type, headers=dict(BINARY_RESPONSE_HEADERS))

QWEN_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN_DEFAULT_DTYPE = "bfloat16"
QWEN_DEFAULT_LANGUAGE = "Auto"
QWEN_DEFAULT_MAX_NEW_TOKENS = 360
QWEN_DEFAULT_TEMPERATURE = 0.80
QWEN_DEFAULT_TOP_K = 20
QWEN_DEFAULT_TOP_P = 0.95
QWEN_DEFAULT_REPETITION_PENALTY = 1.01
QWEN_DEFAULT_XVEC_ONLY = False
QWEN_DEFAULT_NON_STREAMING_MODE = True
QWEN_DEFAULT_EXPRESSIVENESS = 1.0
QWEN_DP_BUDGET_DEFAULT = True
QWEN_CONTEXT_REPLACEMENTS_ENABLED = True
QWEN_DEFAULT_VIETNAMESE_LORA_MODEL = ""
QWEN_BATCH_BUCKETS = (1, 3, 5)
QWEN_MODEL_MAX_BATCH_SIZE = max(QWEN_BATCH_BUCKETS)
QWEN_MAX_BATCH_SIZE = max(
    QWEN_MODEL_MAX_BATCH_SIZE,
    int(os.environ.get("QWEN_TTS_MAX_BATCH_SIZE", "64")),
)
QWEN_BATCH_DUMMY_TEXT = ""
ENABLE_SILENCE_TRIM_DEFAULT = True
SILERO_VAD_SAMPLE_RATE = 16000
SILERO_VAD_CHUNK_SAMPLES = 480
SILERO_VAD_THRESHOLD = 0.2
SILERO_VAD_ONNX_PROVIDERS_DEFAULT = "CUDAExecutionProvider,CPUExecutionProvider"
SILENCE_TRIM_PADDING_MS = 250
QWEN_DEFAULT_OUTPUT_BUFFER_SILENCE_MS = 100
SILENCE_TRIM_MIN_CHUNK_RMS = 0.003
SILENCE_TRIM_RELATIVE_CHUNK_RMS = 0.03
SILENCE_TRIM_EMPTY_OUTPUT_MS = 200
EXPRESSIVENESS_PRESETS = {
    0.0: {"temperature": 0.60, "top_k": 12, "top_p": 0.85, "repetition_penalty": 1.02},
    0.25: {"temperature": 0.65, "top_k": 14, "top_p": 0.88, "repetition_penalty": 1.018},
    0.5: {"temperature": 0.70, "top_k": 16, "top_p": 0.90, "repetition_penalty": 1.015},
    0.75: {"temperature": 0.75, "top_k": 18, "top_p": 0.93, "repetition_penalty": 1.012},
    1.0: {"temperature": 0.80, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.01},
    1.25: {"temperature": 0.85, "top_k": 28, "top_p": 0.97, "repetition_penalty": 1.005},
    1.5: {"temperature": 0.90, "top_k": 35, "top_p": 0.98, "repetition_penalty": 1.0},
    1.75: {"temperature": 0.95, "top_k": 45, "top_p": 0.99, "repetition_penalty": 1.0},
    2.0: {"temperature": 1.00, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.0},
}

router = APIRouter(prefix="/qwen3", tags=["qwen3"])
model: Optional[Any] = None
vietnamese_model: Optional[Any] = None
reference_transcription_model: Optional[Any] = None
reference_transcription_worker: Optional[WorkerProcessClient] = None
silero_vad_detector: Optional[Any] = None
dp_budget_model: Optional[Any] = None
inference_lock = threading.Lock()
vietnamese_model_load_lock = threading.Lock()
reference_transcription_lock = threading.Lock()
reference_transcription_worker_lock = threading.Lock()
model_load_lock = threading.Lock()
dp_budget_load_lock = threading.Lock()
silero_vad_lock = threading.Lock()
model_ready_event = threading.Event()
vietnamese_model_ready_event = threading.Event()
model_loading = False
vietnamese_model_loading = False
model_load_error: Optional[str] = None
vietnamese_model_load_error: Optional[str] = None
model_load_started_at: Optional[float] = None
model_load_finished_at: Optional[float] = None
vietnamese_model_load_started_at: Optional[float] = None
vietnamese_model_load_finished_at: Optional[float] = None
_qwen_language_splitter: Optional[Any] = None

QWEN_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    "vi": "Vietnamese",
}
QWEN_LANGUAGE_LOCALES = {
    "zh": "zh-CN",
    "en": "en-US",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "fr": "fr-FR",
    "de": "de-DE",
    "ru": "ru-RU",
    "pt": "pt-PT",
    "es": "es-ES",
    "it": "it-IT",
    "vi": "vi-VN",
}
QWEN_LANGUAGE_CODES = {
    language_name.lower(): language_name
    for language_name in QWEN_LANGUAGE_NAMES.values()
}
QWEN_NAME_TO_CODE = {
    language_name.lower(): language_code
    for language_code, language_name in QWEN_LANGUAGE_NAMES.items()
}


@dataclass(frozen=True)
class ResolvedQwenLanguage:
    qwen_language: str
    dp_language: str


def _default_qwen_language() -> ResolvedQwenLanguage:
    return ResolvedQwenLanguage("English", QWEN_LANGUAGE_LOCALES["en"])


def _is_auto_language_value(value: str | None) -> bool:
    return not value or not value.strip() or value.strip().lower() == "auto"


class DpBudgetSettings(BaseModel):
    enabled: bool = True
    preload: bool = True
    use_bert: bool = False
    checkpoint: str = "data/lzspeech-sparrow/model.ckpt"
    config_path: Optional[str] = None
    device: str = "cuda"
    language: str = "multilingual"
    noise_scale: float = 0.8
    length_scale: float = 1.0
    token_rate: float = 12.0
    samples: int = 32
    upper_quantile: float = 0.90
    min_margin: float = 1.0
    max_margin: float = 1.35
    min_extra_tokens: int = 0
    max_extra_tokens: int = 72
    language_profiles: dict[str, dict[str, float | int]] = Field(default_factory=dict)


class QwenValidationSettings(BaseModel):
    enabled: bool = True
    max_retries: int = Field(2, ge=0)
    duration_tolerance: float = Field(0.25, ge=0.0)
    reject_zero_phoneme_duration: bool = True


class QwenSettings(BaseModel):
    preload: bool = True
    preload_background: bool = True
    model: str = QWEN_DEFAULT_MODEL
    vietnamese_model: str = ""
    vietnamese_device: str = ""
    vietnamese_disable_cuda_graph: bool = False
    viet_lora_model: str = QWEN_DEFAULT_VIETNAMESE_LORA_MODEL
    vietnamese_icl_mode: bool = True
    device: str = "cuda"
    precision_mode: str = "config"
    dtype: str = QWEN_DEFAULT_DTYPE
    audio_dtype: str = "auto"
    warmup: bool = True
    attn: str = "sdpa"
    layer_precision: str = "auto"
    predictor_layer_precision: str = "auto"
    audio_decoder_precision: str = "auto"
    large_block_precision: str = "auto"
    extra_precision: str = "auto"
    linear_precision: str = "none"
    max_seq_len: int = 2048
    language: str = QWEN_DEFAULT_LANGUAGE
    max_new_tokens: int = QWEN_DEFAULT_MAX_NEW_TOKENS
    xvec_only: bool = QWEN_DEFAULT_XVEC_ONLY
    non_streaming_mode: bool = QWEN_DEFAULT_NON_STREAMING_MODE
    output_buffer_silence_ms: int = Field(QWEN_DEFAULT_OUTPUT_BUFFER_SILENCE_MS, ge=0)
    disable_cuda_graph: bool = False
    disable_cuda_graph_batch: bool = True
    temperature: float = QWEN_DEFAULT_TEMPERATURE
    top_k: int = QWEN_DEFAULT_TOP_K
    top_p: float = QWEN_DEFAULT_TOP_P
    repetition_penalty: float = QWEN_DEFAULT_REPETITION_PENALTY
    voice_prompt_cache_entries: int = Field(8, ge=0)
    asr: QwenASRSettings = Field(default_factory=QwenASRSettings)
    dp_budget: DpBudgetSettings = Field(default_factory=DpBudgetSettings)
    validation: QwenValidationSettings = Field(default_factory=QwenValidationSettings)


_qwen_settings = QwenSettings()


def apply_env_overrides(settings: QwenSettings) -> QwenSettings:
    """Apply .env overrides for Qwen runtime and precision settings."""

    def env_bool_value(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    mode = os.environ.get("QWEN_TTS_PRECISION_MODE")
    optimized = os.environ.get("QWEN_TTS_OPTIMIZED")
    if mode:
        mode = mode.strip().lower()
    elif optimized is not None:
        mode = "optimized" if env_bool_value("QWEN_TTS_OPTIMIZED", False) else "bf16"

    if mode in {"optimized", "auto", "fp16", "mixed", "stable"}:
        settings.precision_mode = "optimized"
        settings.dtype = "auto"
        settings.audio_dtype = "auto"
        settings.attn = "auto"
        settings.layer_precision = "auto"
        settings.predictor_layer_precision = "none"
        settings.audio_decoder_precision = "none"
        settings.large_block_precision = "none"
        settings.extra_precision = "none"
        settings.linear_precision = "none"
    elif mode in {"balanced"}:
        settings.precision_mode = "balanced"
        settings.dtype = "auto"
        settings.audio_dtype = "auto"
        settings.attn = "auto"
        settings.layer_precision = "auto"
        settings.predictor_layer_precision = "0,1,3,4"
        settings.audio_decoder_precision = "none"
        settings.large_block_precision = "none"
        settings.extra_precision = "none"
        settings.linear_precision = "none"
    elif mode in {"aggressive", "max", "maximum"}:
        settings.precision_mode = "aggressive"
        settings.dtype = "auto"
        settings.audio_dtype = "auto"
        settings.attn = "auto"
        settings.layer_precision = "auto"
        settings.predictor_layer_precision = "0,1,3,4"
        settings.audio_decoder_precision = "fp16"
        settings.large_block_precision = "fp16"
        settings.extra_precision = "fp16_inner"
        settings.linear_precision = "none"
    elif mode in {"bf16", "bfloat16", "off", "disabled", "original"}:
        settings.precision_mode = "bf16"
        settings.dtype = "bfloat16"
        settings.audio_dtype = "same"
        settings.attn = "sdpa"
        settings.layer_precision = "none"
        settings.predictor_layer_precision = "none"
        settings.audio_decoder_precision = "none"
        settings.large_block_precision = "none"
        settings.extra_precision = "none"
        settings.linear_precision = "none"
    elif mode:
        raise ValueError(
            "QWEN_TTS_PRECISION_MODE must be optimized, aggressive, or bf16 "
            f"(got {mode!r})"
        )

    string_overrides = {
        "QWEN_TTS_MODEL": "model",
        "QWEN_TTS_VIETNAMESE_MODEL": "vietnamese_model",
        "QWEN_TTS_VIETNAMESE_DEVICE": "vietnamese_device",
        "QWEN_TTS_VIETNAMESE_LORA_MODEL": "viet_lora_model",
        "QWEN_TTS_DEVICE": "device",
        "QWEN_TTS_DTYPE": "dtype",
        "QWEN_TTS_AUDIO_DTYPE": "audio_dtype",
        "QWEN_TTS_ATTN": "attn",
        "QWEN_TTS_LAYER_PRECISION": "layer_precision",
        "QWEN_TTS_PREDICTOR_LAYER_PRECISION": "predictor_layer_precision",
        "QWEN_TTS_AUDIO_DECODER_PRECISION": "audio_decoder_precision",
        "QWEN_TTS_LARGE_BLOCK_PRECISION": "large_block_precision",
        "QWEN_TTS_EXTRA_PRECISION": "extra_precision",
        "QWEN_TTS_LINEAR_PRECISION": "linear_precision",
        "QWEN_ASR_MODEL": "asr.model",
        "QWEN_ASR_DEVICE": "asr.device",
        "QWEN_ASR_DTYPE": "asr.dtype",
        "QWEN_ASR_ATTN": "asr.attn",
        "QWEN_TTS_REFERENCE_TRANSCRIPTION_MODEL": "asr.model",
        "QWEN_TTS_REFERENCE_TRANSCRIPTION_DEVICE": "asr.device",
        "QWEN_TTS_REFERENCE_TRANSCRIPTION_DTYPE": "asr.dtype",
    }
    for env_name, attr in string_overrides.items():
        value = os.environ.get(env_name)
        if value is not None and value.strip():
            if attr.startswith("asr."):
                setattr(settings.asr, attr.split(".", 1)[1], value.strip())
            else:
                setattr(settings, attr, value.strip())
            if attr in {
                "dtype",
                "audio_dtype",
                "attn",
                "layer_precision",
                "predictor_layer_precision",
                "audio_decoder_precision",
                "large_block_precision",
                "extra_precision",
                "linear_precision",
            }:
                settings.precision_mode = "custom"

    int_overrides = {
        "QWEN_TTS_MAX_SEQ_LEN": "max_seq_len",
        "QWEN_TTS_MAX_NEW_TOKENS": "max_new_tokens",
        "QWEN_TTS_VOICE_PROMPT_CACHE_ENTRIES": "voice_prompt_cache_entries",
        "QWEN_TTS_OUTPUT_BUFFER_SILENCE_MS": "output_buffer_silence_ms",
        "QWEN_TTS_VALIDATION_MAX_RETRIES": "validation.max_retries",
        "QWEN_ASR_MAX_NEW_TOKENS": "asr.max_new_tokens",
        "QWEN_ASR_MAX_INFERENCE_BATCH_SIZE": "asr.max_inference_batch_size",
    }
    for env_name, attr in int_overrides.items():
        value = os.environ.get(env_name)
        if value is not None and value.strip():
            if attr.startswith("validation."):
                setattr(settings.validation, attr.split(".", 1)[1], int(value.strip()))
            elif attr.startswith("asr."):
                setattr(settings.asr, attr.split(".", 1)[1], int(value.strip()))
            else:
                setattr(settings, attr, int(value.strip()))

    bool_overrides = {
        "QWEN_TTS_PRELOAD": "preload",
        "QWEN_TTS_PRELOAD_BACKGROUND": "preload_background",
        "QWEN_TTS_WARMUP": "warmup",
        "QWEN_TTS_XVEC_ONLY": "xvec_only",
        "QWEN_TTS_VIETNAMESE_ICL_MODE": "vietnamese_icl_mode",
        "QWEN_TTS_VIETNAMESE_DISABLE_CUDA_GRAPH": "vietnamese_disable_cuda_graph",
        "QWEN_TTS_NON_STREAMING_MODE": "non_streaming_mode",
        "QWEN_TTS_DISABLE_CUDA_GRAPH": "disable_cuda_graph",
        "QWEN_TTS_DISABLE_CUDA_GRAPH_BATCH": "disable_cuda_graph_batch",
        "QWEN_TTS_VALIDATION_ENABLED": "validation.enabled",
        "QWEN_TTS_VALIDATION_REJECT_ZERO_PHONEME_DURATION": "validation.reject_zero_phoneme_duration",
        "QWEN_ASR_ENABLED": "asr.enabled",
        "QWEN_ASR_ISOLATED": "asr.isolated",
        "QWEN_ASR_PRELOAD": "asr.preload",
    }
    for env_name, attr in bool_overrides.items():
        if os.environ.get(env_name) is not None:
            if attr.startswith("validation."):
                key = attr.split(".", 1)[1]
                setattr(settings.validation, key, env_bool_value(env_name, getattr(settings.validation, key)))
            elif attr.startswith("asr."):
                key = attr.split(".", 1)[1]
                setattr(settings.asr, key, env_bool_value(env_name, getattr(settings.asr, key)))
            else:
                setattr(settings, attr, env_bool_value(env_name, getattr(settings, attr)))

    float_overrides = {
        "QWEN_TTS_VALIDATION_DURATION_TOLERANCE": "validation.duration_tolerance",
    }
    for env_name, attr in float_overrides.items():
        value = os.environ.get(env_name)
        if value is not None and value.strip():
            if attr.startswith("validation."):
                setattr(settings.validation, attr.split(".", 1)[1], float(value.strip()))
            else:
                setattr(settings, attr, float(value.strip()))

    return settings


def configure(settings: QwenSettings) -> None:
    global _qwen_settings, dp_budget_model, model, vietnamese_model, reference_transcription_model
    global model_loading, vietnamese_model_loading, model_load_error, vietnamese_model_load_error
    global model_load_started_at, model_load_finished_at, vietnamese_model_load_started_at, vietnamese_model_load_finished_at
    stop_reference_transcription_worker()
    _qwen_settings = settings
    dp_budget_model = None
    model = None
    vietnamese_model = None
    reference_transcription_model = None
    model_loading = False
    vietnamese_model_loading = False
    model_load_error = None
    vietnamese_model_load_error = None
    model_load_started_at = None
    model_load_finished_at = None
    vietnamese_model_load_started_at = None
    vietnamese_model_load_finished_at = None
    model_ready_event.clear()
    vietnamese_model_ready_event.clear()


def demo_defaults() -> dict[str, Any]:
    return {
        "language": _qwen_settings.language,
        "temperature": _qwen_settings.temperature,
        "top_k": _qwen_settings.top_k,
        "top_p": _qwen_settings.top_p,
        "repetition_penalty": _qwen_settings.repetition_penalty,
        "xvec_only": _qwen_settings.xvec_only,
        "vietnamese_icl_mode": _qwen_settings.vietnamese_icl_mode,
        "non_streaming_mode": _qwen_settings.non_streaming_mode,
        "output_buffer_silence_ms": _qwen_settings.output_buffer_silence_ms,
        "validation": _qwen_settings.validation.model_dump(),
        "asr": _qwen_settings.asr.model_dump(),
        "disable_cuda_graph": _qwen_settings.disable_cuda_graph,
        "disable_cuda_graph_batch": _qwen_settings.disable_cuda_graph_batch,
        "dp_budget": _qwen_settings.dp_budget.enabled,
        "precision_mode": _qwen_settings.precision_mode,
    }


@router.get("/health")
async def healthcheck():
    return {
        "status": "ok",
        "backend": "faster-qwen3-tts",
        **model_status(),
    }


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_expressiveness(value: Optional[float]) -> tuple[float, dict[str, float | int]]:
    if value is None:
        value = QWEN_DEFAULT_EXPRESSIVENESS
    if value < 0 or value > 2:
        raise HTTPException(400, "expressiveness must be between 0 and 2")

    level = round(value * 4) / 4
    level = min(EXPRESSIVENESS_PRESETS, key=lambda preset: abs(preset - level))
    return level, EXPRESSIVENESS_PRESETS[level]


def _get_qwen_language_splitter() -> Any:
    global _qwen_language_splitter
    if _qwen_language_splitter is None:
        from src.multilingual_splitter import MultilingualSplitter

        _qwen_language_splitter = MultilingualSplitter()
    return _qwen_language_splitter


def _language_weight(text: str) -> int:
    return len(re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE))


def _canonical_locale(language_code: str) -> str:
    code = language_code.strip().replace("_", "-")
    parts = [part for part in code.split("-") if part]
    if not parts:
        return ""
    base = parts[0].lower()
    if len(parts) == 1:
        return base
    region = parts[1].upper()
    rest = parts[2:]
    return "-".join([base, region, *rest])


def _locale_base(locale: str) -> str:
    return locale.strip().lower().replace("_", "-").split("-", 1)[0]


def _qwen_name_for_code(language_code: str) -> Optional[str]:
    return QWEN_LANGUAGE_NAMES.get(_locale_base(language_code))


def detect_qwen_language(text: str) -> ResolvedQwenLanguage:
    splitter = _get_qwen_language_splitter()
    result = splitter.split(text)

    weights: dict[str, int] = {}
    locales: dict[str, str] = {}
    for segment in result.segments:
        language = (segment.language if segment.language and segment.language != "und" else result.main_language).lower()
        qwen_language = QWEN_LANGUAGE_NAMES.get(language)
        if not qwen_language:
            continue
        locales[qwen_language] = QWEN_LANGUAGE_LOCALES.get(language, language)
        weight = _language_weight(segment.text)
        if weight:
            weights[qwen_language] = weights.get(qwen_language, 0) + weight

    if not weights:
        main_code = (result.main_language or "").lower()
        main_language = QWEN_LANGUAGE_NAMES.get(main_code)
        if main_language:
            return ResolvedQwenLanguage(main_language, QWEN_LANGUAGE_LOCALES.get(main_code, main_code))
        return ResolvedQwenLanguage("Auto", "multilingual")

    total_weight = sum(weights.values())
    prominence_threshold = max(4, int(total_weight * 0.20))
    prominent = [
        language
        for language, weight in weights.items()
        if weight >= prominence_threshold
    ]
    if len(prominent) == 1:
        qwen_language = prominent[0]
        return ResolvedQwenLanguage(qwen_language, locales.get(qwen_language, "multilingual"))
    return ResolvedQwenLanguage("Auto", "multilingual")


def normalize_qwen_language(language: str) -> str:
    requested = language.strip()
    requested_lower = requested.lower()
    if requested_lower == "auto":
        return "Auto"
    if "-" in requested or "_" in requested:
        qwen_language = _qwen_name_for_code(requested)
        if qwen_language:
            return qwen_language
    return QWEN_LANGUAGE_NAMES.get(requested_lower) or QWEN_LANGUAGE_CODES.get(requested_lower) or requested


def _is_vietnamese_qwen_language(language: str) -> bool:
    return language.strip().lower() == "vietnamese"


def _qwen_model_key_for_language(language: str) -> str:
    if _is_vietnamese_qwen_language(language) and (
        _qwen_settings.vietnamese_model.strip() or _qwen_settings.viet_lora_model.strip()
    ):
        return "vietnamese"
    return "base"


def _qwen_model_label_for_key(key: str) -> str:
    if key != "vietnamese":
        return "base"
    return "Vietnamese full model" if _qwen_settings.vietnamese_model.strip() else "Vietnamese LoRA"


def _qwen_model_context(model_obj: Any, use_vietnamese_lora: bool):
    if use_vietnamese_lora:
        return nullcontext()
    adapter_owner = getattr(getattr(model_obj, "model", None), "model", None)
    adapter = getattr(adapter_owner, "disable_adapter", None)
    if callable(adapter):
        return adapter()
    return nullcontext()


def resolve_qwen_language_code(language_code: str) -> ResolvedQwenLanguage:
    requested = language_code.strip()
    if _is_auto_language_value(requested):
        return _default_qwen_language()

    locale = _canonical_locale(requested)
    if "-" not in locale:
        raise HTTPException(400, "language_code must be a full locale code like ja-JP or en-US")

    qwen_language = _qwen_name_for_code(locale)
    if qwen_language is None:
        raise HTTPException(400, f"Unsupported Qwen language_code: {language_code}")

    return ResolvedQwenLanguage(qwen_language, locale)


def resolve_qwen_language(
    text: str,
    language: Optional[str],
    language_code: Optional[str] = None,
) -> ResolvedQwenLanguage:
    if language_code is not None and language_code.strip() and not _is_auto_language_value(language_code):
        return resolve_qwen_language_code(language_code)

    requested = (language or _qwen_settings.language).strip()
    if _is_auto_language_value(requested):
        return detect_qwen_language(text)

    qwen_language = normalize_qwen_language(requested)
    language_base = QWEN_NAME_TO_CODE.get(qwen_language.lower()) or _locale_base(requested)
    dp_language = QWEN_LANGUAGE_LOCALES.get(language_base, "multilingual")
    return ResolvedQwenLanguage(qwen_language, dp_language)


def model_status() -> dict[str, Any]:
    return {
        "model_loaded": model is not None,
        "vietnamese_model_loaded": (
            vietnamese_model is not None
            if _qwen_settings.vietnamese_model.strip()
            else model is not None and bool(_qwen_settings.viet_lora_model.strip())
        ),
        "model_loading": model_loading,
        "vietnamese_model_loading": vietnamese_model_loading,
        "model_load_error": model_load_error,
        "vietnamese_model_load_error": vietnamese_model_load_error,
        "model_load_started_at": model_load_started_at,
        "model_load_finished_at": model_load_finished_at,
        "vietnamese_model_load_started_at": vietnamese_model_load_started_at,
        "vietnamese_model_load_finished_at": vietnamese_model_load_finished_at,
        "precision_mode": _qwen_settings.precision_mode,
        "model": _qwen_settings.model,
        "vietnamese_model": _qwen_settings.vietnamese_model,
        "vietnamese_device": _qwen_settings.vietnamese_device,
        "vietnamese_disable_cuda_graph": _qwen_settings.vietnamese_disable_cuda_graph,
        "viet_lora_model": _qwen_settings.viet_lora_model,
        "vietnamese_icl_mode": _qwen_settings.vietnamese_icl_mode,
        "output_buffer_silence_ms": _qwen_settings.output_buffer_silence_ms,
        "validation": _qwen_settings.validation.model_dump(),
        "asr": reference_transcription_status(),
        "dtype": _qwen_settings.dtype,
        "audio_dtype": _qwen_settings.audio_dtype,
        "attn": _qwen_settings.attn,
        "layer_precision": _qwen_settings.layer_precision,
        "predictor_layer_precision": _qwen_settings.predictor_layer_precision,
        "audio_decoder_precision": _qwen_settings.audio_decoder_precision,
        "large_block_precision": _qwen_settings.large_block_precision,
        "extra_precision": _qwen_settings.extra_precision,
        "linear_precision": _qwen_settings.linear_precision,
        "disable_cuda_graph": _qwen_settings.disable_cuda_graph,
        "disable_cuda_graph_batch": _qwen_settings.disable_cuda_graph_batch,
    }


def _load_model_unlocked(
    model_name: str,
    *,
    device: str | None = None,
    capture_cuda_graph: bool | None = None,
) -> Any:
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    device = device or _qwen_settings.device
    capture_cuda_graph = (
        not _qwen_settings.disable_cuda_graph
        if capture_cuda_graph is None
        else capture_cuda_graph
    )
    dtype_name = _qwen_settings.dtype
    if dtype_name.lower() == "auto":
        dtype = "auto"
    else:
        dtype = getattr(torch, dtype_name, torch.bfloat16)
    audio_dtype = _qwen_settings.audio_dtype
    attn_implementation = _qwen_settings.attn
    max_seq_len = _qwen_settings.max_seq_len
    _LOGGER.info(
        "Loading FasterQwen3TTS precision_mode=%s model=%s device=%s dtype=%s "
        "audio_dtype=%s attn=%s layer_precision=%s predictor_layer_precision=%s "
        "audio_decoder_precision=%s large_block_precision=%s extra_precision=%s "
        "linear_precision=%s max_seq_len=%s",
        _qwen_settings.precision_mode,
        model_name,
        device,
        dtype_name,
        audio_dtype,
        attn_implementation,
        _qwen_settings.layer_precision,
        _qwen_settings.predictor_layer_precision,
        _qwen_settings.audio_decoder_precision,
        _qwen_settings.large_block_precision,
        _qwen_settings.extra_precision,
        _qwen_settings.linear_precision,
        max_seq_len,
    )
    loaded_model = FasterQwen3TTS.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        audio_dtype=audio_dtype,
        layer_precision=_qwen_settings.layer_precision,
        predictor_layer_precision=_qwen_settings.predictor_layer_precision,
        audio_decoder_precision=_qwen_settings.audio_decoder_precision,
        large_block_precision=_qwen_settings.large_block_precision,
        extra_precision=_qwen_settings.extra_precision,
        linear_precision=_qwen_settings.linear_precision,
        attn_implementation=attn_implementation,
        max_seq_len=max_seq_len,
    )
    if capture_cuda_graph and _qwen_settings.warmup and hasattr(loaded_model, "_warmup"):
        _LOGGER.info("Capturing CUDA graphs...")
        loaded_model._warmup(prefill_len=100)
        if not _qwen_settings.disable_cuda_graph_batch and hasattr(loaded_model, "capture_batch_graphs"):
            _LOGGER.info("Capturing Qwen batch CUDA graph buckets: %s...", QWEN_BATCH_BUCKETS)
            loaded_model.capture_batch_graphs(QWEN_BATCH_BUCKETS, prefill_len=100)
    elif not capture_cuda_graph:
        _LOGGER.info("CUDA graph capture disabled by configuration; using eager/dynamic-cache path.")
    if hasattr(loaded_model, "max_voice_prompt_cache_entries"):
        loaded_model.max_voice_prompt_cache_entries = _qwen_settings.voice_prompt_cache_entries
    _LOGGER.info("FasterQwen3TTS loaded. Sample rate: %s", loaded_model.sample_rate)
    return loaded_model


def _load_base_model_unlocked() -> Any:
    loaded_model = _load_model_unlocked(_qwen_settings.model)
    viet_lora_model = _qwen_settings.viet_lora_model.strip()
    if viet_lora_model:
        adapter_path = Path(viet_lora_model)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Vietnamese LoRA checkpoint not found: {adapter_path}")
        _LOGGER.info("Attaching Vietnamese LoRA adapter checkpoint: %s", viet_lora_model)
        from peft import PeftModel

        loaded_model.model.model = PeftModel.from_pretrained(
            loaded_model.model.model,
            str(adapter_path),
            is_trainable=False,
        )
        loaded_model.model.model.eval()
        rebuild_graph_wrappers = getattr(loaded_model, "rebuild_graph_wrappers", None)
        if callable(rebuild_graph_wrappers):
            rebuild_graph_wrappers()
        active_adapters = getattr(loaded_model.model.model, "active_adapters", None)
        if callable(active_adapters):
            active_adapters = active_adapters()
        _LOGGER.info(
            "Vietnamese LoRA adapter attached: model_type=%s active_adapters=%s",
            type(loaded_model.model.model).__name__,
            active_adapters or getattr(loaded_model.model.model, "active_adapter", None),
        )
    return loaded_model


def _load_vietnamese_model_unlocked() -> Any:
    vietnamese_model_path = _qwen_settings.vietnamese_model.strip()
    if not vietnamese_model_path:
        return get_model()
    checkpoint_path = Path(vietnamese_model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Vietnamese Qwen checkpoint not found: {checkpoint_path}")
    _LOGGER.info("Loading Vietnamese Qwen full checkpoint: %s", vietnamese_model_path)
    vietnamese_device = _qwen_settings.vietnamese_device.strip() or None
    return _load_model_unlocked(
        vietnamese_model_path,
        device=vietnamese_device,
        capture_cuda_graph=not _qwen_settings.vietnamese_disable_cuda_graph,
    )


def get_model() -> Any:
    global model, model_loading, model_load_error, model_load_started_at, model_load_finished_at
    if model is not None:
        return model

    with model_load_lock:
        if model is not None:
            return model
        model_loading = True
        model_load_error = None
        model_load_started_at = time.time()
        model_load_finished_at = None
        model_ready_event.clear()
        try:
            model = _load_base_model_unlocked()
            model_load_error = None
            return model
        except Exception as e:
            model_load_error = str(e)
            raise
        finally:
            model_loading = False
            model_load_finished_at = time.time()
            model_ready_event.set()


def get_vietnamese_model() -> Any | None:
    global vietnamese_model, vietnamese_model_loading, vietnamese_model_load_error
    global vietnamese_model_load_started_at, vietnamese_model_load_finished_at

    if not _qwen_settings.vietnamese_model.strip():
        return get_model()
    if vietnamese_model is not None:
        return vietnamese_model

    with vietnamese_model_load_lock:
        if vietnamese_model is not None:
            return vietnamese_model
        vietnamese_model_loading = True
        vietnamese_model_load_error = None
        vietnamese_model_load_started_at = time.time()
        vietnamese_model_load_finished_at = None
        vietnamese_model_ready_event.clear()
        try:
            vietnamese_model = _load_vietnamese_model_unlocked()
            vietnamese_model_load_error = None
            return vietnamese_model
        except Exception as e:
            vietnamese_model_load_error = str(e)
            raise
        finally:
            vietnamese_model_loading = False
            vietnamese_model_load_finished_at = time.time()
            vietnamese_model_ready_event.set()


def get_qwen_model_for_language(language: str) -> Any:
    if _qwen_model_key_for_language(language) == "vietnamese":
        viet_model = get_vietnamese_model()
        if viet_model is not None:
            return viet_model
    return get_model()


def _apply_context_replacements(text: str, dp_language: str) -> str:
    """Apply context-aware text replacements (e.g. AI -> ây ai) if enabled."""
    if not QWEN_CONTEXT_REPLACEMENTS_ENABLED:
        return text
    from ..piper.context_replacer import get_replacer
    replacer = get_replacer()
    return replacer.apply_replacements(text, language=dp_language)


def _apply_context_replacements_batch(texts: list[str], dp_languages: list[str]) -> list[str]:
    """Apply context-aware text replacements for a batch of texts."""
    if not QWEN_CONTEXT_REPLACEMENTS_ENABLED:
        return texts
    from ..piper.context_replacer import get_replacer
    replacer = get_replacer()
    unique_langs = set(dp_languages)
    results = list(texts)
    for lang in unique_langs:
        indices = [i for i, l in enumerate(dp_languages) if l == lang]
        lang_texts = [texts[i] for i in indices]
        replaced = replacer.apply_replacements_many(lang_texts, language=lang)
        for i, new_text in zip(indices, replaced):
            results[i] = new_text
    return results


def _normalize_qwen_text(text: str, dp_language: str) -> str:
    from ..text_norm import normalize_text

    return normalize_text(text, dp_language)


def _prepare_qwen_texts_batch(texts: list[str], dp_languages: list[str]) -> list[str]:
    normalized = [
        _normalize_qwen_text(text, dp_language)
        for text, dp_language in zip(texts, dp_languages)
    ]
    return _apply_context_replacements_batch(normalized, dp_languages)


def _preload_worker(include_dp_budget: bool) -> None:
    try:
        get_model()
        if _qwen_settings.vietnamese_model.strip():
            get_vietnamese_model()
        if include_dp_budget:
            get_dp_budget_model()
    except Exception:
        _LOGGER.exception("Qwen3 preload failed")


def start_preload_background(include_dp_budget: bool = False) -> None:
    if model is not None or model_loading:
        return
    thread = threading.Thread(
        target=_preload_worker,
        args=(include_dp_budget,),
        name="qwen3-preload",
        daemon=True,
    )
    thread.start()


def preload_model(background: bool = False, include_dp_budget: bool = False) -> None:
    if background:
        start_preload_background(include_dp_budget=include_dp_budget)
        return
    get_model()
    if _qwen_settings.vietnamese_model.strip():
        get_vietnamese_model()
    if include_dp_budget:
        get_dp_budget_model()


def _wait_for_qwen_model_load_before_dp_budget() -> None:
    if model is not None or not model_loading:
        return
    model_ready_event.wait()
    if model_load_error:
        raise RuntimeError(f"Qwen3 model load failed before DP budget load: {model_load_error}")


def get_dp_budget_model() -> Any:
    global dp_budget_model
    if dp_budget_model is not None:
        return dp_budget_model

    _wait_for_qwen_model_load_before_dp_budget()

    with dp_budget_load_lock:
        if dp_budget_model is not None:
            return dp_budget_model

        from src.qwen_dp_budget import DpBudgetConfig, QwenDpBudget

        _LOGGER.info("Loading Qwen DP budget model...")
        dp_settings = _qwen_settings.dp_budget
        loaded_dp_budget_model = QwenDpBudget(
            DpBudgetConfig(
                checkpoint=Path(dp_settings.checkpoint),
                config_path=Path(dp_settings.config_path) if dp_settings.config_path else None,
                device=dp_settings.device,
                language=dp_settings.language,
                noise_scale=dp_settings.noise_scale,
                length_scale=dp_settings.length_scale,
                token_rate=dp_settings.token_rate,
                samples=dp_settings.samples,
                upper_quantile=dp_settings.upper_quantile,
                min_margin=dp_settings.min_margin,
                max_margin=dp_settings.max_margin,
                min_extra_tokens=dp_settings.min_extra_tokens,
                max_extra_tokens=dp_settings.max_extra_tokens,
                language_profiles=dp_settings.language_profiles,
                use_bert=dp_settings.use_bert,
                enable_alignment_validation=_qwen_settings.validation.enabled,
            )
        )
        loaded_dp_budget_model.load()
        dp_budget_model = loaded_dp_budget_model
        _LOGGER.info("Qwen DP budget model ready.")
    return dp_budget_model


def predict_dp_budget(text: str, language: Optional[str] = None) -> dict[str, Any]:
    return get_dp_budget_model().predict(text, language=language)


def predict_dp_budget_batch(
    texts: list[str],
    languages: Optional[list[str | None]] = None,
) -> list[dict[str, Any]]:
    return get_dp_budget_model().predict_batch(texts, languages=languages)


class QwenValidationError(RuntimeError):
    def __init__(self, info: dict[str, Any]):
        super().__init__(str(info.get("reason") or "validation_failed"))
        self.info = info


def _ensure_reference_transcription_worker() -> WorkerProcessClient:
    global reference_transcription_worker
    with reference_transcription_worker_lock:
        if reference_transcription_worker is None:
            reference_transcription_worker = WorkerProcessClient(
                name="qwen-asr",
                target=qwen_asr_worker_main,
                args=(_qwen_settings.asr.model_dump(mode="json"),),
            )
        return reference_transcription_worker


def stop_reference_transcription_worker() -> None:
    global reference_transcription_worker
    with reference_transcription_worker_lock:
        if reference_transcription_worker is not None:
            reference_transcription_worker.stop()
            reference_transcription_worker = None


def reference_transcription_status() -> dict[str, Any]:
    status = {
        **_qwen_settings.asr.model_dump(mode="json"),
        "model_loaded": reference_transcription_model is not None,
        "worker_started": reference_transcription_worker is not None,
    }
    return status


def get_reference_transcription_model() -> QwenASRBackend:
    global reference_transcription_model

    if not _qwen_settings.asr.enabled:
        raise RuntimeError("Qwen ASR is disabled")
    if reference_transcription_model is None:
        reference_transcription_model = QwenASRBackend(_qwen_settings.asr)
    assert isinstance(reference_transcription_model, QwenASRBackend)
    return reference_transcription_model


def preload_reference_transcription_model() -> None:
    if not _qwen_settings.asr.enabled:
        return
    if _qwen_settings.asr.isolated:
        _ensure_reference_transcription_worker().call("preload")
    else:
        get_reference_transcription_model().load()


def get_silero_vad_detector() -> Any:
    global silero_vad_detector
    if silero_vad_detector is None:
        with silero_vad_lock:
            if silero_vad_detector is None:
                from ..piper.norm_audio import make_silence_detector

                providers = [
                    provider.strip()
                    for provider in os.environ.get(
                        "SILERO_VAD_ONNX_PROVIDERS",
                        SILERO_VAD_ONNX_PROVIDERS_DEFAULT,
                    ).split(",")
                    if provider.strip()
                ]
                _LOGGER.info("Loading Silero VAD postprocessor...")
                silero_vad_detector = make_silence_detector(providers=providers)
                _LOGGER.info("Silero VAD ready. Providers: %s", silero_vad_detector.providers)
    return silero_vad_detector


def get_cache_dir(url: str) -> Path:
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    d = CACHE_DIR / url_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


MAX_AUDIO_BYTES = 10 * 1024 * 1024
ASSEMBLYAI_TRANSCRIPT_TIMEOUT_SECONDS = int(os.environ.get("ASSEMBLYAI_TRANSCRIPT_TIMEOUT_SECONDS", "120"))
ASSEMBLYAI_TRANSCRIPT_POLL_SECONDS = float(os.environ.get("ASSEMBLYAI_TRANSCRIPT_POLL_SECONDS", "2"))
INCOMPLETE_PROMPT_ENDINGS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "because",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "including",
    "into",
    "like",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "when",
    "where",
    "which",
    "while",
    "with",
    "without",
}


class EmptyVoiceTranscriptError(RuntimeError):
    """Raised when transcription succeeds but produces no usable prompt text."""


def download_and_cache(url: str) -> Path:
    d = get_cache_dir(url)
    metadata_file = d / "source.json"
    if metadata_file.exists():
        try:
            cached_name = json.loads(metadata_file.read_text(encoding="utf-8")).get("file")
            if cached_name:
                audio_file = d / cached_name
                if audio_file.exists():
                    return audio_file
        except Exception:
            pass

    audio_file = d / "reference_audio"
    if audio_file.exists():
        return audio_file

    import urllib.request
    from urllib.parse import urlparse

    ext = Path(urlparse(url).path).suffix or ".wav"
    raw_file = d / f"raw{ext}"

    req = urllib.request.Request(url, headers={"User-Agent": "vc-temp/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(raw_file, "wb") as f:
        f.write(resp.read())

    if raw_file.stat().st_size > MAX_AUDIO_BYTES:
        raw_file.unlink(missing_ok=True)
        raise RuntimeError(f"voice sample is too large; max is {MAX_AUDIO_BYTES / 1024 / 1024:.1f} MB")

    audio_file = d / f"reference_audio{ext}"
    raw_file.replace(audio_file)
    metadata_file.write_text(json.dumps({"file": audio_file.name}), encoding="utf-8")

    return audio_file


def assemblyai_json_request(url: str, api_key: str, payload: Optional[dict] = None, timeout: int = 30) -> dict:
    import urllib.request
    import urllib.error

    data = None
    headers = {"Authorization": api_key}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"AssemblyAI request failed: HTTP {e.code}: {body}") from e


def upload_to_assemblyai(audio_file: Path, api_key: str) -> str:
    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        "https://api.assemblyai.com/v2/upload",
        data=audio_file.read_bytes(),
        headers={
            "Authorization": api_key,
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"AssemblyAI upload failed: HTTP {e.code}: {body}") from e

    upload_url = result.get("upload_url")
    if not upload_url:
        raise RuntimeError("AssemblyAI upload did not return upload_url")
    return upload_url


def sanitize_prompt_text(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return text

    words = re.findall(r"[A-Za-z']+", text)
    if not words or words[-1].lower() not in INCOMPLETE_PROMPT_ENDINGS:
        return text

    sentence_ends = list(re.finditer(r"[.!?。！？]", text))
    if len(sentence_ends) >= 2:
        return text[:sentence_ends[-2].end()].strip()

    comma_index = max(text.rfind(","), text.rfind("，"), text.rfind("、"))
    if comma_index > 0:
        return text[:comma_index].strip()

    return text


def _qwen_asr_language_for_locale(locale: str | None) -> str | None:
    if not locale or locale == "multilingual":
        return None
    base = _locale_base(locale)
    return QWEN_LANGUAGE_NAMES.get(base)


def transcribe_voice_sample(audio_file: Path, language: str | None = None) -> str:
    transcript_file = audio_file.with_suffix(".txt")
    if transcript_file.exists():
        transcript = transcript_file.read_text(encoding="utf-8").strip()
        if transcript:
            return transcript

    if not _qwen_settings.asr.enabled:
        raise RuntimeError("Qwen ASR is disabled")
    asr_language = _qwen_asr_language_for_locale(language)
    if _qwen_settings.asr.isolated:
        response = _ensure_reference_transcription_worker().call(
            "transcribe",
            {"audio_file": str(audio_file), "language": asr_language},
        )
        result = response.get("data") or {}
    else:
        with reference_transcription_lock:
            result = get_reference_transcription_model().transcribe(audio_file, language=asr_language)
    text = str(result.get("text") or "").strip()
    if not text:
        try:
            audio_info = sf.info(audio_file)
            debug = (
                f"path={audio_file} format={audio_info.format} subtype={audio_info.subtype} "
                f"samplerate={audio_info.samplerate} channels={audio_info.channels} "
                f"frames={audio_info.frames} duration={audio_info.duration:.3f}s "
                f"size_bytes={audio_file.stat().st_size}"
            )
        except Exception as e:
            debug = f"path={audio_file} size_bytes={audio_file.stat().st_size} info_error={e}"
        raise EmptyVoiceTranscriptError(f"Qwen ASR transcript completed with empty text ({debug})")
    transcript_file.write_text(text, encoding="utf-8")
    _LOGGER.info(
        "Reference audio transcribed with Qwen ASR: path=%s language=%s detected_language=%s text=%s",
        audio_file,
        asr_language,
        result.get("language"),
        text,
    )
    return text


def _silero_vad_bounds(wav_data: np.ndarray, sample_rate: int) -> tuple[int, int] | None:
    import torchaudio

    audio = np.asarray(wav_data, dtype=np.float32).flatten()
    if audio.size == 0:
        return None

    detector_audio = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != SILERO_VAD_SAMPLE_RATE:
        detector_audio = torchaudio.functional.resample(
            detector_audio,
            sample_rate,
            SILERO_VAD_SAMPLE_RATE,
        )
    detector_audio_np = detector_audio.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    if detector_audio_np.size == 0:
        return None

    first_chunk: int | None = None
    last_chunk: int | None = None
    chunk_count = int(np.ceil(detector_audio_np.size / SILERO_VAD_CHUNK_SAMPLES))
    detector = get_silero_vad_detector()
    chunks: list[np.ndarray] = []
    chunk_rms: list[float] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * SILERO_VAD_CHUNK_SAMPLES
        chunk = detector_audio_np[start : start + SILERO_VAD_CHUNK_SAMPLES]
        if chunk.size < SILERO_VAD_CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, SILERO_VAD_CHUNK_SAMPLES - chunk.size))
        chunks.append(chunk)
        chunk_rms.append(float(np.sqrt(np.mean(np.square(chunk)))))

    if not chunks:
        return None

    energy_threshold = max(
        SILENCE_TRIM_MIN_CHUNK_RMS,
        max(chunk_rms) * SILENCE_TRIM_RELATIVE_CHUNK_RMS,
    )

    with silero_vad_lock:
        if hasattr(detector, "reset"):
            detector.reset()
        for chunk_index, chunk in enumerate(chunks):
            prob = float(np.asarray(detector(chunk, sample_rate=SILERO_VAD_SAMPLE_RATE)).reshape(-1)[0])
            if prob >= SILERO_VAD_THRESHOLD and chunk_rms[chunk_index] >= energy_threshold:
                if first_chunk is None:
                    first_chunk = chunk_index
                last_chunk = chunk_index

    if first_chunk is None or last_chunk is None:
        return None

    vad_start = first_chunk * SILERO_VAD_CHUNK_SAMPLES
    vad_end = min(detector_audio_np.size, (last_chunk + 1) * SILERO_VAD_CHUNK_SAMPLES)
    start = int(np.floor(vad_start * sample_rate / SILERO_VAD_SAMPLE_RATE))
    end = int(np.ceil(vad_end * sample_rate / SILERO_VAD_SAMPLE_RATE))
    return max(0, start), min(audio.size, max(start + 1, end))


def trim_silence(wav_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float, float]:
    bounds = _silero_vad_bounds(wav_data, sample_rate)
    if bounds is None:
        empty_samples = max(1, int(sample_rate * SILENCE_TRIM_EMPTY_OUTPUT_MS / 1000))
        empty_audio = np.zeros(empty_samples, dtype=np.float32)
        trimmed_tail_seconds = max(0.0, (wav_data.size - empty_samples) / sample_rate)
        return empty_audio, 0.0, trimmed_tail_seconds

    voiced_start, voiced_end = bounds
    pad = int(sample_rate * SILENCE_TRIM_PADDING_MS / 1000)
    start = max(0, voiced_start - pad)
    end = min(wav_data.size, voiced_end + pad)
    trimmed = wav_data[start:end]
    return trimmed, start / sample_rate, (wav_data.size - end) / sample_rate


def _ensure_output_buffer_silence(wav_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    buffer_ms = max(0, int(_qwen_settings.output_buffer_silence_ms))
    buffer_samples = int(sample_rate * buffer_ms / 1000)
    if buffer_samples <= 0 or wav_data.size == 0:
        return wav_data, 0.0
    silence = np.zeros(buffer_samples, dtype=np.float32)
    return np.concatenate([silence, wav_data.astype(np.float32, copy=False), silence]), buffer_ms / 1000


def postprocess_audio(wav_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": False,
        "vad": "silero",
        "trim": False,
        "trim_head_seconds": 0.0,
        "trim_tail_seconds": 0.0,
        "buffer_silence_seconds": 0.0,
    }

    if env_bool("ENABLE_SILENCE_TRIM", ENABLE_SILENCE_TRIM_DEFAULT):
        wav_data, head_seconds, tail_seconds = trim_silence(wav_data, sample_rate)
        info.update(
            {
                "enabled": bool(head_seconds or tail_seconds),
                "trim": bool(head_seconds or tail_seconds),
                "trim_head_seconds": head_seconds,
                "trim_tail_seconds": tail_seconds,
            }
        )

    wav_data, buffer_seconds = _ensure_output_buffer_silence(wav_data, sample_rate)
    info["buffer_silence_seconds"] = buffer_seconds

    return wav_data, info


def wav_to_mp3(wav_data: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav_data, sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)

    proc = subprocess.run(
        [
            "ffmpeg",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "0",
            "-b:a",
            "320k",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=buf.read(),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    return proc.stdout


def _encoded_mp3_duration_seconds(mp3_bytes: bytes, fallback: float) -> float:
    try:
        from pydub import AudioSegment

        return float(AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3").duration_seconds)
    except Exception:
        _LOGGER.warning("Failed to read encoded Qwen MP3 duration; using sample-count fallback", exc_info=True)
        return fallback


class SynthesizeRequest(BaseModel):
    text: str
    voice_url: str
    voice_text: Optional[str] = None
    speed: float = 1.0
    language: Optional[str] = None
    language_code: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    xvec_only: Optional[bool] = None
    non_streaming_mode: Optional[bool] = None
    expressiveness: Optional[float] = None
    dp_budget: Optional[bool] = None
    validation_enabled: Optional[bool] = None


class BatchSynthesizeRequest(BaseModel):
    items: list[SynthesizeRequest] = Field(..., min_length=1, max_length=QWEN_MAX_BATCH_SIZE)


class BatchSynthesizeItem(BaseModel):
    text: str
    audio_base64: str
    sample_rate: int
    raw_audio_seconds: float
    audio_seconds: float
    max_new_tokens: int
    hit_token_cap: bool
    language: str
    dp_language: str


class BatchSynthesizeResponse(BaseModel):
    items: list[BatchSynthesizeItem]
    count: int
    wall_seconds: float
    audio_seconds_total: float


def _is_qwen_batch_dummy(req: SynthesizeRequest) -> bool:
    return req.max_new_tokens == 0 and not req.text.strip()


def _log_qwen_synthesize_request(
    *,
    status: str,
    started: float,
    req: SynthesizeRequest,
    settings: Any | None = None,
    prompt_text: str | None = None,
    xvec_only: bool | None = None,
    info: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    log_data: dict[str, Any] = {
        "route": "/qwen3/synthesize",
        "method": "POST",
        "status": status,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "backend": "faster-qwen3-tts",
        "input": req.model_dump(),
        "xvec_only": xvec_only,
        "prompt_text": prompt_text,
    }
    if settings is not None:
        log_data["resolved_settings"] = {
            "language": settings.language,
            "dp_language": settings.dp_language,
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_k": settings.top_k,
            "top_p": settings.top_p,
            "repetition_penalty": settings.repetition_penalty,
            "xvec_only": xvec_only,
            "non_streaming_mode": settings.non_streaming_mode,
            "expressiveness_level": settings.expressiveness_level,
            "dp_budget_enabled": settings.dp_budget_enabled,
            "dp_budget_info": settings.dp_budget_info,
            "validation_enabled": settings.validation_enabled,
        }
    if info is not None:
        log_data["result"] = info
    if error is not None:
        log_data["error"] = error
    _LOGGER.info("Qwen3 synthesize request: %s", json.dumps(log_data, ensure_ascii=False))


def _qwen_resolved_settings_log(
    settings: "_ResolvedGenerationSettings",
    *,
    xvec_only: bool | None = None,
) -> dict[str, Any]:
    return {
        "language": settings.language,
        "dp_language": settings.dp_language,
        "prepared_text": settings.prepared_text,
        "prepared_text_chars": len(settings.prepared_text),
        "max_new_tokens": settings.max_new_tokens,
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "top_p": settings.top_p,
        "repetition_penalty": settings.repetition_penalty,
        "xvec_only": xvec_only,
        "non_streaming_mode": settings.non_streaming_mode,
        "expressiveness_level": settings.expressiveness_level,
        "dp_budget_enabled": settings.dp_budget_enabled,
        "dp_budget_info": settings.dp_budget_info,
        "validation_enabled": settings.validation_enabled,
    }


def _log_qwen_batch_synthesize_request(
    *,
    status: str,
    started: float,
    req: BatchSynthesizeRequest,
    item_requests: list[SynthesizeRequest] | None = None,
    settings_list: list["_ResolvedGenerationSettings"] | None = None,
    resolved_prompts: list[tuple[Path, str, bool]] | None = None,
    info: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    items = item_requests if item_requests is not None else req.items
    log_data: dict[str, Any] = {
        "route": "/qwen3/synthesize/batch",
        "method": "POST",
        "status": status,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "backend": "faster-qwen3-tts",
        "count": len(items),
        "input": [item.model_dump() for item in items],
    }
    if settings_list is not None:
        log_data["resolved_settings"] = [
            _qwen_resolved_settings_log(
                settings,
                xvec_only=resolved_prompts[index][2] if resolved_prompts is not None else None,
            )
            for index, settings in enumerate(settings_list)
        ]
    if info is not None:
        log_data["result"] = info
    if error is not None:
        log_data["error"] = error
    _LOGGER.info("Qwen3 batch synthesize request: %s", json.dumps(log_data, ensure_ascii=False))


@dataclass(frozen=True)
class _ResolvedGenerationSettings:
    language: str
    dp_language: str
    prepared_text: str
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    non_streaming_mode: bool
    dp_budget_enabled: bool
    dp_budget_info: Optional[dict[str, Any]]
    validation_enabled: bool
    expressiveness_level: float


def _resolve_generation_settings(req: SynthesizeRequest) -> _ResolvedGenerationSettings:
    return _resolve_generation_settings_batch([req])[0]


def _resolve_generation_settings_batch(
    reqs: list[SynthesizeRequest],
) -> list[_ResolvedGenerationSettings]:
    if not reqs:
        return []

    dummy_mask = [_is_qwen_batch_dummy(req) for req in reqs]
    resolved_languages = [
        ResolvedQwenLanguage("English", "en-US")
        if dummy_mask[index]
        else resolve_qwen_language(req.text, req.language, req.language_code)
        for index, req in enumerate(reqs)
    ]
    prepared_texts = _prepare_qwen_texts_batch(
        [req.text for req in reqs],
        [resolved_language.dp_language for resolved_language in resolved_languages],
    )
    dp_budget_indices = [
        index
        for index, req in enumerate(reqs)
        if not dummy_mask[index]
        and req.max_new_tokens is None
        and (req.dp_budget if req.dp_budget is not None else _qwen_settings.dp_budget.enabled)
    ]
    dp_budget_language_inputs: list[str] = []
    dp_budget_texts: list[str] = []
    for index in dp_budget_indices:
        dp_budget_texts.append(prepared_texts[index])
        dp_budget_language_inputs.append(resolved_languages[index].dp_language)
    dp_budget_info_list: list[dict[str, Any] | None] = [None] * len(reqs)
    if dp_budget_indices:
        try:
            budgets = predict_dp_budget_batch(dp_budget_texts, languages=dp_budget_language_inputs)
        except Exception as e:
            raise HTTPException(500, f"DP budget failed: {e}") from e
        for request_index, budget_info in zip(dp_budget_indices, budgets):
            dp_budget_info_list[request_index] = budget_info

    settings_list: list[_ResolvedGenerationSettings] = []
    for index, req in enumerate(reqs):
        resolved_language = resolved_languages[index]
        language = resolved_language.qwen_language
        expressiveness_level, expressiveness_config = resolve_expressiveness(req.expressiveness)
        dp_budget_enabled = (
            req.dp_budget
            if req.dp_budget is not None
            else _qwen_settings.dp_budget.enabled
        )
        validation_enabled = (
            req.validation_enabled
            if req.validation_enabled is not None
            else _qwen_settings.validation.enabled
        )
        dp_budget_info = dp_budget_info_list[index]
        if dummy_mask[index]:
            max_new_tokens = 0
            dp_budget_enabled = False
        elif req.max_new_tokens is not None:
            max_new_tokens = req.max_new_tokens
        elif dp_budget_enabled:
            max_new_tokens = int(dp_budget_info["max_tokens"])  # type: ignore[index]
        else:
            max_new_tokens = _qwen_settings.max_new_tokens
        temperature = (
            req.temperature
            if req.temperature is not None
            else expressiveness_config["temperature"]
            if req.expressiveness is not None
            else _qwen_settings.temperature
        )
        top_k = (
            req.top_k
            if req.top_k is not None
            else expressiveness_config["top_k"]
            if req.expressiveness is not None
            else _qwen_settings.top_k
        )
        top_p = (
            req.top_p
            if req.top_p is not None
            else expressiveness_config["top_p"]
            if req.expressiveness is not None
            else _qwen_settings.top_p
        )
        repetition_penalty = (
            req.repetition_penalty
            if req.repetition_penalty is not None
            else expressiveness_config["repetition_penalty"]
            if req.expressiveness is not None
            else _qwen_settings.repetition_penalty
        )
        non_streaming_mode = (
            req.non_streaming_mode
            if req.non_streaming_mode is not None
            else _qwen_settings.non_streaming_mode
        )
        settings_list.append(
            _ResolvedGenerationSettings(
                language=language,
                dp_language=resolved_language.dp_language,
                prepared_text=prepared_texts[index],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                non_streaming_mode=non_streaming_mode,
                dp_budget_enabled=dp_budget_enabled,
                dp_budget_info=dp_budget_info,
                validation_enabled=validation_enabled,
                expressiveness_level=expressiveness_level,
            )
        )
    return settings_list


def _resolve_voice_prompt(req: SynthesizeRequest, settings: _ResolvedGenerationSettings) -> tuple[Path, str, bool]:
    try:
        sample_path = download_and_cache(req.voice_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to download voice sample: {e}") from e

    if req.xvec_only is not None:
        xvec_only = req.xvec_only
    elif _is_vietnamese_qwen_language(settings.language):
        xvec_only = not _qwen_settings.vietnamese_icl_mode
    else:
        xvec_only = _qwen_settings.xvec_only
    if req.voice_text and req.voice_text.strip():
        prompt_text = req.voice_text.strip()
    elif xvec_only:
        prompt_text = ""
    else:
        try:
            prompt_text = transcribe_voice_sample(
                sample_path,
                language=settings.dp_language,
            )
        except EmptyVoiceTranscriptError as e:
            _LOGGER.warning(
                "Qwen3 voice sample transcription was empty; falling back to xvec-only: voice_url=%s sample_path=%s error=%s",
                req.voice_url,
                sample_path,
                e,
            )
            prompt_text = ""
            xvec_only = True
        except Exception as e:
            raise HTTPException(500, f"Failed to transcribe voice sample: {e}") from e
    return sample_path, prompt_text, xvec_only


def _generate_qwen_raw(
    model_obj: Any,
    sample_path: Path,
    prompt_text: str,
    xvec_only: bool,
    use_vietnamese_lora: bool,
    settings: _ResolvedGenerationSettings,
) -> tuple[np.ndarray, int]:
    with _qwen_model_context(model_obj, use_vietnamese_lora):
        audio_list, sample_rate = model_obj.generate_voice_clone(
            text=settings.prepared_text,
            language=settings.language,
            ref_audio=str(sample_path),
            ref_text=prompt_text,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
            xvec_only=xvec_only,
            non_streaming_mode=settings.non_streaming_mode,
            append_silence=True,
            parity_mode=_qwen_settings.disable_cuda_graph,
        )

    wav_data = _concat_qwen_audio(audio_list)
    if wav_data.size == 0:
        raise HTTPException(500, "Model produced no output")

    return wav_data, sample_rate


def _expected_validation_duration_seconds(settings: _ResolvedGenerationSettings) -> float | None:
    info = settings.dp_budget_info or {}
    if not info:
        try:
            info = predict_dp_budget(settings.prepared_text, language=settings.dp_language)
        except Exception:
            _LOGGER.exception("Failed to predict Qwen validation duration")
            info = {}
    for key in ("p50_seconds", "mean_seconds", "seconds"):
        value = info.get(key)
        if value is not None:
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                continue
            if seconds > 0:
                return seconds
    return None


def _qwen_generation_validation_info(
    wav_data: np.ndarray,
    sample_rate: int,
    settings: _ResolvedGenerationSettings,
) -> dict[str, Any]:
    validation = _qwen_settings.validation
    if not settings.validation_enabled:
        return {"enabled": False, "valid": True}

    info = get_dp_budget_model().validate_alignment(
        text=settings.prepared_text,
        wav_data=wav_data,
        sample_rate=sample_rate,
        language=settings.dp_language,
        expected_seconds=_expected_validation_duration_seconds(settings),
        duration_tolerance=validation.duration_tolerance,
        reject_zero_phoneme_duration=validation.reject_zero_phoneme_duration,
    )
    info.update(
        {
            "qwen_language": settings.language,
            "validation_language": settings.dp_language,
            "prepared_text_chars": len(settings.prepared_text),
            "dp_budget_enabled": settings.dp_budget_enabled,
            "dp_budget_info": settings.dp_budget_info,
        }
    )
    return info


def _validate_qwen_generation(
    wav_data: np.ndarray,
    sample_rate: int,
    settings: _ResolvedGenerationSettings,
) -> dict[str, Any]:
    info = _qwen_generation_validation_info(wav_data, sample_rate, settings)
    if not info.get("valid", False):
        raise QwenValidationError(info)
    return info


def _generate_qwen_mp3(
    model_obj: Any,
    sample_path: Path,
    prompt_text: str,
    xvec_only: bool,
    use_vietnamese_lora: bool,
    settings: _ResolvedGenerationSettings,
) -> tuple[bytes, dict[str, Any]]:
    attempts = max(1, _qwen_settings.validation.max_retries + 1) if settings.validation_enabled else 1
    validation_info: dict[str, Any] = {"enabled": settings.validation_enabled, "valid": True}
    failed_candidates: list[tuple[float, int, bytes, dict[str, Any]]] = []
    for attempt in range(1, attempts + 1):
        wav_data, sample_rate = _generate_qwen_raw(
            model_obj,
            sample_path,
            prompt_text,
            xvec_only,
            use_vietnamese_lora,
            settings,
        )
        try:
            validation_info = _validate_qwen_generation(wav_data, sample_rate, settings)
            validation_info["attempt"] = attempt
            mp3_bytes, info = _encode_qwen_audio(wav_data, sample_rate, settings.max_new_tokens)
            info["validation"] = validation_info
            return mp3_bytes, info
        except QwenValidationError as exc:
            validation_info = exc.info | {"attempt": attempt}
            mp3_bytes, info = _encode_qwen_audio(wav_data, sample_rate, settings.max_new_tokens)
            info["validation"] = validation_info
            duration_ratio = validation_info.get("duration_ratio")
            try:
                duration_score = abs(float(duration_ratio) - 1.0)
            except (TypeError, ValueError):
                duration_score = float("inf")
            failed_candidates.append((duration_score, attempt, mp3_bytes, info))
            if attempt >= attempts:
                if not xvec_only:
                    _LOGGER.warning(
                        "Qwen3 generation failed validation after retries; falling back to xvec-only mode failures=%s",
                        [candidate[3]["validation"] for candidate in failed_candidates],
                    )
                    fallback_wav_data, fallback_sample_rate = _generate_qwen_raw(
                        model_obj,
                        sample_path,
                        "",
                        True,
                        use_vietnamese_lora,
                        settings,
                    )
                    fallback_mp3_bytes, fallback_info = _encode_qwen_audio(
                        fallback_wav_data,
                        fallback_sample_rate,
                        settings.max_new_tokens,
                    )
                    fallback_validation_info = _qwen_generation_validation_info(
                        fallback_wav_data,
                        fallback_sample_rate,
                        settings,
                    )
                    fallback_validation_info.update(
                        {
                            "attempt": attempt + 1,
                            "fallback_after_retries": "xvec_only",
                            "fallback_from_xvec_only": xvec_only,
                            "previous_attempts": attempts,
                            "previous_failures": [
                                candidate[3]["validation"]
                                for candidate in failed_candidates
                            ],
                        }
                    )
                    fallback_info["validation"] = fallback_validation_info
                    return fallback_mp3_bytes, fallback_info

                _, selected_attempt, selected_mp3_bytes, selected_info = min(
                    failed_candidates,
                    key=lambda item: (item[0], item[1]),
                )
                selected_info["validation"]["selected_after_retries"] = True
                selected_info["validation"]["selected_attempt"] = selected_attempt
                _LOGGER.warning(
                    "Qwen3 xvec-only generation failed validation after retries; returning closest-duration attempt selected_attempt=%d selected_info=%s",
                    selected_attempt,
                    selected_info["validation"],
                )
                return selected_mp3_bytes, selected_info
            _LOGGER.warning(
                "Qwen3 generation failed validation; retrying attempt=%d/%d reason=%s info=%s",
                attempt,
                attempts,
                exc.info.get("reason"),
                exc.info,
            )

    raise HTTPException(500, "Qwen3 generation failed unexpectedly")


def _concat_qwen_audio(audio: Any) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return np.asarray(audio, dtype=np.float32).flatten()

    parts = []
    for chunk in audio or []:
        chunk_data = np.asarray(chunk, dtype=np.float32).flatten()
        if chunk_data.size:
            parts.append(chunk_data)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def _encode_qwen_audio(
    wav_data: np.ndarray,
    sample_rate: int,
    max_new_tokens: int,
) -> tuple[bytes, dict[str, Any]]:
    raw_audio_seconds = wav_data.size / sample_rate
    raw_level = _qwen_audio_level(wav_data)
    wav_data, postprocess_info = postprocess_audio(wav_data, sample_rate)
    output_level = _qwen_audio_level(wav_data)
    postprocess_audio_seconds = wav_data.size / sample_rate
    cap_seconds = max_new_tokens / 12.0
    hit_token_cap = raw_audio_seconds >= max(0.0, cap_seconds - 0.25)
    mp3_bytes = wav_to_mp3(wav_data, sample_rate)
    audio_seconds = _encoded_mp3_duration_seconds(mp3_bytes, postprocess_audio_seconds)
    return mp3_bytes, {
        "sample_rate": sample_rate,
        "raw_audio_seconds": raw_audio_seconds,
        "postprocess_audio_seconds": postprocess_audio_seconds,
        "audio_seconds": audio_seconds,
        "cap_seconds": cap_seconds,
        "hit_token_cap": hit_token_cap,
        "raw_level": raw_level,
        "output_level": output_level,
        "postprocess": postprocess_info,
    }


def _qwen_audio_level(wav_data: np.ndarray) -> dict[str, float]:
    if wav_data.size == 0:
        return {"rms": 0.0, "peak": 0.0}
    samples = np.asarray(wav_data, dtype=np.float32).flatten()
    return {
        "rms": round(float(np.sqrt(np.mean(np.square(samples)))), 8),
        "peak": round(float(np.max(np.abs(samples))), 8),
    }


def _generate_qwen_batch_audio_for_indices(
    model_obj: Any,
    item_indices: list[int],
    voice_clone_prompts: list[Any],
    settings_list: list[_ResolvedGenerationSettings],
    xvec_only: bool,
    use_vietnamese_lora: bool,
) -> tuple[list[int | None], list[Any], int]:
    if not item_indices:
        raise ValueError("item_indices must not be empty")

    template_index = item_indices[0]
    template = settings_list[template_index]
    bucket_size = _qwen_batch_bucket_size(len(item_indices))
    active_indices: list[int | None] = list(item_indices)
    active_indices.extend([None] * (bucket_size - len(active_indices)))

    dummy_settings = _make_qwen_batch_dummy_settings(template)
    dummy_prompt = voice_clone_prompts[template_index]
    active_settings = [
        settings_list[item_index] if item_index is not None else dummy_settings
        for item_index in active_indices
    ]
    active_prompts = [
        voice_clone_prompts[item_index] if item_index is not None else dummy_prompt
        for item_index in active_indices
    ]
    active_texts = [settings.prepared_text for settings in active_settings]

    with _qwen_model_context(model_obj, use_vietnamese_lora):
        audio_list, sample_rate = model_obj.generate_voice_clone_batch(
            texts=active_texts,
            language=[settings.language for settings in active_settings],
            ref_audio=None,
            ref_text="",
            max_new_tokens=[settings.max_new_tokens for settings in active_settings],
            temperature=template.temperature,
            top_k=template.top_k,
            top_p=template.top_p,
            repetition_penalty=template.repetition_penalty,
            xvec_only=xvec_only,
            non_streaming_mode=template.non_streaming_mode,
            append_silence=True,
            voice_clone_prompt=active_prompts,
            parity_mode=_qwen_settings.disable_cuda_graph_batch,
        )

    if len(audio_list) != len(active_texts):
        raise HTTPException(500, f"Model produced {len(audio_list)} outputs for {len(active_texts)} inputs")
    return active_indices, audio_list, sample_rate


def _generate_qwen_batch_mp3(
    model_obj: Any,
    item_requests: list[SynthesizeRequest],
    sample_paths: list[Path],
    prompt_texts: list[str],
    voice_clone_prompts: list[Any],
    settings_list: list[_ResolvedGenerationSettings],
    xvec_only: bool,
    use_vietnamese_lora: bool,
) -> list[tuple[bytes, dict[str, Any]] | None]:
    if (
        len(item_requests) != len(settings_list)
        or len(item_requests) != len(voice_clone_prompts)
        or len(item_requests) != len(sample_paths)
        or len(item_requests) != len(prompt_texts)
    ):
        raise ValueError(
            "item_requests, sample_paths, prompt_texts, voice_clone_prompts, and settings_list length mismatch"
        )

    dummy_mask = [_is_qwen_batch_dummy(item) for item in item_requests]
    if not hasattr(model_obj, "generate_voice_clone_batch"):
        raise HTTPException(500, "Loaded Qwen backend does not support true batch generation")

    first_real_index = next(
        (index for index, is_dummy in enumerate(dummy_mask) if not is_dummy),
        None,
    )
    if first_real_index is None:
        raise HTTPException(400, "at least one non-dummy item is required")
    attempts = (
        max(1, _qwen_settings.validation.max_retries + 1)
        if any(settings.validation_enabled for settings in settings_list)
        else 1
    )

    results: list[tuple[bytes, dict[str, Any]] | None] = [None] * len(item_requests)
    pending_indices = [
        item_index
        for item_index, is_dummy in enumerate(dummy_mask)
        if not is_dummy
    ]

    for attempt in range(1, attempts + 1):
        if not pending_indices:
            return results

        active_indices, audio_list, sample_rate = _generate_qwen_batch_audio_for_indices(
            model_obj=model_obj,
            item_indices=pending_indices,
            voice_clone_prompts=voice_clone_prompts,
            settings_list=settings_list,
            xvec_only=xvec_only,
            use_vietnamese_lora=use_vietnamese_lora,
        )

        invalid_items = []
        next_pending_indices: list[int] = []
        for batch_index, item_index in enumerate(active_indices):
            if item_index is None:
                continue
            settings = settings_list[item_index]
            audio = audio_list[batch_index]
            wav_data = np.asarray(audio, dtype=np.float32).flatten()
            validation_info = _qwen_generation_validation_info(wav_data, sample_rate, settings)
            validation_info["attempt"] = attempt
            mp3_bytes, info = _encode_qwen_audio(wav_data, sample_rate, settings.max_new_tokens)
            info["validation"] = validation_info

            if validation_info.get("valid", False):
                results[item_index] = (mp3_bytes, info)
                continue

            invalid_item = {
                "item_index": item_index,
                "text": settings.prepared_text,
                "validation": validation_info,
            }
            invalid_items.append(invalid_item)
            if attempt >= attempts:
                results[item_index] = (mp3_bytes, info)
            else:
                next_pending_indices.append(item_index)

        if not next_pending_indices:
            if invalid_items:
                _LOGGER.warning(
                    "Qwen3 batch generation failed validation after retries; returning final items invalid_items=%s",
                    invalid_items,
                )
            return results

        if invalid_items:
            _LOGGER.warning(
                "Qwen3 batch generation failed validation; retrying invalid items attempt=%d/%d invalid_items=%s",
                attempt,
                attempts,
                invalid_items,
            )
        pending_indices = next_pending_indices

    missing_indices = [
        item_index
        for item_index, is_dummy in enumerate(dummy_mask)
        if not is_dummy and results[item_index] is None
    ]
    if missing_indices:
        raise HTTPException(500, f"Qwen3 batch generation did not produce results for items {missing_indices}")
    return results


def _validate_qwen_batch_shared_settings(settings_list: list[_ResolvedGenerationSettings]) -> None:
    if not settings_list:
        return
    first = settings_list[0]
    shared_fields = (
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "non_streaming_mode",
    )
    for idx, settings in enumerate(settings_list[1:], start=1):
        mismatched = [
            field
            for field in shared_fields
            if getattr(settings, field) != getattr(first, field)
        ]
        if mismatched:
            raise HTTPException(
                400,
                "Qwen3 batch generation requires shared settings for "
                f"{', '.join(shared_fields)}; item {idx} differs in {', '.join(mismatched)}",
            )


def _qwen_batch_bucket_size(count: int) -> int:
    for bucket_size in QWEN_BATCH_BUCKETS:
        if count <= bucket_size:
            return bucket_size
    raise ValueError(f"Qwen3 internal batch chunk size must be <= {QWEN_MODEL_MAX_BATCH_SIZE}; got {count}")


def _qwen_batch_chunks(count: int) -> list[range]:
    chunks: list[range] = []
    start = 0
    while start < count:
        end = min(start + QWEN_MODEL_MAX_BATCH_SIZE, count)
        chunks.append(range(start, end))
        start = end
    return chunks


def _make_qwen_batch_dummy_request(template: SynthesizeRequest) -> SynthesizeRequest:
    return template.model_copy(
        update={
            "text": QWEN_BATCH_DUMMY_TEXT,
            "max_new_tokens": 0,
            "dp_budget": False,
        }
    )


def _make_qwen_batch_dummy_settings(
    template: _ResolvedGenerationSettings,
) -> _ResolvedGenerationSettings:
    return _ResolvedGenerationSettings(
        language=template.language,
        dp_language=template.dp_language,
        prepared_text=QWEN_BATCH_DUMMY_TEXT,
        max_new_tokens=0,
        temperature=template.temperature,
        top_k=template.top_k,
        top_p=template.top_p,
        repetition_penalty=template.repetition_penalty,
        non_streaming_mode=template.non_streaming_mode,
        dp_budget_enabled=False,
        dp_budget_info=None,
        validation_enabled=template.validation_enabled,
        expressiveness_level=template.expressiveness_level,
    )


def _create_voice_clone_prompt_item(
    model_obj: Any,
    sample_path: Path,
    prompt_text: str,
    xvec_only: bool,
    use_vietnamese_lora: bool,
) -> Any:
    with _qwen_model_context(model_obj, use_vietnamese_lora):
        if xvec_only:
            prompt_items = model_obj.model.create_voice_clone_prompt(
                ref_audio=str(sample_path),
                ref_text="",
                x_vector_only_mode=True,
            )
        else:
            ref_audio_input = model_obj._load_ref_audio_with_silence(sample_path, silence_secs=0.5)
            prompt_items = model_obj.model.create_voice_clone_prompt(
                ref_audio=ref_audio_input,
                ref_text=prompt_text,
            )
    if not prompt_items:
        raise HTTPException(500, "Failed to create Qwen voice clone prompt")
    return prompt_items[0]


@router.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    started = time.perf_counter()
    _log_qwen_synthesize_request(status="received", started=started, req=req, xvec_only=req.xvec_only)

    prompt_text: str | None = None
    xvec_only: bool | None = None
    settings: _ResolvedGenerationSettings | None = None
    try:
        if not req.text.strip():
            raise HTTPException(400, "text is required")

        settings = _resolve_generation_settings(req)
        sample_path, prompt_text, xvec_only = _resolve_voice_prompt(req, settings)
        backend_key = _qwen_model_key_for_language(settings.language)
        qwen_model = get_qwen_model_for_language(settings.language)
        use_vietnamese_lora = backend_key == "vietnamese" and not _qwen_settings.vietnamese_model.strip()

        with inference_lock:
            mp3_bytes, info = _generate_qwen_mp3(
                qwen_model,
                sample_path,
                prompt_text,
                xvec_only,
                use_vietnamese_lora,
                settings,
            )
            info["backend"] = backend_key

        _log_qwen_synthesize_request(
            status="ok",
            started=started,
            req=req,
            settings=settings,
            prompt_text=prompt_text,
            xvec_only=xvec_only,
            info=info,
        )

        return _binary_response(mp3_bytes, "audio/mpeg")
    except HTTPException as exc:
        _log_qwen_synthesize_request(
            status=f"http_{exc.status_code}",
            started=started,
            req=req,
            settings=settings,
            prompt_text=prompt_text,
            xvec_only=xvec_only,
            error=str(exc.detail),
        )
        raise
    except Exception as exc:
        _log_qwen_synthesize_request(
            status="error",
            started=started,
            req=req,
            settings=settings,
            prompt_text=prompt_text,
            xvec_only=xvec_only,
            error=type(exc).__name__,
        )
        raise


@router.post("/synthesize/batch", response_model=BatchSynthesizeResponse)
@router.post("/synthesize-batch", response_model=BatchSynthesizeResponse)
def synthesize_batch(req: BatchSynthesizeRequest):
    started = time.perf_counter()
    if len(req.items) > QWEN_MAX_BATCH_SIZE:
        raise HTTPException(400, f"Qwen3 batch size must be <= {QWEN_MAX_BATCH_SIZE}")

    item_requests = [
        item.model_copy(update={"text": item.text.strip()})
        for item in req.items
    ]
    _log_qwen_batch_synthesize_request(
        status="received",
        started=started,
        req=req,
        item_requests=item_requests,
    )
    texts = [item.text for item in item_requests]
    for index, text in enumerate(texts):
        if not text:
            raise HTTPException(400, f"items[{index}].text is required")

    settings_list = _resolve_generation_settings_batch(item_requests)
    resolved_prompts = [
        _resolve_voice_prompt(item, settings)
        for item, settings in zip(item_requests, settings_list)
    ]
    _log_qwen_batch_synthesize_request(
        status="resolved",
        started=started,
        req=req,
        item_requests=item_requests,
        settings_list=settings_list,
        resolved_prompts=resolved_prompts,
    )

    items: list[BatchSynthesizeItem | None] = [None] * len(item_requests)
    with inference_lock:
        for index, (item, settings, resolved_prompt) in enumerate(
            zip(item_requests, settings_list, resolved_prompts)
        ):
            sample_path, prompt_text, xvec_only = resolved_prompt
            backend_key = _qwen_model_key_for_language(settings.language)
            model_obj = get_qwen_model_for_language(settings.language)
            use_vietnamese_lora = backend_key == "vietnamese" and not _qwen_settings.vietnamese_model.strip()
            mp3_bytes, info = _generate_qwen_mp3(
                model_obj,
                sample_path,
                prompt_text,
                xvec_only,
                use_vietnamese_lora,
                settings,
            )
            items[index] = BatchSynthesizeItem(
                text=item.text,
                audio_base64=base64.b64encode(mp3_bytes).decode("ascii"),
                sample_rate=int(info["sample_rate"]),
                raw_audio_seconds=float(info["raw_audio_seconds"]),
                audio_seconds=float(info["audio_seconds"]),
                max_new_tokens=settings.max_new_tokens,
                hit_token_cap=bool(info["hit_token_cap"]),
                language=settings.language,
                dp_language=settings.dp_language,
            )
    wall_seconds = time.perf_counter() - started
    final_items = [item for item in items if item is not None]
    result_info = {
        "count": len(final_items),
        "audio_seconds_total": sum(item.audio_seconds for item in final_items),
        "items": [
            {
                "index": index,
                "text": item.text,
                "language": item.language,
                "dp_language": item.dp_language,
                "max_new_tokens": item.max_new_tokens,
                "raw_audio_seconds": item.raw_audio_seconds,
                "audio_seconds": item.audio_seconds,
                "hit_token_cap": item.hit_token_cap,
            }
            for index, item in enumerate(final_items)
        ],
    }
    _log_qwen_batch_synthesize_request(
        status="ok",
        started=started,
        req=req,
        item_requests=item_requests,
        settings_list=settings_list,
        resolved_prompts=resolved_prompts,
        info=result_info,
    )
    return BatchSynthesizeResponse(
        items=final_items,
        count=len(final_items),
        wall_seconds=wall_seconds,
        audio_seconds_total=result_info["audio_seconds_total"],
    )
