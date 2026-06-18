import base64
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

_LOGGER = logging.getLogger(__name__)

CACHE_DIR = Path("cache/voice_samples")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

QWEN_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN_DEFAULT_DTYPE = "bfloat16"
QWEN_DEFAULT_LANGUAGE = "Auto"
QWEN_DEFAULT_MAX_NEW_TOKENS = 360
QWEN_DEFAULT_TEMPERATURE = 0.9
QWEN_DEFAULT_TOP_K = 50
QWEN_DEFAULT_TOP_P = 1.0
QWEN_DEFAULT_REPETITION_PENALTY = 1.03
QWEN_DEFAULT_XVEC_ONLY = True
QWEN_DEFAULT_NON_STREAMING_MODE = True
QWEN_DEFAULT_EXPRESSIVENESS = 1.0
QWEN_DP_BUDGET_DEFAULT = True
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
SILENCE_TRIM_MIN_CHUNK_RMS = 0.003
SILENCE_TRIM_RELATIVE_CHUNK_RMS = 0.03
SILENCE_TRIM_EMPTY_OUTPUT_MS = 200
EXPRESSIVENESS_PRESETS = {
    1.0: {"temperature": 0.90, "top_k": 50, "repetition_penalty": 1.03},
    0.8: {"temperature": 0.84, "top_k": 48, "repetition_penalty": 1.035},
    0.6: {"temperature": 0.78, "top_k": 46, "repetition_penalty": 1.04},
    0.4: {"temperature": 0.72, "top_k": 44, "repetition_penalty": 1.045},
    0.2: {"temperature": 0.66, "top_k": 42, "repetition_penalty": 1.05},
    0.0: {"temperature": 0.60, "top_k": 40, "repetition_penalty": 1.055},
}

router = APIRouter(prefix="/qwen3", tags=["qwen3"])
model: Optional[Any] = None
parakeet_model: Optional[Any] = None
parakeet_device: Optional[torch.device] = None
silero_vad_detector: Optional[Any] = None
dp_budget_model: Optional[Any] = None
inference_lock = threading.Lock()
parakeet_lock = threading.Lock()
model_load_lock = threading.Lock()
dp_budget_load_lock = threading.Lock()
silero_vad_lock = threading.Lock()
model_ready_event = threading.Event()
model_loading = False
model_load_error: Optional[str] = None
model_load_started_at: Optional[float] = None
model_load_finished_at: Optional[float] = None
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


class QwenSettings(BaseModel):
    preload: bool = True
    preload_background: bool = True
    model: str = QWEN_DEFAULT_MODEL
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
    temperature: float = QWEN_DEFAULT_TEMPERATURE
    top_k: int = QWEN_DEFAULT_TOP_K
    top_p: float = QWEN_DEFAULT_TOP_P
    repetition_penalty: float = QWEN_DEFAULT_REPETITION_PENALTY
    voice_prompt_cache_entries: int = Field(8, ge=0)
    dp_budget: DpBudgetSettings = Field(default_factory=DpBudgetSettings)


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
    }
    for env_name, attr in string_overrides.items():
        value = os.environ.get(env_name)
        if value is not None and value.strip():
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
    }
    for env_name, attr in int_overrides.items():
        value = os.environ.get(env_name)
        if value is not None and value.strip():
            setattr(settings, attr, int(value.strip()))

    bool_overrides = {
        "QWEN_TTS_PRELOAD": "preload",
        "QWEN_TTS_PRELOAD_BACKGROUND": "preload_background",
        "QWEN_TTS_WARMUP": "warmup",
        "QWEN_TTS_XVEC_ONLY": "xvec_only",
        "QWEN_TTS_NON_STREAMING_MODE": "non_streaming_mode",
    }
    for env_name, attr in bool_overrides.items():
        if os.environ.get(env_name) is not None:
            setattr(settings, attr, env_bool_value(env_name, getattr(settings, attr)))

    return settings


def configure(settings: QwenSettings) -> None:
    global _qwen_settings, dp_budget_model
    _qwen_settings = settings
    dp_budget_model = None


def demo_defaults() -> dict[str, Any]:
    return {
        "language": _qwen_settings.language,
        "temperature": _qwen_settings.temperature,
        "top_k": _qwen_settings.top_k,
        "top_p": _qwen_settings.top_p,
        "repetition_penalty": _qwen_settings.repetition_penalty,
        "xvec_only": _qwen_settings.xvec_only,
        "non_streaming_mode": _qwen_settings.non_streaming_mode,
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
    if value < 0 or value > 1:
        raise HTTPException(400, "expressiveness must be between 0 and 1")

    level = round(value * 5) / 5
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
        return _default_qwen_language()

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
        "model_loading": model_loading,
        "model_load_error": model_load_error,
        "model_load_started_at": model_load_started_at,
        "model_load_finished_at": model_load_finished_at,
        "precision_mode": _qwen_settings.precision_mode,
        "model": _qwen_settings.model,
        "dtype": _qwen_settings.dtype,
        "audio_dtype": _qwen_settings.audio_dtype,
        "attn": _qwen_settings.attn,
        "layer_precision": _qwen_settings.layer_precision,
        "predictor_layer_precision": _qwen_settings.predictor_layer_precision,
        "audio_decoder_precision": _qwen_settings.audio_decoder_precision,
        "large_block_precision": _qwen_settings.large_block_precision,
        "extra_precision": _qwen_settings.extra_precision,
        "linear_precision": _qwen_settings.linear_precision,
    }


def _load_model_unlocked() -> Any:
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    model_name = _qwen_settings.model
    device = _qwen_settings.device
    dtype_name = _qwen_settings.dtype
    if dtype_name.lower() == "auto":
        dtype = "auto"
    else:
        dtype = getattr(torch, dtype_name, torch.bfloat16)
    audio_dtype = _qwen_settings.audio_dtype
    attn_implementation = _qwen_settings.attn
    max_seq_len = _qwen_settings.max_seq_len
    print(
        "Loading FasterQwen3TTS "
        f"precision_mode={_qwen_settings.precision_mode} "
        f"model={model_name} device={device} dtype={dtype_name} "
        f"audio_dtype={audio_dtype} attn={attn_implementation} "
        f"layer_precision={_qwen_settings.layer_precision} "
        f"predictor_layer_precision={_qwen_settings.predictor_layer_precision} "
        f"audio_decoder_precision={_qwen_settings.audio_decoder_precision} "
        f"large_block_precision={_qwen_settings.large_block_precision} "
        f"extra_precision={_qwen_settings.extra_precision} "
        f"linear_precision={_qwen_settings.linear_precision} "
        f"max_seq_len={max_seq_len}...",
        file=sys.stderr,
        flush=True,
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
    if _qwen_settings.warmup and hasattr(loaded_model, "_warmup"):
        print("Capturing CUDA graphs...", file=sys.stderr, flush=True)
        loaded_model._warmup(prefill_len=100)
        if hasattr(loaded_model, "capture_batch_graphs"):
            print(
                f"Capturing Qwen batch CUDA graph buckets: {QWEN_BATCH_BUCKETS}...",
                file=sys.stderr,
                flush=True,
            )
            loaded_model.capture_batch_graphs(QWEN_BATCH_BUCKETS, prefill_len=100)
    if hasattr(loaded_model, "max_voice_prompt_cache_entries"):
        loaded_model.max_voice_prompt_cache_entries = _qwen_settings.voice_prompt_cache_entries
    print(f"FasterQwen3TTS loaded. Sample rate: {loaded_model.sample_rate}", file=sys.stderr, flush=True)
    return loaded_model


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
            model = _load_model_unlocked()
            model_load_error = None
            return model
        except Exception as e:
            model_load_error = str(e)
            raise
        finally:
            model_loading = False
            model_load_finished_at = time.time()
            model_ready_event.set()


def _preload_worker(include_dp_budget: bool) -> None:
    try:
        get_model()
        if include_dp_budget:
            get_dp_budget_model()
    except Exception as e:
        print(f"Qwen3 preload failed: {e}", file=sys.stderr, flush=True)


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

        print("Loading Qwen DP budget model...", file=sys.stderr, flush=True)
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
            )
        )
        loaded_dp_budget_model.load()
        dp_budget_model = loaded_dp_budget_model
        print("Qwen DP budget model ready.", file=sys.stderr, flush=True)
    return dp_budget_model


def predict_dp_budget(text: str, language: Optional[str] = None) -> dict[str, Any]:
    return get_dp_budget_model().predict(text, language=language)


def predict_dp_budget_batch(
    texts: list[str],
    languages: Optional[list[str | None]] = None,
) -> list[dict[str, Any]]:
    return get_dp_budget_model().predict_batch(texts, languages=languages)


def get_parakeet_model() -> Any:
    global parakeet_model, parakeet_device
    if parakeet_model is None:
        from nano_parakeet import from_pretrained as parakeet_from_pretrained
        from nano_parakeet.model import ParakeetTDT

        device = os.environ.get("NANO_PARAKEET_DEVICE", "cuda").strip() or "cuda"
        if device == "cuda":
            device = "cuda:0"
        parakeet_device = torch.device(device)
        disable_cuda_graph = env_bool("NANO_PARAKEET_DISABLE_CUDA_GRAPH", False)
        original_build_decode_graph = None
        if disable_cuda_graph and parakeet_device.type == "cuda":
            original_build_decode_graph = ParakeetTDT._build_decode_graph
            ParakeetTDT._build_decode_graph = lambda self, device: None
        print("Loading transcription model (nano-parakeet)...", file=sys.stderr, flush=True)
        try:
            dtype = torch.float32 if disable_cuda_graph and parakeet_device.type == "cuda" else None
            if parakeet_device.type == "cuda":
                with torch.cuda.device(parakeet_device):
                    parakeet_model = parakeet_from_pretrained(device=device, dtype=dtype)
            else:
                parakeet_model = parakeet_from_pretrained(device=device, dtype=dtype)
            if disable_cuda_graph and parakeet_device.type == "cuda":
                parakeet_model._build_decode_graph = lambda device: None
        finally:
            if original_build_decode_graph is not None:
                ParakeetTDT._build_decode_graph = original_build_decode_graph
        print(
            f"Transcription model ready on {device} "
            f"(cuda_graph={'disabled' if disable_cuda_graph and parakeet_device.type == 'cuda' else 'enabled'}).",
            file=sys.stderr,
            flush=True,
        )
    return parakeet_model


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
                print("Loading Silero VAD postprocessor...", file=sys.stderr, flush=True)
                silero_vad_detector = make_silence_detector(providers=providers)
                print(
                    f"Silero VAD ready. Providers: {silero_vad_detector.providers}",
                    file=sys.stderr,
                    flush=True,
                )
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


def transcribe_voice_sample(audio_file: Path) -> str:
    transcript_file = audio_file.with_suffix(".parakeet.txt")
    if transcript_file.exists():
        transcript = transcript_file.read_text(encoding="utf-8").strip()
        if transcript:
            return transcript

    parakeet = get_parakeet_model()
    with parakeet_lock, torch.inference_mode():
        if parakeet_device is not None and parakeet_device.type == "cuda":
            with torch.cuda.device(parakeet_device):
                text = parakeet.transcribe(str(audio_file)).strip()
        else:
            text = parakeet.transcribe(str(audio_file)).strip()
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
        raise EmptyVoiceTranscriptError(f"nano-parakeet transcript completed with empty text ({debug})")
    transcript_file.write_text(text, encoding="utf-8")
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


def postprocess_audio(wav_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": False,
        "vad": "silero",
        "trim": False,
        "trim_head_seconds": 0.0,
        "trim_tail_seconds": 0.0,
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

    return wav_data, info


def wav_to_mp3(wav_data: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav_data, sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)

    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-codec:a", "libmp3lame", "-q:a", "0", "-f", "mp3", "pipe:1"],
        input=buf.read(),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    return proc.stdout


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
        }
    if info is not None:
        log_data["result"] = info
    if error is not None:
        log_data["error"] = error
    _LOGGER.info("Qwen3 synthesize request: %s", json.dumps(log_data, ensure_ascii=False))


@dataclass(frozen=True)
class _ResolvedGenerationSettings:
    language: str
    dp_language: str
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    non_streaming_mode: bool
    dp_budget_enabled: bool
    dp_budget_info: Optional[dict[str, Any]]
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
        dp_budget_texts.append(reqs[index].text)
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
        top_p = req.top_p if req.top_p is not None else _qwen_settings.top_p
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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                non_streaming_mode=non_streaming_mode,
                dp_budget_enabled=dp_budget_enabled,
                dp_budget_info=dp_budget_info,
                expressiveness_level=expressiveness_level,
            )
        )
    return settings_list


def _resolve_voice_prompt(req: SynthesizeRequest) -> tuple[Path, str, bool]:
    try:
        sample_path = download_and_cache(req.voice_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to download voice sample: {e}") from e

    xvec_only = req.xvec_only if req.xvec_only is not None else _qwen_settings.xvec_only
    if req.voice_text and req.voice_text.strip():
        prompt_text = req.voice_text.strip()
    elif xvec_only:
        prompt_text = ""
    else:
        try:
            prompt_text = transcribe_voice_sample(sample_path)
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


def _generate_qwen_mp3(
    req: SynthesizeRequest,
    sample_path: Path,
    prompt_text: str,
    xvec_only: bool,
    settings: _ResolvedGenerationSettings,
) -> tuple[bytes, dict[str, Any]]:
    m = get_model()
    audio_list, sample_rate = m.generate_voice_clone(
        text=req.text,
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
    )

    wav_data = _concat_qwen_audio(audio_list)
    if wav_data.size == 0:
        raise HTTPException(500, "Model produced no output")

    return _encode_qwen_audio(wav_data, sample_rate, settings.max_new_tokens)


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
    audio_seconds = wav_data.size / sample_rate
    cap_seconds = max_new_tokens / 12.0
    hit_token_cap = raw_audio_seconds >= max(0.0, cap_seconds - 0.25)
    mp3_bytes = wav_to_mp3(wav_data, sample_rate)
    return mp3_bytes, {
        "sample_rate": sample_rate,
        "raw_audio_seconds": raw_audio_seconds,
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


def _generate_qwen_batch_mp3(
    item_requests: list[SynthesizeRequest],
    voice_clone_prompts: list[Any],
    settings_list: list[_ResolvedGenerationSettings],
    xvec_only: bool,
) -> list[tuple[bytes, dict[str, Any]] | None]:
    if len(item_requests) != len(settings_list) or len(item_requests) != len(voice_clone_prompts):
        raise ValueError("item_requests, voice_clone_prompts, and settings_list length mismatch")

    dummy_mask = [_is_qwen_batch_dummy(item) for item in item_requests]
    m = get_model()
    if not hasattr(m, "generate_voice_clone_batch"):
        raise HTTPException(500, "Loaded Qwen backend does not support true batch generation")

    first_real_index = next(
        (index for index, is_dummy in enumerate(dummy_mask) if not is_dummy),
        None,
    )
    if first_real_index is None:
        raise HTTPException(400, "at least one non-dummy item is required")
    first = settings_list[first_real_index]
    texts = [item.text for item in item_requests]
    audio_list, sample_rate = m.generate_voice_clone_batch(
        texts=texts,
        language=[settings.language for settings in settings_list],
        ref_audio=None,
        ref_text="",
        max_new_tokens=[settings.max_new_tokens for settings in settings_list],
        temperature=first.temperature,
        top_k=first.top_k,
        top_p=first.top_p,
        repetition_penalty=first.repetition_penalty,
        xvec_only=xvec_only,
        non_streaming_mode=first.non_streaming_mode,
        append_silence=True,
        voice_clone_prompt=voice_clone_prompts,
    )

    if len(audio_list) != len(texts):
        raise HTTPException(500, f"Model produced {len(audio_list)} outputs for {len(texts)} inputs")

    encoded = []
    for audio, settings, is_dummy in zip(audio_list, settings_list, dummy_mask):
        if is_dummy:
            encoded.append(None)
            continue
        wav_data = np.asarray(audio, dtype=np.float32).flatten()
        encoded.append(_encode_qwen_audio(wav_data, sample_rate, settings.max_new_tokens))
    return encoded


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
        max_new_tokens=0,
        temperature=template.temperature,
        top_k=template.top_k,
        top_p=template.top_p,
        repetition_penalty=template.repetition_penalty,
        non_streaming_mode=template.non_streaming_mode,
        dp_budget_enabled=False,
        dp_budget_info=None,
        expressiveness_level=template.expressiveness_level,
    )


def _create_voice_clone_prompt_item(
    sample_path: Path,
    prompt_text: str,
    xvec_only: bool,
) -> Any:
    m = get_model()
    if xvec_only:
        prompt_items = m.model.create_voice_clone_prompt(
            ref_audio=str(sample_path),
            ref_text="",
            x_vector_only_mode=True,
        )
    else:
        ref_audio_input = m._load_ref_audio_with_silence(sample_path, silence_secs=0.5)
        prompt_items = m.model.create_voice_clone_prompt(
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

        sample_path, prompt_text, xvec_only = _resolve_voice_prompt(req)
        settings = _resolve_generation_settings(req)

        with inference_lock:
            mp3_bytes, info = _generate_qwen_mp3(req, sample_path, prompt_text, xvec_only, settings)
            torch.cuda.empty_cache()

        _log_qwen_synthesize_request(
            status="ok",
            started=started,
            req=req,
            settings=settings,
            prompt_text=prompt_text,
            xvec_only=xvec_only,
            info=info,
        )

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return StreamingResponse(io.BytesIO(mp3_bytes), media_type="audio/mpeg")
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
    if len(req.items) > QWEN_MAX_BATCH_SIZE:
        raise HTTPException(400, f"Qwen3 batch size must be <= {QWEN_MAX_BATCH_SIZE}")

    item_requests = [
        item.model_copy(update={"text": item.text.strip()})
        for item in req.items
    ]
    texts = [item.text for item in item_requests]
    for index, text in enumerate(texts):
        if not text:
            raise HTTPException(400, f"items[{index}].text is required")

    settings_list = _resolve_generation_settings_batch(item_requests)
    _validate_qwen_batch_shared_settings(settings_list)
    resolved_prompts = [
        _resolve_voice_prompt(item)
        for item in item_requests
    ]
    xvec_modes = [xvec_only for _, _, xvec_only in resolved_prompts]
    if any(mode != xvec_modes[0] for mode in xvec_modes):
        raise HTTPException(
            400,
            "Qwen3 batch generation requires all items to share xvec_only mode",
        )

    started = time.perf_counter()
    items: list[BatchSynthesizeItem] = []
    with inference_lock:
        prompt_items = [
            _create_voice_clone_prompt_item(
                sample_path=sample_path,
                prompt_text=prompt_text,
                xvec_only=xvec_only,
            )
            for sample_path, prompt_text, xvec_only in resolved_prompts
        ]

        for chunk in _qwen_batch_chunks(len(item_requests)):
            chunk_indices = list(chunk)
            chunk_requests = [item_requests[index] for index in chunk_indices]
            chunk_settings = [settings_list[index] for index in chunk_indices]
            chunk_prompts = [prompt_items[index] for index in chunk_indices]

            bucket_size = _qwen_batch_bucket_size(len(chunk_requests))
            padding_count = bucket_size - len(chunk_requests)
            if padding_count:
                dummy_request = _make_qwen_batch_dummy_request(chunk_requests[0])
                dummy_settings = _make_qwen_batch_dummy_settings(chunk_settings[0])
                dummy_prompt = chunk_prompts[0]
                chunk_requests.extend(dummy_request for _ in range(padding_count))
                chunk_settings.extend(dummy_settings for _ in range(padding_count))
                chunk_prompts.extend(dummy_prompt for _ in range(padding_count))

            generated = _generate_qwen_batch_mp3(
                item_requests=chunk_requests,
                voice_clone_prompts=chunk_prompts,
                settings_list=chunk_settings,
                xvec_only=xvec_modes[0],
            )

            for index, settings, generated_item in zip(chunk_indices, chunk_settings, generated):
                if generated_item is None:
                    raise HTTPException(500, "Model did not produce audio for a non-dummy item")
                mp3_bytes, info = generated_item
                items.append(
                    BatchSynthesizeItem(
                        text=texts[index],
                        audio_base64=base64.b64encode(mp3_bytes).decode("ascii"),
                        sample_rate=int(info["sample_rate"]),
                        raw_audio_seconds=float(info["raw_audio_seconds"]),
                        audio_seconds=float(info["audio_seconds"]),
                        max_new_tokens=settings.max_new_tokens,
                        hit_token_cap=bool(info["hit_token_cap"]),
                        language=settings.language,
                        dp_language=settings.dp_language,
                    )
                )
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    wall_seconds = time.perf_counter() - started
    return BatchSynthesizeResponse(
        items=items,
        count=len(items),
        wall_seconds=wall_seconds,
        audio_seconds_total=sum(item.audio_seconds for item in items),
    )
