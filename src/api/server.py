"""FastAPI server for Sparrow/VITS TTS inference."""

from __future__ import annotations

import base64
import asyncio
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import secrets
import shutil
import sys
import threading
import time
import httpx
import uuid
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Optional
from urllib.parse import parse_qsl, urlencode

import numpy as np
import torch
from asgi_compression import BrotliAlgorithm, CompressionMiddleware, GzipAlgorithm, ZstdAlgorithm
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pydub import AudioSegment

from ..multilingual_splitter import MultilingualSplitter
from ..piper import PiperInference
from ..ssml import BreakSegment, TextSegment, generate_silence, parse_ssml
from ..matcha_inference import MatchaBackend as ProductionMatchaBackend
from ..matcha_inference import MatchaBatcher as ProductionMatchaBatcher
from . import qwen3
from .qwen3 import router as qwen3_router
from .request_decompression import RequestDecompressionMiddleware
from .rvc import RVCBackend, RVCSettings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
_LOGGER = logging.getLogger(__name__)
load_dotenv()

# Default paths
DATA_DIR = Path("data")
CONFIG_PATH = Path(os.environ.get("LZ_TTS_SERVER_CONFIG", "local/server.json"))
LLMS_TEMPLATE_PATH = Path(__file__).with_name("llms.txt")
DEFAULT_MODEL = "lzspeech-sparrow"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_VC_ROOT = PROJECT_ROOT / "data" / "seed-vc"
SEED_VC_RUNTIME_ROOT = PROJECT_ROOT / "src" / "seed_vc_runtime"
SEED_VC_VOICE_IDS_PATH = SEED_VC_ROOT / "voice_ids.txt"
MATCHA_LANGUAGE_ID_MAP = {
    "en": 1,
    "ar": 2,
    "bn": 3,
    "de": 4,
    "es": 5,
    "fa": 6,
    "fr": 7,
    "hi": 8,
    "id": 9,
    "it": 10,
    "ja": 11,
    "jv": 12,
    "ko": 13,
    "pt": 14,
    "ru": 15,
    "sw": 16,
    "ta": 17,
    "te": 18,
    "tr": 19,
    "ur": 20,
    "vi": 21,
    "zh": 22,
}


@contextlib.contextmanager
def _logged_startup_step(name: str, **details: Any):
    detail_text = " ".join(f"{key}={value}" for key, value in details.items() if value is not None)
    if detail_text:
        _LOGGER.info("Loading %s %s", name, detail_text)
    else:
        _LOGGER.info("Loading %s", name)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        _LOGGER.exception("Failed loading %s elapsed=%.2fs", name, time.perf_counter() - started)
        raise
    _LOGGER.info("Loaded %s elapsed=%.2fs", name, time.perf_counter() - started)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _configured_api_key() -> str:
    return os.environ.get("API_KEY", "").strip()


def _request_api_key(request: Request) -> str:
    return (request.headers.get("X-Api-Key") or request.query_params.get("api_key") or "").strip()


def _is_browser_demo_path(path: str) -> bool:
    return path == "/qwen3/demo" or path.startswith("/qwen3/demo/")


def _render_llms_txt(request: Request) -> str:
    template = LLMS_TEMPLATE_PATH.read_text(encoding="utf-8")
    base_url = str(request.base_url).rstrip("/")
    return template.replace("{{BASE_URL}}", base_url)


def _scrub_api_key_query_param(request: Request) -> None:
    query_string = request.scope.get("query_string", b"")
    if b"api_key=" not in query_string:
        return
    query_items = [
        (key, value)
        for key, value in parse_qsl(query_string.decode("latin1"), keep_blank_values=True)
        if key != "api_key"
    ]
    request.scope["query_string"] = urlencode(query_items, doseq=True).encode("latin1")

class ModelConfig(BaseModel):
    """Per-model configuration override."""

    # Speaker mappings: {"speaker_label": speaker_id_or_null}
    # Use null for single-speaker models that don't need a speaker ID
    speakers: dict[str, Optional[int]] = Field(default_factory=dict)
    # Override espeak voice for phonemization (e.g., "en-us", "en-gb")
    phoneme_voice: Optional[str] = None


class RootVoiceConfig(BaseModel):
    """A configured public voice id backed directly by a Sparrow model."""

    voice_id: str
    model: str
    speaker: Optional[str] = None


class EngineEnableConfig(BaseModel):
    """Global engine switches. Disabled engines are not mounted or loaded."""

    pipertts: bool = Field(default_factory=lambda: _env_bool("PIPER_TTS_ENABLED", True))
    qwen3: bool = Field(default_factory=lambda: _env_bool("QWEN_TTS_ENABLED", True))
    matcha: bool = Field(default_factory=lambda: _env_bool("MATCHA_TTS_ENABLED", False))
    seed_vc: bool = Field(default_factory=lambda: _env_bool("SEED_VC_ENABLED", True))
    rvc: bool = Field(default_factory=lambda: _env_bool("RVC_ENABLED", False))


class PiperTTSConfig(BaseModel):
    """Sparrow/VITS model cache and routing configuration."""

    enabled: bool = Field(default_factory=lambda: _env_bool("PIPER_TTS_ENABLED", True))
    default_model: str = DEFAULT_MODEL
    models: list[str] = Field(default_factory=list)
    max_models_in_cache: int = Field(1, ge=1)
    preload_models: list[str] = Field(default_factory=list)
    model_priority: list[str] = Field(default_factory=list)
    lang_speaker_map: dict[str, str] = Field(default_factory=dict)
    root_voices: dict[str, RootVoiceConfig] = Field(default_factory=dict)
    model_config_overrides: dict[str, ModelConfig] = Field(default_factory=dict, alias="model_config")


class QwenTTSConfig(qwen3.QwenSettings):
    """Qwen3 TTS engine configuration."""

    enabled: bool = Field(default_factory=lambda: _env_bool("QWEN_TTS_ENABLED", True))


class MatchaConfig(BaseModel):
    """Matcha-TTS backend configuration."""

    enabled: bool = Field(default_factory=lambda: _env_bool("MATCHA_TTS_ENABLED", False))
    preload: bool = Field(default_factory=lambda: _env_bool("MATCHA_TTS_PRELOAD", True))
    device: str = Field(default_factory=lambda: os.environ.get("MATCHA_TTS_DEVICE", "cuda:2"))
    checkpoint: str = Field(default_factory=lambda: os.environ.get("MATCHA_TTS_CHECKPOINT", ""))
    icbpe_vocab_path: str = Field(default_factory=lambda: os.environ.get("MATCHA_TTS_ICBPE_VOCAB", ""))
    phoneme_vocab_path: str = Field(default_factory=lambda: os.environ.get("MATCHA_TTS_PHONEME_VOCAB", ""))
    filelist_path: str = Field(default_factory=lambda: os.environ.get("MATCHA_TTS_FILELIST", ""))
    max_batch_size: int = Field(default_factory=lambda: int(os.environ.get("MATCHA_TTS_MAX_BATCH_SIZE", "128")), ge=1)
    batch_wait_ms: float = Field(default_factory=lambda: float(os.environ.get("MATCHA_TTS_BATCH_WAIT_MS", "10")), ge=0)
    n_timesteps: int = Field(default_factory=lambda: int(os.environ.get("MATCHA_TTS_STEPS", "32")), ge=1)
    temperature: float = Field(default_factory=lambda: float(os.environ.get("MATCHA_TTS_TEMPERATURE", "0.667")), ge=0)
    length_scale: float = Field(default_factory=lambda: float(os.environ.get("MATCHA_TTS_LENGTH_SCALE", "1.0")), gt=0)
    sample_rate: int = Field(default_factory=lambda: int(os.environ.get("MATCHA_TTS_SAMPLE_RATE", "24000")), ge=1)
    vocoder: str = Field(default_factory=lambda: os.environ.get("MATCHA_TTS_VOCODER", "vocos24k"))
    n_mels: int = Field(default_factory=lambda: int(os.environ.get("MATCHA_TTS_N_MELS", "100")), ge=1)
    n_fft: int = Field(default_factory=lambda: int(os.environ.get("MATCHA_TTS_N_FFT", "1024")), ge=1)
    hop_length: int = Field(default_factory=lambda: int(os.environ.get("MATCHA_TTS_HOP_LENGTH", "256")), ge=1)
    win_length: int = Field(default_factory=lambda: int(os.environ.get("MATCHA_TTS_WIN_LENGTH", "1024")), ge=1)
    f_min: float = Field(default_factory=lambda: float(os.environ.get("MATCHA_TTS_F_MIN", "0")))
    f_max: Optional[float] = Field(
        default_factory=lambda: None
        if os.environ.get("MATCHA_TTS_F_MAX", "none").lower() in {"none", "null", ""}
        else float(os.environ.get("MATCHA_TTS_F_MAX", "none"))
    )
    mel_mean: float = Field(default_factory=lambda: float(os.environ.get("MATCHA_TTS_MEL_MEAN", "-5.772607")))
    mel_std: float = Field(default_factory=lambda: float(os.environ.get("MATCHA_TTS_MEL_STD", "2.773259")))


class SeedVCConfig(BaseModel):
    """Embedded Seed-VC voice conversion configuration."""

    enabled: bool = Field(default_factory=lambda: _env_bool("SEED_VC_ENABLED", True))
    preload: bool = Field(default_factory=lambda: _env_bool("SEED_VC_PRELOAD", False))
    device: str = Field(default_factory=lambda: os.environ.get("SEED_VC_DEVICE", "cuda:0"))
    runtime_root: str = Field(default_factory=lambda: os.environ.get("SEED_VC_RUNTIME_ROOT", "src/seed_vc_runtime"))
    root: str = Field(default_factory=lambda: os.environ.get("SEED_VC_ROOT", "data/seed-vc"))
    embeddings_hdf5_path: str = Field(
        default_factory=lambda: os.environ.get("SEED_VC_EMBEDDINGS_HDF5", "data/seed-vc/embeddings/vtts_embeddings.h5")
    )
    tmp_dir: str = Field(default_factory=lambda: os.environ.get("SEED_VC_TMP_DIR", "data/seed-vc/tmp"))
    output_dir: str = Field(default_factory=lambda: os.environ.get("SEED_VC_OUTPUT_DIR", "data/seed-vc/output"))
    voice_samples_dir: str = Field(
        default_factory=lambda: os.environ.get("SEED_VC_VOICE_SAMPLES_DIR", "data/seed-vc/voice-samples")
    )
    fp16: bool = Field(default_factory=lambda: _env_bool("SEED_VC_FP16", True))
    embedding_cache_size: int = Field(default_factory=lambda: int(os.environ.get("SEED_VC_EMBEDDING_CACHE_SIZE", "256")), ge=1)
    estimator_cache_batch_size: int = Field(
        default_factory=lambda: int(os.environ.get("SEED_VC_ESTIMATOR_CACHE_BATCH_SIZE", "8")),
        ge=1,
    )
    estimator_cache_seq_length: int = Field(
        default_factory=lambda: int(os.environ.get("SEED_VC_ESTIMATOR_CACHE_SEQ_LENGTH", "4096")),
        ge=1,
    )
    max_chunk_batch_size: int = Field(
        default_factory=lambda: int(os.environ.get("SEED_VC_MAX_CHUNK_BATCH_SIZE", "1")),
        ge=1,
        le=64,
    )


class RVCConfig(BaseModel):
    """RVC voice conversion engine configuration."""

    enabled: bool = Field(default_factory=lambda: _env_bool("RVC_ENABLED", False))
    preload: bool = Field(default_factory=lambda: _env_bool("RVC_PRELOAD", False))
    cache_size: int = Field(default_factory=lambda: int(os.environ.get("RVC_CACHE_SIZE", "5")), ge=1)
    preload_models: list[str] = Field(default_factory=list)
    default_f0_method: str = Field(default_factory=lambda: os.environ.get("RVC_F0_METHOD", "rmvpe"))
    default_pitch: int = Field(default_factory=lambda: int(os.environ.get("RVC_PITCH", "0")))
    default_index_rate: float = Field(default_factory=lambda: float(os.environ.get("RVC_INDEX_RATE", "0.0")))
    default_rms_mix_rate: float = Field(default_factory=lambda: float(os.environ.get("RVC_RMS_MIX_RATE", "0.25")))
    default_protect: float = Field(default_factory=lambda: float(os.environ.get("RVC_PROTECT", "0.33")))


class ServerConfig(BaseModel):
    """Server configuration."""

    engines: EngineEnableConfig = Field(default_factory=EngineEnableConfig)
    pipertts: PiperTTSConfig = Field(default_factory=PiperTTSConfig)
    qwen: QwenTTSConfig = Field(default_factory=QwenTTSConfig)
    matcha: MatchaConfig = Field(default_factory=MatchaConfig)
    seed_vc: SeedVCConfig = Field(default_factory=SeedVCConfig)
    rvc: RVCConfig = Field(default_factory=RVCConfig)


class SparrowSynthesizeOptions(BaseModel):
    """Sparrow/VITS-specific synthesis controls."""

    model_config = {"extra": "forbid"}

    acoustic_noise_scale: Optional[float] = Field(None, description="Acoustic decoder sampling noise")
    length_scale: Optional[float] = Field(None, description="Speech rate multiplier (>1 = slower)")
    duration_noise_scale: Optional[float] = Field(None, description="Stochastic duration predictor noise")
    duration_sdp_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Duration predictor blend: 0 = deterministic DP, 1 = stochastic DP; default model config is 0.2",
    )


class SynthesizeRequest(BaseModel):
    """Request body for text synthesis."""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    text: Optional[str] = Field(None, description="Plain text to synthesize (mutually exclusive with ssml)")
    ssml: Optional[str] = Field(None, description="SSML to synthesize, must be wrapped in <speak> tags (mutually exclusive with text)")
    voice_id: Optional[str] = Field(None, description="Public voice id from data/seed-vc/voice_ids.txt, e.g. msa.en-US.AvaMultilingual")
    sample_url: Optional[str] = Field(None, description="Reference sample URL; output is converted to this voice with Seed-VC")
    language: Optional[str] = Field(None, description="Force full locale for the entire input, e.g. en-GB")
    style: Optional[str] = Field(None, description="Seed-VC speech style for voice_id synthesis")
    style_intensity: Optional[float] = Field(None, alias="styleIntensity", description="Seed-VC speech style intensity")
    options: Optional[SparrowSynthesizeOptions] = Field(None, description="Sparrow/VITS-specific synthesis options")
    format: Literal["wav", "mp3"] = Field("wav", description="Output audio format (wav or mp3)")
    neural: bool = Field(True, description="Use neural heteronym disambiguation for more accurate pronunciation of ambiguous words")


class BatchSynthesizeInputItem(BaseModel):
    """One item in a /synthesize/batch request."""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    text: Optional[str] = Field(None, description="Plain text to synthesize")
    ssml: Optional[str] = Field(None, description="SSML input is not supported for batched synthesis")
    voice_id: Optional[str] = Field(None, description="Public voice id from data/seed-vc/voice_ids.txt, e.g. msa.en-US.AvaMultilingual")
    sample_url: Optional[str] = Field(None, description="Reference sample URL; output is converted to this voice with Seed-VC")
    language: Optional[str] = Field(None, description="Force full locale for this item, e.g. en-GB")
    model: Optional[str] = Field(None, description="Model to use for direct Sparrow batching")
    style: Optional[str] = Field(None, description="Seed-VC speech style for voice_id synthesis")
    style_intensity: Optional[float] = Field(None, alias="styleIntensity", description="Seed-VC speech style intensity")
    options: Optional[SparrowSynthesizeOptions] = Field(None, description="Sparrow/VITS-specific synthesis options")
    format: Literal["wav", "mp3"] = Field("wav", description="Output audio encoding")
    neural: bool = Field(True, description="Use neural heteronym disambiguation")


class BatchSynthesizeRequest(BaseModel):
    """Request body for real batched synthesis with per-item /synthesize inputs."""

    model_config = {"extra": "forbid"}

    items: list[BatchSynthesizeInputItem] = Field(..., min_length=1, max_length=64, description="Synthesize inputs")


@dataclass(frozen=True)
class _SharedBatchSynthesizeRequest:
    texts: list[str]
    voice_id: str | None = None
    sample_url: str | None = None
    language: str | None = None
    model: str | None = None
    style: str | None = None
    style_intensity: float | None = None
    options: SparrowSynthesizeOptions | None = None
    format: Literal["wav", "mp3"] = "wav"
    neural: bool = True


class BatchSynthesizeItem(BaseModel):
    text: str
    audio_base64: str
    sample_rate: int
    audio_seconds: float


class BatchSynthesizeResponse(BaseModel):
    items: list[BatchSynthesizeItem]
    count: int
    model: str
    speaker: Optional[str]
    wall_seconds: float
    audio_seconds_total: float
    rtf: float


class MatchaSynthesizeRequest(BaseModel):
    """Request body for the temporary Matcha backend."""

    text: str
    language: str = Field("en", description="Language code used for phonemization and speaker/language conditioning")
    format: Literal["wav", "json"] = "wav"
    input_type: Literal["aligned"] = "aligned"
    speaker_id: Optional[int] = Field(None, description="Override language speaker id; 0 means auto")
    neural: bool = True
    steps: Optional[int] = None
    temperature: Optional[float] = None
    length_scale: Optional[float] = None


class SeedVCRequest(BaseModel):
    """Request body compatible with the Seed-VC /vc endpoint."""

    audio: str = Field(..., description="Base64 encoded source audio")
    reference_url: Optional[str] = Field(None, description="Reference voice URL; optional when cached sample/embedding exists")
    id: str = Field(..., description="Voice id used for cached reference audio/embeddings")
    style: Optional[str] = "general"
    intensity: Optional[float] = 1.0
    preset: Optional[str] = None
    remove_glitches: Optional[bool] = False


class SeedVCBatchRequest(BaseModel):
    """Batch Seed-VC request.

    The initial batched path is intended for a Matcha -> one target timbre
    pipeline, so all items share the same target voice settings.
    """

    items: list[SeedVCRequest] = Field(..., min_length=1)
    max_chunk_batch_size: int = Field(1, ge=1, le=64)


class SeedVCFindVoiceRequest(BaseModel):
    reference_url: str
    id: str


class SeedVCEnhanceRequest(BaseModel):
    reference_url: str
    id: str


class RVCConvertRequest(BaseModel):
    """Request body for RVC voice conversion."""

    model_config = {"populate_by_name": True}

    audio: str = Field(..., description="Base64 encoded source audio")
    model: str = Field(..., alias="model_name", description="RVC model filename (e.g., 'mrbeast.pth', 'trump.pth')")
    f0_method: str = Field("rmvpe", description="Pitch extraction method (pm, harvest, crepe, rmvpe)")
    pitch: int = Field(0, description="Pitch shift in semitones")
    index_rate: float = Field(0.0, description="FAISS index blending rate 0-1 (0 = no index)")
    rms_mix_rate: float = Field(0.25, description="Volume envelope mix rate 0-1")
    protect: float = Field(0.33, description="Protect voiceless consonants 0-0.5")
    format: Literal["wav", "mp3"] = Field("wav", description="Output audio format (wav or mp3)")


class RVCBatchConvertItem(BaseModel):
    """One source item for batched RVC conversion."""

    audio: str = Field(..., description="Base64 encoded source audio")


class RVCBatchConvertRequest(BaseModel):
    """Request body for real batched RVC voice conversion."""

    model_config = {"populate_by_name": True}

    items: list[RVCBatchConvertItem] = Field(..., min_length=1, max_length=64)
    model: str = Field(..., alias="model_name", description="RVC model filename (e.g., 'mrbeast.pth')")
    f0_method: str = Field("rmvpe", description="Pitch extraction method (pm, harvest, crepe, rmvpe)")
    pitch: int = Field(0, description="Pitch shift in semitones")
    index_rate: float = Field(0.0, description="FAISS index blending rate 0-1 (0 = no index)")
    rms_mix_rate: float = Field(0.25, description="Volume envelope mix rate 0-1")
    protect: float = Field(0.33, description="Protect voiceless consonants 0-0.5")
    format: Literal["wav", "mp3"] = Field("wav", description="Output audio format (wav or mp3)")


class RVCBatchConvertResponseItem(BaseModel):
    audio_base64: str
    sample_rate: int


class RVCBatchConvertResponse(BaseModel):
    items: list[RVCBatchConvertResponseItem]
    count: int


class SynthesizeVoiceInfo(BaseModel):
    """Synthesize voice catalog entry."""

    voice_id: str
    locale: Optional[str] = None


class SynthesizeVoicesResponse(BaseModel):
    """Supported voices and locales for synthesis."""

    locales: list[str]
    voices: list[SynthesizeVoiceInfo]


def _text_length(value: str | None) -> int:
    return len(value.strip()) if value else 0


def _log_synthesize_request(
    *,
    route: str,
    method: str,
    status: str,
    started: float,
    count: int,
    text_chars: int,
    ssml_chars: int = 0,
    input_data: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    log_data = {
        "route": route,
        "method": method,
        "status": status,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "count": count,
        "text_chars": text_chars,
        "ssml_chars": ssml_chars,
        "input": input_data or {},
    }
    if error is not None:
        log_data["error"] = error
    _LOGGER.info("Synthesize request: %s", json.dumps(log_data, ensure_ascii=False))


def _request_model_input(request: BaseModel, **query_fields: Any) -> dict[str, Any]:
    input_data = request.model_dump(by_alias=True)
    input_data.update({key: value for key, value in query_fields.items() if value is not None})
    return input_data


def _log_synthesize_batch_stage(stage: str, **data: Any) -> None:
    _LOGGER.debug(
        "Synthesize batch stage: %s",
        json.dumps({"stage": stage, **data}, ensure_ascii=False, default=str),
    )


def _log_synthesize_batch_summary(**data: Any) -> None:
    _LOGGER.info(
        "Synthesize batch summary: %s",
        json.dumps(data, ensure_ascii=False, default=str),
    )


# Global state
_inference_cache: OrderedDict[str, PiperInference] = OrderedDict()
_server_config: ServerConfig = ServerConfig()
_speaker_routes: dict[str, tuple[str, Optional[int]]] = {}  # speaker -> (model, speaker_id or None)
_lang_speaker_map: dict[str, str] = {}  # canonical locale -> speaker
_splitter: MultilingualSplitter | None = None
_splitter_languages: tuple[str, ...] | None = None
_matcha_backend: "ProductionMatchaBackend | None" = None
_matcha_batcher: "ProductionMatchaBatcher | None" = None
_seed_vc_backend: "_SeedVCBackend | None" = None
_rvc_backend: "RVCBackend | None" = None
_seed_vc_supported_voice_ids: set[str] | None = None
_seed_vc_voice_ids: set[str] | None = None

_inference_counter = 0
_CLEANUP_EVERY = 1


def _maybe_cleanup_gpu() -> None:
    """Call torch.cuda.empty_cache() + gc.collect() every N inference requests."""
    global _inference_counter
    _inference_counter += 1
    if _inference_counter % _CLEANUP_EVERY == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _normalize_locale(lang: str) -> str:
    """Normalize locale code to canonical BCP 47 format (e.g., en-us -> en-US)."""
    parts = lang.lower().split("-")
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1].upper()}"
    return parts[0]


def _normalize_locale_with_region(lang: str) -> str:
    """Normalize locale code and keep only the primary language and region."""
    parts = lang.lower().split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1].upper()}"
    return parts[0]


def _extract_locale_from_voice_id(voice_id: str) -> str | None:
    parts = voice_id.split(".")
    if len(parts) < 2:
        return None
    candidate = parts[1]
    if candidate == "en":
        return "en-US"
    if "-" in candidate:
        return _normalize_locale_with_region(candidate)
    return None


def _is_supported_sparrow_locale(locale: str) -> bool:
    """Return True if Sparrow can resolve this locale to a configured route."""
    if not locale:
        return False
    normalized = _normalize_locale_with_region(locale)
    resolved_speaker = _lang_speaker_map.get(normalized, normalized)
    return resolved_speaker in _speaker_routes


def _supported_sparrow_locales() -> set[str]:
    """Return all locale labels that can be forced to Sparrow."""
    locales: set[str] = set()
    for locale, speaker in _lang_speaker_map.items():
        normalized_locale = _normalize_locale_with_region(locale)
        if "-" in normalized_locale and speaker in _speaker_routes:
            locales.add(normalized_locale)

    for speaker in _speaker_routes.keys():
        if not speaker:
            continue
        normalized_speaker = _normalize_locale_with_region(speaker)
        if "-" in normalized_speaker:
            locales.add(normalized_speaker)

    return locales


def _supported_sparrow_language_codes() -> set[str]:
    """Return base language codes routable by the configured Sparrow models."""
    languages: set[str] = set()
    for locale in _supported_sparrow_locales():
        languages.add(_get_base_language(locale))
    for speaker in _speaker_routes.keys():
        if speaker:
            languages.add(_get_base_language(speaker))
    return {language for language in languages if language and language != "und"}


def _get_multilingual_splitter() -> MultilingualSplitter:
    """Build a splitter constrained to languages supported by Sparrow routing."""
    global _splitter, _splitter_languages
    languages = tuple(sorted(_supported_sparrow_language_codes()))
    if _splitter is None or _splitter_languages != languages:
        _splitter = MultilingualSplitter(languages=list(languages) if languages else None)
        _splitter_languages = languages
    return _splitter


def _routable_detected_language(language: str, main_lang: str) -> str:
    """Clamp auto-detected language to a Sparrow-routable language."""
    normalized = _normalize_locale_with_region(language)
    if _is_supported_sparrow_locale(normalized):
        return normalized
    normalized_main = _normalize_locale_with_region(main_lang)
    if _is_supported_sparrow_locale(normalized_main):
        return normalized_main
    if _is_supported_sparrow_locale("en"):
        return "en"
    return next(iter(_speaker_routes.keys()), "en")


def _resolve_forced_language(locale: str) -> tuple[str, str, str]:
    """Resolve a forced locale into speaker/model tuple and return normalized locale."""
    normalized = _normalize_locale_with_region(locale)
    speaker, model_name = _resolve_speaker_and_model(normalized, explicit=True)
    return normalized, speaker, model_name


def _get_base_language(speaker_or_locale: str) -> str:
    """Extract base language from a speaker/locale string (e.g., 'en-GB' -> 'en')."""
    return speaker_or_locale.lower().split("-")[0]


def _load_config() -> ServerConfig:
    """Load server configuration from local/server.json."""
    if not CONFIG_PATH.exists():
        config = ServerConfig()
        qwen3.apply_env_overrides(config.qwen)
        return config
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    config = ServerConfig(**data)
    qwen3.apply_env_overrides(config.qwen)
    return config


def _engine_enabled(engine: Literal["pipertts", "qwen3", "matcha", "seed_vc", "rvc"], config: ServerConfig | None = None) -> bool:
    cfg = config or _server_config
    return bool(getattr(cfg.engines, engine))


def _find_checkpoint(model_dir: Path) -> Path | None:
    """Find the most recent checkpoint in a model directory."""
    if not model_dir.exists():
        return None
    checkpoints = list(model_dir.glob("*.ckpt"))
    if checkpoints:
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    return None


def _list_available_models() -> list[str]:
    """List all available models in the data directory."""
    if not DATA_DIR.exists():
        return []
    models = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and (d / "config.json").exists():
            models.append(d.name)
    return sorted(models)


def _allowed_models() -> list[str]:
    """Configured Sparrow/VITS models."""
    if not _engine_enabled("pipertts"):
        return []
    return _server_config.pipertts.models or _list_available_models()


def _is_model_allowed(model: str) -> bool:
    """Check whether a model is configured for this server."""
    return model in _allowed_models()


def _append_unique(items: list[str], value: str | None) -> None:
    if value and value not in items:
        items.append(value)


def _required_piper_models() -> list[str]:
    """All Sparrow/VITS models that must be resident after startup."""
    if not _engine_enabled("pipertts"):
        return []

    models: list[str] = []
    for model in _server_config.pipertts.preload_models:
        _append_unique(models, model)
    for model in _server_config.pipertts.model_priority:
        _append_unique(models, model)
    for model in _server_config.pipertts.models:
        _append_unique(models, model)
    _append_unique(models, _server_config.pipertts.default_model)

    if not models:
        models = _allowed_models()
    else:
        for model in _allowed_models():
            _append_unique(models, model)
    return models


def _get_model_speakers(model: str) -> dict[str, int]:
    """Read a model's native speaker map from config without loading weights."""
    config_path = DATA_DIR / model / "config.json"
    if not config_path.exists():
        raise ValueError(f"Model config not found: {model}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {str(label): int(sid) for label, sid in (data.get("speaker_id_map") or {}).items()}


def _enforce_cache_limit() -> None:
    """Evict least-recently-used models until the cache is within its limit."""
    # Startup loads every configured Sparrow model. The historical cache limit is
    # kept as a lower bound for compatibility, not as permission to evict models
    # that the process is expected to serve without lazy reloads.
    limit = max(_server_config.pipertts.max_models_in_cache, len(_required_piper_models()))
    while len(_inference_cache) > limit:
        evicted, _ = _inference_cache.popitem(last=False)
        _LOGGER.info("Evicted model from cache: %s", evicted)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_model(model: str) -> PiperInference:
    """Load a model (used internally, raises ValueError instead of HTTPException)."""
    if not _engine_enabled("pipertts"):
        raise ValueError("PiperTTS backend is disabled")
    if not _is_model_allowed(model):
        raise ValueError(f"Model is not configured for this server: {model}")

    model_dir = DATA_DIR / model
    config_path = model_dir / "config.json"
    checkpoint_path = _find_checkpoint(model_dir)

    if not config_path.exists():
        raise ValueError(f"Model config not found: {model}")
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found for model: {model}")

    _LOGGER.info(
        "Loading Sparrow model model=%s checkpoint=%s config=%s",
        model,
        checkpoint_path,
        config_path,
    )
    started = time.perf_counter()
    try:
        inference = PiperInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
    except Exception:
        _LOGGER.exception("Failed loading Sparrow model model=%s elapsed=%.2fs", model, time.perf_counter() - started)
        raise
    _inference_cache[model] = inference
    _enforce_cache_limit()
    _LOGGER.info(
        "Loaded Sparrow model model=%s speakers=%d elapsed=%.2fs",
        model,
        len(getattr(inference, "speakers", {}) or {}),
        time.perf_counter() - started,
    )
    return inference


def _get_inference(model: str) -> PiperInference:
    """Get an already loaded inference instance for a model."""
    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    if model in _inference_cache:
        inference = _inference_cache.pop(model)
        _inference_cache[model] = inference
        return inference
    if _is_model_allowed(model):
        raise HTTPException(status_code=503, detail=f"Model was not loaded at startup: {model}")
    raise HTTPException(status_code=404, detail=f"Model is not configured for this server: {model}")


def _preload_models(models: list[str], *, strict: bool = False) -> None:
    """Preload specified models into the cache."""
    errors: list[str] = []
    for model in models:
        if model in _inference_cache:
            _LOGGER.info("Model already loaded: %s", model)
            continue
        _LOGGER.info("Preloading model: %s", model)
        try:
            _load_model(model)
            _LOGGER.info("Loaded model: %s", model)
        except ValueError as e:
            message = f"{model}: {e}"
            if strict:
                errors.append(message)
            else:
                _LOGGER.warning("Failed to preload model %s: %s", model, e)
    if errors:
        raise RuntimeError("Failed to preload Sparrow models: " + "; ".join(errors))


def _preload_piper_text_models() -> None:
    """Load Sparrow text-side runtime models required by normal synthesis."""
    semantic_count = 0
    for inference in _inference_cache.values():
        if getattr(inference, "use_bert", False):
            inference.warmup_semantic()
            semantic_count += 1

    device = None
    if _inference_cache:
        first_inference = next(iter(_inference_cache.values()))
        device = str(first_inference.device)

    from ..piper.heteronym import get_resolver

    resolver = get_resolver(device=device)
    resolver.load()
    _LOGGER.info(
        "Loaded Sparrow text models semantic_models=%d heteronym_device=%s",
        semantic_count,
        device or "auto",
    )


def _build_speaker_routes(model_priority: list[str]) -> dict[str, tuple[str, Optional[int]]]:
    """Build speaker routing table based on model priority.

    For each speaker, the first model in the priority list that has that speaker wins.
    Uses model_config overrides first, then falls back to model config metadata.
    """
    routes: dict[str, tuple[str, Optional[int]]] = {}

    for model_name in model_priority:
        if not _is_model_allowed(model_name):
            raise RuntimeError(f"Model {model_name!r} is in priority list but not configured for this server")

        # Check for config override first (useful for single-speaker models with empty labels)
        model_cfg = _server_config.pipertts.model_config_overrides.get(model_name)
        if model_cfg and model_cfg.speakers:
            for speaker, speaker_id in model_cfg.speakers.items():
                if speaker not in routes:
                    routes[speaker] = (model_name, speaker_id)
                    _LOGGER.debug("Routing speaker '%s' -> model '%s' (id=%s) [config override]", speaker, model_name, speaker_id)
        else:
            # Use model's native speaker map
            speakers = _get_model_speakers(model_name)

            for speaker, speaker_id in speakers.items():
                if speaker and speaker not in routes:  # Skip empty speaker labels
                    routes[speaker] = (model_name, speaker_id)
                    _LOGGER.debug("Routing speaker '%s' -> model '%s' (id=%d)", speaker, model_name, speaker_id)

    return routes


def _resolve_speaker_and_model(input_speaker: str | None, *, explicit: bool = False) -> tuple[str | None, str]:
    """Resolve speaker to actual speaker label and model name.

    Simple two-step lookup:
    1. Check lang_speaker_map for alias resolution (e.g., "en-US" → "en")
    2. Check speaker_routes for model selection

    Returns (speaker, model_name).

    If ``explicit`` is True, unresolved speakers/locales are rejected with 400.
    """
    if input_speaker is None:
        return None, _server_config.pipertts.default_model

    # Step 1: Resolve alias through lang_speaker_map
    normalized = _normalize_locale_with_region(input_speaker)
    speaker = _lang_speaker_map.get(normalized, normalized)

    # Step 2: Find model in speaker_routes
    if speaker in _speaker_routes:
        model_name, _ = _speaker_routes[speaker]
        return speaker, model_name

    if explicit:
        supported = sorted(set(_speaker_routes.keys()) | set(_lang_speaker_map.keys()))
        if not supported:
            detail = f"Unsupported speaker or locale {input_speaker!r}; no Piper speakers are available"
        else:
            detail = (
                f"Unsupported speaker or locale {input_speaker!r}; "
                f"supported speakers/locales: {supported}"
            )
        raise HTTPException(status_code=400, detail=detail)

    # Fallback to default model
    return speaker, _server_config.pipertts.default_model


def _synthesize_multilingual(
    text: str,
    primary_speaker: Optional[str] = None,
    language: Optional[str] = None,
    noise_scale: Optional[float] = None,
    length_scale: Optional[float] = None,
    noise_w: Optional[float] = None,
    neural: bool = True,
) -> tuple[np.ndarray, int]:
    """Synthesize multilingual text using multiple models.

    Args:
        text: Text to synthesize.
        primary_speaker: If set, use this speaker for segments matching its base language
                        (e.g., "en-GB" applies to "en" segments only).
        language: If set, force the whole text to this locale.

    Returns (audio, sample_rate).
    """
    if language is not None:
        _resolve_forced_language(language)

    synth_kwargs = {}
    if noise_scale is not None:
        synth_kwargs["noise_scale"] = noise_scale
    if length_scale is not None:
        synth_kwargs["length_scale"] = length_scale
    if noise_w is not None:
        synth_kwargs["noise_w"] = noise_w

    # First pass: compute routing plan
    routing_plan, _ = _plan_text_segments(
        text,
        primary_speaker,
        forced_language=language,
        validate_primary_speaker=primary_speaker is not None,
    )

    _LOGGER.info("Multilingual routing: %s", json.dumps([
        {**p, "text": p["text"][:50] + ("..." if len(p["text"]) > 50 else "")}
        for p in routing_plan
    ], ensure_ascii=False))

    # Second pass: synthesize
    audio_parts: list[np.ndarray] = []
    sample_rate = 22050

    for plan in routing_plan:
        seg_text = plan["text"]
        speaker = plan["speaker"]
        model_name = plan["model"]

        inference = _get_inference(model_name)
        sample_rate = inference.sample_rate

        internal_speaker = _resolve_internal_speaker(model_name, speaker, inference)

        audio = inference.synthesize_span(seg_text, speaker=internal_speaker, neural=neural, **synth_kwargs)
        audio_parts.append(audio)

    if not audio_parts:
        return np.array([], dtype=np.int16), sample_rate

    if len(audio_parts) == 1:
        return audio_parts[0], sample_rate

    return np.concatenate(audio_parts, axis=0), sample_rate


def _configured_root_voice(name: str) -> RootVoiceConfig | None:
    return _server_config.pipertts.root_voices.get(name)


def _root_voice_name(voice_id: str | None) -> str | None:
    if not voice_id:
        return None
    for name, config in _server_config.pipertts.root_voices.items():
        if config.voice_id == voice_id:
            return name
    return None


def _seed_vc_base_id(embedding_key: str) -> str:
    parts = embedding_key.split(".")
    if len(parts) < 3:
        return embedding_key
    return ".".join(parts[:3])


def _load_seed_vc_voice_ids() -> set[str]:
    global _seed_vc_voice_ids
    if _seed_vc_voice_ids is not None:
        return _seed_vc_voice_ids

    if not SEED_VC_VOICE_IDS_PATH.exists():
        raise RuntimeError(f"Seed-VC voice id list not found: {SEED_VC_VOICE_IDS_PATH}")

    try:
        _seed_vc_voice_ids = {
            line.strip()
            for line in SEED_VC_VOICE_IDS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(f"Failed to load Seed-VC voice id list {SEED_VC_VOICE_IDS_PATH}: {exc}") from exc
    if not _seed_vc_voice_ids:
        raise RuntimeError(f"Seed-VC voice id list is empty: {SEED_VC_VOICE_IDS_PATH}")
    return _seed_vc_voice_ids


def _build_synthesize_voices_catalog() -> tuple[list[str], list[SynthesizeVoiceInfo]]:
    supported_voice_ids = set(cfg.voice_id for cfg in _server_config.pipertts.root_voices.values())

    if _engine_enabled("seed_vc"):
        supported_voice_ids.update(_get_seed_vc_supported_voice_ids())

    supported_locales = _supported_sparrow_locales()
    locales = sorted({
        locale for locale in supported_locales
        if _is_supported_sparrow_locale(locale)
    })
    for voice_id in supported_voice_ids:
        locale = _extract_locale_from_voice_id(voice_id)
        if locale is not None and _is_supported_sparrow_locale(locale):
            locales.append(locale)

    unique_locales = sorted({locale for locale in locales if locale})

    voices = [
        SynthesizeVoiceInfo(
            voice_id=voice_id,
            locale=_extract_locale_from_voice_id(voice_id),
        )
        for voice_id in sorted(supported_voice_ids)
    ]
    return unique_locales, voices


def _get_seed_vc_supported_voice_ids() -> set[str]:
    global _seed_vc_supported_voice_ids
    if _seed_vc_supported_voice_ids is not None:
        return _seed_vc_supported_voice_ids

    supported = {cfg.voice_id for cfg in _server_config.pipertts.root_voices.values()}
    backend = _get_seed_vc_backend()
    emb_ids = {_seed_vc_base_id(key) for key in backend.cached_embeddings.keys()} if backend.cached_embeddings else set()
    if not emb_ids:
        raise RuntimeError("Seed-VC backend loaded without cached embeddings")

    configured_ids = _load_seed_vc_voice_ids()
    all_seed_vc_ids = configured_ids & emb_ids
    missing_embeddings = configured_ids - emb_ids
    if missing_embeddings:
        raise RuntimeError(
            "Seed-VC voice id manifest contains id(s) without cached embeddings: "
            f"count={len(missing_embeddings)} first={sorted(missing_embeddings)[:10]}"
        )
    seed_vc_supported = {
        voice_id
        for voice_id in all_seed_vc_ids
        if (locale := _extract_locale_from_voice_id(voice_id))
        and _is_supported_sparrow_locale(locale)
    }
    if not seed_vc_supported:
        raise RuntimeError(
            "Seed-VC voice catalog resolved to zero supported embedding-backed voices "
            f"(manifest={SEED_VC_VOICE_IDS_PATH}, embeddings={len(emb_ids)})"
        )
    supported.update(seed_vc_supported)

    _seed_vc_supported_voice_ids = supported
    return _seed_vc_supported_voice_ids


def _synth_kwargs_from_request(request: SynthesizeRequest | BatchSynthesizeInputItem | _SharedBatchSynthesizeRequest) -> dict[str, float]:
    synth_kwargs: dict[str, float] = {}
    if request.options is None:
        return synth_kwargs
    if request.options.acoustic_noise_scale is not None:
        synth_kwargs["noise_scale"] = request.options.acoustic_noise_scale
    if request.options.length_scale is not None:
        synth_kwargs["length_scale"] = request.options.length_scale
    if request.options.duration_noise_scale is not None:
        synth_kwargs["noise_w"] = request.options.duration_noise_scale
    if request.options.duration_sdp_ratio is not None:
        synth_kwargs["sdp_ratio"] = request.options.duration_sdp_ratio
    return synth_kwargs


def _seed_vc_style_from_request(request: SynthesizeRequest | BatchSynthesizeInputItem | _SharedBatchSynthesizeRequest) -> tuple[str, float]:
    return request.style or "general", request.style_intensity if request.style_intensity is not None else 1.0


def _seed_vc_style_requested(request: SynthesizeRequest | BatchSynthesizeInputItem | _SharedBatchSynthesizeRequest) -> bool:
    style, intensity = _seed_vc_style_from_request(request)
    return style != "general" or intensity != 1.0


def _seed_vc_sample_id(sample_url: str) -> str:
    return f"synthesize-sample-{hashlib.sha256(sample_url.encode()).hexdigest()[:16]}"


def _seed_vc_chunk_batch_size(backend: SeedVCBackend) -> int:
    return max(1, int(backend.settings.max_chunk_batch_size))


async def _convert_generated_audio_to_sample_batch(
    *,
    source_audios: list[np.ndarray],
    source_sample_rates: list[int],
    sample_url: str,
    output_format: Literal["wav", "mp3"],
) -> tuple[list[tuple[bytes, float]], int]:
    backend = _get_seed_vc_backend()
    sample_request = SeedVCRequest(
        audio="",
        reference_url=sample_url,
        id=_seed_vc_sample_id(sample_url),
        style="general",
        intensity=1.0,
    )
    reference_path = await backend._fetch_sample(sample_request)
    vc_source_rate = backend.sample_rate
    vc_source_audios = [
        _resample_audio(audio, source_rate, vc_source_rate)
        for audio, source_rate in zip(source_audios, source_sample_rates)
    ]
    vc_started = time.perf_counter()
    _log_synthesize_batch_stage(
        "seed_vc_sample_batch_start",
        count=len(vc_source_audios),
        sample_url=sample_url,
        output_format=output_format,
        source_sample_rates=source_sample_rates,
    )
    converted = await asyncio.to_thread(
        backend.convert_generated_audio_reference_batch,
        vc_source_audios,
        vc_source_rate,
        reference_path,
        None,
        output_format,
        _seed_vc_chunk_batch_size(backend),
    )
    _log_synthesize_batch_stage(
        "seed_vc_sample_batch_done",
        count=len(converted),
        wall_seconds=round(time.perf_counter() - vc_started, 6),
        output_sample_rate=backend.sample_rate,
    )
    return converted, backend.sample_rate


def _resolve_internal_speaker(model_name: str, speaker: str | None, inference: PiperInference) -> str | None:
    if speaker is None or not str(speaker).strip() or str(speaker).lower() == "und":
        return None
    model_cfg = _server_config.pipertts.model_config_overrides.get(model_name)
    if model_cfg and speaker in model_cfg.speakers:
        return None if model_cfg.speakers[speaker] is None else speaker
    if speaker in inference.speakers:
        return speaker
    raise HTTPException(status_code=500, detail=f"Speaker {speaker!r} is not available in loaded model {model_name!r}")


def _plan_text_segments(
    text: str,
    primary_speaker: str | None,
    forced_language: str | None = None,
    *,
    validate_primary_speaker: bool = False,
) -> tuple[list[dict[str, Any]], set[str]]:
    splitter = _get_multilingual_splitter()

    if forced_language is not None:
        forced_locale, forced_speaker, forced_model = _resolve_forced_language(forced_language)

    result = splitter.split(text)
    main_lang = result.main_language or "en"
    primary_lang = _get_base_language(primary_speaker) if primary_speaker else None
    if primary_speaker is not None and validate_primary_speaker:
        _resolve_speaker_and_model(primary_speaker, explicit=True)

    segments: list[dict[str, Any]] = []
    languages: set[str] = set()

    if forced_language is not None:
        segment_text = text.strip()
        if segment_text:
            languages.add(forced_locale)
            segments.append({
                "lang": forced_locale,
                "speaker": forced_speaker,
                "model": forced_model,
                "text": segment_text,
            })
        return segments, languages or {forced_locale}

    for segment in result.segments:
        segment_text = segment.text.strip()
        if not segment_text:
            continue
        detected_language = (segment.language if segment.language and segment.language != "und" else main_lang) or "en"
        language = _routable_detected_language(detected_language, main_lang)
        languages.add(language)
        if primary_speaker and _get_base_language(language) == primary_lang:
            speaker, model_name = _resolve_speaker_and_model(
                primary_speaker,
                explicit=validate_primary_speaker,
            )
        else:
            speaker, model_name = _resolve_speaker_and_model(language)
        segments.append(
            {
                "lang": language,
                "speaker": speaker,
                "model": model_name,
                "text": segment_text,
            }
        )

    return segments, languages or {main_lang}


async def synthesize_configured_voice_batch(request: _SharedBatchSynthesizeRequest) -> BatchSynthesizeResponse:
    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    if request.voice_id is None:
        raise HTTPException(status_code=400, detail="voice_id is required for configured voice synthesis")
    if request.sample_url is not None:
        raise HTTPException(status_code=400, detail="Use either 'voice_id' or 'sample_url', not both")
    if request.model is not None:
        raise HTTPException(status_code=400, detail="Use either 'voice_id' or 'model', not both")

    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")

    supported_voice_ids = _get_seed_vc_supported_voice_ids()
    if request.voice_id not in supported_voice_ids:
        supported = sorted(supported_voice_ids)
        raise HTTPException(status_code=400, detail=f"Unsupported voice_id {request.voice_id!r}; supported voices: {supported}")

    forced_language = _normalize_locale_with_region(request.language) if request.language is not None else None
    if forced_language is not None:
        _resolve_forced_language(forced_language)

    voice_name = _root_voice_name(request.voice_id)
    primary_speaker: str | None = None
    style, style_intensity = _seed_vc_style_from_request(request)
    style_requested = _seed_vc_style_requested(request)
    if voice_name == "sparrow":
        convert_all = forced_language == "en-GB"
    elif voice_name == "sparrow_en_gb":
        en_gb_voice = _configured_root_voice("sparrow_en_gb")
        primary_speaker = forced_language if forced_language is not None else (en_gb_voice.speaker if en_gb_voice else "en-GB")
        convert_all = forced_language is not None and _get_base_language(forced_language) != "en"
    else:
        convert_all = True

    if style_requested:
        convert_all = True
    if convert_all:
        try:
            _get_seed_vc_backend()._resolve_exact_cached_embeddings(request.voice_id, style, style_intensity)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()
    synth_kwargs = _synth_kwargs_from_request(request)
    item_segments: list[list[dict[str, Any]]] = []
    convert_item = [convert_all for _ in texts]
    segment_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    for item_idx, text in enumerate(texts):
        segments, languages = _plan_text_segments(
            text,
            primary_speaker,
            forced_language=forced_language,
            validate_primary_speaker=False,
        )
        if voice_name == "sparrow_en_gb" and any(_get_base_language(language) != "en" for language in languages):
            convert_item[item_idx] = True
        item_segments.append(segments)
        for segment_idx, segment in enumerate(segments):
            record = {**segment, "item_idx": item_idx, "segment_idx": segment_idx}
            segment_groups.setdefault(segment["model"], []).append(record)

    if any(convert_item):
        try:
            _get_seed_vc_backend()._resolve_exact_cached_embeddings(request.voice_id, style, style_intensity)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    _log_synthesize_batch_stage(
        "configured_voice_routing",
        voice_id=request.voice_id,
        item_count=len(texts),
        item_segment_counts=[len(segments) for segments in item_segments],
        convert_indices=[idx for idx, value in enumerate(convert_item) if value],
        model_groups=[
            {
                "model": model_name,
                "segment_count": len(records),
                "item_indices": sorted({int(record["item_idx"]) for record in records}),
                "speakers": sorted({str(record["speaker"]) for record in records}),
                "languages": sorted({str(record["lang"]) for record in records}),
            }
            for model_name, records in segment_groups.items()
        ],
    )

    generated_segments: list[list[tuple[np.ndarray, int] | None]] = [
        [None for _ in segments]
        for segments in item_segments
    ]

    for model_name, records in segment_groups.items():
        inference = _get_inference(model_name)
        model_sample_rate = inference.sample_rate
        batch_texts = [record["text"] for record in records]
        batch_speakers = [
            _resolve_internal_speaker(model_name, record["speaker"], inference)
            for record in records
        ]
        batch_started = time.perf_counter()
        _log_synthesize_batch_stage(
            "sparrow_batch_start",
            pipeline="configured_voice",
            voice_id=request.voice_id,
            model=model_name,
            item_count=len(texts),
            segment_count=len(batch_texts),
            batch_size=len(batch_texts),
            speakers=sorted({str(speaker) for speaker in batch_speakers}),
            neural=request.neural,
            synth_kwargs=synth_kwargs,
        )
        batch_audios = await asyncio.to_thread(
            inference.synthesize_batch,
            batch_texts,
            speaker=batch_speakers,
            batch_size=len(batch_texts),
            neural=request.neural,
            **synth_kwargs,
        )
        audio_seconds = sum(float(len(audio)) / model_sample_rate for audio in batch_audios) if model_sample_rate else 0.0
        elapsed = time.perf_counter() - batch_started
        _log_synthesize_batch_stage(
            "sparrow_batch_done",
            pipeline="configured_voice",
            voice_id=request.voice_id,
            model=model_name,
            output_count=len(batch_audios),
            audio_seconds=round(audio_seconds, 6),
            wall_seconds=round(elapsed, 6),
            rtf=round(elapsed / audio_seconds, 6) if audio_seconds else 0.0,
            sample_rate=model_sample_rate,
        )
        for record, audio in zip(records, batch_audios):
            generated_segments[record["item_idx"]][record["segment_idx"]] = (audio, model_sample_rate)

    item_audios: list[np.ndarray] = []
    item_source_sample_rates: list[int] = []
    for segments in generated_segments:
        parts = [segment for segment in segments if segment is not None]
        if not parts:
            item_audios.append(np.zeros(0, dtype=np.int16))
            item_source_sample_rates.append(22050)
        elif len(parts) == 1:
            audio, source_rate = parts[0]
            item_audios.append(audio)
            item_source_sample_rates.append(source_rate)
        else:
            target_rate = parts[0][1]
            item_audios.append(np.concatenate([
                _resample_audio(audio, source_rate, target_rate)
                for audio, source_rate in parts
            ], axis=0))
            item_source_sample_rates.append(target_rate)

    encoded_items: list[bytes | None] = [None for _ in item_audios]
    item_sample_rates = list(item_source_sample_rates)
    item_audio_seconds = [
        (float(len(audio)) / item_sample_rates[idx] if item_sample_rates[idx] else 0.0)
        for idx, audio in enumerate(item_audios)
    ]

    convert_indices = [idx for idx, should_convert in enumerate(convert_item) if should_convert]
    if convert_indices:
        backend = _get_seed_vc_backend()
        try:
            backend._resolve_exact_cached_embeddings(request.voice_id, style, style_intensity)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        vc_source_rate = backend.sample_rate
        vc_source_audios = [
            _resample_audio(item_audios[idx], item_source_sample_rates[idx], vc_source_rate)
            for idx in convert_indices
        ]
        vc_started = time.perf_counter()
        _log_synthesize_batch_stage(
            "seed_vc_voice_batch_start",
            pipeline="configured_voice",
            voice_id=request.voice_id,
            count=len(vc_source_audios),
            item_indices=convert_indices,
            style=style,
            style_intensity=style_intensity,
            output_format=request.format,
        )
        converted = await asyncio.to_thread(
            backend.convert_generated_audio_batch,
            vc_source_audios,
            vc_source_rate,
            request.voice_id,
            style,
            style_intensity,
            None,
            request.format,
            _seed_vc_chunk_batch_size(backend),
            strict_embedding=True,
        )
        _log_synthesize_batch_stage(
            "seed_vc_voice_batch_done",
            pipeline="configured_voice",
            voice_id=request.voice_id,
            output_count=len(converted),
            wall_seconds=round(time.perf_counter() - vc_started, 6),
            output_sample_rate=backend.sample_rate,
        )
        for item_idx, (audio_bytes, audio_seconds) in zip(convert_indices, converted):
            encoded_items[item_idx] = audio_bytes
            item_sample_rates[item_idx] = backend.sample_rate
            item_audio_seconds[item_idx] = audio_seconds

    for idx, audio in enumerate(item_audios):
        if encoded_items[idx] is not None:
            continue
        encoded_items[idx] = (
            _audio_to_mp3_bytes(audio, item_source_sample_rates[idx])
            if request.format == "mp3"
            else _audio_to_wav_bytes(audio, item_source_sample_rates[idx])
        )

    items: list[BatchSynthesizeItem] = []
    audio_seconds_total = 0.0
    for text, audio_bytes, item_sample_rate, audio_seconds in zip(texts, encoded_items, item_sample_rates, item_audio_seconds):
        audio_seconds_total += audio_seconds
        items.append(
            BatchSynthesizeItem(
                text=text,
                audio_base64=base64.b64encode(audio_bytes or b"").decode("ascii"),
                sample_rate=item_sample_rate,
                audio_seconds=audio_seconds,
            )
        )

    wall_seconds = time.perf_counter() - started
    return BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=f"voice_id:{request.voice_id}",
        speaker=primary_speaker,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )


async def _synthesize_configured_voice(request: SynthesizeRequest) -> Response:
    if request.ssml:
        raise HTTPException(status_code=400, detail="voice-id synthesis currently supports plain text only")
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required for voice-id synthesis")
    batch_result = await synthesize_configured_voice_batch(
        _SharedBatchSynthesizeRequest(
            texts=[request.text],
            voice_id=request.voice_id,
            language=request.language,
            style=request.style,
            style_intensity=request.style_intensity,
            options=request.options,
            format=request.format,
            neural=request.neural,
        )
    )
    audio_bytes = base64.b64decode(batch_result.items[0].audio_base64)
    media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
    return Response(content=audio_bytes, media_type=media_type)


def _synthesize_ssml(
    ssml_text: str,
    global_speaker: Optional[str] = None,
    primary_speaker: Optional[str] = None,
    language: Optional[str] = None,
    noise_scale: Optional[float] = None,
    length_scale: Optional[float] = None,
    noise_w: Optional[float] = None,
) -> tuple[np.ndarray, int]:
    """Synthesize SSML text with break and multilingual support.

    Args:
        ssml_text: SSML string to synthesize.
        global_speaker: If set, overrides all segment speakers.
        primary_speaker: If set, use this speaker for segments matching its base language.
        language: If set, force SSML text segments without explicit speaker to this locale.

    Returns (audio, sample_rate).
    """
    splitter = _get_multilingual_splitter()

    if language is not None:
        forced_locale, forced_speaker, forced_model = _resolve_forced_language(language)

    segments = parse_ssml(ssml_text)

    # Extract base language from primary_speaker if provided
    primary_lang = _get_base_language(primary_speaker) if primary_speaker else None
    if primary_speaker is not None:
        _resolve_speaker_and_model(primary_speaker, explicit=True)

    synth_kwargs = {}
    if noise_scale is not None:
        synth_kwargs["noise_scale"] = noise_scale
    if length_scale is not None:
        synth_kwargs["length_scale"] = length_scale
    if noise_w is not None:
        synth_kwargs["noise_w"] = noise_w

    # Build routing plan
    routing_plan: list[dict] = []

    for seg in segments:
        if isinstance(seg, BreakSegment):
            routing_plan.append({"type": "break", "duration_ms": seg.duration_ms})
        elif isinstance(seg, TextSegment):
            seg_text = seg.text.strip()
            if not seg_text:
                continue

            # Determine speaker: global override > segment speaker > primary speaker > auto-detect
            if global_speaker is not None:
                # Global override - applies to ALL segments
                resolved_speaker, model_name = _resolve_speaker_and_model(global_speaker, explicit=True)
                routing_plan.append({
                    "type": "text",
                    "speaker": resolved_speaker,
                    "model": model_name,
                    "text": seg_text,
                })
            elif seg.speaker is not None:
                # Segment-level speaker from <voice name="...">
                resolved_speaker, model_name = _resolve_speaker_and_model(seg.speaker, explicit=True)
                routing_plan.append({
                    "type": "text",
                    "speaker": resolved_speaker,
                    "model": model_name,
                    "text": seg_text,
                })
            else:
                # Auto-detect: run through multilingual splitter
                if language is not None:
                    lang = forced_locale
                    routing_plan.append({
                        "type": "text",
                        "lang": lang,
                        "speaker": forced_speaker,
                        "model": forced_model,
                        "text": seg_text,
                    })
                    continue

                result = splitter.split(seg_text)
                main_lang = result.main_language or "en"

                for lang_seg in result.segments:
                    lang_text = lang_seg.text.strip()
                    if not lang_text:
                        continue

                    detected_lang = (lang_seg.language if lang_seg.language and lang_seg.language != "und" else main_lang) or "en"
                    lang = _routable_detected_language(detected_lang, main_lang)

                    # Use primary_speaker if language matches, otherwise normal resolution
                    if primary_speaker and _get_base_language(lang) == primary_lang:
                        speaker, model_name = _resolve_speaker_and_model(primary_speaker, explicit=True)
                    else:
                        speaker, model_name = _resolve_speaker_and_model(lang)

                    routing_plan.append({
                        "type": "text",
                        "lang": lang,
                        "speaker": speaker,
                        "model": model_name,
                        "text": lang_text,
                    })

    # Log routing plan (text segments only, truncated)
    log_plan = []
    for p in routing_plan:
        if p["type"] == "text":
            log_plan.append({
                **{k: v for k, v in p.items() if k != "text"},
                "text": p["text"][:50] + ("..." if len(p["text"]) > 50 else ""),
            })
        else:
            log_plan.append(p)
    _LOGGER.info("SSML routing: %s", json.dumps(log_plan, ensure_ascii=False))

    # Synthesize
    audio_parts: list[np.ndarray] = []
    sample_rate = 22050

    for plan in routing_plan:
        if plan["type"] == "break":
            silence = generate_silence(plan["duration_ms"], sample_rate)
            audio_parts.append(silence)
        elif plan["type"] == "text":
            speaker = plan["speaker"]
            model_name = plan["model"]

            inference = _get_inference(model_name)
            sample_rate = inference.sample_rate

            speaker = _resolve_internal_speaker(model_name, speaker, inference)

            _LOGGER.info("Synthesizing: speaker=%s, text=%r, synth_kwargs=%s", speaker, plan["text"], synth_kwargs)
            audio = inference.synthesize_span(plan["text"], speaker=speaker, **synth_kwargs)
            audio_parts.append(audio)

    if not audio_parts:
        return np.array([], dtype=np.int16), sample_rate

    if len(audio_parts) == 1:
        return audio_parts[0], sample_rate

    return np.concatenate(audio_parts, axis=0), sample_rate


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes."""
    pcm_audio = _audio_to_pcm16(audio)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_audio.tobytes())
    return buffer.getvalue()


def _audio_to_mp3_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to MP3 bytes with highest quality settings."""
    pcm_audio = _audio_to_pcm16(audio)
    # First convert to WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_audio.tobytes())
    wav_buffer.seek(0)

    # Convert WAV to MP3 using pydub with highest quality
    # Use 320kbps CBR (constant bitrate) for maximum quality
    audio_segment = AudioSegment.from_wav(wav_buffer)
    mp3_buffer = io.BytesIO()
    audio_segment.export(
        mp3_buffer,
        format="mp3",
        bitrate="320k",
        parameters=["-q:a", "0"]  # Highest quality VBR setting
    )
    return mp3_buffer.getvalue()


def _audio_to_pcm16(audio: np.ndarray) -> np.ndarray:
    audio_array = np.asarray(audio).squeeze()
    if np.issubdtype(audio_array.dtype, np.floating):
        return (np.clip(audio_array, -1.0, 1.0).astype(np.float32) * 32767.0).astype(np.int16)
    return audio_array.astype(np.int16, copy=False)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    import math
    from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

    audio_array = np.asarray(audio).squeeze()
    gcd = math.gcd(source_rate, target_rate)
    resampled = resample_poly(audio_array.astype(np.float32), target_rate // gcd, source_rate // gcd)
    if np.issubdtype(audio_array.dtype, np.floating):
        return resampled.astype(audio_array.dtype, copy=False)
    return np.clip(resampled, np.iinfo(audio_array.dtype).min, np.iinfo(audio_array.dtype).max).astype(audio_array.dtype)


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return stem[:120] or "voice"


def _audio_file_to_mp3_bytes(audio_path: Path) -> bytes:
    audio_segment = AudioSegment.from_file(audio_path)
    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3", bitrate="320k", parameters=["-q:a", "0"])
    return mp3_buffer.getvalue()


class _SeedVCBackend:
    """Embedded Seed-VC inference backend compatible with the standalone /vc API."""

    model_presets = {
        "default": {"diffusion_steps": 30},
        "distilled": {"diffusion_steps": 30},
        "medium": {"diffusion_steps": 23},
        "fast": {"diffusion_steps": 15},
    }

    def __init__(self, settings: SeedVCConfig):
        self.settings = settings
        self.runtime_root = _resolve_project_path(settings.runtime_root).resolve()
        self.root = _resolve_project_path(settings.root).resolve()
        self.tmp_dir = _resolve_project_path(settings.tmp_dir).resolve()
        self.output_dir = _resolve_project_path(settings.output_dir).resolve()
        self.voice_samples_dir = _resolve_project_path(settings.voice_samples_dir).resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.runtime_root.exists():
            raise FileNotFoundError(f"Seed-VC runtime root not found: {self.runtime_root}")
        if not self.root.exists():
            raise FileNotFoundError(f"Seed-VC asset root not found: {self.root}")
        if str(self.runtime_root) not in sys.path:
            sys.path.insert(0, str(self.runtime_root))

        import torch  # pylint: disable=import-outside-toplevel
        import torchaudio  # pylint: disable=import-outside-toplevel
        import yaml  # pylint: disable=import-outside-toplevel
        from transformers import AutoFeatureExtractor, WhisperModel  # pylint: disable=import-outside-toplevel

        with _temporary_cwd(self.root):
            from hf_utils import load_custom_model_from_hf  # pylint: disable=import-outside-toplevel
            from inference import convert_voice, crossfade, find_silence_boundaries  # pylint: disable=import-outside-toplevel
            from modules.audio import mel_spectrogram  # pylint: disable=import-outside-toplevel
            from modules.bigvgan import bigvgan  # pylint: disable=import-outside-toplevel
            from modules.campplus.DTDNN import CAMPPlus  # pylint: disable=import-outside-toplevel
            from modules.commons import build_model, load_checkpoint, recursive_munch  # pylint: disable=import-outside-toplevel
            from modules.lazy_embedding_loader import HDF5EmbeddingLoader  # pylint: disable=import-outside-toplevel

            self.torch = torch
            self.torchaudio = torchaudio
            self.convert_voice = convert_voice
            self.seed_vc_crossfade = crossfade
            self.find_silence_boundaries = find_silence_boundaries

            if torch.cuda.is_available() and self.settings.device.startswith("cuda"):
                self.device = torch.device(self.settings.device)
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.dtype = torch.float16 if self.settings.fp16 else torch.float32

            _LOGGER.info("Loading Seed-VC models on %s from %s", self.device, self.root)
            medium_checkpoint_path, medium_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            )
            with open(medium_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            self.config = config
            self.model_params = recursive_munch(config["model_params"])
            self.model_params.dit_type = "DiT"
            self.sample_rate = int(config["preprocess_params"]["sr"])
            self.hop_length = int(config["preprocess_params"]["spect_params"]["hop_length"])

            self.default_model = build_model(self.model_params, stage="DiT")
            self.default_model, _, _, _ = load_checkpoint(
                self.default_model,
                None,
                medium_checkpoint_path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
            self._prepare_model_dict(self.default_model)
            self.model_cache = {"default": self.default_model}

            campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu", weights_only=False))
            self.campplus_model.eval().to(self.device)

            whisper_name = self.model_params.speech_tokenizer.name
            self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
            del self.whisper_model.decoder
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(self.model_params.vocoder.name, use_cuda_kernel=False)
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval().to(self.device)

            spect_params = config["preprocess_params"]["spect_params"]
            mel_fn_args = {
                "n_fft": spect_params["n_fft"],
                "win_size": spect_params["win_length"],
                "hop_size": spect_params["hop_length"],
                "num_mels": spect_params["n_mels"],
                "sampling_rate": self.sample_rate,
                "fmin": spect_params.get("fmin", 0),
                "fmax": None,
                "center": False,
            }
            self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

            embeddings_path = _resolve_project_path(self.settings.embeddings_hdf5_path)
            if embeddings_path.exists():
                self.cached_embeddings = HDF5EmbeddingLoader(embeddings_path, cache_size=self.settings.embedding_cache_size)
            else:
                raise RuntimeError(f"Seed-VC embeddings file not found: {embeddings_path}")

            try:
                from find_voice import find_base_voice  # pylint: disable=import-outside-toplevel
            except Exception as exc:  # pylint: disable=broad-exception-caught
                find_base_voice = None
                _LOGGER.warning("Seed-VC find_voice unavailable: %s", exc)
            try:
                from glitch_remover import process_file  # pylint: disable=import-outside-toplevel
            except Exception as exc:  # pylint: disable=broad-exception-caught
                process_file = None
                _LOGGER.warning("Seed-VC glitch remover unavailable: %s", exc)

            self.find_base_voice = find_base_voice
            self.process_file = process_file
            _LOGGER.info("Seed-VC backend ready: sr=%d cached_embeddings=%s", self.sample_rate, bool(self.cached_embeddings))

    def _prepare_model_dict(self, model_dict: dict[str, Any]) -> None:
        for key in model_dict:
            model_dict[key].eval()
            model_dict[key].to(self.device)
        estimator = getattr(model_dict.cfm, "estimator", None)
        if estimator is not None and hasattr(estimator, "setup_caches"):
            estimator.setup_caches(
                max_batch_size=self.settings.estimator_cache_batch_size,
                max_seq_length=self.settings.estimator_cache_seq_length,
            )

    def get_semantic_features(self, waves_16k):
        torch = self.torch
        ori_inputs = self.whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        ori_input_features = self.whisper_model._mask_input_features(
            ori_inputs.input_features,
            attention_mask=ori_inputs.attention_mask,
        ).to(self.device)
        with torch.no_grad():
            ori_outputs = self.whisper_model.encoder(ori_input_features.to(self.whisper_model.encoder.dtype))
        features = ori_outputs.last_hidden_state.to(torch.float32)
        return features[:, : waves_16k.size(-1) // 320 + 1]

    def get_semantic_features_batch(self, waves_16k: list[Any]):
        torch = self.torch
        lengths = [int(wave.size(-1)) for wave in waves_16k]
        ori_inputs = self.whisper_feature_extractor(
            [wave.squeeze(0).cpu().numpy() for wave in waves_16k],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        ori_input_features = self.whisper_model._mask_input_features(
            ori_inputs.input_features,
            attention_mask=ori_inputs.attention_mask,
        ).to(self.device)
        with torch.no_grad():
            ori_outputs = self.whisper_model.encoder(ori_input_features.to(self.whisper_model.encoder.dtype))
        features = ori_outputs.last_hidden_state.to(torch.float32)
        max_frames = max(length // 320 + 1 for length in lengths)
        return features[:, :max_frames], torch.LongTensor([length // 320 + 1 for length in lengths]).to(self.device)

    async def _fetch_sample(self, request: SeedVCRequest) -> Path:
        sample_path = self.voice_samples_dir / f"{_safe_file_stem(request.id)}.mp3"
        if sample_path.exists():
            return sample_path
        if not request.reference_url:
            raise ValueError(f"No cached sample for voice {request.id!r} and no reference_url provided")
        _LOGGER.info("Fetching Seed-VC reference sample: id=%s url=%s", request.id, request.reference_url)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(request.reference_url)
            resp.raise_for_status()
            sample_path.write_bytes(resp.content)
        return sample_path

    @staticmethod
    def _embedding_key(voice_id: str, style: str, intensity: float) -> str:
        if style == "general":
            return f"{voice_id}.general"
        if intensity == 1.0:
            return f"{voice_id}.{style}"
        if intensity == int(intensity):
            return f"{voice_id}.{style}.{int(intensity)}"
        return f"{voice_id}.{style}.{intensity}"

    def _available_styles_for_voice(self, voice_id: str) -> dict[str, list[float]]:
        styles: dict[str, set[float]] = {}
        if not self.cached_embeddings:
            return {}
        prefix = f"{voice_id}."
        for key in self.cached_embeddings.keys():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            style = suffix
            intensity = 1.0
            parts = suffix.split(".")
            for idx in range(1, len(parts)):
                try:
                    intensity = float(".".join(parts[idx:]))
                    style = ".".join(parts[:idx])
                    break
                except ValueError:
                    pass
            styles.setdefault(style, set()).add(intensity)
        return {style: sorted(intensities) for style, intensities in sorted(styles.items())}

    def _resolve_exact_cached_embeddings(self, voice_id: str, style: str, intensity: float) -> tuple[str, Any]:
        embedding_key = self._embedding_key(voice_id, style, intensity)
        if style == "general" and intensity != 1.0:
            embedding_key = f"{voice_id}.general.{intensity:g}"
        cached = self.cached_embeddings.get(embedding_key) if self.cached_embeddings else None
        if cached is not None:
            return embedding_key, cached

        available = self._available_styles_for_voice(voice_id)
        if not available:
            raise ValueError(f"No cached Seed-VC embeddings found for voice {voice_id!r}")
        if style not in available:
            raise ValueError(
                f"Unsupported style {style!r} for voice {voice_id!r}; supported styles: {sorted(available)}"
            )
        raise ValueError(
            f"Unsupported styleIntensity {intensity:g} for voice {voice_id!r} style {style!r}; "
            f"supported intensities: {available[style]}"
        )

    def _resolve_cached_embeddings(self, request: SeedVCRequest) -> tuple[str, Any | None]:
        style = request.style or "general"
        intensity = request.intensity or 1.0
        embedding_key = self._embedding_key(request.id, style, intensity)
        cached = self.cached_embeddings.get(embedding_key) if self.cached_embeddings else None
        if cached or style == "general" or not self.cached_embeddings:
            return embedding_key, cached

        available: list[tuple[float, str]] = []
        prefix = f"{request.id}.{style}"
        for key in self.cached_embeddings.keys():
            if key == prefix:
                available.append((1.0, key))
            elif key.startswith(prefix + "."):
                try:
                    available.append((float(key[len(prefix) + 1 :]), key))
                except ValueError:
                    pass
        if not available:
            return embedding_key, None
        available.sort(key=lambda item: abs(item[0] - intensity))
        closest_intensity, closest_key = available[0]
        _LOGGER.info("Seed-VC exact intensity %.3f not found for %s; using %.3f", intensity, prefix, closest_intensity)
        return closest_key, self.cached_embeddings.get(closest_key)

    def _convert_voice_v1(
        self,
        source_path: Path,
        target_path: Path | None,
        preset_config: dict[str, Any],
        cached_embeddings: Any | None,
        voice_id: str | None,
    ) -> Path:
        torch = self.torch
        import soundfile as sf  # pylint: disable=import-outside-toplevel

        diffusion_steps = int(preset_config.get("diffusion_steps", 25))
        length_adjust = float(preset_config.get("length_adjust", 1.0))
        cfg_rate = float(preset_config.get("cfg_rate", 0.7))
        model = self.model_cache["default"]

        with torch.no_grad(), _temporary_cwd(self.root):
            vc_wave = self.convert_voice(
                source_audio_path=str(source_path),
                ref_audio_path=str(target_path) if target_path is not None else None,
                model=model,
                semantic_fn=self.get_semantic_features,
                vocoder_fn=self.bigvgan_model,
                campplus_model=self.campplus_model,
                mel_fn=self.to_mel,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=cfg_rate,
                f0_condition=False,
                f0_fn=None,
                device=self.device,
                fp16=self.settings.fp16,
                cached_ref_embeddings=cached_embeddings,
            )

        source_name = source_path.stem
        target_name = _safe_file_stem(voice_id or (target_path.stem if target_path is not None else "cached"))
        output_path = self.output_dir / f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{cfg_rate}.wav"
        if torch.is_tensor(vc_wave):
            vc_wave = vc_wave.detach().float().cpu().numpy()
        vc_wave = np.asarray(vc_wave, dtype=np.float32).squeeze()
        sf.write(output_path, vc_wave, self.sample_rate)
        return output_path

    def _prepare_seed_vc_reference(self, target_path: Path | None, cached_ref_embeddings: Any | None, model: Any):
        torch = self.torch
        if cached_ref_embeddings is not None:
            style = cached_ref_embeddings["style"].to(self.device)
            mel_ref = cached_ref_embeddings["mel_ref"].to(self.device)
            prompt_condition = cached_ref_embeddings["prompt_condition"].to(self.device)
            if style.dim() == 1:
                style = style.unsqueeze(0)
            if mel_ref.dim() == 2:
                mel_ref = mel_ref.unsqueeze(0)
            if prompt_condition.dim() == 2:
                prompt_condition = prompt_condition.unsqueeze(0)
            return style, mel_ref, prompt_condition

        if target_path is None:
            raise ValueError("target_path is required when cached reference embeddings are unavailable")

        import librosa  # pylint: disable=import-outside-toplevel

        ref_audio_np = librosa.load(target_path, sr=self.sample_rate)[0]
        ref_audio = torch.tensor(ref_audio_np[: self.sample_rate * 25]).unsqueeze(0).float().to(self.device)
        ref_16k = self.torchaudio.functional.resample(ref_audio, self.sample_rate, 16000)
        semantic_ref = self.get_semantic_features(ref_16k)
        mel_ref = self.to_mel(ref_audio.float())
        ref_lengths = torch.LongTensor([mel_ref.size(2)]).to(self.device)
        feat_ref = self.torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat_ref = feat_ref - feat_ref.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat_ref.unsqueeze(0))
        prompt_condition, _, _, _, _ = model.length_regulator(
            semantic_ref, ylens=ref_lengths, n_quantizers=3, f0=None
        )
        return style, mel_ref, prompt_condition

    @staticmethod
    def _fade_seed_vc_start(result: np.ndarray, sample_rate: int) -> np.ndarray:
        check_window = int(sample_rate * 0.03)
        if len(result) <= check_window:
            return result
        first_10ms = int(sample_rate * 0.01)
        window_20_30ms = result[int(sample_rate * 0.02) : check_window]
        energy_first = np.sqrt(np.mean(result[:first_10ms] ** 2))
        energy_later = np.sqrt(np.mean(window_20_30ms ** 2))
        if energy_first > energy_later * 1.5:
            fade_samples = int(sample_rate * 0.025)
            fade_in = np.linspace(0, 1, fade_samples) ** 4
        else:
            fade_samples = int(sample_rate * 0.005)
            fade_in = np.linspace(0, 1, fade_samples)
        result[:fade_samples] *= fade_in
        return result

    def _convert_voice_v1_batch(
        self,
        source_paths: list[Path],
        target_path: Path | None,
        preset_config: dict[str, Any],
        cached_embeddings: Any | None,
        max_chunk_batch_size: int,
    ) -> list[np.ndarray]:
        torch = self.torch
        import librosa  # pylint: disable=import-outside-toplevel

        diffusion_steps = int(preset_config.get("diffusion_steps", 25))
        length_adjust = float(preset_config.get("length_adjust", 1.0))
        cfg_rate = float(preset_config.get("cfg_rate", 0.7))
        model = self.model_cache["default"]

        style, mel_ref, prompt_condition = self._prepare_seed_vc_reference(target_path, cached_embeddings, model)
        source_audios = [
            np.asarray(librosa.load(path, sr=self.sample_rate)[0], dtype=np.float32)
            for path in source_paths
        ]
        chunk_records: list[tuple[int, int, int, np.ndarray]] = []
        generated_wave_chunks: list[list[np.ndarray]] = [[] for _ in source_audios]
        crossfade_samples = int(self.sample_rate * 0.05)

        for item_idx, source_audio in enumerate(source_audios):
            chunks = self.find_silence_boundaries(
                source_audio,
                self.sample_rate,
                min_silence_duration=0.15,
                silence_threshold=0.02,
                max_chunk_duration=25.0,
                min_chunk_duration=3.0,
            )
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                chunk_records.append(
                    (
                        item_idx,
                        chunk_idx,
                        chunk_start,
                        source_audio[chunk_start:chunk_end],
                    )
                )

        for batch_start in range(0, len(chunk_records), max_chunk_batch_size):
            records = chunk_records[batch_start : batch_start + max_chunk_batch_size]
            chunk_audios = [
                torch.from_numpy(record[3]).unsqueeze(0).to(self.device, dtype=torch.float32)
                for record in records
            ]
            chunk_16k = [
                self.torchaudio.functional.resample(chunk_audio, self.sample_rate, 16000)
                for chunk_audio in chunk_audios
            ]
            semantic_batch, _ = self.get_semantic_features_batch(chunk_16k)

            target_lengths = []
            for chunk_audio in chunk_audios:
                mel_chunk = self.to_mel(chunk_audio.float())
                target_lengths.append(int(mel_chunk.size(2) * length_adjust))
            chunk_target_lengths = torch.LongTensor(target_lengths).to(self.device)

            cond_chunk, _, _, _, _ = model.length_regulator(
                semantic_batch,
                ylens=chunk_target_lengths,
                n_quantizers=3,
                f0=None,
            )

            batch_size = len(records)
            prompt_batch = prompt_condition.expand(batch_size, -1, -1)
            mel_ref_batch = mel_ref.expand(batch_size, -1, -1)
            style_batch = style.expand(batch_size, -1)
            cat_condition = torch.cat([prompt_batch, cond_chunk], dim=1)
            x_lens = torch.LongTensor([prompt_condition.size(1) + length for length in target_lengths]).to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.settings.fp16 else torch.float32):
                vc_target = model.cfm.inference(
                    cat_condition,
                    x_lens,
                    mel_ref_batch,
                    style_batch,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=cfg_rate,
                )
                vc_target = vc_target[:, :, mel_ref.size(-1) :]
                vc_wave_batch = self.bigvgan_model(vc_target.float())

            if vc_wave_batch.dim() == 1:
                vc_wave_batch = vc_wave_batch.unsqueeze(0)
            elif vc_wave_batch.dim() == 3:
                vc_wave_batch = vc_wave_batch.squeeze(1)

            for batch_idx, (item_idx, chunk_idx, _, _) in enumerate(records):
                target_samples = max(1, int(target_lengths[batch_idx] * self.hop_length))
                chunk_output = vc_wave_batch[batch_idx, :target_samples].detach().float().cpu().numpy()
                if chunk_idx > 0 and generated_wave_chunks[item_idx] and crossfade_samples > 0:
                    prev_chunk = generated_wave_chunks[item_idx][-1]
                    if len(prev_chunk) >= crossfade_samples and len(chunk_output) >= crossfade_samples:
                        chunk_output = self.seed_vc_crossfade(prev_chunk, chunk_output, crossfade_samples)
                        generated_wave_chunks[item_idx][-1] = prev_chunk[:-crossfade_samples]
                generated_wave_chunks[item_idx].append(chunk_output)

            del chunk_audios, chunk_16k, semantic_batch, chunk_target_lengths, cond_chunk
            del prompt_batch, mel_ref_batch, style_batch, cat_condition, x_lens, vc_target, vc_wave_batch

        results = []
        for chunks in generated_wave_chunks:
            result = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            results.append(self._fade_seed_vc_start(result, self.sample_rate).astype(np.float32))
        return results

    def convert_batch_request(
        self, request: SeedVCBatchRequest, reference_path: Path | None = None,
        embedding_key: str | None = None, cached_embeddings=None,
    ) -> dict[str, Any]:
        items = request.items
        first = items[0]
        for item in items[1:]:
            if (
                item.id != first.id
                or item.style != first.style
                or item.intensity != first.intensity
                or item.preset != first.preset
                or item.reference_url != first.reference_url
            ):
                raise ValueError("Seed-VC batch currently requires a shared id/style/intensity/preset/reference_url")

        preset = first.preset or "default"
        preset_config = self.model_presets.get(preset)
        if preset_config is None:
            raise ValueError(f"Unknown Seed-VC preset {preset!r}; expected one of {sorted(self.model_presets)}")
        chunk_batch_size = min(
            max(1, int(request.max_chunk_batch_size)),
            max(1, int(self.settings.max_chunk_batch_size)),
        )

        import soundfile as sf  # pylint: disable=import-outside-toplevel

        source_paths = [self.tmp_dir / f"{uuid.uuid4().hex}.input" for _ in items]
        wav_output_paths: list[Path] = []
        try:
            for item, source_path in zip(items, source_paths):
                source_path.write_bytes(base64.b64decode(item.audio))

            with self.lock, self.torch.no_grad(), _temporary_cwd(self.root):
                waves = self._convert_voice_v1_batch(
                    source_paths,
                    reference_path,
                    preset_config,
                    cached_embeddings=cached_embeddings,
                    max_chunk_batch_size=chunk_batch_size,
                )
                encoded = []
                for idx, (item, wave_data) in enumerate(zip(items, waves)):
                    wav_output_path = self.output_dir / f"vc_batch_{uuid.uuid4().hex}_{idx}.wav"
                    wav_output_paths.append(wav_output_path)
                    sf.write(wav_output_path, wave_data, self.sample_rate)
                    if item.remove_glitches:
                        self._remove_glitches(wav_output_path)
                    mp3_bytes = _audio_file_to_mp3_bytes(wav_output_path)
                    encoded.append(base64.b64encode(mp3_bytes).decode("ascii"))

            return {
                "sample_rate": self.sample_rate,
                "format": "mp3",
                "preset": preset,
                "count": len(encoded),
                "audios": encoded,
            }
        finally:
            for path in source_paths:
                path.unlink(missing_ok=True)
            for path in wav_output_paths:
                path.unlink(missing_ok=True)

    def _remove_glitches(self, wav_path: Path) -> None:
        if self.process_file is None:
            _LOGGER.warning("Seed-VC glitch removal requested but glitch_remover is unavailable")
            return
        glitch_temp_dir = self.tmp_dir / f"glitch_temp_{uuid.uuid4().hex}"
        glitch_temp_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "rms_win_ms": 5.0,
            "rms_hop_ms": 1.0,
            "rms_thr": 0.002,
            "hold_ms": 15.0,
            "safety_ms": 2.0,
            "max_cut_ms": 200.0,
            "veto_cut_ms": 50.0,
            "fallback_trim_ms": 0.0,
        }
        try:
            result = self.process_file(str(wav_path), str(glitch_temp_dir), params, do_write=True)
            _LOGGER.info("Seed-VC glitch removal: %s cut_ms=%s", result.get("decision"), result.get("cut_ms"))
            shutil.copyfile(result["out"], wav_path)
        finally:
            shutil.rmtree(glitch_temp_dir, ignore_errors=True)

    def convert_request(self, request: SeedVCRequest) -> bytes:
        emb_key, emb = self._resolve_cached_embeddings(request)
        reference_path = None if emb is not None else self._fetch_sample(request)
        return self._convert_with_reference(
            request, reference_path,
            embedding_key=emb_key if emb is not None else None,
            cached_embeddings=emb,
        )

    def convert_generated_audio_batch(
        self,
        source_audios: list[np.ndarray],
        source_sample_rate: int,
        voice_id: str,
        style: str,
        intensity: float,
        preset: str | None,
        output_format: Literal["wav", "mp3"],
        max_chunk_batch_size: int | None = None,
        strict_embedding: bool = False,
    ) -> list[tuple[bytes, float]]:
        if not source_audios:
            return []

        if strict_embedding:
            emb_key, emb = self._resolve_exact_cached_embeddings(voice_id, style, intensity)
        else:
            request = SeedVCRequest(
                audio="",
                id=voice_id,
                style=style,
                intensity=intensity,
                preset=preset,
            )
            emb_key, emb = self._resolve_cached_embeddings(request)
        if emb is None:
            raise ValueError(f"No cached Seed-VC embedding for voice {emb_key!r}")

        preset_name = preset or "default"
        preset_config = self.model_presets.get(preset_name)
        if preset_config is None:
            raise ValueError(f"Unknown Seed-VC preset {preset_name!r}; expected one of {sorted(self.model_presets)}")
        chunk_batch_size = max_chunk_batch_size or self.settings.max_chunk_batch_size

        import soundfile as sf  # pylint: disable=import-outside-toplevel

        source_paths = [self.tmp_dir / f"{uuid.uuid4().hex}.input.wav" for _ in source_audios]
        wav_output_paths: list[Path] = []
        try:
            for source_audio, source_path in zip(source_audios, source_paths):
                sf.write(source_path, _audio_to_pcm16(source_audio), source_sample_rate)

            _LOGGER.info(
                "Seed-VC convert generated batch: voice=%s preset=%s count=%d",
                voice_id,
                preset_name,
                len(source_paths),
            )
            with self.lock, self.torch.no_grad(), _temporary_cwd(self.root):
                waves = self._convert_voice_v1_batch(
                    source_paths,
                    None,
                    preset_config,
                    cached_embeddings=emb,
                    max_chunk_batch_size=chunk_batch_size,
                )

            encoded: list[tuple[bytes, float]] = []
            for idx, wave_data in enumerate(waves):
                audio_seconds = float(len(wave_data)) / self.sample_rate if self.sample_rate else 0.0
                if output_format == "mp3":
                    wav_output_path = self.output_dir / f"vc_synth_batch_{uuid.uuid4().hex}_{idx}.wav"
                    wav_output_paths.append(wav_output_path)
                    sf.write(wav_output_path, wave_data, self.sample_rate)
                    encoded.append((_audio_file_to_mp3_bytes(wav_output_path), audio_seconds))
                else:
                    encoded.append((_audio_to_wav_bytes(wave_data, self.sample_rate), audio_seconds))
            return encoded
        finally:
            for path in source_paths:
                path.unlink(missing_ok=True)
            for path in wav_output_paths:
                path.unlink(missing_ok=True)

    def convert_generated_audio_reference_batch(
        self,
        source_audios: list[np.ndarray],
        source_sample_rate: int,
        reference_path: Path,
        preset: str | None,
        output_format: Literal["wav", "mp3"],
        max_chunk_batch_size: int | None = None,
    ) -> list[tuple[bytes, float]]:
        if not source_audios:
            return []

        preset_name = preset or "default"
        preset_config = self.model_presets.get(preset_name)
        if preset_config is None:
            raise ValueError(f"Unknown Seed-VC preset {preset_name!r}; expected one of {sorted(self.model_presets)}")
        chunk_batch_size = max_chunk_batch_size or self.settings.max_chunk_batch_size

        import soundfile as sf  # pylint: disable=import-outside-toplevel

        source_paths = [self.tmp_dir / f"{uuid.uuid4().hex}.input.wav" for _ in source_audios]
        wav_output_paths: list[Path] = []
        try:
            for source_audio, source_path in zip(source_audios, source_paths):
                sf.write(source_path, _audio_to_pcm16(source_audio), source_sample_rate)

            _LOGGER.info(
                "Seed-VC convert generated reference batch: reference=%s preset=%s count=%d",
                reference_path,
                preset_name,
                len(source_paths),
            )
            with self.lock, self.torch.no_grad(), _temporary_cwd(self.root):
                waves = self._convert_voice_v1_batch(
                    source_paths,
                    reference_path,
                    preset_config,
                    cached_embeddings=None,
                    max_chunk_batch_size=chunk_batch_size,
                )

            encoded: list[tuple[bytes, float]] = []
            for idx, wave_data in enumerate(waves):
                audio_seconds = float(len(wave_data)) / self.sample_rate if self.sample_rate else 0.0
                if output_format == "mp3":
                    wav_output_path = self.output_dir / f"vc_synth_sample_batch_{uuid.uuid4().hex}_{idx}.wav"
                    wav_output_paths.append(wav_output_path)
                    sf.write(wav_output_path, wave_data, self.sample_rate)
                    encoded.append((_audio_file_to_mp3_bytes(wav_output_path), audio_seconds))
                else:
                    encoded.append((_audio_to_wav_bytes(wave_data, self.sample_rate), audio_seconds))
            return encoded
        finally:
            for path in source_paths:
                path.unlink(missing_ok=True)
            for path in wav_output_paths:
                path.unlink(missing_ok=True)

    def _convert_with_reference(
        self,
        request: SeedVCRequest,
        reference_path: Path | None,
        embedding_key: str | None = None,
        cached_embeddings: torch.Tensor | None = None,
    ) -> bytes:
        preset = request.preset or "default"
        preset_config = self.model_presets.get(preset)
        if preset_config is None:
            raise ValueError(f"Unknown Seed-VC preset {preset!r}; expected one of {sorted(self.model_presets)}")

        source_path = self.tmp_dir / f"{uuid.uuid4().hex}.input"
        wav_output_path: Path | None = None
        try:
            source_path.write_bytes(base64.b64decode(request.audio))
            _LOGGER.info(
                "Seed-VC convert: voice=%s preset=%s cached_embedding=%s reference=%s",
                request.id, preset, bool(cached_embeddings), reference_path,
            )
            with self.lock:
                wav_output_path = self._convert_voice_v1(
                    source_path,
                    reference_path,
                    preset_config,
                    cached_embeddings=cached_embeddings,
                    voice_id=embedding_key if cached_embeddings is not None else None,
                )
                if request.remove_glitches:
                    self._remove_glitches(wav_output_path)
                return _audio_file_to_mp3_bytes(wav_output_path)
        finally:
            source_path.unlink(missing_ok=True)
            if wav_output_path is not None:
                wav_output_path.unlink(missing_ok=True)

    def find_voice(self, request: SeedVCFindVoiceRequest, reference_path: Path) -> str:
        if self.find_base_voice is None:
            raise RuntimeError("Seed-VC find_voice support is unavailable")
        with self.lock, _temporary_cwd(self.root):
            return str(self.find_base_voice(str(reference_path)))

    def enhance(self, request: SeedVCEnhanceRequest, raw_path: Path) -> bytes:
        import subprocess
        enhance_dir = raw_path.parent
        sample_dir = enhance_dir / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        wav_path = enhance_dir / "sample_raw.wav"

        subprocess.run(["ffmpeg", "-i", str(raw_path), "-t", "120", str(wav_path), "-y"],
                       capture_output=True, check=True)
        subprocess.run(["uv", "tool", "run", "ffmpeg-normalize", str(wav_path), "-o", str(sample_dir / "sample.wav"), "-f"],
                       capture_output=True, check=True)
        sample_wav = sample_dir / "sample.wav"
        enhanced_dir = enhance_dir / "enhanced"
        enhanced_dir.mkdir(exist_ok=True)
        enhanced_wav = enhanced_dir / "sample.wav"
        enhanced_wav.write_bytes(sample_wav.read_bytes())
        mp3_path = enhance_dir / "final_sample.mp3"
        subprocess.run(["ffmpeg", "-i", str(enhanced_wav), "-f", "mp3", "-aq", "2", "-b:a", "320k", str(mp3_path), "-y"],
                       capture_output=True, check=True)
        return mp3_path.read_bytes()


def _get_matcha_batcher() -> ProductionMatchaBatcher:
    if not _engine_enabled("matcha"):
        raise HTTPException(status_code=503, detail="Matcha backend is disabled")
    if _matcha_batcher is None:
        raise HTTPException(status_code=503, detail="Matcha backend is not enabled or not loaded")
    return _matcha_batcher


def _get_seed_vc_backend() -> _SeedVCBackend:
    if not _engine_enabled("seed_vc"):
        raise HTTPException(status_code=503, detail="Seed-VC backend is disabled")
    if _seed_vc_backend is None:
        raise HTTPException(status_code=503, detail="Seed-VC backend was not loaded at startup")
    return _seed_vc_backend


def _build_rvc_backend(settings: RVCConfig) -> RVCBackend:
    backend = RVCBackend(RVCSettings(
        enabled=True,
        preload=settings.preload,
        cache_size=settings.cache_size,
        preload_models=settings.preload_models,
        default_f0_method=settings.default_f0_method,
        default_pitch=settings.default_pitch,
        default_index_rate=settings.default_index_rate,
        default_rms_mix_rate=settings.default_rms_mix_rate,
        default_protect=settings.default_protect,
    ))
    backend.preload_models(settings.preload_models)
    return backend


def _get_rvc_backend() -> RVCBackend:
    if not _engine_enabled("rvc"):
        raise HTTPException(status_code=503, detail="RVC backend is disabled")
    if _rvc_backend is None:
        raise HTTPException(status_code=503, detail="RVC backend was not loaded at startup")
    return _rvc_backend


async def synthesize_sparrow_batch(
    request: _SharedBatchSynthesizeRequest,
    *,
    speaker: str | None = None,
    model: str | None = None,
) -> BatchSynthesizeResponse:
    """Run real batched Sparrow/VITS synthesis for one shared model."""
    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    if request.voice_id is not None:
        raise HTTPException(status_code=400, detail="voice_id requests must use configured voice synthesis")
    if request.sample_url is not None and _seed_vc_style_requested(request):
        raise HTTPException(status_code=400, detail="'style' and 'styleIntensity' require 'voice_id'")

    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")

    model_name = model or request.model
    resolved_speaker = speaker
    if model_name is None and resolved_speaker is None:
        raise HTTPException(
            status_code=400,
            detail="real Sparrow batching requires 'model' when voice_id/language routing is not used",
        )

    if model_name is None:
        resolved_speaker, model_name = _resolve_speaker_and_model(resolved_speaker, explicit=True)

    inference = _get_inference(model_name)

    model_cfg = _server_config.pipertts.model_config_overrides.get(model_name)
    if model_cfg and resolved_speaker in model_cfg.speakers:
        internal_speaker = None if model_cfg.speakers[resolved_speaker] is None else resolved_speaker
    elif resolved_speaker is None:
        internal_speaker = None
    elif resolved_speaker in inference.speakers:
        internal_speaker = resolved_speaker
    else:
        raise HTTPException(
            status_code=400,
            detail=f"speaker {resolved_speaker!r} is not available for model {model_name!r}",
        )

    synth_kwargs = _synth_kwargs_from_request(request)
    started = time.perf_counter()
    _log_synthesize_batch_stage(
        "sparrow_batch_start",
        pipeline="direct_sparrow",
        model=model_name,
        speaker=resolved_speaker,
        internal_speaker=internal_speaker,
        item_count=len(texts),
        segment_count=len(texts),
        batch_size=len(texts),
        neural=request.neural,
        synth_kwargs=synth_kwargs,
        sample_url=bool(request.sample_url),
    )
    audios = await asyncio.to_thread(
        inference.synthesize_batch,
        texts,
        speaker=internal_speaker,
        batch_size=len(texts),
        neural=request.neural,
        **synth_kwargs,
    )
    wall_seconds = time.perf_counter() - started

    sample_rate = inference.sample_rate
    audio_seconds = sum(float(len(audio)) / sample_rate for audio in audios) if sample_rate else 0.0
    _log_synthesize_batch_stage(
        "sparrow_batch_done",
        pipeline="direct_sparrow",
        model=model_name,
        output_count=len(audios),
        audio_seconds=round(audio_seconds, 6),
        wall_seconds=round(wall_seconds, 6),
        rtf=round(wall_seconds / audio_seconds, 6) if audio_seconds else 0.0,
        sample_rate=sample_rate,
    )
    if request.sample_url is not None:
        converted, converted_sample_rate = await _convert_generated_audio_to_sample_batch(
            source_audios=audios,
            source_sample_rates=[sample_rate for _ in audios],
            sample_url=request.sample_url,
            output_format=request.format,
        )
        items = []
        audio_seconds_total = 0.0
        for text, (encoded, audio_seconds) in zip(texts, converted):
            audio_seconds_total += audio_seconds
            items.append(
                BatchSynthesizeItem(
                    text=text,
                    audio_base64=base64.b64encode(encoded).decode("ascii"),
                    sample_rate=converted_sample_rate,
                    audio_seconds=audio_seconds,
                )
            )
        total_wall_seconds = time.perf_counter() - started
        return BatchSynthesizeResponse(
            items=items,
            count=len(items),
            model=model_name,
            speaker=resolved_speaker,
            wall_seconds=total_wall_seconds,
            audio_seconds_total=audio_seconds_total,
            rtf=(total_wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
        )

    items: list[BatchSynthesizeItem] = []
    audio_seconds_total = 0.0
    for text, audio in zip(texts, audios):
        audio_seconds = float(len(audio)) / sample_rate if sample_rate else 0.0
        audio_seconds_total += audio_seconds
        if request.format == "mp3":
            encoded = _audio_to_mp3_bytes(audio, sample_rate)
        else:
            encoded = _audio_to_wav_bytes(audio, sample_rate)
        items.append(
            BatchSynthesizeItem(
                text=text,
                audio_base64=base64.b64encode(encoded).decode("ascii"),
                sample_rate=sample_rate,
                audio_seconds=audio_seconds,
            )
        )

    return BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=model_name,
        speaker=resolved_speaker,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )


async def synthesize_multilingual_sparrow_batch(request: _SharedBatchSynthesizeRequest) -> BatchSynthesizeResponse:
    """Run real batched Sparrow synthesis for auto-routed multilingual text items."""
    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    if request.voice_id is not None:
        raise HTTPException(status_code=400, detail="voice_id requests must use configured voice synthesis")
    if request.model is not None:
        raise HTTPException(status_code=400, detail="model-specific requests must use direct Sparrow batching")
    if request.sample_url is not None and _seed_vc_style_requested(request):
        raise HTTPException(status_code=400, detail="'style' and 'styleIntensity' require 'voice_id'")

    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")

    started = time.perf_counter()
    synth_kwargs = _synth_kwargs_from_request(request)
    item_segments: list[list[dict[str, Any]]] = []
    segment_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    for item_idx, text in enumerate(texts):
        segments, _ = _plan_text_segments(text, primary_speaker=None, forced_language=None)
        item_segments.append(segments)
        for segment_idx, segment in enumerate(segments):
            record = {**segment, "item_idx": item_idx, "segment_idx": segment_idx}
            segment_groups.setdefault(segment["model"], []).append(record)

    generated_segments: list[list[tuple[np.ndarray, int] | None]] = [
        [None for _ in segments]
        for segments in item_segments
    ]

    for model_name, records in segment_groups.items():
        inference = _get_inference(model_name)
        model_sample_rate = inference.sample_rate
        batch_texts = [record["text"] for record in records]
        batch_speakers = [
            _resolve_internal_speaker(model_name, record["speaker"], inference)
            for record in records
        ]
        batch_started = time.perf_counter()
        _log_synthesize_batch_stage(
            "sparrow_batch_start",
            pipeline="auto_multilingual",
            model=model_name,
            item_count=len(texts),
            segment_count=len(batch_texts),
            batch_size=len(batch_texts),
            speakers=sorted({str(speaker) for speaker in batch_speakers}),
            neural=request.neural,
            synth_kwargs=synth_kwargs,
            sample_url=bool(request.sample_url),
        )
        batch_audios = await asyncio.to_thread(
            inference.synthesize_batch,
            batch_texts,
            speaker=batch_speakers,
            batch_size=len(batch_texts),
            neural=request.neural,
            **synth_kwargs,
        )
        audio_seconds = sum(float(len(audio)) / model_sample_rate for audio in batch_audios) if model_sample_rate else 0.0
        elapsed = time.perf_counter() - batch_started
        _log_synthesize_batch_stage(
            "sparrow_batch_done",
            pipeline="auto_multilingual",
            model=model_name,
            output_count=len(batch_audios),
            audio_seconds=round(audio_seconds, 6),
            wall_seconds=round(elapsed, 6),
            rtf=round(elapsed / audio_seconds, 6) if audio_seconds else 0.0,
            sample_rate=model_sample_rate,
        )
        for record, audio in zip(records, batch_audios):
            generated_segments[record["item_idx"]][record["segment_idx"]] = (audio, model_sample_rate)

    item_audios: list[np.ndarray] = []
    item_sample_rates: list[int] = []
    for segments in generated_segments:
        parts = [segment for segment in segments if segment is not None]
        if not parts:
            item_audios.append(np.zeros(0, dtype=np.int16))
            item_sample_rates.append(22050)
        elif len(parts) == 1:
            audio, source_rate = parts[0]
            item_audios.append(audio)
            item_sample_rates.append(source_rate)
        else:
            target_rate = parts[0][1]
            item_audios.append(np.concatenate([
                _resample_audio(audio, source_rate, target_rate)
                for audio, source_rate in parts
            ], axis=0))
            item_sample_rates.append(target_rate)

    if request.sample_url is not None:
        converted, converted_sample_rate = await _convert_generated_audio_to_sample_batch(
            source_audios=item_audios,
            source_sample_rates=item_sample_rates,
            sample_url=request.sample_url,
            output_format=request.format,
        )
        items = []
        audio_seconds_total = 0.0
        for text, (encoded, audio_seconds) in zip(texts, converted):
            audio_seconds_total += audio_seconds
            items.append(
                BatchSynthesizeItem(
                    text=text,
                    audio_base64=base64.b64encode(encoded).decode("ascii"),
                    sample_rate=converted_sample_rate,
                    audio_seconds=audio_seconds,
                )
            )
        wall_seconds = time.perf_counter() - started
        return BatchSynthesizeResponse(
            items=items,
            count=len(items),
            model="auto",
            speaker=None,
            wall_seconds=wall_seconds,
            audio_seconds_total=audio_seconds_total,
            rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
        )

    items: list[BatchSynthesizeItem] = []
    audio_seconds_total = 0.0
    for text, audio, sample_rate in zip(texts, item_audios, item_sample_rates):
        audio_seconds = float(len(audio)) / sample_rate if sample_rate else 0.0
        audio_seconds_total += audio_seconds
        encoded = (
            _audio_to_mp3_bytes(audio, sample_rate)
            if request.format == "mp3"
            else _audio_to_wav_bytes(audio, sample_rate)
        )
        items.append(
            BatchSynthesizeItem(
                text=text,
                audio_base64=base64.b64encode(encoded).decode("ascii"),
                sample_rate=sample_rate,
                audio_seconds=audio_seconds,
            )
        )

    wall_seconds = time.perf_counter() - started
    return BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model="auto",
        speaker=None,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )


def _batch_item_group_key(item: BatchSynthesizeInputItem) -> tuple[Any, ...]:
    options_key = None
    if item.options is not None:
        options_key = json.dumps(item.options.model_dump(mode="json"), sort_keys=True)
    kind = "voice" if item.voice_id is not None else "sparrow"
    return (
        kind,
        item.voice_id,
        item.sample_url,
        item.language,
        item.model,
        item.style,
        item.style_intensity,
        options_key,
        item.format,
        item.neural,
    )


def _validate_batch_item(item: BatchSynthesizeInputItem, item_idx: int) -> str:
    if item.text and item.ssml:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: Provide either 'text' or 'ssml', not both")
    if item.ssml is not None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: SSML is not supported in /synthesize/batch")
    if item.text is None or not item.text.strip():
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: text is required")
    if item.sample_url is not None and item.voice_id is not None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: Use either 'voice_id' or 'sample_url', not both")
    if item.voice_id is not None and item.model is not None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: Use either 'voice_id' or 'model', not both")
    if item.language is not None and item.model is not None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: Use either 'language' or 'model', not both")
    if item.sample_url is not None and _seed_vc_style_requested(item):
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'style' and 'styleIntensity' require 'voice_id'")
    if item.voice_id is None and _seed_vc_style_requested(item):
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'style' and 'styleIntensity' require 'voice_id'")
    return item.text.strip()


def _shared_batch_from_items(records: list[tuple[int, BatchSynthesizeInputItem, str]]) -> _SharedBatchSynthesizeRequest:
    first = records[0][1]
    return _SharedBatchSynthesizeRequest(
        texts=[text for _, _, text in records],
        voice_id=first.voice_id,
        sample_url=first.sample_url,
        language=first.language,
        model=first.model,
        style=first.style,
        style_intensity=first.style_intensity,
        options=first.options,
        format=first.format,
        neural=first.neural,
    )


async def synthesize_mixed_batch(request: BatchSynthesizeRequest) -> BatchSynthesizeResponse:
    """Group independent /synthesize-shaped inputs and run compatible real batches."""
    started = time.perf_counter()
    groups: OrderedDict[tuple[Any, ...], list[tuple[int, BatchSynthesizeInputItem, str]]] = OrderedDict()
    for item_idx, item in enumerate(request.items):
        text = _validate_batch_item(item, item_idx)
        groups.setdefault(_batch_item_group_key(item), []).append((item_idx, item, text))

    _log_synthesize_batch_stage(
        "request_grouping",
        item_count=len(request.items),
        group_count=len(groups),
        groups=[
            {
                "group_index": group_idx,
                "item_indices": [item_idx for item_idx, _, _ in records],
                "count": len(records),
                "voice_id": records[0][1].voice_id,
                "sample_url": bool(records[0][1].sample_url),
                "language": records[0][1].language,
                "model": records[0][1].model,
                "style": records[0][1].style,
                "styleIntensity": records[0][1].style_intensity,
                "format": records[0][1].format,
                "neural": records[0][1].neural,
            }
            for group_idx, records in enumerate(groups.values())
        ],
    )

    output_items: list[BatchSynthesizeItem | None] = [None for _ in request.items]
    group_results: list[BatchSynthesizeResponse] = []

    for group_idx, records in enumerate(groups.values()):
        shared_request = _shared_batch_from_items(records)
        group_started = time.perf_counter()
        _log_synthesize_batch_stage(
            "group_start",
            group_index=group_idx,
            item_indices=[item_idx for item_idx, _, _ in records],
            item_count=len(records),
            voice_id=shared_request.voice_id,
            sample_url=bool(shared_request.sample_url),
            language=shared_request.language,
            model=shared_request.model,
            format=shared_request.format,
        )
        if shared_request.voice_id is not None:
            group_result = await synthesize_configured_voice_batch(shared_request)
        elif shared_request.language is not None:
            _, forced_speaker, forced_model = _resolve_forced_language(shared_request.language)
            group_result = await synthesize_sparrow_batch(
                _SharedBatchSynthesizeRequest(
                    texts=shared_request.texts,
                    sample_url=shared_request.sample_url,
                    model=forced_model,
                    options=shared_request.options,
                    format=shared_request.format,
                    neural=shared_request.neural,
                ),
                speaker=forced_speaker,
            )
        elif shared_request.model is not None:
            group_result = await synthesize_sparrow_batch(shared_request)
        else:
            group_result = await synthesize_multilingual_sparrow_batch(shared_request)

        _log_synthesize_batch_stage(
            "group_done",
            group_index=group_idx,
            item_count=len(records),
            output_count=len(group_result.items),
            model=group_result.model,
            speaker=group_result.speaker,
            audio_seconds=round(group_result.audio_seconds_total, 6),
            wall_seconds=round(time.perf_counter() - group_started, 6),
            group_rtf=round(group_result.rtf, 6),
        )
        _log_synthesize_batch_summary(
            group_index=group_idx,
            item_indices=[item_idx for item_idx, _, _ in records],
            item_count=len(records),
            output_count=len(group_result.items),
            voice_id=shared_request.voice_id,
            sample_url=bool(shared_request.sample_url),
            language=shared_request.language,
            model=group_result.model,
            speaker=group_result.speaker,
            format=shared_request.format,
            audio_seconds=round(group_result.audio_seconds_total, 6),
            wall_seconds=round(time.perf_counter() - group_started, 6),
            rtf=round(group_result.rtf, 6),
        )
        group_results.append(group_result)
        for (item_idx, _, _), result_item in zip(records, group_result.items):
            output_items[item_idx] = result_item

    items = [item for item in output_items if item is not None]
    if len(items) != len(request.items):
        raise RuntimeError("internal batch response ordering error")

    audio_seconds_total = sum(item.audio_seconds for item in items)
    wall_seconds = time.perf_counter() - started
    models = {result.model for result in group_results}
    speakers = {result.speaker for result in group_results}
    return BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=next(iter(models)) if len(models) == 1 else "mixed",
        speaker=next(iter(speakers)) if len(speakers) == 1 else None,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create the FastAPI application."""
    global _server_config, _speaker_routes

    if config is None:
        config = _load_config()
    _server_config = config
    if _engine_enabled("qwen3", config):
        qwen3.configure(config.qwen)

    app = FastAPI(
        title="LZ-TTS API",
        description="Piper TTS inference API",
        version="0.1.0",
    )
    app.add_middleware(
        CompressionMiddleware,
        algorithms=[
            ZstdAlgorithm(level=3),
            BrotliAlgorithm(quality=4),
            GzipAlgorithm(compresslevel=5),
        ],
        minimum_size=1024,
    )
    app.add_middleware(RequestDecompressionMiddleware)

    @app.middleware("http")
    async def api_key_auth_middleware(request: Request, call_next):
        provided_api_key = _request_api_key(request)
        _scrub_api_key_query_param(request)

        if _is_browser_demo_path(request.url.path):
            return await call_next(request)

        expected_api_key = _configured_api_key()
        if not expected_api_key:
            return JSONResponse(
                status_code=503,
                content={"detail": "API_KEY is not configured"},
            )
        if not provided_api_key or not secrets.compare_digest(provided_api_key, expected_api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)

    if _engine_enabled("qwen3", config):
        app.include_router(qwen3_router)
        _mount_qwen_demo(app)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        if request.url.path in {"/synthesize", "/synthesize/batch"}:
            try:
                input_data = dict(request.query_params) if request.method == "GET" else await request.json()
            except Exception:
                input_data = {}
            _log_synthesize_request(
                route=request.url.path,
                method=request.method,
                status="http_400",
                started=time.perf_counter(),
                count=0,
                text_chars=0,
                input_data=input_data,
                error=str(exc.errors()),
            )
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})

    @app.on_event("startup")
    async def startup_event():
        """Load enabled engines in one deterministic startup sequence."""
        global _speaker_routes, _lang_speaker_map, _splitter, _splitter_languages
        global _matcha_backend, _matcha_batcher, _seed_vc_backend, _rvc_backend
        global _seed_vc_supported_voice_ids, _seed_vc_voice_ids

        startup_started = time.perf_counter()
        _LOGGER.info("Loading server startup order=pipertts,qwen3,matcha,seed_vc,rvc config=%s", CONFIG_PATH)
        with _logged_startup_step("reset_runtime_state"):
            _inference_cache.clear()
            _lang_speaker_map.clear()
            _speaker_routes.clear()
            _splitter = None
            _splitter_languages = None
            _matcha_backend = None
            _matcha_batcher = None
            _seed_vc_backend = None
            _rvc_backend = None
            _seed_vc_supported_voice_ids = None
            _seed_vc_voice_ids = None

        if _engine_enabled("pipertts"):
            with _logged_startup_step("pipertts"):
                required_models = _required_piper_models()
                if not required_models:
                    raise RuntimeError("PiperTTS is enabled but no Sparrow models are configured or available")

                for locale, speaker in _server_config.pipertts.lang_speaker_map.items():
                    canonical = _normalize_locale(locale)
                    _lang_speaker_map[canonical] = speaker

                _LOGGER.info("Sparrow required models count=%d models=%s", len(required_models), required_models)
                _preload_models(required_models, strict=True)
                _LOGGER.info("Sparrow loaded models=%s", list(_inference_cache.keys()))
                _preload_piper_text_models()

                route_models = _server_config.pipertts.model_priority or _allowed_models()
                if route_models:
                    _LOGGER.info("Loading PiperTTS speaker routes models=%s", route_models)
                    _speaker_routes = _build_speaker_routes(route_models)
                    _LOGGER.info("Loaded PiperTTS speaker routes speakers=%d locales=%d", len(_speaker_routes), len(_lang_speaker_map))
        else:
            _LOGGER.info("PiperTTS backend disabled")

        if _engine_enabled("qwen3"):
            with _logged_startup_step(
                "qwen3",
                model=_server_config.qwen.model,
                device=_server_config.qwen.device,
                dtype=_server_config.qwen.dtype,
                dp_budget=_server_config.qwen.dp_budget.enabled,
            ):
                qwen3.preload_model(
                    background=False,
                    include_dp_budget=_server_config.qwen.dp_budget.enabled,
                )
        else:
            _LOGGER.info("Qwen3 TTS backend disabled")

        if _engine_enabled("matcha"):
            with _logged_startup_step(
                "matcha",
                device=_server_config.matcha.device,
                checkpoint=_server_config.matcha.checkpoint,
                vocoder=_server_config.matcha.vocoder,
            ):
                _matcha_backend = await asyncio.to_thread(ProductionMatchaBackend, _server_config.matcha)
                _matcha_batcher = ProductionMatchaBatcher(_matcha_backend, _server_config.matcha)
                _matcha_batcher.start()
        else:
            _LOGGER.info("Matcha backend disabled")

        if _engine_enabled("seed_vc"):
            with _logged_startup_step(
                "seed_vc",
                device=_server_config.seed_vc.device,
                root=_server_config.seed_vc.root,
                embeddings=_server_config.seed_vc.embeddings_hdf5_path,
            ):
                _seed_vc_backend = await asyncio.to_thread(_SeedVCBackend, _server_config.seed_vc)
                _LOGGER.info("Loading Seed-VC voice catalog manifest=%s", SEED_VC_VOICE_IDS_PATH)
                catalog_started = time.perf_counter()
                _seed_vc_supported_voice_ids = _get_seed_vc_supported_voice_ids()
                _LOGGER.info(
                    "Loaded Seed-VC voice catalog voices=%d elapsed=%.2fs",
                    len(_seed_vc_supported_voice_ids),
                    time.perf_counter() - catalog_started,
                )
        else:
            _LOGGER.info("Seed-VC backend disabled")

        if _engine_enabled("rvc"):
            with _logged_startup_step(
                "rvc",
                cache_size=_server_config.rvc.cache_size,
                preload_models=_server_config.rvc.preload_models,
            ):
                _rvc_backend = await asyncio.to_thread(_build_rvc_backend, _server_config.rvc)
        else:
            _LOGGER.info("RVC backend disabled")

        _LOGGER.info("Loaded server startup elapsed=%.2fs", time.perf_counter() - startup_started)

    @app.get("/")
    async def health():
        """Health check and server info."""
        # Build speaker list with locale mappings
        speakers = []
        seen_locales: set[str] = set()

        for locale, speaker in _lang_speaker_map.items():
            if speaker in _speaker_routes:
                model, sid = _speaker_routes[speaker]
                speakers.append({
                    "locale": locale,
                    "speaker": speaker,
                    "model": model,
                    "speaker_id": sid,
                })
                seen_locales.add(locale)

        for speaker, (model, sid) in _speaker_routes.items():
            if speaker not in seen_locales:
                speakers.append({
                    "locale": speaker,
                    "speaker": speaker,
                    "model": model,
                    "speaker_id": sid,
                })

        speakers.sort(key=lambda x: x["locale"])

        return {
            "status": "ok",
            "version": "0.1.0",
            "engines": {
                "pipertts": _engine_enabled("pipertts"),
                "qwen3": _engine_enabled("qwen3"),
                "matcha": _engine_enabled("matcha"),
                "seed_vc": _engine_enabled("seed_vc"),
                "rvc": _engine_enabled("rvc"),
            },
            "pipertts": {
                "enabled": _engine_enabled("pipertts"),
                "models_loaded": list(_inference_cache.keys()),
                "models_enabled": _allowed_models(),
                "max_models_in_cache": _server_config.pipertts.max_models_in_cache,
                "default_model": _server_config.pipertts.default_model,
            },
            "qwen3": {
                "enabled": _engine_enabled("qwen3"),
                **(qwen3.model_status() if _engine_enabled("qwen3") else {}),
            },
            "matcha": {
                "enabled": _engine_enabled("matcha"),
                "loaded": _matcha_backend is not None,
                "device": _server_config.matcha.device,
                "checkpoint": _server_config.matcha.checkpoint,
                "vocoder": _server_config.matcha.vocoder,
                "sample_rate": getattr(_matcha_backend, "sample_rate", _server_config.matcha.sample_rate),
                "n_mels": _server_config.matcha.n_mels,
                "max_batch_size": _server_config.matcha.max_batch_size,
                "batch_wait_ms": _server_config.matcha.batch_wait_ms,
            },
            "seed_vc": {
                "enabled": _engine_enabled("seed_vc"),
                "loaded": _seed_vc_backend is not None,
                "device": _server_config.seed_vc.device,
                "root": _server_config.seed_vc.root,
                "runtime_root": _server_config.seed_vc.runtime_root,
                "presets": sorted(_SeedVCBackend.model_presets),
            },
            "rvc": {
                "enabled": _engine_enabled("rvc"),
                "loaded": _rvc_backend is not None,
            },
            "speakers": speakers,
        }

    @app.get("/llms.txt", response_class=Response)
    async def llms_txt(request: Request):
        """LLM-readable API documentation."""
        return Response(content=_render_llms_txt(request), media_type="text/plain")

    @app.get("/synthesize/voices", response_model=SynthesizeVoicesResponse)
    async def list_synthesize_voices():
        """List voices and locales supported by /synthesize endpoints."""
        locales, voices = _build_synthesize_voices_catalog()
        return SynthesizeVoicesResponse(locales=locales, voices=voices)

    @app.get("/matcha/status")
    async def matcha_status():
        """Temporary Matcha backend status."""
        return {
            "enabled": _engine_enabled("matcha"),
            "loaded": _matcha_backend is not None,
            "device": _server_config.matcha.device,
            "checkpoint": _server_config.matcha.checkpoint,
            "vocoder": _server_config.matcha.vocoder,
            "sample_rate": getattr(_matcha_backend, "sample_rate", _server_config.matcha.sample_rate),
            "n_mels": _server_config.matcha.n_mels,
            "n_timesteps": _server_config.matcha.n_timesteps,
            "semantic": "always",
            "max_batch_size": _server_config.matcha.max_batch_size,
            "batch_wait_ms": _server_config.matcha.batch_wait_ms,
        }

    @app.post("/matcha/synthesize")
    async def matcha_synthesize(request: MatchaSynthesizeRequest):
        """Temporary Matcha synthesis endpoint with dynamic request batching."""
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        result = await _get_matcha_batcher().submit(request)
        headers = {
            "X-Matcha-Audio-Seconds": f"{result.audio_seconds:.6f}",
            "X-Matcha-Backend-Seconds": f"{result.backend_seconds:.6f}",
            "X-Matcha-Backend-RTF": f"{result.backend_rtf:.6f}",
            "X-Matcha-Model-RTF": f"{result.model_rtf:.6f}",
            "X-Matcha-Batch-Size": str(result.batch_size),
            "X-Matcha-Queue-Seconds": f"{result.queue_seconds:.6f}",
        }
        if request.format == "json":
            return {
                "audio_seconds": result.audio_seconds,
                "backend_seconds": result.backend_seconds,
                "backend_rtf": result.backend_rtf,
                "model_rtf": result.model_rtf,
                "batch_size": result.batch_size,
                "queue_seconds": result.queue_seconds,
                "text": result.text,
                "phoneme": result.phoneme_text,
            }
        _maybe_cleanup_gpu()
        return Response(content=_audio_to_wav_bytes(result.audio, result.sample_rate), media_type="audio/wav", headers=headers)

    @app.get("/seed-vc/status")
    async def seed_vc_status():
        """Embedded Seed-VC backend status."""
        return {
            "enabled": _engine_enabled("seed_vc"),
            "loaded": _seed_vc_backend is not None,
            "device": _server_config.seed_vc.device,
            "root": _server_config.seed_vc.root,
            "runtime_root": _server_config.seed_vc.runtime_root,
            "presets": sorted(_SeedVCBackend.model_presets),
        }

    @app.post("/vc")
    async def seed_vc_convert(request: SeedVCRequest):
        """Seed-VC voice conversion endpoint compatible with the standalone API."""
        if not request.audio:
            raise HTTPException(status_code=400, detail="audio is required")
        try:
            backend = _get_seed_vc_backend()
            # Resolve cached embedding check async; download reference in background
            emb_key, emb = backend._resolve_cached_embeddings(request)
            reference_path = None if emb is not None else await backend._fetch_sample(request)
            mp3_bytes = await asyncio.to_thread(
                backend._convert_with_reference, request, reference_path, emb_key if emb is not None else None, emb
            )
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("Seed-VC conversion failed")
            raise HTTPException(status_code=500, detail=f"Seed-VC conversion failed: {exc}") from exc
        _maybe_cleanup_gpu()
        return Response(content=mp3_bytes, media_type="audio/mpeg")

    @app.post("/vc-batch")
    async def seed_vc_convert_batch(request: SeedVCBatchRequest):
        """Batched Seed-VC conversion endpoint for shared target voice settings."""
        if not request.items:
            raise HTTPException(status_code=400, detail="items is required")
        if any(not item.audio for item in request.items):
            raise HTTPException(status_code=400, detail="all items require audio")
        try:
            started = time.perf_counter()
            backend = _get_seed_vc_backend()
            first = request.items[0]
            emb_key, emb = backend._resolve_cached_embeddings(first)
            reference_path = None if emb is not None else await backend._fetch_sample(first)
            result = await asyncio.to_thread(
                backend.convert_batch_request, request, reference_path,
                emb_key if emb is not None else None, emb,
            )
            result["wall_time_sec"] = time.perf_counter() - started
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("Seed-VC batch conversion failed")
            raise HTTPException(status_code=500, detail=f"Seed-VC batch conversion failed: {exc}") from exc
        _maybe_cleanup_gpu()
        return result

    @app.post("/find-voice")
    async def seed_vc_find_voice(request: SeedVCFindVoiceRequest):
        """Seed-VC compatible voice lookup endpoint."""
        try:
            backend = _get_seed_vc_backend()
            sample_request = SeedVCRequest(audio="", reference_url=request.reference_url, id=request.id)
            reference_path = await backend._fetch_sample(sample_request)
            voice_id = await asyncio.to_thread(backend.find_voice, request, reference_path)
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("Seed-VC find-voice failed")
            raise HTTPException(status_code=500, detail=f"Seed-VC find-voice failed: {exc}") from exc
        return {"voice_id": voice_id}

    @app.post("/enhance")
    async def seed_vc_enhance(request: SeedVCEnhanceRequest):
        """Seed-VC compatible audio enhancement endpoint."""
        try:
            backend = _get_seed_vc_backend()
            raw_path = backend.tmp_dir / request.id / "sample_raw.mp3"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(request.reference_url)
                resp.raise_for_status()
                raw_path.write_bytes(resp.content)
            mp3_bytes = await asyncio.to_thread(backend.enhance, request, raw_path)
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("Seed-VC enhance failed")
            raise HTTPException(status_code=500, detail=f"Seed-VC enhance failed: {exc}") from exc
        _maybe_cleanup_gpu()
        return Response(content=mp3_bytes, media_type="audio/mpeg")

    @app.get("/rvc/status")
    async def rvc_status():
        """RVC voice conversion backend status."""
        backend = _rvc_backend
        if backend is None:
            return {
                "enabled": _engine_enabled("rvc"),
                "loaded": False,
                "available_models": [],
            }
        return backend.status()

    @app.get("/rvc/models", response_model=list[str])
    async def rvc_models():
        """List available RVC model weights."""
        backend = _get_rvc_backend()
        return backend.list_models()

    @app.post("/rvc/convert")
    async def rvc_convert(request: RVCConvertRequest):
        """Convert audio through an RVC voice model.

        Provide ``audio`` as base64-encoded source audio.
        Returns converted audio in the requested ``format``.
        """
        if not request.audio:
            raise HTTPException(status_code=400, detail="audio is required")
        try:
            audio_bytes = base64.b64decode(request.audio)
            backend = _get_rvc_backend()
            result_bytes, sr = await asyncio.to_thread(
                backend.convert,
                audio_bytes=audio_bytes,
                model=request.model,
                f0_method=request.f0_method,
                pitch=request.pitch,
                index_rate=request.index_rate,
                rms_mix_rate=request.rms_mix_rate,
                protect=request.protect,
                output_format=request.format,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("RVC conversion failed")
            raise HTTPException(status_code=500, detail=f"RVC conversion failed: {exc}") from exc

        media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
        _maybe_cleanup_gpu()
        return Response(content=result_bytes, media_type=media_type)

    @app.post("/rvc/convert/batch", response_model=RVCBatchConvertResponse)
    async def rvc_convert_batch(request: RVCBatchConvertRequest):
        """Convert multiple audio items through one real RVC batch call."""
        if any(not item.audio for item in request.items):
            raise HTTPException(status_code=400, detail="all items must include audio")
        if request.index_rate != 0:
            raise HTTPException(
                status_code=400,
                detail="RVC real batch conversion does not support index_rate != 0",
            )
        try:
            audio_items = [base64.b64decode(item.audio) for item in request.items]
            backend = _get_rvc_backend()
            converted = await asyncio.to_thread(
                backend.convert_batch,
                audio_items=audio_items,
                model=request.model,
                f0_method=request.f0_method,
                pitch=request.pitch,
                index_rate=request.index_rate,
                rms_mix_rate=request.rms_mix_rate,
                protect=request.protect,
                output_format=request.format,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("RVC batch conversion failed")
            raise HTTPException(status_code=500, detail=f"RVC batch conversion failed: {exc}") from exc

        _maybe_cleanup_gpu()
        return RVCBatchConvertResponse(
            items=[
                RVCBatchConvertResponseItem(
                    audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
                    sample_rate=sample_rate,
                )
                for audio_bytes, sample_rate in converted
            ],
            count=len(converted),
        )

    async def _handle_synthesize(
        request: SynthesizeRequest,
        model: str | None = None,
    ) -> Response:
        # Validate exactly one of text or ssml is provided
        if request.text and request.ssml:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'ssml', not both")
        if not request.text and not request.ssml:
            raise HTTPException(status_code=400, detail="Must provide either 'text' or 'ssml'")
        if request.language is not None and model is not None:
            raise HTTPException(status_code=400, detail="Use either 'language' or 'model', not both")
        if not _engine_enabled("pipertts"):
            raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
        if request.sample_url is not None and request.voice_id is not None:
            raise HTTPException(status_code=400, detail="Use either 'voice_id' or 'sample_url', not both")
        if request.sample_url is not None and _seed_vc_style_requested(request):
            raise HTTPException(status_code=400, detail="'style' and 'styleIntensity' require 'voice_id'")
        if request.voice_id is not None:
            if model is not None:
                raise HTTPException(status_code=400, detail="Use either 'voice_id' or 'model', not both")
            response = await _synthesize_configured_voice(request)
            _maybe_cleanup_gpu()
            return response
        if _seed_vc_style_requested(request):
            raise HTTPException(status_code=400, detail="'style' and 'styleIntensity' require 'voice_id'")

        synth_kwargs = _synth_kwargs_from_request(request)

        if request.ssml:
            audio, sample_rate = _synthesize_ssml(
                request.ssml,
                language=request.language,
                **synth_kwargs,
            )
        elif model is None:
            audio, sample_rate = _synthesize_multilingual(
                request.text,
                language=request.language,
                neural=request.neural,
                **synth_kwargs,
            )
        else:
            inference = _get_inference(model)
            internal_speaker = None
            batch_audios = await asyncio.to_thread(
                inference.synthesize_batch,
                [request.text],
                speaker=internal_speaker,
                batch_size=1,
                neural=request.neural,
                **synth_kwargs,
            )
            audio = batch_audios[0]
            sample_rate = inference.sample_rate

        if request.sample_url is not None:
            converted, converted_sample_rate = await _convert_generated_audio_to_sample_batch(
                source_audios=[audio],
                source_sample_rates=[sample_rate],
                sample_url=request.sample_url,
                output_format=request.format,
            )
            audio_bytes, _ = converted[0]
            media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
            sample_rate = converted_sample_rate
            _maybe_cleanup_gpu()
            return Response(content=audio_bytes, media_type=media_type)

        # Convert to requested format
        if request.format == "mp3":
            audio_bytes = _audio_to_mp3_bytes(audio, sample_rate)
            media_type = "audio/mpeg"
        else:
            audio_bytes = _audio_to_wav_bytes(audio, sample_rate)
            media_type = "audio/wav"

        _maybe_cleanup_gpu()
        return Response(content=audio_bytes, media_type=media_type)

    @app.post("/synthesize")
    async def synthesize(
        request: SynthesizeRequest,
        fastapi_request: Request,
        model: str = Query(None, description="Model to use (overrides auto routing)"),
    ):
        """Synthesize text or SSML to speech.

        Provide either `text` (plain text) or `ssml` (SSML with <speak> wrapper), not both.

        By default, text is split by language and routed automatically.
        Use `language` to force one supported locale for the entire text.
        """
        started = time.perf_counter()
        try:
            response = await _handle_synthesize(request, model=model)
        except HTTPException as exc:
            _log_synthesize_request(
                route=fastapi_request.url.path,
                method=fastapi_request.method,
                status=f"http_{exc.status_code}",
                started=started,
                count=1,
                text_chars=_text_length(request.text),
                ssml_chars=_text_length(request.ssml),
                input_data=_request_model_input(request, model=model),
                error=str(exc.detail),
            )
            raise
        except Exception as exc:
            _log_synthesize_request(
                route=fastapi_request.url.path,
                method=fastapi_request.method,
                status="error",
                started=started,
                count=1,
                text_chars=_text_length(request.text),
                ssml_chars=_text_length(request.ssml),
                input_data=_request_model_input(request, model=model),
                error=type(exc).__name__,
            )
            raise
        _log_synthesize_request(
            route=fastapi_request.url.path,
            method=fastapi_request.method,
            status="ok",
            started=started,
            count=1,
            text_chars=_text_length(request.text),
            ssml_chars=_text_length(request.ssml),
            input_data=_request_model_input(request, model=model),
        )
        return response

    @app.post("/synthesize/batch", response_model=BatchSynthesizeResponse)
    async def synthesize_batch(request: BatchSynthesizeRequest, fastapi_request: Request):
        """Batched synthesis with independent /synthesize-shaped inputs."""
        started = time.perf_counter()
        try:
            result = await synthesize_mixed_batch(request)
        except HTTPException as exc:
            _log_synthesize_request(
                route=fastapi_request.url.path,
                method=fastapi_request.method,
                status=f"http_{exc.status_code}",
                started=started,
                count=len(request.items),
                text_chars=sum(_text_length(item.text) for item in request.items),
                ssml_chars=sum(_text_length(item.ssml) for item in request.items),
                input_data=_request_model_input(request),
                error=str(exc.detail),
            )
            raise
        except Exception as exc:
            _log_synthesize_request(
                route=fastapi_request.url.path,
                method=fastapi_request.method,
                status="error",
                started=started,
                count=len(request.items),
                text_chars=sum(_text_length(item.text) for item in request.items),
                ssml_chars=sum(_text_length(item.ssml) for item in request.items),
                input_data=_request_model_input(request),
                error=type(exc).__name__,
            )
            raise
        _log_synthesize_request(
            route=fastapi_request.url.path,
            method=fastapi_request.method,
            status="ok",
            started=started,
            count=len(request.items),
            text_chars=sum(_text_length(item.text) for item in request.items),
            ssml_chars=sum(_text_length(item.ssml) for item in request.items),
            input_data=_request_model_input(request),
        )
        _maybe_cleanup_gpu()
        return result

    @app.get("/synthesize")
    async def synthesize_get(
        fastapi_request: Request,
        text: Optional[str] = Query(None, description="Plain text to synthesize (mutually exclusive with ssml)"),
        ssml: Optional[str] = Query(None, description="SSML to synthesize, must be wrapped in <speak> tags (mutually exclusive with text)"),
        model: str = Query(None, description="Model to use (overrides auto routing)"),
        voice_id: Optional[str] = Query(None, description="Public voice id from data/seed-vc/voice_ids.txt"),
        sample_url: Optional[str] = Query(None, description="Reference sample URL; output is converted to this voice with Seed-VC"),
        language: Optional[str] = Query(None, description="Force full locale for the entire text, e.g. en-GB"),
        style: Optional[str] = Query(None, description="Seed-VC speech style for voice_id synthesis"),
        style_intensity: Annotated[
            Optional[float],
            Query(alias="styleIntensity", description="Seed-VC speech style intensity"),
        ] = None,
        options: Optional[str] = Query(
            None,
            description='Sparrow options as JSON, e.g. {"length_scale":1.2,"duration_sdp_ratio":0.2}',
        ),
        format: Literal["wav", "mp3"] = Query("wav", description="Output audio format (wav or mp3)"),
        neural: bool = Query(True, description="Use neural heteronym disambiguation"),
    ):
        """Synthesize text or SSML to speech (GET endpoint for easy testing).

        Provide either `text` (plain text) or `ssml` (SSML with <speak> wrapper), not both.

        By default, text is split by language and routed automatically.
        """
        started = time.perf_counter()
        get_input = {
            "text": text,
            "ssml": ssml,
            "model": model,
            "voice_id": voice_id,
            "sample_url": sample_url,
            "language": language,
            "style": style,
            "styleIntensity": style_intensity,
            "options": options,
            "format": format,
            "neural": neural,
        }
        try:
            parsed_options = SparrowSynthesizeOptions.model_validate_json(options) if options else None
        except ValidationError as exc:
            detail = jsonable_encoder(exc.errors())
            _log_synthesize_request(
                route=fastapi_request.url.path,
                method=fastapi_request.method,
                status="http_400",
                started=started,
                count=1,
                text_chars=_text_length(text),
                ssml_chars=_text_length(ssml),
                input_data=get_input,
                error=str(detail),
            )
            raise HTTPException(status_code=400, detail=detail) from exc

        synth_request = SynthesizeRequest(
            text=text,
            ssml=ssml,
            voice_id=voice_id,
            sample_url=sample_url,
            language=language,
            style=style,
            style_intensity=style_intensity,
            options=parsed_options,
            format=format,
            neural=neural,
        )
        try:
            response = await _handle_synthesize(synth_request, model=model)
        except HTTPException as exc:
            _log_synthesize_request(
                route=fastapi_request.url.path,
                method=fastapi_request.method,
                status=f"http_{exc.status_code}",
                started=started,
                count=1,
                text_chars=_text_length(synth_request.text),
                ssml_chars=_text_length(synth_request.ssml),
                input_data=get_input,
                error=str(exc.detail),
            )
            raise
        except Exception as exc:
            _log_synthesize_request(
                route=fastapi_request.url.path,
                method=fastapi_request.method,
                status="error",
                started=started,
                count=1,
                text_chars=_text_length(synth_request.text),
                ssml_chars=_text_length(synth_request.ssml),
                input_data=get_input,
                error=type(exc).__name__,
            )
            raise
        _log_synthesize_request(
            route=fastapi_request.url.path,
            method=fastapi_request.method,
            status="ok",
            started=started,
            count=1,
            text_chars=_text_length(synth_request.text),
            ssml_chars=_text_length(synth_request.ssml),
            input_data=get_input,
        )
        return response

    return app


class _BasicAuthWrapper:
    def __init__(self, app, username: str, password: str):
        self.app = app
        self.username = username
        self.password = password

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        auth_header = ""
        for key, value in scope.get("headers", []):
            if key.lower() == b"authorization":
                auth_header = value.decode("latin1")
                break

        if not _basic_auth_matches(auth_header, self.username, self.password):
            response = _basic_auth_challenge()
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _basic_auth_matches(auth_header: str, username: str, password: str) -> bool:
    if not auth_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
        submitted_username, submitted_password = decoded.split(":", 1)
    except Exception:
        return False
    return secrets.compare_digest(submitted_username, username) and secrets.compare_digest(
        submitted_password,
        password,
    )


def _basic_auth_challenge() -> Response:
    return Response(
        "Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="qwen3-demo"'},
    )


def _mount_qwen_demo(app: FastAPI) -> None:
    """Mount the bundled faster-qwen3-tts demo inside this server."""
    if not qwen3.env_bool("QWEN_TTS_DEMO", True):
        return

    demo_password = os.environ.get("QWEN_TTS_DEMO_PASSWORD")
    if not demo_password:
        _LOGGER.warning("Qwen3 demo disabled: QWEN_TTS_DEMO_PASSWORD is not set")
        return

    demo_server = Path(__file__).resolve().parents[1] / "qwen3_demo" / "server.py"
    if not demo_server.exists():
        _LOGGER.warning("Qwen3 demo server not found: %s", demo_server)
        return

    os.environ["LZ_TTS_EMBEDDED_DEMO"] = "1"
    spec = importlib.util.spec_from_file_location("lz_tts_faster_qwen3_demo", demo_server)
    if spec is None or spec.loader is None:
        _LOGGER.warning("Could not load Qwen3 demo server: %s", demo_server)
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @app.get("/qwen3/demo", include_in_schema=False)
    async def qwen_demo_entrypoint(request: Request):
        if not _basic_auth_matches(request.headers.get("authorization", ""), "admin", demo_password):
            return _basic_auth_challenge()
        return RedirectResponse("/qwen3/demo/")

    app.mount("/qwen3/demo", _BasicAuthWrapper(module.app, "admin", demo_password))
    _LOGGER.info("Mounted Qwen3 demo at /qwen3/demo")


app = create_app()


def run():
    """Run the server with uvicorn."""
    import os

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    _LOGGER.info("Starting server at http://%s:%d", host, port)
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)
