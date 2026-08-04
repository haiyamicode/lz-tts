"""FastAPI server for Sparrow/VITS TTS inference."""

from __future__ import annotations

import base64
import asyncio
import contextlib
import gc
import hashlib
import json
import logging
import os
import re
import secrets
import signal
import sys
import threading
import time
import httpx
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Literal, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit

import numpy as np
import torch
from asgi_compression import BrotliAlgorithm, CompressionMiddleware, GzipAlgorithm, ZstdAlgorithm
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from ..multilingual_splitter import MultilingualSplitter
from ..piper import PiperInference
from ..ssml import BreakSegment, TextSegment, generate_silence, parse_ssml
from ..text_norm import normalize_spoken_text
from ..matcha_inference import MatchaBackend as ProductionStarlingBackend
from ..matcha_inference import MatchaBatchRequest
from ..matcha_inference import MatchaBatcher as ProductionStarlingBatcher
from .request_decompression import RequestDecompressionMiddleware
from .audio_utils import _audio_to_mp3_bytes, _audio_to_wav_bytes, _resample_audio
from .model_workers import seed_vc_worker_main, sparrow_worker_main, starling_worker_main
from .seed_vc_backend import (
    SeedVCBackend as _SeedVCBackend,
    SeedVCBatchRequest,
    SeedVCEnhanceRequest,
    SeedVCFindVoiceRequest,
    SeedVCRequest,
)
from .voxcpm_runtime import VoxCPMRuntime
from .worker_common import WorkerProcessClient

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
_LOGGER = logging.getLogger(__name__)
load_dotenv()

BINARY_RESPONSE_HEADERS = {"Content-Encoding": "identity"}


def _process_exit_description(exitcode: int | None) -> str:
    if exitcode is None:
        return "exitcode unknown"
    if exitcode < 0:
        signum = -exitcode
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"signal {signum}"
        return f"exitcode {exitcode} ({signame})"
    return f"exitcode {exitcode}"


def _binary_response(content: bytes, media_type: str, headers: dict[str, str] | None = None) -> Response:
    response_headers = dict(BINARY_RESPONSE_HEADERS)
    response_headers.update(headers or {})
    response_headers.setdefault("Content-Encoding", "identity")
    return Response(content=content, media_type=media_type, headers=response_headers)

# Default paths
DATA_DIR = Path("data")
CONFIG_PATH = Path(os.environ.get("LZ_TTS_SERVER_CONFIG", "local/server.json"))
LLMS_TEMPLATE_PATH = Path(__file__).with_name("llms.txt")
DEFAULT_MODEL = "lzspeech-sparrow"
PUBLIC_SPARROW_MODEL = "sparrow"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_VC_ROOT = PROJECT_ROOT / "data" / "seed-vc"
SEED_VC_RUNTIME_ROOT = PROJECT_ROOT / "src" / "seed_vc_runtime"
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
    languages: Optional[list[str]] = None


class EngineEnableConfig(BaseModel):
    """Global engine switches. Disabled engines are not mounted or loaded."""

    pipertts: bool = Field(default_factory=lambda: _env_bool("PIPER_TTS_ENABLED", True))
    voxcpm: bool = Field(default_factory=lambda: _env_bool("VOXCPM_ENABLED", False))
    starling: bool = Field(default_factory=lambda: _env_bool("STARLING_TTS_ENABLED", _env_bool("MATCHA_TTS_ENABLED", False)))
    matcha: bool = Field(default_factory=lambda: _env_bool("MATCHA_TTS_ENABLED", False))
    seed_vc: bool = Field(default_factory=lambda: _env_bool("SEED_VC_ENABLED", True))


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


class VoxCPMDurationBudgetConfig(BaseModel):
    """Sparrow DP settings used to derive a VoxCPM generation limit per text."""

    enabled: bool = True
    preload: bool = True
    use_bert: bool = False
    checkpoint: str = "data/lzspeech-sparrow/model.ckpt"
    config_path: Optional[str] = None
    device: str = "cuda:0"
    language: str = "multilingual"
    noise_scale: float = Field(default=0.8, ge=0)
    length_scale: float = Field(default=1.0, gt=0)
    token_rate: float = Field(default=6.25, gt=0)
    samples: int = Field(default=32, ge=1)
    upper_quantile: float = Field(default=0.90, ge=0, le=1)
    min_margin: float = Field(default=1.0, gt=0)
    max_margin: float = Field(default=1.35, gt=0)
    min_extra_tokens: int = Field(default=0, ge=0)
    max_extra_tokens: int = Field(default=38, ge=0)
    language_profiles: dict[str, dict[str, float | int]] = Field(default_factory=dict)


class VoxCPMConfig(BaseModel):
    """Optimized nano-vLLM VoxCPM2 serving configuration."""

    enabled: bool = Field(default_factory=lambda: _env_bool("VOXCPM_ENABLED", False))
    preload: bool = Field(default_factory=lambda: _env_bool("VOXCPM_PRELOAD", True))
    model_id: Literal["voxcpm"] = "voxcpm"
    model_path: str = Field(default_factory=lambda: os.environ.get("VOXCPM_MODEL_PATH", "data/voxcpm2-stable"))
    device: int = Field(default_factory=lambda: int(os.environ.get("VOXCPM_DEVICE", "1")), ge=0)
    inference_timesteps: int = Field(default=10, ge=1)
    max_num_batched_tokens: int = Field(default=8192, ge=1)
    max_num_seqs: int = Field(default=12, ge=1)
    max_model_len: int = Field(default=4096, ge=1)
    gpu_memory_utilization: float = Field(default=0.62, gt=0, le=1)
    num_kvcache_blocks: int = Field(default=192, ge=1)
    enforce_eager: bool = False
    fallback_max_generate_length: int = Field(default=4096, ge=1)
    duration_budget: VoxCPMDurationBudgetConfig = Field(default_factory=VoxCPMDurationBudgetConfig)
    temperature: float = Field(default=1.0, gt=0)
    cfg_value: float = Field(default=2.0, ge=0)
    reference_cache_size: int = Field(default=128, ge=1)
    preset_voice_catalog_path: str = "data/voice-presets.json"


class MatchaConfig(BaseModel):
    """Starling backend configuration."""

    enabled: bool = Field(default_factory=lambda: _env_bool("STARLING_TTS_ENABLED", _env_bool("MATCHA_TTS_ENABLED", False)))
    preload: bool = Field(default_factory=lambda: _env_bool("STARLING_TTS_PRELOAD", _env_bool("MATCHA_TTS_PRELOAD", True)))
    device: str = Field(default_factory=lambda: os.environ.get("STARLING_TTS_DEVICE", os.environ.get("MATCHA_TTS_DEVICE", "cuda:2")))
    checkpoint: str = Field(default_factory=lambda: os.environ.get("STARLING_TTS_CHECKPOINT", os.environ.get("MATCHA_TTS_CHECKPOINT", "")))
    config_path: str = Field(default_factory=lambda: os.environ.get("STARLING_TTS_CONFIG", "data/lzspeech-starling/config.json"))
    semantic_model_name: str = Field(
        default_factory=lambda: os.environ.get("STARLING_TTS_SEMANTIC_MODEL", "distilbert/distilbert-base-multilingual-cased")
    )
    semantic_max_tokens: Optional[int] = Field(
        default_factory=lambda: None
        if os.environ.get("STARLING_TTS_SEMANTIC_MAX_TOKENS", "").lower() in {"", "none", "null"}
        else int(os.environ.get("STARLING_TTS_SEMANTIC_MAX_TOKENS", "0"))
    )
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
        default_factory=lambda: os.environ.get(
            "SEED_VC_EMBEDDINGS_HDF5",
            "data/seed-vc/embeddings/vtts_embeddings_sparrow_fallback.h5",
        )
    )
    tmp_dir: str = Field(default_factory=lambda: os.environ.get("SEED_VC_TMP_DIR", "data/seed-vc/tmp"))
    output_dir: str = Field(default_factory=lambda: os.environ.get("SEED_VC_OUTPUT_DIR", "data/seed-vc/output"))
    voice_samples_dir: str = Field(
        default_factory=lambda: os.environ.get("SEED_VC_VOICE_SAMPLES_DIR", "data/seed-vc/voice-samples")
    )
    fp16: bool = Field(default_factory=lambda: _env_bool("SEED_VC_FP16", True))
    embedding_cache_size: int = Field(
        default_factory=lambda: int(os.environ.get("SEED_VC_EMBEDDING_CACHE_SIZE", "64")),
        ge=1,
    )
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


class ServerConfig(BaseModel):
    """Server configuration."""

    engines: EngineEnableConfig = Field(default_factory=EngineEnableConfig)
    pipertts: PiperTTSConfig = Field(default_factory=PiperTTSConfig)
    voxcpm: VoxCPMConfig = Field(default_factory=VoxCPMConfig)
    starling: MatchaConfig = Field(default_factory=MatchaConfig)
    matcha: MatchaConfig = Field(default_factory=MatchaConfig)
    seed_vc: SeedVCConfig = Field(default_factory=SeedVCConfig)


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
    voice_id: Optional[str] = Field(None, description="Configured preset voice id, e.g. msa.en-US.AvaMultilingual")
    sample_url: Optional[str] = Field(
        None,
        description="Reference sample URL; used natively by VoxCPM or for Seed-VC conversion with Sparrow",
    )
    reference_version: Optional[str] = Field(
        None,
        min_length=1,
        max_length=256,
        description="Opaque reference revision or content hash used for cache invalidation; requires sample_url",
    )
    language: Optional[str] = Field(None, description="Force full locale for the entire input, e.g. en-GB")
    model: Optional[str] = Field(None, description="Public model family, e.g. sparrow or voxcpm")
    seed: Optional[int] = Field(None, ge=0, description="Optional VoxCPM sampling seed")
    style: Optional[str] = Field(None, description="Preset reference style for voice_id synthesis")
    style_intensity: Optional[float] = Field(None, alias="styleIntensity", description="Legacy preset style intensity")
    options: Optional[SparrowSynthesizeOptions] = Field(None, description="Sparrow/VITS-specific synthesis options")
    format: Literal["wav", "mp3"] = Field("wav", description="Output audio format (wav or mp3)")
    neural: bool = Field(True, description="Use neural heteronym disambiguation for more accurate pronunciation of ambiguous words")


class BatchSynthesizeInputItem(BaseModel):
    """One item in a /synthesize/batch request."""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    text: Optional[str] = Field(None, description="Plain text to synthesize")
    ssml: Optional[str] = Field(None, description="SSML input is not supported for batched synthesis")
    voice_id: Optional[str] = Field(None, description="Configured preset voice id, e.g. msa.en-US.AvaMultilingual")
    sample_url: Optional[str] = Field(
        None,
        description="Reference sample URL; used natively by VoxCPM or for Seed-VC conversion with Sparrow",
    )
    reference_version: Optional[str] = Field(
        None,
        min_length=1,
        max_length=256,
        description="Opaque reference revision or content hash used for cache invalidation; requires sample_url",
    )
    language: Optional[str] = Field(None, description="Force full locale for this item, e.g. en-GB")
    model: Optional[str] = Field(None, description="Public model family, e.g. sparrow or voxcpm")
    seed: Optional[int] = Field(None, ge=0, description="Optional VoxCPM sampling seed")
    style: Optional[str] = Field(None, description="Preset reference style for voice_id synthesis")
    style_intensity: Optional[float] = Field(None, alias="styleIntensity", description="Legacy preset style intensity")
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
    seeds: list[int | None] | None = None
    voice_id: str | None = None
    sample_url: str | None = None
    reference_version: str | None = None
    language: str | None = None
    languages: list[str | None] | None = None
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
    """Request body for the Starling backend."""

    text: str
    language: str = Field("en", description="Language code used for phonemization and speaker/language conditioning")
    format: Literal["wav", "json"] = "wav"
    input_type: Literal["aligned"] = "aligned"
    speaker_id: Optional[int] = Field(None, description="Override language speaker id; 0 means auto")
    neural: bool = True
    steps: Optional[int] = None
    temperature: Optional[float] = None
    length_scale: Optional[float] = None



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
    _LOGGER.info(
        "Synthesize stage: %s",
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
_starling_backend: "ProductionStarlingBackend | None" = None
_starling_batcher: "ProductionStarlingBatcher | None" = None
_seed_vc_backend: "_SeedVCBackend | None" = None
_sparrow_worker: WorkerProcessClient | None = None
_sparrow_model_info: dict[str, dict[str, Any]] = {}
_starling_worker: WorkerProcessClient | None = None
_starling_info: dict[str, Any] = {}
_seed_vc_worker: WorkerProcessClient | None = None
_seed_vc_info: dict[str, Any] = {}
_voxcpm_runtime: VoxCPMRuntime | None = None
_voxcpm_reference_download_cache: OrderedDict[
    tuple[str, str], tuple[bytes, str]
] = OrderedDict()
_voxcpm_reference_download_tasks: dict[
    tuple[str, str], asyncio.Task[tuple[bytes, str]]
] = {}
_voxcpm_reference_download_lock = asyncio.Lock()
_voice_preset_catalog: dict[str, dict[str, Any]] | None = None
_voxcpm_preset_voices: dict[str, dict[str, Any]] | None = None
_seed_vc_fallback_voices: dict[str, dict[str, Any]] | None = None
_startup_loader_task: asyncio.Task | None = None


@dataclass
class _EngineLoadState:
    name: str
    ready: threading.Event
    status: str = "disabled"
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


_engine_load_states: dict[str, _EngineLoadState] = {
    name: _EngineLoadState(name=name, ready=threading.Event())
    for name in ("pipertts", "voxcpm", "starling", "seed_vc")
}

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
        return ServerConfig()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "starling" not in data and "matcha" in data:
        data["starling"] = data["matcha"]
    engines = data.get("engines")
    if isinstance(engines, dict) and "starling" not in engines and "matcha" in engines:
        engines["starling"] = engines["matcha"]
    return ServerConfig(**data)


def _engine_enabled(
    engine: Literal["pipertts", "voxcpm", "starling", "matcha", "seed_vc"],
    config: ServerConfig | None = None,
) -> bool:
    cfg = config or _server_config
    if engine == "starling":
        return bool(cfg.engines.starling or cfg.engines.matcha)
    if engine == "matcha":
        return bool(cfg.engines.matcha or cfg.engines.starling)
    return bool(getattr(cfg.engines, engine))


def _is_voxcpm_model(model: str | None) -> bool:
    if model is None:
        return False
    return model.strip().lower() == _server_config.voxcpm.model_id


def _public_model_name(model: str) -> str:
    """Collapse internal Sparrow checkpoints into one public model family."""
    return PUBLIC_SPARROW_MODEL if model in _allowed_models() else model


def _public_model_names(models: list[str]) -> list[str]:
    return list(dict.fromkeys(_public_model_name(model) for model in models))


def _resolve_api_model(model: str | None) -> str | None:
    """Resolve a public API model family to its internal serving model."""
    if model is None:
        return None

    normalized = model.strip().lower()
    if normalized in {PUBLIC_SPARROW_MODEL, DEFAULT_MODEL}:
        return PUBLIC_SPARROW_MODEL
    if _is_voxcpm_model(normalized):
        return _server_config.voxcpm.model_id
    if model in _allowed_models():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Internal Sparrow model {model!r} is not exposed; use model='sparrow' "
                "with language or voice_id routing"
            ),
        )
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported model {model!r}; supported public models: ['sparrow', '{_server_config.voxcpm.model_id}']",
    )


def _resolved_synthesize_model(body_model: str | None, query_model: str | None) -> str | None:
    resolved_body = _resolve_api_model(body_model)
    resolved_query = _resolve_api_model(query_model)
    if resolved_body and resolved_query and resolved_body != resolved_query:
        raise HTTPException(
            status_code=400,
            detail="Conflicting 'model' values in request body and query string",
        )
    return resolved_body or resolved_query


def _engine_state(engine: str) -> _EngineLoadState:
    return _engine_load_states[engine]


def _mark_engine_disabled(engine: str) -> None:
    state = _engine_state(engine)
    state.status = "disabled"
    state.error = None
    state.started_at = None
    state.finished_at = None
    state.ready.clear()


def _mark_engine_loading(engine: str) -> None:
    state = _engine_state(engine)
    state.status = "loading"
    state.error = None
    state.started_at = time.perf_counter()
    state.finished_at = None
    state.ready.clear()


def _mark_engine_ready(engine: str) -> None:
    state = _engine_state(engine)
    state.status = "ready"
    state.error = None
    state.finished_at = time.perf_counter()
    state.ready.set()


def _mark_engine_failed(engine: str, exc: BaseException) -> None:
    state = _engine_state(engine)
    state.status = "error"
    state.error = str(exc)
    state.finished_at = time.perf_counter()
    state.ready.set()


def _engine_status(engine: str) -> dict[str, Any]:
    state = _engine_state(engine)
    elapsed = None
    if state.started_at is not None:
        end = state.finished_at or time.perf_counter()
        elapsed = round(end - state.started_at, 3)
    return {
        "status": state.status,
        "ready": state.status == "ready",
        "loading_seconds": elapsed,
        **({"error": state.error} if state.error else {}),
    }


def _wait_for_engine_ready(engine: str, *, timeout: float | None = None) -> None:
    if not _engine_enabled("starling" if engine == "starling" else engine):  # type: ignore[arg-type]
        raise HTTPException(status_code=503, detail=f"{engine} backend is disabled")

    state = _engine_state(engine)
    if state.status == "ready":
        return
    if state.status == "disabled":
        raise HTTPException(status_code=503, detail=f"{engine} backend is disabled")

    if state.status == "loading":
        _LOGGER.info("Waiting for %s backend readiness", engine)
        if not state.ready.wait(timeout=timeout):
            elapsed = time.perf_counter() - state.started_at if state.started_at is not None else None
            elapsed_text = f" after {elapsed:.1f}s" if elapsed is not None else ""
            raise HTTPException(
                status_code=503,
                detail=f"{engine} backend is still loading{elapsed_text}",
            )

    if state.status == "ready":
        return
    if state.status == "error":
        raise HTTPException(status_code=503, detail=f"{engine} backend failed to load: {state.error}")
    raise HTTPException(status_code=503, detail=f"{engine} backend is still loading")


async def _await_engine_ready(engine: str, *, timeout: float | None = None) -> None:
    await asyncio.to_thread(_wait_for_engine_ready, engine, timeout=timeout)


def _worker_settings_data() -> dict[str, Any]:
    return _server_config.model_dump(mode="json")


def _ensure_sparrow_worker() -> WorkerProcessClient:
    global _sparrow_worker
    if _sparrow_worker is None:
        _sparrow_worker = WorkerProcessClient(
            name="sparrow",
            target=sparrow_worker_main,
            args=(_worker_settings_data(),),
        )
    return _sparrow_worker


def _ensure_starling_worker() -> WorkerProcessClient:
    global _starling_worker
    if _starling_worker is None:
        _starling_worker = WorkerProcessClient(
            name="starling",
            target=starling_worker_main,
            args=(_worker_settings_data(),),
        )
    return _starling_worker


def _ensure_seed_vc_worker() -> WorkerProcessClient:
    global _seed_vc_worker
    if _seed_vc_worker is None:
        _seed_vc_worker = WorkerProcessClient(
            name="seed-vc",
            target=seed_vc_worker_main,
            args=(_worker_settings_data(),),
        )
    return _seed_vc_worker


def _stop_model_workers() -> None:
    global _sparrow_worker, _starling_worker, _seed_vc_worker
    for worker in (_sparrow_worker, _starling_worker, _seed_vc_worker):
        if worker is not None:
            worker.stop()
    _sparrow_worker = None
    _starling_worker = None
    _seed_vc_worker = None


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


class _SparrowInferenceProxy:
    """Parent-process stand-in for a Sparrow model owned by the worker."""

    def __init__(self, model: str, info: dict[str, Any]):
        self.model = model
        self.sample_rate = int(info.get("sample_rate") or 22050)
        self.speakers = dict(info.get("speakers") or {})
        self.use_bert = bool(info.get("use_bert", False))

    def synthesize_batch(
        self,
        texts: list[str],
        *,
        speaker: Any = None,
        batch_size: int = 1,
        neural: bool = True,
        **synth_kwargs: Any,
    ) -> list[np.ndarray]:
        response = _ensure_sparrow_worker().call(
            "synthesize_batch",
            {
                "model": self.model,
                "texts": texts,
                "speaker": speaker,
                "batch_size": batch_size,
                "neural": neural,
                "synth_kwargs": synth_kwargs,
            },
        )
        data = response.get("data") or {}
        self.sample_rate = int(data.get("sample_rate") or self.sample_rate)
        return list(data.get("audios") or [])

    def synthesize_span(
        self,
        text: str,
        *,
        speaker: Any = None,
        neural: bool = True,
        **synth_kwargs: Any,
    ) -> np.ndarray:
        response = _ensure_sparrow_worker().call(
            "synthesize_span",
            {
                "model": self.model,
                "text": text,
                "speaker": speaker,
                "neural": neural,
                "synth_kwargs": synth_kwargs,
            },
        )
        data = response.get("data") or {}
        self.sample_rate = int(data.get("sample_rate") or self.sample_rate)
        return data.get("audio")


def _get_inference(model: str) -> PiperInference:
    """Get an already loaded inference instance for a model."""
    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    _wait_for_engine_ready("pipertts")
    if _sparrow_worker is not None:
        if model in _sparrow_model_info:
            return _SparrowInferenceProxy(model, _sparrow_model_info[model])
        if _is_model_allowed(model):
            raise HTTPException(status_code=503, detail=f"Model was not loaded at startup: {model}")
        raise HTTPException(status_code=404, detail=f"Model is not configured for this server: {model}")
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

    try:
        from ..piper.context_replacer import get_replacer
        replacer = get_replacer(device=device)
        replacer.load()
        _LOGGER.info("Loaded context replacer")
    except FileNotFoundError:
        _LOGGER.info("Context replacer checkpoint not found, skipping")
    except Exception:
        _LOGGER.debug("Context replacer not available", exc_info=True)

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


def _configured_root_voice_for_voice_id(voice_id: str | None) -> RootVoiceConfig | None:
    if not voice_id:
        return None
    for config in _server_config.pipertts.root_voices.values():
        if config.voice_id == voice_id:
            return config
    return None


@dataclass(frozen=True)
class _VoxCPMPresetReference:
    voice_id: str
    style: str
    url: str | None = None
    path: Path | None = None


def _load_voice_preset_catalog() -> dict[str, dict[str, Any]]:
    global _voice_preset_catalog
    if _voice_preset_catalog is not None:
        return _voice_preset_catalog
    catalog_path = _resolve_project_path(_server_config.voxcpm.preset_voice_catalog_path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to load preset voice catalog {catalog_path}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise RuntimeError(f"Preset voice catalog {catalog_path} must have version 1")
    voices = payload.get("voices")
    if not isinstance(voices, list):
        raise RuntimeError(f"Preset voice catalog {catalog_path} must contain a voices array")

    catalog: dict[str, dict[str, Any]] = {}
    for entry in voices:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Preset voice catalog {catalog_path} contains a non-object voice")
        voice_id = entry.get("id")
        language = entry.get("language")
        pipeline = entry.get("pipeline")
        if not isinstance(voice_id, str) or not voice_id:
            raise RuntimeError(f"Preset voice catalog {catalog_path} contains an invalid voice id")
        if voice_id in catalog:
            raise RuntimeError(f"Preset voice catalog {catalog_path} contains duplicate voice {voice_id!r}")
        if not isinstance(language, str) or not language:
            raise RuntimeError(f"Preset voice {voice_id!r} has no language")
        if pipeline == "voxcpm":
            references = entry.get("references")
            if not isinstance(references, dict) or "general" not in references:
                raise RuntimeError(f"VoxCPM preset voice {voice_id!r} has no general reference")
            for style, reference in references.items():
                if not isinstance(style, str) or not style or not isinstance(reference, dict):
                    raise RuntimeError(f"VoxCPM preset voice {voice_id!r} has an invalid reference")
                url = reference.get("url")
                path = reference.get("path")
                if bool(url) == bool(path):
                    raise RuntimeError(
                        f"VoxCPM preset voice {voice_id!r} style {style!r} must define exactly one URL or path"
                    )
                if url and (not isinstance(url, str) or not url.startswith(("https://", "http://"))):
                    raise RuntimeError(f"VoxCPM preset voice {voice_id!r} style {style!r} has an invalid URL")
                if path:
                    reference_path = _resolve_project_path(path)
                    if not reference_path.is_file():
                        raise RuntimeError(
                            f"VoxCPM preset voice {voice_id!r} style {style!r} path does not exist: "
                            f"{reference_path}"
                        )
        elif pipeline == "sparrow_seed_vc":
            if entry.get("embedding_style") != "general":
                raise RuntimeError(f"Seed-VC preset voice {voice_id!r} must use the general embedding")
        else:
            raise RuntimeError(f"Preset voice {voice_id!r} has unsupported pipeline {pipeline!r}")
        catalog[voice_id] = entry

    if not catalog:
        raise RuntimeError(f"Preset voice catalog {catalog_path} contains no voices")
    _voice_preset_catalog = catalog
    _LOGGER.info(
        "Loaded production voice preset catalog path=%s voices=%d voxcpm=%d seed_vc=%d",
        catalog_path,
        len(catalog),
        sum(entry["pipeline"] == "voxcpm" for entry in catalog.values()),
        sum(entry["pipeline"] == "sparrow_seed_vc" for entry in catalog.values()),
    )
    return catalog


def _load_voxcpm_preset_voices() -> dict[str, dict[str, Any]]:
    global _voxcpm_preset_voices
    if _voxcpm_preset_voices is None:
        _voxcpm_preset_voices = {
            voice_id: entry
            for voice_id, entry in _load_voice_preset_catalog().items()
            if entry["pipeline"] == "voxcpm"
        }
    return _voxcpm_preset_voices


def _seed_vc_fallback_language_codes() -> set[str]:
    return {
        _get_base_language(entry["language"])
        for entry in _load_seed_vc_fallback_voices().values()
    }


def _load_seed_vc_fallback_voices() -> dict[str, dict[str, Any]]:
    global _seed_vc_fallback_voices
    if _seed_vc_fallback_voices is not None:
        return _seed_vc_fallback_voices
    fallback_voices = {
        voice_id: entry
        for voice_id, entry in _load_voice_preset_catalog().items()
        if entry["pipeline"] == "sparrow_seed_vc"
    }

    if not fallback_voices:
        raise RuntimeError("Preset voice catalog contains no Seed-VC fallback voices")
    _seed_vc_fallback_voices = fallback_voices
    return fallback_voices


def _is_seed_vc_fallback_voice(voice_id: str | None) -> bool:
    return bool(voice_id) and voice_id in _load_seed_vc_fallback_voices()


def _validate_seed_vc_fallback_embeddings(worker_info: dict[str, Any]) -> None:
    fallback_voices = _load_seed_vc_fallback_voices()
    available_keys = set(worker_info.get("embedding_keys") or [])
    expected_keys = {f"{voice_id}.general" for voice_id in fallback_voices}
    missing_keys = sorted(expected_keys - available_keys)
    if missing_keys:
        raise RuntimeError(f"Seed-VC fallback embeddings are missing keys: {missing_keys}")

    unroutable_languages = sorted(
        language
        for language in _seed_vc_fallback_language_codes()
        if not _is_supported_sparrow_locale(language)
    )
    if unroutable_languages:
        raise RuntimeError(f"Seed-VC fallback languages are not routable by Sparrow: {unroutable_languages}")

    _LOGGER.info(
        "Validated Seed-VC fallback catalog languages=%s voices=%d embeddings=%d",
        sorted(_seed_vc_fallback_language_codes()),
        len(fallback_voices),
        len(available_keys),
    )


def _configured_voice_ids() -> set[str]:
    voice_ids = {config.voice_id for config in _server_config.pipertts.root_voices.values()}
    if _engine_enabled("voxcpm"):
        voice_ids.update(_load_voxcpm_preset_voices())
    if _engine_enabled("seed_vc"):
        voice_ids.update(_load_seed_vc_fallback_voices())
    return voice_ids


def _resolve_voxcpm_preset_reference(voice_id: str, requested_style: str | None) -> _VoxCPMPresetReference:
    entry = _load_voxcpm_preset_voices().get(voice_id)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"Unsupported VoxCPM preset voice_id {voice_id!r}")

    style = (requested_style or "general").strip() or "general"
    references = entry["references"]
    reference = references.get(style)
    if reference is None and entry.get("style_fallback") == "general":
        reference = references.get("general")
    if reference is None:
        available = sorted(references)
        raise HTTPException(
            status_code=400,
            detail=f"Voice {voice_id!r} has no reference sample for style {style!r}; available styles: {available}",
        )
    url = reference.get("url")
    path = reference.get("path")
    return _VoxCPMPresetReference(
        voice_id=voice_id,
        style=style,
        url=url,
        path=_resolve_project_path(path) if path else None,
    )


async def _synthesize_voxcpm_preset_batch(
    request: _SharedBatchSynthesizeRequest,
) -> BatchSynthesizeResponse:
    if request.voice_id is None:
        raise HTTPException(status_code=400, detail="voice_id is required for preset voice synthesis")

    return await _synthesize_voxcpm_preset_inputs(
        texts=request.texts,
        seeds=request.seeds or [None] * len(request.texts),
        voice_ids=[request.voice_id] * len(request.texts),
        languages=request.languages or [request.language] * len(request.texts),
        styles=[request.style] * len(request.texts),
        style_intensities=[request.style_intensity] * len(request.texts),
        options=request.options,
        output_format=request.format,
        neural=request.neural,
    )


async def _load_voxcpm_preset_reference(
    reference: _VoxCPMPresetReference,
) -> tuple[bytes, str]:
    if reference.path is not None:
        audio = await asyncio.to_thread(reference.path.read_bytes)
        return audio, reference.path.suffix.lower().lstrip(".") or "wav"
    if reference.url is not None:
        return await _download_voxcpm_reference(reference.url)
    raise RuntimeError(f"VoxCPM preset voice {reference.voice_id!r} has no reference source")


async def _synthesize_voxcpm_preset_inputs(
    *,
    texts: list[str],
    seeds: list[int | None],
    voice_ids: list[str],
    languages: list[str | None],
    styles: list[str | None],
    style_intensities: list[float | None],
    options: SparrowSynthesizeOptions | None,
    output_format: Literal["wav", "mp3"],
    neural: bool,
) -> BatchSynthesizeResponse:
    item_count = len(texts)
    per_item_values = {
        "seeds": seeds,
        "voice_ids": voice_ids,
        "languages": languages,
        "styles": styles,
        "style_intensities": style_intensities,
    }
    for name, values in per_item_values.items():
        if len(values) != item_count:
            raise RuntimeError(f"{name} length must match preset VoxCPM texts length")

    references = [
        _resolve_voxcpm_preset_reference(voice_id, style)
        for voice_id, style in zip(voice_ids, styles)
    ]
    unique_references: dict[tuple[str, str], _VoxCPMPresetReference] = {}
    reference_keys: list[tuple[str, str]] = []
    for reference in references:
        if reference.path is not None:
            key = ("path", str(reference.path))
        elif reference.url is not None:
            key = ("url", reference.url)
        else:
            raise RuntimeError(f"VoxCPM preset voice {reference.voice_id!r} has no reference source")
        unique_references.setdefault(key, reference)
        reference_keys.append(key)

    loaded_references = await asyncio.gather(
        *(_load_voxcpm_preset_reference(reference) for reference in unique_references.values())
    )
    loaded_by_key = dict(zip(unique_references, loaded_references))
    reference_audios = [loaded_by_key[key][0] for key in reference_keys]
    reference_formats = [loaded_by_key[key][1] for key in reference_keys]

    for voice_id, reference, intensity in zip(voice_ids, references, style_intensities):
        if intensity not in {None, 1.0}:
            _LOGGER.info(
                "Ignoring legacy styleIntensity for VoxCPM preset voice_id=%s style=%s intensity=%s",
                voice_id,
                reference.style,
                intensity,
            )

    _log_synthesize_batch_stage(
        "configured_voice_routing",
        pipeline="voxcpm_preset",
        voice_ids=voice_ids,
        styles=[reference.style for reference in references],
        reference_urls=[reference.url for reference in references],
        reference_paths=[
            str(reference.path) if reference.path is not None else None
            for reference in references
        ],
        item_count=item_count,
    )
    result = await synthesize_voxcpm_batch(
        _SharedBatchSynthesizeRequest(
            texts=texts,
            seeds=seeds,
            languages=languages,
            model=_server_config.voxcpm.model_id,
            options=options,
            format=output_format,
            neural=neural,
        ),
        reference_audios=reference_audios,
        reference_formats=reference_formats,
    )
    unique_voice_ids = set(voice_ids)
    model = f"voice_id:{voice_ids[0]}" if len(unique_voice_ids) == 1 else "voxcpm:mixed-presets"
    return result.model_copy(update={"model": model, "speaker": None})


async def _synthesize_voxcpm_preset_items(
    records: list[tuple[int, BatchSynthesizeInputItem, str]],
) -> BatchSynthesizeResponse:
    return await _synthesize_voxcpm_preset_inputs(
        texts=[text for _, _, text in records],
        seeds=[item.seed for _, item, _ in records],
        voice_ids=[item.voice_id or "" for _, item, _ in records],
        languages=[item.language for _, item, _ in records],
        styles=[item.style for _, item, _ in records],
        style_intensities=[item.style_intensity for _, item, _ in records],
        options=records[0][1].options,
        output_format=records[0][1].format,
        neural=records[0][1].neural,
    )


def _default_root_voice() -> RootVoiceConfig | None:
    for config in _server_config.pipertts.root_voices.values():
        if config.languages is None:
            return config
    return None


def _root_voice_can_synthesize_language(config: RootVoiceConfig, language: str | None) -> bool:
    if config.languages is None:
        return True
    if language is None:
        return False

    normalized = _normalize_locale_with_region(language)
    base_language = _get_base_language(normalized)

    for config_language in config.languages:
        config_language = _normalize_locale_with_region(config_language)
        config_base_language = _get_base_language(config_language)
        if "-" in config_language:
            if normalized == config_language or normalized == config_base_language:
                return True
        elif base_language == config_base_language:
            return True

    return False


def _build_synthesize_voices_catalog() -> tuple[list[str], list[SynthesizeVoiceInfo]]:
    supported_voice_ids = _configured_voice_ids()

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


def _seed_vc_sample_id(sample_url: str, reference_version: str | None = None) -> str:
    identity = f"{sample_url}\0{reference_version or ''}"
    return f"synthesize-sample-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def _seed_vc_chunk_batch_size(backend: SeedVCBackend) -> int:
    return max(1, int(backend.settings.max_chunk_batch_size))


async def _convert_generated_audio_to_sample_batch(
    *,
    source_audios: list[np.ndarray],
    source_sample_rates: list[int],
    sample_url: str,
    reference_version: str | None,
    output_format: Literal["wav", "mp3"],
) -> tuple[list[tuple[bytes, float]], int]:
    await _await_engine_ready("seed_vc")
    backend = _get_seed_vc_backend()
    sample_request = SeedVCRequest(
        audio="",
        reference_url=sample_url,
        id=_seed_vc_sample_id(sample_url, reference_version),
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
    if request.voice_id is None:
        raise HTTPException(status_code=400, detail="voice_id is required for configured voice synthesis")
    if request.sample_url is not None:
        raise HTTPException(status_code=400, detail="Use either 'voice_id' or 'sample_url', not both")
    if request.model is not None:
        raise HTTPException(status_code=400, detail="Use either 'voice_id' or 'model', not both")

    root_voice = _configured_root_voice_for_voice_id(request.voice_id)
    seed_vc_fallback = root_voice is None and _is_seed_vc_fallback_voice(request.voice_id)
    if root_voice is None and not seed_vc_fallback:
        return await _synthesize_voxcpm_preset_batch(request)

    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    if any(seed is not None for seed in request.seeds or []):
        raise HTTPException(status_code=400, detail="'seed' is only supported by VoxCPM preset voices")
    if root_voice is not None and _seed_vc_style_requested(request):
        raise HTTPException(status_code=400, detail="Native Sparrow root voices do not support preset styles")

    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")

    await _await_engine_ready("pipertts")
    supported_voice_ids = _configured_voice_ids()
    if request.voice_id not in supported_voice_ids:
        supported = sorted(supported_voice_ids)
        raise HTTPException(status_code=400, detail=f"Unsupported voice_id {request.voice_id!r}; supported voices: {supported}")

    requested_language = request.language
    if requested_language is None and seed_vc_fallback:
        requested_language = str(_load_seed_vc_fallback_voices()[request.voice_id].get("language") or "")
    elif requested_language is None and root_voice is not None and root_voice.languages:
        # A locale-specific root voice should use its configured language even
        # when the client omits `language`. Script/language detection is not
        # reliable for every LFL language (for example, Latin-script Bosnian).
        if len(root_voice.languages) == 1:
            requested_language = root_voice.languages[0]

    forced_language = _normalize_locale_with_region(requested_language) if requested_language else None
    if forced_language is not None and not _is_supported_sparrow_locale(forced_language):
        base_language = _get_base_language(forced_language)
        if seed_vc_fallback and _is_supported_sparrow_locale(base_language):
            forced_language = base_language
    if forced_language is not None:
        _resolve_forced_language(forced_language)

    primary_speaker: str | None = (
        forced_language
        if forced_language is not None
        else root_voice.speaker if root_voice is not None else None
    )
    style, style_intensity = _seed_vc_style_from_request(request)
    if seed_vc_fallback:
        await _await_engine_ready("seed_vc")
        try:
            _get_seed_vc_backend()._resolve_exact_cached_embeddings(
                request.voice_id,
                style,
                style_intensity,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()
    synth_kwargs = _synth_kwargs_from_request(request)
    item_segments: list[list[dict[str, Any]]] = []
    segment_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    for item_idx, text in enumerate(texts):
        segments, _ = _plan_text_segments(
            text,
            primary_speaker,
            forced_language=forced_language,
            validate_primary_speaker=False,
        )
        if root_voice is not None:
            fallback_root_voice = _default_root_voice()
            for segment in segments:
                if _root_voice_can_synthesize_language(root_voice, segment["lang"]):
                    if root_voice.speaker is not None:
                        segment["speaker"] = root_voice.speaker
                    segment["model"] = root_voice.model
                elif fallback_root_voice is not None:
                    if fallback_root_voice.speaker is not None:
                        segment["speaker"] = fallback_root_voice.speaker
                    segment["model"] = fallback_root_voice.model
        item_segments.append(segments)
        for segment_idx, segment in enumerate(segments):
            record = {**segment, "item_idx": item_idx, "segment_idx": segment_idx}
            segment_groups.setdefault(segment["model"], []).append(record)

    _log_synthesize_batch_stage(
        "configured_voice_routing",
        voice_id=request.voice_id,
        forced_language=forced_language,
        root_voice=bool(root_voice),
        root_voice_id=root_voice.voice_id if root_voice is not None else None,
        root_voice_model=root_voice.model if root_voice is not None else None,
        root_voice_speaker=root_voice.speaker if root_voice is not None else None,
        root_voice_languages=root_voice.languages if root_voice is not None else None,
        primary_speaker=primary_speaker,
        seed_vc_fallback=seed_vc_fallback,
        style=style,
        style_intensity=style_intensity,
        item_count=len(texts),
        item_segment_counts=[len(segments) for segments in item_segments],
        convert_indices=list(range(len(texts))) if seed_vc_fallback else [],
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
        if _is_starling_model(model_name):
            await _await_engine_ready("starling")
            starling_batcher = _get_starling_batcher()
            starling_length_scale = request.options.length_scale if request.options and request.options.length_scale is not None else None
            batch_started = time.perf_counter()
            _log_synthesize_batch_stage(
                "starling_batch_start",
                pipeline="configured_voice",
                voice_id=request.voice_id,
                model=model_name,
                item_count=len(texts),
                segment_count=len(records),
                batch_size=len(records),
                languages=sorted({str(record["lang"]) for record in records}),
                neural=request.neural,
            )
            batch_results = await asyncio.gather(
                *[
                    starling_batcher.submit(
                        MatchaSynthesizeRequest(
                            text=record["text"],
                            language=record["lang"],
                            format="wav",
                            neural=request.neural,
                            length_scale=starling_length_scale,
                        )
                    )
                    for record in records
                ]
            )
            audio_seconds = sum(result.audio_seconds for result in batch_results)
            elapsed = time.perf_counter() - batch_started
            _log_synthesize_batch_stage(
                "starling_batch_done",
                pipeline="configured_voice",
                voice_id=request.voice_id,
                model=model_name,
                output_count=len(batch_results),
                audio_seconds=round(audio_seconds, 6),
                wall_seconds=round(elapsed, 6),
                rtf=round(elapsed / audio_seconds, 6) if audio_seconds else 0.0,
                sample_rate=_server_config.starling.sample_rate,
            )
            for record, result in zip(records, batch_results):
                generated_segments[record["item_idx"]][record["segment_idx"]] = (result.audio, result.sample_rate)
            continue

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

    if seed_vc_fallback:
        backend = _get_seed_vc_backend()
        vc_source_rate = backend.sample_rate
        vc_source_audios = [
            _resample_audio(audio, source_rate, vc_source_rate)
            for audio, source_rate in zip(item_audios, item_source_sample_rates)
        ]
        vc_started = time.perf_counter()
        _log_synthesize_batch_stage(
            "seed_vc_voice_batch_start",
            pipeline="sparrow_seed_vc_fallback",
            voice_id=request.voice_id,
            count=len(vc_source_audios),
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
            pipeline="sparrow_seed_vc_fallback",
            voice_id=request.voice_id,
            output_count=len(converted),
            wall_seconds=round(time.perf_counter() - vc_started, 6),
            output_sample_rate=backend.sample_rate,
        )
        for idx, (audio_bytes, audio_seconds) in enumerate(converted):
            encoded_items[idx] = audio_bytes
            item_sample_rates[idx] = backend.sample_rate
            item_audio_seconds[idx] = audio_seconds

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
    response = BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=_public_model_name(root_voice.model) if root_voice is not None else PUBLIC_SPARROW_MODEL,
        speaker=primary_speaker,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )
    return response


async def _synthesize_configured_voice(request: SynthesizeRequest) -> Response:
    if request.ssml:
        raise HTTPException(status_code=400, detail="voice-id synthesis currently supports plain text only")
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required for voice-id synthesis")
    batch_result = await synthesize_configured_voice_batch(
        _SharedBatchSynthesizeRequest(
            texts=[request.text],
            seeds=[request.seed],
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
    return _binary_response(audio_bytes, media_type)


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



class _StarlingBatcherProxy:
    """Parent-process async proxy for the Starling worker."""

    async def submit(self, request: MatchaSynthesizeRequest) -> Any:
        queued_at = time.perf_counter()
        response = await asyncio.to_thread(
            _ensure_starling_worker().call,
            "synthesize_batch",
            {
                "items": [
                    {
                        "text": request.text,
                        "language": request.language,
                        "input_type": request.input_type,
                        "speaker_id": request.speaker_id,
                        "neural": request.neural,
                        "steps": request.steps,
                        "temperature": request.temperature,
                        "length_scale": request.length_scale,
                        "queued_at": queued_at,
                    }
                ]
            },
        )
        results = (response.get("data") or {}).get("results") or []
        if not results:
            raise HTTPException(status_code=502, detail="Starling worker returned no result")
        return results[0]


def _get_starling_batcher() -> ProductionStarlingBatcher:
    if not _engine_enabled("starling"):
        raise HTTPException(status_code=503, detail="Starling backend is disabled")
    _wait_for_engine_ready("starling")
    if _starling_worker is not None:
        return _StarlingBatcherProxy()
    if _starling_batcher is None:
        raise HTTPException(status_code=503, detail="Starling backend is not enabled or not loaded")
    return _starling_batcher


def _is_starling_model(model_name: str | None) -> bool:
    return bool(model_name) and str(model_name).lower() in {"lzspeech-starling", "starling"}


class _SeedVCBackendProxy:
    """Parent-process proxy for a Seed-VC worker-owned backend."""

    def __init__(self, info: dict[str, Any]):
        self.settings = _server_config.seed_vc
        self.sample_rate = int(info.get("sample_rate") or 22050)
        self.device = info.get("device") or self.settings.device
        self.root = Path(info.get("root") or _resolve_project_path(self.settings.root)).resolve()
        self.runtime_root = Path(info.get("runtime_root") or _resolve_project_path(self.settings.runtime_root)).resolve()
        self.tmp_dir = Path(info.get("tmp_dir") or _resolve_project_path(self.settings.tmp_dir)).resolve()
        self.output_dir = Path(info.get("output_dir") or _resolve_project_path(self.settings.output_dir)).resolve()
        self.voice_samples_dir = Path(
            info.get("voice_samples_dir") or _resolve_project_path(self.settings.voice_samples_dir)
        ).resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.cached_embeddings = None

    async def _fetch_sample(self, request: SeedVCRequest) -> Path:
        response = await asyncio.to_thread(
            _ensure_seed_vc_worker().call,
            "fetch_sample",
            {"request": request.model_dump(mode="json")},
        )
        return Path((response.get("data") or {})["path"])

    def _resolve_exact_cached_embeddings(self, voice_id: str, style: str, intensity: float) -> tuple[str, Any]:
        response = _ensure_seed_vc_worker().call(
            "resolve_exact_cached_embeddings",
            {"voice_id": voice_id, "style": style, "intensity": intensity},
        )
        return str((response.get("data") or {})["embedding_key"]), True

    def _resolve_cached_embeddings(self, request: SeedVCRequest) -> tuple[str, Any | None]:
        response = _ensure_seed_vc_worker().call(
            "resolve_cached_embeddings",
            {"request": request.model_dump(mode="json")},
        )
        data = response.get("data") or {}
        return str(data["embedding_key"]), True if data.get("cached") else None

    def _convert_with_reference(
        self,
        request: SeedVCRequest,
        reference_path: Path | None,
        embedding_key: str | None = None,
        cached_embeddings: Any | None = None,
    ) -> bytes:
        response = _ensure_seed_vc_worker().call(
            "convert_with_reference",
            {
                "request": request.model_dump(mode="json"),
                "reference_path": str(reference_path) if reference_path is not None else None,
                "embedding_key": embedding_key if cached_embeddings is not None else None,
            },
        )
        data = response.get("data") or {}
        self.sample_rate = int(data.get("sample_rate") or self.sample_rate)
        return data["audio"]

    def convert_batch_request(
        self,
        request: SeedVCBatchRequest,
        reference_path: Path | None = None,
        embedding_key: str | None = None,
        cached_embeddings: Any | None = None,
    ) -> dict[str, Any]:
        response = _ensure_seed_vc_worker().call(
            "convert_batch_request",
            {
                "request": request.model_dump(mode="json"),
                "reference_path": str(reference_path) if reference_path is not None else None,
                "embedding_key": embedding_key if cached_embeddings is not None else None,
            },
        )
        return response.get("data") or {}

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
        response = _ensure_seed_vc_worker().call(
            "convert_generated_audio_batch",
            {
                "source_audios": source_audios,
                "source_sample_rate": source_sample_rate,
                "voice_id": voice_id,
                "style": style,
                "intensity": intensity,
                "preset": preset,
                "output_format": output_format,
                "max_chunk_batch_size": max_chunk_batch_size,
                "strict_embedding": strict_embedding,
            },
        )
        data = response.get("data") or {}
        self.sample_rate = int(data.get("sample_rate") or self.sample_rate)
        return list(data.get("items") or [])

    def convert_generated_audio_reference_batch(
        self,
        source_audios: list[np.ndarray],
        source_sample_rate: int,
        reference_path: Path,
        preset: str | None,
        output_format: Literal["wav", "mp3"],
        max_chunk_batch_size: int | None = None,
    ) -> list[tuple[bytes, float]]:
        response = _ensure_seed_vc_worker().call(
            "convert_generated_audio_reference_batch",
            {
                "source_audios": source_audios,
                "source_sample_rate": source_sample_rate,
                "reference_path": str(reference_path),
                "preset": preset,
                "output_format": output_format,
                "max_chunk_batch_size": max_chunk_batch_size,
            },
        )
        data = response.get("data") or {}
        self.sample_rate = int(data.get("sample_rate") or self.sample_rate)
        return list(data.get("items") or [])

    def find_voice(self, request: SeedVCFindVoiceRequest, reference_path: Path) -> str:
        response = _ensure_seed_vc_worker().call(
            "find_voice",
            {"request": request.model_dump(mode="json"), "reference_path": str(reference_path)},
        )
        return str((response.get("data") or {})["voice_id"])

    def enhance(self, request: SeedVCEnhanceRequest, raw_path: Path) -> bytes:
        response = _ensure_seed_vc_worker().call(
            "enhance",
            {"request": request.model_dump(mode="json"), "raw_path": str(raw_path)},
        )
        return (response.get("data") or {})["audio"]


def _get_seed_vc_backend() -> _SeedVCBackend:
    if not _engine_enabled("seed_vc"):
        raise HTTPException(status_code=503, detail="Seed-VC backend is disabled")
    _wait_for_engine_ready("seed_vc")
    if _seed_vc_worker is not None:
        return _SeedVCBackendProxy(_seed_vc_info)
    if _seed_vc_backend is None:
        raise HTTPException(status_code=503, detail="Seed-VC backend was not loaded at startup")
    return _seed_vc_backend


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
            reference_version=request.reference_version,
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
        response = BatchSynthesizeResponse(
            items=items,
            count=len(items),
            model=PUBLIC_SPARROW_MODEL,
            speaker=resolved_speaker,
            wall_seconds=total_wall_seconds,
            audio_seconds_total=audio_seconds_total,
            rtf=(total_wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
        )
        return response

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

    response = BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=PUBLIC_SPARROW_MODEL,
        speaker=resolved_speaker,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )
    return response


async def synthesize_multilingual_sparrow_batch(request: _SharedBatchSynthesizeRequest) -> BatchSynthesizeResponse:
    """Run real batched Sparrow synthesis for auto-routed multilingual text items."""
    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    if request.voice_id is not None:
        raise HTTPException(status_code=400, detail="voice_id requests must use configured voice synthesis")
    if request.model not in {None, PUBLIC_SPARROW_MODEL}:
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

    _log_synthesize_batch_stage(
        "auto_multilingual_routing",
        item_count=len(texts),
        item_segment_counts=[len(segments) for segments in item_segments],
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
            reference_version=request.reference_version,
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
        response = BatchSynthesizeResponse(
            items=items,
            count=len(items),
            model=PUBLIC_SPARROW_MODEL,
            speaker=None,
            wall_seconds=wall_seconds,
            audio_seconds_total=audio_seconds_total,
            rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
        )
        return response

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
    response = BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=PUBLIC_SPARROW_MODEL,
        speaker=None,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )
    return response


def _prepare_voxcpm_input(text: str, language: str | None) -> tuple[str, str]:
    if language is not None:
        return normalize_spoken_text(text, language), language

    result = _get_multilingual_splitter().split(text)
    main_language = result.main_language or "en"
    prepared_segments = []
    for segment in result.segments:
        segment_text = segment.text.strip()
        if not segment_text:
            continue
        segment_language = (
            segment.language
            if segment.language and segment.language != "und"
            else main_language
        )
        prepared_segments.append(normalize_spoken_text(segment_text, segment_language))
    prepared_text = (
        " ".join(prepared_segments)
        if prepared_segments
        else normalize_spoken_text(text, main_language)
    )
    return prepared_text, main_language


def _prepare_voxcpm_text(text: str, language: str | None) -> str:
    return _prepare_voxcpm_input(text, language)[0]


def _get_voxcpm_runtime() -> VoxCPMRuntime:
    if _voxcpm_runtime is None:
        raise HTTPException(status_code=503, detail="VoxCPM backend was not loaded at startup")
    return _voxcpm_runtime


async def _fetch_voxcpm_reference(sample_url: str) -> tuple[bytes, str]:
    parsed = urlsplit(sample_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="VoxCPM sample_url must use http or https")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(sample_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch VoxCPM sample_url: {exc}") from exc

    audio = response.content
    if not audio:
        raise HTTPException(status_code=400, detail="VoxCPM sample_url returned an empty response")
    suffix = Path(parsed.path).suffix.lower().lstrip(".")
    return audio, suffix or "wav"


async def _download_voxcpm_reference(
    sample_url: str,
    reference_version: str | None = None,
) -> tuple[bytes, str]:
    if reference_version is None:
        return await _fetch_voxcpm_reference(sample_url)

    cache_key = (sample_url, reference_version)
    async with _voxcpm_reference_download_lock:
        cached = _voxcpm_reference_download_cache.get(cache_key)
        if cached is not None:
            _voxcpm_reference_download_cache.move_to_end(cache_key)
            return cached
        task = _voxcpm_reference_download_tasks.get(cache_key)
        if task is None:
            task = asyncio.create_task(_fetch_voxcpm_reference(sample_url))
            _voxcpm_reference_download_tasks[cache_key] = task

    try:
        loaded = await asyncio.shield(task)
    except BaseException:
        async with _voxcpm_reference_download_lock:
            if _voxcpm_reference_download_tasks.get(cache_key) is task and task.done():
                _voxcpm_reference_download_tasks.pop(cache_key, None)
        raise

    async with _voxcpm_reference_download_lock:
        _voxcpm_reference_download_tasks.pop(cache_key, None)
        cached = _voxcpm_reference_download_cache.get(cache_key)
        if cached is not None:
            _voxcpm_reference_download_cache.move_to_end(cache_key)
            return cached
        _voxcpm_reference_download_cache[cache_key] = loaded
        cache_size = int(_server_config.voxcpm.reference_cache_size)
        while len(_voxcpm_reference_download_cache) > cache_size:
            _voxcpm_reference_download_cache.popitem(last=False)
    return loaded


async def synthesize_voxcpm_batch(
    request: _SharedBatchSynthesizeRequest,
    *,
    reference_audio: bytes | None = None,
    reference_format: str | None = None,
    reference_audios: list[bytes | None] | None = None,
    reference_formats: list[str] | None = None,
) -> BatchSynthesizeResponse:
    """Synthesize one compatible request group with optimized nano-vLLM."""
    if not _engine_enabled("voxcpm"):
        raise HTTPException(status_code=503, detail="VoxCPM backend is disabled")
    if request.voice_id is not None:
        raise HTTPException(status_code=400, detail="VoxCPM model routing does not support voice_id")
    if request.options is not None:
        raise HTTPException(status_code=400, detail="Sparrow options are not valid for VoxCPM")

    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")
    languages = request.languages or [request.language] * len(texts)
    if len(languages) != len(texts):
        raise HTTPException(status_code=400, detail="VoxCPM languages length must match texts length")

    await _await_engine_ready("voxcpm")
    runtime = _get_voxcpm_runtime()
    prepared_inputs = [
        _prepare_voxcpm_input(text, language)
        for text, language in zip(texts, languages)
    ]
    prepared_texts = [prepared_text for prepared_text, _ in prepared_inputs]
    dp_languages = [dp_language for _, dp_language in prepared_inputs]
    if reference_audio is not None and reference_audios is not None:
        raise HTTPException(status_code=400, detail="VoxCPM received both shared and per-item reference audio")
    if request.sample_url is not None and (reference_audio is not None or reference_audios is not None):
        raise HTTPException(status_code=400, detail="VoxCPM received both inline and URL reference audio")
    resolved_reference_format = reference_format or "wav"
    if request.sample_url is not None:
        reference_audio, resolved_reference_format = await _download_voxcpm_reference(
            request.sample_url,
            request.reference_version,
        )

    started = time.perf_counter()
    audios = await runtime.synthesize_batch(
        prepared_texts,
        languages=dp_languages,
        seeds=request.seeds,
        reference_audio=reference_audio,
        reference_format=resolved_reference_format,
        reference_audios=reference_audios,
        reference_formats=reference_formats,
    )
    generation_wall_seconds = time.perf_counter() - started
    sample_rate = runtime.sample_rate

    items = []
    for text, audio in zip(texts, audios):
        audio_seconds = float(len(audio)) / sample_rate if sample_rate else 0.0
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

    audio_seconds_total = sum(item.audio_seconds for item in items)
    wall_seconds = time.perf_counter() - started
    _log_synthesize_batch_stage(
        "voxcpm_batch_done",
        model=_server_config.voxcpm.model_id,
        item_count=len(items),
        sample_rate=sample_rate,
        generation_wall_seconds=round(generation_wall_seconds, 6),
        wall_seconds=round(wall_seconds, 6),
        audio_seconds=round(audio_seconds_total, 6),
        rtf=round(wall_seconds / audio_seconds_total, 6) if audio_seconds_total else 0.0,
    )
    response = BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=_server_config.voxcpm.model_id,
        speaker=None,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )
    return response


async def _synthesize_voxcpm_items(
    records: list[tuple[int, BatchSynthesizeInputItem, str]],
) -> BatchSynthesizeResponse:
    reference_keys = [
        (item.sample_url, item.reference_version)
        for _, item, _ in records
    ]
    unique_keys = list(
        dict.fromkeys(key for key in reference_keys if key[0] is not None)
    )
    loaded_references = await asyncio.gather(
        *(
            _download_voxcpm_reference(sample_url, reference_version)
            for sample_url, reference_version in unique_keys
            if sample_url is not None
        )
    )
    loaded_by_key = dict(zip(unique_keys, loaded_references))
    reference_audios = [
        loaded_by_key[key][0] if key[0] is not None else None
        for key in reference_keys
    ]
    reference_formats = [
        loaded_by_key[key][1] if key[0] is not None else "wav"
        for key in reference_keys
    ]
    first = records[0][1]
    return await synthesize_voxcpm_batch(
        _SharedBatchSynthesizeRequest(
            texts=[text for _, _, text in records],
            seeds=[item.seed for _, item, _ in records],
            languages=[item.language for _, item, _ in records],
            model=_server_config.voxcpm.model_id,
            options=first.options,
            format=first.format,
            neural=first.neural,
        ),
        reference_audios=reference_audios,
        reference_formats=reference_formats,
    )


def _batch_item_group_key(item: BatchSynthesizeInputItem) -> tuple[Any, ...]:
    options_key = None
    if item.options is not None:
        options_key = json.dumps(item.options.model_dump(mode="json"), sort_keys=True)
    model_key = item.model
    if item.voice_id is not None:
        if (
            _configured_root_voice_for_voice_id(item.voice_id) is None
            and not _is_seed_vc_fallback_voice(item.voice_id)
        ):
            return (
                "voxcpm_preset",
                options_key,
                item.format,
                item.neural,
            )
        kind = "voice"
    elif _is_voxcpm_model(item.model):
        return (
            "voxcpm",
            options_key,
            item.format,
            item.neural,
        )
    else:
        kind = "sparrow"
    return (
        kind,
        item.voice_id,
        item.sample_url,
        item.reference_version,
        item.language,
        model_key,
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
    if item.reference_version is not None and item.sample_url is None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'reference_version' requires 'sample_url'")
    if item.voice_id is not None and item.model is not None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: Use either 'voice_id' or 'model', not both")
    if item.language is not None and item.model is not None and not _is_voxcpm_model(item.model):
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: Use either 'language' or 'model', not both")
    seed_uses_voxcpm_preset = (
        item.voice_id is not None
        and _configured_root_voice_for_voice_id(item.voice_id) is None
        and not _is_seed_vc_fallback_voice(item.voice_id)
    )
    if item.seed is not None and not _is_voxcpm_model(item.model) and not seed_uses_voxcpm_preset:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'seed' requires model='voxcpm'")
    if item.sample_url is not None and _seed_vc_style_requested(item):
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'style' and 'styleIntensity' require 'voice_id'")
    if item.voice_id is None and _seed_vc_style_requested(item):
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'style' and 'styleIntensity' require 'voice_id'")
    return item.text.strip()


def _shared_batch_from_items(records: list[tuple[int, BatchSynthesizeInputItem, str]]) -> _SharedBatchSynthesizeRequest:
    first = records[0][1]
    return _SharedBatchSynthesizeRequest(
        texts=[text for _, _, text in records],
        seeds=[item.seed for _, item, _ in records],
        voice_id=first.voice_id,
        sample_url=first.sample_url,
        reference_version=first.reference_version,
        language=first.language,
        languages=[item.language for _, item, _ in records],
        model=_resolve_api_model(first.model),
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
                "voice_ids": [item.voice_id for _, item, _ in records],
                "sample_url": bool(records[0][1].sample_url),
                "languages": [item.language for _, item, _ in records],
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
        is_voxcpm_preset_group = (
            records[0][1].voice_id is not None
            and _configured_root_voice_for_voice_id(records[0][1].voice_id) is None
            and not _is_seed_vc_fallback_voice(records[0][1].voice_id)
        )
        group_started = time.perf_counter()
        _log_synthesize_batch_stage(
            "group_start",
            group_index=group_idx,
            item_indices=[item_idx for item_idx, _, _ in records],
            item_count=len(records),
            voice_ids=[item.voice_id for _, item, _ in records],
            sample_url=bool(shared_request.sample_url),
            languages=[item.language for _, item, _ in records],
            model=shared_request.model,
            format=shared_request.format,
        )
        if is_voxcpm_preset_group:
            group_result = await _synthesize_voxcpm_preset_items(records)
        elif shared_request.voice_id is not None:
            group_result = await synthesize_configured_voice_batch(shared_request)
        elif _is_voxcpm_model(shared_request.model):
            group_result = await _synthesize_voxcpm_items(records)
        elif shared_request.language is not None:
            await _await_engine_ready("pipertts")
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
        elif shared_request.model == PUBLIC_SPARROW_MODEL:
            await _await_engine_ready("pipertts")
            group_result = await synthesize_multilingual_sparrow_batch(shared_request)
        elif shared_request.model is not None:
            await _await_engine_ready("pipertts")
            group_result = await synthesize_sparrow_batch(shared_request)
        else:
            await _await_engine_ready("pipertts")
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
        """Schedule enabled model engines to load in background worker processes."""
        global _speaker_routes, _lang_speaker_map, _splitter, _splitter_languages
        global _starling_backend, _starling_batcher, _seed_vc_backend
        global _voxcpm_runtime, _voice_preset_catalog, _voxcpm_preset_voices, _seed_vc_fallback_voices
        global _sparrow_model_info, _starling_info, _seed_vc_info
        global _startup_loader_task

        startup_started = time.perf_counter()
        _LOGGER.info("Scheduling server startup mode=early-online-parallel-workers config=%s", CONFIG_PATH)
        with _logged_startup_step("reset_runtime_state"):
            if _startup_loader_task is not None and not _startup_loader_task.done():
                _startup_loader_task.cancel()
            if _voxcpm_runtime is not None:
                await _voxcpm_runtime.stop()
                _voxcpm_runtime = None
            _voice_preset_catalog = None
            _voxcpm_preset_voices = None
            _seed_vc_fallback_voices = None
            _stop_model_workers()
            _inference_cache.clear()
            _lang_speaker_map.clear()
            _speaker_routes.clear()
            _splitter = None
            _splitter_languages = None
            _starling_backend = None
            _starling_batcher = None
            _seed_vc_backend = None
            _sparrow_model_info = {}
            _starling_info = {}
            _seed_vc_info = {}
            for engine in _engine_load_states:
                if _engine_enabled("starling" if engine == "starling" else engine):  # type: ignore[arg-type]
                    _mark_engine_loading(engine)
                else:
                    _mark_engine_disabled(engine)

        async def run_loader(engine: str, loader: Callable[[], Awaitable[None]]) -> None:
            try:
                await loader()
                _mark_engine_ready(engine)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.exception("Failed loading %s backend", engine)
                _mark_engine_failed(engine, exc)

        async def load_models_background() -> None:
            load_started = time.perf_counter()
            startup_tasks: list[asyncio.Task] = []

            if _engine_enabled("pipertts"):
                required_models = _required_piper_models()
                if not required_models:
                    _mark_engine_failed("pipertts", RuntimeError("PiperTTS is enabled but no Sparrow models are configured or available"))
                else:
                    async def start_sparrow() -> None:
                        global _sparrow_model_info
                        with _logged_startup_step("sparrow_worker", models=required_models):
                            worker = _ensure_sparrow_worker()
                            worker.start()
                            response = await asyncio.to_thread(worker.call, "health")
                            data = response.get("data") or {}
                            _sparrow_model_info = dict(data.get("models") or {})
                            _LOGGER.info("Sparrow worker loaded models=%s", list(_sparrow_model_info.keys()))

                    startup_tasks.append(asyncio.create_task(run_loader("pipertts", start_sparrow)))
            else:
                _LOGGER.info("PiperTTS backend disabled")

            if _engine_enabled("voxcpm"):
                async def start_voxcpm() -> None:
                    global _voxcpm_runtime
                    with _logged_startup_step(
                        "voxcpm_nanovllm",
                        model=_server_config.voxcpm.model_path,
                        model_id=_server_config.voxcpm.model_id,
                        device=_server_config.voxcpm.device,
                        kv_blocks=_server_config.voxcpm.num_kvcache_blocks,
                    ):
                        runtime = VoxCPMRuntime(_server_config.voxcpm.model_dump(mode="python"))
                        await runtime.start()
                        _voxcpm_runtime = runtime

                startup_tasks.append(asyncio.create_task(run_loader("voxcpm", start_voxcpm)))
            else:
                _LOGGER.info("VoxCPM backend disabled")

            if _engine_enabled("starling"):
                async def start_starling() -> None:
                    global _starling_info
                    with _logged_startup_step(
                        "starling_worker",
                        device=_server_config.starling.device,
                        checkpoint=_server_config.starling.checkpoint,
                        vocoder=_server_config.starling.vocoder,
                    ):
                        worker = _ensure_starling_worker()
                        worker.start()
                        response = await asyncio.to_thread(worker.call, "health")
                        _starling_info = dict(response.get("data") or {})

                startup_tasks.append(asyncio.create_task(run_loader("starling", start_starling)))
            else:
                _LOGGER.info("Starling backend disabled")

            if _engine_enabled("seed_vc"):
                async def start_seed_vc() -> None:
                    global _seed_vc_info
                    with _logged_startup_step(
                        "seed_vc_worker",
                        device=_server_config.seed_vc.device,
                        root=_server_config.seed_vc.root,
                    ):
                        worker = _ensure_seed_vc_worker()
                        worker.start()
                        response = await asyncio.to_thread(worker.call, "health")
                        _seed_vc_info = dict(response.get("data") or {})
                        _validate_seed_vc_fallback_embeddings(_seed_vc_info)

                startup_tasks.append(asyncio.create_task(run_loader("seed_vc", start_seed_vc)))
            else:
                _LOGGER.info("Seed-VC backend disabled")

            if startup_tasks:
                await asyncio.gather(*startup_tasks)
            _LOGGER.info("Loaded server background models elapsed=%.2fs", time.perf_counter() - load_started)

        if _engine_enabled("pipertts"):
            for locale, speaker in _server_config.pipertts.lang_speaker_map.items():
                canonical = _normalize_locale(locale)
                _lang_speaker_map[canonical] = speaker
            route_models = _server_config.pipertts.model_priority or _allowed_models()
            if route_models:
                _LOGGER.info("Loading PiperTTS speaker routes models=%s", route_models)
                _speaker_routes = _build_speaker_routes(route_models)
                _LOGGER.info("Loaded PiperTTS speaker routes speakers=%d locales=%d", len(_speaker_routes), len(_lang_speaker_map))
        else:
            _LOGGER.info("PiperTTS backend disabled")

        _startup_loader_task = asyncio.create_task(load_models_background())
        _LOGGER.info("Server startup scheduled background loading elapsed=%.2fs", time.perf_counter() - startup_started)

    @app.on_event("shutdown")
    async def shutdown_event():
        global _startup_loader_task, _voxcpm_runtime, _voice_preset_catalog
        global _voxcpm_preset_voices, _seed_vc_fallback_voices
        if _startup_loader_task is not None and not _startup_loader_task.done():
            _startup_loader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _startup_loader_task
        _startup_loader_task = None
        if _voxcpm_runtime is not None:
            await _voxcpm_runtime.stop()
            _voxcpm_runtime = None
        _voice_preset_catalog = None
        _voxcpm_preset_voices = None
        _seed_vc_fallback_voices = None
        _stop_model_workers()

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
                    "model": _public_model_name(model),
                    "speaker_id": sid,
                })
                seen_locales.add(locale)

        for speaker, (model, sid) in _speaker_routes.items():
            if speaker not in seen_locales:
                speakers.append({
                    "locale": speaker,
                    "speaker": speaker,
                    "model": _public_model_name(model),
                    "speaker_id": sid,
                })

        speakers.sort(key=lambda x: x["locale"])

        return {
            "status": "ok",
            "version": "0.1.0",
            "engines": {
                "pipertts": _engine_enabled("pipertts"),
                "voxcpm": _engine_enabled("voxcpm"),
                "starling": _engine_enabled("starling"),
                "seed_vc": _engine_enabled("seed_vc"),
            },
            "pipertts": {
                "enabled": _engine_enabled("pipertts"),
                **_engine_status("pipertts"),
                "models_loaded": _public_model_names(
                    list(_sparrow_model_info.keys()) if _sparrow_worker is not None else list(_inference_cache.keys())
                ),
                "models_enabled": _public_model_names(_allowed_models()),
                "max_models_in_cache": _server_config.pipertts.max_models_in_cache,
                "default_model": PUBLIC_SPARROW_MODEL,
            },
            "voxcpm": {
                "enabled": _engine_enabled("voxcpm"),
                **_engine_status("voxcpm"),
                "loaded": _voxcpm_runtime is not None,
                "model": _server_config.voxcpm.model_id,
                "model_path": _server_config.voxcpm.model_path,
                "device": _server_config.voxcpm.device,
                "num_kvcache_blocks": _server_config.voxcpm.num_kvcache_blocks,
                "sample_rate": (
                    _voxcpm_runtime.sample_rate
                    if _voxcpm_runtime is not None
                    else None
                ),
            },
            "starling": {
                "enabled": _engine_enabled("starling"),
                **_engine_status("starling"),
                "loaded": (_starling_worker is not None and bool(_starling_info)) or _starling_backend is not None,
                "device": _server_config.starling.device,
                "checkpoint": _server_config.starling.checkpoint,
                "vocoder": _server_config.starling.vocoder,
                "sample_rate": _starling_info.get("sample_rate") or getattr(_starling_backend, "sample_rate", _server_config.starling.sample_rate),
                "n_mels": _server_config.starling.n_mels,
                "max_batch_size": _server_config.starling.max_batch_size,
                "batch_wait_ms": _server_config.starling.batch_wait_ms,
            },
            "seed_vc": {
                "enabled": _engine_enabled("seed_vc"),
                **_engine_status("seed_vc"),
                "loaded": (_seed_vc_worker is not None and bool(_seed_vc_info)) or _seed_vc_backend is not None,
                "device": _server_config.seed_vc.device,
                "root": _server_config.seed_vc.root,
                "runtime_root": _server_config.seed_vc.runtime_root,
                "presets": sorted(_SeedVCBackend.model_presets),
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
        if _engine_enabled("seed_vc"):
            await _await_engine_ready("seed_vc", timeout=15)
        locales, voices = _build_synthesize_voices_catalog()
        return SynthesizeVoicesResponse(locales=locales, voices=voices)

    @app.get("/starling/status")
    @app.get("/matcha/status")
    async def starling_status():
        """Starling backend status."""
        return {
            "enabled": _engine_enabled("starling"),
            **_engine_status("starling"),
            "loaded": (_starling_worker is not None and bool(_starling_info)) or _starling_backend is not None,
            "device": _server_config.starling.device,
            "checkpoint": _server_config.starling.checkpoint,
            "vocoder": _server_config.starling.vocoder,
            "sample_rate": _starling_info.get("sample_rate") or getattr(_starling_backend, "sample_rate", _server_config.starling.sample_rate),
            "n_mels": _server_config.starling.n_mels,
            "n_timesteps": _server_config.starling.n_timesteps,
            "semantic": "always",
            "max_batch_size": _server_config.starling.max_batch_size,
            "batch_wait_ms": _server_config.starling.batch_wait_ms,
        }

    @app.get("/seed-vc/status")
    async def seed_vc_status():
        """Embedded Seed-VC backend status."""
        return {
            "enabled": _engine_enabled("seed_vc"),
            **_engine_status("seed_vc"),
            "loaded": (_seed_vc_worker is not None and bool(_seed_vc_info)) or _seed_vc_backend is not None,
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
            await _await_engine_ready("seed_vc")
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
        return _binary_response(mp3_bytes, "audio/mpeg")

    @app.post("/vc-batch")
    async def seed_vc_convert_batch(request: SeedVCBatchRequest):
        """Batched Seed-VC conversion endpoint for shared target voice settings."""
        if not request.items:
            raise HTTPException(status_code=400, detail="items is required")
        if any(not item.audio for item in request.items):
            raise HTTPException(status_code=400, detail="all items require audio")
        try:
            await _await_engine_ready("seed_vc")
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
            await _await_engine_ready("seed_vc")
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
            await _await_engine_ready("seed_vc")
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
        return _binary_response(mp3_bytes, "audio/mpeg")

    async def _handle_synthesize(
        request: SynthesizeRequest,
        model: str | None = None,
    ) -> Response:
        # Validate exactly one of text or ssml is provided
        if request.text and request.ssml:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'ssml', not both")
        if not request.text and not request.ssml:
            raise HTTPException(status_code=400, detail="Must provide either 'text' or 'ssml'")
        if request.language is not None and model is not None and not _is_voxcpm_model(model):
            raise HTTPException(status_code=400, detail="Use either 'language' or 'model', not both")
        seed_uses_voxcpm_preset = (
            request.voice_id is not None
            and _configured_root_voice_for_voice_id(request.voice_id) is None
            and not _is_seed_vc_fallback_voice(request.voice_id)
        )
        if request.seed is not None and not _is_voxcpm_model(model) and not seed_uses_voxcpm_preset:
            raise HTTPException(status_code=400, detail="'seed' requires model='voxcpm'")
        if request.sample_url is not None and request.voice_id is not None:
            raise HTTPException(status_code=400, detail="Use either 'voice_id' or 'sample_url', not both")
        if request.reference_version is not None and request.sample_url is None:
            raise HTTPException(status_code=400, detail="'reference_version' requires 'sample_url'")
        if request.sample_url is not None and _seed_vc_style_requested(request):
            raise HTTPException(status_code=400, detail="'style' and 'styleIntensity' require 'voice_id'")
        if _is_voxcpm_model(model):
            if request.ssml is not None:
                raise HTTPException(status_code=400, detail="VoxCPM synthesis supports plain text only")
            if request.voice_id is not None:
                raise HTTPException(status_code=400, detail="VoxCPM model routing does not support voice_id")
            if _seed_vc_style_requested(request):
                raise HTTPException(status_code=400, detail="'style' and 'styleIntensity' require 'voice_id'")
            result = await synthesize_voxcpm_batch(
                _SharedBatchSynthesizeRequest(
                    texts=[request.text or ""],
                    seeds=[request.seed],
                    sample_url=request.sample_url,
                    reference_version=request.reference_version,
                    language=request.language,
                    model=model,
                    options=request.options,
                    format=request.format,
                    neural=request.neural,
                )
            )
            audio_bytes = base64.b64decode(result.items[0].audio_base64)
            media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
            return _binary_response(audio_bytes, media_type)

        if request.voice_id is not None:
            if model is not None:
                raise HTTPException(status_code=400, detail="Use either 'voice_id' or 'model', not both")
            response = await _synthesize_configured_voice(request)
            _maybe_cleanup_gpu()
            return response
        if _seed_vc_style_requested(request):
            raise HTTPException(status_code=400, detail="'style' and 'styleIntensity' require 'voice_id'")

        if not _engine_enabled("pipertts"):
            raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
        await _await_engine_ready("pipertts")
        synth_kwargs = _synth_kwargs_from_request(request)

        if request.ssml:
            audio, sample_rate = _synthesize_ssml(
                request.ssml,
                language=request.language,
                **synth_kwargs,
            )
        elif model is None or model == PUBLIC_SPARROW_MODEL:
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
            converted, _ = await _convert_generated_audio_to_sample_batch(
                source_audios=[audio],
                source_sample_rates=[sample_rate],
                sample_url=request.sample_url,
                reference_version=request.reference_version,
                output_format=request.format,
            )
            audio_bytes, _ = converted[0]
            media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
            _maybe_cleanup_gpu()
            return _binary_response(audio_bytes, media_type)

        # Convert to requested format
        if request.format == "mp3":
            audio_bytes = _audio_to_mp3_bytes(audio, sample_rate)
            media_type = "audio/mpeg"
        else:
            audio_bytes = _audio_to_wav_bytes(audio, sample_rate)
            media_type = "audio/wav"

        _maybe_cleanup_gpu()
        return _binary_response(audio_bytes, media_type)

    @app.post("/synthesize")
    async def synthesize(
        request: SynthesizeRequest,
        fastapi_request: Request,
        model_query: str = Query(None, alias="model", description="Legacy query-string model override"),
    ):
        """Synthesize text or SSML to speech.

        Provide either `text` (plain text) or `ssml` (SSML with <speak> wrapper), not both.

        By default, text is split by language and routed automatically.
        Use `language` to force one supported locale for the entire text.
        """
        started = time.perf_counter()
        model = _resolved_synthesize_model(request.model, model_query)
        _log_synthesize_request(
            route=fastapi_request.url.path,
            method=fastapi_request.method,
            status="received",
            started=started,
            count=1,
            text_chars=_text_length(request.text),
            ssml_chars=_text_length(request.ssml),
            input_data=_request_model_input(request, model=model),
        )
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
        _log_synthesize_request(
            route=fastapi_request.url.path,
            method=fastapi_request.method,
            status="received",
            started=started,
            count=len(request.items),
            text_chars=sum(_text_length(item.text) for item in request.items),
            ssml_chars=sum(_text_length(item.ssml) for item in request.items),
            input_data=_request_model_input(request),
        )
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
        model: str = Query(None, description="Public model family, e.g. sparrow or voxcpm"),
        voice_id: Optional[str] = Query(None, description="Configured preset voice id"),
        sample_url: Optional[str] = Query(
            None,
            description="Reference sample URL; used natively by VoxCPM or for Seed-VC conversion with Sparrow",
        ),
        reference_version: Optional[str] = Query(
            None,
            min_length=1,
            max_length=256,
            description="Opaque reference revision or content hash used for cache invalidation",
        ),
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
        resolved_model = _resolved_synthesize_model(None, model)
        get_input = {
            "text": text,
            "ssml": ssml,
            "model": model,
            "voice_id": voice_id,
            "sample_url": sample_url,
            "reference_version": reference_version,
            "language": language,
            "style": style,
            "styleIntensity": style_intensity,
            "options": options,
            "format": format,
            "neural": neural,
        }
        _log_synthesize_request(
            route=fastapi_request.url.path,
            method=fastapi_request.method,
            status="received",
            started=started,
            count=1,
            text_chars=_text_length(text),
            ssml_chars=_text_length(ssml),
            input_data=get_input,
        )
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
            reference_version=reference_version,
            language=language,
            style=style,
            style_intensity=style_intensity,
            options=parsed_options,
            format=format,
            neural=neural,
        )
        try:
            response = await _handle_synthesize(synth_request, model=resolved_model)
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


app = create_app()


def run():
    """Run the server with uvicorn."""
    import os

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    _LOGGER.info("Starting server at http://%s:%d", host, port)
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)
