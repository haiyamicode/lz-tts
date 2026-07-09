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
import signal
import sys
import threading
import time
import httpx
import uuid
from collections import OrderedDict
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import wait as mp_connection_wait
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Literal, Optional
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
from ..matcha_inference import MatchaBackend as ProductionStarlingBackend
from ..matcha_inference import MatchaBatchRequest
from ..matcha_inference import MatchaBatcher as ProductionStarlingBatcher
from . import qwen3
from .qwen_worker import qwen_worker_main
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
from .rvc import RVCBackend, RVCSettings
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
    languages: Optional[list[str]] = None


class EngineEnableConfig(BaseModel):
    """Global engine switches. Disabled engines are not mounted or loaded."""

    pipertts: bool = Field(default_factory=lambda: _env_bool("PIPER_TTS_ENABLED", True))
    qwen3: bool = Field(default_factory=lambda: _env_bool("QWEN_TTS_ENABLED", True))
    starling: bool = Field(default_factory=lambda: _env_bool("STARLING_TTS_ENABLED", _env_bool("MATCHA_TTS_ENABLED", False)))
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
    starling: MatchaConfig = Field(default_factory=MatchaConfig)
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
    rvc_model: Optional[str] = Field(None, alias="rvcModel", description="Optional RVC model filename to apply as the final conversion step")
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
    rvc_model: Optional[str] = Field(None, alias="rvcModel", description="Optional RVC model filename to apply as the final conversion step")
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
    rvc_model: str | None = None
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
_rvc_backend: "RVCBackend | None" = None
_seed_vc_supported_voice_ids: set[str] | None = None
_seed_vc_voice_ids: set[str] | None = None
_qwen_worker_processes: dict[str, mp.Process | None] = {"primary": None, "vietnamese": None}
_qwen_worker_requests: dict[str, Any | None] = {"primary": None, "vietnamese": None}
_qwen_worker_responses: dict[str, Any | None] = {"primary": None, "vietnamese": None}
_qwen_worker_locks: dict[str, threading.Lock] = {
    "primary": threading.Lock(),
    "vietnamese": threading.Lock(),
}
_qwen_worker_start_lock = threading.Lock()
_sparrow_worker: WorkerProcessClient | None = None
_sparrow_model_info: dict[str, dict[str, Any]] = {}
_starling_worker: WorkerProcessClient | None = None
_starling_info: dict[str, Any] = {}
_seed_vc_worker: WorkerProcessClient | None = None
_seed_vc_info: dict[str, Any] = {}
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
    for name in ("pipertts", "qwen3", "starling", "seed_vc", "rvc")
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
        config = ServerConfig()
        qwen3.apply_env_overrides(config.qwen)
        return config
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "starling" not in data and "matcha" in data:
        data["starling"] = data["matcha"]
    engines = data.get("engines")
    if isinstance(engines, dict) and "starling" not in engines and "matcha" in engines:
        engines["starling"] = engines["matcha"]
    config = ServerConfig(**data)
    qwen3.apply_env_overrides(config.qwen)
    return config


def _engine_enabled(engine: Literal["pipertts", "qwen3", "starling", "matcha", "seed_vc", "rvc"], config: ServerConfig | None = None) -> bool:
    cfg = config or _server_config
    if engine == "starling":
        return bool(cfg.engines.starling or cfg.engines.matcha)
    if engine == "matcha":
        return bool(cfg.engines.matcha or cfg.engines.starling)
    return bool(getattr(cfg.engines, engine))


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


def _separate_vietnamese_qwen_worker_enabled() -> bool:
    return bool(
        _server_config.qwen.vietnamese_model.strip()
        and _server_config.qwen.vietnamese_device.strip()
        and _server_config.qwen.vietnamese_device.strip() != _server_config.qwen.device.strip()
    )


def _qwen_worker_settings(worker_name: str) -> dict[str, Any]:
    settings = _server_config.qwen.model_dump(mode="json")
    settings["asr"] = {**settings.get("asr", {}), "enabled": False, "preload": False}
    if worker_name == "primary" and _separate_vietnamese_qwen_worker_enabled():
        settings["vietnamese_model"] = ""
        settings["viet_lora_model"] = ""
        settings["vietnamese_device"] = ""
        return settings
    if worker_name == "vietnamese":
        if not _separate_vietnamese_qwen_worker_enabled():
            raise RuntimeError("Separate Vietnamese Qwen worker is not configured")
        settings["model"] = _server_config.qwen.vietnamese_model
        settings["device"] = _server_config.qwen.vietnamese_device
        settings["dp_budget"] = {
            **settings.get("dp_budget", {}),
            "device": _server_config.qwen.vietnamese_device,
        }
        settings["disable_cuda_graph"] = _server_config.qwen.vietnamese_disable_cuda_graph
        settings["vietnamese_model"] = ""
        settings["viet_lora_model"] = ""
        settings["vietnamese_device"] = ""
        return settings
    return settings


def _qwen_worker_name_for_request(req: qwen3.SynthesizeRequest) -> str:
    if not _separate_vietnamese_qwen_worker_enabled():
        return "primary"
    resolved = qwen3.resolve_qwen_language(req.text, req.language, req.language_code)
    return "vietnamese" if qwen3._is_vietnamese_qwen_language(resolved.qwen_language) else "primary"


def _qwen_worker_groups_for_batch(
    req: qwen3.BatchSynthesizeRequest,
) -> OrderedDict[str, list[tuple[int, qwen3.SynthesizeRequest]]]:
    groups: OrderedDict[str, list[tuple[int, qwen3.SynthesizeRequest]]] = OrderedDict()
    for index, item in enumerate(req.items):
        worker_name = _qwen_worker_name_for_request(item)
        groups.setdefault(worker_name, []).append((index, item))
    return groups


def _call_qwen_batch_worker(
    worker_name: str,
    indexed_items: list[tuple[int, qwen3.SynthesizeRequest]],
) -> list[tuple[int, qwen3.BatchSynthesizeItem]]:
    req = qwen3.BatchSynthesizeRequest(items=[item for _, item in indexed_items])
    result = _call_qwen_worker(
        "synthesize_batch",
        req.model_dump(mode="json"),
        worker_name=worker_name,
    )
    data = result.get("data")
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail=f"Qwen3 {worker_name} worker returned invalid batch data")
    response = qwen3.BatchSynthesizeResponse(**data)
    if len(response.items) != len(indexed_items):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Qwen3 {worker_name} worker returned {len(response.items)} batch items "
                f"for {len(indexed_items)} requests"
            ),
        )
    return [
        (original_index, item)
        for (original_index, _), item in zip(indexed_items, response.items)
    ]


def _prepare_qwen_request_for_worker(req: qwen3.SynthesizeRequest) -> qwen3.SynthesizeRequest:
    if req.random_voice_embedding:
        return req
    settings = qwen3._resolve_generation_settings(req)
    _, prompt_text, xvec_only = qwen3._resolve_voice_prompt(req, settings)
    return req.model_copy(update={"voice_text": prompt_text, "xvec_only": xvec_only})


def _prepare_qwen_batch_for_worker(req: qwen3.BatchSynthesizeRequest) -> qwen3.BatchSynthesizeRequest:
    settings_list = qwen3._resolve_generation_settings_batch(req.items)
    items = []
    for item, settings in zip(req.items, settings_list):
        if item.random_voice_embedding:
            items.append(item)
            continue
        _, prompt_text, xvec_only = qwen3._resolve_voice_prompt(item, settings)
        items.append(item.model_copy(update={"voice_text": prompt_text, "xvec_only": xvec_only}))
    return req.model_copy(update={"items": items})


def _start_qwen_worker(worker_name: str = "primary") -> None:
    """Start the isolated Qwen worker process."""
    if worker_name == "vietnamese" and not _separate_vietnamese_qwen_worker_enabled():
        return
    with _qwen_worker_start_lock:
        process = _qwen_worker_processes.get(worker_name)
        if process is not None and process.is_alive():
            return

        ctx = mp.get_context("spawn")
        requests = ctx.Queue()
        responses = ctx.Queue()
        settings = _qwen_worker_settings(worker_name)
        process = ctx.Process(
            target=qwen_worker_main,
            args=(settings, requests, responses, worker_name),
            name=f"lz-tts-qwen-{worker_name}-worker",
            daemon=False,
        )
        _qwen_worker_requests[worker_name] = requests
        _qwen_worker_responses[worker_name] = responses
        _qwen_worker_processes[worker_name] = process
        process.start()


def _stop_qwen_worker(worker_name: str | None = None) -> None:
    """Stop the isolated Qwen worker process."""
    with _qwen_worker_start_lock:
        worker_names = [worker_name] if worker_name else list(_qwen_worker_processes.keys())
        for name in worker_names:
            process = _qwen_worker_processes.get(name)
            requests = _qwen_worker_requests.get(name)
            if process is not None and process.is_alive() and requests is not None:
                try:
                    requests.put({"action": "shutdown", "payload": None})
                    process.join(timeout=10)
                except Exception:
                    _LOGGER.exception("Failed graceful Qwen worker shutdown name=%s", name)
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=10)
            _qwen_worker_processes[name] = None
            _qwen_worker_requests[name] = None
            _qwen_worker_responses[name] = None


def _call_qwen_worker(
    action: str,
    payload: Any | None = None,
    *,
    wait_ready: bool = True,
    worker_name: str = "primary",
) -> dict[str, Any]:
    """Call the Qwen worker process and return its serialized response."""
    if not _engine_enabled("qwen3"):
        raise HTTPException(status_code=503, detail="Qwen3 TTS backend is disabled")
    if wait_ready:
        _wait_for_engine_ready("qwen3")
    _start_qwen_worker(worker_name)
    process = _qwen_worker_processes.get(worker_name)
    requests = _qwen_worker_requests.get(worker_name)
    responses = _qwen_worker_responses.get(worker_name)
    if (
        process is None
        or requests is None
        or responses is None
        or not process.is_alive()
    ):
        raise HTTPException(status_code=503, detail=f"Qwen3 {worker_name} worker is not running")

    request_id = uuid.uuid4().hex
    _qwen_worker_locks[worker_name].acquire()
    started = time.perf_counter()
    _LOGGER.info("Qwen worker request start name=%s action=%s request_id=%s", worker_name, action, request_id)
    try:
        requests.put({"request_id": request_id, "action": action, "payload": payload})
        response_reader = getattr(responses, "_reader", None)
        if response_reader is None:
            raise HTTPException(status_code=502, detail=f"Qwen3 {worker_name} worker response queue is not readable")
        while True:
            ready = mp_connection_wait([response_reader, process.sentinel])
            if response_reader not in ready:
                process.join()
                exit_detail = _process_exit_description(process.exitcode)
                with _qwen_worker_start_lock:
                    if _qwen_worker_processes.get(worker_name) is process:
                        _qwen_worker_processes[worker_name] = None
                        _qwen_worker_requests[worker_name] = None
                        _qwen_worker_responses[worker_name] = None
                _LOGGER.error(
                    "Qwen worker exited without response name=%s action=%s request_id=%s %s elapsed=%.2fs",
                    worker_name,
                    action,
                    request_id,
                    exit_detail,
                    time.perf_counter() - started,
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"Qwen3 {worker_name} worker exited without response ({exit_detail})",
                )
            response = responses.get()
            if not isinstance(response, dict):
                raise HTTPException(status_code=502, detail="Qwen3 worker returned invalid response")
            if response.get("request_id") == request_id:
                break
            _LOGGER.warning(
                "Discarding stale Qwen worker response name=%s action=%s expected_request_id=%s got_request_id=%s",
                worker_name,
                action,
                request_id,
                response.get("request_id"),
            )
    finally:
        _qwen_worker_locks[worker_name].release()

    _LOGGER.info(
        "Qwen worker request done name=%s action=%s request_id=%s elapsed=%.2fs",
        worker_name,
        action,
        request_id,
        time.perf_counter() - started,
    )
    if response.get("ok"):
        return response

    status_code = int(response.get("status_code") or 500)
    detail = response.get("detail") or response.get("error") or "Qwen3 worker failed"
    raise HTTPException(status_code=status_code, detail=detail)


def _qwen_worker_status() -> dict[str, Any]:
    """Best-effort Qwen status without loading Qwen in the parent process."""
    if not _engine_enabled("qwen3"):
        return {"enabled": False}
    state = _engine_status("qwen3")
    if state["status"] != "ready":
        return {"enabled": True, "worker": state["status"], **state}
    worker_names = ["primary"]
    if _separate_vietnamese_qwen_worker_enabled():
        worker_names.append("vietnamese")
    workers: dict[str, Any] = {}
    for worker_name in worker_names:
        try:
            response = _call_qwen_worker("health", wait_ready=False, worker_name=worker_name)
            data = response.get("data")
            workers[worker_name] = data if isinstance(data, dict) else {"worker": "invalid_response"}
        except HTTPException as exc:
            detail = str(exc.detail)
            workers[worker_name] = {
                "worker": "busy" if exc.status_code == 504 and "request slot" in detail else "error",
                "status_code": exc.status_code,
                "detail": detail,
            }
    primary = workers.get("primary")
    return {
        **state,
        "enabled": True,
        "worker": primary.get("worker") if isinstance(primary, dict) else "unknown",
        "separate_vietnamese_worker": _separate_vietnamese_qwen_worker_enabled(),
        "workers": workers,
    }



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
    if _seed_vc_info.get("embedding_keys"):
        embedding_keys = list(_seed_vc_info.get("embedding_keys") or [])
    else:
        backend = _get_seed_vc_backend()
        embedding_keys = list(backend.cached_embeddings.keys()) if backend.cached_embeddings else []
    emb_ids = {_seed_vc_base_id(key) for key in embedding_keys}
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
    await _await_engine_ready("seed_vc")
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


def _encoded_audio_duration_seconds(audio_bytes: bytes, output_format: Literal["wav", "mp3"], fallback: float) -> float:
    try:
        if output_format == "mp3":
            return float(AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3").duration_seconds)
        import soundfile as sf  # pylint: disable=import-outside-toplevel

        info = sf.info(io.BytesIO(audio_bytes))
        return float(info.frames) / float(info.samplerate) if info.samplerate else fallback
    except Exception:
        return fallback


async def _convert_encoded_audio_rvc_batch(
    *,
    audio_items: list[bytes],
    rvc_model: str | None,
    output_format: Literal["wav", "mp3"],
) -> list[tuple[bytes, int]]:
    if rvc_model is None:
        raise ValueError("rvc_model is required")
    if not audio_items:
        return []

    await _await_engine_ready("rvc")
    backend = _get_rvc_backend()
    started = time.perf_counter()
    _log_synthesize_batch_stage(
        "rvc_batch_start",
        model=rvc_model,
        count=len(audio_items),
        output_format=output_format,
    )
    try:
        converted = await asyncio.to_thread(
            backend.convert_batch,
            audio_items=audio_items,
            model=rvc_model,
            f0_method=backend.settings.default_f0_method,
            pitch=backend.settings.default_pitch,
            index_rate=backend.settings.default_index_rate,
            rms_mix_rate=backend.settings.default_rms_mix_rate,
            protect=backend.settings.default_protect,
            output_format=output_format,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _log_synthesize_batch_stage(
        "rvc_batch_done",
        model=rvc_model,
        count=len(converted),
        wall_seconds=round(time.perf_counter() - started, 6),
    )
    return converted


async def _apply_rvc_to_synthesize_response(
    response: BatchSynthesizeResponse,
    *,
    rvc_model: str | None,
    output_format: Literal["wav", "mp3"],
) -> BatchSynthesizeResponse:
    if rvc_model is None:
        return response

    audio_items = [base64.b64decode(item.audio_base64) for item in response.items]
    started = time.perf_counter()
    converted = await _convert_encoded_audio_rvc_batch(
        audio_items=audio_items,
        rvc_model=rvc_model,
        output_format=output_format,
    )
    if len(converted) != len(response.items):
        raise RuntimeError("internal RVC batch response count mismatch")

    items: list[BatchSynthesizeItem] = []
    audio_seconds_total = 0.0
    for original, (audio_bytes, sample_rate) in zip(response.items, converted):
        audio_seconds = _encoded_audio_duration_seconds(audio_bytes, output_format, original.audio_seconds)
        audio_seconds_total += audio_seconds
        items.append(
            BatchSynthesizeItem(
                text=original.text,
                audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
                sample_rate=sample_rate,
                audio_seconds=audio_seconds,
            )
        )

    wall_seconds = response.wall_seconds + (time.perf_counter() - started)
    return response.model_copy(
        update={
            "items": items,
            "wall_seconds": wall_seconds,
            "audio_seconds_total": audio_seconds_total,
            "rtf": (wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
            "model": f"{response.model}+rvc:{rvc_model}",
        }
    )


async def _apply_rvc_to_encoded_audio(
    *,
    audio_bytes: bytes,
    rvc_model: str | None,
    output_format: Literal["wav", "mp3"],
) -> tuple[bytes, int] | None:
    if rvc_model is None:
        return None
    converted = await _convert_encoded_audio_rvc_batch(
        audio_items=[audio_bytes],
        rvc_model=rvc_model,
        output_format=output_format,
    )
    if len(converted) != 1:
        raise RuntimeError("internal RVC response count mismatch")
    return converted[0]


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

    await _await_engine_ready("seed_vc")
    supported_voice_ids = _get_seed_vc_supported_voice_ids()
    if request.voice_id not in supported_voice_ids:
        supported = sorted(supported_voice_ids)
        raise HTTPException(status_code=400, detail=f"Unsupported voice_id {request.voice_id!r}; supported voices: {supported}")

    forced_language = _normalize_locale_with_region(request.language) if request.language is not None else None
    if forced_language is not None:
        _resolve_forced_language(forced_language)

    root_voice = _configured_root_voice_for_voice_id(request.voice_id)
    primary_speaker: str | None = None
    style, style_intensity = _seed_vc_style_from_request(request)
    style_requested = _seed_vc_style_requested(request)
    if root_voice is not None:
        primary_speaker = forced_language if forced_language is not None else root_voice.speaker
        convert_all = forced_language is not None and not _root_voice_can_synthesize_language(root_voice, forced_language)
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
        if root_voice is not None and any(not _root_voice_can_synthesize_language(root_voice, language) for language in languages):
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
        forced_language=forced_language,
        root_voice=bool(root_voice),
        root_voice_id=root_voice.voice_id if root_voice is not None else None,
        root_voice_model=root_voice.model if root_voice is not None else None,
        root_voice_speaker=root_voice.speaker if root_voice is not None else None,
        root_voice_languages=root_voice.languages if root_voice is not None else None,
        primary_speaker=primary_speaker,
        style_requested=style_requested,
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
    response = BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=f"voice_id:{request.voice_id}",
        speaker=primary_speaker,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )
    return await _apply_rvc_to_synthesize_response(
        response,
        rvc_model=request.rvc_model,
        output_format=request.format,
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
            rvc_model=request.rvc_model,
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


class _CachedEmbeddingKeysProxy:
    def __init__(self, keys: list[str]):
        self._keys = list(keys)

    def keys(self) -> list[str]:
        return list(self._keys)

    def __bool__(self) -> bool:
        return bool(self._keys)


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
        self.cached_embeddings = _CachedEmbeddingKeysProxy(list(info.get("embedding_keys") or []))

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
    _wait_for_engine_ready("rvc")
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
        response = BatchSynthesizeResponse(
            items=items,
            count=len(items),
            model=model_name,
            speaker=resolved_speaker,
            wall_seconds=total_wall_seconds,
            audio_seconds_total=audio_seconds_total,
            rtf=(total_wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
        )
        return await _apply_rvc_to_synthesize_response(
            response,
            rvc_model=request.rvc_model,
            output_format=request.format,
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

    response = BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model=model_name,
        speaker=resolved_speaker,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )
    return await _apply_rvc_to_synthesize_response(
        response,
        rvc_model=request.rvc_model,
        output_format=request.format,
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
            model="auto",
            speaker=None,
            wall_seconds=wall_seconds,
            audio_seconds_total=audio_seconds_total,
            rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
        )
        return await _apply_rvc_to_synthesize_response(
            response,
            rvc_model=request.rvc_model,
            output_format=request.format,
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
    response = BatchSynthesizeResponse(
        items=items,
        count=len(items),
        model="auto",
        speaker=None,
        wall_seconds=wall_seconds,
        audio_seconds_total=audio_seconds_total,
        rtf=(wall_seconds / audio_seconds_total) if audio_seconds_total else 0.0,
    )
    return await _apply_rvc_to_synthesize_response(
        response,
        rvc_model=request.rvc_model,
        output_format=request.format,
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
        item.rvc_model,
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
        rvc_model=first.rvc_model,
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
                "rvcModel": records[0][1].rvc_model,
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
            rvcModel=shared_request.rvc_model,
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
                    rvc_model=shared_request.rvc_model,
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
            rvcModel=shared_request.rvc_model,
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
    qwen3.configure(_server_config.qwen)

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
        _mount_qwen_demo(app)

        @app.get("/qwen3/health")
        async def qwen3_health():
            return await asyncio.to_thread(_qwen_worker_status)

        @app.post("/qwen3/synthesize")
        async def qwen3_synthesize(request: Request):
            payload = await request.json()
            req = qwen3.SynthesizeRequest(**payload)
            await _await_engine_ready("qwen3")
            req = await asyncio.to_thread(_prepare_qwen_request_for_worker, req)
            worker_name = _qwen_worker_name_for_request(req)
            result = await asyncio.to_thread(
                _call_qwen_worker,
                "synthesize",
                req.model_dump(mode="json"),
                worker_name=worker_name,
            )
            media_type = str(result.get("media_type") or "audio/mpeg")
            content = result.get("content") or b""
            if not isinstance(content, bytes):
                raise HTTPException(status_code=502, detail="Qwen3 worker returned invalid audio")
            return _binary_response(content, media_type)

        @app.post("/qwen3/synthesize/batch", response_model=qwen3.BatchSynthesizeResponse)
        @app.post("/qwen3/synthesize-batch", response_model=qwen3.BatchSynthesizeResponse)
        async def qwen3_synthesize_batch(request: Request):
            started = time.perf_counter()
            payload = await request.json()
            req = qwen3.BatchSynthesizeRequest(**payload)
            await _await_engine_ready("qwen3")
            req = await asyncio.to_thread(_prepare_qwen_batch_for_worker, req)
            worker_groups = _qwen_worker_groups_for_batch(req)
            results = await asyncio.gather(
                *(
                    asyncio.to_thread(_call_qwen_batch_worker, worker_name, indexed_items)
                    for worker_name, indexed_items in worker_groups.items()
                )
            )
            items: list[qwen3.BatchSynthesizeItem | None] = [None] * len(req.items)
            for worker_items in results:
                for index, item in worker_items:
                    items[index] = item
            final_items = [item for item in items if item is not None]
            if len(final_items) != len(req.items):
                raise HTTPException(status_code=502, detail="Qwen3 worker batch response was incomplete")
            return qwen3.BatchSynthesizeResponse(
                items=final_items,
                count=len(final_items),
                wall_seconds=time.perf_counter() - started,
                audio_seconds_total=sum(item.audio_seconds for item in final_items),
            )

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
        global _starling_backend, _starling_batcher, _seed_vc_backend, _rvc_backend
        global _seed_vc_supported_voice_ids, _seed_vc_voice_ids
        global _sparrow_model_info, _starling_info, _seed_vc_info
        global _startup_loader_task

        startup_started = time.perf_counter()
        _LOGGER.info("Scheduling server startup mode=early-online-parallel-workers config=%s", CONFIG_PATH)
        with _logged_startup_step("reset_runtime_state"):
            if _startup_loader_task is not None and not _startup_loader_task.done():
                _startup_loader_task.cancel()
            _stop_model_workers()
            _stop_qwen_worker()
            qwen3.stop_reference_transcription_worker()
            _inference_cache.clear()
            _lang_speaker_map.clear()
            _speaker_routes.clear()
            _splitter = None
            _splitter_languages = None
            _starling_backend = None
            _starling_batcher = None
            _seed_vc_backend = None
            _rvc_backend = None
            _seed_vc_supported_voice_ids = None
            _seed_vc_voice_ids = None
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

            if _engine_enabled("qwen3"):
                async def start_qwen() -> None:
                    with _logged_startup_step(
                        "qwen3_worker",
                        model=_server_config.qwen.model,
                        device=_server_config.qwen.device,
                        separate_vietnamese_worker=_separate_vietnamese_qwen_worker_enabled(),
                        vietnamese_model=_server_config.qwen.vietnamese_model,
                        vietnamese_device=_server_config.qwen.vietnamese_device,
                        dtype=_server_config.qwen.dtype,
                        dp_budget=_server_config.qwen.dp_budget.enabled,
                        asr_model=_server_config.qwen.asr.model,
                        asr_device=_server_config.qwen.asr.device,
                        asr_isolated=_server_config.qwen.asr.isolated,
                    ):
                        worker_names = ["primary"]
                        if _separate_vietnamese_qwen_worker_enabled():
                            worker_names.append("vietnamese")

                        async def start_tts_workers() -> None:
                            for worker_name in worker_names:
                                _start_qwen_worker(worker_name)
                            for worker_name in worker_names:
                                _LOGGER.info("Checking Qwen worker health name=%s", worker_name)
                                await asyncio.to_thread(
                                    _call_qwen_worker,
                                    "health",
                                    None,
                                    wait_ready=False,
                                    worker_name=worker_name,
                                )
                                _LOGGER.info("Qwen worker health ok name=%s", worker_name)

                        async def start_asr_worker() -> None:
                            if not (_server_config.qwen.asr.enabled and _server_config.qwen.asr.preload):
                                return
                            _LOGGER.info(
                                "Preloading Qwen ASR model model=%s device=%s isolated=%s",
                                _server_config.qwen.asr.model,
                                _server_config.qwen.asr.device,
                                _server_config.qwen.asr.isolated,
                            )
                            await asyncio.to_thread(qwen3.preload_reference_transcription_model)
                            _LOGGER.info("Qwen ASR preload ok")

                        await asyncio.gather(start_tts_workers(), start_asr_worker())

                startup_tasks.append(asyncio.create_task(run_loader("qwen3", start_qwen)))
            else:
                _LOGGER.info("Qwen3 TTS backend disabled")

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
                    global _seed_vc_info, _seed_vc_supported_voice_ids
                    with _logged_startup_step(
                        "seed_vc_worker",
                        device=_server_config.seed_vc.device,
                        root=_server_config.seed_vc.root,
                        embeddings=_server_config.seed_vc.embeddings_hdf5_path,
                    ):
                        worker = _ensure_seed_vc_worker()
                        worker.start()
                        response = await asyncio.to_thread(worker.call, "health")
                        _seed_vc_info = dict(response.get("data") or {})
                        _LOGGER.info("Loading Seed-VC voice catalog manifest=%s", SEED_VC_VOICE_IDS_PATH)
                        catalog_started = time.perf_counter()
                        _seed_vc_supported_voice_ids = _get_seed_vc_supported_voice_ids()
                        _LOGGER.info(
                            "Loaded Seed-VC voice catalog voices=%d elapsed=%.2fs",
                            len(_seed_vc_supported_voice_ids),
                            time.perf_counter() - catalog_started,
                        )

                startup_tasks.append(asyncio.create_task(run_loader("seed_vc", start_seed_vc)))
            else:
                _LOGGER.info("Seed-VC backend disabled")

            if _engine_enabled("rvc"):
                async def start_rvc() -> None:
                    global _rvc_backend
                    with _logged_startup_step(
                        "rvc",
                        cache_size=_server_config.rvc.cache_size,
                        preload_models=_server_config.rvc.preload_models,
                    ):
                        _rvc_backend = await asyncio.to_thread(_build_rvc_backend, _server_config.rvc)

                startup_tasks.append(asyncio.create_task(run_loader("rvc", start_rvc)))
            else:
                _LOGGER.info("RVC backend disabled")

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
        global _startup_loader_task
        if _startup_loader_task is not None and not _startup_loader_task.done():
            _startup_loader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _startup_loader_task
        _startup_loader_task = None
        _stop_model_workers()
        _stop_qwen_worker()
        qwen3.stop_reference_transcription_worker()

    @app.get("/")
    async def health():
        """Health check and server info."""
        qwen_status = await asyncio.to_thread(_qwen_worker_status) if _engine_enabled("qwen3") else {}

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
                "starling": _engine_enabled("starling"),
                "seed_vc": _engine_enabled("seed_vc"),
                "rvc": _engine_enabled("rvc"),
            },
            "pipertts": {
                "enabled": _engine_enabled("pipertts"),
                **_engine_status("pipertts"),
                "models_loaded": list(_sparrow_model_info.keys()) if _sparrow_worker is not None else list(_inference_cache.keys()),
                "models_enabled": _allowed_models(),
                "max_models_in_cache": _server_config.pipertts.max_models_in_cache,
                "default_model": _server_config.pipertts.default_model,
            },
            "qwen3": {
                "enabled": _engine_enabled("qwen3"),
                **qwen_status,
                "asr": qwen3.reference_transcription_status(),
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
            "rvc": {
                "enabled": _engine_enabled("rvc"),
                **_engine_status("rvc"),
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

    @app.get("/rvc/status")
    async def rvc_status():
        """RVC voice conversion backend status."""
        backend = _rvc_backend
        if backend is None:
            return {
                "enabled": _engine_enabled("rvc"),
                **_engine_status("rvc"),
                "loaded": False,
                "available_models": [],
            }
        return {**_engine_status("rvc"), **backend.status()}

    @app.get("/rvc/models", response_model=list[str])
    async def rvc_models():
        """List available RVC model weights."""
        await _await_engine_ready("rvc")
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
            await _await_engine_ready("rvc")
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
        return _binary_response(result_bytes, media_type)

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
            await _await_engine_ready("rvc")
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
        await _await_engine_ready("pipertts")
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
            rvc_result = await _apply_rvc_to_encoded_audio(
                audio_bytes=audio_bytes,
                rvc_model=request.rvc_model,
                output_format=request.format,
            )
            if rvc_result is not None:
                audio_bytes, sample_rate = rvc_result
            _maybe_cleanup_gpu()
            return _binary_response(audio_bytes, media_type)

        # Convert to requested format
        if request.format == "mp3":
            audio_bytes = _audio_to_mp3_bytes(audio, sample_rate)
            media_type = "audio/mpeg"
        else:
            audio_bytes = _audio_to_wav_bytes(audio, sample_rate)
            media_type = "audio/wav"

        rvc_result = await _apply_rvc_to_encoded_audio(
            audio_bytes=audio_bytes,
            rvc_model=request.rvc_model,
            output_format=request.format,
        )
        if rvc_result is not None:
            audio_bytes, sample_rate = rvc_result

        _maybe_cleanup_gpu()
        return _binary_response(audio_bytes, media_type)

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
            await _await_engine_ready("pipertts")
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
        rvc_model: Annotated[
            Optional[str],
            Query(alias="rvcModel", description="Optional RVC model filename to apply as the final conversion step"),
        ] = None,
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
            "rvcModel": rvc_model,
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
            rvc_model=rvc_model,
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
