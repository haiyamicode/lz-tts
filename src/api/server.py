"""FastAPI server for Sparrow/VITS TTS inference."""

from __future__ import annotations

import base64
import asyncio
import contextlib
import gc
import hashlib
import io
import json
import logging
import os
import secrets
import signal
import sys
import threading
import time
import httpx
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..multilingual_splitter import MultilingualSplitter, SplitResult
from ..piper import PiperInference
from ..ctc_forced_alignment import CtcAlignmentConfig, CtcForcedAligner, CtcLanguageSpan
from ..aligned_pauses import ResolvedPause, insert_resolved_pauses
from ..ssml import BreakOperation, PronunciationOperation, SSMLDocument, parse_ssml
from ..ssml_postprocessing import (
    insert_ssml_breaks,
    resolve_ssml_breaks,
)
from ..text_norm import normalize_spoken_text
from ..voxcpm_ipa_adapter import approximate_ipa_spelling, resolve_ipa_control_schedules
from ..matcha_inference import MatchaBackend as ProductionStarlingBackend
from ..matcha_inference import MatchaBatcher as ProductionStarlingBatcher
from .audio_utils import _audio_to_mp3_bytes, _audio_to_wav_bytes, _resample_audio
from .audio_adjustments import adjust_audio
from .locale_utils import normalize_locale as _normalize_locale_with_region
from .model_workers import seed_vc_worker_main, sparrow_worker_main, starling_worker_main
from .seed_vc_backend import (
    SeedVCBackend as _SeedVCBackend,
    SeedVCBatchRequest,
    SeedVCRequest,
)
from .voice_enhance import VoiceEnhanceRequest, VoiceEnhancer
from .voxcpm_runtime import VoxCPMRuntime
from .worker_common import ChildWorkerDied, WorkerProcessClient

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


# Process-wide readiness flag driven by the worker's Taskflow join lifecycle.
# Surfaced through the in-process HTTP server's /health endpoint so container
# orchestrators / load balancers can avoid routing traffic to a worker that
# hasn't joined the cluster yet.
#
#   starting -- process is up, models may still be loading, no cluster join yet.
#   ok       -- joined the Taskflow cluster and the inference runtime is healthy.
#   error    -- joined but currently unhealthy (lost the Taskflow session, the
#               inference runtime threw, etc.). Orchestrators should pull
#               traffic; the worker recovers to ``ok`` once it reconnects.
_HEALTH_STATUS: dict[str, str] = {"status": "starting", "reason": ""}
_VALID_HEALTH_STATUSES = {"starting", "ok", "error"}


def get_health_status() -> dict[str, str]:
    return {
        "status": _HEALTH_STATUS.get("status", "starting"),
        "reason": _HEALTH_STATUS.get("reason", ""),
    }


def set_status(status: str, reason: str = "") -> None:
    """Set the worker's /health status. Unknown values are coerced to ``error``."""
    if status not in _VALID_HEALTH_STATUSES:
        status = "error"
    _HEALTH_STATUS["status"] = status
    _HEALTH_STATUS["reason"] = reason


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _configured_api_key() -> str:
    return os.environ.get("API_KEY", "").strip()


def _request_api_key(request: Request) -> str:
    return (request.headers.get("X-Api-Key") or request.query_params.get("api_key") or "").strip()


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


class SparrowVoiceAdapterConfig(BaseModel):
    """An adapter artifact bound to one persistent Sparrow base model."""

    path: str
    model: str


class RootVoiceConfig(BaseModel):
    """A configured public voice id backed directly by a Sparrow model."""

    voice_id: str
    model: str
    speaker: Optional[str] = None
    languages: Optional[list[str]] = None
    adapter: Optional[str] = None


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
    text_preprocessor_device: str = "cuda:0"
    model_priority: list[str] = Field(default_factory=list)
    lang_speaker_map: dict[str, str] = Field(default_factory=dict)
    root_voices: dict[str, RootVoiceConfig] = Field(default_factory=dict)
    voice_adapter_cache_size: int = Field(1, ge=1)
    voice_adapters: dict[str, SparrowVoiceAdapterConfig] = Field(default_factory=dict)
    model_config_overrides: dict[str, ModelConfig] = Field(default_factory=dict, alias="model_config")

    @model_validator(mode="after")
    def validate_voice_adapters(self) -> "PiperTTSConfig":
        for root_name, root_voice in self.root_voices.items():
            if root_voice.adapter is None:
                continue
            adapter = self.voice_adapters.get(root_voice.adapter)
            if adapter is None:
                raise ValueError(
                    f"Root voice {root_name!r} uses unknown Sparrow adapter "
                    f"{root_voice.adapter!r}"
                )
            if adapter.model != root_voice.model:
                raise ValueError(
                    f"Root voice {root_name!r} model {root_voice.model!r} does not "
                    f"match adapter {root_voice.adapter!r} model {adapter.model!r}"
                )
        return self


class VoxCPMDurationBudgetConfig(BaseModel):
    """Sparrow deterministic-DP settings for VoxCPM generation limits."""

    enabled: bool = True
    preload: bool = True
    use_bert: bool = False
    checkpoint: str = "data/lzspeech-sparrow/model.ckpt"
    config_path: Optional[str] = None
    device: str = "auto"
    language: str = "multilingual"
    length_scale: float = Field(default=1.0, gt=0)
    token_rate: float = Field(default=6.25, gt=0)
    min_margin: float = Field(default=1.0, gt=0)
    max_margin: float = Field(default=1.35, gt=0)
    min_extra_tokens: int = Field(default=0, ge=0)
    max_extra_tokens: int = Field(default=38, ge=0)
    soft_text_token_limit: int = Field(default=64, ge=1)
    hard_text_token_limit: int = Field(default=80, ge=1)
    include_word_spans: bool = True
    language_profiles: dict[str, dict[str, float | int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_text_token_limits(self) -> "VoxCPMDurationBudgetConfig":
        if self.hard_text_token_limit < self.soft_text_token_limit:
            raise ValueError(
                "hard_text_token_limit must be greater than or equal to "
                "soft_text_token_limit"
            )
        return self


class VoxCPMConfig(BaseModel):
    """Optimized nano-vLLM VoxCPM2 serving configuration."""

    enabled: bool = Field(default_factory=lambda: _env_bool("VOXCPM_ENABLED", False))
    preload: bool = Field(default_factory=lambda: _env_bool("VOXCPM_PRELOAD", True))
    model_id: Literal["voxcpm"] = "voxcpm"
    model_path: str = Field(default_factory=lambda: os.environ.get("VOXCPM_MODEL_PATH", "data/voxcpm2-stable"))
    device: int = Field(default_factory=lambda: int(os.environ.get("VOXCPM_DEVICE", "1")), ge=0)
    dtype: Literal["auto", "bfloat16", "float16"] = Field(
        default_factory=lambda: os.environ.get("VOXCPM_DTYPE", "auto")
    )
    inference_timesteps: int = Field(default=10, ge=1)
    max_num_batched_tokens: int = Field(default=8192, ge=1)
    max_num_seqs: int = Field(default=8, ge=1)
    max_model_len: int = Field(default=4096, ge=1)
    gpu_memory_utilization: float = Field(default=0.62, gt=0, le=1)
    num_kvcache_blocks: int = Field(default=192, ge=1)
    enforce_eager: bool = False
    ipa_adapter_path: str | None = None
    fallback_max_generate_length: int = Field(default=4096, ge=1)
    duration_budget: VoxCPMDurationBudgetConfig = Field(default_factory=VoxCPMDurationBudgetConfig)
    temperature: float = Field(default=1.0, gt=0)
    cfg_value: float = Field(default=2.0, ge=0)
    reference_cache_size: int = Field(default=128, ge=1)
    max_reference_seconds: float = Field(default=25.0, gt=0)
    supported_languages: list[str] = Field(default_factory=lambda: [
        "ar", "da", "de", "el", "en", "es", "fi", "fil", "fr", "he", "hi",
        "id", "it", "ja", "km", "ko", "lo", "ms", "my", "nb", "nl", "pl",
        "pt", "ru", "sv", "sw", "th", "tr", "vi", "wuu", "yue", "zh",
    ])
    default_locales: list[str] = Field(default_factory=list)
    locale_loras: dict[str, str] = Field(default_factory=dict)
    applicable_loras: dict[str, str] = Field(default_factory=dict)
    max_concurrent_loras: int = Field(default=3, ge=1)
    max_loras_per_request: int = Field(default=2, ge=1)
    lora_composition_cache_path: str = "cache/voxcpm-lora-compositions"

    @model_validator(mode="after")
    def validate_locale_policy(self) -> "VoxCPMConfig":
        supported = {item.lower().split("-", 1)[0] for item in self.supported_languages}
        configured_locales = [*self.default_locales, *self.locale_loras]
        unsupported = sorted(
            locale
            for locale in configured_locales
            if locale.lower().split("-", 1)[0] not in supported
        )
        if unsupported:
            raise ValueError(f"VoxCPM locale policy contains unsupported locales: {unsupported}")
        unknown_loras = sorted(set(self.locale_loras.values()) - set(self.applicable_loras))
        if unknown_loras:
            raise ValueError(f"VoxCPM locale policy contains unknown LoRAs: {unknown_loras}")
        return self


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


class SSMLConfig(BaseModel):
    """Settings for SSML operations that require audio alignment."""

    enabled: bool = True
    ctc_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    ctc_device: str = Field(default_factory=lambda: os.environ.get("SSML_CTC_DEVICE", "cuda:0"))
    ctc_dtype: str = Field(default_factory=lambda: os.environ.get("SSML_CTC_DTYPE", "float16"))
    voxcpm_ipa_stop_cushion_patches: int = Field(default=1, ge=0)
    voxcpm_ipa_max_length_cushion_patches: int = Field(default=12, ge=1)
    voxcpm_ipa_alignment_tolerance_patches: int = Field(default=2, ge=0, le=10)
    voxcpm_ipa_refinement_passes: int = Field(default=1, ge=0, le=3)


class ServerConfig(BaseModel):
    """Server configuration."""

    engines: EngineEnableConfig = Field(default_factory=EngineEnableConfig)
    pipertts: PiperTTSConfig = Field(default_factory=PiperTTSConfig)
    voxcpm: VoxCPMConfig = Field(default_factory=VoxCPMConfig)
    starling: MatchaConfig = Field(default_factory=MatchaConfig)
    matcha: MatchaConfig = Field(default_factory=MatchaConfig)
    seed_vc: SeedVCConfig = Field(default_factory=SeedVCConfig)
    ssml: SSMLConfig = Field(default_factory=SSMLConfig)


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
    voice_id: Optional[str] = Field(None, description="Opaque product voice id")
    reference_url: Optional[str] = Field(
        None,
        description="Reference sample URL; used natively by VoxCPM or for Seed-VC conversion with Sparrow",
    )
    reference_language: Optional[str] = Field(
        None,
        description="Language/locale spoken by the reference sample",
    )
    language: Optional[str] = Field(
        None,
        description="Language/locale routing hint; forced for the entire input only when language_override is true",
    )
    language_override: bool = Field(
        False,
        description="Whether language was explicitly selected rather than detected",
    )
    model: Optional[str] = Field(None, description="Public model family, e.g. sparrow or voxcpm")
    seed: Optional[int] = Field(None, ge=0, description="Optional VoxCPM sampling seed")
    voxcpm_loras: list[str] = Field(
        default_factory=list,
        description="Configured VoxCPM LoRA names to apply",
    )
    options: Optional[SparrowSynthesizeOptions] = Field(None, description="Sparrow/VITS-specific synthesis options")
    format: Literal["wav", "mp3"] = Field("wav", description="Output audio format (wav or mp3)")
    neural: bool = Field(True, description="Use neural heteronym disambiguation for more accurate pronunciation of ambiguous words")
    speed: float = Field(1.0, ge=0.5, le=1.5, description="Playback speed ratio")
    pitch: float = Field(1.0, ge=0.5, le=1.5, description="Pitch ratio without changing duration")
    volume: float = Field(1.0, ge=0.0, le=1.0, description="Output volume multiplier")


class BatchSynthesizeInputItem(BaseModel):
    """One item in a /synthesize/batch request."""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    text: Optional[str] = Field(None, description="Plain text to synthesize")
    ssml: Optional[str] = Field(None, description="SSML input is not supported for batched synthesis")
    voice_id: Optional[str] = Field(None, description="Opaque product voice id")
    reference_url: Optional[str] = Field(
        None,
        description="Reference sample URL; used natively by VoxCPM or for Seed-VC conversion with Sparrow",
    )
    reference_language: Optional[str] = Field(
        None,
        description="Language/locale spoken by the reference sample",
    )
    language: Optional[str] = Field(
        None,
        description="Language/locale routing hint; forced for this item only when language_override is true",
    )
    language_override: bool = Field(
        False,
        description="Whether language was explicitly selected rather than detected",
    )
    model: Optional[str] = Field(None, description="Public model family, e.g. sparrow or voxcpm")
    seed: Optional[int] = Field(None, ge=0, description="Optional VoxCPM sampling seed")
    voxcpm_loras: list[str] = Field(
        default_factory=list,
        description="Configured VoxCPM LoRA names to apply",
    )
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
    reference_url: str | None = None
    reference_language: str | None = None
    language: str | None = None
    language_override: bool = False
    languages: list[str | None] | None = None
    model: str | None = None
    voxcpm_loras: tuple[str, ...] = ()
    options: SparrowSynthesizeOptions | None = None
    format: Literal["wav", "mp3"] = "wav"
    neural: bool = True


@dataclass(frozen=True)
class _BatchCompatibilityKey:
    """Collision-safe identity for one backend-compatible request configuration."""

    digest: str
    canonical_config: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _BatchCompatibilityKey:
        canonical_config = json.dumps(
            config,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
        return cls(digest=digest, canonical_config=canonical_config)


@dataclass(frozen=True)
class _SynthesisBatchPlan:
    """One ordered group that can share a backend inference call."""

    compatibility_key: _BatchCompatibilityKey
    pipeline: str
    records: list[tuple[int, BatchSynthesizeInputItem, str]]


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



class SynthesisCapabilitiesResponse(BaseModel):
    """Model capabilities needed by the task dispatcher."""

    locales: list[str]
    rootVoiceIds: list[str]


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
_voxcpm_reference_download_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
_voxcpm_reference_download_tasks: dict[str, asyncio.Task[tuple[bytes, str]]] = {}
_voxcpm_reference_download_lock = asyncio.Lock()
_startup_loader_task: asyncio.Task | None = None
_ssml_aligner: CtcForcedAligner | None = None


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


def _split_multilingual_text(text: str, language_hint: str | None = None) -> SplitResult:
    """Split text while using a routable locale only as the primary-language hint."""
    hinted_main_language = (
        _get_base_language(_normalize_locale_with_region(language_hint))
        if language_hint
        else None
    )
    if hinted_main_language not in _supported_sparrow_language_codes():
        hinted_main_language = None
    return _get_multilingual_splitter().split(text, main_lang=hinted_main_language)


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


def _worker_name_to_engine(name: str) -> str | None:
    if name == "sparrow":
        return "pipertts"
    if name in {"starling", "matcha"}:
        return "starling"
    if name == "seed-vc":
        return "seed_vc"
    return None


def _mark_engine_failed_from_child(worker_name: str, exc: BaseException) -> None:
    """Mark the engine backing a child worker as failed in ``_engine_load_states``.

    Subsequent ``_wait_for_engine_ready`` calls will return 503 instead of
    silently cold-loading the dead process again on every request.
    """
    engine = _worker_name_to_engine(worker_name)
    if engine is None or engine not in _engine_load_states:
        return
    state = _engine_state(engine)
    if state.status in {"loading", "ready", "error"}:
        _mark_engine_failed(engine, exc)


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


def _preloaded_piper_models() -> list[str]:
    """Configured Sparrow/VITS models to make resident at startup."""
    if not _engine_enabled("pipertts"):
        return []

    models: list[str] = []
    for model in _server_config.pipertts.preload_models:
        if not _is_model_allowed(model):
            raise ValueError(f"Preloaded model is not configured for this server: {model}")
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


def _make_cache_room() -> None:
    """Evict LRU models before loading another complete model stack."""
    limit = _server_config.pipertts.max_models_in_cache
    while len(_inference_cache) >= limit:
        evicted, inference = _inference_cache.popitem(last=False)
        _LOGGER.info("Evicted model from cache: %s", evicted)
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _configure_model_voice_adapters(model: str, inference: PiperInference) -> None:
    adapters = {
        name: _resolve_project_path(config.path)
        for name, config in _server_config.pipertts.voice_adapters.items()
        if config.model == model
    }
    if adapters:
        inference.configure_voice_adapters(
            adapters,
            cache_size=_server_config.pipertts.voice_adapter_cache_size,
        )


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

    _make_cache_room()
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
        _configure_model_voice_adapters(model, inference)
    except Exception:
        _LOGGER.exception("Failed loading Sparrow model model=%s elapsed=%.2fs", model, time.perf_counter() - started)
        raise
    _inference_cache[model] = inference
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
        voice_adapter: str | None = None,
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
                "voice_adapter": voice_adapter,
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
        voice_adapter: str | None = None,
        **synth_kwargs: Any,
    ) -> np.ndarray:
        response = _ensure_sparrow_worker().call(
            "synthesize_span",
            {
                "model": self.model,
                "text": text,
                "speaker": speaker,
                "neural": neural,
                "voice_adapter": voice_adapter,
                "synth_kwargs": synth_kwargs,
            },
        )
        data = response.get("data") or {}
        self.sample_rate = int(data.get("sample_rate") or self.sample_rate)
        return data.get("audio")

    def synthesize_with_ipa_overrides(
        self,
        text: str,
        overrides: Any,
        *,
        speaker: Any = None,
        neural: bool = True,
        voice_adapter: str | None = None,
        **synth_kwargs: Any,
    ) -> np.ndarray:
        response = _ensure_sparrow_worker().call(
            "synthesize_with_ipa_overrides",
            {
                "model": self.model,
                "text": text,
                "overrides": list(overrides),
                "speaker": speaker,
                "neural": neural,
                "voice_adapter": voice_adapter,
                "synth_kwargs": synth_kwargs,
            },
        )
        data = response.get("data") or {}
        self.sample_rate = int(data.get("sample_rate") or self.sample_rate)
        return data.get("audio")


def _get_inference(model: str) -> PiperInference:
    """Get a Sparrow model, loading it on demand when necessary."""
    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    _wait_for_engine_ready("pipertts")
    if _sparrow_worker is not None:
        if model in _sparrow_model_info:
            return _SparrowInferenceProxy(model, _sparrow_model_info[model])
        if _is_model_allowed(model):
            config_path = DATA_DIR / model / "config.json"
            if not config_path.exists():
                raise HTTPException(status_code=503, detail=f"Model config not found: {model}")
            with config_path.open("r", encoding="utf-8") as handle:
                model_config = json.load(handle)
            return _SparrowInferenceProxy(
                model,
                {
                    "sample_rate": (model_config.get("audio") or {}).get("sample_rate", 22050),
                    "speakers": model_config.get("speaker_id_map") or {},
                },
            )
        raise HTTPException(status_code=404, detail=f"Model is not configured for this server: {model}")
    if model in _inference_cache:
        inference = _inference_cache.pop(model)
        _inference_cache[model] = inference
        return inference
    if _is_model_allowed(model):
        return _load_model(model)
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

    device = _server_config.pipertts.text_preprocessor_device

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
        device,
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


def _validate_language_speaker_routes() -> None:
    """Fail startup when a configured locale points at no loaded Sparrow speaker."""
    unresolved = sorted(
        {
            speaker
            for speaker in _lang_speaker_map.values()
            if speaker not in _speaker_routes
        }
    )
    if unresolved:
        raise RuntimeError(
            "PiperTTS lang_speaker_map contains unresolved speakers: "
            + ", ".join(unresolved)
        )


def _synthesize_multilingual(
    text: str,
    primary_speaker: Optional[str] = None,
    language_hint: Optional[str] = None,
    forced_language: Optional[str] = None,
    noise_scale: Optional[float] = None,
    length_scale: Optional[float] = None,
    noise_w: Optional[float] = None,
    sdp_ratio: Optional[float] = None,
    neural: bool = True,
) -> tuple[np.ndarray, int]:
    """Synthesize multilingual text using multiple models.

    Args:
        text: Text to synthesize.
        primary_speaker: If set, use this speaker for segments matching its base language
                        (e.g., "en-GB" applies to "en" segments only).
        language_hint: Preferred locale for ambiguous same-script text.
        forced_language: If set, force the whole text to this locale.

    Returns (audio, sample_rate).
    """
    if forced_language is not None:
        _resolve_forced_language(forced_language)

    synth_kwargs = {}
    if noise_scale is not None:
        synth_kwargs["noise_scale"] = noise_scale
    if length_scale is not None:
        synth_kwargs["length_scale"] = length_scale
    if noise_w is not None:
        synth_kwargs["noise_w"] = noise_w
    if sdp_ratio is not None:
        synth_kwargs["sdp_ratio"] = sdp_ratio

    # First pass: compute routing plan
    routing_plan, _ = _plan_text_segments(
        text,
        primary_speaker,
        language_hint=language_hint,
        forced_language=forced_language,
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


def _configured_voice_language(voice_id: str | None, language: str | None) -> str | None:
    """Resolve the configured default locale of a locale-specific root voice."""
    if language is not None:
        return language
    root_voice = _configured_root_voice_for_voice_id(voice_id)
    if root_voice and root_voice.languages and len(root_voice.languages) == 1:
        return root_voice.languages[0]
    return None


def _explicit_language(language: str | None, language_override: bool) -> str | None:
    """Return the locale only when the caller explicitly requested it."""
    return language if language_override else None


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


def _build_synthesis_capabilities() -> SynthesisCapabilitiesResponse:
    locales = {
        locale
        for locale in _supported_sparrow_locales()
        if _is_supported_sparrow_locale(locale)
    }
    locales.update(_server_config.voxcpm.supported_languages)
    locales.update(_server_config.voxcpm.default_locales)
    locales.update(_server_config.voxcpm.locale_loras)
    return SynthesisCapabilitiesResponse(
        locales=sorted(locales),
        rootVoiceIds=sorted(
            config.voice_id for config in _server_config.pipertts.root_voices.values()
        ),
    )


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


def _seed_vc_reference_id(reference_url: str) -> str:
    identity = reference_url
    return f"synthesize-sample-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def _seed_vc_chunk_batch_size(backend: _SeedVCBackend) -> int:
    return max(1, int(backend.settings.max_chunk_batch_size))


async def _convert_generated_audio_to_sample_batch(
    *,
    source_audios: list[np.ndarray],
    source_sample_rates: list[int],
    reference_url: str,
    output_format: Literal["wav", "mp3"],
) -> tuple[list[tuple[bytes, float]], int]:
    await _await_engine_ready("seed_vc")
    backend = _get_seed_vc_backend()
    sample_request = SeedVCRequest(
        audio="",
        reference_url=reference_url,
        id=_seed_vc_reference_id(reference_url),
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
        reference_url=reference_url,
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
    language_hint: str | None = None,
    forced_language: str | None = None,
    *,
    validate_primary_speaker: bool = False,
) -> tuple[list[dict[str, Any]], set[str]]:
    if forced_language is not None:
        forced_locale, forced_speaker, forced_model = _resolve_forced_language(forced_language)
        source_start = len(text) - len(text.lstrip())
        source_end = len(text.rstrip())
        segment_text = text[source_start:source_end]
        segments: list[dict[str, Any]] = []
        if segment_text:
            segments.append({
                "lang": forced_locale,
                "speaker": forced_speaker,
                "model": forced_model,
                "text": segment_text,
                "source_start": source_start,
                "source_end": source_end,
            })
        return segments, {forced_locale}

    result = _split_multilingual_text(text, language_hint)
    main_lang = result.effective_main_language(language_hint or "en-us")
    primary_lang = _get_base_language(primary_speaker) if primary_speaker else None
    if primary_speaker is not None and validate_primary_speaker:
        _resolve_speaker_and_model(primary_speaker, explicit=True)

    segments = []
    languages: set[str] = set()

    for segment in result.segments:
        source_start = int(segment.start)
        source_end = int(segment.end)
        while source_start < source_end and text[source_start].isspace():
            source_start += 1
        while source_end > source_start and text[source_end - 1].isspace():
            source_end -= 1
        segment_text = text[source_start:source_end]
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
                "source_start": source_start,
                "source_end": source_end,
            }
        )

    return segments, languages or {main_lang}


async def synthesize_configured_voice_batch(request: _SharedBatchSynthesizeRequest) -> BatchSynthesizeResponse:
    if request.voice_id is None:
        raise HTTPException(status_code=400, detail="voice_id is required for root voice synthesis")

    root_voice = _configured_root_voice_for_voice_id(request.voice_id)
    if root_voice is None:
        raise HTTPException(status_code=400, detail=f"Voice {request.voice_id!r} is not a Sparrow root voice")

    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    if any(seed is not None for seed in request.seeds or []):
        raise HTTPException(status_code=400, detail="'seed' is only supported by VoxCPM")

    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")

    await _await_engine_ready("pipertts")
    forced_language = _resolve_sparrow_forced_language(
        _explicit_language(request.language, request.language_override)
    )

    primary_speaker: str | None = (
        forced_language
        if forced_language is not None
        else root_voice.speaker if root_voice is not None else None
    )
    started = time.perf_counter()
    synth_kwargs = _synth_kwargs_from_request(request)
    item_segments: list[list[dict[str, Any]]] = []
    segment_groups: OrderedDict[
        tuple[str, str | None], list[dict[str, Any]]
    ] = OrderedDict()

    for item_idx, text in enumerate(texts):
        segments, _ = _plan_text_segments(
            text,
            primary_speaker,
            language_hint=request.language,
            forced_language=forced_language,
            validate_primary_speaker=False,
        )
        for segment in segments:
            segment["voice_adapter"] = None
        if root_voice is not None:
            for segment in segments:
                if _root_voice_can_synthesize_language(root_voice, segment["lang"]):
                    if root_voice.speaker is not None:
                        segment["speaker"] = root_voice.speaker
                    segment["model"] = root_voice.model
                    segment["voice_adapter"] = root_voice.adapter
        item_segments.append(segments)
        for segment_idx, segment in enumerate(segments):
            record = {**segment, "item_idx": item_idx, "segment_idx": segment_idx}
            group_key = (segment["model"], segment["voice_adapter"])
            segment_groups.setdefault(group_key, []).append(record)

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
        item_count=len(texts),
        item_segment_counts=[len(segments) for segments in item_segments],
        convert_indices=[],
        model_groups=[
            {
                "model": model_name,
                "voice_adapter": voice_adapter,
                "segment_count": len(records),
                "item_indices": sorted({int(record["item_idx"]) for record in records}),
                "speakers": sorted({str(record["speaker"]) for record in records}),
                "languages": sorted({str(record["lang"]) for record in records}),
            }
            for (model_name, voice_adapter), records in segment_groups.items()
        ],
    )

    generated_segments: list[list[tuple[np.ndarray, int] | None]] = [
        [None for _ in segments]
        for segments in item_segments
    ]

    for (model_name, voice_adapter), records in segment_groups.items():
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
                voice_adapter=voice_adapter,
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
            voice_adapter=voice_adapter,
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
            voice_adapter=voice_adapter,
            **synth_kwargs,
        )
        audio_seconds = sum(float(len(audio)) / model_sample_rate for audio in batch_audios) if model_sample_rate else 0.0
        elapsed = time.perf_counter() - batch_started
        _log_synthesize_batch_stage(
            "sparrow_batch_done",
            pipeline="configured_voice",
            voice_id=request.voice_id,
            model=model_name,
            voice_adapter=voice_adapter,
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
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required for voice-id synthesis")
    batch_result = await synthesize_configured_voice_batch(
        _SharedBatchSynthesizeRequest(
            texts=[request.text],
            seeds=[request.seed],
            voice_id=request.voice_id,
            reference_language=request.reference_language,
            language=request.language,
            language_override=request.language_override,
            voxcpm_loras=tuple(request.voxcpm_loras),
            options=request.options,
            format=request.format,
            neural=request.neural,
        )
    )
    audio_bytes = base64.b64decode(batch_result.items[0].audio_base64)
    media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
    return _binary_response(audio_bytes, media_type)


def _decode_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    else:
        audio = audio[:, 0]
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _get_ssml_aligner() -> CtcForcedAligner:
    global _ssml_aligner
    settings = _server_config.ssml
    if not settings.enabled:
        raise HTTPException(status_code=503, detail="SSML audio operations are disabled")
    if _ssml_aligner is None:
        _ssml_aligner = CtcForcedAligner(
            CtcAlignmentConfig(
                model=settings.ctc_model,
                device=settings.ctc_device,
                dtype=settings.ctc_dtype,
            )
        )
    return _ssml_aligner


def _ssml_language_plan(
    text: str,
    language_hint: str | None,
    forced_language: str | None,
) -> tuple[str, list[CtcLanguageSpan]]:
    if forced_language:
        language = _normalize_locale_with_region(forced_language)
        return language, [CtcLanguageSpan(0, len(text), language)]

    result = _split_multilingual_text(text, language_hint)
    main_language = result.effective_main_language(language_hint or "en-us")
    spans: list[CtcLanguageSpan] = []
    cursor = 0

    def append_span(start: int, end: int, language: str) -> None:
        if end <= start:
            return
        if spans and spans[-1].source_end == start and spans[-1].language == language:
            previous = spans[-1]
            spans[-1] = CtcLanguageSpan(previous.source_start, end, language)
        else:
            spans.append(CtcLanguageSpan(start, end, language))

    for segment in sorted(result.segments, key=lambda item: item.start):
        if segment.start > cursor:
            append_span(cursor, segment.start, main_language)
        detected = segment.language if segment.language and segment.language != "und" else main_language
        language = _routable_detected_language(detected, main_language)
        append_span(segment.start, segment.end, language)
        cursor = max(cursor, segment.end)
    if cursor < len(text):
        append_span(cursor, len(text), main_language)
    if not spans:
        spans = [CtcLanguageSpan(0, len(text), main_language)]
    return main_language, spans


async def _align_ssml_audio(
    text: str,
    audio: np.ndarray,
    sample_rate: int,
    language_hint: str | None,
    forced_language: str | None,
) -> list[dict[str, Any]]:
    language, language_spans = _ssml_language_plan(
        text,
        language_hint,
        forced_language,
    )
    result = await asyncio.to_thread(
        _get_ssml_aligner().align_words,
        text,
        audio,
        sample_rate,
        language=language,
        language_spans=language_spans,
    )
    if not result.get("valid") or not result.get("word_timestamps"):
        raise ValueError(f"Could not force-align SSML audio: {result.get('reason') or 'unknown error'}")
    return list(result["word_timestamps"])


def _language_at_source_position(
    text: str,
    position: int,
    language_hint: str | None,
    forced_language: str | None,
) -> str:
    if forced_language:
        return _normalize_locale_with_region(forced_language)
    result = _split_multilingual_text(text, language_hint)
    main_language = result.effective_main_language(language_hint or "en-us")
    for segment in result.segments:
        if segment.start <= position < segment.end:
            detected = segment.language if segment.language and segment.language != "und" else main_language
            return _routable_detected_language(detected, main_language)
    return main_language


def _resolve_sparrow_forced_language(language: str | None) -> str | None:
    if language is None:
        return None

    forced_language = _normalize_locale_with_region(language)
    if not _is_supported_sparrow_locale(forced_language):
        base_language = _get_base_language(forced_language)
        if _is_supported_sparrow_locale(base_language):
            forced_language = base_language
    _resolve_forced_language(forced_language)
    return forced_language


def _ssml_sparrow_route(
    request: SynthesizeRequest,
    text: str,
    operation: PronunciationOperation,
    resolved_model: str | None,
) -> tuple[str | None, str]:
    forced_language = _explicit_language(
        request.language,
        request.language_override,
    )
    language = _language_at_source_position(
        text,
        operation.start,
        request.language,
        forced_language,
    )
    _locale, speaker, model_name = _resolve_forced_language(language)
    root_voice = _configured_root_voice_for_voice_id(request.voice_id)
    if root_voice is not None:
        if _root_voice_can_synthesize_language(root_voice, language):
            return root_voice.speaker or speaker, root_voice.model
    if resolved_model and resolved_model not in {
        PUBLIC_SPARROW_MODEL,
        _server_config.voxcpm.model_id,
    }:
        return None, resolved_model
    return speaker, model_name


def _voice_request_routes_to_voxcpm(
    *,
    voice_id: str | None,
    reference_url: str | None,
    language: str | None,
    model: str | None,
    reference_language: str | None = None,
) -> bool:
    if reference_url is None:
        return False

    # An explicit public model selects the backend. Capability-based routing is
    # only used when the client leaves model unset.
    if model is not None:
        return _is_voxcpm_model(model)

    effective_language = _configured_voice_language(voice_id, language)
    normalized_language = _normalize_locale_with_region(effective_language or "en")
    base_language = _get_base_language(normalized_language)
    if base_language not in {
        _get_base_language(item)
        for item in _server_config.voxcpm.supported_languages
    }:
        return False

    # A bare language selects VoxCPM's normal accent for that language. A full
    # locale must be one VoxCPM can actually honor: its configured default
    # locale, an adapter-backed locale, or the accent already in the reference.
    if "-" not in normalized_language:
        return True
    if normalized_language in {
        _normalize_locale_with_region(item)
        for item in _server_config.voxcpm.default_locales
    }:
        return True
    if normalized_language in {
        _normalize_locale_with_region(item)
        for item in _server_config.voxcpm.locale_loras
    }:
        return True
    return bool(
        reference_language
        and normalized_language == _normalize_locale_with_region(reference_language)
    )


def _request_routes_to_voxcpm(
    request: SynthesizeRequest,
    resolved_model: str | None,
) -> bool:
    return _voice_request_routes_to_voxcpm(
        voice_id=request.voice_id,
        reference_url=request.reference_url,
        language=request.language,
        model=resolved_model,
        reference_language=request.reference_language,
    )


def _request_routes_to_seed_vc(
    request: SynthesizeRequest,
    resolved_model: str | None,
) -> bool:
    if request.reference_url is None or _request_routes_to_voxcpm(request, resolved_model):
        return False

    return _voice_reference_requires_seed_vc(
        voice_id=request.voice_id,
        reference_url=request.reference_url,
        language=request.language,
    )


def _voice_reference_requires_seed_vc(
    *,
    voice_id: str | None,
    reference_url: str | None,
    language: str | None,
) -> bool:
    """Return whether Sparrow output must be converted to the requested voice."""
    if reference_url is None:
        return False

    root_voice = _configured_root_voice_for_voice_id(voice_id)
    if root_voice is None:
        return True

    effective_language = _configured_voice_language(voice_id, language)
    return not _root_voice_can_synthesize_language(root_voice, effective_language)


async def _synthesize_sparrow_ipa_ssml(
    request: SynthesizeRequest,
    document: SSMLDocument,
    resolved_model: str | None,
) -> tuple[np.ndarray, int]:
    """Synthesize Sparrow IPA overrides natively in one pass per language span."""
    await _await_engine_ready("pipertts")
    root_voice = _configured_root_voice_for_voice_id(request.voice_id)
    forced_language = _resolve_sparrow_forced_language(
        _explicit_language(request.language, request.language_override)
    )
    primary_speaker = (
        forced_language
        if forced_language is not None
        else root_voice.speaker if root_voice is not None else None
    )

    if (
        root_voice is None
        and resolved_model
        and resolved_model not in {PUBLIC_SPARROW_MODEL, _server_config.voxcpm.model_id}
    ):
        source_start = len(document.text) - len(document.text.lstrip())
        source_end = len(document.text.rstrip())
        segments = [{
            "lang": forced_language or "und",
            "speaker": None,
            "model": resolved_model,
            "text": document.text[source_start:source_end],
            "source_start": source_start,
            "source_end": source_end,
        }]
    else:
        segments, _ = _plan_text_segments(
            document.text,
            primary_speaker,
            language_hint=request.language,
            forced_language=forced_language,
            validate_primary_speaker=False,
        )

    for segment in segments:
        segment["voice_adapter"] = None

    if root_voice is not None:
        for segment in segments:
            if _root_voice_can_synthesize_language(root_voice, str(segment["lang"])):
                if root_voice.speaker is not None:
                    segment["speaker"] = root_voice.speaker
                segment["model"] = root_voice.model
                segment["voice_adapter"] = root_voice.adapter

    assigned_operations: set[int] = set()
    for segment in segments:
        source_start = int(segment["source_start"])
        source_end = int(segment["source_end"])
        local_overrides: list[tuple[int, int, str]] = []
        for index, operation in enumerate(document.pronunciations):
            if operation.end <= source_start or operation.start >= source_end:
                continue
            if operation.start < source_start or operation.end > source_end:
                raise ValueError("SSML <phoneme> span crosses a Sparrow language boundary")
            local_overrides.append((
                operation.start - source_start,
                operation.end - source_start,
                operation.phonemes,
            ))
            assigned_operations.add(index)
        segment["ipa_overrides"] = local_overrides

    if len(assigned_operations) != len(document.pronunciations):
        raise ValueError("Could not assign every SSML <phoneme> span to a Sparrow language segment")

    synth_kwargs = _synth_kwargs_from_request(request)
    generated: list[tuple[np.ndarray, int]] = []
    for segment in segments:
        model_name = str(segment["model"])
        if _is_starling_model(model_name):
            raise ValueError("SSML IPA pronunciation requires a Sparrow model")
        inference = _get_inference(model_name)
        internal_speaker = _resolve_internal_speaker(
            model_name,
            str(segment["speaker"]) if segment["speaker"] is not None else None,
            inference,
        )
        overrides = list(segment["ipa_overrides"])
        if overrides:
            audio = await asyncio.to_thread(
                inference.synthesize_with_ipa_overrides,
                str(segment["text"]),
                overrides,
                speaker=internal_speaker,
                neural=request.neural,
                voice_adapter=segment["voice_adapter"],
                **synth_kwargs,
            )
        else:
            audio = (
                await asyncio.to_thread(
                    inference.synthesize_batch,
                    [str(segment["text"])],
                    speaker=internal_speaker,
                    batch_size=1,
                    neural=request.neural,
                    voice_adapter=segment["voice_adapter"],
                    **synth_kwargs,
                )
            )[0]
        generated.append((audio, inference.sample_rate))

    if not generated:
        raise ValueError("SSML input produced no Sparrow language segments")
    sample_rate = generated[0][1]
    audio = np.concatenate([
        _resample_audio(item, item_rate, sample_rate)
        if item_rate != sample_rate
        else item
        for item, item_rate in generated
    ])

    if _request_routes_to_seed_vc(request, resolved_model):
        assert request.reference_url is not None
        converted, _converted_rate = await _convert_generated_audio_to_sample_batch(
            source_audios=[audio],
            source_sample_rates=[sample_rate],
            reference_url=request.reference_url,
            output_format="wav",
        )
        audio, sample_rate = _decode_wav_bytes(converted[0][0])
    return audio, sample_rate


def _ipa_marker(index: int) -> str:
    suffix = ""
    value = index
    while True:
        suffix = chr(ord("a") + value % 26) + suffix
        value = value // 26 - 1
        if value < 0:
            break
    return f"lzvoiceipaoverride{suffix}marker"


def _voxcpm_ipa_guide_text(
    document: SSMLDocument,
    operation: PronunciationOperation,
    language_hint: str | None,
    forced_language: str | None,
) -> str:
    """Return the visible LM guide; the adapter receives exact IPA separately."""
    language = _language_at_source_position(
        document.text,
        operation.start,
        language_hint,
        forced_language,
    )
    if _get_base_language(language) == "en":
        spelling = approximate_ipa_spelling(operation.phonemes, language)
        if spelling:
            return spelling
    # Non-English, or English IPA with no speakable spelling: the visible
    # text is already the best guide.
    guide = document.text[operation.start : operation.end].strip()
    if not guide:
        raise ValueError("SSML pronunciation spans must contain visible text")
    return guide


def _prepare_voxcpm_ipa_text(
    document: SSMLDocument,
    language_hint: str | None,
    forced_language: str | None,
    fallback_language: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    operations = sorted(document.pronunciations, key=lambda item: item.start)
    cursor = 0
    marked_parts: list[str] = []
    replacements: list[tuple[str, PronunciationOperation]] = []
    for index, operation in enumerate(operations):
        if operation.start < cursor:
            raise ValueError("Overlapping SSML pronunciation spans are not supported")
        marker = _ipa_marker(index)
        if marker in document.text.lower():
            raise ValueError("SSML text collides with an internal pronunciation marker")
        marked_parts.extend((document.text[cursor : operation.start], marker))
        replacements.append((marker, operation))
        cursor = operation.end
    marked_parts.append(document.text[cursor:])

    marked_text, detected_language = _prepare_voxcpm_input(
        "".join(marked_parts), forced_language, fallback_language
    )
    located_replacements: list[tuple[int, str, PronunciationOperation]] = []
    for marker, operation in replacements:
        if marked_text.count(marker) != 1:
            raise ValueError(
                "Spoken-text normalization did not preserve an SSML pronunciation span"
            )
        located_replacements.append((marked_text.index(marker), marker, operation))
    located_replacements.sort(key=lambda item: item[0])

    controlled_parts: list[str] = []
    controls: list[dict[str, Any]] = []
    marked_cursor = 0
    controlled_length = 0
    for marker_start, marker, operation in located_replacements:
        segment = marked_text[marked_cursor:marker_start]
        controlled_parts.append(segment)
        controlled_length += len(segment)
        spelling = _voxcpm_ipa_guide_text(
            document,
            operation,
            language_hint,
            forced_language,
        )
        controlled_parts.append(spelling)
        controls.append(
            {
                "source_start": operation.start,
                "source_end": operation.end,
                "controlled_start": controlled_length,
                "controlled_end": controlled_length + len(spelling),
                "target_ipa": operation.phonemes,
            }
        )
        controlled_length += len(spelling)
        marked_cursor = marker_start + len(marker)
    controlled_parts.append(marked_text[marked_cursor:])
    controlled_text = "".join(controlled_parts)
    return controlled_text, detected_language, controls


def _aligned_pronunciation_spans(
    operations: tuple[PronunciationOperation, ...],
    timestamps: list[dict[str, Any]],
) -> list[dict[str, float]]:
    spans: list[dict[str, float]] = []
    for operation in operations:
        matching = [
            item
            for item in timestamps
            if int(item.get("source_start", -1)) < operation.end
            and int(item.get("source_end", -1)) > operation.start
        ]
        if not matching:
            raise ValueError(
                "Forced alignment did not locate VoxCPM IPA source span "
                f"[{operation.start}, {operation.end})"
            )
        spans.append(
            {
                "start_seconds": min(float(item["start_seconds"]) for item in matching),
                "end_seconds": max(float(item["end_seconds"]) for item in matching),
            }
        )
    return spans


def _voxcpm_controlled_operations(
    controls: list[dict[str, Any]],
) -> tuple[PronunciationOperation, ...]:
    return tuple(
        PronunciationOperation(
            start=int(control["controlled_start"]),
            end=int(control["controlled_end"]),
            alphabet="ipa",
            phonemes=str(control["target_ipa"]),
        )
        for control in controls
    )


def _voxcpm_controlled_breaks(
    document: SSMLDocument,
    controls: list[dict[str, Any]],
) -> tuple[BreakOperation, ...]:
    mapped: list[BreakOperation] = []
    ordered = sorted(controls, key=lambda control: int(control["source_start"]))
    for operation in document.breaks:
        shift = 0
        for control in ordered:
            source_start = int(control["source_start"])
            source_end = int(control["source_end"])
            if source_start < operation.position < source_end:
                raise ValueError("SSML breaks inside IPA pronunciation spans are not supported")
            if source_end <= operation.position:
                shift += (
                    int(control["controlled_end"])
                    - int(control["controlled_start"])
                    - (source_end - source_start)
                )
        mapped.append(BreakOperation(operation.position + shift, operation.duration_seconds))
    return tuple(mapped)


def _voxcpm_ipa_pass_controls(
    controls: list[dict[str, Any]],
    schedules: list[dict[str, object]],
) -> list[dict[str, Any]]:
    """Keep all text controls fixed while enabling an observed audio prefix."""
    result = [{**control, "audio_enabled": False} for control in controls]
    for index, schedule in enumerate(schedules):
        result[index].update(schedule)
        result[index]["audio_enabled"] = True
    for index in range(len(schedules) - 1):
        start = int(result[index]["start_patch"])
        next_start = int(result[index + 1]["start_patch"])
        if next_start <= start:
            raise RuntimeError("Observed VoxCPM IPA control starts are not monotonic")
        result[index]["gates"] = list(result[index]["gates"][: next_start - start])
        if not result[index]["gates"]:
            raise RuntimeError("Observed VoxCPM IPA controls have no non-overlapping frames")
    return result


async def _predict_ssml_ipa_durations(
    request: SynthesizeRequest,
    document: SSMLDocument,
    resolved_model: str | None,
) -> list[float]:
    await _await_engine_ready("pipertts")
    routes = [
        _ssml_sparrow_route(request, document.text, operation, resolved_model)
        for operation in document.pronunciations
    ]
    if len(set(routes)) != 1:
        raise ValueError("VoxCPM IPA controls must use one Sparrow duration model")
    speaker, model_name = routes[0]
    inference = _get_inference(model_name)
    internal_speaker = _resolve_internal_speaker(model_name, speaker, inference)
    _audio, timestamps = await asyncio.to_thread(
        inference.synthesize_with_ipa_overrides,
        document.text,
        [
            (operation.start, operation.end, operation.phonemes)
            for operation in document.pronunciations
        ],
        speaker=internal_speaker,
        noise_scale=0.0,
        length_scale=1.0,
        noise_w=0.0,
        sdp_ratio=0.0,
        neural=False,
        return_alignment=True,
    )
    spans = _aligned_pronunciation_spans(document.pronunciations, list(timestamps))
    durations = [span["end_seconds"] - span["start_seconds"] for span in spans]
    if any(duration <= 0.0 for duration in durations):
        raise ValueError("Sparrow predicted a non-positive IPA pronunciation duration")
    return durations


async def _load_voxcpm_ssml_reference(
    request: SynthesizeRequest,
) -> tuple[bytes | None, str]:
    if request.reference_url is not None:
        return await _download_voxcpm_reference(request.reference_url)
    return None, "wav"


async def _synthesize_voxcpm_ipa_ssml(
    request: SynthesizeRequest,
    document: SSMLDocument,
    baseline_audio: np.ndarray,
    baseline_sample_rate: int,
    resolved_model: str | None,
) -> tuple[np.ndarray, int, list[ResolvedPause]]:
    await _await_engine_ready("voxcpm")
    runtime = _get_voxcpm_runtime()
    if baseline_sample_rate != runtime.sample_rate:
        raise ValueError(
            "VoxCPM SSML baseline sample rate does not match the configured runtime"
        )
    if request.seed is None:
        raise RuntimeError("VoxCPM IPA synthesis requires a resolved sampling seed")

    forced_language = _explicit_language(
        request.language,
        request.language_override,
    )
    controlled_text, _language, controls = _prepare_voxcpm_ipa_text(
        document,
        request.language,
        forced_language,
        _configured_voice_language(request.voice_id, request.language),
    )
    controlled_operations = _voxcpm_controlled_operations(controls)
    baseline_timestamps = await _align_ssml_audio(
        document.text,
        baseline_audio,
        baseline_sample_rate,
        request.language,
        forced_language,
    )
    baseline_spans = _aligned_pronunciation_spans(
        document.pronunciations, baseline_timestamps
    )
    target_durations = await _predict_ssml_ipa_durations(
        request, document, resolved_model
    )
    patch_samples = runtime.output_patch_samples
    baseline_patch_count, remainder = divmod(len(baseline_audio), patch_samples)
    if baseline_patch_count <= 0 or remainder:
        raise ValueError(
            "VoxCPM baseline audio does not contain an integral number of output patches"
        )
    initial_expected_patches = baseline_patch_count
    for span, target_duration in zip(baseline_spans, target_durations, strict=True):
        resolved, _expected = resolve_ipa_control_schedules(
            [span],
            [target_duration],
            baseline_patch_count=baseline_patch_count,
            patch_seconds=patch_samples / runtime.sample_rate,
            fade_out_ratio=runtime.ipa_fade_out_ratio,
        )
        initial_expected_patches += int(resolved[0]["duration_shift_patches"])
    initial_expected_patches = max(2, initial_expected_patches)
    stop_cushion = _server_config.ssml.voxcpm_ipa_stop_cushion_patches
    max_cushion = _server_config.ssml.voxcpm_ipa_max_length_cushion_patches
    requested_loras = _effective_voxcpm_lora_names(
        request.voxcpm_loras,
        request.language,
        request.language_override,
    )
    try:
        lora_name = await runtime.resolve_lora_combination(requested_loras)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not apply VoxCPM LoRAs: {exc}") from exc
    reference_audio, reference_format = await _load_voxcpm_ssml_reference(request)

    async def generate_pass(
        schedules: list[dict[str, object]],
        expected_patches: int,
    ) -> np.ndarray:
        pass_controls = _voxcpm_ipa_pass_controls(controls, schedules)
        target_end = max(
            (
                int(schedule["start_patch"])
                + int(schedule["target_patch_count"])
                for schedule in schedules
            ),
            default=0,
        )
        gate_end = max(
            (
                int(control["start_patch"]) + len(control["gates"])
                for control in pass_controls
                if control["audio_enabled"]
            ),
            default=0,
        )
        return await runtime.synthesize_controlled(
            controlled_text,
            ipa_controls=pass_controls,
            min_generate_length=max(
                2,
                expected_patches - stop_cushion,
                target_end,
            ),
            max_generate_length=max(
                expected_patches + max_cushion,
                gate_end,
            ),
            seed=request.seed,
            reference_audio=reference_audio,
            reference_format=reference_format,
            lora_name=lora_name,
        )

    async def align_controlled(
        audio: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
        timestamps = await _align_ssml_audio(
            controlled_text,
            audio,
            runtime.sample_rate,
            request.language,
            forced_language,
        )
        return timestamps, _aligned_pronunciation_spans(
            controlled_operations,
            timestamps,
        )

    # Establish one invariant text-conditioned timeline before enabling any
    # audio gate. Each subsequent pass adds exactly one future audio control.
    audio = await generate_pass([], initial_expected_patches)
    timestamps, observed_spans = await align_controlled(audio)
    schedules: list[dict[str, object]] = []
    generation_passes = 1

    async def build_schedule_suffix(start_index: int) -> None:
        nonlocal audio, timestamps, observed_spans, generation_passes
        del schedules[start_index:]
        for control_index in range(start_index, len(controls)):
            previous_patch_count, remainder = divmod(len(audio), patch_samples)
            if previous_patch_count <= 0 or remainder:
                raise RuntimeError(
                    "VoxCPM IPA pass did not contain an integral number of output patches"
                )
            resolved, expected_patches = resolve_ipa_control_schedules(
                [observed_spans[control_index]],
                [target_durations[control_index]],
                baseline_patch_count=previous_patch_count,
                patch_seconds=patch_samples / runtime.sample_rate,
                fade_out_ratio=runtime.ipa_fade_out_ratio,
            )
            schedules.append(resolved[0])
            audio = await generate_pass(schedules, expected_patches)
            generation_passes += 1
            timestamps, observed_spans = await align_controlled(audio)

    await build_schedule_suffix(0)

    tolerance = _server_config.ssml.voxcpm_ipa_alignment_tolerance_patches
    drifts: list[int] = []
    for refinement in range(_server_config.ssml.voxcpm_ipa_refinement_passes + 1):
        observed_starts = [
            max(0, int(float(span["start_seconds"]) * runtime.sample_rate // patch_samples))
            for span in observed_spans
        ]
        drifts = [
            observed_start - int(schedule["start_patch"])
            for observed_start, schedule in zip(observed_starts, schedules, strict=True)
        ]
        mismatches = [
            index for index, drift in enumerate(drifts) if abs(drift) > tolerance
        ]
        if not mismatches:
            break
        if refinement >= _server_config.ssml.voxcpm_ipa_refinement_passes:
            raise RuntimeError(
                "VoxCPM IPA control timeline did not converge after observed-alignment refinement: "
                f"drift_patches={drifts}"
            )
        await build_schedule_suffix(mismatches[0])

    _LOGGER.info(
        "VoxCPM IPA SSML generated controls=%d passes=%d baseline_patches=%d "
        "actual_patches=%d schedule_starts=%s observed_drift_patches=%s eager=true",
        len(controls),
        generation_passes,
        baseline_patch_count,
        len(audio) // patch_samples,
        [int(schedule["start_patch"]) for schedule in schedules],
        drifts,
    )
    controlled_breaks = (
        resolve_ssml_breaks(
            controlled_text,
            len(audio),
            runtime.sample_rate,
            _voxcpm_controlled_breaks(document, controls),
            timestamps,
        )
        if document.breaks
        else []
    )
    return audio, runtime.sample_rate, controlled_breaks


async def _postprocess_ssml_response(
    request: SynthesizeRequest,
    document: SSMLDocument,
    baseline_response: Response,
    resolved_model: str | None,
) -> Response:
    baseline_audio, sample_rate = _decode_wav_bytes(bytes(baseline_response.body))
    audio = baseline_audio
    resolved_breaks: list[ResolvedPause] | None = None

    if document.pronunciations:
        if not _request_routes_to_voxcpm(request, resolved_model):
            raise RuntimeError("Native Sparrow IPA must be synthesized before SSML postprocessing")
        audio, sample_rate, resolved_breaks = await _synthesize_voxcpm_ipa_ssml(
            request,
            document,
            baseline_audio,
            sample_rate,
            resolved_model,
        )

    if document.breaks:
        if resolved_breaks is not None:
            audio, _report = insert_resolved_pauses(
                audio,
                sample_rate,
                resolved_breaks,
            )
        else:
            timestamps = await _align_ssml_audio(
                document.text,
                audio,
                sample_rate,
                request.language,
                _explicit_language(request.language, request.language_override),
            )
            audio, _report = insert_ssml_breaks(
                document.text,
                audio,
                sample_rate,
                document.breaks,
                timestamps,
            )

    if request.format == "mp3":
        return _binary_response(_audio_to_mp3_bytes(audio, sample_rate), "audio/mpeg")
    return _binary_response(_audio_to_wav_bytes(audio, sample_rate), "audio/wav")



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
        reference_url=bool(request.reference_url),
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
    if request.reference_url is not None:
        converted, converted_sample_rate = await _convert_generated_audio_to_sample_batch(
            source_audios=audios,
            source_sample_rates=[sample_rate for _ in audios],
            reference_url=request.reference_url,
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

    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")

    started = time.perf_counter()
    synth_kwargs = _synth_kwargs_from_request(request)
    item_segments: list[list[dict[str, Any]]] = []
    segment_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    for item_idx, text in enumerate(texts):
        segments, _ = _plan_text_segments(
            text,
            primary_speaker=None,
            language_hint=request.language,
        )
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
            reference_url=bool(request.reference_url),
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

    if request.reference_url is not None:
        converted, converted_sample_rate = await _convert_generated_audio_to_sample_batch(
            source_audios=item_audios,
            source_sample_rates=item_sample_rates,
            reference_url=request.reference_url,
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


def _prepare_voxcpm_input(
    text: str,
    language: str | None,
    fallback_language: str | None = None,
) -> tuple[str, str]:
    if language is not None:
        return normalize_spoken_text(text, language), language

    result = _get_multilingual_splitter().split(text)
    main_language = result.effective_main_language(fallback_language or "en-us")
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


def _prepare_voxcpm_text(
    text: str,
    language: str | None,
    fallback_language: str | None = None,
) -> str:
    return _prepare_voxcpm_input(text, language, fallback_language)[0]


def _get_voxcpm_runtime() -> VoxCPMRuntime:
    if _voxcpm_runtime is None:
        raise HTTPException(status_code=503, detail="VoxCPM backend was not loaded at startup")
    return _voxcpm_runtime


def _resolve_voxcpm_lora_names(lora_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    names = list(lora_names)
    if not names:
        return ()
    if len(names) != len(set(names)):
        raise HTTPException(status_code=400, detail="voxcpm_loras must not contain duplicates")
    max_loras = _server_config.voxcpm.max_loras_per_request
    if len(names) > max_loras:
        raise HTTPException(
            status_code=400,
            detail=f"voxcpm_loras supports at most {max_loras} entries per request",
        )
    unknown = sorted(name for name in names if not name or name not in _server_config.voxcpm.applicable_loras)
    if unknown:
        available = sorted(_server_config.voxcpm.applicable_loras)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported VoxCPM LoRAs {unknown}; available: {available}",
        )
    return tuple(names)


def _configured_voxcpm_locale_lora(language: str | None) -> str | None:
    if language is None:
        return None
    locale = _normalize_locale_with_region(language)
    return next(
        (
            lora
            for configured_locale, lora in _server_config.voxcpm.locale_loras.items()
            if _normalize_locale_with_region(configured_locale) == locale
        ),
        None,
    )


def _effective_voxcpm_lora_names(
    lora_names: list[str] | tuple[str, ...],
    language: str | None,
    language_override: bool = False,
) -> tuple[str, ...]:
    names = list(lora_names)
    configured_lora = (
        _configured_voxcpm_locale_lora(language) if language_override else None
    )
    if configured_lora and configured_lora not in names:
        names.append(configured_lora)
    return _resolve_voxcpm_lora_names(names)


async def _fetch_voxcpm_reference(reference_url: str) -> tuple[bytes, str]:
    parsed = urlsplit(reference_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="reference_url must use http or https")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(reference_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch reference_url: {exc}") from exc

    audio = response.content
    if not audio:
        raise HTTPException(status_code=400, detail="reference_url returned an empty response")
    suffix = Path(parsed.path).suffix.lower().lstrip(".")
    return audio, suffix or "wav"


async def _download_voxcpm_reference(
    reference_url: str,
) -> tuple[bytes, str]:
    cache_key = reference_url
    async with _voxcpm_reference_download_lock:
        cached = _voxcpm_reference_download_cache.get(cache_key)
        if cached is not None:
            _voxcpm_reference_download_cache.move_to_end(cache_key)
            return cached
        task = _voxcpm_reference_download_tasks.get(cache_key)
        if task is None:
            task = asyncio.create_task(_fetch_voxcpm_reference(reference_url))
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
    if request.options is not None:
        raise HTTPException(status_code=400, detail="Sparrow options are not valid for VoxCPM")

    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="all texts must be non-empty")
    languages = request.languages or [
        _explicit_language(request.language, request.language_override)
    ] * len(texts)
    if len(languages) != len(texts):
        raise HTTPException(status_code=400, detail="VoxCPM languages length must match texts length")

    await _await_engine_ready("voxcpm")
    runtime = _get_voxcpm_runtime()
    requested_loras = _effective_voxcpm_lora_names(
        request.voxcpm_loras,
        request.language,
        request.language_override,
    )
    try:
        lora_name = await runtime.resolve_lora_combination(requested_loras)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not apply VoxCPM LoRAs: {exc}") from exc
    fallback_language = _configured_voice_language(request.voice_id, request.language)
    prepared_inputs = [
        _prepare_voxcpm_input(text, language, fallback_language)
        for text, language in zip(texts, languages)
    ]
    prepared_texts = [prepared_text for prepared_text, _ in prepared_inputs]
    dp_languages = [dp_language for _, dp_language in prepared_inputs]
    if reference_audio is not None and reference_audios is not None:
        raise HTTPException(status_code=400, detail="VoxCPM received both shared and per-item reference audio")
    if request.reference_url is not None and (reference_audio is not None or reference_audios is not None):
        raise HTTPException(status_code=400, detail="VoxCPM received both inline and URL reference audio")
    resolved_reference_format = reference_format or "wav"
    if request.reference_url is not None:
        reference_audio, resolved_reference_format = await _download_voxcpm_reference(
            request.reference_url,
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
        lora_names=[lora_name] * len(prepared_texts),
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
        loras=list(requested_loras),
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
    reference_keys = [item.reference_url for _, item, _ in records]
    unique_keys = list(dict.fromkeys(key for key in reference_keys if key is not None))
    loaded_references = await asyncio.gather(
        *(
            _download_voxcpm_reference(reference_url)
            for reference_url in unique_keys
            if reference_url is not None
        )
    )
    loaded_by_key = dict(zip(unique_keys, loaded_references))
    reference_audios = [
        loaded_by_key[key][0] if key is not None else None
        for key in reference_keys
    ]
    reference_formats = [
        loaded_by_key[key][1] if key is not None else "wav"
        for key in reference_keys
    ]
    first = records[0][1]
    routing_languages = [
        _configured_voice_language(item.voice_id, item.language)
        for _, item, _ in records
    ]
    forced_languages = [
        _explicit_language(language, item.language_override)
        for language, (_, item, _) in zip(routing_languages, records, strict=True)
    ]
    return await synthesize_voxcpm_batch(
        _SharedBatchSynthesizeRequest(
            texts=[text for _, _, text in records],
            seeds=[item.seed for _, item, _ in records],
            reference_language=first.reference_language,
            language=routing_languages[0],
            language_override=first.language_override,
            languages=forced_languages,
            model=_server_config.voxcpm.model_id,
            voxcpm_loras=tuple(first.voxcpm_loras),
            options=first.options,
            format=first.format,
            neural=first.neural,
        ),
        reference_audios=reference_audios,
        reference_formats=reference_formats,
    )


def _batch_item_pipeline(item: BatchSynthesizeInputItem) -> str:
    resolved_model = _resolve_api_model(item.model)
    if _voice_request_routes_to_voxcpm(
        voice_id=item.voice_id,
        reference_url=item.reference_url,
        language=item.language,
        model=resolved_model,
        reference_language=item.reference_language,
    ):
        return "voxcpm"
    if _voice_reference_requires_seed_vc(
        voice_id=item.voice_id,
        reference_url=item.reference_url,
        language=item.language,
    ):
        return "sparrow_reference"
    if _explicit_language(item.language, item.language_override) is not None:
        return "sparrow_forced_language"
    return "sparrow"


_BATCH_PER_ITEM_FIELDS = frozenset({"text", "ssml", "seed", "language_override"})
_BATCH_PIPELINE_PER_ITEM_FIELDS = {
    "voxcpm": frozenset({"reference_url", "reference_language", "language", "voice_id"}),
}


def _batch_item_compatibility_key(item: BatchSynthesizeInputItem) -> _BatchCompatibilityKey:
    """Hash every setting that must be shared by one backend batch call."""
    pipeline = _batch_item_pipeline(item)
    per_item_fields = _BATCH_PER_ITEM_FIELDS | _BATCH_PIPELINE_PER_ITEM_FIELDS.get(
        pipeline,
        frozenset(),
    )
    config = {
        key: value
        for key, value in item.model_dump(mode="json").items()
        if key not in per_item_fields
    }
    config["pipeline"] = pipeline

    # LoRA composition is additive, so request order does not affect inference.
    compatibility_loras = list(item.voxcpm_loras)
    if pipeline == "voxcpm":
        compatibility_loras = list(
            _effective_voxcpm_lora_names(
                compatibility_loras,
                item.language,
                item.language_override,
            )
        )
    config["voxcpm_loras"] = sorted(compatibility_loras)
    if config.get("language") is not None:
        config["language"] = _normalize_locale_with_region(config["language"])
    if pipeline == "voxcpm":
        config["model"] = _server_config.voxcpm.model_id
    elif pipeline == "sparrow":
        config["model"] = _resolve_api_model(item.model) or PUBLIC_SPARROW_MODEL
    if pipeline.startswith("sparrow"):
        effective_options = {
            key: value
            for key, value in (config.get("options") or {}).items()
            if value is not None
        }
        config["options"] = effective_options or None

    return _BatchCompatibilityKey.from_config(config)


def _validate_batch_item(item: BatchSynthesizeInputItem, item_idx: int) -> str:
    if item.text and item.ssml:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: Provide either 'text' or 'ssml', not both")
    if item.ssml is not None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: SSML is not supported in /synthesize/batch")
    if item.text is None or not item.text.strip():
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: text is required")
    if item.reference_url is not None and item.voice_id is None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'reference_url' requires 'voice_id'")
    if item.voice_id is not None and item.reference_url is None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: voices require 'reference_url'")
    resolved_model = _resolve_api_model(item.model)
    if _is_voxcpm_model(resolved_model) and item.reference_url is None:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: model='voxcpm' requires 'reference_url'")
    routes_to_voxcpm = _voice_request_routes_to_voxcpm(
        voice_id=item.voice_id,
        reference_url=item.reference_url,
        language=item.language,
        model=resolved_model,
        reference_language=item.reference_language,
    )
    if item.seed is not None and not routes_to_voxcpm:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'seed' requires model='voxcpm'")
    if item.voxcpm_loras and not routes_to_voxcpm:
        raise HTTPException(status_code=400, detail=f"items[{item_idx}]: 'voxcpm_loras' requires model='voxcpm'")
    if item.voxcpm_loras:
        try:
            _resolve_voxcpm_lora_names(item.voxcpm_loras)
        except HTTPException as exc:
            raise HTTPException(status_code=exc.status_code, detail=f"items[{item_idx}]: {exc.detail}") from exc
    return item.text.strip()


def _shared_batch_from_items(records: list[tuple[int, BatchSynthesizeInputItem, str]]) -> _SharedBatchSynthesizeRequest:
    first = records[0][1]
    pipeline = _batch_item_pipeline(first)
    root_voice = _configured_root_voice_for_voice_id(first.voice_id)
    native_root_voice = root_voice is not None and pipeline in {
        "sparrow",
        "sparrow_forced_language",
    }
    forced_languages = [
        _explicit_language(
            _configured_voice_language(item.voice_id, item.language),
            item.language_override,
        )
        for _, item, _ in records
    ]
    return _SharedBatchSynthesizeRequest(
        texts=[text for _, _, text in records],
        seeds=[item.seed for _, item, _ in records],
        voice_id=first.voice_id if native_root_voice else None,
        # Once routing selected native Sparrow, its registry reference is
        # metadata only. Passing it into the Sparrow backend means Seed-VC.
        reference_url=(
            first.reference_url
            if pipeline in {"voxcpm", "sparrow_reference"}
            else None
        ),
        reference_language=first.reference_language,
        language=_configured_voice_language(first.voice_id, first.language),
        language_override=first.language_override,
        languages=forced_languages,
        model=_resolve_api_model(first.model),
        voxcpm_loras=tuple(first.voxcpm_loras),
        options=first.options,
        format=first.format,
        neural=first.neural,
    )


_BATCH_PIPELINE_PRIORITY = {
    "sparrow": 0,
    "sparrow_forced_language": 0,
    "sparrow_reference": 1,
    "voxcpm": 2,
}


def _plan_synthesis_batches(
    items: list[BatchSynthesizeInputItem],
) -> list[_SynthesisBatchPlan]:
    """Resolve routes once, then partition requests by actual backend compatibility."""
    groups: OrderedDict[
        _BatchCompatibilityKey,
        tuple[str, list[tuple[int, BatchSynthesizeInputItem, str]]],
    ] = OrderedDict()
    for item_idx, item in enumerate(items):
        text = _validate_batch_item(item, item_idx)
        pipeline = _batch_item_pipeline(item)
        key = _batch_item_compatibility_key(item)
        if key not in groups:
            groups[key] = (pipeline, [])
        groups[key][1].append((item_idx, item, text))

    plans = [
        _SynthesisBatchPlan(
            compatibility_key=key,
            pipeline=pipeline,
            records=records,
        )
        for key, (pipeline, records) in groups.items()
    ]
    return sorted(
        plans,
        key=lambda plan: _BATCH_PIPELINE_PRIORITY.get(plan.pipeline, 99),
    )


async def _execute_synthesis_batch_plan(
    plan: _SynthesisBatchPlan,
) -> BatchSynthesizeResponse:
    records = plan.records
    shared_request = _shared_batch_from_items(records)
    if plan.pipeline == "voxcpm":
        return await _synthesize_voxcpm_items(records)
    if _configured_root_voice_for_voice_id(shared_request.voice_id) is not None:
        return await synthesize_configured_voice_batch(shared_request)
    forced_language = (shared_request.languages or [None])[0]
    if forced_language is not None:
        await _await_engine_ready("pipertts")
        _, forced_speaker, forced_model = _resolve_forced_language(forced_language)
        return await synthesize_sparrow_batch(
            _SharedBatchSynthesizeRequest(
                texts=shared_request.texts,
                reference_url=shared_request.reference_url,
                model=forced_model,
                options=shared_request.options,
                format=shared_request.format,
                neural=shared_request.neural,
            ),
            speaker=forced_speaker,
        )
    await _await_engine_ready("pipertts")
    if shared_request.model in {None, PUBLIC_SPARROW_MODEL}:
        return await synthesize_multilingual_sparrow_batch(shared_request)
    return await synthesize_sparrow_batch(shared_request)


async def synthesize_mixed_batch(request: BatchSynthesizeRequest) -> BatchSynthesizeResponse:
    """Group independent /synthesize-shaped inputs and run compatible real batches."""
    started = time.perf_counter()
    plans = _plan_synthesis_batches(request.items)

    _log_synthesize_batch_stage(
        "request_grouping",
        item_count=len(request.items),
        group_count=len(plans),
        groups=[
            {
                "group_index": group_idx,
                "config_hash": plan.compatibility_key.digest,
                "pipeline": plan.pipeline,
                "item_indices": [item_idx for item_idx, _, _ in plan.records],
                "count": len(plan.records),
                "voice_ids": [item.voice_id for _, item, _ in plan.records],
                "reference_url": bool(plan.records[0][1].reference_url),
                "languages": [item.language for _, item, _ in plan.records],
                "model": plan.records[0][1].model,
                "format": plan.records[0][1].format,
                "neural": plan.records[0][1].neural,
            }
            for group_idx, plan in enumerate(plans)
        ],
    )

    output_items: list[BatchSynthesizeItem | None] = [None for _ in request.items]
    group_results: list[BatchSynthesizeResponse] = []

    for group_idx, plan in enumerate(plans):
        compatibility_key = plan.compatibility_key
        records = plan.records
        shared_request = _shared_batch_from_items(records)
        group_started = time.perf_counter()
        _log_synthesize_batch_stage(
            "group_start",
            group_index=group_idx,
            config_hash=compatibility_key.digest,
            item_indices=[item_idx for item_idx, _, _ in records],
            item_count=len(records),
            voice_ids=[item.voice_id for _, item, _ in records],
            reference_url=bool(shared_request.reference_url),
            languages=[item.language for _, item, _ in records],
            model=shared_request.model,
            format=shared_request.format,
        )
        group_result = await _execute_synthesis_batch_plan(plan)

        _log_synthesize_batch_stage(
            "group_done",
            group_index=group_idx,
            config_hash=compatibility_key.digest,
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
            reference_url=bool(shared_request.reference_url),
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


@dataclass(frozen=True)
class InferenceResult:
    """Transport-independent result returned by the preloaded inference session."""

    kind: Literal["audio", "json"]
    content_type: str | None = None
    audio: bytes | None = None
    data: dict[str, Any] | None = None


async def _apply_audio_adjustments(
    result: InferenceResult,
    *,
    speed: float,
    pitch: float,
    volume: float,
) -> InferenceResult:
    if result.kind != "audio" or (speed == 1.0 and pitch == 1.0 and volume == 1.0):
        return result
    if not result.audio:
        raise RuntimeError("cannot adjust an empty inference result")

    audio_format: Literal["mp3", "wav"] = (
        "mp3" if result.content_type == "audio/mpeg" else "wav"
    )
    audio = await asyncio.to_thread(
        adjust_audio,
        result.audio,
        input_format=audio_format,
        output_format=audio_format,
        speed=speed,
        pitch=pitch,
        volume=volume,
    )
    return InferenceResult(
        kind="audio",
        content_type=result.content_type,
        audio=audio,
    )


class InferenceOperationError(RuntimeError):
    """An inference failure with retry semantics independent of HTTP."""

    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(str(detail))


async def _start_inference_runtime(config: ServerConfig) -> None:
    """Initialize the process-wide model runtime without constructing an API app."""
    global _server_config, _speaker_routes, _ssml_aligner
    global _lang_speaker_map, _splitter, _splitter_languages
    global _starling_backend, _starling_batcher, _seed_vc_backend
    global _voxcpm_runtime
    global _sparrow_model_info, _starling_info, _seed_vc_info
    global _startup_loader_task

    _server_config = config
    _ssml_aligner = None
    startup_started = time.perf_counter()
    _LOGGER.info("Scheduling inference runtime startup config=%s", CONFIG_PATH)
    with _logged_startup_step("reset_runtime_state"):
        if _startup_loader_task is not None and not _startup_loader_task.done():
            _startup_loader_task.cancel()
        if _voxcpm_runtime is not None:
            await _voxcpm_runtime.stop()
            _voxcpm_runtime = None
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
            allowed_models = _allowed_models()
            preload_models = _preloaded_piper_models()
            if not allowed_models:
                _mark_engine_failed(
                    "pipertts",
                    RuntimeError("PiperTTS is enabled but no Sparrow models are configured or available"),
                )
            else:
                async def start_sparrow() -> None:
                    global _sparrow_model_info
                    with _logged_startup_step("sparrow_worker", preload_models=preload_models):
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

            startup_tasks.append(asyncio.create_task(run_loader("seed_vc", start_seed_vc)))
        else:
            _LOGGER.info("Seed-VC backend disabled")

        if startup_tasks:
            await asyncio.gather(*startup_tasks)

        failed_engines = {
            name: state.error
            for name, state in _engine_load_states.items()
            if state.status == "error"
        }
        if failed_engines:
            details = "; ".join(
                f"{name}: {error}"
                for name, error in failed_engines.items()
            )
            raise RuntimeError(f"Configured inference engines failed to load: {details}")

        if _server_config.ssml.enabled:
            with _logged_startup_step(
                "ssml_ctc_aligner",
                model=_server_config.ssml.ctc_model,
                device=_server_config.ssml.ctc_device,
                dtype=_server_config.ssml.ctc_dtype,
            ):
                await asyncio.to_thread(_get_ssml_aligner().load)
        _LOGGER.info("Loaded inference runtime models elapsed=%.2fs", time.perf_counter() - load_started)

    if _engine_enabled("pipertts"):
        for locale, speaker in _server_config.pipertts.lang_speaker_map.items():
            _lang_speaker_map[_normalize_locale_with_region(locale)] = speaker
        route_models = _server_config.pipertts.model_priority or _allowed_models()
        if route_models:
            _LOGGER.info("Loading PiperTTS speaker routes models=%s", route_models)
            _speaker_routes = _build_speaker_routes(route_models)
            _validate_language_speaker_routes()
            _LOGGER.info(
                "Loaded PiperTTS speaker routes speakers=%d locales=%d",
                len(_speaker_routes),
                len(_lang_speaker_map),
            )
    else:
        _LOGGER.info("PiperTTS backend disabled")

    _startup_loader_task = asyncio.create_task(load_models_background())
    await _startup_loader_task
    _LOGGER.info("Inference runtime startup complete elapsed=%.2fs", time.perf_counter() - startup_started)


async def _stop_inference_runtime() -> None:
    global _startup_loader_task, _voxcpm_runtime
    if _startup_loader_task is not None and not _startup_loader_task.done():
        _startup_loader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _startup_loader_task
    _startup_loader_task = None
    if _voxcpm_runtime is not None:
        await _voxcpm_runtime.stop()
        _voxcpm_runtime = None
    _stop_model_workers()


async def _synthesize(request: SynthesizeRequest, model: str | None = None) -> Response:
    """Run one synthesis directly against the preloaded runtime."""
    if request.text and request.ssml:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'ssml', not both")
    if not request.text and not request.ssml:
        raise HTTPException(status_code=400, detail="Must provide either 'text' or 'ssml'")
    effective_language = _configured_voice_language(request.voice_id, request.language)
    if effective_language != request.language:
        request = request.model_copy(update={"language": effective_language})
    forced_language = _explicit_language(
        request.language,
        request.language_override,
    )
    if request.ssml is not None:
        try:
            document = parse_ssml(request.ssml)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if document.pronunciations and not _request_routes_to_voxcpm(request, model):
            try:
                audio, sample_rate = await _synthesize_sparrow_ipa_ssml(
                    request,
                    document,
                    model,
                )
                if document.breaks:
                    timestamps = await _align_ssml_audio(
                        document.text,
                        audio,
                        sample_rate,
                        request.language,
                        forced_language,
                    )
                    audio, _report = insert_ssml_breaks(
                        document.text,
                        audio,
                        sample_rate,
                        document.breaks,
                        timestamps,
                    )
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if request.format == "mp3":
                return _binary_response(_audio_to_mp3_bytes(audio, sample_rate), "audio/mpeg")
            return _binary_response(_audio_to_wav_bytes(audio, sample_rate), "audio/wav")
        has_operations = bool(document.operations)
        effective_request = request
        if (
            document.pronunciations
            and _request_routes_to_voxcpm(request, model)
            and request.seed is None
        ):
            effective_request = request.model_copy(update={"seed": secrets.randbits(63)})
        plain_request = effective_request.model_copy(
            update={
                "text": document.text,
                "ssml": None,
                "format": "wav" if has_operations else request.format,
            }
        )
        baseline_response = await _synthesize(plain_request, model=model)
        if not has_operations:
            return baseline_response
        try:
            return await _postprocess_ssml_response(
                effective_request,
                document,
                baseline_response,
                model,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.reference_url is not None and request.voice_id is None:
        raise HTTPException(status_code=400, detail="'reference_url' requires 'voice_id'")
    if request.voice_id is not None and request.reference_url is None:
        raise HTTPException(status_code=400, detail="Voices require 'reference_url'")
    if _is_voxcpm_model(model) and request.reference_url is None:
        raise HTTPException(status_code=400, detail="model='voxcpm' requires 'reference_url'")

    routes_to_voxcpm = _request_routes_to_voxcpm(request, model)
    _LOGGER.info(
        "Synthesis route voice=%s language=%s requested_model=%s backend=%s "
        "reference_language=%s text_chars=%d",
        request.voice_id,
        request.language,
        model,
        "voxcpm" if routes_to_voxcpm else "sparrow",
        request.reference_language,
        len(request.text or ""),
    )
    if request.seed is not None and not routes_to_voxcpm:
        raise HTTPException(status_code=400, detail="'seed' requires model='voxcpm'")
    if request.voxcpm_loras and not routes_to_voxcpm:
        raise HTTPException(status_code=400, detail="'voxcpm_loras' requires model='voxcpm'")
    if request.voxcpm_loras:
        _resolve_voxcpm_lora_names(request.voxcpm_loras)

    if routes_to_voxcpm:
        result = await synthesize_voxcpm_batch(
            _SharedBatchSynthesizeRequest(
                texts=[request.text or ""],
                seeds=[request.seed],
                voice_id=request.voice_id,
                reference_url=request.reference_url,
                reference_language=request.reference_language,
                language=request.language,
                language_override=request.language_override,
                languages=[forced_language],
                model=_server_config.voxcpm.model_id,
                voxcpm_loras=tuple(request.voxcpm_loras),
                options=request.options,
                format=request.format,
                neural=request.neural,
            )
        )
        audio_bytes = base64.b64decode(result.items[0].audio_base64)
        media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
        return _binary_response(audio_bytes, media_type)

    root_voice = _configured_root_voice_for_voice_id(request.voice_id)
    if (
        root_voice is not None
        and model in {None, PUBLIC_SPARROW_MODEL}
        and not _request_routes_to_seed_vc(request, model)
    ):
        return await _synthesize_configured_voice(request)

    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    await _await_engine_ready("pipertts")
    synth_kwargs = _synth_kwargs_from_request(request)

    if model is None or model == PUBLIC_SPARROW_MODEL:
        audio, sample_rate = _synthesize_multilingual(
            request.text,
            language_hint=request.language,
            forced_language=forced_language,
            neural=request.neural,
            **synth_kwargs,
        )
    else:
        inference = _get_inference(model)
        batch_audios = await asyncio.to_thread(
            inference.synthesize_batch,
            [request.text],
            speaker=None,
            batch_size=1,
            neural=request.neural,
            **synth_kwargs,
        )
        audio = batch_audios[0]
        sample_rate = inference.sample_rate

    if _request_routes_to_seed_vc(request, model):
        assert request.reference_url is not None
        converted, _ = await _convert_generated_audio_to_sample_batch(
            source_audios=[audio],
            source_sample_rates=[sample_rate],
            reference_url=request.reference_url,
            output_format=request.format,
        )
        audio_bytes, _ = converted[0]
        media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
        _maybe_cleanup_gpu()
        return _binary_response(audio_bytes, media_type)

    if request.format == "mp3":
        audio_bytes = _audio_to_mp3_bytes(audio, sample_rate)
        media_type = "audio/mpeg"
    else:
        audio_bytes = _audio_to_wav_bytes(audio, sample_rate)
        media_type = "audio/wav"
    _maybe_cleanup_gpu()
    return _binary_response(audio_bytes, media_type)


def _prepare_batchable_synthesis_request(
    request_data: dict[str, Any],
) -> SynthesizeRequest | None:
    """Return plain synthesis input, or None when SSML needs postprocessing."""
    request = SynthesizeRequest.model_validate(request_data)
    if request.ssml is None:
        return request
    if request.text is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'text' or 'ssml', not both",
        )
    try:
        document = parse_ssml(request.ssml)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if document.operations:
        return None
    return request.model_copy(update={"text": document.text, "ssml": None})


class LzTtsInferenceSession:
    """One preloaded runtime with direct task-operation dispatch."""

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or _load_config()
        self.voice_enhancer = VoiceEnhancer(
            os.environ.get("VOICE_ENHANCE_TMP_DIR", "data/voice-enhance/tmp")
        )
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await _start_inference_runtime(self.config)
        self._started = True

    async def close(self) -> None:
        if not self._started:
            return
        await _stop_inference_runtime()
        self._started = False

    def synthesis_capabilities(self) -> dict[str, Any]:
        return _build_synthesis_capabilities().model_dump(mode="json")

    async def execute_many(
        self,
        operations: list[tuple[str, dict[str, Any]]],
    ) -> list[InferenceResult | BaseException]:
        """Plan compatible synthesis operations into real backend batches."""
        if not self._started:
            raise RuntimeError("Inference session has not been started")

        outcomes: list[InferenceResult | BaseException | None] = [None] * len(operations)
        batch_items: list[BatchSynthesizeInputItem] = []
        batch_requests: list[SynthesizeRequest] = []
        batch_operation_indices: list[int] = []
        singleton_indices: list[int] = []

        for operation_index, (operation, request_data) in enumerate(operations):
            if operation != "synthesize":
                singleton_indices.append(operation_index)
                continue
            try:
                request = _prepare_batchable_synthesis_request(request_data)
                if request is None:
                    singleton_indices.append(operation_index)
                    continue
                item = BatchSynthesizeInputItem.model_validate(
                    request.model_dump(
                        mode="python",
                        exclude={"speed", "pitch", "volume"},
                    )
                )
                _validate_batch_item(item, operation_index)
            except ValidationError as error:
                outcomes[operation_index] = InferenceOperationError(
                    400, jsonable_encoder(error.errors())
                )
            except HTTPException as error:
                outcomes[operation_index] = InferenceOperationError(
                    error.status_code, error.detail
                )
            else:
                batch_items.append(item)
                batch_requests.append(request)
                batch_operation_indices.append(operation_index)

        plans = _plan_synthesis_batches(batch_items) if batch_items else []
        _LOGGER.info(
            "Inference task batch planned: %s",
            json.dumps(
                {
                    "operation_count": len(operations),
                    "batchable_synthesis_count": len(batch_items),
                    "singleton_count": len(singleton_indices),
                    "groups": [
                        {
                            "pipeline": plan.pipeline,
                            "count": len(plan.records),
                            "operation_indices": [
                                batch_operation_indices[item_index]
                                for item_index, _, _ in plan.records
                            ],
                        }
                        for plan in plans
                    ],
                }
            ),
        )

        for plan in plans:
            try:
                response = await _execute_synthesis_batch_plan(plan)
                if len(response.items) != len(plan.records):
                    raise RuntimeError(
                        "backend batch returned a different number of results"
                    )
                for (batch_item_index, item, _), response_item in zip(
                    plan.records, response.items, strict=True
                ):
                    request = batch_requests[batch_item_index]
                    result = InferenceResult(
                        kind="audio",
                        content_type=(
                            "audio/mpeg" if item.format == "mp3" else "audio/wav"
                        ),
                        audio=base64.b64decode(response_item.audio_base64),
                    )
                    outcomes[batch_operation_indices[batch_item_index]] = (
                        await _apply_audio_adjustments(
                            result,
                            speed=request.speed,
                            pitch=request.pitch,
                            volume=request.volume,
                        )
                    )
            except HTTPException as error:
                outcome: BaseException = InferenceOperationError(
                    error.status_code, error.detail
                )
                for batch_item_index, _, _ in plan.records:
                    outcomes[batch_operation_indices[batch_item_index]] = outcome
            except ChildWorkerDied as error:
                # A backend subprocess is gone (OOM, crash). Don't retry the
                # leases in this batch — escalate to the worker so it surfaces
                # ``error`` and stops pulling new leases.
                _mark_engine_failed_from_child(error.name, error)
                raise
            except Exception as error:
                _LOGGER.exception(
                    "Inference task batch failed pipeline=%s count=%d",
                    plan.pipeline,
                    len(plan.records),
                )
                for batch_item_index, _, _ in plan.records:
                    outcomes[batch_operation_indices[batch_item_index]] = error

        for operation_index in singleton_indices:
            operation, request_data = operations[operation_index]
            try:
                outcomes[operation_index] = await self.execute(operation, request_data)
            except ChildWorkerDied:
                # Backend subprocess died -- propagate so the worker surfaces
                # ``error`` instead of retrying leases against a dead process.
                raise
            except Exception as error:
                outcomes[operation_index] = error

        if any(outcome is None for outcome in outcomes):
            raise RuntimeError("internal inference batch result ordering error")
        return [outcome for outcome in outcomes if outcome is not None]

    async def execute(self, operation: str, request_data: dict[str, Any]) -> InferenceResult:
        if not self._started:
            raise RuntimeError("Inference session has not been started")
        started = time.perf_counter()
        context = {
            "operation": operation,
            "model": request_data.get("model"),
            "voice": request_data.get("voice_id") or request_data.get("id"),
            "language": request_data.get("language") or request_data.get("locale"),
            "text_chars": len(request_data.get("text") or ""),
            "ssml_chars": len(request_data.get("ssml") or ""),
            "has_reference": bool(request_data.get("reference_url")),
            "speed": request_data.get("speed", 1.0),
            "pitch": request_data.get("pitch", 1.0),
            "volume": request_data.get("volume", 1.0),
        }
        _LOGGER.info("Inference operation started: %s", json.dumps(context, default=str))
        try:
            if operation == "synthesize":
                request = SynthesizeRequest.model_validate(request_data)
                response = await _synthesize(request, model=_resolve_api_model(request.model))
                result = InferenceResult(
                    kind="audio",
                    content_type=response.media_type,
                    audio=bytes(response.body),
                )
                result = await _apply_audio_adjustments(
                    result,
                    speed=request.speed,
                    pitch=request.pitch,
                    volume=request.volume,
                )
            elif operation == "voice-enhance":
                result = await self._enhance(VoiceEnhanceRequest.model_validate(request_data))
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported TTS operation: {operation}")
        except ValidationError as exc:
            _LOGGER.warning("Inference input rejected context=%s errors=%s", context, exc.errors())
            raise InferenceOperationError(400, jsonable_encoder(exc.errors())) from exc
        except HTTPException as exc:
            _LOGGER.warning(
                "Inference operation rejected context=%s status=%s detail=%s",
                context,
                exc.status_code,
                exc.detail,
            )
            raise InferenceOperationError(exc.status_code, exc.detail) from exc
        except ChildWorkerDied as exc:
            # Backend subprocess is gone (OOM, crash). Mark the engine as
            # failed and re-raise so the worker surfaces ``error`` rather than
            # spinning this lease on retries against a dead process.
            _mark_engine_failed_from_child(exc.name, exc)
            raise
        except Exception:
            _LOGGER.exception("Inference operation failed context=%s", context)
            raise

        _LOGGER.info(
            "Inference operation completed context=%s kind=%s bytes=%s wall_seconds=%.3f",
            context,
            result.kind,
            len(result.audio or b"") if result.kind == "audio" else None,
            time.perf_counter() - started,
        )
        return result

    async def _enhance(self, request: VoiceEnhanceRequest) -> InferenceResult:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(request.reference_url)
            response.raise_for_status()
        audio = await asyncio.to_thread(self.voice_enhancer.enhance, response.content)
        return InferenceResult(kind="audio", content_type="audio/mpeg", audio=audio)


class SyncTaskInput(BaseModel):
    """Generic development task accepted by /task/sync."""

    model_config = {"extra": "forbid"}

    operation: Literal["synthesize", "voice-enhance"]
    request: dict[str, Any]


class SyncTaskRequest(BaseModel):
    """Runpod-style synchronous task envelope."""

    model_config = {"extra": "forbid"}

    input: SyncTaskInput


def create_app(config: ServerConfig | None = None, session: LzTtsInferenceSession | None = None) -> FastAPI:
    """Create the development-only synchronous task adapter."""
    global _server_config
    session = session or LzTtsInferenceSession(config)
    _server_config = session.config

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        await session.start()
        try:
            yield
        finally:
            await session.close()

    app = FastAPI(
        title="LZ-TTS synchronous task adapter",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def api_key_auth_middleware(request: Request, call_next):
        # Health probes must work without credentials so the orchestrator can
        # gate traffic on the worker's Taskflow readiness.
        if request.url.path == "/health":
            return await call_next(request)
        provided_api_key = _request_api_key(request)
        _scrub_api_key_query_param(request)
        expected_api_key = _configured_api_key()
        if not expected_api_key:
            return JSONResponse(status_code=503, content={"error": "API_KEY is not configured"})
        if not provided_api_key or not secrets.compare_digest(provided_api_key, expected_api_key):
            return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        info = get_health_status()
        # 200 only when fully joined and healthy. Both ``starting`` (still
        # booting / reconnecting from cold) and ``error`` (joined but
        # unhealthy) return 503 so orchestrators pull traffic in both cases.
        return JSONResponse(
            status_code=200 if info["status"] == "ok" else 503,
            content=info,
        )

    @app.post("/task/sync")
    async def sync_task(task: SyncTaskRequest):
        started = time.perf_counter()
        try:
            result = await session.execute(task.input.operation, task.input.request)
        except InferenceOperationError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"status": "FAILED", "error": jsonable_encoder(exc.detail)},
            )
        except Exception as exc:
            _LOGGER.exception("Synchronous inference task failed operation=%s", task.input.operation)
            return JSONResponse(
                status_code=500,
                content={"status": "FAILED", "error": str(exc)},
            )

        if result.kind == "json":
            output: dict[str, Any] = {"kind": "json", "data": result.data}
        else:
            output = {
                "kind": "audio",
                "contentType": result.content_type,
                "audioBase64": base64.b64encode(result.audio or b"").decode("ascii"),
            }
        return {
            "status": "COMPLETED",
            "executionTime": time.perf_counter() - started,
            "output": output,
        }

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
