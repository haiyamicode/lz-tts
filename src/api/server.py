"""FastAPI server for Sparrow/VITS TTS inference."""

from __future__ import annotations

import base64
import asyncio
import contextlib
import gc
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
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydub import AudioSegment

from ..multilingual_splitter import MultilingualSplitter
from ..piper import PiperInference
from ..ssml import BreakSegment, TextSegment, generate_silence, parse_ssml
from ..matcha_inference import MatchaBackend as ProductionMatchaBackend
from ..matcha_inference import MatchaBatcher as ProductionMatchaBatcher
from . import qwen3
from .qwen3 import router as qwen3_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
_LOGGER = logging.getLogger(__name__)
load_dotenv()

# Default paths
DATA_DIR = Path("data")
CONFIG_PATH = Path(os.environ.get("LZ_TTS_SERVER_CONFIG", "local/server.json"))
DEFAULT_MODEL = "lzspeech-sparrow"
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

class ModelConfig(BaseModel):
    """Per-model configuration override."""

    # Speaker mappings: {"speaker_label": speaker_id_or_null}
    # Use null for single-speaker models that don't need a speaker ID
    speakers: dict[str, Optional[int]] = Field(default_factory=dict)
    # Override espeak voice for phonemization (e.g., "en-us", "en-gb")
    phoneme_voice: Optional[str] = None


class EngineEnableConfig(BaseModel):
    """Global engine switches. Disabled engines are not mounted or loaded."""

    pipertts: bool = Field(default_factory=lambda: _env_bool("PIPER_TTS_ENABLED", True))
    qwen3: bool = Field(default_factory=lambda: _env_bool("QWEN_TTS_ENABLED", True))
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


class ServerConfig(BaseModel):
    """Server configuration."""

    engines: EngineEnableConfig = Field(default_factory=EngineEnableConfig)
    pipertts: PiperTTSConfig = Field(default_factory=PiperTTSConfig)
    qwen: QwenTTSConfig = Field(default_factory=QwenTTSConfig)
    matcha: MatchaConfig = Field(default_factory=MatchaConfig)
    seed_vc: SeedVCConfig = Field(default_factory=SeedVCConfig)


class SynthesizeRequest(BaseModel):
    """Request body for text synthesis."""

    text: Optional[str] = Field(None, description="Plain text to synthesize (mutually exclusive with ssml)")
    ssml: Optional[str] = Field(None, description="SSML to synthesize, must be wrapped in <speak> tags (mutually exclusive with text)")
    speaker: Optional[str] = Field(None, description="Speaker label (overrides auto language detection for ALL segments)")
    primary_speaker: Optional[str] = Field(None, description="Speaker for the primary language only (e.g., 'en-GB' applies to English segments, other languages use their defaults)")
    format: Literal["wav", "mp3"] = Field("wav", description="Output audio format (wav or mp3)")
    noise_scale: Optional[float] = Field(None, description="Prosody randomness")
    length_scale: Optional[float] = Field(None, description="Speech rate multiplier (>1 = slower)")
    noise_w: Optional[float] = Field(None, description="Duration predictor noise")
    neural: bool = Field(True, description="Use neural heteronym disambiguation for more accurate pronunciation of ambiguous words")


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
    max_chunk_batch_size: int = Field(16, ge=1, le=64)


class SeedVCFindVoiceRequest(BaseModel):
    reference_url: str
    id: str


class SeedVCEnhanceRequest(BaseModel):
    reference_url: str
    id: str


class SpeakerInfo(BaseModel):
    """Speaker information."""

    label: str
    id: int


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    speakers: list[str]
    bert_enabled: bool


# Global state
_inference_cache: OrderedDict[str, PiperInference] = OrderedDict()
_server_config: ServerConfig = ServerConfig()
_speaker_routes: dict[str, tuple[str, Optional[int]]] = {}  # speaker -> (model, speaker_id or None)
_lang_speaker_map: dict[str, str] = {}  # canonical locale -> speaker
_splitter: MultilingualSplitter | None = None
_matcha_backend: "ProductionMatchaBackend | None" = None
_matcha_batcher: "ProductionMatchaBatcher | None" = None
_seed_vc_backend: "_SeedVCBackend | None" = None


def _normalize_locale(lang: str) -> str:
    """Normalize locale code to canonical BCP 47 format (e.g., en-us -> en-US)."""
    parts = lang.lower().split("-")
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1].upper()}"
    return parts[0]


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


def _engine_enabled(engine: Literal["pipertts", "qwen3", "matcha", "seed_vc"], config: ServerConfig | None = None) -> bool:
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
    """Configured models that may be loaded on demand."""
    if not _engine_enabled("pipertts"):
        return []
    return _server_config.pipertts.models or _list_available_models()


def _is_model_allowed(model: str) -> bool:
    """Check whether a model is allowed to be loaded on demand."""
    return model in _allowed_models()


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
    while len(_inference_cache) > _server_config.pipertts.max_models_in_cache:
        evicted, _ = _inference_cache.popitem(last=False)
        _LOGGER.info("Evicted model from cache: %s", evicted)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _load_model(model: str) -> PiperInference:
    """Load a model (used internally, raises ValueError instead of HTTPException)."""
    if not _engine_enabled("pipertts"):
        raise ValueError("PiperTTS backend is disabled")
    if not _is_model_allowed(model):
        raise ValueError(f"Model is not configured for on-demand use: {model}")

    model_dir = DATA_DIR / model
    config_path = model_dir / "config.json"
    checkpoint_path = _find_checkpoint(model_dir)

    if not config_path.exists():
        raise ValueError(f"Model config not found: {model}")
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found for model: {model}")

    inference = PiperInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    _inference_cache[model] = inference
    _enforce_cache_limit()
    return inference


def _get_inference(model: str) -> PiperInference:
    """Get or create an inference instance for a model."""
    if not _engine_enabled("pipertts"):
        raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")
    if model in _inference_cache:
        inference = _inference_cache.pop(model)
        _inference_cache[model] = inference
        return inference

    try:
        return _load_model(model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _preload_models(models: list[str]) -> None:
    """Preload specified models into the cache."""
    for model in models:
        if model in _inference_cache:
            _LOGGER.info("Model already loaded: %s", model)
            continue
        _LOGGER.info("Preloading model: %s", model)
        try:
            _load_model(model)
            _LOGGER.info("Loaded model: %s", model)
        except ValueError as e:
            _LOGGER.warning("Failed to preload model %s: %s", model, e)


def _build_speaker_routes(model_priority: list[str]) -> dict[str, tuple[str, Optional[int]]]:
    """Build speaker routing table based on model priority.

    For each speaker, the first model in the priority list that has that speaker wins.
    Uses model_config overrides first, then falls back to model config metadata.
    """
    routes: dict[str, tuple[str, Optional[int]]] = {}

    for model_name in model_priority:
        if not _is_model_allowed(model_name):
            _LOGGER.warning("Model %s in priority list but not configured for on-demand use, skipping", model_name)
            continue

        # Check for config override first (useful for single-speaker models with empty labels)
        model_cfg = _server_config.pipertts.model_config_overrides.get(model_name)
        if model_cfg and model_cfg.speakers:
            for speaker, speaker_id in model_cfg.speakers.items():
                if speaker not in routes:
                    routes[speaker] = (model_name, speaker_id)
                    _LOGGER.debug("Routing speaker '%s' -> model '%s' (id=%s) [config override]", speaker, model_name, speaker_id)
        else:
            # Use model's native speaker map
            try:
                speakers = _get_model_speakers(model_name)
            except ValueError as e:
                _LOGGER.warning("Failed to read speakers for model %s: %s", model_name, e)
                continue

            for speaker, speaker_id in speakers.items():
                if speaker and speaker not in routes:  # Skip empty speaker labels
                    routes[speaker] = (model_name, speaker_id)
                    _LOGGER.debug("Routing speaker '%s' -> model '%s' (id=%d)", speaker, model_name, speaker_id)

    return routes


def _resolve_speaker_and_model(input_speaker: str | None) -> tuple[str | None, str]:
    """Resolve speaker to actual speaker label and model name.

    Simple two-step lookup:
    1. Check lang_speaker_map for alias resolution (e.g., "en-US" → "en")
    2. Check speaker_routes for model selection

    Returns (speaker, model_name).
    """
    if input_speaker is None:
        return None, _server_config.pipertts.default_model

    # Step 1: Resolve alias through lang_speaker_map
    normalized = _normalize_locale(input_speaker)
    speaker = _lang_speaker_map.get(normalized, normalized)

    # Step 2: Find model in speaker_routes
    if speaker in _speaker_routes:
        model_name, _ = _speaker_routes[speaker]
        return speaker, model_name

    # Fallback to default model
    return speaker, _server_config.pipertts.default_model


def _synthesize_multilingual(
    text: str,
    primary_speaker: Optional[str] = None,
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

    Returns (audio, sample_rate).
    """
    global _splitter
    if _splitter is None:
        _splitter = MultilingualSplitter()

    result = _splitter.split(text)
    segments = result.segments
    main_lang = result.main_language or "en"

    # Extract base language from primary_speaker if provided
    primary_lang = _get_base_language(primary_speaker) if primary_speaker else None

    synth_kwargs = {}
    if noise_scale is not None:
        synth_kwargs["noise_scale"] = noise_scale
    if length_scale is not None:
        synth_kwargs["length_scale"] = length_scale
    if noise_w is not None:
        synth_kwargs["noise_w"] = noise_w

    # First pass: compute routing plan
    routing_plan: list[dict] = []
    for seg in segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue

        lang = (seg.language if seg.language and seg.language != "und" else main_lang) or "en"

        # Use primary_speaker if language matches, otherwise normal resolution
        if primary_speaker and _get_base_language(lang) == primary_lang:
            speaker, model_name = _resolve_speaker_and_model(primary_speaker)
        else:
            speaker, model_name = _resolve_speaker_and_model(lang)

        routing_plan.append({
            "lang": lang,
            "speaker": speaker,
            "model": model_name,
            "text": seg_text,
        })

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

        # Check if this speaker is configured via model_config (may have null speaker_id)
        model_cfg = _server_config.pipertts.model_config_overrides.get(model_name)
        if model_cfg and speaker in model_cfg.speakers:
            # Use None for single-speaker models configured with null speaker_id
            internal_speaker = None if model_cfg.speakers[speaker] is None else speaker
        elif speaker in inference.speakers:
            internal_speaker = speaker
        else:
            _LOGGER.warning("Speaker '%s' not in model '%s', using first available", speaker, model_name)
            internal_speaker = next(iter(inference.speakers.keys()), None)

        audio = inference.synthesize_span(seg_text, speaker=internal_speaker, neural=neural, **synth_kwargs)
        audio_parts.append(audio)

    if not audio_parts:
        return np.array([], dtype=np.int16), sample_rate

    if len(audio_parts) == 1:
        return audio_parts[0], sample_rate

    return np.concatenate(audio_parts, axis=0), sample_rate


def _synthesize_ssml(
    ssml_text: str,
    global_speaker: Optional[str] = None,
    primary_speaker: Optional[str] = None,
    noise_scale: Optional[float] = None,
    length_scale: Optional[float] = None,
    noise_w: Optional[float] = None,
) -> tuple[np.ndarray, int]:
    """Synthesize SSML text with break and multilingual support.

    Args:
        ssml_text: SSML string to synthesize.
        global_speaker: If set, overrides all segment speakers.
        primary_speaker: If set, use this speaker for segments matching its base language.

    Returns (audio, sample_rate).
    """
    global _splitter
    if _splitter is None:
        _splitter = MultilingualSplitter()

    segments = parse_ssml(ssml_text)

    # Extract base language from primary_speaker if provided
    primary_lang = _get_base_language(primary_speaker) if primary_speaker else None

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
                resolved_speaker, model_name = _resolve_speaker_and_model(global_speaker)
                routing_plan.append({
                    "type": "text",
                    "speaker": resolved_speaker,
                    "model": model_name,
                    "text": seg_text,
                })
            elif seg.speaker is not None:
                # Segment-level speaker from <voice name="...">
                resolved_speaker, model_name = _resolve_speaker_and_model(seg.speaker)
                routing_plan.append({
                    "type": "text",
                    "speaker": resolved_speaker,
                    "model": model_name,
                    "text": seg_text,
                })
            else:
                # Auto-detect: run through multilingual splitter
                result = _splitter.split(seg_text)
                main_lang = result.main_language or "en"

                for lang_seg in result.segments:
                    lang_text = lang_seg.text.strip()
                    if not lang_text:
                        continue

                    lang = (lang_seg.language if lang_seg.language and lang_seg.language != "und" else main_lang) or "en"

                    # Use primary_speaker if language matches, otherwise normal resolution
                    if primary_speaker and _get_base_language(lang) == primary_lang:
                        speaker, model_name = _resolve_speaker_and_model(primary_speaker)
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

            if speaker not in inference.speakers:
                _LOGGER.warning("Speaker '%s' not in model '%s', using first available", speaker, model_name)
                speaker = next(iter(inference.speakers.keys()))

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
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
    return buffer.getvalue()


def _audio_to_mp3_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to MP3 bytes with highest quality settings."""
    # First convert to WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
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
                self.cached_embeddings = {}
                _LOGGER.warning("Seed-VC embeddings file not found: %s", embeddings_path)

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
            estimator.setup_caches(max_batch_size=64, max_seq_length=8192)

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
            torch.tensor(librosa.load(path, sr=self.sample_rate)[0]).unsqueeze(0).float().to(self.device)
            for path in source_paths
        ]
        chunk_records: list[tuple[int, int, int, Any]] = []
        generated_wave_chunks: list[list[np.ndarray]] = [[] for _ in source_audios]
        crossfade_samples = int(self.sample_rate * 0.05)

        for item_idx, source_audio in enumerate(source_audios):
            source_audio_np = source_audio.squeeze(0).cpu().numpy()
            chunks = self.find_silence_boundaries(
                source_audio_np,
                self.sample_rate,
                min_silence_duration=0.15,
                silence_threshold=0.02,
                max_chunk_duration=25.0,
                min_chunk_duration=3.0,
            )
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                chunk_records.append((item_idx, chunk_idx, chunk_start, source_audio[:, chunk_start:chunk_end]))

        for batch_start in range(0, len(chunk_records), max_chunk_batch_size):
            records = chunk_records[batch_start : batch_start + max_chunk_batch_size]
            chunk_audios = [record[3] for record in records]
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
            prompt_batch = prompt_condition.repeat(batch_size, 1, 1)
            mel_ref_batch = mel_ref.repeat(batch_size, 1, 1)
            style_batch = style.repeat(batch_size, 1)
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

        results = []
        for chunks in generated_wave_chunks:
            result = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            results.append(self._fade_seed_vc_start(result, self.sample_rate).astype(np.float32))
        return results

    def convert_batch_request(self, request: SeedVCBatchRequest) -> dict[str, Any]:
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

        import soundfile as sf  # pylint: disable=import-outside-toplevel

        source_paths = [self.tmp_dir / f"{uuid.uuid4().hex}.input" for _ in items]
        wav_output_paths: list[Path] = []
        try:
            for item, source_path in zip(items, source_paths):
                source_path.write_bytes(base64.b64decode(item.audio))

            embedding_key, cached_embeddings = self._resolve_cached_embeddings(first)
            reference_path = None if cached_embeddings is not None else self._fetch_sample(first)
            with self.lock, self.torch.no_grad(), _temporary_cwd(self.root):
                waves = self._convert_voice_v1_batch(
                    source_paths,
                    reference_path,
                    preset_config,
                    cached_embeddings=cached_embeddings,
                    max_chunk_batch_size=request.max_chunk_batch_size,
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

    def find_voice(self, request: SeedVCFindVoiceRequest) -> str:
        if self.find_base_voice is None:
            raise RuntimeError("Seed-VC find_voice support is unavailable")
        sample_request = SeedVCRequest(audio="", reference_url=request.reference_url, id=request.id)
        reference_path = self._fetch_sample(sample_request)
        with self.lock, _temporary_cwd(self.root):
            return str(self.find_base_voice(str(reference_path)))

    def enhance(self, request: SeedVCEnhanceRequest) -> bytes:
        import subprocess
        from pathlib import Path as _Path
        enhance_dir = self.tmp_dir / request.id
        sample_dir = enhance_dir / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        raw_path = enhance_dir / "sample_raw.mp3"
        wav_path = enhance_dir / "sample_raw.wav"

        _LOGGER.info("Seed-VC enhance: fetching sample for %s", request.id)
        with httpx.Client(follow_redirects=True) as client:
            resp = client.get(request.reference_url)
            resp.raise_for_status()
            raw_path.write_bytes(resp.content)

        subprocess.run(["ffmpeg", "-i", str(raw_path), "-t", "120", str(wav_path), "-y"],
                       capture_output=True, check=True)
        subprocess.run(["ffmpeg-normalize", str(wav_path), "-o", str(sample_dir / "sample.wav"), "-f"],
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
    global _seed_vc_backend
    if not _engine_enabled("seed_vc"):
        raise HTTPException(status_code=503, detail="Seed-VC backend is disabled")
    if _seed_vc_backend is None:
        try:
            _seed_vc_backend = _SeedVCBackend(_server_config.seed_vc)
        except Exception as exc:
            _LOGGER.exception("Failed to load Seed-VC backend")
            raise HTTPException(status_code=503, detail=f"Failed to load Seed-VC backend: {exc}") from exc
    return _seed_vc_backend


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
    if _engine_enabled("qwen3", config):
        app.include_router(qwen3_router)
        _mount_qwen_demo(app)

    @app.on_event("startup")
    async def startup_event():
        """Preload models and build routing table on startup."""
        global _speaker_routes, _lang_speaker_map, _matcha_backend, _matcha_batcher, _seed_vc_backend

        _lang_speaker_map.clear()
        _speaker_routes.clear()
        if _engine_enabled("pipertts"):
            # Build canonical lookup for lang_speaker_map
            for locale, speaker in _server_config.pipertts.lang_speaker_map.items():
                canonical = _normalize_locale(locale)
                _lang_speaker_map[canonical] = speaker

            if _server_config.pipertts.preload_models:
                _LOGGER.info("Preloading %d PiperTTS models...", len(_server_config.pipertts.preload_models))
                _preload_models(_server_config.pipertts.preload_models)
                _LOGGER.info("PiperTTS preload complete. Loaded models: %s", list(_inference_cache.keys()))

            route_models = _server_config.pipertts.model_priority or _allowed_models()
            if route_models:
                _speaker_routes = _build_speaker_routes(route_models)
                _LOGGER.info("Built PiperTTS speaker routes for %d speakers", len(_speaker_routes))
        else:
            _LOGGER.info("PiperTTS backend disabled")

        if _engine_enabled("qwen3") and _server_config.qwen.preload:
            if _server_config.qwen.preload_background:
                _LOGGER.info("Starting Qwen3 TTS preload in background...")
                qwen3.start_preload_background(
                    include_dp_budget=_server_config.qwen.dp_budget.preload
                )
            else:
                _LOGGER.info("Preloading Qwen3 TTS...")
                qwen3.preload_model(
                    include_dp_budget=_server_config.qwen.dp_budget.preload
                )
                _LOGGER.info("Qwen3 TTS preload complete")

        if _engine_enabled("matcha") and _server_config.matcha.preload:
            _LOGGER.info("Preloading Matcha backend on %s...", _server_config.matcha.device)
            _matcha_backend = await asyncio.to_thread(ProductionMatchaBackend, _server_config.matcha)
            _matcha_batcher = ProductionMatchaBatcher(_matcha_backend, _server_config.matcha)
            _matcha_batcher.start()
            _LOGGER.info("Matcha backend ready")

        if _engine_enabled("seed_vc") and _server_config.seed_vc.preload:
            _LOGGER.info("Preloading Seed-VC backend on %s...", _server_config.seed_vc.device)
            _seed_vc_backend = await asyncio.to_thread(_SeedVCBackend, _server_config.seed_vc)
            _LOGGER.info("Seed-VC backend ready")

        _LOGGER.info("Server ready")

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
                "preload": _server_config.seed_vc.preload,
                "root": _server_config.seed_vc.root,
                "runtime_root": _server_config.seed_vc.runtime_root,
                "presets": sorted(_SeedVCBackend.model_presets),
            },
            "speakers": speakers,
        }

    @app.get("/llms.txt", response_class=Response)
    async def llms_txt():
        """LLM-readable API documentation."""
        content = """# LZ-TTS API Documentation for LLMs

This is a text-to-speech API that converts text to audio using Piper TTS models.

## Base URL
http://localhost:8000 (or your deployed URL)

## Main Endpoint: /synthesize

### POST /synthesize
Synthesize text or SSML to speech audio.

Request Body (JSON):
{
  "text": "Text to synthesize",           // Plain text (mutually exclusive with ssml)
  "ssml": "<speak>SSML content</speak>", // SSML format (mutually exclusive with text)
  "speaker": "en-US",                    // Optional: speaker/language code
  "format": "mp3",                       // Optional: "wav" (default) or "mp3"
  "noise_scale": 0.667,                  // Optional: prosody randomness (default: 0.667)
  "length_scale": 1.0,                   // Optional: speech rate (>1 = slower, default: 1.0)
  "noise_w": 0.8                         // Optional: duration predictor noise (default: 0.8)
}

Response: Binary audio data (audio/wav or audio/mpeg)

### GET /synthesize
Same as POST but with query parameters for easy testing.

Query Parameters:
- text: Plain text to synthesize (mutually exclusive with ssml)
- ssml: SSML to synthesize (mutually exclusive with text)
- speaker: Speaker/language code (optional)
- format: "wav" or "mp3" (optional, default: "wav")
- noise_scale: Prosody randomness (optional)
- length_scale: Speech rate multiplier (optional)
- noise_w: Duration predictor noise (optional)
- model: Specific model to use (optional, overrides auto routing)

## Audio Format Support
- WAV: Lossless, default format
- MP3: 320kbps CBR with highest quality settings (-q:a 0)

## Multilingual Support
The API automatically detects languages and routes to appropriate speakers.
You can override this by specifying a speaker parameter.

## SSML Support
Use SSML for advanced control:
- <speak>: Root element (required)
- <voice name="speaker">: Change speaker
- <break time="500ms"/>: Insert pauses

Example SSML:
<speak>
  <voice name="en-US">Hello</voice>
  <break time="500ms"/>
  <voice name="ja">こんにちは</voice>
</speak>

## Other Endpoints

GET /: Health check and server info
GET /models: List available models
GET /models/{model}: Get model information
GET /models/{model}/speakers: List speakers for a model

## Example Usage

# Simple text to MP3
curl -X POST "http://localhost:8000/synthesize" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello world", "format": "mp3"}' \\
  -o output.mp3

# GET request with text
curl "http://localhost:8000/synthesize?text=Hello+world&format=mp3" -o output.mp3

# Multilingual SSML
curl -X POST "http://localhost:8000/synthesize" \\
  -H "Content-Type: application/json" \\
  -d '{"ssml": "<speak>Hello <break time=\\"500ms\\"/> こんにちは</speak>", "format": "mp3"}' \\
  -o output.mp3

# Custom speech parameters
curl -X POST "http://localhost:8000/synthesize" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Slower speech", "length_scale": 1.5, "format": "mp3"}' \\
  -o output.mp3

## Notes
- Provide either 'text' OR 'ssml', not both
- Default format is WAV (lossless)
- MP3 format requires ffmpeg to be installed on the server
- Automatic language detection and speaker routing when speaker is not specified
- Speaker parameter overrides automatic language detection
"""
        return Response(content=content, media_type="text/plain")

    @app.get("/models", response_model=list[str])
    async def list_models():
        """List models enabled for on-demand use."""
        return _allowed_models()

    @app.get("/models/{model}", response_model=ModelInfo)
    async def get_model_info(model: str):
        """Get information about a specific model."""
        inference = _get_inference(model)
        return ModelInfo(
            name=model,
            speakers=list(inference.speakers.keys()),
            bert_enabled=inference.use_bert,
        )

    @app.get("/models/{model}/speakers", response_model=list[SpeakerInfo])
    async def list_model_speakers(model: str):
        """List speakers for a specific model."""
        inference = _get_inference(model)
        return [SpeakerInfo(label=label, id=sid) for label, sid in inference.speakers.items()]

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
        return Response(content=_audio_to_wav_bytes(result.audio, result.sample_rate), media_type="audio/wav", headers=headers)

    @app.get("/seed-vc/status")
    async def seed_vc_status():
        """Embedded Seed-VC backend status."""
        return {
            "enabled": _engine_enabled("seed_vc"),
            "loaded": _seed_vc_backend is not None,
            "device": _server_config.seed_vc.device,
            "preload": _server_config.seed_vc.preload,
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
            result = await asyncio.to_thread(lambda: _get_seed_vc_backend().convert_batch_request(request))
            result["wall_time_sec"] = time.perf_counter() - started
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("Seed-VC batch conversion failed")
            raise HTTPException(status_code=500, detail=f"Seed-VC batch conversion failed: {exc}") from exc
        return result

    @app.post("/find-voice")
    async def seed_vc_find_voice(request: SeedVCFindVoiceRequest):
        """Seed-VC compatible voice lookup endpoint."""
        try:
            voice_id = await asyncio.to_thread(lambda: _get_seed_vc_backend().find_voice(request))
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
            mp3_bytes = await asyncio.to_thread(lambda: _get_seed_vc_backend().enhance(request))
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("Seed-VC enhance failed")
            raise HTTPException(status_code=500, detail=f"Seed-VC enhance failed: {exc}") from exc
        return Response(content=mp3_bytes, media_type="audio/mpeg")

    @app.post("/synthesize")
    async def synthesize(
        request: SynthesizeRequest,
        model: str = Query(None, description="Model to use (overrides auto routing)"),
    ):
        """Synthesize text or SSML to speech.

        Provide either `text` (plain text) or `ssml` (SSML with <speak> wrapper), not both.

        By default, text is split by language and routed to appropriate speakers automatically.
        Specify `speaker` to override and use a single speaker for the entire text.
        """
        # Validate exactly one of text or ssml is provided
        if request.text and request.ssml:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'ssml', not both")
        if not request.text and not request.ssml:
            raise HTTPException(status_code=400, detail="Must provide either 'text' or 'ssml'")
        if not _engine_enabled("pipertts"):
            raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")

        synth_kwargs = {}
        if request.noise_scale is not None:
            synth_kwargs["noise_scale"] = request.noise_scale
        if request.length_scale is not None:
            synth_kwargs["length_scale"] = request.length_scale
        if request.noise_w is not None:
            synth_kwargs["noise_w"] = request.noise_w

        if request.ssml:
            # SSML synthesis (speaker param overrides all, otherwise auto-detect)
            audio, sample_rate = _synthesize_ssml(
                request.ssml,
                global_speaker=request.speaker,
                primary_speaker=request.primary_speaker,
                **synth_kwargs,
            )
        elif request.speaker is None and model is None:
            # Auto multilingual synthesis (default, with optional primary_speaker)
            audio, sample_rate = _synthesize_multilingual(
                request.text,
                primary_speaker=request.primary_speaker,
                neural=request.neural,
                **synth_kwargs,
            )
        else:
            # Single-model synthesis
            if model is None:
                request_speaker, model = _resolve_speaker_and_model(request.speaker)
                # Use the resolved speaker instead of the original input
                request.speaker = request_speaker
            inference = _get_inference(model)

            audio = inference.synthesize(
                text=request.text,
                speaker=request.speaker,
                neural=request.neural,
                **synth_kwargs,
            )
            sample_rate = inference.sample_rate

        # Convert to requested format
        if request.format == "mp3":
            audio_bytes = _audio_to_mp3_bytes(audio, sample_rate)
            media_type = "audio/mpeg"
        else:
            audio_bytes = _audio_to_wav_bytes(audio, sample_rate)
            media_type = "audio/wav"

        return Response(content=audio_bytes, media_type=media_type)

    @app.get("/synthesize")
    async def synthesize_get(
        text: Optional[str] = Query(None, description="Plain text to synthesize (mutually exclusive with ssml)"),
        ssml: Optional[str] = Query(None, description="SSML to synthesize, must be wrapped in <speak> tags (mutually exclusive with text)"),
        model: str = Query(None, description="Model to use (overrides auto routing)"),
        speaker: Optional[str] = Query(None, description="Speaker label (overrides auto language detection for ALL segments)"),
        primary_speaker: Optional[str] = Query(None, description="Speaker for primary language only (e.g., 'en-GB' applies to English segments)"),
        format: Literal["wav", "mp3"] = Query("wav", description="Output audio format (wav or mp3)"),
        noise_scale: Optional[float] = Query(None, description="Prosody randomness"),
        length_scale: Optional[float] = Query(None, description="Speech rate multiplier"),
        noise_w: Optional[float] = Query(None, description="Duration predictor noise"),
        neural: bool = Query(True, description="Use neural heteronym disambiguation"),
    ):
        """Synthesize text or SSML to speech (GET endpoint for easy testing).

        Provide either `text` (plain text) or `ssml` (SSML with <speak> wrapper), not both.

        By default, text is split by language and routed to appropriate speakers automatically.
        Specify `speaker` to override and use a single speaker for the entire text.
        Specify `primary_speaker` to override only segments matching that language (e.g., 'en-GB' for English).
        """
        # Validate exactly one of text or ssml is provided
        if text and ssml:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'ssml', not both")
        if not text and not ssml:
            raise HTTPException(status_code=400, detail="Must provide either 'text' or 'ssml'")
        if not _engine_enabled("pipertts"):
            raise HTTPException(status_code=503, detail="PiperTTS backend is disabled")

        synth_kwargs = {}
        if noise_scale is not None:
            synth_kwargs["noise_scale"] = noise_scale
        if length_scale is not None:
            synth_kwargs["length_scale"] = length_scale
        if noise_w is not None:
            synth_kwargs["noise_w"] = noise_w

        if ssml:
            # SSML synthesis (speaker param overrides all, otherwise auto-detect)
            audio, sample_rate = _synthesize_ssml(
                ssml,
                global_speaker=speaker,
                primary_speaker=primary_speaker,
                **synth_kwargs,
            )
        elif speaker is None and model is None:
            # Auto multilingual synthesis (default, with optional primary_speaker)
            audio, sample_rate = _synthesize_multilingual(
                text,
                primary_speaker=primary_speaker,
                neural=neural,
                **synth_kwargs,
            )
        else:
            # Single-speaker synthesis
            if model is None:
                speaker, model = _resolve_speaker_and_model(speaker)
            inference = _get_inference(model)

            audio = inference.synthesize(
                text=text,
                speaker=speaker,
                neural=neural,
                **synth_kwargs,
            )
            sample_rate = inference.sample_rate

        # Convert to requested format
        if format == "mp3":
            audio_bytes = _audio_to_mp3_bytes(audio, sample_rate)
            media_type = "audio/mpeg"
        else:
            audio_bytes = _audio_to_wav_bytes(audio, sample_rate)
            media_type = "audio/wav"

        return Response(content=audio_bytes, media_type=media_type)

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
