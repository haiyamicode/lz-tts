"""FastAPI server for Piper TTS inference."""

from __future__ import annotations

import io
import json
import logging
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..multilingual_splitter import MultilingualSplitter
from ..piper import PiperInference
from ..ssml import BreakSegment, TextSegment, generate_silence, parse_ssml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
_LOGGER = logging.getLogger(__name__)

# Default paths
DATA_DIR = Path("data")
CONFIG_PATH = Path("local/server.json")
DEFAULT_MODEL = "lzspeech-enzhja-1000-bert"

class ServerConfig(BaseModel):
    """Server configuration."""

    default_model: str = DEFAULT_MODEL
    preload_models: list[str] = Field(default_factory=list)
    model_priority: list[str] = Field(default_factory=list)
    lang_speaker_map: dict[str, str] = Field(default_factory=dict)


class SynthesizeRequest(BaseModel):
    """Request body for text synthesis."""

    text: Optional[str] = Field(None, description="Plain text to synthesize (mutually exclusive with ssml)")
    ssml: Optional[str] = Field(None, description="SSML to synthesize, must be wrapped in <speak> tags (mutually exclusive with text)")
    speaker: Optional[str] = Field(None, description="Speaker label (overrides auto language detection)")
    noise_scale: Optional[float] = Field(None, description="Prosody randomness")
    length_scale: Optional[float] = Field(None, description="Speech rate multiplier (>1 = slower)")
    noise_w: Optional[float] = Field(None, description="Duration predictor noise")


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
_inference_cache: dict[str, PiperInference] = {}
_server_config: ServerConfig = ServerConfig()
_speaker_routes: dict[str, tuple[str, int]] = {}  # speaker -> (model, speaker_id)
_lang_speaker_map: dict[str, str] = {}  # canonical locale -> speaker
_splitter: MultilingualSplitter | None = None


def _normalize_locale(lang: str) -> str:
    """Normalize locale code to canonical BCP 47 format (e.g., en-us -> en-US)."""
    parts = lang.lower().split("-")
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1].upper()}"
    return parts[0]


def _load_config() -> ServerConfig:
    """Load server configuration from local/server.json."""
    if not CONFIG_PATH.exists():
        return ServerConfig()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ServerConfig(**data)


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


def _load_model(model: str) -> PiperInference:
    """Load a model (used internally, raises ValueError instead of HTTPException)."""
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
    return inference


def _get_inference(model: str) -> PiperInference:
    """Get or create an inference instance for a model."""
    if model in _inference_cache:
        return _inference_cache[model]

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


def _build_speaker_routes(model_priority: list[str]) -> dict[str, tuple[str, int]]:
    """Build speaker routing table based on model priority.

    For each speaker, the first model in the priority list that has that speaker wins.
    """
    routes: dict[str, tuple[str, int]] = {}

    for model_name in model_priority:
        if model_name not in _inference_cache:
            _LOGGER.warning("Model %s in priority list but not loaded, skipping", model_name)
            continue

        inference = _inference_cache[model_name]
        for speaker, speaker_id in inference.speakers.items():
            if speaker not in routes:
                routes[speaker] = (model_name, speaker_id)
                _LOGGER.debug("Routing speaker '%s' -> model '%s' (id=%d)", speaker, model_name, speaker_id)

    return routes


def _resolve_model_for_speaker(speaker: str | None) -> tuple[str, str | None]:
    """Resolve which model to use for a given speaker.

    Returns (model_name, speaker_label). If speaker is None or not found in routes,
    falls back to default model.
    """
    if speaker and speaker in _speaker_routes:
        model_name, _ = _speaker_routes[speaker]
        return model_name, speaker

    return _server_config.default_model, speaker


def _resolve_lang_to_speaker(lang: str) -> str:
    """Resolve a language code to a speaker label.

    Uses lang_speaker_map for explicit mappings, then falls back to
    checking speaker_routes for the language code itself.
    """
    lang_canonical = _normalize_locale(lang)

    # Check explicit mapping first
    if lang_canonical in _lang_speaker_map:
        return _lang_speaker_map[lang_canonical]

    # Build fallback chain
    parts = lang_canonical.split("-")
    base_lang = parts[0]
    if len(parts) == 2:
        if parts[0] == parts[1].lower():
            # "fr-FR" -> just "fr"
            candidates = [base_lang]
        else:
            # "en-GB" -> try "en-GB" first, then "en"
            candidates = [lang_canonical, base_lang]
    else:
        candidates = [base_lang]

    # Find first speaker that exists in routes
    for cand in candidates:
        if cand in _speaker_routes:
            return cand

    return candidates[0]


def _synthesize_multilingual(
    text: str,
    noise_scale: Optional[float] = None,
    length_scale: Optional[float] = None,
    noise_w: Optional[float] = None,
) -> tuple[np.ndarray, int]:
    """Synthesize multilingual text using multiple models.

    Returns (audio, sample_rate).
    """
    global _splitter
    if _splitter is None:
        _splitter = MultilingualSplitter()

    result = _splitter.split(text)
    segments = result.segments
    main_lang = result.main_language or "en"

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
        speaker = _resolve_lang_to_speaker(lang)
        model_name, _ = _resolve_model_for_speaker(speaker)

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

        # Check if speaker exists in this model
        if speaker not in inference.speakers:
            _LOGGER.warning("Speaker '%s' not in model '%s', using first available", speaker, model_name)
            speaker = next(iter(inference.speakers.keys()))

        audio = inference.synthesize_span(seg_text, speaker=speaker, **synth_kwargs)
        audio_parts.append(audio)

    if not audio_parts:
        return np.array([], dtype=np.int16), sample_rate

    if len(audio_parts) == 1:
        return audio_parts[0], sample_rate

    return np.concatenate(audio_parts, axis=0), sample_rate


def _synthesize_ssml(
    ssml_text: str,
    global_speaker: Optional[str] = None,
    noise_scale: Optional[float] = None,
    length_scale: Optional[float] = None,
    noise_w: Optional[float] = None,
) -> tuple[np.ndarray, int]:
    """Synthesize SSML text with break and multilingual support.

    Args:
        ssml_text: SSML string to synthesize.
        global_speaker: If set, overrides all segment speakers.

    Returns (audio, sample_rate).
    """
    global _splitter
    if _splitter is None:
        _splitter = MultilingualSplitter()

    segments = parse_ssml(ssml_text)

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

            # Determine speaker: global override > segment speaker > auto-detect
            if global_speaker is not None:
                # Global override
                model_name, _ = _resolve_model_for_speaker(global_speaker)
                routing_plan.append({
                    "type": "text",
                    "speaker": global_speaker,
                    "model": model_name,
                    "text": seg_text,
                })
            elif seg.speaker is not None:
                # Segment-level speaker from <voice name="...">
                model_name, _ = _resolve_model_for_speaker(seg.speaker)
                routing_plan.append({
                    "type": "text",
                    "speaker": seg.speaker,
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
                    speaker = _resolve_lang_to_speaker(lang)
                    model_name, _ = _resolve_model_for_speaker(speaker)

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

    @app.on_event("startup")
    async def startup_event():
        """Preload models and build routing table on startup."""
        global _speaker_routes, _lang_speaker_map

        # Build canonical lookup for lang_speaker_map
        _lang_speaker_map.clear()
        for locale, speaker in _server_config.lang_speaker_map.items():
            canonical = _normalize_locale(locale)
            _lang_speaker_map[canonical] = speaker

        if _server_config.preload_models:
            _LOGGER.info("Preloading %d models...", len(_server_config.preload_models))
            _preload_models(_server_config.preload_models)
            _LOGGER.info("Preloading complete. Loaded models: %s", list(_inference_cache.keys()))

        if _server_config.model_priority:
            _speaker_routes = _build_speaker_routes(_server_config.model_priority)
            _LOGGER.info("Built speaker routes for %d speakers", len(_speaker_routes))

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
            "models_loaded": list(_inference_cache.keys()),
            "default_model": _server_config.default_model,
            "speakers": speakers,
        }

    @app.get("/models", response_model=list[str])
    async def list_models():
        """List available models."""
        return _list_available_models()

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
                **synth_kwargs,
            )
        elif request.speaker is None and model is None:
            # Auto multilingual synthesis (default)
            audio, sample_rate = _synthesize_multilingual(request.text, **synth_kwargs)
        else:
            # Single-model synthesis
            if model is None:
                model, _ = _resolve_model_for_speaker(request.speaker)
            inference = _get_inference(model)

            audio = inference.synthesize(
                text=request.text,
                speaker=request.speaker,
                **synth_kwargs,
            )
            sample_rate = inference.sample_rate

        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")

    @app.get("/synthesize")
    async def synthesize_get(
        text: Optional[str] = Query(None, description="Plain text to synthesize (mutually exclusive with ssml)"),
        ssml: Optional[str] = Query(None, description="SSML to synthesize, must be wrapped in <speak> tags (mutually exclusive with text)"),
        model: str = Query(None, description="Model to use (overrides auto routing)"),
        speaker: Optional[str] = Query(None, description="Speaker label (overrides auto language detection)"),
        noise_scale: Optional[float] = Query(None, description="Prosody randomness"),
        length_scale: Optional[float] = Query(None, description="Speech rate multiplier"),
        noise_w: Optional[float] = Query(None, description="Duration predictor noise"),
    ):
        """Synthesize text or SSML to speech (GET endpoint for easy testing).

        Provide either `text` (plain text) or `ssml` (SSML with <speak> wrapper), not both.

        By default, text is split by language and routed to appropriate speakers automatically.
        Specify `speaker` to override and use a single speaker for the entire text.
        """
        # Validate exactly one of text or ssml is provided
        if text and ssml:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'ssml', not both")
        if not text and not ssml:
            raise HTTPException(status_code=400, detail="Must provide either 'text' or 'ssml'")

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
                **synth_kwargs,
            )
        elif speaker is None and model is None:
            # Auto multilingual synthesis (default)
            audio, sample_rate = _synthesize_multilingual(text, **synth_kwargs)
        else:
            # Single-speaker synthesis
            if model is None:
                model, _ = _resolve_model_for_speaker(speaker)
            inference = _get_inference(model)

            audio = inference.synthesize(
                text=text,
                speaker=speaker,
                **synth_kwargs,
            )
            sample_rate = inference.sample_rate

        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")

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
