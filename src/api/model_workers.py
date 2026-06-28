"""Import-clean process worker entrypoints for model-serving engines."""

from __future__ import annotations

import gc
import asyncio
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from fastapi import HTTPException

from ..matcha_inference import MatchaBackend as ProductionStarlingBackend
from ..matcha_inference import MatchaBatchRequest
from ..piper import PiperInference
from .seed_vc_backend import (
    SeedVCBackend,
    SeedVCBatchRequest,
    SeedVCEnhanceRequest,
    SeedVCFindVoiceRequest,
    SeedVCRequest,
)
from .worker_common import run_worker_loop

_LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL = "lzspeech-sparrow"

_inference_cache: OrderedDict[str, PiperInference] = OrderedDict()
_server_config: SimpleNamespace | None = None
_speaker_routes: dict[str, tuple[str, Optional[int]]] = {}
_lang_speaker_map: dict[str, str] = {}


def _settings_data_to_config(settings_data: dict[str, Any]) -> SimpleNamespace:
    sections = {
        key: SimpleNamespace(**value) if isinstance(value, dict) else value
        for key, value in settings_data.items()
    }
    return SimpleNamespace(**sections)


def _normalize_locale(lang: str) -> str:
    parts = lang.lower().split("-")
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1].upper()}"
    return parts[0]


def _pipertts_enabled() -> bool:
    if _server_config is None:
        return False
    engines = getattr(_server_config, "engines", SimpleNamespace())
    pipertts = getattr(_server_config, "pipertts", SimpleNamespace(enabled=True))
    return bool(getattr(engines, "pipertts", True) and getattr(pipertts, "enabled", True))


def _list_available_models() -> list[str]:
    if not DATA_DIR.exists():
        return []
    models = []
    for path in DATA_DIR.iterdir():
        if path.is_dir() and (path / "config.json").exists():
            models.append(path.name)
    return sorted(models)


def _allowed_models() -> list[str]:
    if _server_config is None or not _pipertts_enabled():
        return []
    models = list(getattr(_server_config.pipertts, "models", []) or [])
    return models or _list_available_models()


def _is_model_allowed(model: str) -> bool:
    return model in _allowed_models()


def _append_unique(items: list[str], value: str | None) -> None:
    if value and value not in items:
        items.append(value)


def _required_piper_models() -> list[str]:
    if _server_config is None or not _pipertts_enabled():
        return []

    config = _server_config.pipertts
    models: list[str] = []
    for model in getattr(config, "preload_models", []) or []:
        _append_unique(models, model)
    for model in getattr(config, "model_priority", []) or []:
        _append_unique(models, model)
    for model in getattr(config, "models", []) or []:
        _append_unique(models, model)
    _append_unique(models, getattr(config, "default_model", DEFAULT_MODEL))

    if not models:
        models = _allowed_models()
    else:
        for model in _allowed_models():
            _append_unique(models, model)
    return models


def _find_checkpoint(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    checkpoints = list(model_dir.glob("*.ckpt"))
    if checkpoints:
        checkpoints.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return checkpoints[0]
    return None


def _get_model_speakers(model: str) -> dict[str, int]:
    config_path = DATA_DIR / model / "config.json"
    if not config_path.exists():
        raise ValueError(f"Model config not found: {model}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {str(label): int(sid) for label, sid in (data.get("speaker_id_map") or {}).items()}


def _enforce_cache_limit() -> None:
    if _server_config is None:
        return
    limit = max(int(getattr(_server_config.pipertts, "max_models_in_cache", 1)), len(_required_piper_models()))
    while len(_inference_cache) > limit:
        evicted, _ = _inference_cache.popitem(last=False)
        _LOGGER.info("Evicted model from cache: %s", evicted)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_model(model: str) -> PiperInference:
    if not _pipertts_enabled():
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

    _LOGGER.info("Loading Sparrow model model=%s checkpoint=%s config=%s", model, checkpoint_path, config_path)
    started = time.perf_counter()
    try:
        inference = PiperInference(checkpoint_path=checkpoint_path, config_path=config_path)
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


def _preload_models(models: list[str], *, strict: bool = False) -> None:
    errors: list[str] = []
    for model in models:
        if model in _inference_cache:
            _LOGGER.info("Model already loaded: %s", model)
            continue
        _LOGGER.info("Preloading model: %s", model)
        try:
            _load_model(model)
            _LOGGER.info("Loaded model: %s", model)
        except ValueError as exc:
            message = f"{model}: {exc}"
            if strict:
                errors.append(message)
            else:
                _LOGGER.warning("Failed to preload model %s: %s", model, exc)
    if errors:
        raise RuntimeError("Failed to preload Sparrow models: " + "; ".join(errors))


def _preload_piper_text_models() -> None:
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

    _LOGGER.info("Loaded Sparrow text models semantic_models=%d heteronym_device=%s", semantic_count, device or "auto")


def _model_override_speakers(model_name: str) -> dict[str, int | None]:
    if _server_config is None:
        return {}
    overrides = getattr(_server_config.pipertts, "model_config_overrides", {}) or {}
    model_cfg = overrides.get(model_name) if isinstance(overrides, dict) else None
    if model_cfg is None:
        return {}
    if isinstance(model_cfg, dict):
        return dict(model_cfg.get("speakers") or {})
    return dict(getattr(model_cfg, "speakers", {}) or {})


def _build_speaker_routes(model_priority: list[str]) -> dict[str, tuple[str, Optional[int]]]:
    routes: dict[str, tuple[str, Optional[int]]] = {}

    for model_name in model_priority:
        if not _is_model_allowed(model_name):
            raise RuntimeError(f"Model {model_name!r} is in priority list but not configured for this server")

        override_speakers = _model_override_speakers(model_name)
        if override_speakers:
            for speaker, speaker_id in override_speakers.items():
                if speaker not in routes:
                    routes[speaker] = (model_name, speaker_id)
                    _LOGGER.debug(
                        "Routing speaker '%s' -> model '%s' (id=%s) [config override]",
                        speaker,
                        model_name,
                        speaker_id,
                    )
        else:
            for speaker, speaker_id in _get_model_speakers(model_name).items():
                if speaker and speaker not in routes:
                    routes[speaker] = (model_name, speaker_id)
                    _LOGGER.debug("Routing speaker '%s' -> model '%s' (id=%d)", speaker, model_name, speaker_id)

    return routes


def sparrow_worker_main(settings_data: dict[str, Any], request_queue: Any, response_queue: Any) -> None:
    """Own all Sparrow/Piper model instances in a dedicated process."""
    global _server_config, _speaker_routes, _lang_speaker_map

    _server_config = _settings_data_to_config(settings_data)
    _inference_cache.clear()
    _speaker_routes.clear()
    _lang_speaker_map.clear()
    for locale, speaker in getattr(_server_config.pipertts, "lang_speaker_map", {}).items():
        _lang_speaker_map[_normalize_locale(locale)] = speaker

    required_models = _required_piper_models()
    if not required_models:
        raise RuntimeError("PiperTTS is enabled but no Sparrow models are configured or available")
    _preload_models(required_models, strict=True)
    _preload_piper_text_models()

    route_models = list(getattr(_server_config.pipertts, "model_priority", []) or []) or _allowed_models()
    if route_models:
        _speaker_routes = _build_speaker_routes(route_models)

    def model_info() -> dict[str, dict[str, Any]]:
        return {
            name: {
                "sample_rate": inference.sample_rate,
                "speakers": dict(getattr(inference, "speakers", {}) or {}),
                "use_bert": bool(getattr(inference, "use_bert", False)),
            }
            for name, inference in _inference_cache.items()
        }

    def handler(action: str, payload: Any) -> dict[str, Any]:
        if action == "health":
            return {"ok": True, "data": {"worker": "ok", "models_loaded": list(_inference_cache.keys()), "models": model_info()}}
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        if action == "synthesize_batch":
            model_name = str(payload["model"])
            inference = _inference_cache.get(model_name)
            if inference is None:
                raise HTTPException(status_code=503, detail=f"Model was not loaded in Sparrow worker: {model_name}")
            audios = inference.synthesize_batch(
                list(payload.get("texts") or []),
                speaker=payload.get("speaker"),
                batch_size=int(payload.get("batch_size") or len(payload.get("texts") or []) or 1),
                neural=bool(payload.get("neural", True)),
                **dict(payload.get("synth_kwargs") or {}),
            )
            return {"ok": True, "data": {"audios": audios, "sample_rate": inference.sample_rate}}
        if action == "synthesize_span":
            model_name = str(payload["model"])
            inference = _inference_cache.get(model_name)
            if inference is None:
                raise HTTPException(status_code=503, detail=f"Model was not loaded in Sparrow worker: {model_name}")
            audio = inference.synthesize_span(
                str(payload.get("text") or ""),
                speaker=payload.get("speaker"),
                neural=bool(payload.get("neural", True)),
                **dict(payload.get("synth_kwargs") or {}),
            )
            return {"ok": True, "data": {"audio": audio, "sample_rate": inference.sample_rate}}
        raise ValueError(f"unknown worker action: {action}")

    run_worker_loop("Sparrow", handler, request_queue, response_queue)


def starling_worker_main(settings_data: dict[str, Any], request_queue: Any, response_queue: Any) -> None:
    """Own the Starling model in a dedicated process."""
    config = _settings_data_to_config(settings_data)
    backend = ProductionStarlingBackend(config.starling)

    def handler(action: str, payload: Any) -> dict[str, Any]:
        if action == "health":
            return {
                "ok": True,
                "data": {
                    "worker": "ok",
                    "sample_rate": backend.sample_rate,
                    "device": config.starling.device,
                    "checkpoint": config.starling.checkpoint,
                },
            }
        if action == "synthesize_batch":
            if not isinstance(payload, dict):
                raise ValueError("worker payload must be an object")
            items = []
            for item in list(payload.get("items") or []):
                items.append(
                    MatchaBatchRequest(
                        text=str(item.get("text") or ""),
                        language=str(item.get("language") or "en"),
                        input_type=str(item.get("input_type") or "aligned"),
                        speaker_id=item.get("speaker_id"),
                        neural=bool(item.get("neural", True)),
                        steps=item.get("steps"),
                        temperature=item.get("temperature"),
                        length_scale=item.get("length_scale"),
                        future=None,
                        queued_at=float(item.get("queued_at") or time.perf_counter()),
                    )
                )
            results = backend.synthesize_batch(items)
            return {"ok": True, "data": {"results": results}}
        raise ValueError(f"unknown worker action: {action}")

    run_worker_loop("Starling", handler, request_queue, response_queue)


def seed_vc_worker_main(settings_data: dict[str, Any], request_queue: Any, response_queue: Any) -> None:
    """Own the Seed-VC model in a dedicated process."""
    config = _settings_data_to_config(settings_data)
    backend = SeedVCBackend(config.seed_vc)

    def handler(action: str, payload: Any) -> dict[str, Any]:
        if action == "health":
            embedding_keys = list(backend.cached_embeddings.keys()) if backend.cached_embeddings else []
            return {
                "ok": True,
                "data": {
                    "worker": "ok",
                    "sample_rate": backend.sample_rate,
                    "device": str(backend.device),
                    "root": str(backend.root),
                    "runtime_root": str(backend.runtime_root),
                    "tmp_dir": str(backend.tmp_dir),
                    "output_dir": str(backend.output_dir),
                    "voice_samples_dir": str(backend.voice_samples_dir),
                    "embedding_keys": embedding_keys,
                    "presets": sorted(SeedVCBackend.model_presets),
                },
            }
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        if action == "resolve_exact_cached_embeddings":
            voice_id = str(payload["voice_id"])
            style = str(payload.get("style") or "general")
            intensity = float(payload.get("intensity") if payload.get("intensity") is not None else 1.0)
            key, _ = backend._resolve_exact_cached_embeddings(voice_id, style, intensity)
            return {"ok": True, "data": {"embedding_key": key}}
        if action == "resolve_cached_embeddings":
            request = SeedVCRequest(**payload["request"])
            key, emb = backend._resolve_cached_embeddings(request)
            return {"ok": True, "data": {"embedding_key": key, "cached": emb is not None}}
        if action == "fetch_sample":
            request = SeedVCRequest(**payload["request"])
            path = asyncio.run(backend._fetch_sample(request))
            return {"ok": True, "data": {"path": str(path)}}
        if action == "convert_with_reference":
            request = SeedVCRequest(**payload["request"])
            reference_path = Path(payload["reference_path"]) if payload.get("reference_path") else None
            embedding_key = payload.get("embedding_key")
            emb = backend.cached_embeddings.get(embedding_key) if embedding_key and backend.cached_embeddings else None
            data = backend._convert_with_reference(request, reference_path, embedding_key=embedding_key, cached_embeddings=emb)
            return {"ok": True, "data": {"audio": data, "sample_rate": backend.sample_rate}}
        if action == "convert_batch_request":
            request = SeedVCBatchRequest(**payload["request"])
            reference_path = Path(payload["reference_path"]) if payload.get("reference_path") else None
            embedding_key = payload.get("embedding_key")
            emb = backend.cached_embeddings.get(embedding_key) if embedding_key and backend.cached_embeddings else None
            data = backend.convert_batch_request(request, reference_path, embedding_key=embedding_key, cached_embeddings=emb)
            return {"ok": True, "data": data}
        if action == "convert_generated_audio_batch":
            data = backend.convert_generated_audio_batch(
                source_audios=list(payload.get("source_audios") or []),
                source_sample_rate=int(payload["source_sample_rate"]),
                voice_id=str(payload["voice_id"]),
                style=str(payload.get("style") or "general"),
                intensity=float(payload.get("intensity") if payload.get("intensity") is not None else 1.0),
                preset=payload.get("preset"),
                output_format=payload.get("output_format") or "mp3",
                max_chunk_batch_size=payload.get("max_chunk_batch_size"),
                strict_embedding=bool(payload.get("strict_embedding", False)),
            )
            return {"ok": True, "data": {"items": data, "sample_rate": backend.sample_rate}}
        if action == "convert_generated_audio_reference_batch":
            data = backend.convert_generated_audio_reference_batch(
                source_audios=list(payload.get("source_audios") or []),
                source_sample_rate=int(payload["source_sample_rate"]),
                reference_path=Path(payload["reference_path"]),
                preset=payload.get("preset"),
                output_format=payload.get("output_format") or "mp3",
                max_chunk_batch_size=payload.get("max_chunk_batch_size"),
            )
            return {"ok": True, "data": {"items": data, "sample_rate": backend.sample_rate}}
        if action == "find_voice":
            request = SeedVCFindVoiceRequest(**payload["request"])
            result = backend.find_voice(request, Path(payload["reference_path"]))
            return {"ok": True, "data": {"voice_id": result}}
        if action == "enhance":
            request = SeedVCEnhanceRequest(**payload["request"])
            data = backend.enhance(request, Path(payload["raw_path"]))
            return {"ok": True, "data": {"audio": data}}
        raise ValueError(f"unknown worker action: {action}")

    run_worker_loop("Seed-VC", handler, request_queue, response_queue)
