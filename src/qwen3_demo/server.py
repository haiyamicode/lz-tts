#!/usr/bin/env python3
"""
Faster Qwen3-TTS Demo Server

Usage:
    python demo/server.py
    python demo/server.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 7860
    python demo/server.py --no-preload  # skip startup model load
"""

import argparse
import asyncio
import base64
from collections import OrderedDict
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Allow running from any directory.
REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

try:
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError:
    print("Error: faster_qwen3_tts not found.")
    print("Install with:  pip install -e .  (from the repo root)")
    sys.exit(1)

from src.api import qwen3 as shared_qwen3
from nano_parakeet import from_pretrained as _parakeet_from_pretrained


EMBEDDED_IN_LZ_TTS = os.environ.get("LZ_TTS_EMBEDDED_DEMO", "0") == "1"
EMBEDDED_MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")

_ALL_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

_active_models_env = os.environ.get("ACTIVE_MODELS", "")
if EMBEDDED_IN_LZ_TTS:
    AVAILABLE_MODELS = [EMBEDDED_MODEL_ID]
elif _active_models_env:
    _allowed = {m.strip() for m in _active_models_env.split(",") if m.strip()}
    AVAILABLE_MODELS = [m for m in _ALL_MODELS if m in _allowed]
else:
    AVAILABLE_MODELS = list(_ALL_MODELS)

BASE_DIR = Path(__file__).resolve().parent
# Assets that need to be downloaded at runtime go to a writable directory.
# /app is read-only in HF Spaces; fall back to /tmp.
_ASSET_DIR = Path(os.environ.get("ASSET_DIR", "/tmp/faster-qwen3-tts-assets"))
PRESET_TRANSCRIPTS = _ASSET_DIR / "samples" / "parity" / "icl_transcripts.txt"
PRESET_REFS = [] if EMBEDDED_IN_LZ_TTS else [
    ("ref_audio_3", _ASSET_DIR / "ref_audio_3.wav", "Clone 1"),
    ("ref_audio_2", _ASSET_DIR / "ref_audio_2.wav", "Clone 2"),
    ("ref_audio", _ASSET_DIR / "ref_audio.wav", "Clone 3"),
]

_GITHUB_RAW = "https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main"
_PRESET_REMOTE = {
    "ref_audio":   f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_2": f"{_GITHUB_RAW}/ref_audio_2.wav",
    "ref_audio_3": f"{_GITHUB_RAW}/ref_audio_3.wav",
}
_TRANSCRIPT_REMOTE = f"{_GITHUB_RAW}/samples/parity/icl_transcripts.txt"


def _fetch_preset_assets() -> None:
    """Download preset wav files and transcripts from GitHub if not present locally."""
    if EMBEDDED_IN_LZ_TTS:
        return
    import urllib.request
    _ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_TRANSCRIPTS.parent.mkdir(parents=True, exist_ok=True)
    if not PRESET_TRANSCRIPTS.exists():
        try:
            urllib.request.urlretrieve(_TRANSCRIPT_REMOTE, PRESET_TRANSCRIPTS)
        except Exception as e:
            print(f"Warning: could not fetch transcripts: {e}")
    for key, path, _ in PRESET_REFS:
        if not path.exists() and key in _PRESET_REMOTE:
            try:
                urllib.request.urlretrieve(_PRESET_REMOTE[key], path)
                print(f"Downloaded {path.name}")
            except Exception as e:
                print(f"Warning: could not fetch {key}: {e}")

_preset_refs: dict[str, dict] = {}


def _load_preset_transcripts() -> dict[str, str]:
    if not PRESET_TRANSCRIPTS.exists():
        return {}
    transcripts = {}
    for line in PRESET_TRANSCRIPTS.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key_part, text = line.split(":", 1)
        key = key_part.split("(")[0].strip()
        transcripts[key] = text.strip()
    return transcripts


def _load_preset_refs() -> None:
    transcripts = _load_preset_transcripts()
    for key, path, label in PRESET_REFS:
        if not path.exists():
            continue
        content = path.read_bytes()
        cached_path = _get_cached_ref_path(content)
        _preset_refs[key] = {
            "id": key,
            "label": label,
            "filename": path.name,
            "path": cached_path,
            "ref_text": transcripts.get(key, ""),
            "audio_b64": base64.b64encode(content).decode(),
        }


def _prime_preset_voice_cache(model: FasterQwen3TTS) -> None:
    if not _preset_refs:
        return
    for preset in _preset_refs.values():
        ref_path = preset["path"]
        ref_text = preset["ref_text"]
        for xvec_only in (True, False):
            try:
                model._prepare_generation(
                    text="Hello.",
                    ref_audio=ref_path,
                    ref_text=ref_text,
                    language="English",
                    xvec_only=xvec_only,
                    non_streaming_mode=True,
                )
            except Exception:
                continue

app = FastAPI(title="Faster Qwen3-TTS Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: OrderedDict[str, FasterQwen3TTS] = OrderedDict()
_model_cache_max: int = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
_active_model_name: str | None = None
_loading = False
_ref_cache: dict[str, str] = {}
_ref_cache_lock = threading.Lock()
_parakeet = None
_generation_lock = asyncio.Lock()
_generation_waiters: int = 0  # requests waiting for or holding the generation lock
_dp_budget_defaults: dict[str, object] = {
    "enabled": shared_qwen3.env_bool("QWEN_DP_BUDGET", True) if EMBEDDED_IN_LZ_TTS else False,
    "checkpoint": str(REPO_DIR / "data/lzspeech-multilingual-bert/lzspeech-multilingual-bert-189.ckpt"),
    "device": "cpu",
    "language": "multilingual",
    "noise_scale": 0.8,
    "length_scale": 1.0,
    "token_rate": 12.0,
    "samples": 32,
    "upper_quantile": 0.90,
    "min_margin": 1.0,
    "max_margin": 1.0,
    "min_extra_tokens": 0,
    "max_extra_tokens": 24,
    "language_profiles": {},
}

# Guard against inputs that would overflow the static KV cache (max_seq_len=2048).
# At ~3-4 chars/token for English the overhead of system/ref tokens leaves room
# for roughly 1000 chars before we approach the limit.
MAX_TEXT_CHARS = 1000
# ~10 MB covers 1 minute of 44.1 kHz stereo 16-bit WAV.
MAX_AUDIO_BYTES = 10 * 1024 * 1024
_AUDIO_TOO_LARGE_MSG = (
    "Audio file too large ({size_mb:.1f} MB). "
    "Voice cloning works best with short clips under 1 minute — please upload a shorter recording."
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_wav_b64(audio: np.ndarray, sr: int) -> str:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64


def _concat_audio(audio_list) -> np.ndarray:
    if isinstance(audio_list, np.ndarray):
        return audio_list.astype(np.float32).squeeze()
    parts = [np.array(a, dtype=np.float32).squeeze() for a in audio_list if len(a) > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _postprocess_audio(audio: np.ndarray, sr: int, enabled: bool) -> tuple[np.ndarray, dict, float]:
    if not enabled:
        return audio, {"enabled": False}, 0.0
    t0 = time.perf_counter()
    processed, info = shared_qwen3.postprocess_audio(audio, sr)
    info["enabled"] = True
    return processed, info, (time.perf_counter() - t0) * 1000


def _get_cached_ref_path(content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()
    with _ref_cache_lock:
        cached = _ref_cache.get(digest)
        if cached and os.path.exists(cached):
            return cached
        tmp_dir = Path(tempfile.gettempdir())
        path = tmp_dir / f"faster_qwen3_tts_ref_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        _ref_cache[digest] = str(path)
        return str(path)


def _default_non_streaming_mode_for_mode(mode: str) -> bool:
    return mode != "voice_clone"


def _predict_dp_budget(text: str, language: str | None = None) -> dict:
    if EMBEDDED_IN_LZ_TTS:
        return shared_qwen3.predict_dp_budget(text, language=language)
    from src.qwen_dp_budget import DpBudgetConfig, QwenDpBudget

    config = DpBudgetConfig(
        checkpoint=Path(str(_dp_budget_defaults["checkpoint"])),
        device=str(_dp_budget_defaults["device"]),
        language=str(_dp_budget_defaults["language"]),
        noise_scale=float(_dp_budget_defaults["noise_scale"]),
        length_scale=float(_dp_budget_defaults["length_scale"]),
        token_rate=float(_dp_budget_defaults["token_rate"]),
        samples=int(_dp_budget_defaults["samples"]),
        upper_quantile=float(_dp_budget_defaults["upper_quantile"]),
        min_margin=float(_dp_budget_defaults["min_margin"]),
        max_margin=float(_dp_budget_defaults["max_margin"]),
        min_extra_tokens=int(_dp_budget_defaults["min_extra_tokens"]),
        max_extra_tokens=int(_dp_budget_defaults["max_extra_tokens"]),
        language_profiles=dict(_dp_budget_defaults["language_profiles"]),
    )
    return QwenDpBudget(config).predict(text, language=language)


def _max_new_tokens_for_request(
    text: str,
    mode: str,
    use_dp_budget: bool,
    language: str | None = None,
) -> tuple[int, dict | None, float | None]:
    if mode == "voice_clone" and use_dp_budget:
        t0 = time.perf_counter()
        budget = _predict_dp_budget(text, language=language)
        return int(budget["max_tokens"]), budget, (time.perf_counter() - t0) * 1000
    return 360, None, None


# ─── Routes ───────────────────────────────────────────────────────────────────

_fetch_preset_assets()
_load_preset_refs()

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe reference audio using nano-parakeet."""
    parakeet = shared_qwen3.get_parakeet_model() if EMBEDDED_IN_LZ_TTS else _parakeet
    if parakeet is None:
        raise HTTPException(status_code=503, detail="Transcription model not loaded")

    content = await audio.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
        )

    def run():
        wav, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav)
        if sr != 16000:
            wav_t = torchaudio.functional.resample(wav_t.unsqueeze(0), sr, 16000).squeeze(0)
        with torch.inference_mode():
            return parakeet.transcribe(wav_t.cuda())

    text = await asyncio.to_thread(run)
    return {"text": text}


@app.get("/status")
async def get_status():
    speakers = []
    model_type = None
    active = shared_qwen3.model if EMBEDDED_IN_LZ_TTS else _model_cache.get(_active_model_name) if _active_model_name else None
    if active is not None:
        try:
            model_type = active.model.model.tts_model_type
            speakers = active.model.get_supported_speakers() or []
        except Exception:
            speakers = []
    return {
        "loaded": active is not None,
        "model": EMBEDDED_MODEL_ID if EMBEDDED_IN_LZ_TTS and active is not None else _active_model_name,
        "loading": shared_qwen3.model_status()["model_loading"] if EMBEDDED_IN_LZ_TTS else _loading,
        "available_models": AVAILABLE_MODELS,
        "model_type": model_type,
        "speakers": speakers,
        "transcription_available": _parakeet is not None,
        "preset_refs": [
            {"id": p["id"], "label": p["label"], "ref_text": p["ref_text"]}
            for p in _preset_refs.values()
        ],
        "qwen_defaults": shared_qwen3.demo_defaults() if EMBEDDED_IN_LZ_TTS else {
            "language": "Auto",
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.03,
            "xvec_only": False,
            "non_streaming_mode": True,
            "dp_budget": bool(_dp_budget_defaults["enabled"]),
        },
        "dp_budget_default": bool(_dp_budget_defaults["enabled"]),
        "queue_depth": _generation_waiters,
        "cached_models": list(_model_cache.keys()),
    }


@app.get("/preset_ref/{preset_id}")
async def get_preset_ref(preset_id: str):
    preset = _preset_refs.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {
        "id": preset["id"],
        "label": preset["label"],
        "filename": preset["filename"],
        "ref_text": preset["ref_text"],
        "audio_b64": preset["audio_b64"],
    }


@app.post("/load")
async def load_model(model_id: str = Form(...)):
    global _active_model_name, _loading

    if EMBEDDED_IN_LZ_TTS:
        if model_id != EMBEDDED_MODEL_ID:
            raise HTTPException(status_code=400, detail=f"This server is running {EMBEDDED_MODEL_ID}")
        await asyncio.to_thread(shared_qwen3.get_model)
        return {"status": "already_loaded", "model": EMBEDDED_MODEL_ID}

    # Already in cache — instant switch, no GPU work needed
    if model_id in _model_cache:
        _active_model_name = model_id
        _model_cache.move_to_end(model_id)
        return {"status": "already_loaded", "model": model_id}

    _loading = True

    def _do_load():
        global _active_model_name, _loading
        try:
            if len(_model_cache) >= _model_cache_max:
                evicted, old_model = _model_cache.popitem(last=False)
                del old_model
                print(f"Model cache full — evicted: {evicted}")
            new_model = FasterQwen3TTS.from_pretrained(
                model_id,
                device="cuda",
                dtype=torch.bfloat16,
            )
            print("Capturing CUDA graphs…")
            new_model._warmup(prefill_len=100)
            _model_cache[model_id] = new_model
            _model_cache.move_to_end(model_id)
            _active_model_name = model_id
            _prime_preset_voice_cache(new_model)
            print("CUDA graphs captured — model ready.")
        finally:
            _loading = False

    # Hold the generation lock while loading to prevent OOM from concurrent inference
    async with _generation_lock:
        await asyncio.to_thread(_do_load)
    return {"status": "loaded", "model": model_id}


@app.post("/generate/stream")
async def generate_stream(
    text: str = Form(...),
    language: str = Form("Auto"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    xvec_only: bool = Form(False),
    chunk_size: int = Form(8),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.03),
    non_streaming_mode: bool | None = Form(None),
    use_dp_budget: bool = Form(False),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    if not EMBEDDED_IN_LZ_TTS and (not _active_model_name or _active_model_name not in _model_cache):
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )

    tmp_path = None
    tmp_is_cached = False

    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
            )
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    if non_streaming_mode is None:
        non_streaming_mode = _default_non_streaming_mode_for_mode(mode)

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def run_generation():
        try:
            # Resolve the model after the generation lock is held so we always
            # use the currently active model, not a stale reference captured
            # before a concurrent /load request changed the active model.
            model = shared_qwen3.get_model() if EMBEDDED_IN_LZ_TTS else _model_cache.get(_active_model_name)
            if model is None:
                raise RuntimeError("No model loaded. Please load a model first.")

            t0 = time.perf_counter()
            total_audio_s = 0.0
            voice_clone_ms = 0.0
            resolved_language = shared_qwen3.resolve_qwen_language(text, language)
            max_new_tokens, dp_budget, dp_budget_ms = _max_new_tokens_for_request(
                text,
                mode,
                use_dp_budget,
                language=resolved_language.dp_language,
            )

            if mode == "voice_clone":
                gen = model.generate_voice_clone_streaming(
                    text=text,
                    language=resolved_language.qwen_language,
                    ref_audio=tmp_path,
                    ref_text=ref_text,
                    xvec_only=xvec_only,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                )
            elif mode == "custom":
                if not speaker:
                    raise ValueError("Speaker ID is required for custom voice")
                gen = model.generate_custom_voice_streaming(
                    text=text,
                    speaker=speaker,
                    language=resolved_language.qwen_language,
                    instruct=instruct,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                )
            else:
                gen = model.generate_voice_design_streaming(
                    text=text,
                    instruct=instruct,
                    language=resolved_language.qwen_language,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                )

            # Use timing data from the generator itself (measured after voice-clone
            # encoding, so TTFA and RTF reflect pure LLM generation latency).
            ttfa_ms = None
            total_gen_ms = 0.0

            # Prime generator to capture wall-clock time to first chunk
            first_audio = next(gen, None)
            if first_audio is not None:
                audio_chunk, sr, timing = first_audio
                wall_first_ms = (time.perf_counter() - t0) * 1000
                model_ms = timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
                voice_clone_ms = max(0.0, wall_first_ms - model_ms - (dp_budget_ms or 0.0))
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms

                audio_chunk = _concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

                audio_b64 = _to_wav_b64(audio_chunk, sr)
                payload = {
                    "type": "chunk",
                    "audio_b64": audio_b64,
                    "sample_rate": sr,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }
                if dp_budget is not None:
                    payload["dp_budget"] = dp_budget
                    payload["dp_budget_ms"] = round(dp_budget_ms or 0)
                    payload["max_new_tokens"] = max_new_tokens
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            for audio_chunk, sr, timing in gen:
                # prefill_ms is non-zero only on the first chunk
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms  # already in ms

                audio_chunk = _concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

                audio_b64 = _to_wav_b64(audio_chunk, sr)
                payload = {
                    "type": "chunk",
                    "audio_b64": audio_b64,
                    "sample_rate": sr,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }
                if dp_budget is not None:
                    payload["dp_budget"] = dp_budget
                    payload["dp_budget_ms"] = round(dp_budget_ms or 0)
                    payload["max_new_tokens"] = max_new_tokens
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
            done_payload = {
                "type": "done",
                "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
                "voice_clone_ms": round(voice_clone_ms),
                "rtf": round(rtf, 3),
                "total_audio_s": round(total_audio_s, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            }
            if dp_budget is not None:
                done_payload["dp_budget"] = dp_budget
                done_payload["dp_budget_ms"] = round(dp_budget_ms or 0)
                done_payload["max_new_tokens"] = max_new_tokens
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(done_payload))

        except Exception as e:
            import traceback
            err = {"type": "error", "message": str(e), "detail": traceback.format_exc()}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            if tmp_path and os.path.exists(tmp_path) and not tmp_is_cached:
                os.unlink(tmp_path)

    async def sse():
        global _generation_waiters
        lock_acquired = False
        _generation_waiters += 1
        people_ahead = _generation_waiters - 1 + (1 if _generation_lock.locked() else 0)
        try:
            if people_ahead > 0:
                yield f"data: {json.dumps({'type': 'queued', 'position': people_ahead})}\n\n"

            await _generation_lock.acquire()
            lock_acquired = True
            _generation_waiters -= 1

            thread = threading.Thread(target=run_generation, daemon=True)
            thread.start()

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if lock_acquired:
                _generation_lock.release()
            else:
                _generation_waiters -= 1

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )




@app.post("/generate")
async def generate_non_streaming(
    text: str = Form(...),
    language: str = Form("Auto"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    xvec_only: bool = Form(False),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.03),
    non_streaming_mode: bool | None = Form(None),
    use_dp_budget: bool = Form(False),
    use_postprocess: bool = Form(False),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    if not EMBEDDED_IN_LZ_TTS and (not _active_model_name or _active_model_name not in _model_cache):
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )

    tmp_path = None
    tmp_is_cached = False

    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
            )
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    if non_streaming_mode is None:
        non_streaming_mode = _default_non_streaming_mode_for_mode(mode)

    def run():
        # Resolve the model after the generation lock is held.
        model = shared_qwen3.get_model() if EMBEDDED_IN_LZ_TTS else _model_cache.get(_active_model_name)
        if model is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        t0 = time.perf_counter()
        resolved_language = shared_qwen3.resolve_qwen_language(text, language)
        max_new_tokens, dp_budget, dp_budget_ms = _max_new_tokens_for_request(
            text,
            mode,
            use_dp_budget,
            language=resolved_language.dp_language,
        )
        if mode == "voice_clone":
            audio_list, sr = model.generate_voice_clone(
                text=text,
                language=resolved_language.qwen_language,
                ref_audio=tmp_path,
                ref_text=ref_text,
                xvec_only=xvec_only,
                non_streaming_mode=non_streaming_mode,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        elif mode == "custom":
            if not speaker:
                raise ValueError("Speaker ID is required for custom voice")
            audio_list, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=resolved_language.qwen_language,
                instruct=instruct,
                non_streaming_mode=non_streaming_mode,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        else:
            audio_list, sr = model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=resolved_language.qwen_language,
                non_streaming_mode=non_streaming_mode,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        audio = _concat_audio(audio_list)
        raw_dur = len(audio) / sr
        audio, postprocess_info, postprocess_ms = _postprocess_audio(audio, sr, use_postprocess)
        dur = len(audio) / sr
        elapsed = time.perf_counter() - t0
        return audio, sr, elapsed, raw_dur, dur, max_new_tokens, dp_budget, dp_budget_ms, postprocess_info, postprocess_ms

    global _generation_waiters
    _generation_waiters += 1
    lock_acquired = False
    try:
        await _generation_lock.acquire()
        lock_acquired = True
        _generation_waiters -= 1
        audio, sr, elapsed, raw_dur, dur, max_new_tokens, dp_budget, dp_budget_ms, postprocess_info, postprocess_ms = await asyncio.to_thread(run)
        rtf = dur / elapsed if elapsed > 0 else 0.0
        return JSONResponse({
            "audio_b64": _to_wav_b64(audio, sr),
            "sample_rate": sr,
            "dp_budget": dp_budget,
            "max_new_tokens": max_new_tokens,
            "postprocess": postprocess_info,
            "metrics": {
                "total_ms": round(elapsed * 1000),
                "raw_audio_duration_s": round(raw_dur, 3),
                "audio_duration_s": round(dur, 3),
                "rtf": round(rtf, 3),
                "dp_budget_ms": round(dp_budget_ms or 0) if dp_budget is not None else None,
                "postprocess_ms": round(postprocess_ms) if use_postprocess else None,
            },
        })
    finally:
        if lock_acquired:
            _generation_lock.release()
        else:
            _generation_waiters -= 1
        if tmp_path and os.path.exists(tmp_path) and not tmp_is_cached:
            os.unlink(tmp_path)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Faster Qwen3-TTS Demo Server")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model to preload at startup (default: 1.7B-Base)",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip model loading at startup (load via UI instead)",
    )
    parser.add_argument(
        "--dp-budget",
        action="store_true",
        help="Enable DP-derived max_new_tokens budget by default for voice cloning.",
    )
    parser.add_argument("--dp-checkpoint", default=str(REPO_DIR / "data/lzspeech-multilingual-bert/lzspeech-multilingual-bert-189.ckpt"))
    parser.add_argument("--dp-device", default="cpu")
    parser.add_argument("--dp-language", default="multilingual")
    parser.add_argument("--dp-noise-scale", type=float, default=0.8)
    parser.add_argument("--dp-length-scale", type=float, default=1.0)
    parser.add_argument("--token-rate", type=float, default=12.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--upper-quantile", type=float, default=0.90)
    parser.add_argument("--min-margin", type=float, default=1.0)
    parser.add_argument("--max-margin", type=float, default=1.0)
    parser.add_argument("--min-extra-tokens", type=int, default=0)
    parser.add_argument("--max-extra-tokens", type=int, default=24)
    parser.add_argument("--dp-language-profiles", default=os.environ.get("QWEN_DP_BUDGET_LANGUAGE_PROFILES", ""))
    args = parser.parse_args()

    from src.qwen_dp_budget import parse_language_profiles

    _dp_budget_defaults.update({
        "enabled": bool(args.dp_budget),
        "checkpoint": args.dp_checkpoint,
        "device": args.dp_device,
        "language": args.dp_language,
        "noise_scale": args.dp_noise_scale,
        "length_scale": args.dp_length_scale,
        "token_rate": args.token_rate,
        "samples": args.samples,
        "upper_quantile": args.upper_quantile,
        "min_margin": args.min_margin,
        "max_margin": args.max_margin,
        "min_extra_tokens": args.min_extra_tokens,
        "max_extra_tokens": args.max_extra_tokens,
        "language_profiles": parse_language_profiles(args.dp_language_profiles),
    })

    if not args.no_preload:
        global _active_model_name, _parakeet
        print(f"Loading model: {args.model}")
        _startup_model = FasterQwen3TTS.from_pretrained(
            args.model,
            device="cuda",
            dtype=torch.bfloat16,
        )
        print("Capturing CUDA graphs…")
        _startup_model._warmup(prefill_len=100)
        _model_cache[args.model] = _startup_model
        _active_model_name = args.model
        _prime_preset_voice_cache(_startup_model)
        print("TTS model ready.")

        print("Loading transcription model (nano-parakeet)…")
        _parakeet = _parakeet_from_pretrained(device="cuda")
        print("Transcription model ready.")

        print(f"Ready. Open http://localhost:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
