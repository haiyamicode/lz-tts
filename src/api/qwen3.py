import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

CACHE_DIR = Path("cache/voice_samples")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

QWEN_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN_DEFAULT_DTYPE = "bfloat16"
QWEN_DEFAULT_LANGUAGE = "English"
QWEN_DEFAULT_MAX_NEW_TOKENS = 360
QWEN_DEFAULT_TEMPERATURE = 0.9
QWEN_DEFAULT_TOP_K = 50
QWEN_DEFAULT_TOP_P = 1.0
QWEN_DEFAULT_REPETITION_PENALTY = 1.03
QWEN_DEFAULT_XVEC_ONLY = False
QWEN_DEFAULT_NON_STREAMING_MODE = True
QWEN_DEFAULT_EXPRESSIVENESS = 1.0
ENABLE_DEEPFILTER_DEFAULT = True
ENABLE_SILENCE_TRIM_DEFAULT = True
SILENCE_TRIM_THRESHOLD_DB = -45.0
SILENCE_TRIM_RELATIVE_THRESHOLD_DB = -35.0
SILENCE_TRIM_PADDING_MS = 80
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
deepfilter_model: Optional[Any] = None
deepfilter_state: Optional[Any] = None
inference_lock = threading.Lock()


@router.get("/health")
async def healthcheck():
    return {"status": "ok", "backend": "faster-qwen3-tts", "model_loaded": model is not None}


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


def get_model() -> Any:
    global model
    if model is None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        model_name = os.environ.get("QWEN_TTS_MODEL", QWEN_DEFAULT_MODEL)
        device = os.environ.get("QWEN_TTS_DEVICE", "cuda")
        dtype_name = os.environ.get("QWEN_TTS_DTYPE", QWEN_DEFAULT_DTYPE)
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        attn_implementation = os.environ.get("QWEN_TTS_ATTN", "sdpa")
        max_seq_len = int(os.environ.get("QWEN_TTS_MAX_SEQ_LEN", "2048"))
        print(
            "Loading FasterQwen3TTS "
            f"model={model_name} device={device} dtype={dtype_name} "
            f"attn={attn_implementation} max_seq_len={max_seq_len}...",
            file=sys.stderr,
            flush=True,
        )
        model = FasterQwen3TTS.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            max_seq_len=max_seq_len,
        )
        if env_bool("QWEN_TTS_WARMUP", True) and hasattr(model, "_warmup"):
            print("Capturing CUDA graphs...", file=sys.stderr, flush=True)
            model._warmup(prefill_len=100)
        print(f"FasterQwen3TTS loaded. Sample rate: {model.sample_rate}", file=sys.stderr, flush=True)
    return model


def get_parakeet_model() -> Any:
    global parakeet_model
    if parakeet_model is None:
        from nano_parakeet import from_pretrained as parakeet_from_pretrained

        print("Loading transcription model (nano-parakeet)...", file=sys.stderr, flush=True)
        parakeet_model = parakeet_from_pretrained(device="cuda")
        print("Transcription model ready.", file=sys.stderr, flush=True)
    return parakeet_model


def get_deepfilter() -> tuple[Any, Any]:
    global deepfilter_model, deepfilter_state
    if deepfilter_model is None or deepfilter_state is None:
        ensure_torchaudio_backend_compat()
        from df.enhance import init_df

        print("Loading DeepFilterNet postprocessor...", file=sys.stderr, flush=True)
        deepfilter_model, deepfilter_state, _ = init_df(log_level="ERROR", log_file=None)
        print(
            f"DeepFilterNet ready. Sample rate: {deepfilter_state.sr()}",
            file=sys.stderr,
            flush=True,
        )
    return deepfilter_model, deepfilter_state


def ensure_torchaudio_backend_compat() -> None:
    """Provide the torchaudio.backend.common import removed in newer torchaudio."""
    if "torchaudio.backend.common" in sys.modules:
        return

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = "UNKNOWN"

    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules.setdefault("torchaudio.backend.common", common)


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

    import torch
    import torchaudio

    parakeet = get_parakeet_model()
    wav, sr = sf.read(audio_file, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav)
    if sr != 16000:
        wav_t = torchaudio.functional.resample(wav_t.unsqueeze(0), sr, 16000).squeeze(0)

    text = parakeet.transcribe(wav_t.cuda()).strip()
    if not text:
        raise RuntimeError("nano-parakeet transcript completed with empty text")
    transcript_file.write_text(text, encoding="utf-8")
    return text


def trim_silence(wav_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float, float]:
    if wav_data.size == 0:
        return wav_data, 0.0, 0.0

    frame_size = max(1, int(sample_rate * 0.02))
    hop_size = max(1, int(sample_rate * 0.01))
    absolute_threshold = 10 ** (SILENCE_TRIM_THRESHOLD_DB / 20)
    padded = np.pad(wav_data, (0, max(0, frame_size - wav_data.size % hop_size)))
    starts = np.arange(0, max(1, padded.size - frame_size + 1), hop_size)
    if starts.size == 0:
        return wav_data, 0.0, 0.0

    frames = np.stack([padded[start : start + frame_size] for start in starts])
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = max(absolute_threshold, float(rms.max()) * 10 ** (SILENCE_TRIM_RELATIVE_THRESHOLD_DB / 20))
    voiced = np.flatnonzero(rms > threshold)
    if voiced.size == 0:
        return wav_data, 0.0, 0.0

    pad = int(sample_rate * SILENCE_TRIM_PADDING_MS / 1000)
    start = max(0, int(starts[voiced[0]]) - pad)
    end = min(wav_data.size, int(starts[voiced[-1]]) + frame_size + pad)
    trimmed = wav_data[start:end]
    return trimmed, start / sample_rate, (wav_data.size - end) / sample_rate


def deepfilter_audio(wav_data: np.ndarray, sample_rate: int) -> np.ndarray:
    ensure_torchaudio_backend_compat()
    import torch
    import torchaudio
    from df.enhance import enhance

    df_model, df_state = get_deepfilter()
    df_sample_rate = int(df_state.sr())
    wav_tensor = torch.from_numpy(wav_data.astype(np.float32, copy=False)).unsqueeze(0)
    if sample_rate != df_sample_rate:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sample_rate, df_sample_rate)

    enhanced = enhance(df_model, df_state, wav_tensor, pad=True)
    if sample_rate != df_sample_rate:
        enhanced = torchaudio.functional.resample(enhanced, df_sample_rate, sample_rate)
    return enhanced.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def postprocess_audio(wav_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {
        "deepfilter": False,
        "trim": False,
        "trim_head_seconds": 0.0,
        "trim_tail_seconds": 0.0,
    }

    if env_bool("ENABLE_DEEPFILTER", ENABLE_DEEPFILTER_DEFAULT):
        try:
            wav_data = deepfilter_audio(wav_data, sample_rate)
            info["deepfilter"] = True
        except Exception as e:
            info["deepfilter_error"] = str(e)
            print(f"DeepFilterNet postprocessor failed; continuing without it: {e}", file=sys.stderr, flush=True)

    if env_bool("ENABLE_SILENCE_TRIM", ENABLE_SILENCE_TRIM_DEFAULT):
        wav_data, head_seconds, tail_seconds = trim_silence(wav_data, sample_rate)
        info.update(
            {
                "trim": True,
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
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    xvec_only: Optional[bool] = None
    non_streaming_mode: Optional[bool] = None
    expressiveness: Optional[float] = None


@router.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")

    try:
        sample_path = download_and_cache(req.voice_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to download voice sample: {e}")

    m = get_model()
    xvec_only = req.xvec_only if req.xvec_only is not None else env_bool("QWEN_TTS_XVEC_ONLY", QWEN_DEFAULT_XVEC_ONLY)
    try:
        prompt_text = (
            req.voice_text.strip()
            if req.voice_text and req.voice_text.strip()
            else "" if xvec_only else transcribe_voice_sample(sample_path)
        )
    except Exception as e:
        raise HTTPException(400, f"Failed to transcribe voice sample: {e}")

    language = req.language or os.environ.get("QWEN_TTS_LANGUAGE", QWEN_DEFAULT_LANGUAGE)
    expressiveness_level, expressiveness_config = resolve_expressiveness(req.expressiveness)
    max_new_tokens = (
        req.max_new_tokens
        if req.max_new_tokens is not None
        else int(os.environ.get("QWEN_TTS_MAX_NEW_TOKENS", str(QWEN_DEFAULT_MAX_NEW_TOKENS)))
    )
    temperature = (
        req.temperature
        if req.temperature is not None
        else expressiveness_config["temperature"]
        if req.expressiveness is not None
        else float(os.environ.get("QWEN_TTS_TEMPERATURE", str(expressiveness_config["temperature"])))
    )
    top_k = (
        req.top_k
        if req.top_k is not None
        else expressiveness_config["top_k"]
        if req.expressiveness is not None
        else int(os.environ.get("QWEN_TTS_TOP_K", str(expressiveness_config["top_k"])))
    )
    top_p = req.top_p if req.top_p is not None else float(os.environ.get("QWEN_TTS_TOP_P", str(QWEN_DEFAULT_TOP_P)))
    repetition_penalty = (
        req.repetition_penalty
        if req.repetition_penalty is not None
        else expressiveness_config["repetition_penalty"]
        if req.expressiveness is not None
        else float(os.environ.get("QWEN_TTS_REPETITION_PENALTY", str(QWEN_DEFAULT_REPETITION_PENALTY)))
    )
    non_streaming_mode = (
        req.non_streaming_mode
        if req.non_streaming_mode is not None
        else env_bool("QWEN_TTS_NON_STREAMING_MODE", QWEN_DEFAULT_NON_STREAMING_MODE)
    )

    text_hash = hashlib.sha256(req.text.encode()).hexdigest()[:12]
    prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]
    voice_hash = hashlib.sha256(req.voice_url.encode()).hexdigest()[:12]
    print(
        "synthesize request "
        "backend=faster-qwen3-tts "
        f"text_hash={text_hash} "
        f"text_len={len(req.text)} "
        f"prompt_hash={prompt_hash} "
        f"voice_hash={voice_hash} "
        f"voice_text_len={len(prompt_text)} "
        f"voice_text_source={'request' if req.voice_text and req.voice_text.strip() else 'none' if xvec_only else 'nano-parakeet'} "
        f"language={language!r} "
        f"xvec_only={xvec_only} "
        f"non_streaming_mode={non_streaming_mode} "
        f"expressiveness={req.expressiveness if req.expressiveness is not None else QWEN_DEFAULT_EXPRESSIVENESS} "
        f"expressiveness_level={expressiveness_level} "
        f"temperature={temperature} "
        f"top_k={top_k} "
        f"top_p={top_p} "
        f"repetition_penalty={repetition_penalty} "
        f"max_new_tokens={max_new_tokens} "
        f"speed_ignored={req.speed != 1.0} "
        f"prompt_preview={prompt_text[:180]!r} "
        f"preview={req.text[:240]!r}",
        file=sys.stderr,
        flush=True,
    )

    audio_list = None
    sample_rate = m.sample_rate
    postprocess_info: dict[str, Any] = {}
    with inference_lock:
        audio_list, sample_rate = m.generate_voice_clone(
            text=req.text,
            language=language,
            ref_audio=str(sample_path),
            ref_text=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            xvec_only=xvec_only,
            non_streaming_mode=non_streaming_mode,
            append_silence=True,
        )

        if not audio_list:
            raise HTTPException(500, "Model produced no output")

        wav_data = np.asarray(audio_list[0], dtype=np.float32).flatten()
        raw_audio_seconds = wav_data.size / sample_rate
        wav_data, postprocess_info = postprocess_audio(wav_data, sample_rate)

    print(
        "synthesize response "
        f"text_hash={text_hash} "
        f"raw_audio_seconds={raw_audio_seconds:.2f} "
        f"audio_seconds={wav_data.size / sample_rate:.2f} "
        f"deepfilter={postprocess_info.get('deepfilter')} "
        f"silence_trim={postprocess_info.get('trim')} "
        f"trim_head_seconds={postprocess_info.get('trim_head_seconds', 0.0):.2f} "
        f"trim_tail_seconds={postprocess_info.get('trim_tail_seconds', 0.0):.2f}",
        file=sys.stderr,
        flush=True,
    )
    mp3_bytes = wav_to_mp3(wav_data, sample_rate)

    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return StreamingResponse(io.BytesIO(mp3_bytes), media_type="audio/mpeg")
