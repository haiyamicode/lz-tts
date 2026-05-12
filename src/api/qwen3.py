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
from pydantic import BaseModel, Field

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
QWEN_DEFAULT_XVEC_ONLY = False
QWEN_DEFAULT_NON_STREAMING_MODE = True
QWEN_DEFAULT_EXPRESSIVENESS = 1.0
QWEN_DP_BUDGET_DEFAULT = True
ENABLE_DEEPFILTER_DEFAULT = True
ENABLE_SILENCE_TRIM_DEFAULT = True
SILENCE_TRIM_THRESHOLD_DB = -45.0
SILENCE_TRIM_RELATIVE_THRESHOLD_DB = -35.0
SILENCE_TRIM_PADDING_MS = 150
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
dp_budget_model: Optional[Any] = None
inference_lock = threading.Lock()
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


class DpBudgetSettings(BaseModel):
    enabled: bool = True
    preload: bool = True
    checkpoint: str = "data/lzspeech-gm/glow_tts.pt"
    device: str = "cuda"
    language: str = "multilingual"
    noise_scale: float = 0.8
    semantic_guidance_scale: float = 1.7
    length_scale: float = 1.0
    token_rate: float = 12.0
    samples: int = 32
    upper_quantile: float = 0.90
    min_margin: float = 1.0
    max_margin: float = 1.35
    min_extra_tokens: int = 0
    max_extra_tokens: int = 72
    bert_model: str = "distilbert-base-multilingual-cased"
    fusion_weight: float = 0.5
    language_profiles: dict[str, dict[str, float | int]] = Field(default_factory=dict)


class QwenSettings(BaseModel):
    preload: bool = True
    model: str = QWEN_DEFAULT_MODEL
    device: str = "cuda"
    dtype: str = QWEN_DEFAULT_DTYPE
    warmup: bool = True
    attn: str = "sdpa"
    max_seq_len: int = 2048
    language: str = QWEN_DEFAULT_LANGUAGE
    max_new_tokens: int = QWEN_DEFAULT_MAX_NEW_TOKENS
    xvec_only: bool = QWEN_DEFAULT_XVEC_ONLY
    non_streaming_mode: bool = QWEN_DEFAULT_NON_STREAMING_MODE
    temperature: float = QWEN_DEFAULT_TEMPERATURE
    top_k: int = QWEN_DEFAULT_TOP_K
    top_p: float = QWEN_DEFAULT_TOP_P
    repetition_penalty: float = QWEN_DEFAULT_REPETITION_PENALTY
    dp_budget: DpBudgetSettings = Field(default_factory=DpBudgetSettings)


_qwen_settings = QwenSettings()


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
    }


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
        return ResolvedQwenLanguage("Auto", "multilingual")

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
    if not requested or requested.lower() == "auto":
        return ResolvedQwenLanguage("Auto", "multilingual")

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
    if language_code is not None and language_code.strip():
        return resolve_qwen_language_code(language_code)

    requested = (language or _qwen_settings.language).strip()
    if not requested or requested.lower() == "auto":
        return detect_qwen_language(text)

    qwen_language = normalize_qwen_language(requested)
    language_base = QWEN_NAME_TO_CODE.get(qwen_language.lower()) or _locale_base(requested)
    dp_language = QWEN_LANGUAGE_LOCALES.get(language_base, "multilingual")
    return ResolvedQwenLanguage(qwen_language, dp_language)


def get_model() -> Any:
    global model
    if model is None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        model_name = _qwen_settings.model
        device = _qwen_settings.device
        dtype_name = _qwen_settings.dtype
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        attn_implementation = _qwen_settings.attn
        max_seq_len = _qwen_settings.max_seq_len
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
        if _qwen_settings.warmup and hasattr(model, "_warmup"):
            print("Capturing CUDA graphs...", file=sys.stderr, flush=True)
            model._warmup(prefill_len=100)
        print(f"FasterQwen3TTS loaded. Sample rate: {model.sample_rate}", file=sys.stderr, flush=True)
    return model


def preload_model() -> None:
    get_model()


def get_dp_budget_model() -> Any:
    global dp_budget_model
    if dp_budget_model is None:
        from src.qwen_dp_budget import DpBudgetConfig, QwenDpBudget

        print("Loading Qwen DP budget model...", file=sys.stderr, flush=True)
        dp_settings = _qwen_settings.dp_budget
        dp_budget_model = QwenDpBudget(
            DpBudgetConfig(
                checkpoint=Path(dp_settings.checkpoint),
                device=dp_settings.device,
                language=dp_settings.language,
                noise_scale=dp_settings.noise_scale,
                semantic_guidance_scale=dp_settings.semantic_guidance_scale,
                length_scale=dp_settings.length_scale,
                token_rate=dp_settings.token_rate,
                samples=dp_settings.samples,
                upper_quantile=dp_settings.upper_quantile,
                min_margin=dp_settings.min_margin,
                max_margin=dp_settings.max_margin,
                min_extra_tokens=dp_settings.min_extra_tokens,
                max_extra_tokens=dp_settings.max_extra_tokens,
                bert_model=dp_settings.bert_model,
                fusion_weight=dp_settings.fusion_weight,
                language_profiles=dp_settings.language_profiles,
            )
        )
        dp_budget_model.load()
        print("Qwen DP budget model ready.", file=sys.stderr, flush=True)
    return dp_budget_model


def predict_dp_budget(text: str, language: Optional[str] = None) -> dict[str, Any]:
    return get_dp_budget_model().predict(text, language=language)


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


@router.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")

    try:
        sample_path = download_and_cache(req.voice_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to download voice sample: {e}")

    m = get_model()
    xvec_only = req.xvec_only if req.xvec_only is not None else _qwen_settings.xvec_only
    try:
        prompt_text = (
            req.voice_text.strip()
            if req.voice_text and req.voice_text.strip()
            else "" if xvec_only else transcribe_voice_sample(sample_path)
        )
    except Exception as e:
        raise HTTPException(400, f"Failed to transcribe voice sample: {e}")

    resolved_language = resolve_qwen_language(req.text, req.language, req.language_code)
    language = resolved_language.qwen_language
    expressiveness_level, expressiveness_config = resolve_expressiveness(req.expressiveness)
    dp_budget_enabled = (
        req.dp_budget
        if req.dp_budget is not None
        else _qwen_settings.dp_budget.enabled
    )
    dp_budget_info = None
    if req.max_new_tokens is not None:
        max_new_tokens = req.max_new_tokens
    elif dp_budget_enabled:
        try:
            dp_budget_info = predict_dp_budget(req.text, language=resolved_language.dp_language)
            max_new_tokens = int(dp_budget_info["max_tokens"])
        except Exception as e:
            print(f"DP budget failed; falling back to default max_new_tokens: {e}", file=sys.stderr, flush=True)
            max_new_tokens = _qwen_settings.max_new_tokens
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
        f"dp_language={resolved_language.dp_language!r} "
        f"xvec_only={xvec_only} "
        f"non_streaming_mode={non_streaming_mode} "
        f"expressiveness={req.expressiveness if req.expressiveness is not None else QWEN_DEFAULT_EXPRESSIVENESS} "
        f"expressiveness_level={expressiveness_level} "
        f"temperature={temperature} "
        f"top_k={top_k} "
        f"top_p={top_p} "
        f"repetition_penalty={repetition_penalty} "
        f"max_new_tokens={max_new_tokens} "
        f"dp_budget={dp_budget_enabled} "
        f"dp_budget_info={json.dumps(dp_budget_info, ensure_ascii=False) if dp_budget_info else None} "
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

    cap_seconds = max_new_tokens / 12.0
    hit_token_cap = raw_audio_seconds >= max(0.0, cap_seconds - 0.25)
    print(
        "synthesize response "
        f"text_hash={text_hash} "
        f"raw_audio_seconds={raw_audio_seconds:.2f} "
        f"audio_seconds={wav_data.size / sample_rate:.2f} "
        f"cap_seconds={cap_seconds:.2f} "
        f"hit_token_cap={hit_token_cap} "
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
