"""Speech-to-text for the ``transcribe`` task, with Qwen3-ASR-0.6B.

The model runs in its own GPU process (``transcribe_worker_main``, see
model_workers.py), separate from the synthesis engines. The API decodes the input to 16 kHz mono and force-aligns the
transcript for word timestamps with the shared CTC aligner (server.py
``_align_words``), so no second alignment model is loaded.

Weights are the pre-baked ``data/qwen3-asr-0.6b`` snapshot (uploaded with
scripts/upload_data.py --model qwen3-asr-0.6b).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

from .language_id import LanguageInputError, decode_audio_16k_mono

SAMPLE_RATE = 16000
# Longest input decoded for one request. Longer audio is still transcribed in
# full: the model splits it into chunks internally.
MAX_SECONDS = 3600.0
DEFAULT_MODEL_DIR = "data/qwen3-asr-0.6b"

# ISO 639-1 (plus the few codes Qwen uses with a non-ISO name) -> the language
# names Qwen3-ASR accepts for ``language=``. Codes not listed are auto-detected.
_QWEN_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
_CODES_BY_NAME = {name: code for code, name in _QWEN_NAMES.items() if code != "tl"}


class TranscribeInputError(ValueError):
    """The transcribe request cannot be served (bad input, undecodable audio)."""


class TranscribeRequest(BaseModel):
    """Input for the ``transcribe`` operation (one audio clip)."""

    audio_url: str | None = None
    audio_base64: str | None = None
    # BCP-47 locale or language code ("vi", "vi-VN"); omitted means auto-detect.
    language: str | None = None
    # Also return word timestamps (force-aligned on the decoded audio).
    timestamps: bool = False


def qwen_language(language: str | None) -> str | None:
    """Qwen's language name for a locale or code, or None to let the model detect it."""
    if not language:
        return None
    base = language.replace("_", "-").split("-", 1)[0].lower()
    return _QWEN_NAMES.get(base)


def language_code(qwen_name: str | None) -> str | None:
    """The ISO code for a language name Qwen reports (``"Vietnamese"`` -> ``"vi"``)."""
    if not qwen_name:
        return None
    return _CODES_BY_NAME.get(qwen_name.strip())


def decode_transcribe_audio(audio: bytes) -> np.ndarray:
    """Decode the request's audio to float32 16 kHz mono, at most MAX_SECONDS long."""
    try:
        return decode_audio_16k_mono(audio, max_seconds=MAX_SECONDS)
    except LanguageInputError as exc:
        raise TranscribeInputError(str(exc).replace("Language identification", "Transcription")) from exc


def _torch_dtype(name: str) -> torch.dtype:
    dtypes = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in dtypes:
        raise ValueError(f"unsupported transcribe dtype: {name}")
    return dtypes[name]


class QwenTranscriber:
    """Qwen3-ASR loaded once in the speech GPU process."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, device: str = "cuda:0", dtype: str = "float16"):
        self.model_dir = str(model_dir)
        self.device = device
        self.dtype = _torch_dtype(dtype)
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_dir,
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=8,
            max_new_tokens=512,
        )

    def transcribe(self, audio: np.ndarray, language: str | None) -> dict[str, Any]:
        """Return {text, language (Qwen's name), languageCode}; audio is float32 16 kHz mono."""
        self.load()
        results = self._model.transcribe(
            audio=[(np.asarray(audio, dtype=np.float32).reshape(-1), SAMPLE_RATE)],
            language=[qwen_language(language)],
        )
        result = results[0]
        return {
            "text": (result.text or "").strip(),
            "language": result.language,
            "languageCode": language_code(result.language),
        }
