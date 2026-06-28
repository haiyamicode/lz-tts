"""Text-only written normalization helpers."""

from __future__ import annotations

from .chinese import normalize_chinese
from .english import normalize_english
from .japanese import normalize_japanese
from .korean import normalize_korean
from .romance import normalize_french, normalize_spanish
from .vietnamese import normalize_vietnamese


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_text(text: str, language: str) -> str:
    """Normalize written text for a language without phonemizing it."""
    lang = (language or "").strip().lower().replace("_", "-")
    base = lang.split("-", 1)[0]

    if lang in {"zh-mix-en", "zh_mix_en"}:
        return text
    if base in {"en"}:
        return normalize_english(text)
    if base in {"zh", "cmn", "yue"}:
        if lang.startswith("cmn-latn") and not _contains_cjk(text):
            return text
        return normalize_chinese(text)
    if base in {"ja", "jp"}:
        return normalize_japanese(text)
    if base in {"ko", "kr"}:
        return normalize_korean(text)
    if base in {"vi", "vie"}:
        return normalize_vietnamese(text)
    if base == "fr":
        return normalize_french(text)
    if base in {"es", "sp"}:
        return normalize_spanish(text)
    return text


__all__ = [
    "normalize_chinese",
    "normalize_english",
    "normalize_french",
    "normalize_japanese",
    "normalize_korean",
    "normalize_spanish",
    "normalize_text",
    "normalize_vietnamese",
]
