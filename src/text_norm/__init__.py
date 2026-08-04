"""Text-only written normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from typing import Sequence

from .assamese import normalize_assamese
from .chinese import normalize_chinese
from .english import normalize_english
from .japanese import normalize_japanese
from .khmer import normalize_khmer
from .korean import normalize_korean
from .romance import normalize_french, normalize_spanish
from .spoken import verbalize_spoken_text
from .thai import normalize_thai
from .vietnamese import normalize_vietnamese

_CANONICAL_PUNCTUATION = str.maketrans(
    {
        "：": ":",
        "；": ";",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": " ",
    }
)

_LANGUAGE_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "as": normalize_assamese,
    "asm": normalize_assamese,
    "cmn": normalize_chinese,
    "en": normalize_english,
    "es": normalize_spanish,
    "fr": normalize_french,
    "ja": normalize_japanese,
    "jp": normalize_japanese,
    "ko": normalize_korean,
    "kr": normalize_korean,
    "km": normalize_khmer,
    "khm": normalize_khmer,
    "sp": normalize_spanish,
    "th": normalize_thai,
    "tha": normalize_thai,
    "vi": normalize_vietnamese,
    "vie": normalize_vietnamese,
    "yue": normalize_chinese,
    "zh": normalize_chinese,
}


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_text(text: str, language: str) -> str:
    """Normalize written text for a language without phonemizing it."""
    lang = (language or "").strip().lower().replace("_", "-")
    base = lang.split("-", 1)[0]

    if lang in {"zh-mix-en", "zh_mix_en"}:
        return text
    if lang.startswith("cmn-latn") and not _contains_cjk(text):
        return text

    normalizer = _LANGUAGE_NORMALIZERS.get(base)
    return normalizer(text) if normalizer is not None else text


def canonicalize_text(text: str) -> str:
    """Apply backend-agnostic Unicode, punctuation, and whitespace cleanup."""
    canonical = unicodedata.normalize("NFKC", text).translate(_CANONICAL_PUNCTUATION)
    return re.sub(r"\s+", " ", canonical).strip()


def normalize_spoken_text(text: str, language: str) -> str:
    """Produce language-specific spoken text for a frontend without text TN."""
    return verbalize_spoken_text(canonicalize_text(text), language)


def prepare_tts_texts(
    texts: Sequence[str],
    languages: Sequence[str],
    *,
    normalization_enabled: bool = True,
    normalization_profile: str = "legacy",
    context_replacements_enabled: bool = True,
    context_replacer_device: str | None = None,
) -> list[str]:
    """Apply the shared normalization and contextual-replacement pipeline."""
    if len(texts) != len(languages):
        raise ValueError(
            f"texts and languages must have equal lengths: "
            f"{len(texts)} != {len(languages)}"
        )

    profile = normalization_profile.strip().lower()
    normalizers = {
        "canonical": lambda text, _language: canonicalize_text(text),
        "legacy": normalize_text,
        "spoken": normalize_spoken_text,
    }
    if profile not in normalizers:
        supported = ", ".join(sorted(normalizers))
        raise ValueError(
            f"Unknown normalization profile {normalization_profile!r}; "
            f"expected one of: {supported}"
        )
    normalizer = normalizers[profile]
    prepared = [
        normalizer(text, language) if normalization_enabled else text
        for text, language in zip(texts, languages)
    ]
    if not context_replacements_enabled or not prepared:
        return prepared

    from ..piper.context_replacer import get_replacer

    replacer = get_replacer(device=context_replacer_device)
    indices_by_language: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(languages):
        indices_by_language[language].append(index)

    results = list(prepared)
    for language, indices in indices_by_language.items():
        replaced = replacer.apply_replacements_many(
            [prepared[index] for index in indices],
            language=language,
        )
        for index, replaced_text in zip(indices, replaced):
            results[index] = replaced_text
    return results


__all__ = [
    "canonicalize_text",
    "normalize_assamese",
    "normalize_chinese",
    "normalize_english",
    "normalize_french",
    "normalize_japanese",
    "normalize_khmer",
    "normalize_korean",
    "normalize_spanish",
    "normalize_spoken_text",
    "normalize_text",
    "normalize_thai",
    "normalize_vietnamese",
    "prepare_tts_texts",
    "verbalize_spoken_text",
]
