"""Vietnamese written text normalization."""

from __future__ import annotations

import re
import unicodedata

from num2words import num2words

_NUMBER_RE = re.compile(r"\b(\d{1,3}(?:[.,]\d{3}){2,}|\d+[.,]\d+|\d+)\b")
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:[.,]\d{3}){1,}")
_DECIMAL_RE = re.compile(r"\d+[.,]\d+")
_PERCENT_RE = re.compile(r"\s*%")
_WHITESPACE_RE = re.compile(r"\s+")


def _number_to_words(match: re.Match[str]) -> str:
    raw = match.group(1)
    if _GROUPED_NUMBER_RE.fullmatch(raw):
        value = int(re.sub(r"[.,]", "", raw))
    elif _DECIMAL_RE.fullmatch(raw):
        value = float(raw.replace(",", "."))
    else:
        value = int(raw)
    return num2words(value, lang="vi")


def normalize_vietnamese(text: str) -> str:
    """Normalize fixed Vietnamese written forms before phonemization."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _NUMBER_RE.sub(_number_to_words, normalized)
    normalized = _PERCENT_RE.sub(" phần trăm", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()
