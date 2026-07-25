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
_ROUND_THE_CLOCK_RE = re.compile(r"(?<![\w/])24\s*/\s*(7|24)(?![\w/])")
_TERMINAL_FOUR_AFTER_TENS_RE = re.compile(
    r"\b(hai|ba|bốn|năm|sáu|bảy|tám|chín) mươi bốn\b"
)


def vietnamese_number_words(value: int | float) -> str:
    words = num2words(value, lang="vi")
    return _TERMINAL_FOUR_AFTER_TENS_RE.sub(r"\1 mươi tư", words)


def _number_to_words(match: re.Match[str]) -> str:
    raw = match.group(1)
    if _GROUPED_NUMBER_RE.fullmatch(raw):
        value = int(re.sub(r"[.,]", "", raw))
    elif _DECIMAL_RE.fullmatch(raw):
        value = float(raw.replace(",", "."))
    else:
        value = int(raw)
    return vietnamese_number_words(value)


def _round_the_clock_to_words(match: re.Match[str]) -> str:
    numerator = vietnamese_number_words(24)
    denominator = vietnamese_number_words(int(match.group(1)))
    return f"{numerator} trên {denominator}"


def normalize_vietnamese(text: str) -> str:
    """Normalize fixed Vietnamese written forms before phonemization."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ROUND_THE_CLOCK_RE.sub(_round_the_clock_to_words, normalized)
    normalized = _NUMBER_RE.sub(_number_to_words, normalized)
    normalized = _PERCENT_RE.sub(" phần trăm", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()
