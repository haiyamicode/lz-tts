"""Khmer written-text normalization."""

from __future__ import annotations

import re

from km_num2text import num_normalize

_DIGIT_CLASS = "0-9\u17e0-\u17e9"
_GROUPED_INTEGER_RE = re.compile(
    rf"(?<![{_DIGIT_CLASS}])"
    rf"[{_DIGIT_CLASS}]{{1,3}}(?:\s+[{_DIGIT_CLASS}]{{3}})+"
    rf"(?![{_DIGIT_CLASS}])"
)
_DOTTED_SEQUENCE_RE = re.compile(
    rf"(?<![{_DIGIT_CLASS}])"
    rf"[{_DIGIT_CLASS}]+(?:\.[{_DIGIT_CLASS}]+){{2,}}"
    rf"(?![{_DIGIT_CLASS}])"
)
_SPACED_OPERATORS = (
    (re.compile(r"\s*\+\s*"), " បូក "),
    (re.compile(r"\s*=\s*"), " ស្មើ "),
    (re.compile(r"\s+-\s+"), " ដក "),
)


def _remove_grouping_spaces(match: re.Match[str]) -> str:
    return re.sub(r"\s+", "", match.group(0))


def _verbalize_dotted_sequence(match: re.Match[str]) -> str:
    return " ចុច ".join(num_normalize(part) for part in match.group(0).split("."))


def normalize_khmer(text: str) -> str:
    """Verbalize Khmer/Western digits and common numeric notation for TTS."""
    normalized = _GROUPED_INTEGER_RE.sub(_remove_grouping_spaces, text)
    normalized = _DOTTED_SEQUENCE_RE.sub(_verbalize_dotted_sequence, normalized)
    normalized = num_normalize(normalized)
    normalized = normalized.replace("%", " ភាគរយ")
    for pattern, replacement in _SPACED_OPERATORS:
        normalized = pattern.sub(replacement, normalized)
    return re.sub(r"\s+", " ", normalized).strip()
