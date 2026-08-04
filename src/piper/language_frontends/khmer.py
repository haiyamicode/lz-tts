"""Khmer frontend backed by khmercut and khmerphonemizer."""

from __future__ import annotations

import re

from .base import FrontendResult, append_word

_SHORT_VOWEL_TRANSLATION = str.maketrans({"ĕ": "e", "ŏ": "o", "ŭ": "u"})
_LEXICAL_PATTERN = re.compile(r"[A-Za-z]+|[\u1780-\u17b3\u17b6-\u17d3]+")


def phonemize(text: str) -> FrontendResult:
    from khmercut import tokenize
    from khmerphonemizer import phonemize_single

    phonemes: list[str] = []
    word_spans: list[list[int]] = []
    cursor = 0
    for token in tokenize(text):
        token_start = text.find(token, cursor)
        if token_start < 0:
            token_start = cursor
        cursor = token_start + len(token)

        for match in _LEXICAL_PATTERN.finditer(token):
            raw_phones = phonemize_single(match.group(0))
            if raw_phones:
                append_word(
                    phonemes,
                    word_spans,
                    token_start + match.start(),
                    token_start + match.end(),
                    "".join(raw_phones).translate(_SHORT_VOWEL_TRANSLATION),
                )
    return phonemes, word_spans or None
