"""Shared types and span construction for language frontends."""

from __future__ import annotations

import re
from collections.abc import Callable

FrontendResult = tuple[list[str], list[list[int]] | None]
LanguageFrontend = Callable[[str], FrontendResult]

_NONSPACE_PATTERN = re.compile(r"\S+")


def append_word(
    phonemes: list[str],
    word_spans: list[list[int]],
    text_start: int,
    text_end: int,
    ipa: str,
) -> None:
    word_phonemes = list(ipa)
    if not word_phonemes:
        return

    if phonemes:
        phonemes.append(" ")

    ph_start = len(phonemes)
    phonemes.extend(word_phonemes)
    word_spans.append([text_start, text_end, ph_start, len(phonemes)])


def phonemize_phrases(
    text: str,
    transliterate: Callable[[str], str],
) -> FrontendResult:
    phonemes: list[str] = []
    word_spans: list[list[int]] = []
    for match in _NONSPACE_PATTERN.finditer(text):
        append_word(
            phonemes,
            word_spans,
            match.start(),
            match.end(),
            transliterate(match.group(0)),
        )
    return phonemes, word_spans or None
