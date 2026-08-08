"""Locale-aware word segmentation with Python string offsets."""

from __future__ import annotations

from typing import NamedTuple


class WordSpan(NamedTuple):
    start: int
    end: int
    text: str


def _utf16_offset_to_python_index(text: str) -> dict[int, int]:
    """Map ICU's UTF-16 boundaries to Python's Unicode string indices."""
    offsets = {0: 0}
    utf16_offset = 0
    for index, character in enumerate(text):
        utf16_offset += 2 if ord(character) > 0xFFFF else 1
        offsets[utf16_offset] = index + 1
    return offsets


def icu_word_spans(text: str, locale: str = "und") -> list[WordSpan]:
    """Return lexical ICU word spans with exact Python string offsets."""
    from icu import BreakIterator, Locale

    iterator = BreakIterator.createWordInstance(Locale((locale or "und").replace("-", "_")))
    iterator.setText(text)
    utf16_to_python = _utf16_offset_to_python_index(text)

    spans: list[WordSpan] = []
    start_utf16 = iterator.first()
    for end_utf16 in iterator:
        status = iterator.getRuleStatus()
        start = utf16_to_python.get(start_utf16)
        end = utf16_to_python.get(end_utf16)
        start_utf16 = end_utf16

        if start is None or end is None or end <= start:
            continue

        piece = text[start:end]
        if status == 0 or not any(
            character.isalpha() or character.isnumeric() for character in piece
        ):
            continue
        spans.append(WordSpan(start, end, piece))

    return spans


__all__ = ["WordSpan", "icu_word_spans"]
