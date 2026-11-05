"""English-only spoken normalization rules."""

from __future__ import annotations

import re

from num2words import num2words

from .base import RuleContext

_Q_AND_A_RE = re.compile(r"(?<!\w)Q\s*&\s*A(?!\w)", re.IGNORECASE)
_ORDINAL_RE = re.compile(
    r"(?<!\w)(?P<number>\d+)"
    r"(?:st|nd|rd|th)(?!\w)",
    re.IGNORECASE,
)
# 1990s, 1990's, '90s, 90s.
_DECADE_RE = re.compile(
    r"(?<![\w'’])(?:['’](?P<short>\d0)|(?P<decade>\d{1,3}0))['’]?s(?!\w)"
)


def _decade_words(match: re.Match[str]) -> str:
    short = match.group("short")
    if short is not None:
        words = num2words(int(short), lang="en")
    else:
        decade = match.group("decade")
        value = int(decade)
        words = (
            num2words(value, lang="en", to="year")
            if len(decade) >= 3
            else num2words(value, lang="en")
        )
    return f"{words[:-1]}ies" if words.endswith("y") else f"{words}s"


def apply(text: str, context: RuleContext) -> str:
    text = _Q_AND_A_RE.sub("Q and A", text)
    text = _DECADE_RE.sub(_decade_words, text)
    return _ORDINAL_RE.sub(
        lambda match: num2words(
            int(match.group("number")),
            lang="en",
            to="ordinal",
        ),
        text,
    )
