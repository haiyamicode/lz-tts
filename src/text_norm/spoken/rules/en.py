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


def apply(text: str, context: RuleContext) -> str:
    text = _Q_AND_A_RE.sub("Q and A", text)
    return _ORDINAL_RE.sub(
        lambda match: num2words(
            int(match.group("number")),
            lang="en",
            to="ordinal",
        ),
        text,
    )
