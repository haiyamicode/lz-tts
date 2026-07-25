"""French-only spoken normalization rules."""

from __future__ import annotations

import re

from .base import RuleContext

_FIRST_RE = re.compile(
    r"(?<!\w)1(?P<suffix>er|re)(?!\w)",
    re.IGNORECASE,
)
_NUMERO_RE = re.compile(
    r"(?<!\w)(?P<label>n\s*[°º])\s*"
    r"(?P<number>\d+(?:(?:[.,])\d+)*)(?!\w)",
    re.IGNORECASE,
)


def _preserve_initial_case(source: str, replacement: str) -> str:
    if source and source[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply(text: str, context: RuleContext) -> str:
    text = _FIRST_RE.sub(
        lambda match: (
            "premier" if match.group("suffix").lower() == "er" else "première"
        ),
        text,
    )
    return _NUMERO_RE.sub(
        lambda match: (
            f"{_preserve_initial_case(match.group('label'), 'numéro')} "
            f"{context.spoken_number(match.group('number'))}"
        ),
        text,
    )
