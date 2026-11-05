"""French-only spoken normalization rules."""

from __future__ import annotations

import re

from num2words import num2words

from .base import RuleContext

_FIRST_RE = re.compile(
    r"(?<!\w)1(?P<suffix>er|re)(?!\w)",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(
    r"(?<!\w)(?P<number>\d+)(?:e|è|ème|eme|ième)(?!\w)",
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
    text = _ORDINAL_RE.sub(
        lambda match: num2words(
            int(match.group("number")),
            lang="fr",
            to="ordinal",
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
