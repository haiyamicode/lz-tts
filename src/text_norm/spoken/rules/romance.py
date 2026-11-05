"""Spanish, Italian and Portuguese ordinal indicators: 2º, 3ª."""

from __future__ import annotations

import re
from collections.abc import Callable

from ..locales import LOCALES
from .base import RuleContext

_ORDINAL_RE = re.compile(r"(?<!\w)(?P<number>\d+)\.?(?P<indicator>[ºª])(?!\w)")


def _feminine(words: str) -> str:
    return re.sub(r"o\b", "a", words)


def ordinal_indicator_rule(language: str) -> Callable[[str, RuleContext], str]:
    sign_ordinal_words = LOCALES[language].sign_ordinal_words

    def ordinal_words(match: re.Match[str]) -> str:
        value = int(match.group("number"))
        if match.group("indicator") == "ª":
            # The feminine keeps its ending: 1ª planta, primera.
            return _feminine(sign_ordinal_words(value, match.string[: match.start()], ""))
        return sign_ordinal_words(
            value, match.string[: match.start()], match.string[match.end() :]
        )

    def apply(text: str, context: RuleContext) -> str:
        return _ORDINAL_RE.sub(ordinal_words, text)

    return apply
