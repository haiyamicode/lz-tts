"""Shared callback interface for language-specific spoken rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RuleContext:
    spoken_number: Callable[[str], str]
    digit_words: Callable[[str], str]
    currency_words: Callable[[str, str, str | None], str]
    clock_words: Callable[[int, int | None, str | None], str]
