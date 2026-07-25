"""Registry for language-specific spoken normalization rules."""

from __future__ import annotations

from collections.abc import Callable

from . import en, fr, vi, zh
from .base import RuleContext

LanguageRule = Callable[[str, RuleContext], str]

LANGUAGE_RULES: dict[str, LanguageRule] = {
    "en": en.apply,
    "fr": fr.apply,
    "vi": vi.apply,
    "zh": zh.apply,
}


def apply_language_rules(
    text: str,
    language: str,
    context: RuleContext,
) -> str:
    rule = LANGUAGE_RULES.get(language)
    return rule(text, context) if rule is not None else text


__all__ = ["RuleContext", "apply_language_rules"]
