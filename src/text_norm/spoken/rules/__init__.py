"""Registry for language-specific spoken normalization rules."""

from __future__ import annotations

from collections.abc import Callable

from . import en, fr, vi, zh
from .base import RuleContext
from .romance import ordinal_indicator_rule

LanguageRule = Callable[[str, RuleContext], str]

LANGUAGE_RULES: dict[str, LanguageRule] = {
    "en": en.apply,
    "es": ordinal_indicator_rule("es"),
    "fr": fr.apply,
    "it": ordinal_indicator_rule("it"),
    "pt": ordinal_indicator_rule("pt"),
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
