"""Locale registry for spoken-text normalization."""

from __future__ import annotations

from .base import SpokenLocale
from .de import LOCALE as DE
from .en import LOCALE as EN
from .es import LOCALE as ES
from .fr import LOCALE as FR
from .it import LOCALE as IT
from .ja import LOCALE as JA
from .ko import LOCALE as KO
from .pt import LOCALE as PT
from .vi import LOCALE as VI
from .zh import LOCALE as ZH

LOCALES: dict[str, SpokenLocale] = {
    locale.language: locale
    for locale in (DE, EN, ES, FR, IT, JA, KO, PT, VI, ZH)
}

SUPPORTED_LANGUAGES = frozenset(LOCALES)
BASE_LANGUAGE_ALIASES = {
    "jp": "ja",
    "kr": "ko",
    "vie": "vi",
    "sp": "es",
}

CURRENCY_MAGNITUDES = {
    language: locale.currency_magnitudes
    for language, locale in LOCALES.items()
    if locale.currency_magnitudes
}
CURRENCY_ATTRIBUTIVE_NOUNS = {
    language: locale.currency_attributive_nouns
    for language, locale in LOCALES.items()
    if locale.currency_attributive_nouns
}
ROUND_THE_CLOCK_WORDS = {
    language: locale.round_the_clock for language, locale in LOCALES.items()
}
ALL_DAY_WORDS = {
    language: locale.all_day for language, locale in LOCALES.items()
}
CARBON_DIOXIDE_WORDS = {
    language: locale.carbon_dioxide for language, locale in LOCALES.items()
}
TECH_LETTER_WORDS = {
    language: locale.tech_letter_words for language, locale in LOCALES.items()
}
RANGE_WORDS = {
    language: locale.range_template for language, locale in LOCALES.items()
}
DECIMAL_WORDS = {
    language: locale.decimal_word for language, locale in LOCALES.items()
}
PERCENT_WORDS = {
    language: locale.percent_word for language, locale in LOCALES.items()
}
CURRENCY_WORDS = {
    language: locale.currency_words for language, locale in LOCALES.items()
}
CURRENCY_ALIAS_WORDS = {
    language: locale.currency_alias_words
    for language, locale in LOCALES.items()
    if locale.currency_alias_words
}
UNIT_WORDS = {
    language: locale.unit_words for language, locale in LOCALES.items()
}
UNIT_ALIASES = {
    language: {
        **{
            word.casefold(): unit
            for unit, word in locale.unit_words.items()
        },
        **{
            word.casefold(): unit
            for unit, word in locale.unit_singular_words.items()
        },
        **{
            alias.casefold(): unit
            for alias, unit in locale.unit_aliases.items()
        },
    }
    for language, locale in LOCALES.items()
}
UNIT_SINGULAR_WORDS = {
    language: locale.unit_singular_words
    for language, locale in LOCALES.items()
    if locale.unit_singular_words
}
UNIT_FEMININE_ONE = {
    language: locale.unit_feminine_one
    for language, locale in LOCALES.items()
    if locale.unit_feminine_one
}
RATE_TEMPLATES = {
    language: locale.rate_template for language, locale in LOCALES.items()
}
RATE_DENOMINATOR_WORDS = {
    language: locale.rate_denominator_words
    for language, locale in LOCALES.items()
}
RATE_DENOMINATOR_ALIASES = {
    language: {
        **{
            word.casefold(): denominator
            for denominator, word in locale.rate_denominator_words.items()
        },
        **{
            alias.casefold(): denominator
            for alias, denominator in locale.rate_denominator_aliases.items()
        },
    }
    for language, locale in LOCALES.items()
}
CURRENCY_SINGULAR_WORDS = {
    language: locale.currency_singular_words
    for language, locale in LOCALES.items()
    if locale.currency_singular_words
}
CURRENCY_FEMININE_ONE = {
    language: locale.currency_feminine_one
    for language, locale in LOCALES.items()
    if locale.currency_feminine_one
}
FEMININE_ONE_WORD = {
    language: locale.feminine_one_word
    for language, locale in LOCALES.items()
    if locale.feminine_one_word is not None
}
GROUPING_SEPARATOR = {
    language: locale.grouping_separator for language, locale in LOCALES.items()
}
