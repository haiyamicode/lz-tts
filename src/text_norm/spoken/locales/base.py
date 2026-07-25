"""Typed locale data for spoken-text normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from num2words import num2words

NumberWords = Callable[[int], str]
ClockWords = Callable[[int, int | None, str | None], str]


def make_number_words(language: str) -> NumberWords:
    return lambda value: num2words(value, lang=language)


@dataclass(frozen=True)
class SpokenLocale:
    language: str
    number_words: NumberWords
    clock_words: ClockWords
    digit_separator: str
    decimal_template: str
    negative_template: str
    one_quantity_word: str | None
    quantity_unit_separator: str
    grouping_separator: str
    round_the_clock: str
    all_day: str
    all_day_hour_aliases: tuple[str, ...]
    clock_suffix_aliases: tuple[str, ...]
    clock_consumes_meridiem: bool
    carbon_dioxide: str
    tech_letter_words: Mapping[str, str]
    range_template: str
    decimal_word: str
    percent_word: str
    currency_magnitudes: Mapping[str, tuple[str, str]]
    currency_words: Mapping[str, str]
    currency_alias_words: Mapping[str, str]
    currency_singular_words: Mapping[str, str]
    currency_feminine_one: frozenset[str]
    feminine_one_word: str | None
    currency_attributive_nouns: frozenset[str]
    currency_magnitude_one_words: Mapping[str, str]
    currency_connector_elision_vowels: str
    unit_words: Mapping[str, str]
    unit_singular_words: Mapping[str, str]
    unit_feminine_one: frozenset[str]
    unit_aliases: Mapping[str, str]
    rate_template: str
    rate_denominator_words: Mapping[str, str]
    rate_denominator_aliases: Mapping[str, str]
