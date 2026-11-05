"""Typed locale data for spoken-text normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from num2words import num2words

NumberWords = Callable[[int], str]
ClockWords = Callable[[int, int | None, str | None], str]
# (value, text before the number, text after it) -> words.
ContextNumberWords = Callable[[int, str, str], str]


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
    # Readings for the number modes (src/piper/number_modes.py); None reads
    # that mode as a cardinal.
    year_words: NumberWords | None = None
    ordinal_words: ContextNumberWords | None = None
    # A cardinal whose form depends on its neighbours (Korean native counters,
    # Chinese 两); None is number_words.
    cardinal_words: ContextNumberWords | None = None
    # German writes ordinals with a period ("am 4. Mai"): the ordinal reading
    # consumes it, and an ordinal without one reads as a cardinal (um 20 Uhr).
    ordinal_period: bool = False
    # A fraction of up to three digits reads as a number after the decimal
    # word (douze virgule cinquante); otherwise digit by digit (point one four).
    decimal_fraction_number: bool = False
    # Zero in codes and phone numbers, when it differs from number_words(0)
    # (Korean 공); fractions keep number_words(0).
    code_zero_word: str | None = None
    # Between digit groups read digit by digit (phone numbers).
    digit_group_separator: str = ', '
    # "14h" with no minutes reads as clock_words(14): the hour word also
    # counts a duration (3h de route).
    bare_hour_clock: bool = False
    # (singular, plural) for a number followed by a bare degree sign: 30°.
    degree_words: tuple[str, str] | None = None
    # The ordinal a degree sign stands for when the number is an ordinal
    # (al 2° piano); None reads the degree sign as degrees.
    sign_ordinal_words: ContextNumberWords | None = None
