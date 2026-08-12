"""English text normalization without G2P."""

from __future__ import annotations

import re
from typing import Dict

import inflect

_INFLECT = inflect.engine()

_ABBREVIATIONS = [
    (re.compile(r"\b%s\." % key, re.IGNORECASE), value)
    for key, value in [
        ("co", "company"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

_CURRENCY_RE = re.compile(r"(£|\$|¥|€)([0-9,.]*[0-9]+)")
_TIME_RE = re.compile(
    r"""\b
        ((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))
        :
        ([0-5][0-9])
        \s*(a\.m\.?|am|p\.m\.?|pm)?
        (?=\b|[^A-Za-z0-9]|$)""",
    re.IGNORECASE | re.X,
)
_ROUND_THE_CLOCK_RE = re.compile(r"(?<![\w/])24\s*/\s*7(?![\w/])")


def _expand_abbreviations(text: str) -> str:
    for regex, replacement in _ABBREVIATIONS:
        text = re.sub(regex, replacement, text)
    return text


def _expand_currency_value(value: str, inflection: Dict[float, str]) -> str:
    parts = value.replace(",", "").split(".")
    if len(parts) > 2:
        return f"{value} {inflection[2]}"

    text = []
    integer = int(parts[0]) if parts[0] else 0
    if integer > 0:
        integer_unit = inflection.get(integer, inflection[2])
        text.append(f"{integer} {integer_unit}")

    fraction = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if fraction > 0:
        fraction_unit = inflection.get(fraction / 100, inflection[0.02])
        text.append(f"{fraction} {fraction_unit}")

    if not text:
        return f"zero {inflection[2]}"
    return " ".join(text)


def _expand_currency(match: re.Match[str]) -> str:
    currencies = {
        "$": {0.01: "cent", 0.02: "cents", 1: "dollar", 2: "dollars"},
        "€": {0.01: "cent", 0.02: "cents", 1: "euro", 2: "euros"},
        "£": {0.01: "penny", 0.02: "pence", 1: "pound sterling", 2: "pounds sterling"},
        "¥": {0.02: "sen", 2: "yen"},
    }
    return _expand_currency_value(match.group(2), currencies[match.group(1)])


def _expand_num(value: int) -> str:
    return _INFLECT.number_to_words(value)


def _expand_time(match: re.Match[str]) -> str:
    original_hour = int(match.group(1))
    hour = original_hour
    words = []

    if hour > 12:
        hour -= 12
    elif hour == 0:
        hour = 12
    words.append(_expand_num(hour))

    minute = int(match.group(6))
    if minute > 0:
        if minute < 10:
            words.append("oh")
        words.append(_expand_num(minute))

    am_pm = match.group(7)
    if am_pm is None:
        if original_hour == 0:
            words.append("a.m.")
        elif original_hour > 12:
            words.append("p.m.")
    else:
        meridiem = am_pm.lower().replace(".", "")
        words.append(f"{meridiem[0]}.m.")
    return " ".join(words)


def _expand_time_english(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        expanded = _expand_time(match)
        if match.group(0)[-1:].isspace():
            expanded += " "
        return expanded

    text = re.sub(_TIME_RE, replace, text)
    return re.sub(r"\b([ap]\.m\.)\.", r"\1", text)


def normalize_english(text: str) -> str:
    text = text.lower()
    text = _ROUND_THE_CLOCK_RE.sub("twenty four seven", text)
    text = _expand_time_english(text)
    text = re.sub(_CURRENCY_RE, _expand_currency, text)
    return _expand_abbreviations(text)
