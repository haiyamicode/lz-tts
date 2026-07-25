"""Language-aware written-form verbalization for raw-text TTS frontends."""

from __future__ import annotations

import re
import unicodedata

from .rules import RuleContext, apply_language_rules

from .locales import (
    ALL_DAY_WORDS as _ALL_DAY_WORDS,
    BASE_LANGUAGE_ALIASES as _BASE_LANGUAGE_ALIASES,
    CARBON_DIOXIDE_WORDS as _CARBON_DIOXIDE_WORDS,
    CURRENCY_ALIAS_WORDS as _CURRENCY_ALIAS_WORDS,
    CURRENCY_ATTRIBUTIVE_NOUNS as _CURRENCY_ATTRIBUTIVE_NOUNS,
    CURRENCY_FEMININE_ONE as _CURRENCY_FEMININE_ONE,
    CURRENCY_MAGNITUDES as _CURRENCY_MAGNITUDES,
    CURRENCY_SINGULAR_WORDS as _CURRENCY_SINGULAR_WORDS,
    CURRENCY_WORDS as _CURRENCY_WORDS,
    DECIMAL_WORDS as _DECIMAL_WORDS,
    FEMININE_ONE_WORD as _FEMININE_ONE_WORD,
    GROUPING_SEPARATOR as _GROUPING_SEPARATOR,
    LOCALES as _LOCALES,
    PERCENT_WORDS as _PERCENT_WORDS,
    RATE_DENOMINATOR_ALIASES as _RATE_DENOMINATOR_ALIASES,
    RATE_DENOMINATOR_WORDS as _RATE_DENOMINATOR_WORDS,
    RATE_TEMPLATES as _RATE_TEMPLATES,
    RANGE_WORDS as _RANGE_WORDS,
    ROUND_THE_CLOCK_WORDS as _ROUND_THE_CLOCK_WORDS,
    SUPPORTED_LANGUAGES as _SUPPORTED_LANGUAGES,
    TECH_LETTER_WORDS as _TECH_LETTER_WORDS,
    UNIT_ALIASES as _UNIT_ALIASES,
    UNIT_FEMININE_ONE as _UNIT_FEMININE_ONE,
    UNIT_SINGULAR_WORDS as _UNIT_SINGULAR_WORDS,
    UNIT_WORDS as _UNIT_WORDS,
)

_CJK_SCRIPT = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
_TOKEN_LEFT_BOUNDARY = rf"(?:(?<![\w/])|(?<=[{_CJK_SCRIPT}]))"
_TOKEN_RIGHT_BOUNDARY = rf"(?:(?![\w/])|(?=[{_CJK_SCRIPT}]))"
_ROUND_THE_CLOCK_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}24\s*/\s*7{_TOKEN_RIGHT_BOUNDARY}"
)


def _alias_pattern(aliases: tuple[str, ...]) -> str:
    return "|".join(
        re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)
    )


def _all_day_pattern(language: str) -> re.Pattern[str]:
    word_aliases = _alias_pattern(_LOCALES[language].all_day_hour_aliases)
    hour_token = rf"h|{word_aliases}" if word_aliases else "h"
    return re.compile(
        rf"{_TOKEN_LEFT_BOUNDARY}24\s*(?:{hour_token})?\s*/\s*24"
        rf"{_TOKEN_RIGHT_BOUNDARY}",
        re.IGNORECASE,
    )


_ALL_DAY_RES = {
    language: _all_day_pattern(language) for language in _SUPPORTED_LANGUAGES
}
_CARBON_DIOXIDE_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}CO[₂2]{_TOKEN_RIGHT_BOUNDARY}"
)
_COMPACT_TECH_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}(?P<number>\d+)"
    rf"(?P<letter>[KDG]){_TOKEN_RIGHT_BOUNDARY}"
)
_RANGE_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}(?P<start>[-−]?\d{{1,4}})\s*[-–—]\s*"
    rf"(?P<end>[-−]?\d{{1,4}}){_TOKEN_RIGHT_BOUNDARY}"
)


def _clock_pattern(language: str) -> re.Pattern[str]:
    locale = _LOCALES[language]
    suffix_aliases = _alias_pattern(locale.clock_suffix_aliases)
    suffix = rf"(?:\s*(?:{suffix_aliases}))?" if suffix_aliases else ""
    meridiem = r"(?:\s*(?P<meridiem>[ap]\.?\s*m\.?))?"
    return re.compile(
        rf"{_TOKEN_LEFT_BOUNDARY}(?P<hour>[01]?\d|2[0-3])\s*"
        r"(?P<separator>:|h)\s*"
        r"(?P<minute>[0-5]\d)"
        rf"{meridiem}{suffix}{_TOKEN_RIGHT_BOUNDARY}",
        re.IGNORECASE,
    )


_CLOCK_RES = {
    language: _clock_pattern(language) for language in _SUPPORTED_LANGUAGES
}
_NUMERIC_TOKEN = r"[-−]?\d+(?:(?:[.,])\d+)*"
_COMMON_MEASUREMENT_UNIT_ALIASES = {
    "km": "km",
    "mi": "mi",
    "m": "m",
    "cm": "cm",
    "mm": "mm",
    "kg": "kg",
    "g": "g",
}
_CASE_SENSITIVE_MEASUREMENT_UNIT_ALIASES = {
    "mAh": "mah",
    "W": "w",
    "kW": "kw",
    "GB": "gb",
    "MB": "mb",
    "GHz": "ghz",
    "MP": "mp",
    "°C": "°c",
}
_CASE_SENSITIVE_MEASUREMENT_UNIT_CASEFOLDS = frozenset(
    alias.casefold() for alias in _CASE_SENSITIVE_MEASUREMENT_UNIT_ALIASES
)
_COMMON_RATE_DENOMINATOR_ALIASES = {
    "h": "h",
    "s": "s",
    "min": "min",
    "d": "d",
}
_ALL_MEASUREMENT_UNIT_ALIASES = frozenset(
    _COMMON_MEASUREMENT_UNIT_ALIASES
).union(
    _CASE_SENSITIVE_MEASUREMENT_UNIT_ALIASES
).union(
    alias for aliases in _UNIT_ALIASES.values() for alias in aliases
)
_ALL_RATE_DENOMINATOR_ALIASES = frozenset(
    _COMMON_RATE_DENOMINATOR_ALIASES
).union(
    alias
    for aliases in _RATE_DENOMINATOR_ALIASES.values()
    for alias in aliases
)
_MEASUREMENT_UNIT_TOKEN = "|".join(
    re.escape(alias)
    for alias in sorted(_ALL_MEASUREMENT_UNIT_ALIASES, key=len, reverse=True)
)
_RATE_DENOMINATOR_TOKEN = "|".join(
    re.escape(alias)
    for alias in sorted(_ALL_RATE_DENOMINATOR_ALIASES, key=len, reverse=True)
)
_RATE_UNIT_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}(?P<number>{_NUMERIC_TOKEN})\s*"
    rf"(?P<numerator>{_MEASUREMENT_UNIT_TOKEN})\s*/\s*"
    rf"(?P<denominator>{_RATE_DENOMINATOR_TOKEN})"
    rf"{_TOKEN_RIGHT_BOUNDARY}",
    re.IGNORECASE,
)
_CURRENCY_MAGNITUDE_TOKEN = "|".join(
    re.escape(magnitude)
    for magnitude in sorted(
        {
            magnitude
            for magnitudes in _CURRENCY_MAGNITUDES.values()
            for magnitude in magnitudes
        },
        key=len,
        reverse=True,
    )
)
_CURRENCY_ALIASES = {
    "US$": "USD",
    "R$": "BRL",
    "C$": "CAD",
    "A$": "AUD",
    "CN¥": "CNY",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₩": "KRW",
    "₫": "VND",
    "đ": "VND",
    "Đ": "VND",
    "₹": "INR",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "JPY": "JPY",
    "KRW": "KRW",
    "VND": "VND",
    "BRL": "BRL",
    "CNY": "CNY",
    "RMB": "CNY",
    "INR": "INR",
    "CHF": "CHF",
    "CAD": "CAD",
    "AUD": "AUD",
}
_CURRENCY_TOKEN = "|".join(
    re.escape(alias)
    for alias in sorted(_CURRENCY_ALIASES, key=len, reverse=True)
)
_CURRENCY_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}(?:"
    rf"(?P<prefix>{_CURRENCY_TOKEN})\s*(?P<prefix_number>{_NUMERIC_TOKEN})"
    rf"(?:(?P<prefix_magnitude_separator>\s+|[-‐‑‒–—])"
    rf"(?P<prefix_magnitude>{_CURRENCY_MAGNITUDE_TOKEN}))?"
    rf"|"
    rf"(?P<suffix_number>{_NUMERIC_TOKEN})"
    rf"(?:\s+(?P<suffix_magnitude>{_CURRENCY_MAGNITUDE_TOKEN}))?"
    rf"\s*(?P<suffix>{_CURRENCY_TOKEN})"
    rf"){_TOKEN_RIGHT_BOUNDARY}",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}(?P<number>{_NUMERIC_TOKEN})\s*%"
    rf"{_TOKEN_RIGHT_BOUNDARY}"
)
_UNIT_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}(?P<number>{_NUMERIC_TOKEN})\s*"
    rf"(?P<unit>{_MEASUREMENT_UNIT_TOKEN})"
    rf"{_TOKEN_RIGHT_BOUNDARY}",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}{_NUMERIC_TOKEN}{_TOKEN_RIGHT_BOUNDARY}"
)

_OPAQUE_SPAN_RE = re.compile(
    r"https?://\S+"
    r"|www\.\S+"
    r"|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    rf"|{_TOKEN_LEFT_BOUNDARY}(?:[A-Za-z]:)?"
    rf"(?:[/\\][^\s/\\]+){{2,}}{_TOKEN_RIGHT_BOUNDARY}",
    re.IGNORECASE,
)
_MACHINE_NUMERIC_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}(?=[A-Za-z0-9_.-]*\d)"
    rf"[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z0-9_.-]+"
    rf"{_TOKEN_RIGHT_BOUNDARY}"
    rf"|{_TOKEN_LEFT_BOUNDARY}\d+(?:\.\d+)+"
    rf"(?:[-_]?(?:alpha|beta|dev|pre|rc)\d*)"
    rf"{_TOKEN_RIGHT_BOUNDARY}"
    rf"|{_TOKEN_LEFT_BOUNDARY}(?:\d{{1,3}}\.){{3}}\d{{1,3}}"
    rf"{_TOKEN_RIGHT_BOUNDARY}",
    re.IGNORECASE,
)
_AMBIGUOUS_NUMERIC_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}\d+(?:\.\d+){{2,}}{_TOKEN_RIGHT_BOUNDARY}"
    rf"|{_TOKEN_LEFT_BOUNDARY}\d{{1,4}}[/-]\d{{1,2}}"
    rf"[/-]\d{{1,4}}{_TOKEN_RIGHT_BOUNDARY}"
    rf"|{_TOKEN_LEFT_BOUNDARY}\+?\(?\d{{1,4}}\)?"
    rf"(?:[\t \-‐‑‒–—]\d{{2,4}}){{2,}}{_TOKEN_RIGHT_BOUNDARY}"
    rf"|{_TOKEN_LEFT_BOUNDARY}\d{{2,}}"
    rf"(?:[\-‐‑‒–—]\d{{2,}})+{_TOKEN_RIGHT_BOUNDARY}",
    re.IGNORECASE,
)


def _base_language(language: str) -> str:
    base = (language or "").strip().lower().replace("_", "-").split("-", 1)[0]
    return _BASE_LANGUAGE_ALIASES.get(base, base)


def _number_words(value: int, language: str) -> str:
    return _LOCALES[language].number_words(value)


def _digit_words(digits: str, language: str) -> str:
    words = [_number_words(int(digit), language) for digit in digits]
    return _LOCALES[language].digit_separator.join(words)


def _parse_number(raw: str, language: str) -> tuple[int, str | None] | None:
    sign = -1 if raw.startswith(("-", "−")) else 1
    if sign < 0:
        raw = raw[1:]
    separators = [char for char in raw if char in ".,"]
    if not separators:
        return sign * int(raw), None

    groups = re.split(r"[.,]", raw)
    if any(not group for group in groups):
        return None

    unique_separators = set(separators)
    if len(unique_separators) == 1 and len(groups) > 2:
        if len(groups[0]) <= 3 and all(
            len(group) == 3 for group in groups[1:]
        ):
            return sign * int("".join(groups)), None
        return None

    if len(unique_separators) == 2:
        decimal_index = max(raw.rfind("."), raw.rfind(","))
        integer_raw = re.sub(r"[.,]", "", raw[:decimal_index])
        fraction = raw[decimal_index + 1 :]
        if not integer_raw or not fraction:
            return None
        return sign * int(integer_raw), fraction

    separator = separators[0]
    integer_raw, fraction = groups
    if (
        separator == _GROUPING_SEPARATOR[language]
        and len(fraction) == 3
        and len(integer_raw) <= 3
    ):
        return sign * int(integer_raw + fraction), None
    return sign * int(integer_raw), fraction


def _spoken_number(raw: str, language: str) -> str:
    try:
        negative = raw.startswith(("-", "−"))
        unsigned = raw[1:] if negative else raw
        if unsigned.isdigit() and len(unsigned) > 1 and unsigned.startswith("0"):
            result = _digit_words(unsigned, language)
            return (
                _LOCALES[language].negative_template.format(number=result)
                if negative
                else result
            )
        parsed = _parse_number(raw, language)
        if parsed is None:
            return raw
        integer, fraction = parsed
        if fraction is not None:
            integer_words = _number_words(abs(integer), language)
        else:
            result = _number_words(abs(integer), language)
            return (
                _LOCALES[language].negative_template.format(number=result)
                if negative
                else result
            )

        result = _LOCALES[language].decimal_template.format(
            integer=integer_words,
            decimal=_DECIMAL_WORDS[language],
            fraction=_digit_words(fraction, language),
        )
        return (
            _LOCALES[language].negative_template.format(number=result)
            if negative
            else result
        )
    except (ArithmeticError, KeyError, NotImplementedError, OverflowError, ValueError):
        return raw


def _is_one(raw: str, language: str) -> bool:
    parsed = _parse_number(raw, language)
    if parsed is None:
        return False
    integer, fraction = parsed
    return integer == 1 and (
        fraction is None or all(digit == "0" for digit in fraction)
    )


def _quantity_words(raw: str, language: str) -> str:
    one_quantity_word = _LOCALES[language].one_quantity_word
    if _is_one(raw, language) and one_quantity_word is not None:
        return one_quantity_word
    return _spoken_number(raw, language)


def _join_quantity_unit(quantity: str, unit: str, language: str) -> str:
    separator = _LOCALES[language].quantity_unit_separator
    return f"{quantity}{separator}{unit}"


def _canonical_currency(raw: str) -> str:
    canonical = _CURRENCY_ALIASES.get(raw)
    if canonical is None:
        canonical = _CURRENCY_ALIASES[raw.upper()]
    return canonical


def _currency_quantity_words(raw: str, language: str) -> str:
    parsed = _parse_number(raw, language)
    if parsed is not None:
        integer, fraction = parsed
        if fraction is not None and all(digit == "0" for digit in fraction):
            return _number_words(integer, language)
    return _quantity_words(raw, language)


def _currency_words(
    raw: str,
    currency: str,
    language: str,
    magnitude: str | None = None,
    *,
    attributive: bool = False,
    unit_override: str | None = None,
) -> str:
    unit = unit_override or _CURRENCY_WORDS[language][currency]
    if (magnitude is None and _is_one(raw, language)) or attributive:
        unit = _CURRENCY_SINGULAR_WORDS.get(language, {}).get(currency, unit)

    quantity = _currency_quantity_words(raw, language)
    if magnitude is None:
        if (
            _is_one(raw, language)
            and currency in _CURRENCY_FEMININE_ONE.get(language, set())
        ):
            quantity = _FEMININE_ONE_WORD[language]
        return _join_quantity_unit(quantity, unit, language)

    magnitude_entry = _CURRENCY_MAGNITUDES.get(language, {}).get(
        magnitude.casefold()
    )
    if magnitude_entry is None:
        raise ValueError(
            f"Unsupported currency magnitude {magnitude!r} for {language}"
        )
    magnitude_words, connector = magnitude_entry
    magnitude_one_words = _LOCALES[language].currency_magnitude_one_words
    if _is_one(raw, language) and magnitude.casefold() in magnitude_one_words:
        amount = magnitude_one_words[magnitude.casefold()]
    else:
        amount = f"{quantity} {magnitude_words}"

    if connector:
        elision_vowels = _LOCALES[language].currency_connector_elision_vowels
        if elision_vowels and unit[0].lower() in elision_vowels:
            return f"{amount} d'{unit}"
        return f"{amount} {connector} {unit}"
    return f"{amount} {unit}"


def _currency_match_words(match: re.Match[str], language: str) -> str:
    raw = match.group("prefix_number") or match.group("suffix_number")
    token = match.group("prefix") or match.group("suffix")
    magnitude = match.group("prefix_magnitude") or match.group(
        "suffix_magnitude"
    )
    magnitude_separator = match.group("prefix_magnitude_separator")
    attributive = bool(
        magnitude_separator and not magnitude_separator.isspace()
    )
    if (
        match.group("prefix")
        and magnitude is not None
        and not attributive
    ):
        following_word = re.match(
            r"\s+(?P<word>[A-Za-z]+)",
            match.string[match.end() :],
        )
        attributive = bool(
            following_word
            and following_word.group("word").lower()
            in _CURRENCY_ATTRIBUTIVE_NOUNS.get(language, frozenset())
        )
    if (
        magnitude is not None
        and magnitude.casefold() not in _CURRENCY_MAGNITUDES.get(language, {})
    ):
        return match.group(0)
    return _currency_words(
        raw,
        _canonical_currency(token),
        language,
        magnitude,
        attributive=attributive,
        unit_override=_CURRENCY_ALIAS_WORDS.get(language, {}).get(token),
    )


def _unit_words(raw: str, unit_key: str, language: str) -> str:
    unit = _UNIT_WORDS[language][unit_key]
    if _is_one(raw, language):
        unit = _UNIT_SINGULAR_WORDS.get(language, {}).get(unit_key, unit)
    return _join_quantity_unit(
        _unit_quantity_words(raw, unit_key, language),
        unit,
        language,
    )


def _unit_quantity_words(raw: str, unit_key: str, language: str) -> str:
    if (
        _is_one(raw, language)
        and unit_key in _UNIT_FEMININE_ONE.get(language, frozenset())
    ):
        return _FEMININE_ONE_WORD[language]
    return _quantity_words(raw, language)


def _canonical_measurement_unit(raw: str, language: str) -> str | None:
    if raw == "M":
        return None
    case_sensitive = _CASE_SENSITIVE_MEASUREMENT_UNIT_ALIASES.get(raw)
    if case_sensitive is not None:
        return case_sensitive
    normalized = raw.casefold()
    if normalized in _CASE_SENSITIVE_MEASUREMENT_UNIT_CASEFOLDS:
        return None
    return _COMMON_MEASUREMENT_UNIT_ALIASES.get(
        normalized
    ) or _UNIT_ALIASES[language].get(normalized)


def _unit_match_words(match: re.Match[str], language: str) -> str:
    unit_key = _canonical_measurement_unit(match.group("unit"), language)
    if unit_key is None:
        return match.group(0)
    return _unit_words(match.group("number"), unit_key, language)


def _rate_match_words(match: re.Match[str], language: str) -> str:
    numerator_key = _canonical_measurement_unit(
        match.group("numerator"),
        language,
    )
    normalized_denominator = match.group("denominator").casefold()
    denominator_key = _COMMON_RATE_DENOMINATOR_ALIASES.get(
        normalized_denominator
    ) or _RATE_DENOMINATOR_ALIASES[language].get(
        normalized_denominator
    )
    if numerator_key is None or denominator_key is None:
        return match.group(0)

    numerator = _UNIT_WORDS[language][numerator_key]
    if _is_one(match.group("number"), language):
        numerator = _UNIT_SINGULAR_WORDS.get(language, {}).get(
            numerator_key,
            numerator,
        )
    return _RATE_TEMPLATES[language].format(
        quantity=_unit_quantity_words(
            match.group("number"),
            numerator_key,
            language,
        ),
        numerator=numerator,
        denominator=_RATE_DENOMINATOR_WORDS[language][denominator_key],
    )


def _compact_tech_words(raw: str, letter: str, language: str) -> str:
    return _join_quantity_unit(
        _spoken_number(raw, language),
        _TECH_LETTER_WORDS[language][letter],
        language,
    )


def _clock_words(
    hour: int,
    minute: int | None,
    meridiem: str | None,
    language: str,
) -> str:
    return _LOCALES[language].clock_words(hour, minute, meridiem)


def _clock_match_words(match: re.Match[str], language: str) -> str:
    meridiem = match.groupdict().get("meridiem")
    locale = _LOCALES[language]
    spoken = _clock_words(
        int(match.group("hour")),
        int(match.group("minute")),
        meridiem if locale.clock_consumes_meridiem else None,
        language,
    )
    if meridiem and not locale.clock_consumes_meridiem:
        return f"{spoken} {meridiem}"
    return spoken


def _placeholder_name(index: int) -> str:
    name = ""
    value = index
    while True:
        name = chr(ord("A") + value % 26) + name
        value = value // 26 - 1
        if value < 0:
            return f"\ue000{name}\ue001"


def _store_protected(value: str, protected: dict[str, str]) -> str:
    placeholder = _placeholder_name(len(protected))
    protected[placeholder] = value
    return placeholder


def _protect_if_unchanged(
    match: re.Match[str],
    replacement: str,
    protected: dict[str, str],
) -> str:
    if replacement == match.group(0):
        return _store_protected(match.group(0), protected)
    return replacement


def _protect_spans(
    text: str,
    pattern: re.Pattern[str],
    protected: dict[str, str],
    *,
    allow_grouped_number: bool = True,
) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if allow_grouped_number and re.fullmatch(r"\d+(?:\.\d+){2,}", value):
            groups = value.split(".")
            if len(groups[0]) <= 3 and all(
                len(group) == 3 for group in groups[1:]
            ):
                return value
        return _store_protected(value, protected)

    return pattern.sub(replace, text)


def _restore_spans(text: str, protected: dict[str, str]) -> str:
    for placeholder, original in protected.items():
        text = text.replace(placeholder, original)
    return text


def verbalize_spoken_text(text: str, language: str) -> str:
    """Expand unambiguous written forms for a TTS frontend without a TN stage."""
    base = _base_language(language)
    if base not in _SUPPORTED_LANGUAGES:
        return text

    normalized = unicodedata.normalize("NFKC", text)
    protected: dict[str, str] = {}
    normalized = _protect_spans(normalized, _OPAQUE_SPAN_RE, protected)
    normalized = _protect_spans(
        normalized,
        _MACHINE_NUMERIC_RE,
        protected,
        allow_grouped_number=False,
    )
    normalized = _ALL_DAY_RES[base].sub(_ALL_DAY_WORDS[base], normalized)
    normalized = _ROUND_THE_CLOCK_RE.sub(_ROUND_THE_CLOCK_WORDS[base], normalized)
    normalized = apply_language_rules(
        normalized,
        base,
        RuleContext(
            spoken_number=lambda raw: _spoken_number(raw, base),
            digit_words=lambda digits: _digit_words(digits, base),
            currency_words=lambda raw, currency, unit_override: _currency_words(
                raw,
                currency,
                base,
                unit_override=unit_override,
            ),
            clock_words=lambda hour, minute, meridiem: _clock_words(
                hour,
                minute,
                meridiem,
                base,
            ),
        ),
    )
    normalized = _protect_spans(
        normalized,
        _AMBIGUOUS_NUMERIC_RE,
        protected,
    )
    normalized = _CARBON_DIOXIDE_RE.sub(_CARBON_DIOXIDE_WORDS[base], normalized)
    normalized = _COMPACT_TECH_RE.sub(
        lambda match: _compact_tech_words(
            match.group("number"),
            match.group("letter").upper(),
            base,
        ),
        normalized,
    )
    normalized = _RANGE_RE.sub(
        lambda match: _protect_if_unchanged(
            match,
            _RANGE_WORDS[base].format(
                start=_spoken_number(match.group("start"), base),
                end=_spoken_number(match.group("end"), base),
            )
            if (
                len(match.group("start").lstrip("-−"))
                == len(match.group("end").lstrip("-−"))
            )
            else match.group(0),
            protected,
        ),
        normalized,
    )

    normalized = _CLOCK_RES[base].sub(
        lambda match: _clock_match_words(match, base),
        normalized,
    )
    normalized = _RATE_UNIT_RE.sub(
        lambda match: _protect_if_unchanged(
            match,
            _rate_match_words(match, base),
            protected,
        ),
        normalized,
    )
    normalized = _CURRENCY_RE.sub(
        lambda match: _protect_if_unchanged(
            match,
            _currency_match_words(match, base),
            protected,
        ),
        normalized,
    )
    normalized = _PERCENT_RE.sub(
        lambda match: _join_quantity_unit(
            _spoken_number(match.group("number"), base),
            _PERCENT_WORDS[base],
            base,
        ),
        normalized,
    )
    normalized = _UNIT_RE.sub(
        lambda match: _protect_if_unchanged(
            match,
            _unit_match_words(match, base),
            protected,
        ),
        normalized,
    )
    normalized = _NUMBER_RE.sub(
        lambda match: _spoken_number(match.group(0), base),
        normalized,
    )
    return _restore_spans(normalized, protected)


__all__ = ["verbalize_spoken_text"]
