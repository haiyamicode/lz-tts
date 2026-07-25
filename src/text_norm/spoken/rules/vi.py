"""Vietnamese-only spoken normalization rules."""

from __future__ import annotations

import re
from datetime import date

from .base import RuleContext

_NUMERIC_TOKEN = r"\d+(?:(?:[.,])\d+)*"
_OR_RE = re.compile(r"(?<!\w)anh\s*/\s*chị(?!\w)", re.IGNORECASE)
_LEGAL_DOCUMENT_RE = re.compile(
    r"(?<![\w/])"
    r"(?P<number>\d{1,4})\s*/\s*"
    r"(?P<year>(?:19|20)\d{2})\s*/\s*"
    r"(?P<code>[A-Za-zÀ-ỹĐđ]+(?:-[A-Za-zÀ-ỹĐđ0-9]+)*)"
    r"(?![\w/])"
)
_DATE_RE = re.compile(
    r"(?<!\w)(?P<label>ngày\s+)?"
    r"(?P<day>0?[1-9]|[12]\d|3[01])\s*/\s*"
    r"(?P<month>0?[1-9]|1[0-2])\s*/\s*"
    r"(?P<year>(?:19|20)\d{2})(?![\w/])",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"(?<![\w/])(?P<hour>[01]?\d|2[0-3])\s*giờ\s*"
    r"(?P<minute>[0-5]\d)(?![\w/])",
    re.IGNORECASE,
)
_CURRENCY_PER_KG_RE = re.compile(
    rf"(?<!\w)(?P<number>{_NUMERIC_TOKEN})\s*"
    r"(?P<currency>VND|₫|[Đđ]|đồng)\s*/\s*kg(?!\w)",
    re.IGNORECASE,
)
_PERCENT_PER_YEAR_RE = re.compile(
    rf"(?<![\w/])(?P<number>{_NUMERIC_TOKEN})\s*%\s*/\s*năm"
    r"(?![\w/])",
    re.IGNORECASE,
)


def _preserve_initial_case(source: str, replacement: str) -> str:
    if source and source[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _legal_document_words(
    match: re.Match[str],
    context: RuleContext,
) -> str:
    return " ".join(
        (
            context.spoken_number(match.group("number")),
            context.digit_words(match.group("year")),
            match.group("code"),
        )
    )


def _date_words(match: re.Match[str], context: RuleContext) -> str:
    label = match.group("label")
    day = int(match.group("day"))
    month = int(match.group("month"))
    year = int(match.group("year"))
    try:
        date(year, month, day)
    except ValueError:
        return match.group(0)
    prefix = "Ngày" if label and label[0].isupper() else "ngày"
    return (
        f"{prefix} {context.spoken_number(str(day))} "
        f"tháng {context.spoken_number(str(month))} "
        f"năm {context.spoken_number(str(year))}"
    )


def _currency_per_kg_words(
    match: re.Match[str],
    context: RuleContext,
) -> str:
    token = match.group("currency")
    unit_override = None if token.upper() == "VND" else "đồng"
    amount = context.currency_words(
        match.group("number"),
        "VND",
        unit_override,
    )
    return f"{amount} một ki lô gam"


def apply(text: str, context: RuleContext) -> str:
    text = _OR_RE.sub(
        lambda match: _preserve_initial_case(match.group(0), "anh hoặc chị"),
        text,
    )
    text = _LEGAL_DOCUMENT_RE.sub(
        lambda match: _legal_document_words(match, context),
        text,
    )
    text = _DATE_RE.sub(lambda match: _date_words(match, context), text)
    text = _CLOCK_RE.sub(
        lambda match: context.clock_words(
            int(match.group("hour")),
            int(match.group("minute")),
            None,
        ),
        text,
    )
    text = _CURRENCY_PER_KG_RE.sub(
        lambda match: _currency_per_kg_words(match, context),
        text,
    )
    return _PERCENT_PER_YEAR_RE.sub(
        lambda match: (
            f"{context.spoken_number(match.group('number'))} "
            "phần trăm một năm"
        ),
        text,
    )
