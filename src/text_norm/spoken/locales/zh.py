"""Chinese spoken-normalization vocabulary and grammar."""

from __future__ import annotations

import cn2an

from .base import SpokenLocale


def _number_words(value: int) -> str:
    return cn2an.an2cn(str(value))


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = f"{_number_words(hour)}点"
    if minute not in (None, 0):
        result += f"{_number_words(minute)}分"
    return result


LOCALE = SpokenLocale(
    language="zh",
    number_words=_number_words,
    clock_words=_clock_words,
    digit_separator="",
    decimal_template="{integer}{decimal}{fraction}",
    negative_template="负{number}",
    one_quantity_word=None,
    quantity_unit_separator="",
    grouping_separator=",",
    round_the_clock="二十四小时，每周七天",
    all_day="全天二十四小时",
    all_day_hour_aliases=("小时",),
    clock_suffix_aliases=(),
    clock_consumes_meridiem=False,
    carbon_dioxide="二氧化碳",
    tech_letter_words={"K": "K", "D": "D", "G": "G"},
    range_template="{start}到{end}",
    decimal_word="点",
    percent_word="百分比",
    currency_magnitudes={},
    currency_words={
        "USD": "美元",
        "EUR": "欧元",
        "GBP": "英镑",
        "JPY": "日元",
        "KRW": "韩元",
        "VND": "越南盾",
        "BRL": "巴西雷亚尔",
        "CNY": "人民币",
        "INR": "印度卢比",
        "CHF": "瑞士法郎",
        "CAD": "加元",
        "AUD": "澳元",
    },
    currency_alias_words={"¥": "元", "CN¥": "人民币"},
    currency_singular_words={},
    currency_feminine_one=frozenset(),
    feminine_one_word=None,
    currency_attributive_nouns=frozenset(),
    currency_magnitude_one_words={},
    currency_connector_elision_vowels="",
    unit_words={
        "kg": "千克",
        "g": "克",
        "km": "公里",
        "mi": "英里",
        "m": "米",
        "cm": "厘米",
        "mm": "毫米",
        "°c": "摄氏度",
        "mah": "毫安时",
        "w": "瓦",
        "kw": "千瓦",
        "gb": "吉字节",
        "mb": "兆字节",
        "ghz": "吉赫兹",
        "mp": "百万像素",
    },
    unit_singular_words={},
    unit_feminine_one=frozenset(),
    unit_aliases={},
    rate_template="每{denominator}{quantity}{numerator}",
    rate_denominator_words={
        "h": "小时",
        "s": "秒",
        "min": "分钟",
        "d": "天",
    },
    rate_denominator_aliases={},
)
