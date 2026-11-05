"""Chinese spoken-normalization vocabulary and grammar."""

from __future__ import annotations

import re

import cn2an

from .base import SpokenLocale


def _number_words(value: int) -> str:
    return cn2an.an2cn(str(value))


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = "两点" if hour == 2 else f"{_number_words(hour)}点"
    if minute not in (None, 0):
        result += f"{_number_words(minute)}分"
    return result


_DIGITS = "零一二三四五六七八九"
# Measure words and magnitudes that take 两 instead of 二.
_LIANG_FOLLOWERS = (
    "个", "本", "只", "天", "次", "位", "件", "张", "条", "种", "部", "辆", "杯",
    "家", "份", "点", "岁", "年", "周", "倍", "万", "千", "百", "亿", "斤",
    "公斤", "公里", "米", "小时", "分钟", "秒钟", "星期", "礼拜", "台", "双",
    "对", "块", "片", "间", "所", "座", "棵", "头", "匹", "支", "首", "篇",
    "门", "节", "层", "趟", "遍", "口", "名", "颗", "瓶", "碗", "根", "套",
    # Traditional script (Cantonese and Taiwan text).
    "個", "隻", "張", "條", "種", "輛", "歲", "週", "萬", "億", "點", "雙",
    "對", "塊", "間", "頭", "門", "節", "層", "顆", "架", "碟", "分鐘", "秒鐘",
    "小時", "禮拜", "臺",
)


def _year_words(value: int) -> str:
    # Years are read digit by digit: 一九八二年, 二零二四年.
    digits = str(value)
    if len(digits) < 4:
        return _number_words(value)
    return "".join(_DIGITS[int(digit)] for digit in digits)


def _cardinal_words(value: int, before: str, after: str) -> str:
    words = _number_words(value)
    if before.endswith("第"):
        return words
    following = after.lstrip()
    if (
        value == 2
        and following.startswith(_LIANG_FOLLOWERS)
        and not following.startswith(("年级", "年級"))
    ):
        return "两"
    if re.match(r"二[千万亿]", words):
        return "两" + words[1:]
    return words


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
    year_words=_year_words,
    cardinal_words=_cardinal_words,
    digit_group_separator='，',
    degree_words=('度', '度'),
)
