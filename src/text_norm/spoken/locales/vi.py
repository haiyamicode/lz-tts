"""Vietnamese spoken-normalization vocabulary and grammar."""

from __future__ import annotations

import re

from ...vietnamese import vietnamese_number_words
from .base import SpokenLocale


_SCALES = ('', ' nghìn', ' triệu', ' tỷ')


def _number_words(value: int) -> str:
    # num2words reads an inner group below 100 as "lẻ hai mươi tư" (2024);
    # Vietnamese says "hai nghìn không trăm hai mươi tư".
    if value < 1000 or value >= 10**12:
        return vietnamese_number_words(value)
    groups = []
    while value:
        value, group = divmod(value, 1000)
        groups.append(group)
    words = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            continue
        if index == len(groups) - 1 or group >= 100:
            spoken = vietnamese_number_words(group)
        elif group >= 10:
            spoken = f'không trăm {vietnamese_number_words(group)}'
        else:
            spoken = f'không trăm lẻ {vietnamese_number_words(group)}'
        words.append(spoken + _SCALES[index])
    return ' '.join(words)


_NUMBER_WORDS = _number_words


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = f"{_NUMBER_WORDS(hour)} giờ"
    if minute not in (None, 0):
        result += f" {_NUMBER_WORDS(minute)}"
    return result


_ORDER_WORDS = {1: 'nhất', 4: 'tư'}  # "thứ nhất", "thứ hai", "thứ tư"
_PRIZE_WORDS = {1: 'nhất', 2: 'nhì', 4: 'tư'}  # "giải nhì", "hạng nhì"


def _cardinal_words(value: int, before: str, after: str) -> str:
    # The fourth month is "tháng tư"; "thứ 4" is "thứ tư", "hạng 1" is
    # "hạng nhất" and a prize won ("đạt giải 1") is "giải nhất" in any context.
    if value == 4 and re.search(r'tháng\s*$', before, re.IGNORECASE):
        return 'tư'
    if re.search(r'thứ\s*$', before, re.IGNORECASE):
        return _ORDER_WORDS.get(value) or _number_words(value)
    if re.search(r'(?:hạng|(?:đạt|giành|đoạt|được|nhận)\s+giải)\s*$', before, re.IGNORECASE):
        return _PRIZE_WORDS.get(value) or _number_words(value)
    return _number_words(value)


def _ordinal_words(value: int, before: str, after: str) -> str:
    # "giải 1" is "giải nhất" as a prize but "giải một bài toán" (solve) as a
    # count; other ordinals are cardinals.
    if re.search(r'giải\s*$', before, re.IGNORECASE):
        return _PRIZE_WORDS.get(value) or _number_words(value)
    return _cardinal_words(value, before, after)


LOCALE = SpokenLocale(
    language='vi',
    number_words=_NUMBER_WORDS,
    clock_words=_clock_words,
    digit_separator=' ',
    decimal_template='{integer} {decimal} {fraction}',
    negative_template='âm {number}',
    one_quantity_word=None,
    quantity_unit_separator=' ',
    grouping_separator='.',
    round_the_clock='hai mươi tư trên bảy',
    all_day='hai mươi tư trên hai mươi tư',
    all_day_hour_aliases=('giờ',),
    clock_suffix_aliases=(),
    clock_consumes_meridiem=False,
    carbon_dioxide='xê ô hai',
    tech_letter_words={'K': 'ca', 'D': 'đê', 'G': 'gi'},
    range_template='{start} đến {end}',
    decimal_word='phẩy',
    percent_word='phần trăm',
    currency_magnitudes={'nghìn': ('nghìn', ''), 'ngàn': ('ngàn', ''), 'triệu': ('triệu', ''), 'tỷ': ('tỷ', '')},
    currency_words={'USD': 'u ét đê',
     'EUR': 'euro',
     'GBP': 'bảng Anh',
     'JPY': 'yên',
     'KRW': 'won Hàn Quốc',
     'VND': 'Việt Nam đồng',
     'BRL': 'real Brazil',
     'CNY': 'nhân dân tệ',
     'INR': 'rupee Ấn Độ',
     'CHF': 'franc Thụy Sĩ',
     'CAD': 'đô la Canada',
     'AUD': 'đô la Úc'},
    currency_alias_words={
        '$': 'đô la',
        'US$': 'đô la',
        '₫': 'đồng',
        'đ': 'đồng',
        'Đ': 'đồng',
    },
    currency_singular_words={},
    currency_feminine_one=frozenset(),
    feminine_one_word=None,
    currency_attributive_nouns=frozenset(),
    currency_magnitude_one_words={},
    currency_connector_elision_vowels='',
    unit_words={'kg': 'ki lô gam',
     'g': 'gam',
     'km': 'ki lô mét',
     'mi': 'dặm',
     'm': 'mét',
     'cm': 'xen ti mét',
     'mm': 'mi li mét',
     '°c': 'độ C',
     'mah': 'mi li am pe giờ',
     'w': 'oát',
     'kw': 'ki lô oát',
     'gb': 'ghi ga bai',
     'mb': 'mê ga bai',
     'ghz': 'ghi ga héc',
     'mp': 'mê ga pixel'},
    unit_singular_words={},
    unit_feminine_one=frozenset(),
    unit_aliases={},
    rate_template='{quantity} {numerator} trên {denominator}',
    rate_denominator_words={
        'h': 'giờ',
        's': 'giây',
        'min': 'phút',
        'd': 'ngày',
    },
    rate_denominator_aliases={},
    ordinal_words=_ordinal_words,
    cardinal_words=_cardinal_words,
    decimal_fraction_number=True,
    bare_hour_clock=True,
    degree_words=('độ', 'độ'),
)
