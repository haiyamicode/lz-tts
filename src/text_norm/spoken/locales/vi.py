"""Vietnamese spoken-normalization vocabulary and grammar."""

from __future__ import annotations

from ...vietnamese import vietnamese_number_words
from .base import SpokenLocale

_NUMBER_WORDS = vietnamese_number_words


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = f"{_NUMBER_WORDS(hour)} giờ"
    if minute not in (None, 0):
        result += f" {_NUMBER_WORDS(minute)}"
    return result


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
)
