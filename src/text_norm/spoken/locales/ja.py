"""Japanese spoken-normalization vocabulary and grammar."""

from __future__ import annotations

from .base import SpokenLocale, make_number_words

_NUMBER_WORDS = make_number_words('ja')


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = f"{_NUMBER_WORDS(hour)}時"
    if minute not in (None, 0):
        result += f"{_NUMBER_WORDS(minute)}分"
    return result


LOCALE = SpokenLocale(
    language='ja',
    number_words=_NUMBER_WORDS,
    clock_words=_clock_words,
    digit_separator='',
    decimal_template='{integer}{decimal}{fraction}',
    negative_template='マイナス{number}',
    one_quantity_word=None,
    quantity_unit_separator='',
    grouping_separator=',',
    round_the_clock='二十四時間、週七日',
    all_day='二十四時間',
    all_day_hour_aliases=(),
    clock_suffix_aliases=(),
    clock_consumes_meridiem=False,
    carbon_dioxide='シーオーツー',
    tech_letter_words={'K': 'ケー', 'D': 'ディー', 'G': 'ジー'},
    range_template='{start}から{end}',
    decimal_word='点',
    percent_word='パーセント',
    currency_magnitudes={},
    currency_words={'USD': 'ドル',
     'EUR': 'ユーロ',
     'GBP': 'ポンド',
     'JPY': '円',
     'KRW': '韓国ウォン',
     'VND': 'ベトナムドン',
     'BRL': 'ブラジルレアル',
     'CNY': '人民元',
     'INR': 'インドルピー',
     'CHF': 'スイスフラン',
     'CAD': 'カナダドル',
     'AUD': '豪ドル'},
    currency_alias_words={},
    currency_singular_words={},
    currency_feminine_one=frozenset(),
    feminine_one_word=None,
    currency_attributive_nouns=frozenset(),
    currency_magnitude_one_words={},
    currency_connector_elision_vowels='',
    unit_words={'kg': 'キログラム',
     'g': 'グラム',
     'km': 'キロメートル',
     'mi': 'マイル',
     'm': 'メートル',
     'cm': 'センチメートル',
     'mm': 'ミリメートル',
     '°c': '度',
     'mah': 'ミリアンペア時',
     'w': 'ワット',
     'kw': 'キロワット',
     'gb': 'ギガバイト',
     'mb': 'メガバイト',
     'ghz': 'ギガヘルツ',
     'mp': 'メガピクセル'},
    unit_singular_words={},
    unit_feminine_one=frozenset(),
    unit_aliases={},
    rate_template='{denominator}あたり{quantity}{numerator}',
    rate_denominator_words={
        'h': '一時間',
        's': '一秒',
        'min': '一分',
        'd': '一日',
    },
    rate_denominator_aliases={},
)
