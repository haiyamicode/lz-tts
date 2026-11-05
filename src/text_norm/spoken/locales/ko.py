"""Korean spoken-normalization vocabulary and grammar."""

from __future__ import annotations

from .base import SpokenLocale, make_number_words

_NUMBER_WORDS = make_number_words('ko')
_NATIVE_UNITS = ('', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉')
_NATIVE_TENS = ('', '열', '스물', '서른', '마흔', '쉰', '예순', '일흔', '여든', '아흔')
# Counters that take native numbers (세 개, 다섯 명). 번 is left out: 세 번
# (three times) but 삼 번 (number three).
_NATIVE_COUNTERS = (
    '개', '명', '사람', '시', '살', '마리', '잔', '권', '장', '대', '병', '그릇',
    '달', '군데', '가지', '벌', '켤레', '송이', '그루', '채', '척', '통', '줄',
    '쌍', '곳',
)
# Sino-Korean words that start like a native counter (삼 개월, 오 대학).
_SINO_COUNTER_WORDS = ('개월', '시즌', '대학', '대회', '장면', '줄기')


def _native_words(value: int) -> str:
    tens, units = divmod(value, 10)
    if tens == 2 and units == 0:
        return '스무'
    return _NATIVE_TENS[tens] + _NATIVE_UNITS[units]


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = f"{_native_words(hour) if hour else _NUMBER_WORDS(hour)}시"
    if minute not in (None, 0):
        result += f" {_NUMBER_WORDS(minute)}분"
    return result


def _cardinal_words(value: int, before: str, after: str) -> str:
    following = after.lstrip()
    if following.startswith('번째') and 0 < value < 100:
        return '첫' if value == 1 else _native_words(value)
    if following.startswith('월') and value in (6, 10):
        return '유' if value == 6 else '시'
    if (
        0 < value < 100
        and following.startswith(_NATIVE_COUNTERS)
        and not following.startswith(_SINO_COUNTER_WORDS)
    ):
        return _native_words(value)
    return _NUMBER_WORDS(value)


LOCALE = SpokenLocale(
    language='ko',
    number_words=_NUMBER_WORDS,
    clock_words=_clock_words,
    digit_separator=' ',
    decimal_template='{integer} {decimal} {fraction}',
    negative_template='마이너스 {number}',
    one_quantity_word=None,
    quantity_unit_separator=' ',
    grouping_separator=',',
    round_the_clock='하루 이십사 시간, 주 칠일',
    all_day='하루 이십사 시간',
    all_day_hour_aliases=(),
    clock_suffix_aliases=(),
    clock_consumes_meridiem=False,
    carbon_dioxide='씨오투',
    tech_letter_words={'K': '케이', 'D': '디', 'G': '지'},
    # 에서 also reads before a counter: 천구백구십에서 천구백구십오 년.
    range_template='{start}에서 {end}',
    decimal_word='점',
    percent_word='퍼센트',
    currency_magnitudes={},
    currency_words={'USD': '달러',
     'EUR': '유로',
     'GBP': '파운드',
     'JPY': '엔',
     'KRW': '원',
     'VND': '베트남 동',
     'BRL': '브라질 헤알',
     'CNY': '중국 위안',
     'INR': '인도 루피',
     'CHF': '스위스 프랑',
     'CAD': '캐나다 달러',
     'AUD': '호주 달러'},
    currency_alias_words={},
    currency_singular_words={},
    currency_feminine_one=frozenset(),
    feminine_one_word=None,
    currency_attributive_nouns=frozenset(),
    currency_magnitude_one_words={},
    currency_connector_elision_vowels='',
    unit_words={'kg': '킬로그램',
     'g': '그램',
     'km': '킬로미터',
     'mi': '마일',
     'm': '미터',
     'cm': '센티미터',
     'mm': '밀리미터',
     '°c': '도',
     'mah': '밀리암페어시',
     'w': '와트',
     'kw': '킬로와트',
     'gb': '기가바이트',
     'mb': '메가바이트',
     'ghz': '기가헤르츠',
     'mp': '메가픽셀'},
    unit_singular_words={},
    unit_feminine_one=frozenset(),
    unit_aliases={},
    rate_template='{denominator}당 {quantity} {numerator}',
    rate_denominator_words={
        'h': '시간',
        's': '초',
        'min': '분',
        'd': '하루',
    },
    rate_denominator_aliases={},
    cardinal_words=_cardinal_words,
    code_zero_word='공',
    degree_words=('도', '도'),
)
