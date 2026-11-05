"""German spoken-normalization vocabulary and grammar."""

from __future__ import annotations

import re

from num2words import num2words

from .base import SpokenLocale, make_number_words

_NUMBER_WORDS = make_number_words('de')


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = f"{_NUMBER_WORDS(hour)} Uhr"
    if minute not in (None, 0):
        result += f" {_NUMBER_WORDS(minute)}"
    return result


_MONTHS = frozenset({
    'januar', 'jänner', 'februar', 'märz', 'april', 'mai', 'juni', 'juli',
    'august', 'september', 'oktober', 'november', 'dezember',
})
_NOMINATIVE_ARTICLES = frozenset({
    'der', 'die', 'das', 'jede', 'jeder', 'jedes', 'diese', 'dieser', 'dieses',
})
_DATIVE_ARTICLES = frozenset({
    'am', 'im', 'vom', 'zum', 'zur', 'beim', 'den', 'dem', 'des', 'einem',
    'einen', 'eines', 'jeden', 'jedem', 'diesen', 'diesem',
})
_DATIVE_PREPOSITIONS = frozenset({
    'in', 'an', 'auf', 'aus', 'bei', 'mit', 'nach', 'von', 'zu', 'vor', 'seit',
})


def _year_words(value: int) -> str:
    return num2words(value, lang='de', to='year')


def _ordinal_words(value: int, before: str, after: str) -> str:
    stem = num2words(value, lang='de', to='ordinal')
    words = re.findall(r'\w+', before.lower())[-2:]
    previous = words[-1] if words else ''
    if previous == 'der' and len(words) == 2 and words[0] in _DATIVE_PREPOSITIONS:
        return f'{stem}n'
    if previous in _NOMINATIVE_ARTICLES:
        return stem
    if previous in _DATIVE_ARTICLES:
        return f'{stem}n'
    if previous == 'ein':
        return f'{stem}r'
    following = re.match(r'\W*(\w+)', after)
    if following and following.group(1).lower() in _MONTHS:
        return f'{stem}r'
    return stem


LOCALE = SpokenLocale(
    language='de',
    number_words=_NUMBER_WORDS,
    clock_words=_clock_words,
    digit_separator=' ',
    decimal_template='{integer} {decimal} {fraction}',
    negative_template='minus {number}',
    one_quantity_word='ein',
    quantity_unit_separator=' ',
    grouping_separator='.',
    round_the_clock='vierundzwanzig Stunden am Tag, sieben Tage die Woche',
    all_day='vierundzwanzig Stunden am Tag',
    all_day_hour_aliases=('Stunde', 'Stunden'),
    clock_suffix_aliases=('Uhr',),
    clock_consumes_meridiem=False,
    carbon_dioxide='C O zwei',
    tech_letter_words={'K': 'K', 'D': 'D', 'G': 'G'},
    range_template='{start} bis {end}',
    decimal_word='Komma',
    percent_word='Prozent',
    currency_magnitudes={'tausend': ('tausend', ''),
     'million': ('Million', ''),
     'millionen': ('Millionen', ''),
     'milliarde': ('Milliarde', ''),
     'milliarden': ('Milliarden', '')},
    currency_words={'USD': 'Dollar',
     'EUR': 'Euro',
     'GBP': 'Pfund',
     'JPY': 'Yen',
     'KRW': 'südkoreanische Won',
     'VND': 'vietnamesische Dong',
     'BRL': 'brasilianische Real',
     'CNY': 'chinesische Yuan',
     'INR': 'indische Rupien',
     'CHF': 'Schweizer Franken',
     'CAD': 'kanadische Dollar',
     'AUD': 'australische Dollar'},
    currency_alias_words={},
    currency_singular_words={},
    currency_feminine_one=frozenset(),
    feminine_one_word='eine',
    currency_attributive_nouns=frozenset(),
    currency_magnitude_one_words={
        'tausend': 'eintausend',
        'million': 'eine Million',
        'millionen': 'eine Million',
        'milliarde': 'eine Milliarde',
        'milliarden': 'eine Milliarde',
    },
    currency_connector_elision_vowels='',
    unit_words={'kg': 'Kilogramm',
     'g': 'Gramm',
     'km': 'Kilometer',
     'mi': 'Meilen',
     'm': 'Meter',
     'cm': 'Zentimeter',
     'mm': 'Millimeter',
     '°c': 'Grad Celsius',
     'mah': 'Milliamperestunden',
     'w': 'Watt',
     'kw': 'Kilowatt',
     'gb': 'Gigabyte',
     'mb': 'Megabyte',
     'ghz': 'Gigahertz',
     'mp': 'Megapixel'},
    unit_singular_words={'mi': 'Meile'},
    unit_feminine_one=frozenset({'mi'}),
    unit_aliases={},
    rate_template='{quantity} {numerator} pro {denominator}',
    rate_denominator_words={
        'h': 'Stunde',
        's': 'Sekunde',
        'min': 'Minute',
        'd': 'Tag',
    },
    rate_denominator_aliases={
        'Stunden': 'h',
        'Sekunden': 's',
        'Minuten': 'min',
        'Tage': 'd',
    },
    year_words=_year_words,
    ordinal_words=_ordinal_words,
    ordinal_period=True,
    degree_words=('Grad', 'Grad'),
)
