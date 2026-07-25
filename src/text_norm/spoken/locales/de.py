"""German spoken-normalization vocabulary and grammar."""

from __future__ import annotations

from .base import SpokenLocale, make_number_words

_NUMBER_WORDS = make_number_words('de')


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = f"{_NUMBER_WORDS(hour)} Uhr"
    if minute not in (None, 0):
        result += f" {_NUMBER_WORDS(minute)}"
    return result


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
)
