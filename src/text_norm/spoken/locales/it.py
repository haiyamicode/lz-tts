"""Italian spoken-normalization vocabulary and grammar."""

from __future__ import annotations

from num2words import num2words

from .base import SpokenLocale, make_number_words

_NUM2WORDS = make_number_words('it')


def _NUMBER_WORDS(value: int) -> str:
    # num2words elides "cento" + "diciotto" (118, 2118, 118000) to "centodicotto".
    return _NUM2WORDS(value).replace('dicott', 'diciott')


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    result = _NUMBER_WORDS(hour)
    if minute not in (None, 0):
        result += f" e {_NUMBER_WORDS(minute)}"
    return result


def _sign_ordinal_words(value: int, before: str, after: str) -> str:
    # num2words: 118º "centodicottesimo".
    return num2words(value, lang='it', to='ordinal').replace('dicott', 'diciott')


def _ordinal_words(value: int, before: str, after: str) -> str:
    # Only the first day of a month is an ordinal; the rest read as cardinals.
    return 'primo' if value == 1 else _NUMBER_WORDS(value)


LOCALE = SpokenLocale(
    language='it',
    number_words=_NUMBER_WORDS,
    clock_words=_clock_words,
    digit_separator=' ',
    decimal_template='{integer} {decimal} {fraction}',
    negative_template='meno {number}',
    one_quantity_word='un',
    quantity_unit_separator=' ',
    grouping_separator='.',
    round_the_clock='ventiquattro ore su ventiquattro, sette giorni su sette',
    all_day='ventiquattro ore su ventiquattro',
    all_day_hour_aliases=('ora', 'ore'),
    clock_suffix_aliases=(),
    clock_consumes_meridiem=False,
    carbon_dioxide='ci o due',
    tech_letter_words={'K': 'cappa', 'D': 'di', 'G': 'gi'},
    range_template='{start} a {end}',
    decimal_word='virgola',
    percent_word='per cento',
    currency_magnitudes={'mila': ('mila', ''),
     'milione': ('milione', 'di'),
     'milioni': ('milioni', 'di'),
     'miliardo': ('miliardo', 'di'),
     'miliardi': ('miliardi', 'di')},
    currency_words={'USD': 'dollari',
     'EUR': 'euro',
     'GBP': 'sterline',
     'JPY': 'yen',
     'KRW': 'won sudcoreani',
     'VND': 'dong vietnamiti',
     'BRL': 'real brasiliani',
     'CNY': 'yuan cinesi',
     'INR': 'rupie indiane',
     'CHF': 'franchi svizzeri',
     'CAD': 'dollari canadesi',
     'AUD': 'dollari australiani'},
    currency_alias_words={},
    currency_singular_words={'USD': 'dollaro',
     'EUR': 'euro',
     'GBP': 'sterlina',
     'JPY': 'yen',
     'KRW': 'won sudcoreano',
     'VND': 'dong vietnamita',
     'BRL': 'real brasiliano',
     'CNY': 'yuan cinese',
     'INR': 'rupia indiana',
     'CHF': 'franco svizzero',
     'CAD': 'dollaro canadese',
     'AUD': 'dollaro australiano'},
    currency_feminine_one=frozenset({'GBP', 'INR'}),
    feminine_one_word='una',
    currency_attributive_nouns=frozenset(),
    currency_magnitude_one_words={},
    currency_connector_elision_vowels='',
    unit_words={'kg': 'chilogrammi',
     'g': 'grammi',
     'km': 'chilometri',
     'mi': 'miglia',
     'm': 'metri',
     'cm': 'centimetri',
     'mm': 'millimetri',
     '°c': 'gradi Celsius',
     'mah': 'milliamperora',
     'w': 'watt',
     'kw': 'chilowatt',
     'gb': 'gigabyte',
     'mb': 'megabyte',
     'ghz': 'gigahertz',
     'mp': 'megapixel'},
    unit_singular_words={'kg': 'chilogrammo',
     'g': 'grammo',
     'km': 'chilometro',
     'mi': 'miglio',
     'm': 'metro',
     'cm': 'centimetro',
     'mm': 'millimetro',
     '°c': 'grado Celsius',
     'mah': 'milliamperora',
     'w': 'watt',
     'kw': 'chilowatt',
     'gb': 'gigabyte',
     'mb': 'megabyte',
     'ghz': 'gigahertz',
     'mp': 'megapixel'},
    unit_feminine_one=frozenset(),
    unit_aliases={},
    rate_template='{quantity} {numerator} per {denominator}',
    rate_denominator_words={
        'h': 'ora',
        's': 'secondo',
        'min': 'minuto',
        'd': 'giorno',
    },
    rate_denominator_aliases={
        'ore': 'h',
        'secondi': 's',
        'minuti': 'min',
        'giorni': 'd',
    },
    ordinal_words=_ordinal_words,
    decimal_fraction_number=True,
    degree_words=('grado', 'gradi'),
    sign_ordinal_words=_sign_ordinal_words,
)
