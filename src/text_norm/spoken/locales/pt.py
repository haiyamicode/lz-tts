"""Portuguese spoken-normalization vocabulary and grammar."""

from __future__ import annotations

import re

from num2words import num2words

from .base import SpokenLocale, make_number_words

_NUMBER_WORDS = make_number_words('pt')


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    # Hora is feminine: uma hora, duas horas, vinte e duas horas.
    if hour == 1:
        result = 'uma hora'
    else:
        words = re.sub(r'\bum$', 'uma', _NUMBER_WORDS(hour))
        result = re.sub(r'\bdois$', 'duas', words) + ' horas'
    if minute not in (None, 0):
        result += f" e {_NUMBER_WORDS(minute)} minutos"
    return result


def _sign_ordinal_words(value: int, before: str, after: str) -> str:
    return num2words(value, lang='pt', to='ordinal')


def _ordinal_words(value: int, before: str, after: str) -> str:
    # Only the first day of a month is an ordinal; the rest read as cardinals.
    return 'primeiro' if value == 1 else _NUMBER_WORDS(value)


LOCALE = SpokenLocale(
    language='pt',
    number_words=_NUMBER_WORDS,
    clock_words=_clock_words,
    digit_separator=' ',
    decimal_template='{integer} {decimal} {fraction}',
    negative_template='menos {number}',
    one_quantity_word=None,
    quantity_unit_separator=' ',
    grouping_separator='.',
    round_the_clock='vinte e quatro horas por dia, sete dias por semana',
    all_day='vinte e quatro horas por dia',
    all_day_hour_aliases=('hora', 'horas'),
    clock_suffix_aliases=(),
    clock_consumes_meridiem=False,
    carbon_dioxide='cê o dois',
    tech_letter_words={'K': 'cá', 'D': 'dê', 'G': 'gê'},
    range_template='{start} a {end}',
    decimal_word='vírgula',
    percent_word='por cento',
    currency_magnitudes={'mil': ('mil', ''),
     'milhão': ('milhão', 'de'),
     'milhões': ('milhões', 'de'),
     'bilhão': ('bilhão', 'de'),
     'bilhões': ('bilhões', 'de')},
    currency_words={'USD': 'dólares',
     'EUR': 'euros',
     'GBP': 'libras',
     'JPY': 'ienes',
     'KRW': 'wons sul-coreanos',
     'VND': 'dongs vietnamitas',
     'BRL': 'reais brasileiros',
     'CNY': 'yuans chineses',
     'INR': 'rúpias indianas',
     'CHF': 'francos suíços',
     'CAD': 'dólares canadenses',
     'AUD': 'dólares australianos'},
    currency_alias_words={},
    currency_singular_words={'USD': 'dólar',
     'EUR': 'euro',
     'GBP': 'libra',
     'JPY': 'iene',
     'KRW': 'won sul-coreano',
     'VND': 'dong vietnamita',
     'BRL': 'real brasileiro',
     'CNY': 'yuan chinês',
     'INR': 'rúpia indiana',
     'CHF': 'franco suíço',
     'CAD': 'dólar canadense',
     'AUD': 'dólar australiano'},
    currency_feminine_one=frozenset({'GBP', 'INR'}),
    feminine_one_word='uma',
    currency_attributive_nouns=frozenset(),
    currency_magnitude_one_words={},
    currency_connector_elision_vowels='',
    unit_words={'kg': 'quilogramas',
     'g': 'gramas',
     'km': 'quilômetros',
     'mi': 'milhas',
     'm': 'metros',
     'cm': 'centímetros',
     'mm': 'milímetros',
     '°c': 'graus Celsius',
     'mah': 'miliampères-hora',
     'w': 'watts',
     'kw': 'quilowatts',
     'gb': 'gigabytes',
     'mb': 'megabytes',
     'ghz': 'gigahertz',
     'mp': 'megapixels'},
    unit_singular_words={'kg': 'quilograma',
     'g': 'grama',
     'km': 'quilômetro',
     'mi': 'milha',
     'm': 'metro',
     'cm': 'centímetro',
     'mm': 'milímetro',
     '°c': 'grau Celsius',
     'mah': 'miliampère-hora',
     'w': 'watt',
     'kw': 'quilowatt',
     'gb': 'gigabyte',
     'mb': 'megabyte',
     'ghz': 'gigahertz',
     'mp': 'megapixel'},
    unit_feminine_one=frozenset({'mi'}),
    unit_aliases={},
    rate_template='{quantity} {numerator} por {denominator}',
    rate_denominator_words={
        'h': 'hora',
        's': 'segundo',
        'min': 'minuto',
        'd': 'dia',
    },
    rate_denominator_aliases={
        'horas': 'h',
        'segundos': 's',
        'minutos': 'min',
        'dias': 'd',
    },
    ordinal_words=_ordinal_words,
    decimal_fraction_number=True,
    bare_hour_clock=True,
    degree_words=('grau', 'graus'),
    sign_ordinal_words=_sign_ordinal_words,
)
