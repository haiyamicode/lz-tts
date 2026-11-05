"""Spanish spoken-normalization vocabulary and grammar."""

from __future__ import annotations

import re

from num2words import num2words

from .base import SpokenLocale, make_number_words

_NUMBER_WORDS = make_number_words('es')


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    # Hora is feminine (una hora, veintiuna horas), minuto masculine before
    # the noun (un minuto, veintiún minutos).
    if hour == 1:
        result = 'una hora'
    else:
        result = re.sub(r'uno$', 'una', _NUMBER_WORDS(hour)) + ' horas'
    if minute == 1:
        result += ' y un minuto'
    elif minute:
        words = re.sub(r'veintiuno$', 'veintiún', _NUMBER_WORDS(minute))
        result += f" y {re.sub(r'uno$', 'un', words)} minutos"
    return result


def _sign_ordinal_words(value: int, before: str, after: str) -> str:
    words = num2words(value, lang='es', to='ordinal')
    # Primero and tercero drop the -o before a noun: el 1º piso, el 3º puesto.
    if words.endswith(('primero', 'tercero')) and re.match(
        r'\s*(?!(?:de|del|y|o)\b)[^\W\d_]', after, re.IGNORECASE
    ):
        words = words[:-1]
    return words


def _ordinal_words(value: int, before: str, after: str) -> str:
    # Only the first day of a month is an ordinal; the rest read as cardinals.
    return 'primero' if value == 1 else _NUMBER_WORDS(value)


LOCALE = SpokenLocale(
    language='es',
    number_words=_NUMBER_WORDS,
    clock_words=_clock_words,
    digit_separator=' ',
    decimal_template='{integer} {decimal} {fraction}',
    negative_template='menos {number}',
    one_quantity_word='un',
    quantity_unit_separator=' ',
    grouping_separator='.',
    round_the_clock='veinticuatro horas al día, siete días a la semana',
    all_day='veinticuatro horas al día',
    all_day_hour_aliases=('hora', 'horas'),
    clock_suffix_aliases=(),
    clock_consumes_meridiem=False,
    carbon_dioxide='ce o dos',
    tech_letter_words={'K': 'ka', 'D': 'de', 'G': 'ge'},
    range_template='{start} a {end}',
    decimal_word='coma',
    percent_word='por ciento',
    currency_magnitudes={'mil': ('mil', ''),
     'millón': ('millón', 'de'),
     'millones': ('millones', 'de'),
     'mil millones': ('mil millones', 'de')},
    currency_words={'USD': 'dólares',
     'EUR': 'euros',
     'GBP': 'libras',
     'JPY': 'yenes',
     'KRW': 'wones surcoreanos',
     'VND': 'dongs vietnamitas',
     'BRL': 'reales brasileños',
     'CNY': 'yuanes chinos',
     'INR': 'rupias indias',
     'CHF': 'francos suizos',
     'CAD': 'dólares canadienses',
     'AUD': 'dólares australianos'},
    currency_alias_words={},
    currency_singular_words={'USD': 'dólar',
     'EUR': 'euro',
     'GBP': 'libra',
     'JPY': 'yen',
     'KRW': 'won surcoreano',
     'VND': 'dong vietnamita',
     'BRL': 'real brasileño',
     'CNY': 'yuan chino',
     'INR': 'rupia india',
     'CHF': 'franco suizo',
     'CAD': 'dólar canadiense',
     'AUD': 'dólar australiano'},
    currency_feminine_one=frozenset({'GBP', 'INR'}),
    feminine_one_word='una',
    currency_attributive_nouns=frozenset(),
    currency_magnitude_one_words={},
    currency_connector_elision_vowels='',
    unit_words={'kg': 'kilogramos',
     'g': 'gramos',
     'km': 'kilómetros',
     'mi': 'millas',
     'm': 'metros',
     'cm': 'centímetros',
     'mm': 'milímetros',
     '°c': 'grados Celsius',
     'mah': 'miliamperios hora',
     'w': 'vatios',
     'kw': 'kilovatios',
     'gb': 'gigabytes',
     'mb': 'megabytes',
     'ghz': 'gigahercios',
     'mp': 'megapíxeles'},
    unit_singular_words={'kg': 'kilogramo',
     'g': 'gramo',
     'km': 'kilómetro',
     'mi': 'milla',
     'm': 'metro',
     'cm': 'centímetro',
     'mm': 'milímetro',
     '°c': 'grado Celsius',
     'mah': 'miliamperio hora',
     'w': 'vatio',
     'kw': 'kilovatio',
     'gb': 'gigabyte',
     'mb': 'megabyte',
     'ghz': 'gigahercio',
     'mp': 'megapíxel'},
    unit_feminine_one=frozenset({'mi'}),
    unit_aliases={},
    rate_template='{quantity} {numerator} por {denominator}',
    rate_denominator_words={
        'h': 'hora',
        's': 'segundo',
        'min': 'minuto',
        'd': 'día',
    },
    rate_denominator_aliases={
        'horas': 'h',
        'segundos': 's',
        'minutos': 'min',
        'días': 'd',
    },
    ordinal_words=_ordinal_words,
    decimal_fraction_number=True,
    bare_hour_clock=True,
    degree_words=('grado', 'grados'),
    sign_ordinal_words=_sign_ordinal_words,
)
