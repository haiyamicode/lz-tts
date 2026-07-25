"""English spoken-normalization vocabulary and grammar."""

from __future__ import annotations

from .base import SpokenLocale, make_number_words

_NUMBER_WORDS = make_number_words('en')


def _clock_words(hour: int, minute: int | None, meridiem: str | None) -> str:
    normalized_meridiem = (
        meridiem.lower().replace(' ', '').replace('.', '') if meridiem else None
    )
    display_hour = hour
    if normalized_meridiem:
        display_hour = 12 if hour == 0 else hour - 12 if hour > 12 else hour
    result = _NUMBER_WORDS(display_hour)
    if minute not in (None, 0):
        if minute < 10:
            result += ' oh'
        result += f" {_NUMBER_WORDS(minute)}"
    if normalized_meridiem:
        result += ' a.m.' if normalized_meridiem == 'am' else ' p.m.'
    return result


LOCALE = SpokenLocale(
    language='en',
    number_words=_NUMBER_WORDS,
    clock_words=_clock_words,
    digit_separator=' ',
    decimal_template='{integer} {decimal} {fraction}',
    negative_template='minus {number}',
    one_quantity_word=None,
    quantity_unit_separator=' ',
    grouping_separator=',',
    round_the_clock='twenty four seven',
    all_day='twenty four hours a day',
    all_day_hour_aliases=('hour', 'hours'),
    clock_suffix_aliases=(),
    clock_consumes_meridiem=True,
    carbon_dioxide='C O two',
    tech_letter_words={'K': 'K', 'D': 'D', 'G': 'G'},
    range_template='{start} to {end}',
    decimal_word='point',
    percent_word='percent',
    currency_magnitudes={'thousand': ('thousand', ''),
     'million': ('million', ''),
     'billion': ('billion', ''),
     'trillion': ('trillion', '')},
    currency_words={'USD': 'dollars',
     'EUR': 'euros',
     'GBP': 'pounds',
     'JPY': 'yen',
     'KRW': 'South Korean won',
     'VND': 'Vietnamese dong',
     'BRL': 'Brazilian reais',
     'CNY': 'Chinese yuan',
     'INR': 'Indian rupees',
     'CHF': 'Swiss francs',
     'CAD': 'Canadian dollars',
     'AUD': 'Australian dollars'},
    currency_alias_words={},
    currency_singular_words={'USD': 'dollar',
     'EUR': 'euro',
     'GBP': 'pound',
     'JPY': 'yen',
     'KRW': 'South Korean won',
     'VND': 'Vietnamese dong',
     'BRL': 'Brazilian real',
     'CNY': 'Chinese yuan',
     'INR': 'Indian rupee',
     'CHF': 'Swiss franc',
     'CAD': 'Canadian dollar',
     'AUD': 'Australian dollar'},
    currency_feminine_one=frozenset(),
    feminine_one_word=None,
    currency_attributive_nouns=frozenset({'grant', 'investment', 'bond'}),
    currency_magnitude_one_words={},
    currency_connector_elision_vowels='',
    unit_words={'kg': 'kilograms',
     'g': 'grams',
     'km': 'kilometers',
     'mi': 'miles',
     'm': 'meters',
     'cm': 'centimeters',
     'mm': 'millimeters',
     '°c': 'degrees Celsius',
     'mah': 'milliamp hours',
     'w': 'watts',
     'kw': 'kilowatts',
     'gb': 'gigabytes',
     'mb': 'megabytes',
     'ghz': 'gigahertz',
     'mp': 'megapixels'},
    unit_singular_words={'kg': 'kilogram',
     'g': 'gram',
     'km': 'kilometer',
     'mi': 'mile',
     'm': 'meter',
     'cm': 'centimeter',
     'mm': 'millimeter',
     '°c': 'degree Celsius',
     'mah': 'milliamp hour',
     'w': 'watt',
     'kw': 'kilowatt',
     'gb': 'gigabyte',
     'mb': 'megabyte',
     'ghz': 'gigahertz',
     'mp': 'megapixel'},
    unit_feminine_one=frozenset(),
    unit_aliases={
        'kilometre': 'km',
        'kilometres': 'km',
        'metre': 'm',
        'metres': 'm',
        'centimetre': 'cm',
        'centimetres': 'cm',
        'millimetre': 'mm',
        'millimetres': 'mm',
    },
    rate_template='{quantity} {numerator} per {denominator}',
    rate_denominator_words={
        'h': 'hour',
        's': 'second',
        'min': 'minute',
        'd': 'day',
    },
    rate_denominator_aliases={
        'hr': 'h',
        'hrs': 'h',
        'hours': 'h',
        'sec': 's',
        'secs': 's',
        'seconds': 's',
        'mins': 'min',
        'minutes': 'min',
        'days': 'd',
    },
)
