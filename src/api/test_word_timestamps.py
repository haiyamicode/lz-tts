"""Word timestamps aligned on the spoken text land on the written text."""

import re

from .server import _source_word_timestamps, _word_timestamp


def aligned(spoken, pattern=r"\S+"):
    """One aligned unit per spoken word, word n spoken from n to n + 1 seconds."""
    return [
        {"source_start": m.start(), "source_end": m.end(), "start_seconds": float(n), "end_seconds": n + 1.0}
        for n, m in enumerate(re.finditer(pattern, spoken))
    ]


def timestamps(text, spoken, pattern=r"\S+"):
    words = _source_word_timestamps(text, spoken, aligned(spoken, pattern))
    return [(t["text"], t["start"], t["end"]) for t in (_word_timestamp(text, w) for w in words)]


def test_number_words_are_one_timestamp_on_the_number():
    assert timestamps("She was born in 1987 and moved.", "She was born in nineteen eighty-seven and moved.") == [
        ("She", 0, 1), ("was", 1, 2), ("born", 2, 3), ("in", 3, 4), ("1987", 4, 6), ("and", 6, 7), ("moved.", 7, 8),
    ]


def test_symbols_read_after_the_number_stay_on_its_token():
    assert timestamps("Prices rose 3.5% to $1,299.", "Prices rose three point five percent to one thousand, "
                      "two hundred and ninety-nine dollars.") == [
        ("Prices", 0, 1), ("rose", 1, 2), ("3.5%", 2, 6), ("to", 6, 7), ("$1,299.", 7, 14),
    ]


def test_unspaced_text_maps_each_number():
    assert timestamps("会议在3月15日开始。", "会议在三月十五日开始。", r".") == [
        ("会", 0, 1), ("议", 1, 2), ("在", 2, 3), ("3", 3, 4), ("月", 4, 5), ("15", 5, 7), ("日", 7, 8),
        ("开", 8, 9), ("始", 9, 10), ("。", 10, 11),
    ]
