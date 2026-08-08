from __future__ import annotations

from dataclasses import dataclass

from src import ctc_forced_alignment
from src.ctc_forced_alignment import (
    CtcLanguageSpan,
    _merge_words_crossed_by_edges,
    _prepare_ctc_transcript,
)
from src.piper.word_segmentation import WordSpan, icu_word_spans


def test_icu_word_spans_convert_utf16_offsets_to_python_indices() -> None:
    text = "😀Hello 世界"

    spans = icu_word_spans(text, "en-US")

    assert spans == [WordSpan(1, 6, "Hello"), WordSpan(7, 9, "世界")]
    assert all(text[span.start : span.end] == span.text for span in spans)


def test_prepare_ctc_transcript_preserves_mixed_language_source_offsets() -> None:
    text = "FIFA 在重庆举办比赛 and Saoirse attended."
    language_spans = [
        CtcLanguageSpan(0, 5, "en-US"),
        CtcLanguageSpan(5, 13, "zh-CN"),
        CtcLanguageSpan(13, len(text), "en-US"),
    ]

    prepared = _prepare_ctc_transcript(
        text,
        "en-US",
        star_frequency="segment",
        language_spans=language_spans,
    )

    assert prepared.languages == ["eng", "zho"]
    assert "".join(unit.text for unit in prepared.units) == text
    assert [unit.text for unit in prepared.units] == [
        "FIFA ",
        "在",
        "重庆",
        "举办",
        "比赛 ",
        "and ",
        "Saoirse ",
        "attended.",
    ]
    assert all(
        text[unit.source_start : unit.source_end] == unit.text
        for unit in prepared.units
    )
    assert [unit.language for unit in prepared.units] == [
        "eng",
        "zho",
        "zho",
        "zho",
        "zho",
        "eng",
        "eng",
        "eng",
    ]


def test_prepare_ctc_transcript_romanizes_each_language_span_once(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    original = ctc_forced_alignment._romanized_edges

    def record_call(text: str, language: str):
        calls.append((text, language))
        return original(text, language)

    monkeypatch.setattr(ctc_forced_alignment, "_romanized_edges", record_call)

    _prepare_ctc_transcript(
        "FIFA 在重庆举办比赛 and Saoirse attended.",
        "en-US",
        star_frequency="segment",
        language_spans=[
            CtcLanguageSpan(0, 5, "en-US"),
            CtcLanguageSpan(5, 13, "zh-CN"),
            CtcLanguageSpan(13, 34, "en-US"),
        ],
    )

    assert calls == [
        ("FIFA ", "eng"),
        ("在重庆举办比赛 ", "zho"),
        ("and Saoirse attended.", "eng"),
    ]


def test_prepare_ctc_transcript_keeps_punctuation_with_timestamp_units() -> None:
    text = "In conclusion, that is a good thing."

    prepared = _prepare_ctc_transcript(text, "en-US", star_frequency="segment")

    assert "".join(unit.text for unit in prepared.units) == text
    assert prepared.units[1].text == "conclusion, "
    assert prepared.units[-1].text == "thing."
    assert prepared.units[1].token == "c o n c l u s i o n"


@dataclass(frozen=True)
class _Edge:
    start: int
    end: int
    txt: str


def test_uroman_edges_remain_atomic_across_icu_boundaries() -> None:
    words = [WordSpan(0, 1, "a"), WordSpan(2, 3, "b")]

    groups = _merge_words_crossed_by_edges(words, [_Edge(0, 3, "ab")])

    assert groups == [[words[0], words[1]]]


def test_edge_star_frequency_has_only_outer_wildcards() -> None:
    prepared = _prepare_ctc_transcript(
        "One two.",
        "en-US",
        star_frequency="edges",
    )

    assert prepared.tokens == ["<star>", "o n e", "t w o", "<star>"]
    assert prepared.starred_text == ["<star>", "One ", "two.", "<star>"]
