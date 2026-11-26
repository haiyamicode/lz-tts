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


def test_numeral_characters_are_aligned_as_their_syllables() -> None:
    prepared = _prepare_ctc_transcript("会议在三月十五日开始，地点是三零二房间。", "zh-CN", star_frequency="segment")

    assert [(unit.text, unit.token) for unit in prepared.units][2:11] == [
        ("三月", "s a n y u e"),
        ("十五", "s h i w u"),
        ("日", "r i"),
        ("开始，", "k a i s h i"),
        ("地点", "d e d i a n"),
        ("是", "s h i"),
        ("三", "s a n"),
        ("零", "l i n g"),
        ("二", "e r"),
    ]


def test_token_pieces_split_letters_over_source_characters():
    from src.ctc_forced_alignment import _prepare_ctc_transcript

    def pieces(text, language):
        units = _prepare_ctc_transcript(text, language, star_frequency="segment").units
        return [[(text[start:end], letters) for start, end, letters in unit.pieces] for unit in units]

    assert pieces("naïve cat", "en") == [[("n", 1), ("a", 1), ("ï", 1), ("v", 1), ("e", 1)], [("c", 1), ("a", 1), ("t", 1)]]
    assert pieces("今天三月", "zh") == [[("今", 3), ("天", 4)], [("三", 3), ("月", 3)]]
    assert pieces("안녕", "ko") == [[("안", 2), ("녕", 6)]]


def test_padding_keeps_the_cpp_aligner_in_bounds_and_its_path_unchanged() -> None:
    # ctc_forced_aligner's C++ under-sizes its back pointers when the audio barely fits the
    # text; the overflow corrupted the heap (garbage paths, SIGABRT) on short TTS takes.
    import numpy as np
    import torch
    from ctc_forced_aligner.alignment_utils import forced_align

    rng = np.random.default_rng(0)
    padded = 0
    for _ in range(200):
        targets = [int(x) for x in rng.integers(1, 28, int(rng.integers(3, 60)))]
        repeats = sum(a == b for a, b in zip(targets, targets[1:]))
        frames = len(targets) + repeats + int(rng.integers(0, 40))
        emissions = torch.cat(
            [torch.log_softmax(torch.randn(frames, 27) * 3, -1), torch.zeros(frames, 1)], 1
        )
        padding = ctc_forced_alignment._safe_ctc_padding(targets, frames)
        assert not ctc_forced_alignment._ctc_backpointer_overflows(targets, frames + padding)
        if not padding:
            continue
        padded += 1
        pad = torch.full((padding, 28), float("-inf"))
        pad[:, 0] = 0.0
        path, _ = forced_align(torch.cat([emissions, pad]).numpy()[None], np.array([targets]), blank=0)
        # Every label, in order, on the real frames (CTC merges a run of one label).
        merged = [x for i, x in enumerate(path[0, :frames]) if x and (i == 0 or path[0, i - 1] != x)]
        assert [int(x) for x in merged] == targets
        assert not path[0, frames:].any()
    assert padded > 20
