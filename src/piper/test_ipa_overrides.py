from __future__ import annotations

import pytest

from .preprocess import (
    _normalize_text_for_mapping,
    _normalized_override_bounds,
    apply_ipa_overrides,
)


def test_apply_ipa_overrides_rewrites_exact_mapping_and_shifts_following_spans() -> None:
    spans = [
        {
            "text": "hello saoirse left",
            "source_text": "hello Saoirse left",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 18,
            "phonemes": list("abc def ghi"),
            "word_spans": [[0, 5, 0, 3], [6, 13, 4, 7], [14, 18, 8, 11]],
            "speaker_id": 0,
        }
    ]
    result = apply_ipa_overrides(spans, [(6, 13, "sˈɜːɹʃə")])

    assert result[0]["phonemes"] == list("abc sˈɜːɹʃə ghi")
    assert result[0]["word_spans"] == [
        [0, 5, 0, 3],
        [6, 13, 4, 11],
        [14, 18, 12, 15],
    ]
    assert result[0]["phoneme_ids"]


def test_apply_ipa_overrides_preserves_partial_frontend_word_fragments() -> None:
    spans = [
        {
            "text": "saoirse",
            "source_text": "Saoirse",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 7,
            "phonemes": list("sɜʃə"),
            "word_spans": [[0, 7, 0, 4]],
        }
    ]
    result = apply_ipa_overrides(
        spans,
        [(0, 3, "sɜː")],
        partial_word_phonemizer=lambda _text, _voice: list("saʊ"),
    )

    assert result[0]["phonemes"] == list("sɜːʃə")
    assert result[0]["word_spans"] == [[0, 7, 0, 5]]


def test_apply_ipa_overrides_preserves_unwrapped_suffix_from_original_mapping() -> None:
    spans = [
        {
            "text": "cat's",
            "source_text": "cat's",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 5,
            "phonemes": list("kæts"),
            "word_spans": [[0, 5, 0, 4]],
        }
    ]
    result = apply_ipa_overrides(
        spans,
        [(0, 3, "kæt")],
        partial_word_phonemizer=lambda _text, _voice: list("kæt"),
    )

    assert result[0]["phonemes"] == list("kæts")
    assert result[0]["word_spans"] == [[0, 5, 0, 4]]


def test_apply_ipa_overrides_rejects_cross_language_span() -> None:
    spans = [
        {
            "text": "hello ",
            "source_text": "hello ",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 6,
            "phonemes": list("hello"),
            "word_spans": [[0, 5, 0, 5]],
        },
        {
            "text": "世界",
            "source_text": "世界",
            "voice": "cmn",
            "source_start": 6,
            "source_end": 8,
            "phonemes": list("shijie"),
            "word_spans": [[0, 2, 0, 6]],
        },
    ]
    with pytest.raises(ValueError, match="crosses a Sparrow language span"):
        apply_ipa_overrides(spans, [(4, 7, "x")])


def test_apply_ipa_overrides_maps_boundaries_through_text_normalization() -> None:
    spans = [
        {
            "text": "captain smith left",
            "source_text": "Capt. Smith left",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 16,
            "phonemes": list("abc def ghi"),
            "word_spans": [[0, 7, 0, 3], [8, 13, 4, 7], [14, 18, 8, 11]],
        }
    ]
    result = apply_ipa_overrides(spans, [(6, 11, "smɪθ")])

    assert result[0]["phonemes"] == list("abc smɪθ ghi")
    assert result[0]["word_spans"] == [
        [0, 7, 0, 3],
        [8, 13, 4, 8],
        [14, 18, 9, 12],
    ]


def test_ipa_override_boundaries_survive_chinese_normalization() -> None:
    text = "今天我们用新鲜的番茄做一道简单的菜。"
    normalized = _normalize_text_for_mapping(text, "cmn-latn-pinyin")

    assert _normalized_override_bounds(
        text,
        text.index("番茄"),
        text.index("番茄") + len("番茄"),
        "cmn-latn-pinyin",
        normalized,
    ) == (8, 10)
