from __future__ import annotations

import pytest

from .preprocess import (
    _normalize_text_for_mapping,
    _normalized_override_bounds,
    apply_ipa_overrides,
)


def test_override_bounds_survive_context_sensitive_tashkeel() -> None:
    """Arabic tashkeel is a neural, context-sensitive model: it may rewrite
    an existing diacritic (e.g. the final kasra of a mid-sentence word
    becomes a damma) without moving any base character. The span bounds must
    still resolve instead of raising ``Text normalization did not preserve
    IPA override boundaries``.
    """
    source = "الرَّحْمَانِ الرَّحِيمِ"
    normalized = _normalize_text_for_mapping(source, "ar")

    assert _normalized_override_bounds(source, 0, 12, normalized) == (0, 12)
    assert _normalized_override_bounds(source, 13, 23, normalized) == (13, 23)


def test_override_bounds_survive_tashkeel_mark_insertion() -> None:
    """On unvocalized Arabic text tashkeel *adds* diacritics, so the
    normalized text is longer than the source. The alignment must still map
    each word's span onto its normalized region.
    """
    source = "السلام عليكم ورحمة الله وبركاته"
    normalized = _normalize_text_for_mapping(source, "ar")
    assert len(normalized) > len(source)  # tashkeel added marks

    words = normalized.split(" ")
    # First word: tashkeel adds marks inside and before the rest of the text.
    assert _normalized_override_bounds(source, 0, 6, normalized) == (0, len(words[0]))
    # A mid-text word, offset by the inserted marks of the previous word.
    second_start = len(words[0]) + 1
    assert _normalized_override_bounds(source, 7, 12, normalized) == (
        second_start,
        second_start + len(words[1]),
    )
    # Whole text.
    assert _normalized_override_bounds(source, 0, len(source), normalized) == (0, len(normalized))


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


def test_apply_ipa_overrides_skips_cross_language_span_override() -> None:
    """An override crossing language spans cannot be applied per span; it is
    skipped and the text is pronounced as written."""
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
    result = apply_ipa_overrides(spans, [(4, 7, "x")])

    assert result[0]["phonemes"] == list("hello")
    assert result[1]["phonemes"] == list("shijie")


def test_apply_ipa_overrides_skips_span_removed_by_normalization() -> None:
    """A span whose characters are all dropped by normalization has nothing
    to override; it is skipped instead of failing the synthesis."""
    spans = [
        {
            "text": "ab",
            "source_text": "a.b",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 3,
            "phonemes": list("ks"),
            "word_spans": [[0, 1, 0, 1], [1, 2, 1, 2]],
        }
    ]
    result = apply_ipa_overrides(spans, [(1, 2, "x")])  # the "." vanishes

    assert result[0]["phonemes"] == list("ks")
    assert result[0]["word_spans"] == [[0, 1, 0, 1], [1, 2, 1, 2]]


def test_apply_ipa_overrides_skips_span_without_pronounced_unit() -> None:
    """A span that maps onto a region with no phonemes (e.g. a space) is
    skipped instead of failing the synthesis."""
    spans = [
        {
            "text": "a b",
            "source_text": "a b",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 3,
            "phonemes": list("ks"),
            "word_spans": [[0, 1, 0, 1], [2, 3, 1, 2]],
        }
    ]
    result = apply_ipa_overrides(spans, [(1, 2, "x")])  # the space

    assert result[0]["phonemes"] == list("ks")
    assert result[0]["word_spans"] == [[0, 1, 0, 1], [2, 3, 1, 2]]


def test_apply_ipa_overrides_applies_valid_and_skips_invalid_together() -> None:
    spans = [
        {
            "text": "a b c",
            "source_text": "a b c",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 5,
            "phonemes": list("kst"),
            "word_spans": [[0, 1, 0, 1], [2, 3, 1, 2], [4, 5, 2, 3]],
        }
    ]
    result = apply_ipa_overrides(
        spans,
        [(1, 2, "x"), (2, 3, "ɛ")],  # space skipped, "b" overridden
    )

    assert result[0]["phonemes"] == ["k", "ɛ", "t"]
    assert result[0]["word_spans"] == [[0, 1, 0, 1], [2, 3, 1, 2], [4, 5, 2, 3]]


def test_apply_ipa_overrides_skips_malformed_and_overlapping() -> None:
    """Empty IPA, empty spans, and overlaps are skipped; the rest applies."""
    spans = [
        {
            "text": "abcde",
            "source_text": "abcde",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 5,
            "phonemes": list("abcde"),
            "word_spans": [[0, 1, 0, 1], [1, 2, 1, 2], [2, 3, 2, 3], [3, 4, 3, 4], [4, 5, 4, 5]],
        }
    ]
    result = apply_ipa_overrides(
        spans, [(0, 3, "æ"), (1, 1, "x"), (2, 2, ""), (2, 5, "y")]
    )

    assert result[0]["phonemes"] == ["æ", "d", "e"]
    assert result[0]["word_spans"] == [[0, 3, 0, 1], [3, 4, 1, 2], [4, 5, 2, 3]]


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
        normalized,
    ) == (8, 10)
