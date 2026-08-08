from __future__ import annotations

import pytest

from .preprocess import apply_ipa_overrides


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


def test_apply_ipa_overrides_rejects_partial_frontend_word() -> None:
    spans = [
        {
            "text": "saoirse",
            "source_text": "Saoirse",
            "voice": "en-us",
            "source_start": 0,
            "source_end": 7,
            "phonemes": list("sirsha"),
            "word_spans": [[0, 7, 0, 6]],
        }
    ]
    with pytest.raises(ValueError, match="complete word"):
        apply_ipa_overrides(spans, [(0, 3, "sɜː")])


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
