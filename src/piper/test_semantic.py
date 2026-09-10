"""Regression tests for phoneme-aligned BERT input building."""

import pytest

from src.piper.semantic import _EMPTY_PHONEME_ID_COUNT, _build_word2ph_counts


def _token_columns() -> tuple[list[list[int]], list[int], list[int]]:
    """Tokenizer columns for the punctuation-only span '." "'."""
    offsets = [[0, 1], [1, 2], [2, 3]]
    attention_mask = [1, 1, 1]
    special_tokens_mask = [0, 0, 0]
    return offsets, attention_mask, special_tokens_mask


def test_punctuation_only_span_has_no_phoneme_alignment() -> None:
    # Shape of production failure a5pXJx4YCBEiUA4H3Mg2FF: a multilingual split
    # segment that is only punctuation ('.” “') phonemizes to BOS/blank/EOS
    # with word_spans=None. The span carries no phonemes, so there is nothing
    # to align and building the BERT input must not raise.
    offsets, attention_mask, special_tokens_mask = _token_columns()
    counts = _build_word2ph_counts(
        phoneme_length=_EMPTY_PHONEME_ID_COUNT,
        offsets=offsets,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
        word_spans=None,
    )
    assert counts == [0, 0, 0]


def test_span_with_phonemes_still_requires_word_spans() -> None:
    # A span that does contain phonemes keeps failing loudly when the
    # phonemizer produced no word spans.
    offsets, attention_mask, special_tokens_mask = _token_columns()
    with pytest.raises(ValueError, match="word_spans are required"):
        _build_word2ph_counts(
            phoneme_length=5,  # BOS, blank, one phoneme, blank, EOS
            offsets=offsets,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            word_spans=None,
        )
