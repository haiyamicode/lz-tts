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


def test_symbol_only_span_puts_its_phonemes_on_the_leading_token() -> None:
    # Shape of the tts-fallback failure 2026-11-27 (ar-SA forced speaker): eSpeak reads emoji
    # and symbols by name with no word mapping, so a chunk of only those has phonemes and no
    # word spans. Its phonemes are all unmapped and go to the first special token, as unmapped
    # phonemes before a sentence's first word do.
    offsets = [[0, 0], [0, 1], [1, 2], [0, 0]]
    attention_mask = [1, 1, 1, 1]
    special_tokens_mask = [1, 0, 0, 1]
    counts = _build_word2ph_counts(
        phoneme_length=355,
        offsets=offsets,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
        word_spans=None,
    )
    assert counts == [355, 0, 0, 0]


def test_symbol_only_text_builds_bert_input_end_to_end() -> None:
    from src.piper.preprocess import phonemize_texts_for_speaker
    from src.piper.semantic import build_bert_input

    config = "data/lzspeech-sparrow/config.json"
    texts = ["🎉🎉🎉 🔥🔥", "© ® ™", "Great news 🎉 today"]
    spans = phonemize_texts_for_speaker(texts, config, "ar-SA", neural=True)
    assert spans[0]["phonemes"] and not spans[0]["word_spans"]
    phoneme_lengths = [len(span["phoneme_ids"]) for span in spans]
    bert = build_bert_input(
        [span["text"] for span in spans],
        phoneme_lengths=phoneme_lengths,
        word_spans=[span["word_spans"] for span in spans],
    )
    assert bert["word2ph"].sum(dim=1).tolist() == phoneme_lengths
