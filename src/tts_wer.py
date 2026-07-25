from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any

from num2words import num2words
from piper_phonemize import phonemize_espeak_with_mapping
from rapidfuzz.distance import Levenshtein

from src.text_norm import normalize_text


SPOKEN_ENGLISH_ABBREVIATIONS = {
    "mr": "mister",
    "mrs": "misses",
    "ms": "miss",
    "dr": "doctor",
}

LANGUAGE_WER_CONFIGS = {
    "de": {"locale": "de-DE", "espeak": "de", "asr": "German"},
    "en": {"locale": "en-US", "espeak": "en-us", "asr": "English"},
    "es": {"locale": "es-ES", "espeak": "es", "asr": "Spanish"},
    "fr": {"locale": "fr-FR", "espeak": "fr", "asr": "French"},
    "it": {"locale": "it-IT", "espeak": "it", "asr": "Italian"},
    "ja": {"locale": "ja-JP", "espeak": "ja", "asr": "Japanese", "character_error_rate": True},
    "ko": {"locale": "ko-KR", "espeak": "ko", "asr": "Korean"},
    "pt": {"locale": "pt-BR", "espeak": "pt-br", "asr": "Portuguese"},
    "ru": {"locale": "ru-RU", "espeak": "ru", "asr": "Russian"},
    "vi": {"locale": "vi-VN", "espeak": "vi", "asr": "Vietnamese"},
    "zh": {"locale": "zh-CN", "espeak": "cmn", "asr": "Chinese", "character_error_rate": True},
}


def normalize_english_for_wer(text: str) -> str:
    text = normalize_text(unicodedata.normalize("NFKC", text), "en-US").casefold()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    words = []
    for word in text.split():
        word = SPOKEN_ENGLISH_ABBREVIATIONS.get(word, word)
        if word.isdecimal():
            if len(word) > 1 and word.startswith("0"):
                words.extend(num2words(int(digit), lang="en") for digit in word)
            else:
                words.extend(
                    num2words(int(word), lang="en").replace("-", " ").split()
                )
        else:
            words.append(word)
    return " ".join(words)


def _canonical_phonemes(phonemes: list[str]) -> str:
    return "".join(phoneme for phoneme in phonemes if not phoneme.isspace())


@lru_cache(maxsize=4096)
def phonemize_english_aligned(
    normalized_text: str,
) -> tuple[tuple[str, ...], str]:
    """Return context-aware phonemes per word and for the full utterance."""
    if not normalized_text:
        return (), ""
    sentences, sentence_mappings = phonemize_espeak_with_mapping(
        normalized_text,
        "en-us",
    )
    word_phonemes = []
    full_phonemes = []
    for sentence, mappings in zip(sentences, sentence_mappings):
        full_phonemes.extend(sentence)
        for _, _, phoneme_start, phoneme_end, punctuation_length in mappings:
            word_phonemes.append(
                _canonical_phonemes(
                    sentence[
                        phoneme_start : phoneme_end - punctuation_length
                        if punctuation_length
                        else phoneme_end
                    ]
                )
            )
    words = normalized_text.split()
    if len(word_phonemes) != len(words):
        word_phonemes = [
            _canonical_phonemes(
                [
                    phoneme
                    for sentence in phonemize_espeak_with_mapping(word, "en-us")[0]
                    for phoneme in sentence
                ]
            )
            for word in words
        ]
    return tuple(word_phonemes), _canonical_phonemes(full_phonemes)


def calculate_english_phoneme_adjusted_wer(
    reference: str,
    hypothesis: str,
) -> dict[str, Any]:
    reference_words = normalize_english_for_wer(reference).split()
    hypothesis_words = normalize_english_for_wer(hypothesis).split()
    operations = Levenshtein.editops(reference_words, hypothesis_words)
    counts = Counter(operation.tag for operation in operations)
    errors = len(operations)
    denominator = len(reference_words)
    reference_word_phonemes, reference_phonemes = phonemize_english_aligned(
        " ".join(reference_words)
    )
    hypothesis_word_phonemes, hypothesis_phonemes = phonemize_english_aligned(
        " ".join(hypothesis_words)
    )
    utterance_phoneme_match = (
        bool(reference_phonemes) and reference_phonemes == hypothesis_phonemes
    )
    equivalent_substitutions = sum(
        operation.tag == "replace"
        and reference_word_phonemes[operation.src_pos]
        == hypothesis_word_phonemes[operation.dest_pos]
        for operation in operations
    )
    adjusted_errors = (
        0 if utterance_phoneme_match else errors - equivalent_substitutions
    )
    return {
        "target_normalized": " ".join(reference_words),
        "asr_normalized": " ".join(hypothesis_words),
        "target_word_count": denominator,
        "asr_word_count": len(hypothesis_words),
        "substitutions": counts["replace"],
        "deletions": counts["delete"],
        "insertions": counts["insert"],
        "word_errors": errors,
        "wer": errors / denominator if denominator else float(bool(hypothesis_words)),
        "target_phonemes": reference_phonemes,
        "asr_phonemes": hypothesis_phonemes,
        "utterance_phoneme_match": utterance_phoneme_match,
        "phoneme_equivalent_substitutions": equivalent_substitutions,
        "phoneme_adjusted_word_errors": adjusted_errors,
        "phoneme_adjusted_wer": (
            adjusted_errors / denominator
            if denominator
            else float(bool(hypothesis_words))
        ),
    }


def _normalize_multilingual_for_wer(text: str, locale: str) -> str:
    normalized = normalize_text(unicodedata.normalize("NFKC", text), locale).casefold()
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = re.sub(r"(?<=\w)'(?=\w)", "", normalized)
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


@lru_cache(maxsize=16384)
def _phonemize_tokens(
    normalized_text: str,
    espeak_voice: str,
) -> tuple[tuple[str, ...], str]:
    if not normalized_text:
        return (), ""
    sentences, sentence_mappings = phonemize_espeak_with_mapping(
        normalized_text,
        espeak_voice,
    )
    token_phonemes = []
    full_phonemes = []
    for sentence, mappings in zip(sentences, sentence_mappings):
        full_phonemes.extend(sentence)
        for _, _, phoneme_start, phoneme_end, punctuation_length in mappings:
            token_phonemes.append(
                _canonical_phonemes(
                    sentence[
                        phoneme_start : phoneme_end - punctuation_length
                        if punctuation_length
                        else phoneme_end
                    ]
                )
            )
    tokens = normalized_text.split()
    if len(token_phonemes) != len(tokens):
        token_phonemes = [
            _canonical_phonemes(
                [
                    phoneme
                    for sentence in phonemize_espeak_with_mapping(token, espeak_voice)[0]
                    for phoneme in sentence
                ]
            )
            for token in tokens
        ]
    return tuple(token_phonemes), _canonical_phonemes(full_phonemes)


def calculate_multilingual_phoneme_adjusted_wer(
    reference: str,
    hypothesis: str,
    language: str,
) -> dict[str, Any]:
    """Calculate language-aware WER, forgiving aligned homophone substitutions.

    Japanese and Chinese use character tokens because whitespace-delimited WER is
    not meaningful for those scripts. The compatibility fields retain the ``wer``
    name so callers can use one threshold across languages.
    """
    language_code = language.strip().lower().replace("_", "-").split("-", 1)[0]
    if language_code == "en":
        return calculate_english_phoneme_adjusted_wer(reference, hypothesis) | {
            "error_rate_unit": "word",
        }
    if language_code not in LANGUAGE_WER_CONFIGS:
        raise ValueError(f"Unsupported WER language: {language!r}")

    config = LANGUAGE_WER_CONFIGS[language_code]
    target_normalized = _normalize_multilingual_for_wer(reference, config["locale"])
    asr_normalized = _normalize_multilingual_for_wer(hypothesis, config["locale"])
    character_error_rate = bool(config.get("character_error_rate", False))
    if character_error_rate:
        reference_tokens = [char for char in target_normalized if not char.isspace()]
        hypothesis_tokens = [char for char in asr_normalized if not char.isspace()]
    else:
        reference_tokens = target_normalized.split()
        hypothesis_tokens = asr_normalized.split()

    operations = Levenshtein.editops(reference_tokens, hypothesis_tokens)
    counts = Counter(operation.tag for operation in operations)
    errors = len(operations)
    denominator = len(reference_tokens)

    reference_token_phonemes: tuple[str, ...] = ()
    hypothesis_token_phonemes: tuple[str, ...] = ()
    reference_phonemes = ""
    hypothesis_phonemes = ""
    if not character_error_rate:
        reference_token_phonemes, reference_phonemes = _phonemize_tokens(
            target_normalized,
            config["espeak"],
        )
        hypothesis_token_phonemes, hypothesis_phonemes = _phonemize_tokens(
            asr_normalized,
            config["espeak"],
        )
    else:
        _, reference_phonemes = _phonemize_tokens(
            target_normalized,
            config["espeak"],
        )
        _, hypothesis_phonemes = _phonemize_tokens(
            asr_normalized,
            config["espeak"],
        )

    utterance_phoneme_match = (
        bool(reference_phonemes) and reference_phonemes == hypothesis_phonemes
    )
    equivalent_substitutions = 0
    if not character_error_rate:
        equivalent_substitutions = sum(
            operation.tag == "replace"
            and reference_token_phonemes[operation.src_pos]
            == hypothesis_token_phonemes[operation.dest_pos]
            for operation in operations
        )
    adjusted_errors = (
        0 if utterance_phoneme_match else errors - equivalent_substitutions
    )
    error_rate = errors / denominator if denominator else float(bool(hypothesis_tokens))
    adjusted_error_rate = (
        adjusted_errors / denominator
        if denominator
        else float(bool(hypothesis_tokens))
    )
    return {
        "target_normalized": target_normalized,
        "asr_normalized": asr_normalized,
        "target_word_count": denominator,
        "asr_word_count": len(hypothesis_tokens),
        "substitutions": counts["replace"],
        "deletions": counts["delete"],
        "insertions": counts["insert"],
        "word_errors": errors,
        "wer": error_rate,
        "target_phonemes": reference_phonemes,
        "asr_phonemes": hypothesis_phonemes,
        "utterance_phoneme_match": utterance_phoneme_match,
        "phoneme_equivalent_substitutions": equivalent_substitutions,
        "phoneme_adjusted_word_errors": adjusted_errors,
        "phoneme_adjusted_wer": adjusted_error_rate,
        "error_rate_unit": "character" if character_error_rate else "word",
    }
