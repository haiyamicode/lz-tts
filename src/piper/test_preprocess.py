"""Regression tests for espeak phonemization edge cases."""

import json

from pathlib import Path

from src.multilingual_splitter import MultilingualSplitter, SplitResult
from src.piper.preprocess import (
    _map_cld2_to_espeak,
    phonemize_text_for_infer,
)


def _repeated_number_text() -> str:
    # Shape of the production failure payload: a long run of repeated
    # numbers. espeak reads the full sentence differently from the
    # per-token anchors, so the word-span alignment is ambiguous.
    return " ".join(["67", "67", "67"] + ["6767"] * 64 + ["6"])


def test_effective_main_language_falls_back_for_und() -> None:
    result = MultilingualSplitter().split("67 67 67 6767 6767")
    assert result.main_language == "und"
    assert result.effective_main_language() == "en"
    # A request/voice language is used as the fallback base when given.
    assert result.effective_main_language("fr-FR") == "fr-FR"

    normal = SplitResult(original_text="hello", main_language="fr", segments=[])
    assert normal.effective_main_language() == "fr"
    assert normal.effective_main_language("en") == "fr"


def test_und_language_maps_to_primary_espeak_voice() -> None:
    assert _map_cld2_to_espeak("und", "en-us") == "en-us"
    assert _map_cld2_to_espeak("und", "cmn-latn-pinyin") == "cmn-latn-pinyin"
    assert _map_cld2_to_espeak("undetermined", "en-us") == "en-us"
    assert _map_cld2_to_espeak("", "en-us") == "en-us"
    # Normal languages keep mapping through the primary voice.
    assert _map_cld2_to_espeak("en", "en-us") == "en-us"
    assert _map_cld2_to_espeak("fr", "en-us") == "fr"


def test_repeated_number_text_produces_word_spans() -> None:
    # Previously raised "unmatched token '67'" when the DP word-span
    # alignment dropped leading tokens of repetitive text.
    config = {
        "language": {"code": "en"},
        "espeak": {"voice": "en", "primary": "en-us"},
    }
    result = phonemize_text_for_infer(
        _repeated_number_text(), config, neural=False, include_word_spans=True
    )
    assert result["phoneme_ids"]
    assert result["word_spans"]
    # The first token must be covered by the first mapping (0-based start).
    assert result["word_spans"][0][0] == 0


def test_multilingual_phonemization_of_undetectable_text() -> None:
    # The sparrow config default language is "multilingual"; CLD2 reports
    # pure numeric text as "und", which previously reached espeak as voice
    # "und" and raised "Failed to set eSpeak-ng voice".
    config_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "lzspeech-sparrow"
        / "config.json"
    )
    with open(config_path) as f:
        config = json.load(f)
    result = phonemize_text_for_infer(
        _repeated_number_text(), config, neural=False, include_word_spans=True
    )
    assert result["phoneme_ids"]
