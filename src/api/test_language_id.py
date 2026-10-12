"""Spoken language identification tests — real model, real speech, no mocks."""

import os
from pathlib import Path

import numpy as np
import pytest

from src.api.language_id import (
    ACCENT_LOCALES,
    ACCENT_MIN_SCORE,
    MAX_SECONDS,
    SAMPLE_RATE,
    LANGUAGE_DROPS,
    LANGUAGE_FOLDS,
    LANGUAGE_MIN_SCORE,
    LanguageDetectRequest,
    LanguageInputError,
    SpokenLanguageIdentifier,
    accent_locale,
    decode_audio_16k_mono,
)

_MODEL_DIR = "data/language-id/lang-id-voxlingua107-ecapa"
_SAMPLES = Path("local/samples")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(_MODEL_DIR),
    reason="language id snapshot not present under data/language-id",
)


@pytest.fixture(scope="module")
def identifier() -> SpokenLanguageIdentifier:
    return SpokenLanguageIdentifier()


def _sample(name: str) -> bytes:
    path = _SAMPLES / name
    if not path.exists():
        pytest.skip(f"sample audio {path} not present")
    return path.read_bytes()


def test_rejects_empty_input(identifier: SpokenLanguageIdentifier) -> None:
    with pytest.raises(LanguageInputError, match="empty"):
        identifier.identify(b"")


def test_rejects_non_audio_bytes(identifier: SpokenLanguageIdentifier) -> None:
    with pytest.raises(LanguageInputError):
        identifier.identify(b"definitely not audio" * 64)


def test_decode_caps_length() -> None:
    samples = decode_audio_16k_mono(_sample("vntintuc.wav"), max_seconds=2)
    assert samples.dtype == np.float32
    assert samples.size == 2 * SAMPLE_RATE


@pytest.mark.parametrize(
    ("name", "locale"),
    [
        ("vntintuc.wav", "vi"),
        ("ava_zh.wav", "zh"),
        ("andrew.mp3", "en-US"),
        ("clone-sample.mp3", "en-US"),
        ("en-gb/p226_003.wav", "en-GB"),
    ],
)
def test_identifies_locale(identifier: SpokenLanguageIdentifier, name: str, locale: str) -> None:
    result = identifier.identify(_sample(name))
    assert result["locale"] == locale
    language = result["language"]
    assert language["fallback"] is False
    assert result["locale"].split("-")[0] == language["code"]
    assert language["candidates"][0] == {"code": language["code"], "name": language["name"], "score": language["score"]}
    scores = [c["score"] for c in language["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 < sum(scores) <= 1.0 + 1e-4
    assert 0.0 < result["analyzedSeconds"] <= MAX_SECONDS
    if language["code"] == "en":
        accent = result["accent"]
        assert accent["locale"] == ACCENT_LOCALES[accent["code"]]
        assert accent["candidates"][0]["code"] == accent["code"]
        assert result["models"]["accent"] == "accent-id-commonaccent_ecapa"
    else:
        assert result["accent"] is None and "accent" not in result["models"]


def test_accent_locale_fallbacks() -> None:
    assert accent_locale("england", ACCENT_MIN_SCORE) == "en-GB"
    assert accent_locale("scotland", 0.99) == "en-GB"
    assert accent_locale("us", ACCENT_MIN_SCORE - 0.01) == "en"
    assert accent_locale("african", 0.999) == "en"
    assert accent_locale("southatlandtic", 0.999) == "en"


def test_top_k_controls_candidates(identifier: SpokenLanguageIdentifier) -> None:
    audio = _sample("andrew.mp3")
    one = identifier.identify(audio, top_k=1)
    five = identifier.identify(audio, top_k=5)
    for key in ("language", "accent"):
        assert len(one[key]["candidates"]) == 1 and len(five[key]["candidates"]) == 5
        assert one[key]["candidates"][0] == five[key]["candidates"][0]


def test_request_validation() -> None:
    assert LanguageDetectRequest(audio_url="https://x/a.mp3").top_k == 3
    with pytest.raises(Exception):
        LanguageDetectRequest(top_k=0)
    with pytest.raises(Exception):
        LanguageDetectRequest(top_k=11)


def test_batch_matches_single_clip(identifier: SpokenLanguageIdentifier) -> None:
    # Mixed lengths and languages in one padded batch. Pooling masks the
    # padding, but the convolutions still see zeros past a short clip's end:
    # scores drift slightly (~0.01 on low-confidence clips), labels hold.
    names = ["vntintuc.wav", "ava_zh.wav", "andrew.mp3", "en-gb/p226_003.wav"]
    clips = [decode_audio_16k_mono(_sample(name)) for name in names]
    batched = identifier.identify_samples(clips)
    for clip, result in zip(clips, batched, strict=True):
        single = identifier.identify_samples([clip])[0]
        assert result["locale"] == single["locale"]
        assert result["language"]["code"] == single["language"]["code"]
        assert result["language"]["score"] == pytest.approx(single["language"]["score"], abs=0.03)
        assert (result["accent"] is None) == (single["accent"] is None)
        if single["accent"] is not None:
            assert result["accent"]["code"] == single["accent"]["code"]
            assert result["accent"]["score"] == pytest.approx(single["accent"]["score"], abs=0.03)
    assert identifier.identify_samples([]) == []


def test_folds_confusable_minority_languages(identifier: SpokenLanguageIdentifier) -> None:
    # VCTK p225 (southern English) is VoxLingua107's "cy: Welsh" before folding.
    result = identifier.identify(_sample("en-gb/p225_003.wav"))
    assert result["language"]["rawCandidates"][0]["code"] == "cy"
    assert result["language"]["code"] == "en"
    assert result["accent"] is not None
    codes = [c["code"] for c in result["language"]["candidates"]]
    assert not set(codes) & (set(LANGUAGE_FOLDS) | LANGUAGE_DROPS)


def test_fold_and_drop_codes_are_model_labels() -> None:
    text = Path(_MODEL_DIR, "label_encoder.txt").read_text()
    codes = {line.split("'")[1].split(":")[0] for line in text.splitlines() if line.startswith("'") and ": " in line}
    assert len(codes) == 107
    assert set(LANGUAGE_FOLDS) <= codes and set(LANGUAGE_FOLDS.values()) <= codes and LANGUAGE_DROPS <= codes
    # A fold target is never itself folded or dropped.
    assert not set(LANGUAGE_FOLDS.values()) & (set(LANGUAGE_FOLDS) | LANGUAGE_DROPS)


def test_silence_falls_back_to_english(identifier: SpokenLanguageIdentifier) -> None:
    # Digital silence: the model's "no speech" output (Nynorsk 0.27 / Welsh 0.14).
    result = identifier.identify_samples([np.zeros(5 * SAMPLE_RATE, dtype=np.float32)])[0]
    language = result["language"]
    assert result["locale"] == "en"
    assert language["code"] == "en" and language["fallback"] is True
    assert language["candidates"][0]["score"] < LANGUAGE_MIN_SCORE
    assert result["accent"] is None and "accent" not in result["models"]
