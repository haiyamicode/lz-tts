"""Spoken language identification tests — real model, real speech, no mocks."""

import os
from pathlib import Path

import numpy as np
import pytest

from src.api.language_id import (
    MAX_SECONDS,
    SAMPLE_RATE,
    LanguageInputError,
    SpokenLanguageIdentifier,
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
    ("name", "code"),
    [("vntintuc.wav", "vi"), ("andrew.mp3", "en"), ("ava_zh.wav", "zh")],
)
def test_identifies_language(identifier: SpokenLanguageIdentifier, name: str, code: str) -> None:
    result = identifier.identify(_sample(name))
    assert result["code"] == code
    assert result["model"] == "lang-id-voxlingua107-ecapa"
    assert result["candidates"][0] == {"code": result["code"], "name": result["name"], "score": result["score"]}
    scores = [c["score"] for c in result["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 < sum(scores) <= 1.0 + 1e-4
    assert 0.0 < result["analyzedSeconds"] <= MAX_SECONDS
