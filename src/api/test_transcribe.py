from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.api.transcribe import (
    DEFAULT_MODEL_DIR,
    QwenTranscriber,
    TranscribeInputError,
    decode_transcribe_audio,
    language_code,
    qwen_language,
)

_CLIP = Path("local/samples/vi/mautiengviet.mp3")


def test_qwen_language_maps_locales_and_codes():
    assert qwen_language("vi-VN") == "Vietnamese"
    assert qwen_language("zh_CN") == "Chinese"
    assert qwen_language("yue") == "Cantonese"
    assert qwen_language("fil-PH") == "Filipino"
    assert qwen_language("tl") == "Filipino"


def test_qwen_language_auto_detects_when_unset_or_unsupported():
    assert qwen_language(None) is None
    assert qwen_language("") is None
    assert qwen_language("sw-KE") is None


def test_language_code_maps_qwen_names_back_to_iso():
    assert language_code("Vietnamese") == "vi"
    assert language_code("Filipino") == "fil"
    assert language_code("Cantonese") == "yue"
    assert language_code("Klingon") is None
    assert language_code(None) is None


def test_decode_rejects_empty_and_undecodable_input():
    with pytest.raises(TranscribeInputError):
        decode_transcribe_audio(b"")
    with pytest.raises(TranscribeInputError):
        decode_transcribe_audio(b"not audio at all")


@pytest.mark.skipif(
    not Path(DEFAULT_MODEL_DIR, "model.safetensors").exists() or not _CLIP.exists(),
    reason="Qwen3-ASR weights under data/qwen3-asr-0.6b or the Vietnamese sample clip are missing",
)
def test_qwen_transcribes_vietnamese_clip():
    audio = decode_transcribe_audio(_CLIP.read_bytes())
    assert audio.dtype == np.float32
    assert len(audio) / 16000 > 30

    result = QwenTranscriber(DEFAULT_MODEL_DIR, device="cuda:0", dtype="float16").transcribe(audio, "vi-VN")

    assert result["language"] == "Vietnamese"
    assert result["languageCode"] == "vi"
    words = result["text"].lower().replace(".", " ").replace(",", " ").split()
    assert " ".join(words[:4]) == "khám phá vũ trụ"
    assert len(words) > 100
