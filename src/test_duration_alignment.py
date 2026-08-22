import numpy as np
import pytest
import torch

from src.duration_alignment import (
    DpBudgetConfig,
    DurationAlignmentValidator,
    alignment_word_timestamps,
    trim_boundary_silence,
)
from src.piper.preprocess import phonemize_text_for_infer


def _tone(sample_rate: int, seconds: float, amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    return amplitude * np.sin(2 * np.pi * 220 * time)


def test_trim_boundary_silence_preserves_internal_pause() -> None:
    sample_rate = 16_000
    tone = _tone(sample_rate, 0.3)
    internal_pause = np.zeros(round(sample_rate * 0.2), dtype=np.float32)
    audio = np.concatenate(
        [
            np.zeros(round(sample_rate * 0.3), dtype=np.float32),
            tone,
            internal_pause,
            tone,
            np.zeros(round(sample_rate * 0.4), dtype=np.float32),
        ]
    )

    trimmed, start, end = trim_boundary_silence(audio, sample_rate)

    assert 0.23 * sample_rate <= start <= 0.26 * sample_rate
    assert 1.14 * sample_rate <= end <= 1.18 * sample_rate
    np.testing.assert_array_equal(trimmed, audio[start:end])
    assert np.count_nonzero(trimmed == 0) >= internal_pause.size


def test_alignment_word_timestamps_map_phoneme_spans_to_mas_tokens() -> None:
    words = alignment_word_timestamps(
        text="go there",
        phonemes=list("go there"),
        word_spans=[[0, 2, 0, 2], [3, 8, 3, 8]],
        token_durations_frames=torch.ones(19),
        hop_length=256,
        sample_rate=25600,
        trim_start_samples=256,
    )

    assert words[0]["token_start"] == 2
    assert words[0]["token_end"] == 6
    assert words[0]["start_seconds"] == pytest.approx(0.03)
    assert words[0]["end_seconds"] == pytest.approx(0.07)
    assert words[1]["token_start"] == 8
    assert words[1]["token_end"] == 18
    assert words[1]["start_seconds"] == pytest.approx(0.09)
    assert words[1]["end_seconds"] == pytest.approx(0.19)


def test_trim_boundary_silence_keeps_audio_at_boundaries() -> None:
    audio = _tone(16_000, 0.5)

    trimmed, start, end = trim_boundary_silence(audio, 16_000)

    assert start == 0
    assert end == audio.size
    np.testing.assert_array_equal(trimmed, audio)


def test_trim_boundary_silence_returns_empty_for_silence() -> None:
    audio = np.zeros(16_000, dtype=np.float32)

    trimmed, start, end = trim_boundary_silence(audio, 16_000)

    assert trimmed.size == 0
    assert start == 0
    assert end == 0


def test_duration_budget_can_skip_unused_word_span_alignment() -> None:
    text = "清晨的小雨已经停了，工作人员正在准备今天的图书馆活动。"
    config = {
        "language": {"code": "zh-CN"},
        "espeak": {"voice": "zh-CN", "primary": "en-us"},
    }

    mapped = phonemize_text_for_infer(text, config, include_word_spans=True)
    unmapped = phonemize_text_for_infer(text, config, include_word_spans=False)

    assert mapped["word_spans"]
    assert unmapped["word_spans"] is None
    assert unmapped["text"] == mapped["text"]
    assert unmapped["phonemes"] == mapped["phonemes"]
    assert unmapped["phoneme_ids"] == mapped["phoneme_ids"]

    validator = DurationAlignmentValidator(DpBudgetConfig(include_word_spans=False))
    chunks = validator._phonemize_budget_chunks(text, "zh-CN")
    assert chunks
    assert all(chunk["word_spans"] is None for chunk in chunks)
