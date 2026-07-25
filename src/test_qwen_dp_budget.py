import numpy as np

from src.qwen_dp_budget import trim_boundary_silence


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
