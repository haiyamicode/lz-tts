from __future__ import annotations

import subprocess
import unittest

import numpy as np

from .audio_adjustments import adjust_audio

SAMPLE_RATE = 16000


def _tone_mp3(seconds: float) -> bytes:
    return subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-codec:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "-",
        ],
        capture_output=True,
        check=True,
    ).stdout


def _decode(audio: bytes) -> np.ndarray:
    pcm = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "-",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
        ],
        input=audio,
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _seconds(samples: np.ndarray) -> float:
    return len(samples) / SAMPLE_RATE


def _peak(samples: np.ndarray, start: float, end: float) -> float:
    return float(np.abs(samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]).max())


class PauseTests(unittest.TestCase):
    def test_pauses_add_silence_around_the_speech(self) -> None:
        tone = _tone_mp3(1.0)
        audio = adjust_audio(
            tone,
            input_format="mp3",
            output_format="mp3",
            pause_before=0.5,
            pause_after=0.3,
        )
        samples = _decode(audio)
        self.assertAlmostEqual(_seconds(samples) - _seconds(_decode(tone)), 0.8, delta=0.01)
        self.assertLess(_peak(samples, 0.0, 0.45), 0.01)
        self.assertGreater(_peak(samples, 0.6, 1.4), 0.05)
        self.assertLess(_peak(samples, 1.6, 1.75), 0.01)

    def test_speed_does_not_stretch_the_pauses(self) -> None:
        tone = _tone_mp3(1.0)
        audio = adjust_audio(
            tone,
            input_format="mp3",
            output_format="mp3",
            speed=2.0,
            pause_before=0.5,
            pause_after=0.3,
        )
        self.assertAlmostEqual(
            _seconds(_decode(audio)), _seconds(_decode(tone)) / 2 + 0.8, delta=0.02
        )

    def test_no_adjustments_returns_the_input(self) -> None:
        audio = _tone_mp3(0.5)
        self.assertIs(
            adjust_audio(audio, input_format="mp3", output_format="mp3"), audio
        )


if __name__ == "__main__":
    unittest.main()
