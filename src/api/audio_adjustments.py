"""Speed, pitch, and volume adjustment for completed TTS audio."""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal


_LOGGER = logging.getLogger(__name__)


class AudioAdjustmentError(RuntimeError):
    """Raised when an external audio adjustment command fails."""


def _run(command: list[str], *, name: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout).strip()
    raise AudioAdjustmentError(
        f"{name} exited with code {result.returncode}: {detail or 'no diagnostic output'}"
    )


def adjust_audio(
    audio: bytes,
    *,
    input_format: Literal["mp3", "wav"],
    output_format: Literal["mp3", "wav"],
    speed: float = 1.0,
    pitch: float = 1.0,
    volume: float = 1.0,
) -> bytes:
    """Apply all requested adjustments in one SoX pass and preserve the encoding."""
    if speed == 1.0 and pitch == 1.0 and volume == 1.0:
        return audio
    if not audio:
        raise AudioAdjustmentError("cannot adjust empty audio")

    started = time.perf_counter()
    _LOGGER.info(
        "Audio adjustment started speed=%.3f pitch=%.3f volume=%.3f input_bytes=%d",
        speed,
        pitch,
        volume,
        len(audio),
    )

    with tempfile.TemporaryDirectory(prefix="lz-tts-adjust-") as temp_dir:
        root = Path(temp_dir)
        encoded_input = root / f"encoded-input.{input_format}"
        decoded_input = root / "decoded-input.wav"
        adjusted_output = root / "adjusted.wav"
        encoded_output = root / f"output.{output_format}"
        encoded_input.write_bytes(audio)

        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(encoded_input),
                "-vn",
                "-acodec",
                "pcm_s16le",
                str(decoded_input),
            ],
            name="ffmpeg audio decode",
        )

        sox_command = ["sox", str(decoded_input), str(adjusted_output)]
        if volume != 1.0:
            sox_command.extend(["vol", str(volume)])
        if speed != 1.0:
            sox_command.extend(["tempo", "-s", str(speed)])
        if pitch != 1.0:
            cents = round(1200 * math.log2(pitch))
            if cents:
                sox_command.extend(["pitch", str(cents)])
        sox_command.extend(["rate", "-v", "44100"])
        _run(sox_command, name="sox audio adjustment")

        if output_format == "mp3":
            encode_command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(adjusted_output),
                "-vn",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "320k",
                str(encoded_output),
            ]
        else:
            encode_command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(adjusted_output),
                "-vn",
                "-acodec",
                "pcm_s16le",
                str(encoded_output),
            ]
        _run(encode_command, name="ffmpeg audio encode")
        result = encoded_output.read_bytes()

    _LOGGER.info(
        "Audio adjustment completed speed=%.3f pitch=%.3f volume=%.3f output_bytes=%d "
        "wall_seconds=%.3f",
        speed,
        pitch,
        volume,
        len(result),
        time.perf_counter() - started,
    )
    return result
