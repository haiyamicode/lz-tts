"""Speed, pitch, volume and effect adjustment for completed TTS audio."""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


_LOGGER = logging.getLogger(__name__)


AudioEffect = Literal["robot", "deep", "monster", "echo", "radio", "phone"]


@dataclass(frozen=True)
class _Effect:
    """An effect as the adjustment pass runs it: an ffmpeg filter while the input
    is decoded, then SoX effects after the speed and pitch changes."""

    decode_filter: str | None = None
    sox: tuple[str, ...] = ()


# Deterministic DSP on the finished speech. Each ends about as loud as the
# plain speech (within 2 LU on speech); a closing ``gain -l`` is its make-up gain.
_EFFECTS: dict[str, _Effect] = {
    # The spectrum's phases are zeroed, so every frame buzzes at the hop rate:
    # a monotone at 44100 / 256 ≈ 172 Hz.
    "robot": _Effect(
        decode_filter=(
            "aresample=44100,"
            "afftfilt=real='hypot(re,im)*sin(0)':imag='hypot(re,im)*cos(0)'"
            ":win_size=1024:overlap=0.75"
        ),
        sox=("gain", "-l", "13"),
    ),
    "deep": _Effect(sox=("pitch", "-400", "bass", "+4")),
    "monster": _Effect(
        sox=(
            "pitch", "-800",
            "chorus", "0.6", "0.9", "55", "0.4", "0.25", "2", "-t",
            "overdrive", "5", "10",
            "pad", "0", "0.6",
            "reverb", "40", "50", "50",
            "gain", "-l", "-0.5",
        )
    ),
    # A cave: long reverb with two echoes; the padding keeps the tail.
    "echo": _Effect(
        sox=(
            "pad", "0", "1.4",
            "reverb", "75", "40", "100", "100", "30", "0",
            "echos", "0.8", "0.9", "280", "0.25", "560", "0.12",
            "gain", "-l", "1.5",
        )
    ),
    # A walkie-talkie: narrow band, overdriven and squashed.
    "radio": _Effect(
        sox=(
            "sinc", "500-2600",
            "overdrive", "10", "10",
            "gain", "-6",
            "compand", "0.01,0.2", "-60,-60,-30,-12,0,-6", "-8",
            "gain", "-l", "4.5",
        )
    ),
    # A phone line: 300-3400 Hz at 8 kHz, compressed.
    "phone": _Effect(
        sox=(
            "sinc", "300-3400",
            "gain", "-3",
            "rate", "8000",
            "compand", "0.02,0.3", "-60,-60,-40,-20,0,-10", "-6",
            "gain", "-l", "9.5",
        )
    ),
}


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
    pause_before: float = 0.0,
    pause_after: float = 0.0,
    effect: AudioEffect | None = None,
) -> bytes:
    """Apply all requested adjustments in one SoX pass and preserve the encoding.

    ``effect`` goes on after the speed and pitch changes and before the volume,
    so the volume stays the one asked for. ``pause_before``/``pause_after`` add
    that many seconds of silence around the speech, after the speed change so
    the silence keeps its length.
    """
    if (
        speed == 1.0
        and pitch == 1.0
        and volume == 1.0
        and not pause_before
        and not pause_after
        and effect is None
    ):
        return audio
    if not audio:
        raise AudioAdjustmentError("cannot adjust empty audio")

    started = time.perf_counter()
    _LOGGER.info(
        "Audio adjustment started speed=%.3f pitch=%.3f volume=%.3f "
        "pause_before=%.3f pause_after=%.3f effect=%s input_bytes=%d",
        speed,
        pitch,
        volume,
        pause_before,
        pause_after,
        effect,
        len(audio),
    )
    effect_steps = _EFFECTS[effect] if effect else _Effect()

    with tempfile.TemporaryDirectory(prefix="lzai-adjust-") as temp_dir:
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
                *(["-af", effect_steps.decode_filter] if effect_steps.decode_filter else []),
                "-acodec",
                "pcm_s16le",
                str(decoded_input),
            ],
            name="ffmpeg audio decode",
        )

        sox_command = ["sox", str(decoded_input), str(adjusted_output)]
        if speed != 1.0:
            sox_command.extend(["tempo", "-s", str(speed)])
        if pitch != 1.0:
            cents = round(1200 * math.log2(pitch))
            if cents:
                sox_command.extend(["pitch", str(cents)])
        sox_command.extend(effect_steps.sox)
        if volume != 1.0:
            sox_command.extend(["vol", str(volume)])
        if pause_before or pause_after:
            sox_command.extend(["pad", str(pause_before), str(pause_after)])
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
        "Audio adjustment completed speed=%.3f pitch=%.3f volume=%.3f effect=%s "
        "output_bytes=%d wall_seconds=%.3f",
        speed,
        pitch,
        volume,
        effect,
        len(result),
        time.perf_counter() - started,
    )
    return result
