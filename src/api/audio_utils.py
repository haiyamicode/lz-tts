"""Shared audio conversion helpers for API routes and model workers."""

from __future__ import annotations

import contextlib
import io
import math
import os
import re
import wave
from pathlib import Path

import numpy as np
from pydub import AudioSegment
import soundfile as sf

MP3_BITRATE = "320k"
MP3_EXPORT_PARAMETERS = ["-q:a", "0"]
MP3_INTERMEDIATE_WAV_SUBTYPE = "PCM_24"


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes."""
    pcm_audio = _audio_to_pcm16(audio)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_audio.tobytes())
    return buffer.getvalue()


def _audio_to_mp3_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to MP3 bytes with highest quality settings."""
    wav_buffer = io.BytesIO()
    sf.write(
        wav_buffer,
        _audio_to_float32(audio),
        sample_rate,
        format="WAV",
        subtype=MP3_INTERMEDIATE_WAV_SUBTYPE,
    )
    wav_buffer.seek(0)

    # Convert WAV to MP3 using the production quality settings.
    audio_segment = AudioSegment.from_wav(wav_buffer)
    mp3_buffer = io.BytesIO()
    audio_segment.export(
        mp3_buffer,
        format="mp3",
        bitrate=MP3_BITRATE,
        parameters=MP3_EXPORT_PARAMETERS,
    )
    return mp3_buffer.getvalue()


def _audio_to_pcm16(audio: np.ndarray) -> np.ndarray:
    audio_array = np.asarray(audio).squeeze()
    if np.issubdtype(audio_array.dtype, np.floating):
        return (np.clip(audio_array, -1.0, 1.0).astype(np.float32) * 32767.0).astype(np.int16)
    return audio_array.astype(np.int16, copy=False)


def _audio_to_float32(audio: np.ndarray) -> np.ndarray:
    audio_array = np.asarray(audio).squeeze()
    if np.issubdtype(audio_array.dtype, np.floating):
        return np.clip(audio_array, -1.0, 1.0).astype(np.float32, copy=False)
    if np.issubdtype(audio_array.dtype, np.integer):
        info = np.iinfo(audio_array.dtype)
        peak = max(abs(info.min), info.max)
        return np.clip(audio_array.astype(np.float32) / float(peak), -1.0, 1.0)
    return np.clip(audio_array.astype(np.float32), -1.0, 1.0)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    import math
    from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

    audio_array = np.asarray(audio).squeeze()
    gcd = math.gcd(source_rate, target_rate)
    resampled = resample_poly(audio_array.astype(np.float32), target_rate // gcd, source_rate // gcd)
    if np.issubdtype(audio_array.dtype, np.floating):
        return resampled.astype(audio_array.dtype, copy=False)
    return np.clip(resampled, np.iinfo(audio_array.dtype).min, np.iinfo(audio_array.dtype).max).astype(audio_array.dtype)


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return stem[:120] or "voice"


def _audio_file_to_mp3_bytes(audio_path: Path) -> bytes:
    audio_segment = AudioSegment.from_file(audio_path)
    mp3_buffer = io.BytesIO()
    audio_segment.export(
        mp3_buffer,
        format="mp3",
        bitrate=MP3_BITRATE,
        parameters=MP3_EXPORT_PARAMETERS,
    )
    return mp3_buffer.getvalue()
