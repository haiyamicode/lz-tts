"""decode_audio_16k_mono on real ffmpeg output; needs no model snapshot."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.api.language_id import SAMPLE_RATE, LanguageInputError, decode_audio_16k_mono


def _encode(tmp_path: Path, name: str, *args: str) -> bytes:
    out = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=30", *args, str(out)],
        check=True,
    )
    return out.read_bytes()


def test_decodes_m4a_with_moov_at_the_end(tmp_path: Path) -> None:
    # ffmpeg writes the moov atom after mdat unless told +faststart. 30 s is past what ffmpeg
    # buffers from a pipe, so a piped decode of this file returns nothing.
    audio = _encode(tmp_path, "tail.m4a", "-c:a", "aac")
    assert audio.index(b"moov") > audio.index(b"mdat")
    samples = decode_audio_16k_mono(audio)
    assert samples.dtype == np.float32
    assert abs(samples.size - 30 * SAMPLE_RATE) < SAMPLE_RATE // 10


def test_decodes_faststart_m4a(tmp_path: Path) -> None:
    audio = _encode(tmp_path, "fast.m4a", "-c:a", "aac", "-movflags", "+faststart")
    assert audio.index(b"moov") < audio.index(b"mdat")
    assert abs(decode_audio_16k_mono(audio).size - 30 * SAMPLE_RATE) < SAMPLE_RATE // 10


def test_truncated_mp4_is_undecodable(tmp_path: Path) -> None:
    audio = _encode(tmp_path, "tail.m4a", "-c:a", "aac")
    with pytest.raises(LanguageInputError, match="Undecodable audio"):
        decode_audio_16k_mono(audio[: audio.index(b"moov") - 8])
