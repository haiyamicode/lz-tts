"""A render goes up as the file it is: nothing in the worker reads it, and it is deleted after."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .api.server import InferenceResult, ResultFile
from .taskflow_worker import _artifact_duration, _file_chunks


def _wav(seconds: float) -> Path:
    scratch = Path(tempfile.mkdtemp(prefix="upload-test-"))
    t = np.arange(int(seconds * 24000)) / 24000
    sf.write(str(scratch / "out.wav"), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000)
    return scratch


def test_a_file_result_knows_its_size_without_reading_it():
    scratch = _wav(1.0)
    result = InferenceResult(
        kind="audio",
        content_type="audio/wav",
        file=ResultFile("audio/wav", path=scratch / "out.wav"),
        scratch=scratch,
    )
    assert result.size == (scratch / "out.wav").stat().st_size
    assert result.source == scratch / "out.wav"
    result.release()
    assert not scratch.exists()


def test_a_file_goes_up_in_chunks():
    scratch = Path(tempfile.mkdtemp(prefix="upload-test-"))
    path = scratch / "render.bin"
    path.write_bytes(b"x" * 2_500_000)

    async def collect() -> list[int]:
        return [len(chunk) async for chunk in _file_chunks(path, chunk_size=1 << 20)]

    sizes = asyncio.run(collect())
    assert sizes == [1 << 20, 1 << 20, 2_500_000 - 2 * (1 << 20)]


def test_a_file_is_measured_where_it_is():
    scratch = _wav(2.0)
    seconds = _artifact_duration(scratch / "out.wav", "audio/wav")
    assert seconds is not None and abs(seconds - 2.0) < 0.05
