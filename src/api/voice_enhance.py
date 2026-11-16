"""Standalone voice-sample normalization used by the voice-enhance task."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

_VENV_BIN = Path(sys.prefix) / "bin"

from pydantic import BaseModel


class VoiceEnhanceRequest(BaseModel):
    reference_url: str
    id: str


class VoiceEnhancer:
    """Normalize a voice sample without loading or invoking a speech model."""

    def __init__(self, temp_dir: str | Path):
        root = Path(temp_dir)
        self.temp_dir = root if root.is_absolute() else Path(__file__).resolve().parents[2] / root
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _run(command: list[str]) -> None:
        try:
            subprocess.run(command, capture_output=True, check=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Voice enhancement command failed ({command[0]}): {stderr[-2000:]}"
            ) from exc

    def workspace(self) -> tempfile.TemporaryDirectory:
        """Where the sample is fetched and worked on; it goes when the enhancement is done."""
        return tempfile.TemporaryDirectory(prefix="voice-enhance-", dir=self.temp_dir)

    def enhance(self, raw_path: Path) -> bytes:
        """The normalized 320 kbps MP3 of the sample at ``raw_path`` — at most two minutes of it.

        The sample is read by ffmpeg from where it was fetched; only the enhanced take comes back
        as bytes, because that is what the language identifier reads and what the task returns.
        """
        if not raw_path.is_file() or not raw_path.stat().st_size:
            raise ValueError("Voice enhancement input is empty")

        with tempfile.TemporaryDirectory(prefix="voice-enhance-", dir=self.temp_dir) as workspace:
            workspace_path = Path(workspace)
            wav_path = workspace_path / "sample_raw.wav"
            normalized_path = workspace_path / "sample.wav"
            output_path = workspace_path / "sample.mp3"

            self._run(["ffmpeg", "-i", str(raw_path), "-t", "120", str(wav_path), "-y"])
            self._run(
                [
                    str(_VENV_BIN / "ffmpeg-normalize"),
                    str(wav_path),
                    "-o",
                    str(normalized_path),
                    "-f",
                ]
            )
            self._run(
                [
                    "ffmpeg",
                    "-i",
                    str(normalized_path),
                    "-f",
                    "mp3",
                    "-q:a",
                    "0",
                    "-b:a",
                    "320k",
                    str(output_path),
                    "-y",
                ]
            )
            return output_path.read_bytes()


__all__ = ["VoiceEnhanceRequest", "VoiceEnhancer"]
