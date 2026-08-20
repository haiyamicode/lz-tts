"""Standalone voice-sample normalization used by the voice-enhance task."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

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

    def enhance(self, input_audio: bytes) -> bytes:
        if not input_audio:
            raise ValueError("Voice enhancement input is empty")

        with tempfile.TemporaryDirectory(prefix="voice-enhance-", dir=self.temp_dir) as workspace:
            workspace_path = Path(workspace)
            raw_path = workspace_path / "sample_raw.input"
            wav_path = workspace_path / "sample_raw.wav"
            normalized_path = workspace_path / "sample.wav"
            output_path = workspace_path / "sample.mp3"
            raw_path.write_bytes(input_audio)

            self._run(["ffmpeg", "-i", str(raw_path), "-t", "120", str(wav_path), "-y"])
            self._run(
                [
                    "uv",
                    "tool",
                    "run",
                    "ffmpeg-normalize",
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
