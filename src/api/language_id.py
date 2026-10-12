"""Spoken language identification for the ``voice-enhance`` task.

SpeechBrain ``lang-id-voxlingua107-ecapa`` (ECAPA-TDNN, 21M params / 85 MB,
107 VoxLingua107 languages, 16 kHz mono input). Runs on CPU inside the main
worker process — no backend child subprocess, no CUDA. Weights load lazily on
the first identification so workers that never enhance never pay the load
cost.

Model directory resolution: the pre-baked ``data/language-id`` snapshot
(baked into Docker images via scripts/download_data.py) when present, else
the Hugging Face hub.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)

HUB_MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"
SNAPSHOT_DIR = "lang-id-voxlingua107-ecapa"
SAMPLE_RATE = 16000
# The language of a voice sample is settled well within a minute; CPU cost is
# linear in length (~0.9 s for 60 s on the dev box).
MAX_SECONDS = 60
TOP_K = 3


class LanguageInputError(ValueError):
    """Audio that cannot be decoded or contains no samples."""


def decode_audio_16k_mono(audio: bytes, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Decode any ffmpeg-readable audio to float32 16 kHz mono samples."""
    if not audio:
        raise LanguageInputError("Language identification input is empty")
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error",
                "-i", "pipe:0",
                "-t", str(max_seconds),
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "f32le", "pipe:1",
            ],
            input=audio,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise LanguageInputError(f"Undecodable audio: {stderr[-500:]}") from exc
    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    if samples.size == 0:
        raise LanguageInputError("Decoded audio contains no samples")
    return samples


class SpokenLanguageIdentifier:
    """Lazy CPU classifier; returns the top spoken languages for one clip."""

    def __init__(self, model_root: str | Path | None = None):
        repo_root = Path(__file__).resolve().parents[2]
        self._model_root = (
            Path(model_root)
            if model_root is not None
            else repo_root / "data" / "language-id"
        )
        self._snapshot_dir = self._model_root / SNAPSHOT_DIR
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_source(self) -> str:
        """Pre-baked snapshot when available, else the hub model id."""
        if (self._snapshot_dir / "embedding_model.ckpt").exists():
            return str(self._snapshot_dir)
        return HUB_MODEL_ID

    @property
    def model_label(self) -> str:
        """Short model name for result payloads."""
        return Path(self.model_source).name

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from speechbrain.inference.classifiers import EncoderClassifier

                source = self.model_source
                _LOGGER.info("Language identifier loading model=%s", self.model_label)
                # SpeechBrain links every checkpoint into ``savedir``; keep
                # those links out of the baked data directory.
                savedir = Path(tempfile.gettempdir()) / "lz-tts-language-id"
                model = EncoderClassifier.from_hparams(
                    source=source,
                    savedir=str(savedir),
                    overrides={"pretrained_path": source},
                    run_opts={"device": "cpu"},
                )
                model.eval()
                self._model = model
                _LOGGER.info("Language identifier loaded model=%s", self.model_label)
            except Exception as exc:
                _LOGGER.exception("Language identifier load failed model=%s", self.model_label)
                raise RuntimeError(f"Language identifier load failed: {exc}") from exc

    def identify(self, audio: bytes, top_k: int = TOP_K) -> dict[str, Any]:
        """Identify the spoken language of one encoded audio clip."""
        samples = decode_audio_16k_mono(audio)
        self._ensure_loaded()
        wavs = torch.from_numpy(samples.copy()).unsqueeze(0)
        with torch.inference_mode():
            log_probs = self._model.classify_batch(wavs)[0][0]
        probabilities = torch.softmax(log_probs.float(), dim=0)
        scores, indices = probabilities.topk(min(top_k, probabilities.numel()))
        labels = self._model.hparams.label_encoder.decode_ndim(indices.tolist())

        candidates = []
        for label, score in zip(labels, scores.tolist(), strict=True):
            code, _, name = str(label).partition(": ")
            candidates.append({"code": code, "name": name or code, "score": round(float(score), 6)})
        top = candidates[0]
        return {
            "code": top["code"],
            "name": top["name"],
            "score": top["score"],
            "candidates": candidates,
            "analyzedSeconds": round(samples.size / SAMPLE_RATE, 3),
            "model": self.model_label,
        }


__all__ = [
    "LanguageInputError",
    "SpokenLanguageIdentifier",
    "decode_audio_16k_mono",
]
