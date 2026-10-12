"""Spoken language (and English accent) identification.

Used by the ``voice-enhance`` task and the standalone ``detect-language``
task. Two SpeechBrain ECAPA-TDNN classifiers (~21M params / 85 MB each):

- ``speechbrain/lang-id-voxlingua107-ecapa`` — 107 VoxLingua107 languages.
- ``Jzuluaga/accent-id-commonaccent_ecapa`` — 16 CommonAccent English
  accents, run only when the language is English.

The result's ``locale`` is a BCP-47 tag: ``en-GB``-style when the language is
English and the accent is confident and maps to one region, otherwise the
bare language code (``vi``, ``zh``, ``en``).

Both run on CPU inside the main worker process — no backend child
subprocess, no CUDA — and load lazily on first use so workers that never
identify never pay the load cost. Model directory resolution: the pre-baked
``data/language-id`` snapshots (baked into Docker images via
scripts/download_data.py) when present, else the Hugging Face hub.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# The language of a voice sample is settled well within a minute; CPU cost is
# linear in length (~0.9 s per model for 60 s on the dev box).
MAX_SECONDS = 60
TOP_K = 3

# The accent classifier ends in a cosine head trained with an additive
# angular margin loss at scale 30; softmax(30 * cos) recovers its posterior.
ACCENT_COSINE_SCALE = 30.0
# Below this the accent is reported but the locale stays the bare "en".
# On VCTK (out of domain) and GLOBE speakers, >=0.9 keeps ~80% of speakers
# and lifts top-1 accent accuracy from 0.59 to 0.67 (VCTK) / 0.72 to 0.82 (GLOBE).
ACCENT_MIN_SCORE = 0.9

# CommonAccent labels -> BCP-47 English locale. Labels without one region
# (pan-African, South Atlantic) fall back to "en". The upstream label is
# spelled "southatlandtic".
ACCENT_LOCALES: dict[str, str | None] = {
    "us": "en-US",
    "england": "en-GB",
    "scotland": "en-GB",
    "wales": "en-GB",
    "ireland": "en-IE",
    "canada": "en-CA",
    "australia": "en-AU",
    "newzealand": "en-NZ",
    "indian": "en-IN",
    "malaysia": "en-MY",
    "philippines": "en-PH",
    "singapore": "en-SG",
    "hongkong": "en-HK",
    "bermuda": "en-BM",
    "african": None,
    "southatlandtic": None,
}


@dataclass(frozen=True)
class _ModelSpec:
    hub_id: str
    snapshot_dir: str


LANGUAGE_MODEL = _ModelSpec("speechbrain/lang-id-voxlingua107-ecapa", "lang-id-voxlingua107-ecapa")
ACCENT_MODEL = _ModelSpec("Jzuluaga/accent-id-commonaccent_ecapa", "accent-id-commonaccent_ecapa")


class LanguageDetectRequest(BaseModel):
    """Input for the ``detect-language`` operation (one audio clip)."""

    audio_url: str | None = None
    audio_base64: str | None = None
    top_k: int = Field(default=TOP_K, ge=1, le=10)


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


def accent_locale(accent: str, score: float) -> str:
    """Locale for an English accent prediction: region tag or bare "en"."""
    if score < ACCENT_MIN_SCORE:
        return "en"
    return ACCENT_LOCALES.get(accent) or "en"


class SpokenLanguageIdentifier:
    """Lazy CPU classifiers; returns the spoken language/locale for one clip."""

    def __init__(self, model_root: str | Path | None = None):
        repo_root = Path(__file__).resolve().parents[2]
        self._model_root = (
            Path(model_root)
            if model_root is not None
            else repo_root / "data" / "language-id"
        )
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def model_source(self, spec: _ModelSpec) -> str:
        """Pre-baked snapshot when available, else the hub model id."""
        snapshot = self._model_root / spec.snapshot_dir
        if (snapshot / "embedding_model.ckpt").exists():
            return str(snapshot)
        return spec.hub_id

    def _model(self, spec: _ModelSpec) -> Any:
        with self._lock:
            model = self._models.get(spec.snapshot_dir)
            if model is not None:
                return model
            try:
                from speechbrain.inference.classifiers import EncoderClassifier

                source = self.model_source(spec)
                _LOGGER.info("Language identifier loading model=%s", spec.snapshot_dir)
                # SpeechBrain links every checkpoint into ``savedir``; keep
                # those links out of the baked data directory.
                savedir = Path(tempfile.gettempdir()) / "lz-tts-language-id" / spec.snapshot_dir
                model = EncoderClassifier.from_hparams(
                    source=source,
                    savedir=str(savedir),
                    overrides={"pretrained_path": source},
                    run_opts={"device": "cpu"},
                )
                model.eval()
                self._models[spec.snapshot_dir] = model
                _LOGGER.info("Language identifier loaded model=%s", spec.snapshot_dir)
                return model
            except Exception as exc:
                _LOGGER.exception("Language identifier load failed model=%s", spec.snapshot_dir)
                raise RuntimeError(f"Language identifier load failed: {exc}") from exc

    @staticmethod
    def _top(model: Any, wavs: torch.Tensor, top_k: int, logit_scale: float) -> list[tuple[str, float]]:
        with torch.inference_mode():
            outputs = model.classify_batch(wavs)[0][0]
        probabilities = torch.softmax(outputs.float() * logit_scale, dim=0)
        scores, indices = probabilities.topk(min(top_k, probabilities.numel()))
        labels = model.hparams.label_encoder.decode_ndim(indices.tolist())
        return [(str(label), round(float(score), 6)) for label, score in zip(labels, scores.tolist(), strict=True)]

    def identify(self, audio: bytes, top_k: int = TOP_K) -> dict[str, Any]:
        """Identify the spoken language (and English accent) of one clip."""
        samples = decode_audio_16k_mono(audio)
        wavs = torch.from_numpy(samples.copy()).unsqueeze(0)

        # VoxLingua107 outputs log-probabilities; labels are "vi: Vietnamese".
        languages = []
        for label, score in self._top(self._model(LANGUAGE_MODEL), wavs, top_k, 1.0):
            code, _, name = label.partition(": ")
            languages.append({"code": code, "name": name or code, "score": score})
        language = {**languages[0], "candidates": languages}
        locale = language["code"]

        accent = None
        if language["code"] == "en":
            accents = [
                {"code": label, "locale": ACCENT_LOCALES.get(label), "score": score}
                for label, score in self._top(self._model(ACCENT_MODEL), wavs, top_k, ACCENT_COSINE_SCALE)
            ]
            accent = {**accents[0], "candidates": accents}
            locale = accent_locale(accent["code"], accent["score"])

        return {
            "locale": locale,
            "language": language,
            "accent": accent,
            "analyzedSeconds": round(samples.size / SAMPLE_RATE, 3),
            "models": {
                "language": LANGUAGE_MODEL.snapshot_dir,
                **({"accent": ACCENT_MODEL.snapshot_dir} if accent is not None else {}),
            },
        }


__all__ = [
    "ACCENT_LOCALES",
    "LanguageDetectRequest",
    "LanguageInputError",
    "SpokenLanguageIdentifier",
    "accent_locale",
    "decode_audio_16k_mono",
]
