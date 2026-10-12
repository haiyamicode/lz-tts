"""Spoken language (and English accent) identification.

Used by the ``voice-enhance`` task and the standalone ``detect-language``
task. Two SpeechBrain ECAPA-TDNN classifiers (~21M params / 85 MB each):

- ``speechbrain/lang-id-voxlingua107-ecapa`` — 107 VoxLingua107 languages.
- ``Jzuluaga/accent-id-commonaccent_ecapa`` — 16 CommonAccent English
  accents, run only when the language is English.

Language scores fold confusable minority classes into their major neighbour
and drop languages without native speakers (LANGUAGE_FOLDS / LANGUAGE_DROPS);
``language.rawCandidates`` keeps the model's own top labels. When no folded
language reaches LANGUAGE_MIN_SCORE the result falls back to bare ``en``
(``language.fallback``; the accent model is skipped).

The result's ``locale`` is a BCP-47 tag: ``en-GB``-style when the language is
English and the accent is confident and maps to one region, otherwise the
bare language code (``vi``, ``zh``, ``en``).

In the worker both run on CPU inside the main worker process — no backend
child subprocess, no CUDA — and load lazily on first use so workers that
never identify never pay the load cost. Offline jobs (the voice language
backfill, scripts/backfill_language_id.py) pass ``device="cuda"`` and score
padded batches through ``identify_samples``. Model directory resolution: the pre-baked
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

# VoxLingua107 minority-language classes that, on voice-clone audio (accented
# speech, old recordings, YouTube rips), almost always absorb a neighbouring
# major language: Lazybird's backfill of ~1.2k clones put Attenborough, Richard
# Burton and "Older Brit Woman" under Welsh, Pedro Sánchez under Galician and a
# Russian sample under Belarusian. Their probability is folded into that
# language (so British English still reaches the accent model). The rest are
# the same pattern: regional languages, dialect continua and mutually
# intelligible standards that the model splits off their major neighbour.
LANGUAGE_FOLDS: dict[str, str] = {
    "cy": "en",
    "gv": "en",
    "sco": "en",
    "af": "en",
    "gl": "es",
    "be": "ru",
    "lb": "de",
    "yi": "de",
    "nn": "no",
    "fo": "is",
    "oc": "fr",
    "br": "fr",
    "jw": "id",
    "su": "id",
    "ceb": "tl",
    "war": "tl",
    "bs": "hr",
    "sr": "hr",
    "ur": "hi",
}
# Languages without native speakers: never a voice's spoken language. Their
# probability is dropped (not renormalized), so the best real language has to
# be confident on its own.
LANGUAGE_DROPS = frozenset({"la", "eo", "ia", "sa"})
# Below this folded score no language is trusted and the result falls back to
# bare "en" (language.fallback). Silence lands here too: VoxLingua107 scores
# digital silence as Nynorsk 0.27 / Welsh 0.14. On the ~1.2k-clone backfill,
# 104 clips (8%) fall below 0.5 — silence, music, chant and weak niche-language
# guesses (Javanese, Swahili, Maori, ...) — of which 23 were English already.
LANGUAGE_MIN_SCORE = 0.5
FALLBACK_LANGUAGE = {"code": "en", "name": "English"}

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
    """Lazy classifiers; returns the spoken language/locale of clips."""

    def __init__(self, model_root: str | Path | None = None, device: str = "cpu"):
        repo_root = Path(__file__).resolve().parents[2]
        self._model_root = (
            Path(model_root)
            if model_root is not None
            else repo_root / "data" / "language-id"
        )
        self._device = device
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
                    run_opts={"device": self._device},
                )
                model.eval()
                self._models[spec.snapshot_dir] = model
                _LOGGER.info("Language identifier loaded model=%s", spec.snapshot_dir)
                return model
            except Exception as exc:
                _LOGGER.exception("Language identifier load failed model=%s", spec.snapshot_dir)
                raise RuntimeError(f"Language identifier load failed: {exc}") from exc

    @staticmethod
    def _probabilities(
        model: Any, wavs: torch.Tensor, wav_lens: torch.Tensor | None, logit_scale: float
    ) -> tuple[torch.Tensor, list[str]]:
        """(B, classes) probabilities for a padded batch, plus the class labels."""
        with torch.inference_mode():
            outputs = model.classify_batch(wavs, wav_lens)[0]
        probabilities = torch.softmax(outputs.float().reshape(outputs.shape[0], -1) * logit_scale, dim=1)
        labels = model.hparams.label_encoder.decode_ndim(list(range(probabilities.shape[1])))
        return probabilities.cpu(), [str(label) for label in labels]

    @staticmethod
    def _top(row: list[float], labels: list[str], top_k: int) -> list[tuple[str, float]]:
        ranked = sorted(zip(labels, row, strict=True), key=lambda item: item[1], reverse=True)
        return [(label, round(score, 6)) for label, score in ranked[:top_k]]

    @staticmethod
    def _languages(row: list[float], labels: list[str], top_k: int) -> dict[str, Any]:
        """Folded language candidates ({code, name, score}) and the raw top labels."""
        names: dict[str, str] = {}
        folded: dict[str, float] = {}
        for label, score in zip(labels, row, strict=True):
            # VoxLingua107 labels are "vi: Vietnamese".
            code, _, name = label.partition(": ")
            names[code] = name or code
            if code in LANGUAGE_DROPS:
                continue
            target = LANGUAGE_FOLDS.get(code, code)
            folded[target] = folded.get(target, 0.0) + score
        ranked = sorted(folded.items(), key=lambda item: item[1], reverse=True)[:top_k]
        candidates = [{"code": code, "name": names[code], "score": round(score, 6)} for code, score in ranked]
        raw = []
        for label, score in SpokenLanguageIdentifier._top(row, labels, top_k):
            code, _, name = label.partition(": ")
            raw.append({"code": code, "name": name or code, "score": score})
        if candidates[0]["score"] >= LANGUAGE_MIN_SCORE:
            chosen = {**candidates[0], "fallback": False}
        else:
            score = round(folded.get(FALLBACK_LANGUAGE["code"], 0.0), 6)
            chosen = {**FALLBACK_LANGUAGE, "score": score, "fallback": True}
        return {"chosen": chosen, "candidates": candidates, "raw": raw}

    def _batch(self, clips: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zero-pad clips into one (B, T) tensor plus SpeechBrain relative lengths.

        Pooling masks the padding, but the convolutions see zeros past a
        short clip's end, so its scores drift slightly (~0.01 on
        low-confidence clips) from scoring it alone."""
        if len(clips) == 1:
            return torch.from_numpy(clips[0].copy()).unsqueeze(0).to(self._device), None
        longest = max(clip.size for clip in clips)
        wavs = torch.zeros(len(clips), longest, dtype=torch.float32)
        for row, clip in enumerate(clips):
            wavs[row, : clip.size] = torch.tensor(clip)
        lens = torch.tensor([clip.size / longest for clip in clips], dtype=torch.float32)
        return wavs.to(self._device), lens.to(self._device)

    def identify(self, audio: bytes, top_k: int = TOP_K) -> dict[str, Any]:
        """Identify the spoken language (and English accent) of one clip."""
        return self.identify_samples([decode_audio_16k_mono(audio)], top_k)[0]

    def identify_samples(self, clips: list[np.ndarray], top_k: int = TOP_K) -> list[dict[str, Any]]:
        """Identify decoded 16 kHz mono clips as one padded batch per model.

        Callers batching many clips should group similar lengths: padding
        costs compute (the masked pooling keeps it out of the result).
        """
        if not clips:
            return []
        wavs, lens = self._batch(clips)

        # VoxLingua107 outputs log-probabilities: softmax at scale 1 recovers them.
        probabilities, labels = self._probabilities(self._model(LANGUAGE_MODEL), wavs, lens, 1.0)
        languages_per_clip = [self._languages(row, labels, top_k) for row in probabilities.tolist()]

        # A fallback "en" is not evidence of English speech: no accent for it.
        english = [
            row
            for row, languages in enumerate(languages_per_clip)
            if languages["chosen"]["code"] == "en" and not languages["chosen"]["fallback"]
        ]
        accents_per_clip: dict[int, list[dict[str, Any]]] = {}
        if english:
            accent_wavs, accent_lens = (
                (wavs, lens) if len(english) == len(clips) else self._batch([clips[row] for row in english])
            )
            accent_probabilities, accent_labels = self._probabilities(
                self._model(ACCENT_MODEL), accent_wavs, accent_lens, ACCENT_COSINE_SCALE
            )
            for row, accent_row in zip(english, accent_probabilities.tolist(), strict=True):
                accents_per_clip[row] = [
                    {"code": label, "locale": ACCENT_LOCALES.get(label), "score": score}
                    for label, score in self._top(accent_row, accent_labels, top_k)
                ]

        results = []
        for row, (clip, languages) in enumerate(zip(clips, languages_per_clip, strict=True)):
            language = {
                **languages["chosen"],
                "candidates": languages["candidates"],
                "rawCandidates": languages["raw"],
            }
            locale = language["code"]
            accent = None
            if row in accents_per_clip:
                accents = accents_per_clip[row]
                accent = {**accents[0], "candidates": accents}
                locale = accent_locale(accent["code"], accent["score"])
            results.append(
                {
                    "locale": locale,
                    "language": language,
                    "accent": accent,
                    "analyzedSeconds": round(clip.size / SAMPLE_RATE, 3),
                    "models": {
                        "language": LANGUAGE_MODEL.snapshot_dir,
                        **({"accent": ACCENT_MODEL.snapshot_dir} if accent is not None else {}),
                    },
                }
            )
        return results


__all__ = [
    "ACCENT_LOCALES",
    "LANGUAGE_DROPS",
    "LANGUAGE_FOLDS",
    "LANGUAGE_MIN_SCORE",
    "LanguageDetectRequest",
    "LanguageInputError",
    "SpokenLanguageIdentifier",
    "accent_locale",
    "decode_audio_16k_mono",
]
