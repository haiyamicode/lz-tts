"""Duration-predictor budget for Qwen3-TTS codec-token caps."""

from __future__ import annotations

import gc
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class DpBudgetConfig:
    checkpoint: Path = Path("data/lzspeech-multilingual-bert/lzspeech-multilingual-bert-189.ckpt")
    config_path: Optional[Path] = None
    device: str = "cpu"
    language: str = "multilingual"
    noise_scale: float = 0.8
    length_scale: float = 1.0
    token_rate: float = 12.0
    samples: int = 32
    upper_quantile: float = 0.90
    min_margin: float = 1.0
    max_margin: float = 1.25
    min_extra_tokens: int = 0
    max_extra_tokens: int = 36
    language_profiles: dict[str, dict[str, float | int]] = field(default_factory=dict)


LANGUAGE_ALIASES = {
    "auto": "auto",
    "default": "default",
    "multilingual": "multilingual",
    "zh": "chinese",
    "zho": "chinese",
    "chinese": "chinese",
    "en": "english",
    "eng": "english",
    "english": "english",
    "ja": "japanese",
    "jpn": "japanese",
    "japanese": "japanese",
    "ko": "korean",
    "kor": "korean",
    "korean": "korean",
    "de": "german",
    "deu": "german",
    "ger": "german",
    "german": "german",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "french": "french",
    "ru": "russian",
    "rus": "russian",
    "russian": "russian",
    "pt": "portuguese",
    "por": "portuguese",
    "portuguese": "portuguese",
    "es": "spanish",
    "spa": "spanish",
    "spanish": "spanish",
    "it": "italian",
    "ita": "italian",
    "italian": "italian",
}


def normalize_language_key(language: str | None) -> str:
    if not language:
        return "default"
    key = language.strip().lower().replace("_", "-")
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    base = key.split("-", 1)[0]
    return LANGUAGE_ALIASES.get(base, key)


def parse_language_profiles(raw: str) -> dict[str, dict[str, float | int]]:
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("QWEN_DP_BUDGET_LANGUAGE_PROFILES must be a JSON object")

    profiles: dict[str, dict[str, float | int]] = {}
    for language, values in parsed.items():
        if not isinstance(values, dict):
            raise ValueError(f"DP budget profile for {language!r} must be an object")
        profile: dict[str, float | int] = {}
        for key in ("min_margin", "max_margin"):
            if key in values:
                profile[key] = float(values[key])
        for key in ("min_extra_tokens", "max_extra_tokens"):
            if key in values:
                profile[key] = int(values[key])
        profiles[normalize_language_key(str(language))] = profile
    return profiles


class QwenDpBudget:
    """Predict a conservative Qwen codec-token cap from duration samples."""

    def __init__(self, config: DpBudgetConfig | None = None):
        self.config = config or DpBudgetConfig()
        self.device = torch.device(self.config.device)
        self._lock = threading.Lock()
        self._model = None
        self._model_config: dict[str, Any] = {}
        self._semantic_tokenizer = None
        self._build_bert_input = None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            from src.piper.semantic import SemanticTokenizer, build_bert_input
            from src.piper.vits.lightning import VitsModel

            checkpoint_path = Path(self.config.checkpoint)
            config_path = Path(self.config.config_path) if self.config.config_path else checkpoint_path.parent / "config.json"
            with config_path.open("r", encoding="utf-8") as f:
                self._model_config = json.load(f)

            model = VitsModel.load_from_checkpoint(
                str(checkpoint_path),
                dataset=None,
                weights_only=False,
            )
            model.eval()

            model_g = model.model_g
            model_g.dec = None
            model_g.enc_q = None
            model_g.flow = None
            gc.collect()

            self._sync_config_from_checkpoint(model)
            if bool(getattr(model.hparams, "use_bert", False)):
                bert_model_name = getattr(model.hparams, "bert_model_name", None)
                self._semantic_tokenizer = SemanticTokenizer(model_name=bert_model_name)
                self._build_bert_input = build_bert_input

            self._model = model_g.to(self.device).eval()
            gc.collect()

    def _phoneme_config(self, language: str | None = None) -> dict[str, Any]:
        language = (language or self.config.language).strip()
        language_lower = language.lower()
        if language_lower in {"auto", "multilingual"}:
            return {
                "language": {"code": "multilingual"},
                "espeak": {"voice": "multilingual", "primary": "en-us"},
            }

        language_map = {
            "english": "en-us",
            "japanese": "ja",
            "chinese": "zh",
            "korean": "ko",
            "french": "fr-fr",
            "german": "de",
            "spanish": "es",
            "russian": "ru",
            "portuguese": "pt",
            "italian": "it",
        }
        voice = language_map.get(language_lower, language)
        return {
            "language": {"code": voice},
            "espeak": {"voice": voice, "primary": "en-us"},
        }

    def _sync_config_from_checkpoint(self, model: Any) -> None:
        speaker_map = getattr(model.hparams, "speaker_id_map", None)
        if isinstance(speaker_map, dict) and speaker_map:
            self._model_config["speaker_id_map"] = {
                str(label): int(sid)
                for label, sid in speaker_map.items()
            }
        num_speakers = getattr(model.hparams, "num_speakers", None)
        if isinstance(num_speakers, int):
            self._model_config["num_speakers"] = num_speakers

    def _speaker_id_for_language(self, language: str | None) -> int:
        speaker_map = self._model_config.get("speaker_id_map") or {}
        language_speakers = self._model_config.get("language_speakers") or {}
        if not speaker_map:
            return 0

        language_value = (language or self.config.language).strip()
        if not language_value or language_value.lower() in {"auto", "multilingual"}:
            language_value = (self._model_config.get("espeak") or {}).get("primary") or "en-us"

        normalized = language_value.replace("_", "-")
        base = normalized.split("-", 1)[0].lower()
        candidates = [
            normalized,
            normalized.lower(),
            language_speakers.get(normalized),
            language_speakers.get(normalized.lower()),
            base,
            language_speakers.get(base),
        ]
        for candidate in candidates:
            if candidate is not None and candidate in speaker_map:
                return int(speaker_map[candidate])
        return int(next(iter(speaker_map.values())))

    def _budget_profile(self, language: str | None) -> tuple[str, dict[str, float | int]]:
        language_key = normalize_language_key(language)
        profiles = self.config.language_profiles
        profile = profiles.get(language_key) or profiles.get("default") or {}
        return language_key, profile

    @staticmethod
    def _profile_float(profile: dict[str, float | int], key: str, default: float) -> float:
        return float(profile.get(key, default))

    @staticmethod
    def _profile_int(profile: dict[str, float | int], key: str, default: int) -> int:
        return int(profile.get(key, default))

    @torch.no_grad()
    def predict(self, text: str, language: str | None = None) -> dict[str, Any]:
        self.load()
        assert self._model is not None

        from src.piper.preprocess import phonemize_text_for_infer

        phoneme_result = phonemize_text_for_infer(
            text,
            self._phoneme_config(language),
            neural=True,
        )
        phoneme_ids = phoneme_result["phoneme_ids"]
        if not phoneme_ids:
            return self._empty_budget(language)

        frame_values = []
        speaker_id = self._speaker_id_for_language(language)
        x = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)
        x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.long, device=self.device)
        sid = torch.tensor([speaker_id], dtype=torch.long, device=self.device)
        bert_input = self._bert_input(text)
        for _ in range(max(1, self.config.samples)):
            frame_values.append(self._predict_frames(x, x_lengths, sid, bert_input))

        frame_tensor = torch.tensor(frame_values, dtype=torch.float32)
        seconds_tensor = frame_tensor * (256.0 / 22050.0)
        token_tensor = seconds_tensor * self.config.token_rate

        profile_language, profile = self._budget_profile(language)
        min_margin = self._profile_float(profile, "min_margin", self.config.min_margin)
        max_margin = self._profile_float(profile, "max_margin", self.config.max_margin)
        min_extra_tokens = self._profile_int(profile, "min_extra_tokens", self.config.min_extra_tokens)
        max_extra_tokens = self._profile_int(profile, "max_extra_tokens", self.config.max_extra_tokens)
        quantile = self.config.upper_quantile
        mel_frames = int(torch.quantile(frame_tensor, quantile).ceil().item())
        seconds = float(torch.quantile(seconds_tensor, quantile).item())
        estimated_tokens = max(1, round(float(torch.quantile(token_tensor, quantile).item())))
        min_tokens = max(1, round(estimated_tokens * min_margin) + min_extra_tokens)
        max_tokens = max(min_tokens, round(estimated_tokens * max_margin) + max_extra_tokens)

        return {
            "mel_frames": mel_frames,
            "seconds": seconds,
            "estimated_tokens": estimated_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "token_rate": self.config.token_rate,
            "samples": self.config.samples,
            "upper_quantile": self.config.upper_quantile,
            "budget_language": profile_language,
            "budget_profile": profile,
            "min_margin": min_margin,
            "max_margin": max_margin,
            "min_extra_tokens": min_extra_tokens,
            "max_extra_tokens": max_extra_tokens,
            "phoneme_count": len(phoneme_ids),
            "phoneme_language": self._phoneme_config(language)["language"]["code"],
            "speaker_id": speaker_id,
            "sample_frames": frame_values,
            "sample_seconds": [round(float(s), 3) for s in seconds_tensor.tolist()],
            "mean_seconds": float(torch.mean(seconds_tensor).item()),
            "p50_seconds": float(torch.quantile(seconds_tensor, 0.50).item()),
            "p90_seconds": float(torch.quantile(seconds_tensor, 0.90).item()),
            "length_scale": self.config.length_scale,
            "noise_scale": self.config.noise_scale,
        }

    def _bert_input(self, text: str) -> dict[str, torch.Tensor] | None:
        if self._semantic_tokenizer is None or self._build_bert_input is None or not text:
            return None
        bert_dict = self._build_bert_input([text], self._semantic_tokenizer)
        if bert_dict is None:
            return None
        return {
            "input_ids": bert_dict["input_ids"].to(self.device),
            "attention_mask": bert_dict["attention_mask"].to(self.device),
        }

    def _predict_frames(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        bert_input: dict[str, torch.Tensor] | None,
    ) -> int:
        assert self._model is not None
        model = self._model
        if bert_input is not None:
            x_encoded, _m_p, _logs_p, x_mask = model.enc_p(
                x,
                x_lengths,
                bert_input=bert_input,
            )
        else:
            x_encoded, _m_p, _logs_p, x_mask = model.enc_p(x, x_lengths)

        if model.n_speakers > 1:
            g = model.emb_g(sid).unsqueeze(-1)
        else:
            g = None

        if model.use_sdp:
            logw = model.dp(x_encoded, x_mask, g=g, reverse=True, noise_scale=self.config.noise_scale)
        else:
            logw = model.dp(x_encoded, x_mask, g=g)
        w = torch.exp(logw) * x_mask * self.config.length_scale
        w_ceil = torch.ceil(w)
        return int(torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).max().item())

    def _empty_budget(self, language: str | None = None) -> dict[str, Any]:
        profile_language, profile = self._budget_profile(language)
        min_margin = self._profile_float(profile, "min_margin", self.config.min_margin)
        max_margin = self._profile_float(profile, "max_margin", self.config.max_margin)
        min_extra_tokens = self._profile_int(profile, "min_extra_tokens", self.config.min_extra_tokens)
        max_extra_tokens = self._profile_int(profile, "max_extra_tokens", self.config.max_extra_tokens)
        return {
            "mel_frames": 0,
            "seconds": 0.0,
            "estimated_tokens": 1,
            "min_tokens": 1,
            "max_tokens": 1,
            "token_rate": self.config.token_rate,
            "samples": 0,
            "upper_quantile": self.config.upper_quantile,
            "budget_language": profile_language,
            "budget_profile": profile,
            "min_margin": min_margin,
            "max_margin": max_margin,
            "min_extra_tokens": min_extra_tokens,
            "max_extra_tokens": max_extra_tokens,
            "phoneme_count": 0,
            "phoneme_language": self.config.language,
            "speaker_id": self._speaker_id_for_language(language),
            "sample_frames": [],
            "sample_seconds": [],
            "mean_seconds": 0.0,
            "p50_seconds": 0.0,
            "p90_seconds": 0.0,
            "length_scale": self.config.length_scale,
            "noise_scale": self.config.noise_scale,
        }
