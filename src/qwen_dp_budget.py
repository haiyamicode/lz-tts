"""Duration-predictor budget for Qwen3-TTS codec-token caps."""

from __future__ import annotations

import gc
import importlib.machinery
import json
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class DpBudgetConfig:
    checkpoint: Path = Path("data/lzspeech-gm/glow_tts.pt")
    device: str = "cpu"
    language: str = "multilingual"
    noise_scale: float = 0.8
    semantic_guidance_scale: float = 1.7
    length_scale: float = 1.0
    token_rate: float = 12.0
    samples: int = 32
    upper_quantile: float = 0.90
    min_margin: float = 1.0
    max_margin: float = 1.25
    min_extra_tokens: int = 0
    max_extra_tokens: int = 36
    bert_model: str = "distilbert-base-multilingual-cased"
    fusion_weight: float = 0.5
    language_profiles: dict[str, dict[str, float | int]] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "DpBudgetConfig":
        language_profiles = parse_language_profiles(os.environ.get("QWEN_DP_BUDGET_LANGUAGE_PROFILES", ""))
        return cls(
            checkpoint=Path(os.environ.get("QWEN_DP_BUDGET_CHECKPOINT", str(cls.checkpoint))),
            device=os.environ.get("QWEN_DP_BUDGET_DEVICE", cls.device),
            language=os.environ.get("QWEN_DP_BUDGET_LANGUAGE", cls.language),
            noise_scale=float(os.environ.get("QWEN_DP_BUDGET_NOISE_SCALE", cls.noise_scale)),
            semantic_guidance_scale=float(
                os.environ.get("QWEN_DP_BUDGET_SEMANTIC_GUIDANCE_SCALE", cls.semantic_guidance_scale)
            ),
            length_scale=float(os.environ.get("QWEN_DP_BUDGET_LENGTH_SCALE", cls.length_scale)),
            token_rate=float(os.environ.get("QWEN_DP_BUDGET_TOKEN_RATE", cls.token_rate)),
            samples=int(os.environ.get("QWEN_DP_BUDGET_SAMPLES", cls.samples)),
            upper_quantile=float(os.environ.get("QWEN_DP_BUDGET_UPPER_QUANTILE", cls.upper_quantile)),
            min_margin=float(os.environ.get("QWEN_DP_BUDGET_MIN_MARGIN", cls.min_margin)),
            max_margin=float(os.environ.get("QWEN_DP_BUDGET_MAX_MARGIN", cls.max_margin)),
            min_extra_tokens=int(os.environ.get("QWEN_DP_BUDGET_MIN_EXTRA_TOKENS", cls.min_extra_tokens)),
            max_extra_tokens=int(os.environ.get("QWEN_DP_BUDGET_MAX_EXTRA_TOKENS", cls.max_extra_tokens)),
            bert_model=os.environ.get("QWEN_DP_BUDGET_BERT_MODEL", cls.bert_model),
            fusion_weight=float(os.environ.get("QWEN_DP_BUDGET_FUSION_WEIGHT", cls.fusion_weight)),
            language_profiles=language_profiles,
        )


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
        self.config = config or DpBudgetConfig.from_env()
        self.device = torch.device(self.config.device)
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._pp = None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            exp_dir = Path(__file__).resolve().parents[1] / "local" / "exp"
            if str(exp_dir) not in sys.path:
                sys.path.insert(0, str(exp_dir))

            if "matplotlib" not in sys.modules:
                matplotlib = types.ModuleType("matplotlib")
                matplotlib.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
                matplotlib.use = lambda *_args, **_kwargs: None
                pyplot = types.ModuleType("matplotlib.pyplot")
                pyplot.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", loader=None)
                sys.modules["matplotlib"] = matplotlib
                sys.modules["matplotlib.pyplot"] = pyplot
            if "torch.utils.tensorboard" not in sys.modules:
                tensorboard = types.ModuleType("torch.utils.tensorboard")
                tensorboard.__spec__ = importlib.machinery.ModuleSpec("torch.utils.tensorboard", loader=None)
                tensorboard.SummaryWriter = lambda *_args, **_kwargs: None
                sys.modules["torch.utils.tensorboard"] = tensorboard

            import piper_phonemize as pp
            from glow_tts_v2 import GlowTTS
            from transformers import AutoTokenizer

            model = GlowTTS(
                n_vocab=256,
                hidden_channels=192,
                filter_channels=768,
                filter_channels_dp=256,
                out_channels=80,
                kernel_size=3,
                n_heads=2,
                n_layers_enc=6,
                p_dropout=0.1,
                n_blocks_dec=12,
                kernel_size_dec=5,
                dilation_rate=1,
                n_block_layers=4,
                p_dropout_dec=0.05,
                n_split=4,
                n_sqz=2,
                sigmoid_scale=False,
                window_size=4,
                mean_only=True,
                prenet=True,
                stochastic_duration=True,
                use_semantic=True,
                bert_model=self.config.bert_model,
                fusion_weight=self.config.fusion_weight,
            )
            checkpoint = torch.load(self.config.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.decoder = None
            model.eval().to(self.device)

            self._model = model
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model)
            self._pp = pp
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
        assert self._tokenizer is not None
        assert self._pp is not None

        from src.piper.preprocess import phonemize_text_for_infer

        phoneme_result = phonemize_text_for_infer(
            text,
            self._phoneme_config(language),
            neural=True,
        )
        phoneme_ids = phoneme_result["phoneme_ids"]
        if not phoneme_ids:
            return self._empty_budget(language)

        x = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)
        x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.long, device=self.device)
        bert_encoded = self._tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        bert_input = {
            "input_ids": bert_encoded["input_ids"].to(self.device),
            "attention_mask": bert_encoded["attention_mask"].to(self.device),
        }

        frame_values = []
        for _ in range(max(1, self.config.samples)):
            _, _, logw, x_mask = self._model.encoder.infer_durations(
                x,
                x_lengths,
                noise_scale=self.config.noise_scale,
                bert_input=bert_input,
            )
            w = torch.exp(logw) * x_mask * self.config.length_scale
            w_ceil = torch.ceil(w)
            frame_values.append(int(torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).max().item()))

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
            "sample_frames": frame_values,
            "sample_seconds": [round(float(s), 3) for s in seconds_tensor.tolist()],
            "mean_seconds": float(torch.mean(seconds_tensor).item()),
            "p50_seconds": float(torch.quantile(seconds_tensor, 0.50).item()),
            "p90_seconds": float(torch.quantile(seconds_tensor, 0.90).item()),
            "length_scale": self.config.length_scale,
            "noise_scale": self.config.noise_scale,
            "semantic_guidance_scale": self.config.semantic_guidance_scale,
        }

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
            "sample_frames": [],
            "sample_seconds": [],
            "mean_seconds": 0.0,
            "p50_seconds": 0.0,
            "p90_seconds": 0.0,
            "length_scale": self.config.length_scale,
            "noise_scale": self.config.noise_scale,
            "semantic_guidance_scale": self.config.semantic_guidance_scale,
        }
