"""Duration-predictor budget for Qwen3-TTS codec-token caps."""

from __future__ import annotations

import gc
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DpBudgetConfig:
    checkpoint: Path = Path("data/lzspeech-sparrow/model.ckpt")
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
    use_bert: bool = False
    enable_alignment_validation: bool = False


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
    "vi": "vietnamese",
    "vie": "vietnamese",
    "vietnamese": "vietnamese",
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
        self._semantic_model = None
        self._build_bert_input = None
        self._align_phone_features = None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            from src.piper.vits.lightning import VitsModel

            checkpoint_path = Path(self.config.checkpoint)
            config_path = Path(self.config.config_path) if self.config.config_path else checkpoint_path.parent / "config.json"
            _LOGGER.info("Loading Qwen DP budget checkpoint=%s config=%s", checkpoint_path, config_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Qwen DP budget checkpoint not found: {checkpoint_path}")
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Qwen DP budget config not found: {config_path} "
                    f"(checkpoint={checkpoint_path})"
                )
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
            if not self.config.enable_alignment_validation:
                model_g.enc_q = None
                model_g.flow = None
            gc.collect()

            self._sync_config_from_checkpoint(model)
            checkpoint_uses_bert = bool(getattr(model.hparams, "use_bert", False))
            if checkpoint_uses_bert and self.config.use_bert:
                from src.piper.semantic import SemanticTokenizer, build_bert_input
                from src.piper.semantic import align_phone_features

                bert_model_name = getattr(model.hparams, "bert_model_name", None)
                self._semantic_tokenizer = SemanticTokenizer(model_name=bert_model_name)
                self._build_bert_input = build_bert_input
                self._align_phone_features = align_phone_features
                enc_p = getattr(model_g, "enc_p", None)
                if bool(getattr(enc_p, "bert_features_precomputed", False)):
                    from transformers import AutoModel

                    from src.piper.hf_cache import resolve_hf_model_path

                    semantic_path = resolve_hf_model_path(bert_model_name, require_weights=True)
                    self._semantic_model = AutoModel.from_pretrained(semantic_path).to(self.device).eval()
            elif checkpoint_uses_bert:
                self._strip_bert_from_text_encoder(model_g)

            self._model = model_g.to(self.device).eval()
            gc.collect()

    @staticmethod
    def _strip_bert_from_text_encoder(model_g: Any) -> None:
        """Drop semantic-only modules from the DP budget model.

        The DP budget only needs the phoneme encoder and duration predictor.
        BertTextEncoder already supports bert_input=None, so removing these
        modules keeps that path while avoiding a ~514 MiB BERT allocation.
        """
        enc_p = getattr(model_g, "enc_p", None)
        if enc_p is None:
            return
        for attr in ("bert", "bert_projection", "cross_attention", "layer_norm"):
            if hasattr(enc_p, attr):
                setattr(enc_p, attr, None)

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
            "vietnamese": "vi",
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
        return self.predict_batch([text], [language])[0]

    @torch.no_grad()
    def predict_batch(
        self,
        texts: list[str],
        languages: list[str | None] | None = None,
    ) -> list[dict[str, Any]]:
        self.load()
        assert self._model is not None

        if languages is None:
            languages = [None] * len(texts)
        if len(texts) != len(languages):
            raise ValueError("languages length must match texts length")

        from src.piper.preprocess import phonemize_text_for_infer

        phoneme_results: list[dict[str, Any]] = []
        for text, language in zip(texts, languages):
            phoneme_result = phonemize_text_for_infer(
                text,
                self._phoneme_config(language),
                neural=False,
            )
            phoneme_results.append(phoneme_result)

        budgets = [self._empty_budget(language) for language in languages]
        batch_entries: list[tuple[int, str | None, list[int], dict[str, Any]]] = []
        for index, (text, language, phoneme_result) in enumerate(zip(texts, languages, phoneme_results)):
            phoneme_ids = phoneme_result["phoneme_ids"]
            if not phoneme_ids:
                continue
            phoneme_text = phoneme_result.get("text") or text
            batch_entries.append(
                (
                    index,
                    language,
                    phoneme_ids,
                    {
                        "phoneme_text": phoneme_text,
                        "phoneme_length": len(phoneme_ids),
                        "word_spans": phoneme_result.get("word_spans"),
                    },
                )
            )

        if not batch_entries:
            return budgets

        batch_size = len(batch_entries)
        max_len = max(len(phoneme_ids) for _, _, phoneme_ids, _ in batch_entries)
        x = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        x_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        sid_values = []
        batch_texts = []
        batch_phoneme_lengths = []
        batch_word_spans = []
        for batch_idx, (input_index, language, phoneme_ids, payload) in enumerate(batch_entries):
            x[batch_idx, : len(phoneme_ids)] = torch.tensor(
                phoneme_ids,
                dtype=torch.long,
                device=self.device,
            )
            x_lengths[batch_idx] = len(phoneme_ids)
            sid_values.append(self._speaker_id_for_language(language))
            batch_texts.append(payload["phoneme_text"])
            batch_phoneme_lengths.append(payload["phoneme_length"])
            batch_word_spans.append(payload["word_spans"])

        sid = torch.tensor(sid_values, dtype=torch.long, device=self.device)
        bert_input = self._bert_input(
            batch_texts,
            phoneme_length=batch_phoneme_lengths,
            word_spans=batch_word_spans,
        )
        frame_samples = []
        for _ in range(max(1, self.config.samples)):
            frame_samples.append(self._predict_frames(x, x_lengths, sid, bert_input))
        frame_tensor = torch.stack(frame_samples, dim=0).to(dtype=torch.float32)
        seconds_tensor = frame_tensor * (256.0 / 22050.0)
        token_tensor = seconds_tensor * self.config.token_rate
        quantile = self.config.upper_quantile

        for batch_idx, (input_index, language, phoneme_ids, _payload) in enumerate(batch_entries):
            profile_language, profile = self._budget_profile(language)
            min_margin = self._profile_float(profile, "min_margin", self.config.min_margin)
            max_margin = self._profile_float(profile, "max_margin", self.config.max_margin)
            min_extra_tokens = self._profile_int(profile, "min_extra_tokens", self.config.min_extra_tokens)
            max_extra_tokens = self._profile_int(profile, "max_extra_tokens", self.config.max_extra_tokens)

            sample_frames = frame_tensor[:, batch_idx]
            sample_seconds = seconds_tensor[:, batch_idx]
            mel_frames = int(torch.quantile(sample_frames, quantile).ceil().item())
            seconds = float(torch.quantile(sample_seconds, quantile).item())
            estimated_tokens = max(1, round(float(torch.quantile(token_tensor[:, batch_idx], quantile).item())))
            min_tokens = max(1, round(estimated_tokens * min_margin) + min_extra_tokens)
            max_tokens = max(min_tokens, round(estimated_tokens * max_margin) + max_extra_tokens)

            budgets[input_index] = {
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
                "speaker_id": int(self._speaker_id_for_language(language)),
                "sample_frames": [int(v) for v in sample_frames.tolist()],
                "sample_seconds": [round(float(s), 3) for s in sample_seconds.tolist()],
                "mean_seconds": float(torch.mean(sample_seconds).item()),
                "p50_seconds": float(torch.quantile(sample_seconds, 0.50).item()),
                "p90_seconds": float(torch.quantile(sample_seconds, 0.90).item()),
                "length_scale": self.config.length_scale,
                "noise_scale": self.config.noise_scale,
            }

        return budgets

    @torch.no_grad()
    def validate_alignment(
        self,
        text: str,
        wav_data: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        expected_seconds: float | None = None,
        duration_tolerance: float = 0.25,
        reject_zero_phoneme_duration: bool = True,
    ) -> dict[str, Any]:
        self.load()
        assert self._model is not None
        model = self._model
        if getattr(model, "enc_q", None) is None or getattr(model, "flow", None) is None:
            raise RuntimeError("Qwen DP alignment validation requires enc_q/flow; enable_alignment_validation is false")

        from src.piper.preprocess import phonemize_text_for_infer
        from src.piper.vits import monotonic_align
        from src.piper.vits.mel_processing import spectrogram_torch

        phoneme_result = phonemize_text_for_infer(
            text,
            self._phoneme_config(language),
            neural=False,
        )
        phoneme_ids = phoneme_result.get("phoneme_ids") or []
        phoneme_count = len(phoneme_ids)
        if not phoneme_ids:
            return {
                "enabled": True,
                "valid": True,
                "skipped": True,
                "reason": "empty_phonemes",
            }

        audio = np.asarray(wav_data, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {
                "enabled": True,
                "valid": False,
                "reason": "empty_audio",
                "phoneme_count": phoneme_count,
            }

        target_sample_rate = int(getattr(model, "sample_rate", 0) or self._model_config.get("audio", {}).get("sample_rate") or 22050)
        if sample_rate != target_sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate).astype(np.float32, copy=False)

        audio = np.clip(audio, -1.0, 1.0)
        audio_seconds = float(audio.size / target_sample_rate)

        def invalid_alignment_result(
            reason: str,
            *,
            aligned_frames: int | None = None,
            alignment_error: str | None = None,
        ) -> dict[str, Any]:
            duration_ratio = None
            if expected_seconds is not None and expected_seconds > 0:
                duration_ratio = audio_seconds / float(expected_seconds)
            result: dict[str, Any] = {
                "enabled": True,
                "valid": False,
                "reason": reason,
                "audio_seconds": audio_seconds,
                "expected_seconds": float(expected_seconds) if expected_seconds is not None else None,
                "duration_ratio": float(duration_ratio) if duration_ratio is not None else None,
                "duration_tolerance": float(duration_tolerance),
                "phoneme_count": phoneme_count,
                "aligned_frames": int(aligned_frames) if aligned_frames is not None else None,
                "zero_duration_count": None,
                "zero_duration_indices": [],
                "min_phoneme_frames": 0,
                "max_phoneme_frames": 0,
                "sample_rate": target_sample_rate,
            }
            if alignment_error:
                result["alignment_error"] = alignment_error
            return result

        if expected_seconds is not None and expected_seconds > 0:
            duration_ratio = audio_seconds / float(expected_seconds)
            if duration_ratio > 1.0 + duration_tolerance:
                _LOGGER.info(
                    "Qwen DP alignment validation rejected duration before alignment audio_seconds=%.2f expected_seconds=%.2f ratio=%.3f tolerance=%.3f phoneme_count=%d",
                    audio_seconds,
                    float(expected_seconds),
                    duration_ratio,
                    float(duration_tolerance),
                    phoneme_count,
                )
                return invalid_alignment_result("duration_out_of_range")

        y = torch.from_numpy(audio).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        if y.size(1) < int(getattr(model, "hop_length", 256)):
            return invalid_alignment_result("audio_too_short", aligned_frames=0)

        filter_length = int(getattr(model, "filter_length", 1024))
        hop_length = int(getattr(model, "hop_length", 256))
        win_length = int(getattr(model, "win_length", 1024))
        y_spec = spectrogram_torch(
            y,
            n_fft=filter_length,
            sampling_rate=target_sample_rate,
            hop_size=hop_length,
            win_size=win_length,
            center=False,
        )
        y_lengths = torch.tensor([y_spec.size(-1)], dtype=torch.long, device=self.device)
        aligned_frames = int(y_lengths.item())
        if phoneme_count > aligned_frames:
            return invalid_alignment_result(
                "audio_too_short_for_alignment",
                aligned_frames=aligned_frames,
            )

        x = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)
        x_lengths = torch.tensor([phoneme_count], dtype=torch.long, device=self.device)
        sid = torch.tensor([self._speaker_id_for_language(language)], dtype=torch.long, device=self.device)
        bert_input = self._bert_input(
            phoneme_result.get("text") or text,
            phoneme_length=phoneme_count,
            word_spans=phoneme_result.get("word_spans"),
        )
        speaker_id = int(sid.item())

        if model.n_speakers > 1:
            g = model.emb_g(sid).unsqueeze(-1)
        else:
            g = None

        try:
            _x_encoded, m_p, logs_p, x_mask = model.enc_p(x, x_lengths, bert_input=bert_input, g=g)
        except TypeError:
            _x_encoded, m_p, logs_p, x_mask = model.enc_p(x, x_lengths, g=g)
        z, _m_q, _logs_q, y_mask = model.enc_q(y_spec, y_lengths, g=g)
        z_p = model.flow(z, y_mask, g=g)

        s_p_sq_r = torch.exp(-2 * logs_p)
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
        neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
        neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
        neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        try:
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1)
        except ValueError as exc:
            return invalid_alignment_result(
                "alignment_error",
                aligned_frames=aligned_frames,
                alignment_error=str(exc),
            ) | {"speaker_id": speaker_id}
        durations = attn.sum(2).squeeze(0).squeeze(0)
        active_durations = durations[:phoneme_count].detach().cpu()
        zero_indices = torch.nonzero(active_durations <= 0, as_tuple=False).flatten().tolist()

        duration_ratio = None
        duration_valid = True
        if expected_seconds is not None and expected_seconds > 0:
            duration_ratio = audio_seconds / float(expected_seconds)
            duration_valid = (1.0 - duration_tolerance) <= duration_ratio <= (1.0 + duration_tolerance)

        zero_valid = (not reject_zero_phoneme_duration) or not zero_indices
        valid = bool(duration_valid and zero_valid)
        reason = "ok"
        if not zero_valid:
            reason = "zero_phoneme_duration"
        elif not duration_valid:
            reason = "duration_out_of_range"

        return {
            "enabled": True,
            "valid": valid,
            "reason": reason,
            "audio_seconds": audio_seconds,
            "expected_seconds": float(expected_seconds) if expected_seconds is not None else None,
            "duration_ratio": float(duration_ratio) if duration_ratio is not None else None,
            "duration_tolerance": float(duration_tolerance),
            "phoneme_count": phoneme_count,
            "aligned_frames": aligned_frames,
            "zero_duration_count": len(zero_indices),
            "zero_duration_indices": [int(index) for index in zero_indices[:20]],
            "min_phoneme_frames": int(active_durations.min().item()) if active_durations.numel() else 0,
            "max_phoneme_frames": int(active_durations.max().item()) if active_durations.numel() else 0,
            "speaker_id": speaker_id,
            "sample_rate": target_sample_rate,
        }

    def _bert_input(
        self,
        text: str | list[str],
        phoneme_length: int | list[int] | None = None,
        word_spans: list[list[int] | None] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if self._semantic_tokenizer is None or self._build_bert_input is None or not text:
            return None
        if isinstance(text, list):
            bert_dict = self._build_bert_input(
                text,
                self._semantic_tokenizer,
                phoneme_lengths=phoneme_length if isinstance(phoneme_length, list) else None,
                word_spans=word_spans,
            )
        elif phoneme_length is None:
            bert_dict = self._build_bert_input([text], self._semantic_tokenizer)
        else:
            bert_dict = self._build_bert_input(
                [text],
                self._semantic_tokenizer,
                phoneme_lengths=[phoneme_length],
                word_spans=[word_spans],
            )
        if bert_dict is None:
            return None
        if self._semantic_model is not None and self._align_phone_features is not None:
            input_ids = bert_dict["input_ids"].to(self.device)
            attention_mask = bert_dict["attention_mask"].to(self.device)
            word2ph = bert_dict.get("word2ph")
            if word2ph is None:
                return None
            if isinstance(phoneme_length, int):
                phone_lengths = [phoneme_length]
            elif isinstance(phoneme_length, list):
                phone_lengths = [int(length) for length in phoneme_length]
            else:
                return None

            with torch.inference_mode():
                hidden = self._semantic_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

            word2ph = word2ph.to(hidden.device)
            feature_items = [
                self._align_phone_features(hidden[idx], word2ph[idx], phone_len=phone_lengths[idx])
                for idx in range(len(phone_lengths))
            ]
            hidden_dim = int(feature_items[0].size(0))
            max_len = max(phone_lengths)
            features = torch.zeros(
                (len(feature_items), hidden_dim, max_len),
                device=self.device,
                dtype=hidden.dtype,
            )
            for idx, item in enumerate(feature_items):
                features[idx, :, : item.size(1)] = item.to(device=self.device)
            return {"features": features}

        out = {
            "input_ids": bert_dict["input_ids"].to(self.device),
            "attention_mask": bert_dict["attention_mask"].to(self.device),
        }
        if "word2ph" in bert_dict:
            out["word2ph"] = bert_dict["word2ph"].to(self.device)
        return out

    def _predict_frames(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        bert_input: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
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
            dp = model.sdp if model.use_duration_blend else model.dp
            logw = dp(x_encoded, x_mask, g=g, reverse=True, noise_scale=self.config.noise_scale)
        else:
            logw = model.dp(x_encoded, x_mask, g=g)
        w = torch.exp(logw) * x_mask * self.config.length_scale
        w_ceil = torch.ceil(w)
        return torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1)

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
