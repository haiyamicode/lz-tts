"""Sparrow duration prediction and monotonic-alignment validation."""

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

_ALIGNMENT_TRIM_FRAME_MS = 20
_ALIGNMENT_TRIM_HOP_MS = 10
_ALIGNMENT_TRIM_PADDING_MS = 50
_ALIGNMENT_TRIM_TOP_DB = 40.0
_ALIGNMENT_TRIM_MIN_RMS = 1e-4


def trim_boundary_silence(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    """Trim low-energy audio at the boundaries while preserving internal pauses."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio, 0, 0
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    frame_length = max(1, round(sample_rate * _ALIGNMENT_TRIM_FRAME_MS / 1000))
    hop_length = max(1, round(sample_rate * _ALIGNMENT_TRIM_HOP_MS / 1000))
    frame_starts = np.arange(0, audio.size, hop_length, dtype=np.int64)
    frame_ends = np.minimum(frame_starts + frame_length, audio.size)

    squared = np.square(audio, dtype=np.float64)
    cumulative_energy = np.concatenate(([0.0], np.cumsum(squared)))
    frame_energy = cumulative_energy[frame_ends] - cumulative_energy[frame_starts]
    frame_rms = np.sqrt(frame_energy / (frame_ends - frame_starts))
    peak_rms = float(frame_rms.max(initial=0.0))
    threshold = max(
        _ALIGNMENT_TRIM_MIN_RMS,
        peak_rms * 10.0 ** (-_ALIGNMENT_TRIM_TOP_DB / 20.0),
    )
    active_frames = np.flatnonzero(frame_rms >= threshold)
    if active_frames.size == 0:
        return audio[:0], 0, 0

    padding = round(sample_rate * _ALIGNMENT_TRIM_PADDING_MS / 1000)
    start = max(0, int(frame_starts[active_frames[0]]) - padding)
    end = min(audio.size, int(frame_ends[active_frames[-1]]) + padding)
    return np.ascontiguousarray(audio[start:end]), start, end


def alignment_word_timestamps(
    *,
    text: str,
    phonemes: list[str],
    word_spans: list[list[int]] | None,
    token_durations_frames: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    trim_start_samples: int = 0,
) -> list[dict[str, Any]]:
    """Convert a VITS MAS path into word timestamps.

    Piper's text tensor contains BOS, padding tokens between phonemes, and EOS,
    while ``word_spans`` refers to the pre-ID phoneme list. Building the token
    boundary table with ``phoneme_ids_espeak`` keeps the conversion correct if
    an unsupported phoneme is omitted from the ID sequence.
    """
    if not word_spans:
        return []
    if hop_length <= 0 or sample_rate <= 0:
        raise ValueError("hop_length and sample_rate must be positive")

    from piper_phonemize import phoneme_ids_espeak

    durations = token_durations_frames.detach().cpu().to(dtype=torch.int64).reshape(-1)
    frame_boundaries = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(durations, dim=0)]
    )
    # The ID sequence has BOS + one PAD after every retained phoneme + EOS.
    # Encode one phoneme at a time so this remains linear in utterance length;
    # an unsupported phoneme contributes no IDs and therefore no boundary move.
    token_boundaries = [2]
    for phoneme in phonemes:
        encoded_phoneme = phoneme_ids_espeak([phoneme])
        token_boundaries.append(token_boundaries[-1] + max(0, len(encoded_phoneme) - 3))
    frame_seconds = hop_length / sample_rate
    trim_start_seconds = trim_start_samples / sample_rate

    words: list[dict[str, Any]] = []
    for raw_span in word_spans:
        if len(raw_span) != 4:
            raise ValueError(f"Invalid word span: {raw_span!r}")
        text_start, text_end, phoneme_start, phoneme_end = map(int, raw_span)
        if not (0 <= phoneme_start <= phoneme_end <= len(phonemes)):
            raise ValueError(f"Word span exceeds phoneme sequence: {raw_span!r}")

        token_start = token_boundaries[phoneme_start]
        token_end = token_boundaries[phoneme_end]
        if token_end > durations.numel():
            raise ValueError(
                f"Word span maps past MAS token sequence: {raw_span!r} -> "
                f"[{token_start}, {token_end}) of {durations.numel()}"
            )
        start_frame = int(frame_boundaries[token_start].item())
        end_frame = int(frame_boundaries[token_end].item())
        words.append(
            {
                "text": text[text_start:text_end],
                "text_start": text_start,
                "text_end": text_end,
                "phoneme_start": phoneme_start,
                "phoneme_end": phoneme_end,
                "token_start": token_start,
                "token_end": token_end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": trim_start_seconds + start_frame * frame_seconds,
                "end_seconds": trim_start_seconds + end_frame * frame_seconds,
            }
        )
    return words


@dataclass(frozen=True)
class DpBudgetConfig:
    checkpoint: Path = Path("data/lzspeech-sparrow/model.ckpt")
    config_path: Optional[Path] = None
    device: str = "cpu"
    language: str = "multilingual"
    length_scale: float = 1.0
    token_rate: float = 12.0
    min_margin: float = 1.0
    max_margin: float = 1.25
    min_extra_tokens: int = 0
    max_extra_tokens: int = 36
    soft_text_token_limit: int = 250
    hard_text_token_limit: int = 300
    include_word_spans: bool = True
    language_profiles: dict[str, dict[str, float | int]] = field(default_factory=dict)
    use_bert: bool = False
    enable_alignment_validation: bool = False

    def __post_init__(self) -> None:
        if self.soft_text_token_limit < 1:
            raise ValueError("soft_text_token_limit must be positive")
        if self.hard_text_token_limit < self.soft_text_token_limit:
            raise ValueError(
                "hard_text_token_limit must be greater than or equal to "
                "soft_text_token_limit"
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
        raise ValueError("duration alignment language profiles must be a JSON object")

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


class DurationAlignmentValidator:
    """Predict speech duration and validate monotonic phoneme alignment."""

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
            _LOGGER.info("Loading duration alignment checkpoint=%s config=%s", checkpoint_path, config_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Duration alignment checkpoint not found: {checkpoint_path}")
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Duration alignment config not found: {config_path} "
                    f"(checkpoint={checkpoint_path})"
                )
            with config_path.open("r", encoding="utf-8") as f:
                self._model_config = json.load(f)

            # Transformers loading can leave PyTorch's default device as
            # ``meta`` in this worker. Lightning then constructs VITS on meta and
            # checkpoint copies become no-ops, so force real CPU allocation here.
            with torch.device("cpu"):
                model = VitsModel.load_from_checkpoint(
                    str(checkpoint_path),
                    dataset=None,
                    weights_only=False,
                    map_location="cpu",
                )
            model.eval()

            model_g = model.model_g
            model_g.dec = None
            if not self.config.enable_alignment_validation:
                model_g.enc_q = None
                model_g.flow = None
            if model_g.use_duration_blend:
                model_g.sdp = None
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
                    with torch.device("cpu"):
                        semantic_model = AutoModel.from_pretrained(semantic_path)
                    self._semantic_model = semantic_model.to(self.device).eval()
            elif checkpoint_uses_bert:
                self._strip_bert_from_text_encoder(model_g)

            meta_tensors = self._meta_tensor_names(model_g)
            if meta_tensors:
                preview = ", ".join(meta_tensors[:10])
                suffix = (
                    f", ... (+{len(meta_tensors) - 10} more)"
                    if len(meta_tensors) > 10
                    else ""
                )
                raise RuntimeError(
                    "Duration alignment model still has meta tensors after CPU checkpoint load: "
                    f"{preview}{suffix}"
                )
            self._model = model_g.to(self.device).eval()
            gc.collect()

    @staticmethod
    def _meta_tensor_names(model: Any) -> list[str]:
        names = [
            f"parameter:{name}"
            for name, param in model.named_parameters()
            if getattr(param, "is_meta", False)
        ]
        names.extend(
            f"buffer:{name}"
            for name, buffer in model.named_buffers()
            if getattr(buffer, "is_meta", False)
        )
        return names

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

        budgets = [self._empty_budget(language) for language in languages]
        batch_entries: list[tuple[int, str | None, list[int], dict[str, Any]]] = []
        for index, (text, language) in enumerate(zip(texts, languages)):
            chunk_results = self._phonemize_budget_chunks(text, language)
            for chunk_index, phoneme_result in enumerate(chunk_results):
                phoneme_ids = phoneme_result["phoneme_ids"]
                if not phoneme_ids:
                    continue
                batch_entries.append(
                    (
                        index,
                        language,
                        phoneme_ids,
                        {
                            "phoneme_text": phoneme_result.get("text") or text,
                            "phoneme_length": len(phoneme_ids),
                            "word_spans": phoneme_result.get("word_spans"),
                            "keep_bos": chunk_index == 0,
                            "keep_eos": chunk_index == len(chunk_results) - 1,
                        },
                    )
                )

        if not batch_entries:
            return budgets

        frame_tensor = torch.zeros(len(texts), dtype=torch.float32, device=self.device)
        phoneme_counts = [0] * len(texts)
        chunk_counts = [0] * len(texts)
        for input_index, language, phoneme_ids, payload in batch_entries:
            x = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)
            x_lengths = torch.tensor(
                [len(phoneme_ids)], dtype=torch.long, device=self.device
            )
            sid = torch.tensor(
                [self._speaker_id_for_language(language)],
                dtype=torch.long,
                device=self.device,
            )
            bert_input = self._bert_input(
                [payload["phoneme_text"]],
                phoneme_length=[payload["phoneme_length"]],
                word_spans=[payload["word_spans"]],
            )
            token_frames = self._predict_token_frames(x, x_lengths, sid, bert_input)
            token_start = 0 if payload["keep_bos"] else 2
            token_end = (
                len(phoneme_ids)
                if payload["keep_eos"]
                else len(phoneme_ids) - 1
            )
            frame_tensor[input_index] += token_frames[
                0, 0, token_start:token_end
            ].sum(dtype=torch.float32)
            phoneme_counts[input_index] += token_end - token_start
            chunk_counts[input_index] += 1

        seconds_tensor = frame_tensor * (256.0 / 22050.0)
        token_tensor = seconds_tensor * self.config.token_rate

        for input_index, language in enumerate(languages):
            if chunk_counts[input_index] == 0:
                continue
            profile_language, profile = self._budget_profile(language)
            min_margin = self._profile_float(profile, "min_margin", self.config.min_margin)
            max_margin = self._profile_float(profile, "max_margin", self.config.max_margin)
            min_extra_tokens = self._profile_int(profile, "min_extra_tokens", self.config.min_extra_tokens)
            max_extra_tokens = self._profile_int(profile, "max_extra_tokens", self.config.max_extra_tokens)

            mel_frames = int(frame_tensor[input_index].ceil().item())
            seconds = float(seconds_tensor[input_index].item())
            estimated_tokens = max(1, round(float(token_tensor[input_index].item())))
            min_tokens = max(1, round(estimated_tokens * min_margin) + min_extra_tokens)
            max_tokens = max(min_tokens, round(estimated_tokens * max_margin) + max_extra_tokens)

            budgets[input_index] = {
                "mel_frames": mel_frames,
                "seconds": seconds,
                "estimated_tokens": estimated_tokens,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "token_rate": self.config.token_rate,
                "budget_language": profile_language,
                "budget_profile": profile,
                "min_margin": min_margin,
                "max_margin": max_margin,
                "min_extra_tokens": min_extra_tokens,
                "max_extra_tokens": max_extra_tokens,
                "phoneme_count": phoneme_counts[input_index],
                "duration_chunks": chunk_counts[input_index],
                "phoneme_language": self._phoneme_config(language)["language"]["code"],
                "speaker_id": int(self._speaker_id_for_language(language)),
                "length_scale": self.config.length_scale,
            }

        return budgets

    def _phonemize_budget_chunks(
        self,
        text: str,
        language: str | None,
    ) -> list[dict[str, Any]]:
        """Split source text naturally using Lazybird's token limits."""
        from src.piper.preprocess import phonemize_text_for_infer
        from src.text_splitter import count_cl100k_tokens, split_text

        phoneme_config = self._phoneme_config(language)
        chunks = split_text(
            text,
            self.config.hard_text_token_limit,
            soft_max_length=self.config.soft_text_token_limit,
            length_function=count_cl100k_tokens,
            measure_merged_length=True,
        )
        results: list[dict[str, Any]] = []
        for chunk in chunks:
            result = phonemize_text_for_infer(
                chunk,
                phoneme_config,
                neural=False,
                include_word_spans=self.config.include_word_spans,
            )
            if len(result["phoneme_ids"]) <= 3:
                continue
            results.append(result)
        return results

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
        include_word_timestamps: bool = False,
    ) -> dict[str, Any]:
        self.load()
        assert self._model is not None
        model = self._model
        if getattr(model, "enc_q", None) is None or getattr(model, "flow", None) is None:
            raise RuntimeError("Alignment validation requires enc_q/flow; enable_alignment_validation is false")

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
        raw_audio_samples = int(audio.size)
        audio, trim_start, trim_end = trim_boundary_silence(audio, target_sample_rate)
        raw_audio_seconds = float(raw_audio_samples / target_sample_rate)
        trim_head_seconds = float(trim_start / target_sample_rate)
        trim_tail_seconds = float((raw_audio_samples - trim_end) / target_sample_rate)
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
                "raw_audio_seconds": raw_audio_seconds,
                "trim_head_seconds": trim_head_seconds,
                "trim_tail_seconds": trim_tail_seconds,
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

        if audio.size == 0:
            return invalid_alignment_result("silent_audio", aligned_frames=0)

        if expected_seconds is not None and expected_seconds > 0:
            duration_ratio = audio_seconds / float(expected_seconds)
            if duration_ratio > 1.0 + duration_tolerance:
                _LOGGER.info(
                    "Alignment validation rejected duration before alignment audio_seconds=%.2f raw_audio_seconds=%.2f trim_head_seconds=%.2f trim_tail_seconds=%.2f expected_seconds=%.2f ratio=%.3f tolerance=%.3f phoneme_count=%d",
                    audio_seconds,
                    raw_audio_seconds,
                    trim_head_seconds,
                    trim_tail_seconds,
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

        result = {
            "enabled": True,
            "valid": valid,
            "reason": reason,
            "audio_seconds": audio_seconds,
            "raw_audio_seconds": raw_audio_seconds,
            "trim_head_seconds": trim_head_seconds,
            "trim_tail_seconds": trim_tail_seconds,
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
        if include_word_timestamps:
            result["word_timestamps"] = alignment_word_timestamps(
                text=str(phoneme_result.get("text") or text),
                phonemes=list(phoneme_result.get("phonemes") or []),
                word_spans=phoneme_result.get("word_spans"),
                token_durations_frames=active_durations,
                hop_length=hop_length,
                sample_rate=target_sample_rate,
                trim_start_samples=trim_start,
            )
            result["frame_seconds"] = hop_length / target_sample_rate
        return result

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
        token_frames = self._predict_token_frames(x, x_lengths, sid, bert_input)
        return torch.clamp_min(torch.sum(token_frames, [1, 2]), 1)

    def _predict_token_frames(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        bert_input: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        assert self._model is not None
        model = self._model
        with torch.device(self.device):
            if model.n_speakers > 1:
                g = model.emb_g(sid).unsqueeze(-1)
            else:
                g = None

            if bert_input is not None:
                x_encoded, _m_p, _logs_p, x_mask = model.enc_p(
                    x,
                    x_lengths,
                    bert_input=bert_input,
                    g=g,
                )
            else:
                x_encoded, _m_p, _logs_p, x_mask = model.enc_p(
                    x,
                    x_lengths,
                    g=g,
                )

            if model.use_sdp and not model.use_duration_blend:
                raise RuntimeError(
                    "Duration budgeting requires a checkpoint with a deterministic "
                    "duration predictor"
                )
            logw = model.dp(x_encoded, x_mask, g=g)
            w = torch.exp(logw) * x_mask * self.config.length_scale
            token_frames = torch.ceil(w)

            # VITS stores every attention matrix on its module for training-time
            # inspection. The duration-budget runtime never consumes those tensors.
            for module in model.enc_p.modules():
                if hasattr(module, "attn"):
                    module.attn = None
            return token_frames

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
            "budget_language": profile_language,
            "budget_profile": profile,
            "min_margin": min_margin,
            "max_margin": max_margin,
            "min_extra_tokens": min_extra_tokens,
            "max_extra_tokens": max_extra_tokens,
            "phoneme_count": 0,
            "phoneme_language": self.config.language,
            "speaker_id": self._speaker_id_for_language(language),
            "length_scale": self.config.length_scale,
        }
