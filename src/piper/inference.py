"""Piper TTS inference pipeline.

Provides a minimal interface for running TTS inference using a Piper/VITS model checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import autocast

_LOGGER = logging.getLogger(__name__)

from .vits.lightning import VitsModel
from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    noise_scale: float = 0.667
    length_scale: float = 1.0
    noise_w: float = 0.8
    sdp_ratio: float = 0.2
    sample_rate: int = 22050


@dataclass
class _CachedVoiceAdapter:
    path: Path
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]


class PiperInference:
    """Piper TTS inference wrapper.

    Loads a VITS model checkpoint and provides methods for text-to-speech synthesis.
    Supports multi-speaker models with optional BERT semantic conditioning.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: Optional[str] = None,
    ):
        """Initialize the inference pipeline.

        Args:
            checkpoint_path: Path to the .ckpt model checkpoint.
            config_path: Path to the config.json from preprocessing.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self._voice_adapter_specs: dict[str, Path] = {}
        self._voice_adapter_cache: OrderedDict[str, _CachedVoiceAdapter] = OrderedDict()
        self._voice_adapter_cache_size = 1
        self._voice_adapter_runtime_enabled = False
        self._resident_voice_adapter: str | None = None
        self._active_voice_adapter: str | None = None
        self._checkpoint_sha256: str | None = None
        self._voice_adapter_lock = threading.RLock()

        # Load config
        with open(self.config_path) as f:
            self.config = json.load(f)

        # Extract inference parameters
        inference_cfg = self.config.get("inference", {})
        self.inference_config = InferenceConfig(
            noise_scale=inference_cfg.get("noise_scale", 0.667),
            length_scale=inference_cfg.get("length_scale", 1.0),
            noise_w=inference_cfg.get("noise_w", 0.8),
            sdp_ratio=inference_cfg.get("sdp_ratio", 0.2),
            sample_rate=self.config.get("audio", {}).get("sample_rate", 22050),
        )

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Precision mode: enable fp16 on CUDA unless disabled via env
        self.fp16 = self.device.type == "cuda" and bool(
            int(os.environ.get("PIPER_USE_FP16", "1"))
        )

        # Load model
        _LOGGER.info("Loading model: %s", self.checkpoint_path.name)
        self.model = VitsModel.load_from_checkpoint(
            str(self.checkpoint_path), dataset=None, weights_only=False
        )
        self._sync_config_from_checkpoint()
        self._sync_inference_defaults_from_checkpoint(inference_cfg)
        self.model.eval()
        self.model.to(self.device)
        if self.fp16:
            self.model.half()

        # Remove weight norm for inference (suppress warnings)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Removing weight norm.*")
            self.model.model_g.dec.remove_weight_norm()

        # Setup BERT semantic tokenizer if model was trained with it
        self.use_bert = bool(getattr(self.model.hparams, "use_bert", False))
        self.bert_features_precomputed = bool(
            getattr(self.model.hparams, "bert_features_precomputed", False)
        )
        self.bert_model_name = getattr(self.model.hparams, "bert_model_name", None)
        self.semantic_tokenizer = None
        self.semantic_model = None
        self._build_bert_input = None

        if self.use_bert:
            from .semantic import (
                SemanticTokenizer,
                build_bert_input,
            )

            if self.bert_features_precomputed:
                self._load_semantic_model()

            _LOGGER.info("Loading BERT tokenizer: %s", self.bert_model_name or "default")
            self.semantic_tokenizer = SemanticTokenizer(model_name=self.bert_model_name)
            self._build_bert_input = build_bert_input
        self.semantic_fusion_mode = getattr(
            getattr(self.model.model_g, "enc_p", None),
            "semantic_fusion_mode",
            None,
        )

        _LOGGER.info(
            "Model ready: %s (device=%s, speakers=%d, bert=%s)",
            self.checkpoint_path.name,
            self.device,
            len(self.speakers),
            self.use_bert,
        )

    def configure_voice_adapters(
        self,
        adapters: Mapping[str, str | Path],
        *,
        cache_size: int = 1,
    ) -> None:
        """Configure lazily loaded adapters for this persistent base model."""

        if cache_size < 1:
            raise ValueError("Voice adapter cache size must be at least 1")
        if self.model.model_g.voice_adapter_embedding is not None:
            raise RuntimeError(
                "Managed voice adapters require an unmodified base checkpoint"
            )

        specs: dict[str, Path] = {}
        for raw_name, raw_path in adapters.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Voice adapter names must be non-empty")
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata_path = path.with_suffix(".json")
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            specs[name] = path

        self._voice_adapter_specs = specs
        self._voice_adapter_cache_size = int(cache_size)
        self._voice_adapter_runtime_enabled = True
        _LOGGER.info(
            "Configured Sparrow voice adapters names=%s cache_size=%d",
            sorted(specs),
            self._voice_adapter_cache_size,
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_cached_voice_adapter(self, name: str) -> _CachedVoiceAdapter:
        cached = self._voice_adapter_cache.pop(name, None)
        if cached is not None:
            self._voice_adapter_cache[name] = cached
            return cached

        path = self._voice_adapter_specs.get(name)
        if path is None:
            raise ValueError(f"Unknown Sparrow voice adapter: {name}")
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("format") != "sparrow_voice_adapter":
            raise ValueError(f"Unsupported Sparrow voice adapter format: {path}")
        if int(metadata.get("format_version", 0)) != 1:
            raise ValueError(
                f"Unsupported Sparrow voice adapter version: {metadata.get('format_version')}"
            )

        base_hash = str(metadata.get("base_checkpoint_sha256") or "")
        if not base_hash:
            raise ValueError(f"Voice adapter metadata has no base checkpoint hash: {path}")
        if self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = self._sha256_file(self.checkpoint_path)
        if base_hash != self._checkpoint_sha256:
            raise ValueError(
                "Voice adapter base checkpoint mismatch: "
                f"adapter={base_hash} loaded={self._checkpoint_sha256}"
            )

        speaker = str(metadata.get("speaker") or "")
        speaker_id = int(metadata.get("speaker_id", -1))
        if self.speakers.get(speaker) != speaker_id:
            raise ValueError(
                "Voice adapter speaker does not match the loaded model: "
                f"speaker={speaker!r} id={speaker_id}"
            )

        from safetensors.torch import load_file

        started = time.perf_counter()
        cached = _CachedVoiceAdapter(
            path=path,
            metadata=metadata,
            tensors=load_file(str(path), device="cpu"),
        )
        self._voice_adapter_cache[name] = cached
        while len(self._voice_adapter_cache) > self._voice_adapter_cache_size:
            evicted_name, _ = self._voice_adapter_cache.popitem(last=False)
            _LOGGER.info("Evicted Sparrow voice adapter from LRU cache: %s", evicted_name)
        _LOGGER.info(
            "Loaded Sparrow voice adapter name=%s path=%s elapsed=%.3fs cached=%s",
            name,
            path,
            time.perf_counter() - started,
            list(self._voice_adapter_cache),
        )
        return cached

    def _select_voice_adapter(self, name: str | None) -> None:
        if not self._voice_adapter_runtime_enabled:
            if name is not None:
                raise RuntimeError("No managed Sparrow voice adapters are configured")
            return

        model_g = self.model.model_g
        if name is None:
            model_g.set_voice_adapter_enabled(False)
            self._active_voice_adapter = None
            return

        cached = self._load_cached_voice_adapter(name)
        if self._resident_voice_adapter == name:
            model_g.set_voice_adapter_enabled(True)
            self._active_voice_adapter = name
            return

        from .vits.voice_adapter import load_adapter_state_dict

        model_g.remove_voice_adapter()
        self._resident_voice_adapter = None
        self._active_voice_adapter = None
        metadata = cached.metadata
        model_g.configure_voice_adapter(
            speaker_id=int(metadata["speaker_id"]),
            target_modules=tuple(metadata["target_modules"]),
            rank=int(metadata["rank"]),
            alpha=float(metadata["alpha"]),
            dropout=float(metadata["dropout"]),
        )
        load_adapter_state_dict(model_g, cached.tensors)
        model_g.eval()
        model_g.set_voice_adapter_enabled(True)
        self._resident_voice_adapter = name
        self._active_voice_adapter = name
        _LOGGER.info(
            "Activated Sparrow voice adapter name=%s speaker=%s targets=%d",
            name,
            metadata["speaker"],
            len(metadata["target_modules"]),
        )

    @property
    def voice_adapter_status(self) -> dict[str, Any]:
        return {
            "configured": sorted(self._voice_adapter_specs),
            "cached": list(self._voice_adapter_cache),
            "resident": self._resident_voice_adapter,
            "active": self._active_voice_adapter,
            "cache_size": self._voice_adapter_cache_size,
        }

    def _sync_config_from_checkpoint(self) -> None:
        speaker_map = getattr(self.model.hparams, "speaker_id_map", None)
        if not isinstance(speaker_map, dict) or not speaker_map:
            return

        checkpoint_speakers = {str(label): int(sid) for label, sid in speaker_map.items()}
        configured_speakers = {
            str(label): int(sid)
            for label, sid in (self.config.get("speaker_id_map") or {}).items()
        }
        if configured_speakers != checkpoint_speakers:
            _LOGGER.warning(
                "Config speaker_id_map does not match checkpoint; using checkpoint speaker map "
                "(config=%d speakers, checkpoint=%d speakers)",
                len(configured_speakers),
                len(checkpoint_speakers),
            )

        self.config["speaker_id_map"] = checkpoint_speakers
        num_speakers = getattr(self.model.hparams, "num_speakers", None)
        if isinstance(num_speakers, int):
            self.config["num_speakers"] = num_speakers

    def _sync_inference_defaults_from_checkpoint(self, inference_cfg: dict) -> None:
        if not bool(getattr(self.model.hparams, "use_duration_blend", False)):
            return

        # Older dataset configs carry the classic Piper defaults. Duration-blend
        # checkpoints should default to Melo/OpenVoice inference behavior unless
        # the config was explicitly updated with an sdp_ratio field.
        if "sdp_ratio" not in inference_cfg:
            self.inference_config.noise_scale = 0.6
            self.inference_config.noise_w = 0.8
            self.inference_config.sdp_ratio = float(
                getattr(self.model.hparams, "duration_blend_sdp_ratio", 0.2)
            )

    def _load_semantic_model(self):
        if self.semantic_model is not None:
            return self.semantic_model

        from .hf_cache import get_shared_hf_encoder

        local_files_only = any(
            os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
            for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
        )
        model = get_shared_hf_encoder(
            self.bert_model_name,
            device=self.device,
            dtype=torch.float16 if self.fp16 else torch.float32,
            local_files_only=local_files_only,
        )
        self.semantic_model = model
        return model

    def warmup_semantic(self) -> None:
        """Load semantic runtime components needed by this model."""
        if not self.use_bert:
            return
        if self.bert_features_precomputed:
            self._load_semantic_model()
            return

        encoder = getattr(self.model.model_g, "enc_p", None)
        if getattr(encoder, "bert", None) is not None:
            _LOGGER.info("BERT feature model already loaded inside VITS text encoder")

    def _semantic_input_for_span(
        self,
        span_text: str,
        phoneme_ids: list[int],
        word_spans,
    ) -> dict[str, torch.Tensor] | None:
        if not span_text:
            return None
        return self._semantic_input_for_span_batch(
            [{"text": span_text, "word_spans": word_spans, "phoneme_ids": phoneme_ids}],
            [len(phoneme_ids)],
        )

    def _inference_scales(
        self,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sdp_ratio: Optional[float] = None,
    ) -> list[float]:
        return [
            noise_scale if noise_scale is not None else self.inference_config.noise_scale,
            length_scale if length_scale is not None else self.inference_config.length_scale,
            noise_w if noise_w is not None else self.inference_config.noise_w,
            sdp_ratio if sdp_ratio is not None else self.inference_config.sdp_ratio,
        ]

    def _semantic_input_for_span_batch(
        self,
        spans: list[dict],
        phoneme_lengths: list[int],
    ) -> dict[str, torch.Tensor] | None:
        if not self.use_bert or self.semantic_tokenizer is None:
            return None

        texts = [str(span.get("text", "")) for span in spans]
        if not all(texts):
            raise ValueError("Batched BERT inference requires non-empty span text")

        if self.semantic_fusion_mode == "legacy_cross_attention":
            bert_dict = self._build_bert_input(texts, self.semantic_tokenizer)
            if bert_dict is None:
                return None
            return {key: value.to(self.device) for key, value in bert_dict.items()}

        word_spans = [span.get("word_spans") for span in spans]
        bert_dict = self._build_bert_input(
            texts,
            self.semantic_tokenizer,
            phoneme_lengths=phoneme_lengths,
            word_spans=word_spans,
        )
        if bert_dict is None:
            return None

        if not self.bert_features_precomputed:
            return {key: value.to(self.device) for key, value in bert_dict.items()}

        from .semantic import align_phone_features

        semantic_model = self._load_semantic_model()
        input_ids = bert_dict["input_ids"].to(self.device)
        attention_mask = bert_dict["attention_mask"].to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    hidden = semantic_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).last_hidden_state
            else:
                hidden = semantic_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

        dtype = torch.float16 if self.fp16 else torch.float32
        feature_items = [
            align_phone_features(
                hidden[idx],
                bert_dict["word2ph"][idx].to(device=hidden.device),
                phone_len=phoneme_lengths[idx],
            )
            for idx in range(len(spans))
        ]
        hidden_dim = feature_items[0].size(0)
        max_len = max(phoneme_lengths)
        features = torch.zeros(
            (len(spans), hidden_dim, max_len),
            device=self.device,
            dtype=dtype,
        )
        for idx, item in enumerate(feature_items):
            features[idx, :, : item.size(1)] = item.to(device=self.device, dtype=dtype)
        return {"features": features}

    def _infer_prepared_span_batch(
        self,
        spans: list[dict],
        scales: list[float],
    ) -> list[np.ndarray]:
        outputs, _token_durations = self._infer_prepared_span_batch_with_durations(
            spans,
            scales,
        )
        return outputs

    def _infer_prepared_span_batch_with_durations(
        self,
        spans: list[dict],
        scales: list[float],
    ) -> tuple[list[np.ndarray], list[torch.Tensor]]:
        if not spans:
            return [], []

        phoneme_ids = [span["phoneme_ids"] for span in spans]
        phoneme_lengths = [len(ids) for ids in phoneme_ids]
        max_len = max(phoneme_lengths)
        text_tensor = torch.zeros((len(spans), max_len), dtype=torch.long, device=self.device)
        text_lengths = torch.tensor(phoneme_lengths, dtype=torch.long, device=self.device)

        for idx, ids in enumerate(phoneme_ids):
            text_tensor[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)

        sid = None
        if any(span.get("speaker_id") is not None for span in spans):
            speaker_ids = [int(span.get("speaker_id", 0)) for span in spans]
            sid = torch.tensor(speaker_ids, dtype=torch.long, device=self.device)

        bert_input = self._semantic_input_for_span_batch(spans, phoneme_lengths)

        with torch.inference_mode(), autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            enabled=self.fp16,
        ):
            audio, attn, y_mask, _ = self.model.model_g.infer(
                text_tensor,
                text_lengths,
                sid=sid,
                noise_scale=scales[0],
                length_scale=scales[1],
                noise_scale_w=scales[2],
                sdp_ratio=scales[3],
                bert_input=bert_input,
            )

        hop_length = int(np.prod(getattr(self.model.model_g, "upsample_rates", (256,))))
        frame_lengths = y_mask.squeeze(1).sum(dim=1).long().detach().cpu().numpy()
        sample_lengths = frame_lengths * hop_length
        audio_i16 = audio_float_to_int16(audio.detach().float().cpu().numpy())

        outputs: list[np.ndarray] = []
        token_durations: list[torch.Tensor] = []
        for idx, sample_len in enumerate(sample_lengths):
            outputs.append(audio_i16[idx].reshape(-1)[: int(sample_len)])
            frame_len = int(frame_lengths[idx])
            token_durations.append(
                attn[idx, 0, :frame_len, : phoneme_lengths[idx]]
                .sum(dim=0)
                .detach()
                .cpu()
            )
        return outputs, token_durations

    def phonemize(
        self,
        text: str,
        speaker: Optional[str] = None,
        espeak_data_path: Optional[str] = None,
        neural: bool = False,
    ) -> list[dict]:
        """Convert text to phoneme spans with speaker IDs.

        Args:
            text: Input text to phonemize.
            speaker: Optional speaker label to force (skips language detection).
            espeak_data_path: Optional path to espeak-ng data directory.
            neural: If True, use neural heteronym disambiguation.

        Returns:
            List of dicts with 'phoneme_ids', 'speaker_id', and 'text' keys.
        """
        from .preprocess import (
            phonemize_spans_with_speakers,
            phonemize_text_for_speaker,
        )

        if speaker:
            span = phonemize_text_for_speaker(
                text, self.config, speaker, espeak_data_path, neural=neural
            )
            return [span]
        else:
            return phonemize_spans_with_speakers(
                text, self.config, espeak_data_path, neural=neural
            )

    def synthesize_span(
        self,
        text: str,
        speaker: Optional[str] = None,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sdp_ratio: Optional[float] = None,
        neural: bool = True,
        voice_adapter: str | None = None,
    ) -> np.ndarray:
        """Synthesize a single text span with a specific speaker.

        This is a lower-level method for multi-model synthesis where the caller
        handles language detection and model routing.

        Args:
            text: Input text to synthesize.
            speaker: Speaker label (None for single-speaker models, uses default).
            noise_scale: Override for prosody randomness (default from config).
            length_scale: Override for speech rate (default from config).
            noise_w: Override for duration predictor noise (default from config).
            neural: Use neural heteronym disambiguation (default True).

        Returns:
            Audio waveform as int16 numpy array.
        """
        return self.synthesize_batch(
            [text],
            speaker=speaker,
            batch_size=1,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_w=noise_w,
            sdp_ratio=sdp_ratio,
            neural=neural,
            voice_adapter=voice_adapter,
        )[0]

    def synthesize_with_ipa_overrides(
        self,
        text: str,
        overrides: Sequence[tuple[int, int, str]],
        speaker: Optional[str] = None,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sdp_ratio: Optional[float] = None,
        neural: bool = True,
        return_alignment: bool = False,
        voice_adapter: str | None = None,
    ) -> np.ndarray | tuple[np.ndarray, list[dict]]:
        """Synthesize full-context speech with exact IPA source-span overrides."""
        with self._voice_adapter_lock:
            self._select_voice_adapter(voice_adapter)
            return self._synthesize_with_ipa_overrides_unlocked(
                text,
                overrides,
                speaker=speaker,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_w=noise_w,
                sdp_ratio=sdp_ratio,
                neural=neural,
                return_alignment=return_alignment,
            )

    def _synthesize_with_ipa_overrides_unlocked(
        self,
        text: str,
        overrides: Sequence[tuple[int, int, str]],
        speaker: Optional[str] = None,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sdp_ratio: Optional[float] = None,
        neural: bool = True,
        return_alignment: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[dict]]:
        if not overrides:
            audio = self._synthesize_batch_unlocked(
                [text],
                speaker=speaker,
                batch_size=1,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_w=noise_w,
                sdp_ratio=sdp_ratio,
                neural=neural,
            )[0]
            return (audio, []) if return_alignment else audio

        from .preprocess import apply_ipa_overrides

        spans = self.phonemize(text, speaker=speaker, neural=neural)
        if speaker is not None:
            spans[0]["source_start"] = 0
            spans[0]["source_end"] = len(text)

        def phonemize_fragment(fragment: str, voice: str) -> list[str]:
            if not fragment:
                return []
            fragment_spans = self.phonemize(
                fragment,
                speaker=voice,
                neural=neural,
            )
            return [
                phoneme
                for fragment_span in fragment_spans
                for phoneme in fragment_span.get("phonemes", [])
            ]

        prepared = apply_ipa_overrides(
            spans,
            list(overrides),
            partial_word_phonemizer=phonemize_fragment,
        )
        scales = self._inference_scales(
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_w=noise_w,
            sdp_ratio=sdp_ratio,
        )
        if return_alignment:
            from src.duration_alignment import alignment_word_timestamps

            outputs, token_durations = self._infer_prepared_span_batch_with_durations(
                prepared,
                scales,
            )
            hop_length = int(
                np.prod(getattr(self.model.model_g, "upsample_rates", (256,)))
            )
            timestamps: list[dict] = []
            sample_offset = 0
            for prepared_span, span_audio, durations in zip(
                prepared,
                outputs,
                token_durations,
                strict=True,
            ):
                source_start = int(prepared_span.get("source_start", 0))
                offset_seconds = sample_offset / self.sample_rate
                for timestamp in alignment_word_timestamps(
                    text=str(prepared_span.get("text", "")),
                    phonemes=list(prepared_span.get("phonemes") or []),
                    word_spans=prepared_span.get("word_spans"),
                    token_durations_frames=durations,
                    hop_length=hop_length,
                    sample_rate=self.sample_rate,
                ):
                    timestamps.append(
                        {
                            **timestamp,
                            "source_start": source_start + int(timestamp["text_start"]),
                            "source_end": source_start + int(timestamp["text_end"]),
                            "start_seconds": offset_seconds + float(timestamp["start_seconds"]),
                            "end_seconds": offset_seconds + float(timestamp["end_seconds"]),
                        }
                    )
                sample_offset += int(span_audio.size)
        else:
            outputs = self._infer_prepared_span_batch(prepared, scales)
        if not outputs:
            audio = np.zeros(0, dtype=np.int16)
        else:
            audio = outputs[0] if len(outputs) == 1 else np.concatenate(outputs)
        return (audio, timestamps) if return_alignment else audio

    def synthesize_batch(
        self,
        texts: Sequence[str],
        speaker: Optional[str] | Sequence[Optional[str]] = None,
        batch_size: Optional[int] = None,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sdp_ratio: Optional[float] = None,
        neural: bool = False,
        voice_adapter: str | None = None,
    ) -> list[np.ndarray]:
        """Synthesize multiple texts with real batched model inference.

        For one forced speaker, neural heteronym resolution, semantic BERT, and
        VITS inference are batched. Mixed-speaker or auto-routed calls keep the
        existing per-text routing behavior before the model batch is formed.
        """
        with self._voice_adapter_lock:
            self._select_voice_adapter(voice_adapter)
            return self._synthesize_batch_unlocked(
                texts,
                speaker=speaker,
                batch_size=batch_size,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_w=noise_w,
                sdp_ratio=sdp_ratio,
                neural=neural,
            )

    def _synthesize_batch_unlocked(
        self,
        texts: Sequence[str],
        speaker: Optional[str] | Sequence[Optional[str]] = None,
        batch_size: Optional[int] = None,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sdp_ratio: Optional[float] = None,
        neural: bool = False,
    ) -> list[np.ndarray]:
        text_items = list(texts)
        if not text_items:
            return []

        if isinstance(speaker, str) or speaker is None:
            speaker_items: list[Optional[str]] = [speaker for _ in text_items]
        else:
            speaker_items = list(speaker)
            if len(speaker_items) != len(text_items):
                raise ValueError("speaker sequence length must match texts length")

        flat_spans: list[tuple[int, dict]] = []
        unique_speakers = set(speaker_items)
        only_speaker = next(iter(unique_speakers)) if len(unique_speakers) == 1 else None
        if only_speaker is not None:
            from .preprocess import phonemize_texts_for_speaker

            batched_spans = phonemize_texts_for_speaker(
                text_items,
                self.config,
                only_speaker,
                None,
                neural=neural,
            )
            flat_spans.extend((text_idx, span) for text_idx, span in enumerate(batched_spans))
        else:
            for text_idx, (text, item_speaker) in enumerate(zip(text_items, speaker_items)):
                for span in self.phonemize(text, speaker=item_speaker, neural=neural):
                    flat_spans.append((text_idx, span))

        if not flat_spans:
            return [np.zeros(0, dtype=np.int16) for _ in text_items]

        scales = self._inference_scales(
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_w=noise_w,
            sdp_ratio=sdp_ratio,
        )
        effective_batch_size = batch_size or len(flat_spans)
        segments: list[list[np.ndarray]] = [[] for _ in text_items]

        with torch.inference_mode():
            for start in range(0, len(flat_spans), effective_batch_size):
                chunk = flat_spans[start : start + effective_batch_size]
                chunk_outputs = self._infer_prepared_span_batch(
                    [span for _text_idx, span in chunk],
                    scales,
                )
                for (text_idx, _span), audio in zip(chunk, chunk_outputs):
                    segments[text_idx].append(audio)

        return [
            parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            for parts in segments
        ]

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sdp_ratio: Optional[float] = None,
        neural: bool = False,
        voice_adapter: str | None = None,
    ) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize.
            speaker: Optional speaker label (uses auto-detection if not provided).
            noise_scale: Override for prosody randomness (default from config).
            length_scale: Override for speech rate (default from config).
            noise_w: Override for duration predictor noise (default from config).
            neural: If True, use neural heteronym disambiguation.

        Returns:
            Audio waveform as int16 numpy array.
        """
        return self.synthesize_batch(
            [text],
            speaker=speaker,
            batch_size=1,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_w=noise_w,
            sdp_ratio=sdp_ratio,
            neural=neural,
            voice_adapter=voice_adapter,
        )[0]

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        speaker: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """Synthesize speech and save to a WAV file.

        Args:
            text: Input text to synthesize.
            output_path: Path for the output WAV file.
            speaker: Optional speaker label.
            **kwargs: Additional arguments passed to synthesize().

        Returns:
            Path to the output WAV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio = self.synthesize(text, speaker=speaker, **kwargs)
        write_wav(str(output_path), self.inference_config.sample_rate, audio)

        return output_path

    @property
    def speakers(self) -> dict[str, int]:
        """Get available speaker labels and their IDs."""
        return self.config.get("speaker_id_map", {})

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.inference_config.sample_rate
