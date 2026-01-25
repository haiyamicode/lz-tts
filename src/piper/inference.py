"""Piper TTS inference pipeline.

Provides a minimal interface for running TTS inference using a Piper/VITS model checkpoint.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

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
    sample_rate: int = 22050


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

        # Load config
        with open(self.config_path) as f:
            self.config = json.load(f)

        # Extract inference parameters
        inference_cfg = self.config.get("inference", {})
        self.inference_config = InferenceConfig(
            noise_scale=inference_cfg.get("noise_scale", 0.667),
            length_scale=inference_cfg.get("length_scale", 1.0),
            noise_w=inference_cfg.get("noise_w", 0.8),
            sample_rate=self.config.get("audio", {}).get("sample_rate", 22050),
        )

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        _LOGGER.info("Loading model: %s", self.checkpoint_path.name)
        self.model = VitsModel.load_from_checkpoint(
            str(self.checkpoint_path), dataset=None, weights_only=False
        )
        self.model.eval()
        self.model.to(self.device)

        # Remove weight norm for inference (suppress warnings)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Removing weight norm.*")
            self.model.model_g.dec.remove_weight_norm()

        # Setup BERT semantic tokenizer if model was trained with it
        self.use_bert = bool(getattr(self.model.hparams, "use_bert", False))
        self.semantic_tokenizer = None
        self._build_bert_input = None

        if self.use_bert:
            from .semantic import (
                SemanticTokenizer,
                build_bert_input,
            )

            bert_model_name = getattr(self.model.hparams, "bert_model_name", None)
            _LOGGER.info("Loading BERT tokenizer: %s", bert_model_name or "default")
            self.semantic_tokenizer = SemanticTokenizer(model_name=bert_model_name)
            self._build_bert_input = build_bert_input

        _LOGGER.info(
            "Model ready: %s (device=%s, speakers=%d, bert=%s)",
            self.checkpoint_path.name,
            self.device,
            len(self.speakers),
            self.use_bert,
        )

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
                text, self.config_path, speaker, espeak_data_path, neural=neural
            )
            return [span]
        else:
            return phonemize_spans_with_speakers(
                text, self.config_path, espeak_data_path, neural=neural
            )

    def synthesize_span(
        self,
        text: str,
        speaker: Optional[str] = None,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        neural: bool = True,
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
        from .preprocess import phonemize_text_for_speaker

        scales = [
            noise_scale if noise_scale is not None else self.inference_config.noise_scale,
            length_scale if length_scale is not None else self.inference_config.length_scale,
            noise_w if noise_w is not None else self.inference_config.noise_w,
        ]

        # For single-speaker models, speaker can be None - phonemize will use default speaker_id=0
        span = phonemize_text_for_speaker(text, self.config_path, speaker or "", None, neural=neural)
        phoneme_ids = span["phoneme_ids"]
        speaker_id = span.get("speaker_id", 0)

        _LOGGER.info("synthesize_span: phoneme_ids[:20]=%s, len=%d", phoneme_ids[:20], len(phoneme_ids))

        with torch.no_grad():
            text_tensor = torch.LongTensor(phoneme_ids).unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(phoneme_ids)]).to(self.device)
            sid = torch.LongTensor([speaker_id]).to(self.device)

            bert_input = None
            if self.use_bert and self.semantic_tokenizer and text:
                bert_dict = self._build_bert_input([text], self.semantic_tokenizer)
                if bert_dict is not None:
                    bert_input = {
                        "input_ids": bert_dict["input_ids"].to(self.device),
                        "attention_mask": bert_dict["attention_mask"].to(self.device),
                    }

            audio = self.model(
                text_tensor, text_lengths, scales, sid=sid, bert_input=bert_input
            )
            audio = audio.detach().cpu().numpy()
            audio = audio_float_to_int16(audio)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

        return audio

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        noise_scale: Optional[float] = None,
        length_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        neural: bool = False,
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
        # Get inference scales
        scales = [
            noise_scale if noise_scale is not None else self.inference_config.noise_scale,
            length_scale if length_scale is not None else self.inference_config.length_scale,
            noise_w if noise_w is not None else self.inference_config.noise_w,
        ]

        # Phonemize text
        spans = self.phonemize(text, speaker=speaker, neural=neural)

        # Synthesize each span and concatenate
        audio_segments = []

        with torch.no_grad():
            for span in spans:
                phoneme_ids = span["phoneme_ids"]
                speaker_id = span.get("speaker_id")
                span_text = span.get("text", "")

                # Prepare input tensors
                text_tensor = torch.LongTensor(phoneme_ids).unsqueeze(0).to(self.device)
                text_lengths = torch.LongTensor([len(phoneme_ids)]).to(self.device)
                sid = (
                    torch.LongTensor([speaker_id]).to(self.device)
                    if speaker_id is not None
                    else None
                )

                # Prepare BERT input if enabled
                bert_input = None
                if self.use_bert and self.semantic_tokenizer and span_text:
                    bert_dict = self._build_bert_input([span_text], self.semantic_tokenizer)
                    if bert_dict is not None:
                        bert_input = {
                            "input_ids": bert_dict["input_ids"].to(self.device),
                            "attention_mask": bert_dict["attention_mask"].to(self.device),
                        }

                # Run inference
                audio = self.model(
                    text_tensor, text_lengths, scales, sid=sid, bert_input=bert_input
                )
                audio = audio.detach().cpu().numpy()
                audio = audio_float_to_int16(audio)

                # Ensure 1-D array
                if audio.ndim > 1:
                    audio = audio.reshape(-1)

                audio_segments.append(audio)

        # Concatenate all segments
        if len(audio_segments) == 1:
            return audio_segments[0]
        return np.concatenate(audio_segments, axis=0)

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
