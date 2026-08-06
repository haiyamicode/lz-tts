"""Word-level timestamps from MahmoudAshraf97/ctc-forced-aligner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import torch
from langcodes import Language


@dataclass(frozen=True)
class CtcAlignmentConfig:
    model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    device: str = "cuda:2"
    dtype: str = "float16"
    star_frequency: str = "segment"


class CtcForcedAligner:
    """Load one multilingual CTC model and align utterances at word level."""

    def __init__(self, config: CtcAlignmentConfig | None = None):
        self.config = config or CtcAlignmentConfig()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        from ctc_forced_aligner import load_alignment_model

        dtype = getattr(torch, self.config.dtype)
        if self.config.device == "cpu" and dtype != torch.float32:
            dtype = torch.float32
        self._model, self._tokenizer = load_alignment_model(
            self.config.device,
            model_path=self.config.model,
            dtype=dtype,
        )

    def align_words(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
        *,
        language: str,
    ) -> dict[str, Any]:
        self.load()
        assert self._model is not None and self._tokenizer is not None

        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            postprocess_results,
            preprocess_text,
        )

        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16_000:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=16_000,
            ).astype(np.float32, copy=False)
        waveform_tensor = torch.from_numpy(waveform).to(
            device=self._model.device,
            dtype=self._model.dtype,
        )
        emissions, stride_ms = generate_emissions(
            self._model,
            waveform_tensor,
            batch_size=1,
        )

        iso_639_3 = Language.get(language).to_alpha3()
        tokens, starred_text = preprocess_text(
            text,
            romanize=True,
            language=iso_639_3,
            split_size="word",
            star_frequency=self.config.star_frequency,
        )
        segments, scores, blank = get_alignments(
            emissions,
            tokens,
            self._tokenizer,
        )
        spans = get_spans(tokens, segments, blank)
        raw_words = postprocess_results(starred_text, spans, stride_ms, scores)
        word_timestamps = [
            {
                "text": word["text"],
                "start_seconds": float(word["start"]),
                "end_seconds": float(word["end"]),
                "score": float(word["score"]),
            }
            for word in raw_words
        ]
        return {
            "backend": "ctc",
            "valid": bool(word_timestamps),
            "reason": "ok" if word_timestamps else "empty_alignment",
            "model": self.config.model,
            "language": iso_639_3,
            "star_frequency": self.config.star_frequency,
            "stride_seconds": float(stride_ms / 1000),
            "word_timestamps": word_timestamps,
        }


__all__ = ["CtcAlignmentConfig", "CtcForcedAligner"]
