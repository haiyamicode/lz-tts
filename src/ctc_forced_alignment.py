"""Word-level timestamps from MahmoudAshraf97/ctc-forced-aligner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import torch
from langcodes import Language

from .piper.word_segmentation import WordSpan, icu_word_spans


@dataclass(frozen=True)
class CtcAlignmentConfig:
    model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    device: str = "cuda:0"
    dtype: str = "float16"
    star_frequency: str = "segment"


@dataclass(frozen=True)
class CtcLanguageSpan:
    source_start: int
    source_end: int
    language: str


@dataclass(frozen=True)
class _TranscriptUnit:
    text: str
    token: str
    source_start: int
    source_end: int
    language: str


@dataclass(frozen=True)
class _PreparedTranscript:
    tokens: list[str]
    starred_text: list[str]
    units: list[_TranscriptUnit]
    languages: list[str]


def _iso_639_3(language: str, fallback: str) -> str:
    try:
        return Language.get(language).to_alpha3()
    except (LookupError, ValueError):
        return fallback


def _romanized_edges(text: str, language: str) -> list[Any]:
    """Romanize a complete language span while retaining its source offsets."""
    from ctc_forced_aligner.text_utils import uroman_instance
    from uroman import RomFormat

    result = uroman_instance.romanize_string(
        text,
        lcode=language,
        rom_format=RomFormat.EDGES,
    )
    if isinstance(result, str):
        raise TypeError("Uroman did not return offset edges")
    return result


def _edge_has_ctc_text(edge: Any) -> bool:
    from ctc_forced_aligner.text_utils import normalize_uroman

    return bool(normalize_uroman(str(edge.txt)))


def _merge_words_crossed_by_edges(
    words: list[WordSpan],
    edges: list[Any],
) -> list[list[WordSpan]]:
    """Keep Uroman edges atomic when one crosses an ICU word boundary."""
    if not words:
        return []

    groups: list[list[WordSpan]] = [[words[0]]]
    for word in words[1:]:
        boundary = word.start
        crosses_boundary = any(
            int(edge.start) < boundary < int(edge.end) and _edge_has_ctc_text(edge)
            for edge in edges
        )
        if crosses_boundary:
            groups[-1].append(word)
        else:
            groups.append([word])
    return groups


def _ctc_token_for_range(
    edges: list[Any],
    start: int,
    end: int,
) -> str:
    from ctc_forced_aligner.text_utils import normalize_uroman

    romanized = normalize_uroman(
        "".join(
            str(edge.txt)
            for edge in edges
            if int(edge.end) > start and int(edge.start) < end
        )
    )
    return " ".join(character for character in romanized if not character.isspace())


def _span_transcript_units(
    text: str,
    *,
    source_offset: int,
    language: str,
    locale: str,
) -> list[_TranscriptUnit]:
    """Prepare ICU units from one context-preserving language-span transform."""
    words = icu_word_spans(text, locale)
    if not words:
        return []

    edges = _romanized_edges(text, language)
    groups = _merge_words_crossed_by_edges(words, edges)
    tokenized_groups = [
        (group, _ctc_token_for_range(edges, group[0].start, group[-1].end))
        for group in groups
    ]
    tokenized_groups = [item for item in tokenized_groups if item[1]]
    units: list[_TranscriptUnit] = []
    for index, (group, token) in enumerate(tokenized_groups):
        lexical_start = group[0].start
        display_start = 0 if index == 0 else lexical_start
        display_end = (
            tokenized_groups[index + 1][0][0].start
            if index + 1 < len(tokenized_groups)
            else len(text)
        )
        units.append(
            _TranscriptUnit(
                text=text[display_start:display_end],
                token=token,
                source_start=source_offset + display_start,
                source_end=source_offset + display_end,
                language=language,
            )
        )
    return units


def _prepare_ctc_transcript(
    text: str,
    language: str,
    *,
    star_frequency: str,
    language_spans: Sequence[CtcLanguageSpan] | None = None,
) -> _PreparedTranscript:
    """Build one mixed-language CTC transcript from Sparrow language spans."""
    if star_frequency not in {"segment", "edges"}:
        raise ValueError("star_frequency must be 'segment' or 'edges'")

    requested = Language.get(language)
    fallback_iso = requested.to_alpha3()
    if language_spans is None:
        language_spans = [CtcLanguageSpan(0, len(text), language)] if text else []

    source_cursor = 0
    for span in language_spans:
        if span.source_start != source_cursor:
            raise ValueError(
                "CTC language spans must provide contiguous coverage in source order: "
                f"expected start {source_cursor}, got {span.source_start}"
            )
        if span.source_end <= span.source_start or span.source_end > len(text):
            raise ValueError(
                f"Invalid CTC language span [{span.source_start}, {span.source_end})"
            )
        if not span.language:
            raise ValueError("CTC language span has no language")
        source_cursor = span.source_end
    if source_cursor != len(text):
        raise ValueError(
            "CTC language spans do not cover the complete source text: "
            f"covered {source_cursor} of {len(text)} characters"
        )

    units: list[_TranscriptUnit] = []
    languages: list[str] = []
    for span in language_spans:
        span_text = text[span.source_start : span.source_end]
        if not span_text.strip():
            continue
        iso3 = _iso_639_3(span.language, fallback_iso)
        span_units = _span_transcript_units(
            span_text,
            source_offset=span.source_start,
            language=iso3,
            locale=span.language,
        )
        if not span_units:
            continue
        units.extend(span_units)
        if iso3 not in languages:
            languages.append(iso3)

    tokens: list[str] = []
    starred_text: list[str] = []
    if star_frequency == "segment":
        for unit in units:
            tokens.extend(("<star>", unit.token))
            starred_text.extend(("<star>", unit.text))
    elif units:
        tokens = ["<star>", *(unit.token for unit in units), "<star>"]
        starred_text = ["<star>", *(unit.text for unit in units), "<star>"]

    return _PreparedTranscript(
        tokens=tokens,
        starred_text=starred_text,
        units=units,
        languages=languages,
    )


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
        language_spans: Sequence[CtcLanguageSpan] | None = None,
    ) -> dict[str, Any]:
        self.load()
        assert self._model is not None and self._tokenizer is not None

        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            postprocess_results,
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
        prepared = _prepare_ctc_transcript(
            text,
            language,
            star_frequency=self.config.star_frequency,
            language_spans=language_spans,
        )
        if not prepared.tokens:
            return {
                "backend": "ctc",
                "valid": False,
                "reason": "empty_transcript",
                "model": self.config.model,
                "language": iso_639_3,
                "languages": prepared.languages,
                "star_frequency": self.config.star_frequency,
                "stride_seconds": float(stride_ms / 1000),
                "word_timestamps": [],
            }
        segments, scores, blank = get_alignments(
            emissions,
            prepared.tokens,
            self._tokenizer,
        )
        spans = get_spans(prepared.tokens, segments, blank)
        raw_words = postprocess_results(
            prepared.starred_text,
            spans,
            stride_ms,
            scores,
        )
        if len(raw_words) != len(prepared.units):
            raise ValueError(
                "CTC aligner returned a different number of timestamps than transcript units: "
                f"{len(raw_words)} != {len(prepared.units)}"
            )
        word_timestamps = [
            {
                "text": word["text"],
                "start_seconds": float(word["start"]),
                "end_seconds": float(word["end"]),
                "score": float(word["score"]),
                "source_start": unit.source_start,
                "source_end": unit.source_end,
                "language": unit.language,
            }
            for word, unit in zip(raw_words, prepared.units)
        ]
        return {
            "backend": "ctc",
            "valid": bool(word_timestamps),
            "reason": "ok" if word_timestamps else "empty_alignment",
            "model": self.config.model,
            "language": iso_639_3,
            "languages": prepared.languages,
            "star_frequency": self.config.star_frequency,
            "stride_seconds": float(stride_ms / 1000),
            "word_timestamps": word_timestamps,
        }


__all__ = ["CtcAlignmentConfig", "CtcForcedAligner", "CtcLanguageSpan"]
