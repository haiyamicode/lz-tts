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
    # (source_start, source_end, letters): the source characters behind each run of
    # the token's letters, in order; the letters add up to the token's.
    pieces: tuple[tuple[int, int, int], ...] = ()


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
    """Romanize a complete language span while retaining its source offsets.

    Numeral characters (三, 十, 零) keep their syllables instead of the number
    uroman prefers: the CTC vocabulary has no digits, so 三月 would otherwise
    align as 月 alone.
    """
    from ctc_forced_aligner.text_utils import uroman_instance
    from uroman import RomFormat
    from uroman.uroman import NumEdge

    result = uroman_instance.romanize_string(
        text,
        lcode=language,
        rom_format=RomFormat.EDGES,
    )
    if isinstance(result, str):
        raise TypeError("Uroman did not return offset edges")
    if not any(isinstance(edge, NumEdge) for edge in result):
        return result
    syllables = {
        int(edge.start): edge
        for edge in uroman_instance.romanize_string(
            text,
            lcode=language,
            rom_format=RomFormat.LATTICE,
        )
        if not isinstance(edge, NumEdge)
        and int(edge.end) == int(edge.start) + 1
        and _edge_has_ctc_text(edge)
    }
    edges: list[Any] = []
    for edge in result:
        spelled = [syllables.get(position) for position in range(int(edge.start), int(edge.end))]
        if isinstance(edge, NumEdge) and all(spelled):
            edges.extend(spelled)
        else:
            edges.append(edge)
    return edges


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


def _token_pieces(
    edges: list[Any],
    start: int,
    end: int,
    letters: int,
) -> tuple[tuple[int, int, int], ...]:
    """Split a token's letters over the source characters they were romanized from.

    Each Uroman edge is one source character or an atomic run of them (a CJK
    character and its pinyin, a Latin letter, a digraph rule). If the per-edge
    letters don't add up to the token, the token stays one piece.
    """
    from ctc_forced_aligner.text_utils import normalize_uroman

    pieces = []
    for edge in edges:
        edge_start, edge_end = max(int(edge.start), start), min(int(edge.end), end)
        if edge_end <= edge_start:
            continue
        count = sum(1 for character in normalize_uroman(str(edge.txt)) if not character.isspace())
        if count:
            pieces.append((edge_start, edge_end, count))
    if sum(count for _, _, count in pieces) != letters:
        return ((start, end, letters),)
    return tuple(pieces)


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
        lexical_end = group[-1].end
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
                pieces=tuple(
                    (source_offset + piece_start, source_offset + piece_end, count)
                    for piece_start, piece_end, count in _token_pieces(
                        edges, lexical_start, lexical_end, len(token.split(" "))
                    )
                ),
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


def _piece_timestamps(
    pieces: Sequence[tuple[int, int, int]],
    letter_starts: Sequence[float],
    word_start: float,
    word_end: float,
) -> list[dict[str, Any]]:
    """Time each piece of a word from its first letter's frame.

    Pieces tile the word: one ends where the next starts, the first starts with
    the word and the last ends with it (the word's own padded bounds).
    """
    starts: list[float] = []
    letter = 0
    for _, _, count in pieces:
        starts.append(letter_starts[letter] if letter < len(letter_starts) else word_end)
        letter += count
    if starts:
        starts[0] = word_start
    return [
        {
            "source_start": piece_start,
            "source_end": piece_end,
            "start_seconds": max(word_start, min(start, word_end)),
            "end_seconds": max(word_start, min(starts[index + 1] if index + 1 < len(starts) else word_end, word_end)),
        }
        for index, ((piece_start, piece_end, _), start) in enumerate(zip(pieces, starts))
    ]


def _target_indices(tokens: Sequence[str], tokenizer: Any) -> list[int]:
    """The label ids `ctc_forced_aligner.get_alignments` builds from `tokens`."""
    dictionary = {key.lower(): value for key, value in tokenizer.get_vocab().items()}
    dictionary["<star>"] = len(dictionary)
    return [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]


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
        frames = int(emissions.shape[0])
        targets = _target_indices(prepared.tokens, self._tokenizer)
        repeats = sum(a == b for a, b in zip(targets, targets[1:]))
        if frames < len(targets) + repeats:
            # CTC needs a frame per label plus a blank between repeats: the audio is too
            # short to say this text (a cut take). The C++ aligner would throw anyway.
            # (Our fork of ctc_forced_aligner, src/ctc_forced_aligner, sizes its back pointers
            # for any take that fits; upstream 0.3.0 overran them when a take barely fit.)
            raise ValueError(
                f"Audio too short to align its text: {frames} frames for "
                f"{len(targets)} labels + {repeats} repeats"
            )
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
        unit_spans = [
            span for token, span in zip(prepared.tokens, spans) if token != "<star>"
        ]
        stride_seconds = stride_ms / 1000
        word_timestamps = [
            {
                "text": word["text"],
                "start_seconds": float(word["start"]),
                "end_seconds": float(word["end"]),
                "score": float(word["score"]),
                "source_start": unit.source_start,
                "source_end": unit.source_end,
                "language": unit.language,
                "tokens": _piece_timestamps(
                    unit.pieces,
                    [segment.start * stride_seconds for segment in span if segment.label != blank],
                    float(word["start"]),
                    float(word["end"]),
                ),
            }
            for word, unit, span in zip(raw_words, prepared.units, unit_spans)
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
