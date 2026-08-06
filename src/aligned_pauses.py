"""Parse custom pause markers and splice silence at forced-aligned boundaries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np

_PAUSE_RE = re.compile(
    r"\[pause\s+(?P<duration>(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>ms|s)\]",
    re.IGNORECASE,
)
_POSSIBLE_PAUSE_RE = re.compile(r"\[pause\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


@dataclass(frozen=True)
class PauseMarker:
    duration_seconds: float
    word_index: int
    prefix_text: str
    source_start: int
    source_end: int


def parse_pause_markers(text: str, *, maximum_seconds: float = 60.0) -> tuple[str, list[PauseMarker]]:
    """Remove ``[pause 3s]`` markers and anchor each to a word boundary."""
    matches = list(_PAUSE_RE.finditer(text))
    possible = list(_POSSIBLE_PAUSE_RE.finditer(text))
    if len(matches) != len(possible):
        raise ValueError("Malformed pause marker; expected syntax such as [pause 3s] or [pause 500ms]")
    if not matches:
        raise ValueError("Text contains no pause markers")

    markers: list[PauseMarker] = []
    pieces: list[str] = []
    cursor = 0
    prefix_without_markers = ""
    for match in matches:
        fragment = text[cursor : match.start()]
        pieces.append(fragment)
        prefix_without_markers += fragment
        duration = float(match.group("duration"))
        if match.group("unit").lower() == "ms":
            duration /= 1000.0
        if not 0 < duration <= maximum_seconds:
            raise ValueError(f"Pause must be greater than 0 and at most {maximum_seconds:g} seconds")
        markers.append(
            PauseMarker(
                duration_seconds=duration,
                word_index=len(_WORD_RE.findall(prefix_without_markers)),
                prefix_text=prefix_without_markers.rstrip(),
                source_start=match.start(),
                source_end=match.end(),
            )
        )
        pieces.append(" ")
        prefix_without_markers += " "
        cursor = match.end()
    pieces.append(text[cursor:])
    clean_text = re.sub(r"\s+", " ", "".join(pieces)).strip()
    if not clean_text:
        raise ValueError("Pause markers leave no text to synthesize")
    return clean_text, markers


def _normalized_alignment_text(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value)
        if not character.isspace()
    )


def _is_spoken_segment(value: str) -> bool:
    """Return whether an alignment segment represents something pronounced."""
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in value)


def _alignment_boundary(marker: PauseMarker, word_timestamps: list[dict[str, Any]]) -> int:
    """Map a source-text marker to the aligner's language-dependent segments."""
    target = _normalized_alignment_text(marker.prefix_text)
    if not target:
        return 0

    aligned_prefix = ""
    for index, word in enumerate(word_timestamps):
        aligned_prefix += _normalized_alignment_text(str(word["text"]))
        if aligned_prefix == target:
            boundary = index + 1
            while boundary < len(word_timestamps) and not _normalized_alignment_text(
                str(word_timestamps[boundary]["text"])
            ):
                boundary += 1
            return boundary
        if not target.startswith(aligned_prefix):
            break
    raise ValueError(
        "Could not map pause marker source prefix to forced-alignment segments: "
        f"{marker.prefix_text!r}"
    )


def _timestamp(word: dict[str, Any], field: str, index: int) -> float:
    try:
        value = float(word[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Alignment segment {index} has no valid {field}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Alignment segment {index} has non-finite {field}")
    return value


def _resolve_cut(
    audio_samples: int,
    sample_rate: int,
    alignment_word_index: int,
    word_timestamps: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """Resolve a marker using its neighboring pronounced alignment segments."""
    left_index = next(
        (
            index
            for index in range(alignment_word_index - 1, -1, -1)
            if _is_spoken_segment(str(word_timestamps[index]["text"]))
        ),
        None,
    )
    right_index = next(
        (
            index
            for index in range(alignment_word_index, len(word_timestamps))
            if _is_spoken_segment(str(word_timestamps[index]["text"]))
        ),
        None,
    )

    if left_index is None and right_index is None:
        raise ValueError("Forced alignment contains no spoken segments")
    if left_index is None:
        return 0, {
            "left_word": None,
            "right_word": str(word_timestamps[right_index]["text"]),
            "left_alignment_index": None,
            "right_alignment_index": right_index,
            "alignment_gap_start_seconds": None,
            "alignment_gap_end_seconds": _timestamp(
                word_timestamps[right_index], "start_seconds", right_index
            ),
            "cut_strategy": "audio_start",
        }
    if right_index is None:
        return audio_samples, {
            "left_word": str(word_timestamps[left_index]["text"]),
            "right_word": None,
            "left_alignment_index": left_index,
            "right_alignment_index": None,
            "alignment_gap_start_seconds": _timestamp(
                word_timestamps[left_index], "end_seconds", left_index
            ),
            "alignment_gap_end_seconds": None,
            "cut_strategy": "audio_end",
        }

    left_end = _timestamp(word_timestamps[left_index], "end_seconds", left_index)
    right_start = _timestamp(word_timestamps[right_index], "start_seconds", right_index)
    audio_seconds = audio_samples / sample_rate
    if not 0 <= left_end <= audio_seconds:
        raise ValueError(f"Left word end {left_end:.6f}s is outside the audio")
    if not 0 <= right_start <= audio_seconds:
        raise ValueError(f"Right word start {right_start:.6f}s is outside the audio")
    if left_end > right_start:
        raise ValueError(
            "Neighboring spoken-word timestamps overlap at pause boundary: "
            f"left end={left_end:.6f}s, right start={right_start:.6f}s"
        )

    midpoint = (left_end + right_start) / 2
    cut = round(midpoint * sample_rate)
    return cut, {
        "left_word": str(word_timestamps[left_index]["text"]),
        "right_word": str(word_timestamps[right_index]["text"]),
        "left_alignment_index": left_index,
        "right_alignment_index": right_index,
        "alignment_gap_start_seconds": left_end,
        "alignment_gap_end_seconds": right_start,
        "cut_strategy": "aligned_spoken_neighbors_midpoint",
    }


def insert_aligned_pauses(
    audio: np.ndarray,
    sample_rate: int,
    markers: list[PauseMarker],
    word_timestamps: list[dict[str, Any]],
    *,
    fade_seconds: float = 0.005,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Insert silence at forced-aligned word gaps after one continuous generation."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if fade_seconds < 0:
        raise ValueError("fade_seconds must not be negative")
    if not word_timestamps:
        raise ValueError("Forced aligner returned no word timestamps")

    insertions_by_boundary: dict[int, dict[str, Any]] = {}
    for marker in markers:
        alignment_word_index = _alignment_boundary(marker, word_timestamps)
        cut, cut_details = _resolve_cut(
            audio.size,
            sample_rate,
            alignment_word_index,
            word_timestamps,
        )
        insertion = insertions_by_boundary.setdefault(
            alignment_word_index,
            {
                "cut": cut,
                "markers": [],
                "details": {
                    **cut_details,
                    "alignment_word_index": alignment_word_index,
                    "cut_seconds": cut / sample_rate,
                },
            },
        )
        if insertion["cut"] != cut:
            raise ValueError("Identical text boundaries resolved to different audio positions")
        insertion["markers"].append(marker)

    insertions = sorted(insertions_by_boundary.values(), key=lambda item: item["cut"])
    spliced_audio = audio.copy()
    fade_samples = round(fade_seconds * sample_rate)
    if fade_samples:
        for insertion in insertions:
            cut = insertion["cut"]
            if cut == 0 or cut == spliced_audio.size:
                continue
            fade_out_start = max(0, cut - fade_samples)
            fade_in_end = min(spliced_audio.size, cut + fade_samples)
            if fade_out_start < cut:
                spliced_audio[fade_out_start:cut] *= np.linspace(
                    1.0,
                    0.0,
                    cut - fade_out_start,
                    endpoint=True,
                    dtype=np.float32,
                )
            if cut < fade_in_end:
                spliced_audio[cut:fade_in_end] *= np.linspace(
                    0.0,
                    1.0,
                    fade_in_end - cut,
                    endpoint=True,
                    dtype=np.float32,
                )

    pieces: list[np.ndarray] = []
    report: list[dict[str, Any]] = []
    cursor = 0
    inserted_samples = 0
    for insertion in insertions:
        cut = insertion["cut"]
        markers_at_boundary: list[PauseMarker] = insertion["markers"]
        details = insertion["details"]
        if cut < cursor:
            raise ValueError("Pause boundaries are not monotonic")
        pieces.append(spliced_audio[cursor:cut])
        duration_seconds = sum(marker.duration_seconds for marker in markers_at_boundary)
        silence_samples = round(duration_seconds * sample_rate)
        pieces.append(np.zeros(silence_samples, dtype=np.float32))
        final_start = cut + inserted_samples
        report.append(
            {
                **details,
                "duration_seconds": duration_seconds,
                "final_pause_start_seconds": final_start / sample_rate,
                "final_pause_end_seconds": (final_start + silence_samples) / sample_rate,
                "source_word_index": markers_at_boundary[0].word_index,
                "marker_count": len(markers_at_boundary),
                "fade_seconds": fade_seconds if 0 < cut < audio.size else 0.0,
            }
        )
        inserted_samples += silence_samples
        cursor = cut
    pieces.append(spliced_audio[cursor:])
    return np.concatenate(pieces), report


__all__ = ["PauseMarker", "insert_aligned_pauses", "parse_pause_markers"]
