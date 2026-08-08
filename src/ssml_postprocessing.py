"""Alignment-based audio operations described by a parsed SSML document."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .aligned_pauses import PauseMarker, insert_aligned_pauses
from .ssml import BreakOperation, PronunciationOperation


@dataclass(frozen=True)
class AlignedAudioInterval:
    start_sample: int
    end_sample: int
    alignment_indices: tuple[int, ...]


def _is_spoken(value: str) -> bool:
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in value)


def _seconds(timestamp: dict[str, Any], field: str, index: int) -> float:
    try:
        value = float(timestamp[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Alignment segment {index} has no valid {field}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Alignment segment {index} has non-finite {field}")
    return value


def aligned_text_interval(
    operation: PronunciationOperation,
    timestamps: Sequence[dict[str, Any]],
    *,
    audio_samples: int,
    sample_rate: int,
) -> AlignedAudioInterval:
    """Cut a source span at midpoints between its aligned neighboring units."""
    target_indices = tuple(
        index
        for index, timestamp in enumerate(timestamps)
        if _is_spoken(str(timestamp.get("text", "")))
        and int(timestamp.get("source_end", -1)) > operation.start
        and int(timestamp.get("source_start", audio_samples + 1)) < operation.end
    )
    if not target_indices:
        raise ValueError(
            f"Forced alignment did not locate SSML pronunciation span [{operation.start}, {operation.end})"
        )

    first_index, last_index = target_indices[0], target_indices[-1]
    left_index = next(
        (
            index
            for index in range(first_index - 1, -1, -1)
            if _is_spoken(str(timestamps[index].get("text", "")))
        ),
        None,
    )
    right_index = next(
        (
            index
            for index in range(last_index + 1, len(timestamps))
            if _is_spoken(str(timestamps[index].get("text", "")))
        ),
        None,
    )

    if left_index is None:
        start_seconds = 0.0
    else:
        start_seconds = (
            _seconds(timestamps[left_index], "end_seconds", left_index)
            + _seconds(timestamps[first_index], "start_seconds", first_index)
        ) / 2
    if right_index is None:
        end_seconds = audio_samples / sample_rate
    else:
        end_seconds = (
            _seconds(timestamps[last_index], "end_seconds", last_index)
            + _seconds(timestamps[right_index], "start_seconds", right_index)
        ) / 2

    start_sample = round(start_seconds * sample_rate)
    end_sample = round(end_seconds * sample_rate)
    if not 0 <= start_sample < end_sample <= audio_samples:
        raise ValueError(
            "Invalid forced-alignment interval for SSML pronunciation: "
            f"samples [{start_sample}, {end_sample}) of {audio_samples}"
        )
    return AlignedAudioInterval(start_sample, end_sample, target_indices)


def _join_with_crossfade(parts: Sequence[np.ndarray], fade_samples: int) -> np.ndarray:
    nonempty = [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts if np.asarray(part).size]
    if not nonempty:
        return np.zeros(0, dtype=np.float32)
    output = nonempty[0].copy()
    for part in nonempty[1:]:
        overlap = min(fade_samples, output.size, part.size)
        if overlap:
            fade_out = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
            fade_in = 1.0 - fade_out
            output[-overlap:] = output[-overlap:] * fade_out + part[:overlap] * fade_in
            output = np.concatenate((output, part[overlap:]))
        else:
            output = np.concatenate((output, part))
    return output


def splice_pronunciations(
    baseline_audio: np.ndarray,
    replacement_audios: Sequence[np.ndarray],
    sample_rate: int,
    operations: Sequence[PronunciationOperation],
    baseline_timestamps: Sequence[dict[str, Any]],
    replacement_timestamps: Sequence[Sequence[dict[str, Any]]],
    *,
    crossfade_seconds: float = 0.01,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Replace all pronunciation spans from one full-context Sparrow render."""
    baseline = np.asarray(baseline_audio, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if crossfade_seconds < 0:
        raise ValueError("crossfade_seconds must not be negative")

    if len(replacement_audios) != len(operations) or len(replacement_timestamps) != len(operations):
        raise ValueError("Each SSML pronunciation requires one aligned replacement render")

    resolved = []
    for operation, replacement_audio, item_timestamps in zip(
        operations, replacement_audios, replacement_timestamps
    ):
        replacement = np.asarray(replacement_audio, dtype=np.float32).reshape(-1)
        resolved.append(
            (
                operation,
                replacement,
                aligned_text_interval(
                    operation,
                    baseline_timestamps,
                    audio_samples=baseline.size,
                    sample_rate=sample_rate,
                ),
                aligned_text_interval(
                    operation,
                    item_timestamps,
                    audio_samples=replacement.size,
                    sample_rate=sample_rate,
                ),
            )
        )
    resolved.sort(key=lambda item: item[2].start_sample)
    for previous, current in zip(resolved, resolved[1:]):
        if previous[2].end_sample > current[2].start_sample:
            raise ValueError("Aligned SSML pronunciation intervals overlap in baseline audio")

    parts: list[np.ndarray] = []
    report: list[dict[str, Any]] = []
    cursor = 0
    for operation, replacement, source_interval, replacement_interval in resolved:
        parts.append(baseline[cursor : source_interval.start_sample])
        parts.append(replacement[replacement_interval.start_sample : replacement_interval.end_sample])
        report.append(
            {
                "source_start": operation.start,
                "source_end": operation.end,
                "baseline_start_seconds": source_interval.start_sample / sample_rate,
                "baseline_end_seconds": source_interval.end_sample / sample_rate,
                "replacement_start_seconds": replacement_interval.start_sample / sample_rate,
                "replacement_end_seconds": replacement_interval.end_sample / sample_rate,
                "baseline_alignment_indices": list(source_interval.alignment_indices),
                "replacement_alignment_indices": list(replacement_interval.alignment_indices),
            }
        )
        cursor = source_interval.end_sample
    parts.append(baseline[cursor:])
    return _join_with_crossfade(parts, round(crossfade_seconds * sample_rate)), report


def insert_ssml_breaks(
    text: str,
    audio: np.ndarray,
    sample_rate: int,
    operations: Sequence[BreakOperation],
    timestamps: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Convert parsed break positions to the shared forced-alignment inserter."""
    markers = [
        PauseMarker(
            duration_seconds=operation.duration_seconds,
            word_index=0,
            prefix_text=text[: operation.position].rstrip(),
            source_start=operation.position,
            source_end=operation.position,
        )
        for operation in operations
    ]
    return insert_aligned_pauses(audio, sample_rate, markers, list(timestamps))


__all__ = [
    "AlignedAudioInterval",
    "aligned_text_interval",
    "insert_ssml_breaks",
    "splice_pronunciations",
]
