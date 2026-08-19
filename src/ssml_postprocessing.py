"""Alignment-based audio operations described by a parsed SSML document."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .aligned_pauses import (
    PauseMarker,
    ResolvedPause,
    insert_resolved_pauses,
    resolve_aligned_pauses,
)
from .ssml import BreakOperation


def resolve_ssml_breaks(
    text: str,
    audio_samples: int,
    sample_rate: int,
    operations: Sequence[BreakOperation],
    timestamps: Sequence[dict[str, Any]],
) -> list[ResolvedPause]:
    """Resolve parsed SSML breaks to forced-aligned audio sample cuts."""
    markers = [
        PauseMarker(
            duration_seconds=operation.duration_seconds,
            word_index=0,
            prefix_text=text[: operation.position].rstrip(),
            source_start=operation.position,
            source_end=operation.position,
            alignment_position=operation.position,
        )
        for operation in operations
    ]
    return resolve_aligned_pauses(
        audio_samples,
        sample_rate,
        markers,
        list(timestamps),
    )


def insert_ssml_breaks(
    text: str,
    audio: np.ndarray,
    sample_rate: int,
    operations: Sequence[BreakOperation],
    timestamps: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Convert parsed break positions to the shared forced-alignment inserter."""
    flattened = np.asarray(audio, dtype=np.float32).reshape(-1)
    resolved = resolve_ssml_breaks(
        text,
        flattened.size,
        sample_rate,
        operations,
        timestamps,
    )
    return insert_resolved_pauses(flattened, sample_rate, resolved)


__all__ = [
    "insert_ssml_breaks",
    "resolve_ssml_breaks",
]
