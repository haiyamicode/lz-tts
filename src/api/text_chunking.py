"""Split long synthesis input into chunks a backend can generate safely.

Sparrow/VITS rejects inputs whose phoneme sequence overflows the model window
and VoxCPM scales its generation budget with input length, so a single long
request either fails outright or occupies the backend for minutes. Long inputs
are instead split on text boundaries with the shared recursive splitter (the
same cl100k token budget for both backends) and the generated audio is
concatenated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from src.text_splitter import count_cl100k_tokens, split_text

T = TypeVar("T")

# Sparrow's batched VITS call pads every span to the longest span in the batch,
# so one long text costs as much as several short ones. Text is therefore
# counted in batch items of this many characters.
SPARROW_BATCH_ITEM_CHARS = 500


def chunk_synthesis_texts(
    texts: Sequence[str],
    *,
    length_function: Callable[[str], int],
    soft_limit: int,
    hard_limit: int,
) -> tuple[list[str], list[int]]:
    """Expand ``texts`` into generation-sized chunks.

    Returns the flat chunk list plus how many chunks each input produced, so
    callers can concatenate the generated audio back into per-input results.
    Text within the soft limit is returned untouched.
    """
    chunks: list[str] = []
    counts: list[int] = []
    for text in texts:
        stripped = text.strip()
        if not stripped or length_function(stripped) <= soft_limit:
            parts = [text]
        else:
            parts = split_text(
                stripped,
                hard_limit,
                soft_max_length=soft_limit,
                length_function=length_function,
                measure_merged_length=True,
            )
            parts = [part for part in parts if part.strip()] or [text]
        chunks.extend(parts)
        counts.append(len(parts))
    return chunks, counts


def sparrow_batch_weights(
    texts: Sequence[str],
    *,
    item_chars: int = SPARROW_BATCH_ITEM_CHARS,
) -> list[int]:
    """Count texts in batch items, one item per ``item_chars`` characters.

    A 2000-character text counts as 4 items, so a batch holds as many of them
    as its cost allows instead of treating every text as one slot.
    """
    return [max(1, -(-len(text) // item_chars)) for text in texts]


def batch_groups_by_weight(weights: Sequence[int], max_weight: int) -> list[list[int]]:
    """Group item indices into batches whose summed weight fits ``max_weight``.

    An item heavier than the whole budget still gets its own batch: it cannot
    be split any further here.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    total = 0
    for index, weight in enumerate(weights):
        if current and total + weight > max_weight:
            groups.append(current)
            current, total = [], 0
        current.append(index)
        total += weight
    if current:
        groups.append(current)
    return groups


def expand_chunks(values: Sequence[T] | None, chunk_counts: Sequence[int]) -> list[T] | None:
    """Repeat one per-input value for every chunk that input produced."""
    if values is None:
        return None
    return [value for value, count in zip(values, chunk_counts) for _ in range(count)]


def concat_chunk_audios(
    audios: Sequence[np.ndarray],
    chunk_counts: Sequence[int],
) -> list[np.ndarray]:
    """Concatenate per-chunk audio back into one array per original input."""
    results: list[np.ndarray] = []
    offset = 0
    for count in chunk_counts:
        group = audios[offset : offset + count]
        offset += count
        if not group:
            results.append(np.zeros(0, dtype=audios[0].dtype if audios else np.float32))
        elif len(group) == 1:
            results.append(group[0])
        else:
            results.append(np.concatenate(group, axis=0))
    return results
