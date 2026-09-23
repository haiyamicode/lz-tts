"""Tests for long-input chunking.

Long inputs used to reach a backend as a single generation, which either
overflowed Sparrow's model window (tensor mismatch) or spent minutes on one
VoxCPM generation. These tests pin the real contract of the chunking layer
with the real cl100k token counter: pieces must fit the hard token budget and
the text must survive the split unchanged, cut on text boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.api.server import SynthesisChunkingConfig, _sparrow_chunk_segments
from src.api.text_chunking import (
    batch_groups_by_weight,
    chunk_source_ranges,
    chunk_synthesis_texts,
    concat_chunk_audios,
    expand_chunks,
    sparrow_batch_weights,
)
from src.text_splitter import count_cl100k_tokens

SOFT_LIMIT = 100
HARD_LIMIT = 150


def _collapse(text: str) -> str:
    """Content comparison that ignores the whitespace the splitter may move."""
    return "".join(text.split())


def _split(text: str) -> list[str]:
    chunks, counts = chunk_synthesis_texts(
        [text],
        length_function=count_cl100k_tokens,
        soft_limit=SOFT_LIMIT,
        hard_limit=HARD_LIMIT,
    )
    assert counts[0] == len(chunks)
    return chunks


def test_short_texts_are_returned_untouched() -> None:
    texts = ["Hello there.", "  Spaced out.  ", "   ", ""]
    chunks, counts = chunk_synthesis_texts(
        texts,
        length_function=count_cl100k_tokens,
        soft_limit=SOFT_LIMIT,
        hard_limit=HARD_LIMIT,
    )

    assert counts == [1, 1, 1, 1]
    assert chunks == texts


def test_long_text_splits_on_text_boundaries_within_hard_limit() -> None:
    text = (
        "Hormonal and metabolic health is not only about tests or treatment. "
        "Everyday habits matter just as much, and they compound over time. "
        "Sleep, movement, protein and fibre all shape insulin sensitivity. "
        "Small consistent choices beat dramatic short bursts of effort. "
    ) * 12
    assert count_cl100k_tokens(text) > HARD_LIMIT

    chunks = _split(text)

    assert len(chunks) > 1
    assert all(count_cl100k_tokens(chunk) <= HARD_LIMIT for chunk in chunks)
    assert _collapse("".join(chunks)) == _collapse(text)
    # The splitter only cuts between words: the chunks are a word-level
    # partition of the input, never a partial word.
    assert [word for chunk in chunks for word in chunk.split()] == text.split()


def test_chinese_text_without_spaces_still_fits_the_budget() -> None:
    text = "今天天气很好我们一起去公园散步吧" * 24
    assert count_cl100k_tokens(text) > HARD_LIMIT

    chunks = _split(text)

    assert len(chunks) > 1
    assert all(count_cl100k_tokens(chunk) <= HARD_LIMIT for chunk in chunks)
    assert _collapse("".join(chunks)) == _collapse(text)


def test_long_sparrow_segment_expands_into_token_sized_pieces() -> None:
    text = (
        "আমি বাংলায় কথা বলতে পারি, এবং ইংরেজিতেও লিখতে পারি। " * 20
    ).strip()
    assert count_cl100k_tokens(text) > HARD_LIMIT
    segment = {
        "text": text,
        "lang": "bn",
        "model": "lzspeech-sparrow",
        "speaker": "en-US",
    }

    pieces = _sparrow_chunk_segments([segment])

    assert len(pieces) > 1
    assert all(piece["lang"] == "bn" for piece in pieces)
    assert all(piece["model"] == "lzspeech-sparrow" for piece in pieces)
    assert all(piece["speaker"] == "en-US" for piece in pieces)
    assert all(count_cl100k_tokens(piece["text"]) <= HARD_LIMIT for piece in pieces)
    assert _collapse("".join(piece["text"] for piece in pieces)) == _collapse(text)
    assert [word for piece in pieces for word in piece["text"].split()] == text.split()


def test_short_sparrow_segment_is_left_alone() -> None:
    segment = {"text": "One short sentence.", "lang": "en", "model": "m"}

    pieces = _sparrow_chunk_segments([segment])

    assert pieces == [segment]


def test_sparrow_batch_weights_count_long_texts_as_multiple_items() -> None:
    texts = ["x" * 400, "x" * 500, "x" * 501, "x" * 2000, ""]

    assert sparrow_batch_weights(texts) == [1, 1, 2, 4, 1]


def test_batch_groups_by_weight_respects_the_budget() -> None:
    weights = sparrow_batch_weights(["x" * 2000] * 5)
    assert weights == [4, 4, 4, 4, 4]

    assert batch_groups_by_weight(weights, 8) == [[0, 1], [2, 3], [4]]
    # An item heavier than the budget still gets its own batch.
    assert batch_groups_by_weight([12, 1], 8) == [[0], [1]]


def test_expand_chunks_repeats_per_item_values() -> None:
    assert expand_chunks(["en", "bn"], [2, 1]) == ["en", "en", "bn"]
    assert expand_chunks(None, [2, 1]) is None


def test_concat_chunk_audios_groups_in_order() -> None:
    audios = [
        np.full(3, 1, dtype=np.int16),
        np.full(2, 2, dtype=np.int16),
        np.full(4, 3, dtype=np.int16),
    ]

    single = concat_chunk_audios(audios, [1, 1, 1])
    assert len(single) == 3
    assert single[0] is audios[0]
    assert single[2] is audios[2]

    grouped = concat_chunk_audios(audios, [2, 1])
    assert len(grouped) == 2
    assert grouped[0].tolist() == [1, 1, 1, 2, 2]
    assert grouped[1].tolist() == [3, 3, 3, 3]


def test_chunking_config_rejects_inverted_limits() -> None:
    with pytest.raises(ValueError):
        SynthesisChunkingConfig(soft_text_token_limit=300, hard_text_token_limit=200)

    config = SynthesisChunkingConfig()
    assert config.enabled is True
    assert config.soft_text_token_limit == SOFT_LIMIT
    assert config.hard_text_token_limit == HARD_LIMIT


def _is_subsequence(needle: str, hay: str) -> bool:
    iterator = iter(hay)
    return all(char in iterator for char in needle)


def test_chunk_source_ranges_short_text_is_identity() -> None:
    assert chunk_source_ranges("  hello world  ", ["  hello world  "]) == [(0, 15)]
    assert chunk_source_ranges("", [""]) == [(0, 0)]


def test_chunk_source_ranges_recover_despite_eaten_separators() -> None:
    """Chunk ranges must locate each chunk even when the splitter drops the
    separator character at split boundaries."""
    texts = [
        ("The quick brown fox jumps over the lazy dog. " * 30).strip(),
        ("word word word word word word " * 60).strip(),
        ("الرحمن الرحيم الرحمن الرحيم " * 40).strip(),
        ("今天天气很好我们去公园散步 " * 30).strip(),
        "\n\n  " + ("line one. line two, line three; " * 30) + "  \t\n",
    ]
    for text in texts:
        assert count_cl100k_tokens(text) > HARD_LIMIT
        chunks = _split(text)
        ranges = chunk_source_ranges(text, chunks)

        assert all(r is not None for r in ranges), (text[:40], chunks, ranges)
        previous_end = 0
        for chunk, (start, end) in zip(chunks, ranges):
            assert start >= previous_end, (chunk, start, previous_end)
            previous_end = end
            # The chunk is the source slice minus dropped separator chars:
            # every chunk character appears in order inside its range.
            assert _is_subsequence(chunk, text[start:end]), (chunk, text[start:end])
            assert text[start] == chunk[0] or not chunk
        # No content is lost: joining the ranges' content covers all words.
        joined = "".join(text[start:end] for start, end in ranges if (start, end) != (0, 0))
        assert _collapse(joined) == _collapse(text.strip())


def test_chunk_source_ranges_cjk_chunks_are_exact_slices() -> None:
    text = "今天天气很好我们一起去公园散步吧" * 24
    assert count_cl100k_tokens(text) > HARD_LIMIT
    chunks = _split(text)

    ranges = chunk_source_ranges(text, chunks)
    assert all(r is not None for r in ranges)
    # CJK text has no separators to drop: chunks are exact contiguous slices.
    for chunk, (start, end) in zip(chunks, ranges):
        assert text[start:end] == chunk
    assert ranges[-1][1] == len(text)


def test_chunk_source_ranges_fails_closed_on_unplaceable_chunks() -> None:
    # A chunk that cannot be embedded in the source poisons its range and
    # every range after it: callers degrade instead of guessing offsets.
    assert chunk_source_ranges("abc def", ["zzz", "def"]) == [None, None]
    assert chunk_source_ranges("abc def", ["abc", "xyz", "def"]) == [(0, 3), None, None]
