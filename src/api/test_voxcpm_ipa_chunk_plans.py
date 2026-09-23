"""Tests for the VoxCPM IPA chunk plan (controlled text -> source mapping).

The IPA mechanism gates the IPA adapter at per-patch positions inside one
generation, so long controlled documents are split with the shared
token-budget splitter and every chunk is generated on its own. These tests
pin, with the real SSML parser and the real splitter, that chunks locate
exactly in the controlled text, that each control lands in exactly one
chunk (or is skipped when it crosses a boundary), and that the plain-text
twin carries the original source spans for the duration model.
"""

from __future__ import annotations

from src.api.server import (
    _chunk_voxcpm_texts,
    _prepare_voxcpm_ipa_text,
    _voxcpm_ipa_chunk_plans,
)
from src.api.text_chunking import chunk_source_ranges
from src.ssml import PronunciationOperation, SSMLDocument, parse_ssml
from src.text_splitter import count_cl100k_tokens

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
    "Squirrels carry acorns through the autumn leaves while the wind "
    "turns the season and the farmers gather the last of the harvest. "
) * 6


def _document(text: str, spans: list[tuple[int, int, str]]) -> SSMLDocument:
    parts = []
    cursor = 0
    for start, end, ipa in spans:
        parts.append(text[cursor:start])
        parts.append(f'<phoneme alphabet="ipa" ph="{ipa}">{text[start:end]}</phoneme>')
        cursor = end
    parts.append(text[cursor:])
    return parse_ssml("<speak>" + "".join(parts) + "</speak>")


def test_short_document_is_one_identity_plan() -> None:
    document = _document(
        "The quick red fox.",
        [(4, 12, "kwɪk rɛd")],
    )
    controlled_text, _language, controls = _prepare_voxcpm_ipa_text(document, None, None, "en")
    assert len(controls) == 1

    plans = _voxcpm_ipa_chunk_plans(controlled_text, controls, document)

    assert len(plans) == 1
    plan = plans[0]
    assert plan["unplaced"] is False
    assert plan["text"] == controlled_text
    assert (plan["source_start"], plan["source_end"]) == (0, len(document.text))
    # The single control keeps its offsets and its original source span
    # appears verbatim in the plain twin.
    control = controls[0]
    assert plan["controls"] == [control]
    assert plan["control_indices"] == [0]
    twin_start, twin_end, ipa = plan["twin_overrides"][0]
    assert ipa == control["target_ipa"]
    assert plan["twin_text"][twin_start:twin_end] == document.text[
        control["source_start"] : control["source_end"]
    ]


def test_long_document_chunks_locate_and_assign_each_control_once() -> None:
    text = LONG_TEXT.strip()
    assert count_cl100k_tokens(text) > 150
    # Spans scattered across the text, including one near the start and one
    # near the end.
    spans = [
        (0, 3, "ðə"),
        (len(text) // 3, len(text) // 3 + 5, "brʌn fɒks"),
        (2 * len(text) // 3, 2 * len(text) // 3 + 4, "ləzi dɒɡ"),
        (len(text) - 7, len(text) - 1, "hɑːvɪst"),
    ]
    document = _document(text, spans)
    controlled_text, _language, controls = _prepare_voxcpm_ipa_text(document, None, None, "en")
    assert len(controls) == len(spans)

    plans = _voxcpm_ipa_chunk_plans(controlled_text, controls, document)
    chunks = _chunk_voxcpm_texts([controlled_text])[0]
    assert len(plans) == len(chunks) > 1

    # Chunk ranges locate the chunks in the controlled text.
    ranges = chunk_source_ranges(controlled_text, chunks)
    assert all(r is not None for r in ranges)
    previous_end = 0
    for plan, chunk, (start, end) in zip(plans, chunks, ranges):
        assert plan["text"] == chunk
        assert (plan["source_start"], plan["source_end"]) <= (len(document.text), len(document.text))
        assert plan["source_start"] >= previous_end
        previous_end = plan["source_end"]
        # The chunk is the controlled slice minus dropped separators.
        iterator = iter(controlled_text[start:end])
        assert all(char in iterator for char in chunk)

    # Every control is assigned to exactly one chunk (or none), and an
    # assigned control's controlled span sits exactly inside its chunk.
    assigned: dict[int, int] = {}
    for plan_index, plan in enumerate(plans):
        for control, control_index in zip(plan["controls"], plan["control_indices"]):
            assert control_index not in assigned
            assigned[control_index] = plan_index
            global_start = ranges[plan_index][0] + control["controlled_start"]
            global_end = ranges[plan_index][0] + control["controlled_end"]
            original = controls[control_index]
            assert controlled_text[global_start:global_end] == controlled_text[
                original["controlled_start"] : original["controlled_end"]
            ]
            # Chunk-local twin: the original source span appears verbatim.
            twin_start, twin_end, ipa = plan["twin_overrides"][
                plan["controls"].index(control)
            ]
            assert ipa == original["target_ipa"]
            assert plan["twin_text"][twin_start:twin_end] == document.text[
                original["source_start"] : original["source_end"]
            ]
    assert len(assigned) + (len(controls) - len(assigned)) == len(controls)


def test_control_crossing_a_chunk_boundary_is_skipped() -> None:
    controlled_text = "alpha " * 200
    chunks = _chunk_voxcpm_texts([controlled_text])[0]
    ranges = chunk_source_ranges(controlled_text, chunks)
    assert all(r is not None for r in ranges) and len(chunks) > 1
    boundary = ranges[0][1]
    # One control crossing the first boundary, one fully inside chunk two.
    crossing = {
        "source_start": 0,
        "source_end": 5,
        "controlled_start": boundary - 3,
        "controlled_end": boundary + 3,
        "target_ipa": "ɑ",
    }
    inside = {
        "source_start": 5,
        "source_end": 11,
        "controlled_start": ranges[1][0] + 2,
        "controlled_end": ranges[1][0] + 8,
        "target_ipa": "ɛ",
    }
    document = SSMLDocument(
        text="x" * 60,
        operations=(
            PronunciationOperation(start=0, end=5, alphabet="ipa", phonemes="ɑ"),
            PronunciationOperation(start=5, end=11, alphabet="ipa", phonemes="ɛ"),
        ),
    )

    plans = _voxcpm_ipa_chunk_plans(controlled_text, [crossing, inside], document)

    assert [ci for plan in plans for ci in plan["control_indices"]] == [1]
    plan = plans[1]
    assert plan["controls"] == [
        {
            **inside,
            "controlled_start": 2,
            "controlled_end": 8,
        }
    ]
    assert plans[0]["controls"] == []


def test_twin_replaces_only_fully_contained_spellings() -> None:
    controlled_text = "alpha " * 200
    chunks = _chunk_voxcpm_texts([controlled_text])[0]
    ranges = chunk_source_ranges(controlled_text, chunks)
    assert all(r is not None for r in ranges) and len(chunks) > 1
    # A control that ends exactly at the first boundary fits chunk one.
    control = {
        "source_start": 0,
        "source_end": 4,
        "controlled_start": 0,
        "controlled_end": ranges[0][1],
        "target_ipa": "ɑːl fə",
    }
    document = SSMLDocument(
        text="original span text here",
        operations=(
            PronunciationOperation(start=0, end=4, alphabet="ipa", phonemes="ɑːl fə"),
        ),
    )

    plans = _voxcpm_ipa_chunk_plans(controlled_text, [control], document)

    assert plans[0]["control_indices"] == [0]
    assert plans[0]["controls"][0]["controlled_end"] == ranges[0][1]
    twin_start, twin_end, ipa = plans[0]["twin_overrides"][0]
    assert ipa == "ɑːl fə"
    # The control fills the whole chunk: the twin is exactly the original
    # source span (the spelling is gone, the gap after the chunk is outside
    # this chunk).
    assert (twin_start, twin_end) == (0, 4)
    assert plans[0]["twin_text"] == "orig"
