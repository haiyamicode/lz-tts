from __future__ import annotations

import numpy as np

from .ssml import BreakOperation, PronunciationOperation
from .ssml import parse_ssml
from .ssml_postprocessing import aligned_text_interval, insert_ssml_breaks, splice_pronunciations


TIMESTAMPS = [
    {"text": "hello ", "start_seconds": 0.1, "end_seconds": 0.4, "source_start": 0, "source_end": 6},
    {"text": "Saoirse ", "start_seconds": 0.6, "end_seconds": 1.0, "source_start": 6, "source_end": 14},
    {"text": "left", "start_seconds": 1.2, "end_seconds": 1.5, "source_start": 14, "source_end": 18},
]


def test_aligned_text_interval_uses_neighbor_midpoints() -> None:
    interval = aligned_text_interval(
        PronunciationOperation(6, 13, "ipa", "sˈɜːɹʃə"),
        TIMESTAMPS,
        audio_samples=1600,
        sample_rate=1000,
    )
    assert (interval.start_sample, interval.end_sample) == (500, 1100)
    assert interval.alignment_indices == (1,)


def test_splice_pronunciation_uses_separately_aligned_context_render() -> None:
    baseline = np.zeros(1600, dtype=np.float32)
    replacement = np.ones(1800, dtype=np.float32)
    replacement_timestamps = [
        {**TIMESTAMPS[0], "end_seconds": 0.5},
        {**TIMESTAMPS[1], "start_seconds": 0.7, "end_seconds": 1.2},
        {**TIMESTAMPS[2], "start_seconds": 1.4, "end_seconds": 1.7},
    ]
    output, report = splice_pronunciations(
        baseline,
        [replacement],
        1000,
        [PronunciationOperation(6, 13, "ipa", "sˈɜːɹʃə")],
        TIMESTAMPS,
        [replacement_timestamps],
        crossfade_seconds=0,
    )
    assert output.size == 1700
    assert np.all(output[500:1200] == 1)
    assert report[0]["baseline_start_seconds"] == 0.5
    assert report[0]["replacement_start_seconds"] == 0.6


def test_ssml_breaks_use_source_prefix_and_support_bos_eos() -> None:
    audio = np.ones(1600, dtype=np.float32)
    output, report = insert_ssml_breaks(
        "hello Saoirse left",
        audio,
        1000,
        [BreakOperation(0, 0.2), BreakOperation(13, 0.3), BreakOperation(18, 0.4)],
        TIMESTAMPS,
    )
    assert output.size == 2500
    assert [item["cut_strategy"] for item in report] == [
        "audio_start",
        "aligned_spoken_neighbors_midpoint",
        "audio_end",
    ]


def test_adjacent_ssml_words_map_to_distinct_alignment_units() -> None:
    document = parse_ssml('<speak>Hello<break time="1s"/>world</speak>')
    output, report = insert_ssml_breaks(
        document.text,
        np.ones(1000, dtype=np.float32),
        1000,
        document.breaks,
        [
            {
                "text": "Hello ",
                "start_seconds": 0.1,
                "end_seconds": 0.4,
                "source_start": 0,
                "source_end": 6,
            },
            {
                "text": "world",
                "start_seconds": 0.6,
                "end_seconds": 0.9,
                "source_start": 6,
                "source_end": 11,
            },
        ],
    )

    assert output.size == 2000
    assert report[0]["left_word"] == "Hello "
    assert report[0]["right_word"] == "world"
    assert report[0]["cut_seconds"] == 0.5
