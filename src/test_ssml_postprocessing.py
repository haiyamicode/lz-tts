from __future__ import annotations

import numpy as np

from .ssml import BreakOperation
from .ssml import parse_ssml
from .ssml_postprocessing import insert_ssml_breaks


TIMESTAMPS = [
    {"text": "hello ", "start_seconds": 0.1, "end_seconds": 0.4, "source_start": 0, "source_end": 6},
    {"text": "Saoirse ", "start_seconds": 0.6, "end_seconds": 1.0, "source_start": 6, "source_end": 14},
    {"text": "left", "start_seconds": 1.2, "end_seconds": 1.5, "source_start": 14, "source_end": 18},
]


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
