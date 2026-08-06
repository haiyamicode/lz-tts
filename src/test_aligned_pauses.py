import numpy as np
import pytest

from src.aligned_pauses import insert_aligned_pauses, parse_pause_markers


def test_parse_pause_markers_tracks_multiple_word_boundaries() -> None:
    text, markers = parse_pause_markers(
        "First we wait [pause 500ms] then continue [pause 2s] home."
    )

    assert text == "First we wait then continue home."
    assert [(marker.word_index, marker.duration_seconds) for marker in markers] == [
        (3, 0.5),
        (5, 2.0),
    ]


def test_parse_pause_markers_rejects_malformed_marker() -> None:
    with pytest.raises(ValueError, match="Malformed pause marker"):
        parse_pause_markers("Wait [pause three seconds] here")


def test_insert_aligned_pauses_adds_exact_silence_duration() -> None:
    sample_rate = 1000
    audio = np.full(2000, 0.25, dtype=np.float32)
    _, markers = parse_pause_markers("go [pause 3s] there")
    words = [
        {"text": "go", "start_seconds": 0.2, "end_seconds": 0.95},
        {"text": "there", "start_seconds": 1.05, "end_seconds": 1.8},
    ]

    output, report = insert_aligned_pauses(audio, sample_rate, markers, words)

    assert output.size == audio.size + 3000
    assert report[0]["left_word"] == "go"
    assert report[0]["right_word"] == "there"
    assert report[0]["cut_seconds"] == pytest.approx((0.95 + 1.05) / 2)
    assert report[0]["cut_strategy"] == "aligned_spoken_neighbors_midpoint"
    pause_start = round(report[0]["final_pause_start_seconds"] * sample_rate)
    assert np.all(output[pause_start : pause_start + 3000] == 0)


def test_insert_aligned_pauses_rejects_overlapping_neighbor_timestamps() -> None:
    audio = np.full(2000, 0.25, dtype=np.float32)
    _, markers = parse_pause_markers("go [pause 1s] there")
    words = [
        {"text": "go", "start_seconds": 0.2, "end_seconds": 1.05},
        {"text": "there", "start_seconds": 0.95, "end_seconds": 1.8},
    ]

    with pytest.raises(ValueError, match="timestamps overlap"):
        insert_aligned_pauses(audio, 1000, markers, words)


def test_insert_aligned_pauses_maps_japanese_character_segments_from_source() -> None:
    sample_rate = 1000
    audio = np.full(4000, 0.25, dtype=np.float32)
    _, markers = parse_pause_markers("準備ができました。[pause 1s] それでは始めましょう。")
    characters = list("準備ができました。") + [" "] + list("それでは始めましょう。")
    words = [
        {
            "text": character,
            "start_seconds": 0.2 * index,
            "end_seconds": 0.2 * (index + 1),
        }
        for index, character in enumerate(characters)
    ]
    words[8]["end_seconds"] = 1.8
    words[9]["start_seconds"] = 1.8
    words[9]["end_seconds"] = 1.9
    words[10]["start_seconds"] = 2.1

    _, report = insert_aligned_pauses(audio, sample_rate, markers, words)

    assert markers[0].word_index == 1
    assert report[0]["source_word_index"] == 1
    assert report[0]["alignment_word_index"] == 10
    assert report[0]["left_word"] == "た"
    assert report[0]["right_word"] == "そ"
    assert report[0]["cut_seconds"] == pytest.approx((1.6 + 2.1) / 2)


def test_insert_aligned_pauses_handles_bos_and_eos_without_midpoints() -> None:
    sample_rate = 1000
    audio = np.full(2000, 0.25, dtype=np.float32)
    _, markers = parse_pause_markers("[pause 1s] hello world [pause 2s]")
    words = [
        {"text": "hello", "start_seconds": 0.2, "end_seconds": 0.8},
        {"text": "world", "start_seconds": 1.0, "end_seconds": 1.7},
    ]

    output, report = insert_aligned_pauses(audio, sample_rate, markers, words)

    assert output.size == 5000
    assert [entry["cut_strategy"] for entry in report] == ["audio_start", "audio_end"]
    assert [entry["cut_seconds"] for entry in report] == [0.0, 2.0]
    assert np.all(output[:1000] == 0)
    assert np.all(output[-2000:] == 0)


def test_insert_aligned_pauses_combines_consecutive_markers() -> None:
    audio = np.full(2000, 0.25, dtype=np.float32)
    _, markers = parse_pause_markers("go [pause 1s] [pause 500ms] there")
    words = [
        {"text": "go", "start_seconds": 0.2, "end_seconds": 0.9},
        {"text": "there", "start_seconds": 1.1, "end_seconds": 1.8},
    ]

    output, report = insert_aligned_pauses(audio, 1000, markers, words)

    assert output.size == 3500
    assert len(report) == 1
    assert report[0]["duration_seconds"] == pytest.approx(1.5)
    assert report[0]["marker_count"] == 2
