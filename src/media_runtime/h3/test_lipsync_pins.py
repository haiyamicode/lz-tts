"""Lipsync ``pinned_frames``: which frames are pinned, and to which image, in each mode."""

from __future__ import annotations

import numpy as np
import pytest

from h3.lipsync import (
    GENERATED_IMAGE,
    INPUT_IMAGE,
    RESTORED_IMAGE,
    FramePin,
    PinnedSpan,
    PinSettings,
    frame_pins,
    freestyle_plan,
    plan_anchors,
    plan_chunks,
)

FPS, SECONDS = 24.0, 30
TOTAL = int(FPS * SECONDS)


def _speech(seconds: float):
    """16 kHz speech-like bursts with a quiet gap every 2 s, so the planner has pauses to choose."""
    rate = 16000
    t = np.arange(int(seconds * rate)) / rate
    loud = (t % 2.0) < 1.7
    rng = np.random.default_rng(1)
    return (rng.normal(0, 0.2, t.size) * loud + rng.normal(0, 0.001, t.size)).astype(np.float32)


def _plan(settings: PinSettings, *, seed: int = 0, pinned=()):
    fixed = frame_pins(settings, fps=FPS, total_frames=TOTAL, pinned=list(pinned))
    cuts, pins = freestyle_plan(_speech(SECONDS), total_frames=TOTAL, fps=FPS, chunk_cap=226, pinned=list(pinned),
                                continuous=settings.continuous, fixed=fixed)
    anchors = plan_anchors(pins, list(pinned), continuous=settings.continuous, fixed=fixed,
                           images=settings.images or (INPUT_IMAGE,), order=settings.order, seed=seed)
    return cuts, anchors


def test_none_pins_frame_zero_and_restored_boundaries_only():
    cuts, anchors = _plan(PinSettings.from_payload(None))
    assert cuts
    assert [(a.frame, a.label) for a in anchors] == [(0, "start")] + [(cut, "boundary") for cut in cuts]
    assert all(a.image == f"{RESTORED_IMAGE}{a.frame}" for a in anchors[1:])


def test_single_image_pins_every_boundary_and_pause_to_its_image():
    default_cuts, default = _plan(PinSettings.from_payload({"mode": "single_image"}))
    cuts, anchors = _plan(PinSettings.from_payload({"mode": "single_image", "image_url": "http://x/rest.png"}))
    assert cuts == default_cuts
    assert {a.image for a in default} == {INPUT_IMAGE}
    # The same frames, the boundaries, the pauses between them and the last frame, on the caller's image.
    assert [a.frame for a in anchors] == [a.frame for a in default]
    assert set(cuts) < {a.frame for a in anchors} and TOTAL - 1 in {a.frame for a in anchors}
    assert len(anchors) > len(cuts) + 2
    assert anchors[0].image == INPUT_IMAGE and {a.image for a in anchors[1:]} == {"http://x/rest.png"}


def test_multi_image_round_robin_takes_the_images_in_turn():
    images = ["http://x/1.png", "http://x/2.png", "http://x/3.png"]
    _, single = _plan(PinSettings.from_payload({"mode": "single_image"}))
    _, anchors = _plan(PinSettings.from_payload({"mode": "multi_image", "image_urls": images}))
    assert [a.frame for a in anchors] == [a.frame for a in single]
    assert [a.image for a in anchors[1:]] == [images[i % 3] for i in range(len(anchors) - 1)]


def test_multi_image_random_is_seeded_and_never_repeats_an_image_twice_running():
    payload = {"mode": "multi_image", "image_urls": ["http://x/1.png", "http://x/2.png", "http://x/3.png"],
               "order": "random"}
    picks = [a.image for a in _plan(PinSettings.from_payload(payload), seed=7)[1][1:]]
    assert picks == [a.image for a in _plan(PinSettings.from_payload(payload), seed=7)[1][1:]]
    assert picks != [a.image for a in _plan(PinSettings.from_payload(payload), seed=8)[1][1:]]
    assert all(a != b for a, b in zip(picks, picks[1:]))
    assert set(picks) == set(payload["image_urls"])


def test_generate_pins_its_stills_like_multi_image():
    settings = PinSettings.from_payload({"mode": "generate", "count": 4, "order": "random", "max_retries": 0})
    assert settings.images == tuple(f"{GENERATED_IMAGE}{k}" for k in range(4))
    assert (settings.order, settings.max_retries, settings.urls) == ("random", 0, [])
    _, single = _plan(PinSettings.from_payload({"mode": "single_image"}))
    _, anchors = _plan(settings, seed=7)
    assert [a.frame for a in anchors] == [a.frame for a in single]
    assert {a.image for a in anchors[1:]} == set(settings.images)
    default = PinSettings.from_payload({"mode": "generate"})
    assert (len(default.images), default.order, default.max_retries) == (6, "round_robin", 2)


def test_timed_images_pin_only_the_callers_frames_and_cuts_land_on_them():
    # 7.0 s and 16.0 s are mid-speech (the quiet gaps are 1.7-2.0 s into every 2 s): the quietest cut
    # would be in a gap, but a cut in reach of a pin goes onto it.
    frames = [{"image_url": "http://x/a.png", "seconds": 7.0}, {"image_url": "http://x/b.png", "seconds": 16.0},
              {"image_url": "http://x/c.png", "seconds": 25.5}]
    cuts, anchors = _plan(PinSettings.from_payload({"mode": "timed_images", "frames": frames}))
    by_frame = {a.frame: a for a in anchors}
    assert {168, 384} <= set(cuts)
    for frame, url in ((168, "http://x/a.png"), (384, "http://x/b.png"), (612, "http://x/c.png")):
        assert (by_frame[frame].label, by_frame[frame].image) == ("frame", url)
    # Nothing else but frame 0 and the restored boundaries of cuts no pin was in reach of.
    assert {a.label for a in anchors} <= {"start", "frame", "boundary"}
    assert all(a.frame in cuts for a in anchors if a.label == "boundary")
    chunks = plan_chunks(TOTAL, max_frames=226, anchors=anchors, cut_frames=cuts)
    assert chunks[-1].last_frame == TOTAL - 1 and TOTAL - 1 not in by_frame


def test_timed_images_keep_pinned_chunks():
    span = PinnedSpan(first=240, last=360, image="http://x/span.png")
    settings = PinSettings.from_payload({"mode": "timed_images", "frames": [
        {"image_url": "http://x/span.png", "seconds": 10.0},  # on the span's start, with its image: that end
        {"image_url": "http://x/a.png", "seconds": 12.5},  # inside the span: a guide there
    ]})
    cuts, anchors = _plan(settings, pinned=[span])
    by_frame = {a.frame: a for a in anchors}
    assert {240, 360} <= set(cuts)
    assert (by_frame[240].label, by_frame[360].label, by_frame[300].label) == ("pin", "pin", "frame")


@pytest.mark.parametrize("seconds, error", [
    (0.01, "frame 0"),
    (SECONDS + 1, "past the end"),
])
def test_timed_images_reject_frames_outside_the_video(seconds, error):
    settings = PinSettings.from_payload({"mode": "timed_images", "frames": [{"image_url": "u", "seconds": seconds}]})
    with pytest.raises(ValueError, match=error):
        frame_pins(settings, fps=FPS, total_frames=TOTAL)


def test_timed_images_reject_two_pins_on_one_frame_and_a_clash_with_a_pinned_chunk():
    two = PinSettings.from_payload({"mode": "timed_images", "frames": [
        {"image_url": "a", "seconds": 5.0}, {"image_url": "b", "seconds": 5.01}]})
    with pytest.raises(ValueError, match="same frame"):
        frame_pins(two, fps=FPS, total_frames=TOTAL)
    clash = PinSettings.from_payload({"mode": "timed_images", "frames": [{"image_url": "b", "seconds": 10.0}]})
    with pytest.raises(ValueError, match="pinned chunk's end"):
        frame_pins(clash, fps=FPS, total_frames=TOTAL, pinned=[PinnedSpan(first=240, last=300, image="a")])
    assert frame_pins(clash, fps=16.0, total_frames=TOTAL) == [FramePin(frame=160, image="b")]


@pytest.mark.parametrize("payload, error", [
    ({"mode": "solo"}, "mode must be"),
    ({"mode": "none", "image_url": "u"}, "takes no image_url"),
    ({"mode": "single_image", "image_urls": ["a", "b"]}, "takes no image_urls"),
    ({"mode": "multi_image", "image_urls": ["a"]}, "2 to 16"),
    ({"mode": "multi_image", "image_urls": ["a", "b"], "order": "shuffle"}, "order must be"),
    ({"mode": "timed_images"}, "1 to 100"),
    ({"mode": "timed_images", "frames": [{"image_url": "a", "seconds": 1}], "order": "random"}, "takes no order"),
    ({"mode": "generate", "count": 1}, "count of 2 to 12"),
    ({"mode": "generate", "count": 13}, "count of 2 to 12"),
    ({"mode": "generate", "max_retries": 6}, "max_retries"),
    ({"mode": "generate", "image_urls": ["a", "b"]}, "takes no image_urls"),
    ({"mode": "multi_image", "image_urls": ["a", "b"], "count": 3}, "takes no count"),
])
def test_pin_settings_reject_what_their_mode_does_not_take(payload, error):
    with pytest.raises(ValueError, match=error):
        PinSettings.from_payload(payload)
