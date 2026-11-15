"""Clean chunk joins: latent indexing, the preview graph, the pre-clean and the continuous lipsync plan.

The GPU passes themselves are checked end to end on a worker, not here.
"""

from __future__ import annotations

import numpy as np

from h3 import h3_restore
from h3.lipsync import RESTORED_IMAGE, PinnedSpan, freestyle_plan, plan_anchors, plan_chunks
from h3.minimax_h3_runtime import LIPSYNC_PRESETS, VIDEO_PRESETS, build_h3_graph, build_lipsync_graph


def test_pixel_frames_map_onto_the_packed_latents():
    # 175 frames pack into 52 latents; the last kept frame of a 170-frame chunk (169) is latent 50,
    # the one vigen measured decoding closest to the real frame (25 dB vs ~14 dB for its neighbours).
    assert h3_restore.latent_index(169, 175) == 50
    assert h3_restore.latent_index(0, 175) == 0
    assert h3_restore.latent_index(174, 175) == 51
    assert h3_restore.latent_index(500, 175) == 51
    # The shortest clip: 5 frames, 2 latents.
    assert h3_restore.latent_index(4, 5) == 1


def _render_graph(recipe):
    return build_h3_graph(
        prompt="p", seed=1, width=512, height=768, length=73, first_frame_input="jobs/j/first.png",
        last_frame_input=None, filename_prefix="jobs/j/output", recipe=recipe, generate_audio=True,
    )


def test_preview_graph_decodes_one_frame_on_the_preview_schedule():
    graph = _render_graph(VIDEO_PRESETS["turbo-4"])
    sampled_by = graph["13"]["inputs"]["sigmas"]
    save = h3_restore.preview_graph(graph, frame=72, length=73, filename_prefix="jobs/j/preview",
                                    sigmas=VIDEO_PRESETS["turbo-4"].preview_sigmas)
    assert "17" not in graph  # no video output: the full decode and the mp4 never run
    assert sampled_by == ["9", 0]
    assert graph["9"] == {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.8575, 0.7064, 0.0"}}
    assert graph["94"]["class_type"] == h3_restore.DECODE_FRAME_NODE
    assert graph["94"]["inputs"] == {"samples": ["13", 0], "vae": ["4", 0], "latent_index": 21}
    assert graph[save]["inputs"]["images"] == ["94", 0]


def test_unmeasured_recipes_preview_on_their_own_schedule():
    for name in ("standard", "turbo-8"):
        recipe = VIDEO_PRESETS[name]
        assert recipe.preview_sigmas is None
        graph = _render_graph(recipe)
        scheduler = dict(graph["9"])
        h3_restore.preview_graph(graph, frame=72, length=73, filename_prefix="p", sigmas=recipe.preview_sigmas)
        assert graph["9"] == scheduler


def test_lipsync_student_previews_in_one_step():
    recipe = LIPSYNC_PRESETS["turbo-3"]
    graph = build_lipsync_graph(
        prompt="p", seed=1, width=512, height=768, length=73, first_frame_input="f.png", last_frame_input=None,
        guides=[], audio_input="a.wav", filename_prefix="o", recipe=recipe, sparse_attention=False,
    )
    h3_restore.preview_graph(graph, frame=72, length=73, filename_prefix="p", sigmas=recipe.preview_sigmas)
    assert graph["9"]["inputs"]["sigmas"] == "1.0, 0.0"


def test_preclean_matches_the_anchor_colour_and_flattens_speckle(tmp_path):
    import cv2

    rng = np.random.default_rng(0)
    anchor = np.full((64, 64, 3), (110, 120, 130), np.uint8)
    drifted = np.clip(np.full((64, 64, 3), (100, 110, 170), np.float32) + rng.normal(0, 12, (64, 64, 3)), 0, 255)
    paths = [tmp_path / name for name in ("anchor.png", "drifted.png", "clean.png")]
    cv2.imwrite(str(paths[0]), anchor)
    cv2.imwrite(str(paths[1]), drifted.astype(np.uint8))
    cleaned = cv2.imread(str(h3_restore.preclean(paths[1], paths[0], paths[2]))).astype(np.float32)
    assert np.abs(cleaned.reshape(-1, 3).mean(0) - anchor.reshape(-1, 3).mean(0)).max() < 3.0
    assert cleaned.std(axis=(0, 1)).max() < drifted.std(axis=(0, 1)).max() / 3


def _speech(seconds: float, fps: float = 24.0):
    """16 kHz speech-like bursts with a quiet gap every 2 s, so the planner has pauses to choose."""
    rate = 16000
    t = np.arange(int(seconds * rate)) / rate
    loud = (t % 2.0) < 1.7
    rng = np.random.default_rng(1)
    return (rng.normal(0, 0.2, t.size) * loud + rng.normal(0, 0.001, t.size)).astype(np.float32)


def test_continuous_boundaries_pin_restored_frames_and_no_pauses():
    fps, total = 24.0, 24 * 20
    samples = _speech(20)
    cuts, pins = freestyle_plan(samples, total_frames=total, fps=fps, chunk_cap=226)
    c_cuts, c_pins = freestyle_plan(samples, total_frames=total, fps=fps, chunk_cap=226, continuous=True)
    assert c_cuts == cuts and cuts
    # Input-image boundaries pin the cuts, the pauses and the last frame; continuous only frame 0 and the cuts.
    assert c_pins == [0, *cuts]
    assert total - 1 in pins and len(pins) > len(c_pins)

    anchors = plan_anchors(c_pins, [], continuous=True)
    assert [(a.label, a.image) for a in anchors] == [("start", "input")] + [
        ("boundary", f"{RESTORED_IMAGE}{cut}") for cut in cuts
    ]
    chunks = plan_chunks(total, max_frames=226, anchors=anchors, cut_frames=c_cuts)
    # Every chunk but the last ends on its restored boundary; the last ends free.
    for chunk in chunks[:-1]:
        assert chunk.guides[-1] == (chunk.frames - 1, next(a for a in anchors if a.frame == chunk.last_frame))
    assert chunks[-1].guides == ()


def test_continuous_keeps_pinned_chunk_images():
    fps, total = 24.0, 24 * 20
    span = PinnedSpan(first=96, last=192, image="https://example/pin.png")
    _, pins = freestyle_plan(_speech(20), total_frames=total, fps=fps, chunk_cap=226, pinned=[span], continuous=True)
    anchors = {a.frame: a for a in plan_anchors(pins, [span], continuous=True)}
    assert (anchors[96].label, anchors[96].image) == ("pin", span.image)
    assert (anchors[192].label, anchors[192].image) == ("pin", span.image)
    assert all(a.label in {"start", "pin", "boundary"} for a in anchors.values())
