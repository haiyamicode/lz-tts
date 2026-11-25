"""Videogen character swap: a person in a source video replaced by the character in a picture.

akatz's H3 Character Swap V1.5 run with the official h3-relay toolkit graph (``H3-Character-Swap.api.json``,
h3-relay e2a7665, copied unchanged as ``character_swap.api.json``), as tested on a 5 s dance clip on
2026-11-24 (``local/exp/h3-charswap/run_dance.py``): Ref2VA int8, our Ref2V turbo-8 v1.1 LoRA then V1.5 at
1.0, er_sde 8 steps CFG 1, the akatz prompt. SAM 3.1 masks the "person" (largest, tracked through the
clip, widened 64 px) and only that area is regenerated; the scene, camera and audio stay the source's.
The source is prepared at 768 (~1 MP, 24 fps) and rendered in 124-frame windows, each later one taking
the last 18 frames of the one before as history. Changes from the toolkit file: the int4 Qwen encoder
(the nvfp4 one needs Blackwell), the turbo and adapter file names, the job's files, seed and duration,
and the mask preview node dropped (it only writes PNGs for the web UI).

A ``videogen`` job with the addon ``{"name": "character_swap", "source_video_url" | "source_video_path"}`` in
its ``addons`` (:func:`h3.videogen.run_videogen` sends it here): the video whose person is replaced
(<= ``MAX_SOURCE_SECONDS`` of it is rendered). From the job's payload:

- ``image_url`` / ``image_path``: the character to put in;
- ``duration_seconds``: how much of the source, from its start (default: all of it; a source longer than
  ``MAX_SOURCE_SECONDS`` needs it);
- ``seed`` (random when omitted): the window plan's seed;
- ``prompt`` (optional): the swap's instruction, default the toolkit's ``DEFAULT_PROMPT``.

Nothing else applies (preset, sizing, pins, other addons); the API rejects them. The clip comes out at the
render's size with the source's audio. Cost on the dev 3090: 944 s for 5 s at 1344x736 (one window),
8.5 GiB peak allocated.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import cancel
import progress

from .compat import job_dir, log_event
from .lipsync import _probe, download_job_input

GRAPH_PATH = Path(__file__).with_name("character_swap.api.json")
ADAPTER_LORA_NAME = "h3_character_swap_v1.5.safetensors"
SAM_CHECKPOINT_NAME = "sam3.1_multiplex_fp16.safetensors"
#: The longest source a job renders: 6 windows, ~1.5 h on the 3090.
MAX_SOURCE_SECONDS = 30.0
#: The toolkit's window plan: the first window covers this many frames, each later one adds one fewer
#: (its first frame is the previous window's last).
WINDOW_FRAMES = 124
FPS = 24
SAVE_NODE_ID = "save"
#: The toolkit's swap instruction (``reference_conditioning``), what a request without ``prompt`` renders.
DEFAULT_PROMPT = ("Replace the selected person with the reference character. Preserve source performance, scene "
                  "and audio timing.")
#: Nodes the swap needs from the h3-relay pack (data/comfyui/custom_nodes/h3-relay).
RELAY_NODES = ("H3RelayPrepareSource", "H3RelayWindowPlan", "H3RelayWindowInputs", "H3RelayFinishWindows")


def window_count(frames: int) -> int:
    """Windows the toolkit plan (``planning.plan_windows``, 124 / 18) cuts ``frames`` source frames into."""
    return 1 + -(-max(0, frames - WINDOW_FRAMES) // (WINDOW_FRAMES - 1))


def build_graph(*, source_input: str, reference_input: str, seed: int, duration_seconds: float,
                filename_prefix: str, prompt: str | None = None) -> dict[str, Any]:
    from .minimax_h3_runtime import TURBO8_LORA_NAME, clip_name

    graph = json.loads(GRAPH_PATH.read_text())
    graph["text"]["inputs"]["clip_name"] = clip_name()
    graph["turbo"]["inputs"]["lora_name"] = TURBO8_LORA_NAME
    graph["adapter"]["inputs"]["lora_name"] = ADAPTER_LORA_NAME
    graph["source"]["inputs"]["file"] = source_input
    graph["reference"]["inputs"]["image"] = reference_input
    graph["prepare"]["inputs"]["duration_seconds"] = float(duration_seconds)
    graph["plan"]["inputs"]["seed"] = int(seed)
    if prompt:
        graph["reference_conditioning"]["inputs"]["prompt"] = prompt
    graph["save"]["inputs"]["filename_prefix"] = filename_prefix
    del graph["mask_preview"]
    return graph


def _model_paths(comfy_root: Path, graph: dict[str, Any]) -> list[Path]:
    models = comfy_root / "models"
    return [
        models / "diffusion_models" / graph["model"]["inputs"]["unet_name"],
        models / "text_encoders" / graph["text"]["inputs"]["clip_name"],
        models / "vae" / graph["video_vae"]["inputs"]["vae_name"],
        models / "vae" / graph["audio_vae"]["inputs"]["vae_name"],
        models / "loras" / graph["turbo"]["inputs"]["lora_name"],
        models / "loras" / graph["adapter"]["inputs"]["lora_name"],
        models / "checkpoints" / graph["sam"]["inputs"]["ckpt_name"],
    ]


class _WindowProgress:
    """The sampler's steps as the job's progress, a window at a time.

    Each window is one 8-step sampling of the loop body's ``sample`` node (expanded as ``<i>_sample`` per
    iteration); a new sampling node is the next window. Other progress bars (SAM tracking, decodes) are
    ignored.
    """

    def __init__(self, windows: int):
        self.windows = windows
        self.window = -1
        self.node: str | None = None

    def __call__(self, value, total, preview_image=None, prompt_id=None, node_id=None) -> None:
        if cancel.is_cancelled():
            from comfy.model_management import InterruptProcessingException

            raise InterruptProcessingException()
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        node = str(context.node_id) if context is not None else str(node_id or "")
        if not total or not (node == "sample" or node.endswith("_sample")):
            return
        if node != self.node:
            self.node = node
            self.window = min(self.window + 1, self.windows - 1)
            progress.report("rendering", self.window, self.windows)
        progress.step(int(value), int(total))


def run_character_swap(job: dict[str, Any], addon: dict[str, Any]) -> dict[str, Any]:
    """Render the swap of ``job`` (a videogen job) with its ``character_swap`` ``addon`` entry."""
    from .minimax_h3_runtime import _execute_graph, _report_sampling_progress, _retained_runtime, cleanup_h3_job_files

    payload = job["payload"]
    job_id = str(job["id"])
    if not (payload.get("image_url") or payload.get("image_path")):
        raise ValueError("the character_swap addon needs image_url: the character to put in")
    seed = int(payload["seed"]) if payload.get("seed") is not None else random.randint(0, 2**31 - 1)
    scratch_dir = job_dir(job_id)
    progress.report("preparing")
    prompt = str(payload.get("prompt") or "").strip() or None
    source_path = download_job_input(job, addon, "source_video", scratch_dir, ".mp4", kind="videogen")
    image_path = download_job_input(job, payload, "image", scratch_dir, ".png", kind="videogen")
    probe = _probe(source_path)
    if not any(stream.get("codec_type") == "video" for stream in probe.get("streams", [])):
        raise ValueError("source_video has no video stream")
    source_seconds = float(probe.get("format", {}).get("duration") or 0.0)
    if source_seconds <= 0:
        raise ValueError("source_video has no duration")
    if payload.get("duration_seconds"):
        duration = float(payload["duration_seconds"])
        if not 0 < duration <= MAX_SOURCE_SECONDS:
            raise ValueError(f"duration_seconds must be in (0, {MAX_SOURCE_SECONDS:g}] for the character_swap addon")
        duration = min(duration, source_seconds)
    elif source_seconds > MAX_SOURCE_SECONDS:
        raise ValueError(f"the source video lasts {source_seconds:.2f} s, over the swap's {MAX_SOURCE_SECONDS:g} s; "
                         "pass duration_seconds to swap its start")
    else:
        duration = 0.0  # the toolkit's "whole source"
    windows = window_count(round((duration or source_seconds) * FPS))

    retained = _retained_runtime(generate_audio=True)
    comfy_root = retained.comfy_root
    import nodes as comfy_nodes

    if missing := [name for name in RELAY_NODES if name not in comfy_nodes.NODE_CLASS_MAPPINGS]:
        raise FileNotFoundError(f"h3-relay nodes are not registered: {missing} (expected "
                                f"{comfy_root / 'custom_nodes' / 'h3-relay'}; check the ComfyUI 'IMPORT FAILED' lines)")
    job_input_dir = comfy_root / "input" / "jobs" / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    source_input = f"jobs/{job_id}/source{source_path.suffix or '.mp4'}"
    reference_input = f"jobs/{job_id}/reference{image_path.suffix or '.png'}"
    shutil.copyfile(source_path, comfy_root / "input" / source_input)
    shutil.copyfile(image_path, comfy_root / "input" / reference_input)
    graph = build_graph(source_input=source_input, reference_input=reference_input, seed=seed,
                        duration_seconds=duration, filename_prefix=f"jobs/{job_id}/output", prompt=prompt)
    if missing := [str(path) for path in _model_paths(comfy_root, graph) if not path.exists()]:
        raise FileNotFoundError("character swap weights are missing: " + ", ".join(missing))

    output_path = scratch_dir / "outputs" / "output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_event("character_swap_started", job_id=job_id, seed=seed, source_seconds=round(source_seconds, 3),
              duration_seconds=duration or None, windows=windows)
    started = time.perf_counter()
    import comfy.utils

    try:
        comfy.utils.set_progress_bar_global_hook(_WindowProgress(windows))
        with progress.chunk_steps():
            saved = _execute_graph(retained, graph, job_id=job_id, save_node_id=SAVE_NODE_ID)
        progress.report("finishing")
        saved_path = comfy_root / "output" / saved["saved_subfolder"] / saved["saved_filename"]
        shutil.move(str(saved_path), output_path)
    finally:
        comfy.utils.set_progress_bar_global_hook(_report_sampling_progress)
        cleanup_h3_job_files(job_id)
    wall_seconds = time.perf_counter() - started
    video = next(s for s in _probe(output_path)["streams"] if s.get("codec_type") == "video")
    item = {
        "mode": "character_swap",
        "seed": seed,
        "fps": float(FPS),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frames": int(video.get("nb_frames") or 0),
        "duration_seconds": round(float(video.get("duration") or 0.0), 3),
        "source_seconds": round(source_seconds, 3),
        "windows": windows,
        "steps": int(graph["schedule"]["inputs"]["steps"]),
        "loras": [graph["turbo"]["inputs"]["lora_name"], graph["adapter"]["inputs"]["lora_name"]],
        "prompt": graph["reference_conditioning"]["inputs"]["prompt"],
    }
    metrics = {
        "engine": "comfyui_h3",
        "wall_seconds": round(wall_seconds, 3),
        "pipeline_seconds": round(saved["pipeline_seconds"], 3),
        "cuda_peak_allocated_gib": round(saved["cuda_peak_allocated_gib"], 2),
        "cuda_peak_reserved_gib": round(saved["cuda_peak_reserved_gib"], 2),
    }
    log_event("character_swap_complete", job_id=job_id, **metrics)
    return {"path": str(output_path), "item": item, "metrics": metrics, "seed": seed}
