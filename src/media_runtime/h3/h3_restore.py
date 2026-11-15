"""Clean chunk joins for H3: restore a drifted frame toward an anchor image (port of vigen e9791cc).

Chained H3 renders drift: a clip that starts on another clip's last frame copies that frame's damage
(speckle, blotchy colour, painterly skin) and adds its own, so the look degrades clip by clip. The fix,
used by lipsync's ``continuous`` chunk boundaries and videogen's ``continuable`` clips:

1. a preview of the clip's last frame: the same graph, sampled on a short schedule, only that frame's
   latent decoded (:func:`preview_graph`);
2. that frame restored (:func:`restore_frame`): a classical pre-clean (colour statistics matched to the
   anchor, non-local-means denoise), then an H3 still img2img pass on the base model with the anchor as
   ``<Picture 1>``: the look comes from the anchor, the pose from the frame;
3. the real render pinned to the restored frame at its last frame; the next clip opens on it.

The drifted frame never enters the conditioning: as ``<Picture 1>`` H3 reproduces its damage at any
denoise strength. Text-only conditioning loses the person (vigen local/experiments/h3-derift-20261113).
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compat import log_event

#: img2img strength and steps of the restore pass (the tail of a ``steps / denoise`` schedule).
RESTORE_DENOISE = 0.4
RESTORE_STEPS = 20
RESTORE_PROMPT = (
    "The same person from <Picture 1>, in the same place, with the same camera and light. Their face, skin, "
    "hair, clothes and the background look exactly like in <Picture 1>. A real, natural photograph: natural "
    "skin texture with real pores, a natural skin tone, true neutral colors, a clean background, no noise, no "
    "color spots, not painted, not airbrushed, not over-sharpened."
)
RESTORE_NODE = "VigenH3StillRestore"
DECODE_FRAME_NODE = "VigenH3DecodeFrame"
_PREVIEW_DECODE_NODE_ID = "94"
_PREVIEW_SAVE_NODE_ID = "95"


@dataclass(frozen=True)
class FramePreview:
    """Render only ``frame`` (an index into the rendered clip) to the PNG ``output_path``.

    ``sigmas`` (``ManualSigmas``) replace the graph's schedule; ``None`` keeps the recipe's own.
    """

    frame: int
    output_path: Path
    sigmas: str | None = None


def latent_index(frame: int, length: int) -> int:
    """The latent frame holding pixel ``frame`` of a ``length``-frame (17k+5) H3 clip.

    Linear over the packed axis; vigen checked it on a 175-frame render: pixel 169 -> latent 50 of 52
    decodes at 25 dB PSNR against the real decode, the neighbouring latents at ~14 dB.
    """
    latents = 5 * (length - 5) // 17 + 2
    return min(latents - 1, max(0, round(frame * (latents - 1) / max(1, length - 1))))


def preview_graph(
    graph: dict[str, dict[str, Any]],
    *,
    frame: int,
    length: int,
    filename_prefix: str,
    sigmas: str | None,
    sampler_node: str = "13",
    vae_node: str = "4",
    sigmas_node: str = "9",
    video_save_node: str = "17",
) -> str:
    """Turn a render graph into its one-frame preview in place; returns the preview's save node id.

    No video/audio decode or mp4: the sampled latent's frame for ``frame`` goes through the still
    decoder only. ``sigmas`` replace the schedule node; ``None`` keeps it.
    """
    graph.pop(video_save_node)
    if sigmas is not None:
        graph[sigmas_node] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}}
    graph[_PREVIEW_DECODE_NODE_ID] = {
        "class_type": DECODE_FRAME_NODE,
        "inputs": {"samples": [sampler_node, 0], "vae": [vae_node, 0], "latent_index": latent_index(frame, length)},
    }
    graph[_PREVIEW_SAVE_NODE_ID] = {
        "class_type": "SaveImage",
        "inputs": {"images": [_PREVIEW_DECODE_NODE_ID, 0], "filename_prefix": filename_prefix},
    }
    return _PREVIEW_SAVE_NODE_ID


class VigenH3StillRestore:
    """img2img on one H3 still: ``reference`` conditions (<Picture 1>), ``init`` seeds the latent."""

    CATEGORY = "lzai"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 (ComfyUI node API)
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True}),
                "reference": ("IMAGE",),
                "init": ("IMAGE",),
                "width": ("INT", {"default": 768, "min": 32, "max": 4096}),
                "height": ("INT", {"default": 1344, "min": 32, "max": 4096}),
                "denoise": ("FLOAT", {"default": RESTORE_DENOISE, "min": 0.01, "max": 1.0}),
                "steps": ("INT", {"default": RESTORE_STEPS, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def run(self, model, clip, vae, prompt, reference, init, width, height, denoise, steps, seed):
        import comfy.model_management
        import comfy.nested_tensor
        import comfy.samplers
        import comfy.utils
        import torch
        from comfy_extras.nodes_custom_sampler import BasicGuider, Noise_RandomNoise, SamplerCustomAdvanced

        from . import h3_still

        positive, _ = h3_still._prepare(clip, vae, prompt, int(width), int(height), [reference])
        source = comfy.utils.common_upscale(
            init[:1, :, :, :3].movedim(-1, 1), int(width), int(height), "lanczos", "disabled"
        ).movedim(1, -1)
        device = comfy.model_management.intermediate_device()
        latent = {
            "samples": comfy.nested_tensor.NestedTensor(
                (vae.encode(source).to(device), torch.zeros((1, 32, 2, 2), device=device))
            )
        }
        # BasicScheduler's denoise semantics: the last `steps` sigmas of a `steps / denoise` schedule.
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), h3_still.DEFAULT_SCHEDULER, int(steps / denoise)
        ).cpu()[-(int(steps) + 1):]
        guider = BasicGuider.execute(model, positive)[0]
        sampler = comfy.samplers.sampler_object(h3_still.DEFAULT_SAMPLER)
        result = SamplerCustomAdvanced.execute(Noise_RandomNoise(seed), guider, sampler, sigmas, latent)[0]
        return (h3_still.decode_still(vae, result),)


class VigenH3DecodeFrame:
    """Decode ONE latent frame of a sampled H3 AV latent with the still decoder."""

    CATEGORY = "lzai"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 (ComfyUI node API)
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",), "latent_index": ("INT", {"default": 0, "min": 0})}}

    def run(self, samples, vae, latent_index):
        from . import h3_still

        video = samples["samples"]
        if getattr(video, "is_nested", False):
            video = video.unbind()[0]
        index = int(latent_index)
        return (h3_still.decode_still(vae, {"samples": video[:, :, index:index + 1].contiguous()}),)


def register_nodes(comfy_nodes: Any) -> None:
    """Add the restore and frame-decode nodes to a loaded ComfyUI ``nodes`` module."""
    comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(RESTORE_NODE, VigenH3StillRestore)
    comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(DECODE_FRAME_NODE, VigenH3DecodeFrame)


def preclean(frame: Path, anchor: Path, output_path: Path) -> Path:
    """Colour statistics (LAB mean/std) matched to the anchor, then a non-local-means denoise.

    Takes the pink/saturation creep and the speckle off so a light img2img pass can restore the rest.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(frame))
    reference = cv2.imread(str(anchor))
    if image is None or reference is None:
        raise FileNotFoundError(f"cannot read {frame if image is None else anchor}")
    source = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_mean, s_std = source.reshape(-1, 3).mean(0), source.reshape(-1, 3).std(0) + 1e-6
    t_mean, t_std = target.reshape(-1, 3).mean(0), target.reshape(-1, 3).std(0)
    matched = cv2.cvtColor(
        np.clip((source - s_mean) / s_std * t_std + t_mean, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )
    cleaned = cv2.fastNlMeansDenoisingColored(matched, None, h=6, hColor=10, templateWindowSize=7, searchWindowSize=21)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cleaned)
    return output_path


def restore_frame(
    *, job_id: str, frame: Path, anchor: Path, output_path: Path, width: int, height: int, seed: int
) -> dict[str, Any]:
    """Restore ``frame`` toward ``anchor`` on the retained H3 runtime (base model, no LoRA).

    Writes ``output_path`` (PNG, ``width`` x ``height``) and returns metrics.
    """
    from . import minimax_h3_runtime as h3

    began = time.perf_counter()
    # Whichever H3 runtime is loaded: the base, text encoder and video VAE are shared, and a LoRA
    # node clones the model, so the base model here is unpatched.
    retained = h3._retained_runtime(generate_audio=True, check_models=False)
    import nodes as comfy_nodes

    register_nodes(comfy_nodes)
    root = retained.comfy_root
    relative = f"{h3._OUTPUT_SUBDIR_PREFIX}/{job_id}"
    (root / "input" / relative).mkdir(parents=True, exist_ok=True)
    pre = preclean(frame, anchor, output_path.with_name(f"{output_path.stem}.preclean.png"))
    anchor_input = f"{relative}/restore_anchor{anchor.suffix or '.png'}"
    init_input = f"{relative}/restore_init.png"
    shutil.copyfile(anchor, root / "input" / anchor_input)
    shutil.copyfile(pre, root / "input" / init_input)
    graph = {
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": h3.unet_name(), "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": h3.clip_name(), "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": h3.video_vae_name()}},
        "41": {"class_type": "LoadImage", "inputs": {"image": anchor_input}},
        "42": {"class_type": "LoadImage", "inputs": {"image": init_input}},
        "20": {
            "class_type": RESTORE_NODE,
            "inputs": {
                "model": ["2", 0],
                "clip": ["3", 0],
                "vae": ["4", 0],
                "prompt": RESTORE_PROMPT,
                "reference": ["41", 0],
                "init": ["42", 0],
                "width": int(width),
                "height": int(height),
                "denoise": RESTORE_DENOISE,
                "steps": RESTORE_STEPS,
                "seed": int(seed),
            },
        },
        h3._SAVE_IMAGE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {"images": ["20", 0], "filename_prefix": f"{relative}/restored"},
        },
    }
    try:
        saved = h3._execute_graph(retained, graph, job_id=job_id, save_node_id=h3._SAVE_IMAGE_NODE_ID)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root / "output" / saved["saved_subfolder"] / saved["saved_filename"]), output_path)
    finally:
        for name in (anchor_input, init_input):
            (root / "input" / name).unlink(missing_ok=True)
    seconds = round(time.perf_counter() - began, 3)
    log_event("h3_frame_restored", job_id=job_id, seconds=seconds, width=int(width), height=int(height))
    return {"seconds": seconds, "pipeline_seconds": round(saved["pipeline_seconds"], 3)}
