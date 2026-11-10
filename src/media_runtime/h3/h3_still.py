"""Still images from MiniMax H3: ``imagegen`` jobs on a worker with ``IMAGEGEN_BACKEND=h3``.

H3 is a video model, but it renders a good still when it samples exactly ONE
latent frame and that frame is decoded through the video VAE as a short repeated
group (a lone token decodes with banding). Prompt and reference conditioning are
H3's own: the Qwen3-VL encoder reads the references as ``<Picture N>`` and the
model receives their VAE latents as ``minimax_refs``, separate from the
generated frame. So a worker that already serves lipsync or videogen gets
text-to-image and multi-reference editing from the same base, text encoder and
video VAE, with no image model.

The recipe is the one the user approved in local/experiments/h3-image-studio-20261031:
the plain FL2VA INT8 ConvRot base (no LoRA), ``er_sde``/``simple``, 50 steps, no
CFG, on H3's native canvas of about 1 MP. Larger canvases only stretch the
image: they add no real detail and invent hard specks.

Vendored, not installed: the node is a minimal port of
ComfyUI-MiniMax-H3-Image-Studio (github.com/astropuzzo, Unlicense), whose single-
frame decoder is adapted from ComfyUI-Fizgig-H3-Still commit 10d5171 by Peter
Neill (MIT, notice below). It runs on the worker's ComfyUI 0.34 and is
registered into the retained runtime by :func:`register_nodes`.

``decode_still`` - MIT License

Copyright (c) 2026 Peter Neill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import re
from typing import Any

#: ComfyUI node class this module registers.
STILL_NODE_CLASS = "VigenH3StillRender"
#: H3 conditions on up to nine pictures (``<Picture 1>`` .. ``<Picture 9>``).
MAX_REFERENCES = 9
#: H3's canvas grid: 16x VAE downsampling times the 2x2 patch.
CANVAS_MULTIPLE = 32
#: H3 renders natively at about 1 MP; ComfyUI's own H3 node caps the canvas at
#: 768x1344. Above that the image is stretched, not more detailed.
MAX_CANVAS_PIXELS = 768 * 1344
#: Default area: 768x1024, the size the user approved over 3 MP.
DEFAULT_CANVAS_PIXELS = 768 * 1024
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 50
DEFAULT_SAMPLER = "er_sde"
DEFAULT_SCHEDULER = "simple"

_IMAGE_LABEL = re.compile(r"\bImage\s+([1-9])\b")
_PICTURE_1 = re.compile(r"<picture\s+1>", re.IGNORECASE)


def picture_tags(prompt: str) -> str:
    """``/jobs/imagegen`` numbers references ``Image 1``..``Image N``; H3's
    encoder binds them as ``<Picture 1>``..``<Picture N>``."""
    return _IMAGE_LABEL.sub(lambda match: f"<Picture {match.group(1)}>", prompt)


def fit_canvas(width: int, height: int, *, pixels: int = DEFAULT_CANVAS_PIXELS) -> tuple[int, int]:
    """The 32-pixel grid pair with ``width:height``'s aspect closest to ``pixels``.

    Never above :data:`MAX_CANVAS_PIXELS`; used for the default canvas of an edit,
    which follows ``<Picture 1>``'s shape.
    """
    if width <= 0 or height <= 0:
        raise ValueError("fit_canvas needs a positive width and height")
    ratio = width / height
    area = min(float(pixels), float(MAX_CANVAS_PIXELS))
    best: tuple[float, int, int] | None = None
    center_w = max(1, round(math.sqrt(area * ratio) / CANVAS_MULTIPLE))
    center_h = max(1, round(math.sqrt(area / ratio) / CANVAS_MULTIPLE))
    for wi in range(max(1, center_w - 4), center_w + 5):
        for hi in range(max(1, center_h - 4), center_h + 5):
            w, h = wi * CANVAS_MULTIPLE, hi * CANVAS_MULTIPLE
            if w * h > MAX_CANVAS_PIXELS:
                continue
            score = abs(math.log((w / h) / ratio)) * 4 + abs(math.log((w * h) / area))
            if best is None or score < best[0]:
                best = (score, w, h)
    assert best is not None
    return best[1], best[2]


def validate_canvas(width: int, height: int) -> None:
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(f"H3 images need width and height divisible by {CANVAS_MULTIPLE}, got {width}x{height}")
    if width * height > MAX_CANVAS_PIXELS:
        raise ValueError(
            f"H3 renders natively up to about 1 MP (768x1344 = {MAX_CANVAS_PIXELS} pixels); "
            f"{width}x{height} would only be stretched"
        )


def _reference_resize(image: Any, width: int, height: int) -> tuple[Any, int, int]:
    """Scale a reference down to the generation area (never up), on the grid."""
    import comfy.utils

    image = image[:1, ..., :3]
    h, w = int(image.shape[1]), int(image.shape[2])
    scale = min(1.0, math.sqrt((width * height) / max(1, w * h)))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    resized = comfy.utils.common_upscale(image.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1)
    return resized, tw, th


def _prepare(clip: Any, vae: Any, prompt: str, width: int, height: int, references: list[Any]) -> tuple[Any, dict]:
    import comfy.model_management
    import comfy.nested_tensor
    import node_helpers
    import torch

    items, blocks = [], []
    for reference in references:
        resized, rw, rh = _reference_resize(reference, width, height)
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image", "latent_h": rh // 16, "latent_w": rw // 16, "latent": vae.encode(resized)})
    # An anchor keyframe would occupy the ONLY output frame and block editing,
    # so references stay separate from the generated latent.
    if items:
        if len(items) == 1 and not _PICTURE_1.search(prompt):
            prompt = "<Picture 1>. " + prompt
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
    else:
        tokens = clip.tokenize(prompt, images=[])
    positive = clip.encode_from_tokens_scheduled(tokens)
    if blocks:
        positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": blocks})
    device = comfy.model_management.intermediate_device()
    video = torch.zeros((1, 24, 1, height // 16, width // 16), device=device)
    audio = torch.zeros((1, 32, 2, 2), device=device)
    return positive, {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def decode_still(vae: Any, samples: dict) -> Any:
    """Decode one sampled H3 latent without the stock lone-token banding (MIT, see above)."""
    import comfy.model_management
    import torch

    latent = samples["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    if latent.ndim != 5 or latent.shape[1] != 24 or latent.shape[2] != 1:
        raise ValueError("H3 still decode expects one H3 latent frame, not a video packet")
    fsm = vae.first_stage_model
    # Repeat only at decode time: the diffusion model sampled exactly ONE frame.
    group_size, keep = 5, 3
    shape = (1, latent.shape[1], group_size, *latent.shape[-2:])
    memory = vae.memory_used_decode(shape, vae.vae_dtype)
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=memory, force_full_load=getattr(vae, "disable_offload", False)
    )
    output = []
    with torch.no_grad():
        for index in range(latent.shape[0]):
            z = latent[index:index + 1].to(device=vae.device, dtype=vae.vae_dtype)
            mean = fsm.latents_mean.view(1, -1, 1, 1, 1).to(z)
            std = fsm.latents_std.view(1, -1, 1, 1, 1).to(z)
            group = (z * std + mean).repeat(1, 1, group_size, 1, 1)
            raw = fsm._adaptive_decode(group)
            pixels = fsm._finalize_pixels(raw[:, :, keep:keep + 1])
            # Own the storage: a one-frame view must not pin all decoded frames.
            image = pixels[:, :, 0].movedim(1, -1).to(
                device=comfy.model_management.intermediate_device(), dtype=torch.float32, copy=True
            )
            output.append(image)
            del z, mean, std, group, raw, pixels
    return torch.cat(output)


class VigenH3StillRender:
    """Text-to-image or reference edit: one H3 latent frame, sampled and decoded."""

    CATEGORY = "vigen"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 (ComfyUI node API)
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": DEFAULT_WIDTH, "min": CANVAS_MULTIPLE, "max": 4096, "step": CANVAS_MULTIPLE}),
                "height": ("INT", {"default": DEFAULT_HEIGHT, "min": CANVAS_MULTIPLE, "max": 4096, "step": CANVAS_MULTIPLE}),
                "steps": ("INT", {"default": DEFAULT_STEPS, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": DEFAULT_SAMPLER}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": DEFAULT_SCHEDULER}),
            },
            "optional": {f"reference_{index}": ("IMAGE",) for index in range(1, MAX_REFERENCES + 1)},
        }

    def render(self, model, clip, vae, prompt, width, height, steps, seed, sampler_name, scheduler, **references):
        import comfy.samplers
        from comfy_extras.nodes_custom_sampler import BasicGuider, Noise_RandomNoise, SamplerCustomAdvanced

        validate_canvas(int(width), int(height))
        ordered = [
            references[name]
            for name in (f"reference_{index}" for index in range(1, MAX_REFERENCES + 1))
            if references.get(name) is not None
        ]
        positive, latent = _prepare(clip, vae, prompt.strip(), int(width), int(height), ordered)
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
        guider = BasicGuider.execute(model, positive)[0]
        result = SamplerCustomAdvanced.execute(Noise_RandomNoise(seed), guider, sampler, sigmas, latent)[0]
        return (decode_still(vae, result),)


def register_nodes(comfy_nodes: Any) -> None:
    """Add :class:`VigenH3StillRender` to a loaded ComfyUI ``nodes`` module."""
    comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(STILL_NODE_CLASS, VigenH3StillRender)
    comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.setdefault(STILL_NODE_CLASS, "Vigen H3 Still Render")
