"""MiniMax H3 FL2VA video runtime (headless ComfyUI), for ``videogen`` and ``lipsync``.

Served recipe (``H3_VARIANT=int8_turbo``, the default): the pruned
**int8_convrot** FL2VA base, the **int4_convrot** Qwen3-VL 32B text encoder,
fp16 video / fp32 audio VAEs, the ``MiniMaxH3TurboLoRA`` node (``strength=1.0``,
``low_vram=true``) and the ``MiniMaxH3TurboSampler``, 4 steps on ``simple``.
Lipsync swaps the LoRA for our 3-step student (:data:`LIPSYNC_LORA_NAME`) and
samples under SLA block-sparse attention. ``int8_dmd3`` and ``nvfp4_vanilla``
are kept as alternative recipes; the env overrides (``H3_UNET_NAME``,
``H3_CLIP_NAME``, ``H3_TURBO_LORA_NAME``, ``H3_STEPS``...) still apply.

The geometry is ours to choose, the grid is not. H3 encodes latents at 16x and
patches them 2x2, so a canvas whose keyframes condition the DiT has to sit on a
32-px lattice; ``resolve_output_dimensions`` searches that lattice for the pair
holding the requested aspect ratio instead of rounding each axis on its own, and
``fit_video_dimensions`` finishes the clip at exactly the frame size the client
named (uniform scale, sub-percent centre crop - never a squeeze). The frame count
is whatever the node's own ``17k+5`` ``temporal_shape`` snapping makes of the
requested length. The one thing this module changes in the ComfyUI tree is the
keyframe fit: upstream stretches the first keyframe but crops the last one, so
:func:`ensure_minimax_h3_keyframe_geometry` keeps both on the same
aspect-preserving crop.

Provisioning: ``data/comfyui`` (S3-baked like the rest of ``data/``; override
with ``H3_COMFYUI_DIR``) holds the ComfyUI source, the ``ComfyUI-MiniMax-H3-Turbo``
and ``ComfyUI-H3-SLA-Attention`` custom nodes and the weights under
``models/<type>/``.

The headless ComfyUI prompt executor runs in-process; the retained executor
keeps the models warm between jobs, and :func:`unload_retained_h3_runtime`
drops it, unloads every ComfyUI model and clears the CUDA cache (the media
runtime does that only before it loads the music models). One executor serves
every H3 graph (videogen with or without audio, lipsync, imagegen): each graph
loads the VAEs it needs, so nothing is torn down between H3 jobs.

FL2VA always decodes the audio VAE, so a videogen mp4 carries a synchronized
audio track; the Turbo Sampler is what keeps that track intelligible.
"""

from __future__ import annotations

import asyncio
import gc
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import progress

from .compat import log_event
from .compat import COMFYUI_DIR, data_dir, project_path

H3_FPS = 24.0

#: Pruned int8_convrot FL2VA base - the checkpoint the Turbo LoRA is validated
#: against ("default pruned_int8_convrot model"), 21 GB, streamed by ComfyUI.
DEFAULT_UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
#: int4 ConvRot Qwen3-VL 32B text encoder (15 GB; the nvfp4 one needs Blackwell).
DEFAULT_CLIP_NAME = "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
DEFAULT_VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"

#: The Turbo LoRA the reference run uses ("the ema_ckpt850 version", which the
#: author says beat the non-ema one, and which the node ships a silu(t_emb)
#: grid for so one LoRA covers every pruned base). Its keys are bare module
#: names (``blocks.0...``) with the adaln rows; a ``diffusion_model.``-prefixed
#: export patches nothing in this node (the old ckpt500 file did exactly that).
DEFAULT_TURBO_LORA_NAME = "minimax_h3_turbo_4step_ema_ckpt850.safetensors"
#: README: "Keep strength at 1.0. It's tuned for 1.0 and holds up across the
#: 4-8 step range."
DEFAULT_TURBO_STRENGTH = 1.0
#: low_vram ON merges the LoRA into the weights: lowest peak VRAM (the setting
#: that makes a 32 GB card work) and, per the reference report, the difference
#: between a "cooked AI look" and a clean render.
DEFAULT_TURBO_LOW_VRAM = True
#: The node's own directory name under ComfyUI/custom_nodes.
TURBO_NODE_DIRNAME = "ComfyUI-MiniMax-H3-Turbo"
#: ``sparse_attention`` renders: Comfy Kitchen INT8 attention under PlagueKind's
#: training-free block-sparse SLA node, the pick of the H3 attention benchmark
#: (local/experiments/h3-sparse-attn-20261016) confirmed on the 3-step DMD
#: student (h3-dmd-avatar-20261017): 85% of key blocks skipped on every step but
#: the second, which stays dense. Diffusion ~1.75x faster, no visible change.
SLA_NODE_DIRNAME = "ComfyUI-H3-SLA-Attention"
SLA_ATTENTION_INPUTS: dict[str, Any] = {
    "sparsity_ratio": 0.85,
    "block_size": "32",
    "min_seq_len": 12228,
    "dense_last_steps": 0,
    "protect_audio": False,
    "enabled": True,
    "dense_steps": "1",
    "dense_backend": "comfy_kitchen",
    "disable_fp16_accum": True,
    "stabilize_motion": False,
    "reference_protection": "Off",
    "tail_correction": False,
    "use_int8_qk": True,
    "engine": "comfy_kitchen",
}

# H3 canvas constraints, read from the provisioned ComfyUI runtime when present
# (see _stock_h3_nodes()): each axis a multiple of 32, area at or under the
# model's 1 MP ceiling (768*1344), short-edge-768 reference canvas. These
# literals only mirror that module for environments where it cannot be
# imported, i.e. unit tests without a provisioned runtime.
H3_BASE_SHORT_EDGE = 768
H3_MAX_PIXELS = 768 * 1344
H3_CANVAS_MULTIPLE = 32
#: Aspect error we treat as free: 0.005 is 0.5% of scale, about 6 px on a
#: 1280-tall canvas, far below what anyone sees in a moving image.
H3_ASPECT_TOLERANCE = 0.005
#: VRAM a helper outside ComfyUI needs free beside H3 (``make_room_beside_h3``), measured with its weights loading:
#: ZipEnhancer 1.18 GB at any length, RIFE 1.1 at 1 MP, CTC 0.69, S3FD/SyncNet + glitch 0.59 (3090, 2026-11-15).
HELPER_VRAM_BYTES = 1536 << 20
#: How much of the requested pixel budget we refuse to give up to chase a
#: perfect ratio. The 32-px block is ~4.5% of a 700-px axis, so demanding the
#: exact aspect would sometimes cost a quarter of the resolution; we only take
#: a canvas that keeps this fraction of the requested area.
H3_CANVAS_AREA_SLACK = 0.10
#: Aspect error that is worth a warning line. 480x848 is a real client preset and
#: the lattice answer is 480x832 + a 1.9% finish scale: correct, not an error, so
#: the search band above stays at 0.5% and only a crop this size gets logged.
H3_CANVAS_ASPECT_WARN = 0.02
H3_DEFAULT_WIDTH = 1344
H3_DEFAULT_HEIGHT = 768

#: Served recipe: the Turbo LoRA's 4 steps on the ``simple`` schedule (the node's
#: minimum; the upstream reference runs 6).
DEFAULT_STEPS = 4
DEFAULT_SCHEDULER = "simple"
#: Step count reported to the job record; :func:`h3_steps` is what the graph uses.
H3_RECIPE_STEPS = DEFAULT_STEPS

#: Sampling recipe for the H3 FL2VA backend, selected by the ``H3_VARIANT``
#: worker env var. ``int8_turbo`` is the recipe above; ``nvfp4_vanilla`` is the
#: plain pruned-NVFP4 checkpoint with the stock ComfyUI sampler chain (no
#: Turbo LoRA), the setup benchmarked on one RTX 5090 (four files, 31.7 GB).
DEFAULT_H3_VARIANT = "int8_turbo"
NVFP4_VANILLA_VARIANT = "nvfp4_vanilla"
INT8_DMD3_VARIANT = "int8_dmd3"
H3_VARIANTS = (DEFAULT_H3_VARIANT, NVFP4_VANILLA_VARIANT, INT8_DMD3_VARIANT)

#: Pruned NVFP4 FL2VA DiT provisioned beside the int8 base (the
#: ``coolthor/MiniMax-H3-pruned-NVFP4`` file, 11.7 GB).
NVFP4_VANILLA_UNET_NAME = "minimax_h3_fl2va_pruned_nvfp4.safetensors"
#: The benchmarked recipe: 10 steps (20 -> 10 measured 324 s -> 172 s with no
#: visible quality difference; the weights are CFG-distilled).
NVFP4_VANILLA_STEPS = 10
#: Stock ComfyUI ``KSamplerSelect`` sampler the reference run uses.
NVFP4_VANILLA_SAMPLER = "res_multistep"

#: The ``int8_dmd3`` recipe: a DMD2 student distilled from the Turbo recipe on
#: talking-avatar clips, trained on the int8_convrot base. It is a drop-in for
#: the ema850 weights in ``MiniMaxH3TurboLoRA`` (same node, same low_vram merge,
#: same Turbo sampler) and samples ``simple`` at 3 steps, i.e. every other sigma
#: of the 6-step schedule: [1, .9601, .8575, 0]. Validated in the production
#: graph on gesturing noise draws where 2-step students collapse.
DMD3_TURBO_LORA_NAME = "minimax_h3_turbo3_dmd_avatar_ema_g00100.safetensors"
DMD3_STEPS = 3

#: Freestyle lipsync (``lipsync`` jobs): our 3-step avatar
#: student "ff3sla2 g120" (local/experiments/h3-dmd-avatar-20261017), a drop-in
#: for the ema850 weights in ``MiniMaxH3TurboLoRA``. It samples ema850's own
#: 4-step ``simple`` sigmas [1, .973, .923, .8, 0] with the first two steps
#: folded into one, and was trained with :data:`SLA_ATTENTION_INPUTS` on its
#: rollouts, so the job samples sparse by default.
LIPSYNC_LORA_NAME = "minimax_h3_foldfirst3_ff3sla2_ema_g00120.safetensors"
LIPSYNC_SIGMAS = "1.0, 0.9230769276618958, 0.800000011920929, 0.0"
LIPSYNC_STEPS = 3

#: The upstream 8-step FL2V Turbo LoRA (``diffusion_model.*`` keys, so the stock
#: ``LoraLoaderModelOnly`` loads it), sampled ``res_multistep``.
TURBO8_LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


@dataclass(frozen=True)
class H3Recipe:
    """One sampling recipe on the H3 base: the LoRA (if any), step count and sampler.

    ``sampler`` is ``"turbo"`` for the ``MiniMaxH3TurboSampler`` node, else a stock
    ``KSamplerSelect`` name. ``sigmas`` (``ManualSigmas``) replace the ``simple``
    schedule when set; only the lipsync graph honours them. ``preview_sigmas`` are the
    short schedule of a one-frame preview (:mod:`h3.h3_restore`, clean chunk joins);
    ``None`` previews on the recipe's full schedule.
    """

    steps: int
    sampler: str = "turbo"
    lora_name: str | None = None
    lora_loader: str = "MiniMaxH3TurboLoRA"
    sigmas: str | None = None
    preview_sigmas: str | None = None

    @property
    def sampler_label(self) -> str:
        return "minimax_h3_turbo_sampler" if self.sampler == "turbo" else self.sampler


#: Preview schedules of the clean chunk joins, measured by vigen (h3-derift-20261113): the lipsync
#: student's sigma-1 prediction is already a clean frame; the ema850 Turbo LoRA needs 3 steps of its
#: own schedule (1 step is a ghosted blur, 2 still double the edges). Other recipes were never
#: measured and preview on their full schedule.
LIPSYNC_PREVIEW_SIGMAS = "1.0, 0.0"
TURBO_PREVIEW_SIGMAS = "1.0, 0.8575, 0.7064, 0.0"

#: ``preset`` values of videogen jobs. ``standard`` is the plain base on the
#: CFG-distilled checkpoint's benchmarked recipe (res_multistep, 10 steps, no LoRA);
#: ``turbo-N`` are the Turbo LoRAs: ema850 at 4 (the served default) or 6 steps (vigen's
#: production recipe), the upstream 8-step LoRA at 8.
VIDEO_PRESETS: dict[str, H3Recipe] = {
    "standard": H3Recipe(steps=NVFP4_VANILLA_STEPS, sampler=NVFP4_VANILLA_SAMPLER),
    "turbo-8": H3Recipe(steps=8, sampler="res_multistep", lora_name=TURBO8_LORA_NAME, lora_loader="LoraLoaderModelOnly"),
    "turbo-6": H3Recipe(steps=6, lora_name=DEFAULT_TURBO_LORA_NAME, preview_sigmas=TURBO_PREVIEW_SIGMAS),
    "turbo-4": H3Recipe(steps=4, lora_name=DEFAULT_TURBO_LORA_NAME, preview_sigmas=TURBO_PREVIEW_SIGMAS),
}
#: ``preset`` values of lipsync jobs: the video presets plus ``turbo-3``, the lipsync
#: student (:data:`LIPSYNC_LORA_NAME` on :data:`LIPSYNC_SIGMAS`), the default.
LIPSYNC_PRESETS: dict[str, H3Recipe] = {
    **VIDEO_PRESETS,
    "turbo-3": H3Recipe(
        steps=LIPSYNC_STEPS, lora_name=LIPSYNC_LORA_NAME, sigmas=LIPSYNC_SIGMAS, preview_sigmas=LIPSYNC_PREVIEW_SIGMAS
    ),
}
DEFAULT_LIPSYNC_PRESET = "turbo-3"


def h3_preset(name: str, presets: dict[str, H3Recipe]) -> H3Recipe:
    if name not in presets:
        raise ValueError(f"unknown preset {name!r}; expected one of: {', '.join(presets)}")
    return presets[name]

#: Model-artifact name these weights are provisioned under.
DEFAULT_ARTIFACT_NAME = "minimax_h3_comfy_turbo"
#: The ``int8_dmd3`` artifact: the same base, text encoder and VAEs with the
#: DMD3 LoRA instead of ema850, so a DMD3 worker downloads no other LoRA.
DMD3_ARTIFACT_NAME = "minimax_h3_comfy_dmd3"
#: The lipsync artifact: the same base, text encoder and VAEs with the lipsync
#: student's LoRA.
LIPSYNC_ARTIFACT_NAME = "minimax_h3_comfy_lipsync"
#: Every H3 artifact carries the base, text encoder and video VAE that still
#: images need, so a worker holding any of them can render stills.
H3_ARTIFACT_NAMES = frozenset({
    DEFAULT_ARTIFACT_NAME, DMD3_ARTIFACT_NAME, LIPSYNC_ARTIFACT_NAME, "minimax_h3_comfy_nvfp4",
})

#: Log-distance aspect error at which two uploaded keyframes are worth flagging
#: in the job log: about 2% of scale, i.e. a visible framing crop difference.
H3_KEYFRAME_ASPECT_WARN = 0.02

# --------------------------------------------------------------------------
# Keyframe geometry fixup for the stock H3 node
# --------------------------------------------------------------------------
# ``MiniMaxH3ImageToVideo`` fits its two keyframes onto the canvas with
# *different* rules: the first frame is stretched (``crop="disabled"``) and the
# last frame is cover-cropped (``crop="center"``).  That is only harmless when
# the canvas aspect equals the image aspect, which H3's grid makes impossible:
# every axis has to be a multiple of 32, so a 480x848 request renders on a
# 480x832 canvas and an 884x1556 avatar lands on 768x1344 - 1.5-2% off in both
# cases.  The stretch squeezes the first frame by that error while the
# cover-crop zooms the last frame by it, so the clip ends on a shot ~2% larger
# than it started, and the two keyframe latents the model is conditioned on
# disagree about scale.  Same-size first/last pairs - what the product actually
# sends - are hit even though nothing about the inputs is mismatched.
#
# Fixing it in the node (one rule for both keyframes, never a distortion) is
# the only place the geometry can be reconciled without pre-warming the
# uploaded images, so we keep the provisioned runtime file in sync instead of
# pre-cropping keyframes in this repo.  Upstream ComfyUI (0.34.0 and current
# master) still ships the split rule.
H3_GEOMETRY_MARKER = "vigen: keyframes share one aspect-preserving fit"
_H3_STOCK_FIRST_KEYFRAME_FIT = '_resize(first_frame[:1], width, height, "disabled")'
_H3_FIXED_FIRST_KEYFRAME_FIT = '_resize(first_frame[:1], width, height, "center")'
_H3_STOCK_FIT_COMMENT = "plain stretch to canvas"


def h3_node_module_path(comfy_root: Path | None = None) -> Path:
    """Where the provisioned runtime keeps the stock H3 latent nodes."""
    comfy_root = comfy_root or ensure_comfyui_runtime()
    return comfy_root / "comfy_extras" / "nodes_minimax_h3.py"


def h3_keyframe_geometry_state(comfy_root: Path | None = None) -> str:
    """``"patched"``, ``"stock"`` or ``"unknown"`` for the provisioned node.

    ``"unknown"`` means the file no longer contains either shape, i.e. the
    runtime was upgraded and the fixup has to be re-checked by hand.
    """
    path = h3_node_module_path(comfy_root)
    if not path.exists():
        return "missing"
    source = path.read_text(encoding="utf-8")
    if H3_GEOMETRY_MARKER in source:
        return "patched"
    if _H3_STOCK_FIRST_KEYFRAME_FIT in source:
        return "stock"
    return "unknown"


def ensure_minimax_h3_keyframe_geometry(comfy_root: Path | None = None) -> str:
    """Make both H3 keyframes use the same aspect-preserving cover-crop.

    Idempotent, and deliberately narrow: it rewrites the single ``_resize`` call
    for ``first_frame`` plus its now-wrong comment.  Anything it does not
    recognise raises instead of silently rendering with mismatched keyframe
    geometry, so a ComfyUI bump has to be reconciled with this function.
    """
    import os

    path = h3_node_module_path(comfy_root)
    state = h3_keyframe_geometry_state(path.parent.parent)
    if state == "patched":
        return "patched"
    if state != "stock":
        raise RuntimeError(
            "Cannot verify H3 keyframe geometry in "
            f"{path} (state {state!r}). The node must fit first_frame and "
            "last_frame with the same aspect-preserving crop - "
            f"expected either the marker {H3_GEOMETRY_MARKER!r} or the upstream "
            f"line {_H3_STOCK_FIRST_KEYFRAME_FIT!r}. Re-check this fixup against "
            "the new comfy_extras/nodes_minimax_h3.py."
        )

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    changed = False
    for index, line in enumerate(lines):
        if not changed and _H3_STOCK_FIRST_KEYFRAME_FIT in line:
            indent = line[: len(line) - len(line.lstrip())]
            previous = lines[index - 1] if index else ""
            has_comment = _H3_STOCK_FIT_COMMENT in previous
            if has_comment:
                # Replace the misleading "plain stretch" comment with the fix.
                out[-1] = f"{indent}# {H3_GEOMETRY_MARKER}\n"
            fixed = line.replace(_H3_STOCK_FIRST_KEYFRAME_FIT, _H3_FIXED_FIRST_KEYFRAME_FIT)
            if not has_comment:
                fixed = f"{fixed.rstrip(chr(10))}  # {H3_GEOMETRY_MARKER}\n"
            out.append(fixed)
            changed = True
            continue
        out.append(line)

    tmp_path = path.with_name(f".{path.name}.h3.tmp")
    tmp_path.write_text("".join(out), encoding="utf-8")
    os.replace(tmp_path, path)
    if h3_keyframe_geometry_state(path.parent.parent) != "patched":
        raise RuntimeError(f"H3 keyframe geometry fixup did not apply to {path}")
    log_event("minimax_h3_keyframe_geometry_patched", node_module=str(path))
    return "patched"

# Node id whose SaveVideo output is read out of the executor history.
_SAVE_VIDEO_NODE_ID = "17"
# Node id of the SaveImage in the still-image graph.
_SAVE_IMAGE_NODE_ID = "40"
# H3 inference output files land under this subfolder of the ComfyUI output dir.
_OUTPUT_SUBDIR_PREFIX = "jobs"


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def h3_variant() -> str:
    """Active sampling recipe (``H3_VARIANT`` env, ``int8_turbo`` by default)."""
    value = os.getenv("H3_VARIANT", "").strip().lower() or DEFAULT_H3_VARIANT
    if value not in H3_VARIANTS:
        raise ValueError(f"H3_VARIANT must be one of: {', '.join(H3_VARIANTS)}")
    return value


def is_nvfp4_vanilla() -> bool:
    return h3_variant() == NVFP4_VANILLA_VARIANT


def is_int8_dmd3() -> bool:
    return h3_variant() == INT8_DMD3_VARIANT


def h3_artifact_name() -> str:
    """Model artifact that provisions the active variant's weights."""
    return DMD3_ARTIFACT_NAME if is_int8_dmd3() else DEFAULT_ARTIFACT_NAME


def h3_engine_name() -> str:
    """Engine label reported in job metrics for the active variant."""
    if is_nvfp4_vanilla():
        return "comfyui_minimax_h3_nvfp4_vanilla"
    if is_int8_dmd3():
        return "comfyui_minimax_h3_dmd3"
    return "comfyui_minimax_h3_turbo"


def unet_name() -> str:
    default = NVFP4_VANILLA_UNET_NAME if is_nvfp4_vanilla() else DEFAULT_UNET_NAME
    return _env("H3_UNET_NAME", default)


def clip_name() -> str:
    return _env("H3_CLIP_NAME", DEFAULT_CLIP_NAME)


def video_vae_name() -> str:
    return _env("H3_VIDEO_VAE_NAME", DEFAULT_VIDEO_VAE_NAME)


def audio_vae_name() -> str:
    return _env("H3_AUDIO_VAE_NAME", DEFAULT_AUDIO_VAE_NAME)


def h3_steps() -> int:
    if is_nvfp4_vanilla():
        default = NVFP4_VANILLA_STEPS
    elif is_int8_dmd3():
        default = DMD3_STEPS
    else:
        default = DEFAULT_STEPS
    return int(_env("H3_STEPS", str(default)))


def turbo_lora_name() -> str:
    default = DMD3_TURBO_LORA_NAME if is_int8_dmd3() else DEFAULT_TURBO_LORA_NAME
    return _env("H3_TURBO_LORA_NAME", default)


def turbo_strength() -> float:
    return float(_env("H3_TURBO_LORA_STRENGTH", str(DEFAULT_TURBO_STRENGTH)))


def turbo_low_vram() -> bool:
    value = os.getenv("H3_TURBO_LOW_VRAM", "").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return DEFAULT_TURBO_LOW_VRAM


def h3_sampler() -> str:
    """Sampler reported to the job record (node type, or KSamplerSelect name)."""
    return NVFP4_VANILLA_SAMPLER if is_nvfp4_vanilla() else "minimax_h3_turbo_sampler"


def h3_scheduler() -> str:
    return _env("H3_SCHEDULER", DEFAULT_SCHEDULER)


def env_recipe() -> H3Recipe:
    """The worker's own recipe (``H3_VARIANT``/``H3_STEPS``/``H3_TURBO_LORA_NAME``): videogen without a preset."""
    if is_nvfp4_vanilla():
        return H3Recipe(steps=h3_steps(), sampler=NVFP4_VANILLA_SAMPLER)
    lora = turbo_lora_name()
    return H3Recipe(
        steps=h3_steps(),
        lora_name=lora,
        preview_sigmas=TURBO_PREVIEW_SIGMAS if lora == DEFAULT_TURBO_LORA_NAME else None,
    )


def comfyui_runtime_dir() -> Path:
    value = os.getenv("H3_COMFYUI_DIR")
    if value:
        return project_path(value)
    return COMFYUI_DIR


def _stock_h3_nodes():
    """Return the provisioned ComfyUI runtime's own H3 node module, if importable.

    The canvas rules live in ``comfy_extras.nodes_minimax_h3`` (``CANVAS_MULTIPLE``,
    ``MAX_PIXELS``, ``adapt_canvas``). We call that module instead of restating it
    so the runtime can never drift from the graph the reference workflow uses;
    the module-level constants in this file are only a fallback for a process
    without a provisioned runtime (unit tests).
    """
    try:
        from comfy_extras import nodes_minimax_h3  # type: ignore
    except Exception:
        return None
    return nodes_minimax_h3


def _canvas_multiple() -> int:
    nodes = _stock_h3_nodes()
    return int(getattr(nodes, "CANVAS_MULTIPLE", H3_CANVAS_MULTIPLE)) if nodes else H3_CANVAS_MULTIPLE


def _snap_to_grid(width: int, height: int) -> tuple[int, int]:
    """Round each axis to the nearest H3 32-multiple, independently.

    This is what ``comfy_extras.nodes_minimax_h3.adapt_canvas`` does, and it is
    kept only as the last-resort fallback for a request too small to sit on the
    lattice at all: rounding the two axes separately changes their ratio by up
    to half a block per axis, which is exactly the distortion :func:`_grid_pair`
    exists to avoid (720x1280 -> 704x1280 is 2.3% of aspect).
    """
    import math

    w = max(_canvas_multiple(), round(width / _canvas_multiple()) * _canvas_multiple())
    h = max(_canvas_multiple(), round(height / _canvas_multiple()) * _canvas_multiple())
    return int(w), int(h)


def _grid_pair(
    width: int, height: int, *, max_pixels: int | None = None
) -> tuple[int, int]:
    """Canvas on H3's 32-multiple lattice that keeps ``width/height``'s aspect.

    Both axes are free to move; the *ratio* is not. Rounding each axis
    separately - what :func:`adapt_canvas` does - rotates the frame by up to
    half a block per axis, because a 32 px block is 4.5% of a 700 px axis
    (720x1280 -> 704x1280 is 2.3% of aspect, and 720 is 45 latents wide, which
    H3's patchify cannot express).

    So walk the lattice (``w = round(ratio * h)`` snapped to the block, for
    every valid ``h``) and take the pair with the **smallest aspect error**
    among those keeping at least ``1 - H3_CANVAS_AREA_SLACK`` of the pixel
    budget, preferring more pixels on ties. The budget is the requested area
    capped by the model's ``MAX_PIXELS``: we never upscale a request and never
    exceed the resolution the checkpoint was trained for.

    720x1280 -> 704x1248 (0.28%, 0.88 MP of 0.92); 864x480 -> 864x480
    (already on the lattice); 480x848 -> 480x832 (1.9% - the exact-ratio pair
    416x736 would cost 25% of the pixels, so the finish crop absorbs it);
    884x1556 -> 768x1344 (0.58%, the ceiling).
    """
    import math

    block = _canvas_multiple()
    if width <= 0 or height <= 0:
        raise ValueError(f"_grid_pair requires positive dimensions, got {width}x{height}")
    ratio = width / height
    budget = min(int(width) * int(height), H3_MAX_PIXELS if max_pixels is None else int(max_pixels))
    floor = budget * (1.0 - H3_CANVAS_AREA_SLACK)

    candidates: list[tuple[float, int, int, int]] = []
    # w * h == ratio * h^2 <= budget; the extra slack is the tolerance band, so
    # a canvas one block taller than the budget circle still gets considered.
    h_stop = int((budget / (ratio * (1.0 - H3_ASPECT_TOLERANCE))) ** 0.5) + block
    for h in range(block, h_stop, block):
        if h * block > budget:
            break
        nearest = round(h * ratio / block)
        for w in {max(1, nearest - 1) * block, max(1, nearest) * block}:
            if w * h > budget:
                continue
            candidates.append((abs(math.log((w / h) / ratio)), w * h, w, h))
    if not candidates:
        # Smaller than one lattice block: nothing preserves the ratio here.
        return _snap_to_grid(width, height)

    pool = [item for item in candidates if item[1] >= floor] or candidates
    err, _, w, h = min(pool, key=lambda item: (round(item[0], 9), -item[1]))
    return int(w), int(h)


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """The stock H3 canvas: ``comfy_extras.nodes_minimax_h3.adapt_canvas``.

    Delegates to the provisioned runtime's own function; the local mirror below
    exists only so this module stays importable without a provisioned ComfyUI
    tree. Both are the same rule:
    - snap the short edge to ``H3_BASE_SHORT_EDGE`` (768)
    - cap the total area at ``H3_MAX_PIXELS`` (768*1344 ≈ 1 MP)
    - round each axis to a multiple of ``H3_CANVAS_MULTIPLE`` (32)

    The result fits on the H3 AV latent grid (H/16, W/16) and is what the
    ``MiniMaxH3ImageToVideo`` node needs. A square input gets 768x768; a
    16:9 input gets 1344x768; a portrait 9:16 input gets 768x1344.

    **Serving does not use this.** It upscales to the 768 short edge and rounds
    each axis on its own, which is what :func:`_grid_pair` replaced; the
    function stays as the pinned oracle for
    ``test_adapt_canvas_matches_the_stock_node_rule`` so the divergence we
    correct for is written down, not folklore.
    """
    import math

    nodes = _stock_h3_nodes()
    if nodes is not None:
        return tuple(int(v) for v in nodes.adapt_canvas(width, height))  # type: ignore[return-value]
    if width <= 0 or height <= 0:
        raise ValueError(f"adapt_canvas requires positive dimensions, got {width}x{height}")
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = H3_BASE_SHORT_EDGE * ratio, H3_BASE_SHORT_EDGE
    else:
        nom_w, nom_h = H3_BASE_SHORT_EDGE, H3_BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > H3_MAX_PIXELS:
        s = (H3_MAX_PIXELS / (nom_w * nom_h)) ** 0.5
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(H3_CANVAS_MULTIPLE, round(nom_w / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE),
        max(H3_CANVAS_MULTIPLE, round(nom_h / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE),
    )


def probe_image_dimensions(path: Path) -> tuple[int, int]:
    """Read (width, height) from an image file without decoding pixels.

    Falls back to ``(H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT)`` if Pillow can't
    open the file, so a transient probe failure never aborts a job. The EXIF
    orientation is honoured because ComfyUI's ``LoadImage`` runs
    ``ImageOps.exif_transpose``; measuring anything else would size the canvas
    for a framing the node never sees.
    """
    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception:
        return H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT
    try:
        with Image.open(path) as handle:
            oriented = ImageOps.exif_transpose(handle)
            return int(oriented.width), int(oriented.height)
    except Exception:
        return H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT


def _fit_canvas(width: int, height: int) -> tuple[int, int]:
    """Snap a nominal canvas onto H3's lattice without moving its aspect.

    Unlike :func:`adapt_canvas` this never upscales to the 768 short-edge
    base: the missing axis (or the whole canvas, when nothing is requested)
    is derived straight from the input image's aspect and resolution. The
    aspect-preserving lattice search in :func:`_grid_pair` also enforces the
    ``H3_MAX_PIXELS`` ceiling.
    """
    return _grid_pair(width, height)


def resolve_output_dimensions(
    *,
    image_path: Path,
    requested_width: int | None,
    requested_height: int | None,
) -> tuple[int, int]:
    """Pick the (width, height) we render the H3 video at.

    Priority:
    1. Both ``requested_width`` and ``requested_height`` — caller asked for
       an exact frame size; :func:`_grid_pair` finds the biggest lattice canvas
       with *that* aspect ratio under the requested area (864x480 passes
       through unchanged, 720x1280 renders at 704x1248).
    2. One axis requested — the missing axis is derived from the input
       image's aspect ratio. No upscaling to a 768 short edge.
    3. Neither requested — the output is the input image's own dimensions,
       snapped to the grid. A 512x768 input renders at 512x768.

    Downscaling happens only when the resulting canvas exceeds the H3 1 MP
    ceiling, which the model cannot encode.
    """
    src_w, src_h = probe_image_dimensions(image_path)
    if requested_width and requested_height:
        return _grid_pair(int(requested_width), int(requested_height))
    if requested_width:
        return _fit_canvas(int(requested_width), int(round(int(requested_width) * src_h / src_w)))
    if requested_height:
        return _fit_canvas(int(round(int(requested_height) * src_w / src_h)), int(requested_height))
    return _fit_canvas(src_w, src_h)


class H3Canvas(NamedTuple):
    """A render canvas plus the provenance needed to audit how it was chosen.

    Same record the LightX2V runtime returns so a job log reads the same for
    either engine.
    """

    width: int
    height: int
    source: str = "requested"
    aspect_error: float = 0.0
    #: How far the canvas sits from the *requested* ratio (0 when that ratio
    #: already lands on the lattice). The finish crop in
    #: :func:`fit_video_dimensions` removes exactly this much of the frame.
    requested_aspect_error: float = 0.0

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def resolve_canvas(
    *,
    image_path: Path,
    requested_width: int | None = None,
    requested_height: int | None = None,
) -> H3Canvas:
    """Engine-facing canvas resolution: H3's 32-px lattice, aspect kept intact.

    A thin record over :func:`resolve_output_dimensions`. The canvas comes from
    :func:`_grid_pair`, which moves both axes to hold the requested ratio
    instead of rounding them one at a time; :func:`fit_video_dimensions` then
    delivers the caller's exact frame size. The node itself decides how to fit
    an image to the canvas, and :func:`ensure_minimax_h3_keyframe_geometry`
    keeps that fit identical for both keyframes; we never pre-crop or rescale an
    uploaded image here.
    """
    import math

    width, height = resolve_output_dimensions(
        image_path=image_path,
        requested_width=requested_width,
        requested_height=requested_height,
    )
    source_width, source_height = probe_image_dimensions(image_path)
    if requested_width and requested_height:
        source = "requested"
    elif requested_width or requested_height:
        source = "requested_axis+image_aspect"
    else:
        source = "image"
    requested_aspect_error = 0.0
    if requested_width and requested_height and requested_width > 0 and requested_height > 0:
        requested_aspect_error = round(
            abs(math.log((width / height) / (requested_width / requested_height))), 6
        )
    aspect_error = 0.0
    if source_width > 0 and source_height > 0 and height > 0:
        aspect_error = round(abs(math.log((width / height) / (source_width / source_height))), 6)
    return H3Canvas(
        width=width,
        height=height,
        source=source,
        aspect_error=aspect_error,
        requested_aspect_error=requested_aspect_error,
    )


#: Sizes delivered to clients follow the same 16-px convention the API validates.
DELIVERY_MULTIPLE = 16


def delivery_dimensions(
    canvas: H3Canvas,
    requested_width: int | None = None,
    requested_height: int | None = None,
    source_aspect: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """The frame size to hand back for ``canvas``, or ``None`` to deliver the canvas.

    A caller that pinned both axes gets those exact pixels; a caller that named
    one axis gets *that* axis exactly, with the other derived from
    ``source_aspect`` (the input image, i.e. what they left to us) on the API's
    16-px convention, falling back to the render ratio. Nobody gets a surprise
    2% size change - the lattice only ever moves the number of pixels the model
    reasons over, and :func:`fit_video_dimensions` converts that back to what was
    asked for.
    """
    if requested_width and requested_height:
        return int(requested_width), int(requested_height)
    if not (requested_width or requested_height):
        return None
    if source_aspect and source_aspect[0] > 0 and source_aspect[1] > 0:
        ratio = source_aspect[0] / source_aspect[1]
    else:
        ratio = canvas.width / canvas.height
    if requested_height:
        height = int(requested_height)
        width = max(DELIVERY_MULTIPLE, round(height * ratio / DELIVERY_MULTIPLE) * DELIVERY_MULTIPLE)
        return width, height
    width = int(requested_width)
    height = max(DELIVERY_MULTIPLE, round(width / ratio / DELIVERY_MULTIPLE) * DELIVERY_MULTIPLE)
    return width, height


def keyframe_geometry_report(
    *,
    image_path: Path,
    last_frame_path: Path | None,
    canvas: H3Canvas,
) -> dict[str, Any]:
    """How far each uploaded keyframe sits from the render canvas.

    Both keyframes get the same aspect-preserving cover-crop (see
    :func:`ensure_minimax_h3_keyframe_geometry`), so an aspect error no longer
    makes the last frame a different scale than the first - it just crops the
    same border off both. A large error is still worth seeing in the job record:
    it means the crop is discarding most of one frame, which is only visible as
    lost framing in the output.

    Errors are log-distances, like :attr:`H3Canvas.aspect_error`: 0.02 is about
    2% of scale, 0.35 is a 9:16 canvas against a square upload.
    """
    import math

    def error(width: int, height: int) -> float:
        if width <= 0 or height <= 0 or canvas.width <= 0 or canvas.height <= 0:
            return 0.0
        return round(abs(math.log((canvas.width / canvas.height) / (width / height))), 6)

    first_w, first_h = probe_image_dimensions(image_path)
    report: dict[str, Any] = {
        "keyframe_fit": "cover_crop_center",
        "first_keyframe_aspect_error": error(first_w, first_h),
        "last_keyframe_aspect_error": None,
        "keyframe_pair_aspect_error": None,
    }
    if last_frame_path is not None and last_frame_path.exists():
        last_w, last_h = probe_image_dimensions(last_frame_path)
        report["last_keyframe_aspect_error"] = error(last_w, last_h)
        if first_w > 0 and first_h > 0 and last_w > 0 and last_h > 0:
            report["keyframe_pair_aspect_error"] = round(
                abs(math.log((first_w / first_h) / (last_w / last_h))), 6
            )
    return report


def fit_video_dimensions(
    input_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Scale + centre-crop a rendered clip to exactly ``width`` x ``height``.

    Cover-fit on purpose: ``force_original_aspect_ratio=increase`` then ``crop``
    discards the sliver that overflows, so pixels stay square. A fit that
    stretched would put back the very distortion the lattice canvas in
    :func:`_grid_pair` was chosen to avoid; that canvas is never more than a
    couple of percent off this aspect, so the crop is a handful of pixels.
    Audio is stream-copied - the mux is untouched.
    """
    import subprocess

    if width <= 0 or height <= 0:
        raise ValueError(f"fit_video_dimensions requires positive dimensions, got {width}x{height}")
    started = time.perf_counter()
    filter_graph = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            filter_graph,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    return {
        "delivered_width": int(width),
        "delivered_height": int(height),
        "fit_wall_seconds": round(time.perf_counter() - started, 3),
    }


def ensure_comfyui_runtime() -> Path:
    comfy_root = comfyui_runtime_dir()
    if not (comfy_root / "main.py").exists():
        raise FileNotFoundError(
            f"ComfyUI runtime is not provisioned: {comfy_root} has no main.py "
            "(download data/comfyui, or set H3_COMFYUI_DIR to a ComfyUI checkout)."
        )
    return comfy_root


def ensure_minimax_h3_models(
    comfy_root: Path | None = None,
    *,
    generate_audio: bool = True,
    recipe: H3Recipe | None = None,
) -> dict[str, Path]:
    comfy_root = comfy_root or ensure_comfyui_runtime()
    required = {
        "unet": (comfy_root / "models" / "diffusion_models" / unet_name()),
        "clip": (comfy_root / "models" / "text_encoders" / clip_name()),
        "video_vae": (comfy_root / "models" / "vae" / video_vae_name()),
    }
    lora_name = (recipe or env_recipe()).lora_name
    if lora_name:
        required["turbo_lora"] = comfy_root / "models" / "loras" / lora_name
    if generate_audio:
        required["audio_vae"] = comfy_root / "models" / "vae" / audio_vae_name()
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"MiniMax H3 weights are missing under {comfy_root / 'models'}: " + ", ".join(missing)
        )
    return required


def ensure_minimax_h3_turbo_node(comfy_root: Path | None = None) -> Path:
    """Check the Turbo node (LoRA loader + sampler) is installed in the runtime.

    The whole recipe depends on it: without ``MiniMaxH3TurboLoRA`` there is no
    4-step distillation and no ``low_vram`` merge, and without
    ``MiniMaxH3TurboSampler`` the audio stream is stepped on the wrong clock and
    comes out garbled. It also ships the ``silu(t_emb)`` grid it re-injects for
    pruned/quantized bases, so that file is checked too.
    """
    comfy_root = comfy_root or ensure_comfyui_runtime()
    node_dir = comfy_root / "custom_nodes" / TURBO_NODE_DIRNAME
    required = {
        "__init__.py": node_dir / "__init__.py",
        "silu_temb_grid": node_dir / "h3_silu_temb_grid.safetensors",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"MiniMax H3 Turbo ComfyUI node is not provisioned at {node_dir} "
            "(https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo, v1.2.3 or "
            "newer - older versions garble the audio): " + ", ".join(missing)
        )
    return node_dir


def sparse_attention_unavailable(comfy_root: Path) -> str | None:
    """Why this worker cannot sample with SLA attention, or ``None`` when it can.

    The SLA node never says so itself: it passes the model through unchanged
    when its package fails to import, and its Kitchen engine runs dense without
    ``comfy_kitchen.sol_attn`` (comfy-kitchen>=0.2.32 on a CUDA 13 torch, so not
    on the cu128 stack). Checking up front lets a job report what it ran.
    """
    node_dir = comfy_root / "custom_nodes" / SLA_NODE_DIRNAME
    if not (node_dir / "__init__.py").exists():
        return f"the SLA node is not provisioned at {node_dir}"
    import comfy.ldm.modules.attention as comfy_attention
    import comfy_kitchen
    import nodes as comfy_nodes

    if "H3SLAAttention" not in comfy_nodes.NODE_CLASS_MAPPINGS:
        return f"H3SLAAttention did not register from {node_dir}"
    if not comfy_attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE or not hasattr(comfy_kitchen, "sol_attn"):
        return "Comfy Kitchen INT8 attention or its sol_attn kernel is unavailable (needs comfy-kitchen>=0.2.32 on a CUDA 13 torch)"
    return None


def add_sparse_attention(graph: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Route the sampler's model through Kitchen INT8 attention + SLA (:data:`SLA_ATTENTION_INPUTS`)."""
    graph["60"] = {
        "class_type": "ModelAttentionBackend",
        "inputs": {"model": graph["11"]["inputs"]["model"], "attention": "comfy kitchen attention"},
    }
    graph["61"] = {"class_type": "H3SLAAttention", "inputs": {"model": ["60", 0], **SLA_ATTENTION_INPUTS}}
    for node_id in ("9", "11"):  # scheduler and guider
        graph[node_id]["inputs"]["model"] = ["61", 0]
    return graph


def ensure_minimax_h3_runtime_assets(*, generate_audio: bool = True) -> tuple[Path, dict[str, Path]]:
    comfy_root = ensure_comfyui_runtime()
    if not is_nvfp4_vanilla():
        ensure_minimax_h3_turbo_node(comfy_root)
    ensure_minimax_h3_keyframe_geometry(comfy_root)
    return comfy_root, ensure_minimax_h3_models(comfy_root, generate_audio=generate_audio)


def build_h3_graph(
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    length: int,
    first_frame_input: str,
    last_frame_input: str | None,
    filename_prefix: str,
    recipe: H3Recipe | None = None,
    scheduler: str | None = None,
    generate_audio: bool = True,
    fps: float = H3_FPS,
) -> dict[str, dict[str, Any]]:
    """Build the headless MiniMax H3 FL2VA prompt graph.

    For the ``int8_turbo`` variant this is the reference Turbo graph, node for
    node: ``UNETLoader -> MiniMaxH3TurboLoRA -> {BasicGuider, BasicScheduler} ->
    SamplerCustomAdvanced(<- MiniMaxH3TurboSampler)``. The LoRA node is the only
    thing between the model loader and the sampler, and the sampler is the
    Turbo sampler node rather than ``KSamplerSelect``.

    For the ``nvfp4_vanilla`` variant the graph is the stock local H3 chain:
    ``UNETLoader -> {BasicGuider, BasicScheduler} -> SamplerCustomAdvanced(<-
    KSamplerSelect(res_multistep))`` - no LoRA node, the pruned NVFP4
    checkpoint runs unmodified on its benchmarked 10-step recipe.

    ``first_frame_input``/``last_frame_input`` are LoadImage input values
    (relative to the ComfyUI input directory); pass ``None`` for
    ``last_frame_input`` to omit the last-frame keyframe.
    ``filename_prefix`` is the SaveVideo output prefix.
    Set ``generate_audio=False`` to drop the audio VAE decoder (no audio track);
    the FL2VA latent always carries the audio stream, so the default keeps it.

    ``recipe`` (a preset, default :func:`env_recipe`) picks the LoRA node (none,
    ``MiniMaxH3TurboLoRA`` or the stock ``LoraLoaderModelOnly``), the steps and the
    sampler node (the Turbo sampler or ``KSamplerSelect``).
    """
    recipe = recipe or env_recipe()
    scheduler_name = scheduler or h3_scheduler()
    graph: dict[str, dict[str, Any]] = {}

    graph["1"] = {
        "class_type": "LoadImage",
        "inputs": {"image": first_frame_input},
    }
    graph["2"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": unet_name(), "weight_dtype": "default"},
    }
    model_source: list[Any] = ["2", 0]
    if recipe.lora_name and recipe.lora_loader == "MiniMaxH3TurboLoRA":
        graph["18"] = {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {
                "model": ["2", 0],
                "lora_name": recipe.lora_name,
                "strength": turbo_strength(),
                "low_vram": turbo_low_vram(),
            },
        }
        model_source = ["18", 0]
    elif recipe.lora_name:
        graph["18"] = {
            "class_type": recipe.lora_loader,
            "inputs": {"model": ["2", 0], "lora_name": recipe.lora_name, "strength_model": 1.0},
        }
        model_source = ["18", 0]
    graph["3"] = {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": clip_name(),
            "type": "minimax",
            "device": "default",
        },
    }
    graph["4"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": video_vae_name()},
    }
    # MiniMaxH3ImageToVideo outputs (positive conditioning, AV latent).
    h3_inputs: dict[str, Any] = {
        "clip": ["3", 0],
        "vae": ["4", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
        "first_frame": ["1", 0],
    }
    if last_frame_input is not None:
        graph["7"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_frame_input},
        }
        h3_inputs["last_frame"] = ["7", 0]
    graph["8"] = {
        "class_type": "MiniMaxH3ImageToVideo",
        "inputs": h3_inputs,
    }
    graph["9"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_source,
            "scheduler": scheduler_name,
            "steps": recipe.steps,
            "denoise": 1.0,
        },
    }
    graph["10"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }
    graph["11"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model_source, "conditioning": ["8", 0]},
    }
    if recipe.sampler == "turbo":
        graph["12"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
    else:
        graph["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": recipe.sampler}}
    graph["13"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["10", 0],
            "guider": ["11", 0],
            "sampler": ["12", 0],
            "sigmas": ["9", 0],
            "latent_image": ["8", 1],
        },
    }
    graph["14"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["13", 0], "vae": ["4", 0]},
    }
    create_video_inputs: dict[str, Any] = {
        "images": ["14", 0],
        # The frame rate the time axis rendered at (h3.h3_interpolation.h3_time_axis), which the
        # generated audio lasts: 24 unless an interpolated job renders fewer frames a second.
        "fps": fps,
    }
    if generate_audio:
        graph["5"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae_name()},
        }
        graph["15"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["13", 0], "vae": ["5", 0]},
        }
        create_video_inputs["audio"] = ["15", 0]
    graph["16"] = {
        "class_type": "CreateVideo",
        "inputs": create_video_inputs,
    }
    graph[_SAVE_VIDEO_NODE_ID] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["16", 0],
            "filename_prefix": filename_prefix,
            "format": "mp4",
            "codec": {"codec": "h264", "encoding": {"mode": "re-encode", "crf": 18.0}},
        },
    }
    return graph


def add_frame_guides(graph: dict[str, dict[str, Any]], guides: list[tuple[int, str]], *, length: int) -> None:
    """Pin ``(frame_idx, LoadImage input)`` guide images through chained ``MiniMaxH3AddGuide`` nodes.

    Each guide conditions the clip at its pixel frame (an anchor, not an overwrite);
    ``frame_idx`` must sit inside ``[0, length)``.
    """
    conditioning: list[Any] = ["8", 0]
    for index, (frame_idx, image_input) in enumerate(guides):
        if not 0 <= int(frame_idx) < int(length):
            raise ValueError(f"guide frame {frame_idx} is outside the {length}-frame clip")
        load_id, guide_id = str(100 + index), str(200 + index)
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": image_input}}
        graph[guide_id] = {
            "class_type": "MiniMaxH3AddGuide",
            "inputs": {
                "positive": conditioning,
                "vae": ["4", 0],
                "latent": ["8", 1],
                "image": [load_id, 0],
                "frame_idx": int(frame_idx),
            },
        }
        conditioning = [guide_id, 0]
    graph["11"]["inputs"]["conditioning"] = conditioning


def pin_audio(graph: dict[str, dict[str, Any]], audio_input: str) -> None:
    """Pin ``audio_input`` in the AV latent: the audio VAE (node "5") encodes it and an all-zero
    ``SolidMask`` through ``SetLatentNoiseMask`` keeps it out of denoising, so the sampler sees the
    clean audio at every step and moves only the video. The audio is never decoded: the clip comes
    out video-only and the caller muxes the original audio back."""
    graph.pop("15", None)
    graph["16"]["inputs"].pop("audio", None)
    graph["40"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_input}}
    graph["41"] = {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["40", 0], "vae": ["5", 0]}}
    graph["42"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 8, "height": 8}}
    graph["43"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["41", 0], "mask": ["42", 0]}}
    graph["44"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {"video_latent": ["8", 1], "audio_latent": ["43", 0]},
    }
    graph["13"]["inputs"]["latent_image"] = ["44", 0]


def build_lipsync_graph(
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    length: int,
    first_frame_input: str,
    last_frame_input: str | None,
    guides: list[tuple[int, str]],
    audio_input: str,
    filename_prefix: str,
    recipe: H3Recipe = LIPSYNC_PRESETS[DEFAULT_LIPSYNC_PRESET],
    sparse_attention: bool = True,
) -> dict[str, dict[str, Any]]:
    """Pinned-audio lipsync: the Turbo graph re-rendering every frame to a fixed voice.

    The serving graph from :func:`build_h3_graph` with four changes:

    - the caller's audio is encoded by the audio VAE and *pinned*: an all-zero
      ``SolidMask`` through ``SetLatentNoiseMask`` keeps it out of denoising, so
      the sampler sees the clean speech at every step and moves only the video -
      the lips follow this voice instead of one the model invents;
    - keyframes condition the clip: the first frame (and optionally the last)
      through FL2VA, everything else through ``MiniMaxH3AddGuide`` at its frame
      index. A few anchors seconds apart hold identity, pose and framing without
      carrying old mouth shapes often enough to fight the audio;
    - the audio is never decoded: the clip is video-only and the worker muxes the
      caller's original audio afterwards, bit-exact;
    - the recipe's ``sigmas`` (``ManualSigmas``) replace the ``simple`` schedule;
      ``None`` keeps ``BasicScheduler(simple, steps)``.

    Defaults to the lipsync student's recipe (``turbo-3``: :data:`LIPSYNC_LORA_NAME`
    on :data:`LIPSYNC_SIGMAS`, sparse attention), the one this mode ships.
    ``guides`` are ``(frame_idx, LoadImage input)`` pairs inside ``[0, length)``.
    ``sparse_attention`` samples with :func:`add_sparse_attention`.
    """
    graph = build_h3_graph(
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        length=length,
        first_frame_input=first_frame_input,
        last_frame_input=last_frame_input,
        filename_prefix=filename_prefix,
        recipe=recipe,
        generate_audio=True,
    )
    add_frame_guides(graph, guides, length=length)
    pin_audio(graph, audio_input)
    if sparse_attention:
        add_sparse_attention(graph)
    if recipe.sigmas is not None:  # after the attention rewiring: ManualSigmas takes no model
        graph["9"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": recipe.sigmas}}
    return graph


class _ProgressSink:
    """The executor's "server": there is none here, so its messages go nowhere.

    Where the render has got to comes from the sampler's own progress bar instead
    (:func:`_report_sampling_progress`).
    """

    client_id = None
    last_node_id = None
    last_prompt_id = None
    sockets_metadata: dict = {}

    def send_sync(self, event: Any, data: Any, sid: Any = None) -> None:
        return


def _report_sampling_progress(value, total, preview_image=None, prompt_id=None, node_id=None) -> None:
    """One sampler step, as the job's progress: a clip is rendered in one window, step by step.

    ComfyUI's own server installs a hook like this one in its main; headless, we install ours.
    """
    if not total:
        return
    if value in (1, total):
        log_event("minimax_h3_sampling_progress", value=value, max=total)
    progress.step(int(value), int(total))


@dataclass
class _RetainedH3Runtime:
    comfy_root: Path
    initialized_at: float
    init_seconds: float
    executor: Any
    generate_audio: bool = False


_RETAINED_H3_RUNTIME: _RetainedH3Runtime | None = None
_RETAINED_H3_LOCK = threading.RLock()


def _load_comfy_modules(comfy_root: Path) -> None:
    """Import the ComfyUI runtime headlessly (mirrors the benchmark harness).

    Custom nodes are deliberately NOT disabled here: the Turbo LoRA loader and
    the Turbo sampler live in ``custom_nodes/ComfyUI-MiniMax-H3-Turbo``.
    """
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    sys.argv = [
        sys.argv[0],
        "--cache-classic",
        "--disable-api-nodes",
        # DynamicVRAM fills the card with H3 weight pages; without headroom every lipsync step re-faults pages
        # for its activations (warm 5 s lipsync on the 3090: 123-130 s at 0, 88 s at 2/4/8 GB for 16 fps;
        # 12 fps 84 s at 2, 72 s at 4/8). Helpers outside ComfyUI also take this room first (make_room_beside_h3).
        "--vram-headroom",
        os.environ.get("H3_VRAM_HEADROOM_GB", "4"),
        "--input-directory",
        str(comfy_root / "input"),
        "--output-directory",
        str(comfy_root / "output"),
    ]
    import main  # noqa: E402  (ComfyUI entry module from the runtime dir)

    main.apply_custom_paths()


def load_retained_h3_runtime(*, generate_audio: bool = True) -> Path:
    """Import ComfyUI and register its nodes now (no model weights load until a job runs).

    Returns the ComfyUI root; :func:`sparse_attention_unavailable` is only meaningful after this.
    """
    return _retained_runtime(generate_audio=generate_audio).comfy_root


def _retained_runtime(
    *,
    generate_audio: bool = False,
    recipe: H3Recipe | None = None,
    check_models: bool = True,
) -> _RetainedH3Runtime:
    global _RETAINED_H3_RUNTIME

    with _RETAINED_H3_LOCK:
        retained = _RETAINED_H3_RUNTIME
        if retained is not None:
            # The executor is graph-agnostic: a job without audio just leaves the
            # audio VAE out of its graph (callers check their own weights), so a
            # change of job type never reloads the shared weights.
            return retained

        comfy_root = ensure_comfyui_runtime()
        if not is_nvfp4_vanilla():
            ensure_minimax_h3_turbo_node(comfy_root)
        ensure_minimax_h3_keyframe_geometry(comfy_root)
        if check_models:
            ensure_minimax_h3_models(comfy_root, generate_audio=generate_audio, recipe=recipe)
        log_event("minimax_h3_runtime_load_started", comfy_root=str(comfy_root), generate_audio=generate_audio)
        started = time.perf_counter()
        _load_comfy_modules(comfy_root)
        (comfy_root / "input").mkdir(parents=True, exist_ok=True)
        (comfy_root / "output").mkdir(parents=True, exist_ok=True)

        async def _init() -> None:
            import execution as comfy_execution
            import nodes as comfy_nodes

            failed = await comfy_nodes.init_extra_nodes(
                init_custom_nodes=True,
                init_api_nodes=False,
            )
            if failed:
                log_event("minimax_h3_builtin_node_import_failures", nodes=sorted(failed))
            # Fail here, not inside validate_prompt, if the Turbo node did not
            # load: a missing LoRA/sampler turns the render into an un-lora'd
            # 6-step base run with garbled audio. The vanilla recipe needs no
            # custom node - KSamplerSelect is stock ComfyUI.
            required_nodes = () if is_nvfp4_vanilla() else ("MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler")
            missing = sorted(
                name for name in required_nodes
                if name not in comfy_nodes.NODE_CLASS_MAPPINGS
            )
            if missing:
                raise FileNotFoundError(
                    "ComfyUI custom node classes are not registered: "
                    + ", ".join(missing)
                    + f". Expected {TURBO_NODE_DIRNAME} under "
                    f"{comfy_root / 'custom_nodes'} to have loaded "
                    "(check the ComfyUI 'IMPORT FAILED' lines in the worker log)."
                )
            import comfy.utils

            comfy.utils.set_progress_bar_global_hook(_report_sampling_progress)
            executor = comfy_execution.PromptExecutor(
                _ProgressSink(),
                cache_type=comfy_execution.CacheType.CLASSIC,
                cache_args={"lru": 0, "ram": 0, "ram_inactive": 0},
            )
            _retained_runtime_state["executor"] = executor

        _retained_runtime_state: dict[str, Any] = {}
        asyncio.run(_init())

        init_seconds = time.perf_counter() - started
        retained = _RetainedH3Runtime(
            comfy_root=comfy_root,
            initialized_at=time.time(),
            init_seconds=round(init_seconds, 3),
            executor=_retained_runtime_state["executor"],
            generate_audio=generate_audio,
        )
        _RETAINED_H3_RUNTIME = retained
        log_event("minimax_h3_runtime_load_complete", comfy_root=str(comfy_root), seconds=retained.init_seconds, generate_audio=generate_audio)
        return retained


def _free_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return round(free / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        return None


def unload_retained_h3_runtime() -> bool:
    """Drop the retained ComfyUI executor and release all H3 models / VRAM."""
    global _RETAINED_H3_RUNTIME

    with _RETAINED_H3_LOCK:
        retained = _RETAINED_H3_RUNTIME
        _RETAINED_H3_RUNTIME = None

    if retained is None:
        return False

    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except Exception as exc:  # noqa: BLE001
        log_event("minimax_h3_runtime_unload_warning", error=str(exc))
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    log_event("minimax_h3_runtime_unloaded")
    return True


def make_room_beside_h3(need_bytes: int | None = None) -> None:
    """Free ``need_bytes`` (default ``HELPER_VRAM_BYTES``) of VRAM for a model run outside ComfyUI: ZipEnhancer,
    the CTC transcriber, S3FD/SyncNet, the glitch net, RIFE. ComfyUI's DynamicVRAM fills every byte nobody else
    holds with H3 weight pages and never yields them to a torch allocation, so on a full card a helper after an H3
    graph runs out of memory. ``free_memory`` drops only the missing bytes of weight pages (``partially_unload``):
    the models stay loaded and the pages come back on the next H3 step. No-op before the executor exists."""
    if _RETAINED_H3_RUNTIME is None:
        return
    import comfy.model_management as model_management

    device = model_management.get_torch_device()
    need = HELPER_VRAM_BYTES if need_bytes is None else int(need_bytes)
    free = model_management.get_free_memory(device)
    if free >= need:
        return
    began = time.perf_counter()
    model_management.free_memory(need, device)
    log_event("minimax_h3_room_made", need_mib=need >> 20, free_mib=free >> 20,
              now_free_mib=model_management.get_free_memory(device) >> 20,
              seconds=round(time.perf_counter() - began, 3))


def _execute_graph(
    retained: _RetainedH3Runtime,
    graph: dict[str, Any],
    *,
    job_id: str,
    save_node_id: str = _SAVE_VIDEO_NODE_ID,
) -> dict[str, Any]:
    """Validate and run one prompt graph on the retained executor.

    Returns the ``save_node_id`` result (``saved_subfolder``/``saved_filename``
    under the ComfyUI output dir) plus the pipeline wall time and CUDA peaks.
    """
    import torch

    prompt_id = str(uuid.uuid4())

    async def _execute() -> dict[str, Any]:
        import execution as comfy_execution

        valid, error, outputs, node_errors = await comfy_execution.validate_prompt(prompt_id, graph, None)
        if not valid:
            raise ValueError(f"MiniMax H3 graph validation failed: {error} {node_errors}")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        await retained.executor.execute_async(
            graph,
            prompt_id,
            extra_data={"client_id": None},
            execute_outputs=outputs,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - inference_started
        if not retained.executor.success:
            raise RuntimeError(f"MiniMax H3 ComfyUI execution failed: job_id={job_id}")
        # SaveVideo emits a PreviewVideo UI with key "images" (a list of SavedResult
        # dicts with filename/subfolder/type), as SaveImage does. See comfy_api/latest/_ui.py.
        saved_list = retained.executor.history_result["outputs"][save_node_id]["images"]
        saved = saved_list[0]
        return {
            "pipeline_seconds": elapsed,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            "saved_subfolder": saved.get("subfolder", ""),
            "saved_filename": saved["filename"],
        }

    return asyncio.run(_execute())


def run_minimax_h3_inference(
    *,
    job_id: str,
    prompt: str,
    image_path: Path,
    last_frame_path: Path | None,
    output_path: Path,
    seed: int,
    height: int | None = None,
    width: int | None = None,
    length: int,
    preset: str | None = None,
    generate_audio: bool = True,
    output_width: int | None = None,
    output_height: int | None = None,
    guides: list[tuple[int, Path]] | None = None,
    pinned_audio_path: Path | None = None,
    preview: Any = None,
    fps: int | None = None,
) -> dict[str, Any]:
    """Run one H3 FL2VA/i2v generation; returns stage metrics.

    ``preview`` (:class:`h3.h3_restore.FramePreview`) renders only that frame of
    this graph to its PNG instead of the clip (``output_path`` unused); returns
    ``{preview, frame, pipeline_seconds, width, height}``.

    ``guides`` are ``(frame_idx, image)`` pins (:func:`add_frame_guides`).
    ``pinned_audio_path`` pins that audio in the latent (:func:`pin_audio`; it should
    last the snapped clip): the video moves to it and the mp4 comes out silent.

    ``length`` is the requested frame count at 24 fps; the node's own
    ``temporal_shape`` snaps it up to the model's 17k+5 frame grid. ``preset`` is a
    :data:`VIDEO_PRESETS` name; ``None`` runs the worker's :func:`env_recipe`
    (``turbo-4`` on the default variant).
    ``generate_audio`` defaults to ``True``: FL2VA carries a synchronized audio
    stream and the Turbo Sampler is what keeps it intelligible.

    ``width``/``height`` are what the caller asked for, not necessarily what H3
    can render: :func:`resolve_canvas` owns the final canvas (off-lattice or
    oversized numbers move to the lattice pair that holds this aspect), and
    ``None`` means "use the input image's own shape". Keyframes are copied to the
    ComfyUI input directory **as uploaded** - no pre-crop, no resize - so the
    graph behaves exactly like the reference workflow; the node crops them to the
    canvas itself.

    ``output_width``/``output_height`` are the *delivered* frame size. The
    render itself has to sit on the lattice, so when a caller asked for an exact
    size we finish with a uniform scale plus a sub-percent centre crop
    (:func:`fit_video_dimensions`) instead of handing back 704x1248 for a
    720x1280 request.

    ``fps`` below 24 renders that many frames a second of the clip
    (:func:`h3.h3_interpolation.h3_time_axis`, an interpolated job with no pinned
    audio): the frames and the generated audio last ``length / fps`` seconds.
    """
    from .h3_interpolation import h3_time_axis

    time_axis_fps = int(fps or H3_FPS)
    canvas = resolve_canvas(image_path=image_path, requested_width=width, requested_height=height)
    width = canvas.width
    height = canvas.height

    recipe = h3_preset(preset, VIDEO_PRESETS) if preset else env_recipe()
    turbo_lora = recipe.lora_loader == "MiniMaxH3TurboLoRA" and recipe.lora_name is not None
    engine_name = h3_engine_name()
    started = time.perf_counter()
    # A pinned audio needs the audio VAE as its encoder.
    audio_vae = generate_audio or pinned_audio_path is not None
    retained = _retained_runtime(generate_audio=audio_vae, recipe=recipe)
    comfy_root = retained.comfy_root
    # The retained runtime may have been loaded for another recipe; check this one's LoRA.
    ensure_minimax_h3_models(comfy_root, generate_audio=audio_vae, recipe=recipe)
    stage_timings: dict[str, float] = {"minimax_h3_runtime_init_seconds": retained.init_seconds}

    def finish_stage(name: str, began: float) -> None:
        seconds = time.perf_counter() - began
        stage_timings[name] = round(seconds, 3)
        log_event("minimax_h3_stage_complete", job_id=job_id, stage=name, seconds=round(seconds, 3))

    job_input_dir = comfy_root / "input" / "jobs" / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    # Distinct node-side names: the same upload used for both keyframes (the
    # normal FL2VA-off-by-default case) and two different files that happen to
    # share a basename must both survive the copy.
    first_frame_input = f"jobs/{job_id}/first_frame{image_path.suffix}"
    shutil.copyfile(image_path, comfy_root / "input" / first_frame_input)
    last_frame_input: str | None = None
    if last_frame_path is not None:
        last_frame_input = f"jobs/{job_id}/last_frame{last_frame_path.suffix}"
        shutil.copyfile(last_frame_path, comfy_root / "input" / last_frame_input)

    filename_prefix = f"jobs/{job_id}/output"
    geometry = keyframe_geometry_report(
        image_path=image_path, last_frame_path=last_frame_path, canvas=canvas
    )
    log_event("minimax_h3_keyframe_geometry", job_id=job_id, **geometry)
    pair_error = geometry["keyframe_pair_aspect_error"]
    if pair_error is not None and pair_error > H3_KEYFRAME_ASPECT_WARN:
        # Not fatal - the node crops both frames the same way - but this is the
        # case where "the last frame looks zoomed/shifted" has an answer in the
        # inputs rather than in the model.
        log_event(
            "minimax_h3_keyframe_aspect_mismatch",
            job_id=job_id,
            keyframe_pair_aspect_error=pair_error,
            first_keyframe_aspect_error=geometry["first_keyframe_aspect_error"],
            last_keyframe_aspect_error=geometry["last_keyframe_aspect_error"],
            threshold=max(H3_ASPECT_TOLERANCE, H3_KEYFRAME_ASPECT_WARN),
        )
    graph = build_h3_graph(
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        length=length,
        first_frame_input=first_frame_input,
        last_frame_input=last_frame_input,
        filename_prefix=filename_prefix,
        recipe=recipe,
        generate_audio=audio_vae,
        fps=float(time_axis_fps),
    )
    if guides:
        staged = []
        for index, (frame_idx, path) in enumerate(guides):
            relative = f"jobs/{job_id}/guide{index}{path.suffix}"
            shutil.copyfile(path, comfy_root / "input" / relative)
            staged.append((int(frame_idx), relative))
        add_frame_guides(graph, staged, length=length)
    if pinned_audio_path is not None:
        relative = f"jobs/{job_id}/pinned_audio{pinned_audio_path.suffix}"
        shutil.copyfile(pinned_audio_path, comfy_root / "input" / relative)
        pin_audio(graph, relative)
    if preview is not None:
        # The node renders the length snapped up to its grid; the preview indexes those frames.
        try:
            with h3_time_axis(time_axis_fps):
                result = _run_frame_preview(
                    retained, graph, preview, job_id=job_id, length=5 + 17 * -(-max(0, int(length) - 5) // 17),
                    filename_prefix=f"jobs/{job_id}/preview",
                )
        finally:
            _cleanup_job_inputs(job_input_dir)
        return {**result, "width": canvas.width, "height": canvas.height}
    with h3_time_axis(time_axis_fps):
        saved_result = _execute_graph(retained, graph, job_id=job_id)
    stage_timings["minimax_h3_pipeline_seconds"] = round(saved_result["pipeline_seconds"], 3)
    log_event(
        "minimax_h3_inference_complete",
        job_id=job_id,
        seed=int(seed),
        requested_length=int(length),
        seconds=round(saved_result["pipeline_seconds"], 3),
        warm=True,
        generate_audio=generate_audio,
        h3_variant=h3_variant(),
        preset=preset,
        steps=recipe.steps,
        lora=recipe.lora_name,
    )

    comfy_output_dir = comfy_root / "output"
    saved_path = comfy_output_dir / saved_result["saved_subfolder"] / saved_result["saved_filename"]
    move_started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(saved_path), output_path)
    generated_dir = comfy_output_dir / "jobs" / job_id
    if generated_dir.exists():
        shutil.rmtree(generated_dir, ignore_errors=True)
    _cleanup_job_inputs(comfy_root / "input" / "jobs" / job_id)
    if not output_path.exists():
        raise FileNotFoundError(f"MiniMax H3 inference did not produce output at {output_path}")

    finish_stage("minimax_h3_output_move_seconds", move_started)

    # Deliver the frame size the caller asked for. The lattice canvas is within a
    # couple of percent of that aspect (_grid_pair buys ratio with at most
    # H3_CANVAS_AREA_SLACK of pixels), so this is a uniform scale plus a small
    # centre crop - never a squeeze.
    resize_metrics: dict[str, Any] | None = None
    delivered = (int(output_width or 0), int(output_height or 0))
    if delivered[0] > 0 and delivered[1] > 0 and delivered != (canvas.width, canvas.height):
        began = time.perf_counter()
        fit_path = output_path.with_name(f"{output_path.stem}.fit{delivered[0]}x{delivered[1]}.mp4")
        resize_metrics = fit_video_dimensions(
            output_path, fit_path, width=delivered[0], height=delivered[1]
        )
        os.replace(fit_path, output_path)
        finish_stage("minimax_h3_output_fit_seconds", began)
        log_event(
            "minimax_h3_output_resized",
            job_id=job_id,
            canvas_width=canvas.width,
            canvas_height=canvas.height,
            canvas_aspect_error=canvas.aspect_error,
            # How much of the frame the cover-fit above scaled/cropped away.
            requested_aspect_error=canvas.requested_aspect_error,
            **resize_metrics,
        )
    delivered_width, delivered_height = (
        delivered if resize_metrics is not None else (canvas.width, canvas.height)
    )
    return {
        "backend": "minimax_h3",
        "engine": engine_name,
        "warm": True,
        "seed": int(seed),
        "requested_length": int(length),
        "canvas_source": canvas.source,
        "canvas_aspect_error": canvas.aspect_error,
        "requested_aspect_error": canvas.requested_aspect_error,
        "width": canvas.width,
        "height": canvas.height,
        "delivered_width": delivered_width,
        "delivered_height": delivered_height,
        "output_resize": resize_metrics,
        **geometry,
        "fps": float(time_axis_fps),
        "preset": preset,
        "steps": recipe.steps,
        "sampler": recipe.sampler_label,
        "scheduler": h3_scheduler(),
        "unet": unet_name(),
        "text_encoder": clip_name(),
        "h3_variant": h3_variant(),
        "lora": recipe.lora_name,
        "turbo_strength": turbo_strength() if turbo_lora else None,
        "turbo_low_vram": turbo_low_vram() if turbo_lora else None,
        "generate_audio": generate_audio,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "pipeline_seconds": stage_timings["minimax_h3_pipeline_seconds"],
        "stage_timings": stage_timings,
        "cuda_peak_allocated_gib": saved_result["cuda_peak_allocated_gib"],
        "cuda_peak_reserved_gib": saved_result["cuda_peak_reserved_gib"],
        "output": str(output_path),
    }


def _run_frame_preview(
    retained: _RetainedH3Runtime,
    graph: dict[str, dict[str, Any]],
    preview: Any,
    *,
    job_id: str,
    length: int,
    filename_prefix: str,
) -> dict[str, Any]:
    """Run ``graph`` as a one-frame preview (:func:`h3.h3_restore.preview_graph`); moves the PNG to
    ``preview.output_path``."""
    import nodes as comfy_nodes

    from . import h3_restore

    h3_restore.register_nodes(comfy_nodes)
    save_node_id = h3_restore.preview_graph(
        graph, frame=int(preview.frame), length=int(length), filename_prefix=filename_prefix, sigmas=preview.sigmas
    )
    saved = _execute_graph(retained, graph, job_id=job_id, save_node_id=save_node_id)
    preview.output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(
        str(retained.comfy_root / "output" / saved["saved_subfolder"] / saved["saved_filename"]), preview.output_path
    )
    log_event(
        "minimax_h3_frame_preview_complete",
        job_id=job_id,
        frame=int(preview.frame),
        length=int(length),
        sigmas=preview.sigmas,
        seconds=round(saved["pipeline_seconds"], 3),
    )
    return {"preview": str(preview.output_path), "frame": int(preview.frame), "pipeline_seconds": round(saved["pipeline_seconds"], 3)}


def run_lipsync_chunk(
    *,
    job_id: str,
    chunk_index: int,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    length: int,
    first_frame_path: Path,
    last_frame_path: Path | None,
    guides: list[tuple[int, Path]],
    audio_path: Path,
    output_path: Path,
    recipe: H3Recipe = LIPSYNC_PRESETS[DEFAULT_LIPSYNC_PRESET],
    sparse_attention: bool = True,
    preview: Any = None,
) -> dict[str, Any]:
    """Render one pinned-audio lipsync chunk with ``recipe`` (default: the lipsync student).

    ``preview`` (:class:`h3.h3_restore.FramePreview`) renders only that frame
    instead, to its PNG (``output_path`` unused).

    ``width``/``height`` must already sit on the H3 lattice (see
    :func:`resolve_canvas`) and ``length`` on the node's 17k+5 grid, so the
    render has exactly the frames the caller planned; ``audio_path`` must last
    ``length / H3_FPS`` seconds. Writes a video-only mp4 to ``output_path``.
    """
    retained = _retained_runtime(generate_audio=True, recipe=recipe)
    comfy_root = retained.comfy_root
    # The retained runtime may have been loaded by a videogen job; check this
    # recipe's weights per chunk.
    ensure_minimax_h3_turbo_node(comfy_root)
    ensure_minimax_h3_models(comfy_root, generate_audio=True, recipe=recipe)
    if sparse_attention:
        reason = sparse_attention_unavailable(comfy_root)
        if reason is not None:
            # Like the cu128 stack's other kernels: the slower path, not a failure.
            log_event("minimax_h3_sparse_attention_unavailable", job_id=job_id, chunk=chunk_index, reason=reason)
            sparse_attention = False

    relative_dir = f"jobs/{job_id}"
    input_dir = comfy_root / "input" / relative_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"c{chunk_index:03d}"

    def stage(path: Path, name: str) -> str:
        relative = f"{relative_dir}/{prefix}_{name}{path.suffix}"
        shutil.copyfile(path, comfy_root / "input" / relative)
        return relative

    graph = build_lipsync_graph(
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        length=length,
        first_frame_input=stage(first_frame_path, "first"),
        last_frame_input=stage(last_frame_path, "last") if last_frame_path is not None else None,
        guides=[(frame_idx, stage(path, f"guide{index}")) for index, (frame_idx, path) in enumerate(guides)],
        audio_input=stage(audio_path, "audio"),
        filename_prefix=f"{relative_dir}/{prefix}",
        recipe=recipe,
        sparse_attention=sparse_attention,
    )
    if preview is not None:
        try:
            return _run_frame_preview(
                retained, graph, preview, job_id=job_id, length=length, filename_prefix=f"{relative_dir}/{prefix}_preview"
            )
        finally:
            for staged in input_dir.glob(f"{prefix}_*"):
                staged.unlink(missing_ok=True)
    saved_result = _execute_graph(retained, graph, job_id=job_id)
    saved_path = comfy_root / "output" / saved_result["saved_subfolder"] / saved_result["saved_filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(saved_path), output_path)
    for staged in input_dir.glob(f"{prefix}_*"):
        staged.unlink(missing_ok=True)
    log_event(
        "lipsync_chunk_complete",
        job_id=job_id,
        chunk=chunk_index,
        frames=int(length),
        sparse_attention=sparse_attention,
        seconds=round(saved_result["pipeline_seconds"], 3),
        cuda_peak_reserved_gib=round(saved_result["cuda_peak_reserved_gib"], 2),
    )
    return {
        "chunk": chunk_index,
        "frames": int(length),
        "seed": int(seed),
        "guides": [int(frame_idx) for frame_idx, _ in guides],
        "sparse_attention": sparse_attention,
        "pipeline_seconds": round(saved_result["pipeline_seconds"], 3),
        "cuda_peak_allocated_gib": round(saved_result["cuda_peak_allocated_gib"], 3),
        "cuda_peak_reserved_gib": round(saved_result["cuda_peak_reserved_gib"], 3),
        "runtime_init_seconds": retained.init_seconds,
    }


def build_h3_still_graph(
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    preset: str,
    reference_inputs: list[str],
    filename_prefix: str,
) -> dict[str, dict[str, Any]]:
    """One H3 still: ``{UNET, CLIP, VAE} -> [LoRA] -> VigenH3StillRender(<- LoadImage x N) -> SaveImage``.

    The loaders keep the node ids and inputs of :func:`build_h3_graph`, so the
    retained executor's cache hands the still the base model, encoder and VAE
    that videogen/lipsync already loaded. ``preset`` (``h3_still.STILL_PRESETS``)
    sets steps, sampler, scheduler and the optional Turbo LoRA; both LoRA
    loaders patch a clone, never the cached base.
    ``reference_inputs`` are LoadImage values (relative to the ComfyUI input
    directory) in ``<Picture 1>``..``<Picture N>`` order.
    """
    from . import h3_still

    spec = h3_still.still_preset(preset)

    if len(reference_inputs) > h3_still.MAX_REFERENCES:
        raise ValueError(f"H3 conditions on at most {h3_still.MAX_REFERENCES} reference images")
    graph: dict[str, dict[str, Any]] = {
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name(), "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name(), "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_name()}},
    }
    model: list[Any] = ["2", 0]
    lora = spec["lora"]
    if lora is not None:
        graph["18"] = {
            "class_type": lora["class_type"],
            "inputs": {"model": ["2", 0], "lora_name": lora["name"], **lora["inputs"]},
        }
        model = ["18", 0]
    still_inputs: dict[str, Any] = {
        "model": model,
        "clip": ["3", 0],
        "vae": ["4", 0],
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
        "steps": int(spec["steps"]),
        "seed": int(seed),
        "sampler_name": spec["sampler"],
        "scheduler": spec["scheduler"],
    }
    for index, value in enumerate(reference_inputs, start=1):
        node_id = str(30 + index)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": value}}
        still_inputs[f"reference_{index}"] = [node_id, 0]
    graph["20"] = {"class_type": h3_still.STILL_NODE_CLASS, "inputs": still_inputs}
    graph[_SAVE_IMAGE_NODE_ID] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["20", 0], "filename_prefix": filename_prefix},
    }
    return graph


def run_minimax_h3_still(
    *,
    job_id: str,
    prompt: str,
    reference_paths: list[Path],
    output_path: Path,
    width: int,
    height: int,
    seed: int,
    preset: str,
) -> dict[str, Any]:
    """Render one still on the retained H3 runtime with ``preset``; returns stage metrics.

    ``width``/``height`` must already be an H3 canvas (32-pixel grid, at most
    about 1 MP; see :mod:`h3.h3_still`).
    """
    from . import h3_still

    spec = h3_still.still_preset(preset)
    h3_still.validate_canvas(int(width), int(height))
    # Take whichever H3 runtime is loaded (videogen and lipsync both keep the
    # audio VAE), so an image job never tears down their models.
    retained = _retained_runtime(generate_audio=True, check_models=False)
    comfy_root = retained.comfy_root
    for name, path in {
        "unet": comfy_root / "models" / "diffusion_models" / unet_name(),
        "clip": comfy_root / "models" / "text_encoders" / clip_name(),
        "video_vae": comfy_root / "models" / "vae" / video_vae_name(),
        **({"lora": comfy_root / "models" / "loras" / spec["lora"]["name"]} if spec["lora"] else {}),
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"MiniMax H3 {name} for still images is missing: {path}")
    import nodes as comfy_nodes

    h3_still.register_nodes(comfy_nodes)

    job_input_dir = comfy_root / "input" / _OUTPUT_SUBDIR_PREFIX / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    reference_inputs = []
    for index, path in enumerate(reference_paths, start=1):
        value = f"{_OUTPUT_SUBDIR_PREFIX}/{job_id}/picture_{index}{path.suffix or '.png'}"
        shutil.copyfile(path, comfy_root / "input" / value)
        reference_inputs.append(value)
    graph = build_h3_still_graph(
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        preset=preset,
        reference_inputs=reference_inputs,
        filename_prefix=f"{_OUTPUT_SUBDIR_PREFIX}/{job_id}/still",
    )
    try:
        saved = _execute_graph(retained, graph, job_id=job_id, save_node_id=_SAVE_IMAGE_NODE_ID)
        saved_path = comfy_root / "output" / saved["saved_subfolder"] / saved["saved_filename"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(saved_path), output_path)
    finally:
        cleanup_h3_job_files(job_id)
    log_event(
        "minimax_h3_still_complete",
        job_id=job_id,
        seed=int(seed),
        width=int(width),
        height=int(height),
        preset=preset,
        references=len(reference_paths),
        seconds=round(saved["pipeline_seconds"], 3),
    )
    return {
        "engine": "comfyui_minimax_h3_still",
        "unet": unet_name(),
        "preset": preset,
        "steps": int(spec["steps"]),
        "sampler": spec["sampler"],
        "scheduler": spec["scheduler"],
        "lora": spec["lora"]["name"] if spec["lora"] else None,
        "pipeline_seconds": round(saved["pipeline_seconds"], 3),
        "cuda_peak_allocated_gib": round(saved["cuda_peak_allocated_gib"], 3),
        "cuda_peak_reserved_gib": round(saved["cuda_peak_reserved_gib"], 3),
        "runtime_init_seconds": retained.init_seconds,
    }


def cleanup_h3_job_files(job_id: str) -> None:
    """Remove a job's staged inputs and leftover outputs from the ComfyUI tree."""
    comfy_root = comfyui_runtime_dir()
    for base in (comfy_root / "input", comfy_root / "output"):
        shutil.rmtree(base / "jobs" / job_id, ignore_errors=True)


def _cleanup_job_inputs(job_input_dir: Path) -> None:
    try:
        shutil.rmtree(job_input_dir)
    except FileNotFoundError:
        pass
