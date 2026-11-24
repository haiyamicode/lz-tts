"""Long H3 videos as a chain of parts joined in latent space, decoded once.

A part is an ordinary videogen render whose first 22 frames (7 video tokens, the 37-odd 40 Hz audio
steps under them) are the previous part's last 22, written into its latent with noise mask 0 and
added as its frame-0 keyframe (:class:`LzH3LatentHandover`). Nothing is decoded and re-encoded at
a join, so no VAE round trip drifts the picture. After sampling, a part's latent is matched to part
0's (:func:`refine`: per-channel mean/std of its body plus a high-band gain to part 0's high-band
sd, noEmbryo's H3ClipRefiner without its detail resample), which keeps a long chain from washing
out or sharpening part by part, and saved (:class:`LzH3LatentSave`); its tail is the next part's
head. The chain is decoded once (:class:`LzH3ChainDecode`): part 0 whole, then every later part
after its handed-over head, one video VAE pass streaming frames into ffmpeg (VRAM holds one
temporal chunk, host RAM one chunk of frames) and one audio VAE pass. The decoder's causal context
runs across every join.

Measured (local/exp/h3-joiner, arm E_chain, 4 x 5 s at 768x672 turbo-8, 2026-11-24): seam diff
1.74 / 1.14 / 1.25 of the clips' median frame step, sharpness vs part 0 1.06 / 1.18 / 1.01; decoding
430 frames once took ~100 s on the dev 3090.

The H3 grid repeats (1, 4, 4, 4, 4) frames a token every 5 tokens (17 frames), so 22 = 17 + 5
frames starting on a 5-token boundary map token for token onto a new part's first tokens: a part
of 17u + 5 frames adds 17(u - 1) new ones.

Latent files are safetensors (``video`` [1, 24, T, h/16, w/16], ``audio`` [1, 32, 2, Ta], fp32)
with string metadata ``format`` (:data:`LATENT_FORMAT`), ``version``, ``frames``, ``width``,
``height``, ``head_tokens`` / ``head_audio_steps`` (what the chain decode skips: 0 for part 0).
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cancel
import progress

from .compat import log_event

LATENT_FORMAT = "lzai-h3-chain-latent"
LATENT_VERSION = "1"
#: Frames a part takes over from the part before it: one 17-frame unit plus the 5-frame head.
HANDOVER_FRAMES = 22
#: Video tokens of the handover (and the context the refiner leaves out of a part's body).
HANDOVER_TOKENS = 7
UNIT_FRAMES = 17
FPS = 24

HANDOVER_NODE = "LzH3LatentHandover"
SAVE_NODE = "LzH3LatentSave"
DECODE_NODE = "LzH3ChainDecode"
#: Graph ids of the chain nodes in a videogen graph, clear of SLA (60, 61), add-on LoRAs (70+), guides (100+, 200+).
HANDOVER_NODE_ID = "300"
SAVE_NODE_ID = "301"


def grid_frames(frames: int) -> int:
    """``frames`` snapped up to H3's 17k + 5 grid."""
    return 5 + UNIT_FRAMES * max(0, math.ceil((int(frames) - 5) / UNIT_FRAMES))


def continuation_frames(new_frames: int) -> int:
    """Frames a later part renders to add ``new_frames`` (snapped up to whole units) after its handover."""
    return HANDOVER_FRAMES + UNIT_FRAMES * max(1, math.ceil(int(new_frames) / UNIT_FRAMES))


def plan_parts(frames: int, max_part_frames: int) -> list[int]:
    """Each part's frame count (17u + 5, handover included) for a ``frames``-long chain.

    The fewest parts whose renders fit ``max_part_frames``, units spread evenly (longer parts first),
    together covering ``frames`` snapped up to the grid exactly: n parts of u_i units cover
    sum(u_i) - (n - 1) units.
    """
    units = max(1, math.ceil((int(frames) - 5) / UNIT_FRAMES))
    max_units = (int(max_part_frames) - 5) // UNIT_FRAMES
    if max_units < 2:
        raise ValueError(f"a part of {max_part_frames} frames cannot hand over {HANDOVER_FRAMES} and add any")
    if units <= max_units:
        return [5 + UNIT_FRAMES * units]
    count = 1 + math.ceil((units - max_units) / (max_units - 1))
    base, extra = divmod(units + count - 1, count)
    return [5 + UNIT_FRAMES * (base + (1 if index < extra else 0)) for index in range(count)]


def _frames_of_tokens(tokens: int) -> int:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN

    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(tokens)))


def handover_span(previous_video_tokens: int, previous_audio_steps: int) -> tuple[int, int, int]:
    """(first video token, first audio step, audio steps) of a latent's last 22 frames."""
    from comfy.ldm.minimax.model import FRAME_RESCALE

    start_token = int(previous_video_tokens) - HANDOVER_TOKENS
    if start_token < 0 or start_token % 5:
        raise ValueError(f"a {previous_video_tokens}-token latent has no {HANDOVER_FRAMES}-frame tail on the grid")
    audio_start = round(FRAME_RESCALE * (_frames_of_tokens(previous_video_tokens) - HANDOVER_FRAMES))
    return start_token, audio_start, int(previous_audio_steps) - audio_start


# --- latent files ---------------------------------------------------------------------------------


def save_latent(path: Path, video: Any, audio: Any, *, head_tokens: int, head_audio_steps: int) -> dict[str, str]:
    from safetensors.torch import save_file

    metadata = {
        "format": LATENT_FORMAT,
        "version": LATENT_VERSION,
        "frames": str(_frames_of_tokens(video.shape[2])),
        "width": str(int(video.shape[-1]) * 16),
        "height": str(int(video.shape[-2]) * 16),
        "fps": str(FPS),
        "head_tokens": str(int(head_tokens)),
        "head_audio_steps": str(int(head_audio_steps)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    save_file({"video": video.detach().float().cpu().contiguous(), "audio": audio.detach().float().cpu().contiguous()},
              str(tmp), metadata=metadata)
    tmp.replace(path)
    return metadata


def read_metadata(path: Path) -> dict[str, Any]:
    """A chain latent's metadata (ints where numeric); a file that is not one is a ``ValueError``."""
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt") as handle:
            raw = handle.metadata() or {}
            keys = set(handle.keys())
    except Exception as exc:  # noqa: BLE001 - any unreadable file is the caller's bad input
        raise ValueError(f"{path.name} is not an H3 chain latent: {exc}") from exc
    if raw.get("format") != LATENT_FORMAT or not {"video", "audio"} <= keys:
        raise ValueError(f"{path.name} is not an H3 chain latent")
    if raw.get("version") != LATENT_VERSION:
        raise ValueError(f"{path.name} is chain latent version {raw.get('version')}, expected {LATENT_VERSION}")
    return {key: int(value) if value.isdigit() else value for key, value in raw.items()}


def load_latent(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    from safetensors.torch import load_file

    metadata = read_metadata(path)
    tensors = load_file(str(path))
    return tensors["video"], tensors["audio"], metadata


# --- refiner ----------------------------------------------------------------------------------------


def _band_split(t: Any, sigma: float = 2.0) -> tuple[Any, Any]:
    """Separable Gaussian low/high split over the last two dims (replicate padding) -> (lo, hi)."""
    import torch
    import torch.nn.functional as F

    r = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=t.device)
    k = torch.exp(-(x * x) / (2.0 * sigma**2))
    k = (k / k.sum()).to(t.dtype)
    h, w = t.shape[-2:]
    flat = t.reshape(-1, 1, h, w)
    lo = F.conv2d(F.pad(flat, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    lo = F.conv2d(F.pad(lo, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1)).reshape(t.shape)
    return lo, t - lo


def refine(video: Any, reference: Any, context_tokens: int = HANDOVER_TOKENS) -> tuple[Any, dict[str, float]]:
    """``video`` matched to ``reference`` (part 0's latent), both [1, C, T, h, w].

    The body (minus ``context_tokens`` at both ends) gets a per-channel affine to the reference
    body's mean/std, then a high-band gain to the reference's high-band sd (mean over sigma 1, 2, 3,
    clamped to [0.05, 5]); head and tail are matched to the corrected body with the same gain.
    """
    import torch

    v, r = video[0].float(), reference[0].to(video.device).float()
    c, t, h, w = v.shape
    o = int(context_tokens)
    if t <= 2 * o or r.shape[1] <= 2 * o:
        return video, {"refined": 0.0}
    head, body, tail = v[:, :o], v[:, o:t - o], v[:, t - o:]
    rbody = r[:, o:r.shape[1] - o]
    a = rbody.std(dim=(1, 2, 3)) / body.std(dim=(1, 2, 3)).clamp(min=1e-6)
    b = rbody.mean(dim=(1, 2, 3)) - a * body.mean(dim=(1, 2, 3))
    corrected = body * a.view(c, 1, 1, 1) + b.view(c, 1, 1, 1)
    target = sum(float(_band_split(rbody.reshape(-1, *rbody.shape[-2:]), s)[1].std()) for s in (1.0, 2.0, 3.0)) / 3
    lo, hi = _band_split(corrected.reshape(-1, h, w))
    gain = min(max(target / max(float(hi.std()), 1e-6), 0.05), 5.0)
    corrected = (lo + hi * gain).reshape(corrected.shape)
    mean, std = corrected.mean(dim=(1, 2, 3)), corrected.std(dim=(1, 2, 3))

    def match(region: Any) -> Any:
        ra = std / region.std(dim=(1, 2, 3)).clamp(min=1e-6)
        rb = mean - ra * region.mean(dim=(1, 2, 3))
        out = region * ra.view(c, 1, 1, 1) + rb.view(c, 1, 1, 1)
        lo, hi = _band_split(out.reshape(-1, h, w))
        return (lo + hi * gain).reshape(out.shape)

    out = torch.cat([match(head), corrected, match(tail)], dim=1)
    stats = {"a_mean": round(float(a.mean()), 4), "b_mean": round(float(b.mean()), 4), "gain": round(gain, 4)}
    return out.unsqueeze(0).to(video.dtype), stats


# --- ComfyUI nodes ------------------------------------------------------------------------------------


class LzH3LatentHandover:
    """The previous part's last 22 frames (video tokens + 40 Hz audio) as this latent's first, noise
    mask 0, and as the frame-0 keyframe."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"positive": ("CONDITIONING",), "latent": ("LATENT",),
                             "previous_latent": ("STRING", {"default": ""})}}

    @classmethod
    def IS_CHANGED(cls, **_: Any) -> float:
        return float("nan")

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "pin"
    CATEGORY = "lzai/h3"

    def pin(self, positive: Any, latent: dict[str, Any], previous_latent: str) -> tuple[Any, dict[str, Any]]:
        import comfy.nested_tensor
        import node_helpers
        import torch

        video, audio = latent["samples"].unbind()
        pv, pa, meta = load_latent(Path(previous_latent))
        if tuple(pv.shape[-2:]) != tuple(video.shape[-2:]):
            raise ValueError(f"the previous part is {meta['width']}x{meta['height']}, this one "
                             f"{video.shape[-1] * 16}x{video.shape[-2] * 16}")
        start_token, audio_start, steps = handover_span(pv.shape[2], pa.shape[-1])
        tail_video = pv[:, :, start_token:].to(video)
        tail_audio = pa[..., audio_start:].to(audio)
        pinned_video, pinned_audio = video.clone(), audio.clone()
        pinned_video[:, :, :HANDOVER_TOKENS] = tail_video
        pinned_audio[..., :steps] = tail_audio
        video_mask, audio_mask = torch.ones_like(pinned_video), torch.ones_like(pinned_audio)
        video_mask[:, :, :HANDOVER_TOKENS] = 0
        audio_mask[..., :steps] = 0
        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append({"resolved_frame_index": 0, "latent": tail_video, "audio_latent": tail_audio})
        positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})
        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((pinned_video, pinned_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        return (positive, out)


class LzH3LatentSave:
    """After the sampler: refine to part 0 (``reference``) when given, save to ``save_path``, pass
    the latent on (noise mask dropped) to the decoders."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"latent": ("LATENT",), "save_path": ("STRING", {"default": ""}),
                             "reference": ("STRING", {"default": ""}),
                             "previous_latent": ("STRING", {"default": ""})}}

    @classmethod
    def IS_CHANGED(cls, **_: Any) -> float:
        return float("nan")

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "lzai/h3"

    def save(self, latent: dict[str, Any], save_path: str, reference: str, previous_latent: str) -> dict[str, Any]:
        import comfy.nested_tensor

        video, audio = latent["samples"].unbind()
        report: dict[str, Any] = {"refined": False}
        if reference:
            ref_video, _, _ = load_latent(Path(reference))
            video, stats = refine(video, ref_video)
            report.update(refined=True, **stats)
        head_tokens = head_audio = 0
        if previous_latent:
            previous_video, previous_audio, _ = load_latent(Path(previous_latent))
            _, _, head_audio = handover_span(previous_video.shape[2], previous_audio.shape[-1])
            head_tokens = HANDOVER_TOKENS
        path = Path(save_path)
        save_latent(path, video, audio, head_tokens=head_tokens, head_audio_steps=head_audio)
        log_event("minimax_h3_chain_latent_saved", path=str(path), video_tokens=int(video.shape[2]),
                  audio_steps=int(audio.shape[-1]), head_tokens=head_tokens, head_audio_steps=head_audio, **report)
        out = dict(latent)
        out.pop("noise_mask", None)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        return {"ui": {"images": [{"filename": path.name, "subfolder": "", "type": "output"}]}, "result": (out,)}


class _FrameSink:
    """What the H3 VAE's ``decode_temporal`` writes its finished frames into: rgb24 bytes for ffmpeg.

    It only ever writes ``dec[:, :, pos:pos + n].copy_(part)`` in order and reads ``dec.shape``, so
    ``shape[2]`` (the frames to keep) also trims the tail.
    """

    def __init__(self, shape: tuple[int, ...], pipe: Any, on_frames: Any):
        self.shape = shape
        self._pipe = pipe
        self._on_frames = on_frames
        self.written = 0

    def __getitem__(self, _key: Any) -> "_FrameSink":
        return self

    def copy_(self, part: Any) -> "_FrameSink":
        import torch

        cancel.check()
        frames = part[0].mul(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
        self._pipe.write(frames.cpu().numpy().tobytes())
        self.written += int(frames.shape[0])
        self._on_frames(self.written, int(self.shape[2]))
        return self


class LzH3ChainDecode:
    """Decode a chain of part latents (newline-separated paths, in order) into one mp4 at ``output_path``.

    ``max_frames`` > 0 trims the chain to that many frames (audio likewise).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"vae": ("VAE",), "audio_vae": ("VAE",), "latent_paths": ("STRING", {"default": ""}),
                             "output_path": ("STRING", {"default": ""}),
                             "max_frames": ("INT", {"default": 0, "min": 0, "max": 1 << 20}),
                             "with_audio": ("BOOLEAN", {"default": True})}}

    @classmethod
    def IS_CHANGED(cls, **_: Any) -> float:
        return float("nan")

    RETURN_TYPES = ()
    FUNCTION = "decode"
    OUTPUT_NODE = True
    CATEGORY = "lzai/h3"

    def decode(self, vae: Any, audio_vae: Any, latent_paths: str, output_path: str, max_frames: int,
               with_audio: bool) -> dict[str, Any]:
        output = Path(output_path)
        LAST_DECODE.clear()
        LAST_DECODE.update(decode_latents(vae, audio_vae, [Path(p) for p in latent_paths.splitlines() if p.strip()],
                                          output, max_frames=int(max_frames), with_audio=bool(with_audio)))
        return {"ui": {"images": [{"filename": output.name, "subfolder": "", "type": "output"}]}}


#: The last chain decode's metrics (one job renders at a time).
LAST_DECODE: dict[str, Any] = {}


def concat_chain(paths: list[Path]) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Part 0 whole, every later part after its handed-over head: (video, audio, metadata per part)."""
    import torch

    videos, audios, metas = [], [], []
    for index, path in enumerate(paths):
        video, audio, meta = load_latent(path)
        if index == 0:
            if meta["head_tokens"]:
                raise ValueError(f"{path.name} continues another part; a chain starts with part 0")
        else:
            if not meta["head_tokens"]:
                raise ValueError(f"latent {index} ({path.name}) is a part 0, not a continuation")
            if (meta["width"], meta["height"]) != (metas[0]["width"], metas[0]["height"]):
                raise ValueError(f"latent {index} is {meta['width']}x{meta['height']}, part 0 "
                                 f"{metas[0]['width']}x{metas[0]['height']}")
            video = video[:, :, meta["head_tokens"]:]
            audio = audio[..., meta["head_audio_steps"]:]
        videos.append(video)
        audios.append(audio)
        metas.append(meta)
    return torch.cat(videos, dim=2), torch.cat(audios, dim=-1), metas


def decode_latents(vae: Any, audio_vae: Any, paths: list[Path], output: Path, *, max_frames: int = 0,
                   with_audio: bool = True) -> dict[str, Any]:
    """One audio VAE pass and one streaming video VAE pass over the chain, encoded to ``output``."""
    import comfy.model_management as model_management
    import comfy.model_prefetch
    from comfy_extras.nodes_audio import vae_decode_audio

    if not paths:
        raise ValueError("no latents to decode")
    began = time.perf_counter()
    video, audio, metas = concat_chain(paths)
    chain_frames = _frames_of_tokens(video.shape[2])
    frames = min(chain_frames, max_frames) if max_frames > 0 else chain_frames
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_path = output.with_name(f"{output.stem}.audio.f32")
    audio_seconds = 0.0
    sample_rate = channels = 0
    if with_audio:
        started = time.perf_counter()
        decoded = vae_decode_audio(audio_vae, {"samples": audio})
        waveform = decoded["waveform"][0].float().cpu()  # [channels, samples]
        sample_rate, channels = int(decoded["sample_rate"]), int(waveform.shape[0])
        audio_path.write_bytes(waveform.t().contiguous().numpy().tobytes())
        audio_seconds = time.perf_counter() - started
        del decoded, waveform
    width, height = int(video.shape[-1]) * 16, int(video.shape[-2]) * 16
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(FPS), "-i", "-"]
    if with_audio:
        command += ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "-i", str(audio_path)]
    command += ["-map", "0:v"] + (["-map", "1:a", "-c:a", "aac", "-b:a", "192k"] if with_audio else [])
    command += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-t", f"{frames / FPS:.6f}",
                "-movflags", "+faststart", str(output)]
    started = time.perf_counter()
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    sink = _FrameSink((1, 3, frames, height, width), encoder.stdin, progress.step)
    try:
        vae.throw_exception_if_invalid()
        with model_management.cuda_device_context(vae.device):
            with comfy.model_prefetch.pause_malloc_graph():
                model_management.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(
                    video.shape, vae.vae_dtype), force_full_load=vae.disable_offload)
            vae.first_stage_model.decode(video.to(vae.device, vae.vae_dtype), output_buffer=sink)
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode(errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed encoding the chain: {stderr[-2000:]}")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        audio_path.unlink(missing_ok=True)
    if sink.written < frames:
        raise RuntimeError(f"the chain decoded {sink.written} of {frames} frames")
    metrics = {
        "parts": len(paths),
        "frames": frames,
        "chain_frames": chain_frames,
        "video_tokens": int(video.shape[2]),
        "audio_steps": int(audio.shape[-1]),
        "width": width,
        "height": height,
        "audio_decode_seconds": round(audio_seconds, 3),
        "video_decode_seconds": round(time.perf_counter() - started, 3),
        "decode_seconds": round(time.perf_counter() - began, 3),
    }
    log_event("minimax_h3_chain_decoded", output=str(output), **metrics)
    return metrics


def register_nodes(comfy_nodes: Any) -> None:
    """Add the chain nodes to a loaded ComfyUI ``nodes`` module."""
    comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(HANDOVER_NODE, LzH3LatentHandover)
    comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(SAVE_NODE, LzH3LatentSave)
    comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(DECODE_NODE, LzH3ChainDecode)


# --- graph wiring and runners ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainPart:
    """A videogen render as a chain part.

    ``latent_path``: where its (refined) latent is written. ``previous_latent``: the part it
    continues (``None`` = part 0). ``reference_latent``: part 0's latent, the refiner's target
    (``None``: not refined). ``decode``: also decode the part to the job's mp4 (else the latent is
    the result).
    """

    latent_path: Path
    previous_latent: Path | None = None
    reference_latent: Path | None = None
    decode: bool = False


def wire_part(graph: dict[str, dict[str, Any]], part: ChainPart) -> str | None:
    """Add the handover and save nodes to a videogen graph (:func:`build_h3_graph`, guides already
    added). Returns the save node's id when the part is not decoded (the decode nodes are dropped)."""
    if part.previous_latent is not None:
        graph[HANDOVER_NODE_ID] = {
            "class_type": HANDOVER_NODE,
            "inputs": {"positive": graph["11"]["inputs"]["conditioning"], "latent": ["8", 1],
                       "previous_latent": str(part.previous_latent)},
        }
        graph["11"]["inputs"]["conditioning"] = [HANDOVER_NODE_ID, 0]
        graph["13"]["inputs"]["latent_image"] = [HANDOVER_NODE_ID, 1]
    graph[SAVE_NODE_ID] = {
        "class_type": SAVE_NODE,
        "inputs": {"latent": ["13", 0], "save_path": str(part.latent_path),
                   "reference": str(part.reference_latent or ""),
                   "previous_latent": str(part.previous_latent or "")},
    }
    if not part.decode:
        for node_id in ("14", "15", "16", "17"):
            graph.pop(node_id, None)
        return SAVE_NODE_ID
    graph["14"]["inputs"]["samples"] = [SAVE_NODE_ID, 0]
    if "15" in graph:
        graph["15"]["inputs"]["samples"] = [SAVE_NODE_ID, 0]
    return None


def decode_chain(*, job_id: str, latent_paths: list[Path], output_path: Path, with_audio: bool = True,
                 max_frames: int = 0) -> dict[str, Any]:
    """Decode ``latent_paths`` (part 0 first) into ``output_path`` on the retained H3 runtime."""
    from . import minimax_h3_runtime as h3

    retained = h3._retained_runtime(generate_audio=True, check_models=False)
    import nodes as comfy_nodes

    register_nodes(comfy_nodes)
    graph = {
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": h3.video_vae_name()}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": h3.audio_vae_name()}},
        "1": {"class_type": DECODE_NODE,
              "inputs": {"vae": ["4", 0], "audio_vae": ["5", 0],
                         "latent_paths": "\n".join(str(p) for p in latent_paths),
                         "output_path": str(output_path), "max_frames": int(max_frames),
                         "with_audio": bool(with_audio)}},
    }
    result = h3._execute_graph(retained, graph, job_id=job_id, save_node_id="1")
    if not output_path.exists():
        raise FileNotFoundError(f"the chain decode did not produce {output_path}")
    return {**LAST_DECODE, "pipeline_seconds": round(result["pipeline_seconds"], 3),
            "cuda_peak_allocated_gib": result["cuda_peak_allocated_gib"],
            "cuda_peak_reserved_gib": result["cuda_peak_reserved_gib"]}


def chain_summary(paths: list[Path]) -> dict[str, Any]:
    """Frames/size of a chain without loading its tensors (validation before the GPU)."""
    metas = [read_metadata(p) for p in paths]
    frames = metas[0]["frames"] + sum(m["frames"] - HANDOVER_FRAMES for m in metas[1:])
    return {"parts": len(metas), "frames": frames, "width": metas[0]["width"], "height": metas[0]["height"],
            "metadata": json.loads(json.dumps(metas))}
