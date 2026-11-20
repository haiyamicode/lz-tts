"""Paths and logging shared by the H3 videogen and lipsync modules."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RUNTIME_DIR.parents[1]
DATA_DIR = REPO_DIR / "data"

#: ComfyUI source, the two H3 custom nodes and the H3 weights (data/comfyui, S3-baked).
COMFYUI_DIR = DATA_DIR / "comfyui"
#: Support models for lipsync (data/lipsync, S3-baked like the rest of data/).
LIPSYNC_DIR = DATA_DIR / "lipsync"
CTC_ROOT = LIPSYNC_DIR / "mms-300m-ctc"
SPEECH_CLEANUP_ZIPENHANCER_PATH = LIPSYNC_DIR / "speech-cleanup/zipenhancer/pytorch_model.bin"
SPEECH_CLEANUP_ZIPENHANCER_CONFIG_PATH = LIPSYNC_DIR / "speech-cleanup/zipenhancer/configuration.json"
SYNCNET_EVAL_SYNCNET_PATH = LIPSYNC_DIR / "syncnet-eval/syncnet_v2.model"
SYNCNET_EVAL_S3FD_PATH = LIPSYNC_DIR / "syncnet-eval/sfd_face.pth"
GLITCH_DETECTOR_PATH = LIPSYNC_DIR / "glitch-detector/glitch_detector_v1.pt"
#: H3 ``upscale``: the 2x SPAN model (data/span-upscale, S3-baked; vigen 43c32e7), see h3.span_upscale.
SPAN_UPSCALE_MODEL_PATH = DATA_DIR / "span-upscale" / "2xNomosUni_span_multijpg_ldl.safetensors"
#: imagegen ``rest_frame``: MediaPipe FaceLandmarker (Apache-2.0, float16 v1; data/face-landmarks, S3-baked),
#: measures how open a rest still's mouth is (h3.rest_stills).
FACE_LANDMARKER_PATH = DATA_DIR / "face-landmarks" / "face_landmarker.task"
#: imagegen ``rest_frame``: InsightFace buffalo_l ArcFace (w600k_r50; data/face-identity, S3-baked), checks a rest
#: still keeps the person's face (h3.rest_stills).
FACE_IDENTITY_PATH = DATA_DIR / "face-identity" / "w600k_r50.onnx"
#: Rest stills: OpenAI CLIP ViT-L/14 (MIT, fp16; data/realism-clip, S3-baked), zero-shot photo vs cartoon/drawn/3D
#: look of the image, whose face checks a non-realistic character skips (h3.rest_stills).
REALISM_CLIP_DIR = DATA_DIR / "realism-clip"


def data_dir() -> Path:
    """Scratch root (jobs, outputs); H3_DATA_DIR overrides."""
    return Path(os.environ.get("H3_DATA_DIR") or RUNTIME_DIR / ".cache" / "h3")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_DIR / path


def job_dir(job_id: str) -> Path:
    return data_dir() / "jobs" / str(job_id)


def _proc_fields(path: str, sep: str | None = None) -> dict[str, int]:
    try:
        with open(path) as f:
            rows = (line.split(sep, 1) for line in f)
            return {k.strip(): int(v.split()[0]) for k, v in rows if v.split() and v.split()[0].isdigit()}
    except OSError:
        return {}


def host_memory() -> dict:
    """Host RAM, swap and disk reads now (Linux /proc; {} elsewhere). The counters are cumulative: diff two events.
    `read_gb` is this process's storage reads (mmap'd weight pages faulted back in count), `swap_in_gb` the host's."""
    mem, vm, io = _proc_fields("/proc/meminfo", ":"), _proc_fields("/proc/vmstat", " "), _proc_fields("/proc/self/io", ":")
    if not mem:
        return {}
    gib = 1024**2  # meminfo is kB
    status = _proc_fields("/proc/self/status", ":")
    return {
        "avail_gb": round(mem.get("MemAvailable", 0) / gib, 1),
        "cached_gb": round(mem.get("Cached", 0) / gib, 1),
        "locked_gb": round(mem.get("Mlocked", 0) / gib, 1),
        "swap_used_gb": round((mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / gib, 1),
        "rss_gb": round(status.get("VmRSS", 0) / gib, 1),
        "read_gb": round(io.get("read_bytes", 0) / 1024**3, 1),
        "swap_in_gb": round(vm.get("pswpin", 0) * 4096 / 1024**3, 2),
        "majflt": vm.get("pgmajfault", 0),
    }


def log_event(event: str, **fields) -> None:
    fields.setdefault("host", host_memory())
    print(f"[h3] {json.dumps({'event': event, 't': round(time.time(), 3), **fields}, default=str)}", file=sys.stderr, flush=True)
