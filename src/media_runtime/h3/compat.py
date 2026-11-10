"""Paths and logging the vigen H3 modules relied on from its apiserver package."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RUNTIME_DIR.parents[1]
DATA_DIR = REPO_DIR / "data"

#: Support models for lipsync (data/lipsync, S3-baked like the rest of data/).
LIPSYNC_DIR = DATA_DIR / "lipsync"
WHISPER_MEDIUM_ROOT = LIPSYNC_DIR / "faster-whisper-medium"
SPEECH_CLEANUP_DEMUCS_PATH = LIPSYNC_DIR / "speech-cleanup/htdemucs/955717e8-8726e21a.th"
SPEECH_CLEANUP_ZIPENHANCER_PATH = LIPSYNC_DIR / "speech-cleanup/zipenhancer/pytorch_model.bin"
SPEECH_CLEANUP_ZIPENHANCER_CONFIG_PATH = LIPSYNC_DIR / "speech-cleanup/zipenhancer/configuration.json"
SYNCNET_EVAL_SYNCNET_PATH = LIPSYNC_DIR / "syncnet-eval/syncnet_v2.model"
SYNCNET_EVAL_S3FD_PATH = LIPSYNC_DIR / "syncnet-eval/sfd_face.pth"
GLITCH_DETECTOR_PATH = LIPSYNC_DIR / "glitch-detector/glitch_detector_v1.pt"


def data_dir() -> Path:
    """Scratch root (jobs, outputs); H3_DATA_DIR overrides."""
    return Path(os.environ.get("H3_DATA_DIR") or RUNTIME_DIR / ".cache" / "h3")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_DIR / path


def job_dir(job_id: str) -> Path:
    return data_dir() / "jobs" / str(job_id)


def log_event(event: str, **fields) -> None:
    print(f"[h3] {json.dumps({'event': event, 't': round(time.time(), 3), **fields}, default=str)}", file=sys.stderr, flush=True)
