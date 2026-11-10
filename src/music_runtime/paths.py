"""Locations of the music runtime's code and model weights.

Imported before any model library so the Hugging Face cache is fixed to the
pre-baked snapshot and the runtime never reaches the network for weights.
"""

from __future__ import annotations

import os
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
REPO_DIR = RUNTIME_DIR.parents[1]
DATA_DIR = REPO_DIR / "data"

YUE2_DIR = DATA_DIR / "yue2-3b"
YINGMUSIC_DIR = DATA_DIR / "yingmusic-svc"

VENDOR_SVC_DIR = RUNTIME_DIR / "vendor" / "yingmusic" / "svc"
VENDOR_SEPARATION_DIR = RUNTIME_DIR / "vendor" / "yingmusic" / "separation"

os.environ.setdefault("HF_HUB_CACHE", str(YINGMUSIC_DIR / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
