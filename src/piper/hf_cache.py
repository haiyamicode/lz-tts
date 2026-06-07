"""Small helpers for loading Hugging Face assets from the local cache."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_hf_model_path(model_name: str) -> str:
    """Return a local cached snapshot path for a Hugging Face model id if present."""
    path = Path(model_name)
    if path.exists():
        return str(path)

    cache_roots: list[Path] = []
    for env_name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_name)
        if value:
            cache_roots.append(Path(value).expanduser())
    cache_roots.append(Path.home() / ".cache" / "huggingface")

    model_dir_name = "models--" + model_name.replace("/", "--")
    for root in cache_roots:
        candidates = [root / model_dir_name, root / "hub" / model_dir_name]
        for model_dir in candidates:
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue

            ref_path = model_dir / "refs" / "main"
            if ref_path.is_file():
                ref = ref_path.read_text(encoding="utf-8").strip()
                snapshot = snapshots_dir / ref
                if snapshot.is_dir():
                    return str(snapshot)

            snapshots = [item for item in snapshots_dir.iterdir() if item.is_dir()]
            if snapshots:
                return str(max(snapshots, key=lambda item: item.stat().st_mtime))

    return model_name
