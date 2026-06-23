"""Small helpers for loading Hugging Face assets from the local cache."""

from __future__ import annotations

import os
from pathlib import Path


_HF_MODEL_ALIASES = {
    "distilbert-base-multilingual-cased": "distilbert/distilbert-base-multilingual-cased",
}

_MODEL_WEIGHT_FILES = {
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
}


def _has_model_weights(snapshot_dir: Path) -> bool:
    return any((snapshot_dir / filename).exists() for filename in _MODEL_WEIGHT_FILES)


def resolve_hf_model_path(model_name: str, *, require_weights: bool = False) -> str:
    """Return a local cached snapshot path for a Hugging Face model id if present."""
    model_name = _HF_MODEL_ALIASES.get(model_name, model_name)
    path = Path(model_name)
    if path.exists() and (not require_weights or _has_model_weights(path)):
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
                if snapshot.is_dir() and (
                    not require_weights or _has_model_weights(snapshot)
                ):
                    return str(snapshot)

            snapshots = [item for item in snapshots_dir.iterdir() if item.is_dir()]
            if require_weights:
                snapshots = [item for item in snapshots if _has_model_weights(item)]
            if snapshots:
                return str(max(snapshots, key=lambda item: item.stat().st_mtime))

    return model_name
