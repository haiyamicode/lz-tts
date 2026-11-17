"""Model sets: which ``data/`` dirs each servable engine needs (staged Docker bootstrap).

Stdlib only: ``scripts/download_data.py`` imports it before ``uv sync``.

- ``LZAI_MODELS``: the sets this worker downloads and serves, comma-separated. Unset = every set
  the worker's server.json enables; ``all`` = every set. An engine whose set is left out is
  disabled even if server.json enables it.
- ``LZAI_REQUIRED_MODELS``: the sets that must be on disk before the worker loads, joins Taskflow
  and reports ready. Unset = all of ``LZAI_MODELS``; ``none`` = core only. The rest download in
  the background, and each engine loads (and its task types are pulled) once its sets are in.

Falcon downloads only what the server.json serves: ``voxcpm.model_path`` (default
``data/lzspeech-falcon-v1.2``), the ``applicable_loras`` and ``ipa_adapter_path``. Other Falcon
versions, unused LoRAs and the VoxCPM2 base (a LoRA training base) stay on the bucket.

Every ``data/`` dir no set claims is core and always downloads first: Sparrow and Starling
(routes, capabilities and Falcon's duration budget read them at startup), the CTC aligner,
detectors, classifiers, runtime wheels.

While a set is still downloading, ``data/.pending/<set>`` exists; the worker treats engines
needing it as not loaded yet. ``media-venv`` marks the media runtime venv sync.
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

# Set -> data dir patterns. Dirs may belong to several sets (comfyui).
MODEL_SETS: dict[str, tuple[str, ...]] = {
    "transcribe": ("qwen3-asr-*",),
    "falcon": ("lzspeech-falcon-*", "voxcpm2-*", "voxcpm-lora"),
    "seed-vc": ("seed-vc",),
    "music": ("yue2-3b", "yingmusic-svc"),
    "imagegen": ("comfyui", "face-landmarks", "face-identity", "realism-clip"),
    "videogen": ("comfyui", "rife-*", "span-upscale"),
    "lipsync": ("comfyui", "lipsync", "rife-*", "span-upscale", "face-landmarks", "face-identity", "realism-clip"),
}
MEDIA_SETS = ("music", "videogen", "lipsync", "imagegen")
# Worker engine -> the sets it loads from (media: the enabled media features).
ENGINE_SETS: dict[str, str] = {"voxcpm": "falcon", "seed_vc": "seed-vc", "transcribe": "transcribe"}
MEDIA_VENV_MARKER = "media-venv"
DEFAULT_FALCON_MODEL = "data/lzspeech-falcon-v1.2"


def pending_dir(data_dir: Path | str = "data") -> Path:
    return Path(data_dir) / ".pending"


def pending_sets(data_dir: Path | str = "data") -> set[str]:
    """Sets (and ``media-venv``) still downloading in the background."""
    root = pending_dir(data_dir)
    return {path.name for path in root.iterdir()} if root.is_dir() else set()


def set_of_dir(name: str) -> set[str]:
    """The sets that claim data dir ``name`` (empty = core)."""
    return {s for s, patterns in MODEL_SETS.items() if any(fnmatch.fnmatch(name, p) for p in patterns)}


def _raw_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path or os.environ.get("LZAI_SERVER_CONFIG", "local/server.json"))
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def falcon_paths(config_path: Path | str | None = None) -> list[str]:
    """Data-relative paths Falcon serves: the model dir, its LoRAs and the IPA adapter."""
    voxcpm = _raw_config(config_path).get("voxcpm", {})
    paths = [
        voxcpm.get("model_path") or os.environ.get("VOXCPM_MODEL_PATH") or DEFAULT_FALCON_MODEL,
        *voxcpm.get("applicable_loras", {}).values(),
        voxcpm.get("ipa_adapter_path"),
    ]
    out = []
    for path in filter(None, paths):
        parts = Path(path).parts
        if "data" in parts:  # "data/x", "./data/x", "/abs/repo/data/x"
            out.append("/".join(parts[len(parts) - parts[::-1].index("data"):]))
    return out


def wanted_path(rel: str, sets: set[str] | list[str], falcon: list[str] | None = None) -> bool:
    """Whether data-relative ``rel`` downloads for ``sets`` (core always does).

    ``falcon``: ``falcon_paths()``, computed when omitted.
    """
    owners = set_of_dir(rel.split("/")[0])
    if not owners:
        return True
    for name in owners & set(sets):
        if name != "falcon":
            return True
        if falcon is None:
            falcon = falcon_paths()
        if any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in falcon):
            return True
    return False


def _parse(value: str, env: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if names == ["all"]:
        return list(MODEL_SETS)
    if names == ["none"]:
        return []
    unknown = [name for name in names if name not in MODEL_SETS]
    if unknown:
        raise ValueError(f"{env}: unknown model set(s) {', '.join(unknown)} (choose from {', '.join(MODEL_SETS)}, all, none)")
    return list(dict.fromkeys(names))


def config_sets(config_path: Path | str | None = None) -> list[str]:
    """Sets the server.json enables (raw JSON, with ServerConfig's defaults for missing keys)."""
    raw = _raw_config(config_path)

    def env_bool(name: str, default: bool) -> bool:
        return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

    engines = raw.get("engines", {})
    enabled = {
        "transcribe": bool(raw.get("transcribe", {}).get("enabled", False)),
        "falcon": bool(engines.get("voxcpm", env_bool("VOXCPM_ENABLED", False))),
        "seed-vc": bool(engines.get("seed_vc", env_bool("SEED_VC_ENABLED", True))),
        **{name: bool(raw.get("media", {}).get(name, False)) for name in MEDIA_SETS},
    }
    return [name for name in MODEL_SETS if enabled[name]]


def selected_sets(config_path: Path | str | None = None) -> list[str]:
    """``LZAI_MODELS``: the sets to download and serve."""
    value = os.environ.get("LZAI_MODELS", "").strip()
    return _parse(value, "LZAI_MODELS") if value else config_sets(config_path)


def required_sets(config_path: Path | str | None = None) -> list[str]:
    """``LZAI_REQUIRED_MODELS`` within ``LZAI_MODELS``: what must be on disk before the worker starts."""
    models = selected_sets(config_path)
    value = os.environ.get("LZAI_REQUIRED_MODELS", "").strip()
    if not value:
        return models
    required = _parse(value, "LZAI_REQUIRED_MODELS")
    outside = [name for name in required if name not in models]
    if outside:
        raise ValueError(f"LZAI_REQUIRED_MODELS: {', '.join(outside)} not in LZAI_MODELS ({', '.join(models) or 'none'})")
    return required


def explicit_selection() -> list[str] | None:
    """``LZAI_MODELS`` when set (the worker disables engines outside it), else None."""
    value = os.environ.get("LZAI_MODELS", "").strip()
    return _parse(value, "LZAI_MODELS") if value else None


def media_stage() -> str:
    """``required`` / ``deferred`` / ``none``: when the bootstrap syncs the media runtime venv."""
    models = selected_sets()
    if not any(name in MEDIA_SETS for name in models):
        return "none"
    required = required_sets()
    return "required" if any(name in MEDIA_SETS for name in required) else "deferred"


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["media-stage"]:
        raise SystemExit("usage: python -m src.model_sets media-stage")
    print(media_stage())
