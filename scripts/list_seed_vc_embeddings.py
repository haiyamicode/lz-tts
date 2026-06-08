#!/usr/bin/env python3
"""List saved Seed-VC voice embeddings."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_VC_ROOT = Path(os.environ.get("SEED_VC_ROOT", "data/seed-vc"))
SEED_VC_ROOT = SEED_VC_ROOT if SEED_VC_ROOT.is_absolute() else PROJECT_ROOT / SEED_VC_ROOT
SEED_VC_RUNTIME_ROOT = Path(os.environ.get("SEED_VC_RUNTIME_ROOT", "src/seed_vc_runtime"))
SEED_VC_RUNTIME_ROOT = (
    SEED_VC_RUNTIME_ROOT if SEED_VC_RUNTIME_ROOT.is_absolute() else PROJECT_ROOT / SEED_VC_RUNTIME_ROOT
)
sys.path.insert(0, str(SEED_VC_RUNTIME_ROOT))

from modules.lazy_embedding_loader import HDF5EmbeddingLoader  # noqa: E402

DEFAULT_EMBEDDINGS_FILE = SEED_VC_ROOT / "embeddings" / "vtts_embeddings.h5"


def parse_args():
    parser = argparse.ArgumentParser(description="List Seed-VC HDF5 voice embeddings.")
    parser.add_argument("--file", type=Path, default=Path(os.environ.get("SEED_VC_EMBEDDINGS_FILE", DEFAULT_EMBEDDINGS_FILE)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.file.exists():
        print(f"Error: Embeddings file not found: {args.file}")
        print("Run: uv run python scripts/download_seed_vc_embeddings.py")
        return 1

    loader = HDF5EmbeddingLoader(args.file)
    voices: dict[str, list[str]] = {}

    for key in sorted(loader.keys()):
        parts = key.split(".")
        voice_id = ".".join(parts[:3])
        style = ".".join(parts[3:]) if len(parts) > 3 else "default"
        voices.setdefault(voice_id, []).append(style)

    print(f"Total embeddings: {len(loader)}")
    print(f"File: {args.file}")
    print()
    for voice_id in sorted(voices):
        print(voice_id)
        print(f"  styles: {', '.join(voices[voice_id])}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
