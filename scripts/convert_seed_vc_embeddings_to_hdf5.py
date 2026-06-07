#!/usr/bin/env python3
"""Convert Seed-VC monolithic embeddings.pt to compressed HDF5 format."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_VC_ROOT = Path(os.environ.get("SEED_VC_ROOT", "local/seed-vc"))
SEED_VC_ROOT = SEED_VC_ROOT if SEED_VC_ROOT.is_absolute() else PROJECT_ROOT / SEED_VC_ROOT
SEED_VC_RUNTIME_ROOT = Path(os.environ.get("SEED_VC_RUNTIME_ROOT", "src/seed_vc_runtime"))
SEED_VC_RUNTIME_ROOT = (
    SEED_VC_RUNTIME_ROOT if SEED_VC_RUNTIME_ROOT.is_absolute() else PROJECT_ROOT / SEED_VC_RUNTIME_ROOT
)
sys.path.insert(0, str(SEED_VC_RUNTIME_ROOT))

from modules.lazy_embedding_loader import convert_monolithic_to_hdf5  # noqa: E402

DEFAULT_INPUT = SEED_VC_ROOT / "embeddings" / "vtts_embeddings.pt"
DEFAULT_OUTPUT = SEED_VC_ROOT / "embeddings" / "vtts_embeddings.h5"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Seed-VC embeddings.pt to HDF5.")
    parser.add_argument("--input", type=Path, default=Path(os.environ.get("SEED_VC_EMBEDDINGS_PT", DEFAULT_INPUT)))
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("SEED_VC_EMBEDDINGS_H5", DEFAULT_OUTPUT)))
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--compression-level", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert_monolithic_to_hdf5(
        args.input,
        args.output,
        compression=args.compression,
        compression_level=args.compression_level,
    )
    print(f"Converted {args.input} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
