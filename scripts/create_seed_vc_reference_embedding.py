#!/usr/bin/env python3
"""Create one cached Seed-VC reference embedding from a local audio sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py

from src.api.seed_vc_backend import SeedVCBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-config", type=Path, default=Path("local/server.json"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    config = json.loads(args.server_config.read_text(encoding="utf-8"))["seed_vc"]
    settings = SimpleNamespace(
        runtime_root=config["runtime_root"],
        root=config["root"],
        tmp_dir=config["tmp_dir"],
        output_dir=config["output_dir"],
        voice_samples_dir=config["voice_samples_dir"],
        embeddings_hdf5_path=config["embeddings_hdf5_path"],
        device=args.device,
        fp16=config.get("fp16", True),
        embedding_cache_size=1,
        estimator_cache_batch_size=config.get("estimator_cache_batch_size", 8),
        estimator_cache_seq_length=config.get("estimator_cache_seq_length", 4096),
    )
    backend = SeedVCBackend(settings)
    style, mel_ref, prompt_condition = backend._prepare_seed_vc_reference(
        args.reference,
        cached_ref_embeddings=None,
        model=backend.model_cache["default"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)
    embedding_key = f"{args.voice_id}.general"
    try:
        with h5py.File(temporary_output, "w") as output:
            group = output.create_group(embedding_key)
            for name, value in {
                "style": style,
                "mel_ref": mel_ref,
                "prompt_condition": prompt_condition,
            }.items():
                group.create_dataset(
                    name,
                    data=value.detach().float().cpu().numpy(),
                    compression="gzip",
                )
        temporary_output.replace(args.output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    print(f"Created {args.output}: {embedding_key}")


if __name__ == "__main__":
    main()
