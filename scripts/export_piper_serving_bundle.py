#!/usr/bin/env python3
"""Export a stripped Piper serving bundle.

Keeps only the checkpoint fields needed by VitsModel.load_from_checkpoint()
for inference-time serving, and copies the matching config.json into the
destination directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from src.piper.vits.lightning import VitsModel


SLIM_KEYS = (
    "state_dict",
    "hyper_parameters",
    "pytorch-lightning_version",
    "epoch",
    "global_step",
    "hparams_name",
)


def _checkpoint_size(path: Path) -> int:
    return path.stat().st_size


def _fmt_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f}{unit}" if unit != "B" else f"{num_bytes}B"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="model.ckpt")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", action="store_false", dest="validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    source_config = args.source_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    if not source_config.exists():
        raise FileNotFoundError(source_config)

    source_size = _checkpoint_size(source_checkpoint)
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    slim_checkpoint = {key: checkpoint[key] for key in SLIM_KEYS if key in checkpoint}
    required = {"state_dict", "hyper_parameters", "pytorch-lightning_version"}
    missing = sorted(required - set(slim_checkpoint))
    if missing:
        raise ValueError(f"Source checkpoint is missing required keys: {missing}")

    output_checkpoint = output_dir / args.output_name
    tmp_checkpoint = output_dir / f".{args.output_name}.tmp"
    torch.save(slim_checkpoint, tmp_checkpoint)
    tmp_checkpoint.replace(output_checkpoint)

    output_config = output_dir / "config.json"
    if source_config != output_config:
        shutil.copy2(source_config, output_config)

    if args.validate:
        model = VitsModel.load_from_checkpoint(
            str(output_checkpoint),
            dataset=None,
            weights_only=False,
        )
        del model

    print(
        f"saved {output_checkpoint} "
        f"({_fmt_bytes(source_size)} -> {_fmt_bytes(_checkpoint_size(output_checkpoint))})"
    )
    print(f"config {output_config}")


if __name__ == "__main__":
    main()
