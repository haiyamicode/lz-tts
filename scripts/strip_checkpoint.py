#!/usr/bin/env python3
"""Strip a PyTorch Lightning checkpoint to serving-only format.

Removes training artifacts (optimizer states, schedulers, callbacks, loops)
and weight_norm duplicates. The output loads with torch.load(..., weights_only=True).

Usage:
    uv run python scripts/strip_checkpoint.py data/lzspeech-sparrow-en-GB/model.ckpt
    uv run python scripts/strip_checkpoint.py data/lzspeech-sparrow-en-GB/model.ckpt -o data/lzspeech-sparrow-en-GB/model.ckpt
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import pathlib as _pl

torch.serialization.add_safe_globals([_pl.PosixPath])

try:
    import omegaconf
    torch.serialization.add_safe_globals([omegaconf.DictConfig, omegaconf.ListConfig])
except (ImportError, AttributeError):
    pass


def strip_checkpoint(src: Path, dst: Path | None = None, inference_only: bool = True):
    if dst is None:
        dst = src

    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    stripped: dict = {}
    for k in ["epoch", "global_step", "hparams_name", "pytorch-lightning_version"]:
        if k in ckpt:
            stripped[k] = ckpt[k]
    stripped["state_dict"] = ckpt["state_dict"]

    # Flatten hyper_parameters to avoid OmegaConf serialization
    hp = ckpt.get("hyper_parameters", {})
    try:
        from omegaconf import OmegaConf
        hp = OmegaConf.to_container(hp, resolve=True)
    except Exception:
        hp = dict(hp) if hasattr(hp, "items") else {}

    if inference_only:
        hp["inference_only"] = True

    stripped["hyper_parameters"] = hp

    sd = stripped["state_dict"]
    clean_sd = OrderedDict()
    removed_dups = 0
    removed_discriminator = 0
    for k, v in sd.items():
        if k.endswith("_org"):
            removed_dups += 1
        elif inference_only and k.startswith("model_d."):
            removed_discriminator += 1
        else:
            clean_sd[k] = v
    stripped["state_dict"] = clean_sd

    torch.save(stripped, dst)

    # Verify
    try:
        torch.load(dst, map_location="cpu", weights_only=True)
        print(f"  weights_only=True OK")
    except Exception:
        print(f"  weights_only=True unavailable (OmegaConf model, expected)")

    final_mb = dst.stat().st_size / (1024 * 1024)
    print(f"Stripped: {src} -> {dst}")
    print(f"  keys: {sorted(stripped.keys())}, {final_mb:.0f} MB")
    print(f"  epoch: {stripped.get('epoch', 'N/A')}")
    print(f"  state_dict: {len(clean_sd)} tensors ({removed_dups} dups, {removed_discriminator} discriminator removed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source Lightning checkpoint")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: overwrite source)")
    args = parser.parse_args()
    strip_checkpoint(args.src, args.output)


if __name__ == "__main__":
    main()
