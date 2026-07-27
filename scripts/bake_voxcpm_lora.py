#!/usr/bin/env python3
"""Merge a VoxCPM LoRA checkpoint into a standalone production model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _lora_scale(checkpoint: Path) -> float:
    config_path = checkpoint / "lora_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lora = config.get("lora_config", config)
    rank = int(lora["r"])
    alpha = float(lora["alpha"])
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank: {rank}")
    return alpha / rank


def _copy_model_assets(base_model: Path, output: Path) -> None:
    for source in base_model.iterdir():
        if source.name == "model.safetensors":
            continue
        destination = output / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source.resolve(), destination)


def bake(base_model: Path, checkpoint: Path, output: Path) -> None:
    base_weights = base_model / "model.safetensors"
    lora_weights = checkpoint / "lora_weights.safetensors"
    if not base_weights.is_file():
        raise FileNotFoundError(base_weights)
    if not lora_weights.is_file():
        raise FileNotFoundError(lora_weights)

    output.mkdir(parents=True, exist_ok=True)
    scale = _lora_scale(checkpoint)
    merged: dict[str, torch.Tensor] = {}
    merged_names: list[str] = []

    with (
        safe_open(base_weights, framework="pt", device="cpu") as base_file,
        safe_open(lora_weights, framework="pt", device="cpu") as lora_file,
    ):
        base_keys = set(base_file.keys())
        lora_keys = set(lora_file.keys())
        lora_a_keys = sorted(key for key in lora_keys if key.endswith(".lora_A"))
        if not lora_a_keys:
            raise ValueError(f"No LoRA tensors found in {lora_weights}")

        merge_specs: dict[str, tuple[str, str]] = {}
        for a_key in lora_a_keys:
            stem = a_key[: -len(".lora_A")]
            b_key = f"{stem}.lora_B"
            weight_key = f"{stem}.weight"
            if b_key not in lora_keys:
                raise KeyError(f"Missing paired tensor {b_key}")
            if weight_key not in base_keys:
                raise KeyError(f"Missing base tensor {weight_key}")
            merge_specs[weight_key] = (a_key, b_key)

        for weight_key in sorted(base_keys):
            base_tensor = base_file.get_tensor(weight_key)
            spec = merge_specs.get(weight_key)
            if spec is None:
                merged[weight_key] = base_tensor
                continue
            a_key, b_key = spec
            lora_a = lora_file.get_tensor(a_key)
            lora_b = lora_file.get_tensor(b_key)
            expected_shape = (lora_b.shape[0], lora_a.shape[1])
            if tuple(base_tensor.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {weight_key}: base={tuple(base_tensor.shape)} "
                    f"A={tuple(lora_a.shape)} B={tuple(lora_b.shape)}"
                )
            delta = torch.matmul(lora_b.float(), lora_a.float()).mul_(scale)
            merged[weight_key] = base_tensor.float().add_(delta).to(base_tensor.dtype)
            merged_names.append(weight_key)

    if len(merged_names) != len(lora_a_keys):
        raise RuntimeError(
            f"Merged {len(merged_names)} weights but found {len(lora_a_keys)} LoRA pairs"
        )

    save_file(merged, output / "model.safetensors")
    _copy_model_assets(base_model, output)
    metadata = {
        "base_model": str(base_model.resolve()),
        "lora_checkpoint": str(checkpoint.resolve()),
        "lora_scale": scale,
        "merged_weight_count": len(merged_names),
    }
    (output / "bake_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Baked {len(merged_names)} LoRA weights into "
        f"{output / 'model.safetensors'}"
    )


def main() -> None:
    args = _parse_args()
    bake(args.base_model, args.lora_checkpoint, args.output)


if __name__ == "__main__":
    main()
