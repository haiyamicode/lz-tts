from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

_CONFIG_FILENAME = "lora_config.json"
_WEIGHTS_FILENAME = "lora_weights.safetensors"


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with (path / _CONFIG_FILENAME).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    config = payload.get("lora_config")
    if not isinstance(config, dict):
        raise TypeError(f"Invalid VoxCPM LoRA config: {path / _CONFIG_FILENAME}")
    weights = load_file(path / _WEIGHTS_FILENAME, device="cpu")
    return payload, weights


def compose_voxcpm_loras(checkpoint_paths: Sequence[Path], output_path: Path) -> None:
    """Write one LoRA whose delta is the exact sum of the input LoRA deltas."""
    if len(checkpoint_paths) < 2:
        raise ValueError("At least two VoxCPM LoRAs are required for composition")

    checkpoints = [_load_checkpoint(path) for path in checkpoint_paths]
    configs = [payload["lora_config"] for payload, _ in checkpoints]
    ranks = [int(config.get("r", config.get("rank", 0))) for config in configs]
    alphas = [
        float(config.get("alpha", config.get("lora_alpha", 0))) for config in configs
    ]
    if any(rank <= 0 for rank in ranks) or any(alpha <= 0 for alpha in alphas):
        raise ValueError(
            "Every composed VoxCPM LoRA must declare positive rank and alpha values"
        )

    module_names: set[str] = set()
    for _, weights in checkpoints:
        for key in weights:
            if key.endswith(".lora_A"):
                module_names.add(key[: -len(".lora_A")])
            elif key.endswith(".lora_B"):
                module_names.add(key[: -len(".lora_B")])

    combined_weights: dict[str, torch.Tensor] = {}
    for module_name in sorted(module_names):
        a_key = f"{module_name}.lora_A"
        b_key = f"{module_name}.lora_B"
        template_a = next(
            (weights[a_key] for _, weights in checkpoints if a_key in weights), None
        )
        template_b = next(
            (weights[b_key] for _, weights in checkpoints if b_key in weights), None
        )
        if template_a is None or template_b is None:
            raise ValueError(f"Incomplete LoRA tensor pair for module {module_name!r}")

        a_parts: list[torch.Tensor] = []
        b_parts: list[torch.Tensor] = []
        for (_, weights), rank, alpha in zip(checkpoints, ranks, alphas):
            has_a = a_key in weights
            has_b = b_key in weights
            if has_a != has_b:
                raise ValueError(
                    f"Incomplete LoRA tensor pair for module {module_name!r}"
                )
            if has_a:
                a = weights[a_key].to(dtype=torch.float32)
                b = weights[b_key].to(dtype=torch.float32)
                if a.shape != (rank, template_a.shape[1]) or b.shape != (
                    template_b.shape[0],
                    rank,
                ):
                    raise ValueError(
                        f"Incompatible LoRA tensor shape for module {module_name!r}"
                    )
            else:
                a = torch.zeros((rank, template_a.shape[1]), dtype=torch.float32)
                b = torch.zeros((template_b.shape[0], rank), dtype=torch.float32)
            a_parts.append(a)
            b_parts.append(b * (alpha / rank))

        combined_weights[a_key] = torch.cat(a_parts, dim=0).contiguous()
        combined_weights[b_key] = torch.cat(b_parts, dim=1).contiguous()

    combined_rank = sum(ranks)

    def union_list(key: str) -> list[str]:
        return list(
            dict.fromkeys(item for config in configs for item in config.get(key, []))
        )

    base_models = {payload.get("base_model") for payload, _ in checkpoints}
    if len(base_models) > 1:
        raise ValueError(
            f"Cannot compose VoxCPM LoRAs trained from different base models: {base_models}"
        )
    combined_payload = {
        "base_model": next(iter(base_models)),
        "composed_from": [path.name for path in checkpoint_paths],
        "lora_config": {
            "enable_lm": any(bool(config.get("enable_lm")) for config in configs),
            "enable_dit": any(bool(config.get("enable_dit")) for config in configs),
            "enable_proj": any(bool(config.get("enable_proj")) for config in configs),
            "r": combined_rank,
            "alpha": combined_rank,
            "dropout": 0.0,
            "target_modules_lm": union_list("target_modules_lm"),
            "target_modules_dit": union_list("target_modules_dit"),
            "target_proj_modules": union_list("target_proj_modules"),
        },
    }

    output_path.mkdir(parents=True, exist_ok=True)
    save_file(combined_weights, output_path / _WEIGHTS_FILENAME)
    with (output_path / _CONFIG_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(combined_payload, file, indent=2)
        file.write("\n")
