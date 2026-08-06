from __future__ import annotations

import json

import torch
from safetensors.torch import load_file, save_file

from .voxcpm_lora import compose_voxcpm_loras


def _write_lora(path, *, a: torch.Tensor, b: torch.Tensor) -> None:
    path.mkdir()
    (path / "lora_config.json").write_text(
        json.dumps(
            {
                "base_model": "openbmb/VoxCPM2",
                "lora_config": {
                    "enable_lm": True,
                    "enable_dit": False,
                    "enable_proj": False,
                    "r": 1,
                    "alpha": 1,
                    "target_modules_lm": ["q_proj"],
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.0.q_proj.lora_A": a,
            "model.layers.0.q_proj.lora_B": b,
        },
        path / "lora_weights.safetensors",
    )


def test_composition_preserves_the_exact_sum_and_portable_metadata(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "cache" / "combined"
    first_a = torch.tensor([[1.0, 2.0]])
    first_b = torch.tensor([[3.0], [4.0]])
    second_a = torch.tensor([[5.0, 6.0]])
    second_b = torch.tensor([[7.0], [8.0]])
    _write_lora(first, a=first_a, b=first_b)
    _write_lora(second, a=second_a, b=second_b)

    compose_voxcpm_loras([first, second], output)

    weights = load_file(output / "lora_weights.safetensors")
    combined_a = weights["model.layers.0.q_proj.lora_A"]
    combined_b = weights["model.layers.0.q_proj.lora_B"]
    expected_delta = first_b @ first_a + second_b @ second_a
    assert torch.equal(combined_b @ combined_a, expected_delta)
    payload = json.loads((output / "lora_config.json").read_text(encoding="utf-8"))
    assert payload["base_model"] == "openbmb/VoxCPM2"
    assert payload["composed_from"] == ["first", "second"]
