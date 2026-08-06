from __future__ import annotations

from .server import (
    BatchSynthesizeInputItem,
    VoxCPMConfig,
    _batch_item_compatibility_key,
)


def test_voxcpm_release_defaults_use_stabilized_model_and_three_slots() -> None:
    config = VoxCPMConfig()

    assert config.model_path == "data/voxcpm2-stable"
    assert config.max_concurrent_loras == 3
    assert config.max_loras_per_request == 2


def test_voxcpm_batch_grouping_is_adapter_order_independent() -> None:
    first = BatchSynthesizeInputItem(
        text="first",
        model="voxcpm",
        sample_url="https://example.com/first.wav",
        voxcpm_loras=["accent-en-GB", "ipa"],
    )
    second = BatchSynthesizeInputItem(
        text="second",
        model="voxcpm",
        sample_url="https://example.com/second.wav",
        voxcpm_loras=["ipa", "accent-en-GB"],
    )
    different_adapter = BatchSynthesizeInputItem(
        text="third",
        model="voxcpm",
        sample_url="https://example.com/third.wav",
        voxcpm_loras=["accent-en-US"],
    )

    assert _batch_item_compatibility_key(first) == _batch_item_compatibility_key(second)
    assert _batch_item_compatibility_key(first) != _batch_item_compatibility_key(
        different_adapter
    )
