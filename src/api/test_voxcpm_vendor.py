from __future__ import annotations

from src.nanovllm_voxcpm.engine.block_manager import BlockManager
from src.nanovllm_voxcpm.engine.sequence import Sequence


def test_prefix_cache_is_reused_only_by_the_same_adapter() -> None:
    manager = BlockManager(num_blocks=6, block_size=2)

    first = Sequence("first", [10, 20], 2, adapter_id=7)
    manager.allocate(first)
    assert first.num_cached_tokens == 0
    manager.deallocate(first)

    same_adapter = Sequence("same", [10, 20], 2, adapter_id=7)
    manager.allocate(same_adapter)
    assert same_adapter.num_cached_tokens == 2
    manager.deallocate(same_adapter)

    different_adapter = Sequence("different", [10, 20], 2, adapter_id=8)
    manager.allocate(different_adapter)
    assert different_adapter.num_cached_tokens == 0
    manager.deallocate(different_adapter)

    baseline = Sequence("baseline", [10, 20], 2)
    manager.allocate(baseline)
    assert baseline.num_cached_tokens == 0


def test_prefix_cache_is_reused_only_by_the_same_control_namespace() -> None:
    manager = BlockManager(num_blocks=6, block_size=2)

    first = Sequence("first", [10, 20], 2, cache_namespace=b"ipa-a")
    manager.allocate(first)
    assert first.num_cached_tokens == 0
    manager.deallocate(first)

    same_control = Sequence("same", [10, 20], 2, cache_namespace=b"ipa-a")
    manager.allocate(same_control)
    assert same_control.num_cached_tokens == 2
    manager.deallocate(same_control)

    different_control = Sequence("different", [10, 20], 2, cache_namespace=b"ipa-b")
    manager.allocate(different_control)
    assert different_control.num_cached_tokens == 0
