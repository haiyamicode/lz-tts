"""On-demand engine residency: eviction choice, in-use protection, first-come order, pull policy."""

import asyncio

import pytest

from src.api.residency import EngineResidency, plan_eviction
from src.taskflow_worker import on_demand_task_types

FOOT = {"voxcpm": 7.0, "transcribe": 3.0, "media": 18.0}


def test_plan_eviction_unloads_least_recent_idle_engines_only_as_needed():
    # 20 GB: voxcpm + transcribe fit together; media needs both gone.
    assert plan_eviction("transcribe", {"voxcpm": 1.0}, set(), FOOT, 20.0) == []
    assert plan_eviction("media", {"voxcpm": 2.0, "transcribe": 1.0}, set(), FOOT, 20.0) == ["transcribe", "voxcpm"]
    # 9 GB: voxcpm + transcribe do not fit; transcribe goes.
    assert plan_eviction("voxcpm", {"transcribe": 5.0}, set(), FOOT, 9.0) == ["transcribe"]
    # An engine in use is never a victim: wait (None) until it is released.
    assert plan_eviction("media", {"voxcpm": 2.0, "transcribe": 1.0}, {"voxcpm"}, FOOT, 20.0) is None


def test_plan_eviction_lru_order():
    foot = {**FOOT, "media": 8.0}
    # 12 GB, 10 used: media (8) needs 6 freed. Oldest first; stop once it fits.
    assert plan_eviction("media", {"voxcpm": 1.0, "transcribe": 5.0}, set(), foot, 12.0) == ["voxcpm"]
    assert plan_eviction("media", {"voxcpm": 9.0, "transcribe": 5.0}, set(), foot, 12.0) == ["transcribe", "voxcpm"]


def test_budget_smaller_than_an_engine_is_rejected():
    with pytest.raises(ValueError, match="media"):
        EngineResidency(budget_gb=16.0, footprints_gb=FOOT, load=None, unload=None)


class _Engines:
    """Real load/unload bookkeeping that also checks the budget is never exceeded."""

    def __init__(self, budget: float):
        self.budget = budget
        self.loaded: set[str] = set()
        self.events: list[str] = []
        self.peak = 0.0

    async def load(self, engine: str) -> None:
        await asyncio.sleep(0.01)
        self.loaded.add(engine)
        self.events.append(f"load {engine}")
        self.peak = max(self.peak, sum(FOOT[name] for name in self.loaded))

    async def unload(self, engine: str) -> None:
        await asyncio.sleep(0.01)
        self.loaded.discard(engine)
        self.events.append(f"unload {engine}")


def test_swaps_wait_for_jobs_in_use_and_keep_arrival_order():
    async def scenario():
        engines = _Engines(20.0)
        residency = EngineResidency(budget_gb=20.0, footprints_gb=FOOT, load=engines.load, unload=engines.unload)
        log: list[str] = []

        async def job(name: str, engine: str, seconds: float, delay: float = 0.0):
            await asyncio.sleep(delay)
            async with residency.hold(engine):
                assert engine in engines.loaded
                log.append(f"start {name}")
                await asyncio.sleep(seconds)
                log.append(f"end {name}")

        await asyncio.gather(
            job("tts1", "voxcpm", 0.2),
            job("asr", "transcribe", 0.05, delay=0.02),  # fits beside voxcpm: runs at once
            job("video", "media", 0.05, delay=0.05),  # waits for tts1 to finish
            job("tts2", "voxcpm", 0.05, delay=0.08),  # voxcpm is loaded, but queued behind video
        )
        return engines, log, residency

    engines, log, residency = asyncio.run(scenario())
    assert engines.peak <= 20.0
    assert log.index("start asr") < log.index("end tts1")
    assert log.index("start video") > log.index("end tts1")
    assert log.index("start tts2") > log.index("end video")
    # voxcpm was loaded once, unloaded for video, loaded again for tts2.
    assert engines.events.count("load voxcpm") == 2 and "unload voxcpm" in engines.events
    assert residency.snapshot().waiting_for is None


def test_shared_load_and_failed_load_leaves_nothing_resident():
    async def scenario():
        engines = _Engines(20.0)
        attempts = []

        async def flaky_load(engine: str) -> None:
            attempts.append(engine)
            if len(attempts) == 1:
                raise RuntimeError("boom")
            await engines.load(engine)

        residency = EngineResidency(budget_gb=20.0, footprints_gb=FOOT, load=flaky_load, unload=engines.unload)
        with pytest.raises(RuntimeError, match="boom"):
            async with residency.hold("voxcpm"):
                pass
        assert not residency.snapshot().resident

        async def use():
            async with residency.hold("voxcpm"):
                await asyncio.sleep(0.02)

        await asyncio.gather(use(), use(), use())
        return attempts, residency

    attempts, residency = asyncio.run(scenario())
    assert attempts == ["voxcpm", "voxcpm"]  # three concurrent holds share one load
    assert residency.snapshot().resident == frozenset({"voxcpm"})


TYPES = ("tts-synthesis", "voice-design", "voice-enhance", "transcribe", "force-align", "videogen", "imagegen")


def test_on_demand_pull_types():
    # Nothing loaded, or not sticky: everything.
    assert on_demand_task_types(TYPES, frozenset(), None, True) == TYPES
    assert on_demand_task_types(TYPES, frozenset({"media"}), None, False) == TYPES
    # Sticky on the loaded engines (+ the types that need no swappable engine).
    assert on_demand_task_types(TYPES, frozenset({"media"}), None, True) == (
        "voice-enhance",
        "force-align",
        "videogen",
        "imagegen",
    )
    assert on_demand_task_types(TYPES, frozenset({"voxcpm", "transcribe"}), None, True) == (
        "tts-synthesis",
        "voice-design",
        "voice-enhance",
        "transcribe",
        "force-align",
    )
    # A job waits to load media: stop pulling work for the engines it will unload.
    assert on_demand_task_types(TYPES, frozenset({"voxcpm"}), "media", False) == (
        "voice-enhance",
        "force-align",
        "videogen",
        "imagegen",
    )
