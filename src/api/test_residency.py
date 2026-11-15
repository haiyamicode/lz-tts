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
    assert on_demand_task_types(TYPES, frozenset({"media"}), None, True) == ("voice-enhance", "videogen", "imagegen")
    assert on_demand_task_types(TYPES, frozenset({"voxcpm", "transcribe"}), None, True) == (
        "tts-synthesis",
        "voice-design",
        "voice-enhance",
        "transcribe",
    )
    # A job waits to load media: stop pulling work for the engines it will unload.
    assert on_demand_task_types(TYPES, frozenset({"voxcpm"}), "media", False) == (
        "voice-enhance",
        "videogen",
        "imagegen",
    )


def test_pinned_footprints_come_off_the_budget():
    # media (no footprint) gets what the pinned engines leave.
    residency = EngineResidency(
        budget_gb=20.0, footprints_gb={"voxcpm": 7.0, "transcribe": 3.0, "media": None}, load=None, unload=None,
        pinned=["transcribe"],
    )
    assert residency.footprints == {"voxcpm": 7.0, "transcribe": 3.0, "media": 17.0}
    with pytest.raises(ValueError, match="voxcpm"):  # 20 - 14 pinned leaves 6 < 7
        EngineResidency(budget_gb=20.0, footprints_gb={"voxcpm": 7.0, "transcribe": 14.0}, load=None, unload=None,
                        pinned=["transcribe"])
    with pytest.raises(ValueError, match="more than"):
        EngineResidency(budget_gb=20.0, footprints_gb={"voxcpm": 12.0, "transcribe": 9.0}, load=None, unload=None,
                        pinned=["voxcpm", "transcribe"])
    with pytest.raises(ValueError, match="media"):
        EngineResidency(budget_gb=20.0, footprints_gb={"voxcpm": 7.0}, load=None, unload=None, pinned=["media"])


def test_pinned_engine_is_never_unloaded():
    async def scenario():
        engines = _Engines(20.0)
        residency = EngineResidency(
            budget_gb=20.0, footprints_gb={"voxcpm": 7.0, "transcribe": 3.0, "media": None},
            load=engines.load, unload=engines.unload, pinned=["transcribe"],
        )
        async with residency.hold("transcribe"):  # the startup load
            pass

        async def job(engine: str, delay: float):
            await asyncio.sleep(delay)
            async with residency.hold(engine):
                assert engine in engines.loaded and "transcribe" in engines.loaded
                await asyncio.sleep(0.03)

        await asyncio.gather(job("media", 0), job("transcribe", 0.01), job("voxcpm", 0.02), job("media", 0.03))
        return engines, residency

    engines, residency = asyncio.run(scenario())
    assert "unload transcribe" not in engines.events and engines.events.count("load transcribe") == 1
    assert engines.events.count("unload media") == 1 and engines.events.count("unload voxcpm") == 1
    assert residency.snapshot().pinned == frozenset({"transcribe"})


def test_on_demand_pull_types_with_a_pinned_engine():
    pinned = frozenset({"aligner"})  # the flex 3090: only the aligner pinned
    # Only the pinned engine loaded: nothing to stick to, pull everything.
    assert on_demand_task_types(TYPES, pinned, None, True, pinned) == TYPES
    # Sticky on media keeps pulling the pinned engine's work.
    assert on_demand_task_types(TYPES, frozenset({"media", "aligner"}), None, True, pinned) == (
        "voice-enhance",
        "force-align",
        "videogen",
        "imagegen",
    )
    # Waiting to load voxcpm: the pinned engine stays, so its work is still pulled.
    assert on_demand_task_types(TYPES, frozenset({"media", "aligner"}), "voxcpm", False, pinned) == (
        "tts-synthesis",
        "voice-design",
        "voice-enhance",
        "force-align",
    )


# The flex 3090: speech and aligner share one process, the others are their own.
SPLIT = {"speech": 2.5, "aligner": 1.5, "voxcpm": 10.0, "transcribe": 2.5, "media": None}
SYNTH = ("voxcpm", "speech", "aligner")


def test_a_running_job_gets_the_rest_of_its_claim_while_a_swap_waits():
    """Synthesis holds voxcpm, then needs speech and the aligner while a media job is queued:
    the queued swap must not block it (that would deadlock: media waits for voxcpm's job)."""

    async def scenario():
        residency = EngineResidency(
            budget_gb=22.0, footprints_gb=SPLIT, load=_noop, unload=_noop, claims=[SYNTH, ("media",)]
        )
        log: list[str] = []

        async def synth():
            job = residency.job(SYNTH)
            async with residency.hold("voxcpm", job):
                log.append("synth voxcpm")
                await asyncio.sleep(0.05)  # media queues meanwhile
                async with residency.hold("speech", job), residency.hold("aligner", job):
                    log.append("synth speech+aligner")
            log.append("synth done")

        async def video():
            await asyncio.sleep(0.02)
            async with residency.hold("media"):
                log.append("video")

        await asyncio.wait_for(asyncio.gather(synth(), video()), timeout=5)
        return log, residency

    log, residency = asyncio.run(scenario())
    assert log == ["synth voxcpm", "synth speech+aligner", "synth done", "video"]
    assert residency.snapshot().resident == frozenset({"media"})


def test_pinned_aligner_serves_force_align_beside_media():
    async def scenario():
        residency = EngineResidency(
            budget_gb=22.0, footprints_gb=SPLIT, load=_noop, unload=_noop, pinned=["aligner"],
            claims=[("media", "aligner")],
        )
        assert residency.footprints["media"] == 20.5
        async with residency.hold("aligner"):  # the startup load
            pass
        log: list[str] = []

        async def video():
            async with residency.hold("media", residency.job(("media", "aligner"))):
                log.append("video start")
                await asyncio.sleep(0.05)
                log.append("video end")

        async def align():
            await asyncio.sleep(0.03)  # media loaded (0.01) and running
            async with residency.hold("aligner", residency.job(("aligner",))):
                log.append("align")

        await asyncio.wait_for(asyncio.gather(video(), align()), timeout=5)
        return log

    assert asyncio.run(scenario()) == ["video start", "align", "video end"]


def test_a_claim_bigger_than_the_unpinned_budget_needs_a_pin():
    # media takes the whole budget, so a job that also needs the aligner only fits it pinned.
    with pytest.raises(ValueError, match="aligner, media"):
        EngineResidency(budget_gb=22.0, footprints_gb=SPLIT, load=None, unload=None, claims=[("media", "aligner")])
    EngineResidency(
        budget_gb=22.0, footprints_gb=SPLIT, load=None, unload=None, pinned=["aligner"], claims=[("media", "aligner")]
    )


def test_pinning_speech_pins_the_aligner():
    from src.api.server import LoadConfig

    assert LoadConfig(pinned=["speech"]).pinned == ["speech", "aligner"]
    assert LoadConfig(pinned=["aligner"]).pinned == ["aligner"]
    # A footprint override keeps the other defaults.
    assert LoadConfig(footprints_gb={"voxcpm": 9.0}).footprints_gb == {
        "speech": 2.5, "aligner": 1.5, "voxcpm": 9.0, "transcribe": 2.5,
    }


async def _noop(engine: str) -> None:
    await asyncio.sleep(0.01)
