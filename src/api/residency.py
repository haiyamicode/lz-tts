"""On-demand engine residency: keep the swappable engines inside a VRAM budget.

``load.mode`` "on-demand" (the flex worker): every engine starts unloaded and loads when a job
needs it. ``speech`` is Sparrow + Seed-VC, ``aligner`` the CTC aligner + the number classifier
(both live in the one speech process, which unloads them in place); voxcpm, transcribe and media
are their own processes. Before a load, idle engines are unloaded least recently used first until
the engine's footprint fits ``budget_gb``. An engine a job is using is never unloaded; a load that
needs its memory waits for it. A pinned engine (``load.pinned``) loads at startup and is never
unloaded: its footprint is taken off the budget the others share.

A job declares up front every engine it may need (its claim: synthesis may use voxcpm, speech
and the aligner, in any order). It is admitted only once its whole claim fits beside the claims
of the jobs already running; after that, each engine it asks for loads without waiting for any
other job (idle engines outside every claim are unloaded to make room), so a job holding one
engine never waits for memory another job holds. Admission is first come, first served: once a
job waits to be admitted, later jobs wait behind it, so a steady stream on the resident engines
cannot starve a swap. A claim that only names pinned engines needs no admission.

Load and unload are the server's engine loaders and stops; this module only decides when.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass

_LOGGER = logging.getLogger(__name__)

SWAPPABLE_ENGINES = ("speech", "aligner", "voxcpm", "transcribe", "media")


class ResidencyJob:
    """The engines one job may use (its claim) and how many of them it holds right now."""

    def __init__(self, claim: Iterable[str] = ()):
        self.claim: set[str] = set(claim)
        self.admitted: frozenset[str] = frozenset()  # the part of the claim counted in the residency
        self.holds = 0


def plan_eviction(
    engine: str,
    resident: dict[str, float],
    busy: set[str],
    footprints: dict[str, float],
    budget: float,
) -> list[str] | None:
    """Engines to unload so ``engine`` fits, least recently used first; None while it cannot fit yet.

    ``resident``: engine -> last use (monotonic) of every engine holding memory (loading and
    unloading ones included); ``busy``: those that cannot be unloaded now (in use, loading or
    already unloading). Unloads only as many idle engines as needed.
    """
    used = sum(footprints[name] for name in resident)
    need = footprints[engine] - (budget - used)
    victims: list[str] = []
    for name in sorted((name for name in resident if name not in busy), key=resident.__getitem__):
        if need <= 1e-9:
            break
        victims.append(name)
        need -= footprints[name]
    return victims if need <= 1e-9 else None


@dataclass(frozen=True)
class ResidencySnapshot:
    resident: frozenset[str]
    # The engine the oldest waiting job needs, when a job is waiting for memory to free up.
    waiting_for: str | None
    pinned: frozenset[str] = frozenset()


class EngineResidency:
    def __init__(
        self,
        *,
        budget_gb: float,
        footprints_gb: dict[str, float | None],
        load: Callable[[str], Awaitable[None]],
        unload: Callable[[str], Awaitable[None]],
        pinned: Iterable[str] = (),
        claims: Iterable[Iterable[str]] = (),
    ):
        """``footprints_gb``: None = whatever the pinned engines leave of the budget (media).

        ``claims``: the engine sets the server's jobs declare; each must fit the budget at once.
        """
        self.pinned = frozenset(pinned)
        unknown = sorted(self.pinned - footprints_gb.keys())
        if unknown:
            raise ValueError(f"load.pinned names {', '.join(unknown)}, not an engine this worker loads on demand")
        reserved = sum(size for name, size in footprints_gb.items() if name in self.pinned and size is not None)
        self.footprints = {
            name: budget_gb - reserved if size is None else size for name, size in footprints_gb.items()
        }
        pinned_total = sum(self.footprints[name] for name in self.pinned)
        if pinned_total > budget_gb + 1e-9:
            raise ValueError(f"load.pinned engines need {pinned_total:g} GB, more than load.budget_gb {budget_gb:g}")
        too_big = [
            name
            for name, size in self.footprints.items()
            if name not in self.pinned and size > budget_gb - pinned_total + 1e-9
        ]
        if too_big:
            raise ValueError(
                f"load.budget_gb {budget_gb:g} less the pinned engines' {pinned_total:g} GB"
                f" is smaller than the footprint of {', '.join(too_big)}"
            )
        for claim in claims:
            claim = set(claim) & self.footprints.keys()
            needed = sum(self.footprints[name] for name in claim - self.pinned)
            if needed > budget_gb - pinned_total + 1e-9:
                raise ValueError(
                    f"a job using {', '.join(sorted(claim))} needs {needed:g} GB of unpinned engines at once,"
                    f" more than load.budget_gb {budget_gb:g} less the pinned engines' {pinned_total:g} GB"
                    " (pin some of them)"
                )
        self.budget = budget_gb
        self._pinned_total = pinned_total
        self._load = load
        self._unload = unload
        self._cond = asyncio.Condition()
        self._queue: deque[tuple[ResidencyJob, str]] = deque()  # jobs waiting to be admitted
        self._claimed: Counter[str] = Counter()  # admitted claims of the running jobs
        self._resident: dict[str, float] = {}  # loaded, loading or unloading -> last use
        self._in_use: Counter[str] = Counter()
        self._loads: dict[str, asyncio.Task[None]] = {}
        self._unloading: set[str] = set()

    def snapshot(self) -> ResidencySnapshot:
        return ResidencySnapshot(
            resident=frozenset(name for name in self._resident if name not in self._unloading),
            waiting_for=self._queue[0][1] if self._queue else None,
            pinned=self.pinned,
        )

    def is_resident(self, engine: str) -> bool:
        return engine in self._resident and engine not in self._unloading

    def job(self, claim: Iterable[str] = ()) -> ResidencyJob:
        """A job that may use the engines in ``claim`` (the ones this worker swaps; others are dropped)."""
        return ResidencyJob(name for name in claim if name in self.footprints)

    def forget(self, engine: str) -> None:
        """The engine's process is gone (it died): its memory is free, the next hold reloads it."""
        if engine in self._resident and engine not in self._loads and engine not in self._unloading:
            del self._resident[engine]
            asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def hold(self, engine: str, job: ResidencyJob | None = None) -> AsyncIterator[None]:
        """Keep ``engine`` loaded for the body, loading it (and unloading idle ones) first if needed.

        ``job``: the job this hold belongs to (its claim is released when its last hold ends);
        None = a job of its own that claims just this engine.
        """
        job = job or self.job((engine,))
        job.holds += 1
        try:
            load_task = await self._acquire(engine, job)
        except BaseException:
            await self._release(job, None)
            raise
        try:
            if load_task is not None:
                await asyncio.shield(load_task)
            yield
        finally:
            await self._release(job, engine)

    async def _release(self, job: ResidencyJob, engine: str | None) -> None:
        async with self._cond:
            if engine is not None:
                self._in_use[engine] -= 1
                if engine in self._resident:
                    self._resident[engine] = time.monotonic()
            job.holds -= 1
            if job.holds == 0:
                self._claimed.subtract(job.admitted)
                self._claimed = +self._claimed
                job.admitted = frozenset()
            self._cond.notify_all()

    def _fits(self, claim: set[str]) -> bool:
        claimed = set(self._claimed) | claim
        return sum(self.footprints[name] for name in claimed - self.pinned) <= self.budget - self._pinned_total + 1e-9

    async def _acquire(self, engine: str, job: ResidencyJob) -> asyncio.Task[None] | None:
        waited_since: float | None = None
        async with self._cond:
            if engine not in job.claim:
                if job.admitted:
                    _LOGGER.warning("Engine residency: a job admitted for %s also needs %s", sorted(job.claim), engine)
                job.claim.add(engine)
            if (job.claim - job.admitted) - self.pinned:
                # Admission: wait until the claim fits beside the running jobs' claims. A job that
                # already holds an engine goes first, since the jobs behind it may wait for that one.
                entry = (job, engine)
                if job.holds > 1:
                    self._queue.appendleft(entry)
                else:
                    self._queue.append(entry)
                try:
                    while True:
                        new = job.claim - job.admitted  # another hold of this job may have been admitted
                        if not new - self.pinned or (self._queue[0] is entry and self._fits(new)):
                            break
                        if waited_since is None:
                            waited_since = time.monotonic()
                            _LOGGER.info(
                                "Engine residency: %s waits (claim=%s resident=%s claimed=%s)",
                                engine,
                                sorted(job.claim),
                                sorted(self._resident),
                                sorted(self._claimed),
                            )
                        await self._cond.wait()
                finally:
                    self._queue = deque(item for item in self._queue if item is not entry)
                    self._cond.notify_all()
            self._claimed.update(job.claim - job.admitted)
            job.admitted = frozenset(job.claim)
            while True:
                if engine in self._resident and engine not in self._unloading:
                    # Loaded, or loading for an earlier job: share it.
                    self._in_use[engine] += 1
                    self._resident[engine] = time.monotonic()
                    return self._loads.get(engine)
                busy = {name for name, count in self._in_use.items() if count > 0}
                busy |= set(self._loads) | self._unloading | self.pinned | set(self._claimed)
                # Only idle engines outside every claim go; the admitted claims fit together,
                # so this waits at most for engines already being unloaded.
                victims = plan_eviction(engine, self._resident, busy, self.footprints, self.budget)
                if victims is not None:
                    self._in_use[engine] += 1
                    self._unloading.update(victims)
                    self._resident[engine] = time.monotonic()
                    task = asyncio.get_running_loop().create_task(self._swap(engine, victims))
                    self._loads[engine] = task
                    if waited_since is not None:
                        _LOGGER.info(
                            "Engine residency: %s waited %.1fs for memory", engine, time.monotonic() - waited_since
                        )
                    return task
                if waited_since is None:
                    waited_since = time.monotonic()
                await self._cond.wait()

    async def _swap(self, engine: str, victims: list[str]) -> None:
        try:
            for victim in victims:
                started = time.monotonic()
                try:
                    await self._unload(victim)
                finally:
                    async with self._cond:
                        self._unloading.discard(victim)
                        self._resident.pop(victim, None)
                        self._cond.notify_all()
                _LOGGER.info("Engine residency: unloaded %s in %.1fs to fit %s", victim, time.monotonic() - started, engine)
            started = time.monotonic()
            await self._load(engine)
            _LOGGER.info("Engine residency: loaded %s in %.1fs", engine, time.monotonic() - started)
        except BaseException:
            async with self._cond:
                self._resident.pop(engine, None)
                self._cond.notify_all()
            with contextlib.suppress(Exception):
                await self._unload(engine)
            raise
        finally:
            async with self._cond:
                self._loads.pop(engine, None)
                self._cond.notify_all()
