"""On-demand engine residency: keep the swappable engines inside a VRAM budget.

``load.mode`` "on-demand" (the flex worker): voxcpm, transcribe and media start unloaded and
load when a job needs them. Before a load, idle engines are unloaded least recently used first
until the engine's footprint fits ``budget_gb``. An engine a job is using is never unloaded;
a load that needs its memory waits for it. Requests are served first come, first served: once a
job waits for a swap, later jobs wait behind it, so a steady stream on the resident engine
cannot starve the swap. The speech process (Sparrow, Seed-VC, aligner, number classifier) is
not managed here: it always stays loaded, and the budget is what is left beside it.

Load and unload are the server's engine loaders and stops; this module only decides when.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

_LOGGER = logging.getLogger(__name__)

SWAPPABLE_ENGINES = ("voxcpm", "transcribe", "media")


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


class EngineResidency:
    def __init__(
        self,
        *,
        budget_gb: float,
        footprints_gb: dict[str, float],
        load: Callable[[str], Awaitable[None]],
        unload: Callable[[str], Awaitable[None]],
    ):
        too_big = [name for name, size in footprints_gb.items() if size > budget_gb]
        if too_big:
            raise ValueError(f"load.budget_gb {budget_gb:g} is smaller than the footprint of {', '.join(too_big)}")
        self.budget = budget_gb
        self.footprints = dict(footprints_gb)
        self._load = load
        self._unload = unload
        self._cond = asyncio.Condition()
        self._queue: deque[tuple[object, str]] = deque()
        self._resident: dict[str, float] = {}  # loaded, loading or unloading -> last use
        self._in_use: Counter[str] = Counter()
        self._loads: dict[str, asyncio.Task[None]] = {}
        self._unloading: set[str] = set()

    def snapshot(self) -> ResidencySnapshot:
        return ResidencySnapshot(
            resident=frozenset(name for name in self._resident if name not in self._unloading),
            waiting_for=self._queue[0][1] if self._queue else None,
        )

    def is_resident(self, engine: str) -> bool:
        return engine in self._resident and engine not in self._unloading

    def forget(self, engine: str) -> None:
        """The engine's process is gone (it died): its memory is free, the next hold reloads it."""
        if engine in self._resident and engine not in self._loads and engine not in self._unloading:
            del self._resident[engine]
            asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def hold(self, engine: str) -> AsyncIterator[None]:
        """Keep ``engine`` loaded for the body, loading it (and unloading idle ones) first if needed."""
        load_task = await self._acquire(engine)
        try:
            if load_task is not None:
                await asyncio.shield(load_task)
            yield
        finally:
            async with self._cond:
                self._in_use[engine] -= 1
                if engine in self._resident:
                    self._resident[engine] = time.monotonic()
                self._cond.notify_all()

    async def _acquire(self, engine: str) -> asyncio.Task[None] | None:
        ticket = object()
        waited_since: float | None = None
        async with self._cond:
            self._queue.append((ticket, engine))
            try:
                while True:
                    if self._queue[0][0] is ticket:
                        if engine in self._resident and engine not in self._unloading:
                            # Loaded, or loading for an earlier job: share it.
                            self._in_use[engine] += 1
                            self._resident[engine] = time.monotonic()
                            return self._loads.get(engine)
                        busy = {name for name, count in self._in_use.items() if count > 0}
                        busy |= set(self._loads) | self._unloading
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
                        _LOGGER.info(
                            "Engine residency: %s waits (resident=%s in use=%s)",
                            engine,
                            sorted(self._resident),
                            sorted(name for name, count in self._in_use.items() if count > 0),
                        )
                    await self._cond.wait()
            finally:
                self._queue = deque(item for item in self._queue if item[0] is not ticket)
                self._cond.notify_all()

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
