"""Model sessions that load, unload and share device memory least-recently-used.

One mechanism serves pinned and opportunistic models. A session is configured
(server.json) with the device it runs on, the memory it needs while resident
(``vram_mib``) and whether it is ``pinned``:

- a pinned session is loaded at startup and never evicted;
- any other session loads on first use and is evicted, least recently used
  first, when a lease needs its device memory.

Pinning only removes a session from the eviction candidates, so admission has
no per-mode branch.

A lease is what a job holds while it uses a session. ``job_mib`` is extra
memory the job reserves for its activations; it counts against the device
budget for as long as the lease is held.

Unloading a session stops its backend. ``ProcessBackend`` runs each session in
its own child process, so stopping the child is what returns the device memory
to the driver.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from .worker_common import ChildWorkerDied, WorkerProcessClient

_LOGGER = logging.getLogger(__name__)

UNLOADED = "unloaded"
LOADING = "loading"
READY = "ready"
UNLOADING = "unloading"


class ResourceError(RuntimeError):
    """A lease cannot be admitted: it can never fit the device budget, or the memory stayed busy past the timeout."""


class DeviceResourceConfig(BaseModel):
    """Memory budget for one device. Required: the budget is what the server is allowed to use."""

    budget_mib: int = Field(gt=0)


class ResourcesConfig(BaseModel):
    """``resources`` in server.json."""

    devices: dict[str, DeviceResourceConfig] = Field(default_factory=dict)
    acquire_timeout_seconds: float = Field(300.0, gt=0)


class ResidencyConfig(BaseModel):
    """Per-session ``residency`` in server.json."""

    device: str
    vram_mib: int = Field(gt=0)
    pinned: bool = False


class SessionBackend(Protocol):
    """What a session runs on. ``unload`` must return only after the device memory is released."""

    async def load(self) -> None: ...

    async def call(self, action: str, payload: Any = None) -> Any: ...

    async def unload(self) -> None: ...


class ProcessBackend:
    """A session hosted in its own child process (``WorkerProcessClient``)."""

    def __init__(
        self,
        *,
        name: str,
        target: Callable[..., None],
        args: tuple[Any, ...] = (),
        load_action: str | None = "load",
        load_payload: Any = None,
        stop_timeout: float = 10.0,
    ):
        self.client = WorkerProcessClient(name=name, target=target, args=args, stop_timeout=stop_timeout)
        self.load_action = load_action
        self.load_payload = load_payload

    @property
    def pid(self) -> int | None:
        process = self.client.process
        return process.pid if process is not None else None

    async def load(self) -> None:
        await asyncio.to_thread(self.client.start)
        if self.load_action is None:
            return
        try:
            await asyncio.to_thread(self.client.call, self.load_action, self.load_payload)
        except BaseException:
            await self.unload()
            raise

    async def call(self, action: str, payload: Any = None) -> Any:
        # WorkerProcessClient.call restarts a dead child on its own, but that child has no model
        # loaded. A session must never answer from it, so a dead child is reported as a death.
        process = self.client.process
        if process is None or not process.is_alive():
            raise ChildWorkerDied(self.client.name, "child is not running")
        return await asyncio.to_thread(self.client.call, action, payload)

    async def unload(self) -> None:
        await asyncio.to_thread(self.client.stop)


@dataclass(eq=False)
class InferenceSession:
    name: str
    backend: SessionBackend
    device: str
    vram_mib: int
    pinned: bool = False
    state: str = UNLOADED
    active_leases: int = 0
    job_mib: int = 0
    last_used: float = 0.0
    loads: int = 0


class Supervisor:
    """Admits leases against per-device budgets and evicts least-recently-used sessions."""

    def __init__(self, *, budgets: Mapping[str, int], acquire_timeout_seconds: float = 300.0):
        self._budgets = {device: int(mib) for device, mib in budgets.items()}
        self._timeout = float(acquire_timeout_seconds)
        self._sessions: dict[str, InferenceSession] = {}
        self._cond: asyncio.Condition | None = None

    @classmethod
    def from_config(cls, config: ResourcesConfig) -> Supervisor:
        return cls(
            budgets={device: entry.budget_mib for device, entry in config.devices.items()},
            acquire_timeout_seconds=config.acquire_timeout_seconds,
        )

    def add(self, name: str, backend: SessionBackend, residency: ResidencyConfig) -> InferenceSession:
        if name in self._sessions:
            raise ValueError(f"session {name} is already registered")
        budget = self._budgets.get(residency.device)
        if budget is None:
            raise ValueError(f"session {name}: no budget configured for {residency.device}")
        if residency.vram_mib > budget:
            raise ValueError(f"session {name} needs {residency.vram_mib} MiB but {residency.device} budget is {budget} MiB")
        session = InferenceSession(
            name=name,
            backend=backend,
            device=residency.device,
            vram_mib=residency.vram_mib,
            pinned=residency.pinned,
        )
        self._sessions[name] = session
        return session

    def session(self, name: str) -> InferenceSession:
        return self._sessions[name]

    async def start(self) -> None:
        """Load every pinned session. Fails if the pinned sessions do not fit together."""
        for device, budget in self._budgets.items():
            pinned_mib = sum(s.vram_mib for s in self._sessions.values() if s.pinned and s.device == device)
            if pinned_mib > budget:
                raise ResourceError(f"pinned sessions need {pinned_mib} MiB on {device} but the budget is {budget} MiB")
        for session in [s for s in self._sessions.values() if s.pinned]:
            await self.ensure_loaded(session)

    async def ensure_loaded(self, session: InferenceSession) -> None:
        async with self.lease(session):
            pass

    async def call(self, session: InferenceSession, action: str, payload: Any = None, *, job_mib: int = 0) -> Any:
        async with self.lease(session, job_mib=job_mib) as backend:
            return await backend.call(action, payload)

    @contextlib.asynccontextmanager
    async def lease(self, session: InferenceSession, *, job_mib: int = 0) -> AsyncIterator[SessionBackend]:
        if job_mib < 0:
            raise ValueError("job_mib must be >= 0")
        await self._admit(session, job_mib)
        try:
            yield session.backend
        except ChildWorkerDied:
            await self._discard(session)
            raise
        finally:
            await self._release(session, job_mib)

    async def close(self) -> None:
        """Unload every resident session (shutdown)."""
        cond = self._condition()
        async with cond:
            victims = [s for s in self._sessions.values() if s.state == READY]
            for victim in victims:
                victim.state = UNLOADING
        await self._unload(victims)

    def snapshot(self) -> dict[str, Any]:
        return {
            "devices": {
                device: {"budget_mib": budget, "used_mib": self._used(device)}
                for device, budget in self._budgets.items()
            },
            "sessions": [
                {
                    "name": s.name,
                    "device": s.device,
                    "vram_mib": s.vram_mib,
                    "pinned": s.pinned,
                    "state": s.state,
                    "active_leases": s.active_leases,
                    "loads": s.loads,
                }
                for s in self._sessions.values()
            ],
        }

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def _used(self, device: str) -> int:
        used = 0
        for session in self._sessions.values():
            if session.device != device:
                continue
            if session.state != UNLOADED:
                used += session.vram_mib
            used += session.job_mib
        return used

    def _busy(self, session: InferenceSession) -> bool:
        """Another session on this device is in use or changing state: its memory frees later."""
        return any(
            other is not session
            and other.device == session.device
            and (other.active_leases > 0 or other.state in (LOADING, UNLOADING))
            for other in self._sessions.values()
        )

    def _eviction_plan(self, session: InferenceSession, deficit: int) -> list[InferenceSession] | None:
        """Least-recently-used idle unpinned sessions on the device whose memory covers ``deficit``."""
        candidates = sorted(
            (
                other
                for other in self._sessions.values()
                if other is not session
                and other.device == session.device
                and other.state == READY
                and not other.pinned
                and other.active_leases == 0
            ),
            key=lambda other: other.last_used,
        )
        plan: list[InferenceSession] = []
        freed = 0
        for candidate in candidates:
            if freed >= deficit:
                break
            plan.append(candidate)
            freed += candidate.vram_mib
        return plan if freed >= deficit else None

    async def _wait(self, cond: asyncio.Condition, deadline: float, session: InferenceSession) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResourceError(f"timed out waiting for {session.device} memory for {session.name}")
        try:
            await asyncio.wait_for(cond.wait(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise ResourceError(f"timed out waiting for {session.device} memory for {session.name}") from exc

    async def _admit(self, session: InferenceSession, job_mib: int) -> None:
        cond = self._condition()
        deadline = time.monotonic() + self._timeout
        while True:
            async with cond:
                if session.state in (LOADING, UNLOADING):
                    await self._wait(cond, deadline, session)
                    continue
                budget = self._budgets[session.device]
                needs_load = session.state == UNLOADED
                need = job_mib + (session.vram_mib if needs_load else 0)
                if need > budget:
                    raise ResourceError(
                        f"{session.name} needs {need} MiB on {session.device} but the budget is {budget} MiB"
                    )
                free = budget - self._used(session.device)
                if free >= need:
                    if needs_load:
                        session.state = LOADING
                    session.active_leases += 1
                    session.job_mib += job_mib
                    session.last_used = time.monotonic()
                    break
                victims = self._eviction_plan(session, need - free)
                if victims:
                    for victim in victims:
                        victim.state = UNLOADING
                elif self._busy(session):
                    await self._wait(cond, deadline, session)
                    continue
                else:
                    raise ResourceError(
                        f"{session.name} needs {need} MiB on {session.device}; {free} MiB free and nothing "
                        "unpinned and idle can be evicted"
                    )
            await self._unload(victims)

        if needs_load:
            await self._load(session, job_mib)

    async def _load(self, session: InferenceSession, job_mib: int) -> None:
        started = time.monotonic()
        cond = self._condition()
        try:
            await session.backend.load()
        except BaseException:
            async with cond:
                session.state = UNLOADED
                session.active_leases -= 1
                session.job_mib -= job_mib
                cond.notify_all()
            raise
        async with cond:
            session.state = READY
            session.loads += 1
            cond.notify_all()
        _LOGGER.info(
            "Loaded %s on %s (%d MiB) in %.1fs",
            session.name,
            session.device,
            session.vram_mib,
            time.monotonic() - started,
        )

    async def _unload(self, victims: Sequence[InferenceSession]) -> None:
        if not victims:
            return
        results = await asyncio.gather(*(victim.backend.unload() for victim in victims), return_exceptions=True)
        cond = self._condition()
        async with cond:
            for victim, result in zip(victims, results):
                if isinstance(result, BaseException):
                    # The backend's stop escalates to SIGKILL, so the device memory is still released.
                    _LOGGER.error("Unloading %s reported an error: %s", victim.name, result)
                victim.state = UNLOADED
                _LOGGER.info("Unloaded %s from %s (%d MiB)", victim.name, victim.device, victim.vram_mib)
            cond.notify_all()

    async def _discard(self, session: InferenceSession) -> None:
        """The backend died under a lease: drop it so the next lease loads a fresh one."""
        cond = self._condition()
        async with cond:
            session.state = UNLOADING
        with contextlib.suppress(Exception):
            await session.backend.unload()
        async with cond:
            session.state = UNLOADED
            cond.notify_all()

    async def _release(self, session: InferenceSession, job_mib: int) -> None:
        cond = self._condition()
        async with cond:
            session.active_leases -= 1
            session.job_mib -= job_mib
            session.last_used = time.monotonic()
            cond.notify_all()
