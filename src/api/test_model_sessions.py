"""Supervisor tests with real child processes that hold real CUDA allocations.

Each session is a spawned child that allocates ``mib`` MiB on the GPU when loaded.
Eviction is checked against the real processes: an evicted child must be gone,
and a resident one must be alive.
"""

from __future__ import annotations

import asyncio
import os
import signal

import pytest
import torch

from src.api.model_sessions import (
    ChildWorkerDied,
    ProcessBackend,
    ResidencyConfig,
    ResourceError,
    Supervisor,
)
from src.api.worker_common import run_worker_loop

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

DEVICE = "cuda:0"
MIB = 2**20


def _cuda_worker(mib, request_queue, response_queue):
    """Child target: ``load`` allocates ``mib`` MiB on the GPU and keeps it; ``ping`` answers."""
    import torch as child_torch

    block = None

    def handler(action, payload):
        nonlocal block
        if action == "load":
            block = child_torch.empty(mib * MIB, dtype=child_torch.uint8, device=DEVICE)
            block.fill_(1)
            return {"ok": True, "pid": os.getpid(), "allocated_mib": child_torch.cuda.memory_allocated() // MIB}
        if action == "ping":
            return {"ok": True, "pid": os.getpid(), "allocated_mib": child_torch.cuda.memory_allocated() // MIB}
        return {"ok": False, "status_code": 400, "detail": f"unknown action {action}"}

    run_worker_loop("cuda-test", handler, request_queue, response_queue)


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _supervisor(budget_mib: int, sessions: dict[str, tuple[int, bool]]) -> Supervisor:
    supervisor = Supervisor(budgets={DEVICE: budget_mib}, acquire_timeout_seconds=60)
    for name, (mib, pinned) in sessions.items():
        supervisor.add(
            name,
            ProcessBackend(name=name, target=_cuda_worker, args=(mib,), stop_timeout=20),
            ResidencyConfig(device=DEVICE, vram_mib=mib, pinned=pinned),
        )
    return supervisor


async def _touch(supervisor: Supervisor, name: str) -> int:
    session = supervisor.session(name)
    response = await supervisor.call(session, "ping")
    return response["pid"]


def test_load_allocates_real_device_memory_and_lease_reuses_the_child():
    async def run():
        supervisor = _supervisor(300, {"a": (128, False)})
        session = supervisor.session("a")
        response = await supervisor.call(session, "ping")
        pid = response["pid"]
        assert response["allocated_mib"] >= 128
        assert session.state == "ready"
        assert session.loads == 1
        assert await _touch(supervisor, "a") == pid
        assert session.loads == 1
        await supervisor.close()
        assert not _alive(pid)

    asyncio.run(run())


def test_lru_evicts_least_recently_used_idle_session_and_frees_its_child():
    async def run():
        supervisor = _supervisor(400, {"a": (150, False), "b": (150, False), "c": (150, False)})
        pid_a = await _touch(supervisor, "a")
        pid_b = await _touch(supervisor, "b")
        # Use a again, so b is now the least recently used resident session.
        assert await _touch(supervisor, "a") == pid_a
        assert _alive(pid_a) and _alive(pid_b)

        pid_c = await _touch(supervisor, "c")

        assert not _alive(pid_b), "least recently used session must be unloaded"
        assert _alive(pid_a), "recently used session must stay resident"
        assert supervisor.session("b").state == "unloaded"
        assert supervisor.session("a").state == "ready"
        assert supervisor.session("c").state == "ready"
        await supervisor.close()
        assert not _alive(pid_a) and not _alive(pid_c)

    asyncio.run(run())


def test_pinned_session_is_never_evicted():
    async def run():
        supervisor = _supervisor(300, {"pinned": (200, True), "other": (200, False)})
        await supervisor.start()
        pinned_pid = supervisor.session("pinned").backend.pid
        assert _alive(pinned_pid)

        # Only 100 MiB is left and the pinned session is the only resident one: nothing may be evicted.
        with pytest.raises(ResourceError):
            await _touch(supervisor, "other")

        assert _alive(pinned_pid)
        assert supervisor.session("pinned").state == "ready"
        assert supervisor.session("other").state == "unloaded"
        await supervisor.close()

    asyncio.run(run())


def test_session_in_use_is_not_evicted_and_the_waiter_gets_in_after_release():
    async def run():
        supervisor = _supervisor(300, {"busy": (200, False), "next": (200, False)})
        busy = supervisor.session("busy")
        nxt = supervisor.session("next")

        async with supervisor.lease(busy) as backend:
            busy_pid = (await backend.call("ping"))["pid"]
            waiter = asyncio.create_task(_touch(supervisor, "next"))
            await asyncio.sleep(1.0)
            assert not waiter.done(), "must wait while the only other session holds a lease"
            assert _alive(busy_pid)
            assert nxt.state == "unloaded"

        next_pid = await asyncio.wait_for(waiter, timeout=60)
        assert not _alive(busy_pid), "idle session is evicted once its lease is released"
        assert _alive(next_pid)
        await supervisor.close()

    asyncio.run(run())


def test_a_job_that_cannot_fit_even_alone_is_refused_at_once():
    async def run():
        supervisor = _supervisor(200, {"a": (128, False)})
        session = supervisor.session("a")
        with pytest.raises(ResourceError):
            async with supervisor.lease(session, job_mib=150):
                pass
        assert session.state == "unloaded"
        assert session.active_leases == 0

    asyncio.run(run())


def test_job_memory_counts_against_the_budget_and_forces_eviction():
    async def run():
        # a (128) is resident; b (128) + its job reservation (50) = 178 > 300 - 128, so a must go.
        supervisor = _supervisor(300, {"a": (128, False), "b": (128, False)})
        pid_a = await _touch(supervisor, "a")
        async with supervisor.lease(supervisor.session("b"), job_mib=50):
            assert not _alive(pid_a)
            assert supervisor.session("a").state == "unloaded"
        await supervisor.close()

    asyncio.run(run())


def test_child_death_under_lease_is_reported_and_the_next_lease_reloads():
    async def run():
        supervisor = _supervisor(300, {"a": (128, False)})
        session = supervisor.session("a")
        await _touch(supervisor, "a")
        process = session.backend.client.process
        first_pid = process.pid

        os.kill(first_pid, signal.SIGKILL)
        process.join(timeout=30)
        assert not process.is_alive()

        with pytest.raises(ChildWorkerDied):
            await supervisor.call(session, "ping")
        assert session.state == "unloaded"

        second_pid = await _touch(supervisor, "a")
        assert second_pid != first_pid
        assert session.loads == 2
        await supervisor.close()

    asyncio.run(run())


def test_config_is_checked_before_anything_loads():
    supervisor = Supervisor(budgets={DEVICE: 200}, acquire_timeout_seconds=5)
    with pytest.raises(ValueError):
        supervisor.add(
            "too-big",
            ProcessBackend(name="too-big", target=_cuda_worker, args=(10,)),
            ResidencyConfig(device=DEVICE, vram_mib=300),
        )
    with pytest.raises(ValueError):
        supervisor.add(
            "no-device",
            ProcessBackend(name="no-device", target=_cuda_worker, args=(10,)),
            ResidencyConfig(device="cuda:7", vram_mib=10),
        )

    overfull = _supervisor(200, {"p1": (120, True), "p2": (120, True)})
    with pytest.raises(ResourceError):
        asyncio.run(overfull.start())
    assert all(s["state"] == "unloaded" for s in overfull.snapshot()["sessions"])
