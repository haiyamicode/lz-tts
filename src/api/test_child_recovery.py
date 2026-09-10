"""Process-level tests for backend supervision: child death detection.

These exercise the real contract that keeps a headless worker recovering: when
a backend child dies its in-flight requests fail instead of hanging, and no
child process outlives the worker that spawned it.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from src.nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server, VoxCPMServerDied

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stalled_child(queue_in, queue_out, args, kwargs) -> None:
    """Stand-in for ``main_loop``: announces readiness, then stalls on ops."""
    queue_out.put({"type": "init_ok"})
    while True:
        command = queue_in.get()
        if command.get("type") == "stop":
            queue_out.put({"type": "response", "id": command["id"], "data": None})
            return
        # No response on purpose: simulates a child stuck inside a generation.


def test_voxcpm_child_death_fails_pending_requests() -> None:
    async def scenario() -> None:
        server = AsyncVoxCPM2Server(model_path="unused", target=_stalled_child)
        try:
            await server.wait_for_ready()
            pending = asyncio.create_task(server.submit("health"))
            stream = server.generate("hello")
            streaming = asyncio.ensure_future(stream.__anext__())
            await asyncio.sleep(0.2)
            assert not pending.done()
            assert not streaming.done()

            server.process.kill()

            with pytest.raises(VoxCPMServerDied):
                await asyncio.wait_for(pending, timeout=10)
            with pytest.raises(VoxCPMServerDied):
                await asyncio.wait_for(streaming, timeout=10)
            # The death is sticky: later work fails instead of hanging.
            with pytest.raises(VoxCPMServerDied):
                await asyncio.wait_for(asyncio.create_task(server.submit("health")), timeout=1)
        finally:
            await server.stop()

    asyncio.run(scenario())


_HARNESS = """
import multiprocessing as mp
import time


def worker_child(request_queue, response_queue):
    from src.api.worker_common import run_worker_loop

    run_worker_loop(
        "harness",
        lambda action, payload: {"ok": True, "data": None},
        request_queue,
        response_queue,
    )


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    child = ctx.Process(target=worker_child, args=(request_queue, response_queue))
    child.start()
    # Only report the child once it answers, i.e. once it is past its startup
    # (which is where the child registers for parent-death cleanup).
    request_queue.put({"request_id": "ready", "action": "ping", "payload": None})
    assert response_queue.get(timeout=120).get("ok") is True
    print(child.pid, flush=True)
    time.sleep(120)
"""


def _process_is_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, IndexError):
        return False
    return state != "Z"


def test_backend_child_dies_with_the_worker_process(tmp_path: Path) -> None:
    script = tmp_path / "child_harness.py"
    script.write_text(textwrap.dedent(_HARNESS))
    harness = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        text=True,
    )
    try:
        assert harness.stdout is not None
        child_pid = None
        while child_pid is None:
            line = harness.stdout.readline()
            assert line, "harness exited before reporting its child pid"
            token = line.strip()
            if token.isdigit():
                child_pid = int(token)
        assert _process_is_alive(child_pid)

        os.kill(harness.pid, signal.SIGKILL)
        harness.wait(timeout=10)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _process_is_alive(child_pid):
            time.sleep(0.1)
        assert not _process_is_alive(child_pid), "backend child outlived the killed worker process"
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait(timeout=10)
