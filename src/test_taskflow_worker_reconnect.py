from __future__ import annotations

import asyncio

import httpx
import pytest

from . import taskflow_worker
from .taskflow_worker import ProtocolError, TaskflowWorker, _serve_taskflow, _transient_failure

GONE = ProtocolError("Worker has not joined", 404, "INVALID_WORKER", scope="session")


class FakeTaskflow(TaskflowWorker):
    def __init__(self, beats: list[BaseException | None] | None = None, pulls=None):
        super().__init__(base_url="http://taskflow.test/taskflow/v1", worker_token="w", worker_id="w")
        self.heartbeat_interval = 0.05
        self.beats = list(beats or [])
        self.beat_count = 0
        self.joins = 0
        self.pull_results = list(pulls or [])
        self.pull_count = 0
        self.reconnects = 0

    async def reset_client(self):
        self.reconnects += 1

    async def join(self, metadata):
        self.joins += 1
        self.session_id = "s"
        self.session_token = "t"

    async def heartbeat(self):
        self.beat_count += 1
        outcome = self.beats.pop(0) if self.beats else None
        if outcome is not None:
            raise outcome

    async def pull(self, **kwargs):
        self.pull_count += 1
        outcome = self.pull_results.pop(0) if self.pull_results else []
        if isinstance(outcome, BaseException):
            raise outcome
        await asyncio.sleep(0.01)
        return outcome


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(taskflow_worker, "_CONTROL_RETRY_DELAY", 0.01)


async def _beat_for(taskflow: FakeTaskflow, seconds: float) -> asyncio.Event:
    lost = asyncio.Event()
    task = asyncio.create_task(taskflow.heartbeat_loop(lost))
    await asyncio.sleep(seconds)
    task.cancel()
    return lost


def test_a_stalled_heartbeat_is_retried_without_dropping_the_session():
    taskflow = FakeTaskflow(beats=[httpx.ReadTimeout("stalled"), None, None])
    lost = asyncio.run(_beat_for(taskflow, 0.2))
    assert not lost.is_set()
    assert taskflow.beat_count >= 3


def test_the_server_ending_the_session_reconnects():
    taskflow = FakeTaskflow(beats=[GONE])
    lost = asyncio.run(_beat_for(taskflow, 0.1))
    assert lost.is_set()


def test_heartbeats_failing_for_a_whole_session_lifetime_reconnect():
    taskflow = FakeTaskflow(beats=[httpx.ConnectTimeout("down")] * 1000)
    lost = asyncio.run(_beat_for(taskflow, 0.3))
    assert lost.is_set()


def test_transient_failures():
    assert _transient_failure(httpx.ReadTimeout("x"))
    assert _transient_failure(httpx.RemoteProtocolError("x"))
    assert _transient_failure(ProtocolError("bad gateway", 502, scope="session"))
    assert not _transient_failure(GONE)
    assert not _transient_failure(ProtocolError("bad request", 400, "INVALID_INPUT", scope="session"))
    assert not _transient_failure(ProtocolError("Worker has not joined Taskflow"))


def test_a_failed_pull_is_retried_in_the_same_session(monkeypatch):
    monkeypatch.setattr(taskflow_worker, "_update_pull_task_types", lambda taskflow: None)
    taskflow = FakeTaskflow(
        pulls=[httpx.ReadTimeout("stalled"), ProtocolError("bad gateway", 502, scope="session")]
    )

    async def run():
        task = asyncio.create_task(_serve_taskflow(taskflow, None, {}))  # type: ignore[arg-type]
        while taskflow.pull_count < 5:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert taskflow.joins == 1
    assert taskflow.reconnects == 0
