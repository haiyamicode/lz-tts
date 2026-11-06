from __future__ import annotations

import asyncio
from typing import Any

from .api.audio_utils import _audio_to_mp3_bytes
from .api.take_stream import Mp3Segment
from .taskflow_worker import (
    _PLAY_PRIORITY,
    ProtocolError,
    TaskflowWorker,
    _pull_leases,
    _SegmentUploads,
)

import numpy as np

LEASE = {"id": "lease-1", "leaseToken": "token", "payload": {}}


class FakeTaskflow(TaskflowWorker):
    def __init__(self, fail_segment: int | None = None, error: BaseException | None = None):
        super().__init__(base_url="http://taskflow.test/taskflow/v1", worker_token="w", worker_id="w")
        self.segments: list[tuple[int, Mp3Segment, bool]] = []
        self.uploads: list[bytes] = []
        self.pulls: list[dict[str, Any]] = []
        self.fail_segment = fail_segment
        self.error = error or ProtocolError("boom", 500)

    async def upload_segment(self, lease, artifact_id, index, segment, *, last):
        await asyncio.sleep(0)
        if index == self.fail_segment:
            raise self.error
        self.segments.append((index, segment, last))

    async def upload(self, lease, artifact_id, audio):
        self.uploads.append(audio)

    async def pull(self, **kwargs):
        self.pulls.append(kwargs)
        return []


def _take(seconds: float) -> bytes:
    t = np.arange(int(seconds * 48000)) / 48000
    return _audio_to_mp3_bytes((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)


def test_streamed_take_goes_up_in_order_and_ends_with_last():
    audio = _take(3.2)
    taskflow = FakeTaskflow()

    async def run():
        uploads = _SegmentUploads(taskflow, LEASE, "a" * 32)
        for start in range(0, len(audio), 5000):
            uploads.write(audio[start : start + 5000])
        await uploads.finish(audio)

    asyncio.run(run())
    assert [index for index, _, _ in taskflow.segments] == [0, 1, 2, 3]
    assert [last for _, _, last in taskflow.segments] == [False, False, False, True]
    assert b"".join(segment.data for _, segment, _ in taskflow.segments) == audio
    assert taskflow.uploads == []


def test_finished_take_is_segmented_at_the_end():
    audio = _take(1.5)
    taskflow = FakeTaskflow()
    asyncio.run(_SegmentUploads(taskflow, LEASE, "a" * 32).finish(audio))
    assert b"".join(segment.data for _, segment, _ in taskflow.segments) == audio
    assert taskflow.segments[-1][2] is True


def test_failed_segment_falls_back_to_a_whole_upload():
    audio = _take(3.2)
    taskflow = FakeTaskflow(fail_segment=1)

    async def run():
        uploads = _SegmentUploads(taskflow, LEASE, "a" * 32)
        uploads.write(audio)
        await uploads.finish(audio)

    asyncio.run(run())
    assert [index for index, _, _ in taskflow.segments] == [0]
    assert taskflow.uploads == [audio]


def test_lost_lease_is_not_retried_whole():
    taskflow = FakeTaskflow(fail_segment=0, error=ProtocolError("gone", 401, scope="lease"))

    async def run():
        await _SegmentUploads(taskflow, LEASE, "a" * 32).finish(_take(0.5))

    try:
        asyncio.run(run())
    except ProtocolError as error:
        assert error.loses_lease
    else:
        raise AssertionError("expected the lost lease to surface")
    assert taskflow.uploads == []


def test_pull_fills_base_slots_then_only_takes_play_boosts():
    taskflow = FakeTaskflow()
    taskflow.concurrency, taskflow.boost_slots, taskflow.low_priority_batch_size = 8, 3, 3

    async def run():
        await _pull_leases(taskflow, busy=2, background=0)
        await _pull_leases(taskflow, busy=5, background=3)
        await _pull_leases(taskflow, busy=8, background=0)
        assert await _pull_leases(taskflow, busy=11, background=0) == []

    asyncio.run(run())
    assert taskflow.pulls == [
        {"max_tasks": 6, "priority_bands": [{"below": 0, "maxTasks": 3}]},
        {"max_tasks": 3, "min_priority": 0},
        {"max_tasks": 3, "min_priority": _PLAY_PRIORITY},
    ]
