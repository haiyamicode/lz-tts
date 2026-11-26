from __future__ import annotations

import asyncio
from typing import Any

from .api.audio_utils import _audio_to_mp3_bytes
from .api.take_stream import Mp3Segment
from .taskflow_worker import (
    ProtocolError,
    SynthesisAckClient,
    TaskflowWorker,
    TASK_TYPES,
    TaskGroup,
    _pull_leases,
    task_groups,
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


def test_pull_fills_free_slots_only():
    taskflow = FakeTaskflow()
    taskflow.concurrency, taskflow.low_priority_batch_size = 8, 3
    taskflow.task_types = ("tts-synthesis", "voice-design")

    async def run():
        await _pull_leases(taskflow, {"tts-synthesis": 2}, background=0)
        await _pull_leases(taskflow, {"tts-synthesis": 3, "voice-design": 2}, background=3)
        assert await _pull_leases(taskflow, {"tts-synthesis": 8}, background=0) is None

    asyncio.run(run())
    types = ("tts-synthesis", "voice-design")
    assert taskflow.pulls == [
        {"max_tasks": 6, "priority_bands": [{"below": 0, "maxTasks": 3}], "task_types": types},
        {"max_tasks": 3, "min_priority": 0, "task_types": types},
    ]


def test_only_voxcpm_types_run_several_at_once():
    groups = task_groups(TASK_TYPES, 8)
    assert {g.task_types: g.concurrency for g in groups} == {
        ("tts-synthesis", "tts-voxcpm", "voice-design"): 8,
        ("generate-music", "videogen", "videogen-chain-decode", "lipsync", "imagegen"): 1,
        **{(t,): 1 for t in ("voice-enhance", "nudity-detect", "face-detect", "language-detect",
                             "transcribe", "force-align")},
    }
    # Only served types, and no empty groups.
    assert task_groups(("transcribe", "lipsync"), 8) == (TaskGroup(("lipsync",), 1), TaskGroup(("transcribe",), 1))


def test_pull_skips_full_groups():
    taskflow = FakeTaskflow()
    taskflow.concurrency, taskflow.low_priority_batch_size = 8, 3
    taskflow.task_types = ("tts-synthesis", "voice-design", "transcribe", "videogen", "lipsync")
    taskflow.pull_task_types = ("tts-synthesis", "transcribe", "videogen", "lipsync")  # on-demand narrowing

    async def run():
        # A lipsync running fills the media group; one transcribe fills its own.
        await _pull_leases(taskflow, {"lipsync": 1, "transcribe": 1, "tts-synthesis": 2}, background=0)
        # Only one-at-a-time types ready: no more than their free slots.
        taskflow.pull_task_types = ("transcribe", "videogen")
        await _pull_leases(taskflow, {}, background=0)
        assert await _pull_leases(taskflow, {"transcribe": 1, "videogen": 1}, background=0) is None

    asyncio.run(run())
    assert [(pull["task_types"], pull["max_tasks"]) for pull in taskflow.pulls] == [
        (("tts-synthesis",), 4),  # 6 free VoxCPM slots, but 4 left under the box concurrency
        (("transcribe", "videogen"), 2),
    ]


class FakeAckResponse:
    is_error = False

    def __init__(self, items):
        self._items = items

    def json(self):
        return {"results": [{"runId": item["runId"], "ok": item["runId"] != "bad"} for item in self._items]}


class FakeAckHttp:
    def __init__(self):
        self.requests: list[list[str]] = []

    async def post(self, endpoint, json):
        self.requests.append([item["runId"] for item in json["items"]])
        await asyncio.sleep(0.01)
        return FakeAckResponse(json["items"])


def test_acks_finishing_meanwhile_go_in_one_request():
    taskflow = FakeTaskflow()
    http = FakeAckHttp()
    taskflow._client = http  # type: ignore[assignment]  # pylint: disable=protected-access
    acks = SynthesisAckClient(taskflow, "/acks")

    async def run():
        first = asyncio.create_task(acks.send([{"runId": "a"}]))
        await asyncio.sleep(0)
        rest = [asyncio.create_task(acks.send([{"runId": run_id}])) for run_id in ("b", "bad", "c")]
        return await asyncio.gather(first, *rest)

    results = asyncio.run(run())
    assert http.requests[0] == ["a"]
    assert http.requests[1][:1] == ["b"] and sorted(http.requests[1]) == ["b", "bad", "c"]
    assert [list(result) for result in results] == [[], [], ["bad"], []]
