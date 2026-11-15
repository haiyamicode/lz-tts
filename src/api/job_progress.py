"""Where a running media job has got to, from the media runtime back to the lease that asked for it.

A videogen, lipsync, imagegen or music job holds this worker for minutes or hours on one blocking
POST to the media runtime, and until it answers the person who asked for it sees nothing at all.
The runtime says where it is as it goes, up the pipe it was spawned with
(``src/media_runtime/progress.py``); this is the other end: the sink the job's caller left here,
found again by the job's own id.

A job nobody is listening to — a /task/sync call, a bench run — reports into nothing, which costs
the render nothing.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from typing import Literal

#: The stages a job is reported in; they are LazyBird's own ``MediaJobStage``.
Stage = Literal["preparing", "rendering", "finishing"]

#: Told where a job is: the stage, and the chunks finished of the chunks there are.
ProgressSink = Callable[[Stage, int | None, int | None], None]

_LOCK = threading.Lock()
_SINKS: dict[str, ProgressSink] = {}


@contextlib.contextmanager
def listening(job_id: str, sink: ProgressSink | None) -> Iterator[None]:
    """Sends ``job_id``'s progress to ``sink`` for as long as the job runs."""
    if sink is None:
        yield
        return
    with _LOCK:
        _SINKS[job_id] = sink
    try:
        yield
    finally:
        with _LOCK:
            _SINKS.pop(job_id, None)


def report(job_id: str, stage: Stage, done: int | None = None, total: int | None = None) -> None:
    """Hands one report to the job's sink, if anything is listening for it."""
    with _LOCK:
        sink = _SINKS.get(job_id)
    if sink is not None:
        sink(stage, done, total)
