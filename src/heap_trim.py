"""Return freed malloc memory to the OS while a worker process is idle.

The models live on the GPU, but every request allocates host buffers (text
processing, audio decode/resample/encode, base64, queue payloads) from glibc
malloc. glibc keeps freed blocks in per-thread arenas (~90 of them in a torch
process with one thread per core) instead of returning them, so each worker
grew by a gigabyte or more of free-but-held heap. ``malloc_trim(0)`` hands the
free pages back. It costs 1-5 ms when little has accumulated and a few hundred
ms after heavy churn, so it only runs after the process has been idle for
IDLE_SECONDS following real work, never on the request path.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

_LOGGER = logging.getLogger(__name__)

IDLE_SECONDS = 2.0

try:
    _MALLOC_TRIM = ctypes.CDLL("libc.so.6").malloc_trim if sys.platform == "linux" else None
except (OSError, AttributeError):
    _MALLOC_TRIM = None


def release_freed_heap() -> None:
    """Return free glibc heap pages in every arena to the OS."""
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


class IdleHeapTrimmer:
    """Trim the heap once no work has been in flight for IDLE_SECONDS.

    For processes that serve requests on many threads/tasks (the main
    worker): wrap each unit of work in ``busy()``; a daemon thread trims after
    the last one finishes and the process stays idle.
    """

    def __init__(self, idle_seconds: float = IDLE_SECONDS):
        self._idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self._active = 0
        self._dirty = False
        self._last_done = 0.0
        self._thread: threading.Thread | None = None

    @contextmanager
    def busy(self) -> Iterator[None]:
        with self._lock:
            self._active += 1
            if self._thread is None and _MALLOC_TRIM is not None:
                self._thread = threading.Thread(target=self._run, name="heap-trim", daemon=True)
                self._thread.start()
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                self._dirty = True
                self._last_done = time.monotonic()

    def _run(self) -> None:
        while True:
            time.sleep(self._idle_seconds / 2)
            with self._lock:
                due = (
                    self._dirty
                    and self._active == 0
                    and time.monotonic() - self._last_done >= self._idle_seconds
                )
                if due:
                    self._dirty = False
            if due:
                try:
                    release_freed_heap()
                except Exception:  # pylint: disable=broad-exception-caught
                    _LOGGER.debug("malloc_trim failed", exc_info=True)
