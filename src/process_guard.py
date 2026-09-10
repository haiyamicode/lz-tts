"""Keep backend child processes from outliving the worker process."""

from __future__ import annotations

import ctypes
import os
import signal


def exit_with_parent() -> None:
    """Ask the kernel to kill this process when its parent process exits.

    Every backend worker child (Sparrow, Starling, Seed-VC, VoxCPM) owns a
    multi-gigabyte CUDA context. If the worker process is SIGKILLed (pm2 kill
    timeout, OOM killer) plain ``multiprocessing`` cannot clean them up, and the
    orphans keep the GPU memory while the restarted worker tries to load its
    models again. ``PR_SET_PDEATHSIG`` makes the kernel reap them immediately.

    The flag is cleared by ``fork`` but survives ``execve``, so the child has to
    set it for itself; the ``getppid`` re-check closes the race where the parent
    died before the flag was set.
    """
    parent_pid = os.getppid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        # Non-Linux platform or missing libc: fall back to the existing
        # multiprocessing cleanup behaviour.
        return
    if os.getppid() != parent_pid:
        os._exit(1)
