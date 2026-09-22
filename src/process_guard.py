"""Keep backend child processes from outliving the worker process."""

from __future__ import annotations

import ctypes
import logging
import os
import signal

_LOGGER = logging.getLogger(__name__)


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


def _process_children() -> dict[int, list[int]]:
    """Map ppid -> [pid] for every live process, read from /proc."""
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return children
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        # The comm field can contain spaces and parentheses, so parse the
        # fields after the final ')': state, ppid, ...
        close = data.rfind(b")")
        if close < 0:
            continue
        fields = data[close + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            ppid = int(fields[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(int(entry))
    return children


def descendant_pids(root_pid: int | None = None) -> list[int]:
    """Return every live descendant of ``root_pid``, parents before children."""
    root = os.getpid() if root_pid is None else root_pid
    children = _process_children()
    ordered: list[int] = []
    seen: set[int] = set()
    stack = list(children.get(root, []))
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        stack.extend(children.get(pid, []))
    return ordered


def kill_process_tree(sig: int = signal.SIGKILL) -> list[int]:
    """Signal every descendant of this process, deepest first.

    ``PR_SET_PDEATHSIG`` already reaps our direct children when we die, but
    this also covers any process that failed to set the flag and frees GPU
    memory before the worker itself goes down.
    """
    pids = descendant_pids()
    for pid in reversed(pids):  # children before their parents
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            continue
    return pids


def hard_exit(reason: str, *, code: int = 1) -> None:
    """Kill the entire worker process tree and then SIGKILL ourselves.

    In-process engine reloads can fail forever (GPU genuinely exhausted, a
    child stuck in an uninterruptible CUDA call). The only state that reliably
    frees every CUDA context is a brand-new process, which is exactly what a
    manual ``pm2 restart`` gives us. So when recovery is not working, tear the
    whole tree down and let the supervisor restart the worker cleanly instead
    of looping with resident GPU memory.

    ``SIGKILL`` to ourselves is deliberate: an ``os._exit`` still runs
    ``multiprocessing`` atexit handlers, which can block on a wedged child.
    """
    _LOGGER.error("Hard-exiting worker pid=%s: %s", os.getpid(), reason)
    try:
        killed = kill_process_tree()
    except Exception:  # pylint: disable=broad-exception-caught
        _LOGGER.exception("Failed to enumerate/kill descendant processes")
        killed = []
    if killed:
        _LOGGER.error(
            "Sent SIGKILL to %d descendant process(es) before exit: %s",
            len(killed),
            killed,
        )
    logging.shutdown()
    try:
        os.kill(os.getpid(), signal.SIGKILL)
    except OSError:
        pass
    os._exit(code)  # pragma: no cover - only reached if SIGKILL was blocked
