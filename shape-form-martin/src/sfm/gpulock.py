"""sfm.gpulock - one heavy GPU job at a time: an exclusive ``flock`` on ``cache/gpu.lock``.

    with gpu_lock("fit b0_seed0"):
        ...                                   # training / evaluation

Blocking by default: a second job logs who holds the lock, waits, and logs how long it waited. The holder writes
its label, PID, a host id (hash, no host name) and start time into the lock file, so ``cat cache/gpu.lock`` shows who is running. The lock
is released when the block exits (also on exceptions) and automatically by the kernel if the process dies.
``timeout`` (seconds) turns the wait into polling that raises ``TimeoutError``. The lock file can be moved with the
environment variable ``SFM_GPU_LOCK``. ``flock`` locks belong to the open file description, so two ``gpu_lock``
blocks exclude each other even inside one process (do not nest them: that deadlocks without a timeout).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import paths

__all__ = ["gpu_lock", "lock_path", "holder"]


def lock_path() -> Path:
    v = os.environ.get("SFM_GPU_LOCK")
    return Path(v).expanduser() if v else paths.CACHE_DIR / "gpu.lock"


def holder(path: Optional[Path] = None) -> str:
    """The label line written by the current (or last) holder, or '' if none."""
    try:
        return Path(path or lock_path()).read_text().strip()
    except OSError:
        return ""


def _me(label: Optional[str]) -> str:
    script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "<interactive>"
    host_id = hashlib.sha256(b"sfm-host|" + socket.gethostname().encode()).hexdigest()[:12]   # no host name in files
    return (f"{label or script} | pid {os.getpid()} | host_id {host_id} | since "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}")


@contextlib.contextmanager
def gpu_lock(label: Optional[str] = None, path: Optional[Path] = None, log: Optional[Callable] = print,
             timeout: Optional[float] = None, poll: float = 1.0):
    path = Path(path or lock_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    me = _me(label)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            if log:
                log(f"[gpulock] {label or 'job'} (pid {os.getpid()}) waits for {path}; held by: {holder(path) or '?'}")
            t0 = time.time()
            if timeout is None:
                fcntl.flock(fd, fcntl.LOCK_EX)
                acquired = True
            else:
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except BlockingIOError:
                        if time.time() - t0 >= timeout:
                            raise TimeoutError(f"gpu lock {path} not acquired within {timeout} s "
                                               f"(held by: {holder(path) or '?'})")
                        time.sleep(min(poll, max(0.01, timeout / 10)))
            if log:
                log(f"[gpulock] {label or 'job'} acquired {path} after waiting {time.time() - t0:.0f} s")
        os.ftruncate(fd, 0)
        os.pwrite(fd, (me + "\n").encode(), 0)
        if log:
            log(f"[gpulock] held by {me}")
        yield me
    finally:
        if acquired:
            try:
                os.ftruncate(fd, 0)
            except OSError:
                pass
            fcntl.flock(fd, fcntl.LOCK_UN)
            if log:
                log(f"[gpulock] released by {label or 'job'} (pid {os.getpid()})")
        os.close(fd)
