"""Shared helpers of the Track-B (M2B) run scripts ``scripts/run_m2b_*.py``.

    STAGE, OUT                      registry stage "m2b_mlp" and its result directory results/m2b_mlp/
    S1_SEEDS, B0_SEEDS              seeds of the five independent 4x32 networks and of the B0 control copies
    registered_row(name, ckpt)      latest registry row of (STAGE, name) whose checkpoint SHA-256 equals the file's
                                    (lets a re-run after a crash between "row appended" and "done.json written" skip
                                    the evaluation instead of appending a duplicate row)
    stage1_epochs(hist)             (epochs run, best epoch) of a ``common.fit`` history, replaying its rule
    Resources                       wall-clock and peak CUDA memory of one phase
    free_cuda()                     release cached CUDA memory between jobs (the GPU is shared through the lock)

Nothing here touches the GPU lock, the test split or the registry file directly.
"""
from __future__ import annotations

import gc
import hashlib
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from sfm import gpulock, paths, registry  # noqa: E402

STAGE = "m2b_mlp"
OUT = paths.RESULTS_DIR / STAGE
S1_SEEDS = (11, 12, 13, 14, 15)
B0_SEEDS = (0, 1, 2)
STAGE1_RECIPE = {"model": "common.FieldModel(hidden=32, n_layers=4, n_freq=4)",
                 "trainer": "common.fit (unmodified)", "epochs": 60, "lr": 3e-3, "smooth": 1e-4, "patience": 15,
                 "batch": 32, "k_train": 16, "k_val_selection": 16, "min_improvement": 1e-4,
                 "seeding": "common.configure(batch=32) first (re-seeds with common.SEED=1), then random/numpy/torch "
                            "seeded with s immediately before the model is built; common.SEED = s during "
                            "common.fit (its shuffle generator is seeded with common.SEED + start_epoch), "
                            "restored afterwards"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def registered_row(name: str, ckpt=None, need_guards: bool = False):
    """The latest registry row of (STAGE, name) with the same checkpoint SHA-256 (and guards, if required)."""
    want = None
    if ckpt is not None:
        if not Path(ckpt).exists():
            return None
        want = sha256_file(ckpt)
    found = None
    for r in registry.read_registry():
        if r.get("stage") != STAGE or r.get("name") != name:
            continue
        if want is not None and r.get("checkpoint_sha256") != want:
            continue
        if need_guards and not r.get("guards"):
            continue
        found = r
    return found


def stage1_epochs(hist: dict) -> tuple:
    """(epochs run, best epoch) of a ``common.fit`` history; the best epoch (0-based, as common.fit counts) is the
    last epoch at which ``val < best - 1e-4`` reset the patience counter, i.e. the epoch whose state was kept."""
    best, best_ep = math.inf, None
    for i, v in enumerate(hist["val"]):
        if v < best - 1e-4:
            best, best_ep = v, i
    return len(hist["val"]), best_ep


class Resources:
    """with Resources() as r: ...   ->  r.seconds, r.peak_alloc_mb, r.peak_reserved_mb (CUDA peak inside the block)."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_alloc_mb = torch.cuda.max_memory_allocated() / 2 ** 20
            self.peak_reserved_mb = torch.cuda.max_memory_reserved() / 2 ** 20
        else:
            self.peak_alloc_mb = self.peak_reserved_mb = None
        self.seconds = time.time() - self.t0
        return False

    def as_dict(self) -> dict:
        return {"seconds": self.seconds, "peak_alloc_mb": self.peak_alloc_mb, "peak_reserved_mb": self.peak_reserved_mb}


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_on_gpu(label: str, fn, min_free_mb: float = 3000.0, poll: float = 30.0, max_wait: float = 6 * 3600.0):
    """Run ``fn()`` under ``sfm.gpulock.gpu_lock(label)`` once the device has at least ``min_free_mb`` free.

    The lock alone does not guarantee memory: a process that has released the lock can still hold cached CUDA
    memory (observed: another agent's process with ~7 GB while the lock was free, which made two jobs fail with
    OOM while loading data). So after acquiring the lock the free device memory is checked; if it is too small the
    lock is RELEASED again (never wait for memory while holding it), and the check is repeated after ``poll``
    seconds, up to ``max_wait`` seconds in total (then RuntimeError)."""
    t0 = time.time()
    while True:
        with gpulock.gpu_lock(label):
            free = torch.cuda.mem_get_info()[0] / 2 ** 20 if torch.cuda.is_available() else math.inf
            if free >= min_free_mb:
                return fn()
        waited = time.time() - t0
        print(f"[run_on_gpu] {label}: only {free:.0f} MB free on the device (need {min_free_mb:.0f}); lock released, "
              f"retry in {poll:.0f} s (waited {waited:.0f} s)", flush=True)
        if waited > max_wait:
            raise RuntimeError(f"{label}: device memory did not become free within {max_wait:.0f} s")
        time.sleep(poll)
