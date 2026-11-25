"""sfm.refstream - the CUDA random stream of ``torch.rand`` emulated on the CPU, for any GPU launch grid.

Why (Gate 0, PLAN.md section 13). ``common.test_nll`` draws its fold points with
``torch.rand(H, K, 3, device="cuda", generator=g)``. PyTorch launches ``grid`` blocks of 256 threads,
``grid = min(SMs * (maxThreadsPerSM // 256), ceil(numel / 256))``; thread ``idx`` owns the Philox4x32-10
subsequence ``idx`` and fills the elements ``idx + T (4n + i)`` with ``T = 256 grid``. For tensors larger than ``T``
the stream therefore depends on the GPU model. This module re-implements that kernel (``_philox_emul.c``) so the
fold points of any GPU can be produced here:

    local_grid()    the launch grid of the GPU this process runs on (from its properties; never a constant)
    LAPTOP_GRID     = 144    the laptop of 24 Nov 2026 (RTX 4060 Laptop: 24 SMs x 1536/256), whose arrays are
                             records; used to emulate the laptop stream on another machine (row R1b)
    REFERENCE_GRID  = 1056   the reference machine of the earlier study (132 SMs x 8, H100 class), identified in
                             Gate 0 by reproducing all 13 stored validation means to <= 1.9e-6

    test_nll_ref(model, data, grid=REFERENCE_GRID)   ``common.test_nll`` on the fold points of that GPU
    EmulGen(seed, grid).rand(*shape)                 what ``torch.rand(*shape, device="cuda", generator=g)`` gives
    patched_draw(grid)                               ``sfm.metric.draw`` (i.i.d. fold) on that GPU's points, so the
                                                     guard evaluations (K = 1024, bulk) can be emulated as well

Validated in Gate 0 (``results/gates/gate0_checks/{philox_emul.c, emul.py, c02, c04}``); this file and
``_philox_emul.c`` are copies of those (the originals stay in place). ``tests/test_refstream.py`` repeats the checks.

Use. Reference-stream numbers are a verification instrument and an optional, pre-registered secondary column
(they allow paired deltas against the stored per-event arrays of the earlier study). They are never the headline:
the headline is the unmodified ``common.test_nll`` on the machine of record (PLAN.md section 13, A6 revised). On a
GPU whose launch grid equals the emulated one, the emulated and the native points are identical.

Build. The shared object is compiled on first use into ``cache/`` (git-ignored) as ``philox_emul_<sha>.so``, where
``<sha>`` is a hash of the C source, with ``gcc -O2 -fopenmp -shared -fPIC -ffp-contract=off``. It is never committed.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import paths

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common  # noqa: E402  (shape-form-martin/common.py, the metric of record)

__all__ = ["LAPTOP_GRID", "REFERENCE_GRID", "THREADS_PER_BLOCK", "local_grid", "grid_for", "EmulGen", "patched_fold",
           "patched_draw", "test_nll_ref", "library_path"]

THREADS_PER_BLOCK = 256
LAPTOP_GRID = 144            # the laptop of record until 25 Nov 2026 (RTX 4060 Laptop GPU: 24 SMs x (1536 // 256))
REFERENCE_GRID = 1056        # reference machine of the stored arrays: 132 SMs x (2048 // 256)

C_SOURCE = Path(__file__).resolve().parent / "_philox_emul.c"
CFLAGS = ("-O2", "-fopenmp", "-shared", "-fPIC", "-ffp-contract=off")
_lib = None


def grid_for(multi_processor_count: int, max_threads_per_multi_processor: int) -> int:
    """Launch grid of PyTorch's elementwise random kernels on a GPU with these properties."""
    return int(multi_processor_count) * (int(max_threads_per_multi_processor) // THREADS_PER_BLOCK)


def local_grid(device: int = 0) -> int:
    """Launch grid of the local CUDA device (from its properties)."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: the local grid is undefined")
    p = torch.cuda.get_device_properties(device)
    return grid_for(p.multi_processor_count, p.max_threads_per_multi_processor)


def library_path() -> Path:
    """Location of the compiled emulator in the cache directory (depends on the C source and the flags)."""
    h = hashlib.sha256(C_SOURCE.read_bytes() + " ".join(CFLAGS).encode()).hexdigest()[:16]
    return paths.CACHE_DIR / f"philox_emul_{h}.so"


def _load():
    """Compile (once, atomically: temp file + rename) and load the shared object."""
    global _lib
    if _lib is not None:
        return _lib
    so = library_path()
    if not so.exists():
        so.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=so.stem + ".", suffix=".tmp.so", dir=so.parent)
        os.close(fd)
        try:
            subprocess.run(["gcc", *CFLAGS, str(C_SOURCE), "-o", tmp], check=True)
            os.replace(tmp, so)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    lib = ctypes.CDLL(str(so))
    lib.philox_fill.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.philox_fill.restype = None
    _lib = lib
    return lib


class EmulGen:
    """A CUDA ``torch.Generator`` seeded with ``seed`` on a GPU with launch grid ``grid``, emulated on the CPU.

    ``rand(*shape)`` returns what ``torch.rand(*shape, device="cuda", generator=g)`` returns on that GPU (float32),
    and advances the Philox offset exactly as PyTorch does: ``((numel - 1) // (4 T) + 1) * 4`` per call, with
    ``T = 256 * min(grid, ceil(numel / 256))``. Consecutive calls therefore reproduce consecutive calls on one
    generator. The result is placed on ``device`` (default: ``common.DEVICE``).
    """

    def __init__(self, seed: int = 12345, grid: int = REFERENCE_GRID):
        self.seed, self.grid, self.offset = int(seed), int(grid), 0
        if self.grid < 1:
            raise ValueError("grid must be a positive integer")

    def rand(self, *shape, device=None) -> torch.Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        numel = int(np.prod(shape)) if shape else 1
        if numel == 0:
            return torch.empty(shape, dtype=torch.float32, device=device or common.DEVICE)
        g = min(self.grid, (numel + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK)
        T = THREADS_PER_BLOCK * g
        out = np.empty(numel, dtype=np.float32)
        _load().philox_fill(self.seed, self.offset, T, numel, out.ctypes.data_as(ctypes.c_void_p))
        self.offset += ((numel - 1) // (T * 4) + 1) * 4
        return torch.from_numpy(out).reshape(shape).to(device or common.DEVICE)


@contextlib.contextmanager
def patched_fold(gen: EmulGen):
    """Temporarily replace ``common.fold`` (in memory only; ``common.py`` is not modified) by a version that draws
    its offsets from ``gen``; the ``gen`` argument that ``common.event_nll`` passes is ignored. The unmodified
    ``common.event_nll`` then runs on the emulated points. Not thread-safe (it swaps a module attribute)."""
    orig = common.fold

    def fold(b, box, K, gen_ignored=None):
        u = (gen.rand(len(b.xyz), K, 3, device=b.xyz.device) - 0.5) * box
        return b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", b.R[b.ev], u)

    common.fold = fold
    try:
        yield
    finally:
        common.fold = orig


@contextlib.contextmanager
def patched_draw(grid: int, seed: int = 12345):
    """Temporarily replace ``sfm.metric.draw`` (in memory only) by a version whose i.i.d. offsets come from
    ``EmulGen(seed, grid)``; the generator argument is ignored. ``sfm.metric.test_nll`` / ``test_nll_multi`` re-create
    their generator once per call with the same seed, so each call inside this block must be preceded by a fresh
    block (one block = one evaluation). Only ``fold="iid"`` is supported. Not thread-safe."""
    from . import metric
    gen = EmulGen(seed, grid)
    orig = metric.draw

    def draw(b, box, K, gen_ignored=None, fold="iid"):
        if fold != "iid":
            raise NotImplementedError("patched_draw emulates the i.i.d. fold only")
        return (gen.rand(len(b.xyz), K, 3, device=b.xyz.device) - 0.5) * box

    metric.draw = draw
    try:
        yield gen
    finally:
        metric.draw = orig


@torch.no_grad()
def test_nll_ref(model, data, grid: int = REFERENCE_GRID, K: int = common.N_FOLD_EVAL, seed: int = 12345,
                 batch: int = 128) -> np.ndarray:
    """``common.test_nll(model, data, K)`` on the fold points of a GPU with launch grid ``grid``.

    Same loop as ``common.test_nll``: generator seed ``seed`` (12345), ``data.batches(batch)`` (128) in storage order,
    ``common.event_nll`` unmodified. With ``grid=local_grid()`` the points are those of this GPU, so inside
    ``sfm.metric.deterministic()`` the result equals ``common.test_nll`` bit for bit. With ``grid=REFERENCE_GRID``
    it reproduces the arrays of the earlier study's machine (validation means to <= 1.9e-6; on CUDA the per-event
    sums carry the usual ~3e-5 accumulation-order noise of ``index_add_`` outside ``deterministic()``).
    """
    gen = EmulGen(seed, grid)
    with patched_fold(gen):
        return torch.cat([common.event_nll(model, b, data.box, K, None)[0] for b in data.batches(batch)]).cpu().numpy()


test_nll_ref.__test__ = False        # not a pytest test, whatever module imports it
