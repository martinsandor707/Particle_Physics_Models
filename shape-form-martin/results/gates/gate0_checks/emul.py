"""Gate 0 helper: emulate torch.rand on CUDA (float32) for a GPU with a given launch grid, on the CPU.

    gen = EmulGen(seed=12345, grid=G)        # G = multiProcessorCount * (maxThreadsPerMultiProcessor // 256)
    u = gen.rand(H, K, 3, device="cuda")      # what torch.rand(H, K, 3, device="cuda", generator=g) returns on that GPU

See philox_emul.c for the algorithm. `patched_fold(gen)` swaps common.fold for a version that draws from the
emulated stream (in memory only; common.py is not modified), so the unmodified common.event_nll / test_nll code
runs on those points.
"""
import contextlib
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SO = HERE / "philox_emul.so"
if not SO.exists():
    subprocess.run(["gcc", "-O2", "-fopenmp", "-shared", "-fPIC", "-ffp-contract=off", str(HERE / "philox_emul.c"),
                    "-o", str(SO)], check=True)
_lib = ctypes.CDLL(str(SO))
_lib.philox_fill.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
_lib.philox_fill.restype = None


def this_gpu_grid_candidates():
    """Launch grid of the local GPU: SM count is exposed by torch; threads per SM is 1024, 1536 or 2048."""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    return sm, {t: sm * (t // 256) for t in (1024, 1536, 2048)}


class EmulGen:
    def __init__(self, seed=12345, grid=144):
        self.seed, self.grid, self.offset = int(seed), int(grid), 0

    def rand(self, *shape, device="cuda"):
        numel = int(np.prod(shape))
        g = min(self.grid, (numel + 255) // 256)
        T = 256 * g
        out = np.empty(numel, dtype=np.float32)
        _lib.philox_fill(self.seed, self.offset, T, numel, out.ctypes.data_as(ctypes.c_void_p))
        self.offset += ((numel - 1) // (T * 4) + 1) * 4
        return torch.from_numpy(out).reshape(shape).to(device)


@contextlib.contextmanager
def patched_fold(common, gen: EmulGen):
    """common.fold drawing from `gen` (the `gen` argument passed by event_nll is ignored)."""
    orig = common.fold

    def fold(b, box, K, gen_ignored=None):
        u = (gen.rand(len(b.xyz), K, 3, device=b.xyz.device) - 0.5) * box
        return b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", b.R[b.ev], u)

    common.fold = fold
    try:
        yield
    finally:
        common.fold = orig


@torch.no_grad()
def test_nll_emul(common, model, data, grid, K=None, seed=12345):
    """common.test_nll with the fold points of a GPU whose launch grid is `grid` (same loop, same batches)."""
    K = common.N_FOLD_EVAL if K is None else K
    gen = EmulGen(seed, grid)
    with patched_fold(common, gen):
        return torch.cat([common.event_nll(model, b, data.box, K, None)[0] for b in data.batches(128)]).cpu().numpy()
