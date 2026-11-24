"""Tests of sfm.refstream (CPU emulation of the CUDA random stream of torch.rand).

1. With the local launch grid the emulation reproduces ``torch.rand`` on this GPU bit for bit: consecutive calls of
   different sizes on one generator (checks the offset rule), and fresh generators with other seeds.
2. On the validation split (cache needed): ``test_nll_ref(mlp_std, val, grid=<local>)`` equals ``common.test_nll``
   bit for bit inside ``metric.deterministic()``, and with the reference grid 1056 the validation mean of ``mlp_std``
   is 18.118082 +- 4e-6 (Gate 0; the earlier study stored 18.118084).
No test split is opened. GPU: a few seconds, two validation evaluations of mlp_std.
"""
import re

import numpy as np
import pytest
import torch

import common as C
from sfm import metric, paths, refstream

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
TRAINVAL_CACHE = paths.CACHE_DIR / "pion_s1_trainval.pt"
GATE0 = paths.GATES_DIR / "gate0_checks"
SHAPES = [(1000,), (36_863,), (36_864,), (36_865,), (147_456,), (147_457,), (35_000, 64, 3), (7, 16, 3),
          (41_237, 64, 3), (5,)]


def _code(path):
    """C source without its leading comment block (the header was edited, the code must not be)."""
    return re.sub(r"^/\*.*?\*/\s*", "", path.read_text(), count=1, flags=re.S)


def test_c_source_is_the_gate0_emulator():
    orig = GATE0 / "philox_emul.c"
    if not orig.exists():
        pytest.skip("gate-0 emulator not present")
    assert _code(refstream.C_SOURCE) == _code(orig)


def test_library_is_built_in_the_cache_dir():
    so = refstream.library_path()
    refstream._load()
    assert so.exists() and so.parent == paths.CACHE_DIR and so.suffix == ".so"
    assert not list(refstream.C_SOURCE.parent.glob("*.so"))           # nothing compiled next to the sources


def test_grid_rule():
    assert refstream.grid_for(24, 1536) == refstream.LOCAL_GRID == 144
    assert refstream.grid_for(132, 2048) == refstream.REFERENCE_GRID == 1056


@cuda
def test_local_grid_constant_matches_this_gpu():
    assert refstream.local_grid() == refstream.LOCAL_GRID


@cuda
def test_emulation_reproduces_torch_rand_bitwise():
    G = refstream.local_grid()
    g = torch.Generator(device="cuda").manual_seed(12345)
    e = refstream.EmulGen(12345, G)
    for shp in SHAPES:                                               # one generator: also checks the offset advance
        a = torch.rand(*shp, device="cuda", generator=g)
        b = e.rand(*shp, device="cuda")
        assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
        assert torch.equal(a, b), shp
    for seed in (0, 1, 12345, 2 ** 40 + 7):
        g = torch.Generator(device="cuda").manual_seed(seed)
        assert torch.equal(torch.rand(300_001, 3, device="cuda", generator=g),
                           refstream.EmulGen(seed, G).rand(300_001, 3, device="cuda")), seed


@cuda
def test_other_grids_differ_only_beyond_their_thread_count():
    G = refstream.local_grid()
    a = torch.rand(35_000, 64, 3, device="cuda", generator=torch.Generator(device="cuda").manual_seed(12345))
    b = refstream.EmulGen(12345, refstream.REFERENCE_GRID).rand(35_000, 64, 3, device="cuda")
    assert float((a == b).float().mean()) < 0.05                      # another GPU, another stream
    n = 256 * G                                                       # n <= 256 * grid: identical on every GPU
    small = torch.rand(n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    assert torch.equal(small, refstream.EmulGen(7, refstream.REFERENCE_GRID).rand(n, device="cuda"))


@pytest.fixture(scope="module")
def val_and_mlp():
    if not TRAINVAL_CACHE.exists():
        pytest.skip(f"validation cache {TRAINVAL_CACHE} not built")
    from sfm import data, legacy
    sets, _ = data.load_sets(("val",))
    return sets["val"], legacy.load_mlp_std()


@cuda
def test_local_stream_equals_common_test_nll_on_validation(val_and_mlp):
    val, mlp = val_and_mlp
    with metric.deterministic():
        ref = C.test_nll(mlp, val)
        emu = refstream.test_nll_ref(mlp, val, grid=refstream.local_grid())
    assert emu.dtype == ref.dtype == np.float32 and emu.shape == ref.shape == (len(val),)
    assert np.array_equal(emu, ref)
    print(f"mlp_std validation, local stream: {ref.astype(np.float64).mean():.6f} (emulated = common.test_nll, bitwise)")


@cuda
def test_reference_stream_reproduces_the_stored_validation_mean(val_and_mlp):
    val, mlp = val_and_mlp
    with metric.deterministic():
        arr = refstream.test_nll_ref(mlp, val, grid=refstream.REFERENCE_GRID)
    mean32 = float(np.float32(arr.mean()))                           # the earlier study stored a float32 mean
    mean64 = float(arr.astype(np.float64).mean())
    print(f"mlp_std validation, reference stream (grid 1056): float32 mean {mean32:.7f}, float64 mean {mean64:.7f}; "
          f"stored 18.118083953857422")
    assert abs(mean32 - 18.118082) <= 4e-6
    assert abs(mean32 - 18.118083953857422) <= 4e-6
    gate = GATE0 / "val_grid1056_mlp_std.npy"                       # Gate 0's array on the same stream
    if gate.exists():
        assert np.abs(arr.astype(np.float64) - np.load(gate)).max() <= 1e-4
