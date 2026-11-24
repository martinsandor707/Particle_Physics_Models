"""Gate 0 helper: the CPU emulation of the CUDA random stream reproduces the local GPU bit for bit.
No data needed. Prints the local launch grid and the comparison for several tensor sizes and call sequences."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from emul import EmulGen, this_gpu_grid_candidates

sm, cands = this_gpu_grid_candidates()
print("GPU:", torch.cuda.get_device_name(0), "| SM count:", sm, "| torch", torch.__version__)
shapes_seq = [(1000,), (36_863,), (36_864,), (36_865,), (147_456,), (147_457,), (35_000, 64, 3), (7, 16, 3), (41_237, 64, 3), (5,)]
found = None
for tps, G in cands.items():
    g = torch.Generator(device="cuda").manual_seed(12345)
    e = EmulGen(12345, G)
    ok = []
    for shp in shapes_seq:                           # one generator, consecutive calls: also checks the offset advance
        a = torch.rand(*shp, device="cuda", generator=g)
        b = e.rand(*shp, device="cuda")
        ok.append(bool(torch.equal(a, b)))
    print(f"threads/SM {tps}: grid {G:4d} -> bit-identical per call: {ok}")
    if all(ok):
        found = G
print("local launch grid:", found)
assert found is not None
# independent seeds / fresh generators
for seed in (0, 1, 12345, 2**40 + 7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.rand(300_001, 3, device="cuda", generator=g)
    b = EmulGen(seed, found).rand(300_001, 3)
    print(f"seed {seed}: equal {bool(torch.equal(a, b))}")
# how different is the stream for another grid (e.g. an 82-SM GPU, grid 492)?
g = torch.Generator(device="cuda").manual_seed(12345)
a = torch.rand(35_000, 64, 3, device="cuda", generator=g)
b = EmulGen(12345, 492).rand(35_000, 64, 3)
print("grid 492 vs local on a (35000,64,3) tensor: fraction of equal elements", float((a == b).float().mean()),
      "| first 2 elements equal:", bool(a.flatten()[0] == b.flatten()[0]), bool(a.flatten()[1] == b.flatten()[1]))
small = torch.rand(20_000, device="cuda", generator=torch.Generator(device="cuda").manual_seed(12345))
print("small tensor (20000 <= 256*grid for both): equal", bool(torch.equal(small, EmulGen(12345, 492).rand(20_000))))
