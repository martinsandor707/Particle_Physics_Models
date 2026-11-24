import numpy as np
SP = "/tmp/claude-1000/-home-martin-projects-uni-PhD-Particle-Physics-Models/baad5034-857c-48b3-bd15-4b0326ffecf3/scratchpad/"
np.set_printoptions(linewidth=220, suppress=True, precision=3)
d = np.load(SP + "e3_cols.npz"); ux, uy, c2 = d["ux"], d["uy"], d["c2"]
for yv in (26.2, 127.2, 329.2, 430.2, 1036.2, 1137.2, 2599.8):
    xs = np.sort(ux[np.abs(uy - yv) < 1e-3])
    print(f"\ny={yv}: n={len(xs)}")
    print(xs)
    print("diffs", np.round(np.diff(xs), 3))
occ = np.load(SP + "e3_occ.npy")
print("occ per-cell hits: min", occ.min(), "q1%", np.quantile(occ, 0.01), "median", np.median(occ))
zero = np.argwhere(occ == 0)
for c, l in zero: print("never hit: x", ux[c], "y", uy[c], "layer", l, " column total", occ[c].sum(), "neighbours layer counts", occ[c, max(0, l - 2):l + 1])
