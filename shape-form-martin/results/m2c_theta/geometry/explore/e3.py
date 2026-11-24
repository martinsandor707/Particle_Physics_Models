import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, "shape-form-martin/src"); sys.path.insert(0, "shape-form-martin")
import numpy as np, torch
import pyarrow.parquet as pq
from sfm import paths
SP = "/tmp/claude-1000/-home-martin-projects-uni-PhD-Particle-Physics-Models/baad5034-857c-48b3-bd15-4b0326ffecf3/scratchpad/"
np.set_printoptions(linewidth=250, suppress=True)
t0 = time.time()
d = np.load(SP + "e1_train_origin.npz")
evn_tr = d["evn"]
pf = pq.ParquetFile(str(paths.parquet_path()))
def col(name, rows=None):
    a = pf.read(columns=[name]).column(0).to_numpy()
    return a if rows is None else a[rows]
ev = col("event_number")
i = np.searchsorted(evn_tr, ev); i[i >= len(evn_tr)] = 0
rows = np.flatnonzero(evn_tr[i] == ev)
ei = i[rows]; del ev, i
x, y, z = col("x", rows), col("y", rows), col("z", rows)
# integer keys
xi = np.round(x * 1000).astype(np.int64); yi = np.round(y * 1000).astype(np.int64); zi = np.round((z - 3662.4) / 20.5).astype(np.int64)
print("layer idx range", zi.min(), zi.max(), "max dev from lattice", np.abs(z - 3662.4 - 20.5 * zi).max())
key2 = (xi + 5_000_000) * 10_000_000 + (yi + 5_000_000)
u2, inv2, c2 = np.unique(key2, return_inverse=True, return_counts=True)
ux = (u2 // 10_000_000 - 5_000_000) / 1000.0; uy = (u2 % 10_000_000 - 5_000_000) / 1000.0
print("distinct (x,y) columns:", len(u2), " min count", c2.min(), "median", np.median(c2))
# per-row structure
ys = np.unique(uy)
print("rows (y):", len(ys))
for yv in ys:
    xs = np.sort(ux[uy == yv])
    dxs = np.round(np.diff(xs), 3)
    vals, cnts = np.unique(dxs, return_counts=True)
    print(f"y={yv:9.3f} n={len(xs):3d} x[{xs[0]:9.3f}..{xs[-1]:9.3f}] steps {dict(zip(vals.tolist(), cnts.tolist()))}")
np.savez(SP + "e3_cols.npz", ux=ux, uy=uy, c2=c2)
# 3D distinct cells
key3 = inv2.astype(np.int64) * 64 + zi
u3, c3 = np.unique(key3, return_counts=True)
print("distinct 3D cells:", len(u3), "of", len(u2) * 60, "possible (columns x 60)")
occ = np.zeros((len(u2), 60), np.int64); occ[u3 // 64, u3 % 64] = c3
np.save(SP + "e3_occ.npy", occ)
print("cells never hit per layer:", (occ == 0).sum(0))
print("done", time.time() - t0)
