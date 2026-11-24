import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, "shape-form-martin/src"); sys.path.insert(0, "shape-form-martin")
import numpy as np, torch
import pyarrow.parquet as pq
from sfm import paths
SP = "/tmp/claude-1000/-home-martin-projects-uni-PhD-Particle-Physics-Models/baad5034-857c-48b3-bd15-4b0326ffecf3/scratchpad/"
t0 = time.time()
d = np.load(SP + "e1_train_origin.npz")
evn_tr = d["evn"]; O = d["O"]
pf = pq.ParquetFile(str(paths.parquet_path()))
def col(name, rows=None):
    a = pf.read(columns=[name]).column(0).to_numpy()
    return a if rows is None else a[rows]
ev = col("event_number")
i = np.searchsorted(evn_tr, ev); i[i >= len(evn_tr)] = 0
rows = np.flatnonzero(evn_tr[i] == ev)
ei = i[rows]; del ev, i
x, y, z = col("x", rows), col("y", rows), col("z", rows)
e = col("energy", rows); t = col("earliest_contribution_time", rows)
print("train rows", len(rows))
uz = np.unique(np.round(z, 3)); print("z layers:", len(uz), uz[:4], uz[-3:], "steps", np.unique(np.round(np.diff(uz), 3)))
ux = np.unique(np.round(x, 3)); uy = np.unique(np.round(y, 3))
print("x unique:", len(ux), ux[:6], ux[-6:]); print(" x steps:", np.unique(np.round(np.diff(ux), 3), return_counts=True))
print("y unique:", len(uy), uy[:6], uy[-6:]); print(" y steps:", np.unique(np.round(np.diff(uy), 3), return_counts=True))
print("x around 0:", ux[(ux > -200) & (ux < 200)]); print("y around 0:", uy[(uy > -200) & (uy < 200)])
print("energy min/max", e.min(), e.max(), "time min/max", t.min(), t.max(), "frac t<100", (t < 100).mean())
# first-layer centroid check
n = len(evn_tr)
zmin = np.full(n, np.inf); np.minimum.at(zmin, ei, z)
first = z <= zmin[ei] + 1e-6
sw = np.bincount(ei[first], weights=e[first], minlength=n)
cx = np.bincount(ei[first], weights=(e * x)[first], minlength=n) / sw
cy = np.bincount(ei[first], weights=(e * y)[first], minlength=n) / sw
print("ALL rows: origin z == min z:", np.abs(O[:, 2] - zmin).max(), " frac equal:", (np.abs(O[:, 2] - zmin) < 1e-6).mean())
print("ALL rows: centroid err max", np.abs(cx - O[:, 0]).max(), np.abs(cy - O[:, 1]).max(), "frac <1e-6:", ((np.abs(cx - O[:, 0]) < 1e-6) & (np.abs(cy - O[:, 1]) < 1e-6)).mean())
m = t < 100
zmin2 = np.full(n, np.inf); np.minimum.at(zmin2, ei[m], z[m])
first2 = m & (z <= zmin2[ei] + 1e-6)
sw2 = np.bincount(ei[first2], weights=e[first2], minlength=n)
cx2 = np.bincount(ei[first2], weights=(e * x)[first2], minlength=n) / sw2
cy2 = np.bincount(ei[first2], weights=(e * y)[first2], minlength=n) / sw2
print("t<100 rows: origin z == min z frac:", (np.abs(O[:, 2] - zmin2) < 1e-6).mean(), " centroid frac<1e-6:", ((np.abs(cx2 - O[:, 0]) < 1e-6) & (np.abs(cy2 - O[:, 1]) < 1e-6)).mean())
# unweighted centroid?
cnt1 = np.bincount(ei[first], minlength=n)
ux_ = np.bincount(ei[first], weights=x[first], minlength=n) / cnt1; uy_ = np.bincount(ei[first], weights=y[first], minlength=n) / cnt1
print("unweighted centroid frac<1e-6:", ((np.abs(ux_ - O[:, 0]) < 1e-6) & (np.abs(uy_ - O[:, 1]) < 1e-6)).mean())
print("hits in first layer: dist", np.bincount(np.minimum(cnt1, 10)))
np.savez(SP + "e2_first.npz", cnt1=cnt1, zmin=zmin)
# global extent
print("x range", x.min(), x.max(), "y range", y.min(), y.max())
for lim in (2400, 2500, 2550, 2600, 2624.4, 2650, 2700, 2800, 3000, 3500):
    mk = (np.abs(x) > lim) | (np.abs(y) > lim)
    print(f" max(|x|,|y|) > {lim}: hits {mk.sum()} frac {mk.mean():.2e} energy frac {e[mk].sum() / e.sum():.2e}")
print("done", time.time() - t0)
