import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, "shape-form-martin/src"); sys.path.insert(0, "shape-form-martin")
import numpy as np, torch
import pyarrow.parquet as pq
from sfm import paths
t0 = time.time()
obj = torch.load(paths.CACHE_DIR / "pion_s1_trainval.pt", map_location="cpu", weights_only=False)
evn = {s: obj[s]["event_number"].numpy() for s in ("train", "val")}
th_c = {s: obj[s]["theta"].numpy() for s in ("train", "val")}
ph_c = {s: obj[s]["phi"].numpy() for s in ("train", "val")}
zb_c = {s: obj[s]["ev"]["zb"].numpy() for s in ("train", "val")}
nh_c = {s: np.diff(obj[s]["off"].numpy()) for s in ("train", "val")}
del obj
pf = pq.ParquetFile(str(paths.parquet_path()))
def col(name, rows=None):
    a = pf.read(columns=[name]).column(0).to_numpy()
    return a if rows is None else a[rows]
ev = col("event_number")
print("rows", len(ev), "distinct events", len(np.unique(ev)), time.time() - t0)
code = np.zeros(len(ev), np.int8)
for k, s in ((1, "train"), (2, "val")):
    i = np.searchsorted(evn[s], ev); i[i >= len(evn[s])] = 0
    code[evn[s][i] == ev] = k
print("rows train", (code == 1).sum(), "val", (code == 2).sum(), "other", (code == 0).sum())
rows = np.flatnonzero(code == 1)      # TRAIN ONLY
ev = ev[rows]
# event index into train
ei = np.searchsorted(evn["train"], ev)
x, y, z = col("x", rows), col("y", rows), col("z", rows)
xl, yl, zl = col("x_local", rows), col("y_local", rows), col("z_local", rows)
th, ph = col("incoming_theta", rows), col("incoming_phi", rows)
print("theta/phi equal to cache:", np.abs(th - th_c["train"][ei]).max(), np.abs(ph - ph_c["train"][ei]).max())
ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
# global = origin + R^T local ; R rows: e1=(ct cp, ct sp, -st), e2=(-sp, cp, 0), e3=(st cp, st sp, ct)
gx = ct * cp * xl - sp * yl + st * cp * zl
gy = ct * sp * xl + cp * yl + st * sp * zl
gz = -st * xl + ct * zl
ox, oy, oz = x - gx, y - gy, z - gz
n = len(evn["train"])
cnt = np.bincount(ei, minlength=n)
def spread(o):
    mn = np.full(n, np.inf); mx = np.full(n, -np.inf)
    np.minimum.at(mn, ei, o); np.maximum.at(mx, ei, o)
    return mn, mx
res = {}
for name, o in (("x", ox), ("y", oy), ("z", oz)):
    mn, mx = spread(o)
    d = mx - mn
    print(f"origin {name}: per-event spread (max-min over hits): max {d.max():.3e}, 99.9% {np.quantile(d, 0.999):.3e}, median {np.median(d):.3e}")
    res[name] = 0.5 * (mn + mx)
# alternative convention: global = origin + R local (wrong one) -> spread
gx2 = ct * cp * xl + ct * sp * yl - st * zl
gy2 = -sp * xl + cp * yl
gz2 = st * cp * xl + st * sp * yl + ct * zl
for name, o in (("x", x - gx2), ("y", y - gy2), ("z", z - gz2)):
    mn, mx = spread(o)
    print(f"  WRONG convention (R instead of R^T) origin {name} spread: median {np.median(mx - mn):.3e}")
zt = col("z_trans", rows)
print("z - z_trans vs origin z: max abs", np.abs((z - zt) - res["z"][ei]).max())
O = np.stack([res["x"], res["y"], res["z"]], 1)
print("origin z unique (rounded .01):", np.unique(np.round(O[:, 2], 2))[:10], len(np.unique(np.round(O[:, 2], 2))))
# straight-line reconstruction
Z_BACK = 4882.15
zb = zb_c["train"].astype(np.float64); thc, phc = th_c["train"], ph_c["train"]
z0 = Z_BACK - zb * np.cos(thc)
X0, Y0 = z0 * np.tan(thc) * np.cos(phc), z0 * np.tan(thc) * np.sin(phc)
dx, dy, dz = O[:, 0] - X0, O[:, 1] - Y0, O[:, 2] - z0
dr = np.hypot(dx, dy)
print("origin - straight line: dz max abs", np.abs(dz).max())
for q in (0.5, 0.9, 0.99, 0.999, 1.0):
    print(f"  |d_transverse| q{q}: {np.quantile(dr, q):.2f} mm")
print("  mean dx, dy:", dx.mean(), dy.mean(), " rms:", dx.std(), dy.std())
print("  frac |d|>24.3:", (dr > 24.3).mean(), ">48.6:", (dr > 48.6).mean(), ">100:", (dr > 100).mean(), ">500:", (dr > 500).mean())
np.savez("/tmp/claude-1000/-home-martin-projects-uni-PhD-Particle-Physics-Models/baad5034-857c-48b3-bd15-4b0326ffecf3/scratchpad/e1_train_origin.npz", O=O, X0=X0, Y0=Y0, z0=z0, evn=evn["train"], theta=thc, phi=phc, zb=zb, cnt=cnt)
print("done", time.time() - t0)
