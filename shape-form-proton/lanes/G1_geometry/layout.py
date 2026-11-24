import numpy as np, torch
G = torch.load("/home/user/tamas/shape-form-proton/cache/proton_geometry.pt", weights_only=False)
cx = G["cells_xy"].numpy().astype(np.float64)
print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in G.items() if k != "sets"}, {k: {a: b.shape for a, b in v.items()} for k, v in G["sets"].items()})
uy = np.unique(np.round(cx[:, 1], 1))
print("unique y:", len(uy), uy[:8], "...", uy[-4:])
for y in (uy[0], uy[1], uy[2], uy[3], uy[50], uy[51], uy[52], uy[53], uy[103]):
    xs = np.sort(cx[np.abs(cx[:, 1] - y) < 0.06, 0])
    d = np.round(np.diff(xs), 2)
    vals, cnt = np.unique(d, return_counts=True)
    print(f"y={y:8.1f} n={len(xs):3d} x[{xs[0]:8.1f}..{xs[-1]:8.1f}] first x: {np.round(xs[:10],1)} pitch hist: {dict(zip(vals.tolist(), cnt.tolist()))}")
ux = np.unique(np.round(cx[:, 0], 1))
# column structure: for a few x values list y diffs
for x in (ux[0], ux[100], ux[200]):
    ys = np.sort(cx[np.abs(cx[:, 0] - x) < 0.06, 1]); d = np.round(np.diff(ys), 2); vals, cnt = np.unique(d, return_counts=True)
    print(f"x={x:8.1f} n={len(ys)} pitch hist {dict(zip(vals.tolist(), cnt.tolist()))}")
# nearest-neighbour distance distribution
from scipy.spatial import cKDTree
t = cKDTree(cx); d, i = t.query(cx, k=5)
for k in range(1, 5):
    vals, cnt = np.unique(np.round(d[:, k], 1), return_counts=True)
    print("NN", k, dict(zip(vals.tolist()[:12], cnt.tolist()[:12])))
print("area per tower if 48.6^2:", 48.6**2, " total", len(cx) * 48.6**2 / 1e6, "m^2")
