import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, "shape-form-martin/src"); sys.path.insert(0, "shape-form-martin")
import numpy as np, torch
from sfm import paths
SP = "/tmp/claude-1000/-home-martin-projects-uni-PhD-Particle-Physics-Models/baad5034-857c-48b3-bd15-4b0326ffecf3/scratchpad/"
np.set_printoptions(linewidth=220, suppress=True, precision=4)
d = np.load(SP + "e1_train_origin.npz"); O, X0, Y0, z0, th, ph, zb = d["O"], d["X0"], d["Y0"], d["z0"], d["theta"], d["phi"], d["zb"]
f = np.load(SP + "e2_first.npz"); cnt1 = f["cnt1"]
n = len(O)
k = np.round((O[:, 2] - 3662.4) / 20.5).astype(int)
dx, dy = O[:, 0] - X0, O[:, 1] - Y0; dr = np.hypot(dx, dy)
print("k distribution:", np.bincount(np.minimum(k, 6)))
for name, m in (("k=0 & 1 hit in first layer", (k == 0) & (cnt1 == 1)), ("k=0 & 2-3 hits", (k == 0) & (cnt1 >= 2) & (cnt1 <= 3)), ("k=0 & >=10 hits", (k == 0) & (cnt1 >= 10)), ("k=0", k == 0), ("k>=1", k >= 1), ("k>=4", k >= 4)):
    q = np.quantile(dr[m], [0.5, 0.9, 0.99, 0.999])
    print(f"{name:30s} n={m.sum():6d} |d| median {q[0]:7.1f} q90 {q[1]:7.1f} q99 {q[2]:7.1f} q99.9 {q[3]:7.1f}  frac>100 {(dr[m] > 100).mean():.4f}  max|dx| {np.abs(dx[m]).max():.0f}")
m1 = (k == 0) & (cnt1 == 1)
print("single-hit k=0: dx range", dx[m1].min(), dx[m1].max(), "dy range", dy[m1].min(), dy[m1].max(), "frac |dx|<30 & |dy|<30:", ((np.abs(dx[m1]) < 30) & (np.abs(dy[m1]) < 30)).mean())
# radial component of the offset (along phi) vs tangential
er = np.stack([np.cos(ph), np.sin(ph)], 1); et = np.stack([-np.sin(ph), np.cos(ph)], 1)
dpar = dx * er[:, 0] + dy * er[:, 1]; dperp = dx * et[:, 0] + dy * et[:, 1]
print("offset along phi (radial): mean", dpar[k == 0].mean(), "median", np.median(dpar[k == 0]), "rms", dpar[k == 0].std(), "; tangential mean", dperp[k == 0].mean(), "rms", dperp[k == 0].std())
# entry point (straight line at front layer z=3662.4) relative to the detector
Xs, Ys = 3662.4 * np.tan(th) * np.cos(ph), 3662.4 * np.tan(th) * np.sin(ph)
in_hole = (Xs > -406.6) & (Xs < 209.7) & (np.abs(Ys) < 303)
print("straight-line entry in beam hole: frac", in_hole.mean(), " of k>=1:", in_hole[k >= 1].mean(), " of k>=4:", in_hole[k >= 4].mean(), " of k==0:", in_hole[k == 0].mean())
print("k>=1 among in_hole:", (k[in_hole] >= 1).mean(), "; k>=1 among not in hole:", (k[~in_hole] >= 1).mean(), "; k>=4 among not in hole", (k[~in_hole] >= 4).mean())
print("theta: min", th.min(), "max", th.max(), "; entry radius min/max", np.hypot(Xs, Ys).min(), np.hypot(Xs, Ys).max())
# near hole (margin)
for mrg in (0, 50, 100, 200):
    nh = (Xs > -406.6 - mrg) & (Xs < 209.7 + mrg) & (np.abs(Ys) < 303 + mrg)
    print(f" entry within hole+{mrg}mm: frac {nh.mean():.4f}; frac k>=1 inside {(k[nh] >= 1).mean():.3f}; share of all k>=1 events {(nh & (k >= 1)).sum() / (k >= 1).sum():.3f}; share of all k>=4 {(nh & (k >= 4)).sum() / max((k >= 4).sum(), 1):.3f}")
# true origin of k>=4 events
m4 = k >= 4
print("k>=4 events: true origin radius quantiles", np.quantile(np.hypot(O[m4, 0] + 98, O[m4, 1]), [0.1, 0.5, 0.9]), " |O_x+98|<360 & |O_y|<355:", ((np.abs(O[m4, 0] + 98.5) < 360) & (np.abs(O[m4, 1]) < 355)).mean())
# straight-line entry outside outer outline? octagon |x|+|y| ...
oct_out = (np.abs(Xs) > 2624) | (np.abs(Ys) > 2624) | (np.abs(Xs) + np.abs(Ys) > 3710)
print("straight-line entry outside outline (approx octagon):", oct_out.mean(), " among them k>=1:", (k[oct_out] >= 1).mean(), " median |d|:", np.median(dr[oct_out]) if oct_out.any() else None)
big = dr > 500
print("events |d|>500:", big.sum(), big.mean(), " of which k>=1:", (k[big] >= 1).mean(), " in hole(+100):", ((Xs[big] > -506) & (Xs[big] < 310) & (np.abs(Ys[big]) < 403)).mean(), " entry outside outline:", oct_out[big].mean(), " cnt1==1:", (cnt1[big] == 1).mean())
big = (dr > 100) & (dr <= 500)
print("events 100<|d|<=500:", big.sum(), big.mean(), " of which k>=1:", (k[big] >= 1).mean(), " cnt1==1:", (cnt1[big] == 1).mean(), "cnt1>=5:", (cnt1[big] >= 5).mean())
