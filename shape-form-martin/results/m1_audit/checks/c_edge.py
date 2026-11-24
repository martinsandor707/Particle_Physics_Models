"""M1-d Part C helper - is there a lateral edge of the instrumented volume? (training + validation, CPU only)

    CUDA_VISIBLE_DEVICES="" uv run python -B shape-form-martin/results/m1_audit/checks/c_edge.py

Hypothesis tested: the particle starts at the global origin and travels along (theta, phi); the shower origin (centroid
of the first hit layer) then sits at global z0 = z_back - zb cos(theta) and transverse radius z0 tan(theta) in the
direction phi. Absolute hit position = origin + R^T x_local. If the calorimeter has a lateral edge, the absolute
transverse coordinates of all hits stop at a fixed value, whatever theta.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402

Z_BACK = 4882.15


def main():
    sets, extras = chk.load_cpu(("train", "val"))
    out = {}
    for split in ("train", "val"):
        sh, ex = sets[split], extras[split]
        ev = chk.hit_event(sh)
        xyz = sh.xyz.numpy().astype(np.float64)
        w = sh.w.numpy().astype(np.float64)
        th, ph = ex["theta"].astype(np.float64), ex["phi"].astype(np.float64)
        zb = sh.ev["zb"].numpy().astype(np.float64)
        R = sh.ev["R"].numpy().astype(np.float64)
        g = np.einsum("hji,hj->hi", R[ev], xyz)
        z0 = Z_BACK - zb * np.cos(th)
        rho0 = z0 * np.tan(th)
        X = rho0[ev] * np.cos(ph[ev]) + g[:, 0]
        Y = rho0[ev] * np.sin(ph[ev]) + g[:, 1]
        Rabs = np.hypot(X, Y)
        res = {"z0_quantiles": {str(q): float(np.quantile(z0, q)) for q in (0, 0.5, 1)},
               "entry_radius_quantiles": {str(q): float(np.quantile(rho0, q)) for q in (0, 0.5, 0.99, 1)},
               "phi_range": [float(ph.min()), float(ph.max())],
               "abs_X_quantiles": {str(q): float(np.quantile(X, q)) for q in (0, 0.0001, 0.001, 0.5, 0.999, 0.9999, 1)},
               "abs_Y_quantiles": {str(q): float(np.quantile(Y, q)) for q in (0, 0.0001, 0.001, 0.5, 0.999, 0.9999, 1)},
               "abs_R_quantiles": {str(q): float(np.quantile(Rabs, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1)}}
        e = np.arange(2000, 3401, 50)
        res["hist_abs_R"] = {"edges": e, "hits": np.histogram(Rabs, e)[0], "energy_sum": np.histogram(Rabs, e, weights=w)[0]}
        e2 = np.arange(1500, 3401, 50)
        res["hist_max_absXY"] = {"edges": e2, "hits": np.histogram(np.maximum(np.abs(X), np.abs(Y)), e2)[0]}
        # per-event: distance of the entry point from candidate edges, and deep energy share
        out[split] = res
        print(split, {k: v for k, v in res.items() if not k.startswith("hist")})
        print(" R hist", list(zip(e[:-1].tolist(), res["hist_abs_R"]["hits"].tolist())))
        print(" max|X|,|Y| hist", list(zip(e2[:-1].tolist(), res["hist_max_absXY"]["hits"].tolist())))
        # cell-centre lattice in absolute coordinates: do X and Y sit on a 48.6 mm lattice?
        for name, c in (("X", X), ("Y", Y)):
            f = (c / 48.6) % 1.0
            hist = np.histogram(f, np.linspace(0, 1, 11))[0]
            print(f"  lattice phase of {name}/48.6:", hist.tolist())
            res[f"lattice_phase_{name}"] = hist
    chk.write("c_edge.json", out)


if __name__ == "__main__":
    main()
