"""Geometry audit, step 4b - Monte Carlo cross-check of the acceptance maps on TRAINING events (CPU only).

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g4b_montecarlo.py

For 48 events of the g4 training subsample (24 whose axis leaves through a side face or the hole, 24 others) every
grid cell gets 256 random points (uniform in z', in r^2 and in azimuth); a point counts as accepted if
``sfm.geometry.columns_inside`` - the test built directly on the list of training columns, independent of the
outline constants - and the front / back planes accept it. Compared with the quadrature at (nsub, nphi) = (2, 64)
[default] and (4, 256).
"""
import json
import time

import glib as L
from glib import G, common, np, torch

t0 = time.time()
sets, extras, origin = L.load_cpu(("train",))
tr, ex, O = sets["train"], extras["train"], origin["train"]
rng = np.random.default_rng(20261125)
sub_idx = np.sort(rng.choice(len(tr), 4096, replace=False))
R, Os = tr.ev["R"][sub_idx], O[sub_idx]
th, ph, zb = ex["theta"][sub_idx], ex["phi"][sub_idx], tr.ev["zb"].numpy()[sub_idx].astype(np.float64)
leak = G.axis_exit(R.numpy(), Os, G.Volume()) < np.minimum(zb, 1450.0)
nm, Mpts = 48, 256
pick = np.concatenate([np.flatnonzero(leak)[-nm // 2:], np.flatnonzero(~leak)[-nm // 2:]])
cols = np.load(L.HERE / "explore" / "train_columns.npz")
V = {"FS": ("front", "side"), "FSB": ("front", "side", "back")}
quad = {"2x64": G.event_acceptance(R[pick], torch.as_tensor(Os[pick]), G.Volume(), V, nsub=2, nphi=64, chunk=4),
        "4x256": G.event_acceptance(R[pick], torch.as_tensor(Os[pick]), G.Volume(), V, nsub=4, nphi=256, chunk=1)}
ze, re = common.Z_EDGES.cpu().numpy().astype(np.float64), common.R_EDGES.cpu().numpy().astype(np.float64)
mrng = np.random.default_rng(7)
mc = {k: [] for k in V}
for e in pick:
    z = ze[:-1, None, None] + common.DZ * mrng.random((common.N_Z, common.N_R, Mpts))
    r = np.sqrt(re[None, :-1, None] ** 2 + (re[None, 1:, None] ** 2 - re[None, :-1, None] ** 2) * mrng.random((common.N_Z, common.N_R, Mpts)))
    a = mrng.uniform(0, 2 * np.pi, (common.N_Z, common.N_R, Mpts))
    off = G.to_global(np.stack([r * np.cos(a), r * np.sin(a), z], -1), float(th[e]), float(ph[e]))
    g = off + Os[e]
    side = G.columns_inside(g[..., 0], g[..., 1], cols["ux"], cols["uy"])
    front, back = off[..., 2] >= -G.LAYER / 2, g[..., 2] <= G.Z_BACK
    mc["FS"].append((side & front).mean(2))
    mc["FSB"].append((side & front & back).mean(2))
out = {"split": "train", "events": int(len(pick)), "points_per_cell": Mpts, "cells": int(len(pick) * common.N_Z * common.N_R),
       "point_test": "sfm.geometry.columns_inside on explore/train_columns.npz"}
for k in V:
    m = np.stack(mc[k])
    sig2 = m * (1 - m) / Mpts
    for q, A in quad.items():
        d = A[k].numpy().astype(np.float64) - m
        edge = (m > 0) & (m < 1)
        out[f"{k} quadrature {q} minus MC"] = {
            "mean_signed": float(d.mean()), "se_of_mean": float(d.std(ddof=1) / np.sqrt(d.size)),
            "rms": float(np.sqrt((d ** 2).mean())), "rms_expected_from_binomial_noise_alone": float(np.sqrt(sig2.mean())),
            "max_abs": float(np.abs(d).max()), "cells_with_0<MC<1": int(edge.sum()),
            "cells_MC_exactly_0_or_1": int((~edge).sum()), "max_abs_in_cells_MC_0_or_1": float(np.abs(d[~edge]).max()),
            "event_mean_acceptance_max_abs_difference": float(np.abs(d.mean((1, 2))).max())}
        print(k, q, json.dumps(out[f"{k} quadrature {q} minus MC"]), flush=True)
out["seconds"] = time.time() - t0
(L.HERE / "g4b_montecarlo.json").write_text(json.dumps(out, indent=1) + "\n")
print("done", out["seconds"])
