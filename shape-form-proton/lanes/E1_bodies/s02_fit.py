"""Per-bin fits of the body families (exact K = 8 points).  usage: s02_fit.py --bins 0,4,8 --fam drop,gendrop [--fe 0.95 --fev 0.98]
Knot families (name starts with "knots") are started from the best already fitted closed-form bodies of the bin."""
import argparse
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/E1_bodies")
import bodylab as bl
import fitlib as fl
import fam

ap = argparse.ArgumentParser()
ap.add_argument("--bins", default="all")
ap.add_argument("--fam", default="drop,envelope")
ap.add_argument("--fe", type=float, default=0.95)
ap.add_argument("--fev", type=float, default=0.98)
ap.add_argument("--redo", action="store_true")
ap.add_argument("--improve", action="store_true", help="knot families: surrogate + exact optimisation from all fitted bodies; keep only if better")
ap.add_argument("--sur", action="store_true", help="optimise on the histogram surrogate, then polish / evaluate with the exact K = 8 points")
a = ap.parse_args()
bins = list(range(20)) if a.bins == "all" else [int(v) for v in a.bins.split(",")]
fams = a.fam.split(",")
REG = fam.registry()
tag = fl.crit_tag(a.fe, a.fev)
L05, L15, L2 = math.log(0.5), math.log(1.5), math.log(2.0)
SEEDS = {
    "gendrop": [("drop", lambda x: x + [L05, L15])],
    "drop_trunc": [("drop", lambda x: x + [1500.0])],
    "gendrop_trunc": [("gendrop", lambda x: x + [1500.0]), ("drop_trunc", lambda x: x[:2] + [L05, L15] + x[2:])],
    "envelope_trunc": [("envelope", lambda x: x + [1500.0])],
    "superegg": [("egg", lambda x: x + [L2, L2])],
    "superegg_sep": [("superegg", lambda x: x[:3] + [x[3], x[3], x[4], x[4]])],
    "superellipsoid": [("superellipsoid_n", lambda x: x + [L2])],
}
C, geo, data = bl.load()
for b in bins:
    todo = [f for f in fams if a.redo or fl.load_fit(tag, f, b) is None or (a.improve and f.startswith("knots") and not (fl.load_fit(tag, f, b) or {}).get("improved"))]
    if not todo:
        continue
    sets = {k: bl.bin_P(C, data, k, b)[0] for k in ("train", "val", "test")}
    P = sets["train"]
    p_mean = float(torch.exp(data["train"].ev["lnp"][bl.bin_idx(data, "train", b)]).mean())
    for f in todo:
        body = REG[f][0]
        obj = bl.objective_factory(body, P=P, f_e=a.fe, f_ev=a.fev)
        t0 = time.time()
        extra, knots = [], f.startswith("knots")
        if knots:
            cand = []
            for sname in REG:
                if sname == f:
                    continue
                prev = fl.load_fit(tag, sname, b)
                if prev is not None:
                    xs = body.from_body(REG[sname][0], prev["x"])
                    cand.append((obj(xs), sname, xs))
            cand.sort(key=lambda t: t[0])
            extra = [t[2] for t in cand[:2]]
        else:
            for sname, conv in SEEDS.get(f, []):
                prev = fl.load_fit(tag, sname, b)
                if prev is not None:
                    xs = conv(list(prev["x"]))
                    if body.valid(xs):
                        extra.append(xs)
        if (knots and a.improve) or a.sur:
            if "H" not in sets:
                sets["H"] = bl.EventHist(P)
            obj_h = bl.objective_factory(body, hist=sets["H"], f_e=a.fe, f_ev=a.fev)
        if knots and (a.improve or a.sur):
            cands = [t[2] for t in cand]
            old = fl.load_fit(tag, f, b)
            if old is not None:
                cands.append(np.asarray(old["x"], float))
            if not cands:
                cands = [np.asarray(g, float) for g in body.x0_grid]
            x, lnv, nfev = fl.fit_knots2(body, obj_h, obj, cands, light=a.sur and not a.improve)
        elif a.sur:
            x, lnv_h, nfev = fl.fit_body(body, obj_h, x_extra=extra)
            x, lnv, k = fl.nm(obj, x, fl.steps_of(body) * 0.15, 40 * len(x))
            nfev += k
        else:
            x, lnv, nfev = fl.fit_body(body, obj, x_extra=extra, knots=knots)
        res = fl.metrics(C, body, x, {k: v for k, v in sets.items() if k != "H"}, a.fe, a.fev)
        res["improved"] = bool(knots and (a.improve or a.sur))
        res["surrogate"] = bool(a.sur)
        res.update(family=f, bin=b, p_mean=p_mean, f_e=a.fe, f_ev=a.fev, nfev=nfev, n_par=len(body.names) + 1, t_fit=time.time() - t0)
        fl.save_fit(tag, f, b, res)
        print(f"{tag} bin {b:2d} {f:18s} V {res['V_L']:8.1f} L  V_in {res['V_in_L']:8.1f}  rmax {res['r_max']:6.1f}  z {res['z_first']:.0f}..{res['z_last']:.0f}  "
              f"cov tr {res['cov_train']:.4f} va {res['cov_val']:.4f} te {res['cov_test']:.4f}  nfev {nfev}  {time.time() - t0:.0f}s", flush=True)
