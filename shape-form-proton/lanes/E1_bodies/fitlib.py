"""fitlib - per-bin fitting of bodies of revolution with the report's criterion (exact K = 8 points, fast sort).
Used by s02_fit.py.  Results: results/fits/<crit>/<family>/bin<bb>.json"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/E1_bodies")
import bodylab as bl  # noqa: E402
import envlab  # noqa: E402
import fam  # noqa: E402

LANE = bl.LANE
Z_LO, Z_HI, R_CUT = -200.0, 1450.0, 700.0


def steps_of(body):
    if hasattr(body, "steps"):
        return np.asarray(body.steps, float)
    return np.asarray([0.3 if (hi - lo) < 20 else 20.0 for lo, hi in body.bounds], float)


def nm(obj, x0, steps, maxiter):
    n = len(x0)
    sim = np.tile(np.asarray(x0, float), (n + 1, 1))
    for i in range(n):
        sim[i + 1, i] += steps[i]
    res = minimize(obj, x0, method="Nelder-Mead", options=dict(initial_simplex=sim, maxiter=maxiter, maxfev=maxiter, xatol=1e-4, fatol=1e-7))
    return res.x, float(res.fun), int(res.nfev)


def pattern_search(obj, x0, step0=0.4, step_min=0.01, max_eval=2500, seed=0):
    """Coordinate pattern search (robust for the piecewise-smooth objective, used for the knot profiles)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float).copy()
    f, nfev, step = obj(x), 1, step0
    while step >= step_min and nfev < max_eval:
        improved = False
        for i in rng.permutation(len(x)):
            for d in (step, -step):
                moved = False
                while nfev < max_eval:
                    y = x.copy()
                    y[i] += d
                    fy = obj(y)
                    nfev += 1
                    if fy < f - 1e-9:
                        x, f, moved, improved = y, fy, True, True
                    else:
                        break
                if moved:
                    break
        if not improved:
            step *= 0.5
    return x, f, nfev


def fit_body(body, obj, n_start=3, x_extra=(), knots=False):
    """Grid of starting points -> Nelder-Mead from the best n_start -> restarts.  Returns x, ln V[mm^3], number of evaluations."""
    steps = steps_of(body)
    dim = len(steps)
    if knots:
        best, nfev = (float("inf"), None), 0
        for x0 in list(x_extra) + [np.asarray(g, float) for g in body.x0_grid]:
            x, f, k = pattern_search(obj, x0)
            nfev += k
            if f < best[0]:
                best = (f, x)
        return best[1], best[0], nfev
    grid = [np.asarray(g, float) for g in body.x0_grid] + [np.asarray(g, float) for g in x_extra]
    vals = np.array([obj(x) for x in grid])
    nfev, best = len(grid), (float("inf"), None)
    for i in np.argsort(vals)[:n_start]:
        if vals[i] >= 1e6:
            continue
        x, f, k = nm(obj, grid[i], steps, 150 * dim)
        nfev += k
        if f < best[0]:
            best = (f, x)
    for _ in range(4):
        x, f, k = nm(obj, best[1], steps * 0.3, 200 * dim)
        nfev += k
        if f < best[0] - 1e-6:
            best = (f, x)
        else:
            if f < best[0]:
                best = (f, x)
            break
    return best[1], best[0], nfev


def score_of(body, P, x):
    lg = body.logg(P["z"], x)
    return torch.where(torch.isfinite(lg), P["_logr"] - lg, torch.full_like(lg, float("inf")))


def metrics(C, body, x, sets, f_e, f_ev):
    """sets = {"train": P, "val": P, "test": P}.  Size A from the training events (the report's recipe)."""
    s = {k: score_of(body, P, x) for k, P in sets.items()}
    c = bl.fast_threshold(sets["train"], s["train"], f_e, f_ev)
    A = math.exp(c)
    rf = lambda z: body.radius(z, x, A)
    zz = np.linspace(-260, 1500, 3521)
    rr = rf(zz)
    out = dict(body.decode(x))
    out = {k: (float(v) if np.isscalar(v) else v) for k, v in out.items()}
    out.update(x=[float(v) for v in x], A=A, lnA=c,
               V_L=math.exp(math.log(math.pi) + 2 * c + body.log_i2(x)) / 1e6,
               V_in_L=envlab.grid_volume_L(C, rf, clip_r=R_CUT, z_lo=Z_LO, z_hi=Z_HI),
               V_grid_L=envlab.grid_volume_L(C, rf, z_lo=Z_LO, z_hi=Z_HI),
               r_max=float(rr.max()), z_rmax=float(zz[int(rr.argmax())]),
               z_first=float(zz[rr > 1.0][0]) if (rr > 1.0).any() else float("nan"),
               z_last=float(zz[rr > 1.0][-1]) if (rr > 1.0).any() else float("nan"))
    for k, P in sets.items():
        q = bl.fast_quantile(P, s[k], f_e)
        out[f"cov_{k}"] = float((q <= c + 1e-9).float().mean())
        out[f"n_{k}"] = int(P["n"])
        if k != "train":
            # size needed for exactly f_ev coverage on this independent set (shape from train): calibrated volume
            ck = float(np.quantile(q.cpu().numpy(), f_ev, method="higher"))
            out[f"V_cal_{k}_L"] = math.exp(math.log(math.pi) + 2 * ck + body.log_i2(x)) / 1e6 if ck < bl.S_INF - 1e-6 else float("inf")
    return out


def crit_tag(f_e, f_ev):
    return f"E{int(round(f_e * 100)):02d}_EV{int(round(f_ev * 100)):02d}"


def out_path(tag, family, b):
    return os.path.join(LANE, "results", "fits", tag, family, f"bin{b:02d}.json")


def load_fit(tag, family, b):
    p = out_path(tag, family, b)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def save_fit(tag, family, b, res):
    p = out_path(tag, family, b)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(p + ".tmp", p)


def fit_knots2(body, obj_h, obj, cands, log=None, light=False):
    """Improved optimisation of a knot profile: the bulk of the search on the histogram surrogate `obj_h` (Nelder-Mead +
    pattern search from the two best candidates), then a polish with the exact objective `obj`.  Returns x, ln V (exact), nfev."""
    steps = steps_of(body)
    dim = len(steps)
    cands = [np.asarray(c, float) for c in cands]
    vals = np.array([obj(x) for x in cands])
    order = np.argsort(vals)
    best, nfev, sols = (float(vals[order[0]]), cands[order[0]]), len(cands), []
    for i in order[:2]:
        x, f, k = nm(obj_h, cands[i], steps, 250 * dim)
        x, f, k2 = pattern_search(obj_h, x, step0=0.2, step_min=0.01, max_eval=1500)
        nfev += k + k2
        sols.append(x)
    for x in sols:
        x, f, k = pattern_search(obj, x, step0=0.1, step_min=0.005 if not light else 0.02, max_eval=400 if not light else 100)
        nfev += k
        if f < best[0]:
            best = (f, x)
    if not light:
        x, f, k = nm(obj, best[1], steps * 0.2, 40 * dim)
        nfev += k
        if f < best[0]:
            best = (f, x)
    return best[1], best[0], nfev
