"""bodylab - fast tools for the envelope criterion (lane E1_bodies).

Everything is built on envlab (the report's functions).  Additions:
  * fast_quantile / fast_threshold: the same numbers as envlab.shower_quantile / threshold, but with ONE float64 sort
    (the K = 8 points of `envlab.deposits` are already ordered by event), ~5x faster.
  * EventHist: per-event energy histograms on a fine (z', r) grid -> containment as a cumulative sum over cells sorted
    by score (surrogate for exploration).
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import envlab  # noqa: E402
import shapelab  # noqa: E402

LANE = "/home/user/tamas/shape-form-proton/lanes/E1_bodies"
F_E, F_EV = envlab.F_E, envlab.F_EV
S_INF = 60.0            # score used for "never inside" in the fast sort key


def load():
    C, geo, data = shapelab.setup(batch=256, out_dir=os.path.join(LANE, "results"))
    return C, geo, data


def bin_idx(data, split, b):
    return (data[split].ev["bin"] == b).nonzero().squeeze(1)


def bin_P(C, data, split, b, K=8, seed=3, xy=False):
    """Deposits (K fold points per hit) of one momentum bin of one split - exactly envlab.deposits."""
    sh = data[split].subset(bin_idx(data, split, b))
    P = envlab.deposits(C, sh, K=K, seed=seed, xy=xy)
    prep(P)
    return P, sh


def prep(P):
    """Pre-computations for the fast quantile (the points are ordered by event)."""
    ev = P["ev"]
    assert bool((ev[1:] >= ev[:-1]).all())
    cnt = torch.bincount(ev, minlength=P["n"])
    start = torch.cumsum(cnt, 0) - cnt
    P["_cnt"], P["_start"] = cnt, start
    P["_evkey"] = ev.double() * 256.0
    P["_wd"] = P["w"].double()
    P["_logr"] = torch.log(P["r"].clamp_min(1e-6))
    return P


def fast_quantile(P, s, f_e=F_E):
    """Per shower: energy-weighted f_e quantile of the score s (same as envlab.shower_quantile; inf -> S_INF)."""
    sc = torch.nan_to_num(s, nan=S_INF, posinf=S_INF, neginf=-S_INF).clamp(-S_INF, S_INF)
    o = torch.argsort(P["_evkey"] + sc.double())
    sc, w = sc[o], P["_wd"][o]
    c = torch.cumsum(w, 0)
    start, cnt = P["_start"], P["_cnt"]
    last = (start + cnt - 1).clamp_min(0)
    before = c[start.clamp_max(len(c) - 1)] - w[start.clamp_max(len(c) - 1)]
    idx = torch.minimum(torch.searchsorted(c, before + f_e * (c[last] - before)), last)
    return sc[idx]


def fast_threshold(P, s, f_e=F_E, f_ev=F_EV):
    q = fast_quantile(P, s, f_e).cpu().numpy()
    c = float(np.quantile(q, f_ev, method="higher"))
    return c if c < S_INF - 1e-6 else float("inf")


def fast_coverage(P, s, c, f_e=F_E):
    return float((fast_quantile(P, s, f_e) <= c + 1e-9).float().mean())


# ----------------------------------------------------------------------------------------------------------------------
# Surrogate: per-event histograms on a fine (z', r) grid
# ----------------------------------------------------------------------------------------------------------------------
class EventHist:
    """Per-event energy on a (z', r) grid of dz x dr cells.  Containment of {score <= c}: cells sorted by the score of
    their centre, cumulative sum of the energy along that order."""

    def __init__(self, P, dz=10.0, dr=10.0, z_lo=-240.0, z_hi=1490.0, r_hi=740.0):
        dev = P["z"].device
        self.nz, self.nr = int(round((z_hi - z_lo) / dz)), int(round(r_hi / dr))
        iz = ((P["z"] - z_lo) / dz).floor().long().clamp(0, self.nz - 1)
        ir = (P["r"] / dr).floor().long().clamp(0, self.nr - 1)
        n = P["n"]
        H = torch.zeros(n, self.nz * self.nr, device=dev, dtype=torch.float32)
        H.view(-1).index_add_(0, P["ev"] * (self.nz * self.nr) + iz * self.nr + ir, P["w"])
        self.H, self.n = H, n
        self.tot = H.sum(1)
        zc = z_lo + (torch.arange(self.nz, device=dev) + 0.5) * dz
        rc = (torch.arange(self.nr, device=dev) + 0.5) * dr
        self.Z, self.R = [t.reshape(-1) for t in torch.meshgrid(zc, rc, indexing="ij")]
        self.zc, self.rc, self.dz, self.dr = zc, rc, dz, dr
        self.logR = torch.log(self.R)
        self.cellvol = (2 * math.pi * self.R * dr * dz)

    def quantile(self, s, f_e=F_E):
        """Per event: the score level that contains f_e of its energy (s = score of every cell)."""
        s = torch.nan_to_num(s, nan=S_INF, posinf=S_INF, neginf=-S_INF).clamp(-S_INF, S_INF)
        ss, o = torch.sort(s)
        c = torch.cumsum(self.H[:, o], 1)
        idx = (c < (f_e * self.tot)[:, None] - 1e-7).sum(1).clamp_max(len(ss) - 1)
        return ss[idx]

    def threshold(self, s, f_e=F_E, f_ev=F_EV):
        q = self.quantile(s, f_e).cpu().numpy()
        c = float(np.quantile(q, f_ev, method="higher"))
        return c if c < S_INF - 1e-6 else float("inf")

    def body_score(self, body, x):
        lg = body.logg(self.Z, x)
        return torch.where(torch.isfinite(lg), self.logR - lg, torch.full_like(lg, float("inf")))


def objective_factory(body, P=None, hist=None, f_e=F_E, f_ev=F_EV):
    """ln V [mm^3] of the smallest body of the family with shape x that fulfils the criterion (exact points or histogram)."""
    def obj(x):
        if not body.valid(x):
            return 1e6
        if hist is not None:
            c = hist.threshold(hist.body_score(body, x), f_e, f_ev)
        else:
            lg = body.logg(P["z"], x)
            s = torch.where(torch.isfinite(lg), P["_logr"] - lg, torch.full_like(lg, float("inf")))
            c = fast_threshold(P, s, f_e, f_ev)
        if not np.isfinite(c):
            return 1e6
        v = math.log(math.pi) + 2 * c + body.log_i2(x)
        return v if np.isfinite(v) else 1e6
    return obj
