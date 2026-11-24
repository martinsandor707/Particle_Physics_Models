"""envlab – burok-típusú 3D testek általános illesztése a report kritériumával.

Kritérium (02_envelope, 03_drop): a legkisebb térfogatú test, amelyen belül a záporok F_EV = 98 %-ában
a látható energia legalább F_E = 95 %-a van. A lerakódások a cellán belül szétosztott pontok (K = 8, seed = 3),
pontosan úgy, mint a 02/03 notebookokban.

Általánosítás: egy testcsalád egymásba ágyazott testekből áll, {pont : s(pont) <= c}, ahol s a pont „pontszáma"
(pl. forgástestnél s = ln r − ln g(z'), a szint c = ln A). Adott alakparaméterekre a legkisebb megfelelő c EGZAKTUL számolható:
záporonként az s energiával súlyozott F_E-kvantilise, majd ezek F_EV-percentilise a záporok fölött. Így csak az
alakparamétereket kell keresni (Nelder–Mead), a méretet nem.
"""
import math

import numpy as np
import torch
from scipy.optimize import minimize

F_E, F_EV = 0.95, 0.98


def deposits(C, sh, K=8, seed=3, xy=False):
    """Minden találat -> K pont a (forgatott) cellájában, w/K súllyal (a 02/03 notebook `deposits` függvénye).
    xy=True: az x', y' koordinátákat és az eseményenkénti cos/sin(theta), D (= hátlap mélysége) értékeket is megtartja."""
    dev = C["DEVICE"]
    gen = torch.Generator(device=dev).manual_seed(seed)
    b = sh.batch(torch.arange(len(sh), device=dev))
    pts = C["fold"](b, sh.box, K, gen)
    P = {"z": pts[..., 2].reshape(-1), "r": torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2).reshape(-1),
         "w": (b.w[:, None] / K).expand(-1, K).reshape(-1), "ev": b.ev[:, None].expand(-1, K).reshape(-1), "n": len(sh)}
    if xy:
        cos_t = b.R[:, 2, 2]
        P.update(x=pts[..., 0].reshape(-1), y=pts[..., 1].reshape(-1), cos_t=cos_t,
                 sin_t=torch.sqrt((1 - cos_t ** 2).clamp_min(0)), D=b.zb * cos_t, half=float(sh.box[2]) / 2)
    return P


def shower_quantile(P, s, f_e=F_E):
    """Záporonként: az s pontszám energiával súlyozott f_e-kvantilise (a 03 notebook függvénye)."""
    o = torch.argsort(s, stable=True)
    o = o[torch.argsort(P["ev"][o], stable=True)]                        # zápor, azon belül s szerint rendezve
    s, w, e = s[o], P["w"][o].double(), P["ev"][o]
    c = torch.cumsum(w, 0)
    cnt = torch.bincount(e, minlength=P["n"])
    start = torch.cumsum(cnt, 0) - cnt
    before = c[start] - w[start]
    idx = torch.minimum(torch.searchsorted(c, before + f_e * (c[start + cnt - 1] - before)), start + cnt - 1)
    return s[idx]


def threshold(P, s, f_e=F_E, f_ev=F_EV):
    """A legkisebb c szint, amellyel a záporok f_ev hányadában az energia >= f_e része a {s <= c} testben van."""
    return float(np.quantile(shower_quantile(P, s, f_e).cpu().numpy(), f_ev, method="higher"))


def coverage(P, s, c, f_e=F_E):
    """Azon záporok hányada, amelyek energiájának >= f_e része a {s <= c} testben van."""
    return float((shower_quantile(P, s, f_e) <= c + 1e-9).float().mean())


# ----------------------------------------------------------------------------------------------------------------------
# Forgástestek: r(z') = A g(z'; alak).  Pontszám: s = ln r − ln g(z'), szint: c = ln A.
# ----------------------------------------------------------------------------------------------------------------------
class Body:
    """Forgástest-család.  `logg(z, x)`: ln g a z pontokban (torch; -inf ahol a test nem létezik);
    `log_i2(x)`: ln ∫ g(z')^2 dz' (a teljes test, analitikusan) -> ln V = ln(pi) + 2 ln A + log_i2."""
    names, x0_grid, bounds = [], [], []

    def logg(self, z, x):
        raise NotImplementedError

    def log_i2(self, x):
        raise NotImplementedError

    def decode(self, x):
        """Keresési változók -> fizikai alakparaméterek (dict)."""
        raise NotImplementedError

    def score(self, P, x):
        lg = self.logg(P["z"], x)
        return torch.where(torch.isfinite(lg), torch.log(P["r"].clamp_min(1e-6)) - lg, torch.full_like(lg, float("inf")))

    def radius(self, z, x, A):
        """r(z') numpy-ban, ábrához és rácsos térfogathoz."""
        zt = torch.as_tensor(np.asarray(z, dtype=np.float32))
        lg = self.logg(zt, x).double().numpy()
        return np.where(np.isfinite(lg), A * np.exp(lg), 0.0)

    def valid(self, x):
        return all(lo < v < hi for v, (lo, hi) in zip(x, self.bounds))

    def fit_bin(self, P, f_e=F_E, f_ev=F_EV, maxiter=400):
        def objective(x):
            if not self.valid(x):
                return 1e6
            c = threshold(P, self.score(P, x), f_e, f_ev)
            return math.log(math.pi) + 2 * c + self.log_i2(x) if np.isfinite(c) else 1e6
        x0 = min(self.x0_grid, key=objective)
        x = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5}).x
        c = threshold(P, self.score(P, x), f_e, f_ev)
        out = dict(self.decode(x))
        out.update(x=[float(v) for v in x], A=math.exp(c), V_L=math.exp(math.log(math.pi) + 2 * c + self.log_i2(x)) / 1e6)
        return out


class Envelope(Body):
    """02: r = A u^a e^{-bu}, u = z' - z0 (keresés: ln a, ln z_env = ln(a/b), z0)."""
    names = ["a", "z_env", "z0"]
    bounds = [(math.log(0.05), math.log(10)), (math.log(3), math.log(1500)), (-200, 50)]
    x0_grid = [(la, lz, z0) for la in np.linspace(math.log(0.1), math.log(6), 12)
               for lz in np.linspace(math.log(5), math.log(1200), 12) for z0 in (-60, -30, -10)]

    def logg(self, z, x):
        a, zenv, z0 = math.exp(x[0]), math.exp(x[1]), x[2]
        u = z - z0
        return torch.where(u > 0, a * torch.log(u.clamp_min(1e-12)) - (a / zenv) * u, torch.full_like(u, -float("inf")))

    def log_i2(self, x):
        a, zenv = math.exp(x[0]), math.exp(x[1])
        return math.lgamma(2 * a + 1) - (2 * a + 1) * math.log(2 * a / zenv)

    def decode(self, x):
        a, zenv = math.exp(x[0]), math.exp(x[1])
        return {"a": a, "b": a / zenv, "z_env": zenv, "z0": x[2]}


class BetaDrop(Body):
    """Általánosított csepp: r = A g(t), g = t^alpha (1-t)^beta / g_max, t = (z' - z0)/L.
    alpha = 1/2, beta = 3/2: a 03 notebook cseppje; alpha = beta = 1/2: ellipszoid; alpha = 0, beta = 1: kúp; alpha = beta -> 0: henger.
    `fixed`: rögzített (alpha, beta) – ekkor csak (ln L, z0) a keresési változó."""

    def __init__(self, fixed=None, z0_range=(-200.0, 50.0), L_range=(20.0, 8000.0)):
        self.fixed = fixed
        self.names = ["L", "z0"] + ([] if fixed else ["alpha", "beta"])
        self.bounds = [(math.log(L_range[0]), math.log(L_range[1])), z0_range] + ([] if fixed else [(math.log(0.02), math.log(4)), (math.log(0.02), math.log(6))])
        grid = [(lL, z0) for lL in np.linspace(math.log(80), math.log(6000), 20) for z0 in (-120, -80, -50, -30, -10, 10)]
        self.x0_grid = grid if fixed else [g + (math.log(a), math.log(b)) for g in grid[::3] for a, b in ((0.5, 1.5), (0.5, 0.5), (0.25, 1.0), (1.0, 2.0), (0.15, 0.4))]

    def _ab(self, x):
        return self.fixed if self.fixed else (math.exp(x[2]), math.exp(x[3]))

    def logg(self, z, x):
        a, b = self._ab(x)
        t = (z - x[1]) / math.exp(x[0])
        ok = (t > 0) & (t < 1)
        tc = t.clamp(1e-12, 1 - 1e-7)
        lgmax = a * math.log(a / (a + b)) + b * math.log(b / (a + b))
        return torch.where(ok, a * torch.log(tc) + b * torch.log1p(-tc) - lgmax, torch.full_like(t, -float("inf")))

    def log_i2(self, x):
        a, b = self._ab(x)
        lgmax = a * math.log(a / (a + b)) + b * math.log(b / (a + b))
        return x[0] + math.lgamma(2 * a + 1) + math.lgamma(2 * b + 1) - math.lgamma(2 * a + 2 * b + 2) - 2 * lgmax

    def decode(self, x):
        a, b = self._ab(x)
        L = math.exp(x[0])
        return {"L": L, "z0": x[1], "alpha": a, "beta": b, "z_env": x[1] + L * a / (a + b)}


def grid_volume_L(C, radius_fn, clip_r=None, z_lo=None, z_hi=None, n=6601):
    """∫ pi r(z')^2 dz' [liter] a [z_lo, z_hi] mélységtartományon (alap: a rács), opcionálisan r <= clip_r-re vágva."""
    z = np.linspace(C["Z_MIN"] if z_lo is None else z_lo, C["Z_MAX"] if z_hi is None else z_hi, n)
    r = radius_fn(z)
    if clip_r is not None:
        r = np.minimum(r, clip_r)
    return float(np.trapezoid(np.pi * r ** 2, z)) / 1e6


def bin_sets(C, data, b):
    """A b. impulzus-bin tanító- és teszteseményei."""
    tr = (data["train"].ev["bin"] == b).nonzero().squeeze(1)
    te = (data["test"].ev["bin"] == b).nonzero().squeeze(1)
    return data["train"].subset(tr), data["test"].subset(te)
