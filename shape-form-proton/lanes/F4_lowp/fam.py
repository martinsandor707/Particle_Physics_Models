"""fam.py - model families of lane F4_lowp (low-momentum / stopping protons, energy variable). No side effects at import.

All classes derive from shapelab.PolyFormula: forward(lnp) -> log P (B, 128, 32), n_params() = number of free coefficients.
The only input is ln p; the kinetic energy T = sqrt(p^2 + m_p^2) - m_p is a fixed function of it.

Classes
-------
GCTVar     gamma x core+tail (the reference family) with the polynomial variable v(p) chosen by `var`
           ("lnp", "lnT", "lnE", "beta", "lnTsat", "lnR"): Q3 and Q4 of the lane.
StopMix    shower (gamma x core+tail) + stopping track (tube along the axis that ends at the range
           R(T) = R1 T^k - R0) + entry spike (blob at the origin): Q2.
"""
import math
import sys

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import torch
import torch.nn.functional as F

import families
from shapelab import PolyFormula

M_P = 0.938272          # proton mass [GeV]


def kinetic(lnp):
    """Kinetic energy T [GeV] of a proton of momentum p = exp(lnp) [GeV]."""
    p = torch.exp(lnp)
    return p * p / (torch.sqrt(p * p + M_P ** 2) + M_P)          # = sqrt(p^2 + m^2) - m, without cancellation


VARS = {
    "lnp": lambda lnp: lnp,
    "lnT": lambda lnp: torch.log(kinetic(lnp)),
    "lnE": lambda lnp: 0.5 * torch.log(torch.exp(2 * lnp) + M_P ** 2),
    "beta": lambda lnp: torch.exp(lnp) / torch.sqrt(torch.exp(2 * lnp) + M_P ** 2),
    "lnTsat": lambda lnp: torch.log(kinetic(lnp) / (kinetic(lnp) + M_P)),                 # ~ln T at low T, saturates at high T
    "lnR": lambda lnp: torch.log(F.softplus((kinetic(lnp) ** 1.77 - 0.2 ** 1.77) * 20) / 20 + 0.005),   # ~ln of the range beyond the threshold
    "asinhT": lambda lnp: torch.asinh(kinetic(lnp) / 0.3),                                  # linear in T at low T, logarithmic above 0.3 GeV
}


class VarPoly(PolyFormula):
    """PolyFormula whose polynomial variable is x = (v(p) - v0) / vs, with v a fixed function of p (`var`).
    v0, vs: centre and half-width of v over the momentum range [lnp0 - lnp_s, lnp0 + lnp_s] (so x is in [-1, 1])."""

    def __init__(self, C, degree=1, degrees=None, lnp0=0.0, lnp_s=1.0, var="lnp"):
        super().__init__(C, degree, degrees=degrees, lnp0=lnp0, lnp_s=lnp_s)
        self.var = var
        lo, hi = [float(VARS[var](torch.tensor(float(lnp0) + sg * float(lnp_s)))) for sg in (-1.0, 1.0)]
        self.register_buffer("v0", torch.tensor((hi + lo) / 2))
        self.register_buffer("vs", torch.tensor((hi - lo) / 2))

    def xvar(self, lnp):
        return (VARS[self.var](lnp) - self.v0) / self.vs

    def params(self, lnp):
        x = self.xvar(lnp)
        return torch.stack([x ** k for k in range(self.degree + 1)], 1) @ (self.coef * self.mask).T


def _shower_logdens(th, Z, R):
    """gamma x (Cauchy core + Cauchy tail), lateral parameters linear in depth: identical to families.GammaCoreTail."""
    zk = Z / 1000
    rc = torch.exp(th["log_rc"] + th["s_rc"] * zk)
    rt = rc * (1 + torch.exp(th["log_ratio"] + th["s_ratio"] * zk))
    lp = th["logit_pc"] + th["s_pc"] * zk
    lat = torch.logaddexp(F.logsigmoid(lp) + families.lat_cauchy(R, rc), F.logsigmoid(-lp) + families.lat_cauchy(R, rt))
    return families.long_gamma(Z, th["z0"], torch.exp(th["log_tmax"]), torch.exp(th["log_am1"])) + lat


class GCTVar(VarPoly):
    """Reference family (gamma x core+tail, 9 raw parameters) with a selectable polynomial variable and per-parameter degrees."""
    SPEC = list(families.GammaCoreTail.SPEC)

    def __init__(self, C, degree=1, degrees=None, **kw):
        d = {"s_rc": 0, "s_ratio": 0, "s_pc": 0}
        d.update(degrees or {})
        super().__init__(C, degree, degrees=d, **kw)

    def log_density(self, th, Z, R):
        return _shower_logdens(th, Z, R)


class StopMix(VarPoly):
    r"""Shower + stopping track + entry spike.

    rho(z', r | p) = w_sh S(z', r) + w_tu U(z', r) + w_sp D(z', r), every component normalised on the grid.

    S (shower): gamma(z') x [core + tail Cauchy](r), parameters polynomial in x(v(p)).
    U (tube, the stopping primary beyond the first layer):
        U ~ sigma((z' - zt0)/4 mm) * sigma((R - z')/(wR R + 3 mm)) * exp(-kap min(z'/R, 1.5)) * exp(-r^2 / (2 st^2)),
        R(T) = softplus(R1 T^k - R0)   [mm, T in GeV]: range-energy power law (Bragg-Kleeman) with an offset.
    D (entry spike = the first hit layer, whose centroid defines the origin):
        D ~ exp(-(z' - sp_z)^2 / (2 sz^2)) * exp(-r^2 / (2 sr^2)).

    Weights, tie=False:  w_sp = sigma(logit_sp), w_tu = (1 - w_sp) sigma(logit_tr), w_sh = (1 - w_sp)(1 - sigma(logit_tr)).
    Weights, tie=True (the spike of the stopping primary is its first layer, f1 = d1 / (d1 + Lt), Lt = softplus(R - zt0)):
        w_trk = sigma(logit_tr), w_sp = w_trk f1 + (1 - w_trk) sigma(logit_sp), w_tu = w_trk (1 - f1),
        w_sh = (1 - w_trk)(1 - sigma(logit_sp)).
    gate="poly": logit_tr is a polynomial in x; gate="range": logit_tr = g0 - g1 ln R(T) (a smooth gate in the range).
    """
    SHOWER = list(families.GammaCoreTail.SPEC)
    SPIKE = [("logit_sp", -3.0), ("sp_z", 0.0), ("log_sp_sz", math.log(6.0)), ("log_sp_sr", math.log(10.0))]
    TRACK = [("logit_tr", -1.5), ("log_R1", math.log(700.0)), ("k_R", 1.77), ("R0", 15.0), ("log_wR", math.log(0.12)),
             ("kap", 0.3), ("zt0", 10.0), ("log_st", math.log(20.0))]
    TIE = [("log_d1", math.log(22.0))]
    GATE_R = [("g1", 1.0)]
    CONST = ("s_rc", "s_ratio", "s_pc", "sp_z", "log_sp_sz", "log_sp_sr", "log_R1", "k_R", "R0", "log_wR", "kap", "zt0", "log_st",
             "log_d1", "g1")

    def __init__(self, C, degree=2, degrees=None, spike=True, track=True, tie=False, gate="poly", var="lnT", **kw):
        spec = list(self.SHOWER) + (list(self.SPIKE) if spike else []) + (list(self.TRACK) if track else [])
        if tie and spike and track:
            spec += list(self.TIE)
        if gate == "range" and track:
            spec += list(self.GATE_R)
        object.__setattr__(self, "SPEC", spec)
        d = {n: 0 for n, _ in spec if n in self.CONST}
        if gate == "range":
            d["logit_tr"] = 0
        d.update(degrees or {})
        super().__init__(C, degree, degrees=d, var=var, **kw)
        self.use_spike, self.use_track, self.tie, self.gate = bool(spike), bool(track), bool(tie and spike and track), gate

    def theta(self, lnp):
        th = super().theta(lnp)
        th["lnT"] = VARS["lnT"](lnp)[:, None, None]
        return th

    # ---- pieces ----
    def range_mm(self, th):
        """R(T) = softplus(R1 T^k - R0) [mm]; smooth floor of a few mm so that the tube never has negative length."""
        raw = torch.exp(th["log_R1"] + th["k_R"] * th["lnT"]) - th["R0"]
        return F.softplus(raw / 5.0) * 5.0 + 1.0

    def log_tube(self, th, Z, R):
        Rr = self.range_mm(th)
        wR = torch.exp(th["log_wR"]) * Rr + 3.0
        lon = F.logsigmoid((Z - th["zt0"]) / 4.0) + F.logsigmoid((Rr - Z) / wR) - th["kap"] * (Z / Rr).clamp(-1.0, 1.5)
        st = torch.exp(th["log_st"])
        return lon - R * R / (2 * st * st)

    def log_spike(self, th, Z, R):
        sz, sr = torch.exp(th["log_sp_sz"]), torch.exp(th["log_sp_sr"])
        return -(Z - th["sp_z"]) ** 2 / (2 * sz * sz) - R * R / (2 * sr * sr)

    def log_weights(self, th):
        """log of (w_shower, w_tube, w_spike), each (B, 1, 1) or None."""
        zero = torch.zeros_like(th["lnT"])
        if self.use_track:
            if self.gate == "range":
                lt = th["logit_tr"] - th["g1"] * torch.log(self.range_mm(th))
            else:
                lt = th["logit_tr"]
            l_tr, l_ntr = F.logsigmoid(lt), F.logsigmoid(-lt)
        else:
            l_tr, l_ntr = None, zero
        if not self.use_spike:
            return l_ntr, l_tr, None
        l_sp, l_nsp = F.logsigmoid(th["logit_sp"]), F.logsigmoid(-th["logit_sp"])
        if not self.use_track:
            return l_nsp, None, l_sp
        if not self.tie:
            return l_nsp + l_ntr, l_nsp + l_tr, l_sp
        d1 = torch.exp(th["log_d1"])
        Lt = F.softplus((self.range_mm(th) - th["zt0"]) / 5.0) * 5.0
        l_f1, l_nf1 = torch.log(d1) - torch.log(d1 + Lt), torch.log(Lt.clamp_min(1e-12)) - torch.log(d1 + Lt)
        return l_ntr + l_nsp, l_tr + l_nf1, torch.logaddexp(l_tr + l_f1, l_ntr + l_sp)

    def components(self, lnp):
        """(log weights, normalised log P of the components) - for figures and diagnostics."""
        th = self.theta(lnp)
        Z, R, B = self.ZZ[None], self.RR[None], len(lnp)
        norm = lambda ld: F.log_softmax((ld + self.LOG_VOL).reshape(B, -1), 1).reshape(B, self.N_Z, self.N_R)
        lw = self.log_weights(th)
        comp = [norm(_shower_logdens(th, Z, R)),
                norm(self.log_tube(th, Z, R)) if self.use_track else None,
                norm(self.log_spike(th, Z, R)) if self.use_spike else None]
        return lw, comp, th

    def forward(self, lnp):
        lw, comp, _ = self.components(lnp)
        terms = [w + c for w, c in zip(lw, comp) if c is not None]
        return torch.logsumexp(torch.stack(terms, 0), 0)

    def log_density(self, th, Z, R):          # not used (forward is overridden); kept for the PolyFormula interface
        return _shower_logdens(th, Z, R)
