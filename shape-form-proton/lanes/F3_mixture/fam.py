"""fam - lane F3: mixtures of separable components ("several nested droplets") for the proton shower shape.

    rho(z', r | p) = sum_k pi_k(p) * L_k(z' | p) * T_k(r | z', p)

Every component is normalised on the (z', r) grid (log_softmax of log density + LOG_VOL), the components are mixed with
log-weights through logsumexp, so pi_k(p) are energy fractions (on the full grid, before any detector cut).
Every raw parameter is a polynomial of x = (v(p) - v0) / vs with its own degree (shapelab.PolyFormula); v = ln p (default)
or v = ln T, T = kinetic energy of the proton (a fixed, parameter-free function of p).

No side effects at import (apart from making shapelab importable).
"""
import math
import re
import sys

import torch
import torch.nn.functional as F

if "/home/user/tamas/shape-form-proton" not in sys.path:
    sys.path.insert(0, "/home/user/tamas/shape-form-proton")
from shapelab import PolyFormula  # noqa: E402

LOG_PI, LOG_2PI = math.log(math.pi), math.log(2 * math.pi)
EPS_R = 1.0          # [mm] regularisation of 1/r on the axis (fixed constant)
M_P = 0.938272       # proton mass [GeV]


def kinetic(lnp):
    """Kinetic energy T [GeV] of a proton of momentum p = exp(lnp) [GeV] (no cancellation at small p)."""
    p = torch.exp(lnp)
    return p * p / (torch.sqrt(p * p + M_P ** 2) + M_P)


VARS = {"lnp": lambda lnp: lnp, "lnT": lambda lnp: torch.log(kinetic(lnp))}


def _log_softplus(x):
    """log(softplus(x)), stable (value and gradient) for very negative x."""
    xc = x.clamp_min(-20.0)
    return torch.log(F.softplus(xc)) + (x - xc)


# ----------------------------------------------------------------------------------------------------------------------
# longitudinal building blocks: name -> ([(parameter, initial raw value)], log L(z'))      (un-normalised)
# ----------------------------------------------------------------------------------------------------------------------
def _long_gamma0(t, z):
    """The report's gamma: s^(a-1) exp(-(a-1) s / tmax), s = 2 softplus((z - z0)/2) + 1e-3 (floor -> flat pedestal upstream)."""
    s = F.softplus((z - t["z0"]) / 2) * 2 + 1e-3
    am1 = torch.exp(t["lam1"])
    return am1 * torch.log(s) - am1 / torch.exp(t["ltmax"]) * s


def _long_gamma(t, z, w=2.0):
    """Clean gamma: s^(a-1) exp(-(a-1) s / tmax), s = w softplus((z - z0)/w); upstream it dies as exp((a-1)(z - z0)/w)."""
    x = (z - t["z0"]) / w
    am1 = torch.exp(t["lam1"])
    return am1 * (_log_softplus(x) + (math.log(w) if isinstance(w, float) else torch.log(w))) - am1 / torch.exp(t["ltmax"]) * F.softplus(x) * w


def _long_gammaw(t, z):
    """Gamma with a free onset softness w: backward exponential tail of length w/(a-1), forward gamma."""
    return _long_gamma(t, z, torch.exp(t["lw"]))


def _long_alap(t, z):
    """Asymmetric Laplace: exp(-(z - z0)/lf) downstream of z0, exp(-(z0 - z)/lb) upstream."""
    d = z - t["z0"]
    return -F.relu(d) / torch.exp(t["llf"]) - F.relu(-d) / torch.exp(t["llb"])


def _long_expo(t, z):
    """Exponential in depth with a logistic onset: sigmoid((z - z0)/w) exp(-z/lf)."""
    return F.logsigmoid((z - t["z0"]) / torch.exp(t["lw"])) - z / torch.exp(t["llf"])


def _long_gauss(t, z):
    return -0.5 * ((z - t["z0"]) / torch.exp(t["lsz"])) ** 2


def _long_track(t, z):
    """Stopping primary: sigmoid((z - z0)/4 mm) * sigmoid((Rg - z)/(wr Rg + 3 mm)) * exp(kap z / Rg), Rg = exp(lrng) = range."""
    rg = torch.exp(t["lrng"])
    return (F.logsigmoid((z - t["z0"]) / 4.0) + F.logsigmoid((rg - z) / (torch.exp(t["lwr"]) * rg + 3.0))
            + t["kap"] * (z / rg).clamp(-1.0, 1.5))


LONG = {
    "gamma0": ([("z0", -20.0), ("ltmax", math.log(100)), ("lam1", math.log(3))], _long_gamma0),
    "gamma": ([("z0", -20.0), ("ltmax", math.log(100)), ("lam1", math.log(3))], _long_gamma),
    "gammaw": ([("z0", -20.0), ("ltmax", math.log(100)), ("lam1", math.log(3)), ("lw", math.log(10))], _long_gammaw),
    "alap": ([("z0", 0.0), ("llf", math.log(100)), ("llb", math.log(20))], _long_alap),
    "expo": ([("z0", 0.0), ("lw", math.log(10)), ("llf", math.log(300))], _long_expo),
    "gauss": ([("z0", 0.0), ("lsz", math.log(8))], _long_gauss),
    "track": ([("z0", 5.0), ("lrng", math.log(150)), ("lwr", math.log(0.15)), ("kap", 0.0)], _long_track),
}


# ----------------------------------------------------------------------------------------------------------------------
# lateral building blocks: name -> ([(parameter, initial raw value)], name of the width parameter, log T(r; width, t))
# (per unit area; the constant factors are irrelevant because every component is normalised on the grid)
# ----------------------------------------------------------------------------------------------------------------------
def _lat_cauchy(t, r, R):
    """R^2 / (pi (r^2 + R^2)^2)"""
    return 2 * torch.log(R) - LOG_PI - 2 * torch.log(r * r + R * R)


def _lat_moffat(t, r, R):
    """(q-1)/(pi R^2) (1 + r^2/R^2)^(-q), q = 1 + exp(lq)   (q = 2: the Cauchy-like term)"""
    q = 1 + torch.exp(t["lq"])
    return t["lq"] - LOG_PI - 2 * torch.log(R) - q * torch.log1p(r * r / (R * R))


def _lat_gauss(t, r, s):
    return -r * r / (2 * s * s) - LOG_2PI - 2 * torch.log(s)


def _lat_exp(t, r, lam):
    """exp(-r/lam) / (2 pi lam^2)"""
    return -r / lam - LOG_2PI - 2 * torch.log(lam)


def _lat_expr(t, r, lam):
    """line source: exp(-r/lam) / (2 pi lam (r + eps))   (dE/dr purely exponential)"""
    return -r / lam - LOG_2PI - torch.log(lam) - torch.log(r + EPS_R)


def _lat_pexp(t, r, lam):
    """(r + eps)^(-q) exp(-r/lam), q = 2 sigmoid(qraw) in (0, 2)"""
    q = 2 * torch.sigmoid(t["qraw"])
    return -q * torch.log(r + EPS_R) - r / lam - (2 - q) * torch.log(lam)


def _lat_k0(t, r, lam):
    """K0(r/lam) / (2 pi lam^2): steady-state diffusion from a line source"""
    x = (r / lam).clamp_min(1e-6)
    return torch.log(torch.special.scaled_modified_bessel_k0(x)) - x - LOG_2PI - 2 * torch.log(lam)


def _lat_sexp(t, r, lam):
    """stretched exponential exp(-(r/lam)^b) , b = exp(lb)"""
    return -torch.exp(torch.exp(t["lb"]) * torch.log((r / lam).clamp_min(1e-9))) - 2 * torch.log(lam)


LAT = {
    "cauchy": ([("lR", math.log(10))], "lR", _lat_cauchy),
    "moffat": ([("lR", math.log(10)), ("lq", 0.0)], "lR", _lat_moffat),
    "gauss": ([("lsg", math.log(10))], "lsg", _lat_gauss),
    "exp": ([("llam", math.log(40))], "llam", _lat_exp),
    "expr": ([("llam", math.log(90))], "llam", _lat_expr),
    "pexp": ([("llam", math.log(90)), ("qraw", 0.0)], "llam", _lat_pexp),
    "k0": ([("llam", math.log(90))], "llam", _lat_k0),
    "sexp": ([("llam", math.log(40)), ("lb", 0.0)], "llam", _lat_sexp),
}

LONG_TEX = {
    "gamma0": r"s^{a}e^{-a s/t_{max}},\ s=2\,\mathrm{softplus}((z'-z_0)/2)+10^{-3},\ a=e^{lam1},\ t_{max}=e^{ltmax}",
    "gamma": r"s^{a}e^{-a s/t_{max}},\ s=2\,\mathrm{softplus}((z'-z_0)/2)",
    "gammaw": r"s^{a}e^{-a s/t_{max}},\ s=w\,\mathrm{softplus}((z'-z_0)/w),\ w=e^{lw}",
    "alap": r"e^{-(z'-z_0)_+/\ell_f-(z_0-z')_+/\ell_b}",
    "expo": r"\sigma((z'-z_0)/w)\,e^{-z'/\ell_f}",
    "gauss": r"e^{-(z'-z_0)^2/2\sigma_z^2}",
    "track": r"\sigma((z'-z_0)/4)\,\sigma((R_g-z')/(w_r R_g+3))\,e^{\kappa z'/R_g},\ R_g=e^{lrng}",
}
LAT_TEX = {
    "cauchy": r"R^2/(\pi(r^2+R^2)^2)", "moffat": r"(1+r^2/R^2)^{-q}", "gauss": r"e^{-r^2/2\sigma^2}/(2\pi\sigma^2)",
    "exp": r"e^{-r/\lambda}/(2\pi\lambda^2)", "expr": r"e^{-r/\lambda}/(2\pi\lambda(r+1))", "pexp": r"(r+1)^{-q}e^{-r/\lambda}",
    "k0": r"K_0(r/\lambda)/(2\pi\lambda^2)", "sexp": r"e^{-(r/\lambda)^b}",
}


class Mixture(PolyFormula):
    """rho = sum_k pi_k(p) L_k(z'|p) T_k(r|z',p): mixture of separable components, each normalised on the grid.

    comps   : list of "long/lat" strings, e.g. ["gamma/cauchy", "gammaw/expr+z"]; "+z" adds a depth slope to the lateral width,
              width(z') = exp(l + s z'/1000).  Parameter names: c{k}_{name} (c0_z0, c0_ltmax, c1_llam, c1_sz ...) and the
              mixture logits w1 ... w{K-1} (relative to component 0).
    degree  : default polynomial degree of every parameter in x.
    degrees : {regex: degree} overrides (full match on the parameter name, later entries win); depth slopes default to degree 0.
    fixed   : {regex: value} parameters frozen at a constant (not trainable, not counted).
    init    : {name: raw initial value, or list of polynomial coefficients [c0, c1, ...]} overrides.
    var     : "lnp" (x = (ln p - lnp0)/lnp_s) or "lnT" (x = (ln T - v0)/vs, T = kinetic energy; v0, vs = centre and half-width
              of ln T over the same momentum range, so x is in [-1, 1] as well).
    """

    def __init__(self, C, comps, degree=1, degrees=None, fixed=None, init=None, lnp0=0.0, lnp_s=1.0, var="lnp"):
        self.comps = []
        spec = []
        for k, c in enumerate(comps):
            lg, lt = c.split("/")
            slope = lt.endswith("+z")
            lt = lt[:-2] if slope else lt
            self.comps.append((lg, lt, slope))
            spec += [(f"c{k}_{n}", v) for n, v in LONG[lg][0]] + [(f"c{k}_{n}", v) for n, v in LAT[lt][0]]
            if slope:
                spec.append((f"c{k}_sz", 0.0))
        spec += [(f"w{k}", 0.0) for k in range(1, len(comps))]
        init = dict(init or {})
        unknown = set(init) - {n for n, _ in spec}
        if unknown:
            raise KeyError(f"unknown init names: {sorted(unknown)}")
        first = lambda v: float(v[0]) if isinstance(v, (list, tuple)) else float(v)
        spec = [(n, first(init.get(n, v))) for n, v in spec]
        self.fixed_values = {}
        for pat, val in (fixed or {}).items():
            for n, _ in spec:
                if re.fullmatch(pat, n):
                    self.fixed_values[n] = float(val)
        self.SPEC = [(n, v) for n, v in spec if n not in self.fixed_values]
        deg = {n: (0 if n.endswith("_sz") else degree) for n, _ in self.SPEC}
        for pat, d in (degrees or {}).items():
            for n in deg:
                if re.fullmatch(pat, n):
                    deg[n] = d
        super().__init__(C, degree, degrees=deg, lnp0=lnp0, lnp_s=lnp_s)
        self.K = len(self.comps)
        self.var = var
        lo, hi = [float(VARS[var](torch.tensor(float(lnp0) + sg * float(lnp_s)))) for sg in (-1.0, 1.0)]
        self.register_buffer("v0", torch.tensor((hi + lo) / 2))
        self.register_buffer("vs", torch.tensor((hi - lo) / 2))
        with torch.no_grad():                                   # higher-order initial coefficients
            for n, v in init.items():
                if isinstance(v, (list, tuple)) and n in self.names:
                    i = self.names.index(n)
                    for k, x in enumerate(v[:self.deg[n] + 1]):
                        self.coef[i, k] = float(x)

    def xvar(self, lnp):
        return (VARS[self.var](lnp) - self.v0) / self.vs

    def params(self, lnp):
        x = self.xvar(lnp)
        return torch.stack([x ** k for k in range(self.degree + 1)], 1) @ (self.coef * self.mask).T

    def theta(self, lnp):
        th = super().theta(lnp)
        for n, v in self.fixed_values.items():
            th[n] = torch.full((len(lnp), 1, 1), v, device=lnp.device)
        return th

    def components(self, lnp):
        """(log pi (B, K), log P_k (B, K, N_Z, N_R)): mixture log-weights and the grid-normalised component maps."""
        th = self.theta(lnp)
        Z, R, B = self.ZZ[None], self.RR[None], len(lnp)
        maps = []
        for k, (lg, lt, slope) in enumerate(self.comps):
            t = {n[len(f"c{k}_"):]: v for n, v in th.items() if n.startswith(f"c{k}_")}
            _, wname, fn = LAT[lt]
            lw = t[wname] + (t["sz"] * Z / 1000 if slope else 0.0)
            ld = LONG[lg][1](t, Z) + fn(t, R, torch.exp(lw)) + self.LOG_VOL
            maps.append(F.log_softmax(ld.reshape(B, -1), 1))
        logits = torch.cat([torch.zeros(B, 1, device=lnp.device)] + [th[f"w{k}"].reshape(B, 1) for k in range(1, self.K)], 1)
        return F.log_softmax(logits, 1), torch.stack(maps, 1).reshape(B, self.K, self.N_Z, self.N_R)

    def forward(self, lnp):
        lw, maps = self.components(lnp)
        return torch.logsumexp(maps + lw[:, :, None, None], 1)

    @torch.no_grad()
    def physical(self, p_values=(0.5, 1.5, 5.0, 15.0)):
        """Physical values of the parameters at a few momenta (l* -> exp, weights -> fractions), for inspection."""
        lnp = torch.log(torch.tensor(p_values, dtype=torch.float32, device=self.coef.device))
        th = {n: v.reshape(-1).cpu() for n, v in self.theta(lnp).items()}
        lw, _ = self.components(lnp)
        out = {}
        for k in range(self.K):
            out[f"pi{k}"] = lw[:, k].exp().cpu().tolist()
        for n, v in th.items():
            base = n.split("_", 1)[-1]
            if n.startswith("w"):
                continue
            if base.startswith("l"):
                out[n.replace("_l", "_", 1)] = v.exp().tolist()
            else:
                out[n] = v.tolist()
        return out

    def formula_md(self, title="", roles=None):
        """Markdown: the closed form of every component and the fitted polynomial coefficients."""
        c, s = float(self.v0), float(self.vs)
        v = "\\ln(p/\\mathrm{GeV})" if self.var == "lnp" else "\\ln(T/\\mathrm{GeV}),\\ T=\\sqrt{p^2+m_p^2}-m_p"
        out = [f"### {title}", "", r"$\rho(z',r\,|\,p)=\sum_k \pi_k(p)\,\hat L_k(z')\,\hat T_k(r)$, every component normalised on the grid; "
               r"$\pi=\mathrm{softmax}(0,w_1,\dots)$; raw parameter $=\sum_j c_j x^j$, " + f"$x=({v.split(',')[0]} {-c:+.4f})/{s:.4f}$"
               + ("" if self.var == "lnp" else f" (${v}$)") + "; `l*` parameters are logarithms [mm]; `+z`: width $\\times e^{sz\\,z'/1000}$.", ""]
        for k, (lg, lt, slope) in enumerate(self.comps):
            role = f" ({roles[k]})" if roles else ""
            out.append(f"- component {k}{role}: $L_{k}\\propto {LONG_TEX[lg]}$; $T_{k}\\propto {LAT_TEX[lt]}$" + (" , width with depth slope" if slope else ""))
        out += ["", "| parameter | coefficients $c_0, c_1, \\dots$ |", "|---|---|"]
        cm = (self.coef * self.mask).detach().cpu()
        for i, n in enumerate(self.names):
            out.append(f"| `{n}` | " + ", ".join(f"{float(x):+.5g}" for x in cm[i, :self.deg[n] + 1]) + " |")
        for n, x in self.fixed_values.items():
            out.append(f"| `{n}` (fixed) | {x:+.5g} |")
        out += ["", f"Number of free parameters: **{self.n_params()}**"]
        return "\n".join(out)
