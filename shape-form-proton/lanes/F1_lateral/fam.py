"""fam - lateral-profile families for the proton shower (lane F1_lateral).

rho(z', r | p) = L_gamma(z' | p) * T(r | z', p),   T = sum_k f_k(z', p) T_k(r; w_k(z', p), shape_k)

Every lateral block T_k is normalised to unit area on the infinite plane (int T 2 pi r dr = 1); the longitudinal factor is the
baseline gamma (families.long_gamma), so only the lateral part differs from families.GammaCoreTail.
Every raw parameter is a polynomial in L = (ln p - lnp0)/lnp_s (shapelab.PolyFormula); depth-dependence coefficients have
degree 0 (p-independent) unless overridden through `degrees`.

No side effects at import.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from shapelab import PolyFormula
from families import long_gamma

LOG_PI, LOG_2PI = math.log(math.pi), math.log(2 * math.pi)
HAT_KNOTS = (-200.0, -100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0, 1450.0)


# ---------------------------------------------------------------------------------------------------------------------
# log K0 with a gradient (torch.special.*bessel_k0 has no autograd): d/dx ln K0(x) = -K1(x)/K0(x)
# ---------------------------------------------------------------------------------------------------------------------
class _LogK0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        k0e = torch.special.scaled_modified_bessel_k0(x)
        k1e = torch.special.scaled_modified_bessel_k1(x)
        ctx.save_for_backward(k1e / k0e)
        return torch.log(k0e) - x

    @staticmethod
    def backward(ctx, g):
        (ratio,) = ctx.saved_tensors
        return -g * ratio


def log_k0(x):
    return _LogK0.apply(x)


# ---------------------------------------------------------------------------------------------------------------------
# lateral blocks: log T(r), int T 2 pi r dr = 1
# ---------------------------------------------------------------------------------------------------------------------
def lat_cauchy(r, w):
    """GFlash term: w^2 / (pi (r^2 + w^2)^2)."""
    return 2 * torch.log(w) - LOG_PI - 2 * torch.log(r * r + w * w)


def lat_gauss(r, w):
    return -r * r / (2 * w * w) - LOG_2PI - 2 * torch.log(w)


def lat_exp(r, w):
    """e^{-r/w} / (2 pi w^2): dE/dr = r e^{-r/w} / w^2."""
    return -r / w - LOG_2PI - 2 * torch.log(w)


def lat_expr(r, w, eps=1.0):
    """'line source': e^{-r/w} / (2 pi w (r + eps)): dE/dr is a pure exponential; eps = 1 mm regularises the axis."""
    return -r / w - LOG_2PI - torch.log(w) - torch.log(r + eps)


def lat_k0(r, w):
    """2D diffusion (Yukawa) kernel: K0(r/w) / (2 pi w^2)."""
    return log_k0(r / w) - LOG_2PI - 2 * torch.log(w)


def lat_student(r, w, nu):
    """Generalised Cauchy / Student-t: (nu-1)/(pi w^2) (1 + r^2/w^2)^(-nu); nu = 2 is the GFlash term, tail ~ r^(-2 nu)."""
    return torch.log(nu - 1) - LOG_PI - 2 * torch.log(w) - nu * torch.log1p(r * r / (w * w))


def lat_ggamma(r, w, k, eps=1.0):
    """Gamma-distributed radius: dE/dr ~ r^(k-1) e^{-r/w}:  T = (r+eps)^(k-2) e^{-r/w} / (2 pi w^k Gamma(k)).
    k = 1: 'line source' e^{-r/w}/r;  k = 2: plain exponential density."""
    return (k - 2) * torch.log(r + eps) - r / w - LOG_2PI - k * torch.log(w) - torch.lgamma(k)


def lat_sexp(r, w, b):
    """Stretched exponential: exp(-(r/w)^b) b / (2 pi w^2 Gamma(2/b));  b = 1: exponential, b = 2: Gauss."""
    return -(r / w) ** b - LOG_2PI - 2 * torch.log(w) - torch.lgamma(2 / b) + torch.log(b)


def lat_sexpr(r, w, b, eps=1.0):
    """Stretched exponential over r: dE/dr ~ exp(-(r/w)^b):  T = exp(-(r/w)^b) b / (2 pi w Gamma(1/b) (r+eps))."""
    return -(r / w) ** b - LOG_2PI - torch.log(w) - torch.lgamma(1 / b) + torch.log(b) - torch.log(r + eps)


# kind -> (function, shape parameters [(raw name, initial raw value, transform)])
KINDS = {
    "cauchy": (lat_cauchy, []),
    "gauss": (lat_gauss, []),
    "exp": (lat_exp, []),
    "expr": (lat_expr, []),
    "k0": (lat_k0, []),
    "student": (lat_student, [("lnu", 0.0, lambda q: 1 + torch.exp(q))]),          # nu = 1 + e^q   (q = 0: Cauchy term)
    "ggamma": (lat_ggamma, [("lk", 0.0, torch.exp)]),                              # k = e^q        (q = 0: line source)
    "sexp": (lat_sexp, [("lb", 0.0, torch.exp)]),                                  # b = e^q        (q = 0: exponential)
    "sexpr": (lat_sexpr, [("lb", 0.0, torch.exp)]),                                # b = e^q        (q = 0: line source)
}
W_INIT = {1: (15.0,), 2: (10.0, 60.0), 3: (6.0, 30.0, 100.0), 4: (4.0, 15.0, 50.0, 120.0)}
F_INIT = {1: (), 2: (-0.85,), 3: (-0.5, -1.0), 4: (-0.5, -1.0, -1.5)}


class LatMix(PolyFormula):
    """rho = gamma(z') x sum_k f_k T_k(r; w_k): a mixture of lateral blocks with depth-dependent widths and fractions.

    comps  : kinds of the blocks, e.g. ("cauchy", "expr") (see KINDS)
    depth  : depth dependence of the lateral raw parameters. A string applies the same bases to every log-width `log_w<k>`
             and fraction logit `lf<k>`; a dict {raw name: (bases...)} sets it per parameter (others: none).
             Bases g(z'):  "lin" z'/1000 | "pos" z'_+/1000 | "neg" z'_-/100 | "sqrt" sqrt(z'_+/1000) |
                           "sat" 1 - exp(-z'_+/zs) (zs free) | "log" ln(1 + z'_+/zs) (zs free) | "hat" piecewise linear (HAT_KNOTS)
             raw(z') = raw(p) + sum_g s_g(p) g(z')      (z'_+ = max(z', 0), z'_- = max(-z', 0))
    frac   : "logit": f = softmax(0, lf1, lf2, ...);
             "expdecay": f_0 = sigmoid(lf_a) [sigmoid(lf_b) + (1 - sigmoid(lf_b)) exp(-z'_+/lam_c)], the others share the rest.
    """

    def __init__(self, C, degree=1, comps=("cauchy", "cauchy"), depth="lin", frac="logit", w_init=None, f_init=None,
                 degrees=None, **kw):
        comps = tuple(comps)
        n = len(comps)
        spec = [("z0", -20.0), ("log_tmax", math.log(100)), ("log_am1", math.log(3))]
        lat = []
        w_init = w_init or W_INIT[n]
        for i, kind in enumerate(comps):
            spec.append((f"log_w{i}", math.log(w_init[i])))
            lat.append(f"log_w{i}")
            for sn, sv, _ in KINDS[kind][1]:
                spec.append((f"{sn}{i}", sv))
        f_init = f_init or F_INIT[n]
        first_logit = 1
        deg = {}
        if frac == "expdecay":
            spec += [("lf_a", 0.85), ("lf_b", -2.0), ("log_lamc", math.log(200.0))]
            deg.update({"lf_b": 0, "log_lamc": 0})
            first_logit = 2
        for i in range(first_logit, n):
            spec.append((f"lf{i}", f_init[i - 1]))
            lat.append(f"lf{i}")
        if depth is None or isinstance(depth, str):
            depth = {q: ((depth,) if depth else ()) for q in lat}
        depth = {q: tuple(b) for q, b in depth.items()}
        for q, bases in depth.items():
            for b in bases:
                if b == "hat":
                    for j, k in enumerate(HAT_KNOTS):
                        if k != 0.0:
                            spec.append((f"h{j}_{q}", 0.0))
                            deg[f"h{j}_{q}"] = 0
                else:
                    spec.append((f"s_{b}_{q}", 0.0))
                    deg[f"s_{b}_{q}"] = 0
                    if b in ("sat", "log"):
                        spec.append((f"lzs_{b}_{q}", math.log(150.0)))
                        deg[f"lzs_{b}_{q}"] = 0
        deg.update(degrees or {})
        object.__setattr__(self, "SPEC", spec)
        super().__init__(C, degree, degrees=deg, **kw)
        self.comps, self.depth_spec, self.frac, self.first_logit = comps, depth, frac, first_logit
        zc = C["ZC"].detach().cpu().numpy()
        eye = np.eye(len(HAT_KNOTS))
        self.register_buffer("HAT", torch.tensor(np.stack([np.interp(zc, HAT_KNOTS, eye[j]) for j in range(len(HAT_KNOTS))], 1),
                                                 dtype=torch.float32))

    # ---- depth dependence ----
    def _basis(self, b, q, th, Z):
        zp = Z.clamp_min(0.0)
        if b == "lin":
            return Z / 1000
        if b == "pos":
            return zp / 1000
        if b == "neg":
            return (-Z).clamp_min(0.0) / 100
        if b == "sqrt":
            return torch.sqrt(zp / 1000 + 1e-6)
        if b == "sat":
            return 1 - torch.exp(-zp / torch.exp(th[f"lzs_sat_{q}"]))
        if b == "log":
            return torch.log1p(zp / torch.exp(th[f"lzs_log_{q}"]))
        raise KeyError(b)

    def raw(self, th, q, Z):
        """Raw parameter q as a function of depth: (B, N_Z, N_R)-broadcastable."""
        v = th[q]
        for b in self.depth_spec.get(q, ()):
            if b == "hat":
                for j, k in enumerate(HAT_KNOTS):
                    if k != 0.0:
                        v = v + th[f"h{j}_{q}"] * self.HAT[None, :, j, None]
            else:
                v = v + th[f"s_{b}_{q}"] * self._basis(b, q, th, Z)
        return v

    def log_fractions(self, th, Z):
        n = len(self.comps)
        zero = torch.zeros_like(th["z0"] + Z)
        if self.frac == "expdecay":
            a, b = self.raw(th, "lf_a", Z), th["lf_b"]
            lam = torch.exp(th["log_lamc"])
            lf0 = F.logsigmoid(a) + torch.logaddexp(F.logsigmoid(b), F.logsigmoid(-b) - Z.clamp_min(0.0) / lam) + zero
            rest = torch.log1p(-torch.exp(lf0).clamp_max(1 - 1e-7))
            if n == 2:
                return [lf0, rest]
            lg = torch.log_softmax(torch.stack([zero] + [self.raw(th, f"lf{i}", Z) + zero for i in range(2, n)], 0), 0)
            return [lf0] + [rest + lg[i] for i in range(n - 1)]
        if n == 1:
            return [zero]
        lg = torch.log_softmax(torch.stack([zero] + [self.raw(th, f"lf{i}", Z) + zero for i in range(1, n)], 0), 0)
        return [lg[i] for i in range(n)]

    def log_lateral(self, th, Z, R):
        lf = self.log_fractions(th, Z)
        terms = []
        for i, kind in enumerate(self.comps):
            fn, shapes = KINDS[kind]
            w = torch.exp(self.raw(th, f"log_w{i}", Z))
            args = [tr(self.raw(th, f"{sn}{i}", Z)) for sn, _, tr in shapes]
            terms.append(lf[i] + fn(R, w, *args))
        return terms[0] if len(terms) == 1 else torch.logsumexp(torch.stack(terms, 0), 0)

    def log_density(self, th, Z, R):
        return long_gamma(Z, th["z0"], torch.exp(th["log_tmax"]), torch.exp(th["log_am1"])) + self.log_lateral(th, Z, R)
