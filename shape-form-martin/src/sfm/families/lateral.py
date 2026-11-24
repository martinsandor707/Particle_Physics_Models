"""Radial (lateral) families, defined through the survival function of enclosed energy, and their mixtures.

For a term with area density T(r) (int_0^inf T 2 pi r dr = 1):

    S(r) = int_r^inf T(r') 2 pi r' dr'        energy fraction outside radius r;   dE/dr = -S'(r) = 2 pi r T(r)
    ring mass  = S(r_j) - S(r_{j+1})          exact, whatever T does at the axis
    disc mass  = 1 - S(700 mm)

    family        ln S(r)                         T(r) at the axis        nests / is nested in
    gauss         -r^2 / (2 sigma^2)              finite                  weibull_core at b = 2
    cauchy        -log1p(r^2 / R^2)               finite                  burr at (c, k) = (2, 1)   [GFlash term]
    student       -k log1p(r^2 / R^2)             finite                  burr at c = 2; 2-D Student-t, nu = 2k
    burr          -k log1p((r/R)^c)               ~ r^(c-2)               Lomax in r at c = 1; Weibull as k -> inf
    exp           -r / lam                        ~ 1/r                   weibull at b = 1; the legacy exp(-r/lam)/r
                                                                          halo without its regulariser
    weibull       -(r/lam)^b                      ~ r^(b-2)               stretched exponential (b < 1)
    weibull_core  -(r/lam)^b / 2                  ~ r^(b-2)               gauss at b = 2 with lam = sigma
    exp_reg       no closed form in torch         finite                  legacy T = exp(-r/lam) / (2 pi lam (r + eps));
                                                                          plane mass 1 - (eps/lam) e^(eps/lam) E1(eps/lam) < 1

``exp_reg`` is integrated with ``Grid.ring_logmass_quad`` (Gauss-Legendre in ln(r + 1 mm)); everything else is a
difference of closed-form survival values.

Hazards
  * T diverges at the axis for burr with c < 2, weibull with b < 2 and exp. The ring masses are exact and finite;
    the cell-centre rule (``integration="centre"``) evaluates T at RC_0 = 1.37 mm and is then a poor approximation of
    the first ring. c > 2 or b > 2 makes T vanish at the axis (a hollow profile).
  * ln r at the edge r_0 = 0 is taken at 1e-30 mm inside the formulas (finite values and gradients) and S(0) = 1 is
    then set exactly.
  * float32: a ring mass is a difference of two rounded ln S values; where ln S changes by less than ~1e-3 of its
    size over one ring (weibull with b < 0.3 on the outer rings, any term much wider than the grid) the relative
    accuracy of that ring mass drops to ~1e-4. Double precision does not have the problem.

Mixtures (``LateralMixture``): stick-breaking weights f_1 = sigma(l_1), f_2 = (1 - f_1) sigma(l_2), ...; every raw
scale and logit may depend on depth as  raw + s zk + q zk^2, zk = z'/1000 (the legacy convention), evaluated at the
z nodes of the cell quadrature. ``ordered=True`` builds the scales cumulatively in the median radius (r50 of term k
= r50 of term k-1 times exp(softplus(raw_k))) at every depth, which removes label switching between like terms.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .cellint import Grid, log_disc_mass, ring_logmass
from .longitudinal import raw_of, softplus_floor

__all__ = ["LatFamily", "GaussLat", "CauchyLat", "BurrLat", "ExpLat", "WeibullLat", "ExpRegLat", "LAT_FAMILIES",
           "make_lat", "LatTerm", "LateralMixture", "legacy_shower_lateral"]

LOG_PI, LOG_2PI = math.log(math.pi), math.log(2.0 * math.pi)
R_TINY = 1e-30


def _scale(log_v):
    """exp with the exponent clamped to [-12, 35]: widths between 6e-6 mm and 1.6e15 mm (inactive for fitted values;
    it keeps every intermediate of the values *and gradients* below inside float32 for any raw input)."""
    return torch.exp(log_v.clamp(-12.0, 35.0))


def _log_ratio(r, scale):
    """ln(r / scale) without forming the ratio (1e-30 mm / 1e15 mm underflows in float32); r = 0 is taken at 1e-30."""
    return torch.log(r.clamp_min(R_TINY)) - torch.log(scale)


class LatFamily:
    """Base class of a radial family with one scale and optional shape parameters.

    SHAPES = ((local raw name, default raw), ...). ``shapes(raw)`` applies the links. All evaluation methods take
    r (..., N), scale (..., 1) and the dict of natural shape parameters (each (..., 1))."""

    name = ""
    SHAPES: tuple = ()
    closed_form = True

    def shapes(self, raw: dict) -> dict:
        return {}

    def log_sf(self, r, scale, sh):                                           # pragma: no cover - interface
        raise NotImplementedError

    def log_dEdr(self, r, scale, sh):                                         # pragma: no cover - interface
        raise NotImplementedError

    def log_pdf_area(self, r, scale, sh):
        """ln T(r) = ln(dE/dr) - ln(2 pi r)  [1/mm^2]."""
        return self.log_dEdr(r, scale, sh) - LOG_2PI - torch.log(r.clamp_min(R_TINY))

    def log_median(self, sh):
        """ln(r50 / scale)."""
        raise NotImplementedError

    def ring_logmass(self, grid: Grid, scale, sh):
        """ln ring masses (..., N_R) of the plane-normalised term."""
        if self.closed_form:
            return ring_logmass(self.log_sf(grid.r_edges, scale, sh))
        return grid.ring_logmass_quad(lambda r: self.log_dEdr(r, scale, sh))

    def log_disc_mass(self, grid: Grid, scale, sh, ring=None):
        """ln of the mass inside r < R_max (..., 1)."""
        if self.closed_form:
            return log_disc_mass(self.log_sf(grid.r_edges[-1:], scale, sh))
        ring = self.ring_logmass(grid, scale, sh) if ring is None else ring
        return torch.logsumexp(ring, -1, keepdim=True)


class GaussLat(LatFamily):
    """ln S = -r^2 / (2 sigma^2);  T = exp(-r^2 / 2 sigma^2) / (2 pi sigma^2)   (``families.lat_gauss``)."""

    name = "gauss"

    # written as -(r/sigma)^2 / 2: the derivative of r^2 / (2 sigma^2) through 1/(2 sigma^2) overflows float32 for
    # sigma < 1e-5 mm although the result is finite
    def log_sf(self, r, scale, sh):
        return -0.5 * (r / scale) ** 2

    def log_dEdr(self, r, scale, sh):
        return torch.log(r.clamp_min(R_TINY)) - 2 * torch.log(scale) - 0.5 * (r / scale) ** 2

    def log_pdf_area(self, r, scale, sh):
        return -0.5 * (r / scale) ** 2 - LOG_2PI - 2 * torch.log(scale)

    def log_median(self, sh):
        return 0.5 * math.log(2.0 * math.log(2.0))


class CauchyLat(LatFamily):
    """ln S = -log1p(r^2 / R^2);  T = R^2 / (pi (r^2 + R^2)^2)   (``families.lat_cauchy``, the GFlash term).
    S ~ R^2 / r^2: the mean radius exists, the mean square radius does not."""

    name = "cauchy"

    def log_sf(self, r, scale, sh):
        return -torch.log1p((r / scale) ** 2)

    def log_dEdr(self, r, scale, sh):
        return math.log(2.0) + torch.log(r.clamp_min(R_TINY)) + 2 * torch.log(scale) - 2 * torch.log(r * r + scale * scale)

    def log_pdf_area(self, r, scale, sh):
        return 2 * torch.log(scale) - LOG_PI - 2 * torch.log(r * r + scale * scale)

    def log_median(self, sh):
        return 0.0


class BurrLat(LatFamily):
    """Burr XII in r: ln S = -k log1p((r/R)^c) = -k softplus(c ln(r/R)).

    Shapes: rc (c = c_lo + softplus, c_lo = 0.3; omitted when ``c`` is fixed), rk (k = 0.05 + softplus).
    c = 2: 2-D Student-t / Pearson VII body T ~ (1 + r^2/R^2)^(-k-1) (nu = 2k); (c, k) = (2, 1): the Cauchy-like term.
    Tail S ~ (r/R)^(-ck); axis T ~ r^(c-2).
    Not identifiable: k -> infinity at fixed R k^(-1/c) (Weibull limit); ck only from r >> R, which the 700 mm disc
    cuts off for R >~ 150 mm; c and a neighbouring narrower term both shape the first rings (ring 0 ends at 1.9 mm).
    """

    name = "burr"

    def __init__(self, c: Optional[float] = None, c_lo: float = 0.3, init=(2.0, 1.0)):
        self.c_fixed, self.c_lo = c, float(c_lo)
        sh = () if c is not None else (("rc", raw_of(init[0], self.c_lo)),)
        self.SHAPES = sh + (("rk", raw_of(init[1], 0.05)),)
        if c is not None:
            self.name = "student"

    def shapes(self, raw):
        c = self.c_fixed if self.c_fixed is not None else softplus_floor(raw["rc"], self.c_lo)
        return {"c": c, "k": softplus_floor(raw["rk"], 0.05)}

    def log_sf(self, r, scale, sh):
        v = -sh["k"] * F.softplus(sh["c"] * _log_ratio(r, scale))
        return torch.where(r > 0, v, torch.zeros_like(v))                 # S(0) = 1 exactly

    def log_dEdr(self, r, scale, sh):
        y = _log_ratio(r, scale)
        c, k = sh["c"], sh["k"]
        return torch.log(c * k / scale) + (c - 1) * y - (k + 1) * F.softplus(c * y)

    def log_median(self, sh):
        c, k = sh["c"], sh["k"]
        return torch.log(torch.expm1(math.log(2.0) / k)) / c


class ExpLat(LatFamily):
    """ln S = -r/lam: exponential dE/dr, T = exp(-r/lam) / (2 pi lam r). The legacy halo without the regulariser."""

    name = "exp"

    def log_sf(self, r, scale, sh):
        return -r / scale

    def log_dEdr(self, r, scale, sh):
        return -r / scale - torch.log(scale)

    def log_median(self, sh):
        return math.log(math.log(2.0))


class WeibullLat(LatFamily):
    """ln S = -coef (r/lam)^b: Weibull in r (stretched exponential for b < 1).

    coef = 1 (``weibull``): b = 1 is ``exp`` with the same lam.  coef = 1/2 (``weibull_core``): b = 2 is ``gauss``
    with lam = sigma. Shape: rb (b = b_lo + (b_hi - b_lo) sigmoid, default range (0.1, 4); the
    exponent b ln(r/lam) is capped at 80, far beyond any resolvable mass).
    Tail exp(-coef (r/lam)^b); axis T ~ r^(b-2).
    Not identifiable: b -> b_lo with lam -> 0 (ln S ~ -(r/lam)^b flattens to a constant drop: a point mass at the
    axis plus a nearly r-independent remainder); with two Weibull terms of similar b only the mixture is defined
    (use ``ordered``).
    """

    def __init__(self, coef: float = 1.0, b_range=(0.1, 4.0), init: float = 1.0):
        self.coef, self.b_lo, self.b_hi = float(coef), float(b_range[0]), float(b_range[1])
        f = (init - self.b_lo) / (self.b_hi - self.b_lo)
        self.SHAPES = (("rb", math.log(f / (1 - f))),)
        self.name = "weibull" if coef == 1.0 else "weibull_core"

    def shapes(self, raw):
        return {"b": self.b_lo + (self.b_hi - self.b_lo) * torch.sigmoid(raw["rb"])}

    def log_sf(self, r, scale, sh):
        v = -self.coef * torch.exp((sh["b"] * _log_ratio(r, scale)).clamp_max(80.0))
        return torch.where(r > 0, v, torch.zeros_like(v))                 # S(0) = 1 exactly

    def log_dEdr(self, r, scale, sh):
        b = sh["b"]
        y = _log_ratio(r, scale)
        return math.log(self.coef) + torch.log(b / scale) + (b - 1) * y - self.coef * torch.exp((b * y).clamp_max(80.0))

    def log_median(self, sh):
        return math.log(math.log(2.0) / self.coef) / sh["b"]


class ExpRegLat(LatFamily):
    """Legacy regularised halo: T = exp(-r/lam) / (2 pi lam (r + eps))   (``families.lat_exp_over_r``, eps = 1 mm;
    the far halo uses eps = 10 mm).

    dE/dr = [r / (r + eps)] exp(-r/lam) / lam. Its integral over the plane is 1 - (eps/lam) e^(eps/lam) E1(eps/lam)
    (0.947 at lam = 70 mm, 0.979 at lam = 220 mm for eps = 1 mm), so in a mixture the halo fraction is slightly
    smaller than its stick-breaking weight. torch has no exponential integral: ring masses come from
    ``Grid.ring_logmass_quad``.
    """

    name = "exp_reg"
    closed_form = False

    def __init__(self, eps: float = 1.0):
        self.eps = float(eps)

    def log_sf(self, r, scale, sh):
        raise NotImplementedError("exp_reg has no closed-form survival function in torch; use ring_logmass")

    def log_dEdr(self, r, scale, sh):
        return torch.log(r.clamp_min(R_TINY)) - torch.log(r + self.eps) - r / scale - torch.log(scale)

    def log_pdf_area(self, r, scale, sh):
        return -r / scale - LOG_2PI - torch.log(scale) - torch.log(r + self.eps)

    def log_median(self, sh):
        return math.log(math.log(2.0))                                  # of the unregularised form (ordering only)


LAT_FAMILIES = {
    "gauss": GaussLat, "cauchy": CauchyLat, "burr": BurrLat, "student": lambda **kw: BurrLat(c=2.0, **kw),
    "exp": ExpLat, "weibull": lambda **kw: WeibullLat(1.0, **kw),
    "weibull_core": lambda **kw: WeibullLat(0.5, init=2.0, **kw), "exp_reg": ExpRegLat,
}


def make_lat(name, **kw) -> LatFamily:
    if isinstance(name, LatFamily):
        return name
    if name not in LAT_FAMILIES:
        raise KeyError(f"unknown lateral family '{name}'; available: {sorted(LAT_FAMILIES)}")
    return LAT_FAMILIES[name](**kw)


# ----------------------------------------------------------------------------------------------------------------------
class LatTerm:
    """One mixture term: a family, the raw names of its log-scale and of the depth coefficients, a shape prefix.

    scale = (base, slope, quad): ln(scale) = base + slope zk + quad zk^2 with zk = z'/1000; slope / quad may be None.
    ``const`` fixes natural shape values instead of reading them from raw parameters."""

    def __init__(self, family, scale, shape_prefix: str = "", default: float = math.log(50.0)):
        self.family = make_lat(family)
        self.scale = tuple(scale) + (None,) * (3 - len(scale))
        self.shape_prefix, self.default = shape_prefix, float(default)

    @property
    def names(self):
        return [n for n in self.scale if n is not None] + [self.shape_prefix + n for n, _ in self.family.SHAPES]

    @property
    def defaults(self):
        d = {self.scale[0]: self.default}
        d.update({n: 0.0 for n in self.scale[1:] if n is not None})
        d.update({self.shape_prefix + n: float(v) for n, v in self.family.SHAPES})
        return d

    @property
    def depth_dependent(self):
        return self.scale[1] is not None or self.scale[2] is not None


def _poly_z(th, names, zk):
    """raw + s zk + q zk^2 -> (B, Nz, 1) (or (B, 1, 1) when there is no depth term)."""
    base, s, q = names
    out = th[base][:, None, None]
    if s is not None:
        out = out + th[s][:, None, None] * zk[None, :, None]
    if q is not None:
        out = out + th[q][:, None, None] * (zk * zk)[None, :, None]
    return out


class LateralMixture:
    """K-term radial mixture with stick-breaking weights and depth-dependent raw parameters.

    LateralMixture(terms, logits, disc_norm="mixture", ordered=False)

    terms      list of ``LatTerm``
    logits     K - 1 triples (base, slope, quad) of raw names (None where absent): l_k(z) = base + slope zk + quad zk^2
               f_1 = sigma(l_1), f_k = sigma(l_k) prod_{i<k} sigma(-l_i), f_K = prod sigma(-l_i)
    disc_norm  "mixture": the mixture is renormalised on the grid disc r < 700 mm at every depth (legacy
               ``lat_norm="grid"``; f_k are plane fractions);
               "term": every term is renormalised on the disc first (f_k are the fractions inside the disc, literal);
               "none": plane-normalised terms, mass beyond 700 mm is lost depth by depth (legacy track profile).
    ordered    cumulative ordering of the median radii: the first raw scale is ln r50_1, each further raw scale is
               the raw increment d_k with ln r50_k = ln r50_{k-1} + softplus(d_k), at every depth (exact unless a
               scale reaches the clamp of the link, 6e-6 mm or 1.6e15 mm). Not contained in the legacy fit: there
               the Gaussian "core" is the widest term at z' = 600 mm for p >= 2 GeV.
    """

    def __init__(self, terms: Sequence[LatTerm], logits: Sequence = (), disc_norm: str = "mixture",
                 ordered: bool = False):
        self.terms = list(terms)
        self.logits = [tuple(l) + (None,) * (3 - len(l)) for l in logits]
        if len(self.logits) != len(self.terms) - 1:
            raise ValueError("LateralMixture: need K - 1 logit triples for K terms")
        if disc_norm not in ("mixture", "term", "none"):
            raise ValueError("disc_norm must be 'mixture', 'term' or 'none'")
        self.disc_norm, self.ordered = disc_norm, bool(ordered)

    @property
    def names(self):
        out = [n for t in self.terms for n in t.names]
        return out + [n for l in self.logits for n in l if n is not None]

    @property
    def defaults(self):
        d = {}
        for t in self.terms:
            d.update(t.defaults)
        for l in self.logits:
            d.update({n: 0.0 for n in l if n is not None})
        return d

    # ---- parameters at depth ----
    def log_weights(self, th, zk):
        """K log-fractions, each (B, Nz or 1, 1)."""
        out, rest = [], 0.0
        for l in self.logits:
            v = _poly_z(th, l, zk)
            out.append(rest + F.logsigmoid(v))
            rest = rest + F.logsigmoid(-v)
        out.append(rest if torch.is_tensor(rest) else torch.zeros_like(th[self.terms[0].scale[0]])[:, None, None])
        return out

    def scales(self, th, zk):
        """[(scale (B, Nz or 1, 1), shapes dict of (B, 1, 1))] per term."""
        out, prev = [], None
        for t in self.terms:
            sh = t.family.shapes({n: th[t.shape_prefix + n][:, None, None] for n, _ in t.family.SHAPES})
            raw = _poly_z(th, t.scale, zk)
            if self.ordered:
                lmed = raw if prev is None else prev + F.softplus(raw)
                prev = lmed
                raw = lmed - t.family.log_median(sh)
            out.append((_scale(raw), sh))
        return out

    def median_radii(self, th, zk):
        """r50 of every term, each (B, Nz or 1, 1)."""
        return [s * torch.exp(torch.as_tensor(t.family.log_median(sh), dtype=s.dtype, device=s.device))
                for t, (s, sh) in zip(self.terms, self.scales(th, zk))]

    # ---- exact ring masses ----
    def ring_logmass(self, th, zk, grid: Grid):
        """ln of the mixture's ring masses at the depths zk: (B, Nz, N_R)."""
        lw, acc = self.log_weights(th, zk), None
        for t, w, (scale, sh) in zip(self.terms, lw, self.scales(th, zk)):
            m = t.family.ring_logmass(grid, scale, sh)
            if self.disc_norm == "term":                                  # by the sum of its rings: exactly 1
                m = m - torch.logsumexp(m, -1, keepdim=True)
            m = m + w
            acc = m if acc is None else torch.logaddexp(acc, m)
        if self.disc_norm == "mixture":
            acc = acc - torch.logsumexp(acc, -1, keepdim=True)
        return acc

    # ---- cell-centre rule (legacy) ----
    def log_area_density(self, th, zk, grid: Grid):
        """ln T at the ring centres RC for the depths zk: (B, Nz, N_R). With disc_norm != "none" it is normalised
        with the centre rule on the disc, sum_j T_j A_j = 1 (legacy ``lat_norm="grid"``)."""
        lw, acc = self.log_weights(th, zk), None
        for t, w, (scale, sh) in zip(self.terms, lw, self.scales(th, zk)):
            m = t.family.log_pdf_area(grid.rc, scale, sh)
            if self.disc_norm == "term":
                m = m - torch.logsumexp(m + grid.log_area, -1, keepdim=True)
            m = m + w
            acc = m if acc is None else torch.logaddexp(acc, m)
        if self.disc_norm == "mixture":
            acc = acc - torch.logsumexp(acc + grid.log_area, -1, keepdim=True)
        return acc


def legacy_shower_lateral(families=("gauss", "cauchy", "exp_reg"), curv: bool = True, disc_norm: str = "mixture",
                          ordered: bool = False, family_kw: Optional[Sequence[dict]] = None) -> LateralMixture:
    """The shower cross-section with the legacy raw names:
    scales  log_sig + s_sig zk (+ q_sig zk^2), log_R + s_R zk, log_lam + s_lam zk;
    logits  logit_1 + s_1 zk (+ q_1 zk^2), logit_2 + s_2 zk (+ q_2 zk^2);  shape parameters lat0_*, lat1_*, lat2_*.
    ``families`` picks the three radial forms: ("gauss", "cauchy", "exp_reg") is the legacy ``gce``;
    ("gauss", "cauchy", "exp") drops the 1 mm regulariser; ("gauss", "burr", "weibull") frees the body and the halo
    (identical to the previous one at the default shapes c = 2, k = 1, b = 1)."""
    kw = list(family_kw or [{}, {}, {}])
    q = (lambda n: n) if curv else (lambda n: None)
    terms = [LatTerm(make_lat(families[0], **kw[0]), ("log_sig", "s_sig", q("q_sig")), "lat0_", math.log(15.0)),
             LatTerm(make_lat(families[1], **kw[1]), ("log_R", "s_R", None), "lat1_", math.log(60.0)),
             LatTerm(make_lat(families[2], **kw[2]), ("log_lam", "s_lam", None), "lat2_", math.log(110.0))]
    logits = [("logit_1", "s_1", q("q_1")), ("logit_2", "s_2", q("q_2"))]
    return LateralMixture(terms, logits, disc_norm, ordered)
