"""sfm.sr.audit - the two audits of PLAN.md section 7, exact mathematics, no data, NumPy only (no torch).

    rep = audit("tanh((zt * 1.3) + -0.4) * 0.5")           # default: the extended box of operators.extended_box()
    rep.passed, rep.domain["status"], rep.magnitude["status"]

Box (user requirement): z' in [-300, 2000] mm, r in [0, 1500] mm, p from 0.0005 to 50 GeV, i.e. in the search
variables zt in [-1.154, 7.692], rh in [0, 68.49], x in [-4.11, 1.33]. A model that floors x is audited down to its
exact floor: ``audit(..., box=audit_box(x_floor=...))``.

THE AUDIT IS A STATEMENT ABOUT THE EXPRESSION WITH THE CONSTANTS AS GIVEN. A refit moves the constants, so the audit
must be repeated on the refitted equation (``sfm.sr.refit`` does it and refuses to hand over a result otherwise).

(a) Domain safety - SOUND (``domain_safety``)
    Interval arithmetic on every node of the tree. Each elementary enclosure is computed in float64 and rounded
    outward (one ulp for + - * / sqrt, which IEEE rounds correctly; a relative margin of 64 ulp plus one ulp for
    library functions, which are accurate to a few ulp but not correctly rounded), then intersected with the
    operator's mathematical range. By induction the enclosure of every node contains its exact range on the box.
    Restricted operators and their forbidden argument sets:
        /  inv  negative integer power     denominator / base = 0
        log  sqrt  real power              argument / base <= 0   (0 excluded for sqrt: infinite slope)
        log1p                              argument <= -1
    plus a rule that also applies to the pole-free Stage-1 vocabulary: no node may exceed OVERFLOW_CAP = 1e30 in
    magnitude (exp, square, ppow and products can overflow; the refit has no clamps, and Inf - Inf or Inf * 0 is NaN).
    A box is certified when no enclosure touches a forbidden set dilated by
        tau = TAU_REL * max(1, largest magnitude in the argument's subtree)        (TAU_REL = 1e-5)
    - an explicit allowance for evaluation in float32, not part of the rigour. Boxes that are not certified are
    bisected (only along variables the offending node depends on) down to 2^-10 of each side or a box budget.
    Outcomes: "safe" (every leaf certified), "unsafe_proven" (a witness: a sample point inside the dilated forbidden
    set or with an overflow, or a sign change of a denominator between two corners of a box on which the
    denominator itself is continuous - a zero by the intermediate value theorem), "not_proven_safe" (anything
    else: budget, minimum size). Only "safe" passes: over-estimation costs false rejections, never a missed pole.

(b) Magnitude bound - TIGHT (``magnitude_bound``)
    Naive intervals over-estimate (dependency problem), so: (0) a coarse grid - an actual value above the limit
    fails at once; (1) per box the natural enclosure intersected with the centred (mean-value) form
    f(c) + sum_i G_i(B) [-h_i, h_i], G from forward-mode interval derivatives; a box is certified if its bound is
    <= limit; (2) otherwise the centre and corners are evaluated - a value above the limit FAILS (witness kept);
    (3) otherwise the box is bisected along the side with the largest sup|G_i| w_i; (4) at the minimum size
    (2^-10 of a side) or at the box budget the remaining boxes are verified on a dense grid of n^d sub-cell
    centres with the margin sum_i sup|G_i| w_i / (2 n): a value above the limit fails, max + margin <= limit
    certifies, anything else is left "not_refuted" with the interval [sup_lower, sup_upper].
    As the plan says, an equation FAILS (b) only if an actual evaluation exceeds the bound: status "certified" and
    "not_refuted" both pass, "failed" does not. All comparisons use limit * (1 + 1e-9) so that forms bounded by
    exactly 1 (tanh templates) are not failed by the last bit.

    Gauge (decision of this module's author, see STATUS.md section 8; needs the user's confirmation). The softmax
    ignores f -> f + c(x), so "sup |delta| <= 1 nat" is not a property of the model unless a gauge is fixed.
        gauge="oscillation" (default)   H = sup_x (1/2) [ sup_{z,r} f - inf_{z,r} f ] <= limit
                                        = the bound in the most favourable gauge, min over c(x) of sup |f - c(x)|.
                                        Reference-free and gauge-invariant. It is implied by both other readings.
        gauge="reference"               sup | f(z, r, x) - f(z_ref, r_ref, x) | <= limit    (the earlier proposal)
        gauge="none"                    sup | f | <= limit as written (trajectories, gauge-fixed targets)
    Whatever the gauge, the normalised correction delta_eff = f - ln E_base[exp f] obeys |delta_eff| <= 2 H (for
    every baseline), so limit 1 guarantees |delta_eff| <= 2 nat and limit 0.5 guarantees 1 nat.
    The oscillation is computed as a one-sided bound of h = f(z1, r1, x) - f(z2, r2, x) on the product box
    (top-level additive terms that depend on x alone are dropped first: they cancel exactly).

Limits of the statement: real arithmetic with the constants as parsed; library functions assumed accurate to 64 ulp;
the float32 allowance tau is heuristic.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import special as _sp

from . import operators as O
from . import translate as T

__all__ = ["OVERFLOW_CAP", "TAU_REL", "MIN_FRAC", "LIMIT_NAT", "audit_box", "enclose", "domain_safety",
           "magnitude_bound", "audit", "audit_trajectory", "AuditReport", "drop_gauge_terms", "oscillation_expr"]

OVERFLOW_CAP = 1e30          # eight decades below the float32 limit 3.4e38
TAU_REL = 1e-5               # float32 allowance around a forbidden set (relative to the subtree's magnitude)
MIN_FRAC = 2.0 ** -10        # minimum box side as a fraction of the audit box (2.2 mm, 1.5 mm, 0.005 in x)
LIMIT_NAT = 1.0              # PLAN: sup |delta| <= 1 nat
SLACK = 1e-9                 # relative slack of every comparison with the limit
_LIB = 64 * np.finfo(np.float64).eps
_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi
_C_ERF = 2.0 / math.sqrt(math.pi)
_LINV_D = 3.0 * math.sqrt(3.0) / 8.0         # max |linv'| at |x| = 1/sqrt(3)
_INV_SQRT3 = 1.0 / math.sqrt(3.0)
_INF = np.inf

Interval = Tuple[np.ndarray, np.ndarray]


def audit_box(x_floor: Optional[float] = None, **kw) -> Dict[str, Tuple[float, float]]:
    """The extended audit box; with ``x_floor`` the lower x-limit is the model's exact floored x_min instead of
    x(0.0005 GeV) (PLAN: "if a model floors x, down to its exact floored x_min instead")."""
    box = O.extended_box(**kw)
    if x_floor is not None:
        if not (float(x_floor) < box["x"][1]):
            raise ValueError("x_floor must lie below the upper x-limit")
        box["x"] = (float(x_floor), box["x"][1])
    return box


# ----------------------------------------------------------------------------------------------------------------------
# Outward-rounded interval arithmetic on arrays
# ----------------------------------------------------------------------------------------------------------------------
def _dn(a, rel: float = 0.0):
    a = np.asarray(a, dtype=np.float64)
    if rel:
        a = np.where(np.isfinite(a), a - rel * np.abs(a), a)
    return np.nextafter(a, -_INF)


def _up(a, rel: float = 0.0):
    a = np.asarray(a, dtype=np.float64)
    if rel:
        a = np.where(np.isfinite(a), a + rel * np.abs(a), a)
    return np.nextafter(a, _INF)


def _fix(lo, hi) -> Interval:
    """NaN endpoints (Inf - Inf and the like) become the unbounded end."""
    return np.where(np.isnan(lo), -_INF, lo), np.where(np.isnan(hi), _INF, hi)


def _add(a: Interval, b: Interval) -> Interval:
    return _fix(_dn(a[0] + b[0]), _up(a[1] + b[1]))


def _sub(a: Interval, b: Interval) -> Interval:
    return _fix(_dn(a[0] - b[1]), _up(a[1] - b[0]))


def _neg(a: Interval) -> Interval:
    return -a[1], -a[0]


def _mul(a: Interval, b: Interval) -> Interval:
    """Interval product with 0 * Inf = 0 (intervals are sets of reals: IEEE 1788)."""
    ps = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    ps = [np.where(np.isnan(p), 0.0, p) for p in ps]
    lo = np.minimum(np.minimum(ps[0], ps[1]), np.minimum(ps[2], ps[3]))
    hi = np.maximum(np.maximum(ps[0], ps[1]), np.maximum(ps[2], ps[3]))
    return _dn(lo), _up(hi)


def _scale(a: Interval, c: float) -> Interval:
    c = float(c)
    lo, hi = (a[0] * c, a[1] * c) if c >= 0 else (a[1] * c, a[0] * c)
    lo, hi = np.where(np.isnan(lo), 0.0, lo), np.where(np.isnan(hi), 0.0, hi)
    return _dn(lo), _up(hi)


def _abs(a: Interval) -> Interval:
    al, ah = np.abs(a[0]), np.abs(a[1])
    cross = (a[0] <= 0) & (a[1] >= 0)
    return np.where(cross, 0.0, np.minimum(al, ah)), np.maximum(al, ah)


def _square(a: Interval) -> Interval:
    b = _abs(a)
    return np.maximum(_dn(b[0] * b[0]), 0.0), _up(b[1] * b[1])


def _recip(a: Interval) -> Interval:
    """1 / a for an interval that does not contain 0 (the caller has checked); garbage otherwise."""
    return _fix(_dn(1.0 / a[1]), _up(1.0 / a[0]))


def _mono(f, a: Interval, increasing: bool = True, rel: float = _LIB) -> Interval:
    lo, hi = (f(a[0]), f(a[1])) if increasing else (f(a[1]), f(a[0]))
    return _fix(_dn(lo, rel), _up(hi, rel))


def _clip(v: Interval, lo: float, hi: float) -> Interval:
    return np.clip(v[0], lo, hi), np.clip(v[1], lo, hi)


def _trig(a: Interval, phase: float) -> Interval:
    """Enclosure of sin(a + phase) for phase in {0, pi/2} (sin, cos)."""
    f = np.sin if phase == 0.0 else np.cos
    s0, s1 = f(a[0]), f(a[1])
    lo, hi = _dn(np.minimum(s0, s1), _LIB), _up(np.maximum(s0, s1), _LIB)
    big = ~(np.isfinite(a[0]) & np.isfinite(a[1])) | ((a[1] - a[0]) >= _TWO_PI) | \
        (np.maximum(np.abs(a[0]), np.abs(a[1])) > 1e6)
    al = np.where(big, 0.0, a[0])
    ah = np.where(big, 0.0, a[1])
    slack = 1e-9 * (1.0 + np.abs(ah))
    tmax = _HALF_PI - phase                              # maxima at tmax + 2 pi k, minima at tmax - pi + 2 pi k
    kmax = np.ceil((al - tmax) / _TWO_PI - 1e-9)
    kmin = np.ceil((al - (tmax - math.pi)) / _TWO_PI - 1e-9)
    has_max = (tmax + _TWO_PI * kmax) <= ah + slack
    has_min = ((tmax - math.pi) + _TWO_PI * kmin) <= ah + slack
    hi = np.where(big | has_max, 1.0, hi)
    lo = np.where(big | has_min, -1.0, lo)
    return np.clip(lo, -1.0, 1.0), np.clip(hi, -1.0, 1.0)


def _softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _linv_val(a: Interval) -> Interval:
    s = _square(a)
    return _clip(_fix(_dn(1.0 / (1.0 + s[1]), _LIB), _up(1.0 / (1.0 + s[0]), _LIB)), 0.0, 1.0)


def _linv_der(a: Interval) -> Interval:
    def h(x):
        with np.errstate(all="ignore"):
            v = -2.0 * x / np.square(1.0 + x * x)
        return np.where(np.isfinite(v), v, 0.0)          # the limit at +-Inf (and after overflow of x^4) is 0
    h0, h1 = h(a[0]), h(a[1])
    lo, hi = _dn(np.minimum(h0, h1), _LIB), _up(np.maximum(h0, h1), _LIB)
    hi = np.where((a[0] <= -_INV_SQRT3) & (a[1] >= -_INV_SQRT3), _up(_LINV_D, _LIB), hi)
    lo = np.where((a[0] <= _INV_SQRT3) & (a[1] >= _INV_SQRT3), _dn(-_LINV_D, _LIB), lo)
    return lo, hi


def _l1p_der(a: Interval) -> Interval:
    pos, neg_ = a[0] >= 0, a[1] <= 0
    lo = np.where(pos, 1.0 / (1.0 + np.abs(a[1])), np.where(neg_, -1.0 / (1.0 + np.abs(a[1])), -1.0))
    hi = np.where(pos, 1.0 / (1.0 + np.abs(a[0])), np.where(neg_, -1.0 / (1.0 + np.abs(a[0])), 1.0))
    return _clip((_dn(lo, _LIB), _up(hi, _LIB)), -1.0, 1.0)


def _sqa_der(a: Interval) -> Interval:
    pos, neg_ = a[0] > 0, a[1] < 0
    r0, r1 = 0.5 / np.sqrt(np.abs(a[0])), 0.5 / np.sqrt(np.abs(a[1]))
    lo = np.where(pos, r1, np.where(neg_, -r1, -_INF))
    hi = np.where(pos, r0, np.where(neg_, -r0, _INF))
    return _fix(_dn(lo, _LIB), _up(hi, _LIB))


def _ipow(a: Interval, n: int) -> Interval:
    """a^n for an integer n >= 0."""
    if n == 0:
        one = np.ones_like(a[0])
        return one, one.copy()
    if n == 1:
        return a
    if n % 2 == 0:
        b = _abs(a)
        lo, hi = _fix(_dn(np.power(b[0], n), _LIB), _up(np.power(b[1], n), _LIB))
        return np.maximum(lo, 0.0), hi
    return _fix(_dn(np.power(a[0], n), _LIB), _up(np.power(a[1], n), _LIB))


def _corner_pow(b: Interval, y: Interval) -> Interval:
    """b^y for b > 0: monotone in each argument separately, so the range is spanned by the four corners."""
    cs = [np.power(b[i], y[j]) for i in (0, 1) for j in (0, 1)]
    lo = np.minimum(np.minimum(cs[0], cs[1]), np.minimum(cs[2], cs[3]))
    hi = np.maximum(np.maximum(cs[0], cs[1]), np.maximum(cs[2], cs[3]))
    lo, hi = _fix(_dn(lo, _LIB), _up(hi, _LIB))
    return np.maximum(lo, 0.0), hi


# value enclosure f(a), derivative enclosure d(a, v) for the total unary operators
_UNARY_VAL = {
    "exp": lambda a: (np.maximum(_mono(np.exp, a)[0], 0.0), _mono(np.exp, a)[1]),
    "square": _square,
    "tanh": lambda a: _clip(_mono(np.tanh, a), -1.0, 1.0),
    "atan": lambda a: _clip(_mono(np.arctan, a), -_HALF_PI, _HALF_PI),
    "erf": lambda a: _clip(_mono(_sp.erf, a), -1.0, 1.0),
    "erfc": lambda a: _clip(_mono(_sp.erfc, a, increasing=False), 0.0, 2.0),
    "softplus": lambda a: (np.maximum(_mono(_softplus, a)[0], 0.0), _mono(_softplus, a)[1]),
    "linv": _linv_val,
    "l1p": lambda a: (np.maximum(_mono(np.log1p, _abs(a))[0], 0.0), _mono(np.log1p, _abs(a))[1]),
    "sqa": lambda a: (np.maximum(_mono(np.sqrt, _abs(a), rel=0.0)[0], 0.0), _mono(np.sqrt, _abs(a), rel=0.0)[1]),
    "sin": lambda a: _trig(a, 0.0),
    "cos": lambda a: _trig(a, _HALF_PI),
    "neg": _neg,
    "abs": _abs,
    "cube": lambda a: _mono(lambda x: x * x * x, a),
}


def _tanh_der(a, v):
    s = _square(v)
    return _clip((_dn(1.0 - s[1]), _up(1.0 - s[0])), 0.0, 1.0)


def _erf_der(a, v):
    s = _square(a)
    return np.maximum(_dn(_C_ERF * np.exp(-s[1]), _LIB), 0.0), _up(_C_ERF * np.exp(-s[0]), _LIB)


def _abs_der(a, v):
    return np.where(a[0] > 0, 1.0, -1.0), np.where(a[1] < 0, -1.0, 1.0)


_UNARY_DER = {
    "exp": lambda a, v: v,
    "square": lambda a, v: _fix(2.0 * a[0], 2.0 * a[1]),
    "tanh": _tanh_der,
    "atan": lambda a, v: _linv_val(a),
    "erf": _erf_der,
    "erfc": lambda a, v: _neg(_erf_der(a, v)),
    "softplus": lambda a, v: _clip(_mono(_sp.expit, a), 0.0, 1.0),
    "linv": lambda a, v: _linv_der(a),
    "l1p": lambda a, v: _l1p_der(a),
    "sqa": lambda a, v: _sqa_der(a),
    "sin": lambda a, v: _trig(a, _HALF_PI),
    "cos": lambda a, v: _neg(_trig(a, 0.0)),
    "neg": lambda a, v: (-np.ones_like(a[0]), -np.ones_like(a[0])),
    "abs": _abs_der,
    "cube": lambda a, v: _scale(_square(a), 3.0),
}


# ----------------------------------------------------------------------------------------------------------------------
# Compiled tree and its interval evaluation
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class _Node:
    kind: str
    name: str
    ch: Tuple[int, ...]
    value: Optional[float]
    fixed: bool
    deps: frozenset
    text: str


class _Prog:
    def __init__(self, expr: T.Expr, names: Sequence[str]):
        self.names = list(names)
        self.nodes: List[_Node] = []
        self.root = self._build(expr)

    def _build(self, e: T.Expr) -> int:
        if e.kind not in ("const", "var", "op"):
            raise ValueError(f"cannot audit a {e.kind!r} node ({e.name}): translate templates / parameters first")
        ch = tuple(self._build(a) for a in e.args)
        if e.kind == "var":
            if e.name not in self.names:
                raise KeyError(f"variable {e.name!r} is not in the audit box {self.names}")
            deps = frozenset([self.names.index(e.name)])
        else:
            deps = frozenset().union(*[self.nodes[c].deps for c in ch]) if ch else frozenset()
        self.nodes.append(_Node(e.kind, e.name, ch, e.value, e.fixed, deps, T.to_string(e)[:160]))
        return len(self.nodes) - 1


@dataclass
class _Run:
    V: list
    G: Optional[list]
    M: list
    flags: list            # (node index, reason, mask (N,), index of the offending argument node or -1)


def _gmul(d: Interval, g: Interval) -> Interval:
    return _mul((d[0][:, None], d[1][:, None]), g)


def _run(prog: _Prog, lo: np.ndarray, hi: np.ndarray, grad: bool = False, tau_rel: float = TAU_REL,
         cap: float = OVERFLOW_CAP) -> _Run:
    """Enclosures of every node on N boxes ``[lo, hi]`` (N, d); with ``grad`` also of the partial derivatives."""
    N, d = lo.shape
    V: list = []
    G: Optional[list] = [] if grad else None
    M: list = []
    flags: list = []
    zeros = np.zeros((N, d))
    with np.errstate(all="ignore"):
        for idx, nd in enumerate(prog.nodes):
            g = None
            if nd.kind == "const":
                c = float(nd.value)
                v = (np.full(N, c), np.full(N, c))
                if not math.isfinite(c):
                    flags.append((idx, "non-finite constant", np.ones(N, bool), -1))
                    v = (np.full(N, -_INF), np.full(N, _INF))
                if grad:
                    g = (zeros, zeros)
            elif nd.kind == "var":
                j = prog.names.index(nd.name)
                v = (lo[:, j].copy(), hi[:, j].copy())
                if grad:
                    u = np.zeros((N, d))
                    u[:, j] = 1.0
                    g = (u, u)
            else:
                name = nd.name
                a = V[nd.ch[0]]
                ga = G[nd.ch[0]] if grad else None
                b = V[nd.ch[1]] if len(nd.ch) > 1 else None
                gb = G[nd.ch[1]] if (grad and len(nd.ch) > 1) else None
                bad = None
                bad_arg = -1
                if name in _UNARY_VAL:
                    v = _UNARY_VAL[name](a)
                    if grad:
                        g = _gmul(_UNARY_DER[name](a, v), ga)
                elif name == "+":
                    v = _add(a, b)
                    if grad:
                        g = _add(ga, gb)
                elif name == "-":
                    v = _sub(a, b)
                    if grad:
                        g = _sub(ga, gb)
                elif name == "*":
                    v = _mul(a, b)
                    if grad:
                        g = _add(_gmul(a, gb), _gmul(b, ga))
                elif name in ("/", "inv"):
                    den, gden, den_idx = (b, gb, nd.ch[1]) if name == "/" else (a, ga, nd.ch[0])
                    tau = tau_rel * np.maximum(1.0, M[den_idx])
                    bad, bad_arg = (den[0] <= tau) & (den[1] >= -tau), den_idx
                    r = _recip(den)
                    v = _mul(a, r) if name == "/" else r
                    if grad:
                        if name == "/":
                            g = _gmul(r, _sub(ga, _gmul(v, gden)))
                        else:
                            g = _gmul(_neg(_square(v)), gden)
                elif name in ("log", "log1p", "sqrt"):
                    tau = tau_rel * np.maximum(1.0, M[nd.ch[0]])
                    edge = -1.0 if name == "log1p" else 0.0
                    bad, bad_arg = a[0] <= edge + tau, nd.ch[0]
                    f = {"log": np.log, "log1p": np.log1p, "sqrt": np.sqrt}[name]
                    v = _mono(f, a, rel=0.0 if name == "sqrt" else _LIB)
                    if grad:
                        if name == "log":
                            dd = _recip(a)
                        elif name == "log1p":
                            dd = _recip(_add(a, (np.ones(N), np.ones(N))))
                        else:
                            dd = _recip(_scale(v, 2.0))
                        g = _gmul(dd, ga)
                elif name == "ppow":
                    base = _abs(a)
                    base = (np.maximum(_dn(1.0 + base[0]), 1.0), _up(1.0 + base[1]))
                    v = _corner_pow(base, b)
                    if grad:
                        l1 = _UNARY_VAL["l1p"](a)
                        dx = _mul(_mul(b, v), _l1p_der(a))
                        g = _add(_gmul(dx, ga), _gmul(_mul(v, l1), gb))
                elif name == "pow":
                    ex = prog.nodes[nd.ch[1]]
                    if ex.kind == "const" and ex.fixed and float(ex.value).is_integer():
                        n = int(ex.value)
                        if n >= 0:
                            v = _ipow(a, n)
                            if grad:
                                g = _gmul(_scale(_ipow(a, n - 1), float(n)), ga) if n >= 1 else (zeros, zeros)
                        else:
                            tau = tau_rel * np.maximum(1.0, M[nd.ch[0]])
                            bad, bad_arg = (a[0] <= tau) & (a[1] >= -tau), nd.ch[0]
                            v = _recip(_ipow(a, -n))
                            if grad:
                                g = _gmul(_scale(_recip(_ipow(a, -n + 1)), float(n)), ga)
                    else:
                        tau = tau_rel * np.maximum(1.0, M[nd.ch[0]])
                        bad, bad_arg = a[0] <= tau, nd.ch[0]
                        apos = (np.maximum(a[0], np.finfo(float).tiny), np.maximum(a[1], np.finfo(float).tiny))
                        v = _corner_pow(apos, b)
                        if grad:
                            lg = _mono(np.log, apos)
                            g = _gmul(v, _add(_gmul(lg, gb), _gmul(_mul(b, _recip(apos)), ga)))
                else:
                    raise ValueError(f"the audit has no enclosure for operator {name!r}")
                if bad is not None:
                    if bad.any():
                        flags.append((idx, f"{name}: argument can reach its forbidden set", bad, bad_arg))
                        v = (np.where(bad, -_INF, v[0]), np.where(bad, _INF, v[1]))
                        if grad:
                            g = (np.where(bad[:, None], -_INF, g[0]), np.where(bad[:, None], _INF, g[1]))
                r_lo, r_hi = O.OPS[name].range
                v = (np.maximum(v[0], r_lo), np.minimum(v[1], r_hi)) if name != "pow" else v
                v = _fix(*v)
            mag = np.maximum(np.abs(v[0]), np.abs(v[1]))
            sub = mag
            for c in nd.ch:
                sub = np.maximum(sub, M[c])
            if nd.kind == "op":
                over = mag > cap
                if over.any():
                    flags.append((idx, f"{nd.name}: magnitude can exceed {cap:g} (overflow)", over, -1))
            V.append(v)
            M.append(sub)
            if grad:
                G.append(_fix(*g))
    return _Run(V, G, M, flags)


def _resolve(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]]):
    e = T.parse(expr) if isinstance(expr, str) else expr
    box = dict(O.extended_box() if box is None else box)
    used = T.variables(e)
    missing = [v for v in used if v not in box]
    if missing:
        raise KeyError(f"variable(s) {missing} have no range in the audit box {sorted(box)}")
    names = [n for n in box if n in used]
    lo = np.array([[float(box[n][0]) for n in names]], dtype=np.float64)
    hi = np.array([[float(box[n][1]) for n in names]], dtype=np.float64)
    if np.any(hi < lo) or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("the audit box needs finite limits with lo <= hi")
    return e, names, lo, hi


def enclose(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None,
            grad: bool = False) -> dict:
    """Natural interval enclosure of the expression on one box: {"lo", "hi", "certified" (no domain or overflow
    flag), "flags", and with ``grad`` "grad_lo"/"grad_hi" per variable}. Sound: every value of the expression on
    the box lies in [lo, hi] (if "certified"; otherwise the enclosure may be the whole line)."""
    e, names, lo, hi = _resolve(expr, box)
    prog = _Prog(e, names)
    out = _run(prog, lo, hi, grad=grad)
    res = {"lo": float(out.V[prog.root][0][0]), "hi": float(out.V[prog.root][1][0]), "certified": not out.flags,
           "flags": [{"node": prog.nodes[i].text, "reason": r} for i, r, m, _ in out.flags], "variables": names}
    if grad:
        res["grad_lo"] = {n: float(out.G[prog.root][0][0, j]) for j, n in enumerate(names)}
        res["grad_hi"] = {n: float(out.G[prog.root][1][0, j]) for j, n in enumerate(names)}
    return res


def _corners(lo: np.ndarray, hi: np.ndarray, centre: bool = True) -> np.ndarray:
    """(N, P, d) corner points of N boxes (plus the centre)."""
    d = lo.shape[1]
    if d == 0:
        return np.zeros((len(lo), 1, 0))
    pat = np.array(list(itertools.product((0.0, 1.0), repeat=d)))
    if centre:
        pat = np.concatenate([pat, np.full((1, d), 0.5)])
    return lo[:, None, :] + pat[None, :, :] * (hi - lo)[:, None, :]


# ----------------------------------------------------------------------------------------------------------------------
# (a) Domain safety
# ----------------------------------------------------------------------------------------------------------------------
def domain_safety(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None, *,
                  min_frac: float = MIN_FRAC, max_boxes: int = 200_000, tau_rel: float = TAU_REL,
                  cap: float = OVERFLOW_CAP) -> dict:
    """Check (a). Returns a dict with ``status`` in {"safe", "unsafe_proven", "not_proven_safe"}, ``safe`` (bool,
    True only for "safe"), ``witness`` / ``unresolved`` details and the box counts. See the module docstring."""
    t0 = time.time()
    e, names, lo, hi = _resolve(expr, box)
    prog = _Prog(e, names)
    d = len(names)
    root_w = np.where(hi[0] > lo[0], hi[0] - lo[0], 1.0)
    res = {"status": "safe", "safe": True, "variables": names, "n_boxes": 0, "n_certified": 0, "witness": None,
           "unresolved": [], "restricted_nodes": [nd.text for nd in prog.nodes if nd.kind == "op" and
                                                  not O.OPS[nd.name].total],
           "tau_rel": tau_rel, "overflow_cap": cap, "min_frac": min_frac, "max_boxes": max_boxes}
    qlo, qhi = lo, hi
    budget_hit = False
    while len(qlo):
        nlo, nhi = [], []
        for s in range(0, len(qlo), 4096):
            blo, bhi = qlo[s:s + 4096], qhi[s:s + 4096]
            out = _run(prog, blo, bhi, tau_rel=tau_rel, cap=cap)
            res["n_boxes"] += len(blo)
            flagged = np.zeros(len(blo), bool)
            depmask = np.zeros((len(blo), d), bool)
            for i, _, m, _ in out.flags:
                flagged |= m
                deps = sorted(prog.nodes[i].deps)
                if deps:
                    depmask[np.ix_(m, deps)] = True
            res["n_certified"] += int((~flagged).sum())
            if not flagged.any():
                continue
            fi = np.flatnonzero(flagged)
            # --- proof attempts on the flagged boxes: thin evaluations at the corners and the centre ---------------
            pts = _corners(blo[fi], bhi[fi])
            P = pts.shape[1]
            flat = pts.reshape(-1, d)
            thin = _run(prog, flat, flat, tau_rel=tau_rel, cap=cap)
            for i, reason, m, _ in thin.flags:
                if m.any():
                    j = int(np.flatnonzero(m)[0])
                    res.update(status="unsafe_proven", safe=False,
                               witness={"kind": "point", "node": prog.nodes[i].text, "reason": reason,
                                        "point": {n: float(flat[j, k]) for k, n in enumerate(names)}})
                    res["seconds"] = round(time.time() - t0, 3)
                    return res
            for i, reason, m, arg in out.flags:
                if arg < 0 or prog.nodes[i].name not in ("/", "inv", "pow"):
                    continue
                # a denominator that is continuous on the box (no flag inside its own subtree) and changes sign
                inner = np.zeros(len(blo), bool)
                for i2, _, m2, _ in out.flags:
                    if i2 != i and i2 <= arg and _in_subtree(prog, arg, i2):
                        inner |= m2
                vlo = thin.V[arg][0].reshape(len(fi), P)
                vhi = thin.V[arg][1].reshape(len(fi), P)
                change = (vhi < 0).any(1) & (vlo > 0).any(1) & m[fi] & ~inner[fi]
                if change.any():
                    j = int(np.flatnonzero(change)[0])
                    res.update(status="unsafe_proven", safe=False,
                               witness={"kind": "sign change of a denominator (intermediate value theorem)",
                                        "node": prog.nodes[i].text, "denominator": prog.nodes[arg].text,
                                        "box": {n: (float(blo[fi[j], k]), float(bhi[fi[j], k]))
                                                for k, n in enumerate(names)}})
                    res["seconds"] = round(time.time() - t0, 3)
                    return res
            # --- bisect along the widest relative side among the variables the offending nodes depend on ------------
            relw = (bhi[fi] - blo[fi]) / root_w
            score = np.where(depmask[fi], relw, -1.0)
            dim = np.argmax(score, 1) if d else np.zeros(len(fi), int)
            can = (score[np.arange(len(fi)), dim] > min_frac) if d else np.zeros(len(fi), bool)
            for j in np.flatnonzero(~can)[:5]:
                if len(res["unresolved"]) < 5:
                    res["unresolved"].append({n: (float(blo[fi[j], k]), float(bhi[fi[j], k]))
                                              for k, n in enumerate(names)})
            if (~can).any():
                res.update(status="not_proven_safe", safe=False)
            fi, dim = fi[can], dim[can]
            if len(fi):
                mid = 0.5 * (blo[fi, dim] + bhi[fi, dim])
                l1, h1, l2, h2 = blo[fi].copy(), bhi[fi].copy(), blo[fi].copy(), bhi[fi].copy()
                h1[np.arange(len(fi)), dim] = mid
                l2[np.arange(len(fi)), dim] = mid
                nlo += [l1, l2]
                nhi += [h1, h2]
        if not nlo:
            break
        qlo, qhi = np.concatenate(nlo), np.concatenate(nhi)
        if res["n_boxes"] + len(qlo) > max_boxes:
            budget_hit = True
            break
    if budget_hit:
        res.update(status="not_proven_safe", safe=False, budget_exhausted=True)
        if len(res["unresolved"]) < 5:
            res["unresolved"].append({n: (float(qlo[0, k]), float(qhi[0, k])) for k, n in enumerate(names)})
    res["seconds"] = round(time.time() - t0, 3)
    return res


def _in_subtree(prog: _Prog, root: int, idx: int) -> bool:
    if idx == root:
        return True
    return any(_in_subtree(prog, c, idx) for c in prog.nodes[root].ch)


# ----------------------------------------------------------------------------------------------------------------------
# (b) Magnitude bound
# ----------------------------------------------------------------------------------------------------------------------
def _split_sum(e: T.Expr, sign: int = 1) -> List[Tuple[int, T.Expr]]:
    if e.kind == "op" and e.name == "+":
        return _split_sum(e.args[0], sign) + _split_sum(e.args[1], sign)
    if e.kind == "op" and e.name == "-":
        return _split_sum(e.args[0], sign) + _split_sum(e.args[1], -sign)
    if e.kind == "op" and e.name == "neg":
        return _split_sum(e.args[0], -sign)
    return [(sign, e)]


def drop_gauge_terms(e: T.Expr, gauge_vars: Sequence[str] = ("x",)) -> Tuple[Optional[T.Expr], List[str]]:
    """Remove the top-level additive terms that depend on the gauge variables alone (constants included): the
    softmax cannot see them. Returns (remaining expression or None if nothing remains, dropped terms as text)."""
    gv = set(gauge_vars)
    keep, dropped = [], []
    for s, t in _split_sum(e):
        if set(T.variables(t)) <= gv:
            dropped.append(("-" if s < 0 else "+") + " " + T.to_string(t))
        else:
            keep.append((s, t))
    if not keep:
        return None, dropped
    s0, acc = keep[0]
    if s0 < 0:
        acc = T.op("neg", acc)
    for s, t in keep[1:]:
        acc = T.op("+" if s > 0 else "-", acc, t)
    return acc, dropped


def oscillation_expr(e: T.Expr, spatial: Sequence[str]) -> Tuple[T.Expr, Dict[str, str], Dict[str, str]]:
    """h = f(spatial -> copy a) - f(spatial -> copy b): sup h over the product box is the largest oscillation of f
    over the spatial variables at fixed gauge variables."""
    a = {s: f"{s}__a" for s in spatial}
    b = {s: f"{s}__b" for s in spatial}
    fa = T.substitute(e, {s: T.var(n) for s, n in a.items()})
    fb = T.substitute(e, {s: T.var(n) for s, n in b.items()})
    return T.op("-", fa, fb), a, b


def _values(e: T.Expr, names: Sequence[str], pts: np.ndarray) -> np.ndarray:
    """Actual float64 evaluations of the expression at points (P, d)."""
    if not len(names):
        return np.full(len(pts), float(T.evaluate(e, {})))
    return np.asarray(T.evaluate(e, {n: pts[:, j] for j, n in enumerate(names)}), dtype=np.float64) + np.zeros(len(pts))


def _bnb(e: T.Expr, names: List[str], lo: np.ndarray, hi: np.ndarray, limit: float, two_sided: bool, *,
         min_frac: float, max_boxes: int, dense_points: int, root_w: Optional[np.ndarray] = None) -> dict:
    """Branch and bound for sup e <= limit (or sup |e| <= limit). See the module docstring, steps (1)-(4).
    ``root_w``: the side lengths that ``min_frac`` refers to (default: those of the box given)."""
    prog = _Prog(e, names)
    d = len(names)
    lim = limit * (1.0 + SLACK)
    root_w = np.where(hi[0] > lo[0], hi[0] - lo[0], 1.0) if root_w is None else np.asarray(root_w, dtype=np.float64)
    st = {"status": "certified", "sup_lower": -_INF, "sup_upper": -_INF, "witness": None, "n_boxes": 0,
          "n_certified": 0, "n_dense_boxes": 0, "n_not_certified": 0, "not_certified": [], "max_depth": 0}

    def score_of(v):
        return np.abs(v) if two_sided else v

    def fail(pt, val):
        st.update(status="failed", witness={"point": {n: float(pt[k]) for k, n in enumerate(names)},
                                            "value": float(val)})
        st["sup_lower"] = max(st["sup_lower"], float(score_of(val)) if np.isfinite(val) else _INF)
        st["sup_upper"] = max(st["sup_upper"], st["sup_lower"])
        return st

    qlo, qhi = lo, hi
    leaves_lo, leaves_hi, leaves_ub, leaves_g = [], [], [], []
    depth = 0
    while len(qlo):
        nlo, nhi = [], []
        for s in range(0, len(qlo), 2048):
            blo, bhi = qlo[s:s + 2048], qhi[s:s + 2048]
            n = len(blo)
            st["n_boxes"] += n
            out = _run(prog, blo, bhi, grad=True)
            v = out.V[prog.root]
            ctr = 0.5 * (blo + bhi)
            thin = _run(prog, ctr, ctr)
            h = 0.5 * (bhi - blo)
            gmax = np.maximum(np.abs(out.G[prog.root][0]), np.abs(out.G[prog.root][1]))
            with np.errstate(all="ignore"):
                contrib = np.where(h > 0, gmax * h, 0.0)
            rad = _up(contrib.sum(1), 4 * np.finfo(float).eps * max(d, 1))
            c_lo, c_hi = _fix(_dn(thin.V[prog.root][0] - rad), _up(thin.V[prog.root][1] + rad))
            e_lo, e_hi = np.maximum(v[0], c_lo), np.minimum(v[1], c_hi)
            ub = np.maximum(e_hi, -e_lo) if two_sided else e_hi
            ub = np.where(np.isnan(ub), _INF, ub)
            cert = ub <= lim
            # the thin enclosure of f(centre) is a rigorous lower bound of an attained value
            low_c = np.maximum(thin.V[prog.root][0], -thin.V[prog.root][1]) if two_sided else thin.V[prog.root][0]
            low_c = low_c[np.isfinite(low_c)]
            if len(low_c):
                st["sup_lower"] = max(st["sup_lower"], float(low_c.max()))
            st["n_certified"] += int(cert.sum())
            if cert.any():
                st["sup_upper"] = max(st["sup_upper"], float(ub[cert].max()))
            ri = np.flatnonzero(~cert)
            if not len(ri):
                continue
            pts = _corners(blo[ri], bhi[ri])
            flat = pts.reshape(-1, d)
            vals = _values(e, names, flat)
            sc = score_of(vals)
            badv = ~np.isfinite(vals) | (sc > lim)
            fin = np.isfinite(sc)
            if fin.any():
                st["sup_lower"] = max(st["sup_lower"], float(sc[fin].max()))
            if badv.any():
                j = int(np.flatnonzero(badv)[np.argmax(np.where(np.isfinite(sc[badv]), sc[badv], _INF))])
                return fail(flat[j], vals[j])
            relw = (bhi[ri] - blo[ri]) / root_w
            score = np.where(relw > min_frac, np.where(np.isnan(contrib[ri]), _INF, contrib[ri]), -1.0)
            score = np.where((score == 0.0) & (relw > min_frac), relw * 1e-300, score)
            dim = np.argmax(score, 1) if d else np.zeros(len(ri), int)
            can = (score[np.arange(len(ri)), dim] > 0) if d else np.zeros(len(ri), bool)
            leaf = ri[~can]
            if len(leaf):
                leaves_lo.append(blo[leaf]); leaves_hi.append(bhi[leaf]); leaves_ub.append(ub[leaf])
                leaves_g.append(gmax[leaf])
            ri, dim = ri[can], dim[can]
            if len(ri):
                mid = 0.5 * (blo[ri, dim] + bhi[ri, dim])
                l1, h1, l2, h2 = blo[ri].copy(), bhi[ri].copy(), blo[ri].copy(), bhi[ri].copy()
                h1[np.arange(len(ri)), dim] = mid
                l2[np.arange(len(ri)), dim] = mid
                nlo += [l1, l2]
                nhi += [h1, h2]
        if not nlo:
            break
        qlo, qhi = np.concatenate(nlo), np.concatenate(nhi)
        depth += 1
        st["max_depth"] = depth
        if st["n_boxes"] + len(qlo) > max_boxes:
            # budget: the pending boxes go to the dense verification with a freshly computed gradient enclosure
            for s in range(0, len(qlo), 2048):
                blo, bhi = qlo[s:s + 2048], qhi[s:s + 2048]
                out = _run(prog, blo, bhi, grad=True)
                v = out.V[prog.root]
                ubq = np.maximum(v[1], -v[0]) if two_sided else v[1]
                leaves_lo.append(blo); leaves_hi.append(bhi); leaves_ub.append(np.where(np.isnan(ubq), _INF, ubq))
                leaves_g.append(np.maximum(np.abs(out.G[prog.root][0]), np.abs(out.G[prog.root][1])))
            st["budget_exhausted"] = True
            break
    if leaves_lo:
        L_lo, L_hi = np.concatenate(leaves_lo), np.concatenate(leaves_hi)
        L_ub, L_g = np.concatenate(leaves_ub), np.concatenate(leaves_g)
        nper = max(2, int(math.floor(dense_points ** (1.0 / max(d, 1)) + 1e-9)))
        while len(L_lo) * nper ** d > 3e7 and nper > 2:
            nper -= 1
        offs = np.array(list(itertools.product((np.arange(nper) + 0.5) / nper, repeat=d))) if d else np.zeros((1, 0))
        with np.errstate(all="ignore"):
            margin = np.where((L_hi - L_lo) > 0, L_g * (L_hi - L_lo) / (2.0 * nper), 0.0).sum(1)
        margin = np.where(np.isnan(margin), _INF, margin)
        chunk = max(1, int(2e6 // len(offs)))
        st["n_dense_boxes"] = int(len(L_lo))
        st["dense_points_per_box"] = int(len(offs))
        for s in range(0, len(L_lo), chunk):
            blo, bhi = L_lo[s:s + chunk], L_hi[s:s + chunk]
            pts = (blo[:, None, :] + offs[None, :, :] * (bhi - blo)[:, None, :]).reshape(-1, d)
            vals = _values(e, names, pts)
            sc = score_of(vals)
            badv = ~np.isfinite(vals) | (sc > lim)
            if badv.any():
                j = int(np.flatnonzero(badv)[np.argmax(np.where(np.isfinite(sc[badv]), sc[badv], _INF))])
                return fail(pts[j], vals[j])
            mx = sc.reshape(len(blo), -1).max(1)
            st["sup_lower"] = max(st["sup_lower"], float(mx.max()))
            ubd = np.minimum(L_ub[s:s + chunk], mx + margin[s:s + chunk] * (1.0 + 1e-12))
            cert = ubd <= lim
            st["n_certified"] += int(cert.sum())
            st["n_not_certified"] += int((~cert).sum())
            st["sup_upper"] = max(st["sup_upper"], float(ubd.max()))
            for j in np.flatnonzero(~cert)[:3]:
                if len(st["not_certified"]) < 5:
                    st["not_certified"].append({"box": {nm: (float(blo[j, k]), float(bhi[j, k]))
                                                        for k, nm in enumerate(names)},
                                                "max_evaluated": float(mx[j]), "upper_bound": float(ubd[j])})
        if st["n_not_certified"]:
            st["status"] = "not_refuted"
    return st


def magnitude_bound(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None, *,
                    limit: float = LIMIT_NAT, gauge: str = "oscillation", gauge_vars: Sequence[str] = ("x",),
                    reference: Optional[Mapping[str, float]] = None, min_frac: float = MIN_FRAC,
                    max_boxes: int = 60_000, dense_points: int = 512, coarse: int = 33) -> dict:
    """Check (b). ``status`` in {"certified", "not_refuted", "failed"}; ``passed`` is False only for "failed"
    (PLAN: an equation fails only if an actual evaluation exceeds the bound).

    ``sup_lower`` is the largest actually evaluated value of the bounded quantity, ``sup_upper`` a rigorous upper
    bound of its supremum on the box (Inf when not available). The bounded quantity ``quantity`` is
        gauge="oscillation"   H = sup_x (1/2)(sup_{z,r} f - inf_{z,r} f)      (default; the plan's rule in the most
                              favourable gauge; |delta_eff| <= 2 H for every baseline)
        gauge="reference"     sup |f - f(reference, x)|, reference default zt = rh = 1 (260 mm, 21.9 mm)
        gauge="none"          sup |f|
    The domain check is separate (``domain_safety``): this function assumes a domain-safe expression and reports a
    non-finite evaluation as a failure.
    """
    if gauge not in ("oscillation", "reference", "none"):
        raise ValueError("gauge must be 'oscillation', 'reference' or 'none'")
    t0 = time.time()
    e0 = T.parse(expr) if isinstance(expr, str) else expr
    box = dict(O.extended_box() if box is None else box)
    res: dict = {"gauge": gauge, "limit": float(limit), "gauge_vars": list(gauge_vars), "dropped_gauge_terms": []}
    e = e0
    if gauge != "none":
        e, res["dropped_gauge_terms"] = drop_gauge_terms(e0, gauge_vars)
    spatial = [v for v in (T.variables(e) if e is not None else []) if v not in gauge_vars]
    if gauge != "none" and (e is None or not spatial):
        res.update(status="certified", passed=True, sup_lower=0.0, sup_upper=0.0, quantity="oscillation",
                   note="the expression depends on the gauge variables alone: it does not change the model",
                   seconds=round(time.time() - t0, 3))
        return res
    if gauge == "oscillation":
        _, names, lo, hi = _resolve(e, box)
        res["quantity"] = "half-oscillation over (" + ", ".join(spatial) + ") at fixed (" + \
            ", ".join(v for v in gauge_vars if v in names) + ")"
        st = _osc_bound(e, names, lo, hi, [n in gauge_vars for n in names], float(limit), min_frac=min_frac,
                        max_boxes=max_boxes, dense_points=dense_points, coarse=coarse)
        res.update(st)
        res["passed"] = st["status"] != "failed"
        res["delta_eff_bound"] = 2.0 * res["sup_upper"]
        res["seconds"] = round(time.time() - t0, 3)
        return res
    if gauge == "reference":
        ref = {s: 1.0 for s in spatial}
        ref.update({k: float(v) for k, v in (reference or {}).items() if k in spatial})
        target = T.op("-", e, T.substitute(e, {s: T.const(v, fixed=True) for s, v in ref.items()}))
        res["reference"] = ref
        res["quantity"] = "sup |f - f(reference, gauge variables)|"
    else:
        target = e
        res["quantity"] = "sup |f|"
    # (0) coarse grid: an actual value above the limit fails at once
    _, names, lo, hi = _resolve(target, box)
    axes = [np.linspace(lo[0, j], hi[0, j], coarse) for j in range(len(names))]
    pts = np.array(list(itertools.product(*axes))) if names else np.zeros((1, 0))
    tv = _values(target, names, pts)
    fin = np.isfinite(tv)
    q0 = float(np.max(np.abs(tv[fin]))) if fin.any() else 0.0
    if (not fin.all()) or q0 > limit * (1 + SLACK):
        j = int(np.argmax(np.where(fin, np.abs(tv), _INF)))
        res.update(status="failed", passed=False, sup_lower=(q0 if fin.all() else _INF), sup_upper=_INF,
                   witness={"point": {n: float(pts[j, k]) for k, n in enumerate(names)}, "value": float(tv[j])},
                   stage="coarse grid", seconds=round(time.time() - t0, 3))
        return res
    st = _bnb(target, names, lo, hi, float(limit), True, min_frac=min_frac, max_boxes=max_boxes,
              dense_points=dense_points)
    res.update({k: v for k, v in st.items() if k not in ("sup_lower", "sup_upper")})
    res["sup_lower"] = max(q0, st["sup_lower"]) if st["status"] != "failed" else max(q0, st["sup_lower"])
    res["sup_upper"] = max(st["sup_upper"], res["sup_lower"]) if st["status"] != "failed" else _INF
    res["passed"] = st["status"] != "failed"
    res["delta_eff_bound"] = 2.0 * res["sup_upper"] if gauge == "reference" else None
    res["seconds"] = round(time.time() - t0, 3)
    return res


def _osc_bound(e: T.Expr, names: List[str], lo: np.ndarray, hi: np.ndarray, gmask: Sequence[bool], limit: float, *,
               min_frac: float, max_boxes: int, dense_points: int, coarse: int) -> dict:
    """H = sup over the gauge variables of (1/2)(sup - inf over the spatial variables) <= limit.

    The gauge range is cut into slabs (adaptively). On a slab the whole range of f is enclosed in a window
    [mid - limit, mid + limit] by the two-sided bound of ``_bnb`` on f - mid; if that is certified, the oscillation
    at every gauge point of the slab is <= 2 limit. A slab whose window cannot be certified is bisected along
    its gauge variables down to ``min_frac``. FAILED is declared only for an actual pair of evaluations at one
    gauge point whose difference exceeds 2 limit."""
    d = len(names)
    gidx = [j for j in range(d) if gmask[j]]
    sidx = [j for j in range(d) if not gmask[j]]
    lim2 = 2.0 * limit * (1.0 + SLACK)
    root_w = np.where(hi[0] > lo[0], hi[0] - lo[0], 1.0)
    res = {"status": "certified", "sup_lower": 0.0, "sup_upper": 0.0, "witness": None, "n_boxes": 0, "n_slabs": 0,
           "n_slabs_certified": 0, "n_not_certified": 0, "not_certified": [], "max_slab_depth": 0}

    def planes(slo, shi, gpts):
        axes = [np.linspace(slo[j], shi[j], coarse) for j in sidx]
        sp = np.array(list(itertools.product(*axes))) if sidx else np.zeros((1, 0))
        pts = np.zeros((len(gpts), len(sp), d))
        pts[:, :, sidx] = sp[None, :, :]
        pts[:, :, gidx] = np.asarray(gpts)[:, None, :]
        return _values(e, names, pts.reshape(-1, d)).reshape(len(gpts), len(sp)), pts

    def fail(vals_row, pts_row, extra_val=None, extra_pt=None):
        v, pp = vals_row, pts_row
        if extra_val is not None:
            v, pp = np.append(v, extra_val), np.concatenate([pp, extra_pt[None, :]])
        if not np.all(np.isfinite(v)):
            j = int(np.argmax(~np.isfinite(v)))
            res.update(status="failed", sup_lower=_INF, sup_upper=_INF,
                       witness={"point": {n: float(pp[j, k]) for k, n in enumerate(names)}, "value": float(v[j])})
            return res
        ja, jb = int(np.argmax(v)), int(np.argmin(v))
        res.update(status="failed", sup_upper=_INF, sup_lower=max(res["sup_lower"], 0.5 * float(v[ja] - v[jb])),
                   witness={"gauge_point": {names[k]: float(pp[ja, k]) for k in gidx},
                            "max_at": {names[k]: float(pp[ja, k]) for k in sidx}, "max": float(v[ja]),
                            "min_at": {names[k]: float(pp[jb, k]) for k in sidx}, "min": float(v[jb]),
                            "half_oscillation": 0.5 * float(v[ja] - v[jb])})
        return res

    stack = [(lo[0].copy(), hi[0].copy(), 0)]
    per_slab = max(2000, max_boxes // 8)
    while stack:
        slo, shi, depth = stack.pop()
        res["n_slabs"] += 1
        res["max_slab_depth"] = max(res["max_slab_depth"], depth)
        gax = [np.linspace(slo[j], shi[j], 3) if shi[j] > slo[j] else np.array([slo[j]]) for j in gidx]
        gpts = np.array(list(itertools.product(*gax))) if gidx else np.zeros((1, 0))
        vals, pts = planes(slo, shi, gpts)
        if not np.all(np.isfinite(vals)):
            j = int(np.argmax(~np.isfinite(vals).all(1)))
            return fail(vals[j], pts[j])
        osc = vals.max(1) - vals.min(1)
        res["sup_lower"] = max(res["sup_lower"], 0.5 * float(osc.max()))
        if osc.max() > lim2:
            j = int(np.argmax(osc))
            return fail(vals[j], pts[j])
        m, n = float(vals.max()), float(vals.min())
        st = None
        for _ in range(4):
            if m - n > lim2 or res["n_boxes"] >= max_boxes:
                st = None
                break
            mid = 0.5 * (m + n)
            st = _bnb(T.op("-", e, T.const(mid, fixed=True)), names, slo[None, :], shi[None, :], limit, True,
                      min_frac=min_frac, max_boxes=min(per_slab, max(1, max_boxes - res["n_boxes"])),
                      dense_points=dense_points, root_w=root_w)
            res["n_boxes"] += st["n_boxes"]
            if st["status"] != "failed":
                break
            # an evaluation left the window: a violation only if it happens at ONE gauge point
            wp = np.array([st["witness"]["point"][nm] for nm in names])
            fp = float(st["witness"]["value"]) + mid
            v2, p2 = planes(slo, shi, wp[gidx][None, :])
            if (not math.isfinite(fp)) or max(float(v2.max()), fp) - min(float(v2.min()), fp) > lim2:
                return fail(v2[0], p2[0], fp, wp)
            res["sup_lower"] = max(res["sup_lower"], 0.5 * (max(float(v2.max()), fp) - min(float(v2.min()), fp)))
            m, n, st = max(m, fp), min(n, fp), None
        if st is not None and st["status"] == "certified":
            res["sup_upper"] = max(res["sup_upper"], float(st["sup_upper"]))
            res["n_slabs_certified"] += 1
            continue
        relw = np.array([(shi[j] - slo[j]) / root_w[j] for j in gidx]) if gidx else np.zeros(0)
        if len(relw) and relw.max() > min_frac and res["n_boxes"] < max_boxes:
            j = gidx[int(np.argmax(relw))]
            mid_g = 0.5 * (slo[j] + shi[j])
            h1, l2 = shi.copy(), slo.copy()
            h1[j], l2[j] = mid_g, mid_g
            stack += [(slo, h1, depth + 1), (l2, shi, depth + 1)]
            continue
        if st is not None and st["status"] == "not_refuted":
            ub = float(st["sup_upper"])
        else:
            r = _run(_Prog(e, names), slo[None, :], shi[None, :])
            ub = 0.5 * float(r.V[-1][1][0] - r.V[-1][0][0])
            ub = _INF if math.isnan(ub) else ub
        res["status"] = "not_refuted"
        res["n_not_certified"] += 1
        res["sup_upper"] = max(res["sup_upper"], ub)
        if len(res["not_certified"]) < 5:
            res["not_certified"].append({"slab": {nm: (float(slo[k]), float(shi[k])) for k, nm in enumerate(names)},
                                         "upper_bound": ub})
    res["sup_upper"] = max(res["sup_upper"], res["sup_lower"])
    return res


# ----------------------------------------------------------------------------------------------------------------------
# The audit of one equation
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class AuditReport:
    equation: str
    passed: bool
    domain: dict
    magnitude: dict
    box: dict
    k: int
    complexity: int
    limit: float
    gauge: str
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        def clean(o):
            if isinstance(o, dict):
                return {str(a): clean(b) for a, b in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(a) for a in o]
            if isinstance(o, (np.floating, float)):
                return float(o) if math.isfinite(float(o)) else repr(float(o))
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return o
        return clean(asdict(self))

    def summary(self) -> str:
        m = self.magnitude
        return (f"{'PASS' if self.passed else 'REJECT'}  domain={self.domain['status']}  magnitude={m.get('status')} "
                f"[{m.get('sup_lower', float('nan')):.4g}, {m.get('sup_upper', float('nan')):.4g}] "
                f"(limit {self.limit:g}, gauge {self.gauge})  k={self.k} C={self.complexity}  {self.equation[:100]}")


def audit(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None, *,
          limit: float = LIMIT_NAT, gauge: str = "oscillation", gauge_vars: Sequence[str] = ("x",),
          reference: Optional[Mapping[str, float]] = None, domain_kw: Optional[dict] = None,
          magnitude_kw: Optional[dict] = None) -> AuditReport:
    """Both audits of one equation (string or tree; a template result is translated first with
    ``translate.parse_template``). ``passed`` requires (a) "safe" and (b) not "failed". If (a) does not pass, (b)
    is not run (its coarse grid would only repeat the finding) and is reported as "skipped".

    Run it on the search result AND on the refitted equation: the constants move in the refit."""
    e = T.parse(expr) if isinstance(expr, str) else expr
    box = dict(O.extended_box() if box is None else box)
    dom = domain_safety(e, box, **(domain_kw or {}))
    notes = []
    if dom["safe"]:
        mag = magnitude_bound(e, box, limit=limit, gauge=gauge, gauge_vars=gauge_vars, reference=reference,
                              **(magnitude_kw or {}))
    else:
        mag = {"status": "skipped", "passed": False, "gauge": gauge, "limit": float(limit),
               "note": "domain safety did not pass"}
    if mag.get("status") == "not_refuted":
        notes.append("magnitude bound not certified: no evaluation exceeds the limit, the rigorous upper bound "
                     f"is {mag['sup_upper']:.6g}")
    return AuditReport(equation=T.to_string(e), passed=bool(dom["safe"] and mag.get("passed", False)), domain=dom,
                       magnitude=mag, box={k: tuple(v) for k, v in box.items()}, k=T.n_constants(e),
                       complexity=T.complexity(e), limit=float(limit), gauge=gauge, notes=notes)


def audit_trajectory(expr: Union[str, T.Expr], x_range: Tuple[float, float] = O.TRAJECTORY_BOX,
                     variable: str = "x", train_range: Tuple[float, float] = O.TRAIN_X_RANGE,
                     value_range: Optional[Tuple[float, float]] = None, n_grid: int = 4001) -> dict:
    """Audit of a 1-D trajectory (a raw parameter as a function of x) on the full interval x in [-4.2, 1.4]:
    domain safety (sound), a rigorous enclosure of the value range (natural intervals on 1024 slices), the
    numerically evaluated range on the full interval and on the training range, and their ratio (how far the
    extrapolation leaves the fitted band). With ``value_range`` = (lo, hi) the trajectory must stay inside it:
    checked with the tight machinery of (b) on f - (lo + hi)/2 against (hi - lo)/2."""
    e = T.parse(expr) if isinstance(expr, str) else expr
    extra = [v for v in T.variables(e) if v != variable]
    if extra:
        raise ValueError(f"a trajectory depends on {variable!r} only; found {extra}")
    box = {variable: (float(x_range[0]), float(x_range[1]))}
    dom = domain_safety(e, box)
    xs = np.linspace(box[variable][0], box[variable][1], n_grid)
    v = np.asarray(T.evaluate(e, {variable: xs}), dtype=np.float64) + np.zeros(n_grid)
    tr = (xs >= train_range[0]) & (xs <= train_range[1])
    out = {"equation": T.to_string(e), "domain": dom, "finite": bool(np.all(np.isfinite(v))),
           "range_evaluated": (float(np.nanmin(v)), float(np.nanmax(v))),
           "range_train": (float(np.nanmin(v[tr])), float(np.nanmax(v[tr]))) if tr.any() else None,
           "k": T.n_constants(e), "complexity": T.complexity(e)}
    if T.variables(e):
        edges = np.linspace(box[variable][0], box[variable][1], 1025)
        r = _run(_Prog(e, [variable]), edges[:-1, None], edges[1:, None])
        root = len(r.V) - 1
        out["range_enclosure"] = (float(r.V[root][0].min()), float(r.V[root][1].max()))
    else:
        out["range_enclosure"] = out["range_evaluated"]
    if out["range_train"] is not None:
        w_tr = out["range_train"][1] - out["range_train"][0]
        w_all = out["range_evaluated"][1] - out["range_evaluated"][0]
        out["extrapolation_ratio"] = float(w_all / w_tr) if w_tr > 0 else (1.0 if w_all == 0 else math.inf)
    out["passed"] = bool(dom["safe"] and out["finite"])
    if value_range is not None:
        mid, half = 0.5 * (value_range[0] + value_range[1]), 0.5 * (value_range[1] - value_range[0])
        mag = magnitude_bound(T.op("-", e, T.const(mid, fixed=True)), box, limit=half, gauge="none") \
            if dom["safe"] else {"status": "skipped", "passed": False}
        out["value_range"] = tuple(float(a) for a in value_range)
        out["value_range_check"] = mag
        out["passed"] = bool(out["passed"] and mag.get("passed", False))
    return out
