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

(c) Numerical hygiene (``numerics``; added after the user's confirmation 1 of 25 Nov 2026)
    OVERFLOW_CAP keeps every node finite. It does NOT bound the raw value of f and does NOT prevent cancellation:
    f = g(z, r, x) + c(x) with |c| = 1e4 passes (a) and (b) (the gauge-invariant H does not see c) while a float32
    evaluation loses 1e4 * 2^-24 = 6e-4 nat of g. Three mechanisms close this, and all hold after a refit because
    the audit is repeated on the refitted constants:
      * top-level additive terms in the gauge variables alone are stripped before translation and refit
        (``drop_gauge_terms``; ``refit.expression_correction`` does it): they carry no likelihood, and their
        constants are neither trained nor counted;
      * the RAW, unshifted |f| of what remains must stay below RAW_LIMIT_NAT = 16 nat on the extended box (tight
        machinery, gauge "none"); adding f to log P_base then costs < 4e-6 nat in float32, 1e-14 in float64;
      * a first-order rounding-error bound of f (``rounding_error_bound``) must stay below 1e-6 nat in float64, the
        precision in which ``refit.CorrectedModel`` evaluates the correction and the softmax (it refuses a float32
        correction); the float32 bound is reported with the verdict ``float32_safe`` (<= 1e-5 nat).

Screening and finalists (user's confirmation 2). ``audit().passed`` is the screening rule: an undecided magnitude
    bound passes. A finalist needs ``certified`` - (a) safe, (b) and (c) PROVEN - together with the tail check, on
    the constants after the joint refit: ``certify_finalist``.

Tail check (PLAN 17.11, S4, and the user's decision on the depth rule; ``tail_check``). Slopes are those of the 3-D
    density rho, not of the ring mass 2 pi r rho. Every bound comes from the record of the FIXED reference baseline
    (``checks.decide_tail_criterion``; the Track A finalist before any PySR refit, or the refit-only control C1),
    never from a jointly refitted baseline: a joint refit cannot bend its own bound. Radial: "d ln rho / d ln r <=
    -2 on r in [200, 1500] mm" if the reference satisfies it, otherwise "not flatter than the reference" (which is
    monotone). Depth: beyond z_90, d ln rho_model / d z' <= max(0, d ln rho_reference / d z'), point by point.
    Stage 2 adds rising_stretches(model) <= rising_stretches(reference) per depth profile (sign-change counts are a
    diagnostic). Tolerances of the comparisons with the reference: 1e-5 / mm, 1e-3. The record carries the reference's
    slope tables and the cells where it rises itself, and is bound to the baseline's hash.

Limits of the statement: real arithmetic with the constants as parsed; library functions assumed accurate to 64 ulp;
the float32 allowance tau is heuristic; the rounding-error bound is first order (derivative enclosures on sub-boxes,
4 ulp per library call assumed).
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
           "magnitude_bound", "audit", "audit_trajectory", "AuditReport", "drop_gauge_terms", "oscillation_expr",
           "RAW_LIMIT_NAT", "ROUNDING_TOL_NAT", "FLOAT32_TOL_NAT", "rounding_error_bound", "numerics",
           "tail_slopes", "certify_finalist", "canonical_gauge_form", "gauge_term_records", "tail_check",
           "baseline_against_reference",
           "FLAT_SLACK", "RADIAL_CRITERIA"]

OVERFLOW_CAP = 1e30          # eight decades below the float32 limit 3.4e38
TAU_REL = 1e-5               # float32 allowance around a forbidden set (relative to the subtree's magnitude)
MIN_FRAC = 2.0 ** -10        # minimum box side as a fraction of the audit box (2.2 mm, 1.5 mm, 0.005 in x)
LIMIT_NAT = 1.0              # PLAN: sup |delta| <= 1 nat
SLACK = 1e-9                 # relative slack of every comparison with the limit
RAW_LIMIT_NAT = 16.0         # bound on the RAW, unshifted |f| (top-level gauge terms stripped) on the extended box
ROUNDING_TOL_NAT = 1e-6      # largest admissible rounding error of f in the precision the refit uses (float64)
FLOAT32_TOL_NAT = 1e-5       # an equation is called float32-safe if its float32 rounding-error bound is below this
U32, U64 = 2.0 ** -24, 2.0 ** -53      # unit roundoff of float32 / float64
ULP_LIB = 4.0                # assumed accuracy of a library function call, in units of the unit roundoff
TAIL_R_FROM_MM, TAIL_Z_FROM_MM = 700.0, 1450.0       # where the grid of common.py ends
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
    if not dropped:
        return e, []                                       # nothing to strip: the tree itself, not a re-association
    s0, acc = keep[0]
    if s0 < 0:
        acc = T.op("neg", acc)
    for s, t in keep[1:]:
        acc = T.op("+" if s > 0 else "-", acc, t)
    return acc, dropped


def gauge_term_records(e: T.Expr, gauge_vars: Sequence[str] = ("x",), stage: str = "") -> List[dict]:
    """One record per top-level additive term that depends on the gauge variables alone (what ``drop_gauge_terms``
    strips), for the audit trail (PLAN 17.11, S3): {"term": signed text, "sign", "constants": the values of its
    free constants, "n_constants", "variables", "stage": where it was stripped}. These constants are neither
    trained nor counted in k."""
    gv = set(gauge_vars)
    out = []
    for sg, t in _split_sum(e):
        if set(T.variables(t)) <= gv:
            cs = T.constants(t)
            out.append({"term": ("-" if sg < 0 else "+") + " " + T.to_string(t), "sign": "-" if sg < 0 else "+",
                        "constants": cs, "n_constants": len(cs), "variables": T.variables(t), "stage": str(stage)})
    return out


def canonical_gauge_form(e: T.Expr, gauge_vars: Sequence[str] = ("x",)) -> Optional[T.Expr]:
    """The expression without its top-level gauge terms, with the remaining top-level sum ALWAYS rebuilt in one
    association ((t1 + t2) - t3 ...), whether or not anything was dropped. Two hall-of-fame members that differ
    only by gauge terms or by the bracketing of their top-level sum get the same ``translate.expr_hash`` of this
    form (``drop_gauge_terms`` returns an untouched tree when nothing is dropped, so its hash can differ by
    bracketing). None for a pure gauge expression. The order of the terms is kept."""
    gv = set(gauge_vars)
    keep = [(sg, t) for sg, t in _split_sum(e) if not set(T.variables(t)) <= gv]
    if not keep:
        return None
    s0, acc = keep[0]
    if s0 < 0:
        acc = T.op("neg", acc)
    for sg, t in keep[1:]:
        acc = T.op("+" if sg > 0 else "-", acc, t)
    return acc


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
                      min_frac=min_frac, max_boxes=max(1, max_boxes - res["n_boxes"]) if not gidx else
                      min(per_slab, max(1, max_boxes - res["n_boxes"])),
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
# (c) Numerical hygiene: raw magnitude and rounding-error bound
# ----------------------------------------------------------------------------------------------------------------------
def _supabs(iv: Interval) -> np.ndarray:
    return np.maximum(np.abs(iv[0]), np.abs(iv[1]))


def _t(a, b):
    """a * b for non-negative arrays with 0 * Inf = 0."""
    q = a * b
    return np.where(np.isnan(q), 0.0, q)


def _rounding_error(prog: _Prog, V: list, u: float, c_lib: float = ULP_LIB) -> np.ndarray:
    """First-order forward bound of the rounding error of the root, per box: every operation contributes
    (ulps) * u * sup|value| and inherits the errors of its arguments times sup|partial derivative| on the box.
    ``sqa`` and ``sqrt`` use min(first-order term, sqrt(error)): |sqrt p - sqrt q| <= sqrt|p - q| holds exactly."""
    N = len(V[0][0])
    E: list = []
    ones = np.ones(N)
    with np.errstate(all="ignore"):
        for idx, nd in enumerate(prog.nodes):
            if nd.kind != "op":                             # a constant given in more digits than the working
                c = abs(float(nd.value)) if nd.kind == "const" else 0.0    # precision is rounded once: u |c|
                E.append(np.full(N, u * c if math.isfinite(c) else _INF))
                continue
            v, name = V[idx], nd.name
            mag = _supabs(v)
            a, Ea = V[nd.ch[0]], E[nd.ch[0]]
            b, Eb = (V[nd.ch[1]], E[nd.ch[1]]) if len(nd.ch) > 1 else (None, None)
            local = c_lib * u * mag
            if name in _UNARY_DER:
                prop = _t(_supabs(_UNARY_DER[name](a, v)), Ea)
                if name == "sqa":
                    prop = np.minimum(prop, np.sqrt(Ea))
                if name in ("neg", "abs"):
                    local = 0.0 * mag
                elif name == "square":
                    local = u * mag
            elif name in ("+", "-"):
                prop, local = Ea + Eb, u * mag
            elif name == "*":
                prop, local = _t(_supabs(b), Ea) + _t(_supabs(a), Eb), u * mag
            elif name == "/":
                r = _supabs(_recip(b))
                prop, local = _t(r, Ea) + _t(_t(mag, r), Eb), u * mag
            elif name == "inv":
                prop, local = _t(_supabs(_square(v)), Ea), u * mag
            elif name == "log":
                prop = _t(_supabs(_recip(a)), Ea)
            elif name == "log1p":
                prop = _t(_supabs(_recip(_add(a, (ones, ones)))), Ea)
            elif name == "sqrt":
                prop, local = np.minimum(_t(_supabs(_recip(_scale(v, 2.0))), Ea), np.sqrt(Ea)), u * mag
            elif name == "ppow":
                dx = _supabs(_mul(_mul(b, v), _l1p_der(a)))
                dy = _supabs(_mul(v, _UNARY_VAL["l1p"](a)))
                prop = _t(dx, Ea) + _t(dy, Eb)
            elif name == "pow":
                ex = prog.nodes[nd.ch[1]]
                if ex.kind == "const" and ex.fixed and float(ex.value).is_integer():
                    n = int(ex.value)
                    if n == 0:
                        prop = np.zeros(N)
                    elif n > 0:
                        prop = _t(_supabs(_scale(_ipow(a, n - 1), float(n))), Ea)
                    else:
                        prop = _t(_supabs(_scale(_recip(_ipow(a, -n + 1)), float(n))), Ea)
                else:
                    tiny = np.finfo(float).tiny
                    apos = (np.maximum(a[0], tiny), np.maximum(a[1], tiny))
                    prop = _t(_supabs(_mul(v, _mul(b, _recip(apos)))), Ea) + _t(_supabs(_mul(v, _mono(np.log, apos))), Eb)
            else:
                raise ValueError(f"no rounding model for operator {name!r}")
            e = prop + local
            E.append(np.where(np.isnan(e), _INF, e))
    return E[prog.root]


def _partition(lo: np.ndarray, hi: np.ndarray, n_split: int, max_boxes: float = 3e4):
    d = lo.shape[1]
    if d == 0:
        return lo, hi
    n = int(n_split)
    while n ** d > max_boxes and n > 2:
        n -= 1
    edges = [np.linspace(lo[0, j], hi[0, j], n + 1) for j in range(d)]
    idx = np.array(list(itertools.product(range(n), repeat=d)))
    blo = np.stack([edges[j][idx[:, j]] for j in range(d)], 1)
    bhi = np.stack([edges[j][idx[:, j] + 1] for j in range(d)], 1)
    return blo, bhi


def rounding_error_bound(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None,
                         n_split: int = 16) -> dict:
    """Upper bound (first order) of the rounding error of evaluating the expression in float32 and in float64,
    anywhere on the box: {"float32": nat, "float64": nat}. Inputs are taken as exact in the working precision (the
    model IS the function of its float inputs); a constant is charged one rounding to that precision, each
    arithmetic operation one unit roundoff of its result, each library call ``ULP_LIB`` = 4. The bound is the maximum over a partition of the box
    into n_split^d sub-boxes, with interval enclosures of the values and of the derivatives on each: it can only
    over-estimate through the size of the sub-boxes. It is what detects cancellation: f = c(x) + g with |c| = 1e4
    has a float32 bound of about 1e-3 nat although the model only uses g."""
    e, names, lo, hi = _resolve(expr, box)
    prog = _Prog(e, names)
    blo, bhi = _partition(lo, hi, n_split)
    out = {"float32": 0.0, "float64": 0.0, "n_boxes": int(len(blo)), "ulp_per_library_call": ULP_LIB}
    for s0 in range(0, len(blo), 4096):
        run = _run(prog, blo[s0:s0 + 4096], bhi[s0:s0 + 4096])
        out["float32"] = max(out["float32"], float(np.max(_rounding_error(prog, run.V, U32))))
        out["float64"] = max(out["float64"], float(np.max(_rounding_error(prog, run.V, U64))))
    return out


def numerics(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None, *,
             raw_limit: float = RAW_LIMIT_NAT, gauge_vars: Sequence[str] = ("x",),
             magnitude_kw: Optional[dict] = None) -> dict:
    """Check (c), numerical hygiene of the correction as the refit evaluates it (top-level additive terms in the
    gauge variables alone are stripped first, exactly as ``refit.expression_correction`` does):

      raw       sup |f| of the RAW, unshifted expression on the box, against ``raw_limit`` (16 nat), with the tight
                machinery of (b) in gauge "none". The half-oscillation H <= 1 does not bound it: f = g + 1e4 x
                linv(0 rh) has H < 1 and |f| = 4e4. |f| <= 16 guarantees that adding f to log P_base loses at most
                2^-24 (16 + |log P|) < 4e-6 nat in float32 and 1e-14 nat in float64.
      rounding  ``rounding_error_bound``: float64 must stay below ROUNDING_TOL_NAT = 1e-6 nat (the refit evaluates
                f in float64); ``float32_safe`` says whether the float32 bound is below 1e-5 nat as well.

    ``passed`` = raw not failed and the float64 bound within tolerance; ``certified`` = raw certified and the
    float64 bound within tolerance."""
    e0 = T.parse(expr) if isinstance(expr, str) else expr
    e, dropped = drop_gauge_terms(e0, gauge_vars)
    res: dict = {"dropped_gauge_terms": dropped, "raw_limit": float(raw_limit), "rounding_tol": ROUNDING_TOL_NAT,
                 "float32_tol": FLOAT32_TOL_NAT}
    if e is None:
        res.update(expression=None, raw={"status": "certified", "passed": True, "sup_lower": 0.0, "sup_upper": 0.0},
                   rounding={"float32": 0.0, "float64": 0.0, "n_boxes": 0}, float32_safe=True, passed=True,
                   certified=True, note="nothing remains after the gauge terms are stripped")
        return res
    raw = magnitude_bound(e, box, limit=raw_limit, gauge="none", **(magnitude_kw or {}))
    rnd = rounding_error_bound(e, box)
    ok64 = bool(rnd["float64"] <= ROUNDING_TOL_NAT)
    res.update(expression=T.to_string(e), raw=raw, rounding=rnd,
               float32_safe=bool(raw["passed"] and rnd["float32"] <= FLOAT32_TOL_NAT),
               passed=bool(raw["passed"] and ok64), certified=bool(raw["status"] == "certified" and ok64))
    return res


def _tail_component(prog: _Prog, l: np.ndarray, h: np.ndarray, j: int, key: str) -> Interval:
    run = _run(prog, l, h, grad=True)
    g = (run.G[prog.root][0][:, j], run.G[prog.root][1][:, j])
    g = _mul(g, (l[:, j], h[:, j])) if key == "dlnr" else _scale(g, 1.0 / O.ZETA_SCALE_MM)
    return _fix(*g)


def tail_slopes(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None, *,
                r_from_mm: float = TAIL_R_FROM_MM, z_from_mm: float = TAIL_Z_FROM_MM, tol_dlnr: float = 0.01,
                tol_dz_per_mm: float = 2e-5, max_boxes: int = 60_000, n_split: int = 6) -> dict:
    """What the equation adds to the tail slopes of the density, on the part of the extended box beyond the grid:

        "dlnr"        d f / d ln r   = rh * d f / d rh    for r  >= r_from_mm (700 mm), all z', x of the box
        "dz_per_mm"   d f / d z'     = (d f / d zt) / 260 for z' >= z_from_mm (1450 mm), all r, x of the box

    each with a rigorous enclosure ("sup", "inf": forward-mode interval derivatives, refined by bisection until
    the enclosure is within ``tol`` of the largest / smallest value found at a box centre, or ``max_boxes``) and
    those centre values ("sup_sampled", "inf_sampled"). ln rho_model = ln rho_base + f - ln Z(x), so these add
    to the baseline's tail slopes; ``certify_finalist`` combines them. Tolerances: 0.01 in the log-log slope,
    2e-5 per mm (0.005 per 260 mm) in the depth slope."""
    e = T.parse(expr) if isinstance(expr, str) else expr
    box = dict(O.extended_box() if box is None else box)
    out: dict = {"r_from_mm": float(r_from_mm), "z_from_mm": float(z_from_mm)}
    spec = (("dlnr", "rh", r_from_mm / O.RHO_SCALE_MM, tol_dlnr), ("dz_per_mm", "zt", z_from_mm / O.ZETA_SCALE_MM,
                                                                  tol_dz_per_mm))
    for key, vname, start, tol in spec:
        if vname not in T.variables(e) or vname not in box or start >= box[vname][1]:
            out[key] = {"sup": 0.0, "inf": 0.0, "sup_sampled": 0.0, "inf_sampled": 0.0, "n_boxes": 0,
                        "note": "no dependence, or no tail region in the box"}
            continue
        sub = dict(box)
        sub[vname] = (max(float(start), box[vname][0]), box[vname][1])
        _, names, lo, hi = _resolve(e, sub)
        prog = _Prog(e, names)
        d, j = len(names), names.index(vname)
        root_w = np.where(hi[0] > lo[0], hi[0] - lo[0], 1.0)
        qlo, qhi = _partition(lo, hi, n_split)
        sup_s, inf_s, sup_done, inf_done, n_eval = -_INF, _INF, -_INF, _INF, 0
        while len(qlo):
            g = _tail_component(prog, qlo, qhi, j, key)
            c = 0.5 * (qlo + qhi)
            gc = _tail_component(prog, c, c, j, key)
            gc = 0.5 * (gc[0] + gc[1])
            gc = gc[np.isfinite(gc)]
            n_eval += 2 * len(qlo)
            if len(gc):
                sup_s, inf_s = max(sup_s, float(gc.max())), min(inf_s, float(gc.min()))
            relw = (qhi - qlo) / root_w
            act = ((g[1] > sup_s + tol) | (g[0] < inf_s - tol)) & (relw.max(1) > MIN_FRAC)
            if n_eval > max_boxes:
                act[:] = False
            if (~act).any():
                sup_done, inf_done = max(sup_done, float(g[1][~act].max())), min(inf_done, float(g[0][~act].min()))
            if not act.any():
                break
            alo, ahi = qlo[act], qhi[act]
            best_w, best = np.full(len(alo), _INF), np.zeros(len(alo), int)
            for k in range(d):                              # bisect along the side that tightens the enclosure most
                mid = 0.5 * (alo[:, k] + ahi[:, k])
                h1, l2 = ahi.copy(), alo.copy()
                h1[:, k], l2[:, k] = mid, mid
                g1, g2 = _tail_component(prog, alo, h1, j, key), _tail_component(prog, l2, ahi, j, key)
                n_eval += 2 * len(alo)
                w = np.maximum(g1[1] - g1[0], g2[1] - g2[0])
                w = np.where(np.isnan(w) | ((ahi[:, k] - alo[:, k]) / root_w[k] <= MIN_FRAC), _INF, w)
                better = w < best_w
                best_w, best = np.where(better, w, best_w), np.where(better, k, best)
            rows = np.arange(len(alo))
            mid = 0.5 * (alo[rows, best] + ahi[rows, best])
            h1, l2 = ahi.copy(), alo.copy()
            h1[rows, best], l2[rows, best] = mid, mid
            qlo, qhi = np.concatenate([alo, l2]), np.concatenate([h1, ahi])
        out[key] = {"sup": sup_done, "inf": inf_done, "sup_sampled": sup_s, "inf_sampled": inf_s,
                    "n_boxes": int(n_eval), "tolerance": float(tol),
                    "converged": bool(sup_done <= sup_s + tol * 1.0001 and inf_done >= inf_s - tol * 1.0001)}
    return out


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
    numerics: dict = field(default_factory=dict)
    certified: bool = False                # (a) safe, (b) CERTIFIED, (c) certified: what a finalist needs
    require_certified: bool = False        # True: ``passed`` is ``certified``
    stage: str = ""                        # where in the pipeline this audit was made
    k_as_written: int = 0                  # free constants of the equation as written (``k`` excludes gauge terms)
    k_stripped: int = 0                    # free constants inside the stripped gauge terms: not trained, not in k
    stripped_gauge_terms: List[dict] = field(default_factory=list)   # every stripped term, with its stage

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
        m, nm = self.magnitude, self.numerics
        up = m.get("sup_upper", float("nan"))
        raw = nm.get("raw", {}).get("status", "n/a") if nm else "n/a"
        e64 = nm.get("rounding", {}).get("float64", float("nan")) if nm else float("nan")
        return (f"{'PASS' if self.passed else 'REJECT'}  domain={self.domain['status']}  magnitude={m.get('status')} "
                f"H in [{m.get('sup_lower', float('nan')):.4g}, {up:.4g}] (limit {self.limit:g}, gauge {self.gauge}; "
                f"|delta_eff| <= 2H = {2 * up:.4g})  raw={raw}  err64<={e64:.2g}  certified={self.certified}  "
                f"k={self.k} C={self.complexity}  {self.equation[:100]}")


def audit(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None, *,
          limit: float = LIMIT_NAT, gauge: str = "oscillation", gauge_vars: Sequence[str] = ("x",),
          reference: Optional[Mapping[str, float]] = None, domain_kw: Optional[dict] = None,
          magnitude_kw: Optional[dict] = None, require_certified: bool = False,
          raw_limit: float = RAW_LIMIT_NAT, stage: str = "audit",
          carried_stripped: Optional[Sequence[Mapping]] = None) -> AuditReport:
    """The audit of one equation (string or tree; a template result is translated first with
    ``translate.parse_template``): (a) domain safety, (b) magnitude bound, (c) numerical hygiene (``numerics``).

    ``passed``      (a) "safe", (b) not "failed", (c) raw magnitude not failed and float64 rounding within tolerance.
                    This is the SCREENING rule of the search stage: as the plan says, (b) fails only on an actual
                    evaluation above the limit, so an undecided bound passes with its interval reported.
    ``certified``   (a) "safe", (b) "certified", (c) certified. A FINALIST needs this, on the constants after the
                    joint refit: call with ``require_certified=True`` (then ``passed`` is ``certified``), or use
                    ``certify_finalist``, which adds the tail check.
    If (a) does not pass, (b) and (c) are not run and are reported as "skipped".

    Gauge terms (PLAN 17.11, S3): top-level terms in the gauge variables alone are stripped for (c) and for the
    count ``k`` (their constants are neither trained nor counted); every stripped term is logged in
    ``stripped_gauge_terms`` with its text, its constants and ``stage`` (pass "before refit" / "after refit";
    ``carried_stripped`` hands on the records of earlier stages, e.g. those of ``refit.expression_correction``,
    so that the post-refit record still shows what was removed before translation). ``k_as_written`` is the
    count including them.

    Run it on the search result AND on the refitted equation: the constants move in the refit."""
    e = T.parse(expr) if isinstance(expr, str) else expr
    box = dict(O.extended_box() if box is None else box)
    here = gauge_term_records(e, gauge_vars, stage) if gauge != "none" else []
    stripped = [dict(r) for r in (carried_stripped or [])] + here
    k_here = sum(r["n_constants"] for r in here)
    dom = domain_safety(e, box, **(domain_kw or {}))
    notes = []
    if dom["safe"]:
        mag = magnitude_bound(e, box, limit=limit, gauge=gauge, gauge_vars=gauge_vars, reference=reference,
                              **(magnitude_kw or {}))
        num = numerics(e, box, raw_limit=raw_limit, gauge_vars=(gauge_vars if gauge != "none" else ()),
                       magnitude_kw=magnitude_kw)
    else:
        mag = {"status": "skipped", "passed": False, "gauge": gauge, "limit": float(limit),
               "note": "domain safety did not pass"}
        num = {"status": "skipped", "passed": False, "certified": False, "note": "domain safety did not pass"}
    if mag.get("status") == "not_refuted":
        notes.append("magnitude bound not certified: no evaluation exceeds the limit, the rigorous upper bound "
                     f"is {mag['sup_upper']:.6g}")
    if dom["safe"] and not num["passed"]:
        notes.append("numerical hygiene: raw |f| " + str(num["raw"].get("status")) +
                     f" against {raw_limit:g} nat, float64 rounding bound {num['rounding']['float64']:.3g} nat")
    screen = bool(dom["safe"] and mag.get("passed", False) and num.get("passed", False))
    cert = bool(dom["safe"] and mag.get("status") == "certified" and num.get("certified", False))
    if require_certified and screen and not cert:
        notes.append("not certified: a finalist needs a proven magnitude bound (and proven raw bound), not only "
                     "the absence of a violating evaluation")
    if stripped:
        notes.append(f"{len(stripped)} gauge term(s) stripped (not trained, not counted in k): " +
                     "; ".join(f"{r['term']} [{r['stage']}]" for r in stripped))
    return AuditReport(equation=T.to_string(e), passed=(cert if require_certified else screen), domain=dom,
                       magnitude=mag, box={k: tuple(v) for k, v in box.items()}, k=T.n_constants(e) - k_here,
                       complexity=T.complexity(e), limit=float(limit), gauge=gauge, notes=notes, numerics=num,
                       certified=cert, require_certified=bool(require_certified), stage=str(stage),
                       k_as_written=T.n_constants(e), k_stripped=k_here, stripped_gauge_terms=stripped)


def _grad_sup(prog: _Prog, lo: np.ndarray, hi: np.ndarray, j: int, kind: str, n_sub: int = 4) -> np.ndarray:
    """Rigorous upper bound, per box, of d f / d ln r (kind "dlnr": rh * df/drh) or d f / d z' in 1/mm (kind
    "dz"), the box cut into ``n_sub`` pieces along every side of non-zero width (maximum over the pieces)."""
    N, d = lo.shape
    wide = [k for k in range(d) if np.any(hi[:, k] > lo[:, k])]
    best = np.full(N, -_INF)
    for idx in itertools.product(range(n_sub), repeat=len(wide)):
        l, h = lo.copy(), hi.copy()
        for k, i in zip(wide, idx):
            w = (hi[:, k] - lo[:, k]) / n_sub
            l[:, k], h[:, k] = lo[:, k] + i * w, lo[:, k] + (i + 1) * w
        run = _run(prog, l, h, grad=True)
        g = (run.G[prog.root][0][:, j], run.G[prog.root][1][:, j])
        g = _mul(g, (l[:, j], h[:, j])) if kind == "dlnr" else _scale(g, 1.0 / O.ZETA_SCALE_MM)
        best = np.maximum(best, np.where(np.isnan(g[1]), _INF, g[1]))
    return best


def _grad_sup_refined(prog: _Prog, lo: np.ndarray, hi: np.ndarray, j: int, kind: str, need: np.ndarray,
                      levels: Sequence[int] = (4, 16, 64)) -> np.ndarray:
    """``_grad_sup``, refined only where it matters: boxes whose bound exceeds ``need`` (the room the baseline
    leaves) are cut finer, level by level. A bound that stays above ``need`` after the last level is a refusal:
    over-estimation costs false refusals only."""
    sup = _grad_sup(prog, lo, hi, j, kind, levels[0])
    d_wide = int(np.sum(np.any(hi > lo, axis=0)))
    for n_sub in levels[1:]:
        bad = np.flatnonzero(sup > need)
        if not len(bad) or len(bad) * n_sub ** max(d_wide, 1) > 4e6:
            break
        sup[bad] = np.minimum(sup[bad], _grad_sup(prog, lo[bad], hi[bad], j, kind, n_sub))
    return sup


FLAT_SLACK = 1e-9            # numerical slack of "<= 0" for a slope enclosure (log-log units, or nat / mm * 260)
RADIAL_CRITERIA = ("loglog_slope_le_limit", "monotone_and_no_flattening")


def _table(a) -> np.ndarray:
    """A slope table of a tail-criterion record (nested lists, None = no finite slope) as a float array with NaN."""
    return np.array(a, dtype=np.float64)                   # None -> NaN


def _sup_chunks(prog: _Prog, lo: np.ndarray, hi: np.ndarray, j: int, kind: str, need: np.ndarray,
                levels: Sequence[int]) -> np.ndarray:
    out = np.empty(len(lo))
    for s0 in range(0, len(lo), 16384):
        out[s0:s0 + 16384] = _grad_sup_refined(prog, lo[s0:s0 + 16384], hi[s0:s0 + 16384], j, kind,
                                               need[s0:s0 + 16384], levels)
    return out


def tail_check(expr: Union[str, T.Expr], criterion: Mapping, box: Optional[Mapping] = None, *,
               measured: Optional[Mapping] = None, baseline_frozen: Optional[bool] = None,
               enforce_stage2: bool = False, enforce_sign_changes: Optional[bool] = None) -> dict:
    """The tail part of a finalist's certificate (PLAN 17.11, S4, and the user's decision on the depth rule).

    ``criterion`` is the record of ``checks.decide_tail_criterion`` for the FIXED reference baseline (the Track A
    finalist before any PySR refit, or the refit-only control C1): it supplies the rules and every bound.
    ``measured`` is the same kind of record for the jointly REFITTED baseline (``reference_kind =
    "refitted_baseline"``): it supplies the model side. The model is

        ln rho_model = ln rho_refitted_baseline + f - ln Z(x),

    so a slope of the model between two neighbouring nodes of the record's grid is the refitted baseline's
    finite-difference slope there plus the mean of the correction's derivative on the segment, which is bounded
    by the supremum of d f / d z' (or d f / d ln r) on it - enclosed rigorously by forward-mode interval
    derivatives, refined where the bound leaves little room. All slopes are of the 3-D density rho, never of the
    ring mass. Without ``measured`` the reference table stands in for the model side, which is correct only for a
    frozen baseline: with ``baseline_frozen=False`` and no ``measured`` the check does not pass.

    CERTIFIED, at every pre-registered momentum p, every depth interval beyond the reference's z_90(p) and every
    radius of the record ("depth"), and at every depth row and every radial interval overlapping [200, 1500] mm
    ("radial"), with x = x(p):
      depth    "inherit_reference":  slope_model <= max(0, slope_reference)        - the model may not rise where
               the fixed reference falls, and may not rise faster where it rises;
      radial   "loglog_slope_le_limit":       slope_model <= -2;
               "monotone_and_no_flattening":  slope_model <= slope_reference (< 0) - not flatter than the fixed
               reference; with a frozen baseline also d f / d ln r <= 0 on the whole audit box beyond 200 mm (all
               z', all x: the continuum statement of S4).
    Tolerances of the two comparisons WITH THE REFERENCE: 1e-5 / mm in depth (0.001 nat per 100 mm) and 1e-3 in
    the log-log slope - the record's ``sign_change_tolerance``, i.e. what the sign-change count treats as no
    slope. Without them a correction with an exponentially small tail (tanh, erf) inside a cell where the
    reference rises would fail for a slope of 1e-8 / mm. The absolute rule "<= -2" and the continuum statement
    keep the rounding slack 1e-9.
    The certificate holds for the model's finite-difference slopes on the record's grid, at those momenta; the
    baselines enter numerically (their tables), not as enclosures.

    EVALUATED, not certified - the Stage-2 shape condition (PLAN section 7 and section 20, item 6). The model's
    finite-difference slopes on the record's grid are the refitted baseline's table plus the correction evaluated
    at the nodes. Per profile - d ln rho / d z' beyond z_90 along depth at each (momentum, radius) - the number of
    RISING STRETCHES is counted with ``checks.profile_rising_lobes`` and compared with the reference's:

        rising_stretches(model) <= rising_stretches(reference)      for every depth profile.

    A rising stretch: skip non-finite and exactly-zero slopes; a lobe is a maximal run of consecutive slopes of
    one sign; lobes whose largest |slope| is below the tolerance (1e-5 / mm in depth) are dropped; neighbouring
    surviving lobes of equal sign are one stretch; a rising stretch is a surviving stretch of positive slope. The
    grid and the tolerance belong to the record, so the count cannot be changed by choosing a resolution. It is a
    pass condition with ``enforce_stage2=True`` (``certify_finalist(stage2=True)``), reported otherwise.
    Radial profiles (d ln rho / d ln r for r > 200 mm along r at each (momentum, depth row)): the analogous
    quantity - stretches in which the density increases with r, tolerance 1e-3 - is counted and reported, but NOT
    separately enforced: the user's decision is worded for the depth check, and both radial slope rules above
    already forbid any radial rise of the model on these segments wherever the reference is monotone (a reference
    that is not monotone cannot be certified against at all). Reading of this module, for confirmation.
    The raw sign-change counts of the same profiles (``checks.profile_sign_changes``), total and per profile, are
    printed side by side as a DIAGNOSTIC that decides nothing: a purely falling correction can add a sign change
    where a profile begins inside a rising stretch of the reference ("+ -" becomes "- + -").
    Returns {"passed", "radial", "depth", "rising_stretches", "sign_changes", "reference", "reasons"}."""
    if enforce_sign_changes:
        raise ValueError("enforce_sign_changes is superseded (PLAN section 20, item 6): the enforced Stage-2 "
                         "condition counts rising stretches - pass enforce_stage2=True; sign changes are a diagnostic")
    from . import checks as _checks                        # NumPy only; imported here to avoid a module cycle

    e = T.parse(expr) if isinstance(expr, str) else expr
    box = dict(O.extended_box() if box is None else box)
    rad_rule = criterion["criterion"]["radial"]
    if rad_rule not in RADIAL_CRITERIA:
        raise ValueError(f"unknown radial criterion {rad_rule!r}")
    mside = criterion if measured is None else measured
    if measured is not None:
        pairs = [(key, measured[key], criterion[key]) for key in ("momenta_GeV", "z_nodes_mm", "r_nodes_mm")]
        pairs.append(("depth radii", measured["depth"]["r_nodes_mm"], criterion["depth"]["r_nodes_mm"]))
        for key, a_, b_ in pairs:
            if len(a_) != len(b_) or not np.allclose(a_, b_, rtol=1e-9, atol=1e-9):
                raise ValueError(f"the measured table and the reference differ in {key}: both must be made on the "
                                 "same grid and momenta")
    limit = float(criterion["slope_limit"])
    mom = np.asarray(criterion["momenta_GeV"], dtype=np.float64)
    xs = O.x_of_p(mom)
    z = np.asarray(criterion["z_nodes_mm"], dtype=np.float64)
    rn = np.asarray(criterion["r_nodes_mm"], dtype=np.float64)
    rd = np.asarray(criterion["depth"]["r_nodes_mm"], dtype=np.float64)
    k0 = list(criterion["depth"]["k0"])
    used = T.variables(e)
    names = [n for n in ("zt", "rh", "x") if n in used]
    prog = _Prog(e, names) if names else None
    reasons: List[str] = []
    joint_blind = measured is None and baseline_frozen is False
    out: dict = {"criterion": dict(criterion["criterion"]),
                 "reference": {"kind": criterion.get("reference_kind"), "label": criterion.get("label"),
                               "baseline_hash": criterion.get("baseline_hash"),
                               "record_hash": criterion.get("record_hash")},
                 "model_side": "reference table (baseline frozen)" if measured is None else
                 f"table of the refitted baseline ({measured.get('label')}, record {measured.get('record_hash')})",
                 "density": "3-D density rho, not ring mass"}
    if joint_blind:
        reasons.append("the baseline was refitted jointly and no slope table of the refitted baseline was given: "
                       "the model's slopes are unknown (pass measured=decide_tail_criterion(..., reference_kind="
                       "'refitted_baseline'))")

    def boxes(zlo, zhi, rlo, rhi, x):
        lo = np.zeros((len(zlo), len(names)))
        hi = np.zeros((len(zlo), len(names)))
        for k, n in enumerate(names):
            a_, b_ = {"zt": (zlo / O.ZETA_SCALE_MM, zhi / O.ZETA_SCALE_MM),
                      "rh": (rlo / O.RHO_SCALE_MM, rhi / O.RHO_SCALE_MM), "x": (x, x)}[n]
            lo[:, k], hi[:, k] = a_, b_
        return lo, hi

    def f_at(zz, rr, xx):                                  # the correction at grid nodes (actual evaluations)
        env = {"zt": zz / O.ZETA_SCALE_MM, "rh": rr / O.RHO_SCALE_MM, "x": xx}
        shape = np.broadcast_shapes(zz.shape, rr.shape, xx.shape)
        return np.broadcast_to(np.asarray(T.evaluate(e, {n: env[n] for n in used}), dtype=np.float64), shape) \
            if used else np.full(shape, float(T.evaluate(e, {})))

    # ---- depth beyond z_90 of the reference: slope_model <= max(0, slope_reference) ---------------------------------
    Dref, Dm = _table(criterion["depth"]["dz_per_mm"]), _table(mside["depth"]["dz_per_mm"])
    P, nk, nr = Dref.shape
    beyond = np.zeros(Dref.shape, bool)
    for j in range(P):
        beyond[j, k0[j]:, :] = True
    ok = beyond & np.isfinite(Dref) & np.isfinite(Dm)
    jj, kk, qq = np.nonzero(ok)
    bound = np.maximum(Dref[ok], 0.0)
    slack_z = float(criterion["sign_change_tolerance"]["depth_per_mm"])     # 1e-5 / mm: what counts as "no slope"
    if "zt" in names and len(jj):
        lo, hi = boxes(z[kk], z[kk + 1], rd[qq], rd[qq], xs[jj])
        Sz = _sup_chunks(prog, lo, hi, names.index("zt"), "dz", bound - Dm[ok], (4, 16, 64, 256))
    else:
        Sz = np.zeros(len(jj))
    tot = Dm[ok] + Sz
    exc = tot - bound
    w = int(np.argmax(exc)) if len(exc) else 0
    okz = bool(len(exc) == 0 or exc.max() <= slack_z)
    out["depth"] = {
        "rule": criterion["criterion"]["depth"], "passed": bool(okz and not joint_blind),
        "tolerance_per_mm": slack_z, "n_points": int(len(jj)), "n_violations": int(np.sum(exc > slack_z)),
        "n_reference_rising": int(np.sum(Dref[ok] > 0)), "n_skipped_nonfinite": int((beyond & ~ok).sum()),
        "largest_excess_per_mm": float(exc.max()) if len(exc) else 0.0,
        "model_max_dz_per_mm": float(tot.max()) if len(tot) else 0.0,
        "equation_sup_dz_per_mm": float(Sz.max()) if len(Sz) else 0.0,
        "would_pass_absolute_rule": bool(len(tot) == 0 or tot.max() <= slack_z),
        "z90_mm": list(criterion["depth"]["z90_mm"]), "z_max_mm": float(z[-1]),
        "worst": None if not len(exc) else {
            "p_GeV": float(mom[jj[w]]), "z_mm": [float(z[kk[w]]), float(z[kk[w] + 1])], "r_mm": float(rd[qq[w]]),
            "reference": float(Dref[ok][w]), "bound": float(bound[w]), "refitted_baseline": float(Dm[ok][w]),
            "equation_sup": float(Sz[w]), "model": float(tot[w])}}
    if not okz:
        ww = out["depth"]["worst"]
        reasons.append(f"depth tail: beyond z_90 the model's d ln rho / d z' can reach {ww['model']:.4g} / mm where "
                       f"the reference allows {ww['bound']:.4g} (reference slope {ww['reference']:.4g}) at p = "
                       f"{ww['p_GeV']:g} GeV, z' = {ww['z_mm'][0]:.0f}-{ww['z_mm'][1]:.0f} mm, r = {ww['r_mm']:.0f} mm; "
                       f"{out['depth']['n_violations']} of {len(jj)} points")
    # ---- radial, r in [200, 1500] mm -----------------------------------------------------------------------------
    Rref, Rm = _table(criterion["radial"]["loglog_slope"]), _table(mside["radial"]["loglog_slope"])
    okr = np.isfinite(Rref) & np.isfinite(Rm)
    jj, kk, ii = np.nonzero(okr)
    rbound = np.full(len(jj), limit) if rad_rule == "loglog_slope_le_limit" else Rref[okr]
    if "rh" in names and len(jj):
        lo, hi = boxes(z[kk], z[kk], rn[ii], rn[ii + 1], xs[jj])
        Sr = _sup_chunks(prog, lo, hi, names.index("rh"), "dlnr", rbound - Rm[okr], (4, 16, 64))
    else:
        Sr = np.zeros(len(jj))
    rtot = Rm[okr] + Sr
    rexc = rtot - rbound
    w = int(np.argmax(rexc)) if len(rexc) else 0
    slack_r = FLAT_SLACK if rad_rule == "loglog_slope_le_limit" else \
        float(criterion["sign_change_tolerance"]["radial_loglog"])          # 1e-3 against the reference's own slope
    rok = bool(len(rexc) == 0 or rexc.max() <= slack_r)
    rad: dict = {"rule": rad_rule, "slope_limit": limit, "tolerance": slack_r, "n_points": int(len(jj)),
                 "n_violations": int(np.sum(rexc > slack_r)), "r_range_mm": [float(rn[0]), float(rn[-1])],
                 "model_max_loglog_slope": float(rtot.max()) if len(rtot) else None,
                 "equation_sup_dlnr_at_registered_momenta": float(Sr.max()) if len(Sr) else 0.0,
                 "worst": None if not len(rexc) else {
                     "p_GeV": float(mom[jj[w]]), "z_mm": float(z[kk[w]]),
                     "r_interval_mm": [float(rn[ii[w]]), float(rn[ii[w] + 1])], "reference": float(Rref[okr][w]),
                     "bound": float(rbound[w]), "refitted_baseline": float(Rm[okr][w]), "equation_sup": float(Sr[w]),
                     "model": float(rtot[w])}}
    if not rok:
        ww = rad["worst"]
        reasons.append((f"radial tail: d ln rho / d ln r can reach {ww['model']:.4g} (needs <= {limit:g})"
                        if rad_rule == "loglog_slope_le_limit" else
                        f"radial tail: the model is flatter than the fixed reference, d ln rho / d ln r can reach "
                        f"{ww['model']:.4g} where the reference has {ww['reference']:.4g}") +
                       f" at p = {ww['p_GeV']:g} GeV, z' = {ww['z_mm']:.0f} mm, r in [{ww['r_interval_mm'][0]:.0f}, "
                       f"{ww['r_interval_mm'][1]:.0f}] mm; {rad['n_violations']} of {len(jj)} points")
    if rad_rule == "monotone_and_no_flattening":
        base_mono = bool(criterion["radial"]["baseline_monotone"])
        rad["reference_monotone"] = base_mono
        if not base_mono:
            rok = False
            reasons.append("radial tail: the reference baseline itself is not monotone in r beyond 200 mm (d_r ln rho "
                           "< 0 fails for delta = 0): to be reported, no correction can be certified under this rule")
        cont = tail_slopes(e, dict(box), r_from_mm=float(criterion["r_min_mm"]), z_from_mm=1e30)["dlnr"]
        rad.update(equation_sup_dlnr_continuum=float(cont["sup"]), equation_sup_dlnr_sampled=float(cont["sup_sampled"]),
                   enclosure_converged=cont.get("converged"),
                   continuum_region="r > %g mm, all z' and x of the audit box" % float(criterion["r_min_mm"]),
                   continuum_required=bool(measured is None))
        if measured is None and cont["sup"] > FLAT_SLACK:
            rok = False
            reasons.append(f"radial tail: the correction flattens the profile, d delta / d ln r can reach "
                           f"{cont['sup']:.4g} > 0 for r > {float(criterion['r_min_mm']):g} mm")
    rad["passed"] = bool(rok and not joint_blind)
    out["radial"] = rad
    # ---- sign changes per profile (evaluated on the record's grid) ----------------------------------------------------
    tol = criterion["sign_change_tolerance"]
    Fz = f_at(z[None, :, None], rd[None, None, :], xs[:, None, None])                 # (P, NZ, nr)
    mdz = Dm + np.diff(Fz, axis=1) / np.diff(z)[None, :, None]
    Fr = f_at(z[None, :, None], rn[None, None, :], xs[:, None, None])                 # (P, NZ, ni + 1)
    mdr = Rm + np.diff(Fr, axis=2) / np.diff(np.log(rn))[None, None, :]
    quantity = {"depth": "d ln rho / d z' beyond z_90, along depth at each (momentum, radius)",
                "radial": "d ln rho / d ln r for r > 200 mm, along r at each (momentum, depth row)"}
    depth_prof_m = [np.where(beyond[j], mdz[j], np.nan).T for j in range(P)]
    depth_prof_r = [np.where(beyond[j], Dref[j], np.nan).T for j in range(P)]

    def compare(m_, r_, axis_names, coords):
        more = m_ > r_
        wi = np.unravel_index(int(np.argmax(m_ - r_)), m_.shape)
        return {"model_total": int(m_.sum()), "reference_total": int(r_.sum()), "n_profiles": int(m_.size),
                "n_profiles_exceeding": int(more.sum()), "passed": bool(not more.any()),
                "model_per_momentum": m_.sum(1).tolist(), "reference_per_momentum": r_.sum(1).tolist(),
                "worst_profile": {axis_names[0]: float(coords[0][wi[0]]), axis_names[1]: float(coords[1][wi[1]]),
                                  "model": int(m_[wi]), "reference": int(r_[wi])}}

    # ---- enforced for Stage 2: rising stretches per depth profile; radial counted and reported ---------------------
    rs: dict = {"tolerance": {"depth_per_mm": tol["depth_per_mm"], "radial_loglog": tol["radial_loglog"]},
                "quantity": quantity, "enforced": bool(enforce_stage2),
                "definition": "per profile on the record's grid: non-finite and zero slopes skipped; lobes = maximal "
                              "runs of one sign; lobes with largest |slope| below the tolerance dropped; equal-sign "
                              "neighbours merged; a rising stretch = a surviving stretch of positive slope",
                "condition": "rising_stretches(model) <= rising_stretches(reference) for every depth profile"}
    rs["depth"] = compare(np.stack([_checks.profile_rising_lobes(v, tol["depth_per_mm"]) for v in depth_prof_m]),
                          np.stack([_checks.profile_rising_lobes(v, tol["depth_per_mm"]) for v in depth_prof_r]),
                          ("p_GeV", "r_mm"), (mom, rd))
    rs["radial"] = dict(compare(_checks.profile_rising_lobes(mdr, tol["radial_loglog"]),
                                _checks.profile_rising_lobes(Rref, tol["radial_loglog"]), ("p_GeV", "z_mm"), (mom, z)),
                        enforced=False, note="reported; not separately enforced: the certified radial slope rule "
                                             "already forbids a radial rise of the model where the reference is "
                                             "monotone (reading of this module, for confirmation)")
    rs["passed"] = bool(rs["depth"]["passed"])
    # ---- diagnostic only: raw sign-change counts of the same profiles ----------------------------------------------
    sc: dict = {"tolerance": dict(tol), "enforced": False, "decides": "nothing (diagnostic)", "quantity": quantity}
    sc["depth"] = compare(np.stack([_checks.profile_sign_changes(v, tol["depth_per_mm"]) for v in depth_prof_m]),
                          np.stack([_checks.profile_sign_changes(v, tol["depth_per_mm"]) for v in depth_prof_r]),
                          ("p_GeV", "r_mm"), (mom, rd))
    sc["radial"] = compare(_checks.profile_sign_changes(mdr, tol["radial_loglog"]),
                           _checks.profile_sign_changes(Rref, tol["radial_loglog"]), ("p_GeV", "z_mm"), (mom, z))
    sc["passed"] = bool(sc["depth"]["passed"] and sc["radial"]["passed"])

    def lines(d, keys):
        return [f"  {key:6s} total {d[key]['model_total']:5d} | {d[key]['reference_total']:5d}   profiles exceeding "
                f"{d[key]['n_profiles_exceeding']} of {d[key]['n_profiles']}   per momentum "
                f"{d[key]['model_per_momentum']} | {d[key]['reference_per_momentum']}" for key in keys]
    rs["text"] = "\n".join(
        ["rising stretches, model | reference (per depth profile: model <= reference " +
         ("REQUIRED)" if enforce_stage2 else "- Stage-2 condition, reported here, not enforced)")] +
        lines(rs, ("depth",)) + [ln + "   [reported, not enforced]" for ln in lines(rs, ("radial",))])
    sc["text"] = "\n".join(["sign changes, model | reference (diagnostic: decides nothing)"] +
                           lines(sc, ("depth", "radial")))
    out["rising_stretches"] = rs
    out["sign_changes"] = sc
    if enforce_stage2 and not rs["passed"]:
        wp = rs["depth"]["worst_profile"]
        reasons.append(f"rising stretches (depth): {rs['depth']['n_profiles_exceeding']} profile(s) have more rising "
                       f"stretches beyond z_90 than the reference (worst: {wp['model']} against {wp['reference']} at "
                       f"p = {wp['p_GeV']:g} GeV, r = {wp['r_mm']:.0f} mm)")
    out["passed"] = bool(out["depth"]["passed"] and rad["passed"] and (rs["passed"] or not enforce_stage2))
    out["reasons"] = reasons
    return out


def certify_finalist(expr: Union[str, T.Expr], box: Optional[Mapping[str, Tuple[float, float]]] = None, *,
                     tail_criterion: Optional[Mapping] = None, baseline_hash: Optional[str] = None,
                     measured_tail: Optional[Mapping] = None, baseline_frozen: Optional[bool] = None,
                     stage2: bool = False, limit: float = LIMIT_NAT, gauge: str = "oscillation",
                     gauge_vars: Sequence[str] = ("x",), domain_kw: Optional[dict] = None,
                     magnitude_kw: Optional[dict] = None, carried_stripped: Optional[Sequence[Mapping]] = None,
                     baseline_tail: Optional[Mapping] = None) -> dict:
    """The certificate a FINALIST needs. Call it on the equation WITH THE CONSTANTS AFTER THE JOINT REFIT
    (``refit.fit_candidate(..., finalist=True)`` does); a certificate of the pre-refit constants says nothing.

      1. ``audit(..., require_certified=True, stage="after refit")``: domain safe, magnitude bound PROVEN (status
         "certified", not merely "not_refuted"), raw magnitude proven, rounding within tolerance;
      2. tail check (``tail_check``) against the record of the FIXED reference baseline, ``tail_criterion`` =
         ``checks.decide_tail_criterion(...)``, bound to it by ``baseline_hash`` (the hash given here must equal
         the one in the record, and the record's own hash must be intact - otherwise ``ValueError``). A table of a
         jointly refitted baseline is refused as the reference; it belongs in ``measured_tail``, where it supplies
         the model side. ``baseline_frozen``: True if the baseline was not refitted (then no table is needed).
      3. ``stage2=True``: the Stage-2 shape condition is enforced as well -
         rising_stretches(model) <= rising_stretches(reference) for every depth profile beyond z_90 (PLAN section
         20, item 6); the raw sign-change counts are reported beside it and decide nothing.

    Returns {"certified", "audit_certified", "tail", "reasons", "audit", "equation", "hash", "baseline_hash",
    "tail_criterion_hash", "reference"}. Without ``tail_criterion`` the tail check cannot be evaluated and
    ``certified`` is False (reason given). ``baseline_tail`` (the two-number form of an earlier version) is
    refused."""
    if baseline_tail is not None:
        raise ValueError("baseline_tail is superseded (PLAN 17.11, S4): pass tail_criterion = "
                         "checks.decide_tail_criterion(...) and baseline_hash")
    e = T.parse(expr) if isinstance(expr, str) else expr
    if tail_criterion is not None:
        from . import checks as _checks                    # NumPy only; imported here to avoid a module cycle
        if not baseline_hash:
            raise ValueError("baseline_hash is required with tail_criterion: the criterion is bound to one baseline "
                             "and must be checked against the baseline in hand")
        _checks.verify_tail_criterion(tail_criterion, baseline_hash, as_reference=True)
        if measured_tail is not None:
            _checks.verify_tail_criterion(measured_tail, as_reference=False)
    rep = audit(e, box, limit=limit, gauge=gauge, gauge_vars=gauge_vars, domain_kw=domain_kw,
                magnitude_kw=magnitude_kw, require_certified=True, stage="after refit",
                carried_stripped=carried_stripped)
    reasons: List[str] = []
    if not rep.domain["safe"]:
        reasons.append(f"domain: {rep.domain['status']}")
    else:
        if rep.magnitude.get("status") != "certified":
            reasons.append(f"magnitude bound {rep.magnitude.get('status')}: a finalist needs it certified")
        if not rep.numerics.get("certified", False):
            reasons.append("numerical hygiene not certified (raw magnitude / rounding)")
    tail: dict = {"evaluated": False, "passed": False}
    if rep.domain["safe"]:
        if tail_criterion is None:
            reasons.append("tail check not evaluated: the criterion record of the reference baseline is needed "
                           "(checks.decide_tail_criterion on the machine that holds the baseline)")
        else:
            tail = dict(tail_check(e, tail_criterion, box, measured=measured_tail, baseline_frozen=baseline_frozen,
                                   enforce_stage2=stage2), evaluated=True)
            reasons += tail["reasons"]
    return {"certified": bool(rep.passed and tail.get("passed", False)), "audit_certified": bool(rep.passed),
            "tail": tail, "reasons": reasons, "audit": rep.to_json(), "equation": T.to_string(e),
            "hash": T.expr_hash(e), "baseline_hash": baseline_hash, "stage2": bool(stage2),
            "tail_criterion_hash": None if tail_criterion is None else tail_criterion.get("record_hash"),
            "reference": None if tail_criterion is None else {
                "kind": tail_criterion.get("reference_kind"), "label": tail_criterion.get("label"),
                "baseline_hash": tail_criterion.get("baseline_hash"),
                "measured_table": None if measured_tail is None else measured_tail.get("record_hash")}}


def baseline_against_reference(criterion: Mapping, measured: Mapping, *, baseline_hash: str,
                               stage2: bool = False) -> dict:
    """Test a baseline WITHOUT any correction (delta = 0) against the fixed reference: the control C1 (the baseline
    refitted alone with the candidates' schedule), or any jointly refitted baseline taken by itself.

    criterion   the record of the reference of record - the frozen Track A finalist before any PySR refit (PLAN
                section 20, item 4) - bound to ``baseline_hash``
    measured    the slope table of the baseline to test, made on the same grid with
                ``checks.decide_tail_criterion(..., reference_kind="refitted_baseline")`` (for a closed-form model:
                ``refit.c1_against_reference`` builds it and calls this function)
    Runs ``tail_check`` with the zero correction: the inheriting depth rule, the radial rule and (``stage2``) the
    rising-stretch condition, all on the baseline's own slopes. If C1 fails, the schedule bends the baseline, not
    the equation: report it beside every SR verdict. Returns the ``tail_check`` result plus the two record hashes."""
    from . import checks as _checks

    if not baseline_hash:
        raise ValueError("baseline_hash is required: the reference is bound to one baseline")
    _checks.verify_tail_criterion(criterion, baseline_hash, as_reference=True)
    _checks.verify_tail_criterion(measured, as_reference=False)
    res = tail_check(T.const(0.0, fixed=True), criterion, measured=measured, baseline_frozen=False,
                     enforce_stage2=stage2)
    res.update(correction="zero (delta = 0): the tested baseline alone", stage2=bool(stage2),
               tested_table=measured.get("record_hash"), tested_label=measured.get("label"),
               tested_baseline_hash=measured.get("baseline_hash"))
    return res


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
