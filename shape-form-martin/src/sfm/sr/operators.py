"""sfm.sr.operators - the single source of truth for the PySR vocabularies of Milestone 3 (PLAN.md section 7).

Plain Python data; imports neither ``pysr`` / ``juliacall`` nor ``torch`` (torch implementations are built on demand
by ``torch_impl()``). The search runner passes ``search_config(...)`` to ``PySRRegressor`` (PySR 2.7.3: custom
operators are Julia definition strings, constraints are dictionaries keyed by the operator *name*).

Contents
    OPS                     name -> OpSpec: Julia string, complexity, mathematical range, domain, smoothness
    numpy_impl / torch_impl name -> callable; the same formulas as the Julia strings (see "Agreement" below)
    sympy_mappings()        name -> sympy callable for the custom operators (``extra_sympy_mappings``)
    STAGE1, STAGE1_RAW, STAGE2   Vocabulary objects: operator lists, ``constraints``, ``nested_constraints``,
                            ``complexity_of_operators`` in exactly the form PySR expects
    TEMPLATES               TemplateSpec objects (combine string, sub-expression names and their arguments)
    violates_nesting / violates_constraints / violations   pure-Python scans of an expression tree
    tree_complexity         operator-weighted size C, the same weights as the search
    WL2_ELEMENTWISE, gauge_loss_julia(), wl2_loss_numpy(), gauge_loss_numpy()   losses and their references
    MAXSIZE_1D / MAXSIZE_ND / PARETO_TIERS, VARIABLE_NAMES, scales, the extended audit box, x_of_p()

Variables of the search (dimensionless, O(1) on the data)
    zt = z' / 260 mm   (zeta)        rh = r / 21.9 mm   (varrho)        x = (ln T - v0) / vs
    ``zeta`` itself cannot be used as a name: it is a sympy function and PySR refuses it.

Stage 1 - pole-free by construction
    binary  +  -  *  ppow(x, y) = (1 + |x|)^y           (exponent: complexity-1 argument, see below)
    unary   exp  square  tanh  atan  erf  erfc  softplus  l1p  sqa  linv
    Every operator is defined and continuous on all of R (R^2), so no composition has a pole or a forbidden sign.
    What can still go wrong is *overflow* (exp, square, ppow, products) - found by the audit, not by construction.

Stage 1 extra arm (labelled STAGE1_RAW) adds raw ``/`` and ``inv``. "Protected" means nothing there: there is no
    epsilon in a denominator and no clamp anywhere. Inside the search SymbolicRegression.jl discards a candidate
    whose evaluation is non-finite *on a data row*; a pole between rows is invisible to it (this is how
    ``c/(r - 0.017)`` was found in the earlier attempt). Only ``sfm.sr.audit`` decides.

Stage 2 adds sin, cos: never inside any other unary operator or ``ppow`` (they combine through + - * only),
    no trig / exp / ppow / square inside a trig argument, argument size <= 9.

Overflow (where, and what happens)
    exp      x > 709.78 (float64), x > 88.72 (float32)          -> +Inf in Julia, NumPy and torch alike
    square   |x| > 1.34e154 (float64), |x| > 1.84e19 (float32)   -> +Inf
    ppow     y * log1p(|x|) > 709.78 / 88.72                     -> +Inf     (underflow to 0 is harmless)
    The search (Float32 by default, Float64 with ``precision=64``) treats any non-finite intermediate on a data row
    as an invalid candidate. Off the rows nothing protects: the audit bounds every *intermediate* node by interval
    arithmetic and rejects an expression if a node can exceed ``audit.OVERFLOW_CAP`` (1e30, eight decades below
    the float32 limit 3.4e38) anywhere in the extended box, because the refit runs in float32 without clamps and
    Inf * 0 or Inf - Inf would give NaN.

Agreement of the three implementations
    The NumPy and torch functions use literally the formulas of the Julia strings (``softplus`` is *not*
    ``F.softplus``, whose threshold shortcut differs by 2e-9 relative). exp, tanh, atan, erf, erfc, log1p, sqrt and
    pow come from each library's libm and agree to a few ulp; ``tests/test_sr_operators.py`` checks NumPy against
    torch at 1e-13 relative over |x| from 1e-300 to 1e300 including the overflow points. Agreement with Julia itself:
    ``tests/test_sr_translate.py`` compares with PySR's own predictions on exported rows (1e-12, every operator of
    the three vocabularies).

Complexity (``complexity_of_operators``): PLAN "transcendental 2, erf/erfc 3". + - * square: 1; exp tanh atan
    softplus l1p ppow sin cos: 2; erf erfc: 3. Two choices beyond the plan's sentence: ``sqa`` and ``linv`` are
    algebraic but composite (sqrt(|x|) hides an abs, linv hides 1, +, *, /), so they cost 2 as well. Raw ``/`` and
    ``inv`` cost 1. Constants and variables cost 1.

Confirmed against Julia on 25 Nov 2026: see ``CONFIRMED_WITH_JULIA`` (``UNCONFIRMED`` is now empty).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import special as _sp

__all__ = [
    "OpSpec", "OPS", "Vocabulary", "TemplateSpec", "STAGE1", "STAGE1_RAW", "STAGE2", "VOCABULARIES", "TEMPLATES",
    "numpy_impl", "torch_impl", "sympy_mappings", "tree_complexity", "violates_nesting", "violates_constraints",
    "violations", "search_config", "WL2_ELEMENTWISE", "gauge_loss_julia", "wl2_loss_numpy", "gauge_loss_numpy",
    "MAXSIZE_1D", "MAXSIZE_ND", "PARETO_TIERS", "VARIABLE_NAMES", "ZETA_SCALE_MM", "RHO_SCALE_MM", "M_PION",
    "V0_PION", "VS_PION", "x_of_p", "extended_box", "data_box", "TRAJECTORY_BOX", "TRAIN_X_RANGE", "pareto_tier",
    "SATURATING", "UNCONFIRMED", "CONFIRMED_WITH_JULIA", "AUDIT_ONLY_OPS",
]

# ----------------------------------------------------------------------------------------------------------------------
# Variables, scales, boxes
# ----------------------------------------------------------------------------------------------------------------------
VARIABLE_NAMES: Tuple[str, str, str] = ("zt", "rh", "x")
ZETA_SCALE_MM = 260.0                # zt = z' / 260 mm
RHO_SCALE_MM = 21.9                  # rh = r / 21.9 mm
M_PION = 0.139570                    # GeV
V0_PION = -0.325860                  # centre of ln T over the training momentum range
VS_PION = 3.314613                   # half-width of ln T over the training momentum range
TRAIN_P_MIN, TRAIN_P_MAX = 0.08951619267463684, 19.9999942779541     # GeV (sfm.models.field)
TRAIN_X_RANGE: Tuple[float, float] = (-1.0, 1.0)
TRAJECTORY_BOX: Tuple[float, float] = (-4.2, 1.4)                     # 1-D audit range in x

MAXSIZE_1D = 18                      # 1-D trajectory searches
MAXSIZE_ND = 30                      # 2-D / 3-D searches
PARETO_TIERS: Tuple[int, int, int] = (12, 20, 30)


def x_of_p(p, m: float = M_PION, v0: float = V0_PION, vs: float = VS_PION):
    """x = (ln T - v0) / vs with T = p^2 / (sqrt(p^2 + m^2) + m) (GeV). x(0.0005) = -4.1030, x(0.0895) = -1.0001,
    x(20) = +1.0000, x(50) = +1.2777."""
    p = np.asarray(p, dtype=np.float64)
    T = p * p / (np.sqrt(p * p + m * m) + m)
    return (np.log(T) - v0) / vs


def extended_box(z_mm=(-300.0, 2000.0), r_mm=(0.0, 1500.0), x=(-4.11, 1.33)) -> Dict[str, Tuple[float, float]]:
    """The audit box in the search variables. Momentum 0.0005 ... 50 GeV is x = -4.1030 ... +1.2777; the default
    x-range (-4.11, 1.33) contains it with a margin (x = 1.33 is p = 59 GeV)."""
    return {"zt": (z_mm[0] / ZETA_SCALE_MM, z_mm[1] / ZETA_SCALE_MM),
            "rh": (r_mm[0] / RHO_SCALE_MM, r_mm[1] / RHO_SCALE_MM), "x": (float(x[0]), float(x[1]))}


def data_box() -> Dict[str, Tuple[float, float]]:
    """The region that holds data: the grid of ``common.py`` (z' -200 ... 1450 mm, r < 700 mm) and the training
    momenta (x in [-1, 1])."""
    return extended_box((-200.0, 1450.0), (0.0, 700.0), TRAIN_X_RANGE)


def pareto_tier(complexity: int) -> Optional[int]:
    """The smallest reporting tier (12, 20, 30) that contains ``complexity``; None above 30."""
    for t in PARETO_TIERS:
        if complexity <= t:
            return t
    return None


# ----------------------------------------------------------------------------------------------------------------------
# Operator table
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    """One operator.

    name        the name that appears in PySR's equation strings (infix symbol for + - * /)
    arity       1 or 2
    julia       the string given to PySR: a built-in name, or a Julia definition for a custom operator
    builtin     True if SymbolicRegression.jl knows the name
    complexity  operator weight in ``complexity_of_operators``
    range       closed hull (lo, hi) of the mathematical range over the whole domain
    range_note  which ends are attained
    total       defined and finite-valued on all of R^arity (no pole, no forbidden sign)
    domain      the forbidden set of the argument(s) if not total
    smooth      False if the function has a kink or an infinite slope somewhere (stated in ``range_note``)
    group       "arith" | "growth" | "saturating" | "protected" | "trig" | "raw" | "audit_only"
    """
    name: str
    arity: int
    julia: str
    builtin: bool
    complexity: int
    range: Tuple[float, float]
    range_note: str
    total: bool = True
    domain: str = "all reals"
    smooth: bool = True
    group: str = "arith"


_INF = math.inf
_HALF_PI = math.pi / 2

OPS: Dict[str, OpSpec] = {o.name: o for o in [
    # --- arithmetic --------------------------------------------------------------------------------------------------
    OpSpec("+", 2, "+", True, 1, (-_INF, _INF), "unbounded"),
    OpSpec("-", 2, "-", True, 1, (-_INF, _INF), "unbounded"),
    OpSpec("*", 2, "*", True, 1, (-_INF, _INF), "unbounded"),
    # --- growth ------------------------------------------------------------------------------------------------------
    OpSpec("exp", 1, "exp", True, 2, (0.0, _INF), "(0, inf): 0 not attained; overflows above 709.78 / 88.72",
           group="growth"),
    OpSpec("square", 1, "square", True, 1, (0.0, _INF), "[0, inf): 0 attained at 0", group="growth"),
    # --- saturating --------------------------------------------------------------------------------------------------
    OpSpec("tanh", 1, "tanh", True, 2, (-1.0, 1.0), "(-1, 1) open; +-1 reached in floating point for |x| > 19.1",
           group="saturating"),
    OpSpec("atan", 1, "atan", True, 2, (-_HALF_PI, _HALF_PI), "(-pi/2, pi/2) open", group="saturating"),
    OpSpec("erf", 1, "erf", True, 3, (-1.0, 1.0), "(-1, 1) open; +-1 in floating point for |x| > 5.93",
           group="saturating"),
    OpSpec("erfc", 1, "erfc", True, 3, (0.0, 2.0), "(0, 2) open; underflows to 0 for x > 26.6", group="saturating"),
    OpSpec("softplus", 1, "softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))", False, 2, (0.0, _INF),
           "(0, inf): 0 not attained; softplus(x) - x -> 0 for large x, never overflows", group="saturating"),
    OpSpec("linv", 1, "linv(x) = 1 / (1 + x * x)", False, 2, (0.0, 1.0),
           "(0, 1]: 1 attained at 0, 0 not attained (underflows for |x| > 1e154)", group="saturating"),
    # --- protected forms of pow / log / sqrt -------------------------------------------------------------------------
    OpSpec("ppow", 2, "ppow(x, y) = (1 + abs(x))^y", False, 2, (0.0, _INF),
           "(0, inf): base >= 1, so >= 1 for y >= 0 and in (0, 1] for y <= 0; kink in x at 0", smooth=False,
           group="protected"),
    OpSpec("l1p", 1, "l1p(x) = log1p(abs(x))", False, 2, (0.0, _INF), "[0, inf): 0 attained at 0; kink at 0 "
           "(slope jumps from -1 to +1)", smooth=False, group="protected"),
    OpSpec("sqa", 1, "sqa(x) = sqrt(abs(x))", False, 2, (0.0, _INF), "[0, inf): 0 attained at 0; infinite slope "
           "at 0 (torch_impl defines the backward pass as 0 at exactly 0; plain autograd would give NaN)",
           smooth=False, group="protected"),
    # --- Stage 2 -----------------------------------------------------------------------------------------------------
    OpSpec("sin", 1, "sin", True, 2, (-1.0, 1.0), "[-1, 1] attained", group="trig"),
    OpSpec("cos", 1, "cos", True, 2, (-1.0, 1.0), "[-1, 1] attained", group="trig"),
    # --- labelled extra arm ------------------------------------------------------------------------------------------
    OpSpec("/", 2, "/", True, 1, (-_INF, _INF), "unbounded; pole where the denominator is 0", total=False,
           domain="denominator != 0", group="raw"),
    OpSpec("inv", 1, "inv", True, 1, (-_INF, _INF), "unbounded; pole at 0", total=False, domain="argument != 0",
           group="raw"),
    # --- never searched: accepted by the parser and the audit so that foreign equations can be judged ---------------
    OpSpec("neg", 1, "neg", True, 1, (-_INF, _INF), "unbounded", group="audit_only"),
    OpSpec("abs", 1, "abs", True, 1, (0.0, _INF), "[0, inf); kink at 0", smooth=False, group="audit_only"),
    OpSpec("cube", 1, "cube", True, 1, (-_INF, _INF), "unbounded", group="audit_only"),
    OpSpec("log", 1, "log", True, 2, (-_INF, _INF), "unbounded", total=False, domain="argument > 0",
           group="audit_only"),
    OpSpec("log1p", 1, "log1p", True, 2, (-_INF, _INF), "unbounded", total=False, domain="argument > -1",
           group="audit_only"),
    OpSpec("sqrt", 1, "sqrt", True, 1, (0.0, _INF), "[0, inf); infinite slope at 0", total=False,
           domain="argument > 0 (0 excluded: the slope is infinite there)", smooth=False, group="audit_only"),
    OpSpec("pow", 2, "^", True, 2, (-_INF, _INF), "x^y; a fixed integer exponent is an integer power, any other "
           "exponent needs a positive base", total=False,
           domain="base > 0 (integer power with negative exponent: base != 0)", group="audit_only"),
]}

SATURATING: Tuple[str, ...] = ("tanh", "atan", "erf", "erfc", "softplus", "linv")
AUDIT_ONLY_OPS: Tuple[str, ...] = tuple(n for n, o in OPS.items() if o.group == "audit_only")


# ----------------------------------------------------------------------------------------------------------------------
# NumPy / torch / sympy implementations
# ----------------------------------------------------------------------------------------------------------------------
def _np_softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _np_ppow(x, y):
    return np.power(1.0 + np.abs(x), y)


_NUMPY: Dict[str, Callable] = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide,
    "exp": np.exp, "square": np.square, "tanh": np.tanh, "atan": np.arctan, "erf": _sp.erf, "erfc": _sp.erfc,
    "softplus": _np_softplus, "linv": lambda x: 1.0 / (1.0 + x * x), "ppow": _np_ppow,
    "l1p": lambda x: np.log1p(np.abs(x)), "sqa": lambda x: np.sqrt(np.abs(x)),
    "sin": np.sin, "cos": np.cos, "inv": lambda x: 1.0 / x,
    "neg": np.negative, "abs": np.abs, "cube": lambda x: x * x * x, "log": np.log, "log1p": np.log1p,
    "sqrt": np.sqrt, "pow": np.power,
}


def numpy_impl(name: Optional[str] = None):
    """NumPy implementation of operator ``name`` (the whole table if ``name`` is None). No error state is changed:
    callers that expect overflow wrap the call in ``np.errstate``."""
    return dict(_NUMPY) if name is None else _NUMPY[name]


_TORCH_CACHE: Dict[str, Callable] = {}


def torch_impl(name: Optional[str] = None):
    """torch implementation of operator ``name`` (the whole table if None); torch is imported here, not at module
    import. The formulas are those of the Julia strings; no clamp, no ``nan_to_num``: every forward value is the
    formula, non-finite inputs propagate.

    Gradients at exactly 0 (and -0.0) are finite for every total operator: ``abs``, ``l1p`` and ``ppow`` (kinks) give
    0 there through ``sign(0) = 0``; ``sqa`` (cusp, infinite one-sided slopes) has its backward pass defined as 0 at
    0 instead of autograd's ``inf * 0 = NaN``; ``softplus`` has the exact derivative ``sigmoid(x)`` everywhere.
    Next to 0 the slope of ``sqa`` is 1 / (2 sqrt|x|): finite for every non-zero float, but arbitrarily large."""
    if not _TORCH_CACHE:
        import torch

        class _Softplus(torch.autograd.Function):
            """Forward: literally the Julia formula. Backward: the exact derivative sigmoid(x). Autograd through
            the formula itself gives 0 at exactly x = 0 (sign(0) = 0 in both pieces) instead of 1/2."""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return torch.clamp_min(x, 0.0) + torch.log1p(torch.exp(-torch.abs(x)))     # max(x, 0), not a guard

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return grad_output * torch.sigmoid(x)

        def sqa(x):
            # Forward: sqrt(|x|), bit for bit (NaN and Inf propagate: "a != 0" is True for NaN).
            # Backward: sign(x) / (2 sqrt|x|) for x != 0 and 0 AT exactly x = +-0, where autograd through
            # sqrt(abs(x)) would give inf * 0 = NaN. Only the derivative at the cusp is defined here; the value
            # is the formula. (0 lies in the generalised gradient of sqrt|x| at 0 and is its symmetric derivative.)
            a = torch.abs(x)
            nz = a != 0
            return torch.where(nz, torch.sqrt(torch.where(nz, a, torch.ones_like(a))), torch.zeros_like(a))

        _TORCH_CACHE.update({
            "+": torch.add, "-": torch.sub, "*": torch.mul, "/": torch.div,
            "exp": torch.exp, "square": torch.square, "tanh": torch.tanh, "atan": torch.atan,
            "erf": torch.special.erf, "erfc": torch.special.erfc, "softplus": _Softplus.apply,
            "linv": lambda x: 1.0 / (1.0 + x * x), "ppow": lambda x, y: torch.pow(1.0 + torch.abs(x), y),
            "l1p": lambda x: torch.log1p(torch.abs(x)), "sqa": sqa,
            "sin": torch.sin, "cos": torch.cos, "inv": lambda x: 1.0 / x,
            "neg": torch.neg, "abs": torch.abs, "cube": lambda x: x * x * x, "log": torch.log,
            "log1p": torch.log1p, "sqrt": torch.sqrt, "pow": torch.pow,
        })
    return dict(_TORCH_CACHE) if name is None else _TORCH_CACHE[name]


def sympy_mappings(names: Optional[Iterable[str]] = None) -> Dict[str, Callable]:
    """``extra_sympy_mappings`` for the custom operators (PySR already maps the built-ins). sympy is imported here."""
    import sympy

    table = {
        "ppow": lambda x, y: (1 + sympy.Abs(x)) ** y,
        "softplus": lambda x: sympy.log(1 + sympy.exp(x)),
        "l1p": lambda x: sympy.log(1 + sympy.Abs(x)),
        "sqa": lambda x: sympy.sqrt(sympy.Abs(x)),
        "linv": lambda x: 1 / (1 + x ** 2),
    }
    return table if names is None else {n: table[n] for n in names if n in table}


# ----------------------------------------------------------------------------------------------------------------------
# Trees: a minimal protocol shared with sfm.sr.translate
# ----------------------------------------------------------------------------------------------------------------------
# A node is any object with ``kind`` in {"const", "var", "param", "op"}, ``name`` (operator / variable name) and
# ``args`` (tuple of child nodes) - ``sfm.sr.translate.Expr`` - or, for hand-written tests, a nested tuple
# ``("tanh", ("+", "zt", 1.5))`` whose leaves are strings (variables) and numbers (constants).
def _kind(node) -> str:
    if isinstance(node, tuple):
        return "op"
    if isinstance(node, str):
        return "var"
    if isinstance(node, (int, float)):
        return "const"
    return node.kind


def _name(node) -> str:
    if isinstance(node, tuple):
        return node[0]
    if isinstance(node, str):
        return node
    if isinstance(node, (int, float)):
        return ""
    return node.name


def _args(node) -> tuple:
    if isinstance(node, tuple):
        return tuple(node[1:])
    if isinstance(node, (str, int, float)):
        return ()
    return tuple(node.args)


def _walk(node):
    yield node
    for a in _args(node):
        yield from _walk(a)


COMPLEXITY_OF_OPERATORS_ALL: Dict[str, int] = {n: o.complexity for n, o in OPS.items()}


def tree_complexity(node, complexity_of_operators: Optional[Mapping[str, float]] = None,
                    complexity_of_constants: float = 1, complexity_of_variables: float = 1) -> int:
    """Operator-weighted size C of a tree: the sum of the node weights, rounded to the nearest integer as
    SymbolicRegression.jl does. Default weights: this module's table (the search's ``complexity_of_operators``)."""
    w = COMPLEXITY_OF_OPERATORS_ALL if complexity_of_operators is None else complexity_of_operators
    total = 0.0
    for n in _walk(node):
        k = _kind(n)
        if k == "op":
            total += w.get(_name(n), 1)
        elif k == "var":
            total += complexity_of_variables
        else:
            total += complexity_of_constants
    return int(math.floor(total + 0.5))


def _max_nestedness(node, inner: str) -> int:
    """SymbolicRegression.jl ``count_max_nestedness``: the largest number of ``inner`` nodes on a root-to-leaf path
    of the subtree at ``node``, not counting ``node`` itself."""
    def depth(n) -> int:
        here = 1 if (_kind(n) == "op" and _name(n) == inner) else 0
        ch = _args(n)
        return here + (max(depth(c) for c in ch) if ch else 0)
    full = depth(node)
    return full - (1 if (_kind(node) == "op" and _name(node) == inner) else 0)


def violates_nesting(expr_tree, nested_constraints: Optional[Mapping[str, Mapping[str, int]]] = None) -> List[str]:
    """Violations of ``nested_constraints`` in a tree, with SymbolicRegression.jl's semantics
    (``flag_illegal_nests``): for every node whose operator is ``outer``, the number of ``inner`` operators nested
    below it (largest count on any path, the node itself excluded) must not exceed the limit.
    Returns a list of messages, empty if the tree complies. Default constraints: Stage 1."""
    nc = STAGE1.nested_constraints if nested_constraints is None else nested_constraints
    out: List[str] = []
    for n in _walk(expr_tree):
        if _kind(n) != "op":
            continue
        rules = nc.get(_name(n))
        if not rules:
            continue
        for inner, limit in rules.items():
            got = _max_nestedness(n, inner)
            if got > limit:
                out.append(f"{inner} nested {got}x inside {_name(n)} (limit {limit})")
    return out


def violates_constraints(expr_tree, constraints: Optional[Mapping[str, object]] = None,
                         complexity_of_operators: Optional[Mapping[str, float]] = None,
                         constant_exponent: bool = True) -> List[str]:
    """Violations of the argument-size limits (``constraints``: an int for a unary operator, a tuple for a binary
    one; -1 = no limit), sizes counted with the operator weights. With ``constant_exponent`` a ``ppow`` whose
    exponent is not a constant is reported too: PySR's size limit 1 admits a single constant *or* a single
    variable, so this part of the PLAN can only be enforced by this scan."""
    cons = STAGE1.constraints if constraints is None else constraints
    out: List[str] = []
    for n in _walk(expr_tree):
        if _kind(n) != "op":
            continue
        name, args = _name(n), _args(n)
        lim = cons.get(name)
        if lim is not None:
            lims = (lim,) if isinstance(lim, int) else tuple(lim)
            if len(lims) != len(args):
                out.append(f"{name}: constraint {lim!r} does not match arity {len(args)}")
            else:
                for i, (a, m) in enumerate(zip(args, lims)):
                    if m >= 0:
                        c = tree_complexity(a, complexity_of_operators)
                        if c > m:
                            out.append(f"{name}: argument {i + 1} has complexity {c} (limit {m})")
        if constant_exponent and name == "ppow" and len(args) == 2 and _kind(args[1]) not in ("const", "param"):
            out.append("ppow: exponent is not a constant")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Vocabularies
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Vocabulary:
    """An operator set with its constraints, in the form ``PySRRegressor`` expects."""
    name: str
    label: str
    binary: Tuple[str, ...]
    unary: Tuple[str, ...]
    constraints: Dict[str, object]
    nested_constraints: Dict[str, Dict[str, int]]
    pole_free: bool

    @property
    def names(self) -> Tuple[str, ...]:
        return self.binary + self.unary

    def binary_operators(self) -> List[str]:
        return [OPS[n].julia for n in self.binary]

    def unary_operators(self) -> List[str]:
        return [OPS[n].julia for n in self.unary]

    def complexity_of_operators(self) -> Dict[str, int]:
        """Only the weights that differ from PySR's default 1."""
        return {n: OPS[n].complexity for n in self.names if OPS[n].complexity != 1}

    def extra_sympy_mappings(self) -> Dict[str, Callable]:
        return sympy_mappings([n for n in self.names if not OPS[n].builtin])

    def violations(self, expr_tree) -> List[str]:
        return violations(expr_tree, self)


_UNARY_S1 = ("exp", "square", "tanh", "atan", "erf", "erfc", "softplus", "l1p", "sqa", "linv")
_BINARY_S1 = ("+", "-", "*", "ppow")

# Argument-size limits (weighted complexity of the argument). An affine form in three variables,
# a*zt + b*rh + c*x + d, has size 13; in two variables 9; a*(v - b) squared has size 6.
_CONSTRAINTS_S1: Dict[str, object] = {
    "ppow": (7, 1),          # base: an affine form in one variable (5) with room for one more node; exponent: size 1
    "exp": 9,                # e.g. c * square((zt - a) * b), or an affine form in two variables
    "square": 7,
    "tanh": 13, "atan": 13, "erf": 13, "erfc": 13, "softplus": 13,
    "linv": 9,
    "l1p": 7, "sqa": 7,
}


def _nested_stage1() -> Dict[str, Dict[str, int]]:
    nc: Dict[str, Dict[str, int]] = {}
    for o in SATURATING:                                   # never inside each other, never wrapping exp / ppow
        nc[o] = {i: 0 for i in SATURATING}
        nc[o].update({"exp": 0, "ppow": 0})
    nc["exp"] = {"exp": 0}                                 # no exp in exp
    for o in ("ppow", "l1p", "sqa", "square"):             # not self-nested
        nc[o] = {o: 0}
    return nc


def _nested_stage2() -> Dict[str, Dict[str, int]]:
    nc = _nested_stage1()
    for o in _UNARY_S1 + ("ppow",):                        # trig never inside another operator: only + - * above it
        nc.setdefault(o, {}).update({"sin": 0, "cos": 0})
    for t in ("sin", "cos"):                               # phase: no trig, nothing that makes a chirp
        nc[t] = {"sin": 0, "cos": 0, "exp": 0, "ppow": 0, "square": 0}
    return nc


def _nested_raw() -> Dict[str, Dict[str, int]]:
    nc = _nested_stage1()
    nc["inv"] = {"inv": 0, "/": 0}
    nc["/"] = {"/": 1, "inv": 0}
    return nc


STAGE1 = Vocabulary(
    name="stage1", label="Stage 1 (pole-free by construction)", binary=_BINARY_S1, unary=_UNARY_S1,
    constraints=dict(_CONSTRAINTS_S1), nested_constraints=_nested_stage1(), pole_free=True)

STAGE1_RAW = Vocabulary(
    name="stage1_raw", label="EXTRA ARM: Stage 1 + raw '/' and 'inv' (no protection; the audit decides)",
    binary=_BINARY_S1 + ("/",), unary=_UNARY_S1 + ("inv",),
    constraints={**_CONSTRAINTS_S1, "/": (-1, 9), "inv": 9}, nested_constraints=_nested_raw(), pole_free=False)

STAGE2 = Vocabulary(
    name="stage2", label="Stage 2 (Stage 1 + sin, cos; gated ablation)", binary=_BINARY_S1,
    unary=_UNARY_S1 + ("sin", "cos"), constraints={**_CONSTRAINTS_S1, "sin": 9, "cos": 9},
    nested_constraints=_nested_stage2(), pole_free=True)

VOCABULARIES: Dict[str, Vocabulary] = {v.name: v for v in (STAGE1, STAGE1_RAW, STAGE2)}


def violations(expr_tree, vocabulary: Vocabulary = STAGE1) -> List[str]:
    """Everything a hall-of-fame member can violate: operators outside the vocabulary, nesting, argument sizes,
    non-constant ``ppow`` exponents. Empty list = compliant. The Gate 2-pilot scan runs this on every member."""
    allowed = set(vocabulary.names)
    out = [f"operator {_name(n)!r} is not in vocabulary {vocabulary.name}"
           for n in _walk(expr_tree) if _kind(n) == "op" and _name(n) not in allowed]
    out += violates_nesting(expr_tree, vocabulary.nested_constraints)
    out += violates_constraints(expr_tree, vocabulary.constraints, vocabulary.complexity_of_operators())
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Templates (PySR ``TemplateExpressionSpec``)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TemplateSpec:
    """A template expression: ``combine`` is Julia source over the sub-expressions, written in the subset that
    ``sfm.sr.translate.parse_template`` reads (calls, + - * / ^, numeric literals, optional ``name = expr;``
    statements). ``arguments[name]`` lists the data variables the sub-expression is called with: in PySR's output
    string its ``#1, #2, ...`` are these, in this order.

    Numeric literals of ``combine`` are structural (the 0.3 cap of the modulation): the translator marks them as
    fixed, so they are neither trained in the refit nor counted in k.
    """
    name: str
    label: str
    expressions: Tuple[str, ...]
    arguments: Dict[str, Tuple[str, ...]]
    combine64: str
    vocabulary: str = "stage1"
    variable_names: Tuple[str, ...] = VARIABLE_NAMES
    modulation: Optional[str] = None           # the part of ``combine`` that is the Stage-2 modulation (documentation)
    bound: Optional[float] = None              # sup |value| guaranteed by the form itself, if any

    def combine(self, precision: int = 64) -> str:
        """The combine string for the data precision: Float32 searches need Float32 literals (``0.3f0``)."""
        if precision == 64:
            return self.combine64
        if precision != 32:
            raise ValueError("precision must be 32 or 64")
        import re
        return re.sub(r"(?<![\w.])(\d+\.\d+)(?![\w.])", r"\1f0", self.combine64)

    def pysr_kwargs(self, precision: int = 64) -> Dict[str, object]:
        """Keyword arguments of ``pysr.TemplateExpressionSpec``."""
        return {"combine": self.combine(precision), "expressions": list(self.expressions),
                "variable_names": list(self.variable_names)}


TEMPLATES: Dict[str, TemplateSpec] = {t.name: t for t in [
    TemplateSpec(
        name="s1_tanh_sum", label="Stage 1: tanh(u(z,x) + v(r,x) + w(z,r))", expressions=("u", "v", "w"),
        arguments={"u": ("zt", "x"), "v": ("rh", "x"), "w": ("zt", "rh")},
        combine64="tanh(u(zt, x) + v(rh, x) + w(zt, rh))", bound=1.0),
    TemplateSpec(
        name="s1_amp_shape", label="Stage 1: a(x) * f(z,r)", expressions=("a", "f"),
        arguments={"a": ("x",), "f": ("zt", "rh")}, combine64="a(x) * f(zt, rh)"),
    TemplateSpec(
        name="s2_modulation",
        label="Stage 2 (PLAN, literal): tanh(f) + 0.3 * tanh(a) * envelope * sin(phase); |value| <= 1.3",
        expressions=("f", "a", "e", "ph"),
        arguments={"f": ("zt", "rh", "x"), "a": ("x",), "e": ("zt", "rh"), "ph": ("zt", "rh", "x")},
        combine64="tanh(f(zt, rh, x)) + 0.3 * tanh(a(x)) * (1 / (1 + e(zt, rh)^2)) * sin(ph(zt, rh, x))",
        vocabulary="stage1", modulation="0.3 * tanh(a(x)) * (1 / (1 + e(zt, rh)^2)) * sin(ph(zt, rh, x))",
        bound=1.3),
    TemplateSpec(
        name="s2_modulation_unit",
        label="Stage 2 (variant): 0.7 * tanh(f) + 0.3 * tanh(a) * envelope * sin(phase); |value| <= 1 by form",
        expressions=("f", "a", "e", "ph"),
        arguments={"f": ("zt", "rh", "x"), "a": ("x",), "e": ("zt", "rh"), "ph": ("zt", "rh", "x")},
        combine64="0.7 * tanh(f(zt, rh, x)) + 0.3 * tanh(a(x)) * (1 / (1 + e(zt, rh)^2)) * sin(ph(zt, rh, x))",
        vocabulary="stage1", modulation="0.3 * tanh(a(x)) * (1 / (1 + e(zt, rh)^2)) * sin(ph(zt, rh, x))",
        bound=1.0),
]}
# In the Stage-2 templates the sub-expressions are searched with the *Stage-1* vocabulary: the only trigonometric
# function is the template's own sin(ph), so it cannot be nested. The envelope 1 / (1 + e^2) lies in (0, 1] and
# |tanh(a)| < 1, so the modulation is bounded by 0.3 nat by construction.


# ----------------------------------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------------------------------
# Weighted L2, elementwise. Data layout: any row order; ``weights`` has the shape of ``y`` ((N,) or (N, n_out),
# one column per output). SymbolicRegression.jl evaluates  sum_i loss(pred_i, y_i, w_i) / sum_i w_i .
WL2_ELEMENTWISE = "sfm_wl2(prediction, target, weight) = weight * (prediction - target)^2"


def wl2_loss_numpy(prediction, y, weights) -> float:
    """Reference for ``WL2_ELEMENTWISE``: sum_i w_i (f_i - y_i)^2 / sum_i w_i  (one output)."""
    p, y, w = (np.asarray(a, dtype=np.float64) for a in (prediction, y, weights))
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def gauge_loss_julia(x_index: int = 3, name: str = "sfm_gauge_loss") -> str:
    """Julia source of the gauge-invariant loss for the residual target (iii): the weighted within-slice variance
    of the error,

        L(f) = sum_s sum_{i in s} w_i (e_i - ebar_s)^2 / sum_i w_i ,   e_i = f(X_i) - y_i ,
        ebar_s = sum_{i in s} w_i e_i / sum_{i in s} w_i .

    L(f + c(x)) = L(f) for every function c of the slice variable alone - the freedom the softmax ignores - and
    L is about 2 x KL per slice (second order in e), weighted by the slice masses.

    Data layout the function relies on:
      * rows of one momentum slice are contiguous;
      * a slice is a maximal run of rows with the *same value* of feature number ``x_index`` (1-based row of
        ``dataset.X``; 3 = ``x`` for the variables (zt, rh, x)). Two different slices must not share that value,
        or be adjacent with equal values;
      * ``weights`` of shape (N,), non-negative (missing weights count as 1);
      * the whole data set in every evaluation: run with ``batching=False`` (a random batch breaks the slices).

    The same source serves ``loss_function`` (first argument: the tree) and ``loss_function_expression`` (first
    argument: the expression object, needed for template expressions): ``eval_tree_array`` accepts both.
    The mean is removed in a first pass and the squares are summed in a second one, so a large additive gauge term
    in f does not cancel catastrophically in Float32.
    """
    xi = int(x_index)
    if xi < 1:
        raise ValueError("x_index is the 1-based feature row of the slice variable")
    return f"""function {name}(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    !flag && return L(Inf)
    X = dataset.X
    y = dataset.y
    w = dataset.weights
    n = dataset.n
    num = zero(L)
    den = zero(L)
    i = 1
    @inbounds while i <= n
        key = X[{xi}, i]
        j = i
        sw = zero(L)
        swe = zero(L)
        while j <= n && X[{xi}, j] == key
            wj = w === nothing ? one(L) : L(w[j])
            sw += wj
            swe += wj * L(prediction[j] - y[j])
            j += 1
        end
        if sw > zero(L)
            m = swe / sw
            for k in i:(j - 1)
                wk = w === nothing ? one(L) : L(w[k])
                d = L(prediction[k] - y[k]) - m
                num += wk * d * d
            end
            den += sw
        end
        i = j
    end
    return den > zero(L) ? num / den : L(Inf)
end"""


def gauge_loss_numpy(prediction, y, weights, slice_key) -> float:
    """Reference for ``gauge_loss_julia``. ``slice_key``: the slice variable per row (the ``x`` column); slices are
    the maximal runs of equal consecutive values, exactly as in the Julia code."""
    p, y, w, key = (np.asarray(a, dtype=np.float64) for a in (prediction, y, weights, slice_key))
    if not (p.shape == y.shape == w.shape == key.shape and p.ndim == 1):
        raise ValueError("prediction, y, weights and slice_key must be 1-D arrays of one length")
    if not np.all(np.isfinite(p)):
        return math.inf
    starts = np.concatenate([[0], np.flatnonzero(key[1:] != key[:-1]) + 1, [len(key)]])
    num = den = 0.0
    for a, b in zip(starts[:-1], starts[1:]):
        sw = float(np.sum(w[a:b]))
        if sw > 0:
            e = p[a:b] - y[a:b]
            m = float(np.sum(w[a:b] * e)) / sw
            num += float(np.sum(w[a:b] * (e - m) ** 2))
            den += sw
    return num / den if den > 0 else math.inf


# ----------------------------------------------------------------------------------------------------------------------
# What the runner passes to PySRRegressor
# ----------------------------------------------------------------------------------------------------------------------
def search_config(stage: str = "stage1", ndim: int = 3, loss: str = "gauge", template: Optional[str] = None,
                  precision: int = 64, x_index: int = 3) -> Dict[str, object]:
    """The math-critical ``PySRRegressor`` keyword arguments of one search (the runner adds iterations,
    populations, parallelism, output paths).

    stage      "stage1" | "stage1_raw" | "stage2"
    ndim       1 (trajectories: maxsize 18) or 2 / 3 (maxsize 30)
    loss       "gauge" (residual target; full objective, needs slice-contiguous rows and ``batching=False``) or
               "wl2" (weighted L2; trajectories and any gauge-fixed target)
    template   a key of ``TEMPLATES``: the result then also holds ``template`` = the keyword arguments of
               ``pysr.TemplateExpressionSpec`` (the runner builds the object: this module does not import pysr),
               the gauge loss is passed as ``loss_function_expression`` and no sympy mappings are needed
    precision  64 recommended: the audit and the translator work in float64, and ``print_precision`` must then be
               17 so that the constants in ``hall_of_fame.csv`` round-trip (PySR's default prints 5 digits)

    ``complexity_of_constants`` / ``complexity_of_variables`` are left at 1 (PLAN: a cubic has C = 13).
    """
    if stage not in VOCABULARIES:
        raise KeyError(f"unknown stage {stage!r}; available: {sorted(VOCABULARIES)}")
    if loss not in ("gauge", "wl2"):
        raise ValueError("loss must be 'gauge' or 'wl2'")
    v = VOCABULARIES[stage]
    cfg: Dict[str, object] = {
        "binary_operators": v.binary_operators(),
        "unary_operators": v.unary_operators(),
        "constraints": dict(v.constraints),
        "nested_constraints": {k: dict(d) for k, d in v.nested_constraints.items()},
        "complexity_of_operators": v.complexity_of_operators(),
        "complexity_of_constants": 1,
        "complexity_of_variables": 1,
        "maxsize": MAXSIZE_1D if int(ndim) == 1 else MAXSIZE_ND,
        "precision": int(precision),
        "print_precision": 17 if int(precision) == 64 else 9,
    }
    tmpl = None
    if template is not None:
        tmpl = TEMPLATES[template]
        if stage != tmpl.vocabulary:
            raise ValueError(f"template {template!r} is searched with vocabulary {tmpl.vocabulary!r}, not {stage!r}")
        cfg["template"] = tmpl.pysr_kwargs(precision)
    else:
        cfg["extra_sympy_mappings"] = v.extra_sympy_mappings()
    if loss == "wl2":
        cfg["elementwise_loss"] = WL2_ELEMENTWISE
    else:
        src = gauge_loss_julia(x_index)
        cfg["loss_function_expression" if tmpl is not None else "loss_function"] = src
        cfg["batching"] = False
    return cfg


# Confirmed against PySR 2.7.3 / SymbolicRegression.jl 2.7.2 / Julia 1.13.4 with tiny fits on 25 Nov 2026
# (src/sfm/sr/STATUS.md section 9; the evidence that can be kept is embedded in tests/test_sr_translate.py).
CONFIRMED_WITH_JULIA: Tuple[str, ...] = (
    "custom operator strings (ppow, softplus, l1p, sqa, linv) are accepted, in Float64 and in Float32 searches; "
    "their values agree with numpy_impl to 1e-12 on exported rows",
    "nested_constraints / constraints / complexity_of_operators keyed by custom operator *names* are accepted",
    "argument-size limits (constraints) are counted with the operator weights (check_constraints rejects "
    "exp(tanh(zt) + tanh(rh) + tanh(x)): weighted 11 > 9, unit count 8), as violates_constraints assumes",
    "constraints['ppow'] = (7, 1) admits a single variable as exponent (check_constraints accepts ppow(zt, rh)): "
    "members with a non-constant exponent have to be removed by operators.violations",
    "nesting: flag_illegal_nests agrees with violates_nesting on 14 hand-made trees; compute_complexity agrees with "
    "tree_complexity on all of them and on every hall-of-fame member seen",
    "elementwise weighted loss is normalised by sum(weights) (wl2_loss_numpy equals the CSV loss to 1e-15)",
    "the full-objective gauge loss (Dataset fields X, y, weights, n; eval_tree_array for a tree and, as "
    "loss_function_expression, for a template expression) equals gauge_loss_numpy on the exported rows to 1e-15",
    "batching=False: the reported loss is the loss on the full, slice-contiguous data set",
    "template combine strings: ValidVector arithmetic with numeric literals (0.3 *, 1 / (1 + e^2)), tanh and sin "
    "of a ValidVector; a Float32 search runs with the 0.3f0 form (TemplateSpec.combine(32))",
    "template complexity = sum of the sub-expressions' complexities (the combine function's own nodes are free)",
    "hall_of_fame.csv prints constants with print_precision significant digits (17: exact round trip)",
    "multi-output fit: weights of shape (N, n_out), guesses as list[list[str]] (a Horner cubic comes back with "
    "complexity 13), one hall_of_fame_output{i}.csv per output",
)
UNCONFIRMED: Tuple[str, ...] = ()        # nothing left of the first session's list
