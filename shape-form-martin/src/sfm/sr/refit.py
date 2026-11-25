"""sfm.sr.refit - likelihood refit of an audited equation and the controls C1 / C2 (PLAN.md section 7).

    model = CorrectedModel(base, expression_correction(expr))      # log P = log_softmax(log P_base + f(zt, rh, x))
    out = fit_candidate(model, train, val, config, label="...")    # sfm.train.fit + the audit of the refitted tree

WHERE THIS RUNS (PLAN section 13, A6 revised, point 7). Refits, controls and every registered score are made on the
H200. No likelihood computed on the laptop may select, rank or screen an equation. ``fit_candidate`` therefore refuses
to run unless ``SFM_SR_REFIT_ALLOWED=1`` is set in the environment (the H200 run-book sets it) or the call passes
``synthetic=True`` (tests on synthetic showers). This module does not import ``sfm.registry``; the registration hook
``register_refit`` is a documented stub.

The model
    f is the translated equation (``translate.TorchExpr``) with ALL free constants trainable, evaluated in float64 at
    the cell centres zt = ZC / 260 mm, rh = RC / 21.9 mm and x = (ln T - v0) / vs of each event - the same float
    operations as ``sfm.families.params.x_of_lnp``. No ``nan_to_num``, no clamp. A non-finite value of f raises
    ``FloatingPointError`` (strict mode, default): the audit has certified that this cannot happen on the audit box,
    so it would mean that the constants have left the audited region. The detector fold is the metric's own
    (``sfm.train.fit`` -> ``sfm.metric.event_nll``): the refit is a likelihood fit on training hits.
    k of the equation = ``n_correction()`` (free constants; structural template literals are not counted);
    ``n_params()`` adds the baseline's own count when it is refitted too (``freeze_base=False``, the default:
    PLAN "all constants trainable"). Both numbers are reported, never mixed.

After the fit (mandatory): the constants have moved, so ``fit_candidate`` audits ``model.current_expr()`` again with
``sfm.sr.audit.audit`` and returns the report; a refit whose audit fails must not be registered. A FINALIST needs
more than the screening rule: ``fit_candidate(..., finalist=True, tail_criterion=..., baseline_hash=...)`` runs
``audit.certify_finalist`` on the refitted constants (magnitude bound proven, domain, numerics, tail check).

Finalists (S4): the tail check uses the criterion record of the baseline of record, decided once from the baseline
alone (``tail_criterion_for_baseline`` -> ``checks.decide_tail_criterion``) and bound to its hash.
Clip log (S5): ``fit_candidate`` records the pre-clip gradient norm of every step and the gradient of every
constant (``ClipMonitor``), and flags a cusp-stalled run (definitions in ``CLIP_DEFINITIONS``).

Numerical hygiene (user's confirmation 1): top-level gauge terms are stripped before translation
(``expression_correction``); the correction, the sum with log P_base and the softmax are evaluated in float64 and a
float32 correction is refused; the audit bounds the raw |f| by 16 nat and the float64 rounding error by 1e-6 nat,
before and after the refit. ``sqa`` cannot give a NaN gradient at 0 (``operators.torch_impl``).

Controls (same schedule = the same ``FitConfig`` as the candidate, only ``name`` / ``out_dir`` differ)
    C1  ``control_c1(base)``: the baseline alone, all its constants trainable, continued with the candidate's
        schedule (what the refit alone buys; needed because the candidate's refit moves the baseline too).
    C2  ``control_c2(base, k)``: a smooth tensor polynomial with the same number of constants k,
            f = sum_j c_j u^a v^b t^c,  u, v in [-1, 1] the rescaled zt and sqrt(rh) over the data box, t = x,
        the first k monomials in graded lexicographic order, excluding the constant and the pure powers of x
        (the softmax cannot see them), zero start (the model starts exactly at the baseline). SR is credited with
        Delta(SR) - Delta(C2). REGISTERED FORM (user decision S1, PLAN 17.11): f = tanh(that polynomial), bounded by
        form, so that C2 and SR are compared inside the same bounded feasible set on the extended box - this is
        the default of ``control_c2`` / ``PolyControl`` and the only C2 accepted by ``fit_candidate(role=
        "control")``. The plain polynomial (``bounded=False``) is an optional diagnostic row
        (``role="diagnostic"``) with its audit verdict printed beside it (``c2_report``); it governs nothing.
    C3  (seed-null) is a search control: ``targets.build_residual_target(..., halves=...)``.

Trajectories: ``SubstitutedProvider(provider, {name: expr})`` replaces chosen raw parameters of a Track-A provider by
translated functions of x; k = the provider's count minus that of the replaced rows plus the equations' constants.
"""
from __future__ import annotations

import copy
import math
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import paths
from . import audit as A
from . import operators as O
from . import translate as T

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common as C  # noqa: E402

__all__ = ["CorrectedModel", "expression_correction", "PolyControl", "c2_monomials", "c2_tree", "control_c1",
           "control_c2", "audit_control", "audit_verdict", "c2_report", "C2_GOVERNING_LABEL", "C2_DIAGNOSTIC_LABEL",
           "ClipMonitor", "clip_statistics", "constants_inside_sqa", "CLIP_DEFINITIONS", "cusp_hover_statistics",
           "extended_ring_grid",
           "continued_log_density", "tail_criterion_for_baseline", "STAGE_BEFORE", "STAGE_TRANSLATION",
           "STAGE_AFTER",
           "SubstitutedProvider", "fit_candidate", "check_gradients", "register_refit", "refit_summary",
           "RefitNotAllowed", "ENV_ALLOW"]

ENV_ALLOW = "SFM_SR_REFIT_ALLOWED"
STAGE_BEFORE, STAGE_TRANSLATION, STAGE_AFTER = "before refit", "before translation", "after refit"


class RefitNotAllowed(RuntimeError):
    """Refits and scores of equations are made on the machine of record (PLAN section 13, A6 revised, point 7)."""


def _x_of_lnp(lnp: torch.Tensor, v0: float, vs: float, mass: float) -> torch.Tensor:
    p = torch.exp(lnp)
    return (torch.log(p * p / (torch.sqrt(p * p + mass * mass) + mass)) - v0) / vs


def expression_correction(expr, variable_names: Sequence[str] = O.VARIABLE_NAMES, dtype=torch.float64,
                          strip_gauge: bool = True, gauge_vars: Sequence[str] = ("x",)):
    """The translated equation as a correction module f(zt, rh, x); all free constants trainable.

    strip_gauge (default True): top-level additive terms that depend on ``gauge_vars`` alone (constants included)
    are removed first (``audit.drop_gauge_terms``). The softmax cannot see them, their constants have zero
    gradient, and a large one would cost float32 accuracy for nothing; they are neither trained nor counted in k.
    The removed terms are kept in ``module.dropped_gauge_terms`` (texts) and ``module.stripped_gauge_terms``
    (records with text, constants and the stage "before translation"; ``fit_candidate`` carries them into the
    post-refit audit record). An equation that is a pure gauge term is refused.
    Use ``strip_gauge=False`` for trajectories, where x is the argument and nothing is a gauge."""
    e = T.parse(expr) if isinstance(expr, str) else expr
    dropped: List[str] = []
    records: List[dict] = []
    k_written = T.n_constants(e)
    if strip_gauge:
        records = A.gauge_term_records(e, gauge_vars, STAGE_TRANSLATION)
        e2, dropped = A.drop_gauge_terms(e, gauge_vars)
        if e2 is None:
            raise ValueError("the equation depends on the gauge variables alone: it does not change the model "
                             f"(dropped: {dropped})")
        e = e2
    m = T.TorchExpr(e, variable_names, dtype=dtype, trainable=True)
    m.dropped_gauge_terms = list(dropped)
    m.stripped_gauge_terms = records                       # text, constants, stage: for the audit trail (S3)
    m.k_as_written = k_written
    return m


class CorrectedModel(nn.Module):
    """log P(lnp) = log_softmax(log P_base(lnp) + f(zt, rh, x)) on the grid of ``common.py``.

    base        a model lnp (B,) -> log P (B, N_Z, N_R)
    correction  a module f(zt, rh, x) broadcasting over (1, N_Z, N_R) and (B, 1, 1): ``expression_correction``
                or ``PolyControl``
    freeze_base False (default; PLAN "all constants trainable"): the baseline's constants are refitted together
                with the equation's, and control C1 (the baseline alone on the same schedule) measures what the
                refit alone buys. True: only the correction's constants are trained (C1 is then not needed)
    strict      True (default): a non-finite correction raises ``FloatingPointError``
    """

    def __init__(self, base: nn.Module, correction: nn.Module, freeze_base: bool = False, strict: bool = True,
                 v0: float = O.V0_PION, vs: float = O.VS_PION, mass: float = O.M_PION):
        super().__init__()
        self.base, self.correction = base, correction
        self.freeze_base, self.strict = bool(freeze_base), bool(strict)
        if self.freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
        self.v0, self.vs, self.mass = float(v0), float(vs), float(mass)
        self.register_buffer("zt", (C.ZZ.double() / O.ZETA_SCALE_MM)[None].clone())
        self.register_buffer("rh", (C.RR.double() / O.RHO_SCALE_MM)[None].clone())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()                               # a frozen baseline stays in eval mode
        return self

    def x(self, lnp: torch.Tensor) -> torch.Tensor:
        return _x_of_lnp(lnp, self.v0, self.vs, self.mass)

    def f(self, lnp: torch.Tensor) -> torch.Tensor:
        """The correction (B, N_Z, N_R) in float64."""
        x = self.x(lnp).double()[:, None, None]
        f = self.correction(self.zt, self.rh, x)
        if f.dtype != torch.float64:
            raise TypeError("the correction must be evaluated in float64: in float32 a gauge-like term of size G "
                            "costs G * 6e-8 nat of the (z', r) dependence (build it with dtype=torch.float64)")
        f = f.expand(len(lnp), C.N_Z, C.N_R) if f.dim() < 3 or f.shape[0] != len(lnp) else f
        if self.strict and not bool(torch.isfinite(f).all()):
            bad = int((~torch.isfinite(f)).sum())
            raise FloatingPointError(f"the correction is not finite in {bad} cells: the constants have left the "
                                     "audited region (no clamp is applied; audit the current equation)")
        return f

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        if self.freeze_base:
            with torch.no_grad():
                logp = self.base(lnp)
        else:
            logp = self.base(lnp)
        z = logp.double() + self.f(lnp)
        return F.log_softmax(z.reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R).to(logp.dtype)

    def n_correction(self) -> int:
        return sum(p.numel() for p in self.correction.parameters() if p.requires_grad)

    def n_params(self) -> int:
        """k: the trainable constants (the correction's; plus the baseline's if it is refitted)."""
        k = self.n_correction()
        if not self.freeze_base:
            k += int(self.base.n_params()) if hasattr(self.base, "n_params") else \
                sum(p.numel() for p in self.base.parameters())
        return k

    def current_expr(self):
        """The equation with its present constants (None for a polynomial control)."""
        return self.correction.current_expr() if hasattr(self.correction, "current_expr") else None


# ----------------------------------------------------------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------------------------------------------------------
def c2_monomials(k: int) -> List[Tuple[int, int, int]]:
    """The first k exponent triples (a, b, c) of u^a v^b t^c in graded lexicographic order with a + b >= 1:
    total degree first, then a, then b descending. The constant and the pure powers of t = x are excluded - the
    softmax cannot see them, so they would be constants that buy nothing."""
    out: List[Tuple[int, int, int]] = []
    deg = 1
    while len(out) < k:
        for a in range(deg, -1, -1):
            for b in range(deg - a, -1, -1):
                c = deg - a - b
                if a + b >= 1 and len(out) < k:
                    out.append((a, b, c))
        deg += 1
    return out


def c2_tree(k: int, bounded: bool = True, coefficients: Optional[Sequence[float]] = None) -> T.Expr:
    """The control polynomial as an expression tree in (zt, rh, x), so that it is evaluated, printed and AUDITED
    by the same code as an equation:

        p = sum_j c_j u^a_j v^b_j x^c_j,   u = 2 (zt - z_lo) / (z_hi - z_lo) - 1,   v = 2 sqa(rh) / sqrt(rh_hi) - 1

    (u, v in [-1, 1] on the data box z' in [-200, 1450] mm, r in [0, 700] mm; ``sqa(rh) = sqrt(rh)`` for rh >= 0).
    The scales are structural literals; the k coefficients are the free constants (zero unless given).
    ``bounded=True`` (default, the registered control: PLAN 17.11, S1) returns tanh(p): |f| <= 1 by form, like the
    Stage-1 template; ``bounded=False`` returns the plain polynomial (diagnostic only)."""
    box = O.data_box()
    z_lo, z_hi = box["zt"]
    fx = lambda v: T.const(v, fixed=True)                                  # noqa: E731
    u = T.op("-", T.op("*", T.op("-", T.var("zt"), fx(z_lo)), fx(2.0 / (z_hi - z_lo))), fx(1.0))
    v = T.op("-", T.op("*", T.op("sqa", T.var("rh")), fx(2.0 / float(np.sqrt(box["rh"][1])))), fx(1.0))
    coef = [0.0] * k if coefficients is None else [float(c) for c in coefficients]
    if len(coef) != k or k < 1:
        raise ValueError("k must be >= 1 and match the number of coefficients")

    def power(e: T.Expr, n: int) -> Optional[T.Expr]:
        if n == 0:
            return None
        return e if n == 1 else (T.op("square", e) if n == 2 else T.op("pow", e, fx(float(n))))

    total = None
    for cj, (a, b, c) in zip(coef, c2_monomials(k)):
        term = T.const(cj)
        for fac in (power(u, a), power(v, b), power(T.var("x"), c)):
            if fac is not None:
                term = T.op("*", term, fac)
        total = term if total is None else T.op("+", total, term)
    return T.op("tanh", total) if bounded else total


C2_GOVERNING_LABEL = "C2 tanh(polynomial), bounded by form - the registered control"
C2_DIAGNOSTIC_LABEL = "C2 plain polynomial, unbounded - diagnostic row, not used for Delta(SR) - Delta(C2)"


class PolyControl(nn.Module):
    """Control C2 as a correction module: the polynomial of ``c2_tree`` with k trainable coefficients (zero start:
    the model starts at the baseline).

    ``bounded=True`` (default): f = tanh(polynomial). This is THE registered C2 (user decision S1, PLAN 17.11):
    it is used for Delta(SR) - Delta(C2), so that C2 and SR are compared inside the same bounded feasible set on the
    extended box; it passes the magnitude audit by construction.
    ``bounded=False``: the plain polynomial, which is not bounded outside the data. It may be recorded as a
    non-governing DIAGNOSTIC row with its audit verdict beside it (``fit_candidate(..., role="diagnostic")``); it
    never enters Delta(SR) - Delta(C2)."""

    def __init__(self, k: int, dtype=torch.float64, bounded: bool = True):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k, self.bounded = int(k), bool(bounded)
        self.governing = self.bounded
        self.exponents = c2_monomials(self.k)
        self.poly = T.TorchExpr(c2_tree(self.k, self.bounded), O.VARIABLE_NAMES, dtype=dtype)
        self.label = C2_GOVERNING_LABEL if self.bounded else C2_DIAGNOSTIC_LABEL

    @property
    def c(self) -> torch.Tensor:
        return self.poly.c

    def forward(self, zt, rh, x):
        return self.poly(zt, rh, x)

    def n_params(self) -> int:
        return self.k

    def current_expr(self) -> T.Expr:
        return self.poly.current_expr()


def control_c1(base: nn.Module) -> nn.Module:
    """C1: a deep copy of the baseline with every parameter trainable - "refit only, same schedule"."""
    m = copy.deepcopy(base)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def control_c2(base: nn.Module, k: int, freeze_base: bool = False, bounded: bool = True) -> CorrectedModel:
    """C2: the baseline plus tanh(tensor polynomial) with k constants (k = the candidate equation's free
    constants, ``CorrectedModel.n_correction()``), zero start; ``freeze_base`` as in the candidate's refit. This is
    the registered, governing control (S1): fit it with ``fit_candidate(..., role="control")``.
    ``bounded=False`` gives the plain polynomial for the optional diagnostic row: fit it with
    ``fit_candidate(..., role="diagnostic")`` (``role="control"`` refuses it). Either way the same extended-box
    audit as for an equation is run on the fitted coefficients and returned."""
    return CorrectedModel(copy.deepcopy(base), PolyControl(k, bounded=bounded), freeze_base=freeze_base)


def c2_report(governing: Mapping, diagnostic: Optional[Mapping] = None) -> dict:
    """The C2 rows of a results table from ``fit_candidate`` records: the governing control (tanh(polynomial),
    used for Delta(SR) - Delta(C2)) and, optionally, the plain polynomial as a labelled diagnostic row, each with
    its audit verdict. Raises if the roles are mixed up: a record with ``role="control"`` must be the bounded
    control, a diagnostic record must not be passed as governing. No likelihood is computed here."""
    if governing.get("role") != "control" or not governing.get("governing"):
        raise ValueError("the governing C2 row must come from fit_candidate(control_c2(base, k), role='control')")
    if diagnostic is not None and (diagnostic.get("role") != "diagnostic" or diagnostic.get("governing")):
        raise ValueError("the diagnostic row must come from fit_candidate(..., role='diagnostic')")

    def row(rec, used_for):
        v = rec.get("audit_verdict") or {}
        return {"label": rec.get("control_label") or rec["label"], "role": rec["role"], "k": rec.get("k_equation"),
                "used_for": used_for, "audit_passed": rec.get("audit_passed"), "audit_verdict": v,
                "violates_extended_box_bound": v.get("violates_extended_box_bound"), "equation": rec.get("expr_refit")}
    out = {"governing": row(governing, "Delta(SR) - Delta(C2)"),
           "diagnostic": None if diagnostic is None else row(diagnostic, "nothing: diagnostic row only"),
           "warnings": []}
    if not out["governing"]["audit_passed"]:
        out["warnings"].append("the governing control did not pass its audit although it is bounded by form: "
                               "check the audit record (overflow / numerics) before using the difference")
    lines = []
    for key in ("governing", "diagnostic"):
        r = out[key]
        if r is None:
            continue
        v = r["audit_verdict"]
        hu = v.get("H_upper")
        lines.append(f"{key:10s} | {r['label']} | k = {r['k']} | audit: {'pass' if r['audit_passed'] else 'FAIL'} "
                     f"(domain {v.get('domain')}, magnitude {v.get('magnitude')}, H <= "
                     f"{'n/a' if hu is None else format(hu, '.4g')}) | used for: {r['used_for']}")
    out["text"] = "\n".join(lines)
    return out


def audit_verdict(report_json: Mapping) -> dict:
    """The few numbers of an audit that are printed beside a likelihood difference: verdicts and the bound."""
    m, nm = report_json.get("magnitude", {}), report_json.get("numerics", {})

    def num(v):
        return float(v) if isinstance(v, (int, float)) else (float("inf") if v == "inf" else None)
    return {"passed": bool(report_json.get("passed")), "certified": bool(report_json.get("certified")),
            "domain": report_json.get("domain", {}).get("status"), "magnitude": m.get("status"),
            "H_lower": num(m.get("sup_lower")), "H_upper": num(m.get("sup_upper")),
            "limit": report_json.get("limit"), "gauge": report_json.get("gauge"),
            "raw": (nm.get("raw") or {}).get("status"), "violates_extended_box_bound": m.get("status") == "failed"}


def audit_control(model_or_correction, box: Optional[Mapping] = None, **audit_kw) -> dict:
    """The extended-box audit of a control's correction with its PRESENT (fitted) coefficients - the same
    ``audit.audit`` (domain, magnitude with the same gauge and limit, numerics) as for an equation. Returns
    {"label", "equation", "verdict": audit_verdict(...), "audit": full report}. For C2 this answers "does the
    control violate the bound that every SR candidate has to respect?"; print ``verdict`` beside
    Delta(SR) - Delta(C2)."""
    corr = getattr(model_or_correction, "correction", model_or_correction)
    if not hasattr(corr, "current_expr"):
        raise TypeError("the control has no expression to audit")
    e = corr.current_expr()
    rep = A.audit(e, box, **audit_kw).to_json()
    return {"label": getattr(corr, "label", type(corr).__name__), "equation": T.to_string(e), "k": T.n_constants(e),
            "verdict": audit_verdict(rep), "audit": rep}


# ----------------------------------------------------------------------------------------------------------------------
# Trajectories
# ----------------------------------------------------------------------------------------------------------------------
class SubstitutedProvider(nn.Module):
    """A parameter provider whose columns ``names`` are replaced by translated functions of x.

    provider       ``names`` and ``forward(lnp) -> (B, n)`` (``sfm.families.params``)
    replacements   {raw parameter name: equation string or tree in the single variable "x"}
    replaced_k     the provider's constants in the replaced rows; default: the row sums of ``provider.mask`` for a
                   ``PolyProvider``, otherwise it must be given
    """

    def __init__(self, provider: nn.Module, replacements: Mapping[str, object],
                 replaced_k: Optional[int] = None):
        super().__init__()
        self.provider = provider
        self.names = list(provider.names)
        unknown = [n for n in replacements if n not in self.names]
        if unknown:
            raise KeyError(f"provider has no parameter(s) {unknown}")
        self.columns = [self.names.index(n) for n in replacements]
        self.exprs = nn.ModuleList([expression_correction(e, ("x",), strip_gauge=False)
                                    for e in replacements.values()])
        if replaced_k is None:
            mask = getattr(provider, "mask", None)
            if mask is None:
                raise ValueError("replaced_k is needed for a provider without a degree mask")
            replaced_k = int(sum(float(mask[i].sum()) for i in self.columns))
        self.replaced_k = int(replaced_k)
        self.v0 = float(getattr(provider, "v0", O.V0_PION))
        self.vs = float(getattr(provider, "vs", O.VS_PION))
        self.mass = float(getattr(provider, "mass", O.M_PION))

    def forward(self, lnp: torch.Tensor) -> torch.Tensor:
        raw = self.provider(lnp)
        x = _x_of_lnp(lnp, self.v0, self.vs, self.mass).double()
        cols = list(raw.unbind(1))
        for i, m in zip(self.columns, self.exprs):
            v = m(x)
            if not bool(torch.isfinite(v).all()):
                raise FloatingPointError(f"trajectory of {self.names[i]!r} is not finite")
            cols[i] = v.expand(len(lnp)).to(raw.dtype)
        return torch.stack(cols, 1)

    def n_params(self) -> int:
        base_k = int(self.provider.n_params()) if hasattr(self.provider, "n_params") else \
            sum(p.numel() for p in self.provider.parameters())
        return base_k - self.replaced_k + sum(int(m.n_constants()) for m in self.exprs)

    def current_exprs(self) -> Dict[str, object]:
        return {self.names[i]: m.current_expr() for i, m in zip(self.columns, self.exprs)}


# ----------------------------------------------------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------------------------------------------------
def check_gradients(model: nn.Module, data, n_events: int = 64, K: int = 16, seed: int = 0) -> dict:
    """One backward pass on a few events: are the loss and every gradient finite? (``sqa`` at exactly 0 has an
    infinite slope and gives a NaN gradient; the cell centres avoid r = 0, this check confirms it for the model.)"""
    from .. import metric

    was = model.training
    model.train()
    try:
        b = data.batch(torch.arange(min(int(n_events), len(data)), device=data.off.device))
        gen = torch.Generator(device=data.xyz.device).manual_seed(int(seed))
        loss = metric.event_nll(model, b, data.box, int(K), gen)[0].mean()
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        finite = all(g is None or bool(torch.isfinite(g).all()) for g in grads)
        return {"loss_finite": bool(torch.isfinite(loss)), "gradients_finite": bool(finite),
                "unused_parameters": int(sum(g is None for g in grads)), "n_parameters": len(params)}
    finally:
        model.train(was)


# ----------------------------------------------------------------------------------------------------------------------
# Gradient-clip log (PLAN 17.11, S5)
# ----------------------------------------------------------------------------------------------------------------------
PERSIST_RUN = 20             # consecutive steps
PERSIST_TAIL_FRACTION = 0.25  # the final quarter of the run
PERSIST_TAIL_SHARE = 0.5
DRIVE_SHARE = 0.5
STALL_RATIO = 0.25
CLIP_DEFINITIONS = {
    "clipped step": "the gradient norm over ALL trainable parameters, before clipping, exceeds FitConfig.clip",
    "driven by constant i": "a clipped step in which g_i^2 >= 0.5 * (squared gradient norm): the constant alone "
                            "carries at least half of the squared norm",
    "persistent (constant i)": "the constant drives at least 20 consecutive steps, or more than half of the steps "
                               "of the final quarter of the run (if that quarter has at least 8 steps)",
    "stalled (constant i)": "persistent, and over that window |net change of the constant| <= 0.25 * (sum of its "
                            "absolute step-to-step changes): it moves back and forth, or not at all",
    "cusp-stalled (run)": "some constant that lies inside the argument of an sqa node is stalled",
}


def clip_statistics(total_norm, const_grads, const_values, clip: float, *, in_cusp=None, start_values=None) -> dict:
    """Clip counts per constant and per step, from what ``ClipMonitor`` recorded (pure NumPy; definitions in
    ``CLIP_DEFINITIONS``).

    total_norm    (n,) pre-clip gradient norm over all trainable parameters per step (NaN: no backward ran -
                  the loss was not finite; Inf / NaN: a non-finite gradient - the harness skips such batches)
    const_grads   (n, k) raw gradient of each constant of the correction per step
    const_values  (n, k) value of each constant after each step;  start_values (k,) before the first step
    in_cusp       (k,) bool: the constant lies inside the argument of an ``sqa`` node
    """
    tn = np.asarray(total_norm, dtype=np.float64).reshape(-1)
    n = len(tn)
    cg = np.asarray(const_grads, dtype=np.float64).reshape(n, -1) if n else np.zeros((0, 0))
    cv = np.asarray(const_values, dtype=np.float64).reshape(n, -1) if n else np.zeros((0, 0))
    k = cg.shape[1] if n else 0
    cusp = np.zeros(k, bool) if in_cusp is None else np.asarray(in_cusp, dtype=bool)
    valid = np.isfinite(tn)
    clipped = valid & (tn > float(clip))
    with np.errstate(all="ignore"):
        share = np.where(clipped[:, None], cg * cg / (tn * tn)[:, None], 0.0) if k else np.zeros((n, 0))
    driven = clipped[:, None] & (share >= DRIVE_SHARE) if k else np.zeros((n, 0), bool)
    n_tail = int(math.ceil(PERSIST_TAIL_FRACTION * n))
    tail = slice(n - n_tail, n)
    prev = np.vstack([np.asarray(start_values, dtype=np.float64).reshape(1, -1), cv[:-1]]) if (n and start_values
                                                                                               is not None) \
        else (np.vstack([cv[:1], cv[:-1]]) if n else cv)
    steps = np.abs(cv - prev) if n else cv

    def window_ratio(i, lo, hi):                            # steps lo .. hi - 1
        path = float(steps[lo:hi, i].sum())
        net = float(abs(cv[hi - 1, i] - prev[lo, i]))
        return (net / path) if path > 0 else 0.0

    per = []
    for i in range(k):
        d = driven[:, i]
        best, best_start, run = 0, 0, 0
        for t in range(n):
            run = run + 1 if d[t] else 0
            if run > best:
                best, best_start = run, t - run + 1
        tail_frac = float(d[tail].mean()) if n_tail else 0.0
        by_run = best >= PERSIST_RUN
        by_tail = n_tail >= 8 and tail_frac > PERSIST_TAIL_SHARE
        ratios = []
        if by_run:
            ratios.append(window_ratio(i, best_start, best_start + best))
        if by_tail:
            ratios.append(window_ratio(i, n - n_tail, n))
        stalled = bool((by_run or by_tail) and min(ratios) <= STALL_RATIO)
        g = cg[d, i]
        flips = float(np.mean(np.sign(g[1:]) != np.sign(g[:-1]))) if len(g) > 1 else 0.0
        per.append({"index": i, "value_end": float(cv[-1, i]) if n else None, "inside_sqa_argument": bool(cusp[i]),
                    "n_driven": int(d.sum()), "longest_driven_run": int(best), "driven_fraction_final_quarter": tail_frac,
                    "persistent": bool(by_run or by_tail), "net_over_path": (min(ratios) if ratios else None),
                    "gradient_sign_flip_fraction_when_driven": flips, "stalled": stalled,
                    "max_abs_gradient": float(np.nanmax(np.abs(cg[:, i]))) if n and np.isfinite(cg[:, i]).any() else None})
    flags = {"persistent_clipping": any(c["persistent"] for c in per),
             "stalled": any(c["stalled"] for c in per),
             "cusp_stalled": any(c["stalled"] and c["inside_sqa_argument"] for c in per)}
    out = {"definitions": dict(CLIP_DEFINITIONS), "clip_norm": float(clip), "n_steps": int(n),
           "n_clipped": int(clipped.sum()), "clipped_fraction": float(clipped.mean()) if n else 0.0,
           "n_no_backward": int(np.isnan(tn).sum()), "n_nonfinite_gradient": int((~valid & ~np.isnan(tn)).sum()),
           "max_norm": float(tn[valid].max()) if valid.any() else None, "per_constant": per, "flags": flags}
    if n <= 20000:
        out["per_step"] = {"clipped": clipped.astype(int).tolist(),
                           "driven_by": [np.flatnonzero(driven[t]).tolist() for t in range(n)] if k else []}
    else:
        out["per_step"] = {"note": f"{n} steps: per-step lists omitted, counts above are complete"}
    return out


def constants_inside_sqa(e) -> List[bool]:
    """For each free constant of the tree (order of ``translate.constants``): does it lie inside the argument of an
    ``sqa`` node? Those are the constants that can sit on a cusp."""
    out: List[bool] = []

    def rec(n, inside):
        if n.kind == "const":
            if not n.fixed:
                out.append(inside)
            return
        for a_ in n.args:
            rec(a_, inside or (n.kind == "op" and n.name == "sqa"))
    rec(e, False)
    return out


def cusp_hover_statistics(expr, const_history, zt, rh, x_values=(-1.0, -0.5, 0.0, 0.5, 1.0),
                          tail_fraction: float = PERSIST_TAIL_FRACTION, near: float = 3.0) -> dict:
    """SUPPLEMENTARY to the clip log, and independent of the clip norm: does a constant hover at a cusp of ``sqa``?

    On a grid the likelihood has a kink wherever the argument of an ``sqa`` node vanishes in a cell. For every
    ``sqa`` node the argument is evaluated in all grid cells (``zt``, ``rh`` broadcastable arrays; ``x_values``)
    with the constants of each step of the final quarter of the run (``const_history`` (n, k), order of
    ``translate.constants``). Reported per node: the number of cells in which the argument changed sign during
    that window (the cusp was crossed), the smallest |argument| reached, and that distance in units of the
    argument's own excursion in the same cell. ``suspect`` = a crossing, or a distance of at most ``near`` (3)
    excursions: the constant's jitter is comparable to its distance from a cusp.
    This is NOT the user's definition of a cusp-stalled run (persistent clipping, ``clip_statistics``): at the
    harness's default clip norm a constant can hover at a cusp without a single clipped step, because the
    gradient at a V-shaped kink is bounded. For a constant unrelated to any cusp the chance of ``suspect`` is
    roughly (6 excursions) / (grid spacing of the argument)."""
    hist = np.asarray(const_history, dtype=np.float64)
    n = len(hist)
    nodes = [nd for nd in T.walk(expr) if nd.kind == "op" and nd.name == "sqa"]
    out = {"n_steps_window": 0, "nodes": [], "suspect": False,
           "definition": "supplementary, clip-independent: a sign change of an sqa argument in a grid cell during "
                         "the final quarter, or its smallest |value| within 3 of its own excursions there"}
    if not nodes or n < 8:
        return out
    n_tail = max(int(math.ceil(tail_fraction * n)), 2)
    zt, rh = np.asarray(zt, dtype=np.float64), np.asarray(rh, dtype=np.float64)
    xv = np.asarray(x_values, dtype=np.float64)
    env = {"zt": zt[..., None], "rh": rh[..., None], "x": xv.reshape((1,) * zt.ndim + (-1,))}
    out["n_steps_window"] = n_tail
    for which in range(len(nodes)):
        vals = []
        for cvec in hist[n - n_tail:]:
            tree = T.with_constants(expr, cvec)
            arg = [nd for nd in T.walk(tree) if nd.kind == "op" and nd.name == "sqa"][which].args[0]
            used = {v: env[v] for v in T.variables(arg)}
            a = np.asarray(T.evaluate(arg, used), dtype=np.float64)
            vals.append(np.broadcast_to(a, np.broadcast_shapes(a.shape, *[u.shape for u in used.values()]) if used
                                        else a.shape).reshape(-1))
        v = np.stack(vals)
        lo, hi = v.min(0), v.max(0)
        crossed = int(np.sum((lo < 0) & (hi > 0)))
        dist = np.abs(v).min(0)
        exc = hi - lo
        with np.errstate(all="ignore"):
            rel = np.where(exc > 0, dist / exc, np.inf)
        rec = {"node": T.to_string(nodes[which])[:120], "cells_crossed": crossed,
               "min_abs_argument": float(dist.min()),
               "min_distance_in_excursions": float(rel.min()) if np.isfinite(rel).any() else None,
               "argument_depends_on_constants": bool(np.any(exc > 0))}
        rec["suspect"] = bool(crossed > 0 or (rec["min_distance_in_excursions"] is not None
                                              and rec["min_distance_in_excursions"] <= near))
        out["nodes"].append(rec)
    out["suspect"] = any(r["suspect"] for r in out["nodes"])
    return out


class ClipMonitor:
    """Records, for every step of ``sfm.train.fit``, the gradient norm BEFORE clipping (tensor hooks on all trainable
    parameters) and the raw gradient and value of every constant of the correction; ``summary()`` =
    ``clip_statistics``. Use: ``mon.attach(); fit(..., callback=mon.after_step); mon.detach()``.
    The state is not part of the harness's checkpoint: after a resume the log covers the steps of this process."""

    def __init__(self, model: nn.Module, clip_norm: float):
        self.clip = float(clip_norm)
        self.params = [p for p in model.parameters() if p.requires_grad]
        corr = getattr(model, "correction", None)
        self.cparams = [p for p in corr.parameters() if p.requires_grad] if corr is not None else []
        e = model.current_expr() if hasattr(model, "current_expr") else None
        k = sum(p.numel() for p in self.cparams)
        cusp = constants_inside_sqa(e) if e is not None else []
        self.in_cusp = cusp if len(cusp) == k else [False] * k
        self.start = self._values()
        self._buf: Dict[int, torch.Tensor] = {}
        self._handles: list = []
        self.total_norm: List[float] = []
        self.c_grad: List[np.ndarray] = []
        self.c_val: List[np.ndarray] = []

    def _values(self) -> np.ndarray:
        if not self.cparams:
            return np.zeros(0)
        return torch.cat([p.detach().reshape(-1).double().cpu() for p in self.cparams]).numpy().copy()

    def attach(self) -> "ClipMonitor":
        for p in self.params:
            self._handles.append(p.register_hook(lambda g, key=id(p): self._buf.__setitem__(key, g.detach().clone())))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def after_step(self, step: int, model: nn.Module) -> None:
        k = sum(p.numel() for p in self.cparams)
        if self._buf:
            sq = sum(float((g.double() ** 2).sum()) for g in self._buf.values())
            norm = math.sqrt(sq) if math.isfinite(sq) else math.inf
            cg = np.concatenate([self._buf[id(p)].reshape(-1).double().cpu().numpy() if id(p) in self._buf
                                 else np.zeros(p.numel()) for p in self.cparams]) if self.cparams else np.zeros(0)
        else:                                               # no backward in this step: the loss was not finite
            norm, cg = math.nan, np.full(k, np.nan)
        self.total_norm.append(norm)
        self.c_grad.append(cg)
        self.c_val.append(self._values())
        self._buf = {}

    def summary(self) -> dict:
        n = len(self.total_norm)
        k = len(self.in_cusp)
        out = clip_statistics(self.total_norm, np.array(self.c_grad).reshape(n, k), np.array(self.c_val).reshape(n, k),
                              self.clip, in_cusp=self.in_cusp, start_values=self.start)
        if n == 0:
            out["note"] = "no training step ran in this call (finished run loaded): the clip log is empty"
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Tail criterion of a closed-form baseline (PLAN 17.11, S4)
# ----------------------------------------------------------------------------------------------------------------------
def extended_ring_grid(grid, r_max_mm: float = 1500.0, n_extra: int = 24):
    """A ``sfm.families.Grid`` with the same depth cells and the same rings as ``grid`` plus ``n_extra``
    logarithmically spaced rings from the grid's edge (700 mm) to ``r_max_mm``, in float64. A model built on it
    with the same parameters is the baseline's own density functional beyond the grid."""
    from ..families import Grid

    re_std, ze, zc = grid.r_edges.double().cpu(), grid.z_edges.double().cpu(), grid.zc.double().cpu()
    if float(re_std[-1]) >= r_max_mm:
        raise ValueError("the grid already reaches r_max_mm")
    extra = torch.exp(torch.linspace(math.log(float(re_std[-1])), math.log(float(r_max_mm)), n_extra + 1,
                                     dtype=torch.float64))[1:]
    re = torch.cat([re_std, extra])
    rc = torch.sqrt(0.5 * (re[1:] ** 2 + re[:-1] ** 2))
    dz = (grid.z_max - grid.z_min) / grid.N_Z
    log_vol = (math.log(dz) + torch.log(math.pi * (re[1:] ** 2 - re[:-1] ** 2)))[None, :].expand(grid.N_Z, len(rc)).clone()
    rc[:grid.N_R] = grid.rc.double().cpu()                  # the standard rings exactly as in the standard grid
    log_vol[:, :grid.N_R] = grid.log_vol.double().cpu()
    return Grid(grid.n_z, grid.n_r, grid.r0, grid.z_min, grid.z_max, grid.N_Z, float(r_max_mm), len(rc),
                dtype=torch.float64, n_zf=grid.n_zf, _edges=(ze, re, zc, rc, log_vol))


def continued_log_density(model_std: nn.Module, model_ext: nn.Module, lnp: torch.Tensor) -> dict:
    """ln rho(z', r) of a component model beyond its grid, as the continuation of the model's own density.

    ``model_std`` is the baseline on its grid, ``model_ext`` the same model (same parameters) built on
    ``extended_ring_grid``. Each component is normalised on the grid it lives on, so the two differ, component
    by component, by a factor that does not depend on r (a constant, or a function of depth where the lateral
    profile is renormalised depth by depth). That factor is measured on the rings both grids share and applied
    to the extended component; the mixture is then rebuilt with the model's own weights. ``match_error`` = the
    largest |ln P| difference between the rebuilt map and ``model_std.core_map`` on the shared rings (cells with
    ln P > -40): it must be at rounding level, otherwise the model is not of this kind.
    Density BEFORE the acceptance template (a detector effect defined on the grid only), 3-D density
    rho = P / cell volume. Returns {"log_rho" (B, NZ, NR_ext), "z_mm", "r_mm", "z_edges_mm", "log_p_std",
    "match_error"} as NumPy arrays."""
    with torch.no_grad():
        lw, cs = model_std.components(lnp)
        _, ce = model_ext.components(lnp)
        core = model_std.core_map(lnp)
        nr = model_std.grid.N_R
        cont = []
        for name in cs:
            d = cs[name] - ce[name][:, :, :nr]
            ok = torch.isfinite(d) & (cs[name] > -600.0)
            cnt = ok.sum(2, keepdim=True)
            tot = torch.where(ok, d, torch.zeros_like(d)).sum(2, keepdim=True)
            off = torch.where(cnt > 0, tot / torch.where(cnt > 0, cnt, torch.ones_like(cnt)), torch.zeros_like(tot))
            cont.append(lw[name][:, None, None] + ce[name] + off)
        rec = torch.logsumexp(torch.stack(cont, 0), 0)
        sel = core > -40.0
        err = float((rec[:, :, :nr] - core)[sel].abs().max()) if bool(sel.any()) else 0.0
        log_rho = rec - model_ext.grid.log_vol[None]
    return {"log_rho": log_rho.double().cpu().numpy(), "z_mm": model_ext.grid.zc.double().cpu().numpy(),
            "r_mm": model_ext.grid.rc.double().cpu().numpy(),
            "z_edges_mm": model_std.grid.z_edges.double().cpu().numpy(),
            "log_p_std": core.double().cpu().numpy(), "match_error": err}


def tail_criterion_for_baseline(model_std: nn.Module, model_ext: nn.Module, *, baseline_hash: str, label: str,
                                momenta_gev: Sequence[float] = (0.05, 0.1, 1.0, 5.0, 20.0, 30.0, 50.0),
                                max_match_error: float = 1e-6, **kw) -> dict:
    """The tail-criterion record (``checks.decide_tail_criterion``) of a closed-form baseline, decided from the
    baseline alone: its density is continued to r = 1500 mm (``continued_log_density``) and evaluated at the
    pre-registered momenta and at every depth row of the grid; z_90 comes from the baseline's own map.
    On the H200 this is run ONCE for the refitted Track A finalist, with the SHA-256 of its checkpoint as
    ``baseline_hash``; every finalist is then certified against that record."""
    from . import checks as K

    mom = torch.tensor(list(momenta_gev), dtype=model_ext.grid.zc.dtype)
    cd = continued_log_density(model_std, model_ext, torch.log(mom).to(model_ext.grid.zc.device))
    if cd["match_error"] > max_match_error:
        raise ValueError(f"the extended model does not continue the baseline (mismatch {cd['match_error']:.3g} on "
                         "the shared rings): build it with the same parameters on extended_ring_grid(model.grid)")
    P = np.exp(cd["log_p_std"])
    z90 = [float(np.interp(0.9, np.concatenate([[0.0], np.cumsum(P[j].sum(1)) / P[j].sum()]), cd["z_edges_mm"]))
           for j in range(len(mom))]
    note = kw.pop("note", "")
    return K.decide_tail_criterion(cd["log_rho"], cd["z_mm"], cd["r_mm"], list(momenta_gev), z90,
                                   baseline_hash=baseline_hash, label=label,
                                   note=(note + " | " if note else "") + "density before the acceptance template; "
                                   f"continuation mismatch on the shared rings {cd['match_error']:.2e}", **kw)


# ----------------------------------------------------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------------------------------------------------
def fit_candidate(model: nn.Module, train, val, config, *, label: str, synthetic: bool = False,
                  audit_box: Optional[Mapping] = None, audit_kw: Optional[dict] = None, log=print,
                  role: str = "candidate", finalist: bool = False, tail_criterion: Optional[Mapping] = None,
                  baseline_hash: Optional[str] = None, measured_tail: Optional[Mapping] = None,
                  pre_refit_audit: Optional[Mapping] = None, baseline_tail: Optional[Mapping] = None) -> dict:
    """Likelihood refit through the existing harness ``sfm.train.fit`` (all trainable constants, training hits, the
    metric's fold, no clamps), followed by the mandatory audit of the REFITTED equation.

    Allowed only on the machine of record (environment ``SFM_SR_REFIT_ALLOWED=1``) or with ``synthetic=True`` for
    synthetic showers; otherwise ``RefitNotAllowed``.

    role       "candidate": an equation; ``audit_passed`` is a gate for registration.
               "control": the governing control. For C2 that is tanh(polynomial) (``control_c2(base, k)``; user
               decision S1) - the plain polynomial is refused here. Its audit is run and returned.
               "diagnostic": a non-governing row, e.g. the plain polynomial ``control_c2(base, k, bounded=False)``;
               its audit verdict is returned to be printed beside it; it never enters Delta(SR) - Delta(C2).
    finalist   True: after the refit ``audit.certify_finalist`` is run on the refitted constants - domain safe,
               magnitude bound PROVEN, numerics proven, and the tail check under ``tail_criterion`` (the record of
               ``checks.decide_tail_criterion`` for the baseline of record, bound to ``baseline_hash``;
               ``measured_tail``: the slopes of the jointly refitted baseline in the same format, optional).
               The result is ``finalist_certified``; a finalist without it must not enter LOCK.json.
    pre_refit_audit   the JSON of the audit made before the refit (laptop): its stripped gauge terms are carried
               into the record, and compared with what was stripped before translation.

    Gauge terms (S3): ``stripped_gauge_terms`` lists every stripped term with its text, constants and stage
    ("before refit" from ``pre_refit_audit``, "before translation" from ``expression_correction``, "after refit"
    from the post-refit audit); none of their constants is trained or counted (``k_equation``); ``k_as_written``
    is the count including them.
    Clip log (S5): ``clip_log`` = ``clip_statistics`` of the run (per constant and per step; definitions inside),
    ``cusp_stalled`` = its flag (persistent clipping driven by a constant inside an sqa argument, without net
    movement). ``cusp_hover`` / ``cusp_hover_suspect``: the supplementary, clip-independent indicator of
    ``cusp_hover_statistics`` - reported, not part of the flag.

    Returns {"label", "role", "governing", "control_label", "fit", "k", "k_equation", "k_as_written", "expr_start",
             "expr_refit", "expr_refit_dict", "dropped_gauge_terms", "stripped_gauge_terms", "audit_refit",
             "audit_passed", "audit_certified", "audit_verdict", "finalist", "finalist_certified", "clip_log",
             "cusp_stalled", "warnings", "registered": False}.
    The validation numbers inside ``fit`` are the harness's monitoring values; the score of record is produced by
    the registry on the H200 (``register_refit``).
    """
    if role not in ("candidate", "control", "diagnostic"):
        raise ValueError("role must be 'candidate', 'control' or 'diagnostic'")
    if baseline_tail is not None:
        raise ValueError("baseline_tail is superseded (PLAN 17.11, S4): pass tail_criterion and baseline_hash")
    corr = getattr(model, "correction", None)
    if isinstance(corr, PolyControl):
        if role == "candidate":
            raise ValueError("a PolyControl is a control, not a candidate equation")
        if role == "control" and not corr.bounded:
            raise ValueError("the governing C2 is tanh(polynomial) (PLAN 17.11, S1): fit the plain polynomial with "
                             "role='diagnostic'")
    if not synthetic and os.environ.get(ENV_ALLOW) != "1":
        raise RefitNotAllowed(
            "refits and scores of PySR equations are produced on the H200 (PLAN section 13, A6 revised, point 7): "
            f"set {ENV_ALLOW}=1 there, or pass synthetic=True for synthetic showers")
    from .. import train as sfm_train

    cfg = config if isinstance(config, sfm_train.FitConfig) else sfm_train.FitConfig.from_dict(dict(config))
    start = model.current_expr() if hasattr(model, "current_expr") else None
    warn: List[str] = []
    carried = [dict(r) for r in (pre_refit_audit or {}).get("stripped_gauge_terms", [])]
    translated = [dict(r) for r in getattr(corr, "stripped_gauge_terms", [])]
    if pre_refit_audit is not None:
        if sorted(r["term"] for r in carried) != sorted(r["term"] for r in translated):
            warn.append("the gauge terms stripped before translation differ from those of the pre-refit audit: "
                        "the audited equation and the refitted one are not the same")
    carried += translated
    mon = ClipMonitor(model, cfg.clip).attach()
    try:
        res = sfm_train.fit(model, train, val, cfg, meta={"sr_label": label, "synthetic": bool(synthetic),
                                                          "sr_role": role}, log=log, callback=mon.after_step)
    finally:
        mon.detach()
    clip_log = mon.summary()
    hover = None
    if start is not None and isinstance(corr, nn.Module) and hasattr(model, "zt") and len(mon.c_val) and \
            len(mon.in_cusp) == T.n_constants(start):
        hover = cusp_hover_statistics(start, np.array(mon.c_val), model.zt[0].detach().cpu().numpy(),
                                      model.rh[0].detach().cpu().numpy())
    if clip_log["flags"]["cusp_stalled"]:
        warn.append("cusp-stalled: a constant inside an sqa argument drives persistent gradient clipping without "
                    "net movement (see clip_log)")
    elif clip_log["flags"]["persistent_clipping"]:
        warn.append("persistent gradient clipping (see clip_log)")
    if hover is not None and hover["suspect"] and not clip_log["flags"]["cusp_stalled"]:
        warn.append("supplementary indicator: a constant hovers at a cusp of sqa without persistent clipping "
                    "(see cusp_hover)")
    out = {"label": label, "role": role, "governing": role == "control", "synthetic": bool(synthetic),
           "control_label": getattr(corr, "label", None) if role != "candidate" else None, "fit": res.summary(),
           "k": int(model.n_params()) if hasattr(model, "n_params") else None,
           "k_equation": int(model.n_correction()) if hasattr(model, "n_correction") else None,
           "k_as_written": getattr(corr, "k_as_written", None),
           "expr_start": T.to_string(start) if start is not None else None, "expr_refit": None,
           "expr_refit_dict": None, "dropped_gauge_terms": list(getattr(corr, "dropped_gauge_terms", [])),
           "stripped_gauge_terms": carried, "audit_refit": None, "audit_passed": None, "audit_certified": None,
           "audit_verdict": None, "finalist": None, "finalist_certified": None, "clip_log": clip_log,
           "cusp_stalled": bool(clip_log["flags"]["cusp_stalled"]), "cusp_hover": hover,
           "cusp_hover_suspect": None if hover is None else bool(hover["suspect"]), "warnings": warn,
           "registered": False}
    refit = model.current_expr() if hasattr(model, "current_expr") else None
    if refit is not None:
        rep = A.audit(refit, audit_box, stage=STAGE_AFTER, carried_stripped=carried, **(audit_kw or {}))
        rj = rep.to_json()
        out.update(expr_refit=T.to_string(refit), expr_refit_dict=T.to_dict(refit), audit_refit=rj,
                   audit_passed=bool(rep.passed), audit_certified=bool(rep.certified),
                   audit_verdict=audit_verdict(rj), stripped_gauge_terms=rj["stripped_gauge_terms"])
        if log is not None:
            log(f"[{label}] audit of the refitted {'equation' if role == 'candidate' else role}: {rep.summary()}")
        if finalist:
            kw = {k: v for k, v in (audit_kw or {}).items() if k in ("limit", "gauge", "gauge_vars", "domain_kw",
                                                                     "magnitude_kw")}
            cert = A.certify_finalist(refit, audit_box, tail_criterion=tail_criterion, baseline_hash=baseline_hash,
                                      measured_tail=measured_tail, carried_stripped=carried, **kw)
            out.update(finalist=cert, finalist_certified=bool(cert["certified"]))
            if log is not None:
                log(f"[{label}] finalist certificate on the refitted constants: "
                    f"{'CERTIFIED' if cert['certified'] else 'NOT CERTIFIED: ' + '; '.join(cert['reasons'])}")
    elif finalist:
        out.update(finalist={"certified": False, "reasons": ["the model has no equation to certify"]},
                   finalist_certified=False)
    return out


def refit_summary(out: Mapping) -> str:
    f, v = out["fit"], out.get("audit_verdict")
    txt = (f"{out['label']}: k={out['k']} steps={f['step']} skipped={f['skipped_loss'] + f['skipped_grad']} "
           f"audit_refit={'n/a' if out['audit_passed'] is None else ('PASS' if out['audit_passed'] else 'REJECT')}")
    if v is not None:
        up = v["H_upper"]
        txt += (f" magnitude={v['magnitude']} H<={'n/a' if up is None else format(up, '.4g')} "
                f"certified={v['certified']}")
    if out.get("finalist_certified") is not None:
        txt += f" finalist_certified={out['finalist_certified']}"
    cl = out.get("clip_log")
    if cl is not None:
        txt += f" clipped_steps={cl['n_clipped']}/{cl['n_steps']} cusp_stalled={out.get('cusp_stalled')}"
    if out.get("stripped_gauge_terms"):
        txt += f" stripped_gauge_terms={len(out['stripped_gauge_terms'])}"
    return txt


def register_refit(*args, **kwargs):
    """REGISTRATION HOOK - STUB, deliberately not implemented in this session.

    What it has to do on the H200 (PLAN section 13, A6 revised, points 4, 5, 7 and Gate 2):
      * refuse a candidate whose ``audit_passed`` is not True (the audit of the REFITTED constants); refuse a
        FINALIST whose ``finalist_certified`` is not True (``fit_candidate(..., finalist=True, tail_criterion=...,
        baseline_hash=...)``: proven magnitude bound, domain, numerics and tail check, all on the refitted
        constants);
      * the governing C2 is tanh(polynomial) (``control_c2(base, k)``, ``role="control"``; S1): only it enters
        Delta(SR) - Delta(C2). The plain polynomial may be registered as a diagnostic row (``role="diagnostic"``)
        with its ``audit_verdict`` beside it (``c2_report``);
      * store ``stripped_gauge_terms`` (S3) and ``clip_log`` / ``cusp_stalled`` (S5) with the row; a finalist is
        certified against the tail-criterion record of the baseline of record (S4: ``tail_criterion_for_baseline``
        run once for the refitted Track A finalist, bound to its checkpoint hash);
      * evaluate the refitted model with the registry's evaluation (unmodified ``common.test_nll`` inside
        ``sfm.metric.deterministic()``, validation split) and write one registry row with the machine tag, the
        hash of the equation (``translate.expr_hash``), k, C, the search seed and tier, and the audit report;
      * pair the per-event array only against benchmark arrays of the same machine and fold points;
      * do the same for the controls C1 and C2 of the same schedule, so that Delta(SR) - Delta(C2) is a paired
        quantity; count every audited front member sent to the H200 for the winner's-curse correction;
      * write ``search.record_val_nll_check(seed_dir, ...)`` for the plateau rule from these H200 numbers only.
    ``sfm.registry`` was being changed (machine tags) while this module was written, so nothing here calls it.
    """
    raise NotImplementedError("register_refit is a stub: registration is done on the H200 through sfm.registry "
                              "(see the docstring for what it must do)")
