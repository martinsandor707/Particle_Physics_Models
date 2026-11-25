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
``sfm.sr.audit.audit`` and returns the report; a refit whose audit fails must not be registered.

Controls (same schedule = the same ``FitConfig`` as the candidate, only ``name`` / ``out_dir`` differ)
    C1  ``control_c1(base)``: the baseline alone, all its constants trainable, continued with the candidate's
        schedule (what the refit alone buys; needed because the candidate's refit moves the baseline too).
    C2  ``control_c2(base, k)``: a smooth tensor polynomial with the same number of constants k,
            f = sum_j c_j u^a v^b t^c,  u, v in [-1, 1] the rescaled zt and sqrt(rh) over the data box, t = x,
        the first k monomials in graded lexicographic order, excluding the constant and the pure powers of x
        (the softmax cannot see them), zero start (the model starts exactly at the baseline). SR is credited with
        Delta(SR) - Delta(C2).
    C3  (seed-null) is a search control: ``targets.build_residual_target(..., halves=...)``.

Trajectories: ``SubstitutedProvider(provider, {name: expr})`` replaces chosen raw parameters of a Track-A provider by
translated functions of x; k = the provider's count minus that of the replaced rows plus the equations' constants.
"""
from __future__ import annotations

import copy
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

__all__ = ["CorrectedModel", "expression_correction", "PolyControl", "c2_monomials", "control_c1", "control_c2",
           "SubstitutedProvider", "fit_candidate", "check_gradients", "register_refit", "refit_summary",
           "RefitNotAllowed", "ENV_ALLOW"]

ENV_ALLOW = "SFM_SR_REFIT_ALLOWED"


class RefitNotAllowed(RuntimeError):
    """Refits and scores of equations are made on the machine of record (PLAN section 13, A6 revised, point 7)."""


def _x_of_lnp(lnp: torch.Tensor, v0: float, vs: float, mass: float) -> torch.Tensor:
    p = torch.exp(lnp)
    return (torch.log(p * p / (torch.sqrt(p * p + mass * mass) + mass)) - v0) / vs


def expression_correction(expr, variable_names: Sequence[str] = O.VARIABLE_NAMES, dtype=torch.float64):
    """The translated equation as a correction module f(zt, rh, x); all free constants trainable."""
    e = T.parse(expr) if isinstance(expr, str) else expr
    return T.TorchExpr(e, variable_names, dtype=dtype, trainable=True)


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


class PolyControl(nn.Module):
    """Control C2: f = sum_j c_j u^a_j v^b_j t^c_j with u = zt and v = sqrt(rh) rescaled to [-1, 1] over the data box
    (z' in [-200, 1450] mm, r in [0, 700] mm) and t = x; k constants, zero start."""

    def __init__(self, k: int, dtype=torch.float64):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.exponents = c2_monomials(self.k)
        box = O.data_box()
        self.z_lo, self.z_hi = box["zt"]
        self.s_hi = float(np.sqrt(box["rh"][1]))
        self.c = nn.Parameter(torch.zeros(self.k, dtype=dtype))

    def features(self, zt, rh, x):
        u = 2.0 * (zt - self.z_lo) / (self.z_hi - self.z_lo) - 1.0
        v = 2.0 * torch.sqrt(rh) / self.s_hi - 1.0         # rh >= 0 at the cell centres: no guard needed
        return [u ** a * v ** b * x ** c for a, b, c in self.exponents]

    def forward(self, zt, rh, x):
        zt, rh, x = (t.to(self.c.dtype) for t in (zt, rh, x))
        out = 0.0
        for cj, m in zip(self.c, self.features(zt, rh, x)):
            out = out + cj * m
        return out

    def n_params(self) -> int:
        return self.k


def control_c1(base: nn.Module) -> nn.Module:
    """C1: a deep copy of the baseline with every parameter trainable - "refit only, same schedule"."""
    m = copy.deepcopy(base)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def control_c2(base: nn.Module, k: int, freeze_base: bool = False) -> CorrectedModel:
    """C2: the baseline plus a tensor polynomial with k constants (k = the candidate equation's free constants,
    ``CorrectedModel.n_correction()``), zero start; ``freeze_base`` as in the candidate's refit."""
    return CorrectedModel(copy.deepcopy(base), PolyControl(k), freeze_base=freeze_base)


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
        self.exprs = nn.ModuleList([expression_correction(e, ("x",)) for e in replacements.values()])
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


def fit_candidate(model: nn.Module, train, val, config, *, label: str, synthetic: bool = False,
                  audit_box: Optional[Mapping] = None, audit_kw: Optional[dict] = None, log=print) -> dict:
    """Likelihood refit through the existing harness ``sfm.train.fit`` (all trainable constants, training hits, the
    metric's fold, no clamps), followed by the mandatory audit of the refitted equation.

    Allowed only on the machine of record (environment ``SFM_SR_REFIT_ALLOWED=1``) or with ``synthetic=True`` for
    synthetic showers; otherwise ``RefitNotAllowed``. Returns
        {"label", "fit": FitResult.summary(), "k", "expr_start", "expr_refit", "expr_refit_dict", "audit_refit",
         "audit_passed", "registered": False}
    The validation numbers inside ``fit`` are the harness's monitoring values; the score of record is produced by
    the registry on the H200 (``register_refit``).
    """
    if not synthetic and os.environ.get(ENV_ALLOW) != "1":
        raise RefitNotAllowed(
            "refits and scores of PySR equations are produced on the H200 (PLAN section 13, A6 revised, point 7): "
            f"set {ENV_ALLOW}=1 there, or pass synthetic=True for synthetic showers")
    from .. import train as sfm_train

    start = model.current_expr() if hasattr(model, "current_expr") else None
    res = sfm_train.fit(model, train, val, config, meta={"sr_label": label, "synthetic": bool(synthetic)}, log=log)
    out = {"label": label, "synthetic": bool(synthetic), "fit": res.summary(),
           "k": int(model.n_params()) if hasattr(model, "n_params") else None,
           "expr_start": T.to_string(start) if start is not None else None, "expr_refit": None,
           "expr_refit_dict": None, "audit_refit": None, "audit_passed": None, "registered": False}
    refit = model.current_expr() if hasattr(model, "current_expr") else None
    if refit is not None:
        rep = A.audit(refit, audit_box, **(audit_kw or {}))
        out.update(expr_refit=T.to_string(refit), expr_refit_dict=T.to_dict(refit), audit_refit=rep.to_json(),
                   audit_passed=bool(rep.passed))
        if log is not None:
            log(f"[{label}] audit of the refitted equation: {rep.summary()}")
    return out


def refit_summary(out: Mapping) -> str:
    f = out["fit"]
    return (f"{out['label']}: k={out['k']} steps={f['step']} skipped={f['skipped_loss'] + f['skipped_grad']} "
            f"audit_refit={'n/a' if out['audit_passed'] is None else ('PASS' if out['audit_passed'] else 'REJECT')}")


def register_refit(*args, **kwargs):
    """REGISTRATION HOOK - STUB, deliberately not implemented in this session.

    What it has to do on the H200 (PLAN section 13, A6 revised, points 4, 5, 7 and Gate 2):
      * refuse a refit whose ``audit_passed`` is not True (the audit of the REFITTED constants);
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
