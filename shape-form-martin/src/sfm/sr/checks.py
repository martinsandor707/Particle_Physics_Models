"""sfm.sr.checks - the shape checks of PLAN.md section 7 (Stage 2 gate and Gate 2), NumPy only.

Every function returns numbers and a verdict against a tolerance that is an ARGUMENT with a stated default; the
defaults are this module's proposals (STATUS.md section 7) except where the plan fixes the number.

    sign_changes(v, min_peak)             sign changes of a sequence, counting only lobes whose peak is >= min_peak
    count_extrema(v, prominence)          turning points of a profile with at least this prominence
    containment(P, z_edges, r_edges)      r50, r90, z50, z90 of a normalised map
    monotone_sequence(values, tol)        "r50, r90, z50, z90 monotone from 20 -> 30 -> 50 GeV": no reversal beyond tol
    slope_sign_changes(log_rho, coord)    sign changes of d ln rho / d coord along a profile (tails)
    modulation_expr(expr)                 f minus f with every sin / cos replaced by 0 (free-form Stage 2), or
    template_modulation(parts)            the modulation term of the Stage-2 templates
    modulation_amplitude(expr, box)       sup |modulation| on the extended box (tight machinery of the audit), <= 0.3
    trig_periods(expr, box)               local period ranges in mm of every sin / cos along z' and r
    period_check(period_mm)               nearest 1650/k mm (k = 1..8), relative distance, verdict
    seed_null_comparison(...)             loss reduction on the target against the seed-null target

Period check (PLAN sections 13 and 14, corrected values): the MLP's depth features have periods 1650/k mm. A fitted
period "matches" if it lies within ``rel_tol`` (default 3 %) of 1650/k for some k = 1..8; periods above
``trend_above`` (default 570 mm, i.e. k = 1, 2) count as trend, not as a match. Honest size of the test: bands of
+-3 % around 1650/k for k = 3..8 cover about a third of the interval 190-570 mm, so a non-match is weak evidence and
a match is not proof of copying.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import audit as A
from . import operators as O
from . import translate as T

__all__ = ["MLP_BASE_PERIOD_MM", "sign_changes", "count_extrema", "containment", "monotone_sequence",
           "slope_sign_changes", "modulation_expr", "template_modulation", "modulation_amplitude", "trig_periods",
           "period_check", "period_band_coverage", "seed_null_comparison", "CHECK_MOMENTA_GEV",
           "MODULATION_LIMIT_NAT"]

MLP_BASE_PERIOD_MM = 1650.0                     # periods of the MLP's depth features: 1650 / k mm (PLAN section 13)
CHECK_MOMENTA_GEV = (0.05, 0.1, 1.0, 5.0, 20.0, 30.0, 50.0)       # PLAN section 7, Stage 2
MODULATION_LIMIT_NAT = 0.3


def sign_changes(v, min_peak: float = 0.0) -> int:
    """Number of sign changes of a sequence. Lobes (maximal runs of one sign) whose largest |value| is below
    ``min_peak`` are ignored, so numerical ripple around zero does not count."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v) & (v != 0)]
    if not len(v):
        return 0
    sgn = np.sign(v)
    starts = np.concatenate([[0], np.flatnonzero(sgn[1:] != sgn[:-1]) + 1, [len(v)]])
    lobes = [(sgn[a], float(np.max(np.abs(v[a:b])))) for a, b in zip(starts[:-1], starts[1:])]
    kept = [s for s, peak in lobes if peak >= min_peak]
    return int(sum(1 for a, b in zip(kept[:-1], kept[1:]) if a != b))


def count_extrema(v, prominence: float = 0.01) -> int:
    """Turning points of a profile with prominence >= ``prominence`` (default 0.01 nat for log-profiles: below the
    0.02-0.05 nat reproducible ringing of the MLP map). A monotone profile has 0, a single bump 1."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return 0
    n, direction, lo, hi = 0, 0, v[0], v[0]                # hysteresis: a turn counts once it is undone by >= prominence
    for x in v[1:]:
        if direction == 0:
            lo, hi = min(lo, x), max(hi, x)
            if x - lo >= prominence:
                direction, hi = 1, x
            elif hi - x >= prominence:
                direction, lo = -1, x
        elif direction == 1:
            if x > hi:
                hi = x
            elif hi - x >= prominence:
                n, direction, lo = n + 1, -1, x
        else:
            if x < lo:
                lo = x
            elif x - lo >= prominence:
                n, direction, hi = n + 1, 1, x
    return n


def containment(P, z_edges, r_edges, fractions: Sequence[float] = (0.5, 0.9)) -> Dict[str, float]:
    """z and r below which the given fractions of a map P (NZ, NR) lie (linear interpolation inside the bin):
    {"z50": ..., "z90": ..., "r50": ..., "r90": ...} in the units of the edges."""
    P = np.asarray(P, dtype=np.float64)
    P = P / P.sum()
    out = {}
    for name, prof, edges in (("z", P.sum(1), np.asarray(z_edges, float)), ("r", P.sum(0), np.asarray(r_edges, float))):
        cdf = np.concatenate([[0.0], np.cumsum(prof)])
        for f in fractions:
            out[f"{name}{int(round(100 * f))}"] = float(np.interp(f, cdf, edges))
    return out


def monotone_sequence(values: Sequence[float], tol: float = 0.5) -> dict:
    """A sequence is monotone if its steps do not change direction by more than ``tol`` (default 0.5 mm for
    containment radii at 20 -> 30 -> 50 GeV). Returns the steps, the direction and the verdict."""
    v = np.asarray(values, dtype=np.float64)
    steps = np.diff(v)
    up, down = bool(np.all(steps >= -tol)), bool(np.all(steps <= tol))
    return {"values": v.tolist(), "steps": steps.tolist(), "monotone": up or down,
            "direction": "flat" if (up and down) else ("up" if up else ("down" if down else "reversal")),
            "largest_reversal": float(min(np.max(np.maximum(-steps, 0.0)), np.max(np.maximum(steps, 0.0))))}


def slope_sign_changes(log_rho, coord, start: Optional[float] = None, min_peak: float = 0.0) -> dict:
    """Sign changes of d ln rho / d coord along a profile for coord >= start ("no sign change of d_r ln rho for
    r > 200 mm", "of d_z' ln rho beyond z90"). ``min_peak`` is in nat per unit of coord."""
    y, c = np.asarray(log_rho, dtype=np.float64), np.asarray(coord, dtype=np.float64)
    keep = np.isfinite(y) & (c >= (start if start is not None else -math.inf))
    y, c = y[keep], c[keep]
    if len(y) < 3:
        return {"n": 0, "max_slope": math.nan, "min_slope": math.nan, "points": int(len(y))}
    d = np.diff(y) / np.diff(c)
    return {"n": sign_changes(d, min_peak), "max_slope": float(d.max()), "min_slope": float(d.min()),
            "points": int(len(y))}


# ----------------------------------------------------------------------------------------------------------------------
# Modulation
# ----------------------------------------------------------------------------------------------------------------------
def modulation_expr(e: T.Expr) -> T.Expr:
    """m = f - f|_{sin = cos = 0}: the part of a free-form Stage-2 expression that exists because of the
    trigonometric operators (the plan's "modulation" is defined only for the template; this is the extension)."""
    def zero_trig(n: T.Expr) -> T.Expr:
        if n.kind == "op" and n.name in ("sin", "cos"):
            return T.const(0.0, fixed=True)
        if not n.args:
            return n
        return T.Expr(n.kind, n.name, tuple(zero_trig(a) for a in n.args), n.value, n.fixed)
    return T.op("-", e, zero_trig(e))


def template_modulation(subexpressions: Mapping[str, str], template: str = "s2_modulation") -> T.Expr:
    """The modulation term 0.3 tanh(a) / (1 + e^2) sin(ph) of a Stage-2 template result."""
    full = T.parse_template(template, subexpressions)
    if not (full.kind == "op" and full.name == "+"):
        raise ValueError("the template is not a sum of a trend and a modulation")
    return full.args[1]


def modulation_amplitude(modulation: T.Expr, box: Optional[Mapping] = None, limit: float = MODULATION_LIMIT_NAT,
                         data_box: Optional[Mapping] = None) -> dict:
    """sup |modulation| on the extended box against the plan's 0.3 nat (gauge "none": the modulation is a
    definite function), and the number of its sign changes along z' and r profiles inside / outside the data box
    (PLAN: at most 4 inside the data range, none outside; lobes below 0.003 nat are ignored)."""
    box = dict(O.extended_box() if box is None else box)
    dbox = dict(O.data_box() if data_box is None else data_box)
    mag = A.magnitude_bound(modulation, box, limit=limit, gauge="none")
    used = T.variables(modulation)
    res = {"limit": limit, "amplitude": mag, "passed_amplitude": bool(mag["passed"]), "sign_changes": {}}
    worst_in = worst_out = 0
    for prof in ("zt", "rh"):
        if prof not in used:
            continue
        others = [v for v in used if v != prof]
        grid = np.linspace(box[prof][0], box[prof][1], 4001)
        inside = (grid >= dbox[prof][0]) & (grid <= dbox[prof][1])
        n_in = n_out = 0
        combos = np.array(np.meshgrid(*[np.linspace(dbox[o][0], dbox[o][1], 7) for o in others])).reshape(len(others), -1).T \
            if others else np.zeros((1, 0))
        for row in combos:
            env = {prof: grid, **{o: np.full_like(grid, v) for o, v in zip(others, row)}}
            m = np.asarray(T.evaluate(modulation, env)) + np.zeros_like(grid)
            n_in = max(n_in, sign_changes(m[inside], 0.003))
            n_out = max(n_out, sign_changes(m[grid > dbox[prof][1]], 0.003), sign_changes(m[grid < dbox[prof][0]], 0.003))
        res["sign_changes"][prof] = {"inside": n_in, "outside": n_out}
        worst_in, worst_out = max(worst_in, n_in), max(worst_out, n_out)
    res["passed_sign_changes"] = bool(worst_in <= 4 and worst_out == 0)
    return res


def trig_periods(e: T.Expr, box: Optional[Mapping] = None) -> List[dict]:
    """For every sin / cos node: the range of the local period 2 pi / |d phase / d coordinate| in mm along z' and
    r over the box (default: the data box), from interval derivatives of the phase (sound enclosures)."""
    box = dict(O.data_box() if box is None else box)
    scale = {"zt": O.ZETA_SCALE_MM, "rh": O.RHO_SCALE_MM}
    out = []
    for n in T.walk(e):
        if n.kind == "op" and n.name in ("sin", "cos"):
            phase = n.args[0]
            enc = A.enclose(phase, box, grad=True)
            item = {"node": T.to_string(n)[:120], "periods_mm": {}}
            for v in ("zt", "rh"):
                if v in enc["variables"]:
                    glo, ghi = enc["grad_lo"][v], enc["grad_hi"][v]
                    gmax = max(abs(glo), abs(ghi))
                    gmin = 0.0 if glo <= 0 <= ghi else min(abs(glo), abs(ghi))
                    item["periods_mm"][v] = (2 * math.pi * scale[v] / gmax if gmax > 0 else math.inf,
                                             2 * math.pi * scale[v] / gmin if gmin > 0 else math.inf)
            out.append(item)
    return out


def period_check(period_mm: float, base: float = MLP_BASE_PERIOD_MM, k_max: int = 8, rel_tol: float = 0.03,
                 trend_above: float = 570.0) -> dict:
    """Compare a fitted period with the MLP's feature periods base / k, k = 1..k_max."""
    p = float(period_mm)
    ks = np.arange(1, k_max + 1)
    rel = np.abs(p - base / ks) / (base / ks)
    k = int(ks[np.argmin(rel)])
    trend = bool(p > trend_above)
    return {"period_mm": p, "nearest_k": k, "nearest_mm": base / k, "relative_distance": float(rel.min()),
            "trend": trend, "matches_mlp_feature": bool((not trend) and rel.min() <= rel_tol),
            "verdict": "trend" if trend else ("match" if rel.min() <= rel_tol else "no match")}


def period_band_coverage(base: float = MLP_BASE_PERIOD_MM, k_max: int = 8, rel_tol: float = 0.03,
                         trend_above: float = 570.0) -> float:
    """Fraction of the period interval [base/k_max (1 - tol), trend_above] covered by the match bands: the chance
    that an arbitrary period in that interval "matches" (the specificity of the check)."""
    lo = base / k_max * (1 - rel_tol)
    bands = [(max(lo, base / k * (1 - rel_tol)), min(trend_above, base / k * (1 + rel_tol))) for k in range(1, k_max + 1)]
    bands = sorted((a, b) for a, b in bands if b > a)
    cov, end = 0.0, lo
    for a, b in bands:
        a = max(a, end)
        if b > a:
            cov, end = cov + (b - a), b
    return cov / (trend_above - lo)


def seed_null_comparison(loss_null_target: float, loss_fit_target: float, loss_null_seednull: float,
                         loss_fit_seednull: float) -> dict:
    """Control C3 / Stage-2 gate "smaller loss reduction on the seed-null target": relative reductions
    1 - L_fit / L_null of the same search on the target and on the difference of two ensemble halves. The absolute
    reductions are comparable too, because the seed-null is scaled to the teacher's noise variance."""
    r_t = 1.0 - loss_fit_target / loss_null_target
    r_n = 1.0 - loss_fit_seednull / loss_null_seednull
    a_t, a_n = loss_null_target - loss_fit_target, loss_null_seednull - loss_fit_seednull
    return {"relative_reduction_target": r_t, "relative_reduction_seed_null": r_n, "absolute_reduction_target": a_t,
            "absolute_reduction_seed_null": a_n, "noise_share_of_target_reduction": (a_n / a_t if a_t > 0 else math.inf),
            "passed": bool(a_n < a_t and r_n < r_t)}
