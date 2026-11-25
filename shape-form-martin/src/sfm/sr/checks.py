"""sfm.sr.checks - the shape checks of PLAN.md section 7 (Stage 2 gate and Gate 2), NumPy only.

Every function returns numbers and a verdict against a tolerance that is an ARGUMENT with a stated default; the
defaults are this module's proposals (STATUS.md section 7) except where the plan fixes the number.

    sign_changes(v, min_peak)             sign changes of a sequence, counting only lobes whose peak is >= min_peak
    count_extrema(v, prominence)          turning points of a profile with at least this prominence
    containment(P, z_edges, r_edges)      r50, r90, z50, z90 of a normalised map
    monotone_sequence(values, tol)        "r50, r90, z50, z90 monotone from 20 -> 30 -> 50 GeV": no reversal beyond tol
    slope_sign_changes(log_rho, coord)    sign changes of d ln rho / d coord along a profile (tails)
    baseline_tail_slopes(log_rho, z, r)   largest tail slopes of a grid density beyond the grid (summary numbers)
    decide_tail_criterion(...)            the tail criterion of a baseline, decided once from the baseline alone
    verify_tail_criterion(record, hash)   integrity and baseline binding of that record
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

import hashlib
import json
import math
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import audit as A
from . import operators as O
from . import translate as T

__all__ = ["MLP_BASE_PERIOD_MM", "sign_changes", "count_extrema", "containment", "monotone_sequence",
           "slope_sign_changes", "baseline_tail_slopes", "decide_tail_criterion", "verify_tail_criterion",
           "format_tail_criterion", "TAIL_R_MIN_MM", "TAIL_R_MAX_MM", "RADIAL_SLOPE_LIMIT", "modulation_expr", "template_modulation", "modulation_amplitude", "trig_periods",
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


def baseline_tail_slopes(log_rho, z_mm, r_mm, n_fit: int = 4) -> dict:
    """Summary tail slopes of a grid density at the edge of the grid (two numbers, for a table). The finalist's
    tail check does NOT use them any more: it uses the record of ``decide_tail_criterion`` (PLAN 17.11, S4).

    log_rho   (NZ, NR) or (B, NZ, NR) log-density (log P - log cell volume) of the REFITTED model's baseline part
              at the momenta of interest, on the grid with cell centres ``z_mm`` (NZ,), ``r_mm`` (NR,)
    Returns   "dlnr": the largest, over depth rows (and maps), least-squares slope of ln rho against ln r on the
              last ``n_fit`` interior rings (the outermost ring is left out: it is the border cell of the metric);
              "dz_per_mm": the largest, over rings (and maps), least-squares slope of ln rho against z' on the
              last ``n_fit`` rows before the last one; plus the medians, for information.
    The grid ends at r = 700 mm and z' = 1450 mm; beyond it the baseline is taken to continue as this power law
    in r and exponential in z' (an assumption: nothing constrains the baseline there)."""
    lr = np.asarray(log_rho, dtype=np.float64)
    lr = lr[None] if lr.ndim == 2 else lr
    z, r = np.asarray(z_mm, dtype=np.float64), np.asarray(r_mm, dtype=np.float64)
    if lr.shape[1:] != (len(z), len(r)) or n_fit < 2 or len(r) < n_fit + 1 or len(z) < n_fit + 1:
        raise ValueError("log_rho must be (NZ, NR) or (B, NZ, NR) with at least n_fit + 1 rings and rows")

    def ls_slope(y, t):                                     # y (..., n), t (n,)
        tc = t - t.mean()
        return np.sum((y - y.mean(-1, keepdims=True)) * tc, -1) / np.sum(tc * tc)
    sr = ls_slope(lr[:, :, -(n_fit + 1):-1], np.log(r[-(n_fit + 1):-1]))                 # (B, NZ)
    sz = ls_slope(np.moveaxis(lr[:, -(n_fit + 1):-1, :], 1, -1), z[-(n_fit + 1):-1])     # (B, NR)
    sr, sz = sr[np.isfinite(sr)], sz[np.isfinite(sz)]
    if not len(sr) or not len(sz):
        raise ValueError("no finite tail slope: the map is not finite in its last rings / rows")
    return {"dlnr": float(sr.max()), "dz_per_mm": float(sz.max()), "dlnr_median": float(np.median(sr)),
            "dz_per_mm_median": float(np.median(sz)), "n_fit": int(n_fit),
            "r_fit_mm": [float(r[-(n_fit + 1)]), float(r[-2])], "z_fit_mm": [float(z[-(n_fit + 1)]), float(z[-2])]}


# ----------------------------------------------------------------------------------------------------------------------
# Tail criterion of a baseline (PLAN 17.11, S4): decided once, from the baseline alone
# ----------------------------------------------------------------------------------------------------------------------
TAIL_SCHEMA = "sfm.sr.tail_criterion/1"
TAIL_R_MIN_MM, TAIL_R_MAX_MM = 200.0, 1500.0
RADIAL_SLOPE_LIMIT = -2.0


def _tail_record_hash(rec: Mapping) -> str:
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def decide_tail_criterion(log_rho, z_mm, r_mm, momenta_gev, z90_mm, *, baseline_hash: str, label: str,
                          r_min_mm: float = TAIL_R_MIN_MM, r_max_mm: float = TAIL_R_MAX_MM,
                          slope_limit: float = RADIAL_SLOPE_LIMIT, n_z_bands: int = 8, n_depth_bands: int = 4,
                          note: str = "", allow_depth_alternative: bool = False) -> dict:
    """Decide the tail criterion of a baseline ONCE, from the baseline alone (delta = 0), and return the record
    that ``audit.certify_finalist`` needs.

    log_rho      (P, NZ, NR) ln of the 3-D DENSITY rho(z', r) of the baseline at the points (z_mm[i], r_mm[j]) for
                 each momentum - log P minus the log cell volume, or the density functional itself. NOT the ring
                 mass 2 pi r rho (whose log-log slope is larger by one) and NOT log P of unequal cells.
                 A constant per momentum (normalisation) does not matter.
    z_mm, r_mm   ascending coordinates; r_mm should reach ``r_max_mm`` (1500 mm): a grid map ends at 700 mm, a
                 closed-form baseline can be continued (``refit.continued_log_density``)
    momenta_gev  (P,) the pre-registered momenta, ``CHECK_MOMENTA_GEV`` = 0.05, 0.1, 1, 5, 20, 30, 50 GeV
    z90_mm       (P,) depth containing 90 % of the baseline's energy at each momentum (``containment``)
    baseline_hash, label   what the record is bound to (e.g. the SHA-256 of the baseline's checkpoint)

    Radial rule. d ln rho / d ln r is taken by finite differences between adjacent radii, at every depth given and
    every momentum, for all intervals that overlap [r_min_mm, r_max_mm]. If its maximum is <= ``slope_limit`` (-2:
    rho integrable in r dr) the rule is "loglog_slope_le_limit"; otherwise - the baseline is itself flatter, as
    with the legacy far halo exp(-r/R) / (r + 1 cm) - the rule is "monotone_and_no_flattening": d_r ln rho < 0 and
    d_r delta <= 0 for r > 200 mm. Depth rule: d ln rho / d z' <= 0 beyond z_90, by finite differences between
    adjacent depths beyond z_90(p), at every radius up to ``r_max_mm``.

    Depth, when the baseline itself fails (NOT covered by the user's decision S4, which names an alternative for
    the radial rule only): the record says so (``depth.baseline_satisfies`` False, where and by how much) and the
    governing rule stays the accepted "dz_le_0_beyond_z90" - under which no correction can then be certified -
    unless ``allow_depth_alternative=True`` is passed, which is to be done only after the user has decided it:
    the rule then becomes "no_depth_flattening", d_z' delta <= 0 beyond z_90, the analogue of the radial
    alternative. ``audit.tail_check`` always reports both.

    The record stores the measured slopes (maxima per momentum, depth band and radial interval; per momentum,
    depth band beyond z_90 and radius), whether the baseline itself satisfies each rule, the chosen rule, the
    baseline hash and its own hash (``verify_tail_criterion``). The choice is a property of the baseline: it is
    not, and cannot be, made per equation."""
    lr = np.asarray(log_rho, dtype=np.float64)
    z, r = np.asarray(z_mm, dtype=np.float64), np.asarray(r_mm, dtype=np.float64)
    mom, z90 = np.asarray(momenta_gev, dtype=np.float64), np.asarray(z90_mm, dtype=np.float64)
    if lr.ndim != 3 or lr.shape != (len(mom), len(z), len(r)) or z90.shape != mom.shape:
        raise ValueError("log_rho must be (P, NZ, NR) with momenta (P,), z90 (P,), z (NZ,), r (NR,)")
    if np.any(np.diff(z) <= 0) or np.any(np.diff(r) <= 0) or r[0] < 0:
        raise ValueError("z_mm and r_mm must be ascending, r_mm non-negative")
    if not baseline_hash or not isinstance(baseline_hash, str):
        raise ValueError("baseline_hash is required: the criterion is bound to one baseline")
    P, NZ, NR = lr.shape
    # ---- radial --------------------------------------------------------------------------------------------------
    iv = np.flatnonzero((r[1:] > r_min_mm) & (r[:-1] < r_max_mm * (1 + 1e-9)) & (r[:-1] > 0))
    if len(iv) < 1:
        raise ValueError("no radial interval overlaps [r_min_mm, r_max_mm]")
    i0, i1 = int(iv[0]), int(iv[-1]) + 1
    rn = r[i0:i1 + 1]
    with np.errstate(invalid="ignore"):
        sl = np.diff(lr[:, :, i0:i1 + 1], axis=2) / np.diff(np.log(rn))[None, None, :]          # (P, NZ, ni)
    fin = np.isfinite(sl)
    bands = [b for b in np.array_split(np.arange(NZ), min(n_z_bands, NZ)) if len(b)]
    zb = [float(z[b[0]]) for b in bands] + [float(z[-1])]
    M = np.stack([np.where(fin[:, b, :], sl[:, b, :], -np.inf).max(1) for b in bands], 1)       # (P, nb, ni)
    slm = np.where(fin, sl, -np.inf)
    gmax = float(slm.max())
    w = np.unravel_index(int(np.argmax(slm)), slm.shape)
    per_p = slm.reshape(P, -1).max(1)
    frac = float(np.mean(sl[fin] > slope_limit)) if fin.any() else float("nan")
    satisfies = bool(gmax <= slope_limit + 1e-12)
    monotone = bool(gmax < 0.0)
    # ---- depth beyond z_90 ---------------------------------------------------------------------------------------
    jr = np.flatnonzero(r <= r_max_mm * (1 + 1e-9))
    with np.errstate(invalid="ignore"):
        dz = np.diff(lr[:, :, jr], axis=1) / np.diff(z)[None, :, None]                           # (P, NZ - 1, nr)
    D = np.full((P, n_depth_bands, len(jr)), -np.inf)
    dbe = np.zeros((P, n_depth_bands + 1))
    dmax, dwhere = -np.inf, None
    for j in range(P):
        ks = np.flatnonzero(z[:-1] > z90[j])
        if len(ks) < 1:
            raise ValueError(f"no depth interval beyond z_90 = {z90[j]:.0f} mm at p = {mom[j]:g} GeV")
        groups = [g for g in np.array_split(ks, min(n_depth_bands, len(ks))) if len(g)]
        edges = [float(z[g[0]]) for g in groups] + [float(z[ks[-1] + 1])]
        for b in range(n_depth_bands):
            g = groups[b] if b < len(groups) else groups[-1][-1:]
            v = dz[j][g]
            D[j, b] = np.where(np.isfinite(v), v, -np.inf).max(0)
            dbe[j, b], dbe[j, b + 1] = (edges[b], edges[b + 1]) if b < len(groups) else (edges[-1], edges[-1])
        if D[j].max() > dmax:
            dmax = float(D[j].max())
            bw, rw = np.unravel_index(int(np.argmax(D[j])), D[j].shape)
            dwhere = {"p_GeV": float(mom[j]), "z_band_mm": [float(dbe[j, bw]), float(dbe[j, bw + 1])],
                      "r_mm": float(r[jr][rw])}
    rec = {
        "schema": TAIL_SCHEMA, "label": str(label), "baseline_hash": str(baseline_hash), "note": str(note),
        "density": "slopes of the 3-D density rho(z', r) (probability per volume), not of the ring mass 2 pi r rho",
        "momenta_GeV": mom.tolist(), "x": O.x_of_p(mom).tolist(), "r_min_mm": float(r_min_mm),
        "r_max_mm": float(r_max_mm), "slope_limit": float(slope_limit), "r_nodes_mm": rn.tolist(),
        "z_range_mm": [float(z[0]), float(z[-1])], "n_depths": int(NZ),
        "covers_r_max": bool(rn[-1] >= r_max_mm * (1 - 0.05)),
        "radial": {"z_band_edges_mm": zb, "max_loglog_slope": np.where(np.isfinite(M), M, -1e300).tolist(),
                   "baseline_max": gmax, "baseline_max_at": {"p_GeV": float(mom[w[0]]), "z_mm": float(z[w[1]]),
                                                             "r_mm": float(np.sqrt(rn[w[2]] * rn[w[2] + 1]))},
                   "baseline_max_per_momentum": per_p.tolist(), "fraction_above_limit": frac,
                   "baseline_satisfies_limit": satisfies, "baseline_monotone": monotone,
                   "n_nonfinite": int((~fin).sum())},
        "depth": {"z90_mm": z90.tolist(), "z_band_edges_mm": dbe.tolist(), "r_nodes_mm": r[jr].tolist(),
                  "max_dz_per_mm": np.where(np.isfinite(D), D, -1e300).tolist(), "baseline_max": dmax,
                  "baseline_max_at": dwhere, "baseline_satisfies": bool(dmax <= 1e-12),
                  "baseline_max_per_momentum": D.reshape(P, -1).max(1).tolist(),
                  "alternative_allowed": bool(allow_depth_alternative)},
        "criterion": {"radial": "loglog_slope_le_limit" if satisfies else "monotone_and_no_flattening",
                      "depth": "no_depth_flattening" if (allow_depth_alternative and dmax > 1e-12)
                      else "dz_le_0_beyond_z90",
                      "decided_from": "the baseline alone (delta = 0); not selectable per equation"},
    }
    rec["record_hash"] = _tail_record_hash(rec)
    return rec


def verify_tail_criterion(record: Mapping, baseline_hash: Optional[str] = None) -> None:
    """Raise ``ValueError`` unless the record is intact (its hash matches its content) and, if ``baseline_hash``
    is given, bound to that baseline. ``certify_finalist`` calls it with the hash of the baseline in hand."""
    if record.get("schema") != TAIL_SCHEMA:
        raise ValueError(f"not a tail-criterion record (schema {record.get('schema')!r})")
    if record.get("record_hash") != _tail_record_hash(record):
        raise ValueError("the tail-criterion record was altered after it was made (hash mismatch)")
    if record["criterion"]["radial"] != ("loglog_slope_le_limit" if record["radial"]["baseline_satisfies_limit"]
                                         else "monotone_and_no_flattening"):
        raise ValueError("the radial rule of the record does not follow from its measured baseline slopes")
    dep = record["depth"]
    want = "no_depth_flattening" if (dep.get("alternative_allowed") and not dep["baseline_satisfies"]) \
        else "dz_le_0_beyond_z90"
    if record["criterion"]["depth"] != want:
        raise ValueError("the depth rule of the record does not follow from its measured baseline slopes")
    if baseline_hash is None:
        return
    if record.get("baseline_hash") != baseline_hash:
        raise ValueError(f"the tail criterion belongs to baseline {record.get('baseline_hash')!r}, not to "
                         f"{baseline_hash!r}: decide it for this baseline (checks.decide_tail_criterion)")


def format_tail_criterion(record: Mapping) -> str:
    """Three lines for a log: what was measured on the baseline and which rule follows."""
    ra, de = record["radial"], record["depth"]
    at = ra["baseline_max_at"]
    return "\n".join([
        f"tail criterion of [{record['label']}] (baseline {record['baseline_hash'][:16]}, record "
        f"{record['record_hash'][:12]}); slopes of the 3-D density rho",
        f"  radial, r in [{record['r_nodes_mm'][0]:.0f}, {record['r_nodes_mm'][-1]:.0f}] mm: largest d ln rho / d ln r"
        f" = {ra['baseline_max']:.3f} at p = {at['p_GeV']:g} GeV, z' = {at['z_mm']:.0f} mm, r = {at['r_mm']:.0f} mm; "
        f"<= {record['slope_limit']:g} everywhere: {ra['baseline_satisfies_limit']}; monotone: "
        f"{ra['baseline_monotone']}  ->  rule \"{record['criterion']['radial']}\"",
        f"  depth beyond z_90: largest d ln rho / d z' = {de['baseline_max']:.3g} / mm; <= 0 everywhere: "
        f"{de['baseline_satisfies']}  ->  rule \"{record['criterion']['depth']}\""])


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
