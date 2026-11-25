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
           "format_tail_criterion", "TAIL_R_MIN_MM", "TAIL_R_MAX_MM", "RADIAL_SLOPE_LIMIT", "profile_sign_changes",
           "profile_rising_lobes",
           "SIGN_TOL_DEPTH_PER_MM", "SIGN_TOL_RADIAL", "REFERENCE_KINDS", "MEASURED_KIND", "DEPTH_RULE",
           "TAIL_SCHEMA", "modulation_expr", "template_modulation", "modulation_amplitude", "trig_periods",
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
# Tail criterion of a reference baseline (PLAN 17.11, S4, and the user's decision on the depth rule): decided once,
# from the FIXED non-SR baseline alone
# ----------------------------------------------------------------------------------------------------------------------
TAIL_SCHEMA = "sfm.sr.tail_criterion/2"
TAIL_R_MIN_MM, TAIL_R_MAX_MM = 200.0, 1500.0
RADIAL_SLOPE_LIMIT = -2.0
SIGN_TOL_DEPTH_PER_MM = 1e-5         # 0.001 nat per 100 mm: lobes of d ln rho / d z' below this are not counted
SIGN_TOL_RADIAL = 1e-3               # in d ln rho / d ln r
REFERENCE_KINDS = ("track_a_finalist", "c1_refit_only", "legacy_preview", "synthetic")
MEASURED_KIND = "refitted_baseline"  # a table of the jointly refitted baseline: model side only, never a reference
DEPTH_RULE = "inherit_reference"     # d_z' ln rho_model <= max(0, d_z' ln rho_reference) beyond z_90


def _tail_record_hash(rec: Mapping) -> str:
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def _r9(a) -> list:
    """Nested list of floats rounded to 9 significant digits; non-finite values become None."""
    a = np.asarray(a, dtype=np.float64)
    out = np.array([float(f"{v:.9g}") if np.isfinite(v) else np.nan for v in a.ravel()]).reshape(a.shape)
    return json.loads(json.dumps(np.where(np.isfinite(out), out, None).tolist()))


def profile_sign_changes(slopes, tol: float) -> np.ndarray:
    """Sign changes of slope profiles along the LAST axis, counted in a way that depends only on the grid of the
    profile and on ``tol``: non-finite and zero entries are skipped; a lobe is a maximal run of consecutive
    entries of one sign; lobes whose largest |slope| is below ``tol`` are dropped; the count is the number of
    sign alternations between the remaining consecutive lobes. Returns an integer array without the last axis."""
    a = np.asarray(slopes, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1])
    out = np.zeros(len(flat), dtype=np.int64)
    for i, v in enumerate(flat):
        v = v[np.isfinite(v) & (v != 0)]
        if len(v) < 2:
            continue
        sg = np.sign(v)
        starts = np.concatenate([[0], np.flatnonzero(sg[1:] != sg[:-1]) + 1, [len(v)]])
        keep = [sg[a_] for a_, b_ in zip(starts[:-1], starts[1:]) if np.max(np.abs(v[a_:b_])) >= tol]
        out[i] = sum(1 for x, y in zip(keep[:-1], keep[1:]) if x != y)
    return out.reshape(a.shape[:-1])


def profile_rising_lobes(slopes, tol: float) -> np.ndarray:
    """Number of rising stretches per profile along the last axis: lobes as in ``profile_sign_changes`` (same
    skipping of zeros, same tolerance), counting those with positive slope. Unlike the number of sign changes it
    does not depend on whether a profile happens to begin inside a rising stretch."""
    a = np.asarray(slopes, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1])
    out = np.zeros(len(flat), dtype=np.int64)
    for i, v in enumerate(flat):
        v = v[np.isfinite(v) & (v != 0)]
        if not len(v):
            continue
        sg = np.sign(v)
        starts = np.concatenate([[0], np.flatnonzero(sg[1:] != sg[:-1]) + 1, [len(v)]])
        keep = [sg[a_] for a_, b_ in zip(starts[:-1], starts[1:]) if np.max(np.abs(v[a_:b_])) >= tol]
        merged = [x for n_, x in enumerate(keep) if n_ == 0 or x != keep[n_ - 1]]
        out[i] = sum(1 for x in merged if x > 0)
    return out.reshape(a.shape[:-1])


def decide_tail_criterion(log_rho, z_mm, r_mm, momenta_gev, z90_mm, *, baseline_hash: str, label: str,
                          reference_kind: str = "track_a_finalist", r_min_mm: float = TAIL_R_MIN_MM,
                          r_max_mm: float = TAIL_R_MAX_MM, slope_limit: float = RADIAL_SLOPE_LIMIT, note: str = "",
                          rising_annotations: Optional[Mapping] = None) -> dict:
    """The tail-criterion record of a baseline: its slope tables at the pre-registered momenta, where it rises in
    depth beyond z_90, and the rules that follow. Made ONCE for the fixed reference baseline and bound to its hash.

    log_rho      (P, NZ, NR) ln of the 3-D DENSITY rho(z', r) of the baseline at the points (z_mm[i], r_mm[j]) for
                 each momentum - log P minus the log cell volume, or the density functional itself. NOT the ring
                 mass 2 pi r rho (whose log-log slope is larger by one) and NOT log P of unequal cells.
                 A constant per momentum (normalisation) does not matter.
    z_mm, r_mm   ascending coordinates; r_mm should reach ``r_max_mm`` (1500 mm): a grid map ends at 700 mm, a
                 closed-form baseline can be continued (``refit.continued_log_density``)
    momenta_gev  (P,) the pre-registered momenta, ``CHECK_MOMENTA_GEV`` = 0.05, 0.1, 1, 5, 20, 30, 50 GeV
    z90_mm       (P,) depth containing 90 % of the baseline's energy at each momentum (``containment``)
    baseline_hash, label   what the record is bound to (e.g. the SHA-256 of the baseline's checkpoint)
    reference_kind   "track_a_finalist" (default: the finalist BEFORE any PySR refit - the reference of record),
                 "c1_refit_only" (the baseline refitted alone with the candidate's schedule), "legacy_preview",
                 "synthetic"; or ``MEASURED_KIND`` = "refitted_baseline" for the slope table of a jointly refitted
                 baseline, which ``audit.tail_check`` uses on the MODEL side and which is refused as a reference:
                 a joint refit must not be able to bend its own bound.
    rising_annotations   {momentum index: dict} merged into ``depth.rising_regions`` - which component makes the
                 baseline rise there (``refit.tail_criterion_for_baseline`` fills it)

    Radial rule (S4). d ln rho / d ln r by finite differences between adjacent radii, at every depth given and
    every momentum, for all intervals that overlap [r_min_mm, r_max_mm]. If its maximum is <= ``slope_limit`` (-2:
    rho integrable in r dr) the rule is "loglog_slope_le_limit"; otherwise - the baseline is itself flatter, as
    with the legacy far halo - "monotone_and_no_flattening": the model may nowhere be flatter than THIS baseline.
    Depth rule (user decision after the investigation of STATUS section 14), always "inherit_reference": beyond
    z_90 the model may not rise where this baseline falls and may not rise faster where it rises,
    d_z' ln rho_model <= max(0, d_z' ln rho_reference), point by point. Every cell in which the reference itself
    rises is logged (``depth.rising_cells``: momentum index, depth interval index, radius index, slope) and
    summarised per momentum (``depth.rising_regions``), so that a reviewer can check them against the physics.
    Sign changes (Stage 2): ``radial.sign_changes`` / ``depth.sign_changes`` count them per profile with
    ``profile_sign_changes`` (tolerances ``SIGN_TOL_*``) on this record's own grid.

    The record holds the full slope tables, is self-hashed (``verify_tail_criterion``) and cannot be chosen or
    edited per equation."""
    lr = np.asarray(log_rho, dtype=np.float64)
    z, r = np.asarray(z_mm, dtype=np.float64), np.asarray(r_mm, dtype=np.float64)
    mom, z90 = np.asarray(momenta_gev, dtype=np.float64), np.asarray(z90_mm, dtype=np.float64)
    if lr.ndim != 3 or lr.shape != (len(mom), len(z), len(r)) or z90.shape != mom.shape:
        raise ValueError("log_rho must be (P, NZ, NR) with momenta (P,), z90 (P,), z (NZ,), r (NR,)")
    if np.any(np.diff(z) <= 0) or np.any(np.diff(r) <= 0) or r[0] < 0:
        raise ValueError("z_mm and r_mm must be ascending, r_mm non-negative")
    if not baseline_hash or not isinstance(baseline_hash, str):
        raise ValueError("baseline_hash is required: the criterion is bound to one baseline")
    if reference_kind not in REFERENCE_KINDS + (MEASURED_KIND,):
        raise ValueError(f"reference_kind must be one of {REFERENCE_KINDS + (MEASURED_KIND,)}")
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
    slm = np.where(fin, sl, -np.inf)
    gmax = float(slm.max())
    w = np.unravel_index(int(np.argmax(slm)), slm.shape)
    satisfies = bool(gmax <= slope_limit + 1e-12)
    # ---- depth beyond z_90 ---------------------------------------------------------------------------------------
    jr = np.flatnonzero(r <= r_max_mm * (1 + 1e-9))
    with np.errstate(invalid="ignore"):
        dz = np.diff(lr[:, :, jr], axis=1) / np.diff(z)[None, :, None]                           # (P, NZ - 1, nr)
    k0 = []
    for j in range(P):
        ks = np.flatnonzero(z[:-1] > z90[j])
        if len(ks) < 1:
            raise ValueError(f"no depth interval beyond z_90 = {z90[j]:.0f} mm at p = {mom[j]:g} GeV")
        k0.append(int(ks[0]))
    beyond = np.zeros(dz.shape, bool)
    for j in range(P):
        beyond[j, k0[j]:, :] = True
    dfin = np.isfinite(dz) & beyond
    dzm = np.where(dfin, dz, -np.inf)
    dmax = float(dzm.max())
    wj, wk, wr = np.unravel_index(int(np.argmax(dzm)), dzm.shape)
    rising = np.argwhere(dfin & (dz > 0))
    cells = [[int(j), int(k), int(q), float(f"{dz[j, k, q]:.9g}")] for j, k, q in rising]
    regions = []
    for j in range(P):
        sel = rising[rising[:, 0] == j]
        reg = {"p_GeV": float(mom[j]), "z90_mm": float(z90[j]), "n_cells": int(len(sel)),
               "share_of_cells_beyond_z90": float(len(sel) / max(int(dfin[j].sum()), 1))}
        if len(sel):
            v = dz[j][sel[:, 1], sel[:, 2]]
            m = int(np.argmax(v))
            col = lr[j, k0[j]:, jr[sel[m, 2]]]
            reg.update({"z_range_mm": [float(z[sel[:, 1].min()]), float(z[sel[:, 1].max() + 1])],
                        "r_range_mm": [float(r[jr][sel[:, 2].min()]), float(r[jr][sel[:, 2].max()])],
                        "max_slope_per_100mm": float(v.max() * 100.0),
                        "max_at": {"z_mm": [float(z[sel[m, 1]]), float(z[sel[m, 1] + 1])], "r_mm": float(r[jr][sel[m, 2]])},
                        "total_rise_at_that_radius_nat": float(np.sum(np.clip(np.diff(col[np.isfinite(col)]), 0, None)))})
        if rising_annotations and j in rising_annotations:
            reg["attribution"] = json.loads(json.dumps(rising_annotations[j]))
        regions.append(reg)
    rec = {
        "schema": TAIL_SCHEMA, "label": str(label), "baseline_hash": str(baseline_hash),
        "reference_kind": str(reference_kind), "note": str(note),
        "density": "slopes of the 3-D density rho(z', r) (probability per volume), not of the ring mass 2 pi r rho",
        "momenta_GeV": mom.tolist(), "x": O.x_of_p(mom).tolist(), "r_min_mm": float(r_min_mm),
        "r_max_mm": float(r_max_mm), "slope_limit": float(slope_limit), "z_nodes_mm": z.tolist(),
        "r_nodes_mm": rn.tolist(), "z_range_mm": [float(z[0]), float(z[-1])], "n_depths": int(NZ),
        "covers_r_max": bool(rn[-1] >= r_max_mm * (1 - 0.05)),
        "sign_change_tolerance": {"depth_per_mm": SIGN_TOL_DEPTH_PER_MM, "radial_loglog": SIGN_TOL_RADIAL,
                                  "counting": "per profile on this record's grid; lobes below the tolerance dropped"},
        "radial": {"loglog_slope": _r9(sl), "baseline_max": gmax,
                   "baseline_max_at": {"p_GeV": float(mom[w[0]]), "z_mm": float(z[w[1]]),
                                       "r_mm": float(np.sqrt(rn[w[2]] * rn[w[2] + 1]))},
                   "baseline_max_per_momentum": slm.reshape(P, -1).max(1).tolist(),
                   "fraction_above_limit": float(np.mean(sl[fin] > slope_limit)) if fin.any() else float("nan"),
                   "baseline_satisfies_limit": satisfies, "baseline_monotone": bool(gmax < 0.0),
                   "n_nonfinite": int((~fin).sum()),
                   "sign_changes": profile_sign_changes(sl, SIGN_TOL_RADIAL).tolist()},
        "depth": {"z90_mm": z90.tolist(), "k0": k0, "r_nodes_mm": r[jr].tolist(), "dz_per_mm": _r9(dz),
                  "baseline_max": dmax,
                  "baseline_max_at": {"p_GeV": float(mom[wj]), "z_mm": [float(z[wk]), float(z[wk + 1])],
                                      "r_mm": float(r[jr][wr])},
                  "baseline_max_per_momentum": dzm.reshape(P, -1).max(1).tolist(),
                  "baseline_falls_everywhere": bool(dmax <= 1e-12), "n_rising_cells": int(len(cells)),
                  "rising_cells": cells, "rising_cells_columns": ["momentum index", "depth interval index k (between "
                                                                  "z_nodes k and k + 1)", "radius index", "slope / mm"],
                  "rising_regions": regions,
                  "sign_changes": [profile_sign_changes(np.where(beyond[j], dz[j], np.nan).T, SIGN_TOL_DEPTH_PER_MM).tolist()
                                   for j in range(P)]},
        "criterion": {"radial": "loglog_slope_le_limit" if satisfies else "monotone_and_no_flattening",
                      "depth": DEPTH_RULE,
                      "decided_from": "the fixed reference baseline alone (delta = 0); not selectable per equation"},
    }
    rec["record_hash"] = _tail_record_hash(rec)
    return rec


def verify_tail_criterion(record: Mapping, baseline_hash: Optional[str] = None, as_reference: bool = True) -> None:
    """Raise ``ValueError`` unless the record is intact (its hash matches its content), its rules follow from its
    own slopes and - if ``baseline_hash`` is given - it is bound to that baseline. With ``as_reference`` (default)
    a table of a jointly refitted baseline (``reference_kind == "refitted_baseline"``) is refused: the bound of the
    depth rule and of the radial alternative must come from the fixed non-SR baseline."""
    if record.get("schema") != TAIL_SCHEMA:
        raise ValueError(f"not a tail-criterion record of schema {TAIL_SCHEMA} (found {record.get('schema')!r})")
    if record.get("record_hash") != _tail_record_hash(record):
        raise ValueError("the tail-criterion record was altered after it was made (hash mismatch)")
    if record["criterion"]["radial"] != ("loglog_slope_le_limit" if record["radial"]["baseline_satisfies_limit"]
                                         else "monotone_and_no_flattening"):
        raise ValueError("the radial rule of the record does not follow from its measured baseline slopes")
    if record["criterion"]["depth"] != DEPTH_RULE:
        raise ValueError(f"the depth rule of the record is not {DEPTH_RULE!r}")
    if as_reference and record.get("reference_kind") == MEASURED_KIND:
        raise ValueError("a slope table of the jointly refitted baseline cannot be the reference: the bound has to "
                         "come from the fixed baseline (Track A finalist, or the refit-only control C1)")
    if baseline_hash is None:
        return
    if record.get("baseline_hash") != baseline_hash:
        raise ValueError(f"the tail criterion belongs to baseline {record.get('baseline_hash')!r}, not to "
                         f"{baseline_hash!r}: decide it for this baseline (checks.decide_tail_criterion)")


def format_tail_criterion(record: Mapping) -> str:
    """A few lines for a log: what was measured on the baseline, where it rises in depth, which rules follow."""
    ra, de = record["radial"], record["depth"]
    at = ra["baseline_max_at"]
    lines = [
        f"tail criterion of [{record['label']}] ({record['reference_kind']}, baseline {record['baseline_hash'][:16]}, "
        f"record {record['record_hash'][:12]}); slopes of the 3-D density rho",
        f"  radial, r in [{record['r_nodes_mm'][0]:.0f}, {record['r_nodes_mm'][-1]:.0f}] mm: largest d ln rho / d ln r"
        f" = {ra['baseline_max']:.3f} at p = {at['p_GeV']:g} GeV, z' = {at['z_mm']:.0f} mm, r = {at['r_mm']:.0f} mm; "
        f"<= {record['slope_limit']:g} everywhere: {ra['baseline_satisfies_limit']}; monotone: "
        f"{ra['baseline_monotone']}  ->  rule \"{record['criterion']['radial']}\"",
        f"  depth beyond z_90: rule \"{record['criterion']['depth']}\" (model slope <= max(0, reference slope)); the "
        f"reference itself rises in {de['n_rising_cells']} cells:"]
    for reg in de["rising_regions"]:
        if not reg["n_cells"]:
            continue
        att = reg.get("attribution", {})
        lines.append(
            f"    p = {reg['p_GeV']:g} GeV: z' {reg['z_range_mm'][0]:.0f}-{reg['z_range_mm'][1]:.0f} mm, r "
            f"{reg['r_range_mm'][0]:.0f}-{reg['r_range_mm'][1]:.0f} mm, {reg['n_cells']} cells "
            f"({100 * reg['share_of_cells_beyond_z90']:.1f} %), up to {reg['max_slope_per_100mm']:+.3f} nat / 100 mm"
            + (f"; removed by leaving out: {att['removed_by']}" if att.get("removed_by") is not None else ""))
    return "\n".join(lines)


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
