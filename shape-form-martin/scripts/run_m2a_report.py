"""Track A (M2A) report: the pre-registered rules of results/m1_audit/AUDIT.md sections 4.0 / 4.1 applied mechanically to
THIS machine's registered arrays, the guard tables, the secondary columns, RESULTS.md and results.json.

    uv run python -B shape-form-martin/scripts/run_m2a_report.py [--smoke]      # also the "report" phase of run_m2a.py

CPU only (reads registry rows of the live stream and their tagged arrays; the guard arrays were computed by the
"guards" phase). Every delta is ``sfm.provenance.paired_delta`` (tag-checked), printed as mean +- paired SE (n), in
units of 1e-4 nat per event. Report-metric numbers (K = 64, all events) and guard numbers (K = 1024, bulk-restricted
stratum, G-region parts, mass monitors) are in separate tables and never in one column or one delta. Nulls and
negative results are printed as measured. Rules (nat per event):
    keep          Delta <= -3e-4 and Delta <= -2 SE against the comparator of the same seed
    replication   seed 1: Delta <= -1.5e-4 and Delta < 0 (same sign); mean of the two seeds' per-event differences
                  <= -3e-4 (its paired SE printed)
    equal NLL     Delta <= +1 SE and Delta < +1e-4, on both seeds (one-seed rungs: "seed 0 only")
    cleaner       (Stage 2 wording, shown per rung) Delta vs THIS machine's formula array <= -3e-4 (paired form of
                  "3e-4 better than the formula / the A0 value"), or >= 10 % fewer constants than A2s (101) at equal NLL
    convergence   |val(last) - val(last - 1000)| < 0.5e-4 from the fit history (this machine)
    guards        G-K / G-bulk persistence = paired margin guard - 0.5 main with its SE, resolved only at |z| >= 2
                  (sfm.stats.guard_persistence); G-region: hit term by hit-centre class (deep z' >= 1221 mm, border
                  = last ring or first / last z-bin, Abar <= 0.3), the normalisation term as its own row; pass if
                  the flagged classes carry <= 50 % of the gain (margin flagged - 0.5 main >= 0, SE, resolved at 2 SE);
                  G-mass: per-event mass monitors, paired
    1-SE          only where the list says so (the Stage-2 finalist), not in the Stage-1 tables
    reruns        (review B1) when the doubled-step rerun pair (<rung>_s<seed>_x2 vs A2s_s<seed>_x2) of a
                  non-converged primary exists, it is the pair of that seed for keep, replication, equal NLL,
                  "cleaner" and kept_and_replicated; the original is printed as a labelled row that no rule uses
    alternatives  (review B3) printed beside the literal readings, labelled "alternative - decides nothing": the
                  two-seed mean also at <= -2 SE; the G-region share relative to the hit-term total
    initialisation (review B2) kl_map of every run against its start / initialisation / fit_to target; net runs
                  with kl_map > 1e-4 nat are flagged "warm start degraded"; target columns of a net warm start whose
                  relative spread lies between the 2^-20 floor and 1e-2 are listed (reporting only)
    templates     (review B4) the limitations Q1 / Q2 of the computed templates are printed beside A3c / A3x

    uv run python -B shape-form-martin/scripts/run_m2a_report.py --without-guards   (review C2) the tables without
                  guard columns, written to RESULTS_without_guards.md / results_without_guards.json (results.json,
                  the file a Stage-2 approval is bound to, is written only by the full report)

Stage 2 (REVIEW.md section 10.5 R5): report_stage2(), called by `run_m2a.py stage2 | a11p | stage2-report`, writes
RESULTS_stage2.md / results_stage2.json - the comparisons of REVIEW 10.4, the finalist rule, "cleaner" (c1), (c2)
against the formula array, the A11p screen and acceptance (REVIEW 11), the A11n initialisation, the Stage-2 guards
with the flagged* waiver line, the conditional controls (PLAN 20 item 1) and the warnings for Gate 1 (item 2).
results.json and RESULTS.md of Stage 1 are not touched and do not change while Stage 2 runs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_m2a_lib as L  # noqa: E402
from run_m2a_lib import log  # noqa: E402

from sfm import models, paths, provenance, registry, stats  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402
from sfm.models.paramnet import ParamNet  # noqa: E402

E4 = 1e4
P_SPLIT_GEV = 2.0
Z_SIDE_MM = 1450.0
A2S_CONSTANTS = 101
KL_DEGRADED = 1e-4                 # nat; review B2: a net warm start above this is "degraded" (fixed before any run)
KL_EXACT = 1e-6                    # nat; "exactly A2s" claims hold up to float rounding
NEAR_CONSTANT_HI = 1e-2            # review B2 / ruling on fit_to: relative spread in (2^-20, 1e-2) degrades fit_to
EXACT_START = ("A7", "A8", "A78", "A7s", "A8c", "A12")   # AUDIT 4.1 / tests: start exactly on A2s
ALT = "alternative - decides nothing"


def _m():
    import run_m2a as M
    return M


def fmt(p: dict | None) -> str:
    if not p:
        return "-"
    return f"{p['mean'] * E4:+.2f} ± {p['se'] * E4:.2f} (n {p['n']})"


def yn(v) -> str:
    return "-" if v is None else ("yes" if v else "no")


def have(name: str) -> bool:
    return L.registered_row(name) is not None


def pdict(a: str, b: str, mask=None) -> dict:
    ta = L.tagged(a)
    tb = L.formula_array() if b == "formula" else L.tagged(b)
    return L.pd(provenance.paired_delta(ta, tb, mask=mask))


# ----------------------------------------------------------------------------------------------------------------------
def secondary_masks() -> dict:
    """Pre-registered secondary strata (AUDIT 4.0): p < 2 GeV / p >= 2 GeV; contained / side-leaking events (axis exit
    depth from checks/c_side.py::exit_depth, ported to sfm.models.dist_factory.exit_depth: side-leaking if the axis
    crosses a side face before z' = 1450 mm)."""
    sets, extras = L.data_val()
    v = sets["val"]
    p = v.ev["p"].numpy().astype(np.float64)
    zb = v.ev["zb"].numpy().astype(np.float64)
    th, ph = (np.asarray(extras["val"][k], dtype=np.float64) for k in ("theta", "phi"))
    z_exit, _, _ = dfac.exit_depth(th, ph, dfac.Z_BACK - zb * np.cos(th))
    side = z_exit < Z_SIDE_MM
    return {"p<2GeV": p < P_SPLIT_GEV, "p>=2GeV": p >= P_SPLIT_GEV, "contained": ~side, "side-leaking": side}


def comparison(run: str, comp: str, masks: dict) -> dict:
    M = _m()
    p = provenance.paired_delta(L.tagged(run), L.tagged(comp))
    row = L.registered_row(run)
    out = {"run": run, "vs": comp, "paired": L.pd(p), "keep": M.keep_rule(p), "equal_one": M.equal_rule_one(p),
           "k": row["k"], "vs_formula": row["vs_formula"], "secondary": {}}
    for k, m in masks.items():
        out["secondary"][k] = pdict(run, comp, mask=m)
    return out


def replication(c0: dict, c1: dict | None, run0: str, comp0: str, run1: str | None, comp1: str | None) -> dict | None:
    """AUDIT 4.0 replication: seed 1 Delta <= half the threshold (-1.5e-4) with the same sign, and the mean of the two
    seeds meets the threshold (-3e-4). The two-seed mean is the mean of the per-event differences (same events).
    Not evaluated (None) when a seed has no comparison (diverged / blocked)."""
    if c1 is None or c0 is None or c0.get("paired") is None or c1.get("paired") is None:
        return None
    arrs = [L.tagged(run0), L.tagged(run1), L.tagged(comp0), L.tagged(comp1)]
    provenance.check_pairable(*arrs)                  # all four on the same points and arithmetic
    t = arrs[0]
    mean_run = provenance.TaggedArray.make(0.5 * (arrs[0].values.astype(np.float64) + arrs[1].values), t.points,
                                           L.live_machine(), label=f"mean({run0},{run1})")
    mean_comp = provenance.TaggedArray.make(0.5 * (arrs[2].values.astype(np.float64) + arrs[3].values), t.points,
                                            L.live_machine(), label=f"mean({comp0},{comp1})")
    both = provenance.paired_delta(mean_run, mean_comp)
    m1 = c1["paired"]["mean"]
    ok = bool(m1 <= -0.5 * _m().KEEP_GAIN and (m1 < 0) == (c0["paired"]["mean"] < 0) and both.mean <= -_m().KEEP_GAIN)
    alt = bool(both.mean <= -2.0 * both.se)
    return {"seed1_half_threshold": m1 <= -0.5 * _m().KEEP_GAIN, "same_sign": (m1 < 0) == (c0["paired"]["mean"] < 0),
            "two_seed_mean": L.pd(both), "replicated": ok, "pairs": [[run0, comp0], [run1, comp1]],
            "alternative_two_seed_mean_2se": alt, "replicated_alternative_2se": bool(ok and alt)}


def convergence(name: str) -> dict | None:
    if L.done_marker(L.out_dir() / name / "done.json") is None:
        return None
    ok, mv = _m().converged(name)
    return {"converged": ok, "moved": mv}


# ----------------------------------------------------------------------------------------------------------------------
def guard_block(run: str, comp: str) -> dict | None:
    """Guard tables of one comparison: G-K, G-bulk (+ short stratum row), G-mass, G-region."""
    r_run, r_comp = L.registered_row(run, need_guards=True), L.registered_row(comp, need_guards=True)
    if r_run is None or r_comp is None:
        return None
    main = provenance.paired_difference(L.tagged(run), L.tagged(comp))
    out = {"run": run, "vs": comp}
    # G-K
    k_run, k_comp = L.tagged(run, "k1024"), L.tagged(comp, "k1024")
    gk = provenance.paired_difference(k_run, k_comp)
    dk = stats.guard_persistence(main, gk)
    out["G-K"] = {"delta_k1024": L.pd(provenance.paired_delta(k_run, k_comp)),
                  "persistence": {"passed": dk.passed, "checks": dk.checks, "numbers": dk.numbers, "flags": dk.flags}}
    # G-bulk on the stratum zb >= 1221 mm (amendment A4); bulk arrays are per stratum event
    b_run, b_comp = L.tagged(run, "bulk_zb1221"), L.tagged(comp, "bulk_zb1221")
    sets, _ = L.data_val()
    zb = sets["val"].ev["zb"].numpy().astype(np.float64)
    in_s = zb >= registry.ZB_STRATUM_MM
    ok = np.isfinite(b_run.values) & np.isfinite(b_comp.values)
    gb = provenance.paired_difference(b_run, b_comp, mask=ok)
    main_s = main[in_s][ok]
    db = stats.guard_persistence(main_s, gb)
    out["G-bulk"] = {"delta_bulk": L.pd(provenance.paired_delta(b_run, b_comp, mask=ok)),
                     "n_stratum": int(in_s.sum()), "n_excluded_nonfinite": int((~ok).sum()),
                     "persistence": {"passed": db.passed, "checks": db.checks, "numbers": db.numbers,
                                     "flags": db.flags}}
    # report metric on the strata (own rows; report-metric numbers)
    out["report_metric_strata"] = {"zb>=1221 (bulk events)": pdict(run, comp, mask=in_s),
                                   "zb<1221 (short events)": pdict(run, comp, mask=~in_s) if (~in_s).sum() >= 2
                                   else None}
    # G-mass: per-event mass monitors (npz of the guard rows; file SHA-256 checked), paired per monitor
    mons = {}
    f_run = L.paths.OUT_RESULTS / r_run["guards"]["arrays"]["mass_monitors"]["path"]
    f_comp = L.paths.OUT_RESULTS / r_comp["guards"]["arrays"]["mass_monitors"]["path"]
    for f, r in ((f_run, r_run), (f_comp, r_comp)):
        if provenance.sha256_file(f) != r["guards"]["arrays"]["mass_monitors"]["file_sha256"]:
            raise RuntimeError(f"{paths.portable(f)}: SHA-256 differs from its registry row")
    pts = L.live()["points"]
    with np.load(f_run) as a, np.load(f_comp) as b:
        for key in sorted(set(a.files) & set(b.files)):
            tp = provenance.points_tag(grid=pts["grid"], K=pts["K"], split_fingerprint=pts["split_fingerprint"],
                                       seed=pts["seed"], batch=pts["batch"], subset=f"mass_monitor:{key}")
            ta = provenance.TaggedArray.make(a[key].astype(np.float64), tp, L.live_machine(), label=f"{run}:{key}")
            tb = provenance.TaggedArray.make(b[key].astype(np.float64), tp, L.live_machine(), label=f"{comp}:{key}")
            mons[key] = {"run_mean": float(ta.values.mean()), "comp_mean": float(tb.values.mean()),
                         "delta": L.pd(provenance.paired_delta(ta, tb))}
    out["G-mass"] = mons
    # G-region
    out["G-region"] = g_region_block(run, comp, main)
    return out


def _gr_tagged(name: str, cls: str) -> provenance.TaggedArray:
    sdir = L.scores_dir(name)
    mk = L.read_json(sdir / "g_region.json")
    if not mk:
        raise FileNotFoundError(f"{L.stage()}/{name}: no G-region arrays (guards phase)")
    ta = provenance.load_tagged(sdir / f"g_region_{cls}.npy", expect_machine_id=L.live_machine().id)
    if ta.sha256 != mk["arrays"][cls]["sha256"]:
        raise RuntimeError(f"{name}: G-region array {cls} differs from its marker")
    return ta


def g_region_block(run: str, comp: str, main: np.ndarray) -> dict | None:
    try:
        rows = {}
        diffs = {}
        for cls in ("deep", "border_ring", "border_zbin", "low_abar", "flagged", "inner", "normalisation"):
            a, b = _gr_tagged(run, cls), _gr_tagged(comp, cls)
            rows[cls] = L.pd(provenance.paired_delta(a, b))
            diffs[cls] = provenance.paired_difference(a, b)
    except FileNotFoundError:
        return None
    gain = float(main.mean())
    margin = stats.paired(diffs["flagged"] - 0.5 * main, np.zeros_like(main))
    hit_total = diffs["flagged"] + diffs["inner"]
    return {"hit_term_by_class": {k: v for k, v in rows.items() if k != "normalisation"},
            "normalisation_term": rows["normalisation"],
            "hit_term_total": L.pd(stats.paired(hit_total, np.zeros_like(hit_total))),
            "decomposition_residual_max_abs": float(np.abs(hit_total + diffs["normalisation"] - main).max()),
            "flagged_share_of_gain": (rows["flagged"]["mean"] / gain) if gain != 0 else None,
            "margin_flagged_minus_half_main": L.pd(margin),
            "alternative_flagged_share_of_hit_term": (rows["flagged"]["mean"] / float(hit_total.mean())
                                                      if float(hit_total.mean()) != 0 else None),
            "alternative_margin_flagged_minus_half_hit_term": L.pd(stats.paired(diffs["flagged"] - 0.5 * hit_total,
                                                                                np.zeros_like(main))),
            "alternative_passed_hit_term": bool(gain < 0 and float(hit_total.mean()) < 0          # review R2
                                                and float((diffs["flagged"] - 0.5 * hit_total).mean()) >= 0.0),
            "passed": bool(gain < 0 and margin.mean >= 0.0), "resolved": abs(margin.z) >= 2.0,
            "rule": "<= 50 % of the gain from deep, border or Abar <= 0.3 cells (hit term, by hit centre); "
                    "normalisation term reported separately"}


# ----------------------------------------------------------------------------------------------------------------------
def warm_start_targets(done: dict) -> dict:
    """Relative spreads sd / max(1, |mean|) of the target columns of a net warm start (uniform weights on
    default_lnp(48), as make_paramnet calls fit_to), recomputed on the CPU from the fitted poly run (net_from) or,
    without net_from, from the model's own legacy provider. Columns at or below the 2^-20 floor are constant (fit_to
    sets them exactly); columns in (floor, 1e-2) are "nearly constant": the review measured a degraded fit there."""
    from sfm.families.components import default_lnp
    kw = dict(done.get("build_kwargs") or done["spec"]["kwargs"])
    ctx = L.build_context("cpu", None, 0, skip_init=True)
    if kw.get("net_from"):
        tgt = models.load_run(kw["net_from"], ctx)
    else:
        kw2 = {k: v for k, v in kw.items() if k not in ("provider", "fit_to_steps", "fit_to_lbfgs", "net_from")}
        tgt = models.build(done["spec"]["factory"], ctx, **kw2)
    base = tgt.base if isinstance(tgt, dfac.LastRing) else tgt
    names_net, _ = dfac.net_name_split(base)
    poly = base.provider
    with torch.no_grad():
        t = poly(default_lnp(48, device="cpu")).double()
    rel, std = {}, {}
    for n in names_net:
        col = t[:, poly.names.index(n)]
        std[n] = float(col.std(unbiased=False))
        rel[n] = std[n] / max(1.0, abs(float(col.mean())))
    floor = ParamNet.SCALE_FLOOR_REL
    varying = {n: r for n, r in rel.items() if r > floor}
    return {"target": kw.get("net_from") or "the model's own (legacy) provider", "std": std,
            "below_scale_floor": sorted(n for n in varying if std[n] < ParamNet.SCALE_FLOOR_ABS),
            "constant": sorted(n for n, r in rel.items() if r <= floor),
            "nearly_constant": {n: r for n, r in varying.items() if r < NEAR_CONSTANT_HI},
            "smallest_varying": (min(varying, key=varying.get), min(varying.values())) if varying else None}


def initialisation_table(stage2: bool = False) -> list:
    """Review B2: kl_map of every finished run against what it was started / initialised / fitted from, the
    "exactly A2s" claims, and for net runs the warm-start report with the degraded-start flag. ``stage2``: the
    Stage-2 runs instead (the Stage-2 report; REVIEW 10.5 R5)."""
    out = []
    chk = (L.decisions().get("warmstart_check") or {}).get("runs") or {}
    sup = L.superseded()
    for f in sorted(L.out_dir().glob("*/done.json")):
        d = L.done_marker(f)
        init = L.read_json(f.parent / "init.json", {}) or {}
        info = init.get("init_info") or {}
        spec = d.get("spec") or {}
        if (spec.get("cls") == "stage2") != stage2:            # Stage 2 has its own report (keeps R6 stable)
            continue
        row = {"run": f.parent.name, "rung": spec.get("rung"), "provider": spec.get("provider"),
               "cls": spec.get("cls"), "kl_map": d.get("kl_map", info.get("kl_map")), "kl_ref": info.get("kl_ref"),
               "sources_verified": sorted((d.get("sources_verified") or {}).keys())}
        if ((spec.get("rung") in EXACT_START and spec.get("cls") in ("primary", "exploratory", "rerun"))
                or spec.get("rung") == "A2c") and spec.get("provider") == "poly":                # review C8-5
            row["claim"] = "exactly A2s"
            row["claim_holds"] = row["kl_map"] is not None and abs(row["kl_map"]) <= KL_EXACT
        if spec.get("provider") == "net":
            ft = info.get("fit_to") or {}
            name = f.parent.name
            row.update({"fit_to_rms_max": ft.get("rms_max"), "fit_to_rms_max_name": ft.get("rms_max_name"),
                        "fit_to_constant_columns": ft.get("constant_columns"),
                        "fit_to_lbfgs_stopped": ft.get("lbfgs_stopped"),
                        "fit_to_version": ft.get("fit_to_version") or 2,
                        "fit_to_below_scale_floor": ft.get("below_scale_floor"),
                        "degraded": row["kl_map"] is not None and row["kl_map"] > KL_DEGRADED,
                        "superseded_by": sup.get(name)})
            if init.get("step0_parity"):
                row["step0_parity"] = {**init["step0_parity"], "origin": "logged at initialisation"}
            elif (chk.get(name) or {}).get("step0_parity"):
                row["step0_parity"] = {**chk[name]["step0_parity"],
                                       "origin": "rebuilt old warm start (warmstart-check; bit for bit: "
                                                 f"{chk[name].get('bit_for_bit')})"}
            try:
                row["targets"] = warm_start_targets(d)
            except Exception as err:                   # reporting only: never stops the report
                row["targets"] = {"error": f"{type(err).__name__}: {err}"}
            std = (ft.get("column_std") or (row["targets"].get("std") if isinstance(row["targets"], dict) else None)
                   or {})
            nm, rm = row["fit_to_rms_max_name"], row["fit_to_rms_max"]
            row["fit_to_rms_max_rel"] = (rm / std[nm]) if (rm is not None and std.get(nm)) else None   # review R5
        out.append(row)
    return out


def template_limitations() -> dict | None:
    """Review B4 (rulings Q1, Q2): the measured limitations of the computed templates, from this machine's files."""
    tdir = dfac.template_dir()
    meta = L.read_json(tdir / "templates.json")
    f = tdir / "templates_all.npz"
    if not meta or not f.exists():
        return None
    with np.load(f) as t:
        bc, den = t["back_corrected"], t["below_zb"]
    zc = L.common.Z_EDGES.cpu().numpy().astype(np.float64)
    zc = 0.5 * (zc[:-1] + zc[1:])
    above = bc > 1.01
    iz = np.argwhere(above)[:, 0]
    return {"fb_vs_stored_abar": meta.get("fb_vs_stored_abar"), "cells_above_1": int((bc > 1.0).sum()),
            "cells_above_1.01": int(above.sum()), "cells_above_1.1": int((bc > 1.1).sum()),
            "max_back_corrected": float(bc.max()),
            "lowest_zbin_centre_above_1.01_mm": float(zc[iz.min()]) if above.any() else None,
            "smallest_denominator_where_above_1.01": float(den[above].min()) if above.any() else None,
            "templates_json_sha256": provenance.sha256_file(tdir / "templates.json")}


TEMPLATE_NOTE_Q1 = ("Q1 (A3c, A3x): numerator by the 2 x 2-point rule of `c_side.py`; in the z-bin of the front face it "
                    "overestimates the innermost rings (0.50 against 0.35); effect on the validation NLL at the legacy "
                    "coefficients -0.1 to -0.2 (review: stored template with the 2-point bin 14 -0.10 ± 0.01; A3c as "
                    "coded - A3c with the stored numerator -0.16 ± 0.02; A3x as coded - A3x with 4 sub-points "
                    "-0.14 ± 0.02; laptop diagnostic, not numbers of record).")
TEMPLATE_NOTE_Q2 = ("Q2 (A3c): the back_corrected template exceeds 1 by more than 1 % in {n101} cells at z' >= {z} mm "
                    "(maximum {mx:.2f}; smallest denominator there {den:.3f}): it adds density behind the flat cut, where "
                    "the metric's normaliser does not count mass. A gain of A3c is to be read with the G-region class "
                    "`deep` and the `beyond_zb` mass monitor.")


# ----------------------------------------------------------------------------------------------------------------------
def run_status(name: str) -> str:
    return _m()._status_text(name)


def _diagnosed(name: str) -> bool:
    return L.diverged_marker(name) is not None and bool((L.diagnosis(name) or {}).get("valid"))


def _lr_net_of_record():
    try:
        return L.frozen_lr("net")
    except RuntimeError:
        return None


def entry(run: str, comp: str, masks: dict) -> dict | None:
    """The comparison of run vs comp if both are registered; a status record if the run (or its comparator) diverged
    or is blocked (user decision of 25 Nov 2026: never silently omitted); None if the run does not exist at all."""
    if have(run) and have(comp):
        return comparison(run, comp, masks)
    st_r = run_status(run)
    if st_r == "missing" and not have(run):
        return None
    st = st_r if not have(run) else f"comparator {comp}: {run_status(comp)}"
    return {"run": run, "vs": comp, "status": st, "paired": None, "keep": None, "equal_one": None, "k": None,
            "vs_formula": None, "secondary": {}}


def _kept_status(v: dict) -> str:
    """'yes', 'no', 'no (failed, diagnosed: <run>)' or 'not evaluable (<run>: <status>)' for the kept line (review
    B8-3): kept = keep on seed 0 and replication by seed 1. Seed 0 without a comparison -> not evaluable (or 'no' if
    its divergence was diagnosed as a failure); seed 0 kept but seed 1 without a comparison -> not evaluable."""
    c0, c1 = v["seeds"].get(0), v["seeds"].get(1)
    for c in (c0, c1):
        if c is None or not c.get("status"):
            continue
        if c is c1 and not (c0 and c0.get("keep")):
            continue                                   # seed 0 evaluated and not kept: seed 1 cannot change it
        if c["status"] == "failed (diagnosed)":
            return f"no (failed, diagnosed: {c['run']})"
        return f"not evaluable ({c['run']}: {c['status']})"
    if c0 is None:
        return "not evaluable (seed 0 missing)"
    rep_ = v.get("replication")
    return "yes" if (c0.get("keep") and rep_ and rep_.get("replicated")) else "no"


def primary_pairs(r: str, masks: dict) -> tuple:
    """Review B1: ({seed: comparison used by the rules}, {seed: original not used}). For each seed the doubled-step
    rerun pair (<r>_s<seed>_x2 vs A2s_s<seed>_x2) replaces the non-converged original when it exists. Diverged or
    blocked runs appear with their status (PLAN 17.11)."""
    c, orig = {}, {}
    for s in (0, 1):
        o = entry(f"{r}_s{s}", f"A2s_s{s}", masks)
        x = entry(f"{r}_s{s}_x2", f"A2s_s{s}_x2", masks)
        if x is not None:                         # the converged measurement of this seed is the rerun pair
            x["pair"] = "rerun"
            c[s] = x
            if o is not None:
                o["pair"] = "original (not converged; not used by the rules)"
                orig[s] = o
        elif o is not None:
            o["pair"] = "original"
            c[s] = o
    return c, orig


STAGE1_EXCLUDED_KEYS = ("conditional_controls",)   # PLAN 20 item 1: printed by the Stage-2 report (REVIEW 10.5 R5)


def stage1_decisions(raw: dict) -> dict:
    """decisions.json as results.json carries it: without the Stage-2 records (keys 'stage2*', the conditional-control
    record of PLAN 20, and in every per-run map - diverged, blocked, warnings, step0_parity - the entries of Stage-2
    runs). results.json and RESULTS.md of Stage 1 stay byte-identical (apart from the commit) while the additions of
    26 Nov 2026 and Stage 2 run (REVIEW 10.5 R5, R8, R9)."""
    out = {}
    for k, v in raw.items():
        if k.startswith("stage2") or k in STAGE1_EXCLUDED_KEYS:
            continue
        if isinstance(v, dict) and any(L.is_stage2_run(n) for n in v):
            v = {n: x for n, x in v.items() if not L.is_stage2_run(n)}
        out[k] = v
    return out


def collect(with_guards: bool = True) -> dict:
    M = _m()
    dec = stage1_decisions(L.read_json(L.decisions_file(), {}) or {})
    from sfm import machine as _machine
    gs = _machine.git_state()
    E = L.effective
    res = {"stage": L.stage(), "smoke": L.CTX.smoke, "machine_id": L.MACHINE_ID, "stream_id": L.live_stream_id(),
           "git": {"commit": gs.get("commit"), "clean": gs.get("clean")},              # review R8
           "units": "1e-4 nat per event (deltas); paired SE; n events", "decisions": dec,
           "plr": L.read_json(L.plr_file()), "refset": paths.portable(L.refset().dir),
           "refset_json_sha256": L.refset().manifest_sha256}
    masks = secondary_masks()
    res["secondary_strata_n"] = {k: int(v.sum()) for k, v in masks.items()}
    # Stage 0
    s0, s0_status, s0_superseded = {}, {}, {}
    for a, b in (("A0_s0", "formula"), ("A0_s1", "formula"), ("A1_s0", "A0_s0"), ("A1_s1", "A0_s1"),
                 ("A2e_s0", "A0_s0"), ("A2e_s1", "A0_s1"), ("A2s_s0", "A2e_s0"), ("A2s_s1", "A2e_s1"),
                 ("A2n_s0", "A2s_s0"), ("A2n_s1", "A2s_s1"), ("A2n_s2", "A2s_s2"),
                 ("A2s_s1", "A2s_s0"), ("A2s_s2", "A2s_s0"), ("A2s_s2", "A2s_s1")):
        ea, eb = E(a), (b if b == "formula" else E(b))
        if have(ea) and (eb == "formula" or have(eb)):
            s0[f"{ea} - {eb}"] = pdict(ea, eb)
            if ea != a and have(a) and (b == "formula" or have(b)):
                s0_superseded[f"{a} - {b}"] = pdict(a, b)
        elif run_status(ea) not in ("missing", "done"):
            s0_status[f"{ea} - {eb}"] = run_status(ea)
    means = [float(np.mean(L.tagged(f"A2s_s{s}").values, dtype=np.float64)) for s in (0, 1, 2) if have(f"A2s_s{s}")]
    res["stage0"] = {"deltas": s0, "status": s0_status, "superseded_warm_start": s0_superseded,
                     "A2s_seed_sd_of_means": float(np.std(means, ddof=1)) if len(means) > 1 else None}
    # step-0 Delta of A9
    res["A9_step0"] = {f"A9_s{s}_init - A2s_s{s}": pdict(f"A9_s{s}_init", f"A2s_s{s}")
                       for s in (0, 1) if have(f"A9_s{s}_init") and have(f"A2s_s{s}")}
    # Stage 1 primaries (poly), the net column, exploratory
    prim, net, expl = {}, {}, {}
    for r in M.PRIMARIES:
        c, orig = primary_pairs(r, masks)
        if not c:
            continue
        rep = (replication(c[0], c.get(1), c[0]["run"], c[0]["vs"], (c.get(1) or {}).get("run"),
                           (c.get(1) or {}).get("vs")) if 0 in c else None)
        prim[r] = {"seeds": c, "originals_not_used": orig, "replication": rep,
                   "equal_both": (all(v["equal_one"] for v in c.values()) if len(c) == 2 else None),
                   "convergence": {s: convergence(c[s]["run"]) for s in c},
                   "convergence_original": {s: convergence(f"{r}_s{s}") for s in (0, 1) if have(f"{r}_s{s}")}}
        nr, nc = E(f"{r}n_s0"), E("A2n_s0")
        x = entry(nr, nc, masks)
        if x is not None:
            net[r + "n"] = x
            x["convergence"] = convergence(nr)
            if nr != f"{r}n_s0" or nc != "A2n_s0":
                x["supersedes"] = entry(f"{r}n_s0", "A2n_s0", masks)
            nif = L.rerun_not_in_force().get(f"{r}n_s0")
            if nif:                                    # review C8-4: the original stays in force
                x["rerun_not_in_force"] = {"run": nif[0], "status": run_status(nif[0]),
                                           "diagnosis": L.diagnosis(nif[0])}
    for r in M.EXPLORATORY:
        c = {}
        for s in (0, 1):
            x = entry(f"{r}_s{s}", f"A2s_s{s}", masks)
            if x is not None:
                c[s] = x
        if not c:
            continue
        for x in c.values():
            x["pair"] = "original"
        rep = replication(c[0], c.get(1), f"{r}_s0", "A2s_s0", f"{r}_s1", "A2s_s1") if 0 in c else None
        expl[r] = {"seeds": c, "replication": rep,
                   "equal_both": (all(v["equal_one"] for v in c.values()) if len(c) == 2 else None),
                   "convergence": {s: convergence(f"{r}_s{s}") for s in c}}
    res["primaries"], res["net"], res["exploratory"] = prim, net, expl
    # cleaner columns and kept-and-replicated (input of Stage 2; no decision is taken here)
    kr, kr_alt, kst = {}, {}, {}
    for grp in (prim, expl):
        for r, v in grp.items():
            c0 = v["seeds"].get(0)
            kr[r] = bool(c0 and c0["keep"] and v["replication"] and v["replication"]["replicated"])
            kr_alt[r] = bool(c0 and c0["keep"] and v["replication"] and v["replication"]["replicated_alternative_2se"])
            # review B8-3: a seed without a comparison (diverged / blocked) makes the rule not evaluable, never "no"
            kst[r] = _kept_status(v)
            if any(c.get("status") for c in v["seeds"].values()):
                v["equal_both"] = None
                v["equal_both_status"] = "not evaluable (" + "; ".join(
                    f"{c['run']}: {c['status']}" for c in v["seeds"].values() if c.get("status")) + ")"
            for s, c in v["seeds"].items():
                if c.get("paired") is None:
                    c["cleaner_by_gain"] = c["cleaner_by_constants"] = None
                    continue
                c["cleaner_by_gain"] = c["vs_formula"]["mean"] <= -M.KEEP_GAIN
                c["cleaner_by_constants"] = bool(c["k"] <= 0.9 * A2S_CONSTANTS and v["equal_both"])
    # A2c control column (user decision of 25 Nov 2026; decides nothing)
    a2c = {"note": M.A2C_NOTE, "groups": M.A2C_GROUPS, "runs": {}, "control_column": {}}
    for s in (0, 1):
        x = entry(f"A2c_s{s}", f"A2s_s{s}", masks)
        a2c["runs"][s] = x if x is not None else {"run": f"A2c_s{s}", "status": "not run yet"}
    a2c["complete"] = all(have(f"A2c_s{s}") or _diagnosed(f"A2c_s{s}") for s in (0, 1))      # review B8-2
    a2c["diagnosed_divergence"] = {f"A2c_s{s}": L.diagnosis(f"A2c_s{s}") for s in (0, 1)
                                   if not have(f"A2c_s{s}") and _diagnosed(f"A2c_s{s}")}
    for grp in (prim, expl):
        for r, v in grp.items():
            for s, c in v["seeds"].items():
                if r in M.A2C_MATCHED and c.get("pair") in ("original", None) and have(c["run"]) \
                        and have(f"A2c_s{s}"):
                    a2c["control_column"][c["run"]] = pdict(c["run"], f"A2c_s{s}")
                elif r in M.A2C_UNMATCHED:
                    a2c["control_column"][c["run"]] = {"not_matched": f"{M.A2C_UNMATCHED[r]} steps, init_from_target"}
                elif c.get("pair") == "rerun":
                    a2c["control_column"][c["run"]] = {"not_matched": "doubled-step rerun"}
    res["a2c"] = a2c
    res["diverged"] = {n: {**v, "status_now": run_status(n), "diagnosis": L.diagnosis(n)}
                       for n, v in (dec.get("diverged") or {}).items()}
    res["blocked"] = dict(dec.get("blocked") or {})
    res["superseded_warm_start"] = L.superseded()
    res["warm_start_reruns_not_in_force"] = {o: {"rerun": n, "status": st, "status_now": run_status(n),
                                                 "diagnosis": L.diagnosis(n)}
                                             for o, (n, st) in L.rerun_not_in_force().items()}
    res["lr_net_of_record"] = _lr_net_of_record()
    res["kept_and_replicated"] = kr
    res["kept_and_replicated_alternative_2se"] = kr_alt
    res["kept_status"] = kst
    res["primary_rungs"], res["exploratory_rungs"] = list(M.PRIMARIES), list(M.EXPLORATORY)
    # guards (C2: may be left out; a pair whose guard data is missing is listed, never dropped silently)
    g, missing = {}, []
    if with_guards:
        for run, comp in (dec.get("guard_targets") or {}).get("pairs", []):
            blk = guard_block(run, comp)
            if blk is not None:
                g[f"{run} vs {comp}"] = blk
            else:
                missing.append(f"{run} vs {comp}")
    res["with_guards"] = bool(with_guards)
    res["guards"] = g
    res["guards_missing"] = missing
    res["initialisation"] = initialisation_table()
    res["templates"] = template_limitations()
    fisher = L.read_json(L.out_dir() / "fisher" / "A6b_s0_sh_rk.json")
    res["A6b_fisher_k"] = fisher
    # convergence of every finished run
    conv = {}
    for f in sorted(L.out_dir().glob("*/done.json")):
        if ((L.done_marker(f) or {}).get("spec") or {}).get("cls") == "stage2":
            continue
        conv[f.parent.name] = convergence(f.parent.name)
    res["convergence_all"] = conv
    return res


# ----------------------------------------------------------------------------------------------------------------------
def _cell(c) -> str:
    """A comparison cell: mean ± SE (n), or the status of a diverged / blocked run (never omitted)."""
    if c is None:
        return "-"
    if c.get("status"):
        return f"**{c['status']}**"
    return fmt(c.get("paired"))


def _k(*cs) -> str:
    for c in cs:
        if c and c.get("k") is not None:
            return str(c["k"])
    return "-"


def _yn(c, key):
    return yn(None if (c is None or c.get("status")) else c.get(key))


def markdown(res: dict) -> str:
    out = []
    w = out.append
    title = "Track A (M2A) - SMOKE RUN (shortened fits; not numbers of record)" if res["smoke"] else \
        "Track A (M2A) - results on the machine of record"
    w(f"# {title}\n")
    w(f"Stage `{res['stage']}`, machine `{res['machine_id']}`, stream `{res['stream_id']}`, report code at commit "
      f"`{(res.get('git') or {}).get('commit')}` (clean clone: {(res.get('git') or {}).get('clean')}). Reference set "
      f"`{res['refset']}` (refset.json SHA-256 `{res['refset_json_sha256'][:16]}...`). Validation only. Every delta: "
      "mean ± paired SE (n events), unit 1e-4 nat per event, through `sfm.provenance.paired_delta`. Report-metric "
      "tables (K = 64) and guard tables are separate. Nulls and negative results as measured.\n")
    a2c = res.get("a2c") or {}
    if not a2c.get("complete"):
        w("**NOT FINAL: the A2c control (user decision of 25 Nov 2026) is not finished - `run_m2a.py a2c` (a diverged "
          "seed needs its independent diagnosis instead).**\n")
    if res.get("diverged") or res.get("blocked"):
        w(f"**Diverged or blocked runs: {len(res.get('diverged') or {})} diverged, {len(res.get('blocked') or {})} "
          "blocked - see the section 'Diverged and blocked runs'. None is omitted from the tables.**\n")
    dec = res["decisions"]
    par = dec.get("parity", {})
    w("## Harness\n")
    if par:
        pp = par.get("paired") or {}
        w(f"- A0 at step 0 vs this machine's formula array: {pp.get('mean', float('nan')):+.3e} ± "
          f"{pp.get('se', float('nan')):.1e} nat (n {pp.get('n')}), max |event| "
          f"{par.get('max_abs_event', float('nan')):.2e} -> **{'PASS' if par.get('passed') else 'FAIL'}** "
          "(criterion: 1e-4 per event, 1e-6 on the mean)")
    dr = dec.get("A0_drift")
    if dr:
        w(f"- A0 drift rule (stop if |A0 - formula| > 2): " + ", ".join(f"{k} {fmt(v)}" for k, v in dr["runs"].items())
          + f" -> {'STOP' if dr['stop'] else 'continue'}"
          + (f"; not evaluable: {dr['not_evaluable']}" if dr.get("not_evaluable") else ""))
    if res.get("plr"):
        p = res["plr"]
        w(f"- P-lr: lr_poly = {p.get('lr_poly')}, lr_net = {p.get('lr_net')}; " + "; ".join(
            f"{grp} lr {lr}: " + (fmt(v['vs_lowest']) + " vs lowest" if v.get("vs_lowest") else
                                  f"{v.get('status')} - loses (P-lr rule)")
            for grp in ("poly", "net") for lr, v in (p.get(grp) or {}).items()))
    w(f"- Learning rate of record for network runs: {res.get('lr_net_of_record')} (plr.json, or the rate decided again "
      "by warmstart-rerun when re-run pilots changed it; review B8-1)")
    ws = dec.get("warmstart_rerun") or {}
    if (ws.get("lr_net_after_rerun") or {}):
        x = ws["lr_net_after_rerun"]
        w(f"- lr_net after the warm-start re-run of the network pilots: {x.get('lr_net_after')} (before: "
          f"{x.get('lr_net_before')})")
        if x.get("escalated"):                                                       # PLAN 20 item 3
            w(f"- **lr_net ESCALATED to the user (PLAN 20 item 3):** {x.get('reason')}; the re-run pilots "
              f"{x.get('not_used')} are not used and the rate of record stays {x.get('lr_net_before')} until the "
              "user decides (Stage 2 is refused until then)")
    w(f"- Dropped runs: {dec.get('dropped', [])} ({dec.get('budget_rule', 'budget rule: not applied')})\n")
    w("## Stage 0 (report metric)\n")
    w("| comparison | Δ |\n|---|---|")
    for k, v in res["stage0"]["deltas"].items():
        w(f"| {k} | {fmt(v)} |")
    for k, v in (res["stage0"].get("status") or {}).items():
        w(f"| {k} | **{v}** |")
    for k, v in (res["stage0"].get("superseded_warm_start") or {}).items():
        w(f"| {k} (superseded: warm start; not used) | {fmt(v)} |")
    for o, v in (res.get("warm_start_reruns_not_in_force") or {}).items():
        w(f"| {v['rerun']} (warm-start re-run of {o}: **{v['status_now']}** - NOT in force; {o} stays in force) | - |")
    sd = res["stage0"]["A2s_seed_sd_of_means"]
    w(f"\nA2s seed scatter (sd of the validation means of the seeds): {sd * E4:.2f}e-4\n" if sd is not None else "")
    if res["A9_step0"]:
        w("A9 at step 0 (logit_dh = -5): " + "; ".join(f"{k} {fmt(v)}" for k, v in res["A9_step0"].items()) + "\n")
    # A2c
    w(f"## A2c - {a2c.get('note', '')}\n")
    w("A2s continued unchanged from the fitted A2s of the same seed, with the schedule of the rungs that start from "
      "it (T_A, 4000 steps, the frozen lr_poly, the same seed). No pre-registered rule consumes it: keep, replication "
      "and equal NLL stay against A2s as registered. The column 'control: Δ vs A2c' below decides nothing.\n")
    for s_, x in (a2c.get("runs") or {}).items():
        w(f"- A2c_s{s_} - A2s_s{s_}: {_cell(x) if x.get('vs') else x.get('status')}")
    for nme, dg in (a2c.get("diagnosed_divergence") or {}).items():
        w(f"- {nme} diverged; diagnosis on file ({dg.get('verdict')}): no control column for that seed. The report "
          "is final without it (review B8-2).")
    w("- Matched step-budget groups: " + "; ".join(f"**{k}**: {v}" for k, v in (a2c.get("groups") or {}).items())
      + "\n")
    cc = a2c.get("control_column") or {}

    def ctrl(c):
        if c is None or c.get("status"):
            return "-"
        x = cc.get(c["run"])
        if x is None:
            return "A2c missing" if not a2c.get("complete") else "-"
        return f"not matched ({x['not_matched']})" if "not_matched" in x else fmt(x)

    w("## Stage 1 - primaries, poly (pre-registered comparisons, vs A2s of the same seed; report metric)\n")
    w("Multiplicity: these six rungs are the pre-registered primary comparisons of Track A; every other row of this "
      "report is exploratory and is not a test.\n")
    w("| rung | k | pairs used by the rules (s0 / s1) | Δ seed 0 | keep s0 | Δ seed 1 | keep s1 | replicated "
      "(two-seed mean) | replicated under the alternative reading (literal conditions AND two-seed mean <= -2 SE; "
      f"{ALT}) | equal NLL (both) | Δ vs formula s0 | converged s0/s1 | control: Δ vs A2c s0 / s1 (decides nothing) |\n"
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r, v in res["primaries"].items():
        c0, c1 = v["seeds"].get(0), v["seeds"].get(1)
        rep = v["replication"]
        cv = v["convergence"]
        used = " / ".join((v["seeds"][s_].get("pair", "original") if s_ in v["seeds"] else "-") for s_ in (0, 1))
        w(f"| {r} | {_k(c0, c1)} | {used} | {_cell(c0)} | {_yn(c0, 'keep')} | {_cell(c1)} | {_yn(c1, 'keep')} | "
          f"{yn(rep and rep['replicated'])} ({fmt(rep and rep['two_seed_mean'])}) | "
          f"{yn(rep and rep['replicated_alternative_2se'])} | {v.get('equal_both_status') or yn(v['equal_both'])} | "
          f"{fmt(c0 and c0.get('vs_formula'))} | "
          f"{'/'.join(yn((cv.get(s_) or {}).get('converged')) for s_ in (0, 1))} | {ctrl(c0)} / {ctrl(c1)} |")
        for s_, x in v["originals_not_used"].items():
            co = (v["convergence_original"].get(s_) or {})
            w(f"| {r} s{s_} original - not converged, NOT used by the rules | {_k(x)} | {x['run']} vs {x['vs']} | "
              f"{_cell(x) if s_ == 0 else ''} | {_yn(x, 'keep') if s_ == 0 else ''} | "
              f"{_cell(x) if s_ == 1 else ''} | {_yn(x, 'keep') if s_ == 1 else ''} | | | | | "
              f"{yn(co.get('converged'))} | {ctrl(x)} |")
    w("\nRules use the rerun pair (`<rung>_s<seed>_x2` vs `A2s_s<seed>_x2`) of a seed whenever it exists (AUDIT 4.0, "
      "convergence rule); seeds may then mix an original and a rerun, each paired with its own comparator. Kept = "
      "keep rule on seed 0 and replication by seed 1 (rule applied in seed order): a rung that passes keep on seed 1 "
      "only is not kept; the table shows both keep flags. A primary run that diverged is shown as 'diverged - "
      "awaiting diagnosis'; no rule marks it failed before an independent diagnosis (PLAN 17.11).")
    a2s_mv = {s_: (res["convergence_all"].get(f"A2s_s{s_}") or {}).get("moved") for s_ in (0, 1)}
    w("Reading aid (review B6; decides nothing): every rung that starts at the fitted A2s also carries a common "
      "'continued training' term, measured by the A2c column; convergence movement of the base over its last "
      "window: " + ", ".join(f"A2s_s{s_} {('-' if m is None else f'{m * E4:.2f}e-4')}" for s_, m in a2s_mv.items())
      + ". Differences between rungs that share the start (paired) are free of that term.\n")
    w("\n## Stage 1 - net column (exploratory; vs A2n seed 0, or its warm-start re-run)\n")
    w("| run | k | Δ | keep | equal NLL (seed 0) | converged |\n|---|---|---|---|---|---|")
    for r, c in res["net"].items():
        w(f"| {c['run']} vs {c['vs']} | {_k(c)} | {_cell(c)} | {_yn(c, 'keep')} | {_yn(c, 'equal_one')} | "
          f"{yn((c.get('convergence') or {}).get('converged'))} |")
        sp = c.get("supersedes")
        if sp:
            w(f"| {sp['run']} vs {sp['vs']} (**superseded: warm start**; rules use the row above) | {_k(sp)} | "
              f"{_cell(sp)} | {_yn(sp, 'keep')} | {_yn(sp, 'equal_one')} | |")
        nif = c.get("rerun_not_in_force")
        if nif:
            dg = nif.get("diagnosis") or {}
            w(f"| {nif['run']} (warm-start re-run: **{nif['status']}** - NOT in force; the original above stays in "
              f"force with its warm start flagged{'; diagnosis: ' + str(dg.get('verdict')) if dg.get('valid') else ''}"
              ") | | | | | |")
    w("\n## Stage 1 - exploratory (vs A2s of the same seed)\n")
    w("| rung | k | Δ seed 0 | keep s0 | equal s0 | Δ seed 1 | replicated | replicated under the alternative reading "
      f"(literal conditions AND two-seed mean <= -2 SE; {ALT}) | equal NLL (both) | cleaner (gain / constants) | "
      "control: Δ vs A2c s0 / s1 (decides nothing) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r, v in res["exploratory"].items():
        c0, c1 = v["seeds"].get(0), v["seeds"].get(1)
        rep = v["replication"]
        mark = {"A3c": " (template notes Q1, Q2)", "A3x": " (template note Q1)"}.get(r, "")       # review R5
        w(f"| {r}{mark} | {_k(c0, c1)} | {_cell(c0)} | {_yn(c0, 'keep')} | {_yn(c0, 'equal_one')} | {_cell(c1)} | "
          f"{yn(rep and rep['replicated'])} | {yn(rep and rep['replicated_alternative_2se'])} | "
          f"{v.get('equal_both_status') or yn(v['equal_both'])} | {_yn(c0, 'cleaner_by_gain')} / "
          f"{_yn(c0, 'cleaner_by_constants')} | "
          f"{ctrl(c0)} / {ctrl(c1)} |")
    tl = res.get("templates")
    w("\n**Template limitations (A3c / A3x; review rulings Q1 and Q2, runs as registered):**\n")
    w(f"- {TEMPLATE_NOTE_Q1}")
    if tl:
        fb = tl.get("fb_vs_stored_abar") or {}
        w(f"  This machine: computed front+back vs stored abar max {fb.get('max_abs', float('nan')):.3f}, mean "
          f"{fb.get('mean_abs', float('nan')):.1e}.")
        w("- " + TEMPLATE_NOTE_Q2.format(n101=tl["cells_above_1.01"], z=(
            f"{tl['lowest_zbin_centre_above_1.01_mm']:.0f}" if tl["lowest_zbin_centre_above_1.01_mm"] else "-"),
            mx=tl["max_back_corrected"], den=tl["smallest_denominator_where_above_1.01"] or float("nan"))
          + f" (cells above 1: {tl['cells_above_1']}, above 1.1: {tl['cells_above_1.1']}.)")
    else:
        w("- Q2: templates.json / templates_all.npz not found on this machine.")
    w("\n## Secondary columns (pre-registered, never selection criteria; report metric on strata)\n")
    n = res["secondary_strata_n"]
    w("| run vs comparator | " + " | ".join(f"{k} (n {n[k]})" for k in n) + " |\n|---|" + "---|" * len(n))

    def srow(c, label=""):
        if c.get("status"):
            w(f"| {c['run']} vs {c['vs']}{label} | " + " | ".join(f"**{c['status']}**" for _ in n) + " |")
        else:
            w(f"| {c['run']} vs {c['vs']}{label} | " + " | ".join(fmt(c["secondary"][k]) for k in n) + " |")
    for grp in ("primaries", "exploratory"):
        for r, v in res[grp].items():
            for s_, c in v["seeds"].items():
                srow(c)
            for s_, c in (v.get("originals_not_used") or {}).items():
                srow(c, " (original, not used by the rules)")
    for r, c in res["net"].items():
        srow(c)
        if c.get("supersedes"):
            srow(c["supersedes"], " (superseded: warm start)")
    w("\n## Guards (own tables; K = 1024, bulk stratum, mass monitors, G-region)\n")
    not_used = {c["run"] for v in res["primaries"].values() for c in (v.get("originals_not_used") or {}).values()}
    not_used |= set((res.get("superseded_warm_start") or {}).keys())
    if not res.get("with_guards", True):
        w("**This report was produced WITHOUT guard columns (`--without-guards`): the guards phase failed or has not "
          "run. No guard statement can be read from it.**\n")
    elif "guard_targets" not in dec:                                                          # review R3
        w("**The guards phase has not run** (no guard targets recorded): no guard statement can be read yet.\n")
    elif not res["guards"] and not res.get("guards_missing"):
        w("No run passed the keep or the equal-NLL rule: no guard columns.\n")
    if res.get("guards_missing"):
        w("Guard data MISSING (guards phase failed or not finished) for: " + ", ".join(res["guards_missing"]) + "\n")
    for key, g in res["guards"].items():
        label = " (original, NOT the pair used by the rules)" if g["run"] in not_used else ""          # review R4
        w(f"### {key}{label}\n")
        gk, gb = g["G-K"], g["G-bulk"]
        nk, nb = gk["persistence"]["numbers"], gb["persistence"]["numbers"]
        w("| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |\n"
          "|---|---|---|---|---|")
        w(f"| G-K (K = 1024) | {fmt(gk['delta_k1024'])} | {nk['margin'] * E4:+.2f} ± {nk['se_margin'] * E4:.2f} | "
          f"{yn(gk['persistence']['passed'])} | {yn(gk['persistence']['flags']['resolved'])} |")
        w(f"| G-bulk (zb >= 1221 mm, n {gb['n_stratum']}, excluded {gb['n_excluded_nonfinite']}) | "
          f"{fmt(gb['delta_bulk'])} | {nb['margin'] * E4:+.2f} ± {nb['se_margin'] * E4:.2f} | "
          f"{yn(gb['persistence']['passed'])} | {yn(gb['persistence']['flags']['resolved'])} |")
        w("\nReport metric on the strata (own rows): " + "; ".join(f"{k} {fmt(v)}" for k, v in
                                                                g["report_metric_strata"].items()) + "\n")
        gr = g.get("G-region")
        if gr:
            w("| G-region (hit term by hit centre) | Δ |\n|---|---|")
            for cls, v in gr["hit_term_by_class"].items():
                w(f"| hit term: {cls} | {fmt(v)} |")
            w(f"| hit term: total | {fmt(gr['hit_term_total'])} |")
            w(f"| normalisation term (own row) | {fmt(gr['normalisation_term'])} |")
            sh = gr["flagged_share_of_gain"]
            w(f"\nflagged share of the gain {sh if sh is None else round(sh, 3)}; margin flagged - 0.5 main "
              f"{fmt(gr['margin_flagged_minus_half_main'])} -> passed {yn(gr['passed'])}, resolved "
              f"{yn(gr['resolved'])}; decomposition residual {gr['decomposition_residual_max_abs']:.1e}")
            sa = gr.get("alternative_flagged_share_of_hit_term")
            w(f"*{ALT}:* flagged share of the hit-term total {sa if sa is None else round(sa, 3)}; margin flagged - "
              f"0.5 hit term {fmt(gr.get('alternative_margin_flagged_minus_half_hit_term'))} -> would pass "
              f"{yn(gr.get('alternative_passed_hit_term'))} (requires a gain in the total as well)\n")
        w("| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |\n|---|---|---|---|")
        for k, v in g["G-mass"].items():
            w(f"| {k} | {v['run_mean']:.4e} | {v['comp_mean']:.4e} | {fmt(v['delta'])} |")
        w("")
    if res.get("A6b_fisher_k"):
        f = res["A6b_fisher_k"]
        w(f"## A6b: Fisher information of k\n\nObserved information of the raw coefficient of `sh_rk` on the "
          f"training events: {f['observed_information']:.4g} (SE of the coefficient {f['se_coefficient']}).\n")
    w("## Diverged and blocked runs (user decision of 25 Nov 2026, PLAN 17.11)\n")
    w(f"Definition: {L.DIVERGENCE_DEFINITION_25NOV}\n")
    if not res.get("diverged") and not res.get("blocked"):
        w("None.\n")
    for nme, v in (res.get("diverged") or {}).items():
        lf = v.get("last_finite") or {}
        gr_ = v.get("gradient") or {}
        dg = v.get("diagnosis")
        w(f"- **{nme}** ({v.get('cls')}): {v.get('status_now')}; criteria: {'; '.join(v.get('criteria') or [])}; "
          f"step {v.get('step')}; last finite validation {lf.get('val')} at step {lf.get('step')} (train NLL "
          f"{lf.get('train_nll')}); parameters: {(v.get('params') or {}).get('n_nonfinite')} non-finite, max |finite| "
          f"{(v.get('params') or {}).get('max_abs_finite')}; gradient norm at the last parameters "
          f"{gr_.get('grad_norm')} (finite: {gr_.get('finite')})"
          + (f"; diagnosis: {dg.get('verdict')} (valid: {dg.get('valid')})" if dg else ""))
    for nme, v in (res.get("blocked") or {}).items():
        w(f"- **{nme}** ({v.get('cls')}): blocked by {v.get('blocked_by')} - not trained")
    w("")
    w("## Convergence (|val(last) - val(last - window)| < 0.5e-4)\n")
    w("| run | converged | moved |\n|---|---|---|")
    for k, v in res["convergence_all"].items():
        if v:
            moved = "-" if v["moved"] is None else f"{v['moved'] * E4:.2f}e-4"
            w(f"| {k} | {yn(v['converged'])} | {moved} |")
    w("\n## Initialisation (review B2; kl_map in nat against the start / initialisation / fit_to target)\n")
    w(f"Net runs with kl_map > {KL_DEGRADED:g} nat are flagged: warm start degraded - net column not interpretable "
      "for this run (reference: the pre-registered warm start of A1 gives 5.5e-6 nat). 'step-0 parity' = paired "
      "difference on validation between the network at step 0 and its source polynomial run (user decision of 25 Nov "
      "2026; for runs made before fit_to version 3 it refers to the warm start rebuilt by warmstart-check). "
      "'rms_max / std' is the worst column's rms error divided by that column's spread (review R5). Target columns "
      f"with a relative spread between the 2^-20 floor and {NEAR_CONSTANT_HI:g} and columns below the scale floor "
      "(std < 1e-2, fit_to version 3) are listed.\n")
    w("| run | provider | kl_map | reference | claim | net: fit_to version / rms_max / std (column) / L-BFGS stopped | "
      "step-0 parity Δ(step 0 - source) | flag | nearly-constant / below-floor target columns |\n"
      "|---|---|---|---|---|---|---|---|---|")
    for x in res.get("initialisation") or []:
        kl = x.get("kl_map")
        claim = (f"exactly A2s: {yn(x.get('claim_holds'))}" if x.get("claim") else "")
        netc, flag, near, par0 = "", "", "", ""
        if x.get("provider") == "net":
            rr = x.get("fit_to_rms_max_rel")
            netc = f"v{x.get('fit_to_version')} / {'-' if rr is None else f'{rr * 100:.2f} %'} " \
                   f"({x.get('fit_to_rms_max_name')}) / {x.get('fit_to_lbfgs_stopped') or 'no'}"
            flag = "**warm start degraded**" if x.get("degraded") else "ok"
            if x.get("superseded_by"):
                flag += f"; **superseded: warm start** -> {x['superseded_by']}"
            tg = x.get("targets") or {}
            if "error" in tg:
                near = f"not recomputed ({tg['error'][:80]})"
            else:
                nc = tg.get("nearly_constant") or {}
                bf = tg.get("below_scale_floor") or []
                near = (", ".join(f"{k} {v:.1e}" for k, v in nc.items()) or "none") + \
                       (f"; below floor: {', '.join(bf)}" if bf else "")
            p0 = x.get("step0_parity") or {}
            if p0.get("available"):
                par0 = f"{fmt(p0.get('paired'))} vs {p0.get('source')}, max {p0.get('max_abs_event', 0):.1e}" \
                       + (" (rebuilt)" if "rebuilt" in str(p0.get("origin")) else "")
            elif p0:
                par0 = f"n/a ({p0.get('reason')})"
        w(f"| {x['run']} | {x.get('provider')} | {'-' if kl is None else f'{kl:.2e}'} | {x.get('kl_ref') or '-'} | "
          f"{claim} | {netc} | {par0} | {flag} | {near} |")
    nearly = [(x["run"], (x.get("targets") or {}).get("nearly_constant")) for x in res.get("initialisation") or []
              if (x.get("targets") or {}).get("nearly_constant")]
    w("\nNearly-constant target columns in a net warm start: "
      + ("; ".join(f"{r}: {', '.join(nc)}" for r, nc in nearly) if nearly else "none") + ".\n")
    w("\n## Kept and replicated (the input of Stage 2; no combination formed here)\n")
    prs, exs = set(res.get("primary_rungs") or []), set(res.get("exploratory_rungs") or [])
    kst = res.get("kept_status") or {}
    for label, keys in (("primaries", prs), ("exploratory", exs)):
        w(f"- {label}: " + ", ".join(f"{r}: {kst.get(r) or yn(v)}" for r, v in res["kept_and_replicated"].items()
                                     if r in keys))
    w(f"- *{ALT}* (literal conditions AND two-seed mean <= -2 SE): " + ", ".join(
        f"{r}: {yn(v)}" for r, v in (res.get("kept_and_replicated_alternative_2se") or {}).items()) + "\n")
    return "\n".join(out) + "\n"


def report(with_guards: bool = True) -> dict:
    """The full report (results.json, RESULTS.md) or, with ``with_guards=False`` (review C2), the tables without guard
    columns in RESULTS_without_guards.md / results_without_guards.json (results.json is left untouched)."""
    res = collect(with_guards=with_guards)
    od = L.out_dir()
    suffix = "" if with_guards else "_without_guards"
    jf = od / f"results{suffix}.json"
    L.write_json(res, jf)
    md = paths.scrub_str(markdown(res))
    f = paths.check_output_path(od / f"RESULTS{suffix}.md")
    tmp = f.with_name(f".{f.name}.tmp")
    tmp.write_text(md)
    tmp.replace(f)
    log(f"report: {paths.portable(f)} and {jf.name} written ({len(res['primaries'])} primaries, "
        f"{len(res['exploratory'])} exploratory rungs, {len(res['guards'])} guard blocks"
        + (f", guard data missing for {len(res['guards_missing'])} pairs" if res["guards_missing"] else "")
        + ("" if with_guards else ", WITHOUT guard columns") + ")")
    return res


# ---------------------------------------------------------------------------------------------------- stage 2 ---------
def gate1_warnings(stage2: bool = False) -> dict:
    """PLAN 20 item 2: runs whose fit recovered from a non-finite validation mean before its final value (a warning
    for Gate 1, not a divergence), from the fit histories of the done.json files (Stage-1 or Stage-2 runs)."""
    out = {}
    for f in sorted(L.out_dir().glob("*/done.json")):
        d = L.done_marker(f) or {}
        if ((d.get("spec") or {}).get("cls") == "stage2") != stage2:
            continue
        w = L.recovered_nonfinite(d.get("history") or [])
        if w["count"]:
            out[f.parent.name] = w
    return out


def reclassification() -> dict:
    """REVIEW 10.5 R3: every finished run of this stage classified by criterion (i) of the definition of 25 Nov 2026
    (any non-finite validation mean of the history) and of the present one (a non-finite final value); the other
    criteria are the same in both. Runs whose class would differ are listed."""
    n, nonfinite, differ = 0, {}, []
    for f in sorted(L.out_dir().glob("*/done.json")):
        d = L.done_marker(f) or {}
        hist = d.get("history") or []
        bad = [h["step"] for h in hist if h.get("val") is None or not math.isfinite(float(h["val"]))]
        n += 1
        if bad:
            nonfinite[f.parent.name] = bad
            old = True
            new = bool(hist) and (hist[-1].get("val") is None or not math.isfinite(float(hist[-1]["val"])))
            if old != new:
                differ.append(f.parent.name)
    return {"finished_runs": n, "runs_with_a_non_finite_validation_mean": nonfinite, "classified_differently": differ,
            "diverged_records": sorted((L.decisions().get("diverged") or {}).keys())}


def _gr_star_tagged(name: str) -> provenance.TaggedArray:
    sdir = L.scores_dir(name)
    mk = L.read_json(sdir / "g_region_star.json")
    if not mk:
        raise FileNotFoundError(f"{L.stage()}/{name}: no flagged* array (guards phase, REVIEW 10.5 R6)")
    ta = provenance.load_tagged(sdir / f"g_region_{_m().G_STAR}.npy", expect_machine_id=L.live_machine().id)
    if ta.sha256 != mk["array"]["sha256"]:
        raise RuntimeError(f"{name}: flagged* array differs from its marker")
    return ta


def g_region_star_check(run: str, comp: str, cls: str = "low_abar") -> dict:
    """The waiver line of REVIEW 10.5 R6 (user decision of PLAN 20): the rule of G-region with flagged* (deep, last
    ring, first / last z-bin: flagged for a reason other than low acceptance) in place of flagged - gain < 0 and at
    most 50 % of it from flagged*, i.e. margin flagged* - 0.5 main >= 0 (paired SE, resolved at 2 SE). 'covered' =
    that rule holds, i.e. the failure of G-region as registered lies within the waived class. It is never a
    statement that G-region passes."""
    try:
        a, b = _gr_star_tagged(run), _gr_star_tagged(comp)
    except FileNotFoundError as err:
        return {"class": cls, "available": False, "covered": False, "reason": str(err)}
    main = provenance.paired_difference(L.tagged(run), L.tagged(comp))
    st = provenance.paired_difference(a, b)
    margin = stats.paired(st - 0.5 * main, np.zeros_like(main))
    gain = float(main.mean())
    return {"class": cls, "available": True, "hit_term_flagged_star": L.pd(provenance.paired_delta(a, b)),
            "flagged_star_share_of_gain": (float(st.mean()) / gain) if gain != 0 else None,
            "margin_flagged_star_minus_half_main": L.pd(margin), "resolved": abs(margin.z) >= 2.0,
            "covered": bool(gain < 0 and margin.mean >= 0.0)}


def _r3(x):
    return "-" if x is None else round(x, 3)


FORMULA_CONSTANTS = 101
CLEANER_RULE = ("'cleaner' (AUDIT 4.1 Stage 2; REVIEW 10.4; PLAN 20: the legacy formula stays the fallback "
                "comparator), against THIS machine's reference array of the legacy formula (k = 101): (c1) better by at "
                "least 3: Δ_0 <= -3e-4, with the replication rule of AUDIT 4.0 in its literal reading: Δ_1 <= -1.5e-4 and "
                "(Δ_0 + Δ_1)/2 <= -3e-4; or (c2) k <= 90 (0.9 x 101: at least 10 % fewer constants) and equal NLL "
                "against the formula on both seeds (Δ_s <= +1 SE_s and Δ_s < +1e-4). Seeds 0 and 1; seed 2 of A11 is "
                "reported and decides nothing.")
FALLBACK_NOTE = ("If no Stage-2 model meets the 'cleaner' bar against the formula, or the finalist fails Gate 1, the "
                 "legacy formula remains the Track A baseline for PySR (user decision of 26 Nov 2026, PLAN 20).")
FINALIST_RULE = ("REVIEW 10.4: candidates A2s, A11, A11k and A11p (A11p only if accepted without hurting the short "
                 "events); for each of the seeds 0 and 1 sfm.stats.one_se_select on the candidates' registered arrays "
                 "of that seed (best = lowest mean; eligible if Δ < SE, strict); a candidate is eligible if it is "
                 "eligible on both seeds; the finalist is the eligible candidate with the fewest constants, ties to the "
                 "lower two-seed mean. No candidate eligible on both seeds: no automatic finalist (the table goes to "
                 "the user). A11n is not a candidate (PLAN 23 item 6): the table is printed once more with A11n for "
                 "information, and the report escalates to the user at Gate 1 if A11n beats every polynomial "
                 "candidate by more than 1 SE on both seeds.")
SHORT = "short (zb < 1221 mm)"


def stage2_masks() -> dict:
    """The registered secondary strata plus the short / bulk validation strata (REVIEW 10.4)."""
    m = secondary_masks()
    sets, _ = L.data_val()
    zb = sets["val"].ev["zb"].numpy().astype(np.float64)
    m[SHORT] = zb < registry.ZB_STRATUM_MM
    m["bulk (zb >= 1221 mm)"] = ~m[SHORT]
    return m


def s2_pair(label: str, run: str, comp: str | None, masks: dict) -> dict:
    """One comparison row of REVIEW 10.4 (provenance.paired_delta on registered arrays), or its status."""
    row = {"label": label, "run": run, "vs": comp}
    if comp is None:
        return {**row, "paired": None, "status": "no comparator"}
    if not have(run) or not (comp == "formula" or have(comp)):
        st = run_status(run) if not have(run) else f"comparator {comp}: {run_status(comp)}"
        return {**row, "paired": None, "status": st}
    row["paired"] = pdict(run, comp)
    row["strata"] = {k: (pdict(run, comp, mask=m) if m.sum() >= 2 else None) for k, m in masks.items()}
    return row


def reproducibility(run: str, ref: str) -> dict | None:
    """REVIEW 10.3 item 2: A11 with a one-rung content is the same computation as that rung (same arguments, start,
    seed, schedule); the maximum absolute per-event difference of the registered arrays must be 0."""
    if not (have(run) and have(ref)):
        return None
    a, b = L.tagged(run), L.tagged(ref)
    provenance.check_pairable(a, b)
    d = a.values.astype(np.float64) - b.values.astype(np.float64)
    return {"run": run, "ref": ref, "max_abs_per_event": float(np.abs(d).max()),
            "bit_identical": bool(np.array_equal(a.values, b.values))}


def stage2_comparisons(content: list, masks: dict) -> list:
    rows = []
    for s in (0, 1, 2):
        rows.append(s2_pair("A11 - A2s", f"A11_s{s}", f"A2s_s{s}", masks))
    for s in (0, 1):
        rows.append(s2_pair("A11 - A2c", f"A11_s{s}", f"A2c_s{s}", masks))
    if have("A11_s2"):
        rows.append({"label": "A11 - A2c", "run": "A11_s2", "vs": None, "paired": None,
                     "status": "no matched control exists for seed 2"})
    if len(content) == 1:
        for s in (0, 1):
            rows.append(s2_pair(f"A11 - {content[0]} (reproducibility)", f"A11_s{s}", f"{content[0]}_s{s}", masks))
    for s in (0, 1):
        rows.append(s2_pair("A11k - A11", f"A11k_s{s}", f"A11_s{s}", masks))
        rows.append(s2_pair("A11k - A2s", f"A11k_s{s}", f"A2s_s{s}", masks))
    for s in (0, 1):
        rows.append(s2_pair("A11p - A11", f"A11p_s{s}", f"A11_s{s}", masks))
        rows.append(s2_pair("A11p - A2s", f"A11p_s{s}", f"A2s_s{s}", masks))
        rows.append(s2_pair("A11p - A2c", f"A11p_s{s}", f"A2c_s{s}", masks))
    for s in (0, 1):
        rows.append(s2_pair("A11n - A2n (in force)", f"A11n_s{s}", L.effective(f"A2n_s{s}"), masks))
        rows.append(s2_pair("A11n - A11", f"A11n_s{s}", f"A11_s{s}", masks))
    for rung, seeds in (("A2s", (0, 1, 2)), ("A11", (0, 1, 2)), ("A11k", (0, 1)), ("A11p", (0, 1)), ("A11n", (0, 1))):
        for s in seeds:
            rows.append(s2_pair(f"{rung} - formula", f"{rung}_s{s}", "formula", masks))
    return [r for r in rows if r.get("paired") is not None or r["status"] not in ("missing",)]


def _k_of(run: str):
    row = L.registered_row(run)
    return None if row is None else int(row["k"])


def cleaner(rung: str) -> dict:
    """(c1) and (c2) of CLEANER_RULE for one rung (seeds 0 and 1) against the formula array."""
    M = _m()
    if not (have(f"{rung}_s0") and have(f"{rung}_s1")):
        return {"rung": rung, "evaluable": False}
    p = [L.paired(f"{rung}_s{s}", "formula") for s in (0, 1)]
    d0, d1 = p[0].mean, p[1].mean
    k = _k_of(f"{rung}_s0")
    c1 = bool(d0 <= -M.KEEP_GAIN and d1 <= -0.5 * M.KEEP_GAIN and 0.5 * (d0 + d1) <= -M.KEEP_GAIN)
    eq = [bool(M.equal_rule_one(x)) for x in p]
    c2 = bool(k <= 0.9 * FORMULA_CONSTANTS and all(eq))
    return {"rung": rung, "evaluable": True, "k": k, "delta": [L.pd(x) for x in p], "mean_of_seeds": 0.5 * (d0 + d1),
            "c1": c1, "c1_parts": {"d0<=-3": bool(d0 <= -M.KEEP_GAIN), "d1<=-1.5": bool(d1 <= -0.5 * M.KEEP_GAIN),
                                   "mean<=-3": bool(0.5 * (d0 + d1) <= -M.KEEP_GAIN)},
            "c2": c2, "c2_parts": {"k<=90": bool(k <= 0.9 * FORMULA_CONSTANTS), "equal_s0": eq[0], "equal_s1": eq[1]},
            "cleaner": bool(c1 or c2)}


def finalist(candidates: list) -> dict:
    """FINALIST_RULE on the given candidate rungs (those with both seeds registered)."""
    usable = [c for c in candidates if have(f"{c}_s0") and have(f"{c}_s1")]
    out = {"candidates": usable, "not_candidates": [c for c in candidates if c not in usable], "tables": {},
           "eligible_both": [], "finalist": None}
    if not usable:
        return out
    within, means = {c: True for c in usable}, {c: [] for c in usable}
    for s in (0, 1):
        arrs = [L.tagged(f"{c}_s{s}") for c in usable]
        provenance.check_pairable(*arrs)
        sel = stats.one_se_select([stats.Candidate(c, a.values.astype(np.float64), k=_k_of(f"{c}_s{s}"))
                                   for c, a in zip(usable, arrs)])
        out["tables"][s] = sel.table
        for row in sel.table:
            within[row["name"]] = within[row["name"]] and row["within_1se"]
            means[row["name"]].append(row["mean"])
    elig = [c for c in usable if within[c]]
    out["eligible_both"] = elig
    if elig:
        out["finalist"] = min(elig, key=lambda c: (_k_of(f"{c}_s0"), float(np.mean(means[c]))))
    out["two_seed_mean"] = {c: float(np.mean(v)) for c, v in means.items()}
    return out


def a11n_escalation(candidates: list) -> dict:
    """PLAN 23 item 6: A11n stays out of the automatic polynomial finalist set; it is escalated to the user at Gate 1
    if it beats every polynomial candidate by more than 1 paired SE on both seeds (Δ = A11n - candidate < -SE)."""
    poly = [c for c in candidates if have(f"{c}_s0") and have(f"{c}_s1")]
    if not (have("A11n_s0") and have("A11n_s1")) or not poly:
        return {"evaluable": False, "escalate": False}
    rows = {}
    for c in poly:
        rows[c] = {}
        for s in (0, 1):
            p = L.paired(f"A11n_s{s}", f"{c}_s{s}")
            rows[c][s] = {"paired": L.pd(p), "beats_by_more_than_1se": bool(p.mean < -p.se)}
    esc = all(v[s]["beats_by_more_than_1se"] for v in rows.values() for s in (0, 1))
    return {"evaluable": True, "polynomial_candidates": poly, "pairs": rows, "escalate": esc}


def u2_max(n_s: int) -> int:
    """REVIEW 11.6 (b): N_s <= 95 for n_S = 166, the level P(Binomial(166, 1/2) > 95); for another n_S (smoke data
    are the same split, so this is a guard) the smallest N with P(Binomial(n_S, 1/2) > N) <= that level."""
    def tail(n, N):
        return sum(math.comb(n, j) for j in range(N + 1, n + 1)) / 2 ** n
    n0, N0 = _m().A11P_U2_REF
    if n_s == n0:
        return N0
    alpha = tail(n0, N0)
    N = 0
    while tail(n_s, N) > alpha:
        N += 1
    return N


def a11p_acceptance(share: float | None, name: str = "A11p") -> dict | None:
    """REVIEW 11.6 on the registered validation arrays: registered (equal NLL against A11_s on both seeds) and the
    short-event conditions U1 (Δ^S + 2 SE^S < share x 1e-4 / f) and U2 (N_s <= u2_max(n_S)) on both seeds."""
    M = _m()
    if share is None or not all(have(f"{name}_s{s}") and have(f"A11_s{s}") for s in (0, 1)):
        return None
    sets, _ = L.data_val()
    S = sets["val"].ev["zb"].numpy().astype(np.float64) < registry.ZB_STRATUM_MM
    f = float(S.mean())
    margin = share * 1e-4 / f
    nmax = u2_max(int(S.sum()))
    seeds = {}
    for s in (0, 1):
        a, b = L.tagged(f"{name}_s{s}"), L.tagged(f"A11_s{s}")
        p = provenance.paired_delta(a, b)
        d = provenance.paired_difference(a, b)
        dS = stats.paired(d[S], np.zeros(int(S.sum())))
        n_worse = int((d[S] > 1e-4).sum())
        seeds[s] = {"paired": L.pd(p), "equal": bool(M.equal_rule_one(p)), "short": L.pd(dS),
                    "U1_bound": dS.mean + 2 * dS.se, "U1": bool(dS.mean + 2 * dS.se < margin),
                    "N_worse_by_1e-4": n_worse, "U2": bool(n_worse <= nmax)}
    reg = all(v["equal"] for v in seeds.values())
    short_ok = all(v["U1"] and v["U2"] for v in seeds.values())
    return {"run": name, "share": share, "f_short": f, "n_short": int(S.sum()), "U1_margin": margin, "U2_max": nmax,
            "seeds": seeds, "accepted": reg, "accepted_without_hurting_short": bool(reg and short_ok),
            "verdict": ("accepted without hurting the short events" if reg and short_ok else
                        "equal NLL overall, short events hurt" if reg else "not accepted (not equal NLL)")}


def collect_stage2() -> dict:
    M = _m()
    dec = L.decisions()
    content = dec.get("stage2_content")
    if content is None:
        raise SystemExit("stage2 report: no stage2_content record (the stage2 phase has not started)")
    run = content.get("a11_content") or []
    s1 = json.loads((L.out_dir() / "results.json").read_text())
    from sfm import machine as _machine
    gs = _machine.git_state()
    masks = stage2_masks()
    specs = M.stage2_specs(run)
    par = content.get("a11p_parameters") or {}
    screen = None
    sc_rec = dec.get("stage2_a11p_screen")
    if sc_rec:
        f = L.out_dir() / M.A11P_DIR / "screen.json"
        screen = json.loads(f.read_text()) if f.exists() else None
        if screen is not None and provenance.sha256_file(f) != sc_rec["sha256"]:
            raise RuntimeError("stage2 report: a11p/screen.json differs from its recorded SHA-256")
    acc = a11p_acceptance(par.get("short_event_share"))
    acc_z0 = a11p_acceptance(par.get("short_event_share"), name="A11pz")
    cc = dec.get("conditional_controls")
    ccr = {}
    for ctl, mk in (("A2c_x2", M.a2c_x2_specs), ("A2c6k", M.a2c6k_specs)):
        if cc and cc[ctl]["triggered"]:
            for sp in mk(cc[ctl]["seeds"]):
                ccr[sp.name] = entry(sp.name, sp.comparator, {}) or {"run": sp.name, "vs": sp.comparator,
                                                                    "status": "not run yet"}
    cands = ["A2s", "A11", "A11k"] + (["A11p"] if acc and acc["accepted_without_hurting_short"] else [])
    fin = finalist(cands)
    fin_net = finalist(cands + ["A11n"])
    esc = a11n_escalation(cands)
    rungs = ["A11", "A11k", "A11p", "A11n"] + (["A2s"] if fin.get("finalist") == "A2s" else [])
    cl = {r: cleaner(r) for r in rungs}
    guards = {}
    pairs = (dec.get("stage2_guard_targets") or {}).get("pairs", []) + \
        (dec.get("stage2_a11p_guard_targets") or {}).get("pairs", [])
    for r_, c_ in pairs:
        blk = guard_block(r_, c_)
        if blk is None:
            guards[f"{r_} vs {c_}"] = {"run": r_, "vs": c_, "missing": True}
            continue
        blk["waiver_lines"] = [{**wv, "check": g_region_star_check(r_, c_, wv["class"])}
                               for wv in content.get("waivers") or []]
        guards[f"{r_} vs {c_}"] = blk
    names = [sp.name for sp in specs] + [f"{r}_s{s}" for r in ("A11p", "A11pz") for s in (0, 1)]
    s2dec = {k: {n: v for n, v in (dec.get(k) or {}).items() if L.is_stage2_run(n)}
             for k in ("diverged", "blocked", "warnings")}
    fa = L.formula_array()
    return {"stage": L.stage(), "smoke": L.CTX.smoke, "machine_id": L.MACHINE_ID, "stream_id": L.live_stream_id(),
            "git": {"commit": gs.get("commit"), "clean": gs.get("clean")},
            "units": "1e-4 nat per event (deltas); paired SE; n events",
            "approval": dec.get("stage2_approval"), "content": content,
            "stage1_results_json_sha256": provenance.sha256_file(L.out_dir() / "results.json"),
            "stage1_guards": {k: (g.get("G-region") or {}) for k, g in (s1.get("guards") or {}).items()},
            "memory_probe": dec.get("stage2_memory_probe"), "a11p_status": dec.get("stage2_a11p"),
            "formula": {"constants": FORMULA_CONSTANTS, "validation_mean": float(np.mean(fa.values, dtype=np.float64)),
                        "reference_array": paths.portable(L.refset().dir)},
            "reproducibility": ([reproducibility(f"A11_s{s}", f"{run[0]}_s{s}") for s in (0, 1)]
                                if len(run) == 1 else None),
            "comparisons": stage2_comparisons(run, masks),
            "strata_n": {k: int(v.sum()) for k, v in masks.items()},
            "finalist_rule": FINALIST_RULE, "finalist": fin, "finalist_with_A11n": fin_net,
            "finalist_changed_by_A11n": fin_net.get("finalist") != fin.get("finalist"), "a11n_escalation": esc,
            "cleaner_rule": CLEANER_RULE, "cleaner": cl, "fallback": FALLBACK_NOTE,
            "a11p_screen": screen, "a11p_screen_sha256": (sc_rec or {}).get("sha256"), "a11p_acceptance": acc,
            "a11p_z0_acceptance": acc_z0, "conditional_controls": cc, "conditional_controls_runs": ccr,
            "warnings_gate1_stage1": gate1_warnings(stage2=False), "reclassification": reclassification(),
            "divergence_definition": L.DIVERGENCE_DEFINITION,
            "initialisation": initialisation_table(stage2=True),
            "convergence": {n: convergence(n) for n in names if L.done_marker(L.out_dir() / n / "done.json")},
            "guards": guards, "diverged": s2dec["diverged"], "blocked": s2dec["blocked"],
            "warnings_gate1": gate1_warnings(stage2=True)}


def _s2_guard_text(g: dict) -> list:
    out = []
    gk, gb = g["G-K"], g["G-bulk"]
    out.append(f"- G-K: {fmt(gk['delta_k1024'])}, passed {yn(gk['persistence']['passed'])}, resolved "
               f"{yn(gk['persistence']['flags']['resolved'])}")
    out.append(f"- G-bulk: {fmt(gb['delta_bulk'])}, passed {yn(gb['persistence']['passed'])}, resolved "
               f"{yn(gb['persistence']['flags']['resolved'])}")
    out.append("- report metric on the strata: " + "; ".join(f"{k} {fmt(v)}" for k, v in
                                                           g["report_metric_strata"].items()))
    gr = g.get("G-region")
    if gr:
        out.append("- G-region as registered, hit term by class: " + "; ".join(
            f"{k} {fmt(v)}" for k, v in gr["hit_term_by_class"].items()) + f"; normalisation "
            f"{fmt(gr['normalisation_term'])}; flagged share {_r3(gr['flagged_share_of_gain'])}, margin "
            f"{fmt(gr['margin_flagged_minus_half_main'])} -> " + ("**FAILED**" if not gr["passed"] else "not failed"))
    for wv in g.get("waiver_lines") or []:
        ch = wv["check"]
        if not ch.get("available"):
            out.append(f"- User's waiver of PLAN 20 ({wv['date']}; {wv['rung']}, class {wv['class']}, subject to "
                       f"{wv['subject_to']}): flagged* line NOT AVAILABLE ({ch.get('reason')}) - not covered")
            continue
        out.append(f"- User's waiver of PLAN 20 ({wv['date']}; {wv['rung']} is part of this model; {wv['guard']}, "
                   f"class {wv['class']} only, subject to {wv['subject_to']}): flagged* share "
                   f"{_r3(ch['flagged_star_share_of_gain'])}, margin {fmt(ch['margin_flagged_star_minus_half_main'])} "
                   f"(resolved {yn(ch['resolved'])}) -> the failure lies within the waived class: "
                   f"**{yn(ch['covered'])}**" + ("" if ch["covered"] else " - NOT covered by the waiver"))
    out.append("- G-mass: " + "; ".join(f"{k} {fmt(v['delta'])}" for k, v in g["G-mass"].items()) + "\n")
    return out


def markdown_stage2(res: dict) -> str:
    out = []
    w = out.append
    w("# Track A (M2A) Stage 2 - " + ("SMOKE RUN (shortened fits; not numbers of record)" if res["smoke"] else
                                      "results on the machine of record") + "\n")
    c = res["content"]
    ap = res.get("approval") or {}
    w(f"Stage `{res['stage']}`, machine `{res['machine_id']}`, stream `{res['stream_id']}`, report code at commit "
      f"`{(res.get('git') or {}).get('commit')}` (clean clone: {(res.get('git') or {}).get('clean')}). Approved "
      f"Stage-1 results.json `{str(ap.get('results_json_sha256'))[:16]}...` (approved {ap.get('approved_utc')} by "
      f"{ap.get('approved_by')}; reading '{c.get('replication_reading')}'). Validation only; deltas in 1e-4 nat per "
      "event, mean ± paired SE (n), through `sfm.provenance.paired_delta`.\n")
    w("## Content of A11 (user decisions of 26 Nov 2026, PLAN 20; REVIEW 10.5 R1)\n")
    w(f"- Computed kept-and-replicated list: {c.get('kept_and_replicated')}")
    for r, e in (c.get("excluded") or {}).items():
        w(f"- **User exclusion** ({e['date']}): **{r} does not enter A11** - {e['reason']}")
    for wv in c.get("waivers") or []:
        w(f"- **Guard waiver** ({wv['date']}): {wv['rung']} enters under an explicit waiver of **{wv['guard']}** for "
          f"the class **{wv['class']}** only, subject to {wv['subject_to']} - {wv['basis']}")
    for rung, chks in (c.get("waiver_checks") or {}).items():
        for key, chk in chks.items():
            w(f"  - Stage 1, {key}: G-region as registered FAILED: {yn(chk.get('g_region_passed') is False)}; "
              + (f"flagged* share {_r3(chk['flagged_star_share_of_gain'])}, margin "
                 f"{fmt(chk['margin_flagged_star_minus_half_main'])} -> within the waived class "
                 f"{yn(chk['covered'])}" if chk.get("available") else "flagged* line not available"))
    w(f"- **Content = computed list minus user exclusions: {c.get('a11_content')}** -> A11 = {c.get('A11')}; A11n: "
      f"{c.get('A11n')}; lr_net of record {c.get('lr_net_of_record')}")
    par = c.get("a11p_parameters") or {}
    w(f"- A11p: {(res.get('a11p_status') or {}).get('status', 'not started')}; parameters of the [confirm] choices of "
      f"REVIEW 11: short-event share {par.get('short_event_share')}, if the short events veto coefficients: "
      f"{par.get('if_vetoed')} ({par.get('source')})")
    mp = res.get("memory_probe")
    if mp:
        w(f"- Memory probe before the first k_train = 64 fit (REVIEW 10.5 R2): peak {mp['peak_gb']:.2f} GB for "
          f"{mp['hits_in_batch']} hits, limit {mp['limit_gb']} GB (cap {mp['cap_gb']}) -> "
          f"{'fits' if mp['fits'] else 'DOES NOT FIT'}")
    rp = res.get("reproducibility")
    if rp:
        w("- **Reproducibility (REVIEW 10.3 item 2):** A11 with this content is the same computation as the rung "
          "(same arguments, start, seed and schedule); the two runs are not a further replication of it, only seed "
          "2 is new information. " + "; ".join(
              f"{x['run']} - {x['ref']}: max |Δ per event| {x['max_abs_per_event']:.3e} nat, bit for bit: "
              f"**{yn(x['bit_identical'])}**" for x in rp if x) + ("" if all(x for x in rp) else " (a run is missing)"))
    w("")
    w("## Comparisons (REVIEW 10.4)\n")
    n = res["strata_n"]
    w("| comparison | run vs comparator | Δ (all events) | " + " | ".join(f"{k} (n {n[k]})" for k in n) + " |\n|---|---|"
      + "---|" * (1 + len(n)))
    for r in res["comparisons"]:
        if r.get("paired") is None:
            w(f"| {r['label']} | {r['run']} vs {r['vs']} | **{r.get('status')}** |" + " |" * len(n))
            continue
        w(f"| {r['label']} | {r['run']} vs {r['vs']} | {fmt(r['paired'])} | "
          + " | ".join(fmt(r["strata"].get(k)) for k in n) + " |")
    w("\n## Finalist (REVIEW 10.4)\n")
    w(res["finalist_rule"] + "\n")
    for label, fin in (("candidates", res["finalist"]), ("with A11n", res["finalist_with_A11n"])):
        w(f"**{label}:** {fin['candidates']} (not candidates / incomplete: {fin['not_candidates']})\n")
        for s_, tab in fin["tables"].items():
            w(f"seed {s_}:\n\n| candidate | k | mean | Δ vs best | SE | within 1 SE | best |\n|---|---|---|---|---|---|---|")
            for row in tab:
                w(f"| {row['name']} | {row['k']} | {row['mean']:.6f} | {row['delta'] * E4:+.2f} | {row['se'] * E4:.2f} | "
                  f"{yn(row['within_1se'])} | {yn(row['best'])} |")
            w("")
        fz_ = fin["finalist"] or "none (no automatic finalist; the table goes to the user)"
        w(f"eligible on both seeds: {fin['eligible_both']} -> finalist: **{fz_}**\n")
    if res["finalist_changed_by_A11n"]:
        w("Including A11n changes the 1-SE selection (information; A11n is not a candidate, PLAN 23 item 6).\n")
    esc = res.get("a11n_escalation") or {}
    if esc.get("evaluable"):
        w("A11n against every polynomial candidate (PLAN 23 item 6; beats = Δ < -1 SE): " + "; ".join(
            f"{c} s{s_}: {fmt(v[s_]['paired'])} {'beats' if v[s_]['beats_by_more_than_1se'] else 'does not beat'}"
            for c, v in esc["pairs"].items() for s_ in (0, 1)) + "\n")
        if esc["escalate"]:
            w("**ESCALATE TO THE USER AT GATE 1: A11n beats every polynomial model by more than 1 SE on both "
              "seeds.**\n")
    w("## 'Cleaner' against the legacy formula\n")
    f = res["formula"]
    w(f"Legacy formula: k = {f['constants']}, validation mean on this machine {f['validation_mean']:.6f} (reference "
      f"array `{f['reference_array']}`). {res['cleaner_rule']}\n")
    w("| rung | k | Δ vs formula s0 | Δ vs formula s1 | (c1) Δ_0 <= -3, Δ_1 <= -1.5, mean <= -3 | (c2) k <= 90 and equal "
      "NLL s0 / s1 | cleaner |\n|---|---|---|---|---|---|---|")
    for r, v in res["cleaner"].items():
        if not v["evaluable"]:
            w(f"| {r} | - | - | - | not evaluable (a seed is missing) | - | - |")
            continue
        p1, p2 = v["c1_parts"], v["c2_parts"]
        w(f"| {r} | {v['k']} | {fmt(v['delta'][0])} | {fmt(v['delta'][1])} | {yn(v['c1'])} ({yn(p1['d0<=-3'])}, "
          f"{yn(p1['d1<=-1.5'])}, {yn(p1['mean<=-3'])}; mean {v['mean_of_seeds'] * E4:+.2f}) | {yn(v['c2'])} "
          f"({yn(p2['k<=90'])}; {yn(p2['equal_s0'])} / {yn(p2['equal_s1'])}) | **{yn(v['cleaner'])}** |")
    fz = res["finalist"].get("finalist")
    fc = res["cleaner"].get(fz) if fz else None
    if fz is None:
        w("\nNo automatic finalist, so no finalist 'cleaner' statement; " + res["fallback"] + "\n")
    elif fc and fc.get("evaluable") and not fc["cleaner"]:
        w(f"\n**The finalist {fz} is not cleaner than the legacy formula; the legacy formula remains the Track A "
          "baseline (PLAN 20).**\n")
    elif fc and fc.get("cleaner"):
        w(f"\nThe finalist {fz} is cleaner than the legacy formula by ({'c1' if fc['c1'] else 'c2'}); Gate 1 "
          "decides.\n")
    else:
        w(f"\nThe finalist {fz}: 'cleaner' not evaluable. " + res["fallback"] + "\n")
    w("## A11p (REVIEW 11)\n")
    sc = res.get("a11p_screen")
    if sc is None:
        w(f"Screen not run: {(res.get('a11p_status') or {}).get('status', 'not started')}.\n")
    else:
        w(f"Screen `a11p/screen.json` (SHA-256 `{str(res.get('a11p_screen_sha256'))[:16]}...`): {sc['n_candidates']} "
          f"candidates; threshold Q_j < {sc['threshold_nat'] * E4:.1f}e-4 nat; short events f_tr Q^S_j < "
          f"{sc['short_threshold_nat'] * E4:.2f}e-4 nat (share {sc['short_event_share']}). Z_0 (registered condition "
          f"alone, both seeds): {len(sc['Z0'])}; **Z (with the short-event condition): {len(sc['Z'])}**; vetoed by the "
          f"short events (Z_0 minus Z): {sc['Z0_minus_Z']}. k {sc['k_A11']} -> {sc['k_A11p']}.")
        for s_, j in sc["joint"].items():
            w(f"- seed {s_}: sum of Q_j over Z {j['sum_Q_dropped'] * E4:.3f}e-4; joint cost before the refit J "
              f"{j['J'] * E4:+.3f}e-4, J^S {j['J_S'] * E4:+.3f}e-4")
        cc_ = sc.get("curvature_check") or {}

        def _g(x, f=".4g"):
            return "non-finite" if x is None else format(x, f)
        w(f"- autodiff check of H_j (REVIEW 11.2; 256 training events, the most resolved candidates): passed "
          f"**{yn(cc_.get('passed'))}** (tolerance {cc_.get('tolerance')}, finite comparisons {cc_.get('n_finite')}): "
          + "; ".join(f"{k}: autodiff {_g(v['autodiff'])}, second difference {_g(v['second_difference'])} (rel. "
                      f"{_g(v['rel_diff'], '.1e')})" for k, v in (cc_.get("coefficients") or {}).items()))
        w(f"- reduced degrees: " + ", ".join(f"{k} {v}" for k, v in sc["prune"].items()) + "\n")
        for s_, t in sc["tables"].items():
            w(f"Seed {s_} ({t['run']}; n_train {t['n_train']}, short {t['n_short']}, f_tr {t['f_tr']:.6f}):\n")
            w("| row | k | c | h | H | Q (1e-4) | -g c (1e-4) | H^S | Q^S (1e-4) | f_tr Q^S (1e-4) | pass reg. | pass "
              "short |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in t["rows"]:
                w(f"| {r['row']} | {r['k']} | {r['c']:.4g} | {r['h']:.3g} | {r['H']:.4g} | {r['Q'] * E4:.4f} | "
                  f"{r['minus_g_c'] * E4:+.4f} | {r['H_S']:.4g} | {r['Q_S'] * E4:.3f} | {r['f_tr_Q_S'] * E4:.4f} | "
                  f"{yn(r['pass_registered'])} | {yn(r['pass_short'])} |")
            w("")
    acc = res.get("a11p_acceptance")
    if acc:
        w(f"Acceptance (REVIEW 11.6): margin U1 M = {acc['U1_margin'] * E4:.2f}e-4 (share {acc['share']}, f "
          f"{acc['f_short']:.6f}), U2: N_s <= {acc['U2_max']} of {acc['n_short']}.\n")
        w("| seed | Δ A11p - A11 | equal NLL | Δ^S (short) | Δ^S + 2 SE^S | U1 | N_s worse by > 1e-4 | U2 |\n"
          "|---|---|---|---|---|---|---|---|")
        for s_, v in acc["seeds"].items():
            w(f"| {s_} | {fmt(v['paired'])} | {yn(v['equal'])} | {fmt(v['short'])} | {v['U1_bound'] * E4:+.2f} | "
              f"{yn(v['U1'])} | {v['N_worse_by_1e-4']} | {yn(v['U2'])} |")
        w(f"\n**A11p: {acc['verdict']}.** " + (
            f"The coefficients of Z can be removed from A11 without hurting the short events (within the margins of "
            f"REVIEW 11.6)." if acc["accepted_without_hurting_short"] else
            "The coefficients of Z cannot be removed from A11 without hurting the short events (within the margins "
            "of REVIEW 11.6), or not at equal NLL.")
            + (f" Z_0 minus Z: {sc['Z0_minus_Z']}." if sc else "") + "\n")
    az = res.get("a11p_z0_acceptance")
    if az:
        w(f"Z_0 fits (A11pz, fitted because the parameter 'if_vetoed' asks for it; reported, not a finalist "
          f"candidate): " + "; ".join(f"seed {s_}: {fmt(v['paired'])}, equal {yn(v['equal'])}, U1 {yn(v['U1'])}, "
                                      f"U2 {yn(v['U2'])}" for s_, v in az["seeds"].items()) + f" -> {az['verdict']}\n")
    w("## Initialisation of the network runs (A11n)\n")
    w("| run | kl_map | reference | fit_to version / rms_max / std (column) | step-0 parity | below-floor target "
      "columns |\n|---|---|---|---|---|---|")
    for x in res.get("initialisation") or []:
        if x.get("provider") != "net":
            continue
        p0 = x.get("step0_parity") or {}
        rr = x.get("fit_to_rms_max_rel")
        tg = x.get("targets") or {}
        w(f"| {x['run']} | {x.get('kl_map')} | {x.get('kl_ref')} | v{x.get('fit_to_version')} / "
          f"{'-' if rr is None else f'{rr * 100:.2f} %'} ({x.get('fit_to_rms_max_name')}) | "
          f"{fmt(p0.get('paired')) if p0.get('available') else '-'} | "
          f"{', '.join(tg.get('below_scale_floor') or []) or 'none'} |")
    w("\n## Convergence (|val(last) - val(last - window)| < 0.5e-4)\n")
    for k, v in res["convergence"].items():
        if v:
            mv = "-" if v["moved"] is None else f"{v['moved'] * E4:.2f}e-4"
            w(f"- {k}: converged {yn(v['converged'])}, moved {mv}")
    w("\n## Guards of the Stage-2 runs (own tables)\n")
    if not res["guards"]:
        w("No guard data yet (the Stage-2 guards run inside the stage2 phase).\n")
    for key, g in res["guards"].items():
        w(f"### {key}\n")
        if g.get("missing"):
            w("**Guard data MISSING.**\n")
            continue
        out.extend(_s2_guard_text(g))
    w("## Conditional controls (user decision of 26 Nov 2026, PLAN 20 item 1)\n")
    ccd = res.get("conditional_controls")
    if not ccd:
        w("**Not evaluated** (`run_m2a.py conditional-controls` has not run).\n")
    else:
        for ctl, v in ccd.items():
            w(f"- **{ctl}**: {'TRIGGERED for seeds ' + str(v['seeds']) if v['triggered'] else '**not triggered**'} - "
              f"rule: {v['rule']}")
            for nme, x in v["evaluated_pairs"].items():
                w(f"  - {nme} vs {x['vs']}: {fmt(x['paired'])}; keep {yn(x['keep'])}, equal NLL (one seed) "
                  f"{yn(x['equal_one_seed'])}")
            if not v["evaluated_pairs"]:
                w("  - no registered pair of this kind on this machine")
        for nme, x in (res.get("conditional_controls_runs") or {}).items():
            w(f"- control run {nme} vs {x.get('vs')}: " + (fmt(x.get("paired")) if x.get("paired") else
                                                          str(x.get("status"))))
        w("")
    w("## Divergence definition and warnings for Gate 1 (PLAN 20 item 2; REVIEW 10.5 R3)\n")
    w(f"Present definition: {res['divergence_definition']}\n")
    rc = res["reclassification"]
    w(f"Re-classification of every finished run of this stage ({rc['finished_runs']}) from its stored fit history: "
      f"runs with a non-finite validation mean anywhere in the history: {len(rc['runs_with_a_non_finite_validation_mean'])}"
      f" {rc['runs_with_a_non_finite_validation_mean'] or ''}; classified differently by the present definition than "
      f"by that of 25 Nov 2026: **{rc['classified_differently'] or 'none'}**.\n")
    wg = res.get("warnings_gate1_stage1") or {}
    w("Recovered non-finite validation values (warnings for Gate 1), Stage-1 runs: " + (
        "; ".join(f"{k}: {v['count']} at steps {v['steps']}" for k, v in wg.items()) if wg else "none") + "\n")
    w("## Diverged, blocked and warnings for Gate 1 (Stage 2)\n")
    for nme, v in (res.get("diverged") or {}).items():
        w(f"- **{nme}** diverged: {'; '.join(v.get('criteria') or [])}")
    for nme, v in (res.get("blocked") or {}).items():
        w(f"- **{nme}** blocked by {v.get('blocked_by')}")
    for nme, v in (res.get("warnings_gate1") or {}).items():
        w(f"- **{nme}**: {v['count']} recovered non-finite validation mean(s) at steps {v['steps']} (warning)")
    if not (res.get("diverged") or res.get("blocked") or res.get("warnings_gate1")):
        w("None.")
    return "\n".join(out) + "\n"


def report_stage2() -> dict:
    """The Stage-2 report: results_stage2.json and RESULTS_stage2.md (results.json and RESULTS.md of Stage 1, the
    files the approval is bound to, are not touched)."""
    res = collect_stage2()
    od = L.out_dir()
    L.write_json(res, od / "results_stage2.json")
    f = paths.check_output_path(od / "RESULTS_stage2.md")
    tmp = f.with_name(f".{f.name}.tmp")
    tmp.write_text(paths.scrub_str(markdown_stage2(res)))
    tmp.replace(f)
    log(f"stage2 report: {paths.portable(f)} written ({len(res['comparisons'])} comparisons, {len(res['guards'])} "
        f"guard blocks, finalist {res['finalist'].get('finalist')})")
    return res


def compare_results(new: Path, approved: Path) -> int:
    """REVIEW 10.5 R9: the results.json re-issued at a new commit against the one the user approved; 0 if they differ
    in the key 'git' only, 1 otherwise (the differing keys are printed; a hard stop, PLAN 23 item 13)."""
    a, b = json.loads(Path(approved).read_text()), json.loads(Path(new).read_text())
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print(f"keys that differ: {diff}")
    if diff and set(diff) != {"git"}:
        print("STOP (PLAN 23 item 13): results.json differs from the approved one beyond the commit; waiting for the "
              "user")
        return 1
    print(f"OK: only {'git' if diff else 'nothing'} differs (approved commit {(a.get('git') or {}).get('commit')}, "
          f"new {(b.get('git') or {}).get('commit')})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--without-guards", action="store_true", help="tables without guard columns (review C2)")
    ap.add_argument("--compare", nargs=2, metavar=("NEW", "APPROVED"),
                    help="R9: exit 1 unless two results.json differ in 'git' only (reads the two files, writes nothing)")
    args = ap.parse_args(argv)
    if args.compare:
        return compare_results(Path(args.compare[0]), Path(args.compare[1]))
    L.configure(smoke=args.smoke)
    L.start("run_m2a_report")
    report(with_guards=not args.without_guards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
