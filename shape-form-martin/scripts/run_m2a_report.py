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
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_m2a_lib as L  # noqa: E402
from run_m2a_lib import log  # noqa: E402

from sfm import paths, provenance, registry, stats  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402

E4 = 1e4
P_SPLIT_GEV = 2.0
Z_SIDE_MM = 1450.0
A2S_CONSTANTS = 101


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
    seeds meets the threshold (-3e-4). The two-seed mean is the mean of the per-event differences (same events)."""
    if c1 is None:
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
    return {"seed1_half_threshold": m1 <= -0.5 * _m().KEEP_GAIN, "same_sign": (m1 < 0) == (c0["paired"]["mean"] < 0),
            "two_seed_mean": L.pd(both), "replicated": ok}


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
            "passed": bool(gain < 0 and margin.mean >= 0.0), "resolved": abs(margin.z) >= 2.0,
            "rule": "<= 50 % of the gain from deep, border or Abar <= 0.3 cells (hit term, by hit centre); "
                    "normalisation term reported separately"}


# ----------------------------------------------------------------------------------------------------------------------
def collect() -> dict:
    M = _m()
    dec = L.read_json(L.decisions_file(), {}) or {}
    res = {"stage": L.stage(), "smoke": L.CTX.smoke, "machine_id": L.MACHINE_ID, "stream_id": L.live_stream_id(),
           "units": "1e-4 nat per event (deltas); paired SE; n events", "decisions": dec,
           "plr": L.read_json(L.plr_file()), "refset": paths.portable(L.refset().dir),
           "refset_json_sha256": L.refset().manifest_sha256}
    masks = secondary_masks()
    res["secondary_strata_n"] = {k: int(v.sum()) for k, v in masks.items()}
    # Stage 0
    s0 = {}
    for a, b in (("A0_s0", "formula"), ("A0_s1", "formula"), ("A1_s0", "A0_s0"), ("A1_s1", "A0_s1"),
                 ("A2e_s0", "A0_s0"), ("A2e_s1", "A0_s1"), ("A2s_s0", "A2e_s0"), ("A2s_s1", "A2e_s1"),
                 ("A2n_s0", "A2s_s0"), ("A2n_s1", "A2s_s1"), ("A2n_s2", "A2s_s2"),
                 ("A2s_s1", "A2s_s0"), ("A2s_s2", "A2s_s0"), ("A2s_s2", "A2s_s1")):
        if have(a) and (b == "formula" or have(b)):
            s0[f"{a} - {b}"] = pdict(a, b)
    means = [float(np.mean(L.tagged(f"A2s_s{s}").values, dtype=np.float64)) for s in (0, 1, 2) if have(f"A2s_s{s}")]
    res["stage0"] = {"deltas": s0, "A2s_seed_sd_of_means": float(np.std(means, ddof=1)) if len(means) > 1 else None}
    # step-0 Delta of A9
    res["A9_step0"] = {f"A9_s{s}_init - A2s_s{s}": pdict(f"A9_s{s}_init", f"A2s_s{s}")
                       for s in (0, 1) if have(f"A9_s{s}_init") and have(f"A2s_s{s}")}
    # Stage 1 primaries (poly), the net column, exploratory
    prim, net, expl = {}, {}, {}
    for r in M.PRIMARIES:
        c = {s: comparison(f"{r}_s{s}", f"A2s_s{s}", masks) for s in (0, 1) if have(f"{r}_s{s}") and have(f"A2s_s{s}")}
        if not c:
            continue
        rep = replication(c[0], c.get(1), f"{r}_s0", "A2s_s0", f"{r}_s1", "A2s_s1") if 0 in c else None
        x2 = {s: comparison(f"{r}_s{s}_x2", f"A2s_s{s}_x2", masks) for s in (0, 1)
              if have(f"{r}_s{s}_x2") and have(f"A2s_s{s}_x2")}
        prim[r] = {"seeds": c, "replication": rep, "equal_both": (all(v["equal_one"] for v in c.values())
                                                                 if len(c) == 2 else None),
                   "convergence": {s: convergence(f"{r}_s{s}") for s in c}, "reruns_x2": x2}
        if have(f"{r}n_s0") and have("A2n_s0"):
            net[r + "n"] = comparison(f"{r}n_s0", "A2n_s0", masks)
            net[r + "n"]["convergence"] = convergence(f"{r}n_s0")
    for r in M.EXPLORATORY:
        c = {s: comparison(f"{r}_s{s}", f"A2s_s{s}", masks) for s in (0, 1) if have(f"{r}_s{s}") and have(f"A2s_s{s}")}
        if not c:
            continue
        rep = replication(c[0], c.get(1), f"{r}_s0", "A2s_s0", f"{r}_s1", "A2s_s1") if 0 in c else None
        expl[r] = {"seeds": c, "replication": rep,
                   "equal_both": (all(v["equal_one"] for v in c.values()) if len(c) == 2 else None),
                   "convergence": {s: convergence(f"{r}_s{s}") for s in c}}
    res["primaries"], res["net"], res["exploratory"] = prim, net, expl
    # cleaner columns and kept-and-replicated (input of Stage 2; no decision is taken here)
    kr = {}
    for grp in (prim, expl):
        for r, v in grp.items():
            c0 = v["seeds"].get(0)
            kr[r] = bool(c0 and c0["keep"] and v["replication"] and v["replication"]["replicated"])
            for s, c in v["seeds"].items():
                c["cleaner_by_gain"] = c["vs_formula"]["mean"] <= -M.KEEP_GAIN
                c["cleaner_by_constants"] = bool(c["k"] <= 0.9 * A2S_CONSTANTS and v["equal_both"])
    res["kept_and_replicated"] = kr
    # guards
    g = {}
    for run, comp in (dec.get("guard_targets") or {}).get("pairs", []):
        blk = guard_block(run, comp)
        if blk is not None:
            g[f"{run} vs {comp}"] = blk
    res["guards"] = g
    fisher = L.read_json(L.out_dir() / "fisher" / "A6b_s0_sh_rk.json")
    res["A6b_fisher_k"] = fisher
    # convergence of every finished run
    conv = {}
    for f in sorted(L.out_dir().glob("*/done.json")):
        conv[f.parent.name] = convergence(f.parent.name)
    res["convergence_all"] = conv
    return res


# ----------------------------------------------------------------------------------------------------------------------
def markdown(res: dict) -> str:
    out = []
    w = out.append
    title = "Track A (M2A) - SMOKE RUN (shortened fits; not numbers of record)" if res["smoke"] else \
        "Track A (M2A) - results on the machine of record"
    w(f"# {title}\n")
    w(f"Stage `{res['stage']}`, machine `{res['machine_id']}`, stream `{res['stream_id']}`. Reference set "
      f"`{res['refset']}` (refset.json SHA-256 `{res['refset_json_sha256'][:16]}...`). Validation only. Every delta: "
      "mean ± paired SE (n events), unit 1e-4 nat per event, through `sfm.provenance.paired_delta`. Report-metric "
      "tables (K = 64) and guard tables are separate. Nulls and negative results as measured.\n")
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
          + f" -> {'STOP' if dr['stop'] else 'continue'}")
    if res.get("plr"):
        p = res["plr"]
        w(f"- P-lr: lr_poly = {p.get('lr_poly')}, lr_net = {p.get('lr_net')}; " + "; ".join(
            f"{grp} lr {lr}: {fmt(v['vs_lowest'])} vs lowest" for grp in ("poly", "net") for lr, v in
            (p.get(grp) or {}).items()))
    w(f"- Dropped runs: {dec.get('dropped', [])} ({dec.get('budget_rule', 'budget rule: not applied')})\n")
    w("## Stage 0 (report metric)\n")
    w("| comparison | Δ |\n|---|---|")
    for k, v in res["stage0"]["deltas"].items():
        w(f"| {k} | {fmt(v)} |")
    sd = res["stage0"]["A2s_seed_sd_of_means"]
    w(f"\nA2s seed scatter (sd of the validation means of the seeds): {sd * E4:.2f}e-4\n" if sd is not None else "")
    if res["A9_step0"]:
        w("A9 at step 0 (logit_dh = -5): " + "; ".join(f"{k} {fmt(v)}" for k, v in res["A9_step0"].items()) + "\n")
    w("## Stage 1 - primaries, poly (pre-registered comparisons, vs A2s of the same seed; report metric)\n")
    w("| rung | k | Δ seed 0 | keep s0 | Δ seed 1 | keep s1 | replicated (two-seed mean) | equal NLL (both) | "
      "Δ vs formula s0 | converged s0/s1 |\n|---|---|---|---|---|---|---|---|---|---|")
    for r, v in res["primaries"].items():
        c0, c1 = v["seeds"].get(0), v["seeds"].get(1)
        rep = v["replication"]
        cv = v["convergence"]
        w(f"| {r} | {(c0 or c1)['k']} | {fmt(c0 and c0['paired'])} | {yn(c0 and c0['keep'])} | "
          f"{fmt(c1 and c1['paired'])} | {yn(c1 and c1['keep'])} | "
          f"{yn(rep and rep['replicated'])} ({fmt(rep and rep['two_seed_mean'])}) | {yn(v['equal_both'])} | "
          f"{fmt(c0 and c0['vs_formula'])} | {'/'.join(yn((cv.get(s) or {}).get('converged')) for s in (0, 1))} |")
        for s, x in v["reruns_x2"].items():
            w(f"| {r} rerun s{s} (doubled steps) | {x['k']} | {fmt(x['paired']) if s == 0 else ''} | "
              f"{yn(x['keep']) if s == 0 else ''} | {fmt(x['paired']) if s == 1 else ''} | "
              f"{yn(x['keep']) if s == 1 else ''} | | | | |")
    w("\n## Stage 1 - net column (exploratory; vs A2n seed 0)\n")
    w("| run | k | Δ | keep | equal NLL (seed 0) | converged |\n|---|---|---|---|---|---|")
    for r, c in res["net"].items():
        w(f"| {r} | {c['k']} | {fmt(c['paired'])} | {yn(c['keep'])} | {yn(c['equal_one'])} | "
          f"{yn((c.get('convergence') or {}).get('converged'))} |")
    w("\n## Stage 1 - exploratory (vs A2s of the same seed)\n")
    w("| rung | k | Δ seed 0 | keep s0 | equal s0 | Δ seed 1 | replicated | equal NLL (both) | cleaner (gain / "
      "constants) |\n|---|---|---|---|---|---|---|---|---|")
    for r, v in res["exploratory"].items():
        c0, c1 = v["seeds"].get(0), v["seeds"].get(1)
        rep = v["replication"]
        w(f"| {r} | {(c0 or c1)['k']} | {fmt(c0 and c0['paired'])} | {yn(c0 and c0['keep'])} | "
          f"{yn(c0 and c0['equal_one'])} | {fmt(c1 and c1['paired'])} | {yn(rep and rep['replicated'])} | "
          f"{yn(v['equal_both'])} | {yn(c0 and c0['cleaner_by_gain'])} / {yn(c0 and c0['cleaner_by_constants'])} |")
    w("\n## Secondary columns (pre-registered, never selection criteria; report metric on strata)\n")
    n = res["secondary_strata_n"]
    w("| run vs comparator | " + " | ".join(f"{k} (n {n[k]})" for k in n) + " |\n|---|" + "---|" * len(n))
    for grp in ("primaries", "exploratory"):
        for r, v in res[grp].items():
            for s, c in v["seeds"].items():
                w(f"| {c['run']} vs {c['vs']} | " + " | ".join(fmt(c["secondary"][k]) for k in n) + " |")
    for r, c in res["net"].items():
        w(f"| {c['run']} vs {c['vs']} | " + " | ".join(fmt(c["secondary"][k]) for k in n) + " |")
    w("\n## Guards (own tables; K = 1024, bulk stratum, mass monitors, G-region)\n")
    if not res["guards"]:
        w("No run passed the keep or the equal-NLL rule: no guard columns.\n")
    for key, g in res["guards"].items():
        w(f"### {key}\n")
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
              f"{yn(gr['resolved'])}; decomposition residual {gr['decomposition_residual_max_abs']:.1e}\n")
        w("| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |\n|---|---|---|---|")
        for k, v in g["G-mass"].items():
            w(f"| {k} | {v['run_mean']:.4e} | {v['comp_mean']:.4e} | {fmt(v['delta'])} |")
        w("")
    if res.get("A6b_fisher_k"):
        f = res["A6b_fisher_k"]
        w(f"## A6b: Fisher information of k\n\nObserved information of the raw coefficient of `sh_rk` on the "
          f"training events: {f['observed_information']:.4g} (SE of the coefficient {f['se_coefficient']}).\n")
    w("## Convergence (|val(last) - val(last - window)| < 0.5e-4)\n")
    w("| run | converged | moved |\n|---|---|---|")
    for k, v in res["convergence_all"].items():
        if v:
            moved = "-" if v["moved"] is None else f"{v['moved'] * E4:.2f}e-4"
            w(f"| {k} | {yn(v['converged'])} | {moved} |")
    w("\n## Kept and replicated (the input of Stage 2; no combination formed here)\n")
    w(", ".join(f"{r}: {yn(v)}" for r, v in res["kept_and_replicated"].items()) + "\n")
    return "\n".join(out) + "\n"


def report() -> dict:
    res = collect()
    od = L.out_dir()
    L.write_json(res, od / "results.json")
    md = paths.scrub_str(markdown(res))
    f = paths.check_output_path(od / "RESULTS.md")
    tmp = f.with_name(".RESULTS.md.tmp")
    tmp.write_text(md)
    tmp.replace(f)
    log(f"report: {paths.portable(f)} and results.json written ({len(res['primaries'])} primaries, "
        f"{len(res['exploratory'])} exploratory rungs, {len(res['guards'])} guard blocks)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    L.configure(smoke=args.smoke)
    L.start("run_m2a_report")
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
