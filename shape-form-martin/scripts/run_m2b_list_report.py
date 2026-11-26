"""Report of the Track B list (scripts/run_m2b_list.py): the pre-registered rules of AUDIT.md section 4.0 / 4.2 and
PLAN.md section 6 applied MECHANICALLY to the registered arrays of the live stream. No training, no test data.

    .venv/bin/python -B shape-form-martin/scripts/run_m2b_list_report.py [--smoke]

Writes <machine dir>/runs/<stage>/LIST_RESULTS.md and list_results.json. Units in the tables: 1e-4 nat per event,
paired, +- SE over the 14,950 validation events. Every delta goes through sfm.provenance.paired_delta on the live
stream (tags checked); untagged laptop rows are never read. Exit 3 with a "HARD STOP" last line when a check fails
(the scoring check below), 0 otherwise.

Rules: the pre-registered text (PLAN 6, AUDIT 4.0 / 4.2) in the readings RULED by the second reviewer
(results/m2b_mlp/REVIEW.md section 9.1). One function per ruling (RULINGS lists them); [confirm] items carry a
constant that the user's answer changes (I11_READING, I4_OPTION) and nothing else.
    promotion of a MODEL, against mlp_std only (I11, statement P), part by part:
        (a) Delta vs mlp_std <= -5            (b) Delta vs mlp_std <= -3 SE
        (c) I12: d_e = run(e) - (1/3) sum_k b0h_seed-k(e); Delta_B0 = mean(d), SE = sd(d, ddof 1)/sqrt(n); (c) iff
            Delta_B0 / SE < -1; formed only when all three controls are registered ("B0 incomplete" otherwise).
            Printed beside it as description: Delta against each single control and the seed sd of the controls.
        (d) I13, the literal reading ("the threshold" = the Delta part, in both clauses): seed-1 Delta <= -2.5 with
            the sign of seed 0, and the two-seed mean (per-event mean of the two runs against mlp_std) <= -5. Labelled
            alternative column (decides nothing): the same and the two-seed mean <= -3 SE; where the two disagree on a
            promotion row or a primary comparison, the row goes to the user. Seed 0 is the result, seed 1 the
            replicate; a row that passes on seed 1 only is printed and not promoted. B1d: first start = result,
            second start = replicate, labelled "replicated over starting networks (one teacher)".
        (e) J-a / J-b, three values per guard: passed (passed and resolved at 2 SE), failed (failed and resolved),
            undecided (not resolved). G-K / G-bulk: the registry's persistence decision (margin = mean(guard -
            0.5 main) <= 0 with gains negative, plus its sign checks). G-region: main < 0 and mean(flagged - 0.5
            main) >= 0 (at most half of the gain from deep / border / Abar <= 0.3 cells; the opposite sign to G-K
            because here at most half may come from the flagged cells), resolved iff |z| >= 2; the normalisation
            term and the decomposition residual are printed. (e) = failed if a guard failed, passed if all passed,
            undecided otherwise.
        promotion: "promoted" iff (a), (b), (c), (d) and (e) passed. (a)-(d) with (e) undecided: "undecided -
            Gate 1" (neither promoted nor rejected). A B1d run meeting (a)-(c) while its teacher E5 does not:
            "anomaly - Gate 1" (not promoted automatically). Evaluated for the primaries' runs of record only; every
            other row prints "n/a (<class>)" (review B3) and the multiplicity sentence.
    primary comparisons (I11, statement Q; option (i) as ruled, NEEDS THE USER'S CONFIRMATION): B1d vs mlp_std;
        B2b - B2a, B2c - B2a, B4f - B2a, B6c - B6a, same seed. A comparison counts iff seed-0 Delta <= -5 and
        <= -3 SE and the replication rule (I13) holds on the comparison. Never combined with promotion: a promoted
        model whose comparison does not count is promoted as a model and "the mechanism is not shown". Printed
        beside the verdict, deciding nothing: option (iii) (-3 and -2 SE, half -1.5, two-seed mean -3; the Track A
        keep rule) and option (ii) (descriptive: Delta +- SE per seed and the two-seed mean, no verdict); a neutral
        note "between -3 and -5" where it applies.
    1-SE rule (I17): per seed, sfm.stats.one_se_select over B6a, B6b, B6c (pairs of record); a capacity is eligible
        if eligible on both seeds; the selection is the eligible capacity with the fewest weights; if none is
        eligible on both seeds: "no automatic selection - to Gate 1".
    convergence reruns (review B2): where <name>_x2 exists it is the run of record, with its comparator's rerun, for
        (a)-(e), the primary comparisons, replication, the 1-SE rule and the guard candidates; the original is printed
        as "not converged, NOT used by the rules"; a column names the pairs used when the seeds mix an original and a
        rerun.
    B1d convergence (I4, [confirm]): option (alpha) implemented (no convergence statement; the KL trajectory as
        logged - initial, final and the five intermediate lines - is printed); option (beta) prepared (companions
        with 12,000 KL steps, phase b1d_companions of the driver; the 6,000-step run is of record iff
        |Delta(12000 - 6000)| < 0.5, else the 12,000-step run) and applied only with I4_OPTION = "beta".
    B1 / B1d: B1d - E5 (paired) for both starts (the distillation loss) in place of a retained fraction.
    scoring check (A2 (i), boundary case): for every finished run the fit's last validation mean (main and EMA
        stream) must equal the registry's mean of the same weights to 1e-6 (run_m2b_list.scoring_check); otherwise a
        hard stop before any table is written.
    B3: EMA stream minus last iterate of the same run. B-ceil: the registered last iterate, and the minimum of the
        validation trajectory (K = 64 means at every 100 steps; selected on validation and therefore optimistic).
    secondary columns (AUDIT 4.0, never selection criteria): p < 2 GeV / p >= 2 GeV; contained / side-leaking events.
Nulls and negative results are printed as measured. Report-metric and bulk-restricted (guard) numbers are in separate
tables.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2b_lib as lib  # noqa: E402
import run_m2b_list as D  # noqa: E402

import numpy as np  # noqa: E402

from sfm import machine, provenance, registry, stats  # noqa: E402

E4 = 1e4
ALT_GAIN, ALT_Z = -3e-4, -2.0                  # I11 option (iii): the Track A keep rule (-3, -2 SE; half -1.5)
RESOLVED_Z = 2.0                               # guards: "called resolved only at 2 SE" (PLAN 13)
B1D_CONV_TOL = 0.5e-4                          # I4 option (beta)
I11_READING = "i"                              # [confirm] the ruled option; the user's answer changes only this
I4_OPTION = "alpha"                            # [confirm] "alpha" (implemented) | "beta" (prepared)
LIST_PRIMARY_RUNGS = ("B1d", "B2b", "B2c", "B4f", "B6c")
COMPARISONS = (("B1d", "mlp_std"), ("B2b", "B2a"), ("B2c", "B2a"), ("B4f", "B2a"), ("B6c", "B6a"))
B1D_REPLICATION_LABEL = "replicated over starting networks (one teacher)"
B1D_REPLICATION_NOTE = ("both B1d runs distil the same teacher (E5, one realisation) on the same grid; part (d) shows "
                        "that the result does not depend on the start, it does not replicate E5")
MULTIPLICITY_NOTE = ("Multiplicity: the six pre-registered primaries (B1, B1d, B2b - B2a, B2c - B2a, B4f - B2a, "
                     "B6c - B6a) are the basis of the multiple-testing accounting (A6 revised, point 6); every other "
                     "row is exploratory, a comparator or a diagnostic and is reported as measured, not as a test "
                     "(AUDIT 4.0).")

RULINGS = {   # results/m2b_mlp/REVIEW.md section 9.1 (Fable); [within] = follows from the text, [confirm] = the user
    "I11": {"status": "ruled: option (i), two separate statements [confirm - needs the user's confirmation]",
            "function": "rule_primary_comparison (constant I11_READING)",
            "text": "promotion of a model against mlp_std only; a primary comparison counts iff seed-0 Delta <= -5 and "
                    "<= -3 SE with replication on the comparison; options (iii) and (ii) printed, deciding nothing"},
    "I12": {"status": "ruled [within]", "function": "rule_better_than_b0",
            "text": "paired SE against the per-event mean of the three controls, no between-seed term; Delta against "
                    "each control and the controls' seed sd printed as description"},
    "I13": {"status": "ruled [within]", "function": "rule_replication",
            "text": "the Delta part only in both clauses: seed 1 <= -2.5 with the same sign, two-seed mean <= -5; the "
                    "version with <= -3 SE is a labelled alternative; a disagreement goes to the user"},
    "I17": {"status": "ruled [within]", "function": "rule_one_se",
            "text": "per seed, then eligibility on both seeds; fewest weights among the eligible; none: to Gate 1"},
    "J-a": {"status": "ruled [within]", "function": "rule_guard_part",
            "text": "three values; unresolved = undecided (not promoted automatically, not rejected, Gate 1)"},
    "J-b": {"status": "ruled [within]", "function": "rule_g_region",
            "text": "passed iff main < 0 and mean(flagged - 0.5 main) >= 0; resolved at 2 SE; three values"},
    "B1d": {"status": "ruled [within]", "function": "b1d_anomaly, B1D_REPLICATION_LABEL",
            "text": "(d) on the two starts, labelled; B1d - E5 printed; a B1d run meeting (a)-(c) while E5 does not "
                    "is an anomaly for Gate 1"},
    "I4": {"status": "with the user [confirm]; option (alpha) implemented, option (beta) prepared",
           "function": "rule_b1d_convergence (constant I4_OPTION)",
           "text": "(alpha): no convergence statement for B1d, KL trajectory printed; (beta): companions with 12,000 "
                   "KL steps (driver phase b1d_companions, not in the default list), the 6,000-step run of record iff "
                   "|Delta| < 0.5"},
}


def fmt(p) -> str:
    return "-" if not p or p.get("mean") is None else f"{p['mean'] * E4:+.2f} ± {p['se'] * E4:.2f}"


def fmtz(p) -> str:
    return "-" if not p or p.get("mean") is None else f"{p['mean'] * E4:+.2f} ± {p['se'] * E4:.2f} (z {p['z']:+.2f})"


def _num(v) -> str:
    return "-" if v is None else (f"{v:.4g}" if isinstance(v, (int, float)) else str(v))


def pd(p: stats.Paired) -> dict:
    return {"mean": p.mean, "se": p.se, "z": p.z, "n": p.n}


def tagged(name: str, stage: str | None = None, kind: str = "k64"):
    r = D.row(name, stage=stage, need_guards=(kind != "k64"))
    return None if r is None else registry.row_array(r, kind=kind)


def mlp_array():
    return registry.reference_set().bench["mlp"]


def _mean_array(arrs, label: str):
    provenance.check_pairable(*arrs)
    t = arrs[0]
    return provenance.TaggedArray.make(np.mean([a.values.astype(np.float64) for a in arrs], axis=0), t.points,
                                       machine.MachineTag.from_fields(t.machine), label=label)


def control_arrays():
    """The three H200 controls b0h_seed0..2 (stage m2b_mlp) as {name: array}; None unless all three are registered."""
    arrs = {n: tagged(n, stage=D.REAL_STAGE) for n in lib.CONTROL_NAMES}
    if any(a is None for a in arrs.values()):
        return None
    provenance.check_pairable(*arrs.values())
    return arrs


def b0_mean(ctrls=None):
    """Per-event mean of exactly the three H200 controls; None if not all three are registered."""
    ctrls = control_arrays() if ctrls is None else ctrls
    return None if not ctrls else _mean_array(list(ctrls.values()), "B0 mean (b0h_seed0..2)")


def comparator_array(name: str):
    return mlp_array() if name == "mlp_std" else tagged(name)


def secondary_masks() -> dict:
    """AUDIT 4.0 secondary strata: p < 2 GeV / p >= 2 GeV; contained / side-leaking (axis leaves a side face before
    z' = 1450 mm; sfm.models.dist_factory.exit_depth, the port of checks/c_side.py used by Track A)."""
    from sfm.models import dist_factory as dfac
    sets, extras = D.data_cpu()
    v = sets["val"]
    p = v.ev["p"].numpy().astype(np.float64)
    zb = v.ev["zb"].numpy().astype(np.float64)
    th, ph = (np.asarray(extras["val"][k], dtype=np.float64) for k in ("theta", "phi"))
    z_exit, _, _ = dfac.exit_depth(th, ph, dfac.Z_BACK - zb * np.cos(th))
    side = z_exit < 1450.0
    return {"p<2GeV": p < 2.0, "p>=2GeV": p >= 2.0, "contained": ~side, "side-leaking": side}


# ======================================================================================================================
# The ruled rules (REVIEW.md 9.1), one function each
# ======================================================================================================================
def between_note(mean) -> str | None:
    """I11 report requirement: a neutral note, not a second pass / fail."""
    return "between −3 and −5" if mean is not None and D.PROMO_GAIN < mean <= ALT_GAIN else None


def rule_replication(p0: dict | None, p1: dict | None, both: dict | None, gain: float = D.PROMO_GAIN,
                     z: float = D.PROMO_Z) -> dict | None:
    """I13 (ruled, [within]): AUDIT 4.0 "the second seed gives Delta <= half the threshold with the same sign and the
    mean of the two seeds meets the threshold", "the threshold" = the Delta part (``gain``) in both clauses. The
    alternative (decides nothing) adds two-seed mean <= ``z`` SE. p0 = seed 0 (the result), p1 = seed 1."""
    if p0 is None or p1 is None or both is None:
        return None
    m0, m1 = p0["mean"], p1["mean"]
    half = bool(m1 <= 0.5 * gain)
    same = bool((m1 < 0) == (m0 < 0))
    mean_ok = bool(both["mean"] <= gain)
    lit = half and same and mean_ok
    alt = bool(lit and both["z"] <= z)
    return {"seed1_half_threshold": half, "same_sign": same, "two_seed_mean_meets_threshold": mean_ok,
            "two_seed_mean": both, "replicated": lit, "replicated_alternative_se": alt,
            "alternative": f"the literal conditions and the two-seed mean <= {z:g} SE (decides nothing)",
            "disagree": bool(lit != alt), "constants": {"half": 0.5 * gain, "mean": gain, "alternative_z": z}}


replication = rule_replication                     # former name


def rule_primary_comparison(p0: dict | None, p1: dict | None, both: dict | None) -> dict:
    """I11 (ruled: option (i); [confirm]): statement Q, the primary comparison of a mechanism against its same-seed
    comparator. Counts iff seed-0 Delta <= -5 and <= -3 SE and replication (I13) on the comparison. Options (iii)
    and (ii) are computed beside it and decide nothing; ``I11_READING`` names the verdict."""
    if p0 is None:
        return {"available": False}
    rep_i = rule_replication(p0, p1, both, D.PROMO_GAIN, D.PROMO_Z)
    rep_iii = rule_replication(p0, p1, both, ALT_GAIN, ALT_Z)
    i = bool(p0["mean"] <= D.PROMO_GAIN and p0["z"] <= D.PROMO_Z and rep_i is not None and rep_i["replicated"])
    iii = bool(p0["mean"] <= ALT_GAIN and p0["z"] <= ALT_Z and rep_iii is not None and rep_iii["replicated"])
    readings = {"i": i, "ii": None, "iii": iii}
    return {"available": True, "reading": I11_READING, "counts": readings[I11_READING], "option_i": i,
            "option_iii": iii, "option_ii": "descriptive (Delta ± SE, no verdict)", "replication": rep_i,
            "replication_option_iii": rep_iii, "replication_disagree": bool(rep_i is not None and rep_i["disagree"])}


def rule_better_than_b0(q: stats.Paired) -> bool:
    """I12 (ruled, [within]): (c) iff Delta_B0 / SE < -1, paired against the per-event mean of the three controls."""
    return bool(q.z < D.B0_Z)


def controls_seed_sd(ctrls, ref) -> float | None:
    """I12 description: the seed sd (ddof 1) of the three controls' means against mlp_std."""
    if not ctrls:
        return None
    return float(np.std([provenance.paired_delta(a, ref).mean for a in ctrls.values()], ddof=1))


def rule_one_se(per_seed: dict) -> dict:
    """I17 (ruled, [within]): per seed sfm.stats.one_se_select over the capacities; a capacity is eligible if it is
    eligible on both seeds; the selection is the eligible capacity with the fewest weights; none: to Gate 1.
    per_seed: {seed: [(capacity, run, k, per-event array or None), ...]} in capacity order."""
    out = {"per_seed": {}, "eligible_on_both_seeds": None, "combined": None, "per_seed_selections_differ": None}
    for seed, cands in per_seed.items():
        runs = {cap: run for cap, run, _, _ in cands}
        missing = [run for _, run, _, a in cands if a is None]
        if missing:
            out["per_seed"][seed] = {"runs": runs, "not_evaluated": missing}
            continue
        sel = stats.one_se_select([stats.Candidate(name=cap, nll=np.asarray(a), k=int(k)) for cap, _, k, a in cands])
        out["per_seed"][seed] = {"runs": runs, "best": sel.best, "selected": sel.selected,
                                 "eligible": [t["name"] for t in sel.table if t["within_1se"]], "table": sel.table}
    done = [v for v in out["per_seed"].values() if "eligible" in v]
    if not per_seed or len(done) != len(per_seed):
        out["statement"] = "not evaluated (a run of record is missing or diverged)"
        return out
    first = next(iter(per_seed.values()))
    caps = [c for c, _, _, _ in first]
    ks = {c: int(k) for c, _, k, _ in first}
    both = [c for c in caps if all(c in v["eligible"] for v in done)]
    out["eligible_on_both_seeds"] = both
    out["per_seed_selections_differ"] = len({v["selected"] for v in done}) > 1
    if both:
        out["combined"] = min(both, key=lambda c: (ks[c], caps.index(c)))
        out["statement"] = f"selected {out['combined']}" + (
            " (the per-seed selections differ; the combined selection is the eligible capacity with the fewest "
            "weights)" if out["per_seed_selections_differ"] else "")
    else:
        out["statement"] = "no automatic selection - to Gate 1 (no capacity is eligible on both seeds)"
    return out


def rule_guard_part(passed: bool, resolved: bool) -> str:
    """J-a (ruled, [within]): passed and resolved = passed; failed and resolved = failed; unresolved = undecided."""
    if not resolved:
        return "undecided"
    return "passed" if passed else "failed"


def rule_g_region(gain: float, margin: stats.Paired) -> dict:
    """J-b (ruled, [within]): passed iff main < 0 and mean(flagged - 0.5 main) >= 0 (gains negative: share of the gain
    in flagged cells <= 1/2 <=> flagged >= 0.5 main <=> margin >= 0); resolved iff |z| >= 2; three values."""
    passed = bool(gain < 0 and margin.mean >= 0.0)
    resolved = bool(abs(margin.z) >= RESOLVED_Z)
    return {"passed": passed, "resolved": resolved, "verdict": rule_guard_part(passed, resolved)}


def combine_guards(verdicts) -> str:
    """Part (e) from the three guards: failed if one failed, passed if all passed, undecided otherwise (not evaluated
    when a guard is missing and none failed)."""
    v = list(verdicts)
    if "failed" in v:
        return "failed"
    if any(x is None for x in v):
        return "not evaluated"
    return "passed" if all(x == "passed" for x in v) else "undecided"


def meets_abc(p: dict | None) -> bool:
    return bool(p and p.get("a_gain") and p.get("b_z") and p.get("c_b0"))


def b1d_anomaly(run_parts: dict | None, e5_parts: dict | None) -> bool:
    """Ruling B1d: a B1d run meeting (a)-(c) while its teacher E5 does not (the student would beat the teacher by
    more than 4): listed for Gate 1 as an anomaly, not promoted automatically."""
    return meets_abc(run_parts) and not meets_abc(e5_parts)


def promotion(e: dict, rep: dict | None, e_verdict: str | None, anomaly: bool) -> str:
    """Promotion of a model (statement P): (a)-(d) and (e) passed; J-a for an undecided guard; ruling B1d."""
    if not (meets_abc(e) and rep and rep["replicated"]):
        return "not promoted"
    if anomaly:
        return "anomaly - Gate 1 (meets (a)-(d); the teacher E5 does not meet (a)-(c); not promoted automatically)"
    if e_verdict == "passed":
        return "promoted"
    if e_verdict == "failed":
        return "not promoted (guard failed)"
    return f"undecided - Gate 1 (meets (a)-(d), guard {e_verdict or 'not evaluated'})"


def rule_b1d_convergence(a6000, a12000) -> dict:
    """I4 option (beta) (prepared; applied only with I4_OPTION = "beta"): the 6,000-step run is of record iff
    |Delta(12,000 - 6,000)| < 0.5e-4 on the registered arrays, else the 12,000-step run."""
    d = provenance.paired_delta(a12000, a6000)
    return {"delta_12000_minus_6000": pd(d), "of_record": "6000" if abs(d.mean) < B1D_CONV_TOL else "12000"}


# ----------------------------------------------------------------------------------------------------------------------
def parts(a, ref_mlp, b0, ctrls=None) -> dict:
    p = provenance.paired_delta(a, ref_mlp)
    out = {"vs_mlp": pd(p), "a_gain": p.mean <= D.PROMO_GAIN, "b_z": p.z <= D.PROMO_Z}
    if b0 is None:
        out["vs_b0"], out["c_b0"], out["vs_controls"] = None, None, None
    else:
        q = provenance.paired_delta(a, b0)
        out["vs_b0"], out["c_b0"] = pd(q), rule_better_than_b0(q)
        out["vs_controls"] = ({n: pd(provenance.paired_delta(a, c)) for n, c in ctrls.items()} if ctrls else None)
    return out


def two_seed(a0, a1, ref) -> dict:
    provenance.check_pairable(a0, a1, ref)
    return pd(provenance.paired_delta(_mean_array([a0, a1], "two-seed mean"), ref))


def guard_decision(persistence: dict) -> dict:
    """Review B1 / J-a: the persistence decision of sfm.stats.guard_persistence as the registry stored it. Its margin
    is mean(guard - 0.5 main) with gains negative: the guard keeps at least half the gain iff the margin is <= 0, which
    is what ``passed`` says (with its sign checks); "resolved" (|z of the margin| >= 2) is its own flag."""
    passed = bool(persistence.get("passed"))
    resolved = bool((persistence.get("flags") or {}).get("resolved"))
    return {"passed": passed, "resolved": resolved, "margin": (persistence.get("numbers") or {}).get("margin"),
            "verdict": rule_guard_part(passed, resolved)}


def guards(name: str, main_vs_mlp: np.ndarray | None) -> dict | None:
    r = D.row(name, need_guards=True)
    if r is None:
        return None
    g = r["guards"]["vs_mlp"]
    out = {"gk": g["gk"], "gk_persistence": g["gk_persistence"], "bulk": g["bulk"],
           "bulk_persistence": g["bulk_persistence"], "report_on_stratum": g["report_on_stratum"],
           "report_on_short": g["report_on_short"], "mass": r["guards"]["mass"]}
    out["gk_decision"] = guard_decision(g["gk_persistence"])
    out["bulk_decision"] = guard_decision(g["bulk_persistence"])
    out["g_region"] = g_region(name, main_vs_mlp)
    out["e"] = combine_guards([out["gk_decision"]["verdict"], out["bulk_decision"]["verdict"],
                               out["g_region"]["verdict"] if out["g_region"] else None])
    return out


def _gr(dirname: str, cls: str):
    d = D.out_dir() / dirname / "g_region"
    mk = D.read_json(d / "g_region.json")
    if mk is None:
        return None
    ta = provenance.load_tagged(d / f"g_region_{cls}.npy", expect_machine_id=lib.MACHINE.id)
    if mk["arrays"][cls]["sha256"] != ta.sha256:
        raise D.HardStop("guard", f"{dirname}: G-region array {cls} differs from its marker")
    return ta


def _recon_residual(dirname: str, registered) -> float:
    recon = sum(_gr(dirname, c).values.astype(np.float64) for c in ("flagged", "inner", "normalisation"))
    return float(np.abs(recon - registered.values.astype(np.float64)).max())


def g_region(name: str, main) -> dict | None:
    ref = "_mlp_std_reference"
    if main is None or _gr(name, "flagged") is None or _gr(ref, "flagged") is None:
        return None
    diffs = {c: provenance.paired_difference(_gr(name, c), _gr(ref, c))
             for c in ("deep", "border_ring", "border_zbin", "low_abar", "flagged", "inner", "normalisation")}
    gain = float(main.mean())
    margin = stats.paired(diffs["flagged"] - 0.5 * main, np.zeros_like(main))
    v = rule_g_region(gain, margin)
    return {"hit_term_by_class": {c: pd(stats.paired(d, np.zeros_like(d))) for c, d in diffs.items()
                                  if c != "normalisation"},
            "normalisation_term": pd(stats.paired(diffs["normalisation"], np.zeros_like(main))),
            "flagged_share_of_gain": (float(diffs["flagged"].mean()) / gain) if gain else None,
            "margin": pd(margin), **v,
            "decomposition_residual": {"run": _recon_residual(name, tagged(name)),
                                       "mlp_std": _recon_residual(ref, mlp_array()),
                                       "driver_check": (D.read_json(D.out_dir() / name / "g_region" / "g_region.json")
                                                        or {}).get("identity")}}


# ----------------------------------------------------------------------------------------------------------------------
# runs of record: convergence reruns (review B2) and the I4 option
# ----------------------------------------------------------------------------------------------------------------------
def all_specs() -> dict:
    """The list plus the convergence reruns (<name>_x2) and the I4 companions that ran on the live stream."""
    sm = dict(D.spec_map())
    for s in list(sm.values()):
        if s.cls in ("primary", "comparator") and not s.data_free:
            x = D.x2_spec(s)
            if D.row(x.name) is not None or D.diverged_marker(x.name) is not None:
                sm[x.name] = x
    for n, s in D.companion_map().items():
        if D.row(n) is not None or D.diverged_marker(n) is not None:
            sm[n] = s
    return sm


def _seed_runs(rung: str) -> list:
    """The original runs of a rung in seed (or start) order."""
    return [n for n, s in D.spec_map().items() if s.rung == rung]


def record_name(name: str, sm: dict) -> str:
    return f"{name}_x2" if f"{name}_x2" in sm else name


def pair_of_record(primary: str, comparator: str | None, sm: dict) -> tuple:
    """(run, comparator, rerun?): a primary that was rerun is represented by its rerun together with its comparator's
    rerun (AUDIT 4.0: "rerun once with doubled steps together with its comparator"); otherwise the originals."""
    x = f"{primary}_x2"
    if x in sm:
        cx = comparator if comparator in (None, "mlp_std") else f"{comparator}_x2"
        return x, (cx if cx in (None, "mlp_std") or cx in sm else comparator), True
    return primary, comparator, False


def b1d_record(name: str, sm: dict) -> tuple:
    """(run of record, I4 record) of a B1d run: under option (alpha) the 6,000-step run; under (beta) the rule."""
    comp = f"{name}{D.COMPANION_SUFFIX}"
    if I4_OPTION != "beta":
        return name, None
    a6, a12 = tagged(name), (tagged(comp) if comp in sm else None)
    if a6 is None or a12 is None:
        return name, {"note": "I4 (beta): the companion is not registered; the 6,000-step run is used"}
    r = rule_b1d_convergence(a6, a12)
    return (name if r["of_record"] == "6000" else comp), r


def records(sm: dict) -> dict:
    """The runs of record of the primary rungs of the list, in seed (start) order."""
    out = {}
    for rung in LIST_PRIMARY_RUNGS:
        out[rung] = ([b1d_record(n, sm)[0] for n in _seed_runs(rung)] if rung == "B1d"
                     else [record_name(n, sm) for n in _seed_runs(rung)])
    return out


def comparison_pairs(sm: dict, rec: dict) -> dict:
    """{rung: [(original, run, comparator, rerun?), ...]} - the pairs of record of the primary comparisons."""
    out = {}
    for rung, comp in COMPARISONS:
        rows = []
        for i, n in enumerate(_seed_runs(rung)):
            if rung == "B1d":
                rows.append((n, rec["B1d"][i], "mlp_std", False))
            else:
                run, cn, rerun = pair_of_record(n, f"{comp}_s{sm[n].seed}", sm)
                rows.append((n, run, cn, rerun))
        out[rung] = rows
    return out


def labels(sm: dict, rec: dict, pairs: dict) -> dict:
    """{name: label} for the rows that no rule uses (or that a rule uses only in part)."""
    used_comp = {}
    for rung, rows in pairs.items():
        for _, run, cn, _ in rows:
            used_comp.setdefault(cn, []).append(run)
    of_record = {x for v in rec.values() for x in v}
    out = {}
    for n, s in sm.items():
        if f"{n}_x2" in sm:
            if s.cls == "primary":
                out[n] = f"not converged, NOT used by the rules (rerun: {n}_x2)"
            elif n in used_comp:
                out[n] = (f"comparator of record for {', '.join(used_comp[n])} only; its rerun {n}_x2 is the "
                          "comparator of the rerun pair(s)")
            else:
                out[n] = f"NOT used by the rules (rerun: {n}_x2)"
        if s.cls == "companion" and n not in of_record:
            out[n] = (f"I4 option (beta) companion - NOT used by the rules (I4_OPTION = {I4_OPTION})"
                      if I4_OPTION != "beta" else "I4 (beta) companion, not of record (|Delta| < 0.5)")
        if s.rung == "B1d" and s.cls == "primary" and n not in of_record:
            out[n] = "not of record under I4 option (beta) (|Delta(12000 - 6000)| >= 0.5)"
    return out


def guard_candidates(sm: dict | None = None) -> list:
    """Runs of record of a primary rung (reruns where they exist, review B2) when one of them meets the promotion
    parts (a)-(c): every run of record of that rung (AUDIT 4.2 note: the candidate and its other seed)."""
    sm = sm or all_specs()
    ref, b0 = mlp_array(), b0_mean()
    hits = set()
    for rung, recs in records(sm).items():
        arrs = {n: tagged(n) for n in recs}
        if any(a is not None and meets_abc(parts(a, ref, b0)) for a in arrs.values()):
            hits.update(n for n, a in arrs.items() if a is not None)
    return sorted(hits)


def scoring_consistency(sm: dict) -> dict:
    """A2 (i), boundary case: for every finished run with a fit history, the fit's last validation mean (and the EMA
    stream's) against the registry's mean of the same weights (run_m2b_list.scoring_check; hard stop on a
    disagreement). Data-free runs (B1d and companions) have no validation inside the fit."""
    out = {}
    for n, s in sm.items():
        d = D.done_marker(n)
        if d is None:
            continue
        if s.data_free:
            out[n] = "not applicable: data-free"
            continue
        r = D.row(n)
        if r is None:
            D.hard_stop("scoring", f"{n}: finished (done.json) but no registry row on the live stream")
        h = d.get("history") or []
        out[n] = D.scoring_check(n, h[-1].get("val") if h else None, r["val_mean"])
        if d.get("ema_row"):
            re_ = D.row(f"{n}_ema")
            if re_ is None:
                D.hard_stop("scoring", f"{n}_ema: named in done.json but no registry row on the live stream")
            out[f"{n}_ema"] = D.scoring_check(f"{n}_ema", h[-1].get("val_ema") if h else None, re_["val_mean"])
    return out


# ----------------------------------------------------------------------------------------------------------------------
def collect(smoke: bool) -> dict:
    sm = all_specs()
    scoring = scoring_consistency(sm)                    # a failed scoring check stops before any table
    rec = records(sm)
    pairs = comparison_pairs(sm, rec)
    lab = labels(sm, rec, pairs)
    ref = mlp_array()
    ctrls = control_arrays()
    b0 = b0_mean(ctrls)
    masks = secondary_masks()
    res = {"stage": D.CTX.stage, "stream_id": lib.live_stream_id(), "machine_id": lib.MACHINE.id, "smoke": smoke,
           "b0_complete": b0 is not None, "runs": {}, "primary": {}, "one_se": {}, "ema": {}, "bceil": {},
           "status": {}, "decisions": D.decisions(), "rulings": RULINGS, "i11_reading": I11_READING,
           "i4_option": I4_OPTION, "records": rec, "labels": lab, "scoring_check": scoring, "gate1": []}
    res["b0"] = {"complete": b0 is not None, "controls_seed_sd": controls_seed_sd(ctrls, ref),
                 "controls_vs_mlp": ({n: pd(provenance.paired_delta(a, ref)) for n, a in ctrls.items()}
                                     if ctrls else None)}
    gate1 = res["gate1"]
    of_record = {x for v in rec.values() for x in v}
    for n, s in sm.items():
        st = D.status(s)
        dg = D.diagnosis(n) if st.startswith("diverged") else None
        res["status"][n] = {"status": st, "cls": s.cls, "rung": s.rung,
                            "diagnosis": (dg["verdict"] if dg and dg["valid"] else None)}
        a = tagged(n)
        if a is None:
            continue
        r = D.row(n)
        e = {"rung": s.rung, "cls": s.cls, "seed": s.seed, "k": r["k"], "val_mean": r["val_mean"],
             "primary": n in of_record, "used_by_rules": n not in lab or lab[n].startswith("comparator of record"),
             "label": lab.get(n), **parts(a, ref, b0, ctrls)}
        e["secondary_vs_mlp"] = {k: pd(provenance.paired_delta(a, ref, mask=m)) for k, m in masks.items()}
        if s.comparator and s.comparator != "mlp_std":
            c = comparator_array(s.comparator)
            e["vs_comparator"] = ({"comparator": s.comparator, **pd(provenance.paired_delta(a, c))}
                                  if c is not None else {"comparator": s.comparator, "available": False})
        e["guards"] = guards(n, provenance.paired_difference(a, ref))
        e["d_replication"], e["promotion"] = None, None
        res["runs"][n] = e
    # B1 (E5) and its distillation
    e5 = tagged("E5_smoke" if smoke else "E5_s2", stage=(D.SMOKE_STAGE if smoke else D.REAL_STAGE))
    e5_parts = parts(e5, ref, b0, ctrls) if e5 is not None else None
    res["b1"] = {"E5": e5_parts, "B1d_minus_E5": {}}
    if e5 is not None:
        for n in rec["B1d"]:
            if n in res["runs"]:
                res["b1"]["B1d_minus_E5"][n] = pd(provenance.paired_delta(tagged(n), e5))
    # promotion (statement P) on the runs of record of the primaries (review B3), with replication (d)
    for rung in LIST_PRIMARY_RUNGS:
        recs = rec[rung]
        have = [x for x in recs if x in res["runs"]]
        if len(recs) != 2 or len(have) != 2:
            for x in have:
                res["runs"][x]["promotion"] = "not evaluated (the other seed of record is missing or diverged)"
            continue
        e0, e1 = res["runs"][recs[0]], res["runs"][recs[1]]
        rep = rule_replication(e0["vs_mlp"], e1["vs_mlp"], two_seed(tagged(recs[0]), tagged(recs[1]), ref))
        rep["runs_used"] = list(recs)
        rep["mixed_original_and_rerun"] = len({x.endswith("_x2") for x in recs}) > 1
        if rung == "B1d":
            rep["label"], rep["note"] = B1D_REPLICATION_LABEL, B1D_REPLICATION_NOTE
        e0["d_replication"], e1["d_replication"] = rep, f"see {recs[0]}"
        anomaly = False
        if rung == "B1d":
            for x in recs:
                if b1d_anomaly(res["runs"][x], e5_parts):
                    anomaly = True
                    gate1.append({"row": x, "reason": "anomaly: a B1d run meets (a)-(c) while its teacher E5 does not "
                                                      "(the student would beat the teacher by more than 4)"})
        e0["promotion"] = promotion(e0, rep, e0["guards"]["e"] if e0["guards"] else None, anomaly)
        if e0["promotion"].startswith("undecided"):
            gate1.append({"row": recs[0], "reason": "meets (a)-(d), guard undecided (J-a / J-b)"})
        if rep["disagree"]:
            gate1.append({"row": recs[0], "reason": "replication (d): the literal reading and the alternative "
                                                    "(two-seed mean <= -3 SE) disagree - to the user (I13)"})
        e1["promotion"] = f"n/a (replicate; the rung's result is {recs[0]})"
        if meets_abc(e1) and not meets_abc(e0):
            e1["note"] = "passes (a)-(c) on seed 1 only: printed, not promoted (I13: seed 0 is the result)"
    # primary comparisons (statement Q) on the pairs of record
    for rung, comp in COMPARISONS:
        out = {"comparator": comp, "seeds": {}, "rung": None}
        got = []
        for orig, run, cn, rerun in pairs[rung]:
            a, c = tagged(run), comparator_array(cn)
            if a is None or c is None:
                out["seeds"][run] = {"available": False, "comparator": cn}
                continue
            p = pd(provenance.paired_delta(a, c))
            out["seeds"][run] = {"available": True, "comparator": cn, "rerun_pair": rerun, **p,
                                 "note": between_note(p["mean"]),
                                 "secondary": {k: pd(provenance.paired_delta(a, c, mask=m)) for k, m in masks.items()}}
            got.append((run, a, cn, c))
        if len(got) == 2:
            (r0, a0, c0n, c0), (r1, a1, c1n, c1) = got
            provenance.check_pairable(a0, a1, c0, c1)
            both = pd(provenance.paired_delta(_mean_array([a0, a1], "variant mean"),
                                              _mean_array([c0, c1], "comparator mean")))
            v = rule_primary_comparison(out["seeds"][r0], out["seeds"][r1], both)
            v.update({"two_seed_mean": both, "note": between_note(both["mean"]),
                      "pairs_used": [[r0, c0n], [r1, c1n]],
                      "mixed_original_and_rerun": len({r0.endswith("_x2"), r1.endswith("_x2")}) > 1})
            if rung == "B1d":
                v["label"] = B1D_REPLICATION_LABEL
            if v["replication_disagree"]:
                gate1.append({"row": f"{rung} - {comp}", "reason": "primary comparison: the literal replication and "
                                                                   "the alternative disagree - to the user (I13)"})
            prom = res["runs"].get(rec[rung][0], {}).get("promotion")
            if rung != "B1d" and prom == "promoted" and v["counts"] is False:
                v["mechanism_note"] = (f"{rec[rung][0]} is promoted as a model; the mechanism ({rung} - {comp}) is not "
                                       "shown")
            out["rung"] = v
        res["primary"][rung] = out
    # 1-SE (B6), on the pairs of record (B6b is never rerun)
    per_seed = {}
    for seed in (0, 1):
        c_run, a_run, _ = pair_of_record(f"B6c_s{seed}", f"B6a_s{seed}", sm)
        cands = []
        for cap, run in (("B6a", a_run), ("B6b", f"B6b_s{seed}"), ("B6c", c_run)):
            r = D.row(run)
            cands.append((cap, run, (r or {}).get("k", 0), tagged(run).values if r else None))
        per_seed[seed] = cands
    arrs = [tagged(run) for c in per_seed.values() for _, run, _, a in c if a is not None]
    if arrs:
        provenance.check_pairable(*arrs)
    res["one_se"] = rule_one_se(per_seed)
    if res["one_se"].get("statement", "").startswith("no automatic"):
        gate1.append({"row": "B6 (1-SE)", "reason": "no capacity is eligible on both seeds (I17)"})
    # B3 EMA
    for n, s in D.spec_map().items():
        if s.ema_row and tagged(n) is not None and tagged(f"{n}_ema") is not None:
            res["ema"][n] = pd(provenance.paired_delta(tagged(f"{n}_ema"), tagged(n)))
    # B-ceil
    mlp_mean = float(ref.values.astype(np.float64).mean())
    for n, s in sm.items():
        if s.rung != "B-ceil" or n not in res["runs"]:
            continue
        d = D.done_marker(n) or {}
        h = [x for x in (d.get("history") or []) if x.get("val") is not None]
        mn = min(h, key=lambda x: x["val"]) if h else None
        res["bceil"][n] = {"last_vs_mlp": res["runs"][n]["vs_mlp"],
                           "trajectory_min": ({"step": mn["step"], "val": mn["val"],
                                               "minus_mlp_mean": mn["val"] - mlp_mean} if mn else None),
                           "train_subset": d.get("train_subset")}
    # I4: KL trajectories of the B1d runs (alpha) and the prepared companions (beta)
    res["i4"] = {"option": I4_OPTION, "kl": {}, "companions": {}}
    for n in _seed_runs("B1d") + list(D.companion_map()):
        d = D.done_marker(n)
        if d is not None:
            res["i4"]["kl"][n] = d.get("kl")
    for n in _seed_runs("B1d"):
        comp = f"{n}{D.COMPANION_SUFFIX}"
        a6, a12 = tagged(n), (tagged(comp) if comp in sm else None)
        res["i4"]["companions"][n] = ({"companion": comp, **rule_b1d_convergence(a6, a12)}
                                      if a6 is not None and a12 is not None else None)
    return res


def markdown(res: dict) -> str:
    L = []
    w = L.append
    yn = lambda v: "-" if v is None else ("yes" if v else "no")  # noqa: E731
    w(f"# Track B list (AUDIT.md section 4.2) - {'SMOKE (not numbers of record)' if res['smoke'] else 'results'}\n")
    w(f"Stage `{res['stage']}`, machine `{res['machine_id']}`, stream `{res['stream_id']}`. Validation only; "
      "1e-4 nat per event, paired, ± SE. NOT gated. Readings as ruled in results/m2b_mlp/REVIEW.md section 9.1.\n")
    sc = [v for v in res["scoring_check"].values() if isinstance(v, dict)]
    w(f"Scoring check (A2 (i)): {len(sc)} finished fits (main and EMA rows) - the fit's last validation mean equals the "
      f"registry's mean of the same weights to {D.SCORING_TOL:g} in every case (largest difference "
      f"{max([v['abs_diff'] for v in sc], default=0.0):.1e}).\n")
    if not res["b0_complete"]:
        w("**B0 INCOMPLETE:** fewer than the three controls b0h_seed0..2 are registered on this stream; part (c) "
          "of the promotion rule is not evaluated.\n")
    else:
        sd = res["b0"]["controls_seed_sd"]
        w("B0 = per-event mean of b0h_seed0..2; controls against mlp_std: "
          + ", ".join(f"{n} {fmt(p)}" for n, p in res["b0"]["controls_vs_mlp"].items())
          + f"; seed sd of the controls {sd * E4:.2f} (description beside (c), I12).\n")
    w("## Report metric (K = 64): promotion of a model against mlp_std (statement P)\n")
    w("| run | rung | class | k | val | Δ mlp_std | (a) ≤ −5 | (b) ≤ −3 SE | Δ B0 (z) | (c) z < −1 | Δ per control "
      "(description) | (d) replicated (literal) | (d) alternative ≤ −3 SE (decides nothing) | (e) guards | promotion | "
      "Δ comparator |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, e in res["runs"].items():
        rep = e["d_replication"]
        if isinstance(rep, dict):
            used = f" [{' + '.join(rep['runs_used'])}]" if rep.get("mixed_original_and_rerun") else ""
            lab = f" ({rep['label']})" if rep.get("label") else ""
            d_lit, d_alt = yn(rep["replicated"]) + lab + used, yn(rep["replicated_alternative_se"])
        else:
            d_lit, d_alt = (rep or "-"), "-"
        vc = e.get("vs_comparator")
        promo = e["promotion"] or ("n/a (not used)" if not e["used_by_rules"] else f"n/a ({e['cls']})")
        ctl = ", ".join(f"{fmt(p)}" for p in (e.get("vs_controls") or {}).values()) or "-"
        name = n + (f" — {e['label']}" if e.get("label") else "") + (f" — {e['note']}" if e.get("note") else "")
        w(f"| {name} | {e['rung']}{' (primary, of record)' if e['primary'] else ''} | {e['cls']} | {e['k']} | "
          f"{e['val_mean']:.6f} | {fmt(e['vs_mlp'])} | {yn(e['a_gain'])} | {yn(e['b_z'])} | {fmtz(e['vs_b0'])} | "
          f"{yn(e['c_b0'])} | {ctl} | {d_lit} | {d_alt} | {e['guards']['e'] if e['guards'] else 'not evaluated'} | "
          f"{promo} | {(vc['comparator'] + ' ' + fmt(vc) if vc and vc.get('mean') is not None else '-')} |")
    w("\nPromotion is evaluated for the primaries' runs of record only; every other row prints n/a with its class "
      "(review B3). " + MULTIPLICITY_NOTE)
    w(f"\n*B1d, part (d):* {B1D_REPLICATION_LABEL} - {B1D_REPLICATION_NOTE}.")
    # B1 / B1d
    if res["b1"].get("E5"):
        e5 = res["b1"]["E5"]
        w(f"\nB1 (E5, the teacher) vs mlp_std {fmt(e5['vs_mlp'])}; (a) {yn(e5['a_gain'])}, (b) {yn(e5['b_z'])}, "
          f"(c) {yn(e5['c_b0'])}. B1d − E5 (paired, the distillation loss; replaces the retained fraction): "
          + (", ".join(f"{n} {fmtz(p)}" for n, p in res["b1"]["B1d_minus_E5"].items()) or "-") + ".")
    # primary comparisons
    hdr = {"i": "(i) ≤ −5, ≤ −3 SE, replicated", "iii": "(iii) ≤ −3, ≤ −2 SE, half −1.5, mean −3"}
    verdict_lab = (f"verdict: option ({res['i11_reading']}) as ruled, AWAITING THE USER'S CONFIRMATION"
                   if res["i11_reading"] != "ii" else "verdict: option (ii), descriptive")
    w("\n## Primary comparisons (statement Q, I11; never combined with promotion)\n")
    w(f"| rung | row | pair used | Δ ± SE (z) | note | {verdict_lab} | option (i) | option (iii) (decides nothing) | "
      "replication (i): seed 1 ≤ −2.5 / same sign / mean ≤ −5 / alt. ≤ −3 SE | p < 2 GeV | p ≥ 2 GeV | contained | "
      "side-leaking |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rung, out in res["primary"].items():
        for run, c in out["seeds"].items():
            if not c.get("available"):
                w(f"| {rung} | seed row | {run} − {c['comparator']} | not available (diverged, blocked or missing) | | | | | "
                  "| | | | |")
                continue
            sec = c["secondary"]
            w(f"| {rung} | seed row | {run} − {c['comparator']}{' (rerun pair)' if c['rerun_pair'] else ''} | "
              f"{fmtz(c)} | {c['note'] or ''} | | | | | {fmt(sec['p<2GeV'])} | {fmt(sec['p>=2GeV'])} | "
              f"{fmt(sec['contained'])} | {fmt(sec['side-leaking'])} |")
        v = out["rung"]
        if v:
            r = v["replication"]
            rp = (f"{yn(r['seed1_half_threshold'])} / {yn(r['same_sign'])} / {yn(r['two_seed_mean_meets_threshold'])}"
                  f" / {yn(r['replicated_alternative_se'])}") if r else "-"
            pu = " + ".join(f"{a} − {b}" for a, b in v["pairs_used"])
            verdict = ("descriptive" if v["counts"] is None else ("counts" if v["counts"] else "does not count"))
            w(f"| {rung} | two-seed mean{' (' + v['label'] + ')' if v.get('label') else ''} | {pu}"
              f"{' (mixed original / rerun)' if v['mixed_original_and_rerun'] else ''} | {fmtz(v['two_seed_mean'])} | "
              f"{v['note'] or ''} | **{verdict}** | {yn(v['option_i'])} | {yn(v['option_iii'])} | {rp} | | | | |")
            if v.get("mechanism_note"):
                w(f"| {rung} | note | | {v['mechanism_note']} | | | | | | | | | |")
    w(f"\nOption (i) is the ruling (REVIEW.md 9.1, I11) and needs the user's confirmation; options (iii) and (ii) "
      f"(the Δ ± SE columns) are printed beside it and decide nothing. Headers: {hdr['i']}; {hdr['iii']}.")
    # guards
    w("\n## Guards (bulk-restricted and other guard numbers; own table; J-a / J-b: three values)\n")
    w("| run | G-K Δ | G-K passed / resolved → verdict | G-bulk Δ (stratum) | G-bulk passed / resolved → verdict | "
      "report metric on stratum | G-region flagged share | G-region margin (z) | G-region passed / resolved → verdict | "
      "normalisation term | decomposition residual (run / mlp_std) | (e) | mass beyond zb |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    any_g = False
    for n, e in res["runs"].items():
        g = e["guards"]
        if not g:
            continue
        any_g = True
        gr = g["g_region"]
        share = gr.get("flagged_share_of_gain") if gr else None
        gk, gb = g["gk_decision"], g["bulk_decision"]
        dec = lambda x: f"{yn(x['passed'])} / {yn(x['resolved'])} → {x['verdict']}"  # noqa: E731
        res_s = (f"{gr['decomposition_residual']['run']:.1e} / {gr['decomposition_residual']['mlp_std']:.1e}"
                 if gr else "-")
        w(f"| {n} | {fmt(g['gk'])} | {dec(gk)} | {fmt(g['bulk'])} | {dec(gb)} | {fmt(g['report_on_stratum'])} | "
          f"{f'{share:.2f}' if share is not None else '-'} | {fmtz(gr['margin']) if gr else '-'} | "
          f"{dec(gr) if gr else '-'} | {fmt(gr['normalisation_term']) if gr else '-'} | {res_s} | {g['e']} | "
          f"{_num(g['mass'].get('beyond_zb', {}).get('mean'))} |")
    if not any_g:
        w("No run of record met the promotion parts (a)-(c); no guard evaluation was required (AUDIT 4.2 note).")
    # 1-SE
    w("\n## 1-SE rule (B6; I17: per seed, then eligibility on both seeds)\n")
    o = res["one_se"]
    for seed, v in o["per_seed"].items():
        if "eligible" in v:
            w(f"- seed {seed} ({', '.join(f'{c}: {r}' for c, r in v['runs'].items())}): best {v['best']}, eligible "
              f"{v['eligible']}, selected {v['selected']}")
        else:
            w(f"- seed {seed}: not evaluated (missing or diverged: {', '.join(v['not_evaluated'])})")
    w(f"- combined: **{o.get('statement')}**" + (f"; eligible on both seeds {o['eligible_on_both_seeds']}"
                                                 if o.get("eligible_on_both_seeds") is not None else ""))
    # Gate 1 list
    w("\n## Listed for Gate 1 / for the user (not decided automatically)\n")
    if res["gate1"]:
        for g in res["gate1"]:
            w(f"- {g['row']}: {g['reason']}")
    else:
        w("- nothing")
    # I4
    w(f"\n## B1d convergence (I4, with the user; option ({res['i4_option']}) applied)\n")
    w("Option (alpha): no convergence statement for B1d (no validation trajectory); the KL of the data-free fit as "
      "logged (initial, final and every steps/5 steps, 3 digits):")
    for n, k in res["i4"]["kl"].items():
        if not k:
            w(f"- {n}: no KL record")
            continue
        tr = ", ".join(f"{t['step']}: {t['kl']:.3e}" if t["kl"] is not None else f"{t['step']}: -"
                       for t in k.get("trajectory") or [])
        ini, fin = k.get("initial"), k.get("final")
        w(f"- {n}: KL {ini:.4e} -> {fin:.4e} ({k.get('steps')} steps); trajectory {tr}" if ini is not None and
          fin is not None else f"- {n}: trajectory {tr}")
    comps = {n: v for n, v in res["i4"]["companions"].items() if v}
    if comps:
        w("\nOption (beta), prepared (companions with 12,000 KL steps; decides only with I4_OPTION = \"beta\"):")
        for n, v in comps.items():
            w(f"- {v['companion']} − {n}: {fmtz(v['delta_12000_minus_6000'])} -> of record under (beta): the "
              f"{v['of_record']}-step run")
    else:
        w("\nOption (beta) prepared, not run (driver phase b1d_companions, not in the default list).")
    # B3, B-ceil, status, convergence
    w("\n## B3: EMA stream minus last iterate\n")
    for n, p in res["ema"].items():
        w(f"- {n}: {fmt(p)}")
    w("\n## B-ceil (diagnostic)\n")
    w("| run | last iterate Δ mlp_std | trajectory minimum (step, mean − mlp_std mean, not paired) | train subset |")
    w("|---|---|---|---|")
    for n, b in res["bceil"].items():
        tm = b["trajectory_min"]
        tm_s = f"{tm['step']}, {tm['minus_mlp_mean'] * E4:+.2f}" if tm else "-"
        frac = (b["train_subset"] or {}).get("fraction", 1.0)
        w(f"| {n} | {fmt(b['last_vs_mlp'])} | {tm_s} | {frac} |")
    w("\n## Status of every run (divergences, blocked runs)\n")
    for n, st in res["status"].items():
        if st["status"] != "done":
            w(f"- {n} ({st['cls']}): {st['status']}" + (f"; diagnosis: {st['diagnosis']}" if st["diagnosis"] else ""))
    for n, st in (res["decisions"].get("diverged") or {}).items():
        if n.endswith("_ema"):
            w(f"- {n}: {st.get('status')} (EMA row; non-finite in the fit and in the registry: recorded as decided, "
              "A2 (i))")
    conv = res["decisions"].get("convergence") or {}
    if conv:
        w("\n## Convergence (AUDIT 4.0)\n")
        for n, c in conv.items():
            w(f"- {n}: " + (f"moved {c['move'] * E4:.2f} over {c['window_steps']} steps -> "
                            f"{'converged' if c['converged'] else 'NOT converged, reruns ' + str(c.get('reruns'))}"
                            if c.get("available") else "not available"))
    w("\n## Rulings applied (results/m2b_mlp/REVIEW.md section 9.1; one function each in this script)\n")
    for k, v in res["rulings"].items():
        w(f"- **{k}** ({v['status']}; `{v['function']}`): {v['text']}")
    return "\n".join(L) + "\n"


def run(argv=None) -> int:
    """The report; a failed check raises run_m2b_list.HardStop (the driver's phase 'report' lets it through)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    D.configure(args.smoke)
    if not D.CTX.batch:
        D.CTX.batch = lib.start(f"run_m2b_list_report ({D.CTX.stage})")
    res = collect(args.smoke)
    res["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    D.write_json(res, D.out_dir() / "list_results.json")
    md = markdown(res)
    f = D.out_dir() / "LIST_RESULTS.md"
    lib.paths.check_output_path(f)
    f.write_text(lib.paths.scrub_str(md))
    print(md)
    print(f"wrote {lib.paths.portable(f)}")
    return 0


def main(argv=None) -> int:
    try:
        return run(argv)
    except D.HardStop as err:
        D._stop_line("hard", str(err))
        return D.HARD_STOP_EXIT


if __name__ == "__main__":
    sys.exit(main())
