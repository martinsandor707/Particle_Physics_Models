"""Track A (M2A) driver: the pre-registered run list of results/m1_audit/AUDIT.md section 4.1, executed in order.

    uv run python -B shape-form-martin/scripts/run_m2a.py <phase> [<phase> ...]

Phases (each skips finished runs - done.json - and resumes interrupted fits from their checkpoint.pt):
    templates   back_corrected / full_box acceptance templates from the TRAINING events, and the training thetas for
                the detector-frame halo -> results/m2a_dist/templates/
    parity      A0 at step 0 (legacy_centre, legacy coefficients) registered as A0_init; must be 18.120061 to 1e-5
    plr         learning-rate pilot P-lr: A2s at lr 0.003 / 0.01 / 0.03 (seed 0, 3000 steps) -> lr_poly frozen;
                Stage-0 run A2s_s0 (the base the net pilot is fitted to); A2n at lr 0.001 / 0.003 / 0.01 -> lr_net
    stage0      A0, A1, A2e (seeds 0, 1), A2s, A2n (seeds 0, 1, 2); A0 drift rule (|A0 - formula| > 2: stop)
    stage1p     primaries A4, A5, A7, A8, A9, A10: poly seeds 0 and 1, then net seed 0; convergence reruns
    stage1x     exploratory runs in table order (seed 0; seed 1 if seed 0 passes keep or equal-NLL), budget/drop order
    guards      registry guard columns (+ G-region, secondary columns) for kept / equal-NLL rungs and comparators;
                Fisher information of k for A6b
    stage2      A11 (or the statement that A11 = A2s), A11k (k_train = 64), A11n
    all         everything above in order
The rules, tables and RESULTS.md: scripts/run_m2a_report.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_m2a_lib as L  # noqa: E402
from run_m2a_lib import OUT, PLR_FILE, STAGE, Spec, log  # noqa: E402

from sfm import registry, stats, train  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402

LN20 = math.log(20.0)
FORMULA_OF_RECORD = 18.120061
BUDGET_S = 4.0 * 3600.0
DECISIONS = OUT / "decisions.json"


def decisions() -> dict:
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}


def note_decision(key: str, value) -> None:
    d = decisions()
    d[key] = value
    train.atomic_write_json(d, DECISIONS)


def run_name(rung: str, seed: int) -> str:
    return f"{rung}_s{seed}"


def a2s(seed: int) -> str:
    return f"{STAGE}/{run_name('A2s', seed)}"


# ----------------------------------------------------------------------------------------------------------------------
# the run list (AUDIT 4.1)
# ----------------------------------------------------------------------------------------------------------------------
def stage0_specs() -> list:
    out = []
    for s in (0, 1):
        out.append(Spec(run_name("A0", s), "A0", "component", dict(variant="legacy_centre"), s, "poly", "control",
                        comparator="formula"))
    for s in (0, 1):
        out.append(Spec(run_name("A1", s), "A1", "component", dict(variant="legacy_centre", provider="net"), s, "net",
                        "control", comparator=run_name("A0", s)))
    for s in (0, 1):
        out.append(Spec(run_name("A2e", s), "A2e", "component", dict(variant="legacy_exact"), s, "poly",
                        "exploratory", comparator=run_name("A0", s)))
    for s in (0, 1, 2):
        out.append(Spec(run_name("A2s", s), "A2s", "component", dict(variant="survival_exact"), s, "poly", "base",
                        comparator=run_name("A2e", s) if s < 2 else None))
    for s in (0, 1, 2):
        out.append(Spec(run_name("A2n", s), "A2n", "component",
                        dict(variant="survival_exact", provider="net", net_from=a2s(s)), s, "net", "base",
                        comparator=run_name("A2s", s)))
    return out


def primary_kwargs(rung: str, seed: int) -> tuple:
    """(factory, kwargs, steps) of a primary's poly version."""
    st = a2s(seed)
    if rung == "A4":
        return "component", dict(shower_long="emg", start=st, init="shower_long"), 4000
    if rung == "A5":
        return "component", dict(shower_long="gengamma", shower_long_kw={"soft": "free"}, start=st,
                                 init="shower_long"), 4000
    if rung == "A7":
        return "component", dict(shower_lat=["gauss", "burr", "exp"], start=st), 4000
    if rung == "A8":
        return "component", dict(shower_lat=["gauss", "cauchy", "weibull"], start=st), 4000
    if rung == "A9":
        return "component", dict(det_halo=True, far=True, start=st), 4000
    if rung == "A10":
        return "mixture", dict(K=3, long="emg", lat="burr", depth="linear", order=True, fine=True, template="stored",
                               init="target", init_ref=st, init_steps=[1500, 1500], init_search=400), 6000
    raise KeyError(rung)


PRIMARIES = ("A4", "A5", "A7", "A8", "A9", "A10")


def primary_specs() -> tuple:
    poly, net = [], []
    for s in (0, 1):
        for r in PRIMARIES:
            fac, kw, steps = primary_kwargs(r, s)
            poly.append(Spec(run_name(r, s), r, fac, kw, s, "poly", "primary", comparator=run_name("A2s", s),
                             steps=steps, init_eval=(r == "A9")))
    for r in PRIMARIES:
        fac, _, steps = primary_kwargs(r, 0)
        net.append(Spec(run_name(r + "n", 0), r, fac, dict(provider="net", net_from=f"{STAGE}/{run_name(r, 0)}"), 0,
                        "net", "primary-net", comparator=run_name("A2n", 0), steps=steps))
    return poly, net


EXPLORATORY = ("A4t", "A5h", "A6i", "A6l", "A6b", "A7s", "A7l", "A8c", "A78", "A9r", "A3f", "A3c", "A3x",
               "A10-2", "A10-4", "A10-5", "A10g", "A12")
DROP_ORDER = ("A10-5", "A10-4", "A10g", "A6b", "A3f", "A3c")      # then: net runs of primaries failing keep on both seeds


def exploratory_spec(rung: str, seed: int) -> Spec:
    st = a2s(seed)
    mix = dict(depth="linear", order=True, fine=True, template="stored", init="target", init_ref=st,
               init_steps=[1500, 1500], init_search=400)
    table = {
        "A4t": ("component", dict(shower_long="emg_tied", start=st, init="shower_long"), 4000),
        "A5h": ("component", dict(shower_long="gengamma", shower_long_kw={"soft": 2.0}, start=st, init="shower_long"),
                4000),
        "A6i": ("component", dict(shower_long="invgauss", shower_long_kw={"soft": "free"}, start=st,
                                  init="shower_long"), 4000),
        "A6l": ("component", dict(shower_long="lognormal", shower_long_kw={"soft": "free"}, start=st,
                                  init="shower_long"), 4000),
        "A6b": ("component", dict(shower_long="burr", shower_long_kw={"soft": "free"}, degrees={"sh_rk": 0}, start=st,
                                  init="shower_long"), 4000),
        "A7s": ("component", dict(shower_lat=["gauss", "student", "exp"], start=st), 4000),
        "A7l": ("component", dict(shower_lat=["gauss", "cauchy", "burr"],
                                  lat_family_kw=[{}, {}, {"c": 1.0, "init": [1.0, 20.0]}],
                                  coef_shift={"log_lam": LN20}, start=st), 4000),
        "A8c": ("component", dict(shower_lat=["weibull_core", "cauchy", "exp"], start=st), 4000),
        "A78": ("component", dict(shower_lat=["gauss", "burr", "weibull"], start=st), 4000),
        "A9r": ("component", dict(det_halo=True, far=False, logit_dh_from_far=True, start=st), 4000),
        "A3f": ("component", dict(template="front", start=st), 4000),
        "A3c": ("component", dict(template="back_corrected", start=st), 4000),
        "A3x": ("component", dict(template="full_box", start=st), 4000),
        "A10-2": ("mixture", dict(K=2, long="emg", lat="burr", **mix), 6000),
        "A10-4": ("mixture", dict(K=4, long="emg", lat="burr", **mix), 6000),
        "A10-5": ("mixture", dict(K=5, long="emg", lat="burr", **mix), 6000),
        "A10g": ("mixture", dict(K=3, long="gengamma", long_kw={"soft": "free"}, lat="burr", **mix), 6000),
        "A12": ("component", dict(last_ring=True, start=st), 4000),
    }
    fac, kw, steps = table[rung]
    return Spec(run_name(rung, seed), rung, fac, kw, seed, "poly", "exploratory", comparator=run_name("A2s", seed),
                steps=steps)


# ----------------------------------------------------------------------------------------------------------------------
# rules used while running (the full tables are in run_m2a_report.py)
# ----------------------------------------------------------------------------------------------------------------------
def paired(a: str, b: str) -> stats.Paired:
    if b == "formula":
        return stats.paired(L.val_array(a), np.load(L.paths.M0_DIR / "formula_val_nll.npy"))
    return stats.paired(L.val_array(a), L.val_array(b))


def keep_rule(p: stats.Paired) -> bool:
    return p.mean <= -3e-4 and p.mean <= -2.0 * p.se


def equal_rule_one(p: stats.Paired) -> bool:
    """One seed of the equal-NLL rule: Delta within +1 paired SE and below +1 (1e-4)."""
    return p.mean <= p.se and p.mean < 1e-4


def converged(name: str) -> tuple:
    d = json.loads((OUT / name / "done.json").read_text())
    h = d["history"]
    last = h[-1]
    prev = [x for x in h if x["step"] == last["step"] - 1000]
    if not prev:
        return None, None
    mv = abs(last["val"] - prev[0]["val"])
    return mv < 0.5e-4, mv


# ----------------------------------------------------------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------------------------------------------------------
def phase_templates():
    f = dfac.TEMPLATE_DIR / "templates.json"
    if f.exists() and dfac.THETA_FILE.exists():
        log("templates: already computed")
        return
    sets, extras = L.data_cpu()
    dfac.TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(dfac.THETA_FILE, np.asarray(extras["train"]["theta"], dtype=np.float64))

    def job():
        t0 = time.time()
        out = dfac.compute_templates(sets["train"], extras["train"]["theta"], extras["train"]["phi"], log=log)
        stored = torch.load(L.paths.PION_RESULTS_DIR / "abar.pt", map_location="cpu",
                            weights_only=False).double().numpy()
        front = torch.load(L.paths.M0_DIR / "abar_front.pt", map_location="cpu", weights_only=False).double().numpy()
        for k in dfac.NEW_TEMPLATES:
            torch.save(torch.tensor(out[k], dtype=torch.float32), dfac.TEMPLATE_DIR / f"{k}.pt")
        np.savez_compressed(dfac.TEMPLATE_DIR / "templates_all.npz", **out)
        meta = {"definition": {"back_corrected": "mean_e A_front&back / mean_e 1[z' < zb_e]",
                               "full_box": "mean_e A_front&back&side / mean_e 1[z' < zb_e]"},
                "events": "training split (sfm.data.load_sets('train')), 69,783 events",
                "code": "sfm.models.dist_factory.compute_templates / acceptances (port of "
                        "results/m1_audit/checks/c_side.py::acceptances, ::exit_depth)",
                "geometry": {"X_E_mm": dfac.X_E, "z_back_mm": dfac.Z_BACK, "layer_mm": dfac.LAYER,
                             "nsub": dfac.NSUB, "nphi": dfac.NPHI},
                "fb_vs_stored_abar": {"max_abs": float(np.abs(out["fb"] - stored).max()),
                                      "mean_abs": float(np.abs(out["fb"] - stored).mean())},
                "front_vs_abar_front": {"max_abs": float(np.abs(out["front"] - front).max()),
                                        "mean_abs": float(np.abs(out["front"] - front).mean())},
                "ranges": {k: {"min": float(out[k].min()), "max": float(out[k].max()), "mean": float(out[k].mean())}
                           for k in out},
                "cells_above_1": {k: int((out[k] > 1).sum()) for k in dfac.NEW_TEMPLATES},
                "seconds": time.time() - t0}
        train.atomic_write_json(meta, f)
        log(f"templates: {json.dumps(meta['fb_vs_stored_abar'])} ranges {json.dumps(meta['ranges'])}")

    L.on_gpu("templates", job)


def phase_parity():
    row = L.registered_row("A0_init")
    if row is None:
        sets, extras = L.data_cpu()

        def job():
            dev = L.common.DEVICE
            va = sets["val"].to(dev)
            val = registry.ValidationSet(va, extras["val"]["event_number"])
            train.seed_everything(0)
            ctx = L.models.BuildContext(device=dev, train=None, seed=0, log=log, skip_init=False)
            model = L.models.build("component", ctx, variant="legacy_centre")
            info = getattr(model, "init_info", {})
            return registry.evaluate(model, "A0_init", STAGE, val,
                                     meta={"seed": 0, "k": int(model.n_params()), "rung": "A0", "kl_map":
                                           info.get("kl_map"), "init": info,
                                           "config": {"factory": "component", "kwargs": {"variant": "legacy_centre"},
                                                      "step": 0},
                                           "notes": "harness parity: A0 at step 0 (legacy coefficients)"})

        row = L.on_gpu("parity A0_init", job)
    v = row["val_mean"]
    ok = abs(v - FORMULA_OF_RECORD) <= 1e-5
    note_decision("parity", {"A0_init_val": v, "formula_of_record": FORMULA_OF_RECORD, "abs_diff": abs(v - FORMULA_OF_RECORD),
                             "vs_formula_mean": row["vs_formula"]["mean"], "passed": ok})
    log(f"parity: A0 at step 0 = {v:.7f} (record {FORMULA_OF_RECORD}), |diff| {abs(v - FORMULA_OF_RECORD):.2e}, "
        f"paired vs formula {row['vs_formula']['mean']:+.2e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("STOP: harness parity failed (A0 at step 0 differs from 18.120061 by more than 1e-5)")


def _plr_decide(rows: dict) -> float:
    """Lowest last-iterate validation mean wins; within 0.5e-4 the smaller lr wins."""
    best = min(rows.values())
    return min(lr for lr, v in rows.items() if v <= best + 0.5e-4)


def phase_plr():
    if not decisions().get("parity", {}).get("passed"):
        raise SystemExit("parity not established; run the parity phase first")
    d = json.loads(PLR_FILE.read_text()) if PLR_FILE.exists() else {}
    if d.get("lr_poly") is None:
        rows = {}
        for lr in (0.003, 0.01, 0.03):
            sp = Spec(f"plr_A2s_lr{lr:g}", "plr", "component", dict(variant="survival_exact"), 0, "poly", "pilot",
                      steps=3000, lr=lr, notes="learning-rate pilot P-lr (not a comparison)")
            rows[lr] = L.run_spec(sp)["val_mean"]
        d["poly"] = {f"{k:g}": v for k, v in rows.items()}
        d["lr_poly"] = _plr_decide(rows)
        train.atomic_write_json(d, PLR_FILE)
        log(f"P-lr poly: {d['poly']} -> lr_poly = {d['lr_poly']}")
    # the A2s seed-0 base of Stage 0 (needed as the fit_to target of the net pilot)
    L.run_spec(next(s for s in stage0_specs() if s.name == "A2s_s0"))
    d = json.loads(PLR_FILE.read_text())
    if d.get("lr_net") is None:
        rows = {}
        for lr in (0.001, 0.003, 0.01):
            sp = Spec(f"plr_A2n_lr{lr:g}", "plr", "component",
                      dict(variant="survival_exact", provider="net", net_from=a2s(0)), 0, "net", "pilot", steps=3000,
                      lr=lr, notes="learning-rate pilot P-lr (not a comparison); ParamNet fit_to A2s_s0")
            rows[lr] = L.run_spec(sp)["val_mean"]
        d["net"] = {f"{k:g}": v for k, v in rows.items()}
        d["lr_net"] = _plr_decide(rows)
        train.atomic_write_json(d, PLR_FILE)
        log(f"P-lr net: {d['net']} -> lr_net = {d['lr_net']}")


def phase_stage0():
    specs = stage0_specs()
    for sp in specs[:2]:                                       # A0 seeds 0, 1 and the drift rule
        L.run_spec(sp)
    drift = {}
    for sp in specs[:2]:
        p = paired(sp.name, "formula")
        drift[sp.name] = {"mean": p.mean, "se": p.se}
    stop = any(abs(v["mean"]) > 2e-4 for v in drift.values())
    note_decision("A0_drift", {"runs": drift, "rule": "|A0 - formula| > 2e-4: stop", "stop": stop})
    log(f"A0 drift: {drift} -> {'STOP' if stop else 'continue'}")
    if stop:
        raise SystemExit("STOP: A0 drift rule (|A0 - formula| > 2e-4): harness or optimum differs")
    for sp in specs[2:]:
        L.run_spec(sp)


def _converge_and_rerun(poly_specs):
    """Convergence rule for the poly primaries: a non-converged primary is rerun once with doubled steps together
    with its comparator (A2s of the same seed, doubled steps); nothing else changes."""
    out = {}
    for sp in poly_specs:
        ok, mv = converged(sp.name)
        out[sp.name] = {"converged": ok, "moved_last_1000": mv}
        if ok is False:
            comp = next(s for s in stage0_specs() if s.name == sp.comparator)
            comp2 = replace(comp, name=comp.name + "_x2", steps=comp.steps * 2, cls="rerun",
                            notes=f"comparator rerun with doubled steps for {sp.name}")
            sp2 = replace(sp, name=sp.name + "_x2", steps=sp.steps * 2, cls="rerun", comparator=comp2.name,
                          init_eval=False, notes=f"convergence rerun of {sp.name} with doubled steps")
            L.run_spec(comp2)
            L.run_spec(sp2)
            ok2, mv2 = converged(sp2.name)
            out[sp.name]["rerun"] = {"name": sp2.name, "comparator": comp2.name, "converged": ok2,
                                     "moved_last_1000": mv2}
    note_decision("convergence_primaries", out)
    return out


def phase_stage1p():
    poly, net = primary_specs()
    for sp in poly:
        L.run_spec(sp)
    for sp in net:
        L.run_spec(sp)
    _converge_and_rerun(poly)


def _keep_or_equal(name: str, comp: str) -> dict:
    p = paired(name, comp)
    return {"mean": p.mean, "se": p.se, "keep": keep_rule(p), "equal": equal_rule_one(p)}


def _est_seconds(sp: Spec) -> float:
    """Expected locked seconds of a run from finished runs of the same kind (steps scale linearly)."""
    def per_step(pred):
        xs = []
        for f in OUT.glob("*/done.json"):
            d = json.loads(f.read_text())
            s = d["spec"]
            if pred(s) and d.get("job_seconds"):
                xs.append(d["job_seconds"] / s["steps"])
        return float(np.median(xs)) if xs else None
    if sp.factory == "mixture":
        K = sp.kwargs.get("K", 3)
        ps = per_step(lambda s: s["factory"] == "mixture" and s["provider"] == "poly")
        return (ps or 0.07) * sp.steps * (K / 3.0)
    ps = per_step(lambda s: s["factory"] == "component" and s["provider"] == "poly"
                  and s["kwargs"].get("variant", "survival_exact") == "survival_exact")
    return (ps or 0.055) * sp.steps


def _reserve_seconds() -> float:
    """GPU time to keep for what follows the exploratory runs: Stage 2 (A11k two seeds at k_train = 64, plus A11 /
    A11n reruns if anything is kept) and the guard evaluations (about 3 min per model)."""
    base = _est_seconds(Spec("x", "x", "component", {}, 0, "poly", "x", steps=4000))
    return 2 * base * 1.8 + 6 * 180.0


def phase_stage1x():
    used = L.gpu_seconds_total()
    poly, net = primary_specs()
    dropped = decisions().get("dropped", [])
    # which net runs of primaries failed keep on both poly seeds (last in the drop order; they already ran)
    order = list(EXPLORATORY)
    done_seed1 = []
    for r in order:
        sp = exploratory_spec(r, 0)
        if not sp.done():
            used = L.gpu_seconds_total()
            pending = [exploratory_spec(x, 0) for x in order[order.index(r):] if x not in dropped]
            need = sum(_est_seconds(s) for s in pending if not s.done()) + _reserve_seconds()
            while used + need > BUDGET_S:
                cand = [x for x in DROP_ORDER if x not in dropped and x in [s.rung for s in pending]]
                if not cand:
                    break
                dropped.append(cand[0])
                note_decision("dropped", dropped)
                log(f"budget: used {used / 3600:.2f} h, projected {(used + need) / 3600:.2f} h > 4 h -> drop {cand[0]}")
                pending = [s for s in pending if s.rung not in dropped]
                need = sum(_est_seconds(s) for s in pending if not s.done()) + _reserve_seconds()
            if r in dropped:
                log(f"{r}: dropped (budget, pre-registered drop order)")
                continue
        L.run_spec(sp)
        ke = _keep_or_equal(sp.name, sp.comparator)
        log(f"{sp.name}: Delta vs {sp.comparator} = {ke['mean'] * 1e4:+.2f} +- {ke['se'] * 1e4:.2f} (1e-4); "
            f"keep {ke['keep']}, equal-NLL (seed 0) {ke['equal']}")
        if ke["keep"] or ke["equal"]:
            L.run_spec(exploratory_spec(r, 1))
            done_seed1.append(r)
    note_decision("exploratory_seed1", done_seed1)
    note_decision("dropped", dropped)


def phase_guards():
    import run_m2a_report as R
    R.guards_phase()


def phase_stage2():
    import run_m2a_report as R
    R.stage2_phase()


PHASES = {"templates": phase_templates, "parity": phase_parity, "plr": phase_plr, "stage0": phase_stage0,
          "stage1p": phase_stage1p, "stage1x": phase_stage1x, "guards": phase_guards, "stage2": phase_stage2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="+", choices=list(PHASES) + ["all"])
    args = ap.parse_args(argv)
    phases = list(PHASES) if "all" in args.phases else args.phases
    for ph in phases:
        log(f"=== phase {ph} (GPU time used so far {L.gpu_seconds_total() / 3600:.2f} h) ===")
        PHASES[ph]()
    log(f"=== finished {phases}; GPU time used {L.gpu_seconds_total() / 3600:.2f} h ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
