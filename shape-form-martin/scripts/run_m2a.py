"""Track A (M2A) driver: the pre-registered run list of results/m1_audit/AUDIT.md section 4.1, run IN FULL, in order,
one GPU job at a time, in ONE process, resumable (PLAN.md section 13 A6 revised points 4 and 6, section 16.7 E4).

    uv run python -B shape-form-martin/scripts/run_m2a.py                     # the default list, in order:
                                                                               # templates parity plr stage0 stage1p
                                                                               # stage1x guards report
    uv run python -B shape-form-martin/scripts/run_m2a.py <phase> [...]       # selected phases (same order rules)
    uv run python -B shape-form-martin/scripts/run_m2a.py --list              # dry listing of the real run (no GPU)
    SFM_M2A_SMOKE=1 uv run python -B shape-form-martin/scripts/run_m2a.py --smoke   # shortened fits, stage m2a_smoke
    uv run python -B shape-form-martin/scripts/run_m2a.py stage2              # Stage 2 - NEVER part of the default

Resume: re-issue the same command. Finished runs are skipped by their done.json (written on this machine; a marker of
another machine is refused), an interrupted fit resumes from its own checkpoint.pt (sfm.train.fit refuses a
checkpoint of another machine), a registry row is never written twice for the same checkpoint (SHA-256 on the live
stream), decisions are recomputed from the registered arrays (identical numbers) and written again.

Phases
    templates   back_corrected / full_box acceptance templates from the TRAINING events and the training thetas for
                the detector-frame halo -> runs/<stage>/../m2a_dist/templates (this machine; sfm.models.dist_factory)
    parity      A0 at step 0 (legacy_centre, legacy coefficients) registered as A0_init. PAIRED criterion (A6 revised,
                point 4): the A0_init array equals THIS machine's formula reference array within 1e-4 per event and
                1e-6 on the mean (replaces the laptop formula mean to 1e-5). Fail: stop.
    plr         learning-rate pilot P-lr: A2s at lr 0.003 / 0.01 / 0.03 (seed 0, 3000 steps) -> lr_poly frozen;
                Stage-0 run A2s_s0 (the base the net pilot is fitted to); A2n at lr 0.001 / 0.003 / 0.01 -> lr_net.
                Lowest last-iterate validation mean wins, within 0.5e-4 the smaller lr (paired deltas vs the lowest).
    stage0      A0 (seeds 0, 1) -> drift rule |paired(A0, formula array)| > 2e-4: stop; A1, A2e (0, 1), A2s, A2n
                (0, 1, 2)
    stage1p     primaries A4, A5, A7, A8, A9, A10: poly seeds 0 and 1 (A9: step-0 rows), then net seed 0; convergence
                rule (validation mean moved < 0.5e-4 over the last 1000 steps); a non-converged poly primary is rerun
                once with doubled steps together with its comparator (<name>_x2)
    stage1x     all 18 exploratory rungs at seed 0, in table order; seed 1 whenever seed 0 passes the keep rule or the
                equal-NLL rule. NO budget and NO drop order (A6 revised point 6; 16.5 R1): decisions.json records
                dropped = [].
    guards      guard columns (registry.evaluate(guards=True): G-K, G-bulk, G-mass) and G-region arrays for every run
                of a rung that passes the keep or the equal-NLL rule on any seed, and for their comparators; Fisher
                information of k for A6b. Since 26 Nov 2026 also the flagged* arrays (REVIEW 10.5 R6; new files under
                their own subset tag and marker, nothing existing rewritten) and a timing record per run and part
                (guard_timing_<UTC>.json beside the arrays; a part over 60 s is logged as a WARNING)
    report      scripts/run_m2a_report.py: the rules applied to the registered arrays, guard tables, secondary
                columns, RESULTS.md / results.json

Added phases (never in the default list): warmstart-check, warmstart-rerun, a2c (PLAN 17.11); conditional-controls
(PLAN 20 item 1: A2c_x2 / A2c6k evaluated on the registered arrays, trained only if triggered); and, each only on
its own, after the user has seen the Stage-1 table and the approval file exists (REVIEW 10.5):
    stage2         gate (approval R1, preconditions) -> A11 (0, 1, 2), A11k (0, 1; memory probe first), A11n (0, 1)
                   -> their guards -> A11p screen (REVIEW 11; hashed) -> A11p fits -> their guards -> Stage-2 report
    a11p           the A11p part alone (re-issue), behind the same gate
    stage2-report  RESULTS_stage2.md / results_stage2.json (results.json of Stage 1 is never touched)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_m2a_lib as L  # noqa: E402
from run_m2a_lib import Spec, log  # noqa: E402

from sfm import machine, metric, paths, provenance, registry  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402

LN20 = math.log(20.0)
DEFAULT_PHASES = ("templates", "parity", "plr", "stage0", "stage1p", "stage1x", "guards", "report")
# pre-registered thresholds (AUDIT 4.0 / 4.1; nat per event)
KEEP_GAIN, KEEP_Z = 3e-4, 2.0
EQUAL_ABS = 1e-4
DRIFT_STOP = 2e-4
PARITY_EVENT_TOL, PARITY_MEAN_TOL = 1e-4, 1e-6
PLR_TIE = 0.5e-4
CONVERGED_MOVE = 0.5e-4


def decisions() -> dict:
    return L.read_json(L.decisions_file(), {}) or {}


def note_decision(key: str, value) -> None:
    d = decisions()
    d[key] = value
    L.write_json(d, L.decisions_file())


def run_name(rung: str, seed: int) -> str:
    return f"{rung}_s{seed}"


def a2s(seed: int) -> str:
    return f"{L.stage()}/{run_name('A2s', seed)}"


# ----------------------------------------------------------------------------------------------------------------------
# the run list (AUDIT 4.1)
# ----------------------------------------------------------------------------------------------------------------------
def plr_specs() -> tuple:
    poly = [Spec(f"plr_A2s_lr{lr:g}", "plr", "component", dict(variant="survival_exact"), 0, "poly", "pilot",
                 steps=3000, lr=lr, notes="learning-rate pilot P-lr (not a comparison)") for lr in (0.003, 0.01, 0.03)]
    net = [Spec(f"plr_A2n_lr{lr:g}", "plr", "component",
                dict(variant="survival_exact", provider="net", net_from=a2s(0)), 0, "net", "pilot", steps=3000, lr=lr,
                notes="learning-rate pilot P-lr (not a comparison); ParamNet fit_to A2s_s0") for lr in (0.001, 0.003, 0.01)]
    return poly, net


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
        net.append(Spec(run_name(r + "n", 0), r, fac, dict(provider="net", net_from=f"{L.stage()}/{run_name(r, 0)}"),
                        0, "net", "primary-net", comparator=run_name("A2n", 0), steps=steps))
    return poly, net


EXPLORATORY = ("A4t", "A5h", "A6i", "A6l", "A6b", "A7s", "A7l", "A8c", "A78", "A9r", "A3f", "A3c", "A3x",
               "A10-2", "A10-4", "A10-5", "A10g", "A12")


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


def rerun_specs(sp: Spec) -> tuple:
    """Convergence rerun of a poly primary: (comparator rerun, primary rerun), doubled steps, nothing else changed."""
    comp = next(s for s in stage0_specs() if s.name == sp.comparator)
    comp2 = replace(comp, name=comp.name + "_x2", steps=comp.steps * 2, cls="rerun",
                    notes=f"comparator rerun with doubled steps for {sp.name}")
    sp2 = replace(sp, name=sp.name + "_x2", steps=sp.steps * 2, cls="rerun", comparator=comp2.name, init_eval=False,
                  notes=f"convergence rerun of {sp.name} with doubled steps")
    return comp2, sp2


# ----------------------------------------------------------------------------------------------------------------------
# rules used while running (paired deltas on this machine's registered arrays; the full tables: run_m2a_report.py)
# ----------------------------------------------------------------------------------------------------------------------
def keep_rule(p) -> bool:
    """AUDIT 4.0 keep rule: Delta <= -3e-4 and Delta <= -2 SE against the base."""
    return p.mean <= -KEEP_GAIN and p.mean <= -KEEP_Z * p.se


def equal_rule_one(p) -> bool:
    """One seed of the equal-NLL rule: Delta within +1 paired SE and below +1e-4."""
    return p.mean <= p.se and p.mean < EQUAL_ABS


def converged(name: str) -> tuple:
    """(converged, |val(last) - val(last - window)|) from the fit history of THIS machine's run (window 1000 steps;
    smoke: 10). (None, None) if the history has no point at last - window."""
    d = L.done_marker(L.out_dir() / name / "done.json")
    h = d["history"]
    last = h[-1]
    prev = [x for x in h if x["step"] == last["step"] - L.convergence_window()]
    if not prev:
        return None, None
    mv = abs(last["val"] - prev[0]["val"])
    return mv < CONVERGED_MOVE, mv


# ----------------------------------------------------------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------------------------------------------------------
def phase_templates():
    tdir = dfac.template_dir()
    f = tdir / "templates.json"
    if f.exists() and dfac.theta_file().exists() and all((tdir / f"{k}.pt").exists() for k in dfac.NEW_TEMPLATES):
        log(f"templates: already computed ({paths.portable(tdir)})")
        return
    sets, extras = L.data_cpu()
    paths.check_output_path(tdir)
    tdir.mkdir(parents=True, exist_ok=True)

    def job():
        t0 = time.time()
        out = dfac.compute_templates(sets["train"], extras["train"]["theta"], extras["train"]["phi"], log=log)
        stored = torch.load(paths.PION_RESULTS_DIR / "abar.pt", map_location="cpu",
                            weights_only=False).double().numpy()
        front = torch.load(paths.M0_DIR / "abar_front.pt", map_location="cpu", weights_only=False).double().numpy()
        for k in dfac.NEW_TEMPLATES:
            train_save = tdir / f"{k}.pt"
            L.train.atomic_save(torch.tensor(out[k], dtype=torch.float32), train_save)
        tmp = tdir / f".templates_all.tmp.npz"
        np.savez_compressed(tmp, **out)
        tmp.replace(tdir / "templates_all.npz")
        theta = np.asarray(extras["train"]["theta"], dtype=np.float64)
        tmp = tdir / ".train_theta.tmp.npy"
        np.save(tmp, theta)
        tmp.replace(dfac.theta_file())
        meta = {"definition": {"back_corrected": "mean_e A_front&back / mean_e 1[z' < zb_e]",
                               "full_box": "mean_e A_front&back&side / mean_e 1[z' < zb_e]"},
                "events": f"training split (sfm.data.load_sets('train')), {len(sets['train'])} events",
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
                "sha256": {k: provenance.sha256_file(tdir / f"{k}.pt") for k in dfac.NEW_TEMPLATES},
                "train_theta_sha256": provenance.sha256_array(theta),
                "machine_id": L.MACHINE_ID, "machine_tag": L.MACHINE.to_dict(), "seconds": time.time() - t0}
        L.write_json(meta, f)
        log(f"templates: fb vs stored abar {json.dumps(meta['fb_vs_stored_abar'])}; front vs abar_front "
            f"{json.dumps(meta['front_vs_abar_front'])}")

    L.on_gpu("templates", job)


def phase_parity():
    row = L.registered_row("A0_init")
    if row is None:
        sets, extras = L.data_cpu()

        def job():
            dev = L.common.DEVICE
            val = registry.ValidationSet(sets["val"].to(dev), extras["val"]["event_number"])
            L.train.seed_everything(0)
            model = L.models.build("component", L.build_context(dev, None, 0), variant="legacy_centre")
            info = getattr(model, "init_info", {})
            return registry.evaluate(model, "A0_init", L.stage(), val,
                                     meta={"seed": 0, "k": int(model.n_params()), "rung": "A0", "kl_map":
                                           info.get("kl_map"), "init": info, "smoke": L.CTX.smoke,
                                           "config": {"factory": "component", "kwargs": {"variant": "legacy_centre"},
                                                      "step": 0},
                                           "trained_on": {"machine_id": L.MACHINE_ID, "source": "step 0 (untrained)"},
                                           "notes": "harness parity: A0 at step 0 (legacy coefficients)"})

        row = L.on_gpu("parity A0_init", job)
    a0, f = L.tagged("A0_init"), L.formula_array()
    p = provenance.paired_delta(a0, f)
    d = provenance.paired_difference(a0, f)
    max_abs = float(np.abs(d).max())
    ok = bool(max_abs <= PARITY_EVENT_TOL and abs(p.mean) <= PARITY_MEAN_TOL)
    note_decision("parity", {
        "criterion": "A0 at step 0 equals this machine's formula reference array within 1e-4 per event and 1e-6 on "
                     "the mean (PLAN 13, A6 revised point 4; replaces the laptop formula mean to 1e-5)",
        "A0_init_val_mean": row["val_mean"], "formula_array_mean": float(np.mean(f.values, dtype=np.float64)),
        "paired": L.pd(p), "max_abs_event": max_abs, "n_events_above_1e-4": int((np.abs(d) > PARITY_EVENT_TOL).sum()),
        "formula_array_sha256": f.sha256, "A0_init_sha256": a0.sha256, "stream_id": L.live_stream_id(), "passed": ok})
    log(f"parity: A0 at step 0 - formula array = {p.mean:+.3e} +- {p.se:.1e} (n {p.n}), max |event| {max_abs:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}")
    if not ok and L.CTX.smoke:
        log("SMOKE MODE: the parity stop is recorded but not enforced")
    elif not ok:
        raise SystemExit("STOP: harness parity failed (A0 at step 0 differs from this machine's formula array by more "
                         "than 1e-4 in an event or 1e-6 on the mean)")


def _plr_decide(names: dict) -> tuple:
    """{lr: run name} -> (winner lr, table). Lowest last-iterate validation mean wins; within 0.5e-4 the smaller lr
    (paired deltas against the lowest, on this machine's registered arrays). A pilot without a registry row (diverged
    or blocked) has no finite last-iterate validation mean and loses (review ruling on C1: [within] the P-lr rule;
    user decision of 25 Nov 2026). Returns (None, table) if no pilot finished."""
    tab = {}
    lost = {lr: n for lr, n in names.items() if L.registered_row(n) is None}
    for lr, n in lost.items():
        dv = L.diverged_marker(n)
        tab[f"{lr:g}"] = {"run": n, "status": ("diverged" if dv else "no registry row (blocked or missing)"),
                          "within": False, "loses": "no finite last-iterate validation mean (P-lr rule)"}
    names = {lr: n for lr, n in names.items() if lr not in lost}
    if not names:
        return None, tab
    means = {lr: float(np.mean(L.tagged(n).values, dtype=np.float64)) for lr, n in names.items()}
    best = min(means, key=lambda lr: (means[lr], lr))
    for lr, n in names.items():
        p = provenance.paired_delta(L.tagged(n), L.tagged(names[best]))
        tab[f"{lr:g}"] = {"run": n, "val_mean": means[lr], "vs_lowest": L.pd(p), "within": bool(p.mean <= PLR_TIE)}
    win = min(lr for lr in names if tab[f"{lr:g}"]["within"])
    return win, tab


def phase_plr():
    if not decisions().get("parity", {}).get("passed") and not (L.CTX.smoke and decisions().get("parity")):
        raise SystemExit("parity not established; run the parity phase first")
    poly, net = plr_specs()
    d = L.read_json(L.plr_file(), {}) or {}
    for sp in poly:
        L.run_spec(sp)
    lr_poly, tab = _plr_decide({sp.lr: sp.name for sp in poly})
    if lr_poly is None:
        raise SystemExit("STOP: every polynomial learning-rate pilot diverged; no polynomial run can be configured")
    d.update({"poly": tab, "lr_poly": lr_poly, "rule": "lowest last-iterate validation mean; within 0.5e-4 the "
                                                        "smaller lr (paired deltas, this machine)"})
    L.write_json(d, L.plr_file())
    log(f"P-lr poly -> lr_poly = {lr_poly}")
    L.run_spec(next(s for s in stage0_specs() if s.name == "A2s_s0"))       # the base the net pilot is fitted to
    for sp in net:
        L.run_spec(sp)
    lr_net, tab = _plr_decide({sp.lr: sp.name for sp in net})
    d.update({"net": tab, "lr_net": lr_net})
    if lr_net is None:
        d["lr_net_undecided"] = "every network pilot diverged or is blocked: the network runs are blocked"
    L.write_json(d, L.plr_file())
    log(f"P-lr net -> lr_net = {lr_net}")


def phase_stage0():
    specs = stage0_specs()
    for sp in specs[:2]:                                       # A0 seeds 0, 1 and the drift rule
        L.run_spec(sp)
    fin = [sp for sp in specs[:2] if L.registered_row(sp.name) is not None]
    drift = {sp.name: L.pd(L.paired(sp.name, "formula")) for sp in fin}
    stop = any(abs(v["mean"]) > DRIFT_STOP for v in drift.values())
    rec = {"runs": drift, "rule": "|paired(A0, this machine's formula array)| > 2e-4: stop", "stop": stop,
           "enforced": not L.CTX.smoke}
    if len(fin) < 2:                    # a diverged control: recorded, independent runs continue (PLAN 17.11)
        rec["not_evaluable"] = {sp.name: "diverged - awaiting diagnosis" for sp in specs[:2] if sp not in fin}
    note_decision("A0_drift", rec)
    log(f"A0 drift: {json.dumps({k: round(v['mean'] * 1e4, 3) for k, v in drift.items()})} (1e-4) -> "
        f"{'STOP' if stop else 'continue'}")
    if stop and L.CTX.smoke:
        log("SMOKE MODE: the drift stop is recorded but not enforced (shortened fits are not converged)")
    elif stop:
        raise SystemExit("STOP: A0 drift rule (|A0 - formula| > 2e-4): harness or optimum differs")
    for sp in specs[2:]:
        L.run_spec(sp)


def phase_stage1p():
    poly, net = primary_specs()
    for sp in poly:
        L.run_spec(sp)
    for sp in net:
        L.run_spec(sp)
    out = {}
    for sp in poly:                     # convergence rule; reruns only for the pre-registered (poly) primaries
        if L.done_marker(sp.done_file) is None:                    # diverged or blocked (PLAN 17.11)
            out[sp.name] = {"converged": None, "status": _status_text(sp.name)}
            continue
        ok, mv = converged(sp.name)
        out[sp.name] = {"converged": ok, "moved_last_window": mv, "window_steps": L.convergence_window()}
        if ok is False:
            comp2, sp2 = rerun_specs(sp)
            L.run_spec(comp2)
            r2 = L.run_spec(sp2)
            if L.status_of(r2) != "done":
                out[sp.name]["rerun"] = {"name": sp2.name, "comparator": comp2.name, "status": _status_text(sp2.name)}
                continue
            ok2, mv2 = converged(sp2.name)
            out[sp.name]["rerun"] = {"name": sp2.name, "comparator": comp2.name, "converged": ok2,
                                     "moved_last_window": mv2}
    note_decision("convergence_primaries", out)


def phase_stage1x():
    seed1 = {}
    for r in EXPLORATORY:
        sp = exploratory_spec(r, 0)
        res = L.run_spec(sp)
        if L.status_of(res) != "done":                 # diverged or blocked: recorded, never omitted (PLAN 17.11)
            seed1[r] = {"vs": sp.comparator, "status": _status_text(sp.name), "seed1_run": False}
            log(f"{sp.name}: {seed1[r]['status']} - no second seed; the batch continues")
            continue
        p = L.paired(sp.name, sp.comparator)
        ke = {"vs": sp.comparator, "paired": L.pd(p), "keep": keep_rule(p), "equal_seed0": equal_rule_one(p)}
        log(f"{sp.name}: Delta vs {sp.comparator} = {p.mean * 1e4:+.2f} +- {p.se * 1e4:.2f} (1e-4, n {p.n}); "
            f"keep {ke['keep']}, equal-NLL (seed 0) {ke['equal_seed0']}")
        if ke["keep"] or ke["equal_seed0"]:
            L.run_spec(exploratory_spec(r, 1))
        seed1[r] = {**ke, "seed1_run": bool(ke["keep"] or ke["equal_seed0"])}
    note_decision("exploratory_seed1", seed1)
    note_decision("dropped", [])
    note_decision("budget_rule", "not applied: the lists of AUDIT 4.1 run in full on the machine of record "
                                 "(PLAN 13, A6 revised point 6; PLAN 16.5 recommendation R1). No run was dropped.")


def _status_text(name: str) -> str:
    """'diverged', 'diverged - awaiting diagnosis', 'failed (diagnosed)', 'blocked by <run>' or 'missing'."""
    dv = L.diverged_marker(name)
    if dv is not None:
        if dv.get("cls") in L.EXPLORATORY_CLASSES or dv.get("cls") == "pilot":
            return "diverged"
        dg = L.diagnosis(name)
        if dg and dg.get("valid"):
            return "failed (diagnosed)" if dg["verdict"] == "failed" else f"diverged - diagnosis: {dg['verdict']}"
        return "diverged - awaiting diagnosis"
    b = (L.decisions().get("blocked") or {}).get(name)
    if b:
        return f"blocked by {b['blocked_by']}"
    return "done" if L.done_marker(L.out_dir() / name / "done.json") is not None else "missing"


# ---------------------------------------------------------------------------------------------------- guards phase ----
def stage1_comparisons() -> list:
    """(run, comparator, rung, kind) of every Stage-1 comparison that exists (registered) on this machine. A network
    run superseded by its warm-start re-run (PLAN 17.11) is replaced by the re-run, in both places."""
    poly, net = primary_specs()
    E = L.effective
    cand = [(sp.name, sp.comparator, sp.rung, "primary") for sp in poly]
    cand += [(E(sp.name), E(sp.comparator), sp.rung, "primary-net") for sp in net]
    for r in EXPLORATORY:
        for s in (0, 1):
            sp = exploratory_spec(r, s)
            cand.append((sp.name, sp.comparator, r, "exploratory"))
    for sp in poly:
        cand.append((sp.name + "_x2", sp.comparator + "_x2", sp.rung, "rerun"))
    return [c for c in cand if L.registered_row(c[0]) is not None and L.registered_row(c[1]) is not None]


def guard_targets() -> tuple:
    """Runs that get guard columns: every run of a rung that passes the keep rule or the equal-NLL rule (one seed) on
    any of its registered seeds, and the comparator of each such run (same seed). Smoke: at most 2 rungs, and the
    first rung of the list if none passes (so that the code path runs)."""
    comps = stage1_comparisons()
    rung_pass = {}
    for run, comp, rung, kind in comps:
        key = (rung, kind == "primary-net")
        p = L.paired(run, comp)
        rung_pass[key] = rung_pass.get(key, False) or keep_rule(p) or equal_rule_one(p)
    keys = [k for k, v in rung_pass.items() if v]
    if L.CTX.smoke:
        keys = keys[:L.SMOKE_MAX_GUARD_RUNGS] or ([next(iter(rung_pass))] if rung_pass else [])
    pairs = [(run, comp) for run, comp, rung, kind in comps if (rung, kind == "primary-net") in keys]
    targets = []
    for run, comp in pairs:
        for n in (run, comp):
            if n not in targets:
                targets.append(n)
    return targets, pairs, {f"{k[0]}{' (net)' if k[1] else ''}": v for k, v in rung_pass.items()}


def _g_region_masks(val_cpu) -> dict:
    """Per-hit masks of the G-region classes, by the hit centre (PLAN 6 / 13; the partition of
    results/m1_audit/checks/c_gap.py): deep z' >= 1221 mm; border = last ring (r >= 700 (31/32)^1.7 mm) or the first /
    last z-bin; low_abar = stored acceptance Abar <= 0.3 in the hit-centre cell; flagged = their union."""
    C = L.common
    xyz = val_cpu.xyz.numpy().astype(np.float64)
    z, r = xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])
    ze = C.Z_EDGES.cpu().numpy().astype(np.float64)
    re = C.R_EDGES.cpu().numpy().astype(np.float64)
    iz = np.clip(np.searchsorted(ze, z, side="right") - 1, 0, C.N_Z - 1)
    ir = np.clip(np.searchsorted(re, r, side="right") - 1, 0, C.N_R - 1)
    abar = dfac.acceptance_template("stored", "cpu").double().numpy()
    deep = z >= 1221.0
    ring = ir >= C.N_R - 1
    zbin = (iz == 0) | (iz == C.N_Z - 1)
    low = abar[iz, ir] <= 0.3
    flagged = deep | ring | zbin | low
    return {"deep": deep, "border_ring": ring, "border_zbin": zbin, "low_abar": low, "flagged": flagged,
            "inner": ~flagged}


_GR_STATIC: dict = {}


def _g_region_static(val_cpu) -> tuple:
    """(N, event index of every hit, class masks by hit centre): they depend only on the validation hits and the
    stored template, not on the model, so they are computed once per process (the guards phase evaluates many runs
    on the same validation split). Before, every run recomputed them, with a float64 copy of all hit coordinates."""
    key = (id(val_cpu), len(val_cpu), int(val_cpu.off[-1]))
    if key not in _GR_STATIC:
        _GR_STATIC.clear()
        N = len(val_cpu)
        off = val_cpu.off.numpy()
        _GR_STATIC[key] = (N, np.repeat(np.arange(N), np.diff(off)), _g_region_masks(val_cpu))
    return _GR_STATIC[key]


def g_region_compute(model, val, val_cpu, timing: Optional[dict] = None, run: str = "") -> dict:
    """The G-region arrays of one model, without I/O: per class the per-event hit-term contribution
    np.bincount(ev, weights=-hit * mask) (float64, the expression of the original code, so bit-identical), and the
    normalisation term weight_sum * log_mass. Returns {"arrays": {class: array, "normalisation": array}, "hit",
    "parts_nll"}."""
    model.eval()
    rec = timing if timing is not None else {}
    with L.PartTimer("G-region pass (GPU)", rec, run):
        parts = metric.test_nll(model, val, return_parts=True, deterministic_sum=True)
    with L.PartTimer("G-region host work", rec, run):
        hit = parts["hit"][0].astype(np.float64)
        norm = (parts["weight_sum"].astype(np.float64) * parts["log_mass"][0].astype(np.float64))
        N, ev, masks = _g_region_static(val_cpu)
        arrays = {cls: np.bincount(ev, weights=-hit * m, minlength=N) for cls, m in masks.items()}
        arrays["normalisation"] = norm
    return {"arrays": arrays, "hit": hit, "parts_nll": parts["nll"][0]}


def g_region_arrays(name: str, model, val, val_cpu, timing: Optional[dict] = None, comp: Optional[dict] = None) -> dict:
    """Per-event NLL decomposition of one run on the registry's fold points (K = 64, seed 12345, batch 128):
    hit-term contribution -sum_{i in e, class} omega_i ln g_i per class and the normalisation term
    weight_sum_e ln M_e, saved as tagged arrays (subset "g_region:<class>"); paired across runs by the report.
    ``comp``: the result of g_region_compute for this model, if the caller has it already."""
    base = L.live()["points"]
    sdir = L.scores_dir(name)                          # where the registry put this run's arrays
    marker = sdir / "g_region.json"
    if marker.exists():
        return L.read_json(marker)
    if comp is None:
        comp = g_region_compute(model, val, val_cpu, timing=timing, run=name)
    hit, norm, parts_nll = comp["hit"], comp["arrays"]["normalisation"], comp["parts_nll"]
    N, ev, masks = _g_region_static(val_cpu)
    recon = np.bincount(ev, weights=-hit, minlength=N) + norm
    reg = L.tagged(name).values.astype(np.float64)
    identity = {"parts_nll_vs_parts_sum_max_abs": float(np.abs(recon - parts_nll).max()),
                "parts_nll_vs_registered_k64": provenance.tolerance_check(parts_nll, reg, top=3)}
    if not identity["parts_nll_vs_registered_k64"].get("strict_pass") or \
            identity["parts_nll_vs_parts_sum_max_abs"] > IDENTITY_TOL:            # PLAN 23 item 13: hard stop
        tc = identity["parts_nll_vs_registered_k64"]
        raise SystemExit(f"STOP (PLAN 23 item 13, guard check): the G-region pass of {name} does not reproduce its "
                         f"registered array (strict {tc.get('strict_pass')}, max {tc.get('max_abs')}) or its own sum "
                         f"({identity['parts_nll_vs_parts_sum_max_abs']:.2e}); nothing written; waiting for the user")
    out = {"name": name, "arrays": {}, "n_hits": int(len(hit))}
    for cls in masks:
        a = comp["arrays"][cls]
        pts = provenance.points_tag(grid=base["grid"], K=base["K"], split_fingerprint=base["split_fingerprint"],
                                    seed=base["seed"], batch=base["batch"], subset=f"g_region:{cls}")
        ta = provenance.save_tagged(sdir / f"g_region_{cls}.npy", a, pts, L.live_machine(),
                                    label=f"{L.stage()}/{name}:g_region:{cls}")
        out["arrays"][cls] = {"path": paths.portable(ta.path), "sha256": ta.sha256}
    pts = provenance.points_tag(grid=base["grid"], K=base["K"], split_fingerprint=base["split_fingerprint"],
                                seed=base["seed"], batch=base["batch"], subset="g_region:normalisation")
    ta = provenance.save_tagged(sdir / "g_region_normalisation.npy", norm, pts, L.live_machine(),
                                label=f"{L.stage()}/{name}:g_region:normalisation")
    out["arrays"]["normalisation"] = {"path": paths.portable(ta.path), "sha256": ta.sha256}
    out["identity_check"] = identity
    L.write_json(out, marker)
    return out


G_STAR = "flagged_star"
IDENTITY_TOL = 1e-4                 # nat per event: parts sum vs parts NLL (the H200 markers: <= 1.1e-5)
STAR_TOL = 1e-9                     # nat per event: flagged = flagged* + (low_abar and not flagged*), float64 sums
G_STAR_DEFINITION = ("flagged* = deep (z' >= 1221 mm) or last ring or first / last z-bin, by the hit centre: the hits "
                     "flagged for a reason other than low acceptance alone (REVIEW 10.3 item 3, R6); the waiver line "
                     "of G-region for the class low_abar applies the rule of G-region with flagged* in place of "
                     "flagged")


def g_region_star_arrays(name: str, comp: dict, val_cpu) -> dict:
    """REVIEW 10.5 R6: the per-event hit-term contribution of the flagged* cells (G_STAR_DEFINITION) from the same
    per-hit values as the G-region classes, saved as a new tagged array (subset "g_region:flagged_star") with its own
    marker g_region_star.json; the existing arrays and marker are not touched. The marker records whether the
    flagged class recomputed in this pass equals the stored one bit for bit, and the identity
    flagged = flagged* + (low_abar and not flagged*)."""
    base = L.live()["points"]
    sdir = L.scores_dir(name)
    marker = sdir / "g_region_star.json"
    if marker.exists():
        return L.read_json(marker)
    N, ev, masks = _g_region_static(val_cpu)
    star = masks["deep"] | masks["border_ring"] | masks["border_zbin"]
    hit = comp["hit"]
    a = np.bincount(ev, weights=-hit * star, minlength=N)
    low_only = np.bincount(ev, weights=-hit * (masks["low_abar"] & ~star), minlength=N)
    stored = None
    gr = L.read_json(sdir / "g_region.json")
    if gr:
        st = provenance.load_tagged(sdir / "g_region_flagged.npy", expect_machine_id=L.live_machine().id)
        stored = bool(st.sha256 == gr["arrays"]["flagged"]["sha256"]
                      and np.array_equal(st.values, comp["arrays"]["flagged"]))
    resid = float(np.abs(comp["arrays"]["flagged"] - a - low_only).max())
    if stored is False or resid > STAR_TOL:                                       # PLAN 23 item 13: hard stop
        raise SystemExit(f"STOP (PLAN 23 item 13, guard check): {name}: the flagged class recomputed for the flagged* "
                         f"array is not the stored one bit for bit ({stored}) or the identity fails ({resid:.2e}); "
                         "nothing written; waiting for the user")
    pts = provenance.points_tag(grid=base["grid"], K=base["K"], split_fingerprint=base["split_fingerprint"],
                                seed=base["seed"], batch=base["batch"], subset=f"g_region:{G_STAR}")
    ta = provenance.save_tagged(sdir / f"g_region_{G_STAR}.npy", a, pts, L.live_machine(),
                                label=f"{L.stage()}/{name}:g_region:{G_STAR}")
    out = {"name": name, "class": G_STAR, "definition": G_STAR_DEFINITION,
           "array": {"path": paths.portable(ta.path), "sha256": ta.sha256},
           "flagged_recomputed_bit_identical_to_stored": stored,
           "max_abs_flagged_minus_star_minus_low_only": resid}
    L.write_json(out, marker)
    return out


def fisher_k_a6b(name: str, model, tr) -> dict:
    """Observed Fisher information of the Burr-in-depth shape k of A6b (raw row sh_rk, degree 0: one coefficient) on
    the TRAINING events: I = d^2/dc^2 sum_e nll_e at fixed fold points (K = 16, seed 12345, batch 128, inside
    deterministic()), central second difference with h = 0.01 in the raw coefficient; SE(c) = 1/sqrt(I)."""
    f = L.out_dir() / "fisher" / f"{name}_sh_rk.json"
    if f.exists():
        return L.read_json(f)
    base = model.base if hasattr(model, "base") else model
    prov = base.provider
    i = prov.index("sh_rk")
    c0 = float(prov.coef[i, 0])
    h = 0.01
    tot = {}
    model.eval()
    for s in (-1, 0, 1):
        with torch.no_grad():
            prov.coef[i, 0] = c0 + s * h
        tot[s] = float(np.sum(metric.test_nll(model, tr, K=16, deterministic_sum=True), dtype=np.float64))
    with torch.no_grad():
        prov.coef[i, 0] = c0
    info = (tot[1] - 2 * tot[0] + tot[-1]) / h ** 2
    out = {"run": name, "row": "sh_rk", "coefficient": c0, "h": h, "sum_nll": {str(k): v for k, v in tot.items()},
           "observed_information": info, "se_coefficient": (1.0 / math.sqrt(info)) if info > 0 else None,
           "n_train_events": len(tr), "definition": fisher_k_a6b.__doc__}
    L.write_json(out, f)
    return out


@contextlib.contextmanager
def _time_registry_parts(rec: dict, name: str):
    """Time the two guard parts inside registry.evaluate (K = 1024 + bulk; mass monitors) by wrapping the two module
    functions for the duration of the call. Timing only: the wrapped functions are called unchanged."""
    orig_g, orig_m = registry._guard_arrays, registry._mass

    def g(*a, **k):
        with L.PartTimer("guard K=1024 + bulk", rec, name):
            return orig_g(*a, **k)

    def m(*a, **k):
        with L.PartTimer("guard mass monitors", rec, name):
            return orig_m(*a, **k)
    registry._guard_arrays, registry._mass = g, m
    try:
        yield
    finally:
        registry._guard_arrays, registry._mass = orig_g, orig_m


def guard_run(name: str, sets, extras) -> Optional[dict]:
    """Guard columns (registry.evaluate(guards=True)) and G-region arrays of one run, if missing, with a timing record
    per part (STATUS R5): wall and CPU seconds, page faults, load, available memory, swap, pressure-stall figures.
    The record goes to the log and to guard_timing_<UTC>.json next to the run's arrays (not to decisions.json, so
    results.json does not depend on it); a part longer than 60 s is logged as a WARNING. Timing fields only: no array
    and no registry number changes."""
    need_row = L.registered_row(name, need_guards=True) is None
    need_gr = not (L.scores_dir(name) / "g_region.json").exists()
    need_star = not (L.scores_dir(name) / "g_region_star.json").exists()        # REVIEW 10.5 R6 (new arrays only)
    if not (need_row or need_gr or need_star):
        return None

    def job():
        rec = {}
        dev = L.common.DEVICE
        va = sets["val"].to(dev)
        val = registry.ValidationSet(va, extras["val"]["event_number"])
        model = L.rebuild(name, dev)
        mp = L.out_dir() / name / "model.pt"
        if need_row:
            row0 = L.registered_row(name)
            with L.PartTimer("registry evaluate (total)", rec, name), _time_registry_parts(rec, name):
                registry.evaluate(model, name, L.stage(), val,
                                  meta={"k": row0["k"], "seed": row0.get("seed"), "checkpoint": str(mp),
                                        "config_hash": row0.get("config_hash"), "smoke": L.CTX.smoke,
                                        "trained_on": {"machine_id": L.MACHINE_ID, "source": "trained on this machine"},
                                        "notes": "guard columns (G-K, G-bulk, G-mass) of a Track A run"},
                                  guards=True)
            inner = sum(rec[k]["wall_s"] for k in ("guard K=1024 + bulk", "guard mass monitors") if k in rec)
            rec["K=64 metric of record and row (total minus guard parts)"] = {
                "wall_s": round(rec["registry evaluate (total)"]["wall_s"] - inner, 3)}
        if need_gr or need_star:
            comp = g_region_compute(model, va, sets["val"], timing=rec, run=name)
            if need_gr:
                g_region_arrays(name, model, va, sets["val"], comp=comp)
            if need_star:
                g_region_star_arrays(name, comp, sets["val"])
            del comp
        del model, va, val
        return rec

    rec = L.on_gpu(f"guards {name}", job)
    slow = {k: v["wall_s"] for k, v in rec.items() if v.get("wall_s", 0) > L.SLOW_PART_S}
    log(f"[{name}] guard timing: " + ", ".join(f"{k} {v['wall_s']:.1f} s" for k, v in rec.items()))
    f = L.scores_dir(name) / f"guard_timing_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if not f.exists():
        L.write_json({"run": name, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "parts": rec,
                      "slow_parts": slow, "threshold_s": L.SLOW_PART_S,
                      "note": "timing fields only (STATUS R5); no array or registry number depends on them"}, f)
    return rec


def phase_guards():
    targets, pairs, passing = guard_targets()
    note_decision("guard_targets", {"rungs_passing_keep_or_equal": passing, "runs": targets, "pairs": pairs,
                                    "smoke_limited": L.CTX.smoke})
    log(f"guards: {len(targets)} runs ({targets})")
    sets, extras = L.data_cpu()
    for name in targets:
        guard_run(name, sets, extras)
    a6b = "A6b_s0"
    if L.done_marker(L.out_dir() / a6b / "done.json") is not None and \
            not (L.out_dir() / "fisher" / f"{a6b}_sh_rk.json").exists():
        def fjob():
            dev = L.common.DEVICE
            tr = sets["train"].to(dev)
            out = fisher_k_a6b(a6b, L.rebuild(a6b, dev), tr)
            del tr
            return out
        res = L.on_gpu("fisher A6b", fjob)
        log(f"A6b: observed information of sh_rk = {res['observed_information']:.4g}, SE {res['se_coefficient']}")


# ------------------------------------------------------------------------- additions of 25 Nov 2026 (PLAN 17.11) ----
A2C_NOTE = "control (added by the user on 25 Nov 2026, exploratory, outside the six primaries)"
A2C_MATCHED = ("A4", "A5", "A7", "A8", "A9", "A4t", "A5h", "A6i", "A6l", "A6b", "A7s", "A7l", "A8c", "A78", "A9r",
               "A3f", "A3c", "A3x", "A12")       # start = the fitted A2s of the same seed, T_A with 4000 steps, lr_poly
A2C_UNMATCHED = {"A10": 6000, "A10-2": 6000, "A10-4": 6000, "A10-5": 6000, "A10g": 6000}   # init_from_target, 6000
A2C_GROUPS = {
    "matched (A2c)": "component rungs that START FROM the fitted A2s of the same seed (copy by name, or "
                     "init_shower_long on it): " + ", ".join(A2C_MATCHED) + "; T_A with 4000 steps, the frozen "
                     "lr_poly, train seed = the rung's seed, fresh optimiser and cosine cycle",
    "not matched: mixtures": "A10, A10-2, A10-4, A10-5, A10g: 6000 steps and initialised by init_from_target on "
                             "the MAP of A2s (function space, a new structure), not continued from its weights. "
                             "Proposal (awaiting the user): 'A2c6k' = A2c with 6000 steps, seeds 0 and 1",
    "not matched: convergence reruns": "<rung>_s<seed>_x2: 8000 / 12000 steps from the original A2s; their "
                                       "comparator A2s_s<seed>_x2 is itself a rerun. Proposal (awaiting the user): "
                                       "'A2c_x2' = A2c with doubled steps, only for seeds with a rerun",
    "not applicable": "network runs (fitted to their own polynomial run) and Stage 0"}


def a2c_specs() -> list:
    """A2c (user decision of 25 Nov 2026, PLAN 17.11; review B6): the A2s structure started from the fitted A2s of
    the same seed and trained with exactly the schedule of the rungs that start from it (T_A, 4000 steps, the frozen
    lr_poly, the same seed). A control: no pre-registered rule consumes it."""
    return [Spec(run_name("A2c", s), "A2c", "component", dict(variant="survival_exact", start=a2s(s)), s, "poly",
                 "control", comparator=run_name("A2s", s), steps=4000,
                 notes=f"A2c {A2C_NOTE}: A2s continued unchanged from A2s_s{s} with the schedule of the rungs that "
                       "start from the fitted A2s") for s in (0, 1)]


def phase_a2c():
    out = {}
    for sp in a2c_specs():
        res = L.run_spec(sp)
        out[sp.name] = {"status": _status_text(sp.name)}
        if L.status_of(res) == "done":
            out[sp.name]["vs_A2s"] = L.pd(L.paired(sp.name, sp.comparator))
    note_decision("a2c", {"runs": out, "note": A2C_NOTE, "groups": A2C_GROUPS,
                          "rule": "none: keep, replication and equal NLL stay against A2s as registered"})


def _net_runs_executed() -> list:
    """Network runs of this stage finished with the fit_to before version 3 (no fit_to_version >= 3 in init.json),
    in the order of the run list; warm-start re-runs (<name>_ws) are not checked again."""
    order = [s.name for s in plr_specs()[1]] + [s.name for s in stage0_specs() if s.provider == "net"] + \
        [s.name for s in primary_specs()[1]]
    out = []
    for n in order:
        d = L.done_marker(L.out_dir() / n / "done.json")
        if d is None:
            continue
        ft = ((L.read_json(L.out_dir() / n / "init.json", {}) or {}).get("init_info") or {}).get("fit_to") or {}
        if (ft.get("fit_to_version") or 2) < 3:
            out.append(n)
    return out


def target_columns(done: dict) -> dict:
    """Per-column statistics of the targets of a network warm start, rebuilt on the CPU from the source polynomial
    run (net_from) or the legacy provider (A1): float64 standard deviation on default_lnp(48) with uniform weights
    (fit_to's weights=None), constant columns by fit_to's 2^-20 rule, below the floor = varying with std < 1e-2."""
    from sfm.families.components import default_lnp
    from sfm.models.paramnet import ParamNet
    kw = dict(done.get("build_kwargs") or done["spec"]["kwargs"])
    ctx = L.build_context("cpu", None, 0, skip_init=True)
    if kw.get("net_from"):
        tgt = L.models.load_run(kw["net_from"], ctx)
    else:
        kw2 = {k: v for k, v in kw.items() if k not in ("provider", "fit_to_steps", "fit_to_lbfgs", "net_from",
                                                         "fit_to_scale_floor")}
        tgt = L.models.build(done["spec"]["factory"], ctx, **kw2)
    base = tgt.base if isinstance(tgt, dfac.LastRing) else tgt
    names_net, _ = dfac.net_name_split(base)
    poly = base.provider
    with torch.no_grad():
        t = poly(default_lnp(48, device="cpu")).double()
    cols = {}
    for n in names_net:
        c = t[:, poly.names.index(n)]
        mu, sd = float(c.mean()), float(c.std(unbiased=False))
        const = sd <= ParamNet.SCALE_FLOOR_REL * max(1.0, abs(mu))
        cols[n] = {"std": sd, "mean": mu, "constant": bool(const),
                   "below_floor": bool((not const) and sd < ParamNet.SCALE_FLOOR_ABS)}
    return {"source": kw.get("net_from") or "legacy provider (A1)", "columns": cols,
            "below_floor": [n for n, c in cols.items() if c["below_floor"]],
            "constant": [n for n, c in cols.items() if c["constant"]]}


WARMSTART_RULE = ("PLAN 17.11 (fixed before any network result of the job was read): a network run is re-run if and "
                  "only if (a) one of its varying target columns lies below the floor, so that the fix changes its "
                  "warm start, or (b) its recorded kl_map at initialisation exceeds 1e-4 nat. Both are properties of "
                  "the initialisation, not of the outcome. A re-run gets a new run name; the original is kept and "
                  "labelled as superseded.")
KL_AFFECTED = 1e-4


def warmstart_affected(target_cols: dict, kl_map) -> tuple:
    """PLAN 17.11, applied verbatim: (rule a, rule b, re-run). (a) a varying target column lies below the floor
    (target_columns: 'below_floor' = not constant and std < 1e-2); (b) the RECORDED kl_map at initialisation exceeds
    1e-4 nat. A missing kl_map counts as not exceeding (it is reported)."""
    a = bool(target_cols.get("below_floor"))
    b = bool(kl_map is not None and float(kl_map) > KL_AFFECTED)
    return a, b, bool(a or b)


def phase_warmstart_check():
    """Trains nothing. For every network run executed with fit_to version < 3: target columns (std, below the floor),
    the recorded kl_map, the step-0 parity of the OLD warm start (rebuilt deterministically: same seed, same build
    order, fit_to version 2) against the source's registered array, and the PLAN 17.11 decision."""
    from sfm.models.paramnet import ParamNet
    names = _net_runs_executed()
    sets, extras = L.data_cpu()
    runs, rerun = {}, []
    prior = (decisions().get("warmstart_check") or {}).get("runs") or {}
    for n in names:
        d = L.done_marker(L.out_dir() / n / "done.json")
        init = L.read_json(L.out_dir() / n / "init.json", {}) or {}
        kl = (init.get("init_info") or {}).get("kl_map", d.get("kl_map"))
        tc = target_columns(d)
        kw = dict(d.get("build_kwargs") or d["spec"]["kwargs"])
        seed = int(d["spec"]["seed"])
        old = _reusable_check_record(n, prior.get(n), tc, kl)
        if old is not None:                 # review C8-3: re-issue - the verified record is reused, nothing rebuilt
            runs[n] = old
            if old["rerun"]:
                rerun.append(n)
            log(f"warmstart-check {n}: record of the first check reused (step-0 array verified by its tag); "
                f"{'RE-RUN' if old['rerun'] else 'keep'}")
            continue

        def job(n=n, kw=kw, seed=seed, kl=kl):
            dev = L.common.DEVICE
            tr, va = sets["train"].to(dev), sets["val"].to(dev)
            val = registry.ValidationSet(va, extras["val"]["event_number"])
            registry.check_validation(val)
            L.train.seed_everything(seed)                  # the order of run_spec: seed, context, build
            ctx = L.build_context(dev, tr, seed, skip_init=False)
            m = L.models.build(d["spec"]["factory"], ctx, **{**kw, "fit_to_scale_floor": None})
            kl2 = (getattr(m, "init_info", {}) or {}).get("kl_map")
            par = L.step0_parity(m, n, L.warm_start_source(kw), val, subdir="step0_old_warmstart",
                                 base=L.out_dir() / "warmstart_check" / n)
            del m, tr, va, val
            return {"kl_map_rebuilt": kl2, "bit_for_bit": (kl2 == kl), "step0_parity": par}

        reb = L.on_gpu(f"warmstart-check {n}", job)
        a, b, _ = warmstart_affected(tc, kl)
        runs[n] = {"kl_map_recorded": kl, "target_columns": tc, **reb, "rule_a_below_floor": a,
                   "rule_b_kl_map": b, "rerun": bool(a or b),
                   "rebuild_note": ("rebuilt bit for bit (kl_map equal to the recorded value)" if reb["bit_for_bit"]
                                    else "NOT bit for bit (kl_map differs from the recorded value): the step-0 "
                                         "parity refers to the rebuilt warm start, the decision uses the RECORDED "
                                         "kl_map")}
        if a or b:
            rerun.append(n)
        log(f"warmstart-check {n}: kl_map {kl} (rebuilt {reb['kl_map_rebuilt']}, bit for bit {reb['bit_for_bit']}), "
            f"below floor {tc['below_floor']} -> {'RE-RUN' if (a or b) else 'keep'}")
    note_decision("warmstart_check", {"rule": WARMSTART_RULE, "floor": ParamNet.SCALE_FLOOR_ABS,
                                      "kl_threshold_nat": KL_AFFECTED, "runs": runs, "rerun": rerun,
                                      "trains_nothing": True})
    log(f"warmstart-check: {len(rerun)} of {len(names)} network runs to be re-run: {rerun}")


def _reusable_check_record(name: str, rec, tc: dict, kl) -> Optional[dict]:
    """The record of an earlier warmstart-check of this run, if its inputs are unchanged (same target columns, same
    recorded kl_map, hence the same PLAN 17.11 decision) and its step-0 array still carries its tag and hash.
    Reusing it avoids a second rebuild, which save_tagged would refuse if it were not bit-identical (review C8-3)."""
    if not rec:
        return None
    p0 = rec.get("step0_parity") or {}
    try:
        if p0.get("available"):
            f = L.out_dir() / "warmstart_check" / name / "step0_old_warmstart" / Path(p0["array"]).name
            if provenance.load_tagged(f, expect_machine_id=L.live_machine().id).sha256 != p0["sha256"]:
                return None
    except (FileNotFoundError, provenance.UntaggedArray, provenance.MixedTags):
        return None
    a, b, r = warmstart_affected(tc, kl)
    same = (rec.get("target_columns") == json.loads(json.dumps(L._jsonable(tc))) and rec.get("kl_map_recorded") == kl
            and (rec.get("rule_a_below_floor"), rec.get("rule_b_kl_map"), rec.get("rerun")) == (a, b, r))
    return rec if same else None


def _spec_of(name: str) -> Spec:
    d = L.done_marker(L.out_dir() / name / "done.json")
    return Spec(**d["spec"])


def phase_warmstart_rerun():
    """Re-runs exactly the runs listed by warmstart-check under <name>_ws (fit_to version 3, step-0 parity logged),
    never touching the originals; then whatever depends on them: if a network pilot was re-run, lr_net is decided
    again among the pilots (re-runs in place of their originals); if it changes, every other network run is re-run
    with the new rate as well (listed as dependent). Then the guards of the re-run pairs (guards phase)."""
    chk = decisions().get("warmstart_check")
    if chk is None:
        raise SystemExit("warmstart-rerun refused: run warmstart-check first")
    listed = list(chk.get("rerun") or [])
    rec = decisions().get("warmstart_rerun") or {"superseded": {}, "dependent": {}, "listed": listed}
    sup = dict(rec.get("superseded") or {})

    def in_force(n: str) -> str:
        """REVIEW 9.5 Rm1 (aligned with C8-4): a recorded re-run stands for its original only if it is registered."""
        r = sup.get(n)
        return r if (r is not None and L.registered_row(r) is not None) else n

    def rerun(name: str, lr=None, reason="listed by warmstart-check"):
        sp = _spec_of(name)
        kw = {k: v for k, v in sp.kwargs.items() if k != "fit_to_scale_floor"}     # always fit_to version 3
        new = replace(sp, name=name + "_ws", kwargs=kw, lr=(lr if lr is not None else sp.lr),
                      comparator=in_force(sp.comparator) if sp.comparator else sp.comparator,
                      notes=f"warm-start re-run of {name} (PLAN 17.11; {reason}); fit_to version 3")
        res = L.run_spec(new)
        sup[name] = new.name
        rec["superseded"] = sup
        note_decision("warmstart_rerun", rec)
        return res

    pilots = [n for n in listed if n.startswith("plr_A2n_")]
    for n in pilots:
        rerun(n)
    lr_net = (L.read_json(L.plr_file(), {}) or {}).get("lr_net")
    if pilots:
        net_p = plr_specs()[1]
        new_lr, tab = _plr_decide({sp.lr: in_force(sp.name) for sp in net_p})
        rec["lr_net_after_rerun"] = {"lr_net_before": lr_net, "lr_net_after": new_lr, "table": tab}
        note_decision("warmstart_rerun", rec)
        under_a = [p for p in pilots if ((chk.get("runs") or {}).get(p) or {}).get("rule_a_below_floor")]
        if new_lr is not None and new_lr != lr_net and not under_a:
            # PLAN 20 item 3: replicate re-runs (no pilot target below the floor) changed lr_net. The change is not
            # caused by the fix: this branch stops, the re-run pilots are not used, the rate stays, the user decides.
            rec["lr_net_after_rerun"].update({"escalated": True, "status": "awaiting the user", "not_used": pilots,
                                              "reason": "replicate re-runs of the network pilots (no target column "
                                                        "below the floor) changed lr_net"})
            note_decision("warmstart_rerun", rec)
            log(f"warmstart-rerun: lr_net would change {lr_net} -> {new_lr} on replicate re-runs (no column below the "
                "floor): ESCALATED to the user; the re-run pilots are not used and no dependent run is started")
        elif new_lr is not None and new_lr != lr_net:
            for n in _net_runs_executed():
                if n not in listed and not n.startswith("plr_A2n_"):
                    rec.setdefault("dependent", {})[n] = f"lr_net changed from {lr_net} to {new_lr}"
            lr_net = new_lr
    todo = [n for n in _net_runs_executed() if not n.startswith("plr_A2n_")
            and (n in listed or n in (rec.get("dependent") or {}))]
    lr_changed = bool((rec.get("lr_net_after_rerun") or {}).get("lr_net_after") not in
                      (None, (rec.get("lr_net_after_rerun") or {}).get("lr_net_before"))
                      and not (rec.get("lr_net_after_rerun") or {}).get("escalated"))
    for n in todo:
        rerun(n, lr=(lr_net if lr_changed else None),
              reason="listed by warmstart-check" if n in listed else rec["dependent"][n])
    rec["done"] = True
    note_decision("warmstart_rerun", rec)
    log(f"warmstart-rerun: superseded {sup}")
    phase_guards()


# ------------------------------------------------------------------------- additions of 26 Nov 2026 (PLAN 20) --------
MIXTURE_RUNGS = ("A10", "A10-2", "A10-4", "A10-5", "A10g")
CONDITIONAL_RULE = {
    "A2c_x2": "run only if a doubled-step rerun pair (<rung>_s<seed>_x2 vs A2s_s<seed>_x2) passes the keep rule or the "
              "one-seed equal-NLL rule (user decision of 26 Nov 2026, PLAN 20 item 1); then for the seeds concerned",
    "A2c6k": "run only if a mixture rung (A10, A10-2, A10-4, A10-5, A10g; any seed) passes the keep rule or the "
             "one-seed equal-NLL rule against A2s of its seed (PLAN 20 item 1); then for the seeds concerned"}


def evaluate_conditional_controls() -> dict:
    """The PLAN 20 rule evaluated on the registered Stage-1 arrays (paired deltas, this machine)."""
    out = {}
    pairs = {"A2c_x2": [(f"{r}_s{s}_x2", f"A2s_s{s}_x2", s) for r in PRIMARIES for s in (0, 1)],
             "A2c6k": [(f"{r}_s{s}", f"A2s_s{s}", s) for r in MIXTURE_RUNGS for s in (0, 1)]}
    for ctl, cand in pairs.items():
        rows, seeds = {}, []
        for run, comp, s in cand:
            if L.registered_row(run) is None or L.registered_row(comp) is None:
                continue
            p = L.paired(run, comp)
            k, e = keep_rule(p), equal_rule_one(p)
            rows[run] = {"vs": comp, "paired": L.pd(p), "keep": k, "equal_one_seed": e}
            if (k or e) and s not in seeds:
                seeds.append(s)
        out[ctl] = {"rule": CONDITIONAL_RULE[ctl], "evaluated_pairs": rows, "triggered": bool(seeds),
                    "seeds": sorted(seeds)}
    return out


def a2c_x2_specs(seeds) -> list:
    return [Spec(f"A2c_x2_s{s}", "A2c_x2", "component", dict(variant="survival_exact", start=a2s(s)), s, "poly",
                 "control", comparator=f"A2s_s{s}_x2", steps=8000,
                 notes="conditional control (PLAN 20 item 1): A2s continued for the doubled step budget of a rerun "
                       "pair") for s in seeds]


def a2c6k_specs(seeds) -> list:
    return [Spec(f"A2c6k_s{s}", "A2c6k", "component", dict(variant="survival_exact", start=a2s(s)), s, "poly",
                 "control", comparator=f"A2s_s{s}", steps=6000,
                 notes="conditional control (PLAN 20 item 1): A2s continued for the 6000-step budget of the "
                       "mixtures") for s in seeds]


def phase_conditional_controls():
    """Evaluate the PLAN 20 rule, record it, and train a control only if its condition is met."""
    out = evaluate_conditional_controls()
    note_decision("conditional_controls", out)
    for ctl, mk in (("A2c_x2", a2c_x2_specs), ("A2c6k", a2c6k_specs)):
        if out[ctl]["triggered"]:
            log(f"conditional control {ctl}: TRIGGERED for seeds {out[ctl]['seeds']}")
            for sp in mk(out[ctl]["seeds"]):
                L.run_spec(sp)
        else:
            log(f"conditional control {ctl}: not triggered (" + "; ".join(
                f"{n} {v['paired']['mean'] * 1e4:+.2f} +- {v['paired']['se'] * 1e4:.2f}"
                for n, v in out[ctl]["evaluated_pairs"].items()) + ")")


def phase_report():
    import run_m2a_report as R
    R.report(with_guards=not REPORT_WITHOUT_GUARDS)


REPORT_WITHOUT_GUARDS = False          # set by --without-guards (review C2; the "report" phase alone)


# ---------------------------------------------------------------------------------------------------- stage 2 ---------
STAGE2_MERGE = {          # rung -> component kwargs it contributes to A11 (no new mathematics; AUDIT 4.1 Stage 2)
    "A4": {"shower_long": "emg"}, "A5": {"shower_long": "gengamma", "shower_long_kw": {"soft": "free"}},
    "A4t": {"shower_long": "emg_tied"}, "A5h": {"shower_long": "gengamma", "shower_long_kw": {"soft": 2.0}},
    "A6i": {"shower_long": "invgauss", "shower_long_kw": {"soft": "free"}},
    "A6l": {"shower_long": "lognormal", "shower_long_kw": {"soft": "free"}},
    "A6b": {"shower_long": "burr", "shower_long_kw": {"soft": "free"}, "degrees": {"sh_rk": 0}},
    "A7": {"shower_lat": ["gauss", "burr", "exp"]}, "A8": {"shower_lat": ["gauss", "cauchy", "weibull"]},
    "A78": {"shower_lat": ["gauss", "burr", "weibull"]}, "A7s": {"shower_lat": ["gauss", "student", "exp"]},
    "A8c": {"shower_lat": ["weibull_core", "cauchy", "exp"]},
    "A7l": {"shower_lat": ["gauss", "cauchy", "burr"], "lat_family_kw": [{}, {}, {"c": 1.0, "init": [1.0, 20.0]}],
            "coef_shift": {"log_lam": LN20}},
    "A9": {"det_halo": True, "far": True}, "A9r": {"det_halo": True, "far": False, "logit_dh_from_far": True},
    "A3f": {"template": "front"}, "A3c": {"template": "back_corrected"}, "A3x": {"template": "full_box"},
}


def stage2_specs(kept: list) -> list:
    """A11 (seeds 0, 1, 2), A11k (k_train 64; seeds 0, 1), A11n (net; seeds 0, 1). Kept rungs must merge without
    conflict (a conflict, a mixture rung or A12 is refused: the combination is the user's decision)."""
    merged = {}
    for r in kept:
        if r not in STAGE2_MERGE:
            raise SystemExit(f"stage2: rung {r} cannot be merged into a component model (mixture or diagnostic); "
                             "decide the content of A11 by hand")
        for k, v in STAGE2_MERGE[r].items():
            if k in merged and merged[k] != v:
                raise SystemExit(f"stage2: kept rungs conflict on {k} ({merged[k]} vs {v}); decide A11 by hand")
            merged[k] = v
    out = []
    if not kept:
        for s in (0, 1):
            out.append(Spec(run_name("A11k", s), "A11k", "component", dict(variant="survival_exact"), s, "poly",
                            "stage2", comparator=run_name("A2s", s), k_train=64,
                            notes="A11k: nothing kept, A11 = A2s; the same with k_train = 64"))
        return out
    init = {"init": "shower_long"} if "shower_long" in merged else {}
    for s in (0, 1, 2):
        out.append(Spec(run_name("A11", s), "A11", "component", dict(**merged, **init, start=a2s(s)), s, "poly",
                        "stage2", comparator=run_name("A2s", s), notes=f"A11 = A2s + {kept}"))
    for s in (0, 1):
        out.append(Spec(run_name("A11k", s), "A11k", "component", dict(**merged, **init, start=a2s(s)), s, "poly",
                        "stage2", comparator=run_name("A2s", s), k_train=64, notes=f"A11k = A11 with k_train 64"))
    for s in (0, 1):
        out.append(Spec(run_name("A11n", s), "A11n", "component",
                        dict(provider="net", net_from=f"{L.stage()}/{run_name('A11', s)}"), s, "net", "stage2",
                        comparator=L.effective(run_name("A2n", s)), notes="A11n = net version of A11"))
    return out


STAGE2_APPROVAL = "stage2_approval.json"
REPLICATION_READINGS = {"literal": "kept_and_replicated", "two_se": "kept_and_replicated_alternative_2se"}
WAIVABLE = {"G-region": ("low_abar",)}            # REVIEW 10.3 item 3 / R6: the waiver line uses flagged*
LR_NET_ESCALATION_DECISIONS = ("keep_rate_of_record",)


def _nonempty(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


def check_stage2_decisions(ud: dict, where: str) -> dict:
    """Format of the user's decisions on the Stage-2 content carried by the approval file (PLAN 20; REVIEW 10.5 R1):

        "excluded": {"A9r": {"reason": "...", "date": "2026-11-26"}},
        "waivers": [{"rung": "A9", "guard": "G-region", "class": "low_abar", "basis": "...",
                     "subject_to": "Gate 1", "date": "2026-11-26"}],
        "lr_net_escalation": null        # or {"decision": "keep_rate_of_record", "date": ..., "reason": ...}

    Returns {"excluded", "waivers", "lr_net_escalation"}; SystemExit on any malformed entry."""
    out = {"excluded": ud.get("excluded") or {}, "waivers": ud.get("waivers") or [],
           "lr_net_escalation": ud.get("lr_net_escalation")}
    if not isinstance(out["excluded"], dict):
        raise SystemExit(f"{where}: 'excluded' maps rung -> {{'reason': ..., 'date': ...}}")
    for r, e in out["excluded"].items():
        if not (isinstance(e, dict) and _nonempty(e.get("reason")) and _nonempty(e.get("date"))):
            raise SystemExit(f"{where}: the exclusion of {r} needs a 'reason' and a 'date': {e}")
    for w in out["waivers"]:
        if not (isinstance(w, dict) and _nonempty(w.get("rung")) and _nonempty(w.get("date"))
                and _nonempty(w.get("basis")) and _nonempty(w.get("subject_to"))):
            raise SystemExit(f"{where}: every waiver needs 'rung', 'guard', 'class', 'basis', 'subject_to' and "
                             f"'date': {w}")
        if w.get("guard") not in WAIVABLE:
            raise SystemExit(f"{where}: only these guards can be waived: {sorted(WAIVABLE)} (got {w.get('guard')})")
        if w.get("class") not in WAIVABLE[w["guard"]]:
            raise SystemExit(f"{where}: a {w['guard']} waiver is for one of the classes {list(WAIVABLE[w['guard']])} "
                             f"(got {w.get('class')})")
    x = out["lr_net_escalation"]
    if x is not None and not (isinstance(x, dict) and x.get("decision") in LR_NET_ESCALATION_DECISIONS
                              and _nonempty(x.get("date")) and _nonempty(x.get("reason"))):
        raise SystemExit(f"{where}: lr_net_escalation needs 'decision' in {LR_NET_ESCALATION_DECISIONS}, 'date' and "
                         f"'reason' (any other decision is implemented by hand): {x}")
    return out


def lr_net_escalation_pending(appr: Optional[dict] = None) -> bool:
    """PLAN 20 item 3: re-run pilots changed lr_net without a column below the floor and the user's decision (the
    approval key 'lr_net_escalation') is not on file."""
    x = ((decisions().get("warmstart_rerun") or {}).get("lr_net_after_rerun") or {})
    return bool(x.get("escalated")) and (appr or {}).get("lr_net_escalation") is None


def stage2_approval() -> tuple:
    """Review B5 and REVIEW 10.5 R1: Stage 2 runs only with an approval file written AFTER the user has seen the
    Stage-1 table and approved it (created by the orchestrator, never by this code), in this stage's directory:

        runs/<stage>/stage2_approval.json
        {"results_json_sha256": "<SHA-256 of the results.json the user saw>",
         "replication_reading": "literal" | "two_se",      # the user's choice (review ruling 5, [confirm])
         "kept_and_replicated": ["A9", "A9r"],             # the computed list, under that reading (unchanged)
         "a11_content": ["A9"],                            # = computed list minus the exclusions
         "excluded": {"A9r": {"reason": "...", "date": "2026-11-26"}},
         "waivers": [{"rung": "A9", "guard": "G-region", "class": "low_abar", "basis": "...",
                      "subject_to": "Gate 1", "date": "2026-11-26"}],
         "lr_net_escalation": null,                        # the user's decision if warmstart-rerun escalated
         "a11p_parameters": {"short_event_share": 0.5, "if_vetoed": "fit_Z"},   # optional; these are the defaults
         "approved_utc": "2026-11-26T10:00:00Z", "approved_by": "user (recorded by the orchestrator)",
         "notes": "..."}

    Refused if the file is absent, if results.json is absent or its SHA-256 differs (the table was recomputed after
    the approval), if the reading is unknown, if the approved list differs from the list results.json gives under
    that reading, or if the content check (stage2_content_check) fails. Returns (approval, kept list, results.json
    SHA-256, content record)."""
    rf, af = L.out_dir() / "results.json", L.out_dir() / STAGE2_APPROVAL
    if not rf.exists():
        raise SystemExit("stage2 refused: the Stage-1 report (results.json) does not exist")
    if not af.exists():
        raise SystemExit(f"stage2 refused: no approval file {paths.portable(af)} (written after the user has seen and "
                         "approved the Stage-1 table; format in run_m2a.stage2_approval)")
    appr = json.loads(af.read_text())
    sha = provenance.sha256_file(rf)
    if appr.get("results_json_sha256") != sha:
        raise SystemExit(f"stage2 refused: the approval names results.json {str(appr.get('results_json_sha256'))[:16]}"
                         f"..., the file on disk is {sha[:16]}... (the table changed after the approval)")
    if "kept_and_replicated" not in appr:                      # review R7
        raise SystemExit("stage2 refused: the approval file has no key 'kept_and_replicated' (write [] if empty)")
    reading = appr.get("replication_reading")
    if reading not in REPLICATION_READINGS:
        raise SystemExit(f"stage2 refused: replication_reading must be one of {sorted(REPLICATION_READINGS)}")
    res = json.loads(rf.read_text())
    kept = [r for r, v in (res.get(REPLICATION_READINGS[reading]) or {}).items() if v]
    if sorted(appr.get("kept_and_replicated") or []) != sorted(kept):
        raise SystemExit(f"stage2 refused: the approved list {appr.get('kept_and_replicated')} differs from the list "
                         f"of results.json under the reading '{reading}': {kept}")
    for k in ("approved_utc", "approved_by"):
        if not appr.get(k):
            raise SystemExit(f"stage2 refused: the approval file has no '{k}'")
    content = stage2_content_check(appr, res, kept)
    content["a11p_parameters"] = a11p_parameters(appr)        # format checked now; used by the A11p part
    return appr, kept, sha, content


def _rule_guard_blocks(res: dict, rung: str) -> dict:
    """The Stage-1 guard blocks of the poly pairs of ``rung`` that the rules use (not the originals replaced by a
    rerun pair, not superseded warm starts)."""
    not_used = {c["run"] for v in (res.get("primaries") or {}).values()
                for c in (v.get("originals_not_used") or {}).values()}
    not_used |= set((res.get("superseded_warm_start") or {}).keys())
    return {k: g for k, g in (res.get("guards") or {}).items()
            if L.rung_of(g["run"]) == rung and g["run"] not in not_used}


def _failed_guards(g: dict) -> list:
    return [n for n, ok in (("G-K", g["G-K"]["persistence"]["passed"]),
                            ("G-bulk", g["G-bulk"]["persistence"]["passed"]),
                            ("G-region", (g.get("G-region") or {}).get("passed"))) if not ok]


def stage2_content_check(appr: dict, res: dict, kept: list, star_check=None) -> dict:
    """REVIEW 10.5 R1 and PLAN 20: refused unless the content is the computed kept-and-replicated list minus the
    user's exclusions (every other rung of the list excluded with a reason and a date, no exclusion outside it);
    every content rung whose guard block in results.json fails a guard has a waiver for exactly that guard, and its
    waiver line (flagged*, REVIEW 10.5 R6, computed now from the tagged arrays) covers the failure on every pair the
    rules use; no waiver exists without such a failure; the content merges without conflict. ``star_check(run, comp,
    class)`` defaults to run_m2a_report.g_region_star_check."""
    if star_check is None:
        import run_m2a_report as R
        star_check = R.g_region_star_check
    for k in ("a11_content", "excluded", "waivers"):
        if k not in appr:
            raise SystemExit(f"stage2 refused: the approval file has no key '{k}' (write [] / {{}} if empty)")
    a = check_stage2_decisions(appr, "stage2 approval")
    content = appr.get("a11_content")
    if not isinstance(content, list):
        raise SystemExit("stage2 refused: 'a11_content' must be a list of rungs")
    stray = [r for r in content if r not in kept]
    if stray:
        raise SystemExit(f"stage2 refused: content rungs {stray} are not in the kept-and-replicated list {kept}")
    missing = [r for r in kept if r not in content and r not in a["excluded"]]
    if missing:
        raise SystemExit(f"stage2 refused: kept-and-replicated rungs {missing} are neither in the content nor excluded "
                         "with a reason")
    odd = [r for r in a["excluded"] if r not in kept or r in content]
    if odd:
        raise SystemExit(f"stage2 refused: exclusions {odd} are not rungs of the list {kept} outside the content")
    want = [r for r in kept if r not in a["excluded"]]
    if sorted(content) != sorted(want):                       # implied by the three checks; stated as the rule
        raise SystemExit(f"stage2 refused: the content {content} is not the computed list minus the exclusions {want}")
    try:
        stage2_specs(content)
    except SystemExit as err:
        raise SystemExit(f"stage2 refused: the content does not merge: {err}")
    waived = {}
    for w in a["waivers"]:
        if w["rung"] not in content:
            raise SystemExit(f"stage2 refused: a waiver names {w['rung']}, which is not in the content {content}")
        if (w["rung"], w["guard"]) in waived:
            raise SystemExit(f"stage2 refused: more than one waiver of {w['guard']} for {w['rung']}")
        waived[(w["rung"], w["guard"])] = w
    checks = {}
    failing = set()
    for r in content:
        blocks = _rule_guard_blocks(res, r)
        if not blocks:
            raise SystemExit(f"stage2 refused: no guard data for {r} in results.json (guards phase)")
        for key, g in blocks.items():
            for gname in _failed_guards(g):
                failing.add((r, gname))
                w = waived.get((r, gname))
                if w is None:
                    raise SystemExit(f"stage2 refused: {key}: {gname} failed and the user has neither excluded {r} "
                                     "nor waived that guard explicitly")
                chk = {**star_check(g["run"], g["vs"], w["class"]), "g_region_passed": False, "guard": gname}
                checks.setdefault(r, {})[key] = chk
                if not chk.get("covered") or chk.get("class") != w["class"]:
                    raise SystemExit(f"stage2 refused: {key}: the {gname} waiver for the class {w['class']} does not "
                                     f"cover the failure (flagged* line: {chk or 'missing - run the guards phase'})")
    extra = [k for k in waived if k not in failing]
    if extra:
        raise SystemExit(f"stage2 refused: waivers without a failure of that guard: {extra}")
    return {"computed_kept_and_replicated": kept, "replication_reading": appr["replication_reading"],
            "excluded": a["excluded"], "waivers": a["waivers"], "lr_net_escalation": a["lr_net_escalation"],
            "content": content, "waiver_checks": checks}


def _resolved(name: str) -> bool:
    """Review B8-2: a run the gate waits for is resolved if it is registered, or if it diverged and an independent
    diagnosis of its divergence is on file (L.diagnosis valid, any verdict). Consequences, printed by the report: a
    diagnosed A2c seed leaves its control column empty ("diverged (diagnosed)"); a diagnosed warm-start re-run does
    not supersede: its original stays in force with its warm start flagged."""
    if L.registered_row(name) is not None:
        return True
    return L.diverged_marker(name) is not None and bool((L.diagnosis(name) or {}).get("valid"))


def stage2_preconditions(appr: Optional[dict] = None) -> None:
    """User decisions of 25 / 26 Nov 2026 and review R6: refused while (i) A2c_s0 / A2c_s1 are not finished, (ii)
    the warm-start check has not run or a run it listed has no finished re-run, (iii) a Stage-1 run diverged and has
    no valid diagnosis, (iv) an lr_net escalation awaits the user, (v) the conditional controls are not evaluated
    (or their record differs from the re-evaluation, or a triggered control is not finished), or (vi) results.json
    differs from the report recomputed now from the registered arrays (only a table that follows from the arrays
    can be approved)."""
    import run_m2a_report as R
    miss = [sp.name for sp in a2c_specs() if not _resolved(sp.name)]
    if miss:
        raise SystemExit(f"stage2 refused: the A2c control is not finished or diverged without a valid diagnosis "
                         f"({miss}); run `run_m2a.py a2c`, or for a diverged seed obtain the diagnosis")
    if _net_runs_executed() and decisions().get("warmstart_check") is None:
        raise SystemExit("stage2 refused: run `run_m2a.py warmstart-check` (and warmstart-rerun) first")
    rec = L.superseded_recorded()
    open_ = [n for n in (decisions().get("warmstart_check") or {}).get("rerun", [])
             if n not in rec or not _resolved(rec[n])]
    if open_:
        raise SystemExit(f"stage2 refused: warm-start re-runs missing, or diverged without a valid diagnosis, for "
                         f"{open_}; run `run_m2a.py warmstart-rerun`, or for a diverged re-run obtain the diagnosis")
    waiting = [n for n, v in (decisions().get("diverged") or {}).items()
               if v.get("status") == "diverged - awaiting diagnosis" and not (L.diagnosis(n) or {}).get("valid")
               and not L.is_stage2_run(n)]
    if waiting:
        raise SystemExit(f"stage2 refused: diverged runs await an independent diagnosis: {waiting}")
    if lr_net_escalation_pending(appr):
        raise SystemExit("stage2 refused: the change of lr_net by replicate re-runs of the network pilots awaits the "
                         "user (PLAN 20 item 3; decisions.json warmstart_rerun.lr_net_after_rerun)")
    cc = decisions().get("conditional_controls")
    if cc is None:
        raise SystemExit("stage2 refused: run `run_m2a.py conditional-controls` first (PLAN 20 item 1)")
    if json.loads(json.dumps(L._jsonable(evaluate_conditional_controls()))) != cc:
        raise SystemExit("stage2 refused: the recorded conditional-control decision differs from its re-evaluation "
                         "on the registered arrays; re-run `run_m2a.py conditional-controls`")
    trig = [sp.name for ctl, mk in (("A2c_x2", a2c_x2_specs), ("A2c6k", a2c6k_specs)) if cc[ctl]["triggered"]
            for sp in mk(cc[ctl]["seeds"]) if not _resolved(sp.name)]
    if trig:
        raise SystemExit(f"stage2 refused: triggered conditional controls not finished (or diverged without a valid "
                         f"diagnosis): {trig}")
    on_disk = json.loads((L.out_dir() / "results.json").read_text())
    now = json.loads(paths.scrub_str(json.dumps(L._jsonable(R.collect(with_guards=True)))))
    if now != on_disk:
        diff = sorted(k for k in set(now) | set(on_disk) if now.get(k) != on_disk.get(k))
        raise SystemExit(f"stage2 refused: results.json differs from the report recomputed from the registered arrays "
                         f"(keys {diff}); re-run `run_m2a.py report` and renew the approval")


def _lr_net_or_none():
    """The network learning rate of record, or None if it is undecided (then the network runs are blocked)."""
    try:
        return L.frozen_lr("net")
    except RuntimeError:
        return None


# ------------------------------------------------------------------------- stage 2: memory, guards, A11p ---------
MEMORY_HEADROOM = 0.9           # a k_train = 64 step must stay below 90 % of the job's cap (REVIEW 10.5 R2)


def memory_probe_k64(name: str) -> dict:
    """REVIEW 10.5 R2: before the first k_train = 64 fit, the peak GPU memory of one training step of the fitted
    structure ``name`` at k_train = 64 on the 256 training events with the most hits (forward and backward, train
    and validation sets resident as in sfm.train.fit), against the job's memory cap (sfm.machine.caps; the device
    memory where no cap applies). Trains nothing; the model is discarded."""
    rec = decisions().get("stage2_memory_probe")
    if rec is not None:
        return rec
    sets, _ = L.data_cpu()

    def job():
        dev = L.common.DEVICE
        tr, va = sets["train"].to(dev), sets["val"].to(dev)
        model = L.rebuild(name, dev).train()
        n_hits = (tr.off[1:] - tr.off[:-1])
        idx = torch.argsort(n_hits, descending=True)[:256]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        b = tr.batch(idx)
        gen = torch.Generator(device=dev).manual_seed(0)
        nll, _ = metric.event_nll(model, b, tr.box, 64, gen)
        nll.mean().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        hits = int(n_hits[idx].sum())
        del model, tr, va, b, nll
        return {"peak_gb": peak, "hits_in_batch": hits}
    out = L.on_gpu("memory probe k_train 64", job)
    caps = machine.caps()
    limit = caps.get("gpu_mem_cap_gb") or caps.get("device_total_gb")
    out.update({"structure": name, "k_train": 64, "batch": 256, "cap_gb": caps.get("gpu_mem_cap_gb"),
                "limit_gb": limit, "headroom": MEMORY_HEADROOM,
                "fits": bool(limit is not None and out["peak_gb"] <= MEMORY_HEADROOM * limit)})
    note_decision("stage2_memory_probe", out)
    log(f"memory probe (k_train 64, batch 256, {out['hits_in_batch']} hits): peak {out['peak_gb']:.2f} GB, limit "
        f"{limit} GB -> {'fits' if out['fits'] else 'DOES NOT FIT'}")
    return out


def stage2_guard_pairs(specs) -> list:
    """(run, comparator) of every registered Stage-2 run: A11 / A11k vs A2s of the same seed, A11n vs the A2n run in
    force, A11p vs A11 of the same seed; the guard columns of Stage 1 plus the flagged* arrays (REVIEW 10.5 R4)."""
    return [(sp.name, sp.comparator) for sp in specs
            if L.registered_row(sp.name) is not None and L.registered_row(sp.comparator) is not None]


def phase_stage2_guards(specs, key: str = "stage2_guard_targets") -> None:
    pairs = stage2_guard_pairs(specs)
    targets = list(dict.fromkeys(n for pr in pairs for n in pr))
    note_decision(key, {"runs": targets, "pairs": pairs})
    log(f"stage2 guards: {len(targets)} runs ({targets})")
    sets, extras = L.data_cpu()
    for name in targets:
        guard_run(name, sets, extras)


A11P_THRESHOLD = 0.2e-4             # nat per training event: "removal costs < 0.2" (REVIEW 11.2)
A11P_N_CANDIDATES = {("A9",): 58}   # REVIEW 11.1: 14 x 3 + 8 x 2 coefficients with k >= 1 for the content A9
A11P_K, A11P_SEED, A11P_BATCH = 16, 12345, 128
A11P_DIR = "a11p"
A11P_U2_REF = (166, 95)             # REVIEW 11.6 (b): N_s <= 95 of 166 short validation events
A11P_DEFAULTS = {"short_event_share": 0.5, "if_vetoed": "fit_Z"}     # REVIEW 11.6 [confirm]: the reviewer's values
A11P_IF_VETOED = {"fit_Z": "Z is fitted; Z_0 is recorded and not fitted",
                  "fit_Z_and_Z0": "Z is fitted, and Z_0 as well (A11pz_s0, A11pz_s1; reported, not a candidate)"}


def a11p_parameters(appr: dict) -> dict:
    """The two [confirm] choices of REVIEW 11 as recorded inputs (approval key 'a11p_parameters', optional): the
    share x of each registered tolerance the short events may take (screen f_tr Q^S_j < x * 0.2e-4 nat; margin of
    U1 M = x * 1e-4 / f) and what is fitted when the short events veto coefficients (A11P_IF_VETOED). Defaults: the
    reviewer's values (x = 1/2, fit Z only). The refit is the A11 procedure (REVIEW 11.5)."""
    x = appr.get("a11p_parameters") or {}
    out = {**A11P_DEFAULTS, **x}
    try:
        out["short_event_share"] = float(out["short_event_share"])
    except (TypeError, ValueError):
        out["short_event_share"] = float("nan")
    if not (0.0 < out["short_event_share"] <= 1.0) or out["if_vetoed"] not in A11P_IF_VETOED \
            or set(x) - set(A11P_DEFAULTS) - {"date", "basis"}:
        raise SystemExit(f"stage2 refused: a11p_parameters: short_event_share in (0, 1], if_vetoed in "
                         f"{sorted(A11P_IF_VETOED)}, optionally 'date' and 'basis' (got {x})")
    out["source"] = "approval file" if x else "defaults of REVIEW 11 (the reviewer's values)"
    return out


def a11p_candidates(prov) -> list:
    """(row, k) for every coefficient with k >= 1, per row from the top (REVIEW 11.1, 11.3)."""
    return [(n, k) for n in prov.names for k in range(int(prov.deg[n]), 0, -1)]


def _nll_events(model, data) -> np.ndarray:
    """Per-event NLL at the fixed fold points of the screen (K = 16, seed 12345, batch 128, deterministic sums)."""
    return metric.test_nll(model, data, K=A11P_K, seed=A11P_SEED, batch=A11P_BATCH,
                           deterministic_sum=True).astype(np.float64)


@torch.no_grad()
def second_difference(prov, i: int, k: int, evaluate, l0: np.ndarray) -> tuple:
    """REVIEW 11.2: per-event central second and first differences in the coefficient c = coef[i, k] with
    h = max(0.1 |c|, 1e-3). The coefficient is stored in float32, so the realised steps h+ = fl(c + h) - c and
    h- = c - fl(c - h) are used: d2 = 2 [(l+ - l0)/h+ + (l- - l0)/h-] / (h+ + h-) (= (l+ - 2 l0 + l-)/h^2 when the
    steps are equal), d1 = (l+ - l-)/(h+ + h-). The coefficient is restored exactly."""
    c = float(prov.coef[i, k])
    h = max(0.1 * abs(c), 1e-3)
    prov.coef[i, k] = c + h
    hp = float(prov.coef[i, k]) - c
    lp = evaluate()
    prov.coef[i, k] = c - h
    hm = c - float(prov.coef[i, k])
    lm = evaluate()
    prov.coef[i, k] = c
    if float(prov.coef[i, k]) != c:
        raise RuntimeError("second_difference: the coefficient was not restored exactly")
    d2 = 2.0 * ((lp - l0) / hp + (lm - l0) / hm) / (hp + hm)
    d1 = (lp - lm) / (hp + hm)
    return c, h, d2, d1


def curvature_autodiff(model, prov, coefs: list, data, K: int = A11P_K, seed: int = A11P_SEED,
                       batch: int = A11P_BATCH) -> dict:
    """Exact second derivatives d^2 L / dc^2 (L = mean per-event NLL) by automatic differentiation at fixed fold
    points (generator on the device of ``data``, seeded as test_nll, the batches of test_nll; accumulated batch by
    batch), for the check of REVIEW 11.2; and the second difference of the same function in float32 evaluation.
    {"row[k]": {"autodiff", "second_difference", "h", "rel_diff"}}. The agreement is meaningful for coefficients
    whose curvature is resolved by the float32 per-event values (|H| h^2 well above ~1e-8 nat on a few hundred
    events); the screen checks the three most resolved candidates."""
    model.eval()

    def per_event():
        gen = torch.Generator(device=data.xyz.device).manual_seed(seed)
        with torch.no_grad(), metric.deterministic():           # reproducible per-event sums (as the screen)
            return torch.cat([metric.event_nll(model, b, data.box, K, gen)[0] for b in data.batches(batch)])
    res = {}
    for n, k in coefs:
        i = prov.index(n)
        gen = torch.Generator(device=data.xyz.device).manual_seed(seed)
        hsum, ntot = 0.0, 0
        for b in data.batches(batch):
            with torch.enable_grad():
                tot = metric.event_nll(model, b, data.box, K, gen)[0].double().sum()
                g = torch.autograd.grad(tot, prov.coef, create_graph=True)[0]
                hsum += float(torch.autograd.grad(g[i, k], prov.coef)[0][i, k])
            ntot += len(b.lnp)
        hess = hsum / ntot
        l0 = per_event().double().cpu().numpy()
        _, h, d2, _ = second_difference(prov, i, k, lambda: per_event().double().cpu().numpy(), l0)
        fd = float(d2.mean())
        ok = math.isfinite(hess) and math.isfinite(fd)
        res[f"{n}[{k}]"] = {"autodiff": hess if math.isfinite(hess) else None, "second_difference": fd, "h": h,
                            "rel_diff": (abs(fd - hess) / max(abs(hess), 1e-300)) if ok else None}
    return res


CURVATURE_TOL = 0.02                # REVIEW 11.2: agreement of H_j with automatic differentiation within 2 %


def curvature_check(model, prov, rows: list, data, want: int = 3, tries: int = 8) -> dict:
    """REVIEW 11.2 inside the screen: the most resolved candidates (largest |H_j| h_j^2 in the table of seed 0) are
    compared with automatic differentiation on ``data`` until ``want`` comparisons with a finite derivative exist
    (an autodiff second derivative can be non-finite through a branch of the model that the second difference does
    not see; it is recorded and skipped). passed = ``want`` finite comparisons, each within CURVATURE_TOL."""
    order = sorted(rows, key=lambda r: -abs(r["H"]) * r["h"] ** 2)[:tries]
    out = {}
    for r in order:
        out.update(curvature_autodiff(model, prov, [(r["row"], r["k"])], data))
        if sum(1 for v in out.values() if v["rel_diff"] is not None) >= want:
            break
    fin = [v for v in out.values() if v["rel_diff"] is not None]
    return {"coefficients": out, "n_finite": len(fin), "tolerance": CURVATURE_TOL,
            "passed": bool(len(fin) >= want and all(v["rel_diff"] <= CURVATURE_TOL for v in fin))}


def a11p_screen_seed(name: str, model, tr, short: np.ndarray, share: float) -> dict:
    """The table of REVIEW 11.7 item 1 for one seed: H_j, Q_j = 0.5 |H_j| c_j^2, -g_j c_j, H^S_j, Q^S_j, f_tr Q^S_j and
    the two pass flags, for every candidate; the model is left bit for bit as it was."""
    base = model.base if hasattr(model, "base") else model
    prov = base.provider
    model.eval()
    w0 = provenance.weights_sha256(model.state_dict())
    l0 = _nll_events(model, tr)
    f_tr = float(short.mean())
    rows = []
    for n, k in a11p_candidates(prov):
        c, h, d2, d1 = second_difference(prov, prov.index(n), k, lambda: _nll_events(model, tr), l0)
        H, HS, g = float(d2.mean()), float(d2[short].mean()), float(d1.mean())
        Q, QS = 0.5 * abs(H) * c * c, 0.5 * abs(HS) * c * c
        rows.append({"row": n, "k": k, "c": c, "h": h, "H": H, "Q": Q, "minus_g_c": -g * c, "H_S": HS, "Q_S": QS,
                     "f_tr_Q_S": f_tr * QS, "pass_registered": bool(Q < A11P_THRESHOLD),
                     "pass_short": bool(f_tr * QS < share * A11P_THRESHOLD)})
    if provenance.weights_sha256(model.state_dict()) != w0:
        raise RuntimeError(f"a11p screen: the weights of {name} changed during the screen")
    return {"run": name, "n_train": int(len(l0)), "n_short": int(short.sum()), "f_tr": f_tr,
            "L": float(l0.mean()), "L_S": float(l0[short].mean()), "n_params": int(model.n_params())
            if hasattr(model, "n_params") else None, "rows": rows}


def a11p_select(tables: dict, flags: tuple) -> list:
    """REVIEW 11.3 / 11.4: a coefficient passes if all ``flags`` hold on every seed; per row from the top, drop while
    the coefficient passes and stop at the first that does not. Returns [[row, k], ...]."""
    seeds = sorted(tables)
    by = {s: {(r["row"], r["k"]): r for r in tables[s]["rows"]} for s in seeds}
    order = [(r["row"], r["k"]) for r in tables[seeds[0]]["rows"]]
    stopped, out = set(), []
    for n, k in order:
        if n in stopped:
            continue
        if all(all(by[s][(n, k)][f] for f in flags) for s in seeds):
            out.append([n, k])
        else:
            stopped.add(n)
    return out


def prune_of(Z: list, prov_deg: dict) -> dict:
    """{row: reduced degree} for a set Z of dropped (row, k), each row dropped from the top."""
    out = {}
    for n, k in Z:
        out[n] = min(out.get(n, int(prov_deg[n])), int(k) - 1)
    return out


def a11p_screen(content: list, share: float, params: Optional[dict] = None) -> dict:
    """REVIEW 11.1-11.4 and 11.7: the screen on A11_s0 and A11_s1 (training events only), the sets Z_0 and Z, the
    joint cost J, J^S, and the autodiff check; written to runs/<stage>/a11p/screen.json and hashed (decisions.json
    'stage2_a11p_screen') before any A11p fit. Never overwritten: an existing screen is reused only if its SHA-256
    is the recorded one and its share equals ``share``."""
    d = L.out_dir() / A11P_DIR
    f = d / "screen.json"
    rec = decisions().get("stage2_a11p_screen")
    if f.exists():
        if rec and provenance.sha256_file(f) == rec.get("sha256"):
            out = json.loads(f.read_text())
            if out["short_event_share"] != share:
                raise SystemExit(f"a11p refused: the hashed screen used the share {out['short_event_share']}, the "
                                 f"approval confirms {share}; nothing is recomputed")
            return out
        raise SystemExit(f"a11p refused: {paths.portable(f)} exists but is not the recorded screen; nothing is "
                         "overwritten")
    sets, _ = L.data_cpu()
    short = sets["train"].ev["zb"].numpy().astype(np.float64) < registry.ZB_STRATUM_MM

    def job():
        dev = L.common.DEVICE
        tr = sets["train"].to(dev)
        models_, tables = {}, {}
        for s in (0, 1):
            models_[s] = L.rebuild(run_name("A11", s), dev)
            tables[s] = a11p_screen_seed(run_name("A11", s), models_[s], tr, short, share)
        base0 = models_[0].base if hasattr(models_[0], "base") else models_[0]
        prov0 = base0.provider
        cands = a11p_candidates(prov0)
        check = curvature_check(models_[0], prov0, tables[0]["rows"],
                                tr.subset(torch.arange(256, device=tr.off.device)))
        Z0 = a11p_select(tables, ("pass_registered",))
        Z = a11p_select(tables, ("pass_registered", "pass_short"))
        joint = {}
        for s in (0, 1):
            m = models_[s]
            prov = (m.base if hasattr(m, "base") else m).provider
            l0 = _nll_events(m, tr)
            keep = {(n, k): float(prov.coef[prov.index(n), k]) for n, k in Z}
            with torch.no_grad():
                for n, k in Z:
                    prov.coef[prov.index(n), k] = 0.0
            lz = _nll_events(m, tr) if Z else l0
            with torch.no_grad():
                for (n, k), c in keep.items():
                    prov.coef[prov.index(n), k] = c
            J = lz - l0
            joint[s] = {"sum_Q_dropped": float(sum(r["Q"] for r in tables[s]["rows"] if [r["row"], r["k"]] in Z)),
                        "J": float(J.mean()), "J_S": float(J[short].mean())}
        out = {"spec": "REVIEW.md section 11", "parameters": params, "short_event_share": share,
               "threshold_nat": A11P_THRESHOLD,
               "short_threshold_nat": share * A11P_THRESHOLD, "points": {"K": A11P_K, "seed": A11P_SEED,
                                                                        "batch": A11P_BATCH},
               "content": content, "n_candidates": len(cands), "tables": {str(s): tables[s] for s in (0, 1)},
               "Z0": Z0, "Z": Z, "Z0_minus_Z": [z for z in Z0 if z not in Z],
               "prune": prune_of(Z, prov0.deg), "prune_Z0": prune_of(Z0, prov0.deg),
               "degrees_after": {n: prune_of(Z, prov0.deg).get(n, int(prov0.deg[n])) for n in prov0.names},
               "k_A11": int(tables[0]["n_params"]), "k_A11p": int(tables[0]["n_params"]) - len(Z),
               "joint": {str(s): joint[s] for s in (0, 1)}, "curvature_check": check}
        del models_, tr
        return out
    out = L.on_gpu("a11p screen", job)
    want = A11P_N_CANDIDATES.get(tuple(sorted(content)))
    if want is not None and out["n_candidates"] != want:
        raise SystemExit(f"a11p refused: {out['n_candidates']} candidates, REVIEW 11.1 fixes {want} for {content}")
    L.write_json(out, f)
    sha = provenance.sha256_file(f)
    note_decision("stage2_a11p_screen", {"file": paths.portable(f), "sha256": sha, "Z": out["Z"], "Z0": out["Z0"],
                                         "k_A11p": out["k_A11p"], "short_event_share": share})
    log(f"a11p screen: {len(out['Z'])} of {out['n_candidates']} coefficients pass (Z_0: {len(out['Z0'])}); "
        f"k {out['k_A11']} -> {out['k_A11p']}; screen.json {sha[:16]}...")
    return out


def a11p_specs(content: list, screen: dict, which: str = "Z") -> list:
    """REVIEW 11.5: A11p_s = the pruned structure fitted by the procedure of A11 (start = the fitted A2s of the seed,
    copy by name up to the reduced degrees, T_A, seed s), comparator A11_s, step-0 row registered. which = "Z0":
    the same with Z_0 (A11pz_s, only when the parameter if_vetoed asks for it)."""
    a11 = {sp.seed: sp for sp in stage2_specs(content) if sp.rung == "A11"}
    sha = (decisions().get("stage2_a11p_screen") or {}).get("sha256", "")
    rung, prune, z = ("A11p", screen["prune"], screen["Z"]) if which == "Z" else \
        ("A11pz", screen["prune_Z0"], screen["Z0"])
    return [replace(a11[s], name=run_name(rung, s), rung=rung, kwargs={**a11[s].kwargs, "prune": prune},
                    comparator=run_name("A11", s), init_eval=True,
                    notes=f"{rung} (REVIEW 11): A11 without {len(z)} coefficients ({which}; screen.json "
                          f"{sha[:16]}...)")
            for s in (0, 1)]


def a11p_run(content: list, conf: dict) -> None:
    """REVIEW 10.5 R7 / R9: screen (after A11_s0 and A11_s1 are registered), then the fits with exactly the hashed
    pruning set, then their guards."""
    if not content:
        note_decision("stage2_a11p", {"status": "not applicable: the content is empty (A11 = A2s)"})
        return
    miss = [run_name("A11", s) for s in (0, 1) if L.registered_row(run_name("A11", s)) is None]
    if miss:
        note_decision("stage2_a11p", {"status": f"not run: {miss} not registered"})
        log(f"a11p: not run, {miss} not registered")
        return
    screen = a11p_screen(content, conf["short_event_share"], conf)
    rec = decisions().get("stage2_a11p_screen") or {}
    if provenance.sha256_file(L.out_dir() / A11P_DIR / "screen.json") != rec.get("sha256"):
        raise SystemExit("a11p refused: screen.json is not the hashed screen")
    if not (screen.get("curvature_check") or {}).get("passed"):                    # PLAN 23 item 13: hard stop
        raise SystemExit("STOP (PLAN 23 item 13, screen check): the second differences of the A11p screen do not agree "
                         f"with automatic differentiation within {CURVATURE_TOL:.0%} on the most resolved candidates "
                         f"({screen.get('curvature_check')}); no A11p fit; waiting for the user")
    if not screen["Z"]:
        note_decision("stage2_a11p", {"status": "no coefficient passes the screen", "parameters": conf})
        log("a11p: no coefficient passes the screen; nothing is fitted")
        return
    specs = a11p_specs(content, screen)
    if conf["if_vetoed"] == "fit_Z_and_Z0" and screen["Z0"] != screen["Z"]:
        specs += a11p_specs(content, screen, which="Z0")
    for sp in specs:
        init = L.read_json(sp.out_dir / "init.json", {}) or {}
        got = ((init.get("build_kwargs") or {}).get("prune"))
        want = sp.kwargs["prune"]
        if init and got != want:
            raise SystemExit(f"a11p refused: {sp.name} was started with the pruning set {got}, the hashed screen "
                             f"gives {want}")
        run_stage2_spec(sp)
    note_decision("stage2_a11p", {"status": "fitted", "parameters": conf, "runs": [sp.name for sp in specs],
                                  "prune": screen["prune"]})
    phase_stage2_guards(specs, key="stage2_a11p_guard_targets")


def run_stage2_spec(sp: Spec) -> dict:
    """run_spec for a Stage-2 run with the hard stop of PLAN 23 item 13: a Stage-2 run that diverged (criterion (ii)
    is a failed scoring check) or is blocked stops the phase with a non-zero exit until the user responds."""
    res = L.run_spec(sp)
    st = L.status_of(res)
    if st != "done":
        why = "; ".join(res.get("criteria") or []) or res.get("blocked_by")
        raise SystemExit(f"STOP (PLAN 23 item 13): {sp.name} {st} ({why}); Stage 2 halts here and waits for the user "
                         "(decisions.json, the run's diverged.json)")
    return res


def _stage2_gate() -> tuple:
    appr, kept, sha, content = stage2_approval()
    stage2_preconditions(appr)
    note_decision("stage2_approval", {"results_json_sha256": sha, "replication_reading": appr["replication_reading"],
                                      "approved_utc": appr["approved_utc"], "approved_by": appr["approved_by"],
                                      "approval_file_sha256": provenance.sha256_file(L.out_dir() / STAGE2_APPROVAL)})
    run = content["content"]
    note_decision("stage2_content", {
        "kept_and_replicated": kept, "replication_reading": content["replication_reading"],
        "excluded": content["excluded"], "waivers": content["waivers"],
        "lr_net_escalation": content["lr_net_escalation"], "a11_content": run,
        "rule": "content = computed kept-and-replicated list minus the user's exclusions (PLAN 20, REVIEW 10.5 R1)",
        "A11": "A2s (no rerun)" if not run else f"A2s + {run}",
        "A11n": (f"A2n (no rerun; in force: {[L.effective(f'A2n_s{s}') for s in (0, 1)]})"
                 if not run else "net version of A11"),
        "lr_net_of_record": _lr_net_or_none(), "waiver_checks": content["waiver_checks"],
        "a11p_parameters": content["a11p_parameters"]})
    return appr, run, content


def phase_stage2():
    """REVIEW 10.5 R9: fits (A11, then A11k after the memory probe, A11n) -> their guards -> A11p screen -> A11p fits
    -> their guards -> Stage-2 report. A re-issue on a finished state starts no fit."""
    appr, run, content = _stage2_gate()
    specs = stage2_specs(run)
    for sp in specs:
        if sp.k_train == 64 and not sp.done():
            ref = next((x.name for x in specs if x.rung == "A11" and x.seed == sp.seed), None) or "A2s_s0"
            probe = memory_probe_k64(ref if L.registered_row(ref) is not None else "A2s_s0")
            if not probe["fits"]:
                raise SystemExit(f"stage2 stopped before {sp.name}: one k_train = 64 step needs "
                                 f"{probe['peak_gb']:.2f} GB, more than {MEMORY_HEADROOM:.0%} of the limit "
                                 f"{probe['limit_gb']} GB (REVIEW 10.5 R2: the batch is part of T_A and is not "
                                 "changed; report to the user)")
        run_stage2_spec(sp)
    phase_stage2_guards(specs)
    a11p_run(run, content["a11p_parameters"])
    import run_m2a_report as R
    R.report_stage2()


def phase_a11p():
    """A11p alone (re-issue after an interruption): the gate again, then the screen (reused if hashed), the fits,
    their guards and the Stage-2 report."""
    appr, run, content = _stage2_gate()
    a11p_run(run, content["a11p_parameters"])
    import run_m2a_report as R
    R.report_stage2()


def phase_stage2_report():
    import run_m2a_report as R
    R.report_stage2()


PHASES = {"templates": phase_templates, "parity": phase_parity, "plr": phase_plr, "stage0": phase_stage0,
          "stage1p": phase_stage1p, "stage1x": phase_stage1x,
          "warmstart-check": phase_warmstart_check, "warmstart-rerun": phase_warmstart_rerun, "a2c": phase_a2c,
          "conditional-controls": phase_conditional_controls,
          "guards": phase_guards, "report": phase_report, "stage2": phase_stage2, "stage2-report": phase_stage2_report,
          "a11p": phase_a11p}
ADDED_PHASES = ("warmstart-check", "warmstart-rerun", "a2c")   # PLAN 17.11; never in DEFAULT_PHASES
ADDED_PHASES_26NOV = ("conditional-controls",)                  # PLAN 20; never in DEFAULT_PHASES
STAGE2_PHASES = ("stage2", "stage2-report", "a11p")             # each only on its own


# ---------------------------------------------------------------------------------------------------- listing ---------
def listing() -> dict:
    """The real run list in execution order (no GPU, no file written)."""
    rows = []

    def add(phase, sp, condition=None, extra_steps=0):
        rows.append({"phase": phase, "name": sp.name, "rung": sp.rung, "seed": sp.seed, "provider": sp.provider,
                     "class": sp.cls, "steps": sp.steps, "k_train": sp.k_train, "comparator": sp.comparator,
                     "condition": condition, "step0_row": sp.init_eval})
    rows.append({"phase": "parity", "name": "A0_init", "rung": "A0", "seed": 0, "provider": "poly", "class": "step-0",
                 "steps": 0, "k_train": None, "comparator": "formula array", "condition": None, "step0_row": True})
    poly_p, net_p = plr_specs()
    for sp in poly_p:
        add("plr", sp)
    add("plr", next(s for s in stage0_specs() if s.name == "A2s_s0"))
    for sp in net_p:
        add("plr", sp)
    for sp in stage0_specs():
        if sp.name != "A2s_s0":
            add("stage0", sp)
    poly, net = primary_specs()
    for sp in poly + net:
        add("stage1p", sp)
    for sp in poly:
        comp2, sp2 = rerun_specs(sp)
        add("stage1p", comp2, condition=f"only if {sp.name} has not converged (comparator rerun, shared per seed)")
        add("stage1p", sp2, condition=f"only if {sp.name} has not converged")
    for r in EXPLORATORY:
        add("stage1x", exploratory_spec(r, 0))
        add("stage1x", exploratory_spec(r, 1), condition=f"only if {r}_s0 passes the keep or the equal-NLL rule")
    uncond = [x for x in rows if x["condition"] is None]
    cond = [x for x in rows if x["condition"] is not None]
    comp_reruns = {x["name"] for x in cond if x["name"].startswith("A2s_s")}
    cond_steps = sum(x["steps"] for x in cond if not x["name"].startswith("A2s_s")) + 2 * 8000
    return {"rows": rows, "n_unconditional_fits": sum(1 for x in uncond if x["steps"] > 0),
            "n_conditional_fits_max": len([x for x in cond if not x["name"].startswith("A2s_s")]) + len(comp_reruns),
            "training_steps_unconditional": sum(x["steps"] for x in uncond),
            "training_steps_conditional_max": cond_steps,
            "batch_events_per_step": 256, "k_train": 16,
            "not_in_default": "stage2 (A11, A11k, A11n; A11p not implemented) - explicit phase only"}


def print_listing() -> None:
    lst = listing()
    print(f"{'#':>3} {'phase':8} {'name':18} {'rung':6} {'seed':>4} {'prov':4} {'class':12} {'steps':>6} "
          f"{'comparator':16} condition")
    for i, x in enumerate(lst["rows"], 1):
        print(f"{i:3d} {x['phase']:8} {x['name']:18} {x['rung']:6} {x['seed']:4d} {x['provider']:4} {x['class']:12} "
              f"{x['steps']:6d} {str(x['comparator']):16} {x['condition'] or ''}")
    print(f"\nunconditional fits: {lst['n_unconditional_fits']}, training steps {lst['training_steps_unconditional']:,}"
          f" (batch 256, K_train 16)\nconditional fits at most: {lst['n_conditional_fits_max']}, training steps at "
          f"most {lst['training_steps_conditional_max']:,}\nnothing dropped (no budget rule); "
          f"{lst['not_in_default']}")
    print_listing_additions()


def listing_additions() -> list:
    """Runs added by the user's decisions of 25 Nov 2026 (PLAN 17.11), after the run list above had started. They
    are not in the default phase list and are issued after the present job has ended."""
    rows = [{"phase": "a2c", "name": sp.name, "rung": sp.rung, "seed": sp.seed, "provider": sp.provider,
             "class": sp.cls, "steps": sp.steps, "comparator": sp.comparator, "condition": None} for sp in a2c_specs()]
    rows.append({"phase": "warmstart-check", "name": "(trains nothing)", "rung": "-", "seed": -1, "provider": "net",
                 "class": "check", "steps": 0, "comparator": None,
                 "condition": "every network run executed with fit_to version < 3"})
    rows.append({"phase": "warmstart-rerun", "name": "<name>_ws", "rung": "-", "seed": -1, "provider": "net",
                 "class": "as original", "steps": -1, "comparator": "superseding re-run where it exists",
                 "condition": "exactly the runs listed by warmstart-check (PLAN 17.11 rule), plus all network runs "
                              "if a re-run network pilot changes lr_net; steps as the original"})
    return rows


def print_listing_additions() -> None:
    print("\n# Additions of 25 Nov 2026 (user decisions, PLAN.md section 17.11) - NOT in the default phase list; issued "
          "after the present job has ended, in this order: warmstart-check, warmstart-rerun, a2c, report")
    for x in listing_additions():
        print(f"  + {x['phase']:16} {x['name']:18} {x['rung']:6} {x['seed']:4d} {x['provider']:4} {x['class']:12} "
              f"{x['steps']:6d} {str(x['comparator']):16} {x['condition'] or ''}")
    print_listing_additions_26nov()


def listing_additions_26nov() -> list:
    """Runs and phases added by the user's decisions of 26 Nov 2026 (PLAN 20); none in the default phase list."""
    rows = []
    for sp in a2c_x2_specs((0, 1)):
        rows.append({"phase": "conditional-controls", "name": sp.name, "rung": sp.rung, "seed": sp.seed,
                     "provider": sp.provider, "class": sp.cls, "steps": sp.steps, "comparator": sp.comparator,
                     "condition": "only if a doubled-step rerun pair of that seed passes keep or one-seed equal NLL"})
    for sp in a2c6k_specs((0, 1)):
        rows.append({"phase": "conditional-controls", "name": sp.name, "rung": sp.rung, "seed": sp.seed,
                     "provider": sp.provider, "class": sp.cls, "steps": sp.steps, "comparator": sp.comparator,
                     "condition": "only if a mixture rung of that seed passes keep or one-seed equal NLL"})
    for rung, seeds, cmp_, k in (("A11", (0, 1, 2), "A2s_s<seed>", 16), ("A11k", (0, 1), "A2s_s<seed>", 64),
                                 ("A11n", (0, 1), "A2n_s<seed>*", 16)):
        for s_ in seeds:
            rows.append({"phase": "stage2", "name": run_name(rung, s_), "rung": rung, "seed": s_,
                         "provider": "net" if rung == "A11n" else "poly", "class": "stage2", "steps": 4000,
                         "comparator": cmp_, "condition": f"approval file; content = computed list minus user "
                                                          f"exclusions (k_train {k})"})
    rows.append({"phase": "stage2", "name": "(guards)", "rung": "-", "seed": -1, "provider": "-", "class": "guards",
                 "steps": 0, "comparator": None, "condition": "every registered Stage-2 run and its comparator, with "
                                                              "the flagged* arrays (REVIEW 10.5 R4, R6)"})
    rows.append({"phase": "stage2", "name": "(A11p screen)", "rung": "A11p", "seed": -1, "provider": "poly",
                 "class": "screen", "steps": 0, "comparator": None,
                 "condition": "REVIEW 11: 117 passes over the training events per seed after A11_s0, A11_s1; "
                              "written and hashed before any A11p fit"})
    for s_ in (0, 1):
        rows.append({"phase": "stage2", "name": run_name("A11p", s_), "rung": "A11p", "seed": s_, "provider": "poly",
                     "class": "stage2", "steps": 4000, "comparator": run_name("A11", s_),
                     "condition": "only if the screen drops a coefficient (Z non-empty); step-0 row registered"})
    for s_ in (0, 1):
        rows.append({"phase": "stage2", "name": run_name("A11pz", s_), "rung": "A11pz", "seed": s_,
                     "provider": "poly", "class": "stage2", "steps": 4000, "comparator": run_name("A11", s_),
                     "condition": "only if a11p_parameters.if_vetoed = fit_Z_and_Z0 and Z_0 differs from Z"})
    rows.append({"phase": "stage2", "name": "(guards)", "rung": "-", "seed": -1, "provider": "-", "class": "guards",
                 "steps": 0, "comparator": None, "condition": "A11p_s<seed> and A11_s<seed>"})
    return rows


def print_listing_additions_26nov() -> None:
    print("\n# Additions of 26 Nov 2026 (user decisions, PLAN.md section 20; REVIEW.md sections 10, 11) - NOT in the "
          "default phase list; in this order: conditional-controls, guards (flagged* arrays), report, then stage2 "
          "alone after the approval (a11p, stage2-report: alone); * = the A2n run in force")
    for x in listing_additions_26nov():
        print(f"  + {x['phase']:20} {x['name']:14} {x['rung']:6} {x['seed']:4d} {x['provider']:4} {x['class']:12} "
              f"{x['steps']:6d} {str(x['comparator']):16} {x['condition'] or ''}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="*", choices=list(PHASES) + ["all"], help="default: " + " ".join(DEFAULT_PHASES))
    ap.add_argument("--smoke", action="store_true", help="end-to-end test with shortened fits (stage m2a_smoke; "
                                                         "needs SFM_M2A_SMOKE=1)")
    ap.add_argument("--list", action="store_true", help="print the real run list and exit (no GPU)")
    ap.add_argument("--without-guards", action="store_true",
                    help="only with the single phase 'report': the tables without guard columns, for a run whose "
                         "guards phase failed or has not run (review C2); writes RESULTS_without_guards.md")
    args = ap.parse_args(argv)
    if args.list:
        print_listing()
        return 0
    if args.without_guards:
        if args.phases != ["report"]:
            raise SystemExit("--without-guards goes with the single phase 'report' (run_m2a.py report --without-guards)")
        global REPORT_WITHOUT_GUARDS
        REPORT_WITHOUT_GUARDS = True
    phases = list(DEFAULT_PHASES) if (not args.phases or "all" in args.phases) else args.phases
    for ph2 in STAGE2_PHASES:
        if ph2 in phases and phases != [ph2]:
            raise SystemExit(f"{ph2} runs only on its own (`run_m2a.py {ph2}`), after the user has seen the Stage-1 "
                             "table")
    order = list(PHASES)
    phases = sorted(dict.fromkeys(phases), key=order.index)
    L.configure(smoke=args.smoke)
    L.start("run_m2a" + (" --smoke" if args.smoke else ""))
    for ph in phases:
        log(f"=== phase {ph} (stage {L.stage()}; GPU seconds so far {L.gpu_seconds_total():.0f}, information only) ===")
        PHASES[ph]()
    log(f"=== finished {phases} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
