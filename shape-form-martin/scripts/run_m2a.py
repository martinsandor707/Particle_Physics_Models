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
                information of k for A6b
    report      scripts/run_m2a_report.py: the rules applied to the registered arrays, guard tables, secondary
                columns, RESULTS.md / results.json
    stage2      A11 / A11k / A11n (explicit only; refused before the report exists: the user sees the Stage-1 table
                before any combination is formed)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_m2a_lib as L  # noqa: E402
from run_m2a_lib import Spec, log  # noqa: E402

from sfm import metric, paths, provenance, registry  # noqa: E402
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


def g_region_arrays(name: str, model, val, val_cpu) -> dict:
    """Per-event NLL decomposition of one run on the registry's fold points (K = 64, seed 12345, batch 128):
    hit-term contribution -sum_{i in e, class} omega_i ln g_i per class and the normalisation term
    weight_sum_e ln M_e, saved as tagged arrays (subset "g_region:<class>"); paired across runs by the report."""
    base = L.live()["points"]
    sdir = L.scores_dir(name)                          # where the registry put this run's arrays
    marker = sdir / "g_region.json"
    if marker.exists():
        return L.read_json(marker)
    model.eval()
    parts = metric.test_nll(model, val, return_parts=True, deterministic_sum=True)
    hit = parts["hit"][0].astype(np.float64)
    norm = (parts["weight_sum"].astype(np.float64) * parts["log_mass"][0].astype(np.float64))
    N = len(val_cpu)
    off = val_cpu.off.numpy()
    ev = np.repeat(np.arange(N), np.diff(off))
    masks = _g_region_masks(val_cpu)
    out = {"name": name, "arrays": {}, "n_hits": int(len(hit))}
    for cls, m in masks.items():
        a = np.bincount(ev, weights=-hit * m, minlength=N)
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
    recon = np.bincount(ev, weights=-hit, minlength=N) + norm
    reg = L.tagged(name).values.astype(np.float64)
    out["identity_check"] = {"parts_nll_vs_parts_sum_max_abs": float(np.abs(recon - parts["nll"][0]).max()),
                             "parts_nll_vs_registered_k64": provenance.tolerance_check(parts["nll"][0], reg, top=3)}
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


def phase_guards():
    targets, pairs, passing = guard_targets()
    note_decision("guard_targets", {"rungs_passing_keep_or_equal": passing, "runs": targets, "pairs": pairs,
                                    "smoke_limited": L.CTX.smoke})
    log(f"guards: {len(targets)} runs ({targets})")
    sets, extras = L.data_cpu()
    for name in targets:
        need_row = L.registered_row(name, need_guards=True) is None
        need_gr = not (L.scores_dir(name) / "g_region.json").exists()
        if not (need_row or need_gr):
            continue

        def job(name=name, need_row=need_row, need_gr=need_gr):
            dev = L.common.DEVICE
            va = sets["val"].to(dev)
            val = registry.ValidationSet(va, extras["val"]["event_number"])
            model = L.rebuild(name, dev)
            mp = L.out_dir() / name / "model.pt"
            if need_row:
                row0 = L.registered_row(name)
                registry.evaluate(model, name, L.stage(), val,
                                  meta={"k": row0["k"], "seed": row0.get("seed"), "checkpoint": str(mp),
                                        "config_hash": row0.get("config_hash"), "smoke": L.CTX.smoke,
                                        "trained_on": {"machine_id": L.MACHINE_ID, "source": "trained on this machine"},
                                        "notes": "guard columns (G-K, G-bulk, G-mass) of a Track A run"},
                                  guards=True)
            if need_gr:
                g_region_arrays(name, model, va, sets["val"])
            del model, va, val

        L.on_gpu(f"guards {name}", job)
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

    def rerun(name: str, lr=None, reason="listed by warmstart-check"):
        sp = _spec_of(name)
        kw = {k: v for k, v in sp.kwargs.items() if k != "fit_to_scale_floor"}     # always fit_to version 3
        new = replace(sp, name=name + "_ws", kwargs=kw, lr=(lr if lr is not None else sp.lr),
                      comparator=sup.get(sp.comparator, sp.comparator),
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
        new_lr, tab = _plr_decide({sp.lr: sup.get(sp.name, sp.name) for sp in net_p})
        rec["lr_net_after_rerun"] = {"lr_net_before": lr_net, "lr_net_after": new_lr, "table": tab}
        note_decision("warmstart_rerun", rec)
        if new_lr is not None and new_lr != lr_net:
            for n in _net_runs_executed():
                if n not in listed and not n.startswith("plr_A2n_"):
                    rec.setdefault("dependent", {})[n] = f"lr_net changed from {lr_net} to {new_lr}"
            lr_net = new_lr
    todo = [n for n in _net_runs_executed() if not n.startswith("plr_A2n_")
            and (n in listed or n in (rec.get("dependent") or {}))]
    lr_changed = bool((rec.get("lr_net_after_rerun") or {}).get("lr_net_after") not in
                      (None, (rec.get("lr_net_after_rerun") or {}).get("lr_net_before")))
    for n in todo:
        rerun(n, lr=(lr_net if lr_changed else None),
              reason="listed by warmstart-check" if n in listed else rec["dependent"][n])
    rec["done"] = True
    note_decision("warmstart_rerun", rec)
    log(f"warmstart-rerun: superseded {sup}")
    phase_guards()


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


def stage2_approval() -> tuple:
    """Review B5: Stage 2 runs only with an approval file written AFTER the user has seen the Stage-1 table and
    approved it (created by the orchestrator, never by this code), in this stage's directory:

        runs/<stage>/stage2_approval.json
        {"results_json_sha256": "<SHA-256 of the results.json the user saw>",
         "replication_reading": "literal" | "two_se",      # the user's choice (review ruling 5, [confirm])
         "kept_and_replicated": ["A7", ...],               # the list the user approved, under that reading
         "approved_utc": "2026-11-26T10:00:00Z", "approved_by": "user (recorded by the orchestrator)",
         "notes": "..."}

    Refused if the file is absent, if results.json is absent or its SHA-256 differs (the table was recomputed after
    the approval), if the reading is unknown, or if the approved list differs from the list results.json gives
    under that reading. Returns (approval, kept list, results.json SHA-256)."""
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
    return appr, kept, sha


def _resolved(name: str) -> bool:
    """Review B8-2: a run the gate waits for is resolved if it is registered, or if it diverged and an independent
    diagnosis of its divergence is on file (L.diagnosis valid, any verdict). Consequences, printed by the report: a
    diagnosed A2c seed leaves its control column empty ("diverged (diagnosed)"); a diagnosed warm-start re-run does
    not supersede: its original stays in force with its warm start flagged."""
    if L.registered_row(name) is not None:
        return True
    return L.diverged_marker(name) is not None and bool((L.diagnosis(name) or {}).get("valid"))


def stage2_preconditions() -> None:
    """User decisions of 25 Nov 2026 and review R6: refused while (i) A2c_s0 / A2c_s1 are not finished, (ii) the
    warm-start check has not run or a run it listed has no finished re-run, (iii) a primary, base, control or rerun
    run diverged and has no valid diagnosis, or (iv) results.json differs from the report recomputed now from the
    registered arrays (only a table that follows from the arrays can be approved)."""
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
               if v.get("status") == "diverged - awaiting diagnosis" and not (L.diagnosis(n) or {}).get("valid")]
    if waiting:
        raise SystemExit(f"stage2 refused: diverged runs await an independent diagnosis: {waiting}")
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


def phase_stage2():
    appr, kept, sha = stage2_approval()
    stage2_preconditions()
    note_decision("stage2_approval", {"results_json_sha256": sha, "replication_reading": appr["replication_reading"],
                                      "approved_utc": appr["approved_utc"], "approved_by": appr["approved_by"],
                                      "approval_file_sha256": provenance.sha256_file(L.out_dir() / STAGE2_APPROVAL)})
    note_decision("stage2_content", {"kept_and_replicated": kept,
                                     "A11": "A2s (no rerun)" if not kept else f"A2s + {kept}",
                                     "A11n": (f"A2n (no rerun; in force: {[L.effective(f'A2n_s{s}') for s in (0, 1)]})"
                                              if not kept else "net version of A11"),
                                     "lr_net_of_record": _lr_net_or_none(),
                                     "A11p": "not implemented (pruning by diagonal Fisher information)"})
    for sp in stage2_specs(kept):
        L.run_spec(sp)


PHASES = {"templates": phase_templates, "parity": phase_parity, "plr": phase_plr, "stage0": phase_stage0,
          "stage1p": phase_stage1p, "stage1x": phase_stage1x,
          "warmstart-check": phase_warmstart_check, "warmstart-rerun": phase_warmstart_rerun, "a2c": phase_a2c,
          "guards": phase_guards, "report": phase_report, "stage2": phase_stage2}
ADDED_PHASES = ("warmstart-check", "warmstart-rerun", "a2c")   # PLAN 17.11; never in DEFAULT_PHASES


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
    if "stage2" in phases and phases != ["stage2"]:
        raise SystemExit("stage2 runs only on its own (`run_m2a.py stage2`), after the user has seen the Stage-1 table")
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
