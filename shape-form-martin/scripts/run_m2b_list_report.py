"""Report of the Track B list (scripts/run_m2b_list.py): the pre-registered rules of AUDIT.md section 4.0 / 4.2 and
PLAN.md section 6 applied MECHANICALLY to the registered arrays of the live stream. No training, no test data.

    .venv/bin/python -B shape-form-martin/scripts/run_m2b_list_report.py [--smoke]

Writes <machine dir>/runs/<stage>/LIST_RESULTS.md and list_results.json. Units in the tables: 1e-4 nat per event,
paired, +- SE over the 14,950 validation events. Every delta goes through sfm.provenance.paired_delta on the live
stream (tags checked); untagged laptop rows are never read.

Rules (written here before any result of this list exists)
    promotion (PLAN 6, AUDIT 4.0), shown part by part:
        (a) Delta vs mlp_std <= -5            (b) Delta vs mlp_std <= -3 SE
        (c) better than B0 by more than 1 SE: Delta vs the per-event mean of the three controls b0h_seed0..2, z < -1
            (B0 is formed ONLY when exactly these three are registered; otherwise the part is "B0 incomplete")
        (d) replication by the second seed: seed-1 Delta vs mlp_std <= -2.5 with the same sign, and the two-seed mean
            (per-event mean of the two runs) meets (a) and (b)
        (e) guards passed (G-K and G-bulk: persistence margin guard - 0.5 main >= 0 resolved at 2 SE, from the
            registry's guard columns vs mlp_std; G-region: <= 50 % of the gain from deep / border / Abar <= 0.3 cells,
            margin flagged - 0.5 main >= 0; G-mass reported)
        promoted = (a) and (b) and (c) and (d) and (e). For B1d the two starts are the replicates (AUDIT 4.2 "2 starts").
    primary comparisons (AUDIT 4.2): B1d vs mlp_std; B2b - B2a, B2c - B2a, B4f - B2a, B6c - B6a, same seed; the same
        thresholds (-5 and -3 SE) and the same replication rule are applied to the comparison (reading flagged in
        results/m2b_mlp/STATUS.md); all other runs are exploratory, comparators or diagnostics and are labelled so.
    1-SE rule (B6): per seed, among B6a, B6b, B6c, the run with the fewest weights whose validation mean is within one
        paired SE of the best (sfm.stats.one_se_select through registry.select_1se).
    B3: EMA stream minus last iterate of the same run. B-ceil: the registered last iterate, and the minimum of the
        validation trajectory (K = 64 means at every 100 steps; a diagnostic: the minimum over a trajectory is selected
        on validation and therefore optimistic).
    secondary columns (AUDIT 4.0, never selection criteria): p < 2 GeV / p >= 2 GeV; contained / side-leaking events.
Nulls and negative results are printed as measured. Report-metric and bulk-restricted (guard) numbers are in separate
tables.
"""
from __future__ import annotations

import argparse
import json
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


def fmt(p) -> str:
    return "-" if not p else f"{p['mean'] * E4:+.2f} ± {p['se'] * E4:.2f}"


def pd(p: stats.Paired) -> dict:
    return {"mean": p.mean, "se": p.se, "z": p.z, "n": p.n}


def tagged(name: str, stage: str | None = None, kind: str = "k64"):
    r = D.row(name, stage=stage, need_guards=(kind != "k64"))
    return None if r is None else registry.row_array(r, kind=kind)


def mlp_array():
    return registry.reference_set().bench["mlp"]


def b0_mean():
    """Per-event mean of exactly the three H200 controls (stage m2b_mlp); None if not all three are registered."""
    arrs = [tagged(n, stage=D.REAL_STAGE) for n in lib.CONTROL_NAMES]
    if any(a is None for a in arrs):
        return None
    provenance.check_pairable(*arrs)
    t = arrs[0]
    return provenance.TaggedArray.make(np.mean([a.values.astype(np.float64) for a in arrs], axis=0), t.points,
                                       machine.MachineTag.from_fields(t.machine), label="B0 mean (b0h_seed0..2)")


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


def parts(a, ref_mlp, b0) -> dict:
    p = provenance.paired_delta(a, ref_mlp)
    out = {"vs_mlp": pd(p), "a_gain": p.mean <= D.PROMO_GAIN, "b_z": p.z <= D.PROMO_Z}
    if b0 is None:
        out["vs_b0"], out["c_b0"] = None, None
    else:
        q = provenance.paired_delta(a, b0)
        out["vs_b0"], out["c_b0"] = pd(q), bool(q.z < D.B0_Z)
    return out


def two_seed(a0, a1, ref) -> dict:
    t = a0
    m = provenance.TaggedArray.make(0.5 * (a0.values.astype(np.float64) + a1.values), t.points,
                                    machine.MachineTag.from_fields(t.machine), label="two-seed mean")
    provenance.check_pairable(a0, a1, ref)
    return pd(provenance.paired_delta(m, ref))


def replication(p0: dict, p1: dict | None, both: dict | None) -> dict | None:
    if p1 is None or both is None:
        return None
    m0, m1 = p0["mean"], p1["mean"]
    ok = bool(m1 <= 0.5 * D.PROMO_GAIN and (m1 < 0) == (m0 < 0) and both["mean"] <= D.PROMO_GAIN
              and both["z"] <= D.PROMO_Z)
    return {"seed1_half_threshold": m1 <= 0.5 * D.PROMO_GAIN, "same_sign": (m1 < 0) == (m0 < 0),
            "two_seed_mean": both, "replicated": ok}


def guards(name: str, main_vs_mlp: np.ndarray | None) -> dict | None:
    r = D.row(name, need_guards=True)
    if r is None:
        return None
    g = r["guards"]["vs_mlp"]
    out = {"gk": g["gk"], "gk_persistence": g["gk_persistence"], "bulk": g["bulk"],
           "bulk_persistence": g["bulk_persistence"], "report_on_stratum": g["report_on_stratum"],
           "report_on_short": g["report_on_short"], "mass": r["guards"]["mass"]}
    gk_ok = bool(g["gk_persistence"]["numbers"].get("margin", -1) >= 0 and g["gk_persistence"]["flags"]["resolved"])
    gb_ok = bool(g["bulk_persistence"]["numbers"].get("margin", -1) >= 0 and g["bulk_persistence"]["flags"]["resolved"])
    out["gk_ok"], out["bulk_ok"] = gk_ok, gb_ok
    out["g_region"] = g_region(name, main_vs_mlp)
    out["passed"] = bool(gk_ok and gb_ok and out["g_region"] is not None and out["g_region"]["passed"])
    return out


def _gr(dirname: str, cls: str):
    d = D.out_dir() / dirname / "g_region"
    mk = D.read_json(d / "g_region.json")
    if mk is None:
        return None
    ta = provenance.load_tagged(d / f"g_region_{cls}.npy", expect_machine_id=lib.MACHINE.id)
    if mk["arrays"][cls]["sha256"] != ta.sha256:
        raise RuntimeError(f"{dirname}: G-region array {cls} differs from its marker")
    return ta


def g_region(name: str, main) -> dict | None:
    if main is None or _gr(name, "flagged") is None or _gr("_mlp_std_reference", "flagged") is None:
        return None
    diffs = {c: provenance.paired_difference(_gr(name, c), _gr("_mlp_std_reference", c))
             for c in ("deep", "border_ring", "border_zbin", "low_abar", "flagged", "inner", "normalisation")}
    gain = float(main.mean())
    margin = stats.paired(diffs["flagged"] - 0.5 * main, np.zeros_like(main))
    return {"hit_term_by_class": {c: pd(stats.paired(d, np.zeros_like(d))) for c, d in diffs.items()
                                  if c != "normalisation"},
            "normalisation_term": pd(stats.paired(diffs["normalisation"], np.zeros_like(main))),
            "flagged_share_of_gain": (float(diffs["flagged"].mean()) / gain) if gain else None,
            "margin": pd(margin), "passed": bool(gain < 0 and margin.mean >= 0.0)}


def guard_candidates(sm: dict | None = None) -> list:
    """Runs that meet the promotion parts (a)-(c) on a seed, and the other seed of their rung (AUDIT 4.2 note)."""
    sm = sm or D.spec_map()
    ref, b0 = mlp_array(), b0_mean()
    hits = set()
    for n, s in sm.items():
        a = tagged(n)
        if a is None:
            continue
        pr = parts(a, ref, b0)
        if pr["a_gain"] and pr["b_z"] and pr["c_b0"]:
            hits.add(n)
            for m, t in sm.items():
                if t.rung == s.rung and m != n:
                    hits.add(m)
    return sorted(hits)


def collect(smoke: bool) -> dict:
    sm = D.spec_map()
    ref, b0 = mlp_array(), b0_mean()
    masks = secondary_masks()
    res = {"stage": D.CTX.stage, "stream_id": lib.live_stream_id(), "machine_id": lib.MACHINE.id, "smoke": smoke,
           "b0_complete": b0 is not None, "runs": {}, "primary": {}, "one_se": {}, "ema": {}, "bceil": {},
           "status": {}, "decisions": D.decisions()}
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
             "primary": s.rung in D.PRIMARY_RUNGS, **parts(a, ref, b0)}
        e["secondary_vs_mlp"] = {k: pd(provenance.paired_delta(a, ref, mask=m)) for k, m in masks.items()}
        if s.comparator and s.comparator != "mlp_std":
            c = comparator_array(s.comparator)
            e["vs_comparator"] = ({"comparator": s.comparator, **pd(provenance.paired_delta(a, c))}
                                  if c is not None else {"comparator": s.comparator, "available": False})
        main = provenance.paired_difference(a, ref)
        e["guards"] = guards(n, main)
        res["runs"][n] = e
    # replication and promotion
    for n, e in res["runs"].items():
        s = sm[n]
        partner = next((m for m, t in sm.items() if t.replicate_of == n), None)
        if partner and partner in res["runs"]:
            both = two_seed(tagged(n), tagged(partner), ref)
            e["d_replication"] = replication(e["vs_mlp"], res["runs"][partner]["vs_mlp"], both)
        elif s.replicate_of:
            e["d_replication"] = "see " + s.replicate_of
        else:
            e["d_replication"] = None
        rep = e["d_replication"] if isinstance(e["d_replication"], dict) else None
        e["promoted"] = bool(e["a_gain"] and e["b_z"] and e["c_b0"] and rep and rep["replicated"]
                             and e["guards"] and e["guards"]["passed"])
    # primary comparisons
    for rung, comp in (("B1d", "mlp_std"), ("B2b", "B2a"), ("B2c", "B2a"), ("B4f", "B2a"), ("B6c", "B6a")):
        names = [n for n, s in sm.items() if s.rung == rung]
        out = {}
        arrs = {}
        for n in names:
            s = sm[n]
            cn = "mlp_std" if comp == "mlp_std" else f"{comp}_s{s.seed}"
            a, c = tagged(n), comparator_array(cn)
            if a is None or c is None:
                out[n] = {"available": False, "comparator": cn}
                continue
            p = provenance.paired_delta(a, c)
            out[n] = {"comparator": cn, **pd(p), "a": p.mean <= D.PROMO_GAIN, "b": p.z <= D.PROMO_Z,
                      "secondary": {k: pd(provenance.paired_delta(a, c, mask=m)) for k, m in masks.items()}}
            arrs[n] = (a, c)
        if len(arrs) == 2:
            (n0, (a0, c0)), (n1, (a1, c1)) = list(arrs.items())
            provenance.check_pairable(a0, a1, c0, c1)
            t = a0
            mk = lambda x, y, lab: provenance.TaggedArray.make(0.5 * (x.values.astype(np.float64) + y.values),  # noqa
                                                                t.points, machine.MachineTag.from_fields(t.machine),
                                                                label=lab)
            both = pd(provenance.paired_delta(mk(a0, a1, "variant mean"), mk(c0, c1, "comparator mean")))
            out["replication"] = replication(out[n0], out[n1], both)
        res["primary"][rung] = out
    # B1 retention
    e5 = tagged("E5_smoke" if smoke else "E5_s2", stage=(D.SMOKE_STAGE if smoke else D.REAL_STAGE))
    if e5 is not None:
        g = provenance.paired_delta(e5, ref)
        res["b1"] = {"E5_vs_mlp": pd(g)}
        for n in ("B1d_best", "B1d_second"):
            if n in res["runs"]:
                # "retains 30-70 % of B1" (AUDIT 4.2) is defined only when B1 itself is a gain
                res["b1"][f"{n}_retained_fraction"] = (res["runs"][n]["vs_mlp"]["mean"] / g.mean) if g.mean < 0 else None
    # 1-SE (B6) per seed
    for seed in (0, 1):
        rows = [D.row(f"{r}_s{seed}") for r in ("B6a", "B6b", "B6c")]
        if all(rows):
            sel = registry.select_1se(rows=rows)
            res["one_se"][f"seed{seed}"] = {"selected": sel.selected, "best": sel.best, "table": sel.table}
        else:
            missing = [f"{r}_s{seed}" for r, x in zip(("B6a", "B6b", "B6c"), rows) if not x]
            res["one_se"][f"seed{seed}"] = {"selected": None, "best": None, "not_evaluated": missing}
    # B3 EMA
    for n, s in sm.items():
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
    return res


def markdown(res: dict) -> str:
    L = []
    w = L.append
    w(f"# Track B list (AUDIT.md section 4.2) - {'SMOKE (not numbers of record)' if res['smoke'] else 'results'}\n")
    w(f"Stage `{res['stage']}`, machine `{res['machine_id']}`, stream `{res['stream_id']}`. Validation only; "
      "1e-4 nat per event, paired, ± SE. NOT gated.\n")
    if not res["b0_complete"]:
        w("**B0 INCOMPLETE:** fewer than the three controls b0h_seed0..2 are registered on this stream; part (c) "
          "of the promotion rule is not evaluated.\n")
    w("## Report metric (K = 64)\n")
    w("| run | rung | class | k | val | Δ mlp_std | (a) ≤ −5 | (b) ≤ −3 SE | Δ B0 | (c) z < −1 | (d) replicated | "
      "(e) guards | promoted | Δ comparator |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    yn = lambda v: "-" if v is None else ("yes" if v else "no")  # noqa: E731
    for n, e in res["runs"].items():
        rep = e["d_replication"]
        rs = rep if isinstance(rep, str) else yn(rep["replicated"] if rep else None)
        vc = e.get("vs_comparator")
        w(f"| {n} | {e['rung']}{' (primary)' if e['primary'] else ''} | {e['cls']} | {e['k']} | {e['val_mean']:.6f} | "
          f"{fmt(e['vs_mlp'])} | {yn(e['a_gain'])} | {yn(e['b_z'])} | {fmt(e['vs_b0'])} | {yn(e['c_b0'])} | {rs} | "
          f"{yn(e['guards']['passed']) if e['guards'] else 'not evaluated'} | {yn(e['promoted'])} | "
          f"{(vc['comparator'] + ' ' + fmt(vc) if vc and vc.get('mean') is not None else '-')} |")
    w("\n## Primary comparisons (AUDIT 4.2; the same thresholds and replication rule)\n")
    w("| rung | run | comparator | Δ | (a) | (b) | p < 2 GeV | p ≥ 2 GeV | contained | side-leaking |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for rung, out in res["primary"].items():
        for n, c in out.items():
            if n == "replication":
                continue
            if not c.get("mean") and c.get("available") is False:
                w(f"| {rung} | {n} | {c['comparator']} | not available | | | | | | |")
                continue
            sec = c["secondary"]
            w(f"| {rung} | {n} | {c['comparator']} | {fmt(c)} | {yn(c['a'])} | {yn(c['b'])} | {fmt(sec['p<2GeV'])} | "
              f"{fmt(sec['p>=2GeV'])} | {fmt(sec['contained'])} | {fmt(sec['side-leaking'])} |")
        rp = out.get("replication")
        if rp:
            w(f"| {rung} | two-seed mean | | {fmt(rp['two_seed_mean'])} | replicated: {yn(rp['replicated'])} | | | | | |")
    if res.get("b1"):
        fr = [f"{k} {v:.2f}" for k, v in res["b1"].items() if k.endswith("fraction") and v is not None]
        w(f"\nB1 (E5) vs mlp_std {fmt(res['b1']['E5_vs_mlp'])}; retained fraction of B1 by B1d: "
          + (", ".join(fr) if fr else "not defined (B1 shows no gain against mlp_std)"))
    w("\n## Guards (bulk-restricted and other guard numbers; own table)\n")
    w("| run | G-K Δ | G-K margin ok | G-bulk Δ (stratum) | G-bulk margin ok | report metric on stratum | "
      "G-region flagged share | G-region passed | mass beyond zb |")
    w("|---|---|---|---|---|---|---|---|---|")
    any_g = False
    for n, e in res["runs"].items():
        g = e["guards"]
        if not g:
            continue
        any_g = True
        gr = g["g_region"]
        share = gr.get("flagged_share_of_gain") if gr else None
        share_s = f"{share:.2f}" if share is not None else "-"
        gr_ok = yn(gr["passed"]) if gr else "-"
        beyond = g["mass"].get("beyond_zb", {}).get("mean", "-")
        w(f"| {n} | {fmt(g['gk'])} | {yn(g['gk_ok'])} | {fmt(g['bulk'])} | {yn(g['bulk_ok'])} | "
          f"{fmt(g['report_on_stratum'])} | {share_s} | {gr_ok} | {beyond} |")
    if not any_g:
        w("No run met the promotion parts (a)-(c); no guard evaluation was required (AUDIT 4.2 note).")
    w("\n## 1-SE rule (B6)\n")
    for k, v in res["one_se"].items():
        if v.get("selected") is None:
            w(f"- {k}: not evaluated (missing or diverged: {', '.join(v.get('not_evaluated', []))})")
        else:
            w(f"- {k}: selected **{v['selected']}** (best {v['best']})")
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
    conv = res["decisions"].get("convergence") or {}
    if conv:
        w("\n## Convergence (AUDIT 4.0)\n")
        for n, c in conv.items():
            w(f"- {n}: " + (f"moved {c['move'] * E4:.2f} over {c['window_steps']} steps -> "
                            f"{'converged' if c['converged'] else 'NOT converged, reruns ' + str(c.get('reruns'))}"
                            if c.get("available") else "not available"))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
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


if __name__ == "__main__":
    sys.exit(main())
