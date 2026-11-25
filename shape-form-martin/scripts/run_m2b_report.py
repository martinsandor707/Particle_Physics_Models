"""Track B report tables from the registry, the done-markers and the stored per-event arrays (no model is run, no
test data). Writes THIS machine's runs/m2b_mlp/summary.json (OUT_RESULTS/machines/<id>/...) and prints the tables.

    .venv/bin/python -B shape-form-martin/scripts/run_m2b_report.py

Machine tags (A6 revised; user decision P5): only rows and arrays of the LIVE stream are read (this machine's
registry; ``registry.row_array`` checks tag and SHA-256; every delta goes through ``sfm.provenance.paired_delta``).
The control is b0h_seed0..2 (trained on this machine): the B0 mean, the seed scatter and the "vs B0 mean" column use
these three only. The laptop's b0_seed0, re-scored here, is shown as "b0_seed0 (laptop-trained record)" and enters
no scatter, mean or rule; its paired difference to b0h_seed0 (equal initial weights and batch order, two machines)
is reported without a pass/fail.

Columns: name, free weights k, stage-1 epochs run / best epoch (common.fit, 0-based), validation NLL (K = 64, the
registry value), paired delta +- SE against mlp_std and the component formula (registry), against B0 (per-event
mean of the three B0 copies; computed here from the stored arrays), and for guarded entries the K = 1024 and bulk
guard columns (G-K: mean at K = 1024 and the paired delta vs mlp_std at K = 1024 with the persistence verdict;
G-bulk: paired delta vs mlp_std on the stratum zb >= 1221 mm with the persistence verdict; "res" = verdict resolved
at 2 SE). Plus: B0 seed scatter, stage-2 bookkeeping (selected step, skipped batches), wall-clock and peak CUDA
memory, and the ensemble scaling table (from results/m2b_mlp/ensemble_scaling.json).
"""
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2b_lib as lib  # noqa: E402

import numpy as np  # noqa: E402

from sfm import machine, paths, provenance, registry, stats, train  # noqa: E402

ORDER = ([f"s1_seed{s}" for s in lib.S1_SEEDS] + [f"s2_seed{s}" for s in lib.S1_SEEDS]
         + list(lib.CONTROL_NAMES) + [lib.LAPTOP_CONTROL_RECORD]
         + ["E2_s2_seeds11-12", "E3_s2_seeds11-13", "E4_s2_seeds11-14", "E5_s2", "E6_s2_mlpstd", "E5_s1"])
LABEL = {lib.LAPTOP_CONTROL_RECORD: "b0_seed0 (laptop-trained record)"}


def _pd(p) -> dict:
    return {"mean": p.mean, "se": p.se, "z": p.z}


def _f(p, w=8) -> str:
    return f"{p['mean'] * 1e4:+{w}.2f} ± {p['se'] * 1e4:4.2f}" if p else ""


def main() -> int:
    batch = lib.start("run_m2b_report")
    sid = lib.live_stream_id()
    rows = {r["name"]: r for r in registry.latest(registry.read_registry(), stream=sid) if r["stage"] == lib.STAGE}
    have = [n for n in ORDER if n in rows]
    tarr = {n: registry.row_array(rows[n]) for n in have}               # tag + SHA-256 checked, live stream only
    provenance.check_pairable(*tarr.values())
    arrays = {n: t.values.astype(np.float64) for n, t in tarr.items()}
    b0_names = [n for n in lib.CONTROL_NAMES if n in arrays]
    b0_mean = None
    if b0_names:
        t0 = tarr[b0_names[0]]
        b0_mean = provenance.TaggedArray.make(np.mean([arrays[n] for n in b0_names], axis=0), t0.points,
                                              machine.MachineTag.from_fields(t0.machine), label="B0 mean (b0h)")

    table = []
    for n in have:
        r = rows[n]
        done_f = lib.OUT / n / "done.json"
        if not done_f.exists() and n in lib.LAPTOP_TRAINED:
            done_f = lib.REPO_OUT / n / "done.json"                   # training bookkeeping of a laptop record
        done = json.loads(done_f.read_text()) if done_f.exists() else {}
        e = {"name": LABEL.get(n, n), "k": r["k"], "val_mean": r["val_mean"], "vs_mlp": r["vs_mlp"],
             "vs_formula": r["vs_formula"], "seed": r.get("seed"), "checkpoint": r.get("checkpoint"),
             "timestamp": r["timestamp"], "trained_on": r.get("trained_on"), "scored_on": r.get("scored_on")}
        if b0_mean is not None:
            e["vs_b0_mean"] = _pd(provenance.paired_delta(tarr[n], b0_mean))
        if "stage1" in done:
            s1 = done["stage1"]
            e["stage1"] = {k: s1[k] for k in ("epochs_run", "best_epoch", "best_val_k16", "early_stopped",
                                              "skipped_loss", "skipped_grad", "epochs_with_record",
                                              "mean_train_seconds_per_epoch")}
        if "fit" in done:
            fs = done["fit"]
            e["stage2"] = {k: fs.get(k) for k in ("step", "selected_step", "selected_val", "best_val", "final_val",
                                                  "skipped_loss", "skipped_grad", "resumed_from", "seconds")}
        if "resources" in done:
            res = done["resources"]
            segs = res.get("segments") if isinstance(res, dict) else None
            if segs:
                e["resources"] = {"segments": len(segs),
                                  "fit_seconds_sum": sum(s["fit"]["seconds"] for s in segs if s.get("fit")),
                                  "eval_seconds_sum": sum(s["evaluate"]["seconds"] for s in segs if s.get("evaluate")),
                                  "peak_alloc_mb_fit": max((s["fit"]["peak_alloc_mb"] or 0) for s in segs),
                                  "peak_alloc_mb_eval": max((s["evaluate"]["peak_alloc_mb"] or 0) for s in segs),
                                  "resumed": [s.get("resumed_from_epoch", s.get("resumed_from_step")) for s in segs]}
            elif isinstance(res, dict) and "evaluate" in res:
                e["resources"] = {"segments": 1, "eval_seconds_sum": res["evaluate"]["seconds"],
                                  "peak_alloc_mb_eval": res["evaluate"]["peak_alloc_mb"]}
        g = r.get("guards")
        if g:
            gm, gf = g["vs_mlp"], g["vs_formula"]
            bulk = registry.row_array(r, kind="bulk_zb1221").values
            e["guards"] = {"val_mean_k1024": g["val_mean_k1024"], "n_stratum": g["n_stratum"], "n_short": g["n_short"],
                           "n_bulk_nan": g["n_bulk_nan"], "bulk_mean": float(np.nanmean(bulk.astype(np.float64))),
                           "gk_vs_mlp": gm["gk"], "gk_vs_formula": gf["gk"],
                           "gk_persist_mlp": {"passed": gm["gk_persistence"]["passed"],
                                              "resolved": gm["gk_persistence"]["flags"]["resolved"],
                                              "fraction": gm["gk_persistence"]["numbers"]["fraction"]},
                           "bulk_vs_mlp": gm["bulk"], "bulk_vs_formula": gf["bulk"],
                           "report_on_stratum_vs_mlp": gm["report_on_stratum"],
                           "report_on_short_vs_mlp": gm["report_on_short"],
                           "bulk_persist_mlp": {"passed": gm["bulk_persistence"]["passed"],
                                                "resolved": gm["bulk_persistence"]["flags"]["resolved"],
                                                "fraction": gm["bulk_persistence"]["numbers"]["fraction"]},
                           "mass": g["mass"]}
        table.append(e)

    # B0 scatter
    b0 = None
    if b0_names:
        means = [float(arrays[n].mean()) for n in b0_names]
        b0 = {"names": b0_names, "val_means": means,
              "sd_over_seeds": float(np.std(means, ddof=1)) if len(means) > 1 else None,
              "range": float(max(means) - min(means)),
              "pairwise": {f"{a}-{b}": _pd(provenance.paired_delta(tarr[a], tarr[b]))
                           for a, b in itertools.combinations(b0_names, 2)},
              "vs_mlp": {n: rows[n]["vs_mlp"] for n in b0_names}}
    laptop_vs_h200_seed0 = None
    if lib.LAPTOP_CONTROL_RECORD in tarr and "b0h_seed0" in tarr:          # reported, no pass/fail (review 3(ii))
        laptop_vs_h200_seed0 = _pd(provenance.paired_delta(tarr[lib.LAPTOP_CONTROL_RECORD], tarr["b0h_seed0"]))
    # fine-tune gain per seed (stage 2 - stage 1)
    ft = {s: _pd(provenance.paired_delta(tarr[f"s2_seed{s}"], tarr[f"s1_seed{s}"])) for s in lib.S1_SEEDS
          if f"s2_seed{s}" in arrays and f"s1_seed{s}" in arrays}
    s2_names = [f"s2_seed{s}" for s in lib.S1_SEEDS if f"s2_seed{s}" in arrays]
    s2_scatter = ({"val_means": [float(arrays[n].mean()) for n in s2_names],
                   "sd_over_seeds": float(np.std([arrays[n].mean() for n in s2_names], ddof=1))}
                  if len(s2_names) > 1 else None)
    s1_names = [f"s1_seed{s}" for s in lib.S1_SEEDS if f"s1_seed{s}" in arrays]
    s1_scatter = ({"val_means": [float(arrays[n].mean()) for n in s1_names],
                   "sd_over_seeds": float(np.std([arrays[n].mean() for n in s1_names], ddof=1))}
                  if len(s1_names) > 1 else None)
    scal_f = lib.OUT / "ensemble_scaling.json"
    scaling = json.loads(scal_f.read_text()) if scal_f.exists() else None
    if scaling is not None and scaling.get("stream_id") != sid:
        raise RuntimeError(f"{paths.portable(scal_f)} belongs to stream {scaling.get('stream_id')}, live is {sid}")
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stream_id": sid,
           "machine_id": lib.MACHINE.id, "batch": batch, "control_names": list(lib.CONTROL_NAMES),
           "laptop_b0_seed0_minus_b0h_seed0": laptop_vs_h200_seed0,
           "metric": registry.METRIC_OF_RECORD, "units": "nat per event", "table": table, "b0_scatter": b0,
           "s1_scatter": s1_scatter, "s2_scatter": s2_scatter, "finetune_gain_s2_minus_s1": ft,
           "ensemble_scaling": scaling and {"per_M": scaling["per_M"], "extra": scaling["extra"],
                                            "best_single": scaling["best_single"],
                                            "representative_bitwise_equal": scaling["representative_bitwise_equal"]}}
    train.atomic_write_json(out, lib.OUT / "summary.json")

    print("deltas in units of 1e-4 nat (paired, ± SE); val = registry K=64 validation mean")
    print(f"{'name':<18} {'k':>6} {'ep/best':>8} {'val K64':>10} {'Δ mlp_std':>15} {'Δ formula':>15} {'Δ B0mean':>15} "
          f"{'val K1024':>10} {'G-K Δ mlp':>15} {'pass/res':>8} {'bulk Δ mlp':>15} {'pass/res':>8}")
    for e in table:
        s1 = e.get("stage1")
        ep = f"{s1['epochs_run']}/{s1['best_epoch']}" if s1 else ""
        g = e.get("guards")
        gk = g and g["gk_vs_mlp"]
        print(f"{e['name']:<18} {e['k']:>6} {ep:>8} {e['val_mean']:>10.6f} {_f(e['vs_mlp']):>15} "
              f"{_f(e['vs_formula']):>15} {_f(e.get('vs_b0_mean')):>15} "
              f"{(g['val_mean_k1024'] if g else float('nan')):>10.6f} {_f(gk):>15} "
              f"{(str(int(g['gk_persist_mlp']['passed'])) + '/' + str(int(g['gk_persist_mlp']['resolved'])) if g else ''):>8} "
              f"{_f(g and g['bulk_vs_mlp']):>15} "
              f"{(str(int(g['bulk_persist_mlp']['passed'])) + '/' + str(int(g['bulk_persist_mlp']['resolved'])) if g else ''):>8}")
    print("\nstage 2 / B0 bookkeeping")
    for e in table:
        if "stage2" in e:
            s = e["stage2"]
            r = e.get("resources", {})
            print(f"  {e['name']:<12} selected step {s['selected_step']:>5}  best val K16 {s['best_val']:.6f}  "
                  f"final val K16 {s['final_val']:.6f}  skipped loss/grad {s['skipped_loss']}/{s['skipped_grad']}  "
                  f"fit {r.get('fit_seconds_sum', float('nan')):.0f}s  eval {r.get('eval_seconds_sum', float('nan')):.0f}s  "
                  f"peak {r.get('peak_alloc_mb_fit', float('nan')):.0f}/{r.get('peak_alloc_mb_eval', float('nan')):.0f} MB  "
                  f"resumed {r.get('resumed')}")
        if "stage1" in e:
            s = e["stage1"]
            r = e.get("resources", {})
            print(f"  {e['name']:<12} epochs {s['epochs_run']} best {s['best_epoch']} (val K16 {s['best_val_k16']:.6f})  "
                  f"skipped loss/grad {s['skipped_loss']}/{s['skipped_grad']} ({s['epochs_with_record']} epochs "
                  f"observed)  {s['mean_train_seconds_per_epoch']:.1f} s/epoch  fit {r.get('fit_seconds_sum', 0):.0f}s  "
                  f"peak {r.get('peak_alloc_mb_fit', float('nan')):.0f} MB  resumed {r.get('resumed')}")
    if b0:
        print(f"\nB0 seed scatter: means {[round(m, 6) for m in b0['val_means']]}  SD {b0['sd_over_seeds']}  "
              f"range {b0['range']:.6f}")
        for k, p in b0["pairwise"].items():
            print(f"  {k}: {_f(p)} e-4")
    if s1_scatter:
        print(f"stage-1 seed SD {s1_scatter['sd_over_seeds']:.6f}; stage-2 seed SD "
              f"{s2_scatter['sd_over_seeds'] if s2_scatter else float('nan'):.6f}")
    for s, p in ft.items():
        print(f"  fine-tune gain seed {s}: {_f(p)} e-4")
    if laptop_vs_h200_seed0:
        print(f"b0_seed0 (laptop-trained) - b0h_seed0 (this machine), both scored here: {_f(laptop_vs_h200_seed0)} e-4 "
              "(reported, no pass/fail)")
    print(f"\nwrote {paths.portable(lib.OUT / 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
