"""Re-score the eight laptop-trained Track B networks on this machine (run-book step 8, row R4 of PLAN.md section
16.4; spec point 8 of transfer/REVIEW_STATUS.md section 4). Validation only; NEVER opens the test split.

    .venv/bin/python -B shape-form-martin/scripts/rescore_registered.py                     # the eight networks
    .venv/bin/python -B shape-form-martin/scripts/rescore_registered.py --emulate-laptop    # + R1b (grid 144)

For each of s1_seed11..15, s2_seed11, s2_seed12, b0_seed0 (the laptop rows of 24 Nov 2026 in the tracked
results/registry.jsonl, schema 1, read-only):
  1. ``model.pt`` is taken from the clone (results/m2b_mlp/<name>/model.pt, a copied input) and its SHA-256 must equal
     the laptop row's ``checkpoint_sha256``; otherwise that network is not scored (reported, exit code 1).
  2. ``sfm.registry.evaluate`` on the live stream (reference set of the live tag required; created by
     ``val_parity.py --create-refset``), with guards exactly where the laptop row had them (b0_seed0, s2_seed11,
     s2_seed12). One new row per network in THIS machine's registry (results/machines/<id>/registry.jsonl), with
     ``trained_on`` = the laptop record label, the laptop row's timestamp and hashes. A network already re-scored on
     the live stream with the same checkpoint is not scored again (idempotent).
  3. Checks (tolerance checks / table comparisons; never deltas across machines):
       grid 1056   the predicted table of PLAN.md section 16.3 (means within 1e-6, delta vs mlp_std within 0.01e-4) and
                   the guard table D of the review (K = 1024 and bulk means within 1e-6)
       grid 144    (the laptop itself) the new arrays against the laptop's registered arrays: bit-identical expected
       any grid    mass monitors within 1e-4 relative per event of the laptop's files;
                   ``--emulate-laptop``: R1b, the laptop stream emulated here (grid 144) against the laptop's
                   registered val_nll.npy, every event within 1e-4 and means within 1e-6 (relaxed clause reported)
Report: results/machines/<id>/checks/rescore/report_<utc>.json. Exit code 0 if all applicable checks passed.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2b_lib as lib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import machine, metric, models, paths, provenance, refstream, registry  # noqa: E402

common = registry.C
PREDICTED_1056 = {   # PLAN.md section 16.3 (validation, K = 64): mean, delta vs mlp_std in 1e-4
    "s1_seed11": (18.1202702, 21.87), "s1_seed12": (18.1200090, 19.26), "s1_seed13": (18.1199865, 19.04),
    "s1_seed14": (18.1193732, 12.90), "s1_seed15": (18.1214207, 33.38), "s2_seed11": (18.1182652, 1.82),
    "s2_seed12": (18.1184227, 3.40), "b0_seed0": (18.1181833, 1.00)}
GUARDS_1056 = {      # transfer/REVIEW_STATUS.md measurement D: K = 1024 mean, bulk mean on the stratum
    "b0_seed0": (18.1156983, 18.0430490), "s2_seed11": (18.1157633, 18.0431667), "s2_seed12": (18.1159234, 18.0433190)}


def within(x, target, tol) -> dict:
    return {"value": x, "target": target, "tol": tol, "diff": x - target, "pass": bool(abs(x - target) <= tol)}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", default=list(lib.LAPTOP_TRAINED))
    ap.add_argument("--emulate-laptop", action="store_true", help="R1b: also score on the emulated laptop stream")
    ap.add_argument("--source-registry", default=str(registry.LEGACY_REGISTRY),
                    help="registry holding the laptop rows (default: the clone's results/registry.jsonl)")
    args = ap.parse_args(argv)
    batch = lib.start("rescore_registered")
    tag = machine.machine_tag()
    grid = tag.fields["launch_grid"]
    env = registry.RegistryEnv()
    src_rows = registry.read_registry(Path(args.source_registry))
    laptop = {}
    for r in src_rows:                                                   # the last untagged row per name
        if r.get("stage") == lib.STAGE and registry.row_stream_id(r) is None:
            laptop[r["name"]] = r
    missing = [n for n in args.names if n not in laptop]
    if missing:
        raise SystemExit(f"no laptop row for {missing} in {paths.portable(args.source_registry)}")
    st = registry.live_stream(env)
    registry.reference_set(env, stream=st, need_guards=any(laptop[n].get("guards") for n in args.names))
    hashes = provenance.laptop_record_hashes()
    report = {"created_utc": utc(), "batch": batch, "machine": tag.to_dict(), "stream_id": st["stream_id"],
              "networks": {}, "notes": []}
    ok = []

    def job():
        out = {}
        val = registry.load_validation()
        registry.check_validation(val, st["reference"])
        ctx = models.BuildContext(device=common.DEVICE, results_dir=paths.REPO_RESULTS)
        for n in args.names:
            lr = laptop[n]
            ck = paths.REPO_RESULTS / lib.STAGE / n / "model.pt"
            e = {"laptop_row": {"timestamp": lr["timestamp"], "val_mean": lr["val_mean"],
                                "checkpoint_sha256": lr["checkpoint_sha256"], "val_nll_sha256": lr["val_nll_sha256"],
                                "guards": bool(lr.get("guards"))}}
            if not ck.exists():
                e["error"] = f"{paths.portable(ck)} missing"
                out[n] = e
                continue
            sha = provenance.sha256_file(ck)
            e["checkpoint_sha256"] = sha
            e["checkpoint_matches_laptop_row"] = sha == lr["checkpoint_sha256"]
            if not e["checkpoint_matches_laptop_row"]:
                e["error"] = "checkpoint SHA-256 differs from the laptop row: not scored"
                out[n] = e
                continue
            model = models.load_run(f"repo:{lib.STAGE}/{n}", ctx).eval()
            row = registry.registered_row(lib.STAGE, n, stream_id=st["stream_id"], ckpt=ck,
                                          need_guards=bool(lr.get("guards")), path=env.registry_path)
            if row is None:
                row = registry.evaluate(
                    model, n, lib.STAGE, val,
                    meta={"k": lr["k"], "C": lr.get("C"), "seed": lr.get("seed"), "config_hash": lr.get("config_hash"),
                          "checkpoint": str(ck), "trained_on": {"label": machine.LAPTOP_RECORD_LABEL,
                                                                "laptop": machine.LAPTOP_RECORD},
                          "notes": "R4: re-score of a laptop-trained network (record); enters no scatter, mean or "
                                   "rule as a control", "rescore_of": e["laptop_row"]},
                    env=env, guards=bool(lr.get("guards")))
                e["status"] = "scored"
            else:
                e["status"] = "already re-scored on this stream (not scored again)"
            e["row"] = {"val_mean": row["val_mean"], "vs_mlp": row["vs_mlp"], "vs_formula": row["vs_formula"],
                        "val_nll_path": row["val_nll_path"], "val_nll_sha256": row["val_nll_sha256"],
                        "weights_sha256": row["weights_sha256"],
                        "guards": ({"val_mean_k1024": row["guards"]["val_mean_k1024"],
                                    "arrays": row["guards"]["arrays"]} if row.get("guards") else None)}
            if args.emulate_laptop:
                with metric.deterministic():
                    e["_emul144"] = refstream.test_nll_ref(model, val.showers, grid=refstream.LAPTOP_GRID)
            out[n] = e
            del model
        return out

    res = lib.run_on_gpu("rescore_registered (exclusive)", job, min_free_mb=3000)
    lib.free_cuda()

    for n in args.names:
        e = res[n]
        if "error" in e:
            ok.append(False)
            report["networks"][n] = e
            continue
        chk = {}
        mine = np.load(env.results_dir / e["row"]["val_nll_path"])
        rec = provenance.laptop_record(f"results/{lib.STAGE}/{n}/val_nll.npy", hashes=hashes)
        rec_arr = np.load(rec["path"]) if rec["path"] is not None else None
        chk["laptop_record"] = {k: (paths.portable(v) if isinstance(v, Path) else v) for k, v in rec.items()}
        if grid == refstream.REFERENCE_GRID and n in PREDICTED_1056:
            mu, d = PREDICTED_1056[n]
            chk["predicted_mean"] = within(e["row"]["val_mean"], mu, 1e-6 + 5e-8)
            chk["predicted_delta_vs_mlp_e4"] = within(e["row"]["vs_mlp"]["mean"] * 1e4, d, 0.01 + 0.005)
            if n in GUARDS_1056 and e["row"]["guards"]:
                g = e["row"]["guards"]
                chk["predicted_k1024_mean"] = within(g["val_mean_k1024"], GUARDS_1056[n][0], 1e-6 + 5e-8)
                bulk = np.load(env.results_dir / g["arrays"]["bulk_zb1221"]["path"])
                chk["predicted_bulk_mean"] = within(float(np.nanmean(bulk.astype(np.float64))), GUARDS_1056[n][1],
                                                    1e-6 + 5e-8)
        if grid == refstream.LAPTOP_GRID and rec_arr is not None:
            chk["laptop_registered_array"] = provenance.tolerance_check(mine, rec_arr, top=5)
            chk["laptop_registered_array"]["pass"] = chk["laptop_registered_array"]["bit_identical"]
            chk["laptop_registered_array"]["sha_equals_laptop_row"] = (provenance.sha256_array(mine)
                                                                       == e["laptop_row"]["val_nll_sha256"])
            if e["row"]["guards"]:
                for kind, rel in (("k1024", "val_nll_k1024.npy"), ("bulk_zb1221", "val_bulk_zb1221.npy")):
                    r2 = provenance.laptop_record(f"results/{lib.STAGE}/{n}/{rel}", hashes=hashes)
                    if r2["path"] is not None:
                        a = np.load(env.results_dir / e["row"]["guards"]["arrays"][kind]["path"])
                        tc = provenance.tolerance_check(a, np.load(r2["path"]), top=3)
                        chk[f"laptop_{kind}"] = {"bit_identical": tc["bit_identical"], "max_abs": tc["max_abs"],
                                                 "pass": tc["bit_identical"]}
        if e["row"]["guards"]:
            r3 = provenance.laptop_record(f"results/{lib.STAGE}/{n}/val_mass_monitors.npz", hashes=hashes)
            if r3["path"] is not None:
                with np.load(env.results_dir / e["row"]["guards"]["arrays"]["mass_monitors"]["path"]) as a, \
                        np.load(r3["path"]) as b:
                    rel = {}
                    for k in a.files:
                        if k in b.files:
                            x, y = a[k].astype(np.float64), b[k].astype(np.float64)
                            den = np.maximum(np.abs(y), 1e-30)
                            rel[k] = {"max_rel_per_event": float(np.max(np.abs(x - y) / den)),
                                      "rel_mean": float(abs(x.mean() - y.mean()) / max(abs(y.mean()), 1e-30))}
                    chk["mass_monitors"] = {"per_key": rel, "pass": all(v["max_rel_per_event"] <= 1e-4 and
                                                                        v["rel_mean"] <= 1e-6 for v in rel.values())}
        if "_emul144" in e:
            emu = e.pop("_emul144")
            if rec_arr is not None:
                tc = provenance.tolerance_check(emu, rec_arr, top=5)
                chk["R1b_emulated_144_vs_laptop"] = {k: tc[k] for k in ("max_abs", "rms", "mean_diff", "n_above",
                                                                         "bit_identical", "strict_pass", "relaxed_pass",
                                                                         "largest")}
                chk["R1b_emulated_144_vs_laptop"]["pass"] = tc["strict_pass"]
        e["checks"] = chk
        e["pass"] = all(v.get("pass", True) for v in chk.values() if isinstance(v, dict))
        ok.append(e["pass"] and e.get("checkpoint_matches_laptop_row", False))
        report["networks"][n] = e

    report["all_applicable_checks_pass"] = bool(all(ok))
    cdir = provenance.checks_dir(env.results_dir, tag.id, "rescore")
    paths.check_output_path(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    rf = cdir / f"report_{report['created_utc']}.json"

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, Path):
            return paths.portable(o)
        return o
    rf.write_text(paths.scrub_str(json.dumps(_clean(report), indent=1, default=str)) + "\n")

    print("\n============ rescore_registered summary (validation; laptop-trained records re-scored here) ============")
    print(f"machine {tag.id}, grid {grid}, stream {st['stream_id']}")
    print(f"{'name':<10} {'sha ok':>6} {'status':<12} {'val K64':>11} {'d mlp e-4':>16} {'checks':<8}")
    for n, e in report["networks"].items():
        if "error" in e:
            print(f"{n:<10} {'':>6} ERROR: {e['error']}")
            continue
        v = e["row"]
        print(f"{n:<10} {str(e['checkpoint_matches_laptop_row']):>6} {e['status'][:12]:<12} {v['val_mean']:>11.7f} "
              f"{v['vs_mlp']['mean'] * 1e4:+8.2f} +- {v['vs_mlp']['se'] * 1e4:4.2f} "
              f"{'PASS' if e['pass'] else 'FAIL'}  " +
              "; ".join(f"{k}={'ok' if c.get('pass', True) else 'FAIL'}" for k, c in e["checks"].items()
                        if isinstance(c, dict) and "pass" in c))
    print(f"report: {paths.portable(rf)}")
    print(f"ALL APPLICABLE CHECKS: {'PASS' if report['all_applicable_checks_pass'] else 'FAIL'}")
    return 0 if report["all_applicable_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
