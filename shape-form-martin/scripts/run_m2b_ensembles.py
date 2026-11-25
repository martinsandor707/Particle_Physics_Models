"""Track B, zero-training ensembles (PLAN.md section 6, B1): probability averages of finished 4x32 networks with
uniform weights (``sfm.models.Ensemble`` through the "ensemble" factory; ``n_params`` = sum over the members),
registered on validation with guards, plus the M-scaling over all subsets of the five stage-2 networks.

    .venv/bin/python -B shape-form-martin/scripts/run_m2b_ensembles.py             # register + subsets + scaling table
    .venv/bin/python -B shape-form-martin/scripts/run_m2b_ensembles.py --table-only  # scaling table from saved arrays

Machine tags (A6 revised, points 4-5; H200 constraints of 25 Nov 2026). Members: s2_seed11 and s2_seed12 are the
laptop's fine-tunes (records in the clone, weights identified by SHA-256); s2_seed13..15 must have been trained on
THIS machine (its run directory). Every array read here carries the live stream tag: single networks through their
registry rows on the live stream (``registry.row_array``), mlp_std from the reference set of the live tag, subsets
from the tagged subset directory; every difference goes through ``sfm.provenance.paired_difference`` /
``paired_delta``, which refuse mixed tags. No laptop array and no laptop row is read. Outputs go to this machine's
directory OUT_RESULTS/machines/<id>/runs/m2b_mlp/ (ensemble model.pt, done.json, ensemble_subsets/<points id>/,
ensemble_scaling.json).

Registered ensembles (guards on), fixed before any of them was scored:
    E5_s2              the five stage-2 networks s2_seed11..15
    E6_s2_mlpstd       E5_s2 + mlp_std.pt (mlp_std first, as the factory's include_mlp_std puts it)
    E2_s2_seeds11-12   representative of M = 2: the lexicographically first subset of the seeds (no selection)
    E3_s2_seeds11-13   representative of M = 3, same rule
    E4_s2_seeds11-14   representative of M = 4, same rule
    E5_s1              the five stage-1 networks s1_seed11..15 (before the fine-tune), for comparison
Each is written as results/m2b_mlp/<name>/model.pt ({"state", "meta": {"model": {"factory": "ensemble", ...}}}, so
``sfm.models.load_run`` rebuilds it) and registered through ``sfm.registry.evaluate``; done.json marks it finished.

Subsets: for M = 2, 3, 4 every subset of the five stage-2 networks (10 + 10 + 5) is scored with exactly the call
the registry makes (``common.test_nll`` inside ``sfm.metric.deterministic()``, model in eval mode) and saved as
results/m2b_mlp/ensemble_subsets/s2_<seeds>.npy (existing files are kept: resumable). The representatives' arrays
are checked bit for bit against their registered val_nll.npy. Not registered.

Scaling table (results/m2b_mlp/ensemble_scaling.json), per M = 1..5 over all C(5, M) subsets:
    mean / SD / min / max of the subset validation means;
    paired gain against mlp_std, against the subset's best member (by validation mean), against the mean of its
    members (per event) and against the best of the five single networks - each as the mean over subsets of the
    per-event difference, with the SE of that subset-averaged per-event array (events as the unit; the networks
    are held fixed, so the SE does not include the network-to-network variation, which the SD over subsets shows).

GPU work runs under sfm.gpulock.gpu_lock (run_m2b_lib.run_on_gpu: only with >= 3 GB free device memory, else the
lock is released and the job retried), one ensemble (or the subset batch) at a time. Never opens the test split.
"""
import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2b_lib as lib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import metric, models, paths, provenance, registry, stats, train  # noqa: E402

common = train.C
S2_NAMES = [f"s2_seed{s}" for s in lib.S1_SEEDS]
S1_NAMES = [f"s1_seed{s}" for s in lib.S1_SEEDS]


def _rep_name(seeds) -> str:
    return f"E{len(seeds)}_s2_seeds{seeds[0]}-{seeds[-1]}"


ENSEMBLES = {
    "E5_s2": {"members": S2_NAMES, "include_mlp_std": False},
    "E6_s2_mlpstd": {"members": S2_NAMES, "include_mlp_std": True},
    **{_rep_name(lib.S1_SEEDS[:m]): {"members": S2_NAMES[:m], "include_mlp_std": False} for m in (2, 3, 4)},
    "E5_s1": {"members": S1_NAMES, "include_mlp_std": False},
}


def subset_dir() -> Path:
    return lib.OUT / "ensemble_subsets" / provenance.points_id(registry.live_stream()["points"])


def _member_refs(spec) -> list:
    return [lib.member(n) for n in spec["members"]]


def _member_file(ref) -> Path:
    return models._run_dir(ref, models.BuildContext()) / "model.pt"


def check_members(names) -> dict:
    """Every member file must be the checkpoint of a row of the LIVE stream (its own registration or re-score on this
    machine), by SHA-256 (review B3). Returns {ref: file SHA-256}."""
    out = {}
    for n in names:
        ref = lib.member(n)
        f = _member_file(ref)
        row = lib.registered_row(n, f)
        if row is None:
            raise SystemExit(f"member {n} ({ref}): no row on the live stream {lib.live_stream_id()} with this file's "
                             "SHA-256; register / re-score it on this machine first")
        out[ref] = lib.sha256_file(f)
    return out


def _member_info(refs, include_mlp_std) -> dict:
    out = {}
    for r in refs:
        f = models._run_dir(r, models.BuildContext()) / "model.pt"
        obj = torch.load(f, map_location="cpu", weights_only=False)
        tag = obj.get("machine_tag") or (obj.get("meta") or {}).get("machine_tag")
        out[r] = {"sha256": lib.sha256_file(f),
                  "trained_on": (tag or {}).get("id") or lib.machine.LAPTOP_RECORD_LABEL}
    if include_mlp_std:
        out["mlp_std"] = {"sha256": lib.sha256_file(lib.paths.PION_RESULTS_DIR / "mlp_std.pt"),
                          "trained_on": "earlier study (shape-form-proton)"}
    return out


def register_ensemble(name: str, spec: dict) -> dict:
    out = lib.OUT / name
    done = lib.done_marker(out / "done.json")
    if done is not None:
        print(f"[{name}] already done on this machine")
        return done
    refs = _member_refs(spec)
    check_members(spec["members"])

    def job():
        val = registry.load_validation()
        ctx = models.BuildContext(device=common.DEVICE)
        kwargs = {"members": list(refs), "include_mlp_std": bool(spec["include_mlp_std"])}
        ens = models.build("ensemble", ctx, **kwargs).eval()
        paths.check_output_path(out)
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / "model.pt"
        info = _member_info(refs, spec["include_mlp_std"])
        if model_path.exists():
            obj = torch.load(model_path, map_location=common.DEVICE, weights_only=False)
            st = ens.state_dict()
            if set(obj["state"]) != set(st) or not all(torch.equal(obj["state"][k], st[k]) for k in st):
                raise RuntimeError(f"{paths.portable(model_path)} holds other weights than its members")
        else:
            train.atomic_save({"state": {k: v.detach().clone() for k, v in ens.state_dict().items()},
                               "meta": {"model": {"factory": "ensemble", "kwargs": kwargs}, "members": info},
                               "machine_tag": lib.MACHINE.to_dict(),
                               "saved_utc": datetime.now(timezone.utc).isoformat()}, model_path)
        config = {"factory": "ensemble", **kwargs, "weights": "uniform (probability average)"}
        with lib.Resources() as r_eval:
            row = lib.registered_row(name, model_path, need_guards=True)
            if row is None:
                row = registry.evaluate(ens, name, lib.STAGE, val,
                                        meta={"config": config, "checkpoint": str(model_path),
                                              "notes": "Track B zero-training ensemble (uniform weights), B1",
                                              "members": info, "n_members": len(ens.members),
                                              "trained_on": {"zero_training_ensemble_of": info}},
                                        guards=True)
            else:
                print(f"[{name}] already registered ({row['timestamp']}); not evaluated again")
        print(f"[{name}] k {row['k']}  val {row['val_mean']:.6f}  vs mlp {row['vs_mlp']['mean']:+.6f} +- "
              f"{row['vs_mlp']['se']:.6f}  vs formula {row['vs_formula']['mean']:+.6f} +- {row['vs_formula']['se']:.6f}")
        done = {"name": name, "stage": lib.STAGE, "config": config, "model": "model.pt",
                "model_sha256": lib.sha256_file(model_path), "k": row["k"], "val_mean": row["val_mean"],
                "registry_timestamp": row["timestamp"], "resources": {"evaluate": r_eval.as_dict()},
                "machine_tag": lib.MACHINE.to_dict(), "stream_id": lib.live_stream_id(), "batch": BATCH,
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        train.atomic_write_json(done, out / "done.json")
        del ens, val
        return done
    done = lib.run_on_gpu(f"m2b ensemble {name}", job, min_free_mb=3000)
    lib.free_cuda()
    return done


def subset_file(seeds) -> Path:
    return subset_dir() / f"s2_{'-'.join(str(s) for s in seeds)}.npy"


def score_subsets() -> dict:
    todo = [c for m in (2, 3, 4) for c in itertools.combinations(lib.S1_SEEDS, m) if not subset_file(c).exists()]
    timing = {}
    if todo:
        st = registry.live_stream()
        refs = {s: lib.member(f"s2_seed{s}") for s in lib.S1_SEEDS}
        member_sha = check_members([f"s2_seed{s}" for s in lib.S1_SEEDS])

        def job():
            val = registry.load_validation()
            registry.check_validation(val, st["reference"])
            ctx = models.BuildContext(device=common.DEVICE)
            nets = {s: models.load_run(refs[s], ctx).eval() for s in lib.S1_SEEDS}
            with lib.Resources() as r_all:
                for c in todo:
                    ens = models.Ensemble([nets[s] for s in c]).eval()
                    with lib.Resources() as r1, metric.deterministic():
                        nll = common.test_nll(ens, val.showers)        # the registry's call, unmodified metric
                    if not np.isfinite(nll).all():
                        raise FloatingPointError(f"subset {c}: non-finite NLL")
                    provenance.save_tagged(subset_file(c), nll, st["points"], st["machine"],
                                           label=f"ensemble subset s2 {c}",
                                           extra={"members": [refs[s] for s in c],
                                                  "member_sha256": {refs[s]: member_sha[refs[s]] for s in c}})
                    timing["-".join(map(str, c))] = r1.as_dict()
                    print(f"subset {c}: val {float(np.mean(nll, dtype=np.float64)):.6f}  ({r1.seconds:.0f}s)")
            timing["_all"] = r_all.as_dict()
            del nets, val
        lib.run_on_gpu(f"m2b ensemble subsets ({len(todo)} left)", job, min_free_mb=3000)
        lib.free_cuda()
    return timing


def _live_rows() -> dict:
    sid = lib.live_stream_id()
    return {r["name"]: r for r in registry.latest(registry.read_registry(), stream=sid) if r["stage"] == lib.STAGE}


def _arr(rows, name) -> provenance.TaggedArray:
    if name not in rows:
        raise KeyError(f"{lib.STAGE}/{name} is not registered on the live stream {lib.live_stream_id()}")
    return registry.row_array(rows[name])


def _pd(p) -> dict:
    return {"mean": p.mean, "se": p.se, "z": p.z}


def _mean_tagged(arrs, label) -> provenance.TaggedArray:
    provenance.check_pairable(*arrs)
    a0 = arrs[0]
    return provenance.TaggedArray.make(np.mean([a.values.astype(np.float64) for a in arrs], axis=0), a0.points,
                                       lib.machine.MachineTag.from_fields(a0.machine), label=label)


def scaling_table() -> dict:
    st = registry.live_stream()
    rows = _live_rows()
    mlp = registry.reference_set(stream=st).bench["mlp"]
    single = {s: _arr(rows, f"s2_seed{s}") for s in lib.S1_SEEDS}
    single_mean = {s: float(single[s].values.astype(np.float64).mean()) for s in lib.S1_SEEDS}
    best_single = min(lib.S1_SEEDS, key=lambda s: single_mean[s])
    arrays = {(s,): single[s] for s in lib.S1_SEEDS}
    member_sha = check_members([f"s2_seed{s}" for s in lib.S1_SEEDS])
    for m in (2, 3, 4):
        for c in itertools.combinations(lib.S1_SEEDS, m):
            f = subset_file(c)
            arrays[c] = provenance.load_tagged(f, expect_points=st["points"], expect_machine_id=st["machine"].id,
                                               label=f"subset {c}")
            side = json.loads(provenance.sidecar_path(f).read_text())
            want = {lib.member(f"s2_seed{s}"): member_sha[lib.member(f"s2_seed{s}")] for s in c}
            if side.get("member_sha256") != want:
                raise SystemExit(f"{paths.portable(f)}: its members are not today's files (sidecar "
                                 f"{side.get('member_sha256')} vs {want}); refused")
    arrays[tuple(lib.S1_SEEDS)] = _arr(rows, "E5_s2")
    checks = {}                                                       # representatives: registered == recomputed
    for m in (2, 3, 4):
        c = tuple(lib.S1_SEEDS[:m])
        checks[_rep_name(c)] = bool(np.array_equal(arrays[c].values, _arr(rows, _rep_name(c)).values))
    zero = np.zeros(mlp.values.shape)
    agg = lambda ds: _pd(stats.paired(np.mean(ds, axis=0), zero))  # noqa: E731
    per_m, subsets = {}, {}
    for m in range(1, 6):
        cs = list(itertools.combinations(lib.S1_SEEDS, m))
        means, d_mlp, d_best_mem, d_avg_mem, d_best_single = [], [], [], [], []
        for c in cs:
            a = arrays[c]
            bm = min(c, key=lambda s: single_mean[s])
            avg = _mean_tagged([single[s] for s in c], f"member mean {c}")
            means.append(float(a.values.astype(np.float64).mean()))
            d_mlp.append(provenance.paired_difference(a, mlp))
            d_best_mem.append(provenance.paired_difference(a, single[bm]))
            d_avg_mem.append(provenance.paired_difference(a, avg))
            d_best_single.append(provenance.paired_difference(a, single[best_single]))
            subsets["-".join(map(str, c))] = {
                "M": m, "val_mean": means[-1], "best_member": bm,
                "vs_mlp": _pd(provenance.paired_delta(a, mlp)), "vs_best_member": _pd(provenance.paired_delta(a, single[bm])),
                "vs_member_mean": _pd(provenance.paired_delta(a, avg)),
                "vs_best_single": _pd(provenance.paired_delta(a, single[best_single]))}
        per_m[m] = {"n_subsets": len(cs), "val_mean_avg": float(np.mean(means)),
                    "val_mean_sd": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                    "val_mean_min": float(np.min(means)), "val_mean_max": float(np.max(means)),
                    "vs_mlp": agg(d_mlp), "vs_best_member": agg(d_best_mem), "vs_member_mean": agg(d_avg_mem),
                    "vs_best_single": agg(d_best_single),
                    "vs_mlp_subset_range": [float(min(d.mean() for d in d_mlp)), float(max(d.mean() for d in d_mlp))]}
    extra = {}
    e5, e6 = arrays[tuple(lib.S1_SEEDS)], _arr(rows, "E6_s2_mlpstd")
    extra["E6_minus_E5"] = _pd(provenance.paired_delta(e6, e5))
    extra["E6_minus_mlp"] = _pd(provenance.paired_delta(e6, mlp))
    s1 = {s: _arr(rows, f"s1_seed{s}") for s in lib.S1_SEEDS}
    e5s1 = _arr(rows, "E5_s1")
    bm1 = min(lib.S1_SEEDS, key=lambda s: float(s1[s].values.astype(np.float64).mean()))
    extra["E5_s1"] = {"val_mean": float(e5s1.values.astype(np.float64).mean()), "best_member": bm1,
                      "vs_best_member": _pd(provenance.paired_delta(e5s1, s1[bm1])),
                      "vs_member_mean": _pd(provenance.paired_delta(e5s1, _mean_tagged(list(s1.values()), "s1 mean"))),
                      "vs_mlp": _pd(provenance.paired_delta(e5s1, mlp)),
                      "E5_s2_minus_E5_s1": _pd(provenance.paired_delta(e5, e5s1))}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "metric": registry.METRIC_OF_RECORD, "stream_id": st["stream_id"], "machine_id": st["machine"].id,
           "members": {n: lib.member(n) for n in S2_NAMES}, "single_val_means": single_mean,
           "best_single": best_single, "per_M": per_m, "subsets": subsets, "representative_bitwise_equal": checks,
           "extra": extra, "batch": BATCH}
    train.atomic_write_json(out, lib.OUT / "ensemble_scaling.json")
    print(f"\nscaling over subsets of the five stage-2 networks (best single: s2_seed{best_single}, "
          f"{single_mean[best_single]:.6f}); stream {st['stream_id']}")
    print(f"{'M':>2} {'n':>3} {'val mean':>10} {'SD':>9} {'vs mlp_std':>20} {'vs best member':>20} "
          f"{'vs member mean':>20} {'vs best single':>20}")
    for m, d in per_m.items():
        f = lambda p: f"{p['mean']:+.6f}+-{p['se']:.6f}"  # noqa: E731
        print(f"{m:>2} {d['n_subsets']:>3} {d['val_mean_avg']:>10.6f} {d['val_mean_sd']:>9.6f} {f(d['vs_mlp']):>20} "
              f"{f(d['vs_best_member']):>20} {f(d['vs_member_mean']):>20} {f(d['vs_best_single']):>20}")
    print("representatives bit-equal to registered arrays:", checks)
    return out


BATCH: dict = {}


def representatives_verdict(checks: dict) -> tuple:
    """(ok, failing names): the acceptance of R6 - all three representatives bit-equal (review B3)."""
    expected = {_rep_name(tuple(lib.S1_SEEDS[:m])) for m in (2, 3, 4)}
    bad = sorted(k for k in expected if not checks.get(k, False))
    return not bad, bad


def main(argv=None) -> int:
    global BATCH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args(argv)
    BATCH = lib.start("run_m2b_ensembles")
    if not args.table_only:
        for name, spec in ENSEMBLES.items():
            register_ensemble(name, spec)
        t = score_subsets()
        if t:
            tf = subset_dir() / "timing.json"
            prev = json.loads(tf.read_text()) if tf.exists() else {}
            prev.update({k: v for k, v in t.items() if k != "_all"})
            prev.setdefault("_runs", []).append(t.get("_all"))
            train.atomic_write_json(prev, tf)
    out = scaling_table()
    ok, bad = representatives_verdict(out["representative_bitwise_equal"])
    if not ok:
        print(f"FAIL: representatives not bit-equal to their registered arrays: {bad} (acceptance of R6)")
        return 1
    print("PASS: all three representatives bit-equal to their registered arrays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
