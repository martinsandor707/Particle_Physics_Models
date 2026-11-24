"""Track B, zero-training ensembles (PLAN.md section 6, B1): probability averages of finished 4x32 networks with
uniform weights (``sfm.models.Ensemble`` through the "ensemble" factory; ``n_params`` = sum over the members),
registered on validation with guards, plus the M-scaling over all subsets of the five stage-2 networks.

    uv run python -B shape-form-martin/scripts/run_m2b_ensembles.py             # register + subsets + scaling table
    uv run python -B shape-form-martin/scripts/run_m2b_ensembles.py --table-only  # scaling table from saved arrays

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

from sfm import metric, models, registry, stats, train  # noqa: E402

common = train.C
S2 = [f"{lib.STAGE}/s2_seed{s}" for s in lib.S1_SEEDS]
S1 = [f"{lib.STAGE}/s1_seed{s}" for s in lib.S1_SEEDS]


def _rep_name(seeds) -> str:
    return f"E{len(seeds)}_s2_seeds{seeds[0]}-{seeds[-1]}"


ENSEMBLES = {
    "E5_s2": {"members": S2, "include_mlp_std": False},
    "E6_s2_mlpstd": {"members": S2, "include_mlp_std": True},
    **{_rep_name(lib.S1_SEEDS[:m]): {"members": S2[:m], "include_mlp_std": False} for m in (2, 3, 4)},
    "E5_s1": {"members": S1, "include_mlp_std": False},
}
SUBSET_DIR = lib.OUT / "ensemble_subsets"


def _member_shas(spec) -> dict:
    out = {r: lib.sha256_file(lib.paths.RESULTS_DIR / r / "model.pt") for r in spec["members"]}
    if spec["include_mlp_std"]:
        out["mlp_std"] = lib.sha256_file(lib.paths.PION_RESULTS_DIR / "mlp_std.pt")
    return out


def register_ensemble(name: str, spec: dict) -> dict:
    out = lib.OUT / name
    if (out / "done.json").exists():
        print(f"[{name}] already done")
        return json.loads((out / "done.json").read_text())

    def job():
        val = registry.load_validation()
        ctx = models.BuildContext(device=common.DEVICE)
        kwargs = {"members": list(spec["members"]), "include_mlp_std": bool(spec["include_mlp_std"])}
        ens = models.build("ensemble", ctx, **kwargs).eval()
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / "model.pt"
        shas = _member_shas(spec)
        if model_path.exists():
            obj = torch.load(model_path, map_location=common.DEVICE, weights_only=False)
            st = ens.state_dict()
            if set(obj["state"]) != set(st) or not all(torch.equal(obj["state"][k], st[k]) for k in st):
                raise RuntimeError(f"{model_path} holds other weights than its members")
        else:
            train.atomic_save({"state": {k: v.detach().clone() for k, v in ens.state_dict().items()},
                               "meta": {"model": {"factory": "ensemble", "kwargs": kwargs}, "member_sha256": shas},
                               "saved_utc": datetime.now(timezone.utc).isoformat()}, model_path)
        config = {"factory": "ensemble", **kwargs, "weights": "uniform (probability average)"}
        with lib.Resources() as r_eval:
            row = lib.registered_row(name, model_path, need_guards=True)
            if row is None:
                row = registry.evaluate(ens, name, lib.STAGE, val,
                                        meta={"config": config, "checkpoint": str(model_path),
                                              "notes": "Track B zero-training ensemble (uniform weights), B1",
                                              "member_sha256": shas, "n_members": len(ens.members)},
                                        guards=True)
            else:
                print(f"[{name}] already registered ({row['timestamp']}); not evaluated again")
        print(f"[{name}] k {row['k']}  val {row['val_mean']:.6f}  vs mlp {row['vs_mlp']['mean']:+.6f} +- "
              f"{row['vs_mlp']['se']:.6f}  vs formula {row['vs_formula']['mean']:+.6f} +- {row['vs_formula']['se']:.6f}")
        done = {"name": name, "stage": lib.STAGE, "config": config, "model": str(model_path),
                "model_sha256": lib.sha256_file(model_path), "k": row["k"], "val_mean": row["val_mean"],
                "registry_timestamp": row["timestamp"], "resources": {"evaluate": r_eval.as_dict()},
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        train.atomic_write_json(done, out / "done.json")
        del ens, val
        return done
    done = lib.run_on_gpu(f"m2b ensemble {name}", job, min_free_mb=3000)
    lib.free_cuda()
    return done


def subset_file(seeds) -> Path:
    return SUBSET_DIR / f"s2_{'-'.join(str(s) for s in seeds)}.npy"


def score_subsets() -> dict:
    todo = [c for m in (2, 3, 4) for c in itertools.combinations(lib.S1_SEEDS, m) if not subset_file(c).exists()]
    timing = {}
    if todo:
        SUBSET_DIR.mkdir(parents=True, exist_ok=True)

        def job():
            val = registry.load_validation()
            registry.check_validation(val)
            ctx = models.BuildContext(device=common.DEVICE)
            nets = {s: models.load_run(r, ctx).eval() for s, r in zip(lib.S1_SEEDS, S2)}
            with lib.Resources() as r_all:
                for c in todo:
                    ens = models.Ensemble([nets[s] for s in c]).eval()
                    with lib.Resources() as r1, metric.deterministic():
                        nll = common.test_nll(ens, val.showers)        # the registry's call, unmodified metric
                    if not np.isfinite(nll).all():
                        raise FloatingPointError(f"subset {c}: non-finite NLL")
                    f = subset_file(c)
                    tmp = f.with_name(f".{f.stem}.tmp{os.getpid()}.npy")
                    np.save(tmp, nll)
                    os.replace(tmp, f)
                    timing["-".join(map(str, c))] = r1.as_dict()
                    print(f"subset {c}: val {float(np.mean(nll, dtype=np.float64)):.6f}  ({r1.seconds:.0f}s)")
            timing["_all"] = r_all.as_dict()
            del nets, val
        lib.run_on_gpu(f"m2b ensemble subsets ({len(todo)} left)", job, min_free_mb=3000)
        lib.free_cuda()
    return timing


def _row(name):
    return lib.registered_row(name)


def _arr(name) -> np.ndarray:
    r = _row(name)
    if r is None:
        raise KeyError(f"{lib.STAGE}/{name} is not registered")
    a = np.load(lib.paths.RESULTS_DIR / r["val_nll_path"])
    if registry._sha256_array(a) != r["val_nll_sha256"]:
        raise RuntimeError(f"{r['val_nll_path']}: content differs from its registry row")
    return a


def _pd(p) -> dict:
    return {"mean": p.mean, "se": p.se, "z": p.z}


def scaling_table() -> dict:
    mlp = np.load(lib.paths.M0_DIR / "mlp_val_nll.npy").astype(np.float64)
    single = {s: _arr(f"s2_seed{s}").astype(np.float64) for s in lib.S1_SEEDS}
    single_mean = {s: float(single[s].mean()) for s in lib.S1_SEEDS}
    best_single = min(lib.S1_SEEDS, key=lambda s: single_mean[s])
    arrays = {(s,): single[s] for s in lib.S1_SEEDS}
    for m in (2, 3, 4):
        for c in itertools.combinations(lib.S1_SEEDS, m):
            arrays[c] = np.load(subset_file(c)).astype(np.float64)
    arrays[tuple(lib.S1_SEEDS)] = _arr("E5_s2").astype(np.float64)
    checks = {}                                                       # representatives: registered == recomputed
    for m in (2, 3, 4):
        c = tuple(lib.S1_SEEDS[:m])
        checks[_rep_name(c)] = bool(np.array_equal(np.load(subset_file(c)), _arr(_rep_name(c))))
    per_m, subsets = {}, {}
    for m in range(1, 6):
        cs = list(itertools.combinations(lib.S1_SEEDS, m))
        means, d_mlp, d_best_mem, d_avg_mem, d_best_single = [], [], [], [], []
        for c in cs:
            a = arrays[c]
            bm = min(c, key=lambda s: single_mean[s])
            avg = np.mean([single[s] for s in c], axis=0)
            means.append(float(a.mean()))
            d_mlp.append(a - mlp)
            d_best_mem.append(a - single[bm])
            d_avg_mem.append(a - avg)
            d_best_single.append(a - single[best_single])
            subsets["-".join(map(str, c))] = {
                "M": m, "val_mean": float(a.mean()), "best_member": bm,
                "vs_mlp": _pd(stats.paired(a, mlp)), "vs_best_member": _pd(stats.paired(a, single[bm])),
                "vs_member_mean": _pd(stats.paired(a, avg)), "vs_best_single": _pd(stats.paired(a, single[best_single]))}
        zero = np.zeros(len(mlp))
        agg = lambda ds: _pd(stats.paired(np.mean(ds, axis=0), zero))  # noqa: E731
        per_m[m] = {"n_subsets": len(cs), "val_mean_avg": float(np.mean(means)),
                    "val_mean_sd": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                    "val_mean_min": float(np.min(means)), "val_mean_max": float(np.max(means)),
                    "vs_mlp": agg(d_mlp), "vs_best_member": agg(d_best_mem), "vs_member_mean": agg(d_avg_mem),
                    "vs_best_single": agg(d_best_single),
                    "vs_mlp_subset_range": [float(min(d.mean() for d in d_mlp)), float(max(d.mean() for d in d_mlp))]}
    # E6 and E5_s1 in context
    extra = {}
    e5, e6 = arrays[tuple(lib.S1_SEEDS)], _arr("E6_s2_mlpstd").astype(np.float64)
    extra["E6_minus_E5"] = _pd(stats.paired(e6, e5))
    extra["E6_minus_mlp"] = _pd(stats.paired(e6, mlp))
    s1 = {s: _arr(f"s1_seed{s}").astype(np.float64) for s in lib.S1_SEEDS}
    e5s1 = _arr("E5_s1").astype(np.float64)
    bm1 = min(lib.S1_SEEDS, key=lambda s: float(s1[s].mean()))
    extra["E5_s1"] = {"val_mean": float(e5s1.mean()), "best_member": bm1,
                      "vs_best_member": _pd(stats.paired(e5s1, s1[bm1])),
                      "vs_member_mean": _pd(stats.paired(e5s1, np.mean([s1[s] for s in lib.S1_SEEDS], axis=0))),
                      "vs_mlp": _pd(stats.paired(e5s1, mlp)), "E5_s2_minus_E5_s1": _pd(stats.paired(e5, e5s1))}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "metric": registry.METRIC_OF_RECORD, "single_val_means": single_mean, "best_single": best_single,
           "per_M": per_m, "subsets": subsets, "representative_bitwise_equal": checks, "extra": extra}
    train.atomic_write_json(out, lib.OUT / "ensemble_scaling.json")
    print(f"\nscaling over subsets of the five stage-2 networks (best single: s2_seed{best_single}, "
          f"{single_mean[best_single]:.6f})")
    print(f"{'M':>2} {'n':>3} {'val mean':>10} {'SD':>9} {'vs mlp_std':>20} {'vs best member':>20} "
          f"{'vs member mean':>20} {'vs best single':>20}")
    for m, d in per_m.items():
        f = lambda p: f"{p['mean']:+.6f}+-{p['se']:.6f}"  # noqa: E731
        print(f"{m:>2} {d['n_subsets']:>3} {d['val_mean_avg']:>10.6f} {d['val_mean_sd']:>9.6f} {f(d['vs_mlp']):>20} "
              f"{f(d['vs_best_member']):>20} {f(d['vs_member_mean']):>20} {f(d['vs_best_single']):>20}")
    print("representatives bit-equal to registered arrays:", checks)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.table_only:
        for name, spec in ENSEMBLES.items():
            register_ensemble(name, spec)
        t = score_subsets()
        if t:
            prev = json.loads((SUBSET_DIR / "timing.json").read_text()) if (SUBSET_DIR / "timing.json").exists() else {}
            prev.update({k: v for k, v in t.items() if k != "_all"})
            prev.setdefault("_runs", []).append(t.get("_all"))
            train.atomic_write_json(prev, SUBSET_DIR / "timing.json")
    scaling_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
