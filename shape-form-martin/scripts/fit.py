"""Run one fit from a config file: build the model, train (resumable), evaluate on validation, register.

    uv run python -B shape-form-martin/scripts/fit.py shape-form-martin/configs/<stage>/<name>.toml
    uv run python -B shape-form-martin/scripts/fit.py <config> --dry-run      # check config, build model, no training

Steps
    1. ``sfm.train.load_run_config``: typed TOML loader ([run], [model], [fit], [evaluate]; unknown keys fail).
    2. Takes the GPU lock (``sfm.gpulock``, ``cache/gpu.lock``) for the whole job, so heavy jobs never overlap.
    3. Loads train and validation from the cache (never the test split), seeds ``random/numpy/torch`` with
       ``run.seed``, builds the model with the factory registry ``sfm.models`` (``[model] factory = ...``).
    4. ``sfm.train.fit`` into ``results/<stage>/<name>/`` (checkpoint.pt, model.pt, fit_done.json, fit.log);
       re-running resumes an interrupted fit or loads the finished one.
    5. ``sfm.registry.evaluate``: unmodified ``common.test_nll`` on validation inside ``deterministic()``, paired
       deltas vs both benchmarks, optional guards; appends to ``results/registry.jsonl``.
    6. Writes ``results/<stage>/<name>/done.json`` (the marker ``scripts/fit_queue.py`` skips on).
"""
import argparse
import contextlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfm import data, gpulock, machine, models, paths, registry, train  # noqa: E402

common = train.C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="TOML run file (see configs/README section in shape-form-martin/README.md)")
    ap.add_argument("--dry-run", action="store_true", help="load the config and build the model only")
    ap.add_argument("--no-lock", action="store_true", help="do not take the GPU lock (CPU smoke tests only)")
    ap.add_argument("--no-evaluate", action="store_true", help="skip the registry evaluation")
    args = ap.parse_args(argv)

    tag = machine.startup("fit.py")                                  # E10: CUDA or a hard failure; TF32 off; caps
    gitst = machine.git_state()
    rc = train.load_run_config(args.config)
    print(f"run {rc.stage}/{rc.name}  (config {paths.portable(rc.source)}, hash {rc.hash()})  model {rc.model_factory} "
          f"{rc.model_kwargs}  -> {paths.portable(rc.out_dir)}  (machine {tag.id})")
    legacy = paths.REPO_RESULTS / rc.stage / rc.name / train.RUN_DONE_FILE
    if legacy.exists() and not args.dry_run:
        raise SystemExit(f"{paths.portable(legacy)}: this run was finished before machine tags existed (a laptop "
                         "record); it is not retrained or continued here. Use another run name.")
    if rc.done_file.exists() and not args.dry_run:
        print(f"already done: {paths.portable(rc.done_file)}")
        return 0
    lock = (contextlib.nullcontext() if (args.no_lock or args.dry_run)
            else gpulock.gpu_lock(f"fit {rc.stage}/{rc.name}"))
    t0 = time.time()
    with lock:
        sets, extras = data.load_sets(("train", "val"), device=common.DEVICE)
        train.seed_everything(rc.seed)
        ctx = models.BuildContext(device=common.DEVICE, train=sets["train"], seed=rc.seed,
                                  results_dir=paths.REPO_RESULTS)
        model = models.build(rc.model_factory, ctx, **rc.model_kwargs)
        n_params = models.count_params(model)
        print(f"model: {type(model).__name__}, {n_params} parameters "
              f"({sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable)")
        if args.dry_run:
            print("dry run: config valid, model built; nothing trained")
            return 0
        meta = {"model": {"factory": rc.model_factory, "kwargs": rc.model_kwargs}, "run": rc.raw,
                "run_hash": rc.hash(), "config_path": paths.portable(rc.source)}
        res = train.fit(model, sets["train"], sets["val"], rc.fit, meta=meta)
        row = None
        if rc.evaluate.registry and not args.no_evaluate:
            val = registry.ValidationSet(sets["val"], extras["val"]["event_number"])
            row = registry.evaluate(
                model, rc.name, rc.stage, val,
                meta={"k": rc.evaluate.k if rc.evaluate.k is not None else n_params, "C": rc.evaluate.C,
                      "seed": rc.seed, "config": rc.raw, "config_hash": rc.hash(), "checkpoint": str(res.model_path),
                      "notes": rc.notes, "fit": res.summary()},
                guards=rc.evaluate.guards)
            print(f"registered: val {row['val_mean']:.6f}  vs mlp {row['vs_mlp']['mean']:+.6f} +- "
                  f"{row['vs_mlp']['se']:.6f}  vs formula {row['vs_formula']['mean']:+.6f} +- "
                  f"{row['vs_formula']['se']:.6f}")
        done = {"name": rc.name, "stage": rc.stage, "config": paths.portable(rc.source), "run_hash": rc.hash(),
                "machine_tag": tag.to_dict(), "git": gitst, "caps": machine.caps(),
                "fit": res.summary(), "registered": row is not None,
                "val_mean": row["val_mean"] if row else None, "seconds": time.time() - t0,
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        train.atomic_write_json(done, rc.done_file)
    print(f"done: {paths.portable(rc.done_file)}")
    if res.skipped:
        print(f"WARNING: {res.skipped_loss} batches skipped for a non-finite loss, {res.skipped_grad} for a "
              "non-finite gradient")
    return 0


if __name__ == "__main__":
    sys.exit(main())
