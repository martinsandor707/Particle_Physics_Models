"""Track B fits from TOML run files (stage 2 fine-tunes and the B0 control): the steps of ``scripts/fit.py`` plus
per-phase wall-clock and peak CUDA memory, and no duplicate registry rows on a restart.

    uv run python -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/b0_seed0.toml [...]

For every config, in the order given (a config whose run directory holds done.json is skipped):
    1. ``sfm.train.load_run_config`` (typed TOML loader; unknown keys fail).
    2. Under ``sfm.gpulock.gpu_lock`` for the whole job (``run_m2b_lib.run_on_gpu``: started only when >= 3 GB of
       device memory are free, otherwise the lock is released and the job retried later): train + validation from the cache (never test), seed
       random/numpy/torch with run.seed, build the model with the ``sfm.models`` factory.
    3. ``sfm.train.fit`` into results/<stage>/<name>/ (resumes from checkpoint.pt, or loads the finished run).
    4. ``sfm.registry.evaluate`` (unmodified common.test_nll on validation, K = 64, inside deterministic(); guards
       if [evaluate] guards = true) - skipped when the registry already holds a row of this name for the same
       model.pt (SHA-256), i.e. after a crash between the registry append and the done-marker.
    5. resources.json (fit / evaluation seconds and peak CUDA memory per process segment) and done.json (the
       fields of scripts/fit.py's marker plus the resources), so scripts/fit_queue.py would skip the run as well.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2b_lib as lib  # noqa: E402

import torch  # noqa: E402

from sfm import data, models, registry, train  # noqa: E402

common = train.C


def run_config(path: str) -> dict:
    rc = train.load_run_config(path)
    print(f"run {rc.stage}/{rc.name}  (config {rc.source}, hash {rc.hash()})  model {rc.model_factory} "
          f"{rc.model_kwargs}  -> {rc.out_dir}")
    if rc.done_file.exists():
        print(f"already done: {rc.done_file}")
        return json.loads(rc.done_file.read_text())
    t0 = time.time()

    def job():
        sets, extras = data.load_sets(("train", "val"), device=common.DEVICE)
        train.seed_everything(rc.seed)
        ctx = models.BuildContext(device=common.DEVICE, train=sets["train"], seed=rc.seed)
        model = models.build(rc.model_factory, ctx, **rc.model_kwargs)
        n_params = models.count_params(model)
        print(f"model: {type(model).__name__}, {n_params} parameters "
              f"({sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable)")
        meta = {"model": {"factory": rc.model_factory, "kwargs": rc.model_kwargs}, "run": rc.raw,
                "run_hash": rc.hash(), "config_path": str(rc.source)}
        ck = rc.out_dir / train.CHECKPOINT
        resumed_from = None
        if ck.exists() and not (rc.out_dir / train.DONE_FILE).exists():
            resumed_from = int(torch.load(ck, map_location="cpu", weights_only=False)["step"])
        with lib.Resources() as r_fit:
            res = train.fit(model, sets["train"], sets["val"], rc.fit, meta=meta)
        row = None
        with lib.Resources() as r_eval:
            if rc.evaluate.registry:
                row = lib.registered_row(rc.name, res.model_path, need_guards=rc.evaluate.guards)
                if row is None:
                    val = registry.ValidationSet(sets["val"], extras["val"]["event_number"])
                    row = registry.evaluate(
                        model, rc.name, rc.stage, val,
                        meta={"k": rc.evaluate.k if rc.evaluate.k is not None else n_params, "C": rc.evaluate.C,
                              "seed": rc.seed, "config": rc.raw, "config_hash": rc.hash(),
                              "checkpoint": str(res.model_path), "notes": rc.notes, "fit": res.summary()},
                        guards=rc.evaluate.guards)
                else:
                    print(f"already registered ({row['timestamp']}); not evaluated again")
        if row is not None:
            print(f"registered: val {row['val_mean']:.6f}  vs mlp {row['vs_mlp']['mean']:+.6f} +- "
                  f"{row['vs_mlp']['se']:.6f}  vs formula {row['vs_formula']['mean']:+.6f} +- "
                  f"{row['vs_formula']['se']:.6f}")
        res_path = rc.out_dir / "resources.json"
        resources = json.loads(res_path.read_text()) if res_path.exists() else {"segments": []}
        resources["segments"].append({"resumed_from_step": resumed_from, "fit": r_fit.as_dict(),
                                      "fit_seconds_total_in_run": res.seconds, "evaluate": r_eval.as_dict(),
                                      "guards": rc.evaluate.guards,
                                      "finished_utc": datetime.now(timezone.utc).isoformat()})
        train.atomic_write_json(resources, res_path)
        done = {"name": rc.name, "stage": rc.stage, "config": str(rc.source), "run_hash": rc.hash(),
                "fit": res.summary(), "registered": row is not None,
                "val_mean": row["val_mean"] if row else None, "seconds": time.time() - t0,
                "resources": resources, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        train.atomic_write_json(done, rc.done_file)
        del model, sets
        return done, res
    done, res = lib.run_on_gpu(f"m2b fit {rc.stage}/{rc.name}", job, min_free_mb=3000)
    lib.free_cuda()
    print(f"done: {rc.done_file}")
    if res.skipped:
        print(f"WARNING: {res.skipped_loss} batches skipped for a non-finite loss, {res.skipped_grad} for a "
              "non-finite gradient")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args(argv)
    for c in args.configs:                                            # fail early on a broken file
        train.load_run_config(c)
    for c in args.configs:
        run_config(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
