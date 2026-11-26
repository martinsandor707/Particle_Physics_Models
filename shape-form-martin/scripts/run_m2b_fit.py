"""Track B fits from TOML run files (stage 2 fine-tunes and the B0 control): the steps of ``scripts/fit.py`` plus
per-phase wall-clock and peak CUDA memory, and no duplicate registry rows on a restart.

    .venv/bin/python -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/b0h_seed0.toml [...]

Machine tags (PLAN.md section 13, A6 revised; H200 constraints of 25 Nov 2026): the run directory is THIS machine's
OUT_RESULTS/machines/<machine id>/runs/<stage>/<name> (outside the clone on the H200). A run whose laptop done.json
is in the clone (the eight laptop-trained networks, e.g. b0_seed0) is a record: it is refused here, never retrained
or resumed (re-score it with scripts/rescore_registered.py). Inputs from the clone (stage-1 networks of the
fine-tunes, mlp_std.pt) are read-only. Every done.json carries the machine tag and the git state.

For every config, in the order given (a config whose run directory holds this machine's done.json is skipped):
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

from sfm import data, models, paths, registry, train  # noqa: E402

common = train.C
BATCH: dict = {}
VERIFY_ONLY = False


def run_config(path: str) -> dict:
    rc = train.load_run_config(path)
    print(f"run {rc.stage}/{rc.name}  (config {paths.portable(rc.source)}, hash {rc.hash()})  model {rc.model_factory} "
          f"{rc.model_kwargs}  -> {paths.portable(rc.out_dir)}")
    rec = lib.laptop_record_done(rc.name) if rc.stage == lib.STAGE else None
    if rec is not None:
        raise SystemExit(f"{rc.stage}/{rc.name} was trained on the laptop ({paths.portable(rec)} is a record): it is "
                         "never retrained or continued here. Re-score it with scripts/rescore_registered.py; the H200 "
                         "controls are b0h_seed0..2 (P5).")
    done = lib.done_marker(rc.done_file)
    if done is not None:
        print(f"already done on this machine: {paths.portable(rc.done_file)}")
        return done
    lib.check_pinned_config(rc)                                              # N5: tracked file, pinned recipe
    init = lib.verify_init(rc.name, rc.model_factory, rc.model_kwargs)        # B2: pre-registered + SHA-256
    print(f"starting point verified: {init['factory']} {init['run'] or ''} file {init['file_sha256'][:12]} weights "
          f"{init['weights_sha256'][:12]} ({init['trained_on']})")
    t0 = time.time()

    def job():
        sets, extras = data.load_sets(("train", "val"), device=common.DEVICE)
        train.seed_everything(rc.seed)
        ctx = models.BuildContext(device=common.DEVICE, train=sets["train"], seed=rc.seed,
                                  results_dir=paths.REPO_RESULTS)     # from_run: the laptop's stage-1 networks
        model = models.build(rc.model_factory, ctx, **rc.model_kwargs)
        lib.check_built_weights(model, init)
        if VERIFY_ONLY:
            print("verify-only: starting point and built weights agree; nothing trained or written")
            return {"verified_init": init}, None
        n_params = models.count_params(model)
        print(f"model: {type(model).__name__}, {n_params} parameters "
              f"({sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable)")
        meta = {"model": {"factory": rc.model_factory, "kwargs": rc.model_kwargs}, "run": rc.raw,
                "run_hash": rc.hash(), "config_path": paths.portable(rc.source), "init_from": init}
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
                              "checkpoint": str(res.model_path), "notes": rc.notes, "fit": res.summary(),
                              "trained_on": {"machine_id": lib.MACHINE.id, "source": "trained in this job",
                                             "init_from": init}},
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
        done = {"name": rc.name, "stage": rc.stage, "config": paths.portable(rc.source), "run_hash": rc.hash(),
                "machine_tag": lib.MACHINE.to_dict(), "stream_id": lib.live_stream_id(), "batch": BATCH,
                "init_from": init,
                "fit": res.summary(), "registered": row is not None,
                "val_mean": row["val_mean"] if row else None, "seconds": time.time() - t0,
                "resources": resources, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        train.atomic_write_json(done, rc.done_file)
        del model, sets
        return done, res
    done, res = lib.run_on_gpu(f"m2b fit {rc.stage}/{rc.name}", job, min_free_mb=3000)
    lib.free_cuda()
    if res is None:                                                      # --verify-init-only
        return done
    print(f"done: {paths.portable(rc.done_file)}")
    if res.skipped:
        print(f"WARNING: {res.skipped_loss} batches skipped for a non-finite loss, {res.skipped_grad} for a "
              "non-finite gradient")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--verify-init-only", action="store_true",
                    help="verify the starting points and the built weights, then stop (nothing trained or written)")
    args = ap.parse_args(argv)
    global BATCH, VERIFY_ONLY
    VERIFY_ONLY = bool(args.verify_init_only)
    BATCH = lib.start("run_m2b_fit")
    for c in args.configs:                                            # fail early on a broken file
        train.load_run_config(c)
    for c in args.configs:
        run_config(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
