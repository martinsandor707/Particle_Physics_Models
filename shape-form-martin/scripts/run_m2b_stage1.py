"""Track B, stage 1: independent 4x32 networks trained with the report recipe (the first half of how mlp_std.pt was
made), one per seed, registered on validation (no guards).

    uv run python -B shape-form-martin/scripts/run_m2b_stage1.py                 # seeds 11..15, finished ones skipped
    uv run python -B shape-form-martin/scripts/run_m2b_stage1.py --seeds 11 12

Recipe per seed s (PLAN.md section 6, B1; the notebook 05_mlp_pion_copy.ipynb / common.select_architecture)
    common.configure(batch=32, out_dir=results/m2b_mlp)   - common.BATCH = 32; NOTE: configure re-seeds torch and
                                                            numpy with common.SEED (= 1), so it is called first
    random / numpy / torch seeded with s                  - immediately before the model is built
    model = common.FieldModel(hidden=32, n_layers=4).to(common.DEVICE)
    common.SEED = s for the duration of common.fit, restored afterwards
                                                          - common.fit seeds its batch-shuffling generator with
                                                            common.SEED + start_epoch, so this makes the shuffle stream
                                                            seed-specific. common.SEED is otherwise only read by
                                                            common.configure (already called) and common.load_data
                                                            (never called here).
    common.fit(model, {"train", "val"}, results/m2b_mlp/s1_seed<s>/stage1_fit.pt, epochs=60, lr=3e-3, smooth=1e-4)
                                                          - UNMODIFIED: Adam, constant lr, batch 32, K = 16 fold points
                                                            from the global CUDA generator, early stopping with
                                                            patience 15 on the K = 16 validation NLL (improvement
                                                            > 1e-4), best state kept; checkpoint every 5 epochs.

Machine tags (A6 revised): outputs go to THIS machine's directory OUT_RESULTS/machines/<id>/runs/m2b_mlp/s1_seed<s>/
(outside the clone on the H200). A seed whose laptop done.json is in the clone (seeds 11-15: all of them) is a
record and is never retrained; the fine-tunes s2_seed13..15 start from those laptop networks.

Outputs in <machine dir>/runs/m2b_mlp/s1_seed<s>/
    stage1_fit.pt        common.fit's own checkpoint (resumed by common.fit itself when the job is restarted)
    stage1_monitor.json  per-epoch counters observed through passive hooks (see below) and timings
    model.pt             the selected weights in the sfm format ({"state", "meta": {"model": {"factory": "field", ...}}})
                         so that ``sfm.models.load_run("m2b_mlp/s1_seed<s>")`` and the "from_run" factory find them
    val_nll.npy          written by sfm.registry.evaluate (K = 64, the metric of record)
    resources.json       wall-clock and peak CUDA memory of the training and evaluation phases
    done.json            done-marker; a seed with done.json is skipped

Skipped-batch counters. common.fit silently skips batches with a non-finite loss or gradient and counts nothing.
Passive hooks (they return None, so they change no value) count them without modifying common.fit: a forward hook
counts the gradient-enabled forward passes (one per training batch; validation runs under no_grad), a gradient hook
on the first weight matrix counts backward passes (common.fit calls backward iff the loss is finite), and gradient
hooks on every parameter flag a batch with a non-finite gradient (accumulated on the GPU, no synchronisation).
skipped_loss = forwards - backwards, skipped_grad = flagged batches. The first validation forward after training
marks the end of an epoch; the per-epoch record is written then (an epoch re-run after a resume overwrites its
record).

Resuming. Interrupting the job and starting it again continues from the last 5-epoch checkpoint of common.fit (its
own mechanism). common.fit does not store the global CUDA generator, and it re-seeds the shuffle generator with
common.SEED + start_epoch, so a resumed run follows other fold points and permutations than an uninterrupted one
would (statistically equivalent, not bitwise). A resume is recorded in resources.json and in the monitor file.

GPU: each seed runs under sfm.gpulock.gpu_lock (training + registration), data are loaded inside the lock and
released afterwards. Never opens the test split.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2b_lib as lib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import data, registry, train  # noqa: E402

common = train.C


class SkipMonitor:
    """Passive per-epoch counters of common.fit's training loop (module docstring)."""

    def __init__(self, model, path: Path, start_epoch: int):
        self.path = Path(path)
        self.records = json.loads(self.path.read_text())["epochs"] if self.path.exists() else {}
        self.segments = json.loads(self.path.read_text()).get("segments", []) if self.path.exists() else []
        self.segments.append({"start_epoch": start_epoch, "started_utc": datetime.now(timezone.utc).isoformat()})
        self.epoch = start_epoch
        dev = next(model.parameters()).device
        self.bad = torch.zeros((), dtype=torch.int64, device=dev)
        self.batch_bad = torch.zeros((), dtype=torch.bool, device=dev)
        self.n_fwd = self.n_bwd = 0
        self.in_train = False
        self.t0 = None
        params = list(model.parameters())
        self.handles = [model.register_forward_hook(self._fwd)]
        self.handles += [p.register_hook(self._grad) for p in params]
        self.handles.append(params[0].register_hook(self._count_bwd))

    def _grad(self, g):
        self.batch_bad.logical_or_(~torch.isfinite(g).all())
        return None

    def _count_bwd(self, g):
        self.n_bwd += 1
        return None

    def _commit_batch(self):
        self.bad += self.batch_bad
        self.batch_bad.zero_()

    def _fwd(self, module, inputs, output):
        if torch.is_grad_enabled():                                   # a training batch
            if not self.in_train:
                self.in_train, self.t0 = True, time.time()
                self.n_fwd = self.n_bwd = 0
                self.bad.zero_()
                self.batch_bad.zero_()
            else:
                self._commit_batch()
            self.n_fwd += 1
        elif self.in_train:                                           # first validation forward: the epoch ended
            self._commit_batch()
            self.records[str(self.epoch)] = {"n_batches": self.n_fwd, "skipped_loss": self.n_fwd - self.n_bwd,
                                             "skipped_grad": int(self.bad), "train_seconds": time.time() - self.t0}
            self.epoch += 1
            self.in_train = False
            self.save()
        return None

    def save(self):
        train.atomic_write_json({"epochs": self.records, "segments": self.segments}, self.path)

    def close(self):
        for h in self.handles:
            h.remove()
        self.segments[-1]["ended_utc"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def totals(self, n_epochs: int) -> dict:
        recs = [self.records.get(str(e)) for e in range(n_epochs)]
        have = [r for r in recs if r is not None]
        return {"epochs_with_record": len(have), "epochs_run": n_epochs,
                "skipped_loss": sum(r["skipped_loss"] for r in have),
                "skipped_grad": sum(r["skipped_grad"] for r in have),
                "n_batches": sum(r["n_batches"] for r in have),
                "mean_train_seconds_per_epoch": (float(np.mean([r["train_seconds"] for r in have])) if have else None)}


def peek_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "start_epoch": 0, "done": False}
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {"exists": True, "start_epoch": int(ck["epoch"]) + 1, "done": bool(ck["done"])}


def run_seed(s: int) -> dict:
    name = f"s1_seed{s}"
    out = lib.OUT / name
    out.mkdir(parents=True, exist_ok=True)
    ck_path, model_path = out / "stage1_fit.pt", out / "model.pt"
    res_path = out / "resources.json"
    resources = json.loads(res_path.read_text()) if res_path.exists() else {"segments": []}

    common.configure(batch=32, out_dir=str(lib.OUT))                 # re-seeds torch/numpy with common.SEED: call first
    assert common.BATCH == 32 and common.N_FOLD == 16
    sets, extras = data.load_sets(("train", "val"), device=common.DEVICE)
    dat = {"train": sets["train"], "val": sets["val"]}
    before = peek_checkpoint(ck_path)

    train.seed_everything(s)                                          # immediately before the model is built
    model = common.FieldModel(hidden=32, n_layers=4).to(common.DEVICE)
    init_sha = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                      for v in model.state_dict().values())).hexdigest()[:16]
    mon = SkipMonitor(model, out / "stage1_monitor.json", before["start_epoch"])
    old_seed = common.SEED
    common.SEED = s
    try:
        with lib.Resources() as r_fit:
            hist = common.fit(model, dat, str(ck_path), epochs=60, lr=3e-3, smooth=1e-4)
    finally:
        common.SEED = old_seed
        mon.close()
    n_ep, best_ep = lib.stage1_epochs(hist)
    counters = mon.totals(n_ep)
    model.eval()
    summary = {"epochs_run": n_ep, "best_epoch": best_ep, "best_val_k16": float(hist["val"][best_ep]),
               "last_val_k16": float(hist["val"][-1]), "early_stopped": n_ep < 60, "init_state_sha16": init_sha,
               **{k: counters[k] for k in ("skipped_loss", "skipped_grad", "epochs_with_record",
                                            "mean_train_seconds_per_epoch")}}
    print(f"[{name}] stage 1: {n_ep} epochs, best epoch {best_ep} (val K16 {summary['best_val_k16']:.6f}), "
          f"skipped loss/grad {counters['skipped_loss']}/{counters['skipped_grad']} "
          f"(records for {counters['epochs_with_record']}/{n_ep} epochs)")

    if model_path.exists():                                           # written atomically: complete if present
        obj = torch.load(model_path, map_location=common.DEVICE, weights_only=False)
        st = model.state_dict()
        if set(obj["state"]) != set(st) or not all(torch.equal(obj["state"][k], st[k]) for k in st):
            raise RuntimeError(f"{model_path} holds other weights than {ck_path}")
    else:
        obj = {"state": {k: v.detach().clone() for k, v in model.state_dict().items()},
               "config": lib.STAGE1_RECIPE, "history": hist, "summary": summary,
               "meta": {"model": {"factory": "field", "kwargs": {"hidden": 32, "n_layers": 4, "n_freq": 4}},
                        "seed": s, "common_fit_checkpoint": lib.paths.portable(ck_path)},
               "machine_tag": lib.MACHINE.to_dict(),
               "saved_utc": datetime.now(timezone.utc).isoformat()}
        train.atomic_save(obj, model_path)

    with lib.Resources() as r_eval:
        row = lib.registered_row(name, model_path)
        if row is None:
            val = registry.ValidationSet(sets["val"], extras["val"]["event_number"])
            row = registry.evaluate(model, name, lib.STAGE, val,
                                    meta={"seed": s, "config": lib.STAGE1_RECIPE, "checkpoint": str(model_path),
                                          "notes": "Track B stage 1: report recipe (common.fit), independent seed",
                                          "stage1": summary, "common_fit_checkpoint": str(ck_path)},
                                    guards=False)
        else:
            print(f"[{name}] already registered ({row['timestamp']}); not evaluated again")
    print(f"[{name}] registered: val {row['val_mean']:.6f}  vs mlp {row['vs_mlp']['mean']:+.6f} +- "
          f"{row['vs_mlp']['se']:.6f}  vs formula {row['vs_formula']['mean']:+.6f} +- {row['vs_formula']['se']:.6f}")

    resources["segments"].append({"resumed_from_epoch": before["start_epoch"] if before["exists"] else None,
                                  "checkpoint_done_at_start": before["done"], "fit": r_fit.as_dict(),
                                  "evaluate": r_eval.as_dict(),
                                  "finished_utc": datetime.now(timezone.utc).isoformat()})
    train.atomic_write_json(resources, res_path)
    done = {"name": name, "stage": lib.STAGE, "seed": s, "recipe": lib.STAGE1_RECIPE, "stage1": summary,
            "machine_tag": lib.MACHINE.to_dict(),
            "model": str(model_path), "model_sha256": lib.sha256_file(model_path),
            "val_mean": row["val_mean"], "vs_mlp": row["vs_mlp"], "vs_formula": row["vs_formula"],
            "registry_timestamp": row["timestamp"], "resources": resources,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    train.atomic_write_json(done, out / "done.json")
    del model, sets, dat
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(lib.S1_SEEDS))
    args = ap.parse_args(argv)
    lib.start("run_m2b_stage1")
    for s in args.seeds:
        rec = lib.laptop_record_done(f"s1_seed{s}")
        if rec is not None:
            print(f"[s1_seed{s}] trained on the laptop (record {lib.paths.portable(rec)}); never retrained here")
            continue
        done_file = lib.OUT / f"s1_seed{s}" / "done.json"
        if lib.done_marker(done_file) is not None:
            print(f"[s1_seed{s}] already done on this machine: {lib.paths.portable(done_file)}")
            continue
        lib.run_on_gpu(f"m2b stage1 s1_seed{s}", lambda: run_seed(s), min_free_mb=2000)
        lib.free_cuda()
    return 0


if __name__ == "__main__":
    sys.exit(main())
