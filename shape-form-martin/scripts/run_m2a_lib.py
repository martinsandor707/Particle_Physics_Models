"""Shared helpers of the Track-A (M2A) run scripts ``scripts/run_m2a*.py`` (AUDIT.md section 4.1).

    Spec          one run of the pre-registered list: registry name, factory + kwargs, seed, provider, steps,
                  learning-rate class, pre-registered comparator, class (pilot / control / base / primary /
                  exploratory / stage2), rung id
    fit_config    training configuration T_A (AUDIT 4.1): batch 256, k_train 16, iid folds, cosine lr -> lr/15,
                  eval_every 500, k_val 64, selection "last", smooth 0, dedicated fold generator, clip 5;
                  weight_decay 0 for poly providers, 1e-4 for net providers
    run_spec      execute one run under ``sfm.gpulock.gpu_lock``: build (``sfm.models`` factories registered by
                  ``sfm.models.dist_factory``), record the initialisation (init.json), optional step-0 registry row,
                  ``sfm.train.fit`` (resumable from checkpoint.pt), ``sfm.registry.evaluate`` (validation only),
                  done.json. Re-running skips finished runs and resumes interrupted ones.
    Data          train / val from ``sfm.data.load_sets(("train", "val"))`` on the CPU, moved to the GPU inside the
                  lock for each job and released afterwards (the GPU is shared with the Track-B worker).

Nothing here opens the test split; all evaluation goes through ``sfm.registry.evaluate``.
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import data as sfm_data  # noqa: E402
from sfm import gpulock, models, paths, registry, train  # noqa: E402
from sfm.models import dist_factory  # noqa: E402,F401  (registers the "component" and "mixture" factories)

common = train.C
STAGE = "m2a_dist"
OUT = paths.RESULTS_DIR / STAGE
PLR_FILE = OUT / "plr.json"
GPU_LOG = OUT / "gpu_time.jsonl"


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# run specification
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Spec:
    name: str                       # registry name = results/m2a_dist/<name>/
    rung: str                       # A0, A1, ..., A10-5, A12, A11k, plr
    factory: str                    # "component" | "mixture"
    kwargs: dict
    seed: int
    provider: str                   # "poly" | "net"
    cls: str                        # pilot | control | base | primary | primary-net | exploratory | stage2 | rerun
    comparator: Optional[str] = None
    steps: int = 4000
    lr: Optional[float] = None      # None: the frozen P-lr value of the provider class
    k_train: int = 16
    init_eval: bool = False         # register the model at step 0 as <name>_init
    notes: str = ""

    @property
    def out_dir(self) -> Path:
        return OUT / self.name

    @property
    def done_file(self) -> Path:
        return self.out_dir / "done.json"

    def done(self) -> bool:
        return self.done_file.exists()


def frozen_lr(provider: str) -> float:
    if not PLR_FILE.exists():
        raise RuntimeError(f"{PLR_FILE} missing: the learning-rate pilot has not been decided")
    d = json.loads(PLR_FILE.read_text())
    key = {"poly": "lr_poly", "net": "lr_net"}[provider]
    if d.get(key) is None:
        raise RuntimeError(f"{PLR_FILE}: {key} not frozen yet")
    return float(d[key])


def fit_config(spec: Spec) -> train.FitConfig:
    lr = spec.lr if spec.lr is not None else frozen_lr(spec.provider)
    return train.FitConfig(steps=int(spec.steps), lr=float(lr), lr_end=float(lr) / 15.0, batch=256,
                           k_train=int(spec.k_train), fold="iid", k_val=64, eval_every=500, selection="last",
                           ema_decay=None, smooth=0.0, weight_decay=(1e-4 if spec.provider == "net" else 0.0),
                           seed=int(spec.seed), name=spec.name, out_dir=str(spec.out_dir), ckpt_every=500, clip=5.0,
                           val_seed=12345, val_batch=128, fold_rng="dedicated", deterministic=False).validate()


# ----------------------------------------------------------------------------------------------------------------------
# data and GPU
# ----------------------------------------------------------------------------------------------------------------------
_DATA = {}


def data_cpu():
    """(sets, extras) of train and val on the CPU, loaded once per process (never the test split)."""
    if "sets" not in _DATA:
        log("loading train / val from the cache (CPU)")
        _DATA["sets"], _DATA["extras"] = sfm_data.load_sets(("train", "val"), device="cpu")
    return _DATA["sets"], _DATA["extras"]


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def on_gpu(label: str, fn, min_free_mb: float = 2500.0, poll: float = 30.0, max_wait: float = 12 * 3600.0):
    """``fn()`` under ``gpu_lock(label)`` once the device has ``min_free_mb`` free; the lock is released while
    waiting for memory (as in the Track-B helper ``run_m2b_lib.run_on_gpu``, re-implemented here). Appends the
    time spent holding the lock to results/m2a_dist/gpu_time.jsonl."""
    t_wait = time.time()
    while True:
        with gpulock.gpu_lock(f"m2a {label}", log=log):
            free = torch.cuda.mem_get_info()[0] / 2 ** 20 if torch.cuda.is_available() else math.inf
            if free >= min_free_mb:
                t0 = time.time()
                try:
                    return fn()
                finally:
                    free_cuda()
                    OUT.mkdir(parents=True, exist_ok=True)
                    with open(GPU_LOG, "a") as f:
                        f.write(json.dumps({"label": label, "seconds": time.time() - t0,
                                            "waited": t0 - t_wait,
                                            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
        if time.time() - t_wait > max_wait:
            raise RuntimeError(f"{label}: device memory did not become free within {max_wait:.0f} s")
        log(f"[on_gpu] {label}: only {free:.0f} MB free (need {min_free_mb:.0f}); lock released, retry in {poll:.0f} s")
        time.sleep(poll)


def gpu_seconds_total() -> float:
    if not GPU_LOG.exists():
        return 0.0
    return sum(json.loads(l)["seconds"] for l in GPU_LOG.read_text().splitlines() if l.strip())


# ----------------------------------------------------------------------------------------------------------------------
# registry helpers
# ----------------------------------------------------------------------------------------------------------------------
def sha256_file(path) -> str:
    return train.sha256_file(Path(path))


def registered_row(name: str, ckpt=None, need_guards: bool = False):
    """Latest registry row of (STAGE, name) [with this checkpoint's SHA-256] [with guards]."""
    want = sha256_file(ckpt) if ckpt is not None and Path(ckpt).exists() else None
    found = None
    for r in registry.read_registry():
        if r.get("stage") != STAGE or r.get("name") != name:
            continue
        if ckpt is not None and r.get("checkpoint_sha256") != want:
            continue
        if need_guards and not r.get("guards"):
            continue
        found = r
    return found


def val_array(name: str) -> np.ndarray:
    """The registered per-event validation NLL of a run (checked against the registry row's SHA-256)."""
    row = registered_row(name)
    if row is None:
        raise KeyError(f"{name}: not in the registry")
    arr = np.load(paths.RESULTS_DIR / row["val_nll_path"])
    if registry._sha256_array(arr) != row["val_nll_sha256"]:
        raise RuntimeError(f"{name}: per-event array differs from its registry row")
    return arr


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ----------------------------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------------------------
def run_spec(spec: Spec) -> dict:
    """Execute (or skip, or resume) one run; returns its done.json content."""
    if spec.done():
        return json.loads(spec.done_file.read_text())
    cfg = fit_config(spec)
    sets, extras = data_cpu()

    def job():
        t0 = time.time()
        dev = common.DEVICE
        tr, va = sets["train"].to(dev), sets["val"].to(dev)
        val = registry.ValidationSet(va, extras["val"]["event_number"])
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        resuming = (spec.out_dir / train.CHECKPOINT).exists() or (spec.out_dir / train.DONE_FILE).exists()
        init_file = spec.out_dir / "init.json"
        train.seed_everything(spec.seed)
        ctx = models.BuildContext(device=dev, train=tr, seed=spec.seed, log=log, skip_init=resuming)
        tb = time.time()
        model = models.build(spec.factory, ctx, **spec.kwargs)
        build_s = time.time() - tb
        info = getattr(model, "init_info", None)
        if not resuming:
            train.atomic_write_json(_jsonable({"init_info": info, "build_seconds": build_s,
                                               "n_params": int(model.n_params()), "spec": asdict(spec)}), init_file)
        init = json.loads(init_file.read_text()) if init_file.exists() else {}
        info = init.get("init_info") or {}
        log(f"[{spec.name}] built {spec.factory} {spec.kwargs}: {model.n_params()} constants, "
            f"kl_map {info.get('kl_map')}, {'resuming' if resuming else f'init {build_s:.0f}s'}")
        meta_common = {"seed": spec.seed, "k": int(model.n_params()), "start": spec.kwargs.get("start")
                       or spec.kwargs.get("net_from") or spec.kwargs.get("init_ref"),
                       "kl_map": info.get("kl_map"), "init": info, "rung": spec.rung, "cls": spec.cls,
                       "comparator": spec.comparator, "provider": spec.provider}
        # step-0 registration (A0 parity, A9 Delta at step 0)
        if spec.init_eval and registered_row(spec.name + "_init") is None:
            if resuming:
                raise RuntimeError(f"{spec.name}: the step-0 row is missing but the run has already started")
            row0 = registry.evaluate(model, spec.name + "_init", STAGE, val,
                                     meta={**meta_common, "config": {"spec": asdict(spec), "step": 0},
                                           "notes": f"{spec.rung} at step 0 (before training)"})
            log(f"[{spec.name}] step 0 registered: val {row0['val_mean']:.6f} vs formula "
                f"{row0['vs_formula']['mean'] * 1e4:+.3f}e-4")
        meta = {"model": {"factory": spec.factory, "kwargs": spec.kwargs}, "spec": asdict(spec), "init": info}
        res = train.fit(model, tr, va, cfg, meta=_jsonable(meta), log=log)
        row = registered_row(spec.name, res.model_path)
        if row is None:
            row = registry.evaluate(model, spec.name, STAGE, val,
                                    meta={**meta_common, "config": {"spec": asdict(spec), "fit": cfg.result_dict()},
                                          "config_hash": cfg.hash(), "checkpoint": str(res.model_path),
                                          "notes": spec.notes or None, "fit": res.summary()})
        log(f"[{spec.name}] registered: val {row['val_mean']:.6f}  vs formula {row['vs_formula']['mean'] * 1e4:+.3f}"
            f" +- {row['vs_formula']['se'] * 1e4:.3f}  vs mlp {row['vs_mlp']['mean'] * 1e4:+.3f}e-4  "
            f"skipped {res.skipped_loss}/{res.skipped_grad}")
        done = {"name": spec.name, "spec": asdict(spec), "fit": res.summary(), "history": res.history,
                "val_mean": row["val_mean"], "k": row["k"], "kl_map": info.get("kl_map"),
                "build_seconds": init.get("build_seconds"), "job_seconds": time.time() - t0,
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        train.atomic_write_json(_jsonable(done), spec.done_file)
        del model, tr, va, val
        return done

    return on_gpu(spec.name, job)


def rebuild(name: str, device=None):
    """The fitted model of a finished run (``sfm.models.load_run`` with the factories of dist_factory)."""
    dev = device or common.DEVICE
    ctx = models.BuildContext(device=dev, train=None, seed=0, log=log, skip_init=True)
    return models.load_run(f"{STAGE}/{name}", ctx)
