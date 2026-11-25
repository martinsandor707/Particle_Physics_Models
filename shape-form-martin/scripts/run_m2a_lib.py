"""Shared helpers of the Track-A (M2A) run scripts ``scripts/run_m2a*.py`` (AUDIT.md section 4.1, PLAN.md section 13
A6 revised, section 16.7 E4).

Where things go (pristine-clone rule, ``transfer/ENABLING_STATUS.md``)
    Every output of Track A is written in THIS machine's directory below the output root:
        R/machines/<machine id>/runs/<stage>/                     (R = $SFM_OUT_ROOT/shape-form-martin/results)
            templates/            back_corrected / full_box templates, train_theta.npy, templates.json
            plr.json              learning-rate pilot decision
            decisions.json        every decision of the driver (parity, drift, convergence, second seeds, dropped=[])
            gpu_time.jsonl        seconds per GPU job (information only: NO decision depends on time)
            <run name>/           checkpoint.pt, model.pt, fit_done.json, init.json, done.json, fit.log and, under
                                  scores/<points id>/, the tagged per-event arrays written by sfm.registry.evaluate
            g_region/, fisher/    guard arrays and the A6b Fisher information (guards phase)
            RESULTS.md, results.json   the report
        R/machines/<machine id>/registry.jsonl                     this machine's registry rows (sfm.registry)
    <stage> is "m2a_dist" for the real list and "m2a_smoke" for the smoke mode (``configure``), so smoke rows can never
    be mistaken for real ones. Inputs from the clone (legacy checkpoints, abar*.pt, the cache) are read-only.

    Spec          one run of the pre-registered list: registry name, factory + kwargs, seed, provider, steps,
                  learning-rate class, pre-registered comparator, class, rung id
    fit_config    training configuration T_A (AUDIT 4.1): batch 256, k_train 16, iid folds, cosine lr -> lr/15,
                  eval_every 500, k_val 64, selection "last", smooth 0, dedicated fold generator, clip 5;
                  weight_decay 0 for poly providers, 1e-4 for net providers (smoke: steps / 100, eval every steps / 4)
    run_spec      one run under ``sfm.gpulock.gpu_lock``: build, init.json, optional step-0 row, ``sfm.train.fit``
                  (resumable from this machine's checkpoint.pt; a checkpoint of another machine is refused by fit),
                  ``sfm.registry.evaluate`` (validation only, live stream; skipped if a row of this checkpoint's
                  SHA-256 exists on the live stream), done.json with the machine tag. Finished runs are skipped.
    tagged / paired   the registered arrays of the live stream (tag + SHA-256 checked) and the ONE delta function
                  ``sfm.provenance.paired_delta`` (against a run or against this machine's formula reference array)

Nothing here opens the test split; all evaluation goes through ``sfm.registry.evaluate``.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import data as sfm_data  # noqa: E402
from sfm import gpulock, machine, models, paths, provenance, registry, stats, train  # noqa: E402
from sfm.models import dist_factory  # noqa: E402,F401  (registers the "component" and "mixture" factories)

common = train.C
REAL_STAGE = "m2a_dist"
SMOKE_STAGE = "m2a_smoke"
SMOKE_ENV = "SFM_M2A_SMOKE"                    # must be "1" for the smoke mode (refused otherwise)
# The machine directory and id come from the API (sfm.machine.machine_dir_live); they are never built here. The tag
# of a computation (settings) is recorded, but identity checks use the machine id of the directory.
MACHINE_DIR = Path(machine.machine_dir_live())
MACHINE_ID = MACHINE_DIR.name
MACHINE = machine.machine_tag()                # recorded in markers (settings tag at import); not an identity
RUNS = MACHINE_DIR / "runs"                    # BuildContext.results_dir: "<stage>/<name>" -> RUNS/<stage>/<name>
SOURCE_KEYS = ("start", "init_ref", "net_from")   # factory arguments naming another run whose weights are used
CTX = SimpleNamespace(stage=REAL_STAGE, smoke=False, batch={})

# Smoke mode (end-to-end testing only; never a number of record): every fit is shortened, the factories' own
# initialisation loops too, and the guard evaluations are limited to SMOKE_MAX_GUARD_RUNGS rungs.
SMOKE_STEP_DIV = 100
SMOKE_FACTORY_KW = {"fit_to_steps": 40, "fit_to_lbfgs": 0, "init_steps": [15, 15], "init_search": 8}
SMOKE_MAX_GUARD_RUNGS = 2
CONVERGENCE_WINDOW = 1000                      # steps (AUDIT 4.0); smoke: CONVERGENCE_WINDOW // SMOKE_STEP_DIV


def configure(smoke: bool = False) -> None:
    """Select the real list (stage m2a_dist) or the smoke mode (stage m2a_smoke; refused unless SFM_M2A_SMOKE=1)."""
    if smoke and os.environ.get(SMOKE_ENV) != "1":
        raise SystemExit(f"smoke mode refused: set {SMOKE_ENV}=1 explicitly (end-to-end testing only; its rows are "
                         f"registered under stage '{SMOKE_STAGE}' and are never numbers of record)")
    CTX.stage = SMOKE_STAGE if smoke else REAL_STAGE
    CTX.smoke = bool(smoke)


def stage() -> str:
    return CTX.stage


def out_dir() -> Path:
    return RUNS / CTX.stage


def plr_file() -> Path:
    return out_dir() / "plr.json"


def decisions_file() -> Path:
    return out_dir() / "decisions.json"


def gpu_log() -> Path:
    return out_dir() / "gpu_time.jsonl"


def steps_of(real_steps: int) -> int:
    return max(4, int(real_steps) // SMOKE_STEP_DIV) if CTX.smoke else int(real_steps)


def convergence_window() -> int:
    return max(1, CONVERGENCE_WINDOW // SMOKE_STEP_DIV) if CTX.smoke else CONVERGENCE_WINDOW


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}", flush=True)


def start(what: str) -> dict:
    """E10 start-up (``sfm.machine.startup``: CUDA or a hard failure, TF32 off, caps, clean clone in pristine mode)
    and the batch record written into every done.json."""
    tag = machine.startup(what)
    from sfm import guard
    rec = {"what": what, "machine_id": MACHINE_ID, "startup_tag": tag.to_dict(), "host_id": machine.host_id(), "caps": machine.caps(),
           "git": machine.git_state(), "pristine_mode": paths.pristine_mode(), "stage": CTX.stage,
           "smoke": CTX.smoke, "out": paths.portable(out_dir()), "file_access_guard_active": guard.active()}
    CTX.batch = rec
    log(f"[{what}] machine {MACHINE_ID} (GPU {tag.fields['gpu_name']}, grid {tag.fields['launch_grid']}, torch "
        f"{tag.fields['torch']}); caps {rec['caps']}; git {rec['git']}; stage {CTX.stage}; outputs under "
        f"{rec['out']}; file-access guard {'ACTIVE' if rec['file_access_guard_active'] else 'off'}")
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# run specification
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Spec:
    name: str                       # registry name = runs/<stage>/<name>/
    rung: str                       # A0, A1, ..., A10-5, A12, A11k, plr
    factory: str                    # "component" | "mixture"
    kwargs: dict
    seed: int
    provider: str                   # "poly" | "net"
    cls: str                        # pilot | control | base | primary | primary-net | exploratory | stage2 | rerun
    comparator: Optional[str] = None
    steps: int = 4000               # the pre-registered number (smoke: divided at run time, see fit_config)
    lr: Optional[float] = None      # None: the frozen P-lr value of the provider class
    k_train: int = 16
    init_eval: bool = False         # register the model at step 0 as <name>_init
    notes: str = ""

    @property
    def out_dir(self) -> Path:
        return out_dir() / self.name

    @property
    def done_file(self) -> Path:
        return self.out_dir / "done.json"

    def done(self) -> bool:
        return done_marker(self.done_file) is not None

    def build_kwargs(self) -> dict:
        kw = dict(self.kwargs)
        if CTX.smoke:
            if self.provider == "net":
                kw.setdefault("fit_to_steps", SMOKE_FACTORY_KW["fit_to_steps"])
                kw.setdefault("fit_to_lbfgs", SMOKE_FACTORY_KW["fit_to_lbfgs"])
            if self.factory == "mixture" and kw.get("init", "target") == "target" and self.provider == "poly":
                kw["init_steps"] = list(SMOKE_FACTORY_KW["init_steps"])
                kw["init_search"] = SMOKE_FACTORY_KW["init_search"]
        return kw


def done_marker(path) -> Optional[dict]:
    """A done.json written on THIS machine, else None (absent). A marker of another machine raises."""
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    mid = d.get("machine_id") or (d.get("machine_tag") or {}).get("id")
    if mid != MACHINE_ID:
        raise RuntimeError(f"{paths.portable(path)} was written on machine {mid!r}, this is {MACHINE_ID!r}; "
                           "markers are never shared across machines")
    return d


def frozen_lr(provider: str) -> float:
    f = plr_file()
    if not f.exists():
        raise RuntimeError(f"{paths.portable(f)} missing: the learning-rate pilot has not been decided")
    d = json.loads(f.read_text())
    key = {"poly": "lr_poly", "net": "lr_net"}[provider]
    if d.get(key) is None:
        raise RuntimeError(f"{paths.portable(f)}: {key} not frozen yet")
    return float(d[key])


def fit_config(spec: Spec) -> train.FitConfig:
    lr = spec.lr if spec.lr is not None else frozen_lr(spec.provider)
    steps = steps_of(spec.steps)
    every = max(1, steps // 4) if CTX.smoke else 500
    return train.FitConfig(steps=int(steps), lr=float(lr), lr_end=float(lr) / 15.0, batch=256,
                           k_train=int(spec.k_train), fold="iid", k_val=64, eval_every=int(every), selection="last",
                           ema_decay=None, smooth=0.0, weight_decay=(1e-4 if spec.provider == "net" else 0.0),
                           seed=int(spec.seed), name=spec.name, out_dir=str(spec.out_dir), ckpt_every=int(every),
                           clip=5.0, val_seed=12345, val_batch=128, fold_rng="dedicated",
                           deterministic=False).validate()


# ----------------------------------------------------------------------------------------------------------------------
# data and GPU
# ----------------------------------------------------------------------------------------------------------------------
_DATA = {}


def data_cpu():
    """(sets, extras) of train and val on the CPU, loaded once per process (never the test split)."""
    if "tv" not in _DATA:
        log("loading train / val from the cache (CPU)")
        _DATA["tv"] = sfm_data.load_sets(("train", "val"), device="cpu")
    return _DATA["tv"]


def data_val():
    """(sets, extras) with the validation split only (CPU; the report needs no training events)."""
    if "tv" in _DATA:
        return _DATA["tv"]
    if "v" not in _DATA:
        log("loading val from the cache (CPU)")
        _DATA["v"] = sfm_data.load_sets(("val",), device="cpu")
    return _DATA["v"]


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def on_gpu(label: str, fn, min_free_mb: float = 2500.0, poll: float = 30.0, max_wait: float = 6 * 3600.0):
    """``fn()`` under ``sfm.gpulock.gpu_lock`` - the rule of ``run_m2b_lib.run_on_gpu``: on a capped (shared, large)
    device the job is refused at once when less than cap + 4 GB is free (``machine.check_free_memory``); uncapped (the
    laptop) the lock is released while waiting for memory. Tensors go to the device only inside ``fn``; the cache is
    emptied before the lock is released. The seconds spent in ``fn`` are appended to gpu_time.jsonl (information
    only; no decision of the driver depends on time)."""
    machine.require_cuda(label)
    capped = bool(machine.apply_caps().get("gpu_mem_cap_gb"))
    t_wait = time.time()
    while True:
        with gpulock.gpu_lock(f"m2a {label}", log=log):
            free = None
            if capped:
                info = machine.check_free_memory(label)
                log(f"[on_gpu] {label}: {info['free_gb']:.1f} GB free, cap {info['cap_gb']:.1f} GB")
            else:
                free = torch.cuda.mem_get_info()[0] / 2 ** 20
            if capped or free >= min_free_mb:
                t0 = time.time()
                try:
                    return fn()
                finally:
                    free_cuda()
                    f = gpu_log()
                    f.parent.mkdir(parents=True, exist_ok=True)
                    paths.check_output_path(f)
                    with open(f, "a") as fh:
                        fh.write(json.dumps({"label": label, "seconds": time.time() - t0, "waited": t0 - t_wait,
                                             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
        if time.time() - t_wait > max_wait:
            raise RuntimeError(f"{label}: device memory did not become free within {max_wait:.0f} s")
        log(f"[on_gpu] {label}: only {free:.0f} MB free (need {min_free_mb:.0f}); lock released, retry in {poll:.0f} s")
        time.sleep(poll)


def gpu_seconds_total() -> float:
    f = gpu_log()
    if not f.exists():
        return 0.0
    return sum(json.loads(x)["seconds"] for x in f.read_text().splitlines() if x.strip())


# ----------------------------------------------------------------------------------------------------------------------
# registry helpers (live stream only) and the delta function
# ----------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def live() -> dict:
    """{"stream_id", "points", "machine"} of this process (sfm.registry.live_stream)."""
    st = registry.live_stream()
    env_dir = Path(registry.RegistryEnv().machine_dir)
    if env_dir.resolve() != MACHINE_DIR.resolve():
        raise RuntimeError(f"the registry writes to {paths.portable(env_dir)}, this driver to "
                           f"{paths.portable(MACHINE_DIR)}")
    return {"stream_id": st["stream_id"], "points": st["points"], "machine": st["machine"]}


def live_machine():
    """The tag the registry puts on this machine's arrays (used for the derived tagged arrays of the guards)."""
    return live()["machine"]


def scores_dir(name: str) -> Path:
    """Where the registry stored the run's tagged arrays on the live stream (from its row; never rebuilt here)."""
    row = registered_row(name)
    if row is None:
        raise KeyError(f"{CTX.stage}/{name}: no registry row on the live stream")
    return (paths.OUT_RESULTS / row["val_nll_path"]).parent


def live_stream_id() -> str:
    return live()["stream_id"]


def registered_row(name: str, ckpt=None, need_guards: bool = False) -> Optional[dict]:
    """The latest row of (stage, name) on the LIVE stream [with this checkpoint's SHA-256] [with guards]."""
    return registry.registered_row(CTX.stage, name, stream_id=live_stream_id(), ckpt=ckpt, need_guards=need_guards)


def tagged(name: str, kind: str = "k64") -> provenance.TaggedArray:
    """The registered tagged array of a run on the live stream (tag and SHA-256 checked by registry.row_array)."""
    row = registered_row(name, need_guards=(kind != "k64"))
    if row is None:
        raise KeyError(f"{CTX.stage}/{name}: no registry row{' with guards' if kind != 'k64' else ''} on the live "
                       f"stream {live_stream_id()}")
    return registry.row_array(row, kind=kind)


@lru_cache(maxsize=1)
def refset() -> registry.RefSet:
    return registry.reference_set()


def formula_array() -> provenance.TaggedArray:
    """THIS machine's formula reference array (K = 64, live points): the paired form of every laptop constant."""
    return refset().bench["formula"]


def paired(a: str, b: str) -> stats.Paired:
    """paired_delta(run a, run b) or, with b == "formula", against this machine's formula reference array."""
    tb = formula_array() if b == "formula" else tagged(b)
    return provenance.paired_delta(tagged(a), tb)


def pd(p: stats.Paired) -> dict:
    return {"mean": p.mean, "se": p.se, "n": p.n, "z": p.z}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return paths.portable(o)
    return o


def write_json(obj, path: Path) -> None:
    paths.check_output_path(Path(path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    train.atomic_write_json(json.loads(paths.scrub_str(json.dumps(_jsonable(obj)))), Path(path))


def read_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


# ----------------------------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------------------------
def build_context(dev, train_set=None, seed: int = 0, skip_init: bool = False) -> models.BuildContext:
    """Factories resolve "<stage>/<name>" (start, init_ref, net_from) inside THIS machine's runs/ directory."""
    return models.BuildContext(device=dev, train=train_set, seed=seed, log=log, skip_init=skip_init, results_dir=RUNS)


def verify_sources(kwargs: dict) -> dict:
    """Every run whose weights a new run starts from, is initialised from or is fitted to (factory arguments
    ``start`` - copy by name -, ``init_ref`` - init_shower_long reference, init_from_target target -, ``net_from`` -
    the fit_to target and the rebuilt poly model) must be a finished run of THIS machine whose model.pt has the
    SHA-256 of a registry row on the live stream and whose weight tensors have that row's ``weights_sha256``.
    Returns the record {key: {run, model_sha256, weights_sha256, machine_id, row}}; raises otherwise."""
    out = {}
    for key in SOURCE_KEYS:
        ref = kwargs.get(key)
        if not ref:
            continue
        st, _, name = str(ref).partition("/")
        if st != CTX.stage or not name:
            raise RuntimeError(f"{key}={ref!r}: a source run must be '<{CTX.stage}>/<name>' of this stage")
        rdir = out_dir() / name
        if done_marker(rdir / "done.json") is None:
            raise RuntimeError(f"{key}={ref}: not a finished run of this machine ({paths.portable(rdir)})")
        f = rdir / train.MODEL_FILE
        fsha = provenance.sha256_file(f)
        row = registered_row(name, ckpt=f)
        if row is None:
            raise RuntimeError(f"{key}={ref}: model.pt (SHA-256 {fsha[:12]}...) matches no registry row on the live "
                               f"stream {live_stream_id()}")
        obj = torch.load(f, map_location="cpu", weights_only=False)
        wsha = provenance.weights_sha256(obj["state"])
        if row.get("weights_sha256") != wsha:
            raise RuntimeError(f"{key}={ref}: weight tensors differ from the registry row (weights_sha256)")
        out[key] = {"run": ref, "model_sha256": fsha, "weights_sha256": wsha, "machine_id": MACHINE_ID,
                    "trained_on": row.get("trained_on"),
                    "row": {"stage": row["stage"], "name": row["name"], "timestamp": row.get("timestamp"),
                            "stream_id": live_stream_id(), "val_nll_sha256": row.get("val_nll_sha256"),
                            "checkpoint_sha256": row.get("checkpoint_sha256")}}
    return out


def run_spec(spec: Spec) -> dict:
    """Execute (or skip, or resume) one run; returns its done.json content."""
    d = done_marker(spec.done_file)
    if d is not None:
        return d
    cfg = fit_config(spec)
    sets, extras = data_cpu()

    def job():
        t0 = time.time()
        dev = common.DEVICE
        tr, va = sets["train"].to(dev), sets["val"].to(dev)
        val = registry.ValidationSet(va, extras["val"]["event_number"])
        paths.check_output_path(spec.out_dir)
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        resuming = (spec.out_dir / train.CHECKPOINT).exists() or (spec.out_dir / train.DONE_FILE).exists()
        init_file = spec.out_dir / "init.json"
        train.seed_everything(spec.seed)
        ctx = build_context(dev, tr, spec.seed, skip_init=resuming)
        tb = time.time()
        kw = spec.build_kwargs()
        sources = verify_sources(kw)                       # before any weights are read by the factory
        model = models.build(spec.factory, ctx, **kw)
        build_s = time.time() - tb
        info = getattr(model, "init_info", None)
        if not resuming:
            write_json({"init_info": info, "build_seconds": build_s, "n_params": int(model.n_params()),
                        "spec": asdict(spec), "build_kwargs": kw, "sources_verified": sources,
                        "machine_id": MACHINE_ID, "machine_tag": MACHINE.to_dict()}, init_file)
        init = read_json(init_file, {})
        info = init.get("init_info") or {}
        log(f"[{spec.name}] built {spec.factory} {kw}: {model.n_params()} constants, kl_map {info.get('kl_map')}, "
            f"{'resuming' if resuming else f'init {build_s:.0f}s'}")
        meta_common = {"seed": spec.seed, "k": int(model.n_params()),
                       "start": kw.get("start") or kw.get("net_from") or kw.get("init_ref"),
                       "kl_map": info.get("kl_map"), "init": info, "rung": spec.rung, "cls": spec.cls,
                       "comparator": spec.comparator, "provider": spec.provider, "smoke": CTX.smoke,
                       "sources": sources}
        if spec.init_eval and registered_row(spec.name + "_init") is None:          # A0 parity / A9 Delta at step 0
            if resuming:
                raise RuntimeError(f"{spec.name}: the step-0 row is missing but the run has already started")
            row0 = registry.evaluate(model, spec.name + "_init", CTX.stage, val,
                                     meta={**meta_common, "config": {"spec": asdict(spec), "step": 0},
                                           "trained_on": {"machine_id": MACHINE_ID, "source": "step 0 (untrained)"},
                                           "notes": f"{spec.rung} at step 0 (before training)"})
            log(f"[{spec.name}] step 0 registered: val {row0['val_mean']:.6f}, vs formula "
                f"{row0['vs_formula']['mean'] * 1e4:+.3f}e-4")
        meta = {"model": {"factory": spec.factory, "kwargs": kw}, "spec": asdict(spec), "init": info,
                "stage": CTX.stage, "smoke": CTX.smoke}
        res = train.fit(model, tr, va, cfg, meta=_jsonable(meta), log=log)
        row = registered_row(spec.name, res.model_path)
        if row is None:
            row = registry.evaluate(model, spec.name, CTX.stage, val,
                                    meta={**meta_common, "config": {"spec": asdict(spec), "fit": cfg.result_dict()},
                                          "config_hash": cfg.hash(), "checkpoint": str(res.model_path),
                                          "notes": spec.notes or None, "fit": res.summary(),
                                          "trained_on": {"machine_id": MACHINE_ID, "source": "trained in this job"}})
        log(f"[{spec.name}] registered: val {row['val_mean']:.6f}  vs formula {row['vs_formula']['mean'] * 1e4:+.3f}"
            f" +- {row['vs_formula']['se'] * 1e4:.3f}e-4  skipped {res.skipped_loss}/{res.skipped_grad}")
        done = {"name": spec.name, "stage": CTX.stage, "smoke": CTX.smoke, "spec": asdict(spec), "build_kwargs": kw,
                "fit_config": cfg.result_dict(), "fit": res.summary(), "history": res.history,
                "val_mean": row["val_mean"], "k": row["k"], "kl_map": info.get("kl_map"),
                "checkpoint_sha256": row.get("checkpoint_sha256"), "val_nll_sha256": row.get("val_nll_sha256"),
                "build_seconds": init.get("build_seconds"), "job_seconds": time.time() - t0,
                "sources_verified": sources, "machine_id": MACHINE_ID, "machine_tag": MACHINE.to_dict(),
                "stream_id": live_stream_id(), "batch": CTX.batch,
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_json(done, spec.done_file)
        del model, tr, va, val
        return done

    return on_gpu(spec.name, job)


def rebuild(name: str, device=None):
    """The fitted model of a finished run of the current stage (``sfm.models.load_run``, this machine's runs/)."""
    dev = device or common.DEVICE
    return models.load_run(f"{CTX.stage}/{name}", build_context(dev, None, 0, skip_init=True))
