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
import re
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
    if not smoke and os.environ.get("SFM_M2A_INJECT_DIVERGENCE"):
        raise SystemExit("SFM_M2A_INJECT_DIVERGENCE is a smoke-mode testing hook; refused in the real mode")
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
    """The learning rate of record of a provider class: plr.json (the P-lr decision); for the network class the rate
    decided again by ``warmstart-rerun`` when re-run pilots changed it (review B8-1: every later network run, A11n
    of Stage 2 among them, must use the rate the re-runs used). plr.json itself stays the record of the pilot."""
    f = plr_file()
    if not f.exists():
        raise RuntimeError(f"{paths.portable(f)} missing: the learning-rate pilot has not been decided")
    d = json.loads(f.read_text())
    key = {"poly": "lr_poly", "net": "lr_net"}[provider]
    if provider == "net":
        x = (decisions().get("warmstart_rerun") or {}).get("lr_net_after_rerun") or {}
        if x.get("lr_net_after") is not None and not x.get("escalated"):     # PLAN 20 item 3: an escalated change
            return float(x["lr_net_after"])                                  # is not used before the user decides
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


# ----------------------------------------------------------------------------------------------------------------------
# decisions written by the library (one key per kind; the driver's note_decision writes whole keys of its own)
# ----------------------------------------------------------------------------------------------------------------------
def record_decision(key: str, name: str, value) -> None:
    d = read_json(decisions_file(), {}) or {}
    d.setdefault(key, {})[name] = value
    write_json(d, decisions_file())


def decisions() -> dict:
    return read_json(decisions_file(), {}) or {}


_RUN_RE = re.compile(r"^(?P<rung>.+?)_s\d+(?:_x2)?(?:_ws)?(?:_init)?$")
_STAGE2_RUN_RE = re.compile(r"^A11[a-z]*_s\d+")


def rung_of(name: str) -> str:
    """The rung of a run name: A9_s0 -> A9, A4_s1_x2 -> A4, A9n_s0_ws -> A9n, A2c_x2_s0 -> A2c_x2."""
    m = _RUN_RE.match(str(name))
    return m.group("rung") if m else str(name)


def is_stage2_run(name: str) -> bool:
    """A run of Stage 2 (A11, A11k, A11n, A11p ...): its records stay out of the Stage-1 results.json (review R6)."""
    return bool(_STAGE2_RUN_RE.match(str(name)))


# ----------------------------------------------------------------------------------------------------------------------
# divergence (user decision of 25 Nov 2026, PLAN.md section 17.11)
# ----------------------------------------------------------------------------------------------------------------------
DIVERGED_FILE = "diverged.json"
BLOWUP_NAT = 1.0          # nat per event (= 10,000 in the 1e-4 units of the tables)
FROZEN_FRACTION = 0.5     # skipped batches in the last evaluation window
EXPLORATORY_CLASSES = ("exploratory", "primary-net")     # AUDIT 4.1: the net column is exploratory
INJECT_ENV = "SFM_M2A_INJECT_DIVERGENCE"                 # smoke only: comma-separated run names (testing hook)
DIVERGENCE_DEFINITION = (
    "A run is 'diverged' iff (i) its FINAL validation mean is not finite, or (ii) the registry evaluation of the "
    "finished fit finds a non-finite per-event NLL (FloatingPointError), or (iii) 'frozen': in the last evaluation "
    "window at least 50 % of the training batches were skipped by sfm.train.fit for a non-finite loss or gradient, "
    "or (iv) 'blow-up': the last validation mean exceeds the reference by more than 1 nat per event (10,000 in table "
    "units), the reference being the lowest of: the finite validation means of the fit history, the step-0 "
    "validation mean of a network warm start, and the registered validation mean of the run's comparator. A "
    "non-finite validation mean in the middle of a fit that recovered (final value finite, not frozen, no blow-up) is "
    "NOT a divergence: it is a warning, counted in the run's metadata and listed in the report for Gate 1 (user "
    "decision of 26 Nov 2026, PLAN 20 item 2). Isolated skipped batches of a fit that ends finite are not a divergence "
    "either (they are reported with the run).")
# The definition of 25 Nov 2026 (PLAN 17.11) under which the Stage-1 runs were classified; the Stage-1 report keeps
# printing it (its RESULTS.md stays byte-identical, REVIEW 10.5 R5 / R9). It differs from the present one only in
# criterion (i) - any non-finite validation mean of the history - and the Stage-2 report shows from the stored
# histories that every registered run is classified the same under both.
DIVERGENCE_DEFINITION_25NOV = (
    "A run is 'diverged' iff (i) a validation mean in its fit history is not finite, or (ii) the registry evaluation "
    "of the finished fit finds a non-finite per-event NLL (FloatingPointError), or (iii) 'frozen': in the last "
    "evaluation window at least 50 % of the training batches were skipped by sfm.train.fit for a non-finite loss or "
    "gradient, or (iv) 'blow-up': the last validation mean exceeds the reference by more than 1 nat per event "
    "(10,000 in table units), the reference being the lowest of: the validation means of the fit history, the "
    "step-0 validation mean of a network warm start, and the registered validation mean of the run's comparator. "
    "Isolated skipped batches of a fit that ends finite are NOT a divergence (they are reported with the run).")


def diverged_marker(name: str) -> Optional[dict]:
    """The diverged.json of a run of THIS machine and stage, or None."""
    f = out_dir() / name / DIVERGED_FILE
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if d.get("machine_id") != MACHINE_ID:
        raise RuntimeError(f"{paths.portable(f)} was written on machine {d.get('machine_id')!r}, this is "
                           f"{MACHINE_ID!r}")
    return d


def status_of(res) -> str:
    """'done' (a done.json content), 'diverged' or 'blocked' (the records returned by run_spec)."""
    return (res or {}).get("status", "done")


def source_names(kwargs: dict) -> list:
    out = []
    for key in SOURCE_KEYS:
        ref = kwargs.get(key)
        if ref:
            out.append(str(ref).partition("/")[2])
    return out


def blocked_by(spec: Spec) -> Optional[str]:
    """The run this spec depends on (start / init_ref / net_from) that diverged or is itself blocked, if any."""
    blk = decisions().get("blocked") or {}
    if spec.provider == "net" and spec.lr is None and plr_file().exists():
        plr = read_json(plr_file(), {}) or {}
        if "lr_net" in plr and plr["lr_net"] is None:
            return "P-lr (lr_net undecided: every network pilot diverged or is blocked)"
    for n in source_names(spec.kwargs):
        if diverged_marker(n) is not None:
            return n
        if n in blk:
            return n
    return None


def diagnosis(name: str) -> Optional[dict]:
    """The independent diagnosis of a diverged run (runs/<stage>/diagnosis/<name>.json, written by the orchestrator
    after a Fable diagnosis; never by code). Valid only if it names the SHA-256 of the run's diverged.json and a
    verdict in DIAGNOSIS_VERDICTS. Format:
        {"run": "<name>", "diverged_json_sha256": "<sha256 of runs/<stage>/<name>/diverged.json>",
         "verdict": "failed" | "rerun_allowed", "diagnosed_by": "Fable (independent mathematical diagnosis)",
         "diagnosed_utc": "...", "summary": "..."}"""
    f = out_dir() / "diagnosis" / f"{name}.json"
    dv = out_dir() / name / DIVERGED_FILE
    if not (f.exists() and dv.exists()):
        return None
    d = json.loads(f.read_text())
    ok = (d.get("run") == name and d.get("diverged_json_sha256") == provenance.sha256_file(dv)
          and d.get("verdict") in DIAGNOSIS_VERDICTS and d.get("diagnosed_by") and d.get("diagnosed_utc"))
    return {**d, "valid": bool(ok)}


DIAGNOSIS_VERDICTS = ("failed", "rerun_allowed")


def _inject_callback(spec: Spec, cfg):
    """Smoke only: make the named runs diverge (all parameters of the first tensor set to NaN half-way)."""
    names = [x for x in os.environ.get(INJECT_ENV, "").split(",") if x]
    if spec.name not in names:
        return None
    if not CTX.smoke:
        raise SystemExit(f"{INJECT_ENV} is a smoke-mode testing hook; refused in the real mode")
    at = max(2, int(cfg.steps) // 2)

    def cb(step, model):
        if step == at:
            with torch.no_grad():
                next(iter(model.parameters())).fill_(float("nan"))
            log(f"[{spec.name}] SMOKE INJECTION: parameters set to NaN at step {step}")
    return cb


def _grad_norm(model, tr) -> dict:
    """Gradient norm of the mean training NLL at the final parameters, on the first 256 training events with K = 16
    and a fixed fold generator (a diagnostic; sfm.train.fit does not record gradient norms)."""
    from sfm import metric
    try:
        model.train()
        b = tr.batch(torch.arange(min(256, len(tr)), device=tr.off.device))
        gen = torch.Generator(device=tr.off.device).manual_seed(0)
        model.zero_grad(set_to_none=True)
        nll, _ = metric.event_nll(model, b, tr.box, 16, gen)
        loss = nll.mean()
        loss.backward()
        g = [p.grad for p in model.parameters() if p.grad is not None]
        norm = float(torch.sqrt(sum((x.double() ** 2).sum() for x in g))) if g else None
        return {"loss": float(loss.detach()), "grad_norm": norm,
                "finite": bool(norm is not None and math.isfinite(norm) and math.isfinite(float(loss.detach())))}
    except Exception as err:                                                   # diagnostic only
        return {"error": f"{type(err).__name__}: {err}"}
    finally:
        model.zero_grad(set_to_none=True)


def divergence_check(history: list, ref_vals: list) -> list:
    """The criteria (i), (iii), (iv) of DIVERGENCE_DEFINITION that a fit history meets (criterion (ii) is the
    registry's FloatingPointError)."""
    crit = []
    vals = [h.get("val") for h in history]
    if vals and (vals[-1] is None or not math.isfinite(float(vals[-1]))):        # PLAN 20 item 2: the final value
        crit.append("non-finite final validation mean")
    if len(history) >= 1:
        last = history[-1]
        prev = history[-2] if len(history) >= 2 else {"step": 0, "skipped_loss": 0, "skipped_grad": 0}
        n = max(1, int(last["step"]) - int(prev["step"]))
        sk = (last.get("skipped_loss", 0) + last.get("skipped_grad", 0)
              - prev.get("skipped_loss", 0) - prev.get("skipped_grad", 0))
        if sk >= FROZEN_FRACTION * n:
            crit.append(f"frozen: {sk} of {n} batches skipped in the last evaluation window")
        lv = last.get("val")
        refs = [float(v) for v in list(vals) + list(ref_vals) if v is not None and math.isfinite(float(v))]
        if lv is not None and math.isfinite(float(lv)) and refs and float(lv) - min(refs) > BLOWUP_NAT:
            crit.append(f"blow-up: last validation {float(lv):.4f} exceeds the reference {min(refs):.4f} by more "
                        f"than {BLOWUP_NAT} nat")
    return crit


def recovered_nonfinite(history: list) -> dict:
    """Non-finite validation means before the last evaluation of a fit (PLAN 20 item 2: a warning, not a divergence,
    when the fit recovered). {"count": n, "steps": [...]}."""
    steps = [h["step"] for h in history[:-1] if h.get("val") is None or not math.isfinite(float(h["val"]))]
    return {"count": len(steps), "steps": steps}


def _diverged_record(spec, cfg, res, model, tr, crit, eval_error=None, step0=None) -> dict:
    hist = res.history if res is not None else []
    finite = [h for h in hist if h.get("val") is not None and math.isfinite(float(h["val"]))]
    first_bad = next((h["step"] for h in hist if h.get("val") is None or not math.isfinite(float(h["val"]))), None)
    state = model.state_dict()
    tens = [v for v in state.values() if torch.is_tensor(v) and v.is_floating_point()]
    n_bad = int(sum(int((~torch.isfinite(t)).sum()) for t in tens))
    mx = max((float(t[torch.isfinite(t)].abs().max()) for t in tens if bool(torch.isfinite(t).any())), default=None)
    rec = {"status": "diverged", "run": spec.name, "rung": spec.rung, "cls": spec.cls, "seed": spec.seed,
           "comparator": spec.comparator, "criteria": crit, "definition": DIVERGENCE_DEFINITION,
           "step": first_bad if first_bad is not None else (hist[-1]["step"] if hist else None),
           "last_finite": ({"step": finite[-1]["step"], "val": finite[-1]["val"],
                            "train_nll": finite[-1].get("train_nll")} if finite else None),
           "history": hist, "fit": res.summary() if res is not None else None,
           "params": {"n_nonfinite": n_bad, "max_abs_finite": mx,
                      "weights_sha256": provenance.weights_sha256(state),
                      "note": "sfm.train.fit skips non-finite updates, so these are the last parameters it kept"},
           "gradient": _grad_norm(model, tr), "eval_error": eval_error, "step0_parity": step0,
           "files_kept": [f for f in (train.MODEL_FILE, train.CHECKPOINT, train.DONE_FILE, train.LOG_FILE)
                          if (spec.out_dir / f).exists()],
           "machine_id": MACHINE_ID, "stream_id": live_stream_id(), "batch": CTX.batch,
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(rec, spec.out_dir / DIVERGED_FILE)
    record_decision("diverged", spec.name, {k: rec[k] for k in ("rung", "cls", "seed", "criteria", "step",
                                                                "last_finite", "params", "gradient", "eval_error")}
                    | {"status": ("diverged" if spec.cls in EXPLORATORY_CLASSES or spec.cls == "pilot"
                                  else "diverged - awaiting diagnosis")})
    log(f"[{spec.name}] DIVERGED ({'; '.join(crit)}); recorded in {DIVERGED_FILE} and decisions.json; the job "
        "continues with the runs that do not depend on it")
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# timing of guard parts (STATUS R5; the guards phase records it for every run, timing fields only)
# ----------------------------------------------------------------------------------------------------------------------
SLOW_PART_S = 60.0


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def host_state() -> dict:
    """Load average, available memory, swap in use, resident set size and the pressure-stall totals (Linux)."""
    mi = {ln.split(":")[0]: int(ln.split()[1]) for ln in _read_text("/proc/meminfo").splitlines() if ln.split()[1:]}
    st = {ln.split(":")[0]: ln.split()[1] for ln in _read_text("/proc/self/status").splitlines() if ln.split()[1:]}
    psi = {}
    for res in ("cpu", "memory", "io"):
        for line in _read_text(f"/proc/pressure/{res}").splitlines():
            f = dict(x.split("=") for x in line.split()[1:])
            psi[f"{res}_{line.split()[0]}_avg10"] = float(f.get("avg10", "nan"))
            psi[f"{res}_{line.split()[0]}_total_us"] = int(f.get("total", 0))
    return {"load1": float((_read_text("/proc/loadavg").split() or ["nan"])[0]),
            "mem_available_mb": mi.get("MemAvailable", 0) // 1024,
            "swap_used_mb": (mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) // 1024,
            "rss_mb": int(st.get("VmRSS", 0)) // 1024, "psi": psi}


class PartTimer:
    """with PartTimer(label, rec): ...  ->  rec[label] = wall / CPU seconds, page faults, host state and the stall
    seconds of the part; a part longer than SLOW_PART_S is logged as a loud WARNING."""

    def __init__(self, label: str, rec: dict, run: str = ""):
        self.label, self.rec, self.run = label, rec, run

    def __enter__(self):
        import resource
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self.t0, self.c0, self.f0, self.m0 = time.perf_counter(), time.process_time(), ru.ru_majflt, ru.ru_minflt
        self.h0 = host_state()
        return self

    def __exit__(self, *exc):
        import resource
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        h1 = host_state()
        stall = {k.replace("_total_us", "_stall_s"): round((v - self.h0["psi"].get(k, 0)) / 1e6, 3)
                 for k, v in h1["psi"].items() if k.endswith("_total_us")}
        r = {"wall_s": round(time.perf_counter() - self.t0, 3), "cpu_s": round(time.process_time() - self.c0, 3),
             "majflt": ru.ru_majflt - self.f0, "minflt": ru.ru_minflt - self.m0,
             **{k: v for k, v in h1.items() if k != "psi"},
             **{k: v for k, v in h1["psi"].items() if k.endswith("avg10")}, **stall}
        self.rec[self.label] = r
        if r["wall_s"] > SLOW_PART_S:
            log(f"WARNING: guard part '{self.label}' of {self.run} took {r['wall_s']:.0f} s (> {SLOW_PART_S:.0f} s): "
                f"CPU {r['cpu_s']:.0f} s, load {r['load1']}, MemAvailable {r['mem_available_mb']} MB, swap "
                f"{r['swap_used_mb']} MB, major faults {r['majflt']}, stall s cpu/memory/io "
                f"{stall.get('cpu_some_stall_s')}/{stall.get('memory_some_stall_s')}/{stall.get('io_some_stall_s')}")
        return False


# ----------------------------------------------------------------------------------------------------------------------
# step-0 parity of a network warm start (user decision of 25 Nov 2026)
# ----------------------------------------------------------------------------------------------------------------------
def warm_start_source(kwargs: dict) -> Optional[str]:
    """The registered polynomial run a network warm start reproduces: net_from, or A0_init (the legacy coefficients
    at step 0) for a net model fitted to its own legacy provider (A1)."""
    if kwargs.get("net_from"):
        return str(kwargs["net_from"]).partition("/")[2]
    if kwargs.get("provider") == "net" and kwargs.get("variant") == "legacy_centre":
        return "A0_init"
    return None


def step0_parity(model, name: str, source: Optional[str], val, subdir: str = "step0", base: Optional[Path] = None) -> dict:
    """Paired difference on validation between a network at step 0 and its source polynomial run's registered array:
    the per-event NLL of the step-0 model with the unmodified common.test_nll inside metric.deterministic() (the
    registry's computation, same points), saved as a tagged array (not a registry row) and paired through
    provenance.paired_delta with the source's registered array."""
    from sfm import metric
    if source is None or registered_row(source) is None:
        return {"source": source, "available": False, "reason": "no registered source array"}
    was = model.training
    model.eval()
    with metric.deterministic():
        nll = common.test_nll(model, val.showers)
    model.train(was)
    f = (Path(base) if base is not None else out_dir() / name) / subdir / "val_nll.npy"
    if not np.isfinite(nll).all():
        return {"source": source, "available": False, "reason": f"{int((~np.isfinite(nll)).sum())} non-finite values"}
    extra = {}
    if f.exists():
        # review C8-3: a restart before the first checkpoint rebuilds the warm start. The array of the first build
        # is verified by its tag (sidecar SHA-256) and reused if the rebuild equals it bit for bit; otherwise the
        # training now starts from the REBUILT model, so its parity is what counts: it is written as a new tagged
        # array next to the first one (nothing is overwritten) and the difference is recorded.
        old = provenance.load_tagged(f, expect_machine_id=live_machine().id)
        if not (old.values.shape == nll.shape and np.array_equal(old.values, nll)):
            k = 1
            while f.with_name(f"val_nll_rebuild{k}.npy").exists():
                k += 1
            first = {"array": paths.portable(f), "sha256": old.sha256,
                     "max_abs_vs_rebuild": float(np.abs(old.values.astype(np.float64) - nll.astype(np.float64)).max())}
            f = f.with_name(f"val_nll_rebuild{k}.npy")
            extra = {"rebuilt_after_restart": True, "first_build": first}
    ta = provenance.save_tagged(f, nll, live()["points"], live_machine(), label=f"{CTX.stage}/{name}:{subdir}")
    src = tagged(source)
    p = provenance.paired_delta(ta, src)
    d = provenance.paired_difference(ta, src)
    return {"source": source, "available": True, "paired": pd(p), "max_abs_event": float(np.abs(d).max()),
            "step0_val_mean": float(np.mean(nll, dtype=np.float64)), "array": paths.portable(f), "sha256": ta.sha256,
            **extra}


# ----------------------------------------------------------------------------------------------------------------------
# supersession (warm-start re-runs, PLAN 17.11)
# ----------------------------------------------------------------------------------------------------------------------
def superseded_recorded() -> dict:
    """The map {original: re-run} as warmstart-rerun recorded it (every re-run it started)."""
    return dict((decisions().get("warmstart_rerun") or {}).get("superseded") or {})


def superseded() -> dict:
    """The supersessions IN FORCE: only a re-run with a registry row supersedes its original. A re-run that diverged
    (or is blocked, or did not finish) does not: the original stays in force and the report says so (review C8-4)."""
    not_used = set(((decisions().get("warmstart_rerun") or {}).get("lr_net_after_rerun") or {}).get("not_used") or [])
    return {o: n for o, n in superseded_recorded().items() if registered_row(n) is not None and o not in not_used}


def rerun_not_in_force() -> dict:
    """{original: (re-run, status)} for recorded re-runs without a registry row (diverged, blocked, missing)."""
    out = {}
    for o, n in superseded_recorded().items():
        if registered_row(n) is None:
            dv = diverged_marker(n)
            out[o] = (n, "diverged" if dv is not None else ("blocked" if n in (decisions().get("blocked") or {})
                                                            else "missing"))
    return out


def effective(name: Optional[str]) -> Optional[str]:
    """The run that stands for ``name`` in the rules: its warm-start re-run if that supersession is in force (the
    re-run is registered), else itself."""
    if name is None:
        return None
    return superseded().get(name, name)


# ----------------------------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------------------------
def run_spec(spec: Spec) -> dict:
    """Execute (or skip, or resume) one run. Returns its done.json content, or - user decision of 25 Nov 2026 - the
    record {"status": "diverged", ...} (diverged.json; DIVERGENCE_DEFINITION) or {"status": "blocked",
    "blocked_by": <run>} when a run it starts from / is initialised from / is fitted to diverged. A diverged or
    blocked run is terminal: re-issuing the command does not retrain it."""
    d = done_marker(spec.done_file)
    if d is not None:
        return d
    dv = diverged_marker(spec.name)
    if dv is not None:
        return dv
    b = blocked_by(spec)
    if b is not None:
        rec = {"status": "blocked", "run": spec.name, "blocked_by": b, "rung": spec.rung, "cls": spec.cls,
               "seed": spec.seed}
        record_decision("blocked", spec.name, {"blocked_by": b, "rung": spec.rung, "cls": spec.cls,
                                               "seed": spec.seed})
        log(f"[{spec.name}] BLOCKED by {b} (diverged or blocked); not trained, recorded in decisions.json")
        return rec
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
            parity0 = None
            if spec.provider == "net":                     # step-0 parity of the warm start (PLAN 17.11)
                registry.check_validation(val)
                parity0 = step0_parity(model, spec.name, warm_start_source(kw), val)
                record_decision("step0_parity", spec.name, parity0)
                if parity0.get("available"):
                    log(f"[{spec.name}] step-0 parity vs {parity0['source']}: "
                        f"{parity0['paired']['mean'] * 1e4:+.3f} +- {parity0['paired']['se'] * 1e4:.3f}e-4")
            write_json({"init_info": info, "build_seconds": build_s, "n_params": int(model.n_params()),
                        "spec": asdict(spec), "build_kwargs": kw, "sources_verified": sources,
                        "step0_parity": parity0, "machine_id": MACHINE_ID, "machine_tag": MACHINE.to_dict()},
                       init_file)
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
        res = train.fit(model, tr, va, cfg, meta=_jsonable(meta), log=log, callback=_inject_callback(spec, cfg))
        # divergence (PLAN 17.11; DIVERGENCE_DEFINITION): checked before the registry is touched
        refs = []
        p0 = init.get("step0_parity") or {}
        if p0.get("available"):
            refs.append(p0["step0_val_mean"])
        crow = registered_row(spec.comparator) if spec.comparator and spec.comparator != "formula" else None
        if crow is not None:
            refs.append(crow["val_mean"])
        crit = divergence_check(res.history, refs)
        if crit:
            rec = _diverged_record(spec, cfg, res, model, tr, crit, step0=p0 or None)
            del model, tr, va, val
            return rec
        row = registered_row(spec.name, res.model_path)
        if row is None:
            try:
                row = registry.evaluate(model, spec.name, CTX.stage, val,
                                        meta={**meta_common, "config": {"spec": asdict(spec),
                                                                        "fit": cfg.result_dict()},
                                              "config_hash": cfg.hash(), "checkpoint": str(res.model_path),
                                              "notes": spec.notes or None, "fit": res.summary(),
                                              "trained_on": {"machine_id": MACHINE_ID,
                                                             "source": "trained in this job"}})
            except FloatingPointError as err:              # criterion (ii): non-finite per-event validation NLL
                rec = _diverged_record(spec, cfg, res, model, tr,
                                       ["non-finite per-event validation NLL at registration"],
                                       eval_error=str(err), step0=p0 or None)
                del model, tr, va, val
                return rec
        log(f"[{spec.name}] registered: val {row['val_mean']:.6f}  vs formula {row['vs_formula']['mean'] * 1e4:+.3f}"
            f" +- {row['vs_formula']['se'] * 1e4:.3f}e-4  skipped {res.skipped_loss}/{res.skipped_grad}")
        warn = recovered_nonfinite(res.history)
        if warn["count"]:                                  # PLAN 20 item 2: a warning for Gate 1, not a divergence
            record_decision("warnings", spec.name, {"recovered_nonfinite_validation": warn, "cls": spec.cls})
            log(f"[{spec.name}] WARNING: {warn['count']} non-finite validation mean(s) before the final value (steps "
                f"{warn['steps']}); the fit recovered - listed for Gate 1")
        done = {"name": spec.name, "stage": CTX.stage, "smoke": CTX.smoke, "spec": asdict(spec), "build_kwargs": kw,
                "fit_config": cfg.result_dict(), "fit": res.summary(), "history": res.history,
                "val_mean": row["val_mean"], "k": row["k"], "kl_map": info.get("kl_map"),
                "checkpoint_sha256": row.get("checkpoint_sha256"), "val_nll_sha256": row.get("val_nll_sha256"),
                "build_seconds": init.get("build_seconds"), "job_seconds": time.time() - t0,
                "sources_verified": sources, "step0_parity": init.get("step0_parity"),
                "warnings": ({"recovered_nonfinite_validation": warn} if warn["count"] else None),
                "machine_id": MACHINE_ID, "machine_tag": MACHINE.to_dict(),
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
