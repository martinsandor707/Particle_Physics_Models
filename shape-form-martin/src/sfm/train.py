"""sfm.train - one resumable fit routine for any model ``lnp (B,) -> log P (B, N_Z, N_R)``.

    cfg = FitConfig(steps=3000, lr=3e-3, lr_end=3e-4, name="b0", out_dir="results/m2b_mlp/b0")
    res = fit(model, train, val, cfg)          # resumes from out_dir/checkpoint.pt, or returns the finished result

Objective and optimiser
    loss   = mean over the batch of the per-event NLL of ``sfm.metric.event_nll`` (= ``common.event_nll`` at the
             defaults: K = k_train i.i.d. fold points per hit; ``fold="stratified"`` = jittered strata)
             + smooth * mean((logP[:, 2:] - 2 logP[:, 1:-1] + logP[:, :-2])^2)   (``common.fit``'s depth curvature)
    Adam (L2 weight decay ``weight_decay`` on parameters with ndim >= 2 only - weight matrices; biases, scalars and
    tables are not decayed), cosine decay of the learning rate from ``lr`` to ``lr_end`` over ``steps``
    (``CosineAnnealingLR``, stepped once per batch), gradient-norm clipping at ``clip`` (5.0).
    A batch whose loss or gradient is not finite is skipped (no optimiser step, no EMA update) and COUNTED in
    ``skipped_loss`` / ``skipped_grad``; the first skips are logged one by one, the totals appear in every
    validation line, in the history, in the result and in the done-marker. As in ``fit_fast``, a skipped batch
    still counts as a step (the schedule advances).
Batches
    events of the training split in a fresh random permutation per epoch (``torch.randperm`` on a dedicated CPU
    generator seeded with 1000 + seed, the convention of ``shapelab.fit_fast``), ``batch`` events per step; the
    last, partial batch of an epoch is used as well (as ``Showers.batches`` does).
Fold points
    drawn from a dedicated generator on the data's device, seeded with 20000 + seed (``fold_rng="dedicated"``), so
    the run does not depend on, or disturb, other code's use of the global RNG. ``fold_rng="global"`` draws from the
    global generator instead, as ``fit_fast`` / ``common.fit`` do.
Seeding
    on a fresh start ``random``, ``numpy`` and ``torch`` (CPU and all CUDA devices) are seeded with ``seed``; their
    states are part of every checkpoint. Seed the RNG before *building* the model as well (``seed_everything``) -
    the model's initialisation happens outside ``fit``.
Validation
    every ``eval_every`` steps and at the last step: ``model.eval()`` under ``torch.inference_mode()``, per-event NLL
    with K = k_val, generator seed ``val_seed`` (12345), batches of ``val_batch`` (128) in storage order - the loop
    of ``common.test_nll``, inside ``sfm.metric.deterministic()`` (so the history is reproducible on CUDA). With the
    data on ``common.DEVICE`` and k_val = 64 these are the numbers of ``common.test_nll``. The EMA weights, if any,
    are evaluated the same way and logged separately.
Selection
    "last" (default; the plan's rule for new fits) keeps the final iterate; "best_val" keeps the state with the
    lowest validation mean (strictly lower than all earlier ones, as ``fit_fast``). The EMA stream is selected by
    the same rule on its own validation numbers and saved separately (``model.pt["ema_state"]``).
Resumability
    ``out_dir/checkpoint.pt`` every ``ckpt_every`` steps and at the end: model, optimiser, scheduler, EMA, the shuffle
    and fold generators, the global RNG states, the current permutation and position in the epoch, step, history,
    skip counters, the best states, the config. Written atomically (temp file + fsync + rename). At the end
    ``model.pt`` (selected weights, EMA weights, history, config, meta) and the done-marker ``fit_done.json`` are
    written the same way. Calling ``fit`` again returns the finished result (weights loaded) when the done-marker
    exists, resumes from the checkpoint otherwise; a checkpoint written with a different config (``config_hash``,
    all fields except ``out_dir`` and ``ckpt_every``) is refused.
Machine tag (``sfm.machine``)
    ``fit`` starts with the E10 checks (hard failure without CUDA - unless ``SFM_ALLOW_CPU=1`` is set explicitly, for
    CPU unit tests - and TF32 off) and stores the live machine tag (with ``deterministic`` = ``cfg.deterministic``)
    in ``checkpoint.pt``, ``model.pt`` and ``fit_done.json``. A checkpoint or a done-marker with another machine tag,
    or without one (written before tags existed, e.g. on the laptop), is refused (``MachineMismatch``): no fit is
    resumed on another machine and a run finished elsewhere is never returned as this machine's training.
    Resume equivalence: an interrupted and resumed run ends with the same weights as an uninterrupted one -
    bit for bit on CPU, and on CUDA inside ``deterministic=True`` (otherwise up to the accumulation order of
    ``index_add_``); ``tests/test_train.py`` checks both.

Relation to ``shapelab.fit_fast`` (shape-form-proton/shapelab.py:215-250)
    With ``k_train=16, k_val=16, selection="best_val", fold_rng="global"``, the same lr, lr_end, steps, eval_every and
    seed, batch 256 and the same global RNG state, ``fit`` performs the same sequence of operations as ``fit_fast``
    (same permutation generator and seed, same loss, skip rule, clipping, Adam, cosine schedule and selection) and
    reaches the same weights (``tests/test_train.py::test_matches_fit_fast``). Differences by default:
    (1) fold points from a dedicated generator instead of the global one; (2) validation at K = 64 instead of 16,
    inside ``deterministic()``, with ``inference_mode`` instead of ``no_grad`` (identical numbers); (3) "last"
    instead of "best_val"; (4) extras: weight decay, EMA, smoothness penalty, stratified folds, checkpoints and
    resume, skip counters, history with training loss; (5) the model is left in ``eval()`` mode at the end
    (``fit_fast`` leaves it in train mode).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from . import machine as sfm_machine
from . import metric, paths

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common as C  # noqa: E402

__all__ = ["FitConfig", "FitResult", "fit", "seed_everything", "val_nll", "config_hash", "atomic_save",
           "atomic_write_json", "ConfigMismatch", "SHUFFLE_SEED_BASE", "FOLD_SEED_BASE", "RunConfig", "EvalSpec",
           "load_run_config", "RUN_DONE_FILE", "MachineMismatch"]

SHUFFLE_SEED_BASE = 1000          # shapelab.fit_fast: torch.Generator().manual_seed(1000 + seed)
FOLD_SEED_BASE = 20000
CHECKPOINT, MODEL_FILE, DONE_FILE, LOG_FILE = "checkpoint.pt", "model.pt", "fit_done.json", "fit.log"
RUN_DONE_FILE = "done.json"       # written by scripts/fit.py after fit + registry evaluation; scripts/queue.py skips on it
_NON_RESULT_FIELDS = ("out_dir", "ckpt_every")


class ConfigMismatch(RuntimeError):
    """A checkpoint or finished run in out_dir was produced with a different configuration."""


class MachineMismatch(RuntimeError):
    """A checkpoint or finished run in out_dir was produced on another machine (or carries no machine tag): no fit is
    continued, and no finished fit is returned, across machines (PLAN.md section 13, A6 revised, point 5)."""


# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class FitConfig:
    steps: int = 3000
    lr: float = 0.03
    lr_end: float = 0.002
    batch: int = 256
    k_train: int = 16
    fold: str = "iid"                     # "iid" | "stratified" (training points only; validation is always i.i.d.)
    k_val: int = 64
    eval_every: int = 250
    selection: str = "last"               # "last" | "best_val"
    ema_decay: Optional[float] = None     # None or a decay in (0, 1), e.g. 0.999
    smooth: float = 0.0
    weight_decay: float = 0.0
    seed: int = 0
    name: str = "run"
    out_dir: Optional[str] = None         # default: <results>/runs/<name>
    ckpt_every: int = 500
    clip: float = 5.0
    val_seed: int = 12345
    val_batch: int = 128
    fold_rng: str = "dedicated"           # "dedicated" | "global"
    deterministic: bool = False           # train inside sfm.metric.deterministic() (bitwise-reproducible on CUDA)

    def validate(self) -> "FitConfig":
        ints = ("steps", "batch", "k_train", "k_val", "eval_every", "seed", "ckpt_every", "val_seed", "val_batch")
        for k in ints:
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError(f"FitConfig.{k} must be an int, got {v!r}")
        for k in ("steps", "batch", "k_train", "k_val", "eval_every", "ckpt_every", "val_batch"):
            if getattr(self, k) < 1:
                raise ValueError(f"FitConfig.{k} must be >= 1")
        for k in ("lr", "lr_end", "smooth", "weight_decay", "clip"):
            v = getattr(self, k)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or float(v) < 0:
                raise ValueError(f"FitConfig.{k} must be a finite number >= 0, got {v!r}")
        if not self.lr > 0 or self.lr_end > self.lr:
            raise ValueError("FitConfig: need lr > 0 and lr_end <= lr")
        if self.fold not in metric.FOLDS:
            raise ValueError(f"FitConfig.fold must be one of {metric.FOLDS}")
        if self.selection not in ("last", "best_val"):
            raise ValueError("FitConfig.selection must be 'last' or 'best_val'")
        if self.ema_decay is not None and not (0.0 < float(self.ema_decay) < 1.0):
            raise ValueError("FitConfig.ema_decay must be None or in (0, 1)")
        if self.fold_rng not in ("dedicated", "global"):
            raise ValueError("FitConfig.fold_rng must be 'dedicated' or 'global'")
        if not isinstance(self.name, str) or not self.name or "/" in self.name or self.name.startswith("."):
            raise ValueError("FitConfig.name must be a non-empty name without '/'")
        return self

    def result_dict(self) -> dict:
        """The fields that determine the result (all but out_dir and ckpt_every)."""
        return {k: v for k, v in asdict(self).items() if k not in _NON_RESULT_FIELDS}

    def hash(self) -> str:
        return config_hash(self.result_dict())

    def resolved_out_dir(self) -> Path:
        return Path(self.out_dir) if self.out_dir else run_output_dir("adhoc", self.name)

    @classmethod
    def from_dict(cls, d: dict) -> "FitConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown FitConfig field(s): {sorted(unknown)}")
        cfg = cls(**d)
        for k in ("lr", "lr_end", "smooth", "weight_decay", "clip"):         # TOML ints for float fields
            v = getattr(cfg, k)
            if isinstance(v, int) and not isinstance(v, bool):
                setattr(cfg, k, float(v))
        if isinstance(cfg.ema_decay, int) and not isinstance(cfg.ema_decay, bool):
            cfg.ema_decay = float(cfg.ema_decay)
        return cfg.validate()


def config_hash(d: dict) -> str:
    """sha256 (first 16 hex digits) of the canonical JSON of a config dict."""
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class FitResult:
    name: str
    out_dir: Path
    model_path: Path
    config: dict
    config_hash: str
    history: list
    step: int
    skipped_loss: int
    skipped_grad: int
    selection: str
    selected_step: int
    selected_val: Optional[float]
    final_val: Optional[float]
    final_val_ema: Optional[float]
    best_val: Optional[float]
    best_step: Optional[int]
    has_ema: bool
    resumed_from: Optional[int]
    seconds: float
    meta: dict = field(default_factory=dict)

    @property
    def skipped(self) -> int:
        return self.skipped_loss + self.skipped_grad

    def summary(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "config_hash", "step", "skipped_loss", "skipped_grad", "selection",
                                           "selected_step", "selected_val", "final_val", "final_val_ema", "best_val",
                                           "best_step", "has_ema", "resumed_from", "seconds")}
        d["out_dir"], d["model_path"] = str(self.out_dir), str(self.model_path)
        return d


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU and every CUDA device)."""
    random.seed(int(seed))
    np.random.seed(int(seed) % 2 ** 32)
    torch.manual_seed(int(seed))                       # also seeds all CUDA devices


def _rng_states() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_states(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_save(obj, path: Path) -> None:
    """``torch.save`` to a temporary file in the same directory, fsync, then rename over ``path``."""
    path = paths.check_output_path(Path(path))
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(obj, path: Path) -> None:
    path = paths.check_output_path(Path(path))
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(paths.scrub_str(json.dumps(obj, indent=1, default=_json_default)) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _clone_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def val_nll(model: torch.nn.Module, data, K: int = C.N_FOLD_EVAL, seed: int = 12345, batch: int = 128,
            deterministic_sum: bool = True) -> np.ndarray:
    """Per-event NLL with a fixed seed: the loop of ``common.test_nll`` (generator on the data's device, batches in
    storage order), in ``eval()`` mode under ``inference_mode``. Equal to ``common.test_nll(model, data, K)`` for data
    on ``common.DEVICE`` (bit for bit inside ``deterministic()``); also works for data on another device."""
    was = model.training
    model.eval()
    try:
        with torch.inference_mode(), (metric.deterministic() if deterministic_sum else contextlib.nullcontext()):
            gen = torch.Generator(device=data.xyz.device).manual_seed(int(seed))
            out = [metric.event_nll(model, b, data.box, int(K), gen)[0] for b in data.batches(int(batch))]
        return torch.cat(out).cpu().numpy()
    finally:
        model.train(was)


val_nll.__test__ = False


class _EMA:
    """Exponential moving average of all parameters (initialised at the starting weights); buffers are taken from
    the live model when the EMA weights are materialised."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        w = 1.0 - self.decay
        for k, p in model.named_parameters():
            self.shadow[k].lerp_(p.detach(), w)

    def state(self, model: torch.nn.Module) -> dict:
        st = _clone_state(model)
        for k, v in self.shadow.items():
            st[k] = v.detach().clone()
        return st


class _Log:
    def __init__(self, out_dir: Path, log: Optional[Callable]):
        self.path, self.log = paths.check_output_path(out_dir / LOG_FILE), log

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        if self.log:
            self.log(line)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def _param_groups(params, weight_decay: float):
    if not weight_decay:
        return [{"params": params, "weight_decay": 0.0}]
    decay = [p for p in params if p.ndim >= 2]
    rest = [p for p in params if p.ndim < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if rest:
        groups.append({"params": rest, "weight_decay": 0.0})
    return groups


def _weights_sha256(state: dict) -> str:
    from .provenance import weights_sha256
    return weights_sha256(state)


def _check_machine(stored: Optional[dict], live, what: str) -> None:
    """Refuse a checkpoint / done-marker written on another machine, or without a machine tag."""
    if not stored:
        raise MachineMismatch(f"{what} carries no machine tag (written before machine tags existed, e.g. on the "
                              f"laptop); it is not resumed or returned on this machine ({live.id}). Use another run "
                              "name; a laptop-trained network is re-scored with scripts/rescore_registered.py")
    if stored.get("fields") != live.fields:
        sf, lf = stored.get("fields") or {}, live.fields
        diff = sorted(k for k in set(sf) | set(lf) if sf.get(k) != lf.get(k))
        raise MachineMismatch(f"{what} was written on machine {stored.get('id')}, this is {live.id} (differs in "
                              f"{diff}); no fit is continued or returned across machines")


def _result_from_model_file(obj: dict, out: Path, model_path: Path, seconds: float = 0.0) -> FitResult:
    s = obj["summary"]
    return FitResult(name=s["name"], out_dir=out, model_path=model_path, config=obj["config"],
                     config_hash=obj["config_hash"], history=obj["history"], step=s["step"],
                     skipped_loss=s["skipped_loss"], skipped_grad=s["skipped_grad"], selection=s["selection"],
                     selected_step=s["selected_step"], selected_val=s["selected_val"], final_val=s["final_val"],
                     final_val_ema=s["final_val_ema"], best_val=s["best_val"], best_step=s["best_step"],
                     has_ema=obj.get("ema_state") is not None, resumed_from=s.get("resumed_from"),
                     seconds=seconds or s.get("seconds", 0.0), meta=obj.get("meta") or {})


# ----------------------------------------------------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------------------------------------------------
def fit(model: torch.nn.Module, train, val, config, *, meta: Optional[dict] = None, log: Optional[Callable] = print,
        callback: Optional[Callable] = None) -> FitResult:
    """Train ``model`` on ``train`` (a ``common.Showers``), monitor on ``val``; see the module docstring.

    meta      JSON-serialisable information stored in ``model.pt`` (e.g. the model factory spec for ``load_run``)
    log       callable for progress lines (default ``print``); every line is also appended to ``out_dir/fit.log``
    callback  ``callback(step, model)`` after every step (testing hook; an exception there interrupts the run
              exactly like a crash, leaving the last checkpoint behind)
    Returns a ``FitResult``; ``model`` holds the selected weights and is in ``eval()`` mode.
    """
    cfg = config if isinstance(config, FitConfig) else FitConfig.from_dict(dict(config))
    cfg.validate()
    sfm_machine.require_cuda("sfm.train.fit", allow_cpu_env=True)       # E10: no silent CPU fallback
    sfm_machine.disable_tf32()
    mtag = sfm_machine.machine_tag(deterministic=cfg.deterministic)
    out = paths.check_output_path(cfg.resolved_out_dir())
    out.mkdir(parents=True, exist_ok=True)
    ck_path, model_path, done_path = out / CHECKPOINT, out / MODEL_FILE, out / DONE_FILE
    L = _Log(out, log)
    chash = cfg.hash()
    device = train.xyz.device
    model.to(device)

    # ---- finished run: load and return -------------------------------------------------------------------------------
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done.get("config_hash") != chash:
            raise ConfigMismatch(f"{out}: finished run has config {done.get('config_hash')}, this call {chash}; "
                                 "use another name/out_dir")
        _check_machine(done.get("machine_tag"), mtag, f"{done_path} (finished run)")
        if sha256_file(model_path) != done.get("model_sha256"):
            raise RuntimeError(f"{model_path}: SHA-256 differs from the done-marker")
        obj = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(obj["state"])
        model.eval()
        L(f"[{cfg.name}] finished run loaded from {model_path}")
        return _result_from_model_file(obj, out, model_path)

    # ---- set up ------------------------------------------------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("fit: the model has no trainable parameters")
    opt = torch.optim.Adam(_param_groups(params, cfg.weight_decay), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr_end)
    shuffle_gen = torch.Generator().manual_seed(SHUFFLE_SEED_BASE + cfg.seed)
    fold_gen = (torch.Generator(device=device).manual_seed(FOLD_SEED_BASE + cfg.seed)
                if cfg.fold_rng == "dedicated" else None)
    ema = _EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    n_ev = len(train)
    n_batches = (n_ev + cfg.batch - 1) // cfg.batch
    st = {"step": 0, "order": None, "pos": 0, "history": [], "skipped_loss": 0, "skipped_grad": 0,
          "best": {"main": (math.inf, None, None), "ema": (math.inf, None, None)},    # (val, step, state)
          "win_sum": 0.0, "win_n": 0, "seconds": 0.0, "resumed_from": None}

    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["config_hash"] != chash:
            raise ConfigMismatch(f"{ck_path}: checkpoint has config {ck['config_hash']}, this call {chash}; "
                                 "use another name/out_dir or delete the checkpoint")
        _check_machine(ck.get("machine_tag"), mtag, f"{ck_path} (checkpoint)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        shuffle_gen.set_state(ck["shuffle_gen"])
        if fold_gen is not None:
            fold_gen.set_state(ck["fold_gen"])
        if ema is not None:
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        _set_rng_states(ck["rng"])
        for k in ("step", "order", "pos", "history", "skipped_loss", "skipped_grad", "win_sum", "win_n", "seconds"):
            st[k] = ck[k]
        st["best"] = {s: (v, stp, ({k: t.to(device) for k, t in state.items()} if state is not None else None))
                      for s, (v, stp, state) in ck["best"].items()}
        st["resumed_from"] = st["step"]
        L(f"[{cfg.name}] resuming from step {st['step']} of {cfg.steps} ({ck_path})")
    else:
        seed_everything(cfg.seed)
        L(f"[{cfg.name}] start: {cfg.steps} steps, batch {cfg.batch}, K_train {cfg.k_train} ({cfg.fold}), K_val "
          f"{cfg.k_val}, lr {cfg.lr:g} -> {cfg.lr_end:g}, selection {cfg.selection}, EMA {cfg.ema_decay}, "
          f"wd {cfg.weight_decay:g}, smooth {cfg.smooth:g}, seed {cfg.seed}, {sum(p.numel() for p in params)} "
          f"trainable parameters, device {device}")

    win_sum = torch.tensor(st["win_sum"], dtype=torch.float64, device=device)
    t_start = time.time() - st["seconds"]

    def save_checkpoint():
        atomic_save({"config": cfg.result_dict(), "config_hash": chash, "model": _clone_state(model),
                     "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "shuffle_gen": shuffle_gen.get_state(),
                     "fold_gen": fold_gen.get_state() if fold_gen is not None else None,
                     "ema": ({k: v.detach().clone() for k, v in ema.shadow.items()} if ema is not None else None),
                     "rng": _rng_states(), "step": st["step"], "order": st["order"], "pos": st["pos"],
                     "history": st["history"], "skipped_loss": st["skipped_loss"], "skipped_grad": st["skipped_grad"],
                     "best": st["best"], "win_sum": float(win_sum), "win_n": st["win_n"],
                     "seconds": time.time() - t_start, "saved_utc": datetime.now(timezone.utc).isoformat(),
                     "machine_tag": mtag.to_dict(), "host_id": sfm_machine.host_id()}, ck_path)

    def note_skip(kind: str):
        st[kind] += 1
        n = st["skipped_loss"] + st["skipped_grad"]
        if n <= 10 or n % 100 == 0:
            L(f"[{cfg.name}] step {st['step'] + 1}: non-finite {'loss' if kind == 'skipped_loss' else 'gradient'}, "
              f"batch skipped (skipped so far: loss {st['skipped_loss']}, gradient {st['skipped_grad']})")

    def validate_now():
        v = float(np.mean(val_nll(model, val, cfg.k_val, cfg.val_seed, cfg.val_batch), dtype=np.float64))
        v_ema = None
        if ema is not None:
            live = _clone_state(model)
            model.load_state_dict(ema.state(model))
            v_ema = float(np.mean(val_nll(model, val, cfg.k_val, cfg.val_seed, cfg.val_batch), dtype=np.float64))
            model.load_state_dict(live)
        return v, v_ema

    ctx = metric.deterministic() if cfg.deterministic else contextlib.nullcontext()
    with ctx:
        model.train()
        while st["step"] < cfg.steps:
            if st["order"] is None or st["pos"] >= n_batches:
                st["order"], st["pos"] = torch.randperm(n_ev, generator=shuffle_gen), 0
            idx = st["order"][st["pos"] * cfg.batch:(st["pos"] + 1) * cfg.batch].to(train.off.device)
            st["pos"] += 1
            b = train.batch(idx)
            nll, logP = metric.event_nll(model, b, train.box, cfg.k_train, fold_gen, fold=cfg.fold)
            loss = nll.mean()
            if cfg.smooth:
                loss = loss + cfg.smooth * ((logP[:, 2:] - 2 * logP[:, 1:-1] + logP[:, :-2]) ** 2).mean()
            if not bool(torch.isfinite(loss)):
                note_skip("skipped_loss")
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grads = [p.grad for p in params if p.grad is not None]
                if grads and not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                    note_skip("skipped_grad")
                else:
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip)
                    opt.step()
                    if ema is not None:
                        ema.update(model)
                    win_sum += nll.detach().mean().double()
                    st["win_n"] += 1
            lr_used = opt.param_groups[0]["lr"]
            sched.step()
            st["step"] += 1
            step = st["step"]
            if step % cfg.eval_every == 0 or step == cfg.steps:
                v, v_ema = validate_now()
                tr = float(win_sum) / st["win_n"] if st["win_n"] else float("nan")
                st["history"].append({"step": step, "lr": lr_used, "train_nll": tr, "n_train_steps": st["win_n"],
                                      "val": v, "val_ema": v_ema, "skipped_loss": st["skipped_loss"],
                                      "skipped_grad": st["skipped_grad"], "seconds": time.time() - t_start})
                win_sum.zero_()
                st["win_n"] = 0
                for stream, value in (("main", v), ("ema", v_ema)):
                    if value is not None and value < st["best"][stream][0]:
                        state = (_clone_state(model) if stream == "main" else ema.state(model)) \
                            if cfg.selection == "best_val" else None
                        st["best"][stream] = (value, step, state)
                skips = (f"  skipped loss/grad {st['skipped_loss']}/{st['skipped_grad']}"
                         if st["skipped_loss"] + st["skipped_grad"] else "")
                L(f"[{cfg.name}] step {step:6d}  lr {lr_used:.3g}  train {tr:.5f}  val {v:.6f}"
                  + (f"  val_ema {v_ema:.6f}" if v_ema is not None else "")
                  + f"  best {st['best']['main'][0]:.6f}@{st['best']['main'][1]}{skips}  ({time.time() - t_start:.0f}s)")
            if step % cfg.ckpt_every == 0 or step == cfg.steps:
                save_checkpoint()
            if callback is not None:
                callback(step, model)

    # ---- finish ------------------------------------------------------------------------------------------------------
    hist = st["history"]
    final_val = hist[-1]["val"] if hist else None
    final_val_ema = hist[-1]["val_ema"] if hist else None
    if cfg.selection == "best_val" and st["best"]["main"][2] is not None:
        sel_val, sel_step, sel_state = st["best"]["main"]
    else:
        sel_val, sel_step, sel_state = final_val, st["step"], _clone_state(model)
    ema_state = None
    if ema is not None:
        if cfg.selection == "best_val" and st["best"]["ema"][2] is not None:
            ema_state = st["best"]["ema"][2]
        else:
            ema_state = ema.state(model)
    seconds = time.time() - t_start
    summary = {"name": cfg.name, "step": st["step"], "skipped_loss": st["skipped_loss"],
               "skipped_grad": st["skipped_grad"], "selection": cfg.selection, "selected_step": sel_step,
               "selected_val": sel_val, "final_val": final_val, "final_val_ema": final_val_ema,
               "best_val": st["best"]["main"][0] if math.isfinite(st["best"]["main"][0]) else None,
               "best_step": st["best"]["main"][1], "resumed_from": st["resumed_from"], "seconds": seconds}
    obj = {"state": sel_state, "ema_state": ema_state, "config": cfg.result_dict(), "config_hash": chash,
           "history": hist, "summary": summary, "meta": copy.deepcopy(meta) if meta else {},
           "saved_utc": datetime.now(timezone.utc).isoformat(), "machine_tag": mtag.to_dict(),
           "host_id": sfm_machine.host_id()}
    atomic_save(obj, model_path)
    done = {**summary, "config_hash": chash, "config": cfg.result_dict(), "model": MODEL_FILE,
            "model_sha256": sha256_file(model_path), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "machine_tag": mtag.to_dict(), "host_id": sfm_machine.host_id(),
            "weights_sha256": _weights_sha256(sel_state)}
    atomic_write_json(done, done_path)
    model.load_state_dict(sel_state)
    model.eval()
    L(f"[{cfg.name}] done: {st['step']} steps, skipped batches loss/grad {st['skipped_loss']}/{st['skipped_grad']}, "
      f"selected {cfg.selection} @ step {sel_step} (val {sel_val}), {seconds:.0f}s -> {model_path}")
    return _result_from_model_file(obj, out, model_path, seconds)


# ----------------------------------------------------------------------------------------------------------------------
# Run configuration files (configs/*.toml), used by scripts/fit.py and scripts/queue.py
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class EvalSpec:
    registry: bool = True            # call sfm.registry.evaluate after the fit
    guards: bool = False             # add the guard columns (K = 1024, bulk stratum, mass monitors)
    k: Optional[int] = None          # free-constant count for the registry (default: model.n_params())
    C: Optional[float] = None        # operator-weighted complexity (closed forms / SR), if meaningful


@dataclass
class RunConfig:
    name: str
    stage: str
    seed: int
    model_factory: str
    model_kwargs: dict
    fit: FitConfig
    evaluate: EvalSpec
    notes: Optional[str]
    source: Optional[Path]
    raw: dict

    @property
    def out_dir(self) -> Path:
        """This machine's run directory: OUT_RESULTS/machines/<machine id>/runs/<stage>/<name> (pristine-clone rule:
        a run trained here never writes into the repository)."""
        return run_output_dir(self.stage, self.name)

    @property
    def done_file(self) -> Path:
        return self.out_dir / RUN_DONE_FILE

    def hash(self) -> str:
        """Hash of everything that determines the run (the whole file except ``run.notes``)."""
        raw = copy.deepcopy(self.raw)
        raw.get("run", {}).pop("notes", None)
        return config_hash(raw)


def run_output_dir(stage: str, name: str) -> Path:
    """OUT_RESULTS/machines/<live machine id>/runs/<stage>/<name>."""
    from .provenance import run_dir
    return run_dir(paths.OUT_RESULTS, sfm_machine.machine_tag().id, stage, name)


def load_run_config(path) -> RunConfig:
    """Typed loader of a run file (TOML)::

        [run]       name (required), stage (required), seed (default 0), notes (optional)
        [model]     factory (required, a name in sfm.models.FACTORIES) + its keyword arguments
        [fit]       FitConfig fields except name, seed and out_dir (out_dir = this machine's run directory,
                    OUT_RESULTS/machines/<machine id>/runs/<stage>/<name>)
        [evaluate]  registry (true), guards (false), k, C   (optional table)

    Unknown sections or keys are errors.
    """
    import tomllib

    path = Path(path)
    raw = tomllib.loads(path.read_text())
    extra = set(raw) - {"run", "model", "fit", "evaluate"}
    if extra:
        raise KeyError(f"{path}: unknown section(s) {sorted(extra)}")
    for sec in ("run", "model"):
        if sec not in raw:
            raise KeyError(f"{path}: missing section [{sec}]")
    run = dict(raw["run"])
    unknown = set(run) - {"name", "stage", "seed", "notes"}
    if unknown:
        raise KeyError(f"{path}: unknown key(s) in [run]: {sorted(unknown)}")
    name, stage = run.get("name"), run.get("stage")
    for what, v in (("name", name), ("stage", stage)):
        if not isinstance(v, str) or not v or "/" in v or v.startswith(".") or ".." in v:
            raise ValueError(f"{path}: run.{what} must be a non-empty name without '/'")
    seed = run.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"{path}: run.seed must be an int")
    model = dict(raw["model"])
    factory = model.pop("factory", None)
    if not isinstance(factory, str):
        raise KeyError(f"{path}: [model] needs factory = \"...\"")
    fit_d = dict(raw.get("fit", {}))
    for k in ("name", "seed", "out_dir"):
        if k in fit_d:
            raise KeyError(f"{path}: [fit] must not set {k} (it comes from [run])")
    fcfg = FitConfig.from_dict({**fit_d, "name": name, "seed": seed, "out_dir": str(run_output_dir(stage, name))})
    ev_d = dict(raw.get("evaluate", {}))
    unknown = set(ev_d) - {f.name for f in fields(EvalSpec)}
    if unknown:
        raise KeyError(f"{path}: unknown key(s) in [evaluate]: {sorted(unknown)}")
    ev = EvalSpec(**ev_d)
    if not isinstance(ev.registry, bool) or not isinstance(ev.guards, bool):
        raise TypeError(f"{path}: evaluate.registry and evaluate.guards must be booleans")
    if ev.k is not None and (isinstance(ev.k, bool) or not isinstance(ev.k, int)):
        raise TypeError(f"{path}: evaluate.k must be an int")
    if ev.C is not None:
        ev.C = float(ev.C)
    return RunConfig(name=name, stage=stage, seed=seed, model_factory=factory, model_kwargs=model, fit=fcfg,
                     evaluate=ev, notes=run.get("notes"), source=path.resolve(), raw=raw)
