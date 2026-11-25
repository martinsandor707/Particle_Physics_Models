"""Shared helpers of the Track-B (M2B) run scripts ``scripts/run_m2b_*.py`` and of the batch commands
``scripts/val_parity.py`` / ``scripts/rescore_registered.py``.

    start(what)                     E10 start-up (``sfm.machine.startup``: CUDA or a hard failure, TF32 off, per-job
                                    caps, clean clone in pristine mode); prints and returns the batch record
    STAGE                           registry stage "m2b_mlp"
    OUT                             THIS machine's directory of the stage: OUT_RESULTS/machines/<id>/runs/m2b_mlp
                                    (outputs; on the H200 outside the clone)
    REPO_OUT                        the clone's results/m2b_mlp (laptop records and inputs, read-only)
    S1_SEEDS, B0_SEEDS              seeds of the five independent 4x32 networks and of the B0 control copies
    CONTROL_NAMES                   the H200 controls b0h_seed0..2 (user decision P5); LAPTOP_CONTROL_RECORD b0_seed0
    LAPTOP_TRAINED                  the eight networks trained on the laptop (records; re-scored, never retrained)
    live_stream_id()                stream id of this process (machine + points tag)
    registered_row(name, ckpt)      latest row of (STAGE, name) on the LIVE stream whose checkpoint SHA-256 equals the
                                    file's (laptop rows and rows of other machines never match; spec point 11)
    done_marker(path)               a done.json / fit_done.json written on THIS machine, else None; refuses a marker
                                    of another machine
    member(run)                     "out:..." for a run trained on this machine, "repo:..." for a laptop record
    stage1_epochs(hist)             (epochs run, best epoch) of a ``common.fit`` history, replaying its rule
    Resources                       wall-clock and peak CUDA memory of one phase
    free_cuda()                     release cached CUDA memory between jobs (the GPU is shared through the lock)
    run_on_gpu(label, fn)           ``fn()`` under the GPU lock; on a capped (shared, large) device the job is refused
                                    when less than cap + 4 GB is free; on the laptop the lock is released and retried

Nothing here touches the test split or the registry file directly.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from sfm import gpulock, machine, paths, provenance, registry  # noqa: E402

STAGE = "m2b_mlp"
MACHINE = machine.machine_tag()                       # TF32 is already off (sfm.machine import)
MACHINE_DIR = provenance.machine_dir(paths.OUT_RESULTS, MACHINE.id)
OUT = MACHINE_DIR / "runs" / STAGE
REPO_OUT = paths.REPO_RESULTS / STAGE
S1_SEEDS = (11, 12, 13, 14, 15)
B0_SEEDS = (0, 1, 2)
CONTROL_NAMES = tuple(f"b0h_seed{s}" for s in B0_SEEDS)        # P5: the H200 controls
LAPTOP_CONTROL_RECORD = "b0_seed0"
LAPTOP_TRAINED = tuple([f"s1_seed{s}" for s in S1_SEEDS] + ["s2_seed11", "s2_seed12", "b0_seed0"])
STAGE1_RECIPE = {"model": "common.FieldModel(hidden=32, n_layers=4, n_freq=4)",
                 "trainer": "common.fit (unmodified)", "epochs": 60, "lr": 3e-3, "smooth": 1e-4, "patience": 15,
                 "batch": 32, "k_train": 16, "k_val_selection": 16, "min_improvement": 1e-4,
                 "seeding": "common.configure(batch=32) first (re-seeds with common.SEED=1), then random/numpy/torch "
                            "seeded with s immediately before the model is built; common.SEED = s during "
                            "common.fit (its shuffle generator is seeded with common.SEED + start_epoch), "
                            "restored afterwards"}


def start(what: str) -> dict:
    """E10 start-up of a batch command; returns (and prints) what every batch records."""
    tag = machine.startup(what)
    from sfm import guard
    rec = {"what": what, "machine_id": tag.id, "host_id": machine.host_id(), "caps": machine.caps(),
           "git": machine.git_state(), "pristine_mode": paths.pristine_mode(),
           "out_results": paths.portable(paths.OUT_RESULTS), "file_access_guard_active": guard.active(),
           "laptop_of_record": machine.is_laptop_of_record(tag), "env": machine.env_record()}
    print(f"[{what}] machine {tag.id} (GPU {tag.fields['gpu_name']}, grid {tag.fields['launch_grid']}, torch "
          f"{tag.fields['torch']}, Python {tag.fields['python']}, TF32 {tag.fields['tf32_matmul']}/"
          f"{tag.fields['tf32_cudnn']}); caps {rec['caps']}; git {rec['git']}; outputs under {rec['out_results']}; "
          f"file-access guard {'ACTIVE' if rec['file_access_guard_active'] else 'off'}",
          flush=True)
    return rec


def finish(rec: dict) -> dict:
    """End-of-batch record: the git state again (the clone must still be clean in pristine mode)."""
    rec = dict(rec)
    rec["git_at_end"] = machine.git_state()
    if paths.pristine_mode() and rec["git_at_end"].get("clean") is not True:
        print("WARNING: the clone is no longer clean at the end of the batch: "
              f"{rec['git_at_end']}", flush=True)
    return rec


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


@lru_cache(maxsize=1)
def live_stream_id() -> str:
    return registry.live_stream()["stream_id"]


def registered_row(name: str, ckpt=None, need_guards: bool = False):
    """The latest row of (STAGE, name) on the live stream with the same checkpoint SHA-256 (and guards, if
    required), from this machine's registry."""
    return registry.registered_row(STAGE, name, stream_id=live_stream_id(), ckpt=ckpt, need_guards=need_guards)


def done_marker(path) -> dict | None:
    """The marker at ``path`` if it exists and was written on this machine; None if absent. A marker of another
    machine, or without a machine tag inside this machine's directory, raises."""
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    tag = d.get("machine_tag") or {}
    if tag.get("id") != MACHINE.id:
        raise RuntimeError(f"{paths.portable(path)} was written on machine {tag.get('id')!r}, this is {MACHINE.id!r}; "
                           "markers are never shared across machines")
    return d


def laptop_record_done(name: str) -> Path | None:
    """The laptop's done.json of a run in the clone (a record), if any."""
    p = REPO_OUT / name / "done.json"
    return p if p.exists() else None


def member(run_name: str) -> str:
    """Run reference for the ensemble factory: this machine's run if it exists, else the laptop record in the clone
    (only for the networks of LAPTOP_TRAINED)."""
    mine = OUT / run_name / "model.pt"
    if mine.exists():
        return f"out:{os.path.relpath(OUT / run_name, paths.OUT_RESULTS)}"
    if run_name in LAPTOP_TRAINED and (REPO_OUT / run_name / "model.pt").exists():
        return f"repo:{STAGE}/{run_name}"
    raise FileNotFoundError(f"{STAGE}/{run_name}: not trained on this machine ({paths.portable(mine)}) and not a "
                            "laptop-trained record")


def stage1_epochs(hist: dict) -> tuple:
    """(epochs run, best epoch) of a ``common.fit`` history; the best epoch (0-based, as common.fit counts) is the
    last epoch at which ``val < best - 1e-4`` reset the patience counter, i.e. the epoch whose state was kept."""
    best, best_ep = math.inf, None
    for i, v in enumerate(hist["val"]):
        if v < best - 1e-4:
            best, best_ep = v, i
    return len(hist["val"]), best_ep


class Resources:
    """with Resources() as r: ...   ->  r.seconds, r.peak_alloc_mb, r.peak_reserved_mb (CUDA peak inside the block)."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_alloc_mb = torch.cuda.max_memory_allocated() / 2 ** 20
            self.peak_reserved_mb = torch.cuda.max_memory_reserved() / 2 ** 20
        else:
            self.peak_alloc_mb = self.peak_reserved_mb = None
        self.seconds = time.time() - self.t0
        return False

    def as_dict(self) -> dict:
        return {"seconds": self.seconds, "peak_alloc_mb": self.peak_alloc_mb, "peak_reserved_mb": self.peak_reserved_mb}


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_on_gpu(label: str, fn, min_free_mb: float = 3000.0, poll: float = 30.0, max_wait: float = 6 * 3600.0):
    """Run ``fn()`` under ``sfm.gpulock.gpu_lock(label)``.

    Shared, capped device (``SFM_GPU_MEM_GB`` in force, default on devices > 24 GB): after the lock is taken the free
    device memory must be at least the job's cap + 4 GB (``sfm.machine.check_free_memory``), otherwise the job is
    refused at once (``InsufficientGPUMemory``; the device is never empty, so there is no waiting).
    Uncapped device (the laptop): the lock alone does not guarantee memory (another process can hold cached memory),
    so if less than ``min_free_mb`` is free the lock is RELEASED (never wait for memory while holding it) and the
    check is repeated after ``poll`` seconds, up to ``max_wait`` seconds in total (then RuntimeError)."""
    machine.require_cuda(label)
    capped = bool(machine.apply_caps().get("gpu_mem_cap_gb"))
    t0 = time.time()
    while True:
        with gpulock.gpu_lock(label):
            if capped:
                info = machine.check_free_memory(label)
                print(f"[run_on_gpu] {label}: {info['free_gb']:.1f} GB free, cap {info['cap_gb']:.1f} GB", flush=True)
                return fn()
            free = torch.cuda.mem_get_info()[0] / 2 ** 20
            if free >= min_free_mb:
                return fn()
        waited = time.time() - t0
        print(f"[run_on_gpu] {label}: only {free:.0f} MB free on the device (need {min_free_mb:.0f}); lock released, "
              f"retry in {poll:.0f} s (waited {waited:.0f} s)", flush=True)
        if waited > max_wait:
            raise RuntimeError(f"{label}: device memory did not become free within {max_wait:.0f} s")
        time.sleep(poll)

# ----------------------------------------------------------------------------------------------------------------------
# Starting points of the H200 fits (review B2): only the pre-registered ones, verified by SHA-256 and recorded
# ----------------------------------------------------------------------------------------------------------------------
ALLOWED_INIT = {**{f"s2_seed{s}": ("from_run", {"run": f"{STAGE}/s1_seed{s}"}) for s in S1_SEEDS},
                **{n: ("field_mlp_std", {}) for n in CONTROL_NAMES}}
MLP_STD_REL = "shape-form-proton/results/particles/pion/mlp_std.pt"


def input_hashes(sums_file=None) -> dict:
    """{repo-relative path: SHA-256} of the "inputs" section of the tracked transfer/SHA256SUMS."""
    import re
    sums_file = Path(sums_file or provenance.SHA256SUMS)
    out, on = {}, False
    for line in sums_file.read_text().splitlines():
        if re.match(r"^# BEGIN inputs$", line):
            on = True
            continue
        if re.match(r"^# END inputs$", line):
            on = False
            continue
        if on and line and not line.startswith("#"):
            h, rel = line.split(None, 1)
            out[rel.strip()] = h
    return out


def verify_init(name: str, factory: str, kwargs: dict, *, repo_results=None, legacy_registry=None,
                sums_file=None) -> dict:
    """Check that ``name`` starts from its pre-registered starting point and that the starting file is the one on
    record; return the ``init_from`` record. Raises SystemExit otherwise (nothing is trained)."""
    from sfm import provenance as P
    exp = ALLOWED_INIT.get(name)
    if exp is None:
        raise SystemExit(f"{STAGE}/{name}: no pre-registered starting point (allowed: {sorted(ALLOWED_INIT)})")
    if (factory, dict(kwargs or {})) != exp:
        raise SystemExit(f"{STAGE}/{name}: starting point {factory} {kwargs} is not the pre-registered {exp}")
    repo_results = Path(repo_results or paths.REPO_RESULTS)
    if factory == "from_run":
        run = kwargs["run"]
        stage, src = run.split("/")
        f = repo_results / stage / src / "model.pt"
        rows = [r for r in registry.read_registry(Path(legacy_registry or registry.LEGACY_REGISTRY))
                if r.get("stage") == stage and r.get("name") == src and registry.row_stream_id(r) is None]
        if not rows:
            raise SystemExit(f"{run}: no laptop registry row to verify the starting point against")
        row = rows[-1]
        sha = P.sha256_file(f)
        if sha != row.get("checkpoint_sha256"):
            raise SystemExit(f"{run}: starting file SHA-256 {sha[:12]} differs from the laptop row "
                             f"({str(row.get('checkpoint_sha256'))[:12]}); not trained")
        obj = torch.load(f, map_location="cpu", weights_only=False)
        return {"run": run, "factory": factory, "file": paths.portable(f), "file_sha256": sha,
                "weights_sha256": P.weights_sha256(obj["state"]), "trained_on": machine.LAPTOP_RECORD_LABEL,
                "source_row": {"stage": stage, "name": src, "timestamp": row.get("timestamp"),
                               "checkpoint_sha256": row.get("checkpoint_sha256"),
                               "val_nll_sha256": row.get("val_nll_sha256")}}
    f = paths.PION_RESULTS_DIR / "mlp_std.pt"
    want = input_hashes(sums_file).get(MLP_STD_REL)
    sha = P.sha256_file(f)
    if not want or sha != want:
        raise SystemExit(f"mlp_std.pt: SHA-256 {sha[:12]} differs from transfer/SHA256SUMS ({str(want)[:12]})")
    obj = torch.load(f, map_location="cpu", weights_only=False)
    return {"run": None, "factory": factory, "file": paths.portable(f), "file_sha256": sha,
            "weights_sha256": P.weights_sha256(obj["state"]), "trained_on": "earlier study (shape-form-proton)",
            "source_row": None}


def check_built_weights(model, init: dict) -> None:
    """The model built by the factory must hold exactly the verified starting weights."""
    w = provenance.weights_sha256({k: v for k, v in model.state_dict().items()})
    if w != init["weights_sha256"]:
        raise SystemExit(f"built model weights {w[:12]} differ from the verified starting file "
                         f"({init['weights_sha256'][:12]}); not trained")
