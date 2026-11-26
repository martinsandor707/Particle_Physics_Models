"""Track B (M2B) driver: the remaining pre-registered runs of results/m1_audit/AUDIT.md section 4.2, run IN FULL, in
table order, one GPU job at a time, in ONE process, resumable (PLAN.md section 16.7 E6; section 13 A6 revised points
4-6; the pattern of scripts/run_m2a.py).

    .venv/bin/python -B shape-form-martin/scripts/run_m2b_list.py              # default phases, in order (below)
    .venv/bin/python -B shape-form-martin/scripts/run_m2b_list.py <phase> ...  # selected phases (same order rules)
    .venv/bin/python -B shape-form-martin/scripts/run_m2b_list.py --list       # the list derived from AUDIT 4.2 with
                                                                                # the status of every run (no training)
    .venv/bin/python -B shape-form-martin/scripts/run_m2b_list.py --pin        # laptop only: write the pinned recipe
                                                                                # file configs/m2b/list_recipes.json
    SFM_M2B_SMOKE=1 ... run_m2b_list.py --smoke                                 # shortened fits, stage m2b_smoke

Phases (default order): verify b1d b1f b2 b3s b4 b5 b6 bceil converge guards report
    verify    the inputs of the list: this machine's reference set; the batch-1 runs it builds on (S2-11..15, the
              B0 controls b0h_seed0..2, B1 = E5_s2) registered on the live stream with their model files; mlp_std.pt
              and the acceptance templates by SHA-256 (front / stored: transfer/SHA256SUMS; full_box: the Track A
              templates.json of this machine); every recipe equal to the pinned file (N5)
    b1d       B1d: FieldNet(32, 4) started from s2_seed11 and from s2_seed12 - the fixed, validation-blind seed order
              decided by the user (PLAN section 23 item 7; never a validation ranking) - KL-distilled data-free to E5
              on 256 training-momentum quantiles, 6000 steps; B1d_m12 is the replicate of B1d_m11
    b1f       B1f: B1d_m11 (the first replicate) + 3000 steps 3e-4 -> 3e-5, K = 16, T_B otherwise, seed 0
    b2        B2a (K = 16 iid), B2b (K = 64 iid), B2c (K = 16 stratified): field_mlp_std, T_B, seeds 0 and 1; the EMA
              stream of each is registered as <name>_ema (B3)
    b3s       B3s: fresh field(32, 4), 12000 steps 3e-3 -> 1e-4, K = 16, smooth 1e-4, EMA 0.999, seeds 0 and 1
    b4        B4f (front template, seeds 0, 1), B4s (stored) and B4x (full_box) seed 0: field_acc(base mlp_std,
              warm_steps 500), T_B K = 16
    b5        B5f: field_mlp_std, T_B K = 16, smooth 1e-4, seeds 0 and 1
    b6        B6a / B6b / B6c: fresh field(32, 4) / (32, 5) / (64, 4), KL warm start to mlp_std (256 quantiles,
              3000 steps, 1e-3), then T_B K = 16, seeds 0 and 1
    bceil     B-ceil: residual_table(base mlp_std, knots uniform, freeze_base): (1 knot; wd 0, 1e-4; 100 %),
              (6 knots; wd 0, 1e-4; 25, 50, 100 % of train), 3000 steps 3e-3 -> 3e-4, K = 16, eval_every 100, seed 0
    converge  AUDIT 4.0: a primary whose validation mean moved by >= 0.5e-4 over its last 1000 steps is rerun once
              with doubled steps together with its comparator (<name>_x2); nothing else changes
    guards    registry.evaluate(guards=True) (G-K, G-bulk, G-mass) and G-region arrays for every run that meets the
              promotion parts (a)-(c) on a seed, for its other seed, and G-region arrays of mlp_std as reference
    report    scripts/run_m2b_list_report.py: the pre-registered rules applied to the registered arrays
Prepared, NOT in the default phases (runs only when named):
    b1d_companions  I4 option (beta) of the Fable ruling (REVIEW.md 9.1; the choice is with the user): one data-free
              companion per B1d run with 12,000 KL steps (B1d_m11_kl12000, B1d_m12_kl12000; pinned). Under option
              (alpha), the one implemented, B1d has no convergence statement; its KL trajectory (the logged lines and
              the initial / final KL) is stored in init.json, model.pt and done.json and printed by the report.

Hard stops (user, PLAN section 23 item 13): a failed parity, reference, scoring, guard or file-access / deferred-file
check ends the job at once with exit code 3 and the last line "HARD STOP (PLAN.md section 23, item 13) - ..."; too
little free GPU memory ends it with exit code 4 ("STOPPED (resources)", resumable). See results/m2b_mlp/STATUS.md 3b.
Scoring check (Fable ruling A2 (i), boundary case): the fit's last validation mean and the registry's mean of the same
weights (main and EMA row) must agree to 1e-6; a finite fit value with a non-finite registry score is a hard stop, not
a divergence (SCORING_CHECK_NOTE). A candidate's own non-finite outcome (non-finite last validation mean, non-finite KL
in its warm start, the data-free B1d with a non-finite score) stays a recorded divergence.
No validation value chooses what is trained: selection "last" everywhere, fixed B1d/B1f starts; validation values
only drive pre-registered rules on what is re-run (convergence) or reported (guards, 1-SE) and the divergence labels.
Never re-run, re-pair or changed: the runs registered by batch 1 (recognised by their rows on the live stream). The
eight laptop-trained networks stay records (used only as members / starting points, verified by SHA-256).
Divergence (PLAN section 20 item 2): a run is diverged iff its LAST validation mean is not finite, the registry finds a
non-finite per-event NLL, the last evaluation window is frozen (>= 50 % of its batches skipped), or the last value
exceeds the reference (best of history, comparator) by more than 1 nat; recovered non-finite values in the middle of a
fit are warnings (count in done.json). Exploratory runs: recorded, the job continues. Primaries, comparators and
diagnostics: "diverged - awaiting diagnosis" (never "failed" without runs/<stage>/diagnosis/<name>.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
if __name__ == "__main__":
    # ONE module instance: the report (and the guards phase through it) imports this file as "run_m2b_list"; without
    # this alias it would get a second copy with its own CTX (stage, smoke flag) and its own HardStop class
    sys.modules.setdefault("run_m2b_list", sys.modules[__name__])
import run_m2b_lib as lib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import data as sfm_data  # noqa: E402
from sfm import machine, models, paths, provenance, registry, train  # noqa: E402

common = train.C
REAL_STAGE, SMOKE_STAGE = "m2b_mlp", "m2b_smoke"
SMOKE_ENV = "SFM_M2B_SMOKE"
SMOKE_DIV = 100
SMOKE_FORCED_RERUN = "B2b_s0"       # smoke mode only: treated as not converged, so the rerun path runs end to end
PIN_FILE = paths.MARTIN_DIR / "configs" / "m2b" / "list_recipes.json"
DEFAULT_PHASES = ("verify", "b1d", "b1f", "b2", "b3s", "b4", "b5", "b6", "bceil", "converge", "guards", "report")
# I4 option (beta) of the Fable ruling (REVIEW.md 9.1) is PREPARED as its own phase and is NOT in the default list:
# it runs only when named (the choice between (alpha) and (beta) is with the user)
PHASE_ORDER = ("verify", "b1d", "b1d_companions", "b1f", "b2", "b3s", "b4", "b5", "b6", "bceil", "converge", "guards",
               "report")
PRIMARY_RUNGS = ("B1", "B1d", "B2b", "B2c", "B4f", "B6c")             # AUDIT 4.2: the six primaries
CONVERGED_MOVE, CONVERGENCE_WINDOW = 0.5e-4, 1000                     # AUDIT 4.0
BLOWUP_NAT, FROZEN_FRACTION = 1.0, 0.5
# promotion rule (PLAN 6 / AUDIT 4.0), nat per event
PROMO_GAIN, PROMO_Z, B0_Z = -5e-4, -3.0, -1.0
# A2 (i) boundary case (Fable ruling, REVIEW.md 9.1): the fit's last validation mean and the registry's mean of the
# same weights are two scorings of one thing; a larger difference is a failed scoring check (hard stop)
SCORING_TOL = 1e-6
COMPANION_SUFFIX, COMPANION_KL_FACTOR = "_kl12000", 2                 # I4 (beta): 12,000 instead of 6,000 KL steps


class HardStop(RuntimeError):
    """A failed parity, reference, scoring, guard or file-access / deferred-file check (PLAN.md section 23, item 13):
    the job stops at once, before any later phase, and waits for the user."""

    def __init__(self, check: str, msg: str):
        super().__init__(f"{check}: {msg}")
        self.check = check


def hard_stop(check: str, msg: str):
    raise HardStop(check, msg)


HARD_STOP_EXIT, RESOURCE_EXIT = 3, 4


class Ctx:
    stage = REAL_STAGE
    smoke = False
    batch: dict = {}


CTX = Ctx()


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}", flush=True)


def configure(smoke: bool) -> None:
    if smoke and os.environ.get(SMOKE_ENV) != "1":
        raise SystemExit(f"smoke mode refused: set {SMOKE_ENV}=1 explicitly (end-to-end testing only; rows under stage "
                         f"'{SMOKE_STAGE}', never numbers of record)")
    if not smoke and os.environ.get("SFM_M2B_INJECT_DIVERGENCE"):
        raise SystemExit("SFM_M2B_INJECT_DIVERGENCE is a smoke-mode testing hook; refused in the real mode")
    CTX.stage = SMOKE_STAGE if smoke else REAL_STAGE
    CTX.smoke = bool(smoke)


def out_dir() -> Path:
    return lib.MACHINE_DIR / "runs" / CTX.stage


def decisions_file() -> Path:
    return out_dir() / "list_decisions.json"


# ----------------------------------------------------------------------------------------------------------------------
# the list (AUDIT 4.2) - the tracked spec; every recipe is pinned by hash in configs/m2b/list_recipes.json
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Spec:
    name: str
    rung: str
    cls: str                          # primary | comparator | exploratory | diagnostic | rerun
    factory: str
    kwargs: dict
    seed: int = 0
    steps: int = 6000                 # T_B (AUDIT 4.2)
    lr: float = 1e-3
    lr_end: float = 1e-4
    batch: int = 256
    k_train: int = 16
    fold: str = "iid"
    smooth: float = 0.0
    weight_decay: float = 0.0
    ema_decay: Optional[float] = 0.999
    eval_every: int = 250
    selection: str = "last"
    train_fraction: float = 1.0
    data_free: bool = False           # B1d: the factory's KL fit is the whole training (no data)
    ema_row: bool = False             # B3: register the EMA stream as <name>_ema
    comparator: Optional[str] = None  # a run of this list, a batch-1 run, or "mlp_std"
    replicate_of: Optional[str] = None   # the seed-0 run of the same rung (second-seed replication)
    notes: str = ""

    def recipe(self) -> dict:
        d = asdict(self)
        d.pop("notes")
        return d

    def recipe_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.recipe(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    @property
    def out_dir(self) -> Path:
        return out_dir() / self.name


# User decision of 26 Nov 2026 (PLAN.md section 23, item 7): the two starting networks of B1d / B1f are NOT picked by
# a validation score; the strictly validation-blind fixed seed order is used. B1f starts from the first replicate.
B1D_START_ORDER = ("s2_seed11", "s2_seed12")
T_B = {}                                   # Spec defaults are T_B: 6000 steps, cosine 1e-3 -> 1e-4, batch 256,
                                           # k_val 64, eval_every 250, selection last, smooth 0, wd 0, dedicated
                                           # fold generator, EMA 0.999


def build_list() -> list:
    """The remaining runs of AUDIT 4.2, in table order. Batch-1 runs (S2-11..15, B0 = b0h_seed0..2, B1 = E5_s2,
    B1c = the subset arrays + E6) are listed separately (``given``)."""
    L = []
    for member in B1D_START_ORDER:                                      # B1d: two starts, data-free; fixed seed
        L.append(Spec(f"B1d_m{member[-2:]}", "B1d", "primary", "field_distill",   # order (user, PLAN 23 item 7)
                      {"start": member, "target": "<E5>", "grid": 256, "kl_steps": 6000, "kl_lr": 1e-3},
                      steps=0, data_free=True, ema_decay=None, comparator="mlp_std",
                      replicate_of=(None if member == B1D_START_ORDER[0] else f"B1d_m{B1D_START_ORDER[0][-2:]}"),
                      notes=f"distillation of E5 into one FieldNet(32, 4) started from {member} (validation-blind "
                            "fixed seed order); the second start is the replicate"))
    first = f"B1d_m{B1D_START_ORDER[0][-2:]}"
    L.append(Spec("B1f_s0", "B1f", "exploratory", "from_run", {"run": f"<{first}>"}, steps=3000, lr=3e-4, lr_end=3e-5,
                  comparator=first, notes=f"B1d + data fine-tune, from the first B1d replicate ({first})"))
    for rung, kt, fold, cls, comp in (("B2a", 16, "iid", "comparator", "mlp_std"),
                                      ("B2b", 64, "iid", "primary", "B2a"),
                                      ("B2c", 16, "stratified", "primary", "B2a")):
        for s in (0, 1):
            L.append(Spec(f"{rung}_s{s}", rung, cls, "field_mlp_std", {}, seed=s, k_train=kt, fold=fold, ema_row=True,
                          comparator=(comp if comp == "mlp_std" else f"{comp}_s{s}"),
                          replicate_of=(f"{rung}_s0" if s == 1 else None)))
    for s in (0, 1):
        L.append(Spec(f"B3s_s{s}", "B3s", "exploratory", "field", {"hidden": 32, "n_layers": 4}, seed=s, steps=12000,
                      lr=3e-3, lr_end=1e-4, smooth=1e-4, ema_row=True, comparator="mlp_std",
                      replicate_of=("B3s_s0" if s == 1 else None)))
    for s in (0, 1):
        L.append(Spec(f"B4f_s{s}", "B4f", "primary", "field_acc", {"template": "front", "base": "mlp_std",
                                                                  "warm_steps": 500}, seed=s, comparator=f"B2a_s{s}",
                      replicate_of=("B4f_s0" if s == 1 else None)))
    for rung, tpl in (("B4s", "stored"), ("B4x", "full_box")):
        L.append(Spec(f"{rung}_s0", rung, "exploratory", "field_acc", {"template": tpl, "base": "mlp_std",
                                                                       "warm_steps": 500}, seed=0, comparator="B2a_s0"))
    for s in (0, 1):
        L.append(Spec(f"B5f_s{s}", "B5f", "exploratory", "field_mlp_std", {}, seed=s, smooth=1e-4,
                      comparator=f"B2a_s{s}", replicate_of=("B5f_s0" if s == 1 else None)))
    for rung, h, nl, cls in (("B6a", 32, 4, "comparator"), ("B6b", 32, 5, "exploratory"), ("B6c", 64, 4, "primary")):
        for s in (0, 1):
            L.append(Spec(f"{rung}_s{s}", rung, cls, "field_kl",
                          {"hidden": h, "n_layers": nl, "target": "mlp_std", "grid": 256, "kl_steps": 3000,
                           "kl_lr": 1e-3}, seed=s, comparator=("mlp_std" if rung == "B6a" else f"B6a_s{s}"),
                          replicate_of=(f"{rung}_s0" if s == 1 else None)))
    for nk, wd, frac in ((1, 0.0, 1.0), (1, 1e-4, 1.0), (6, 0.0, 0.25), (6, 0.0, 0.5), (6, 0.0, 1.0),
                         (6, 1e-4, 0.25), (6, 1e-4, 0.5), (6, 1e-4, 1.0)):
        L.append(Spec(f"Bceil_k{nk}_wd{'0' if wd == 0 else '1e-4'}_f{int(round(frac * 100))}", "B-ceil", "diagnostic",
                      "residual_table", {"base": "mlp_std", "n_knots": nk, "knots": "uniform", "freeze_base": True},
                      seed=0, steps=3000, lr=3e-3, lr_end=3e-4, eval_every=100, weight_decay=wd, train_fraction=frac,
                      comparator="mlp_std"))
    return L


BATCH1_NAMES = frozenset([f"s1_seed{x}" for x in lib.S1_SEEDS] + [f"s2_seed{x}" for x in lib.S1_SEEDS]
                         + list(lib.CONTROL_NAMES) + ["b0_seed0", "E2_s2_seeds11-12", "E3_s2_seeds11-13",
                                                      "E4_s2_seeds11-14", "E5_s2", "E6_s2_mlpstd", "E5_s1"])
GIVEN = {   # AUDIT 4.2 "taken as given" + B1 / B1c: registered by batch 1 (stage m2b_mlp); never re-run here
    "S2-11..15": [f"s2_seed{s}" for s in lib.S1_SEEDS],
    "B0": list(lib.CONTROL_NAMES),
    "B1": ["E5_s2"],
    "B1c": ["E2_s2_seeds11-12", "E3_s2_seeds11-13", "E4_s2_seeds11-14", "E6_s2_mlpstd"],
}
CLOSED = {"B7": "closed (D6(c): -0.5 +- 0.8 at K = 64; PLAN section 14)", "B3": "the EMA rows of B2a/b/c (<name>_ema)"}


def spec_map() -> dict:
    return {s.name: s for s in build_list()}


def x2_spec(s: Spec) -> Spec:
    """The doubled-step rerun of a primary or of its comparator (AUDIT 4.0 convergence rule)."""
    return replace(s, name=f"{s.name}_x2", cls="rerun", steps=2 * s.steps,
                   comparator=(None if s.comparator in (None, "mlp_std") else f"{s.comparator}_x2"),
                   replicate_of=None, ema_row=False, notes=f"convergence rerun of {s.name} (doubled steps)")


def companion_specs() -> list:
    """I4 option (beta) of the Fable ruling (REVIEW.md 9.1), PREPARED, NOT in the default phases: one unconditional
    data-free companion per B1d run with 12,000 instead of 6,000 KL steps, nothing else changed. No validation number
    decides whether they run (phase ``b1d_companions``, only when named). Under (beta) the 6,000-step run is of
    record iff |Delta(12,000 - 6,000)| < 0.5e-4 on the registered arrays, else the 12,000-step run (the report);
    B1f starts from the 6,000-step run in either case ("B1d + 3000 steps")."""
    out = []
    for s in build_list():
        if s.rung == "B1d":
            out.append(replace(s, name=f"{s.name}{COMPANION_SUFFIX}", cls="companion",
                               kwargs={**s.kwargs, "kl_steps": COMPANION_KL_FACTOR * int(s.kwargs["kl_steps"])},
                               replicate_of=None,
                               notes=f"I4 option (beta): companion of {s.name} with 12,000 KL steps (prepared; runs "
                                     "only when the phase b1d_companions is named)"))
    return out


def companion_map() -> dict:
    return {s.name: s for s in companion_specs()}


# ----------------------------------------------------------------------------------------------------------------------
# pinned recipes (review note N5): the spec above may only run with the hashes recorded in the tracked pin file
# ----------------------------------------------------------------------------------------------------------------------
def pinned() -> dict:
    if not PIN_FILE.exists():
        hard_stop("recipe pin", f"{paths.portable(PIN_FILE)} missing: the recipes are not pinned (run --pin on the laptop "
                         "and commit the file)")
    return json.loads(PIN_FILE.read_text())


def check_pins(specs) -> None:
    clash = sorted(set(s.name for s in specs) & BATCH1_NAMES)
    if clash:
        hard_stop("reference", f"list names collide with batch-1 runs: {clash}")
    pins = pinned().get("recipes", {})
    bad = [(s.name, s.recipe_hash(), pins.get(s.name)) for s in specs if pins.get(s.name) != s.recipe_hash()]
    if bad:
        hard_stop("recipe pin", f"recipes differ from the pinned file {paths.portable(PIN_FILE)}: {bad[:5]} ... "
                         "(N5: only the tracked, pinned recipes run)")
    if paths.pristine_mode() and not str(PIN_FILE.resolve()).startswith(str(paths.REPO_ROOT.resolve())):
        hard_stop("recipe pin", "the pin file must be the tracked file inside the clone")


def check_companion_pins() -> None:
    pins = pinned().get("companion_recipes", {})
    bad = [(s.name, s.recipe_hash(), pins.get(s.name)) for s in companion_specs() if pins.get(s.name) != s.recipe_hash()]
    if bad:
        hard_stop("recipe pin", f"I4 companion recipes differ from the pinned file: {bad}")


def write_pins() -> None:
    if not machine.is_laptop_of_record() or paths.pristine_mode():
        raise SystemExit("--pin runs on the laptop of record only (the file is committed from there)")
    specs = build_list()
    out = {"source": "scripts/run_m2b_list.py build_list() - AUDIT 4.2", "n": len(specs),
           "recipes": {s.name: s.recipe_hash() for s in specs},
           "x2_recipes": {x2_spec(s).name: x2_spec(s).recipe_hash() for s in specs
                          if s.cls in ("primary", "comparator") and not s.data_free},
           "companion_recipes": {s.name: s.recipe_hash() for s in companion_specs()}}
    PIN_FILE.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"pinned {len(specs)} recipes in {paths.portable(PIN_FILE)}")


# ----------------------------------------------------------------------------------------------------------------------
# helpers: decisions, markers, live rows
# ----------------------------------------------------------------------------------------------------------------------
def read_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return paths.portable(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return o


def write_json(obj, path) -> None:
    path = paths.check_output_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    train.atomic_write_json(json.loads(paths.scrub_str(json.dumps(_jsonable(obj)))), path)


def decisions() -> dict:
    return read_json(decisions_file(), {}) or {}


def record_decision(key: str, name: str, value) -> None:
    d = decisions()
    d.setdefault(key, {})[name] = value
    write_json(d, decisions_file())


def done_marker(name: str) -> Optional[dict]:
    return lib.done_marker(out_dir() / name / "done.json")


def diverged_marker(name: str) -> Optional[dict]:
    f = out_dir() / name / "diverged.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if d.get("machine_id") != lib.MACHINE.id:
        raise RuntimeError(f"{paths.portable(f)}: written on machine {d.get('machine_id')!r}")
    return d


def row(name: str, stage: Optional[str] = None, ckpt=None, need_guards: bool = False) -> Optional[dict]:
    return registry.registered_row(stage or CTX.stage, name, stream_id=lib.live_stream_id(), ckpt=ckpt,
                                   need_guards=need_guards)


def given_row(name: str) -> Optional[dict]:
    """A batch-1 run (stage m2b_mlp) on the live stream."""
    return registry.registered_row(REAL_STAGE, name, stream_id=lib.live_stream_id())


def status(s: Spec) -> str:
    if done_marker(s.name) is not None:
        return "done"
    dv = diverged_marker(s.name)
    if dv is not None:
        return dv.get("status_label", "diverged")
    if s.name in (decisions().get("blocked") or {}):
        return "blocked"
    if s.name in (decisions().get("skipped_in_smoke") or {}):
        return "skipped in the smoke mode"
    if (s.out_dir / train.CHECKPOINT).exists():
        return "interrupted (resumes)"
    return "pending"


# ----------------------------------------------------------------------------------------------------------------------
# sources: resolution and verification (B2 of the review: starting points by SHA-256 against a live registry row)
# ----------------------------------------------------------------------------------------------------------------------
def registered_members() -> list:
    """The stage-2 members S2-11..15 that are registered on the live stream, in seed order (no validation number is
    used to order or choose anything). Real mode: all five must be registered (batch 1)."""
    out = []
    for n in GIVEN["S2-11..15"]:
        if given_row(n) is not None:
            out.append(n)
        elif not CTX.smoke:
            hard_stop("reference", f"{n}: not registered on the live stream {lib.live_stream_id()} (batch 1 must be "
                      "done)")
    if len(out) < 2:
        hard_stop("reference", "fewer than two stage-2 members registered on this machine")
    return out


def ensemble_ref() -> str:
    """The B1 ensemble run (E5_s2 of batch 1; smoke: the smoke ensemble of the registered members)."""
    if CTX.smoke:
        return f"out:{os.path.relpath(out_dir() / 'E5_smoke', paths.OUT_RESULTS)}"
    return f"out:{os.path.relpath(lib.OUT / 'E5_s2', paths.OUT_RESULTS)}"


def resolve(s: Spec) -> dict:
    kw = dict(s.kwargs)
    if s.factory == "field_distill":
        if kw["start"] not in GIVEN["S2-11..15"]:
            hard_stop("reference", f"{s.name}: start {kw['start']!r} is not a stage-2 member")
        kw["start"] = lib.member(kw["start"])          # fixed by the spec (validation-blind); file verified later
        kw["target"] = ensemble_ref()
    if s.factory == "from_run" and str(kw.get("run", "")).startswith("<B1d_"):
        kw["run"] = f"out:{os.path.relpath(out_dir() / kw['run'][1:-1], paths.OUT_RESULTS)}"
    if CTX.smoke:
        for k in ("kl_steps", "warm_steps"):
            if k in kw:
                kw[k] = max(2, int(kw[k]) // SMOKE_DIV)
    return kw


def _ref_name(ref: str) -> tuple:
    """(stage, name) of a "repo:<stage>/<name>" / "out:machines/<id>/runs/<stage>/<name>" reference."""
    parts = ref.split(":", 1)[1].split("/")
    return parts[-2], parts[-1]


def verify_sources(s: Spec, kw: dict) -> dict:
    """Every run a new run starts from or is fitted to must have a row on the live stream whose checkpoint SHA-256 is
    the file's and whose weights_sha256 is the file's state; mlp_std.pt and the templates by SHA-256."""
    out = {}
    for key in ("start", "target", "run"):
        ref = kw.get(key)
        if not ref or not isinstance(ref, str) or ":" not in ref:
            continue
        stage, name = _ref_name(ref)
        f = models._run_dir(ref, models.BuildContext()) / "model.pt"
        fsha = provenance.sha256_file(f)
        r = registry.registered_row(stage, name, stream_id=lib.live_stream_id(), ckpt=f)
        if r is None:
            hard_stop("reference", f"{s.name}: source {key}={ref} (SHA-256 {fsha[:12]}) matches no row on the live stream")
        obj = torch.load(f, map_location="cpu", weights_only=False)
        wsha = provenance.weights_sha256(obj["state"])
        if r.get("weights_sha256") != wsha:
            hard_stop("reference", f"{s.name}: source {key}={ref}: weight tensors differ from its row")
        out[key] = {"run": ref, "stage": stage, "name": name, "file_sha256": fsha, "weights_sha256": wsha,
                    "trained_on": r.get("trained_on"), "row_timestamp": r.get("timestamp"),
                    "row_val_nll_sha256": r.get("val_nll_sha256")}
    uses_mlp = (s.factory in ("field_mlp_std",) or kw.get("base") == "mlp_std" or kw.get("target") == "mlp_std")
    if uses_mlp:
        f = paths.PION_RESULTS_DIR / "mlp_std.pt"
        want = lib.input_hashes().get(lib.MLP_STD_REL)
        sha = provenance.sha256_file(f)
        if sha != want:
            hard_stop("reference", f"{s.name}: mlp_std.pt SHA-256 {sha[:12]} differs from transfer/SHA256SUMS")
        out["mlp_std"] = {"file_sha256": sha}
    if s.factory == "field_acc":
        out["template"] = verify_template(kw["template"])
    return out


def verify_template(kind: str) -> dict:
    if kind in ("front", "stored"):
        rel = {"front": "shape-form-martin/results/m0_parity/abar_front.pt",
               "stored": "shape-form-proton/results/particles/pion/abar.pt"}[kind]
        f = paths.REPO_ROOT / rel
        want = lib.input_hashes().get(rel)
        sha = provenance.sha256_file(f)
        if sha != want:
            hard_stop("reference", f"template {kind}: {rel} SHA-256 differs from transfer/SHA256SUMS")
        return {"kind": kind, "file": rel, "file_sha256": sha}
    from sfm.models import dist_factory
    tdir = dist_factory.template_dir()
    meta = read_json(tdir / "templates.json")
    if meta is None:
        hard_stop("reference", f"template {kind}: {paths.portable(tdir)}/templates.json missing (Track A templates phase)")
    sha = provenance.sha256_file(tdir / f"{kind}.pt")
    if (meta.get("sha256") or {}).get(kind) != sha or meta.get("machine_id") != lib.MACHINE.id:
        hard_stop("reference", f"template {kind}: file SHA-256 or machine differs from the Track A templates.json")
    return {"kind": kind, "file": paths.portable(tdir / f"{kind}.pt"), "file_sha256": sha, "track_a_record": True}


# ----------------------------------------------------------------------------------------------------------------------
# data, divergence
# ----------------------------------------------------------------------------------------------------------------------
_DATA = {}


def data_cpu():
    if "tv" not in _DATA:
        log("loading train / val from the cache (CPU)")
        _DATA["tv"] = sfm_data.load_sets(("train", "val"), device="cpu")
    return _DATA["tv"]


def train_subset(tr, frac: float, seed: int):
    """B-ceil: the first round(frac N) events of a permutation seeded with ``seed`` (storage order kept)."""
    if frac >= 1.0:
        return tr, None
    n = len(tr)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    idx = torch.sort(perm[:int(round(frac * n))]).values
    return tr.subset(idx.to(tr.off.device)), {"fraction": frac, "n": int(len(idx)), "seed": int(seed),
                                              "indices_sha256": provenance.sha256_array(idx.numpy())}


DIVERGENCE_DEFINITION = (
    "Diverged iff (i) the LAST validation mean of the fit is not finite, or (ii) the registry finds a non-finite "
    "per-event NLL, or (iii) frozen: >= 50 % of the batches of the last evaluation window were skipped, or (iv) "
    "blow-up: the last validation mean exceeds the lowest of the fit history and of the comparator's registered mean by "
    "more than 1 nat per event. A non-finite validation mean in the middle of a fit that ends finite is a warning "
    "(PLAN section 20, item 2).")


def divergence_check(history: list, refs: list) -> tuple:
    crit, warn = [], []
    vals = [h.get("val") for h in history]
    bad = [h["step"] for h in history if h.get("val") is None or not math.isfinite(float(h["val"]))]
    if history:
        last = history[-1]
        lv = last.get("val")
        if lv is None or not math.isfinite(float(lv)):
            crit.append("last validation mean not finite")
        elif bad:
            warn.append(f"recovered non-finite validation mean at steps {bad}")
        prev = history[-2] if len(history) >= 2 else {"step": 0, "skipped_loss": 0, "skipped_grad": 0}
        n = max(1, int(last["step"]) - int(prev["step"]))
        sk = (last.get("skipped_loss", 0) + last.get("skipped_grad", 0) - prev.get("skipped_loss", 0)
              - prev.get("skipped_grad", 0))
        if sk >= FROZEN_FRACTION * n:
            crit.append(f"frozen: {sk} of {n} batches skipped in the last evaluation window")
        good = [float(v) for v in list(vals) + list(refs) if v is not None and math.isfinite(float(v))]
        if lv is not None and math.isfinite(float(lv)) and good and float(lv) - min(good) > BLOWUP_NAT:
            crit.append(f"blow-up: last {float(lv):.4f} > reference {min(good):.4f} + {BLOWUP_NAT} nat")
    return crit, warn


def status_label(s: Spec) -> str:
    return "diverged" if s.cls == "exploratory" else "diverged - awaiting diagnosis"


def record_divergence(s: Spec, crit, res=None, eval_error=None, extra=None) -> dict:
    rec = {"status": "diverged", "status_label": status_label(s), "run": s.name, "rung": s.rung, "cls": s.cls,
           "seed": s.seed, "criteria": crit, "definition": DIVERGENCE_DEFINITION, "eval_error": eval_error,
           "history": res.history if res is not None else None, "fit": res.summary() if res is not None else None,
           "machine_id": lib.MACHINE.id, "stream_id": lib.live_stream_id(), "batch": CTX.batch,
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}
    write_json(rec, s.out_dir / "diverged.json")
    record_decision("diverged", s.name, {"criteria": crit, "status": rec["status_label"], "cls": s.cls})
    log(f"[{s.name}] DIVERGED ({'; '.join(crit)}) -> {rec['status_label']}; the job continues")
    return rec


def diagnosis(name: str) -> Optional[dict]:
    """runs/<stage>/diagnosis/<name>.json (written by the orchestrator after a Fable diagnosis, never by code); valid
    only with the SHA-256 of the run's diverged.json and a verdict "failed" | "rerun_allowed"."""
    f, dv = out_dir() / "diagnosis" / f"{name}.json", out_dir() / name / "diverged.json"
    if not (f.exists() and dv.exists()):
        return None
    d = json.loads(f.read_text())
    ok = (d.get("run") == name and d.get("diverged_json_sha256") == provenance.sha256_file(dv)
          and d.get("verdict") in ("failed", "rerun_allowed") and d.get("diagnosed_by"))
    return {**d, "valid": bool(ok)}


SCORING_CHECK_NOTE = (
    "Fable ruling A2 (i), boundary case (REVIEW.md 9.1): the fit's last validation mean and the registry's mean of the "
    "same weights are two scorings of one thing. They are computed alike: K = 64 (FitConfig.k_val = registry.EVAL_K), "
    "the same fold points (generator on the data's CUDA device seeded 12345, batches of 128 in storage order: "
    "train.val_nll is the loop of common.test_nll, metric.event_nll at its defaults is common.event_nll), both inside "
    "sfm.metric.deterministic() (the fit's TRAINING runs non-deterministically, its validation does not), and the same "
    "weights (selection 'last': the fit validates the final iterate at step == steps and saves exactly that state; "
    "the EMA row is the saved EMA state, validated the same way as 'val_ema'). Remaining differences: "
    "inference_mode (fit) vs grad mode (registry) - identical numbers - and the float64 mean of the same float32 "
    "array. So they must agree to 1e-6 nat per event; a finite fit value with a non-finite registry score, or a "
    "larger difference, is a failed scoring check (hard stop). Not applicable to the data-free B1d runs (no "
    "validation inside their KL fit): there a non-finite registry score stays a recorded divergence (a candidate's "
    "outcome).")


def check_scoring_settings(cfg) -> None:
    """The comparison of SCORING_CHECK_NOTE is like with like only if the fit validates on the registry's points."""
    pts = registry.live_stream()["points"]
    want = {"K": registry.EVAL_K, "seed": registry.EVAL_SEED, "batch": registry.EVAL_BATCH, "fold": "iid"}
    have = {"K": cfg.k_val, "seed": cfg.val_seed, "batch": cfg.val_batch, "fold": "iid"}   # train.val_nll: iid
    live = {k: pts.get(k) for k in want}
    if have != want or live != want:
        hard_stop("scoring", f"the fit validates with {have}, the registry scores with {want} (live stream {live}): "
                  "the two scorings would not be of the same points; the scoring-consistency check cannot be made")


def scoring_check(name: str, fit_value, reg_value=None, reg_error: Optional[str] = None) -> dict:
    """Compare the fit's last validation mean with the registry's mean of the same weights (SCORING_CHECK_NOTE).
    Returns the record; raises HardStop("scoring") on any disagreement. Both non-finite: the run's (or EMA stream's)
    own outcome, returned with consistent=None for the caller to record as a divergence."""
    fin = fit_value is not None and math.isfinite(float(fit_value))
    if reg_error is not None:
        if fin:
            hard_stop("scoring", f"{name}: the fit's last validation mean {float(fit_value):.6f} is finite, the "
                      f"registry finds non-finite per-event NLLs for the same weights ({reg_error})")
        return {"fit_last_val": None, "registry_val_mean": None, "consistent": None,
                "note": "non-finite in the fit and in the registry: the run's own outcome (recorded divergence)"}
    if not fin:
        hard_stop("scoring", f"{name}: the registry's mean {reg_value} is finite, the fit's last validation mean of the "
                  f"same weights is {fit_value}")
    d = abs(float(fit_value) - float(reg_value))
    if d > SCORING_TOL:
        hard_stop("scoring", f"{name}: the fit's last validation mean {float(fit_value):.9f} and the registry's mean "
                  f"{float(reg_value):.9f} of the same weights differ by {d:.3e} > {SCORING_TOL:g}")
    return {"fit_last_val": float(fit_value), "registry_val_mean": float(reg_value), "abs_diff": d,
            "tolerance": SCORING_TOL, "consistent": True}


KL_LINE = "warm start step"


def kl_record(lines: list, init_info) -> Optional[dict]:
    """I4 option (alpha): the KL trajectory of a warm start / distillation as logged by models.field.kl_warm_start
    (a line every steps // 5 steps, 3 significant digits) plus the initial and final KL of the factory's init_info.
    Copied into init.json, model.pt (data-free runs) and done.json; read by the report."""
    import re
    traj = []
    for ln in lines or []:
        m = re.search(r"warm start step\s+(\d+)\s+KL\s+(\S+)", ln)
        if m:
            try:
                traj.append({"step": int(m.group(1)), "kl": float(m.group(2))})
            except ValueError:
                traj.append({"step": int(m.group(1)), "kl": None})
    kw = (init_info or {}).get("kl_warm_start") if isinstance(init_info, dict) else None
    if not traj and not kw and not lines:
        return None
    return {"initial": (kw or {}).get("kl_initial"), "final": (kw or {}).get("kl_final"), "steps": (kw or {}).get("steps"),
            "trajectory": traj, "log_lines": list(lines or [])}


def latest_init(s: Spec) -> Optional[dict]:
    """The init record of the build whose weights the run carries: the last of init.json, init_rebuild<k>.json."""
    files = [s.out_dir / "init.json"] + sorted(s.out_dir.glob("init_rebuild*.json"),
                                               key=lambda p: int(p.stem.replace("init_rebuild", "") or 0))
    files = [f for f in files if f.exists()]
    return read_json(files[-1]) if files else None


def _inject(s: Spec, steps: int):
    names = [x for x in os.environ.get("SFM_M2B_INJECT_DIVERGENCE", "").split(",") if x]
    if s.name not in names:
        return None
    if not CTX.smoke:
        raise SystemExit("SFM_M2B_INJECT_DIVERGENCE: smoke mode only")
    at = max(1, steps - 1)

    def cb(step, model):
        if step >= at:
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(float("nan"))
            log(f"[{s.name}] SMOKE INJECTION: parameters set to NaN at step {step}")
    return cb


# ----------------------------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------------------------
def fit_config(s: Spec) -> train.FitConfig:
    steps = max(4, s.steps // SMOKE_DIV) if CTX.smoke else s.steps
    every = max(1, steps // 4) if CTX.smoke else s.eval_every
    return train.FitConfig(steps=int(steps), lr=float(s.lr), lr_end=float(s.lr_end), batch=int(s.batch),
                           k_train=int(s.k_train), fold=s.fold, k_val=64, eval_every=int(every),
                           selection=s.selection, ema_decay=s.ema_decay, smooth=float(s.smooth),
                           weight_decay=float(s.weight_decay), seed=int(s.seed), name=s.name, out_dir=str(s.out_dir),
                           ckpt_every=int(every), clip=5.0, val_seed=12345, val_batch=128, fold_rng="dedicated",
                           deterministic=False).validate()


def _comparator_mean(s: Spec) -> list:
    if not s.comparator or s.comparator == "mlp_std":
        return []
    r = row(s.comparator)
    return [r["val_mean"]] if r else []


def run_spec(s: Spec) -> dict:
    """Execute, skip or resume one run; returns its done.json, or a diverged / blocked record."""
    d = done_marker(s.name)
    if d is not None:
        if d.get("recipe_hash") != s.recipe_hash():
            hard_stop("reference", f"{s.name}: the finished run carries recipe {d.get('recipe_hash')}, the pinned "
                      f"recipe is {s.recipe_hash()} (a finished run is never re-run or re-interpreted)")
        return d
    dv = diverged_marker(s.name)
    if dv is not None:
        return dv
    if s.name in BATCH1_NAMES:
        hard_stop("reference", f"{s.name}: a batch-1 run name; batch-1 runs are never re-run")
    if CTX.smoke and s.kwargs.get("template") in ("full_box", "back_corrected"):
        from sfm.models import dist_factory
        if not (dist_factory.template_dir() / "templates.json").exists():
            record_decision("skipped_in_smoke", s.name, "needs the Track A full_box template of this machine "
                            "(run_m2a.py templates); exercised only on the H200, where it exists")
            log(f"[{s.name}] skipped in the smoke mode (no Track A template on this machine)")
            return {"status": "skipped_in_smoke", "run": s.name}
    kw = resolve(s)
    for key in ("start", "target", "run"):                              # blocked by a diverged source of this list
        ref = kw.get(key)
        if isinstance(ref, str) and ref.startswith("out:") and f"/runs/{CTX.stage}/" in ref:
            src = _ref_name(ref)[1]
            if diverged_marker(src) is not None or src in (decisions().get("blocked") or {}):
                record_decision("blocked", s.name, {"blocked_by": src})
                log(f"[{s.name}] BLOCKED by {src}")
                return {"status": "blocked", "run": s.name, "blocked_by": src}
    sets, extras = data_cpu()

    def job():
        t0 = time.time()
        dev = common.DEVICE
        tr_full, va = sets["train"].to(dev), sets["val"].to(dev)
        tr, subset = train_subset(tr_full, s.train_fraction, s.seed)
        val = registry.ValidationSet(va, extras["val"]["event_number"])
        paths.check_output_path(s.out_dir)
        s.out_dir.mkdir(parents=True, exist_ok=True)
        resuming = (s.out_dir / train.CHECKPOINT).exists() or (s.out_dir / train.MODEL_FILE).exists()
        sources = verify_sources(s, kw)                                  # before the factory reads any weights
        train.seed_everything(s.seed)
        kl_lines = []

        def build_log(msg):                                              # I4 (alpha): keep the KL lines of the build
            if " KL " in str(msg):
                kl_lines.append(str(msg).strip())
            log(msg)
        ctx = models.BuildContext(device=dev, train=tr_full, seed=s.seed, log=build_log, skip_init=resuming,
                                  results_dir=paths.REPO_RESULTS)
        try:
            model = models.build(s.factory, ctx, **kw)
        except FloatingPointError as err:                                # non-finite KL in a warm start / distillation
            return record_divergence(s, [f"non-finite KL in the factory's warm start: {err}"],
                                     extra={"kl": kl_record(kl_lines, None)})
        init_file = s.out_dir / "init.json"
        rec = {"spec": asdict(s), "recipe_hash": s.recipe_hash(), "build_kwargs": kw,
               "init_info": getattr(model, "init_info", None), "sources_verified": sources,
               "train_subset": subset, "n_params": models.count_params(model),
               "weights_sha256_at_start": provenance.weights_sha256(model.state_dict()),
               "kl": kl_record(kl_lines, getattr(model, "init_info", None)) if not resuming else None,
               "machine_id": lib.MACHINE.id, "batch": CTX.batch}
        if not init_file.exists():
            write_json(rec, init_file)
        elif not resuming:                  # killed before the first checkpoint: rebuilt; the first record is kept
            k = 1
            while (s.out_dir / f"init_rebuild{k}.json").exists():
                k += 1
            write_json({**rec, "rebuilt_after_restart": True}, s.out_dir / f"init_rebuild{k}.json")
        meta_model = {"factory": s.factory, "kwargs": kw}
        if s.data_free:                                                  # B1d: the KL fit was the training
            mf = s.out_dir / train.MODEL_FILE
            if not mf.exists():                                          # this build's weights, this build's record
                train.atomic_save({"state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                                   "meta": {"model": meta_model, "spec": asdict(s), "stage": CTX.stage},
                                   "machine_tag": lib.MACHINE.to_dict(), "init_info": rec["init_info"],
                                   "kl": rec["kl"], "saved_utc": datetime.now(timezone.utc).isoformat()}, mf)
                kl = rec["kl"]
            else:
                model = models.load_run(f"out:{os.path.relpath(s.out_dir, paths.OUT_RESULTS)}", ctx)
                kl = torch.load(mf, map_location="cpu", weights_only=False).get("kl")
            res, history, warn = None, [], []
            model_path = mf
        else:
            cfg = fit_config(s)
            check_scoring_settings(cfg)
            res = train.fit(model, tr, va, cfg, meta=_jsonable({"model": meta_model, "spec": asdict(s),
                                                                "stage": CTX.stage, "sources": sources,
                                                                "train_subset": subset}),
                            log=log, callback=_inject(s, cfg.steps))
            crit, warn = divergence_check(res.history, _comparator_mean(s))
            if crit:
                return record_divergence(s, crit, res)
            history, model_path = res.history, res.model_path
            kl = (latest_init(s) or {}).get("kl")                         # the build the fit started from
        meta = {"k": models.count_params(model), "seed": s.seed, "config": {"spec": asdict(s), "build_kwargs": kw,
                                                                            "fit": (fit_config(s).result_dict()
                                                                                    if not s.data_free else None)},
                "config_hash": s.recipe_hash(), "checkpoint": str(model_path), "rung": s.rung, "cls": s.cls,
                "comparator": s.comparator, "sources": sources, "train_subset": subset, "smoke": CTX.smoke,
                "trained_on": {"machine_id": lib.MACHINE.id, "source": "trained in this job",
                               "init_from": sources},
                "notes": s.notes or None, "warnings": warn or None}
        fit_last = history[-1].get("val") if history else None
        r = row(s.name, ckpt=model_path)
        if r is None:
            try:
                r = registry.evaluate(model, s.name, CTX.stage, val, meta=meta)
            except FloatingPointError as err:
                if not s.data_free:                  # A2 (i) boundary case: a finite fit value -> hard stop
                    scoring_check(s.name, fit_last, reg_error=str(err))
                return record_divergence(s, ["non-finite per-event validation NLL at registration"], res,
                                         eval_error=str(err), extra={"kl": kl})
        checks = {"note": SCORING_CHECK_NOTE,
                  "main": (scoring_check(s.name, fit_last, r["val_mean"]) if not s.data_free
                           else "not applicable: data-free run (no validation inside the KL fit)")}
        ema = None
        if s.ema_row and not s.data_free:
            obj = torch.load(model_path, map_location=dev, weights_only=False)
            if obj.get("ema_state") is not None:
                m_ema = models.build(s.factory, models.BuildContext(device=dev, train=tr_full, seed=s.seed, log=log,
                                                                    skip_init=True, results_dir=paths.REPO_RESULTS),
                                     **kw)
                m_ema.load_state_dict(obj["ema_state"])
                fit_last_ema = history[-1].get("val_ema") if history else None
                ema = row(f"{s.name}_ema", ckpt=model_path)
                if ema is None:
                    try:
                        ema = registry.evaluate(m_ema, f"{s.name}_ema", CTX.stage, val,
                                                meta={**meta, "rung": "B3", "cls": "exploratory", "comparator": s.name,
                                                      "notes": f"EMA stream (decay {s.ema_decay}) of {s.name} (B3)"})
                    except FloatingPointError as err:
                        # a finite val_ema in the fit: failed scoring check (hard stop, Fable ruling 9.1); non-finite
                        # there too: the EMA stream's own outcome, recorded (orchestrator ruling A2 (i))
                        checks["ema"] = scoring_check(f"{s.name}_ema", fit_last_ema, reg_error=str(err))
                        warn = list(warn) + [f"EMA row {s.name}_ema: non-finite per-event NLL ({err}); recorded as "
                                             "diverged, not registered"]
                        record_decision("diverged", f"{s.name}_ema", {"criteria": ["non-finite EMA score"],
                                                                      "status": "diverged", "cls": "exploratory"})
                        ema = None
                if ema is not None:
                    checks["ema"] = scoring_check(f"{s.name}_ema", fit_last_ema, ema["val_mean"])
        log(f"[{s.name}] registered: val {r['val_mean']:.6f}  vs mlp {r['vs_mlp']['mean'] * 1e4:+.2f} +- "
            f"{r['vs_mlp']['se'] * 1e4:.2f}e-4" + (f"; EMA {ema['val_mean']:.6f}" if ema else ""))
        done = {"name": s.name, "stage": CTX.stage, "smoke": CTX.smoke, "rung": s.rung, "cls": s.cls,
                "spec": asdict(s), "recipe_hash": s.recipe_hash(), "build_kwargs": kw,
                "fit": res.summary() if res is not None else None, "history": history, "warnings": warn,
                "val_mean": r["val_mean"], "k": r["k"], "checkpoint_sha256": r.get("checkpoint_sha256"),
                "weights_sha256": r.get("weights_sha256"), "val_nll_sha256": r.get("val_nll_sha256"),
                "ema_row": ({"name": ema["name"], "val_mean": ema["val_mean"]} if ema else None),
                "scoring_check": checks, "kl": kl,
                "sources_verified": sources, "train_subset": subset, "job_seconds": time.time() - t0,
                "machine_tag": lib.MACHINE.to_dict(), "stream_id": lib.live_stream_id(), "batch": CTX.batch,
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_json(done, s.out_dir / "done.json")
        return done

    out = lib.run_on_gpu(f"m2b list {CTX.stage}/{s.name}", job)
    lib.free_cuda()
    return out


# ----------------------------------------------------------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------------------------------------------------------
def phase_verify() -> dict:
    out = {"reference_set": None, "given": {}, "templates": {}}
    rs = registry.reference_set(need_guards=True)
    out["reference_set"] = {"dir": paths.portable(rs.dir), "parity": rs.manifest.get("parity", {}).get("verdict")}
    for k, names in GIVEN.items():
        for n in names:
            r = given_row(n)
            out["given"][n] = bool(r)
            if r is None and not CTX.smoke:
                hard_stop("reference", f"verify: batch-1 run {n} ({k}) is not registered on the live stream")
    out["mlp_std"] = verify_sources(Spec("verify", "-", "-", "field_mlp_std", {}), {})
    for t in ("front", "stored") + (() if CTX.smoke else ("full_box",)):
        out["templates"][t] = verify_template(t)
    check_pins(build_list())
    out["recipes_pinned"] = True
    record_decision("verify", "inputs", out)
    log(f"verify: reference set {out['reference_set']}; batch-1 runs present: "
        f"{sum(out['given'].values())}/{len(out['given'])}; templates {sorted(out['templates'])}; recipes pinned")
    return out


def smoke_prep() -> None:
    """Smoke only: the B1 ensemble stand-in E5_smoke (uniform ensemble of the registered stage-2 members)."""
    if not CTX.smoke or done_marker("E5_smoke") is not None:
        return
    names = registered_members()
    members = [lib.member(n) for n in names]
    for n in names:
        f = models._run_dir(lib.member(n), models.BuildContext()) / "model.pt"
        if given_row(n) is None or registry.registered_row(REAL_STAGE, n, stream_id=lib.live_stream_id(),
                                                           ckpt=f) is None:
            hard_stop("reference", f"smoke: member {n} has no row with its file on the live stream")

    def job():
        val = registry.load_validation()
        ens = models.build("ensemble", models.BuildContext(device=common.DEVICE), members=members).eval()
        d = out_dir() / "E5_smoke"
        paths.check_output_path(d)
        d.mkdir(parents=True, exist_ok=True)
        mf = d / "model.pt"
        if not mf.exists():
            train.atomic_save({"state": {k: v.detach().clone() for k, v in ens.state_dict().items()},
                               "meta": {"model": {"factory": "ensemble", "kwargs": {"members": members}}},
                               "machine_tag": lib.MACHINE.to_dict()}, mf)
        r = row("E5_smoke", ckpt=mf) or registry.evaluate(ens, "E5_smoke", CTX.stage, val,
                                                          meta={"checkpoint": str(mf), "notes": "smoke stand-in for "
                                                                "E5_s2 (B1d target)", "trained_on": "ensemble"})
        write_json({"name": "E5_smoke", "val_mean": r["val_mean"], "machine_tag": lib.MACHINE.to_dict()},
                   d / "done.json")
    lib.run_on_gpu("m2b list smoke ensemble", job)
    lib.free_cuda()


def run_rungs(rungs) -> list:
    out = []
    for s in build_list():
        if s.rung in rungs:
            out.append(run_spec(s))
    return out


def phase_converge() -> dict:
    """AUDIT 4.0 convergence rule for the primaries with a fit history (B1d is data-free: no validation trajectory)."""
    sm = spec_map()
    res = {}
    win = max(1, CONVERGENCE_WINDOW // SMOKE_DIV) if CTX.smoke else CONVERGENCE_WINDOW
    for s in sm.values():
        if s.cls != "primary" or s.data_free:
            continue
        d = done_marker(s.name)
        if d is None:
            res[s.name] = {"available": False}
            continue
        h = d.get("history") or []
        last = h[-1] if h else None
        prev = next((x for x in reversed(h) if x["step"] <= last["step"] - win), None) if last else None
        if last is None or prev is None:
            res[s.name] = {"available": False, "reason": "history shorter than the window"}
            continue
        move = abs(float(last["val"]) - float(prev["val"]))
        forced = CTX.smoke and s.name == SMOKE_FORCED_RERUN          # smoke only: exercise the rerun path
        conv = move < CONVERGED_MOVE and not forced
        res[s.name] = {"available": True, "move": move, "window_steps": last["step"] - prev["step"], "converged": conv,
                       **({"forced_in_the_smoke_mode": True} if forced else {})}
        if not conv:
            for t in (s, sm.get(s.comparator)) if s.comparator in sm else (s,):
                x = x2_spec(t)
                pins = pinned().get("x2_recipes", {})
                if pins.get(x.name) != x.recipe_hash():
                    hard_stop("recipe pin", f"{x.name}: rerun recipe not pinned")
                res[s.name].setdefault("reruns", []).append(x.name)
                run_spec(x)
    write_json({**decisions(), "convergence": res}, decisions_file())
    return res


def phase_guards() -> dict:
    """Guards for every run meeting the promotion parts (a)-(c) on a seed, and its other seed (AUDIT 4.2 note)."""
    import run_m2b_list_report as rep
    specs = rep.all_specs()                                            # the list + registered x2 reruns (B2)
    cands = rep.guard_candidates(sm=specs)
    if CTX.smoke and not cands:                                        # exercise the path in the smoke mode, on
        cands = [n for n in rep.records(specs)["B2b"] if row(n)]       # the runs of record of B2b (reruns if any)
    out = {}
    for n in cands:
        s = specs.get(n)
        if s is None or row(n) is None:
            continue
        try:
            out[n] = guard_one(s)
        except HardStop:
            raise
        except Exception as err:                                          # a guard that cannot be evaluated
            hard_stop("guard", f"guard evaluation of {n} failed: {type(err).__name__}: {err}")
    write_json({**decisions(), "guards": out}, decisions_file())
    return out


def guard_one(s: Spec) -> dict:
    r0 = row(s.name, need_guards=True)
    sdir_marker = s.out_dir / "g_region.json"
    if r0 is not None and sdir_marker.exists():
        return {"row": "exists", "g_region": "exists"}
    sets, extras = data_cpu()

    def job():
        dev = common.DEVICE
        va = sets["val"].to(dev)
        val = registry.ValidationSet(va, extras["val"]["event_number"])
        model = models.load_run(f"out:{os.path.relpath(s.out_dir, paths.OUT_RESULTS)}",
                                models.BuildContext(device=dev, train=sets["train"], skip_init=True,
                                                    results_dir=paths.REPO_RESULTS))
        base = row(s.name)
        r = r0 or registry.evaluate(model, s.name, CTX.stage, val, guards=True,
                                    meta={"k": base["k"], "seed": s.seed, "config_hash": s.recipe_hash(),
                                          "checkpoint": str(s.out_dir / train.MODEL_FILE), "rung": s.rung,
                                          "cls": s.cls, "trained_on": base.get("trained_on"),
                                          "notes": "guard evaluation (AUDIT 4.2 note)"})
        g = g_region(s.name, model, val, sets["val"])
        return {"row_with_guards": r["timestamp"], "g_region": g}
    out = lib.run_on_gpu(f"m2b list guards {s.name}", job)
    lib.free_cuda()
    return out


def g_region(name: str, model, val, val_cpu) -> dict:
    """G-region arrays (hit term by hit-centre class, normalisation) with the Track A implementation
    (scripts/run_m2a.py g_region_compute, imported, not duplicated), saved as tagged arrays next to the run; the
    reference arrays of mlp_std are computed once the same way."""
    from run_m2a import g_region_compute                                # read-only use of the Track A code
    st = registry.live_stream()
    pts0 = st["points"]

    def save(dirname, label, comp):
        d = out_dir() / dirname / "g_region"
        out = {}
        for cls, a in comp["arrays"].items():
            pts = provenance.points_tag(grid=pts0["grid"], K=pts0["K"], split_fingerprint=pts0["split_fingerprint"],
                                        seed=pts0["seed"], batch=pts0["batch"], subset=f"g_region:{cls}")
            ta = provenance.save_tagged(d / f"g_region_{cls}.npy", a, pts, st["machine"], label=f"{label}:{cls}")
            out[cls] = {"path": paths.portable(ta.path), "sha256": ta.sha256}
        write_json({"name": label, "arrays": out, "identity": comp.get("identity")}, d / "g_region.json")
        return out
    def checked(comp, registered, label):
        """The decomposition must add up to the per-event NLL, and that NLL must equal the registered array
        (tolerance check: every event within 1e-4, mean within 1e-6); otherwise a hard stop (guard check). The
        residuals are stored in the marker (the report prints them, finding C6)."""
        a = comp["arrays"]
        recon = a["flagged"] + a["inner"] + a["normalisation"]
        resid = float(np.abs(recon - comp["parts_nll"].astype(np.float64)).max())
        tc = provenance.tolerance_check(comp["parts_nll"], registered.values, top=3)
        if resid > 1e-4 or not tc["strict_pass"]:
            hard_stop("guard", f"G-region of {label}: decomposition residual {resid:.2e}, against the registered array "
                      f"max {tc['max_abs']:.2e} / mean {tc['mean_diff']:.2e}")
        comp["identity"] = {"decomposition_residual_max": resid, "vs_registered_max_abs": float(tc["max_abs"]),
                            "vs_registered_mean_diff": float(tc["mean_diff"]), "tolerance": "1e-4 per event"}
        return comp

    ref_marker = out_dir() / "_mlp_std_reference" / "g_region" / "g_region.json"
    if not ref_marker.exists():
        mref = registry.reference_set().bench["mlp"]
        save("_mlp_std_reference", "mlp_std",
             checked(g_region_compute(models.mlp_std_start(common.DEVICE).eval(), val.showers, val_cpu), mref,
                     "mlp_std"))
    return save(name, f"{CTX.stage}/{name}",
                checked(g_region_compute(model, val.showers, val_cpu), registry.row_array(row(name)), name))


def phase_report() -> None:
    import run_m2b_list_report as rep
    rep.run(["--smoke"] if CTX.smoke else [])                         # a failed check raises HardStop -> exit 3


def phase_b1d_companions() -> list:
    """I4 option (beta), prepared, not in the default phases: the two 12,000-KL-step companions of B1d."""
    check_companion_pins()
    smoke_prep()
    return [run_spec(s) for s in companion_specs()]


PHASES = {"verify": phase_verify, "b1d": lambda: (smoke_prep(), run_rungs(("B1d",))),
          "b1d_companions": phase_b1d_companions,
          "b1f": lambda: run_rungs(("B1f",)), "b2": lambda: run_rungs(("B2a", "B2b", "B2c")),
          "b3s": lambda: run_rungs(("B3s",)), "b4": lambda: run_rungs(("B4f", "B4s", "B4x")),
          "b5": lambda: run_rungs(("B5f",)), "b6": lambda: run_rungs(("B6a", "B6b", "B6c")),
          "bceil": lambda: run_rungs(("B-ceil",)), "converge": phase_converge, "guards": phase_guards,
          "report": phase_report}


def print_list() -> None:
    print("Track B list (results/m1_audit/AUDIT.md section 4.2), derived by build_list(); status on the live stream "
          f"{lib.live_stream_id()} (stage {CTX.stage})\n")
    print("Given (registered by batch 1; never re-run here):")
    for k, names in GIVEN.items():
        print(f"  {k:<10} " + ", ".join(f"{n} [{'registered' if given_row(n) else 'MISSING'}]" for n in names))
    for k, v in CLOSED.items():
        print(f"  {k:<10} {v}")
    print(f"\n{'name':<22} {'rung':<7} {'class':<12} {'seed':>4} {'steps':>6} {'K':>3} {'fold':<10} {'other':<34} "
          f"{'comparator':<12} status")
    total = 0
    for s in build_list():
        other = []
        if s.smooth:
            other.append(f"smooth {s.smooth:g}")
        if s.weight_decay:
            other.append(f"wd {s.weight_decay:g}")
        if s.train_fraction < 1:
            other.append(f"train {int(s.train_fraction * 100)} %")
        if s.lr != 1e-3 or s.lr_end != 1e-4:
            other.append(f"lr {s.lr:g}->{s.lr_end:g}")
        if s.ema_row:
            other.append("EMA row")
        if "start" in s.kwargs:
            other.append(f"from {s.kwargs['start']}")
        if "run" in s.kwargs:
            other.append(f"from {s.kwargs['run'].strip('<>')}")
        for k, lab in (("template", "tpl"), ("hidden", "h"), ("n_layers", "L"), ("kl_steps", "KL"),
                       ("n_knots", "knots")):
            if k in s.kwargs:
                other.append(f"{lab} {s.kwargs[k]}")
        total += s.steps
        print(f"{s.name:<22} {s.rung:<7} {s.cls:<12} {s.seed:>4} {s.steps:>6} {s.k_train:>3} {s.fold:<10} "
              f"{', '.join(other):<34} {str(s.comparator):<12} {status(s)}")
    print(f"\n{len(build_list())} runs; {total:,} training steps (+ KL warm starts: B1d 2 x 6000, B6 6 x 3000, B4 4 x 500); "
          "convergence reruns of non-converged primaries come on top (doubled steps, with their comparators)")
    print("\nPrepared, NOT in the default phases (I4 option (beta), with the user; phase b1d_companions only when named):")
    for s in companion_specs():
        print(f"{s.name:<22} {s.rung:<7} {s.cls:<12} {s.seed:>4} {s.steps:>6} {s.k_train:>3} {s.fold:<10} "
              f"{'from ' + s.kwargs['start'] + ', KL ' + str(s.kwargs['kl_steps']):<34} {str(s.comparator):<12} "
              f"{status(s)}")


TOLERATED_GUARD_REASON = "import of a Parquet reader while the guard is active"   # pandas probes pyarrow; tolerated


def check_guard_log() -> None:
    """File-access / deferred-file check: any refusal of THIS process other than the tolerated Parquet-reader
    import is a hard stop (a refused open may have been swallowed by library code)."""
    from sfm import guard
    f = guard.guard_log_path()
    if not f.exists():
        return
    bad = [ln for ln in f.read_text().splitlines()
           if f"\tpid {os.getpid()}\t" in ln and TOLERATED_GUARD_REASON not in ln]
    if bad:
        refused = [ln.split("\t")[3] + " (" + ln.split("\t")[4] + ")" if ln.count("\t") >= 4 else ln for ln in bad]
        hard_stop("file access", f"{len(bad)} refusal(s) by the file-access guard in this job: "
                  + "; ".join(refused[:20]) + (" ..." if len(refused) > 20 else ""))


def _stop_line(kind: str, msg: str) -> None:
    line = (f"HARD STOP (PLAN.md section 23, item 13) - {msg} - no later phase was run; waiting for the user"
            if kind == "hard" else f"STOPPED (resources) - {msg} - re-issue the same command later (resumes)")
    try:
        record_decision("stops", datetime.now(timezone.utc).isoformat(timespec="seconds"), {"kind": kind, "msg": msg})
    except Exception:
        pass
    print(paths.scrub_str(line), flush=True)


def main(argv=None) -> int:
    """Exit codes: 0 finished; 3 HARD STOP (a failed check, waiting for the user); 4 stopped for lack of GPU memory
    (resumable); anything else is an unexpected error (traceback, exit 1)."""
    from sfm import guard
    hard = (HardStop, registry.NoReferenceSet, registry.ReferenceSetNotLicensed, registry.ReferenceSetDisputed,
            registry.NotValidationSet, registry.UntaggedRow, provenance.MixedTags, provenance.UntaggedArray,
            guard.GuardRefused, FileExistsError)
    try:
        return _main(argv)
    except hard as err:
        check = getattr(err, "check", None) or {
            "NoReferenceSet": "reference", "ReferenceSetNotLicensed": "parity", "ReferenceSetDisputed": "reference",
            "NotValidationSet": "scoring", "UntaggedRow": "scoring", "MixedTags": "scoring",
            "UntaggedArray": "scoring", "GuardRefused": "file access", "FileExistsError": "scoring"}.get(
            type(err).__name__, "check")
        _stop_line("hard", f"{check} check failed: {type(err).__name__}: {err}" if not isinstance(err, HardStop)
                   else f"{err}")
        return HARD_STOP_EXIT
    except machine.InsufficientGPUMemory as err:
        _stop_line("resource", str(err))
        return RESOURCE_EXIT


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    configure(args.smoke)
    if args.pin:
        write_pins()
        return 0
    if args.list:
        print_list()
        return 0
    CTX.batch = lib.start(f"run_m2b_list ({CTX.stage})")
    phases = args.phases or list(DEFAULT_PHASES)
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; known: {list(PHASES)}")
    if "verify" not in phases:
        check_pins(build_list())
    check_guard_log()
    for p in [x for x in PHASE_ORDER if x in phases]:
        log(f"=== phase {p}")
        PHASES[p]()
        check_guard_log()
    log(f"=== finished: {phases}")
    lib.finish(CTX.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
