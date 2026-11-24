"""sfm.registry - validation evaluation of finished models; the ONLY writer of ``results/registry.jsonl``.

    val = registry.load_validation()                                   # the cached validation split, checked
    row = registry.evaluate(model, "b0_seed0", "m2b_mlp", val, meta={"seed": 0, "config": cfg, "checkpoint": path})
    tab = registry.table()                                             # pandas.DataFrame, latest row per (stage, name)
    sel = registry.select_1se(stage="m2b_mlp")                         # sfm.stats.one_se_select over chosen rows

What ``evaluate`` does
    1. Refuses anything that is not the validation split of the cache (``check_validation``): the number of
       events, the event numbers *in order* (equal to ``results/m0_parity/val_event_number.npy``, the order of the
       pairing partners) and a SHA-256 fingerprint of every tensor of the ``Showers`` object (hits, weights, cell
       box, per-event p / ln p / zb / R / bin / evis) must equal those of the cached validation split. A test set,
       a subset, a reordering or modified tensors are rejected (``NotValidationSet``) before any model is run.
    2. Per-event validation NLL with the UNMODIFIED ``common.test_nll(model, val)`` (K = 64, CUDA generator seed
       12345, batches of 128) called inside ``sfm.metric.deterministic()``, so the stored array is bit-reproducible on
       this machine (PLAN.md section 13). Saved as ``results/<stage>/<name>/val_nll.npy``; an existing file with
       different content is never overwritten silently (``replace=True`` is required, and the old row keeps the
       old SHA-256, so the trail stays visible).
    3. Paired difference against both benchmarks, ``sfm.stats.paired(val_nll, bench)`` with the Gate 0 arrays
       ``results/m0_parity/{mlp,formula}_val_nll.npy`` (numbers of record, this machine's stream).
    4. Optional guard columns (``guards=True``; plan section 6 with the section-13 refinements):
         G-K      ``sfm.metric.test_nll`` at K = 1024 (chunked, same seed and batches), paired against the
                  benchmarks at K = 1024 on the same points, and ``stats.guard_persistence`` (K64 gain vs K1024 gain)
         G-bulk   bulk-restricted NLL (``metric.Bulk(short="raise")``, z' < 1150 mm, last ring dropped) on the
                  stratum ``zb >= 1221 mm``, paired against the benchmarks on the same stratum and points, with
                  the report-metric gain on the same events and ``stats.guard_persistence``; the complementary
                  stratum ``zb < 1221 mm`` is reported as its own row (report-metric gain, share of the total gain)
         G-mass   ``sfm.metric.mass_monitors`` averaged over the validation events (mass beyond zb, last ring,
                  first / last z-bin, outside the bulk region, in cells with stored acceptance Abar <= 0.01)
       Benchmark guard arrays are computed once and cached in ``results/registry_refs/`` (with their spec).
       Guard numbers live in their own columns and never share a column or a delta with the report metric.
    5. One JSON line appended under an exclusive ``flock``: name, stage, k (free constants), C (operator
       complexity, optional), validation mean, paired deltas +- SE vs both benchmarks, guards, seed, config hash,
       checkpoint path and SHA-256, git commit (and whether the tree was dirty), timestamp, the array's SHA-256.

Nothing here opens the test split; ``check_validation`` makes it impossible to score one through this module.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from . import data as sfm_data
from . import metric, paths, stats

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common as C  # noqa: E402

__all__ = ["NotValidationSet", "ValidationReference", "ValidationSet", "RegistryEnv", "fingerprint",
           "validation_reference", "load_validation", "check_validation", "evaluate", "read_registry", "table",
           "select_1se", "latest", "BENCHMARKS", "ZB_STRATUM_MM", "K_GUARD"]

EXPECTED_VAL_EVENTS = 14950
BENCHMARKS = ("mlp", "formula")
ZB_STRATUM_MM = 1221.0          # smallest zb of a front-layer start (Gate 0, amendment A4)
K_GUARD = 1024
METRIC_OF_RECORD = "common.test_nll (unmodified), K=64, CUDA generator seed 12345, batch 128, inside " \
                   "sfm.metric.deterministic()"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class NotValidationSet(PermissionError):
    """The data passed to the registry is not the cached validation split."""


# ----------------------------------------------------------------------------------------------------------------------
# Locations (overridable for tests)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class RegistryEnv:
    results_dir: Path = field(default_factory=lambda: paths.RESULTS_DIR)
    registry_path: Optional[Path] = None              # default <results>/registry.jsonl
    bench_files: Optional[Mapping[str, Path]] = None  # default <results>/m0_parity/{mlp,formula}_val_nll.npy
    refs_dir: Optional[Path] = None                   # default <results>/registry_refs

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.registry_path = Path(self.registry_path) if self.registry_path else self.results_dir / "registry.jsonl"
        if self.bench_files is None:
            m0 = self.results_dir / "m0_parity"
            self.bench_files = {b: m0 / f"{b}_val_nll.npy" for b in BENCHMARKS}
        self.bench_files = {k: Path(v) for k, v in self.bench_files.items()}
        self.refs_dir = Path(self.refs_dir) if self.refs_dir else self.results_dir / "registry_refs"


# ----------------------------------------------------------------------------------------------------------------------
# The validation set and its identity
# ----------------------------------------------------------------------------------------------------------------------
def fingerprint(showers) -> str:
    """SHA-256 over every tensor of a ``common.Showers`` object (dtype, shape and bytes, in a fixed order)."""
    h = hashlib.sha256()

    def add(name, t):
        t = t.detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())

    for name in ("off", "xyz", "w", "box"):
        add(name, getattr(showers, name))
    for k in sorted(showers.ev):
        add(f"ev.{k}", showers.ev[k])
    return h.hexdigest()


@dataclass(frozen=True)
class ValidationReference:
    event_number: np.ndarray
    fingerprint: str
    n_events: int
    source: str = ""


@dataclass
class ValidationSet:
    showers: object                    # common.Showers on common.DEVICE
    event_number: np.ndarray


def _val_fingerprint_file(cache_dir=None) -> Path:
    return Path(cache_dir or paths.CACHE_DIR) / f"{sfm_data.CACHE_STEM}_val_fingerprint.json"


def validation_reference(cache_dir=None, results_dir=None) -> ValidationReference:
    """Identity of the cached validation split: event numbers in the order of the Gate 0 pairing partners
    (``results/m0_parity/val_event_number.npy``) and the fingerprint of the validation tensors in the train/val
    cache file. The fingerprint is computed once from the cache (val split only, on the CPU) and remembered in
    ``cache/pion_s1_val_fingerprint.json``, keyed by the cache file's SHA-256 recorded in the cache metadata."""
    results_dir = Path(results_dir or paths.RESULTS_DIR)
    ev_file = results_dir / "m0_parity" / "val_event_number.npy"
    event_number = np.load(ev_file)
    meta = sfm_data.load_meta(cache_dir)
    key = meta["files"]["trainval"]["sha256"]
    fp_file = _val_fingerprint_file(cache_dir)
    fp = None
    if fp_file.exists():
        stored = json.loads(fp_file.read_text())
        if stored.get("trainval_sha256") == key:
            fp = stored["fingerprint"]
    if fp is None:
        sets, extras = sfm_data.load_sets(("val",), device="cpu", cache_dir=cache_dir)
        if not np.array_equal(extras["val"]["event_number"], event_number):
            raise RuntimeError("cached validation event numbers differ from results/m0_parity/val_event_number.npy")
        fp = fingerprint(sets["val"])
        tmp = fp_file.with_name(fp_file.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"trainval_sha256": key, "fingerprint": fp, "n_events": len(sets["val"]),
                                   "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
        os.replace(tmp, fp_file)
    if len(event_number) != meta["counts"]["val"]:
        raise RuntimeError("validation event-number file and cache metadata disagree on the number of events")
    return ValidationReference(event_number=event_number, fingerprint=fp, n_events=int(len(event_number)),
                               source=f"cache {meta['files']['trainval']['name']} ({key[:12]})")


def load_validation(device=None, cache_dir=None) -> ValidationSet:
    """The validation split of the cache (train/val file only; the test file is never opened)."""
    sets, extras = sfm_data.load_sets(("val",), device=device or C.DEVICE, cache_dir=cache_dir)
    return ValidationSet(sets["val"], extras["val"]["event_number"])


def _unpack_val(val_set):
    if isinstance(val_set, ValidationSet):
        return val_set.showers, np.asarray(val_set.event_number)
    if isinstance(val_set, tuple) and len(val_set) == 2:
        sh, ex = val_set
        en = ex["event_number"] if isinstance(ex, Mapping) else ex
        return sh, np.asarray(en)
    raise NotValidationSet("evaluate() needs a ValidationSet (registry.load_validation()) or a (Showers, extras) pair "
                           "with the event numbers")


def check_validation(val_set, reference: Optional[ValidationReference] = None) -> object:
    """Raise ``NotValidationSet`` unless ``val_set`` is the cached validation split, exactly. Returns the Showers."""
    sh, en = _unpack_val(val_set)
    ref = reference or validation_reference()
    if len(sh) != ref.n_events or len(en) != ref.n_events:
        raise NotValidationSet(f"not the validation split: {len(sh)} events / {len(en)} event numbers, "
                               f"expected {ref.n_events}")
    if en.shape != ref.event_number.shape or not np.array_equal(en, ref.event_number):
        raise NotValidationSet("not the validation split: event numbers differ from the validation split (in order)")
    if fingerprint(sh) != ref.fingerprint:
        raise NotValidationSet("not the validation split: the tensors differ from the cached validation split")
    return sh


# ----------------------------------------------------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def _git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(paths.REPO_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except Exception:
        return {"commit": None, "dirty": None}


def _config_hash(cfg) -> Optional[str]:
    if cfg is None:
        return None
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _count_params(model) -> int:
    return int(model.n_params()) if hasattr(model, "n_params") else sum(p.numel() for p in model.parameters())


def _paired_dict(p: stats.Paired) -> dict:
    return {"mean": p.mean, "se": p.se, "z": p.z, "n": p.n}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _append(row: dict, path: Path) -> None:
    """Append one JSON line under an exclusive flock (the only write access to the registry)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=_json_default, allow_nan=True)
    if "\n" in line:
        raise ValueError("registry row serialises to more than one line")
    with open(path, "a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:                                   # a crashed writer may have left a partial line
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write(line.encode() + b"\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _save_array(path: Path, a: np.ndarray, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path)
        if old.shape == a.shape and old.dtype == a.dtype and np.array_equal(old, a):
            return
        if not replace:
            raise FileExistsError(f"{path} exists with different content; pass replace=True to overwrite it "
                                  "(the earlier registry row keeps the earlier SHA-256)")
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npy")
    np.save(tmp, a)
    os.replace(tmp, path)


def _check_name(s: str, what: str) -> str:
    if not isinstance(s, str) or not _NAME_RE.match(s) or ".." in s:
        raise ValueError(f"{what} must match {_NAME_RE.pattern} (no path separators), got {s!r}")
    return s


# ----------------------------------------------------------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------------------------------------------------------
def _guard_spec(n_events: int, ref_fp: str) -> dict:
    return {"K": K_GUARD, "chunk": 64, "seed": 12345, "batch": 128, "bulk": {"z_cut": 1150.0, "drop_rings": 1,
            "renormalise": True, "short": "raise"}, "stratum": f"zb >= {ZB_STRATUM_MM}", "val_fingerprint": ref_fp,
            "n_events": n_events}


def _guard_arrays(model, val, stratum_idx) -> dict:
    """Per-event guard arrays of one model (same points for every model)."""
    with torch.inference_mode():
        k1024 = metric.test_nll(model, val, K=K_GUARD, chunk=64, deterministic_sum=True)
        sub = val.subset(torch.as_tensor(stratum_idx, device=val.off.device))
        bulk = metric.test_nll(model, sub, bulk=metric.Bulk(short="raise"), deterministic_sum=True)
    return {"k1024": k1024, "bulk_stratum": bulk}


def _mass(model, val, abar=None, batch: int = 128) -> dict:
    masks = {"abar_le_0.01": (abar <= 0.01)} if abar is not None else None
    acc = {}
    model.eval()
    with torch.inference_mode():
        for b in val.batches(batch):
            mon = metric.mass_monitors(model(b.lnp), b, masks=masks)
            for k, v in mon.items():
                acc.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def _bench_guard_arrays(env: RegistryEnv, val, stratum_idx, ref_fp: str, bench_models: Optional[Callable]) -> dict:
    """Benchmark guard arrays, cached in <refs_dir> with their spec; computed once from the benchmark models."""
    spec = _guard_spec(len(val), ref_fp)
    spec_file = env.refs_dir / "guard_spec.json"
    files = {b: {k: env.refs_dir / f"{b}_val_{k}.npy" for k in ("k1024", "bulk_stratum")} for b in BENCHMARKS}
    ok = spec_file.exists() and json.loads(spec_file.read_text()) == spec and all(
        f.exists() for d in files.values() for f in d.values())
    if not ok:
        models = (bench_models or _default_bench_models)()
        env.refs_dir.mkdir(parents=True, exist_ok=True)
        for b in BENCHMARKS:
            arr = _guard_arrays(models[b].eval(), val, stratum_idx)
            for k, f in files[b].items():
                np.save(f, arr[k])
        spec_file.write_text(json.dumps(spec, indent=1) + "\n")
    return {b: {k: np.load(f) for k, f in files[b].items()} for b in BENCHMARKS}


def _default_bench_models() -> dict:
    """mlp_std and the component formula through the read-only adapter (the formula needs the training ln p range)."""
    from . import legacy
    sets, _ = sfm_data.load_sets(("train",), device="cpu")
    return {"mlp": legacy.load_mlp_std(), "formula": legacy.load_component_formula(sets["train"])}


def _guards(model, val, nll: np.ndarray, bench: dict, env: RegistryEnv, ref_fp: str, run_dir: Path,
            bench_models: Optional[Callable], abar) -> dict:
    zb = val.ev["zb"].detach().cpu().numpy()
    in_stratum = zb >= ZB_STRATUM_MM
    idx = np.flatnonzero(in_stratum)
    mine = _guard_arrays(model, val, idx)
    ref = _bench_guard_arrays(env, val, idx, ref_fp, bench_models)
    np.save(run_dir / "val_nll_k1024.npy", mine["k1024"])
    np.save(run_dir / "val_bulk_zb1221.npy", mine["bulk_stratum"])
    mass = _mass(model, val, abar)
    np.savez(run_dir / "val_mass_monitors.npz", **{k.replace(":", "_"): v for k, v in mass.items()})
    out = {"K": K_GUARD, "stratum": f"zb >= {ZB_STRATUM_MM:g} mm", "n_stratum": int(len(idx)),
           "n_short": int((~in_stratum).sum()), "val_mean_k1024": float(np.mean(mine["k1024"], dtype=np.float64))}
    bulk_ok = np.isfinite(mine["bulk_stratum"])
    out["n_bulk_nan"] = int((~bulk_ok).sum())
    for b in BENCHMARKS:
        main = nll.astype(np.float64) - bench[b].astype(np.float64)
        gk = mine["k1024"].astype(np.float64) - ref[b]["k1024"].astype(np.float64)
        g = {"gk": _paired_dict(stats.paired(mine["k1024"], ref[b]["k1024"]))}
        dk = stats.guard_persistence(main, gk)
        g["gk_persistence"] = {"passed": dk.passed, "checks": dk.checks, "numbers": dk.numbers, "flags": dk.flags}
        okb = bulk_ok & np.isfinite(ref[b]["bulk_stratum"])
        gb = mine["bulk_stratum"][okb].astype(np.float64) - ref[b]["bulk_stratum"][okb].astype(np.float64)
        main_s = main[idx][okb]
        g["bulk"] = _paired_dict(stats.paired(gb, np.zeros_like(gb)))
        g["report_on_stratum"] = _paired_dict(stats.paired(main_s, np.zeros_like(main_s)))
        db = stats.guard_persistence(main_s, gb)
        g["bulk_persistence"] = {"passed": db.passed, "checks": db.checks, "numbers": db.numbers, "flags": db.flags}
        short = main[~in_stratum]
        g["report_on_short"] = (_paired_dict(stats.paired(short, np.zeros_like(short))) if len(short) >= 2 else None)
        tot = float(main.sum())
        g["short_share_of_gain"] = float(short.sum()) / tot if tot != 0 else None
        out[f"vs_{b}"] = g
    out["mass"] = {k: {"mean": float(v.mean()), "max": float(v.max())} for k, v in mass.items()}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------------------------------------------------
_META_KEYS = ("k", "C", "seed", "config", "config_hash", "checkpoint", "notes")


def evaluate(model, name: str, stage: str, val_set, meta: Optional[Mapping] = None, *, guards: bool = False,
             replace: bool = False, env: Optional[RegistryEnv] = None, reference: Optional[ValidationReference] = None,
             bench_models: Optional[Callable] = None, abar: Optional[torch.Tensor] = None) -> dict:
    """Score ``model`` on the validation split, store the per-event array, append one registry row, return it.

    meta (all optional): k (free constants; default ``model.n_params()`` or the parameter count), C (operator
    complexity), seed, config (dict; hashed), config_hash, checkpoint (path; hashed), notes; other keys go to
    ``extra``. ``guards=True`` adds the guard columns (module docstring). ``env``/``reference``/``bench_models`` exist
    for tests; the defaults are the campaign's files.
    """
    env = env or RegistryEnv()
    _check_name(name, "name")
    _check_name(stage, "stage")
    meta = dict(meta or {})
    ref = reference if reference is not None else validation_reference(results_dir=env.results_dir)
    val = check_validation(val_set, ref)
    if val.xyz.device.type != torch.device(C.DEVICE).type:
        raise ValueError(f"the validation set must be on common.DEVICE ({C.DEVICE}) for common.test_nll")
    bench = {}
    for b in BENCHMARKS:
        arr = np.load(env.bench_files[b])
        if arr.shape != (len(val),) or not np.isfinite(arr).all():
            raise ValueError(f"benchmark array {env.bench_files[b]} has shape {arr.shape} or non-finite values")
        bench[b] = arr

    was = model.training
    model.eval()
    with metric.deterministic():
        nll = C.test_nll(model, val)                                     # the metric of record, unmodified
    model.train(was)
    if nll.shape != (len(val),):
        raise RuntimeError(f"common.test_nll returned shape {nll.shape}")
    if not np.isfinite(nll).all():
        raise FloatingPointError(f"{int((~np.isfinite(nll)).sum())} non-finite per-event validation NLL(s)")

    run_dir = env.results_dir / stage / name
    nll_path = run_dir / "val_nll.npy"
    _save_array(nll_path, nll, replace)

    ckpt = meta.get("checkpoint")
    row = {"name": name, "stage": stage,
           "k": int(meta["k"]) if meta.get("k") is not None else _count_params(model),
           "C": float(meta["C"]) if meta.get("C") is not None else None,
           "n_events": int(len(nll)), "val_mean": float(np.mean(nll, dtype=np.float64)),
           "vs_mlp": _paired_dict(stats.paired(nll, bench["mlp"])),
           "vs_formula": _paired_dict(stats.paired(nll, bench["formula"])),
           "guards": None,
           "seed": meta.get("seed"),
           "config_hash": meta.get("config_hash") or _config_hash(meta.get("config")),
           "checkpoint": str(ckpt) if ckpt else None,
           "checkpoint_sha256": _sha256_file(Path(ckpt)) if ckpt else None,
           "val_nll_path": os.path.relpath(nll_path, env.results_dir),
           "val_nll_sha256": _sha256_array(nll),
           "metric": METRIC_OF_RECORD,
           "benchmarks": {b: os.path.relpath(env.bench_files[b], env.results_dir) for b in BENCHMARKS},
           "device": str(C.DEVICE), "torch": torch.__version__,
           "git": _git_state(),
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "notes": meta.get("notes"),
           "extra": {k: v for k, v in meta.items() if k not in _META_KEYS} or None}
    if guards:
        if abar is None:
            from .models.field import acceptance_template
            abar = acceptance_template("stored", val.xyz.device)
        row["guards"] = _guards(model, val, nll, bench, env, ref.fingerprint, run_dir, bench_models, abar)
    _append(row, env.registry_path)
    return row


# ----------------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------------
def read_registry(path: Optional[Path] = None, strict: bool = True) -> list:
    """All rows in file order. A line that is not valid JSON raises (strict) or is skipped."""
    path = Path(path) if path else RegistryEnv().registry_path
    if not path.exists():
        return []
    rows = []
    with open(path, "rb") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            lines = f.read().decode().splitlines()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as err:
            if strict:
                raise ValueError(f"{path}:{i}: invalid registry line ({err})") from err
    return rows


def latest(rows: Sequence[dict]) -> list:
    """The last row of every (stage, name), in order of first appearance."""
    last = {}
    for r in rows:
        last[(r["stage"], r["name"])] = r
    return list(last.values())


def table(path: Optional[Path] = None, stage: Optional[str] = None, latest_only: bool = True):
    """Registry as a flat ``pandas.DataFrame`` (one row per entry; latest per (stage, name) by default)."""
    import pandas as pd
    rows = read_registry(path)
    if latest_only:
        rows = latest(rows)
    if stage is not None:
        rows = [r for r in rows if r["stage"] == stage]
    flat = []
    for r in rows:
        d = {k: r.get(k) for k in ("stage", "name", "k", "C", "n_events", "val_mean", "seed", "config_hash",
                                   "timestamp")}
        for b in BENCHMARKS:
            p = r.get(f"vs_{b}") or {}
            d[f"d_{b}"], d[f"se_{b}"], d[f"z_{b}"] = p.get("mean"), p.get("se"), p.get("z")
        g = r.get("guards")
        if g:
            d["val_mean_k1024"] = g.get("val_mean_k1024")
            for b in BENCHMARKS:
                gb = g.get(f"vs_{b}") or {}
                d[f"gk_{b}"] = (gb.get("gk") or {}).get("mean")
                d[f"bulk_{b}"] = (gb.get("bulk") or {}).get("mean")
        d["git_commit"] = (r.get("git") or {}).get("commit")
        flat.append(d)
    return pd.DataFrame(flat)


def select_1se(names: Optional[Sequence[str]] = None, stage: Optional[str] = None, rows: Optional[Sequence[dict]] = None,
               path: Optional[Path] = None, results_dir: Optional[Path] = None) -> stats.OneSE:
    """``sfm.stats.one_se_select`` over chosen registry entries (latest row per (stage, name)).

    Pick by ``stage`` and/or ``names`` (bare names or "stage/name"), or pass ``rows`` directly. Each candidate's
    per-event array is read from its ``val_nll_path`` and checked against the row's SHA-256; complexity key (k, C)."""
    results_dir = Path(results_dir) if results_dir else (Path(path).parent if path else paths.RESULTS_DIR)
    if rows is None:
        rows = latest(read_registry(path))
        if stage is not None:
            rows = [r for r in rows if r["stage"] == stage]
        if names is not None:
            want = set(names)
            rows = [r for r in rows if r["name"] in want or f"{r['stage']}/{r['name']}" in want]
            if len(rows) < len(want):
                found = {r["name"] for r in rows} | {f"{r['stage']}/{r['name']}" for r in rows}
                raise KeyError(f"not in the registry: {sorted(want - found)}")
    if not rows:
        raise ValueError("select_1se: no registry entries selected")
    cands = []
    for r in rows:
        arr = np.load(results_dir / r["val_nll_path"])
        if _sha256_array(arr) != r["val_nll_sha256"]:
            raise RuntimeError(f"{r['val_nll_path']}: content differs from the registry row (SHA-256)")
        label = r["name"] if len({x["stage"] for x in rows}) == 1 else f"{r['stage']}/{r['name']}"
        cands.append(stats.Candidate(name=label, nll=arr, k=int(r["k"]), C=float(r["C"] or 0.0)))
    return stats.one_se_select(cands)
