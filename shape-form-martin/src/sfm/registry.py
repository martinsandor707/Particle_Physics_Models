"""sfm.registry - validation evaluation of finished models; the ONLY writer of ``results/registry.jsonl``.

    val = registry.load_validation()                                   # the cached validation split, checked
    row = registry.evaluate(model, "b0h_seed0", "m2b_mlp", val, meta={"seed": 0, "config": cfg, "checkpoint": path})
    tab = registry.table()                                             # pandas.DataFrame, latest row per stream
    sel = registry.select_1se(stage="m2b_mlp")                         # sfm.stats.one_se_select, one stream only

Machine and stream tags (schema 2; ``transfer/REVIEW_STATUS.md`` section 4, PLAN.md section 13 A6 revised)
    Every row and every per-event array carries the live *machine tag* (``sfm.machine``: GPU, launch grid, torch /
    CUDA build, Python, TF32 flags, deterministic flag, native draw) and *points tag* (``sfm.provenance``: grid, seed,
    K, batch, fold scheme, validation fingerprint, subset). Pairing partners come ONLY from the reference set of the
    live tag, ``results/machines/<machine id>/refsets/<points id>/`` (created once by
    ``scripts/val_parity.py --create-refset``, read-only afterwards, hashes in its tracked ``refset.json``).
    ``evaluate`` raises ``NoReferenceSet`` when that set does not exist; there is no fallback to the untagged laptop
    files ``results/m0_parity/*_val_nll.npy`` or ``results/registry_refs/*.npy``. Rows written before tags existed
    (the laptop's eight rows of 24 Nov 2026, schema 1) are records: ``latest`` keys them under stream ``None``,
    ``row_array`` refuses them, and nothing pairs with them.

What ``evaluate`` does
    0. ``sfm.machine.startup``: hard failure without CUDA (common.py would fall back to the CPU silently), TF32 off.
    1. Refuses anything that is not the validation split of the cache (``check_validation``): the number of
       events, the event numbers *in order* (equal to ``results/m0_parity/val_event_number.npy``, the order of the
       pairing partners) and a SHA-256 fingerprint of every tensor of the ``Showers`` object (hits, weights, cell
       box, per-event p / ln p / zb / R / bin / evis) must equal those of the cached validation split. A test set,
       a subset, a reordering or modified tensors are rejected (``NotValidationSet``) before any model is run.
    2. Loads the reference set of the live tag (``reference_set``; raises without it, without a passing parity
       record, or when a recomputation was found to differ).
    3. Per-event validation NLL with the UNMODIFIED ``common.test_nll(model, val)`` (K = 64, CUDA generator seed
       12345, batches of 128) called inside ``sfm.metric.deterministic()``, so the stored array is bit-reproducible on
       the machine that computed it (PLAN.md section 13). Saved, with its tag sidecar, as
       ``R/machines/<machine id>/runs/<stage>/<name>/scores/<points id>/val_nll.npy`` (R = ``paths.OUT_RESULTS``).
       Tagged arrays are never overwritten.
    4. Paired difference against both benchmarks of the reference set through ``sfm.provenance.paired_delta`` (the
       one delta function; it refuses arrays with different tags).
    5. Optional guard columns (``guards=True``; plan section 6 with the section-13 refinements):
         G-K      ``sfm.metric.test_nll`` at K = 1024 (chunked, same seed and batches), paired against the
                  benchmarks at K = 1024 on the same points, and ``stats.guard_persistence`` (K64 gain vs K1024 gain)
         G-bulk   bulk-restricted NLL (``metric.Bulk(short="raise")``, z' < 1150 mm, last ring dropped) on the
                  stratum ``zb >= 1221 mm``, paired against the benchmarks on the same stratum and points, with
                  the report-metric gain on the same events and ``stats.guard_persistence``; the complementary
                  stratum ``zb < 1221 mm`` is reported as its own row (report-metric gain, share of the total gain)
         G-mass   ``sfm.metric.mass_monitors`` averaged over the validation events (mass beyond zb, last ring,
                  first / last z-bin, outside the bulk region, in cells with stored acceptance Abar <= 0.01)
       The benchmark guard arrays are part of the reference set (never computed here).
       Guard numbers live in their own columns and never share a column or a delta with the report metric.
    6. One JSON line appended under an exclusive ``flock``: name, stage, k (free constants), C (operator
       complexity, optional), validation mean, paired deltas +- SE vs both benchmarks, guards, seed, config hash,
       checkpoint path and SHA-256, the SHA-256 of the weight tensors, ``trained_on`` / ``scored_on``, the stream
       tag, the SHA-256 of the benchmark arrays paired with, git commit (and whether the tree was dirty), host,
       timestamp, the array's SHA-256.

Nothing here opens the test split; ``check_validation`` makes it impossible to score one through this module.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from . import data as sfm_data
from . import machine as sfm_machine
from . import metric, paths, provenance, stats
from .provenance import MixedTags, TaggedArray

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common as C  # noqa: E402

__all__ = ["NotValidationSet", "NoReferenceSet", "UntaggedRow", "ReferenceSetNotLicensed", "ReferenceSetDisputed", "ValidationReference", "ValidationSet", "RegistryEnv",
           "RefSet", "fingerprint", "validation_reference", "load_validation", "check_validation", "evaluate",
           "read_registry", "table", "select_1se", "latest", "row_stream_id", "live_stream", "reference_set",
           "create_reference_set", "registered_row", "row_array", "BENCHMARKS", "ZB_STRATUM_MM", "K_GUARD",
           "ROW_SCHEMA", "VAL_EVENT_NUMBER_FILE"]

EXPECTED_VAL_EVENTS = 14950
BENCHMARKS = ("mlp", "formula")
ZB_STRATUM_MM = 1221.0          # smallest zb of a front-layer start (Gate 0, amendment A4)
K_GUARD = 1024
GUARD_CHUNK = 64
EVAL_SEED, EVAL_BATCH, EVAL_K = 12345, 128, 64
BULK_SPEC = {"z_cut": 1150.0, "drop_rings": 1, "renormalise": True, "short": "raise"}
ROW_SCHEMA = 2
METRIC_OF_RECORD = "common.test_nll (unmodified), K=64, CUDA generator seed 12345, batch 128, inside " \
                   "sfm.metric.deterministic()"
VAL_EVENT_NUMBER_FILE = paths.MARTIN_DIR / "results" / "m0_parity" / "val_event_number.npy"   # an input (b1)
REFSET_FILE = "refset.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class NotValidationSet(PermissionError):
    """The data passed to the registry is not the cached validation split."""


class NoReferenceSet(FileNotFoundError):
    """No reference set exists for the live machine and points tag (no fallback to untagged arrays)."""


class UntaggedRow(ValueError):
    """A registry row without a stream tag (laptop record, schema 1) was about to be used as a live row."""


class ReferenceSetNotLicensed(PermissionError):
    """A reference set without a passing parity record (review A3): never created, never used."""


class ReferenceSetDisputed(PermissionError):
    """A recomputation of the reference set differed from the stored one (``exists_DIFFERENT``): the set is blocked
    for every use until the user has decided (review A3)."""


# ----------------------------------------------------------------------------------------------------------------------
# Locations (overridable for tests)
# ----------------------------------------------------------------------------------------------------------------------
LEGACY_REGISTRY = paths.REPO_RESULTS / "registry.jsonl"   # the laptop's rows of 24 Nov 2026 (records, read-only)


@dataclass
class RegistryEnv:
    results_dir: Path = field(default_factory=lambda: paths.OUT_RESULTS)   # the output root's results/
    registry_path: Optional[Path] = None              # default <results>/machines/<machine id>/registry.jsonl
    machine: Optional[sfm_machine.MachineTag] = None  # default: the live machine tag

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        if self.machine is None:
            self.machine = sfm_machine.machine_tag(deterministic=True)
        self.registry_path = (Path(self.registry_path) if self.registry_path
                              else provenance.machine_dir(self.results_dir, self.machine.id) / "registry.jsonl")

    def machine_tag(self) -> sfm_machine.MachineTag:
        return self.machine

    @property
    def machine_dir(self) -> Path:
        return provenance.machine_dir(self.results_dir, self.machine.id)


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


def validation_reference(cache_dir=None, results_dir=None, event_number_file=None) -> ValidationReference:
    """Identity of the cached validation split: event numbers in the order of the Gate 0 pairing partners
    (``results/m0_parity/val_event_number.npy``, an input file: taken from ``results_dir`` if it holds one, else from
    the repository) and the fingerprint of the validation tensors in the train/val cache file. The fingerprint is
    computed once from the cache (val split only, on the CPU) and remembered in ``cache/pion_s1_val_fingerprint.json``,
    keyed by the cache file's SHA-256 recorded in the cache metadata."""
    ev_file = Path(event_number_file) if event_number_file else None
    if ev_file is None and results_dir is not None:
        cand = Path(results_dir) / "m0_parity" / "val_event_number.npy"
        ev_file = cand if cand.exists() else None
    ev_file = ev_file or VAL_EVENT_NUMBER_FILE
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
    return provenance.sha256_file(path)


def _sha256_array(a: np.ndarray) -> str:
    return provenance.sha256_array(a)


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
    paths.check_output_path(path)
    if Path(path).resolve() == LEGACY_REGISTRY.resolve():
        raise PermissionError(f"{LEGACY_REGISTRY} holds the laptop's records and is never appended to; rows go to the "
                              "machine's own registry under results/machines/<machine id>/")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = paths.scrub_str(json.dumps(row, default=_json_default, allow_nan=True))
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


def _check_name(s: str, what: str) -> str:
    if not isinstance(s, str) or not _NAME_RE.match(s) or ".." in s:
        raise ValueError(f"{what} must match {_NAME_RE.pattern} (no path separators), got {s!r}")
    return s


# ----------------------------------------------------------------------------------------------------------------------
# Streams and the reference set
# ----------------------------------------------------------------------------------------------------------------------
def _points(ref_fp: str, grid, kind: str) -> dict:
    """Points tag of an array of the given kind ("k64", "k1024", "bulk_zb1221") on the validation split."""
    if kind == "k64":
        return provenance.points_tag(grid=grid, K=EVAL_K, split_fingerprint=ref_fp, seed=EVAL_SEED, batch=EVAL_BATCH)
    if kind == "k1024":
        return provenance.points_tag(grid=grid, K=K_GUARD, split_fingerprint=ref_fp, seed=EVAL_SEED, batch=EVAL_BATCH,
                                     chunk=GUARD_CHUNK)
    if kind == "bulk_zb1221":
        return provenance.points_tag(grid=grid, K=EVAL_K, split_fingerprint=ref_fp, seed=EVAL_SEED, batch=EVAL_BATCH,
                                     subset=f"zb >= {ZB_STRATUM_MM:g} mm", bulk=BULK_SPEC)
    raise ValueError(kind)


def live_stream(env: Optional[RegistryEnv] = None, reference: Optional[ValidationReference] = None) -> dict:
    """{"machine": MachineTag, "points": base points tag (K = 64), "stream_id": ...} of this process."""
    env = env or RegistryEnv()
    m = env.machine_tag()
    ref = reference if reference is not None else validation_reference(results_dir=env.results_dir)
    pts = _points(ref.fingerprint, m.fields.get("launch_grid"), "k64")
    return {"machine": m, "points": pts, "stream_id": provenance.stream_id(pts, m.id), "reference": ref}


def _guard_spec(n_events: int, ref_fp: str, machine: Optional[sfm_machine.MachineTag] = None,
                points: Optional[Mapping] = None) -> dict:
    """Spec of the benchmark guard arrays; contains the stream tag (spec point 11)."""
    d = {"K": K_GUARD, "chunk": GUARD_CHUNK, "seed": EVAL_SEED, "batch": EVAL_BATCH, "bulk": dict(BULK_SPEC),
         "stratum": f"zb >= {ZB_STRATUM_MM}", "val_fingerprint": ref_fp, "n_events": n_events}
    if machine is not None:
        d["machine_id"] = machine.id
        d["machine"] = machine.fields
    if points is not None:
        d["points_id"] = provenance.points_id(points)
    return d


@dataclass
class RefSet:
    dir: Path
    manifest: dict
    bench: dict                         # {"mlp": TaggedArray, "formula": TaggedArray}  (K = 64)
    guards: Optional[dict]              # {"mlp": {"k1024": TaggedArray, "bulk_zb1221": TaggedArray}, ...} or None
    manifest_sha256: str

    def hashes(self) -> dict:
        out = {b: self.bench[b].sha256 for b in BENCHMARKS}
        if self.guards:
            for b in BENCHMARKS:
                for k, ta in self.guards[b].items():
                    out[f"{b}_{k}"] = ta.sha256
        return out


def _refset_files(d: Path) -> dict:
    return {b: {"k64": d / f"{b}_val_nll.npy", "k1024": d / f"{b}_val_k1024.npy",
                "bulk_zb1221": d / f"{b}_val_bulk_zb1221.npy"} for b in BENCHMARKS}


def reference_set(env: Optional[RegistryEnv] = None, reference: Optional[ValidationReference] = None,
                  stream: Optional[dict] = None, need_guards: bool = False) -> RefSet:
    """The reference set of the live tag. Raises ``NoReferenceSet`` if it does not exist (no fallback), and
    ``provenance.UntaggedArray`` / ``MixedTags`` if a file in it does not carry the expected tag and hash."""
    env = env or RegistryEnv()
    st = stream or live_stream(env, reference)
    m, pts = st["machine"], st["points"]
    d = provenance.refset_dir(env.results_dir, m.id, pts)
    mf = d / REFSET_FILE
    disputes = sorted(d.parent.glob(f"{d.name}.DIFFERENT*.json")) if d.parent.exists() else []
    if disputes:
        raise ReferenceSetDisputed(f"{d}: a recomputation differed from the stored reference set "
                                   f"({', '.join(x.name for x in disputes)}); it is blocked for every use until the "
                                   "user has decided (nothing is deleted by code)")
    if not mf.exists():
        raise NoReferenceSet(
            f"no reference set for the live tag (machine {m.id}, points {provenance.points_id(pts)}): {d} does not "
            "exist. Create it once with scripts/val_parity.py --create-refset (run-book steps 6-7). There is no "
            "fallback to results/m0_parity/*_val_nll.npy or results/registry_refs/*.npy (laptop records).")
    manifest = json.loads(mf.read_text())
    if (manifest.get("parity") or {}).get("verdict") != "PASS":
        raise ReferenceSetNotLicensed(f"{mf}: no passing parity record (key 'parity'); a reference set is used only "
                                      "if it was created after a passing scripts/val_parity.py run in that process")
    if manifest.get("machine_id") != m.id or manifest.get("points_id") != provenance.points_id(pts):
        raise MixedTags(f"{mf}: holds machine {manifest.get('machine_id')} / points {manifest.get('points_id')}, "
                        f"live is {m.id} / {provenance.points_id(pts)}")
    files = _refset_files(d)
    fp = pts["split_fingerprint"]
    grid = pts["grid"]
    bench, guards = {}, None
    for b in BENCHMARKS:
        ta = provenance.load_tagged(files[b]["k64"], expect_points=_points(fp, grid, "k64"), expect_machine_id=m.id,
                                    label=f"refset:{b}:k64")
        if manifest["arrays"][f"{b}_k64"]["sha256"] != ta.sha256:
            raise provenance.UntaggedArray(f"{files[b]['k64']}: SHA-256 differs from {mf}")
        bench[b] = ta
    if manifest.get("guards"):
        guards = {}
        for b in BENCHMARKS:
            guards[b] = {}
            for k in ("k1024", "bulk_zb1221"):
                ta = provenance.load_tagged(files[b][k], expect_points=_points(fp, grid, k), expect_machine_id=m.id,
                                            label=f"refset:{b}:{k}")
                if manifest["arrays"][f"{b}_{k}"]["sha256"] != ta.sha256:
                    raise provenance.UntaggedArray(f"{files[b][k]}: SHA-256 differs from {mf}")
                guards[b][k] = ta
    elif need_guards:
        raise NoReferenceSet(f"{d}: the reference set holds no guard arrays; it is created with guards by "
                             "scripts/val_parity.py --create-refset (guards are never computed inside evaluate)")
    return RefSet(dir=d, manifest=manifest, bench=bench, guards=guards, manifest_sha256=provenance.sha256_file(mf))


def _guard_arrays(model, val, stratum_idx) -> dict:
    """Per-event guard arrays of one model (same points for every model)."""
    model.eval()
    with torch.inference_mode():
        k1024 = metric.test_nll(model, val, K=K_GUARD, chunk=GUARD_CHUNK, deterministic_sum=True)
        sub = val.subset(torch.as_tensor(stratum_idx, device=val.off.device))
        bulk = metric.test_nll(model, sub, bulk=metric.Bulk(short="raise"), deterministic_sum=True)
    return {"k1024": k1024, "bulk_zb1221": bulk}


def _default_bench_models() -> dict:
    """mlp_std and the component formula through the read-only adapter (the formula needs the training ln p range)."""
    from . import legacy
    sets, _ = sfm_data.load_sets(("train",), device="cpu")
    return {"mlp": legacy.load_mlp_std(), "formula": legacy.load_component_formula(sets["train"])}


def create_reference_set(val_set, *, env: Optional[RegistryEnv] = None,
                         reference: Optional[ValidationReference] = None, bench_models: Optional[Callable] = None,
                         guards: bool = True, k64_arrays: Optional[Mapping] = None, seal: bool = True,
                         log: Callable = print, parity: Optional[Mapping] = None) -> dict:
    """Create the reference set of the live tag ONCE (spec point 9): K = 64 pairing arrays of both benchmarks with
    the unmodified ``common.test_nll`` inside ``deterministic()`` (or the arrays passed in ``k64_arrays``, which must
    have been computed that way in this process), and - with ``guards`` - the K = 1024 and bulk guard references.
    Written into a temporary directory and renamed into place, then made read-only (``seal``; spec point 12).
    If the set exists already, everything is recomputed and compared bit for bit; nothing is overwritten.
    ``parity`` is REQUIRED for a new set: the record of the passing parity run that licenses it ({"verdict": "PASS",
    "report_sha256", ...}); it is stored in ``refset.json`` and ``reference_set`` refuses a set without it (A3).
    If the set exists, a recomputation that differs writes ``<set>.DIFFERENT_<utc>.json`` beside it, which blocks
    the set for every use (``ReferenceSetDisputed``).
    Returns {"status": "created" | "exists_identical" | "exists_DIFFERENT", "dir", "hashes", ...}."""
    sfm_machine.startup("registry.create_reference_set")
    env = env or RegistryEnv()
    st = live_stream(env, reference)
    ref, m, pts = st["reference"], st["machine"], st["points"]
    val = check_validation(val_set, ref)
    models = (bench_models or _default_bench_models)()
    arrays = {}
    for b in BENCHMARKS:
        if k64_arrays is not None:
            arrays[(b, "k64")] = np.asarray(k64_arrays[b])
        else:
            mdl = models[b].eval()
            with metric.deterministic():
                arrays[(b, "k64")] = C.test_nll(mdl, val)
    stratum_idx = None
    if guards:
        zb = val.ev["zb"].detach().cpu().numpy()
        stratum_idx = np.flatnonzero(zb >= ZB_STRATUM_MM)
        for b in BENCHMARKS:
            g = _guard_arrays(models[b].eval(), val, stratum_idx)
            for k, a in g.items():
                arrays[(b, k)] = a
    for (b, k), a in arrays.items():
        if not np.isfinite(a).all():
            raise FloatingPointError(f"reference array {b}/{k}: non-finite values")
    final = provenance.refset_dir(env.results_dir, m.id, pts)
    fp, grid = pts["split_fingerprint"], pts["grid"]
    if not (final / REFSET_FILE).exists() and (not parity or parity.get("verdict") != "PASS"):
        raise ReferenceSetNotLicensed("create_reference_set: a new reference set needs a passing parity record "
                                      "(parity={'verdict': 'PASS', ...}); none given")
    if (final / REFSET_FILE).exists():
        existing = reference_set(env, stream=st)
        diffs = {}
        for (b, k), a in arrays.items():
            old = existing.bench[b] if k == "k64" else (existing.guards or {}).get(b, {}).get(k)
            if old is None:
                diffs[f"{b}_{k}"] = "missing in the existing set"
            elif not np.array_equal(old.values, a):
                diffs[f"{b}_{k}"] = float(np.abs(old.values.astype(np.float64) - a.astype(np.float64)).max())
        status = "exists_identical" if not diffs else "exists_DIFFERENT"
        log(f"[refset] {paths.portable(final)} exists; recomputation: {status} {diffs or ''}")
        if diffs:
            marker = paths.check_output_path(final.parent / (f"{final.name}.DIFFERENT_"
                                             f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"))
            marker.write_text(paths.scrub_str(json.dumps({"refset": final.name, "differences": diffs,
                                                          "machine_id": m.id, "git": sfm_machine.git_state(),
                                                          "parity_of_this_run": parity}, indent=1,
                                                         default=_json_default)) + "\n")
            log(f"[refset] BLOCKED: {paths.portable(marker)} written; the set is refused for every use")
        return {"status": status, "dir": str(final), "differences": diffs, "hashes": existing.hashes()}
    tmp = paths.check_output_path(final.parent / f".{final.name}.tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files = _refset_files(tmp)
    manifest_arrays = {}
    for (b, k), a in arrays.items():
        ta = provenance.save_tagged(files[b][k], a, _points(fp, grid, k), m, label=f"refset:{b}:{k}")
        manifest_arrays[f"{b}_{k}"] = {"file": files[b][k].name, "sha256": ta.sha256,
                                       "file_sha256": provenance.sha256_file(files[b][k]),
                                       "mean": float(np.nanmean(a.astype(np.float64))), "n": int(a.size),
                                       "points": _points(fp, grid, k)}
    manifest = {"refset_schema": 1, "machine_id": m.id, "machine": m.fields, "points_id": provenance.points_id(pts),
                "points": pts, "stream_id": st["stream_id"], "host_id": sfm_machine.host_id(),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": sfm_machine.git_state(),
                "metric": METRIC_OF_RECORD, "validation": {"fingerprint": fp, "n_events": len(val),
                                                           "source": ref.source},
                "guards": bool(guards), "guard_spec": _guard_spec(len(val), fp, m, pts) if guards else None,
                "n_stratum": int(len(stratum_idx)) if stratum_idx is not None else None,
                "arrays": manifest_arrays, "parity": dict(parity),
                "common_py_sha256": provenance.sha256_file(paths.MARTIN_DIR / "common.py"),
                "benchmarks": {"mlp": "mlp_std.pt (legacy.load_mlp_std)",
                                                          "formula": "new3_std_n3.pt (legacy.load_component_formula)"}}
    (tmp / REFSET_FILE).write_text(paths.scrub_str(json.dumps(manifest, indent=1, default=_json_default)) + "\n")
    paths.check_output_path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(tmp, final)                                   # fails if another process created it meanwhile
    sealed = provenance.make_read_only([p for p in final.iterdir() if p.is_file()]) if seal else []
    log(f"[refset] created {paths.portable(final)} ({len(manifest_arrays)} arrays){'; sealed read-only' if seal else ''}")
    return {"status": "created", "dir": str(final), "hashes": {k: v["sha256"] for k, v in manifest_arrays.items()},
            "manifest": manifest, "sealed": sealed}


# ----------------------------------------------------------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------------------------------------------------------
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


def _guards(model, val, nll_t: TaggedArray, rs: RefSet, st: dict, run_dir: Path, abar, label: str) -> dict:
    m, pts = st["machine"], st["points"]
    fp, grid = pts["split_fingerprint"], pts["grid"]
    zb = val.ev["zb"].detach().cpu().numpy()
    in_stratum = zb >= ZB_STRATUM_MM
    idx = np.flatnonzero(in_stratum)
    mine = _guard_arrays(model, val, idx)
    k1024 = provenance.save_tagged(run_dir / "val_nll_k1024.npy", mine["k1024"], _points(fp, grid, "k1024"), m,
                                   label=f"{label}:k1024")
    bulk = provenance.save_tagged(run_dir / "val_bulk_zb1221.npy", mine["bulk_zb1221"],
                                  _points(fp, grid, "bulk_zb1221"), m, label=f"{label}:bulk_zb1221")
    mass = _mass(model, val, abar)
    mass_path = paths.check_output_path(run_dir / "val_mass_monitors.npz")
    mass_arrays = {k.replace(":", "_"): v for k, v in mass.items()}
    if mass_path.exists():                                 # kept if identical (a re-run after a crash), never replaced
        with np.load(mass_path) as old:
            same = set(old.files) == set(mass_arrays) and all(np.array_equal(old[k], v) for k, v in mass_arrays.items())
        if not same:
            raise FileExistsError(f"{mass_path} exists with other content; tagged outputs are never overwritten")
    else:
        tmp = mass_path.with_name(f".{mass_path.stem}.tmp{os.getpid()}.npz")
        np.savez(tmp, **mass_arrays)
        os.replace(tmp, mass_path)
    out = {"K": K_GUARD, "stratum": f"zb >= {ZB_STRATUM_MM:g} mm", "n_stratum": int(len(idx)),
           "n_short": int((~in_stratum).sum()), "val_mean_k1024": float(np.mean(mine["k1024"], dtype=np.float64)),
           "arrays": {"k1024": {"path": str(k1024.path), "sha256": k1024.sha256},
                      "bulk_zb1221": {"path": str(bulk.path), "sha256": bulk.sha256},
                      "mass_monitors": {"path": str(mass_path), "file_sha256": provenance.sha256_file(mass_path)}}}
    bulk_ok = np.isfinite(mine["bulk_zb1221"])
    out["n_bulk_nan"] = int((~bulk_ok).sum())
    for b in BENCHMARKS:
        main = provenance.paired_difference(nll_t, rs.bench[b])
        gk = provenance.paired_difference(k1024, rs.guards[b]["k1024"])
        g = {"gk": _paired_dict(provenance.paired_delta(k1024, rs.guards[b]["k1024"]))}
        dk = stats.guard_persistence(main, gk)
        g["gk_persistence"] = {"passed": dk.passed, "checks": dk.checks, "numbers": dk.numbers, "flags": dk.flags}
        okb = bulk_ok & np.isfinite(rs.guards[b]["bulk_zb1221"].values)
        gb = provenance.paired_difference(bulk, rs.guards[b]["bulk_zb1221"], mask=okb)
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
_META_KEYS = ("k", "C", "seed", "config", "config_hash", "checkpoint", "notes", "trained_on")


def _trained_on_from_checkpoint(ckpt) -> Optional[dict]:
    """The machine tag stored by ``sfm.train.fit`` in ``model.pt`` (meta key "machine_tag"), if any."""
    try:
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception:
        return None
    for holder in (obj, obj.get("meta") or {}):
        t = holder.get("machine_tag") if isinstance(holder, dict) else None
        if t:
            return {"machine_id": t.get("id"), "machine": t.get("fields"), "source": "model.pt machine_tag"}
    return None


def evaluate(model, name: str, stage: str, val_set, meta: Optional[Mapping] = None, *, guards: bool = False,
             replace: bool = False, env: Optional[RegistryEnv] = None, reference: Optional[ValidationReference] = None,
             abar: Optional[torch.Tensor] = None, **_ignored) -> dict:
    """Score ``model`` on the validation split, store the tagged per-event array, append one registry row, return it.

    meta (all optional): k (free constants; default ``model.n_params()`` or the parameter count), C (operator
    complexity), seed, config (dict; hashed), config_hash, checkpoint (path; hashed), notes, trained_on (str or dict;
    default: the machine tag stored in the checkpoint by ``sfm.train.fit``, else "unknown"); other keys go to
    ``extra``. ``guards=True`` adds the guard columns (module docstring). ``env``/``reference`` exist for tests; the
    defaults are the campaign's files. ``replace`` is refused: tagged arrays are never overwritten.
    """
    if replace:
        raise ValueError("replace=True is not supported: tagged arrays are never overwritten (use another run name)")
    if _ignored.get("bench_models") is not None:
        raise TypeError("bench_models is no longer accepted: benchmark arrays come only from the reference set")
    sfm_machine.startup("registry.evaluate")
    env = env or RegistryEnv()
    _check_name(name, "name")
    _check_name(stage, "stage")
    meta = dict(meta or {})
    st = live_stream(env, reference)
    ref, m, pts = st["reference"], st["machine"], st["points"]
    val = check_validation(val_set, ref)
    if val.xyz.device.type != torch.device(C.DEVICE).type:
        raise ValueError(f"the validation set must be on common.DEVICE ({C.DEVICE}) for common.test_nll")
    rs = reference_set(env, stream=st, need_guards=guards)

    was = model.training
    model.eval()
    with metric.deterministic():
        nll = C.test_nll(model, val)                                     # the metric of record, unmodified
    model.train(was)
    if nll.shape != (len(val),):
        raise RuntimeError(f"common.test_nll returned shape {nll.shape}")
    if not np.isfinite(nll).all():
        raise FloatingPointError(f"{int((~np.isfinite(nll)).sum())} non-finite per-event validation NLL(s)")

    run_dir = provenance.run_scores_dir(env.results_dir, stage, name, m.id, pts)
    label = f"{stage}/{name}"
    nll_t = provenance.save_tagged(run_dir / "val_nll.npy", nll, pts, m, label=label)

    ckpt = meta.get("checkpoint")
    trained_on = meta.get("trained_on") or (_trained_on_from_checkpoint(ckpt) if ckpt else None) or "unknown"
    row = {"name": name, "stage": stage, "schema": ROW_SCHEMA,
           "k": int(meta["k"]) if meta.get("k") is not None else _count_params(model),
           "C": float(meta["C"]) if meta.get("C") is not None else None,
           "n_events": int(len(nll)), "val_mean": float(np.mean(nll, dtype=np.float64)),
           "vs_mlp": _paired_dict(provenance.paired_delta(nll_t, rs.bench["mlp"])),
           "vs_formula": _paired_dict(provenance.paired_delta(nll_t, rs.bench["formula"])),
           "guards": None,
           "seed": meta.get("seed"),
           "config_hash": meta.get("config_hash") or _config_hash(meta.get("config")),
           "checkpoint": paths.portable(ckpt) if ckpt else None,
           "checkpoint_sha256": _sha256_file(Path(ckpt)) if ckpt else None,
           "weights_sha256": provenance.weights_sha256(model.state_dict()),
           "val_nll_path": os.path.relpath(nll_t.path, env.results_dir),
           "val_nll_sha256": nll_t.sha256,
           "metric": METRIC_OF_RECORD,
           "common_py_sha256": provenance.sha256_file(paths.MARTIN_DIR / "common.py"),
           "stream": {"stream_id": st["stream_id"], "points_id": provenance.points_id(pts), "points": pts,
                      "machine_id": m.id, "machine": m.fields},
           "trained_on": trained_on,
           "scored_on": {"machine_id": m.id, "host_id": sfm_machine.host_id()},
           "benchmarks": {"refset": os.path.relpath(rs.dir, env.results_dir), "refset_json_sha256": rs.manifest_sha256,
                          **{b: {"path": os.path.relpath(rs.bench[b].path, env.results_dir),
                                 "sha256": rs.bench[b].sha256} for b in BENCHMARKS}},
           "device": str(C.DEVICE), "torch": torch.__version__,
           "git": sfm_machine.git_state(),
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "notes": meta.get("notes"),
           "extra": {k: v for k, v in meta.items() if k not in _META_KEYS} or None}
    if guards:
        if abar is None:
            from .models.field import acceptance_template
            abar = acceptance_template("stored", val.xyz.device)
        row["guards"] = _guards(model, val, nll_t, rs, st, run_dir, abar, label)
        row["guards"]["refset_hashes"] = {f"{b}_{k}": rs.guards[b][k].sha256 for b in BENCHMARKS
                                          for k in ("k1024", "bulk_zb1221")}
        for k in ("k1024", "bulk_zb1221", "mass_monitors"):
            row["guards"]["arrays"][k]["path"] = os.path.relpath(row["guards"]["arrays"][k]["path"], env.results_dir)
    _append(row, env.registry_path)
    return row


# ----------------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------------
def read_registry(path: Optional[Path] = None, strict: bool = True) -> list:
    """All rows in file order (default: this machine's registry). A line that is not valid JSON raises (strict) or is
    skipped. The laptop's records: ``read_registry(registry.LEGACY_REGISTRY)``."""
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


def row_stream_id(r: Mapping) -> Optional[str]:
    """Stream id of a row; None for an untagged (schema 1, laptop) row."""
    return ((r.get("stream") or {}).get("stream_id")) if isinstance(r.get("stream"), Mapping) else None


def latest(rows: Sequence[dict], stream: Optional[str] = "__all__") -> list:
    """The last row of every (stage, name, stream id), in order of first appearance. ``stream``: keep only rows of
    this stream id (None = the untagged laptop rows); default: all streams."""
    last = {}
    for r in rows:
        sid = row_stream_id(r)
        if stream != "__all__" and sid != stream:
            continue
        last[(r["stage"], r["name"], sid)] = r
    return list(last.values())


def registered_row(stage: str, name: str, *, stream_id: str, ckpt=None, need_guards: bool = False,
                   path: Optional[Path] = None) -> Optional[dict]:
    """The latest row of (stage, name) on the stream ``stream_id`` [with this checkpoint's SHA-256] [with guards]
    (spec point 11). Rows of other streams - and the untagged laptop rows - never match (``stream_id=None`` is
    refused: review C5)."""
    if not stream_id:
        raise ValueError("registered_row needs a stream id (untagged laptop rows are never 'registered' here)")
    want = None
    if ckpt is not None:
        if not Path(ckpt).exists():
            return None
        want = _sha256_file(Path(ckpt))
    found = None
    for r in read_registry(path):
        if r.get("stage") != stage or r.get("name") != name or row_stream_id(r) != stream_id:
            continue
        if want is not None and r.get("checkpoint_sha256") != want:
            continue
        if need_guards and not r.get("guards"):
            continue
        found = r
    return found


def row_array(r: Mapping, results_dir: Optional[Path] = None, kind: str = "k64") -> TaggedArray:
    """The tagged per-event array of a row ("k64", "k1024" or "bulk_zb1221"), checked against the row's SHA-256 and
    stream tag. Raises ``UntaggedRow`` for a row without a stream tag (laptop record)."""
    results_dir = Path(results_dir) if results_dir else paths.OUT_RESULTS
    sid = row_stream_id(r)
    if sid is None:
        raise UntaggedRow(f"{r.get('stage')}/{r.get('name')} ({r.get('timestamp')}): untagged registry row (laptop "
                          "record, schema 1); it is never used in a delta")
    s = r["stream"]
    if kind == "k64":
        rel, sha = r["val_nll_path"], r["val_nll_sha256"]
    else:
        g = (r.get("guards") or {}).get("arrays") or {}
        if kind not in g:
            raise KeyError(f"{r['stage']}/{r['name']}: no {kind} array in the row")
        rel, sha = g[kind]["path"], g[kind]["sha256"]
    ta = provenance.load_tagged(results_dir / rel, expect_machine_id=s["machine_id"], label=f"{r['stage']}/{r['name']}")
    if kind == "k64" and provenance._diff_keys(ta.points, s["points"]):
        raise MixedTags(f"{rel}: points tag differs from the row's stream")
    if ta.sha256 != sha:
        raise RuntimeError(f"{rel}: content differs from the registry row (SHA-256)")
    return ta


def table(path: Optional[Path] = None, stage: Optional[str] = None, latest_only: bool = True,
          stream: Optional[str] = "__all__"):
    """Registry as a flat ``pandas.DataFrame`` (one row per entry; latest per (stage, name, stream) by default)."""
    import pandas as pd
    rows = read_registry(path)
    if latest_only:
        rows = latest(rows, stream)
    elif stream != "__all__":
        rows = [r for r in rows if row_stream_id(r) == stream]
    if stage is not None:
        rows = [r for r in rows if r["stage"] == stage]
    flat = []
    for r in rows:
        d = {k: r.get(k) for k in ("stage", "name", "k", "C", "n_events", "val_mean", "seed", "config_hash",
                                   "timestamp")}
        d["stream_id"] = row_stream_id(r)
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
               path: Optional[Path] = None, results_dir: Optional[Path] = None,
               stream: Optional[str] = None) -> stats.OneSE:
    """``sfm.stats.one_se_select`` over chosen registry entries of ONE stream (latest row per (stage, name)).

    Pick by ``stage`` and/or ``names`` (bare names or "stage/name"), or pass ``rows`` directly. ``stream``: the stream
    id to select from; default: the single stream of the selected tagged rows (an error if they span several).
    Untagged laptop rows are never selected. Every candidate's array is loaded through ``row_array`` (tag and
    SHA-256 checked) and all candidates must pass ``provenance.check_pairable`` before the 1-SE rule pairs them."""
    results_dir = Path(results_dir) if results_dir else (Path(path).parents[2] if path else paths.OUT_RESULTS)
    if rows is None:
        rows = [r for r in latest(read_registry(path)) if row_stream_id(r) is not None]
        if stream is not None:
            rows = [r for r in rows if row_stream_id(r) == stream]
        if stage is not None:
            rows = [r for r in rows if r["stage"] == stage]
        if names is not None:
            want = set(names)
            rows = [r for r in rows if r["name"] in want or f"{r['stage']}/{r['name']}" in want]
            found = {r["name"] for r in rows} | {f"{r['stage']}/{r['name']}" for r in rows}
            if want - found:
                raise KeyError(f"not in the registry (on a tagged stream): {sorted(want - found)}")
    if not rows:
        raise ValueError("select_1se: no registry entries selected")
    sids = {row_stream_id(r) for r in rows}
    if None in sids:
        raise UntaggedRow("select_1se: untagged (laptop) rows cannot be selected")
    if len(sids) > 1:
        raise MixedTags(f"select_1se: the selected rows span several streams {sorted(sids)}; pass stream=...")
    arrays = [row_array(r, results_dir) for r in rows]
    provenance.check_pairable(*arrays)
    cands = []
    for r, ta in zip(rows, arrays):
        label = r["name"] if len({x["stage"] for x in rows}) == 1 else f"{r['stage']}/{r['name']}"
        cands.append(stats.Candidate(name=label, nll=ta.values, k=int(r["k"]), C=float(r["C"] or 0.0)))
    return stats.one_se_select(cands)
