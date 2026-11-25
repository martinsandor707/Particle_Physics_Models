"""sfm.provenance - stream tags on per-event arrays, tag-named locations, and the ONE function that forms deltas.

Specification: ``transfer/REVIEW_STATUS.md`` section 4 (points 1-13, 15); PLAN.md section 13 (A6 revised, points 3-5)
and section 16.6 (user decision P6).

Two parts of a tag (spec point 10)
    points   where the fold points come from: launch grid, generator seed, K, batch size, fold scheme, split and the
             fingerprint of its tensors, and which events (all, or a stratum). ``points_tag(...)``.
    machine  the arithmetic and its provenance: ``sfm.machine.MachineTag`` (GPU, grid, torch / CUDA build, Python,
             TF32 flags, deterministic flag, native or emulated draw).
    A delta of record needs BOTH equal (``paired_delta``). A comparison between machines (for example an H200 array
    against a laptop-emulated array) is a ``tolerance_check``: it reports differences and never a paired SE.

Tagged arrays
    ``save_tagged(path, values, points, machine)`` writes ``<name>.npy`` (git-ignored) and the sidecar
    ``<name>.tag.json`` (tracked: points, machine, array SHA-256, shape, dtype). It never overwrites an existing file
    with other content. ``load_tagged(path)`` refuses an array without a sidecar, or whose content differs from the
    sidecar's SHA-256: an untagged laptop array copied to a live path is never picked up.

Locations (``R`` = ``paths.OUT_RESULTS``: ``$SFM_OUT_ROOT/shape-form-martin/results``, by default the repository's
``results/``; nothing authoritative inside the git-ignored ``cache/``, P6). One directory per machine:
    R/machines/<machine id>/identity.json                         R0 identity record (tracked once copied back)
    R/machines/<machine id>/registry.jsonl                        this machine's registry rows
    R/machines/<machine id>/refsets/<points id>/                  reference set: benchmark pairing arrays and guard
                                                                  references + refset.json (tracked)
    R/machines/<machine id>/runs/<stage>/<name>/                  runs trained on this machine (checkpoint, model,
                                                                  done-markers, logs) and, under scores/<points id>/,
                                                                  the per-run arrays of every run scored here
    R/machines/<machine id>/checks/<label>/                       tolerance-check reports (never pairing partners)
    The laptop's files keep their names (results/m0_parity/*.npy, results/<stage>/<name>/val_nll.npy, ...), so a
    machine directory copied back cannot collide with them. There is NO fallback from a tagged location to an
    untagged one. Paths inside rows are relative to R, so they resolve on the laptop after the copy-back.

Laptop records (the H200's quarantine)
    ``laptop_record(rel)`` finds a laptop file by its path relative to ``shape-form-martin/`` in the quarantine folder
    ``shape-form-martin/cache/laptop_records/shape-form-martin/<rel>`` (the layout the tar/rsync of PLAN.md section 16.9
    produces), then ``.../laptop_records/<rel>``, then at its original path (on the laptop itself), and verifies its
    file SHA-256 against the tracked ``transfer/SHA256SUMS``. Records are comparison targets only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from . import paths, stats
from .machine import MachineTag

__all__ = ["MixedTags", "UntaggedArray", "TaggedArray", "points_tag", "points_id", "stream_id", "sha256_array",
           "sha256_file", "weights_sha256", "save_tagged", "load_tagged", "check_pairable", "paired_delta",
           "paired_difference", "tolerance_check", "machine_dir", "refset_dir", "run_dir", "run_scores_dir", "checks_dir",
           "laptop_record", "laptop_record_hashes", "make_read_only", "sidecar_path", "ARITHMETIC_KEYS"]

TAG_SCHEMA = 1
ARITHMETIC_KEYS = ("gpu_name", "sm_count", "max_threads_per_sm", "launch_grid", "torch", "cuda", "cudnn", "python",
                   "tf32_matmul", "tf32_cudnn", "deterministic", "draw", "cublas_workspace_config")


class MixedTags(ValueError):
    """Two arrays with different stream tags were about to be paired."""


class UntaggedArray(FileNotFoundError):
    """An array without a (matching) tag sidecar: it is never used as a live array."""


# ----------------------------------------------------------------------------------------------------------------------
# Hashes
# ----------------------------------------------------------------------------------------------------------------------
def sha256_array(a: np.ndarray) -> str:
    """Same definition as ``sfm.registry._sha256_array`` (dtype | shape | raw bytes)."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def weights_sha256(state: Mapping) -> str:
    """Hash over the weight tensors (spec point 13): key, dtype, shape and bytes of every tensor of a state dict, in
    sorted key order. Unlike the file hash of ``model.pt`` (which contains wall time) it is equal for two
    bit-identical trainings. Non-tensor entries are hashed through their repr."""
    import torch
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        else:
            h.update(f"{k}|{v!r}|".encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Tags
# ----------------------------------------------------------------------------------------------------------------------
def points_tag(*, grid: Optional[int], K: int, split_fingerprint: str, seed: int = 12345, batch: int = 128,
               fold: str = "iid", split: str = "val", subset: str = "all", chunk: Optional[int] = None,
               bulk: Optional[dict] = None) -> dict:
    """Where the fold points of an array come from (spec point 1, 'points'). ``chunk`` and ``bulk`` describe the
    evaluation of the guard arrays (they do not change the points, but they change the estimator)."""
    d = {"grid": None if grid is None else int(grid), "seed": int(seed), "K": int(K), "batch": int(batch),
         "fold": str(fold), "split": str(split), "split_fingerprint": str(split_fingerprint), "subset": str(subset)}
    if chunk is not None:
        d["chunk"] = int(chunk)
    if bulk is not None:
        d["bulk"] = dict(bulk)
    return d


def points_id(points: Mapping) -> str:
    """Directory name of the base stream: grid, seed, batch, fold, split and fingerprint (K and subset are per array)."""
    g = points.get("grid")
    return (f"g{g if g is not None else 'na'}-s{points['seed']}-b{points['batch']}-{points['fold']}-"
            f"{points['split']}{str(points['split_fingerprint'])[:8]}")


def stream_id(points: Mapping, machine_id: str) -> str:
    """Key of a stream: base points id @ machine id (both parts; ``latest`` is keyed by it)."""
    return f"{points_id(points)}@{machine_id}"


@dataclass(frozen=True)
class TaggedArray:
    values: np.ndarray = field(repr=False)
    points: dict
    machine: dict                                      # MachineTag fields
    machine_id: str
    sha256: str
    label: str = ""
    path: Optional[str] = None

    @classmethod
    def make(cls, values: np.ndarray, points: Mapping, machine: MachineTag, label: str = "",
             path: Optional[str] = None) -> "TaggedArray":
        return cls(np.asarray(values), dict(points), machine.fields, machine.id, sha256_array(np.asarray(values)),
                   label, path)

    @property
    def stream_id(self) -> str:
        return stream_id(self.points, self.machine_id)

    def describe(self) -> str:
        return f"{self.label or self.path or 'array'} [{self.stream_id}, K={self.points.get('K')}, " \
               f"subset={self.points.get('subset')}]"


def _diff_keys(a: Mapping, b: Mapping) -> list:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def check_pairable(*arrays: TaggedArray) -> None:
    """Raise ``MixedTags`` unless every array has the same points tag and the same machine tag (and length)."""
    if not arrays:
        return
    first = arrays[0]
    for x in arrays:
        if not isinstance(x, TaggedArray):
            raise MixedTags(f"untagged array passed to a delta ({type(x).__name__}); only TaggedArray can be paired")
    for x in arrays[1:]:
        dp = _diff_keys(first.points, x.points)
        dm = _diff_keys(first.machine, x.machine)
        if dp or dm or first.machine_id != x.machine_id:
            raise MixedTags(f"refusing to pair {first.describe()} with {x.describe()}: points differ in {dp}, "
                            f"machine differs in {dm}. A delta of record needs identical fold points AND identical "
                            "arithmetic; a cross-machine comparison is a tolerance check, never a delta.")
        if x.values.shape != first.values.shape:
            raise MixedTags(f"shapes differ: {first.values.shape} vs {x.values.shape}")


def paired_difference(a: TaggedArray, b: TaggedArray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-event a - b in float64 after ``check_pairable`` (optionally restricted to ``mask``)."""
    check_pairable(a, b)
    d = a.values.astype(np.float64) - b.values.astype(np.float64)
    return d if mask is None else d[mask]


def paired_delta(a: TaggedArray, b: TaggedArray, mask: Optional[np.ndarray] = None) -> stats.Paired:
    """THE delta function of the campaign: ``stats.paired(a, b)`` after ``check_pairable``."""
    check_pairable(a, b)
    if mask is None:
        return stats.paired(a.values, b.values)
    return stats.paired(a.values[mask], b.values[mask])


def tolerance_check(values: np.ndarray, target: np.ndarray, *, event_tol: float = 1e-4, mean_tol: float = 1e-6,
                    flag_tol: float = 5e-5, relaxed: bool = True, top: int = 20) -> dict:
    """Agreement of an array with a target computed elsewhere (another machine, another stream emulation, a laptop
    record). NOT a delta: no SE, no pairing claim. Reports the strict criterion (every event within ``event_tol``,
    mean within ``mean_tol``), the relaxed criterion of PLAN.md section 16.4 (at most three events above 1e-4, none
    above 1e-3, rms <= 1e-5, means within 1e-6), the P3 flag (any event above ``flag_tol`` = 5e-5) and the largest
    deviations (index, both values)."""
    a = np.asarray(values, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.shape != b.shape:
        return {"comparable": False, "reason": f"shapes differ: {a.shape} vs {b.shape}"}
    d = a - b
    ad = np.abs(d)
    order = np.argsort(-ad, kind="stable")[:top]
    mean_diff = float(a.mean() - b.mean())
    out = {"comparable": True, "n": int(a.size), "max_abs": float(ad.max()), "rms": float(np.sqrt(np.mean(d * d))),
           "mean_values": float(a.mean()), "mean_target": float(b.mean()), "mean_diff": mean_diff,
           "n_above": {f"{t:g}": int((ad > t).sum()) for t in (1e-6, 1e-5, 3e-5, 5e-5, 1e-4, 1e-3)},
           "bit_identical": bool(np.array_equal(np.asarray(values), np.asarray(target))),
           "largest": [{"index": int(i), "value": float(a[i]), "target": float(b[i]), "diff": float(d[i])}
                       for i in order if ad[i] > 0]}
    out["strict_pass"] = bool(out["max_abs"] <= event_tol and abs(mean_diff) <= mean_tol)
    out["relaxed_pass"] = bool(relaxed and int((ad > 1e-4).sum()) <= 3 and out["max_abs"] <= 1e-3
                               and out["rms"] <= 1e-5 and abs(mean_diff) <= 1e-6)
    out["p3_flag_events_above_5e-5"] = int((ad > flag_tol).sum())
    out["tolerances"] = {"event": event_tol, "mean": mean_tol, "p3_flag": flag_tol,
                         "relaxed": "<= 3 events > 1e-4, none > 1e-3, rms <= 1e-5, means within 1e-6"}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------------------------------------------------------
def machine_dir(results_dir, machine_id: str) -> Path:
    return Path(results_dir) / "machines" / machine_id


def refset_dir(results_dir, machine_id: str, points: Mapping) -> Path:
    return machine_dir(results_dir, machine_id) / "refsets" / points_id(points)


def checks_dir(results_dir, machine_id: str, label: str) -> Path:
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$", label):
        raise ValueError(f"bad label {label!r}")
    return machine_dir(results_dir, machine_id) / "checks" / label


def run_dir(results_dir, machine_id: str, stage: str, name: str) -> Path:
    """Directory of a run trained (or scored) on this machine: R/machines/<id>/runs/<stage>/<name>."""
    return machine_dir(results_dir, machine_id) / "runs" / stage / name


def run_scores_dir(results_dir, stage: str, name: str, machine_id: str, points: Mapping) -> Path:
    return run_dir(results_dir, machine_id, stage, name) / "scores" / points_id(points)


def sidecar_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.name.rsplit(".", 1)[0] + ".tag.json")


# ----------------------------------------------------------------------------------------------------------------------
# Saving and loading tagged arrays
# ----------------------------------------------------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _atomic_write(path: Path, data: bytes) -> None:
    paths.check_output_path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_tagged(path, values: np.ndarray, points: Mapping, machine: MachineTag, *, label: str = "",
                extra: Optional[Mapping] = None) -> TaggedArray:
    """Write ``path`` (.npy) and its sidecar. An existing array is kept if bit-identical (sidecar must then agree);
    an existing array with other content raises ``FileExistsError`` - nothing is ever overwritten."""
    path = paths.check_output_path(Path(path))
    values = np.asarray(values)
    ta = TaggedArray.make(values, points, machine, label=label, path=str(path))
    side = sidecar_path(path)
    meta = {"tag_schema": TAG_SCHEMA, "array": path.name, "sha256": ta.sha256, "shape": list(values.shape),
            "dtype": str(values.dtype), "points": ta.points, "points_id": points_id(ta.points),
            "machine_id": ta.machine_id, "machine": ta.machine, "stream_id": ta.stream_id, "label": label,
            **(dict(extra) if extra else {})}
    if path.exists():
        old = np.load(path)
        if not (old.shape == values.shape and old.dtype == values.dtype and np.array_equal(old, values)):
            raise FileExistsError(f"{path} exists with other content; tagged arrays are never overwritten")
        if side.exists():
            sm = json.loads(side.read_text())
            if sm.get("sha256") != ta.sha256 or sm.get("points") != ta.points or sm.get("machine_id") != ta.machine_id:
                raise FileExistsError(f"{side} exists with another tag")
            return ta
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npy")
        np.save(tmp, values)
        os.replace(tmp, path)
    _atomic_write(side, (paths.scrub_str(json.dumps(meta, indent=1, default=_json_default)) + "\n").encode())
    return ta


def load_tagged(path, *, expect_points: Optional[Mapping] = None, expect_machine_id: Optional[str] = None,
                label: str = "") -> TaggedArray:
    """Load an array with its sidecar; refuse a missing sidecar, a content/sidecar mismatch, or (if given) another
    points tag / machine id than expected."""
    path = Path(path)
    side = sidecar_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such tagged array")
    if not side.exists():
        raise UntaggedArray(f"{path}: no tag sidecar ({side.name}); untagged arrays (e.g. laptop records) are never "
                            "used as live arrays")
    meta = json.loads(side.read_text())
    values = np.load(path)
    if sha256_array(values) != meta.get("sha256"):
        raise UntaggedArray(f"{path}: content differs from its tag sidecar (SHA-256); the file was replaced")
    ta = TaggedArray(values, dict(meta["points"]), dict(meta["machine"]), meta["machine_id"], meta["sha256"],
                     label or meta.get("label", ""), str(path))
    if expect_points is not None and _diff_keys(ta.points, expect_points):
        raise MixedTags(f"{path}: points tag differs from the expected one in {_diff_keys(ta.points, expect_points)}")
    if expect_machine_id is not None and ta.machine_id != expect_machine_id:
        raise MixedTags(f"{path}: machine {ta.machine_id}, expected {expect_machine_id}")
    return ta


def make_read_only(paths_: Iterable) -> list:
    """chmod a-w on files (and the directories that hold them, last). Returns what was changed."""
    done, dirs = [], set()
    for p in paths_:
        p = Path(p)
        if p.is_file():
            mode = p.stat().st_mode
            p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            done.append(str(p))
            dirs.add(p.parent)
    for d in sorted(dirs, key=lambda x: -len(str(x))):
        d.chmod(d.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        done.append(str(d))
    return done


# ----------------------------------------------------------------------------------------------------------------------
# Laptop records
# ----------------------------------------------------------------------------------------------------------------------
SHA256SUMS = paths.MARTIN_DIR / "transfer" / "SHA256SUMS"
QUARANTINE = paths.MARTIN_DIR / "cache" / "laptop_records"


def laptop_record_hashes(sums_file: Optional[Path] = None) -> dict:
    """{path relative to shape-form-martin/: file SHA-256} of the laptop records listed in transfer/SHA256SUMS
    (sections laptop_records and test_derived_records)."""
    sums_file = Path(sums_file or SHA256SUMS)
    out, section = {}, None
    if not sums_file.exists():
        return out
    for line in sums_file.read_text().splitlines():
        m = re.match(r"^# BEGIN (\S+)$", line)
        if m:
            section = m.group(1)
            continue
        if re.match(r"^# END ", line):
            section = None
            continue
        if section in ("laptop_records", "test_derived_records") and line and not line.startswith("#"):
            h, rel = line.split(None, 1)
            rel = rel.strip()
            if rel.startswith(paths.MARTIN_DIR.name + "/"):
                rel = rel[len(paths.MARTIN_DIR.name) + 1:]
            out[rel] = h
    return out


def laptop_record(rel: str, *, quarantine: Optional[Path] = None, allow_original: bool = True,
                  hashes: Optional[Mapping] = None) -> dict:
    """Locate a laptop record (path relative to shape-form-martin/, e.g. "results/m0_parity/mlp_val_nll_ref1056.npy").
    Returns {"path": Path or None, "where": "quarantine" | "original" | None, "sha256", "expected_sha256",
    "verified"}. Test-derived records are refused outright (they must not be opened before Gate 2c)."""
    if "test" in Path(rel).name:
        raise PermissionError(f"{rel}: test-derived records are not opened by this code")
    q = Path(quarantine or QUARANTINE)
    hashes = laptop_record_hashes() if hashes is None else hashes
    cands = [(q / paths.MARTIN_DIR.name / rel, "quarantine"), (q / rel, "quarantine")]
    if allow_original:
        cands.append((paths.MARTIN_DIR / rel, "original"))
    for p, where in cands:
        if p.exists():
            h = sha256_file(p)
            exp = hashes.get(rel)
            return {"path": p, "where": where, "sha256": h, "expected_sha256": exp,
                    "verified": bool(exp is not None and exp == h)}
    return {"path": None, "where": None, "sha256": None, "expected_sha256": hashes.get(rel), "verified": False}
