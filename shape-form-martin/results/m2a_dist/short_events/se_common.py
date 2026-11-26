"""Shared helpers of the short-event diagnostic (read-only on the machine-of-record arrays copied to the laptop).

Nothing here writes outside results/m2a_dist/short_events/. The validation split only (never the test cache).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parents[2]                       # shape-form-martin
sys.path.insert(0, str(MARTIN / "src"))
sys.path.insert(0, str(MARTIN))

import numpy as np  # noqa: E402

from sfm import provenance, stats  # noqa: E402

MID = "h200-g1056-97b197a7"
MDIR = MARTIN / "results" / "machines" / MID
RUNS = MDIR / "runs"
STAGE_DIR = RUNS / "m2a_dist"
PID = "g1056-s12345-b128-iid-val731a1997"
ZB_STRATUM_MM = 1221.0
E4 = 1e4

_ROWS = None


def registry_rows() -> list:
    global _ROWS
    if _ROWS is None:
        _ROWS = [json.loads(x) for x in (MDIR / "registry.jsonl").read_text().splitlines() if x.strip()]
    return _ROWS


def latest_row(name: str):
    r = [x for x in registry_rows() if x.get("name") == name and x.get("stage") == "m2a_dist"]
    return r[-1] if r else None


def tagged(name: str, kind: str = "k64") -> provenance.TaggedArray:
    """The registered per-event array of a run (tag sidecar and the registry row's SHA-256 checked)."""
    fn = {"k64": "val_nll.npy", "k1024": "val_nll_k1024.npy", "bulk": "val_bulk_zb1221.npy"}[kind]
    p = STAGE_DIR / name / "scores" / PID / fn
    ta = provenance.load_tagged(p, expect_machine_id=MID)
    row = latest_row(name)
    if kind == "k64" and row is not None and row["val_nll_sha256"] != ta.sha256:
        raise RuntimeError(f"{name}: array differs from its registry row")
    return ta


def gr_tagged(name: str, cls: str) -> provenance.TaggedArray:
    return provenance.load_tagged(STAGE_DIR / name / "scores" / PID / f"g_region_{cls}.npy", expect_machine_id=MID)


def mass_monitors(name: str):
    p = STAGE_DIR / name / "scores" / PID / "val_mass_monitors.npz"
    if not p.exists():
        return None
    row = latest_row(name)
    if row and row.get("guards") and provenance.sha256_file(p) != row["guards"]["arrays"]["mass_monitors"]["file_sha256"]:
        raise RuntimeError(f"{name}: mass monitor npz differs from its row")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def pd(p: stats.Paired) -> dict:
    return {"mean": p.mean * E4, "se": p.se * E4, "n": p.n, "z": p.z}


def fmt(p) -> str:
    if p is None:
        return "-"
    if isinstance(p, dict):
        return f"{p['mean']:+.2f} ± {p['se']:.2f} (n {p['n']})"
    return f"{p.mean * E4:+.2f} ± {p.se * E4:.2f} (n {p.n})"


def delta(a: str, b: str, mask=None) -> stats.Paired:
    return provenance.paired_delta(tagged(a), tagged(b), mask=mask)


def diff(a: str, b: str) -> np.ndarray:
    return provenance.paired_difference(tagged(a), tagged(b))


_VAL = {}


def val_info() -> dict:
    """Per-event quantities of the validation split (CPU): zb, p, theta, phi, evis, nhits, event_number."""
    if "v" not in _VAL:
        from sfm import data as sfm_data
        sets, extras = sfm_data.load_sets(("val",), device="cpu")
        v = sets["val"]
        off = v.off.numpy()
        _VAL["v"] = {"zb": v.ev["zb"].numpy().astype(np.float64), "p": v.ev["p"].numpy().astype(np.float64),
                     "evis": v.ev["evis"].numpy().astype(np.float64), "nhits": np.diff(off).astype(np.int64),
                     "theta": np.asarray(extras["val"]["theta"], dtype=np.float64),
                     "phi": np.asarray(extras["val"]["phi"], dtype=np.float64),
                     "event_number": np.asarray(extras["val"]["event_number"]), "_sets": sets, "_extras": extras}
    return _VAL["v"]


def short_mask() -> np.ndarray:
    """The 'zb<1221 (short events)' stratum of run_m2a_report.guard_block: ~(zb >= registry.ZB_STRATUM_MM)."""
    zb = val_info()["zb"]
    return ~(zb >= ZB_STRATUM_MM)


def write_json(obj, name: str) -> None:
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    (HERE / name).write_text(json.dumps(conv(obj), indent=1) + "\n")
