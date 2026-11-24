"""Shared helpers of the M1-d reviewer checks (results/m1_audit/checks/). Validation and training data only.

Nothing here opens the test cache: ``load_val`` / ``load_trainval`` call ``sfm.data.load_sets`` with the default
``allow_test=False`` and never name the test split. Data are loaded to the CPU first and moved to the GPU only inside
``gpu_lock`` blocks of the calling script.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parents[2]
for p in (MARTIN, MARTIN / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import data, legacy, metric, paths, stats  # noqa: E402,F401
from sfm.gpulock import gpu_lock  # noqa: E402,F401
import common  # noqa: E402

DEV = common.DEVICE
AUDIT = MARTIN / "results" / "m1_audit"
M0 = MARTIN / "results" / "m0_parity"

P_EDGES = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0 + 1e-6]
P_LABELS = ["<1", "1-2", "2-5", "5-10", "10-15", "15-20"]


def load_cpu(splits=("val",)):
    """(sets, extras) on the CPU. ``splits`` may only contain 'train' and 'val'."""
    splits = tuple(splits)
    assert set(splits) <= {"train", "val"}, "the reviewer checks never open the test split"
    return data.load_sets(splits, device="cpu")


def mean_se(x):
    x = np.asarray(x, np.float64)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def fmt(x, scale=1e4, nd=2):
    m, s = mean_se(x)
    return f"{m * scale:+.{nd}f} ± {s * scale:.{nd}f}"


def p_group(p):
    return np.clip(np.searchsorted(P_EDGES, np.asarray(p, np.float64), side="right") - 1, 0, len(P_LABELS) - 1)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return jsonable(o.detach().cpu().numpy())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write(name: str, obj: dict, t0: float | None = None):
    obj = dict(obj)
    obj["_run"] = {"script": f"results/m1_audit/checks/{Path(sys.argv[0]).name}",
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "seconds": None if t0 is None else round(time.time() - t0, 1), "torch": torch.__version__}
    dest = HERE / name
    dest.write_text(json.dumps(jsonable(obj), indent=1) + "\n")
    print(f"wrote {dest.relative_to(MARTIN)}")


def hit_event(sh):
    off = sh.off.cpu().numpy()
    return np.repeat(np.arange(len(off) - 1), np.diff(off))
