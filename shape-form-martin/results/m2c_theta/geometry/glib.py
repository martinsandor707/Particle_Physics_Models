"""Shared helpers of the geometry audit scripts (g2 ... g6). Read-only use of the campaign library; no registry."""
import json
import socket
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MARTIN = HERE.parents[2]
for _p in (MARTIN / "src", MARTIN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _retry(fn, tries=6, wait=60):
    """Another worker edits library files; an import can fail transiently."""
    for i in range(tries):
        try:
            return fn()
        except (SyntaxError, ImportError, AttributeError) as e:      # noqa: PERF203
            if i == tries - 1:
                raise
            print(f"[glib] import failed ({type(e).__name__}: {e}); retry in {wait} s", flush=True)
            time.sleep(wait)


def _imports():
    import common
    from sfm import data, geometry, gpulock, legacy, metric, paths, stats
    return common, data, geometry, gpulock, legacy, metric, paths, stats


common, data, G, gpulock, legacy, metric, paths, stats = _retry(_imports)
DEV = common.DEVICE
ARR = HERE / "arrays"
FIG = HERE / "figures"
ARR.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
K, SEED, BATCH = 64, 12345, 128
FLOOR = 1e-3                      # fixed before any validation score (see g4_acceptance.py for the training scan)
NSUB, NPHI = 2, 64


def machine_tag():
    from sfm import refstream
    tag = {"host": socket.gethostname(), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        tag["launch_grid"] = int(refstream.local_grid())
    except Exception as e:      # noqa: BLE001
        tag["launch_grid"] = f"unavailable ({type(e).__name__})"
    return tag


def sidecar(path, **kw):
    meta = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
    Path(str(path) + ".json").write_text(json.dumps(meta, indent=1, default=str) + "\n")


def load_cpu(splits=("train",)):
    """Showers on the CPU plus extras and the absolute origins (train / val only)."""
    assert set(splits) <= {"train", "val"}
    sets, extras = data.load_sets(tuple(splits), device="cpu")
    origin = {s: G.load_origins(s, extras[s]["event_number"]) for s in splits}
    return sets, extras, origin


def paired0(d):
    """mean, SE, n of a per-event difference."""
    d = np.asarray(d, dtype=np.float64)
    p = stats.paired(d, np.zeros_like(d))
    return {"mean": p.mean, "se": p.se, "n": p.n}


def fmt(p, scale=1e4):
    return f"{p['mean'] * scale:+.1f} ± {p['se'] * scale:.1f}"


def acceptance_sets(sh, extras, origin, device, which=("true",), nsub=NSUB, nphi=NPHI):
    """Acceptance arrays (CPU float32, (N, N_Z, N_R)) of every event of ``sh``.
        F, FB            front / front + back (no transverse geometry)
        FS, FSB          stepped volume, true origin
        FS_box, FSB_box  the audit's box, true origin
        FS_oct, FSB_oct  straight-edged octagon with rectangular hole, true origin
        FS_line, FSB_line  stepped volume, straight-line origin (no entry-position input)
    """
    R = sh.ev["R"].cpu()
    out = {}
    o = torch.as_tensor(origin)
    v = {"F": ("front",), "FB": ("front", "back"), "FS": ("front", "side"), "FSB": ("front", "side", "back")}
    A = G.event_acceptance(R, o, G.Volume("stepped"), v, nsub=nsub, nphi=nphi, device=device)
    out.update(A)
    two = {"FS": ("front", "side"), "FSB": ("front", "side", "back")}
    if "box" in which:
        A = G.event_acceptance(R, o, G.Volume("box"), two, nsub=nsub, nphi=nphi, device=device)
        out.update({k + "_box": a for k, a in A.items()})
    if "oct" in which:
        A = G.event_acceptance(R, o, G.Volume("octagon"), two, nsub=nsub, nphi=nphi, device=device)
        out.update({k + "_oct": a for k, a in A.items()})
    if "line" in which:
        ol = torch.as_tensor(G.straight_line_origin(extras["theta"], extras["phi"], sh.ev["zb"].cpu().numpy()))
        A = G.event_acceptance(R, ol, G.Volume("stepped"), two, nsub=nsub, nphi=nphi, device=device)
        out.update({k + "_line": a for k, a in A.items()})
    return out
