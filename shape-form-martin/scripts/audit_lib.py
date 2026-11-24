"""Shared helpers of the Milestone-1 diagnostic audit (scripts/audit_*.py). Validation and training data only.

Every audit script imports this module first: it puts ``shape-form-martin`` and ``shape-form-martin/src`` on
``sys.path``, disables bytecode writing, and provides
    load()                 cached (sets, extras, mlp_std, formula) - train + val only; the test cache is never opened
    parts_val()            per-hit / per-event parts of both benchmarks on validation, K=64, seed 12345, batch 128,
                           computed once by audit_parts.py (results/m1_audit/parts_val_k64.npz)
    paired, mean_se        means and standard errors (float64)
    momentum groups, coarse (z', r) regions, the 12 event-quantile momenta
    LogMix, Hybrid         zero-training models built from existing maps (ensembles, D3 hybrids)
    write_json             JSON writer with numpy conversion and run time
Convention: d_e = NLL_formula - NLL_MLP (positive = formula worse); "gap" = mean of d_e.
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

MARTIN = Path(__file__).resolve().parents[1]
for p in (MARTIN, MARTIN / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sfm import data, legacy, metric, paths, stats  # noqa: E402,F401
import common  # noqa: E402

assert legacy.common is common, "two different common modules on sys.path"

OUT = paths.RESULTS_DIR / "m1_audit"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
M0 = paths.M0_DIR
DEV = common.DEVICE

# ---- numbers of record (PLAN.md section 13, A6) ----------------------------------------------------------------------
RECORD = {"mlp_val": 18.117857, "formula_val": 18.120061, "gap_val": 0.002204, "gap_val_se": 0.000293}

# ---- binning ---------------------------------------------------------------------------------------------------------
P_GROUP_EDGES = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0 + 1e-6]
P_GROUP_LABELS = ["<1", "1-2", "2-5", "5-10", "10-15", "15-20"]
COARSE_Z = np.array([-200, -20, 12, 35, 60, 100, 200, 400, 700, 1450], dtype=np.float64)
COARSE_R = np.array([0, 15, 30, 60, 100, 200, 350, 700], dtype=np.float64)
NZC, NRC = len(COARSE_Z) - 1, len(COARSE_R) - 1
ZB_FULL = 1221.0          # smallest zb of a front-layer start (gate0 section 4)
MLP_Z_FEATURE_PERIODS_PLAN = (3300.0, 1650.0, 1100.0, 825.0)   # as written in PLAN.md section 4 (D9)

# ---- plotting roles (dataviz reference palette, light mode) -----------------------------------------------------------
COL = {"mlp": "#2a78d6", "formula": "#eb6834", "h1": "#1baf7a", "h2": "#eda100", "ink": "#0b0b0b",
       "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb", "neutral": "#f0efec", "red": "#e34948",
       "violet": "#6250d6", "magenta": "#e87ba4"}


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bgr", ["#184f95", COL["mlp"], COL["neutral"], COL["red"], "#9c2323"])


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": COL["surface"], "axes.facecolor": COL["surface"],
                         "savefig.facecolor": COL["surface"], "axes.edgecolor": COL["ink2"],
                         "axes.labelcolor": COL["ink"], "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                         "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 110})
    return plt


# ---- loading ---------------------------------------------------------------------------------------------------------
_CACHE: dict = {}


def load():
    """(sets, extras, mlp, formula): train + val Showers on the GPU, per-event extras, both benchmarks in eval mode."""
    if "sets" not in _CACHE:
        sets, extras = data.load_sets(("train", "val"))
        _CACHE.update(sets=sets, extras=extras, mlp=legacy.load_mlp_std(),
                      formula=legacy.load_component_formula(sets["train"]))
    return _CACHE["sets"], _CACHE["extras"], _CACHE["mlp"], _CACHE["formula"]


def load_local_mlp(H: int):
    """``common.FieldModel(hidden=H, n_layers=3)`` from shape-form-original/results/pion/mlp_L3xH{H}.pt (read only).
    Returns (model in eval mode, checkpoint metadata)."""
    path = paths.ORIGINAL_DIR / "results" / "pion" / f"mlp_L3xH{H}.pt"
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = common.FieldModel(hidden=H, n_layers=3).to(DEV)
    m.load_state_dict(ck["model"])
    meta = {"file": str(path.relative_to(paths.REPO_ROOT)), "done": bool(ck.get("done")), "epoch": int(ck["epoch"]),
            "best_val_K16": float(ck["best"]), "n_params": int(sum(p.numel() for p in m.parameters()))}
    return m.eval(), meta


# ---- statistics ------------------------------------------------------------------------------------------------------
def mean_se(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = int(len(x))
    if n == 0:
        return {"mean": None, "se": None, "n": 0}
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else None, "n": n}


def paired(a, b) -> dict:
    """mean +- SE of a - b (per event, float64)."""
    return mean_se(np.asarray(a, np.float64) - np.asarray(b, np.float64))


def quantile_bins(x, q: int):
    """Integer class 0..q-1 by quantiles of x (edges from np.quantile; ties go to the upper class)."""
    x = np.asarray(x, np.float64)
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    cls = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, q - 1)
    return cls, edges


def p_group(p) -> np.ndarray:
    return np.clip(np.searchsorted(P_GROUP_EDGES, np.asarray(p, np.float64), side="right") - 1, 0,
                   len(P_GROUP_LABELS) - 1)


def coarse_index(z, r) -> np.ndarray:
    """Coarse region index iz*NRC + ir for points (z', r) [mm]; -1 outside [-200, 1450) x [0, 700)."""
    z, r = np.asarray(z, np.float64), np.asarray(r, np.float64)
    iz = np.searchsorted(COARSE_Z, z, side="right") - 1
    ir = np.searchsorted(COARSE_R, r, side="right") - 1
    # the grid upper edge 1450 belongs to the last z region (hit centres are cut at z' <= 1450)
    iz = np.where(z == COARSE_Z[-1], NZC - 1, iz)
    ok = (iz >= 0) & (iz < NZC) & (ir >= 0) & (ir < NRC)
    return np.where(ok, iz * NRC + ir, -1)


def region_labels():
    z = [f"[{COARSE_Z[i]:g},{COARSE_Z[i + 1]:g})" for i in range(NZC)]
    r = [f"[{COARSE_R[j]:g},{COARSE_R[j + 1]:g})" for j in range(NRC)]
    return z, r


def region_name(k: int) -> str:
    z, r = region_labels()
    return f"z'{z[k // NRC]} r{r[k % NRC]}"


# ---- per-hit helpers -------------------------------------------------------------------------------------------------
def hit_event(sh) -> np.ndarray:
    off = sh.off.cpu().numpy()
    return np.repeat(np.arange(len(off) - 1), np.diff(off))


def hit_zr(sh):
    xyz = sh.xyz.cpu().numpy().astype(np.float64)
    return xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])


def n_hits(sh) -> np.ndarray:
    return np.diff(sh.off.cpu().numpy())


def first_layer_index(zb, theta) -> np.ndarray:
    """k = 59.5 - zb cos(theta) / 20.5 (gate0 section 4), rounded to the nearest integer; also the raw value."""
    k = 59.5 - np.asarray(zb, np.float64) * np.cos(np.asarray(theta, np.float64)) / 20.5
    return np.rint(k).astype(int), k


# ---- parts cache -----------------------------------------------------------------------------------------------------
PARTS_FILE = OUT / "parts_val_k64.npz"
TRAIN_NLL_FILE = OUT / "train_nll_k64.npz"


def compute_parts(models, sh, **kw) -> dict:
    """test_nll_multi with return_parts=True inside deterministic() (K=64, seed 12345, batch 128 unless overridden)."""
    return metric.test_nll_multi(models, sh, return_parts=True, deterministic_sum=True, **kw)


def parts_val() -> dict:
    """Validation parts of [mlp, formula] (rows 0, 1); computed by audit_parts.py, recomputed here if missing."""
    if not PARTS_FILE.exists():
        sets, _, mlp, formula = load()
        out = compute_parts([mlp, formula], sets["val"])
        np.savez(PARTS_FILE, **out)
    with np.load(PARTS_FILE) as f:
        return {k: f[k] for k in f.files}


# ---- maps at fixed momenta -------------------------------------------------------------------------------------------
def momenta12(train) -> np.ndarray:
    """12 momenta at event quantiles (j + 1/2)/12 of the training momentum distribution [GeV]."""
    p = train.ev["p"].cpu().numpy().astype(np.float64)
    return np.quantile(p, (np.arange(12) + 0.5) / 12)


@torch.inference_mode()
def logP_at(model, p_values) -> np.ndarray:
    """log P (n, N_Z, N_R) in float64 at momenta p [GeV]."""
    lnp = torch.log(torch.as_tensor(np.asarray(p_values, np.float64), dtype=torch.float32, device=DEV))
    return model(lnp).double().cpu().numpy()


def kl_map(logP, logQ) -> np.ndarray:
    """KL(P || Q) per map, summed over all cells; inputs (n, N_Z, N_R) log-probabilities."""
    P = np.exp(logP)
    return (P * (logP - logQ)).sum((1, 2))


def kl_chain(logP, logQ) -> dict:
    """Exact chain-rule split KL(P||Q) = KL(Pz||Qz) + sum_z Pz KL(T_P(.|z) || T_Q(.|z)) (float64), and the
    per-cell contributions of the conditional part (n, N_Z, N_R)."""
    from scipy.special import logsumexp
    lPz, lQz = logsumexp(logP, axis=2), logsumexp(logQ, axis=2)
    Pz = np.exp(lPz)
    kl_z = (Pz * (lPz - lQz)).sum(1)
    lT, lU = logP - lPz[..., None], logQ - lQz[..., None]
    cell = np.exp(logP) * (lT - lU)                       # Pz * T_P * ln(T_P/T_Q)
    kl_c = cell.sum((1, 2))
    return {"kl": kl_map(logP, logQ), "kl_z": kl_z, "kl_cond": kl_c, "cond_cell": cell}


# ---- zero-training models --------------------------------------------------------------------------------------------
class LogMix(nn.Module):
    """Uniform probability average of several maps: log P = logsumexp_m(log P_m) - ln M."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, lnp):
        lp = torch.stack([m(lnp) for m in self.models], 0)
        return torch.logsumexp(lp, 0) - math.log(len(self.models))


class Hybrid(nn.Module):
    """P = Pz_long(z') * T_lat(r | z') with Pz = sum_r P and T = P / Pz (in log space). Guard: a row in which the
    lateral model has no mass (log Pz = -inf) gets T uniform over the rings; such rows are counted in empty_rows."""

    def __init__(self, long_model, lat_model):
        super().__init__()
        self.long_model, self.lat_model = long_model, lat_model
        self.empty_rows = 0

    def forward(self, lnp):
        lpl, lpt = self.long_model(lnp), self.lat_model(lnp)
        lz_long = torch.logsumexp(lpl, 2, keepdim=True)
        lz_lat = torch.logsumexp(lpt, 2, keepdim=True)
        bad = ~torch.isfinite(lz_lat)
        self.empty_rows += int(bad.sum())
        lT = torch.where(bad, torch.full_like(lpt, -math.log(common.N_R)), lpt - lz_lat)
        out = lz_long + lT
        return out - torch.logsumexp(out.reshape(len(lnp), -1), 1)[:, None, None]   # exact renormalisation (rounding)


# ---- output ----------------------------------------------------------------------------------------------------------
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, torch.Tensor):
        return to_jsonable(o.detach().cpu().numpy())
    return o


class Run:
    """Timer + JSON writer for one audit script."""

    def __init__(self, name: str):
        self.name, self.t0 = name, time.time()
        if DEV.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def write(self, obj: dict):
        obj = dict(obj)
        script = Path(sys.argv[0]).name if sys.argv and sys.argv[0].endswith(".py") else f"audit_{self.name}.py"
        obj["_run"] = {"script": f"scripts/{script}", "seconds": round(time.time() - self.t0, 1),
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "gpu_peak_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if DEV.type == "cuda" else None,
                       "torch": torch.__version__, "device": str(DEV)}
        dest = OUT / f"{self.name}.json"
        dest.write_text(json.dumps(to_jsonable(obj), indent=1) + "\n")
        print(f"wrote {dest.relative_to(paths.REPO_ROOT)} ({obj['_run']['seconds']} s)")
        return dest


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()
