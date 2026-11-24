"""common – shared tools for the shape-form notebooks (importable version of `00_common.ipynb`).

The code is the notebook's code, cell by cell and in the same order. Four things differ from the notebook:

* Settings. The notebook was loaded with `%run -i ./00_common.ipynb` and read its settings from the caller's variables
  (`DATA_DIR = ...` before the `%run` line). A module cannot see the caller's variables, so use `configure(...)`:

      import common
      common.configure(data_dir="data/", out_dir="results")       # where `%run -i ./00_common.ipynb` used to be
      from common import DEVICE, load_data, select_architecture   # ... whatever the notebook needs

  `configure` also does what running the settings cell did: it seeds torch and numpy with `SEED`, creates `OUT_DIR`
  and prints the device and the paths.
* `read_csv` also accepts a `FILE_PATTERN` ending in `.parquet`. The default pattern is still the CSV.
* The overview figure, which the notebook drew when it was opened on its own, is `overview_figure()`;
  `python common.py` draws it.
* Four imports that the notebook made only for the notebooks loading it are gone: `json`, `scipy.optimize.minimize`
  and `IPython.display.Markdown` / `display`. Import them in the notebook that uses them.

Python imports a module only once per kernel: after editing this file, restart the kernel (or `importlib.reload(common)`).

Contents
1. Settings (paths, grid, training).
2. Data: CSV loading, detector geometry, compact event storage, train/val/test split.
3. The common "language": every model returns a probability map in the shower's own coordinates
   (z' depth, r distance from the axis).
4. The detector as a convolution, and the likelihood.
5. Training with simple checkpointing.
6. Plotting (profiles, silhouette, 3D bodies, interactive 3D).
7. Analytic formulas and distillation (symbolic regression); degree and architecture selection; the MLP (`FieldModel`).
8. Export: every model saves its results to a uniform file, which is read by the summary notebook.
"""

import copy
import math
import os
import time
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
# 1. Settings
# ----------------------------------------------------------------------------------------------------------------------
# Five settings can be changed, with `configure(...)` below: DATA_DIR, OUT_DIR, FILE_PATTERN, MAX_EVENTS and BATCH.
# For the 100 000-event run usually only FILE_PATTERN, OUT_DIR and BATCH change. The functions in this module read these
# names when they are called, so call `configure` before loading data or training.
#
# Caution: `from common import OUT_DIR` copies the value at import time. Assigning to that copy in a notebook does not
# change the setting used by this module; only `configure` (or `common.OUT_DIR = ...`) does.

DATA_DIR = "data"
OUT_DIR = "results"
FILE_PATTERN = "{particle}_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.csv"
MAX_EVENTS = None                                    # e.g. 300 for a quick test
BATCH = 32                                           # 100k events, GPU: 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1
TIME_CUT_NS = 100.0                                  # drop very late (slow-neutron) deposits
Z_MIN, Z_MAX, N_Z = -200.0, 1450.0, 128              # depth z' [mm] along the shower axis
R_MAX, N_R = 700.0, 32                               # distance from the axis r [mm]
N_FOLD, N_FOLD_EVAL = 16, 64                         # Monte Carlo points per hit (training / evaluation)

_UNSET = object()                                    # "argument not given" (None is a valid value for MAX_EVENTS)


def configure(data_dir=_UNSET, out_dir=_UNSET, file_pattern=_UNSET, max_events=_UNSET, batch=_UNSET):
    """Change the settings and do what running the notebook's settings cell did.

    Arguments that are left out keep their current value. Like the notebook cell, this seeds the global torch and numpy
    random generators with SEED, creates OUT_DIR and prints the device and the paths. Call it once at the top of a
    notebook, where `%run -i ./00_common.ipynb` used to be.
    """
    global DATA_DIR, OUT_DIR, FILE_PATTERN, MAX_EVENTS, BATCH
    if data_dir is not _UNSET:
        DATA_DIR = data_dir
    if out_dir is not _UNSET:
        OUT_DIR = out_dir
    if file_pattern is not _UNSET:
        FILE_PATTERN = file_pattern
    if max_events is not _UNSET:
        MAX_EVENTS = max_events
    if batch is not _UNSET:
        BATCH = batch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device: {DEVICE} | data: {DATA_DIR} | output: {OUT_DIR}")


# ----------------------------------------------------------------------------------------------------------------------
# 2. Data
# ----------------------------------------------------------------------------------------------------------------------
# One CSV row is one detector-cell hit; an event consists of many rows. The x_local, y_local, z_local columns are in the
# shower's own frame: the origin is the centroid of the first hit layer, the z' axis is the direction of incidence.
#
# Events are stored compactly: the hits of event i are rows off[i]:off[i+1]. The hit weight is the deposited energy,
# normalised to 1 per event, so every event counts equally in the mean shape.
#
# Role of the CSV columns (same for every model):
#   x_local, y_local, z_local     fitted feature: hit position in the shower frame -> z' and r = hypot(x_local, y_local)
#                                 (Showers.xyz, event_nll)
#   energy                        fitted feature: hit weight w, normalised to 1 per event (Showers.w)
#   incoming_momentum             condition: ln p; the model gives the shape as a function of it (ev["lnp"])
#   event_number                  grouping into events: normalisation, per-event NLL (off, b.ev)
#   incoming_theta, incoming_phi  rotation -> the rotated cell box in the detector convolution (ev["R"], fold)
#   z, z_trans (+ incoming_theta) shower start point -> distance to the back face along the axis (z_b), used to
#                                 normalise the likelihood (ev["zb"])
#   x, y, z                       geometry: cell size, layer spacing, back-face position (detector_geometry)
#   earliest_contribution_time    cut: hits later than 100 ns are dropped (TIME_CUT_NS)
#   incoming_momentum_bin         only for stratifying the train/val/test split, and the envelope bins (load_data)
#
# Further cuts: Z_MIN <= z' <= Z_MAX and r < R_MAX. The visible energy (ev["evis"]) does not enter the shape fit; it is
# used only by the summary.


def read_csv(particle):
    """The hit table of one particle type. Not in the notebook: a FILE_PATTERN ending in .parquet is read as parquet."""
    path = os.path.join(DATA_DIR, FILE_PATTERN.format(particle=particle))
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if MAX_EVENTS:
        df = df[df.event_number.isin(df.event_number.unique()[:MAX_EVENTS])]
    return df


def detector_geometry(df):
    """Layer pitch, transverse cell size and back face (global z) estimated from the hit positions."""
    z = np.unique(df.z.round(2))
    dz = np.diff(z)
    d = np.concatenate([np.diff(np.unique(df[c].round(2))) for c in ("x", "y")])
    return {"layer": float(np.median(dz[dz > 1])), "cell": float(np.median(d[(d > 20) & (d < 80)])),
            "z_back": float(z.max() + np.median(dz[dz > 1]) / 2)}


def rotation(theta, phi):
    """Global -> shower frame (the formula from the data description)."""
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    return np.stack([np.stack([ct * cp, ct * sp, -st], -1),
                     np.stack([-sp, cp, 0 * sp], -1),
                     np.stack([st * cp, st * sp, ct], -1)], 1).astype(np.float32)


def _ranges(starts, counts):
    owner = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
    pos = torch.arange(int(counts.sum()), device=counts.device) - (torch.cumsum(counts, 0) - counts)[owner]
    return starts[owner] + pos, owner


class Showers:
    """All events of one particle type (or a subset)."""

    def __init__(self, ev, off, xyz, w, box):
        self.ev, self.off, self.xyz, self.w, self.box = ev, off, xyz, w, box

    @classmethod
    def from_dataframe(cls, df, geo):
        df = df[df.earliest_contribution_time < TIME_CUT_NS].sort_values("event_number", kind="stable")
        r = np.hypot(df.x_local, df.y_local)
        df = df[(df.z_local >= Z_MIN) & (df.z_local <= Z_MAX) & (r < R_MAX)]
        codes, uniq = pd.factorize(df.event_number, sort=True)
        cnt = np.bincount(codes)
        w = df.energy.values / np.bincount(codes, weights=df.energy.values)[codes]
        first = df.groupby("event_number").first().loc[uniq]
        theta, phi = first.incoming_theta.values, first.incoming_phi.values
        z0 = (first.z - first.z_trans).values                       # global z of the shower origin
        ev = {"p": torch.tensor(first.incoming_momentum.values, dtype=torch.float32),
              "bin": torch.tensor(first.incoming_momentum_bin.values.astype(np.int64)),
              "zb": torch.tensor((geo["z_back"] - z0) / np.cos(theta), dtype=torch.float32),   # back face along the axis
              "R": torch.tensor(rotation(theta, phi)),
              "evis": torch.tensor(np.bincount(codes, weights=df.energy.values), dtype=torch.float32)}
        ev["lnp"] = torch.log(ev["p"])
        return cls(ev, torch.tensor(np.concatenate([[0], np.cumsum(cnt)])),
                   torch.tensor(df[["x_local", "y_local", "z_local"]].to_numpy(np.float32)),
                   torch.tensor(w, dtype=torch.float32),
                   torch.tensor([geo["cell"], geo["cell"], geo["layer"]], dtype=torch.float32))

    def __len__(self):
        return len(self.ev["p"])

    def to(self, dev):
        return Showers({k: v.to(dev) for k, v in self.ev.items()}, self.off.to(dev), self.xyz.to(dev),
                       self.w.to(dev), self.box.to(dev))

    def subset(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.off.device)
        cnt = self.off[idx + 1] - self.off[idx]
        hid, _ = _ranges(self.off[idx], cnt)
        off = torch.cat([cnt.new_zeros(1), torch.cumsum(cnt, 0)])
        return Showers({k: v[idx] for k, v in self.ev.items()}, off, self.xyz[hid], self.w[hid], self.box)

    def batch(self, idx):
        hid, owner = _ranges(self.off[idx], self.off[idx + 1] - self.off[idx])
        return SimpleNamespace(lnp=self.ev["lnp"][idx], p=self.ev["p"][idx], R=self.ev["R"][idx],
                               zb=self.ev["zb"][idx], ev=owner, xyz=self.xyz[hid], w=self.w[hid])

    def batches(self, size, shuffle=False, gen=None):
        n = len(self)
        order = (torch.randperm(n, generator=gen) if shuffle else torch.arange(n)).to(self.off.device)
        for s in range(0, n, size):
            yield self.batch(order[s:s + size])

    def hits(self):
        owner = torch.repeat_interleave(torch.arange(len(self), device=self.off.device), self.off[1:] - self.off[:-1])
        xyz = self.xyz.cpu().numpy()
        return pd.DataFrame({"z": xyz[:, 2], "r": np.hypot(xyz[:, 0], xyz[:, 1]), "w": self.w.cpu().numpy(),
                             "p": self.ev["p"][owner].cpu().numpy(), "ev": owner.cpu().numpy()})


def load_data(particle, fracs=(0.70, 0.15, 0.15)):
    """CSV -> (raw DataFrame, geometry, {'train','val','test'} on DEVICE); stratified by 1-GeV momentum bin."""
    df = read_csv(particle)
    geo = detector_geometry(df)
    sh = Showers.from_dataframe(df, geo)
    rng = np.random.default_rng(SEED)
    parts = {"train": [], "val": [], "test": []}
    bins = sh.ev["bin"].numpy()
    for b in np.unique(bins):
        idx = rng.permutation(np.where(bins == b)[0])
        n1, n2 = round(fracs[0] * len(idx)), round((fracs[0] + fracs[1]) * len(idx))
        parts["train"] += idx[:n1].tolist()
        parts["val"] += idx[n1:n2].tolist()
        parts["test"] += idx[n2:].tolist()
    sets = {k: sh.subset(np.sort(v)).to(DEVICE) for k, v in parts.items()}
    print(f"{particle}: {len(sh)} events, geometry {geo}, split " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    return df, geo, sets


# ----------------------------------------------------------------------------------------------------------------------
# 3. The common language: a probability map on the (z', r) grid
# ----------------------------------------------------------------------------------------------------------------------
# The shower is nearly cylindrically symmetric about its axis, so two coordinates suffice: z' (depth) and r (distance
# from the axis). This is divided into a 128 x 32 grid; the r bins are finer near the axis.
#
# Every model does exactly one thing: for a given momentum, ln p, it gives the fraction of the shower energy that falls
# into each grid cell: model(lnp) -> log P, shape (B, N_Z, N_R), sums to 1. Everything else can be computed from this.
#
# Coordinates (shower-local):
#   z' (z_local)  depth along the direction of incidence, in mm. The origin is the centroid of the first hit layer, so
#                 for oblique incidence negative z' also occurs.
#   r             hypot(x_local, y_local): the hit's perpendicular distance from the shower axis (the line of the
#                 incoming particle), in mm. The shape is assumed cylindrically symmetric, so the azimuthal angle does
#                 not appear. 2D plots show r mirrored (+-r).
# The measured r is that of the cell centre (cell ~48 mm), so the measured distribution is coarse; the model is compared
# to it after averaging over the cell.

Z_EDGES = torch.linspace(Z_MIN, Z_MAX, N_Z + 1, device=DEVICE)
R_EDGES = R_MAX * (torch.arange(N_R + 1, device=DEVICE, dtype=torch.float32) / N_R) ** 1.7
DZ = (Z_MAX - Z_MIN) / N_Z
ZC = 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1])
RC = torch.sqrt(0.5 * (R_EDGES[1:] ** 2 + R_EDGES[:-1] ** 2))
LOG_VOL = torch.log(DZ * math.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2))[None, :].expand(N_Z, N_R)   # cell volumes
ZZ, RR = torch.meshgrid(ZC, RC, indexing="ij")


def locate(z, r):
    """Grid cell (iz, ir) of points; points outside are clamped to the border cells."""
    iz = ((z - Z_MIN) / DZ).floor().long().clamp(0, N_Z - 1)
    ir = torch.bucketize(r.contiguous(), R_EDGES[1:-1].contiguous())
    return iz, ir


# ----------------------------------------------------------------------------------------------------------------------
# 4. The detector as a convolution, and the likelihood
# ----------------------------------------------------------------------------------------------------------------------
# The measured hit is the cell centre: the true deposit could have been anywhere in the approx. 48 x 48 x 20.5 mm cell.
# Because the axis is tilted, this box appears rotated in the shower frame. Therefore the model density rho is averaged
# (convolved) over the cell before comparing with the data:
#
#     g(c) = E_{u in cell}[ rho(c + R u) ]          computed per hit with a few random points (Monte Carlo)
#
# Likelihood (per event):  NLL_e = - sum_i w_i [ log g(c_i) - log M_e ],  where M_e is the part of the profile in front
# of the detector back face (the back face lies at a different depth along the axis depending on the angle of
# incidence). Its minimum is exactly the mean normalised shape. Lower NLL = better model.
#
# So what affects the fit (for every model): hit positions (z', r) and weights (w) given the condition ln p; the cell
# size and angle of incidence (convolution); the back-face position (z_b, truncation); and the cuts. The grid resolution
# (128 x 32) limits how fine a detail can be learned.


def fold(b, box, K, gen=None):
    """K random points inside each hit cell (rotated into the shower frame)."""
    u = (torch.rand(len(b.xyz), K, 3, device=b.xyz.device, generator=gen) - 0.5) * box
    return b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", b.R[b.ev], u)


def event_nll(model, b, box, K=N_FOLD, gen=None):
    """Per-event negative log-likelihood (detector-folded, truncated at the back face)."""
    logP = model(b.lnp)
    logrho = logP - LOG_VOL
    pts = fold(b, box, K, gen)
    iz, ir = locate(pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
    logg = torch.logsumexp(logrho[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(K)
    Pz = logP.exp().sum(2)                                            # longitudinal profile
    cdf = torch.cat([Pz.new_zeros(len(Pz), 1), torch.cumsum(Pz, 1)], 1)
    t = ((b.zb - Z_MIN) / DZ).clamp(0, N_Z)
    k = t.floor().long().clamp(max=N_Z - 1)
    ar = torch.arange(len(Pz), device=Pz.device)
    mass = cdf[ar, k] + Pz[ar, k] * (t - k)                           # part of the profile inside the detector
    ll = logg - torch.log(mass.clamp_min(1e-8))[b.ev]
    return -torch.zeros(len(b.lnp), device=ll.device).index_add_(0, b.ev, b.w * ll), logP


@torch.no_grad()
def test_nll(model, data, K=N_FOLD_EVAL):
    """Per-event NLL with a fixed random seed (comparable between models)."""
    gen = torch.Generator(device=DEVICE).manual_seed(12345)
    return torch.cat([event_nll(model, b, data.box, K, gen)[0] for b in data.batches(128)]).cpu().numpy()


# ----------------------------------------------------------------------------------------------------------------------
# 5. Training with simple checkpointing
# ----------------------------------------------------------------------------------------------------------------------
# `fit` saves the model every 5 epochs and at the end to results/<particle>/<name>.pt.
#   - If the file contains a finished training, it is loaded and not retrained.
#   - If it contains an interrupted one, training resumes from there.
#   - Delete the file for full retraining.
#
# Every model minimises the same loss (event_nll, Adam, gradient clipping at 5.0). Differences between models:
#   - lr and epochs differ per model.
#   - With smooth > 0 (neural networks only), a curvature penalty along depth is also applied.
#   - Early stopping: based on the validation NLL (patience); the best validation state is kept.
#   - BATCH and SEED affect the averaging and the random cell points (N_FOLD).


def fit(model, data, path, epochs, lr, patience=15, smooth=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, best, best_state, bad, start = {"train": [], "val": []}, float("inf"), None, 0, 0
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        if ck["done"]:
            model.load_state_dict(ck["model"])
            print(f"loaded finished model from {path}")
            return ck["history"]
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        hist, best, best_state, bad, start = ck["history"], ck["best"], ck["best_state"], ck["bad"], ck["epoch"] + 1
        print(f"resuming {path} from epoch {start}")
        if start >= epochs:                                           # nothing left to do: finish with the best state
            model.load_state_dict(best_state or ck["model"])
            return hist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gen = torch.Generator().manual_seed(SEED + start)
    t0 = time.time()
    for ep in range(start, epochs):
        tot = 0.0
        for b in data["train"].batches(BATCH, shuffle=True, gen=gen):
            nll, logP = event_nll(model, b, data["train"].box)
            loss = nll.mean()
            if smooth:                                                # small curvature penalty for the neural maps
                loss = loss + smooth * ((logP[:, 2:] - 2 * logP[:, 1:-1] + logP[:, :-2]) ** 2).mean()
            if not torch.isfinite(loss):                              # skip a numerically broken batch instead of poisoning the weights
                continue
            opt.zero_grad()
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(nll.detach().sum())
        val = float(test_nll(model, data["val"], N_FOLD).mean())
        hist["train"].append(tot / len(data["train"]))
        hist["val"].append(val)
        if val < best - 1e-4:
            best, bad, best_state = val, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        done = bad >= patience or ep == epochs - 1
        if ep % 10 == 0 or done:
            print(f"epoch {ep:4d}  train {hist['train'][-1]:.4f}  val {val:.4f}  ({time.time() - t0:.0f}s)")
        if (ep + 1) % 5 == 0 or done:
            torch.save({"model": (best_state or model.state_dict()) if done else model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                        "history": hist, "best": best, "best_state": best_state, "bad": bad, "done": done}, path)
        if done:
            break
    if best_state is None:                                            # validation never finite: keep the current weights
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return hist


# ----------------------------------------------------------------------------------------------------------------------
# 6. Plotting
# ----------------------------------------------------------------------------------------------------------------------
#   sample_folded   draws points from the model and passes them through the detector (cell convolution). This can be
#                   compared directly with the measured hits.
#   profile_plot    longitudinal (dE/dz') or lateral (dE/dr) profile, data vs. model, in five momentum bins.
#   shape_map       the model's true (deconvolved) shape as a mirrored silhouette.
#   bodies_3d       nested bodies containing 50/90/95% of the energy (cut open to show the interior).
#   interactive_3d  rotatable 3D page with a momentum slider.

SHOW_BINS = (1, 4, 9, 14, 19)
COLORS = {"data": "black", "gflash": "tab:blue", "cylinder": "tab:blue", "mlp": "tab:orange", "mlp_formula": "tab:red",
          "cnn": "tab:green", "cnn_formula": "tab:purple", "envelope": "tab:brown"}


@torch.no_grad()
def sample_folded(model, data, M=200, seed=7):
    """Deposits drawn from the model for every event, truncated at its back face, folded with its cell."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    out = []
    for b in data.batches(128):
        idx = torch.arange(len(b.lnp), device=DEVICE)
        behind = Z_EDGES[None, :-1] > b.zb[:, None]                  # cells behind this event's back face
        behind[:, 0] = False
        logP = model(b.lnp).masked_fill(behind[:, :, None], -float("inf"))
        cell = torch.multinomial(torch.softmax(logP.reshape(len(idx), -1), 1), M, True, generator=gen)
        iz, ir = cell // N_R, cell % N_R
        u = torch.rand(len(idx), M, 3, device=DEVICE, generator=gen)
        z = Z_EDGES[iz] + u[..., 0] * DZ
        r = torch.sqrt(R_EDGES[ir] ** 2 + u[..., 1] * (R_EDGES[ir + 1] ** 2 - R_EDGES[ir] ** 2))
        pts = torch.stack([r * torch.cos(6.2832 * u[..., 2]), r * torch.sin(6.2832 * u[..., 2]), z], -1)
        box = (torch.rand(len(idx), M, 3, device=DEVICE, generator=gen) - 0.5) * data.box
        pts = pts + torch.einsum("bij,bmj->bmi", b.R, box)
        out.append(pd.DataFrame({"z": pts[..., 2].flatten().cpu().numpy(),
                                 "r": torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2).flatten().cpu().numpy(),
                                 "p": b.p[:, None].expand(-1, M).flatten().cpu().numpy()}))
    s = pd.concat(out, ignore_index=True)
    s["w"] = 1.0 / M
    return s


def profile_plot(hits, samples, which="z", title=""):
    edges = np.linspace(Z_MIN, 1300, 70) if which == "z" else np.linspace(0, 350, 50)
    fig, axes = plt.subplots(1, len(SHOW_BINS), figsize=(3.3 * len(SHOW_BINS), 2.9), sharey=(which == "r"))
    for ax, b in zip(axes, SHOW_BINS):
        for name, s in {"data": hits, **samples}.items():
            m = (s.p >= b) & (s.p < b + 1)
            if m.sum() == 0:
                continue
            h, _ = np.histogram(s[which][m], edges, weights=s.w[m])
            ax.step(edges[:-1], h / h.sum() / np.diff(edges), where="post", color=COLORS.get(name), lw=2 if name == "data" else 1.2,
                    label=name)
        ax.set_title(f"{b}–{b + 1} GeV", fontsize=9)
        ax.set_xlabel("z' [mm]" if which == "z" else "r [mm]")
        if which == "r":
            ax.set_yscale("log")
    axes[0].set_ylabel("dE/dz' (norm.)" if which == "z" else "dE/dr (norm.)")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def shape_map(model, p_values=(1, 5, 10, 18), title="", r_show=150):
    lnp = torch.log(torch.tensor(p_values, dtype=torch.float32, device=DEVICE))
    dens = (model(lnp) - LOG_VOL).exp().cpu().numpy()
    rs = np.linspace(-r_show, r_show, 121)
    fig, axes = plt.subplots(1, len(p_values), figsize=(3.6 * len(p_values), 2.6))
    for ax, d, p in zip(axes, dens, p_values):
        img = np.log10(np.clip(np.stack([np.interp(np.abs(rs), RC.cpu().numpy(), row) for row in d], 1), 1e-14, None))
        ax.imshow(img, origin="lower", aspect="auto", extent=[Z_MIN, Z_MAX, -r_show, r_show], cmap="magma",
                  vmin=img.max() - 5, vmax=img.max())
        ax.set_title(f"p = {p:g} GeV", fontsize=9)
        ax.set_xlabel("z' [mm]")
    axes[0].set_ylabel("r [mm] (mirrored)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()


def iso_radius(P, f):
    """Outer radius r_f(z') of the smallest region holding fraction f of the energy (P: cell probabilities)."""
    dens = P / np.exp(LOG_VOL.cpu().numpy())
    order = np.argsort(-dens.ravel())
    k = min(np.searchsorted(np.cumsum(P.ravel()[order]), f), len(order) - 1)
    mask = dens >= dens.ravel()[order[k]]
    r_up = R_EDGES[1:].cpu().numpy()
    return np.array([r_up[np.nonzero(row)[0].max()] if row.any() else 0.0 for row in mask])


def revolve(z, r, phi0=0.5 * math.pi, phi1=2 * math.pi, n=36):
    Z, PH = np.meshgrid(z, np.linspace(phi0, phi1, n), indexing="ij")
    return Z, r[:, None] * np.cos(PH), r[:, None] * np.sin(PH)


@torch.no_grad()
def bodies_3d(models, p_values=(1.5, 4.5, 9.5, 14.5, 19.5), fractions=(0.5, 0.9, 0.95), title=""):
    colors = {0.5: "#b2182b", 0.9: "#ef8a62", 0.95: "#67a9cf"}
    zc = ZC.cpu().numpy()
    lnp = torch.log(torch.tensor(p_values, dtype=torch.float32, device=DEVICE))
    P = {n: m(lnp).exp().cpu().numpy() for n, m in models.items()}
    radii = {(n, j, f): iso_radius(P[n][j], f) for n in models for j in range(len(p_values)) for f in fractions}
    rmax = max(r.max() for r in radii.values())
    zs = [zc[r > 0] for r in radii.values() if (r > 0).any()]
    zlim = (min(z.min() for z in zs) - 30, max(z.max() for z in zs) + 30)
    fig = plt.figure(figsize=(4.0 * len(p_values), 3.6 * len(models)))
    for i, n in enumerate(models):
        for j, p in enumerate(p_values):
            ax = fig.add_subplot(len(models), len(p_values), i * len(p_values) + j + 1, projection="3d")
            for f in fractions:
                ax.plot_surface(*revolve(zc, radii[(n, j, f)]), color=colors[f], alpha=0.35, linewidth=0)
            ax.set_xlim(*zlim); ax.set_ylim(-rmax, rmax); ax.set_zlim(-rmax, rmax)
            ax.set_box_aspect((min((zlim[1] - zlim[0]) / (2 * rmax), 2.2), 1, 1))
            ax.set_title(f"{n}, p = {p:g} GeV", fontsize=9)
            ax.set_xlabel("z' [mm]", fontsize=7); ax.tick_params(labelsize=6)
            ax.view_init(elev=24, azim=-52)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=colors[f]) for f in fractions],
               [f"{f:.0%} of the energy" for f in fractions], loc="lower center", ncol=3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.show()


@torch.no_grad()
def interactive_3d(models, data=None, envelope=None, p_values=None, fraction=0.9, path=None, title=""):
    """Rotatable 3D page: the f-containment body of every model, a slider over p, one real shower, the envelope."""
    import plotly.graph_objects as go
    p_values = p_values or [b + 0.5 for b in range(0, 20, 2)]
    zc = ZC.cpu().numpy()
    fig, groups = go.Figure(), []
    for p in p_values:
        ids = []
        lnp = torch.tensor([math.log(p)], device=DEVICE)
        for n, m in models.items():
            X, Y, Z = revolve(zc, iso_radius(m(lnp)[0].exp().cpu().numpy(), fraction), 0, 2 * math.pi, 48)
            c = matplotlib.colors.to_hex(COLORS.get(n, "grey"))
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, c], [1, c]], showscale=False, opacity=0.35,
                                     name=n, showlegend=True, visible=False))
            ids.append(len(fig.data) - 1)
        if envelope is not None and int(p) in envelope:
            e = envelope[int(p)]
            zz = np.linspace(-150, 900, 160)
            u = np.clip(zz - e["z0"], 0, None)
            X, Y, Z = revolve(zz, e["A"] * u ** e["a"] * np.exp(-e["b"] * u), 0, 2 * math.pi, 48)
            c = matplotlib.colors.to_hex(COLORS["envelope"])
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, c], [1, c]], showscale=False, opacity=0.2,
                                     name="envelope", showlegend=True, visible=False))
            ids.append(len(fig.data) - 1)
        if data is not None:
            e = np.where(data.ev["bin"].cpu().numpy() == int(p))[0]
            if len(e):
                h = slice(int(data.off[e[0]]), int(data.off[e[0] + 1]))
                xyz, w = data.xyz[h].cpu().numpy(), data.w[h].cpu().numpy()
                fig.add_trace(go.Scatter3d(x=xyz[:, 2], y=xyz[:, 0], z=xyz[:, 1], mode="markers", visible=False,
                                           name=f"one shower (p={float(data.ev['p'][e[0]]):.1f})",
                                           marker=dict(size=2 + 30 * np.sqrt(w), color=np.log10(w + 1e-6), colorscale="Inferno")))
                ids.append(len(fig.data) - 1)
        groups.append(ids)
    steps = []
    for k, p in enumerate(p_values):
        vis = [False] * len(fig.data)
        for i in groups[k]:
            vis[i] = True
        steps.append(dict(method="update", args=[{"visible": vis}], label=f"{p:g} GeV"))
    for i in groups[0]:
        fig.data[i].visible = True
    fig.update_layout(sliders=[dict(active=0, steps=steps, currentvalue={"prefix": "p = "})], height=650,
                      scene=dict(xaxis_title="z' [mm]", yaxis_title="x' [mm]", zaxis_title="y' [mm]", aspectmode="data"),
                      title=title or f"{fraction:.0%} containment bodies", margin=dict(l=0, r=0, t=40, b=0))
    if path:
        fig.write_html(path, include_plotlyjs=True)
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# 7. Analytic formulas and distillation
# ----------------------------------------------------------------------------------------------------------------------
# A formula has two parts:  rho(r, z' | p) = L(z' | p) T(r | z', p).
#   Longitudinal L  `gamma` (Gamma distribution: rise, maximum, decay) or `plateau` (cylinder: fast rise, nearly constant
#                   section, cut-off at the range).
#   Lateral T       (normalised per unit area) `core_tail` (two Cauchy terms: narrow core + wide tail, standard in
#                   GFlash) or `gauss_exp` (Gaussian core + exponential tail).
#
# Every parameter is a small polynomial in L = ln(p/GeV): theta(p) = c0 + c1 L + c2 L^2 + ... This is the energy
# dependence. The lateral parameters may also vary linearly with depth.
#
# Distillation: a few formulas are fitted (KL divergence) to the trained network's deconvolved maps, and the simplest one
# that is close enough to the best is kept. Distillation does not fit the data but the trained network's map: the (z', r)
# grid probabilities at 30 ln p values (between the 2-98% quantiles of the training data). This step has no detector
# convolution. The map is used only up to depth z_cut (the depth still covered by the back face for 90% of training
# events), because beyond it the network is not constrained by data. Initial values come from the mean network map. The
# formula family and degree are chosen by KL and parameter count (`tol`).


def _cauchy(r, R):
    return 2 * torch.log(R) - math.log(math.pi) - 2 * torch.log(r * r + R * R)


def _gauss(r, s):
    return -r * r / (2 * s * s) - math.log(2 * math.pi) - 2 * torch.log(s)


def _exp2(r, lam):
    return -r / lam - math.log(2 * math.pi) - 2 * torch.log(lam)


def long_gamma(t, z):
    s = F.softplus((z - t["z0"]) / 2) * 2 + 1e-3
    a = 1 + torch.exp(t["log_am1"])
    return (a - 1) * torch.log(s) - (a - 1) / torch.exp(t["log_tmax"]) * s


def long_plateau(t, z):
    return (F.logsigmoid((z - t["zs"]) / torch.exp(t["log_w1"])) + F.logsigmoid((torch.exp(t["log_ell"]) - z) / torch.exp(t["log_w2"]))
            + t["kappa"] * z / 1000)


def lat_core_tail(t, r):
    rc = torch.exp(t["log_rc"])
    lpc, lqc = F.logsigmoid(t["logit_pc"]), F.logsigmoid(-t["logit_pc"])      # log pc, log(1-pc), stable when pc -> 0 or 1
    return torch.logaddexp(lpc + _cauchy(r, rc), lqc + _cauchy(r, rc * (1 + torch.exp(t["log_ratio"]))))


def lat_gauss_exp(t, r):
    lpc, lqc = F.logsigmoid(t["logit_pc"]), F.logsigmoid(-t["logit_pc"])
    return torch.logaddexp(lpc + _gauss(r, torch.exp(t["log_sig"])), lqc + _exp2(r, torch.exp(t["log_lam"])))


# name: (parameter names, default raw values, function, LaTeX)
LONG = {"gamma": (["z0", "log_tmax", "log_am1"], [-20.0, math.log(100), math.log(3)], long_gamma,
                  r"L \propto s^{\alpha-1}e^{-(\alpha-1)s/t_{max}},\; s=z'-z_0,\; \alpha=1+e^{\mathrm{log\_am1}},\; t_{max}=e^{\mathrm{log\_tmax}}"),
        "plateau": (["zs", "log_w1", "log_ell", "log_w2", "kappa"], [-10.0, math.log(15), math.log(3000), math.log(30), 0.0],
                    long_plateau, r"L \propto \sigma\!\left(\frac{z'-z_s}{w_1}\right)\sigma\!\left(\frac{\ell-z'}{w_2}\right)e^{\kappa z'/1000}")}
LAT = {"core_tail": (["log_rc", "log_ratio", "logit_pc"], [math.log(10), math.log(3), 0.85], lat_core_tail,
                     r"T = \frac{p_c}{\pi}\frac{R_C^2}{(r^2+R_C^2)^2}+\frac{1-p_c}{\pi}\frac{R_T^2}{(r^2+R_T^2)^2}"),
       "gauss_exp": (["log_sig", "log_lam", "logit_pc"], [math.log(10), math.log(40), 0.4], lat_gauss_exp,
                     r"T = p_c\frac{e^{-r^2/2\sigma^2}}{2\pi\sigma^2}+(1-p_c)\frac{e^{-r/\lambda}}{2\pi\lambda^2}")}


class Formula(nn.Module):
    """rho = L(z'|p) T(r|z',p); every parameter = polynomial in ln p (+ depth slope for the lateral ones)."""

    def __init__(self, long, lat, degree=1, depth=True, lnp0=0.0, lnp_s=1.0):
        super().__init__()
        # polynomial variable L = (ln p - lnp0) / lnp_s; centring/scaling keeps high degrees well conditioned
        self.register_buffer("lnp0", torch.tensor(float(lnp0)))
        self.register_buffer("lnp_s", torch.tensor(float(lnp_s)))
        self.long, self.lat, self.degree = long, lat, degree
        self.names = LONG[long][0] + LAT[lat][0]
        coef = torch.zeros(len(self.names), degree + 1)
        coef[:, 0] = torch.tensor(LONG[long][1] + LAT[lat][1])
        self.coef = nn.Parameter(coef)
        self.slope = nn.Parameter(torch.zeros(len(LAT[lat][0]))) if depth else None

    def params(self, lnp):
        x = (lnp - self.lnp0) / self.lnp_s
        return torch.stack([x ** k for k in range(self.degree + 1)], 1) @ self.coef.T            # (B, n_params)

    def _load_from_state_dict(self, sd, prefix, *args, **kw):
        # checkpoints saved before the centring existed: L = ln p
        sd.setdefault(prefix + "lnp0", torch.tensor(0.0))
        sd.setdefault(prefix + "lnp_s", torch.tensor(1.0))
        super()._load_from_state_dict(sd, prefix, *args, **kw)

    def forward(self, lnp):
        th = self.params(lnp)[:, :, None, None]
        nl = len(LONG[self.long][0])
        Z, R = ZZ[None], RR[None]
        tl = {n: th[:, i] for i, n in enumerate(LONG[self.long][0])}
        tt = {n: th[:, nl + j] + (self.slope[j] * Z / 1000 if self.slope is not None else 0)
              for j, n in enumerate(LAT[self.lat][0])}
        logd = LONG[self.long][2](tl, Z) + LAT[self.lat][2](tt, R) + LOG_VOL
        return F.log_softmax(logd.reshape(len(lnp), -1), 1).reshape(len(lnp), N_Z, N_R)

    def init_from(self, z, r, w):
        """Starting values from the energy-weighted mean depth, its spread and the median radius."""
        w = w / w.sum()
        m = float((w * z).sum())
        v = max(float((w * (z - m) ** 2).sum()), 25.0)
        rmed = max(float(np.interp(0.5, np.cumsum(w[np.argsort(r)]), np.sort(r))), 2.0)
        c = dict(zip(self.names, self.coef.data[:, 0].tolist()))
        if self.long == "gamma":
            s = max(m + 20, 20)
            a = min(max(s * s / v, 1.3), 60)
            c.update(z0=-20.0, log_am1=math.log(a - 1), log_tmax=math.log(max((a - 1) * v / s, 5)))
        else:
            c.update(log_ell=math.log(max(float(np.interp(0.97, np.cumsum(w[np.argsort(z)]), np.sort(z))), 30)))
        if self.lat == "core_tail":
            c.update(log_rc=math.log(0.5 * rmed))
        else:
            c.update(log_sig=math.log(rmed / 1.18), log_lam=math.log(rmed))
        self.coef.data[:, 0] = torch.tensor([c[n] for n in self.names])
        return self

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def variable_md(self):
        c, s = float(self.lnp0), float(self.lnp_s)
        if c == 0 and s == 1:
            return "$L=\\ln(p/\\mathrm{GeV})$"
        return f"$L=(\\ln(p/\\mathrm{{GeV}}) {-c:+.4g})/{s:.4g}$"

    def describe(self, title=""):
        L = "L"
        rows = []
        for i, n in enumerate(self.names):
            terms = " ".join(f"{float(v):+.4g}" + ("" if k == 0 else rf"\,{L}" if k == 1 else rf"\,{L}^{k}")
                             for k, v in enumerate(self.coef[i]))
            j = i - len(LONG[self.long][0])
            slope = f"{float(self.slope[j]):+.4g}" if (self.slope is not None and j >= 0) else "–"
            rows.append(f"| `{n}` | ${terms}$ | {slope} |")
        return "\n".join([f"### {title}", "", f"$${LONG[self.long][3]}$$", "", f"$${LAT[self.lat][3]}$$", "",
                          "| parameter | value, " + self.variable_md() + " | depth slope (× z'/1000) |", "|---|---|---|"]
                         + rows + ["", f"Number of parameters: **{self.n_params()}**"])


@torch.no_grad()
def formula_curves(f, title=""):
    """Physical value of every formula parameter as a function of p (log_ -> exp, logit_ -> sigmoid)."""
    p = torch.logspace(math.log10(0.3), math.log10(20), 60, device=DEVICE)
    th = f.params(torch.log(p)).cpu().numpy()
    n = len(f.names)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.4))
    for i, (ax, name) in enumerate(zip(axes, f.names)):
        v = np.exp(th[:, i]) if name.startswith("log_") else 1 / (1 + np.exp(-th[:, i])) if name.startswith("logit_") else th[:, i]
        ax.plot(p.cpu().numpy(), v)
        ax.set_xscale("log")
        ax.set_title(name.replace("log_", "").replace("logit_", ""), fontsize=9)
        ax.set_xlabel("p [GeV]", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()


def distill(net, train, candidates, degrees=(1, 2), steps=800, tol=0.005):
    """Fit candidate formulas to the network's de-convolved maps on a p grid; keep the simplest good one.

    Only depths seen by >= 90 % of the training events are used (beyond the back face the network is unconstrained)."""
    lnp = torch.linspace(*np.quantile(train.ev["lnp"].cpu().numpy(), [0.02, 0.98]).tolist(), 30, device=DEVICE)
    with torch.no_grad():
        P = net(lnp).exp().reshape(len(lnp), -1)
    z_cut = float(torch.quantile(train.ev["zb"], 0.10))
    cols = (ZZ.reshape(-1) < z_cut)
    Pt = P[:, cols] / P[:, cols].sum(1, keepdim=True)
    H = -torch.special.xlogy(Pt, Pt).sum(1)
    Pm = P.mean(0).reshape(N_Z, N_R).cpu().numpy()
    z0, r0 = ZZ.cpu().numpy().ravel(), RR.cpu().numpy().ravel()
    rows, fits = [], []
    for long, lat in candidates:
        for deg in degrees:
            f = Formula(long, lat, deg).init_from(z0, r0, Pm.ravel()).to(DEVICE)
            opt = torch.optim.Adam(f.parameters(), lr=0.03)
            kl_fn = lambda: (-(Pt * F.log_softmax(f(lnp).reshape(len(lnp), -1)[:, cols], 1)).sum(1) - H).mean()
            for _ in range(steps):
                kl = kl_fn()
                if not torch.isfinite(kl):                            # skip a broken step instead of poisoning Adam
                    continue
                opt.zero_grad()
                kl.backward()
                if all(torch.isfinite(p.grad).all() for p in f.parameters() if p.grad is not None):
                    opt.step()
            with torch.no_grad():
                kl = kl_fn()                                          # KL of the final parameters
            rows.append({"longitudinal": long, "lateral": lat, "degree": deg, "params": f.n_params(), "KL": float(kl)})
            fits.append(f)
            print(f"  {long:8s} x {lat:9s} degree {deg}: {f.n_params():2d} params, KL = {float(kl):.4f}")
    table = pd.DataFrame(rows)
    ok = table[table.KL <= table.KL.min() + tol]
    best = ok.sort_values(["params", "KL"]).index[0]
    return fits[best], table, z_cut


# ----------------------------------------------------------------------------------------------------------------------
# Degree selection (hyperparameter optimisation) – common protocol
# ----------------------------------------------------------------------------------------------------------------------
# Every model uses the same split (load_data: 70/15/15%, stratified by momentum bin, fixed SEED), with the same roles:
#   train       fitting the parameters;
#   validation  early stopping and selecting the hyperparameter (here the degree n of the ln p polynomial);
#   test        only for measuring the final, selected model. It plays no part in the selection.
#
# select_degree trains the formula for every n. The variable is L = (ln p - c)/s, where c and s are the centre and
# half-width of the training set's ln p range, so L is in [-1, 1]. This keeps high powers well conditioned. The degree-n
# model starts from the degree-(n-1) solution (new coefficients zero), so the degrees are nested.
#
# Selection rule (1-SE rule): take n* with the lowest validation NLL. The chosen n is the smallest degree whose per-event
# (paired) validation NLL difference from n* is below one standard error. Thus we only move to a higher degree if the
# improvement is statistically significant. This guards against overfitting, since overfitting shows on validation.
#
# The same for the networks (select_architecture). For neural models the hyperparameter is not the degree but the
# architecture (number of layers, neurons). Every candidate trains on the same split; the validation NLL is compared
# event-by-event (paired) with the best (c*), and the architecture with the fewest parameters within one standard error
# of c* is chosen. This also answers "is the network large enough": if larger networks no longer improve significantly on
# validation, capacity is saturated.


def select_degree(long, lat, data, particle, name, degrees=range(1, 7), epochs=150, lr=0.03, init_hits=None):
    """Train Formula(long, lat, n) for every n (warm start from n-1), choose n on the validation set (1-SE rule).

    Returns (best model, table, {n: model}). Checkpoints: results/<particle>/<name>_n<n>.pt."""
    lnp_tr = data["train"].ev["lnp"].cpu().numpy()
    c, s = float((lnp_tr.max() + lnp_tr.min()) / 2), float((lnp_tr.max() - lnp_tr.min()) / 2)
    h = init_hits if init_hits is not None else data["train"].hits()
    models, val, rows, prev = {}, {}, [], None
    for n in degrees:
        m = Formula(long, lat, degree=n, lnp0=c, lnp_s=s).init_from(h.z.values, h.r.values, h.w.values).to(DEVICE)
        if prev is not None:                                               # warm start: same function as degree n-1
            with torch.no_grad():
                m.coef.zero_()
                m.coef[:, :prev.coef.shape[1]] = prev.coef
                m.slope.copy_(prev.slope)
        print(f"--- degree {n} ---")
        hist = fit(m, data, f"{OUT_DIR}/{particle}/{name}_n{n}.pt", epochs=epochs, lr=lr)
        m.eval()
        val[n] = test_nll(m, data["val"])
        rows.append({"n": n, "params": m.n_params(), "train NLL": float(test_nll(m, data["train"]).mean()),
                     "val NLL": float(val[n].mean()), "epochs": len(hist["val"])})
        models[n] = prev = m
    tab = pd.DataFrame(rows).set_index("n")
    n_star = int(tab["val NLL"].idxmin())
    d = {n: val[n] - val[n_star] for n in models}
    tab["Δval vs n*"] = [float(d[n].mean()) for n in tab.index]
    tab["SE"] = [float(d[n].std(ddof=1) / np.sqrt(len(d[n]))) if n != n_star else 0.0 for n in tab.index]
    ok = [n for n in tab.index if n == n_star or tab.loc[n, "Δval vs n*"] < tab.loc[n, "SE"]]
    n_sel = int(min(ok))
    tab["chosen"] = ["◀" if n == n_sel else "" for n in tab.index]
    print(f"lowest validation NLL: n* = {n_star};  chosen by the 1-SE rule: n = {n_sel}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].plot(tab.index, tab["train NLL"] - tab["train NLL"].iloc[0], "o-", label="train")
    ax[0].plot(tab.index, tab["val NLL"] - tab["val NLL"].iloc[0], "o-", label="validation")
    ax[0].set(xlabel="degree n", ylabel="NLL − NLL(n=1)", title="Lower = better"); ax[0].legend()
    ax[1].errorbar(tab.index, tab["Δval vs n*"], tab["SE"], fmt="o-", capsize=3)
    ax[1].axhline(0, c="grey", lw=0.8); ax[1].axvline(n_sel, ls=":", c="r", label=f"chosen n = {n_sel}")
    ax[1].set(xlabel="degree n", ylabel="Δ validation NLL vs n*", title="Paired difference ± 1 SE"); ax[1].legend()
    plt.tight_layout(); plt.show()
    return models[n_sel], tab, models


def select_architecture(make, configs, data, particle, name, epochs=60, lr=3e-3, smooth=0.0, label=None):
    """Train make(**cfg) for every cfg, choose on the validation set with the 1-SE rule (fewest parameters within 1 SE of the best).

    Returns (chosen cfg, table, {key: model}). Checkpoints: results/<particle>/<name>_<key>.pt (finished ones are only loaded)."""
    label = label or (lambda cfg: "_".join(f"{k}{v}" for k, v in cfg.items()))
    models, val, rows = {}, {}, []
    for cfg in configs:
        key = label(cfg)
        m = make(**cfg)
        print(f"--- {key} ---")
        hist = fit(m, data, f"{OUT_DIR}/{particle}/{name}_{key}.pt", epochs=epochs, lr=lr, smooth=smooth)
        m.eval()
        val[key] = test_nll(m, data["val"])
        rows.append({"config": key, **cfg, "params": sum(p.numel() for p in m.parameters()),
                     "train NLL": float(test_nll(m, data["train"]).mean()), "val NLL": float(val[key].mean()),
                     "epochs": len(hist["val"])})
        models[key] = m
    tab = pd.DataFrame(rows).set_index("config")
    k_star = tab["val NLL"].idxmin()
    d = {k: val[k] - val[k_star] for k in models}
    tab["Δval vs best"] = [float(d[k].mean()) for k in tab.index]
    tab["SE"] = [float(d[k].std(ddof=1) / np.sqrt(len(d[k]))) if k != k_star else 0.0 for k in tab.index]
    ok = tab[(tab.index == k_star) | (tab["Δval vs best"] < tab["SE"])]
    k_sel = ok.sort_values(["params", "val NLL"]).index[0]
    tab["chosen"] = ["◀" if k == k_sel else "" for k in tab.index]
    print(f"lowest validation NLL: {k_star};  chosen by the 1-SE rule: {k_sel}")
    t = tab.sort_values("params")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].plot(t["params"], t["train NLL"] - t["train NLL"].iloc[0], "o-", label="train")
    ax[0].plot(t["params"], t["val NLL"] - t["val NLL"].iloc[0], "o-", label="validation")
    ax[0].set(xscale="log", xlabel="trainable parameters", ylabel="NLL − NLL(smallest)", title="Lower = better"); ax[0].legend()
    ax[1].errorbar(t["params"], t["Δval vs best"], t["SE"], fmt="o-", capsize=3)
    for k, row in t.iterrows():
        ax[1].annotate(k, (row["params"], row["Δval vs best"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].axhline(0, c="grey", lw=0.8); ax[1].axvline(t.loc[k_sel, "params"], ls=":", c="r", label=f"chosen {k_sel}")
    ax[1].set(xscale="log", xlabel="trainable parameters", ylabel="Δ validation NLL vs best", title="Paired difference ± 1 SE"); ax[1].legend()
    plt.tight_layout(); plt.show()
    return next(cfg for cfg in configs if label(cfg) == k_sel), tab, models


# ----------------------------------------------------------------------------------------------------------------------
# Common definition of the neural models
# ----------------------------------------------------------------------------------------------------------------------
# FieldModel (MLP, 05) is defined here so that other notebooks (e.g. 06, which checks the CNN, and the summary 07) can
# load the already trained, selected MLP (load_mlp). The selected architecture is in
# results/<particle>/mlp_architecture.csv.


class FieldModel(nn.Module):
    """MLP(z', r, ln p) -> logit for every grid cell -> softmax."""

    def __init__(self, hidden=64, n_layers=3, n_freq=4):
        super().__init__()
        zn = (ZZ - Z_MIN) / (Z_MAX - Z_MIN) * 2 - 1          # depth scaled to [-1, 1]
        rn = torch.sqrt(RR / R_MAX)                            # radius, stretched near the axis
        feats = [zn, rn]
        for k in range(1, n_freq + 1):                         # sin/cos features: help to learn sharp structures
            feats += [torch.sin(math.pi * k * zn), torch.cos(math.pi * k * zn),
                      torch.sin(math.pi * k * rn), torch.cos(math.pi * k * rn)]
        self.register_buffer("static", torch.stack(feats, -1))
        layers, d = [], self.static.shape[-1] + 1
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, lnp):
        x = torch.cat([self.static[None].expand(len(lnp), -1, -1, -1),
                       (lnp / 3)[:, None, None, None].expand(-1, N_Z, N_R, 1)], -1)
        logits = self.mlp(x).squeeze(-1)
        return F.log_softmax(logits.reshape(len(lnp), -1), 1).reshape(len(lnp), N_Z, N_R)


def load_mlp(particle):
    """The MLP chosen in 05_mlp_<particle> (architecture from mlp_architecture.csv), or None."""
    path = f"{OUT_DIR}/{particle}/mlp_architecture.csv"
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path).drop_duplicates("config")
    row = t[t["chosen"] == "◀"].iloc[-1]
    m = FieldModel(int(row["hidden"]), int(row["n_layers"])).to(DEVICE)
    m.load_state_dict(torch.load(f"{OUT_DIR}/{particle}/mlp_{row['config']}.pt", map_location=DEVICE, weights_only=False)["model"])
    return m.eval()


# ----------------------------------------------------------------------------------------------------------------------
# 8. Export – what the summary notebook reads
# ----------------------------------------------------------------------------------------------------------------------
# At the end, every model notebook saves a uniform file (results/<particle>/export_<name>.pt):
#   - the model's probability map on a dense p grid (0.1-20 GeV), from which any p can be interpolated;
#   - the per-event test NLL (for paired comparison);
#   - the formula description (if any).
# This way the summary notebook does not need to know the models' code.

EXPORT_LNP = torch.linspace(math.log(0.1), math.log(20.0), 80)


@torch.no_grad()
def export(particle, name, model, data, formula_md=None, extra=None):
    model.eval()
    obj = {"name": name, "lnp": EXPORT_LNP, "P": model(EXPORT_LNP.to(DEVICE)).exp().float().cpu(),
           "test_nll": test_nll(model, data["test"]), "formula": formula_md, "extra": extra,
           "n_params": sum(p.numel() for p in model.parameters())}
    os.makedirs(os.path.join(OUT_DIR, particle), exist_ok=True)
    torch.save(obj, os.path.join(OUT_DIR, particle, f"export_{name}.pt"))
    print(f"exported {particle}/{name}: test NLL = {obj['test_nll'].mean():.4f}")


class TableModel(nn.Module):
    """A model rebuilt from an export file: interpolates the stored maps in ln p."""

    def __init__(self, obj):
        super().__init__()
        self.name, self.n_params, self.formula = obj["name"], obj["n_params"], obj["formula"]
        self.register_buffer("lnp", obj["lnp"].to(DEVICE))
        self.register_buffer("P", obj["P"].to(DEVICE))

    def forward(self, lnp):
        x = lnp.clamp(self.lnp[0], self.lnp[-1])
        i = torch.searchsorted(self.lnp, x.contiguous()).clamp(1, len(self.lnp) - 1)
        t = ((x - self.lnp[i - 1]) / (self.lnp[i] - self.lnp[i - 1]))[:, None, None]
        P = self.P[i - 1] * (1 - t) + self.P[i] * t
        return torch.log(P.clamp_min(1e-30) / P.sum((1, 2), keepdim=True))


def load_exports(particle):
    d = os.path.join(OUT_DIR, particle)
    return {f[7:-3]: torch.load(os.path.join(d, f), weights_only=False) for f in sorted(os.listdir(d))
            if f.startswith("export_") and f.endswith(".pt")} if os.path.isdir(d) else {}


# ----------------------------------------------------------------------------------------------------------------------
# Overview figure
# ----------------------------------------------------------------------------------------------------------------------


def overview_figure():
    """The shower frame and the detector cell (the notebook drew this when it was opened on its own).

    Left: detector layers (vertical bands), the axis of the obliquely incident particle, and the shower frame (z' along
    the axis, r perpendicular). Right: a cell in the shower frame. The measured point is the cell centre (black); the
    true deposit could have been anywhere in the cell (grey points). Therefore the model is compared with the data after
    averaging over the cell.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k in range(12):
        ax[0].axvspan(k * 20.5, k * 20.5 + 4, color="#9ecae1", alpha=0.6)
    th = 0.45
    t = np.linspace(0, 240, 50)
    ax[0].plot(t * np.cos(th), 60 + t * np.sin(th), "r-", lw=2, label="particle axis (z')")
    ax[0].annotate("", xy=(60 * np.cos(th) - 25 * np.sin(th), 60 + 60 * np.sin(th) + 25 * np.cos(th)),
                   xytext=(60 * np.cos(th), 60 + 60 * np.sin(th)), arrowprops=dict(arrowstyle="->", color="k"))
    ax[0].text(40, 120, "r", fontsize=12)
    rng = np.random.default_rng(0)
    s = rng.gamma(3, 25, 400)
    q = rng.normal(0, 8 + 0.08 * s)
    ax[0].scatter(s * np.cos(th) - q * np.sin(th), 60 + s * np.sin(th) + q * np.cos(th), s=4, c="k", alpha=0.4)
    ax[0].set_xlabel("global z [mm] (layers in blue)")
    ax[0].set_title("Shower frame: z' along the axis, r across it")
    ax[0].legend(loc="lower right")
    c = np.array([0.0, 0.0])
    box = np.array([[-24, -10], [24, -10], [24, 10], [-24, 10], [-24, -10]])
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ax[1].plot(*(box @ rot.T).T, "b-")
    pts = (rng.uniform(-0.5, 0.5, (300, 2)) * [48, 20]) @ rot.T
    ax[1].scatter(*pts.T, s=6, c="grey", alpha=0.6, label="possible true positions")
    ax[1].scatter(*c, s=80, c="k", label="measured: cell centre")
    ax[1].set_aspect("equal")
    ax[1].set_title("Detector = convolution with the (rotated) cell")
    ax[1].set_xlabel("x' [mm]"); ax[1].set_ylabel("z' [mm]")
    ax[1].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    overview_figure()
