"""R2_numbers review helpers (independent re-evaluation).

- the report metric is computed with 00_common's own `test_nll` (unchanged notebook code via protonlab.load_common);
- the 3D (slab) metric is computed with an independent implementation of the slab normalisation:
  M_e = sum_cells P(cell) * A_e(cell), A_e = volume fraction of the cell's ring inside -h <= z' cos(t) - x' sin(t) <= D,
  integrated with an nz x nr sub-cell midpoint rule (equal-area in r) and the exact azimuthal fraction (asin form);
  the per-hit folded density g uses the notebook's `fold`/`locate` with the same seed/batching as `test_nll`
  (so per-event values are paired with every other model and with the report metric).
"""
import math
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import protonlab
import shapelab
from proton_shape import ProtonShape

OUT = os.path.join(ROOT, "review", "R2_numbers", "out")
FINAL = os.path.join(ROOT, "results", "final", "proton")
G2 = os.path.join(ROOT, "lanes", "G2_reference")


def setup():
    C = protonlab.load_common(OUT_DIR=os.path.join(OUT, "common_out"), BATCH=256)
    geo, data = protonlab.load_proton(C)
    exec(shapelab.SLAB_GEOM_SRC, C)          # only adds slab_accept / event_slab (needed for shapelab.mean_acceptance)
    return C, geo, data


def strip(sd):
    return {k[5:]: v for k, v in sd.items() if k.startswith("base.")}


def ld(p, dev):
    return torch.load(p, map_location=dev, weights_only=False)


DEG = {1: dict(degree=1, degrees={"logit_tr": 2}), **{n: dict(degree=n) for n in range(2, 7)}}


def build_models(C, data, A_bar, rungs=True, extra_mlp_acc=True):
    dev = C["DEVICE"]
    c, s = shapelab.lnp_center_scale(data)
    M = {}
    # report models
    bm = shapelab.baseline_models(C, data)
    M["report_gflash_n6"] = bm["gflash_n6"]
    M["report_mlp"] = bm["mlp"]
    # chosen ProtonShape (degree from the *_chosen.json)
    import json
    for tag in ("slab", "stdacc"):
        js = json.load(open(f"{FINAL}/{tag}_gce_chosen.json"))
        n = js["extra"]["degree"]
        core = ProtonShape(C, lnp0=c, lnp_s=s, lateral="gce", **DEG[n]).to(dev)
        m = shapelab.WithAcceptance(core, A_bar).to(dev) if tag == "stdacc" else core
        m.load_state_dict(ld(f"{FINAL}/{tag}_gce_chosen.pt", dev)["state"])
        M[f"PS_{tag}_chosen"] = m.eval()
        if rungs:
            for k in range(1, 7):
                core = ProtonShape(C, lnp0=c, lnp_s=s, lateral="gce", **DEG[k]).to(dev)
                m = shapelab.WithAcceptance(core, A_bar).to(dev) if tag == "stdacc" else core
                m.load_state_dict(ld(f"{FINAL}/{tag}_gce_n{k}.pt", dev)["state"])
                M[f"PS_{tag}_n{k}"] = m.eval()
    # reference refits
    m = shapelab.WithAcceptance(C["Formula"]("gamma", "core_tail", degree=6, lnp0=c, lnp_s=s), A_bar).to(dev)
    m.load_state_dict(ld(f"{FINAL}/ref_std_gflash_acc_n6.pt", dev)["state"])
    M["ref_std_gflash_acc_n6"] = m.eval()
    m = C["Formula"]("gamma", "core_tail", degree=6, lnp0=c, lnp_s=s).to(dev)
    m.load_state_dict(strip(ld(f"{FINAL}/ref_slab_gflash_n6.pt", dev)["state"]))
    M["ref_slab_gflash_n6"] = m.eval()
    m = C["FieldModel"](16, 3).to(dev)
    m.load_state_dict(strip(ld(f"{FINAL}/ref_slab_mlp_L3xH16.pt", dev)["state"]))
    M["ref_slab_mlp_L3xH16"] = m.eval()
    if extra_mlp_acc:   # G2 lane: report MLP architecture x Abar trained with the report protocol, standard likelihood
        m = shapelab.WithAcceptance(C["FieldModel"](16, 3), A_bar).to(dev)
        m.load_state_dict(ld(f"{G2}/results_std/proton/ref_std_mlp_L3xH16_accboth.pt", dev)["state"])
        M["g2_std_mlp_acc"] = m.eval()
    return M


# ------------------------------------------------------------------ independent slab acceptance
def my_accept(C, cos_t, sin_t, D, half, nz=8, nr=8):
    """(B, N_Z, N_R) fraction of each cell (ring volume) inside -half <= z cos - r cos(phi) sin <= D.
    Midpoint rule with nz x nr sub-points per cell (equal-area in r), exact in phi:
    frac(z, r) = [asin(clip(u)) - asin(clip(l))] / pi,  l = (z cos - D)/(r sin),  u = (z cos + half)/(r sin)."""
    dev = cos_t.device
    Ze, Re = C["Z_EDGES"].double(), C["R_EDGES"].double()
    kz = (torch.arange(nz, device=dev, dtype=torch.float64) + 0.5) / nz
    kr = (torch.arange(nr, device=dev, dtype=torch.float64) + 0.5) / nr
    zs = (Ze[:-1, None] + (Ze[1:, None] - Ze[:-1, None]) * kz)                                 # (NZ, nz)
    rs = torch.sqrt(Re[:-1, None] ** 2 + (Re[1:, None] ** 2 - Re[:-1, None] ** 2) * kr)          # (NR, nr)
    ct, st, Dd = cos_t.double(), sin_t.double(), D.double()
    zc = zs[None, :, :, None, None] * ct[:, None, None, None, None]                            # (B, NZ, nz, 1, 1)
    den = rs[None, None, None, :, :] * st[:, None, None, None, None]                            # (B, 1, 1, NR, nr)
    den = den.clamp_min(1e-12)
    u = ((zc + half) / den).clamp(-1, 1)
    l = ((zc - Dd[:, None, None, None, None]) / den).clamp(-1, 1)
    frac = (torch.asin(u) - torch.asin(l)) / math.pi
    return frac.mean((2, 4)).float()


def geom(b):
    cos_t = b.R[:, 2, 2]
    sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
    return cos_t, sin_t, b.zb * cos_t


@torch.no_grad()
def eval_parts(C, model, sh, K=None, nsub=8, want=("std", "slab_mine", "slab_sl2")):
    """Per-event: hit part H = -sum w log g (same fold seed/batching as test_nll) and log-masses for each normalisation.
    Returns dict of per-event NLL arrays for each normalisation, plus diagnostics."""
    K = K or C["N_FOLD_EVAL"]
    dev = C["DEVICE"]
    gen = torch.Generator(device=dev).manual_seed(12345)
    LOG_VOL, Z_MIN, DZ, N_Z = C["LOG_VOL"], C["Z_MIN"], C["DZ"], C["N_Z"]
    half = float(sh.box[2]) / 2
    out = {k: [] for k in want}
    out["H"], out["norm_err"], out["Wsum"] = [], [], []
    for b in sh.batches(128):
        logP = model(b.lnp)
        out["norm_err"].append((torch.logsumexp(logP.reshape(len(logP), -1), 1)).abs().max().item())
        logrho = logP - LOG_VOL
        pts = C["fold"](b, sh.box, K, gen)
        iz, ir = C["locate"](pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
        logg = torch.logsumexp(logrho[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(K)
        B = len(b.lnp)
        H = -torch.zeros(B, device=dev).index_add_(0, b.ev, b.w * logg)
        Wsum = torch.zeros(B, device=dev).index_add_(0, b.ev, b.w)
        out["H"].append(H)
        out["Wsum"].append(Wsum)
        P = logP.exp()
        if "std" in want:
            Pz = P.sum(2)
            cdf = torch.cat([Pz.new_zeros(B, 1), torch.cumsum(Pz, 1)], 1)
            t = ((b.zb - Z_MIN) / DZ).clamp(0, N_Z)
            k = t.floor().long().clamp(max=N_Z - 1)
            ar = torch.arange(B, device=dev)
            mass = cdf[ar, k] + Pz[ar, k] * (t - k)
            out["std"].append(H + Wsum * torch.log(mass.clamp_min(1e-8)))
        cos_t, sin_t, D = geom(b)
        if "slab_mine" in want:
            A = my_accept(C, cos_t, sin_t, D, half, nsub, nsub)
            mass = (P * A).sum((1, 2))
            out["slab_mine"].append(H + Wsum * torch.log(mass.clamp_min(1e-8)))
        if "slab_sl2" in want:          # the study's closed form with nsub=2 (shapelab.SLAB_GEOM_SRC), for comparison only
            A2 = C["event_slab"](b, sh.box)
            mass = (P * A2).sum((1, 2))
            out["slab_sl2"].append(H + Wsum * torch.log(mass.clamp_min(1e-8)))
    res = {k: torch.cat(v).cpu().numpy() for k, v in out.items() if k not in ("norm_err",)}
    res["norm_err"] = float(max(out["norm_err"]))
    return res


def paired(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
