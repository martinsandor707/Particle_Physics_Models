"""R1 audit, step 3: the closed-form slab acceptance (shapelab.slab_accept, nsub = 2, used by the 3D likelihood) against
(i) an exact reference: z-integral done analytically (antiderivative of arccos), r^2 integral by a 512-point midpoint rule,
    float64;
(ii) brute-force Monte Carlo in Cartesian coordinates (uniform points in every grid cell, direct test of
     -h <= z' cos(t) - x' sin(t) <= D; no arccos), 4096 points per cell;
(iii) model-sampling Monte Carlo (points drawn from the piecewise-constant model density, counted inside the slab).

Then the error of ln M (nat, = error of the per-event NLL) for every model, per event, on the whole test set.
Outputs: r03_accept_check.json, r03_lnM_test.npz (per-event ln M under every normalisation, every model)."""
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R1_likelihood")
import numpy as np
import torch

import shapelab
import rmodels

OUT = "/home/user/tamas/shape-form-proton/review/R1_likelihood"
t00 = time.time()
C, geo, data = shapelab.setup(batch=256, out_dir=OUT + "/scratch", slab=False)
dev = C["DEVICE"]
ZE, RE = C["Z_EDGES"].double(), C["R_EDGES"].double()
DZ = float(C["DZ"])
HALF = float(data["test"].box[2]) / 2


def H(u):
    """Antiderivative of arccos(clip(u, -1, 1))."""
    uc = u.clamp(-1, 1)
    return u * torch.acos(uc) - torch.sqrt((1 - uc * uc).clamp_min(0))


@torch.no_grad()
def exact_accept(cos_t, sin_t, D, half, nr=256, front=True, back=True, chunk=12):
    """Volume fraction of every (z', r) grid cell inside the slab. (B, N_Z, N_R) float64."""
    if len(cos_t) > chunk:
        return torch.cat([exact_accept(cos_t[i:i + chunk], sin_t[i:i + chunk], D[i:i + chunk], half, nr, front, back, chunk)
                          for i in range(0, len(cos_t), chunk)])
    cos_t, sin_t, D = cos_t.double(), sin_t.double(), D.double()
    k = (torch.arange(nr, device=dev, dtype=torch.float64) + 0.5) / nr
    rs = torch.sqrt(RE[:-1, None] ** 2 + (RE[1:, None] ** 2 - RE[:-1, None] ** 2) * k)          # (N_R, nr)
    a = (cos_t[:, None, None] / (rs[None] * sin_t[:, None, None]))[:, None]                         # (B,1,N_R,nr)
    z1, z2 = ZE[:-1][None, :, None, None], ZE[1:][None, :, None, None]
    if back:
        b = (-D[:, None, None] / (rs[None] * sin_t[:, None, None]))[:, None]
        I_lo = (H(a * z2 + b) - H(a * z1 + b)) / a
    else:
        I_lo = torch.full_like(a * z2, math.pi * DZ)
    if front:
        b = (half / (rs[None] * sin_t[:, None, None]))[:, None]
        I_hi = (H(a * z2 + b) - H(a * z1 + b)) / a
    else:
        I_hi = torch.zeros_like(a * z2)
    return ((I_lo - I_hi) / (math.pi * DZ)).mean(-1)


@torch.no_grad()
def mc_accept(cos_t, sin_t, D, half, N=4096, seed=0):
    """Brute force: N uniform points per cell (z uniform, r^2 uniform, phi uniform), direct slab test. (B, N_Z, N_R)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for i in range(len(cos_t)):
        u = torch.rand(C["N_Z"], C["N_R"], N, 3, device=dev, generator=g, dtype=torch.float64)
        z = ZE[:-1, None, None] + DZ * u[..., 0]
        r = torch.sqrt(RE[:-1][None, :, None] ** 2 + (RE[1:] ** 2 - RE[:-1] ** 2)[None, :, None] * u[..., 1])
        x = r * torch.cos(2 * math.pi * u[..., 2])
        zt = z * float(cos_t[i]) - x * float(sin_t[i])
        out.append(((zt >= -half) & (zt <= float(D[i]))).double().mean(-1))
    return torch.stack(out)


def geom(b):
    cos_t = b.R[:, 2, 2]
    sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
    return cos_t, sin_t, b.zb * cos_t


res = {}
te = data["test"]
D_all = (te.ev["zb"] * te.ev["R"][:, 2, 2])
th_all = torch.acos(te.ev["R"][:, 2, 2])
g0 = torch.Generator().manual_seed(7)
rand_idx = torch.randperm(len(te), generator=g0)[:40]
sets = {"random": rand_idx, "smallest_D": torch.argsort(D_all.cpu())[:40], "largest_theta": torch.argsort(th_all.cpu())[-40:],
        "smallest_theta": torch.argsort(th_all.cpu())[:40]}

# ---------------- (A) per-cell accuracy: closed form (nsub = 2) vs exact vs brute force -----------------------------------
for name, idx in sets.items():
    b = te.batch(idx.to(dev))
    ct, stt, D = geom(b)
    A2 = C["event_slab"](b, te.box).double()
    Aex = exact_accept(ct, stt, D, HALF)
    Amc = mc_accept(ct, stt, D, HALF, N=4096, seed=1)
    A64 = torch.cat([C["slab_accept"](ct[i:i + 4], stt[i:i + 4], D[i:i + 4], HALF, nsub=64).double() for i in range(0, len(ct), 4)])
    Aex1024 = exact_accept(ct[:8], stt[:8], D[:8], HALF, nr=1024, chunk=2)             # convergence of the reference in r^2
    Af = exact_accept(ct, stt, D, HALF, back=False)
    Ab = exact_accept(ct, stt, D, HALF, front=False)
    both = (Af < 0.999) & (Ab < 0.999)
    r0 = torch.zeros_like(A2, dtype=torch.bool)
    r0[:, :, :2] = True
    part = (Aex > 1e-9) & (Aex < 1 - 1e-9)
    d = {
        "D range [mm]": [float(D.min()), float(D.max())], "theta range": [float(torch.acos(ct).min()), float(torch.acos(ct).max())],
        "exact vs brute-force MC: mean |dA|, max |dA| (MC sd <= 0.0078)": [float((Aex - Amc).abs().mean()), float((Aex - Amc).abs().max())],
        "exact vs brute-force MC: mean signed dA over partially cut cells": float((Aex - Amc)[part].mean()),
        "closed form nsub=64 vs exact: max |dA|": float((A64 - Aex).abs().max()),
        "exact reference nr=256 vs nr=1024: max |dA| (8 events)": float((Aex1024 - Aex[:8]).abs().max()),
        "closed form nsub=2 vs exact: mean |dA| (all cells), max |dA|": [float((A2 - Aex).abs().mean()), float((A2 - Aex).abs().max())],
        "closed form nsub=2 vs exact: mean |dA| over partially cut cells": float((A2 - Aex)[part].abs().mean()),
        "r->0 (first two r bins), partially cut cells: mean |dA|, max |dA|, n": [float((A2 - Aex)[part & r0].abs().mean()) if (part & r0).any() else None,
                                                                              float((A2 - Aex)[part & r0].abs().max()) if (part & r0).any() else None, int((part & r0).sum())],
        "cells cut by BOTH planes: n, mean |dA| nsub=2, max |dA| nsub=2, mean |exact - MC|": [int(both.sum()),
              float((A2 - Aex)[both].abs().mean()) if both.any() else None, float((A2 - Aex)[both].abs().max()) if both.any() else None,
              float((Aex - Amc)[both].abs().mean()) if both.any() else None],
    }
    res[f"A per-cell [{name}, {len(idx)} test events]"] = d
    print(name, json.dumps(d), flush=True)
    del Amc
    torch.cuda.empty_cache()

# a worked example of the r -> 0 edge: the z-cell holding the front plane on the axis
b = te.batch(sets["largest_theta"][:1].to(dev))
ct, stt, D = geom(b)
zf = -HALF / float(ct[0])
iz = int(((zf - C["Z_MIN"]) / DZ) // 1)
A2 = C["event_slab"](b, te.box)[0, iz, :3].tolist()
Aex = exact_accept(ct, stt, D, HALF)[0, iz, :3].tolist()
res["edge example: theta, front plane on the axis z'=-h/cos(t), z-cell, A(nsub=2) first 3 r bins, exact"] = [
    float(torch.acos(ct[0])), zf, [float(C["Z_EDGES"][iz]), float(C["Z_EDGES"][iz + 1])], A2, Aex]
print(res["edge example: theta, front plane on the axis z'=-h/cos(t), z-cell, A(nsub=2) first 3 r bins, exact"], flush=True)

# ---------------- (B) ln M per model, whole test set ------------------------------------------------------------------------
models = rmodels.load_all(C, data)
print("models:", list(models), f"({time.time() - t00:.0f}s)", flush=True)
std_mass = None


def report_mass(logP, zb):
    Pz = logP.exp().sum(2)
    cdf = torch.cat([Pz.new_zeros(len(Pz), 1), torch.cumsum(Pz, 1)], 1)
    t = ((zb - C["Z_MIN"]) / C["DZ"]).clamp(0, C["N_Z"])
    k = t.floor().long().clamp(max=C["N_Z"] - 1)
    ar = torch.arange(len(Pz), device=Pz.device)
    return cdf[ar, k] + Pz[ar, k] * (t - k)


keys = ["std", "slab2", "slab_exact", "slab_exact_half0", "front_exact", "back_exact"]
L = {m: {k: [] for k in keys} for m in models}
with torch.no_grad():
    for b in te.batches(48):
        ct, stt, D = geom(b)
        A2 = C["event_slab"](b, te.box).double()
        Aex = exact_accept(ct, stt, D, HALF)
        A0 = exact_accept(ct, stt, D, 0.0)
        Af = exact_accept(ct, stt, D, HALF, back=False)
        Ab = exact_accept(ct, stt, D, HALF, front=False)
        for mn, m in models.items():
            logP = m(b.lnp)
            P = logP.exp().double()
            L[mn]["std"].append(torch.log(report_mass(logP, b.zb).clamp_min(1e-8)).double().cpu())
            for k, A in (("slab2", A2), ("slab_exact", Aex), ("slab_exact_half0", A0), ("front_exact", Af), ("back_exact", Ab)):
                L[mn][k].append(torch.log((P * A).sum((1, 2))).cpu())
L = {m: {k: torch.cat(v).numpy() for k, v in d.items()} for m, d in L.items()}
np.savez(os.path.join(OUT, "r03_lnM_test.npz"), **{f"{m}|{k}": v for m, d in L.items() for k, v in d.items()},
         p=te.ev["p"].cpu().numpy(), bin=te.ev["bin"].cpu().numpy(), D=D_all.cpu().numpy(), theta=th_all.cpu().numpy())
print(f"ln M computed ({time.time() - t00:.0f}s)", flush=True)

bins = te.ev["bin"].cpu().numpy()
groups = {"all": np.ones_like(bins, bool), "bin0 (p<1)": bins == 0, "bin1": bins == 1, "bins2-4": (bins >= 2) & (bins <= 4), "bins5-19": bins >= 5}
tab = {}
for mn, d in L.items():
    e = d["slab2"] - d["slab_exact"]
    row = {g: [float(e[m].mean()), float(e[m].std(ddof=1) / np.sqrt(m.sum()))] for g, m in groups.items()}
    row["max |err| per event"] = float(np.abs(e).max())
    row["mean ln M_std - ln M_slab(exact)  (= std NLL - slab NLL)"] = float((d["std"] - d["slab_exact"]).mean())
    row["mean ln M_std - ln M_slab(nsub=2)  (as used)"] = float((d["std"] - d["slab2"]).mean())
    row["mean M_slab(exact)"] = float(np.exp(d["slab_exact"]).mean())
    row["mean ln M_slab(half=10.25) - ln M_slab(half=0)  (half-layer offset)"] = float((d["slab_exact"] - d["slab_exact_half0"]).mean())
    row["mean -ln M_front, -ln M_back (exact)"] = [float(-d["front_exact"].mean()), float(-d["back_exact"].mean())]
    tab[mn] = row
res["B: error of ln M with the nsub=2 closed form, ln M(nsub=2) - ln M(exact) [nat] (mean, SE) per group; = error of the slab NLL"] = tab
ref = L["report_gflash_n6"]["slab2"] - L["report_gflash_n6"]["slab_exact"]
pair = {}
for mn, d in L.items():
    e = (d["slab2"] - d["slab_exact"]) - ref
    pair[mn] = [float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e)))]
res["B: paired difference of that error vs report GFlash n=6 [nat] (mean, SE): bias of model comparisons"] = pair
print(json.dumps(tab, indent=1), flush=True)
print(json.dumps(pair, indent=1), flush=True)

# ---------------- (C) model-sampling Monte Carlo of M (independent of any acceptance formula) --------------------------------
mc = {}
with torch.no_grad():
    idx = torch.randperm(len(te), generator=torch.Generator().manual_seed(11))[:600].to(dev)
    for mn in ("report_gflash_n6", "report_mlp", "new_slab_gce_n6", "ref_slab_gflash_n6"):
        m = models[mn]
        dl = []
        for s in range(0, len(idx), 50):
            b = te.batch(idx[s:s + 50])
            ct, stt, D = geom(b)
            P = m(b.lnp).exp()
            gen = torch.Generator(device=dev).manual_seed(100 + s)
            Ms = 200000
            cell = torch.multinomial(P.reshape(len(P), -1), Ms, True, generator=gen)
            iz, ir = cell // C["N_R"], cell % C["N_R"]
            u = torch.rand(len(P), Ms, 3, device=dev, generator=gen, dtype=torch.float64)
            z = ZE[iz] + u[..., 0] * DZ
            r = torch.sqrt(RE[ir] ** 2 + u[..., 1] * (RE[ir + 1] ** 2 - RE[ir] ** 2))
            x = r * torch.cos(2 * math.pi * u[..., 2])
            zt = z * ct.double()[:, None] - x * stt.double()[:, None]
            Mmc = ((zt >= -HALF) & (zt <= D.double()[:, None])).double().mean(1)
            Aex = exact_accept(ct, stt, D, HALF)
            A2 = C["event_slab"](b, te.box).double()
            Pd = P.double()
            dl.append(torch.stack([torch.log((Pd * Aex).sum((1, 2))) - torch.log(Mmc), torch.log((Pd * A2).sum((1, 2))) - torch.log(Mmc)], 1).cpu())
        dl = torch.cat(dl).numpy()
        mc[mn] = {"ln M_exact - ln M_MC (mean, SE)": [float(dl[:, 0].mean()), float(dl[:, 0].std(ddof=1) / np.sqrt(len(dl)))],
                  "ln M_nsub2 - ln M_MC (mean, SE)": [float(dl[:, 1].mean()), float(dl[:, 1].std(ddof=1) / np.sqrt(len(dl)))]}
        print(mn, mc[mn], flush=True)
res["C: model-sampling MC (600 test events, 200k points each)"] = mc
json.dump(res, open(os.path.join(OUT, "r03_accept_check.json"), "w"), indent=1)
print(f"done ({time.time() - t00:.0f}s)", flush=True)
