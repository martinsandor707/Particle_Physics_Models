"""R1 audit, step 4: is the slab-normalised NLL a properly normalised likelihood of the observed hits?

For the observed cell centres c_i the model's cell probability is q(c) = V_c g(c) / M; it is normalised iff
M = integral of rho over the union of the cells that can carry a hit.  That union is
   slab  AND  inside an existing detector tower (lateral outline, beam hole, module gaps)  AND  cell centre passing the cuts.
The study uses M = mass in the infinite slab (lateral detector boundaries ignored); the hit term g also clamps fold points
that leave the grid to the border cells (locate()), which over-counts for hits at the grid border.

This script
 (1) re-checks lane G1's cached per-event acceptances (acc_cells_test / acc_full_test) against an independent brute-force
     Monte Carlo over the tower list (nearest tower box, layer, cell-centre cuts) for 10 test events;
 (2) evaluates, per test event and per model, the hit term S with and without clamping and ln M for every normalisation,
     with the exact random stream of test_nll (batches of 128, seed 12345, K = 64), so S + ln M_std reproduces test_nll;
 (3) reports how the paired model differences change between the study's slab metric and the properly normalised one.
Outputs: r04_properness.json, r04_terms_test.npz"""
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
ACC = "/home/user/tamas/shape-form-proton/lanes/G1_geometry/cache"
t00 = time.time()
C, geo, data = shapelab.setup(batch=256, out_dir=OUT + "/scratch", slab=False)
dev = C["DEVICE"]
G = torch.load("/home/user/tamas/shape-form-proton/cache/proton_geometry.pt", weights_only=False)
te = data["test"]
gte = G["sets"]["test"]
assert torch.allclose(torch.acos(te.ev["R"][:, 2, 2]).cpu(), gte["theta"], atol=1e-5)
origin = gte["origin"].to(dev).double()
towers = G["cells_xy"].to(dev).double()
z_first = float(G["layers_z"][0])
LAYER = 20.5
HALF = LAYER / 2
res = {}

acc_cells = torch.load(os.path.join(ACC, "acc_cells_test.pt"), weights_only=False)
acc_full = torch.load(os.path.join(ACC, "acc_full_test.pt"), weights_only=False)
print("G1 caches", acc_cells.shape, acc_cells.dtype, f"({time.time() - t00:.0f}s)", flush=True)


# ---------------- (1) independent brute force of the "cells" acceptance for a few events ---------------------------------
@torch.no_grad()
def brute_cells(i, N=192, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    R = te.ev["R"][i].double()
    o = origin[i]
    D = float(te.ev["zb"][i] * te.ev["R"][i, 2, 2])
    ZE, RE = C["Z_EDGES"].double(), C["R_EDGES"].double()
    # towers that can matter: transverse distance to the axis segment < 800 mm
    tt = torch.linspace(-250, 1500, 64, device=dev, dtype=torch.float64)
    axis = o[None, :2] + tt[:, None] * R[2, :2][None]
    dmin = torch.cdist(towers, axis).min(1).values
    tw = towers[dmin < 800]
    u = torch.rand(C["N_Z"], C["N_R"], N, 3, device=dev, generator=g, dtype=torch.float64)
    z = ZE[:-1, None, None] + C["DZ"] * u[..., 0]
    r = torch.sqrt(RE[:-1][None, :, None] ** 2 + (RE[1:] ** 2 - RE[:-1] ** 2)[None, :, None] * u[..., 1])
    ph = 2 * math.pi * u[..., 2]
    loc = torch.stack([r * torch.cos(ph), r * torch.sin(ph), z], -1).reshape(-1, 3)
    glob = o[None] + loc @ R                                    # global = o + R^T local  (row-vector form: loc @ R)
    ok = torch.zeros(len(glob), dtype=torch.bool, device=dev)
    for s in range(0, len(glob), 60000):
        gp = glob[s:s + 60000]
        dz = gp[:, 2] - o[2]
        in_slab = (dz >= -HALF) & (dz <= D)
        cheb = torch.maximum((gp[:, None, 0] - tw[None, :, 0]).abs(), (gp[:, None, 1] - tw[None, :, 1]).abs())
        dm, it = cheb.min(1)
        in_tower = dm <= 24.3
        lay = torch.round((gp[:, 2] - z_first) / LAYER)
        cen = torch.stack([tw[it, 0], tw[it, 1], z_first + LAYER * lay], 1)
        cl = (cen - o[None]) @ R.T                                 # local of the cell centre
        cut = (cl[:, 2] >= C["Z_MIN"]) & (cl[:, 2] <= C["Z_MAX"]) & (cl[:, 0] ** 2 + cl[:, 1] ** 2 < C["R_MAX"] ** 2)
        ok[s:s + 60000] = in_slab & in_tower & cut
    return ok.reshape(C["N_Z"], C["N_R"], N).double().mean(-1)


models = rmodels.load_all(C, data)
chk = {}
sel = torch.randperm(len(te), generator=torch.Generator().manual_seed(3))[:10]
with torch.no_grad():
    rows = []
    for i in sel.tolist():
        Abf = brute_cells(i)
        Ag = acc_cells[i].to(dev).double()
        b = te.batch(torch.tensor([i], device=dev))
        row = {"event": i, "theta": float(torch.acos(te.ev["R"][i, 2, 2])), "mean |A_G1 - A_bruteforce| over cells": float((Ag - Abf).abs().mean())}
        for mn in ("report_gflash_n6", "report_mlp", "new_slab_gce_n6"):
            P = models[mn](b.lnp)[0].exp().double()
            row[f"{mn}: ln M_cells(G1) - ln M_cells(brute force)"] = float(torch.log((P * Ag).sum()) - torch.log((P * Abf).sum()))
            row[f"{mn}: ln M_slab - ln M_cells(brute force)"] = float(torch.log((P * C["event_slab"](b, te.box)[0].double()).sum()) - torch.log((P * Abf).sum()))
        rows.append(row)
        print(json.dumps(row), flush=True)
res["(1) G1 'cells' acceptance vs independent brute force (10 test events, 192 points per grid cell)"] = rows
print(f"brute force done ({time.time() - t00:.0f}s)", flush=True)


# ---------------- (2) per-event terms on the whole test set ------------------------------------------------------------------
def report_mass(logP, zb):
    Pz = logP.exp().sum(2)
    cdf = torch.cat([Pz.new_zeros(len(Pz), 1), torch.cumsum(Pz, 1)], 1)
    t = ((zb - C["Z_MIN"]) / C["DZ"]).clamp(0, C["N_Z"])
    k = t.floor().long().clamp(max=C["N_Z"] - 1)
    ar = torch.arange(len(Pz), device=Pz.device)
    return cdf[ar, k] + Pz[ar, k] * (t - k)


terms = {}
n_allout = {}
with torch.no_grad():
    for mn, m in models.items():
        gen = torch.Generator(device=dev).manual_seed(12345)
        acc = {k: [] for k in ("S", "S_unclamped", "std", "slab", "full", "cells")}
        nout = 0
        start = 0
        for b in te.batches(128):
            n = len(b.lnp)
            idx = torch.arange(start, start + n)
            start += n
            logP = m(b.lnp)
            logrho = logP - C["LOG_VOL"]
            pts = C["fold"](b, te.box, C["N_FOLD_EVAL"], gen)
            rr = torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
            iz, ir = C["locate"](pts[..., 2], rr)
            lr = logrho[b.ev[:, None].expand_as(iz), iz, ir]
            logg = torch.logsumexp(lr, 1) - math.log(C["N_FOLD_EVAL"])
            inside = (pts[..., 2] >= C["Z_MIN"]) & (pts[..., 2] <= C["Z_MAX"]) & (rr < C["R_MAX"])
            lgu = torch.logsumexp(lr.masked_fill(~inside, -float("inf")), 1) - math.log(C["N_FOLD_EVAL"])
            nout += int((~torch.isfinite(lgu)).sum())
            lgu = torch.where(torch.isfinite(lgu), lgu, logg - 30.0)
            acc["S"].append((-torch.zeros(n, device=dev).index_add_(0, b.ev, b.w * logg)).double().cpu())
            acc["S_unclamped"].append((-torch.zeros(n, device=dev).index_add_(0, b.ev, b.w * lgu)).double().cpu())
            P = logP.exp().double()
            acc["std"].append(torch.log(report_mass(logP, b.zb).double().clamp_min(1e-8)).cpu())
            acc["slab"].append(torch.log((P * C["event_slab"](b, te.box).double()).sum((1, 2)).clamp_min(1e-8)).cpu())
            acc["full"].append(torch.log((P * acc_full[idx].to(dev).double()).sum((1, 2)).clamp_min(1e-8)).cpu())
            acc["cells"].append(torch.log((P * acc_cells[idx].to(dev).double()).sum((1, 2)).clamp_min(1e-8)).cpu())
        terms[mn] = {k: torch.cat(v).numpy() for k, v in acc.items()}
        n_allout[mn] = nout
        print(mn, f"({time.time() - t00:.0f}s)", flush=True)

# sanity: S + ln M_std reproduces test_nll of the harness, S + ln M_slab the slab test_nll
Cs, _, _ = shapelab.setup(batch=256, out_dir=OUT + "/scratch", slab=True)
chk = {}
for mn in ("report_gflash_n6", "new_slab_gce_n6"):
    ref_std = C["test_nll"](models[mn], te)
    ref_slab = Cs["test_nll"](models[mn], te)
    chk[mn] = [float(np.abs(terms[mn]["S"] + terms[mn]["std"] - ref_std).max()), float(np.abs(terms[mn]["S"] + terms[mn]["slab"] - ref_slab).max())]
res["sanity: max |S + lnM_std - test_nll(std)|, max |S + lnM_slab - test_nll(slab)| per event"] = chk
res["events with all 64 fold points outside the grid (per model)"] = n_allout
print(chk, flush=True)

bins = te.ev["bin"].cpu().numpy()
zb = te.ev["zb"].cpu().numpy()
D = (te.ev["zb"] * te.ev["R"][:, 2, 2]).cpu().numpy()
first_layer_idx = np.round((gte["origin"][:, 2].numpy() - z_first) / LAYER).astype(int)
np.savez(os.path.join(OUT, "r04_terms_test.npz"), **{f"{m}|{k}": v for m, d in terms.items() for k, v in d.items()},
         bin=bins, D=D, first_layer_idx=first_layer_idx, theta=gte["theta"].numpy())

NORMS = {"report (std)": ("S", "std"), "study slab": ("S", "slab"), "slab, unclamped g": ("S_unclamped", "slab"),
         "full outline, unclamped": ("S_unclamped", "full"), "cells (proper), unclamped": ("S_unclamped", "cells"),
         "cells, clamped": ("S", "cells")}
tab = {}
for mn, d in terms.items():
    tab[mn] = {k: float((d[a] + d[b]).mean()) for k, (a, b) in NORMS.items()}
    tab[mn]["mean ln M_slab - ln M_cells (over-normalisation left in the slab metric)"] = float((d["slab"] - d["cells"]).mean())
    tab[mn]["mean ln M_slab - ln M_full"] = float((d["slab"] - d["full"]).mean())
    tab[mn]["mean S_clamped - S_unclamped (grid-border clamping)"] = float((d["S"] - d["S_unclamped"]).mean())
res["(2) test NLL of every model under each normalisation"] = tab
print(json.dumps(tab, indent=1), flush=True)

pairs = [("new_slab_gce_n6", "report_gflash_n6"), ("new_slab_gce_n6", "ref_slab_gflash_n6"), ("new_slab_gce_n6", "report_mlp"),
         ("new_slab_gce_n6", "ref_slab_mlp"), ("report_mlp", "report_gflash_n6"), ("ref_slab_mlp", "ref_slab_gflash_n6"),
         ("new_stdacc_gce_n6", "report_gflash_n6"), ("ref_std_gflash_acc_n6", "report_gflash_n6"), ("new_slab_gce_n1", "ref_slab_gflash_n6"),
         ("report_cnn", "report_mlp")]
ptab = {}
for a, b_ in pairs:
    row = {}
    for k, (x, y) in NORMS.items():
        dd = (terms[a][x] + terms[a][y]) - (terms[b_][x] + terms[b_][y])
        row[k] = [float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(len(dd)))]
    e = ((terms[a]["slab"] - terms[a]["cells"]) - (terms[b_]["slab"] - terms[b_]["cells"]))
    row["bias of the slab metric vs proper (cells) normalisation: paired d(lnM_slab - lnM_cells)"] = [float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e)))]
    e2 = (terms[a]["S"] - terms[a]["S_unclamped"]) - (terms[b_]["S"] - terms[b_]["S_unclamped"])
    row["bias from grid-border clamping: paired d(S - S_unclamped)"] = [float(e2.mean()), float(e2.std(ddof=1) / np.sqrt(len(e2)))]
    ptab[f"{a} - {b_}"] = row
res["(3) paired test differences (mean, SE) under each normalisation"] = ptab
print(json.dumps(ptab, indent=1), flush=True)

# events whose first hit layer is not the first physical layer / whose origin layer has no kept hit
grp = {"first hit layer = layer 0": first_layer_idx == 0, "first hit layer > 0": first_layer_idx > 0}
gt = {}
for g, msk in grp.items():
    gt[g] = {"n": int(msk.sum())}
    for mn in ("report_gflash_n6", "report_mlp", "new_slab_gce_n6", "ref_slab_gflash_n6", "ref_slab_mlp"):
        d = terms[mn]
        gt[g][mn] = {"std NLL": float((d["S"] + d["std"])[msk].mean()), "slab NLL": float((d["S"] + d["slab"])[msk].mean()),
                     "ln M_std - ln M_slab": float((d["std"] - d["slab"])[msk].mean()),
                     "ln M_slab - ln M_cells": float((d["slab"] - d["cells"])[msk].mean())}
    for a, b_ in (("new_slab_gce_n6", "ref_slab_gflash_n6"), ("report_mlp", "report_gflash_n6")):
        dd = (terms[a]["S"] + terms[a]["slab"] - terms[b_]["S"] - terms[b_]["slab"])[msk]
        gt[g][f"slab: {a} - {b_}"] = [float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(msk.sum()))]
        dd = (terms[a]["S"] + terms[a]["std"] - terms[b_]["S"] - terms[b_]["std"])[msk]
        gt[g][f"std: {a} - {b_}"] = [float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(msk.sum()))]
res["(4) events by first hit layer"] = gt
print(json.dumps(gt, indent=1), flush=True)
json.dump(res, open(os.path.join(OUT, "r04_properness.json"), "w"), indent=1)
print(f"done ({time.time() - t00:.0f}s)", flush=True)
