"""R1 audit, step 5: independent brute-force estimate of the observable region (union of detector cells that can carry a
hit) for a random subsample of test events, to confirm the model-dependent residual of the slab normalisation found with
lane G1's cached acceptances (r04).  Also: convergence of G1's quadrature, where the slab-trained models put the mass that
lies outside the slab, and the geometry of the events whose first hit layer is not the first physical layer.

acceptances per grid cell (uniform random points in the cell volume, 128 per cell):
  slab      : -h <= z_trans <= D                                   (study)
  outline   : slab AND inside a tower box enlarged to +-28.8 mm   (module gaps of 3.8 / 8.5 mm closed; outer edge, beam hole)
  cells     : slab AND inside a +-24.3 mm tower box AND that cell's centre passes the report's cuts
  cells_ng  : as cells but with the enlarged (gap-closing) boxes"""
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R1_likelihood")
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/G1_geometry")
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
origin = gte["origin"].to(dev).double()
towers = G["cells_xy"].to(dev).float()
z_first = float(G["layers_z"][0])
LAYER, HALF = 20.5, 10.25
ZE, RE = C["Z_EDGES"].double(), C["R_EDGES"].double()
res = {}


@torch.no_grad()
def brute(i, N=128, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed + i)
    R = te.ev["R"][i].double()
    o = origin[i]
    D = float(te.ev["zb"][i] * te.ev["R"][i, 2, 2])
    tt = torch.linspace(-250, 1500, 64, device=dev, dtype=torch.float64)
    axis = (o[None, :2] + tt[:, None] * R[2, :2][None]).float()
    tw = towers[torch.cdist(towers, axis).min(1).values < 800]
    u = torch.rand(C["N_Z"], C["N_R"], N, 3, device=dev, generator=g, dtype=torch.float64)
    z = ZE[:-1, None, None] + C["DZ"] * u[..., 0]
    r = torch.sqrt(RE[:-1][None, :, None] ** 2 + (RE[1:] ** 2 - RE[:-1] ** 2)[None, :, None] * u[..., 1])
    ph = 2 * math.pi * u[..., 2]
    loc = torch.stack([r * torch.cos(ph), r * torch.sin(ph), z], -1).reshape(-1, 3)
    glob = o[None] + loc @ R
    out = {k: torch.zeros(len(glob), dtype=torch.bool, device=dev) for k in ("slab", "outline", "cells", "cells_ng")}
    for s in range(0, len(glob), 65536):
        gp = glob[s:s + 65536]
        dz = gp[:, 2] - o[2]
        in_slab = (dz >= -HALF) & (dz <= D)
        gpf = gp[:, :2].float()
        cheb = torch.maximum((gpf[:, None, 0] - tw[None, :, 0]).abs(), (gpf[:, None, 1] - tw[None, :, 1]).abs())
        dm, it = cheb.min(1)
        lay = torch.round((gp[:, 2] - z_first) / LAYER)
        cen = torch.stack([tw[it, 0].double(), tw[it, 1].double(), z_first + LAYER * lay], 1)
        cl = (cen - o[None]) @ R.T
        cut = (cl[:, 2] >= C["Z_MIN"]) & (cl[:, 2] <= C["Z_MAX"]) & (cl[:, 0] ** 2 + cl[:, 1] ** 2 < C["R_MAX"] ** 2)
        out["slab"][s:s + 65536] = in_slab
        out["outline"][s:s + 65536] = in_slab & (dm <= 28.8)
        out["cells"][s:s + 65536] = in_slab & (dm <= 24.3) & cut
        out["cells_ng"][s:s + 65536] = in_slab & (dm <= 28.8) & cut
    return {k: v.reshape(C["N_Z"], C["N_R"], N).double().mean(-1) for k, v in out.items()}


models = rmodels.load_all(C, data, which=["report_gflash_n6", "report_mlp", "report_cnn", "new_slab_gce_n6", "new_stdacc_gce_n6",
                                          "ref_slab_gflash_n6", "ref_slab_mlp", "ref_std_gflash_acc_n6"])
acc_cells = torch.load(os.path.join(ACC, "acc_cells_test.pt"), weights_only=False)
acc_full = torch.load(os.path.join(ACC, "acc_full_test.pt"), weights_only=False)
NSEL = int(os.environ.get("NSEL", 1200))
sel = torch.randperm(len(te), generator=torch.Generator().manual_seed(5))[:NSEL].sort().values
lnM = {m: {k: [] for k in ("slab2", "bf_slab", "bf_outline", "bf_cells", "bf_cells_ng", "g1_cells", "g1_full")} for m in models}
with torch.no_grad():
    for j, i in enumerate(sel.tolist()):
        A = brute(i)
        b = te.batch(torch.tensor([i], device=dev))
        A2 = C["event_slab"](b, te.box)[0].double()
        Ag = {"g1_cells": acc_cells[i].to(dev).double(), "g1_full": acc_full[i].to(dev).double()}
        for mn, m in models.items():
            P = m(b.lnp)[0].exp().double()
            lnM[mn]["slab2"].append(float(torch.log((P * A2).sum())))
            for k in ("slab", "outline", "cells", "cells_ng"):
                lnM[mn]["bf_" + k].append(float(torch.log((P * A[k]).sum().clamp_min(1e-12))))
            for k, a in Ag.items():
                lnM[mn][k].append(float(torch.log((P * a).sum().clamp_min(1e-12))))
        if j % 200 == 0:
            print(f"  {j}/{len(sel)} ({time.time() - t00:.0f}s)", flush=True)
lnM = {m: {k: np.array(v) for k, v in d.items()} for m, d in lnM.items()}
np.savez(os.path.join(OUT, "r05_lnM_bruteforce.npz"), sel=sel.numpy(), **{f"{m}|{k}": v for m, d in lnM.items() for k, v in d.items()})

tab = {}
for mn, d in lnM.items():
    tab[mn] = {"ln M_slab(nsub2) - ln M_slab(brute force) (MC check of the slab part)": float((d["slab2"] - d["bf_slab"]).mean()),
               "ln M_slab - ln M_outline (bf)": float((d["slab2"] - d["bf_outline"]).mean()),
               "ln M_slab - ln M_cells (bf)": float((d["slab2"] - d["bf_cells"]).mean()),
               "ln M_slab - ln M_cells_nogap (bf)": float((d["slab2"] - d["bf_cells_ng"]).mean()),
               "ln M_slab - ln M_cells (G1 cache, same events)": float((d["slab2"] - d["g1_cells"]).mean()),
               "ln M_slab - ln M_full (G1 cache, same events)": float((d["slab2"] - d["g1_full"]).mean())}
res[f"residual over-normalisation of the slab metric, {len(sel)} random test events (mean nat per event)"] = tab
pairs = [("new_slab_gce_n6", "report_gflash_n6"), ("report_mlp", "report_gflash_n6"), ("new_slab_gce_n6", "report_mlp"),
         ("new_slab_gce_n6", "ref_slab_gflash_n6"), ("new_slab_gce_n6", "ref_slab_mlp"), ("new_stdacc_gce_n6", "report_gflash_n6"),
         ("ref_std_gflash_acc_n6", "report_gflash_n6"), ("report_cnn", "report_mlp")]
ptab = {}
for a, b_ in pairs:
    row = {}
    for k in ("bf_outline", "bf_cells", "bf_cells_ng", "g1_cells", "g1_full"):
        e = (lnM[a]["slab2"] - lnM[a][k]) - (lnM[b_]["slab2"] - lnM[b_][k])
        row[f"paired d(ln M_slab - ln M_{k})"] = [float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e)))]
    ptab[f"{a} - {b_}"] = row
res["model-dependent part (= bias of the slab metric's paired differences relative to a properly normalised one), mean, SE"] = ptab
print(json.dumps(res, indent=1), flush=True)

# ---- G1 quadrature convergence on 10 events -------------------------------------------------------------------------------
import g1lib
det = g1lib.Detector(G, dev)
conv = []
for i in sel[:10].tolist():
    R_, zb_, org_ = te.ev["R"][i:i + 1], te.ev["zb"][i:i + 1], gte["origin"][i:i + 1].to(dev)
    a_lo = g1lib.acceptance(C, det, R_, zb_, org_, te.box, nsub=2, nphi=24)
    a_hi = g1lib.acceptance(C, det, R_, zb_, org_, te.box, nsub=6, nphi=96)
    b = te.batch(torch.tensor([i], device=dev))
    P = models["report_gflash_n6"](b.lnp)[0].exp().double()
    Abf = brute(i, N=512, seed=99)
    conv.append({"event": i, "ln M_cells: G1(nsub2,nphi24) - G1(nsub6,nphi96)": float(torch.log((P * a_lo["cells"][0].double()).sum()) - torch.log((P * a_hi["cells"][0].double()).sum())),
                 "ln M_cells: G1(nsub6,nphi96) - brute force(512/cell)": float(torch.log((P * a_hi["cells"][0].double()).sum()) - torch.log((P * Abf["cells"]).sum())),
                 "ln M_full: G1(nsub2) - G1(nsub6)": float(torch.log((P * a_lo["full"][0].double()).sum()) - torch.log((P * a_hi["full"][0].double()).sum()))})
    print(conv[-1], flush=True)
res["G1 quadrature convergence (report GFlash n=6 weights)"] = conv

# ---- where do slab-trained models put the mass outside the slab? -------------------------------------------------------------
with torch.no_grad():
    Amax = torch.zeros(C["N_Z"], C["N_R"], device=dev)
    Asum = torch.zeros(C["N_Z"], C["N_R"], device=dev)
    for b in data["train"].batches(512):
        A = C["event_slab"](b, data["train"].box, nsub=4)
        Amax = torch.maximum(Amax, A.max(0).values)
        Asum += A.sum(0)
    Abar = Asum / len(data["train"])
    never = Amax <= 0
    mt = {}
    for mn, m in models.items():
        out_m, never_m, rare_m = [], [], []
        for b in te.batches(256):
            P = m(b.lnp).exp()
            A = C["event_slab"](b, te.box)
            out_m.append((P * (1 - A)).sum((1, 2)))
            never_m.append((P * never.float()).sum((1, 2)))
            rare_m.append((P * (Abar < 0.05).float()).sum((1, 2)))
        mt[mn] = {"mean mass outside the event's slab": float(torch.cat(out_m).mean()),
                  "mean mass in cells outside EVERY training event's slab": float(torch.cat(never_m).mean()),
                  "mean mass in cells with Abar(train) < 0.05": float(torch.cat(rare_m).mean())}
    res["mass outside the slab (test events)"] = mt
    print(json.dumps(mt, indent=1), flush=True)

# ---- events whose first hit layer is not layer 0 -------------------------------------------------------------------------------
fl = np.round((gte["origin"][:, 2].numpy() - z_first) / LAYER).astype(int)
rho0 = np.hypot(gte["origin"][:, 0].numpy(), gte["origin"][:, 1].numpy())
th = gte["theta"].numpy()
# where would the incident line cross the front face of layer 0?  (straight line from the origin backwards along -e_r)
rho_front = rho0 - (gte["origin"][:, 2].numpy() - (z_first - HALF)) * np.tan(th)
m = fl > 0
res["first hit layer > 0: n, theta quartiles, origin rho quartiles [mm], rho of the incident line at the detector front face quartiles"] = [
    int(m.sum()), np.quantile(th[m], [0.25, 0.5, 0.75]).tolist(), np.quantile(rho0[m], [0.25, 0.5, 0.75]).tolist(),
    np.quantile(rho_front[m], [0.25, 0.5, 0.75]).tolist()]
res["first hit layer = 0: theta quartiles, origin rho quartiles"] = [np.quantile(th[~m], [0.25, 0.5, 0.75]).tolist(), np.quantile(rho0[~m], [0.25, 0.5, 0.75]).tolist()]
json.dump(res, open(os.path.join(OUT, "r05_cells_bruteforce.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k.startswith("first")}, indent=1))
print(f"done ({time.time() - t00:.0f}s)", flush=True)
