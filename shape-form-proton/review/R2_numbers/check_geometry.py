"""Geometry / normalisation checks for the independent slab metric (R2_numbers)."""
import json, math, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
import numpy as np, torch
import r2lib, shapelab

t0 = time.time()
C, geo, data = r2lib.setup()
dev = C["DEVICE"]
res = {"geo": geo}
A_bar = shapelab.mean_acceptance(C, data["train"], front=True, back=True)
M = r2lib.build_models(C, data, A_bar, rungs=False)

# 1. hits and fold points inside the slab?
for split in ("val", "test"):
    sh = data[split]
    half = float(sh.box[2]) / 2
    zmin, over, fz, fo = [], [], [], []
    gen = torch.Generator(device=dev).manual_seed(12345)
    for b in sh.batches(128):
        ct, st, D = r2lib.geom(b)
        zt = b.xyz[:, 2] * ct[b.ev] - b.xyz[:, 0] * st[b.ev]
        zmin.append(float(zt.min())); over.append(float((zt - D[b.ev]).max()))
        pts = C["fold"](b, sh.box, 64, gen)
        ztp = pts[..., 2] * ct[b.ev, None] - pts[..., 0] * st[b.ev, None]
        fz.append(float(ztp.min())); fo.append(float((ztp - D[b.ev, None]).max()))
    res[f"{split}: hits min z_trans"] = min(zmin)
    res[f"{split}: hits max (z_trans - D)"] = max(over)
    res[f"{split}: fold pts min z_trans (should be >= -h={half})"] = min(fz)
    res[f"{split}: fold pts max (z_trans - D) (should be <= 0)"] = max(fo)

# 2. sub-cell resolution of the normalisation: nsub=2 (study) vs 8 vs 16 (mine)
te = data["test"]
idx = torch.randperm(len(te), generator=torch.Generator().manual_seed(3))[:512].to(dev)
b = te.batch(idx)
ct, st, D = r2lib.geom(b)
half = float(te.box[2]) / 2
A2 = C["event_slab"](b, te.box)
A8 = r2lib.my_accept(C, ct, st, D, half, 8, 8)
A16 = r2lib.my_accept(C, ct, st, D, half, 16, 16)
A2m = r2lib.my_accept(C, ct, st, D, half, 2, 2)
res["max |A_study(nsub2) - A_mine(nsub2)|"] = float((A2 - A2m).abs().max())
res["mean |A2 - A16|"] = float((A2 - A16).abs().mean())
res["mean |A8 - A16|"] = float((A8 - A16).abs().mean())
with torch.no_grad():
    for name in ("PS_slab_chosen", "ref_slab_gflash_n6", "report_gflash_n6", "report_mlp", "ref_slab_mlp_L3xH16"):
        P = M[name](b.lnp).exp()
        lm = {k: torch.log((P * A).sum((1, 2))) for k, A in (("2", A2), ("8", A8), ("16", A16))}
        res[f"{name}: mean lnM(nsub2) - lnM(nsub16)"] = float((lm["2"] - lm["16"]).mean())
        res[f"{name}: mean lnM(nsub8) - lnM(nsub16)"] = float((lm["8"] - lm["16"]).mean())
        res[f"{name}: mean slab mass M (nsub16)"] = float(torch.exp(lm["16"]).mean())
        # 3. brute-force MC of the slab mass (points drawn from the model, uniform inside each cell, uniform azimuth)
        g = torch.Generator(device=dev).manual_seed(5)
        Mmc = []
        for j in range(0, len(idx), 64):
            Pj = P[j:j + 64].reshape(-1 if False else P[j:j + 64].shape[0], -1)
            NS = 40000
            cell = torch.multinomial(Pj, NS, True, generator=g)
            iz, ir = cell // C["N_R"], cell % C["N_R"]
            u = torch.rand(Pj.shape[0], NS, 3, device=dev, generator=g)
            z = C["Z_EDGES"][iz] + u[..., 0] * C["DZ"]
            r = torch.sqrt(C["R_EDGES"][ir] ** 2 + u[..., 1] * (C["R_EDGES"][ir + 1] ** 2 - C["R_EDGES"][ir] ** 2))
            x = r * torch.cos(2 * math.pi * u[..., 2])
            zt = z * ct[j:j + 64, None] - x * st[j:j + 64, None]
            Mmc.append(((zt >= -half) & (zt <= D[j:j + 64, None])).float().mean(1))
        Mmc = torch.cat(Mmc)
        d = lm["16"] - torch.log(Mmc)
        res[f"{name}: MC check mean(lnM_mine16 - lnM_MC) +- SE"] = [float(d.mean()), float(d.std() / math.sqrt(len(d)))]

# 4. Abar: recomputed (train only, front+back) vs the buffers stored in the checkpoints
Ackpt = {k: torch.exp(M[k].logA) for k in ("PS_stdacc_chosen", "ref_std_gflash_acc_n6", "g2_std_mlp_acc")}
for k, A in Ackpt.items():
    res[f"Abar recomputed vs ckpt {k}: max |diff|"] = float((A - A_bar.clamp_min(1e-6)).abs().max())
A_vt = shapelab.mean_acceptance(C, data["val"], front=True, back=True)
res["Abar(train) vs Abar(val): max |diff|"] = float((A_vt - A_bar).abs().max())
# lnp range
for k in ("train", "val", "test"):
    p = data[k].ev["p"]
    res[f"{k}: p range"] = [float(p.min()), float(p.max())]
res["sec"] = round(time.time() - t0)
json.dump(res, open(f"{r2lib.OUT}/check_geometry.json", "w"), indent=1)
print(json.dumps(res, indent=1))
