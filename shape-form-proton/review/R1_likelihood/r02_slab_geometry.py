"""R1 audit, step 2: are all hits and all fold() Monte-Carlo points inside the event's slab -h <= z' cos(t) - x' sin(t) <= D ?

Uses the cached data of s1 (train/val/test) and s2, the UNCHANGED 00_common code (fold, Showers) and the exact random stream
of test_nll (batches of 128, generator seed 12345 on the device, K = 64).  Also counts fold points that leave the grid
(locate() clamps them to border cells) and checks the algebra of z_trans with the full rotation matrix."""
import json
import math
import os
import sys

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np
import torch

import protonlab
import shapelab

OUT = "/home/user/tamas/shape-form-proton/review/R1_likelihood"
C, geo, data = shapelab.setup(batch=256, out_dir=OUT + "/scratch", slab=False)
dev = C["DEVICE"]
ck2 = torch.load(os.path.join(protonlab.HERE, "cache", "proton_s2_sets.pt"), weights_only=False)
sets = dict(data)
for k, v in ck2["sets"].items():
    sets["s2_" + k] = C["Showers"](v["ev"], v["off"], v["xyz"], v["w"], v["box"]).to(dev)
print("geo s1", geo, "| geo s2", ck2["geo"], flush=True)

res = {"geo_s1": geo, "geo_s2": ck2["geo"]}
LAYER = float(data["train"].box[2])
h = LAYER / 2
for name, sh in sets.items():
    gen = torch.Generator(device=dev).manual_seed(12345)
    st = {"n_events": len(sh), "n_hits": int(len(sh.xyz)), "box": [float(x) for x in sh.box]}
    mins, maxs, lat, fmin, fmax, w_out, w_tot, n_out, n_pts = [], [], [], [], [], 0.0, 0.0, 0, 0
    sin_dev, r12 = 0.0, 0.0
    first_layer_frac, zt_min_ev = [], []
    for b in sh.batches(128):
        cos_t = b.R[:, 2, 2]
        sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
        sin_dev = max(sin_dev, float((sin_t + b.R[:, 0, 2]).abs().max()))      # R[0,2] = -sin(theta)
        r12 = max(r12, float(b.R[:, 1, 2].abs().max()))                          # R[1,2] = 0
        D = b.zb * cos_t
        # z_trans of the hit = third component of R^T x_local (the full inverse rotation)
        zt_full = (b.R[b.ev][:, :, 2] * b.xyz).sum(1)
        zt = b.xyz[:, 2] * cos_t[b.ev] - b.xyz[:, 0] * sin_t[b.ev]
        st.setdefault("max |z_trans(formula) - z_trans(R^T x)| [mm]", 0.0)
        st["max |z_trans(formula) - z_trans(R^T x)| [mm]"] = max(st["max |z_trans(formula) - z_trans(R^T x)| [mm]"], float((zt - zt_full).abs().max()))
        mins.append(float(zt.min()))
        maxs.append(float((zt - (D[b.ev] - h)).max()))
        lat.append(float((zt / LAYER - torch.round(zt / LAYER)).abs().max() * LAYER))
        # per-event minimum z_trans (is the origin layer populated after the cuts?)
        zmin = torch.full((len(b.lnp),), float("inf"), device=dev).scatter_reduce(0, b.ev, zt, "amin")
        zt_min_ev.append(zmin.cpu())
        # the fold points of test_nll (same generator, same K)
        pts = C["fold"](b, sh.box, C["N_FOLD_EVAL"], gen)
        ztp = pts[..., 2] * cos_t[b.ev][:, None] - pts[..., 0] * sin_t[b.ev][:, None]
        fmin.append(float((ztp + h).min()))
        fmax.append(float((ztp - D[b.ev][:, None]).max()))
        r = torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
        out = (pts[..., 2] < C["Z_MIN"]) | (pts[..., 2] > C["Z_MAX"]) | (r >= C["R_MAX"])
        w_out += float((b.w[:, None] * out.float()).sum() / pts.shape[1])
        w_tot += float(b.w.sum())
        n_out += int(out.sum())
        n_pts += out.numel()
    zmin_all = torch.cat(zt_min_ev)
    st.update({"min hit z_trans [mm]": min(mins), "max hit (z_trans - (D - h)) [mm]": max(maxs),
               "max |hit z_trans - nearest layer multiple| [mm]": max(lat),
               "min fold-point (z_trans + h) [mm]": min(fmin), "max fold-point (z_trans - D) [mm]": max(fmax),
               "fold points outside the grid: fraction of points": n_out / n_pts,
               "fold points outside the grid: hit-weight fraction per event": w_out / w_tot * len(sh) / len(sh),
               "max |sin_t + R[0,2]|": sin_dev, "max |R[1,2]|": r12,
               "events whose origin layer has no hit after the cuts (min z_trans > 1 mm)": float((zmin_all > 1).float().mean()),
               "D range [mm]": [float((sh.ev["zb"] * sh.ev["R"][:, 2, 2]).min()), float((sh.ev["zb"] * sh.ev["R"][:, 2, 2]).max())],
               "theta range": [float(torch.acos(sh.ev["R"][:, 2, 2]).min()), float(torch.acos(sh.ev["R"][:, 2, 2]).max())]})
    res[name] = st
    print(name, json.dumps(st), flush=True)
json.dump(res, open(os.path.join(OUT, "r02_slab_geometry.json"), "w"), indent=1)
