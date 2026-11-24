"""Diagnostic: where (in z', r and p) does the reference formula lose against the slab-renormalised MLP?  Validation set.

Prints, per momentum range, the energy fractions of the data in coarse (z', r) boxes, the slab-normalised model masses of the
reference and of the MLP in the same boxes, and the contribution of the hits of every box to the NLL difference (ref - MLP).
"""
import math
import sys

import numpy as np
import torch

import common
from common import shapelab, families

C, data, base, c, s = common.setup(slab=True)
dev = C["DEVICE"]
ref = families.GammaCoreTail(C, degree=2, lnp0=c, lnp_s=s).to(dev)
ref.load_state_dict(torch.load(common.RES + "/proton/ref_gct2_slab.pt", weights_only=False)["state"])
ref.eval()
mlp = shapelab.baseline_models(C, data)["mlp"]
sh = data[sys.argv[1] if len(sys.argv) > 1 else "val"]

ZB = [-200, -20, 12, 35, 60, 100, 200, 400, 1450]
RB = [0, 1, 30, 60, 120, 250, 700]
PB = [(0.2, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2, 3), (3, 5), (5, 10), (10, 20)]
zb_t, rb_t = torch.tensor(ZB[1:-1], device=dev, dtype=torch.float32), torch.tensor(RB[1:-1], device=dev, dtype=torch.float32)
# grid cell -> coarse box
gz = torch.bucketize(C["ZC"].contiguous(), zb_t)
gr = torch.bucketize(C["RC"].contiguous(), rb_t)
G = (gz[:, None] * (len(RB) - 1) + gr[None, :]).reshape(-1)
NB = (len(ZB) - 1) * (len(RB) - 1)

acc = {k: torch.zeros(len(PB), NB, device=dev, dtype=torch.float64) for k in ("data", "ref", "mlp", "dnll")}
nev = torch.zeros(len(PB), device=dev, dtype=torch.float64)
gen = torch.Generator(device=dev).manual_seed(12345)
with torch.no_grad():
    for b in sh.batches(128):
        pb = torch.full_like(b.p, -1, dtype=torch.long)
        for i, (lo, hi) in enumerate(PB):
            pb[(b.p >= lo) & (b.p < hi)] = i
        A = C["event_slab"](b, sh.box)
        pts = C["fold"](b, sh.box, 64, gen)
        iz, ir = C["locate"](pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
        ll = {}
        for name, m in (("ref", ref), ("mlp", mlp)):
            logP = m(b.lnp)
            PA = logP.exp() * A
            mass = PA.sum((1, 2))
            acc[name].index_put_((pb[:, None].expand(-1, G.numel()).reshape(-1), G[None].expand(len(pb), -1).reshape(-1)),
                                 (PA / mass[:, None, None]).reshape(-1).double(), accumulate=True)
            logrho = logP - C["LOG_VOL"]
            logg = torch.logsumexp(logrho[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(64)
            ll[name] = logg - torch.log(mass)[b.ev]
        hz = torch.bucketize(b.xyz[:, 2].contiguous(), zb_t)
        hr = torch.bucketize(torch.sqrt(b.xyz[:, 0] ** 2 + b.xyz[:, 1] ** 2).contiguous(), rb_t)
        hb = hz * (len(RB) - 1) + hr
        acc["data"].index_put_((pb[b.ev], hb), b.w.double(), accumulate=True)
        acc["dnll"].index_put_((pb[b.ev], hb), (b.w * (ll["mlp"] - ll["ref"])).double(), accumulate=True)
        nev.index_add_(0, pb, torch.ones_like(pb, dtype=torch.float64))

np.set_printoptions(linewidth=220, suppress=True)
for i, (lo, hi) in enumerate(PB):
    n = float(nev[i])
    print(f"\n===== p in [{lo}, {hi}) GeV: {int(n)} events;  NLL(ref) - NLL(mlp) per event = {float(acc['dnll'][i].sum()) / n:+.4f}"
          f"   (contribution to the total: {float(acc['dnll'][i].sum()) / len(sh):+.4f})")
    for k in ("data", "ref", "mlp", "dnll"):
        M = (acc[k][i] / n).reshape(len(ZB) - 1, len(RB) - 1).cpu().numpy()
        print(f"--- {k}: rows z' boxes {ZB}, columns r boxes {RB}; last column = row sum" if k == "data" else f"--- {k}")
        for j, row in enumerate(M):
            print(f"  z'[{ZB[j]:5d},{ZB[j + 1]:5d})  " + " ".join(f"{v:8.4f}" for v in row) + f"   | {row.sum():8.4f}")
        print("  " + " " * 18 + " ".join(f"{v:8.4f}" for v in M.sum(0)) + f"   | {M.sum():8.4f}")
