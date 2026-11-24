"""A belépés előtti tartomány (z' < 0) a beesési szög függvényében (független s2 minta, p ≥ 2 GeV).

A profil-mérés szerint minden sima (z', r)-modell (az új formula és a háló is) alulbecsli a z' < 0 tartományba eső energiát.
Ha ez a ferde belépési sík geometriájából jön (a forgástest-modell nem függ θ-tól), akkor az eltérésnek θ-val nőnie kell.
Kimenet: results/particles/<p>/front_region.json — θ-harmadonként az adat és a modellek z' < 0 energiaaránya.
    python front_region.py <particle>"""
import json
import math
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch

import shapelab

particle = sys.argv[1]
D = os.path.join(ROOT, "results", "particles", particle)
C, geo, d2 = shapelab.setup_particle(particle, out_dir=D, slab=False, sample="s2")
dev = C["DEVICE"]
new = shapelab.build_final(C, particle, D, 0.0, 1.0, metric="slab")[0].eval()
sm = torch.load(os.path.join(D, "mlp_slab.pt"), map_location=dev, weights_only=False)["state"]
n_lin = sum(1 for k in sm if k.startswith("mlp.") and k.endswith(".weight"))
mlp = C["FieldModel"](sm["mlp.0.weight"].shape[0], n_lin - 1).to(dev)
mlp.load_state_dict(sm); mlp.eval()
ZE, RE, DZ, N_R = C["Z_EDGES"], C["R_EDGES"], C["DZ"], C["N_R"]


@torch.no_grad()
def front_share_model(model, sh, M=256, seed=7):
    """Per event: share of the model's in-slab deposits (rho x cell x cell) with z' < 0."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for b in sh.batches(128):
        n = len(b.lnp)
        A = C["event_slab"](b, sh.box, 4)
        cell = torch.multinomial((model(b.lnp).exp() * A).reshape(n, -1) + 1e-30, M, True, generator=gen)
        iz, ir = cell // N_R, cell % N_R
        u = torch.rand(n, M, 3, device=dev, generator=gen)
        z = ZE[iz] + u[..., 0] * DZ
        r = torch.sqrt(RE[ir] ** 2 + u[..., 1] * (RE[ir + 1] ** 2 - RE[ir] ** 2))
        pts = torch.stack([r * torch.cos(2 * math.pi * u[..., 2]), r * torch.sin(2 * math.pi * u[..., 2]), z], -1)
        ct = b.R[:, 2, 2][:, None]
        zt = pts[..., 2] * ct - pts[..., 0] * torch.sqrt(1 - ct ** 2)
        keep = (zt >= -float(sh.box[2]) / 2) & (zt <= b.zb[:, None] * ct)
        box = (torch.rand(n, M, 3, device=dev, generator=gen) - 0.5) * sh.box + (torch.rand(n, M, 3, device=dev, generator=gen) - 0.5) * sh.box
        pts = pts + torch.einsum("bij,bmj->bmi", b.R, box)
        a = A[torch.arange(n, device=dev)[:, None], iz, ir].clamp_min(1e-6)      # 1/A weight: kept deposits follow P*A, not P*A^2
        wk = keep.float() / a
        f = (wk * (pts[..., 2] < 0)).sum(1) / wk.sum(1).clamp_min(1e-30)
        out.append(f)
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def front_share_data(sh, K=8, seed=3):
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for b in sh.batches(256):
        pts = C["fold"](b, sh.box, K, gen)
        w = (b.w / K)[:, None].expand(-1, K)
        f = torch.zeros(len(b.lnp), device=dev).index_add_(0, b.ev, (w * (pts[..., 2] < 0)).sum(1))
        out.append(f)
    return torch.cat(out).cpu().numpy()


sets = [s.subset((s.ev["p"] >= 2).nonzero().squeeze(1)) for s in d2.values()]
theta = np.concatenate([torch.acos(s.ev["R"][:, 2, 2]).cpu().numpy() for s in sets])
fd = np.concatenate([front_share_data(s) for s in sets])
fn = np.concatenate([front_share_model(new, s) for s in sets])
fm = np.concatenate([front_share_model(mlp, s) for s in sets])
q = np.quantile(theta, [0, 1 / 3, 2 / 3, 1])
res = {"particle": particle, "theta_edges_rad": q.tolist(), "bins": []}
for i in range(3):
    s = (theta >= q[i]) & (theta <= q[i + 1])
    res["bins"].append({"theta_mean": float(theta[s].mean()), "n": int(s.sum()), "data": float(fd[s].mean()),
                        "new formula": float(fn[s].mean()), "MLP": float(fm[s].mean())})
json.dump(res, open(os.path.join(D, "front_region.json"), "w"), indent=1)
print(particle, json.dumps(res["bins"]), flush=True)
