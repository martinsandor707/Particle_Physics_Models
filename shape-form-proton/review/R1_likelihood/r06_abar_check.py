"""R1 audit, step 6 (claim C2): is the fixed map Abar(z', r) inside the "x Abar" checkpoints computed from the TRAINING events
only, is it really fixed (not trained), and how much do the reported gains depend on it / on the optimiser?

 1. recompute Abar = shapelab.mean_acceptance(C, split, nsub=4, front=True, back=True) for train / val / test / all and compare with
    the logA buffers stored in every "x Abar" checkpoint;
 2. sensitivity: evaluate the stored models with Abar(test) in place of the stored map (what a leak could have bought);
 3. paired test/val differences in the report's metric: GFlash n=6 x Abar (fit_fast) vs plain GFlash n=6 refitted with the same
    fit_fast ladder (lane G2) and vs the report's own fit; and the same differences under a properly normalised likelihood
    (union of detector cells, lane G1 acceptance, see r04)."""
import json
import math
import os
import sys

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R1_likelihood")
import numpy as np
import torch

import shapelab
import rmodels

OUT = "/home/user/tamas/shape-form-proton/review/R1_likelihood"
G2 = "/home/user/tamas/shape-form-proton/lanes/G2_reference/results_std/proton"
FINAL = "/home/user/tamas/shape-form-proton/results/final/proton"
C, geo, data = shapelab.setup(batch=256, out_dir=OUT + "/scratch", slab=False)
dev = C["DEVICE"]
res = {}
Ab = {k: shapelab.mean_acceptance(C, data[k], nsub=4, front=True, back=True) for k in ("train", "val", "test")}
n = {k: len(data[k]) for k in Ab}
Ab["all"] = sum(Ab[k] * n[k] for k in n) / sum(n.values())
logAb = {k: torch.log(v.clamp_min(1e-6)) for k, v in Ab.items()}
res["max |Abar(train) - Abar(test)|, max |Abar(train) - Abar(val)|"] = [float((Ab["train"] - Ab["test"]).abs().max()), float((Ab["train"] - Ab["val"]).abs().max())]
ckpts = {f"stdacc_gce_n{k}": os.path.join(FINAL, f"stdacc_gce_n{k}.pt") for k in range(1, 7)}
ckpts.update({"stdacc_gce_chosen": os.path.join(FINAL, "stdacc_gce_chosen.pt"), "ref_std_gflash_acc_n6": os.path.join(FINAL, "ref_std_gflash_acc_n6.pt"),
              "G2 fast_std_gflash_accboth_n6": os.path.join(G2, "fast_std_gflash_accboth_n6.pt"),
              "G2 ref_std_gflash_accboth_n6 (report protocol)": os.path.join(G2, "ref_std_gflash_accboth_n6.pt")})
tab = {}
for name, p in ckpts.items():
    if not os.path.exists(p):
        tab[name] = "missing"
        continue
    st = torch.load(p, map_location=dev, weights_only=False)["state"]
    la = st["logA"].float()
    tab[name] = {f"max |logA - log Abar({k})| over cells with Abar>1e-3": float((la - logAb[k])[Ab[k] > 1e-3].abs().max()) for k in logAb}
res["1) stored logA buffers vs recomputed Abar (front+back, nsub=4)"] = tab
print(json.dumps(res, indent=1), flush=True)

# 2) sensitivity to the split used for Abar
models = rmodels.load_all(C, data, which=["new_stdacc_gce_n6", "ref_std_gflash_acc_n6", "report_gflash_n6", "report_mlp"])
sens = {}
for mn in ("new_stdacc_gce_n6", "ref_std_gflash_acc_n6"):
    m = models[mn]
    t0 = C["test_nll"](m, data["test"])
    keep = m.logA.clone()
    with torch.no_grad():
        m.logA.copy_(logAb["test"])
    t1 = C["test_nll"](m, data["test"])
    with torch.no_grad():
        m.logA.copy_(keep)
    d = t1 - t0
    sens[mn] = {"test NLL with stored Abar(train)": float(t0.mean()), "test NLL with Abar(test)": float(t1.mean()),
                "paired change (mean, SE)": [float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))],
                "logA is a buffer (not a Parameter)": "logA" in dict(m.named_buffers()) and "logA" not in dict(m.named_parameters()),
                "n_params() of the wrapper": int(m.n_params())}
res["2) Abar(test) instead of Abar(train) on the test set"] = sens
print(json.dumps(sens, indent=1), flush=True)

# 3) optimiser vs Abar: plain GFlash n=6 refitted with the same fit_fast ladder (lane G2)
c, s = shapelab.lnp_center_scale(data)
f = C["Formula"]("gamma", "core_tail", degree=6, lnp0=c, lnp_s=s).to(dev)
st = torch.load(os.path.join(G2, "fast_std_gflash_n6.pt"), map_location=dev, weights_only=False)["state"]
f.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("base.")} if any(k.startswith("base.") for k in st) else st)
f.eval()
ev = {}
for split in ("val", "test"):
    a = C["test_nll"](models["ref_std_gflash_acc_n6"], data[split])
    p = C["test_nll"](f, data[split])
    r = C["test_nll"](models["report_gflash_n6"], data[split])
    ev[split] = {"GFlash n6 x Abar (fit_fast)": float(a.mean()), "GFlash n6 plain (fit_fast, lane G2)": float(p.mean()), "report GFlash n6": float(r.mean()),
                 "x Abar - plain fit_fast (paired)": list(shapelab.paired(a, p)), "x Abar - report (paired)": list(shapelab.paired(a, r)),
                 "plain fit_fast - report (paired)": list(shapelab.paired(p, r))}
res["3) report metric: GFlash n=6 x Abar vs plain refit with the same optimiser"] = ev
print(json.dumps(ev, indent=1), flush=True)

# the same under a properly normalised likelihood (G1 'cells' acceptance), test set
acc = torch.load("/home/user/tamas/shape-form-proton/lanes/G1_geometry/cache/acc_cells_test.pt", weights_only=False)
te = data["test"]


@torch.no_grad()
def nll_cells(m):
    gen = torch.Generator(device=dev).manual_seed(12345)
    out, start = [], 0
    for b in te.batches(128):
        k = len(b.lnp)
        logP = m(b.lnp)
        pts = C["fold"](b, te.box, C["N_FOLD_EVAL"], gen)
        iz, ir = C["locate"](pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
        logg = torch.logsumexp((logP - C["LOG_VOL"])[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(C["N_FOLD_EVAL"])
        S = -torch.zeros(k, device=dev).index_add_(0, b.ev, b.w * logg)
        M = (logP.exp().double() * acc[start:start + k].to(dev).double()).sum((1, 2))
        out.append((S.double() + torch.log(M)).cpu())
        start += k
    return torch.cat(out).numpy()


a, p, r = nll_cells(models["ref_std_gflash_acc_n6"]), nll_cells(f), nll_cells(models["report_gflash_n6"])
res["3b) properly normalised (cells) test NLL: x Abar - plain fit_fast, x Abar - report"] = [list(shapelab.paired(a, p)), list(shapelab.paired(a, r))]
print(res["3b) properly normalised (cells) test NLL: x Abar - plain fit_fast, x Abar - report"], flush=True)

# 4) does Abar depend on p (incidence-angle distribution per momentum bin)?
tr = data["train"]
th = torch.acos(tr.ev["R"][:, 2, 2]).cpu().numpy()
bins = tr.ev["bin"].cpu().numpy()
res["4) train: mean theta per momentum bin (0..19)"] = [float(th[bins == k].mean()) for k in range(20)]
json.dump(res, open(os.path.join(OUT, "r06_abar_check.json"), "w"), indent=1)
print("done", flush=True)
