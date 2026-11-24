"""R1 audit, step 8: contribution of events whose origin layer has no hit after the cuts (origin set by late hits) and of
events whose first hit layer is not layer 0, to the per-model NLL and to paired differences (test set, terms from r04)."""
import json, os, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import shapelab
OUT = "/home/user/tamas/shape-form-proton/review/R1_likelihood"
C, geo, data = shapelab.setup(batch=256, out_dir=OUT + "/scratch", slab=False)
te = data["test"]
ev = torch.repeat_interleave(torch.arange(len(te), device=te.off.device), te.off[1:] - te.off[:-1])
ct = te.ev["R"][:, 2, 2][ev]; st = torch.sqrt(1 - ct ** 2)
zt = te.xyz[:, 2] * ct - te.xyz[:, 0] * st
zmin = torch.full((len(te),), float("inf"), device=zt.device).scatter_reduce(0, ev, zt, "amin").cpu().numpy()
T = np.load(os.path.join(OUT, "r04_terms_test.npz"))
lost = zmin > 1.0
fl = T["first_layer_idx"] > 0
res = {"n events origin layer empty after cuts": int(lost.sum()), "n events first hit layer > 0": int(fl.sum()), "overlap": int((lost & fl).sum())}
nll = lambda m, a, b: T[f"{m}|{a}"] + T[f"{m}|{b}"]
for g, msk in (("origin layer empty", lost), ("first hit layer > 0", fl)):
    d = {}
    for a, b in (("report_mlp", "report_gflash_n6"), ("new_slab_gce_n6", "report_gflash_n6"), ("new_slab_gce_n6", "ref_slab_gflash_n6"), ("new_slab_gce_n6", "report_mlp")):
        for met, (x, y) in (("std", ("S", "std")), ("slab", ("S", "slab")), ("cells", ("S_unclamped", "cells"))):
            dd = nll(a, x, y) - nll(b, x, y)
            d[f"{met}: {a} - {b}: mean in group, contribution to the overall mean"] = [float(dd[msk].mean()), float(dd[msk].sum() / len(dd))]
    for m in ("report_gflash_n6", "report_mlp", "new_slab_gce_n6"):
        d[f"{m}: mean slab NLL in group vs rest"] = [float(nll(m, "S", "slab")[msk].mean()), float(nll(m, "S", "slab")[~msk].mean())]
    res[g] = d
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(OUT, "r08_origin_events.json"), "w"), indent=1)
