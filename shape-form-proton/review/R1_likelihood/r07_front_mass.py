"""R1 audit, step 7: where the slab-trained models put the mass that the slab likelihood does not constrain (by momentum)."""
import json, math, os, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton"); sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R1_likelihood")
import numpy as np, torch
import shapelab, rmodels
OUT = "/home/user/tamas/shape-form-proton/review/R1_likelihood"
C, geo, data = shapelab.setup(batch=256, out_dir=OUT + "/scratch", slab=False)
dev = C["DEVICE"]
models = rmodels.load_all(C, data, which=["report_gflash_n6", "report_mlp", "new_slab_gce_n6", "new_stdacc_gce_n6", "ref_slab_gflash_n6", "ref_slab_mlp"])
with torch.no_grad():
    Amax = torch.zeros(C["N_Z"], C["N_R"], device=dev)
    for b in data["train"].batches(512):
        Amax = torch.maximum(Amax, C["event_slab"](b, data["train"].box, nsub=4).max(0).values)
never = (Amax <= 0).float()
front = (C["ZZ"] < -C["DZ"]).float()            # cells entirely in front of z' = -12.9 mm
res = {}
P_SHOW = (0.5, 1.0, 2.0, 4.5, 14.5)
with torch.no_grad():
    lnp = torch.log(torch.tensor(P_SHOW, device=dev))
    for mn, m in models.items():
        P = m(lnp).exp()
        res[mn] = {f"p={p}": {"mass in cells outside every training slab": float((P[i] * never).sum()), "mass at z' < -12.9 mm": float((P[i] * front).sum())}
                   for i, p in enumerate(P_SHOW)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(OUT, "r07_front_mass.json"), "w"), indent=1)
