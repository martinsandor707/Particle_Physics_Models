"""Gyors próba: a baseline-formula × átlagos akceptancia (Ā) a report standard metrikájában."""
import sys, json
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import torch, numpy as np
import shapelab, families
C, geo, data = shapelab.setup(batch=256, out_dir="/home/user/tamas/shape-form-proton/results_smoke", slab=False)
base = shapelab.baseline_nll(C, data)
c, s = shapelab.lnp_center_scale(data)
h = data["train"].hits()
f0 = C["Formula"]("gamma", "core_tail", degree=1).init_from(h.z.values, h.r.values, h.w.values)
for tag, kw in (("front", dict(front=True, back=False)), ("front+back", dict(front=True, back=True))):
    A = shapelab.mean_acceptance(C, data["train"], **kw)
    for deg in (1, 2, 3):
        core = families.GammaCoreTail(C, degree=deg, lnp0=c, lnp_s=s).to(C["DEVICE"])
        with torch.no_grad():
            for i, n in enumerate(f0.names):
                core.coef[core.names.index(n), 0] = f0.coef[i, 0]
        m = shapelab.WithAcceptance(core, A).to(C["DEVICE"])
        shapelab.fit_fast(C, data, m, steps=3000, log=None)
        res, _, _ = shapelab.evaluate(C, data, m, f"quick_abar_{tag}_n{deg}", base, save=False)
        print(tag, "degree", deg, "params", res["n_params"], "val", round(res["val_nll"], 4), "test", round(res["test_nll"], 4),
              "| test vs gflash_n6:", [round(x, 4) for x in res["test_vs_gflash_n6"]], "| test vs mlp:", [round(x, 4) for x in res["test_vs_mlp"]], flush=True)
