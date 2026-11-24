"""PS chosen vs GFlash n6 refitted with the same fit_fast budget (4000 steps/rung), paired, S1 val/test."""
import json, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
import numpy as np, torch
from r2lib import OUT, paired
Z = np.load(f"{OUT}/per_event_s1.npz")
res = {}
for mode, ps, metric, ref in (("slab", "PS_slab_chosen", "slab2", "ref_slab_gflash_n6"), ("stdacc", "PS_stdacc_chosen", "std", "ref_std_gflash_acc_n6")):
    ck = torch.load(f"{OUT}/refit_gflash/{mode}_gflash_s4000_n6.pt", map_location="cpu", weights_only=False)
    for sp in ("val", "test"):
        g4 = np.asarray(ck[f"{sp}_nll"])
        res[f"{mode} {sp}: GFlash4000 n6 mean"] = float(g4.mean())
        res[f"{mode} {sp}: GFlash4000 - GFlash3000 (study ref)"] = paired(g4, Z[f"{ref}|{metric}|{sp}"])
        res[f"{mode} {sp}: PS chosen - GFlash4000"] = paired(Z[f"{ps}|{metric}|{sp}"], g4)
        res[f"{mode} {sp}: PS chosen - GFlash3000 (study ref)"] = paired(Z[f"{ps}|{metric}|{sp}"], Z[f"{ref}|{metric}|{sp}"])
json.dump(res, open(f"{OUT}/compare_refit4000.json", "w"), indent=1)
print(json.dumps(res, indent=1))
