"""R2_numbers: evaluate the fit_fast fine-tuned report MLPs (slab / std) on S1 val/test and on s2; paired vs the new formula."""
import json, os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
import numpy as np, torch
import r2lib
from r2lib import OUT, paired

C, geo, data = r2lib.setup()
dev = C["DEVICE"]
Z1 = np.load(f"{OUT}/per_event_s1.npz")
Z2 = np.load(f"{OUT}/per_event_s2.npz")
ck2 = torch.load("/home/user/tamas/shape-form-proton/cache/proton_s2_sets.pt", weights_only=False)
S2 = {k: C["Showers"](v["ev"], v["off"], v["xyz"], v["w"], v["box"]).to(dev) for k, v in ck2["sets"].items()}
res = {}
for metric, f in (("slab", "mlp_slab_s6000_lr0.001_report.pt"), ("std", "mlp_std_s6000_lr0.001_report.pt")):
    p = os.path.join(OUT, "mlp_fastfit", f)
    if not os.path.exists(p):
        res[metric] = "missing"; continue
    m = C["FieldModel"](16, 3).to(dev)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state"]); m.eval()
    per = {}
    for sp in ("val", "test"):
        parts = r2lib.eval_parts(C, m, data[sp], nsub=8)
        per[("std", sp)], per[("slab8", sp)], per[("slab2", sp)] = C["test_nll"](m, data[sp]), parts["slab_mine"], parts["slab_sl2"]
    s_std, s_8, s_2 = [], [], []
    for k, sh in S2.items():
        s_std.append(C["test_nll"](m, sh))
        parts = r2lib.eval_parts(C, m, sh, nsub=8, want=("slab_mine", "slab_sl2"))
        s_8.append(parts["slab_mine"]); s_2.append(parts["slab_sl2"])
    per[("std", "s2")], per[("slab8", "s2")], per[("slab2", "s2")] = np.concatenate(s_std), np.concatenate(s_8), np.concatenate(s_2)
    r = {f"{mm}:{sp}": float(v.mean()) for (mm, sp), v in per.items()}
    if metric == "slab":
        for mm in ("slab2", "slab8"):
            for sp in ("val", "test", "s2"):
                ps = Z1[f"PS_slab_chosen|{mm}|{sp}"] if sp != "s2" else Z2[f"PS_slab_chosen|{mm}"]
                ref = Z1[f"ref_slab_mlp_L3xH16|{mm}|{sp}"] if sp != "s2" else Z2[f"ref_slab_mlp_L3xH16|{mm}"]
                g = Z1[f"ref_slab_gflash_n6|{mm}|{sp}"] if sp != "s2" else Z2[f"ref_slab_gflash_n6|{mm}"]
                r[f"PS_slab - MLP_slab_ft [{mm},{sp}]"] = paired(ps, per[(mm, sp)])
                r[f"MLP_slab_ft - MLP 3D-trained (G2) [{mm},{sp}]"] = paired(per[(mm, sp)], ref)
                r[f"like-with-like gap fraction vs MLP_slab_ft [{mm},{sp}]"] = float((g.mean() - ps.mean()) / (g.mean() - per[(mm, sp)].mean()))
    else:
        for sp in ("val", "test", "s2"):
            ps = Z1[f"PS_stdacc_chosen|std|{sp}"] if sp != "s2" else Z2["PS_stdacc_chosen|std"]
            ref = Z1[f"report_mlp|std|{sp}"] if sp != "s2" else Z2["report_mlp|std"]
            g = Z1[f"ref_std_gflash_acc_n6|std|{sp}"] if sp != "s2" else Z2["ref_std_gflash_acc_n6|std"]
            r[f"PS_stdacc - MLP_std_ft [std,{sp}]"] = paired(ps, per[("std", sp)])
            r[f"MLP_std_ft - report MLP [std,{sp}]"] = paired(per[("std", sp)], ref)
            r[f"like-with-like gap fraction vs MLP_std_ft [std,{sp}]"] = float((g.mean() - ps.mean()) / (g.mean() - per[("std", sp)].mean()))
    res[metric] = r
    print(metric, json.dumps(r, indent=1), flush=True)
json.dump(res, open(f"{OUT}/eval_mlp_ft.json", "w"), indent=1)
