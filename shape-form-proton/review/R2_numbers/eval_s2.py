"""R2_numbers: evaluation on the independent second sample (cache/proton_s2_sets.pt, all ~100k events), both metrics."""
import json, math, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
import numpy as np, torch
import r2lib, shapelab
from r2lib import paired

t0 = time.time()
C, geo, data = r2lib.setup()
dev = C["DEVICE"]
A_bar = shapelab.mean_acceptance(C, data["train"], front=True, back=True)        # from the ORIGINAL training events (part of the model)
M = r2lib.build_models(C, data, A_bar, rungs=True)
keep = ["PS_slab_chosen", "PS_slab_n1", "PS_slab_n2", "PS_stdacc_chosen", "PS_stdacc_n1", "PS_stdacc_n2", "report_gflash_n6",
        "report_mlp", "ref_std_gflash_acc_n6", "ref_slab_gflash_n6", "ref_slab_mlp_L3xH16", "g2_std_mlp_acc"]
ck2 = torch.load("/home/user/tamas/shape-form-proton/cache/proton_s2_sets.pt", weights_only=False)
res = {"s2 geo": ck2["geo"], "s1 geo": geo}
S2 = {k: C["Showers"](v["ev"], v["off"], v["xyz"], v["w"], v["box"]).to(dev) for k, v in ck2["sets"].items()}
res["s2 sets"] = {k: len(v) for k, v in S2.items()}
res["s2 n events"] = int(sum(len(v) for v in S2.values()))
res["s2 box"] = [float(x) for x in next(iter(S2.values())).box]
res["s1 box"] = [float(x) for x in data["train"].box]
p2 = torch.cat([v.ev["p"] for v in S2.values()])
ptr = data["train"].ev["p"]
res["s2 p range"] = [float(p2.min()), float(p2.max())]
res["s2 events below/above the s1 training p range"] = [int((p2 < ptr.min()).sum()), int((p2 > ptr.max()).sum())]


# ---- overlap check: identical events in s1 and s2? key = (p, cos theta, R[0,1], evis, n hits)
def keys(sh):
    nh = (sh.off[1:] - sh.off[:-1]).cpu().numpy()
    a = np.stack([sh.ev["p"].cpu().numpy(), sh.ev["R"][:, 2, 2].cpu().numpy(), sh.ev["R"][:, 0, 1].cpu().numpy(),
                  sh.ev["evis"].cpu().numpy()], 1)
    return {(round(float(x[0]), 6), round(float(x[1]), 6), round(float(x[2]), 6), round(float(x[3]), 5), int(n)) for x, n in zip(a, nh)}
k1 = set().union(*[keys(data[s]) for s in ("train", "val", "test")])
k2 = set().union(*[keys(v) for v in S2.values()])
res["identical event keys s1 & s2"] = len(k1 & k2)
k1p = set().union(*[{round(float(x), 6) for x in data[s].ev["p"].cpu().numpy()} for s in ("train", "val", "test")])
k2p = {round(float(x), 6) for x in p2.cpu().numpy()}
res["identical momentum values (6 digits) s1 & s2"] = len(k1p & k2p)

PER = {}
for name in keep:
    m = M[name]
    std, s8, s2_ = [], [], []
    for k, sh in S2.items():
        std.append(C["test_nll"](m, sh))
        parts = r2lib.eval_parts(C, m, sh, nsub=8, want=("slab_mine", "slab_sl2"))
        s8.append(parts["slab_mine"]); s2_.append(parts["slab_sl2"])
    PER[(name, "std")] = np.concatenate(std)
    PER[(name, "slab8")] = np.concatenate(s8)
    PER[(name, "slab2")] = np.concatenate(s2_)
    print(f"{name:24s} s2: std {PER[(name,'std')].mean():.5f} | slab8 {PER[(name,'slab8')].mean():.5f} | slab2 {PER[(name,'slab2')].mean():.5f} ({time.time()-t0:.0f}s)", flush=True)
bins2 = torch.cat([v.ev["bin"] for v in S2.values()]).cpu().numpy()
np.savez_compressed(f"{r2lib.OUT}/per_event_s2.npz", **{"|".join(k): v for k, v in PER.items()}, bins=bins2)
res["means"] = {f"{n}|{m}": round(float(v.mean()), 5) for (n, m), v in PER.items()}

comps = {
    "3D: PS_slab - report GFlash n6 (renorm)": ("PS_slab_chosen", "report_gflash_n6", "slab"),
    "3D: PS_slab - GFlash n6 3D refit": ("PS_slab_chosen", "ref_slab_gflash_n6", "slab"),
    "3D: PS_slab - report MLP (renorm)": ("PS_slab_chosen", "report_mlp", "slab"),
    "3D: PS_slab - MLP 3D-trained": ("PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab"),
    "3D: PS_slab n2 - GFlash n6 3D refit": ("PS_slab_n2", "ref_slab_gflash_n6", "slab"),
    "3D: PS_slab n1 - GFlash n6 3D refit": ("PS_slab_n1", "ref_slab_gflash_n6", "slab"),
    "3D: PS_slab n6 - PS_slab n2": ("PS_slab_chosen", "PS_slab_n2", "slab"),
    "3D: GFlash 3D refit - report GFlash renorm": ("ref_slab_gflash_n6", "report_gflash_n6", "slab"),
    "3D: MLP 3D-trained - report MLP renorm": ("ref_slab_mlp_L3xH16", "report_mlp", "slab"),
    "std: PS_stdacc - report GFlash n6": ("PS_stdacc_chosen", "report_gflash_n6", "std"),
    "std: PS_stdacc - GFlash n6 x Abar refit": ("PS_stdacc_chosen", "ref_std_gflash_acc_n6", "std"),
    "std: PS_stdacc - report MLP": ("PS_stdacc_chosen", "report_mlp", "std"),
    "std: PS_stdacc - MLP x Abar (G2)": ("PS_stdacc_chosen", "g2_std_mlp_acc", "std"),
    "std: PS_stdacc n2 - GFlash n6 x Abar refit": ("PS_stdacc_n2", "ref_std_gflash_acc_n6", "std"),
    "std: PS_stdacc n1 - GFlash n6 x Abar refit": ("PS_stdacc_n1", "ref_std_gflash_acc_n6", "std"),
    "std: PS_stdacc n6 - PS_stdacc n2": ("PS_stdacc_chosen", "PS_stdacc_n2", "std"),
    "std: GFlash x Abar refit - report GFlash": ("ref_std_gflash_acc_n6", "report_gflash_n6", "std"),
    "std: MLP x Abar (G2) - report MLP": ("g2_std_mlp_acc", "report_mlp", "std"),
}
pc = {}
for lab, (a, b, metric) in comps.items():
    for mm in (("slab8", "slab2") if metric == "slab" else ("std",)):
        pc[f"{lab} [{mm}]"] = [round(x, 6) for x in paired(PER[(a, mm)], PER[(b, mm)])]
res["paired s2 (mean, SE)"] = pc


def frac(base, new, top, mm):
    g = PER[(base, mm)].mean() - PER[(top, mm)].mean()
    n = PER[(base, mm)].mean() - PER[(new, mm)].mean()
    return [round(float(n / g), 4), round(float(n), 5), round(float(g), 5)]


res["gap fractions s2 (fraction, closed, gap)"] = {
    "notebook def std": frac("report_gflash_n6", "PS_stdacc_chosen", "report_mlp", "std"),
    "notebook def 3D slab2": frac("report_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab2"),
    "like-with-like std (GFlash x Abar refit, MLP x Abar)": frac("ref_std_gflash_acc_n6", "PS_stdacc_chosen", "g2_std_mlp_acc", "std"),
    "like-with-like 3D slab2 (GFlash 3D refit, MLP 3D-trained)": frac("ref_slab_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab2"),
    "like-with-like 3D slab8": frac("ref_slab_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab8")}
pb = {}
for lab, (a, b, mm) in {"3D slab8: PS_slab - MLP 3D-trained": ("PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab8"),
                        "std: PS_stdacc - report MLP": ("PS_stdacc_chosen", "report_mlp", "std")}.items():
    d = PER[(a, mm)].astype(np.float64) - PER[(b, mm)].astype(np.float64)
    row = {}
    for nm, sel in (("bin0", bins2 == 0), ("bin1", bins2 == 1), ("bins2-4", (bins2 >= 2) & (bins2 < 5)), ("bins5-19", bins2 >= 5),
                    ("bins2-19", bins2 >= 2)):
        x = d[sel]
        row[nm] = [round(float(x.mean()), 5), round(float(x.std(ddof=1) / np.sqrt(len(x))), 5), round(float(x.sum() / len(d)), 5)]
    row["per bin 2..19"] = [round(float(d[bins2 == k].mean()), 4) for k in range(2, 20)]
    pb[lab] = row
res["per bin s2 (mean, SE, contribution)"] = pb
res["sec"] = round(time.time() - t0)
json.dump(res, open(f"{r2lib.OUT}/eval_s2.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "means"}, indent=1))
