"""R2_numbers: tables / paired differences / ladder re-selection / per-bin / gap fractions from per_event_s1.npz."""
import json, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
import numpy as np
from r2lib import OUT, FINAL, paired

Z = np.load(f"{OUT}/per_event_s1.npz")
P = {tuple(k.split("|")): Z[k] for k in Z.files if "|" in k}
bins = {"val": Z["bins_val"], "test": Z["bins_test"]}
res = {}
mean = lambda n, m, s: float(P[(n, m, s)].mean())

# ---- 1. table: every model, both metrics (std = 00_common test_nll; slab8 = my normalisation; slab2 = study's nsub=2 closed form)
tab = {}
for (n, m, s), v in P.items():
    tab.setdefault(n, {})[f"{m}:{s}"] = round(float(v.mean()), 5)
res["table"] = tab

# ---- 2. claimed numbers (JSON) vs recomputed
cl = {}
for tag, metric, mine in (("slab", "slab", "slab2"), ("stdacc", "std", "std")):
    js = json.load(open(f"{FINAL}/{tag}_gce_chosen.json"))
    name = f"PS_{tag}_chosen"
    r = {"claimed val": js["val_nll"], "recomputed val (study norm.)": mean(name, mine, "val"),
         "claimed test": js["test_nll"], "recomputed test (study norm.)": mean(name, mine, "test")}
    if tag == "slab":
        r["recomputed val (my nsub=8 norm.)"] = mean(name, "slab8", "val")
        r["recomputed test (my nsub=8 norm.)"] = mean(name, "slab8", "test")
    base_metric = "slab2" if tag == "slab" else "std"
    for k, ref in (("gflash_n6", "report_gflash_n6"), ("mlp", "report_mlp")):
        for s in ("val", "test"):
            r[f"claimed {s}_vs_{k}"] = js[f"{s}_vs_{k}"]
            r[f"recomputed {s}_vs_{k}"] = paired(P[(name, base_metric, s)], P[(ref, base_metric, s)])
    cl[tag] = r
res["claimed_vs_recomputed"] = cl

# ---- 3. like-with-like paired comparisons (test and val), both slab normalisations for the 3D metric
comps = {
    "3D: PS_slab - report GFlash n6 (renorm, no refit)": ("PS_slab_chosen", "report_gflash_n6", "slab"),
    "3D: PS_slab - GFlash n6 3D refit (fit_fast)": ("PS_slab_chosen", "ref_slab_gflash_n6", "slab"),
    "3D: PS_slab - report MLP (renorm, no refit)": ("PS_slab_chosen", "report_mlp", "slab"),
    "3D: PS_slab - MLP 3D-trained": ("PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab"),
    "3D: PS_slab n2 (46 par) - GFlash n6 3D refit": ("PS_slab_n2", "ref_slab_gflash_n6", "slab"),
    "3D: PS_slab n1 (37 par) - GFlash n6 3D refit": ("PS_slab_n1", "ref_slab_gflash_n6", "slab"),
    "3D: GFlash 3D refit - report GFlash renorm": ("ref_slab_gflash_n6", "report_gflash_n6", "slab"),
    "3D: MLP 3D-trained - report MLP renorm": ("ref_slab_mlp_L3xH16", "report_mlp", "slab"),
    "std: PS_stdacc - report GFlash n6": ("PS_stdacc_chosen", "report_gflash_n6", "std"),
    "std: PS_stdacc - GFlash n6 x Abar refit": ("PS_stdacc_chosen", "ref_std_gflash_acc_n6", "std"),
    "std: PS_stdacc - report MLP": ("PS_stdacc_chosen", "report_mlp", "std"),
    "std: PS_stdacc - MLP x Abar (G2, report protocol)": ("PS_stdacc_chosen", "g2_std_mlp_acc", "std"),
    "std: PS_stdacc n2 (46 par) - GFlash n6 x Abar refit": ("PS_stdacc_n2", "ref_std_gflash_acc_n6", "std"),
    "std: PS_stdacc n1 (37 par) - GFlash n6 x Abar refit": ("PS_stdacc_n1", "ref_std_gflash_acc_n6", "std"),
    "std: GFlash x Abar refit - report GFlash": ("ref_std_gflash_acc_n6", "report_gflash_n6", "std"),
    "std: MLP x Abar (G2) - report MLP": ("g2_std_mlp_acc", "report_mlp", "std"),
}
pc = {}
for lab, (a, b, metric) in comps.items():
    ms = ("slab8", "slab2") if metric == "slab" else ("std",)
    for mm in ms:
        for s in ("val", "test"):
            pc[f"{lab} [{mm}, {s}]"] = [round(x, 6) for x in paired(P[(a, mm, s)], P[(b, mm, s)])]
res["paired"] = pc

# ---- 4. ladder re-selection (1-SE rule on validation) in each metric, also with my normalisation
lad = {}
for tag, metric_list in (("slab", ("slab2", "slab8")), ("stdacc", ("std",))):
    for mm in metric_list:
        val = {n: P[(f"PS_{tag}_n{n}", mm, "val")] for n in range(1, 7)}
        nstar = min(val, key=lambda k: val[k].mean())
        d = {n: paired(val[n], val[nstar]) if n != nstar else (0.0, 0.0) for n in val}
        ok = [n for n in val if n == nstar or d[n][0] < d[n][1]]
        lad[f"{tag}/{mm}"] = {"val": {n: round(float(v.mean()), 6) for n, v in val.items()}, "n*": nstar,
                              "delta vs n* (mean, SE)": {n: [round(x, 6) for x in d[n]] for n in d}, "chosen (1-SE)": min(ok),
                              "test of each rung": {n: round(float(P[(f'PS_{tag}_n{n}', mm, 'test')].mean()), 6) for n in range(1, 7)}}
res["ladder"] = lad

# ---- 5. per-bin (test): PS vs MLP refs and GFlash refs
pb = {}
for lab, (a, b, mm) in {"3D slab8: PS_slab - MLP 3D-trained": ("PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab8"),
                        "3D slab8: PS_slab - report MLP renorm": ("PS_slab_chosen", "report_mlp", "slab8"),
                        "3D slab8: PS_slab - GFlash 3D refit": ("PS_slab_chosen", "ref_slab_gflash_n6", "slab8"),
                        "std: PS_stdacc - report MLP": ("PS_stdacc_chosen", "report_mlp", "std"),
                        "std: PS_stdacc - GFlash x Abar refit": ("PS_stdacc_chosen", "ref_std_gflash_acc_n6", "std")}.items():
    for s in ("val", "test"):
        d = P[(a, mm, s)].astype(np.float64) - P[(b, mm, s)].astype(np.float64)
        bb = bins[s]
        row = {}
        for nm, sel in (("bin0 (p<1)", bb == 0), ("bin1 (1-2)", bb == 1), ("bins2-4 (2-5)", (bb >= 2) & (bb < 5)),
                        ("bins5-19 (>=5)", bb >= 5), ("bins2-19 (>=2)", bb >= 2)):
            x = d[sel]
            row[nm] = [round(float(x.mean()), 5), round(float(x.std(ddof=1) / np.sqrt(len(x))), 5),
                       round(float(x.sum() / len(d)), 5)]        # mean per event in the bin, SE, contribution to the total mean
        row["per bin 2..19 mean"] = [round(float(d[bb == k].mean()), 4) for k in range(2, 20)]
        pb[f"{lab} [{s}]"] = row
res["per_bin"] = pb

# ---- 6. gap fractions
gf = {}
def frac(base, new, top, mm, s="test"):
    g = mean(base, mm, s) - mean(top, mm, s)
    n = mean(base, mm, s) - mean(new, mm, s)
    return round(n / g, 4), round(n, 5), round(g, 5)
gf["notebook definition, std: (report GFlash - PS_stdacc)/(report GFlash - report MLP)"] = frac("report_gflash_n6", "PS_stdacc_chosen", "report_mlp", "std")
gf["notebook definition, 3D: (report GFlash renorm - PS_slab)/(report GFlash renorm - MLP 3D-trained) [slab2]"] = frac("report_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab2")
gf["same, 3D [slab8]"] = frac("report_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab8")
gf["like-with-like std: (GFlash x Abar refit - PS_stdacc)/(GFlash x Abar refit - MLP x Abar G2)"] = frac("ref_std_gflash_acc_n6", "PS_stdacc_chosen", "g2_std_mlp_acc", "std")
gf["like-with-like std vs report MLP: (GFlash x Abar refit - PS_stdacc)/(GFlash x Abar refit - report MLP)"] = frac("ref_std_gflash_acc_n6", "PS_stdacc_chosen", "report_mlp", "std")
gf["like-with-like 3D: (GFlash 3D refit - PS_slab)/(GFlash 3D refit - MLP 3D-trained) [slab2]"] = frac("ref_slab_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab2")
gf["like-with-like 3D [slab8]"] = frac("ref_slab_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab8")
gf["3D with the renormalised (not 3D-trained) report MLP [slab2]"] = frac("report_gflash_n6", "PS_slab_chosen", "report_mlp", "slab2")
for k in list(gf):
    if "[slab" in k or "std" in k:
        pass
res["gap_fraction (fraction, closed, gap) on test"] = gf
gfv = {}
gfv["notebook def std (val)"] = frac("report_gflash_n6", "PS_stdacc_chosen", "report_mlp", "std", "val")
gfv["notebook def 3D slab2 (val)"] = frac("report_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab2", "val")
gfv["like-with-like std (val)"] = frac("ref_std_gflash_acc_n6", "PS_stdacc_chosen", "g2_std_mlp_acc", "std", "val")
gfv["like-with-like 3D slab2 (val)"] = frac("ref_slab_gflash_n6", "PS_slab_chosen", "ref_slab_mlp_L3xH16", "slab2", "val")
res["gap_fraction on val"] = gfv
json.dump(res, open(f"{OUT}/analysis_s1.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=1))
