"""Aggregate the per-bin fits: per family mean V (analytic, whole body), V inside the data domain, coverage, calibrated volumes.
usage: s03_table.py [crit_tag]   -> prints the table, writes results/table_<tag>.csv (per family) and results/perbin_<tag>.csv"""
import csv, glob, json, os, sys
import numpy as np
import torch
LANE = "/home/user/tamas/shape-form-proton/lanes/E1_bodies"
tag = sys.argv[1] if len(sys.argv) > 1 else "E95_EV98"
root = os.path.join(LANE, "results", "fits", tag)
rows, per = [], []
for fdir in sorted(glob.glob(os.path.join(root, "*"))):
    f = os.path.basename(fdir)
    R = {}
    for p in glob.glob(os.path.join(fdir, "bin*.json")):
        with open(p) as fh:
            d = json.load(fh)
        R[d["bin"]] = d
    if len(R) < 20:
        print(f"{f}: only {len(R)} bins", file=sys.stderr)
        if len(R) == 0:
            continue
    bins = sorted(R)
    g = lambda k: np.array([R[b][k] for b in bins], float)
    nt, nv, ns = g("n_train"), g("n_val"), g("n_test")
    rows.append(dict(family=f, n_bins=len(bins), n_par=int(R[bins[0]]["n_par"]), V_L=g("V_L").mean(), V_in_L=g("V_in_L").mean(),
                     cov_train=(g("cov_train") * nt).sum() / nt.sum(), cov_val=(g("cov_val") * nv).sum() / nv.sum(),
                     cov_test=(g("cov_test") * ns).sum() / ns.sum(), cov_test_binmean=g("cov_test").mean(),
                     V_cal_val_L=g("V_cal_val_L").mean(), V_cal_test_L=g("V_cal_test_L").mean(),
                     r_max=g("r_max").mean(), z_first=g("z_first").mean(), z_last=g("z_last").mean(), t_fit=g("t_fit").sum()))
    for b in bins:
        d = R[b]
        per.append(dict(family=f, bin=b, p_mean=d["p_mean"], n_par=d["n_par"], V_L=d["V_L"], V_in_L=d["V_in_L"], A=d["A"], r_max=d["r_max"],
                        z_first=d["z_first"], z_last=d["z_last"], cov_train=d["cov_train"], cov_val=d["cov_val"], cov_test=d["cov_test"],
                        V_cal_val_L=d["V_cal_val_L"], V_cal_test_L=d["V_cal_test_L"],
                        params=json.dumps({k: v for k, v in d.items() if k not in ("x", "A", "lnA", "V_L", "V_in_L", "V_grid_L", "r_max", "z_rmax", "z_first", "z_last", "cov_train", "n_train", "cov_val", "n_val", "V_cal_val_L", "cov_test", "n_test", "V_cal_test_L", "family", "bin", "p_mean", "f_e", "f_ev", "nfev", "n_par", "t_fit")})))
rows.sort(key=lambda r: r["V_in_L"])
print(f"criterion {tag}")
print(f"{'family':18s} npar  V_full   V_in   covTr  covVal covTest  Vcal_val Vcal_test  rmax  z_first z_last")
for r in rows:
    print(f"{r['family']:18s} {r['n_par']:3d} {r['V_L']:8.1f} {r['V_in_L']:7.1f} {r['cov_train']:.4f} {r['cov_val']:.4f} {r['cov_test']:.4f}  {r['V_cal_val_L']:8.1f} {r['V_cal_test_L']:8.1f} {r['r_max']:6.1f} {r['z_first']:6.0f} {r['z_last']:6.0f}" + ("" if r["n_bins"] == 20 else f"   ({r['n_bins']} bins)"))
for name, rr in (("table", rows), ("perbin", per)):
    if rr:
        with open(os.path.join(LANE, "results", f"{name}_{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0].keys()))
            w.writeheader(); w.writerows(rr)
if tag == "E95_EV98":
    ENV = torch.load("/home/user/sofiya/shape-form/results/proton/envelope.pt", weights_only=False)
    DROP = torch.load("/home/user/sofiya/shape-form/results/proton/drop.pt", weights_only=False)
    print("report saved: envelope mean V", np.mean([ENV[b]["V_L"] for b in range(20)]), " drop mean V", np.mean([DROP[b]["V_L"] for b in range(20)]))
    print("keys env", list(ENV[0].keys()), "drop", list(DROP[0].keys()))
