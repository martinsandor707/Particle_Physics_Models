"""Időben szétválasztott záporalak (3D-mérce): a végleges formulát külön illesztjük a gyors (az esemény első találata után < 5 ns) és
a késleltetett (5 ns … a 100 ns-os ablak vége) találatokra, ugyanazzal a felosztással (SEED = 1), a végleges formulából indítva.
Mindkét részhalmazban a találatok energiája eseményenként 1-re normált, tehát a két alak külön-külön az adott időablak átlagos alakja;
az eseményenkénti késleltetett hányadot (f_d) az adatból mérjük. Kimenet: results/particles/<p>/time_split.json (+ checkpointok)
    python time_split.py <particle>"""
import copy
import json
import os
import sys
import time

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import torch

import protonlab
import shapelab

particle = sys.argv[1]
D = os.path.join(ROOT, "results", "particles", particle)
C = protonlab.load_common_for(particle, "s1", OUT_DIR=D, BATCH=256)
exec(shapelab.SLAB_GEOM_SRC, C)
exec(shapelab.SLAB_NLL_SRC, C)
dev, Showers = C["DEVICE"], C["Showers"]
T_SPLIT = 5.0
t0 = time.time()
df = C["read_csv"](particle)
geo = C["detector_geometry"](df)
r = np.hypot(df.x_local.values, df.y_local.values)
dff = df[(df.earliest_contribution_time < C["TIME_CUT_NS"]) & (df.z_local >= C["Z_MIN"]) & (df.z_local <= C["Z_MAX"]) & (r < C["R_MAX"])].copy()
del df
print(particle, "read", len(dff), f"{time.time() - t0:.0f}s", flush=True)

# the same train/val/test split as load_data (events sorted by number, stratified by momentum bin, SEED)
uniq = np.sort(dff.event_number.unique())
first = dff.sort_values("event_number", kind="stable").groupby("event_number").first().loc[uniq]
bins = first.incoming_momentum_bin.values.astype(np.int64)
rng = np.random.default_rng(C["SEED"])
parts = {"train": [], "val": [], "test": []}
for b in np.unique(bins):
    idx = rng.permutation(np.where(bins == b)[0])
    n1, n2 = round(0.70 * len(idx)), round(0.85 * len(idx))
    parts["train"] += idx[:n1].tolist(); parts["val"] += idx[n1:n2].tolist(); parts["test"] += idx[n2:].tolist()
split_ev = {k: uniq[np.sort(v)] for k, v in parts.items()}

dff["trel"] = dff.earliest_contribution_time - dff.groupby("event_number").earliest_contribution_time.transform("min")
E_all = dff.groupby("event_number").energy.sum()
E_del = dff[dff.trel >= T_SPLIT].groupby("event_number").energy.sum().reindex(E_all.index, fill_value=0.0)
fd = (E_del / E_all).loc[split_ev["test"]]
p_test = first.loc[split_ev["test"]].incoming_momentum.values
res = {"particle": particle, "t_split_ns": T_SPLIT, "delayed_fraction_test": {}}
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20.5)):
    s = (p_test >= lo) & (p_test < hi)
    res["delayed_fraction_test"][f"{lo}-{hi} GeV"] = {"mean": float(fd.values[s].mean()), "median": float(np.median(fd.values[s])),
                                                     "p90": float(np.quantile(fd.values[s], 0.9))}
print(particle, "delayed fraction", res["delayed_fraction_test"], flush=True)


def make_sets(sub):
    sh = Showers.from_dataframe(sub, geo)
    u = np.sort(sub.event_number.unique())
    out = {}
    for k, ev in split_ev.items():
        ev = ev[np.isin(ev, u)]
        out[k] = sh.subset(np.searchsorted(u, ev)).to(dev)
    return out


SETS = {"prompt": make_sets(dff[dff.trel < T_SPLIT]), "delayed": make_sets(dff[dff.trel >= T_SPLIT]), "all": make_sets(dff)}
for k, v in SETS.items():
    print(particle, k, {s: len(x) for s, x in v.items()}, flush=True)
abar = torch.load(os.path.join(D, "abar.pt"), map_location=dev)


def describe(model, P=(1.0, 2.0, 5.0, 10.0, 20.0)):
    """Energy-weighted mean r, r90, mean z', z90 of the model's in-detector (A_bar-weighted) mean map."""
    with torch.no_grad():
        Pm = model(torch.log(torch.tensor(P, device=dev))).exp() * abar
        Pm = Pm / Pm.sum((1, 2), keepdim=True)
        rc, zc = C["RC"], C["ZC"]
        pr, pz = Pm.sum(1), Pm.sum(2)
        cr, cz = torch.cumsum(pr, 1), torch.cumsum(pz, 1)
        r90 = rc[(cr < 0.9).sum(1).clamp(max=len(rc) - 1)]
        z90 = zc[(cz < 0.9).sum(1).clamp(max=len(zc) - 1)]
        return {"p": list(P), "mean_r_mm": (pr * rc).sum(1).tolist(), "r90_mm": r90.tolist(), "mean_z_mm": (pz * zc).sum(1).tolist(), "z90_mm": z90.tolist()}


def data_describe(sh):
    h = sh.hits()
    out = {}
    for lo, hi in ((1, 2), (2, 5), (5, 10), (10, 20.5)):
        s = (h.p >= lo) & (h.p < hi)
        w = h.w[s] / h.w[s].sum()
        rs, zs = h.r[s].values, h.z[s].values
        o = np.argsort(rs); cw = np.cumsum(w.values[o])
        out[f"{lo}-{hi} GeV"] = {"mean_r_mm": float((w * h.r[s]).sum()), "r90_mm": float(rs[o][np.searchsorted(cw, 0.9)]),
                                 "mean_z_mm": float((w * h.z[s]).sum())}
    return out


final = shapelab.build_final(C, particle, D, 0.0, 1.0, metric="slab")[0]
res["final_describe"] = describe(final)
for name in ("prompt", "delayed"):
    data = SETS[name]
    m = copy.deepcopy(final).train()
    torch.manual_seed(0)
    shapelab.fit_fast(C, data, m, steps=4000, lr=3e-3, lr_end=3e-4, log=None)
    m.eval()
    torch.save({"state": m.state_dict()}, os.path.join(D, f"time_split_{name}.pt"))
    t_fit = C["test_nll"](m, data["test"])
    t_fin = C["test_nll"](final, data["test"])
    res[name] = {"n_test_events": len(data["test"]), "test_nll_fit": float(t_fit.mean()), "test_nll_final": float(t_fin.mean()),
                 "fit_minus_final": shapelab.paired(t_fit, t_fin), "describe": describe(m), "data_describe": data_describe(data["test"])}
    with torch.no_grad():
        lw, comp, th = m.components(torch.log(torch.tensor([2.0, 5.0, 10.0, 20.0], device=dev)))
        g = lambda k_: th[k_].flatten().cpu().numpy() if k_ in th else None
        zk = 0.1
        res[name]["halo_lambda_z100_mm"] = np.exp(g("log_lam") + g("s_lam") * zk).tolist()
        res[name]["cauchy_R_z100_mm"] = np.exp(g("log_R") + g("s_R") * zk).tolist()
    print(particle, name, json.dumps({k: v for k, v in res[name].items() if k in ("n_test_events", "test_nll_fit", "test_nll_final", "fit_minus_final")}),
          "mean r", [round(x, 1) for x in res[name]["describe"]["mean_r_mm"]], "r90", [round(x) for x in res[name]["describe"]["r90_mm"]], flush=True)
res["all_data_describe"] = data_describe(SETS["all"]["test"])
json.dump(res, open(os.path.join(D, "time_split.json"), "w"), indent=1)
print(particle, "done", f"{time.time() - t0:.0f}s", flush=True)
