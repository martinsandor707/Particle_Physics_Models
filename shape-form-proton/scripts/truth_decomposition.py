"""A formula tagjainak fizikai értelmezése a szimulációs „igazsággal” (független s2 minta, a v3 CSV találatonkénti oszlopai):
ki adta le az energiát (particle_type, is_secondary, parent_type) és mikor (earliest_contribution_time).

Kategóriák (energiával súlyozva, eseményenként normálva, a fit térbeli vágásaival):
  primer        – maga a beeső részecske (is_secondary = 0);
  neutronból    – egy neutron közvetlen leánya vagy maga a neutron (parent_type vagy particle_type = neutron);
  EM            – egyéb másodlagos e-, e+, gamma;
  hadron        – egyéb másodlagos töltött hadron, mag-töredék stb.
Tartományok: mag r < 50 mm, test 50–300 mm, távoli r ≥ 300 mm. Idő az esemény első találatához képest: < 5 ns, 5–20 ns, ≥ 20 ns;
a fit a globálisan ≥ 100 ns találatokat elhagyja („késői”). Összevetés a végleges (3D) formula komponenseivel ugyanezekben a
tartományokban (a detektorba eső energia, Ā-val súlyozva).
    python truth_decomposition.py <particle>       kimenet: results/particles/<p>/truth_decomposition.json"""
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
C = protonlab.load_common_for(particle, "s2", OUT_DIR=D)
Z_MIN, Z_MAX, R_MAX, TCUT = C["Z_MIN"], C["Z_MAX"], C["R_MAX"], C["TIME_CUT_NS"]
path = os.path.join(protonlab.data_dir(particle), C["FILE_PATTERN"].format(particle=particle))
t0 = time.time()
df = pd.read_csv(path, usecols=["event_number", "x_local", "y_local", "z_local", "energy", "earliest_contribution_time",
                                "incoming_momentum", "particle_type", "is_secondary", "parent_type"],
                 dtype={"particle_type": "category", "parent_type": "category"})
print(particle, "read", len(df), "rows", f"{time.time() - t0:.0f}s", flush=True)
r = np.hypot(df.x_local.values, df.y_local.values)
keep = (df.z_local.values >= Z_MIN) & (df.z_local.values <= Z_MAX) & (r < R_MAX)
df, r = df[keep], r[keep]
ev, uniq = pd.factorize(df.event_number, sort=True)
n_ev = len(uniq)
E = df.energy.values.astype(np.float64)
tg = df.earliest_contribution_time.values
tmin = np.full(n_ev, np.inf); np.minimum.at(tmin, ev, tg)
trel = tg - tmin[ev]
p_ev = np.zeros(n_ev); p_ev[ev] = df.incoming_momentum.values

pt = df.particle_type.astype(str).values
pa = df.parent_type.astype(str).values
prim = df.is_secondary.values == 0
neut = (~prim) & ((pa == "neutron") | (pt == "neutron"))
em = (~prim) & (~neut) & np.isin(pt, ["e-", "e+", "gamma"])
had = (~prim) & (~neut) & (~em)
CAT = {"primer": prim, "neutronból": neut, "EM": em, "hadron": had}
REG = {"mag (r<50)": r < 50, "test (50–300)": (r >= 50) & (r < 300), "távoli (r≥300)": r >= 300}
fit = tg < TCUT
TW = {"<5 ns": trel < 5, "5–20 ns": (trel >= 5) & (trel < 20), "≥20 ns": trel >= 20}


def per_event(mask, denom):
    """Per-event energy fraction (mask energy / denom energy), returned per event."""
    num = np.bincount(ev, weights=E * mask, minlength=n_ev)
    return num / np.maximum(denom, 1e-30)


E_all = np.bincount(ev, weights=E, minlength=n_ev)
E_fit = np.bincount(ev, weights=E * fit, minlength=n_ev)
GROUPS = {"0–2 GeV": (0, 2), "2–5 GeV": (2, 5), "5–10 GeV": (5, 10), "10–20 GeV": (10, 20.5)}
RB = np.linspace(0, 700, 29)                                            # 25 mm radial bins for dE/dr
out = {"particle": particle, "sample": "s2", "n_events": int(n_ev), "groups": {}}
for g, (lo, hi) in GROUPS.items():
    sel = (p_ev >= lo) & (p_ev < hi) & (E_fit > 0)
    if sel.sum() == 0:
        continue
    m = lambda x: float(np.mean(x[sel]))
    res = {"n_events": int(sel.sum()), "late_frac_of_all": m(1 - E_fit / np.maximum(E_all, 1e-30))}
    res["cat"] = {c: m(per_event(fit & k, E_fit)) for c, k in CAT.items()}
    res["region"] = {rn: m(per_event(fit & rk, E_fit)) for rn, rk in REG.items()}
    res["region_cat"] = {rn: {c: m(per_event(fit & rk & k, E_fit)) for c, k in CAT.items()} for rn, rk in REG.items()}
    res["region_time"] = {rn: {tw: m(per_event(fit & rk & tk, E_fit)) for tw, tk in TW.items()} for rn, rk in REG.items()}
    res["time"] = {tw: m(per_event(fit & tk, E_fit)) for tw, tk in TW.items()}
    # radial dE/dr shapes (per-event normalised, averaged) for all in-fit energy, for the delayed part and for neutron daughters
    w_ev = np.where(sel, 1.0 / np.maximum(E_fit, 1e-30), 0.0)[ev]
    for lab, k in (("all", fit), ("≥20 ns", fit & TW["≥20 ns"]), ("neutronból", fit & neut), ("EM", fit & em)):
        h = np.histogram(r[k], RB, weights=(E * w_ev)[k])[0] / sel.sum()
        res[f"dEdr_{lab}"] = h.tolist()
    out["groups"][g] = res
    print(particle, g, "cat", {c: round(v, 4) for c, v in res["cat"].items()}, "far", round(res["region"]["távoli (r≥300)"], 4),
          "late", round(res["late_frac_of_all"], 4), flush=True)

# ---------------- the final 3D formula in the same terms (in-detector energy, A_bar-weighted)
C1 = protonlab.load_common_for(particle, "s1", OUT_DIR=D)
exec(shapelab.SLAB_GEOM_SRC, C1)
model, core, t, K = shapelab.build_final(C1, particle, D, 0.0, 1.0, metric="slab")
abar = torch.load(os.path.join(D, "abar.pt"), map_location=C1["DEVICE"])
RR = C1["RR"]
RMASK = {"mag (r<50)": RR < 50, "test (50–300)": (RR >= 50) & (RR < 300), "távoli (r≥300)": RR >= 300}
names = ["zápor", "nyom", "csúcs"] + (["folt"] if core.endblob else []) + (["távoli udvar"] if core.far else [])
for g, (lo, hi) in GROUPS.items():
    if g not in out["groups"]:
        continue
    ps = np.sort(p_ev[(p_ev >= lo) & (p_ev < hi)])
    ps = ps[np.linspace(0, len(ps) - 1, min(256, len(ps))).astype(int)]
    lo_p, hi_p = float(np.exp(float(core.lnp0) - float(core.lnp_s))), float(np.exp(float(core.lnp0) + float(core.lnp_s)))
    ps = np.clip(ps, lo_p, hi_p)                                            # stay inside the training range of ln p
    with torch.no_grad():
        lw, comp, th = core.components(torch.log(torch.tensor(ps, dtype=torch.float32, device=C1["DEVICE"])))
        mass = {nm: torch.exp(w).reshape(-1, 1, 1) * torch.exp(c) * abar for nm, w, c in zip(names, lw, comp) if w is not None}
        tot = sum(v.sum((1, 2)) for v in mass.values())
        nm_ = lambda x: float(torch.nanmean(x))
        res = {"share": {nm: nm_(v.sum((1, 2)) / tot) for nm, v in mass.items()},
               "region": {rn: nm_(sum(v[:, rm].sum(1) for v in mass.values()) / tot) for rn, rm in RMASK.items()},
               "region_comp": {rn: {nm: nm_(v[:, rm].sum(1) / tot) for nm, v in mass.items()} for rn, rm in RMASK.items()},
               "p_range_used": [float(ps.min()), float(ps.max())]}
    out["groups"][g]["model"] = res
    print(particle, g, "model share", {k: round(v, 4) for k, v in res["share"].items()}, "far", round(res["region"]["távoli (r≥300)"], 4), flush=True)
json.dump(out, open(os.path.join(D, "truth_decomposition.json"), "w"), indent=1, ensure_ascii=False)
print(particle, "done", flush=True)
