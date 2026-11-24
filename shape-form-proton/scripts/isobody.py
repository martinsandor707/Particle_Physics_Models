"""A 98 %-os kritériumra kalibrált izosűrűség-testek (a report következő javasolt lépése) + a detektor-szelettel vágott térfogat.

Minden 1 GeV-es binre és minden sűrűségmodellre (ProtonShape, report MLP, report GFlash n=6):
  pontszám  s(z', r) = -ln rho(z', r | p_bin)   (rho a modell VALÓDI, dekonvolvált sűrűsége a rácson; p_bin = a bin átlagos impulzusa)
  test      {s <= c}, ahol c-t a TANÍTÓ záporokon kalibráljuk egzaktul (envlab.threshold: záporonként F_E-kvantilis, majd F_EV-percentilis)
  térfogat  a rácscellák térfogatának összege a testben (a rács az adattartomány: -200 <= z' <= 1450, r < 700)
  lefedés   a teszt záporokon (envlab.coverage), ugyanazokkal a cellán belül szétosztott pontokkal (K = 8, seed = 3), mint a 02/03 notebook.
A szelettel vágott térfogat: sum(cella térfogat x Ā_bin(cella)), Ā_bin a bin tanító eseményeinek átlagos szelet-akceptanciája (elöl + hátul).
Összevetés: az E1 szál binenkénti testjei (drop = a report cseppje újraillesztve, ellipszoid, superegg, knots12) ugyanazon a rácson raszterizálva.

Használat: python isobody.py <F_E> <F_EV>
"""
import importlib.util
import json
import math
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch

import envlab
import shapelab
from proton_shape import ProtonShape

F_E, F_EV = float(sys.argv[1]), float(sys.argv[2])
TAG = f"E{int(round(F_E * 100)):02d}_EV{int(round(F_EV * 100)):02d}"
OUT = os.path.join(ROOT, "results", "isobody")
os.makedirs(OUT, exist_ok=True)

C, geo, data = shapelab.setup(batch=256, out_dir=os.path.join(ROOT, "results_smoke"), slab=False)
dev = C["DEVICE"]
c0, s0 = shapelab.lnp_center_scale(data)
_ch = json.load(open(os.path.join(ROOT, "results/final/proton/slab_gceg_chosen.json")))["extra"]["degree"]   # the final 3D model
_deg = dict(degree=1, degrees={"logit_tr": 2}) if _ch == 1 else dict(degree=_ch)
ps = ProtonShape(C, lnp0=c0, lnp_s=s0, lateral="gce", lat_norm="grid", **_deg).to(dev)
ps.load_state_dict(torch.load(os.path.join(ROOT, f"results/final/proton/slab_gceg_n{_ch}.pt"), weights_only=False)["state"])
bm = shapelab.baseline_models(C, data)
# the 3D-trained model is restricted to the observable region (cells covered by at least one training slab): the 3D likelihood does not
# constrain the mass elsewhere (in front of the entrance face, outside the calorimeter)
_obs = (shapelab.mean_acceptance(C, data["train"], nsub=4, front=True, back=True) > 0).float()


class _Obs(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, lnp):
        lp = self.base(lnp) + torch.log(_obs.clamp_min(1e-30))
        return torch.log_softmax(lp.reshape(len(lnp), -1), 1).reshape(lp.shape)


MODELS = {"ProtonShape": _Obs(ps.eval()), "MLP (report)": bm["mlp"], "GFlash n=6 (report)": bm["gflash_n6"]}

# az E1 szál testcsaládjai (binenkénti illesztések: results/fits/<krit>/<család>/binXX.json)
spec = importlib.util.spec_from_file_location("e1fam", os.path.join(ROOT, "lanes/E1_bodies/fam.py"))
e1fam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e1fam)
REG = e1fam.registry()
FITS = os.path.join(ROOT, "lanes/E1_bodies/results/fits", TAG)
BODIES = ["drop", "envelope", "ellipsoid", "egg", "gendrop", "superegg", "capsule", "knots12"]

LOG_VOL = C["LOG_VOL"]
VOL = torch.exp(LOG_VOL)                                   # cella-térfogatok [mm^3]  (N_Z, N_R)
ZE, RE = C["Z_EDGES"].cpu().numpy(), C["R_EDGES"].cpu().numpy()


def cell_fraction(radius_fn, nz=8, nr=8):
    """Az r <= r(z') test által lefedett hányad minden rácscellában (cellánként nz x nr részpont, r-ben egyenlő területű)."""
    kz = (np.arange(nz) + 0.5) / nz
    kr = (np.arange(nr) + 0.5) / nr
    z = (ZE[:-1, None] + np.diff(ZE)[:, None] * kz[None, :]).ravel()            # (N_Z * nz)
    rz = radius_fn(z).reshape(len(ZE) - 1, nz)
    r = np.sqrt(RE[:-1, None] ** 2 + (RE[1:, None] ** 2 - RE[:-1, None] ** 2) * kr[None, :])   # (N_R, nr)
    inside = rz[:, None, :, None] >= r[None, :, None, :]                       # (N_Z, N_R, nz, nr)
    return torch.tensor(inside.mean((2, 3)), dtype=torch.float32, device=dev)


rows = []
for b in range(20):
    tr, te = envlab.bin_sets(C, data, b)
    Ptr, Pte = envlab.deposits(C, tr), envlab.deposits(C, te)
    p_mean = float(tr.ev["p"].mean())
    with torch.no_grad():
        Abar = sum(C["event_slab"](bb, tr.box, 4, True, True).sum(0) for bb in tr.batches(256)) / len(tr)
    iz_tr, ir_tr = C["locate"](Ptr["z"], Ptr["r"])
    iz_te, ir_te = C["locate"](Pte["z"], Pte["r"])
    for name, m in MODELS.items():
        with torch.no_grad():
            logrho = m(torch.log(torch.tensor([p_mean], device=dev)))[0] - LOG_VOL
        s_tr, s_te = -logrho[iz_tr, ir_tr], -logrho[iz_te, ir_te]
        c = envlab.threshold(Ptr, s_tr, F_E, F_EV)
        inside = (-logrho <= c + 1e-9).float()
        rup = np.array([RE[1:][np.nonzero(row)[0].max()] if row.any() else 0.0 for row in inside.cpu().numpy() > 0])
        rows.append({"bin": b, "p_mean": p_mean, "body": f"iso-density {name}", "n_par_bin": 1,
                     "V_L": float((VOL * inside).sum()) / 1e6, "V_hull_L": float((np.pi * rup ** 2 * np.diff(ZE)).sum()) / 1e6,
                     "V_slab_L": float((VOL * inside * Abar).sum()) / 1e6,
                     "cov_train": envlab.coverage(Ptr, s_tr, c, F_E), "cov_test": envlab.coverage(Pte, s_te, c, F_E),
                     "r_max": float(rup.max()), "level": float(c)})
    for fam in BODIES:
        f = os.path.join(FITS, fam, f"bin{b:02d}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        body = REG[fam][0]
        frac = cell_fraction(lambda z: body.radius(z, d["x"], d["A"]))
        rows.append({"bin": b, "p_mean": p_mean, "body": fam, "n_par_bin": d.get("n_par"), "V_L": d["V_L"], "V_in_L": d["V_in_L"],
                     "V_grid_L": float((VOL * frac).sum()) / 1e6, "V_slab_L": float((VOL * frac * Abar).sum()) / 1e6,
                     "cov_train": d["cov_train"], "cov_test": d["cov_test"], "r_max": d["r_max"], "x": json.dumps(d["x"]), "A": d["A"]})
    print(f"bin {b:2d} done: " + " | ".join(f"{r['body'][:22]} {r.get('V_grid_L', r['V_L']):.0f}L/{r['V_slab_L']:.0f}L cov {r['cov_test']:.3f}"
                                         for r in rows if r["bin"] == b), flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f"isobody_{TAG}.csv"), index=False)
agg = df.groupby("body").agg(n_bins=("bin", "count"), V_L=("V_L", "mean"), V_in_L=("V_in_L", "mean"), V_grid_L=("V_grid_L", "mean"),
                             V_hull_L=("V_hull_L", "mean"), V_slab_L=("V_slab_L", "mean"), cov_test=("cov_test", "mean"),
                             r_max=("r_max", "mean")).sort_values("V_slab_L")
pd.set_option("display.width", 220)
print(agg.round(3).to_string())
agg.to_csv(os.path.join(OUT, f"isobody_{TAG}_summary.csv"))
