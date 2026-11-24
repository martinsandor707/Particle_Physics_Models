"""Alakok egy részecskére a report 2. táblázatának definíciójával, kiegészítve.  python particle_shapes.py <particle>

Minden 1 GeV-es binre (a teszt-lefedés a 02/03/05 notebookok módján: találatonként 8 pont a cellában, seed = 3):
  r95     a modell átlagos térképének legkisebb, az energia 95 %-át tartalmazó tartománya (iso_radius, mint a reportban);
          térfogat = sum pi r95(z')^2 dz' (a report definíciója), lefedés = a tesztzáporok hányada, amelyekben az energia >= 95 %-a bent van;
          modellek: a report baseline-ja, MLP-je, MLP-formulája (exportált térképek) és az új formula (report-mérce, × Ā);
  98 %    a report mentett burkolója és cseppje (02/03), az újraillesztett csepp és a szuper-tojás (+ müon: henger, kapszula),
          és az új formula 98 %-ra kalibrált izoszint-teste (a report ajánlott következő lépése);
  simított a csepp és a szuper-tojás paraméterei (és ln A) ln p polinomjaként (fokszám BIC-kel, mint a 02/03-ban), a teszt-lefedés
          binenként validálva (a report 3. korlátja: ezt nem validálták);
  V_szelet minden alak térfogata a detektor-szeletben (a bin tanító eseményeinek átlagos szelet-elfogadásával).
Kimenet: results/particles/<részecske>/shapes.json és shapes_perbin.csv
"""
import json
import math
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lanes/E1_bodies"))
import numpy as np
import pandas as pd
import torch

import envlab
import protonlab
import shapelab
import fam
from particle_shape import MASS, ParticleShape

particle = sys.argv[1]
D = os.path.join(ROOT, "results", "particles", particle)
R = os.path.join(protonlab.REPORT_RESULTS, particle)
C, geo, data = shapelab.setup_particle(particle, out_dir=D)
dev = C["DEVICE"]
VOL = torch.exp(C["LOG_VOL"])
ZE, RE, DZ = C["Z_EDGES"].cpu().numpy(), C["R_EDGES"].cpu().numpy(), C["DZ"]
ZC = C["ZC"].cpu().numpy()
REG = fam.registry()
G_MAX = 0.5 * 0.75 ** 1.5
c0, s0 = shapelab.lnp_center_scale(data)
VARIANTS = {"full": dict(long="gamma", track=True, spike=True), "full_att": dict(long="gamma", track=True, spike=True, att=True),
            "notrack": dict(long="gamma", track=False, spike=True), "plateau": dict(long="plateau", track=False, spike=True),
            "plateau_track": dict(long="plateau", track=True, spike=True, att=True)}
NEW, core, test, ROUND = shapelab.build_final(C, particle, D, c0, s0, metric="std")       # the final formula (latest round), × A_bar
n = test["degrees"]["new_std"]
ABAR = torch.load(os.path.join(D, "abar.pt"), map_location=dev)
NEW.eval()
MAPS = {}
for key, label in ((shapelab.REPORT_BASE[particle][0], "baseline (report)"), ("mlp", "MLP (report)"), ("mlp_formula", "MLP-formula (report)")):
    MAPS[label] = C["TableModel"](torch.load(os.path.join(R, f"export_{key}.pt"), weights_only=False)).eval()
MAPS["new formula"] = NEW


def step_fn(r95):
    return lambda z: np.where((z >= ZE[0]) & (z < ZE[-1]), r95[np.clip(np.searchsorted(ZE, z) - 1, 0, len(r95) - 1)], 0.0)


def profile_coverage(P, radius_fn, f_e=0.95):
    inside = torch.as_tensor(P["r"].cpu().numpy() <= radius_fn(P["z"].cpu().numpy()), device=dev)
    s = torch.where(inside, torch.zeros_like(P["r"]), torch.full_like(P["r"], float("inf")))
    return float((envlab.shower_quantile(P, s, f_e) <= 0).float().mean())


def energy_inside(P, inside):
    """Mean over the showers of the fraction of their energy inside the shape (for r95: should be 0.95)."""
    e = torch.zeros(P["n"], device=dev, dtype=torch.float64).index_add_(0, P["ev"], P["w"].double() * inside.double())
    return float(e.mean())


def cell_fraction(radius_fn, nz=8, nr=8):
    kz, kr = (np.arange(nz) + 0.5) / nz, (np.arange(nr) + 0.5) / nr
    z = (ZE[:-1, None] + np.diff(ZE)[:, None] * kz[None, :]).ravel()
    rz = radius_fn(z).reshape(len(ZE) - 1, nz)
    r = np.sqrt(RE[:-1, None] ** 2 + (RE[1:, None] ** 2 - RE[:-1, None] ** 2) * kr[None, :])
    return torch.tensor((rz[:, None, :, None] >= r[None, :, None, :]).mean((2, 3)), dtype=torch.float32, device=dev)


def envelope_r(z, e):
    u = np.clip(np.asarray(z, float) - e["z0"], 0, None)
    return e["A"] * u ** e["a"] * np.exp(-e["b"] * u)


def drop_r(z, e):
    t = (np.asarray(z, float) - e["z0"]) / e["L"]
    return np.where((t > 0) & (t < 1), e["A"] * np.sqrt(np.clip(t, 0, 1)) * np.clip(1 - t, 0, 1) ** 1.5 / G_MAX, 0.0)


REPORT_BODIES = {}
for name, fn in (("envelope", envelope_r), ("drop", drop_r)):
    f = os.path.join(R, f"{name}.pt")
    if os.path.exists(f):
        REPORT_BODIES[name] = (torch.load(f, weights_only=False), fn)
BODY_DIR = os.path.join(D, "bodies_E95_EV98")
FAMS = [f for f in ("drop", "superegg", "cylinder", "capsule") if os.path.exists(os.path.join(BODY_DIR, f"{f}_bin00.json"))]
rows, per_bin_fit = [], {f: [] for f in FAMS}
for b in range(20):
    trb = data["train"].subset((data["train"].ev["bin"] == b).nonzero().squeeze(1))
    teb = data["test"].subset((data["test"].ev["bin"] == b).nonzero().squeeze(1))
    if len(teb) == 0 or len(trb) == 0:
        continue
    Ptr, Pte = envlab.deposits(C, trb), envlab.deposits(C, teb)
    p_te, p_tr = float(teb.ev["p"].mean()), float(trb.ev["p"].mean())
    with torch.no_grad():
        abar = sum(C["event_slab"](bb, trb.box, 4, True, True).sum(0) for bb in trb.batches(256)) / len(trb)
    base = {"bin": b, "p_test": p_te, "p_train": p_tr}
    # r95 shapes (report definition)
    for label, m in MAPS.items():
        with torch.no_grad():
            P = m(torch.log(torch.tensor([p_te], device=dev, dtype=torch.float32))).exp()[0].cpu().numpy()
        r95 = C["iso_radius"](P, 0.95)
        fn = step_fn(r95)
        mask = torch.tensor(RE[1:][None, :] <= r95[:, None] + 1e-9, dtype=torch.float32, device=dev)
        ins = torch.as_tensor(Pte["r"].cpu().numpy() <= fn(Pte["z"].cpu().numpy()), device=dev)
        rows.append({**base, "shape": f"r95 {label}", "V": float(np.sum(np.pi * r95 ** 2) * DZ) / 1e6, "V_in": float(np.sum(np.pi * r95 ** 2) * DZ) / 1e6,
                     "V_slab": float((VOL * mask * abar).sum()) / 1e6, "cov": profile_coverage(Pte, fn), "E_in": energy_inside(Pte, ins)})
    # the report's saved 98 % bodies
    for name, (fits, fn) in REPORT_BODIES.items():
        if b in fits:
            e = fits[b]
            rf = lambda z, e=e, fn=fn: fn(z, e)
            frac = cell_fraction(rf)
            rows.append({**base, "shape": f"98% {name} (report)", "V": e["V_L"], "V_in": envlab.grid_volume_L(C, rf, clip_r=700.0),
                         "V_slab": float((VOL * frac * abar).sum()) / 1e6, "cov": profile_coverage(Pte, rf)})
    # refitted bodies of this study
    for f in FAMS:
        d = json.load(open(os.path.join(BODY_DIR, f"{f}_bin{b:02d}.json")))
        per_bin_fit[f].append((b, p_tr, d["x"], math.log(d["A"])))
        rows.append({**base, "shape": f"98% {f} (refit)", "V": d["V_L"], "V_in": d["V_in_L"], "V_slab": d["V_slab_L"], "cov": d["cov_test"]})
    # calibrated iso-density body of the new formula (report's recommended next step)
    with torch.no_grad():
        logrho = NEW(torch.log(torch.tensor([p_tr], device=dev)))[0] - C["LOG_VOL"]
    iz, ir = C["locate"](Ptr["z"], Ptr["r"])
    lvl = envlab.threshold(Ptr, -logrho[iz, ir], 0.95, 0.98)
    ins = (-logrho <= lvl + 1e-9).float()
    iz2, ir2 = C["locate"](Pte["z"], Pte["r"])
    rows.append({**base, "shape": "98% iso-density (new formula)", "V": float((VOL * ins).sum()) / 1e6, "V_in": float((VOL * ins).sum()) / 1e6,
                 "V_slab": float((VOL * ins * abar).sum()) / 1e6, "cov": envlab.coverage(Pte, -logrho[iz2, ir2], lvl, 0.95)})
    base_store = {"Pte": Pte, "abar": abar}
    per_bin_fit.setdefault("_store", {})[b] = base_store
    print(f"{particle} bin {b:2d}: " + " | ".join(f"{r['shape'][:24]} {r['V']:.1f}/{r['cov']:.2f}" for r in rows if r["bin"] == b), flush=True)

# smoothed (ln p polynomial) bodies, validated per bin on the test showers
for f in FAMS:
    pts = per_bin_fit[f]
    L = np.log([p for _, p, _, _ in pts])
    Y = np.array([list(x) + [la] for _, _, x, la in pts])
    coefs = []
    for j in range(Y.shape[1]):
        best = None
        for deg in range(3):
            cf = np.polynomial.polynomial.polyfit(L, Y[:, j], deg)
            rss = np.sum((np.polynomial.polynomial.polyval(L, cf) - Y[:, j]) ** 2) + 1e-12
            bic = len(L) * np.log(rss / len(L)) + (deg + 1) * np.log(len(L))
            if best is None or bic < best[0]:
                best = (bic, cf)
        coefs.append(best[1])
    body = REG[f][0]
    for (b, p_tr, _, _), l in zip(pts, L):
        y = [float(np.polynomial.polynomial.polyval(l, cf)) for cf in coefs]
        x, A = y[:-1], math.exp(y[-1])
        if not body.valid(x):
            continue
        rf = lambda z, x=x, A=A: body.radius(z, x, A)
        st = per_bin_fit["_store"][b]
        frac = cell_fraction(rf)
        rows.append({"bin": b, "p_test": float("nan"), "p_train": p_tr, "shape": f"98% {f} (smoothed in ln p)",
                     "V": math.exp(math.log(math.pi) + 2 * math.log(A) + body.log_i2(x)) / 1e6,
                     "V_in": envlab.grid_volume_L(C, rf, clip_r=700.0), "V_slab": float((VOL * frac * st["abar"]).sum()) / 1e6,
                     "cov": profile_coverage(st["Pte"], rf)})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(D, "shapes_perbin.csv"), index=False)
if "E_in" not in df:
    df["E_in"] = np.nan
summ = df.groupby("shape").agg(V=("V", "mean"), V_in=("V_in", "mean"), V_slab=("V_slab", "mean"), cov=("cov", "mean"),
                               cov_min=("cov", "min"), E_in=("E_in", "mean"), n_bins=("bin", "count")).reset_index()
json.dump({"particle": particle, "variant": test["variant"], "degree": n, "shapes": summ.to_dict(orient="records")},
          open(os.path.join(D, "shapes.json"), "w"), indent=1)
pd.set_option("display.width", 200)
print(summ.round(3).to_string(index=False))
