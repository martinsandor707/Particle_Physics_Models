"""Független ellenőrzés egy jelölt sűrűségmodellre.

Mit ellenőriz:
 1. normálás: exp(log P) összege 1 minden impulzusra, nincs NaN/inf;
 2. szabad paraméterek száma: hány együttható változtat ténylegesen a kimeneten (vs. a modell n_params() értéke);
 3. teszt-NLL a 00_common VÁLTOZATLAN `test_nll` függvényével (standard likelihood) – a shapelab kiértékelője nélkül;
 4. teszt-NLL a szelet-likelihooddal, és a szelet-normálás független Monte-Carlo ellenőrzése (a zárt alakú akceptancia nélkül);
 5. ugyanez a FÜGGETLEN s2 mintán (másik szimulációs mag, ~100k esemény, a modell soha nem látta), a report modelljeivel párosítva.

Használat:
  python audit_candidate.py --module lanes/X/fam.py --ctor "fam.MyClass(C, degree=2, lnp0=c, lnp_s=s)" --ckpt path.pt --out audit.json
A --ctor kifejezésben elérhető: fam (a modul), C, c, s (a tanítóhalmaz ln p centrálása), data, shapelab, families, torch.
"""
import argparse
import importlib.util
import json
import math
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch

import families
import protonlab
import shapelab

ap = argparse.ArgumentParser()
ap.add_argument("--module", required=True)
ap.add_argument("--ctor", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mc_events", type=int, default=1500)
ap.add_argument("--skip_s2", type=int, default=0)
a = ap.parse_args()

res = {"module": a.module, "ctor": a.ctor, "ckpt": a.ckpt}


def build(C, data):
    spec = importlib.util.spec_from_file_location("fam", a.module)
    fam = importlib.util.module_from_spec(spec)
    sys.modules["fam"] = fam
    spec.loader.exec_module(fam)
    c, s = shapelab.lnp_center_scale(DATA_S1)
    m = eval(a.ctor, {"fam": fam, "C": C, "c": c, "s": s, "data": data, "shapelab": shapelab, "families": families, "torch": torch})
    ck = torch.load(a.ckpt, map_location=C["DEVICE"], weights_only=False)
    m = m.to(C["DEVICE"])
    m.load_state_dict(ck["state"] if "state" in ck else ck["model"])
    return m.eval(), ck


# ---------- standard likelihood: csak a 00_common kódja ----------
C = protonlab.load_common(OUT_DIR=os.path.join(ROOT, "results_smoke"))
exec(shapelab.SLAB_GEOM_SRC, C)                      # csak geometria (a likelihoodot nem érinti)
geo, DATA_S1 = protonlab.load_proton(C)
model, ck = build(C, DATA_S1)
dev = C["DEVICE"]

with torch.no_grad():
    lnp = torch.linspace(math.log(0.25), math.log(20.0), 64, device=dev)
    lp = model(lnp)
    res["normalisation: max |sum P - 1|"] = float((lp.exp().sum((1, 2)) - 1).abs().max())
    res["finite"] = bool(torch.isfinite(lp).all())
    res["shape"] = list(lp.shape)
    # szabad paraméterek: melyik együttható változtat a kimeneten?
    eff = 0
    total = 0
    for p in model.parameters():
        flat = p.data.view(-1)
        for i in range(flat.numel()):
            old = float(flat[i])
            flat[i] = old + 0.05
            d = float((model(lnp) - lp).abs().max())
            flat[i] = old
            eff += d > 1e-7
            total += 1
    res["n_params: effective (output-changing)"] = int(eff)
    res["n_params: raw tensor entries"] = int(total)
    res["n_params: claimed by model"] = int(model.n_params()) if hasattr(model, "n_params") else None

test = C["test_nll"](model, DATA_S1["test"])
val = C["test_nll"](model, DATA_S1["val"])
res["std: test NLL (common test_nll)"] = float(test.mean())
res["std: val NLL"] = float(val.mean())
if "test_nll" in ck and not C.get("SLAB"):
    res["std: checkpoint-stored test NLL (may be slab)"] = float(np.mean(ck["test_nll"]))
base_std = shapelab.baseline_nll({**C, "SLAB": False}, DATA_S1)
for k in ("gflash_n6", "mlp"):
    res[f"std: test Δ vs {k} (paired, SE)"] = shapelab.paired(test, base_std[k]["test"])

# ---------- szelet-likelihood ----------
C2, _, D2 = shapelab.setup(batch=256, out_dir=os.path.join(ROOT, "results_smoke"), slab=True)
model2, _ = build(C2, D2)
test2 = C2["test_nll"](model2, D2["test"])
val2 = C2["test_nll"](model2, D2["val"])
res["slab: test NLL"] = float(test2.mean())
res["slab: val NLL"] = float(val2.mean())
base_slab = shapelab.baseline_nll(C2, D2)
for k in ("gflash_n6", "mlp"):
    res[f"slab: test Δ vs {k} renormalised, no refit (paired, SE)"] = shapelab.paired(test2, base_slab[k]["test"])

# a szelet-normálás független Monte-Carlo ellenőrzése: pontok húzása a modellből, szelet-tagság számlálása
with torch.no_grad():
    te = D2["test"]
    g = torch.Generator().manual_seed(11)
    idx = torch.randperm(len(te), generator=g)[:a.mc_events].to(dev)
    b = te.batch(idx)
    gen = torch.Generator(device=dev).manual_seed(12)
    M = 20000
    P = model2(b.lnp).exp()
    cell = torch.multinomial(P.reshape(len(idx), -1), M, True, generator=gen)
    iz, ir = cell // C2["N_R"], cell % C2["N_R"]
    u = torch.rand(len(idx), M, 3, device=dev, generator=gen)
    z = C2["Z_EDGES"][iz] + u[..., 0] * C2["DZ"]
    r = torch.sqrt(C2["R_EDGES"][ir] ** 2 + u[..., 1] * (C2["R_EDGES"][ir + 1] ** 2 - C2["R_EDGES"][ir] ** 2))
    x = r * torch.cos(2 * math.pi * u[..., 2])
    cos_t = b.R[:, 2, 2][:, None]
    sin_t = torch.sqrt(1 - cos_t ** 2)
    zt = z * cos_t - x * sin_t
    half = float(te.box[2]) / 2
    inside = ((zt >= -half) & (zt <= (b.zb[:, None] * cos_t))).float().mean(1)
    m_closed = (P * C2["event_slab"](b, te.box)).sum((1, 2))
    d = torch.log(m_closed) - torch.log(inside)
    res["slab norm: mean (ln M_closed - ln M_MC)"] = float(d.mean())
    res["slab norm: SE of that mean"] = float(d.std() / math.sqrt(len(d)))
    res["slab norm: mean M"] = float(m_closed.mean())

# ---------- független s2 minta ----------
s2 = os.path.join(ROOT, "cache", "proton_s2_sets.pt")
if os.path.exists(s2) and not a.skip_s2:
    ck2 = torch.load(s2, weights_only=False)
    for tag, CC, mm in (("std", C, model), ("slab", C2, model2)):
        Sh = CC["Showers"]
        bm = shapelab.baseline_models(CC, DATA_S1)
        tot = {"cand": [], "gflash_n6": [], "mlp": []}
        for k, v in ck2["sets"].items():
            sh = Sh(v["ev"], v["off"], v["xyz"], v["w"], v["box"]).to(dev)
            tot["cand"].append(CC["test_nll"](mm, sh))
            for kk in ("gflash_n6", "mlp"):
                tot[kk].append(CC["test_nll"](bm[kk], sh))
            del sh
        tot = {k: np.concatenate(v) for k, v in tot.items()}
        res[f"s2 {tag}: n events"] = int(len(tot["cand"]))
        res[f"s2 {tag}: NLL candidate"] = float(tot["cand"].mean())
        for kk in ("gflash_n6", "mlp"):
            res[f"s2 {tag}: NLL {kk}"] = float(tot[kk].mean())
            res[f"s2 {tag}: Δ candidate - {kk} (paired, SE)"] = shapelab.paired(tot["cand"], tot[kk])
else:
    res["s2"] = "not available / skipped"

with open(a.out, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
