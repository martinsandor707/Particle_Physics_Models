"""Elektron: a GFlash EM laterális profilja (két Cauchy-tag, Grindhammer–Peters) a mi szerkezetünkben, a választott változattal
(plató-zápor + csúcs) és gamma-zápor + csúcs mellett, report-mérce × Ā, n = 2, validáció."""
import json, os, sys, time
ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import torch
import shapelab
from particle_shape import MASS, ParticleShape
OUT = os.path.join(ROOT, "results", "particles", "electron")
C, geo, data = shapelab.setup_particle("electron", out_dir=OUT)
c0, s0 = shapelab.lnp_center_scale(data)
ABAR = torch.load(os.path.join(OUT, "abar.pt"), map_location=C["DEVICE"])
res = {}
for name, kw in {"plateau_cc": dict(long="plateau", track=False, lateral="cc"), "gamma_cc": dict(long="gamma", track=False, lateral="cc")}.items():
    core = ParticleShape(C, MASS["electron"], degree=2, lnp0=c0, lnp_s=s0, **kw).to(C["DEVICE"]).init_from_data(C, data)
    m = shapelab.WithAcceptance(core, ABAR).to(C["DEVICE"])
    torch.manual_seed(0); t = time.time()
    shapelab.fit_fast(C, data, m, steps=4000, log=None)
    res[name] = {"val": float(C["test_nll"](m.eval(), data["val"]).mean()), "params": int(m.n_params()), "sec": round(time.time() - t)}
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(OUT, "variants_em_lateral.json"), "w"), indent=1)
