"""Kontroll a szimbolikus korrekcióhoz: ugyanaz a finomhangolás (3000 lépés, lr 3e-3 -> 3e-4) korrekciós tag NÉLKÜL.
Ha ez ugyanannyit javít, a korrekció nyeresége csak a hosszabb tanításból jön.  Kimenet: results/particles/<p>/sr_control.json"""
import json, os, sys
ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import torch
import shapelab
from particle_shape import MASS, ParticleShape
particle = sys.argv[1]
D = os.path.join(ROOT, "results", "particles", particle)
C, geo, data = shapelab.setup_particle(particle, out_dir=D)
dev = C["DEVICE"]; c0, s0 = shapelab.lnp_center_scale(data)
t2 = json.load(open(os.path.join(D, "test_v2.json")))
n = t2["degrees"]["new_std"]
kw = dict(t2["variant_kwargs"]); spk = kw.pop("spike_degrees", None)
degrees = {"logit_tr": 2} if (n == 1 and kw.get("track")) else {}
if spk:
    degrees.update({k: min(n, v) for k, v in spk.items()})
core = ParticleShape(C, MASS[particle], lnp0=c0, lnp_s=s0, degree=n, degrees=degrees or None, **kw).to(dev)
m = shapelab.WithAcceptance(core, torch.load(os.path.join(D, "abar.pt"), map_location=dev)).to(dev)
m.load_state_dict(torch.load(os.path.join(D, f"new2_std_n{n}.pt"), map_location=dev, weights_only=False)["state"])
v0 = C["test_nll"](m.eval(), data["val"])
shapelab.fit_fast(C, data, m, steps=3000, lr=3e-3, lr_end=3e-4, log=None)
v1 = C["test_nll"](m.eval(), data["val"])
sr = json.load(open(os.path.join(D, "sr_residual.json")))
best = min(sr["candidates"], key=lambda c: c["val"])
r = {"base_val": float(v0.mean()), "control_val": float(v1.mean()), "control_minus_base": shapelab.paired(v1, v0),
     "best_sr_val": best["val"], "best_sr_equation": best["equation"], "best_sr_minus_control": best["val"] - float(v1.mean())}
json.dump(r, open(os.path.join(D, "sr_control.json"), "w"), indent=1)
print(particle, json.dumps(r), flush=True)
