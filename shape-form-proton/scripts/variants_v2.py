"""2. kör: fizikai kiegészítések a választott változathoz (report-mérce × Ā, n = 2, 4000 lépés, validáció).
  spk  – a belépési csúcs alakja impulzusfüggő (sp_z, σ_z, σ_r polinom ln T-ben)
  far  – diffúz távoli udvar (messzire jutó neutronok/fotonok)
  end  – folt a hatótáv végén (Bragg-csúcs, befogási termékek) – csak nyommal
  all  – mind
Választás: a legkevesebb paraméteres a legjobbtól 1 SE-n belül (a régi változatokkal együtt). Kimenet: results/particles/<p>/variants_v2.json"""
import json, os, sys, time
ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import torch
import shapelab
from particle_shape import MASS, ParticleShape
particle = sys.argv[1]
D = os.path.join(ROOT, "results", "particles", particle)
C, geo, data = shapelab.setup_particle(particle, out_dir=D)
c0, s0 = shapelab.lnp_center_scale(data)
ABAR = torch.load(os.path.join(D, "abar.pt"), map_location=C["DEVICE"])
VARIANTS = {"full": dict(long="gamma", track=True, spike=True), "full_att": dict(long="gamma", track=True, spike=True, att=True),
            "notrack": dict(long="gamma", track=False, spike=True), "plateau": dict(long="plateau", track=False, spike=True),
            "plateau_track": dict(long="plateau", track=True, spike=True, att=True)}
base_name = json.load(open(os.path.join(D, "test.json")))["variant"]
base = VARIANTS[base_name]
SPK = {"sp_z": 2, "log_sp_sz": 2, "log_sp_sr": 2}
cands = {f"{base_name}+spk": dict(base, degrees=SPK), f"{base_name}+far": dict(base, far=True)}
if base.get("track"):
    cands[f"{base_name}+end"] = dict(base, endblob=True)
cands[f"{base_name}+all"] = dict(base, far=True, endblob=bool(base.get("track")), degrees=SPK)
path = os.path.join(D, "variants_v2.json")
res = json.load(open(path)) if os.path.exists(path) else {"val": {}, "params": {}}
vals = {}
# the base variant (pipeline checkpoint at n = 2) for the paired comparison
core = ParticleShape(C, MASS[particle], degree=2, lnp0=c0, lnp_s=s0, **base).to(C["DEVICE"])
m = shapelab.WithAcceptance(core, ABAR).to(C["DEVICE"])
m.load_state_dict(torch.load(os.path.join(D, f"variant_{base_name}.pt"), map_location=C["DEVICE"], weights_only=False)["state"])
vals[base_name] = C["test_nll"](m.eval(), data["val"]); res["val"][base_name], res["params"][base_name] = float(vals[base_name].mean()), int(m.n_params())
for name, kw in cands.items():
    kw = dict(kw)
    degrees = kw.pop("degrees", None)
    core = ParticleShape(C, MASS[particle], degree=2, degrees=degrees, lnp0=c0, lnp_s=s0, **kw).to(C["DEVICE"]).init_from_data(C, data)
    m = shapelab.WithAcceptance(core, ABAR).to(C["DEVICE"])
    ck = os.path.join(D, f"variant2_{name}.pt")
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location=C["DEVICE"], weights_only=False)["state"])
    else:
        torch.manual_seed(0); t = time.time()
        shapelab.fit_fast(C, data, m, steps=4000, log=None)
        torch.save({"state": m.state_dict()}, ck)
    vals[name] = C["test_nll"](m.eval(), data["val"])
    res["val"][name], res["params"][name] = float(vals[name].mean()), int(m.n_params())
    res.setdefault("kwargs", {})[name] = {**kw, "degrees": degrees}
    print(particle, name, round(res["val"][name], 4), res["params"][name], flush=True)
best = min(vals, key=lambda k: vals[k].mean())
dif = {k: shapelab.paired(vals[k], vals[best]) for k in vals}
ok = [k for k in vals if k == best or dif[k][0] < dif[k][1]]
res.update({"best": best, "chosen": min(ok, key=lambda k: (res["params"][k], res["val"][k])), "diff_vs_best": dif,
            "diff_vs_base": {k: shapelab.paired(vals[k], vals[base_name]) for k in vals}})
json.dump(res, open(path, "w"), indent=1)
print(particle, "chosen:", res["chosen"], {k: (round(v, 4), res["params"][k]) for k, v in res["val"].items()}, flush=True)
