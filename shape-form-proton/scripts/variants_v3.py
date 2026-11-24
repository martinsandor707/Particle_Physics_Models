"""3. kör: a végleges (2. körös) változat további fizikai kiegészítései (report-mérce × Ā, n = 2, 4000 lépés, validáció).
  halo   – udvar a primer nyom körül (delta-elektronok): (1-f) Gauss + f e^(-r/l)/(2 pi l r)      [csak nyommal]
  bragg  – Bragg–Kleeman-felfutás a hatótáv vége felé, (R - z')^(-q)                           [csak nyommal]
  curv   – a laterális magszélesség és a keverési arányok mélységfüggése négyzetes taggal
  slopes – a laterális mélység-meredekségek impulzusfüggők (2. fokú polinom)
  ms     – a nyom szélessége a többszörös szórás miatt a mélységgel nő (Highland): σ² = σ0² + (c (z'/1 m)^k / βp)²   [csak nyommal]
Választás: a legkevesebb paraméteres a legjobbtól 1 SE-n belül (a kiindulással együtt). Kimenet: results/particles/<p>/variants_v3.json"""
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
V2J = json.load(open(os.path.join(D, "variants_v2.json")))
base_name = V2J["chosen"]
base = dict(V2J["kwargs"][base_name]); base_deg = base.pop("degrees", None) or {}
SLOPES = {k: 2 for k in ("s_sig", "s_R", "s_lam", "s_1", "s_2")}
track = bool(base.get("track"))
cands = {}
if track:
    cands.update({"halo": dict(track_halo=True), "bragg": dict(bragg=True), "halo+bragg": dict(track_halo=True, bragg=True)})
cands.update({"curv": dict(curv=True), "slopes": dict(slopes=True)})
cands["all3"] = dict(curv=True, slopes=True, **(dict(track_halo=True, bragg=True) if track else {}))
if track:                                                        # multiple scattering of the primary (Highland)
    cands.update({"ms": dict(ms=True), "ms+halo": dict(ms=True, track_halo=True), "ms+curv+slopes": dict(ms=True, curv=True, slopes=True),
                  "all4": dict(ms=True, curv=True, slopes=True, track_halo=True, bragg=True)})
path = os.path.join(D, "variants_v3.json")
res = json.load(open(path)) if os.path.exists(path) else {"val": {}, "params": {}, "kwargs": {}}
vals = {}
core = ParticleShape(C, MASS[particle], degree=2, degrees=base_deg or None, lnp0=c0, lnp_s=s0, **base).to(C["DEVICE"])
m = shapelab.WithAcceptance(core, ABAR).to(C["DEVICE"])
m.load_state_dict(torch.load(os.path.join(D, f"variant2_{base_name}.pt"), map_location=C["DEVICE"], weights_only=False)["state"])
vals["base"] = C["test_nll"](m.eval(), data["val"]); res["val"]["base"], res["params"]["base"] = float(vals["base"].mean()), int(m.n_params())
res["kwargs"]["base"] = {**base, "degrees": base_deg}
for name, ext in cands.items():
    ext = dict(ext); slopes = ext.pop("slopes", False)
    kw, deg = {**base, **ext}, {**base_deg, **(SLOPES if slopes else {})}
    core = ParticleShape(C, MASS[particle], degree=2, degrees=deg or None, lnp0=c0, lnp_s=s0, **kw).to(C["DEVICE"]).init_from_data(C, data)
    m = shapelab.WithAcceptance(core, ABAR).to(C["DEVICE"])
    ck = os.path.join(D, f"variant3_{name}.pt")
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location=C["DEVICE"], weights_only=False)["state"])
    else:
        torch.manual_seed(0)
        shapelab.fit_fast(C, data, m, steps=4000, log=None)
        torch.save({"state": m.state_dict()}, ck)
    vals[name] = C["test_nll"](m.eval(), data["val"])
    res["val"][name], res["params"][name] = float(vals[name].mean()), int(m.n_params())
    res["kwargs"][name] = {**kw, "degrees": deg}
    print(particle, name, round(res["val"][name], 4), res["params"][name], flush=True)
best = min(vals, key=lambda k: vals[k].mean())
dif = {k: shapelab.paired(vals[k], vals[best]) for k in vals}
ok = [k for k in vals if k == best or dif[k][0] < dif[k][1]]
res.update({"base_name": base_name, "best": best, "chosen": min(ok, key=lambda k: (res["params"][k], res["val"][k])), "diff_vs_best": dif,
            "diff_vs_base": {k: shapelab.paired(vals[k], vals["base"]) for k in vals}})
json.dump(res, open(path, "w"), indent=1)
print(particle, "chosen:", res["chosen"], {k: (round(v, 4), res["params"][k]) for k, v in res["val"].items()}, flush=True)
