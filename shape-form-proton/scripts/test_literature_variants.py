"""Irodalmi alak-változatok a hadronokra (proton, pion): Bock-féle mélységprofil (két gamma, a kezdőpont-eloszlással konvolválva),
nyommal (csillapítással) és csúccsal, ill. nyom nélkül. Ugyanaz a protokoll, mint a pipeline változatválasztásánál (report-mérce × Ā,
n = 2, 4000 lépés, validáció). Kimenet: results/particles/<p>/variants_literature.json
    python test_literature_variants.py <particle>"""
import json, os, sys, time
ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import torch
import shapelab
from particle_shape import MASS, ParticleShape
particle = sys.argv[1]
OUT = os.path.join(ROOT, "results", "particles", particle)
C, geo, data = shapelab.setup_particle(particle, out_dir=OUT)
c0, s0 = shapelab.lnp_center_scale(data)
ABAR = torch.load(os.path.join(OUT, "abar.pt"), map_location=C["DEVICE"])
VAR = json.load(open(os.path.join(OUT, "variants.json")))           # round-1 choice (validation) for the paired comparison
VARIANTS = {"full": dict(long="gamma", track=True, spike=True), "full_att": dict(long="gamma", track=True, spike=True, att=True),
            "notrack": dict(long="gamma", track=False, spike=True), "plateau": dict(long="plateau", track=False, spike=True),
            "plateau_track": dict(long="plateau", track=True, spike=True, att=True)}
bc = ParticleShape(C, MASS[particle], degree=2, lnp0=c0, lnp_s=s0, **VARIANTS[VAR["chosen"]]).to(C["DEVICE"])
bm = shapelab.WithAcceptance(bc, ABAR).to(C["DEVICE"])
bm.load_state_dict(torch.load(os.path.join(OUT, f"variant_{VAR['chosen']}.pt"), map_location=C["DEVICE"], weights_only=False)["state"])
VBASE = C["test_nll"](bm.eval(), data["val"])
V = {"bock_track": dict(long="bock", track=True, spike=True, att=True), "bock": dict(long="bock", track=False, spike=True)}
res = json.load(open(os.path.join(OUT, "variants_literature.json"))) if os.path.exists(os.path.join(OUT, "variants_literature.json")) else {}
for name, kw in V.items():
    if name in res:
        continue
    core = ParticleShape(C, MASS[particle], degree=2, lnp0=c0, lnp_s=s0, **kw).to(C["DEVICE"]).init_from_data(C, data)
    m = shapelab.WithAcceptance(core, ABAR).to(C["DEVICE"])
    with torch.no_grad():
        P = m(torch.log(torch.tensor([0.5, 5.0, 18.0], device=C["DEVICE"]))).exp()
    assert torch.allclose(P.sum((1, 2)), torch.ones(3, device=C["DEVICE"]), atol=1e-4)
    torch.manual_seed(0); t = time.time()
    shapelab.fit_fast(C, data, m, steps=4000, log=None)
    torch.save({"state": m.state_dict()}, os.path.join(OUT, f"variant_{name}.pt"))
    v = C["test_nll"](m.eval(), data["val"])
    res[name] = {"val": float(v.mean()), "params": int(m.n_params()), "sec": round(time.time() - t), "base": VAR["chosen"],
                 "base_val": float(VBASE.mean()), "diff_vs_base": shapelab.paired(v, VBASE)}
    with torch.no_grad():
        th = core.theta(torch.log(torch.tensor([1.0, 5.0, 18.0], device=C["DEVICE"])))
        res[name]["fitted_at_p_1_5_18GeV"] = {k: [round(float(x), 4) for x in th[k].flatten()] for k in
                                              ("logit_w", "log_a", "log_b", "log_d", "log_X0", "log_lamI", "z_st", "log_fb", "log_lb") if k in th}
    json.dump(res, open(os.path.join(OUT, "variants_literature.json"), "w"), indent=1)
    print(particle, name, res[name], flush=True)
