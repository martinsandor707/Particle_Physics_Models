"""A végleges formula kiegészítő tagjainak illesztett értékei (energiahányad és skálák) néhány impulzusnál – a README értelmezéséhez.
Kimenet: results/particles/extension_params.json"""
import json, math, os, sys
ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import torch
import shapelab
from particle_shape import MASS, ParticleShape
out = {}
for particle in ("electron", "muon", "proton", "pion"):
    D = os.path.join(ROOT, "results", "particles", particle)
    C, geo, data = shapelab.setup_particle(particle, out_dir=D, slab=False)
    dev = C["DEVICE"]; c0, s0 = shapelab.lnp_center_scale(data)
    m, core, t, K = shapelab.build_final(C, particle, D, c0, s0, metric="std")
    P = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    with torch.no_grad():
        lw, comp, th = core.components(torch.log(torch.tensor(P, device=dev)))
    names = ["shower", "track", "spike"] + (["end"] if core.endblob else []) + (["far"] if core.far else [])
    W = {nm: [round(float(torch.exp(w).flatten()[i]), 4) for i in range(len(P))] for nm, w in zip(names, lw) if w is not None}
    sc = lambda k: round(float(torch.exp(th[k]).flatten()[0]), 1)
    r = {"round": K, "variant": t["variant"], "p_GeV": P, "weights": W}
    if getattr(core, "ms", False):
        r["ms_c_mm"], r["ms_pow"] = sc("log_cms"), round(float(th["ms_pow"].flatten()[0]), 2)
        r["track_sigma0_mm"] = sc("log_st")
    if getattr(core, "track_halo", False):
        r["track_halo_frac"] = [round(float(torch.sigmoid(th["logit_th"]).flatten()[i]), 3) for i in range(len(P))]
        r["track_halo_l_mm"] = sc("log_lh")
    if getattr(core, "bragg", False):
        r["bragg_q"] = round(float(th["q_bk"].flatten()[0]), 2)
    if core.far:
        r["far_L_mm"], r["far_R_mm"] = sc("log_Lf"), sc("log_Rf")
    if core.endblob:
        r["end_sigma_z_mm"], r["end_sigma_r_mm"] = sc("log_sez"), sc("log_ser")
        r["range_mm"] = [round(float(v), 0) for v in core.range_mm(th).flatten()]
    if core.use_spike:
        r["spike_sigma_z_mm"] = [round(float(v), 1) for v in torch.exp(th["log_sp_sz"]).flatten()]
        r["spike_sigma_r_mm"] = [round(float(v), 1) for v in torch.exp(th["log_sp_sr"]).flatten()]
    out[particle] = r
    print(particle, json.dumps(r), flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "particles", "extension_params.json"), "w"), indent=1)
