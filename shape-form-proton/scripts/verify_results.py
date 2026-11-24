"""Független ellenőrzés a pipeline eredményeire (minden részecskére, amelynek kész a test/s2 lépése).

Mit ellenőriz:
 1. a végleges modelleket a mentett checkpointokból újraépíti, és a teszt-NLL-t újraszámolja:
    report-mérce: a 00_common változatlan test_nll-jével; 3D-mérce: SAJÁT normálással (nsub = 8 részpont cellánként, a pipeline 2-t használ);
 2. az eseményenkénti értékek egyeznek-e a pipeline npz-fájljával (report-mérce: pontosan; 3D: a kvadratúra-eltérésen belül);
 3. a páros különbségek és a rés-hányadok újraszámolva egyeznek-e a JSON-nal;
 4. normáltság (sum P = 1) és a választások: a variants/ladder JSON-ok csak validációs számokat tartalmaznak.
Eredmény: results/particles/verify.json
"""
import json
import math
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch

import shapelab
from particle_shape import MASS, ParticleShape

P_ROOT = os.path.join(ROOT, "results", "particles")
VARIANTS = {"full": dict(long="gamma", track=True, spike=True), "full_att": dict(long="gamma", track=True, spike=True, att=True),
            "notrack": dict(long="gamma", track=False, spike=True), "plateau": dict(long="plateau", track=False, spike=True),
            "plateau_track": dict(long="plateau", track=True, spike=True, att=True)}
out = {}
for particle in ("electron", "pion", "proton", "muon"):
    D = os.path.join(P_ROOT, particle)
    K, SUF, TAG = shapelab.final_round(D)
    if not os.path.exists(os.path.join(D, f"s2{SUF}.json")):
        out[particle] = "not finished"
        continue
    J = lambda n: json.load(open(os.path.join(D, f"{n}.json")))
    test, s2, var = J("test" + SUF), J("s2" + SUF), J("variants")
    C, geo, data = shapelab.setup_particle(particle, out_dir=D, slab=False)
    dev = C["DEVICE"]
    c0, s0 = shapelab.lnp_center_scale(data)
    ABAR = torch.load(os.path.join(D, "abar.pt"), map_location=dev)
    deg = test["degrees"]

    def new(n, acc):
        return shapelab.build_final(C, particle, D, c0, s0, metric="std" if acc else "slab", n=n)[0]

    m_std, m_slab = new(deg["new_std"], True), new(deg["new_slab"], False)
    npz = dict(np.load(os.path.join(D, f"test{SUF}_nll.npz")))
    r = {"variant": test["variant"], "degrees": deg, "round": K}
    # 1-2: report metric, unchanged code
    t_std = C["test_nll"](m_std, data["test"])
    r["std_new_test"] = float(t_std.mean())
    r["std_new_max_abs_diff_vs_pipeline"] = float(np.abs(t_std - npz["std:new"]).max())
    # 3D metric with an independent, finer normalisation (nsub = 8)
    with torch.no_grad():
        tot, gen = [], torch.Generator(device=dev).manual_seed(12345)
        for b in data["test"].batches(128):
            logP = m_slab(b.lnp)
            pts = C["fold"](b, data["test"].box, C["N_FOLD_EVAL"], gen)
            iz, ir = C["locate"](pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
            logg = torch.logsumexp((logP - C["LOG_VOL"])[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(C["N_FOLD_EVAL"])
            mass = (logP.exp() * C["event_slab"](b, data["test"].box, 8)).sum((1, 2))
            ll = logg - torch.log(mass.clamp_min(1e-8))[b.ev]
            tot.append(-torch.zeros(len(b.lnp), device=dev).index_add_(0, b.ev, b.w * ll))
        t_slab = torch.cat(tot).cpu().numpy()
    r["slab_new_test_nsub8"] = float(t_slab.mean())
    r["slab_new_pipeline"] = float(npz["slab:new"].mean())
    r["slab_new_mean_diff_nsub8_minus_nsub2"] = float((t_slab - npz["slab:new"]).mean())
    # 3: paired differences and gap fractions recomputed from the stored per-event values
    rec = {}
    for metric, ref, mlp in (("std", "std:base_acc", "std:mlp_ft"), ("slab", "slab:base", "slab:mlp_ft")):
        d = shapelab.paired(npz[f"{metric}:new"], npz[ref])
        g = (npz[ref].mean() - npz[f"{metric}:new"].mean()) / (npz[ref].mean() - npz[mlp].mean())
        rec[metric] = {"delta_new_vs_base": d, "gap_closed": float(g),
                       "json_delta": test["delta_vs_base"][f"{metric}:new"], "json_gap": test["gap_closed"][f"{metric}:new"]}
    r["recomputed"] = rec
    r["consistent"] = all(abs(rec[m]["delta_new_vs_base"][0] - rec[m]["json_delta"][0]) < 1e-9 and abs(rec[m]["gap_closed"] - rec[m]["json_gap"]) < 1e-9
                          for m in rec)
    # 4: normalisation and selection hygiene
    with torch.no_grad():
        lnp = torch.linspace(math.log(0.25), math.log(20.0), 32, device=dev)
        r["max_abs_sum_P_minus_1"] = max(float((m(lnp).exp().sum((1, 2)) - 1).abs().max()) for m in (m_std, m_slab))
    sel = [J(n) for n in ("variants", "base_std", "base_slab", f"{TAG}_std", f"{TAG}_slab")] + [J(f"variants_v{k}") for k in range(2, K + 1)]
    r["selection_files_mention_test"] = any("test" in json.dumps(s).lower() for s in sel)
    r["s2_events"] = s2["n_events"]
    r["s2_delta_new_vs_base"] = {m: s2["delta_vs_base"][f"{m}:new"] for m in ("std", "slab")}
    r["test_delta_new_vs_base"] = {m: test["delta_vs_base"][f"{m}:new"] for m in ("std", "slab")}
    r["same_sign_test_s2"] = all(np.sign(r["s2_delta_new_vs_base"][m][0]) == np.sign(r["test_delta_new_vs_base"][m][0]) for m in ("std", "slab"))
    out[particle] = r
    print(particle, json.dumps(r, indent=1), flush=True)
json.dump(out, open(os.path.join(P_ROOT, "verify.json"), "w"), indent=1)
