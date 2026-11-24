"""Hol veszít még az új formula a hálóhoz képest?  (report-mérce, teszt: új formula × Ā vs. a finomhangolt MLP)

ΔNLL_e = NLL_új − NLL_MLP = Σ_i w_i [ln g_MLP(i) − ln g_új(i)]  +  [ln M_új − ln M_MLP]
A találatonkénti tagot a találat (z', r) tartománya szerint összegezzük (átlag az eseményekre), külön p < 2 és p ≥ 2 GeV-re;
a normálási tag eseményenként egy szám. Pozitív = ott a formula rosszabb.
Kimenet: results/particles/<p>/diag_residual.json
"""
import json, math, os, sys
ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np, torch
import shapelab
from particle_shape import MASS, ParticleShape

VARIANTS = {"full": dict(long="gamma", track=True, spike=True), "full_att": dict(long="gamma", track=True, spike=True, att=True),
            "notrack": dict(long="gamma", track=False, spike=True), "plateau": dict(long="plateau", track=False, spike=True),
            "plateau_track": dict(long="plateau", track=True, spike=True, att=True)}
ZB = [-200, -20, 12, 35, 60, 100, 200, 400, 700, 1450]
RB = [0, 15, 30, 60, 100, 200, 350, 700]
out = {}
for particle in sys.argv[1:]:
    D = os.path.join(ROOT, "results", "particles", particle)
    C, geo, data = shapelab.setup_particle(particle, out_dir=D, slab=False)
    dev = C["DEVICE"]
    c0, s0 = shapelab.lnp_center_scale(data)
    new, core, t, K = shapelab.build_final(C, particle, D, c0, s0, metric="std")          # final formula (latest round) × A_bar
    n = t["degrees"]["new_std"]
    rm = shapelab.report_models(C, data, particle)
    mlp = C["FieldModel"](rm["mlp"].mlp[0].out_features, (len(rm["mlp"].mlp) - 1) // 2).to(dev)
    mlp.load_state_dict(torch.load(os.path.join(D, "mlp_std.pt"), map_location=dev, weights_only=False)["state"])
    new.eval(); mlp.eval()

    sh = data["test"]
    acc = {k: np.zeros((len(ZB) - 1, len(RB) - 1)) for k in ("lo", "hi")}
    norm, nev = {"lo": 0.0, "hi": 0.0}, {"lo": 0, "hi": 0}
    gen = torch.Generator(device=dev).manual_seed(12345)
    with torch.no_grad():
        for b in sh.batches(128):
            pts = C["fold"](b, sh.box, 64, gen)
            iz, ir = C["locate"](pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
            res = []
            for m in (new, mlp):
                logP = m(b.lnp)
                logg = torch.logsumexp((logP - C["LOG_VOL"])[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(64)
                Pz = logP.exp().sum(2)                                         # report metric: mass in front of the back face
                cdf = torch.cat([Pz.new_zeros(len(Pz), 1), torch.cumsum(Pz, 1)], 1)
                tt = ((b.zb - C["Z_MIN"]) / C["DZ"]).clamp(0, C["N_Z"]); k = tt.floor().long().clamp(max=C["N_Z"] - 1)
                ar = torch.arange(len(Pz), device=dev)
                res.append((logg, torch.log((cdf[ar, k] + Pz[ar, k] * (tt - k)).clamp_min(1e-8))))
            dh = (b.w * (res[1][0] - res[0][0])).cpu().numpy()                 # per hit: ln g_MLP - ln g_new (weighted)
            dn = (res[0][1] - res[1][1]).cpu().numpy()                         # per event: ln M_new - ln M_MLP
            z, r = b.xyz[:, 2].cpu().numpy(), torch.sqrt(b.xyz[:, 0] ** 2 + b.xyz[:, 1] ** 2).cpu().numpy()
            pe = b.p.cpu().numpy(); ev = b.ev.cpu().numpy()
            for key, sel_ev in (("lo", pe < 2), ("hi", pe >= 2)):
                hs = sel_ev[ev]
                H, _, _ = np.histogram2d(z[hs], r[hs], bins=[ZB, RB], weights=dh[hs])
                acc[key] += H
                norm[key] += float(dn[sel_ev].sum()); nev[key] += int(sel_ev.sum())
    res = {}
    for key in ("lo", "hi"):
        M = acc[key] / max(nev[key], 1)
        res[key] = {"n_events": nev[key], "hit_term_total": float(M.sum()), "norm_term": norm[key] / max(nev[key], 1),
                    "total": float(M.sum() + norm[key] / max(nev[key], 1)), "map": M.round(5).tolist()}
    out[particle] = {"z_bins": ZB, "r_bins": RB, **res}
    print(f"===== {particle} (variant {t['variant']}, n={n})", flush=True)
    for key in ("lo", "hi"):
        r_ = res[key]
        print(f"  p {'<' if key == 'lo' else '>='} 2 GeV: {r_['n_events']} events, ΔNLL total {r_['total']:+.4f} = hits {r_['hit_term_total']:+.4f} + norm {r_['norm_term']:+.4f}")
        print("     z' \\ r " + " ".join(f"{RB[j]:>4d}-{RB[j + 1]:<4d}" for j in range(len(RB) - 1)) + "   | sum")
        for i in range(len(ZB) - 1):
            row = r_["map"][i]
            print(f"  [{ZB[i]:5d},{ZB[i + 1]:5d}) " + " ".join(f"{v:+9.4f}" for v in row) + f"   | {sum(row):+.4f}")
    json.dump(out[particle], open(os.path.join(D, "diag_residual_std.json"), "w"), indent=1)
