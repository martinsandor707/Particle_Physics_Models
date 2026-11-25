"""Geometry audit, step 4 - checks of the per-event acceptance on a TRAINING subsample (no validation event).

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g4_acceptance.py

(1) quadrature convergence of A (nsub, nphi) and of the NLL of MLP x A;  (2) Monte Carlo with the independent
column-based point test;  (3) numeric front (and front + back) against the closed form of the legacy code
(``event_slab``);  (4) acceptance floor: scan on the training subsample, share of hit / fold-point energy in cells
below the floor. AUDIT numbers on the laptop's fold points; nothing here is a score of record.
"""
import json
import time

import glib as L
from glib import G, common, legacy, metric, np, torch

t0 = time.time()
sets, extras, origin = L.load_cpu(("train",))
tr, ex, O = sets["train"], extras["train"], origin["train"]
rng = np.random.default_rng(20261125)
sub_idx = np.sort(rng.choice(len(tr), 4096, replace=False))
sub = tr.subset(sub_idx)
exs = {k: v[sub_idx] for k, v in ex.items()}
Os = O[sub_idx]
N = len(sub)
out = {"split": "train (random subsample, seed 20261125)", "n_events": N, "machine": None, "K": L.K, "seed": L.SEED}
zex = G.axis_exit(sub.ev["R"].numpy(), Os, G.Volume())
zb = sub.ev["zb"].numpy().astype(np.float64)
leak = zex < np.minimum(zb, 1450.0)
out["side_leaking_fraction"] = float(leak.mean())
cols = np.load(L.HERE / "explore" / "train_columns.npz")
V3 = {"F": ("front",), "FS": ("front", "side"), "FSB": ("front", "side", "back")}

with L.gpulock.gpu_lock("geometry audit g4 (acceptance checks, train subsample)"):
    out["machine"] = L.machine_tag()
    DEV = L.DEV
    mlp = legacy.load_mlp_std()
    data = sub.to(DEV)
    R = sub.ev["R"]
    # ---- (1) quadrature convergence ---------------------------------------------------------------------------------
    n1 = 256
    pick = np.concatenate([np.flatnonzero(leak)[:n1 // 2], np.flatnonzero(~leak)[:n1 // 2]])
    Rp, Op = R[pick], torch.as_tensor(Os[pick])
    ref = G.event_acceptance(Rp, Op, G.Volume(), V3, nsub=4, nphi=512, device=DEV, chunk=4)
    ref8 = G.event_acceptance(Rp, Op, G.Volume(), V3, nsub=8, nphi=128, device=DEV, chunk=4)
    conv = {"events": int(len(pick)), "reference": "nsub 4, nphi 512", "settings": {}}
    for nsub, nphi in ((2, 16), (2, 32), (2, 64), (2, 128), (4, 64), (4, 128), (8, 128)):
        A = ref8 if (nsub, nphi) == (8, 128) else G.event_acceptance(Rp, Op, G.Volume(), V3, nsub=nsub, nphi=nphi, device=DEV, chunk=8)
        conv["settings"][f"{nsub}x{nphi}"] = {
            k: {"mean_abs": float((A[k] - ref[k]).abs().mean()), "max_abs": float((A[k] - ref[k]).abs().max()),
                "mean_signed": float((A[k] - ref[k]).mean()),
                "frac_cells_zero_where_ref_positive": float(((A[k] == 0) & (ref[k] > 0)).float().mean()),
                "max_ref_where_zero": float(ref[k][(A[k] == 0)].max()) if bool((A[k] == 0).any()) else 0.0}
            for k in V3}
        print("conv", nsub, nphi, json.dumps(conv["settings"][f"{nsub}x{nphi}"]["FSB"]), flush=True)
    del ref8
    out["quadrature_convergence_A"] = conv

    # ---- default-quadrature arrays for the whole subsample -----------------------------------------------------------
    A = L.acceptance_sets(sub, exs, Os, DEV, which=())
    A_fine = G.event_acceptance(R, torch.as_tensor(Os), G.Volume(), V3, nsub=4, nphi=256, device=DEV, chunk=8)
    A_coarse = G.event_acceptance(R, torch.as_tensor(Os), G.Volume(), V3, nsub=2, nphi=32, device=DEV)

    # ---- (3) numeric front / front+back against the closed form of the legacy code -----------------------------------
    leg_f, leg_fb = [], []
    with torch.inference_mode():
        for s in range(0, N, 256):
            b = data.batch(torch.arange(s, min(s + 256, N), device=DEV))
            leg_f.append(legacy.C["event_slab"](b, data.box, 2, True, False).cpu())
            leg_fb.append(legacy.C["event_slab"](b, data.box, 2, True, True).cpu())
    leg_f, leg_fb = torch.cat(leg_f), torch.cat(leg_fb)
    out["front_numeric_vs_legacy_closed_form"] = {
        "nsub": 2, "nphi": L.NPHI,
        "front": {"max_abs": float((A["F"] - leg_f).abs().max()), "mean_abs": float((A["F"] - leg_f).abs().mean()),
                  "mean_signed": float((A["F"] - leg_f).mean())},
        "front_back": {"max_abs": float((A["FB"] - leg_fb).abs().max()), "mean_abs": float((A["FB"] - leg_fb).abs().mean()),
                       "mean_signed": float((A["FB"] - leg_fb).mean())}}
    print("legacy check", json.dumps(out["front_numeric_vs_legacy_closed_form"]), flush=True)

    # ---- (4) floor scan and (1b) NLL-level convergence: MLP x A on the training subsample ----------------------------
    floors = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)

    def make(b, idx):
        lp = mlp(b.lnp)
        i = idx.cpu()
        d = {"mlp": lp}
        for f in floors:
            for k in ("FS", "FSB"):
                d[f"mlp*{k}@{f:g}"] = G.apply_acceptance(lp, A[k][i].to(DEV), f)
        d["mlp*F"] = G.apply_acceptance(lp, A["F"][i].to(DEV), L.FLOOR)
        d["mlp*F_legacy"] = G.apply_acceptance(lp, leg_f[i].to(DEV), L.FLOOR)
        d["mlp*FB"] = G.apply_acceptance(lp, A["FB"][i].to(DEV), L.FLOOR)
        d["mlp*FB_legacy"] = G.apply_acceptance(lp, leg_fb[i].to(DEV), L.FLOOR)
        for k in V3:
            d[f"mlp*{k}|4x256"] = G.apply_acceptance(lp, A_fine[k][i].to(DEV), L.FLOOR)
            d[f"mlp*{k}|2x32"] = G.apply_acceptance(lp, A_coarse[k][i].to(DEV), L.FLOOR)
        return d

    res = G.score_variants(make, data, K=L.K, seed=L.SEED, batch=L.BATCH, monitors=False)
    base = res["mlp"]["nll"]
    out["floor_scan_train"] = {k: L.paired0(res[k]["nll"] - base) for k in res if "@" in k}
    for k, v in out["floor_scan_train"].items():
        print("floor", k, L.fmt(v), flush=True)
    out["nll_level_checks_train"] = {
        "F_numeric_minus_F_legacy": L.paired0(res["mlp*F"]["nll"] - res["mlp*F_legacy"]["nll"]),
        "FB_numeric_minus_FB_legacy": L.paired0(res["mlp*FB"]["nll"] - res["mlp*FB_legacy"]["nll"]),
        **{f"{k}: default(2x64) minus 4x256": L.paired0(res[f"mlp*{k}@0.001" if k != "F" else "mlp*F"]["nll"] - res[f"mlp*{k}|4x256"]["nll"]) for k in V3},
        **{f"{k}: 2x32 minus 4x256": L.paired0(res[f"mlp*{k}|2x32"]["nll"] - res[f"mlp*{k}|4x256"]["nll"]) for k in V3},
        "gain_F": L.paired0(res["mlp*F"]["nll"] - base), "gain_FB": L.paired0(res["mlp*FB"]["nll"] - base)}
    for k, v in out["nll_level_checks_train"].items():
        print("nll-level", k, L.fmt(v), flush=True)

    # ---- share of hit-centre and fold-point energy in cells with small acceptance ------------------------------------
    thr = (0.0, 1e-3, 1e-2, 0.1, 0.5)
    share = {k: {"centre": np.zeros(len(thr)), "fold": np.zeros(len(thr))} for k in ("FS", "FSB")}
    allk = np.zeros(2)
    with torch.inference_mode():
        gen = torch.Generator(device=DEV).manual_seed(L.SEED)
        for s in range(0, N, L.BATCH):
            idx = torch.arange(s, min(s + L.BATCH, N), device=DEV)
            b = data.batch(idx)
            u = metric.draw(b, data.box, L.K, gen)
            pts = b.xyz[:, None, :] + torch.einsum("hij,hkj->hki", b.R[b.ev], u)
            iz, ir = common.locate(pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
            izc, irc = common.locate(b.xyz[:, 2], torch.sqrt(b.xyz[:, 0] ** 2 + b.xyz[:, 1] ** 2))
            for k in share:
                Ab = A[k][idx.cpu()].to(DEV)
                af = Ab[b.ev[:, None].expand_as(iz), iz, ir]
                ac = Ab[b.ev, izc, irc]
                for j, t in enumerate(thr):
                    share[k]["fold"][j] += float((b.w[:, None] * (af <= t)).sum() / L.K)
                    share[k]["centre"][j] += float((b.w * (ac <= t)).sum())
                if k == "FSB":
                    allk[0] += float((b.w * (af <= 0).all(1)).sum())           # hits with ALL fold points in A = 0 cells
                    allk[1] += float((b.w * (af <= 1e-3).all(1)).sum())
    out["energy_share_in_low_acceptance_cells_train"] = {
        k: {"thresholds_A_le": list(thr), "hit_centres": (v["centre"] / N).tolist(), "fold_points": (v["fold"] / N).tolist()}
        for k, v in share.items()}
    out["energy_share_hits_with_all_fold_points_in_cells_A_eq_0_FSB"] = float(allk[0] / N)
    out["energy_share_hits_with_all_fold_points_in_cells_A_le_1e-3_FSB"] = float(allk[1] / N)
    print(json.dumps(out["energy_share_in_low_acceptance_cells_train"]), allk / N, flush=True)
    del data, mlp
    torch.cuda.empty_cache()

# ---- (2) Monte Carlo with the independent column-based point test (CPU) ---------------------------------------------
nm, Mpts = 48, 256
pick = np.concatenate([np.flatnonzero(leak)[-nm // 2:], np.flatnonzero(~leak)[-nm // 2:]])
ze, re = common.Z_EDGES.cpu().numpy().astype(np.float64), common.R_EDGES.cpu().numpy().astype(np.float64)
mrng = np.random.default_rng(7)
dif = {k: [] for k in ("FS", "FSB")}
zsc = {k: [] for k in ("FS", "FSB")}
for e in pick:
    th, ph = float(exs["theta"][e]), float(exs["phi"][e])
    z = ze[:-1, None, None] + common.DZ * mrng.random((common.N_Z, common.N_R, Mpts))
    r = np.sqrt(re[None, :-1, None] ** 2 + (re[None, 1:, None] ** 2 - re[None, :-1, None] ** 2) * mrng.random((common.N_Z, common.N_R, Mpts)))
    a = mrng.uniform(0, 2 * np.pi, (common.N_Z, common.N_R, Mpts))
    off = G.to_global(np.stack([r * np.cos(a), r * np.sin(a), z], -1), th, ph)
    g = off + Os[e]
    side = G.columns_inside(g[..., 0], g[..., 1], cols["ux"], cols["uy"])
    front = off[..., 2] >= -G.LAYER / 2
    back = g[..., 2] <= G.Z_BACK
    for k, m in (("FS", side & front), ("FSB", side & front & back)):
        mc = m.mean(2)
        d = A[k][e].numpy().astype(np.float64) - mc
        dif[k].append(d)
        zsc[k].append(d / np.hypot(np.sqrt(np.clip(mc * (1 - mc), 0, None) / Mpts), 1e-9 + 0.0))
out["monte_carlo_cross_check_train"] = {"events": int(len(pick)), "points_per_cell": Mpts,
                                        "point_test": "sfm.geometry.columns_inside on explore/train_columns.npz (independent of the constants)"}
for k in dif:
    d = np.stack(dif[k])
    mcz = np.stack(zsc[k])
    fin = np.isfinite(mcz)
    out["monte_carlo_cross_check_train"][k] = {
        "mean_signed": float(d.mean()), "se_of_mean": float(d.std() / np.sqrt(d.size)), "mean_abs": float(np.abs(d).mean()),
        "max_abs": float(np.abs(d).max()),
        "rms_z_cells_with_binomial_noise": float(np.sqrt((mcz[fin] ** 2).mean())),
        "cells_mc_0_or_1_with_quad_difference_gt_0.02": int(((~fin) & (np.abs(d) > 0.02)).sum()), "cells": int(d.size)}
    print("MC", k, json.dumps(out["monte_carlo_cross_check_train"][k]), flush=True)
out["seconds"] = time.time() - t0
(L.HERE / "g4_acceptance.json").write_text(json.dumps(out, indent=1) + "\n")
print("done", out["seconds"])
