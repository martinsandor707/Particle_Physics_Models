"""Geometry audit, step 4c - (i) training-mean acceptance templates Abar = mean over ALL training events of A_e,
(ii) choice of the acceptance FORM on the training subsample of g4 (no validation event):

    naive   log P + ln max(A_e, f)                      (the map times the event's acceptance)
    ratio   log P + ln max(A_e, f) - ln max(Abar, f)    (for a map that was fitted to event-averaged data and so
                                                         already contains the mean acceptance)

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g4c_forms.py

Abar is pure geometry (theta, phi, origin of the training events); nothing is fitted to hits. AUDIT numbers.
"""
import json
import time

import glib as L
from glib import G, legacy, np, torch

t0 = time.time()
sets, extras, origin = L.load_cpu(("train",))
tr, ex, O = sets["train"], extras["train"], origin["train"]
rng = np.random.default_rng(20261125)
sub_idx = np.sort(rng.choice(len(tr), 4096, replace=False))             # the subsample of g4_acceptance.py
sub = tr.subset(sub_idx)
N = len(sub)
out = {"split": "train", "n_events_abar": len(tr), "n_events_scored": N, "K": L.K, "seed": L.SEED}
with L.gpulock.gpu_lock("geometry audit g4c (Abar templates, form choice on train subsample)"):
    out["machine"] = L.machine_tag()
    DEV = L.DEV
    # ---- (i) templates over all training events, in chunks of 4096 events ------------------------------------------------
    abar, A_sub = None, {}
    pos = {int(i): j for j, i in enumerate(sub_idx)}
    for s in range(0, len(tr), 4096):
        idx = np.arange(s, min(s + 4096, len(tr)))
        part = tr.subset(idx)
        A = L.acceptance_sets(part, {k: v[idx] for k, v in ex.items()}, O[idx], DEV, which=("box", "oct", "line"))
        if abar is None:
            abar = {k: torch.zeros(A[k].shape[1:], dtype=torch.float64) for k in A}
        take = np.flatnonzero(np.isin(idx, sub_idx))
        for k, a in A.items():
            abar[k] += a.double().sum(0)
            if k in ("F", "FB", "FS", "FSB"):
                A_sub.setdefault(k, []).append(a[take].clone())
        del A
        print(f"abar {s + len(idx)}/{len(tr)}  {time.time() - t0:.0f} s", flush=True)
    abar = {k: (v / len(tr)).float() for k, v in abar.items()}
    A_sub = {k: torch.cat(v) for k, v in A_sub.items()}
    assert len(A_sub["F"]) == N
    np.savez_compressed(L.ARR / "abar_train.npz", **{k: v.numpy() for k, v in abar.items()})
    L.sidecar(L.ARR / "abar_train.npz", split="train", n_events=len(tr), nsub=L.NSUB, nphi=L.NPHI,
              note="mean over all training events of the per-event acceptance A_e(z', r); pure geometry, no hits, no fit",
              keys={"F": "front", "FB": "front+back", "FS": "front+side (stepped volume, true origin)", "FSB": "all faces",
                    "*_box": "audit box", "*_oct": "straight-edged octagon", "*_line": "stepped volume, straight-line origin"})
    stored = legacy.load_abar().cpu()
    out["abar_FB_vs_stored_legacy_abar"] = {"max_abs": float((abar["FB"] - stored).abs().max()), "mean_abs": float((abar["FB"] - stored).abs().mean())}
    out["abar_summary"] = {k: {"min": float(v.min()), "mean": float(v.mean()), "cells_below_1e-3": int((v < 1e-3).sum()),
                               "cells_below_0.5": int((v < 0.5).sum())} for k, v in abar.items()}
    print(json.dumps(out["abar_FB_vs_stored_legacy_abar"]), flush=True)

    # ---- (ii) forms on the training subsample -----------------------------------------------------------------------------
    mlp = legacy.load_mlp_std()
    formula = legacy.load_component_formula(tr)
    core = formula.base
    data = sub.to(DEV)
    ab = {k: v.to(DEV) for k, v in abar.items()}
    floors = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
    import torch.nn.functional as F

    def ratio(lp, A, Ab, f):
        x = lp + torch.log(A.clamp_min(f)) - torch.log(Ab.clamp_min(f))
        return F.log_softmax(x.reshape(len(x), -1), 1).reshape(x.shape)

    def make(b, idx):
        i = idx.cpu()
        lm, lf, lc = mlp(b.lnp), formula(b.lnp), core(b.lnp)
        lc = F.log_softmax(lc.reshape(len(lc), -1), 1).reshape(lc.shape)
        d = {"mlp": lm, "formula": lf, "core": lc}
        for k in ("F", "FB", "FS", "FSB"):
            Ae = A_sub[k][i].to(DEV)
            d[f"mlp*{k}"] = G.apply_acceptance(lm, Ae, L.FLOOR)
            d[f"formula*{k}"] = G.apply_acceptance(lf, Ae, L.FLOOR)
            d[f"core*{k}"] = G.apply_acceptance(lc, Ae, L.FLOOR)
            for f in floors:
                d[f"mlp*{k}/abar@{f:g}"] = ratio(lm, Ae, ab[k], f)
            d[f"formula*{k}/abar@0.001"] = ratio(lf, Ae, ab[k], 1e-3)
        return d

    res = G.score_variants(make, data, K=L.K, seed=L.SEED, batch=L.BATCH, monitors=True)
tab = {}
for k, r in res.items():
    if k.startswith("_"):
        continue
    ref = "mlp" if k.startswith("mlp") else "formula"
    tab[k] = {"vs": ref, **L.paired0(r["nll"] - res[ref]["nll"]), "mean_nll": float(r["nll"].mean()),
              "mass_beyond_zb_mean": float(r["beyond_zb"].mean()), "mass_beyond_zb_max": float(r["beyond_zb"].max())}
    print(f"{k:28s} vs {ref:8s} {L.fmt(tab[k]):>16s}   mass beyond zb {tab[k]['mass_beyond_zb_mean']:.4f} (max {tab[k]['mass_beyond_zb_max']:.3f})", flush=True)
out["train_subsample_delta_1e-4_nat_is_x1e4"] = tab
out["paired_ratio_minus_naive"] = {k: L.paired0(res[f"mlp*{k}/abar@0.001"]["nll"] - res[f"mlp*{k}"]["nll"]) for k in ("F", "FB", "FS", "FSB")}
out["paired_core_minus_formula_ratio"] = {k: L.paired0(res[f"core*{k}"]["nll"] - res[f"formula*{k}/abar@0.001"]["nll"]) for k in ("F", "FB", "FS", "FSB")}
out["paired_FSB_minus_FS"] = {m: L.paired0(res[m.format("FSB")]["nll"] - res[m.format("FS")]["nll"])
                              for m in ("mlp*{}", "mlp*{}/abar@0.001", "core*{}", "formula*{}/abar@0.001")}
for k in ("paired_ratio_minus_naive", "paired_core_minus_formula_ratio", "paired_FSB_minus_FS"):
    print(k, {a: L.fmt(b) for a, b in out[k].items()}, flush=True)
out["seconds"] = time.time() - t0
(L.HERE / "g4c_forms.json").write_text(json.dumps(out, indent=1) + "\n")
print("done", out["seconds"])
