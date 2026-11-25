"""Geometry audit, step 5k - guard G-K for the primary maps on VALIDATION: the report metric's estimator at K = 1024.

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g5k_gk.py

Same events, seed 12345, batches of 128, K = 1024 in chunks of 64 (as sfm.registry._guard_arrays). G-K measures a
difference of Jensen biases (PLAN section 13). AUDIT numbers; no registry row.
"""
import json
import time

import glib as L
from glib import G, legacy, np, torch
import torch.nn.functional as F

t0 = time.time()
sets, extras, origin = L.load_cpu(("train", "val"))
val, ex, O = sets["val"], extras["val"], origin["val"]
N = len(val)
abar = {k: torch.from_numpy(v) for k, v in np.load(L.ARR / "abar_train.npz").items()}
KG = 1024
with L.gpulock.gpu_lock("geometry audit g5k (validation, G-K at K=1024)"):
    machine = L.machine_tag()
    DEV = L.DEV
    V = {"F": ("front",), "FS": ("front", "side"), "FSB": ("front", "side", "back")}
    A = G.event_acceptance(val.ev["R"], torch.as_tensor(O), G.Volume(), V, nsub=L.NSUB, nphi=L.NPHI, device=DEV)
    mlp = legacy.load_mlp_std()
    formula = legacy.load_component_formula(sets["train"])
    core = formula.base
    ab = {k: abar[k].to(DEV) for k in V}

    def make(b, idx):
        i = idx.cpu()
        lm, lf, lc = mlp(b.lnp), formula(b.lnp), core(b.lnp)
        lc = F.log_softmax(lc.reshape(len(lc), -1), 1).reshape(lc.shape)
        d = {"mlp": lm, "formula": lf}
        for k in V:
            Ae = A[k][i].to(DEV)
            d[f"mlp*{k}/abar"] = G.apply_acceptance(lm, Ae, L.FLOOR, template=ab[k])
            d[f"formula*{k}/abar"] = G.apply_acceptance(lf, Ae, L.FLOOR, template=ab[k])
            if k != "F":
                d[f"mlp*{k}"] = G.apply_acceptance(lm, Ae, L.FLOOR)
                d[f"core*{k}"] = G.apply_acceptance(lc, Ae, L.FLOOR)
        return d

    res = G.score_variants(make, val.to(DEV), K=KG, seed=L.SEED, batch=L.BATCH, chunk=64, monitors=False)
names = [k for k in res if not k.startswith("_")]
np.savez_compressed(L.ARR / "val_k1024.npz", event_number=ex["event_number"], **{f"{k}|nll": res[k]["nll"] for k in names})
L.sidecar(L.ARR / "val_k1024.npz", split="val", n_events=N, metric="report-metric estimator at K = 1024 (guard G-K), chunk 64", K=KG,
          seed=L.SEED, batch=L.BATCH, machine=machine, floor=L.FLOOR,
          label="AUDIT numbers; a guard metric: never in a column or delta with K = 64 report-metric numbers")
rep = np.load(L.ARR / "val_report.npz")
out = {"K": KG, "n": N, "machine": machine, "maps": {}}
for k in names:
    if k in ("mlp", "formula"):
        continue
    b = "mlp" if k.startswith("mlp") else "formula"
    main = rep[f"{k}|nll"] - rep[f"{b}|nll"]
    gk = res[k]["nll"] - res[b]["nll"]
    dec = L.stats.guard_persistence(main, gk)
    out["maps"][k] = {"delta_K1024": L.paired0(gk), "delta_K64_report": L.paired0(main), "margin": dec.numbers["margin"],
                      "se_margin": dec.numbers["se_margin"], "z_margin": dec.numbers["z_margin"], "fraction": dec.numbers["fraction"],
                      "passed": bool(dec.passed), "checks": {a: bool(v) for a, v in dec.checks.items()}}
    print(f"{k:22s} K=1024: {L.fmt(out['maps'][k]['delta_K1024'])}   [K=64: {L.fmt(out['maps'][k]['delta_K64_report'])}]   "
          f"margin {dec.numbers['margin'] * 1e4:+.1f} ± {dec.numbers['se_margin'] * 1e4:.1f}  passed {dec.passed}", flush=True)
out["seconds"] = time.time() - t0
(L.HERE / "g5k_gk.json").write_text(json.dumps(out, indent=1) + "\n")
print("done", out["seconds"])
