"""Geometry audit, step 5 - question (b): acceptance gains on VALIDATION (no training, no registry row).

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g5_gains.py

Models: the frozen benchmark MLP (mlp_std) and the component formula through sfm.legacy - stand-ins for the
theta-blind finalists, which are not frozen yet. Every map is scored on the fold points of common.test_nll
(one generator draw per batch of 128, seed 12345, K = 64) inside sfm.metric.deterministic(); the base maps must
reproduce results/m0_parity/{mlp,formula}_val_nll.npy bit for bit. Two passes:
    report metric   all 14,950 validation events
    G-bulk          stratum zb >= 1221 mm, Bulk(z' < 1150 mm, without the last ring, short="raise") - PLAN A4
AUDIT numbers on this laptop's fold points (RTX 4060, launch grid 144): they decide whether and in which form the
geometry arm is run and are not scores of record. Per-event arrays -> arrays/val_*.npz (+ .json).
"""
import json
import time

import glib as L
from glib import G, legacy, metric, np, torch
import torch.nn.functional as F

t0 = time.time()
sets, extras, origin = L.load_cpu(("train", "val"))
val, ex, O = sets["val"], extras["val"], origin["val"]
N = len(val)
assert N == 14950
zb = val.ev["zb"].numpy().astype(np.float64)
abar = {k: torch.from_numpy(v) for k, v in np.load(L.ARR / "abar_train.npz").items()}
FOUR = ("F", "FB", "FS", "FSB")
ALT = ("FS_box", "FSB_box", "FS_oct", "FSB_oct", "FS_line", "FSB_line")
stratum = np.flatnonzero(zb >= 1221.0)
out = {"split": "val", "n_events": N, "n_stratum_zb_ge_1221": int(len(stratum)), "K": L.K, "seed": L.SEED, "batch": L.BATCH,
       "floor": L.FLOOR, "nsub": L.NSUB, "nphi": L.NPHI}

with L.gpulock.gpu_lock("geometry audit g5 (validation gains, report metric + G-bulk)"):
    out["machine"] = L.machine_tag()
    DEV = L.DEV
    A = L.acceptance_sets(val, ex, O, DEV, which=("box", "oct", "line"))
    print(f"acceptance maps done {time.time() - t0:.0f} s", flush=True)
    mlp = legacy.load_mlp_std()
    formula = legacy.load_component_formula(sets["train"])
    core = formula.base
    ab = {k: v.to(DEV) for k, v in abar.items()}
    lnS = {}

    def run(data, index, bulk):
        """index: positions of data's events in the validation set (for the acceptance arrays)."""
        index_t = torch.as_tensor(index)

        def make(b, idx):
            i = index_t[idx.cpu()]
            lm, lf, lc = mlp(b.lnp), formula(b.lnp), core(b.lnp)
            lc = F.log_softmax(lc.reshape(len(lc), -1), 1).reshape(lc.shape)
            d = {"mlp": lm, "formula": lf, "core": lc}
            fac = {}
            for k in FOUR + ALT:
                Ae = A[k][i].to(DEV)
                d[f"mlp*{k}"] = G.apply_acceptance(lm, Ae, L.FLOOR)
                d[f"mlp*{k}/abar"] = G.apply_acceptance(lm, Ae, L.FLOOR, template=ab[k])
                d[f"core*{k}"] = G.apply_acceptance(lc, Ae, L.FLOOR)
                if k in FOUR:
                    d[f"formula*{k}"] = G.apply_acceptance(lf, Ae, L.FLOOR)
                    d[f"formula*{k}/abar"] = G.apply_acceptance(lf, Ae, L.FLOOR, template=ab[k])
                if bulk is None:                       # ln sum_c P_c f_c of the factor that was applied (before renormalising)
                    la = torch.log(Ae.clamp_min(L.FLOOR))
                    lt = torch.log(ab[k].clamp_min(L.FLOOR))
                    fac[f"mlp*{k}"] = torch.logsumexp((lm + la).reshape(len(lm), -1), 1)
                    fac[f"mlp*{k}/abar"] = torch.logsumexp((lm + la - lt).reshape(len(lm), -1), 1)
                    fac[f"core*{k}"] = torch.logsumexp((lc + la).reshape(len(lm), -1), 1)
                    if k in FOUR:
                        fac[f"formula*{k}"] = torch.logsumexp((lf + la).reshape(len(lm), -1), 1)
                        fac[f"formula*{k}/abar"] = torch.logsumexp((lf + la - lt).reshape(len(lm), -1), 1)
            for k, v in fac.items():
                lnS.setdefault(k, []).append(v.double().cpu())
            return d

        return G.score_variants(make, data, K=L.K, seed=L.SEED, batch=L.BATCH, bulk=bulk)

    data = val.to(DEV)
    rep = run(data, np.arange(N), None)
    print(f"report metric done {time.time() - t0:.0f} s", flush=True)
    sub = data.subset(torch.as_tensor(stratum, device=DEV))
    blk = run(sub, stratum, metric.Bulk(short="raise"))
    print(f"G-bulk done {time.time() - t0:.0f} s", flush=True)
    del data, sub
    torch.cuda.empty_cache()

# ---- parity of the base maps with the stored laptop arrays --------------------------------------------------------------
par = {}
for name in ("mlp", "formula"):
    ref = np.load(L.paths.M0_DIR / f"{name}_val_nll.npy")
    mine = rep[name]["nll"].astype(np.float32)
    par[name] = {"bit_identical": bool(np.array_equal(mine, ref)), "max_abs_diff": float(np.abs(mine.astype(np.float64) - ref).max()),
                 "mean": float(mine.mean(dtype=np.float64)), "stored_mean": float(ref.mean(dtype=np.float64))}
out["parity_with_m0_arrays"] = par
print(json.dumps(par), flush=True)
assert all(v["max_abs_diff"] < 1e-4 for v in par.values())

lnS = {k: torch.cat(v).numpy() for k, v in lnS.items()}
names = [k for k in rep if not k.startswith("_")]
save = {"event_number": ex["event_number"], "weight_sum": rep["_shared"]["weight_sum"]}
for k in names:
    for q, v in rep[k].items():
        save[f"{k}|{q}"] = v
    if k in lnS:
        save[f"{k}|ln_factor_mass"] = lnS[k]
np.savez_compressed(L.ARR / "val_report.npz", **save)
L.sidecar(L.ARR / "val_report.npz", split="val", n_events=N, metric="report metric (common.test_nll fold points)", K=L.K, seed=L.SEED,
          batch=L.BATCH, machine=out["machine"], floor=L.FLOOR, nsub=L.NSUB, nphi=L.NPHI, parity=par,
          keys="<map>|nll, |hit (sum_i w_i ln g_i), |log_mass (ln M_e), |beyond_zb, |last_ring, |outside_bulk, |total, "
               "|ln_factor_mass (ln sum_c P_c f_c before renormalisation)",
          maps="base maps mlp, formula (= core x stored abar), core; X*A naive form; X*A/abar ratio form (template = arrays/abar_train.npz)",
          label="AUDIT numbers on the laptop's fold points; not scores of record; inputs of the geometry maps: p, theta, phi, entry position")
saveb = {"event_number": ex["event_number"][stratum], "stratum_index": stratum, **{f"_shared|{k}": v for k, v in blk["_shared"].items()}}
for k in names:
    for q, v in blk[k].items():
        saveb[f"{k}|{q}"] = v
np.savez_compressed(L.ARR / "val_bulk_zb1221.npz", **saveb)
L.sidecar(L.ARR / "val_bulk_zb1221.npz", split="val, stratum zb >= 1221 mm", n_events=int(len(stratum)),
          metric="G-bulk: Bulk(z_cut=1150, drop_rings=1, renormalise=True, short='raise') on val.subset(stratum); own fold points "
                 "(batches of 128 of the subset, seed 12345, K 64) as sfm.registry._guard_arrays does",
          K=L.K, seed=L.SEED, batch=L.BATCH, machine=out["machine"], floor=L.FLOOR,
          label="AUDIT numbers, bulk-restricted: never in a column or delta with report-metric numbers")
# ---- per-event descriptors ------------------------------------------------------------------------------------------------
Lo = G.straight_line_origin(ex["theta"], ex["phi"], zb)
zex = G.axis_exit(val.ev["R"].numpy(), O, G.Volume())
zex_box = G.axis_exit(val.ev["R"].numpy(), O, G.Volume("box"))
ent_hole = ~G.Volume().inside_xy_np(O[:, 0], O[:, 1])
np.savez_compressed(L.ARR / "val_events.npz", event_number=ex["event_number"], theta=ex["theta"], phi=ex["phi"], zb=zb,
                    p=val.ev["p"].numpy().astype(np.float64), origin=O, first_hit_layer=np.rint((O[:, 2] - G.Z_LAYER0) / G.LAYER).astype(int),
                    axis_exit_stepped=zex, axis_exit_box=zex_box, origin_displacement_mm=np.hypot(O[:, 0] - Lo[:, 0], O[:, 1] - Lo[:, 1]),
                    mean_A=np.stack([A[k].mean((1, 2)).numpy() for k in FOUR], 1))
L.sidecar(L.ARR / "val_events.npz", split="val", n_events=N, note="descriptors for the splits of g6_tables.py; mean_A columns: F, FB, FS, FSB")
out["seconds"] = time.time() - t0
(L.HERE / "g5_gains.json").write_text(json.dumps(out, indent=1) + "\n")
print("done", out["seconds"])
