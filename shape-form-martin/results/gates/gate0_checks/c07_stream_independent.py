"""Gate 0 checks 3 (stream-independent part) and 8: everything that does not depend on the fold points.

  1. ln p centring (lnp0, lnp_s) recomputed from the training set vs the checkpoint buffers; what it would be if
     validation or test momenta had been included (test p range from the Parquet-derived events_table.npz).
  2. v0, vs (ParticleShape.__init__ rule) vs the checkpoint buffers.
  3. formula_values.json["std"] (data-free numbers written on the reference machine by scripts/formula_card.py:
     T, component weights, in-detector shares with Abar, physical parameters at p = 0.5..20 GeV) recomputed here.
  4. audit_v3.json (normalisation, effective parameter count by perturbation) recomputed here.
  5. Abar: recomputed from training events only (front+back, nsub=4) vs abar.pt; front-only; sensitivity to
     including validation events, to nsub; logA buffer of the checkpoint; brute-force azimuthal integration.
  6. grid buffers stored in the checkpoints vs the local grid; model maps CPU vs GPU.
Training and validation data only. Writes c07_stream_independent.json.
"""
import ast
import json
import math

from g0common import *  # noqa: F401,F403

out = {}
sets, extras = load_trainval()
train, val = sets["train"], sets["val"]
C = legacy.C

# ---- 1. centring
ck = torch.load(PION / "new3_std_n3.pt", map_location="cpu", weights_only=False)["state"]
c0, s0 = shapelab.lnp_center_scale({"train": train})
tab = np.load(HERE / "events_table.npz")
lnp_all = np.log(tab["p"].astype(np.float32))


def cs(mask):
    v = lnp_all[mask]
    return float((v.max() + v.min()) / 2), float((v.max() - v.min()) / 2)


out["centring"] = {"recomputed_train": [c0, s0], "checkpoint": [float(ck["base.lnp0"]), float(ck["base.lnp_s"])],
                   "equal_float32": bool(np.float32(c0) == ck["base.lnp0"].numpy() and np.float32(s0) == ck["base.lnp_s"].numpy()),
                   "if_train+val": cs(tab["split"] <= 1), "if_all_events": cs(tab["split"] >= 0), "if_val_only": cs(tab["split"] == 1),
                   "same_buffers_in_all_13_ladder_checkpoints": all(
                       bool(torch.equal(torch.load(PION / f, map_location="cpu", weights_only=False)["state"][k], ck["base.lnp0"] if "lnp0" in k else ck["base.lnp_s"]))
                       for f, ks in [(f"new3_std_n{n}.pt", ("base.lnp0", "base.lnp_s")) for n in range(1, 7)]
                       + [(f"base_std_n{n}.pt", ("base.f.lnp0", "base.f.lnp_s")) for n in range(1, 7)] for k in ks)}

# ---- 2. v0, vs
from particle_shape import MASS, kinetic  # noqa: E402
lo, hi = [float(torch.log(kinetic(torch.tensor(float(c0) + sg * float(s0)), float(MASS["pion"])))) for sg in (-1.0, 1.0)]
out["v0_vs"] = {"recomputed": [float(torch.tensor((hi + lo) / 2)), float(torch.tensor((hi - lo) / 2))],
                "checkpoint": [float(ck["base.v0"]), float(ck["base.vs"])], "mass_buffer": float(ck["base.mass"]),
                "equal_float32": bool(torch.tensor((hi + lo) / 2) == ck["base.v0"] and torch.tensor((hi - lo) / 2) == ck["base.vs"])}

# ---- 3. formula_values.json["std"]
formula = legacy.load_component_formula(train)
core = formula.base
src = (paths.PROTON_DIR / "scripts" / "formula_card.py").read_text()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
ns = {"torch": torch, "np": np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "formula_card.evaluate", "exec"), ns)
abar = legacy.load_abar()
mine = ns["evaluate"](core, [0.5, 1.0, 2.0, 5.0, 10.0, 20.0], abar)
stored = jload(PION / "formula_values.json")["std"]
worst, rows = 0.0, {}
for k, v in stored.items():
    if isinstance(v[0], str):
        rows[k] = "identical strings" if v == mine[k] else "DIFFERENT"
        continue
    a, b = np.array(mine[k], dtype=np.float64), np.array(v, dtype=np.float64)
    rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))
    rows[k] = rel
    worst = max(worst, rel)
out["formula_values_std"] = {"n_rows": len(stored), "keys_equal": sorted(stored) == sorted(mine), "max_rel_diff": worst,
                             "rows_above_1e-4": {k: v for k, v in rows.items() if not isinstance(v, str) and v > 1e-4},
                             "string_rows": {k: v for k, v in rows.items() if isinstance(v, str)}}

# ---- 4. audit_v3.json (new_std)
lnp = torch.linspace(math.log(0.25), math.log(20.0), 48, device=DEV)
with torch.no_grad():
    lp = formula(lnp)
    eff = 0
    for p in formula.parameters():
        flat = p.data.view(-1)
        for i in range(flat.numel()):
            old = float(flat[i]); flat[i] = old + 0.05
            eff += float((formula(lnp) - lp).abs().max()) > 1e-7
            flat[i] = old
out["audit_new_std"] = {"recomputed": {"max_abs_sum_minus_1": float((lp.exp().sum((1, 2)) - 1).abs().max()), "finite": bool(torch.isfinite(lp).all()),
                                       "effective_params": int(eff), "claimed_params": int(formula.n_params())},
                        "stored": jload(PION / "audit_v3.json")["new_std"]}
formula = legacy.load_component_formula(train)        # fresh copy (the perturbation loop restores values only to float precision)
mlp = legacy.load_mlp_std()

# ---- 5. Abar
a_fb = legacy.mean_acceptance(train, front=True, back=True, nsub=4)
a_f = legacy.mean_acceptance(train, front=True, back=False, nsub=4)
a_fb2 = legacy.mean_acceptance(train, front=True, back=True, nsub=2)
a_val = legacy.mean_acceptance(val, front=True, back=True, nsub=4)
a_tv = (a_fb * len(train) + a_val * len(val)) / (len(train) + len(val))
d = lambda a, b: {"max_abs": float((a - b).abs().max()), "mean_abs": float((a - b).abs().mean())}   # noqa: E731
w0 = torch.load(M0 / "abar_front_back.pt").to(DEV)
w1 = torch.load(M0 / "abar_front.pt").to(DEV)
out["abar"] = {"train_front_back_nsub4_vs_stored": d(a_fb, abar), "bit_identical": bool(torch.equal(a_fb, abar)),
               "front_only_vs_stored": d(a_f, abar), "nsub2_vs_stored": d(a_fb2, abar),
               "train+val_vs_stored": d(a_tv, abar), "val_only_vs_stored": d(a_val, abar),
               "worker_saved_front_back_equals_mine": bool(torch.equal(w0, a_fb)), "worker_saved_front_equals_mine": bool(torch.equal(w1, a_f)),
               "stored_range": [float(abar.min()), float(abar.max())],
               "cells_le_1e-6 (floored in logA)": int((abar <= 1e-6).sum()), "cells_le_0.01": int((abar <= 0.01).sum()),
               "cells_le_0.3": int((abar <= 0.3).sum()), "cells_eq_1": int((abar == 1).sum()),
               "front_only_minus_front_back": {"max": float((a_f - a_fb).max()), "min": float((a_f - a_fb).min()),
                                               "cells_differing_by_>1e-3": int(((a_f - a_fb).abs() > 1e-3).sum()),
                                               "first_z_bin_with_diff_>1e-3 (z' lower edge, mm)": float(common.Z_EDGES[int(torch.nonzero(((a_f - a_fb).abs() > 1e-3).any(1))[0])])},
               "checkpoint_logA_equals_log_abar_clamped": bool(torch.equal(ck["logA"].to(DEV), torch.log(abar.clamp_min(1e-6))))}
# P-weighted effect of the template choice on the two benchmarks is not a Gate-0 quantity; only the template difference is.

# brute-force check of the closed-form azimuthal acceptance on random (event, z', r) triples
g = torch.Generator().manual_seed(0)
b = train.batch(torch.randint(0, len(train), (64,), generator=g).to(DEV))
A = C["event_slab"](b, train.box, 1, True, True)                         # nsub=1: cell midpoints (k = 0.5)
cos_t = b.R[:, 2, 2].double()
sin_t = torch.sqrt(1 - cos_t ** 2)
D = b.zb.double() * cos_t
half = float(train.box[2]) / 2
zs = (common.Z_EDGES[:-1] + common.DZ * 0.5).double()
rs = torch.sqrt(common.R_EDGES[:-1] ** 2 + (common.R_EDGES[1:] ** 2 - common.R_EDGES[:-1] ** 2) * 0.5).double()
phi = (torch.arange(20000, device=DEV, dtype=torch.float64) + 0.5) / 20000 * 2 * math.pi
worst = 0.0
for e in range(0, 64, 8):
    zt = zs[:, None, None] * cos_t[e] - rs[None, :, None] * sin_t[e] * torch.cos(phi)[None, None, :]     # z_trans of ring points
    inside = ((zt >= -half) & (zt <= D[e])).double().mean(-1)
    worst = max(worst, float((inside - A[e].double()).abs().max()))
out["abar"]["closed_form_vs_bruteforce_phi_max_abs_diff"] = worst

# ---- 6. buffers and device dependence of the maps
m_local = common.FieldModel(32, 4)
ckm = torch.load(PION / "mlp_std.pt", map_location="cpu", weights_only=False)["state"]
out["buffers"] = {"mlp_static_ckpt_vs_local_max_abs": float((ckm["static"] - m_local.static.cpu()).abs().max()),
                  "formula_ZZ_ckpt_vs_local": float((ck["base.ZZ"] - common.ZZ.cpu()).abs().max()),
                  "formula_RR_ckpt_vs_local": float((ck["base.RR"] - common.RR.cpu()).abs().max()),
                  "formula_LOG_VOL_ckpt_vs_local": float((ck["base.LOG_VOL"] - common.LOG_VOL.cpu()).abs().max())}
lnp = torch.log(torch.tensor([0.0895, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]))
with torch.no_grad():
    gm, gf = mlp(lnp.to(DEV)).cpu(), formula(lnp.to(DEV)).cpu()
    cm, cf = mlp.cpu()(lnp), formula.cpu()(lnp)
pm = lambda a, b: float(((a - b).abs() * a.exp()).sum((1, 2)).max())     # noqa: E731  probability-weighted |d logP|
out["maps_cpu_vs_gpu"] = {"mlp_max_abs_dlogP": float((gm - cm).abs().max()), "mlp_P_weighted_abs_dlogP": pm(gm, cm),
                          "formula_max_abs_dlogP": float((gf - cf).abs().max()), "formula_P_weighted_abs_dlogP": pm(gf, cf)}
(HERE / "c07_stream_independent.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps(out, indent=1, default=float))
