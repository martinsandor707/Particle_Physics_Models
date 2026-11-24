"""Gate 0 checks 6 and 7: units / ranges and the low-momentum boundary. Training and validation data; for the test
split only the meta file and the Parquet-derived per-event table (momentum, hit count); the model maps are evaluated
at given momenta (data-free). Writes c12_units_lowp.json.
"""
import json
import math

import pyarrow as pa
import pyarrow.parquet as pq

from g0common import *  # noqa: F401,F403
from particle_shape import MASS, kinetic

out = {}
sets, extras = load_trainval()
meta = data.load_meta()
tab = np.load(HERE / "events_table.npz")
train, val = sets["train"], sets["val"]
mlp = legacy.load_mlp_std()
formula = legacy.load_component_formula(train)
core = formula.base
v0, vs = float(core.v0), float(core.vs)

# ---------------------------------------------------------------------------------------------------------- units
u = {"geometry_mm": meta["geometry"], "box_mm": train.box.cpu().tolist(), "mass_pion_GeV": {"particle_shape.MASS": MASS["pion"],
     "checkpoint_buffer": float(core.mass), "sfm.data.M_PION": data.M_PION},
     "grid": {"z_mm": [common.Z_MIN, common.Z_MAX, common.N_Z], "r_mm": [0.0, common.R_MAX, common.N_R], "DZ_mm": common.DZ,
              "r_edges_first3_mm": common.R_EDGES[:4].cpu().tolist(), "r_last_ring_inner_edge_mm": float(common.R_EDGES[-2])},
     "time_cut_ns": common.TIME_CUT_NS, "N_FOLD": [common.N_FOLD, common.N_FOLD_EVAL]}
for s in ("train", "val"):
    ds = sets[s]
    th, ph = extras[s]["theta"], extras[s]["phi"]
    R = ds.ev["R"].double()
    n = np.diff(ds.off.cpu().numpy())
    owner = torch.repeat_interleave(torch.arange(len(ds), device=DEV), ds.off[1:] - ds.off[:-1])
    wsum = torch.zeros(len(ds), dtype=torch.float64, device=DEV).index_add_(0, owner, ds.w.double())
    u[s] = {"p_GeV": [float(ds.ev["p"].min()), float(ds.ev["p"].max())],
            "theta_rad": [float(th.min()), float(th.max())], "theta_deg": [float(np.degrees(th.min())), float(np.degrees(th.max()))],
            "phi_rad": [float(ph.min()), float(ph.max())], "phi_over_pi": [float(ph.min() / np.pi), float(ph.max() / np.pi)],
            "zb_mm": [float(ds.ev["zb"].min()), float(ds.ev["zb"].max())],
            "zb_times_cos_theta_max_mm (= 59.5 layers?)": float((ds.ev["zb"].double().cpu().numpy() * np.cos(th)).max()),
            "R_orthonormal_max_dev": float((R @ R.transpose(1, 2) - torch.eye(3, dtype=torch.float64, device=DEV)).abs().max()),
            "R_det_range": [float(torch.linalg.det(R).min()), float(torch.linalg.det(R).max())],
            "R22_minus_cos_theta_max": float((R[:, 2, 2].cpu() - torch.tensor(np.cos(th))).abs().max()),
            "hits_per_event": [int(n.min()), float(np.median(n)), int(n.max())], "n_events_with_1_hit": int((n == 1).sum()),
            "weights_sum_max_dev_from_1": float((wsum - 1).abs().max()), "w_min": float(ds.w.min()),
            "z_local_mm": [float(ds.xyz[:, 2].min()), float(ds.xyz[:, 2].max())],
            "r_mm_max": float(torch.sqrt(ds.xyz[:, 0].double() ** 2 + ds.xyz[:, 1].double() ** 2).max()),
            "evis_over_p_median (sampling fraction if evis in GeV)": float((ds.ev["evis"] / ds.ev["p"]).median()),
            "lnp_equals_log_p": bool(torch.equal(ds.ev["lnp"], torch.log(ds.ev["p"]))),
            "non_finite": {k: int((~torch.isfinite(v)).sum()) for k, v in ds.ev.items() if v.is_floating_point()}}
u["test_from_meta"] = {k: meta["splits"]["test"][k] for k in ("p_min_GeV", "p_max_GeV", "theta_min_rad", "theta_max_rad", "zb_min_mm", "zb_max_mm",
                                                               "hits_per_event_min", "hits_per_event_median", "hits_per_event_max", "n_p_below_0.1GeV")}
# time of flight: earliest hit time vs distance of the calorimeter front from the origin (c = 299.79 mm/ns)
pf = pq.ParquetFile(str(paths.parquet_path()))
t0 = pf.read_row_group(0, columns=["event_number", "x", "y", "z", "x_local", "y_local", "z_local", "z_trans", "earliest_contribution_time",
                                   "incoming_theta", "incoming_phi", "energy"])
df = t0.to_pandas()
dist = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
u["time_of_flight"] = {"earliest_time_ns": float(df.earliest_contribution_time.min()), "z_min_mm": float(df.z.min()),
                       "min over hits of (t - |x|/c) [ns], c = 299.792 mm/ns": float((df.earliest_contribution_time - dist / 299.792).min()),
                       "fraction of hits with t < |x|/c - 0.05 ns": float((df.earliest_contribution_time < dist / 299.792 - 0.05).mean()),
                       "energy_min_max": [float(df.energy.min()), float(df.energy.max())]}
# rotation convention: local = R(theta, phi) (global - origin) with one origin per event?
sub = df[df.event_number.isin(df.event_number.unique()[:400])]
dev_max, zt_dev = 0.0, 0.0
for e, g in sub.groupby("event_number"):
    if len(g) < 3:
        continue
    Rm = common.rotation(np.array([g.incoming_theta.iloc[0]]), np.array([g.incoming_phi.iloc[0]]))[0].astype(np.float64)
    glob = g[["x", "y", "z"]].to_numpy()
    loc = g[["x_local", "y_local", "z_local"]].to_numpy()
    origin = glob - loc @ Rm                       # global = origin + R^T local  ->  origin = global - R^T local (row form)
    dev_max = max(dev_max, float(np.abs(origin - origin.mean(0)).max()))
    zt_dev = max(zt_dev, float(np.abs((g.z - g.z_trans).to_numpy() - origin[:, 2]).max()))
u["rotation_convention"] = {"events_checked": int(sub.event_number.nunique()),
                            "max_spread_of_reconstructed_origin_mm (0 if local = R (global - origin))": dev_max,
                            "max_abs(z - z_trans - origin_z)_mm": zt_dev}
out["units"] = u

# ---------------------------------------------------------------------------------------------------- low momentum
xof = lambda p: (np.log(data.kinetic(p)) - v0) / vs            # noqa: E731
lm = {"v0": v0, "vs": vs}
p_all, split, nh = tab["p"], tab["split"], tab["n_hits"]
for i, s in enumerate(("train", "val", "test")):
    m = split == i
    p = p_all[m]
    j = np.argsort(p)[:5]
    lm[s] = {"p_min_GeV": float(p.min()), "T_min_GeV": float(data.kinetic(p.min())), "x_min": float(xof(p.min())),
             "lnp_over_3_min (MLP input)": float(np.log(p.min()) / 3), "n_p_below_0.1": int((p < 0.1).sum()),
             "n_p_below_0.2": int((p < 0.2).sum()), "n_p_below_0.3": int((p < 0.3).sum()),
             "five_lowest": [{"event_number": int(tab["event_number"][m][k]), "p": float(p[k]), "n_hits": int(nh[m][k]), "x": float(xof(p[k]))} for k in j],
             "n_below_train_min": int((p < p_all[split == 0].min()).sum()), "n_above_train_max": int((p > p_all[split == 0].max()).sum())}
lm["test_p_min_from_meta"] = meta["splits"]["test"]["p_min_GeV"]
out["low_momentum"] = lm

# model maps at special momenta
P = {"lowest_val": lm["val"]["p_min_GeV"], "lowest_train": lm["train"]["p_min_GeV"], "lowest_test(meta)": meta["splits"]["test"]["p_min_GeV"],
     "0.05": 0.05, "0.02": 0.02, "0.005": 0.005, "0.0005": 0.0005, "1.0": 1.0}
zc = common.ZC.double()


@torch.no_grad()
def describe_map(model, p):
    lp = model(torch.tensor([math.log(p)], dtype=torch.float32, device=DEV))[0].double()
    Pm = lp.exp()
    Pz = Pm.sum(1)
    return lp, {"finite": bool(torch.isfinite(lp).all()), "sum": float(Pm.sum()), "abs_sum_minus_1": float((Pm.sum() - 1).abs()),
                "max_cell_prob": float(Pm.max()), "logP_min": float(lp.min()), "mean_depth_mm": float((Pz * zc).sum()),
                "mass_z_lt_0": float(Pz[zc < 0].sum()), "mass_z_gt_500": float(Pz[zc > 500].sum()),
                "mass_r_gt_200": float(Pm[:, common.RC > 200].sum()), "entropy_nat": float(-(Pm * lp).sum())}


maps, md = {}, {}
for name, model in (("mlp", mlp), ("formula", formula)):
    md[name] = {}
    for lab, p in P.items():
        lp, d = describe_map(model, p)
        maps[(name, lab)] = lp
        d["p_GeV"], d["x"] = p, float(xof(p))
        md[name][lab] = d
    kl = lambda a, b: float((maps[(name, a)].exp() * (maps[(name, a)] - maps[(name, b)])).sum())   # noqa: E731
    md[name]["KL(lowest_test || lowest_train)"] = kl("lowest_test(meta)", "lowest_train")
    md[name]["KL(lowest_train || lowest_val)"] = kl("lowest_train", "lowest_val")
    md[name]["KL(0.05 || lowest_train)"] = kl("0.05", "lowest_train")
md["KL(mlp || formula) at"] = {lab: float((maps[("mlp", lab)].exp() * (maps[("mlp", lab)] - maps[("formula", lab)])).sum()) for lab in P}
out["maps"] = md
# raw polynomial parameters of the formula at the boundary
with torch.no_grad():
    lnp = torch.log(torch.tensor([P["lowest_test(meta)"], P["lowest_train"], P["lowest_val"], 0.0005], dtype=torch.float32, device=DEV))
    th = core.params(lnp).cpu().numpy()
    rng_mm = core.range_mm(core.theta(lnp)).flatten().cpu().numpy()
out["formula_raw_params"] = {"all_finite": bool(np.isfinite(th).all()), "max_abs_at_lowest_test": float(np.abs(th[0]).max()),
                             "max_abs_change_lowest_test_vs_lowest_train": float(np.abs(th[0] - th[1]).max()),
                             "param_with_max_change": core.names[int(np.abs(th[0] - th[1]).argmax())],
                             "max_abs_at_0.0005": float(np.abs(th[3]).max()), "param_with_max_abs_at_0.0005": core.names[int(np.abs(th[3]).argmax())],
                             "range_mm(lowest_test, lowest_train, lowest_val, 0.0005)": rng_mm.tolist()}

# leverage: NLL of the lowest-momentum TRAINING events as a function of an assumed momentum (proxy for the one test event
# below the training range: same multiplicity class, 1-2 hits)
order = torch.argsort(train.ev["p"])[:6]
lev = []
gridp = [0.05, 0.06, lm["test"]["p_min_GeV"], 0.08, lm["train"]["p_min_GeV"], 0.1, 0.12, 0.15, 0.2]
for idx in order.tolist():
    one = train.subset([idx])
    row = {"event_number": int(extras["train"]["event_number"][idx]), "p": float(one.ev["p"][0]), "n_hits": int(one.off[-1]),
           "zb": float(one.ev["zb"][0]), "theta": float(extras["train"]["theta"][idx])}
    for name, model in (("mlp", mlp), ("formula", formula)):
        vals = []
        for p in gridp:
            one.ev["lnp"] = torch.tensor([math.log(p)], dtype=torch.float32, device=DEV)
            vals.append(float(common.test_nll(model, one)[0]))
        row[f"nll_{name}_vs_assumed_p"] = dict(zip([f"{p:.4f}" for p in gridp], vals))
    lev.append(row)
out["leverage_lowest_train_events"] = lev
# NLL of the lowest-p validation and training events at their true momentum vs the split mean
with torch.no_grad():
    for s in ("train", "val"):
        ds = sets[s]
        idx = torch.argsort(ds.ev["p"])[:200]
        sub = ds.subset(idx)
        a = {name: common.test_nll(model, sub).astype(np.float64) for name, model in (("mlp", mlp), ("formula", formula))}
        pp = sub.ev["p"].cpu().numpy()
        out[f"lowest_200_{s}"] = {"p_range": [float(pp.min()), float(pp.max())], "nll_mlp_mean": float(a["mlp"].mean()), "nll_formula_mean": float(a["formula"].mean()),
                                   "nll_mlp_max": float(a["mlp"].max()), "nll_formula_max": float(a["formula"].max()),
                                   "F-M_mean": float((a["formula"] - a["mlp"]).mean()), "all_finite": bool(np.isfinite(a["mlp"]).all() and np.isfinite(a["formula"]).all()),
                                   "lowest5": [{"p": float(pp[i]), "n_hits": int(sub.off[i + 1] - sub.off[i]), "mlp": float(a["mlp"][i]), "formula": float(a["formula"][i])} for i in range(5)]}
(HERE / "c12_units_lowp.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps(out, indent=1, default=float))
