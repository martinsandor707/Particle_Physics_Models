"""Gate 0 check 5 (data part): the back-face distance zb and the bulk-restricted metric. Training and validation only.

  * distribution of zb and of the remaining depth D = zb cos(theta) = z_back - z0 (in layers);
  * events with zb < 1150 mm ("short"): how many, where in momentum / angle / multiplicity / visible energy,
    energy behind the flat cut z' > zb, energy in front of the tilted entrance plane;
  * the standard metric on them: normaliser M_e (clamp 1e-8?), model mass beyond zb, per-event NLL, leverage on the
    mean and on the formula - MLP difference (validation, local stream);
  * the bulk metric (sfm.metric.Bulk, z_cut 1150, last ring dropped) on validation: retained energy, NaN events,
    formula - MLP under clip / full-depth only, against the report metric on the same events.
Writes c10_zb_bulk.json.
"""
import json

from g0common import *  # noqa: F401,F403
from sfm import metric as M
from sfm import stats as ST

sets, extras = load_trainval()
out = {"z_cut": 1150.0, "geometry": data.load_meta()["geometry"]}
models = {"mlp": legacy.load_mlp_std(), "formula": legacy.load_component_formula(sets["train"])}
LAYER = 20.5


def q(a, qs=(0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)):
    return [float(x) for x in np.quantile(np.asarray(a, dtype=np.float64), qs)]


for s in ("train", "val"):
    ds = sets[s]
    zb = ds.ev["zb"].double().cpu().numpy()
    th = extras[s]["theta"]
    p = ds.ev["p"].double().cpu().numpy()
    n = np.diff(ds.off.cpu().numpy())
    evis = ds.ev["evis"].double().cpu().numpy()
    D = zb * np.cos(th)                                           # z_back - z0: depth left behind the first hit layer
    D0 = D.max()
    k = np.rint((D0 - D) / LAYER).astype(int)                     # first hit layer index (0 = the usual one)
    short = zb < 1150.0
    r = {"events": int(len(zb)), "zb_quantiles(0,1,5,25,50,75,95,99,100%)": q(zb), "n_zb_lt_0": int((zb < 0).sum()),
         "n_zb_lt_1150": int(short.sum()), "frac_zb_lt_1150": float(short.mean()),
         "n_zb_lt_1221": int((zb < 1221).sum()), "n_zb_gt_1450(beyond grid)": int((zb > 1450).sum()),
         "depth_left_max_mm": float(D0), "depth_left_in_layers_max": float(D0 / LAYER),
         "depth_left_is_multiple_of_layer_max_dev_mm": float(np.abs(D0 - D - k * LAYER).max()),
         "first_hit_layer_k": {"k=0": int((k == 0).sum()), "k=1": int((k == 1).sum()), "k=2": int((k == 2).sum()),
                               "k=3..5": int(((k >= 3) & (k <= 5)).sum()), "k=6..20": int(((k >= 6) & (k <= 20)).sum()),
                               "k=21..40": int(((k >= 21) & (k <= 40)).sum()), "k>40": int((k > 40).sum()), "k_max": int(k.max())},
         "zb_of_k0_events_range": [float(zb[k == 0].min()), float(zb[k == 0].max())],
         "smallest_k_with_zb_lt_1150": int(k[short].min()) if short.any() else None,
         "n_k_ge_1": int((k >= 1).sum())}
    # hits behind the flat cut and in front of the entrance plane
    owner = torch.repeat_interleave(torch.arange(len(ds), device=DEV), ds.off[1:] - ds.off[:-1])
    zh = ds.xyz[:, 2]
    w = ds.w.double()
    seg = lambda v: torch.zeros(len(ds), dtype=torch.float64, device=DEV).index_add_(0, owner, v).cpu().numpy()   # noqa: E731
    e_behind = seg(w * (zh > ds.ev["zb"][owner]))                 # energy fraction with hit centre z' > zb
    e_neg = seg(w * (zh < 0))
    zmean = seg(w * zh.double())
    r["energy_fraction_with_hit_centre_behind_zb"] = {"all_mean": float(e_behind.mean()), "short_mean": float(e_behind[short].mean()),
                                                      "full_mean": float(e_behind[~short].mean()), "short_max": float(e_behind[short].max()),
                                                      "n_events_with_any": int((e_behind > 0).sum()), "n_short_with_any": int((e_behind[short] > 0).sum())}
    sh = {"p_quantiles": q(p[short]), "p_quantiles_all": q(p), "theta_quantiles": q(th[short]), "theta_quantiles_all": q(th),
          "n_hits_quantiles": q(n[short]), "n_hits_quantiles_all": q(n), "evis_over_p_median": float(np.median(evis[short] / p[short])),
          "evis_over_p_median_all": float(np.median(evis / p)), "zb_quantiles": q(zb[short]),
          "mean_depth_zmean_quantiles": q(zmean[short]), "frac_n_hits_le_5": float((n[short] <= 5).mean()),
          "frac_n_hits_le_5_all": float((n <= 5).mean()), "hits_share_of_split": float(n[short].sum() / n.sum()),
          "evis_share_of_split": float(evis[short].sum() / evis.sum()),
          "by_momentum_bin(short/all)": {int(b): [int((short & (ds.ev['bin'].cpu().numpy() == b)).sum()), int((ds.ev['bin'].cpu().numpy() == b).sum())] for b in range(20)},
          "by_theta_tercile": {}}
    te = np.quantile(th, [0, 1 / 3, 2 / 3, 1])
    for i in range(3):
        m_ = (th >= te[i]) & (th <= te[i + 1])
        sh["by_theta_tercile"][f"{te[i]:.3f}-{te[i + 1]:.3f}"] = [int((short & m_).sum()), int(m_.sum())]
    sh["k_quantiles"] = q(k[short])
    r["short_events"] = sh

    # standard metric: normaliser and masses
    mm = {}
    with torch.no_grad():
        for name, m in models.items():
            ins, bey = [], []
            for b in ds.batches(512):
                mon = M.mass_monitors(m(b.lnp), b)
                ins.append(mon["inside_zb"]); bey.append(mon["beyond_zb"])
            ins, bey = torch.cat(ins).cpu().numpy(), torch.cat(bey).cpu().numpy()
            mm[name] = {"M_e_min": float(ins.min()), "M_e_quantiles": q(ins), "M_e_short_quantiles": q(ins[short]),
                        "n_M_e_below_1e-8(clamp active)": int((ins < 1e-8).sum()), "n_M_e_below_1e-3": int((ins < 1e-3).sum()),
                        "n_M_e_below_0.05": int((ins < 0.05).sum()),
                        "mass_beyond_zb_mean_full": float(bey[~short].mean()), "mass_beyond_zb_mean_short": float(bey[short].mean()),
                        "mass_beyond_zb_max": float(bey.max())}
    r["standard_metric_normaliser"] = mm
    out[s] = r
    if s == "val":
        keep = {"short": short, "k": k, "zb": zb, "n": n, "p": p}

# ---- validation: per-event NLLs, leverage of short events, bulk metric
val = sets["val"]
short = keep["short"]
with M.deterministic():
    std = M.test_nll_multi(list(models.values()), val).astype(np.float64)                 # == common.test_nll per model
    blk = M.test_nll_multi(list(models.values()), val, bulk=M.Bulk(), return_parts=True)
nll_b = blk["nll"].astype(np.float64)
ret = blk["retained"].astype(np.float64)
nan = np.isnan(nll_b[0])
d_std = std[1] - std[0]                                                                    # formula - mlp, report metric
d_blk = nll_b[1] - nll_b[0]
pm = lambda x: [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]                   # noqa: E731
v = {"report_metric": {"mlp_mean": float(std[0].mean()), "formula_mean": float(std[1].mean()),
                       "nll_mean_short": [float(std[0][short].mean()), float(std[1][short].mean())],
                       "nll_mean_full": [float(std[0][~short].mean()), float(std[1][~short].mean())],
                       "nll_sd_short": float(std[0][short].std()), "nll_sd_full": float(std[0][~short].std()),
                       "F-M_all": pm(d_std), "F-M_full_depth_only": pm(d_std[~short]), "F-M_short_only": pm(d_std[short]),
                       "share_of_F-M_gap_from_short_events": float(d_std[short].sum() / d_std.sum()),
                       "shift_of_mlp_mean_if_short_removed": float(std[0][~short].mean() - std[0].mean())},
     "bulk_metric(z_cut=1150, last ring dropped, short='clip')": {
         "n_nan(no retained energy)": int(nan.sum()), "nan_events_are_short": bool(short[nan].all()) if nan.any() else None,
         "nan_event_details": [{"zb": float(keep["zb"][i]), "n_hits": int(keep["n"][i]), "p": float(keep["p"][i])} for i in np.flatnonzero(nan)],
         "retained_energy_mean": float(ret.mean()), "retained_quantiles": q(ret), "retained_mean_full": float(ret[~short].mean()),
         "retained_mean_short": float(ret[short].mean()), "n_retained_lt_0.5": int((ret < 0.5).sum()), "n_retained_lt_0.5_full": int((ret[~short] < 0.5).sum()),
         "F-M_bulk_all_finite(clip)": pm(d_blk[~nan]), "F-M_bulk_full_depth_only": pm(d_blk[~short & ~nan]),
         "F-M_bulk_short_only": pm(d_blk[short & ~nan]),
         "F-M_report_same_events_all_finite": pm(d_std[~nan]), "F-M_report_full_depth_only": pm(d_std[~short & ~nan]),
         "n_full_depth_nan": int((nan & ~short).sum())}}
g_all = ST.guard_persistence(d_std[~nan], d_blk[~nan])
g_full = ST.guard_persistence(d_std[~short & ~nan], d_blk[~short & ~nan])
v["guard_persistence_of_the_F-M_gap(sign flipped: formula is worse)"] = {
    "note": "illustration with MLP as 'candidate' and formula as 'reference' (gain = MLP - formula < 0)"}
g_all = ST.guard_persistence(-d_std[~nan], -d_blk[~nan])
g_full = ST.guard_persistence(-d_std[~short & ~nan], -d_blk[~short & ~nan])
for nm, g in (("clip_all_events", g_all), ("full_depth_only", g_full)):
    v["guard_persistence_of_the_F-M_gap(sign flipped: formula is worse)"][nm] = {"passed": g.passed, **{kk: float(vv) for kk, vv in g.numbers.items()}, **g.flags}
out["val_metrics"] = v
(HERE / "c10_zb_bulk.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps(out, indent=1, default=float))
