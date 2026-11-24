"""Gate 0 check 5 (numerical side of the review of sfm.stats / sfm.metric). Validation data only.

  a. one_se_select against a literal transcription of the rule in common.select_architecture (float32 arithmetic,
     idxmin, Delta < SE, sort by [params, val NLL]) on random candidate sets, and against the two stored ladders.
  b. paired == shapelab.paired; analytic SE vs event bootstrap (plain and stratified by momentum bin); lag
     autocorrelation of the per-event differences (independence).
  c. stratified fold points on real hits: unbiased for the cell average, smaller variance, lower Jensen bias.
  d. low-momentum test events in the SAVED test arrays (leverage of the one event below the training range).
Writes c14_stats_review.json.
"""
import json

from g0common import *  # noqa: F401,F403
from sfm import metric as M
from sfm import stats as ST

out = {}
rng = np.random.default_rng(1)


# ---- a. 1-SE rule ---------------------------------------------------------------------------------------------
def rule_common(vals, params):
    """common.select_architecture, lines 846-851, without pandas: vals = list of float32 per-event arrays."""
    mean = [float(v.mean()) for v in vals]                                   # float32 mean, as float(val.mean())
    k_star = int(np.argmin(mean))                                            # idxmin: first minimum
    d = [v - vals[k_star] for v in vals]                                     # float32 differences
    delta = [float(x.mean()) for x in d]
    se = [float(x.std(ddof=1) / np.sqrt(len(x))) if k != k_star else 0.0 for k, x in enumerate(d)]
    ok = [k for k in range(len(vals)) if k == k_star or delta[k] < se[k]]
    return min(ok, key=lambda k: (params[k], mean[k], k)), k_star           # sort_values(["params", "val NLL"]).index[0]


agree = agree_best = trials = 0
for t in range(3000):
    m, n = int(rng.integers(2, 8)), 1500
    base = rng.normal(18.0, 1.4, n)
    gaps = rng.normal(0, rng.choice([1e-4, 1e-3, 1e-2]), m)
    vals = [(base + g + rng.normal(0, rng.choice([0.01, 0.05]), n)).astype(np.float32) for g in gaps]
    params = [int(x) for x in rng.permutation(m) * 10 + 5]
    sel_c, best_c = rule_common(vals, params)
    r = ST.one_se_select([ST.Candidate(f"c{k}", vals[k], k=params[k]) for k in range(m)])
    agree += r.selected == f"c{sel_c}"
    agree_best += r.best == f"c{best_c}"
    trials += 1
out["one_se_vs_common_rule_random"] = {"trials": trials, "same_selection": int(agree), "same_best": int(agree_best)}

lad = {}
for G in (1056, 144):
    for name in ("new3_std", "base_std"):
        j = jload(PION / f"{name}.json")
        vals = [np.load(HERE / f"val_grid{G}_{name}_n{n}.npy") for n in range(1, 7)]
        params = [j["rungs"][str(n)]["params"] for n in range(1, 7)]
        r = ST.one_se_select([ST.Candidate(f"n{n}", vals[n - 1], k=params[n - 1]) for n in range(1, 7)])
        sel_c, best_c = rule_common(vals, params)
        lad[f"{name}@grid{G}"] = {"sfm_selected": r.selected, "sfm_best": r.best, "common_rule_selected": f"n{sel_c + 1}",
                                  "stored_chosen": f"n{j['chosen']}", "stored_n_star": f"n{j['n_star']}",
                                  "table": [{k: (round(v, 9) if isinstance(v, float) else v) for k, v in row.items()} for row in r.table]}
out["one_se_on_stored_ladders"] = lad

# ---- b. paired SE ------------------------------------------------------------------------------------------------
sets, extras = load_trainval()
val = sets["val"]
a, b = np.load(HERE / "val_grid144_new3_std_n3.npy"), np.load(HERE / "val_grid144_mlp_std.npy")
p = ST.paired(a, b)
sp = shapelab.paired(a, b)
bins = val.ev["bin"].cpu().numpy()
bo = ST.bootstrap_paired(a, b, n_boot=4000, seed=0)
bs = ST.bootstrap_paired(a, b, n_boot=4000, seed=0, strata=bins)
d = a.astype(np.float64) - b
dc = d - d.mean()
ac = {L: float((dc[L:] * dc[:-L]).mean() / dc.var()) for L in (1, 2, 5, 10)}
# sd of the difference within momentum bins (heteroscedasticity) and the effect of a few events
out["paired"] = {"sfm": [p.mean, p.se, p.n], "shapelab": list(sp), "identical": bool(p.mean == sp[0] and p.se == sp[1]),
                 "bootstrap_se": bo.se, "bootstrap_ci95": [bo.lo, bo.hi], "stratified_bootstrap_se": bs.se,
                 "se_ratio_bootstrap_over_analytic": bo.se / p.se, "se_ratio_stratified_over_analytic": bs.se / p.se,
                 "lag_autocorrelation_of_d": ac, "autocorr_noise_level_1_over_sqrt_n": float(1 / np.sqrt(len(d))),
                 "excess_kurtosis_of_d": float(((dc / dc.std()) ** 4).mean() - 3),
                 "share_of_sum_d2_from_top_10_events": float(np.sort(dc ** 2)[-10:].sum() / (dc ** 2).sum()),
                 "mean_without_top10_abs_events": float(np.delete(d, np.argsort(np.abs(dc))[-10:]).mean())}

# ---- c. stratified fold points on real hits -------------------------------------------------------------------------
mlp = legacy.load_mlp_std()
bat = next(iter(val.batches(128)))
res = {}
with torch.no_grad():
    logP = mlp(bat.lnp)
    for fold in ("iid", "stratified"):
        for K in (16, 64):
            s1 = torch.zeros(len(bat.w), dtype=torch.float64, device=DEV)
            s2 = torch.zeros_like(s1)
            nl = []
            S = 300
            for seed in range(S):
                gen = torch.Generator(device=DEV).manual_seed(1000 + seed)
                nll, _, parts = M.event_nll(logP, bat, val.box, K, gen, fold=fold, return_parts=True)
                g = (parts.hit.double() / bat.w.double()).exp()            # ghat_i
                s1 += g
                s2 += g * g
                nl.append(float(nll.double().mean()))
            mean = s1 / S
            var = (s2 / S - mean ** 2).clamp_min(0) * S / (S - 1)
            res[(fold, K)] = (mean, var, float(np.mean(nl)), float(np.std(nl, ddof=1) / np.sqrt(S)))
    c = {}
    for K in (16, 64):
        mi, vi, ni, ei = res[("iid", K)]
        ms, vs_, ns, es = res[("stratified", K)]
        se = torch.sqrt((vi + vs_) / 300)
        ok = se > 0
        z = ((ms - mi) / se)[ok]
        c[f"K={K}"] = {"hits": int(ok.sum()), "z_mean (0 if both unbiased for the same g)": float(z.mean()), "z_sd (1 expected)": float(z.std()),
                       "weighted_rel_diff_of_mean_ghat": float((bat.w.double() * (ms - mi) / mi).sum() / bat.w.double().sum()),
                       "variance_ratio_strat_over_iid (weighted by w)": float((bat.w.double() * vs_).sum() / (bat.w.double() * vi).sum()),
                       "batch_mean_nll_iid": [ni, ei], "batch_mean_nll_stratified": [ns, es], "nll_strat_minus_iid": ns - ni}
    c["jensen_check_iid: NLL(K=16) - NLL(K=64)"] = res[("iid", 16)][2] - res[("iid", 64)][2]
    c["jensen_check_strat: NLL(K=16) - NLL(K=64)"] = res[("stratified", 16)][2] - res[("stratified", 64)][2]
out["stratified_fold_real_hits"] = c

# ---- d. the low-momentum test events in the saved arrays ----------------------------------------------------------
ten = np.load(M0 / "test_event_number.npy")
tl = {"mlp": np.load(M0 / "mlp_test_nll.npy").astype(np.float64), "formula": np.load(M0 / "formula_test_nll.npy").astype(np.float64)}
npz = np.load(PION / "test_v3_nll.npz")
rows = []
for ev in (181812, 46398, 87408):
    i = int(np.flatnonzero(ten == ev)[0])
    rows.append({"event_number": ev, "index": i, "mlp_local": tl["mlp"][i], "formula_local": tl["formula"][i],
                 "mlp_ref": float(npz["std:mlp_ft"][i]), "formula_ref": float(npz["std:new"][i]),
                 "F-M_local": tl["formula"][i] - tl["mlp"][i],
                 "contribution_to_test_F-M_mean": (tl["formula"][i] - tl["mlp"][i]) / len(ten),
                 "contribution_to_mlp_mean_vs_average_event": (tl["mlp"][i] - tl["mlp"].mean()) / len(ten)})
dF = tl["formula"] - tl["mlp"]
i = int(np.flatnonzero(ten == 181812)[0])
out["test_low_momentum_events_saved_arrays"] = {"events": rows, "test_F-M_all": [float(dF.mean()), float(dF.std(ddof=1) / np.sqrt(len(dF)))],
                                                "test_F-M_without_event_below_train_range": float(np.delete(dF, i).mean()),
                                                "all_finite": bool(np.isfinite(tl["mlp"]).all() and np.isfinite(tl["formula"]).all()),
                                                "max_nll": [float(tl["mlp"].max()), float(tl["formula"].max())], "min_nll": [float(tl["mlp"].min()), float(tl["formula"].min())]}
(HERE / "c14_stats_review.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
o2 = json.loads(json.dumps(out, default=float))
for k in o2["one_se_on_stored_ladders"].values():
    k.pop("table")
print(json.dumps(o2, indent=1))
