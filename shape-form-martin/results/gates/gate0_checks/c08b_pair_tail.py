"""Gate 0 check 3 (addendum to c08): the cross-machine difference of the PAIRED quantity formula - MLP,
D_e = (F - M)_local - (F - M)_reference, on test (saved arrays) against its validation analogue (local stream
minus emulated reference stream): distribution test, tails, multiplicity of the largest |D|.
Also the bulk-metric leverage of events with a small retained energy fraction (validation).
Writes c08b_pair_tail.json."""
import json

from scipy import stats as sps

from g0common import *  # noqa: F401,F403

out = {}
tab = np.load(HERE / "events_table.npz")
f64 = lambda a: np.asarray(a, dtype=np.float64)                 # noqa: E731
npz = np.load(PION / "test_v3_nll.npz")
Dt = (f64(np.load(M0 / "formula_test_nll.npy")) - f64(np.load(M0 / "mlp_test_nll.npy"))) - (f64(npz["std:new"]) - f64(npz["std:mlp_ft"]))
Dv = (f64(np.load(HERE / "val_grid144_new3_std_n3.npy")) - f64(np.load(HERE / "val_grid144_mlp_std.npy"))) \
    - (f64(np.load(HERE / "val_grid1056_new3_std_n3.npy")) - f64(np.load(HERE / "val_grid1056_mlp_std.npy")))
nt, nv = tab["n_hits"][tab["split"] == 2], tab["n_hits"][tab["split"] == 1]
ks = sps.ks_2samp(np.abs(Dt), np.abs(Dv))
q = lambda a: [float(x) for x in np.quantile(np.abs(a), [0.5, 0.9, 0.99, 0.999, 1.0])]      # noqa: E731


def trimmed_rms(a, k):
    return float(np.sqrt(np.sort(a ** 2)[:len(a) - k].mean()))


out["D_formula_minus_mlp"] = {
    "ks_abs_D_test_vs_val": {"D": float(ks.statistic), "p": float(ks.pvalue)},
    "abs_quantiles(50,90,99,99.9,100%)": {"test": q(Dt), "val": q(Dv)},
    "rms": {"test": float(np.sqrt((Dt ** 2).mean())), "val": float(np.sqrt((Dv ** 2).mean()))},
    "rms_without_20_largest": {"test": trimmed_rms(Dt, 20), "val": trimmed_rms(Dv, 20)},
    "share_of_sum_D2_from_20_largest": {"test": float(np.sort(Dt ** 2)[-20:].sum() / (Dt ** 2).sum()), "val": float(np.sort(Dv ** 2)[-20:].sum() / (Dv ** 2).sum())},
    "n_hits_of_10_largest_abs_D": {"test": nt[np.argsort(np.abs(Dt))[-10:]].tolist(), "val": nv[np.argsort(np.abs(Dv))[-10:]].tolist()},
    "largest_abs_D": {"test": np.sort(np.abs(Dt))[-5:].tolist(), "val": np.sort(np.abs(Dv))[-5:].tolist()},
    "rms_by_multiplicity(n<=5, 6-50, 51-300, >300)": {
        "test": [float(np.sqrt((Dt[m] ** 2).mean())) for m in (nt <= 5, (nt > 5) & (nt <= 50), (nt > 50) & (nt <= 300), nt > 300)],
        "val": [float(np.sqrt((Dv[m] ** 2).mean())) for m in (nv <= 5, (nv > 5) & (nv <= 50), (nv > 50) & (nv <= 300), nv > 300)]}}

# bulk metric: leverage of events with little retained energy (validation)
from sfm import metric as M  # noqa: E402
sets, extras = load_trainval()
val = sets["val"]
models = [legacy.load_mlp_std(), legacy.load_component_formula(sets["train"])]
with M.deterministic():
    blk = M.test_nll_multi(models, val, bulk=M.Bulk(), return_parts=True)
ret = blk["retained"].astype(np.float64)
d = blk["nll"][1].astype(np.float64) - blk["nll"][0]
zb = val.ev["zb"].cpu().numpy()
ok = np.isfinite(d)
full = zb >= 1150.0
full1221 = zb >= 1221.0
se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))            # noqa: E731
out["bulk_retained_energy"] = {
    "n_events": int(len(d)), "n_nan": int((~ok).sum()),
    "full_depth(zb>=1150)": {"n": int(full.sum()), "retained_min": float(ret[full].min()), "n_retained_lt_0.05": int((ret[full] < 0.05).sum()),
                             "n_retained_lt_0.5": int((ret[full] < 0.5).sum())},
    "F-M_bulk_by_retained(full depth)": {
        "W>=0.5": [float(d[full & (ret >= 0.5)].mean()), se(d[full & (ret >= 0.5)]), int((full & (ret >= 0.5)).sum())],
        "W<0.5": [float(d[full & (ret < 0.5)].mean()), se(d[full & (ret < 0.5)]), int((full & (ret < 0.5)).sum())]},
    "contribution_of_W<0.5_events_to_bulk_F-M_mean(full depth)": float(d[full & (ret < 0.5)].sum() / full.sum()),
    "sd_of_bulk_F-M_per_event": {"W>=0.5": float(d[full & (ret >= 0.5)].std()), "W<0.5": float(d[full & (ret < 0.5)].std())},
    "strata_sizes": {"zb<1150": int((~full).sum()), "1150<=zb<1221": int((full & ~full1221).sum()), "zb>=1221": int(full1221.sum())},
    "F-M_bulk(zb>=1221)": [float(d[full1221 & ok].mean()), se(d[full1221 & ok])],
    "F-M_bulk(1150<=zb<1221)": [float(d[full & ~full1221 & ok].mean()), se(d[full & ~full1221 & ok])]}
tr_zb = sets["train"].ev["zb"].cpu().numpy()
out["bulk_retained_energy"]["train_strata_sizes"] = {"zb<1150": int((tr_zb < 1150).sum()), "1150<=zb<1221": int(((tr_zb >= 1150) & (tr_zb < 1221)).sum()),
                                                     "zb>=1221": int((tr_zb >= 1221).sum())}
(HERE / "c08b_pair_tail.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
