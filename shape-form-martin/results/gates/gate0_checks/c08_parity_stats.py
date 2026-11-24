"""Gate 0 check 3 (statistical part): is the cross-machine per-event difference on TEST what a change of fold points
alone produces?

Null distributions come from VALIDATION, where everything can be recomputed here:
  * seeds:  common.test_nll's loop with CUDA generator seeds 1..8 on this machine (re-run here, 4 models);
  * cross:  local stream (launch grid 144) minus the emulated reference stream (grid 1056), same models - the exact
            validation analogue of "this machine minus the reference machine".
TEST uses only the saved arrays: results/m0_parity/{mlp,formula}_test_nll.npy (local stream) and test_v3_nll.npz
(reference machine). Per-event momentum bin / hit multiplicity of test events come from events_table.npz (derived
from the Parquet file by c05; the test cache is not opened, no model is evaluated on test events here).
Writes c08_parity_stats.json.
"""
import itertools
import json

from scipy import stats as sps

from g0common import *  # noqa: F401,F403

rng = np.random.default_rng(0)
out = {}
sets, extras = load_trainval()
val, train = sets["val"], sets["train"]
models = {"mlp": legacy.load_mlp_std(), "formula": new3_std(3, train), "new_n2": new3_std(2, train), "base_acc": base_std(5, train)}
fname = {"mlp": "mlp_std", "formula": "new3_std_n3", "new_n2": "new3_std_n2", "base_acc": "base_std_n5"}
f64 = lambda a: np.asarray(a, dtype=np.float64)                         # noqa: E731


def boot_rms(d, n=400):
    idx = rng.integers(0, len(d), size=(n, len(d)))
    return float(np.sqrt((d[idx] ** 2).mean(1)).std(ddof=1))


def desc(a, b, boot=True):
    d = f64(a) - f64(b)
    ad = np.abs(d)
    r = {"n": int(len(d)), "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
         "rms": float(np.sqrt((d ** 2).mean())), "median_abs": float(np.median(ad)),
         "q90_abs": float(np.quantile(ad, 0.9)), "q99_abs": float(np.quantile(ad, 0.99)), "q999_abs": float(np.quantile(ad, 0.999)),
         "max_abs": float(ad.max()), "pearson_r": float(np.corrcoef(f64(a), f64(b))[0, 1])}
    r["z_mean"] = r["mean"] / r["se"]
    if boot:
        r["rms_boot_se"] = boot_rms(d)
    return r


# ---------------------------------------------------------------------------------------------------------------- val
@torch.no_grad()
def nll_seed(model, ds, seed, K=64):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.cat([common.event_nll(model, b, ds.box, K, gen)[0] for b in ds.batches(128)]).cpu().numpy()


seeds = list(range(1, 9))
S = {k: {s: nll_seed(m, val, s) for s in seeds} for k, m in models.items()}
loc = {k: np.load(HERE / f"val_grid144_{fname[k]}.npy") for k in models}           # local stream (== common.test_nll here)
ref = {k: np.load(HERE / f"val_grid1056_{fname[k]}.npy") for k in models}          # emulated reference stream
now = {k: common.test_nll(m, val) for k, m in models.items()}
out["val_local_stream_checks"] = {
    k: {"max_abs(common.test_nll now - emulated grid144)": float(np.abs(f64(now[k]) - loc[k]).max())} for k in models}
for k, f in (("mlp", "mlp_val_nll.npy"), ("formula", "formula_val_nll.npy")):
    w = np.load(M0 / f)
    out["val_local_stream_checks"][k]["max_abs(worker m0 array - common.test_nll now)"] = float(np.abs(f64(w) - now[k]).max())
    out["val_local_stream_checks"][k]["mean_worker_m0"] = f64mean(w)

v = {}
for k in models:
    means = np.array([f64mean(S[k][s]) for s in seeds])
    pairs = [desc(S[k][a], S[k][b], boot=False) for a, b in itertools.combinations(seeds, 2)]
    ev_sd = np.std(np.stack([f64(S[k][s]) for s in seeds]), axis=0, ddof=1)
    v[k] = {"seed_means": means.tolist(), "sd_of_mean_across_seeds": float(means.std(ddof=1)), "mean_of_seed_means": float(means.mean()),
            "per_event_fold_sd": {"mean": float(ev_sd.mean()), "rms": float(np.sqrt((ev_sd ** 2).mean())), "median": float(np.median(ev_sd))},
            "predicted_sd_of_mean_from_per_event_sd": float(np.sqrt((ev_sd ** 2).mean() / len(ev_sd))),
            "seed_pairs": {q: [float(np.min([p[q] for p in pairs])), float(np.mean([p[q] for p in pairs])), float(np.max([p[q] for p in pairs]))]
                           for q in ("rms", "median_abs", "q99_abs", "max_abs", "pearson_r", "mean")},
            "local_minus_reference_stream": desc(loc[k], ref[k]),
            "local_stream_mean": f64mean(loc[k]), "reference_stream_mean": f64mean(ref[k]),
            "z_of_local_stream_mean_in_seed_distribution": float((f64mean(loc[k]) - means.mean()) / means.std(ddof=1)),
            "z_of_reference_stream_mean_in_seed_distribution": float((f64mean(ref[k]) - means.mean()) / means.std(ddof=1))}
out["val"] = v


def pair_stats(A, B, a="formula", b="mlp"):
    """A, B: dicts of arrays on two streams. D = (a-b)_A - (a-b)_B."""
    dA, dB = f64(A[a]) - f64(A[b]), f64(B[a]) - f64(B[b])
    D = dA - dB
    da, db = f64(A[a]) - f64(B[a]), f64(A[b]) - f64(B[b])
    return {"delta_stream_A": [float(dA.mean()), float(dA.std(ddof=1) / np.sqrt(len(dA)))],
            "delta_stream_B": [float(dB.mean()), float(dB.std(ddof=1) / np.sqrt(len(dB)))],
            "D_mean": float(D.mean()), "D_se": float(D.std(ddof=1) / np.sqrt(len(D))), "D_z": float(D.mean() / (D.std(ddof=1) / np.sqrt(len(D)))),
            "D_rms": float(np.sqrt((D ** 2).mean())), "D_rms_boot_se": boot_rms(D), "D_median_abs": float(np.median(np.abs(D))),
            "corr(d_a, d_b)": float(np.corrcoef(da, db)[0, 1]),
            "rms_d_a": float(np.sqrt((da ** 2).mean())), "rms_d_b": float(np.sqrt((db ** 2).mean())),
            "noise_reduction_rms_D_over_rms_d_a": float(np.sqrt((D ** 2).mean()) / np.sqrt((da ** 2).mean()))}


prs = [("formula", "mlp"), ("new_n2", "mlp"), ("formula", "new_n2"), ("base_acc", "mlp"), ("formula", "base_acc")]
out["val_pairs_local_vs_reference_stream"] = {f"{a}-{b}": pair_stats(loc, ref, a, b) for a, b in prs}
sp = {}
for a, b in prs:
    dm = np.array([(f64(S[a][s]) - f64(S[b][s])).mean() for s in seeds])
    rr = [pair_stats({a: S[a][s1], b: S[b][s1]}, {a: S[a][s2], b: S[b][s2]}, a, b) for s1, s2 in itertools.combinations(seeds, 2)]
    sp[f"{a}-{b}"] = {"delta_by_seed": dm.tolist(), "sd_of_delta_across_seeds": float(dm.std(ddof=1)),
                      "D_rms_pairs_min_mean_max": [float(np.min([r["D_rms"] for r in rr])), float(np.mean([r["D_rms"] for r in rr])), float(np.max([r["D_rms"] for r in rr]))],
                      "corr_pairs_mean": float(np.mean([r["corr(d_a, d_b)"] for r in rr]))}
out["val_pairs_seed_to_seed"] = sp

# ---------------------------------------------------------------------------------------------------------------- test
tab = np.load(HERE / "events_table.npz")
ten = np.load(M0 / "test_event_number.npy")
tm = tab["split"] == 2
assert np.array_equal(tab["event_number"][tm], ten)                    # same events, same order as the saved test arrays
vm = tab["split"] == 1
assert np.array_equal(tab["event_number"][vm], extras["val"]["event_number"])
meta_t = {q: tab[q][tm] for q in ("bin", "n_hits", "p")}
meta_v = {q: tab[q][vm] for q in ("bin", "n_hits", "p")}
npz = np.load(PION / "test_v3_nll.npz")
tl = {"mlp": np.load(M0 / "mlp_test_nll.npy"), "formula": np.load(M0 / "formula_test_nll.npy")}
tr = {"mlp": npz["std:mlp_ft"], "formula": npz["std:new"]}
t = {}
for k in ("mlp", "formula"):
    d_t = f64(tl[k]) - f64(tr[k])
    d_v = f64(loc[k]) - f64(ref[k])
    r = {"local_minus_reference_machine": desc(tl[k], tr[k]), "val_analogue": desc(loc[k], ref[k], boot=False),
         "mean_local": f64mean(tl[k]), "mean_reference": f64mean(tr[k])}
    ks = sps.ks_2samp(np.abs(d_t), np.abs(d_v))
    r["ks_abs_d_test_vs_val_cross"] = {"D": float(ks.statistic), "p": float(ks.pvalue)}
    ks = sps.ks_2samp(d_t, d_v)
    r["ks_signed_d_test_vs_val_cross"] = {"D": float(ks.statistic), "p": float(ks.pvalue)}
    r["rms_ratio_test_over_val_cross"] = r["local_minus_reference_machine"]["rms"] / r["val_analogue"]["rms"]
    r["median_abs_ratio_test_over_val_cross"] = r["local_minus_reference_machine"]["median_abs"] / r["val_analogue"]["median_abs"]
    # per momentum bin
    pb = []
    for b in range(20):
        a_t, a_v = d_t[meta_t["bin"] == b], d_v[meta_v["bin"] == b]
        pb.append({"bin": b, "n_test": int(len(a_t)), "rms_test": float(np.sqrt((a_t ** 2).mean())), "rms_val": float(np.sqrt((a_v ** 2).mean())),
                   "med_test": float(np.median(np.abs(a_t))), "med_val": float(np.median(np.abs(a_v))),
                   "mean_test": float(a_t.mean()), "z_mean_test": float(a_t.mean() / (a_t.std(ddof=1) / np.sqrt(len(a_t))))})
    r["per_bin"] = pb
    r["per_bin_median_abs_ratio_range"] = [float(min(x["med_test"] / x["med_val"] for x in pb)), float(max(x["med_test"] / x["med_val"] for x in pb))]
    r["per_bin_chi2_of_means (20 dof)"] = float(sum(x["z_mean_test"] ** 2 for x in pb))
    # vs hit multiplicity
    edges = [1, 6, 21, 51, 101, 201, 301, 451, 2000]
    hm = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a_t = d_t[(meta_t["n_hits"] >= lo) & (meta_t["n_hits"] < hi)]
        a_v = d_v[(meta_v["n_hits"] >= lo) & (meta_v["n_hits"] < hi)]
        hm.append({"n_hits": [lo, hi - 1], "n_test": int(len(a_t)), "n_val": int(len(a_v)),
                   "rms_test": float(np.sqrt((a_t ** 2).mean())), "rms_val": float(np.sqrt((a_v ** 2).mean())),
                   "med_test": float(np.median(np.abs(a_t))), "med_val": float(np.median(np.abs(a_v)))})
    r["vs_hit_multiplicity"] = hm
    # alignment
    x, y = f64(tl[k]), f64(tr[k])
    r["alignment"] = {"r_by_lag": {int(L): float(np.corrcoef(x[max(0, L):len(x) + min(0, L)], y[max(0, -L):len(y) + min(0, -L)])[0, 1]) for L in range(-3, 4)},
                      "r_random_permutation": float(np.corrcoef(x, rng.permutation(y))[0, 1]),
                      "min_r_over_30_consecutive_blocks": float(min(np.corrcoef(a, b)[0, 1] for a, b in zip(np.array_split(x, 30), np.array_split(y, 30)))),
                      "n_abs_d_gt_0.2": int((np.abs(d_t) > 0.2).sum()), "n_abs_d_gt_0.5": int((np.abs(d_t) > 0.5).sum()), "n_abs_d_gt_1": int((np.abs(d_t) > 1).sum()),
                      "val_cross_n_abs_d_gt_0.2": int((np.abs(d_v) > 0.2).sum()), "val_cross_n_abs_d_gt_0.5": int((np.abs(d_v) > 0.5).sum()),
                      "sd_of_per_event_nll": float(x.std()),
                      "rms_if_shifted_by_one_event": float(np.sqrt(((x[1:] - y[:-1]) ** 2).mean()))}
    # which stored array does the local array match?
    r["rms_vs_each_stored_std_array"] = {q: float(np.sqrt(((x - f64(npz[q])) ** 2).mean())) for q in npz.files if q.startswith("std:")}
    t[k] = r
out["test"] = t
out["test_pair_formula_minus_mlp"] = pair_stats(tl, tr, "formula", "mlp")
out["test_pair_reference_numbers"] = {"stored_json_new_minus_mlp_std": jload(PION / "test_v3.json")["new_minus_mlp"]["std"]}
# does any other stored model pair look like the local pair?  (identification of the model pair)
x = f64(tl["formula"]) - f64(tl["mlp"])
out["test_pair_identification_rms(local F-M minus stored pair)"] = {
    f"{a}-{b}": float(np.sqrt(((x - (f64(npz[a]) - f64(npz[b]))) ** 2).mean()))
    for a, b in [("std:new", "std:mlp_ft"), ("std:new_n2", "std:mlp_ft"), ("std:new", "std:report_mlp"), ("std:base_acc", "std:mlp_ft"), ("std:new_n2", "std:report_mlp")]}
(HERE / "c08_parity_stats.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


def short(o, depth=0):
    return json.dumps(o, indent=1, default=float)


for k in ("val_local_stream_checks",):
    print(k, short(out[k]))
for k in models:
    x = out["val"][k]
    print(f"VAL {k}: seed sd of mean {x['sd_of_mean_across_seeds']:.2e} (pred {x['predicted_sd_of_mean_from_per_event_sd']:.2e}); seed pairs rms {x['seed_pairs']['rms']}, r {x['seed_pairs']['pearson_r']}, max {x['seed_pairs']['max_abs']}")
    y = x["local_minus_reference_stream"]
    print(f"     local-ref stream: mean {y['mean']:+.2e} +- {y['se']:.2e} (z {y['z_mean']:+.2f}) rms {y['rms']:.5f}+-{y['rms_boot_se']:.5f} med {y['median_abs']:.5f} q99 {y['q99_abs']:.4f} max {y['max_abs']:.3f} r {y['pearson_r']:.6f}; z(ref mean in seeds) {x['z_of_reference_stream_mean_in_seed_distribution']:+.2f}, z(local) {x['z_of_local_stream_mean_in_seed_distribution']:+.2f}")
for k in ("mlp", "formula"):
    x = out["test"][k]
    y = x["local_minus_reference_machine"]
    print(f"TEST {k}: local-ref machine: mean {y['mean']:+.2e} +- {y['se']:.2e} (z {y['z_mean']:+.2f}) rms {y['rms']:.5f}+-{y['rms_boot_se']:.5f} med {y['median_abs']:.5f} q99 {y['q99_abs']:.4f} max {y['max_abs']:.3f} r {y['pearson_r']:.6f}")
    print(f"     KS |d| test vs val-cross: {x['ks_abs_d_test_vs_val_cross']}; signed: {x['ks_signed_d_test_vs_val_cross']}; rms ratio {x['rms_ratio_test_over_val_cross']:.3f}; median ratio {x['median_abs_ratio_test_over_val_cross']:.3f}")
    print(f"     per-bin median ratio range {x['per_bin_median_abs_ratio_range']}; chi2 of bin means {x['per_bin_chi2_of_means (20 dof)']:.1f}")
    print("     multiplicity:", [(h['n_hits'], round(h['med_test'], 4), round(h['med_val'], 4), round(h['rms_test'], 4), round(h['rms_val'], 4)) for h in x["vs_hit_multiplicity"]])
    print("     alignment:", x["alignment"])
    print("     rms vs stored arrays:", x["rms_vs_each_stored_std_array"])
print("VAL pairs local vs ref stream:", short(out["val_pairs_local_vs_reference_stream"]))
print("VAL pairs seed-to-seed:", short(out["val_pairs_seed_to_seed"]))
print("TEST pair F-M:", short(out["test_pair_formula_minus_mlp"]))
print(short(out["test_pair_identification_rms(local F-M minus stored pair)"]))
