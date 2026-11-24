"""M1 audit D6a - dependence on the number of fold points K (validation).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d6a.py

Both benchmarks on identical points for K in {16, 64, 256, 1024} and fold seeds {12345, 1, 2}
(sfm.metric.test_nll_multi, chunk=64, batch 128 for K <= 256 and batch 32 for K = 1024 - the batch changes the points,
not the estimator; deterministic sums). Reports mean NLL per model, gap +- SE per K, seed-to-seed scatter of the
means, sfm.metric.jensen_fit (powers (1,) and (1, 2); fold-noise errors) for each model and for the gap, and the
table NLL(K) - NLL(1024). Per-event arrays: d6a_kscan.npz (git-ignored). Writes d6a.json, figures/d6a_kscan.png.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402

KS = (16, 64, 256, 1024)
SEEDS = (12345, 1, 2)
NAMES = ("mlp", "formula")


def fit_dict(f):
    return {"powers": f.powers, "value_inf": f.value_inf, "value_inf_se": f.value_inf_se, "coef": f.coef,
            "coef_se": f.coef_se, "chi2": f.chi2, "dof": f.dof, "mean": f.mean, "mean_se_fold": f.mean_se,
            "bias_at_64": f.bias(64), "bias_at_64_se": f.bias_se(64), "bias_at_16": f.bias(16),
            "bias_at_1024": f.bias(1024)}


def main():
    run = L.Run("d6a")
    sets, extras, mlp, formula = L.load()
    val = sets["val"]
    N = len(val)
    cache = L.OUT / "d6a_kscan.npz"
    arr = np.empty((2, len(KS), len(SEEDS), N), np.float64)
    secs = {}
    if cache.exists():
        with np.load(cache) as f:
            arr = f["nll"].astype(np.float64)
            secs = {k: float(v) for k, v in zip(f["sec_keys"], f["sec_vals"])}
    else:
        for i, K in enumerate(KS):
            for j, seed in enumerate(SEEDS):
                L.sync()
                t = time.time()
                arr[:, i, j] = L.metric.test_nll_multi([mlp, formula], val, K, seed, 128 if K < 1024 else 32,
                                                       chunk=64, deterministic_sum=True)
                L.sync()
                secs[f"K{K}_seed{seed}"] = time.time() - t
                print(f"K={K} seed={seed}: {arr[0, i, j].mean():.6f} {arr[1, i, j].mean():.6f} ({secs[f'K{K}_seed{seed}']:.1f} s)",
                      flush=True)
        np.savez(cache, nll=arr.astype(np.float32), Ks=np.array(KS), seeds=np.array(SEEDS),
                 sec_keys=np.array(list(secs)), sec_vals=np.array(list(secs.values())))
    parts = L.parts_val()
    ref_check = [float(np.abs(arr[m, 1, 0] - parts["nll"][m]).max()) for m in range(2)]
    gap = arr[1] - arr[0]                                                 # (K, seed, N)
    out = {"Ks": KS, "seeds": SEEDS, "batch": {str(K): 128 if K < 1024 else 32 for K in KS}, "chunk": 64,
           "K64_seed12345_vs_parts_file_max_abs": ref_check, "seconds": secs}
    perK = []
    for i, K in enumerate(KS):
        row = {"K": K}
        for m, n in enumerate(NAMES):
            means = arr[m, i].mean(1)
            row[n] = {"mean_over_seeds": float(means.mean()), "per_seed": means,
                      "seed_sd_of_mean": float(means.std(ddof=1))}
        gm = gap[i].mean(1)
        row["gap"] = {"per_seed": [L.mean_se(gap[i, j]) for j in range(len(SEEDS))],
                      "seed_avg_events": L.mean_se(gap[i].mean(0)), "seed_sd_of_mean": float(gm.std(ddof=1))}
        perK.append(row)
    out["per_K"] = perK
    # NLL(K) - NLL(1024), seed-averaged per event; SE over events (includes fold noise and event scatter)
    tab = []
    for i, K in enumerate(KS):
        r = {"K": K}
        for m, n in enumerate(NAMES):
            r[n] = L.mean_se((arr[m, i] - arr[m, -1]).mean(0)) if K != 1024 else {"mean": 0.0, "se": 0.0}
        r["gap"] = L.mean_se((gap[i] - gap[-1]).mean(0)) if K != 1024 else {"mean": 0.0, "se": 0.0}
        tab.append(r)
    out["minus_K1024"] = tab
    fits = {}
    for powers in ((1.0,), (1.0, 2.0)):
        key = "p" + "_".join(str(int(p)) for p in powers)
        fits[key] = {n: fit_dict(L.metric.jensen_fit(arr[m], KS, powers)) for m, n in enumerate(NAMES)}
        fits[key]["gap"] = fit_dict(L.metric.jensen_fit(gap, KS, powers))
    out["jensen_fit"] = fits
    out["notes"] = {"jensen_fit errors": "fold noise only (events fixed), as in sfm.metric.jensen_fit; the event-sampling "
                                         "SE of a single model's mean is ~0.012 and of the gap ~2.9e-4",
                    "minus_K1024 SE": "SE over events of the seed-averaged per-event difference"}
    run.write(out)
    figure(perK, fits)
    for r in tab:
        print(r)


def figure(perK, fits):
    plt = L.style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.array([1 / r["K"] for r in perK])
    xx = np.linspace(0, x.max() * 1.05, 100)
    for n in NAMES:
        y = np.array([r[n]["mean_over_seeds"] for r in perK])
        axes[0].plot(x, y, "o", ms=5, color=L.COL[n], label=n)
        f = fits["p1"][n]
        axes[0].plot(xx, f["value_inf"] + f["coef"][0] * xx, "-", lw=1, color=L.COL[n])
    axes[0].set_xlabel("1/K")
    axes[0].set_ylabel("mean NLL (3 seeds) [nat]")
    axes[0].legend(fontsize=7)
    axes[0].set_title("NLL(K) with fit v∞ + c/K", fontsize=9)
    g = np.array([r["gap"]["seed_avg_events"]["mean"] for r in perK]) * 1e3
    s = np.array([r["gap"]["seed_avg_events"]["se"] for r in perK]) * 1e3
    axes[1].errorbar(x, g, s, fmt="o", ms=5, color=L.COL["formula"], capsize=3)
    f = fits["p1"]["gap"]
    axes[1].plot(xx, (f["value_inf"] + f["coef"][0] * xx) * 1e3, "-", lw=1, color=L.COL["formula"])
    axes[1].set_xlabel("1/K")
    axes[1].set_ylabel("gap formula − MLP [10⁻³ nat] (± event SE)")
    axes[1].set_title("gap vs 1/K", fontsize=9)
    fig.suptitle("D6a · dependence on the number of fold points, validation", fontsize=10)
    fig.tight_layout()
    fig.savefig(L.FIG / "d6a_kscan.png", dpi=110)


if __name__ == "__main__":
    main()
