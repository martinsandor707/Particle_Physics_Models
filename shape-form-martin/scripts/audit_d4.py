"""M1 audit D4 - data vs each model on coarse (z', r) regions (validation).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d4.py

For each model: folded deposits ``common.sample_folded(model, val, M=400)`` (default seed 7; truncated at each event's
back face - cells whose lower z edge lies beyond zb are removed, straddling cells are kept whole - and smeared with
the rotated cell box, so their (z', r) distribution is comparable with hit centres). Regions: the coarse grid of D2
(by point coordinates) plus an "outside" bin for samples beyond [-200, 1450) x [0, 700) mm.
Per group (all; six momentum groups; theta terciles) and region:
    observed share  = mean over events of sum_i w_i [hit centre in region]
    expected share  = mean over events of (samples in region) / M
    diff = observed - expected;  SE = sd over events of the observed share / sqrt(n);  pull = diff / SE
(the per-event SE is the analytic equivalent of a bootstrap by event). Also reported: SE of the per-event difference
(includes the sampling noise of M = 400) and the Monte-Carlo SE of the expected share.
Writes d4.json and figures/d4_pulls.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402

common = L.common
M = 400


def main():
    run = L.Run("d4")
    sets, extras, mlp, formula = L.load()
    val = sets["val"]
    N = len(val)
    NREG = L.NZC * L.NRC
    z, r = L.hit_zr(val)
    reg_hit = L.coarse_index(z, r)
    ev_hit = L.hit_event(val)
    w = val.w.cpu().numpy().astype(np.float64)
    obs = np.bincount(ev_hit * (NREG + 1) + np.where(reg_hit < 0, NREG, reg_hit), weights=w,
                      minlength=N * (NREG + 1)).reshape(N, NREG + 1)
    exp = {}
    timings = {}
    import time
    for name, model in (("mlp", mlp), ("formula", formula)):
        t = time.time()
        s = common.sample_folded(model, val, M=M)
        timings[name] = round(time.time() - t, 1)
        assert len(s) == N * M
        reg_s = L.coarse_index(s["z"].to_numpy(), s["r"].to_numpy())
        ev_s = np.repeat(np.arange(N), M)
        # the sample rows are event-major (pts (B, M, 3).flatten()); check the momentum column
        assert np.allclose(s["p"].to_numpy()[::M], val.ev["p"].cpu().numpy())
        exp[name] = np.bincount(ev_s * (NREG + 1) + np.where(reg_s < 0, NREG, reg_s),
                                minlength=N * (NREG + 1)).reshape(N, NREG + 1) / M
        del s
    p = val.ev["p"].cpu().numpy().astype(np.float64)
    grp = L.p_group(p)
    tcls, tedges = L.quantile_bins(extras["val"]["theta"], 3)
    groups = [("all", np.ones(N, bool))] + [(f"p {lab} GeV", grp == g) for g, lab in enumerate(L.P_GROUP_LABELS)] \
        + [(f"theta T{i + 1} [{tedges[i]:.3f},{tedges[i + 1]:.3f}] rad", tcls == i) for i in range(3)]
    names = [L.region_name(k) for k in range(NREG)] + ["outside grid"]
    out = {"M": M, "seed": 7, "groups": {}, "region_names": names, "theta_tercile_edges_rad": tedges,
           "sampling_seconds": timings}
    for gname, m in groups:
        n = int(m.sum())
        o = obs[m]
        g = {"n_events": n, "observed": o.mean(0), "observed_se": o.std(0, ddof=1) / np.sqrt(n)}
        for name in ("mlp", "formula"):
            e = exp[name][m]
            diff = o.mean(0) - e.mean(0)
            se = g["observed_se"]
            with np.errstate(divide="ignore", invalid="ignore"):
                pull = np.where(se > 0, diff / se, np.nan)          # undefined where no event has energy there
            dse = (o - e).std(0, ddof=1) / np.sqrt(n)
            g[name] = {"expected": e.mean(0), "expected_mc_se": e.std(0, ddof=1) / np.sqrt(n), "diff": diff,
                       "pull": pull, "diff_se_paired": dse, "outside_grid_share": float(e[:, NREG].mean())}
            fin = np.where(np.isfinite(pull[:NREG]), np.abs(pull[:NREG]), -1.0)
            g[name]["regions_abs_pull_gt3"] = [
                {"region": names[k], "observed": float(o.mean(0)[k]), "expected": float(e.mean(0)[k]),
                 "diff": float(diff[k]), "se": float(se[k]), "pull": float(pull[k]),
                 "pull_with_paired_se": float(diff[k] / dse[k]) if dse[k] > 0 else None}
                for k in np.argsort(-fin) if fin[k] > 3]
            g[name]["regions_zero_observed_se"] = [
                {"region": names[k], "observed": float(o.mean(0)[k]), "expected": float(e.mean(0)[k]),
                 "expected_mc_se": float(g[name]["expected_mc_se"][k])}
                for k in range(NREG) if not se[k] > 0]
        both = sorted(set(x["region"] for x in g["mlp"]["regions_abs_pull_gt3"])
                      & set(x["region"] for x in g["formula"]["regions_abs_pull_gt3"]))
        g["regions_abs_pull_gt3_both_models"] = both
        out["groups"][gname] = g
    run.write(out)
    figure(out, NREG)
    for gname, g in out["groups"].items():
        print(gname, len(g["mlp"]["regions_abs_pull_gt3"]), len(g["formula"]["regions_abs_pull_gt3"]),
              len(g["regions_abs_pull_gt3_both_models"]))


def figure(out, NREG):
    plt = L.style()
    gnames = list(out["groups"])[:7]
    fig, axes = plt.subplots(2, 7, figsize=(15, 4.8), sharex=True, sharey=True)
    zl, rl = L.region_labels()
    for j, gname in enumerate(gnames):
        for i, model in enumerate(("mlp", "formula")):
            ax = axes[i, j]
            pull = np.clip(np.asarray(out["groups"][gname][model]["pull"][:NREG], float), -10, 10).reshape(L.NZC, L.NRC)
            im = ax.imshow(pull.T, origin="lower", cmap=L.diverging_cmap(), vmin=-10, vmax=10, aspect="auto")
            ax.grid(False)
            if i == 0:
                ax.set_title(gname.replace("p ", "p "), fontsize=8)
            ax.set_xticks(range(L.NZC))
            ax.set_xticklabels([f"{L.COARSE_Z[k]:g}" for k in range(L.NZC)], rotation=90, fontsize=6)
            ax.set_yticks(range(L.NRC))
            ax.set_yticklabels([f"{L.COARSE_R[k]:g}" for k in range(L.NRC)], fontsize=6)
        axes[0, 0].set_ylabel("MLP\nr lower edge [mm]", fontsize=8)
        axes[1, 0].set_ylabel("formula\nr lower edge [mm]", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("z' lower edge [mm]", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="pull (observed − expected)/SE, clipped ±10")
    fig.suptitle("D4 · energy-share pulls on coarse (z', r) regions, validation (positive = data above model)",
                 fontsize=10)
    fig.savefig(L.FIG / "d4_pulls.png", dpi=100)


if __name__ == "__main__":
    main()
