"""M1 audit D1 - where the formula-MLP gap lives in event space (validation; two training events).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d1.py

d_e = NLL_formula - NLL_MLP per validation event (K=64, seed 12345, identical points; parts_val_k64.npz).
Strata: 1-GeV momentum bin, fine bins below 2 GeV, theta quintile, back-face strata (zb >= / < 1221 mm, zb > 1450 mm,
first-hit-layer index k), hit-multiplicity quintile; each with mean +- SE, count, contribution to the overall gap and
the leave-that-stratum-out gap. Low-momentum inspection (PLAN A5): validation events with p < 0.3 GeV and training
events with p < 0.1 GeV one by one. Writes results/m1_audit/d1.json and figures/d1_strata.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402


def stratum_rows(d, nll_m, nll_f, cls, labels, extra=None):
    N, S = len(d), d.sum()
    rows = []
    for c, lab in enumerate(labels):
        m = cls == c
        row = {"stratum": lab, "n": int(m.sum())}
        row["d"] = L.mean_se(d[m])
        row["contribution"] = float(d[m].sum() / N)            # share of the overall mean carried by the stratum
        row["share_of_gap"] = float(d[m].sum() / S)
        row["leave_out_gap"] = L.mean_se(d[~m])
        row["nll_mlp"] = float(nll_m[m].mean()) if m.any() else None
        row["nll_formula"] = float(nll_f[m].mean()) if m.any() else None
        if extra is not None:
            row.update(extra(m))
        rows.append(row)
    return rows


def main():
    run = L.Run("d1")
    sets, extras, _, _ = L.load()
    val, train = sets["val"], sets["train"]
    parts = L.parts_val()
    nll_m, nll_f = parts["nll"][0].astype(np.float64), parts["nll"][1].astype(np.float64)
    d = nll_f - nll_m
    N = len(d)
    p = val.ev["p"].cpu().numpy().astype(np.float64)
    pbin = val.ev["bin"].cpu().numpy()
    theta = extras["val"]["theta"]
    zb = val.ev["zb"].cpu().numpy().astype(np.float64)
    nh = L.n_hits(val)
    evis = val.ev["evis"].cpu().numpy().astype(np.float64)
    out = {"convention": "d_e = NLL_formula - NLL_MLP [nat/event], validation, K=64 seed 12345"}

    # ---- overall -------------------------------------------------------------------------------------------------------
    qs = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
    order = np.argsort(-np.abs(d))
    top = order[: int(round(0.01 * N))]
    mtop = np.zeros(N, bool)
    mtop[top] = True
    out["overall"] = {"gap": L.mean_se(d), "nll_mlp": L.mean_se(nll_m), "nll_formula": L.mean_se(nll_f),
                      "quantiles": {f"{q:g}": float(np.quantile(d, q)) for q in qs},
                      "sd": float(d.std(ddof=1)), "frac_positive": float((d > 0).mean()),
                      "top1pct_abs": {"n": int(len(top)), "abs_d_threshold": float(np.abs(d[top]).min()),
                                      "sum_d_over_N": float(d[top].sum() / N),
                                      "share_of_gap": float(d[top].sum() / d.sum()),
                                      "n_positive": int((d[top] > 0).sum()),
                                      "leave_out_gap": L.mean_se(d[~mtop]),
                                      "median_hits": float(np.median(nh[top])),
                                      "median_hits_all": float(np.median(nh))}}

    # ---- strata --------------------------------------------------------------------------------------------------------
    st = {}
    st["p_bin_1GeV"] = stratum_rows(d, nll_m, nll_f, pbin, [f"{b}-{b + 1} GeV" for b in range(20)])
    fine = np.array([p.min(), 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
    fcls = np.where(p < 2.0, np.clip(np.searchsorted(fine, p, side="right") - 1, 0, 5), -1)
    st["p_fine_below_2GeV"] = stratum_rows(d, nll_m, nll_f, fcls,
                                           [f"[{fine[i]:.4g},{fine[i + 1]:g}) GeV" for i in range(6)])
    tcls, tedges = L.quantile_bins(theta, 5)
    st["theta_quintile"] = stratum_rows(d, nll_m, nll_f, tcls,
                                        [f"Q{i + 1} [{tedges[i]:.3f},{tedges[i + 1]:.3f}] rad" for i in range(5)])
    st["zb_1221"] = stratum_rows(d, nll_m, nll_f, (zb < L.ZB_FULL).astype(int), ["zb >= 1221 mm", "zb < 1221 mm"])
    st["zb_1450"] = stratum_rows(d, nll_m, nll_f, (zb > 1450).astype(int), ["zb <= 1450 mm", "zb > 1450 mm"])
    k, kraw = L.first_layer_index(zb, theta)
    kedges = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 20), (21, 40), (41, 59)]
    kcls = np.full(N, -1)
    for i, (a, b) in enumerate(kedges):
        kcls[(k >= a) & (k <= b)] = i
    st["first_layer_k"] = stratum_rows(d, nll_m, nll_f, kcls,
                                       [f"k={a}" if a == b else f"k={a}-{b}" for a, b in kedges])
    hcls, hedges = L.quantile_bins(nh, 5)
    st["hits_quintile"] = stratum_rows(d, nll_m, nll_f, hcls,
                                       [f"Q{i + 1} [{hedges[i]:g},{hedges[i + 1]:g}] hits" for i in range(5)])
    out["strata"] = st
    out["notes"] = {"k_rounding_max_abs": float(np.abs(kraw - k).max()), "k_unassigned": int((kcls < 0).sum()),
                    "theta_quintile_edges_rad": tedges, "hits_quintile_edges": hedges,
                    "hits_quintile_counts": np.bincount(hcls, minlength=5),
                    "fine_bin_edges_GeV": fine}

    # ---- low-momentum inspection (PLAN A5) -----------------------------------------------------------------------------
    z_hit, r_hit = L.hit_zr(val)
    ev_hit = L.hit_event(val)
    w = val.w.cpu().numpy().astype(np.float64)

    def event_rows(idx, sh_z, sh_r, sh_ev, sh_w, nm, nf, ev_numbers, p_, th_, nh_, evis_):
        Nn, Sd, Sm, Sf = len(nm), (nf - nm).sum(), nm.sum(), nf.sum()
        rows = []
        for e in idx:
            h = sh_ev == e
            de = nf[e] - nm[e]
            rows.append({"event_number": int(ev_numbers[e]), "p_GeV": float(p_[e]), "theta_rad": float(th_[e]),
                         "n_hits": int(nh_[e]), "evis": float(evis_[e]),
                         "deepest_hit_z_mm": float(sh_z[h].max()), "largest_hit_r_mm": float(sh_r[h].max()),
                         "ew_mean_z_mm": float((sh_w[h] * sh_z[h]).sum()), "ew_mean_r_mm": float((sh_w[h] * sh_r[h]).sum()),
                         "nll_mlp": float(nm[e]), "nll_formula": float(nf[e]), "d_e": float(de),
                         "gap_change_if_left_out": float((Sd - de) / (Nn - 1) - Sd / Nn),
                         "mlp_mean_change_if_left_out": float((Sm - nm[e]) / (Nn - 1) - Sm / Nn),
                         "formula_mean_change_if_left_out": float((Sf - nf[e]) / (Nn - 1) - Sf / Nn)})
        return rows

    low = np.flatnonzero(p < 0.3)
    low = low[np.argsort(p[low])]
    mlow = np.zeros(N, bool)
    mlow[low] = True
    out["low_p_val"] = {"threshold_GeV": 0.3, "n": int(len(low)),
                        "events": event_rows(low, z_hit, r_hit, ev_hit, w, nll_m, nll_f, extras["val"]["event_number"],
                                             p, theta, nh, evis),
                        "subset_d": L.mean_se(d[mlow]), "leave_all_out_gap": L.mean_se(d[~mlow]),
                        "gap_change_leave_all_out": float(d[~mlow].mean() - d.mean()),
                        "mlp_mean_change_leave_all_out": float(nll_m[~mlow].mean() - nll_m.mean()),
                        "formula_mean_change_leave_all_out": float(nll_f[~mlow].mean() - nll_f.mean()),
                        "evis_median_all_val": float(np.median(evis)), "hits_median_all_val": float(np.median(nh))}

    tr = np.load(L.TRAIN_NLL_FILE)
    tm, tf = tr["nll"][0].astype(np.float64), tr["nll"][1].astype(np.float64)
    assert np.array_equal(tr["event_number"], extras["train"]["event_number"])
    pt = train.ev["p"].cpu().numpy().astype(np.float64)
    low_t = np.flatnonzero(pt < 0.1)
    low_t = low_t[np.argsort(pt[low_t])]
    # hit info only for those events (avoid moving all training hits)
    sub = train.subset(low_t)
    zs, rs = L.hit_zr(sub)
    evs = L.hit_event(sub)
    ws = sub.w.cpu().numpy().astype(np.float64)
    nh_t = L.n_hits(train)
    evis_t = train.ev["evis"].cpu().numpy().astype(np.float64)
    rows = []
    Nn, Sd, Sm, Sf = len(tm), (tf - tm).sum(), tm.sum(), tf.sum()
    for j, e in enumerate(low_t):
        h = evs == j
        de = tf[e] - tm[e]
        rows.append({"event_number": int(extras["train"]["event_number"][e]), "p_GeV": float(pt[e]),
                     "theta_rad": float(extras["train"]["theta"][e]), "n_hits": int(nh_t[e]), "evis": float(evis_t[e]),
                     "deepest_hit_z_mm": float(zs[h].max()), "largest_hit_r_mm": float(rs[h].max()),
                     "ew_mean_z_mm": float((ws[h] * zs[h]).sum()), "ew_mean_r_mm": float((ws[h] * rs[h]).sum()),
                     "nll_mlp": float(tm[e]), "nll_formula": float(tf[e]), "d_e": float(de),
                     "gap_change_if_left_out": float((Sd - de) / (Nn - 1) - Sd / Nn),
                     "mlp_mean_change_if_left_out": float((Sm - tm[e]) / (Nn - 1) - Sm / Nn),
                     "formula_mean_change_if_left_out": float((Sf - tf[e]) / (Nn - 1) - Sf / Nn)})
    out["low_p_train"] = {"threshold_GeV": 0.1, "n": int(len(low_t)), "events": rows,
                          "train_gap": L.mean_se(tf - tm), "train_nll_mlp": float(tm.mean()),
                          "train_nll_formula": float(tf.mean()),
                          "evis_median_all_train": float(np.median(evis_t)), "hits_median_all_train": float(np.median(nh_t))}
    run.write(out)
    figure(st)


def figure(st):
    plt = L.style()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), gridspec_kw={"width_ratios": [3.2, 1.6, 1.4, 1.4]})
    for ax, key, xl in zip(axes, ("p_bin_1GeV", "p_fine_below_2GeV", "theta_quintile", "hits_quintile"),
                           ("momentum bin [GeV]", "p below 2 GeV", "θ quintile", "hit-multiplicity quintile")):
        rows = st[key]
        x = np.arange(len(rows))
        m = np.array([r["d"]["mean"] for r in rows]) * 1e3
        s = np.array([r["d"]["se"] for r in rows]) * 1e3
        ax.axhline(0, color=L.COL["ink2"], lw=0.8)
        ax.errorbar(x, m, s, fmt="o", ms=4, color=L.COL["formula"], ecolor=L.COL["formula"], capsize=2, lw=1.2)
        ax.set_xticks(x)
        if key == "p_bin_1GeV":
            ax.set_xticklabels([str(i) for i in range(20)], fontsize=7)
        elif key == "p_fine_below_2GeV":
            ax.set_xticklabels([r["stratum"].split(")")[0].strip("[").replace(",", "–") for r in rows], fontsize=6,
                               rotation=30)
        else:
            ax.set_xticklabels([r["stratum"].split()[0] for r in rows], fontsize=7)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("mean d_e ± SE  [10⁻³ nat]")
    fig.suptitle("D1 · d_e = NLL(formula) − NLL(MLP) by stratum, validation (positive = formula worse)", fontsize=10)
    fig.tight_layout()
    fig.savefig(L.FIG / "d1_strata.png", dpi=110)


if __name__ == "__main__":
    main()
