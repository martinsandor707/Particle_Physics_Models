"""M1 audit D9 - roughness / ringing of the MLP map (no data used beyond the training momentum quantiles).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d9.py

At the 12 momenta of D3 (event quantiles of the training momenta):
  * RMS second difference of ln P along z' (the quantity inside common.fit's smoothness penalty:
    mean((lnP[z+1] - 2 lnP[z] + lnP[z-1])^2) over all z and r; reported as its square root and as the mean square) and
    along the ring index, for the MLP and the formula. Extra, labelled: the same restricted to z'-bin centres in
    [20, 1150] mm and rings 0-30; the formula core without ln A_bar; the three local MLPs mlp_L3xH{16,32,64}.
  * for rings 0, 4, 8, 16: power spectrum along z' of Delta = ln P_MLP - ln P_formula on the z'-bin centres of a window
    (as specified: -100 < z' < 1150 mm, 97 bins; extra, labelled: 20 < z' < 1150 mm, 88 bins), linear trend removed,
    Hann window, zero-padded to 4096 points; the strongest local maxima with period <= window length (longer periods
    are not resolved), the power share at longer periods, and the spectral power at the MLP's z-feature periods.
MLP z-features (common.FieldModel): sin/cos(pi k zn), zn = 2 (z' + 200)/1650 - 1, k = 1..4  ->  periods 1650/k mm
= 1650, 825, 550, 412.5 mm. PLAN.md section 4 lists 3300, 1650, 1100, 825 mm; both lists are evaluated.
Writes d9.json and figures/d9_ringing.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402

common = L.common
RINGS = (0, 4, 8, 16)
NFFT = 4096
WINDOWS = {"spec: -100 < z' < 1150 mm": (-100.0, 1150.0), "extra: 20 < z' < 1150 mm": (20.0, 1150.0)}


def rough(lp, zsel=None, rsel=None):
    """(rms along z, rms along the ring index, mean square along z) per map, lp (n, N_Z, N_R); second differences
    whose three cells all lie in the selection (default: everything)."""
    nz, nr = lp.shape[1:]
    zsel = np.ones(nz, bool) if zsel is None else zsel
    rsel = np.ones(nr, bool) if rsel is None else rsel
    d2z = lp[:, 2:] - 2 * lp[:, 1:-1] + lp[:, :-2]
    d2r = lp[:, :, 2:] - 2 * lp[:, :, 1:-1] + lp[:, :, :-2]
    mz = (zsel[2:] & zsel[1:-1] & zsel[:-2])[:, None] & rsel[None, :]
    mr = zsel[:, None] & (rsel[2:] & rsel[1:-1] & rsel[:-2])[None, :]
    msz = (d2z[:, mz] ** 2).mean(1)
    return np.sqrt(msz), np.sqrt((d2r[:, mr] ** 2).mean(1)), msz


def spectrum(y, dz):
    n = len(y)
    x = np.arange(n)
    y = y - np.polyval(np.polyfit(x, y, 1), x)
    win = np.hanning(n)
    F = np.fft.rfft(y * win, NFFT)
    P = np.abs(F) ** 2 / (win ** 2).sum()
    f = np.fft.rfftfreq(NFFT, dz)
    return f, P, float(np.sqrt((y ** 2).mean()))


def peaks(f, P, fmin, n=3):
    """Strongest local maxima with period <= window length (frequency >= fmin)."""
    i = np.flatnonzero((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])) + 1
    i = i[f[i] >= fmin]
    i = i[np.argsort(-P[i])][:n]
    ref = P[f >= fmin].max()
    return [{"period_mm": float(1 / f[k]), "freq_per_mm": float(f[k]), "power_rel_to_max_resolved": float(P[k] / ref)}
            for k in i]


def low_frac(f, P, fmin):
    """Share of the spectral power (f > 0) at periods longer than the window."""
    m = f > 0
    return float(P[m & (f < fmin)].sum() / P[m].sum())


def power_at(f, P, period, fmin):
    """Spectral power at a period, relative to the largest power at resolved periods (<= window length)."""
    return float(np.interp(1.0 / period, f, P) / P[f >= fmin].max())


def main():
    run = L.Run("d9")
    sets, extras, mlp, formula = L.load()
    p12 = L.momenta12(sets["train"])
    lm, lf = L.logP_at(mlp, p12), L.logP_at(formula, p12)
    lcore = L.logP_at(formula.base, p12)
    lcore = lcore - np.log(np.exp(lcore).sum((1, 2), keepdims=True))                 # normalised core map
    others = {}
    for H in (16, 32, 64):
        m, _ = L.load_local_mlp(H)
        others[f"L3xH{H}"] = L.logP_at(m, p12)
    zc = common.ZC.cpu().numpy()
    zsel_r = (zc >= 20) & (zc <= 1150)
    rsel_r = np.arange(common.N_R) <= 30
    rows = {}
    for name, lp in (("mlp_std", lm), ("formula", lf), ("formula_core_without_lnAbar (extra)", lcore),
                     *[(f"{k} (extra)", v) for k, v in others.items()]):
        rz, rr, msz = rough(lp)
        qz, qr, qmsz = rough(lp, zsel_r, rsel_r)
        rows[name] = {"rms_d2_z": rz, "rms_d2_ring": rr, "ms_d2_z": msz, "rms_d2_z_mean": float(rz.mean()),
                      "rms_d2_ring_mean": float(rr.mean()), "ms_d2_z_mean": float(msz.mean()),
                      "restricted_extra": {"rms_d2_z": qz, "rms_d2_ring": qr, "rms_d2_z_mean": float(qz.mean()),
                                           "rms_d2_ring_mean": float(qr.mean()), "ms_d2_z_mean": float(qmsz.mean())}}
    dz = common.DZ
    feat_actual = [1650.0 / k for k in range(1, 5)]
    feat_plan = list(L.MLP_Z_FEATURE_PERIODS_PLAN)
    delta = lm - lf
    spectra, mean_spec, window_info = {}, {}, {}
    for wname, (z0, z1) in WINDOWS.items():
        sel = (zc > z0) & (zc < z1)
        fmin = 1.0 / (sel.sum() * dz)
        window_info[wname] = {"z_bin_centres_mm": [float(zc[sel][0]), float(zc[sel][-1])], "n_bins": int(sel.sum()),
                              "length_mm": float(sel.sum() * dz), "frequency_resolution_per_mm": float(fmin)}
        spec = {}
        for ring in RINGS:
            per_p, Ps = [], []
            for i, p in enumerate(p12):
                f, P, rms = spectrum(delta[i, sel, ring], dz)
                Ps.append(P / P[f >= fmin].max())
                per_p.append({"p_GeV": float(p), "rms_detrended_delta_nat": rms, "peaks": peaks(f, P, fmin),
                              "power_share_periods_longer_than_window": low_frac(f, P, fmin),
                              "power_at_actual_feature_periods": {f"{q:g}": power_at(f, P, q, fmin) for q in feat_actual},
                              "power_at_plan_feature_periods": {f"{q:g}": power_at(f, P, q, fmin) for q in feat_plan}})
            Pm = np.mean(Ps, 0)
            mean_spec[(wname, ring)] = (f, Pm, fmin)
            spec[f"ring_{ring}"] = {
                "ring_r_range_mm": [float(common.R_EDGES[ring]), float(common.R_EDGES[ring + 1])],
                "rms_detrended_delta_nat_mean": float(np.mean([x["rms_detrended_delta_nat"] for x in per_p])),
                "peaks_of_momentum_averaged_spectrum": peaks(f, Pm, fmin, 4),
                "power_share_periods_longer_than_window_avg_spectrum": low_frac(f, Pm, fmin),
                "power_at_actual_feature_periods_avg_spectrum": {f"{q:g}": power_at(f, Pm, q, fmin) for q in feat_actual},
                "power_at_plan_feature_periods_avg_spectrum": {f"{q:g}": power_at(f, Pm, q, fmin) for q in feat_plan},
                "per_momentum": per_p}
        spectra[wname] = spec
    # orientation for the window choice: the maps at z' < 0 on the inner rings
    neg = zc < 0
    inner_neg = {"mlp_lnP_min_max": [float(lm[:, neg][:, :, :5].min()), float(lm[:, neg][:, :, :5].max())],
                 "formula_lnP_min_max": [float(lf[:, neg][:, :, :5].min()), float(lf[:, neg][:, :, :5].max())],
                 "formula_lnAbar_floor": float(formula.logA.min())}
    out = {"momenta_GeV": p12, "roughness": rows, "spectra": spectra, "windows": window_info,
           "nfft": NFFT, "dz_mm": dz, "inner_rings_z_lt_0": inner_neg,
           "mlp_feature_periods_mm": {"from_common_FieldModel": feat_actual, "as_listed_in_PLAN": feat_plan},
           "notes": {"roughness": "ms_d2_z is the per-map mean square entering common.fit's penalty term "
                                  "(smooth * mean(...)); rms = its square root. restricted_extra: second differences "
                                  "with all three cells in z'-bin centres [20, 1150] mm and rings 0-30",
                     "formula": "formula = WithAcceptance(core, A_bar): its ln P includes ln A_bar (template with "
                                "front and back faces, floored at 1e-6)",
                     "window": "the specified window includes z' < 0, where on the inner rings both maps are almost "
                               "empty (see inner_rings_z_lt_0); Delta there is a step of tens of nat at z' ~ 0"}}
    run.write(out)
    figure(delta, zc, p12, mean_spec, feat_actual, feat_plan)
    for k, v in rows.items():
        print(f"{k:40s} rms_d2_z {v['rms_d2_z_mean']:.4f} ring {v['rms_d2_ring_mean']:.4f} | restricted "
              f"{v['restricted_extra']['rms_d2_z_mean']:.4f} {v['restricted_extra']['rms_d2_ring_mean']:.4f}")
    for wname, spec in spectra.items():
        for k, v in spec.items():
            print(wname, k, round(v["rms_detrended_delta_nat_mean"], 4),
                  round(v["power_share_periods_longer_than_window_avg_spectrum"], 3),
                  [(round(x["period_mm"]), round(x["power_rel_to_max_resolved"], 2))
                   for x in v["peaks_of_momentum_averaged_spectrum"]])


def figure(delta, zc, p12, mean_spec, feat_actual, feat_plan):
    plt = L.style()
    import matplotlib as mpl
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#9ec5f4", "#2a78d6", "#0d366b"])
    wspec, wextra = list(WINDOWS)
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    sel = (zc > WINDOWS[wextra][0]) & (zc < WINDOWS[wextra][1])
    for j, ring in enumerate(RINGS):
        ax = axes[0, j]
        for i in range(len(p12)):
            ax.plot(zc[sel], delta[i, sel, ring], lw=1, color=cmap(i / 11))
        ax.set_title(f"ring {ring}: ln P(MLP) − ln P(formula), 20<z'<1150", fontsize=8)
        ax.set_xlabel("z' [mm]")
        for row, wname in ((1, wspec), (2, wextra)):
            ax = axes[row, j]
            f, P, fmin = mean_spec[(wname, ring)]
            m = f > 0
            ax.plot(1 / f[m], P[m], color=L.COL["ink"], lw=1.2)
            ax.axvspan(1 / fmin, 4000, color=L.COL["grid"], alpha=0.7, lw=0)
            for q in feat_actual:
                ax.axvline(q, color=L.COL["mlp"], lw=0.8, ls="--")
            for q in feat_plan:
                ax.axvline(q, color=L.COL["formula"], lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_xlim(40, 4000)
            ax.set_ylim(0, 1.15)
            ax.set_xlabel("period [mm]")
            ax.set_title(f"ring {ring}, window {wname.split(': ')[1]}", fontsize=7)
    axes[0, 0].set_ylabel("Δ ln P [nat] (light→dark: low→high p)")
    axes[1, 0].set_ylabel("rel. power (window as specified)")
    axes[2, 0].set_ylabel("rel. power (extra window)")
    axes[1, 0].plot([], [], color=L.COL["mlp"], ls="--", lw=0.8, label="MLP z-feature periods 1650/k")
    axes[1, 0].plot([], [], color=L.COL["formula"], ls=":", lw=0.8, label="periods listed in PLAN")
    axes[1, 0].legend(fontsize=6)
    fig.suptitle("D9 · ringing of the MLP map relative to the formula; spectra are means over 12 momenta, each "
                 "normalised to its largest resolved peak (grey: periods longer than the window)", fontsize=9)
    fig.tight_layout()
    fig.savefig(L.FIG / "d9_ringing.png", dpi=95)


if __name__ == "__main__":
    main()
