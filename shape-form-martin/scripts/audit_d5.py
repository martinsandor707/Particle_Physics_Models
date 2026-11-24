"""M1 audit D5 - front and back faces (validation).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d5.py

Per event and model (float64):
    beyond_zb, last_ring, first_z, last_z     sfm.metric.mass_monitors (mass with z' >= zb; ring 31; z-bins 0 / 127)
    front                                     model mass in front of the tilted ENTRANCE plane:
                                              sum_cells P * (1 - A_front),  A_front = event_slab(b, box, 4, True, False)
                                              (shapelab.SLAB_GEOM_SRC: event_slab(b, box, nsub, front, back))
Data side, for orientation: energy share of hit centres with z' > zb (behind the flat cut).
Means +- SE by theta quintile and by back-face stratum; NLL of each model and d_e by theta quintile; Pearson and
Spearman correlations of d_e with theta and with zb. Writes d5.json and figures/d5_masses.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sst  # noqa: E402

common = L.common
KEYS = ("beyond_zb", "last_ring", "first_z", "last_z", "front")


@torch.inference_mode()
def masses(models, sh):
    out = {n: {k: [] for k in KEYS} for n in models}
    event_slab = L.legacy.C["event_slab"]
    for b in sh.batches(128):
        A = event_slab(b, sh.box, 4, True, False).double()                       # (B, N_Z, N_R)
        for n, m in models.items():
            logP = m(b.lnp)
            mm = L.metric.mass_monitors(logP, b)
            for k in ("beyond_zb", "last_ring", "first_z", "last_z"):
                out[n][k].append(mm[k].cpu().numpy())
            out[n]["front"].append((logP.double().exp() * (1.0 - A)).sum((1, 2)).cpu().numpy())
    return {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in out.items()}


def main():
    run = L.Run("d5")
    sets, extras, mlp, formula = L.load()
    val = sets["val"]
    N = len(val)
    ms = masses({"mlp": mlp, "formula": formula}, val)
    np.savez(L.OUT / "d5_masses.npz", **{f"{n}_{k}": v for n, d in ms.items() for k, v in d.items()})
    parts = L.parts_val()
    nll_m, nll_f = parts["nll"][0].astype(np.float64), parts["nll"][1].astype(np.float64)
    d = nll_f - nll_m
    theta = extras["val"]["theta"]
    zb = val.ev["zb"].cpu().numpy().astype(np.float64)
    z, r = L.hit_zr(val)
    ev = L.hit_event(val)
    w = val.w.cpu().numpy().astype(np.float64)
    obs_behind = np.bincount(ev, weights=w * (z > zb[ev]), minlength=N)

    def summarize(mask):
        row = {"n": int(mask.sum())}
        for n in ("mlp", "formula"):
            row[n] = {k: L.mean_se(ms[n][k][mask]) for k in KEYS}
        row["formula_minus_mlp"] = {k: L.paired(ms["formula"][k][mask], ms["mlp"][k][mask]) for k in KEYS}
        row["data_energy_share_behind_zb"] = L.mean_se(obs_behind[mask])
        row["nll_mlp"] = L.mean_se(nll_m[mask])
        row["nll_formula"] = L.mean_se(nll_f[mask])
        row["d"] = L.mean_se(d[mask])
        return row

    tcls, tedges = L.quantile_bins(theta, 5)
    out = {"definitions": {"front": "sum_cells P (1 - A_front), A_front = event_slab(b, box, nsub=4, front=True, back=False)",
                           "beyond_zb": "model mass with z' >= zb (flat cut; linear inside the z-bin of zb)",
                           "last_ring": "mass in ring 31 (r >= 641.9 mm)", "first_z": "mass in z-bin 0",
                           "last_z": "mass in z-bin 127",
                           "data_energy_share_behind_zb": "event mean of sum_i w_i [z'_i > zb] (hit centres)"},
           "all": summarize(np.ones(N, bool)),
           "theta_quintile": {f"Q{i + 1} [{tedges[i]:.3f},{tedges[i + 1]:.3f}] rad": summarize(tcls == i) for i in range(5)},
           "zb_strata": {"zb >= 1221 mm": summarize(zb >= L.ZB_FULL), "zb < 1221 mm": summarize(zb < L.ZB_FULL),
                         "zb <= 1450 mm": summarize(zb <= 1450), "zb > 1450 mm": summarize(zb > 1450)}}
    corr = {}
    for name, x in (("theta", theta), ("zb", zb)):
        pr = sst.pearsonr(d, x)
        sr = sst.spearmanr(d, x)
        corr[name] = {"pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
                      "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue), "n": N}
    # additional, labelled: d_e against the difference of the front masses (formula - MLP)
    dfront = ms["formula"]["front"] - ms["mlp"]["front"]
    corr["front_mass_formula_minus_mlp (extra)"] = {
        "pearson_r": float(sst.pearsonr(d, dfront).statistic), "spearman_rho": float(sst.spearmanr(d, dfront).statistic)}
    out["correlations_of_d_e"] = corr
    run.write(out)
    figure(out)
    print(json_short(out))


def json_short(out):
    a = out["all"]
    return {n: {k: round(a[n][k]["mean"], 6) for k in KEYS} for n in ("mlp", "formula")} | {"corr": out["correlations_of_d_e"]}


def figure(out):
    plt = L.style()
    rows = out["theta_quintile"]
    labs = [k.split()[0] for k in rows]
    x = np.arange(len(labs))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for ax, key, title in ((axes[0], "front", "mass in front of entrance plane"),
                           (axes[1], "beyond_zb", "mass beyond zb")):
        for n, off in (("mlp", -0.1), ("formula", 0.1)):
            m = [rows[k][n][key]["mean"] for k in rows]
            s = [rows[k][n][key]["se"] for k in rows]
            ax.errorbar(x + off, m, s, fmt="o", ms=4, color=L.COL[n], capsize=2, label=n)
        if key == "beyond_zb":
            ax.errorbar(x, [rows[k]["data_energy_share_behind_zb"]["mean"] for k in rows],
                        [rows[k]["data_energy_share_behind_zb"]["se"] for k in rows], fmt="s", ms=4,
                        color=L.COL["ink"], capsize=2, label="data: energy behind zb")
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_xlabel("θ quintile")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("event-mean mass fraction")
    m = [rows[k]["d"]["mean"] * 1e3 for k in rows]
    s = [rows[k]["d"]["se"] * 1e3 for k in rows]
    axes[2].axhline(0, color=L.COL["ink2"], lw=0.8)
    axes[2].errorbar(x, m, s, fmt="o", ms=4, color=L.COL["formula"], capsize=2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labs)
    axes[2].set_xlabel("θ quintile")
    axes[2].set_ylabel("d_e [10⁻³ nat]")
    axes[2].set_title("d_e = NLL(formula) − NLL(MLP)", fontsize=9)
    fig.suptitle("D5 · front / back-face masses and the gap by θ quintile, validation", fontsize=10)
    fig.tight_layout()
    fig.savefig(L.FIG / "d5_masses.png", dpi=110)


if __name__ == "__main__":
    main()
