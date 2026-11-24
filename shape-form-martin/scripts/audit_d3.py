"""M1 audit D3 - longitudinal vs lateral share of the formula-MLP difference (validation; no training).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d3.py

Hybrids built from the two maps (Pz = sum_r P, T = P / Pz, log space):
    H1 = Pz_MLP(z') * T_F(r | z')        MLP longitudinal profile, formula lateral conditionals
    H2 = Pz_F(z')   * T_MLP(r | z')      formula longitudinal profile, MLP lateral conditionals
scored with the MLP and the formula on identical fold points (K=64, seed 12345, batch 128, deterministic sums).
Map level, at 12 momenta at event quantiles of the training momentum distribution:
    KL(MLP || F) = KL(Pz_MLP || Pz_F) + sum_z Pz_MLP(z) KL(T_MLP(.|z) || T_F(.|z))     (exact chain rule)
with the share of the conditional part coming from r < 30 mm (rings 0-4, upper edge 29.96 mm).
Writes d3.json and figures/d3_kl_split.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402

common = L.common


def main():
    run = L.Run("d3")
    sets, extras, mlp, formula = L.load()
    val, train = sets["val"], sets["train"]
    h1, h2 = L.Hybrid(mlp, formula), L.Hybrid(formula, mlp)
    nll = L.metric.test_nll_multi([mlp, formula, h1, h2], val, deterministic_sum=True).astype(np.float64)
    parts = L.parts_val()
    same = [float(np.abs(nll[j] - parts["nll"][j]).max()) for j in range(2)]
    names = ["mlp", "formula", "H1 = Pz_MLP x T_F", "H2 = Pz_F x T_MLP"]
    out = {"scores": {n: L.mean_se(nll[j]) for j, n in enumerate(names)},
           "max_abs_diff_vs_parts_file": {"mlp": same[0], "formula": same[1]},
           "empty_rows_guarded": {"H1": h1.empty_rows, "H2": h2.empty_rows}}
    out["paired"] = {
        "H1 - MLP": L.paired(nll[2], nll[0]), "H1 - formula": L.paired(nll[2], nll[1]),
        "H2 - MLP": L.paired(nll[3], nll[0]), "H2 - formula": L.paired(nll[3], nll[1]),
        "formula - MLP": L.paired(nll[1], nll[0]),
        "(H1 - MLP) + (H2 - MLP) - (formula - MLP)": L.mean_se((nll[2] - nll[0]) + (nll[3] - nll[0]) - (nll[1] - nll[0]))}
    # by momentum group, for orientation
    grp = L.p_group(val.ev["p"].cpu().numpy())
    out["paired_by_p_group"] = {lab: {"n": int((grp == g).sum()),
                                      "H1 - MLP": L.paired(nll[2][grp == g], nll[0][grp == g]),
                                      "H2 - MLP": L.paired(nll[3][grp == g], nll[0][grp == g]),
                                      "formula - MLP": L.paired(nll[1][grp == g], nll[0][grp == g])}
                                for g, lab in enumerate(L.P_GROUP_LABELS)}

    # ---- map-level KL split --------------------------------------------------------------------------------------------
    p12 = L.momenta12(train)
    lm, lf = L.logP_at(mlp, p12), L.logP_at(formula, p12)
    kc = L.kl_chain(lm, lf)
    r_up = common.R_EDGES[1:].cpu().numpy()
    inner = r_up <= 30.0 + 1e-9
    cond_inner = kc["cond_cell"][:, :, inner].sum((1, 2))
    sd = sign_definite_split(lm, lf, inner)
    rows = []
    for i, p in enumerate(p12):
        rows.append({"p_GeV": float(p), "kl": float(kc["kl"][i]), "kl_z": float(kc["kl_z"][i]),
                     "kl_cond": float(kc["kl_cond"][i]), "kl_cond_r_lt_30mm": float(cond_inner[i]),
                     "share_cond_r_lt_30mm": float(cond_inner[i] / kc["kl_cond"][i]),
                     "chain_rule_residual": float(kc["kl"][i] - kc["kl_z"][i] - kc["kl_cond"][i]),
                     "cond_split_inner_outer_mass": float(sd["binary"][i]),
                     "cond_split_within_inner": float(sd["inner"][i]),
                     "cond_split_within_outer": float(sd["outer"][i]),
                     "cond_split_residual": float(kc["kl_cond"][i] - sd["binary"][i] - sd["inner"][i] - sd["outer"][i])})
    keys = ("kl", "kl_z", "kl_cond", "kl_cond_r_lt_30mm", "cond_split_inner_outer_mass", "cond_split_within_inner",
            "cond_split_within_outer")
    avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    avg["share_cond_r_lt_30mm_of_averages"] = avg["kl_cond_r_lt_30mm"] / avg["kl_cond"]
    avg["share_z_of_averages"] = avg["kl_z"] / avg["kl"]
    out["map_kl"] = {"momenta_GeV": p12, "quantile_levels": (np.arange(12) + 0.5) / 12, "rows": rows,
                     "event_weighted_average": avg, "rings_r_lt_30mm": np.flatnonzero(inner),
                     "ring_upper_edges_mm": r_up[inner],
                     "note": "the 12 momenta sit at equal-count quantiles of the training momenta, so the plain mean "
                             "over them is the event-weighted average. kl_cond_r_lt_30mm sums the per-cell terms "
                             "Pz T_MLP ln(T_MLP/T_F) over rings 0-4; these terms are not sign-definite, so its share can "
                             "exceed 1. cond_split_* is an exact sign-definite split of kl_cond (chain rule over the "
                             "partition r < 30 / r >= 30 mm): Pz-weighted binary KL of the inner/outer mass fractions, "
                             "plus inner-fraction-weighted KL of the renormalised inner conditionals, plus the same "
                             "for the outer rings"}
    run.write(out)
    figure(rows)
    print(out["scores"], out["paired"], avg)


def sign_definite_split(lm, lf, inner):
    """sum_z Pz_MLP [ KL_bin(a_z || b_z) + a_z KL(T_MLP^in || T_F^in) + (1-a_z) KL(T_MLP^out || T_F^out) ],
    a_z / b_z = inner mass fraction of the MLP / formula conditional at depth z (float64, log space)."""
    from scipy.special import logsumexp
    lPz_m, lPz_f = logsumexp(lm, 2), logsumexp(lf, 2)
    lTm, lTf = lm - lPz_m[..., None], lf - lPz_f[..., None]
    Pz = np.exp(lPz_m)
    res = {}
    la_m, la_f = logsumexp(lTm[..., inner], 2), logsumexp(lTf[..., inner], 2)
    lo_m, lo_f = logsumexp(lTm[..., ~inner], 2), logsumexp(lTf[..., ~inner], 2)
    a, o = np.exp(la_m), np.exp(lo_m)
    res["binary"] = (Pz * (a * (la_m - la_f) + o * (lo_m - lo_f))).sum(1)
    for key, mask, lm_part, lf_part, w in (("inner", inner, la_m, la_f, a), ("outer", ~inner, lo_m, lo_f, o)):
        ltm = lTm[..., mask] - lm_part[..., None]
        ltf = lTf[..., mask] - lf_part[..., None]
        kl = (np.exp(ltm) * (ltm - ltf)).sum(2)
        res[key] = (Pz * w * kl).sum(1)
    return res


def figure(rows):
    plt = L.style()
    p = np.array([r["p_GeV"] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.4))
    for key, lab, col in (("kl", "KL(MLP‖formula), map", L.COL["ink"]), ("kl_z", "longitudinal marginal", L.COL["mlp"]),
                          ("kl_cond", "lateral conditionals (Pz-weighted)", L.COL["formula"]),
                          ("kl_cond_r_lt_30mm", "… of which r < 30 mm", L.COL["h1"])):
        ax.plot(p, [r[key] for r in rows], "o-", ms=4, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p [GeV] (12 event quantiles of the training momenta)")
    ax.set_ylabel("KL [nat]")
    ax.legend(fontsize=7)
    ax.set_title("D3 · map-level KL(MLP‖formula), chain-rule split", fontsize=10)
    fig.tight_layout()
    fig.savefig(L.FIG / "d3_kl_split.png", dpi=110)


if __name__ == "__main__":
    main()
