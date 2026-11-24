"""M1 audit D2 - where the formula-MLP gap lives in (z', r) (validation).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_d2.py

Exact split per event (same fold points, parts_val_k64.npz):
    d_e = sum_i [w_i ln g_i^MLP - w_i ln g_i^F] + ws_e [ln M_F - ln M_MLP]        (ws_e = sum_i w_i = 1 up to rounding)
Hit term accumulated per hit on the native 128x32 grid by the cell of the hit CENTRE (common.locate) and on the coarse
regions z' [-200,-20,12,35,60,100,200,400,700,1450] x r [0,15,30,60,100,200,350,700] mm (by hit-centre coordinates),
averaged over events (sum over cells = mean hit term), for all events and six momentum groups. Normalisation term per
group. Gap budget: regions ranked by signed and by absolute contribution. Writes d2.json, d2_maps.npz (git-ignored),
figures/d2_native_maps.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

common = L.common


def budget(c, names, fracs=(0.5, 0.8)):
    """Regions needed to reach each fraction of the signed total (ranked by signed value, largest first) and of the
    absolute total (ranked by |value|)."""
    c = np.asarray(c, np.float64)
    tot, atot = c.sum(), np.abs(c).sum()
    out = {"signed_total": float(tot), "abs_total": float(atot)}
    o = np.argsort(-c)
    cs = np.cumsum(c[o])
    oa = np.argsort(-np.abs(c))
    ca = np.cumsum(np.abs(c[oa]))
    for f in fracs:
        n = int(np.searchsorted(cs / tot, f) + 1) if tot > 0 else None
        out[f"signed_{int(f * 100)}pct"] = {"n_regions": n, "regions": [
            {"region": names[k], "value": float(c[k]), "cum_share": float(cs[i] / tot)} for i, k in enumerate(o[:n])]}
        na = int(np.searchsorted(ca / atot, f) + 1)
        out[f"abs_{int(f * 100)}pct"] = {"n_regions": na, "regions": [
            {"region": names[k], "value": float(c[k]), "abs_cum_share": float(ca[i] / atot)} for i, k in enumerate(oa[:na])]}
    return out


def main():
    run = L.Run("d2")
    sets, extras, _, _ = L.load()
    val = sets["val"]
    parts = L.parts_val()
    N = len(val)
    nll_m, nll_f = parts["nll"][0].astype(np.float64), parts["nll"][1].astype(np.float64)
    d = nll_f - nll_m
    ht = parts["hit"][0].astype(np.float64) - parts["hit"][1].astype(np.float64)     # per hit: w ln g_MLP - w ln g_F
    norm = parts["weight_sum"].astype(np.float64) * (parts["log_mass"][1].astype(np.float64)
                                                     - parts["log_mass"][0].astype(np.float64))
    ev = L.hit_event(val)
    # native cell of the hit centre, with common.locate itself
    z_t = val.xyz[:, 2]
    r_t = torch.sqrt(val.xyz[:, 0] ** 2 + val.xyz[:, 1] ** 2)
    iz, ir = common.locate(z_t, r_t)
    cell = (iz * common.N_R + ir).cpu().numpy()
    z, r = L.hit_zr(val)
    reg = L.coarse_index(z, r)
    assert (reg >= 0).all(), "hit centre outside the coarse grid"
    p = val.ev["p"].cpu().numpy().astype(np.float64)
    grp = L.p_group(p)
    names = [L.region_name(k) for k in range(L.NZC * L.NRC)]
    zl, rl = L.region_labels()
    NREG = L.NZC * L.NRC

    per_ev_reg = np.bincount(ev * NREG + reg, weights=ht, minlength=N * NREG).reshape(N, NREG)
    hit_ev = per_ev_reg.sum(1)
    out = {"convention": "contributions in nat per event; positive = formula worse; hit term = w ln g_MLP - w ln g_F "
                         "summed over hits in the region, averaged over the events of the group",
           "coarse_z_edges_mm": L.COARSE_Z, "coarse_r_edges_mm": L.COARSE_R, "z_labels": zl, "r_labels": rl}
    maps = {}
    groups = [("all", np.ones(N, bool))] + [(lab, grp == g) for g, lab in enumerate(L.P_GROUP_LABELS)]
    res = {}
    for gname, m in groups:
        n = int(m.sum())
        hm = np.isin(ev, np.flatnonzero(m))
        native = np.bincount(cell[hm], weights=ht[hm], minlength=common.N_Z * common.N_R) / n
        maps[f"native_{gname}"] = native.reshape(common.N_Z, common.N_R)
        c = per_ev_reg[m].mean(0)
        se = per_ev_reg[m].std(0, ddof=1) / np.sqrt(n)
        absmap = np.bincount(reg[hm], weights=np.abs(ht[hm]), minlength=NREG) / n
        hit_mean, norm_mean, gap = float(hit_ev[m].mean()), float(norm[m].mean()), float(d[m].mean())
        res[gname] = {
            "n_events": n, "gap": L.mean_se(d[m]), "hit_term": L.mean_se(hit_ev[m]), "norm_term": L.mean_se(norm[m]),
            "closure_hit_plus_norm_minus_gap": hit_mean + norm_mean - gap,
            "closure_native_sum_minus_hit_term": float(native.sum() - hit_mean),
            "coarse_signed": c.reshape(L.NZC, L.NRC), "coarse_signed_se": se.reshape(L.NZC, L.NRC),
            "coarse_abs": np.abs(c).reshape(L.NZC, L.NRC),
            "coarse_sum_abs_per_hit_terms": absmap.reshape(L.NZC, L.NRC),
            "budget": budget(c, names)}
        maps[f"coarse_{gname}"] = c.reshape(L.NZC, L.NRC)
    out["groups"] = res
    out["closure_max_abs"] = max(abs(v["closure_hit_plus_norm_minus_gap"]) for v in res.values())
    out["closure_pass_1e-6"] = bool(out["closure_max_abs"] <= 1e-6)
    out["notes"] = {"absolute contribution": "coarse_abs = |signed regional total|; coarse_sum_abs_per_hit_terms = "
                                             "event average of sum_i |w_i ln g_MLP - w_i ln g_F| (a scale, not a budget)",
                    "native cell": "common.locate(z', r) of the hit centre; coarse region by hit-centre coordinates"}
    np.savez(L.OUT / "d2_maps.npz", **maps)
    run.write(out)
    figure(maps)
    print("closure", out["closure_max_abs"], {g: (v["gap"]["mean"], v["hit_term"]["mean"], v["norm_term"]["mean"])
                                              for g, v in res.items()})


def figure(maps):
    plt = L.style()
    from matplotlib.colors import SymLogNorm
    names = ["all"] + L.P_GROUP_LABELS
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.6), sharex=True, sharey=True)
    ze = common.Z_EDGES.cpu().numpy()
    re_idx = np.arange(common.N_R + 1)
    vmax = np.quantile(np.abs(maps["native_all"]), 0.999)
    norm = SymLogNorm(linthresh=vmax / 100, vmin=-vmax, vmax=vmax)
    for ax, g in zip(axes.ravel(), names):
        im = ax.pcolormesh(ze, re_idx, maps[f"native_{g}"].T, cmap=L.diverging_cmap(), norm=norm, shading="flat")
        ax.set_title(f"p {g} GeV" if g != "all" else "all events", fontsize=9)
        ax.grid(False)
    axes.ravel()[-1].axis("off")
    rt = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    redges = common.R_EDGES.cpu().numpy()
    for ax in axes[:, 0]:
        ax.set_yticks(rt)
        ax.set_yticklabels([f"{redges[i]:.0f}" for i in rt], fontsize=7)
        ax.set_ylabel("r [mm] (ring edges)")
    for ax in axes[1, :]:
        ax.set_xlabel("z' [mm]")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="hit term per cell [nat/event] (symlog)")
    fig.suptitle("D2 · hit term w ln g(MLP) − w ln g(formula) on the native grid, by hit-centre cell "
                 "(positive = formula worse)", fontsize=10)
    fig.savefig(L.FIG / "d2_native_maps.png", dpi=100)


if __name__ == "__main__":
    main()
