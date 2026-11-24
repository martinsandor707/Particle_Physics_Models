"""M1-d Part C helper - lateral (side-face) acceptance of the calorimeter and what it does to the theta-blind benchmarks
(validation; no training).

    uv run python -B shape-form-martin/results/m1_audit/checks/c_side.py

Geometry hypothesis (supported by c_edge.py: the absolute hit coordinates stop at |X|, |Y| ~ 2.62 m for every theta):
the particle starts at the global origin and flies along (theta, phi); the calorimeter is a box |X|, |Y| <= X_E with
X_E = 54 cells = 2624.4 mm, front face half a layer in front of the first layer centre (global z0 = z_back - zb cos
theta), back face at z_back. The shower-frame origin sits at transverse position z0 tan(theta) (cos phi, sin phi).

Per event
    z_exit        depth z' at which the shower AXIS leaves through a lateral face (inf if it reaches the back face first)
    A_x(z', r)    azimuthal fraction of the ring (z', r) inside the box, for x in {front, front+back, side, all};
                  2 x 2 sub-points per grid cell, 24 azimuth nodes
    mass          sum_c t_e(c) (1 - A_x(c)),  t_e = model map truncated at z' < zb_e as in the metric's normaliser
    first-order acceptance gain   -ln sum_c t_e(c) A_x(c)   (what a model gains by removing exactly the mass that the
                  event's geometry makes undetectable, everything else unchanged; no parameters)
and from the data: energy share of hits whose cell centre lies outside the lateral box, and beyond z_exit.
Strata of z_exit: NLL of both benchmarks, their gap d_e = NLL_formula - NLL_MLP.
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

common, legacy, metric = chk.common, chk.legacy, chk.metric
DEV = chk.DEV
Z_BACK, X_E, LAYER = 4882.15, 54 * 48.6, 20.5
NSUB, NPHI = 2, 24


def exit_depth(theta, phi, z0):
    """Depth along the axis at which the axis crosses |X| = X_E or |Y| = X_E (inf if never)."""
    rho0 = z0 * np.tan(theta)
    X0, Y0 = rho0 * np.cos(phi), rho0 * np.sin(phi)
    cx, cy = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(np.abs(cx) > 1e-12, (X_E - np.sign(cx) * X0) / np.abs(cx), np.inf)
        ty = np.where(np.abs(cy) > 1e-12, (X_E - np.sign(cy) * Y0) / np.abs(cy), np.inf)
    return np.minimum(tx, ty), X0, Y0


@torch.no_grad()
def acceptances(b, X0, Y0, zbcos):
    """dict of A (B, N_Z, N_R): 'front', 'fb', 'side', 'all'."""
    k = (torch.arange(NSUB, device=DEV, dtype=torch.float32) + 0.5) / NSUB
    zs = (common.Z_EDGES[:-1, None] + common.DZ * k).reshape(-1)                                  # (N_Z * NSUB,)
    rs = torch.sqrt(common.R_EDGES[:-1, None] ** 2 + (common.R_EDGES[1:, None] ** 2 - common.R_EDGES[:-1, None] ** 2) * k).reshape(-1)
    al = (torch.arange(NPHI, device=DEV, dtype=torch.float32) + 0.5) / NPHI * 2 * math.pi
    R = b.R                                                                                      # global -> shower; global = R^T x
    # shower-frame point (r cos a, r sin a, z) -> global offset: sum_j R[j, i] x_j
    ca, sa = torch.cos(al), torch.sin(al)
    B = len(b.lnp)
    out = {}
    gx = (R[:, 0, 0, None, None, None] * rs[None, None, :, None] * ca + R[:, 1, 0, None, None, None] * rs[None, None, :, None] * sa
          + R[:, 2, 0, None, None, None] * zs[None, :, None, None])                               # (B, Z, Rr, A)
    gy = (R[:, 0, 1, None, None, None] * rs[None, None, :, None] * ca + R[:, 1, 1, None, None, None] * rs[None, None, :, None] * sa
          + R[:, 2, 1, None, None, None] * zs[None, :, None, None])
    gz = (R[:, 0, 2, None, None, None] * rs[None, None, :, None] * ca + R[:, 1, 2, None, None, None] * rs[None, None, :, None] * sa
          + R[:, 2, 2, None, None, None] * zs[None, :, None, None])
    front = gz >= -LAYER / 2
    back = gz <= zbcos[:, None, None, None]
    side = ((X0[:, None, None, None] + gx).abs() <= X_E) & ((Y0[:, None, None, None] + gy).abs() <= X_E)

    def red(m):
        return m.float().mean(3).reshape(B, common.N_Z, NSUB, common.N_R, NSUB).mean((2, 4))
    out["front"], out["fb"], out["side"], out["all"] = red(front), red(front & back), red(side), red(front & back & side)
    out["front_side"] = red(front & side)
    return out


def main():
    t0 = time.time()
    sets, extras = chk.load_cpu(("train", "val"))
    sh, ex = sets["val"], extras["val"]
    N = len(sh)
    th, ph = ex["theta"].astype(np.float64), ex["phi"].astype(np.float64)
    zb = sh.ev["zb"].numpy().astype(np.float64)
    p = sh.ev["p"].numpy().astype(np.float64)
    z0 = Z_BACK - zb * np.cos(th)
    zex, X0, Y0 = exit_depth(th, ph, z0)
    # ---- data: energy outside the lateral box / beyond the axis exit ---------------------------------------------------
    ev = chk.hit_event(sh)
    xyz = sh.xyz.numpy().astype(np.float64)
    w = sh.w.numpy().astype(np.float64)
    R = sh.ev["R"].numpy().astype(np.float64)
    g = np.einsum("hji,hj->hi", R[ev], xyz)
    outside = (np.abs(X0[ev] + g[:, 0]) > X_E) | (np.abs(Y0[ev] + g[:, 1]) > X_E)
    data_out = np.bincount(ev, weights=w * outside, minlength=N)
    data_beyond_exit = np.bincount(ev, weights=w * (xyz[:, 2] > zex[ev]), minlength=N)
    data_deep = np.bincount(ev, weights=w * (xyz[:, 2] >= 400), minlength=N)
    names = ("front", "fb", "side", "front_side", "all")
    res = {m: {n: np.zeros(N) for n in names} for m in ("mlp", "formula")}
    model_beyond_exit = {m: np.zeros(N) for m in ("mlp", "formula")}
    model_deep = {m: np.zeros(N) for m in ("mlp", "formula")}
    abar_side = None
    with chk.gpu_lock("m1d side acceptance"):
        val = sh.to(DEV)
        mlp = legacy.load_mlp_std()
        formula = legacy.load_component_formula(sets["train"])
        X0t, Y0t = torch.tensor(X0, dtype=torch.float32, device=DEV), torch.tensor(Y0, dtype=torch.float32, device=DEV)
        zbc = torch.tensor(zb * np.cos(th), dtype=torch.float32, device=DEV)
        zext = torch.tensor(np.minimum(zex, 1e9), dtype=torch.float32, device=DEV)
        idx_all = torch.arange(N, device=DEV)
        asum = {n: torch.zeros(common.N_Z, common.N_R, dtype=torch.float64, device=DEV) for n in names}
        with torch.no_grad():
            for s in range(0, N, 32):
                idx = idx_all[s:s + 32]
                b = val.batch(idx)
                A = acceptances(b, X0t[idx], Y0t[idx], zbc[idx])
                for n in names:
                    asum[n] += A[n].double().sum(0)
                m = ((b.zb[:, None].double() - common.Z_EDGES[None, :-1].double()) / common.DZ).clamp(0, 1)
                mex = ((zext[idx][:, None].double() - common.Z_EDGES[None, :-1].double()) / common.DZ).clamp(0, 1)
                for name, model in (("mlp", mlp), ("formula", formula)):
                    P = model(b.lnp).double().exp()
                    tm = P * m[:, :, None]
                    t = tm / tm.sum((1, 2), keepdim=True)
                    for n in names:
                        res[name][n][s:s + 32] = (t * (1 - A[n].double())).sum((1, 2)).cpu().numpy()
                    model_beyond_exit[name][s:s + 32] = (t * (1 - mex)[:, :, None]).sum((1, 2)).cpu().numpy()
                    model_deep[name][s:s + 32] = t[:, common.ZC.double() >= 400].sum((1, 2)).cpu().numpy()
        abar = {n: (asum[n] / N).cpu().numpy() for n in names}
        stored = legacy.load_abar().double().cpu().numpy()
    nm = np.load(chk.M0 / "mlp_val_nll.npy").astype(np.float64)
    nf = np.load(chk.M0 / "formula_val_nll.npy").astype(np.float64)
    d = nf - nm
    out = {"X_E_mm": X_E, "n": N,
           "data_energy_outside_box": chk.mean_se(data_out), "events_with_any_energy_outside": int((data_out > 0).sum()),
           "data_energy_outside_box_quantiles": {str(q): float(np.quantile(data_out, q)) for q in (0.5, 0.9, 0.99, 0.999, 1)},
           "abar_fb_vs_stored": {"max_abs": float(np.abs(abar["fb"] - stored).max()), "mean_abs": float(np.abs(abar["fb"] - stored).mean())},
           "exit_depth_fractions": {f"z_exit<{c}": float((zex < c).mean()) for c in (200, 400, 700, 1000, 1150, 1450)}}
    out["exit_depth_fractions"]["z_exit<zb"] = float((zex < zb).mean())
    out["model_mass_outside"] = {m: {n: chk.mean_se(res[m][n]) for n in names} for m in res}
    out["first_order_acceptance_gain"] = {m: {n: chk.mean_se(-np.log1p(-np.clip(res[m][n], 0, 1 - 1e-12))) for n in names} for m in res}
    # strata of the axis exit depth
    edges = [0, 200, 400, 700, 1000, 1450, np.inf]
    labs = ["<200", "200-400", "400-700", "700-1000", "1000-1450", "none (>=1450)"]
    cls = np.clip(np.searchsorted(edges, zex, side="right") - 1, 0, len(labs) - 1)
    strata = {}
    for i, lab in enumerate(labs):
        mk = cls == i
        n = int(mk.sum())
        if n < 2:
            continue
        strata[lab] = {"n": n, "share_of_events": n / N, "theta_mean": float(th[mk].mean()),
                       "nll_mlp": chk.mean_se(nm[mk]), "nll_formula": chk.mean_se(nf[mk]),
                       "d_e": chk.mean_se(d[mk]), "contribution_to_gap": float(d[mk].sum() / N),
                       "data_share_z_ge_400": chk.mean_se(data_deep[mk]), "mlp_share_z_ge_400": chk.mean_se(model_deep["mlp"][mk]),
                       "data_share_beyond_exit": chk.mean_se(data_beyond_exit[mk]),
                       "mlp_mass_beyond_exit": chk.mean_se(model_beyond_exit["mlp"][mk]),
                       "mlp_mass_outside_side": chk.mean_se(res["mlp"]["side"][mk]),
                       "mlp_mass_outside_front": chk.mean_se(res["mlp"]["front"][mk]),
                       "gain_side_mlp": chk.mean_se(-np.log1p(-np.clip(res["mlp"]["side"][mk], 0, 1 - 1e-12))),
                       "gain_front_mlp": chk.mean_se(-np.log1p(-np.clip(res["mlp"]["front"][mk], 0, 1 - 1e-12))),
                       "gain_all_mlp": chk.mean_se(-np.log1p(-np.clip(res["mlp"]["all"][mk], 0, 1 - 1e-12)))}
    out["strata_axis_exit_depth"] = strata
    contained = zex >= 1450
    out["gap_contained_axis"] = chk.mean_se(d[contained])
    out["gap_side_leaking"] = chk.mean_se(d[~contained])
    low_side = res["mlp"]["side"] < 0.01
    out["gap_mlp_side_mass_lt_1pct"] = {"n": int(low_side.sum()), "d_e": chk.mean_se(d[low_side]), "contribution": float(d[low_side].sum() / N)}
    out["gap_mlp_side_mass_ge_1pct"] = {"n": int((~low_side).sum()), "d_e": chk.mean_se(d[~low_side]), "contribution": float(d[~low_side].sum() / N)}
    # theta quintiles
    qe = np.quantile(th, np.linspace(0, 1, 6))
    qc = np.clip(np.searchsorted(qe[1:-1], th, side="right"), 0, 4)
    out["by_theta_quintile"] = [{"theta": [float(qe[q]), float(qe[q + 1])],
                                 "mlp_front": chk.mean_se(res["mlp"]["front"][qc == q]), "mlp_side": chk.mean_se(res["mlp"]["side"][qc == q]),
                                 "mlp_all": chk.mean_se(res["mlp"]["all"][qc == q]),
                                 "gain_front": chk.mean_se(-np.log1p(-res["mlp"]["front"][qc == q])),
                                 "gain_all": chk.mean_se(-np.log1p(-np.clip(res["mlp"]["all"][qc == q], 0, 1 - 1e-12))),
                                 "frac_axis_exit_lt_1450": float((zex[qc == q] < 1450).mean())} for q in range(5)]
    np.savez_compressed(chk.HERE / "c_side.npz", z_exit=zex, X0=X0, Y0=Y0, data_out=data_out, data_beyond_exit=data_beyond_exit,
                        abar_side=abar["side"], abar_all=abar["all"], abar_front=abar["front"], abar_fb=abar["fb"],
                        **{f"{m}_{n}": res[m][n] for m in res for n in names})
    chk.write("c_side.json", out, t0)
    for k, v in out.items():
        if k not in ("strata_axis_exit_depth", "by_theta_quintile"):
            print(k, v)
    for k, v in strata.items():
        print(k, {a: (round(b[0], 5), round(b[1], 5)) if isinstance(b, tuple) else (round(b, 5) if isinstance(b, float) else b) for a, b in v.items()})
    for r in out["by_theta_quintile"]:
        print(r)


if __name__ == "__main__":
    main()
