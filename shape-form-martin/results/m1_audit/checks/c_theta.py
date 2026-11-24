"""M1-d Part C helper - how the measured shower-frame energy distribution depends on the incidence angle theta
(training + validation events, CPU only, no model).

    uv run python -B shape-form-martin/results/m1_audit/checks/c_theta.py

1. Depth shares inside the bulk (z' < 1150 mm, events with zb >= 1221 mm) by theta quintile and momentum group:
   is the theta dependence seen by D4 at z' >= 400 mm a back-face effect or a property of the bulk?
2. Detector-frame geometry of the hits: offset from the shower origin rotated back to the detector frame
   (R^T x_local): depth dz (must sit on the 20.5 mm layer pitch) and transverse offsets; looks for a lateral edge of
   the instrumented volume (a cliff in the transverse offset distribution).
3. Energy share in front of / behind the tilted faces as a function of theta (data only).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402

BANDS = [(-200, 0), (0, 60), (60, 200), (200, 400), (400, 700), (700, 1150)]


def main():
    t0 = time.time()
    sets, extras = chk.load_cpu(("train", "val"))
    out = {}
    for split in ("val", "train"):
        sh, ex = sets[split], extras[split]
        N = len(sh)
        ev = chk.hit_event(sh)
        xyz = sh.xyz.numpy().astype(np.float64)
        w = sh.w.numpy().astype(np.float64)
        z, r = xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])
        p = sh.ev["p"].numpy().astype(np.float64)
        zb = sh.ev["zb"].numpy().astype(np.float64)
        th = ex["theta"].astype(np.float64)
        full = zb >= 1221.0
        res = {"n": N}
        # ---- 1. bulk depth shares ------------------------------------------------------------------------------------
        inb = z < 1150.0
        Wb = np.bincount(ev, weights=w * inb, minlength=N)
        share = np.stack([np.bincount(ev, weights=w * ((z >= lo) & (z < hi)), minlength=N) for lo, hi in BANDS], 1)
        share_b = share / np.clip(Wb, 1e-12, None)[:, None]
        zmean = np.bincount(ev, weights=w * inb * z, minlength=N) / np.clip(Wb, 1e-12, None)
        rmean = np.bincount(ev, weights=w * inb * r, minlength=N) / np.clip(Wb, 1e-12, None)
        qe = np.quantile(th[full], np.linspace(0, 1, 6))
        qcls = np.clip(np.searchsorted(qe[1:-1], th, side="right"), 0, 4)
        grp = chk.p_group(p)
        tab = {}
        for g, lab in list(enumerate(chk.P_LABELS)) + [(-1, "all")]:
            rows = []
            for q in range(5):
                m = full & (qcls == q) & ((grp == g) if g >= 0 else True)
                n = int(m.sum())
                row = {"theta": [float(qe[q]), float(qe[q + 1])], "n": n,
                       "mean_z_bulk": [float(zmean[m].mean()), float(zmean[m].std(ddof=1) / np.sqrt(n))],
                       "mean_r_bulk": [float(rmean[m].mean()), float(rmean[m].std(ddof=1) / np.sqrt(n))],
                       "bulk_energy_fraction": float(Wb[m].mean())}
                for (lo, hi), col in zip(BANDS, share_b.T):
                    row[f"share[{lo},{hi})"] = [float(col[m].mean()), float(col[m].std(ddof=1) / np.sqrt(n))]
                rows.append(row)
            tab[lab] = rows
        res["bulk_depth_shares_by_theta_quintile"] = tab
        # slope of the deep share (z' in [400, 1150) within the bulk) against 1/cos(theta) - 1, per momentum group
        deep = share_b[:, 4] + share_b[:, 5]
        u = 1.0 / np.cos(th) - 1.0
        sl = {}
        for g, lab in list(enumerate(chk.P_LABELS)) + [(-1, "all")]:
            m = full & ((grp == g) if g >= 0 else True)
            x, y = u[m] - u[m].mean(), deep[m]
            b = float((x * (y - y.mean())).sum() / (x * x).sum())
            resid = y - y.mean() - b * x
            se = float(np.sqrt((resid ** 2).sum() / (len(x) - 2) / (x * x).sum()))
            sl[lab] = {"mean_deep_share": float(y.mean()), "slope_per_unit_u": b, "se": se, "z": b / se,
                       "u_range": [float(u[m].min()), float(u[m].max())]}
        res["deep_share_slope_vs_u"] = sl
        # ---- 2. detector-frame geometry ------------------------------------------------------------------------------
        R = sh.ev["R"].numpy().astype(np.float64)                         # global -> shower
        g_off = np.einsum("hji,hj->hi", R[ev], xyz)                        # R^T x_local
        dz = g_off[:, 2]
        k = dz / 20.5
        res["layer_pitch_check"] = {"max_abs_dz_minus_layer": float(np.abs(k - np.rint(k)).max() * 20.5),
                                    "dz_min": float(dz.min()), "dz_max": float(dz.max())}
        phi = ex["phi"].astype(np.float64)[ev]
        ut = g_off[:, 0] * np.cos(phi) + g_off[:, 1] * np.sin(phi)         # along the tilt direction
        vt = -g_off[:, 0] * np.sin(phi) + g_off[:, 1] * np.cos(phi)
        res["transverse_offsets_mm"] = {
            "x_abs_quantiles": {str(q): float(np.quantile(np.abs(g_off[:, 0]), q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)},
            "y_abs_quantiles": {str(q): float(np.quantile(np.abs(g_off[:, 1]), q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)},
            "along_tilt_quantiles": {str(q): float(np.quantile(ut, q)) for q in (0.0, 0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999, 1.0)},
            "across_tilt_abs_quantiles": {str(q): float(np.quantile(np.abs(vt), q)) for q in (0.5, 0.99, 0.999, 0.9999, 1.0)}}
        # hit-count histogram of the along-tilt offset for the deepest third of layers, largest theta quintile
        mdeep = (dz > 800) & (qcls[ev] == 4) & full[ev]
        h, e = np.histogram(ut[mdeep], bins=np.arange(-400, 1601, 50))
        hw, _ = np.histogram(ut[mdeep], bins=e, weights=w[mdeep])
        res["along_tilt_hist_deep_large_theta"] = {"edges": e, "hits": h, "energy": hw}
        # is there any hit in front of the first layer or behind the last one (detector depth)?
        kk = np.rint(k).astype(int)
        res["layer_index_range"] = [int(kk.min()), int(kk.max())]
        # ---- 3. data energy behind the flat cut / at negative z' by theta quintile -----------------------------------
        neg = np.bincount(ev, weights=w * (z < -20), minlength=N)
        beh = np.bincount(ev, weights=w * (z > zb[ev]), minlength=N)
        res["face_shares_by_theta_quintile"] = [
            {"theta": [float(qe[q]), float(qe[q + 1])], "n": int((full & (qcls == q)).sum()),
             "share_z_lt_-20": [float(neg[full & (qcls == q)].mean()), float(neg[full & (qcls == q)].std(ddof=1) / np.sqrt((full & (qcls == q)).sum()))],
             "share_behind_zb": [float(beh[full & (qcls == q)].mean()), float(beh[full & (qcls == q)].std(ddof=1) / np.sqrt((full & (qcls == q)).sum()))]}
            for q in range(5)]
        out[split] = res
        print(split, "layer check", res["layer_pitch_check"], res["layer_index_range"])
        print(" transverse", res["transverse_offsets_mm"])
        for lab in ("all", "5-10", "15-20"):
            print(" ", lab)
            for row in tab[lab]:
                print("   th %.3f-%.3f n=%5d <z>=%.1f±%.1f <r>=%.1f±%.1f bulkfrac=%.4f " % (row["theta"][0], row["theta"][1], row["n"], *row["mean_z_bulk"], *row["mean_r_bulk"], row["bulk_energy_fraction"])
                      + " ".join(f"{k[5:]}={v[0]:.4f}±{v[1]:.4f}" for k, v in row.items() if k.startswith("share")))
        print(" slopes", sl)
        print(" hist", list(zip(e[:-1].tolist(), h.tolist())))
        print(" faces", res["face_shares_by_theta_quintile"])
    chk.write("c_theta.json", out, t0)


if __name__ == "__main__":
    main()
