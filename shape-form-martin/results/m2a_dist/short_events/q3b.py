"""Questions 3b, 3c and 5 from the recomputed per-hit decomposition (decomp.py output, a scratch npz):
normalisation term and hit term of paired differences, the hit term by region of the hit centre in (z', r), the model
masses in front of zb, depth profiles around the short events' zb, and the defect checks. CPU only, read-only.

    uv run python -B shape-form-martin/results/m2a_dist/short_events/q3b.py <decomp.npz>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import se_common as S  # noqa: E402
from sfm import provenance, stats  # noqa: E402

E4 = S.E4
Z_MIN, Z_MAX, N_Z, R_MAX, N_R = -200.0, 1450.0, 128, 700.0, 32
DZ = (Z_MAX - Z_MIN) / N_Z
R_EDGES = R_MAX * (np.arange(N_R + 1) / N_R) ** 1.7
ENTRANCE_MM = 150.0
LAST_MM = 200.0


def pdm(x: np.ndarray) -> dict:
    """mean +- SE of a per-event array (1e-4 nat units)."""
    return S.pd(stats.paired(x, np.zeros_like(x)))


def main():
    f = sys.argv[1]
    Z = np.load(f)
    runs = [str(x) for x in Z["runs"]]
    R = {n: i for i, n in enumerate(runs)}
    done = Z["done"]
    nll, logm, wsum, hit = Z["nll"].astype(np.float64), Z["log_mass"].astype(np.float64), Z["weight_sum"].astype(
        np.float64), Z["hit"].astype(np.float64)
    V = S.val_info()
    val = V["_sets"]["val"]
    sh = S.short_mask()
    off = val.off.numpy()
    N = len(val)
    ev = np.repeat(np.arange(N), np.diff(off))
    xyz = val.xyz.numpy().astype(np.float64)
    zp, r = xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])
    zb = V["zb"]
    zbh = zb[ev]
    out = {"runs": runs, "n_evaluated": int(done.sum()), "full_pass": bool(done.all())}
    print(f"evaluated events {int(done.sum())} of {N} (full pass: {bool(done.all())})")

    # ---- recomputation vs registered arrays --------------------------------------------------------------------------
    agree = {}
    for n in runs:
        reg = S.tagged(n).values
        tc = provenance.tolerance_check(nll[R[n], done], reg[done], top=3)
        hs = np.bincount(ev, weights=hit[R[n]], minlength=N)
        recon = wsum * logm[R[n]] - hs
        agree[n] = {"max_abs": tc["max_abs"], "rms": tc["rms"], "mean_diff": tc["mean_diff"],
                    "n_above_1e-4": tc["n_above"]["0.0001"], "n_above_1e-5": tc["n_above"]["1e-05"],
                    "max_abs_short": float(np.abs(nll[R[n], sh & done] - reg[sh & done]).max()),
                    "parts_identity_max_abs": float(np.abs(recon[done] - nll[R[n], done]).max())}
        print(f"agree {n:10s} max|d| {tc['max_abs']:.2e} (short {agree[n]['max_abs_short']:.2e}) rms {tc['rms']:.2e} "
              f"mean {tc['mean_diff']:+.2e} n>1e-4 {agree[n]['n_above_1e-4']}  parts identity "
              f"{agree[n]['parts_identity_max_abs']:.2e}")
    out["agreement"] = agree

    # ---- regions of the hit centre -----------------------------------------------------------------------------------
    front = zp < 0
    beyond = zp >= zbh
    last = (~beyond) & (zp >= zbh - LAST_MM) & (~front)
    entr = (~front) & (~beyond) & (~last) & (zp < ENTRANCE_MM)
    bulk = (~front) & (~beyond) & (~last) & (~entr)
    zreg = {"front z'<0": front, f"entrance 0<=z'<{ENTRANCE_MM:g}": entr, "bulk": bulk,
            f"last {LAST_MM:g} mm before zb": last, "beyond zb (z'>=zb)": beyond}
    ir = np.clip(np.searchsorted(R_EDGES, r, side="right") - 1, 0, N_R - 1)
    rreg = {"core r<30": r < 30, "30<=r<200": (r >= 30) & (r < 200),
            f"200<=r<{R_EDGES[N_R - 1]:.0f}": (r >= 200) & (ir < N_R - 1), "last ring": ir == N_R - 1}
    zabs = {f"[{a:.0f},{b:.0f})": (zp >= a) & (zp < b) for a, b in
            [(-200, 0), (0, 100), (100, 300), (300, 600), (600, 900), (900, 1100), (1100, 1221), (1221, 1450)]}
    # hit counts and weights of the short events by region
    hw = val.w.numpy().astype(np.float64)
    reg_desc = {}
    for lab, m in {**zreg, **rreg}.items():
        reg_desc[lab] = {"hits_short": int((m & sh[ev]).sum()), "weight_share_short": float(hw[m & sh[ev]].sum() / sh.sum()),
                         "hits_other": int((m & ~sh[ev]).sum()),
                         "weight_share_other": float(hw[m & ~sh[ev]].sum() / (~sh).sum())}
    out["region_occupancy"] = reg_desc
    print("occupancy (energy share per event) short / other:")
    for lab, d in reg_desc.items():
        print(f"    {lab:28s} short {d['weight_share_short']:.4f} ({d['hits_short']} hits)  other "
              f"{d['weight_share_other']:.4f}")

    comps = [("A9_s0", "A2c_s0"), ("A9_s1", "A2c_s1"), ("A9_s0", "A2s_s0"), ("A9_s1", "A2s_s1"),
             ("A2c_s0", "A2s_s0"), ("A2c_s1", "A2s_s1"), ("A7_s0", "A2c_s0"), ("A7_s1", "A2c_s1"),
             ("A8_s0", "A2c_s0"), ("A8_s1", "A2c_s1"), ("A7s_s0", "A2c_s0"), ("A9_s0", "A7_s0"),
             ("A9_s1", "A7_s1"), ("A9_s0", "A8_s0"), ("A9_s0", "A7s_s0"), ("A12_s0", "A2c_s0"), ("A9r_s0", "A9_s0"),
             ("A2s_s0_x2", "A2c_s0"), ("A2s_s1", "A2s_s0"), ("A2c_s1", "A2c_s0"), ("A9_s1", "A9_s0")]
    comps = [c for c in comps if c[0] in R and c[1] in R]
    res = {}
    for a, b in comps:
        ia, ib = R[a], R[b]
        dn = wsum * (logm[ia] - logm[ib])                                    # normalisation term per event
        dh_hit = -(hit[ia] - hit[ib])                                        # per hit
        dh = np.bincount(ev, weights=dh_hit, minlength=N)                    # hit term per event
        tot = nll[ia] - nll[ib]
        regd = S.diff(a, b)
        row = {"total_recomputed": {}, "total_registered": {}, "norm": {}, "hit": {}, "hit_z": {}, "hit_r": {},
               "hit_zabs": {}, "hit_zr": {}}
        for key, m in (("short", sh & done), ("other", ~sh & done)):
            row["total_recomputed"][key] = pdm(tot[m])
            row["total_registered"][key] = pdm(regd[m])
            row["norm"][key] = pdm(dn[m])
            row["hit"][key] = pdm(dh[m])
            row["hit_z"][key] = {lab: pdm(np.bincount(ev, weights=dh_hit * mm, minlength=N)[m]) for lab, mm in zreg.items()}
            row["hit_r"][key] = {lab: pdm(np.bincount(ev, weights=dh_hit * mm, minlength=N)[m]) for lab, mm in rreg.items()}
            row["hit_zabs"][key] = {lab: pdm(np.bincount(ev, weights=dh_hit * mm, minlength=N)[m])
                                    for lab, mm in zabs.items()}
            if key == "short":
                row["hit_zr"][key] = {f"{zl} x {rl}": pdm(np.bincount(ev, weights=dh_hit * zm * rm, minlength=N)[m])
                                      for zl, zm in zreg.items() for rl, rm in rreg.items()}
        res[f"{a} - {b}"] = row
        s = row
        print(f"\n== {a} - {b}: short total {S.fmt(s['total_recomputed']['short'])} (registered "
              f"{S.fmt(s['total_registered']['short'])}) = norm {S.fmt(s['norm']['short'])} + hit "
              f"{S.fmt(s['hit']['short'])};  other total {S.fmt(s['total_recomputed']['other'])} = norm "
              f"{S.fmt(s['norm']['other'])} + hit {S.fmt(s['hit']['other'])}")
        for lab in zreg:
            print(f"    z  {lab:28s} short {S.fmt(s['hit_z']['short'][lab])}   other {S.fmt(s['hit_z']['other'][lab])}")
        for lab in rreg:
            print(f"    r  {lab:28s} short {S.fmt(s['hit_r']['short'][lab])}   other {S.fmt(s['hit_r']['other'][lab])}")
        for lab in zabs:
            print(f"    z' {lab:28s} short {S.fmt(s['hit_zabs']['short'][lab])}   other "
                  f"{S.fmt(s['hit_zabs']['other'][lab])}")
    out["decomposition"] = res
    zr = res.get("A9_s0 - A2c_s0", {}).get("hit_zr", {}).get("short", {})
    if zr:
        print("\nA9_s0 - A2c_s0, short events, hit term by (z' region x r region):")
        for k, v in zr.items():
            if abs(v["mean"]) > 0.5:
                print(f"    {k:50s} {S.fmt(v)}")

    # per-event: is the normalisation-term difference a function of zb?
    corr = {}
    for a, b in [("A9_s0", "A2c_s0"), ("A7_s0", "A2c_s0"), ("A2c_s0", "A2s_s0")]:
        if a in R and b in R:
            dn = (wsum * (logm[R[a]] - logm[R[b]]))[sh]
            dh = np.bincount(ev, weights=-(hit[R[a]] - hit[R[b]]), minlength=N)[sh]
            corr[f"{a}-{b}"] = {"corr_norm_hit": float(np.corrcoef(dn, dh)[0, 1]),
                                "by_zb": {f"[{lo},{hi})": {"n": int(((zb[sh] >= lo) & (zb[sh] < hi)).sum()),
                                                            "norm": float(dn[(zb[sh] >= lo) & (zb[sh] < hi)].mean() * E4),
                                                            "hit": float(dh[(zb[sh] >= lo) & (zb[sh] < hi)].mean() * E4)}
                                          for lo, hi in [(0, 600), (600, 1000), (1000, 1150), (1150, 1221)]}}
            print(f"{a}-{b} short: corr(norm, hit) per event {corr[f'{a}-{b}']['corr_norm_hit']:+.3f}; by zb",
                  {k: (v['n'], round(v['norm'], 1), round(v['hit'], 1)) for k, v in corr[f'{a}-{b}']['by_zb'].items()})
    out["norm_vs_hit_by_zb"] = corr

    # ---- Q3c: masses in front of zb and depth profiles ---------------------------------------------------------------
    masses = {}
    for n in runs:
        M = np.exp(logm[R[n]])
        masses[n] = {"short_mean_M": float(M[sh & done].mean()), "other_mean_M": float(M[~sh & done].mean()),
                     "short_mean_lnM": float(logm[R[n], sh & done].mean()),
                     "other_mean_lnM": float(logm[R[n], ~sh & done].mean()),
                     "short_min_M": float(M[sh & done].min()), "max_lnM": float(logm[R[n], done].max())}
    out["masses"] = masses
    print("\nmass in front of zb (mean M_e; mean ln M_e):")
    for n, d in masses.items():
        print(f"    {n:10s} short M {d['short_mean_M']:.5f} lnM {d['short_mean_lnM']:+.5f} (min M {d['short_min_M']:.4f})"
              f"   other M {d['other_mean_M']:.6f} lnM {d['other_mean_lnM']:+.6f}   max lnM {d['max_lnM']:+.2e}")
    # stored mass monitors (guarded runs) vs recomputed M_e
    mm_chk = {}
    for n in runs:
        mon = S.mass_monitors(n)
        if mon is None:
            continue
        ins = mon.get("inside_zb")
        if ins is not None:
            rel = np.abs(np.exp(logm[R[n], done]) - ins[done]) / ins[done]
            mm_chk[n] = {"max_rel_diff_inside_zb": float(rel.max()), "keys": sorted(mon)}
            print(f"    stored mass monitor inside_zb vs recomputed M_e, {n}: max rel diff {rel.max():.2e}")
            mm_chk[n]["short_beyond_zb_mean"] = float(mon["beyond_zb"][sh].mean())
            mm_chk[n]["other_beyond_zb_mean"] = float(mon["beyond_zb"][~sh].mean())
            mm_chk[n]["total_range"] = [float(mon["total"].min()), float(mon["total"].max())]
    out["mass_monitor_check"] = mm_chk

    idx = Z["pz_short_idx"]
    pz = Z["pz_short"]                              # (n_short, M, N_Z)
    zc = Z_MIN + DZ * (np.arange(N_Z) + 0.5)
    prof = {}
    if len(idx):
        zbs = zb[idx]
        rel_bins = [(-600, -400), (-400, -200), (-200, -100), (-100, 0), (0, 100), (100, 300)]
        for n in runs:
            P = pz[:, R[n], :]
            cdf = np.concatenate([np.zeros((len(P), 1)), np.cumsum(P, 1)], 1)
            mono = bool((np.diff(cdf, axis=1) >= -1e-15).all())
            # mass in windows relative to each event's zb (by bin centre)
            win = {}
            for lo, hi in rel_bins:
                msk = (zc[None, :] >= zbs[:, None] + lo) & (zc[None, :] < zbs[:, None] + hi)
                win[f"zb{lo:+d}..{hi:+d}"] = float((P * msk).sum(1).mean())
            prof[n] = {"cdf_monotone": mono, "total_range": [float(P.sum(1).min()), float(P.sum(1).max())],
                       "mean_mass_rel_zb": win, "mean_profile_short": P.mean(0).tolist()}
        print("\ndepth profile around the short events' zb (mean model mass in windows relative to zb):")
        for n in runs:
            print(f"    {n:10s} monotone cdf {prof[n]['cdf_monotone']} total {prof[n]['total_range'][0]:.6f}.."
                  f"{prof[n]['total_range'][1]:.6f}  " + "  ".join(f"{k} {v:.5f}" for k, v in
                                                                prof[n]['mean_mass_rel_zb'].items()))
        for a, b in [("A9_s0", "A2c_s0"), ("A7_s0", "A2c_s0"), ("A2c_s0", "A2s_s0"), ("A9_s1", "A2c_s1")]:
            if a in R and b in R:
                ra = {k: prof[a]["mean_mass_rel_zb"][k] / prof[b]["mean_mass_rel_zb"][k] - 1 for k in
                      prof[a]["mean_mass_rel_zb"]}
                print(f"    relative mass difference {a} vs {b}: " + "  ".join(f"{k} {v:+.4f}" for k, v in ra.items()))
                out.setdefault("profile_ratio", {})[f"{a} vs {b}"] = ra
    out["profiles"] = {n: {k: v for k, v in d.items() if k != "mean_profile_short"} for n, d in prof.items()}
    np.save(S.HERE / "mean_profile_short.npy", np.array([prof[n]["mean_profile_short"] for n in runs]) if prof else
            np.zeros(0))

    # ---- Q5 checks ---------------------------------------------------------------------------------------------------
    q5 = {}
    q5["nonfinite"] = {n: {"nll": int((~np.isfinite(nll[R[n], done])).sum()),
                           "hit": int((~np.isfinite(hit[R[n], np.repeat(done, np.diff(off))])).sum()),
                           "log_mass": int((~np.isfinite(logm[R[n], done])).sum())} for n in runs}
    q5["lnM_positive_max"] = {n: float(logm[R[n], done].max()) for n in runs}
    q5["M_le_1e-3"] = {n: int((logm[R[n], done] < np.log(1e-3)).sum()) for n in runs}
    q5["weight_sum_range"] = [float(wsum[done].min()), float(wsum[done].max())]
    t = (zb - Z_MIN) / DZ
    q5["zb_at_bin_edge_1e-3"] = int((np.abs(t - np.round(t)) < 1e-3).sum())
    q5["zb_at_bin_edge_short"] = int((np.abs(t - np.round(t)) < 1e-3)[sh].sum())
    q5["zb_beyond_grid"] = int((zb >= Z_MAX).sum())
    q5["zb_beyond_grid_short"] = int((zb >= Z_MAX)[sh].sum())
    q5["zb_min"] = float(zb.min())
    q5["short_nhits_1"] = int((V["nhits"][sh] == 1).sum())
    q5["short_nhits_le3"] = int((V["nhits"][sh] <= 3).sum())
    q5["hits_beyond_zb_short"] = int((beyond & sh[ev]).sum())
    q5["hits_beyond_zb_other"] = int((beyond & ~sh[ev]).sum())
    q5["weight_beyond_zb_short_per_event"] = float(hw[beyond & sh[ev]].sum() / sh.sum())
    q5["hits_front_short"] = int((front & sh[ev]).sum())
    # few-hit events: their contribution to A9 - A2c
    if "A9_s0" in R and "A2c_s0" in R:
        d = (nll[R["A9_s0"]] - nll[R["A2c_s0"]])
        for k in (1, 3, 10):
            m = sh & (V["nhits"] <= k)
            q5[f"A9-A2c_s0_short_nhits_le{k}"] = {"n": int(m.sum()), "sum_share": float(d[m].sum() / d[sh].sum()),
                                                 "mean": float(d[m].mean() * E4) if m.any() else None}
    # events with zero visible energy or a single hit at the first layer
    q5["short_evis_min"] = float(V["evis"][sh].min())
    out["q5"] = q5
    print("\nQ5 checks:", q5)
    S.write_json(out, "q3b.json")


if __name__ == "__main__":
    main()
