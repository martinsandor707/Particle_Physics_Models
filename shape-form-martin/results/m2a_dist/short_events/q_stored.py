"""Questions 1, 2, 3a and the stored-array part of 4: paired deltas of the registered per-event arrays (K = 64,
machine of record), restricted to the short events (zb < 1221 mm) and to the others. CPU only, read-only.

    uv run python -B shape-form-martin/results/m2a_dist/short_events/q_stored.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import se_common as S  # noqa: E402
from sfm import stats  # noqa: E402

E4 = S.E4


def main():
    V = S.val_info()
    sh = S.short_mask()
    bk = ~sh
    n_s, n_b = int(sh.sum()), int(bk.sum())
    print(f"validation events {len(sh)}; short {n_s}; others {n_b}; min zb {V['zb'].min():.1f}")
    out = {"n_short": n_s, "n_other": n_b}

    # ---- Q1 ----------------------------------------------------------------------------------------------------------
    pairs = [("A9", "A2c"), ("A9", "A2s"), ("A2c", "A2s"), ("A7", "A2c"), ("A8", "A2c"), ("A7s", "A2c"),
             ("A9r", "A9"), ("A9r", "A2c"), ("A7", "A2s"), ("A8", "A2s"), ("A12", "A2c"), ("A12", "A2s")]
    q1 = []
    for a, b in pairs:
        for s in (0, 1):
            ra, rb = f"{a}_s{s}", f"{b}_s{s}"
            d = S.diff(ra, rb)
            q1.append({"pair": f"{ra} - {rb}", "short": S.pd(S.delta(ra, rb, sh)), "other": S.pd(S.delta(ra, rb, bk)),
                       "all": S.pd(S.delta(ra, rb)), "short_median": float(np.median(d[sh]) * E4),
                       "short_n_worse_1unit": int((d[sh] > 1e-4).sum()),
                       "short_n_better_1unit": int((d[sh] < -1e-4).sum())})
            r = q1[-1]
            print(f"Q1 {r['pair']:20s} short {S.fmt(r['short'])}  other {S.fmt(r['other'])}  all {S.fmt(r['all'])}  "
                  f"median {r['short_median']:+.1f}  worse>1 {r['short_n_worse_1unit']}")
    out["q1"] = q1

    # ---- Q2: seed-to-seed --------------------------------------------------------------------------------------------
    groups = {"A2s": [0, 1, 2], "A2c": [0, 1], "A0": [0, 1], "A9": [0, 1], "A7": [0, 1], "A8": [0, 1], "A9r": [0, 1],
              "A7s": [0, 1], "A7l": [0, 1], "A8c": [0, 1], "A78": [0, 1], "A12": [0, 1], "A3x": [0, 1], "A2n": [0, 1, 2],
              "A2e": [0, 1], "A4": [0, 1], "A5": [0, 1], "A10": [0, 1], "A1": [0, 1]}
    q2 = []
    pooled = {"short": [0.0, 0], "other": [0.0, 0]}
    pooled_poly_from_a2s = {"short": [0.0, 0], "other": [0.0, 0]}
    for g, seeds in groups.items():
        names = [f"{g}_s{s}" for s in seeds]
        if any(S.latest_row(n) is None for n in names):
            continue
        for a, b in itertools.combinations(names, 2):
            q2.append({"pair": f"{b} - {a}", "group": g, "short": S.pd(S.delta(b, a, sh)),
                       "other": S.pd(S.delta(b, a, bk)), "all": S.pd(S.delta(b, a))})
            r = q2[-1]
            print(f"Q2 {r['pair']:20s} short {S.fmt(r['short'])}  other {S.fmt(r['other'])}")
        for key, m in (("short", sh), ("other", bk)):
            means = np.array([S.tagged(n).values[m].astype(np.float64).mean() for n in names])
            ss = float(((means - means.mean()) ** 2).sum())
            pooled[key][0] += ss
            pooled[key][1] += len(names) - 1
            if g in ("A2c", "A9", "A7", "A8", "A9r", "A7s", "A7l", "A8c", "A78", "A12", "A3x"):
                pooled_poly_from_a2s[key][0] += ss
                pooled_poly_from_a2s[key][1] += len(names) - 1
    tau = {k: float(np.sqrt(v[0] / v[1]) * E4) for k, v in pooled.items()}
    tau_a2s = {k: float(np.sqrt(v[0] / v[1]) * E4) for k, v in pooled_poly_from_a2s.items()}
    print(f"run-to-run SD of a stratum mean (pooled over {pooled['short'][1]} dof): short {tau['short']:.2f}, "
          f"other {tau['other']:.3f}; rungs started from A2s only ({pooled_poly_from_a2s['short'][1]} dof): "
          f"short {tau_a2s['short']:.2f}, other {tau_a2s['other']:.3f}")
    out["q2_pairs"] = q2
    out["q2_tau"] = {"all_groups": tau, "dof": pooled["short"][1], "from_a2s_groups": tau_a2s,
                     "dof_from_a2s": pooled_poly_from_a2s["short"][1]}

    # seed-to-seed variation of the DIFFERENCES that matter (difference of differences, event-paired)
    dd = {}
    for a, b in [("A9", "A2c"), ("A2c", "A2s"), ("A9", "A2s"), ("A7", "A2c"), ("A8", "A2c"), ("A7s", "A2c"),
                 ("A9", "A7"), ("A9", "A8"), ("A12", "A2c")]:
        d0 = S.diff(f"{a}_s0", f"{b}_s0")[sh]
        d1 = S.diff(f"{a}_s1", f"{b}_s1")[sh]
        x = stats.paired(d1, d0)
        dd[f"{a}-{b}"] = {"s0": d0.mean() * E4, "s1": d1.mean() * E4, "s1_minus_s0": S.pd(x),
                          "corr_per_event": float(np.corrcoef(d0, d1)[0, 1]),
                          "two_seed_mean": S.pd(stats.paired(0.5 * (d0 + d1), np.zeros_like(d0)))}
        print(f"Q2 diff-of-diff {a}-{b}: s0 {d0.mean()*E4:+.2f} s1 {d1.mean()*E4:+.2f}  s1-s0 {S.fmt(x)}  "
              f"per-event corr {dd[f'{a}-{b}']['corr_per_event']:.3f}")
    out["q2_diff_of_diff"] = dd

    # bootstrap by event (short events) and influence
    rng = np.random.default_rng(20261126)
    boot = {}
    infl = {}
    for a, b in [("A9", "A2c"), ("A2c", "A2s"), ("A9", "A2s"), ("A7", "A2c"), ("A9r", "A9")]:
        for s in (0, 1):
            ra, rb = f"{a}_s{s}", f"{b}_s{s}"
            d = S.diff(ra, rb)[sh] * E4
            idx = rng.integers(0, len(d), size=(20000, len(d)))
            bm = d[idx].mean(1)
            boot[f"{ra}-{rb}"] = {"mean": float(d.mean()), "boot_sd": float(bm.std(ddof=1)),
                                  "ci68": [float(np.percentile(bm, 16)), float(np.percentile(bm, 84))],
                                  "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                  "frac_le_0": float((bm <= 0).mean())}
            order = np.argsort(-d)
            order_abs = np.argsort(-np.abs(d))
            ev = np.flatnonzero(sh)
            top = []
            for j in order[:8]:
                e = ev[j]
                top.append({"val_index": int(e), "d": float(d[j]), "share_of_sum": float(d[j] / d.sum()),
                            "p": V["p"][e], "zb": V["zb"][e], "theta_deg": float(np.degrees(V["theta"][e])),
                            "evis": V["evis"][e], "nhits": int(V["nhits"][e])})
            wo = {}
            for k in (1, 3, 5):
                keep = np.ones(len(d), bool)
                keep[order[:k]] = False
                wo[f"without_top{k}_positive"] = S.pd(stats.paired(d[keep] / E4, np.zeros(keep.sum())))
                keep = np.ones(len(d), bool)
                keep[order_abs[:k]] = False
                wo[f"without_top{k}_abs"] = S.pd(stats.paired(d[keep] / E4, np.zeros(keep.sum())))
            infl[f"{ra}-{rb}"] = {"top": top, **wo,
                                  "trimmed_mean_10pct": float(np.mean(np.sort(d)[len(d) // 10: -len(d) // 10])),
                                  "median": float(np.median(d)),
                                  "sign_test_n_pos": int((d > 0).sum()), "n": len(d)}
            print(f"boot {ra}-{rb}: mean {d.mean():+.2f} boot sd {bm.std(ddof=1):.2f} 95% "
                  f"[{boot[f'{ra}-{rb}']['ci95'][0]:+.1f}, {boot[f'{ra}-{rb}']['ci95'][1]:+.1f}]  "
                  f"w/o top1 {S.fmt(wo['without_top1_positive'])} top3 {S.fmt(wo['without_top3_positive'])} "
                  f"top5 {S.fmt(wo['without_top5_positive'])}")
    out["q2_bootstrap"] = boot
    out["q2_influence"] = infl

    # ---- Q3a: strata of the short events -----------------------------------------------------------------------------
    p, zb, th, ev_, nh = V["p"], V["zb"], np.degrees(V["theta"]), V["evis"], V["nhits"]
    print("short events: p quantiles", np.percentile(p[sh], [0, 25, 50, 75, 100]).round(2),
          " zb", np.percentile(zb[sh], [0, 25, 50, 75, 100]).round(0),
          " theta", np.percentile(th[sh], [0, 25, 50, 75, 100]).round(1),
          " nhits", np.percentile(nh[sh], [0, 25, 50, 75, 100]).round(0),
          " evis", np.percentile(ev_[sh], [0, 25, 50, 75, 100]).round(3))
    print("other events: p quantiles", np.percentile(p[bk], [0, 25, 50, 75, 100]).round(2),
          " theta", np.percentile(th[bk], [0, 25, 50, 75, 100]).round(1),
          " nhits", np.percentile(nh[bk], [0, 25, 50, 75, 100]).round(0))
    out["q3a_describe"] = {k: {"short": np.percentile(x[sh], [0, 25, 50, 75, 100]).tolist(),
                               "other": np.percentile(x[bk], [0, 25, 50, 75, 100]).tolist()}
                           for k, x in (("p", p), ("zb", zb), ("theta_deg", th), ("nhits", nh), ("evis", ev_))}
    strata = {
        "p": [(0, 2), (2, 5), (5, 10), (10, 21)],
        "zb": [(0, 600), (600, 1000), (1000, 1150), (1150, 1221)],
        "theta_deg": [(0, 15), (15, 30), (30, 90)],
        "nhits": [(0, 20), (20, 60), (60, 10 ** 6)],
    }
    vals = {"p": p, "zb": zb, "theta_deg": th, "nhits": nh}
    q3a = {}
    for a, b in [("A9", "A2c"), ("A2c", "A2s"), ("A9", "A2s"), ("A7", "A2c"), ("A8", "A2c")]:
        for s in (0, 1):
            ra, rb = f"{a}_s{s}", f"{b}_s{s}"
            rows = {}
            for key, bins in strata.items():
                for lo, hi in bins:
                    m = sh & (vals[key] >= lo) & (vals[key] < hi)
                    rows[f"{key} [{lo},{hi})"] = S.pd(S.delta(ra, rb, m)) if m.sum() >= 2 else {"n": int(m.sum())}
                    if key == "p":       # the same momentum bins on the others, for comparison
                        mb = bk & (vals[key] >= lo) & (vals[key] < hi)
                        rows[f"OTHER {key} [{lo},{hi})"] = S.pd(S.delta(ra, rb, mb))
            q3a[f"{ra}-{rb}"] = rows
    for k, rows in q3a.items():
        print("Q3a", k)
        for lab, r in rows.items():
            print(f"    {lab:22s} {S.fmt(r) if 'mean' in r else r}")
    out["q3a"] = q3a

    # ---- Q4 (stored part) --------------------------------------------------------------------------------------------
    q4 = []
    for ra, rb in [("A9_s0", "A7_s0"), ("A9_s1", "A7_s1"), ("A9_s0", "A8_s0"), ("A9_s1", "A8_s1"),
                   ("A9_s0", "A7s_s0"), ("A9_s1", "A7s_s1"), ("A7_s0", "A8_s0"), ("A7_s1", "A8_s1"),
                   ("A7_s0", "A7s_s0"), ("A9n_s0", "A2n_s0"), ("A7n_s0", "A2n_s0"), ("A8n_s0", "A2n_s0"),
                   ("A4n_s0", "A2n_s0"), ("A2n_s0", "A2s_s0"), ("A9n_s0", "A9_s0"),
                   ("A12_s0", "A2c_s0"), ("A12_s1", "A2c_s1"), ("A12_s0", "A2s_s0"), ("A12_s1", "A2s_s1"),
                   ("A9r_s0", "A2s_s0"), ("A9r_s1", "A2s_s1"), ("A7l_s0", "A2c_s0"), ("A7l_s1", "A2c_s1"),
                   ("A8c_s0", "A2c_s0"), ("A8c_s1", "A2c_s1"), ("A78_s0", "A2c_s0"), ("A3x_s0", "A2c_s0"),
                   ("A2s_s0_x2", "A2s_s0"), ("A2s_s1_x2", "A2s_s1"), ("A2s_s0_x2", "A2c_s0"), ("A2s_s1_x2", "A2c_s1"),
                   ("A9_s0_init", "A2s_s0"), ("A9_s1_init", "A2s_s1"), ("A4_s0", "A2c_s0"), ("A5_s0", "A2c_s0"),
                   ("A10_s0", "A2c_s0"), ("A4_s1", "A2c_s1"), ("A5_s1", "A2c_s1")]:
        if S.latest_row(ra) is None or S.latest_row(rb) is None:
            continue
        q4.append({"pair": f"{ra} - {rb}", "short": S.pd(S.delta(ra, rb, sh)), "other": S.pd(S.delta(ra, rb, bk))})
        print(f"Q4 {q4[-1]['pair']:22s} short {S.fmt(q4[-1]['short'])}  other {S.fmt(q4[-1]['other'])}")
    out["q4_stored"] = q4

    # stratum means of every run (for the 'which runs sit where' picture)
    means = {}
    for r in S.registry_rows():
        n = r["name"]
        if r.get("stage") != "m2a_dist" or n in means:
            continue
        try:
            v = S.tagged(n).values.astype(np.float64)
        except Exception:
            continue
        means[n] = {"short": float(v[sh].mean()), "other": float(v[bk].mean()), "all": float(v.mean())}
    out["stratum_means"] = means
    for n in sorted(means, key=lambda k: means[k]["short"]):
        if any(n.startswith(x) for x in ("A2", "A7", "A8", "A9", "A12", "A0")):
            print(f"  mean short {means[n]['short']:.4f}  other {means[n]['other']:.5f}  {n}")
    S.write_json(out, "q_stored.json")


if __name__ == "__main__":
    main()
