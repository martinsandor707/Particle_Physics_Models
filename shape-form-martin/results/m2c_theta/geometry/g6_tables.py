"""Geometry audit, step 6 - tables for question (b) from the per-event arrays of g5_gains.py (CPU, no data access).

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g6_tables.py

All numbers are AUDIT numbers (validation, laptop fold points). Unit of the tables: 1e-4 nat per event, mean +- paired
SE over events. Report-metric tables and bulk-restricted tables are separate; the only quantity that combines the
two is the persistence margin of PLAN section 13 (guard - 0.5 * main, sfm.stats.guard_persistence), in its own table.
"""
import json

import glib as L
from glib import np, stats

rep = np.load(L.ARR / "val_report.npz")
blk = np.load(L.ARR / "val_bulk_zb1221.npz")
evd = np.load(L.ARR / "val_events.npz")
N = len(rep["event_number"])
st = blk["stratum_index"]
zb, theta, p, kfirst = evd["zb"], evd["theta"], evd["p"], evd["first_hit_layer"]
in_st = np.zeros(N, bool)
in_st[st] = True
assert np.array_equal(in_st, zb >= 1221.0) and np.array_equal(blk["event_number"], rep["event_number"][st])
short = ~in_st
FOUR = ("F", "FB", "FS", "FSB")
FACE = {"F": "front", "FB": "front + back", "FS": "front + side", "FSB": "front + side + back"}


def base_of(m):
    return "mlp" if m.startswith("mlp") else "formula"


def P(d):
    d = np.asarray(d, dtype=np.float64)
    q = stats.paired(d, np.zeros_like(d))
    return {"mean": q.mean, "se": q.se, "n": q.n}


def f(q, dig=1):
    return f"{q['mean'] * 1e4:+.{dig}f} ± {q['se'] * 1e4:.{dig}f}"


def d_rep(m, ref=None, q="nll"):
    return rep[f"{m}|{q}"] - rep[f"{ref or base_of(m)}|{q}"]


def d_blk(m, ref=None, q="nll"):
    return blk[f"{m}|{q}"] - blk[f"{ref or base_of(m)}|{q}"]


FAM = {"MLP x A_e (naive)": "mlp*{}", "MLP x A_e / Abar (ratio)": "mlp*{}/abar",
       "formula x A_e (naive; stored template kept)": "formula*{}", "formula x A_e / Abar (ratio)": "formula*{}/abar",
       "core x A_e (template replaced)": "core*{}"}
out = {"n_val": N, "n_stratum": int(in_st.sum()), "n_short": int(short.sum()), "unit": "nat (tables print 1e-4 nat)",
       "label": "AUDIT numbers, validation, laptop fold points (RTX 4060, launch grid 144, K 64, seed 12345); not scores of record"}
md = []
ws = rep["weight_sum"]

# ---- T0 base maps ------------------------------------------------------------------------------------------------------------
out["base"] = {m: {"mean_nll_report": float(rep[f"{m}|nll"].mean()), "mass_beyond_zb": float(rep[f"{m}|beyond_zb"].mean())} for m in ("mlp", "formula", "core")}
out["base"]["formula_minus_mlp_report"] = P(rep["formula|nll"] - rep["mlp|nll"])
out["base"]["core_minus_formula_report"] = P(rep["core|nll"] - rep["formula|nll"])
ok_b = np.isfinite(blk["mlp|nll"]) & np.isfinite(blk["formula|nll"])
out["base"]["bulk_nan_events"] = int((~ok_b).sum())

# ---- T1 report metric, all events, with the two decompositions ---------------------------------------------------------------
md.append("### T1 Report metric, all validation events (n = %d). Δ against the map's own base; 1e-4 nat\n" % N)
md.append("| map | faces | Δ report metric | Δ hit term | Δ normaliser term | ln(share of the mass in front of z_b kept by the factor) | hit-side term (Δ minus the previous column) | mass beyond z_b: mean (base) | max |")
md.append("|---|---|---|---|---|---|---|---|---|")
T1 = {}
for fam, pat in FAM.items():
    for k in FOUR:
        m = pat.format(k)
        b = base_of(m)
        tot = d_rep(m)
        dh = -(rep[f"{m}|hit"] - rep[f"{b}|hit"])
        dn = ws * (rep[f"{m}|log_mass"] - rep[f"{b}|log_mass"])
        assert np.abs(tot - dh - dn).max() < 5e-4
        # identity: Delta = [cost paid by hits in partially accepted cells] + ln(accepted share of the mass with z' < zb)
        lnacc = rep[f"{m}|log_mass"] + rep[f"{m}|ln_factor_mass"] - rep[f"{b}|log_mass"]
        if fam.startswith("core"):                       # factor mass was computed on the core, base is the formula: use the core's own mass
            lnacc = rep[f"{m}|log_mass"] + rep[f"{m}|ln_factor_mass"] - rep["core|log_mass"]
            tot_core = d_rep(m, "core")
            cost = tot_core - lnacc
        else:
            cost = tot - lnacc
        T1[m] = {"family": fam, "faces": FACE[k], "delta": P(tot), "delta_hit": P(dh), "delta_norm": P(dn), "ln_accepted_share": P(lnacc),
                 "hit_cost": P(cost), "beyond_zb_mean": float(rep[f"{m}|beyond_zb"].mean()), "beyond_zb_base": float(rep[f"{b}|beyond_zb"].mean()),
                 "beyond_zb_max": float(rep[f"{m}|beyond_zb"].max()), "last_ring_mean": float(rep[f"{m}|last_ring"].mean()),
                 "outside_bulk_mean": float(rep[f"{m}|outside_bulk"].mean())}
        if fam.startswith("core"):
            T1[m]["delta_vs_core"] = P(tot_core)
        r = T1[m]
        md.append(f"| {fam} | {FACE[k]} | {f(r['delta'])} | {f(r['delta_hit'])} | {f(r['delta_norm'])} | {f(r['ln_accepted_share'])} | {f(r['hit_cost'])} | "
                  f"{r['beyond_zb_mean']:.4f} ({r['beyond_zb_base']:.4f}) | {r['beyond_zb_max']:.3f} |")
out["T1_report_all"] = T1
md.append("")

# ---- T2 G-bulk on the stratum ------------------------------------------------------------------------------------------------
md.append("### T2 G-bulk (PLAN A4): stratum z_b ≥ 1221 mm, z′ < 1150 mm without the last ring, Bulk(short=\"raise\"); n = %d. Bulk-restricted numbers only; 1e-4 nat\n" % int(ok_b.sum()))
md.append("| map | faces | Δ G-bulk | Δ hit term (bulk) | Δ normaliser term (bulk) |")
md.append("|---|---|---|---|---|")
T2, T3 = {}, {}
for fam, pat in FAM.items():
    for k in FOUR:
        m = pat.format(k)
        b = base_of(m)
        okm = ok_b & np.isfinite(blk[f"{m}|nll"])
        gb = d_blk(m)[okm]
        dh = -(blk[f"{m}|hit"] - blk[f"{b}|hit"])[okm]
        dn = (blk["_shared|weight_sum"] * (blk[f"{m}|log_mass"] - blk[f"{b}|log_mass"]))[okm]
        T2[m] = {"family": fam, "faces": FACE[k], "delta_bulk": P(gb), "delta_hit": P(dh), "delta_norm": P(dn), "n_nan": int((~okm).sum())}
        md.append(f"| {fam} | {FACE[k]} | {f(T2[m]['delta_bulk'])} | {f(T2[m]['delta_hit'])} | {f(T2[m]['delta_norm'])} |")
        main_s = d_rep(m)[st][okm]
        dec = stats.guard_persistence(main_s, gb)
        T3[m] = {"family": fam, "faces": FACE[k], "report_on_stratum": P(main_s), "bulk": P(gb), "margin": dec.numbers["margin"],
                 "se_margin": dec.numbers["se_margin"], "z_margin": dec.numbers["z_margin"], "fraction": dec.numbers["fraction"],
                 "passed": bool(dec.passed), "checks": {a: bool(v) for a, v in dec.checks.items()},
                 "resolved": bool(dec.flags.get("resolved", abs(dec.numbers["z_margin"]) >= 2))}
out["T2_bulk_stratum"] = T2
md.append("")
md.append("### T3 Persistence under G-bulk (PLAN §13): margin = Δ(G-bulk) − 0.5·Δ(report metric), same events (stratum), paired SE; pass if the margin is ≤ 0, resolved at 2 SE; 1e-4 nat\n")
md.append("| map | faces | report metric on the stratum | G-bulk on the stratum | margin ± SE | z | verdict |")
md.append("|---|---|---|---|---|---|---|")
for m, r in T3.items():
    verdict = ("pass" if r["passed"] else "fail") + (", resolved" if r["resolved"] else ", not resolved")
    if not r["checks"]["main_is_gain"]:
        verdict = "no gain to persist (report-metric Δ ≥ 0)"
    md.append(f"| {r['family']} | {r['faces']} | {f(r['report_on_stratum'])} | {f(r['bulk'])} | {r['margin'] * 1e4:+.1f} ± {r['se_margin'] * 1e4:.1f} | {r['z_margin']:+.1f} | {verdict} |")
out["T3_bulk_persistence"] = T3
md.append("")

# ---- T4 short events (own row) and leave-out ----------------------------------------------------------------------------------
md.append("### T4 Short events (z_b < 1221 mm, n = %d) as their own row, report metric; 1e-4 nat\n" % int(short.sum()))
md.append("| map | faces | Δ on short events | Δ on the stratum (n = %d) | Δ all events | share of the total gain from short events |" % int(in_st.sum()))
md.append("|---|---|---|---|---|---|")
T4 = {}
for fam, pat in FAM.items():
    for k in ("FS", "FSB"):
        m = pat.format(k)
        d = d_rep(m)
        T4[m] = {"short": P(d[short]), "stratum": P(d[in_st]), "all": P(d), "short_share": float(d[short].sum() / d.sum())}
        md.append(f"| {fam} | {FACE[k]} | {f(T4[m]['short'])} | {f(T4[m]['stratum'])} | {f(T4[m]['all'])} | {T4[m]['short_share'] * 100:.1f} % |")
out["T4_short_events"] = T4
md.append("")

# ---- T5 back-face share and front + side only ----------------------------------------------------------------------------------
md.append("### T5 Back-face share, report metric (all events): paired differences between face sets of the same map family; 1e-4 nat\n")
md.append("| family | Δ(front+side) | Δ(all faces) | back-face share = Δ(all) − Δ(front+side) | Δ(front+back) − Δ(front) | Δ(front+side) / Δ(all) |")
md.append("|---|---|---|---|---|---|")
T5 = {}
for fam, pat in FAM.items():
    fs, fsb, fr, fb = (d_rep(pat.format(k)) for k in ("FS", "FSB", "F", "FB"))
    T5[fam] = {"FS": P(fs), "FSB": P(fsb), "back_share": P(fsb - fs), "back_without_sides": P(fb - fr), "ratio": float(fs.mean() / fsb.mean())}
    md.append(f"| {fam} | {f(T5[fam]['FS'])} | {f(T5[fam]['FSB'])} | {f(T5[fam]['back_share'])} | {f(T5[fam]['back_without_sides'])} | {T5[fam]['ratio']:.3f} |")
out["T5_back_share_report"] = T5
md.append("")
md.append("### T5a Back-face share by part, report metric: Δ(all faces) − Δ(front + side), paired; 1e-4 nat\n")
md.append("| family | all events | stratum z_b ≥ 1221 mm (n = %d) | short events (n = %d) | of which hit term (all events) | of which normaliser term (all events) |" % (int(in_st.sum()), int(short.sum())))
md.append("|---|---|---|---|---|---|")
T5a = {}
for fam, pat in FAM.items():
    mf, ms = pat.format("FSB"), pat.format("FS")
    dd = rep[f"{mf}|nll"] - rep[f"{ms}|nll"]
    dh = -(rep[f"{mf}|hit"] - rep[f"{ms}|hit"])
    dn = ws * (rep[f"{mf}|log_mass"] - rep[f"{ms}|log_mass"])
    T5a[fam] = {"all": P(dd), "stratum": P(dd[in_st]), "short": P(dd[short]), "hit": P(dh), "norm": P(dn)}
    md.append(f"| {fam} | {f(T5a[fam]['all'])} | {f(T5a[fam]['stratum'])} | {f(T5a[fam]['short'])} | {f(T5a[fam]['hit'])} | {f(T5a[fam]['norm'])} |")
out["T5a_back_share_parts"] = T5a
md.append("")
md.append("### T5b Back-face share under G-bulk (stratum; bulk-restricted numbers only); 1e-4 nat\n")
md.append("| family | Δ G-bulk (front+side) | Δ G-bulk (all faces) | Δ(all) − Δ(front+side) |")
md.append("|---|---|---|---|")
T5b = {}
for fam, pat in FAM.items():
    okm = ok_b & np.isfinite(blk[pat.format("FS") + "|nll"]) & np.isfinite(blk[pat.format("FSB") + "|nll"])
    fs, fsb = d_blk(pat.format("FS"))[okm], d_blk(pat.format("FSB"))[okm]
    T5b[fam] = {"FS": P(fs), "FSB": P(fsb), "back_share": P(fsb - fs)}
    md.append(f"| {fam} | {f(T5b[fam]['FS'])} | {f(T5b[fam]['FSB'])} | {f(T5b[fam]['back_share'])} |")
out["T5b_back_share_bulk"] = T5b
md.append("")

# ---- T5c forms against each other and the finalists against each other -----------------------------------------------------------
md.append("### T5c Paired differences between forms and between the two stand-in finalists, report metric (all events); 1e-4 nat\n")
md.append("| comparison | front | front + back | front + side | all faces |")
md.append("|---|---|---|---|---|")
T5c = {}
for name, a, b in (("MLP: ratio − naive", "mlp*{}/abar", "mlp*{}"), ("formula: ratio − naive", "formula*{}/abar", "formula*{}"),
                   ("core × A_e − formula ratio", "core*{}", "formula*{}/abar"),
                   ("formula ratio − MLP ratio (the finalists' gap inside the arm)", "formula*{}/abar", "mlp*{}/abar")):
    T5c[name] = {k: P(rep[a.format(k) + "|nll"] - rep[b.format(k) + "|nll"]) for k in FOUR}
    md.append(f"| {name} | " + " | ".join(f(T5c[name][k]) for k in FOUR) + " |")
T5c["formula − MLP, θ-blind bases (for reference, not an arm number)"] = P(rep["formula|nll"] - rep["mlp|nll"])
out["T5c_forms"] = T5c
md.append("")
md.append("θ-blind bases, for reference only: formula − MLP = " + f(T5c["formula − MLP, θ-blind bases (for reference, not an arm number)"]) + ".\n")

# ---- T6 splits -----------------------------------------------------------------------------------------------------------------
zex = evd["axis_exit_stepped"]
leak = zex < np.minimum(zb, 1450.0)
qe = np.quantile(theta, np.linspace(0, 1, 6))
qc = np.clip(np.searchsorted(qe[1:-1], theta, side="right"), 0, 4)
splits = {f"θ quintile {i + 1} [{qe[i]:.3f}, {qe[i + 1]:.3f}] rad": qc == i for i in range(5)}
splits.update({"axis contained (no side face / hole before min(z_b, 1450 mm))": ~leak, "axis side-leaking": leak,
               "p < 2 GeV": p < 2.0, "p ≥ 2 GeV": p >= 2.0, "first layer hit (k = 0)": kfirst == 0, "late start (k ≥ 1)": kfirst >= 1})
PRIMARY = ["mlp*FS/abar", "mlp*FSB/abar", "formula*FS/abar", "formula*FSB/abar", "mlp*FS", "mlp*FSB", "core*FS", "core*FSB"]
FRONT = ["mlp*F/abar", "formula*F/abar", "mlp*FB/abar", "formula*FB/abar", "mlp*F", "core*F"]
md.append("### T6 Splits, report metric; Δ against the map's own base; 1e-4 nat (each cell: mean ± paired SE within the stratum of the split)\n")
md.append("| split | n | " + " | ".join(PRIMARY) + " |")
md.append("|---|---|" + "---|" * len(PRIMARY))
T6 = {}
for name, mk in splits.items():
    T6[name] = {"n": int(mk.sum()), **{m: P(d_rep(m)[mk]) for m in PRIMARY},
                "contribution_to_total": {m: float(d_rep(m)[mk].sum() / N) for m in PRIMARY}}
    md.append(f"| {name} | {int(mk.sum())} | " + " | ".join(f(T6[name][m]) for m in PRIMARY) + " |")
out["T6_splits_report"] = T6
md.append("")
md.append("### T6b Splits under G-bulk (stratum; bulk-restricted numbers only); 1e-4 nat\n")
md.append("| split | n | " + " | ".join(PRIMARY) + " |")
md.append("|---|---|" + "---|" * len(PRIMARY))
T6b = {}
for name, mk in splits.items():
    mks = mk[st] & ok_b
    if mks.sum() < 2:
        continue
    T6b[name] = {"n": int(mks.sum()), **{m: P(d_blk(m)[mks]) for m in PRIMARY}}
    md.append(f"| {name} | {int(mks.sum())} | " + " | ".join(f(T6b[name][m]) for m in PRIMARY) + " |")
out["T6b_splits_bulk"] = T6b
md.append("")
md.append("### T6c Splits for the front (and front + back) acceptance alone, report metric; 1e-4 nat\n")
md.append("| split | n | " + " | ".join(FRONT) + " |")
md.append("|---|---|" + "---|" * len(FRONT))
T6c = {}
for name, mk in splits.items():
    T6c[name] = {"n": int(mk.sum()), **{m: P(d_rep(m)[mk]) for m in FRONT}}
    md.append(f"| {name} | {int(mk.sum())} | " + " | ".join(f(T6c[name][m]) for m in FRONT) + " |")
out["T6c_splits_front_report"] = T6c
md.append("")

# ---- T7 sensitivity: outline model and origin ------------------------------------------------------------------------------------
md.append("### T7 Sensitivity to the volume model and to the origin, report metric (all events); 1e-4 nat\n")
md.append("| family | faces | stepped volume, true origin | audit box, true origin | octagon, true origin | stepped volume, straight-line origin | box − stepped | octagon − stepped | straight line − true origin |")
md.append("|---|---|---|---|---|---|---|---|---|")
T7 = {}
for fam, pat in (("MLP x A_e (naive)", "mlp*{}"), ("MLP x A_e / Abar (ratio)", "mlp*{}/abar"), ("core x A_e (template replaced)", "core*{}")):
    for k in ("FS", "FSB"):
        a0 = d_rep(pat.format(k))
        alt = {s: d_rep(pat.format(f"{k}_{s}")) for s in ("box", "oct", "line")}
        T7[pat.format(k)] = {"stepped": P(a0), **{s: P(v) for s, v in alt.items()}, **{f"{s}_minus_stepped": P(v - a0) for s, v in alt.items()}}
        r = T7[pat.format(k)]
        md.append(f"| {fam} | {FACE[k]} | {f(r['stepped'])} | {f(r['box'])} | {f(r['oct'])} | {f(r['line'])} | {f(r['box_minus_stepped'])} | {f(r['oct_minus_stepped'])} | {f(r['line_minus_stepped'])} |")
out["T7_sensitivity_report"] = T7
md.append("")

# ---- T8 mass monitors -------------------------------------------------------------------------------------------------------------
md.append("### T8 Mass monitors (validation, per-event maps): model mass beyond z_b, in the last ring, outside the bulk region\n")
md.append("| map | mass beyond z_b: mean | max | events with > 0.5 | last ring: mean | outside bulk region: mean |")
md.append("|---|---|---|---|---|---|")
T8 = {}
for m in ["mlp", "formula", "core"] + [pat.format(k) for pat in FAM.values() for k in FOUR]:
    bz = rep[f"{m}|beyond_zb"]
    T8[m] = {"beyond_zb_mean": float(bz.mean()), "beyond_zb_max": float(bz.max()), "n_gt_0.5": int((bz > 0.5).sum()),
             "last_ring_mean": float(rep[f"{m}|last_ring"].mean()), "outside_bulk_mean": float(rep[f"{m}|outside_bulk"].mean())}
    md.append(f"| {m} | {T8[m]['beyond_zb_mean']:.4f} | {T8[m]['beyond_zb_max']:.3f} | {T8[m]['n_gt_0.5']} | {T8[m]['last_ring_mean']:.4f} | {T8[m]['outside_bulk_mean']:.4f} |")
out["T8_mass_monitors"] = T8
md.append("")
# data side: energy of the events beyond zb (hits with z' >= zb) is not available here; mean accepted ring fraction per event:
mA = evd["mean_A"]
out["mean_grid_acceptance"] = {k: float(mA[:, i].mean()) for i, k in enumerate(FOUR)}
out["side_leaking_fraction"] = float(leak.mean())
out["tail"] = {m: {"worst_event_delta": float(d_rep(m).max()), "best_event_delta": float(d_rep(m).min()),
                   "events_worse_by_more_than_0.1": int((d_rep(m) > 0.1).sum()), "events_better_by_more_than_0.1": int((d_rep(m) < -0.1).sum()),
                   "median_delta": float(np.median(d_rep(m)))} for m in PRIMARY}
# who loses: events made worse by more than 0.1 nat by the primary map
lose = d_rep("mlp*FS/abar") > 0.1
win = d_rep("mlp*FS/abar") < -0.1
out["losers_mlp_FS_ratio"] = {"n": int(lose.sum()), "side_leaking_fraction": float(leak[lose].mean()), "late_start_fraction": float((kfirst[lose] >= 1).mean()),
                              "median_theta": float(np.median(theta[lose])), "median_origin_displacement_mm": float(np.median(evd["origin_displacement_mm"][lose])),
                              "sum_of_their_deltas_per_event_of_all": float(d_rep("mlp*FS/abar")[lose].sum() / N)}
out["winners_mlp_FS_ratio"] = {"n": int(win.sum()), "side_leaking_fraction": float(leak[win].mean()), "median_theta": float(np.median(theta[win])),
                               "sum_of_their_deltas_per_event_of_all": float(d_rep("mlp*FS/abar")[win].sum() / N)}
out["all_events_reference"] = {"side_leaking_fraction": float(leak.mean()), "late_start_fraction": float((kfirst >= 1).mean()),
                               "median_theta": float(np.median(theta)), "median_origin_displacement_mm": float(np.median(evd["origin_displacement_mm"]))}
# ---- T9 guard G-K (own table) ---------------------------------------------------------------------------------------------------
gk_file = L.ARR / "val_k1024.npz"
if gk_file.exists():
    gk = np.load(gk_file)
    md.append("### T9 Guard G-K: the same maps at K = 1024 (own fold points; a guard metric, not the report metric); margin = Δ(K=1024) − 0.5·Δ(K=64); 1e-4 nat\n")
    md.append("| map | Δ at K = 1024 | margin ± SE | verdict |")
    md.append("|---|---|---|---|")
    T9 = {"base_mean_K1024": {m: float(gk[f"{m}|nll"].mean()) for m in ("mlp", "formula")},
          "base_mean_K64": {m: float(rep[f"{m}|nll"].mean()) for m in ("mlp", "formula")}}
    for key in gk.files:
        m = key[:-4]
        if not key.endswith("|nll") or m in ("mlp", "formula"):
            continue
        b = base_of(m)
        g = gk[key] - gk[f"{b}|nll"]
        dec = stats.guard_persistence(d_rep(m), g)
        T9[m] = {"delta_K1024": P(g), "margin": dec.numbers["margin"], "se_margin": dec.numbers["se_margin"], "passed": bool(dec.passed),
                 "main_is_gain": bool(dec.checks["main_is_gain"])}
        verdict = ("pass" if dec.passed else "fail") if dec.checks["main_is_gain"] else "no gain to persist"
        md.append(f"| {m} | {f(T9[m]['delta_K1024'])} | {dec.numbers['margin'] * 1e4:+.1f} ± {dec.numbers['se_margin'] * 1e4:.1f} | {verdict} |")
    out["T9_guard_K"] = T9
    md.append("")
(L.HERE / "g6_tables.json").write_text(json.dumps(out, indent=1) + "\n")
(L.HERE / "g6_tables.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(json.dumps(out["base"], indent=1))
print(json.dumps({k: out[k] for k in ("losers_mlp_FS_ratio", "winners_mlp_FS_ratio", "all_events_reference", "mean_grid_acceptance", "side_leaking_fraction")}, indent=1))
print(json.dumps(out.get("T9_guard_K", {}).get("base_mean_K1024")), json.dumps(out.get("T9_guard_K", {}).get("base_mean_K64")))
