"""Geometry audit, step 3 - question (a): why did 1.35 % of events not fit the simple box?  TRAINING split only.

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g3_box135.py

Reproduces the test of results/m1_audit/checks/c_side.py (hit centre outside |X|, |Y| <= 2624.4 mm when the frame
origin is put on the straight line from the global origin) on the training events, repeats it with the true origin,
and attributes the flagged events. A second pass over the TRAINING rows of the Parquet file (filter on the training
event numbers) characterises the first hit layer, whose energy-weighted centroid is the origin. CPU only.
"""
import json
import time

import glib as L
from glib import G, common, np, torch

t0 = time.time()
sets, extras, origin = L.load_cpu(("train",))
tr, ex, O = sets["train"], extras["train"], origin["train"]
N = len(tr)
th, ph = ex["theta"].astype(np.float64), ex["phi"].astype(np.float64)
zb = tr.ev["zb"].numpy().astype(np.float64)
p = tr.ev["p"].numpy().astype(np.float64)
ev = np.repeat(np.arange(N), np.diff(tr.off.numpy()))
xyz = tr.xyz.numpy().astype(np.float64)
w = tr.w.numpy().astype(np.float64)
Lo = G.straight_line_origin(th, ph, zb)
off = G.to_global(xyz, th[ev], ph[ev])                 # R^T local, float64 angles
X_E = G.BOX_HALF
out = {"split": "train", "n_events": N, "n_hits_after_cuts": int(len(w)), "X_E_mm": X_E}

# ---- the audit's test, on training events ------------------------------------------------------------------------------
gL = Lo[ev] + off
gT = O[ev] + off
outside_L = (np.abs(gL[:, 0]) > X_E) | (np.abs(gL[:, 1]) > X_E)
outside_T = (np.abs(gT[:, 0]) > X_E) | (np.abs(gT[:, 1]) > X_E)
eL = np.bincount(ev, weights=w * outside_L, minlength=N)
eT = np.bincount(ev, weights=w * outside_T, minlength=N)
flag = eL > 0
out["straight_line_origin"] = {"events_with_any_hit_outside_box": int(flag.sum()), "fraction": float(flag.mean()),
                               "mean_energy_share_outside": float(eL.mean()), "se": float(eL.std(ddof=1) / np.sqrt(N)),
                               "hits_outside": int(outside_L.sum()),
                               "validation_reference_c_side": {"events": 202, "of": 14950, "fraction": 202 / 14950,
                                                               "mean_energy_share_outside": 0.0037909}}
out["true_origin"] = {"events_with_any_hit_outside_box": int((eT > 0).sum()), "hits_outside": int(outside_T.sum()),
                      "max_abs_X": float(np.abs(gT[:, 0]).max()), "max_abs_Y": float(np.abs(gT[:, 1]).max()),
                      "min_Z": float(gT[:, 2].min()), "max_Z": float(gT[:, 2].max())}
print(json.dumps(out["straight_line_origin"]), json.dumps(out["true_origin"]), flush=True)
# cache hits (float32 local coordinates) against the lattice: distance to the nearest training column
cols = np.load(L.HERE / "explore" / "train_columns.npz")
lat = G.columns_inside(gT[:, 0], gT[:, 1], cols["ux"], cols["uy"], half=0.01, max_step=0.0)
out["true_origin"]["hits_within_0.01mm_of_a_training_column_centre"] = float(lat.mean())

# ---- attribution -------------------------------------------------------------------------------------------------------
d = O[:, :2] - Lo[:, :2]
dr = np.hypot(d[:, 0], d[:, 1])
k = np.rint((O[:, 2] - G.Z_LAYER0) / G.LAYER).astype(int)
# nominal entry point: the straight line at the front layer centre, and whether it is inside the hole aperture
ent = np.stack([G.Z_LAYER0 * np.tan(th) * np.cos(ph), G.Z_LAYER0 * np.tan(th) * np.sin(ph)], 1)
vol = G.Volume()
in_hole = ~vol.inside_xy_np(ent[:, 0], ent[:, 1]) & G.Volume(hole=False).inside_xy_np(ent[:, 0], ent[:, 1])
# overshoot of the flagged hits beyond the box edge (straight-line frame) and the displacement along that direction
over = np.maximum(np.abs(gL[:, 0]) - X_E, np.abs(gL[:, 1]) - X_E)
max_over = np.zeros(N)
np.maximum.at(max_over, ev[outside_L], over[outside_L])
att = {"flagged": int(flag.sum()),
       "displacement_mm_of_flagged": {f"q{q}": float(np.quantile(dr[flag], q)) for q in (0.0, 0.01, 0.1, 0.5, 0.9, 1.0)},
       "displacement_mm_of_all": {f"q{q}": float(np.quantile(dr, q)) for q in (0.5, 0.9, 0.99, 1.0)},
       "flagged_with_displacement_gt": {str(c): int((dr[flag] > c).sum()) for c in (23.4, 50, 100, 200, 500)},
       "all_with_displacement_gt": {str(c): int((dr > c).sum()) for c in (23.4, 50, 100, 200, 500)},
       "flag_rate_by_displacement_bin": {}, "max_overshoot_mm_of_flagged": {f"q{q}": float(np.quantile(max_over[flag], q)) for q in (0.1, 0.5, 0.9, 1.0)},
       "flagged_with_overshoot_le_displacement": int((max_over[flag] <= np.maximum(np.abs(d[flag, 0]), np.abs(d[flag, 1])) + 1e-6).sum())}
edges = [0, 23.4, 50, 100, 200, 500, 1e9]
for a, b in zip(edges[:-1], edges[1:]):
    m = (dr >= a) & (dr < b)
    att["flag_rate_by_displacement_bin"][f"[{a:g},{b:g})"] = {"events": int(m.sum()), "flagged": int((flag & m).sum()),
                                                              "energy_outside_sum_share": float(eL[m].sum() / max(eL.sum(), 1e-300))}
cls = np.where(k >= 1, "late start (first hit layer k >= 1)", np.where(dr > 100, "k = 0, origin displaced > 100 mm",
               np.where(dr > 23.4, "k = 0, displaced 23.4 - 100 mm", "k = 0, displaced < 23.4 mm")))
att["classes"] = {}
for c in np.unique(cls):
    m = cls == c
    att["classes"][c] = {"events": int(m.sum()), "flagged": int((flag & m).sum()), "share_of_flagged": float((flag & m).sum() / flag.sum()),
                         "flag_rate": float(flag[m].mean()), "share_of_energy_outside": float(eL[m].sum() / eL.sum()),
                         "nominal_entry_in_hole_among_flagged": int((flag & m & in_hole).sum())}
att["nominal_entry_in_hole"] = {"events": int(in_hole.sum()), "flagged": int((flag & in_hole).sum())}
# energy outside per flagged event
att["energy_share_outside_of_flagged"] = {f"q{q}": float(np.quantile(eL[flag], q)) for q in (0.1, 0.5, 0.9, 0.99, 1.0)}
att["flagged_with_more_than_half_of_energy_outside"] = int((eL > 0.5).sum())
# where would the straight-line hits fall for the correct outline (corners and hole are not instrumented)?
out_step_L = ~vol.inside_xy_np(gL[:, 0], gL[:, 1])
att["straight_line_origin_vs_stepped_volume"] = {"events_with_hit_centre_outside": int((np.bincount(ev, weights=out_step_L, minlength=N) > 0).sum()),
                                                 "mean_energy_share_outside": float(np.bincount(ev, weights=w * out_step_L, minlength=N).mean())}
out_step_T = ~vol.inside_xy_np(gT[:, 0], gT[:, 1])
att["true_origin_vs_stepped_volume"] = {"hits_outside": int(out_step_T.sum())}
out["attribution"] = att
print(json.dumps(att, indent=1), flush=True)

# ---- the first hit layer, from all TRAINING rows (second pass over the Parquet file, training filter) ------------------
acc = {k_: np.zeros(N) for k_ in ("n", "e", "ex", "ey", "er2", "e_late", "n_late", "emax")}


def hook(index, c):
    z = c["z"]
    first = np.abs(z - O[index, 2]) < 1.0
    if not first.any():
        return
    i, e = index[first], c["energy"][first].astype(np.float64)
    x, y, t = c["x"][first], c["y"][first], c["earliest_contribution_time"][first]
    acc["n"] += np.bincount(i, minlength=N)
    acc["e"] += np.bincount(i, weights=e, minlength=N)
    acc["ex"] += np.bincount(i, weights=e * x, minlength=N)
    acc["ey"] += np.bincount(i, weights=e * y, minlength=N)
    acc["er2"] += np.bincount(i, weights=e * (x * x + y * y), minlength=N)
    late = t >= common.TIME_CUT_NS
    acc["e_late"] += np.bincount(i, weights=e * late, minlength=N)
    acc["n_late"] += np.bincount(i, weights=late.astype(np.float64), minlength=N)
    np.maximum.at(acc["emax"], i, e)


res = G.origins_from_parquet(ex["event_number"], row_hook=hook, extra_columns=("energy", "earliest_contribution_time"))
assert np.array_equal(res["event_number"], ex["event_number"]) and np.abs(res["origin"] - O).max() < 1e-9
cx, cy = acc["ex"] / acc["e"], acc["ey"] / acc["e"]
cen = np.hypot(cx - O[:, 0], cy - O[:, 1])
rms = np.sqrt(np.clip(acc["er2"] / acc["e"] - cx ** 2 - cy ** 2, 0, None))
late_share = acc["e_late"] / acc["e"]
fl = {"energy_weighted_centroid_of_first_hit_layer_equals_origin": {"max_abs_mm": float(cen.max()), "events_gt_1e-6mm": int((cen > 1e-6).sum())},
      "rows_in_first_hit_layer": {f"q{q}": float(np.quantile(acc["n"], q)) for q in (0.5, 0.9, 0.99, 1.0)},
      "single_row_first_layer": int((acc["n"] == 1).sum()),
      "groups": {}}
groups = {"all": np.ones(N, bool), "k=0, displaced <= 100 mm": (k == 0) & (dr <= 100), "k=0, displaced > 100 mm": (k == 0) & (dr > 100),
          "k>=1": k >= 1, "flagged by the box test": flag}
for name, m in groups.items():
    fl["groups"][name] = {"events": int(m.sum()), "median_rows_in_first_layer": float(np.median(acc["n"][m])),
                          "single_row_fraction": float((acc["n"][m] == 1).mean()),
                          "median_rms_radius_mm": float(np.median(rms[m])), "frac_rms_gt_100mm": float((rms[m] > 100).mean()),
                          "mean_late_energy_share_first_layer": float(late_share[m].mean()),
                          "frac_first_layer_all_late": float((late_share[m] >= 1 - 1e-12).mean()),
                          "frac_first_layer_any_late": float((acc["n_late"][m] > 0).mean()),
                          "frac_leading_hit_below_half_of_layer_energy": float((acc["emax"][m] < 0.5 * acc["e"][m]).mean()),
                          "median_p_GeV": float(np.median(p[m])), "median_theta": float(np.median(th[m])),
                          "nominal_entry_in_hole_fraction": float(in_hole[m].mean())}
out["first_hit_layer_all_training_rows"] = fl
print(json.dumps(fl, indent=1), flush=True)
np.savez_compressed(L.ARR / "g3_train_box.npz", event_number=ex["event_number"], energy_outside_line=eL, flagged=flag,
                    displacement_mm=dr, first_hit_layer=k, nominal_entry_in_hole=in_hole, first_layer_rows=acc["n"],
                    first_layer_rms_mm=rms, first_layer_late_share=late_share)
L.sidecar(L.ARR / "g3_train_box.npz", split="train", n_events=N, note="question (a); per-event arrays in cache order; no model, no fold points")
out["seconds"] = time.time() - t0
(L.HERE / "g3_box135.json").write_text(json.dumps(out, indent=1) + "\n")
print("done", out["seconds"])
