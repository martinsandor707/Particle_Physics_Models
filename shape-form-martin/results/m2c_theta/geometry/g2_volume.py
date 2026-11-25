"""Geometry audit, step 2 - the instrumented volume from TRAINING hits: remaining checks and figures.

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g2_volume.py

Inputs: arrays/g1_train_cells.npz (training rows per (column, layer) cell and the first training event that hits
each cell; written by g1_origins.py from the TRAINING rows of the Parquet file) and the training cache hits with
their absolute origins. CPU only; no validation event, no model.
"""
import json
import time

import glib as L
from glib import G, np, torch

t0 = time.time()
c = np.load(L.ARR / "g1_train_cells.npz")
ux, uy, occ, first = c["ux"], c["uy"], c["occ"], c["first_event"]
NC, NL = occ.shape
N_TR = 69783
out = {"split": "train", "columns": int(NC), "layers": int(NL), "rows": int(occ.sum())}

# ---- saturation: distinct cells / columns seen in the first n training events --------------------------------------------
grid = [100, 300, 1000, 3000, 10000, 20000, 30000, 40000, 50000, 60000, N_TR]
fc = np.sort(first.ravel())
fcol = np.sort(first.min(1))
out["saturation"] = {"n_events": grid, "distinct_cells": [int(np.searchsorted(fc, n)) for n in grid],
                     "distinct_columns": [int(np.searchsorted(fcol, n)) for n in grid],
                     "cells_total": int(NC * NL), "last_new_column_at_event_index": int(fcol.max()),
                     "new_cells_in_last_10000_events": int((fc >= N_TR - 10000).sum() - (fc >= N_TR).sum())}
print(json.dumps(out["saturation"]), flush=True)

# ---- the never-hit cells against a Poisson expectation -------------------------------------------------------------------
# mu(cell) = mean occupancy of the neighbouring columns (within 80 mm, the cell's own column excluded) in the same layer
from scipy.spatial import cKDTree  # noqa: E402

tree = cKDTree(np.stack([ux, uy], 1))
nb = tree.query_ball_point(np.stack([ux, uy], 1), r=80.0)
mu = np.empty_like(occ, dtype=np.float64)
for i, lst in enumerate(nb):
    lst = [j for j in lst if j != i]
    mu[i] = occ[lst].mean(0)
empty = occ == 0
p0 = np.exp(-mu)
exp_empty = float(p0.sum())
var_empty = float((p0 * (1 - p0)).sum())
out["never_hit_cells"] = {"observed": int(empty.sum()), "by_layer": {str(l): int(empty[:, l].sum()) for l in range(NL) if empty[:, l].any()},
                          "expected_poisson_from_neighbours": exp_empty, "sd": var_empty ** 0.5,
                          "expected_by_layer_58_59": [float(p0[:, 58].sum()), float(p0[:, 59].sum())],
                          "neighbour_mean_occupancy_of_the_empty_cells": [float(v) for v in np.sort(mu[empty])],
                          "min_hits_per_cell_layers_0_to_55": int(occ[:, :56].min()),
                          "median_hits_per_cell_by_layer": {str(l): float(np.median(occ[:, l])) for l in (0, 10, 30, 50, 57, 58, 59)}}
print(json.dumps(out["never_hit_cells"]), flush=True)

# ---- pile-up test at the outer edge and at the hole: hits per column, edge column against its inner neighbours ----------
col = occ.sum(1).astype(np.float64)
ys = np.unique(np.round(uy, 3))
r1, r2, h1, h2 = [], [], [], []
for y in ys:
    m = np.abs(uy - y) < 1e-2
    o = np.argsort(ux[m])
    xs, cc = ux[m][o], col[m][o]
    r1 += [cc[0] / cc[1], cc[-1] / cc[-2]]
    r2 += [cc[1] / cc[2], cc[-2] / cc[-3]]
    g = np.flatnonzero(np.diff(xs) > 60)
    if len(g):                                   # hole row: columns next to the hole against the next ones outward
        i = g[0]
        h1 += [cc[i] / cc[i - 1], cc[i + 1] / cc[i + 2]]
        h2 += [cc[i - 1] / cc[i - 2], cc[i + 2] / cc[i + 3]]
# top / bottom edge: outermost cell row against the next
top = [col[np.abs(uy - ys[-1]) < 1e-2].mean() / col[np.abs(uy - ys[-2]) < 1e-2].mean(),
       col[np.abs(uy - ys[0]) < 1e-2].mean() / col[np.abs(uy - ys[1]) < 1e-2].mean()]
out["edge_occupancy"] = {"outer_x_edge: edge/next (median, min, max)": [float(np.median(r1)), float(np.min(r1)), float(np.max(r1))],
                         "outer_x_edge: next/next-but-one (median)": float(np.median(r2)),
                         "hole_edge: adjacent/next outward (median, min, max)": [float(np.median(h1)), float(np.min(h1)), float(np.max(h1))],
                         "hole_edge: next/next-but-one (median)": float(np.median(h2)),
                         "top_bottom_row_mean / next_row_mean": [float(v) for v in top],
                         "reading": "a coordinate clip would pile hits into the edge cells (ratio > 1 against a falling profile)"}
print(json.dumps(out["edge_occupancy"]), flush=True)

# ---- hits outside / volume without cells, for the three volume models ----------------------------------------------------
sets, extras, origin = L.load_cpu(("train",))
tr, ex, O = sets["train"], extras["train"], origin["train"]
ev = np.repeat(np.arange(len(tr)), np.diff(tr.off.numpy()))
gl = O[ev] + G.to_global(tr.xyz.numpy().astype(np.float64), ex["theta"][ev], ex["phi"][ev])
w = tr.w.numpy().astype(np.float64)
rng = np.random.default_rng(5)
P = rng.uniform(-2700, 2700, (4_000_000, 2))
truth = G.columns_inside(P[:, 0], P[:, 1], ux, uy)
cells_only = G.columns_inside(P[:, 0], P[:, 1], ux, uy, max_step=0.0)          # union of the 48.6 mm boxes (gaps open)
vm = {}
for kind in ("stepped", "octagon", "box"):
    v = G.Volume(kind)
    ins_c = v.inside_xy_np(gl[:, 0], gl[:, 1])
    q = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):                         # 2 x 2 quarter-cell centres: share of the fold box outside
            q = q + ~v.inside_xy_np(gl[:, 0] + sx * G.HALF_CELL / 2, gl[:, 1] + sy * G.HALF_CELL / 2)
    q = q / 4
    m = v.inside_xy_np(P[:, 0], P[:, 1])
    vm[kind] = {"hit_centres_outside": int((~ins_c).sum()), "energy_share_centres_outside": float((w * ~ins_c).sum() / len(tr)),
                "hits_with_part_of_fold_box_outside": int((q > 0).sum()), "energy_share_of_fold_box_outside": float((w * q).sum() / len(tr)),
                "area_m2": float(m.mean() * 5.4 ** 2), "share_of_model_area_without_cells": float((m & ~truth).sum() / m.sum()),
                "share_of_instrumented_area_missing_from_model": float((truth & ~m).sum() / truth.sum())}
out["volume_models_training_hits"] = {"n_hits": int(len(w)), "n_events": int(len(tr)), **vm,
                                      "instrumented_envelope_area_m2": float(truth.mean() * 5.4 ** 2),
                                      "share_of_envelope_in_module_gaps": float(1 - cells_only.sum() / truth.sum()),
                                      "z_range_of_hit_centres": [float(gl[:, 2].min()), float(gl[:, 2].max())]}
print(json.dumps(out["volume_models_training_hits"], indent=1), flush=True)
out["seconds"] = time.time() - t0
(L.HERE / "g2_volume.json").write_text(json.dumps(out, indent=1) + "\n")

# ---- figures ---------------------------------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"
ORANGE, AQUA = "#eb6834", "#1baf7a"
seq = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False})


def outline_xy():
    """Polygon of Volume('stepped') (outer staircase), from the module constants."""
    hw = np.asarray(G.X_LAST) - G.X_CENTRE + G.HALF_CELL
    xs, ysq = [], []
    for a in range(26):
        y0, y1 = a * G.MODULE_PITCH_Y, min((a + 1) * G.MODULE_PITCH_Y, G.Y_MAX)
        xs += [hw[a], hw[a]]
        ysq += [y0, y1]
    xs, ysq = np.array(xs), np.array(ysq)
    X = np.concatenate([xs, -xs[::-1], -xs, xs[::-1]]) + G.X_CENTRE
    Y = np.concatenate([ysq, ysq[::-1], -ysq, -ysq[::-1]])
    return np.append(X, X[0]), np.append(Y, Y[0])


def hole_xy():
    X, Y = [], []
    for a in (2, 1, 0):
        X += [G.HOLE_LEFT[a] + G.HALF_CELL] * 2
        Y += [(a + 1) * 101.0, a * 101.0]
    for a in (0, 1, 2):
        X += [G.HOLE_LEFT[a] + G.HALF_CELL] * 2
        Y += [-a * 101.0, -(a + 1) * 101.0]
    for a in (2, 1, 0):
        X += [G.HOLE_RIGHT[a] - G.HALF_CELL] * 2
        Y += [-(a + 1) * 101.0, -a * 101.0]
    for a in (0, 1, 2):
        X += [G.HOLE_RIGHT[a] - G.HALF_CELL] * 2
        Y += [a * 101.0, (a + 1) * 101.0]
    return np.append(X, X[0]), np.append(Y, Y[0])


ox, oy = outline_xy()
hx, hy = hole_xy()
b = G.BOX_HALF
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
for ax, (xl, yl, title, ms) in zip(axs, (((-2750, 2750), (-2750, 2750), "all 8,752 cell columns seen in training hits", 1.6),
                                         ((1000, 2750), (1000, 2750), "upper right corner: one module (2 cells) per module row", 14),
                                         ((-700, 500), (-600, 600), "beam hole, off-centre (144 missing columns)", 16))):
    sc = ax.scatter(ux, uy, c=col, s=ms, marker="s", cmap=seq, norm=LogNorm(vmin=col.min(), vmax=col.max()), linewidths=0)
    ax.plot(ox, oy, color=ORANGE, lw=1.6, label="Volume('stepped'): outline and hole")
    ax.plot(hx, hy, color=ORANGE, lw=1.6)
    ax.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], color=INK, lw=1.2, ls="--", label="audit box, |X|, |Y| ≤ 2624.4 mm")
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel("X [mm]")
axs[0].set_ylabel("Y [mm]")
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower left", bbox_to_anchor=(0.12, -0.04), ncol=2, frameon=False, fontsize=9)
cb = fig.colorbar(sc, ax=axs, fraction=0.02, pad=0.01)
cb.set_label("training hits per column")
cb.outline.set_visible(False)
fig.savefig(L.FIG / "volume_outline.png", dpi=140, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.6))
ns = np.unique(np.round(np.logspace(1, np.log10(N_TR), 60)).astype(int))
ax.plot(ns, [np.searchsorted(fc, n) / (NC * NL) for n in ns], color="#2a78d6", lw=2, label="(column, layer) cells")
ax.plot(ns, [np.searchsorted(fcol, n) / NC for n in ns], color=ORANGE, lw=2, label="columns")
ax.set_xscale("log")
ax.set_xlabel("training events used (cache order)")
ax.set_ylabel("fraction seen at least once")
ax.set_ylim(0, 1.02)
ax.grid(color="#e6e5e1", lw=0.6)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Saturation of the distinct cells seen in training hits", fontsize=9, loc="left")
fig.savefig(L.FIG / "saturation.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("done", time.time() - t0)
