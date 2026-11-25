"""Geometry audit, step 7 - figure for question (b): where the acceptance gain sits (validation, audit numbers).

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g7_figures.py

Reads arrays/val_report.npz, val_bulk_zb1221.npz, val_controls.npz, val_events.npz. CPU only.
"""
import glib as L
from glib import np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rep, blk, ctl, evd = (np.load(L.ARR / f) for f in ("val_report.npz", "val_bulk_zb1221.npz", "val_controls.npz", "val_events.npz"))
theta, zb = evd["theta"], evd["zb"]
INK, MUTED, SURF, GRID = "#0b0b0b", "#898781", "#fcfcfb", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False})
NB = 20
edges = np.quantile(theta, np.linspace(0, 1, NB + 1))
b = np.clip(np.searchsorted(edges[1:-1], theta, side="right"), 0, NB - 1)
xc = np.array([theta[b == i].mean() for i in range(NB)])


def prof(d):
    m = np.array([d[b == i].mean() for i in range(NB)]) * 1e4
    s = np.array([d[b == i].std(ddof=1) / np.sqrt((b == i).sum()) for i in range(NB)]) * 1e4
    return m, s


fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"wspace": 0.28})
ax = axs[0]
series = (("inputs (p, θ): θ-binned training-mean acceptance [diagnostic]", ctl["mlp*FS_theta/abar|nll"] - ctl["mlp|nll"], AQUA),
          ("inputs (p, θ, φ, z_b): straight-line origin [diagnostic]", ctl["mlp*FS_line/abar|nll"] - ctl["mlp|nll"], ORANGE),
          ("inputs (p, θ, φ, entry position): geometry arm", rep["mlp*FS/abar|nll"] - rep["mlp|nll"], BLUE))
for lab, d, col in series:
    m, s = prof(d)
    ax.fill_between(xc, m - s, m + s, color=col, alpha=0.18, lw=0)
    ax.plot(xc, m, color=col, lw=2, marker="o", ms=4, label=f"{lab}: {d.mean() * 1e4:+.0f}")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("incidence angle θ [rad] (20 equal-count bins)")
ax.set_ylabel("Δ report metric vs θ-blind MLP [10⁻⁴ nat]")
ax.set_title("Front + side acceptance, ratio form, benchmark MLP\nvalidation, n = 14,950; band: paired SE", fontsize=9, loc="left")
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.42), title="mean over all events [10⁻⁴ nat]", title_fontsize=8, alignment="left")

ax = axs[1]
zex = evd["axis_exit_stepped"]
depth = np.where(np.isfinite(zex), np.minimum(zex, 1450.0), 1500.0)
cuts = [0, 200, 400, 700, 1000, 1450.0001, 2000]
labs = ["<200", "200–400", "400–700", "700–1000", "1000–1450", "no exit\nbefore 1450"]
d1 = rep["mlp*FS/abar|nll"] - rep["mlp|nll"]
d2 = rep["mlp*FSB/abar|nll"] - rep["mlp|nll"]
x = np.arange(len(labs))
for j, (lab, d, col) in enumerate((("front + side", d1, BLUE), ("front + side + back", d2, ORANGE))):
    m, s, n = [], [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        k = (depth >= lo) & (depth < hi)
        m.append(d[k].mean() * 1e4)
        s.append(d[k].std(ddof=1) / np.sqrt(k.sum()) * 1e4)
        n.append(int(k.sum()))
    ax.bar(x + (j - 0.5) * 0.36, m, 0.34, yerr=s, color=col, label=lab, error_kw={"ecolor": INK, "lw": 0.8})
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{a}\nn = {b_}" for a, b_ in zip(labs, n)])
ax.set_xlabel("depth z′ [mm] at which the shower axis leaves the outline or enters the hole")
ax.set_ylabel("Δ report metric vs θ-blind MLP [10⁻⁴ nat]")
ax.set_title("Same map by exit depth of the shower axis\nerror bars: paired SE", fontsize=9, loc="left")
ax.grid(color=GRID, lw=0.6, axis="y")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.text(0.01, -0.08, "AUDIT numbers on the laptop's fold points (RTX 4060, launch grid 144, K = 64); not scores of record. "
         "Never to be compared with θ-blind deltas.", fontsize=8, color="#52514e")
fig.savefig(L.FIG / "gain_by_theta_and_exit.png", dpi=140, bbox_inches="tight")
print("ok")
