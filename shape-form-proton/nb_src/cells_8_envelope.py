# %% [markdown]
# ## 7. Shapes in the sense of the report (Table 2): mean-shower regions and envelope bodies
#
# The report compares two kinds of 3D shapes, per 1 GeV bin:
#
# - **r95**: the smallest region of a model's mean map that holds 95 % of the energy (`iso_radius`), with the report's volume
#   $V = \sum \pi r_{95}(z')^2 \Delta z'$ and coverage $c$ = fraction of test showers with ≥ 95 % of their energy inside;
# - **98 % bodies**: the smallest body of revolution holding ≥ 95 % of the energy in 98 % of the showers (envelope 02, drop 03).
#
# `scripts/particle_shapes.py` recomputes the report's shapes (they reproduce its Table 2) and adds: the r95 region of the new formula,
# the drop refitted with a multi-start search, the **super-egg** (two half super-ellipsoids $r = A(1-|\tau|^n)^{1/m}$ with different
# front and back lengths), the **calibrated iso-density body** of the new formula (the report's recommended next step), the bodies
# **smoothed in $\ln p$** and validated per bin (the report's limitation 3), the volume inside the data domain and **inside the detector
# slab** (the part in front of the tilted entrance face never holds a deposit), and the mean energy fraction inside ($E_{in}$; an r95
# region should hold 0.95 of the actual energy).

# %%
SH = pd.DataFrame(json.load(open(f"{FINAL}/shapes.json"))["shapes"]).set_index("shape")
ORDER = ["r95 baseline (report)", "r95 MLP (report)", "r95 MLP-formula (report)", "r95 new formula", "98% envelope (report)",
         "98% drop (report)", "98% drop (refit)", "98% drop (smoothed in ln p)", "98% superegg (refit)", "98% superegg (smoothed in ln p)",
         "98% iso-density (new formula)"]
tab = SH.loc[[o for o in ORDER if o in SH.index], ["V", "V_in", "V_slab", "cov", "cov_min", "E_in"]]
tab.columns = ["V (report def.) [L]", "V in data domain [L]", "V in detector slab [L]", "coverage", "min coverage (bin)", "E_in"]
display(tab.round(3))
e1 = pd.read_csv("lanes/E1_bodies/results/table_E95_EV98.csv").set_index("family")
print(f"non-parametric bound (ln r piecewise linear, 12 knots; lanes/E1_bodies): V in domain {e1.loc['knots12', 'V_in_L']:.0f} L, "
      f"coverage {e1.loc['knots12', 'cov_test']:.3f}")
iso90 = pd.read_csv("results/isobody/isobody_E90_EV90.csv")
for fam_ in ("knots12", "superegg", "drop"):
    print(f"criterion 90 % / 90 %: {fam_:9s} V in domain {iso90[iso90.body == fam_]['V_grid_L'].mean():6.1f} L, "
          f"slab {iso90[iso90.body == fam_]['V_slab_L'].mean():6.1f} L, coverage {iso90[iso90.body == fam_]['cov_test'].mean():.3f}")

# %% [markdown]
# **What to look for.**
# - **r95.** The new formula's region is close to the network's and holds 0.95 of the actual test energy ($E_{in}$). The report's
#   coverage $c$ is not a quality measure of an r95 region: it counts showers, and it spreads every hit uniformly over its 48 mm cell
#   while the region lives in the de-convolved space.
# - **98 % bodies.** The report's drop is close to optimal: the super-egg is ≈ 3.5 % smaller inside the data domain and the
#   non-parametric bound ≈ 4.4 %. Every good body fills most of the domain ($r_{max} \approx$ 650–700 mm against the cut $r < 700$ mm):
#   the 2 % most extreme showers decide (a hadron shower holds 95 % of its energy within ≈ 1 $\lambda_I$ only on average). With
#   90 % / 90 % the volumes fall by a factor ≈ 3 and the ranking is the same.
# - **Calibrated iso-density** (the report's recommendation) is not smaller than the drop: the criterion is set by atypical showers,
#   not by the mean shape.
# - **Smoothing in ln p**: the drop's parameters can be smoothed (coverage stays ≥ 0.955 in every bin); the six-parameter super-egg
#   loses coverage in some bins.
# - **Inside the detector slab** every body occupies 8–11 % less volume.

# %%
import importlib.util
spec = importlib.util.spec_from_file_location("e1fam", "lanes/E1_bodies/fam.py")
e1fam = importlib.util.module_from_spec(spec); spec.loader.exec_module(e1fam)
E1REG = e1fam.registry()


def body_fit(path):
    d = json.load(open(path))
    return d["x"], d["A"]


def env_figure(bins_show=(1, 5, 10, 18), r_show=750):
    vol = np.exp(LOG_VOL.cpu().numpy()); ze_ = Z_EDGES.cpu().numpy(); re_ = R_EDGES.cpu().numpy()
    h = data["train"].hits()
    zz = np.linspace(-260, 1500, 1200)
    fig, axes = plt.subplots(1, len(bins_show), figsize=(4.2 * len(bins_show), 3.3))
    for ax, b in zip(axes, bins_show):
        m = (h.p >= b) & (h.p < b + 1)
        H, _, _ = np.histogram2d(h.z[m], h.r[m], bins=[ze_, re_], weights=h.w[m])
        d = H / max(h.ev[m].nunique(), 1) / vol
        ax.pcolormesh(ze_, re_, np.log10(np.clip(d.T, 1e-14, None)), cmap="Greys", vmin=np.log10(d.max()) - 6, vmax=np.log10(d.max()))
        for key, path, col, ls, lab in (("drop", f"{FINAL}/bodies_E95_EV98/drop_bin{b:02d}.json", "#1f77b4", "--", "drop (report 03, refit)"),
                                        ("superegg", f"{FINAL}/bodies_E95_EV98/superegg_bin{b:02d}.json", "#008300", "-", "super-egg"),
                                        ("knots12", f"lanes/E1_bodies/results/fits/E95_EV98/knots12/bin{b:02d}.json", "#6250d6", ":", "non-parametric (12 knots)")):
            if os.path.exists(path):
                x, A = body_fit(path)
                ax.plot(zz, E1REG[key][0].radius(zz, x, A), ls, color=col, lw=1.8, label=lab)
        ax.axhline(700, color="0.4", lw=0.8, ls=":"); ax.set(xlim=(-260, 1500), ylim=(0, r_show), xlabel="z' [mm]", title=f"{b}–{b + 1} GeV")
    axes[0].set_ylabel("r [mm]"); axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("98 % envelope bodies on the data density (proton)", fontsize=10)
    plt.tight_layout(); plt.show()


env_figure()
