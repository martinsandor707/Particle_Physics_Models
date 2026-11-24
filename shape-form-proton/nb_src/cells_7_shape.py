# %% [markdown]
# ### 6.5 The 3D shape
#
# **Top:** the true (de-convolved) density of the new formula as a body of revolution, $\log_{10}\rho$ in $(z', \pm r)$ (as
# `shape_map` in 01–06), restricted to the observable region (grey: cells that no event's slab covers). **Middle:** the same body cut
# by the detector planes for a typical incidence angle ($\theta = 0.45$ rad), side view $(z', x')$ through the axis: this is the 3D
# shape the detector records. **Bottom:** the components at 1 GeV.

# %%
P_SHOW = (0.8, 2.0, 5.0, 15.0)
with torch.no_grad():
    lnp_s = torch.log(torch.tensor(P_SHOW, device=DEVICE))
    dens = (SHAPE(lnp_s) - LOG_VOL).exp().cpu().numpy()
zc_np, rc_np = ZC.cpu().numpy(), RC.cpu().numpy()
xs = np.linspace(-500, 500, 201)
fig, axes = plt.subplots(2, 4, figsize=(17, 6.2))
TH = 0.45
for j, p in enumerate(P_SHOW):
    lin = np.stack([np.interp(np.abs(xs), rc_np, row) for row in dens[j]], 1)
    img = np.where(lin > 1e-25, np.log10(np.clip(lin, 1e-14, None)), np.nan)
    vmax = np.nanmax(img)
    axes[0, j].imshow(img, origin="lower", aspect="auto", extent=[Z_MIN, Z_MAX, -500, 500], cmap="magma", vmin=vmax - 5, vmax=vmax)
    axes[0, j].set_facecolor("0.85")
    axes[0, j].set_title(f"new formula, p = {p:g} GeV: body of revolution", fontsize=9)
    Zg, Xg = np.meshgrid(zc_np, xs, indexing="ij")
    cut = (Zg * np.cos(TH) - Xg * np.sin(TH)) >= 0
    axes[1, j].imshow(np.where(cut.T, img, np.nan), origin="lower", aspect="auto", extent=[Z_MIN, Z_MAX, -500, 500], cmap="magma",
                      vmin=vmax - 5, vmax=vmax)
    axes[1, j].set_facecolor("0.85"); axes[1, j].set_title(f"cut by the front face (θ = {TH})", fontsize=9)
    for a_ in axes[:, j]:
        a_.set_xlabel("z' [mm]"); a_.set_xlim(-200, 1300)
axes[0, 0].set_ylabel("r [mm] (mirrored)"); axes[1, 0].set_ylabel("x' [mm]")
plt.tight_layout(); plt.show()

with torch.no_grad():
    lw, comp, _ = NEW["slab"].components(torch.log(torch.tensor([1.0], device=DEVICE)))
    obs = OBS_MASK.float()
    wv = [float(x.exp()) * float((c_[0].exp() * obs).sum()) for x, c_ in zip(lw, comp)]
fig, axes = plt.subplots(1, len(comp), figsize=(4 * len(comp), 2.8))
for a_, c_, t_, w_ in zip(axes, comp, COMP_NAMES, wv):
    d = ((c_[0].exp() * obs) / (c_[0].exp() * obs).sum() / LOG_VOL.exp()).cpu().numpy()
    img = np.log10(np.clip(np.stack([np.interp(np.abs(xs), rc_np, row) for row in d], 1), 1e-14, None))
    a_.imshow(img, origin="lower", aspect="auto", extent=[Z_MIN, Z_MAX, -500, 500], cmap="magma", vmin=img.max() - 5, vmax=img.max())
    a_.set(title=f"{t_}: {w_ / sum(wv):.1%} (p = 1 GeV)", xlabel="z' [mm]", xlim=(-200, 800))
axes[0].set_ylabel("r [mm] (mirrored)")
plt.tight_layout(); plt.show()

# %% [markdown]
# The 95 %-energy bodies of the mean shower (`bodies_3d` of 00), new formula vs the report's GFlash and MLP:

# %%
bodies_3d({"new formula": SHAPE, "GFlash (report)": REPORT["gflash_n6"], "MLP (report)": REPORT["mlp"]},
          p_values=(1.0, 4.5, 14.5), fractions=(0.5, 0.9, 0.95), title="50 / 90 / 95 % bodies of the mean (de-convolved) shower")
