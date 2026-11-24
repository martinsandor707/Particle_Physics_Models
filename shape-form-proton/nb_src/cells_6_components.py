# %% [markdown]
# ### 6.3 The fitted formula
#
# Energy fractions of the components and the fitted range of the stopping track as functions of the momentum, and the
# shower's cross-section parameters at two depths. All curves come from the chosen 3D model; the parameter table follows.

# %%
M = NEW["slab"]
with torch.no_grad():
    pg = torch.logspace(math.log10(0.3), math.log10(20), 80, device=DEVICE)
    lw, comp, th = M.components(torch.log(pg))
    obs = OBS_MASK.float()                                         # energy fractions inside the observable region
    wv = [x.exp().flatten() * (c_.exp() * obs).sum((1, 2)) if x is not None else None for x, c_ in zip(lw, comp)]
    tot = sum(v for v in wv if v is not None)
    w = [(v / tot).cpu().numpy() if v is not None else None for v in wv]
    Rr = M.range_mm(th).flatten().cpu().numpy()
    T = (pg ** 2 / (torch.sqrt(pg ** 2 + 0.938272 ** 2) + 0.938272)).cpu().numpy()
pgn = pg.cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
COMP_NAMES = ["shower", "primary track", "entry spike"] + (["end-of-range blob"] if M.endblob else []) + (["far halo"] if M.far else [])
COMP_COLS = ["#1f77b4", "#ff7f0e", "#008300", "#d62728", "#7f7f7f"]
for lab, ww, col in zip(COMP_NAMES, w, COMP_COLS):
    if ww is not None:
        ax[0].plot(pgn, ww, color=col, lw=2, label=lab)
ax[0].set(xscale="log", xlabel="p [GeV]", ylabel="energy fraction", title="Component fractions (observable region)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
ax[1].plot(pgn, Rr, color="#ff7f0e", lw=2, label="fitted track end R(T)")
ax[1].set(xscale="log", yscale="log", xlabel="p [GeV]", ylabel="R [mm of z']", title="End of the primary track (range)"); ax[1].grid(alpha=0.25)
ax1b = ax[1].secondary_xaxis("top", functions=(lambda p: np.sqrt(np.clip(p, 1e-6, None) ** 2 + 0.938272 ** 2) - 0.938272,
                                               lambda t: np.sqrt(np.clip((np.clip(t, 1e-6, None) + 0.938272) ** 2 - 0.938272 ** 2, 1e-12, None))))
ax1b.set_xlabel("T [GeV]", fontsize=8)
zsel = [100.0, 600.0]
for k, zz_ in enumerate(zsel):
    zk = zz_ / 1000
    q = lambda k_: th[k_] * zk * zk if k_ in th else 0.0
    sig = torch.exp(th["log_sig"] + th["s_sig"] * zk + q("q_sig")).flatten().cpu().numpy()
    Rm = torch.exp(th["log_R"] + th["s_R"] * zk).flatten().cpu().numpy()
    lam = torch.exp(th["log_lam"] + th["s_lam"] * zk).flatten().cpu().numpy()
    ls = "-" if k == 0 else "--"
    ax[2].plot(pgn, sig, ls, color="#1f77b4", label=f"Gauss core σ, z'={zz_:.0f}")
    ax[2].plot(pgn, Rm, ls, color="#6250d6", label=f"Cauchy body R, z'={zz_:.0f}")
    ax[2].plot(pgn, lam, ls, color="#ff7f0e", label=f"halo λ, z'={zz_:.0f}")
ax[2].set(xscale="log", yscale="log", xlabel="p [GeV]", ylabel="[mm]", title="Cross-section scales"); ax[2].legend(fontsize=7, ncol=2); ax[2].grid(alpha=0.25)
plt.tight_layout(); plt.show()
display(Markdown(M.describe(f"New proton formula (3D likelihood, degree {CHOSEN['slab']})")))

# %% [markdown]
# **Reading the curves.** Below 1 GeV most of the energy is in the entry spike, the track and the end-of-range blob (a stopping
# proton); the shower takes over between 1 and 3 GeV and carries ≈ 80 % at high momentum. The track keeps 16–20 % above 5 GeV: the
# primary proton before its first nuclear interaction (fitted attenuation length ≈ 270 mm) together with the narrow core of the shower,
# which its small halo takes over – above 2 GeV this split is effective. The track end grows with the kinetic energy as a power law,
# $R \propto T^{k}$ with $k \approx 1.8$, the Bragg–Kleeman exponent of the proton range (1.77); as an effective length of the mean energy
# profile it is shorter than the CSDA range of a free proton. Below ≈ 0.6 GeV the split is effective as well: the proton stops in the
# first layers, and the spike and the end blob describe a short, wide deposit. The far halo stays below 1 %. In the shower's
# cross-section the Gaussian term widens strongly with depth (≈ 50 mm at $z' = 100$ mm, ≈ 300 mm at 600 mm), the Cauchy body has
# R ≈ 70–80 mm at 100 mm and the halo λ ≈ 200–260 mm.
#
# ### 6.4 Data vs model: profiles (3D metric)
#
# Deposits sampled from each model inside the event's slab and folded with its cell (the 3D analogue of `sample_folded` of 00),
# compared with the test hits in five momentum bins.

# %%
@torch.no_grad()
def sample_folded_slab(model, data_, M_=200, seed=7):
    """Deposits drawn from the model's body of revolution, kept only inside the event's slab (rejection), folded with the cell."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    out = []
    for b in data_.batches(128):
        n = len(b.lnp)
        cell = torch.multinomial(model(b.lnp).exp().reshape(n, -1), M_, True, generator=gen)
        iz, ir = cell // N_R, cell % N_R
        u = torch.rand(n, M_, 3, device=DEVICE, generator=gen)
        z = Z_EDGES[iz] + u[..., 0] * DZ
        r = torch.sqrt(R_EDGES[ir] ** 2 + u[..., 1] * (R_EDGES[ir + 1] ** 2 - R_EDGES[ir] ** 2))
        pts = torch.stack([r * torch.cos(6.2832 * u[..., 2]), r * torch.sin(6.2832 * u[..., 2]), z], -1)
        ct = b.R[:, 2, 2][:, None]
        zt = pts[..., 2] * ct - pts[..., 0] * torch.sqrt(1 - ct ** 2)
        keep = (zt >= -float(data_.box[2]) / 2) & (zt <= b.zb[:, None] * ct)
        box = (torch.rand(n, M_, 3, device=DEVICE, generator=gen) - 0.5) * data_.box
        pts = pts + torch.einsum("bij,bmj->bmi", b.R, box)
        w_ev = (1.0 / keep.sum(1).clamp_min(1)).float()[:, None].expand(-1, M_)
        out.append(pd.DataFrame({"z": pts[..., 2][keep].cpu().numpy(), "r": torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)[keep].cpu().numpy(),
                                 "p": b.p[:, None].expand(-1, M_)[keep].cpu().numpy(), "w": w_ev[keep].cpu().numpy()}))
    return pd.concat(out, ignore_index=True)


COLORS.update({"GFlash 3D refit": "#1f77b4", "new formula": "#008300", "MLP 3D": "#ff7f0e"})
hits_te = data["test"].hits()
SAMP = {"GFlash 3D refit": sample_folded_slab(REF["GFlash (3D refit)"], data["test"]),
        "new formula": sample_folded_slab(NEW["slab"], data["test"]),
        "MLP 3D": sample_folded_slab(REF["MLP fine-tuned (3D)"], data["test"])}
SHOW_BINS = (0, 1, 4, 9, 18)
profile_plot(hits_te, SAMP, "z", "Depth profile: test data vs models (inside the detector slab, cell-folded)")
profile_plot(hits_te, SAMP, "r", "Lateral profile: test data vs models")
