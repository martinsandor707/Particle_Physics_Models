# %% [markdown]
# ## 3. Geometry: the detector slab in the shower frame
#
# The shower frame is obtained from the detector frame by a translation to the origin (the energy-weighted centroid of the hits in the
# first hit layer, $z_0 = \min z$ of the event) and the rotation $R$ whose rows are $e_\theta$, $e_\varphi$, $e_r$ (`rotation` in 00).
# Inverting it, the depth of a point **along the detector axis** is
#
# $$z_{trans} = z'\cos\theta - x'\sin\theta .$$
#
# By construction no hit can have $z_{trans} < 0$, and the last layer ends at $D = z_{back} - z_0$. With the half layer thickness $h$
# of the cell boxes, every recorded deposit lies in the **slab**
#
# $$-h \;\le\; z'\cos\theta - x'\sin\theta \;\le\; D .$$
#
# In cylindrical coordinates around the shower axis, $x' = r\cos\phi$, so a ring $(z', r)$ is inside the slab for
# $\cos\phi \in \big[\tfrac{z'\cos\theta - D}{r\sin\theta},\; \tfrac{z'\cos\theta + h}{r\sin\theta}\big]$ and the accepted fraction of its azimuth is
#
# $$A_\theta(z', r) = \frac{1}{\pi}\Big[\arccos\big(\mathrm{clip}(\ell)\big) - \arccos\big(\mathrm{clip}(u)\big)\Big],\qquad
# \ell = \frac{z'\cos\theta - D}{r\sin\theta},\quad u = \frac{z'\cos\theta + h}{r\sin\theta},$$
#
# with both arguments clipped to $[-1, 1]$. There is **no free parameter**: $\theta$ and $D$ are known for every event (they are already
# used by the likelihood of 00 for the rotated cell box and for the back-face distance $z_b = D/\cos\theta$).

# %%
def slab_accept(cos_t, sin_t, D, half, nsub=2, front=True, back=True):
    """Azimuthal acceptance A (B, N_Z, N_R): fraction of every (z', r) ring inside the detector slab of each event.
    Each grid cell is sampled at nsub x nsub points (equal-area in r)."""
    k = (torch.arange(nsub, device=cos_t.device, dtype=torch.float32) + 0.5) / nsub
    zs = Z_EDGES[:-1, None] + DZ * k                                                                      # (N_Z, nsub)
    rs = torch.sqrt(R_EDGES[:-1, None] ** 2 + (R_EDGES[1:, None] ** 2 - R_EDGES[:-1, None] ** 2) * k)     # (N_R, nsub)
    zc = zs[None, :, :, None, None] * cos_t[:, None, None, None, None]
    den = (rs[None, None, None, :, :] * sin_t[:, None, None, None, None]).clamp_min(1e-9)
    lo = ((zc - D[:, None, None, None, None]) / den).clamp(-1, 1) if back else -torch.ones_like(zc / den)
    hi = ((zc + half) / den).clamp(-1, 1) if front else torch.ones_like(zc / den)
    return ((torch.acos(lo) - torch.acos(hi)) / math.pi).mean((2, 4))


def event_slab(b, box, nsub=2, front=True, back=True):
    cos_t = b.R[:, 2, 2]
    sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
    return slab_accept(cos_t, sin_t, b.zb * cos_t, float(box[2]) / 2, nsub, front, back)


# check 1: every hit (and every Monte-Carlo point of its cell box) lies inside the slab
te = data["test"]
b = te.batch(torch.arange(len(te), device=DEVICE))
cos_t, half = b.R[:, 2, 2], float(te.box[2]) / 2
sin_t = torch.sqrt(1 - cos_t ** 2)
z_trans = b.xyz[:, 2] * cos_t[b.ev] - b.xyz[:, 0] * sin_t[b.ev]
print(f"hits: min z_trans = {float(z_trans.min()):.3f} mm, max (z_trans − D) = {float((z_trans - (b.zb * cos_t)[b.ev]).max()):.2f} mm (half layer = {half} mm)")

# check 2: the closed form against brute-force Monte Carlo (random points in every grid cell, 48 random events)
gen = torch.Generator(device=DEVICE).manual_seed(0)
sel = te.batch(torch.randperm(len(te), generator=torch.Generator().manual_seed(0))[:48].to(DEVICE))
ct = sel.R[:, 2, 2]; st = torch.sqrt(1 - ct ** 2); Dd = sel.zb * ct
u = torch.rand(4000, 3, device=DEVICE, generator=gen)
zz_ = Z_EDGES[:-1, None, None] + DZ * u[None, None, :, 0]
rr_ = torch.sqrt(R_EDGES[:-1, None] ** 2 + (R_EDGES[1:, None] ** 2 - R_EDGES[:-1, None] ** 2) * u[None, :, 1])[None]
cph = torch.cos(2 * math.pi * u[:, 2])[None, None]
A_mc = torch.stack([(((zz_ * ct[i] - rr_ * cph * st[i]) >= -half) & ((zz_ * ct[i] - rr_ * cph * st[i]) <= Dd[i])).float().mean(2) for i in range(len(ct))])
A_cf = slab_accept(ct, st, Dd, half)
print(f"closed form vs Monte Carlo: mean |ΔA| = {float((A_cf - A_mc).abs().mean()):.4f} (MC noise ≈ 0.008 per cell)")

# %% [markdown]
# ### The cut in the data
#
# Energy-weighted hit maps of the training showers ($p \ge 5$ GeV) for four ranges of the incidence angle.
# **Top:** side view $(z', x')$ for $|y'| < 60$ mm with the front face (dashed, $x' = z'/\tan\theta$) and the back face of the slab.
# **Bottom:** the cross-section $(x', y')$ just behind the entry point ($0 \le z' < 100$ mm): half of the disc is missing.
# At the largest angles the outer edge of the calorimeter cuts the shower on the $+x'$ side as well (see section 8).

# %%
tr = data["train"]
bt = tr.batch(torch.arange(len(tr), device=DEVICE))
th_ev = torch.acos(bt.R[:, 2, 2])
hx, hy, hz, hw = (bt.xyz[:, 0].cpu().numpy(), bt.xyz[:, 1].cpu().numpy(), bt.xyz[:, 2].cpu().numpy(), bt.w.cpu().numpy())
h_th, h_p = th_ev[bt.ev].cpu().numpy(), bt.p[bt.ev].cpu().numpy()
D_ev = (bt.zb * bt.R[:, 2, 2])
fig, axes = plt.subplots(2, 4, figsize=(17, 7.2))
for j, (t0, t1) in enumerate(((0.06, 0.2), (0.3, 0.4), (0.45, 0.55), (0.58, 0.63))):
    m = (h_th >= t0) & (h_th < t1) & (h_p >= 5)
    tm = 0.5 * (t0 + t1)
    mm = m & (np.abs(hy) < 60)
    H, xe, ye = np.histogram2d(hz[mm], hx[mm], bins=[np.linspace(-200, 1450, 166), np.linspace(-700, 700, 141)], weights=hw[mm])
    ax = axes[0, j]
    ax.pcolormesh(xe, ye, H.T + 1e-12, norm=matplotlib.colors.LogNorm(vmin=H.max() * 1e-5, vmax=H.max()), cmap="magma")
    zq = np.linspace(-200, 1450, 50)
    Dm = float(D_ev[(th_ev >= t0) & (th_ev < t1)].median())
    ax.plot(zq, zq / math.tan(tm), "c--", lw=1.1, label="front face"); ax.plot(zq, (zq * math.cos(tm) - Dm) / math.sin(tm), "w--", lw=1.1, label="back face")
    ax.set_ylim(-700, 700); ax.set_title(f"θ = {t0}–{t1} rad: side view", fontsize=9); ax.set_xlabel("z' [mm]")
    mm = m & (hz >= 0) & (hz < 100)
    H, xe, ye = np.histogram2d(hx[mm], hy[mm], bins=[np.linspace(-500, 500, 81), np.linspace(-500, 500, 81)], weights=hw[mm])
    ax = axes[1, j]
    ax.pcolormesh(xe, ye, H.T + 1e-12, norm=matplotlib.colors.LogNorm(vmin=H.max() * 1e-5, vmax=H.max()), cmap="magma")
    ax.set_aspect("equal"); ax.set_title("cross-section at 0 ≤ z' < 100 mm", fontsize=9); ax.set_xlabel("x' [mm]")
axes[0, 0].set_ylabel("x' [mm]"); axes[1, 0].set_ylabel("y' [mm]"); axes[0, 0].legend(fontsize=7, loc="lower right", facecolor="0.3", labelcolor="w")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### The event-averaged acceptance $\bar A(z', r)$
#
# Averaging $A_\theta$ over the incidence angles of the sample gives a fixed map $\bar A(z', r)$. A body of revolution multiplied by
# $\bar A$ is the best *pure* $(z', r \mid p)$ description of what the detector records: this is the cone-shaped front seen in the
# network maps of notebooks 05/06. It is computed from the training events only and contains no fitted number.

# %%
@torch.no_grad()
def mean_acceptance(sh, nsub=4, front=True, back=True):
    tot, n = 0.0, 0
    for bb in sh.batches(256):
        A = event_slab(bb, sh.box, nsub, front, back)
        tot, n = tot + A.sum(0), n + len(A)
    return tot / n


A_BAR = mean_acceptance(data["train"])
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
im = ax[0].pcolormesh(Z_EDGES.cpu().numpy(), R_EDGES.cpu().numpy(), A_BAR.T.cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
ax[0].set(title="Event-averaged acceptance  Ā(z', r)", xlabel="z' [mm]", ylabel="r [mm]"); plt.colorbar(im, ax=ax[0], label="accepted fraction of the ring")
zc_np = ZC.cpu().numpy()
for r_sel, ls in ((30, "-"), (100, "--"), (250, "-."), (500, ":")):
    ir = int(torch.argmin((RC - r_sel).abs()))
    ax[1].plot(zc_np, A_BAR[:, ir].cpu().numpy(), ls, color="#1f77b4", label=f"r ≈ {r_sel} mm")
ax[1].set(title="Ā along the depth", xlabel="z' [mm]", ylabel="Ā", xlim=(-200, 1450)); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
plt.tight_layout(); plt.show()
