# %% [markdown]
# ## 5. The model: shower + primary track + entry spike + end-of-range blob + far halo
#
# With the geometry taken out, the remaining deficit of the GFlash formula is concentrated at low momentum. A proton (mass
# $m_p = 0.938$ GeV) differs from a pion here: its kinetic energy $T = \sqrt{p^2 + m_p^2} - m_p$ is small (30 MeV at $p = 0.24$ GeV,
# 0.43 GeV at 1 GeV, 1.27 GeV at 2 GeV), and below about 1.5 GeV it mostly **stops by ionisation** – a short track ending
# at its range – instead of starting a hadronic shower. In addition the origin of the shower frame is the centroid of the first hit
# layer, so every event has energy at $(z', r) \approx (0, 0)$: the first layer holds 43 % of the visible energy below 1 GeV and
# 2.5–5 % above 5 GeV.
#
# The new formula is a sum of physically distinct components, each normalised on the grid, with energy fractions $w_k(p)$:
#
# $$\rho(z', r \mid p) = w_{sh}\, S + w_{tr}\, U + w_{sp}\, D + w_{end}\, E + w_{far}\, H .$$
#
# - **Shower** $S = L(z')\,T(r \mid z')$. Depth: a plateau profile $L \propto \sigma\!\big(\tfrac{z'-z_s}{w_1}\big)\,\sigma\!\big(\tfrac{\ell-z'}{w_2}\big)\,e^{\kappa z'}$
#   (fast rise at the shower start $z_s$, exponential fall-off, cut-off at $\ell$; chosen on validation over the GFlash gamma profile
#   and over the Bock two-gamma profile of the literature). Cross-section: three terms,
#   $$T = f_1\,\frac{e^{-r^2/2\sigma^2}}{2\pi\sigma^2} + f_2\,\frac{1}{\pi}\frac{R^2}{(r^2+R^2)^2} + f_3\,\frac{e^{-r/\lambda}}{2\pi\lambda r},$$
#   a narrow electromagnetic core, a Cauchy body and a halo with an **exponential $dE/dr$** (the data are an almost perfect
#   exponential in $dE/dr$ between 50 and 350 mm). Widths and fractions vary with depth (linear terms with momentum-dependent slopes,
#   plus quadratic terms for the core width and the fractions). The cross-section is normalised on the disc of the grid ($r < 700$ mm)
#   at every depth, so that the depth profile is set by $L(z')$ alone: normalised over the infinite plane, a very wide halo whose share
#   grows with depth would remove energy from the grid and act as a hidden depth-dependent sink.
# - **Primary track** $U$: a tube on the axis that starts behind the first layer, is attenuated as $e^{-z'/\lambda_{att}}$ (the
#   primary survives until its first nuclear interaction) and ends at the **range** $R(T) = R_1 T^{k} - R_0$ (a Bragg–Kleeman-type
#   range–energy law: low-momentum protons stop), with the Bragg–Kleeman rise of $dE/dx$ towards the end, $(R - z')^{-q}$. Its width
#   grows with depth by **multiple scattering** (Highland): $\sigma^2(z') = \sigma_0^2 + \big(c\,(z'/1\,\mathrm{m})^{k_{ms}}/\beta p\big)^2$;
#   a small halo $e^{-r/l}/(2\pi l r)$ around it carries the knock-on electrons.
# - **Entry spike** $D$: a small Gaussian blob at the origin (the first hit layer), position and widths depending on the momentum. For a
#   stopping proton the first layer is part of its track, so the spike fraction is tied to the track length: $f_1 = d_1/(d_1 + L_t)$.
# - **End-of-range blob** $E$: a Gaussian at $z' = R(T)$ – the Bragg peak and the products of the stopping particle.
# - **Far halo** $H \propto e^{-z'/L_f}\, e^{-r/R_f}/(r + 10\ \mathrm{mm})$ from the entry on: a diffuse deposit far from the axis
#   (neutrons and soft photons of the shower).
#
# Every raw parameter is a polynomial in $x = (\ln T - c)/s$ (the kinetic energy instead of $\ln p$; $x \in [-1, 1]$ over the
# training range). Parameters that need no energy dependence (depth slopes, track and spike shapes, the range law) are constants.
# The only input of the model is $\ln p$.
#
# The variant (gamma or plateau shower, with or without track and attenuation, Bock profile) and the additional terms (spike shape,
# end blob, far halo; multiple scattering, knock-on halo, Bragg rise, depth curvature) were chosen on the validation set at degree 2
# (`scripts/particle_pipeline.py`, `scripts/variants_v2.py`, `scripts/variants_v3.py`; the same protocol for all four particles, see
# `README.md`); the checkpoints below come from that run (`results/particles/proton/`).
#
# Fitting: the same likelihood (3D or the report's), Adam with a cosine-decaying learning rate (`shapelab.fit_fast`, batch 256,
# 4000 steps, best validation state kept). The degree $n$ is chosen as in notebook 01: train $n = 1, \dots, 6$ (each started from
# the previous solution), $n \le 6$ as for GFlash in 01, take the lowest validation NLL, and choose the smallest $n$ within one paired
# standard error of it. As for GFlash in 01, the rule picks the top rung: the paired SE between nested, warm-started rungs is tiny,
# while the gains beyond $n = 2$ are small. The degree-2 rung is therefore reported alongside.

# %%
from particle_shape import ParticleShape, MASS
from proton_shape import ProtonShape
import inspect
print(inspect.getsource(ParticleShape.log_shower))
print(inspect.getsource(ProtonShape.log_track))
print(inspect.getsource(ParticleShape.log_track))
print(inspect.getsource(ProtonShape.range_mm))
print(inspect.getsource(ParticleShape.components))

# %%
LNP0, LNP_S = shapelab.lnp_center_scale(data)
A_BAR_DEV = A_BAR.to(DEVICE)
FINAL = "results/particles/proton"                              # written by scripts/particle_pipeline.py proton
KW, _, _, TAG, PIPE_TEST, ROUND = shapelab.final_kwargs(FINAL)   # the variant chosen on validation (final round)
print("final formula:", PIPE_TEST["variant"], KW)


def make(n, metric):
    kw, degrees, *_ = shapelab.final_kwargs(FINAL, n=n, metric=metric)
    core = ParticleShape(C_STD, MASS["proton"], lnp0=LNP0, lnp_s=LNP_S, degree=n, degrees=degrees or None, **kw).to(DEVICE)
    return shapelab.WithAcceptance(core, A_BAR_DEV).to(DEVICE) if metric == "std" else core


def load_or_fit(n, metric, prev=None):
    """Load the trained rung (written by scripts/particle_pipeline.py); otherwise train it here."""
    path = f"{FINAL}/{TAG}_{'std' if metric == 'std' else 'slab'}_n{n}.pt"
    m = make(n, metric)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["state"])
        return m.eval()
    core = m.base if metric == "std" else m
    core.init_from_data(C_STD, data)
    if prev is not None:
        core.warm_start(prev.base if metric == "std" else prev)
    shapelab.fit_fast(C_SLAB if metric == "slab" else C_STD, data, m, steps=4000, log=None)
    os.makedirs(FINAL, exist_ok=True)
    torch.save({"state": m.state_dict()}, path)
    return m.eval()


LADDER, CHOSEN = {}, {}
for metric, tnll in (("slab", test_nll_slab), ("std", test_nll)):
    prev, val = None, {}
    for n in range(1, 7):
        prev = LADDER[(metric, n)] = load_or_fit(n, metric, prev)
        val[n] = tnll(prev, data["val"])
    n_star = min(val, key=lambda k: val[k].mean())
    d = {n: paired(val[n], val[n_star]) for n in val}
    CHOSEN[metric] = min(n for n in val if n == n_star or d[n][0] < d[n][1])
    print(f"{metric:4s} metric:", " | ".join(f"n={n}: {LADDER[(metric, n)].n_params()} params, val {val[n].mean():.4f} (Δ vs n* {d[n][0]:+.4f} ± {d[n][1]:.4f})" for n in val),
          f"→ chosen n = {CHOSEN[metric]}")


# %% [markdown]
# **The observable region.** The 3D likelihood is blind to model mass in cells that no event's slab ever covers (the small-$r$
# region in front of the entrance face, $z' \lesssim -r\tan\theta_{max}$): such mass changes no event's normalisation, so the fit is
# free to put mass there (here 5–9 % of the map). Physically that region is outside the calorimeter. For every figure and body below the
# 3D-trained model is therefore restricted to the cells covered by at least one training slab ($\bar A > 0$) and renormalised; the 3D
# NLL is unchanged by this (verified below).

# %%
class Observable(nn.Module):
    """The model restricted to the cells that at least one training slab covers (A_bar > 0), renormalised."""

    def __init__(self, base, mask):
        super().__init__()
        self.base = base
        self.register_buffer("logm", torch.log(mask.float().clamp_min(1e-30)))

    def forward(self, lnp):
        lp = self.base(lnp) + self.logm
        return F.log_softmax(lp.reshape(len(lnp), -1), 1).reshape(lp.shape)

    def n_params(self):
        return self.base.n_params()


OBS_MASK = (A_BAR_DEV > 0)
SHAPE = Observable(LADDER[("slab", CHOSEN["slab"])], OBS_MASK).eval()
with torch.no_grad():
    lost = (LADDER[("slab", CHOSEN["slab"])](torch.log(torch.tensor([0.5, 2.0, 15.0], device=DEVICE))).exp() * (~OBS_MASK)).sum((1, 2))
t_full, t_obs = test_nll_slab(LADDER[("slab", CHOSEN["slab"])], data["test"]), test_nll_slab(SHAPE, data["test"])
print(f"never-observable cells: {int((~OBS_MASK).sum())} of {OBS_MASK.numel()}; mass there at p = 0.5 / 2 / 15 GeV: "
      + " / ".join(f"{float(x):.3f}" for x in lost)
      + f" | 3D test NLL {t_full.mean():.6f} -> {t_obs.mean():.6f} after restriction (max per-event change {np.abs(t_obs - t_full).max():.1e})")
