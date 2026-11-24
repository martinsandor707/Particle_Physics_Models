# %% [markdown]
# # 08 – The 3D shape of the proton shower: a body of revolution cut by the detector planes
#
# ## 1. Introduction
#
# Notebooks 01–06 describe the proton shower as a **body of revolution** around the direction of incidence: a density
# $\rho(z', r \mid p)$ on the $(z', r)$ grid. This notebook asks whether a better 3D shape exists, and finds three things.
#
# 1. **The measured shower is not a body of revolution.** The calorimeter layers are perpendicular to the detector axis, the proton
#    arrives at an angle $\theta = 3^\circ$–$36^\circ$, and the origin of the shower frame is the first hit layer. Hence every hit satisfies
#    $z_{trans} = z'\cos\theta - x'\sin\theta \ge 0$: the body of revolution is **cut by an oblique plane through the origin** (and by a second
#    one at the back face). Averaged over the azimuth this gives the cone-shaped front that the networks (05, 06) learn and that a product
#    $L(z')\,T(r)$ cannot represent.
# 2. This cut is pure geometry and **costs no parameter**: for a ring $(z', r)$ the accepted fraction of the azimuth is a closed-form
#    $\arccos$ expression. It can be used in two ways: as an event-wise normalisation (the "3D likelihood"), or – averaged over the
#    incidence angles – as a fixed factor $\bar A(z', r)$ that turns any formula into a better pure $(z', r \mid p)$ model in the
#    report's own metric.
# 3. With the geometry taken out, the remaining deficit of the GFlash formula is concentrated below 2 GeV, where the proton
#    (unlike the pion) mostly **stops by ionisation** instead of showering.
#
# **Coordinates** (as in 00–06): $z'$ = depth along the direction of incidence, $r=\sqrt{x'^2+y'^2}$; in addition $x'$ is the transverse
# coordinate in the plane spanned by the shower axis and the detector axis ($x' > 0$ points radially outward and upstream), $y'$ lies
# in the layer plane.

# %%
import numpy as np, matplotlib.pyplot as plt

# illustrative shower (the same toy as in 01-03) seen from the side: in 3D it is cut by the tilted front and back faces
LAM, Rc, Rt, pc, TH, DEPTH = 170.0, 25.0, 110.0, 0.6, 0.45, 1210.0
z, x = np.meshgrid(np.linspace(-200, 1450, 420), np.linspace(-600, 600, 301))
tmax = LAM * (0.2 * np.log(8) + 0.7); bb = 0.5 / LAM; aa = 1 + bb * tmax
s = np.clip(z + 150, 1e-3, None); rc = Rc * (1 + np.clip(z, 0, None) / 800)
rho = s ** (aa - 1) * np.exp(-bb * s) * (pc * rc**2 / (x**2 + rc**2)**2 + (1 - pc) * (4 * rc)**2 / (x**2 + (4 * rc)**2)**2)
zt = z * np.cos(TH) - x * np.sin(TH)                       # depth along the detector axis
inside = (zt >= 0) & (zt <= DEPTH)
fig, ax = plt.subplots(1, 3, figsize=(16, 3.9))
for a_, d, t in ((ax[0], rho, "Body of revolution (what 01-06 assume)"), (ax[1], np.where(inside, rho, np.nan), "What the detector can record (θ = 0.45 rad)")):
    a_.imshow(np.log10(d), extent=[-200, 1450, -600, 600], origin="lower", aspect="auto", cmap="magma", vmin=np.log10(rho.max()) - 5, vmax=np.log10(rho.max()))
    a_.set(title=t, xlabel="z' [mm]", ylabel="x' [mm]"); a_.set_facecolor("0.85")
zz = np.linspace(-200, 1450, 10)
ax[1].plot(zz, zz / np.tan(TH), "c--", lw=1.2, label="front face  z_trans = 0"); ax[1].plot(zz, (zz * np.cos(TH) - DEPTH) / np.sin(TH), "w--", lw=1.2, label="back face")
ax[1].set_ylim(-600, 600); ax[1].legend(fontsize=8, loc="lower right")
zq = np.linspace(-200, 300, 400)
for r_ in (50, 150, 300, 500):
    ax[2].plot(zq, 1 - np.arccos(np.clip(zq / (r_ * np.tan(TH)), -1, 1)) / np.pi, label=f"r = {r_} mm")
ax[2].set(title="Accepted fraction of the azimuth, A(z', r)", xlabel="z' [mm]", ylabel="A"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Left:** the body of revolution assumed so far. **Middle:** the same shower as the detector can see it: everything in front of the
# (tilted) front face and behind the back face is missing. **Right:** the fraction of the ring $(z', r)$ that lies behind the front face,
# $A = 1 - \arccos\!\big(z'/(r\tan\theta)\big)/\pi$: half of every ring is lost at $z' = 0$, and rings of large radius are partly
# visible already at $z' < 0$. This is the cone in the $(z', r)$ maps of the data.
#
# ## 2. Loading data

# %%
PARTICLES = ['proton']
DATA_DIR = "/home/user/data/simulations/regression-proton/"     # location of the CSVs
OUT_DIR = "results"                                             # models and exports go here
%run -i ./00_common.ipynb

# %%
# Loading the 16.7 GB CSV takes about 15 minutes; the compact tensors are therefore cached after the first run.
CACHE = "cache/proton_sets.pt"
if os.path.exists(CACHE):
    ck = torch.load(CACHE, weights_only=False)
    geo = ck["geo"]
    data = {k: Showers(v["ev"], v["off"], v["xyz"], v["w"], v["box"]).to(DEVICE) for k, v in ck["sets"].items()}
    print("loaded the cached split (identical to load_data('proton'), SEED = 1)")
else:
    df, geo, data = load_data("proton")
    os.makedirs("cache", exist_ok=True)
    torch.save({"geo": geo, "sets": {k: {"ev": {a: b.cpu() for a, b in s.ev.items()}, "off": s.off.cpu(), "xyz": s.xyz.cpu(),
                                         "w": s.w.cpu(), "box": s.box.cpu()} for k, s in data.items()}}, CACHE)
print("geometry:", geo)
for k, s in data.items():
    print(f"{k:5s}: {len(s):6d} events, {len(s.w):9d} hits, p = {float(s.ev['p'].min()):.2f}–{float(s.ev['p'].max()):.2f} GeV")
theta_tr = torch.acos(data["train"].ev["R"][:, 2, 2])
print(f"incidence angle θ: {float(theta_tr.min()):.3f}–{float(theta_tr.max()):.3f} rad, median {float(theta_tr.median()):.3f}")
display(data["train"].hits().describe().T)
