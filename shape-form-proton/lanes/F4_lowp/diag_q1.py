"""Q1: where does the reference (GammaCoreTail n=2, slab) lose to the slab-renormalised MLP, and what do the data look like at low p?"""
import json, math, os, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/F4_lowp")
import numpy as np, torch
import lane, shapelab, families

C, geo, data, base, c, s = lane.setup()
dev = C["DEVICE"]
ref, ref_test = lane.ref_val(C, data, base, c, s)
mlp = base["mlp"]["val"]
res = {"geo": geo, "lnp_center_scale": [c, s], "ref_val": float(ref.mean()), "ref_test": float(ref_test.mean()),
       "mlp_val": float(mlp.mean()), "gflash_n6_slab_val": float(base["gflash_n6"]["val"].mean())}
va = data["val"]
bins = va.ev["bin"].cpu().numpy(); p = va.ev["p"].cpu().numpy()
print("ref val", res["ref_val"], "test", res["ref_test"], "| mlp val", res["mlp_val"])
print("\nper 1-GeV bin (validation): n, ref, mlp, ref-mlp ± SE, share of the total gap")
tot = float((ref - mlp).sum())
rows = []
for b in range(20):
    m = bins == b
    d, se = shapelab.paired(ref[m], mlp[m])
    rows.append((b, int(m.sum()), float(ref[m].mean()), float(mlp[m].mean()), d, se, float((ref - mlp)[m].sum() / tot)))
    print(f"  bin {b:2d} n={m.sum():4d} ref {ref[m].mean():8.4f} mlp {mlp[m].mean():8.4f}  d={d:+.4f} ± {se:.4f}  share {rows[-1][-1]:.3f}")
res["by_bin"] = rows
edges = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.01]
print("\nfine p bins (validation): ref - mlp")
fine = []
for a, bq in zip(edges[:-1], edges[1:]):
    m = (p >= a) & (p < bq)
    if m.sum() < 2:
        continue
    d, se = shapelab.paired(ref[m], mlp[m])
    fine.append((a, bq, int(m.sum()), float(ref[m].mean()), d, se))
    print(f"  p {a:5.2f}-{bq:5.2f} n={m.sum():5d} ref {ref[m].mean():8.4f}  d={d:+.4f} ± {se:.4f}")
res["fine"] = fine

# ---- data: what do low-momentum events look like (train) ----
tr = data["train"]
b = tr.batch(torch.arange(len(tr), device=dev))
z, x, y, w, ev = b.xyz[:, 2], b.xyz[:, 0], b.xyz[:, 1], b.w, b.ev
r = torch.sqrt(x * x + y * y)
cos_t = b.R[:, 2, 2]; sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
zt = z * cos_t[ev] - x * sin_t[ev]                       # detector-axis depth of the hit (multiples of the layer pitch)
layer = torch.round(zt / geo["layer"]).long()
pe = b.p[ev]
n = len(tr)
print("\nmax |zt - layer*pitch|:", float((zt - layer * geo["layer"]).abs().max()))
# per-event quantities
nh = torch.zeros(n, device=dev).index_add_(0, ev, torch.ones_like(w))
e0 = torch.zeros(n, device=dev).index_add_(0, ev, w * (layer == 0))                       # energy fraction in first hit layer
eorig = torch.zeros(n, device=dev).index_add_(0, ev, w * ((layer == 0) & (r < 1.0)))      # exactly at the origin
n0 = torch.zeros(n, device=dev).index_add_(0, ev, (layer == 0).float())                   # hits in first layer
lmax = torch.full((n,), -1, device=dev, dtype=torch.long).scatter_reduce(0, ev, layer, "amax")
zmean = torch.zeros(n, device=dev).index_add_(0, ev, w * z)
# energy-weighted last layer holding the top 90% of energy along depth: per-event 90% quantile of layer
order = torch.argsort(layer + ev * 1000, stable=True)
evo, lo_, wo = ev[order], layer[order], w[order]
cw = torch.cumsum(wo.double(), 0)
start = tr.off[:-1]
before = torch.cat([cw.new_zeros(1), cw])[start]
frac = (cw - before[evo]).float()
def layer_q(q):
    big = torch.where(frac >= q - 1e-6, lo_, torch.full_like(lo_, 10 ** 6))
    return torch.full((n,), 10 ** 6, device=dev, dtype=torch.long).scatter_reduce(0, evo, big, "amin")
l50, l90 = layer_q(0.5), layer_q(0.9)
pp = b.p.cpu().numpy(); T = np.sqrt(pp ** 2 + lane.M_P ** 2) - lane.M_P
tab = []
print("\nfine p bins (train): n, <T>, nhits(med), E-frac first layer, E-frac at origin, P(1 hit in first layer), median l50, l90, lmax; <z'>")
pe_edges = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.01]
for a, bq in zip(pe_edges[:-1], pe_edges[1:]):
    m = torch.as_tensor((pp >= a) & (pp < bq), device=dev)
    if m.sum() < 5:
        continue
    row = dict(p0=a, p1=bq, n=int(m.sum()), T=float(T[m.cpu().numpy()].mean()), nh=float(nh[m].median()), e0=float(e0[m].mean()),
               eorig=float(eorig[m].mean()), one=float((n0[m] == 1).float().mean()), l50=float(l50[m].float().median()),
               l90=float(l90[m].float().median()), l90mean=float(l90[m].float().mean()), lmax=float(lmax[m].float().median()),
               zmean=float(zmean[m].mean()), l90q=[float(v) for v in torch.quantile(l90[m].float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev))])
    tab.append(row)
    print(f"  p {a:5.2f}-{bq:5.2f} n={row['n']:5d} T={row['T']:.3f} nh={row['nh']:.0f} e0={row['e0']:.3f} eorig={row['eorig']:.3f} one={row['one']:.3f} "
          f"l50={row['l50']:.0f} l90={row['l90']:.0f} (mean {row['l90mean']:.1f}, q10-90 {row['l90q']}) lmax={row['lmax']:.0f} <z'>={row['zmean']:.1f}")
res["train_fine"] = tab
# layer-resolved energy profile for the five ranges (energy-weighted mean fraction per layer)
rng = [(0.24, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
prof = {}
for a, bq in rng:
    m = (pe >= a) & (pe < bq)
    ne = int(((b.p >= a) & (b.p < bq)).sum())
    h = torch.zeros(40, device=dev).index_add_(0, layer[m].clamp(0, 39), w[m]) / ne
    prof[f"{a}-{bq}"] = h.cpu().numpy().round(4).tolist()
    print(f"\nlayer profile p {a}-{bq} (n={ne}):", " ".join(f"{v:.3f}" for v in h[:24].cpu().numpy()))
res["layer_profile"] = prof
json.dump(res, open(os.path.join(lane.LANE, "diag_q1.json"), "w"), indent=1)

# ---- figure: longitudinal and lateral energy-weighted profiles, data vs reference vs MLP (unfolded model maps folded by sampling) ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
m_ref = families.GammaCoreTail(C, degree=2, lnp0=c, lnp_s=s).to(dev)
m_ref.load_state_dict(torch.load(os.path.join(lane.OUT, "proton", "ref_gct_n2.pt"), weights_only=False)["state"]); m_ref.eval()
m_mlp = shapelab.baseline_models(C, data)["mlp"]
hits = tr.hits()
idx = (tr.ev["p"] < 3.0).nonzero().squeeze(1)
sub = tr.subset(idx)
samp = {"GammaCoreTail n=2 (ref)": C["sample_folded"](m_ref, sub, M=400), "MLP": C["sample_folded"](m_mlp, sub, M=400)}
col = {"data": "black", "GammaCoreTail n=2 (ref)": "#2a78d6", "MLP": "#eb6834"}
fig, axes = plt.subplots(3, 5, figsize=(18, 10))
for j, (a, bq) in enumerate(rng):
    T0, T1 = [math.sqrt(q * q + lane.M_P ** 2) - lane.M_P for q in (a, bq)]
    zmax = [150, 400, 700, 900, 1200][j]
    for i, (which, edges) in enumerate((("z", np.arange(-60, zmax, 10.0)), ("r", np.arange(0, 200, 5.0)), ("r", np.arange(0, 700, 20.0)))):
        ax = axes[i, j]
        for name, sdf in {"data": hits, **samp}.items():
            mm = (sdf.p >= a) & (sdf.p < bq)
            h, _ = np.histogram(sdf[which][mm], edges, weights=sdf.w[mm])
            ne = len(np.unique(sdf.p[mm])) if name == "data" else mm.sum() / 400
            ax.step(edges[:-1], h / max(sdf.w[mm].sum(), 1e-9) / np.diff(edges), where="post", color=col[name], lw=2 if name == "data" else 1.3, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("z' [mm]" if which == "z" else "r [mm]")
        if i == 0:
            ax.set_title(f"p = {a}-{bq} GeV  (T = {T0:.2f}-{T1:.2f} GeV)", fontsize=10)
        if j == 0:
            ax.set_ylabel(["dE/dz' (norm.)", "dE/dr (norm.), r < 200 mm", "dE/dr (norm.), full"][i])
axes[0, 0].legend(fontsize=8)
fig.suptitle("Proton, energy-weighted profiles at low momentum: data (train) vs detector-folded samples of the reference and the MLP", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(lane.FIG, "q1_profiles_lowp.png"), dpi=110)
print("saved figure")
