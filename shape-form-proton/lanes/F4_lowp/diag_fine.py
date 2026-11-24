"""Fine-binned look at the stopping regime (train): spike / tube / halo energy fractions and the layer profile of the tube."""
import json, math, os, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/F4_lowp")
import numpy as np, torch
import lane

C, geo, data, base, c, s = lane.setup()
dev = C["DEVICE"]
tr = data["train"]
b = tr.batch(torch.arange(len(tr), device=dev))
z, x, y, w, ev = b.xyz[:, 2], b.xyz[:, 0], b.xyz[:, 1], b.w, b.ev
r = torch.sqrt(x * x + y * y)
cos_t = b.R[:, 2, 2]; sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
zt = z * cos_t[ev] - x * sin_t[ev]
layer = torch.round(zt / geo["layer"]).long()
pe = b.p[ev]
print("theta quantiles (deg->rad):", [round(float(v), 3) for v in torch.quantile(torch.acos(cos_t), torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=dev))])
print("mean 1/cos:", float((1 / cos_t).mean()))
edges = [0.24, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.01]
out = []
print("p range | n | T | spike(origin) | first layer | tube r<40 (layer>=1) | halo r>=40 | <z'> tube | tube layer profile 1..16 (fraction of tube energy) ")
for a, bq in zip(edges[:-1], edges[1:]):
    me = (b.p >= a) & (b.p < bq)
    ne = int(me.sum())
    if ne < 5:
        continue
    m = (pe >= a) & (pe < bq)
    T = float((torch.sqrt(b.p[me] ** 2 + lane.M_P ** 2) - lane.M_P).mean())
    sp = float(w[m & (layer == 0) & (r < 1.0)].sum()) / ne
    fl = float(w[m & (layer == 0)].sum()) / ne
    tube = m & (layer >= 1) & (r < 40.0)
    halo = m & (layer >= 1) & (r >= 40.0)
    ft, fh = float(w[tube].sum()) / ne, float(w[halo].sum()) / ne
    hp = torch.zeros(60, device=dev).index_add_(0, layer[tube].clamp(0, 59), w[tube]) / max(float(w[tube].sum()), 1e-9)
    hh = torch.zeros(60, device=dev).index_add_(0, layer[halo].clamp(0, 59), w[halo]) / max(float(w[halo].sum()), 1e-9)
    zt_tube = float((w[tube] * z[tube]).sum() / w[tube].sum().clamp_min(1e-9))
    zt_halo = float((w[halo] * z[halo]).sum() / w[halo].sum().clamp_min(1e-9))
    r_halo = float((w[halo] * r[halo]).sum() / w[halo].sum().clamp_min(1e-9))
    out.append(dict(p0=a, p1=bq, n=ne, T=T, spike=sp, first=fl, tube=ft, halo=fh, z_tube=zt_tube, z_halo=zt_halo, r_halo=r_halo,
                    tube_prof=hp.cpu().numpy().round(4).tolist(), halo_prof=hh.cpu().numpy().round(4).tolist()))
    print(f"{a:5.2f}-{bq:5.2f} n={ne:5d} T={T:.3f} sp={sp:.3f} first={fl:.3f} tube={ft:.3f} halo={fh:.3f} z_tube={zt_tube:6.1f} z_halo={zt_halo:6.1f} r_halo={r_halo:5.1f} | "
          + " ".join(f"{v:.2f}" for v in hp[1:17].cpu().numpy()))
json.dump(out, open(os.path.join(lane.LANE, "diag_fine.json"), "w"), indent=1)
# r distribution of the tube region hits in layers>=1 for p<1: where are the cell centres relative to the axis?
m = (pe < 1.0) & (pe >= 0.6) & (layer >= 1)
redges = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300, 700.0], device=dev)
h = torch.zeros(len(redges) - 1, device=dev).index_add_(0, torch.bucketize(r[m].contiguous(), redges[1:-1].contiguous()), w[m])
print("\nr histogram (energy) of layer>=1 hits, 0.6<=p<1:", [f"{float(redges[i]):.0f}-{float(redges[i+1]):.0f}: {float(v/h.sum()):.3f}" for i, v in enumerate(h)])
m = (pe < 1.0) & (pe >= 0.6) & (layer == 0)
h = torch.zeros(len(redges) - 1, device=dev).index_add_(0, torch.bucketize(r[m].contiguous(), redges[1:-1].contiguous()), w[m])
print("r histogram (energy) of layer==0 hits, 0.6<=p<1:", [f"{float(redges[i]):.0f}-{float(redges[i+1]):.0f}: {float(v/h.sum()):.3f}" for i, v in enumerate(h)])
m = (pe >= 5.0) & (layer == 0)
h = torch.zeros(len(redges) - 1, device=dev).index_add_(0, torch.bucketize(r[m].contiguous(), redges[1:-1].contiguous()), w[m])
print("r histogram (energy) of layer==0 hits, p>=5:", [f"{float(redges[i]):.0f}-{float(redges[i+1]):.0f}: {float(v/h.sum()):.3f}" for i, v in enumerate(h)])
