"""Shape of the entry spike in (z', r): cell-smeared deposits of the hits that sit exactly at the origin (and of the whole first layer),
plus a fine-p check of the existing checkpoints (ref, mix_tie_lnT_n2) against the slab MLP. Train data for shapes, validation for NLL."""
import json, math, os, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/F4_lowp")
import numpy as np, torch
import lane, fam, shapelab, envlab

C, geo, data, base, c, s = lane.setup()
dev = C["DEVICE"]
ref, ref_test = lane.ref_val(C, data, base, c, s)
mlp = base["mlp"]["val"]
va = data["val"]; p = va.ev["p"].cpu().numpy()
ck = torch.load(os.path.join(lane.OUT, "proton", "mix_tie_lnT_n2.pt"), weights_only=False)
mix = ck["val_nll"]
print("ref val %.4f test %.4f | mix_tie val %.4f test %.4f | mlp val %.4f" % (ref.mean(), ref_test.mean(), mix.mean(), ck["test_nll"].mean(), mlp.mean()))
edges = [0.2, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.01]
print("fine p (val): n | ref | mlp | mix_tie | ref-mlp | mix-mlp ± se")
for a, b in zip(edges[:-1], edges[1:]):
    m = (p >= a) & (p < b)
    d, se = shapelab.paired(mix[m], mlp[m])
    print(f"  {a:5.2f}-{b:5.2f} n={m.sum():5d} ref {ref[m].mean():8.4f} mlp {mlp[m].mean():8.4f} mix {mix[m].mean():8.4f}  ref-mlp {ref[m].mean()-mlp[m].mean():+.4f}  mix-mlp {d:+.4f} ± {se:.4f}")

# ---- spike shape from the data (train) ----
tr = data["train"]
P = envlab.deposits(C, tr, K=16, seed=3, xy=True)
b = tr.batch(torch.arange(len(tr), device=dev))
cos_t = b.R[:, 2, 2]; sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
zt = b.xyz[:, 2] * cos_t[b.ev] - b.xyz[:, 0] * sin_t[b.ev]
layer = torch.round(zt / geo["layer"]).long()
r_hit = torch.sqrt(b.xyz[:, 0] ** 2 + b.xyz[:, 1] ** 2)
K = 16
lay_k = layer[:, None].expand(-1, K).reshape(-1); rh_k = r_hit[:, None].expand(-1, K).reshape(-1)
pe_k = b.p[b.ev][:, None].expand(-1, K).reshape(-1)
zed = torch.arange(-60, 61, 5.0, device=dev); red = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 70.0], device=dev)
def hist2(mask):
    iz = torch.bucketize(P["z"][mask].contiguous(), zed) - 1; ir = torch.bucketize(P["r"][mask].contiguous(), red) - 1
    ok = (iz >= 0) & (iz < len(zed) - 1) & (ir >= 0) & (ir < len(red) - 1)
    H = torch.zeros((len(zed) - 1) * (len(red) - 1), device=dev).index_add_(0, (iz * (len(red) - 1) + ir)[ok], P["w"][mask][ok])
    return H.reshape(len(zed) - 1, len(red) - 1), float(P["w"][mask][~ok].sum())
for label, mask in (("origin hits (layer 0, r<1mm), all p", (lay_k == 0) & (rh_k < 1.0)),
                    ("first-layer hits NOT at origin, all p", (lay_k == 0) & (rh_k >= 1.0)),
                    ("layer-1 hits, p<1", (lay_k == 1) & (pe_k < 1.0))):
    H, out = hist2(mask)
    tot = float(H.sum()) + out
    area = math.pi * (red[1:] ** 2 - red[:-1] ** 2)
    print(f"\n{label}: total weight {tot:.1f}, outside window {out/tot:.4f}")
    pr = H.sum(0) / tot
    print("  r bin        :", " ".join(f"{float(a):4.0f}-{float(bq):<4.0f}" for a, bq in zip(red[:-1], red[1:])))
    print("  E fraction   :", " ".join(f"{float(v):9.4f}" for v in pr))
    print("  dens/area e-4:", " ".join(f"{float(v)*1e4:9.3f}" for v in pr / area))
    zc = (zed[:-1] + zed[1:]) / 2
    mz = (H * zc[:, None]).sum(0) / H.sum(0).clamp_min(1e-9)
    sd = torch.sqrt((H * (zc[:, None] - mz[None]) ** 2).sum(0) / H.sum(0).clamp_min(1e-9))
    print("  <z'> per r   :", " ".join(f"{float(v):9.2f}" for v in mz))
    print("  sd z' per r  :", " ".join(f"{float(v):9.2f}" for v in sd))
    print("  z' profile (all r), per 5 mm from -60:", " ".join(f"{float(v)/tot:.3f}" for v in H.sum(1)))
    for j in (0, 3, 5, 7):
        print(f"  z' profile r-bin {j}:", " ".join(f"{float(v)/max(float(H[:, j].sum()),1e-9):.3f}" for v in H[:, j]))
th = torch.acos(cos_t)
print("\ntheta quantiles:", [round(float(v), 3) for v in torch.quantile(th, torch.tensor([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98], device=dev))])
