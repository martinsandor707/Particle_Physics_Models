"""Data exploration for lane F2 (train set only): longitudinal profile, entry spike, on-axis track."""
import json, math, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import shapelab

LANE = "/home/user/tamas/shape-form-proton/lanes/F2_longitudinal"
C, geo, data = shapelab.setup(batch=256, out_dir=LANE + "/results", slab=True)
dev = C["DEVICE"]; tr = data["train"]
b = tr.batch(torch.arange(len(tr), device=dev))
cos_t = b.R[:, 2, 2]; sin_t = torch.sqrt(1 - cos_t ** 2)
x, y, z, w = b.xyz[:, 0], b.xyz[:, 1], b.xyz[:, 2], b.w
r = torch.sqrt(x * x + y * y)
zt = z * cos_t[b.ev] - x * sin_t[b.ev]
pbin = tr.ev["bin"][b.ev]
layer = torch.round(zt / geo["layer"]).long()
np.set_printoptions(linewidth=250, precision=4, suppress=True)
print("events per bin:", torch.bincount(tr.ev["bin"]).tolist())
print("mean p per bin:", [round(float(tr.ev["p"][tr.ev["bin"] == k].mean()), 3) for k in range(20)])
print("theta quantiles:", np.quantile(torch.acos(cos_t).cpu().numpy(), [0, .1, .5, .9, 1]).round(3))

# 1. energy per detector layer (z_trans / 20.5), split on-axis (r<30) / off-axis, per momentum bin
print("\n== energy fraction per detector layer k = z_trans/20.5 (rows: bin; cols: layer 0..14), total")
for kbin in (0, 1, 2, 4, 9, 14, 19):
    m = pbin == kbin
    tot = float(w[m].sum())
    e = torch.zeros(60, device=dev).index_add_(0, layer[m].clamp(0, 59), w[m]) / tot
    print(f"bin {kbin:2d} all  ", (e[:15].cpu().numpy() * 100).round(2))
    m2 = m & (r < 30)
    e2 = torch.zeros(60, device=dev).index_add_(0, layer[m2].clamp(0, 59), w[m2]) / tot
    print(f"bin {kbin:2d} r<30 ", (e2[:15].cpu().numpy() * 100).round(2))

# 2. first hit layer: r distribution (energy-weighted, % of the bin's energy)
print("\n== first hit layer (z_trans=0): energy % by r range [0,1) [1,10) [10,25) [25,50) [50,100) [100,700)")
edges = torch.tensor([0, 1, 10, 25, 50, 100, 700.], device=dev)
for kbin in (0, 1, 2, 4, 9, 14, 19):
    m = (pbin == kbin) & (layer == 0)
    tot = float(w[pbin == kbin].sum())
    ir = torch.bucketize(r[m], edges[1:-1])
    e = torch.zeros(6, device=dev).index_add_(0, ir, w[m]) / tot
    print(f"bin {kbin:2d}", (e.cpu().numpy() * 100).round(2), " n_hits/event in first layer:", round(float(m.sum()) / float((tr.ev['bin'] == kbin).sum()), 2))

# 3. energy-weighted (z', r) table near the origin, all bins >= 5 and bin 0
zed = torch.tensor([-200, -45, -32, -19.5, -6.6, 6.3, 19.1, 32, 45, 58, 84, 110, 160], device=dev)
red = torch.tensor([0, 1.93, 6.3, 12.5, 20.4, 29.8, 40.7, 53, 80, 120, 700.], device=dev)
for name, m in (("bin0", pbin == 0), ("bin1", pbin == 1), ("bins5-19", pbin >= 5)):
    iz = torch.bucketize(z[m], zed[1:-1]); ir = torch.bucketize(r[m], red[1:-1])
    H = torch.zeros(len(zed) - 1, len(red) - 1, device=dev).index_put_((iz, ir), w[m], accumulate=True) / float(w[m].sum())
    print(f"\n== {name}: energy % in (z' rows, r cols); r edges {red.tolist()}")
    for i in range(len(zed) - 1):
        print(f"z' [{float(zed[i]):7.1f},{float(zed[i+1]):7.1f})", (H[i].cpu().numpy() * 100).round(3))

# 4. longitudinal profile on the model grid (energy fraction per 12.9 mm cell), on-axis and total -> npz for figures
Z_MIN, DZ, N_Z = C["Z_MIN"], C["DZ"], C["N_Z"]
iz = ((z - Z_MIN) / DZ).floor().long().clamp(0, N_Z - 1)
prof = np.zeros((20, 3, N_Z))
for kbin in range(20):
    m = pbin == kbin
    tot = float(w[m].sum())
    for j, mm in enumerate((m, m & (r < 30), m & (layer == 0))):
        prof[kbin, j] = (torch.zeros(N_Z, device=dev).index_add_(0, iz[mm], w[mm]) / tot).cpu().numpy()
np.savez(LANE + "/results/data_profiles_train.npz", prof=prof, zc=C["ZC"].cpu().numpy())
print("\n== on-axis (r<30) energy % per z' cell, z' cells 13..40 (centres %s ...)" % C["ZC"][13:18].cpu().numpy().round(1))
for kbin in (0, 1, 2, 4, 9, 19):
    print(f"bin {kbin:2d} r<30:", (prof[kbin, 1, 13:41] * 100).round(2))
    print(f"bin {kbin:2d} all :", (prof[kbin, 0, 13:41] * 100).round(2))
# 5. slope of the on-axis profile: fit ln(e) vs z' for 60 < z' < 400
zc = C["ZC"].cpu().numpy()
sel = (zc > 60) & (zc < 400)
print("\n== exponential attenuation of the on-axis (r<30) energy, 60<z'<400: lambda [mm] per bin")
print([round(float(-1 / np.polyfit(zc[sel], np.log(np.clip(prof[k, 1, sel], 1e-9, None)), 1)[0]), 0) for k in range(20)])
print("== exponential slope of the total profile, 600<z'<1100: lambda [mm] per bin")
sel2 = (zc > 600) & (zc < 1100)
print([round(float(-1 / np.polyfit(zc[sel2], np.log(np.clip(prof[k, 0, sel2], 1e-9, None)), 1)[0]), 0) for k in range(20)])
