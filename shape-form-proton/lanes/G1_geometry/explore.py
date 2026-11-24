"""G1 step 0: look at the detector cell layout, the per-event geometry and basic invariants of the slab picture."""
import json, math, os, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import shapelab

LANE = "/home/user/tamas/shape-form-proton/lanes/G1_geometry"
C, geo, data = shapelab.setup(batch=256, out_dir=f"{LANE}/results", slab=False)
dev = C["DEVICE"]
G = torch.load("/home/user/tamas/shape-form-proton/cache/proton_geometry.pt", weights_only=False)
print("geo", geo)
cx = G["cells_xy"].numpy().astype(np.float64)
lz = G["layers_z"].numpy().astype(np.float64)
print("towers", cx.shape, "layers", len(lz), "z first/last", lz[0], lz[-1], "pitch", np.unique(np.round(np.diff(lz), 3)))
ux, uy = np.unique(np.round(cx[:, 0], 1)), np.unique(np.round(cx[:, 1], 1))
print("n unique x", len(ux), "x range", ux.min(), ux.max(), "pitches", np.unique(np.round(np.diff(ux), 2), return_counts=True))
print("n unique y", len(uy), "y range", uy.min(), uy.max(), "pitches", np.unique(np.round(np.diff(uy), 2), return_counts=True))
pitch = 48.6
ix = np.round((cx[:, 0] - ux.min()) / pitch).astype(int); iy = np.round((cx[:, 1] - uy.min()) / pitch).astype(int)
print("grid residual x", np.abs(cx[:, 0] - (ux.min() + ix * pitch)).max(), "y", np.abs(cx[:, 1] - (uy.min() + iy * pitch)).max())
occ = np.zeros((ix.max() + 1, iy.max() + 1), bool); occ[ix, iy] = True
print("occupancy grid", occ.shape, "filled", occ.sum(), "of", occ.size)
rho = np.hypot(cx[:, 0], cx[:, 1])
print("tower rho min/max", rho.min(), rho.max(), "| |x| max", np.abs(cx[:, 0]).max(), "|y| max", np.abs(cx[:, 1]).max())
# print a coarse ascii picture of the occupancy (every 2nd cell)
for j in range(occ.shape[1] - 1, -1, -3):
    print("".join("#" if occ[i, j] else "." for i in range(0, occ.shape[0], 2)))
# the inner hole: rows through the centre
ic = np.argmin(np.abs(ux)); jc = np.argmin(np.abs(uy))
print("central row occupancy (x at y~0):", [(float(ux.min() + i * pitch)) for i in range(occ.shape[0]) if not occ[i, jc] and abs(ux.min() + i * pitch) < 600])
print("central col occupancy (y at x~0):", [(float(uy.min() + j * pitch)) for j in range(occ.shape[1]) if not occ[ic, j] and abs(uy.min() + j * pitch) < 600])

for k in ("train", "val", "test"):
    sh = data[k]; g = G["sets"][k]
    th = g["theta"].numpy().astype(np.float64); ph = g["phi"].numpy().astype(np.float64); o = g["origin"].numpy().astype(np.float64)
    R = sh.ev["R"].cpu().numpy(); zb = sh.ev["zb"].cpu().numpy().astype(np.float64)
    th_R = np.arccos(R[:, 2, 2]); D = zb * np.cos(th)
    print(f"[{k}] n={len(sh)} theta from geometry vs R: max diff {np.abs(th - th_R).max():.2e}; theta range {th.min():.4f}-{th.max():.4f}; phi range {ph.min():.3f}..{ph.max():.3f}")
    print(f"   z0 unique-ish: D = z_back - z0 check: max|D - (z_back - z0)| = {np.abs(D - (geo['z_back'] - o[:, 2])).max():.3e}")
    layer = (o[:, 2] - lz[0]) / 20.5
    print(f"   first-hit-layer index: max frac dev from integer {np.abs(layer - np.round(layer)).max():.3e}; hist", dict(zip(*np.unique(np.round(layer).astype(int), return_counts=True))))
    rho0 = np.hypot(o[:, 0], o[:, 1]); phi0 = np.arctan2(o[:, 1], o[:, 0])
    print(f"   origin rho range {rho0.min():.1f}-{rho0.max():.1f}; rho0 - z0*tan(theta): mean {np.mean(rho0 - o[:, 2] * np.tan(th)):.2f} sd {np.std(rho0 - o[:, 2] * np.tan(th)):.2f}; "
          f"dphi (origin azimuth - phi): sd {np.std(np.angle(np.exp(1j * (phi0 - ph)))):.4f}")
    # min/max z_trans of the hits after cuts
    b_ev = torch.repeat_interleave(torch.arange(len(sh), device=dev), sh.off[1:] - sh.off[:-1])
    ct = sh.ev["R"][:, 2, 2][b_ev]; st = torch.sqrt(1 - ct ** 2)
    zt = sh.xyz[:, 2] * ct - sh.xyz[:, 0] * st
    zmin = torch.full((len(sh),), 1e9, device=dev).scatter_reduce(0, b_ev, zt, "amin")
    zmax = torch.full((len(sh),), -1e9, device=dev).scatter_reduce(0, b_ev, zt, "amax")
    Dt = torch.tensor(D, device=dev, dtype=torch.float32)
    print(f"   min z_trans per event: min {float(zmin.min()):.3f} max {float(zmin.max()):.3f}; frac(min z_trans > 1 mm) = {float((zmin > 1).float().mean()):.5f}; "
          f"max(z_trans - (D - 10.25)) = {float((zmax - (Dt - 10.25)).max()):.3f}")
    lay = zt / 20.5
    print(f"   hit z_trans on the layer lattice: max |frac| = {float((lay - lay.round()).abs().max()):.4f}")
    # transverse global position of the hits relative to the tower lattice
    Rh = sh.ev["R"][b_ev]
    trans = torch.einsum("hji,hj->hi", Rh, sh.xyz)                     # R^T local
    og = torch.tensor(o, device=dev, dtype=torch.float32)[b_ev]
    gx, gy = trans[:, 0] + og[:, 0], trans[:, 1] + og[:, 1]
    fx = (gx - float(ux.min())) / pitch; fy = (gy - float(uy.min())) / pitch
    print(f"   hit global (x,y) on the tower lattice: max |frac| x {float((fx - fx.round()).abs().max()):.4f} y {float((fy - fy.round()).abs().max()):.4f}; "
          f"x range {float(gx.min()):.1f}..{float(gx.max()):.1f}, y range {float(gy.min()):.1f}..{float(gy.max()):.1f}")
    occ_t = torch.tensor(occ, device=dev)
    inside = occ_t[fx.round().long().clamp(0, occ.shape[0] - 1), fy.round().long().clamp(0, occ.shape[1] - 1)]
    print(f"   hits on occupied towers: {float(inside.float().mean()):.6f}")
    nh = (sh.off[1:] - sh.off[:-1]).float()
    print(f"   hits/event mean {float(nh.mean()):.1f}")
