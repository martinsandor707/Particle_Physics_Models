"""Lateral profile of the proton: data (cell centres) vs models (sampled, cut by each event's detector slab, folded with the cell).

usage: latprof.py <split> <out tag> [label=ctor@checkpoint ...]
  The reference (families.GammaCoreTail degree 2, slab fit) is always included.
Writes figs/lateral_<tag>.png (dE/dr), figs/lateral_ratio_<tag>.png (model/data) and results/lateral_<tag>.json (slice numbers).
"""
import json
import math
import os
import sys

import numpy as np
import torch

import lab
import fam
import families
import shapelab

P_EDGES = [0.0, 1.0, 2.0, 5.0, 10.0, 20.5]
Z_SLICES = [-200.0, 0.0, 50.0, 150.0, 300.0, 600.0, 1450.0]
R_BINS = np.concatenate([np.arange(0, 100, 10.0), np.arange(100, 701, 25.0)])


@torch.no_grad()
def sample_slab_folded(C, model, sh, M=256, seed=7):
    """Points drawn from the model for every event, kept only inside the event's detector slab, then folded with the cell box.
    Returns z, r, w (w sums to 1 per event), p."""
    dev = C["DEVICE"]
    gen = torch.Generator(device=dev).manual_seed(seed)
    half = float(sh.box[2]) / 2
    N_R, ZE, RE, DZ = C["N_R"], C["Z_EDGES"], C["R_EDGES"], C["DZ"]
    out = []
    for b in sh.batches(256):
        B = len(b.lnp)
        P = model(b.lnp).exp().reshape(B, -1)
        cell = torch.multinomial(P, M, True, generator=gen)
        iz, ir = cell // N_R, cell % N_R
        u = torch.rand(B, M, 3, device=dev, generator=gen)
        z = ZE[iz] + u[..., 0] * DZ
        r = torch.sqrt(RE[ir] ** 2 + u[..., 1] * (RE[ir + 1] ** 2 - RE[ir] ** 2))
        x, y = r * torch.cos(2 * math.pi * u[..., 2]), r * torch.sin(2 * math.pi * u[..., 2])
        cos_t = b.R[:, 2, 2][:, None]
        sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
        zt = z * cos_t - x * sin_t
        inside = ((zt >= -half) & (zt <= b.zb[:, None] * cos_t)).float()
        box = (torch.rand(B, M, 3, device=dev, generator=gen) - 0.5) * sh.box
        pts = torch.stack([x, y, z], -1) + torch.einsum("bij,bmj->bmi", b.R, box)
        w = inside / inside.sum(1, keepdim=True).clamp_min(1.0)
        out.append((pts[..., 2].flatten(), torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2).flatten(), w.flatten(),
                    b.p[:, None].expand(-1, M).flatten()))
    return [torch.cat(v) for v in zip(*out)]


def data_points(sh):
    owner = torch.repeat_interleave(torch.arange(len(sh), device=sh.off.device), sh.off[1:] - sh.off[:-1])
    return sh.xyz[:, 2], torch.sqrt(sh.xyz[:, 0] ** 2 + sh.xyz[:, 1] ** 2), sh.w, sh.ev["p"][owner]


def hist3(z, r, w, p, dev):
    """Energy per event in (p slice, z' slice, r bin): array (n_p, n_z, n_r)."""
    pe, ze, re = (torch.tensor(v, device=dev, dtype=torch.float32) for v in (P_EDGES, Z_SLICES, R_BINS))
    ip = (torch.bucketize(p, pe[1:-1], right=True))
    iz = torch.bucketize(z.contiguous(), ze[1:-1].contiguous(), right=True)
    ir = torch.bucketize(r.contiguous(), re[1:-1].contiguous(), right=True)
    ok = (z >= ze[0]) & (z <= ze[-1]) & (r < re[-1])
    n_p, n_z, n_r = len(P_EDGES) - 1, len(Z_SLICES) - 1, len(R_BINS) - 1
    idx = (ip * n_z + iz) * n_r + ir
    h = torch.zeros(n_p * n_z * n_r, device=dev, dtype=torch.float64).index_add_(0, idx[ok], w[ok].double())
    return h.reshape(n_p, n_z, n_r).cpu().numpy()


def main():
    split, tag = sys.argv[1], sys.argv[2]
    C, geo, data, c, s = lab.setup(slab=True)
    dev = C["DEVICE"]
    sh = data[split]
    models = {}
    ref, _ = lab.reference(C, data, c, s, 2)
    models["two-Cauchy baseline (21)"] = ref
    for spec in sys.argv[3:]:
        label, rest = spec.split("=", 1)
        ctor, ck = rest.rsplit("@", 1)
        m = eval(ctor, {"fam": fam, "families": families, "C": C, "c": c, "s": s}).to(dev)
        st = torch.load(ck, map_location=dev, weights_only=False)
        m.load_state_dict(st["state"])
        models[f"{label} ({m.n_params()})"] = m.eval()
    n_ev = np.histogram(sh.ev["p"].cpu().numpy(), P_EDGES)[0]
    H = {"data": hist3(*data_points(sh), dev) / n_ev[:, None, None]}
    for k, m in models.items():
        H[k] = hist3(*sample_slab_folded(C, m, sh), dev) / n_ev[:, None, None]
    rc, dr = 0.5 * (R_BINS[1:] + R_BINS[:-1]), np.diff(R_BINS)

    # ---- numbers per slice: energy share, core share (r < 30 mm), exponential slope of dE/dr in 75 < r < 350 mm, KL(data || model) of the r histogram
    rows = []
    sel = (rc > 75) & (rc < 350)
    for ip in range(len(P_EDGES) - 1):
        for iz in range(len(Z_SLICES) - 1):
            row = {"p": [P_EDGES[ip], P_EDGES[ip + 1]], "z": [Z_SLICES[iz], Z_SLICES[iz + 1]]}
            for k, h in H.items():
                v = h[ip, iz]
                tot = v.sum()
                d = v / dr
                good = sel & (d > 0)
                slope = -1.0 / np.polyfit(rc[good], np.log(d[good]), 1)[0] if good.sum() > 3 else float("nan")
                row[k] = {"share": float(tot), "core30": float(v[rc < 30].sum() / max(tot, 1e-30)), "lam_75_350": float(slope),
                          "mean_r": float((v * rc).sum() / max(tot, 1e-30))}
                if k != "data":
                    q, pd_ = v / max(tot, 1e-30), H["data"][ip, iz] / max(H["data"][ip, iz].sum(), 1e-30)
                    okb = pd_ > 0
                    row[k]["kl_r"] = float((pd_[okb] * np.log(pd_[okb] / np.maximum(q[okb], 1e-12))).sum())
            rows.append(row)
    with open(os.path.join(lab.RES, f"lateral_{tag}.json"), "w") as fh:
        json.dump({"split": split, "P_EDGES": P_EDGES, "Z_SLICES": Z_SLICES, "R_BINS": R_BINS.tolist(), "rows": rows,
                   "H": {k: v.tolist() for k, v in H.items()}}, fh)
    names = list(H)
    print("slice table: share of the event energy | core30 = share of the slice within r < 30 mm | lam = slope of dE/dr (75-350 mm) | KL_r")
    for row in rows:
        txt = f"p {row['p'][0]:4.1f}-{row['p'][1]:4.1f} z' {row['z'][0]:6.0f}..{row['z'][1]:5.0f} |"
        for k in names:
            v = row[k]
            txt += f" {k[:14]:14s} sh {v['share']:.4f} c30 {v['core30']:.3f} lam {v['lam_75_350']:6.1f}" + (f" KL {v['kl_r']:.4f} |" if "kl_r" in v else " |")
        print(txt)

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid = "#0b0b0b", "#e1e0d9"
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    n_p, n_z = len(P_EDGES) - 1, len(Z_SLICES) - 1
    for kind in ("profile", "ratio"):
        fig, axes = plt.subplots(n_p, n_z, figsize=(2.9 * n_z, 2.15 * n_p), sharex=True, sharey=(kind == "ratio"))
        for ip in range(n_p):
            for iz in range(n_z):
                ax = axes[ip, iz]
                d = H["data"][ip, iz] / dr
                if kind == "profile":
                    ax.plot(rc, np.where(d > 0, d, np.nan), color=ink, lw=0, marker="o", ms=2.6, label="data (cell centres)")
                    for j, k in enumerate(names[1:]):
                        ax.plot(rc, H[k][ip, iz] / dr, color=cols[j], lw=1.6, label=k)
                    ax.set_yscale("log")
                    top = np.nanmax(np.where(d > 0, d, np.nan))
                    ax.set_ylim(top * 3e-5, top * 2.5)
                else:
                    ax.axhline(1.0, color=ink, lw=0.8)
                    for j, k in enumerate(names[1:]):
                        ax.plot(rc, np.where(d > 0, (H[k][ip, iz] / dr) / np.where(d > 0, d, 1), np.nan), color=cols[j], lw=1.6, label=k)
                    ax.set_ylim(0.4, 1.9)
                ax.grid(True, color=grid, lw=0.6)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(labelsize=7, color="#888")
                if ip == 0:
                    ax.set_title(f"{Z_SLICES[iz]:.0f} < z' < {Z_SLICES[iz + 1]:.0f} mm", fontsize=9)
                if iz == 0:
                    ax.set_ylabel(f"p {P_EDGES[ip]:g}-{min(P_EDGES[ip + 1], 20):g} GeV\n" + ("dE/dr [1/mm]" if kind == "profile" else "model / data"), fontsize=8)
                if ip == n_p - 1:
                    ax.set_xlabel("r [mm]", fontsize=8)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(("Proton lateral profile dE/dr per event" if kind == "profile" else "Proton lateral profile: model / data")
                     + f" in slices of depth and momentum ({split} set; models cut by the detector slab and folded with the cell)", fontsize=10, y=1.035)
        fig.tight_layout()
        path = os.path.join(lab.LANE, "figs", f"lateral_{'ratio_' if kind == 'ratio' else ''}{tag}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
