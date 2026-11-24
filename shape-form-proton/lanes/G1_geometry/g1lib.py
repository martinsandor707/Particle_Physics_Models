"""g1lib - lane G1 (true 3D geometry): detector outline raster, per-event full acceptance, multi-normalisation NLL.

Everything here is lane-local; the shared harness (shapelab / 00_common) is used unchanged.

Normalisations of the SAME folded per-hit density g (S_e = -sum_i w_i log g(c_i) is common to all of them):
  std    : M = model mass with z' < z_b                                   (report; 00_common.event_nll)
  slab   : M = sum P * A_slab,  A_slab = closed-form azimuthal fraction of the ring inside the tilted detector slab
  full   : M = sum P * A_full,  A_full = azimuthal fraction inside slab AND transverse detector outline (gaps closed)
  cells  : M = sum P * A_cells, A_cells = fraction inside slab AND inside an existing tower box whose CELL CENTRE passes the
           report's cuts (-200 <= z' <= 1450, r < 700)  -> the union of cells that can actually carry a hit (module gaps excluded)
"""
import copy
import math
import os
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np
import torch

import shapelab

LANE = "/home/user/tamas/shape-form-proton/lanes/G1_geometry"
GEOM = "/home/user/tamas/shape-form-proton/cache/proton_geometry.pt"
ACC_DIR = os.path.join(LANE, "cache")
PIX, LIM = 1.0, 2720.0           # raster pixel [mm] and half-size [mm] of the transverse raster
LAYER = 20.5


def load(batch=256, slab=False):
    C, geo, data = shapelab.setup(batch=batch, out_dir=f"{LANE}/results", slab=slab)
    G = torch.load(GEOM, weights_only=False)
    return C, geo, data, G


def build_raster(cells_xy, half=24.3, close_px=5):
    """Transverse raster of the calorimeter. Returns (tower index int16 (n,n), -1 = no tower box; smooth-inside bool (n,n)).

    Pixel (i, j) has centre x = -LIM + (i + .5) PIX, y = -LIM + (j + .5) PIX. A pixel belongs to the nearest tower whose
    2*half square box contains it. `smooth` = morphological closing of the union (module gaps of 3.8 / 8.5 mm filled)."""
    from scipy import ndimage
    n = int(round(2 * LIM / PIX))
    idx = -np.ones((n, n), np.int16)
    best = np.full((n, n), 1e9, np.float32)
    cx = np.asarray(cells_xy, np.float64)
    for t, (X, Y) in enumerate(cx):
        i0, i1 = int(math.ceil((X - half + LIM) / PIX - 0.5)), int(math.floor((X + half + LIM) / PIX - 0.5))
        j0, j1 = int(math.ceil((Y - half + LIM) / PIX - 0.5)), int(math.floor((Y + half + LIM) / PIX - 0.5))
        xs = -LIM + (np.arange(i0, i1 + 1) + 0.5) * PIX - X
        ys = -LIM + (np.arange(j0, j1 + 1) + 0.5) * PIX - Y
        d = np.maximum(np.abs(xs)[:, None], np.abs(ys)[None, :]).astype(np.float32)
        blk = best[i0:i1 + 1, j0:j1 + 1]
        upd = d < blk
        blk[upd] = d[upd]
        idx[i0:i1 + 1, j0:j1 + 1][upd] = t
    smooth = ndimage.binary_closing(idx >= 0, structure=np.ones((2 * close_px + 1, 2 * close_px + 1), bool))
    smooth |= idx >= 0
    return idx, smooth


class Detector:
    """Raster + tower list on the GPU."""

    def __init__(self, G, dev):
        path = os.path.join(ACC_DIR, "raster.npz")
        if os.path.exists(path):
            z = np.load(path)
            idx, smooth = z["idx"], z["smooth"]
        else:
            idx, smooth = build_raster(G["cells_xy"].numpy())
            os.makedirs(ACC_DIR, exist_ok=True)
            np.savez_compressed(path, idx=idx, smooth=smooth)
        self.idx_np, self.smooth_np = idx, smooth
        self.idx = torch.tensor(idx.astype(np.int64), device=dev)
        self.smooth = torch.tensor(smooth, device=dev)
        self.towers = torch.tensor(G["cells_xy"].numpy(), dtype=torch.float32, device=dev)
        self.z_first = float(G["layers_z"][0])
        self.n = idx.shape[0]

    def pix(self, gx, gy):
        ix = ((gx + LIM) / PIX).floor().long().clamp_(0, self.n - 1)
        iy = ((gy + LIM) / PIX).floor().long().clamp_(0, self.n - 1)
        return ix, iy


@torch.no_grad()
def acceptance(C, det, R, zb, origin, box, nsub=2, nphi=24, want_cells=True, front=True, back=True):
    """Azimuthal acceptance of every (z', r) ring. Returns dict of (B, N_Z, N_R) float32 tensors:
       slab  : closed form (identical to shapelab.slab_accept with the same nsub)
       full  : slab AND smooth transverse outline (numerical: 2*nphi midpoint samples on the slab-allowed arcs)
       cells : slab AND point inside an existing tower box AND that cell's centre passes the report's cuts
       mx, mx2: <cos phi>, <cos 2 phi> of the allowed arc under `full` times `full` (first moments; for task 2/4)"""
    dev = R.device
    Z_EDGES, R_EDGES, DZ = C["Z_EDGES"], C["R_EDGES"], C["DZ"]
    N_Z, N_R = C["N_Z"], C["N_R"]
    k = (torch.arange(nsub, device=dev, dtype=torch.float32) + 0.5) / nsub
    zs = (Z_EDGES[:-1, None] + DZ * k).reshape(-1)                                                    # (N_Z nsub)
    rs = torch.sqrt(R_EDGES[:-1, None] ** 2 + (R_EDGES[1:, None] ** 2 - R_EDGES[:-1, None] ** 2) * k).reshape(-1)
    B = len(R)
    cos_t = R[:, 2, 2]
    sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
    D = zb * cos_t
    half = float(box[2]) / 2
    zc = zs[None, :, None] * cos_t[:, None, None]
    den = (rs[None, None, :] * sin_t[:, None, None]).clamp_min(1e-9)
    lo = ((zc - D[:, None, None]) / den).clamp(-1, 1) if back else -torch.ones(B, len(zs), len(rs), device=dev)
    hi = ((zc + half) / den).clamp(-1, 1) if front else torch.ones(B, len(zs), len(rs), device=dev)
    p_hi, p_lo = torch.acos(hi), torch.acos(lo)                                                       # p_hi <= p_lo
    arc = (p_lo - p_hi) / math.pi                                                                     # (B, NZs, NRs)
    j = (torch.arange(nphi, device=dev, dtype=torch.float32) + 0.5) / nphi
    phi = p_hi[..., None] + (p_lo - p_hi)[..., None] * j                                              # (B, NZs, NRs, nphi)
    cph, sph = torch.cos(phi), torch.sin(phi)
    rr = rs[None, None, :, None]
    x, yabs = rr * cph, rr * sph
    z = zs[None, :, None, None]
    o = origin[:, :, None, None, None]
    Rm = R[:, :, :, None, None, None]                                                                 # R[b, i, j]
    full = torch.zeros_like(arc)
    cells = torch.zeros_like(arc)
    m1 = torch.zeros_like(arc)
    m2 = torch.zeros_like(arc)
    c2 = 2 * cph * cph - 1
    for sgn in (1.0, -1.0):
        y = sgn * yabs
        gx = o[:, 0] + Rm[:, 0, 0] * x + Rm[:, 1, 0] * y + Rm[:, 2, 0] * z
        gy = o[:, 1] + Rm[:, 0, 1] * x + Rm[:, 1, 1] * y + Rm[:, 2, 1] * z
        ix, iy = det.pix(gx, gy)
        ins = det.smooth[ix, iy].float()
        full += ins.mean(-1)
        m1 += (ins * cph).mean(-1)
        m2 += (ins * c2).mean(-1)
        if want_cells:
            t = det.idx[ix, iy]
            gz = o[:, 2] + Rm[:, 0, 2] * x + Rm[:, 2, 2] * z
            lay = torch.round((gz - det.z_first) / LAYER)
            tw = det.towers[t.clamp_min(0)]
            dx, dy, dz = tw[..., 0] - o[:, 0], tw[..., 1] - o[:, 1], det.z_first + LAYER * lay - o[:, 2]
            xc = Rm[:, 0, 0] * dx + Rm[:, 0, 1] * dy + Rm[:, 0, 2] * dz
            yc = Rm[:, 1, 0] * dx + Rm[:, 1, 1] * dy
            zc_ = Rm[:, 2, 0] * dx + Rm[:, 2, 1] * dy + Rm[:, 2, 2] * dz
            ok = (t >= 0) & (zc_ >= C["Z_MIN"]) & (zc_ <= C["Z_MAX"]) & (xc * xc + yc * yc < C["R_MAX"] ** 2)
            cells += ok.float().mean(-1)
    red = lambda a: (a * arc / 2).reshape(B, N_Z, nsub, N_R, nsub).mean((2, 4))
    out = {"slab": arc.reshape(B, N_Z, nsub, N_R, nsub).mean((2, 4)), "full": red(full), "mx": red(m1), "mx2": red(m2)}
    if want_cells:
        out["cells"] = red(cells)
    return out


def geometry_tensors(data, G, split, dev):
    sh, g = data[split], G["sets"][split]
    return sh.ev["R"], sh.ev["zb"], g["origin"].to(dev)


@torch.no_grad()
def build_acceptance(C, det, data, G, split, keys=("full", "cells"), bs=96, nsub=2, nphi=24, log=print):
    """Cache A_<key>(event, z', r) as float16 for a whole split."""
    dev = C["DEVICE"]
    R, zb, org = geometry_tensors(data, G, split, dev)
    box = data[split].box
    out = {k: torch.empty(len(R), C["N_Z"], C["N_R"], dtype=torch.float16) for k in keys}
    t0 = time.time()
    for s in range(0, len(R), bs):
        a = acceptance(C, det, R[s:s + bs], zb[s:s + bs], org[s:s + bs], box, nsub, nphi, want_cells="cells" in keys)
        for k in keys:
            out[k][s:s + bs] = a[k].half().cpu()
        if log and (s // bs) % 100 == 0:
            log(f"  {split}: {s}/{len(R)}  ({time.time() - t0:.0f}s)")
    return out


def load_acc(split, key, dev=None):
    a = torch.load(os.path.join(ACC_DIR, f"acc_{key}_{split}.pt"), weights_only=False)
    return a.to(dev) if dev is not None else a


def batches_idx(sh, size, shuffle=False, gen=None):
    """Same order as Showers.batches, but every batch also carries the event indices (b.idx)."""
    n = len(sh)
    order = (torch.randperm(n, generator=gen) if shuffle else torch.arange(n)).to(sh.off.device)
    for s in range(0, n, size):
        idx = order[s:s + size]
        b = sh.batch(idx)
        b.idx = idx
        yield b


def std_mass(C, logP, zb):
    """The report's normalisation: profile mass in front of the back face (copied from 00_common.event_nll)."""
    N_Z, DZ, Z_MIN = C["N_Z"], C["DZ"], C["Z_MIN"]
    Pz = logP.exp().sum(2)
    cdf = torch.cat([Pz.new_zeros(len(Pz), 1), torch.cumsum(Pz, 1)], 1)
    t = ((zb - Z_MIN) / DZ).clamp(0, N_Z)
    k = t.floor().long().clamp(max=N_Z - 1)
    ar = torch.arange(len(Pz), device=Pz.device)
    return cdf[ar, k] + Pz[ar, k] * (t - k)


def hit_term(C, logP, b, box, K, gen=None):
    """S_e = -sum_i w_i log g(c_i): the folded per-hit part of event_nll (identical code path, identical random numbers)."""
    logrho = logP - C["LOG_VOL"]
    pts = C["fold"](b, box, K, gen)
    iz, ir = C["locate"](pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
    logg = torch.logsumexp(logrho[b.ev[:, None].expand_as(iz), iz, ir], 1) - math.log(K)
    return -torch.zeros(len(b.lnp), device=logg.device).index_add_(0, b.ev, b.w * logg)


def event_nll_acc(C, model, b, box, A, K=None, gen=None):
    """event_nll with the normalisation mass = sum(P * A), A = acceptance of the batch's events (B, N_Z, N_R)."""
    logP = model(b.lnp)
    S = hit_term(C, logP, b, box, K or C["N_FOLD"], gen)
    mass = (logP.exp() * A).sum((1, 2))
    return S + torch.log(mass.clamp_min(1e-8)), logP


@torch.no_grad()
def eval_norms(C, model, sh, accs, K=None):
    """Per-event S_e and log-mass under every normalisation. accs = {name: (N, N_Z, N_R) tensor on the device}.
    Same batches (128), same generator seed and same random-number consumption as 00_common.test_nll, so
    S + logM['std'] reproduces test_nll(model, sh) of the standard harness exactly."""
    dev = C["DEVICE"]
    K = K or C["N_FOLD_EVAL"]
    gen = torch.Generator(device=dev).manual_seed(12345)
    S, logM = [], {k: [] for k in ["std", "slab", *accs]}
    for b in batches_idx(sh, 128):
        logP = model(b.lnp)
        S.append(hit_term(C, logP, b, sh.box, K, gen))
        P = logP.exp()
        logM["std"].append(torch.log(std_mass(C, logP, b.zb).clamp_min(1e-8)))
        logM["slab"].append(torch.log((P * C["event_slab"](b, sh.box)).sum((1, 2)).clamp_min(1e-8)))
        for k, A in accs.items():
            logM[k].append(torch.log((P * A[b.idx].float()).sum((1, 2)).clamp_min(1e-8)))
    return torch.cat(S).cpu().numpy().astype(np.float64), {k: torch.cat(v).cpu().numpy().astype(np.float64) for k, v in logM.items()}


def fit_fast_acc(C, data, model, A_train, A_val, steps=3000, lr=0.03, lr_end=0.002, eval_every=250, seed=0, log=print, clip=5.0):
    """shapelab.fit_fast with the acceptance-normalised likelihood (mass = sum P * A_e); same optimiser, schedule, batches."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr_end)
    gen = torch.Generator().manual_seed(1000 + seed)
    best, best_state, hist, step, t0 = float("inf"), None, [], 0, time.time()
    tr = data["train"]

    @torch.no_grad()
    def val_nll():
        g = torch.Generator(device=C["DEVICE"]).manual_seed(12345)
        return float(torch.cat([event_nll_acc(C, model, b, data["val"].box, A_val[b.idx].float(), C["N_FOLD"], g)[0]
                                for b in batches_idx(data["val"], 128)]).mean())

    model.train()
    while step < steps:
        for b in batches_idx(tr, C["BATCH"], shuffle=True, gen=gen):
            nll, _ = event_nll_acc(C, model, b, tr.box, A_train[b.idx].float())
            loss = nll.mean()
            if torch.isfinite(loss):
                opt.zero_grad()
                loss.backward()
                if all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    opt.step()
            sched.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                model.eval()
                val = val_nll()
                model.train()
                hist.append((step, val))
                if val < best:
                    best, best_state = val, copy.deepcopy(model.state_dict())
                if log:
                    log(f"  step {step:5d}  val {val:.4f}  best {best:.4f}  ({time.time() - t0:.0f}s)")
            if step >= steps:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def paired(a, b):
    return shapelab.paired(a, b)
