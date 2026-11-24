"""M1-d diagnostic D7, pass 1 (GPU) - per-event score of the fine-tuned MLP with respect to free per-cell log-factors,
and the exact curvature blocks. Validation only; no parameter is updated.

    uv run python -B shape-form-martin/results/m1_audit/checks/b3_score_pass.py [--events N]

Perturbation.  log P_delta(c | p) = log P(c | p) + delta(c, p) - log sum_c' P(c' | p) exp(delta(c', p)),
               delta(c, p) = sum_k h_k(ln p) delta[k, c],  h_k = hat functions on momentum knots (partition of unity).

Per-event score (derivation in AUDIT.md). With the fold points of the report metric (K = 64, seed 12345, batch 128)
    q_i(c) = sum_k 1[x_ik in c] rho(c) / sum_k rho(x_ik)        responsibility of cell c for hit i (sums to 1 over c)
    d_e(c) = sum_i w_i q_i(c)                                    energy of event e assigned to cell c by the model
    t_e(c) = P(c) m_e(c) / M_e                                   model truncated at the event's back face
                                                                 (m_e = fraction of the z-bin below zb_e, M_e = sum P m_e)
    dNLL_e / d delta(c) = (sum_i w_i) t_e(c) - d_e(c) =: G_e(c)  (sums to 0 over c: the softmax gauge)
The formula is checked against autograd on the first batch.

Accumulated (float64):
    block sums   S[set][b, k, c] = sum_{e in block b} h_k(e) G_e(c)      200 random event blocks, four knot sets
    a[set][k, c] = sum_e h_k(e) t_e(c);  n_block[b]
    for the main knot set: T_k = sum_e h_k t_e t_e^T  and  Q_k = sum_e h_k sum_i w_i q_i q_i^T   (4096 x 4096 each),
    which give the exact curvature of the knot-lumped problem:  H_k = diag(g_k) + (Q_k - T_k) / N.
Written to b3_score_pass.npz (blocks, a, counts) and b3_score_curv.npy files (Q, T; git-ignored size ~1.6 GB, kept in
the scratch directory given by --scratch).
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

common, metric, legacy = chk.common, chk.metric, chk.legacy
DEV = chk.DEV
NC = common.N_Z * common.N_R

KNOT_SETS = {"k1": [1.0], "k3": [1.0, 6.0, 20.0], "k6": [0.5, 1.5, 4.0, 8.0, 13.0, 20.0],
             "k12": [0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 8.0, 10.5, 13.0, 16.5, 20.0]}
MAIN = "k6"
N_BLOCKS = 200


def hats(lnp, knots):
    """Hat functions in ln p on the knots (constant beyond the end knots): (B,) -> (B, Kn), rows sum to 1."""
    Kn = len(knots)
    h = torch.zeros(len(lnp), Kn, dtype=torch.float64, device=lnp.device)
    if Kn == 1:
        return h + 1.0
    x = lnp.double().clamp(float(knots[0]), float(knots[-1]))
    i = torch.searchsorted(knots, x).clamp(1, Kn - 1)
    t = (x - knots[i - 1]) / (knots[i] - knots[i - 1])
    h.scatter_(1, (i - 1)[:, None], (1 - t)[:, None])
    h.scatter_add_(1, i[:, None], t[:, None])
    return h


def batch_score(logP, b, pts):
    """G (B, NC) float64, t (B, NC) float64, and the pieces needed for Q: cell (H, K) long, resp (H, K) float64."""
    B, H = len(b.lnp), len(b.xyz)
    logrho = logP - common.LOG_VOL
    iz, ir = common.locate(pts[..., 2], torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2))
    lr = logrho[b.ev[:, None].expand_as(iz), iz, ir].double()
    resp = torch.softmax(lr, 1)
    cell = iz * common.N_R + ir
    d = torch.zeros(B * NC, dtype=torch.float64, device=DEV)
    d.index_add_(0, (b.ev[:, None] * NC + cell).reshape(-1), (b.w[:, None].double() * resp).reshape(-1))
    d = d.reshape(B, NC)
    P = logP.double().exp()
    m = ((b.zb[:, None].double() - common.Z_EDGES[None, :-1].double()) / common.DZ).clamp(0, 1)       # (B, N_Z)
    tm = P * m[:, :, None]
    t = (tm / tm.sum((1, 2), keepdim=True)).reshape(B, NC)
    ws = torch.zeros(B, dtype=torch.float64, device=DEV).index_add_(0, b.ev, b.w.double())
    return ws[:, None] * t - d, t, cell, resp


def autograd_check(mlp, b, box):
    """dNLL_e/d delta from autograd through sfm.metric.event_nll, on the same points as batch_score."""
    gen = torch.Generator(device=DEV).manual_seed(12345)
    with torch.no_grad():
        logP = mlp(b.lnp)
    delta = torch.zeros_like(logP, requires_grad=True)
    lp = F.log_softmax((logP + delta).reshape(len(b.lnp), -1), 1).reshape(logP.shape)
    nll = metric.event_nll(lp, b, box, 64, gen)[0]
    (g,) = torch.autograd.grad(nll.sum(), delta)
    gen = torch.Generator(device=DEV).manual_seed(12345)
    with torch.no_grad():
        pts = common.fold(b, box, 64, gen)
        G = batch_score(logP, b, pts)[0]
    diff = (g.reshape(len(b.lnp), -1).double() - G).abs()
    return {"max_abs_diff": float(diff.max()), "max_abs_grad": float(G.abs().max()),
            "max_abs_rowsum_G": float(G.sum(1).abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=0)
    ap.add_argument("--scratch", default=str(chk.HERE / "_scratch"))
    ap.add_argument("--no-curv", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    sets, extras = chk.load_cpu(("val",))
    val_cpu = sets["val"]
    if a.events:
        val_cpu = val_cpu.subset(np.arange(a.events))
    N = len(val_cpu)
    rng = np.random.default_rng(20261124)
    block = torch.as_tensor(rng.permutation(N) % N_BLOCKS)                # random blocks of ~75 events
    with chk.gpu_lock("m1d D7 score pass"):
        val = val_cpu.to(DEV)
        block = block.to(DEV)
        mlp = legacy.load_mlp_std()
        knots = {s: torch.log(torch.tensor(v, dtype=torch.float64, device=DEV)) for s, v in KNOT_SETS.items()}
        S = {s: torch.zeros(N_BLOCKS, len(v), NC, dtype=torch.float64, device=DEV) for s, v in KNOT_SETS.items()}
        A = {s: torch.zeros(len(v), NC, dtype=torch.float64, device=DEV) for s, v in KNOT_SETS.items()}
        W = {s: torch.zeros(len(v), dtype=torch.float64, device=DEV) for s, v in KNOT_SETS.items()}
        Kn = len(KNOT_SETS[MAIN])
        curv = not a.no_curv
        if curv:
            Q = torch.zeros(Kn, NC, NC, dtype=torch.float64, device=DEV)
            T = torch.zeros(Kn, NC, NC, dtype=torch.float64, device=DEV)
        nblk = torch.zeros(N_BLOCKS, dtype=torch.float64, device=DEV)
        nll_sum = 0.0
        first = next(iter(val.batches(128)))
        check = autograd_check(mlp, first, val.box)
        print("autograd check", check, flush=True)
        gen = torch.Generator(device=DEV).manual_seed(12345)
        pos = 0
        with torch.no_grad():
            for nb, b in enumerate(val.batches(128)):
                B = len(b.lnp)
                blk = block[pos:pos + B]
                pos += B
                logP = mlp(b.lnp)
                pts = common.fold(b, val.box, 64, gen)
                G, t, cell, resp = batch_score(logP, b, pts)
                nblk.index_add_(0, blk, torch.ones(B, dtype=torch.float64, device=DEV))
                for s in KNOT_SETS:
                    h = hats(b.lnp, knots[s])                                        # (B, Kn_s)
                    W[s] += h.sum(0)
                    A[s] += h.T @ t
                    for k in range(h.shape[1]):
                        S[s][:, k, :].index_add_(0, blk, h[:, k, None] * G)
                if curv:
                    h = hats(b.lnp, knots[MAIN])
                    R = torch.zeros(len(b.xyz), NC, dtype=torch.float32, device=DEV)
                    R.scatter_add_(1, cell, resp.float())
                    hw = h[b.ev] * b.w[:, None].double()                             # (H, Kn): w_i h_k(e_i)
                    for k in range(Kn):
                        rows = torch.nonzero(hw[:, k] > 0)[:, 0]
                        if len(rows) == 0:
                            continue
                        Rk = R[rows] * hw[rows, k].sqrt().float()[:, None]
                        Q[k] += (Rk.T @ Rk).double()
                        del Rk
                        ev_rows = torch.nonzero(h[:, k] > 0)[:, 0]
                        tk = t[ev_rows] * h[ev_rows, k].sqrt()[:, None]
                        T[k] += tk.T @ tk
                    del R
                if nb % 20 == 0:
                    print(f"batch {nb}  t = {time.time() - t0:.0f} s", flush=True)
        out = {"n_events": N, "n_blocks": N_BLOCKS, "nblk": nblk.cpu().numpy(), "block_of_event": block.cpu().numpy()}
        for s in KNOT_SETS:
            out[f"S_{s}"] = S[s].cpu().numpy()
            out[f"A_{s}"] = A[s].cpu().numpy()
            out[f"W_{s}"] = W[s].cpu().numpy()
            out[f"knots_{s}"] = np.array(KNOT_SETS[s])
        tag = f"_n{a.events}" if a.events else ""
        if curv:
            np.save(scratch / f"b3_Q{tag}.npy", Q.cpu().numpy())
            np.save(scratch / f"b3_T{tag}.npy", T.cpu().numpy())
    np.savez_compressed(chk.HERE / f"b3_score_pass{tag}.npz", **out)
    chk.write(f"b3_score_pass{tag}.json", {"autograd_check": check, "n_events": N, "knot_sets": KNOT_SETS,
                                           "main": MAIN, "n_blocks": N_BLOCKS, "curvature": curv,
                                           "scratch": str(scratch)}, t0)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
