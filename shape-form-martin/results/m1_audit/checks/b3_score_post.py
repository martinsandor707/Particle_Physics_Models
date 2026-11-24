"""M1-d diagnostic D7, pass 2 (CPU) - cross-validated score test from the block sums of ``b3_score_pass.py``.

    CUDA_VISIBLE_DEVICES="" uv run python -B shape-form-martin/results/m1_audit/checks/b3_score_post.py --scratch <dir>

Notation (per knot k; N events, mean validation NLL = (1/N) sum_e NLL_e)
    g_k(c) = (1/N) sum_e h_k(e) G_e(c)        gradient of the mean NLL w.r.t. delta[k, c]   (sum_c g_k = 0)
    a_k(c) = (1/N) sum_e h_k(e) t_e(c)        knot-weighted truncated model mass;  w_k = sum_c a_k(c) = mean h_k
    Ftil_k = diag(a_k) - a_k a_k^T / w_k      multinomial ("softmax") metric of the knot-lumped problem
    H_k    = diag(g_k) + (Q_k - T_k)/N        exact curvature of the knot-lumped problem (hit term included)
Bounds (AUDIT.md): H (exact, with knot coupling) <= blockdiag(H_k-lumped) ... <= blockdiag(Ftil_k), so for the
second-order model  min_delta [g.delta + delta^T H delta / 2] <= -(1/2) g^T Ftil^+ g = -(1/2) sum_k sum_c g_k(c)^2/a_k(c).

Estimators (all cross-validated: products of gradients from DISJOINT sets of events only)
    block U-statistic   U = sum_{b != b'} x_b^T M x_b' / sum_{b != b'} n_b n_b'    (x_b = block sum of h_k G_e; 200 blocks)
                        gain = U / 2;  SE from the delete-one-block jackknife
    half splits         (1/2) g_A^T M g_B for random halves of the blocks (mean and spread over 400 splits)
    M = Ftil^+ (lambda_min = 1) or the damped Newton metric: in the eigenbasis of H relative to Ftil (eigenvalues
    lambda_j <= 1), M = sum_j v_j v_j^T / max(lambda_j, lambda_min). Every lambda_min gives a lower bound of the
    second-order optimum; lambda_min -> 0 is the Newton value.
Cells with a_k(c) < floor * w_k are left unperturbed (delta = 0 there), and the gauge sum_c a_k(c) delta[k, c] = 0 is
fixed by projection (then delta[k, c] is the first-order change of ln P).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

NZ, NR = 128, 32
NC = NZ * NR
Z_MIN, Z_MAX, R_MAX = -200.0, 1450.0, 700.0
ZE = np.linspace(Z_MIN, Z_MAX, NZ + 1)
ZC = 0.5 * (ZE[1:] + ZE[:-1])
RE = R_MAX * (np.arange(NR + 1) / NR) ** 1.7
LMINS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02)


def regions():
    """Disjoint partition of the 128 x 32 cells (by z-bin centre and ring index)."""
    reg = np.empty((NZ, NR), dtype=object)
    zc = ZC[:, None] + np.zeros((1, NR))
    ring = np.arange(NR)[None, :] + np.zeros((NZ, 1), dtype=int)
    reg[:] = "bulk (60<=z'<1221, rings 5-30)"
    reg[(zc >= 60) & (zc < 1221) & (ring <= 4)] = "core r<30 mm (60<=z'<1221)"
    reg[(zc >= -20) & (zc < 60)] = "entry (-20<=z'<60)"
    reg[zc < -20] = "backward (z'<-20)"
    reg[zc >= 1221] = "beyond typical back face (z'>=1221)"
    reg[:, NR - 1] = "border: last ring"
    reg[0, :] = "border: first/last z-bin"
    reg[NZ - 1, :] = "border: first/last z-bin"
    names = ["entry (-20<=z'<60)", "core r<30 mm (60<=z'<1221)", "bulk (60<=z'<1221, rings 5-30)", "backward (z'<-20)",
             "beyond typical back face (z'>=1221)", "border: last ring", "border: first/last z-bin"]
    idx = np.array([names.index(v) for v in reg.reshape(-1)])
    return names, idx


def block_partition(bz, br):
    """Cell -> block index for delta constant on bz x br blocks of cells."""
    iz = np.arange(NZ)[:, None] // bz
    ir = np.arange(NR)[None, :] // br
    return (iz * (NR // br) + ir).reshape(-1)


def ustat(Xb, nb, m=None):
    """Block U-statistic of sum_j m_j x_j x'_j with jackknife SE. Xb (n_blocks, d); m (d,) or None (= 1).
    Returns (U, se, U_jack (n_blocks,))."""
    w = 1.0 if m is None else m
    tot = Xb.sum(0)
    diag = (Xb * Xb * w).sum(1)                                   # x_b^T M x_b
    full = (tot * tot * w).sum() - diag.sum()
    den = nb.sum() ** 2 - (nb ** 2).sum()
    U = full / den
    # leave one block out
    cross = (Xb * tot * w).sum(1)                                 # x_b^T M tot
    full_j = (tot * tot * w).sum() - 2 * cross + diag - (diag.sum() - diag)
    nj = nb.sum() - nb
    den_j = nj ** 2 - ((nb ** 2).sum() - nb ** 2)
    Uj = full_j / den_j
    J = len(nb)
    se = np.sqrt((J - 1) / J * ((Uj - Uj.mean()) ** 2).sum())
    return float(U), float(se), Uj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=str(chk.HERE / "_scratch"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--floor", type=float, default=1e-8)
    ap.add_argument("--splits", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    f = np.load(chk.HERE / f"b3_score_pass{a.tag}.npz")
    meta = json.loads((chk.HERE / f"b3_score_pass{a.tag}.json").read_text())
    N, nb = int(f["n_events"]), f["nblk"]
    J = len(nb)
    names, ridx = regions()
    rng = np.random.default_rng(7)
    halves = np.stack([rng.permutation(J) < J // 2 for _ in range(a.splits)])        # (R, J) bool
    out = {"n_events": N, "n_blocks": J, "floor": a.floor, "autograd_check": meta["autograd_check"], "sets": {}}

    def half_cross(Xb, m=None):
        """(1/2) g_A^T M g_B over the random half splits: mean, sd, and the in-sample (1/2) g_A^T M g_A mean."""
        w = 1.0 if m is None else m
        SA = halves.astype(np.float64) @ Xb
        SB = (~halves).astype(np.float64) @ Xb
        nA, nB = halves @ nb, (~halves) @ nb
        gA, gB = SA / nA[:, None], SB / nB[:, None]
        cr = 0.5 * (gA * gB * w).sum(1)
        self_ = 0.5 * 0.5 * ((gA * gA * w).sum(1) + (gB * gB * w).sum(1))
        return float(cr.mean()), float(cr.std(ddof=1)), float(self_.mean())

    # ---- multinomial metric for every knot set -------------------------------------------------------------------------
    for s in ("k1", "k3", "k6", "k12"):
        S, A, W = f[f"S_{s}"], f[f"A_{s}"] / N, f[f"W_{s}"] / N                      # S (J, Kn, NC)
        Kn = S.shape[1]
        res = {"knots_GeV": f[f"knots_{s}"], "w_k": W, "per_knot": [], "n_cells_kept": []}
        Xw_all = []
        for k in range(Kn):
            keep = A[k] >= a.floor * W[k]
            ak = A[k][keep]
            Xb = S[:, k, :][:, keep]
            Xb = Xb - ak[None, :] * (Xb.sum(1, keepdims=True) / ak.sum())            # gauge: sum_c a delta = 0 on the kept cells
            Xw = Xb / np.sqrt(ak)[None, :]                                           # whitened: Ftil^+ -> identity
            U, se, _ = ustat(Xw, nb)
            res["per_knot"].append({"gain": U / 2, "se": se / 2})
            res["n_cells_kept"].append(int(keep.sum()))
            full = np.zeros((J, NC))
            full[:, keep] = Xw
            Xw_all.append(full)
        Xw_all = np.concatenate(Xw_all, 1)                                           # (J, Kn * NC)
        U, se, Uj = ustat(Xw_all, nb)
        hm, hs, hself = half_cross(Xw_all)
        gtot = Xw_all.sum(0) / N
        res["multinomial"] = {"gain_U": U / 2, "se_jack": se / 2, "half_split_mean": hm, "half_split_sd": hs,
                              "in_sample_half_mean": hself, "in_sample_full": float(0.5 * (gtot ** 2).sum())}
        # noise trace: E[(1/2) g_A M g_A] = S/2 + T/(2 n_A)  ->  T/(2 n) for a fit on n events
        nA = N / 2
        Tr = 2 * nA * (hself - U / 2)
        Ssig = U                                                                      # gamma^T M gamma
        res["noise"] = {"trace_T": Tr, "signal_S": Ssig,
                        "one_step_gain_unshrunk": {str(n): 0.5 * (Ssig - Tr / n) for n in (N // 2, N, 17446, 34892, 69783)},
                        "one_step_gain_best_scalar_shrinkage": {str(n): 0.5 * Ssig ** 2 / (Ssig + Tr / n) if Ssig > 0 else 0.0
                                                                for n in (N // 2, N, 17446, 34892, 69783)}}
        # regions (diagonal metric: additive over cells)
        reg = {}
        ridx_all = np.tile(ridx, Kn)
        for i, nm in enumerate(names):
            sel = ridx_all == i
            Ur, ser, _ = ustat(Xw_all[:, sel], nb)
            reg[nm] = {"gain": Ur / 2, "se": ser / 2, "n_cells": int((ridx == i).sum())}
        res["regions"] = reg
        # coarser parametrisations: delta constant on blocks of cells (still the multinomial metric)
        coarse = {}
        for bz, br in ((2, 2), (4, 4), (8, 8), (16, 16)):
            part = block_partition(bz, br)
            nblkc = part.max() + 1
            Xc = []
            for k in range(Kn):
                ac = np.bincount(part, weights=A[k], minlength=nblkc)
                Sc = np.stack([np.bincount(part, weights=S[b, k], minlength=nblkc) for b in range(J)])
                keep = ac >= a.floor * W[k]
                Sc, ac = Sc[:, keep], ac[keep]
                Sc = Sc - ac[None, :] * (Sc.sum(1, keepdims=True) / ac.sum())
                Xc.append(Sc / np.sqrt(ac)[None, :])
            Xc = np.concatenate(Xc, 1)
            Uc, sec, _ = ustat(Xc, nb)
            coarse[f"{bz}x{br}"] = {"gain": Uc / 2, "se": sec / 2, "n_free": int(Xc.shape[1])}
        res["coarse_blocks"] = coarse
        out["sets"][s] = res
        print(s, "multinomial", res["multinomial"], "per knot", [(round(x["gain"] * 1e4, 3), round(x["se"] * 1e4, 3)) for x in res["per_knot"]])
        print("   regions", {k: (round(v["gain"] * 1e4, 3), round(v["se"] * 1e4, 3)) for k, v in reg.items()})
        print("   coarse", {k: (round(v["gain"] * 1e4, 3), round(v["se"] * 1e4, 3), v["n_free"]) for k, v in coarse.items()})
        print("   noise", res["noise"])

    # ---- exact curvature of the main set: damped Newton ------------------------------------------------------------------
    qfile = Path(a.scratch) / f"b3_Q{a.tag}.npy"
    if qfile.exists():
        s = meta["main"]
        S, A, W = f[f"S_{s}"], f[f"A_{s}"] / N, f[f"W_{s}"] / N
        Kn = S.shape[1]
        Q = np.load(qfile, mmap_mode="r")
        T = np.load(Path(a.scratch) / f"b3_T{a.tag}.npy", mmap_mode="r")
        C_all, lam_all, cellstep = [], [], []
        spec = []
        for k in range(Kn):
            keep = A[k] >= a.floor * W[k]
            ak = A[k][keep]
            gk = S[:, k, :].sum(0) / N
            Hk = (np.asarray(Q[k]) - np.asarray(T[k]))[np.ix_(keep, keep)] / N
            Hk[np.diag_indices_from(Hk)] += gk[keep]
            d = 1.0 / np.sqrt(ak)
            Ht = Hk * d[:, None] * d[None, :]
            Ht = 0.5 * (Ht + Ht.T)
            sv = np.sqrt(ak / ak.sum())
            Hs = Ht @ sv
            Ht = Ht - np.outer(sv, Hs) - np.outer(Hs, sv) + np.outer(sv, sv) * (sv @ Hs)   # P Ht P, P = I - s s^T
            lam, V = sla.eigh(Ht)
            # drop the gauge direction (eigenvalue 0 along s)
            ov = np.abs(V.T @ sv)
            j0 = int(np.argmax(ov))
            sel = np.ones(len(lam), bool)
            sel[j0] = False
            lam, V = lam[sel], V[:, sel]
            Xb = S[:, k, :][:, keep]
            Xb = Xb - ak[None, :] * (Xb.sum(1, keepdims=True) / ak.sum())
            C = (Xb * d[None, :]) @ V                                                # (J, n_dir) block sums in the eigenbasis
            C_all.append(C)
            lam_all.append(lam)
            cellstep.append((keep, d, V))
            qs = np.quantile(lam, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
            spec.append({"knot_GeV": float(f[f"knots_{s}"][k]), "n_dir": int(len(lam)), "quantiles": qs,
                         "frac_lam_lt_0": float((lam < 0).mean()), "frac_lam_lt_0.1": float((lam < 0.1).mean()),
                         "frac_lam_lt_0.5": float((lam < 0.5).mean()), "frac_lam_gt_1.05": float((lam > 1.05).mean())})
            print("knot", k, "eig quantiles", np.round(qs, 4), "t =", round(time.time() - t0), flush=True)
        Call, lam = np.concatenate(C_all, 1), np.concatenate(lam_all)
        newton = {}
        offs = np.cumsum([0] + [c.shape[1] for c in C_all])
        for lmin in LMINS:
            m = 1.0 / np.maximum(lam, lmin)
            U, se, _ = ustat(Call, nb, m)
            hm, hs, hself = half_cross(Call, m)
            # implied step: delta = -M g (whitened norm^2 = sum_j (c_j m_j)^2; model-weighted variance of the log-factor)
            U2, se2, _ = ustat(Call, nb, m * m)
            per_knot, rms = [], []
            for k in range(Kn):
                sl = slice(offs[k], offs[k + 1])
                Uk, sek, _ = ustat(Call[:, sl], nb, m[sl])
                U2k, _, _ = ustat(Call[:, sl], nb, m[sl] ** 2)
                per_knot.append({"gain": Uk / 2, "se": sek / 2, "rms_log_factor": float(np.sqrt(max(U2k, 0.0) / W[k]))})
            # the same split by curvature band (where does the signal live?)
            bands = {}
            for lo, hi in ((-np.inf, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, np.inf)):
                selb = (lam >= lo) & (lam < hi)
                if selb.any():
                    Ub, seb, _ = ustat(Call[:, selb], nb, m[selb])
                    U1, se1, _ = ustat(Call[:, selb], nb)
                    bands[f"[{lo},{hi})"] = {"n_dir": int(selb.sum()), "gain": Ub / 2, "se": seb / 2,
                                             "gain_multinomial_metric": U1 / 2, "se_multinomial_metric": se1 / 2}
            # regions: -1/2 sum_{c in region} g(c) delta(c), delta = -M g in the gauge sum a delta = 0 (cross-validated)
            reg = {nm: 0.0 for nm in names}
            regj = {nm: np.zeros(J) for nm in names}
            for k in range(Kn):
                keep, d, V = cellstep[k]
                sl = slice(offs[k], offs[k + 1])
                Xb = S[:, k, :][:, keep]
                Yb = ((C_all[k] * m[sl][None, :]) @ V.T) * d[None, :]                # M x_b in cell space (kept cells)
                rk = ridx[keep]
                Xt, Yt = Xb.sum(0), Yb.sum(0)
                den = nb.sum() ** 2 - (nb ** 2).sum()
                for i, nm in enumerate(names):
                    c = rk == i
                    val = (Xt[c] * Yt[c]).sum() - (Xb[:, c] * Yb[:, c]).sum()
                    reg[nm] += 0.5 * val / den
                    # jackknife pieces
                    Xj, Yj = Xt[None, c] - Xb[:, c], Yt[None, c] - Yb[:, c]
                    dj = (Xb[:, c] * Yb[:, c]).sum(1)
                    valj = (Xj * Yj).sum(1) - ((Xb[:, c] * Yb[:, c]).sum() - dj)
                    nj = nb.sum() - nb
                    regj[nm] += 0.5 * valj / (nj ** 2 - ((nb ** 2).sum() - nb ** 2))
            reg_out = {nm: {"gain": float(reg[nm]),
                            "se": float(np.sqrt((J - 1) / J * ((regj[nm] - regj[nm].mean()) ** 2).sum()))} for nm in names}
            newton[str(lmin)] = {"gain_U": U / 2, "se_jack": se / 2, "half_split_mean": hm, "half_split_sd": hs,
                                 "in_sample_half_mean": hself, "rms_log_factor_all": float(np.sqrt(max(U2, 0.0))),
                                 "per_knot": per_knot, "bands": bands, "regions": reg_out}
            print("lambda_min", lmin, {k: v for k, v in newton[str(lmin)].items() if k not in ("per_knot", "bands", "regions")})
            print("    per knot", [(round(x["gain"] * 1e4, 3), round(x["se"] * 1e4, 3), round(x["rms_log_factor"], 3)) for x in per_knot])
            print("    regions", {k: (round(v["gain"] * 1e4, 3), round(v["se"] * 1e4, 3)) for k, v in reg_out.items()})
            print("    bands", {k: (v["n_dir"], round(v["gain"] * 1e4, 3), round(v["se"] * 1e4, 3), round(v["gain_multinomial_metric"] * 1e4, 3)) for k, v in bands.items()})
        out["curvature"] = {"set": s, "spectrum": spec, "damped_newton": newton}
    chk.write(f"b3_score{a.tag}.json", out, t0)


if __name__ == "__main__":
    main()
