"""M1-d D7 - sanity checks of the cross-validated score statistic (CPU, from b3_score_pass.npz).

    CUDA_VISIBLE_DEVICES="" uv run python -B shape-form-martin/results/m1_audit/checks/b3_score_sanity.py

1. Null by random sign flips of the block sums (destroys any common gradient, keeps the noise): the U-statistic must
   average to ~0 and its spread must match the jackknife SE.
2. Sensitivity to the floor on a_k(c) and to the number of blocks used (100 merged blocks instead of 200).
3. Sub-totals of the multinomial-metric estimate for the main knot set: knots <= 4 GeV vs >= 8 GeV, crossed with regions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chk  # noqa: E402
import b3_score_post as post  # noqa: E402

import numpy as np  # noqa: E402


def whitened(f, s, floor):
    N = int(f["n_events"])
    S, A, W = f[f"S_{s}"], f[f"A_{s}"] / N, f[f"W_{s}"] / N
    out = []
    for k in range(S.shape[1]):
        keep = A[k] >= floor * W[k]
        ak = A[k][keep]
        Xb = S[:, k, :][:, keep]
        Xb = Xb - ak[None, :] * (Xb.sum(1, keepdims=True) / ak.sum())
        full = np.zeros((S.shape[0], post.NC))
        full[:, keep] = Xb / np.sqrt(ak)[None, :]
        out.append(full)
    return out


def main():
    f = np.load(chk.HERE / "b3_score_pass.npz")
    nb = f["nblk"]
    names, ridx = post.regions()
    res = {}
    rng = np.random.default_rng(3)
    for s in ("k1", "k6"):
        X = np.concatenate(whitened(f, s, 1e-8), 1)
        U, se, _ = post.ustat(X, nb)
        null = []
        for _ in range(300):
            sg = rng.choice([-1.0, 1.0], size=len(nb))
            null.append(post.ustat(X * sg[:, None], nb)[0] / 2)
        null = np.array(null)
        res[f"{s}_sign_flip_null"] = {"gain": U / 2, "se_jack": se / 2, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                                      "null_frac_ge_observed": float((null >= U / 2).mean())}
        for floor in (1e-6, 1e-10, 0.0):
            Xf = np.concatenate(whitened(f, s, max(floor, 1e-300)), 1)
            Uf, sef, _ = post.ustat(Xf, nb)
            res[f"{s}_floor_{floor:g}"] = {"gain": Uf / 2, "se": sef / 2}
        X2 = X.reshape(100, 2, -1).sum(1)
        U2, se2, _ = post.ustat(X2, nb.reshape(100, 2).sum(1))
        res[f"{s}_100_blocks"] = {"gain": U2 / 2, "se": se2 / 2}
    Xk = whitened(f, "k6", 1e-8)
    knots = f["knots_k6"]
    lowp = np.concatenate(Xk[:3], 1)
    highp = np.concatenate(Xk[3:], 1)
    for lab, X in (("knots 0.5-4 GeV", lowp), ("knots 8-20 GeV", highp)):
        U, se, _ = post.ustat(X, nb)
        res[lab] = {"gain": U / 2, "se": se / 2, "regions": {}}
        rr = np.tile(ridx, X.shape[1] // post.NC)
        for i, nm in enumerate(names):
            Ur, ser, _ = post.ustat(X[:, rr == i], nb)
            res[lab]["regions"][nm] = {"gain": Ur / 2, "se": ser / 2}
    # entry + core + bulk together (the part of the map where the metric is a proper score), all knots
    Xall = np.concatenate(Xk, 1)
    rr = np.tile(ridx, 6)
    inner = np.isin(rr, [0, 1, 2, 3])
    U, se, _ = post.ustat(Xall[:, inner], nb)
    res["inner regions (entry, core, bulk, backward)"] = {"gain": U / 2, "se": se / 2}
    U, se, _ = post.ustat(Xall[:, ~inner], nb)
    res["approximate-score regions (beyond back face, last ring, first/last z-bin)"] = {"gain": U / 2, "se": se / 2}
    chk.write("b3_score_sanity.json", res)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
