"""Záporonkénti ingadozás (az AtlFast3/FastCaloSimV2 mintájára): főkomponens-elemzés az eseményenkénti mélységi és sugárirányú
energiaprofilokon, és egy erre épülő egyszerű eseménygenerátor kiértékelése.

Eseményenként: a lerakódott energia hányada 34 mélységi binben (z' < 0; 0–1312 mm 41 mm-enként; 1312–1450 mm) és 9 sugárirányú binben
(0, 25, 50, 100, 150, 200, 300, 400, 550, 700 mm) – a kettő együtt egy 43 elemű vektor. Impulzus-tartományonként (0–2, 2–5, 5–10,
10–20 GeV) a tanítóhalmazon PCA; a generátor a tanító átlagához k főkomponenst ad (Gauss-együtthatókkal, ill. a tanító együtthatóinak
véletlen újrahúzásával), negatív részeket levágva és újranormálva. Kiértékelés a teszthalmazon, eseményenkénti mennyiségekkel:
a mélységi súlypont és szórás, az átlagos sugár, az r < 50 mm-es és a z' < 205 mm-es (első ~10 réteg) energiahányad. Mérce: a
szeparációs erő az adat 25 kvantilis-binjén (× 10⁻³); padló: a teszthalmaz két véletlen felének S-e.
    python fluctuations_pca.py <particle>        kimenet: results/particles/<p>/fluctuations_pca.json"""
import json
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np

import shapelab

particle = sys.argv[1]
D = os.path.join(ROOT, "results", "particles", particle)
C, geo, data = shapelab.setup_particle(particle, out_dir=D, slab=False)
ZB = np.concatenate([[-200.0], np.arange(0.0, 1312.0 + 1, 41.0), [1450.0]])
RB = np.array([0.0, 25, 50, 100, 150, 200, 300, 400, 550, 700])
ZC, RC = (ZB[:-1] + ZB[1:]) / 2, (RB[:-1] + RB[1:]) / 2
NZ = len(ZC)


def profiles(sh):
    h = sh.hits()
    n = len(sh)
    iz = np.clip(np.searchsorted(ZB, h.z.values, side="right") - 1, 0, NZ - 1)
    ir = np.clip(np.searchsorted(RB, h.r.values, side="right") - 1, 0, len(RC) - 1)
    L = np.zeros((n, NZ)); np.add.at(L, (h.ev.values, iz), h.w.values)
    R = np.zeros((n, len(RC))); np.add.at(R, (h.ev.values, ir), h.w.values)
    p = sh.ev["p"].cpu().numpy()
    return np.concatenate([L, R], 1), p


def observables(X):
    L, R = X[:, :NZ], X[:, NZ:]
    L = L / np.maximum(L.sum(1, keepdims=True), 1e-12); R = R / np.maximum(R.sum(1, keepdims=True), 1e-12)
    cz = L @ ZC
    return {"súlypont z'": cz, "mélységi szórás": np.sqrt(np.maximum(L @ ZC ** 2 - cz ** 2, 0)), "átlagos sugár": R @ RC,
            "r < 50 mm hányad": R[:, :2].sum(1), "z' < 205 mm hányad": L[:, :6].sum(1)}


def sep(a, b, nb=25):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, nb + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    h, _ = np.histogram(a, edges); g, _ = np.histogram(b, edges)
    h, g = h / h.sum(), g / g.sum()
    m = (h + g) > 0
    return float(0.5 * np.sum((h[m] - g[m]) ** 2 / (h[m] + g[m])))


def generate(mu, V, scores_tr, k, n, rng, mode):
    if k == 0:
        X = np.repeat(mu[None], n, 0)
    else:
        if mode == "gauss":
            a = rng.normal(size=(n, k)) * scores_tr[:, :k].std(0)
        else:                                                   # resample the empirical training scores (keeps non-Gaussian tails)
            a = scores_tr[rng.integers(0, len(scores_tr), n), :k]
        X = mu[None] + a @ V[:k]
    X = np.clip(X, 0, None)
    X[:, :NZ] /= np.maximum(X[:, :NZ].sum(1, keepdims=True), 1e-12)
    X[:, NZ:] /= np.maximum(X[:, NZ:].sum(1, keepdims=True), 1e-12)
    return X


Xtr, ptr = profiles(data["train"])
Xte, pte = profiles(data["test"])
rng = np.random.default_rng(0)
res = {"particle": particle, "groups": {}}
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20.5)):
    tr, te = (ptr >= lo) & (ptr < hi), (pte >= lo) & (pte < hi)
    A, B = Xtr[tr], Xte[te]
    mu = A.mean(0)
    U, Sv, Vt = np.linalg.svd(A - mu, full_matrices=False)
    var = Sv ** 2 / (len(A) - 1)
    evr = var / var.sum()
    scores = (A - mu) @ Vt.T
    obs_d = observables(B)
    half = rng.random(len(B)) < 0.5
    obs_a, obs_b = observables(B[half]), observables(B[~half])
    g = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "explained_variance": evr[:10].tolist(),
         "pc1_corr_with_cog_z": float(np.corrcoef(scores[:, 0], observables(A)["súlypont z'"])[0, 1]),
         "pc2_corr_with_mean_r": float(np.corrcoef(scores[:, 1], observables(A)["átlagos sugár"])[0, 1]),
         "floor": {o: sep(obs_a[o], obs_b[o]) for o in obs_d}, "S": {}}
    for mode in ("gauss", "empirical"):
        for k in (0, 1, 2, 3, 5, 10):
            if mode == "empirical" and k == 0:
                continue
            Xg = generate(mu, Vt, scores, k, len(B) * 4, rng, mode)
            og = observables(Xg)
            g["S"][f"{mode}, k={k}"] = {o: sep(obs_d[o], og[o]) for o in obs_d}
    res["groups"][f"{lo}-{hi} GeV"] = g
    print(particle, f"{lo}-{hi} GeV", "EVR", [round(x, 3) for x in evr[:5]], "corr", round(g["pc1_corr_with_cog_z"], 2),
          {k_: round(1e3 * np.nanmean(list(v.values())), 2) for k_, v in g["S"].items()}, "floor", round(1e3 * np.nanmean(list(g["floor"].values())), 2), flush=True)
json.dump(res, open(os.path.join(D, "fluctuations_pca.json"), "w"), indent=1, ensure_ascii=False)
