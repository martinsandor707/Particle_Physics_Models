"""Mennyire egyezik a modellek átlagos mélységi (z') és oldalirányú (r) energiaprofilja az adattal? (független s2 minta)

A 3D-mérce szerint tanított modellek (az ugyanígy illesztett baseline, az új formula, a finomhangolt háló) minden s2 eseményre
lerakódásokat sorsolnak az esemény detektor-szeletén belül (P × az esemény szelet-elfogadása, cellán belül pontos vágással).
A likelihood szerint a modell a találatok *cellaközéppontjainak* eloszlását ρ ⊗ cella alakban jósolja; ezért az adat találatait egyszer
szétkenjük a cellájukban (8 pont, mint a fitben), a modell mintáit kétszer (ρ ⊗ cella ⊗ cella), így a cellarács mintázata egyik oldalon
sem jelenik meg. Binek: z' 41 mm (két réteg), r 0, 12, 25, 50, 75, 100, 150, 200, 300, 400, 550, 700 mm. 1 GeV-es binenként:
  - szeparációs erő (a CaloChallenge mércéje): S = ½ Σ_i (h_i − g_i)² / (h_i + g_i) a normált hisztogramokra;
  - a statisztikai padló: az adat két véletlen felének S-e / 4 (ennyi S jönne egy tökéletes modellre is a véges minta miatt);
  - az energiával súlyozott átlagos z' és r eltérése [mm], és az 1. Wasserstein-távolság [mm].
    python profile_metrics.py <particle>          kimenet: results/particles/<p>/profile_metrics.json"""
import json
import math
import os
import sys

ROOT = "/home/user/tamas/shape-form-proton"
sys.path.insert(0, ROOT)
import numpy as np
import torch

import shapelab

particle = sys.argv[1]
D = os.path.join(ROOT, "results", "particles", particle)
C, geo, d2 = shapelab.setup_particle(particle, out_dir=D, slab=False, sample="s2")
dev = C["DEVICE"]
t = json.load(open(os.path.join(D, "test_v3.json")))

# ---------------- the three 3D-trained models
nb = t["degrees"]["base_slab"]
long = shapelab.REPORT_BASE[particle][1]
sd = torch.load(os.path.join(D, f"base_slab_n{nb}.pt"), map_location=dev, weights_only=False)["state"]
base = C["Formula"](long, "core_tail", degree=nb).to(dev)
base.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("f.")})
new = shapelab.build_final(C, particle, D, 0.0, 1.0, metric="slab")[0]
sm = torch.load(os.path.join(D, "mlp_slab.pt"), map_location=dev, weights_only=False)["state"]
n_lin = sum(1 for k in sm if k.startswith("mlp.") and k.endswith(".weight"))
mlp = C["FieldModel"](sm["mlp.0.weight"].shape[0], n_lin - 1).to(dev)
mlp.load_state_dict(sm)
MODELS = {"baseline": base.eval(), "new formula": new.eval(), "MLP": mlp.eval()}

ZE, RE, DZ, N_R = C["Z_EDGES"], C["R_EDGES"], C["DZ"], C["N_R"]
ZB = np.append(np.arange(-200.0, 1450.0, 41.0), 1450.0)       # ~41 mm depth bins (two layers)
RB = np.array([0.0, 12, 25, 50, 75, 100, 150, 200, 300, 400, 550, 700])


@torch.no_grad()
def sample_folded_slab(model, sh, M=256, seed=7):
    """Deposits drawn from the model's body of revolution, kept inside the event's slab (rejection), folded with the cell.
    Returns per-deposit (z', r, event index, weight) with weights summing to 1 per event."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    zs, rs, es, ws = [], [], [], []
    base_ev = 0
    for b in sh.batches(128):
        n = len(b.lnp)
        A = C["event_slab"](b, sh.box, 4)                                     # azimuthal acceptance of every ring cell
        cell = torch.multinomial((model(b.lnp).exp() * A).reshape(n, -1) + 1e-30, M, True, generator=gen)
        iz, ir = cell // N_R, cell % N_R
        u = torch.rand(n, M, 3, device=dev, generator=gen)
        z = ZE[iz] + u[..., 0] * DZ
        r = torch.sqrt(RE[ir] ** 2 + u[..., 1] * (RE[ir + 1] ** 2 - RE[ir] ** 2))
        pts = torch.stack([r * torch.cos(2 * math.pi * u[..., 2]), r * torch.sin(2 * math.pi * u[..., 2]), z], -1)
        ct = b.R[:, 2, 2][:, None]
        zt = pts[..., 2] * ct - pts[..., 0] * torch.sqrt(1 - ct ** 2)
        keep = (zt >= -float(sh.box[2]) / 2) & (zt <= b.zb[:, None] * ct)
        box = (torch.rand(n, M, 3, device=dev, generator=gen) - 0.5) * sh.box \
            + (torch.rand(n, M, 3, device=dev, generator=gen) - 0.5) * sh.box               # rho (x) cell (x) cell
        pts = pts + torch.einsum("bij,bmj->bmi", b.R, box)
        # cells were drawn proportional to P*A and the exact cut keeps a fraction ~A of each ring: weight 1/A so that the kept
        # deposits follow P*A (in-slab density), not P*A^2
        a = A[torch.arange(n, device=dev)[:, None], iz, ir].clamp_min(1e-6)
        wk = keep.float() / a
        w = wk / wk.sum(1, keepdim=True).clamp_min(1e-30)
        evi = (torch.arange(n, device=dev) + base_ev)[:, None].expand(-1, M)
        zs.append(pts[..., 2][keep]); rs.append(torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)[keep])
        es.append(evi[keep]); ws.append(w[keep])
        base_ev += n
    return [torch.cat(x).cpu().numpy() for x in (zs, rs, es, ws)]


@torch.no_grad()
def data_deposits(sh, K=8, seed=3):
    """The hits folded once with their cell (K points each, weight w/K), as in the likelihood."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    zs, rs, es, ws = [], [], [], []
    base_ev = 0
    for b in sh.batches(256):
        pts = C["fold"](b, sh.box, K, gen)                                      # (hits, K, 3) in the shower frame
        zs.append(pts[..., 2].reshape(-1)); rs.append(torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2).reshape(-1))
        es.append((b.ev + base_ev)[:, None].expand(-1, K).reshape(-1)); ws.append((b.w / K)[:, None].expand(-1, K).reshape(-1))
        base_ev += len(b.lnp)
    return [torch.cat(x).cpu().numpy() for x in (zs, rs, es, ws)]


def sep_power(h, g):
    h, g = h / h.sum(), g / g.sum()
    m = (h + g) > 0
    return float(0.5 * np.sum((h[m] - g[m]) ** 2 / (h[m] + g[m])))


def wasserstein(x_edges, h, g):
    """W1 between two binned distributions on the same edges [mm]."""
    ch, cg = np.cumsum(h / h.sum()), np.cumsum(g / g.sum())
    return float(np.sum(np.abs(ch - cg)[:-1] * np.diff(x_edges)[:-1])) if len(h) > 1 else 0.0


out = {"particle": particle, "sample": "s2", "bins": {}, "models": list(MODELS)}
HIST = {}
sets = list(d2.values())
for b in range(20):
    parts = [s.subset((s.ev["bin"] == b).nonzero().squeeze(1)) for s in sets]
    parts = [s for s in parts if len(s) > 0]
    if not parts:
        continue
    dd = [data_deposits(s) for s in parts]
    offs = np.cumsum([0] + [len(s) for s in parts[:-1]])
    dz, dr, dw = (np.concatenate([x[i] for x in dd]) for i in (0, 1, 3))
    de = np.concatenate([x[2] + o for x, o in zip(dd, offs)])                     # event index across the parts
    n_ev = int(sum(len(s) for s in parts))
    hz = np.histogram(dz, ZB, weights=dw)[0]; hr = np.histogram(dr, RB, weights=dw)[0]
    rng = np.random.default_rng(b)
    half = rng.random(n_ev) < 0.5
    hz_a = np.histogram(dz[half[de]], ZB, weights=dw[half[de]])[0]; hz_b = np.histogram(dz[~half[de]], ZB, weights=dw[~half[de]])[0]
    hr_a = np.histogram(dr[half[de]], RB, weights=dw[half[de]])[0]; hr_b = np.histogram(dr[~half[de]], RB, weights=dw[~half[de]])[0]
    HIST.setdefault(b, {})["data"] = (hz / hz.sum(), hr / hr.sum())
    res = {"n_events": int(n_ev), "p_mean": float(np.mean(np.concatenate([s.ev["p"].cpu().numpy() for s in parts]))),
           "floor_S_z": sep_power(hz_a, hz_b) / 4, "floor_S_r": sep_power(hr_a, hr_b) / 4,
           "data_mean_z": float(np.sum(dz * dw) / np.sum(dw)), "data_mean_r": float(np.sum(dr * dw) / np.sum(dw))}
    for name, m in MODELS.items():
        mz, mr, me, mw = (np.concatenate(x) for x in zip(*[sample_folded_slab(m, s) for s in parts]))
        gz = np.histogram(mz, ZB, weights=mw)[0]; gr = np.histogram(mr, RB, weights=mw)[0]
        HIST[b][name] = (gz / gz.sum(), gr / gr.sum())
        res[name] = {"S_z": sep_power(hz, gz), "S_r": sep_power(hr, gr), "W1_z_mm": wasserstein(ZB, hz, gz), "W1_r_mm": wasserstein(RB, hr, gr),
                     "d_mean_z_mm": float(np.sum(mz * mw) / np.sum(mw)) - res["data_mean_z"],
                     "d_mean_r_mm": float(np.sum(mr * mw) / np.sum(mw)) - res["data_mean_r"]}
    out["bins"][b] = res
    print(particle, b, {k: (round(v["S_z"] * 1e3, 2), round(v["S_r"] * 1e3, 2)) for k, v in res.items() if isinstance(v, dict)},
          "floor", round(res["floor_S_z"] * 1e3, 3), round(res["floor_S_r"] * 1e3, 3), flush=True)

# averages over the bins (all, p < 2 GeV, p >= 2 GeV)
B = out["bins"]
summ = {}
for lab, sel in (("all", lambda b: True), ("p<2", lambda b: int(b) < 2), ("p>=2", lambda b: int(b) >= 2)):
    ks = [k for k in B if sel(k)]
    s = {"floor_S_z": float(np.mean([B[k]["floor_S_z"] for k in ks])), "floor_S_r": float(np.mean([B[k]["floor_S_r"] for k in ks]))}
    for name in MODELS:
        for q in ("S_z", "S_r", "W1_z_mm", "W1_r_mm", "d_mean_z_mm", "d_mean_r_mm"):
            s[f"{name}:{q}"] = float(np.mean([B[k][name][q] for k in ks]))
    summ[lab] = s
out["summary"] = summ
# mean normalised histograms over the bins (p >= 2 GeV and p < 2 GeV) for residual plots / diagnostics
out["edges"] = {"z": ZB.tolist(), "r": RB.tolist()}
out["hist"] = {}
for lab, sel in (("p<2", lambda b: int(b) < 2), ("p>=2", lambda b: int(b) >= 2)):
    ks = [k for k in HIST if sel(k)]
    out["hist"][lab] = {name: {"z": np.mean([HIST[k][name][0] for k in ks], 0).tolist(), "r": np.mean([HIST[k][name][1] for k in ks], 0).tolist()}
                        for name in ["data"] + list(MODELS)}
json.dump(out, open(os.path.join(D, "profile_metrics.json"), "w"), indent=1)
print(particle, "summary", json.dumps(summ["all"]), flush=True)
