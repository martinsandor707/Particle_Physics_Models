"""Diagnosztika 1: a report proton-számainak reprodukálása + hol veszít a baseline (bin, régió) + szeletelés-diagnosztika."""
import json, math, os, sys
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import protonlab

OUT = sys.argv[1]
C = protonlab.load_common()
geo, data = protonlab.load_proton(C)
DEVICE, Formula, FieldModel, test_nll = C["DEVICE"], C["Formula"], C["FieldModel"], C["test_nll"]
Z_EDGES, R_EDGES, DZ, N_Z, N_R, Z_MIN = C["Z_EDGES"], C["R_EDGES"], C["DZ"], C["N_Z"], C["N_R"], C["Z_MIN"]
B = protonlab.BASELINE_DIR
res = {"geo": geo, "n": {k: len(v) for k, v in data.items()}}

# ---------- 1. modellek betöltése ----------
lnp_tr = data["train"].ev["lnp"].cpu().numpy()
c, s = float((lnp_tr.max() + lnp_tr.min()) / 2), float((lnp_tr.max() - lnp_tr.min()) / 2)
res["lnp_center_scale"] = [c, s]
models = {}
for n in range(1, 7):
    m = Formula("gamma", "core_tail", degree=n, lnp0=c, lnp_s=s).to(DEVICE)
    m.load_state_dict(torch.load(f"{B}/gflash_n{n}.pt", map_location=DEVICE, weights_only=False)["model"])
    models[f"gflash_n{n}"] = m.eval()
m = FieldModel(16, 3).to(DEVICE)
m.load_state_dict(torch.load(f"{B}/mlp_L3xH16.pt", map_location=DEVICE, weights_only=False)["model"])
models["mlp"] = m.eval()
ck = torch.load(f"{B}/mlp_formula.pt", map_location=DEVICE, weights_only=False)
m = Formula(ck["long"], ck["lat"], ck["degree"]).to(DEVICE); m.load_state_dict(ck["state"]); models["mlp_formula"] = m.eval()

nll = {k: test_nll(m, data["test"]) for k, m in models.items()}
res["test_nll"] = {k: float(v.mean()) for k, v in nll.items()}
exp = {k: torch.load(f"{B}/export_{k}.pt", weights_only=False) for k in ("gflash", "mlp", "mlp_formula", "cnn", "cnn_formula")}
res["export_test_nll"] = {k: float(np.mean(v["test_nll"])) for k, v in exp.items()}
res["max_abs_diff_vs_export"] = {"gflash_n6": float(np.abs(nll["gflash_n6"] - exp["gflash"]["test_nll"]).max()),
                                 "mlp": float(np.abs(nll["mlp"] - exp["mlp"]["test_nll"]).max()),
                                 "mlp_formula": float(np.abs(nll["mlp_formula"] - exp["mlp_formula"]["test_nll"]).max())}
print(json.dumps(res, indent=1)); sys.stdout.flush()

# ---------- 2. binenkénti különbség ----------
te = data["test"]
bins = te.ev["bin"].cpu().numpy()
d = nll["mlp"] - nll["gflash_n6"]
res["mlp_minus_gflash_by_bin"] = {int(b): [float(d[bins == b].mean()), float(d[bins == b].std(ddof=1) / np.sqrt((bins == b).sum())),
                                           float(nll["gflash_n6"][bins == b].mean()), float(nll["mlp"][bins == b].mean())] for b in range(20)}

# ---------- 3. KL-felbontás a térképeken (MLP || GFlash) ----------
with torch.no_grad():
    for p in (0.5, 1.5, 3.0, 5.0, 10.0, 15.0, 19.0):
        lp = torch.log(torch.tensor([p], device=DEVICE))
        Pm, Pg = models["mlp"](lp).exp()[0].double(), models["gflash_n6"](lp).exp()[0].double()
        kl = float((Pm * (torch.log(Pm.clamp_min(1e-300)) - torch.log(Pg.clamp_min(1e-300)))).sum())
        zm, zg = Pm.sum(1), Pg.sum(1)
        klz = float((zm * (torch.log(zm.clamp_min(1e-300)) - torch.log(zg.clamp_min(1e-300)))).sum())
        # hol van a KL: z' < 30, r szerint
        contrib = (Pm * (torch.log(Pm.clamp_min(1e-300)) - torch.log(Pg.clamp_min(1e-300))))
        zc = C["ZC"].cpu().numpy(); rc = C["RC"].cpu().numpy()
        front = float(contrib[torch.as_tensor(zc < 30, device=DEVICE)].sum())
        res.setdefault("kl_mlp_vs_gflash", {})[p] = {"total": kl, "longitudinal": klz, "lateral|z": kl - klz, "from z'<30mm": front,
                                                     "mass z'<0 mlp": float(Pm[torch.as_tensor(zc < 0, device=DEVICE)].sum()),
                                                     "mass z'<0 gflash": float(Pg[torch.as_tensor(zc < 0, device=DEVICE)].sum())}

# ---------- 4. szelet (slab) diagnosztika ----------
half = geo["layer"] / 2


def slab_accept(cos_t, sin_t, D, nsub=2):
    """Azimutális akceptancia A(B, NZ, NR): a (z', r) gyűrű mekkora hányada esik a detektor-szeletbe."""
    k = (torch.arange(nsub, device=DEVICE) + 0.5) / nsub
    zs = Z_EDGES[:-1, None] + DZ * k                                                    # (NZ, nsub)
    rs = torch.sqrt(R_EDGES[:-1, None] ** 2 + (R_EDGES[1:, None] ** 2 - R_EDGES[:-1, None] ** 2) * k)   # (NR, nsub)
    zc_ = zs[None, :, :, None, None] * cos_t[:, None, None, None, None]
    den = (rs[None, None, None, :, :] * sin_t[:, None, None, None, None]).clamp_min(1e-9)
    lo = ((zc_ - D[:, None, None, None, None]) / den).clamp(-1, 1)
    hi = ((zc_ + half) / den).clamp(-1, 1)
    return ((torch.acos(lo) - torch.acos(hi)) / math.pi).mean((2, 4))


@torch.no_grad()
def slab_stats(model, sh):
    out = {"m_slab": [], "m_std": [], "dmin": [], "dmax_over": [], "e_out_std": []}
    for b in sh.batches(128):
        cos_t = b.R[:, 2, 2]; sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
        D = b.zb * cos_t                                                               # globális mélység z0-tól a hátlapig
        P = model(b.lnp).exp()
        A = slab_accept(cos_t, sin_t, D)
        out["m_slab"].append((P * A).sum((1, 2)))
        Pz = P.sum(2); cdf = torch.cat([Pz.new_zeros(len(Pz), 1), torch.cumsum(Pz, 1)], 1)
        t = ((b.zb - Z_MIN) / DZ).clamp(0, N_Z); k = t.floor().long().clamp(max=N_Z - 1); ar = torch.arange(len(Pz), device=DEVICE)
        out["m_std"].append(cdf[ar, k] + Pz[ar, k] * (t - k))
        dd = b.xyz[:, 2] * cos_t[b.ev] - b.xyz[:, 0] * sin_t[b.ev]                       # z_trans a találatokra
        n = len(b.lnp)
        out["dmin"].append(torch.full((n,), 1e9, device=DEVICE).scatter_reduce(0, b.ev, dd, "amin"))
        out["dmax_over"].append(torch.full((n,), -1e9, device=DEVICE).scatter_reduce(0, b.ev, dd - D[b.ev], "amax"))
        out["e_out_std"].append(torch.zeros(n, device=DEVICE).index_add_(0, b.ev, b.w * (b.xyz[:, 2] > b.zb[b.ev]).float()))
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


for name in ("gflash_n6", "gflash_n1", "mlp", "mlp_formula"):
    st = slab_stats(models[name], te)
    dn = np.log(st["m_slab"]) - np.log(np.clip(st["m_std"], 1e-8, None))
    res.setdefault("slab", {})[name] = {"mean m_slab": float(st["m_slab"].mean()), "mean m_std": float(st["m_std"].mean()),
                                        "ΔNLL (slab-std) mean": float(dn.mean()), "SE": float(dn.std(ddof=1) / np.sqrt(len(dn))),
                                        "NLL with slab norm": float((nll[name] + dn).mean()),
                                        "by bin": {int(b): float(dn[bins == b].mean()) for b in range(20)}}
res["slab_data_check"] = {"min z_trans over hits (all events)": float(st["dmin"].min()), "max (z_trans - D)": float(st["dmax_over"].max()),
                          "mean energy fraction beyond z'>zb": float(st["e_out_std"].mean()),
                          "frac events with any hit z'>zb": float((st["e_out_std"] > 0).mean())}

# ---------- 5. azimutális szerkezet az adatban ----------
h = te.batch(torch.arange(len(te), device=DEVICE))
cos_t = h.R[:, 2, 2]; theta = torch.acos(cos_t)
x, y, z, w = h.xyz[:, 0], h.xyz[:, 1], h.xyz[:, 2], h.w
r = torch.sqrt(x * x + y * y)
res["azimuth"] = {"<x'>": float((w * x).sum() / w.sum()), "<y'>": float((w * y).sum() / w.sum()),
                  "<x'^2>^.5": float(torch.sqrt((w * x * x).sum() / w.sum())), "<y'^2>^.5": float(torch.sqrt((w * y * y).sum() / w.sum()))}
tb = theta[h.ev]
rows = []
for (t0, t1) in ((0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7)):
    for (z0, z1) in ((-200, 0), (0, 50), (50, 150), (150, 400), (400, 800), (800, 1450)):
        m = (tb >= t0) & (tb < t1) & (z >= z0) & (z < z1)
        ww = w[m]
        if ww.sum() > 0:
            rows.append({"theta": [t0, t1], "z": [z0, z1], "E frac": float(ww.sum() / w[(tb >= t0) & (tb < t1)].sum()),
                         "<x'>": float((ww * x[m]).sum() / ww.sum()), "<y'>": float((ww * y[m]).sum() / ww.sum()),
                         "rms x'": float(torch.sqrt((ww * x[m] ** 2).sum() / ww.sum())), "rms y'": float(torch.sqrt((ww * y[m] ** 2).sum() / ww.sum())),
                         "<cos2phi>": float((ww * (x[m] ** 2 - y[m] ** 2) / (r[m] ** 2).clamp_min(1e-6)).sum() / ww.sum())})
res["azimuth_by_theta_z"] = rows
res["theta_quantiles"] = np.quantile(theta.cpu().numpy(), [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist()

# ---------- 6. első réteg / origó csúcs ----------
zt = z * cos_t[h.ev] - x * torch.sqrt(1 - cos_t[h.ev] ** 2)
first = zt.abs() < 1.0
pbin = te.ev["bin"][h.ev]
res["first_layer"] = {"E frac in first hit layer (all)": float((w * first).sum() / w.sum()),
                      "E frac at r<1mm & first layer": float((w * (first & (r < 1.0))).sum() / w.sum()),
                      "by bin": {int(b): float((w * (first & (pbin == b))).sum() / w[pbin == b].sum()) for b in range(20)},
                      "rms r in first layer": float(torch.sqrt((w * first * r * r).sum() / (w * first).sum()))}
json.dump(res, open(os.path.join(OUT, "diag1.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in res if k not in ("geo", "n", "test_nll", "export_test_nll", "max_abs_diff_vs_export", "lnp_center_scale")}, indent=1))
