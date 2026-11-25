"""Geometry audit, step 5r - where the hit-side change of the report metric sits (VALIDATION; G-region style).

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g5r_region.py

For a map P.phi / S the report-metric change is (GEOMETRY.md section 4.2)
    Delta_e = sum_i t_i + ln(share of the mass in front of z_b kept by phi),   t_i = - w_i ln( ghat_i[P phi] / ghat_i[P] ).
The mass behind z_b is in neither term, so t_i of hits BEHIND z_b is the only place where the flat cut leaves a
change of the map unpaid (a raised density there would be a free gain; a lowered one an unrewarded cost). This
script sums t_i by the position of the hit centre:
    behind z_b            z' >= z_b of the event
    deep or border        z' < z_b and (z' >= 1150 mm or last ring)         - outside the G-bulk region
    bulk                  the rest
Same fold points as g5_gains.py (per-hit terms from metric.Parts). AUDIT numbers; G-region localises the hit term only.
"""
import json
import time

import glib as L
from glib import G, common, legacy, metric, np, torch

t0 = time.time()
sets, extras, origin = L.load_cpu(("train", "val"))
val = sets["val"]
N = len(val)
abar = {k: torch.from_numpy(v) for k, v in np.load(L.ARR / "abar_train.npz").items()}
rep = np.load(L.ARR / "val_report.npz")
V = {"FS": ("front", "side"), "FSB": ("front", "side", "back")}
MAPS = ["mlp*FS/abar", "mlp*FSB/abar", "formula*FS/abar", "formula*FSB/abar", "mlp*FS", "mlp*FSB"]
REG = ("behind_zb", "deep_or_border", "bulk")
acc = {m: {r: [] for r in REG} for m in MAPS}
share = {r: [] for r in REG}
nll_chk = {m: [] for m in MAPS + ["mlp", "formula"]}
with L.gpulock.gpu_lock("geometry audit g5r (hit-side term by region, validation)"):
    DEV = L.DEV
    A = G.event_acceptance(val.ev["R"], torch.as_tensor(origin["val"]), G.Volume(), V, nsub=L.NSUB, nphi=L.NPHI, device=DEV)
    mlp, formula = legacy.load_mlp_std(), legacy.load_component_formula(sets["train"])
    ab = {k: abar[k].to(DEV) for k in V}
    data = val.to(DEV)
    r_last = float(common.R_EDGES[common.N_R - 1])
    with torch.inference_mode(), metric.deterministic():
        gen = torch.Generator(device=DEV).manual_seed(L.SEED)
        for s in range(0, N, L.BATCH):
            idx = torch.arange(s, min(s + L.BATCH, N), device=DEV)
            b = data.batch(idx)
            i = idx.cpu()
            base = {"mlp": mlp(b.lnp), "formula": formula(b.lnp)}
            maps = dict(base)
            for k in V:
                Ae = A[k][i].to(DEV)
                for name in ("mlp", "formula"):
                    maps[f"{name}*{k}/abar"] = G.apply_acceptance(base[name], Ae, L.FLOOR, template=ab[k])
                maps[f"mlp*{k}"] = G.apply_acceptance(base["mlp"], Ae, L.FLOOR)
            names = list(maps)
            out = metric.event_nll_multi([maps[n] for n in names], b, data.box, L.K, gen, return_parts=True)
            parts = {n: o[2] for n, o in zip(names, out)}
            for n, o in zip(names, out):
                nll_chk[n].append(o[0].double().cpu())
            z, r = b.xyz[:, 2], torch.sqrt(b.xyz[:, 0] ** 2 + b.xyz[:, 1] ** 2)
            behind = z >= b.zb[b.ev]
            deep = ~behind & ((z >= 1150.0) | (r >= r_last))
            masks = {"behind_zb": behind, "deep_or_border": deep, "bulk": ~behind & ~deep}
            for rg, mk in masks.items():
                share[rg].append(torch.zeros(len(idx), device=DEV, dtype=torch.float64).index_add_(0, b.ev, (b.w * mk).double()).cpu())
            for m in MAPS:
                bname = m.split("*")[0]
                lnS = torch.as_tensor(rep[f"{m}|ln_factor_mass"][s:s + len(idx)], device=DEV)
                t = -(parts[m].hit.double() - parts[bname].hit.double()) - b.w.double() * lnS[b.ev]
                for rg, mk in masks.items():
                    acc[m][rg].append(torch.zeros(len(idx), device=DEV, dtype=torch.float64).index_add_(0, b.ev, t * mk).cpu())
res = {"split": "val", "n": N, "K": L.K, "seed": L.SEED, "regions_by": "hit centre", "energy_share": {}, "maps": {}}
for n in nll_chk:
    assert np.array_equal(torch.cat(nll_chk[n]).numpy(), rep[f"{n}|nll"]), n          # same fold points and maps as g5
for rg in REG:
    v = torch.cat(share[rg]).numpy()
    res["energy_share"][rg] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(N)), "events_with_any": int((v > 0).sum())}
save = {}
for m in MAPS:
    bname = m.split("*")[0]
    tot = rep[f"{m}|nll"] - rep[f"{bname}|nll"]
    lnacc = rep[f"{m}|log_mass"] + rep[f"{m}|ln_factor_mass"] - rep[f"{bname}|log_mass"]
    reg = {rg: torch.cat(acc[m][rg]).numpy() for rg in REG}
    closure = float(np.abs(sum(reg.values()) + lnacc - tot).max())
    res["maps"][m] = {**{rg: L.paired0(v) for rg, v in reg.items()}, "kept_mass_term": L.paired0(lnacc), "total": L.paired0(tot),
                      "closure_max_abs": closure}
    for rg, v in reg.items():
        save[f"{m}|{rg}"] = v
    print(m, {rg: L.fmt(L.paired0(v)) for rg, v in reg.items()}, "kept-mass", L.fmt(L.paired0(lnacc)), "total", L.fmt(L.paired0(tot)), f"closure {closure:.1e}", flush=True)
print(json.dumps(res["energy_share"]))
np.savez_compressed(L.ARR / "val_region.npz", event_number=extras["val"]["event_number"], **save)
L.sidecar(L.ARR / "val_region.npz", split="val", n_events=N, K=L.K, seed=L.SEED, batch=L.BATCH, machine=L.machine_tag(),
          note="per-event sums of the hit-side term t_i = -w_i ln(g_i[P phi]/g_i[P]) by region of the hit centre; report-metric fold points",
          label="AUDIT numbers; localises the hit term only")
res["seconds"] = time.time() - t0
(L.HERE / "g5r_region.json").write_text(json.dumps(res, indent=1) + "\n")
print("done", res["seconds"])
