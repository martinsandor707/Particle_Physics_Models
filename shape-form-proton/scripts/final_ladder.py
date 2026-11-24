"""A végső család (ProtonShape, laterális = gce) fokszám-létrája n = 1 … 6 (az n. fokú az (n-1). megoldásából indul),
választás a validációs halmazon az 1-SE szabállyal (a report select_degree protokollja), majd a kiválasztott modell tesztje.
Használat: python final_ladder.py <slab|stdacc> [lateral] [plane|grid]   (a végső modell: gce grid; NMAX=6 alapértelmezés)"""
import json, os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import shapelab, protocol
from proton_shape import ProtonShape

mode = sys.argv[1]
lateral = sys.argv[2] if len(sys.argv) > 2 else "gce"
lat_norm = sys.argv[3] if len(sys.argv) > 3 else "plane"
LAT = lateral + ("" if lat_norm == "plane" else "g")          # file tag: gce (plane-normalised) / gceg (grid-normalised)
OUT = "/home/user/tamas/shape-form-proton/results/final"
C, geo, data = shapelab.setup(batch=256, out_dir=OUT, slab=(mode == "slab"))
base = shapelab.baseline_nll(C, data)
c, s = shapelab.lnp_center_scale(data)
dev = C["DEVICE"]
A = shapelab.mean_acceptance(C, data["train"], front=True, back=True) if mode == "stdacc" else None
DEG = {1: dict(degree=1, degrees={"logit_tr": 2}), **{n: dict(degree=n) for n in range(2, 7)}}
NMAX = int(os.environ.get("NMAX", 6))
models, val, rows, prev = {}, {}, [], None
for n in range(1, NMAX + 1):
    t0 = time.time()
    torch.manual_seed(0)
    core = ProtonShape(C, lnp0=c, lnp_s=s, lateral=lateral, lat_norm=lat_norm, **DEG[n]).to(dev).init_from_data(C, data)
    m = shapelab.WithAcceptance(core, A).to(dev) if A is not None else core
    ckp = os.path.join(OUT, "proton", f"{mode}_{LAT}_n{n}.pt")
    if os.path.exists(ckp):                                            # rung already trained: load it
        m.load_state_dict(torch.load(ckp, map_location=dev, weights_only=False)["state"])
    else:
        if prev is not None:
            core.warm_start(prev)
        shapelab.fit_fast(C, data, m, steps=4000, log=None)
    m.eval()
    val[n] = C["test_nll"](m, data["val"])
    models[n], prev = m, core
    rows.append({"n": n, "params": m.n_params(), "val": float(val[n].mean()), "sec": round(time.time() - t0)})
    print(rows[-1], flush=True)
best, diff = protocol.one_se_choice(val)
ok = [n for n in val if n == best or diff[n][0] < diff[n][1]]
n_sel = min(ok)
print("n* =", best, "| paired Δ vs n*:", {n: (round(d[0], 5), round(d[1], 5)) for n, d in diff.items()}, "| chosen (1-SE):", n_sel)
tag = f"{mode}_{LAT}_chosen"
res, v, t = shapelab.evaluate(C, data, models[n_sel], tag, base, extra={"degree": n_sel, "ladder": rows,
                                                                        "one_se": {str(k): diff[k] for k in diff}})
for n in models:                                   # save every rung too (no test look needed for the choice)
    torch.save({"state": models[n].state_dict(), "val_nll": val[n], "n_params": models[n].n_params(), "lateral": lateral, "lat_norm": lat_norm, "deg": DEG[n]},
               os.path.join(OUT, "proton", f"{mode}_{LAT}_n{n}.pt"))
print(json.dumps({k: v for k, v in res.items() if k != "extra"}, indent=1))
