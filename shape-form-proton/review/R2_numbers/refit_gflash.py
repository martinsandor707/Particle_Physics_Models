"""R2_numbers: like-with-like optimiser check. GFlash (gamma x core_tail) ladder n=1..6 refitted with exactly the ProtonShape
budget (fit_fast steps=4000, warm start, seed 0, batch 256), in the slab metric or the standard metric x Abar.
usage: refit_gflash.py <slab|stdacc> [steps]"""
import json, os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np, torch
import shapelab
import r2lib

mode = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
OUTD = os.path.join(r2lib.OUT, "refit_gflash")
os.makedirs(OUTD, exist_ok=True)
C, geo, data = shapelab.setup(batch=256, out_dir=OUTD, slab=(mode == "slab"))
dev = C["DEVICE"]
c, s = shapelab.lnp_center_scale(data)
A = shapelab.mean_acceptance(C, data["train"], front=True, back=True) if mode == "stdacc" else None
h = data["train"].hits()
rows, val, prev, models = [], {}, None, {}
t0 = time.time()
for n in range(1, 7):
    torch.manual_seed(0)
    f = C["Formula"]("gamma", "core_tail", degree=n, lnp0=c, lnp_s=s).init_from(h.z.values, h.r.values, h.w.values).to(dev)
    if prev is not None:
        with torch.no_grad():
            f.coef.zero_(); f.coef[:, :prev.coef.shape[1]] = prev.coef; f.slope.copy_(prev.slope)
    m = shapelab.WithAcceptance(f, A).to(dev) if A is not None else f
    hist = shapelab.fit_fast(C, data, m, steps=steps, log=None)
    m.eval()
    val[n] = C["test_nll"](m, data["val"])
    te = C["test_nll"](m, data["test"])
    sl8 = {sp: r2lib.eval_parts(C, m, data[sp], nsub=8, want=("slab_mine",))["slab_mine"] for sp in ("val", "test")} if mode == "slab" else None
    row = {"n": n, "params": sum(p.numel() for p in f.parameters()), "val": float(val[n].mean()), "test": float(te.mean()),
           "best_step": int(min(hist, key=lambda x: x[1])[0]), "sec": round(time.time() - t0)}
    if sl8:
        row["val_slab8"], row["test_slab8"] = float(sl8["val"].mean()), float(sl8["test"].mean())
    rows.append(row)
    print(row, flush=True)
    prev = f
    models[n] = m
    torch.save({"state": m.state_dict(), "val_nll": val[n], "test_nll": te}, os.path.join(OUTD, f"{mode}_gflash_s{steps}_n{n}.pt"))
nstar = min(val, key=lambda k: val[k].mean())
d = {n: shapelab.paired(val[n], val[nstar]) if n != nstar else (0.0, 0.0) for n in val}
ok = [n for n in val if n == nstar or d[n][0] < d[n][1]]
out = {"mode": mode, "steps": steps, "rows": rows, "n*": nstar, "chosen": min(ok), "delta": d}
json.dump(out, open(os.path.join(OUTD, f"{mode}_gflash_s{steps}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
