"""R2_numbers: independent re-evaluation on the original sample (val/test) in both metrics."""
import json, math, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton/review/R2_numbers")
import numpy as np, torch
import r2lib, shapelab, protocol

t0 = time.time()
C, geo, data = r2lib.setup()
dev = C["DEVICE"]
A_bar = shapelab.mean_acceptance(C, data["train"], front=True, back=True)
M = r2lib.build_models(C, data, A_bar, rungs=True)
PER = {}
diag = {}
for name, m in M.items():
    for split in ("val", "test"):
        off = C["test_nll"](m, data[split])                       # the report metric, 00_common's own function
        parts = r2lib.eval_parts(C, m, data[split], nsub=8)
        PER[(name, "std", split)] = off
        PER[(name, "slab8", split)] = parts["slab_mine"]
        PER[(name, "slab2", split)] = parts["slab_sl2"]
        diag[f"{name}/{split}"] = {"max|std_mine - test_nll|": float(np.abs(parts["std"] - off).max()),
                                   "max|logsumexp logP|": parts["norm_err"],
                                   "max|sum w - 1|": float(np.abs(parts["Wsum"] - 1).max()),
                                   "finite": bool(np.isfinite(off).all() and np.isfinite(parts["slab_mine"]).all())}
    print(f"{name:26s} std val {PER[(name,'std','val')].mean():.5f} test {PER[(name,'std','test')].mean():.5f} | "
          f"slab8 val {PER[(name,'slab8','val')].mean():.5f} test {PER[(name,'slab8','test')].mean():.5f} | "
          f"slab2 val {PER[(name,'slab2','val')].mean():.5f} test {PER[(name,'slab2','test')].mean():.5f}  ({time.time()-t0:.0f}s)", flush=True)

np.savez_compressed(f"{r2lib.OUT}/per_event_s1.npz", **{"|".join(k): v for k, v in PER.items()},
                    bins_val=data["val"].ev["bin"].cpu().numpy(), bins_test=data["test"].ev["bin"].cpu().numpy())
json.dump(diag, open(f"{r2lib.OUT}/diag_s1.json", "w"), indent=1)
print("worst diag:", max(d["max|std_mine - test_nll|"] for d in diag.values()), max(d["max|logsumexp logP|"] for d in diag.values()),
      max(d["max|sum w - 1|"] for d in diag.values()), all(d["finite"] for d in diag.values()))
print(f"done {time.time()-t0:.0f}s")
