"""lab - helpers of lane F1_lateral (fit, validation-only scoring, log).  Import only; no side effects."""
import json
import os
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
LANE = "/home/user/tamas/shape-form-proton/lanes/F1_lateral"
sys.path.insert(0, LANE)

import numpy as np
import torch

import shapelab
import families

RES = os.path.join(LANE, "results")
EXP = os.path.join(RES, "explore")
LOG = os.path.join(RES, "log.jsonl")


def setup(slab=True, out=RES):
    C, geo, data = shapelab.setup(batch=256, out_dir=out, slab=slab)
    c, s = shapelab.lnp_center_scale(data)
    return C, geo, data, c, s


def gct_init(C, data, m):
    """Same starting values as the report (Formula.init_from) for families.GammaCoreTail."""
    h = data["train"].hits()
    f0 = C["Formula"]("gamma", "core_tail", degree=1).init_from(h.z.values, h.r.values, h.w.values)
    with torch.no_grad():
        for i, n in enumerate(f0.names):
            m.coef[m.names.index(n), 0] = f0.coef[i, 0]
    return m


def reference(C, data, c, s, degree=2, tag="slab"):
    """families.GammaCoreTail of the given degree, fitted with fit_fast (cached): returns (model, per-event val NLL)."""
    name = f"ref_gct_n{degree}_{tag}"
    path = os.path.join(C["OUT_DIR"], "proton", name + ".pt")
    m = families.GammaCoreTail(C, degree=degree, lnp0=c, lnp_s=s).to(C["DEVICE"])
    if os.path.exists(path):
        ck = torch.load(path, map_location=C["DEVICE"], weights_only=False)
        m.load_state_dict(ck["state"])
        return m.eval(), ck["val_nll"]
    gct_init(C, data, m)
    shapelab.fit_fast(C, data, m, steps=3000, log=None)
    res, val, test = shapelab.evaluate(C, data, m, name, shapelab.baseline_nll(C, data))
    print(f"reference {name}: val {res['val_nll']:.4f} test {res['test_nll']:.4f} params {res['n_params']}", flush=True)
    return m.eval(), val


def warm_long(model, ref):
    """Copy the longitudinal coefficients (z0, log_tmax, log_am1) of a fitted reference into `model` (as far as degrees allow)."""
    with torch.no_grad():
        for n in ("z0", "log_tmax", "log_am1"):
            i, j = model.names.index(n), ref.names.index(n)
            k = min(model.coef.shape[1], ref.coef.shape[1])
            model.coef[i, :k] = (ref.coef * ref.mask)[j, :k]
    return model


def run(C, data, model, name, ref_val, ctor="", steps=3000, seed=0, lr=0.03, note="", refs=None):
    """Fit, score on VALIDATION only (K = 64, fixed seed), paired difference to the reference, append to the log."""
    t = time.time()
    shapelab.fit_fast(C, data, model, steps=steps, seed=seed, lr=lr, log=None)
    model.eval()
    val = C["test_nll"](model, data["val"])
    d, se = shapelab.paired(val, ref_val)
    row = {"name": name, "likelihood": "slab" if C.get("SLAB") else "std", "n_params": int(model.n_params()),
           "val": float(val.mean()), "d_ref": d, "se": se, "steps": steps, "seed": seed, "secs": round(time.time() - t, 1),
           "ctor": ctor, "note": note}
    for k, v in (refs or {}).items():
        row[f"d_{k}"], row[f"se_{k}"] = shapelab.paired(val, v)
    os.makedirs(EXP, exist_ok=True)
    torch.save({"state": model.state_dict(), "val": val, "ctor": ctor, "row": row}, os.path.join(EXP, name + ".pt"))
    with open(LOG, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{name:42s} n={row['n_params']:3d}  val {row['val']:.4f}  d_ref {d:+.4f} +- {se:.4f}  ({row['secs']:.0f}s)", flush=True)
    return val


def load_explore(model, name):
    ck = torch.load(os.path.join(EXP, name + ".pt"), map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(ck["state"])
    return model.eval(), ck["val"]
