"""lane.py - shared helpers of lane F4_lowp (setup, reference fit, reporting). No side effects at import."""
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
import numpy as np
import torch

import shapelab
import families

LANE = "/home/user/tamas/shape-form-proton/lanes/F4_lowp"
OUT = os.path.join(LANE, "results")
FIG = os.path.join(LANE, "figs")
M_P = 0.938272


def setup(slab=True, out=OUT):
    C, geo, data = shapelab.setup(batch=256, out_dir=out, slab=slab)
    base = shapelab.baseline_nll(C, data)
    c, s = shapelab.lnp_center_scale(data)
    return C, geo, data, base, c, s


def init_gamma(C, data, m):
    """Same starting values as Formula.init_from (used for every model that has the GammaCoreTail parameters)."""
    h = data["train"].hits()
    f0 = C["Formula"]("gamma", "core_tail", degree=1).init_from(h.z.values, h.r.values, h.w.values)
    with torch.no_grad():
        for i, n in enumerate(f0.names):
            if n in m.names:
                m.coef[m.names.index(n), 0] = f0.coef[i, 0]
    return m


def ref_val(C, data, base, c, s, tag="ref_gct_n2", degree=2, steps=3000):
    """The reference (GammaCoreTail degree 2, slab) fitted once in this lane; cached per-event val/test NLL."""
    d = os.path.join(C["OUT_DIR"], "proton")
    path = os.path.join(d, f"{tag}.pt")
    if os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        return ck["val_nll"], ck["test_nll"]
    m = init_gamma(C, data, families.GammaCoreTail(C, degree=degree, lnp0=c, lnp_s=s).to(C["DEVICE"]))
    shapelab.fit_fast(C, data, m, steps=steps, log=None)
    res, val, test = shapelab.evaluate(C, data, m, tag, base)
    return val, test


def masks(data, split="val"):
    p = data[split].ev["p"].cpu().numpy()
    return {"all": np.ones(len(p), bool), "lo": p < 2.0, "hi": p >= 2.0}, p


def report(C, data, base, val, ref, name, n_params, extra="", notes=True):
    """One table row: overall / p<2 / p>=2 validation NLL and paired differences to the reference and to the slab MLP."""
    mk, p = masks(data)
    mlp = base["mlp"]["val"]
    row = {"name": name, "n_params": int(n_params)}
    for k, m in mk.items():
        row[f"val_{k}"] = float(val[m].mean())
        d, se = shapelab.paired(val[m], ref[m])
        row[f"d_ref_{k}"], row[f"se_ref_{k}"] = d, se
        d, se = shapelab.paired(val[m], mlp[m])
        row[f"d_mlp_{k}"], row[f"se_mlp_{k}"] = d, se
    line = (f"| {name} | {n_params} | {row['val_all']:.4f} | {row['val_lo']:.4f} | {row['val_hi']:.4f} | "
            f"{row['d_ref_all']:+.4f} ± {row['se_ref_all']:.4f} | {row['d_ref_lo']:+.4f} ± {row['se_ref_lo']:.4f} | "
            f"{row['d_ref_hi']:+.4f} ± {row['se_ref_hi']:.4f} | {row['d_mlp_all']:+.4f} | {extra} |")
    print(line, flush=True)
    if notes:
        with open(os.path.join(LANE, "NOTES.md"), "a") as fh:
            fh.write(line + "\n")
    with open(os.path.join(LANE, "rows.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def by_bin(data, val, ref, split="val"):
    b = data[split].ev["bin"].cpu().numpy()
    out = {}
    for k in range(20):
        m = b == k
        d, se = shapelab.paired(val[m], ref[m])
        out[k] = (float(val[m].mean()), d, se, int(m.sum()))
    return out


def fit_eval(C, data, base, ref, model, name, steps=3000, lr=0.03, extra="", seed=0, save=True, notes=True):
    t0 = time.time()
    shapelab.fit_fast(C, data, model, steps=steps, lr=lr, seed=seed, log=None)
    model.eval()
    if save:
        res, val, test = shapelab.evaluate(C, data, model, name, base)
    else:
        val = C["test_nll"](model, data["val"])
    row = report(C, data, base, val, ref, name, model.n_params(), extra=extra + f" ({time.time() - t0:.0f}s)", notes=notes)
    return row, val
