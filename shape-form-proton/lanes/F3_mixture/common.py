"""Shared helpers of lane F3 (driver side): setup, reference fit, one experiment = fit + validation + log line."""
import json
import math
import os
import sys
import time

import numpy as np
import torch

LANE = "/home/user/tamas/shape-form-proton/lanes/F3_mixture"
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
sys.path.insert(0, LANE)
import shapelab  # noqa: E402
import families  # noqa: E402
import fam  # noqa: E402

RES = os.path.join(LANE, "results")
STATES = os.path.join(RES, "states")
LOG = os.path.join(RES, "ablation.jsonl")


def setup(slab=True, out=None):
    C, geo, data = shapelab.setup(batch=256, out_dir=out or RES, slab=slab)
    base = shapelab.baseline_nll(C, data)
    c, s = shapelab.lnp_center_scale(data)
    return C, data, base, c, s


def reference(C, data, base, c, s, tag="slab"):
    """families.GammaCoreTail degree 2 fitted with fit_fast (the lane's reference); per-event validation NLL cached."""
    path = os.path.join(RES, f"ref_gct2_{tag}_val.npy")
    if os.path.exists(path):
        return np.load(path)
    m = families.GammaCoreTail(C, degree=2, lnp0=c, lnp_s=s).to(C["DEVICE"])
    h = data["train"].hits()
    f0 = C["Formula"]("gamma", "core_tail", degree=1).init_from(h.z.values, h.r.values, h.w.values)
    with torch.no_grad():
        for i, n in enumerate(f0.names):
            m.coef[m.names.index(n), 0] = f0.coef[i, 0]
    shapelab.fit_fast(C, data, m, steps=3000, log=None)
    res, val, test = shapelab.evaluate(C, data, m, f"ref_gct2_{tag}", base)
    np.save(path, val)
    print("reference", json.dumps(res), flush=True)
    return val


def build(C, cfg, c, s):
    cfg = dict(cfg)
    cfg.pop("warm", None)
    cfg.pop("steps", None)
    cfg.pop("lr", None)
    return fam.Mixture(C, lnp0=c, lnp_s=s, **cfg).to(C["DEVICE"])


def warm_from(m, state):
    """Copy coefficients of equally named parameters from a saved state (dict with 'names', 'coef', 'mask')."""
    with torch.no_grad():
        for i, n in enumerate(state["names"]):
            if n in m.names:
                j = m.names.index(n)
                k = min(m.coef.shape[1], state["coef"].shape[1])
                m.coef[j].zero_()
                m.coef[j, :k] = (state["coef"][i, :k] * state["mask"][i, :k]).to(m.coef.device)
        m.coef.mul_(m.mask)


def run_one(C, data, base, c, s, ref_val, name, cfg, steps=3000, seed=0, log=LOG, quiet=True):
    t0 = time.time()
    m = build(C, cfg, c, s)
    if cfg.get("warm"):
        st = torch.load(os.path.join(STATES, cfg["warm"] + ".pt"), weights_only=False)
        warm_from(m, st)
    steps = cfg.get("steps", steps)
    shapelab.fit_fast(C, data, m, steps=steps, seed=seed, lr=cfg.get("lr", 0.03), log=None if quiet else print)
    m.eval()
    val = C["test_nll"](m, data["val"])
    d_ref, se_ref = shapelab.paired(val, ref_val)
    d_mlp, se_mlp = shapelab.paired(val, base["mlp"]["val"])
    phys = m.physical()
    row = {"name": name, "n_params": m.n_params(), "val": float(val.mean()), "d_ref": d_ref, "se_ref": se_ref,
           "d_mlp": d_mlp, "se_mlp": se_mlp, "steps": steps, "seed": seed, "sec": round(time.time() - t0, 1),
           "cfg": {k: v for k, v in cfg.items()}, "phys": {k: [round(float(x), 4) for x in v] for k, v in phys.items()}}
    os.makedirs(STATES, exist_ok=True)
    torch.save({"names": m.names, "coef": m.coef.detach().cpu(), "mask": m.mask.cpu(), "state": m.state_dict(), "cfg": cfg,
                "val": val}, os.path.join(STATES, name + ".pt"))
    with open(log, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    pi = " ".join(f"pi{k}=" + "/".join(f"{x:.3f}" for x in phys[f"pi{k}"]) for k in range(m.K))
    print(f"{name:34s} n={row['n_params']:3d} val {row['val']:.4f}  d_ref {d_ref:+.4f}±{se_ref:.4f}  d_mlp {d_mlp:+.4f}±{se_mlp:.4f}"
          f"  {row['sec']:.0f}s  {pi}", flush=True)
    return m, val, row
