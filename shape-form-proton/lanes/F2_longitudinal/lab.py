"""lab.py - helpers of lane F2 (slab likelihood): reference fit, generic experiment runner, logging."""
import json, math, os, sys, time
sys.path.insert(0, "/home/user/tamas/shape-form-proton")
LANE = "/home/user/tamas/shape-form-proton/lanes/F2_longitudinal"
sys.path.insert(0, LANE)
import numpy as np, torch
import shapelab, families
import fam

RES = LANE + "/results"
STATES = RES + "/states"
LOG = RES + "/log.jsonl"
GROUPS = {"b0": [0], "b1": [1], "b2_4": [2, 3, 4], "b5_19": list(range(5, 20))}


def setup(slab=True, out_dir=None):
    C, geo, data = shapelab.setup(batch=256, out_dir=out_dir or RES, slab=slab)
    c, s = shapelab.lnp_center_scale(data)
    os.makedirs(STATES, exist_ok=True)
    return C, geo, data, c, s


def reference(C, data, c, s, tag="slab"):
    """families.GammaCoreTail degree 2 fitted with fit_fast (3000 steps, seed 0); cached state + per-event validation NLL."""
    path = f"{STATES}/ref_gct2_{tag}.pt"
    m = families.GammaCoreTail(C, degree=2, lnp0=c, lnp_s=s).to(C["DEVICE"])
    if os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        m.load_state_dict(ck["state"])
        return m, ck["val"]
    shapelab.fit_fast(C, data, m, steps=3000, log=None)
    m.eval()
    val = C["test_nll"](m, data["val"])
    torch.save({"state": m.state_dict(), "val": val}, path)
    return m, val


def per_group(d, bins):
    out = {}
    for k, bl in GROUPS.items():
        msk = np.isin(bins, bl)
        out[k] = [float(d[msk].mean()), float(d[msk].std(ddof=1) / math.sqrt(msk.sum()))]
    return out


def run(C, data, c, s, ref_val, name, kw, steps=3000, init=None, seed=0, cls="LongMix", lr=0.03, note=""):
    """Fit fam.<cls>(C, lnp0=c, lnp_s=s, **kw), evaluate on validation only, append a row to results/log.jsonl, save the state."""
    t0 = time.time()
    torch.manual_seed(seed)
    m = getattr(fam, cls)(C, lnp0=c, lnp_s=s, **kw).to(C["DEVICE"])
    if init is not None:
        for src in (init if isinstance(init, (list, tuple)) else [init]):
            ck = torch.load(src if os.path.isabs(src) else f"{STATES}/{src}.pt", weights_only=False)
            m.init_from(ck["names"], ck["coef"])
    hist = shapelab.fit_fast(C, data, m, steps=steps, seed=seed, log=None, lr=lr)
    m.eval()
    val = C["test_nll"](m, data["val"])
    bins = data["val"].ev["bin"].cpu().numpy()
    d, se = shapelab.paired(val, ref_val)
    row = {"name": name, "cls": cls, "kw": kw, "steps": steps, "init": init, "seed": seed, "n_params": m.n_params(),
           "val": float(val.mean()), "d_ref": d, "se": se, "groups": per_group(val - ref_val, bins),
           "time": round(time.time() - t0, 1), "note": note}
    with open(LOG, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    torch.save({"state": m.state_dict(), "names": m.names, "coef": (m.coef * m.mask).detach().cpu(), "val": val, "kw": kw, "cls": cls},
               f"{STATES}/{name}.pt")
    g = row["groups"]
    print(f"{name:28s} n={row['n_params']:3d} val {row['val']:.5f}  d_ref {d:+.5f} +- {se:.5f} | b0 {g['b0'][0]:+.4f} b1 {g['b1'][0]:+.4f} "
          f"b2-4 {g['b2_4'][0]:+.4f} b5-19 {g['b5_19'][0]:+.4f} | {row['time']:.0f}s", flush=True)
    return m, val, row


def ref_as_init(C, data, c, s):
    """Store the reference coefficients in the same format as run() so that LongMix can be warm-started from it."""
    m, val = reference(C, data, c, s)
    p = f"{STATES}/ref.pt"
    if not os.path.exists(p):
        torch.save({"names": m.names, "coef": (m.coef * m.mask).detach().cpu(), "val": val}, p)
    return m, val
