"""Non-parametric probe of the depth dependence: every lateral raw parameter gets a piecewise-linear (hat) function of z'
(13 knots, p-independent offsets).  Prints the fitted curves at p = 1, 5, 15 GeV.

usage: hatprobe.py <degree> <name> <comps comma separated> [steps]
"""
import json
import math
import os
import sys

import numpy as np
import torch

import lab
import fam

degree, name, comps = int(sys.argv[1]), sys.argv[2], tuple(sys.argv[3].split(","))
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
C, geo, data, c, s = lab.setup(slab=True)
dev = C["DEVICE"]
ref2, ref_val = lab.reference(C, data, c, s, 2)
refd, refd_val = lab.reference(C, data, c, s, degree)
m = fam.LatMix(C, degree=degree, comps=comps, depth="hat", lnp0=c, lnp_s=s).to(dev)
lab.warm_long(m, refd)
# start from the linear-depth fit of the same form if it exists
lin = os.path.join(lab.EXP, f"{name}_n{degree}.pt")
if os.path.exists(lin):
    st = torch.load(lin, map_location=dev, weights_only=False)["state"]
    ml = fam.LatMix(C, degree=degree, comps=comps, depth="lin", lnp0=c, lnp_s=s).to(dev)
    ml.load_state_dict(st)
    with torch.no_grad():
        for n in ml.names:
            if n.startswith("s_lin_"):
                q = n[len("s_lin_"):]
                sl = float(ml.coef[ml.names.index(n), 0])
                for j, k in enumerate(fam.HAT_KNOTS):
                    if k != 0.0:
                        m.coef[m.names.index(f"h{j}_{q}"), 0] = sl * k / 1000
            elif n in m.names:
                m.coef[m.names.index(n), :] = ml.coef[ml.names.index(n), :]
ctor = f'fam.LatMix(C, degree={degree}, comps={comps!r}, depth="hat", lnp0=c, lnp_s=s)'
lab.run(C, data, m, f"hat_{name}_n{degree}", ref_val, ctor=ctor, steps=steps, refs={f"gct_n{degree}": refd_val})
with torch.no_grad():
    out = {"knots": list(fam.HAT_KNOTS), "curves": {}}
    for p in (0.5, 1.0, 2.0, 5.0, 15.0):
        th = m.theta(torch.tensor([math.log(p)], device=dev))
        Zk = torch.tensor(fam.HAT_KNOTS, device=dev)[None, :, None]
        # evaluate raw parameters at the knots: use the interpolation matrix restricted to the knots (identity)
        cur = {}
        for q in m.depth_spec:
            v = th[q].flatten().item()
            vals = []
            for j, k in enumerate(fam.HAT_KNOTS):
                vals.append(v + (th[f"h{j}_{q}"].flatten().item() if k != 0.0 else 0.0))
            cur[q] = vals
        out["curves"][str(p)] = cur
        print(f"--- p = {p} GeV   (z0 {th['z0'].item():.1f}, tmax {math.exp(th['log_tmax'].item()):.1f}, alpha-1 {math.exp(th['log_am1'].item()):.2f})")
        print("  z' knot : " + " ".join(f"{k:7.0f}" for k in fam.HAT_KNOTS))
        n = len(comps)
        for i in range(n):
            print(f"  w{i} [mm] : " + " ".join(f"{math.exp(v):7.1f}" for v in cur[f'log_w{i}']) + f"   ({comps[i]})")
        if n > 1:
            lf = np.array([[0.0] * len(fam.HAT_KNOTS)] + [cur[f"lf{i}"] for i in range(1, n)])
            f = np.exp(lf - lf.max(0)) / np.exp(lf - lf.max(0)).sum(0)
            for i in range(n):
                print(f"  f{i}      : " + " ".join(f"{v:7.3f}" for v in f[i]))
    with open(os.path.join(lab.RES, f"hat_{name}_n{degree}.json"), "w") as fh:
        json.dump(out, fh)
