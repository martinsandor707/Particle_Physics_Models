"""Re-verify the survey of the earlier (interrupted) attempt: rebuild every model from its logged constructor, load the
explore checkpoint, recompute the validation NLL (K = 64, fixed seed) and compare with the logged value.  Also prints fitted
lateral parameters at a few momenta.   usage: verify.py [degree]"""
import json
import math
import os
import sys

import numpy as np
import torch

import lab
import fam
import families
import shapelab

degree = int(sys.argv[1]) if len(sys.argv) > 1 else 2
C, geo, data, c, s = lab.setup(slab=True)
dev = C["DEVICE"]
ref, ref_val = lab.reference(C, data, c, s, 2)
v = C["test_nll"](ref, data["val"])
print(f"reference GammaCoreTail n=2 slab: stored val {ref_val.mean():.5f}  recomputed {v.mean():.5f}  n={ref.n_params()}")
rows = [json.loads(l) for l in open(lab.LOG)]
seen = set()
for row in rows:
    if not row["name"].endswith(f"_n{degree}") or row["name"] in seen or row["name"].startswith("hat_"):
        continue
    seen.add(row["name"])
    path = os.path.join(lab.EXP, row["name"] + ".pt")
    if not os.path.exists(path):
        print(row["name"], "checkpoint missing")
        continue
    m = eval(row["ctor"], {"fam": fam, "families": families, "C": C, "c": c, "s": s}).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state"])
    m.eval()
    val = C["test_nll"](m, data["val"])
    d, se = shapelab.paired(val, ref_val)
    txt = f"{row['name']:28s} n={m.n_params():3d} logged {row['val']:.5f} recomputed {val.mean():.5f}  d_ref {d:+.5f} +- {se:.5f} |"
    with torch.no_grad():
        for p in (1.0, 5.0, 15.0):
            th = m.theta(torch.tensor([math.log(p)], device=dev))
            Z0 = torch.tensor([[[200.0]]], device=dev)
            ws = [float(torch.exp(m.raw(th, f"log_w{i}", Z0))) for i in range(len(m.comps))]
            fr = [float(torch.exp(x).flatten()[0]) for x in m.log_fractions(th, Z0)]
            sh = []
            for i, kind in enumerate(m.comps):
                for sn, _, tr in fam.KINDS[kind][1]:
                    sh.append(f"{sn}{i}={float(tr(th[f'{sn}{i}'])):.2f}")
            txt += f" p={p:g}: w=" + "/".join(f"{w:.1f}" for w in ws) + " f=" + "/".join(f"{f:.2f}" for f in fr) + (" " + ",".join(sh) if sh else "") + " |"
    print(txt, flush=True)
