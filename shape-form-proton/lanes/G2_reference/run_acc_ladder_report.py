"""Task 3 with the REPORT'S protocol - standard likelihood, GFlash formula x event-averaged acceptance, n = 1..6.

The loop of select_degree (00_common, cell 18) copied verbatim except that the model is fam.GFlashAcc (Formula x fixed Abar):
init_from(train hits), warm start from n-1 (new coefficients zero), fit(epochs=150, lr=0.03, patience=15), batch 32, 1-SE rule.
(select_degree itself cannot be used because it constructs the plain Formula internally - copied here, as the guide asks.)

usage: run_acc_ladder_report.py --acc both|front [--degrees 1,2,3,4,5,6]
Checkpoints of `fit` (resumable): results_std/proton/acc<kind>_gflash_n<n>.pt ; evaluation: ref_std_gflash_acc<kind>_n<n>.json/.pt
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton")
sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/G2_reference")
import numpy as np
import torch

import shapelab
import fam
import g2lib

ap = argparse.ArgumentParser()
ap.add_argument("--acc", default="both")
ap.add_argument("--degrees", default="1,2,3,4,5,6")
a = ap.parse_args()
degrees = [int(x) for x in a.degrees.split(",")]

LANE = "/home/user/tamas/shape-form-proton/lanes/G2_reference"
OUT = LANE + "/results_std"
t0 = time.time()
C, geo, data = shapelab.setup(batch=32, out_dir=OUT, slab=False)
assert C["BATCH"] == 32 and not C["SLAB"]
base = shapelab.baseline_nll(C, data)
c, s = shapelab.lnp_center_scale(data)
dev = C["DEVICE"]
A = shapelab.mean_acceptance(C, data["train"], front=True, back=(a.acc == "both"))
h = data["train"].hits()
print(f"setup {time.time() - t0:.0f}s  lnp centre/scale {c:.6f} {s:.6f}  acceptance {a.acc}", flush=True)

models, val, rows, prev = {}, {}, [], None
for n in degrees:
    m = fam.GFlashAcc(C, degree=n, lnp0=c, lnp_s=s, A=A).init_from_hits(h).to(dev)
    if prev is not None:
        m.warm_start(prev)
    print(f"--- degree {n} ---", flush=True)
    t1 = time.time()
    hist = C["fit"](m, data, f"{OUT}/proton/acc{a.acc}_gflash_n{n}.pt", epochs=150, lr=0.03)
    res, v, t = shapelab.evaluate(C, data, m, f"ref_std_gflash_acc{a.acc}_n{n}", base,
                                  extra={"protocol": "report select_degree loop: fit(epochs=150, lr=0.03, patience=15), batch 32, warm start from n-1",
                                         "epochs_run": len(hist["val"]), "acceptance": a.acc, "fit_seconds": time.time() - t1,
                                         "ctor": f"fam.GFlashAcc(C, degree={n}, lnp0=c, lnp_s=s)"})
    val[n] = v
    rows.append(res)
    models[n] = prev = m
    print(f"n={n} params {res['n_params']}  val {res['val_nll']:.5f}  test {res['test_nll']:.5f}  epochs {len(hist['val'])}  "
          f"val vs report gflash_n6 {res['val_vs_gflash_n6'][0]:+.5f} ± {res['val_vs_gflash_n6'][1]:.5f}  ({time.time() - t1:.0f}s)", flush=True)

n_star, n_sel, diff = g2lib.one_se(val)
for r in rows:
    k = int(r["name"].rsplit("_n", 1)[1])
    r["dval_vs_nstar"], r["chosen_1SE"] = diff[k], bool(k == n_sel)
print(f"lowest validation NLL n* = {n_star}; chosen by the 1-SE rule n = {n_sel}; "
      + "  ".join(f"n{k}: {diff[k][0]:+.5f}±{diff[k][1]:.5f}" for k in sorted(diff)), flush=True)
json.dump({"acceptance": a.acc, "n_star": n_star, "chosen_degree": n_sel, "rows": rows},
          open(f"{OUT}/proton/ref_std_gflash_acc{a.acc}_ladder.json", "w"), indent=1)
print(f"done {time.time() - t0:.0f}s", flush=True)
