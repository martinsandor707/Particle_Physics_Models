"""Tasks 3 + 4 - ladders n = 1..6 of the report's GFlash formula (gamma x core_tail) fitted with shapelab.fit_fast (batch 256).

usage: run_fast_ladders.py --jobs std:none,std:front,std:both,slab:none [--degrees 1,2,3,4,5,6] [--steps 3000] [--warm 1] [--seed 0]
  job = <likelihood used for training>:<acceptance kind>   (kinds: none front both frontP bothP margF margB, see g2lib.acceptance)
Every degree starts from the previous one (new coefficients zero) like select_degree; degree 1 starts from Formula.init_from(train hits).
Each fitted model is evaluated with the common test_nll (K = 64, fixed seed) under BOTH likelihoods; shapelab.evaluate writes
results_<lik>/proton/<name>.json/.pt for the training likelihood; the other metric goes into the ladder JSON.
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/G2_reference")
import g2lib
import numpy as np
import torch

import shapelab

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--degrees", default="1,2,3,4,5,6")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--warm", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--suffix", default="")
a = ap.parse_args()
degrees = [int(x) for x in a.degrees.split(",")]

t0 = time.time()
C, geo, data, use = g2lib.setup_both(batch=256)
assert C["BATCH"] == 256
c, s = shapelab.lnp_center_scale(data)
h = data["train"].hits()
base = {}
for lik in ("std", "slab"):
    use(lik)
    base[lik] = shapelab.baseline_nll(C, data)
print(f"setup {time.time() - t0:.0f}s  centre/scale {c:.6f} {s:.6f}", flush=True)

for job in a.jobs.split(","):
    lik, kind = job.split(":")
    other = "slab" if lik == "std" else "std"
    use(lik)
    acc_kw = g2lib.acceptance(C, data, kind)
    tag = f"fast_{lik}_gflash" + ("" if kind == "none" else f"_acc{kind}") + a.suffix
    print(f"=== {tag}: trained with the {lik} likelihood, acceptance {kind}, fit_fast steps {a.steps}, batch 256, "
          f"{'warm start from n-1' if a.warm else 'cold start'}, seed {a.seed} ===", flush=True)
    rows, val, prev = [], {}, None
    for n in degrees:
        use(lik)
        m = g2lib.build(C, kind, n, c, s, acc_kw).init_from_hits(h)
        if prev is not None and a.warm:
            m.warm_start(prev)
        t1 = time.time()
        hist = shapelab.fit_fast(C, data, m, steps=a.steps, seed=a.seed, log=None)
        fit_s = time.time() - t1
        res, v, t = shapelab.evaluate(C, data, m, f"{tag}_n{n}", base[lik],
                                      extra={"protocol": f"fit_fast(steps={a.steps}, lr=0.03->0.002 cosine), batch 256, "
                                                         + ("warm start from n-1" if a.warm else "cold start") + f", seed {a.seed}",
                                             "acceptance": kind, "ctor": g2lib.ctor(kind, n), "fit_seconds": fit_s,
                                             "best_step": int(min(hist, key=lambda x: x[1])[0]), "val_K16_best": float(min(x[1] for x in hist))})
        val[n] = v
        use(other)
        vo, to = C["test_nll"](m, data["val"]), C["test_nll"](m, data["test"])
        use(lik)
        row = {"name": f"{tag}_n{n}", "n": n, "n_params": res["n_params"], "trained_with": lik, "acceptance": kind,
               f"val_{lik}": res["val_nll"], f"test_{lik}": res["test_nll"], f"val_{other}": float(vo.mean()), f"test_{other}": float(to.mean()),
               "val_vs_report_gflash_n6": res["val_vs_gflash_n6"], "test_vs_report_gflash_n6": res["test_vs_gflash_n6"],
               "val_vs_report_mlp": res["val_vs_mlp"], "test_vs_report_mlp": res["test_vs_mlp"],
               "val_vs_report_same_n": shapelab.paired(v, base[lik][f"gflash_n{n}"]["val"]),
               "test_vs_report_same_n": shapelab.paired(t, base[lik][f"gflash_n{n}"]["test"]),
               "fit_seconds": fit_s, "ctor": g2lib.ctor(kind, n)}
        rows.append(row)
        prev = m
        print(f"n={n} params {row['n_params']:2d}  {lik}: val {row[f'val_{lik}']:.5f} test {row[f'test_{lik}']:.5f} | {other}: val {row[f'val_{other}']:.5f} "
              f"test {row[f'test_{other}']:.5f} | val vs report n={n}: {row['val_vs_report_same_n'][0]:+.5f} ± {row['val_vs_report_same_n'][1]:.5f} "
              f"| val vs report MLP: {row['val_vs_report_mlp'][0]:+.5f} ± {row['val_vs_report_mlp'][1]:.5f}  ({fit_s:.0f}s fit)", flush=True)
    n_star, n_sel, diff = g2lib.one_se(val)
    for r in rows:
        r["dval_vs_nstar"], r["chosen_1SE"] = diff[r["n"]], bool(r["n"] == n_sel)
    print(f"{tag}: lowest validation NLL n* = {n_star}; chosen by the 1-SE rule n = {n_sel}; "
          + "  ".join(f"n{k}: {diff[k][0]:+.5f}±{diff[k][1]:.5f}" for k in sorted(diff)), flush=True)
    json.dump({"tag": tag, "trained_with": lik, "acceptance": kind, "n_star": n_star, "chosen_degree": n_sel, "rows": rows,
               "steps": a.steps, "warm": a.warm, "seed": a.seed},
              open(f"{g2lib.OUT[lik]}/proton/{tag}_ladder.json", "w"), indent=1)
print(f"done {time.time() - t0:.0f}s", flush=True)
