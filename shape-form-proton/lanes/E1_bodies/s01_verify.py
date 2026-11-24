"""Step (i): do envlab.Envelope / envlab.BetaDrop(fixed=(0.5, 1.5)) reproduce the report's saved per-bin volumes?
Also: timing of the exact objective, agreement of the fast sort and of the histogram surrogate."""
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/E1_bodies")
import bodylab as bl
import envlab

C, geo, data = bl.load()
ENV = torch.load("/home/user/sofiya/shape-form/results/proton/envelope.pt", weights_only=False)
DROP = torch.load("/home/user/sofiya/shape-form/results/proton/drop.pt", weights_only=False)
env, drop = envlab.Envelope(), envlab.BetaDrop(fixed=(0.5, 1.5))
bins = [int(v) for v in sys.argv[1:]] or [1, 10, 18]
for b in bins:
    P, sh = bl.bin_P(C, data, "train", b)
    print(f"bin {b}: {P['n']} events, {len(P['z']) / 1e6:.2f} M points", flush=True)
    e, d = ENV[b], DROP[b]
    xe = (math.log(e["a"]), math.log(e["z_env"]), e["z0"])
    xd = (math.log(d["L"]), d["z0"])
    for name, body, x, saved in (("envelope", env, xe, e), ("drop", drop, xd, d)):
        torch.cuda.synchronize(); t0 = time.time()
        s = body.score(P, x)
        c = envlab.threshold(P, s)
        torch.cuda.synchronize(); t1 = time.time()
        cf = bl.fast_threshold(P, s)
        torch.cuda.synchronize(); t2 = time.time()
        V = math.exp(math.log(math.pi) + 2 * c + body.log_i2(x)) / 1e6
        print(f"  {name:8s} saved A {saved['A']:.4f} V {saved['V_L']:.3f} | envlab at saved shape: A {math.exp(c):.4f} V {V:.3f} "
              f"| fast A {math.exp(cf):.4f} | t envlab {t1 - t0:.3f}s fast {t2 - t1:.3f}s", flush=True)
    t0 = time.time()
    H = bl.EventHist(P)
    torch.cuda.synchronize(); t1 = time.time()
    for name, body, x in (("envelope", env, xe), ("drop", drop, xd)):
        t2 = time.time()
        ch = H.threshold(H.body_score(body, x))
        torch.cuda.synchronize(); t3 = time.time()
        print(f"  hist surrogate {name}: A {math.exp(ch):.4f}  (build {t1 - t0:.2f}s, eval {t3 - t2:.4f}s)", flush=True)
    if b == bins[0]:
        # a full re-fit of the drop with the report's recipe (grid + Nelder-Mead) on the exact points, fast objective
        from scipy.optimize import minimize
        obj = bl.objective_factory(drop, P=P)
        t0 = time.time()
        x0 = min(drop.x0_grid, key=obj)
        x = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 300}).x
        print(f"  drop re-fit: L {math.exp(x[0]):.1f} z0 {x[1]:.1f} V {math.exp(obj(x)) / 1e6:.3f} L   (saved L {d['L']:.1f} z0 {d['z0']:.1f} "
              f"V {d['V_L']:.3f})  [{time.time() - t0:.0f}s]", flush=True)
        obj = bl.objective_factory(env, P=P)
        t0 = time.time()
        x0 = min(env.x0_grid, key=obj)
        x = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 300}).x
        print(f"  envelope re-fit: a {math.exp(x[0]):.3f} zenv {math.exp(x[1]):.1f} z0 {x[2]:.1f} V {math.exp(obj(x)) / 1e6:.3f} L   "
              f"(saved a {e['a']:.3f} zenv {e['z_env']:.1f} z0 {e['z0']:.1f} V {e['V_L']:.3f})  [{time.time() - t0:.0f}s]", flush=True)
