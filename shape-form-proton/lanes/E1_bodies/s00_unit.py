"""Unit test of fam.py: analytic log_i2 against numerical integration of the radius; print the report's saved per-bin fits."""
import math, sys
import numpy as np, torch
sys.path.insert(0, "/home/user/tamas/shape-form-proton"); sys.path.insert(0, "/home/user/tamas/shape-form-proton/lanes/E1_bodies")
import fam
rng = np.random.default_rng(0)
z = np.linspace(-400, 12000, 2_000_001)
bad = 0
for name, (body, desc) in fam.registry().items():
    worst = 0.0
    for trial in range(6):
        x = list(body.x0_grid[rng.integers(len(body.x0_grid))])
        x = [v + 0.05 * rng.standard_normal() for v in x]
        if not body.valid(x):
            continue
        r = body.radius(z, x, 1.0)
        num = np.trapezoid(r ** 2, z)
        ana = math.exp(body.log_i2(x))
        worst = max(worst, abs(num / ana - 1))
    flag = "" if worst < 2e-3 else "   <-- CHECK"
    bad += worst >= 2e-3
    print(f"{name:18s} npar/bin {len(body.names) + 1}  max rel. diff analytic vs numeric {worst:.2e}{flag}")
print("bad:", bad)
ENV = torch.load("/home/user/sofiya/shape-form/results/proton/envelope.pt", weights_only=False)
DROP = torch.load("/home/user/sofiya/shape-form/results/proton/drop.pt", weights_only=False)
print(type(ENV), list(ENV.keys())[:5] if isinstance(ENV, dict) else "")
print("ENV[0]:", ENV[0]); print("DROP[0]:", DROP[0])
print("bin |  env: a  z_env  z0  A  V | drop: L z0 A V")
for b in sorted(k for k in ENV if isinstance(k, int)):
    e, d = ENV[b], DROP[b]
    print(f"{b:2d} | {e['a']:.3f} {e['z_env']:.1f} {e['z0']:.1f} {e['A']:.2f} {e['V_L']:.1f} | {d['L']:.1f} {d['z0']:.1f} {d['A']:.1f} {d['V_L']:.1f}")
print("mean V env", np.mean([ENV[b]['V_L'] for b in range(20)]), "drop", np.mean([DROP[b]['V_L'] for b in range(20)]))
