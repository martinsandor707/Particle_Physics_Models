"""Gate 0 helper: which CUDA launch grid (GPU model) produced the reference numbers?

Validation only. For each candidate grid G the fold points of common.test_nll (seed 12345, K=64) are emulated on
the CPU (emul.py) and the validation mean of mlp_std.pt is compared with the value stored by the original run
(mlp_std.json: 18.118083953857422). Seed-to-seed sd of that mean is 1.6e-4, float32 resolution 2e-6.

    uv run python -B c03_find_reference_grid.py [quick|full]
Writes c03_find_reference_grid.json.
"""
import json
import sys
import time

from g0common import *  # noqa: F401,F403
from emul import test_nll_emul

mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
sets, extras = load_trainval()
val = sets["val"]
mlp = legacy.load_mlp_std()
target = jload(PION / "mlp_std.json")["val"]
out = {"target_val_mlp_std": target, "local_grid": 144}

# 1. the emulation with the local grid reproduces common.test_nll on this machine
t = time.time()
loc = common.test_nll(mlp, val)
t_real = time.time() - t
t = time.time()
emu = test_nll_emul(common, mlp, val, 144)
t_emul = time.time() - t
stored = np.load(M0 / "mlp_val_nll.npy")
out["local"] = {"mean_common_test_nll": f64mean(loc), "mean_emulated_grid144": f64mean(emu),
                "max_abs_diff_emul_vs_common": float(np.abs(emu.astype(np.float64) - loc).max()),
                "max_abs_diff_common_vs_stored_m0": float(np.abs(loc.astype(np.float64) - stored).max()),
                "seconds_common": t_real, "seconds_emulated": t_emul}
print(json.dumps(out["local"], indent=1))

# 2. scan
known = {  # SM counts of common GPUs (threads/SM: 1536 Ampere/Ada/Hopper, 1024 Turing, 2048 Pascal/Volta/Kepler)
    6: [10, 16, 20, 24, 26, 28, 34, 36, 38, 46, 48, 56, 58, 60, 64, 66, 68, 70, 72, 76, 80, 82, 84, 100, 108, 112, 128, 132, 142, 144, 170],
    4: [14, 16, 22, 24, 30, 34, 36, 40, 46, 48, 68, 72],
    8: [10, 13, 15, 20, 28, 30, 56, 80, 132, 144, 148],
}
cands = []
for mult, sms in known.items():
    cands += [mult * s for s in sms]
if mode == "full":
    cands += [6 * s for s in range(1, 201)] + [4 * s for s in range(1, 101)] + [8 * s for s in range(1, 201)]
cands = list(dict.fromkeys(cands))
res = {}
best = None
for i, G in enumerate(cands):
    a = test_nll_emul(common, mlp, val, G)
    d = f32mean(a) - target
    res[G] = {"mean_f32": f32mean(a), "mean_f64": f64mean(a), "diff_to_target": d}
    flag = "  <-- MATCH" if abs(d) < 4e-6 else ""
    print(f"[{i + 1}/{len(cands)}] grid {G:5d}: val mean {f32mean(a):.6f}  diff {d:+.2e}{flag}", flush=True)
    if abs(d) < 4e-6:
        best = G
        if mode != "full":
            break
out["scan"] = res
out["match_grid"] = best
(HERE / "c03_find_reference_grid.json").write_text(json.dumps(out, indent=1) + "\n")
print("match:", best)
