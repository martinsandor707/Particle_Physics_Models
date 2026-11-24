"""Run named experiments of exps.py: fit (fit_fast, slab likelihood) + validation NLL + paired differences -> ablation.jsonl."""
import argparse
import fnmatch

import common
import exps

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--verbose", type=int, default=0)
a = ap.parse_args()

C, data, base, c, s = common.setup(slab=True)
ref_val = common.reference(C, data, base, c, s)
print(f"reference val {ref_val.mean():.4f}   mlp(slab) val {base['mlp']['val'].mean():.4f}", flush=True)
names = [n for pat in a.names for n in exps.E if fnmatch.fnmatch(n, pat)]
for name in names:
    try:
        common.run_one(C, data, base, c, s, ref_val, name + (f"_s{a.seed}" if a.seed else ""), exps.E[name], steps=a.steps, seed=a.seed,
                       quiet=not a.verbose)
    except Exception as e:  # keep the batch alive
        print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
