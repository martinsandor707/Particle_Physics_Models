"""Gate 0 helper: confirm that launch grid 1056 (132 SMs x 8 blocks, e.g. an H100) is the reference machine's stream.

Validation only. With the emulated fold points of grid G, evaluate every model whose validation numbers the
original run stored (mlp_std.json, new3_std.json rungs 1-6 with paired differences, base_std.json rungs 1-6) through the
unmodified common.event_nll, and compare. Control: the local grid 144 (what common.test_nll gives on this machine).
Writes c04_confirm_reference_stream.json and the per-event validation arrays val_grid{G}_{model}.npy.
"""
import json
import sys

from g0common import *  # noqa: F401,F403
from emul import test_nll_emul

GRIDS = [int(a) for a in sys.argv[1:]] or [1056, 144]
sets, extras = load_trainval()
val, train = sets["val"], sets["train"]
models = {"mlp_std": legacy.load_mlp_std()}
for n in range(1, 7):
    models[f"new3_std_n{n}"] = new3_std(n, train)
    models[f"base_std_n{n}"] = base_std(n, train)
ref = {"mlp_std": jload(PION / "mlp_std.json")["val"]}
J = {"new3_std": jload(PION / "new3_std.json"), "base_std": jload(PION / "base_std.json")}
for lad, j in J.items():
    for n in range(1, 7):
        ref[f"{lad}_n{n}"] = j["rungs"][str(n)]["val"]

out = {}
for G in GRIDS:
    arr = {k: test_nll_emul(common, m, val, G) for k, m in models.items()}
    for k, a in arr.items():
        np.save(HERE / f"val_grid{G}_{k}.npy", a)
    r = {"means": {}, "ladders": {}}
    for k in models:
        r["means"][k] = {"emulated_f32": f32mean(arr[k]), "stored": ref[k], "diff": f32mean(arr[k]) - ref[k]}
    for lad, j in J.items():
        v = {n: arr[f"{lad}_n{n}"] for n in range(1, 7)}
        best = min(v, key=lambda k: v[k].mean())
        diff = {n: shapelab.paired(v[n], v[best]) for n in v}
        chosen = min(n for n in v if n == best or diff[n][0] < diff[n][1])
        rows = {}
        for n in range(1, 7):
            sm, ss = j["diff_vs_best"][str(n)]
            # paired difference against the STORED best rung (so the comparison does not depend on which rung wins here)
            dm, ds = shapelab.paired(v[n], v[j["n_star"]])
            rows[n] = {"diff_vs_stored_best_emulated": [dm, ds], "stored": [sm, ss], "abs_err_mean": abs(dm - sm),
                       "abs_err_se": abs(ds - ss)}
        r["ladders"][lad] = {"n_star": [best, j["n_star"]], "chosen": [chosen, j["chosen"]], "rungs": rows}
    out[G] = r
    print(f"===== grid {G}")
    for k, x in r["means"].items():
        print(f"  {k:14s} emulated {x['emulated_f32']:.6f}  stored {x['stored']:.6f}  diff {x['diff']:+.2e}")
    for lad, x in r["ladders"].items():
        print(f"  {lad}: n_star (here, stored) {x['n_star']}  chosen {x['chosen']}")
        for n, y in x["rungs"].items():
            print(f"     n={n}: paired vs stored best  {y['diff_vs_stored_best_emulated'][0]:+.9f} +- {y['diff_vs_stored_best_emulated'][1]:.9f}"
                  f"   stored {y['stored'][0]:+.9f} +- {y['stored'][1]:.9f}   |err| {y['abs_err_mean']:.1e} / {y['abs_err_se']:.1e}")
    m = max(abs(x["diff"]) for x in r["means"].values())
    e = max(y["abs_err_mean"] for x in r["ladders"].values() for y in x["rungs"].values())
    print(f"  max |mean - stored| over 13 models: {m:.2e};  max |paired diff - stored| over 12 rungs: {e:.2e}")
    r["max_abs_mean_diff"], r["max_abs_paired_err"] = m, e
(HERE / "c04_confirm_reference_stream.json").write_text(json.dumps(out, indent=1) + "\n")
