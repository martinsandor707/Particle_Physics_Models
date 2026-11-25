"""Geometry audit - parity of the base maps of g5_gains.py with the metric of record (validation).
Compares, for the MLP and the formula: (a) arrays/val_report.npz (score_variants), (b) common.test_nll inside
sfm.metric.deterministic(), (c) the same call outside it, (d) results/m0_parity/*_val_nll.npy (Gate 0)."""
import json

import glib as L
from glib import common, legacy, metric, np, torch

sets, extras, origin = L.load_cpu(("train", "val"))
rep = np.load(L.ARR / "val_report.npz")
out = {}
with L.gpulock.gpu_lock("geometry audit g5p (parity of base maps)"):
    val = sets["val"].to(L.DEV)
    models = {"mlp": legacy.load_mlp_std(), "formula": legacy.load_component_formula(sets["train"])}
    for name, m in models.items():
        with metric.deterministic():
            b = common.test_nll(m, val)
            b2 = common.test_nll(m, val)
        c = common.test_nll(m, val)
        a = rep[f"{name}|nll"].astype(np.float32)
        d = np.load(L.paths.M0_DIR / f"{name}_val_nll.npy")
        mx = lambda x, y: float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())      # noqa: E731
        out[name] = {"g5_vs_common_deterministic": {"bit_identical": bool(np.array_equal(a, b)), "max_abs": mx(a, b)},
                     "common_deterministic_twice_bit_identical": bool(np.array_equal(b, b2)),
                     "common_deterministic_vs_m0_stored": {"bit_identical": bool(np.array_equal(b, d)), "max_abs": mx(b, d)},
                     "common_plain_vs_m0_stored": {"bit_identical": bool(np.array_equal(c, d)), "max_abs": mx(c, d)},
                     "g5_vs_m0_stored": {"max_abs": mx(a, d), "mean_diff": float(a.mean(dtype=np.float64) - d.mean(dtype=np.float64))},
                     "mean_g5": float(a.mean(dtype=np.float64)), "mean_m0": float(d.mean(dtype=np.float64))}
        print(name, json.dumps(out[name]), flush=True)
(L.HERE / "g5p_parity.json").write_text(json.dumps(out, indent=1) + "\n")
