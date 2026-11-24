"""M1 audit, step 0: sanity check and the shared per-hit parts (validation; training NLLs for D1).

    MPLBACKEND=Agg uv run python -B shape-form-martin/scripts/audit_parts.py

* Both benchmarks on validation with ``sfm.metric.test_nll_multi`` (K=64, seed 12345, batch 128, identical fold
  points, inside ``deterministic()``), ``return_parts=True``; compared with results/m0_parity/{mlp,formula}_val_nll.npy
  (pass: |mean diff| <= 1e-6, max per-event |diff| <= 1e-4). Writes parts_val_k64.npz (git-ignored).
* Both benchmarks on the TRAINING set, same settings (needed for the two training events below 0.1 GeV in D1):
  train_nll_k64.npz.
Writes results/m1_audit/sanity.json. Exits with status 1 if the sanity check fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

import numpy as np  # noqa: E402


def main():
    run = L.Run("sanity")
    sets, extras, mlp, formula = L.load()
    val, train = sets["val"], sets["train"]
    ev_ref = np.load(L.M0 / "val_event_number.npy")
    same_events = bool(np.array_equal(ev_ref, extras["val"]["event_number"]))

    L.sync()
    import time
    t = time.time()
    parts = L.compute_parts([mlp, formula], val)
    L.sync()
    t_val = time.time() - t
    np.savez(L.PARTS_FILE, **parts)

    ref = {"mlp": np.load(L.M0 / "mlp_val_nll.npy"), "formula": np.load(L.M0 / "formula_val_nll.npy")}
    checks = {}
    ok = same_events
    for j, name in enumerate(("mlp", "formula")):
        new = parts["nll"][j].astype(np.float64)
        r = ref[name].astype(np.float64)
        d = new - r
        c = {"mean_new": float(new.mean()), "mean_m0": float(r.mean()), "mean_diff": float(d.mean()),
             "max_abs_event_diff": float(np.abs(d).max()), "n_events_diff_gt_1e-6": int((np.abs(d) > 1e-6).sum()),
             "pass_mean_1e-6": bool(abs(d.mean()) <= 1e-6), "pass_event_1e-4": bool(np.abs(d).max() <= 1e-4),
             "record_value": L.RECORD[f"{name}_val"]}
        ok &= c["pass_mean_1e-6"] and c["pass_event_1e-4"]
        checks[name] = c
    gap_new = L.paired(parts["nll"][1], parts["nll"][0])
    gap_ref = L.paired(ref["formula"], ref["mlp"])
    # exactness of the parts: nll = weight_sum * log_mass - sum_i hit_i
    ev = L.hit_event(val)
    recon = []
    for j in range(2):
        hs = np.bincount(ev, weights=parts["hit"][j].astype(np.float64), minlength=len(val))
        rec = parts["weight_sum"].astype(np.float64) * parts["log_mass"][j] - hs
        recon.append(float(np.abs(rec - parts["nll"][j]).max()))

    t = time.time()
    tr = L.compute_parts([mlp, formula], train)
    L.sync()
    t_train = time.time() - t
    np.savez(L.TRAIN_NLL_FILE, nll=tr["nll"], log_mass=tr["log_mass"], weight_sum=tr["weight_sum"],
             event_number=extras["train"]["event_number"])
    out = {"pass": bool(ok), "same_event_order_as_m0": same_events, "checks": checks,
           "gap_val_new": gap_new, "gap_val_m0": gap_ref, "gap_record": [L.RECORD["gap_val"], L.RECORD["gap_val_se"]],
           "parts_reconstruction_max_abs_err": recon,
           "train": {"n_events": len(train), "mlp_mean": float(tr["nll"][0].astype(np.float64).mean()),
                     "formula_mean": float(tr["nll"][1].astype(np.float64).mean()),
                     "gap": L.paired(tr["nll"][1], tr["nll"][0])},
           "settings": {"K": 64, "seed": 12345, "batch": 128, "deterministic_sum": True, "fold": "iid"},
           "seconds": {"val_2_models_parts": round(t_val, 2), "train_2_models": round(t_train, 2)},
           "files": [L.PARTS_FILE.name, L.TRAIN_NLL_FILE.name]}
    run.write(out)
    print({k: (v["mean_new"], v["mean_diff"], v["max_abs_event_diff"]) for k, v in checks.items()}, "gap", gap_new)
    if not ok:
        print("SANITY CHECK FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
