"""P3 deciding test (review C1): are the validation outliers of an H200 parity run bin-edge flips?

    uv run python -B shape-form-martin/scripts/p3_compare.py <machine dir>/checks/val_parity/outliers_<bench>_<utc>.json

Run on the LAPTOP with the outlier material returned from the H200 (``val_parity.py`` writes it when an event differs
from the laptop's emulated array by more than 5e-5). For every outlier event:
  1. the same event is re-derived here with ``EmulGen(12345, grid)`` (the H200's stream, emulated) and the laptop's
     arithmetic: fold points, grid cells (iz, ir) and per-hit ln g_i (``val_parity.fold_points_for_events``);
  2. the cells are compared POINT BY POINT with the H200's record;
  3. the NLL difference is attributed: with the laptop's log rho of the event held fixed,
        dNLL_cells = - sum_i w_i [ln g_i(H200 cells) - ln g_i(laptop cells)]
     is the part caused by the points whose cell differs; it is compared with the recorded difference
     (H200 value - laptop emulated value).
Verdict per event: "flip: explained" (differing cells, |recorded - dNLL_cells| <= 3e-5, the arithmetic noise of two
GPUs on identical cells), "differing cells, NOT explained", or "no differing cell: NOT a bin-edge flip" (then the
strict per-event clause stays). The decision itself is Fable's (A6 revised, point 8); this script provides the test.
Validation only; the test split is never opened. Output: printed table and a JSON report in this machine's
checks/p3/ directory.
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2b_lib as lib  # noqa: E402

import numpy as np  # noqa: E402

from sfm import paths, provenance, registry  # noqa: E402

ARITH_TOL = 3e-5


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def attribute_cells(rec: dict, lap: dict, recorded_diff: float, K: int = 64, tol: float = ARITH_TOL) -> dict:
    """Point-by-point comparison of the grid cells of one event (``rec``: the H200 record, ``lap``: the laptop's
    re-derivation; each with "iz", "ir" (H, K) and "w" (H,); ``lap`` also "logrho" (N_Z, N_R)) and attribution of
    the NLL difference to the differing points, with the laptop's log rho held fixed."""
    iz_r, ir_r, iz_l, ir_l = (np.asarray(rec["iz"]), np.asarray(rec["ir"]), np.asarray(lap["iz"]), np.asarray(lap["ir"]))
    if iz_r.shape != iz_l.shape:
        return {"comparable": False, "reason": f"shapes differ {iz_r.shape} vs {iz_l.shape}"}
    diff_pts = (iz_r != iz_l) | (ir_r != ir_l)
    lr = np.asarray(lap["logrho"], dtype=np.float64)
    w = np.asarray(lap["w"], dtype=np.float64)
    g_rec = _logsumexp(lr[iz_r, ir_r], 1) - math.log(K)
    g_lap = _logsumexp(lr[iz_l, ir_l], 1) - math.log(K)
    d_cells = float(-np.sum(w * (g_rec - g_lap)))
    out = {"comparable": True, "n_points": int(diff_pts.size), "n_points_cell_differs": int(diff_pts.sum()),
           "n_hits_with_a_differing_point": int(diff_pts.any(1).sum()),
           "differing_points": [[int(h), int(k), int(iz_r[h, k]), int(ir_r[h, k]), int(iz_l[h, k]), int(ir_l[h, k])]
                                for h, k in zip(*np.nonzero(diff_pts))][:50],
           "recorded_diff": float(recorded_diff), "dnll_from_differing_cells": d_cells,
           "residual": float(recorded_diff - d_cells), "tol": tol}
    if "logg" in rec and "logg" in lap:
        same_hit = ~diff_pts.any(1)
        if same_hit.any():
            out["max_abs_logg_diff_on_identical_cells"] = float(np.max(np.abs(np.asarray(rec["logg"])[same_hit]
                                                                              - np.asarray(lap["logg"])[same_hit])))
    if out["n_points_cell_differs"] == 0:
        out["verdict"] = "no differing cell: NOT a bin-edge flip"
    elif abs(out["residual"]) <= tol:
        out["verdict"] = "flip: explained"
    else:
        out["verdict"] = "differing cells, NOT explained"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outliers_json")
    args = ap.parse_args(argv)
    lib.start("p3_compare")
    jf = Path(args.outliers_json)
    summ = json.loads(jf.read_text())
    npz = np.load(jf.with_suffix(".npz"))
    grid = int(summ["grid_of_fold_points"])
    bench = summ["benchmark"]
    events = [e["index"] for e in summ["events"]]

    def job():
        from val_parity import fold_points_for_events
        val = registry.load_validation()
        registry.check_validation(val)
        model = registry._default_bench_models()[bench].to(registry.C.DEVICE).eval()
        return fold_points_for_events(val.showers, events, grid, model=model)

    lap = lib.run_on_gpu("p3_compare", job, min_free_mb=2000)
    lib.free_cuda()
    rows = []
    for e in summ["events"]:
        i = e["index"]
        rec = {k: npz[f"ev{i}_{k}"] for k in ("iz", "ir", "w", "logg") if f"ev{i}_{k}" in npz.files}
        res = attribute_cells(rec, lap[i], e["diff_native_minus_target"])
        res.update(index=i, event_number=e["event_number"])
        rows.append(res)
        print(f"event {i} ({e['event_number']}): recorded diff {e['diff_native_minus_target']:+.2e}; "
              f"{res.get('n_points_cell_differs')} of {res.get('n_points')} points in another cell; "
              f"dNLL(cells) {res.get('dnll_from_differing_cells', float('nan')):+.2e}; "
              f"residual {res.get('residual', float('nan')):+.2e} -> {res.get('verdict', res.get('reason'))}")
    out = provenance.checks_dir(paths.OUT_RESULTS, lib.MACHINE.id, "p3") / f"p3_compare_{bench}_" \
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    paths.check_output_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(paths.scrub_str(json.dumps({"input": jf.name, "benchmark": bench, "grid": grid, "events": rows},
                                              indent=1)) + "\n")
    print(f"report: {paths.portable(out)}")
    all_explained = all(r.get("verdict") == "flip: explained" for r in rows)
    print(f"ALL OUTLIERS EXPLAINED AS BIN-EDGE FLIPS: {'yes' if all_explained else 'NO'} (decision: Fable)")
    return 0 if all_explained else 1


if __name__ == "__main__":
    sys.exit(main())
