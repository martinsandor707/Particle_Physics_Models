"""Recompute the guards of already-guarded Track A runs with the current code and compare them bit for bit with the
registered arrays of THIS machine. Overwrites nothing: the recomputed arrays stay in memory; the only file written is
a new report checks/guard_recompute/report_<UTC>.json in this machine's directory. Through the launcher:

    $L verify_guards $P -B shape-form-martin/scripts/verify_m2a_guards.py A7_s0 A2s_s0

Compared, per run: the K = 1024 and bulk-stratum guard arrays (registry._guard_arrays, as registry.evaluate computes
them), the per-event mass monitors (registry._mass; the npz of the guard row, file SHA-256 checked first) and the
G-region arrays (run_m2a.g_region_compute against the tagged g_region_<class>.npy). Exit code 0 iff every array is
byte-identical (dtype, shape and bytes). Bit-identity is expected only on the machine that registered the arrays.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_m2a as M  # noqa: E402
import run_m2a_lib as L  # noqa: E402

from sfm import data as sfm_data, paths, provenance, registry  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402

G_REGION_CLASSES = ("deep", "border_ring", "border_zbin", "low_abar", "flagged", "inner", "normalisation")


def _same(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    eq = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    d = (float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
         if a.shape == b.shape and a.size else None)
    return {"byte_identical": bool(eq), "max_abs_diff": d, "dtype": [str(a.dtype), str(b.dtype)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    L.configure(smoke=args.smoke)
    L.start("verify_m2a_guards")
    C = L.common
    sets, extras = sfm_data.load_sets(("val",), device="cpu")
    val_cpu = sets["val"]
    report = {"what": "guards recomputed with the current code vs the registered arrays (bit for bit)",
              "stage": L.stage(), "machine_id": L.MACHINE_ID, "batch": L.CTX.batch, "runs": {}}

    def job():
        dev = C.DEVICE
        va = val_cpu.to(dev)
        val = registry.ValidationSet(va, extras["val"]["event_number"])
        registry.check_validation(val)
        idx = np.flatnonzero(va.ev["zb"].cpu().numpy() >= registry.ZB_STRATUM_MM)
        abar = dfac.acceptance_template("stored", dev)
        for n in args.runs:
            row = L.registered_row(n, need_guards=True)
            if row is None:
                report["runs"][n] = {"error": "no registered guard row on the live stream"}
                continue
            model = L.rebuild(n, dev)
            res = {}
            g = registry._guard_arrays(model, va, idx)
            for kind in ("k1024", "bulk_zb1221"):
                res[kind] = _same(g[kind], registry.row_array(row, kind=kind).values)
            mp = paths.OUT_RESULTS / row["guards"]["arrays"]["mass_monitors"]["path"]
            if provenance.sha256_file(mp) != row["guards"]["arrays"]["mass_monitors"]["file_sha256"]:
                res["mass_monitors"] = {"byte_identical": False, "error": "registered npz differs from its row"}
            else:
                mass = {k.replace(":", "_"): v for k, v in registry._mass(model, va, abar).items()}
                with np.load(mp) as reg:
                    res["mass_monitors"] = {k: _same(mass[k], reg[k]) for k in sorted(set(mass) | set(reg.files))
                                            if k in mass and k in reg.files}
                    if set(mass) != set(reg.files):
                        res["mass_monitors"]["_keys_differ"] = {"byte_identical": False}
            gr = M.g_region_compute(model, va, val_cpu)["arrays"]
            sdir = L.scores_dir(n)
            res["g_region"] = {}
            for cls in G_REGION_CLASSES:
                f = sdir / f"g_region_{cls}.npy"
                res["g_region"][cls] = (_same(gr[cls], provenance.load_tagged(f).values) if f.exists()
                                        else {"byte_identical": False, "error": "no registered G-region array"})
            flat = [res["k1024"], res["bulk_zb1221"], *res["mass_monitors"].values(), *res["g_region"].values()]
            res["all_byte_identical"] = all(x.get("byte_identical") for x in flat)
            report["runs"][n] = res
            print(f"{n}: K=1024 {res['k1024']['byte_identical']}, bulk {res['bulk_zb1221']['byte_identical']}, "
                  f"mass monitors {all(x.get('byte_identical') for x in res['mass_monitors'].values())}, "
                  f"G-region {all(x.get('byte_identical') for x in res['g_region'].values())} -> "
                  f"{'BIT-IDENTICAL' if res['all_byte_identical'] else 'DIFFERENT'}", flush=True)
            del model
        del va, val

    L.on_gpu("verify_m2a_guards", job)
    ok = bool(report["runs"]) and all(r.get("all_byte_identical") for r in report["runs"].values())
    report["passed"] = ok
    out = L.MACHINE_DIR / "checks" / "guard_recompute" / \
        f"report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if out.exists():
        raise SystemExit(f"{paths.portable(out)} exists; nothing is overwritten")
    L.write_json(report, out)
    print(f"report: {paths.portable(out)} -> {'PASS (all bit-identical)' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
