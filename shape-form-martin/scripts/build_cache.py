"""Build the lean pion cache (cache/pion_s1_{trainval,test}.pt, cache/pion_s1_meta.json) from the Parquet file
and check it bit for bit against the reference pandas code of common.py.

    uv run python -B shape-form-martin/scripts/build_cache.py [--no-full-check]

Checks (written to results/m0_parity/cache_build.json):
  1. row-group subset: common.Showers.from_dataframe vs sfm.data.build_showers on the rows of the first four
     Parquet row groups (~4.2 M rows; the event of the last row dropped), same geometry -> bit-exact equality of
     off, xyz, w, box and every ev tensor (+ event_number/theta/phi and the geometry function on that subset);
  2. full cache: from_dataframe on complete events, in event-number chunks, against the corresponding events of the
     full lean build (equivalent to one full-frame run; see sfm.data.exactness_check_full);
  3. geometry, event count and split counts against the expected values.
Reads only the Parquet file (column by column); the CSV is never opened.
"""
import json
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from sfm import data, paths  # noqa: E402


def main(full_check=True):
    t0 = time.time()
    out = {"parquet": paths.parquet_path().name}
    sh, extras, parts, meta = data.build_cache()
    out["build_seconds"] = round(time.time() - t0, 1)
    geo = meta["geometry"]
    out["geometry"] = geo
    out["geometry_expected"] = data.EXPECTED_GEOMETRY        # rounded values as printed in summary/notebook
    out["geometry_matches_expected"] = all(abs(geo[k] - v) < 1e-9 * abs(v) for k, v in data.EXPECTED_GEOMETRY.items())
    out["events_total"] = len(sh)
    out["events_total_expected"] = 99689
    out["counts"] = meta["counts"]
    out["counts_match_expected"] = meta["counts"] == data.EXPECTED_COUNTS
    out["hits_total"] = meta["hits_total"]
    nan_or_inf = {k: int((~np.isfinite(v.numpy())).sum()) for k, v in sh.ev.items() if v.is_floating_point()}
    nan_or_inf["xyz"] = int((~np.isfinite(sh.xyz.numpy())).sum())
    nan_or_inf["w"] = int((~np.isfinite(sh.w.numpy())).sum())
    out["non_finite_values"] = nan_or_inf

    t1 = time.time()
    print("exactness check 1: first 4 row groups")
    rg = data.exactness_check_rowgroups(geo, n_row_groups=4)
    rg["seconds"] = round(time.time() - t1, 1)
    print({k: v for k, v in rg.items() if k != "df_dtypes"})
    out["exactness_rowgroups"] = rg

    if full_check:
        t2 = time.time()
        print("exactness check 2: full cache vs reference, by event-number chunks")
        full = data.exactness_check_full(sh, extras, geo, n_chunks=12)
        full["seconds"] = round(time.time() - t2, 1)
        print("full check all_equal:", full["all_equal"])
        out["exactness_full"] = full
    out["peak_rss_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    out["total_seconds"] = round(time.time() - t0, 1)
    dest = paths.M0_DIR / "cache_build.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")
    ok = out["counts_match_expected"] and rg["all_equal"] and (not full_check or out["exactness_full"]["all_equal"])
    print("STATUS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(full_check="--no-full-check" not in sys.argv))
