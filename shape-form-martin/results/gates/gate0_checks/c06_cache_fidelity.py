"""Gate 0 check 2: cache fidelity on a subset the worker did not use.

The worker compared (a) the rows of the first four row groups (incomplete events) and (b) twelve contiguous
event-number ranges against the in-memory lean build. Here:
  * events are selected by event_number % 6 == k (complete events scattered over all 47 row groups), k = 0..5,
    i.e. full coverage with a different partition;
  * the reference is common.Showers.from_dataframe on a pandas frame of those rows, with the geometry from
    common.detector_geometry on a pandas frame of the full x, y, z columns (the reference code itself);
  * train/val events are compared with the cache FILE on disk (pion_s1_trainval.pt), bit for bit incl. dtypes;
  * all events of the class (test events included) are compared with sfm.data.build_showers on the same rows.
    The test cache file is never opened.
Writes c06_cache_fidelity.json.
"""
import json
import resource
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from g0common import *  # noqa: F401,F403

K = 6
classes = [int(a) for a in sys.argv[1:]] or list(range(K))
t0 = time.time()
pf = pq.ParquetFile(str(paths.parquet_path()))
out = {"modulus": K, "classes": classes}

# ---- geometry with the reference function on the full file (pandas, three columns)
gdf = pd.DataFrame({c: pf.read(columns=[c]).column(0).to_numpy() for c in ("x", "y", "z")})
geo_ref = common.detector_geometry(gdf)
del gdf
meta = data.load_meta()
obj = torch.load(data.cache_files()["trainval"], map_location="cpu", weights_only=False)
out["geometry"] = {"reference_pandas_full_file": geo_ref, "meta": meta["geometry"], "cache_file": obj["geometry"],
                   "equal_meta": geo_ref == meta["geometry"], "equal_cache_file": geo_ref == obj["geometry"]}
print("geometry", out["geometry"])

cached = {s: common.Showers(obj[s]["ev"], obj[s]["off"], obj[s]["xyz"], obj[s]["w"], obj[s]["box"]) for s in ("train", "val")}
cached_ev = {s: obj[s]["event_number"].numpy() for s in ("train", "val")}
cached_x = {s: {k: obj[s][k].numpy() for k in ("theta", "phi")} for s in ("train", "val")}
out["cache_dtypes"] = {"xyz": str(cached["train"].xyz.dtype), "w": str(cached["train"].w.dtype), "off": str(cached["train"].off.dtype),
                       "box": str(cached["train"].box.dtype), **{f"ev.{k}": str(v.dtype) for k, v in cached["train"].ev.items()},
                       "event_number": str(obj["train"]["event_number"].dtype), "theta": str(obj["train"]["theta"].dtype)}
cols = list(dict.fromkeys(data.HIT_COLUMNS + data.EVENT_COLUMNS))
res, covered = [], {"train": 0, "val": 0}
for k in classes:
    parts = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=cols)
        parts.append(t.filter(pa.array(t.column("event_number").to_numpy() % K == k)))
    tbl = pa.concat_tables(parts)
    del parts
    df = tbl.to_pandas()
    ref = common.Showers.from_dataframe(df, geo_ref)                      # the reference
    rx = data._reference_extras(df)                                       # pandas, as in from_dataframe
    lean, lx = data.build_showers(data.table_getter(tbl), geo_ref)        # the lean builder on the same rows
    r = {"class": k, "rows": int(tbl.num_rows), "events": len(ref), "hits": int(ref.off[-1])}
    c = data.compare_showers(ref, lean)
    r["lean_builder_equals_reference"] = c["all_equal"]
    r["lean_extras_equal"] = all(np.array_equal(rx[q], lx[q]) for q in ("event_number", "theta", "phi"))
    ref_ev = rx["event_number"]
    for s in ("train", "val"):
        pos_ref = np.flatnonzero(np.isin(ref_ev, cached_ev[s]))           # events of this class that are in split s
        pos_cache = np.searchsorted(cached_ev[s], ref_ev[pos_ref])
        assert np.array_equal(cached_ev[s][pos_cache], ref_ev[pos_ref])
        a, b = ref.subset(pos_ref), cached[s].subset(pos_cache)
        cc = data.compare_showers(a, b)
        cc["theta"] = bool(np.array_equal(rx["theta"][pos_ref], cached_x[s]["theta"][pos_cache]))
        cc["phi"] = bool(np.array_equal(rx["phi"][pos_ref], cached_x[s]["phi"][pos_cache]))
        r[f"cache_file_{s}"] = {"events": int(len(pos_ref)), "hits": int(a.off[-1]), "all_equal": bool(all(cc.values())),
                                "failed": [q for q, v in cc.items() if not v]}
        covered[s] += len(pos_ref)
    r["events_not_in_trainval(test)"] = int(len(ref_ev) - sum(r[f"cache_file_{s}"]["events"] for s in ("train", "val")))
    res.append(r)
    print(r, flush=True)
    del df, tbl, ref, lean
out["chunks"] = res
out["covered"] = covered
out["covered_all_trainval"] = covered == {"train": len(cached["train"]), "val": len(cached["val"])} if classes == list(range(K)) else None
out["all_equal"] = all(r["lean_builder_equals_reference"] and r["lean_extras_equal"] and r["cache_file_train"]["all_equal"]
                       and r["cache_file_val"]["all_equal"] for r in res)
out["peak_rss_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
out["seconds"] = round(time.time() - t0, 1)
(HERE / "c06_cache_fidelity.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print({k: out[k] for k in ("covered", "covered_all_trainval", "all_equal", "peak_rss_GB", "seconds", "cache_dtypes")})
