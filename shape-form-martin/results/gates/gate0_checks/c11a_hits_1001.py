"""Gate 0 check 6 (part): is 1001 hits per event a cap? Looks at the two events with exactly 1001 hits after cuts
(one in train, one in val) and at the multiplicity distribution after each cut. Reads only these events' rows."""
import json

import pyarrow.compute as pc
import pyarrow.parquet as pq

from g0common import *  # noqa: F401,F403

t = np.load(HERE / "events_table.npz")
sel = np.flatnonzero(t["n_hits"] >= 960)
evs = t["event_number"][sel]
print("events:", evs, "split", t["split"][sel], "n_hits", t["n_hits"][sel], "raw rows", t["raw_rows"][sel], "p", t["p"][sel])
pf = pq.ParquetFile(str(paths.parquet_path()))
tbl = pq.read_table(str(paths.parquet_path()), filters=[("event_number", "in", [int(e) for e in evs])])
df = tbl.to_pandas()
out = {}
for e, g in df.groupby("event_number"):
    r = np.hypot(g.x_local, g.y_local)
    tcut = g.earliest_contribution_time < 100
    geo = (g.z_local >= -200) & (g.z_local <= 1450) & (r < 700)
    k = g[tcut & geo]
    out[int(e)] = {"raw": len(g), "after_time": int(tcut.sum()), "after_geo_only": int(geo.sum()), "after_all": len(k),
                   "time_max_kept": float(k.earliest_contribution_time.max()), "time_quantiles_raw": np.quantile(g.earliest_contribution_time, [0, .5, .9, .99, 1]).tolist(),
                   "energy_min_kept": float(k.energy.min()), "energy_min_raw": float(g.energy.min()),
                   "unique_cells_raw": int(len(g[["x", "y", "z"]].drop_duplicates())),
                   "first_rows_time": g.earliest_contribution_time.head(5).round(3).tolist(),
                   "energy_sorted_in_file": bool(g.energy.is_monotonic_decreasing or g.energy.is_monotonic_increasing),
                   "time_sorted_in_file": bool(g.earliest_contribution_time.is_monotonic_increasing)}
print(json.dumps(out, indent=1))
# multiplicity tails over all events after cuts
n = t["n_hits"]
print("n_hits after cuts: top 12 values", np.sort(n)[-12:])
print("raw rows: top 12 values", np.sort(t["raw_rows"])[-12:])
(HERE / "c11a_hits_1001.json").write_text(json.dumps(out, indent=1) + "\n")
