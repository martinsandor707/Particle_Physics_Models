"""Gate 0 check 1 (leakage, geometry): common.load_data estimates the detector geometry (layer pitch, cell size,
back face) from the hit positions of ALL events. Is the result the same when only training events are used?
Reads the x, y, z, event_number columns; writes c15_geometry_train_only.json."""
import json

import pandas as pd
import pyarrow.parquet as pq

from g0common import *  # noqa: F401,F403

pf = pq.ParquetFile(str(paths.parquet_path()))
col = lambda n: pf.read(columns=[n]).column(0).to_numpy()      # noqa: E731
tab = np.load(HERE / "events_table.npz")
evn = col("event_number")
out = {}
df = pd.DataFrame({c: col(c) for c in ("x", "y", "z")})
out["all_rows"] = common.detector_geometry(df)
for i, s in enumerate(("train", "val", "test")):
    keep = np.isin(evn, tab["event_number"][tab["split"] == i])
    out[f"{s}_events_only"] = common.detector_geometry(df[keep])
out["train_only_equals_all"] = out["train_events_only"] == out["all_rows"]
out["all_equal"] = all(out[k] == out["all_rows"] for k in ("train_events_only", "val_events_only", "test_events_only"))
z = np.unique(df.z.round(2))
out["layers"] = {"n_distinct_z": int(len(z)), "z_first": float(z.min()), "z_last": float(z.max()), "pitch_min_max": [float(np.diff(z).min()), float(np.diff(z).max())],
                 "depth_first_centre_to_back_mm": float(out["all_rows"]["z_back"] - z.min())}
(HERE / "c15_geometry_train_only.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
