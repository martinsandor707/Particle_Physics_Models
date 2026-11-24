"""R1 audit, step 0: extract the raw CSV rows (all hits, BEFORE any cut) of the test-split events of the s1 v3 sample.

Read-only on the data; writes one parquet file into the review directory.  Used by r01_raw_frame.py to check the frame
definition (origin, rotation, z_trans) on the raw columns, independently of the cached tensors."""
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import torch

OUT = "/home/user/tamas/shape-form-proton/review/R1_likelihood"
SRC = "/home/user/data/simulations/regression-proton/proton_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.csv"
G = torch.load("/home/user/tamas/shape-form-proton/cache/proton_geometry.pt", weights_only=False)
want = np.concatenate([G["sets"]["test"]["event_number"].numpy()])
print("events wanted:", len(want), flush=True)
cols = ["event_number", "x", "y", "z", "x_local", "y_local", "z_local", "x_trans", "y_trans", "z_trans", "energy",
        "earliest_contribution_time", "incoming_momentum", "incoming_theta", "incoming_phi", "particle_type"]
types = {c: pa.float64() for c in cols if c not in ("event_number", "particle_type")}
types["event_number"] = pa.int64()
types["particle_type"] = pa.string()
t0 = time.time()
reader = pacsv.open_csv(SRC, read_options=pacsv.ReadOptions(block_size=1 << 26),
                        convert_options=pacsv.ConvertOptions(include_columns=cols, column_types=types))
wset = pa.array(want, type=pa.int64())
parts, n = [], 0
for batch in reader:
    n += batch.num_rows
    m = pc.is_in(batch.column("event_number"), value_set=wset)
    parts.append(pa.Table.from_batches([batch]).filter(m))
    if len(parts) % 50 == 0:
        print(f"  {n / 1e6:.0f}M rows read, {time.time() - t0:.0f}s", flush=True)
tab = pa.concat_tables(parts)
pq.write_table(tab, os.path.join(OUT, "raw_test_rows.parquet"))
print(f"done: {n} rows read, {tab.num_rows} kept ({time.time() - t0:.0f}s)", flush=True)
