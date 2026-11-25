"""Geometry audit, step 1 - absolute shower origins of the TRAINING and VALIDATION events.

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g1_origins.py

Writes cache/pion_s1_geometry_trainval.pt (+ .json) and g1_origins.json. CPU only. The Parquet file is read through
a filter on the event numbers of cache/pion_s1_trainval.pt (sfm.geometry.origins_from_parquet); no test row is
returned, and every event number written is asserted to belong to that set. In the same pass the TRAINING rows
(only those) are histogrammed for step 2 (g2_volume.py): hits per (column, layer) cell, per event the list of
distinct cells is not kept - see g2 for what is stored.
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sfm import geometry as G  # noqa: E402
from sfm import paths  # noqa: E402

t0 = time.time()
obj = torch.load(paths.CACHE_DIR / "pion_s1_trainval.pt", map_location="cpu", weights_only=False)
evn = {s: obj[s]["event_number"].numpy() for s in ("train", "val")}
theta = {s: obj[s]["theta"].numpy() for s in ("train", "val")}
phi = {s: obj[s]["phi"].numpy() for s in ("train", "val")}
zb = {s: obj[s]["ev"]["zb"].numpy().astype(np.float64) for s in ("train", "val")}
del obj
allowed = np.concatenate([evn["train"], evn["val"]])
assert len(np.unique(allowed)) == len(allowed) == 69783 + 14950
n_tr = len(evn["train"])

# ---- training-row statistics collected in the same pass (rows of validation events are skipped in the hook) ----------
cols_npz = np.load(HERE / "explore" / "train_columns.npz")
ux, uy = cols_npz["ux"], cols_npz["uy"]
ckey = np.round(ux * 1000).astype(np.int64) * 10_000_000 + np.round(uy * 1000).astype(np.int64) + 5_000_000
corder = np.argsort(ckey)
ckey_s = ckey[corder]
NL = G.N_LAYERS
occ = np.zeros((len(ux), NL), dtype=np.int64)            # training rows per (column, layer)
first_event = np.full((len(ux), NL), n_tr, dtype=np.int64)   # smallest training-event index that hits the cell
vols = {"stepped": G.Volume("stepped"), "octagon": G.Volume("octagon"), "box": G.Volume("box")}
out_rows = {k: 0 for k in vols}
out_events = {k: np.zeros(n_tr, dtype=bool) for k in vols}
stat = {"rows": 0, "unknown_column": 0, "bad_layer": 0, "max_abs_x": 0.0, "max_abs_y": 0.0, "zmin": np.inf, "zmax": -np.inf}


def hook(index, c):
    tr = index < n_tr                                     # `allowed` = train then val: index < n_tr <=> training event
    if not tr.any():
        return
    x, y, z, ei = c["x"][tr], c["y"][tr], c["z"][tr], index[tr]
    stat["rows"] += len(x)
    stat["max_abs_x"] = max(stat["max_abs_x"], float(np.abs(x).max()))
    stat["max_abs_y"] = max(stat["max_abs_y"], float(np.abs(y).max()))
    stat["zmin"], stat["zmax"] = min(stat["zmin"], float(z.min())), max(stat["zmax"], float(z.max()))
    lay = np.rint((z - G.Z_LAYER0) / G.LAYER).astype(np.int64)
    bad = (lay < 0) | (lay >= NL) | (np.abs(z - (G.Z_LAYER0 + G.LAYER * lay)) > 1e-3)
    stat["bad_layer"] += int(bad.sum())
    key = np.round(x * 1000).astype(np.int64) * 10_000_000 + np.round(y * 1000).astype(np.int64) + 5_000_000
    j = np.searchsorted(ckey_s, key)
    known = (j < len(ckey_s)) & (ckey_s[np.minimum(j, len(ckey_s) - 1)] == key)
    stat["unknown_column"] += int((~known).sum())
    ok = known & ~bad
    ci = corder[j[ok]]
    np.add.at(occ, (ci, lay[ok]), 1)
    np.minimum.at(first_event, (ci, lay[ok]), ei[ok])
    X, Y = torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
    for k, v in vols.items():
        o = ~v.inside_xy(X, Y).numpy() | (z < G.Z_FRONT) | (z > G.Z_BACK)
        out_rows[k] += int(o.sum())
        if o.any():
            out_events[k][np.unique(ei[o])] = True


res = G.origins_from_parquet(allowed, row_hook=hook)
print(f"scan done in {time.time() - t0:.0f} s; rows read {int(res['rows'].sum())}")
assert np.isin(res["event_number"], allowed).all() and len(res["event_number"]) == len(allowed)
per = {}
for s, sl in (("train", slice(0, n_tr)), ("val", slice(n_tr, None))):
    per[s] = {k: v[sl] for k, v in res.items()}
    assert np.array_equal(per[s]["event_number"], evn[s])
    assert np.array_equal(per[s]["theta"], theta[s]) and np.array_equal(per[s]["phi"], phi[s])
    # origin z against the cache: zb = (z_back - z0) / cos(theta) (float32)
    z0 = G.Z_BACK - zb[s] * np.cos(theta[s])
    dz = np.abs(per[s]["origin"][:, 2] - z0).max()
    k = (per[s]["origin"][:, 2] - G.Z_LAYER0) / G.LAYER
    assert np.abs(k - np.rint(k)).max() < 1e-5 and dz < 1e-3, (dz, np.abs(k - np.rint(k)).max())
    print(s, "events", len(evn[s]), "rows", int(per[s]["rows"].sum()), "max spread", per[s]["spread"].max(), "max |z - z(zb)|", dz)
meta = G.save_origins(per, extra_meta={"source": paths.parquet_path().name,
                                       "written_by": "results/m2c_theta/geometry/g1_origins.py",
                                       "read_filter": "event_number in (train + val event numbers of cache/pion_s1_trainval.pt)"})
# re-load and re-assert: every stored event number is a train or val event
for s in ("train", "val"):
    o = G.load_origins(s, evn[s])
    assert np.array_equal(o, per[s]["origin"])
chk = torch.load(paths.CACHE_DIR / f"{G.GEOMETRY_STEM}.pt", map_location="cpu", weights_only=False)
stored = np.concatenate([chk[s]["event_number"].numpy() for s in ("train", "val")])
assert np.isin(stored, allowed).all() and set(chk) == {"train", "val", "definition", "units"}

# straight-line comparison (both splits; no hits involved)
summary = {"meta": meta, "seconds": time.time() - t0}
for s in ("train", "val"):
    O = per[s]["origin"]
    L = G.straight_line_origin(theta[s], phi[s], zb[s])
    d = np.hypot(O[:, 0] - L[:, 0], O[:, 1] - L[:, 1])
    k = np.rint((O[:, 2] - G.Z_LAYER0) / G.LAYER).astype(int)
    summary[s] = {"events": int(len(O)), "max_spread_mm": float(per[s]["spread"].max()),
                  "transverse_distance_to_straight_line_mm": {f"q{q}": float(np.quantile(d, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
                  "frac_gt_24.3": float((d > 24.3).mean()), "frac_gt_100": float((d > 100).mean()),
                  "frac_gt_500": float((d > 500).mean()),
                  "first_hit_layer_counts": {str(i): int((k == i).sum()) for i in range(6)} | {">=6": int((k >= 6).sum())}}
    print(s, json.dumps(summary[s]))
np.savez_compressed(HERE / "arrays" / "g1_train_cells.npz", ux=ux, uy=uy, occ=occ, first_event=first_event)
summary["training_rows"] = {**{k: (float(v) if isinstance(v, float) else int(v)) for k, v in stat.items()},
                            "rows_outside": out_rows, "events_with_row_outside": {k: int(v.sum()) for k, v in out_events.items()},
                            "cells_hit": int((occ > 0).sum()), "cells_total": int(occ.size)}
print(json.dumps(summary["training_rows"]))
(HERE / "g1_origins.json").write_text(json.dumps(summary, indent=1) + "\n")
print("done", time.time() - t0)
