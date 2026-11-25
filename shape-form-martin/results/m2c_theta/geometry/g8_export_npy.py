"""Geometry audit, step 8 - per-event NLL arrays as plain .npy matrices with JSON sidecars (CPU, no data access).

    uv run python -B shape-form-martin/results/m2c_theta/geometry/g8_export_npy.py

For each .npz of g5*: ``<stem>_nll.npy`` = float64 matrix (n_maps, n_events); the sidecar ``.json`` lists the rows
(map names), the events file, machine, launch grid, K, seed, batch and split. Report-metric, bulk-restricted and
K = 1024 arrays stay in separate files.
"""
import json

import glib as L
from glib import np

SPEC = {"val_report": ("{}|nll", "report metric, K = 64"),
        "val_bulk_zb1221": ("{}|nll", "G-bulk on the stratum zb >= 1221 mm (bulk-restricted; never mixed with report-metric numbers)"),
        "val_k1024": ("{}|nll", "report-metric estimator at K = 1024 (guard G-K)"),
        "val_controls": ("{}|nll", "report metric, K = 64: controls and diagnostics (NOT the geometry arm)")}
for stem, (pat, what) in SPEC.items():
    z = np.load(L.ARR / f"{stem}.npz")
    rows = [k[:-4] for k in z.files if k.endswith("|nll") and not k.startswith("bulk|")]
    mat = np.stack([z[pat.format(r)] for r in rows]).astype(np.float64)
    np.save(L.ARR / f"{stem}_nll.npy", mat)
    meta = json.loads((L.ARR / f"{stem}.npz.json").read_text())
    meta.update({"array": f"{stem}_nll.npy", "shape": list(mat.shape), "dtype": "float64", "rows": rows, "quantity": "per-event NLL [nat]: " + what,
                 "event_numbers": f"{stem}.npz['event_number'] (validation cache order" + (", stratum subset)" if "bulk" in stem else ")"),
                 "launch_grid": meta.get("machine", {}).get("launch_grid"), "label": "AUDIT numbers on the laptop's fold points; not scores of record"})
    (L.ARR / f"{stem}_nll.npy.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    np.save(L.ARR / f"{stem}_event_number.npy", z["event_number"])
    print(stem, mat.shape, len(rows))
z = np.load(L.ARR / "val_controls.npz")
rows = [k[5:-4] for k in z.files if k.startswith("bulk|") and k.endswith("|nll")]
np.save(L.ARR / "val_controls_bulk_zb1221_nll.npy", np.stack([z[f"bulk|{r}|nll"] for r in rows]).astype(np.float64))
meta = json.loads((L.ARR / "val_controls.npz.json").read_text())
meta.update({"array": "val_controls_bulk_zb1221_nll.npy", "rows": rows, "quantity": "per-event G-bulk NLL on the stratum zb >= 1221 mm (bulk-restricted)",
             "event_index": "val_controls.npz['bulk_stratum_index']", "launch_grid": meta.get("machine", {}).get("launch_grid")})
(L.ARR / "val_controls_bulk_zb1221_nll.npy.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
L.sidecar(L.ARR / "g1_train_cells.npz", split="train", n_events=69783, rows=34326739,
          note="training rows (no cut) per (column, layer) cell [occ] and the first training-event index that hits each cell [first_event]; "
               "columns ux, uy in mm; written by g1_origins.py from the TRAINING rows of the Parquet file; no model, no fold points")
print("ok")
