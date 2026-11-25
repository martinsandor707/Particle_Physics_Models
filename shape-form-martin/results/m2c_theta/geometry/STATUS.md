# Geometry acceptance arm — audit status (25 Nov 2026, second session: audit complete)

The deliverable is `GEOMETRY.md` (same directory); it is complete, not partial. This file is the hand-over note.

Split discipline: question (a), the volume, the acceptance checks, the floor, the templates and the choice of the
acceptance form used TRAINING events only; question (b) used VALIDATION only. The Parquet file was read only
through a pushed-down filter on the event numbers of `cache/pion_s1_trainval.pt`
(`sfm.geometry.origins_from_parquet`); no test row was returned, the test cache was not opened, `allow_test` was
never used. (First session, `explore/`: the `event_number` column was read in full to build the row selection; every
other column was indexed by the training rows before use.) No registry row was written and `sfm.registry` was not imported. No training of any kind. No git
command that changes state was run.

## Steps (all done)

| Step | Script | Output |
|---|---|---|
| 1 origins (train + val) | `g1_origins.py` | `cache/pion_s1_geometry_trainval.pt` + `.json`, `g1_origins.json`, `arrays/g1_train_cells.npz` |
| library | `src/sfm/geometry.py`, `tests/test_geometry.py` | 19 tests |
| 2 volume (train) | `g2_volume.py` | `g2_volume.json`, `figures/volume_outline.png`, `figures/saturation.png` |
| 3 question (a) (train) | `g3_box135.py` | `g3_box135.json`, `arrays/g3_train_box.npz` |
| 4 acceptance checks (train subsample) | `g4_acceptance.py`, `g4b_montecarlo.py` | `g4_acceptance.json`, `g4b_montecarlo.json` |
| 4c templates + form choice (train) | `g4c_forms.py` | `arrays/abar_train.npz`, `g4c_forms.json` |
| 5 question (b) (validation) | `g5_gains.py`, `g5p_parity.py`, `g5k_gk.py`, `g5c_controls.py`, `g5r_region.py` | `arrays/val_report.npz`, `val_bulk_zb1221.npz`, `val_k1024.npz`, `val_controls.npz`, `val_region.npz`, `val_events.npz`, `abar_train_by_theta.npz` (each with `.json`), `g5*.json` |
| 6 tables, 7 figure | `g6_tables.py`, `g7_figures.py` | `g6_tables.md`, `g6_tables.json`, `figures/gain_by_theta_and_exit.png` |
| 8 export | `g8_export_npy.py` | `arrays/val_*_nll.npy` + `.json` (per-event NLL matrices, rows named in the sidecar) |
| report | — | `GEOMETRY.md` |

Scripts run from the repository root with `uv run python -B shape-form-martin/results/m2c_theta/geometry/<script>`,
in the order of the table; logs in `logs/`; GPU work only inside `sfm.gpulock.gpu_lock`. `glib.py` holds the fixed
settings (K 64, seed 12345, batch 128, floor 1e-3, quadrature 2 × 64).

## Answers (audit numbers; laptop fold points, RTX 4060, launch grid 144; stand-in models; 1e-4 nat)

- **(a)** Reproduced on training: 951 of 69,783 events (1.363 %; validation in Milestone 1: 1.351 %) have a hit
  centre outside the 2624.4 mm box when the frame origin is put on the straight line; 0 events with the true origin.
  Cause: the origin is the energy-weighted centroid of the first hit layer, not the straight-line point; every
  flagged event is displaced by more than 23.4 mm. Not the outline, not the hole, not a coordinate edge. The box is
  wrong the other way (19.2 % of its area has no cell).
- **(b)** Validation, ratio form, Δ against the map's own θ-blind base:
  report metric, front + side: MLP −993.3 ± 21.8, formula −989.9 ± 21.6 (n = 14,950); all faces −993.7 ± 22.0 /
  −990.0 ± 21.8; back-face share −0.4 ± 2.3 / −0.2 ± 2.3. G-bulk (stratum, n = 14,784): −947.7 ± 21.6 / −942.8 ± 21.4.
  Persistence margin guard − 0.5·main: −455.3 ± 10.7 / −452.1 ± 10.5 (pass, resolved). Short events (n = 166):
  −1758 ± 232 (front + side), −1339 ± 263 (all faces). Not inflated by the flat z_b cut; the cut cancels the
  back-face part (hit term −37.6 ± 5.8, normaliser +37.2 ± 5.3). Hits behind z_b (0.10 % of the energy) contribute
  −3.4 ± 0.3 of the −993.3: the whole amount the flat cut leaves without a counter-term.
- Front acceptance alone: null in the ratio form (−2.1 ± 1.3), a loss in the naive form (+42.6 ± 4.3).
- Recommendation: run the arm; front + side as primary row, all faces secondary; ratio form with the training
  template; true origin and stepped volume. Open decision for the user: whether the ratio form counts as
  "parameter-free" (the naive form also passes: −893.1 ± 22.1, margin −403.6 ± 10.8).
- Side finding for the θ track (needs a decision, nothing changed): a θ-binned training-mean front + side acceptance,
  inputs (p, θ), zero training, gives −634.2 ± 16.8; Θ1 as specified (naive front) is a loss.

Note: `cache/`, `*.npy`, `*.npz`, `*.log` are git-ignored. The origin file, the arrays, `explore/train_columns.npz` and the
logs live only in this working copy; they must travel with the transfer package.

## Not done / open

- No number of record: H200 scores for the frozen finalists, test split, test origins (need the Parquet table on
  the H200 after Gate 2c).
- Registry entry point for per-event maps; transfer-package manifest entries (origin file, `geometry.py`, tests,
  template). Listed in `GEOMETRY.md` §5; not written here.
- Not measured: acceptance with open module gaps; detector-front-face variant of the front plane; floor scan for
  models other than the MLP.
