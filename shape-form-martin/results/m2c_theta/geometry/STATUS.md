# Geometry acceptance arm — checkpoint status (25 Nov 2026, paused on request)

This is a hand-over note, not the deliverable. `GEOMETRY.md` does not exist yet.

**The user's two questions are both still OPEN.** Nothing has been measured on validation: no acceptance has been
computed, no NLL evaluated, no GPU section run. (a) "Does the gain persist under the bulk guard?" — open.
(b) "Does it persist with front + side-only acceptance?" — open. The first-order sizes in the audit (front 155,
all faces 933 ×10⁻⁴ nat) rest on a geometry that is now known to be wrong in three ways (below) and must be
recomputed before they are quoted again.

Data used so far: TRAINING-split rows of the Parquet file only (selected by the training event numbers of
`cache/pion_s1_trainval.pt`). The `event_number` column was read in full to build the row selection; every other
column was indexed by the training rows before use. Validation rows were only counted. The test split was not
opened and no test event was used.

## 1. Established (with numbers)

### 1.1 Rotation convention and origin identity (Task 1, training events)

- Rows: 34,326,739 training, 7,365,844 validation, 7,418,390 other (not used). 69,783 training events.
- `incoming_theta`, `incoming_phi` of the Parquet rows equal the cache extras exactly (max difference 0).
- **`global = origin + Rᵀ · local`, with `R = common.rotation(θ, φ)` (rows = shower axes in global coordinates),
  holds for every hit of every training event**: the per-event spread (max − min over all rows of the event, no
  cuts) of `(x, y, z) − Rᵀ (x_local, y_local, z_local)` is at most 2.5×10⁻¹² mm (x), 2.0×10⁻¹² (y), 1.8×10⁻¹² (z).
  With the transposed convention (`R` instead of `Rᵀ`) the median spread is 1273 / 1025 / 426 mm.
- Origin z equals `z − z_trans` to 4.5×10⁻¹³ mm and takes exactly 60 values, 3662.4 + 20.5·k mm.
- **The origin is the energy-weighted centroid of the lowest-z layer over ALL rows of the event, including hits
  later than 100 ns**: exact (≤ 2.3×10⁻¹² mm) in 100 % of training events. With the 100 ns cut applied first only
  39.4 % of the events reproduce the centroid (98.8 % the layer); an unweighted centroid matches in 28.7 %.
- First hit layer index k (training): k = 0: 67,905; 1: 1,041; 2: 231; 3: 124; 4: 65; 5: 44; ≥ 6: 373.

### 1.2 Straight-line reconstruction of `c_side.py` against the true origin (training)

`c_side.py` puts the origin at `z0 = z_back − zb cos θ`, transverse `z0 tan θ (cos φ, sin φ)`.
- z agrees (max 6×10⁻⁵ mm). Transverse distance to the true origin: median 23.1 mm, 90 % 94.5 mm, 99 % 464 mm,
  99.9 % 917 mm, max 2251 mm. 45.9 % of events differ by more than half a cell (24.3 mm), 20.6 % by more than a
  cell, **9.27 % by more than 100 mm, 0.89 % by more than 500 mm**. Mean offset (+0.6, −0.1) mm, rms 69 mm per axis;
  along φ the mean is +8.6 mm for k = 0.
- By class: k = 0 with a single first-layer hit (18,651 events): median 19.9 mm, 93.2 % inside ±30 mm in both
  axes (cell quantisation), but 3.45 % beyond 100 mm. k = 0 with ≥ 10 first-layer hits: 11.3 % beyond 100 mm.
  k ≥ 1 (1,878 events): 28.8 % beyond 100 mm. k ≥ 4 (482): median 130 mm, 52.9 % beyond 100 mm.
- Reading: the straight line is the particle's nominal path; the frame origin is a hit centroid that includes
  stray and late deposits in the first hit layer. The two are different objects.

### 1.3 The instrumented volume from training hits (Task 2, partial)

- **Layers:** 60, centres 3662.4 + 20.5·k mm (deviation ≤ 10⁻⁴ mm); front face 3652.15, back face 4882.15 mm.
- **Lattice — not a uniform 48.6 mm grid.** y: 104 values `101·m + {26.2, 74.8}` mm (steps alternate 48.6 /
  52.4), range ±2599.8. x: 212 values; tower pitch 48.27 mm inside a module and 57.079 mm across a module
  boundary (module pitch 201.889 mm = 4 towers; 101.0 mm = 2 towers in y), two x-lattices shifted by 1.574 mm in
  alternating module rows, 2-tower modules (pitch 46.07) at x ≈ 0 and at the edge in every second module row.
  The lattice is symmetric about x = +2.5 mm (`x_first + x_last = 5.0` in every row). x range −2596.035 … +2601.035.
- **8,752 distinct (x, y) columns**, each with ≥ 1,692 training hits (median 3,903).
- **Outer outline: a stepped octagon, not a square.** Full width (104 cells, |x| ≲ 2600) for the 11 innermost
  module rows per side (|y| < 1111 mm); beyond that every module row (101 mm) is 4 cells narrower (2 per side,
  100.944 mm): 100, 96, …, 44 cells, down to |x| ≤ 1085.3 at |y| = 2551.2 / 2599.8. Outermost cell centre per
  module row a (y = 26.2 + 101a and 74.8 + 101a, mirrored in y), `x_last` [mm]:
  a = 0…10: 2599.461 (a even), 2601.035 (a odd); a = 11…25: 2498.516, 2397.572, 2296.627, 2195.683, 2094.738,
  1993.794, 1892.849, 1791.905, 1690.961, 1590.016, 1489.072, 1388.127, 1287.183, 1186.238, 1085.294;
  `x_first = 5.0 − x_last`.
- **Beam-pipe hole, off-centre.** The 12 cell rows with |y| ≤ 276.8 mm (|y| < 303) have 92 cells instead of 104:
  cell centres jump from −430.762 to +233.873 mm (rows y = ±26.2, ±74.8; gap 664.635) and from −429.465 to
  +231.815 mm (rows ±127.2, ±175.8; gap 661.28). Rows ±228.2, ±276.8 show the 664.635 gap as well; their edge
  centres were not printed and should be read from `explore/train_columns.npz`. Free aperture with a 24.3 mm
  half-cell: about x ∈ [−406.5, +209.6], centred near x = −98 mm, ≈ 616 × 606 mm. 144 missing columns.
- **No depth dependence.** 525,101 of 8,752 × 60 = 525,120 cells are hit. The 19 never-hit cells are in layers
  58 (2) and 59 (17), scattered over the whole face, in columns with 2,400–4,200 hits whose two preceding layers
  have 2–14 hits: consistent with low occupancy, to be treated as existing cells (Poisson check not yet written).
- **Compared with the audit's box (54 × 48.6 = 2624.4 mm):** of the 104 × 104 lattice positions inside the box,
  2,064 (19.1 %) have no cell: 1,920 in the four cut corners (17.8 %) and 144 in the hole (1.3 %).
- **Coordinate edge:** no training row lies beyond max(|x|, |y|) = 2601.04 mm (0 of 34.3 M rows beyond 2624.4).

### 1.4 The 1.35 % of events that did not fit the box

- With the true coordinates **no training hit centre lies outside the 2624.4 mm box** (0 events, 0 energy). So the
  1.35 % of validation events with hits "outside" in `c_side.py` cannot be cells outside the box, a coordinate
  edge, or the stepped outline: they are produced by the straight-line origin (§1.2: 9.3 % of events are displaced
  by more than 100 mm, 0.9 % by more than 500 mm).
- The box is wrong in the opposite direction, which the hits-outside test cannot see: it counts the cut corners
  and the beam hole as instrumented (§1.3).
- **Not yet done:** re-running the audit's test on training events with the straight-line origin to reproduce
  the 1.35 % / 0.38 % and split the flagged events by origin displacement. Until then the attribution is by
  exclusion (strong), not by direct reproduction.

### 1.5 Beam hole and the "short events" of Gate 0 §4

- Nominal entry point (straight line at z = 3662.4) inside the hole aperture: 0.97 % of training events.
  68.5 % of them start late (k ≥ 1), against 2.0 % of the others.
- **69.7 % of the k ≥ 4 events (Gate 0's short events) have their nominal entry point inside the hole**; among
  events entering elsewhere only 0.21 % have k ≥ 4. Adding a 100 mm margin raises the share to 71.8 %.
- So particles at the lowest angles (θ_min = 0.052 rad → entry radius 192 mm) fly into the beam hole and reach the
  calorimeter through its inner wall. Gate 0 left this mechanism open; it should now be stated as the main cause.
  The remaining 30 % of the k ≥ 4 events are not explained yet.

## 2. Hypotheses, not established

- That the acceptance gain survives the corrected geometry at anything like the audit's size. Unknown.
- That module gaps (8.5 mm in x every 202 mm, 3.8 mm in y every 101 mm between the metric's 48.6 mm fold boxes)
  are irrelevant. Argument: a uniform acceptance factor cancels in the metric (the hit term pays `ln A` for what
  the normaliser gains). Only the argument exists; no measurement.
- That the hole rows a = 2 use the same lattice as a = 0 (see §1.3).
- That the true origins of validation events behave like training ones (validation origins not yet computed).
- Why 3.5 % of single-first-hit, k = 0 events have an origin more than 100 mm from the nominal path (stray or
  albedo hit in the first layer while the primary left no hit there?). Not investigated.

## 3. Task status

| Task | Status |
|---|---|
| 1 Absolute origin | **done for training** (identity, convention, centroid definition, comparison with the straight line); validation origins and the cache file not written |
| 2 Instrumented volume | **partial**: lattice, layers, outline, hole, depth independence, box comparison done; missing: saturation curve of distinct cells vs number of events, Poisson check of the 19 empty cells, occupancy near the edges (pile-up test), final model as code with one simpler alternative and the hits-outside fractions, direct reproduction of the 1.35 %, figures |
| 3 Per-event acceptance | **not started** (design only, §5) |
| 4 Gains on validation with guards | **not started** |
| 5 Code and tests | **not started**: `src/sfm/geometry.py`, `tests/test_geometry.py` and `cache/pion_s1_geometry_trainval.pt` do not exist |

## 4. Files

`results/m2c_theta/geometry/explore/` — exploration scripts, run from the repo root with
`uv run python -B <script>`; CPU only, training rows only. They were run from a session scratch directory and
exchange intermediate files through the hard-coded path `SP` at the top of each script (tmpfs, probably gone):
set `SP` to a writable directory and run them in order.

| File | Produces |
|---|---|
| `e1.py` | row selection, origin identity, wrong-convention control, `z − z_trans`, straight-line comparison; writes `e1_train_origin.npz` (origins, straight-line origins, θ, φ, zb per training event) |
| `e2.py` | layer and lattice values, centroid definition of the origin (all rows / t < 100 ns / unweighted), global extent; writes `e2_first.npz` |
| `e3.py` | distinct columns per row, x-steps per row, 3-D cell occupancy; writes `e3_cols.npz`, `e3_occ.npy` |
| `e4.py` | x positions of representative rows, the 19 never-hit cells |
| `e5.py` | origin displacement by class, nominal entry point against the hole, late starts |
| `train_columns.npz` | the 8,752 training cell columns: `ux`, `uy` [mm], `c2` = training hits per column (copy of `e3_cols.npz`) |

No file outside `results/m2c_theta/geometry/` was created or changed; no git state changed.

## 5. Next steps, in order

1. `src/sfm/geometry.py`:
   - `origins_from_parquet(event_numbers)`: row-group-wise read, origin = `(x, y, z) − Rᵀ local` per row,
     per-event min/max, error if the spread exceeds 10⁻⁶ mm. Call it for training and validation only; store the
     origins in `cache/pion_s1_geometry_trainval.pt` in cache order.
   - Volume as a row table (per cell row: y interval with boundaries at the midpoints between rows, outer edge
     centre ± 24.3 mm; x intervals `[x_first − 24.3, x_last + 24.3]` minus the hole). Embed the stepped outline
     as constants (§1.3) and check them against `train_columns.npz`. Alternatives: the audit's box; a smooth
     octagon with a rectangular hole; optionally the union of the 48.6 mm boxes (with module gaps).
   - `acceptance(...)` on the 128×32 grid with the sub-points of `slab_accept` and numeric azimuth nodes, three
     variants: front (plane half a layer in front of the first HIT layer, as `event_slab` does — equal to the
     detector front face only for k = 0), front + side (outline and hole), front + side + back (`Z ≤ 4882.15`).
     Check the numeric front against the analytic `legacy.C["event_slab"](front=True, back=False)`.
   - Evaluation helper on `metric.event_nll_multi` with precomputed `log_softmax(logP + ln max(A, floor))`, one
     generator draw per batch of 128 so that the fold points equal those of `common.test_nll`.
2. `tests/test_geometry.py`: rotation convention, origin identity on a synthetic Parquet file, acceptance against
   Monte Carlo with an independent point-in-volume test, the two limiting cases of the brief.
3. Finish Task 2 (list in §3) and write the figures (occupancy map with outline, hole, staircase).
4. Task 3: quadrature convergence, Monte Carlo on a sample of events, energy share of hit centres and K = 64 fold
   points in cells below the floor, floor chosen on a training subsample.
5. Task 4 on validation (GPU, inside `gpu_lock`, `metric.deterministic()`): report metric, bulk guard on
   `val.subset(zb ≥ 1221)` as `sfm.registry._guard_arrays` does, short events as their own row, mass monitors,
   splits by θ quintile, contained / side-leaking (recomputed with the true origin and outline), p < 2 / ≥ 2 GeV.
   Planned decomposition with `return_parts=True` on softmax-normalised maps: report Δ = Δ(Σ hit terms) +
   Δ ln M_e. The first term is the whole-grid log score, which is proper for a per-event model; the second is
   the change of the mass fraction in front of the flat `zb` cut, i.e. the part that belongs to the metric's
   normaliser. The back-face share is Δ(full) − Δ(front + side).
   Note for the formula: `load_component_formula` already contains the event-mean template `Ā` (front + back);
   report it as specified and also with the core alone times `A_e`.
6. `GEOMETRY.md` with the recommendation.
