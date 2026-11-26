# Track B (M2B) status - 26 Nov 2026

Validation only throughout; deltas in 1e-4 nat per event, paired, ± SE over the 14,950 validation events. Nothing in
this file is gated.

## 1. Where Track B stands

**Done on the H200 (machine of record `h200-g1056-97b197a7`; batch 1, PLAN.md section 17.10; rows in
`results/machines/h200-g1056-97b197a7/registry.jsonl`, stage `m2b_mlp`):**
- the eight laptop-trained networks re-scored (records, never retrained): `s1_seed11..15`, `s2_seed11`, `s2_seed12`,
  `b0_seed0`;
- fine-tunes `s2_seed13..15` and the controls `b0h_seed0..2` (P5) trained there;
- ensembles E2..E6 and E5 of the stage-1 networks; the subset arrays of the size curve (B1c).

| model | val K = 64 | Δ `mlp_std` |
|---|---|---|
| `s2_seed11..15` | 18.118137 ... 18.118737 | +0.54 ... +6.54 (best member `s2_seed14`, second `s2_seed13`) |
| `b0h_seed0..2` (B0) | 18.118184, 18.118187, 18.118114 | +1.01, +1.04, +0.31 |
| E5 (B1, primary) | 18.118000 | −0.83 ± 1.35 (not promoted) |

The laptop rows of 24 Nov (`results/registry.jsonl`, schema 1) are records and are never paired.

**Not done:** the rest of the pre-registered list of `results/m1_audit/AUDIT.md` section 4.2. The code to run it is
written (this file, sections 2-6) and NOT run on the H200; it waits for the independent review.

## 2. The remaining list, derived from AUDIT 4.2

`scripts/run_m2b_list.py build_list()` is the tracked spec; `--list` prints it with the status of every run on the
live stream. Each recipe is pinned by hash in the tracked `configs/m2b/list_recipes.json` (review note N5); the
driver refuses a recipe that differs. T_B = 6000 steps, cosine 1e-3 → 1e-4, batch 256, `k_val` 64, `eval_every` 250,
selection last, smooth 0, weight decay 0, dedicated fold generator, EMA 0.999 (AUDIT 4.2).

| run(s) | rung | definition (AUDIT 4.2) | seeds | steps | comparator | class |
|---|---|---|---|---|---|---|
| `B1d_best`, `B1d_second` | B1d | FieldNet(32, 4) from the best / second-best stage-2 member, KL-distilled data-free to E5 on 256 training-momentum quantiles, lr 1e-3 | 2 starts | 6000 KL steps, no data | `mlp_std` | **primary** |
| `B1f_s0` | B1f | `B1d_best` + 3000 steps 3e-4 → 3e-5, K = 16, T_B otherwise | 0 | 3000 | `B1d_best` | exploratory |
| `B2a_s0/1` | B2a | `field_mlp_std`, T_B, K = 16 iid | 0, 1 | 6000 | `mlp_std` (and B0) | comparator |
| `B2b_s0/1` | B2b | `field_mlp_std`, T_B, K = 64 iid | 0, 1 | 6000 | `B2a` same seed | **primary** |
| `B2c_s0/1` | B2c | `field_mlp_std`, T_B, K = 16 stratified | 0, 1 | 6000 | `B2a` same seed | **primary** |
| `<B2x>_ema` | B3 | EMA streams of B2a/b/c, registered as their own rows | — | 0 | own last iterate | exploratory |
| `B3s_s0/1` | B3s | fresh `field(32, 4)`, 3e-3 → 1e-4, K = 16, smooth 1e-4, EMA 0.999 (+ EMA row) | 0, 1 | 12000 | `mlp_std` | exploratory |
| `B4f_s0/1` | B4f | `field_acc(template="front", base="mlp_std", warm_steps=500)`, T_B K = 16 | 0, 1 | 6000 | `B2a` same seed | **primary** |
| `B4s_s0`, `B4x_s0` | B4s/B4x | as B4f with `template="stored"` / `"full_box"` | 0 | 6000 | `B2a_s0` | exploratory |
| `B5f_s0/1` | B5f | `field_mlp_std`, T_B K = 16, smooth 1e-4 | 0, 1 | 6000 | `B2a` same seed | exploratory |
| `B6a_s0/1` | B6a | fresh `field(32, 4)`, KL warm start to `mlp_std` (256 quantiles, 3000 steps, 1e-3), T_B K = 16 | 0, 1 | 6000 | `mlp_std` | comparator |
| `B6b_s0/1` | B6b | `field(32, 5)`, same route | 0, 1 | 6000 | `B6a` same seed | exploratory |
| `B6c_s0/1` | B6c | `field(64, 4)`, same route; 1-SE rule | 0, 1 | 6000 | `B6a` same seed | **primary** |
| `Bceil_k1_wd{0,1e-4}_f100`, `Bceil_k6_wd{0,1e-4}_f{25,50,100}` | B-ceil | `residual_table(base="mlp_std", knots="uniform", freeze_base=True)`, 3000 steps 3e-3 → 3e-4, K = 16, `eval_every` 100; last iterate and trajectory minimum | 0 | 3000 each | `mlp_std` | diagnostic |

Given (batch 1, recognised as done, never re-run): S2-11..15, B0 = `b0h_seed0..2`, B1 = `E5_s2`, B1c (subset arrays,
E6). Closed: B7 (PLAN section 14). Primaries: B1 (done), B1d, B2b, B2c, B4f, B6c. **31 runs, 159,000 training steps**
plus KL warm starts (B1d 2 × 6000, B6 6 × 3000, B4 4 × 500) and any convergence reruns.

Rules applied by `scripts/run_m2b_list_report.py` (written before any result of this list exists):
- promotion (PLAN 6, AUDIT 4.0), each part in its own column: (a) Δ `mlp_std` ≤ −5; (b) ≤ −3 SE; (c) better than B0
  by more than 1 SE (z of Δ against the per-event mean of exactly `b0h_seed0..2` < −1; with fewer than three controls
  the part is "B0 incomplete"); (d) replication by the second seed; (e) guards passed (G-K and G-bulk persistence
  margins from the registry's guard columns, G-region ≤ 50 % of the gain in deep / border / Ā ≤ 0.3 cells, G-mass
  reported). Promoted = all five.
- primary comparisons B1d vs `mlp_std`, B2b − B2a, B2c − B2a, B4f − B2a, B6c − B6a (same seed, two-seed mean);
  everything else labelled exploratory / comparator / diagnostic; nulls and negative results as measured.
- 1-SE rule for B6 per seed; B3 EMA − last; B-ceil last iterate and trajectory minimum; secondary columns
  p < 2 GeV / ≥ 2 GeV and contained / side-leaking (never selection criteria); guard numbers in their own table.
- convergence (AUDIT 4.0): a primary whose validation mean moved by ≥ 0.5 over its last 1000 steps is rerun once with
  doubled steps together with its comparator (`<name>_x2`; recipes pinned as well).
- divergence (PLAN section 20, item 2; 17.11): diverged iff the last validation mean is not finite, the registry finds
  a non-finite per-event NLL, the last evaluation window is frozen (≥ 50 % of batches skipped), or the last value is
  more than 1 nat above the best of its history / its comparator. A recovered non-finite value in the middle of a fit
  is a warning (counted in `done.json`). Exploratory: recorded, the job continues. Primary, comparator, diagnostic:
  "diverged - awaiting diagnosis", never "failed" without `runs/m2b_mlp/diagnosis/<name>.json` (orchestrator, after
  a Fable diagnosis, naming the SHA-256 of the run's `diverged.json`). Runs started from a diverged run are blocked.

## 3. Where the list had to be interpreted (most literal reading taken; flagged for the reviewer)

| # | Point | Reading taken |
|---|---|---|
| I1 | B1d "2 starts" | the two starts are the two replicates: part (d) of B1d compares `B1d_second` with `B1d_best` |
| I2 | "best member" / "second-best" | by the registered K = 64 validation mean of `s2_seed11..15` on the live stream (H200: `s2_seed14`, then `s2_seed13`); resolved and recorded at run time |
| I3 | "256 training-momentum quantiles" (B1d, B6) | quantiles of the training ln p at probabilities (i + ½)/256 (midpoints, so the grid stays inside the training range that `kl_warm_start` enforces) |
| I4 | B1d training | data-free: the factory's KL fit is the whole training; no validation trajectory, so the convergence rule is not applied to B1d |
| I5 | B1f "B1d + 3000 steps" | from `B1d_best`; one run (seed 0, 2 min) |
| I6 | B3 "EMA streams of B2a, B2b, B2c" | each EMA state registered as `<run>_ema` (exploratory) and compared with its own last iterate. B3s ("with EMA") also gets an EMA row |
| I7 | unspecified settings of B3s and B-ceil | taken from T_B (batch 256 given, `k_val` 64, selection last, weight decay as stated, dedicated folds, EMA 0.999; no EMA row for B-ceil) |
| I8 | B-ceil `train_fraction` | the first round(f N) events of a permutation of the training split seeded with the run seed (0), in storage order; the index SHA-256 is recorded |
| I9 | B-ceil "minimum over the trajectory" | the minimum of the K = 64 validation means of the fit history (every 100 steps); printed as a diagnostic, not paired, optimistic by construction |
| I10 | comparator of B2a, B3s, B6a, B-ceil | `mlp_std` (B2a also against B0, as "B2a − B0 within ±1" asks); B1f against `B1d_best` |
| I11 | rule for the primary comparisons | the Track B thresholds (−5 and −3 SE) and the AUDIT 4.0 replication rule are applied to the comparison against the comparator; the promotion rule itself is against `mlp_std` |
| I12 | "better than B0 by > 1 SE" | z of the paired Δ against the per-event mean of the three controls below −1 |
| I13 | replication (AUDIT 4.0) | seed 1 Δ ≤ −2.5 with the same sign, and the two-seed mean (per-event mean of the two runs against the reference) meets (a) and (b) |
| I14 | guards | only for runs meeting (a)-(c) on a seed, plus the other seed of that rung (AUDIT 4.2 note); G-region computed with the Track A implementation (`run_m2a.g_region_compute`, imported read-only) with `mlp_std`'s arrays as reference |
| I15 | convergence rerun of a comparator shared by two primaries | `B2a_sX_x2` once (B2b and B2c both use it) |
| I16 | B4 templates | front = `abar_front.pt`, stored = `abar.pt` (both checked against `transfer/SHA256SUMS`), full_box = the Track A template of the H200 (checked against its `templates.json`) |
| I17 | 1-SE (B6) | per seed among B6a, B6b, B6c (fewest weights within one paired SE of the best; `registry.select_1se`) |

## 4. Code (all new or changed for this list)

- `scripts/run_m2b_list.py` (driver) and `scripts/run_m2b_list_report.py` (report), pattern of `run_m2a.py`: phases,
  done-markers in the machine directory, sources verified by SHA-256 against a live-stream row (members, E5, B1d),
  `mlp_std.pt` and templates by SHA-256, `--list`, `--pin`, smoke mode under stage `m2b_smoke` (refused without
  `SFM_M2B_SMOKE=1`), divergence records, convergence reruns, guards, report. Every delta through
  `provenance.paired_delta` on the live stream.
- `src/sfm/models/__init__.py`: factories `field_kl` (B6 route) and `field_distill` (B1d); `field_acc` accepts the
  Track A templates through `dist_factory.acceptance_template` (not duplicated); `train_quantile_grid`.
- `configs/m2b/list_recipes.json`: the pinned recipe hashes (generated by `--pin` on the laptop; tracked).
- N5: `run_m2b_lib.PINNED_CONFIG_HASH` + `check_pinned_config` (the eight pre-registered `run_m2b_fit` configs: tracked
  file inside `configs/`, pinned hash); `fit.py` and `fit_queue.py` refuse to run off the laptop of record.
- Tests: `tests/test_run_m2b_list.py` (list against AUDIT 4.2, pins, switches, divergence rules, promotion and
  replication rules, N5, factories, and - with the H200's identity patched in - recognition of every batch-1 run and
  the SHA-256 verification of B1d's sources on the real H200 records in the repository).

**Files of this work (for explicit staging):** new `scripts/run_m2b_list.py`, `scripts/run_m2b_list_report.py`,
`configs/m2b/list_recipes.json`, `tests/test_run_m2b_list.py`; changed `src/sfm/models/__init__.py`,
`scripts/run_m2b_lib.py`, `scripts/run_m2b_fit.py`, `scripts/fit.py`, `scripts/fit_queue.py`, this file. Not touched:
`common.py`, `src/sfm/registry.py`, `src/sfm/metric.py`, `src/sfm/families/`, `models/dist_factory.py`,
`models/paramnet.py`, `scripts/run_m2a*.py` (the modifications there in the working tree are the Track A worker's).
Nothing registered by batch 1 was changed, re-run or re-paired.

**Needed from other workers (not changed by me):**
- review note N3 (`src/sfm/provenance.py::save_tagged` should compare the sidecar's `machine` fields, not only the
  machine id): this list saves no emulated arrays, so it is not affected; the fix belongs to the owner of provenance.
- `run_m2a.g_region_compute` is imported by the guards phase; if its owner renames it, the guards phase stops with an
  ImportError (it is reached only if a run meets the promotion parts (a)-(c)).

## 5. H200 command line, acceptance, copy-back

```
cd <repo>
export SFM_OUT_ROOT=<area>/sfm_out SFM_USER_CACHE=<area>/sfm_cache
L=shape-form-martin/scripts/launch.sh; P=.venv/bin/python
$L m2b_list_ls $P -B shape-form-martin/scripts/run_m2b_list.py --list        # compare with section 2
$L m2b_list $P -B shape-form-martin/scripts/run_m2b_list.py                  # the whole list, one detached job
# stop only with: shape-form-martin/scripts/stop_job.sh <pid file>; resume: the same command again
```

| phase | acceptance |
|---|---|
| verify | `batch-1 runs present: 13/13`; reference set parity PASS; templates front, stored, full_box verified; recipes pinned |
| b1d ... bceil | one `registered:` line per run (31, plus 8 + 2 `_ema` rows); no `DIVERGED` line, else read its `diverged.json` (exploratory: continues; primary / comparator / diagnostic: "awaiting diagnosis") |
| converge | the move of every primary; `<name>_x2` reruns only where a primary moved ≥ 0.5 |
| guards | either no candidate ("no run met the promotion parts") or one guarded row and G-region arrays per candidate |
| report | `LIST_RESULTS.md` and `list_results.json` written; the job log ends `=== finished`; `.exit` = 0; the guard log holds only `pyarrow` import lines |

Copy back `<area>/sfm_out/shape-form-martin/results/machines/h200-g1056-97b197a7/` as for batch 1: tracked
`registry.jsonl`, `runs/m2b_mlp/{B*,Bceil_*}/{done.json,init.json,fit_done.json,diverged.json}`,
`runs/m2b_mlp/{list_decisions.json,list_results.json,LIST_RESULTS.md}`, every `*.tag.json` and
`g_region.json`; by hand (git-ignored): `model.pt` and the per-event arrays of the new runs; not needed:
`checkpoint.pt`, `fit.log`.

**Estimate.** 159,000 training steps; at the measured 38 s per 6000-step fine-tune (6.3 ms per step) that is about
17 min, plus B2b (K = 64) and B6c (width 64) at an estimated 1.5-2 times the step cost (+2 min), KL warm starts
(about 1 min), 41 registry evaluations at about 2 s, data transfer per job (about 3 s × 31): **about 25 min**. Each
convergence rerun adds its doubled steps (at most 8 primaries + 4 comparators, about +15 min in the worst case).

## 6. Laptop dry-run evidence (26 Nov 2026; scratch output root, guard active, pristine mode)

- `--list` prints the 31 runs of section 2 (159,000 training steps) and the given batch-1 runs. On the laptop the
  batch-1 runs show "MISSING" (they exist on the H200 only). With the H200's identity patched in and the returned H200
  machine directory of the repository as output root (read-only test `test_batch1_runs_recognised_and_sources_verified_
  on_the_h200_records`): all 13 batch-1 runs are recognised as registered, the member ranking is `s2_seed14`,
  `s2_seed13`, ..., B1d's sources (best member and E5_s2) verify by SHA-256 against the real H200 rows, the full_box
  template verifies against the H200's Track A `templates.json`, and the H200 reference set is licensed (parity PASS).
- Prerequisites in the scratch root: `val_parity.py --create-refset` (PASS) and `rescore_registered.py --names
  s2_seed11 s2_seed12` (PASS) - so that the smoke ensemble has two registered members.
- **Smoke run end to end through the launcher** (`SFM_M2B_SMOKE=1 ... run_m2b_list.py --smoke`, stage `m2b_smoke`,
  all steps and KL steps divided by 100, injected divergences `B3s_s1` (exploratory) and `B6c_s1` (primary)):
  - job 1 stopped with `stop_job.sh` (SIGTERM, exit 143) right after `B2a_s1` was registered;
  - job 2 (same command) skipped the finished runs by their done-markers and continued; it was stopped with
    `stop_job.sh` in the MIDDLE of a fit (`B6a_s0` after its step-15 checkpoint, before its done-marker);
  - job 3 (same command) logged `[B6a_s0] resuming from step 15 of 60` and ran to the end: exit 0, all twelve phases
    (`verify ... report`), log ends `=== finished`.
  - `B3s_s1` → `diverged` (exploratory), the job continued; `B6c_s1` → `diverged - awaiting diagnosis` (primary), the
    job continued; B6's 1-SE for seed 1 is reported as "not evaluated (B6c_s1)".
  - convergence phase: five smoke primaries "not converged" (60-step fits) → `<name>_x2` reruns with doubled steps,
    each with its comparator rerun once (`B2a_s0_x2` shared by B2b, B2c, B4f).
  - guards phase (forced onto `B2b_s0` in the smoke mode): a guarded registry row and the G-region arrays of
    `B2b_s0` and of `mlp_std` (reference), read by the report's guard table.
  - B4x skipped in the smoke mode (no Track A template in the scratch root), shown as such in `--list` and the report.
  - report: `LIST_RESULTS.md` / `list_results.json` with every table of section 2 (B0 marked INCOMPLETE on the laptop:
    no b0h rows there; B1 retention "not defined" because the smoke E5 shows no gain).
  - guard log: only the tolerated `pyarrow` imports of pandas; no JSON / MD under the output root contains an absolute
    path or the host name; `results/registry.jsonl` and `results/test_access.log` unchanged.
- Tests: `tests/test_run_m2b_list.py` 9 passed; with the related files (`test_provenance`, `test_review_fixes`,
  `test_registry`, `test_train`, `test_guard`, `test_models`, `test_data`) 96 passed.

## 7. Not tested / open

- A divergence raised inside a factory's KL fit (non-finite KL in B1d / B6 / B4 warm starts) is recorded as a
  divergence (code path added after the smoke run; not exercised).
- The laptop runs in the smoke mode only; no number of this list exists yet. The real run, its step costs and
  whether any primary is non-converged are for the H200.

- No real run of the list anywhere; the smoke mode shortens every fit and KL fit by 100 and uses a stand-in ensemble
  of the two members registered on the laptop (`E5_smoke`).
- B4x in the smoke mode: skipped (the laptop has no Track A templates in the scratch root); `verify_template("full_box")`
  is tested on the real H200 template file in the repository.
- The guards phase in the smoke mode is forced onto `B2b_s0` (no smoke run meets the promotion parts).
- Real step costs of B2b (K = 64) and B6c (width 64) on the H200: estimated.
