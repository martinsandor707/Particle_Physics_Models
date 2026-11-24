# Track B (M2B) status: checkpoint of 25 Nov 2026, 00:17 CET

Stopped on request at a clean point. No job is running, the GPU lock is free, and every done-marker matches what
is on disk. The registry has 8 rows for stage `m2b_mlp`, one per finished run below.

All values are validation only: `common.test_nll`, K=64, seed 12345, through `sfm.registry.evaluate`. Deltas are
paired and given in units of 1e-4 nat (± SE), against `mlp_std` (18.117857) and the component formula (18.120061).

## Finished and registered

| name | k | stage-1 epochs run / best (0-based) | val K64 | Δ mlp_std | Δ formula | guards |
|---|---|---|---|---|---|---|
| s1_seed11 | 3841 | 31 / 15 | 18.120038 | +21.81 ± 3.17 | −0.23 ± 3.75 | no |
| s1_seed12 | 3841 | 36 / 20 | 18.119782 | +19.25 ± 2.94 | −2.79 ± 3.88 | no |
| s1_seed13 | 3841 | 46 / 30 | 18.119757 | +19.00 ± 2.60 | −3.04 ± 3.59 | no |
| s1_seed14 | 3841 | 58 / 42 | 18.119114 | +12.57 ± 2.32 | −9.47 ± 3.50 | no |
| s1_seed15 | 3841 | 22 / 6 | 18.121151 | +32.94 ± 4.19 | +10.90 ± 4.82 | no |
| s2_seed11 | 3841 | — (selected step 5000) | 18.118039 | +1.82 ± 1.73 | −20.22 ± 2.78 | yes |
| s2_seed12 | 3841 | — (selected step 3000) | 18.118186 | +3.29 ± 1.70 | −18.75 ± 3.11 | yes |
| b0_seed0 | 3841 | — (selected step 3250) | 18.117976 | +1.19 ± 0.42 | −20.85 ± 2.99 | yes |

Guard columns, all against mlp_std (formula columns are in `summary.json`):

| name | val K1024 | G-K Δ mlp | G-bulk Δ mlp (stratum zb ≥ 1221 mm, n = 14784) | mass beyond zb |
|---|---|---|---|---|
| s2_seed11 | 18.115799 | +2.02 ± 1.72 | +2.03 ± 1.69 | 0.003100 |
| s2_seed12 | 18.115952 | +3.56 ± 1.72 | +3.60 ± 1.68 | 0.003161 |
| b0_seed0 | 18.115738 | +1.42 ± 0.41 | +1.06 ± 0.40 | 0.003114 |

The persistence checks return "not passed" for all three. This carries no information here: none of the three is
a gain over mlp_std (the `main_is_gain` check is false).

No batch was skipped anywhere. Stage 2 and B0 report 0/0 (loss/gradient) from `sfm.train`. Stage 1 also reports
0/0, counted by passive hooks around the unmodified `common.fit`, with every epoch observed.

## In progress

Nothing. `s2_seed13` was started at 00:13:49 and was paused within seconds by SIGSTOP, then killed. It wrote no
checkpoint (the first one comes at step 250) and no done-marker. Its `fit.log` holds a single start line. A re-run
starts the fit from step 0.

## Not started

`s2_seed13`, `s2_seed14`, `s2_seed15`, `b0_seed1`, `b0_seed2`, all six ensembles (E5_s2, E6_s2_mlpstd, the M = 2, 3, 4
representatives, E5_s1), the subset scaling arrays, and the final report.

## Changes to existing modules

None. Nothing under `src/sfm/`, `common.py` or `scripts/fit*.py` was changed. The new files are all mine:
- `scripts/run_m2b_lib.py`, `run_m2b_stage1.py`, `run_m2b_fit.py`, `run_m2b_ensembles.py`, `run_m2b_report.py`
- `configs/m2b/*.toml` (s2_seed11..15, b0_seed0..2)

## Incidents

**Two jobs failed with CUDA OOM while loading data** (23:27 and 23:29). Both times this script held the GPU lock,
but another process held about 7 GB of GPU memory: PIDs 2381884 and 2387935. Neither PID was mine, both exited
shortly afterwards, and the other agent's lock label appeared right after. No training state was lost: the
`s1_seed12` attempt failed before any checkpoint, and both runs went through on the retry.

**Mitigation.** `run_m2b_lib.run_on_gpu` now checks free device memory after it acquires the lock. Below 2–3 GB it
releases the lock, waits 30 s and tries again. It never waits for memory while holding the lock.

## Continue (from the repo root; finished runs are skipped, interrupted ones resume)

```
uv run python -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/s2_seed13.toml shape-form-martin/configs/m2b/s2_seed14.toml shape-form-martin/configs/m2b/s2_seed15.toml shape-form-martin/configs/m2b/b0_seed1.toml shape-form-martin/configs/m2b/b0_seed2.toml
uv run python -B shape-form-martin/scripts/run_m2b_ensembles.py
uv run python -B shape-form-martin/scripts/run_m2b_report.py
```

Expected time: about 4.5 min per fit, so about 23 min for the five fits, then a few minutes for the ensembles.
Stage 1 is complete; `run_m2b_stage1.py` would skip every seed. Logs go to `results/m2b_mlp/logs/` if the output is
redirected there.
