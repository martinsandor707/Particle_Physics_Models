# Statistics/physics review of the change of machine (laptop → H200) — checkpoint status

Written at the coordinator's checkpoint request. Read-only review; this is the only file the reviewer wrote.
Nothing was trained, the test split was never opened, no git state was changed.

**Marking.** FINAL = verified on the laptop from code or data in this session. PROVISIONAL = argued but not
measured, or it depends on a reading on the H200. INFERENCE = product knowledge or reasoning, not checked.
Unit for deltas: 10⁻⁴ nat per event, paired, ± SE over events, validation only.

## Status per question

| # | Question | Status |
|---|---|---|
| 1 | What depends on the machine, what carries over | answered (two sub-items not checked, listed in §1) |
| 2 | Parity on the H200, test numbers, options | answered; depends on three readings on the H200 |
| 3 | Mixed provenance | answered (reasoned; one size not measurable without training) |
| 4 | Safeguard against mis-pairing | answered (specification) |
| 5 | Running fits side by side | partly answered (reasoning complete, nothing measurable here) |
| 6 | Wording of the A6 amendment | partly answered (draft; two user decisions open) |
| 7 | Silent laptop assumptions | partly answered (list below; README and `dist_factory.py` not read through) |

## Measurements made in this session (FINAL, laptop, validation only)

One GPU job under `gpu_lock`, 2 min. Models: `mlp_std`, component formula, and the eight registered networks.
Twelve fold-point streams: native seeds 12345 and 1–8 (grid 144), emulated grid 1056 with seeds 12345, 1, 2.

Sanity of the instrument:
- emulated grid 144 equals native `common.test_nll` bit for bit for all ten models;
- native arrays equal the registered `val_nll.npy` bit for bit for all eight networks;
- emulated grid 1056 equals `results/m0_parity/{mlp,formula}_val_nll_ref1056.npy` bit for bit;
- `m0_parity/{mlp,formula}_val_nll.npy` were written outside `deterministic()`: a deterministic recomputation
  differs by up to 2.67e-5 per event (mean ~1e-8);
- `m0_parity/*_val_nll_ref1056.npy` and Gate 0's `gate0_checks/val_grid1056_{mlp_std,new3_std_n3}.npy` are not
  bit-equal: max 2.67e-5 / 2.86e-5 per event, mean difference −2.5e-8 / +1.2e-8.

**Predicted H200 validation numbers if its launch grid is 1056** (K = 64, seed 12345). These are the acceptance
targets for the re-scoring; they were not saved as arrays.

| model | laptop mean | laptop Δ mlp | predicted mean (float64) | predicted Δ mlp | predicted Δ formula |
|---|---|---|---|---|---|
| mlp_std | 18.117857 | — | 18.1180829 | — | −23.24 ± 2.94 |
| formula | 18.120061 | +22.04 ± 2.93 | 18.1204074 | +23.24 ± 2.94 | — |
| s1_seed11 | 18.120038 | +21.81 ± 3.17 | 18.1202702 | +21.87 ± 3.17 | −1.37 ± 3.71 |
| s1_seed12 | 18.119782 | +19.25 ± 2.94 | 18.1200090 | +19.26 ± 2.94 | −3.98 ± 3.86 |
| s1_seed13 | 18.119757 | +19.00 ± 2.60 | 18.1199865 | +19.04 ± 2.59 | −4.21 ± 3.60 |
| s1_seed14 | 18.119114 | +12.57 ± 2.32 | 18.1193732 | +12.90 ± 2.31 | −10.34 ± 3.48 |
| s1_seed15 | 18.121151 | +32.94 ± 4.19 | 18.1214207 | +33.38 ± 4.20 | +10.13 ± 4.83 |
| s2_seed11 | 18.118039 | +1.82 ± 1.73 | 18.1182652 | +1.82 ± 1.74 | −21.42 ± 2.78 |
| s2_seed12 | 18.118186 | +3.29 ± 1.70 | 18.1184227 | +3.40 ± 1.73 | −19.85 ± 3.14 |
| b0_seed0 | 18.117976 | +1.19 ± 0.42 | 18.1181833 | +1.00 ± 0.41 | −22.24 ± 2.99 |

**Fold noise** (sd over the twelve streams of a mean, one stream; a change of stream moves a number by √2 times this):
- single-model mean: 1.6e-4 (MLP), 2.0e-4 (formula);
- formula − MLP: 0.70 (so 0.99 between two streams — Gate 0's "≈ 1.1e-4" is this two-stream figure);
- stage-1 network − mlp_std: 0.27–0.54; fine-tuned network − mlp_std: 0.21–0.26; b0_seed0 − mlp_std: 0.11;
- network − network: 0.21–0.54;
- fold points carry 1–6 % of the variance of a paired delta. The paired SE already contains them, because every
  event has its own fold points.

**What a mis-pairing costs** (from stored arrays): `mlp_std` on stream 1056 minus `mlp_std` on stream 144, true
difference zero, gives +2.26 ± 2.71. Formula(1056) − MLP(144) gives +25.5 ± 4.1 instead of +23.2 ± 2.9. So an
H200 score paired with a laptop benchmark array looks 2.3 worse against the MLP and 3.5 worse against the formula;
the reverse pairing produces a fake gain of the same size (keep threshold 3, promotion threshold 5).

**D6(c)** (`results/m1_audit/checks/b2_lookup{,_train}.npz`, validation + training, 84,733 events): refined − standard
lookup at K = 64 is −0.46 ± 0.81. Fold points carry 59 % of that variance (from the covariance with the
independent K = 256 points). Fold sd of the mean on one stream 0.62, between two streams 0.88. The rule value −3.0
is 2.9 of those sd away (3.1 full SE); probability that another stream gives ≤ −3.0 on these events: 0.2 %.

## 1. What depends on the machine — answered

Carries over unchanged (FINAL unless marked):
- **Split, cuts, event order**: `default_rng(1)` on the CPU and Parquet content; no GPU involved.
- **Data cache**: content is machine-independent. Copy it and compare SHA-256 with `cache/pion_s1_meta.json`
  (trainval `9984886fb902…`, test `14955929e0a6…`); validation fingerprint `731a199763cc…` is in the tracked
  `results/registry_refs/guard_spec.json`. Hashing the test file is a byte check, not a test access — state this
  in the amendment, because Gate 0 chose not to hash it. NOT CHECKED: whether the tracked
  `results/m0_parity/cache_build.json` also holds these hashes (`cache/` is git-ignored); whether a rebuild from
  Parquet is byte-identical.
- **Gate 0 conclusions about data, code and checkpoints**: split integrity, cache fidelity, units, `Ā`, centring,
  `v0/vs`, `zb` strata, low-momentum boundary, A1b, A1c, A4, A5. None uses fold points.
- **Weights of the eight networks and of both benchmarks**: files, identified by SHA-256
  (`checkpoint_sha256` in `results/registry.jsonl`; `results/m0_parity/checksums.json` for the benchmarks).
- **Milestone 1 conclusions and the run lists** (AUDIT.md §4): they stand. No decision taken there is within
  reach of a stream change (D6(c) above; the other decisions rest on effects of 4–50 against fold noise ≤ 1).
- **Code**: by git commit. The registry rows were written from a dirty tree (`"dirty": true` in all eight rows).

Must be recomputed on the H200:
- **Every per-event array that enters a delta**: benchmark pairing arrays `results/m0_parity/{mlp,formula}_val_nll.npy`,
  the guard references `results/registry_refs/*.npy` (K = 1024 and bulk), `val_nll.npy`, `val_nll_k1024.npy`,
  `val_bulk_zb1221.npy` of all eight networks.
- **Gate 0's stream statements**: A1d, A2 run-to-run floor, A3, the numbers of record of §2.5, the timing probe.
- **M1 audit numbers** are laptop-stream numbers. Quote them as such; do not recompute unless a table mixes them
  with H200 deltas. Expected shift: formula − MLP +22.0 → +23.2.
- **Scores inside training histories** (`best_val`, selected step) are laptop-stream numbers and are part of how
  those networks were trained, nothing more.

Needs more than one line:
- **C1** certified this chain: test-cache tensors × two checkpoints × `common.event_nll` × emulated stream 1056 =
  stored test arrays (max 3.1e-5 for all 14,956 events). With a byte-identical test cache and checkpoints, the
  only new links on the H200 are its random generator and its arithmetic. Both can be checked without test data
  (§2). C1 does not have to be repeated to stay valid.
- **`val_mass_monitors.npz`** uses no fold points. It should reproduce on the H200 to rounding, which makes it a
  free functional check of each copied checkpoint (PROVISIONAL: not measured).
- **Operational trap (FINAL)**: `scripts/parity.py` has no validation-only mode. Its default mode opens test
  (`load_sets(..., allow_test=True)`, line 368) and `--reference-stream` does too (line 318). Producing the H200
  pairing arrays with it would be a third test access by construction. A validation-only path is needed
  (`scripts/mc_noise_val.py` is validation-only but does not write the pairing arrays).

## 2. Parity on the H200 — answered, three readings open

INFERENCE: an H200 uses the same GPU die as an H100 (132 SMs, 2048 threads per SM), so its grid should be
132 × 8 = 1056. Read `sfm.refstream.local_grid()` there.

Checks in order, none touching test:
1. `local_grid()`; torch and CUDA build (lock file: 2.14.0+cu130).
2. Generator: `EmulGen(seed, local_grid())` equals `torch.rand` bit for bit
   (`tests/test_refstream.py::test_emulation_reproduces_torch_rand_bitwise`). This is the test that detects a
   different RNG kernel layout in another torch build.
3. Validation, native `common.test_nll` inside `deterministic()`, both benchmarks, if the grid is 1056:
   - float32 means within 4e-6 of the stored 18.118083953857422 and 18.120407104492188;
   - per event against the laptop's `m0_parity/{mlp,formula}_val_nll_ref1056.npy`: max |Δ| ≤ 1e-4 (expected
     ≤ 3e-5), |Δ mean| ≤ 1e-6. Gate 0's `val_grid1056_*.npy` are the same quantity computed outside
     `deterministic()`; same tolerance, do not expect bit equality;
   - formula − MLP = +0.002324 ± 0.000294;
   - the eight networks against the table above (means to ~2e-6).
4. Independent of the grid: emulated grid 144 on the H200 must reproduce the laptop's native arrays
   (`m0_parity/*_val_nll.npy`, the eight `val_nll.npy`) to ≤ 1e-4 per event. This proves "H200 arithmetic + copied
   data + copied checkpoints = laptop" directly.

If all pass with grid 1056: the H200's native numbers are the earlier study's numbers. The stored per-event test
arrays of that study (`test_v3_nll.npz`, including the report MLP and report baseline without weights) become
valid pairing partners on test for a labelled secondary column. For the two benchmarks the pairing partners
should still be the arrays recomputed in the same run.

If the grid is not 1056, or check 2 fails: the H200 is a third stream. Stored arrays are not pairing partners.
Numbers of record are the H200's own. Parity is shown as on the laptop: emulated 1056 on the H200 (CPU code,
independent of the GPU) against the stored means and the laptop's emulated arrays, plus A3-type statistical
parity. If only the arithmetic differs (another build), the 1e-4 / 4e-6 tolerances are the criterion, not bit
equality.

**Test numbers without opening test.** The stored 18.1208 / 18.1235 are test numbers. If checks 1–4 pass and the
test cache hash matches, the H200's native test arrays are predicted to equal the laptop's
`{mlp,formula}_test_nll_ref1056.npy`, hence the stored arrays, to ~3e-5 per event. Every link is then verified
except the evaluation on the test tensors themselves; the residual risk is a fault that acts on test bytes but
not on validation bytes. Strong inference, not a measurement.

Options for the user (not decided here):
- **A — no access before the final run.** `final_test.py` recomputes both benchmark test arrays first and aborts
  before any finalist is scored unless max |Δ| ≤ 1e-4 and |Δ mean| ≤ 4e-6 against the stored and the laptop's
  emulated arrays. Log stays at two lines until the final run. A failure is found late; it leaks nothing, but the
  abort must not count as "the single run" and the retry is a further logged access.
- **B — a third logged Gate-0-type parity access on the H200 now.** Finds a problem early. Adds a log line and
  one more occasion on which the test file is open; Gate 2c's "exactly two lines" becomes "two on the laptop, one
  on the H200". No selection information is involved either way (two fixed models, known arrays).
Statistically the two are equivalent; the difference is procedural.

## 3. Mixed provenance — answered (reasoned)

What differs between machines at equal seed (FINAL, from code): training fold points (CUDA generator;
`fold_rng = "global"` in `configs/m2b/*.toml`, `sfm/train.py` lines 427–428), the K = 16 validation points used
for `selection = "best_val"` (`train.py` 307–320, 534–538; `common.fit` line 367), accumulation order. What is
identical: initial weights and batch order (CPU generators: `train.py` line 426, `common.py` line 349;
`init_state_sha16` in each stage-1 `done.json` can confirm this on the H200).

- **(i) Five-member ensemble, two laptop + three H200 fine-tunes, scored on the H200: sound.** An ensemble is a
  fixed function; its score does not depend on where the weights came from. The members are draws from the same
  training recipe. Do not retrain seeds 11 and 12 and then choose: that would be a selection. AUDIT §4.2 takes
  S2-11…15 "as given"; keep the two existing ones. A network is now identified by (seed, training machine).
- **(ii) B0 scatter from one laptop + two H200 runs: unbiased in expectation, but not clean.** Three runs give an
  sd with ±50 % uncertainty, so a machine component could be neither seen nor excluded. Suggested: also train
  `b0_seed0` on the H200 (238 s on the laptop). Scatter of record = the three H200 runs; laptop `b0_seed0`
  reported as a fourth, labelled draw. The same-seed laptop/H200 pair then measures directly how much two runs
  with equal init and batch order differ. Fix this rule before any H200 B0 score is seen.
- **(iii) Variant on the H200 against a comparator from the laptop.** "Same seed" couples variant and comparator
  through init, batch order and (with `fold_rng = "dedicated"`) the training fold points. Across machines the
  fold-point coupling is lost: no bias, more scatter between seed replicates. In practice no such pair exists:
  Track A has no fit yet, B2a and B6a are not trained. Rule: a comparator is trained on the machine of its
  variants. B0 is a level reference with another configuration (global generator, best-val), so nothing is lost
  there. Never resume a `checkpoint.pt` on another machine: `fit` checks only `config_hash` (`train.py` line 438)
  and would continue the fold generator on a different stream. `s2_seed13` has no checkpoint (verified).
- **Cost of repeating everything**: stage 1 took 378 + 451 + 560 + 712 + 264 s = 39 min, each fine-tune and B0
  238 s; all eight ≈ 51 min at laptop speed.
- NOT MEASURABLE HERE: how much two runs of one configuration differ on one machine (training is not
  bit-reproducible on CUDA with `deterministic = False`). Settle on the H200 with two repeats of one cheap fit.

## 4. Safeguard — answered (specification, not implemented)

Hazards found in the code (FINAL):
- `sfm/registry.py` lines 92–94 and 410–414 pair against whatever is in `results/m0_parity/`; the row stores only
  the relative path of the benchmark arrays (line 445), `"device": "cuda"` and the torch version (line 446).
- `_guard_spec` (lines 296–299) has no GPU or grid; copied laptop arrays in `results/registry_refs/` would be
  used silently (lines 328–338). `guard_spec.json` is tracked in git, the arrays are not.
- `registered_row` (`scripts/run_m2b_lib.py` 51–67) matches on checkpoint SHA-256, and tracked `done.json` files
  make the run scripts skip: a copied network would be reported with its laptop score and never re-scored.
- `run_m2b_ensembles.py::scaling_table` (lines 177–226) and `run_m2b_report.py` (lines 50–51, 111) subtract
  registered arrays from each other with no check of where they were computed.
- `_save_array` (lines 273–284) with `replace=True` would overwrite a laptop array.

Minimal specification:
1. **Stream tag** stored with every per-event array (sidecar JSON or registry row, plus the array's SHA-256):
   launch grid, seed, K, batch, fold scheme, split fingerprint, native or emulated. **Machine tag** as
   provenance: GPU name, SM count, torch and CUDA build, `deterministic` flag.
2. Benchmark and guard reference arrays live in a directory named by the tag. `registry.evaluate` computes the
   live tag and raises if no reference set with that tag exists. No fallback.
3. One function forms every delta and refuses arrays with different stream tags; `select_1se`, the report and
   ensemble scripts and the gates use it. Each row records the SHA-256 of the benchmark arrays it was paired with.
4. Rows carry `trained_on` and `scored_on`. `latest()` is keyed by (stage, name, stream tag).
5. `fit` stores the machine tag in `checkpoint.pt` and refuses to resume under another one.
6. `load_sets(allow_test=True)` refuses on any machine that is not the machine of record.
7. Laptop arrays and rows: archived with a hash manifest (for example `results/archive/rtx4060_g144/…` with the
   eight rows verbatim), labelled, read-only, never deleted or overwritten.
8. Re-scoring of the eight networks is its own step: verify the checkpoint hash against the laptop row, score
   with guards, append an H200 row.
9. The reference arrays are created once, by one process, before anything else is registered
   (`_bench_guard_arrays` writes with a plain `np.save`, line 336).

## 5. Side-by-side fits — partly answered (nothing measured)

Reasoned from code (PROVISIONAL until checked on the H200):
- Each process owns its generators (`seed_everything`, dedicated generators), so random streams do not depend on
  what else runs. Nothing in `sfm.train` is time-based: evaluation, checkpoints and stopping count steps.
- Evaluations inside `deterministic()` should be identical under load. `common.event_nll` calls `torch.cumsum`,
  for which the context only warns; Gate 0 found eight sequential runs bit-identical. Not shown under concurrency.
- Training is not bit-reproducible even sequentially, so "concurrent = sequential" can be exact only for
  evaluations and for fits run with `deterministic = True`.
- One job per process: `deterministic()` flips a process-wide flag and `refstream.patched_fold` swaps a module
  attribute.
- Must stay one-at-a-time: creation of the reference arrays; the Track A driver's decision files
  (`decisions.json`, `plr.json` are read-modify-write, `run_m2a.py` lines 52–55); any single run directory (two
  processes on one run would interleave checkpoints — a per-run lock is needed once the global lock is relaxed);
  the final test run.
- Timing-dependent today: the Track A budget rule (`BUDGET_S = 4 h`, `run_m2a.py` line 44, drop logic lines
  384–405) sums wall seconds per job, so machine speed and concurrency change which exploratory runs exist. Fix
  before running: on the H200 run the full list, or define the budget in runs.
- Replacement for the exclusive lock: a counting semaphore with N slots plus a free-memory check; memory is not
  the limit (fits reserved 0.7–1.7 GB, guarded evaluations 4.4 GB). N from a throughput measurement with 1, 2, 4,
  8 copies of the 200-step probe; stop where the next doubling adds under ~20 %. Limit CPU threads per process.
- `flock` must work on the H200's filesystem (network filesystems may not honour it); test with two processes.

Checks that would prove it on the H200: (a) one model evaluated four times at once and four times in sequence:
identical array hashes; (b) one short fit with `deterministic = True` alone and beside three other jobs: identical
weight hash; (c) the spread of repeated default-mode fits, sequential against concurrent.

## 6. Amendment wording — partly answered (draft)

Open user decisions that the text depends on: test-access option A or B (§2); whether `b0_seed0` is retrained (§3).

Draft:

> **A6 (revised, change of machine).** From this amendment on the **H200** is the machine of record for every
> registered score and for the single final test evaluation: Track A fits, all Track B training still to run,
> the re-scoring of the eight networks trained on the laptop, the ensembles, refits and scores of PySR equations,
> the θ arms, the scores of the zero-training geometry arm, and all guards. PySR searches, the geometry audit and
> the gates stay where they are and produce no number that enters a delta.
> **Numbers of record** are the benchmark arrays computed on the H200 with the unmodified `common.test_nll` inside
> `deterministic()` [with launch grid 1056: validation 18.118083 / 18.120407, equal to the earlier study's stored
> values within 4e-6]. The laptop's numbers (validation 18.117857 / 18.120061, test 18.120806 / 18.123333, the
> Milestone 1 audit, the eight Track B rows of 24 Nov) are quoted as "laptop (RTX 4060, launch grid 144), other
> sample points".
> **Pairing rule.** A delta is formed only between per-event arrays with the same stream tag, computed on the
> machine of record. Every pre-registered rule is evaluated on those arrays only and is never re-evaluated with
> the other machine's numbers.
> **Provenance.** Weights are identified by SHA-256 and carry the machine they were trained on. A comparator is
> trained on the machine of the variants paired with it. No interrupted fit is resumed on another machine. The
> five-member ensemble consists of seeds 11 and 12 fine-tuned on the laptop and 13–15 on the H200.
> **Test split.** Opened only on the machine of record. [Option A or B.]
> **Gates.** Gate 1: checkpoint hashes of the laptop-trained networks equal the laptop registry rows; every array
> behind a delta carries the H200 stream tag; the predicted values of the re-scoring table are met. Gate 2c:
> the same for `LOCK.json`; no laptop-stream array in any H200 delta; the test-access logs of both machines
> together show exactly the expected lines. Gate 3: the final run happened once, on the machine of record.

## 7. Silent laptop assumptions — partly answered

Found (FINAL):
- `scripts/run_m2a.py` line 43, `FORMULA_OF_RECORD = 18.120061`, checked to 1e-5 at lines 260–266 with a hard
  stop; also AUDIT.md §4.1 row A0 and `results/m2a_dist/STATUS.md`. With grid 1056 the H200 value is 18.120407
  and Track A would stop at its first phase. Machine-independent form: A0 at step 0 equals the benchmark formula
  array of the same stream, per event ≤ 1e-4 and on the mean ≤ 1e-6 (M1 measured 1.3e-5 and −1.9e-8).
- `src/sfm/refstream.py` line 51, `LOCAL_GRID = 144`. `tests/test_refstream.py` lines 51–52 fail on any other GPU.
  The test at lines 72–80 compares the local stream with grid 1056 and requires them to differ: it fails on a
  grid-1056 machine. Use `local_grid()` and compare against a grid that differs from the local one.
- `scripts/audit_lib.py` line 48, `RECORD` (laptop values; Milestone 1 is closed, label it).
- PLAN §8 "(18.1208)" and "(18.1235)"; PLAN §12; README lines 15–19; registry docstring "this machine".
- Budgets: `BUDGET_S` and the `_est_seconds` defaults (0.055 and 0.07 s per step) in `run_m2a.py`; the cost
  columns and totals of AUDIT §4.1 and §4.2; PLAN §15 "about 4 / 2 GPU-hours"; `min_free_mb` 2000–3000;
  `tests/test_families_components.py` line 493 (budget written for the laptop GPU).
- "One GPU job at a time": PLAN §2 row 12, §10, §15; `sfm/gpulock.py`.
- Gate 2c's "exactly these two" log lines assume one machine. `final_test.py` does not exist yet; its refusal of a
  second run must hold across both working copies.
- A gap in the decision as stated: PySR refits and the geometry arm produce scores of finalist candidates. They
  must be scored on the H200 even though the searches and the audit stay on the laptop.

Not done: README and `src/sfm/models/dist_factory.py` read through for further constants; `tests/test_registry.py`
and `tests/test_train.py` for machine assumptions.

## What I still intended to check

- Whether `results/m0_parity/cache_build.json` records the cache hashes (so the H200 copy can be verified from git).
- The K = 1024 and bulk guard arrays on the emulated 1056 stream (targets for the guard re-scoring).
- `tests/test_registry.py`, `tests/test_train.py`, README, `dist_factory.py` for item 7.
- The exact line of `tests/test_train.py` behind the resume-equivalence statement used in §5.
