# Statistics/physics review of the change of machine (laptop → H200) — FINAL

Final version, 25 Nov 2026. It replaces the checkpoint status written by the first reviewer the same night. That
hand-over is kept below wherever it still holds; changed statements are marked **CORRECTED** or **SUPERSEDED**.
Read-only review. This is the only repository file the reviewers wrote. Nothing was trained, the test split was
never opened (not even hashed by the reviewers), `registry.evaluate` was never called, no git state was changed.

**Marking.** FINAL = verified on the laptop from code or data. PROVISIONAL = argued, or it depends on a reading on
the H200. INFERENCE = reasoning or product knowledge, not checked.
**Words.** *Stream* = the fold points a GPU draws for a given seed; it is fixed by the GPU's launch grid (laptop
144, earlier study 1056). *Native* = drawn by the GPU itself; *emulated* = drawn by `sfm.refstream` on the CPU for a
chosen grid. *Record* = a laptop file kept for comparison, never used in a delta.
**Unit for deltas:** 10⁻⁴ nat per event, paired, ± SE over events, validation only.

## 0. Result in brief

1. Questions 1–4 of the hand-over stand. Two corrections: the cache hashes are tracked in
   `results/m0_parity/parity.json` (not in `cache_build.json`); and the per-event tolerance 1e-4 is safe between two
   GPUs running the same kernels but not between different arithmetic implementations (measured: one event at
   1.4e-4 with CPU arithmetic), so every per-event criterion below has a stated fallback.
2. The hand-over's predicted H200 numbers were reproduced through a second code path, and extended to the guards
   (K = 1024 and bulk). These are the acceptance targets if the H200's launch grid is 1056.
3. Side-by-side fits are safe for the numbers: 104 deterministic evaluations run 2–4 at a time on the laptop were
   bit-identical to the registered arrays. Whether they are *faster* is open: on the laptop the total throughput
   did not rise at all (it fell by 6 % for training steps). The number of slots must be measured on the H200, and
   the proposal should be dropped if two jobs give less than 1.15 times the throughput of one.
4. The Track A budget rule depends on wall time and must be replaced before the first Track A fit. Recommended:
   run the pre-registered lists in full on the H200.
5. The A6 amendment is written for the three decisions the user took on 25 Nov (§6). The stop rule of the final
   run is specified with files, hashes and tolerances (§2.3).
6. The packaging classification is correct. One refinement is proposed: the seven test-derived files need not be
   on the H200 before Gate 2c.

## Status per question

| # | Question | Status |
|---|---|---|
| 1 | What depends on the machine, what carries over | answered; final table in §1.3 |
| 2 | Parity on the H200, test numbers | answered; decision (i) taken; stop rule in §2.3 |
| 3 | Mixed provenance | answered; decision (ii) taken |
| 4 | Safeguard against mis-pairing | answered (specification, six additions) |
| 5 | Running fits side by side | answered (proposal for the user; measured on the laptop) |
| 6 | Wording of the A6 amendment | answered (final text for the decided alternatives) |
| 7 | Silent laptop assumptions | answered (complete list) |
| — | Packaging classification, cache hashes, guard targets | answered (§8, §1.1, measurements) |

## Decisions taken by the user on 25 Nov 2026 (relayed by the coordinator)

- **(i) Test parity: no test access before the final run.** Parity on the H200 is shown on validation only. The
  final run recomputes the two benchmarks' test arrays first and stops before any finalist is scored if they do not
  match the stored per-event arrays. The H200 adds no line to the test-access log until then.
- **(ii) The control `b0_seed0` is retrained on the H200.** All three control seeds come from one machine. The
  laptop run is a labelled record and is not used for the seed scatter. Fixed before any H200 control score exists.
- **(iii) PySR searches on the laptop CPU; every likelihood refit and every registered score on the H200.** The
  geometry audit stays on the laptop (training and validation data only); the geometry arm's zero-training scores
  are produced on the H200.

## Measurements, session 1 (FINAL, laptop, validation only) — all still valid

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

**Predicted H200 validation numbers if its launch grid is 1056** (K = 64, seed 12345). Acceptance targets for the
re-scoring. Reproduced in session 2 through `metric.test_nll_multi`, to every digit shown.

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
| b0_seed0 (laptop-trained) | 18.117976 | +1.19 ± 0.42 | 18.1181833 | +1.00 ± 0.41 | −22.24 ± 2.99 |

The float32 means of the two benchmarks on the emulated stream are 18.118083953857422 and 18.120407104492188,
exactly the earlier study's stored values.

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

## Measurements, session 2 (FINAL, laptop, validation only)

Three jobs under the exclusive `gpu_lock` (09:21–09:30 UTC, about 8 min of GPU) and three short CPU-only jobs.
Scratch scripts and outputs are in the session scratchpad, directory `review2/` (`w.py`, `conc.py`, `gt.py`,
`cpu.py`, `cpu2.py`, `cpu3.py`, `out/*.json`). They are temporary and not part of the package.

**A. Evaluations side by side give the registered arrays, bit for bit.**

| processes at once | what each did | arrays compared | equal to the registered array |
|---|---|---|---|
| 1 | `common.test_nll` inside `deterministic()`, b0_seed0 | 6 | 6 |
| 2 | same call; b0_seed0, s2_seed11 | 16 | 16 |
| 3 | b0_seed0, s2_seed11, s2_seed12 | 24 | 24 |
| 4 | b0_seed0 (twice), s2_seed11, s1_seed15 | 32 | 32 |
| 3 | two K = 64 loops beside one guard evaluation | 32 | 32 |

- The processes overlapped for 99.9 % of their run time (61 % in the last row, the guard job being fully covered).
- Guard arrays (`registry._guard_arrays`: K = 1024 in chunks of 64, and the bulk metric on the stratum
  `zb ≥ 1221 mm`): b0_seed0 alone and s2_seed11 beside two other jobs. All four arrays are bit-identical to the
  stored `val_nll_k1024.npy` and `val_bulk_zb1221.npy`.

**B. Throughput on the laptop does not rise with the number of processes.**

| processes | K = 64 evaluations per second, total | training-like steps per second, total (per process) |
|---|---|---|
| 1 | 0.648 | 28.2 (28.2); repeated: 28.1 |
| 2 | 0.651 | 26.5 (13.2) |
| 3 | 0.651 | 26.3 (8.8) |
| 4 | 0.652 | 26.3 (6.6) |

- A training-like step is the forward and backward pass of the training loss for `mlp_std` at batch 256 and K = 16,
  with the per-step synchronisation of `sfm.train.fit`. No optimiser step was taken; the weights were checked to be
  unchanged. One process gives 0.0355 s per step, as Gate 0's timing probe (0.035 s).
- One process already keeps the GPU busy 86–92 % of the time (`nvidia-smi`). Without a shared-context service the
  driver gives each process the whole GPU in turn, so a second process only waits.

**C. Device memory per process**, with only the validation set on the device. The figures quoted earlier in the
campaign (fit about 1.6 GB allocated, guarded evaluation up to 4.4 GB reserved) come from processes that also held
the training set and, for the evaluation, the memory the allocator kept from the fit; they are consistent with
these (INFERENCE).

| job | peak reserved by torch | device memory in use (`nvidia-smi`) |
|---|---|---|
| K = 64 evaluation | 0.26 GB | 0.39 GB |
| training-like step, batch 256, K = 16 | 1.42 GB (1.32 allocated) | 1.55 GB |
| guard evaluation (K = 1024, bulk) | 2.89 GB | 3.03 GB |

Memory adds up linearly (4 step processes: 6.23 GB). The CUDA context of a process costs about 0.14 GB.

**D. Guard targets on the emulated reference stream (grid 1056).** The emulated draw was put in place of
`sfm.metric.draw` at run time (`refstream.patched_fold` covers only `common.fold`, so no library path exists).
Check of the instrument on grid 144: K = 64 for the eight networks, K = 1024 and bulk for the three guarded
networks and for both benchmarks, 18 arrays in all, are bit-identical to the stored laptop arrays. On grid 1056
the K = 64 arrays of both benchmarks are byte-identical to `m0_parity/*_val_nll_ref1056.npy`.
Stratum `zb ≥ 1221 mm`: 14,784 events; 166 short events; no undefined bulk value.

| model | K = 1024 mean | Δ mlp at K = 1024 | bulk mean on the stratum | Δ mlp, bulk | Δ mlp, report metric on the stratum |
|---|---|---|---|---|---|
| mlp_std | 18.1155556 | — | 18.0429497 | — | — |
| formula | 18.1177364 | +21.81 ± 2.83 | 18.0448778 | +19.28 ± 2.75 | +21.96 ± 2.84 |
| b0_seed0 (laptop-trained) | 18.1156983 | +1.43 ± 0.41 | 18.0430490 | +0.99 ± 0.42 | +0.84 ± 0.41 |
| s2_seed11 | 18.1157633 | +2.08 ± 1.72 | 18.0431667 | +2.17 ± 1.69 | +2.06 ± 1.71 |
| s2_seed12 | 18.1159234 | +3.68 ± 1.72 | 18.0433190 | +3.69 ± 1.64 | +3.98 ± 1.72 |

- Against the formula: b0_seed0 −20.38 ± 2.89 (K = 1024) and −18.29 ± 2.80 (bulk); s2_seed11 −19.73 ± 2.68 and
  −17.11 ± 2.62; s2_seed12 −18.13 ± 3.04 and −15.59 ± 2.90.
- Laptop stream for comparison: MLP 18.1155965 and 18.0423351; formula 18.1177640 and 18.0442180; formula − MLP
  +21.67 ± 2.85 (K = 1024), +18.83 ± 2.75 (bulk), +20.40 ± 2.83 (report metric on the stratum).
- A mis-paired guard, same model on stream 1056 minus stream 144 (true difference zero): K = 1024 gives
  −0.41 ± 0.67 (MLP) and −0.28 ± 0.70 (formula); the bulk guard gives +6.15 ± 2.72 and +6.60 ± 2.79.
- The persistence flags of the guards (passed, resolved) differ between the two streams for some networks. They
  must be read from H200 arrays only.
- The 20 emulated arrays are in the scratchpad (`review2/out/targets_grid1056/`). File SHA-256 of the four guard
  targets: MLP K = 1024 `52c846fc…2fa503`, formula K = 1024 `2b4c117f…b858ac`, MLP bulk `0987de3d…a69cd3`, formula
  bulk `b44f838a…7bccfb`. If per-event targets are wanted on the H200, regenerate them into the quarantine folder
  as laptop records (stream: emulated 1056); the means above suffice otherwise.

**E. Another arithmetic on identical points: CPU instead of the laptop GPU** (emulated grid 144, same cache bytes,
same checkpoints; CPU repeats are bit-identical).

| model | max per-event difference | second largest | rms | difference of the means |
|---|---|---|---|---|
| b0_seed0 | 1.43e-4 (one event) | 2.9e-5 | 4.6e-6 | +6.8e-9 |
| s2_seed11 | 1.41e-4 (the same event) | 2.9e-5 | 4.6e-6 | −4.8e-8 |
| s1_seed15 | 1.32e-4 (the same event) | 2.7e-5 | not recorded | −5.2e-9 |

- 14,949 of 14,950 events agree within 3e-5. One event (validation index 4211: 429 hits, 13.8 GeV, `zb` = 1437 mm)
  differs by 1.3e-4 to 1.4e-4 for all three networks, and for b0_seed0 on grid 1056 as well. It is therefore a
  property of that event's arithmetic and not of the fold points or of one model. The cause was not investigated.
- Mass monitors (no fold points): per event within 7.5e-6 relative, means within 1.3e-7 relative.
- Meaning: two GPUs running the same kernels agreed to 3.1e-5 on all 14,956 test events (C1). A different
  arithmetic implementation can put a single event above 1e-4 while the mean moves by less than 1e-7. A strict
  "every event within 1e-4" rule can therefore fail for a harmless reason.

**F. Files and hashes.**
- `results/m0_parity/cache_build.json` (tracked) holds **no** cache hash. `results/m0_parity/parity.json` (tracked,
  written by the Gate 0 parity run at 16:22:17 UTC on 24 Nov) holds both under `cache_files`: train/val
  `9984886f…6ab4b4` (385,953,467 bytes), test `14955929…c9ec88` (68,476,893 bytes).
- The train/val cache was re-hashed in this session: equal. The test cache was not hashed by the reviewers; the
  packaging worker's byte hash in `transfer/SHA256SUMS` equals the tracked value.
- Both cache files carry the modification time of the build (16:18:25 UTC), before the two logged accesses.
- `results/test_access.log`: two lines, SHA-256 `effa9734743f51959806d7d8b87e2d262c10773a9be25f2104ceeca9557f0c60`,
  unchanged since commit `bcb76fa`.
- No Track A fit exists: the registry has eight rows, all `m2b_mlp`; `results/m2a_dist/` holds only its STATUS file.

## 1. What depends on the machine — answered

### 1.1 Carries over unchanged (hand-over text, FINAL unless marked)

- **Split, cuts, event order**: `default_rng(1)` on the CPU and Parquet content; no GPU involved.
- **Data cache**: content is machine-independent. Copy it and compare SHA-256. Validation fingerprint
  `731a199763cc…` is in the tracked `results/registry_refs/guard_spec.json`. Hashing the test file is a byte check,
  not a test access.
  **CORRECTED:** the reference for the hashes is the tracked `parity.json` (measurement F), not only the copied
  `cache/pion_s1_meta.json`, which travels with the files it describes. `cache_build.json` has no hashes. The Gate 0
  reviewer did not hash the test file; the hash on record was computed by the cache build and copied into
  `parity.json` by the parity run. Whether a rebuild from Parquet is byte-identical remains unchecked and is not
  needed.
- **Gate 0 conclusions about data, code and checkpoints**: split integrity, cache fidelity, units, `Ā`, centring,
  `v0/vs`, `zb` strata, low-momentum boundary, A1b, A1c, A4, A5. None uses fold points.
- **Weights of the eight networks and of both benchmarks**: files, identified by SHA-256
  (`checkpoint_sha256` in `results/registry.jsonl`; `results/m0_parity/checksums.json` for the benchmarks).
- **Milestone 1 conclusions and the run lists** (AUDIT.md §4): they stand. No decision taken there is within
  reach of a stream change (D6(c) above; the other decisions rest on effects of 4–50 against fold noise ≤ 1).
- **Code**: by git commit. The registry rows were written from a dirty tree (`"dirty": true` in all eight rows).

### 1.2 Must be recomputed on the H200 (hand-over text, FINAL)

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
  free functional check of each copied checkpoint. **Now measured with CPU arithmetic** (measurement E): 7.5e-6
  relative per event at most. Tolerance on the H200: 1e-4 relative per event, 1e-6 relative on the means.
- **Operational trap (FINAL)**: `scripts/parity.py` has no validation-only mode. Its default mode opens test
  (`load_sets(..., allow_test=True)`, line 368) and `--reference-stream` does too (line 318). Producing the H200
  pairing arrays with it would be a test access by construction. A validation-only path is needed
  (`scripts/mc_noise_val.py` is validation-only but does not write the pairing arrays, and it overwrites the tracked
  laptop record `mc_noise_val.json`).

### 1.3 Final table: what carries over and what is recomputed (ready to paste into section 16, item 4)

**Carries over unchanged**

| Item | Why it carries over | What shows it on the H200 |
|---|---|---|
| Code and tests | identified by the git commit | checkout descends from `c9b7f72`; the test suite passes except the two launch-grid tests of `tests/test_refstream.py` (§7); `test_emulation_reproduces_torch_rand_bitwise` must pass |
| Split, cuts, event order | made on the CPU with `default_rng(1)`; stored in the cache | follows from the cache hashes |
| Both cache files (byte copies, not rebuilt) | Gate 0 and C1 certified these bytes; a rebuild on another CPU may differ in the last bit | SHA-256 equal to the tracked `parity.json` (`9984886f…`, `14955929…`; test file by byte hash only); validation fingerprint recomputed there equals `731a1997…58a7` |
| Benchmark files `mlp_std.pt`, `new3_std_n3.pt`, `abar.pt`, `test_v3.json` | weights and the formula's structure are bytes | SHA-256 of `transfer/SHA256SUMS`; functionally by row R1 below |
| Weights of the eight laptop-trained networks | a network is a fixed function wherever it was trained | SHA-256 equal to `checkpoint_sha256` of the laptop registry rows |
| `abar_front.pt`, `abar_front_back.pt`, the two event-number arrays | made from training events, respectively from the split; no fold points | SHA-256 |
| Gate 0 conclusions on data, code and checkpoints | none uses fold points; they are statements about copied bytes | nothing to redo |
| C1 (exact test parity on the reference stream) | it certified bytes × code × stream 1056; bytes are identical by hash | the two new links (generator, arithmetic) by R0 and R1; the test evaluation itself by the first step of the final run |
| Milestone 1 audit and the run lists of AUDIT §4 | no decision is within reach of a change of stream | nothing to redo; numbers quoted as "laptop, other sample points" |
| Laptop per-event arrays, the eight registry rows, training histories | they are records | kept in quarantine; never at a live path, never in a delta |

**Not carried over** (laptop statements): Gate 0's A1(d) and A3 (statistical parity of the laptop stream), the A2
run-to-run floor, the numbers of record of Gate 0 §2.5, the timing probe, every cost and budget figure.

**Recomputed on the H200, in this order.** Targets marked ◆ hold only if the launch grid is 1056; the fallback
follows the table.

| # | Step | Why | Acceptance |
|---|---|---|---|
| R0 | Identity of the machine: GPU name, `refstream.local_grid()`, torch and CUDA build, emulator against `torch.rand` | the stream is a property of the GPU and of the torch build | grid read and recorded (expected 1056); torch 2.14.0+cu130; `EmulGen(seed, local_grid())` equals `torch.rand` bit for bit |
| R1 | Benchmark pairing arrays on validation: `common.test_nll` inside `deterministic()`, K = 64, both benchmarks | every delta is paired against them | ◆ float32 means 18.118083953857422 and 18.120407104492188 within 4e-6; ◆ against the laptop's `*_val_nll_ref1056.npy`: every event within 1e-4 (expected ≤ 3e-5), means within 1e-6; ◆ formula − MLP = +23.24 ± 2.94; a second run is bit-identical |
| R1b | The laptop stream emulated on the H200 (grid 144), eight networks | separates arithmetic from fold points, whatever the grid | every event within 1e-4 of the laptop's registered `val_nll.npy`, means within 1e-6 |
| R2 | Guard reference arrays (K = 1024; bulk on the stratum), once, by one process, before any registration | guards are paired against them | ◆ means 18.1155556 / 18.1177364 (K = 1024) and 18.0429497 / 18.0448778 (bulk), within 1e-6; ◆ formula − MLP +21.81 ± 2.83 and +19.28 ± 2.75; 14,784 events in the stratum, 166 short, none undefined |
| R3 | Run-to-run floor outside `deterministic()` | Gate 0's 2.9e-5 is a laptop measurement | two runs within 1e-4 per event and 1e-6 on the mean (amendment A2) |
| R4 | Re-scoring of the eight laptop-trained networks, guards for b0_seed0, s2_seed11, s2_seed12 | their laptop arrays are on other fold points | ◆ K = 64 table of session 1 and guard table D: means within 1e-6, deltas within 0.01; mass monitors within 1e-4 relative per event of the laptop's files |
| R5 | Remaining Track B training: fine-tunes of seeds 13–15; the control for seeds 0, 1 and 2 | not yet trained, or (control seed 0) decision (ii) | no numeric target; configuration equal to the pre-registered files; no checkpoint from the laptop resumed; the H200 control for seed 0 has another SHA-256 than `75de31f2…` |
| R6 | Ensembles and their subsets | functions of weights, scored on H200 points | the three representative subsets are bit-equal to their registered arrays; no laptop array is read |
| R7 | Track A: templates, then A0 at step 0, then the list of AUDIT §4.1 | never run | A0 at step 0 against the H200 formula array: every event within 1e-4, mean within 1e-6 (◆ mean 18.120407) |
| R8 | Remaining Track B list (AUDIT §4.2), θ arms | never run | comparators trained on the H200 |
| R9 | Refits, controls and scores of PySR equations; scores of the geometry arm | decision (iii) | each is an H200 registry row; translator round trip within 1e-5 against reference values shipped with the equation |
| R10 | Final test evaluation | single run on the machine of record | stop rule of §2.3 |
| R11 | Timing: throughput calibration | costs and slot count are machine properties | §5 |

**Fallback when the grid is not 1056** (or R0's generator check fails): the ◆ targets do not apply to the native
arrays. Then (a) R1, R2 and R4 are repeated on the *emulated* grid 1056 on the H200 and must meet the same ◆ targets;
this needs the emulated draw for `sfm.metric.draw`, which exists only for `common.fold` today. (b) R1b must pass.
(c) The native arrays are the new numbers of record and are accepted on A3: correlation with the emulated arrays
≥ 0.9995 and rms of the per-event difference within ±15 % of the laptop's validation values (0.033 for the MLP,
0.036 for the formula; Gate 0 §2.4). (d) Stored arrays are never pairing partners.

**Fallback when a per-event limit is exceeded but the means agree** (another arithmetic, as in measurement E): the
step passes if at most three events exceed 1e-4, none exceeds 1e-3, the rms is ≤ 1e-5 and the means agree within
1e-6. The finding is recorded, and the per-event clause of the final run's stop rule is then set as in §2.3.

## 2. Parity on the H200 — answered

### 2.1 Checks without test data (hand-over text; tolerances completed)

INFERENCE: an H200 uses the same GPU die as an H100 (132 SMs, 2048 threads per SM), so its grid should be
132 × 8 = 1056. Read `sfm.refstream.local_grid()` there. A MIG slice has another SM count. Whether a thread limit
of the multi-process service changes the reported SM count is not known: do not set one, and read the grid inside
the same configuration that the jobs will use.

Checks in order, none touching test (they are rows R0, R1, R1b of §1.3):
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
   - the eight networks against the table above (means within 1e-6).
4. Independent of the grid: emulated grid 144 on the H200 must reproduce the laptop's native arrays
   (the eight `val_nll.npy`) to ≤ 1e-4 per event. This proves "H200 arithmetic + copied data + copied checkpoints
   = laptop" directly. **CORRECTED:** the eight registered arrays are the clean partners. The hand-over also named
   `m0_parity/*_val_nll.npy`; those were written outside `deterministic()` and carry up to 2.7e-5 of noise of their
   own, which the 1e-4 limit absorbs but which rules out bit equality.

If all pass with grid 1056: the H200's native numbers are the earlier study's numbers. The stored per-event test
arrays of that study (`test_v3_nll.npz`, including the report MLP and report baseline without weights) become
valid pairing partners on test for a labelled secondary column. For the two benchmarks the pairing partners
are the arrays recomputed in the same run.

If the grid is not 1056, or check 2 fails: the H200 is a third stream; see the fallback of §1.3.

### 2.2 Test numbers without opening test (hand-over text, FINAL)

The stored 18.1208 / 18.1235 are test numbers. If checks 1–4 pass and the test cache hash matches, the H200's
native test arrays are predicted to equal the laptop's `{mlp,formula}_test_nll_ref1056.npy`, hence the stored
arrays, to ~3e-5 per event. Every link is then verified except the evaluation on the test tensors themselves; the
residual risk is a fault that acts on test bytes but not on validation bytes. Strong inference, not a measurement.

**SUPERSEDED:** the hand-over's two options. The user chose option A (decision (i)). Option B, a third logged
Gate-0-type parity access on the H200 before the final run, is not taken.

### 2.3 Decision (i): what the first step of the final run must compare

One process, on the H200, one line in the test-access log. Order:

0. **Before the test file is opened.** Machine equals the one named in `LOCK.json`; `LOCK.json` has the hash that
   Gate 2c passed; no path override (`SFM_RESULTS_DIR`, `SFM_CACHE_DIR`) is set; SHA-256 of
   `cache/pion_s1_test.pt` equals `14955929e0a6cfad42e43aa6fa81787a072d9f26d37c3054ef627faac4c9ec88`.
1. **Event order.** The test event numbers, in order, equal `results/m0_parity/test_event_number.npy`
   (file SHA-256 `ab5dcf38…5263a4`).
2. **Benchmark arrays.** Per-event test NLL of `mlp_std.pt` and of the component formula with the unmodified
   `common.test_nll` inside `deterministic()` (K = 64, seed 12345, batches of 128).
   - Compared with: the earlier study's stored arrays in
     `shape-form-proton/results/particles/pion/test_v3_nll.npz`, keys `std:mlp_ft` (MLP) and `std:new` (formula);
     file SHA-256 `174524ef…32fd81b`.
   - Tolerance, the criteria of C1: for both models every one of the 14,956 events within 1e-4 and the float64
     means within 4e-6. C1 measured 2.9e-5 and 3.1e-5, and 3.0e-8 and −2.5e-8.
   - Expected values: means 18.120842 and 18.123458; formula − MLP = 0.0026163 ± 0.000292.
   - Reported beside it, not deciding: the same comparison with the laptop's emulated arrays
     `{mlp,formula}_test_nll_ref1056.npy` (`eae19ea0…`, `3ecc78bb…`); expected within 1.5e-4 by the triangle
     inequality.
3. **Only then** are the finalists' weights loaded and scored, paired against the benchmark arrays of step 2.

**Per-event clause when the H200's arithmetic differs.** If R1 left any validation event above 5e-5 against the
laptop's emulated arrays, Gate 2c fixes, before the run, the per-event clause as: at most three events above 1e-4,
none above 1e-3, rms of the differences ≤ 1e-5 (C1: 5.0e-6). The mean clause stays at 4e-6. Reason: measurement E.
The power is unchanged: a wrong event order, model, cache or stream moves 99 % of the events by more than 1e-4
(fold noise alone has an rms of 0.03).

**If the launch grid is not 1056.** Step 2 is made on the emulated reference stream
(`refstream.test_nll_ref(model, test, grid=1056)` inside `deterministic()`), with the same partner and tolerance:
this is C1 repeated on the H200. The native arrays are then computed as pairing partners and must satisfy A3
against the emulated ones (correlation ≥ 0.9995, rms within ±15 % of the validation analogue measured in R1).

**A stop is not the final run.** Nothing but two fixed models with known expected arrays has been evaluated, so no
selection information exists. The log line stays. A stop record with the measured differences is written. A new
start needs the cause, the user's go-ahead, a byte-identical `LOCK.json` and a new log line; Gate 3 reviews all
lines. `final_test.py`'s refusal of a second run therefore distinguishes "stopped before any finalist was loaded"
from "finalists scored".

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
- **(ii) B0 scatter. SUPERSEDED by decision (ii):** the control is trained on the H200 for seeds 0, 1 and 2, and
  the scatter and the B0 mean use these three. The laptop's `b0_seed0` is re-scored on the H200 (16 s, part of R4)
  and shown as a labelled record. The pair "laptop-trained and H200-trained seed 0, both scored on the H200"
  measures directly how far two runs with equal initial weights and batch order are apart; it is reported with its
  paired SE and has no pass or fail attached. Consequences found in the code:
  - `train.fit` returns the stored laptop network whenever `fit_done.json` exists (`train.py` 407–418), and the
    tracked done-markers of `b0_seed0` arrive with git. With the laptop's `model.pt` copied to its live path, a
    "retraining" would silently be the laptop network. The laptop run must not sit at the path the H200 run uses.
  - `run_m2b_report.py` builds the B0 mean and scatter from the names `b0_seed{0,1,2}` (lines 50–51 and 104–113),
    and its column "Δ B0 mean" (lines 60–61) is filled for every row. Either the laptop record is renamed, or the
    H200 run of seed 0 gets a new name and the report takes its three control names from a list. Whichever is
    chosen, the scatter and the mean must come from the three H200-trained runs only.
  - A renamed run has another `config_hash` (the name is hashed). Equality of the recipe is shown by comparing the
    TOML files apart from `run.name`.
- **(iii) Variant on the H200 against a comparator from the laptop.** "Same seed" couples variant and comparator
  through init, batch order and (with `fold_rng = "dedicated"`) the training fold points. Across machines the
  fold-point coupling is lost: no bias, more scatter between seed replicates. In practice no such pair exists:
  Track A has no fit yet, B2a and B6a are not trained. Rule: a comparator is trained on the machine of its
  variants. Never resume a `checkpoint.pt` on another machine: `fit` checks only `config_hash` (`train.py` line
  438) and would continue the fold generator on a different stream. `s2_seed13` has no checkpoint (verified).
- **Cost of repeating everything**: stage 1 took 378 + 451 + 560 + 712 + 264 s = 39 min, each fine-tune and B0
  238 s; all eight ≈ 51 min at laptop speed.
- NOT MEASURABLE HERE: how much two runs of one configuration differ on one machine (training is not
  bit-reproducible on CUDA with `deterministic = False`). Check E4 of §5 delivers it on the H200.

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

Minimal specification (hand-over, FINAL):
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

Additions of the final review:

10. **Two parts of the tag, two uses.** *Points* (grid, seed, K, batch, fold scheme, split fingerprint) and
    *arithmetic* (machine, build, native or emulated). A delta of record needs both equal. The labelled secondary
    column against the earlier study's stored arrays needs equal points only. A comparison of an H200 array with a
    laptop-emulated array is a tolerance check, never a delta.
11. `registered_row` (both libraries) matches on checkpoint SHA-256 **and** stream tag; `_guard_spec` contains the
    tag.
12. The reference set of the H200 is made read-only after R1 and R2, and its array hashes go into a tracked JSON
    file, so that the gates on the laptop can identify it.
13. Rows also store a hash of the weight tensors. `checkpoint_sha256` is a file hash, and `model.pt` contains wall
    time (`seconds` in the history and the summary, `saved_utc`; `train.py` 529–531, 563–571): two bit-identical
    trainings give different file hashes.
14. `final_test.py` and any test access refuse when `SFM_RESULTS_DIR` or `SFM_CACHE_DIR` is set (the log would be
    written outside the tracked file), and take the machine of record from `LOCK.json`.
15. Scripts that overwrite tracked laptop records are blocked on the H200 or write to a machine-named file:
    `mc_noise_val.py` (`mc_noise_val.json`), `build_cache.py` (`cache_build.json`, both caches),
    `check_protected.py record` (`checksums.json`), `run_m2b_report.py` (`summary.json`, intended), `parity.py`.

## 5. Running fits side by side — answered (proposal for the user; replaces the partial answer)

### 5.1 What is established

- **Numbers do not depend on what else runs** (FINAL for evaluations, measurement A). Each process owns its
  generators (`seed_everything`, dedicated generators, and the process-wide CUDA generator used by
  `fold_rng = "global"`). The launch grid depends on the device and the tensor size, not on load.
- **Nothing in a fit depends on time** (FINAL, from `sfm/train.py`): evaluation, checkpoints and stopping count
  steps. Wall time appears only in records (`seconds`, `saved_utc`).
- **Fits are not bit-reproducible even alone** (`deterministic = False` in every campaign configuration), so
  "side by side equals one after another" can be exact only for evaluations and for fits run with
  `deterministic = True` (`tests/test_train.py` lines 300–306 show bitwise resume equivalence in that mode, on
  the weight tensors and on the history without `seconds`; lines 309–319 bound the default mode at 1e-4).
- **Speed is the open point** (measurement B). On the laptop two processes are not faster than one. On a large GPU
  a single small job leaves most of the device idle, but the driver still runs one process at a time unless the
  multi-process service is used. INFERENCE: without that service the gain on the H200 is bounded by the idle time
  of a single job; with it, by how much of the device one job fills.

### 5.2 The proposal (ready to paste into section 16, item 5)

| Point | Proposal |
|---|---|
| Unit | One operating-system process per job (one fit with its registered evaluation, or one evaluation). Never threads: `deterministic()` flips a process-wide switch, `refstream.patched_fold` swaps a module attribute, seeding is process-wide. |
| How many | N is measured on the H200 (§5.3), not chosen in advance. Expected 1–4 without the multi-process service, more only with it (INFERENCE). Memory would allow more than 16. First use: N ≤ 8. |
| Adopt or drop | Adopt only if two jobs give at least 1.15 times the throughput of one. Otherwise N = 1, the exclusive lock stays, and nothing else in this table is built. |
| Lock | N slot files `cache/gpu.slot.<i>`, each an exclusive `flock` taken with the present `gpu_lock` code; a job takes the first free slot. `SFM_GPU_SLOTS` defaults to 1, which is today's behaviour, so the laptop and the tests are unchanged. |
| Exclusive steps | A second file `cache/gpu.excl`: ordinary jobs hold it shared for their lifetime, exclusive steps hold it exclusively. Exclusive steps are started when the queue is empty. |
| One run, one process | An exclusive, non-blocking `flock` on `<run directory>/.run.lock`, taken first and held to the end; the done-marker is read again after it is held. A second process on the same run stops at once and names the holder. One such lock per driver as well. |
| Shared files | Only the single driver process writes `decisions.json`, `plr.json` and the report files. Worker processes write only inside their run directory and append to the registry (already under `flock`). |
| Memory, static | N × P ≤ half of the device memory, where P is the largest device footprint of any job class measured on the H200 (laptop: about 4.6 GB for a fit with a guarded evaluation). |
| Memory, at each start | After taking a slot and before allocating: (a) the sum of the footprints declared by the other slot holders, plus the job's own, plus 4 GB, is at most 80 % of the device; (b) the free memory reported by the device is at least the job's own footprint plus 4 GB. Each slot file records its holder's declared footprint. If either fails: release the slot, wait 30 s, retry. |
| Waiting processes | A process that waits holds only its CUDA context (0.14 GB on the laptop), because `common.py` creates its grid tensors at import. This is harmless on the H200 and is counted by (b). The rules of §15 stay: no tensors on the device before the slot is held, `empty_cache()` before it is released, never wait for memory while holding a slot. |
| CPU | Threads per process limited to cores ÷ N (at most 8); N ≤ physical cores − 2. |
| File system | `flock` must be honoured where the lock files and `registry.jsonl` live. Put the lock files on a local disk (`SFM_GPU_LOCK` already relocates them) and run check E5. |

Why condition (a) is needed: the present check reads the free memory at one instant. With N slots, several jobs can
start together and all see the memory free before any of them has allocated. The declared sum removes that race.
On a dedicated H200 with N ≤ 8 both conditions always hold (8 × 5 GB against 141 GB, the latter INFERENCE); they
matter if the GPU is shared with other users or N is raised.

### 5.3 How N is determined on the H200

Nothing is registered and nothing is written into the campaign results (scratch result directory).

- **Workloads.** P1: the Track B training step (`mlp_std` architecture, batch 256, K = 16). P2: a Track A training
  step (`survival_exact` component model, batch 256, K = 16). P3: a guarded evaluation. The timing probe of
  `scripts/parity.py` cannot be used: that script opens the test split.
- **Measurement.** For n = 1, 2, 3, 4, 6, 8, 12, 16: start n identical processes behind a common start signal; each
  runs 20 warm-up steps and then a timed window of at least 20 s, with `torch.cuda.synchronize()` at both ends.
  Record steps per second per process, the total T(n), the lowest free device memory, GPU utilisation and CPU load.
  Three repeats per n; use the median. Do this without the multi-process service and, if the user allows it, with it.
- **Rule.** N is the smallest n with T(n) ≥ 0.9 × the largest T measured, taken separately for P1 and P2; use the
  smaller of the two. Stop measuring once T has stayed below 0.9 × the running maximum for two consecutive n.
  Caps: the static memory rule, the CPU rule, and a per-job slow-down of at most 4. If T(2) < 1.15 × T(1): N = 1.
  P3 is run once at the chosen N, mixed with N − 1 fits, to confirm memory and throughput.
- **Applied to the laptop numbers:** T(2)/T(1) = 0.94 for steps and 1.00 for evaluations, so N = 1.
- **If the multi-process service is used:** read `local_grid()` inside it (must equal the value of R0), set no
  thread limit, and pass check E1 against a reference computed without it. One faulting client can end all
  clients; the fits resume from their checkpoints.

### 5.4 What stays strictly one at a time, and why

| Step | Why |
|---|---|
| R0–R4 of §1.3 (identity, reference arrays, re-scoring) | every later delta rests on these files; `_bench_guard_arrays` checks and then writes without a lock; the re-scoring also provides the sequential reference of check E1 |
| The throughput calibration | it measures time |
| Every decision of a driver: learning-rate pilot, A0 parity, A0 drift rule, convergence reruns, keep and equal-NLL rules that trigger second seeds, the content of A11, 1-SE selection, promotion | each is a function of registered arrays and must be evaluated once, by one process, in the pre-registered order, after all its inputs exist. The fits between two decision points can run side by side |
| Report scripts and `LOCK.json` | they rewrite shared files |
| The final test run | single test access; alone on the GPU, no other campaign process alive |
| Commits | taken when no job runs, so that no tracked file is caught half-way |

Dependencies inside Track A that limit what can run together (from `scripts/run_m2a.py`): templates → A0 at step 0
→ three polynomial pilot fits → frozen learning rate → A2s seed 0 → three network pilot fits → frozen learning
rate → A0 (two seeds) → drift rule → A1, A2e, A2s → A2n (needs A2s of the same seed) → twelve polynomial primaries
(need A2s of the same seed) → six network primaries (each needs its polynomial run) and the convergence reruns →
eighteen exploratory fits → second seeds of those that pass → guards → Stage 2.

### 5.5 The budget rule must not depend on time

**Found (FINAL).** `scripts/run_m2a.py` line 44 sets `BUDGET_S` to 4 h. Lines 358–405 drop exploratory runs in the
pre-registered order while "seconds used so far + expected seconds of what is pending" exceeds it. Both terms are
wall time: the sum of `gpu_time.jsonl`, and the median seconds per step of finished runs (defaults 0.055 and 0.07 s,
laptop estimates). A faster machine drops fewer runs; jobs side by side take longer each and drop more. The rule of
AUDIT §4.0 ("if the budget runs out, runs are dropped in the order given") is the same quantity. Also: the last
entry of the drop order in AUDIT §4.1, the network runs of primaries that failed on both seeds, cannot be dropped
by the code, because those runs happen before the exploratory ones.

**Recommended (R1): run the lists in full on the H200.** All eighteen exploratory rungs at seed 0, seed 1 whenever
seed 0 passes the keep or the equal-NLL rule, all network primaries; the same for the Track B list of §4.2. The
drop logic is switched off and `decisions.json` must show no dropped run. The set of runs then depends only on the
lists and on validation outcomes. Cost at laptop speed: about 4.0 h plus at most 1.3 h of second seeds for Track A.
Statistically nothing is added that was not pre-registered: the full list *is* the list; the drop order was a
fallback for lack of time; the six primaries are unchanged; a rung that is not run cannot enter A11, so dropping
could only remove candidates for a reason unrelated to their merit. It must be recorded before the first Track A
fit; none exists today (measurement F).

**Alternative (R2), if the cap is to be kept:** a budget in nominal units. Each run costs its "Cost" entry of AUDIT
§4.1 (laptop minutes, fixed in advance); the cap is 240 nominal minutes; the twelve exploratory rungs that are not
on the drop list run first; the number of second seeds they trigger then fixes, by arithmetic, which rungs of the
drop list run. This is independent of speed and of concurrency, but it drops runs on a machine that has the time.

PySR budgets are wall time by design (minutes per seed on 20 threads) and stay on the laptop, so they do not change.

### 5.6 Checks that show, on the H200, that side by side equals one after another

All in a scratch result directory, with the H200 reference arrays passed in explicitly; nothing reaches the
campaign registry.

| # | Check | Criterion |
|---|---|---|
| E0 | `local_grid()` and the generator test inside the configuration used for concurrent jobs | equal to R0 |
| E1 | K = 64 arrays of the eight networks: N processes at once (different networks, then all the same network), other slots busy with the fit probe | array SHA-256 equal to the sequential re-scoring R4, 8 of 8, no tolerance |
| E2 | Guard arrays (K = 1024, bulk) and mass monitors of the three guarded networks, same arrangement | equal hashes |
| E3 | One short fit per model class (Track B network; Track A component model) with `deterministic = True`: alone, beside N − 1 other jobs, alone again | hash over the weight tensors (not the `model.pt` file, §4 point 13), validation history without `seconds`, and registered array: all three runs identical. If the two solitary runs differ, the class has an operation without a deterministic kernel and E3 cannot decide for it |
| E4 | One short fit in the default mode, four times alone and four times under full load, all scored on the same points | mean of "loaded − alone" within 2 SE of zero, SE taken from the alone–alone pairs; rms of per-event differences within a factor 2. Weak with four repeats; E3 is the decisive check. The alone–alone pairs also give the run-to-run scatter that §3 could not measure |
| E5 | Locks and registry: N processes appending 200 rows each to a scratch registry; two processes on one run directory; N + 1 jobs on N slots | N × 200 valid lines; the second process stops with the holder's name; never more than N at once |
| E6 | After the first real batch: two registered runs re-evaluated alone from their `model.pt` | array hash equal to the row's `val_nll_sha256` |

If E1 or E2 fails: evaluations go back to the exclusive lock. The first thing to try is
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the campaign does not set today; if it is introduced, it is part of the
arithmetic tag and the reference arrays are recomputed under it.

**Option for the user (not recommended as default):** run the Track A fits with `deterministic = True`. Every fit
is then bit-reproducible and a gate can repeat any of them exactly. It costs speed (to be measured in the
calibration) and changes the training configuration T_A before its first use. It must not be applied to the Track B
seeds, whose configurations have to equal the laptop's.

## 6. Amendment wording — answered (final text for decisions (i), (ii), (iii))

Ready to paste into PLAN.md §13. Text in square brackets depends on the launch grid, which is read on the H200.
Point 6 is the reviewer's recommendation R1 and needs the user's approval; the alternative is footnote 4.

> **A6 (revised 25 Nov 2026 — change of machine).** Replaces row A6 for everything computed from this amendment on.
>
> **1. Machine of record.** The H200 produces every number that enters a table, a delta, a pre-registered rule or
> `LOCK.json`: the per-event arrays of the two benchmarks, all Track A fits, all Track B training still to run, the
> re-scoring of the eight networks trained on the laptop, the ensembles, the θ arms, every likelihood refit and
> every registered score of a PySR equation (controls C1 and C2 included), the zero-training scores of the geometry
> arm, all guards, and the single final test evaluation.
>
> **2. What stays on the laptop** and produces no number of record: the PySR searches (CPU; they see only exported
> targets and their own loss, and hand over equations as text), the equation audits (exact mathematics, no data),
> the geometry audit (training and validation data only; its gains are audit numbers on the laptop's sample points
> and decide only whether and in which form the arm is run), and the gates. A gate may recompute an H200 array on
> the laptop with the emulated H200 stream (`sfm.refstream` at the H200's launch grid) as a check; such an array is
> never a pairing partner.
>
> **3. Numbers of record** are the benchmark arrays computed on the H200 with the unmodified `common.test_nll`
> inside `sfm.metric.deterministic()`. The H200's launch grid is read with `sfm.refstream.local_grid()` before
> anything else is computed. [If it is 1056, the H200 draws the earlier study's fold points: validation 18.118083
> (MLP) and 18.120407 (formula), formula − MLP = +0.002324 ± 0.000294, to be met within 4e-6 on the float32 means
> and within 1e-4 per event against the laptop's emulated arrays. The stored test values 18.120842 / 18.123459 and
> +0.002616 ± 0.000292 are then the expected test numbers of record; they are confirmed by the first step of the
> final run and not assumed before it. If it is not 1056, the H200's own values are the numbers of record and both
> the stored and the laptop's values are other sample points.] The laptop's numbers — validation 18.117857 /
> 18.120061, test 18.120806 / 18.123333, formula − MLP +0.002204 ± 0.000293 (validation) and +0.002527 ± 0.000304
> (test), the Milestone 1 audit, the eight Track B rows of 24 Nov — are quoted as "laptop (RTX 4060, launch grid
> 144), other sample points". They are records: kept, never overwritten, never part of a delta with an H200 array.
>
> **4. Pairing rule.** A delta of record is formed only between two per-event arrays computed on the H200 on
> identical fold points (launch grid, seed, K, batch size, fold scheme and split all equal). Each pre-registered
> rule (keep, promotion, replication, equal NLL, 1-SE, convergence, guards, plateau) is evaluated once, on such
> arrays, and never again with another machine's numbers. Criteria written as laptop values take their paired
> form: "A0 at step 0 = 18.120061" becomes "A0 at step 0 equals the H200 formula array, within 1e-4 per event and
> 1e-6 on the mean"; "at least 3e-4 better than 18.12041, or than the A0 value" is a paired delta against the
> formula array of the same stream.
>
> **5. Provenance.** Weights are files identified by SHA-256 and labelled with the machine that trained them. The
> five-member ensemble consists of seeds 11 and 12 (fine-tuned on the laptop) and seeds 13–15 (fine-tuned on the
> H200 from the laptop's stage-1 networks); the two existing fine-tunes are not repeated. The control B0 is trained
> on the H200 for all three seeds; its seed scatter and the B0 mean use these three runs only.² The laptop's
> `b0_seed0` is kept, re-scored on the H200 and shown as a labelled record; it enters no scatter, mean or rule.
> This is fixed before any control has been scored on the H200, and no run is repeated in order to choose between
> outcomes. A comparator is trained on the machine of the variants paired with it. No interrupted fit is continued
> on another machine.
>
> **6. Budgets.** The run lists of `AUDIT.md` §4.1 and §4.2 are run in full on the H200. The wall-clock rule
> (4 GPU-hours, drop orders) was written for the laptop and is not applied: which runs exist depends only on the
> lists and on validation outcomes, not on machine speed or on how many jobs run side by side.⁴
>
> **7. PySR.** For each search the complete audited front (every seed, every pre-registered complexity tier) goes to
> the H200. No likelihood computed on the laptop selects, ranks or screens an equation.³ Refits, controls and
> scores are made on the H200; the audit is repeated on the refitted constants; the "refitted validation NLL" of
> the plateau rule is the H200's. The number of candidates entering the winner's-curse correction is the number of
> audited front members sent to the H200.
>
> **8. Test split.** The test cache is opened only on the H200 and only by `final_test.py`, after Gate 2c. There is
> no test access of any kind before that run; benchmark parity on the H200 is shown on validation only; the laptop
> does not open the test cache again.¹ `final_test.py` first recomputes the per-event test arrays of the two
> benchmarks and stops before any finalist is loaded unless (a) the SHA-256 of the test cache equals the value
> recorded by the Gate 0 run in `results/m0_parity/parity.json`, (b) the test event numbers equal
> `results/m0_parity/test_event_number.npy` in order, and (c) for both benchmarks the arrays agree with the earlier
> study's stored arrays (`test_v3_nll.npz`, keys `std:mlp_ft` and `std:new`) within 1e-4 for every one of the
> 14,956 events and within 4e-6 on the mean — the criteria of C1. If the validation parity on the H200 left any
> event above 5e-5, Gate 2c fixes the per-event clause before the run as: at most three events above 1e-4, none
> above 1e-3, rms at most 1e-5. [If the launch grid is not 1056, (c) is evaluated on the emulated reference stream,
> and the native arrays must in addition satisfy A3 against the emulated ones.] A stop is not the final run: only
> the two benchmarks have been evaluated, the access stays in the log, and a new start needs the cause, the user's
> go-ahead and an unchanged `LOCK.json`. Finalists are paired against the benchmark test arrays of the same run.
>
> **9. Test-access log.** `results/test_access.log` is one tracked file. Its two lines of 24 Nov 2026 (Gate 0
> parity, C1) were written on the laptop; the file with exactly these two lines has the SHA-256
> `effa9734743f51959806d7d8b87e2d262c10773a9be25f2104ceeca9557f0c60`. From this amendment on only the H200 adds
> lines, each with host, GPU and launch grid, and the laptop refuses `allow_test`. Until the final run both working
> copies hold the file unchanged: two laptop lines, no H200 line. After the final run the H200's file is committed
> and pulled, so there is one log again: the two laptop lines and one H200 line per start of `final_test.py`. Any
> other line, on either machine, is reported to the user before anything else is done.
>
> **10. Gates.**
> *Gate 1* also checks: the identity record of the H200 (GPU, launch grid, torch build, generator test) and the
> validation parity of the two benchmarks against their targets; that the laptop-trained checkpoints have the
> SHA-256 of their laptop registry rows and that their re-scored values meet the predicted table [grid 1056]; that
> every array behind a delta, a rule or a table was computed on the H200 against the reference arrays created
> there, and that no table reads a laptop array or a laptop row; that the control consists of three H200-trained
> seeds and the H200's `b0_seed0` is not the laptop file; that no fit was continued across machines and every
> comparator was trained on the machine of its variants; by recomputation on the laptop with the emulated H200
> stream, that both benchmarks and every kept or promoted candidate agree with the H200 arrays within 1e-4 per
> event and 1e-6 on the mean; and, if fits ran side by side, the equivalence checks and that the set of runs
> equals the pre-registered lists.
> *Gate 2* also checks that every refit and score of an equation is an H200 registry row and that the search
> record contains no likelihood.
> *Gate 2c* also checks: `LOCK.json` names the machine of record and holds, for every finalist, the hash of the
> checkpoint or equation, the training machine, the H200 validation numbers and the hashes of the benchmark arrays
> they were paired with; every selection step is traceable to H200 arrays; the scores of the geometry arm and of
> the PySR finalists are H200 rows; the test-access log is as in point 9; the test cache on the H200 has the
> recorded SHA-256 (or is still absent, if the test-derived files are transferred only after this gate);
> `final_test.py` refuses any machine other than the one in `LOCK.json`, refuses path overrides, and implements
> the stop rule of point 8 with the tolerances fixed by this gate.
> *Gate 3* also checks: the final run took place once, on the machine of record; its first step passed, with the
> measured differences on file; every test delta is paired against benchmark arrays of the same run; the log holds
> the two laptop lines and exactly one H200 line (or stop lines, each explained, with `LOCK.json` unchanged in
> between); laptop numbers appear in the report only under their label; the validation-to-test shrinkage uses H200
> validation numbers.
>
> ¹ Not chosen: a third Gate-0-type parity access on the H200 before the final run.
> ² Not chosen: a scatter from the laptop's `b0_seed0` and two H200 runs.
> ³ Not chosen: refitting or ranking candidates by likelihood on the laptop and re-scoring only the selected ones
> on the H200.
> ⁴ Alternative: a budget in nominal units (the cost column of `AUDIT.md` §4.1, cap 240 nominal minutes).

**Notes on the wording.**

- *Where exactly the line runs for PySR and the geometry arm.* May stay on the laptop: the search on an exported
  target (teacher and baseline maps are functions of weights; the search loss uses no fold points); the audits; the
  seed-null search C3, whose verdict compares search losses; the geometry audit's study of training hits and its
  validation gains. Must be made on the H200: every refit (training on hits), every validation score, the controls
  C1 and C2, the plateau clause that uses refitted NLLs, the arm's scores for the frozen MLP and the best closed
  form. Target files should be exported once and identified by SHA-256; exporting them on the H200 keeps the
  Fisher weights of trajectory targets on the stream of record.
- *Selection on one stream, reporting on another* (the rejected alternative ³). It would not be invalid. Fold
  points carry 1–6 % of the variance of a paired delta and are independent between streams, so re-scoring removes
  that part of the selection optimism; the event sample is the same on both machines, so the other 94–99 %
  remains and the winner's-curse correction is still needed at almost full size. It becomes a fault in three
  cases: a rule is evaluated on the laptop and reported as met with H200 numbers, or evaluated on both and the
  better one kept; the constants are refitted on the laptop (the equation is then a laptop-trained model); or the
  laptop screen is not counted in the number of candidates. Decision (iii) avoids all three by construction.
- *The plateau rule under decision (iii).* Its second clause needs refitted NLLs. To keep the search time
  independent of the H200, take the verdict once per seed after the scheduled time, from the last three
  hall-of-fame snapshots refitted on the H200, not chunk by chunk.
- *Equations as text.* Each equation should travel with a few hundred reference values from the search process, so
  that the H200 can show it rebuilt the same function (round trip within 1e-5).
- *The geometry arm's cache extension* (absolute shower origin, AUDIT §4.3) must be an additional file keyed by
  event number. Rebuilding the cache would replace the certified bytes.

## 7. Silent laptop assumptions — answered (complete list)

Each entry: where, what it assumes, the machine-independent form. The packaging worker's `MANIFEST.json`
(`machine_constants`, 29 entries) holds the engineering details of entries 1–9; they are not repeated in full.

**Numbers used as criteria**

| # | Where | Assumes | Machine-independent form |
|---|---|---|---|
| 1 | `scripts/run_m2a.py` line 43, check at 260–266; AUDIT §4.1 row A0; `results/m2a_dist/STATUS.md` | A0 at step 0 = 18.120061 to 1e-5, hard stop | A0 at step 0 equals the benchmark formula array of the same stream: 1e-4 per event, 1e-6 on the mean (M1 measured 1.3e-5 and −1.9e-8) |
| 2 | PLAN §5 ("≥ 3e-4 better than 18.12041"); AUDIT §4.1 ("≥ 3 better than the A0 value") | a threshold against a stored mean | paired delta against the formula array of the same stream |
| 3 | `scripts/run_m2a.py` 170–173, 305–317 (A0 drift rule) | the formula array at the live path is the laptop's | pair only with an array carrying the live stream tag |
| 4 | `scripts/audit_lib.py` line 48 (`RECORD`) | laptop validation values | label as laptop record; Milestone 1 is closed |
| 5 | PLAN §3 Gate 0 list, §8 "(18.1208)", "(18.1235)", §12 first bullet; README lines 9–23 | stored or laptop values as "the" numbers; README also calls C1 open | quote with the machine; on a grid-1056 H200 the stored values are the expected numbers of record |
| 6 | PLAN §13 A2 ("differs from itself by up to 2.9e-5") ; Gate 0 §2.5 | run-to-run floor of the laptop GPU | re-measure (R3); the criterion 1e-4 / 1e-6 stays |
| 7 | PLAN §13, registry docstring lines 16 and 20, `refstream.py` lines 10 and 22 | "this machine" | "the machine of record" plus stream tag |
| 8 | PLAN §14 and AUDIT §4: +22.0 ± 2.9, +4.07 ± 0.56, −0.5 ± 0.8, −5.9, 155, 933 and the "Expected" columns | laptop-stream numbers | expectations, not criteria; quote as laptop numbers. D6(c) is the only one that decided something, and it is safe (0.2 %) |
| 9 | Per-event tolerance 1e-4 wherever arrays of two machines are compared | equal arithmetic | the fallback of §1.3 (three events, 1e-3, rms 1e-5, means 1e-6) |

**Launch grid and stream**

| # | Where | Assumes | Machine-independent form |
|---|---|---|---|
| 10 | `src/sfm/refstream.py` line 51; `tests/test_refstream.py` 50–52 | `LOCAL_GRID = 144` | use `local_grid()`; assert only the rule `grid_for` |
| 11 | `tests/test_refstream.py` 72–80 | the local stream differs from grid 1056 | compare with a grid different from the local one |
| 12 | `tests/test_refstream.py` line 113 | a Gate 0 array of the laptop at its live path | read it from the record folder, or skip |
| 13 | PLAN §13 "optional secondary column on the reference stream" | reference stream ≠ native stream | on a grid-1056 H200 the native stream is the reference stream; the column needs no emulation |
| 14 | `common.py` line 64, `refstream.local_grid(device=0)` | one GPU, index 0; silent CPU fallback | pin the device; refuse to run campaign steps on the CPU |

**Time, memory, one job at a time**

| # | Where | Assumes | Machine-independent form |
|---|---|---|---|
| 15 | `scripts/run_m2a.py` 44, 358–405; AUDIT §4.0 and both "Budget" paragraphs; PLAN §15 "about 4 / 2 GPU-hours" | wall time of the laptop decides which runs exist | §5.5 |
| 16 | Cost columns of AUDIT §4.1 and §4.2; `_est_seconds` defaults; PLAN §3 "timing probe calibrates the GPU budget" | laptop speed | re-measure in the calibration; planning only |
| 17 | `min_free_mb` 2500 and 3000 (`run_m2a_lib.py` 121, `run_m2b_lib.py` 110); `tests/test_families_components.py` 492–499 | an 8 GB GPU; a cost bound for the RTX 4060 | per-class footprints measured on the H200 (§5.2) |
| 18 | PLAN §2 row 12, §10, §15; `sfm/gpulock.py`; README lines 44, 111, 115; `tests/test_train.py` 417–433 | one GPU job at a time; the lock test asserts exclusivity | slots with default 1; the test holds for one slot |
| 19 | `checkpoint_sha256`, `fit_done.json` `model_sha256` | a file hash identifies weights | add a hash of the weight tensors (§4 point 13) |
| 20 | `tests/test_train.py` 309–319 | default-mode resume within 1e-4, measured on the laptop GPU | a tolerance, not an identity; re-measure if it fails |
| 21 | PLAN §7 budgets in minutes on 20 threads | laptop CPU | unchanged, the searches do not move; convert to iterations if they ever do |

**Test split, logs, records**

| # | Where | Assumes | Machine-independent form |
|---|---|---|---|
| 22 | PLAN §8 Gate 2c, §13 C1 ("exactly these two" lines); `sfm/data.py` 253–258 | one machine; log lines without host | §6 point 9 |
| 23 | PLAN §8, §12 ("`final_test.py` refusing a second run") | one working copy; no stop-before-finalists case | §2.3 |
| 24 | PLAN §10 ("opened only by the parity script and by `final_test.py`"), §12 first bullet, README line 64 | the parity script may run | on the H200 `parity.py` is never run; parity is R1 plus the first step of the final run |
| 25 | README lines 62–65 ("Running") | a fresh laptop | on the H200 `check_protected.py record`, `build_cache.py`, `parity.py` and `mc_noise_val.py` each overwrite a certified file or a tracked laptop record, or open test |
| 26 | `sfm/paths.py` overrides | harmless | with `SFM_RESULTS_DIR` set, the test-access log and the registry are written outside the tracked files |
| 27 | `tests/test_registry.py` 187–188 | cached benchmark guard arrays are reused whenever the spec matches | reuse only under an equal stream tag; add the refusal case |
| 28 | `tests/test_registry.py` 195–211 | `results/m0_parity/val_event_number.npy` exists (git-ignored); the test fails instead of skipping without it | copy the file (it is in the package) or skip |
| 29 | `scripts/run_m2b_report.py` 50–51, 111–118 | `b0_seed{0,1,2}` are the control | must be the three H200-trained runs (decision (ii)) |
| 30 | `train.fit` 407–418 with tracked `fit_done.json` | a finished run at this path is this machine's | machine tag in the marker; see §3 (ii) |

**Read through and found free of laptop numbers:** `src/sfm/models/dist_factory.py`. Its templates are computed on
the GPU in float32 (`compute_templates`), so they agree between machines only to rounding: generate them once on
the H200 and identify them by hash. Its templates phase needs the copied `abar_front.pt`.
`tests/test_train.py` and `tests/test_registry.py` contain no stream-dependent constant; their CUDA tests compare
runs on the same machine.

**A gap in the decision as first stated (hand-over), now closed by decision (iii):** PySR refits and the geometry
arm produce scores of finalist candidates; they are made on the H200.

## 8. Check of the packaging classification (`transfer/STATUS.md`, `MANIFEST.json`) — answered

**The classification is correct.** Checked file by file against the code:

| Files | Classified as | Verdict |
|---|---|---|
| Parquet file, both caches, cache metadata | not stream-dependent | correct |
| `mlp_std.pt`, `new3_std_n3.pt`, `abar.pt` | not stream-dependent (weights, template) | correct |
| `test_v3.json` | not stream-dependent | correct for what is read: `shapelab.final_kwargs` takes only `degrees` and `variant_kwargs`, and `legacy.load_component_formula` discards the rest. The file also holds the earlier study's test means, which no campaign code reads |
| `mlp_std.json`, `new3_std.json`, `test_v3_nll.npz` | reference-machine values, grid 1056 | correct |
| `val_event_number.npy`, `test_event_number.npy` | not stream-dependent | correct; event numbers carry no shower content |
| `abar_front.pt`, `abar_front_back.pt` | not stream-dependent; cannot be regenerated without the parity script | correct; `parity.py` lines 502–505 build them from the training set only |
| The eight `model.pt` | not stream-dependent | correct as weights. The histories inside are laptop-stream numbers and part of how the networks were trained |
| `m0_parity/*_nll.npy`, `registry_refs/*.npy`, the networks' `val_nll*.npy` and `val_bulk_zb1221.npy` | laptop stream, records, quarantine | correct |
| `*_ref1056.npy`, `gate0_checks/val_grid1056_*` | emulated grid 1056, records | correct; they are comparison targets, not pairing partners (§4 point 10) |
| `val_mass_monitors.npz` | not stream-dependent | correct (no fold points; measurement E gives the cross-arithmetic size) |
| Tracked `done.json`, `fit_done.json`, `summary.json`, `registry.jsonl`, `parity.json`, `mc_noise_val.json` | laptop records with hazards | correct; they also contain laptop-stream numbers (`val_mean`, selected values) |
| Templates of Track A (regenerated) | identical up to rounding | correct |

**"Copy both caches and verify by hash; do not rebuild" is the right call statistically.** Gate 0 and C1 are
statements about these bytes; a rebuild on another CPU can differ in the last bit of a float32, which would turn
every exact identity into an approximate one and change the validation fingerprint; a rebuild also rewrites the
test file and a tracked record. One addition: verify against the tracked `parity.json` as well as against the
copied metadata file (measurement F).

**Can a copied file let test information reach a decision?** No path was found.
- Seven files are derived from the test split: `cache/pion_s1_test.pt`, `test_v3_nll.npz`,
  `test_event_number.npy`, and the four laptop test arrays `m0_parity/{mlp,formula}_test_nll{,_ref1056}.npy`.
  The traced moved steps open none of them. Only `parity.py` and the future `final_test.py` do.
- The per-event test arrays describe the two benchmarks only. A candidate cannot be scored against them without
  opening the test cache, and the registry rejects an array of 14,956 entries (`registry.py` line 412).
- `gate0_checks/events_table.npz` (per-event metadata of all events, test included) is excluded. Keep it excluded.
- **Refinement for the user (§9, point 3):** these seven files are not needed before the final run. If they are
  copied only after Gate 2c, their absence is physical proof that the H200 made no test access; the log alone
  records only accesses through `load_sets`.

**Consequences of the decisions for the package:** group c must contain a retrained `b0_seed0`; the laptop's
`b0_seed0` weights are then a record (re-scored, never trained from); `fit_done.json` of that run must not be at
the path the H200 run uses (§3).

## 9. Decisions that remain with the user

| # | Decision | Options and trade-off | Reviewer's recommendation |
|---|---|---|---|
| 1 | Fits side by side on the H200 | Approve §5.2 as a conditional proposal, or keep one job at a time. Gain: unknown until measured, at most a few hours of GPU on about 7–8 laptop-hours of moved work. Cost: slot and run locks, a driver that schedules, checks E0–E6 (about an hour) | approve conditionally; drop it if two jobs give less than 1.15 times one |
| 2 | Multi-process service of the GPU | Allow it: possibly several times the throughput; another configuration to verify (E0, E1) and all clients fail together. Refuse it: simpler, gain probably small | allow it only if the calibration without it gives N ≤ 2 |
| 3 | Budget rule | R1: full lists, no drops (simplest, independent of speed, up to 1.3 laptop-hours more). R2: nominal-cost cap (keeps the cap, drops runs although time is available) | R1, recorded before the first Track A fit |
| 4 | When the seven test-derived files travel | Now, unopened (one transfer, protection by log and code). After Gate 2c (absence proves no access; a second transfer) | after Gate 2c |
| 5 | Per-event clause of the stop rule | Strict C1 criterion always (a single rounding-sensitive event could stop the run and cost a logged access). Or the conditional clause of §2.3, fixed at Gate 2c from the validation measurement | the conditional clause |
| 6 | If the H200's grid is not 1056 | Accept its numbers as a third set of sample points (stored arrays never pair; the secondary column needs emulation). Or look for a configuration with grid 1056 (no MIG slice, no thread limit) | first find out why; accept the third stream only if no full-device configuration exists |
| 7 | Bit-reproducible Track A fits | `deterministic = True` for Track A: exact repetition by a gate, slower, changes T_A before first use. Or leave as pre-registered | leave as is unless the measured slow-down is under 20 % |
| 8 | Cache extension for the geometry arm (already open in AUDIT §4.3) | an additional file keyed by event number, built on the laptop CPU and copied by hash; never a rebuild of the cache | additional file |

Not decisions, but to be fixed in writing before the H200 produces a control score: the names under which the
three H200 controls and the laptop record are registered (§3).

## 10. What was verified, what was inferred, what only the H200 can settle

**Verified on the laptop (code read, or measured in the two sessions)**
- Evaluations inside `deterministic()` are bit-identical under 2–4 concurrent processes, guards included.
- On the laptop, concurrency gives no throughput; memory adds linearly; a waiting process holds only its context.
- Emulated grid 144 reproduces 18 stored laptop arrays bit for bit, guards included; emulated grid 1056
  reproduces the earlier study's validation means exactly in float32.
- Predicted H200 values for K = 64, K = 1024 and bulk (grid 1056), by two code paths for K = 64.
- Size of a mis-pairing: 2.3 and 3.5 at K = 64, 0.4 at K = 1024, 6 on the bulk guard.
- CPU arithmetic on identical points: one event at 1.4e-4, means within 5e-8.
- The cache hashes are tracked in `parity.json`; the train/val cache has that hash now; the test-access log has two
  lines; no Track A fit exists.
- The budget rule is the only place where time decides a result; `model.pt` file hashes contain wall time.
- `abar_front.pt` comes from training events; `test_v3.json` is read for the formula's structure only.
- Every line reference in §3, §4, §5.5 and §7.

**Inferred**
- The H200's launch grid is 1056 (same die as an H100) and its memory about 141 GB.
- Its GPU kernels round like the laptop's, so per-event differences stay near 3e-5.
- Why concurrency gives nothing on the laptop (the driver alternates between processes), and that only the
  multi-process service can change this on a large GPU.
- That `CUBLAS_WORKSPACE_CONFIG` would not change the bits of evaluations.
- That selection on one stream and reporting on another removes only the fold-point share of the optimism.

**Only a measurement on the H200 can settle**
- The launch grid, and whether the emulator equals its `torch.rand` (R0).
- Whether its arrays meet the targets, and the largest per-event difference against the laptop's emulated arrays.
- The run-to-run floor outside `deterministic()`, and how far two default-mode fits of one configuration are apart.
- The number of slots, the footprints per job class, the effect of the multi-process service on speed and on the
  reported grid.
- Whether `flock` is honoured on its file system; whether a deterministic fit is bit-identical under load (E3).
- The test parity itself: by decision (i), only in the first step of the final run.
