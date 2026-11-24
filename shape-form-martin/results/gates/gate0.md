# Gate 0 — workspace, leakage, cache fidelity and benchmark parity

Verifier: independent gate agent (fresh context), 24 Nov 2026. Branch `shape-form-martin`, HEAD `8c9771e`.
Machine: RTX 4060 Laptop GPU (24 SMs), torch 2.14.0+cu130, numpy 2.5.3.

## Verdict: PASS WITH AMENDMENTS

Milestone 1 may start. Three things have to be read with that verdict:

1. **The plan's parity criteria fail as written.** With `common.test_nll` on this machine, three of the four means are
   further than 5e-5 from the references, no per-event test array is within 1e-4, and the fallback correlation is
   0.99977 / 0.99975, below the required 0.9999. These are recorded as FAIL below, not softened.
2. **The cause is established, and the stored stream can be reproduced.** The fold points of `common.test_nll` come from
   `torch.rand` on CUDA, whose stream for large tensors depends on the number of streaming multiprocessors of the GPU.
   A CPU emulation of that kernel reproduces this GPU bit for bit, and with the launch grid of the reference machine
   (1056 blocks = 132 SMs × 8, an H100-class GPU) it reproduces **all 13 validation means stored by the earlier study
   to within one float32 ulp (≤ 1.9e-6) and all 10 stored paired differences to ≤ 7.2e-8**. Same checkpoints, same
   data, same split, same code: shown exactly, on validation.
3. **One confirmation is open.** I may not open the test split, so the per-event *test* arrays have not been recomputed
   on the reference stream. On test the evidence is: the event sequence is identical (exact fingerprint), and the
   cross-machine differences are statistically indistinguishable from a change of fold points. The exact test-side
   check (expected max |Δ| ≈ 3e-5 per event) needs one more logged test access by the parity script. It does not block
   Milestone 1; it must be closed before the test split is used for any finalist (item C1).

No code defect was found that blocks Milestone 1. The plan's assumption behind the bulk guard (`z_cut` below every
back face) is false for about 1 % of the events and its treatment is amended (A4).

---

## 1. Criteria table

Status: PASS, FAIL (as written in the plan), AMENDED (criterion replaced, see §3 for the justification), OPEN.

### 1.1 Split integrity and leakage

| # | Criterion | Measured | Threshold | Status |
|---|---|---|---|---|
| 1 | Event-number sets pairwise disjoint | overlaps train∩val 0, train∩test 0, val∩test 0; union = all 99,689 events that survive the cuts | 0 | PASS |
| 2 | Counts | 69,783 / 14,950 / 14,956 (re-derived from the Parquet file with independent code; identical event numbers *and order* to the cache extras and to `test_event_number.npy`) | 69,783 / 14,950 / 14,956 | PASS |
| 3 | 70/15/15 per momentum bin | all 20 bins follow `round(0.70 n)`, `round(0.85 n)` exactly; fractions train 0.69990–0.70008, val 0.14986–0.15015, test 0.14992–0.15009 | within rounding | PASS |
| 4 | Split code is the `common.load_data` loop | AST of the loop in `sfm.data.split_indices` identical to `common.load_data` lines 224–229; rng `default_rng(common.SEED)`; my re-derivation exec's the `load_data` source itself | verbatim | PASS |
| 5 | Every hit in exactly one event / split | storage = contiguous `off` ranges per event, `Showers.subset` gathers them, split index sets are a partition; hits 19,423,809 + 4,168,192 + 4,186,037 = 27,778,038 = rows after cuts; per-event hit counts equal the cache `off` (train, val) | partition | PASS |
| 6 | No model input / constant from val or test: momentum centring | `lnp0, lnp_s` recomputed from training events = 0.29119813, 2.70453405 = checkpoint buffers (float32-equal; same in all 12 ladder checkpoints). With all events they would be 0.19203, 2.80370 | train only | PASS |
| 7 | … `v0, vs` | recomputed −0.32585955, 3.31461334 = checkpoint buffers | equal | PASS |
| 8 | … acceptance `Ā` | recomputed from training events only: bit-identical to `abar.pt` (train+val would differ by up to 1.3e-3) | train only | PASS |
| 9 | … detector geometry | `common.load_data` takes it from the hit positions of all events; from training events only it is identical (`layer 20.5, cell 48.59999999999991, z_back 4882.15`), also from val-only and test-only rows | no dependence | PASS |
| 10 | MLP has no data-derived constants | input `ln p / 3`; grid-feature buffer in `mlp_std.pt` identical to the locally built one (max abs diff 0) | — | PASS |

### 1.2 Cache fidelity

| # | Criterion | Measured | Threshold | Status |
|---|---|---|---|---|
| 11 | Lean builder = `common.Showers.from_dataframe` on a subset the worker did not use | six classes `event_number % 6 == k` (complete events spread over all 47 row groups, all 99,689 events): `off, xyz, w, box`, every `ev` tensor, event numbers, θ, φ bit-identical incl. dtypes | bit-identical | PASS |
| 12 | Cache **file on disk** = reference | all 69,783 train and 14,950 val events of `pion_s1_trainval.pt` bit-identical to the pandas reference (values and dtypes: xyz/w/p/zb/R/evis/lnp float32, off/bin int64); file SHA-256 equals the meta entry | bit-identical | PASS |
| 13 | Geometry | `common.detector_geometry` on a pandas frame of the full x, y, z columns = meta = cache file; equals the line printed by the reference machine's own cache build (`logs/build_cache_pion_s1.log`) | equal | PASS |
| 14 | "Cache equals one `common.load_data` run" | done chunk-wise on complete events (the reference acts row-wise / per event with a fixed geometry); peak RSS 4.3 GB | — | PASS (equivalent form) |

The test cache file was not opened. Its content is covered indirectly: same `_pack` code path as train/val, test event
numbers and order identical to my re-derivation, test hit total equal to the Parquet count (4,186,037), and the lean
builder equals the reference on every test event when run on the Parquet rows.

### 1.3 Benchmark parity

| # | Criterion | Measured | Threshold | Status |
|---|---|---|---|---|
| 15 | Means within 5e-5 of the references (this machine, `common.test_nll`) | MLP test 18.120806 (Δ −3.6e-5), val 18.117857 (Δ −2.27e-4); formula test 18.123333 (Δ −1.26e-4), val 18.120061 (Δ −3.46e-4) | 5e-5 | **FAIL** (1 of 4) → AMENDED (A1) |
| 16 | Per-event test arrays within 1e-4 of `test_v3_nll.npz` | max \|Δ\| 0.487 (MLP), 0.691 (formula); 0.6 % / 0.5 % of events within 1e-4 | 1e-4 | **FAIL** → AMENDED (A1), exact form OPEN (C1) |
| 17 | Fallback: correlation > 0.9999 | 0.999770 (MLP), 0.999749 (formula). Two seeds on *this* machine give 0.99971–0.99978, so the threshold cannot be met by any two streams | 0.9999 | **FAIL** as written → AMENDED (A3): r ≥ 0.9995 PASS |
| 18 | Fallback: means within Monte-Carlo noise | local − reference: test −3.6e-5 ± 2.6e-4 (z −0.14), −1.25e-4 ± 2.7e-4 (z −0.46); val z −0.83, −1.19 | \|z\| < 3 | PASS |
| 19 | **A1a** exact parity on the reference stream, validation | 13 stored means (mlp_std, new3_std n=1…6, base_std n=1…6): max \|Δ\| 1.9e-6 (9 identical in float32, 4 off by one ulp); 10 stored paired differences: max \|Δ\| 7.2e-8, their SEs: 6.3e-10; ladder choices (n\*=6, chosen 3 and 5) reproduced. Control with the local stream: 3.5e-4, 3.7e-5, 2.3e-6 | means 4e-6, paired 1e-6 | PASS |
| 20 | **A1b** stream-independent identities | `formula_values.json["std"]` (40 rows written on the reference machine): max rel. diff 0; `audit_v3.json` identical (sum−1 = 2.38e-7, 101 effective parameters); `Ā`, centring, `v0/vs`, `logA` buffer, grid buffers identical; counts in `repro.json` equal | exact | PASS |
| 21 | **A1c** test event sequence identical | the 40 per-bin test means stored in `test_v3.json` (std and slab) recomputed from the stored arrays with *our* bin labels in *our* test order: max abs diff 0 (identical floats). One swapped neighbouring pair changes them by 1.4e-6–4e-5, a shift by one event by 0.05 | exact | PASS |
| 22 | **A1d** test: statistical parity under the local stream | see §2.4: KS p 0.21 / 0.29 (\|d\|), 0.32 / 0.74 (signed), 0.95 (paired D); rms ratio test/val 0.96 / 0.93; lag-0 r 0.99977 / 0.99975, other lags \|r\| ≤ 0.010; no \|d\| > 0.69 (validation null: up to 1.37) | KS p > 0.01, rms within ±15 %, \|z\| < 3, r ≥ 0.9995 | PASS |
| 23 | **A1e** exact per-event test parity on the reference stream | not evaluated (needs a test access) | max \|Δ\| ≤ 1e-4, \|Δ mean\| ≤ 4e-6 | **OPEN (C1)** |
| 24 | Headline numbers from the unmodified `common.test_nll` | `parity.py` calls `common.test_nll(model, set)`; module = `shape-form-martin/common.py`, byte-identical to `HEAD:shape-form-original/common.py`; its metric functions are AST-identical to the notebook the reference run exec'd; my own `common.test_nll` calls reproduce the saved validation arrays to 2.7e-5 (run-to-run floor) | unmodified | PASS |

### 1.4 Metric and statistics code

| # | Criterion | Measured | Threshold | Status |
|---|---|---|---|---|
| 25 | `sfm.metric` at defaults equals `common.test_nll` exactly on validation | `common.test_nll` itself is not run-to-run reproducible on CUDA: 8 runs differ by up to 2.7e-5 (MLP) / 2.9e-5 (formula) per event, 52 % of events differ between two runs, mean moves ≤ 8e-8. Inside `deterministic()`: runs bit-identical, `sfm.metric.test_nll` and `test_nll_multi` bit-identical to `common` for both benchmarks (K=64 and K=16). Default mode: max \|Δ\| 2.5e-5, \|Δ mean\| ≤ 7e-8 | exact | AMENDED (A2): PASS |
| 26 | `pytest shape-form-martin/tests` | 71 passed (incl. the real-data validation test: bit-identical under `deterministic()` for 3 models) | all pass | PASS |
| 27 | Paired SE and 1-SE rule vs `common.select_architecture` | `paired` = `shapelab.paired` (identical floats); bootstrap SE / analytic SE = 0.999 (plain) and 0.998 (stratified); `one_se_select` = literal transcription of the `common` rule in 3000/3000 random cases (best candidate differs in 3 float32-tie cases, as documented); stored ladder choices reproduced on both streams | same rule | PASS |
| 28 | Stratified fold unbiased | derivation correct (equal-volume strata, one uniform point each); on real hits (300 seeds, 34,749 hits): mean z of ĝ_strat − ĝ_iid = 0.004 (K=16), −0.008 (K=64); variance ratio 0.55 / 0.39; NLL lower by 0.0066 / 0.0017 nat (smaller Jensen bias, as claimed) | unbiased | PASS |
| 29 | Bulk guard premise: `z_cut` = 1150 mm below every `zb` | false: `zb` < 1150 mm for 387 training (0.55 %) and 89 validation events (0.60 %), down to 10.27 mm; `zb` < 1221 mm for 678 (0.97 %) / 166 (1.11 %). None negative | all `zb` > `z_cut` | **FAIL** → AMENDED (A4) |

### 1.5 Units, ranges, low momentum, acceptance, repository

| # | Criterion | Measured | Threshold | Status |
|---|---|---|---|---|
| 30 | Units | p GeV (range 0.0895–20.0, bin = floor(p) for 100 %); θ rad (3.00°–36.00°), φ ∈ (−π, π); lengths mm (60 layers, pitch 20.5, z 3662.4…4871.9, back face 4882.15, cell 48.6; box (48.6, 48.6, 20.5)); time ns (earliest hit 12.26 ns = light travel time to 3662 mm; min of t − \|x\|/c = −0.06 ns); m_π = 0.139570 GeV (buffer 0.13956999778747559) | GeV, rad, mm, ns | PASS |
| 31 | Geometry / conventions | `local = R(θ, φ)(global − origin)` holds to 4e-5 mm on 400 events, origin z = `z − z_trans`; R orthonormal to 1e-7, det 1, R₂₂ = cos θ; `zb cos θ` = (59.5 − k)·20.5 mm with k the first hit layer; weights sum to 1 within 5e-8 | consistent | PASS |
| 32 | 1001 hits per event a cap? | No. Exactly two events have 1001 hits after cuts (raw 1860 → time cut 1006 → 1001, and raw 1835 → 1033 → 1001); raw rows per event go up to 1893 with no pile-up; next values 964, 925, 922 | — | PASS (not a cap) |
| 33 | `Ā` recomputed (train, front+back, nsub=4) vs `abar.pt` | max abs diff 0 (bit-identical). Front-only − stored: max 0.821, mean abs 0.0737, never negative, 2,620 of 4,096 cells differ by > 1e-3, from z′ ≈ 393 mm on. nsub=2 differs (0.146). Closed form vs brute-force φ integration: 6e-5 | equal | PASS |
| 34 | Low-momentum boundary (after cuts) | train p_min 0.089516 GeV (T 0.026240, x −1.0000), 2 events < 0.1 GeV; val 0.174682 (T 0.084022, x −0.6489), 0 events; test 0.073411 (T 0.018129, x −1.1116), 2 events. Raw file: 65 events < 0.1 GeV, 61 lose all hits in the cuts; the 4 survivors have 1–2 hits | report | PASS (reported) |
| 35 | No floor / clamp on T or x | confirmed by reading: `kinetic` = p²/(√(p²+m²)+m), `lnT = log(kinetic)`, `x = (lnT − v0)/vs` unclamped (`particle_shape.py` 28–30, 106–111; `proton_shape.py` 82–89). Only `range_mm` has a 1 mm softplus floor; MLP input `ln p / 3` unbounded | none | PASS (as the plan stated) |
| 36 | Benchmark maps finite and normalised at the boundary | lowest val, lowest train, lowest test (0.0734 GeV), 0.05, 0.02, 0.005, 0.0005 GeV: all finite, \|Σ − 1\| ≤ 2e-7 for both models | finite, normalised | PASS |
| 37 | Protected directories | `check_protected.py verify`: PASS (1997 hashed files, 2 stat-only data files, no new bytecode); tracked files under both directories equal HEAD except the two moved ones | unchanged | PASS |
| 38 | `results/test_access.log` | exactly one line (`2026-11-24T16:22:17+00:00  parity.py  gate0 parity of existing benchmarks`); unchanged by this gate | 1 line | PASS |
| 39 | Git state | staged: only the two renames (R100); nothing staged under `shape-form-proton/`; no data file tracked (`*.csv`, `*.parquet` ignored); unstaged: `pyproject.toml`, `uv.lock` (pytest, nbclient), `summary.md` (three path references), the notebook's settings cell | as planned | PASS |
| 40 | Timing probe | 0.035 s/step (K=16), 0.054 s/step (K=64), batch 256, peak 1.85 GiB | present | PASS |

---

## 2. The central question: same models on the same events, differing only by the fold points?

**Yes.** The evidence, from strongest to weakest.

### 2.1 Why a seeded CUDA generator gives another stream on another GPU

`torch.rand(..., device="cuda", generator=g)` runs a kernel with `grid = min(SMs × (maxThreadsPerSM / 256), ⌈n/256⌉)`
blocks of 256 threads. Thread `idx` owns the Philox4x32-10 subsequence `idx`; its j-th block of four numbers goes to
elements `idx + T(4j + i)`, T = 256·grid, and the generator offset then advances by `(⌊(n−1)/(4T)⌋ + 1)·4`. For a tensor
with more than T elements the mapping element → (thread, counter) therefore depends on the SM count of the GPU. One
batch of `common.test_nll` draws about 7 million numbers, far above T on any GPU. Tensors with n ≤ T are identical on
all GPUs, which is why small tests never show the effect.

- `gate0_checks/philox_emul.c` + `emul.py` re-implement the kernel on the CPU. `c02_validate_emulator.py`: with
  grid 144 (24 SMs × 6) the emulation equals `torch.rand` on this GPU bit for bit — ten consecutive calls of different
  shapes on one generator (so the offset rule is right as well) and four seeds. Grids 96 and 192 fail.
- End to end: `common.event_nll` fed with the emulated grid-144 points returns `common.test_nll` to ≤ 3.2e-5 per
  event for the four models tried, the run-to-run floor.
- For another grid only the first T elements of each call coincide (0.5 % of a batch).

### 2.2 The reference machine's stream, identified and confirmed (validation only)

`c03_find_reference_grid.py` scanned 46 launch grids of common GPUs for the stored validation mean of `mlp_std.pt`
(18.118083953857422). Exactly one matches at float32 resolution: **grid 1056**. The next closest are 9.5e-6 (grid 64)
and 1.1e-5 away. The 46 streams have an sd of 1.90e-4, so a single accidental match at that resolution somewhere in
the scan was not improbable (of the order of 10 %) and needed confirmation. `c04_confirm_reference_stream.py` then
evaluated all 13 models whose validation numbers the original run stored:

| | emulated grid 1056 | local stream (control) |
|---|---|---|
| max \|mean − stored\|, 13 models | 1.9e-6 (one float32 ulp) | 3.5e-4 |
| max \|paired Δ vs best rung − stored\|, 10 rungs | 7.2e-8 | 3.7e-5 |
| max \|SE − stored\| | 6.3e-10 | 2.3e-6 |

The paired quantities are stored in float64, so they carry seven to eight significant digits each; the agreement is
500 times better than with any other stream. The GPU model is not recorded in the logs; 1056 = 132 × 8 is what an
H100 would give.

This single result excludes, for validation, every alternative on the list: other data cuts, another event order,
other hit weights, other model weights, another geometry, another acceptance, another K, another batch order, a
difference between the CSV the reference run read and the Parquet copy used here. Any of them would move the means by
far more than 2e-6.

### 2.3 Checks that do not depend on the random stream

- `Ā` recomputed from the cached training events is bit-identical to `abar.pt`.
- `lnp0, lnp_s` (training min/max of ln p) and `v0, vs` equal the checkpoint buffers in float32.
- `formula_values.json` (kinetic energies, component weights, in-detector shares with `Ā`, 30 physical parameters at
  six momenta, written on the reference machine): all 40 rows reproduced with relative difference 0.
- `audit_v3.json`: normalisation defect 2.384185791015625e-07 and 101 effective parameters, both reproduced exactly.
- Geometry line and split counts printed by the reference machine's cache build equal ours.
- **Test split**: the 40 per-momentum-bin test means in `test_v3.json` were computed on the reference machine with its
  own per-event bin labels. Recomputed from the stored arrays with our labels in our order they are identical floats.
  The reference test set is therefore the same event sequence as ours, position by position (to the resolution of
  the bin label, which a single transposition would already break).

### 2.4 Test split under the local stream: statistics on the saved arrays

Null distribution: validation, local stream minus emulated reference stream, same models (the exact analogue of "this
machine minus the reference machine"); cross-checked against eight seeds on this machine, which I re-ran.

| quantity | test, local − reference | validation analogue | seed-to-seed on this machine (28 pairs) |
|---|---|---|---|
| MLP: rms of d_e | 0.0318 ± 0.0008 | 0.0332 ± 0.0013 | 0.0309 – 0.0348 |
| MLP: median \|d_e\|, 99 % quantile, max | 0.0111, 0.129, 0.49 | 0.0112, 0.123, 0.98 | 0.0107–0.0112, 0.117–0.139, 0.48–1.31 |
| MLP: Pearson r | 0.999770 | 0.999748 | 0.999723 – 0.999782 |
| MLP: mean d ± SE | −3.6e-5 ± 2.6e-4 | −2.3e-4 ± 2.7e-4 | sd of a mean across streams 1.9e-4 |
| formula: rms of d_e | 0.0334 ± 0.0010 | 0.0358 ± 0.0021 | 0.0323 – 0.0355 |
| formula: median, 99 %, max | 0.0112, 0.134, 0.69 | 0.0114, 0.126, 1.37 | 0.0109–0.0114, 0.124–0.140, 0.53–0.85 |
| formula: Pearson r | 0.999749 | 0.999708 | 0.999712 – 0.999763 |
| formula: mean d ± SE | −1.25e-4 ± 2.7e-4 | −3.5e-4 ± 2.9e-4 | 1.9e-4 |
| paired D_e = Δ(formula − MLP): mean ± SE | −8.9e-5 ± 1.2e-4 (z −0.74) | −1.2e-4 ± 1.0e-4 | sd of the difference across seeds 7.6e-5 |
| paired D_e: rms; median \|D\| | 0.0147 ± 0.0013; 0.00194 | 0.0128 ± 0.0007; 0.00193 | rms 0.0116 – 0.0134 |
| corr(d_formula, d_MLP) | 0.899 | 0.934 | 0.930 |

- Two-sample KS, test vs validation analogue: \|d\| p = 0.21 (MLP), 0.29 (formula); signed d p = 0.32, 0.74;
  \|D\| p = 0.95.
- By momentum bin (20 bins): median \|d\| ratio test/validation 0.92–1.10 (MLP), 0.89–1.09 (formula); χ² of the bin
  means 16.6 and 17.2 for 20 degrees of freedom.
- By hit multiplicity (8 classes from 1–5 to > 450 hits): rms falls from 0.15–0.17 to 0.013 in the same way on both
  splits. The largest \|D\| on both splits are 1–2-hit events; without the 20 largest the paired rms is 0.0108 (test)
  and 0.0102 (validation).
- Alignment: r = 0.99977 at lag 0, \|r\| ≤ 0.010 at lags ±1…±3 and for a random permutation; minimum r over 30
  consecutive blocks 0.9996; a shift by one event would give an rms of 2.1.
- Model identification: the local MLP array is closest to `std:mlp_ft` (rms 0.0318; next 0.0476), the local formula
  array to `std:new` (0.0334; `std:new_n2` 0.0382); the local pair difference matches `std:new − std:mlp_ft`
  (0.0147) and not `std:new_n2 − std:mlp_ft` (0.0242).

Power of these test-side statistics, stated plainly: they exclude a wrong event order, a wrong rung, and any
difference that changes the distribution of per-event values; from the means alone they cannot exclude a common
shift below about 5e-4 for a single model (2 SE) or 2.4e-4 for the formula − MLP difference. The exact statement on
test is item C1.

### 2.5 What can and cannot be claimed

- **Can be claimed:** the two benchmark checkpoints, the acceptance template, the data, the split and the metric code
  are those of the earlier study (exact on validation, exact stream-independent identities, exact test event
  sequence). The numbers this machine produces are the same estimator on other fold points.
- **Numbers of record for this campaign** (unmodified `common.test_nll`, this machine):

  | | stored by the earlier study | this machine | emulated reference stream |
  |---|---|---|---|
  | MLP validation | 18.118084 | **18.117857** | 18.118082 |
  | formula validation | 18.120407 | **18.120061** | 18.120407 |
  | MLP test | 18.120842 | **18.120806** | not evaluated |
  | formula test | 18.123459 | **18.123333** | not evaluated |
  | formula − MLP, validation | (0.002323 from the means) | **0.002204 ± 0.000293** | 0.002324 ± 0.000294 |
  | formula − MLP, test | 0.002616 ± 0.000292 | **0.002527 ± 0.000304** | not evaluated |

  The difference between the columns is fold noise: 1.9e-4 (1 sd) on a single-model mean, about 1.1e-4 on the
  formula − MLP difference. It is not an uncertainty of the models.
- **Must be recomputed and paired locally:** every delta of the campaign (against `mlp_std.pt`, the component formula,
  B0, A2 …). The saved arrays `results/m0_parity/{mlp,formula}_{val,test}_nll.npy` are the pairing partners; any
  later recomputation on this machine must agree with them to 1e-4 per event and 1e-6 on the mean.
- **Can only be quoted unpaired under the local stream:** all stored per-event arrays and their means — `std:mlp_ft`
  18.120842, `std:new` 18.123459, `std:new_n2` 18.123930, `std:base_acc` 18.142046, the report MLP 18.122070, the
  report baseline 18.156301, the CNN 18.121792 (the last three have no weights in the repository).
- **New possibility:** a model evaluated on the emulated reference stream *can* be paired per event against those
  stored arrays, including the report MLP and baseline. This is a verification instrument and an optional secondary
  column, never the headline (the headline stays the unmodified `common.test_nll`).

---

## 3. Amendments to the plan's criteria

| ID | Plan text | Amended criterion | Reason |
|---|---|---|---|
| A1 | means within 5e-5 and per-event test arrays within 1e-4 of the stored values | (a) on the emulated reference stream (launch grid 1056): validation means within 4e-6 and paired differences within 1e-6 of the stored values — met (1.9e-6, 7.2e-8); (b) stream-independent identities exact — met; (c) test event sequence identical — met; (d) test under the local stream statistically consistent with a change of fold points (table row 22) — met; (e) per-event test arrays on the reference stream within 1e-4 — open (C1) | the stored numbers belong to a GPU-specific random stream; the original thresholds are only meaningful on that stream |
| A2 | `sfm.metric` equals `common.test_nll` exactly | bit-identical inside `sfm.metric.deterministic()`; otherwise \|Δ\| ≤ 1e-4 per event and ≤ 1e-6 on the mean | `common.test_nll` differs from itself by up to 2.9e-5 per event between runs (`index_add_` on CUDA; per-hit terms and normalisers are bit-identical) |
| A3 | fallback correlation > 0.9999 | r ≥ 0.9995, and rms of the per-event difference within ±15 % of the validation analogue | r = 1 − rms²/(2 sd²) = 0.99975 for fold noise 0.033 on an event sd of 1.49; two seeds on one machine give 0.99971–0.99978 |
| A4 | G-bulk: hits and normalisation restricted to z′ < 1150 mm (premise: every `zb` above it) | G-bulk headline on the stratum `zb ≥ 1221 mm` with `Bulk(short="raise")`, persistence measured against the report-metric gain on the same events; events with `zb < 1221 mm` reported as their own row (report-metric gain ± SE on them, their share of the total gain, bulk value with `clip` as information). They are never removed from the report metric | §4 |
| A5 | D1: "the sub-100 MeV events are inspected one by one" (validation) | validation has no event below 0.1747 GeV; inspect the validation events below 0.3 GeV (84) and the two training events below 0.1 GeV; every test delta is also reported without the events outside the training momentum range (one event, 0.0734 GeV), pre-registered | §5 |
| A6 | reference values 18.120842 / 18.118084 / 18.123459 / 18.120407 (§3, §12; A0 parity target 18.12041; README) | the local values of §2.5 are the numbers of record; the stored ones are quoted as "reference machine, other fold points" | A1 |

---

## 4. Back-face distance `zb` and the bulk guard

`zb = (z_back − z0)/cos θ`, and `zb cos θ = (59.5 − k) · 20.5 mm` exactly, k = index of the first layer with a hit
(the shower-frame origin). Training / validation:

| | training (69,783) | validation (14,950) |
|---|---|---|
| k = 0 (zb 1221–1508 mm) | 67,905 (97.3 %) | 14,505 (97.0 %) |
| k = 1 / 2 / 3–5 / 6–20 / 21–40 / > 40 | 1041 / 231 / 233 / 224 / 81 / 68 | 251 / 61 / 55 / 45 / 15 / 18 |
| `zb` < 1221 mm | 678 (0.97 %) | 166 (1.11 %) |
| `zb` < 1150 mm ("short", needs k ≥ 4) | 387 (0.55 %) | 89 (0.60 %) |
| `zb` < 0 | 0 | 0 |
| minimum `zb` | 10.27 mm (first hit in the last layer) | 10.27 mm |
| `zb` > 1450 mm (beyond the grid) | 11,627 (16.7 %) | 2,441 (16.3 %) |

Short events (`zb` < 1150 mm):
- **Momentum:** all 20 bins (9–33 training events per bin); not a low-momentum effect.
- **Angle:** 371 of 387 (training) and 88 of 89 (validation) lie in the lowest θ tercile; median θ 0.089 rad against
  0.44 overall.
- **Content:** median 44 hits (training; 25 in validation) against 258; 11–16 % have ≤ 5 hits against 1.3 %; visible
  energy per momentum a third or less of the typical value; 0.17 % of all hits and 0.22 % of the visible energy.
- **Situation:** the first layer with a recorded hit is at least the fifth (k ≥ 4, 82 mm behind the first one),
  almost only at near-normal incidence and at every momentum. That pattern would fit particles that cross the first
  layers without a recorded hit, but the mechanism (insensitive gaps along z, a late interaction, deposits of a
  secondary) was not investigated here and must not be quoted as established.
- **Back face:** 4.1 % (training) / 5.6 % (validation) of their energy sits in hits whose centre is behind the flat
  cut z′ > `zb`, against 0.08 % for the other events; the benchmark models hold 16–18 % of their mass beyond `zb` for
  them (up to 97 %), against 0.2 %.

**Standard metric.** No numerical problem: the normaliser `M_e` is ≥ 0.026 for both benchmarks on training and
validation, the clamp at 1e-8 is never active (no event below 1e-3, three or four below 0.05), all NLLs are finite.
Statistically they are high-leverage events: mean NLL 19.90 against 18.107, sd 4.3 against 1.44; these 0.6 % of the
events raise the validation mean by 0.0107. For the formula − MLP gap they contribute 2 % (0.0078 ± 0.0113 per
event). A θ-blind model cannot exploit them; for the θ track note that `zb` is a function of θ only for k = 0.

**Bulk metric (`sfm.metric.Bulk`, validation, formula − MLP).**

| events | bulk gap | report-metric gap on the same events |
|---|---|---|
| all, `short="clip"` (1 NaN event masked) | 0.001813 ± 0.000278 | 0.002194 ± 0.000293 |
| `zb` ≥ 1150 mm | 0.001884 ± 0.000275 | 0.002170 ± 0.000287 |
| `zb` ≥ 1221 mm | 0.001855 ± 0.000274 | — |
| `zb` < 1150 mm | −0.0101 ± 0.0088 | 0.0078 ± 0.0113 |

One short event has no energy in the region (NaN). Among the events with `zb` ≥ 1150 mm none is NaN, the smallest
retained energy fraction is 0.082, and the 68 events that retain less than half contribute −8e-6 to the gap.

**Recommendation: separate (A4).** `clip` moves the region boundary onto the flat cut `zb` with no margin, so for
these events the guard contains the zone it is meant to exclude (hits behind the cut, the tilted face, straddling
cells). Excluding them silently would hide the population with the largest exposure to the back-face artefact.
Reporting them as their own stratum keeps the guard exactly as the plan designed it for the events where its premise
holds, and makes the rest visible. The threshold 1221 mm is the smallest `zb` of a front-layer start, i.e. the
premise the plan stated. For the existing pair the choice moves the bulk gap by 4e-5 to 7e-5 (0.15–0.25 SE): small,
but not negligible against the 3e-4 to 5e-4 thresholds and the 50 % persistence rule.

The definition of the bulk likelihood itself is sound: model-independent region, same region for hits and normaliser,
exact grid mass, fractional treatment of straddling hits, NaN (not 0) for events without retained energy.

---

## 5. Low-momentum boundary

- After cuts the lowest momenta are 0.0895 GeV (training), 0.1747 GeV (validation) and 0.0734 GeV (test). The plan's
  expectation is confirmed: the cuts remove 61 of the 65 raw events below 0.1 GeV; the four survivors (two training,
  two test) have one or two hits.
- **One test event (181812, 2 hits, 0.0734 GeV, x = −1.112) lies below the training range** (x ≥ −1.000). Nothing in
  validation probes that region.
- T and x are not floored anywhere (row 35). The training minimum that defines x = −1 is a single one-hit event.
- Both benchmarks are finite and normalised down to 0.0005 GeV. From the lowest training to the lowest test momentum
  the maps change smoothly (KL 0.043 for the MLP, 0.050 for the formula). Further down they are finite but not
  physical: at 0.0005 GeV the formula puts 99.86 % of the energy into one cell; at 0.02 GeV and below the MLP puts
  75–95 % beyond r = 200 mm. The two benchmarks already disagree strongly at the boundary (KL(MLP‖formula) = 0.29 at
  0.175 GeV, 1.06 at 0.0895 GeV, 1.43 at 0.0734 GeV, against 0.024 at 1 GeV).
- **Risk for the benchmarks: small and quantified.** In the saved test arrays event 181812 has NLL 10.98 (MLP) and
  12.25 (formula). It contributes +8.5e-5 to the test gap of 0.002527 (3.4 %, 0.28 SE); without it the gap is
  0.002442.
- **Risk for later refits: real, through leverage.** One nat on that single event is 6.7e-5 on the test mean.
  Evaluating the hit patterns of the lowest-momentum training events at 0.0734 instead of 0.0895 GeV costs the
  formula +0.36 nat for 1–2-hit patterns and the MLP +1.7 nat for a 5-hit pattern. An unclamped polynomial or
  symbolic expression can do much worse 11 % outside its range, and validation cannot see it. Hence A5, and the
  audit range of the PySR stage down to 0.0005 GeV stays necessary.

---

## 6. Before Milestone 1 starts

No code has to be changed. Two items have to be recorded, because later milestones compare against them:

- **B1. Record the amendments A1–A6** in `PLAN.md` (§3, §6, §12) and `README.md`: numbers of record = local stream;
  A0 parity target 18.120061 (or paired on the same stream); the bulk-guard strata; the D1 wording.
- **B2. Decide how C1 is closed** (either option is acceptable; the decision adds or does not add a second line to
  the test-access log and therefore belongs to the user / orchestrator):
  - *now:* a `--reference-stream` option of `scripts/parity.py` evaluates `mlp_std.pt` and the formula on test with
    `emul.test_nll_emul(common, model, test, grid=1056)` and compares with `std:mlp_ft` / `std:new`; pass: max
    \|Δ\| ≤ 1e-4 per event and \|Δ mean\| ≤ 4e-6 (expected about 3e-5 and 2e-6). No selection information is
    involved: both models and both reference arrays are fixed and already known. Gate 2c's "only the Gate 0 parity
    run" then reads "two Gate 0 parity lines";
  - *or later:* the same comparison as a mandatory first step of `final_test.py`, aborting before any finalist is
    evaluated if it fails. This keeps one log line until Gate 3 but finds a problem late.

**C1 (open, not blocking M1):** exact per-event parity of the two benchmarks on test on the reference stream. It is
the only direct check of the tensors inside `pion_s1_test.pt`.

## 7. Recommendations (not blocking)

1. Move `philox_emul.c` / `emul.py` into `src/sfm/` (for example `refstream.py`) with a test that compares the
   emulation with `torch.rand` on the local GPU, so that B2 does not import from a gate directory. The shared object
   is git-ignored and is rebuilt on import (`gcc -O2 -fopenmp -shared -fPIC -ffp-contract=off`).
2. Let `registry.evaluate()` call `common.test_nll` inside `sfm.metric.deterministic()`: the code stays unmodified,
   stored arrays become bit-reproducible on this machine, the mean changes by less than 1e-7.
3. Optional secondary column for the final test: finalists on the reference stream, which allows *paired* deltas
   against the stored report MLP and report baseline arrays. Pre-register it or leave it out.
4. M1 audit: add the stratum `zb` < 1221 mm (k ≥ 1) to D1 and D5 with leave-out deltas; it carries +0.0107 of the
   mean NLL and the largest model mass beyond `zb`.
5. Do not use the benchmark maps as teachers or KL targets below the training range (0.0895 GeV); they are finite
   but unphysical there, and the two benchmarks disagree by more than 1 nat of KL at the boundary.
6. `Ā` with and without the back face differs from z′ ≈ 393 mm on (2,620 cells by more than 1e-3). That early onset
   is caused by the late-start events, not by the nominal back face — relevant for rung A3.
7. The fold noise of 1–5-hit events is large (rms 0.15–0.17 per event at K = 64; 1.4 % of the events). It is paired
   away within one stream but dominates the tails of every cross-stream comparison.
8. Add the component formula to `test_real_data_validation_equals_common_test_nll` (it currently checks two random
   models and `mlp_std`; I checked the formula separately, row 25).
9. Documentation: `README.md` says `results/test_access.log` is ignored by the repo-wide `*.log` rule (the negation in
   `shape-form-martin/.gitignore` re-includes it; it will be tracked) and that the plan lives outside the repo
   (`PLAN.md` is in this directory). `parity.py`: `TOL_CORR = 0.9999` and the `flags` block reflect the unamended
   criteria.
10. The reference checkpoints and `test_v3_nll.npz` are git-ignored; only `results/m0_parity/checksums.json` protects
    them. Commit that manifest with Milestone 0.
11. `sfm.stats`: sound. The winner's-curse estimator is calibrated at the null and over-corrects away from it, as
    documented; it covers "lowest mean of exactly these m candidates" only.

## 8. What this gate touched

- Not done: `pion_s1_test.pt` was never loaded (nor hashed); `load_sets(allow_test=True)` was never called; the CSV
  was never opened; no model was trained; nothing under `shape-form-proton/` or `shape-form-original/` was written
  (no new bytecode there); no git state was changed.
- Done with test-related information: (i) the saved test per-event arrays of the parity run and the stored reference
  arrays were analysed, including three individual low-momentum events; (ii) the raw Parquet columns were read for
  all events to re-derive the split, to run the pandas reference builder against the lean builder, and for the
  geometry; per-event metadata (event number, split, momentum bin, p, θ, φ, hit count) of all 99,689 events is in
  `gate0_checks/events_table.npz`. No model was evaluated on test hits.
- `check_protected.py verify` was run with `SFM_RESULTS_DIR` pointing to a byte-identical copy of the manifest under
  `gate0_checks/protected/`, so `results/m0_parity/checksums_verify.json` (the worker's report) is untouched.
- Two tracked files under the protected directories carry today's change time
  (`shape-form-proton/scripts/sr_residual.py` 12:49, `shape-form-original/05_mlp_pion.ipynb` 00:39). Both predate the
  baseline commit (16:12) and equal HEAD and the manifest: not campaign changes.

## 9. Regenerating the numbers

All from the repo root with `uv run python -B shape-form-martin/results/gates/gate0_checks/<script>`; outputs are the
JSON files of the same name. Run in numerical order: `c04` writes the validation arrays `val_grid{1056,144}_*.npy`
and `c05` writes `events_table.npz`, which later scripts read. One GPU job at a time.

| script | content |
|---|---|
| `c00_inspect_checkpoints.py` | contents of the reference checkpoints (no per-event validation arrays are stored) |
| `c01_notebook_vs_common.py` | metric code of the reference run (`00_common.ipynb`) vs `common.py`; `common.py` vs HEAD |
| `philox_emul.c`, `emul.py`, `c02_validate_emulator.py` | CUDA random-stream emulation and its bit-exact validation |
| `c03_find_reference_grid.py [quick\|full]` | scan of launch grids against the stored validation mean of `mlp_std` |
| `c04_confirm_reference_stream.py 1056 144` | 13 stored validation means and 10 paired differences on the reference and local streams |
| `c05_split_integrity.py` | independent split re-derivation, per-bin fractions, low-momentum counts, `events_table.npz` |
| `c06_cache_fidelity.py` | pandas reference vs lean builder vs cache file, modulo-6 event classes |
| `c07_stream_independent.py` | centring, `v0/vs`, `formula_values.json`, `audit_v3.json`, `Ā` (all variants), buffers |
| `c08_parity_stats.py`, `c08b_pair_tail.py` | seed study, cross-stream and cross-machine statistics, paired differences, bulk retained energy |
| `c09_metric_repro.py` | run-to-run reproducibility of `common.test_nll`, equality with `sfm.metric` |
| `c10_zb_bulk.py` | `zb` distribution, short events, normaliser, bulk metric |
| `c11a_hits_1001.py` | the two events with 1001 hits |
| `c12_units_lowp.py` | units, ranges, rotation convention, time of flight, low-momentum maps and leverage |
| `c14_stats_review.py` | 1-SE rule, paired SE, stratified fold on real hits, low-momentum test events in the saved arrays |
| `c15_geometry_train_only.py` | geometry from training events only |
| `c16_test_order_fingerprint.py` | per-bin test means of `test_v3.json` with our bin labels |
| `g0common.py` | shared setup (never loads the test cache) |

`uv run pytest shape-form-martin/tests -q -rP -p no:cacheprovider` → 71 passed.
