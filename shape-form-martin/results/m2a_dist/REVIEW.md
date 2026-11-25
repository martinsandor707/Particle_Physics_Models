# Track A (M2A) driver - independent review of commit `c4f39fd`

Reviewer: independent session (mathematics, statistics, verification). Date: 25 Nov 2026.
Subject: `scripts/run_m2a.py`, `scripts/run_m2a_lib.py`, `scripts/run_m2a_report.py`, `src/sfm/models/paramnet.py`,
`src/sfm/models/dist_factory.py` and their tests, as committed in `c4f39fd`, judged against the base modules as
settled in commit `0c38b32` (the five Track A files and their three test files are byte-identical in both commits;
`HEAD` at the end of the review is `7fdf96c`; since `0c38b32` only one test file of the base modules and two
documents changed).
Units below: 1e-4 nat per event, mean ± paired SE, unless a unit is written. `<scratch>` is the reviewer's scratch
directory (reproducers and logs; not part of the repository).

**STATUS OF THIS FILE: complete (25 Nov 2026). All tests of section 4 have finished.**

## 0. Files reviewed (SHA-256 of the content in commit `c4f39fd`)

| File | SHA-256 |
|---|---|
| `scripts/run_m2a.py` | `98f13ee96882f71486a274282d571a5601b32997db4fff9dd4dd46c9ca26c6e0` |
| `scripts/run_m2a_lib.py` | `18b211cee78276e9eaa0fafef18849b0138ebef6507f14c451ca3e2209c00ab5` |
| `scripts/run_m2a_report.py` | `086d7f6955779ca040f6a83be29598269164a24b78d301614be9920bab09b113` |
| `src/sfm/models/paramnet.py` | `9f9a14d33c39ce29d76fb126f233b6ccd7e7c0e56842cbd8562d2bc903595a62` |
| `src/sfm/models/dist_factory.py` | `ddc5a7ddbaa28265cb0fb3aba814607047208f19238e217458b648516c1ab728` |
| `tests/test_dist_factory.py` | `ed214b470293859fadd959ceec64c72b6d4fa8a342492d7af0e56880b7d3fbfb` |
| `tests/test_models.py` | `1693096df8cd9afa7f97cfc4a7635868dd9928d1444c524b9ae1883e8542a0e2` |
| `tests/test_run_m2a.py` | `932fdaa3aff7a80eb5be0b442cc5369a2134a712613643ea597e25ff95f731ec` |
| `results/m2a_dist/STATUS.md` (the worker's account) | `42b8a3fa48c8fd9c4c4d53f466ac07fe2a3d676893e534ab6d4a300d9930482c` |

Guardrail files, before and after the review: `common.py` `ece3c67f...ebb3` (no diff against `HEAD`; last changed
in the Milestone 0 commit), `results/registry.jsonl` `017edeba...81de`, `results/test_access.log` `effa9734...0c60`
(the value of PLAN 13, A6 revised point 9). No test array, test cache or `*.csv` file was opened; `load_sets` was
called with `("train", "val")` or `("val",)` only. No real Track A fit was run. In the clone the review wrote this
file only (and touched the git-ignored `cache/gpu.lock`).

## 1. Verdict

**PASS WITH REQUIRED FIXES. No finding of class (A): nothing has to change before the run starts.** What the
driver fits and registers is the pre-registered list and its rules. Six findings of class (B) concern the report,
the reading of the table and Stage 2; they change no fit and no registry row and can be made while the fits run,
but before the Stage-1 table is read (B1 to B4, B6) or before Stage 2 (B5). One robustness weakness (C1) is
recommended before the start if that does not delay it: without it the run is still correct, it can only stop
early and wait for a person.

Conditions of the start, unchanged from STATUS section 4: batch 1 has passed on the H200 (reference set with its
parity record); the command is issued through the launcher; no `--smoke`, no opt-out variable.

Limits of this verdict: no real-length fit exists anywhere, and nothing ran on the H200 (section 4, last two
lists). The review looked for what would make the run's results unusable, biased or other than pre-registered; it
does not predict that every one of up to 86 fits runs through.

## 2. Findings

### (A) To be fixed before the real run starts

None. The run list, the model structures and constant counts, the training configuration, the seeds, the starting
points, the conditions of the conditional runs and the pairing are the pre-registered ones (section 5); the stops
(parity, drift) are enforced in real mode; Stage 2 is not in the default list.

### (B) To be fixed before the Stage-1 table is read, or before Stage 2 (none changes a fit or a registry row)

**B1. The report applies the rules to the non-converged original of a primary even when its rerun exists.**
- *Evidence.* `run_m2a_report.py:240-248`: `c` holds `<rung>_s<seed>` against `A2s_s<seed>`; the reruns go to
  `reruns_x2` and are only printed (`:337-340`). `replication`, `equal_both`, the "cleaner" flags and
  `kept_and_replicated` (`:262-270`) - the input of Stage 2 - are computed from `c` alone. Seen by execution: the
  rerun path was not taken in any smoke run; forced once in this review (section 4), the report printed the rerun
  as an extra row and left every rule column on the original.
- *Why.* AUDIT 4.0: "A non-converged primary is rerun once with doubled steps together with its comparator;
  nothing else changes." The rerun pair is the measurement of that seed; the table would otherwise show "kept"
  from a fit the rule has just declared unfinished, or miss a rung that passes only when converged.
- *Fix.* For each seed use the pair (`_x2`, `A2s_..._x2`) when it exists, mark the row "rerun", and compute
  replication, equal NLL and `kept_and_replicated` from the pairs so chosen (seeds may then mix an original and a
  rerun: each is paired with its own comparator, which is what the rule needs). Keep the original as a labelled
  extra row.

**B2. The initialisation record that AUDIT 4.1 asks for is stored but not shown, and a degraded network warm
start would not be visible in the table.**
- *Evidence.* AUDIT 4.1, column "Initialisation": A4 "log `kl_map`", A7 "exactly A2s at (c, k) = (2, 1)", A8
  "exactly A2s at b = 1". `kl_map` is written to `init.json`, `done.json` and the row (`run_m2a_lib.py:455-459`),
  but `run_m2a_report.py` prints none of it. For the net runs the fit report of `fit_to` (`rms_max`, constant and
  near-floor columns, L-BFGS stop) is stored in `init.json` only.
- *Why.* (i) "Exactly A2s" is a claim the table should carry (smoke: `kl_map` of A7, A7s, A8 between -1e-8 and
  +2e-8, i.e. zero up to rounding; A9 at `logit_dh` = -5: 2.3e-4). (ii) The warm start of a net run can fail
  without any error - see the ruling on `fit_to` in section 3: one target column of small but non-zero spread
  leaves **all** columns at 14-17 % of their spread instead of 0.04 %. The comparison "net version - A2n" would
  then start from a damaged copy and nobody reading the table could tell. The reference value of a healthy warm
  start is `kl_map` = 5.5e-6 nat (control A1 with the pre-registered call, built on the CPU in this review). The
  smoke runs say nothing here: their shortened warm start (40 Adam steps, no L-BFGS) gives 0.06-0.09 nat.
- *Fix.* An "Initialisation" table in `RESULTS.md`: for every run `kl_map` and its reference; for net runs also
  `rms_max` with its column, the columns whose relative spread sd / max(1, |mean|) is below 1e-2, and whether
  L-BFGS stopped. Flag a net run whose `kl_map` exceeds 1e-4 nat (one unit of the tables; fixed here, before any
  net run exists) as "warm start degraded - net column not interpretable for this run". Everything needed is in
  the stored files; the relative spreads can be recomputed from the fitted polynomial run.

**B3. Two flagged readings decide columns of the table; the alternative should be printed beside the literal
one, so that the user's choice needs no recomputation.**
- Replication (`run_m2a_report.py:100-116`): print, next to "replicated", whether the two-seed mean is also
  <= -2 of its paired SE (the numbers are already in `two_seed_mean`).
- G-region (`run_m2a_report.py:190-211`): print the flagged share of the hit-term total beside the flagged share
  of the total gain. With the total gain in the denominator a gain in the normalisation term - the signature of
  mass moved beyond `z_b` - *lowers* the flagged share. Example: hit term flagged -3, inner -1, normalisation -6
  gives 30 % of the total (pass) and 75 % of the hit term (fail).
- Rulings on both: section 3.

**B4. The limitations of the two computed templates must be stated with the A3c / A3x results.** Text and
measured numbers: section 3, Q1 and Q2. `RESULTS.md` should print them from `templates.json` (it already holds
`fb_vs_stored_abar` and `cells_above_1`; add the count above 1.01, the lowest depth at which that happens and the
smallest denominator).

**B5. Stage 2 is not covered by this review and is weakly interlocked.**
- `phase_stage2` (`run_m2a.py:644-655`) is refused only while `results.json` does not exist, and the default run
  writes that file itself: after the default run nothing but the operator stands between the table and A11
  (shown by execution, section 4).
- A11p is not implemented; the content of A11 depends on the replication reading (B3) and on B1.
- *Fix before Stage 2 is started:* require a hand-written approval file that names the SHA-256 of the
  `results.json` the user saw; refuse if the file is absent or the hash differs; review `stage2_specs` and the
  missing A11p then.

**B6. (Statistics; a property of the pre-registered design, not a defect of the code.) Every rung that starts at
the fitted A2s carries a common "continued training" term, and the list has no control for it.**
- *Evidence.* By the list, a rung is A2s (4000 steps) continued for 4000 more steps with a new optimiser state and
  a new learning-rate cycle, and is compared with the A2s it started from. The convergence rule looks at the rung,
  not at A2s, and a small movement at the end of a cosine cycle (rate lr/15) does not show that a second cycle
  would find nothing. In the smoke runs the six rungs that start exactly at A2s and add one to three constants
  all sit at the same offset from A2s: A7 +0.55, A7s +0.50, A8 +0.50, A8c +0.52, A78 +0.52, A12 +0.43 (each
  ± 0.24), while their mutual paired differences are -0.07 ... +0.04 (± 0.05). Six different families with one
  offset: the offset belongs to what they share, the continuation (here 40 smoke steps with a fresh optimiser),
  not to the family. Its sign and size in the real run are not known; after 4000 steps it may well be a gain.
- *Why.* The keep rule (Δ <= -3 and <= -2 SE against A2s) is pre-registered and stays. But a "kept" of size 3 to 5
  cannot be attributed to the family unless the continuation term is known to be small.
- *What to do.* (i) Reading aid, no new run: differences **between** rungs that share the start (for example
  A78 - A7, paired) are free of the common term; the least favourable of the rungs that start exactly at A2s
  (A7, A7s, A8, A8c, A78, A12) bounds it from one side. The report should print the convergence movement of
  A2s_s0 / A2s_s1 beside the primaries table (it is already in the convergence table). (ii) **[confirm]** a
  control "A2c" = A2s continued unchanged (start = A2s of the same seed, T_A, seeds 0 and 1; two fits). It is not
  in the pre-registered list, selects nothing and can be added at any time after A2s exists - also after Stage 1.
  The run may start without it.

### (C) Not blocking

- **C1. A non-finite evaluation stops the track, and a restart stops again at the same run (recommended before
  the start).** `sfm/registry.py:679-680` raises `FloatingPointError` when any per-event value is not finite;
  `run_m2a_lib.py:471-478` does not handle it. `sfm/train.py:544-551` skips batches with a non-finite loss, so a
  fit that leaves the finite region ends frozen there, writes `model.pt` and `fit_done.json`, and fails only at
  registration; on a restart `train.fit` returns the finished fit and the evaluation raises again. Shown by
  injection (section 4). No run is skipped silently and no number is wrong; the run waits for a person.
  Handling proposed here, to be fixed now so that such a stop is not followed by an improvised decision: *a
  learning-rate pilot without finite scores loses* (**[within]**: the rule "the lowest last-iterate validation
  mean wins" already says so); *an exploratory rung or a net version without finite scores is recorded as "FAILED:
  non-finite validation NLL", gets no second seed, and the list continues* (**[confirm]**: the list has no rule
  for it; until confirmed the stop stands); *for a base, control or primary run the stop stands and the user
  decides*. Implementation: catch the exception for the classes `pilot`, `exploratory`, `primary-net`; write
  `failed.json` (reason, fit summary with the skip counters) instead of `done.json`; decide P-lr among the pilots
  that have a registry row and stop if a class has none; print the failures in the report. In the smoke pilot at
  the largest rate (0.03) the validation mean stayed within 25 of the start and no batch was skipped, so the case
  is not expected - which is why this is not class (A).
- **C2. A failure in the guards phase leaves the run without a table.** `guards` precedes `report` in the default
  list; the report needs no GPU. Put a `report` before `guards` as well (order `... stage1x report guards report`).
  The guards phase re-evaluates every target and accepts only a bit-identical K = 64 array; where evaluations are
  not reproducible bit for bit it stops at the first target (laptop: identical; the H200 is checked by batch 1).
- **C3. The drift rule is checked after seven fits.** `phase_plr` trains A2s_s0 and the three net pilots before
  `phase_stage0` trains A0 and applies the drift rule. Nothing is wrong with the numbers; moving A0 (seeds 0, 1)
  and the drift rule to the point where `lr_poly` is frozen would stop a broken harness four fits earlier (about
  10 minutes at laptop speed).
- **C4. Phases started by name do not re-check the recorded stops.** `plr` checks the parity record; `stage1p`,
  `stage1x`, `guards` check neither `parity.passed` nor `A0_drift.stop`. In the default sequence the stops are
  exceptions and nothing runs after them; the check matters only for a hand-issued phase list.
- **C5. Templates.** (i) `phase_templates` skips when the files exist, without comparing `templates.json` with the
  code (`nsub`, `nphi`, definition); after any change of the template code an old directory would be used
  silently. (ii) The detector-halo tables of A9 / A9r / A9n are non-persistent buffers
  (`families/halo_det.py:158-162`) rebuilt from `train_theta.npy`, so they are **not** covered by `model.pt` or
  `weights_sha256` (the acceptance template is: `logA` is a persistent buffer). Record `train_theta_sha256` in
  `init.json` of those runs and verify it when the file is loaded.
- **C6. `ParamNet.fit_to`.** A scale floor for columns of small relative spread (section 3) - a change of the
  warm-start procedure, to be decided with the user, not now. Docstring: explicit `(G, P)` weights are described
  as "used as given", but `paramnet.py:242` normalises every column to sum 1, which removes the relative scale
  between columns (not used by Track A).
- **C7. Absolute paths in logs.** `sfm/train.py:452, 491, 619` print `model_path` / `ck_path` unscrubbed into
  `fit.log` and the job log, and `sfm.gpulock` prints the path of the lock file into the job log (both seen in
  the review's smoke logs). These files are git-ignored; they must stay uncommitted or the lines get
  `paths.portable` (base modules; for `train.py` the worker's proposal in STATUS section 7).
- **C8.** `run_m2a_lib.gpu_seconds_total` parses every line of `gpu_time.jsonl` on every phase start; a truncated
  last line would block every restart. Skip unparsable lines (the number is information only).
- **C9.** The guard row of a run (`run_m2a.py:565-570`) is the "latest row" of that name and lacks the `sources`,
  `rung`, `comparator` and `kl_map` entries of the first row; anything that reads provenance from the latest row
  must go to `done.json`. Copy the first row's `extra` into the guard row.
- **C10. STATUS.md.** Stop a job with `scripts/stop_job.sh <pid file>` (base commit `0c38b32`), not a bare `kill`.
  The start-up failures now also include `UnguardedStart`, `ReferenceSetNotLicensed` and `ReferenceSetDisputed`.
  "Q2: 56 cells" is the count above 1.1; above 1 there are 906 cells, above 1.01 there are 79.
- **C11.** On the capped device a job is refused when less than 14 GB is free, and the process exits
  (`run_m2a_lib.py:255-257`). On a GPU that another user fills at times this ends the unattended run at an
  arbitrary rung; a bounded number of retries (for example three, ten minutes apart, lock released in between)
  would not conflict with "do not wait for an empty GPU". The GPU lock itself is taken without a timeout: a
  campaign job that never releases it makes Track A wait, with the holder named in the log.
- **C12.** `RESULTS.md`: label primaries and exploratory rungs in the "kept and replicated" line; add one sentence
  on multiplicity (six pre-registered comparisons; every other row exploratory and not a test). The replication
  rule is applied in seed order (keep on seed 0, replication by seed 1); a primary that passes keep on seed 1 only
  is "not kept" - the table shows both keep flags, a footnote should say so.
- **C13.** Cosmetic: the listing shows `A2e_s<seed>` as comparator of the `A2s_s<seed>_x2` rows. `--list` reads
  the device properties when the module is imported (it allocates nothing and writes nothing).

## 3. Rulings

Each ruling is marked **[within]** = within the pre-registration (the most literal reading, or a numerical detail
the text leaves open), or **[confirm]** = needs the user's confirmation. For every [confirm] it is said whether the
run may start without it and what would have to be redone. **No ruling requires the user before the start.**

### Q1 - front-face bin of the computed templates **[within]: runs as registered, limitation stated**

The ported `acceptances` is a faithful copy of `results/m1_audit/checks/c_side.py::acceptances`, the code AUDIT 4.1
names ("computes all of them"): 2 x 2 sub-points per cell, 24 azimuth nodes. It is not an incorrect port. Its
quadrature is coarse in exactly one place. Measured (training events, `<scratch>/templates_check.json`):

| Quantity | Value |
|---|---|
| computed front+back vs stored `abar.pt`: max abs difference | 0.146, in z-bin 14 ([-19.5, -6.6) mm), innermost ring: 0.500 against 0.354 |
| the same outside z-bin 14 | <= 0.016 (bins 9-19 and 99-104), mean 7.6e-4 |
| cause outside bin 14 | 24 azimuth nodes: 24 against 96 nodes on the same events differ by up to 0.017 |
| with 4 sub-points (the rule of the stored templates), bin 14 | within 0.005 of `abar.pt` |
| `abar.pt` itself | reproduced exactly (0.0) by the earlier study's rule (analytic in azimuth, 4 sub-points); with 16 sub-points the innermost value of bin 14 is 0.369 |
| ratio computed / stored in bin 14 | 1.41, 1.08, 0.96 in rings 0, 1, 2; from ring 8 outwards the two agree within 1 % |

What it costs, at the legacy coefficients, on validation (K = 64, the laptop's fold points; a diagnostic, never a
number of record, never registered):

| Comparison | Δ |
|---|---|
| stored template with bin 14 replaced by the 2-sub-point values, minus stored | -0.10 ± 0.01 |
| A3c as coded minus A3c with the stored template as numerator | -0.16 ± 0.02 |
| A3x as coded minus A3x with 4 sub-points | -0.14 ± 0.02 |

These are 20 to 30 times below the keep threshold (3) and about a seventh of a typical paired SE. The difference
between A3c / A3x and A2s is therefore the template definition, not the sampling error, to that accuracy.
*Statement to print with the result:* "numerator by the 2 x 2-point rule of `c_side.py`; in the z-bin of the front
face it overestimates the innermost rings (0.50 against 0.35); effect on the validation NLL at the legacy
coefficients -0.1 to -0.2." A reading with the stored template as numerator (`stored` "= mean_e A_front∧back
(exists)" in the same paragraph of the audit) is equally literal for A3c; the measured difference between the two
makes the choice immaterial, so the code stays as it is.

### Q2 - `back_corrected` above 1 **[within]: runs as registered, limitation stated**

The ratio is the definition, literally, and values above 1 are its geometry, not an error: the back plane is
tilted, so a ring at depth z' > zb_e is still partly inside the detector up to z' = zb_e + r tan θ, while
1[z' < zb_e] is already 0. Measured: 906 cells above 1, of which 824 by less than 0.1 %; 79 above 1.01, all at
z' >= 1379 mm; 56 above 1.1 (from 1392 mm); maximum 1.81 in the corner (last ring, z' = 1444 mm), where the
denominator is 0.186. The 2-point denominator differs from the exact event mean (linear in the bin of zb_e, as
the metric's normaliser) by at most 6.4e-4.
*Statement to print with the result:* "the template exceeds 1 by more than 1 % in 79 cells at z' >= 1379 mm
(maximum 1.81): it adds density behind the flat cut, where the metric's normaliser does not count mass. A gain of
A3c is to be read with the G-region class `deep` and the `beyond_zb` mass monitor." Both are computed
automatically if A3c passes the keep or the equal-NLL rule.

**Rungs affected by Q1 and Q2:** A3c and A3x only (and A11 if one of them were kept and replicated). A3f uses the
existing file `abar_front.pt`; A9 / A9r / A9n use the training angles, not the templates. **None is a primary.**
For Track B: a `back_corrected` arm of B4 would inherit both statements; the primary B4f uses `abar_front.pt`.
If the user later prefers another numerator or a finer rule, only A3c and A3x (one seed each, plus a second seed
if triggered) would be refitted; nothing else depends on them in Stage 1.

For orientation only (legacy coefficients, before any refit, same diagnostic): A3f - stored = +11.2 ± 1.2,
A3c - stored = +8.1 ± 1.1, A3x - stored = +13.2 ± 2.8.

### The `ParamNet.fit_to` fix **[within]**

- *Mathematics of the constant-column treatment: correct.* A column with weight 0 contributes `0 * (f - t)^2` to
  the objective, so its last-layer row and bias receive an exactly zero gradient; Adam (no weight decay inside
  `fit_to`) leaves them at 0 / the mean, L-BFGS directions are zero in those entries, and they are reset at the
  end in any case. For an exactly constant target (c, 0) minimises the column's own term for every value of the
  hidden layers, so weight 0 selects a minimiser of the original problem. The float64 detection makes a
  bitwise-constant column constant whatever the float32 mean rounds to.
- *"Other columns bit-identical to the old code": true with two qualifications.* (i) It holds when L-BFGS is
  monotone from chunk to chunk; the new code keeps the best chunk, the old one the last (equal on the test
  trajectories). (ii) Scalars of `names_const` are now the exact constant instead of the float32 mean of 48
  equal numbers - a change of a few ulp, in the right direction; every real net run has such scalars, so "no
  constant column present" never applies to the run list. Neither matters for a number of record.
- *Effect on the pre-registered net runs.* None that matters: the only constant **net** column of the list is
  `logit_dh` of A9n, which the code before the fix could not fit at all. For A1 the pre-registered call
  (48 points, 2000 Adam steps, 500 L-BFGS iterations) gives `kl_map` = 5.5e-6 nat, worst column 1.1 % of its
  spread (`<scratch>/a1_warmstart.json`); the 21 constants are reproduced exactly.
- *A column nearly but not exactly constant falls on the wrong side - shown, not present at the starting
  coefficients, not excluded for fitted targets.* Reproducer `<scratch>/fit_to_floor.py` (CPU, the pre-registered
  call, six ordinary cubic columns plus one column 2.0 + small cubic):

  | relative spread of the extra column, sd / max(1, \|mean\|) | treated as | worst rms / spread of the six ordinary columns |
  |---|---|---|
  | no extra column | - | 0.04 % |
  | 0, or 1.8e-7 | constant | 0.04 % |
  | 1.0e-6 (ratio to the floor 1.1) | varying | 17 % |
  | 1.0e-5, 1.0e-4, 1.0e-3 | varying | 17 %, 15 %, 14 % |
  | 2.5e-3, 5e-3 | varying | 4.2 %, 1.6 % |
  | 1e-2, 2e-2, 4e-2 | varying | 0.7 %, 0.2 %, 0.1 % |

  Cause: the variance normalisation asks for an absolute precision of 1e-3 of the column's spread; Adam's step and
  float32 cannot deliver it for a small spread, the column's term stays of order 1, dominates the shared layers,
  and L-BFGS stalls on rounding noise. The floor 2^-20 removes only the exactly constant end of this range, and
  `near_floor_columns` reports ratios below 1e3 only, i.e. not the cases at 1e-3 to 5e-3 either.
  The 22 momentum-dependent legacy columns have relative spreads from 0.041 to 1.21, outside the range; the
  worker's test covers the eight net models at their **starting** coefficients. The real targets are the
  **fitted** polynomial runs, which do not exist yet. A fitted degree-3 row that barely depends on momentum
  (relative spread below about 5e-3) would damage the warm start of that net run silently. Consequence: finding B2
  (make it visible, with a fixed flag threshold). A cure inside `fit_to` (a scale floor of about 2^-5 max(1,
  |mean|), below every legacy column, so that all starting fits stay bit-identical) changes the pre-registered
  warm start in the affected case and is **[confirm]** - the run may start without it; if the user wants it after
  seeing a flagged run, only the flagged net runs (and, if A2n is flagged, the net column) would be refitted.

### Flagged readings (numbering of STATUS.md section 3)

| # | Reading implemented | Ruling | May the run start? What would be redone? |
|---|---|---|---|
| 1 | parity: A0 at step 0 equals this machine's formula array within 1e-4 per event and 1e-6 on the mean | [within] - A6 revised point 4, verbatim | - |
| 2 | drift: \|A0 - formula array\| > 2 on either seed stops, two-sided | [within] | - |
| 3 | P-lr: lowest mean; within 0.5 of it the smaller rate; net pilots fitted to the Stage-0 run A2s_s0 (4000 steps at the frozen `lr_poly`) | [within] - "A2n: fit_to A2s (same seed)"; the tie rule is the text | - |
| 4 | equal NLL, one-sided: Δ <= +1 SE and Δ < +1; a second exploratory seed on keep or one-seed equal NLL | [within] - "within +1 paired SE" has a sign; the two-sided reading would run fewer second seeds, never more | yes; nothing |
| 5 | replication: seed 1 <= -1.5 with the same sign, two-seed mean <= -3, no 2-SE condition on the mean | [within] as the literal reading: "half the threshold" can only be half of a number, and the same word in the same sentence then means that number. **[confirm] before Stage 2**, because it decides the content of A11 | yes; nothing is refitted - one column of the table and the Stage-2 list change (B3 prints both) |
| 6 | kept and replicated = keep on seed 0 and replication | [within], with B1 (reruns) to be fixed | yes; nothing |
| 7 | convergence: \|val(last) - val(last - 1000)\| < 0.5; reruns for the poly primaries only, comparator rerun shared per seed; the rerun starts from the original A2s, as the first run did | [within] - "nothing else changes"; the net versions are exploratory and are fitted to the first poly run | yes; nothing |
| 8 | guards for every rung that passes keep or one-seed equal NLL, and their comparators | [within] - a superset; guards select nothing in Stage 1; they add registry rows and change no array | - |
| 9 | G-region: hit term by hit centre; flagged share of the **total** gain; margin form with SE | [within] - "the gain" is the report-metric gain, as in G-K and G-bulk. The share of the hit-term total is the stricter number where the normalisation term gains; print both (B3) | yes; nothing |
| 10, 11 | G-mass paired per monitor, no threshold; G-bulk on zb >= 1221 mm, non-finite bulk values masked in both runs and counted | [within] | - |
| 12, 13, 14, 17 | secondary columns; "cleaner" shown per rung against this machine's formula array; no 1-SE rule in Stage 1; A9 step-0 rows | [within] | - |
| 15 | A6b: observed information of the raw coefficient of `sh_rk` on training events, second difference | [within] - exploratory; the audit asks to "report the Fisher information of k" without a definition | - |
| 16 | Stage 2 | not ruled here (B5) | Stage 2 does not start |
| - | net split: radial shape names are constants, not net outputs | [within] - the counts follow the audit's formula 304 + 17 n_net + n_const (A1, A2n: 699) | - |
| - | A9r: `logit_dh` starts at the logit of the training-event mean of the fitted far-halo weight | [within] - exploratory; one constant has to stand for a momentum-dependent weight | - |
| - | A10: `init_from_target(..., seed=0)` for both seeds | [within] - 0 is the default of the pre-registered call | - |
| - | a control A2c (finding B6) | **[confirm]** - an addition to the list | yes; two fits, at any time after A2s |
| - | a run without finite validation scores (finding C1): a pilot loses; an exploratory rung or a net version is recorded as failed and the list continues | pilot: [within] (the P-lr rule); exploratory / net: **[confirm]** | yes; as coded the run stops at such a run and waits. Nothing is refitted either way |

## 4. What was verified, and how

Everything below ran on the laptop (launch grid 144), with `SFM_OUT_ROOT` in the review's scratch directory, on the
base of commit `0c38b32`. Fits ran in smoke mode only (stage `m2a_smoke`, steps / 100). Reproducers and logs:
`<scratch>/` (`counts.py`, `templates_check.py`, `fit_to_floor.py`, `a1_warmstart.py`, `inject.py`,
`force_rerun.py`, `check_state.py`, `rebuild_check.py`, `recompute_report.py`, `common_term.py`, `smoke_seq.sh`,
`smoke_seq2.sh`, `logs/`).

### Verified by execution

| What | Result |
|---|---|
| `run_m2a.py --list` | 97 rows; 54 unconditional fits, 224,000 steps; at most 32 conditional fits; compared with AUDIT 4.1 line by line (section 5) |
| constant counts, every model of the list built on the CPU | 29 of 29 equal the audit's column; `logit_dh` of A9 starts at -5 (weight 0.67 %), degree 0; net versions of the primaries: 665 / 699 / 701 / 700 / 718 / 599 by the audit's formula |
| `pytest tests/test_dist_factory.py tests/test_run_m2a.py tests/test_models.py` | 82 passed, 0 skipped (templates present) |
| reference set in a fresh scratch root: `val_parity.py`, then `val_parity.py --create-refset` | both "ALL APPLICABLE CHECKS: PASS"; set created with its parity record and accepted by the driver |
| real mode, `templates parity` (an evaluation, not a fit; templates copied from the worker's dry run) | parity -1.869e-08 ± 1.1e-08 nat, largest event 1.34e-05 -> PASS; a second `parity` re-reads the registered row and decides the same |
| real mode, refusals | `stage2` before any report: refused; `plr` before parity: refused; `report stage2`: refused; `--smoke` without `SFM_M2A_SMOKE=1`: refused |
| kill inside a fit (pilot, step 25 of 30) | restart resumes at the checkpoint of step 21 |
| kill after the fit, before registration | restart loads the finished fit and registers it |
| kill inside registration (arrays saved, row not appended) | restart recomputes, the identical arrays are accepted, one row |
| kill after the row, before `done.json` | restart finds the row by the checkpoint's SHA-256, writes `done.json`, no second evaluation |
| state after these four kills and Stage 0 | 19 rows, 18 done-markers, no duplicate row, every `done.json` backed by a row of its `model.pt` |
| kill inside the fit of a net run (A4n) | restart resumes without a second warm start (`init.json` kept) |
| non-finite evaluation (one NaN injected into the array of A4t_s0) | `FloatingPointError`, exit 1, no row, no `done.json`; the restart stops at the same place (finding C1) |
| kill inside the report (`results.json` written, `RESULTS.md` not) | restart rewrites both |
| end of the smoke run | 55 finished runs (54 + A9r seed 1, triggered by one-seed equal NLL), 64 rows, no duplicates; drift, P-lr winners, Fisher information of A6b (1442.8) equal to the worker's dry run to the printed digits |
| every Stage-1 number of `results.json`, recomputed with numpy only from the stored arrays | 37 comparisons: means, paired SE, keep / equal flags, two-seed means and replication flags identical (difference 0) |
| guard blocks recomputed the same way (4 blocks) | G-K and G-bulk margins and SE, pass flags, G-region margins identical; bulk arrays have the stratum length (14,784); hit term + normalisation term reproduces the main difference to 1.5e-5 per event |
| a finished run rebuilt from `model.pt` (the path of the guards phase, of Stage 2 and of Gate 1) | 15 of 15 models - poly, net, mixture, net mixture, last-ring wrapper, detector halo, template rungs - give the registered K = 64 array bit for bit |
| the real report after a complete smoke run | "0 primaries, 0 exploratory rungs, 0 guard blocks": smoke rows are not read (other stage) |
| real `stage2` after that (empty) report | passes the gate and stops only because `plr.json` is missing (finding B5) |
| the convergence-rerun path, forced for one primary (A7_s0; it was not taken in any smoke run) | `A2s_s0_x2` (from the legacy coefficients, doubled steps) and `A7_s0_x2` (from the original A2s_s0, `kl_map` 1.4e-8) trained and registered; a plain restart changes nothing; the rerun pair gets its own guard block (5 blocks, recomputed identically); the report prints the rerun as an extra row (+0.59 ± 0.18) and takes replication, equal NLL and `kept_and_replicated` from the original (finding B1); end state 68 rows, 57 finished runs, no duplicates |
| written files | no JSON, registry or Markdown file under the scratch output root contains a local path (308 files); 55 `fit.log` files do (finding C7); nothing written in the clone |
| templates (Q1, Q2), `fit_to` (constant and nearly constant columns, the A1 warm start) | section 3 |
| offset shared by the rungs that start exactly at A2s (smoke, seed 0) | A7 +0.55, A7s +0.50, A8 +0.50, A8c +0.52, A78 +0.52, A12 +0.43 (each ± 0.24); their mutual differences -0.07 ... +0.04 (± 0.05) (finding B6) |

GPU memory: every smoke fit (same batch and K as the real ones), the K = 1024 guard evaluations, the templates and
the Fisher step ran on an 8 GB device that another process shared (1.5 to 3 GB), i.e. within about 5 GB.

### By reading only

- Real-length fits: none was run anywhere. The warm starts of the net runs on **fitted** targets and
  `init_from_target` with its real settings (1500 + 1500 steps, 400 search points) have not been executed.
- Enforcement of the parity and drift stops in real mode (`raise SystemExit`; the smoke path records and
  continues), the tie rule of P-lr with real numbers, the second-seed trigger on "keep".
- The capped-device branch of `on_gpu` (refusal below cap + 4 GB) and the clean-clone refusal.
- Stage 2 (`stage2_specs`, the merge table): not reviewed for execution (B5).
- The mathematics of the families, of `init_shower_long` and of the metric: taken from the earlier gates.

### Only on the H200

- The numbers themselves (launch grid 1056 expected; every delta against the reference set created there).
- Start-up under the launcher: guard active, output root outside the clone, clean clone, `HEAD` pushed.
- The 10 GB cap and the free-memory refusal next to another user's process.
- Bit-identical re-evaluation next to another user's process - needed by the guards phase and by a restart inside
  a registration (batch 1 checks it; C2 protects the table if it fails).
- Speed, and with it the duration of the run.

## 5. Answers to the questions of the review

**The run list.** `run_m2a.py --list` gives 97 rows: the step-0 row, 54 unconditional fits (6 pilots, 12 Stage-0
runs, 12 + 6 primaries, 18 exploratory rungs at seed 0) and at most 32 conditional ones (18 second seeds, 12
primary reruns, 2 comparator reruns). Compared line by line with AUDIT 4.1:

| AUDIT 4.1 | Listing and code | Result |
|---|---|---|
| P-lr: A2s at 0.003 / 0.01 / 0.03, A2n at 0.001 / 0.003 / 0.01, seed 0, 3000 steps, winner frozen per provider class | rows 2-8; `frozen_lr` read by every later run | same |
| Stage 0: A0, A1, A2e seeds 0, 1; A2s, A2n seeds 0, 1, 2; 4000 steps | rows 5, 9-19 | same; A2s_s0 is trained inside the P-lr phase because the net pilots are fitted to it |
| primaries A4 A5 A7 A8 A9 A10: poly seeds 0, 1 from A2s of the same seed, against A2s of the same seed; net seed 0 fitted to the poly run of seed 0, against A2n seed 0; A10 6000 steps; A9 with step-0 rows | rows 20-37 | same |
| reruns: a non-converged primary once, doubled steps, with its comparator | rows 38-61, conditional | same (poly primaries only; ruling 7) |
| 18 exploratory rungs in table order, seed 0; seed 1 on keep or equal NLL; A10 variants 6000 steps | rows 62-97 | same |
| Stage 2 | not in the list; explicit phase | as approved (PLAN 17.6) |
| constants | every model built on the CPU: 29 of 29 counts equal the audit's column | same |
| training configuration T_A | `fit_config`: batch 256, k_train 16, iid, cosine lr -> lr/15, eval and checkpoint every 500, k_val 64, last iterate, smooth 0, dedicated fold generator, clip 5; weight decay 0 (poly) / 1e-4 (net) | same |
| initialisations | A0 / A2e / A2s legacy coefficients; A1 `fit_to` the legacy provider; A2n `fit_to` A2s of the same seed; A4 / A5 and the A4t-A6 rungs `init_shower_long` on A2s; A7 / A8 copy of A2s with the family defaults; A9 `logit_dh` = -5; A10 family `init_from_target` on `A2s.core_map`, steps (1500, 1500), 400 search points; A7l `log_lam` + ln 20; A6b `sh_rk` degree 0; A12 last-ring scalar at 0 | same |

Nothing is missing, nothing is extra. Deviations in order only: A2s_s0 before A0 (finding C3).

**The rules.** Keep, equal NLL, replication, convergence, drift, parity and the P-lr decision are implemented as
ruled in section 3. The conditional runs are triggered by exactly the pre-registered conditions, each computed from
registered arrays of the live stream of this machine (tag and SHA-256 checked) or, for convergence, from the fit
history of this machine's run; a restart recomputes the same numbers from the same arrays. The order is the
order of the tables. Exception: the rules of the *report* ignore the reruns (B1).

**The guards.** G-K: K = 1024 arrays of run and comparator, paired; margin `guard - 0.5 main` with its paired SE
on the same 14,950 events; resolved at 2 SE. G-bulk: `Bulk(short="raise")` on the stratum zb >= 1221 mm, margin
against the report-metric difference of the **same** stratum events; the short events are their own row. G-mass:
paired per monitor. G-region: section 3, ruling 9. Recomputed from the stored arrays with numpy only (section 4).
Applying the guards to a superset (keep or equal NLL, on any seed, plus comparators) is harmless: guards select
nothing in Stage 1; the only side effects are more K = 1024 evaluations and a second registry row per target.

**Multiple testing.** The report keeps the distinction: one table for the six pre-registered poly comparisons, one
for the net column ("exploratory"), one for the exploratory rungs; secondary columns are labelled "never selection
criteria". Nothing in the driver lets an exploratory result into a primary decision: the learning rates come from
the base configurations, the reruns from the primary's own history, and the exploratory phase runs after the
primaries are finished. To add: C12 (labels in the last line, one sentence on multiplicity).

**Step-0 parity and drift.** Correct: the A0 step-0 array (the re-implemented `legacy_centre` in the registry
harness) is paired with the formula array of the reference set, which was computed by another code path
(`legacy.load_component_formula` and `common.test_nll`), so the check is not circular. Real mode, laptop, the
review's own reference set: -1.87e-8 ± 1.1e-8 nat, largest event 1.34e-5 -> PASS. Sufficient for what it claims:
model code, template, data identity and evaluation harness of the centre-rule model. Not covered by it, and by
nothing else that is pre-registered: the training loop (the drift rule sees only a gross failure: 2 units after
4000 steps from the optimum) and the exact-integration variants (covered by the earlier unit tests; A2e is
expected to start at +4.07 on the laptop's points).

**`fit_to`, Q1, Q2.** Section 3.

**Robustness.** Section 4 (kill and resume at six points - inside a fit, between fit and registration, inside the
registration, between registration and done-marker, inside the fit of a net run, inside the report -, non-finite
evaluation, smoke separation, Stage-2 gate) and findings C1, C2, C8, C11. In short: no double registration and no
silently skipped run were found; one diverging fit of any class stops the track and a restart does not get past
it (C1); the default phase list ends with the report and never contains Stage 2; smoke rows live in another stage
and the real report does not read them; all outputs go below the output root; `verify_sources` checks every
starting point against a registry row of this machine by file and weight hashes before the factory reads it.

**`dist_factory.py`.** The commit changes no model: template location per machine, the workaround replica removed
in favour of the fixed `fit_to`, two keyword arguments for the smoke mode whose defaults are the audit's values
(2000 / 500). All constant counts are unchanged and equal the audit's.

## 6. The base changes of commit `0c38b32` and the Track A driver

None of the five changes alters what the driver fits, registers or decides.
1. *Reference set only with a passing parity record.* The driver takes the set only through
   `registry.reference_set()`; a set without the record, or a disputed one, now raises at the first use (the
   parity phase). Checked: a set created in a fresh scratch root by `val_parity.py` and then
   `val_parity.py --create-refset` was accepted; the driver has no fallback.
2. *Stable machine id.* `run_m2a_lib.MACHINE_DIR`, the registry's directory, the done-markers and the fit's
   checkpoints now agree on one id whatever the per-call flags; outputs written under an older id are not found
   (a fresh output root was used here; on the H200 nothing exists yet).
3. *`registered_row(stream_id=None)` raises.* The driver always passes the live stream id.
4. *Fail-closed start-up off the laptop.* `run_m2a_lib.start()`, `train.fit` and `registry.evaluate` all go
   through it; on the H200 the command of STATUS section 4 (through the launcher) satisfies it. `--list` does not
   call the start-up (it computes nothing).
5. *Launcher liveness, `stop_job.sh`.* The stop helper sends SIGTERM; the driver has no handler, so a stop is a
   kill at an arbitrary point, which is the case tested in section 4. `verify_sources`, the resume logic and the
   use of the reference set are unchanged; the three test files pass on this base (82 passed, 0 skipped).
