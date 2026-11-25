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
**Addendum: section 7, re-check of commit `ee462fc` (the fixes for B1-B5 and C2) - GO for starting the real run at
`ee462fc`.** Sections 0 to 6 are as written for `c4f39fd` and were not changed by the addendum.
**Second addendum: section 8, re-review of commit `a05c86e` (A2c control, scale floor, divergence handling) - GO
for `warmstart-check warmstart-rerun a2c report` after the present job has ended.**

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

## 7. Re-check of commit `ee462fc` (fixes for B1-B5 and C2)

25 Nov 2026, same reviewer, same rules as the review above (read-only; scratch output roots with a reference set
made by the two-step `val_parity.py`; smoke mode only; no test split; no real Track A fit).

Subject: commit `ee462fc`. Against the reviewed commit `c4f39fd` three files of Track A differ:

| File | SHA-256 in `ee462fc` |
|---|---|
| `scripts/run_m2a.py` | `d04e6cb95dc91c60f9508443ea06c647d0085f06a48473268cdaada11aa2b411` |
| `scripts/run_m2a_report.py` | `8a1149981bb3eab04756250934bd9f0c768e5303f45ab7d59d922f72749ebc99` |
| `tests/test_run_m2a.py` | `91554e26d4c4dcc60cbf2abbaa11f9159e4b489fa5116e20860e2c6b8c597be2` |

`scripts/run_m2a_lib.py`, `src/sfm/models/paramnet.py`, `src/sfm/models/dist_factory.py`, `tests/test_dist_factory.py`
and `tests/test_models.py` have the SHA-256 of section 0. In the whole repository, between the base of the review
(`0c38b32`) and `ee462fc`, only these files changed: the two scripts, `tests/test_run_m2a.py`,
`tests/test_provenance.py` and four documents. No base module, no family, no configuration; `common.py`,
`results/registry.jsonl` and `results/test_access.log` have the hashes of section 0 before and after this re-check.

### 7.1 Starting the real run at `ee462fc`: **GO**

Nothing that a phase up to and including `guards` fits, registers or decides differs from the reviewed commit.
(Commit `247017d`, where the five Track A files equal `c4f39fd`, stays GO as well; it would produce the same fits
and rows.)

| Check | Result |
|---|---|
| diff of `run_m2a.py`, read line by line | three hunks and no other line: `phase_report` passes one argument (`with_guards`; the default is the full report as before); the new `stage2_approval`, used by `phase_stage2`; `main()` gets the option `--without-guards`, refused unless the phase list is exactly `report` |
| the same, function by function (syntax trees of both versions) | 59 top-level items identical, among them every spec builder (`plr_specs`, `stage0_specs`, `primary_kwargs`, `primary_specs`, `exploratory_spec`, `rerun_specs`), every rule (`keep_rule`, `equal_rule_one`, `converged`, `_plr_decide`), `phase_templates`, `phase_parity`, `phase_plr`, `phase_stage0`, `phase_stage1p`, `phase_stage1x`, `guard_targets`, `phase_guards` with its helpers, `stage2_specs`, `listing`, all thresholds and `DEFAULT_PHASES`; changed: `main`, `phase_report`, `phase_stage2`; new: `stage2_approval` and three constants |
| default invocation | without `--without-guards` the new lines of `main()` are not entered; phase list and order unchanged; `stage2` still refused in any list but its own; `--without-guards` without a phase list is refused, so the default run cannot be started with it |
| the report module | imported only inside `phase_report`; it writes `results*.json` and `RESULTS*.md` and nothing else (registry of the scratch root byte-identical before and after a report) |
| `run_m2a.py --list` | byte-identical to the listing of `c4f39fd` (SHA-256 `8762ba00...`, as the worker states) |
| tests | `tests/test_dist_factory.py tests/test_run_m2a.py tests/test_models.py`: 88 passed, 0 skipped |
| execution: a fresh smoke run with the code of `ee462fc` (second scratch root, own reference set) against the smoke run of the reviewed code (section 4) | 58 registry rows without guards on both sides, same names (the two rows of the forced rerun of section 4 set aside): SHA-256 of the validation array, SHA-256 of the weight tensors, constant count, configuration hash, seed and comparator identical in all 58. The K = 1024, bulk and mass-monitor arrays of the six guard targets: identical. `plr.json`: identical (lr_poly 0.003, lr_net 0.01). The decisions `parity`, `A0_drift`, `convergence_primaries`, `exploratory_seed1`, `dropped`, `budget_rule`, `guard_targets`: identical. The default command (no phase list) at `ee462fc` ran `templates ... guards report` in that order and ended with 55 finished runs and 64 rows, no duplicates, as the reviewed code did |

Why the execution check is decisive: smoke training is bit-reproducible on this laptop - the worker's dry run and
the first smoke run of this review, two independent sessions, agree in all 60 common rows in the SHA-256 of the
validation array and of the weight tensors. Any change in what the new code fits or registers before the report
would show as a different hash.

What is new at the end of the default run: the report executes more code than the reviewed one (it rebuilds the
target model of every net warm start on the CPU, reads the template files, prints more tables). All fits, guard
evaluations and registry rows are complete before it starts; if it failed, `run_m2a.py report` can be repeated at
any time and nothing is lost. The warm-start recomputation catches its own errors per run.

### 7.2 The findings of section 2

| Finding | State | How it was checked |
|---|---|---|
| B1 reruns | **closed** | On the smoke state with a forced rerun of A7_s0 the new report uses the pair `A7_s0_x2` vs `A2s_s0_x2` for seed 0 and the original for seed 1 (column "pairs used by the rules": "rerun / original"), prints the original as "A7 s0 original - not converged, NOT used by the rules", and records `replication.pairs`. The two-seed mean of the mixed pairs (+0.42 ± 0.18), every other Stage-1 number and the keep / equal / replication flags were recomputed with numpy only from the arrays named in `results.json`: identical (37 comparisons, 5 guard blocks). |
| B2 initialisation | **closed** | Table of 57 rows on the smoke state: `kl_map` and its reference for every run; "exactly A2s: yes" for A7, A7s, A8, A8c, A78, A12 and the A7 rerun (\|kl_map\| <= 3.4e-8); all 14 smoke net runs flagged (their shortened warm start gives 0.008-0.09 nat, as expected). Variation: a target model with one momentum-dependent row flattened to a relative spread of 1.0e-4 is listed as nearly constant; flattened to exactly constant it is listed as constant. |
| B3 alternatives | **closed, with remarks R1 and R2 on the labels** | Both alternatives are printed and decide nothing in Stage 1; the two constructed cases of 7.3 show what a reader can misread. |
| B4 template limitations | **closed** | Printed below the exploratory table; the Q2 numbers are computed from the machine's own template files and equal section 3 (79 cells from 1379 mm, maximum 1.81, denominator 0.186; 906 above 1, 56 above 1.1). The Q1 sizes are the laptop diagnostics of section 3, quoted and labelled as such. |
| B5 Stage-2 gate | **closed for the interlock; Stage 2 itself still needs its review** (`stage2_specs`, the missing A11p) | Gate function called directly on the smoke state (no Stage-2 fit started): refused without an approval file; with a stale hash; with A7 added to the approved list; with an unknown reading; without `approved_by`; after `results.json` was changed by one byte; after `results.json` was hand-edited. Accepted: the valid approval (also under the reading `two_se` with the list of that reading). The command `run_m2a.py --smoke stage2` refused without and with a stale file and created no A11 directory. Two limits: remarks R6 and R7. |
| C2 table without guards | **closed as an operator option, not as an automatic step** | `run_m2a.py report --without-guards` writes `RESULTS_without_guards.md` / `results_without_guards.json` with the banner and leaves `results.json` byte-identical; refused with any other phase list. The default phase list is unchanged, so after a guards failure in the unattended run a table exists only once someone issues that command (or the full `report`, which now lists the pairs whose guard data is missing). |
| also: C12, B6 (i) | done | multiplicity sentence, kept line split into primaries and exploratory, footnote on the seed order, convergence movement of A2s_s0 / A2s_s1 beside the primaries table |
| not implemented, awaiting the user (unchanged) | open | the control A2c (B6 ii); a scale floor in `fit_to`; record-and-continue for a non-finite evaluation (C1: the run stops there and waits) |

### 7.3 Remarks on the new report (for whoever reads the Stage-1 table; none blocks the start)

- **R1. The column "alternative - decides nothing: two-seed mean <= -2 SE" shows that one condition, not the verdict
  of the alternative reading.** Constructed case: both seeds at -2.0 ± 0.1 - not kept, not replicated - give
  "replicated: no" and "alternative: **yes**", while the alternative reading (literal conditions *and* 2 SE) says
  no. The verdict of the alternative reading is only in the last section ("kept and replicated ... alternative").
  Read the column as "the mean is 2 SE away from zero", never as "replicated under the alternative". Fix: print
  `replicated_alternative_2se` in the column, or head it "also <= -2 SE (in addition to the literal conditions)".
  The words "decides nothing" hold for this report; in the Stage-2 approval the user may choose that reading.
- **R2. The G-region alternative "would pass" does not require a gain.** It tests the hit term alone
  (`run_m2a_report.py:235-236`). Constructed case: total +1.0 (no gain), hit term -1.0 with the flagged cells at
  -0.2, normalisation +2.0: literal "passed no", alternative "would pass **yes**". Fix: require the total gain to
  be negative there as well, or print "share condition only".
- **R3. A full report made before the guards phase has run says "No run passed the keep or the equal-NLL rule: no
  guard columns".** Seen by execution: in the second scratch root, after `stage1x` and before any `guards`, `run_m2a.py report` printed that sentence although rungs pass the equal-NLL rule there (A7, A9, A9r and five net versions). In the default sequence the guards phase always precedes the report, so this needs a
  report issued by hand while fits are running or after a stop; then the sentence is wrong. Use
  `report --without-guards` for an interim look. Fix: when `decisions.json` has no `guard_targets`, print "the
  guards phase has not run".
- **R4. Guard blocks of an original that the rules do not use are not labelled.** With a rerun, both
  `A7_s0 vs A2s_s0` and `A7_s0_x2 vs A2s_s0_x2` get a block; only the second belongs to the pair the rules use
  (the secondary columns do carry the label).
- **R5. `rms_max` in the Initialisation table is absolute, in raw units of the worst column.** For the healthy
  warm start of A1 it is 0.36 (column `zs`, spread 32.5, i.e. 1.1 %); the number alone looks alarming. The flag
  and `kl_map` are what to read. The row marks "(template note Q1/Q2)" stand on A3x too, although Q2 concerns
  A3c only.
- **R6. (before Stage 2) The gate binds the approval to the file, not the file to the arrays.** A `results.json`
  that was hand-edited (A7 set to kept) together with a new approval naming its hash is accepted. Since the full
  report is byte-stable (two runs on the same state gave the same SHA-256), `phase_stage2` can recompute the
  report in memory and require equality with the file on disk; then only a table that follows from the registered
  arrays can be approved. Corollary: STATUS says any re-run of the full report makes a new approval necessary;
  that is so only when the table changes.
- **R7. (before Stage 2) An approval without the key `kept_and_replicated` is accepted when nothing is kept.**
  The missing key counts as the empty list. Require the key.
- **R8.** `results.json` does not record the commit of the report code; with fits and report meant to be one
  commit, add `machine.git_state()` to it.

None of R1 to R8 changes a number. R1 to R5 are labels; if they are fixed after the start, the report code is a
later commit than the fits (the report is a deterministic function of the registered arrays and can be re-run).
If fits and report shall be one commit, this section is the reading note that goes with the table of `ee462fc`.

### 7.4 What this re-check did not do

Nothing ran on the H200 and no real-length fit exists; the lists "by reading only" and "only on the H200" of
section 4 stand. Stage 2 was not started (the gate function was called; no A11 fit). The new report was exercised
on smoke rows only: its real-run inputs (converged fits, real warm starts, the H200's templates) are first seen on
the H200.

## 8. Re-review of commit `a05c86e` (A2c control, scale floor, divergence handling)

25 Nov 2026, same reviewer, same rules as before (read-only; scratch output roots with reference sets made by the
two-step `val_parity.py`; smoke mode only; no test split; no real Track A fit). Decisions of record: PLAN 17.11.
The worker's account: STATUS section R3. Code at `HEAD` (`aee67a6`) equals `a05c86e` in every script, module and
test. The real job runs on the H200 from `5ddfd9f`, whose five Track A files are those of `ee462fc` (section 7).

| File | SHA-256 in `a05c86e` |
|---|---|
| `scripts/run_m2a.py` | `0a38533507ef7ccb09df5230bf27fd57b1c111c74d743257f1401b28ccff6a9b` |
| `scripts/run_m2a_lib.py` | `e65f1478ee36346c3dc567adf58ed016fb35ff115bb9e52111ab44026354a4ab` |
| `scripts/run_m2a_report.py` | `9acb49a4717fe8af8f4135488e08543dc80c7c0948c3f741d92f2861d85b42a7` |
| `src/sfm/models/paramnet.py` | `1cd02599d66b7f02397e551543e47de1847fabbb3987b90ec401978dd3737d1f` |
| `src/sfm/models/dist_factory.py` | `09612b3fe25e5c8ac7aa7dc7ec4fee08087bec5bd53602e173e28bb964112b8d` |

`common.py`, `results/registry.jsonl`, `results/test_access.log`: hashes of section 0, before and after.

### 8.1 Verdict

**GO for `run_m2a.py warmstart-check warmstart-rerun a2c report` on the H200 at `a05c86e`, after the present job
has ended. Nothing must be fixed first.** The command leaves every run of the present job untouched (8.2), applies
the PLAN 17.11 rule as written (8.5), and A2c is the control proposed in B6 (8.3).

Conditions: (i) pull only after the job's process has exited (8.2, last row); (ii) the findings of class (B) below
before Stage 2 - none of them concerns the planned command's numbers.

### 8.2 Hard constraint: the runs of the present job

| Check | Result |
|---|---|
| run list | `--list`: the 102 lines of the reviewed listing are byte-identical (SHA-256 `8762ba00...` for that part); the additions follow in a block of their own. `listing`, every spec builder, the rules, the thresholds, `DEFAULT_PHASES`, `main`, `fit_config`, `Spec`, `verify_sources`, `guard_targets`, `phase_guards`, `phase_parity`, `phase_templates` are identical syntax trees |
| what changed in existing phases | `_plr_decide`, `phase_plr`, `phase_stage0`, `phase_stage1p`, `phase_stage1x`: branches that are entered only for a run without a registry row (diverged or blocked); with every run registered they compute and record what the reviewed code did. `stage1_comparisons`: maps a superseded network run to its re-run. `run_spec`: a done run is returned at its `done.json` before any new line |
| execution: the planned command on a copy of a finished smoke state made by the code of `ee462fc` | exit 0; the 64 registry lines of the old state byte-identical, 16 appended (14 re-runs, 2 A2c); `done.json`, `model.pt`, `init.json`, `fit_done.json` of all old runs unchanged; all old tagged arrays unchanged; the old keys of `decisions.json` unchanged, four new keys; all Stage-1 comparisons of the polynomial rungs (pairs, deltas, flags, two-seed means, kept lists) identical in the new report |
| the same command a second time | no fit started, registry unchanged, decisions identical, `results.json` byte-identical |
| the existing phases `plr stage0 stage1p stage1x` issued with the new code on that finished state | no fit started, registry unchanged, `decisions.json` and `plr.json` identical |
| would new code fit differently? (only relevant if the default list were re-issued with it) | a polynomial run retrained by the new `run_spec`: same weights and validation array as the old code's row; the 14 network re-runs (fit_to version 3, step-0 parity evaluated before training): bit-identical to their originals |
| comparators | not changed for any existing run. Where a re-run exists, tables and guard pairs name it explicitly (`A4n_s0_ws vs A2n_s0_ws`), and the original pair stays as a labelled row |
| **pulling while the job runs** | **do not.** The report module is imported only when the report phase starts: a pull before that makes the old process load the new report, which calls functions the old library in memory does not have - the report phase would end with an error (fits and rows intact, no table). And every row registered after the pull (guards phase) would record the new commit although old code computed it |

### 8.3 A2c

- *Definition.* `A2c_s0`, `A2c_s1`: the structure of A2s, coefficients copied from the fitted A2s of the same seed
  (`kl_map` 0.0 in smoke), T_A with 4000 steps, the frozen `lr_poly`, the seed of the rung (hence the same batches
  and fold points as the rungs of that seed), fresh optimiser and cosine cycle; class `control`; comparator A2s of
  the same seed. **This is the control of B6 (ii), and it is what "trained for the matched longer step budget"
  means:** A2s plus exactly what every matched rung gets on top of A2s. (A single 8000-step cycle from the legacy
  coefficients would be the other reading; it would not match the optimiser restart that the rungs have.)
- *No rule consumes it.* Keep, replication, equal NLL, "cleaner", the kept lists, the guard targets and
  `stage2_specs` read the comparison against A2s (or the rerun pair) only; A2c appears in its own section, in the
  column "control: Δ vs A2c" and in the Stage-2 precondition "A2c finished". The section says "No pre-registered
  rule consumes it ... decides nothing" and the report carries "NOT FINAL" while A2c is missing. The column stands in
  the table of the pre-registered comparisons; its header should say "(decides nothing)" as the alternative column
  does (C).
- *What it shows (smoke).* A2c_s0 - A2s_s0 = +0.51 ± 0.24: the offset that section 2 (B6) found common to the six
  rungs starting exactly at A2s (+0.50 ± 0.24) is the continuation. Against A2c those rungs are at +0.03 ± 0.07
  (A7), -0.01 ± 0.04 (A8), -0.01 ± 0.02 (A7s), +0.01 ± 0.06 (A78), -0.08 ± 0.05 (A12): the control comparison is
  also two to ten times sharper than the one against A2s.
- *Are A2c6k and A2c_x2 needed?* **Not for the 19 matched rungs, and not unconditionally.** Rule proposed here,
  before any Track A result has been read by the reviewer: **A2c_x2** is needed if and only if the rules use a
  doubled-step rerun pair for some primary (then that row has no matched control: 4000 + 8000 steps against a
  comparator of 8000); **A2c6k** is needed if and only if a mixture rung (A10, A10-2, A10-4, A10-5, A10g) passes the
  keep rule or the one-seed equal-NLL rule (a mixture far worse than A2s needs no control for its training budget;
  it also does not continue A2s's weights, so the control would bound, not match, its budget effect). In either
  case the control runs before the content of A11 is decided. Both are additions to the list: **[confirm]**.
  Until then the report's "not matched (...)" in those cells is the correct statement.

### 8.4 Scale floor (`fit_to` version 3) and step-0 parity

- *Mathematics: correct.* For a varying column the weight is 1 / max(s_j, F)^2, F = 1e-2 in raw units, s_j the
  float64 standard deviation over the grid; the detection `s_j < F` and the constant-column rule are disjoint;
  columns with `s_j >= F` keep the float32 `1 / var_j` of version 2 through an unchanged tensor operation, so
  nothing changes for them; the weight is continuous at `s_j = F`.
- *An absolute floor is the right kind.* Reproducer `<scratch>/fit_to_v3.py` (CPU, the pre-registered call; six
  ordinary columns plus one extra column): worst ordinary column, as a fraction of its spread -

  | extra column: std (raw units) | at mean 2: version 2 / version 3 | at mean 0 | at mean 46 |
  |---|---|---|---|
  | 2e-6 ... 2e-3 (below the floor) | 17 %, 15 %, 14 % / 1.3 %, 0.5 %, 0.7 % | 11 % / 0.4 % (std 2e-3) | 14 % / 1.8 % (std 2e-3) |
  | 5e-3 (below) | 4.3 % / 1.9 % | | |
  | 1e-2 (just above: not floored) | 1.6 % / 1.6 % | 0.7 % | 1.8 % |
  | 2e-2, 5e-2 | 0.7 % (2e-2) | 0.3 %, 0.2 % | 1.1 %, 0.9 % |
  | no extra column | 0.04 % | | |

  The damage follows the absolute spread, not the spread relative to the mean (same picture at mean 0, 2 and 46):
  the optimiser's step acts in raw units. The user's absolute form is therefore better founded than the relative
  one suggested in section 3. With the floor the worst case is of the size of the healthy A1 warm start (worst
  column 1.1 %, `kl_map` 5.5e-6 nat). Units: of the 22 momentum-dependent legacy columns, 21 are logs, logits and
  similar unconstrained arguments by their names (|mean| below 10, std 0.10 ... 3.8), where 1e-2 is about one per
  cent of the constrained quantity; `zs` (mean 46, std 32.5) is a length, for which the floor is practically
  inactive - consistent with the table: what counts is the spread in the units in which the network's output
  layer works, and a column with std above 1e-2 is outside the badly damaged range whatever its mean.
- *"Bit-identical for the existing targets": true where no column lies below the floor* - by construction, in all
  nine such cases of the reproducer (state dictionaries equal), in the worker's test over the eight net models at
  their starting coefficients, and in the execution of 8.2 (14 of 14 re-runs bit-identical). Whether a **fitted**
  target of the job has a column below the floor is what `warmstart-check` finds out on the H200.
- *Step-0 parity: correct.* The step-0 network is scored with the registry's computation (unmodified
  `common.test_nll` inside `deterministic()`, same points), saved as a tagged array (not a registry row) and paired
  through `provenance.paired_delta` with the registered array of its source: `net_from` for every run fitted to a
  polynomial run (pilots and A2n: A2s of the seed; net versions: their polynomial primary), `A0_init` for A1 (the
  legacy coefficients at step 0). Seen in the logs of 8.2 for all 14. It does not disturb the training (re-runs
  bit-identical).

### 8.5 `warmstart-check` and `warmstart-rerun`

- *The rule is PLAN 17.11, verbatim, and nothing else:* `warmstart_affected` returns (a) "a varying target column
  has std < 1e-2" - varying by `fit_to`'s own constant rule, std in float64 on `default_lnp(48)` with uniform
  weights, target rebuilt from the source run - or (b) "recorded `kl_map` > 1e-4 nat", read from the run's
  `init.json`. The rebuilt warm start and its step-0 parity are reported and do not enter the decision.
- *The rebuild is faithful:* same seed, same order (seed, context, build) and same arguments as `run_spec`, with
  `fit_to_scale_floor=None`; judged by equality of the rebuilt and the recorded `kl_map`: 14 of 14 on a state made
  in another session by the code of `ee462fc`, and again identical on the second issue.
- *Cascade.* Re-runs are named `<name>_ws`; pilots first; the fit target stays the (never superseded) polynomial
  run; a net version is compared with `A2n_s0_ws` when A2n_s0 was re-run (order guarantees it exists); `A1_ws` with
  A0, `A2n_ws` with A2s; the guards phase runs on the pairs with the superseding names; conditional second seeds
  and convergence reruns do not depend on a network run. It cannot loop: only runs made with a `fit_to` before
  version 3 are checked, re-runs never. Originals keep their rows and files, are labelled "superseded: warm start"
  in the Stage-0 table, the net column and the Initialisation table, and leave the rules through one mapping
  (`decisions.json: warmstart_rerun.superseded`).
- *If a re-run pilot changes `lr_net`,* every other network run is re-run with the new rate ("dependent"). That
  goes beyond "if and only if (a) or (b)" and is **[within]** the pre-registration all the same: P-lr freezes one
  rate for all network runs. Two remarks: B8-1 and C8-2 below.

### 8.6 Divergence handling - rulings on the worker's definitions

| Definition | Ruling |
|---|---|
| (i) a non-finite validation mean anywhere in the fit history | literal to "non-finite value", **but not conservative in one corner: a fit whose validation was non-finite once and that ends finite and good is declared diverged, is not registered, and is terminal** (checked: history NaN at step 500, then 18.121, 18.1205, 18.1203 -> diverged). Proposed: (i) applies to the last value or to a fit that did not recover; an earlier non-finite value is reported with the run like isolated skipped batches. **[confirm]**; it cannot matter for the planned command unless a re-run or A2c shows it |
| (ii) non-finite per-event value at registration | **[within]** - the case of C1 |
| (iii) "frozen": at least half the batches of the last evaluation window skipped | **[within]**; unambiguous (window = the last two history points); a healthy fit has no skipped batch |
| (iv) "blow-up": last value more than 1 nat above the lowest of history, step-0 value, comparator | **[within]** as a threshold: the worst smoke fit after 60 steps is 0.05 nat above A2s. Property to know: against the comparator it would also label a stable but very bad model "diverged" (checked with a constructed history); no model of the list is expected within a factor 20 of that |
| classes: pilots lose; exploratory rungs and the net column "diverged", no second seed; primary, base, control, rerun "diverged - awaiting diagnosis", no rule marks them failed; dependents "blocked"; the independent rungs finish | **[within]** the user's decision. A diverged A0 seed makes the drift rule "not evaluable" and the list continues: the user's wording ("for primary comparisons and controls the independent rungs still finish"), although A0 is the control behind a pre-registered stop; in the present job A0 is finished |
| Stage 2 refuses while a diagnosis is pending | **sound** - it only adds caution; see B8-2 for two cases in which it can never be lifted |

Not dropped silently: a diverged or blocked run appears with its status in the primaries, net, exploratory,
Stage-0, secondary and P-lr tables and in the section "Diverged and blocked runs" with its record; it has no row in
the convergence and Initialisation tables (those list finished runs). Exercised in 8.8.

### 8.7 Findings

**(A) before the planned command:** none.

**(B) before Stage 2:**
- **B8-1. `plr.json` keeps the old `lr_net` when the re-run pilots change it.** `phase_warmstart_rerun` records
  the new rate in `decisions.json` and passes it to the re-runs, but `frozen_lr("net")` - which every later network
  run reads, A11n of Stage 2 among them - still returns the old one; and `stage2_specs` names `A2n_s<seed>` as
  comparator, not its superseding run. Only if `lr_net` changes; to be fixed with the Stage-2 review of B5.
- **B8-2. Two states that Stage 2 can never leave.** (i) A warm-start re-run that diverged has no registry row, so
  "warm-start re-runs missing" is raised for ever (for the net column there is no diagnosis path at all). (ii) A
  diverged A2c seed keeps "A2c is not finished" even after a diagnosis (seen: refused with exactly that message;
  the report then says "NOT FINAL ... run a2c", which cannot help either). Both need a divergence in the planned
  command to occur; the gate should accept "diverged and diagnosed (or exploratory)" as finished.
- **B8-3. Rule lines for a primary with a diverged seed.** The kept line and "equal NLL (both)" print "no" for it
  (`kept_and_replicated[r]` and `equal_both` are False when a seed has no comparison). "No rule marks it failed"
  asks for "not evaluable" there. Cannot occur in the present job's primaries (all finished).

**(C) not blocking:**
- **C8-1.** Divergence criterion (i): ruling above.
- **C8-2.** If the pilots are re-run under rule (b) alone (no column below the floor), version 3 does not change
  their warm start and the re-runs are replicates; `lr_net` is then decided a second time on replicates. On the
  laptop replicates are bit-identical (14 of 14), so nothing can change; should the H200 not train bit-reproducibly,
  a near-tie could flip the rate and re-run the whole net column. Read the log line "lr_net after the warm-start
  re-run"; a change without a below-floor column in the pilots' target is not caused by the fix and goes to the
  user before the re-runs are used. Simplest guard: decide `lr_net` again only when a pilot was re-run under (a).
- **C8-3.** Tagged step-0 arrays are written once: a restart of a network run killed between its step-0 array and
  its first checkpoint, and every re-issue of `warmstart-check`, need the warm start to be rebuilt bit for bit
  (true on the laptop, twice). Otherwise `save_tagged` refuses loudly; the run or the check then needs the stale
  array moved aside by hand.
- **C8-4.** A re-run that diverges still supersedes its original (`sup[name]` is set whatever the status): the
  rules then see "diverged" where the original had numbers (seen with the injected case: all six net rows lost
  their numbers to "comparator A2n_s0_ws: diverged"). Visible in the table; arguably right (the original's warm
  start was flagged), but it should be said in the superseded row.
- **C8-5.** A2c is not in the "exactly A2s" claims of the Initialisation table (its `kl_map` is printed: 0.0).
  The header of the control column should carry "(decides nothing)".

### 8.8 R1 to R8 of section 7

| Remark | State |
|---|---|
| R1 alternative column | closed - the column is `replicated_alternative_2se`, headed "replicated under the alternative reading (literal conditions AND two-seed mean <= -2 SE; ...)" |
| R2 G-region alternative | closed - requires a gain in the total, and says so |
| R3 report before the guards phase | closed - "The guards phase has not run" when no guard targets are recorded |
| R4 guard blocks of unused originals | closed - "(original, NOT the pair used by the rules)", also for superseded network runs |
| R5 `rms_max`, template marks | closed - printed relative to the column's spread; "Q1, Q2" on A3c, "Q1" on A3x |
| R6 gate bound to the arrays | closed - `stage2_preconditions` recomputes the report; a hand-edited `results.json` with a matching new approval is refused ("differs from the report recomputed from the registered arrays (keys ['kept_and_replicated'])"); the valid approval on a complete state is accepted (function calls only, no Stage-2 fit) |
| R7 approval without the list | closed - refused |
| R8 commit of the report code | closed - `results.json` and the header of `RESULTS.md` carry commit and clean flag; the gate's comparison therefore also binds Stage 2 to the commit of the report |

### 8.9 What was verified, and how

*By execution (laptop, smoke stage, scratch roots):* tests 110 passed, 0 skipped; the listing; the planned command
on a copy of an old-code state, twice (8.2); one polynomial run retrained by the new code; the planned command with two divergences injected through the smoke-only hook - a re-run base (`A2n_s0_ws`) and one control seed (`A2c_s1`): both recorded as "diverged - awaiting diagnosis" with their records, the command ran on to its report (exit 0), the six net versions were trained and are shown as "comparator A2n_s0_ws: diverged - awaiting diagnosis", the report carries "NOT FINAL" and the diverged section, a second issue started no fit, Stage 2 was refused; the new report on the earlier state with a doubled-step rerun ("rerun / original", control "not matched (doubled-step rerun)", all numbers again equal to the numpy-only recomputation); the Stage-2
gate (valid, missing key, forged table); `fit_to` versions 2 and 3 on 15 constructed targets; the divergence
classifier on constructed histories; the syntax-tree comparison of driver and library with the running commit.

*By reading only:* the branches of `phase_plr`, `phase_stage0`, `phase_stage1p`, `phase_stage1x` for diverged or
blocked runs in a default list (the worker's fresh-root run exercised them; not repeated here); the `lr_net`-changed
branch of `warmstart-rerun` (never taken in smoke: the rate did not change); Stage 2 itself.

*Only on the H200:* whether a fitted target has a column below the floor; whether rebuilt warm starts and
replicate trainings are bit for bit as on the laptop (C8-2, C8-3); the real `kl_map` values and step-0 parities.
