# Track A (M2A) - status, 25 Nov 2026 (item E4 of PLAN.md section 16.7)

## R6. Stage 2 (REVIEW.md sections 10, 11), the user's decisions of PLAN 20 and 23, and the guard timing record of PLAN 21 - written, waiting for the final re-check

Nothing was trained for real; everything below ran in smoke mode, in scratch output roots on the laptop
(`<scratch>`) and in fresh output roots of the H200 test copy (slot `ta`). The H200 state of record was read only (`results/machines/h200-g1056-97b197a7/`, copied back). Base modules touched: only
`src/sfm/models/dist_factory.py` (the new `prune` argument of A11p, REVIEW 11.5; default unchanged).

### R6.1 What each requirement became

| Item | Where | What it does |
|---|---|---|
| PLAN 21 timing | `run_m2a.guard_run`, `run_m2a_lib.PartTimer`, `host_state` | per guarded run and part (registry evaluate total; K = 1024 + bulk; mass monitors; the K = 64 rest; G-region GPU pass; G-region host work): wall and CPU seconds, page faults, load, MemAvailable, swap, RSS, PSI avg10 and stall seconds of the part; logged, and written to `guard_timing_<UTC>.json` beside the run's arrays; a part over 60 s logs `WARNING: guard part '...' of <run> took N s (> 60 s): ...`. Timing fields only, not in decisions.json or results.json; nothing moved to the GPU |
| PLAN 20 item 1 / conditional controls | `evaluate_conditional_controls`, phase `conditional-controls` | A2c_x2 only if a doubled-step rerun pair (`<rung>_s<seed>_x2` vs `A2s_s<seed>_x2`) passes keep or one-seed equal NLL; A2c6k only if a mixture rung (A10, A10-2/4/5, A10g) passes either against A2s of its seed; recorded with every evaluated pair's numbers (decisions.json `conditional_controls`, excluded from results.json); trained only if triggered (A2c_x2_s<seed>: A2s from A2s_s<seed>, 8000 steps, vs A2s_s<seed>_x2; A2c6k_s<seed>: 6000 steps, vs A2s_s<seed>); the gate requires the record to equal its re-evaluation and triggered controls to be resolved; printed in the Stage-2 report. On the H200 table: no rerun pair exists, every mixture is worse by +20 to +116 -> neither triggers |
| PLAN 20 item 2 / R3 | `DIVERGENCE_DEFINITION`, `divergence_check`, `recovered_nonfinite`, `run_spec` | criterion (i) is now a non-finite FINAL validation mean; a non-finite value earlier in a fit that recovered registers the run with a WARNING, the count and steps in done.json (`warnings`) and decisions.json `warnings`; listed for Gate 1 in the Stage-2 report together with a re-classification of every finished run from its stored history. Frozen last window and blow-up stay terminal. The Stage-1 report keeps printing the definition of 25 Nov (`DIVERGENCE_DEFINITION_25NOV`) under which its runs were classified, so that its RESULTS.md stays byte-identical |
| PLAN 20 item 3 / R3 | `phase_warmstart_rerun`, `frozen_lr`, `superseded`, `lr_net_escalation_pending` | if re-run network pilots change lr_net and no re-run pilot had a target column below the floor (rule a), the change is escalated: recorded `escalated`, `status: awaiting the user`, `not_used: [pilots]`; no dependent re-run; the rate of record stays (frozen_lr ignores it, superseded() drops the pilots' re-runs); Stage 2 refuses until the approval carries `lr_net_escalation: {decision: keep_rate_of_record, date, reason}` (any other decision is implemented by hand). REVIEW 9.5 Rm1: the comparator of a re-run and the second lr decision use the re-runs in force (registered), not the recorded map |
| R1 | `stage2_approval`, `stage2_content_check`, `check_stage2_decisions` | the approval carries `a11_content`, `excluded` (rung -> {reason, date}), `waivers` ([{rung, guard, class, basis, subject_to, date}]); refused unless the content is the computed kept-and-replicated list minus the exclusions (content within the list; every other rung excluded with a reason; no exclusion outside it), the content merges, every guard failure of a content rung (G-K, G-bulk, G-region on the pairs the rules use) has exactly one waiver of that guard whose flagged* line covers it, and no waiver lacks such a failure. Only G-region / low_abar can be waived. Content, exclusions, waivers and the flagged* lines go to decisions.json (`stage2_content`) and the head of the Stage-2 report |
| R2 | `stage2_specs` (unchanged), `memory_probe_k64` | A11 (0, 1, 2), A11k (0, 1), A11n (0, 1) as coded. Before the first k_train = 64 fit: one training step of the fitted A11 at k_train 64 on the 256 training events with the most hits, peak allocated memory against 90 % of the job's cap (`sfm.machine.caps`; device memory without a cap); if it does not fit, `stage2` stops with the numbers |
| R4 | `phase_stage2_guards`, `guard_run` | guard rows, G-region and flagged* arrays and the timing record for every Stage-2 run and its comparator (A11 / A11k vs A2s, A11n vs the A2n in force, A11p vs A11); existing rows, arrays and markers are reused, none rewritten |
| R5 | `run_m2a_report.report_stage2` | RESULTS_stage2.md / results_stage2.json: the comparisons of REVIEW 10.4 on all events and the strata (secondary columns, short and bulk), A11 - A9 reproducibility (max per-event difference, bit for bit), the finalist rule (one_se_select per seed, eligible on both, fewest constants; A11n never a candidate, PLAN 23 item 6: the line "ESCALATE TO THE USER AT GATE 1" is printed when A11n beats every polynomial candidate by more than 1 SE on both seeds), "cleaner" (c1) and (c2) for every Stage-2 rung against the formula array with the literal sentence when the finalist is not cleaner, the A11p screen tables and acceptance, the A11n initialisation, convergence, the guards with the waiver line, the conditional controls, the warnings and the re-classification. Stage-1 results.json / RESULTS.md: not touched, and the gate's recomputation still passes after Stage 2 (a re-issued `stage2` starts no fit and is not refused) |
| R6 | `g_region_star_arrays` (subset `g_region:flagged_star`, marker `g_region_star.json`), `g_region_star_check` | flagged* = deep, last ring, first / last z-bin by hit centre. The marker also records whether the flagged class recomputed in the same pass equals the stored one bit for bit, and the identity flagged = flagged* + (low_abar and not flagged*). G-region is printed as registered (FAILED) and the waiver line separately; it never says "passes"; a failing flagged* line is "NOT covered by the waiver" (and the gate refuses) |
| R7 / A11p | `a11p_screen`, `a11p_select`, `second_difference`, `curvature_autodiff`, `a11p_specs`, `a11p_run`, `dist_factory._prune_rows` | exactly REVIEW 11: candidates k >= 1 per row from the top (58 asserted for the content A9); H_j by the central second difference (h = max(0.1 |c|, 1e-3), realised float32 steps), Q_j = 0.5 |H_j| c_j^2 < 0.2e-4 nat, the short-event condition f_tr Q^S_j < share x 0.2e-4; pass on both seeds; Z_0 and Z; sum of Q, J and J^S per seed; an autodiff check of H on 256 training events for the three most resolved candidates; written to `a11p/screen.json` and hashed (decisions.json) before any fit, never overwritten; the fits (A11p_s0/s1: A11 procedure with `prune`, comparator A11_s, step-0 rows) refuse any other pruning set; their guards; acceptance (equal NLL; U1 with M = share x 1e-4 / f; U2 N_s <= 95 of 166) in the Stage-2 report with the sentence of 11.7. The two [confirm] choices are recorded parameters (`a11p_parameters` in the approval: `short_event_share`, default 0.5; `if_vetoed`: `fit_Z` (default) or `fit_Z_and_Z0`, which also fits A11pz_s0/s1), written into screen.json and decisions.json; the user confirmed the defaults (PLAN 23 items 3-5: share 1/2, only Z fitted with Z_0 logged in the screen table, refit from the fitted A2s with the A11 schedule) and the proposed approval records them with date and basis |
| R8 | see R6.4 | |
| R9 | `phase_stage2` order | gate -> A11, A11k (after the probe), A11n -> their guards -> A11p screen -> A11p fits -> their guards -> Stage-2 report; on the H200 first `conditional-controls guards report` at the new commit (R6.5) |

### R6.2 Hard stops on the H200 (PLAN 23 item 13)

Every one of these ends the process with a non-zero exit and a last line that starts with `STOP (PLAN 23 item 13` or
`stage2 refused:` (or a Python traceback for an exception), and no later phase of the same command runs:

- **parity / reference**: the harness parity of A0 at step 0 (`phase_parity`, unchanged), the reference set and its
  parity record (`registry`, unchanged), the source checks of every warm start (`verify_sources`: SHA-256 of
  model.pt and weights) and the tag / hash checks of every array read (`provenance.load_tagged`, unchanged);
- **scoring**: a Stage-2 run that diverges or is blocked (`run_stage2_spec`; criterion (ii) is a failed scoring
  check) - Stage 2 stops there instead of continuing as Stage 1 did; the gate's recomputation of results.json; the
  R9 comparison `run_m2a_report.py --compare` (exit 1 if anything other than `git` differs);
- **guards**: the G-region pass of a newly guarded run that does not reproduce its registered K = 64 array (strict
  criterion) or its own sum (> 1e-4) - nothing is written; the flagged* pass whose recomputed flagged class is not the
  stored array bit for bit, or whose identity flagged = flagged* + (low_abar and not flagged*) fails (> 1e-9) -
  nothing is written; any exception of registry.evaluate(guards=True);
- **file access / deferred files**: a missing or altered approval file, results.json, screen.json (SHA-256), marker
  or array (`FileNotFoundError`, hash mismatch), a pruning set other than the hashed one;
- **memory**: the k_train = 64 probe above 90 % of the cap stops before A11k.

A slow guard part (> 60 s) is a WARNING only (it is not a failed check).

### R6.3 Notes for the reviewer

- Reproducibility of a rebuilt source is compared on `weights_sha256`, the arrays and `kl_map`, never on the bytes of
  `init.json` or the file hash of `model.pt` (REVIEW 9.3): time stamps, the latest registry row of the source and the
  checkpoint file hash differ between executions of the same computation.
- The gate's recomputation of results.json includes the report's commit (REVIEW 9.5 Rm3): after the pull on the
  H200 the Stage-1 report must be issued again at the new commit, and the approval written for that file; that the
  new file differs from the approved one in `git` only is step 2 of R6.5.
- The H200 numbers of A9 bound the flagged* line before it is computed: flagged - low_abar = -0.34 (7 % of the gain)
  <= flagged* <= deep + border_ring + border_zbin = -0.80 (17 %), so the margin flagged* - 0.5 main lies between about
  +1.5 and +2.0 (seed 0; seed 1 alike): the waiver is expected to cover the failure. The gate decides on the array.
- The autodiff check (REVIEW 11.2): the second difference agrees with the exact second derivative to < 2 % where the
  float32 per-event values resolve the curvature (k = 1 coefficients: 0.2 - 1 % on 256 events); for leading
  coefficients with tiny h the per-event rounding dominates on a few hundred events (on all training events the
  noise of Q_j is about 0.005e-4, REVIEW 11.2), so the screen checks the three most resolved candidates. Outside
  `deterministic()` the CUDA sums alone differ by ~2e-5 per event and spoil the check; the check and the screen run
  inside it.

### R6.4 Evidence (laptop, smoke, a fresh scratch root; approval files were scratch fixtures and are gone)

- **Tests (laptop, CPU; the one GPU test - the A11p autodiff check - runs in the H200 test copy since the user moved
  test jobs there)**: `test_run_m2a.py`, `test_dist_factory.py`, `test_review_fixes.py`: 130 passed, 3 skipped (the
  three template tests that need the Track A templates in the default output root), 1 deselected.
- **Listing**: the output of `--list` up to the new block is byte-identical to the listing of the HEAD code
  (SHA-256 `5d67b5e2...`, first 102 lines `8762ba00...`); the additions of 26 Nov are a separate block appended after
  the block of 25 Nov (whole output now `1f1808a3...`, 128 lines).
- **A fresh smoke state made with the present code** (val_parity, refset, templates, parity; default list; then
  `warmstart-check warmstart-rerun a2c report`): 67 registry rows; kept list empty; guard pairs A7 and A9.
- **R8** on that state: the Stage-1 report of the HEAD code and of the present code are byte-identical
  (results.json and RESULTS.md). The state continued with the present code (default list again, then
  `warmstart-check warmstart-rerun a2c conditional-controls guards report`): no fit, registry.jsonl byte-identical,
  results.json and RESULTS.md byte-identical, 432 result files (arrays, checkpoints, done/init/marker files)
  unchanged; decisions.json gained only `conditional_controls` (not triggered; the six mixture pairs printed with
  their numbers).
- **R1 gate, real command**: refused without an approval; with content outside the list; with an exclusion outside
  the list; with a waiver for a rung outside the content; with a forged table hash; with an invalid A11p share.
  The valid approval (smoke content []) ran `stage2`: memory probe (1.74 GB for 205,842 hits, laptop limit 7.7 GB),
  A11k_s0, A11k_s1, their guards, A11p "not applicable", Stage-2 report (finalist A11k; "not cleaner than the
  legacy formula ... remains the Track A baseline"). Registry: 67 rows before, 71 after, the 67 a byte-identical
  prefix.
- **R9 in passing**: two commits by other workers landed between the state and the re-issue; the re-issued `stage2`
  was refused ("results.json differs ... (keys ['git'])"), the Stage-1 report re-issued, `run_m2a_report.py
  --compare` reported "only git differs" (exit 0), RESULTS.md differs in its commit line only; with the approval
  renewed for that file, the re-issued `stage2` started no fit, wrote no registry line and was not refused.

- **The content of PLAN 20 (A2s + A9) end to end** (laptop, scratch copy of the state above; the smoke list is not
  [A9, A9r], so a scratch harness fabricated that list IN MEMORY for the content check and replaced only
  `stage2_approval`; preconditions with the recomputation of results.json, the runs, the guards, the screen, the
  fits and the report are the real code). The content check on the smoke arrays refused - correctly: on smoke
  numbers the flagged* line of A9_s0 does not cover the G-region failure (flagged* carries 95 % of the smoke gain).
  Then: A11_s0/s1/s2, memory probe (1.86 GB), A11k_s0/s1, A11n_s0/s1, 12 guarded runs, the A11p screen (10 of 58
  coefficients pass on both seeds, Z = Z_0, k 104 -> 94; autodiff check passed: logit_tr[1] rel. 1.1e-4,
  logit_tr[2] 2.7e-4, s_R[1] non-finite autodiff recorded and skipped; 26 min on the shared laptop GPU), A11p_s0/s1
  with step-0 rows and the hashed pruning set, their guards, RESULTS_stage2.md (34 comparisons, 9 guard blocks).
  **A11_s0 and A11_s1 are bit-identical to A9_s0 and A9_s1** (max |Δ| 0). Smoke verdicts as printed: finalist A11k;
  no rung "cleaner"; A11p "not accepted (not equal NLL)", U1 and U2 met. Registry 67 -> 90 rows, the 67 a
  byte-identical prefix. A re-issued `stage2` (after the Stage-1 report was re-issued for a newer commit of another
  worker; `--compare`: only git differs): no fit, no GPU lock taken, registry unchanged, not refused.
- **First version of the screen, found by this run and fixed**: one candidate's autodiff second derivative was
  non-finite and the report could not format it; the check now takes the most resolved candidates until three
  finite comparisons exist, records the others, and stops A11p (PLAN 23 item 13) if a finite one disagrees by more
  than 2 %.
- **In the H200 test copy** (machine of record's GPU, grid 1056; not a git repository, so the commit fields are
  None there and the R9 comparison and the commit part of the gate's recomputation were exercised on the laptop
  only): the three test files 130 passed, 3 skipped, 1 failed - `test_review_fixes.py::test_C4_git_state_and_env_record`,
  which needs a git repository (expected in the test copy); `test_run_m2a.py` 59 of 59, the GPU autodiff test
  included. A fresh smoke state (5 min), then `warmstart-check warmstart-rerun a2c conditional-controls guards
  report`, then both commands again: no fit, registry.jsonl, results.json and RESULTS.md byte-identical; guard
  timing per run on the H200: K = 1024 + bulk 1.8 s, mass monitors 0.5 s, the K = 64 rest 0.7 s, G-region pass 0.6 s.
  `stage2` with a fixture approval (smoke content []): probe 1.76 GB of the 10 GB cap, A11k_s0/s1, guards, Stage-2
  report; re-issued: no fit, registry unchanged; the Stage-1 results.json / RESULTS.md byte-identical after Stage 2
  (70 -> 74 registry rows, prefix identical). The harness with the content A9 in a second fresh root: A11 (3),
  probe 1.91 GB, A11k (2), A11n (2), guards, **A11p screen in 10 min** (10 of 58 pass, Z = Z_0, k 104 -> 94;
  autodiff check passed with rel. 6.0e-4, 5.8e-4, 7.5e-3; two candidates with non-finite autodiff recorded), A11p
  (2) with step-0 rows, guards, report; **A11_s0 / A11_s1 bit-identical to A9_s0 / A9_s1 on the H200 as well**
  (smoke length); re-issued: no fit, no GPU lock taken.

### R6.5 On the H200 (R9): command sequence, and where it stops

Environment, output root and lock exactly as for the `m2a` job (section 4); a clean clone at the commit that carries
these files (REVIEW 11 is in `d0f04b8`, which precedes it). `<out>` is the machine's output root of record,
`<repo>` the clone. Each step ends with exit 0 or stops (non-zero exit, last line `STOP (PLAN 23 item 13 ...` or
`stage2 refused: ...`, R6.2); nothing after a stopped step is run.

    # 1. conditional controls (evaluated, recorded; nothing triggers on the H200 table), flagged* arrays and timing
    #    record for the 27 Stage-1 guard targets (no registry line), the Stage-1 report at the new commit
    uv run python -B shape-form-martin/scripts/run_m2a.py conditional-controls guards report
    # 2. R9: the re-issued table against the one the user approved (the copy-back in the clone); exit 1 = stop
    uv run python -B shape-form-martin/scripts/run_m2a_report.py --compare \
        <out>/shape-form-martin/results/machines/h200-g1056-97b197a7/runs/m2a_dist/results.json \
        <repo>/shape-form-martin/results/machines/h200-g1056-97b197a7/runs/m2a_dist/results.json
    # 3. (orchestrator) runs/m2a_dist/stage2_approval.json from the user's decisions, with the SHA-256 of the file of
    #    step 1 - the text proposed in the final report of this round
    # 4. Stage 2: gate -> A11 (0, 1, 2) -> memory probe -> A11k (0, 1) -> A11n (0, 1) -> guards -> A11p screen
    #    (written and hashed) -> A11p (0, 1) -> guards -> RESULTS_stage2.md
    uv run python -B shape-form-martin/scripts/run_m2a.py stage2

Steps 1 and 2 can be one job (`step1 && step2`). Expected on the H200: step 1 registers nothing and writes 27
flagged* arrays with markers (each marker records the bit-for-bit reproduction of the stored flagged class; a
failure stops); step 2 prints `keys that differ: ['git']`; step 4 refuses if the flagged* line of A9 does not cover
the G-region failure (expected to cover, R6.3). Copy back as before (`runs/m2a_dist/`, `registry.jsonl`,
`gpu_time.jsonl`, the guard_timing files).

## R5. The guard bottleneck on the H200 (user prerequisite for Stage 2) - cause located, fix bit-identical, H200 timing pending

**Where the time went (measured from the files of the H200 job, `results/machines/h200-g1056-97b197a7/`).** The
guards phase took 8.5 h for 27 runs. All 27 guard rows exist, with their K = 1024, bulk and mass-monitor arrays and
their G-region arrays on disk, including the five runs that took 5 s (A9n_s0, A7s_s0, A7s_s1, A7l_s0, A7l_s1): their
guards were computed. The modification times of the arrays (preserved by the tar copy-back) split each job:

| run | job start | K = 1024 + bulk arrays | mass monitors | G-region files (all, incl. sidecars) | G-region step |
|---|---|---|---|---|---|
| A7_s0 | 21:56:08 | 21:56:11 | 21:56:12 | 22:17:28 | **21 min 16 s** |
| A2s_s0 | 22:17:28 | 22:17:30 | 22:17:31 | 22:46:50 | **29 min 19 s** |
| A8c_s0 | 02:51:20 | 02:51:21 | 02:51:22 | 03:18:22 | **27 min** |
| A8n_s0 | 02:44:40 | 02:44:42 | 02:44:43 | 02:50:52 | 6 min 9 s |
| A9n_s0, A7s_s0 (fast) | 02:50:53, 02:50:58 | +2 s | +2 s | +3 s | 3 s |

- **The K = 1024 guard evaluation is not the bottleneck.** The registry part (K = 1024, bulk stratum, mass monitors)
  took about 3 s for every run, all night.
- The time went into the **G-region step** (`run_m2a.g_region_arrays`, Track A code): one extra validation pass with
  `return_parts=True`, then host-side work on the 4.17 million per-hit values (float64 copies, event index,
  hit-centre class masks, per-class sums).
- It ended with all eight files written within one second, so writes and fsyncs were not slow; the time passed
  before the first write.
- The same step took 3 s for five consecutive runs and 6-37 minutes otherwise, while the GPU-heavy K = 1024 part of
  the same jobs stayed at about 3 s. The cause is therefore a resource contended on the H200 that night and used by
  this step only; GPU throughput is ruled out by the K = 1024 timings.

**Laptop reproduction** (scratch harness, real models of the H200 job loaded read-only; RTX 4060), per run:
- K = 64 metric of record 1.4-2.3 s; K = 1024 + bulk 13.6-14.2 s; mass monitors 0.7-0.9 s;
- G-region validation pass 1.4-1.7 s; G-region host work 0.4 s;
- no major page faults, no allocator retries, peak device memory 0.3 GB for the G-region pass;
- pressure-stall information (cpu / memory / io) about 0.

The laptop does not reproduce the slowness: the step is intrinsically about 2 s. The candidates on the H200 are host
memory pressure (page reclaim / swap: the step is the only one that allocates several hundred MB of host memory per
run), CPU contention, or a stall in the device-to-host copies. Only a measurement there can tell.

**Timing command for the H200** (writes nothing but its log; fail-closed start-up; through the launcher):
`$L time_guards $P -B shape-form-martin/scripts/time_m2a_guards.py A7_s0 A2s_s0 A9n_s0 --repeat 2`

For every part (load, K = 64, K = 1024 + bulk, mass monitors, G-region pass, G-region host work before and after the
fix) it prints:
- wall and process CPU seconds, major and minor page faults;
- RSS, MemAvailable, swap in use, the load average;
- the pressure-stall information (/proc/pressure cpu, memory, io: avg10 and the stall seconds during the part);
- free device memory, peak allocated device memory and allocator retries.

A part whose wall time grows with its memory or I/O stall seconds names the resource. It should be run once at a
quiet and once at a busy hour, and before Stage 2, which will guard A11.

**Fix (bit-identical; `scripts/run_m2a.py`).**
- The model-independent part of the G-region step (event index of every hit, the four hit-centre class masks with a
  float64 copy of all hit coordinates, the stored-template lookup) is computed once per process
  (`_g_region_static`) instead of once per run.
- The computation is split into `g_region_compute` (no I/O, the original expressions) and the writer.
- The per-run host work drops from 0.41 s to 0.085 s on the laptop, together with its host allocations.
- Whether this removes the H200 stall depends on the cause the timing command finds. If it is host memory, this
  removes most of the per-run allocation. If it is the device-to-host copy of the per-hit arrays or CPU contention,
  a further change is needed: per-event sums on the device would not be bit-identical to the float64 `np.bincount`
  of the registered arrays, so it would need a decision.
- The registry guard path (`registry.py`, `metric.py`) was not changed: it is not the bottleneck.

**Proof that no number changes.**
- Laptop, before and after the fix, two benchmarks (mlp_std, component formula) and five component models of the
  H200 job (A2s_s0, A7_s0, A9_s0, A9n_s0, A10_s0): all 119 arrays are byte-identical - K = 1024 7/7, bulk stratum 7/7,
  mass monitors 56/56, G-region 49/49.
- New command `scripts/verify_m2a_guards.py` recomputes the guards of already-guarded runs with the current code and
  compares bit for bit with the registered arrays of the machine. It overwrites nothing; its only output is a new
  report `checks/guard_recompute/report_<UTC>.json`.
- On the laptop, against guards this machine registered with the code before the fix (smoke root): A7_s0, A2s_s0,
  A9_s0 BIT-IDENTICAL (K = 1024, bulk, every mass monitor, every G-region class); registry unchanged.
- For the H200: `$L verify_guards $P -B shape-form-martin/scripts/verify_m2a_guards.py A7_s0 A2s_s0` (exit 0 = PASS).

## R4. Fixes after the re-review of `a05c86e` (REVIEW.md section 8) - for before Stage 2

The re-review gave GO for `warmstart-check warmstart-rerun a2c report` on the H200 at `a05c86e`. The fixes below are
a separate change, committed after that follow-up job has started. **They do not alter what `warmstart-check`,
`warmstart-rerun` or `a2c` fit or register** (proof in R4.2).

### R4.1 What changed

| finding | fix | where |
|---|---|---|
| B8-1 | (i) `frozen_lr("net")` returns the rate decided again by `warmstart-rerun` (`decisions.json: warmstart_rerun.lr_net_after_rerun.lr_net_after`) when re-run pilots changed it, so every later network run (A11n of Stage 2 among them) uses the rate the re-runs used; `plr.json` stays the record of the pilot. (ii) `stage2_specs` names `effective("A2n_s<seed>")` as A11n's comparator; the Stage-2 record names the A2n runs in force and the rate of record. The report prints "Learning rate of record for network runs". Test on the lr_net-changed branch, which no smoke run has taken: `test_b8_1_lr_net_changed_branch_rate_of_record_and_stage2_comparator` (dependents re-run with the new rate, comparator `A2n_s0_ws`, `frozen_lr` and `fit_config` give the new rate, A11n compared with `A2n_s0_ws`). | `run_m2a_lib.frozen_lr`, `run_m2a.stage2_specs`, report |
| B8-2 | The gate accepts a run it waits for as **resolved** if it is registered, OR if it diverged and a valid diagnosis of that divergence is on file (`L.diagnosis`: names the SHA-256 of its `diverged.json`; verdict `failed` or `rerun_allowed`). Applied to the two A2c seeds and to the recorded warm-start re-runs. Consequences, printed: a diagnosed A2c seed has no control column ("A2c_s1 diverged; diagnosis on file (...): no control column for that seed. The report is final without it"), and the "NOT FINAL" banner is lifted; a diagnosed warm-start re-run does not supersede, so its original stays in force with its warm start flagged. Not weakened: without a valid diagnosis the gate refuses as before, and a diagnosis bound to another `diverged.json` is ignored. | `run_m2a._resolved`, `stage2_preconditions`, report |
| B8-3 | Kept line: `yes`, `no`, `no (failed, diagnosed: <run>)` or `not evaluable (<run>: <status>)`. Kept = keep on seed 0 and replication by seed 1, so a seed without a comparison makes the rule not evaluable unless seed 0 was evaluated and not kept. "Equal NLL (both)" prints `not evaluable (...)` when a seed has no comparison. The value in `kept_and_replicated` stays a boolean (False when not evaluable), so the Stage-2 list is unchanged in meaning: only rungs kept and replicated enter A11. | report `_kept_status` |
| C8-3 | `warmstart-check` re-issued: the record of the first check is reused when its step-0 array still carries its tag and SHA-256 and the inputs (target columns, recorded kl_map, decision) are unchanged; nothing is rebuilt. A network run restarted before its first checkpoint: the first step-0 array is verified by its tag and reused if the rebuild equals it bit for bit. Otherwise the training starts from the REBUILT model, so its parity is written as a new tagged array `val_nll_rebuild<k>.npy` beside the first (nothing is overwritten) with `rebuilt_after_restart` and the difference recorded. (Reusing the first array there would describe a start the run no longer has.) | `run_m2a._reusable_check_record`, `run_m2a_lib.step0_parity` |
| C8-4 | A re-run that did not register (diverged, blocked, missing) does not supersede: `L.superseded()` returns only the supersessions in force (re-run registered); `L.superseded_recorded()` is the record as written by `warmstart-rerun` (unchanged); `L.rerun_not_in_force()` lists the others. The rules, the guard pairs and the tables use the original; the table carries the row "<re-run> (warm-start re-run: **diverged** - NOT in force; the original above stays in force with its warm start flagged)". Implemented in the mapping only: `phase_warmstart_rerun` and what it records are unchanged. | `run_m2a_lib`, report |
| C8-5 | A2c carries the claim "exactly A2s" in the Initialisation table (smoke: kl_map 0.0, holds); the control column is headed "control: Δ vs A2c s0 / s1 (decides nothing)". | report |

### R4.2 Proof that `warmstart-check`, `warmstart-rerun`, `a2c` fit and register exactly what `a05c86e` does

Reference states were all produced by code identical to `a05c86e` (the tree equalled `a05c86e` when they were made):
- the old smoke root after `warmstart-check warmstart-rerun`, with the constructed below-floor case;
- the same with `a2c`;
- the fresh root with the injected divergences.

- P1 `--list`: the whole output, the additions block included, is byte-identical to `a05c86e`'s (and its first 102
  lines to the reviewed listing, `8762ba00...`).
- P2 re-issue: `warmstart-check warmstart-rerun a2c` with the new code on a copy of the a05c86e state started 0 fits.
  All 783 files of the machine directory (registry, `decisions.json`, `plr.json`, every run file, every tagged array,
  `results.json`) are byte-identical. The check reused its 14 records without a rebuild. On the fresh root, where the
  two warm-start phases had never been issued, the first issue added only their two (empty) keys to
  `decisions.json`; their content comes from unchanged code paths. The second issue left all 570 files byte-identical.
- P3 `a2c`: with the new code on a copy of the a05c86e state before A2c, both A2c rows are identical to the rows
  `a05c86e` produced on the same state (validation-array and weight hashes, 2 of 2).
- P4 `warmstart-check warmstart-rerun`: the pre-check state was rebuilt (old smoke root plus the deterministic
  construction of the below-floor case), then both phases ran with the new code. Compared with the a05c86e state:
  - all 74 smoke rows (14 re-runs, the constructed pair and every other row) identical in validation-array and
    weight hashes;
  - the 8 guard rows identical (K = 1024 and bulk arrays);
  - `decisions.json` keys `warmstart_check`, `warmstart_rerun`, `guard_targets` and `step0_parity` identical.

  `init.json` of the two runs fitted to the rebuilt A4_s0 (A4n_s0, A4n_s0_ws) differs from the reference. The
  comparison was interrupted before I traced the differing key. Their step-0 records in `decisions.json` are
  identical, so the difference lies elsewhere in `init.json`, probably in the recorded metadata of the source row
  (the timestamp of the independently rebuilt A4_s0 row). The A2n re-runs, whose source was not rebuilt, have
  byte-identical `init.json`.
- These proof runs used the code of this change. Afterwards only `phase_stage2` received one more line (the rate of
  record recorded safely when lr_net is undecided). The scratch roots of P2-P4 were lost in a session interruption;
  the numbers above were read before that.
- **The one case that can differ (by design, C8-4):** if a warm-start re-run DIVERGED, its original stays in force,
  so the guard pairs of the network rungs use the original comparator (a05c86e used the diverged re-run and computed
  no guards for those pairs). On such a state, re-issuing the guards phase with the new code adds guard rows for
  the pairs now in force (K = 1024 / bulk evaluations, no fit). Nothing else changes, and nothing at all when no
  re-run diverged.

### R4.3 Awaiting the user (not changed, by instruction)

1. **Divergence criterion (i)** (review 8.6, C8-1). As implemented, a non-finite validation mean anywhere in the
   history makes a run diverged and terminal, even if the fit recovered and ends finite and good. Reviewer's
   proposal: apply (i) to the LAST value, or to a fit that did not recover; an earlier non-finite value is reported
   with the run, like isolated skipped batches.
2. **A2c_x2 / A2c6k** (review 8.3). Reviewer's conditional rule, stated before any Track A result was read:
   - **A2c_x2** is needed iff the rules use a doubled-step rerun pair for some primary;
   - **A2c6k** is needed iff a mixture rung (A10, A10-2, A10-4, A10-5, A10g) passes the keep rule or the one-seed
     equal-NLL rule;
   - in either case before the content of A11 is decided. Until then the report says "not matched (...)".
3. **Pilot re-runs under rule (b) only** (C8-2). Re-runs of pilots whose targets have no column below the floor are
   replicates. If the H200 does not train bit-reproducibly, a near-tie could change lr_net and re-run the whole net
   column (as implemented). Reviewer's proposal: decide lr_net again only when a pilot was re-run under rule (a). A
   change without a below-floor column goes to the user before the re-runs are used; read the log line "lr_net after
   the warm-start re-run".

### R4.4 Tests

`pytest tests/test_dist_factory.py tests/test_run_m2a.py tests/test_models.py`: **113 passed, 3 skipped** (default
output root, templates absent); **116 passed, 0 skipped** with a dry-run output root. New: one test per finding of
section 8 (B8-1 lr_net-changed branch; B8-2 gate resolution; B8-3 kept status; C8-3 step-0 reuse / rebuild; C8-4
supersession in force; C8-5 A2c claim); the earlier gate test was adapted to the recorded map.

**End-to-end with the current code (laptop, smoke, new scratch root; diagnosis and approval files were TEST
FIXTURES).**
- Setup: two-step reference set ALL PASS; real parity PASS (-1.869e-08 ± 1.1e-08 nat); the smoke default list with
  the network warm starts in fit_to version 2 (scratch harness, as the job in progress made them).
- `warmstart-check warmstart-rerun` with `A2n_s0_ws` made to diverge (smoke-only hook): all 14 network runs listed
  (rule b); `A2n_s0_ws` DIVERGED; the other 13 re-ran.
  - **C8-4:** the report shows "A2n_s0_ws (warm-start re-run of A2n_s0: diverged - awaiting diagnosis - NOT in force;
    A2n_s0 stays in force)". The net rows keep their numbers against the original in force (`A4n_s0_ws vs A2n_s0`:
    -7.48 ± 2.36). Under a05c86e every net row lost its numbers.
- `a2c` with `A2c_s1` made to diverge: A2c_s1 "diverged - awaiting diagnosis"; the report says NOT FINAL.
- Re-issuing `warmstart-check warmstart-rerun a2c`: registry byte-identical (the check reused its 14 records).
- **B8-2:**
  - Stage 2 with an approval fixture: refused (A2c_s1 diverged without a valid diagnosis).
  - Diagnosis fixtures (A2c_s1: failed; A2n_s0_ws: rerun_allowed), then the report: "A2c_s1 diverged; diagnosis on
    file (failed): no control column for that seed. The report is final without it". No NOT FINAL banner. A2n_s0_ws
    "diagnosis: rerun_allowed - NOT in force".
  - The old approval was refused as stale. With a renewed approval Stage 2 ran, and its re-issue passed the R6
    recomputation.
- **B8-3** on a fresh-root state with A7_s0 diverged: "A7: not evaluable (A7_s0: diverged - awaiting diagnosis)" in
  the kept line and in "equal NLL (both)". With a valid "failed" diagnosis: "A7: no (failed, diagnosed: A7_s0)".
- **C8-5:** "A2c_s0 | ... | exactly A2s: yes" and both control column headers "(decides nothing)".
- Repository: `results/registry.jsonl` and `results/test_access.log` unchanged; `--list` SHA-256 `5d67b5e2...` (=
  a05c86e).

**Files for staging (this change):** `shape-form-martin/scripts/run_m2a.py`, `shape-form-martin/scripts/run_m2a_lib.py`,
`shape-form-martin/scripts/run_m2a_report.py`, `shape-form-martin/tests/test_run_m2a.py`,
`shape-form-martin/results/m2a_dist/STATUS.md`. (`paramnet.py`, `dist_factory.py`, `test_models.py`,
`test_dist_factory.py`: unchanged from `a05c86e`.)

## R3. The user's decisions of 25 Nov 2026 (PLAN 17.11): A2c, scale floor, divergence - written, waiting for re-review

The real Track A job runs on the H200 from commit `5ddfd9f` (= the reviewed files). Everything below is for AFTER that
job has ended and after an independent re-review. Nothing here was run on the H200.

**Hard constraint, shown:**
- `run_m2a.py --list`: the first 102 lines (the whole listing of the reviewed code) are byte-identical (SHA-256
  `8762ba00...`). The additions follow as a separate block headed "# Additions of 25 Nov 2026".
- `listing()`, the spec builders, `DEFAULT_PHASES` and every rule and threshold are unchanged.
- A run with a `done.json` is returned by `run_spec` before any new code path is entered, so the job's runs stay
  untouched and are still recognised as done.
- The changes to the phases before `guards` act only when a run diverges or is blocked, which the reviewed code
  handled by stopping.
- Rebuilding an existing network run (`load_run`, `skip_init`) never calls `fit_to`.
- On the starting trajectories of all eight network models of the list, `fit_to` version 3 gives warm starts bit for
  bit equal to version 2 (test).

### R3.1 What changed

| item | what | where |
|---|---|---|
| A2c | `a2c_specs()`: `A2c_s0` and `A2c_s1` are the A2s structure with `start` = A2s of the same seed, T_A with 4000 steps, the frozen lr_poly, seed = the seed (the schedule of the rungs that start from the fitted A2s), class `control`, comparator `A2s_s<seed>`. Phase `a2c`, NOT in `DEFAULT_PHASES`. The report prints, beside every matched rung, "control: Δ vs A2c" (paired, decides nothing), an A2c section marked "control (added by the user on 25 Nov 2026, exploratory, outside the six primaries)", and a banner "NOT FINAL" while A2c is missing. Stage 2 refuses while A2c is not finished. | `run_m2a.py`, `run_m2a_report.py` |
| step budgets (definition to confirm) | The rungs that start from the fitted A2s do NOT all share one budget. **Matched:** the 19 component rungs that start from the fitted A2s (A4, A5, A7, A8, A9 and 14 exploratory rungs), all 4000 steps; the test checks this set against the run list. **Not matched, proposals awaiting the user:** the mixture family A10, A10-2, A10-4, A10-5, A10g (6000 steps, initialised by `init_from_target` on the map of A2s, not continued from its weights), proposal "A2c6k" (A2c with 6000 steps, seeds 0 and 1); the convergence reruns (8000 / 12000 steps, against `A2s_s<seed>_x2`), proposal "A2c_x2". The report prints "not matched (...)" for those rows. | `A2C_GROUPS` |
| scale floor | `ParamNet.fit_to` version 3. The weight of a varying column is 1 / max(std_j, 1e-2)^2 in raw (unconstrained) units. Columns with std >= 1e-2 keep the version-2 float32 normalisation bit for bit, and exactly constant columns keep the earlier treatment. `scale_floor=None` reproduces version 2 (used only to rebuild old warm starts). Recorded in init.json (`init_info.fit_to`): `fit_to_version`, `scale_floor_abs`, `column_std`, `below_scale_floor`, `column_weight`, per-column `rms`. Factories: `fit_to_scale_floor` (default = version 3). | `paramnet.py`, `dist_factory.py` |
| floor: which columns change | On the real run list (starting trajectories of A1, A2n, A4n, A5n, A7n, A8n, A9n, A10n): no varying column below 1e-2 (smallest std 0.041); the warm starts are bit for bit those of version 2 (test `test_scale_floor_leaves_every_net_model_of_the_list_bit_identical`). The fitted targets of the job are checked by `warmstart-check`. The cure, measured (reviewer's case: six cubic columns plus one column of small spread, pre-registered call): worst ordinary column without the floor 60.8 % / 56.5 % / 11.7 % of its spread for an extra column of relative spread 1e-6 / 1e-4 / 1e-3, with the floor 0.79 % / 1.32 % / 0.60 % (0.19 % without the extra column). For spreads >= 5e-3 (std >= 1e-2) nothing changes. | test, `r3/floor_repro.py` (scratch) |
| step-0 parity | For every new warm-started network run, `run_spec` computes before training the per-event validation NLL of the network at step 0 (the registry's computation: unmodified `common.test_nll` inside `deterministic()`). It saves this as a tagged array `runs/<stage>/<run>/step0/val_nll.npy` (not a registry row) and pairs it with the source polynomial run's registered array through `provenance.paired_delta`. The source is `net_from`, or `A0_init` for A1. The result goes to init.json, done.json and `decisions.json: step0_parity`, and is printed in the Initialisation table. | `run_m2a_lib.step0_parity` |
| warm-start check | `run_m2a.py warmstart-check` trains nothing. For every network run executed with `fit_to` version < 3 it records: (i) the target columns from the source run (std, constant, below the floor); (ii) the recorded `kl_map`; (iii) a deterministic **rebuild** of the old warm start (same seed, same build order as `run_spec`, `fit_to_scale_floor=None`), judged bit for bit by exact equality of the rebuilt and recorded `kl_map`, and the step-0 parity of that rebuild (tagged arrays under `runs/<stage>/warmstart_check/<run>/`); (iv) the PLAN 17.11 rule applied by code (`warmstart_affected`): re-run iff (a) a varying target column has std < 1e-2, or (b) the recorded kl_map > 1e-4 nat. It writes `decisions.json: warmstart_check` (with `rerun`). If a rebuild is not bit for bit, this is said; the decision always uses the RECORDED kl_map. | `run_m2a.py` |
| warm-start re-run | `run_m2a.py warmstart-rerun` re-runs exactly the listed runs as `<name>_ws` (fit_to version 3, step-0 parity, comparators mapped to their superseding runs). Originals are never overwritten. `decisions.json: warmstart_rerun.superseded` is written after every run (resumable). Dependents: if a network pilot was re-run, lr_net is decided again among the pilots (re-runs in place of originals); if it changes, every other network run is re-run with the new rate (listed as `dependent`). Then the guards phase runs (targets recomputed with the superseding names). The report shows each original as "superseded: warm start" and applies the rules to the re-run. Stage 2 refuses until every listed run has its finished re-run. | `run_m2a.py`, report |
| divergence | Definition (`DIVERGENCE_DEFINITION`, below). After `train.fit` and before the registry, `run_spec` checks the history; a `FloatingPointError` of the registry is caught. A diverged run writes `diverged.json` (criteria, step, last finite validation and train NLL, the fit summary with skip counters, parameter statistics with the weight hash of the last kept parameters, the gradient norm at those parameters on a fixed batch, kept files) and `decisions.json: diverged`, frees the GPU, and the job continues. It is terminal: re-issuing the command does not retrain it. A run whose `start` / `init_ref` / `net_from` diverged (or is blocked) is **blocked**: not trained, `decisions.json: blocked` ("blocked by <run>"). A pilot that diverged loses under the P-lr rule (recorded); if every network pilot is lost, lr_net is "undecided" and the network runs are blocked; if every polynomial pilot is lost, the job stops. Exploratory rungs (and the net column, exploratory by AUDIT 4.1): status "diverged", no second seed, the rung stays in every table. Primary, base, control and rerun runs: status "diverged - awaiting diagnosis"; NO rule marks them failed; the report says "failed (diagnosed)" only with a valid diagnosis file; Stage 2 refuses while one waits. A diverged A0 seed makes the drift rule "not evaluable" for that seed (recorded); the job continues. | `run_m2a_lib.py`, `run_m2a.py`, report |
| review R1-R8 | R1: the alternative column shows the verdict of the alternative reading (literal conditions AND two-seed mean <= -2 SE). R2: the G-region alternative also requires a gain in the total. R3: "the guards phase has not run" when no guard targets are recorded. R4: guard blocks of an original not used by the rules (or superseded) are labelled. R5: rms_max is printed relative to that column's spread; the template marks are "Q1, Q2" for A3c and "Q1" for A3x. R6: `phase_stage2` recomputes the report in memory and requires equality with `results.json` (decisions of Stage 2 and Stage-2 runs are left out of the report, so a resumed Stage 2 is not refused). R7: the approval must contain the key `kept_and_replicated`. R8: `results.json` records `git: {commit, clean}`. | report, `run_m2a.py` |

**Divergence definition (for the reviewer's and the user's eye):**
> A run is 'diverged' iff (i) a validation mean in its fit history is not finite, or (ii) the registry evaluation of
> the finished fit finds a non-finite per-event NLL (FloatingPointError), or (iii) 'frozen': in the last evaluation
> window at least 50 % of the training batches were skipped by sfm.train.fit for a non-finite loss or gradient, or
> (iv) 'blow-up': the last validation mean exceeds the reference by more than 1 nat per event (10,000 in table units),
> the reference being the lowest of: the validation means of the fit history, the step-0 validation mean of a network
> warm start, and the registered validation mean of the run's comparator. Isolated skipped batches of a fit that ends
> finite are NOT a divergence (they are reported with the run).

It is conservative: `sfm.train.fit` already skips non-finite batches, so only a fit whose result is unusable, or that
has gone more than 1 nat wrong, is declared diverged. No healthy fit of the list is expected within orders of
magnitude of 1 nat; the A10 mixtures in smoke were 0.04 nat worse than A2s.

**Diagnosis file** (written by the orchestrator after an independent mathematical diagnosis; never by code):
`runs/<stage>/diagnosis/<run>.json`
```
{"run": "A7_s0", "diverged_json_sha256": "<sha256 of runs/<stage>/A7_s0/diverged.json>",
 "verdict": "failed" | "rerun_allowed", "diagnosed_by": "Fable (independent mathematical diagnosis)",
 "diagnosed_utc": "2026-11-26T10:00:00Z", "summary": "..."}
```
A file with another hash, another verdict or a missing field is ignored (the run stays "awaiting diagnosis").
"rerun_allowed" changes no number. A re-run would be a new, explicit decision.

**Proposed change to a base module (not made; `sfm.train` is read-only).** The gradient norm of a diverging fit is
known only at its last parameters (the diagnostics compute it on a fixed batch). To record it during the fit:
`src/sfm/train.py` line 553: `torch.nn.utils.clip_grad_norm_(params, cfg.clip)` ->
`gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip)); st["gn_max"] = max(st.get("gn_max", 0.0), gn)`.
Line 568, in the history entry: add `"grad_norm_max": st.pop("gn_max", None),` before `"seconds"`. In `note_skip`
(line 514): `st.setdefault("first_skip_step", st["step"] + 1)`. This costs one device sync per step and changes the
content of `model.pt` / `checkpoint.pt` (history), not the weights.

### R3.2 H200 commands after the present job has ended (in order; one job; resumable by re-issuing)

```
cd <repo>; git pull          # the commit of this round, after its re-review
export SFM_OUT_ROOT=<area>/sfm_out SFM_USER_CACHE=<area>/sfm_cache
L=shape-form-martin/scripts/launch.sh; P=.venv/bin/python
# 0. only if the job stopped at a non-finite evaluation (reviewed code: FloatingPointError): finish the default list
#    with the new code - the stopped run becomes 'diverged' and the list continues
$L m2a $P -B shape-form-martin/scripts/run_m2a.py
# 1.-4. the additions and the final report, in this order, in one process:
$L m2a_after $P -B shape-form-martin/scripts/run_m2a.py warmstart-check warmstart-rerun a2c report
```

| phase | acceptance | output |
|---|---|---|
| warmstart-check | one line per network run: `kl_map ... (rebuilt ..., bit for bit True)`, below-floor columns, `RE-RUN` / `keep`; then `N of M network runs to be re-run` | `decisions.json: warmstart_check`; tagged step-0 arrays under `runs/m2a_dist/warmstart_check/` |
| warmstart-rerun | one `<name>_ws` run per listed run (and dependents), each logging its step-0 parity; then `superseded {...}` and the guards phase | new run directories `<name>_ws`, rows, `decisions.json: warmstart_rerun` |
| a2c | `A2c_s0`, `A2c_s1` registered | rows, `decisions.json: a2c` |
| report | `report: ... written` without "NOT FINAL" | RESULTS.md, results.json |

Then the user reads the table. Stage 2 needs the approval file (section 0) and refuses while: A2c is missing, a listed
warm-start re-run is missing, a diverged primary / base / control waits for its diagnosis, or `results.json` differs
from a recomputation. Stop a job with `shape-form-martin/scripts/stop_job.sh <pid file>`. Copy back the machine
directory as in section 4.

### R3.3 Evidence (laptop, smoke mode, scratch output roots)

Two scratch output roots. Each reference set was made by the two-step `val_parity.py` (both runs ALL PASS), and
real templates + parity passed (A0 at step 0 - formula array = -1.869e-08 ± 1.1e-08 nat, max event 1.34e-05).

**Fresh root (new code from the start).** Smoke default list through the launcher with
`SFM_M2A_INJECT_DIVERGENCE=A4t_s0,A7_s0` (smoke-only hook: the first parameter tensor is set to NaN half-way).
- Kill and resume: the job was stopped at `[A2n_s1] step 20`; re-issuing the same command resumed it ("resuming from
  step 20 of 40").
- A7_s0 (primary): "DIVERGED (non-finite validation mean in the fit history; frozen: 10 of 10 batches skipped in the
  last evaluation window)", recorded as "diverged - awaiting diagnosis". The record holds step 30, the last finite
  validation (step 20), 180 non-finite parameters, the gradient norm at the last parameters, and the kept files.
  A7n_s0: "BLOCKED by A7_s0", not trained.
- A4t_s0 (exploratory): "diverged - no second seed; the batch continues".
- The job ran on through guards and report. In the tables: A7 seed 0 "**diverged - awaiting diagnosis**" (no keep,
  no replication), A4t "**diverged**", A7n_s0 "**blocked by A7_s0**", and a section "Diverged and blocked runs" with
  the full diagnostics.
- Re-issuing after completion started 0 fits; the diverged and blocked runs stayed as recorded.
- `run_m2a.py --smoke a2c`: stopped at `[A2c_s0] step 20`, resumed, both registered. Report: A2c_s0 - A2s_s0 =
  +0.51 ± 0.24, and the control column, e.g. A8_s0: Δ vs A2s +0.50 ± 0.23 but Δ vs A2c -0.01 ± 0.04 (on the smoke
  fits the offset shared by the rungs that start exactly at A2s is the continued training, as the reviewer suspected
  in B6). The mixtures show "not matched (6000 steps, init_from_target)".
- Hard constraint: all 55 rows common to this run and the earlier smoke run of the REVIEWED code are identical in the
  SHA-256 of the validation array and of the weight tensors. This includes the 14 network runs, now with fit_to
  version 3 and the step-0 parity evaluated before training. Different only: the injected runs (A4t_s0, A7_s0), the
  blocked A7n_s0, the earlier forced rerun (A7_s0_x2, A2s_s0_x2), and the additions (A2c_s0/1, A11k_s0/1).
- Stage-2 gate (approval and diagnosis files were TEST FIXTURES written in scratch):
  - refused without an approval file;
  - with an approval: refused, "diverged runs await an independent diagnosis: ['A7_s0']";
  - a diagnosis fixture (verdict failed) changed the table (A7 seed 0 "failed (diagnosed)"), so the old approval
    was refused as stale;
  - with a renewed approval Stage 2 ran (nothing kept: A11k_s0/1);
  - re-issuing Stage 2 after that passed the R6 recomputation (stable with Stage-2 runs present).

**Copy of the old smoke root** (all 14 network runs made with fit_to version 2). Constructed below-floor case, real
code paths, scratch copy only (the original directories were moved aside, never deleted): A4_s0 with the EMG rows
at their defaults (`init="none"`), trained 4 smoke steps at lr 5e-4. Its rows sh_mu, sh_rsig and sh_rtau then have
std 1.4e-3, 1.5e-3 and 4.7e-4. A4n_s0 was warm-started from it with the full pre-registered call in version 2 (kl_map
4.9e-4 nat).
- `warmstart-check`: every old warm start was rebuilt **bit for bit** (rebuilt kl_map equal to the recorded one, 14
  of 14). A4n_s0: rule (a) (sh_mu, sh_rsig, sh_rtau below the floor) and rule (b). The other 13: rule (b) only (the
  smoke warm start of 40 Adam steps gives 0.008-0.09 nat). Re-run list: all 14. A second issue of the command gave
  the identical decision.
- `warmstart-rerun`: stopped at `[A2n_s1_ws] step 20`, resumed ("resuming from step 20 of 40").
  - All 14 re-run as `<name>_ws`; lr_net re-decided among the re-run pilots: 0.01 -> 0.01 (no dependents); guards of
    the re-run pairs computed.
  - **13 of the 14 re-runs are bit-identical to their originals** (validation array and weight hashes): the floor
    changes nothing where no column lies below it.
  - Only the constructed A4n_s0 changed: version 3 floored sh_mu, sh_rsig and sh_rtau (weight 1e4 each); the worst
    column went from 11.5 % to 2.35 % of its spread; step-0 parity against A4_s0 went from -4.30 ± 0.76e-4 (old warm
    start, rebuilt) to +0.11 ± 0.15e-4.
  - Its kl_map is still 3.7e-4 nat, so the B2 flag stays on A4n_s0_ws. The rule is applied once (to runs made before
    version 3), so it does not loop. The full-grid kl_map weights low momenta that the data hardly populate; the
    step-0 parity is the data-weighted measure.
- Report: the net column uses `A4n_s0_ws vs A2n_s0_ws`, with the original row "superseded: warm start; rules use the
  row above". Stage 0 shows `A1_s0_ws - A0_s0` with the original labelled. The Initialisation table shows version,
  floored columns, step-0 parity ("rebuilt" for the old ones) and "superseded: warm start -> <name>_ws".

**Tests:** `pytest tests/test_dist_factory.py tests/test_run_m2a.py tests/test_models.py` gives 107 passed, 3 skipped
(default output root, templates absent) and 110 passed, 0 skipped (with a dry-run output root). Repository:
`results/registry.jsonl` and `results/test_access.log` unchanged; nothing was written in the clone except the
git-ignored `cache/gpu.lock`.

### R3.4 Definitions that need the reviewer's or the user's eye

1. The divergence definition above: the thresholds (1 nat blow-up, 50 % frozen window) and that isolated skipped
   batches are not a divergence.
2. Divergence classes: the net column counts as exploratory ("diverged", no diagnosis needed), as AUDIT 4.1 calls it
   exploratory. Pilots lose under the P-lr rule. Primaries, bases (A2s, A2n), controls (A0, A1, A2c) and reruns wait
   for a diagnosis. A diverged A0 seed makes the drift rule not evaluable for that seed, and the job continues (the
   user's "allow independent rungs to finish").
3. Stage 2 also refuses while a primary / base / control / rerun waits for its diagnosis (my addition; the user asked
   only that no primary be called failed without one).
4. The A2c step-budget groups and the two proposals (A2c6k, A2c_x2), which await the user.
5. The scale floor is absolute (1e-2 in raw units, the user's words), not relative to the column's mean as the
   reviewer suggested (2^-5 max(1, |mean|)). "Below the floor" = varying (not constant by the 2^-20 rule) and float64
   std on `default_lnp(48)` with uniform weights < 1e-2. A column with |mean| >> 1 and std just above 1e-2 is not
   floored.
6. The source of the step-0 parity of A1 is `A0_init` (the legacy coefficients at step 0).
7. Warm-start re-run dependents: an lr_net change re-runs every network run with the new rate; comparators are
   mapped to the superseding runs; the guards of the re-run pairs are computed; nothing else depends on a network
   run in Stage 1.
8. R6: the comparison leaves out Stage-2 decisions and Stage-2 runs; the git record holds `commit` and `clean` only
   (the number of changed files of a dirty laptop clone would make the comparison fragile).

### R3.5 Awaiting the user

A2c6k and A2c_x2 (R3.4, item 4). The proposed `sfm.train` change for gradient norms (R3.1). Unchanged from
section 0: A11p is not implemented, and `stage2_specs` still needs its review.

**Update after the independent review of commit `c4f39fd` (`results/m2a_dist/REVIEW.md`: PASS WITH REQUIRED FIXES,
no class-A finding).** The real run starts on the H200 from the reviewed files. The fixes B1-B5 and C2 (section 0)
change only the report, the Stage-2 gate and an extra report option. **Nothing that any phase before `guards` fits or
registers has changed, and neither has the run list:**
- `run_m2a.py --list` is byte-identical before and after (SHA-256 of the output `8762ba00...`);
- `run_m2a_lib.py`, `paramnet.py` and `dist_factory.py` are unchanged (SHA-256 as in REVIEW.md section 0);
- the diff of `run_m2a.py` touches only `phase_report`, the new `stage2_approval` with `phase_stage2`, and `main()`
  (the `--without-guards` option).

The new report code is applied to the H200 run's registered arrays afterwards (`run_m2a.py report`).

**State.** Track A is ready to run on the machine of record as ONE detached, resumable job (section 4). **No real Track
A fit has been run anywhere.** Nothing was registered under `stage="m2a_dist"` in the campaign results. The laptop
dry run went to a scratch output root (section 6). It covered templates and the step-0 parity for real, then every
phase in smoke mode under the separate stage `m2a_smoke`, including a kill in the middle of a fit and a resume. In
the repository, `results/registry.jsonl` and `results/test_access.log` are byte-identical, and nothing was written in
the clone. The code is waiting for review: the Fable rule review of E4 and the independent mathematical review of
the `fit_to` fix (section 2).

Units: deltas are in nat per event. Tables use 1e-4 nat (the convention of AUDIT.md). `<repo>` is the clone and
`<area>` is the user's folder on the H200.

## 0. Fixes after the review of `c4f39fd` (B1-B5, C2), the reviewer's rulings, and what awaits the user

| item | fix | where |
|---|---|---|
| B1 | When the doubled-step rerun pair (`<rung>_s<seed>_x2` vs `A2s_s<seed>_x2`) of a non-converged primary exists, it is the pair of that seed for keep, replication, equal NLL, "cleaner" and `kept_and_replicated`. Seeds may mix an original and a rerun, each paired with its own comparator. The table has a column "pairs used by the rules (s0 / s1)", and the original is printed as an extra row "original - not converged, NOT used by the rules"; it also appears in the secondary columns, labelled. `results.json` records the pairs used (`replication.pairs`). | `run_m2a_report.primary_pairs`, `collect`, `markdown` |
| B2 | A new "Initialisation" table for every finished run: `kl_map` (nat) and its reference (`kl_ref`); the claim "exactly A2s" for A7, A8, A78, A7s, A8c and A12 (holds if \|kl_map\| <= 1e-6). For net runs: `rms_max` (column), the L-BFGS stop, and the flag **"warm start degraded - net column not interpretable for this run"** when `kl_map` > 1e-4 nat. Target columns of each net warm start are recomputed on the CPU from the fitted poly run (or from the legacy provider for A1), and those with relative spread sd / max(1, \|mean\|) between the 2^-20 floor and 1e-2 are listed in the table and in one summary line. Reporting only: `fit_to` is unchanged. | `warm_start_targets`, `initialisation_table` |
| B3 | Printed beside the literal readings, labelled "alternative - decides nothing": (i) the two-seed mean also at <= -2 SE (a column in both Stage-1 tables, plus a second "kept and replicated" line); (ii) the G-region flagged share of the **hit-term total** with its margin (flagged - 0.5 hit term, SE) and "would pass". | `replication`, `g_region_block` |
| B4 | Beside A3c / A3x, the template limitations are printed with the reviewer's measured sizes. Q1: the front-face z-bin and the 2 x 2-point rule (0.50 against 0.35; -0.10 / -0.16 / -0.14 at the legacy coefficients). Q2: computed from this machine's `templates_all.npz` - cells above 1.01 and the lowest z-bin centre where that happens, the maximum, the smallest denominator there, and the counts above 1 and 1.1; to be read with G-region `deep` and the `beyond_zb` monitor. On the laptop templates: 79 cells from z' = 1379 mm, maximum 1.81, denominator 0.186; 906 above 1, 56 above 1.1. These equal the review's numbers. | `template_limitations`, `TEMPLATE_NOTE_Q1/Q2` |
| B5 | Stage 2 runs only with an approval file bound to the SHA-256 of the `results.json` the user saw (format below). It is refused if the file is absent, if the hash differs (the table was recomputed after the approval), if the replication reading is unknown, if the approved list differs from the list `results.json` gives under that reading, or if `approved_utc` / `approved_by` are missing. `stage2` stays outside the default phase list. | `run_m2a.stage2_approval`, `phase_stage2` |
| C2 | `run_m2a.py report --without-guards` (also `run_m2a_report.py --without-guards`) writes the tables without guard columns to `RESULTS_without_guards.md` / `results_without_guards.json`, with a banner saying that no guard statement can be read from it. `results.json`, the file the Stage-2 approval binds, is written only by the full report. The full report now lists guard pairs whose guard data is missing instead of dropping them silently. The default command and its phases are unchanged. | `report(with_guards=...)`, `main` |
| also (cheap, no rule) | C12: the kept line is split into primaries and exploratory; a multiplicity sentence; a footnote that the replication rule runs in seed order. B6 reading aid: the convergence movement of A2s_s0 / A2s_s1 is printed beside the primaries table. | `markdown` |

**Stage-2 approval file** (written by the orchestrator after the user has approved; never by code):
`R/machines/<machine id>/runs/m2a_dist/stage2_approval.json`

```
{"results_json_sha256": "<sha256sum of runs/m2a_dist/results.json as the user saw it>",
 "replication_reading": "literal",            # or "two_se" (review ruling 5: [confirm] before Stage 2)
 "kept_and_replicated": ["A7", "A8"],         # the list the user approved; must equal results.json under that reading
 "approved_utc": "2026-11-26T10:00:00Z",
 "approved_by": "user (recorded by the orchestrator)",
 "notes": "free text"}
```

Then run `$L m2a_stage2 $P -B shape-form-martin/scripts/run_m2a.py stage2`. Any re-run of the full report rewrites
`results.json`, so the approval has to be renewed. A11p is still not implemented, and `stage2_specs` must be reviewed
before Stage 2 (review B5).

**Verification of the fixes** (laptop, the second dry-run root of section 6, smoke rows, CPU-only report):
- B1:
  - The convergence rerun of A7_s0 was forced with a scratch harness that calls `run_m2a.rerun_specs` and
    `run_m2a_lib.run_spec`, exactly as the driver does for a non-converged primary. It trained and registered
    `A2s_s0_x2` and `A7_s0_x2`.
  - The new report shows `A7 | rerun / original | +0.59 ± 0.18 | ...` (the review measured the same +0.59 ± 0.18),
    with the row `A7 s0 original - not converged, NOT used by the rules | A7_s0 vs A2s_s0 | +0.55 ± 0.25`.
  - Replication used the pairs `[A7_s0_x2, A2s_s0_x2]` and `[A7_s1, A2s_s1]`.
- B2:
  - "exactly A2s: yes" for A7/A8/A78/A7s/A8c/A12, with kl_map between -1.1e-8 and 3.4e-8 nat; A9 2.3e-4.
  - All 14 smoke net runs are flagged "warm start degraded" (0.008-0.09 nat; the smoke warm start is 40 Adam steps
    without L-BFGS, as the review predicted).
  - No nearly-constant target column was found in the smoke targets.
  - The A1 legacy-provider target: smallest relative spread 0.041, nothing nearly constant (unit test).
- B3: both alternative columns and lines are printed; a unit test builds a case where the literal reading replicates
  and the 2-SE alternative does not.
- B4: as in the table above.
- B5: unit test of every refusal and of acceptance.
- C2: `run_m2a.py --smoke report --without-guards` wrote `RESULTS_without_guards.md` with the banner; the option is
  refused with any other phase list.
- Tests: `pytest tests/test_dist_factory.py tests/test_run_m2a.py tests/test_models.py` gives **85 passed, 3
  skipped** (default output root, templates absent) and **88 passed, 0 skipped** (with `SFM_OUT_ROOT` = the dry-run
  root).
- Repository: `results/registry.jsonl` and `results/test_access.log` are unchanged, and nothing was written in the
  clone.

**The reviewer's rulings on the flagged readings** (REVIEW.md section 3; numbering of section 3 below):

| # | reading | ruling |
|---|---|---|
| 1 | paired parity criterion | [within] - A6 revised point 4, verbatim |
| 2 | drift rule, two-sided | [within] |
| 3 | P-lr tie rule; net pilots fitted to A2s_s0 | [within] |
| 4 | equal NLL one-sided; second seed on keep or one-seed equal NLL | [within] |
| 5 | replication: two-seed mean <= -3, no 2-SE condition | [within] as the literal reading; **[confirm] before Stage 2** (it decides A11): both readings are now printed (B3), and the approval file names the reading |
| 6 | kept and replicated | [within], with B1 (now fixed) |
| 7 | convergence; reruns for the poly primaries only, from the original A2s | [within] |
| 8 | guards on a superset (keep or equal NLL, plus comparators) | [within] |
| 9 | G-region share of the total gain, margin form | [within]; the hit-term share is printed as well (B3) |
| 10, 11 | G-mass per monitor without a threshold; G-bulk masking | [within] |
| 12, 13, 14, 17 | secondary columns, "cleaner", no 1-SE in Stage 1, A9 step-0 rows | [within] |
| 15 | A6b observed information on training events | [within] |
| 16 | Stage 2 | not ruled; gated (B5) |
| Q1, Q2 | templates | [within]: runs as registered, limitation printed (B4) |
| `fit_to` fix | constant-column treatment | [within]; mathematics correct (two qualifications in REVIEW.md section 3) |

**Awaiting the user (not implemented, by instruction):**
1. **B6, a control "A2c"**: A2s continued unchanged (start = A2s of the same seed, T_A, seeds 0 and 1), to measure
   the common "continued training" term of every rung that starts at A2s. It is not pre-registered, selects nothing,
   and can be added at any time after A2s exists.
2. **A scale floor in `fit_to`** for columns of small but non-zero relative spread, at about 2^-5 max(1, |mean|): the
   review measured that such a column degrades every column of the warm start to 14-17 % of its spread. Until it is
   decided, B2 makes an affected net run visible. If it is wanted later, only the flagged net runs would be refitted.
3. **C1, "record a failed run and continue"** for a non-finite validation evaluation of an exploratory rung or a net
   version. As coded, such a run stops the track and a restart stops at the same place. A pilot without finite scores
   loses by the P-lr rule ([within]); for the other classes the user decides.

## 1. What changed

| File | Change |
|---|---|
| `src/sfm/models/paramnet.py` | `ParamNet.fit_to`: a scale floor for columns that are constant in momentum, and a guarded L-BFGS (section 2). The return value also carries `constant_columns`, `near_floor_columns`, `scale_floor_rel` and `lbfgs`. |
| `src/sfm/models/dist_factory.py` | The replica workaround is **removed**: `_fit_to_robust`, `_degenerate_net_columns` and `_fit_target` are deleted, and `make_paramnet` calls `fit_to` directly. The fit report records the constant columns. The templates now live **per machine**: `template_dir()` is `OUT_RESULTS/machines/<id>/runs/m2a_dist/templates` (pristine-clone rule). `set_template_dir()` points the factory elsewhere, for example to rebuild A3c/A3x/A9 from a machine directory copied back to the laptop. `TEMPLATE_DIR` and `THETA_FILE` stay readable as module attributes (PEP 562). New keyword arguments `fit_to_steps` and `fit_to_lbfgs` (defaults 2000 / 500, the AUDIT values) are used only by the smoke mode. |
| `scripts/run_m2a_lib.py` | Rewritten for the provenance/registry/paths API. All outputs go to `R/machines/<id>/runs/<stage>/`. `start()` calls `machine.startup`. `registered_row` carries the live stream id. `tagged()` uses `registry.row_array` (tag and SHA-256 checked). `paired()` uses `provenance.paired_delta`, against a run or against **this machine's formula reference array** (`registry.reference_set().bench["formula"]`). A done-marker from another machine is refused. `on_gpu` follows the rule of `run_m2b_lib.run_on_gpu`: a capped device refuses at once, the laptop releases the lock and retries. Smoke mode: `configure(smoke=True)` is refused unless `SFM_M2A_SMOKE=1`; it uses stage `m2a_smoke`, steps divided by 100, and shortened factory initialisations. |
| `scripts/run_m2a.py` | Rewritten. One command runs the default phases `templates parity plr stage0 stage1p stage1x guards report` in order, in one process. `--list` gives a dry listing. `--smoke` selects smoke mode. `stage2` runs only on its own and is refused before `results.json` exists. Every laptop constant is gone (`FORMULA_OF_RECORD`, the absolute drift threshold), and so are the budget and drop logic (`BUDGET_S`, `DROP_ORDER`, `_est_seconds`, `_reserve_seconds`). `decisions.json` records `dropped: []` and the reason. New in the guards phase: the G-region arrays and the A6b Fisher information. |
| `scripts/run_m2a_report.py` (new) | The rules applied to registered arrays, the guard tables, the secondary columns, RESULTS.md and results.json. It runs on the CPU only. |
| `tests/test_models.py` | Four new `fit_to` tests (section 2). |
| `tests/test_dist_factory.py` | The net-version test now checks constant-column reproduction and fit quality, with no replica. New test: constant and varying columns are far apart in all 8 net models. The template tests now read this machine's directory and assert what was measured (section 7, Q1 and Q2). |
| `tests/test_run_m2a.py` (new) | The run list equals AUDIT 4.1. No laptop constant, budget or drop order remains. Outputs go to the machine directory. Smoke mode is refused without its switch. Stage 2 is not in the default list. Also covered: the rule constants, the Stage-2 merge, and the G-region classes. |

Tests: `uv run pytest shape-form-martin/tests/test_dist_factory.py shape-form-martin/tests/test_run_m2a.py
shape-form-martin/tests/test_models.py -q` gives **79 passed, 3 skipped** (after the review's batch-1 fixes had
landed). The 3 skips are the template tests: with
the default output root the templates do not exist. With `SFM_OUT_ROOT=<scratch dry root>`,
`test_dist_factory.py` gives **51 passed, 0 skipped**.

### 1a. Adaptations to the batch-1 review (coordinator's message during this item)

The batch-1 modules are being fixed by the independent review. They were not edited here. Track A was adapted to
the announced changes:

1. **Machine identity comes from the API.** `run_m2a_lib.MACHINE_DIR = machine.machine_dir_live()` and
   `MACHINE_ID = MACHINE_DIR.name`. `dist_factory.template_dir()` uses `machine_dir_live()` too. A run's scores
   directory is read from its registry row (`val_nll_path`) and is never rebuilt from an id.
   - The derived tagged arrays (G-region, mass monitors) carry the registry's live tag (`registry.live_stream()`).
   - Markers record `machine_id` and are checked against it.
   - `live()` checks that `registry.RegistryEnv().machine_dir` is the driver's directory.
   - The settings tag is recorded only (`machine_tag`, `startup_tag`).
2. **The reference set** comes only from `registry.reference_set()` (`NoReferenceSet` otherwise). There is no
   fallback.
3. **Fail-closed start-up.** Every entry point (`run_m2a.py` and `run_m2a_report.py`) goes through
   `run_m2a_lib.start()`, which calls `machine.startup()`. Nothing bypasses the start-up checks. The smoke switch
   only selects a stage and shortened fits. It touches neither the guard, the output root nor the clone check.
4. **Starting points verified by SHA-256** (`run_m2a_lib.verify_sources`, called in every `run_spec` before the
   factory reads any weights). Every `start` (copy by name), `init_ref` (init_shower_long reference, mixture
   init_from_target target) and `net_from` (fit_to target and rebuilt poly model) must be:
   - a finished run of this machine (its done.json carries this machine id);
   - whose `model.pt` SHA-256 matches a registry row on the live stream;
   - whose weight tensors match that row's `weights_sha256`.

   What is recorded (model SHA-256, weights SHA-256, machine id, row stage / name / timestamp / stream /
   val_nll_sha256 / checkpoint SHA-256) goes into `init.json`, into the registry row (`extra.sources`) and into
   `done.json` (`sources_verified`). Tested in `test_source_runs_are_verified_by_sha256`. This covers the six
   primaries, the 18 exploratory rungs, A2n / A1 / the net pilots, the reruns and Stage 2. Convergence reruns keep
   the primary's original source.
5. **Launcher.** Stop only by the PID file, or with the review's stop helper (`scripts/stop_job.sh`, new and
   untracked at the time of writing; not used here). The commands of section 4 are unchanged.
6. **No raw pairing.** The parity criterion, the drift rule, P-lr, keep / equal / replication, the report tables,
   the guards and the secondary columns all form their deltas with `provenance.paired_delta` (or
   `paired_difference` for per-event values), against the live reference set or registered runs. The
   laptop-array pairing that was formerly at `run_m2a.py:172` is gone (`test_no_budget_no_drop_order_no_laptop_constants` checks
   for `m0_parity/formula`).
   - The two-seed replication mean is a `paired_delta` between the averaged tagged arrays, after `check_pairable`
     of all four.
   - The pre-registered *margins* (G-K / G-bulk persistence; the G-region flagged - 0.5 main) are computed with
     `stats.guard_persistence` / `stats.paired` from tag-checked `paired_difference` outputs. This is the pattern of
     `sfm.registry._guards`. The two terms cannot form one TaggedArray pair, because their subsets differ by
     construction.

## 2. The `ParamNet.fit_to` fix - mathematics (for the independent reviewer)

**Problem.** The warm start minimises a weighted least-squares objective over the momentum grid {x_g}. Let p_g be
the point weights, normalised to sum 1, and t_gj the target trajectories. The objective is

  L = sum_j lambda_j sum_g p_g (f_j(x_g) - t_gj)^2,   with lambda_j = 1 / Var_p(t_j)

for normalised weights (weights=None or shape (G,)). Each column's term is then its fraction of unexplained
variance. The output is f_j(x) = b_j + W_j . h(x), where h is the shared hidden layer, W_j is row j of the last
layer, and b_j is its bias.

Some columns are constant in momentum: A9n's `logit_dh`, and degree-0 rows in general. For those, Var_p(t_j) is pure
float32 summation rounding, about 1e-15, so lambda_j is about 1e14 to 1e15. That one column then dominates the
gradient of the shared layers, the varying columns go unfitted, and the strong-Wolfe line search of L-BFGS
overflowed. The old code only covered Var = 0 exactly.

**Rule.** Column statistics are computed in float64, with p renormalised to sum 1 in float64:

  mu_j = sum_g p_g t_gj,   s_j = sqrt(sum_g p_g (t_gj - mu_j)^2).

Column j is **constant** iff

  s_j <= tau * max(1, |mu_j|),   tau = 2^-20 (= 8 float32 ulps at magnitude 1; float32 epsilon = 2^-23).

The scale is relative above 1 and absolute below 1, because raw parameters are logs and logits. float64 is needed
because the float32 mean of a bitwise-constant column differs from the value by a few ulps.

**Treatment of a constant column.** Its scale is set to infinity (weight 0), not to the floor:

1. Its value is set to round_f32(mu_j). For a bitwise-constant target (t_gj = c for all g), round_f32(mu_j) = c
   exactly: the float64 sum is c(1 + O(G * 2^-53)), which rounds back to the float32 c. A names_const column becomes
   that scalar. A net column gets W_j = 0 and b_j = round_f32(mu_j).
2. The column gets weight 0 in the MLP objective and in the reported loss.

*Why weight 0 changes nothing (exactly constant target t_gj = c).* (b_j, W_j) appear in column j's term only, and the
hidden parameters theta reach column j only through W_j . h_theta. So for any weight lambda_j >= 0:

  L(theta, b, W) = L_rest(theta, b_-j, W_-j) + lambda_j sum_g p_g (b_j + W_j . h_theta(x_g) - c)^2.

The second term is >= 0, and it equals 0 at (b_j, W_j) = (c, 0) for **every** theta. So the set of minimisers is
{(c, 0)} x argmin L_rest, which is the same set for every lambda_j in [0, inf). (If h_theta has a constant direction,
there are more zero sets, but (c, 0) is always among them.) Weight 0 therefore picks a solution of the original
problem, and it avoids the conditioning catastrophe of a finite lambda_j around 1e14.

*Exactly zero gradient.* With weight 0, dL/dW_j = 0 and dL/db_j = 0 identically.
- Adam keeps (m, v) = (0, 0) for those entries, so each step is lr * 0 / (0 + eps) = 0.
- L-BFGS directions are linear combinations of the current gradient and of past steps s_k and gradient differences
  y_k. All of these are 0 in those entries, so the entries never move.
- At the end, (W_j, b_j) is reset to (0, round_f32(mu_j)) anyway, which is a bitwise no-op.

*Target constant up to rounding* (|t_gj - c| <= a few ulps). Any finite lambda_j would only fit float rounding noise
through the shared layers. Setting the column to round_f32(mu_j) leaves an error <= max_g |t_gj - mu_j|, a few ulps.
The test bounds this at 1 ulp for a column alternating between c and c + 1 ulp.

**Columns above the floor** keep exactly the old float32 operations: `w / torch.where(var > 0, var, 1)`. So when no
constant column is present, the weights, the optimisation and the outputs are **bit-for-bit those of the old
code**. This is tested against a verbatim copy of the old function on CPU, comparing all weights, rms and loss.

**Margin of the rule on the real trajectories.** Test `test_constant_and_varying_columns_are_far_apart` covers the 8
net models of the list (A1, A2n, A4n, A5n, A7n, A8n, A9n, A10n) at their starting coefficients:
- every column has either s_j <= floor (exactly 0 or rounding),
- or s_j / floor >= 42,889 (the smallest varying column).
- `fit_to` reports any varying column within a factor 1e3 of the floor (`near_floor_columns`); there are none.

**Guarded L-BFGS.**
- Each chunk of at most 100 iterations runs inside `try/except` (RuntimeError, OverflowError, ValueError,
  FloatingPointError), and the objective must be finite after each chunk.
- On failure, the best state seen so far is restored (the start of the polish included) and the polish stops.
- The kept state is the last one whose loss is <= every earlier loss. In the normal monotone case that is the final
  state, as before. This is part of the bit-identity test.
- A non-finite loss during Adam raises `FloatingPointError`, a loud failure instead of NaN weights.

**Tests** (`tests/test_models.py`):
- (iii) `..._bit_identical_to_the_code_before_the_fix`: 4 varying columns; state dict, rms and loss are identical.
- (ii) and (i), `..._constant_net_column_is_reproduced_...`:
  - The column alternates by 1 ulp. Old weight 1/var > 1e12. Now: reproduced within 1 ulp, row = 0, constant outside
    the grid too.
  - A bitwise-constant column is reproduced bit for bit.
  - The varying columns reach rms < 0.5 % of their spread and stay within a factor 5 of a fit without the constant
    column. Measured: mu 1.99e-4 vs 2.06e-4, log_s 9.6e-5 vs 2.9e-5, frac 2.09e-4 vs 1.82e-4.
  - The two optimisations are not bitwise equal: the summation order in the shared layers differs, and the fit is
    chaotic at the net's 1e-3-of-spread limit.
- `..._constant_scalar_column_and_floor_rule` (names_const; a column varying by 2e-3 is not constant; explicit (G, P)
  weights are left untouched).
- `..._lbfgs_guard_restores_the_best_state` (an injected overflow in chunk 2).
- `tests/test_dist_factory.py::test_net_versions_of_the_primaries`:
  - A9n: `logit_dh` is constant and reproduced bit for bit.
  - Varying net columns reach rms/spread <= 1.1 %. A7n is the worst case: 22 outputs, and since it has no constant
    net column its MLP part is identical to the old code.

**Choice recorded.** The replica in `dist_factory` is removed, not bypassed. `init_info["fit_to"]` now reports
`constant_columns`, `near_floor_columns`, `scale_floor_rel` and `lbfgs_stopped` instead of `fit_to_workaround`.

## 3. Readings of the rules (most literal reading; each one flagged)

1. **Parity, paired form** (A6 revised, point 4). The A0_init array must equal this machine's formula reference array
   (refset, K = 64) within 1e-4 for every event and within 1e-6 on the mean. It is evaluated with
   `provenance.paired_delta` / `paired_difference`. Failure stops the run.
2. **Drift rule.** |paired_delta(A0_s, formula array)| > 2e-4 on either seed stops the run. Same as before, now on
   this machine's arrays.
3. **P-lr.** The lowest last-iterate validation mean wins; within 0.5e-4 of it the smaller lr wins. The difference
   is formed as a paired delta against the lowest. Its mean equals the difference of the means, and the SE is
   recorded. The pilots are registered (stage m2a_dist, names `plr_*`).
4. **Equal NLL, one seed:** Delta <= +1 SE and Delta < +1e-4. A seed-1 exploratory run is triggered by keep **or**
   seed-0 equal NLL (AUDIT 4.1). "Equal NLL" in the tables requires both seeds; with one seed it is shown as "-".
5. **Replication.** Seed 1: Delta <= -1.5e-4, with the same sign as seed 0. Two-seed mean: the mean over events of
   (d0 + d1)/2 must be <= -3e-4. I read "meets the threshold" as the magnitude part of the keep rule (-3e-4). The
   two-seed SE is printed, but no 2-SE condition is applied to the mean. **Flag:** the stricter reading would also
   require <= -2 SE.
6. **Kept and replicated**, the input of Stage 2: keep on seed 0, plus replication. This is computed in the report
   (`results.json: kept_and_replicated`). No combination is formed.
7. **Convergence:** |val(last) - val(last - 1000)| < 0.5e-4, from this machine's fit history (same points, K = 64).
   Reruns with doubled steps apply only to the six **poly** primaries, together with A2s of the same seed (`_x2`; the
   comparator rerun is shared per seed). Net primaries are exploratory (AUDIT 4.1), so they are reported and not
   rerun.
8. **Guards: which runs.** AUDIT 4.0 says "kept rungs". I apply the guards (G-K, G-bulk, G-mass, G-region) to every
   run of a rung that passes keep **or** one-seed equal NLL on any registered seed, and to each comparator (same
   seed). This is a superset of the literal set. Guards never select, and on equal-NLL rungs the persistence checks
   fail on `main_is_gain` by construction; this is reported as measured. The net column counts as its own rung.
9. **G-region.**
   - The hit term is attributed by **hit centre**, using the partition of `results/m1_audit/checks/c_gap.py`:
     - deep: z' >= 1221 mm;
     - border: last ring (r >= R_EDGES[31]) or the first / last z-bin;
     - low acceptance: stored Abar <= 0.3 in the hit-centre cell;
     - "flagged" = the union of these; "inner" = the rest.
   - The normalisation term is its own row.
   - Pass: the flagged hit term is <= 50 % of the **report-metric gain**, read as the margin flagged - 0.5 * main >= 0
     (with gain < 0). The margin's SE is reported, and the result counts as resolved at |z| >= 2. The persistence
     margin form for G-region is my extension of the refined definition, which states it for G-K and G-bulk.
   - **Flag:** the denominator is the total gain, not the hit-term gain.
10. **G-mass.** The per-event monitors of `registry.evaluate(guards=True)` (npz, file SHA-256 checked against the row)
    are wrapped as TaggedArrays with subset `mass_monitor:<key>` and paired. There is no pass/fail threshold, because
    the plan states none.
11. **G-bulk** uses the stratum zb >= 1221 mm (amendment A4) and excludes events whose bulk NLL is non-finite in
    either run (the count is printed). The short events (zb < 1221 mm) get their own report-metric row.
12. **Secondary columns:** p < 2 GeV vs p >= 2 GeV, and contained vs side-leaking. An event is side-leaking when the
    axis exit depth from `dist_factory.exit_depth` (port of `c_side.py`) is < 1450 mm. Rows exist for every Stage-1
    comparison.
13. **"Cleaner"** (AUDIT, Stage 2) is shown per rung:
    - by gain: Delta vs this machine's formula array <= -3e-4 (A6 revised, point 4);
    - by constants: k <= 0.9 * 101 together with equal NLL on both seeds.
14. **1-SE:** the list applies it only to the Stage-2 finalist, so it is not applied in the Stage-1 report.
15. **A6b "Fisher information of k".** This is the observed information of the raw coefficient of `sh_rk` (degree 0,
    one coefficient) on the **training** events: the second central difference (h = 0.01) of sum_e nll_e at fixed
    fold points (K = 16, seed 12345, deterministic), with SE = 1/sqrt(I). **Flag:** the alternatives are the
    expected (squared-score) information, or the information on validation.
16. **Stage 2** (separate, explicit phase; refused before `results.json` exists).
    - If nothing is kept: A11 = A2s and A11n = A2n (no reruns), and A11k = A2s with k_train = 64 (seeds 0, 1).
    - Otherwise: A11 merges the component keyword arguments of the kept and replicated rungs. A conflict (two depth
      families, say), a mixture rung or A12 is refused, and the content is the user's decision. A11 runs seeds 0, 1, 2;
      A11k is A11 with k_train = 64 (seeds 0, 1); A11n is its net version (seeds 0, 1).
    - **A11p (pruning) is not implemented.**
17. **The step-0 rows A9_s0_init / A9_s1_init** ("record Delta at step 0") are paired against A2s of the same seed in
    the report.
18. **Smoke mode only:**
    - fits run steps / 100, with validation every steps / 4;
    - factory loops are shortened (fit_to 40 Adam steps and no L-BFGS; init_from_target 15 + 15 steps, 8 search
      points);
    - guards are limited to 2 rungs (or the first rung, if none passes, so that the code path runs);
    - **the parity and drift stops are recorded but not enforced.** The first smoke attempt stopped at the drift rule:
      A0_s1 was +2.83e-4 after 40 steps, which is correct behaviour for an unconverged fit.
    - None of this exists in the real mode.

## 4. On the H200: command line, acceptance per phase, copy-back

Preconditions: batch 1 has passed (ENABLING_STATUS.md, steps 5-11), above all step 6-7, which creates this machine's
reference set. The Fable review of E4 has passed. The user has approved the run (PLAN 16.10, rules 5 and 9; 17.6).
There is one GPU job at a time, the cap is 10 GB, and 4 threads.

```
cd <repo>
export SFM_OUT_ROOT=<area>/sfm_out          # outside the clone (the same root as batch 1)
export SFM_USER_CACHE=<area>/sfm_cache
L=shape-form-martin/scripts/launch.sh
P=.venv/bin/python
$P -B shape-form-martin/scripts/run_m2a.py --list            # optional: the run list (no GPU, writes nothing)
$L m2a $P -B shape-form-martin/scripts/run_m2a.py            # the default phases, in order, in one process
```

Do not set `SFM_M2A_SMOKE`, `SFM_ALLOW_DIRTY_CLONE`, `SFM_RESULTS_DIR`, `SFM_CACHE_DIR`, `SFM_DATA_DIR`,
`SFM_ALLOW_CPU` or `SFM_ALLOW_RECORD_WRITER`. Each phase prints `=== phase <name> ===`, and the job ends with
`=== finished [...] ===`.

| phase | acceptance (read in the log / decisions.json) | on failure |
|---|---|---|
| start | the line `[run_m2a] machine <id> ... file-access guard ACTIVE`, the same machine id as batch 1, `git ... clean: True` | any start-up exception of the base stops the job; report it: `NoCUDA`, `UnguardedStart` (output root inside the clone or guard not active off the laptop), `DirtyClone`, `InsufficientGPUMemory` (less than cap + 4 GB free; re-issue later), `NoReferenceSet`, `ReferenceSetNotLicensed` (set without a passing parity record), `ReferenceSetDisputed` (a recomputation differed; blocked until the user decides) |
| templates | `templates.json` written. Expected from the laptop: fb vs stored abar mean 7.6e-4, max 0.146 in z-bin 14 only; back_corrected max 1.81 (z' >= 1385 mm); full_box <= 1. | - |
| parity | `parity: A0 at step 0 - formula array = ... -> PASS`, i.e. max event <= 1e-4 and |mean| <= 1e-6. Laptop: -1.9e-8 +- 1.1e-8, max 1.34e-5. | STOP (SystemExit): report the numbers |
| plr | `plr.json` has `lr_poly`, `lr_net` | - |
| stage0 | `A0 drift: ... -> continue`, i.e. both seeds within 2e-4 of the formula array | STOP: report (harness or optimum differs) |
| stage1p | 18 rows (12 poly, 6 net) and the step-0 rows A9_s0_init / A9_s1_init; `decisions.json: convergence_primaries`, plus `_x2` reruns where needed | - |
| stage1x | 18 seed-0 rows, plus seed-1 rows where triggered; `decisions.json: dropped = []` | - |
| guards | a guard row (`guards` not null) and `g_region.json` for each run in `decisions.json: guard_targets`; `fisher/A6b_s0_sh_rk.json` | - |
| report | `report: .../RESULTS.md and results.json written` | - |

Volume (from `--list`):
- 54 unconditional fits with 224,000 training steps (batch 256, K = 16);
- at most 32 conditional fits (18 seed-1 runs, 12 primary reruns, 2 comparator reruns) with at most 200,000 more steps;
- in total at most 86 fits, 424,000 steps.

At laptop speed the AUDIT cost column gives about 4.0 h, plus up to 1.3 h of second seeds, plus reruns. **The H200
speed is unknown.** Nothing is dropped and nothing depends on time.

**If the guards phase fails** (or before it has run), the Stage-1 tables can still be produced without guard
columns: `$L m2a_report_ng $P -B shape-form-martin/scripts/run_m2a.py report --without-guards`. This writes
`RESULTS_without_guards.md`, fits nothing and registers nothing (review C2). After the run, the reviewed report code
(section 0) is applied with `$L m2a_report $P -B shape-form-martin/scripts/run_m2a.py report`.

**Stage 2 is not started.** It needs the user's approval, recorded in `runs/m2a_dist/stage2_approval.json` and bound
to the SHA-256 of the `results.json` the user saw (format in section 0). Only then:
`$L m2a_stage2 $P -B shape-form-martin/scripts/run_m2a.py stage2`.

**Copy back.** Copy the whole machine directory, as in ENABLING_STATUS.md section 3:
`tar -C <area>/sfm_out -cf machine.tar shape-form-martin/results/machines/<machine id>`, then unpack it at the
laptop repository root. The Track A part is `runs/m2a_dist/`:

| tracked (JSON / MD) | git-ignored, copy by hand |
|---|---|
| `plr.json`, `decisions.json`, `gpu_time.jsonl`, `results.json`, `RESULTS.md`, `templates/templates.json`, `fisher/*.json`; per run `done.json`, `fit_done.json`, `init.json`, `scores/<points id>/*.tag.json`, `g_region.json`; `registry.jsonl` (machine root) | `templates/*.pt`, `templates/*.npz`, `templates/train_theta.npy`; per run `model.pt`, `scores/<points id>/*.npy`, `val_mass_monitors.npz`; `fit.log` if a gate wants the curves (not `checkpoint.pt`) |

## 5. Stopping and resuming

- **Stop** only with the base's stop helper and the job's PID file:
  - `shape-form-martin/scripts/stop_job.sh <area>/sfm_out/jobs/m2a_<UTC>.pid`. It sends SIGTERM only if the PID still
    belongs to that job (start time, working directory and command line are checked). Never stop a job by name;
  - the driver has no signal handler, so a stop is a kill at an arbitrary point, which is the tested case (REVIEW.md
    section 4);
  - the GPU lock (flock) is released when the process dies.
- **Resume** by re-issuing exactly the same `$L m2a ...` line. Then:
  - finished runs are skipped by their `done.json`, which is refused if written by another machine;
  - an interrupted fit resumes from its own `checkpoint.pt`: `train.fit` checks the config hash and the machine tag,
    and checkpoints are written every 500 steps;
  - a registry row is not written twice for the same `model.pt` (SHA-256 on the live stream);
  - a step-0 row that is missing after training began raises an error rather than registering a trained model as
    step 0;
  - the decisions are recomputed from the registered arrays (the same numbers) and rewritten.
- A run killed between `model.pt` and its registry row is registered on resume. A run killed between the row and
  `done.json` is found by its checkpoint SHA and not evaluated again.
- Guard rows and G-region arrays are redone only if missing. Tagged arrays are never overwritten (an identical
  recomputation is accepted).
- On the shared H200 a job is **refused, not queued**, when less than cap + 4 GB is free at the start of a GPU job
  (`InsufficientGPUMemory`, the batch-1 rule). The process then exits, and the remedy is to re-issue the same command
  later.

**The listing compared with AUDIT 4.1** (`run_m2a.py --list`, 97 rows):

| AUDIT 4.1 | in the listing |
|---|---|
| step-0 parity of A0 | `A0_init` |
| P-lr pilots | A2s lr 0.003 / 0.01 / 0.03 and A2n lr 0.001 / 0.003 / 0.01, seed 0, 3000 steps |
| Stage 0 | A0, A1, A2e at seeds 0 and 1; A2s, A2n at seeds 0, 1 and 2 (A2s_s0 before the net pilots) |
| six primaries | poly at seeds 0 and 1 (A9 with step-0 rows), then net at seed 0, all 4000 steps (A10: 6000) |
| convergence reruns | 12 conditional `_x2` reruns plus 2 comparator reruns, doubled steps |
| 18 exploratory rungs | all at seed 0; seed 1 conditional on keep or equal-NLL; 4000 steps (the A10 family: 6000) |
| Stage 2 | separate explicit phase |

Every rung and seed of the list is present. Nothing is dropped, and no budget logic remains.

## 6. Laptop dry run (evidence)

Scratch output root `<scratch>/m2a_dry` (pristine mode, `SFM_ALLOW_DIRTY_CLONE=1`, because other workers' files are
uncommitted), grid 144. The dry run was done **twice**, because the review's identity change landed in between:
first as machine `rtx-4060-laptop-gpu-g144-629104b0`, then as `...-76c0f59a` (the machine directory now comes from
`machine_dir_live()`).

**First run** (old identity, code before the coordinator's adaptations):
1. `val_parity.py --create-refset`: ALL PASS.
2. Real templates + parity: PASS, same numbers as below.
3. Smoke through the launcher, with the **file-access guard active**:
   - killed by its PID file after A0_s0 had logged step 20 of 40;
   - the same command resumed it (`resuming from step 20 of 40`);
   - the next attempt stopped at the drift rule (A0_s1 +2.83e-4 after 40 steps). This led to smoke reading 18;
   - the third issue ran templates through stage1x and the guard evaluations of 6 runs (A7_s0/s1, A9_s0/s1 and
     A2s_s0/s1);
   - it then failed at the A6b Fisher step: its output directory was not created. Fixed: `write_json` now creates
     parent directories.

**Second run** (new identity, the final code):
1. `val_parity.py --create-refset`: `ALL APPLICABLE CHECKS: PASS`; the reference set was created in the new machine
   directory.
2. **Real** `run_m2a.py templates parity` (stage m2a_dist):
   - templates computed: fb vs stored abar mean 7.6e-4, max 0.146 (Q1);
   - **parity: A0 at step 0 - formula array = -1.869e-08 +- 1.1e-08 (n 14950), max |event| 1.34e-05 -> PASS** (the
     paired criterion, 1e-4 per event and 1e-6 on the mean).
3. **Smoke**, `SFM_M2A_SMOKE=1 launch.sh m2a_smoke2 .venv/bin/python -B .../run_m2a.py --smoke` (guard active):
   - killed by its PID file at `[A0_s0] step 20` of 40;
   - **resumed with the same command**: `[A0_s0] built ... resuming`, `resuming from step 20 of 40`;
   - the drift stop was recorded and not enforced (smoke only);
   - source verification passed for every run started from another run (the plr net pilots, A2n, primaries,
     exploratory rungs);
   - result: every phase ran to `=== finished [templates, parity, plr, stage0, stage1p, stage1x, guards, report] ===`.
     - 55 finished runs: the 54 unconditional fits plus `A9r_s1`, triggered because A9r_s0 passed the one-seed
       equal-NLL rule. 64 `m2a_smoke` rows and 1 `m2a_dist` row (A0_init), all on one stream
       (`g144-s12345-b128-iid-val731a1997@<machine id>`).
     - `decisions.json` holds `dropped: []` and the budget reason.
     - Guard rows and G-region arrays were made for 6 runs (A7 and A9 at seeds 0 and 1 with their A2s comparators;
       the smoke limit is 2 rungs).
     - A6b Fisher information: 1443 (SE of the coefficient 0.026, smoke weights).
     - RESULTS.md / results.json hold all the tables and 4 guard blocks. In the G-region block for A7_s0 vs A2s_s0,
       the hit term + the normalisation term reproduce the main delta to 1.5e-5. G-mass is per monitor.
   - In this run the job also **waited correctly for the GPU lock** while another worker's `val_parity` held it
     (about 4 min).
   - `run_m2a_report.py --smoke` on its own rewrote the report. Without `SFM_M2A_SMOKE=1` it is refused.
4. **Nothing in the repository.**
   - `results/registry.jsonl` and `results/test_access.log` have the same SHA-256 before and after (checked after
     each run).
   - No `results/machines/` exists in the clone. The only clone-side write is the git-ignored `cache/gpu.lock`.
   - No JSON or registry line in the scratch machine directory contains `/home/` or the scratch path (checked);
     paths are written as `<out>/...` and `<repo>/...`.
   - The first run's outputs (old identity) were moved aside to `<scratch>/m2a_dry_firstrun_629104b0`.

## 7. Not tested, and open questions

- **Q1 (templates) - ruled [within] (REVIEW.md section 3); runs as registered, limitation printed with A3c/A3x (B4).**
  The ported `acceptances` (c_side.py, 2 x 2 sub-points per cell) resolve the front-face step only to 1/2. In z-bin
  14 (z' in [-19.5, -6.6) mm) they give 0.5 against 0.354 (stored abar) at r = 0. The effect on the validation NLL at
  the legacy coefficients is -0.1 to -0.2 (review).
- **Q2 - ruled [within]; printed (B4).** The literal back_corrected ratio is above 1 in 906 cells, above 1.01 in 79
  cells (all at z' >= 1379 mm) and above 1.1 in 56 cells; the maximum is 1.81, where mean 1[z' < zb] is 0.186.
  full_box stays <= 1.
- Not run anywhere:
  - the real fits (by instruction);
  - the real guard evaluations on kept rungs (smoke only);
  - Stage 2 (only its spec construction is unit-tested);
  - the A6b Fisher information (smoke only).
- Nothing ran on the H200: the capped-memory path of `on_gpu` and the clean-clone refusal are exercised there only.
- **Proposed change to a module under review (not made).** `src/sfm/train.py` lines 452, 491 and 619 (after the review edits) print absolute
  paths (`model_path`, `ck_path`) into `fit.log`. On the H200 that writes the `<area>` path into a file that may be
  copied back (fit.log is git-ignored, so it is not committed). Exact diff:
  `{model_path}` -> `{paths.portable(model_path)}` (lines 452 and 619), `({ck_path})` -> `({paths.portable(ck_path)})`
  (line 491). Cosmetic: the fit log lines carry a second, local-time stamp.
- `scripts/make_transfer_package.py list` cites `run_m2a.py:43`, `:44`, `:172` and `run_m2a_lib.py:121`, `:166`.
  Those constants are removed, so `list` will report the rows as moved. The section 16.7 table rows for run_m2a are
  resolved by this item.
- The listing shows comparator `A2e_s<seed>` for the `A2s_s<seed>_x2` rerun rows. This is cosmetic: the rerun pairs
  with its primary.

## 8. Files for explicit staging

**Round R3 (user decisions of 25 Nov 2026):** `shape-form-martin/scripts/run_m2a.py`,
`shape-form-martin/scripts/run_m2a_lib.py`, `shape-form-martin/scripts/run_m2a_report.py`,
`shape-form-martin/src/sfm/models/paramnet.py`, `shape-form-martin/src/sfm/models/dist_factory.py`,
`shape-form-martin/tests/test_run_m2a.py`, `shape-form-martin/tests/test_models.py`,
`shape-form-martin/tests/test_dist_factory.py`, `shape-form-martin/results/m2a_dist/STATUS.md`. Nothing else.

**This round (review fixes B1-B5, C2):** `shape-form-martin/scripts/run_m2a_report.py`,
`shape-form-martin/scripts/run_m2a.py`, `shape-form-martin/tests/test_run_m2a.py`,
`shape-form-martin/results/m2a_dist/STATUS.md`. Nothing else was changed.

The first round (commit `c4f39fd`), for reference:

Modified: `shape-form-martin/src/sfm/models/paramnet.py`, `shape-form-martin/src/sfm/models/dist_factory.py`,
`shape-form-martin/scripts/run_m2a.py`, `shape-form-martin/scripts/run_m2a_lib.py`,
`shape-form-martin/tests/test_dist_factory.py`, `shape-form-martin/tests/test_models.py`,
`shape-form-martin/results/m2a_dist/STATUS.md`.
New: `shape-form-martin/scripts/run_m2a_report.py`, `shape-form-martin/tests/test_run_m2a.py`.
Nothing else was written in the repository. The dry-run outputs are in the scratch directory only.
