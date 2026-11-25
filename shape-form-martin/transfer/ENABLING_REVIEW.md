# Review of the H200 enabling code, batch 1 (commit `b3cbd6d`)

25 Nov 2026. Independent review (mathematics, statistics, verification). Reviewed: the code of commit `b3cbd6d`; the
files of that commit are unchanged at the `HEAD` of the review (`git diff b3cbd6d HEAD -- <files of the commit>` is
empty). `transfer/ENABLING_STATUS.md` was read as a list of claims, not as evidence. Nothing under review was edited.
Below, `<scratch>` is the reviewer's scratch directory (outside the repository); every trial ran with `SFM_OUT_ROOT`
inside it. Laptop = RTX 4060, launch grid 144. Nothing was run on the H200.

## 1. Verdict

**PASS WITH REQUIRED FIXES.**

The part that protects the scientific result holds. The tags, the reference set without fallback, the one delta
function, the refusal of untagged rows and arrays, the test-split lock and the metric (`common.py` byte-identical,
evaluations through the unmodified `common.test_nll` inside `deterministic()`, the new emulated draw bit-exact)
withstood every attempt of this review to obtain a mixed delta, selection or table (section 3.1).

What does not hold is the acceptance logic around it. Both new commands can print `ALL APPLICABLE CHECKS: PASS` and
exit 0 when a check failed or was never made (A2, A4), the reference set is created before the checks that should
license it (A3), step 5 as written fails by construction (A1), and the pristine-clone mode and the file-access guard
are off whenever a command is started without the launcher (A5). None of these produces a wrong number silently if
the H200 behaves as predicted; each of them removes a tripwire that is supposed to fire if it does not.

Section 4 gives a step list that closes A1-A5 by procedure for launch grid 1056, so steps 5-8 can run before the code
is fixed if the user prefers; the code fixes are small and should still be made, because an unattended batch is read
by its last line. **If the launch grid read in step 5 is not 1056, stop after step 5:** the fallback is incomplete
(A2 (b), A4 (c)).

## 2. Findings

### (A) Before run-book steps 6-8 on the H200

**A1. Step 5 as written cannot meet its acceptance ("all pass").**
- Evidence (executed, laptop): the worker's command through the launcher, guard active,
  `launch.sh tests .venv/bin/python -B -m pytest shape-form-martin/tests -q -p no:cacheprovider
  --ignore=.../test_geometry.py` (plus `--ignore=.../test_sr_search.py`), gives
  `3 failed, 648 passed, 6 skipped, 24 errors in 65.52s`.
  - 24 errors, `tests/test_transfer_package.py`: its fixture writes synthetic files named `pion_s1_test.pt` in a
    temporary directory; the guard refuses a deferred name wherever it is (`guard.py:128-132`).
  - 2 failures, `tests/test_guard.py`: `_run` writes `area/out/test_event_number.npy` in the pytest process itself
    (`test_guard.py:59`), where the guard is already active because earlier test modules imported `sfm`.
  - 1 failure, `tests/test_provenance.py::test_caps_defaults_follow_the_device_size`: the launcher exports
    `SFM_CPU_THREADS=4`, the test expects no thread cap on a small device. Laptop only; on a device above 24 GB the
    expected values are the launcher's defaults.
  - The same two files alone, same environment: `29 passed in 8.81s`. The failures are an order dependence
    (whether `sfm.paths` was imported in the process), not defects of the code under test.
- Expected on the H200: 26 failures and errors.
- Why it matters: the first thing the orchestrator sees on the machine of record is a red suite. Either the work
  stops, or failures are waved through and a real one (the generator test) hides among them.
- Fix: step 5 of section 4 (two pytest invocations). Better: the two fixtures create their files before the guard
  can be active, or `test_guard.py` / `test_transfer_package.py` are marked to run in a process of their own.

**A2. `val_parity.py`: four checks are computed but not part of the verdict.**
- (a) Reference set differs on recomputation. `registry.create_reference_set` returns `exists_DIFFERENT`
  (`registry.py:489`); `val_parity.py:393-429` copies the status into the report, and `r2["pass"]` is built from the
  guard targets only.
- (b) A3 in the fallback (grid not 1056). `r1["A3_native_vs_emulated"][b]["pass"]` (`val_parity.py:331-338`) is not
  in `r1["targets"]`, from which `r1["pass"]` is formed (`:358`). A3 is the only acceptance criterion of the native
  arrays in that case (PLAN 16.4, fallback (c)).
- (c) R1b for the benchmarks. `r1b["pass"]` (`:380`) is never appended to `ok`.
- (d) Identity of the comparison target. `provenance.laptop_record` returns `verified` (file SHA-256 against
  `transfer/SHA256SUMS`); `record()` (`:200-203`) loads the file whatever its value, and no pass criterion reads it.
  A forged file in a quarantine folder is returned with `verified: False` and used (executed, CPU).
- Reproducer for (a) and (b): `<scratch>/w1_valparity.py`. A valid, sealed reference set is created whose MLP array
  differs in one event by 0.25; `val_parity.A3_RMS` is set to 0.05; then `val_parity.main(["--create-refset",
  "--skip-r3"])`. Output:
  ```
  [refset] ... exists; recomputation: exists_DIFFERENT {'mlp_k64': 0.25}
  A3 native vs emulated 1056: {'mlp': {'correlation': 0.99975, 'rms_diff': 0.03316, ..., 'pass': False}, ...
  R2  reference set: exists_DIFFERENT at ...; guard targets: PASS
  ALL APPLICABLE CHECKS: PASS
  EXIT CODE 0
  ```
- Why it matters: (a) is the cross-process reproducibility check of the pairing partners of every delta of record.
  (b) decides whether a third stream becomes the stream of record. (d) is what makes "agrees with the laptop's
  emulated array" a statement about the laptop's array.
- Fix: append to `ok`: `status in ("created", "exists_identical")`, the A3 passes when the grid is not 1056,
  `r1b["pass"]`, and `info["verified"] is True` for every record used as a target; print each as its own line.

**A3. The reference set is created and sealed before any acceptance check is evaluated.**
- Evidence (reading): `registry.create_reference_set` is called inside the GPU job (`val_parity.py:251-253`); the
  verdicts of R0, R1 and R3 are formed after the job returned (`:272-388`). A run that exits 1 leaves a sealed
  reference set of the live tag, which `registry.evaluate` and `rescore_registered.py` accept; `refset.json`
  records nothing about the run that made it.
- Evidence (executed): the re-score through the launcher in the output root of the A2 reproducer, where
  `val_parity` had just reported `exists_DIFFERENT`, registered two rows against that set (`+21.64 ± 3.17` for
  `s1_seed11` instead of `+21.81`) and ended `ALL APPLICABLE CHECKS: PASS`.
- Why it matters: spec point 9 wants the reference arrays created once, before anything is registered. A set born
  from a failed parity run cannot be removed under operating rule 6 without the user, and nothing downstream knows.
- Fix: create the set only after R0, R1 (and R3) passed in the same process; write the report's SHA-256 and verdict
  into `refset.json`; `registry.reference_set` refuses a set without a passing parity record. Until then: section 4,
  steps 6 and 7 as two invocations.

**A4. `rescore_registered.py`: checks that cannot be made are skipped silently, and the verdict stays PASS.**
- (a) A laptop record that is not found: R1b (`:183-190`), the bit-identity check (`:156`) and the mass-monitor
  check (`:169-171`) are inside `if ... is not None`; `e["pass"]` is `all(v.get("pass", True) ...)` (`:192`).
- (b) `verified` is not read here either (the tracked `val_nll_sha256` of the laptop row is used only on grid 144).
- (c) On a grid that is neither 1056 nor 144 no table check exists (R4 on the emulated grid 1056 is not automated,
  as the worker states), and the result is PASS.
- Reproducer for (a): `<scratch>/w3_rescore_vacuous.py` (real reference set; `provenance.laptop_record` made to
  report "not found", as on an H200 whose quarantine folder is missing or misplaced), then
  `main(["--names", "s1_seed11", "--emulate-laptop"])`:
  ```
  s1_seed11    True scored        18.1200375   +21.81 +- 3.17 PASS
  ALL APPLICABLE CHECKS: PASS
  EXIT CODE 0
  ```
  No check is listed behind PASS. With the records present the same command lists them (`laptop_registered_array=ok;
  R1b_emulated_144_vs_laptop=ok`).
- Why it matters: R1b is the one check that separates arithmetic from fold points whatever the grid. It is the row
  most likely to be lost through a copying mistake, and that is exactly when it disappears from the verdict.
- Fix: with `--emulate-laptop` a missing or unverified record is a FAIL; require the R1b entry for all eight
  networks and the mass-monitor entry for the three guarded ones; on a third grid end with FAIL ("R4 on the emulated
  grid not implemented").

**A5. Pristine-clone mode and the file-access guard are off unless two environment variables are set.**
- Evidence (reading): `paths.OUT_ROOT` defaults to the repository (`paths.py:45`); `guard.install` is inert without
  `SFM_GUARD_AREA` (`guard.py:206-208`); `machine.startup` requires neither. A campaign command started on the H200
  without the launcher (the `uv run python ...` lines of PLAN 16.3, a quick call by hand) writes
  `results/machines/<id>/...` inside the clone, creates a second reference set and registry beside the one under
  the output root, and runs unguarded. The memory and thread caps still apply (device size). The batch record shows
  `pristine_mode: false`, `file_access_guard_active: false`; nothing refuses.
- Why it matters: PLAN 16.10 states that the guard sits "in the start-up function of every campaign command" and
  that the campaign writes no tracked or later-tracked path inside the clone. As coded, both are properties of the
  launcher. A second reference set is also what "created once" excludes.
- Fix: fail closed. On a device above 24 GB (the criterion `apply_caps` already uses), `startup` raises unless
  `paths.pristine_mode()` and `guard.active()`; an explicit, recorded opt-out for the case that is really wanted.
  Until then: section 4, "before every start".

### (B) Before steps 9-11 (training, ensembles)

**B1. The machine id changes with per-call flags and with one environment variable.**
- Evidence (executed): `machine_tag().id` = `rtx-4060-laptop-gpu-g144-629104b0`; with `deterministic=False`
  `...-e1ec1313`; with `draw="emulated"` `...-0f8aada0`; with `CUBLAS_WORKSPACE_CONFIG` set `...-9d3ef65f`.
- `train.fit` stores `machine_tag(deterministic=cfg.deterministic)` in `checkpoint.pt`, `model.pt` and
  `fit_done.json` (`train.py:430`; the default and the Track B configs have `deterministic = False`). The directory
  name, `scored_on` and the row written by `run_m2b_fit.py` use the id with `deterministic=True`.
- Consequences: `registry._trained_on_from_checkpoint` (`registry.py:596-606`, used by `fit.py`) and
  `run_m2b_ensembles._member_info` (`:89-91`) write a `trained_on` id that names no machine directory. For the
  H200-trained members of E5 the ensemble rows will say "trained on `<h200>-g1056-X`, scored on `<h200>-g1056-Y`".
- Why it matters: Gate 1 and Gate 2c check "trained on the machine of record" and "comparator trained on the
  machine of its variants"; `LOCK.json` names the machine by this id. `model.pt` and registry rows are append-only
  records, so the label cannot be repaired later.
- Fix: two ids. Machine identity (GPU, SM count, grid, torch / CUDA / cuDNN / Python, TF32 flags) names directories,
  `trained_on`, `scored_on`, `LOCK.json`. Evaluation mode (`deterministic`, `draw`, cuBLAS variable) stays in the tag
  and in `check_pairable`.

**B2. The starting point of the fine-tunes is neither verified nor recorded.**
- Evidence (reading): `from_run` (`models/__init__.py:179-182`) loads `results/m2b_mlp/s1_seed13/model.pt` from the
  clone through `load_run`, with no comparison to the laptop row's `checkpoint_sha256`. The new run's records hold
  the string `run = "m2b_mlp/s1_seed13"` in `model.pt["meta"]` and `trained_on = {"machine_id": ..., "source":
  "trained in this job"}` (`run_m2b_fit.py:86`). Step 8 verified the eight hashes at its own time; step 9 does not.
- On "does `fit`'s refusal allow exactly this and nothing else": no. `fit` refuses checkpoints and done-markers in
  its own output directory. The starting weights are a model-factory matter: any run directory in the clone can be a
  starting point, and `run_m2b_fit.py` refuses a config only by its run name (`laptop_record_done`).
- Why it matters: A6 revised, point 5, describes seeds 13-15 as "fine-tuned on the H200 from the laptop's stage-1
  networks" and identifies weights by SHA-256. From the records of step 9 Gate 1 cannot show which file a fine-tune
  started from.
- Fix: before training, verify the starting file against the laptop row; record `init_from = {run, file SHA-256,
  weights SHA-256, trained-on label, laptop row timestamp}` in `model.pt["meta"]`, `done.json` and the row's
  `trained_on`. Same for `field_mlp_std` (SHA-256 of `mlp_std.pt`) in the `b0h_*` runs.

**B3. Step 11 has no failing outcome.**
- `run_m2b_ensembles.py` writes `representative_bitwise_equal` (`:221-224`) and returns 0 whatever it holds; this is
  the acceptance of R6.
- Subset arrays are found by file name only (`subset_file`, `:152-157`); an existing file is reused without a check
  that its members are today's files, and the sidecar's `array` name is not compared on load. Executed: a tagged
  pair copied to another file name loads without complaint (trial A16). The representative check covers only the
  subsets (11,12), (11,12,13), (11,...,14).
- Member files are hashed into the row but not compared with the `checkpoint_sha256` of their own live-stream rows.
- `run_m2b_report.py:57-62` forms "B0 mean" from however many of `b0h_seed0..2` are registered, without saying so;
  A6 revised, point 5, fixes it as these three runs.
- Fix: non-zero exit unless all three representatives are bit-equal; member SHA-256 in the subset sidecar, checked
  on load together with the file name; members checked against their rows; the report refuses the B0 columns, or
  labels them incomplete, unless exactly three controls are present.

### (C) Not blocking

**C1. P3: the saved outlier material is sufficient in content, but its summary numbers decide nothing.**
- Sufficient in content (executed on four validation events, laptop): the recorded offsets equal the native draw
  bit for bit; the recorded cells `(iz, ir)` with the weights reproduce each event's NLL to 5e-6 (other summation
  order); the writer runs and produces valid files (2.6 MB for four events).
- The ULP distance is computed correctly as written (`val_parity.py:139-141`: a value one float above an edge gives
  1.0). But it is the wrong yardstick in both directions:
  - Base rate. Of 300 random validation events, 80 (27 %) have a fold point within 2 ULP of a bin edge and 196
    (65 %) within 12 ULP (median 16,928 points per event). `n_points_within_2ulp_of_an_edge > 0` confirms nothing.
  - Reach. A last-bit difference arises in the rotated offset (about 30 mm, ULP 1.9e-6 mm), not in the result. Near
    the inner ring edges (1.93, 6.28, 12.5 mm) a one-ULP change of the offsets moves `r` = 2.4 mm by 2.9e-6 mm,
    which is 12 ULP of `r` (numeric example, float32). The same holds for `z'` near 0. A flip can therefore show a
    distance well above 2.
  - For `z'`, `common.locate` bins by `floor((z - Z_MIN) / DZ)`; at 57 of the 127 inner edges the float32 boundary
    of that expression is not the float32 edge value, so the distance is uncertain by one ULP.
- What decides: re-derive the same events on the laptop with `EmulGen(12345, 1056)`, compare `(iz, ir)` point by
  point with the H200's record, and show that the points whose cell differs account for the NLL difference
  (`-w_i [ln g_i(H200) - ln g_i(laptop)]`). No differing cell means it is not a bin-edge flip and the strict clause
  stays.
- Proposed additions: save per-hit `ln g_i`; give the edge distance in units of the ULP of the larger operand, or
  the float64 distance in mm; write the comparison script before step 6 runs, not after a banner.
- Also: `val_parity.py:343` tags the emulated grid-1056 arrays with the native tag (in memory only, never saved).

**C2. Guard: limits not stated, and one condition for legitimate operation.**
- Executed (`<scratch>/guard_probe.py`, temporary directories). Refused, as claimed: `open` (absolute, relative),
  `os.open`, `np.load`, `np.memmap`, a symbolic link into the other folder, `subprocess` with a path argument or a
  foreign working directory, deferred names anywhere, `pyarrow.parquet`, `pandas.read_parquet` (no engine),
  `enable_final_test_access()` without `LOCK.json`.
- Not seen by the hook, and not in the documented limits: `os.rename` of another project's file into the output
  root, `os.remove` of it, `os.mkdir` in its folder, a hard link followed by `open` (contents read), `os.system`,
  and `guard._STATE["allow_deferred"] = True`. The campaign code does none of this; operating rules 1 and 6 are
  about exactly these operations. Add the audit events `os.remove`, `os.rename`, `os.mkdir`, `os.rmdir`, `os.link`,
  `os.symlink`, `os.truncate`, `os.chmod`, `os.system`, `shutil.rmtree`, `shutil.move`, and name the rest as limits.
- A child-process argument containing `/` is read as a path relative to the working directory:
  `subprocess.run([python, "-c", "print(1/2)"])` is refused with the working directory in the area outside the
  clone, allowed from the clone root. Commands must be started from the clone root (section 4).
- The release of the deferred names cannot be triggered by an environment variable: it needs
  `results/final/LOCK.json` in the clone naming the live machine id (`data.py:263-289`).

**C3. Launcher (`scripts/launch.sh`).**
- The PID written is the Python process (`nohup` executes the command; executed: the PID in the file is the one
  `nvidia-smi` lists). With `uv run` it is the PID of `uv`, the interpreter is its child (executed): never start
  `uv run` through the launcher.
- A command that fails at once is reported as "started job", exit 0, with the PID of a dead process in the file
  (executed with a non-existent interpreter). Add a short liveness check, and write the exit status at the end.
- "Stop with `kill "$(cat <pid file>)"`" omits what operating rule 10 demands (check command line and working
  directory first). Store the start time from `/proc/<pid>/stat` beside the PID and compare before a signal.
- Job logs receive Python tracebacks and the `# command:` line, both with absolute local paths. They are not for
  commit and must not be pasted into the repository.
- Missing conveniences: `PYTHONUNBUFFERED=1`, `setsid`, `export SFM_USER_CACHE` (today it fails closed if the
  variable is not exported, which is acceptable).

**C4. What the dirty-clone check tests.** `git status --porcelain` once, at start-up, in pristine mode only.
- It does not see ignored files (`results/**/*.npy`, `*.pt`, `*.log`, `cache/`), does not compare `HEAD` with a
  pushed commit (PLAN 16.10 wants that recorded), and is not repeated at the end of a job.
- `git status` may refresh the index; use `git --no-optional-locks status` so that two jobs never collide on
  `index.lock` (a failed call is treated as dirty and refuses the start).
- `SFM_ALLOW_DIRTY_CLONE=1` leaves a trace only as `git.clean: false` in rows and reports; the variable itself, like
  `SFM_GUARD_AREA=off`, `SFM_GPU_MEM_GB` and `SFM_CPU_THREADS`, is not written down. Record them verbatim in the
  batch record. `SFM_ALLOW_RECORD_WRITER` has no effect when `SFM_OUT_ROOT` is set, and `SFM_ALLOW_CPU` is honoured
  only by `train.fit` when CUDA is absent, with `gpu_name: "cpu"` in the tag; neither can weaken an H200 result.

**C5. Smaller points.**
- `registry.registered_row(..., stream_id=None, path=LEGACY_REGISTRY)` returns a laptop row (executed); no caller
  passes `None`. `latest()` and `table()` return untagged rows by default if a file contains them; `row_array` and
  `select_1se` refuse them. Reject `None` and filter by default.
- The points tag holds asserted constants (seed 12345, batch 128, K 64, `fold: "iid"`); they equal those of
  `common.test_nll` today (`common.py:311-314`) but are not read from it. The SHA-256 of `common.py` is in no tag or
  row; put it in the row.
- `rescore_registered.py:181-182` adds a criterion that the plan does not have (relative mean of the mass monitors
  within 1e-6); it may fail across machines without meaning anything.
- The memory cap limits PyTorch's allocator, not the CUDA context (a few hundred MB more in `nvidia-smi`).
- `registry.py:29-30` (docstring) names a path layout that the code does not use.
- A re-fit under an existing run name stops at `FileExistsError` in `save_tagged`; correct, but the way out (another
  run name) should be in the run-book, since nothing may be deleted on the H200.
- `scripts/run_m2a.py:172` still pairs with `np.load(M0_DIR / "formula_val_nll.npy")` through a bare
  `stats.paired`. It is outside this batch (E4, being changed by another worker) and outside steps 5-11; on the H200
  it fails loudly (the file is in quarantine). It is the one fallback-type pairing left in the code base.
- The test-access log stays inside the clone (`paths.TEST_ACCESS_LOG`), in conflict with the pristine clone; the
  worker names this as open for E8. Not reachable in steps 5-11.

## 3. What was verified, and how

### 3.1 By execution (laptop, validation only, scratch output roots)

**Metric integrity.**
- `common.py`: SHA-256 `ece3c67f...dbebb3` in the working tree, at `HEAD`, `b3cbd6d`, `c9b7f72`, `1f94b57` and the
  Milestone 0 commit `24305cb`.
- `refstream.patched_draw` (`<scratch>/g1_trials.py`, one process under the GPU lock, 23 s):
  - offsets equal `(torch.rand(H, K, 3, generator=g) - 0.5) * box` bit for bit, K = 16, 64, 1024, four consecutive
    batches each;
  - K = 1024 (chunk 64), first 2,560 events: native = emulated grid 144 = the stored
    `results/registry_refs/mlp_val_k1024.npy`, bit for bit;
  - bulk on the stratum (14,784 events), first 2,560: native = emulated = the stored `mlp_val_bulk_stratum.npy`;
  - K = 64, all 14,950 events: `common.test_nll` inside `deterministic()` = `metric.test_nll` under
    `patched_draw(144)`; `patched_draw(1056)` = the stored `mlp_val_nll_ref1056.npy`, float32 mean
    18.118083953857422.
- `val_parity.py` on the laptop (fallback path), independent run: float32 means 18.118083953857422 and
  18.120407104492188 exactly; formula − MLP 0.00232442 ± 0.00029427; K = 1024 and bulk means within 4e-8 of the
  laptop values; emulated grid-1056 guards PASS; second run bit-identical.

**Mis-pairing (all refused unless noted).**
- `registry.evaluate` without a reference set, and with untagged laptop arrays dropped at the reference-set path:
  `NoReferenceSet`.
- A laptop `val_nll.npy` written over a live tagged array: `UntaggedArray` from `row_array` and from `select_1se`.
- `row_array` and `select_1se` on the laptop rows, alone or mixed with a live row: `UntaggedRow`;
  `select_1se(path=LEGACY_REGISTRY)`: nothing selected.
- A laptop row injected into the machine registry: ignored by `select_1se` and by `latest(stream=live)`.
- The same values tagged as another machine: `MixedTags` from `paired_delta` and from `select_1se`.
- K = 64 against an array tagged K = 1024, and a bare array: `MixedTags`.
- Append to the laptop registry: `PermissionError`.
- Sealing: files 0444, directory 0555; a second creation: `exists_identical`.
- Not refused: `registered_row(stream_id=None)` and a renamed array-plus-sidecar pair (C5, B3).

**Test-split lock.** With the audit hook active in the pytest process, the guard log of the whole suite holds 29
refusals: 26 synthetic files in temporary directories and 3 `pyarrow` imports. No test tried to open the real test
cache, a deferred file of the repository or the raw table. `load_sets(allow_test=True)` is refused (no `LOCK.json`).

**Pristine clone.** A real re-score through the launcher, guard active (`rescore_registered.py --names s1_seed11
b0_seed0 --emulate-laptop`, 26 s): arrays bit-identical to the laptop's registered ones, guards and mass monitors
ok, R1b ok; the only file touched in the clone is the git-ignored `cache/gpu.lock`; guard log: one `pyarrow` import.
No JSON under the output root contains a host name or an absolute path.

**Resource caps** (`<scratch>/g4_caps.py`, under the GPU lock).
- `SFM_GPU_MEM_GB=2 SFM_CPU_THREADS=3`: fraction 0.2607 of the measured 7.673 GiB; 1 GiB allocated; a further
  2.5 GiB refused by PyTorch although 6.4 GiB of the device were free; `torch.get_num_threads()` 3.
- `SFM_GPU_MEM_GB=6`: `check_free_memory` raises `InsufficientGPUMemory` at once (7.4 GiB free, 10 needed).

**The test suite.**
- Cause of the 14 minutes: `tests/test_geometry.py:353` takes the real campaign lock with `timeout=1800`. A wrapper
  that holds the lock around `pytest` makes that test wait 30 minutes for its own parent, then fail. Reproduced with
  a scratch lock file held by another process (the test was still waiting when killed after 30 s).
- Without a held lock and without the guard: `721 passed, 6 skipped in 93.88s` (`test_sr_search.py` and the
  untracked `test_sr_post.py` left out). The skips are template and soft-start tests, none for missing data.
- Under the launcher with the guard, without `test_geometry.py`: 65.5 s, 3 failed and 24 errors (A1). Peak GPU
  memory of the pytest process 904 MiB.
- Caveat: other workers changed files outside this commit during the review (the PySR modules, Track A model code
  and their tests). The counts above are those of the working tree at the time of each run; the files of `b3cbd6d`
  were unchanged throughout.

**Laptop regressions.** The suite passes; the eight laptop rows are readable and refused as pairing partners;
`results/registry.jsonl` (`017edeba...81de`) and `results/test_access.log` (`effa9734...0c60`) have the same
SHA-256 after the review as before it.

### 3.2 By reading only

- Statistics: `stats.paired` (float64 difference, sd with `ddof=1` over √n) is unchanged; the guard statistics of
  `registry._guards` are the same expressions as before the commit, now on tagged arrays.
- `provenance.tolerance_check` reports differences and counts and never an SE. Its relaxed clause is the plan's
  (at most three events above 1e-4, none above 1e-3, rms at most 1e-5, means within 1e-6); the P3 flag is "any event
  above 5e-5".
- Targets coded in the two commands equal the plan's: 4e-6 on the float32 means, 1e-4 and 1e-6 per event and on the
  mean, formula − MLP +0.002324 ± 0.000294, K = 1024 and bulk means, 14,784, the table of 16.3, guard table D.
- The points tag (grid, seed, K, batch, fold, split, fingerprint over every tensor of the validation set, subset,
  chunk, bulk) determines the fold points and the event set and order (remarks in C5).
- `b0h_seed{0,1,2}.toml` differ from `b0_seed{0,1,2}.toml` in `run.name` and `run.notes` only (diffed).
- Every write of steps 6-11 goes under the output root or the git-ignored `cache/` (all write sites listed).
- All 70 files of the two "now" transfer lists are git-ignored, so the copy leaves `git status` empty.
- Steps 9-11 were not run (they train). The code paths of `run_m2b_fit.py`, `run_m2b_ensembles.py` and
  `run_m2b_report.py` were read, not executed.

### 3.3 Only the H200 can settle

- Launch grid; `EmulGen` against `torch.rand` on that GPU and build; arithmetic parity with the laptop's emulated
  arrays (whether a P3 banner appears); the predicted table.
- `tests/test_refstream.py` on a grid-1056 GPU (written to pass, never run there); the CUDA tolerance tests of
  `tests/test_train.py`; the cost test, which skips on a busy GPU.
- Whether every job fits under the 10 GB cap there (they fit in 8 GB on the laptop); `DirtyClone` against a clone
  that really is clean (the laptop's never was during this review, so every trial used `SFM_ALLOW_DIRTY_CLONE=1`);
  `flock` on that file system.
- The guard with the real layout: interpreter outside the area, virtual environment inside the clone, the launcher's
  default area (the parent of the clone) being the user's folder.
- Durations.

## 4. Corrected step list for the H200 (steps 5-8; 9-11 after the (B) fixes)

**Before every start**
- `cd <repo>` (the guard reads child-process arguments relative to the working directory, C2).
- `export SFM_OUT_ROOT=<area>/sfm_out SFM_USER_CACHE=<area>/sfm_cache`; `L=shape-form-martin/scripts/launch.sh`;
  `P=.venv/bin/python`. Every campaign command goes through `$L` with `$P`; never `uv run`, never a bare
  `python ... scripts/...` (A5).
- Do not set `SFM_RESULTS_DIR`, `SFM_CACHE_DIR`, `SFM_DATA_DIR`, `SFM_ALLOW_DIRTY_CLONE`, `SFM_ALLOW_RECORD_WRITER`,
  `SFM_ALLOW_CPU`, `SFM_GUARD_AREA`, `CUBLAS_WORKSPACE_CONFIG` (the last one changes the machine id, B1).
- Read `nvidia-smi` (rule 3). `git status --porcelain` empty; `git rev-parse HEAD` equal to
  `git rev-parse origin/shape-form-martin` after a fetch.
- One job at a time. After each: wait until `kill -0 "$(cat <pid file>)"` fails and read the log to its end. For the
  campaign commands (steps 6-11) the first line after the header must show `outputs under <out>/...` (not
  `<repo>/...`) and `file-access guard ACTIVE`.
- To stop a job: read `/proc/<pid>/cmdline` and `/proc/<pid>/cwd` first, then `kill <pid>` (rule 10).

**The command lines of `ENABLING_STATUS.md` section 2 against the operating rules of PLAN 16.10**

| Rule | Status |
|---|---|
| 1, 8 (only inside the area; caches redirected) | met through the launcher (pip, XDG, matplotlib, CUDA cache, `TORCH_HOME`, `TMPDIR`); not met by a command started without it (A5) |
| 2, 9 (detached, PID file, log) | met; an immediate failure is not reported (C3) |
| 3 (read `nvidia-smi` before every start, no waiting) | no waiting: met in code (`check_free_memory` refuses below cap + 4 GB). The `nvidia-smi` reading is not in the list; added above |
| 4 (10 GB and few threads per job, one job unless a gain is shown) | met by default on a device above 24 GB (fraction from the measured total, 4 threads); the campaign lock keeps one job at a time. The pytest process of step 5 is not capped until a test calls `machine.startup` (peak on the laptop 0.9 GB) |
| 7 (git: read-only) | met: the code calls `git rev-parse HEAD` and `git status --porcelain` only (C4 for `--no-optional-locks`) |
| 10 (signal only a recorded PID, after checking command line and working directory) | the PID is recorded; the documented stop command skips the check (C3) |
| pristine clone, `HEAD` equal to a pushed commit recorded with every batch | clean flag and `HEAD` are recorded; "pushed" is not checked (added above) |
| step 5 | fails as written (A1) |
| steps 6-7 in one invocation | to be split (A3) |

Laptop durations of the commands as run in this review: `val_parity.py --create-refset --skip-r3` 91 s (with the
emulated guards, which a grid-1056 machine skips); the re-score of two networks with R1b 26 s (eight: about 1.5 min).

| Step | Command | Acceptance (read each item; the last line alone is not enough until A2-A4 are fixed) |
|---|---|---|
| 5a | `$L tests_core $P -B -m pytest shape-form-martin/tests/test_refstream.py shape-form-martin/tests/test_provenance.py shape-form-martin/tests/test_registry.py shape-form-martin/tests/test_metric.py shape-form-martin/tests/test_data.py shape-form-martin/tests/test_models.py shape-form-martin/tests/test_train.py -q -p no:cacheprovider -rs` | all pass, `test_emulation_reproduces_torch_rand_bitwise` among them. A failure in `test_train.py` (CUDA tolerances measured on the laptop) or a skip of a busy-GPU test is reported and does not block |
| 5b | `$L tests_guard $P -B -m pytest shape-form-martin/tests/test_guard.py shape-form-martin/tests/test_transfer_package.py -q -p no:cacheprovider` | 29 passed. A process of its own, because the fixtures create synthetic files with deferred names (A1) |
| 5c | optional: `$L tests_rest $P -B -m pytest shape-form-martin/tests -q -p no:cacheprovider -rs` with `--ignore=` for the nine files of 5a and 5b, `test_geometry.py`, `test_sr_search.py` (and `test_sr_post.py` once tracked) | informational: mathematics developed on the laptop, no statement about the H200 |
| 6 | `$L val_parity $P -B shape-form-martin/scripts/val_parity.py` (no `--create-refset`) | launch grid 1056, else stop; R0 PASS; second run bit-identical for both benchmarks; every R1 target PASS, `emulated_equals_native` included; no P3 banner; `R1b benchmarks ... PASS`; R3 PASS; in `checks/val_parity/report_*.json` both `vs_laptop_ref1056.<b>.record.verified` are `true` |
| 7 | only if 6 passed: `$L refset $P -B shape-form-martin/scripts/val_parity.py --create-refset`, then the same command once more | first run: `R2 reference set: created`, all R2 targets PASS, 14,784. Second run: `R2 reference set: exists_identical`. `exists_DIFFERENT` is a stop, whatever the last line says |
| 8 | `$L rescore $P -B shape-form-martin/scripts/rescore_registered.py --emulate-laptop` | `sha ok` True for all eight; for each network the summary lists `predicted_mean=ok`, `predicted_delta_vs_mlp_e4=ok` and `R1b_emulated_144_vs_laptop=ok`; for `b0_seed0`, `s2_seed11`, `s2_seed12` also `predicted_k1024_mean=ok`, `predicted_bulk_mean=ok`, `mass_monitors=ok`; in the report every `laptop_record.verified` is `true`. A network with nothing listed behind PASS has not been checked (A4) |
| 9-11 | as in `ENABLING_STATUS.md` section 2, after B1-B3 | step 11 additionally: `representative_bitwise_equal` all true in `ensemble_scaling.json`; `b0_scatter.names` holds the three `b0h_*` runs |

**On running the suite on the shared machine.** It is cheap: about 65 to 95 s on the laptop, under 1 GB of GPU
memory, four CPU threads under the launcher. Its CUDA tests take no campaign lock and are not capped until one of
them calls `machine.startup`. It is worth running there for 5a and 5b; the remainder says nothing about the H200.
It must never be wrapped in a process that holds the GPU lock.

---

## 5. Re-check of commit `0c38b32` (25 Nov 2026, second pass)

Reviewed: the 22 files of commit `0c38b32` (unchanged in the working tree during the re-check). The worker's account
(`ENABLING_STATUS.md` sections 2, 7, 8, 9) was again treated as claims. Method per finding: my reproducer of the
first pass re-run (adapted to the new interface where needed), one variation, and the fix read. Reproducers of this
pass: `<scratch>/r2/` (`vp.py`, `rs.py`, `g5_recheck.py`, `run_gpu.sh`, `run_e.sh`). Same rules as before; nothing
ran on the H200.

### 5.1 Verdicts

- **Steps 5-8: GO**, with the step list of 5.6 and launch grid 1056. If step 6 reads another grid: stop after step 6
  (the fallback for the re-score is still not implemented; the command now refuses such a grid).
- **Steps 9-11: GO**, with the step list of 5.6.

All eight findings A1-A5 and B1-B3 are closed. The re-check found six new points, all ranked (C); N1 changes how
one line of the step list has to be read and is worth a two-line fix before the first ssh session.

### 5.2 A1-A5 and B1-B3, one line each

| Finding | Status | Evidence of this pass | Variation tried |
|---|---|---|---|
| A1 step 5 red by construction | **closed** | step 5a through the launcher, guard on: `107 passed in 39.78s`, peak GPU memory 544 MiB; step 5b: `26 passed in 2.14s`. Whole suite in one process, no guard: `749 passed, 8 skipped` in 110 s | the launcher reports the 2-second job 5b as `FAILED` (N1) |
| A2 verdict ignored four checks | **closed** | `verdict_items` is the only source of the exit code. On a real report every one of 13 single flipped checks (R0, second run, a mean, a per-event target, both kinds of record identity, A3, R1b, R3, emulated R2, set status, an R2 target, the stratum count) gives exactly one FAIL line | a recomputation that differs by one ULP in one event: `exists_DIFFERENT`, FAIL, exit 1. A record failing the hash check: FAIL, no set. Not part of the verdict: a P3 banner, and R3 when `--skip-r3` is given (N2) |
| A3 set created before the checks | **closed** | forced R0 failure with `--create-refset` on a fresh root: `NOT created`, exit 1, no `refsets/` directory. Same for an unverified record. A created set carries the parity record; the SHA-256 of the parity report on disk equals `parity.report_sha256` | after the differing recomputation a marker `<set>.DIFFERENT_<utc>.json` exists and the re-score stops with `ReferenceSetDisputed`. The licence is a field of a dictionary: the interface accepts any `parity={"verdict": "PASS"}` (5.5) |
| A4 silent skips in the re-score | **closed** | records "not found": every required check `NOT MADE`, `INCOMPLETE`, exit 2. Normal run: PASS with each required check listed (`+21.81 ± 3.17`, `+1.19 ± 0.42`, the laptop's numbers) | records present but failing the hash check (one array, one mass-monitor file): `NOT MADE`, exit 2. Without `--emulate-laptop` R1b is not required and the verdict is PASS: the flag must be in the command (it is, step 8) |
| A5 fail-open pristine mode and guard | **closed** | with the policy list emptied (this machine "not the laptop"): `UnguardedStart` from `machine.startup`, `registry.evaluate`, `create_reference_set` and `train.fit`; also with the output root outside but the guard off | a tag with another Python version is not the laptop (fail closed). The policy accepts any active guard area, including one that does not contain the clone (N4) |
| B1 machine id moved with settings | **closed** | id identical for `deterministic=False`, `draw="emulated"` and a set cuBLAS variable; different for another Python or TF32 on. After a real (shortened) fit through the launcher the id is the same in the row (`scored_on`, `trained_on`), `model.pt` (`deterministic: False`), `fit_done.json`, `done.json` | pairing is as strict as before: `paired_delta` raises `MixedTags` for each of the three settings. One check became weaker: `save_tagged` onto an existing file (N3) |
| B2 starting point unverified | **closed for `run_m2b_fit.py`** | `init_from` (file and weights SHA-256, source row) is in the row, `model.pt` and `done.json` of the real run; step 9-pre through the launcher: `starting point verified`, `built weights agree`, nothing written | refused: another stage-1 seed, another spelling of the same run, another factory, a name outside the eight, a file at the right path with other weights, a model changed after the build. Not pinned: the recipe and the location of the config file (N5). `fit.py` and `fit_queue.py` do not call the check (later batch, run-book step 14) |
| B3 step 11 had no failing outcome | **closed** | renamed array-and-sidecar pair: `UntaggedArray`; `representatives_verdict` false for a failing or missing representative; report with no control: banner, `b0_controls_complete` false, exit 2; ensembles without the members: exit 1 | a pair renamed with its sidecar's `array` field edited as well loads (a sidecar is a consistency check, not a signature; for the subset files the member SHA-256 and the representatives stand behind it). The ensemble path with all members present was read, not run |

### 5.3 The (C) items

- **C1 (P3), fixed, and it is the deciding test.** `scripts/p3_compare.py` re-derives the events on the laptop, compares
  `(iz, ir)` point by point and attributes the difference with the laptop's log rho held fixed.
  - Run on events of my choice with genuine flips. Another arithmetic was imitated by computing the fold-point
    coordinates in float64 and rounding; 36 of the 14,950 validation events then have one point in another cell. For
    seven of them the script finds exactly that one point and reproduces my independently computed NLL difference
    to 3e-8 or better ("flip: explained"); two events without a flip give "no differing cell: NOT a bin-edge flip".
  - The path writer to reader works: material written by the real `save_outliers` (identical cells, a fake target)
    is read and gives "no differing cell", residual equal to the fake difference.
  - Not right yet: the label (N6).
  - For the user and Gate 2c, not a code finding: in that imitation the flips alone move 5 events by more than 1e-4
    (largest 2.5e-4; rms 3e-6). The relaxed clause allows three. The imitation changes every coordinate's rounding
    and is therefore an upper-range case, but it shows that confirmed bin-edge flips and "at most three events above
    1e-4" are not the same condition.
- **C2 (guard events), fixed.** Refused now, the other project's files untouched: rename, replace, remove, unlink,
  mkdir, makedirs, rmdir, hard link, chmod, truncate, utime, `shutil.move`, `shutil.rmtree`, `os.system` (also with
  redirection or `cd`). Legitimate renames, removals and temporary files inside the output root and the cache pass.
  Left, and acceptable as documented limits: a hard link that already exists, `guard._STATE`. Left and not
  documented: a shell command given to `subprocess` as one string (`os.popen`, `shell=True`) is read as a single
  path and passes from the clone root (executed: the foreign file was read). The campaign code has no such call.
- **C3 (launcher, stop helper), fixed except N1.** The PID is the interpreter's; `.job` holds PID, start time,
  working directory and command line; `uv` is refused; a command that cannot start is `FAILED`, exit 1.
  `stop_job.sh` against a PID that is not the job's: refused when the PID file names another live process of mine,
  when the `.job` file was edited to that PID as well, and when its start time was forged too (the command line
  still differs); the other process stayed alive. The genuine job received SIGTERM and its exit status (143) was
  recorded; a second call after its end signals nothing.
- **C4, fixed.** `--no-optional-locks`; `upstream` and `head_equals_upstream` in every git record (seen in the row of
  the real run); the switches in force are in the batch record (`SFM_ALLOW_DIRTY_CLONE: "1"`, `SFM_GUARD_AREA:
  "<area>"`); git state again at the end of a batch.
- **C5, fixed in part, the rest acceptable.** `registered_row(stream_id=None)` raises; `common_py_sha256` is in rows
  and `refset.json`. Left as it was, with reasons I accept: `latest()` and `table()` defaults (callers of other
  workers; `row_array` and `select_1se` refuse untagged rows), the mass-monitor mean criterion (it is in the
  hand-over text, section 1.2 of `REVIEW_STATUS.md`; my remark was wrong on that point), the asserted constants of
  the points tag (now pinned by the hash of `common.py`), the test-access log inside the clone (E8).

### 5.4 New findings (all C)

**N1. The launcher reports a job that succeeds within the grace period as FAILED.**
- Evidence: `launch.sh quick true` prints `FAILED: job quick ended within 3 s (exit status 0)` and returns 1. Step
  5b as written did the same on the laptop (`26 passed in 2.14s`), and so did a repeated step 9 whose run was
  already done. `launch.sh:110-114` tests only whether the job has ended.
- Consequence: the first or second line the orchestrator reads over ssh is a false alarm, and a script that waits
  for the `pid file:` line never gets one (my own driver hung on exactly this).
- Fix: when the `.exit` file exists and holds 0, print `finished within <n> s (exit status 0)` and return 0; FAILED
  only for a non-zero or missing status. Until then: 5.6.

**N2. Two options of `val_parity.py` weaken the parity that licenses a reference set.** `--skip-r3` removes R3 from
the verdict (30 items instead of 32, still PASS). `--no-guards` creates a set without guard arrays, and a later
normal run then finds the guard arrays "missing in the existing set", writes the dispute marker and blocks the set.
Fix: refuse both together with `--create-refset` off the laptop of record. The step list uses neither. A P3 banner
does not change the exit code either; the plan allows the set in that case (every event within 1e-4), so this is a
remark: a distinct exit code would stop an unattended batch at step 6.

**N3. `save_tagged` no longer notices another setting on an existing file.** It compares the sidecar's
`machine_id`, which used to contain the settings. Executed: the same values saved with `draw="emulated"` onto a
native file return a tag that says "emulated" while the sidecar says "native". No campaign code saves emulated
arrays today. Fix: compare the sidecar's `machine` fields with the tag's.

**N4. The policy accepts any active guard.** `require_guarded_start` asks `guard.active()`; an area that does not
contain the clone satisfies it. The launcher's default is right and the step list forbids setting the variable.
Fix: also require the clone to lie inside the guarded area. Related: a process without CUDA is exempt, so on the
H200 `CUDA_VISIBLE_DEVICES=""` with `SFM_ALLOW_CPU=1` would let `train.fit` run unguarded; two deliberate settings.

**N5. `run_m2b_fit.py` pins the starting point, not the recipe.** A copy of `s2_seed13.toml` with 500 steps, kept
outside the clone, was accepted, trained and registered under the name `s2_seed13` (config hash `92c7e44e...`
instead of `0354cc66...`; its absolute path stands in `done.json`). On the H200 the configs are tracked files in a
clean clone, and Gate 1 can compare the hash. Fix: pin the config hashes of the six pre-registered files beside
`ALLOWED_INIT`, or accept only config files inside the clone.

**N6. Two details of the new text handling.**
- `p3_compare.py` labels an event "flip: explained" when the residual is at most 3e-5. Executed: a recorded
  difference of -8.8e-5 of which the flip explains -1.17e-4 (residual +2.9e-5) gets the label. With a banner
  threshold of 5e-5 that is too wide. The numbers are printed and the decision is not the script's; the label
  should demand a residual at the level of events without a differing cell (which the script already measures as
  `max_abs_logg_diff_on_identical_cells` and does not use) and well below the recorded difference.
- `paths.scrub_str` now replaces a root only before `/`, a quote, white space and a few other characters. A path
  followed by `.`, `>`, `}` or `|` is left as it is (executed: `see <output root>.` stays absolute). The earlier
  code replaced unconditionally. No such string was found in the JSON files of this pass; 5.6 adds a search before
  anything is copied back.

### 5.5 The risks asked about

- **Can the laptop lock itself out?** Yes, by design: the policy names it by an id that contains the Python patch
  version, torch, CUDA, cuDNN and both TF32 flags. After an upgrade every CUDA start-up outside the launcher raises
  `UnguardedStart` until `policy.py` is updated. `startup` and `fit` switch TF32 off before the id is read, so a
  flag set elsewhere does not cause it.
- **Can another machine pass as the laptop?** Only a GPU of the same model with the same builds; the host is not
  part of the id. The H200 cannot.
- **Unit tests and CPU-only processes.** Under the launcher the policy is met, so step 5a is not affected (107
  passed with guard and scratch output root here). No test asserts that it runs on the laptop. Without the launcher
  on another CUDA machine the tests that start fits or evaluations fail with `UnguardedStart`, as intended.
- **The parity record.** Through `val_parity.py` a set cannot be created from a failed run (three failure modes
  tried). A foreign report cannot be attached there either: the record is built in the same process from that
  run's own report. Through the function itself the record is a claim (the unit tests pass
  `{"verdict": "PASS", "source": "unit test"}`); what makes it checkable is the report SHA-256 in `refset.json`,
  which I confirmed against the file. The report does not hold the SHA-256 of the arrays it checked; adding them
  would let a gate tie report and arrays together from the report alone.
- **The blocking marker.** Written beside the sealed set by a differing recomputation; `reference_set` (hence
  `evaluate`, the re-score, Track A) refuses the set while any marker exists. Nothing removes it by code. It can
  also be caused without a numerical difference by the option of N2.
- **The new machine id.** The split is right: identity names directories, `trained_on`, `scored_on` and the lock
  file; settings stay in the tag and in `check_pairable`. `load_tagged(expect_machine_id=...)` alone no longer
  separates settings; every use I found is followed by a pairing check on the full fields. N3 is the one place
  that lost strictness.
- **Starting points.** For `run_m2b_fit.py` exactly the pre-registered ones: `s2_seed11..15` from the stage-1 run
  of the same seed, `b0h_seed0..2` from `mlp_std.pt`, each verified by SHA-256 before the build and by weight hash
  after it. See N5 for what is not pinned.
- **Test-split lock.** Not touched by the fixes (`data.py` is not in the commit; the deferred names and their
  release are unchanged). `check_test_access_allowed` goes through `startup`, so on the H200 a test access now also
  needs the launcher's settings. The guard log of every guarded run of this pass holds `pyarrow` import refusals
  only.
- **Writes into the clone.** None added. The marker, the parity report, the job files and the P3 report go under
  the output root. In each launcher run of this pass the only file touched in the clone was the git-ignored
  `cache/gpu.lock`.

**Track A (`scripts/run_m2a*.py`, commit `c4f39fd`; not my subject).** I found no silent change of behaviour. The
changes it will meet are all loud: reference sets made before this commit have no parity record and are refused;
machine directories under the old id are not found; `create_reference_set` needs `parity=` (Track A does not call
it); `registered_row` needs a stream id (it passes one); off the laptop a start without the launcher is refused.
Its own markers compare the directory's machine id, which is now stable. Two things for its reviewer:
`run_m2a_report.py:169-170` wraps arrays read from files in the live tag with `TaggedArray.make`, and
`expect_machine_id` no longer distinguishes settings (the pairing check behind it still does).

### 5.6 Step list endorsed (H200, steps 5-11)

The list of `ENABLING_STATUS.md` section 2 is right as written, with these amendments.

**Before every start** (as there): `cd <repo>`; `export SFM_OUT_ROOT=... SFM_USER_CACHE=...`; `L`, `S`, `P`;
`nvidia-smi`; `git status --porcelain` empty and `HEAD` equal to the pushed branch. In addition:
- Pass no option that the list does not show: no `--skip-r3`, `--no-guards`, `--emulated-guards`,
  `--allow-unchecked-grid`, `--names`, `--source-registry`.
- Use only the tracked config files of `shape-form-martin/configs/m2b/` (N5).
- A job is finished when `<job>.exit` exists. Its content is the result: read it and the end of the log. If the
  launcher prints `FAILED: job ... ended within 3 s (exit status 0)`, the job has finished successfully (N1).
- After every campaign command the log must not contain `WARNING: the clone is no longer clean`.

| Step | Command | Read in the log | Stop when |
|---|---|---|---|
| 5a | as written (eight test files, one process) | `N passed`; `.exit` = 0 | any failure other than a CUDA tolerance test of `test_train.py`; `test_emulation_reproduces_torch_rand_bitwise` must be among the passed |
| 5b | as written (`test_transfer_package.py`) | `26 passed`; `.exit` = 0. The launcher will probably say FAILED (N1) | `.exit` not 0 |
| 6 | `$L val_parity $P -B shape-form-martin/scripts/val_parity.py` | first line: `outputs under <out>/...`, `file-access guard ACTIVE`; `launch grid 1056`; every verdict line PASS, among them `R1 emulated_equals_native`, four `laptop record verified` lines, two R1b and two R3 lines; `P3: no validation event differs ...`; no line beginning `R2 reference set`; `ALL APPLICABLE CHECKS: PASS` | grid not 1056; any FAIL; a P3 banner (the exit code is 0 with a banner: copy `checks/val_parity/outliers_*` back and run `scripts/p3_compare.py` on the laptop before step 7) |
| 7 | only after 6: the same command with `--create-refset`, twice | first run: `[refset] created ...; sealed read-only`, `R2 reference set status: created`, eight further R2 lines PASS. Second run: `R2 reference set status: exists_identical`, exit 0 | `NOT created`; `exists_DIFFERENT` or `BLOCKED` (the set is then refused everywhere; report to the user, delete nothing) |
| 8 | as written, with `--emulate-laptop` | per network `sha ok` and `-> PASS`, with `predicted_mean`, `predicted_delta_vs_mlp_e4` and `R1b_emulated_144_vs_laptop` listed `ok`; for `b0_seed0`, `s2_seed11`, `s2_seed12` also `predicted_k1024_mean`, `predicted_bulk_mean`, `mass_monitors`; `ALL APPLICABLE CHECKS: PASS`, `.exit` = 0 | `INCOMPLETE` (`.exit` = 2: a quarantine record is missing or fails its hash; fix the copy, run again) or `FAIL`. A FAIL of `mass_monitors` alone is reported with its per-key numbers before anything else is decided |
| 9-pre | as written (six configs, `--verify-init-only`) | six `starting point verified` lines and six `verify-only: starting point and built weights agree` lines; no `runs/` entry for these names appears | any `SystemExit` text |
| 9 | as written | per run: `starting point verified`, `start: 6000 steps`, `registered: ...`, `done: ...`; three rows whose `trained_on.init_from.file_sha256` equals the laptop row of `s1_seed13..15` | `MachineMismatch`, `FileExistsError`, `InsufficientGPUMemory` |
| 10 | as written | as 9, starting file `mlp_std.pt`; none of the three `checkpoint_sha256` begins `75de31f2` | as 9 |
| 11 | ensembles, then report | ensembles: `PASS: all three representatives bit-equal ...`, `.exit` = 0. Report: no `INCOMPLETE` banner, `.exit` = 0, `b0_controls_complete: true` in `summary.json` | `FAIL: representatives not bit-equal`; `members are not today's files`; report `.exit` = 2 |

**Before anything is copied back:** search the machine directory for the area's path and the account name
(`grep -rl` over `*.json` and `*.jsonl`); nothing may match (N6). Job logs stay on the H200.

Laptop durations of this pass: 5a 40 s, 5b 3 s, step 6 or 7 about 90 s each (with the emulated guards that a
grid-1056 machine skips), the re-score of two networks 25 s, 9-pre for two configs 15 s, a 500-step fit with its
guarded evaluation 39 s.

### 5.7 Verified in this pass, and what stays open

- By execution: everything in 5.2 to 5.4 marked as executed; the step-7 command and a shortened step 9 through the
  launcher with the guard active; the failing outcomes of step 11; `p3_compare.py` on real validation events.
- By reading only: the grid-1056 branches of `val_parity.py` and of the re-score; the ensemble path with all
  members present (`check_members`, the member hashes in the subset sidecars); `lib.finish`.
- `results/registry.jsonl`, `results/test_access.log` and `common.py` have the SHA-256 of the first pass. The only
  file written in the repository is this one.
- Only the H200 can settle: the launch grid and the generator check; arithmetic parity and whether a P3 banner
  appears; the predicted table; `UnguardedStart` and `DirtyClone` with the real machine id and a clean clone;
  `setsid` and `/proc` as the launcher and the stop helper use them; whether 5b finishes inside the grace period
  there; the 10 GB cap with real jobs; `flock` on its file system; durations; steps 9-11 at full length.

---

## 6. Ruling on the step-5a failures on the H200 (25 Nov 2026)

Subject: step 5a on the H200 at `4dbd0ac` ended 5 failed, 102 passed, all five in `tests/test_provenance.py`; the
worker's diagnosis (test fixtures, not the safeguard) and its fix, commit `7fdf96c`. Reproducers:
`<scratch>/r3/`. Nothing ran on the H200; the H200 facts are those relayed by the orchestrator.

**Ruling: the diagnosis is right for all five. Step 5a may be re-run at `7fdf96c`. If it passes completely, the
batch continues with 5b and 6-8 as in section 5.6. If any test fails, it stays stopped.**

**A miss of mine first.** Section 5.5 said that step 5a is not affected off the laptop and that no test asserts it
runs on the laptop. I had looked for assertions about the laptop and not for fixtures that use the H200's own
values as "another machine". These five were there to be found by reading.

### 6.1 The diagnosis, test by test

- The ten identity fields reported from the H200 (GPU name `NVIDIA H200`, 132 SMs, 2048 threads per SM, grid 1056,
  torch 2.14.0+cu130, CUDA 13.0, cuDNN 92400, Python 3.13.12, both TF32 flags false) hash to `97b197a7`: the id
  `h200-g1056-97b197a7` is reproduced exactly on the laptop. Changing any one field gives another id. So the
  literal values in the old fixtures are the H200's live values.
- With the live identity replaced by these fields for the whole process, the pre-fix test file (taken from
  `4dbd0ac`) fails on the laptop in exactly these five tests, with the messages of the H200 log.

| Test (pre-fix) | The "other machine" | Why nothing is refused on the H200, and why that is correct |
|---|---|---|
| `test_machine_tag_fields_and_id` | `with_(launch_grid=1056)` | the live grid is 1056: the tag is unchanged, so the id is equal (`:66`) |
| `test_the_delta_function_refuses_mixed_tags` | `fake_tag(gpu_name="NVIDIA H200")` | identical tag and identical points: a legitimate pair (`:183`). The loop stopped there, before its `draw="emulated"` case |
| `test_reference_set_of_another_machine_is_not_used` | H200 name, grid, SM count, threads | the "foreign" set is the live machine's own set, so `evaluate` finds it (`:260`) |
| `test_mixed_streams_untagged_rows_and_live_paths` | H200 name and grid | the third row is on the live stream; three rows of one stream may be selected together (`:301`) |
| `test_fit_refuses_checkpoints_and_done_markers_of_another_machine` | H200 name, `deterministic=False` | the tag written into the checkpoint equals the one already there (`:349`) |

None of the production paths these tests exercise (`provenance.check_pairable`, `registry.reference_set`,
`registry.select_1se`, `train._check_machine`, `MachineTag.id`) contains a grid or a GPU name as a constant. I see no
way for any of the five to be a real defect that only shows on a grid-1056 machine.

### 6.2 Same fold points on a grid-1056 machine

Executed with the H200 identity imitated and the real laptop record `mlp_val_nll_ref1056.npy`
(`<scratch>/r3/pairing_1056.py`):

- H200-native against laptop-emulated, points tags equal and even the same values: `MixedTags`, machine differs in
  `draw`, `gpu_name`, `launch_grid` and others.
- Native against emulated on the same machine (same machine id and same stream id since B1): `MixedTags`, machine
  differs in `['draw']`.
- The bare laptop record: `MixedTags` from the delta function; `UntaggedArray` from `load_tagged`.
- The laptop's emulated arrays dropped at the live reference-set path: `NoReferenceSet`; with a hand-made
  `refset.json` beside them: `UntaggedArray` (no sidecar).

The one soft spot is N3 of section 5.4: `save_tagged` would accept an emulated array onto an existing native file
if the values are bit-identical, which is exactly the grid-1056 case. The numbers are then identical by
construction and only the label is wrong; no campaign code saves emulated arrays. It should still be fixed.

### 6.3 Commit `7fdf96c`

- **Tests only:** the commit changes `tests/test_provenance.py` and `transfer/ENABLING_STATUS.md`;
  `git diff 4dbd0ac 7fdf96c` over `src/`, `scripts/`, `configs/` and `common.py` is empty.
- **Nothing weakened.** Every scenario keeps its refusal assertions; the foreign machine is now the live tag with
  GPU name, grid and SM count changed relative to their live values. The id test checks seven single-field changes
  instead of four; the delta test gained a `deterministic=False` case. Fixed file in place on the laptop:
  `20 passed`.
- **The tests notice a broken safeguard** (each mutation applied to the fixed file, laptop):

| Safeguard switched off | Tests that then fail |
|---|---|
| `check_pairable` compares nothing | delta test, mixed-streams test, both imitation cases |
| pairing ignores the settings (`draw`, `deterministic`) | delta test, both imitation cases |
| `fit` does not compare the machine of a checkpoint or marker | fit test, both imitation cases |
| machine directories do not depend on the machine | reference-set test, both imitation cases |

- **The imitation is faithful for what these tests use.** `test_scenarios_are_machine_independent[h200]` replaces
  the live tag's identity by the ten fields above and asserts the id `h200-g1056-97b197a7`; I confirmed the hash
  independently. It does not imitate the device (the generator still draws on grid 144), which none of the five
  scenarios uses. It also sets the start-up policy aside, which is not their subject.
- No other test file contains a literal H200 value.

### 6.4 What the first run already settled, and what to read in the second

- From the reported counts (102 passed, no skips, all failures in one file): on the H200 the launch grid is 1056,
  `test_refstream.py` passes there, the emulator test among it, and the start-up policy and the guard let the suite
  run under the launcher.
- Second run of 5a: the eight files hold 109 tests at `7fdf96c` (107 before, plus the two imitation cases; counted
  by collection on the laptop). Expect `109 passed`, no `failed`, no `error`, `.exit` = 0. A skip would be new and
  should be read before going on.
- Unchanged by this ruling: sections 5.4 and 5.6.

`results/registry.jsonl`, `results/test_access.log` and `common.py` have the SHA-256 of the first pass.
