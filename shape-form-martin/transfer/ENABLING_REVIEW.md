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
