# Enabling code, batch 1 (E1, E2, E3, E5, E10, b0h part of E6) - status

25 Nov 2026. Worker status file; PLAN.md is not edited. Nothing was trained, nothing was registered in the campaign
results, the test split was never opened, `common.py` is unchanged, nothing under `shape-form-proton/` or
`shape-form-original/` was touched, `results/registry.jsonl` and `results/test_access.log` are byte-identical
(SHA-256 checked before and after every run). No git command that changes state was used.

**Design changes forced by the orchestrator's later messages (adopted, they supersede the first brief):**
(1) pristine clone on the H200: every output goes under an *output root* (`SFM_OUT_ROOT`), never into the clone;
(2) shared GPU: per-job memory and thread caps, refuse instead of wait; (3) detached launcher with PID files;
(4) cache redirection; (5) no host names / addresses / local paths in any written file; (6) a file-access guard
(`SFM_GUARD_AREA`). Consequence for E1: the "tag-named locations under results/" are now **one directory per machine**,
`results/machines/<machine id>/`, whose layout under the output root is identical to the place where it lands in the
laptop repository after the copy-back (details below). The first design (per-run `scores/` folders next to the
laptop files inside the clone) was replaced before anything was run with it.

## 1. What exists now, per item

### E10 - start-up checks (`src/sfm/machine.py`, `src/sfm/guard.py`, `src/sfm/paths.py`)
`machine.startup(what)` is called by every campaign entry point I own (`val_parity.py`, `rescore_registered.py`,
`run_m2b_{fit,stage1,ensembles,report}.py` via `run_m2b_lib.start`, `fit.py`, `fit_queue.py`), by
`registry.evaluate`, `registry.create_reference_set` and (without the caps/clean-clone part) by `train.fit`. It does:
- hard failure (`NoCUDA`) when CUDA is unavailable or `common.DEVICE` is not CUDA (common.py would silently use the
  CPU). Only `train.fit` accepts the explicit opt-in `SFM_ALLOW_CPU=1` (used by the CPU unit tests of test_train);
- `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False` (also at the import
  of `sfm.machine`, so every tag in a campaign process reads them off); both flags are part of the machine tag;
- caps: `torch.cuda.set_per_process_memory_fraction` from `SFM_GPU_MEM_GB` (default 10 GB on a device > 24 GB, no cap
  on the laptop, "0" = no cap); threads from `SFM_CPU_THREADS` (default 4 on a device > 24 GB; `torch.set_num_threads`
  and OMP/MKL/OpenBLAS/NumExpr variables);
- after the GPU lock is taken (`run_m2b_lib.run_on_gpu`): on a capped device, refuse (`InsufficientGPUMemory`) when
  less than cap + 4 GB is free, no waiting. Uncapped (laptop): the old release-and-retry rule is kept;
- pristine mode (output root outside the clone): refuse to start unless `git status --porcelain` is empty
  (`DirtyClone`); `git rev-parse HEAD` and the clean flag are recorded by every batch (report, done.json, rows).
  `SFM_ALLOW_DIRTY_CLONE=1` lifts the refusal for laptop dry runs only (recorded); it must never be set on the H200;
- the file-access guard (below).

### File-access guard (`src/sfm/guard.py`; orchestrator's second message)
Active only when `SFM_GUARD_AREA` is set (the launcher sets it by default to the directory that contains the clone;
unset on the laptop = inert). Installed at the import of `sfm.paths` (before `common`/pandas are imported) and again
by `startup`. A Python audit hook refuses (PermissionError / ImportError + one line in `<out root>/guard/guard.log`:
UTC time, PID, script, path with the area written as `<area>`, reason):
1. any `open`, `os.listdir`, `os.scandir`, `shutil.copyfile/copytree`, `ctypes.dlopen` of a path that resolves
   (symlinks followed) under the area but outside the clone, the output root and `SFM_USER_CACHE`; path-like
   arguments of `subprocess.Popen`, `os.exec`, `os.posix_spawn`, `os.spawn`;
2. wherever they are: the eight deferred file names and `*_hits_with_model_outputs.parquet`. The deferred names (not
   the foreign tables) can be released only by `guard.enable_final_test_access()`, which requires
   `data.check_test_access_allowed()` (LOCK.json naming this machine, no overrides). Nothing calls it; off by default;
3. imports of `pyarrow` (all submodules), `fastparquet`, `polars`, `duckdb` (ImportError; pandas 3.0.5 treats pyarrow
   as optional and works without it - checked). `sfm.data` imports pyarrow only inside the cache-build functions
   (checked by a test: importing `sfm.data/registry/legacy` loads no `pyarrow.parquet`). **Note:** importing pandas
   imports `pyarrow` core eagerly, so the guard must be installed before pandas: true for every batch command
   (they import `sfm` first); in pytest, test modules that import `common` first load pyarrow core before the guard,
   but `pyarrow.parquet`/`pyarrow.dataset` (needed to read Parquet) are still refused. The identity record lists the
   Parquet-reader modules loaded before the guard.
**Limits (honest):** C/C++ code opening files directly (an already-imported reader, CUDA/cuDNN, `fopen` in an
extension) raises no audit event; a subprocess runs without the hook (only its command line is checked); `stat` /
`exists` / `readlink` raise no event (metadata of a refused path is visible, contents are not); hooks are per process.

### E1 - safeguard (`src/sfm/provenance.py`, `src/sfm/registry.py`, `src/sfm/train.py`, `src/sfm/data.py`)
Spec points (transfer/REVIEW_STATUS.md section 4):
1. **Tags.** Machine tag (`machine.machine_tag()`): GPU name, SM count, threads/SM, launch grid
   (`refstream.local_grid()`), torch, CUDA build, cuDNN, Python version, both TF32 flags, deterministic flag,
   native/emulated draw, CUBLAS_WORKSPACE_CONFIG; id = readable prefix + 8 hex of the SHA-256 of all fields
   (laptop: `rtx-4060-laptop-gpu-g144-629104b0`). Points tag: grid, seed, K, batch, fold, split + fingerprint,
   subset (and chunk / bulk spec for the guard arrays). Every per-event array is written with a sidecar
   `<name>.tag.json` (points, machine, array SHA-256) by `provenance.save_tagged`; rows carry the full stream tag.
2. **Tag-named locations, no fallback.** `R = $SFM_OUT_ROOT/shape-form-martin/results` (laptop default: the repo's
   `results/`): `R/machines/<id>/{identity.json, registry.jsonl, refsets/<points id>/, runs/<stage>/<name>/
   [scores/<points id>/], checks/<label>/}`. `registry.evaluate` raises `NoReferenceSet` if the reference set of the
   live tag does not exist; it never reads `results/m0_parity/*_val_nll.npy` or `results/registry_refs/*.npy`.
3. **One delta function:** `provenance.paired_delta` (+ `paired_difference`, `check_pairable`) refuses arrays whose
   points or machine tags differ (and bare arrays). Used by `registry.evaluate` (incl. guards), `select_1se`
   (all candidates checked before the 1-SE rule), `run_m2b_ensembles.py`, `run_m2b_report.py`, `val_parity.py`.
   Cross-machine comparisons are `provenance.tolerance_check` (no SE, never a delta). Rows record the SHA-256 of the
   benchmark arrays they were paired with (+ refset path and the SHA-256 of `refset.json`).
4. **Rows:** `schema 2`, `trained_on` (machine tag from `model.pt`, or the laptop label for re-scored records, or the
   members for ensembles), `scored_on` (machine id + host id), `stream`. `latest()` is keyed by
   (stage, name, stream id); untagged laptop rows get stream `None` and `row_array` refuses them (`UntaggedRow`).
5. **`fit`:** machine tag in `checkpoint.pt`, `model.pt`, `fit_done.json` (+ `weights_sha256`); a checkpoint or a
   done-marker of another machine, or without a tag (laptop), is refused (`MachineMismatch`): no resume and no
   "finished run" returned across machines.
6. **Test access (stricter):** `load_sets(..., allow_test=True)` refuses unless `results/final/LOCK.json` exists and
   names this machine (`machine_of_record.machine_id`, optional `host_id`), and refuses when `SFM_RESULTS_DIR` or
   `SFM_CACHE_DIR` is set; all checks run before anything is logged or opened. The log line now has host **id**
   (hash; host names are never written: secrets rule), GPU, launch grid, machine id. Not written: final_test.py,
   LOCK.json. **Open point:** the user asked for "host" in the log line and also "never write a host name": I used
   the host id; and the log file lives in the clone (`results/test_access.log`), which conflicts with the
   pristine-clone rule - to be resolved with E8 (out of scope here, as instructed).
7. **Laptop records:** not moved, not copied: they stay where they are and are read-only inputs
   (`registry.LEGACY_REGISTRY`, the clone's `results/m2b_mlp/*/model.pt`, quarantine via `provenance.laptop_record`,
   which verifies each record's SHA-256 against the tracked `transfer/SHA256SUMS`). `registry._append` refuses to
   append to the legacy registry. **Deviation:** no archive copy (`results/archive/rtx4060_g144/...`) was made: the
   new design never writes at a laptop path, so nothing can overwrite them; an archive copy would only duplicate
   files. If the orchestrator still wants it, it is a copy-and-hash job (not implemented).
8. **Re-scoring as its own step:** `scripts/rescore_registered.py` (E3).
9. **Reference arrays created once** by `registry.create_reference_set` (called by `val_parity.py --create-refset`):
   computed, written to a temp dir, renamed into place (fails if another process created it), never inside
   `evaluate`; a second invocation recomputes and compares bit for bit.
10. Points vs arithmetic: both in every tag; `check_pairable` requires both equal.
11. `registered_row` (registry-level and `run_m2b_lib`) matches stage, name, **stream id** and checkpoint SHA-256;
    the guard spec in `refset.json` contains the machine id and points id. **Not done:** `scripts/run_m2a_lib.py`
    (not in my write set) still has its own `registered_row` by SHA-256 only and reads `registry.read_registry()`
    (now: this machine's registry); it must switch to `registry.registered_row(..., stream_id=...)` with E4.
12. **Read-only reference set + tracked hashes:** files and directory chmod a-w; `refset.json` (tracked once copied
    back) holds every array's SHA-256, file SHA-256, mean, points, machine tag, git state.
13. **Weight hash:** `weights_sha256` over the state-dict tensors in every row and fit_done.json.
15. **Record-writing scripts blocked:** `mc_noise_val.py`, `build_cache.py`, `check_protected.py record`,
    `parity.py` exit unless `SFM_ALLOW_RECORD_WRITER=1`, and always when `SFM_OUT_ROOT` is set. `run_m2b_report.py`
    writes `summary.json` in the machine directory.
14. covered under 6.

### E2 - `scripts/val_parity.py` (run-book steps 6-7; R0, R1, R1b (benchmarks), R2, R3)
Identity record + generator check (R0); both benchmarks with unmodified `common.test_nll` inside
`metric.deterministic()`, twice (bit-identity); tolerance checks against the laptop's emulated arrays
(`*_val_nll_ref1056.npy`, quarantine or original path, SHA-256 verified) and the stored means; emulated grid-1056 and
grid-144 recomputation; R3 floor outside deterministic(); with `--create-refset` the reference set (K = 64 arrays of
run 1, K = 1024 and bulk guard references) and the R2 targets. **P3:** any event > 5e-5 against the laptop's emulated
arrays prints a banner and writes `checks/val_parity/outliers_<bench>_<utc>.json/.npz`: event index and number, both
values, batch, Philox offset of the batch, H of the batch, the event's hit rows, the fold points of every hit
(computed with common.fold's expression on the GPU), grid cells, distance of z'/r to the nearest bin edge in float32
ULPs, and how to re-derive them with `EmulGen` on the laptop. **Fallback (grid != 1056):** the targets are evaluated
on the emulated grid-1056 arrays (K = 64 via `refstream.test_nll_ref`; guards via the new
`refstream.patched_draw`, which emulates `sfm.metric.draw`), and native vs emulated is checked on A3. Missing in the
fallback: R4 on the emulated grid (the eight networks on emulated 1056 points) is not automated (`rescore` scores
native only; easy to add with `test_nll_ref` if needed).

### E3 - `scripts/rescore_registered.py` (step 8, R4)
Reads the laptop rows from the clone's `results/registry.jsonl` (read-only), takes `model.pt` from the clone, refuses
a network whose SHA-256 differs from its row, scores with guards exactly where the laptop row had them (b0_seed0,
s2_seed11, s2_seed12), appends tagged rows to this machine's registry (`trained_on` = laptop label + the laptop row's
hashes), idempotent. Checks: grid 1056 - predicted table of PLAN 16.3 and guard table D; grid 144 - bit-identity with
the laptop's registered arrays (+ guards); any grid - mass monitors within 1e-4 relative per event;
`--emulate-laptop` - R1b for the eight networks.

### E5 - `sfm.refstream` / `tests/test_refstream.py`
`LOCAL_GRID` removed; `LAPTOP_GRID = 144` names the laptop records' grid; tests read `local_grid()` and compare with a
grid different from the local one; the Gate 0 record is looked up in the quarantine first. New test for
`patched_draw`. The suite no longer assumes grid 144 (designed to pass on 1056; **not run on a 1056 GPU**).

### E6 (part) - controls
`configs/m2b/b0h_seed{0,1,2}.toml`: identical recipe to `b0_seed*.toml` (diff apart from name/notes/comments empty;
the name changes the config hash, as the review noted). `run_m2b_report.py` takes the control from `b0h_*` only;
`b0_seed0` is shown as "b0_seed0 (laptop-trained record)" and enters no scatter/mean; the paired difference
b0_seed0 - b0h_seed0 is reported without pass/fail. `run_m2b_fit.py` refuses the eight laptop-trained names
(their laptop done.json is in the clone) and points to the re-score; `run_m2b_stage1.py` never retrains them.
`run_m2b_ensembles.py`: members s2_seed11/12 from the clone (`repo:` reference), s2_seed13..15 must come from this
machine (`out:` reference); subsets saved tagged under `ensemble_subsets/<points id>/`; only live-tag arrays read.

### Launcher - `scripts/launch.sh`
`SFM_OUT_ROOT=<out> SFM_USER_CACHE=<cache> shape-form-martin/scripts/launch.sh <job> <command...>`: exports
PIP_CACHE_DIR, XDG_CACHE_HOME, MPLCONFIGDIR, CUDA_CACHE_PATH, TORCH_HOME, TMPDIR (subdirectories of
`SFM_USER_CACHE`, created), thread variables (`SFM_CPU_THREADS`, default 4), `SFM_GUARD_AREA` (default: the directory
containing the clone), starts the command with nohup, writes `<out>/jobs/<job>_<UTC>.pid` and `.log`, prints both.
**Stop a job only by its PID file:** `kill "$(cat <out>/jobs/<job>_<UTC>.pid)"` (never pkill/killall).
`sfm.refstream` builds its `.so` only in `shape-form-martin/cache/` (git-ignored): `tempfile.mkstemp(dir=cache)` +
rename; gcc's own temporary files follow `TMPDIR` (gcc documents `TMPDIR` for its temporaries; not traced here).

## 2. H200 command lines, run-book steps 5-11 (in order; one step at a time)

```
cd <repo>
export SFM_OUT_ROOT=<area>/sfm_out          # outside the clone
export SFM_USER_CACHE=<area>/sfm_cache
# optional: export SFM_GPU_MEM_GB=10 SFM_CPU_THREADS=4      (defaults on the H200)
L=shape-form-martin/scripts/launch.sh
P=.venv/bin/python
```
Each `$L` prints a PID file and a log; wait until the job has ended (`kill -0 "$(cat <pid file>)"` fails) and read
the last lines of the log before the next step. Do not set `SFM_RESULTS_DIR`, `SFM_CACHE_DIR`, `SFM_DATA_DIR`,
`SFM_ALLOW_DIRTY_CLONE`, `SFM_ALLOW_RECORD_WRITER`, `SFM_ALLOW_CPU`.

| step | command | expected / acceptance |
|---|---|---|
| 5 (R0, suite) | `$L tests $P -B -m pytest shape-form-martin/tests -q -p no:cacheprovider --ignore=shape-form-martin/tests/test_geometry.py` | all pass; `test_refstream.py` passes on grid 1056 (no expected failures any more); `test_emulation_reproduces_torch_rand_bitwise` must pass. `test_geometry.py` (other worker; laptop-only geometry audit) writes and reads synthetic Parquet files through pyarrow, which the guard refuses on the H200, hence `--ignore`. Tests needing the Parquet table or the test cache skip (files absent). |
| 6-7 (R0, R1, R1b, R2, R3) | `$L val_parity $P -B shape-form-martin/scripts/val_parity.py --create-refset` | log ends `ALL APPLICABLE CHECKS: PASS`; grid 1056; float32 means 18.118083953857422 / 18.120407104492188 (4e-6); every event within 1e-4 of the laptop's emulated arrays (no P3 banner; if a banner appears: STOP, send `checks/val_parity/outliers_*` for Fable's 1-ULP review); emulated == native bit for bit; formula - MLP +23.24 +- 2.94; second run bit-identical; R3 within 1e-4 / 1e-6; reference set "created"; K = 1024 means 18.1155556 / 18.1177364, bulk 18.0429497 / 18.0448778 (1e-6), 14,784 events. Must run before anything else is registered. |
| 8 (R4) | `$L rescore $P -B shape-form-martin/scripts/rescore_registered.py --emulate-laptop` | `ALL APPLICABLE CHECKS: PASS`; 8 checkpoints match the laptop rows; means / deltas = PLAN 16.3 table (1e-6 / 0.01e-4); guard means = review table D; mass monitors within 1e-4 rel.; R1b within 1e-4 (relaxed clause reported) |
| 9 | `$L s2_ft $P -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/s2_seed13.toml shape-form-martin/configs/m2b/s2_seed14.toml shape-form-martin/configs/m2b/s2_seed15.toml` | three rows in this machine's registry; fits start from step 0 (no laptop checkpoint exists or is accepted) |
| 10 | `$L b0h $P -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/b0h_seed0.toml shape-form-martin/configs/m2b/b0h_seed1.toml shape-form-martin/configs/m2b/b0h_seed2.toml` | three rows; none with checkpoint SHA-256 `75de31f2...` |
| 11 | `$L ensembles $P -B shape-form-martin/scripts/run_m2b_ensembles.py` then `$L report $P -B shape-form-martin/scripts/run_m2b_report.py` | six ensemble rows, representatives bit-equal (`representative_bitwise_equal` all true), `summary.json` with B0 = b0h_seed0..2 |

Laptop time estimates as in PLAN 16.3 (steps 6-7 about 1 min, step 8 about 1.5 min, 9 and 10 about 13 min each).

## 3. What each step writes, and what to copy back

Everything is under `<area>/sfm_out/` (nothing in the clone except the git-ignored `shape-form-martin/cache/`
files that already exist or are regenerated there: `philox_emul_<sha>.so`, `pion_s1_val_fingerprint.json`,
`gpu.lock`). Copy back **the machine directory** `<area>/sfm_out/shape-form-martin/results/machines/<machine id>/`
to the laptop at `shape-form-martin/results/machines/<machine id>/` (same relative path), e.g. `tar -C
<area>/sfm_out -cf machine.tar shape-form-martin/results/machines/<machine id>` and unpack at the laptop repo root.

| written by | files (relative to the machine directory) | git on the laptop |
|---|---|---|
| step 6-7 | `identity.json`, `checks/val_parity/report_*.json`, `outliers_*.json` (if any) | tracked |
| step 6-7 | `refsets/<points id>/refset.json`, `*.tag.json` | tracked |
| step 6-7 | `refsets/<points id>/*.npy`, `outliers_*.npz` | ignored (copy by hand) |
| steps 6-11 | `registry.jsonl` (this machine's rows) | tracked |
| step 8 | `runs/m2b_mlp/<8 names>/scores/<points id>/{val_nll,val_nll_k1024,val_bulk_zb1221}.{npy,tag.json}`, `val_mass_monitors.npz`, `checks/rescore/report_*.json` | .tag.json/.json tracked; .npy/.npz ignored |
| steps 9-10 | `runs/m2b_mlp/<run>/{done.json, fit_done.json, resources.json}` | tracked |
| steps 9-10 | `runs/m2b_mlp/<run>/{model.pt, checkpoint.pt}`, `fit.log`, scores arrays | ignored (copy model.pt and arrays; checkpoint.pt not needed) |
| step 11 | `runs/m2b_mlp/E*/{done.json, model.pt}`, `ensemble_subsets/<points id>/*.{npy,tag.json}`, `timing.json`, `ensemble_scaling.json`, `summary.json` | JSON tracked; .npy/.pt ignored |
| launcher | `<area>/sfm_out/jobs/*.{pid,log}`, `<area>/sfm_out/guard/guard.log` | not under results/: keep on the H200 or copy separately; not for commit |

No written file contains a host name, address or local directory (host id = hash; paths as `<repo>/...`,
`<out>/...`; JSON writers scrub absolute roots).

## 4. Tests and dry runs on the laptop

- `uv run pytest` on my files (test_provenance, test_registry, test_refstream, test_data, test_train, test_metric,
  test_models: 105 passed in 32 s; test_guard: 3 passed), run while a wrapper process held the GPU lock.
  **The whole suite (`pytest shape-form-martin/tests`, all workers' tests) was started the same way but had not
  finished after about 14 minutes; I stopped it to release the GPU lock** (another worker's job was waiting).
  Probable cause: a test of another worker takes the real `gpu_lock` (it would wait for my wrapper) or the SR tests
  are slow; not investigated. So: my files are green; the full suite with the other workers' files was not
  observed to completion in this session.
- **E2 dry run** (`SFM_OUT_ROOT=<scratch>`, pristine mode, `SFM_ALLOW_DIRTY_CLONE=1`): exit 0, ALL PASS. Grid 144, so
  the fallback path ran: emulated grid-1056 arrays bit-identical to the laptop's `*_val_nll_ref1056.npy`
  (float32 means exactly 18.118083953857422 / 18.120407104492188, formula - MLP +23.24 +- 2.94); laptop native vs
  `m0_parity/*_val_nll.npy` within A2 (max 2.67e-5, mean 1.9e-8; those records were made outside deterministic());
  second run bit-identical; R3 max 2.3e-5; A3 r = 0.99975 / 0.99971, rms 0.0332 / 0.0358; reference set created and
  sealed; guard means equal the review's laptop-stream values (|diff| < 5e-8) and the emulated-1056 guards meet the
  H200 targets.
- **E3 dry run** (same scratch output root, **file-access guard active** with the area = the directory containing the
  clone, `SFM_USER_CACHE`/`TMPDIR` in the scratchpad): exit 0, ALL PASS. All eight checkpoints match their laptop
  rows; all eight re-scored K = 64 arrays are **bit-identical** to the laptop's registered `val_nll.npy` (array
  SHA-256 equal to each row's `val_nll_sha256`); K = 1024 and bulk arrays of the three guarded networks bit-identical
  to the laptop's; mass monitors within 1e-4 relative; R1b (emulated grid 144) within the strict clause for all eight.
  Means and deltas reproduce STATUS.md (e.g. b0_seed0 18.1179757, +1.19 +- 0.42e-4).
- **E2 second run** with the guard active: reference set "exists_identical" (bit-for-bit recomputation, nothing
  overwritten), ALL PASS.
- **Refusals on real outputs** (scratch): a delta between a re-scored array and the same values tagged as an H200
  array -> `MixedTags`; the laptop's untagged row -> `UntaggedRow`; a laptop `val_nll.npy` copied over a live tagged
  array -> `UntaggedArray` (SHA-256 differs from the sidecar); `registered_row` on the live stream never returns a
  laptop row; `run_m2b_fit.py configs/m2b/b0_seed0.toml` -> refused as a laptop record; `run_m2b_report.py` ran on the
  scratch rows (control column empty: no b0h rows) and wrote `summary.json` in the machine directory.
- **"H200 layout" check:** instead of hiding the eight deferred files with the earlier audit-hook harness, the runs
  of E3, E2 (second run), the fit refusal and the report were made with the new guard active, which refuses any
  Python open of the eight deferred names wherever they are (and logs it). The guard log of all these runs holds only
  the refused `pyarrow` imports of pandas (tolerated: pandas works without pyarrow); **no open of a deferred file,
  of a `*_hits_with_model_outputs.parquet`, or of any path in the area outside the clone / output root / cache was
  attempted**. Limits as in section 1 (C-level opens are not seen). The first E2 run was made without the guard.
- `results/registry.jsonl` and `results/test_access.log`: SHA-256 unchanged after every run; nothing was written
  inside the clone (pristine mode with the scratch output root; the only clone-side files are the pre-existing
  git-ignored cache files).
- Mixed-tag refusal and "laptop array at a live path not picked up": `tests/test_provenance.py`
  (`test_the_delta_function_refuses_mixed_tags`, `test_mixed_streams_untagged_rows_and_live_paths`,
  `test_no_reference_set_no_fallback`, `test_reference_set_of_another_machine_is_not_used`) and the real-data
  demonstration above.

## 5. Not tested / could not be tested on the laptop; open questions

- Nothing was run on an H200: grid 1056 code paths (native targets, P3 outlier path with real outliers, caps with a
  10 GB fraction, `InsufficientGPUMemory`, `DirtyClone` with a real clean clone) are exercised only by unit tests or
  not at all. The P3 outlier writer ran only in a unit-free way (no event > 5e-5 on the laptop).
- Steps 9-11 were not run (they train); `run_m2b_fit/ensembles/report` were import-checked only.
- `run_m2a.py` / `run_m2a_lib.py` (not in my write set) still use the old assumptions (E4); with the new registry they
  will raise `NoReferenceSet` until switched; their `registered_row` must take the stream id.
- `scripts/make_transfer_package.py list` re-checks file:line citations of the 29 machine constants; lines in
  registry.py, refstream.py, test_refstream.py, run_m2b_lib.py, data.py, train.py moved, so `list` will report them.
- `results/test_access.log` in the clone vs the pristine-clone rule (E8 decision).
- `fit.py`/`fit_queue.py` now write to the machine directory too (they are not part of steps 6-11).
- Whether `SFM_GUARD_AREA` = parent of the clone is the right area on the H200 (the user said so; the launcher's
  default), and whether the venv lives inside the clone (`<repo>/.venv`, as in PLAN 16.9; a venv elsewhere in the area
  would be refused by the guard).

## 6. Files of this batch (for explicit staging by the orchestrator)

New: `src/sfm/machine.py`, `src/sfm/provenance.py`, `src/sfm/guard.py`, `scripts/val_parity.py`,
`scripts/rescore_registered.py`, `scripts/launch.sh` (executable), `configs/m2b/b0h_seed0.toml`, `b0h_seed1.toml`,
`b0h_seed2.toml`, `tests/test_provenance.py`, `tests/test_guard.py`, `transfer/ENABLING_STATUS.md`.
Modified: `src/sfm/{registry,train,data,paths,refstream,gpulock}.py`, `src/sfm/models/__init__.py`,
`scripts/run_m2b_{lib,fit,stage1,ensembles,report}.py`, `scripts/{fit,fit_queue,mc_noise_val,build_cache,
check_protected,parity}.py`, `tests/test_{registry,refstream,train}.py`.
Not mine (other workers): `src/sfm/sr/`, `src/sfm/geometry.py`, `tests/test_sr_*`, `tests/test_geometry.py`,
`results/m2c_theta/`. Nothing was written under `results/` of the repository.
