# H200 transfer package: status (packaging DONE; section 16 not written)

Resumed and finished on 25 Nov 2026 (second packaging worker). Section 16 of PLAN.md, the commit and the push are the
orchestrator's. The statistics review is in `REVIEW_STATUS.md` (not touched here).

**Produced (all untracked until the orchestrator commits them, explicit paths only):**
- `shape-form-martin/scripts/make_transfer_package.py`: `list` (laptop) and `verify` (H200), standard library only,
  never imports torch or numpy, refuses any `*.csv`, treats the test cache as an opaque byte stream.
- `shape-form-martin/tests/test_transfer_package.py`: 9 tests on synthetic temp repositories, all pass.
- `shape-form-martin/transfer/{files_b_inputs.txt, files_b_laptop_records.txt, SHA256SUMS, MANIFEST.json}`, written
  by `list` at commit `c9b7f72`. The only tracked change since that commit is this STATUS.md, so the manifest
  records `tracked_tree_dirty: true`. `MANIFEST.json` holds every field of every file; this file only summarises it.

**Guardrails kept.** No training, fit or registry write. The test cache was opened only by the byte hash in
`sha256_file` (also shown by the test's audit hook). No CSV was listed, stat'ed or hashed. Nothing under
`shape-form-proton/` or `shape-form-original/` was modified (`find -newer` is empty there). `parity.py` and
`build_cache.py` were not run. `cache/gpu.lock` changed at 10:29:18 because of the reviewer's concurrent job
(`scratchpad/review2/`); this work opened it read-only (strace).

## 1. Trace re-check (part 1 i)

Harness rebuilt (the scratchpad had been cleared): audit hook installed before any import, recording `open`,
`listdir`, `scandir`, `glob`, `dlopen` and `Popen` under the repo. It **raises before the operation** on any open of
the test cache or of a `*.csv`, on any write-mode open, and on any `mkdir`/`rename`/`replace`/`remove` under the repo.
It ran on the CPU (`CUDA_VISIBLE_DEVICES=""`) under `strace -f -e trace=%file`.

Exercised: imports of all `sfm.*` modules; `data.load_sets(("train","val"), device="cpu")` and `load_meta`; the
legacy loaders (`load_mlp_std`, `load_component_formula`, `load_abar`, `families.load_legacy_core`); the registry's
read paths without `evaluate` (`validation_reference`, `check_validation`, `read_registry`, `select_1se`, the benchmark
`np.load`, and `_bench_guard_arrays` with a `bench_models` that raises); all five acceptance templates; the emulator
`.so` and `gpulock.holder`; the Track A spec lists and builds of A0, A1, A2e, A2s, A2n, A3f, A3x, A9 and A10
(`skip_init=True`); Track B builds (`field_mlp_std`, `from_run` s1_seed13, ensembles, `field_acc` with front, stored
and full_box, `residual_table`, `field`); `load_run` of the 8 networks with SHA-256 against the registry;
`_member_shas`; `fit.py --dry-run` (s2_seed13, b0_seed1) and `fit_queue.py configs/m2b --dry-run`.

Result: **the hand-over's reads are reproduced.** No write under the repo (strace: no `O_WRONLY`/`O_CREAT` except
uv's own `.venv/.lock`). No access to the test cache or to any CSV.

Differences from the hand-over and precisions:
- With `skip_init=True`, A3f (`start=A2s_s0`) and A10 (`init_ref=A2s_s0`) build without reading the start run. Only
  `provider="net"` with `net_from` (A2n) reads `results/m2a_dist/A2s_s0/model.pt` at build time. A9 needs
  `templates/train_theta.npy`; A3x and B4x need `full_box` (`field_acc` rejects it with a `ValueError`).
- The Track A spec list has 66 entries (12 stage 0, 12 poly primaries, 6 net primaries, 36 exploratory); the
  hand-over's 75 probably counted pilots or reruns. This makes no difference to the files read.
- Branch: `HEAD` = `origin/shape-form-martin` = `c9b7f72` (the hand-over said `ccd9a6f`).
- New hazard, found statically: `fit_done.json` makes `train.fit` return the stored laptop network (§3).
- New: `tests/test_registry.py` real-data test FAILS, rather than skips, without `results/m0_parity/val_event_number.npy`.
- Reviewer's open question: `results/m0_parity/cache_build.json` does **not** record the cache hashes. They are in
  the ignored `cache/pion_s1_meta.json` and now also in the tracked `MANIFEST.json`.

Settled statically (reads that only happen inside fits, evaluations or resumes; not run):
- `train.fit` reads `checkpoint.pt`, `fit_done.json` and `model.pt`, and appends to `fit.log`, only in the run's own
  directory.
- `registry.evaluate`:
  - `_save_array` compares with an existing `val_nll.npy` at the live path: `FileExistsError` unless `replace=True`;
  - `_append` opens `registry.jsonl` with `a+b`;
  - the guards read `guard_spec.json`, the 4 reference arrays and `abar.pt`; `_default_bench_models` reads the
    train split, `mlp_std.pt`, `new3_std_n3.pt`, `test_v3.json` and `abar.pt`;
  - `_git_state` runs `git`.
- `run_spec` writes and then reads `init.json` and appends to `gpu_time.jsonl`.
- Ensembles read `ensemble_subsets/timing.json`.
- The `run_m2a.py templates` phase reads `abar.pt`, `abar_front.pt` and the train split.

All of these are inside groups b1 or c.

## 2. Groups (exact; from `MANIFEST.json`)

| group | files | bytes | git status (checked per file) |
|---|---|---|---|
| a | 127: 121 tracked + 6 package files to be committed | about 2.3 MB (exact snapshot in `MANIFEST.json`; not verified by `verify`) | `git ls-files --error-unmatch` OK for all 121 |
| b1 inputs | 23 | 3,063,083,399 | all ignored; rule recorded per file; none tracked |
| b2 laptop records | 55 | 5,989,510 | all ignored; none tracked |
| c | 17 entries | n/a | regenerated on the H200 |

No file was found that is untracked and not ignored.

Ignore rules that apply:
- `.gitignore:232 *.parquet`;
- `.gitignore:222/223/227 shape-form-proton/**/*.{pt,npz,json}`;
- `shape-form-martin/.gitignore:2 cache/`;
- `shape-form-martin/.gitignore:7/8/9 results/**/*.{pt,npy,npz}`.

Cross-checks made by `list`; all equal:
- both caches: SHA-256 and size equal `pion_s1_meta.json`;
- the 7 PION files: equal the tracked `checksums.json`;
- the Parquet file: size and mtime equal `checksums.json` `stat_only`;
- the 8 checkpoints: equal the registry's `checkpoint_sha256`;
- the 8 registered `val_nll.npy`: equal the registry's `val_nll_sha256` (array hash, recomputed without numpy).

**b1** (copy to the same path): the Parquet file (2,604,833,654 B); `cache/pion_s1_{trainval,test}.pt` and
`pion_s1_meta.json`; `PION/{mlp_std.pt, new3_std_n3.pt, abar.pt, test_v3.json, mlp_std.json, new3_std.json,
test_v3_nll.npz}`; `results/m0_parity/{val_event_number.npy, test_event_number.npy, abar_front.pt,
abar_front_back.pt}`; the 8 `results/m2b_mlp/<run>/model.pt`.

**b2** (copy only under `shape-form-martin/cache/laptop_records/<same path>`):
- `m0_parity/{mlp,formula}_{val,test}_nll{,_ref1056}.npy` (8);
- `registry_refs/*.npy` (4);
- `m2b_mlp/<8 runs>/val_nll.npy`;
- `{b0_seed0,s2_seed11,s2_seed12}/{val_nll_k1024.npy, val_bulk_zb1221.npy, val_mass_monitors.npz}`;
- `gates/gate0_checks/val_grid{1056,144}_*.npy` (26).

All are stream-dependent except the 3 `val_mass_monitors.npz`. Streams: native grid 144 for the m0_parity native
arrays, registry_refs and m2b arrays; emulated grid 1056 for `*_ref1056` and `val_grid1056_*`; emulated grid 144 for
`val_grid144_*`.

## 3. Copy or rebuild: COPY both caches (confirmed)

- `build_cache.py` has no argument parser, so any invocation rebuilds.
- `data.build_cache` always writes both `.pt` files (`data.py:226-227`) and the meta file (`:246`), and the script
  then rewrites the tracked `cache_build.json` (`build_cache.py:64-66`).
- The meta file is never byte-identical.
- A rebuild's tensors are not guaranteed bit-identical on another CPU: float64 `np.cos`/`np.sin` at `data.py:135-136`
  and `common.py:144`, plus the torch CPU log for lnp.
- The code would not notice a different rebuild: `validation_reference` re-keys the fingerprint by the new trainval
  SHA-256, and `_bench_guard_arrays` recomputes silently.
- Only a byte copy carries Gate 0 and C1 over.

Optional cross-check into scratch: `SFM_CACHE_DIR=<s>/c SFM_RESULTS_DIR=<s>/r uv run python -B
shape-form-martin/scripts/build_cache.py`.

## 4. Tracked laptop records (group a hazards; they arrive with `git pull`)

`registry.jsonl`:
- `registered_row` matches by checkpoint SHA-256, so the run scripts reuse the laptop rows.
- `latest()` makes tables, ensembles and the 1-SE rule read the laptop rows' arrays. That gives `FileNotFoundError`
  while the records are quarantined (correct), and a silent mix if they were copied to the live path.

`test_access.log`:
- The H200 copy starts with the two laptop lines. Its lines hold no host.
- Gate 2c's count must therefore be per machine.

`guard_spec.json`: the laptop guard arrays would be reused silently if they were at the live path.

`m2b_mlp/*/done.json` (8): the run scripts skip these runs. Re-scoring needs a new script.

`fit_done.json` (3): `train.fit` reloads the laptop network instead of training. Retraining `b0_seed0` (user decision
2) needs a new run name.

Others:
- `summary.json` keeps laptop numbers until `run_m2b_report.py` rewrites it.
- `parity.json` and `c1_reference_stream.json` make `parity.py` refuse to run without `--overwrite` (keep that).
- `cache_build.json` is rewritten by any `build_cache.py` call.
- `checksums.json`: `check_protected.py` reports every protected file that was not copied (exit 1).
- `resources.json` and `stage1_monitor.json` are inert.

## 5. Run-book facts (commands checked against argparse or source; laptop = RTX 4060 Laptop)

| action | existing command, or what is missing | laptop duration |
|---|---|---|
| environment | `uv sync --python 3.12`; `uv run python -B -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` (expect `2.14.0+cu130`, `13.0`, `True`); grid: `uv run python -B -c "import sys; sys.path.insert(0,'shape-form-martin/src'); from sfm import refstream; print(refstream.local_grid())"` | not measured |
| paths | leave `SFM_DATA_DIR`, `SFM_RESULTS_DIR`, `SFM_CACHE_DIR` unset (see below) | - |
| copy + verify | the two `rsync -a --files-from` commands in `MANIFEST.json`, then `python3 -B shape-form-martin/scripts/make_transfer_package.py verify` | list 2.1 s, verify 1.5 s (warm page cache, 3.06 GB) |
| validation-only parity | **missing**; partial: `uv run pytest shape-form-martin/tests/test_refstream.py -q` (mlp_std only: emulated grid 1056 against the stored 18.118084 to 4e-6, plus native = emulated local grid bit for bit; the formula is not covered) | test_nll 1.48 s / 0.90 s |
| validation pairing arrays | **missing**: validation-only `common.test_nll` inside `deterministic()`, `np.save` to `m0_parity/{mlp,formula}_val_nll.npy`, logging GPU and grid | 1.48 s + 0.90 s |
| re-score the 8 runs | **missing**: SHA-256 check, `load_run`, `registry.evaluate(guards = as originally)` | about 1.4 min (5 x 1.5 s + 3 x 16.1 s + about 27 s for the reference arrays) |
| Track B resume | (1) `uv run python -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/s2_seed13.toml shape-form-martin/configs/m2b/s2_seed14.toml shape-form-martin/configs/m2b/s2_seed15.toml shape-form-martin/configs/m2b/b0_seed1.toml shape-form-martin/configs/m2b/b0_seed2.toml`; (2) `uv run python -B shape-form-martin/scripts/run_m2b_ensembles.py [--table-only]`; (3) `uv run python -B shape-form-martin/scripts/run_m2b_report.py` | (1) 238 s fit + 16.1 s evaluation per run, about 21 min; (2) never run ("a few minutes", STATUS); (3) seconds, not measured |
| Track A phases | `uv run python -B shape-form-martin/scripts/run_m2a.py templates parity plr stage0 stage1p stage1x` (phases from {templates, parity, plr, stage0, stage1p, stage1x, guards, stage2, all}). **Blocked** on a grid-1056 GPU by `FORMULA_OF_RECORD` (`:43`, `:260`, `:266`); the drift rule needs the H200 formula array | never run; AUDIT 4.1 budget: 49 + 73 + 77 + 38 min, about 4.0 h |
| Track A report, guards, Stage 2 | **missing**: `run_m2a_report.py` (the `guards`/`stage2` phases raise ImportError) | finalist 38 min, inside the 4 h |
| remaining Track B (AUDIT 4.2) | `fit.py <toml> [--dry-run] [--no-lock] [--no-evaluate]` / `fit_queue.py [--dry-run] [--stop-on-error] configs...` work once configs are written for B2a/b/c, B5f, B3s, B4f/B4s. **Missing**: the configs; B4x, B6a/b/c, B1d/B1f and B-ceil need code | AUDIT 4.2: 2.0 h |
| θ track | **no code** | - |
| final test | **missing**: `final_test.py`, `LOCK.json` | C1 7.1 s, Gate 0 parity 52.4 s |

**Path variables.**

`SFM_DATA_DIR` affects only `parquet_path`, which is used by the rebuild and the parity raw-file check. No moved step
reads it while the caches are copied.

`SFM_CACHE_DIR`: the code would look for the caches, the meta file, the fingerprint file, the `.so` and `gpu.lock`
there, while `verify` checks the default path. The quarantine root is repo-relative, so it is not affected.

`SFM_RESULTS_DIR` breaks the most:
- the pairing arrays, `val_event_number.npy` and the templates would not be found;
- the tracked registry, done-markers and guard spec would be ignored, so finished runs would count as not done;
- worst: `test_access.log` would be written outside the tracked log.

`SFM_GPU_LOCK` relocates the lock file. Use it to share one lock between checkouts.

**Machine-specific constants.** 29 in `MANIFEST.json` `machine_constants`, each with file:line and checked by `list`
(the line must contain the quoted text). These are the hand-over's 20 plus:
- silent CPU fallback (`common.py:64`);
- the root cause, the fold stream at `common.py:313`;
- the Gate 0 record used by `test_refstream.py:113`;
- the test-access log line has no host (`data.py:258`);
- one lock per working copy, and flock on network filesystems (`gpulock.py:32`);
- the env overrides (`paths.py:26`);
- Python version not pinned (`pyproject.toml:5`; the laptop uses CPython 3.12);
- torch 2.14.0 +cu130, which needs a CUDA 13 driver (`uv.lock:2450`).

**Campaign tests expected to fail on the H200:**
- `test_refstream.py::test_local_grid_constant_matches_this_gpu` fails on any GPU except the RTX 4060 Laptop.
- `test_refstream.py::test_other_grids_differ_only_beyond_their_thread_count` fails if the grid is 1056.
- All emulator tests fail without gcc with OpenMP.
- The cost test passes or skips.
- The real-data tests need the b1 files.

The campaign suite was not run in this session (it contains tiny fits and GPU evaluations).

## 6. Script outputs

`list` (final run at commit `c9b7f72`; the only tracked change is this file): a 127 files; b1 23 files, 3,063,083,399 B; b2 55 files,
5,989,510 B; c 17 entries.

`verify --source-machine` on the laptop: 80 checks, 0 failures, 6 warnings (the package files are not committed
yet), PASS.

Target mode on the laptop (`--skip-large-hash 0`): FAIL as designed. 113 failures:
- 55 records missing from quarantine;
- 52 stream-dependent records at their live paths, each printed with its `mv`;
- 6 package files not committed.

Plus warnings for the 3 mass monitors.

Tests: `uv run pytest shape-form-martin/tests/test_transfer_package.py -q` gives 9 passed.

## 7. Not verified

**Reads that happen only inside code that was not run:**
- fits, evaluations, guard recomputation and resumes (static reading only);
- `skip_init=False` builds that need finished Track A runs;
- code that does not exist yet: re-scoring, validation-only parity, `run_m2a_report.py`, the AUDIT 4.2 drivers, the
  θ arms, `final_test.py`.

**Laptop durations** of ensembles, report and Track A: never run, so estimates only.

**Cache rebuild:** whether a rebuild on the H200 host would be bit-identical (argued, not tested).

**Anything about the H200:**
- grid 1056 (prediction);
- CUDA 13 driver;
- gcc and OpenMP;
- AVX-512 on the host CPU;
- flock on its filesystem;
- Python 3.12;
- MIG or multi-GPU layout;
- that `rsync -a` keeps mtimes;
- disk space;
- that its torch build reproduces the Philox layout;
- the reviewer's predicted re-scoring values.
