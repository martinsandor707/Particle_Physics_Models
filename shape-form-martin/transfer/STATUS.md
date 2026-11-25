# H200 transfer package: status at the checkpoint (INCOMPLETE)

The task was paused at the user's request, 25 Nov 2026.

**State.** Part 1 (the manifest derived from the code) is traced and mostly verified; the details are below.
Part 2 is **not started**:
- `scripts/make_transfer_package.py` and `tests/test_transfer_package.py` do not exist yet.
- `list` has never run, so this directory holds no `files_b.txt`, `SHA256SUMS` or `MANIFEST.json`.
- Every SHA-256 below comes from an earlier computation that is named in the table. No hashing pass for the package
  has run.

Nothing in the repository was modified. The trace wrote no file: this was checked with `find -newer` and with
strace, which shows no `O_WRONLY`, `O_RDWR` or `O_CREAT` open under the repo. No `.pyc` was written either.
The test cache was never opened. The CSV was never touched. No training, fit or registry write took place.

## 1. How the reads were traced

**Dynamic trace, in-process.** The harness is
`/tmp/claude-1000/-home-martin-projects-uni-PhD-Particle-Physics-Models/baad5034-857c-48b3-bd15-4b0326ffecf3/scratchpad/trace_reads.py`.
The scratchpad may be cleared; the approach is described here so that the harness can be rebuilt.

*Audit hook.* `sys.addaudithook` is installed before any import.
- Events recorded: `open`, `os.listdir`, `os.scandir`, `glob.glob`, `os.mkdir`, `os.rename`, `os.replace`,
  `os.remove`, `os.truncate`, `shutil.copyfile`, `subprocess.Popen`, `ctypes.dlopen`, `os.chmod`, `os.utime`,
  `os.rmdir`, `shutil.rmtree`, `os.link`, `os.symlink`.
- Fields kept: phase, path relative to the repo, mode, and the innermost repo frames (file:line), found with a
  re-entrancy guard around `sys._getframe`.
- The hook **raises before the operation** on any `open` of `shape-form-martin/cache/pion_s1_test.pt` or of any
  `*.csv`.

*How it was run.* From the repo root, with the GPU hidden:

    CUDA_VISIBLE_DEVICES="" strace -f -qq -e trace=%file -e signal=none -o <scratch>/strace_harness.txt \
        uv run python -B <scratch>/trace_reads.py <scratch>/trace_harness.json

With the GPU hidden, `common.DEVICE` is the CPU; the harness asserts this.

*What the harness exercised.*
- **Imports:** `sfm.{paths,data,legacy,registry,metric,train,models,families,refstream,gpulock,stats}` and
  `sfm.models.dist_factory`.
- **Data and legacy models:** `data.load_sets(("train","val"), device="cpu")` and `load_meta`; `legacy.load_mlp_std`,
  `legacy.load_component_formula(train)` (which runs `shapelab.build_final`), `legacy.load_abar` and
  `families.load_legacy_core`.
- **Registry, without `evaluate`:** `load_validation`, `validation_reference`, `check_validation` and
  `read_registry`, plus the benchmark-array `np.load` of `evaluate`.
- **Guard-array reuse check:** `registry._bench_guard_arrays` was called with a `bench_models` callable that raises.
  It therefore could only read, never recompute or write. It **reused** the laptop arrays.
- **Other registry calls:** `acceptance_template("stored")`, `_default_bench_models`, and `select_1se` on
  s2_seed11/12 and b0_seed0.
- **Emulator and lock:** `refstream.library_path`, then `_load` (dlopen of the cached `.so`; compiling was guarded
  out) and a small `EmulGen`; `gpulock.lock_path` and `holder`.
- **Acceptance templates:** all five kinds through `dist_factory.acceptance_template`.
- **Track A driver:** `stage0_specs`, `primary_specs` and `exploratory_spec` for every rung and both seeds (75 specs);
  `decisions`, `gpu_seconds_total`, `registered_row("A0_init")`, `frozen_lr`, `_est_seconds` and `done()`.
- **Track A builds:** every spec through the factory on the CPU with `skip_init=True`. A0, A2e and A2s were also
  built with `skip_init=False`, which only computes `kl_map` and runs no optimisation.
- **Track B configs and models:** every `configs/m2b/*.toml` and `configs/examples/*.toml` (loader plus done-file
  check), each built on the CPU (warm start skipped for `field_acc`).
- **Registered networks:** `load_run` of all 8 registered networks, with each file's SHA-256 compared to its registry
  row.
- **Ensembles:** E5_s1, E2_s2_seeds11-12, E5_s2 and E6 built, and `_member_shas` run.
- **Factories for AUDIT §4.2:** `field(32,4)`, `field(32,5)`, `field(64,4)`, `field_mlp_std`, `field_acc` with the
  front, stored and full_box templates, and `residual_table` with uniform and quantile knots.
- **θ helpers:** `legacy.C["event_slab"]` on 8 validation events, and `dist_factory.exit_depth`.

**Dynamic trace, script dry runs.** Each command ran under the same strace, again with `CUDA_VISIBLE_DEVICES=""`:
- `scripts/fit.py configs/m2b/s2_seed13.toml --dry-run`
- `scripts/fit.py configs/m2b/b0_seed1.toml --dry-run`
- `scripts/fit_queue.py configs/m2b --dry-run`
- `--help` of `run_m2a.py`, `run_m2b_fit.py`, `run_m2b_ensembles.py`, `run_m2b_stage1.py`, `fit.py`, `fit_queue.py`
  and `check_protected.py` (all of them use argparse)

**Not run, because running them would do work:**
- `build_cache.py`, `run_m2b_report.py` and `mc_noise_val.py` have **no argument parser**. Even `--help` would build
  the cache, write `summary.json`, or run on the GPU.
- `parity.py` is forbidden.
- `run_m2a.py` has no dry-run mode.

**Static reading.** Sources read: `src/sfm/*.py`, `src/sfm/models/*.py`, `families/__init__.py`, the
`components.load_legacy_core` path, and `scripts/run_m2a{,_lib}.py`, `run_m2b_{lib,fit,stage1,ensembles,report}.py`,
`fit.py`, `fit_queue.py`, `build_cache.py`, `parity.py`, `mc_noise_val.py`, `check_protected.py` and `audit_lib.py`.
Also read: `configs/**`, the tests that touch files or GPUs, `shapelab.py` (`build_final`, `final_round`,
`final_kwargs`, `lnp_center_scale`, `mean_acceptance`), `protonlab.py`, the AUDIT.md §4, the PLAN.md §10–15, both
STATUS files, `registry.jsonl`, `parity.json`, `c1_reference_stream.json`, `cache_build.json`, `pion_s1_meta.json`
and every `resources.json`.

**Git status.** It was checked with `git status --short --ignored`, `git ls-files` and the two `.gitignore` files.
Rules that apply:
- In `shape-form-martin/.gitignore`: `cache/`, `results/**/*.pt`, `results/**/*.npy`, `results/**/*.npz`.
- In the root `.gitignore`: `*.parquet`, `*.csv`, and `shape-form-proton/**/*.{pt,npz,npy,json,log}`.

`git ls-files --error-unmatch` and `git check-ignore -v` have **not yet** been run file by file. That is planned for
`list`.

## 2. Files read by the moved steps

Abbreviations: **SFM** = `shape-form-martin/`, **PION** = `shape-form-proton/results/particles/pion/`.

Columns:
- **grp**: a = tracked; b-live = copy to the same path; b-rec = laptop record, copy to a quarantine directory only
  (see §6); c = regenerated on the H200.
- **stream**: whether the content depends on the Monte-Carlo fold-point stream of one GPU.

### 2.1 Group b, live: copy by hand to the same path

| path | read by (code path) | git | size (B) | SHA-256 | stream |
|---|---|---|---|---|---|
| `shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet` | `sfm.data.parquet_getter` / `build_cache` (`build_cache.py`), `parity.py` raw-file check. No moved step reads it while the caches are copied. Needed for a cache rebuild or cross-check, and for the cache extension that Θ1b needs (absolute origin, AUDIT §4.3) | ignored (`*.parquet`) | 2604833654 | not computed | no |
| `SFM/cache/pion_s1_trainval.pt` | `sfm.data.load_sets` at `data.py:292`, in every step | ignored (`cache/`) | 385953467 | 9984886fb90214b9c556489bfb3a913aba89509239c9b4a190797a540a6ab4b4 (from `pion_s1_meta.json`, written at build; not re-hashed) | no |
| `SFM/cache/pion_s1_test.pt` | Only `load_sets(..., allow_test=True)`: `parity.py`, and the future `final_test.py`. **Opaque byte stream**: hash and size only | ignored | 68476893 | 14955929e0a6cfad42e43aa6fa81787a072d9f26d37c3054ef627faac4c9ec88 (from meta; not re-hashed) | no |
| `SFM/cache/pion_s1_meta.json` | `data.load_meta` at `data.py:300`; `registry.validation_reference`, which uses the trainval SHA-256 as the fingerprint key | ignored | 4051 | not computed | no |
| `PION/mlp_std.pt` | `legacy.load_mlp_std` (`legacy.py:45`), `models.field.mlp_std_start` (`field.py:70`); factories `field_mlp_std`, `field_acc`, `residual_table`, `ensemble(include_mlp_std)`; `run_m2b_ensembles._member_shas` | ignored (`shape-form-proton/**/*.pt`) | 314242 | not computed | no |
| `PION/new3_std_n3.pt` | `shapelab.build_final` at `shapelab.py:397`, reached from `legacy.load_component_formula` and `families.load_legacy_core`. Used by every Track A model (`dist_factory.legacy_core`) | ignored | 71174 | not computed | no |
| `PION/abar.pt` | `legacy.load_abar` (`legacy.py:66`), `field.acceptance_template("stored")` (`field.py:82`), `shapelab.build_final` (`shapelab.py:396`), and the `run_m2a.py` templates phase | ignored | 17876 | not computed | no |
| `PION/test_v3.json` | `shapelab.final_kwargs` at `shapelab.py:377` (degree and variant of the final formula; `final_round` checks that it exists). Stored *results*, not test data | ignored (`**/*.json`) | 3282 | not computed | no |
| `PION/mlp_std.json`, `PION/new3_std.json` | `parity.py stored_val_means` (stored validation means of the reference machine). Needed for a validation-only parity check on the H200 | ignored | 72 / 781 | not computed | reference-machine values |
| `PION/test_v3_nll.npz` | `parity.py` (A3, C1): stored per-event TEST arrays of the reference machine. Pairing partner for the final test on a GPU with grid 1056 (see §4) | ignored | 600804 | not computed | yes (reference stream, grid 1056) |
| `SFM/results/m0_parity/val_event_number.npy` | `registry.validation_reference` at `registry.py:143`; every registry evaluation | ignored (`results/**/*.npy`) | 119728 | not computed | no |
| `SFM/results/m0_parity/test_event_number.npy` | `parity.py --reference-stream` (`parity.py:316`); test-order check for the final test | ignored | 119776 | not computed | no |
| `SFM/results/m0_parity/abar_front.pt`, `abar_front_back.pt` | `field.acceptance_template("front"/"front_back")` (`field.py:82`). Used by A3f, B4f (`field_acc` front) and the `run_m2a.py` templates phase. Written only by `parity.py` default mode, which opens test, so they cannot be regenerated without opening test | ignored | 17982 / 18017 | not computed | no (deterministic `mean_acceptance`, no RNG) |
| `SFM/results/m2b_mlp/{s1_seed11,s1_seed12,s1_seed13,s1_seed14,s1_seed15,s2_seed11,s2_seed12,b0_seed0}/model.pt` | `models.load_run` (`models/__init__.py:95`): `from_run` for s2_seed13–15 (needs s1_seed13–15), the ensembles (s1_*, s2_11/12), re-scoring, and `run_m2b_lib.registered_row` (`run_m2b_lib.py:45`, matched by SHA-256). `train.fit` checks the `model.pt` SHA-256 against `fit_done.json` | ignored | see below | equals the registry row (verified in this trace) | no |

The 8 checkpoint SHA-256 values below were computed in the trace and equal `checkpoint_sha256` in `registry.jsonl`:

| run | size (B) | SHA-256 |
|---|---|---|
| b0_seed0 | 317250 | 75de31f2540769f68f652979163684213f43e92066f0a6a03f926595ecf6b700 |
| s1_seed11 | 315778 | d8f6e94d04b89e5bca1d9473b547ab61f57cef962e9e56884363a15fc84d907c |
| s1_seed12 | 315906 | be1a67e11dc8171c4a620d167e5b84ac4eba13e49fa414417b0e1142955714f6 |
| s1_seed13 | 316098 | 5e8e7ef9f434550cf60e65e5a7363b8c198a97be2d4893977e8bbcc58a2a22f6 |
| s1_seed14 | 316290 | c0d23ad25b8a555bc9b1bef3429fef0e2704d7865af737e1207a19e6fbe3c1f2 |
| s1_seed15 | 315650 | 95cb4fa4fde34e2dd8a17dd67255fcb7af00bb8b6fc7a458653fd50704752613 |
| s2_seed11 | 317314 | 0a1c0f02c39349aa43459d10713db8e8b0f12d83a814c04133f4c652238cdf6d |
| s2_seed12 | 317314 | 165123ad4ed76680c1bf996b992d8487307369a3f5e2a056d063b1b93790419a |

Running total of the files above (sizes from `stat`): about 3.06 GB, almost all of it the Parquet file (2.60 GB) and
the train/val cache (0.39 GB).

### 2.2 Group b, records: laptop records, never to the live path

All of these are `stream_dependent = true` unless marked otherwise.

| path | why it is a record | size (B) |
|---|---|---|
| `SFM/results/m0_parity/{mlp,formula}_val_nll.npy` | Pairing partners of `registry.evaluate` (`registry.py:411`), on the laptop stream (grid 144). **Dangerous at the live path**: if present, every H200 registry row pairs against them silently. Absent, `evaluate` fails loudly | 59928 each |
| `SFM/results/m0_parity/{mlp,formula}_test_nll.npy` | Laptop-stream test arrays (Gate 0) | 59952 each |
| `SFM/results/m0_parity/{mlp,formula}_{val,test}_nll_ref1056.npy` | Laptop *emulation* of the reference stream (grid 1056). These are the expected values of the H200's own `common.test_nll` if its grid is 1056 (§4), so they are useful verification partners | 59928 / 59952 |
| `SFM/results/registry_refs/{mlp,formula}_val_{k1024,bulk_stratum}.npy` | Benchmark guard arrays. **Dangerous at the live path**: `guard_spec.json` is tracked and holds no GPU identity, so copied arrays are reused silently. Proven in the trace: `_bench_guard_arrays` reused them. Absent, they are recomputed | 59928 / 59264 |
| `SFM/results/m2b_mlp/*/val_nll.npy` (8 files), `val_nll_k1024.npy` (3), `val_bulk_zb1221.npy` (3) | Per-event arrays of the registered runs, on the laptop stream. Read through the latest registry row by `run_m2b_ensembles._arr`, `run_m2b_report.main`, `registry.select_1se` and `run_m2a_lib.val_array` | 59928 / 59928 / 59264 |
| `SFM/results/m2b_mlp/{b0_seed0,s2_seed11,s2_seed12}/val_mass_monitors.npz` | Deterministic map functionals; stream_dependent = **false**. Laptop record; regenerated by guarded re-scoring; read by no code | 958826 each |
| `SFM/results/gates/gate0_checks/val_grid1056_*.npy` (13), `val_grid144_*.npy` (13) | Gate 0 emulator arrays. `tests/test_refstream.py:113` uses `val_grid1056_mlp_std.npy` if present | 1558128 total |

### 2.3 Group a: tracked; arrives with `git clone` or `git pull`

The branch is pushed: `HEAD` and `origin/shape-form-martin` are both `ccd9a6f`.

**Code.**
- `SFM/common.py`
- `SFM/src/sfm/**/*.py` and `SFM/src/sfm/_philox_emul.c`. The `.c` file is read by `refstream.library_path`; its
  hash names the `.so`.
- `SFM/scripts/{run_m2a,run_m2a_lib,run_m2b_lib,run_m2b_fit,run_m2b_stage1,run_m2b_ensembles,run_m2b_report,fit,fit_queue,build_cache}.py`
- `shape-form-proton/{shapelab,particle_shape,proton_shape,families,protonlab}.py`. These are imported by
  `sfm.legacy`. The `__pycache__` holds cpython-310 files only, so the `.py` files are read directly. Nothing is
  written there when run with `-B`.
- `pyproject.toml`, `uv.lock`

**Configs.** `SFM/configs/m2b/*.toml` (8) and `SFM/configs/examples/*.toml` (3).

**State and records.**

| file | role | stream |
|---|---|---|
| `SFM/results/registry.jsonl` | Read by `read_registry`; **appended** by `evaluate`. Its 8 laptop rows are records | dependent |
| `SFM/results/test_access.log` | Appended on every test load. Gate 2c expects exactly 2 lines before the final run | — |
| `SFM/results/registry_refs/guard_spec.json` | Read; rewritten with the same content when the guard arrays are recomputed | — |
| `SFM/results/m2b_mlp/*/done.json` (8) and `{b0_seed0,s2_seed11,s2_seed12}/fit_done.json` | Skip markers. `fit_done.json` holds the `model.pt` SHA-256 | — |
| `SFM/results/m2b_mlp/*/resources.json` | Appended per run segment | — |
| `SFM/results/m2b_mlp/summary.json` | Rewritten by `run_m2b_report.py` | — |
| `SFM/results/m0_parity/{parity,c1_reference_stream,cache_build,mc_noise_val,checksums}.json` | Gate 0 records; `cache_build.json` is rewritten by `build_cache.py` | — |
| `SFM/results/m1_audit/checks/c_side.py` | Source of the ported `exit_depth` (secondary columns: contained vs side-leaking). Not imported at run time | — |

### 2.4 Group c: regenerated on the H200

| file | regenerated by | laptop time | reproducibility |
|---|---|---|---|
| `.venv` | `uv sync` (`uv.lock`: torch 2.14.0, cu130 wheel; needs a driver that supports CUDA 13) | not measured | machine-dependent install |
| `SFM/cache/philox_emul_<sha16>.so` (laptop: `_622ce7d283034712`) | Automatic on first emulator use (`refstream._load`: `gcc -O2 -fopenmp -shared -fPIC -ffp-contract=off`); needs gcc with OpenMP | seconds (not measured) | output bits machine-independent (integer Philox, no FP contraction) |
| `SFM/cache/pion_s1_val_fingerprint.json` | Automatic in `registry.validation_reference` when absent or keyed to another trainval SHA | seconds | tensor hash: with the copied cache it must equal `731a199763cc31c498986574083e95c9c6c61e09d83f69b294cee13cf07858a7` |
| `SFM/cache/gpu.lock` | Created on first `gpu_lock` | — | — |
| `SFM/results/m0_parity/{mlp,formula}_val_nll.npy`, H200 versions | **No command exists** (§5) | `common.test_nll` takes 1.48 s (mlp) and 0.90 s (formula) on validation (`parity.json`) | machine-dependent (stream) |
| `SFM/results/registry_refs/*.npy` | Automatic at the first `registry.evaluate(guards=True)` when the files are absent | about 27 s (b0_seed0's guarded evaluation took 42.8 s against 16 s for the others) | machine-dependent |
| Re-scored arrays of the 8 registered runs | **No command exists** | stage 1: 1–2 s each (no guards); stage 2 and B0: 16 s each (with guards) | machine-dependent |
| `SFM/results/m2a_dist/templates/{back_corrected.pt,full_box.pt,templates_all.npz,templates.json,train_theta.npy}` | `run_m2a.py templates`. Read by A3c, A3x, A9, A9r (traced FileNotFoundError) and B4x | never run | `train_theta.npy` is bit-identical; the templates are GPU float32, so identical only up to rounding |
| `SFM/results/m2a_dist/{plr.json,decisions.json,gpu_time.jsonl}` and `<run>/{model.pt,checkpoint.pt,done.json,init.json}` | Produced by the Track A phases. Later phases read `A2s_s*/model.pt` and `A{4,5,7,8,9,10}_s0/model.pt` (traced) | — | machine-dependent |
| `SFM/results/m2b_mlp/{s2_seed13..15,b0_seed1,b0_seed2}/*`, the ensembles, `ensemble_subsets/*.npy` and `ensemble_scaling.json` | Track B commands (§5) | — | machine-dependent |

### 2.5 Considered and excluded

| file | why it is excluded |
|---|---|
| `SFM/results/m2b_mlp/{b0_seed0,s2_seed11,s2_seed12}/checkpoint.pt` | Read by `train.fit` only when `fit_done.json` is absent, and it is present. A resumed laptop checkpoint would also carry laptop CUDA RNG states |
| `SFM/results/m2b_mlp/s1_seed*/stage1_fit.pt` | Read only for a seed without `done.json` |
| `*/fit.log`, `logs/*.log` | Logs. `s2_seed13` holds a start line only and restarts from step 0 |
| `SFM/results/gates/gate0_checks/{philox_emul.so,events_table.npz,protected/}` | Gate 0 artefacts; no moved step reads them |
| `SFM/results/m1_audit/**/*.npz` | Not read by `dist_factory`, which ports the `c_side.py` code and reads no `c_side.npz`; confirmed in the trace |
| `SFM/results/m2c_theta/geometry/explore/train_columns.npz` | Belongs to the geometry audit, which does not move |
| The other files in `PION/` | Never opened in the trace |
| The Julia environment | PySR does not move |

## 3. The two caches: copy or rebuild (decision: COPY both and check them by SHA-256)

**What a rebuild does.**
- `scripts/build_cache.py` has no argument parser; the only option is the `--no-full-check` string. Any invocation,
  even with `--help`, rebuilds.
- `sfm.data.build_cache` **always writes both** `pion_s1_trainval.pt` and `pion_s1_test.pt` (`data.py:226-227`),
  plus `pion_s1_meta.json`.
- The script then writes `results/m0_parity/cache_build.json`, which is tracked.
- A rebuild therefore overwrites copied files and changes a tracked file, unless both `SFM_CACHE_DIR` and
  `SFM_RESULTS_DIR` point to a scratch directory.
- Laptop cost: 12 s build, 40.1 s total with both exactness checks, 4.05 GB peak RSS.

**Byte reproducibility.**
- `pion_s1_meta.json` is never byte-identical: it holds `created_utc` and the Parquet `mtime_ns`.
- The `.pt` files are byte-reproducible when the tensors are bit-identical, under the same torch version and file
  name. Tested here: two separate processes saving the same tensor produced identical SHA-256. The
  `.data/serialization_id` inside the zip is deterministic, and the archive prefix is the file stem.

**Tensor reproducibility is not guaranteed on another CPU.**
- `build_showers` uses numpy float64 `np.cos` for `zb` and `common.rotation(theta, phi)`.
- numpy dispatches these to AVX-512 SVML kernels where the CPU has AVX-512.
- The laptop CPU (i7-14650HX) has no AVX-512; typical H200 hosts (Xeon SPR, EPYC Genoa) do. Last-bit differences are
  therefore possible.
- `build_cache.py`'s exactness check compares against pandas *on the same machine*, so it would pass while still
  differing from the laptop cache.

**Gate 0 and C1 validated the laptop bytes.**
- Gate 0 showed the train/val tensors bit-identical to the reference loader. C1 showed exact per-event parity on the
  test tensors.
- Only a byte-identical copy carries both results to the H200. The test cache must stay unopened, so a plain SHA-256
  of the copy is the right check.
- If wanted, a rebuild into scratch can serve as a cross-check: an equal SHA-256 proves bit-identity on the H200
  as well.

**The validation fingerprint.** `cache/pion_s1_val_fingerprint.json` is a SHA-256 over the tensor bytes and is keyed
by the trainval file's SHA-256. It is machine-independent if the tensors are identical. With the copied cache the
value must be `731a1997…58a7`.

## 4. Machine-specific constants and assumptions (none fixed; consequences on the H200)

**Main finding.** On the H200 the local stream will probably be the reference stream. The launch grid is
`refstream.grid_for(SMs, maxThreadsPerSM) = SMs × (maxThreadsPerSM // 256)`. An H200 has 132 SMs and 2048 threads
per SM, which gives **1056 = `REFERENCE_GRID`**, the grid of the earlier study's machine.
- `common.test_nll` on the H200 would then draw the earlier study's fold points. Expected validation means:
  18.118083 (MLP) and 18.120407 (formula), as emulated on the laptop and stored. The laptop's "numbers of record"
  (A6: 18.117857 and 18.120061) would not reproduce.
- This prediction is unverified. Check it with
  `uv run python -B -c "import sys; sys.path.insert(0,'shape-form-martin/src'); from sfm import refstream; print(refstream.local_grid())"`.
- A MIG slice has another SM count and therefore another grid.

| # | file:line | constant or assumption | what happens on the H200 if nothing changes |
|---|---|---|---|
| 1 | `scripts/run_m2a.py:43`, check at `:260`, stop at `:266` | `FORMULA_OF_RECORD = 18.120061`, tolerance 1e-5 | The parity phase registers `A0_init` (a registry write happens first), records `passed: false` in `decisions.json`, and raises `SystemExit("STOP: harness parity failed")`. With grid 1056 the difference is about 3.5e-4. Track A cannot proceed |
| 2 | `scripts/run_m2a.py:170-173`, `:305-317` | A0 drift rule pairs against `results/m0_parity/formula_val_nll.npy` | If the laptop array is copied: A0 (H200) − formula (laptop) contains the difference between the two streams, about +3.5e-4, which exceeds 2e-4 and triggers a false STOP. If the array is absent: FileNotFoundError |
| 3 | `scripts/run_m2a.py:44`, `:358-381`, `:397` | `BUDGET_S = 4 h`; laptop-based step costs of 0.055 and 0.07 s per step; 180 s per guard evaluation; drop rule on `gpu_time.jsonl` seconds | The pre-registered drop order depends on machine speed. A faster H200 drops fewer or no exploratory runs |
| 4 | `src/sfm/registry.py:20`, `:86-94`, `:411`, `:446` | Benchmark pairing arrays are "this machine's stream"; rows record only `device: "cuda"`, with no GPU name and no grid | With laptop arrays copied, every H200 row pairs silently against laptop points: not paired on identical points, so the SE is wrong. With the arrays absent, a loud FileNotFoundError |
| 5 | `src/sfm/registry.py:296-338` | `_guard_spec` holds no GPU or stream identity; `guard_spec.json` is tracked | Copied `registry_refs/*.npy` are reused silently, as shown in the trace. If absent, they are recomputed |
| 6 | `scripts/run_m2b_lib.py:51-67`, `scripts/run_m2a_lib.py:159-171` | `registered_row` matches by checkpoint SHA-256 only | Re-running a registered run's script reuses the laptop row ("already registered; not evaluated again"), so re-scoring is impossible through the existing scripts |
| 7 | `scripts/run_m2b_ensembles.py:162-169`, `:177`; `scripts/run_m2b_report.py:42-49`; `scripts/run_m2a_lib.py:174-182`; `src/sfm/registry.py:521-547` | Latest row per name, plus `m0_parity/mlp_val_nll.npy` | Before re-scoring, `scaling_table` would pair H200 ensemble arrays against the laptop arrays of seeds 11 and 12: a silent mix if copied, FileNotFoundError if absent |
| 8 | `src/sfm/refstream.py:51`; `tests/test_refstream.py:52` | `LOCAL_GRID = 144` | `test_local_grid_constant_matches_this_gpu` fails (1056 ≠ 144) |
| 9 | `tests/test_refstream.py:72-79` | Assumes that the local grid ≠ 1056 | `assert (a == b).mean() < 0.05` fails, because the local stream equals the reference stream |
| 10 | `scripts/audit_lib.py:48` (used at `audit_parts.py:46`, `:66`) | `RECORD` holds the laptop validation numbers | M1 audit re-runs on the H200 compare against laptop numbers. Not a moved step |
| 11 | `src/sfm/registry.py:439-441`; `train.py:661`; done-markers; `config_path` in `model.pt` meta | Absolute laptop paths | Informational only; no loader uses them |
| 12 | `src/sfm/train.py:232-244`, `:427`, `:446` | Checkpoints hold CUDA RNG states and the CUDA fold generator | Resuming a laptop checkpoint gives a hybrid fit on other points. No partial checkpoint is needed today |
| 13 | `common.py:64`; `src/sfm/refstream.py:64-69` | Always `cuda:0`; `local_grid(device=0)` | On a multi-GPU node every job runs on GPU 0, and `cache/gpu.lock` serialises them. With MIG the grid changes |
| 14 | `src/sfm/refstream.py:84-94` | gcc with `-fopenmp` must exist | Without it, the first emulator use fails (`test_refstream`, `parity --reference-stream`, reference-stream columns) |
| 15 | `scripts/run_m2b_lib.py:110`, `scripts/run_m2a_lib.py:121` | `min_free_mb` 3000 / 2500, tuned for an 8 GB GPU | Harmless on 141 GB |
| 16 | `tests/test_families_components.py:484-499` | Cost budget written for the RTX 4060 | Passes, or skips on a busy shared GPU |
| 17 | `scripts/check_protected.py` together with `results/m0_parity/checksums.json` | Expects all 1997 protected files, plus the size and mtime of the data files | Not a moved step. On the H200, `verify` reports every uncopied ignored file as removed (exit 1). The Parquet mtime must be preserved (`rsync -a`) |
| 18 | `src/sfm/data.py` build path | numpy float64 transcendental functions | See §3: a rebuild may not be bit-identical |
| 19 | `src/sfm/metric.py:79-97` | `deterministic()` gives bit-reproducibility per machine only | H200 arrays are reproducible on the H200 and differ from the laptop's |
| 20 | `scripts/parity.py:412` vs `src/sfm/registry.py:418-419` | The laptop pairing arrays were computed outside `deterministic()`; registry arrays are computed inside it | A methodological note for the recomputation script on the H200 |

## 5. Run-book: which commands exist

Argument parsers were checked with `--help`, or by reading the source where `--help` would execute work.

| action | existing command, or "missing" |
|---|---|
| Environment check | `uv sync`; `uv run pytest shape-form-martin/tests -q` (expect the two `test_refstream` failures of §4 #8 and #9). There is no script for the grid or GPU identity, only the one-liner in §4 |
| Cache build or verify | Build: `uv run python -B shape-form-martin/scripts/build_cache.py [--no-full-check]` (no argparse; writes both caches, meta and `cache_build.json`). Verify: **missing**; this is the planned `make_transfer_package.py verify` |
| Validation-only benchmark parity | **Missing**. `parity.py` default mode opens TEST, needs `--overwrite` because `parity.json` exists, replaces the pairing arrays and appends to `test_access.log`. Needs a validation-only script that compares with `mlp_std.json` / `new3_std.json` and the `*_ref1056` records, and logs the GPU name and grid |
| Recompute the validation pairing arrays without test | **Missing**. Needs a script that writes `results/m0_parity/{mlp,formula}_val_nll.npy` with `common.test_nll` inside `deterministic()` |
| Re-score the 8 registered runs | **Missing**. `run_m2b_fit` and `run_m2b_stage1` skip on `done.json`, and `registered_row` reuses the laptop rows. Needs a script: `load_run` → `registry.evaluate(guards = as originally)` |
| Resume the Track B seeds | `uv run python -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/s2_seed13.toml shape-form-martin/configs/m2b/s2_seed14.toml shape-form-martin/configs/m2b/s2_seed15.toml shape-form-martin/configs/m2b/b0_seed1.toml shape-form-martin/configs/m2b/b0_seed2.toml` (usage: `run_m2b_fit.py configs [configs ...]`). Alternative: `fit_queue.py [--dry-run] [--stop-on-error] configs...`. **Precondition**: the H200 pairing arrays exist |
| Ensembles | `uv run python -B shape-form-martin/scripts/run_m2b_ensembles.py [--table-only]`. **Precondition**: the 8 runs are re-scored first |
| Track B report | `uv run python -B shape-form-martin/scripts/run_m2b_report.py` (no argparse; rewrites the tracked `summary.json`) |
| Track A phases | `uv run python -B shape-form-martin/scripts/run_m2a.py templates parity plr stage0 stage1p stage1x` (positional phases from {templates, parity, plr, stage0, stage1p, stage1x, guards, stage2, all}). **Blocked** by §4 #1 |
| Track A report, guards, Stage 2 | **Missing**. The phases `guards` and `stage2` import `scripts/run_m2a_report.py`, which does not exist (ImportError). A11p belongs to Fable |
| Remaining Track B runs (AUDIT §4.2) | **No driver**. Configs plus `fit.py` would work for B2a, B2b, B2c, B5f (`field_mlp_std`), B3s (`field`) and B4f / B4s (`field_acc` front / stored). Missing: B4x, because `field_acc` rejects `full_box` (traced: `ValueError`, since `models.field.acceptance_template` knows only stored, front and front_back); B6a/b/c (no factory for "fresh field + `kl_warm_start` to mlp_std"); B1d / B1f (distillation from E5); B-ceil (`train_fraction` option) |
| θ track Θ1–Θ3 | **No code**. Needs its own harness (`forward(lnp, aux)`). Θ1b also needs the absolute origin in the cache, which means extending `sfm/data.py`, rebuilding from the Parquet file, and a user decision |
| Final test | **Missing**: `final_test.py` and `results/final/LOCK.json` |

## 6. Plan for Part 2 (not implemented)

**Files and tables.**
- `make_transfer_package.py`: standard library only, `sys.dont_write_bytecode`, never imports torch.
- It holds explicit tables for groups a, b-live, b-records, c and excluded, as in §2.
- Glob patterns are used only with an expected count.

**`list`:**
- Checks each file with `git ls-files --error-unmatch` and `git check-ignore -v`.
- Computes sizes and chunked SHA-256 values; the test cache is marked opaque.
- Refuses any `*.csv`.
- Writes `files_b.txt` and `SHA256SUMS` (live), `files_b_records.txt` and `SHA256SUMS.records` (records), and
  `MANIFEST.json` (groups, totals, the machine constants of §4, the run-book of §5, and the git commit).

**`verify`:**
- Checks the live files' presence, size and SHA-256 (`--skip-large`, `--large-threshold`).
- Checks group a: tracked, present, and the commit is an ancestor (`git merge-base --is-ancestor`).
- Reports group c as "to be regenerated".
- Records: verified under the quarantine root `shape-form-martin/cache/laptop_records/` (inside the ignored `cache/`);
  a **warning** when a laptop record's exact bytes sit at a live path.
- Exits non-zero on any mismatch.

**Test.** Temp git repo; round trip list → copy → verify; a corrupted file is detected; a CSV is refused. A
subprocess wrapper with an audit hook asserts that the test cache is opened only, `rb`, by the hashing function, and
that no `pickle.find_class` or torch import happens.

## 7. Not yet traced or verified

**Hashes.**
- No SHA-256 computed for the Parquet file, the meta and fingerprint JSON files, the PION files, the m0_parity files,
  or any record.
- The two cache SHA-256 values are copied from `pion_s1_meta.json` and have not been re-hashed.
- No per-file `git ls-files --error-unmatch` or `git check-ignore -v` run yet.

**Read only inside a fit, a registration, a guard evaluation or a resume.** Known from static reading only:
- `train.fit`: `checkpoint.pt`, `fit_done.json`, `fit.log`.
- `registry.evaluate`: `_save_array` replace check, `_append`, guard files.
- `run_m2a_lib.run_spec`: `init.json`, `gpu_time.jsonl`.
- `run_m2b_ensembles`: `ensemble_subsets/timing.json`.
- `run_m2a.py templates`: the job reads `abar.pt` and `abar_front.pt`; both were traced through `acceptance_template`.

**Builds that need files that do not exist yet.**
- Builds with `skip_init=False` that need a finished start run (`init_shower_long`, `init_from_target`,
  `logit_dh_from_far`, net `fit_to`) were not exercised. They need `A2s_s*` runs and would optimise.
- The mixture and net specs whose `net_from` or `start` is missing failed with FileNotFoundError, as expected.

**Not run at all.** `run_m2b_report.py`, `run_m2b_ensembles.py`, `run_m2a.py` phases, `build_cache.py` and
`mc_noise_val.py` would write or compute.

**Not checked.**
- The prediction that the H200 grid is 1056.
- Whether the H200 driver supports CUDA 13.
- Whether gcc with OpenMP is present on the H200.
- The AVX-512 claim for the H200 host CPU (a property of numpy dispatch, not observed).

**Track B and θ details.** Of the AUDIT §4.2 runs, only the factory builds were traced; no code exists for B1d, B6,
B-ceil or the θ arms, so their future file reads are unknown beyond what the AUDIT names.

**Track A timing.** Laptop durations for Track A are estimates only (AUDIT §4.1 about 4 GPU-hours;
`_est_seconds` defaults). Nothing was measured.

## 8. Measured laptop durations (RTX 4060 Laptop, from the result files)

| step | time | source |
|---|---|---|
| Cache build | 12.0 s; 40.1 s with checks (row groups 2.1 s, full 25.9 s); 4.05 GB peak RSS | `results/m0_parity/cache_build.json` |
| Gate 0 parity, total | 52.4 s | `parity.json` |
| `test_nll` per benchmark | val: mlp 1.48 s, formula 0.90 s; test: 1.37 s and 0.89 s | `parity.json` |
| Train step, batch 256 | 0.035 s at K=16, 0.054 s at K=64 | `parity.json` |
| C1 | 7.1 s | `c1_reference_stream.json` |
| Track B stage 1 (`common.fit`) | s1_seed11 378 s, s1_seed12 451 s, s1_seed13 560 s, s1_seed14 712 s, s1_seed15 264 s; evaluation 1–2 s each | `*/resources.json` |
| Track B stage 2 and B0 fits (6000 steps) | 238 s each (s2_seed11, s2_seed12, b0_seed0); guarded evaluation 16 s (b0_seed0: 42.8 s, including the benchmark guard arrays) | `*/resources.json` |
