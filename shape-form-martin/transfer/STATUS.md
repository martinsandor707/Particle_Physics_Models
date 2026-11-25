# H200 transfer package: status (eight deferred files: P4 + the raw Parquet table; PLAN.md not updated)

Packaging finished on 25 Nov 2026 and was amended twice the same day:
- **P4:** "do not transfer the seven test-derived files now; split the file list and copy them to the H200 only after
  Gate 2c, so physical absence proves zero test access" (committed in `83ab7bd`).
- **Raw table:** the Parquet raw hit table is deferred as well, because it holds the raw hits of all events, the test
  events included (this amendment).

PLAN.md section 16 (which still describes the single-phase package), the commit and the push are the orchestrator's.
The statistics review is in `REVIEW_STATUS.md` (not touched here).

**Produced (modified until the orchestrator commits them, explicit paths only):**
- `shape-form-martin/scripts/make_transfer_package.py`: `list` (laptop) and `verify` (H200, two phases), standard
  library only. It never imports torch or numpy, refuses any `*.csv`, and treats the test cache as an opaque byte
  stream.
- `shape-form-martin/tests/test_transfer_package.py`: 26 tests on synthetic temp repositories, all pass.
- `shape-form-martin/transfer/{files_b_inputs.txt, files_b_laptop_records.txt, files_b_test_derived_inputs.txt,
  files_b_test_derived_records.txt, SHA256SUMS, MANIFEST.json}`, written by `list` at commit `83ab7bd`. The manifest
  records `tracked_tree_dirty: true` because the script, its test and this file differ from `83ab7bd`.
  `MANIFEST.json` holds every field of every file; this file only summarises it.

**Guardrails kept:**
- No training, fit or registry write.
- The test cache was opened only by the byte hash in `sha256_file` (also shown by the test's audit hook).
- No CSV was listed, stat'ed or hashed. strace of `verify`, including its tree walk, shows no file syscall naming a
  `.csv`.
- Nothing under `shape-form-proton/` or `shape-form-original/` was modified.
- `parity.py`, `build_cache.py` and `check_protected.py` were not run.
- The real deferred files were never moved, renamed or deleted; their absence was simulated (section 4).

## 1. The split (exact; from `MANIFEST.json`)

| list | group | phase | destination root on the H200 | files | bytes |
|---|---|---|---|---|---|
| `files_b_inputs.txt` | b1 | now | `<repo>/` | 19 | 389,052,272 |
| `files_b_laptop_records.txt` | b2 | now | `<repo>/shape-form-martin/cache/laptop_records/` | 51 | 5,749,702 |
| `files_b_test_derived_inputs.txt` | b1 | after Gate 2c | `<repo>/` | 4 | 2,674,031,127 |
| `files_b_test_derived_records.txt` | b2 | after Gate 2c | `<repo>/shape-form-martin/cache/laptop_records/` | 4 | 239,808 |

| totals | files | bytes |
|---|---|---|
| group a (tracked; arrives with git) | 129 | about 2.6 MB (snapshot at generation; exact value in `MANIFEST.json`, not checked by `verify`) |
| group b1 | 23 | 3,063,083,399 (now 19 / 389,052,272; after Gate 2c 4 / 2,674,031,127) |
| group b2 | 55 | 5,989,510 (now 51 / 5,749,702; after Gate 2c 4 / 239,808) |
| phase now (b1 + b2) | 70 | 394,801,974 |
| phase after Gate 2c (b1 + b2) | 8 | 2,674,270,935 |
| group c | 18 entries | 17 regenerated on the H200, plus 1 not-yet-existing input made on the laptop (absolute shower origin) |

**The eight deferred files.** `files_b_test_derived_inputs.txt` keeps its name but now holds four files; the
SHA256SUMS section header and `MANIFEST.json` `lists.test_derived_inputs.holds` say so.
- Raw table, copied to the same path:
  `shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet` (2,604,833,654 B).
  Flags: `contains_test_events = true`, `test_derived = false`.
- Test-derived inputs (`test_derived = true`), copied to the same path:
  - `shape-form-martin/cache/pion_s1_test.pt` (68,476,893 B);
  - `shape-form-proton/results/particles/pion/test_v3_nll.npz` (600,804 B);
  - `shape-form-martin/results/m0_parity/test_event_number.npy` (119,776 B).
- Test-derived laptop records, copied into the quarantine root: `results/m0_parity/{mlp,formula}_test_nll.npy` and
  `{mlp,formula}_test_nll_ref1056.npy` (59,952 B each).

Every b-file in `MANIFEST.json` has:
- `test_derived` and `contains_test_events` (booleans);
- `transfer_phase` (`now` / `after_gate2c`) and `list`;
- a `test_note` where its name or content mentions the test split.

Policies:
- `policies.raw_table`: one paragraph on the difference between the two flags and the user's decision of
  25 Nov 2026.
- `policies.test_events_in_raw_parquet`: updated to the new state.
- `policies.deferred_files`: the 8 files.
- `policies.test_derived_files`: the 7 files.

Other changes:
- `lists.*` carry `holds`, `test_derived_files` and `contains_test_events_files`.
- `transfer_commands` keeps the four rsync commands.
- `absence_before_gate2c` holds 12 file:line consequences, each line checked by `list`.
- `list` refuses an entry that is both `test_derived` and `contains_test_events`, and `contains_test_events` outside
  b1.

## 2. Other b1/b2 files with test-event information (search result, unchanged)

- No file in the "now" set carries per-event or per-hit information about test events.
- **`pion_s1_meta.json`** travels now, as stated in the manifest. It holds summary metadata only: `counts.test`,
  `files.test` (hash and size of the test cache), and the aggregate statistics in `splits.test`.
- **`test_v3.json`** travels now, as stated in the manifest. It holds the formula structure read by
  `shapelab.final_kwargs`, plus the earlier study's summary test numbers. It has no per-event values.
- **`mlp_std.json` and `new3_std.json`** are validation only.
- **`pion_s1_trainval.pt`:** train and val only. Its geometry constants come from all rows, and its per-event `index`
  implicitly locates the test events (membership only).
- **Checkpoints and templates:** no test content.
- **The Parquet table** was the one "now" file with test-event hits. It is now deferred.

## 3. `verify` (two phases + source machine)

**Default = before Gate 2c.** It checks the "now" files by size and SHA-256 and keeps every existing check. It FAILS
if any of the eight deferred files is on this machine:
- a test-derived file at its live path or under the quarantine root (any content);
- a test-derived file with the same name and size anywhere in the working tree, confirmed by SHA-256. For the test
  cache this is a size compare, then a byte hash only if the sizes match.
- the raw Parquet table at its live path (any content);
- the raw Parquet table with the same name and size elsewhere, confirmed by SHA-256.

How the search treats other files:
- A same-name file of another size is information only. For the Parquet, the message says "(for example a
  train/val-only extract; not the raw table)".
- Any other `*.parquet` file outside `.venv/` is also listed as information.
- Each file is hashed at most once per run. Files above 128 MiB carry "SHA-256 of N B computed once" in the output.
- The walk skips `.git` and skips `*.csv` names before any stat.

Output:
- Each offending path is printed as `DEFERRED FILE ON THIS MACHINE: <path>`.
- The summary reads `deferred files absent: yes/no (8 checked: 7 test-derived + the raw Parquet table)`.
- A `RAW TABLE:` line sits next to the `TEST CACHE:` line.

**`--after-gate2c`** requires all eight present with correct size and SHA-256. The test cache is checked by byte hash
only; the Parquet file by one hash, with an mtime warning against `checksums.json`. The summary reads
`deferred files present and verified (after Gate 2c): yes/no (8 checked: ...)`.

**`--source-machine`** checks all eight at their live source paths; that is not a failure.

## 4. Consequences of the absence before Gate 2c (also in `MANIFEST.json` `absence_before_gate2c`)

**How it was checked.** A laptop run, read-only, on the CPU:
- An audit hook raised `FileNotFoundError` on any open, and a wrapper of `os.stat`/`os.lstat` on any stat, of all
  eight paths.
- Writes under the repo and `*.csv` opens were refused.
- **The whole run was traced with strace**, because pyarrow opens files from C++ without a Python audit event.
- Imports of `pyarrow`/`pandas` were recorded.
- The harness is in the session scratchpad (`absence/hide_trace8.py`, `absence/hideplug8.py`).

**What was exercised:**
- imports of all `sfm` modules;
- `load_sets(("train","val"))` from the copied cache, and `load_meta`;
- the legacy loaders, `shapelab.build_final` and `load_legacy_core`;
- the registry read paths;
- the templates and the emulator library;
- Track A spec builds and Track B factories;
- `load_run` of the 8 networks;
- `fit.py --dry-run` and `fit_queue.py --dry-run`;
- the `run_m2b_fit.py` config start-up;
- import and `--help` of `run_m2a.py`, `run_m2b_fit.py`, `run_m2b_ensembles.py` and `run_m2b_stage1.py`, and import of
  `run_m2b_report.py`;
- `pytest --collect-only` of all 458 campaign tests;
- `tests/test_data.py`: 5 passed.

**Result:**
- Zero opens or stats of any of the eight, apart from the harness's own probe.
- strace shows no file syscall naming the real Parquet file, any test-derived file or a `.csv`.
- `pyarrow.parquet` was never imported by the code under test. `pandas`/`pyarrow` core are imported by
  `sfm/data.py:32`, but nothing reads a Parquet file.

### Parquet file absent

None of these changes a moved step:
- `sfm/data.py:55` (`parquet_getter`, the first I/O of `build_cache`, `data.py:216-217`; also the exactness checks
  `:327`, `:367`): `FileNotFoundError` ("Failed to open local file", shown with a missing scratch path). That happens
  before anything is written: the caches at `:226-227`, the meta file at `:246` and `cache_build.json` at
  `build_cache.py:64-66` all come later. `build_cache.py` cannot rebuild on the H200 before Gate 2c, which is
  intended.
- `scripts/parity.py:472` (raw-file check of the default mode) is not reached: the default mode is refused at `:342`
  and fails at `:368` without the test cache.
- `scripts/check_protected.py:84-86`/`:152`: the Parquet file becomes a second "removed" `stat_only` entry after the
  CSV, and `test_v3_nll.npz` a 1806th removed hashed file. The exit code is 1 on the H200 either way, and the script
  rewrites the tracked `checksums_verify.json` (`:155`).
- Gate 0 record scripts and the laptop geometry audit (not moved) raise `FileNotFoundError`:
  - `gate0_checks/c05_split_integrity.py:19`;
  - `c06_cache_fidelity.py:28`;
  - `c11a_hits_1001.py:14`;
  - `c12_units_lowp.py:55`;
  - `c15_geometry_train_only.py:11`;
  - `m2c_theta/geometry/explore/e1.py:15`, `e2.py:11`, `e3.py:12`.
- **Tests:**
  - No campaign test reads the Parquet file. There is no cache-fidelity test in `tests/`; that check lives in the
    Gate 0 record `c06_cache_fidelity.py`.
  - `tests/test_data.py:17` (`test_paths_layout`) only checks the suffix of `paths.parquet_path()`, a path
    computation with no stat; it passes.
  - Nothing skips or fails because of the absence.
- **The zero-training geometry arm** needs the absolute shower origin of the validation events. The cache does not
  store it (`sfm/data.py:132-135` computes z0 only to derive zb), and the Parquet file is absent until after Gate 2c.
  The origin must therefore be computed on the laptop and shipped as a small file keyed by event number:
  - computed from train/val rows only;
  - origin = energy-weighted centroid of the lowest-z layer over all rows of the event
    (`results/m2c_theta/geometry/STATUS.md` §1.1);
  - about 15,000 validation events, well under 1 MB.

  This is recorded as a not-yet-existing input in group c. No script and no reader exist yet; the geometry
  constants of §1.3 travel as text or code.

### Test-derived files absent (unchanged from the P4 amendment)

- `sfm/data.py:286-287`: `load_sets(test, allow_test=True)` writes the log line before opening, so on the H200 a log
  line before Gate 2c means an attempted access. The locked calls raise earlier: `:283` without `allow_test`, `:285`
  with an empty reason.
- `parity.py:316` (C1 with `--overwrite`): `FileNotFoundError` before `load_sets(test)` at `:318`, but only after
  `:314` has written the val ref1056 arrays at their live paths.
- `parity.py:368` (default mode with `--overwrite`): writes a log line, then raises `FileNotFoundError`.
- Gate 0 record scripts:
  - `c05:102`;
  - `c08:110`, `:117-118`;
  - `c08b:15-16`;
  - `c14:119-121`;
  - `c16:15`, `:19`.
- `sfm/data.py:168`: `cache_files()["test"]` is a path only.
- `tests/test_data.py:34`: passes.

## 5. Script outputs (real run on the laptop)

`list` (commit `83ab7bd`, tracked tree dirty; last run after this file was written):
- group a: 129 files, about 2.6 MB;
- group b1: 23 files, 3,063,083,399 B (now 19 / 389,052,272; after Gate 2c 4 / 2,674,031,127);
- group b2: 55 files, 5,989,510 B (now 51 / 5,749,702; after Gate 2c 4 / 239,808);
- group c: 18 entries;
- phase now: 70 files, 394,801,974 B; phase after Gate 2c: 8 files, 2,674,270,935 B;
- "deferred to after Gate 2c: 8 files = 7 test-derived + 1 raw table(s) with test events".

All cross-checks are equal, and all cited lines (29 constants, 12 absence consequences) were confirmed.

`verify --source-machine` on the laptop: **PASS**. 80 checks, 0 failures, 0 warnings. The Parquet line reads
"[SHA-256 of 2604833654 B computed once]". The summary reads `deferred files: present at their source paths on the
source machine (not a failure); correct: yes (8 checked: 7 test-derived + the raw Parquet table)`.

Default (before Gate 2c) mode on the laptop: **FAIL as designed**. 107 failures:
- 51 records missing from quarantine;
- 48 stream-dependent records at their live paths;
- 8 deferred files present, among them "RAW TABLE (contains test events) PRESENT BEFORE GATE 2c at its live path …
  (identical to the laptop file (size + SHA-256; SHA-256 of 2604833654 B computed once))".

It prints `deferred files absent: no (8 checked: 7 test-derived + the raw Parquet table)`. The information lines are
the other particles' `test_v3_nll.npz` files (another size).

Tests: `uv run python -B -m pytest -p no:cacheprovider -q shape-form-martin/tests/test_transfer_package.py` gives
26 passed. The tests cover:
- the four lists, the flags, the totals per group, list and phase, `policies.raw_table`, and the `holds` texts;
- that the real tables mark exactly 7 test-derived files plus the Parquet (`contains_test_events`), and that the
  group-c origin entry exists;
- default verify passing with all deferred files absent;
- default verify failing for each of the 4 synthetic deferred files at its live path, in quarantine or elsewhere
  (12 cases);
- the raw-table rules:
  - same name with another size, or another name: information;
  - any content at the live path: failure;
  - identical copies: failure, with exactly one SHA-256 per file and the "computed once" note;
- the name-and-size search for test-derived files;
- `--after-gate2c` needing all deferred files, the Parquet included;
- the four `sha256sum -c` one-liners and the `holds` lines;
- corruption, live-path quarantine and source-machine mode;
- `*.csv` refusal, including the walk;
- the flag-exclusivity refusals;
- the audit hook (the test cache is opened only by `sha256_file`);
- no torch or numpy import.

## 6. Unchanged from the first packaging (still valid)

- Caches are copied, not rebuilt. The train/val cache and the meta file travel now. The test cache and the Parquet
  source travel after Gate 2c, so before Gate 2c no rebuild is possible on the H200.
- Tracked laptop records are hazards on the H200. They are listed in `groups.a.laptop_record_hazards`.
- 29 machine constants, each line checked by `list`.
- Leave `SFM_DATA_DIR`, `SFM_RESULTS_DIR` and `SFM_CACHE_DIR` unset.
- **Run-book on the H200:**
  1. `git clone` / `git pull`;
  2. the two "now" rsync commands of `MANIFEST.json` `transfer_commands`;
  3. `python3 -B shape-form-martin/scripts/make_transfer_package.py verify` (must say `deferred files absent: yes
     (8 checked: 7 test-derived + the raw Parquet table)`).

  After Gate 2c:
  1. the two `test_derived_*` rsync commands (they include the Parquet file);
  2. `verify --after-gate2c`.

## 7. Not verified

- **Code that was not run:** fits, evaluations, guard recomputation and resumes (static reading only);
  `skip_init=False` builds; code that does not exist yet (re-scoring, validation-only parity, `run_m2a_report.py`,
  the AUDIT 4.2 drivers, the θ arms including the geometry arm and its origin file, `final_test.py`).
- **Campaign suite:** beyond `test_data.py` it was collected but not run, because it contains tiny fits and GPU
  evaluations.
- **Simulation limits:** the absence was simulated on the laptop. strace covered C-level opens of the run that was
  made; code paths that were not exercised are covered only by reading the code.
- **`check_protected.py verify`:** the H200 outcome is derived from `checksums.json` and `git ls-files`; the script
  was not run, because it writes a tracked file.
- **Origin file size:** the size quoted for the shipped origin file is an estimate.
- **Anything about the H200 itself:**
  - grid 1056;
  - CUDA 13 driver;
  - gcc with OpenMP;
  - AVX-512;
  - flock;
  - Python 3.12;
  - MIG layout;
  - that `rsync -a` keeps mtimes;
  - disk space;
  - Philox layout;
  - the predicted re-scoring values.
