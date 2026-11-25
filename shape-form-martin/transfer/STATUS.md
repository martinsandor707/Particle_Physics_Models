# H200 transfer package: status (P4 implemented: test-derived files split off; PLAN.md not updated)

Packaging finished on 25 Nov 2026; amended the same day for the user's decision P4 ("do not transfer the seven
test-derived files now; split the file list and copy them to the H200 only after Gate 2c, so physical absence proves
zero test access"). PLAN.md section 16 (which still describes the single-phase package), the commit and the push are
the orchestrator's. The statistics review is in `REVIEW_STATUS.md` (not touched here).

**Produced (untracked or modified until the orchestrator commits them, explicit paths only):**
- `shape-form-martin/scripts/make_transfer_package.py`: `list` (laptop) and `verify` (H200, two phases), standard
  library only, never imports torch or numpy, refuses any `*.csv`, treats the test cache as an opaque byte stream.
- `shape-form-martin/tests/test_transfer_package.py`: 22 tests on synthetic temp repositories, all pass.
- `shape-form-martin/transfer/{files_b_inputs.txt, files_b_laptop_records.txt, files_b_test_derived_inputs.txt,
  files_b_test_derived_records.txt, SHA256SUMS, MANIFEST.json}`, written by `list` at commit `fcd223a`. The two
  `files_b_test_derived_*.txt` are new files. The manifest records `tracked_tree_dirty: true`: the script, its test
  and this file differ from `fcd223a`, and so does `PLAN.md`, which another process modified in the working tree
  during this work (not this worker). `MANIFEST.json` holds every field of every file; this file only summarises it.

**Guardrails kept.** No training, fit or registry write. The test cache was opened only by the byte hash in
`sha256_file` (also shown by the test's audit hook). No CSV was listed, stat'ed or hashed (strace of `verify`,
including the new tree walk: no file syscall names a `.csv`). Nothing under `shape-form-proton/` or
`shape-form-original/` was modified. `parity.py`, `build_cache.py` and `check_protected.py` were not run. The real
test-derived files were never moved, renamed or deleted; their absence was simulated by an audit hook (section 4).

## 1. The P4 split (exact; from `MANIFEST.json`)

| list | group | phase | destination root on the H200 | files | bytes |
|---|---|---|---|---|---|
| `files_b_inputs.txt` | b1 | now | `<repo>/` | 20 | 2,993,885,926 |
| `files_b_laptop_records.txt` | b2 | now | `<repo>/shape-form-martin/cache/laptop_records/` | 51 | 5,749,702 |
| `files_b_test_derived_inputs.txt` | b1 | after Gate 2c | `<repo>/` | 3 | 69,197,473 |
| `files_b_test_derived_records.txt` | b2 | after Gate 2c | `<repo>/shape-form-martin/cache/laptop_records/` | 4 | 239,808 |

| totals | files | bytes |
|---|---|---|
| group a (tracked; arrives with git) | 129 (was 127: the two new list files) | about 2.6 MB (snapshot at generation; exact value in `MANIFEST.json`, not checked by `verify`) |
| group b1 | 23 | 3,063,083,399 (now 20 / 2,993,885,926; after Gate 2c 3 / 69,197,473) |
| group b2 | 55 | 5,989,510 (now 51 / 5,749,702; after Gate 2c 4 / 239,808) |
| phase now (b1 + b2) | 71 | 2,999,635,628 |
| phase after Gate 2c (b1 + b2) | 7 | 69,437,281 |
| group c | 17 entries | regenerated on the H200 |

The seven test-derived files:
- inputs, copied after Gate 2c to the same path: `shape-form-martin/cache/pion_s1_test.pt` (68,476,893 B),
  `shape-form-proton/results/particles/pion/test_v3_nll.npz` (600,804 B),
  `shape-form-martin/results/m0_parity/test_event_number.npy` (119,776 B);
- laptop records, copied after Gate 2c into the quarantine root: `results/m0_parity/{mlp,formula}_test_nll.npy` and
  `{mlp,formula}_test_nll_ref1056.npy` (59,952 B each).

Every b-file in `MANIFEST.json` now has `test_derived` (bool), `transfer_phase` (`now` / `after_gate2c`), `list` and,
where its name or content mentions the test split, a `test_note`. New manifest keys: `lists` (per list: totals,
rsync, check one-liner), `totals_by_phase` (top level and per group), the four rsync commands in
`transfer_commands` (plus `phases`), `policies.test_derived` (rule and reason in one paragraph),
`policies.test_derived_files`, `policies.test_summary_metadata_now`, `policies.test_events_in_raw_parquet`, and
`absence_before_gate2c` (consequences, each line checked by `list`). Schema 2.

`SHA256SUMS`: four delimited sections (`# BEGIN <list>` / `# END <list>`), each headed by its phase, destination root,
rsync command and `sha256sum -c` one-liner. The two "now" one-liners were run on the laptop's live files from the
repo root (20 and 51 OK); in the tests all four run on a synthetic H200 copy.

## 2. Did any other b1/b2 file carry test-event information? (search result)

Checked against the tables, the code readers and the files' own contents (JSON read, `.pt` pickles disassembled
with `pickletools` without unpickling, `.npz` member names listed; the test cache and the CSV were not opened):

- **No other file holds per-event or per-hit information derived from test events.**
- **The Parquet raw table contains the test events.** It holds the hits of all 99,689 events; the split is made only
  at cache build (`sfm/data.py:150-160`, seeded per momentum bin), and `build_cache` writes a test cache too
  (`data.py:227`). It is not test-derived and travels now (geometry arm), but it means that the absence of the seven
  files shows that none of the *stored* test-derived files was used, not that the test events are unreachable on the
  H200. Recorded in `policies.test_events_in_raw_parquet`: before Gate 2c never run `build_cache.py` on the H200 (not
  even into scratch), and any code that reads the Parquet file must keep to the train/val event numbers. The
  "optional cross-check rebuild" of the cache decision is now marked "only after Gate 2c".
- `pion_s1_trainval.pt`: train and val packs only. Its geometry constants are computed from all rows
  (`data.py:82-86`), and its per-event `index` is the position in the full event list, so the complement locates the
  test events. That is membership only, with no hits and no scores.
- `pion_s1_meta.json` (travels now, stated in the manifest): `counts.test`, `files.test` (SHA-256 and size of the test
  cache) and `splits.test`. The last is aggregate statistics of the test split: events, hits, extremes of p, theta,
  zb, z_local, r and evis, hits per event min/median/max, and events per momentum bin. No event identities, no
  per-event arrays.
- `test_v3.json` (travels now, stated in the manifest): the formula definition (variant, variant_kwargs, degrees,
  params), which `shapelab.final_round/final_kwargs` read on every legacy formula load. Also the earlier study's
  summary test numbers: n_events, 10 test means, paired differences with SE, gap-closed fractions, and two 20-entry
  per-momentum-bin mean differences. No per-event values.
- `mlp_std.json`: validation only (`val` 18.118084, `val_report_weights`).
- `new3_std.json`: validation only. Per-rung params and validation means, `n_star`, `chosen`, and `diff_vs_best`
  (paired difference and SE on validation, `shape-form-proton/scripts/particle_pipeline.py:121-126`).
- Checkpoints (`mlp_std.pt`, `new3_std_n3.pt`, the 8 `model.pt`) and templates (`abar.pt`, `abar_front.pt`,
  `abar_front_back.pt`): no test strings in their pickles. The templates are computed from the training split
  (`parity.py:494-497`, `particle_pipeline.py` abar).
- b2: every other array is on validation. The `val_mass_monitors.npz` members are validation functionals.

`list` now refuses a b-entry whose file name contains the word "test" unless the entry is `test_derived` or carries
a `test_note`. It also refuses a test cache that is not `test_derived`.

## 3. `verify` (two phases + source machine)

- **Default = before Gate 2c.** Checks:
  - every "now" file at its destination, by size and SHA-256 (b1 at live paths, b2 under the quarantine root);
  - all the existing checks (quarantine rule for stream-dependent records at live paths, group a presence, commit
    ancestry, hazards list, `--skip-large-hash`, `--json`).

  It **FAILS if any of the seven exists on this machine**, at any of these places:
  - its live path (any content);
  - under the quarantine root (any content);
  - elsewhere in the working tree with the same file name and size.

  The tree search is a walk by name. It skips `.git`, does not follow symlinked directories, and skips `*.csv` names
  before any stat. A size match is confirmed by SHA-256; for the test cache that means a size compare, then a byte
  hash only if the sizes match. A same-name, same-size file with another hash is a warning; a same-name file of
  another size is an info line. Each offending path is printed. The summary prints
  `test-derived files absent: yes/no (7 checked)`.
- **`--after-gate2c`** additionally requires the seven present with correct size and SHA-256 at their destinations.
  The test cache is checked by byte hash only, with the sentence that it stays unopened as data until the final
  evaluation. The summary prints `test-derived files present and verified (after Gate 2c): yes/no (7 checked)`.
- **`--source-machine`** (laptop): everything is checked at the live source paths. The test-derived files are present
  by definition, and that is not a failure.

## 4. Consequences of the absence before Gate 2c (also in `MANIFEST.json` `absence_before_gate2c`)

Laptop check, read-only, on the CPU. An audit hook raised `FileNotFoundError` on any open, and a wrapper of
`os.stat`/`os.lstat` on any stat, of the seven paths, so they looked absent. Writes under the repo and `*.csv` opens
were refused. The harness is in the session scratchpad (`absence/hide_trace.py`, `absence/hideplug.py`).

What was exercised:
- imports of all `sfm` modules;
- `load_sets(("train","val"))` and `load_meta`;
- `legacy.load_mlp_std`, `load_component_formula` (`shapelab.build_final`), `load_abar` and
  `families.load_legacy_core`;
- the registry read paths: `validation_reference` (fingerprint `731a1997…`), `check_validation`, `read_registry`,
  `select_1se`, the benchmark arrays and `_bench_guard_arrays` (reuse path);
- the templates and the emulator library;
- Track A spec lists (66) and builds with `skip_init=True`;
- the Track B factories and `load_run` of the 8 networks;
- `fit.py --dry-run` (s2_seed13, b0_seed1) and `fit_queue.py --dry-run`;
- the `run_m2b_fit.py` config start-up;
- import and `--help` of `run_m2a.py`, `run_m2b_fit.py`, `run_m2b_ensembles.py` and `run_m2b_stage1.py`, and
  import of `run_m2b_report.py`;
- `pytest --collect-only` of all 445 campaign tests;
- a run of `tests/test_data.py` (5 passed).

**Result: zero opens or stats of any of the seven.** The only hits came from the harness's own visibility probe.
Every phase behaved as with the files present; the build failures seen (A2n, A9, `full_box`) are the known missing
Track A outputs.

Consequences, each with file:line (none changes a moved step):
- `sfm/data.py:286-287`: `load_sets(("test",), allow_test=True, reason=…)` writes the test-access log line **before**
  opening the cache. Before Gate 2c the call fails at `:287` after logging, so on the H200 such a line marks an
  attempted access, not a read. The locked calls raise earlier: `:283` without `allow_test`, `:285` with an empty
  reason.
- `scripts/parity.py:316` (C1): with `--reference-stream --overwrite`, `FileNotFoundError` on
  `test_event_number.npy`. That is before `load_sets(test)` at `:318`, so there is no log line, but only after
  `:314` has written `{mlp,formula}_val_nll_ref1056.npy` at their live paths. Without `--overwrite`, `:300` refuses
  first.
- `scripts/parity.py:368` (Gate 0 default mode, refused without `--overwrite` at `:342`): with `--overwrite` it
  writes a log line and then raises `FileNotFoundError`, before `test_v3_nll.npz` at `:407`. `parity.py` must not run
  on the H200 before Gate 2c in any case.
- `scripts/check_protected.py:152`: `verify` already fails on the H200 (exit 1), for two reasons:
  - 1805 of the 1812 git-ignored protected files in `checksums.json` are never copied;
  - the CSV in `stat_only` is missing.

  Before Gate 2c, `test_v3_nll.npz` is one more "removed" entry (1806); the exit code is 1 either way. It also
  rewrites the tracked `results/m0_parity/checksums_verify.json` (`:155`).
- Gate 0 record scripts (not moved) raise `FileNotFoundError` on the H200:
  - `c05_split_integrity.py:102`;
  - `c08_parity_stats.py:110`, `:117-118`;
  - `c08b_pair_tail.py:15-16`;
  - `c14_stats_review.py:119-121`;
  - `c16_test_order_fingerprint.py:15`, `:19`.
- `sfm/data.py:168`: `cache_files()["test"]` is a path only, and no code branches on its existence (grep).
- Tests: none reads `test_v3_nll.npz`, `test_event_number.npy` or the test cache.
  `tests/test_data.py:34` (`test_test_split_is_locked`) passes, because it raises at `data.py:283/285` before any
  open.

## 5. Script outputs (real run on the laptop)

`list` (commit `fcd223a`, tracked tree dirty; last run after this file was written):
- group a: 129 files, about 2.6 MB (snapshot);
- group b1: 23 files, 3,063,083,399 B (now 20 / 2,993,885,926; after Gate 2c 3 / 69,197,473);
- group b2: 55 files, 5,989,510 B (now 51 / 5,749,702; after Gate 2c 4 / 239,808);
- group c: 17 entries;
- phase now: 71 files, 2,999,635,628 B; phase after Gate 2c: 7 files, 69,437,281 B.

All cross-checks are equal, and all cited lines (29 constants, 7 absence consequences) were confirmed. Run time
1.8 s (warm page cache).

`verify --source-machine` on the laptop: **PASS**. 80 checks, 0 failures, 2 warnings: the two new list files are not
committed yet. Summary line: `test-derived files: present at their source paths on the source machine (not a
failure); correct: yes (7 checked)`.

Default (before Gate 2c) mode on the laptop: **FAIL as designed**. 108 failures:
- 51 records missing from quarantine;
- 48 stream-dependent records at their live paths, each printed with its `mv`;
- 7 test-derived files present at their live paths, each "identical to the laptop file";
- 2 new package files not committed.

It also gives 3 warnings (the mass monitors) and prints `test-derived files absent: no (7 checked)`. The tree search
also found `shape-form-proton/results/particles/{electron,muon,proton}/test_v3_nll.npz` (other particles; same name,
other size): info lines only. These are not in the package.

Tests: `uv run python -B -m pytest -p no:cacheprovider -q shape-form-martin/tests/test_transfer_package.py` gives
22 passed. The tests cover:
- the four lists and the manifest fields;
- that the real tables mark exactly the seven files;
- default verify passing with the seven absent;
- default verify failing for each of the 3 synthetic test-derived files at its live path, in quarantine, or as a
  copy elsewhere (9 cases);
- the name-and-size search (other size: info; same size, other hash: warning; identical: failure; `.git` ignored);
- `--after-gate2c` failing until all are present and correct;
- the four `sha256sum -c` one-liners;
- corruption, live-path quarantine and source-machine mode;
- `*.csv` refusal, including the tree walk;
- the audit hook: the test cache is only ever opened by `sha256_file` (binary read), never when absent or when a
  same-name file differs in size;
- no torch or numpy import.

## 6. Unchanged from the first packaging (still valid)

- **Trace of the moved steps' reads:** the hand-over's reads are reproduced; everything they read is in b1 or c.
  `fit_done.json` makes `train.fit` return the laptop network. The `test_registry.py` real-data test fails, rather
  than skips, without `val_event_number.npy`.
- **Caches are copied, not rebuilt:**
  - `build_cache.py` has no argument parser and always rewrites both caches, the meta file and the tracked
    `cache_build.json`;
  - a rebuild on another CPU is not guaranteed bit-identical;
  - only a byte copy carries Gate 0 and C1 over.

  The train/val cache and the meta file travel now; the test cache travels after Gate 2c.
- **Tracked laptop records are hazards on the H200** (`registry.jsonl`, `test_access.log`, `guard_spec.json`, the
  `done.json`/`fit_done.json` markers, and others). They are listed with file:line in `MANIFEST.json`
  `groups.a.laptop_record_hazards`, and `verify` prints them.
- **Machine constants:** 29 in `MANIFEST.json` `machine_constants`, each line checked by `list`.
- **Path variables:** leave `SFM_DATA_DIR`, `SFM_RESULTS_DIR` and `SFM_CACHE_DIR` unset; `verify` checks the
  default paths.
- **Run-book on the H200:**
  1. `git clone` / `git pull`;
  2. the two "now" rsync commands of `MANIFEST.json` `transfer_commands`;
  3. `python3 -B shape-form-martin/scripts/make_transfer_package.py verify` (must say
     `test-derived files absent: yes (7 checked)`).

  After Gate 2c:
  1. the two `test_derived_*` rsync commands;
  2. `verify --after-gate2c`.

## 7. Not verified

- **Code that was not run:** reads that only happen inside fits, evaluations, guard recomputation and resumes
  (static reading only); `skip_init=False` builds; code that does not exist yet (re-scoring, validation-only parity,
  `run_m2a_report.py`, the AUDIT 4.2 drivers, the θ arms, `final_test.py`). The campaign suite beyond `test_data.py`
  was collected but not run, because it contains tiny fits and GPU evaluations.
- **Simulation limits:** the absence was simulated on the laptop by an audit hook and a stat wrapper. Code that
  reached the files through C-level paths, without Python's `open`/`os.stat`, would not have been caught; none is
  known to read them.
- **`check_protected.py verify`:** its H200 outcome is derived from `checksums.json` and `git ls-files`; the script
  was not run, because it writes a tracked file.
- **Anything about the H200 itself:**
  - grid 1056;
  - CUDA 13 driver;
  - gcc with OpenMP;
  - AVX-512 on the host;
  - flock on its filesystem;
  - Python 3.12;
  - MIG or multi-GPU layout;
  - that `rsync -a` keeps mtimes;
  - disk space;
  - Philox layout;
  - the predicted re-scoring values.
