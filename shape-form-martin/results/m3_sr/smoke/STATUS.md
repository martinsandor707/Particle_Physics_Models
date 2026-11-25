# M3 PySR runner: live smoke tests a–f (25 Nov 2026, complete)

Julia 1.13.4, PySR 2.7.3, SymbolicRegression.jl 2.7.2, 8 Julia threads unless stated. Synthetic targets only
(`make_data.py`); no real data, no `*.csv` data file, no GPU, no torch in any search process. Total wall time of all
search processes in this session ≈ 18 min (the longest single run 3.6 min), one 77 s run at 20 threads. The Julia
bootstrap of the first session is in §6 (unchanged).

## 1. Results

| Test | What | Verdict | Evidence |
|---|---|---|---|
| a | SIGKILL mid-chunk and resume; SIGINT stop and resume; extension; fallback restart from the saved hall of fame | **PASS** (11 of 11 checks) | `evidence/a.json`, `evidence/a_console.txt`, `evidence/a_resume.json` |
| b | Stage-1 vocabulary + gauge-invariant loss from `operators.search_config` accepted; 7800 rows, 3 inputs, maxsize 30 | **PASS** | `evidence/b_stage1_gauge.json`, `evidence/b_console.txt` |
| c | post-processing (`scripts/sr_post.py`, separate process): scans, complexity, loss recomputation, round trip, gauge de-duplication, plateau | **PASS** on all six run directories | `evidence/*.json` → `post` |
| d | template `tanh(u(zt,x) + v(rh,x) + w(zt,rh))` with custom operators and `loss_function_expression`; extension in a new process (`from_file` on a template checkpoint); template fallback | **PASS** | `evidence/d_template.json`, `evidence/d_template_console.txt` |
| e | multi-output: weights (40, 3), guesses `list[list[str]]`, three halls of fame, extension in a new process | **PASS** | `evidence/e_multiout.json`, `evidence/e_multiout_console.txt` |
| f | thread setting takes effect; scaling 8 → 20 threads | **PASS**: `nthreads` = 20, ×1.86–1.96 evaluations/s | `evidence/f_threads8.json`, `evidence/f_threads20.json` |
| — | search process torch-free | **PASS** in every run (`torch_in_process: false`, recorded by the search process itself from `sys.modules` at start and after every chunk; plus a Julia-free unit test of all search-side imports) | progress.json of each run |

### d — template (PASS)
`smoke_def.py d` (run 74 s, extension 52 s, post 1.3 s) + a fallback probe (44 s). Spec
`[search_config] kwargs = {stage = "stage1", ndim = 3, loss = "gauge", template = "s1_tanh_sum"}` on `residual3d.npz`.
- PySR accepted the template with the Stage-1 custom operators inside the sub-expressions and the gauge loss as
  `loss_function_expression`; fronts like `u = tanh(#1 * l1p((#2 + #1) + 0.139…)); v = -1.0025…; w = l1p(erf(#2 * 0.0356…))`.
- **`from_file` on a template checkpoint works in a new process** (the open question of the first session): restore
  1.2 s, populations and hall of fame identical to the end of chunk 0 (`same_as_chunk_end = true`), front not worse.
- Template fallback: checkpoint truncated, extension to 2.1 min → restart seeded with the 11 members as sub-expression
  dicts (`{"u": …, "v": …, "w": …}`), accepted by PySR, front not worse than before.
- Post: 12 members, round trip 2.3e-16 (float32 torch 1.6e-7), loss 6.3e-15, complexity (sum over sub-expressions)
  equal for all, 0 violations (scanned per sub-expression).
- Overheads: 23.5 s first chunk, 18.3 s after `from_file`, 28 s for the template fallback; 7.5–1.1e4 evaluations/s.
- Bug found and fixed here: the Julia helper that exports the sub-expressions referenced
  `SymbolicRegression.DynamicExpressions` (not reachable in 2.7.2) and printed `x1`; it now uses the exported
  `get_contents` / `get_metadata` and `#k` names, and its output equals PySR's own CSV split for 11 of 11 members.
  Before the fix the template fallback would have had no guesses; `_guesses_from_export` now falls back to PySR's
  `u = …; v = …` string as well.

### e — multi-output (PASS)
`smoke_def.py e` (run 54 s, extension 44 s, post 1.6 s). `traj3.npz`: 40 points, outputs `0.3 + 0.5x − 0.2x² + 0.1x³`,
`0.8 tanh(1.5x)`, `exp(0.5x) − 1` (+ noise), weights (40, 3), `guesses` = three Horner cubics from
`targets.cubic_guesses` (`list[list[str]]`), `[search_config] kwargs = {stage = "stage1", ndim = 1, loss = "wl2"}`
(maxsize 18).
- Accepted; three files `hall_of_fame_output{1,2,3}.csv`, snapshotted per chunk; 14 / 12 / 13 members; best losses
  2.0e-6 / 3.6e-6 / 3.8e-6 at C ≤ 18. Extension in a new process: `from_file`, identical state, fronts not worse.
- Post (per output): round trip ≤ 4.4e-16 (float32 ≤ 2.2e-7), loss (weighted L2 with the output's weight column)
  ≤ 1.2e-14, complexities equal.
- **The scan dropped members live**: output 3 (`exp(0.5x) − 1`) had 4 of 13 members of the form `ppow(c, x)`, e.g.
  `ppow(-0.648…, x) + -1.00004…` = (1 + |c|)^x — an exponential with the exponent a *variable*, admitted by
  `constraints["ppow"] = (7, 1)` and removed by `operators.violations`; that output's front shrinks from 13 to 9.

### f — threads (PASS)
`smoke_def.py f`: `f_threads8.toml` and `f_threads20.toml`, identical except `threads` (populations fixed at 24), one
60 s chunk each on `residual3d.npz` with the Stage-1 gauge setup.
- `PYTHON_JULIACALL_THREADS` takes effect: `Threads.nthreads()` = 8 / 20, GC threads 5 / 11.
- Evaluation rate (Julia's 5-second prints): 8 threads mean 16 900/s (16 500 after the first two prints), 20 threads
  31 400/s (32 400) → **×1.86 (×1.96)** for ×2.5 threads, measured while other workers used the laptop. With
  production's 3 × threads = 60 populations the 20-thread figure may be higher (not measured).
- Wall 80 s / 77 s for a 60 s timeout (compilation 15.8 s / 13.6 s).

### a — kill, resume, SIGINT, extension, fallback (PASS)
`uv run python -B shape-form-martin/results/m3_sr/smoke/smoke_a.py` (216 s wall, 5 search processes). Toy 2-D target
(`toy2d.npz`, operators `+ - * tanh exp square`, maxsize 25, PySR's default L2), 8 threads, 24 populations,
20 s chunks, budget 2.0 min, extended to 2.5 and 2.9 min. All checks are computed by the driver from `progress.json`
and file times (`evidence/a.json` → `checks`):

| Step | Result |
|---|---|
| 1 fresh start, **SIGKILL** 8 s into chunk 2 | exit −9 with 2 chunks complete; `checkpoint.pkl` last written 8.1 s before the kill, i.e. before chunk 2 began (= end of chunk 1); the run directory's `hall_of_fame.csv` was 7.6 s newer (in-flight front) |
| 2 resume | `from_file` in 0.09 s; **population and hall-of-fame hashes of the restored state equal those recorded at the end of chunk 1** (`same_as_chunk_end = true`: 24 populations × 27 members, 25 hall-of-fame entries); the killed chunk was detected (≈ 8 s lost), its in-flight front snapshotted (`hof_snapshots/aborted002_*`) and fed back as 25 guesses; the first resumed chunk's front is not worse than chunk 1's at any complexity; no "Recreating population" warning in Julia's log |
| 2 **SIGINT** 5 s into the next chunk | PySR stopped the chunk after 5.2 s (`interrupted_ = true`), checkpointed; the runner stopped with `stop_reason = sigint`, exit 0, status `stopped` (resumable) |
| 3 resume to the end | restored state identical to the interrupted chunk's end; DONE after 116 s of 120 s |
| 4 extension `--budget-min 2.5` | same seed continues (`from_file`, identical state), `DONE.json` superseded and rewritten, front not worse |
| 5 fallback: `checkpoint.pkl` truncated to a third, extension to 2.9 min | `UnpicklingError` → checkpoint moved to `checkpoint.pkl.broken-*`, restart seeded with the 24 members of `hall_of_fame.json` as guesses; the fallback chunk's front is not worse than the last chunk before at any complexity |

A first attempt with a 1.5 min budget (`evidence/a_run1_budget1.5min*.json/txt`) passed 10 of 11 checks: after the
kill-and-resume the budget was used up, so there was no chunk left to interrupt (a budget choice of the test, not a
runner fault). It also exercised the fallback path after the fix below.

Measured costs (toy target, 400 rows, 8 threads; `fit_wall_s − timeout`):

| Situation | Overhead per chunk |
|---|---|
| first chunk in a fresh process (compilation) | 8.3 s (b: 26.9 s with the custom loss and operators) |
| next chunk in the same process (`warm_start`) | **0.6 s** (b: 5.2 s at 7800 rows) |
| first chunk after `from_file` in a new process | 9.0 s; 13.8 s with 25 salvaged guesses |
| fallback restart (fresh populations + 24 guesses) | 15.5–17.0 s |
| process start outside `fit` (Python, `import pysr` 2.3 s, restore 0.1 s, exports) | ≈ 4.5 s |
| export after a chunk (state hashes, predictions, hall_of_fame.json) | 0.5 s first, then 0.02–0.04 s |

So with 5-minute chunks the warm-start cost is negligible (< 2 %), a restart costs ≈ 15–30 s, and a SIGKILL loses
at most the chunk in flight (≤ 5 min of search).

### b — Stage-1 vocabulary and gauge loss (PASS)
`uv run python -B shape-form-martin/scripts/sr_run.py shape-form-martin/results/m3_sr/smoke/specs/b_stage1_gauge.toml`
(116 s wall). Spec: `[search_config] kwargs = {stage = "stage1", ndim = 3, loss = "gauge"}`; data `residual3d.npz`
(12 slices × 650 rows, slice-contiguous, a gauge term `0.7 x + 0.2` added to y), budget 2 min, chunks 40 s.
- All five custom operators were accepted and used (`ppow`, `softplus`, `l1p`, `sqa`, `linv` appear in the fronts),
  together with the constraints, nested constraints and operator complexities of `STAGE1`, Float64,
  `print_precision = 17`, `batching = false`.
- Front after 112 s: 19 members; best loss at C ≤ 12 / 20 / 30: 4.60e-3 / 1.11e-3 / 8.39e-4 (null loss 0.136).
- Chunk 0: timeout 40 s, wall 66.9 s (26.9 s overhead = first compilation of the custom loss and operators);
  chunk 1: wall 45.2 s (5.2 s overhead). Evaluation rate 1.0–2.2e4 expressions/s at 8 threads, 7800 rows (14 prints).
- Julia import 2.3 s; export after a chunk (state deserialisation, state hashes, predictions of 20 members on 7800
  rows) 0.59 s for the first chunk, 0.04 s afterwards.
- Gauge freedom seen live: the C = 1 member is a bare constant (`-1.3367…`, pure gauge), the C = 27 member carries a
  free additive constant (`+ -0.342…`).

### c — post-processing on b's output (PASS)
`uv run python -B shape-form-martin/scripts/sr_post.py shape-form-martin/results/m3_sr/smoke/specs/b_stage1_gauge.toml --float32` (1.4 s).
- Round trip against the search's own predictions (Julia `eval_tree_array` on the end-of-chunk member trees,
  full constant precision, all 7800 rows): max |torch − Julia| = **4.4e-16** in float64 over 19 members (rule
  ≤ 1e-5), 2.7e-7 with the torch side in float32.
- Loss recomputed in NumPy (`operators.gauge_loss_numpy` on `translate.evaluate` of the 17-digit strings) equals the
  CSV loss to 6.4e-15 (relative); `operators.tree_complexity` equals PySR's complexity for all 19 members; the losses
  stored in the Julia state equal the CSV losses exactly.
- Scan (`operators.violations`): 0 of 19 members violate. The two deliberately non-compliant guesses of the spec
  (`ppow(zt, rh)`, `tanh(tanh(·))`) did not survive into any saved front (dominated), so the *dropping* path is
  exercised by the unit tests (`tests/test_sr_post.py`) and not by live output here.
- Gauge de-duplication (`audit.drop_gauge_terms`): 2 members carry pure-x terms; the C = 27 member is effectively
  C = 25 and displaces the listed C = 25 member, so the front after the scan and de-duplication has 18 members.
  Tier bests unchanged here. Plateau (gauge-aware) needs 4 chunks; b has 2. On a's 7 chunks the plateau over the
  snapshots is computed (`plateau = false`, C ≤ 15 / 25 improved 18 % / 22 % over the last 3 chunks) and
  `search.plateau_status` reports it with `loss_rule_source = sr_post`.
- In f_threads20 the C = 1 member is the bare constant `187.806…`: under the gauge loss a constant is free and
  drifts. In float64 harmless; in float32 its rounding alone is 7.4e-6 (close to the 1e-5 rule). Gauge terms should be
  dropped (`audit.drop_gauge_terms`) before anything is handed to a refit.
- sr_post was also run on every other run directory (a, d, e, f8, f20): all PASS; worst float64 round trip over all
  138 members 8.9e-16.

## 2. Code changes in this session (write set only)

`src/sfm/sr/search.py`
- **`[search_config]`** table → `resolve_search_config`: calls `operators.search_config(**kwargs)` (or another
  module/function) and routes *every* returned key — operator settings → `operators`, loss strings → `loss`,
  `template` → `template` (+ `name`), `maxsize`, `precision` / `print_precision` / `batching` /
  `complexity_of_constants` / `complexity_of_variables` → `[pysr]`. An unknown key, or operators / loss / template given
  twice, or a different maxsize, is refused. The interface was checked end to end (unit test for three
  configurations; b, d, e, f use it live).
- `print_precision` now follows `precision` (17 for Float64, 9 for Float32) unless the spec sets it; was 15.
- **Julia output now reaches `search.log`.** Bug found: Julia (via juliacall) writes to its own duplicate of fd 1 / 2,
  so the old `os.dup2` redirect captured nothing (no evaluation rates were recorded in the first live test). Now
  redirected inside Julia (`redirect_stdout/stderr` + a `SimpleLogger` for `@info`/`@warn`); verified with stdout
  as a file and as a pipe.
- **Prediction export** after every chunk: `predictions/chunkNNN_<UTC>.npz` = the search's own prediction of every
  front member (Julia `eval_tree_array` on the member trees of the end-of-chunk state) on the target rows (all rows
  up to `prediction_max_rows` = 20000, else an even subset, indices stored), with complexities, CSV and state losses
  and equation strings; referenced from `hall_of_fame.json` (`predictions`).
- **State summary** per chunk and per restore: number of populations / members / hall-of-fame entries and SHA-256 of
  all member strings; the restore records `same_as_chunk_end` (exact check that the resumed search starts from the
  saved populations and hall of fame).
- All Julia-side indexing of the search state moved into Julia helpers (the old Python-side `state[1]` was an
  indexing ambiguity through juliacall).
- **Bug fixed:** after a failed restore the fallback model's guesses (the saved hall of fame) were overwritten with
  `None` by the first chunk's `set_params(guesses=...)`, i.e. the fallback restarted from scratch.
- **Bug fixed:** template sub-expression export (see d); template fallback guesses also from PySR's own string.
- `torch_in_process` recorded by the search process (`"torch" in sys.modules`) at start and after every chunk;
  per-chunk `fit_overhead_s`, `export_s`; per start `julia_import_s`; per restore `restore_s`.
- `plateau_status` uses `post/plateau_gauge.json` (written by `sr_post.py`) when it is current, else the raw
  runner value, and says which (`loss_rule_source`).

`scripts/sr_run.py`: resolves `[search_config]`; the `[operators] module = …` path now refuses non-operator keys
(it used to drop `loss`, `maxsize`, `precision`, … from `search_config` with only a note).

`scripts/sr_post.py` (new, separate torch-side process): translation, scans, complexity, loss recomputation, round
trip, gauge de-duplication, gauge-aware plateau over all chunk snapshots; writes `post/post_chunkNNN.json` and
`post/plateau_gauge.json`; exit status 1 on a failed round trip / loss / complexity check. Complexities are counted
with the search's own weights (the spec's `complexity_of_operators`, PySR's default 1 otherwise); a first version
used the operators-module table and wrongly failed a's hand-written vocabulary.

`results/m3_sr/smoke/`: `make_data.py` (synthetic targets), `specs/*.toml`, `smoke_a.py` (kill / SIGINT / extension /
fallback driver with machine-checked verdicts), `smoke_def.py` (d, e, f), `collect.py` (condenses a run directory into
`evidence/<target>.json` and replaces the absolute repository path by `<repo>` in every evidence file).

How to rerun everything (≈ 12 min, 8 threads except f20):
```bash
uv run python -B shape-form-martin/results/m3_sr/smoke/make_data.py
uv run python -B shape-form-martin/scripts/sr_run.py shape-form-martin/results/m3_sr/smoke/specs/b_stage1_gauge.toml
uv run python -B shape-form-martin/scripts/sr_post.py shape-form-martin/results/m3_sr/smoke/specs/b_stage1_gauge.toml --float32
uv run python -B shape-form-martin/results/m3_sr/smoke/smoke_a.py
uv run python -B shape-form-martin/results/m3_sr/smoke/smoke_def.py d e f
uv run python -B shape-form-martin/results/m3_sr/smoke/collect.py
```

## 3. Tests

- `uv run pytest -q shape-form-martin/tests/test_sr_search.py shape-form-martin/tests/test_sr_post.py`:
  27 passed, 1 skipped (the live one), ≈ 2 s. New Julia-free tests: search_config routing for three configurations,
  refusal of unknown keys / conflicts, print_precision default, `[operators]` refusing a full search config, torch-free
  imports of everything the search process imports (fresh subprocess), the post-processing on a fake seed directory
  (scan drops `ppow(zt, rh)` and `tanh(tanh(·))`, gauge duplicate removed, round trip passes and detects a 1e-3 error,
  template translation, gauge-aware plateau).
- Live: `SFM_RUN_JULIA_TESTS=1 uv run pytest -q shape-form-martin/tests/test_sr_search.py -k test_julia_`
  → 1 passed in **66 s** (4 threads): two search processes (1 chunk, then an extension that must restore the identical
  state, front not worse, `torch_in_process` false in both, evaluation rate captured) and `sr_post.py` (round trip
  passes). Selection is by the environment variable plus the name prefix `test_julia_`; without the variable the test
  is skipped. No pytest marker is registered because `tests/conftest.py` is outside this write set — the orchestrator
  may add `config.addinivalue_line("markers", "julia: ...")` there if a marker is preferred.
- Whole `sr_` suite (this worker's and the mathematics worker's tests): `uv run pytest -q shape-form-martin/tests -k sr_`
  → 303 passed, 1 skipped, 15.6 s. (One of my tests had to move into a subprocess: `load_vocabulary` refuses to run
  once torch is imported, which other test modules do in the same pytest process.)

## 4. Files under results/m3_sr/smoke/

Small, worth committing (≈ 0.2 MB together): `STATUS.md`, `make_data.py`, `smoke_a.py`, `smoke_def.py`,
`collect.py`, `specs/*.toml`, `evidence/*.json`, `evidence/*_console.txt`, `bootstrap_log.txt`. The evidence files
contain no absolute paths, host names or keys (`collect.py` rewrites the repository path to `<repo>`).
Bulky or regenerable, keep untracked: `runs/` (13 MB: checkpoints 0.1–0.8 MB each, `predictions/*.npz` ≈ 1.3 MB per
chunk at 7800 rows, `search.log`; it also holds absolute paths in `spec_snapshot.json`), `data/` (0.4 MB, regenerated
deterministically by `make_data.py`).

## 5. What Gate 2-pilot can rely on, what is not verified, problems found

**Can rely on (measured live, PySR 2.7.3 / SR.jl 2.7.2 / Julia 1.13.4):**
- `operators.search_config` → runner interface: every key routed, nothing dropped; Stage-1 custom operators, the
  three constraint dictionaries, Float64, `print_precision = 17`, the gauge loss (free form and as
  `loss_function_expression` for templates) and the weighted L2 are accepted.
- Chunked search with `warm_start`: in-process chunk overhead 0.6 s (toy) to 5 s (7800 rows); `checkpoint.pkl`
  during chunk k+1 = end of chunk k; resume / extension via `from_file` restores *identical* populations and hall of
  fame (hash check) for free-form, template and multi-output searches; SIGKILL loses only the chunk in flight (its
  front is salvaged as guesses); SIGINT stops a chunk within seconds and leaves a resumable state; a broken checkpoint
  falls back to a restart seeded with the saved front (free form and template), front not worse.
- Per-output halls of fame with weights (N, n_out) and `list[list[str]]` guesses.
- Thread setting effective (`nthreads` = requested); 20 threads ≈ 1.9× the throughput of 8 on a shared laptop.
- Round trip: the translator reproduces Julia's own evaluation of the saved member trees to ≤ 8.9e-16 (float64) on
  138 live members across all four losses/forms; NumPy loss recomputation equals PySR's to ≤ 1.2e-14; complexities equal.
- Hall-of-fame scan works on live output and removes `ppow(·, variable)` members; gauge de-duplication and the
  gauge-aware plateau are computed from the snapshots.
- No search process imported torch.

**Not verified:**
- Anything on a real target; convergence per complexity tier (the pilot's purpose); 5-minute chunks (smoke chunks
  were 12–60 s); long runs (memory growth, log size).
- A *nesting* violation in live output: none appeared in ~140 live members (consistent with PySR enforcing
  `nested_constraints`); the dropping of nesting violations is tested only on hand-made members. Whether PySR filters
  non-compliant `guesses` is unknown (b's two deliberately non-compliant guesses were not in any saved front).
- The batching probes (c2 of the first plan): not run; `search_config` sets `batching = false` for the gauge loss and
  the runner warns if a spec turns batching on with a full-objective loss.
- The multi-output fallback restart (list[list] guesses from the export) — only the multi-output `from_file` path ran.
- SIGTERM and the `STOP` file (stop after the current chunk), several seeds in one process, two runners on one seed
  (lock), changing `--threads` on resume (populations are frozen in the snapshot, so it should work).
- A kill between the end of `fit` (checkpoint written) and the progress update (≈ 0.1–0.6 s window): the next start
  would treat the finished chunk as aborted and resume from a checkpoint that is *newer* than recorded (harmless for
  the search; `same_as_chunk_end` would then be false and the chunk's time uncounted).
- Budget accounting counts the wall time of `fit` including compilation (15–27 s for the first chunk of a process), so
  a 15-minute pilot loses ≈ 0.5 min per process start to compilation.
- Thread scaling with 60 populations (production default 3 × threads); the measurement used 24 populations under load.

**Problems found in other modules (none edited; no bug found in operators / translate / audit / targets / checks /
refit):**
1. *For the mathematics worker / Gate 2-pilot (D3 consequence, now measured):* on the trajectory-like target, PySR
   expressed `exp(0.5x) − 1` through `ppow(c, x)` = (1 + |c|)^x; 4 of 13 members of that output violated "exponent of
   `ppow` a constant" and were dropped, leaving holes in the front (13 → 9 members). This is the rule as written, but
   it removes a legitimate positive-base exponential; whether to keep it, or to make it strict inside the search
   (`complexity_of_variables`-type trick, or a unary `ppow`-with-constant), is a decision for the gate.
2. *Gauge constants drift:* with the gauge loss a bare constant member reached 187.8 (f20). The audit's oscillation
   gauge and `drop_gauge_terms` handle it; a float32 consumer would see rounding ≈ 1e-5 from such a term alone.
3. *Ad-hoc vocabularies in `operators.violations`:* the argument-size check counts sizes with the operators-module
   weights (`Vocabulary.complexity_of_operators()`), not with a spec's own `complexity_of_operators`. Exact for the
   `STAGE1/STAGE1_RAW/STAGE2` vocabularies (which is what the campaign uses); approximate for hand-written specs.
4. `translate.round_trip` compares absolute differences; fine for the O(1) targets here, but for a target with large
   values a relative rule may be wanted (not changed).
5. Interface note: `operators.search_config` returns `extra_sympy_mappings` as callables (not serialisable). The runner
   copes (re-created at every resume, recorded as `<callable>` in `spec_snapshot.json`); no change needed.

## 6. Julia bootstrap (first session, unchanged)

Julia 1.13.4 in `.venv/julia_env/pyjuliapkg/install/`, depot `~/.julia` (272 MB). First `import pysr` 125 s
(download and precompile), afterwards 2.3–2.9 s. The PythonCall extension needed one precompile run with
`GLIBC_TUNABLES=glibc.rtld.execstack=2` (glibc 2.44 refuses to `dlopen` the uv-managed libpython that asks for an
executable stack); if the cache is ever invalidated: `GLIBC_TUNABLES=glibc.rtld.execstack=2 uv run python -c "import pysr"`.
`PYTHON_JULIACALL_THREADS=N` gives `Threads.nthreads() == N`; `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` is required.
Full log: `bootstrap_log.txt`.
