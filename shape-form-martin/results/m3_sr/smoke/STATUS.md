# M3 PySR infrastructure: checkpoint status (paused 25 Nov 2026, ~00:20 CET)

## 1. Julia bootstrap: DONE

| Item | Value |
|---|---|
| Trigger | `import pysr` in a fresh `uv run python` process, `PYTHON_JULIACALL_THREADS=8`, `JULIA_NUM_PRECOMPILE_TASKS=8`. Full log: `bootstrap_log.txt` (this directory) |
| Julia | **1.13.4** (juliapkg picked the newest release allowed by juliacall `^1.10.3` and PySR `=1.10.0, 1.10.3`), at `.venv/julia_env/pyjuliapkg/install/` |
| Julia project | `.venv/julia_env/Project.toml` + `Manifest.toml` (1.1 GB total under `.venv/julia_env/`, including the Julia binary) |
| Julia depot | `~/.julia` (272 MB: registries 7.6 MB, packages 23 MB, artifacts 87 MB, compiled 156 MB) |
| Packages | SymbolicRegression **v2.7.2** (PySR 2.7.3 pins `~2.7.2`), DynamicExpressions v2.13.0, PythonCall v0.9.36, Optim v2.2.2, LossFunctions v1.2.2, DynamicQuantities v1.14.0, SpecialFunctions v2.9.2 |
| Time | First `import pysr`: **125 s** in total (download of Julia (287 MB), registry, packages, precompile; SymbolicRegression precompile 26.5 s). After that: `import pysr` 2.9 s, first trivial fit 2.4 s |
| Trivial fit | PASS: `2.5*cos(x0) + x1^2` recovered (`(cos(x0) * 2.5) + (x1 * x1)`, loss 2.6e-14), 8 threads |
| Threads | `PYTHON_JULIACALL_THREADS=N` before juliacall starts gives `Threads.nthreads() == N` (checked: 8). PySR then sets `JULIA_NUM_GC_THREADS = N//2,1` (8 threads gives `ngcthreads() == 5`). `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` is required (PySR's cooperative Ctrl-C depends on it). PySR warns when `PYTHON_JULIACALL_THREADS != auto`; the warning is harmless and `search._pysr()` silences it |
| torch | `import pysr` does **not** import torch (checked in `sys.modules`) |

**Problem found and fixed: the PythonCall extension would not precompile.** The extension
`SymbolicRegressionPythonCallExt` failed to precompile with the message `Python library ".../libpython3.12.so.1.0" could not be opened` / `Could not find Python library`.
The cause: glibc 2.44 refuses `dlopen` of the uv-managed `libpython3.12.so.1.0`, which is flagged as needing an executable stack (`cannot enable executable stack as shared object requires: Invalid argument`).
This only affects the separate Julia process that precompiles the extension. Inside the Python process libpython is already loaded.
The fix: one precompile run with `GLIBC_TUNABLES=glibc.rtld.execstack=2` (11.7 s). Later processes load the cached extension without errors and do not need the tunable.
If the cache is ever invalidated (for example by a package update), the error comes back once. Re-run the same command:
`GLIBC_TUNABLES=glibc.rtld.execstack=2 uv run python -c "import pysr"`.

## 2. Code status

| File | State |
|---|---|
| `src/sfm/sr/search.py` | Written, imports cleanly (no torch, no pysr, no juliacall at import), pyflakes clean. Spec parsing (JSON/TOML), chunk loop with `warm_start`, `progress.json` (atomic), HOF snapshots, `hall_of_fame.json` export (template sub-expressions through a Julia helper), resume via `from_file`, guesses fallback, salvage of a killed chunk's in-flight HOF as guesses, extension, done marker, loss-plateau rule, val-NLL hook (`record_val_nll_check`/`plateau_status`), status without Julia, seed lock (flock), Julia output redirected to `search.log` with evals/s parsed. **Not yet exercised against Julia** |
| `scripts/sr_run.py` | Written. `--help` and `--status` work without Julia. Loads the vocabulary from `[operators] module = "sfm.sr.operators", attr = ...` (a dict, an object with `pysr_kwargs()`, or a callable) and checks that torch did not come in |
| `tests/test_sr_search.py` | 15 Julia-free tests pass. 2 Julia tests are skipped by default: one short search plus resume/extend (needs `SFM_RUN_JULIA_TESTS=1`, never run yet), and a placeholder for the smoke suite, skipped with a reason |
| `src/sfm/sr/__init__.py` | Not created by me. The directory works as a namespace package without it. The math author owns it |

## 3. Smoke tests a–f: NOT STARTED

None of a–f has been run. No toy data and no specs have been written yet. The only Julia run so far is the trivial fit above.

Planned design (for when work resumes), all synthetic, at most 8 threads:
- **a** kill/resume on a toy 2-D target with noise: 1.5 min budget, 20 s chunks. SIGKILL during chunk 2, then resume. Check that the checkpoint holds the end of chunk 1: load it with `from_file` and compare with the chunk001 snapshot. Check that the resumed chunk's `hof_not_worse_than_prev_chunk` is True. Then a SIGINT clean stop and resume, an extension with `--budget-min`, and a kill with `salvage_aborted = true`.
- **b+c+f** in one production-like run: 12×650 = 7800 rows, inputs z, r, x, maxsize 30, the Stage-1 vocabulary as Julia strings with constraints, nested constraints and complexities, and the within-slice weighted-variance `loss_function` (slice ends passed through the `{{npz:slice_ends}}` placeholder). Scan the HOF for nesting and complexity, recompute the loss in numpy, and read evals/s at 8 threads.
- **c2** batching probes (about 15 s each): the 3-argument loss with `batching=true`, and a 4-argument `(tree, dataset, options, idx)` variant.
- **d** template `tanh(u(z, x) + v(r, x) + w(z, r))` with custom operators and `loss_function_expression`, run as 1 chunk, then an extension in a new process (tests `from_file` for templates).
- **e** multi-output y (N, 3), weights (N, 3), `guesses` as `list[list[str]]`. Compare wall time with `timeout_in_seconds` (shared or per-output budget), and check the per-output CSVs and the extension.

## 4. PySR 2.7.3 / SymbolicRegression.jl 2.7.2 facts so far (from the installed source)

- `checkpoint.pkl` is written at the start of a **fresh** `fit` and at the end of every `fit`. With `warm_start=True` and a Julia state present, nothing is written at the start. So during chunk k+1 the file holds the end of chunk k, as the plan assumed. The write is atomic (`.tmp` followed by `os.replace`).
- `timeout_in_seconds` counts from the start of the main search loop (`_main_search_loop!`). Initialisation and compilation come first, so a chunk's wall time is init + timeout + teardown. The runner counts **wall time of the `fit` calls** against the budget.
- `niterations` applies to each `fit` call. The runner sets 1e8, so the timeout ends every chunk.
- Warm start fails only if one of these changes: operators, op_constraints, nested_constraints, complexity_mapping, dimensionless_constants_only, maxsize, maxdepth, populations, population_size, node_type, expression_type, expression_options. Seed, timeout and loss may change. The runner therefore freezes `populations` (default 3 × threads) in `spec_snapshot.json`, so `--threads` can change on resume.
- With warm start, the hall-of-fame losses are **recomputed** on the current data, and `guesses` **are still applied**: they are parsed, constant-optimised, inserted into the HOF and migrated. This is what makes the salvage and fallback paths possible.
- The HOF CSVs (`hall_of_fame.csv`, or `hall_of_fame_output{1..n}.csv` for multi-output) list only the dominating Pareto front. Each is written twice (`.csv` and `.csv.bak`). Equation strings use `print_precision`, which defaults to 5 digits (lossy), so the runner defaults to 15. Float32 losses print as `0.25f0`.
- `loss_function` receives the **inner tree** for template expressions. Templates therefore need `loss_function_expression` (the runner enforces this).
- With batching, a 3-argument full objective receives a random row subset as a `SubDataset`, so row order is lost. If a 4-argument method `(tree, full_dataset, options, idx)` exists, it gets the full data and the batch indices. Expectation (still to be checked in c2): batching must be off for the contiguous-slice loss unless the 4-argument method is written.
- In free-form runs, custom Julia operators need `extra_sympy_mappings` (PySR raises otherwise). These are lambdas, which are not pickled, so the runner passes them again to `from_file`. The spec gives them as strings (`operators.sympy_mappings`). Templates do not need them.
- Template checkpoints rebuild the equation columns from the Julia state, which needs the custom operators defined first. The runner defines them in `Main` before `from_file`. Whether `expression_options` (the template's combine closure) still compares equal after deserialisation in a new process is **unknown**. Smoke d decides this; the guesses fallback is the safety net.
- Graceful stop: SIGINT makes PySR stop the running chunk, return partial results and checkpoint (`interrupted_`). Only the SIGINT byte triggers this. The runner maps SIGTERM and a `STOP` file to "stop after the current chunk".

## 5. Commands to continue

```bash
# from the repo root
uv run pytest -q shape-form-martin/tests/test_sr_search.py                              # 15 pass, 2 skip
SFM_RUN_JULIA_TESTS=1 uv run pytest -q shape-form-martin/tests/test_sr_search.py -k julia_short   # first live test (~1-2 min, 4 threads)

# run / resume / extend / status for a spec (seed dirs under results/m3_sr/<step>/<target>/seed<k>/)
uv run python shape-form-martin/scripts/sr_run.py SPEC.toml --seeds 0 --threads 8      # run, or resume
uv run python shape-form-martin/scripts/sr_run.py SPEC.toml --budget-min 30            # extend
uv run python shape-form-martin/scripts/sr_run.py SPEC.toml --status                   # no Julia
```

Next steps: run the live pytest above and fix whatever it shows. Then write the toy data, the specs and the orchestration script under `results/m3_sr/smoke/`, and run smoke tests a–f as designed in §3.
