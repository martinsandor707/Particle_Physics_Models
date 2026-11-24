# shape-form-martin

Pion shower-shape campaign: can statistical distributions, PySR with an audited operator set, or other
leakage-free changes beat the two pion benchmarks (fine-tuned 4×32 MLP, test NLL 18.1208; 101-parameter
component formula, 18.1235) under the original report metric? The plan and its gates live outside the repo;
this directory holds all new code and results. `shape-form-original/` and `shape-form-proton/` are read-only.

## Layout

```
common.py                 metric of record (moved unchanged from shape-form-original/, see below)
05_mlp_pion_copy.ipynb    pion MLP notebook; its settings cell reads sfm.paths (Parquet data, results/ output)
src/sfm/
  paths.py                repo-relative locations; env overrides SFM_DATA_DIR, SFM_RESULTS_DIR, SFM_CACHE_DIR
  data.py                 lean pyarrow cache (same events/cuts/dtypes/split as common.load_data, plus event_number,
                          theta, phi); load_sets(); test split only with allow_test=True + reason (logged)
  legacy.py               read-only adapter to shape-form-proton (mlp_std.pt, component formula, abar.pt)
configs/  scripts/  tests/
results/m0_parity/        Gate-0 parity outputs (JSON tracked; *.npy / *.pt stay local)
results/gates/            gate reports
results/test_access.log   one line per test-cache access (note: ignored by the repo-wide *.log rule)
cache/                    git-ignored data cache
```

## Running (from the repo root, Python through uv, no bytecode)

```
uv run python -B shape-form-martin/scripts/check_protected.py record   # once, before touching anything
uv run python -B shape-form-martin/scripts/build_cache.py              # ~1 min: cache + bit-exactness checks
uv run python -B shape-form-martin/scripts/parity.py                   # Gate-0 parity (the only test access)
uv run python -B shape-form-martin/scripts/mc_noise_val.py             # validation-only Monte-Carlo noise baseline
uv run python -B shape-form-martin/scripts/check_protected.py verify   # protected directories unchanged
PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider shape-form-martin/tests
```

The data come only from `shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet`,
read column by column; the CSV is never opened. Nothing under `shape-form-proton/` is written: `sfm.legacy` sets
`sys.dont_write_bytecode` before importing from it and never calls `shapelab.setup*` or `protonlab.load_*`.

## Provenance of `common.py`

`common.py` is the importable version of `shape-form-original/00_common.ipynb` (added locally on 23 Nov 2026;
see `summary.md`, section 7). It was moved here with `git mv` in Milestone 0 without any change (SHA-256
`ece3c67f89cd9f4792ccc0e50f3eea342180db72d0b55dfcfd3863c096dbebb3`, checked by `check_protected.py verify`).
`event_nll` / `test_nll` (K=64, CUDA generator seed 12345) are the metric of record and stay unmodified.
