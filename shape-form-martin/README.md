# shape-form-martin

Pion shower-shape campaign: can statistical distributions, PySR with an audited operator set, or other
leakage-free changes beat the two pion benchmarks (fine-tuned 4×32 MLP; 101-parameter component formula) under the
original report metric? The plan is `PLAN.md` in this directory (section 13 holds the amendments after Gate 0); gate
reports are in `results/gates/`. This directory holds all new code and results. `shape-form-original/` and
`shape-form-proton/` are read-only.

## Numbers of record (PLAN.md section 13, amendment A6)

Unmodified `common.test_nll` (K=64, CUDA generator seed 12345) **on this machine**:

| | validation | test |
|---|---|---|
| MLP (`mlp_std.pt`) | **18.117857** | **18.120806** |
| component formula (`new3_std_n3.pt` × `abar.pt`) | **18.120061** | **18.123333** |
| formula − MLP (paired) | **+0.002204 ± 0.000293** | **+0.002527 ± 0.000304** |

The values stored by the earlier study (MLP 18.118084 / 18.120842, formula 18.120407 / 18.123459) are quoted as
"reference machine, other fold points": the CUDA random stream of `torch.rand` depends on the GPU's launch grid
(Gate 0; `sfm.refstream` reproduces the reference machine's stream, grid 1056). Every campaign delta is paired
against the locally recomputed per-event arrays `results/m0_parity/{mlp,formula}_val_nll.npy`. Open item C1 (exact
per-event test parity on the reference stream) is `scripts/parity.py --reference-stream`; whether and when it runs is
the user's decision.

## Layout

```
common.py                 metric of record (moved unchanged from shape-form-original/, see below)
PLAN.md                   research and engineering plan (section 13: amendments after Gate 0)
05_mlp_pion_copy.ipynb    pion MLP notebook; its settings cell reads sfm.paths (Parquet data, results/ output)
src/sfm/
  paths.py                repo-relative locations; env overrides SFM_DATA_DIR, SFM_RESULTS_DIR, SFM_CACHE_DIR
  data.py                 lean pyarrow cache (same events/cuts/dtypes/split as common.load_data, plus event_number,
                          theta, phi); load_sets(); test split only with allow_test=True + reason (logged)
  legacy.py               read-only adapter to shape-form-proton (mlp_std.pt, component formula, abar.pt)
  metric.py               generalised NLL (K, stratified folds, bulk restriction, parts, mass monitors, K scans);
                          equals common.test_nll at defaults (bit for bit inside metric.deterministic())
  stats.py                paired delta ± SE, 1-SE rule, bootstrap, winner's curse, the plan's decision rules
  refstream.py            CPU emulation of the CUDA stream of torch.rand for any launch grid (_philox_emul.c,
                          compiled on first use into cache/); test_nll_ref(model, data, grid=1056)
  train.py                resumable fit (FitConfig, fit), run-config loader (load_run_config)
  registry.py             evaluate(): the only writer of results/registry.jsonl; table(), select_1se()
  gpulock.py              exclusive flock on cache/gpu.lock (one heavy GPU job at a time)
  models/                 paramnet.py (ParamNet), field.py (FieldNet, FieldAcc, warm starts), ensemble.py,
                          table.py (ResidualTable); factory registry used by scripts/fit.py
  families/               closed-form families (Track A)
configs/                  TOML run files (configs/examples/: templates, not pre-registered runs; configs/smoke/)
scripts/                  build_cache, parity, check_protected, mc_noise_val, fit, fit_queue, audit_*
tests/
results/m0_parity/        Gate-0 parity outputs (JSON tracked; *.npy / *.pt stay local)
results/gates/            gate reports
results/registry.jsonl    one JSON line per registered validation evaluation (append-only, sfm.registry only)
results/<stage>/<name>/   one fit: checkpoint.pt, model.pt, fit_done.json, fit.log, val_nll.npy, done.json
results/test_access.log   one line per test-cache access; tracked (re-included by the negation rule in .gitignore)
cache/                    git-ignored: data cache, val fingerprint, compiled emulator, gpu.lock
```

## Running (from the repo root, Python through uv, no bytecode)

```
uv run python -B shape-form-martin/scripts/check_protected.py record   # once, before touching anything
uv run python -B shape-form-martin/scripts/build_cache.py              # ~1 min: cache + bit-exactness checks
uv run python -B shape-form-martin/scripts/parity.py                   # Gate-0 parity (done; re-run needs --overwrite)
uv run python -B shape-form-martin/scripts/mc_noise_val.py             # validation-only Monte-Carlo noise baseline
uv run python -B shape-form-martin/scripts/check_protected.py verify   # protected directories unchanged
PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider shape-form-martin/tests
```

The data come only from `shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet`,
read column by column; the CSV is never opened. Nothing under `shape-form-proton/` is written: `sfm.legacy` sets
`sys.dont_write_bytecode` before importing from it and never calls `shapelab.setup*` or `protonlab.load_*`.

### A fit

A run file has four tables (unknown keys are errors):

```toml
[run]
name = "b2_k64_seed0"         # output: results/<stage>/<name>/
stage = "m2b_mlp"
seed = 0                      # seeds random/numpy/torch before the model is built and for the fit
notes = "..."                 # optional, not hashed

[model]
factory = "field_mlp_std"     # sfm.models factory: field, field_mlp_std, field_acc, residual_table, ensemble, from_run
                              # further keys are the factory's keyword arguments

[fit]                         # sfm.train.FitConfig (name, seed, out_dir come from [run])
steps = 3000
lr = 1e-3
lr_end = 1e-4                 # cosine decay lr -> lr_end over steps
batch = 256
k_train = 16                  # fold points per hit in the loss; fold = "iid" | "stratified"
k_val = 64                    # validation monitor (always i.i.d., seed 12345, like common.test_nll)
eval_every = 250
selection = "last"            # or "best_val"
ema_decay = 0.999             # optional; EMA weights evaluated and saved separately
smooth = 0.0                  # common.fit's depth-curvature penalty
weight_decay = 0.0            # L2 on weight matrices only
ckpt_every = 250

[evaluate]                    # optional
registry = true
guards = false                # K=1024, bulk stratum zb >= 1221 mm, mass monitors
C = 13.0                      # optional operator complexity (closed forms / SR)
```

```
uv run python -B shape-form-martin/scripts/fit.py shape-form-martin/configs/<stage>/<name>.toml [--dry-run]
uv run python -B shape-form-martin/scripts/fit_queue.py shape-form-martin/configs/<stage>       # serial queue
uv run python -B shape-form-martin/scripts/fit_queue.py a.toml b.toml @list.txt --dry-run
```

`fit.py` takes the GPU lock (`cache/gpu.lock`; `cat` it to see the holder), loads train and validation (never
test), builds the model, runs `sfm.train.fit` and registers the result. An interrupted fit resumes from
`checkpoint.pt` when started again (bit for bit on CPU; on CUDA inside `deterministic = true`, otherwise up to the
accumulation order of `index_add_`); a finished one is loaded. `fit_queue.py` runs configs one after another as
separate processes and skips those with `done.json` (it is not called `queue.py`, which would shadow the standard
library module `queue` for every script in `scripts/`). In Python:

```python
from sfm import data, models, registry, train
sets, extras = data.load_sets(("train", "val"))
train.seed_everything(0)
model = models.build("field_mlp_std", models.BuildContext(train=sets["train"]))
cfg = train.FitConfig(steps=3000, lr=1e-3, lr_end=1e-4, name="b0_seed0", out_dir="shape-form-martin/results/m2b_mlp/b0_seed0")
res = train.fit(model, sets["train"], sets["val"], cfg)
```

### The registry

```python
val = registry.load_validation()                            # refuses anything but the cached validation split
row = registry.evaluate(model, "b0_seed0", "m2b_mlp", val, meta={"seed": 0, "checkpoint": res.model_path})
registry.table(stage="m2b_mlp")                             # DataFrame: val mean, delta ± SE vs mlp / formula, guards
registry.select_1se(stage="m2b_mlp")                        # sfm.stats.one_se_select over the chosen rows
```

`evaluate` checks the event numbers (in order) and a SHA-256 fingerprint of every tensor against the cached
validation split, scores the model with the unmodified `common.test_nll` inside `metric.deterministic()`, stores
`results/<stage>/<name>/val_nll.npy` (never silently overwritten), pairs it against the Gate-0 arrays with
`sfm.stats.paired`, and appends one line to `results/registry.jsonl` under an exclusive lock (k, C, seed, config
hash, checkpoint SHA-256, git commit, timestamp). Guard numbers have their own columns.

## Provenance of `common.py`

`common.py` is the importable version of `shape-form-original/00_common.ipynb` (added locally on 23 Nov 2026;
see `summary.md`, section 7). It was moved here with `git mv` in Milestone 0 without any change (SHA-256
`ece3c67f89cd9f4792ccc0e50f3eea342180db72d0b55dfcfd3863c096dbebb3`, checked by `check_protected.py verify`).
`event_nll` / `test_nll` (K=64, CUDA generator seed 12345) are the metric of record and stay unmodified.
