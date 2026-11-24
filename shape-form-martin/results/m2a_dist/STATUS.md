# Track A (M2A) — status at the checkpoint of 25 Nov 2026

**No GPU job has been run yet.** Nothing has been trained, nothing is registered under `stage="m2a_dist"`, and no
checkpoint exists. The code is consistent and importable. The CPU tests pass.

## What is implemented and tested

| File | Content | State |
|---|---|---|
| `src/sfm/models/dist_factory.py` (new) | factories `component` and `mixture`, registered in `sfm.models.FACTORIES` on import; providers `poly` (start = copy by name from a finished run) and `net` (ParamNet + `fit_to`); initialisers `init_shower_long` and `init_from_target`; `kl_map` recorded in `model.init_info`; templates `back_corrected` / `full_box` (`acceptances` and `exit_depth` ported from `results/m1_audit/checks/c_side.py`, cited in the docstring); the `LastRing` wrapper for A12 | done; CPU-tested |
| `tests/test_dist_factory.py` (new) | all 29 model structures of the list build on the CPU, are normalised (\|logsumexp\| < 1e-5), have finite gradients, and have the pre-registered constant counts: A0 101, A1 699, A2e 101, A2s 101, A2n 699, A4 93, A5 101, A7 103, A8 102, A9 104, A10 74, A4t 89, A5h 97, A6i/A6l 97, A6b 98, A7s/A7l/A8c 102, A78 104, A9r 98, A3* 101, A10-2/4/5 48/100/126, A10g 98, A12 102. Also tested: net versions of A4/A7/A9/A10; `legacy_centre` = the legacy formula's map; the stored template = the legacy logA; A7/A8/A78/A7s/A8c start exactly on A2s (A7l within KL 6e-5); the LastRing wrapper; copy by name; the degree override; the geometry port at normal incidence; `exit_depth` | **40 passed, 3 skipped**. The skipped tests are A3c, A3x and the template-range test; they need the templates file, which has not been computed yet |
| `scripts/run_m2a_lib.py` (new) | `Spec`; T_A `fit_config` (weight decay 0 for poly, 1e-4 for net; lr from `plr.json`); `run_spec`, which does: under `gpu_lock` with a free-memory check → build → `init.json` → optional step-0 registry row → `sfm.train.fit` (resumable) → `registry.evaluate` (with a duplicate check by checkpoint SHA-256) → `done.json`; GPU-time log `gpu_time.jsonl` | importable; **not yet exercised on the GPU** |
| `scripts/run_m2a.py` (new) | phases `templates`, `parity`, `plr`, `stage0`, `stage1p` (with the convergence reruns), `stage1x` (seed 1 on keep/equal-NLL; budget check with the drop order) | importable; run list verified by listing (below); **not yet run** |
| `scripts/run_m2a_report.py` | rules, guards (G-K / G-bulk / G-mass through `registry.evaluate(guards=True)`, plus G-region from `metric.event_nll_multi(return_parts=True)`), secondary columns, Fisher information of k for A6b, Stage 2, RESULTS.md | **not written yet**. The `guards` and `stage2` phases of `run_m2a.py` import it lazily, so `run_m2a.py all` would fail at `guards`; run the first six phases explicitly |

Run names: Stage 0 `A0_s0/1, A1_s0/1, A2e_s0/1, A2s_s0/1/2, A2n_s0/1/2`; primaries `A4/A5/A7/A8/A9/A10_s0/_s1`, nets
`A4n…A10n_s0`; exploratory `<rung>_s0` (and `_s1` if passing); pilots `plr_A2s_lr{0.003,0.01,0.03}`,
`plr_A2n_lr{0.001,0.003,0.01}`; step-0 rows `A0_init`, `A9_s0_init`, `A9_s1_init`; convergence reruns `<name>_x2`.

## Execution order — state

| Step | State |
|---|---|
| 1. tests | **done** (CPU) |
| 2. step-0 parity of A0 (18.120061 to 1e-5) | implemented (`parity` phase); not run |
| 3. learning-rate pilot | implemented (`plr` phase); not run |
| 4. Stage 0 | implemented; not run |
| 5. primaries | implemented; not run |
| 6. exploratory | implemented; not run |
| 7. rules, guards, secondary columns | not written |
| 8. Stage 2 | not written |
| 9. RESULTS.md | not written |

## Decisions taken where the list was ambiguous (most literal reading)

1. **ParamNet names.** The audit puts the radial shape names into `names_const`, but they also belong to
   `model.momentum_dependent`. They are therefore removed from `names_net`. `logit_dh` (A9) stays a net output: it is
   in `momentum_dependent` and is not a radial shape name.
2. **ParamNet names for the mixture**, which has no legacy split: `names_net` = the rows of polynomial degree > 0;
   `names_const` = the `c{k}_lat_*` rows.
3. **Net runs with `net_from`** rebuild the whole fitted poly model first. Scalars outside the provider (DetHalo L, R
   in A9) therefore start at their fitted values.
4. **P-lr for A2n.** "fit_to A2s (same seed)" is read as fit_to the Stage-0 run `A2s_s0`. The order is: poly pilots →
   lr_poly frozen → `A2s_s0` (a Stage-0 run, executed early) → net pilots → lr_net frozen. The pilots are registered;
   the decision uses the registered validation means. Winner = the smallest lr within 0.5e-4 of the lowest.
5. **A0 drift rule.** The table's "|Δ| > 2: stop" is applied to the trained A0 − formula on both seeds.
6. **A0 step-0 parity** is registered once (`A0_init`); seeds 0 and 1 have the same initialisation.
7. **Convergence.** |val(last) − val(last − 1000)| < 0.5e-4 from the fit history. Doubled-step reruns are done only
   for **poly** primaries (the pre-registered comparisons): primary and A2s of the same seed, the same initialisation,
   steps doubled. The net column is exploratory, so non-converged nets are not rerun.
8. **A10 and its net/exploratory variants** (A10n, A10-2/4/5, A10g) use 6000 steps ("as A10"). For the mixtures,
   `init_from_target` uses A2s of the same seed as the target.
9. **Depth-family swaps** (A4t, A5h, A6i, A6l, A6b) use `init_shower_long` as A4/A5 do.
10. **A9r** `logit_dh` initial value = logit of the mean over the training events of σ(logit_far(p)) of A2s_s0. DetHalo
    L and R stay at their defaults (the list names only logit_dh).
11. **`kl_map`** for every run = mean over `default_lnp(48)` of KL(P_ref ‖ P_model). The reference is the run it was
    started / initialised / fitted from, or the legacy formula for runs that start from the legacy coefficients.
12. **Templates.** Training events. 1[z′ < zb_e] is averaged over the same two z sub-points per cell as A.
    `stored` uses the legacy checkpoint's logA (1-ulp identical to ln abar.pt).
13. **Stage-2 plan (not yet coded).** If nothing is kept, A11 = A2s (no rerun) and A11n = A2n (no rerun). A11k = A2s
    from the legacy coefficients with `k_train = 64` (seeds 0, 1), compared with A2s of the same seed. Guards will also
    be run on the comparators (A2s / A2n of the same seed), so that guard deltas against the base can be paired.

## Bugs and workarounds

- **`sfm.models.paramnet.ParamNet.fit_to`** (not patched). With `weights=None` it divides each target column by its
  variance whenever var > 0. A column that is constant in momentum has a float-rounding variance (~1e-15) and
  receives a weight of ~1e15. The other columns then go unfitted (rms errors ~1 on the shower parameters, NaN model
  gradients afterwards), and the L-BFGS polish raises "value cannot be converted to type float without overflow".
  In the run list this affects A9n (`logit_dh` is degree 0). Workaround in `dist_factory.make_paramnet`: when such a
  column exists (detected with fit_to's own float operations), a line-by-line replica of `fit_to` is used. Its only
  changes are a variance floor of 1e-12·max(1, mean²) and guarded L-BFGS chunks. Otherwise `fit_to` itself runs.
  `init_info["fit_to"]["fit_to_workaround"]` records which path was taken.
- `fit_to`'s reported `loss` includes the constant columns weighted by 1/var(rounding). This is cosmetic (e.g. 12.0
  for A7n at the legacy coefficients).

## Commands that continue from here (from the repo root)

```
uv run python -B -m pytest -q shape-form-martin/tests/test_dist_factory.py -p no:cacheprovider     # CPU, ~10 s
uv run python -B shape-form-martin/scripts/run_m2a.py templates parity plr stage0 stage1p stage1x
```

Every phase skips finished runs (`results/m2a_dist/<name>/done.json`) and resumes interrupted fits from
`checkpoint.pt`. The `parity` phase stops the track if A0 at step 0 differs from 18.120061 by more than 1e-5, and
`stage0` stops it if |A0 − formula| > 2e-4. Still to write before steps 7–9: `scripts/run_m2a_report.py` (rules,
guards with G-region, secondary columns, A6b Fisher information, Stage 2, RESULTS.md).
