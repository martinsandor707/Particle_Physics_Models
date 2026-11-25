# `sfm.sr` — status of the math-author modules (second session, 25 Nov 2026; updated after the user's review)

Written for whoever continues. All six modules are implemented; each section says what the code does, what was
verified and what was not. Section 8 holds the decisions and the plan rules they touch (**all eight accepted by
the user on 25 Nov 2026**); **section 12 answers the user's four confirmations** and lists what they added;
section 10 holds the interfaces for the runner. **Numbers are measured only where they say so.**

## 1. Status per module

| Module | State | Tests | Notes |
|---|---|---|---|
| `operators.py` | done (first session); `UNCONFIRMED` emptied, `CONFIRMED_WITH_JULIA` added; torch `sqa` / `softplus` gradients fixed (§12.1) | 70 pass | every item of `UNCONFIRMED` is now confirmed against Julia, see §9 |
| `translate.py` | **done** | 91 pass (`tests/test_sr_translate.py`, 1.3 s) | torch only on demand; checked against real PySR strings, predictions and losses |
| `audit.py` | **done** | 77 pass (`tests/test_sr_audit.py`, 9 s) | NumPy only; (a) domain, (b) magnitude, (c) numerics; `certify_finalist` |
| `checks.py` | **done** | 9 pass (`tests/test_sr_checks.py`) | NumPy only; tolerances are arguments with stated defaults |
| `targets.py` | **done**, not yet run on real data | 18 pass (`tests/test_sr_targets.py`) | constructions tested on synthetic maps; event-level estimators on synthetic showers (CPU child process) |
| `refit.py` | **done except registration** (stub, by instruction) | 11 pass (`tests/test_sr_refit.py`, CPU child process) | constants recovered on synthetic showers; no real refit run |
| `__init__.py` | empty, as `test_sr_package_init_is_empty` requires | | |

Run: `uv run pytest shape-form-martin/tests -q -k "sr_ and not search"` → 280 passed (about 18 s): 276 in the six
files of this table plus the 4 of another worker's `tests/test_sr_post.py`, which the same filter selects.

Nothing outside the write set was touched; no git state changed; the test split was not opened; no `*.csv` was
opened; no real data file was read at all (targets and refit were exercised on synthetic inputs only); the GPU was
not used (everything that imports `common.py` runs in child processes with `CUDA_VISIBLE_DEVICES=""`); no
likelihood of a candidate equation was computed; `sfm.registry` is not imported by any module here.
Read-only exception to "no real file": `shape-form-proton/results/particles/*/sr_residual.json` was read once to
copy the earlier study's four pion equations into the audit test.

## 2. `operators.py` — what is fixed (implemented)

**Variables.** `zt = z'/260 mm`, `rh = r/21.9 mm`, `x = (ln T − v0)/vs`. The name `zeta` is not usable: it is a sympy
function and PySR refuses such variable names (`rho`, `zt`, `rh`, `x` are free).

**Audit box.** `extended_box()` = `zt ∈ [−1.1538, 7.6923]`, `rh ∈ [0, 68.493]`, `x ∈ [−4.11, 1.33]`.
Measured: `x(0.0005 GeV) = −4.1030`, `x(50 GeV) = +1.2777` (not +1.33; `x = 1.33` is p = 59 GeV, and −1.3336 is
`x(0.05 GeV)`). The default is a superset of both. 1-D trajectory range `[−4.2, 1.4]`, training range `[−1, 1]`.

**Operators** (`OPS[name]`: Julia string, complexity, range, totality, smoothness):

| name | Julia string | C | range | remark |
|---|---|---|---|---|
| `+ − *` | built-in | 1 | ℝ | |
| `exp` | built-in | 2 | (0, ∞) | overflows above 709.78 (f64) / 88.72 (f32) |
| `square` | built-in | 1 | [0, ∞) | overflows above 1.34e154 / 1.84e19 |
| `tanh` | built-in | 2 | (−1, 1) | |
| `atan` | built-in | 2 | (−π/2, π/2) | |
| `erf` / `erfc` | built-in | 3 | (−1, 1) / (0, 2) | |
| `softplus` | `softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))` | 2 | (0, ∞) | never overflows; not `F.softplus` |
| `linv` | `linv(x) = 1 / (1 + x * x)` | 2 | (0, 1] | |
| `ppow` | `ppow(x, y) = (1 + abs(x))^y` | 2 | (0, ∞) | kink in x at 0; overflows when `y·log1p(|x|)` > 709.78 / 88.72 |
| `l1p` | `l1p(x) = log1p(abs(x))` | 2 | [0, ∞) | kink at 0 |
| `sqa` | `sqa(x) = sqrt(abs(x))` | 2 | [0, ∞) | infinite slope at 0; `torch_impl` defines the backward pass as 0 at exactly 0 (plain autograd gives NaN), forward unchanged |
| `sin`, `cos` (Stage 2) | built-in | 2 | [−1, 1] | |
| `/`, `inv` (extra arm) | built-in | 1 | ℝ | not total |
| `neg abs cube log log1p sqrt pow` | — | — | — | never searched; for parsing and auditing foreign equations |

Deviation from the plan's sentence "transcendental 2, erf/erfc 3": `sqa` and `linv` are algebraic but composite and
cost 2. Constants and variables cost 1 (a Horner cubic is C = 13, k = 4, as in the plan).

**`nested_constraints`** (Stage 1): each of `tanh, atan, erf, erfc, softplus, linv` → `{those six: 0, exp: 0,
ppow: 0}`; `exp → {exp: 0}`; `ppow, l1p, sqa, square` → `{itself: 0}`.
Stage 2 adds: every Stage-1 unary operator and `ppow` → `{sin: 0, cos: 0}` (trig combines through `+ − *` only);
`sin`, `cos` → `{sin: 0, cos: 0, exp: 0, ppow: 0, square: 0}`.
Extra arm adds: `inv → {inv: 0, /: 0}`, `/ → {/: 1, inv: 0}`.

**`constraints`** (argument size, weighted): `ppow (7, 1)`, `exp 9`, `square 7`, `tanh/atan/erf/erfc/softplus 13`,
`linv 9`, `l1p 7`, `sqa 7`; Stage 2 `sin 9`, `cos 9`; extra arm `/ (−1, 9)`, `inv 9`. These limits are my choice
(13 = an affine form in three variables); they are meant to be tuned at the pilot.

**`ppow` exponent.** PySR's size limit 1 admits one constant *or one variable*. "Exponent a constant" is therefore
enforced only by the scan `operators.violations(tree, vocabulary)`. Strict enforcement inside the search would need
`complexity_of_variables = 2`, which changes what size 18 / 30 mean — a decision for the pilot gate.

**Scans.** `violates_nesting` reproduces SymbolicRegression.jl's `flag_illegal_nests` / `count_max_nestedness`
(largest count of the inner operator on any path below an outer node, the node itself excluded), from memory of the
Julia source, not from a run. `violates_constraints`, `violations`, `tree_complexity`. Trees are duck-typed: objects
with `kind` ∈ {const, var, param, op}, `name`, `args`, or nested tuples `("tanh", ("+", "zt", 1.5))`.

**Losses.**
- `WL2_ELEMENTWISE = "sfm_wl2(prediction, target, weight) = weight * (prediction - target)^2"`; reference
  `wl2_loss_numpy = Σ w e² / Σ w`. Any row order; `weights` shaped like `y`.
- `gauge_loss_julia(x_index=3)`: within-slice weighted variance
  `L = Σ_s Σ_{i∈s} w_i (e_i − ē_s)² / Σ_i w_i`, `e = f(X) − y`. Invariant under `f → f + c(x)`. Layout: rows of a
  slice contiguous; a slice = a maximal run of equal values in feature row `x_index` of `dataset.X`; weights (N,);
  `batching=False`. Two passes (mean, then squares) so a large gauge term does not cancel in Float32. Same source
  for `loss_function` and `loss_function_expression`. Reference `gauge_loss_numpy`.

**Templates** (`TEMPLATES`, `TemplateSpec.combine(precision)`; literals become `0.3f0` for Float32):
- `s1_tanh_sum`: `tanh(u(zt, x) + v(rh, x) + w(zt, rh))`
- `s1_amp_shape`: `a(x) * f(zt, rh)`
- `s2_modulation` (plan, literal): `tanh(f(zt, rh, x)) + 0.3 * tanh(a(x)) * (1 / (1 + e(zt, rh)^2)) * sin(ph(zt, rh, x))`
- `s2_modulation_unit`: the same with `0.7 * tanh(f)`, so that |value| ≤ 1 by form.
Sub-expressions of the Stage-2 templates use the Stage-1 vocabulary, so the template's `sin` cannot be nested.

**`search_config(stage, ndim, loss, template, precision)`** returns the math-critical `PySRRegressor` keyword
arguments: operators, the three dictionaries, `maxsize` (18 / 30), loss string, `batching=False` for the gauge loss,
`precision=64`, `print_precision=17`. With `template=` it returns `cfg["template"]` = the keyword arguments of
`pysr.TemplateExpressionSpec` (the runner builds the object).

## 3. `translate.py` — implemented

- `Expr` (immutable; kinds const / var / param / op / sub; `fixed=True` = structural literal, not trained, not in k);
  `parse` (precedence climbing; PySR strings, Julia combine strings, the sympy-style strings of the earlier study);
  `parse_template(name, {sub: string})` and `parse_template_string(name, "u = ...; v = ...")`; `to_string` (PySR's own
  format, reproduces PySR's strings character by character); `to_dict` / `from_dict` (lossless, keeps `fixed`);
  `evaluate` (NumPy float64, optional record of the largest intermediate); `TorchExpr(expr, variable_names, dtype)`
  (all k free constants in one `nn.Parameter` `c`, order of `constants(expr)`; no clamp, no `nan_to_num`;
  `current_expr()` returns the tree with the present constants); `with_constants`; `round_trip(expr, X, names,
  reference, tol=1e-5)`; `horner_string` (cubic: C = 13, k = 4); `expr_hash`.
- Sign folding: `x * -1.5` has one constant; `-x^2 = -(x^2)`; `u^2` → `square(u)`; another literal integer exponent →
  `pow` with a fixed exponent (integer-power semantics); any other exponent → real power (base > 0, audit-enforced).
- **Verified against Julia** (fixture embedded in the test file: 24 rows, 7 runs, 26 equations, Float64 searches,
  `print_precision=17`; plus 2 equations with `ppow` and `softplus` on 10 rows): translation equals PySR's own prediction to ≤ 1e-12 (rule: 1e-5), in float32 torch to
  ≤ 1e-5; the recomputed loss equals the CSV loss to 1e-9 relative, for the weighted L2 and for the gauge loss, free
  form and all four templates; `T.complexity` equals PySR's complexity column; for templates the column is the sum
  over the sub-expressions. A Float32 template search (`0.3f0` literals) prints 8 digits and agrees to < 5e-6.
- Not verified: template *parameters* (`p[1]`; parsed as `param` nodes, nothing downstream uses them — our templates
  have none). A free constant exponent that happens to be an integer is re-read as a fixed integer power after
  `to_string` (use `to_dict` for hand-over).
- For the runner: the round-trip check of the plan needs the search's own predictions on the target rows; the runner
  should export `model.predict(X, index=i)` per hall-of-fame member into the seed directory (NumPy, no torch), and
  the check is `translate.round_trip(expr, X, names, reference=pred)`. Without predictions the loss comparison
  (`operators.gauge_loss_numpy` on `translate.evaluate`) is the fallback; both were exercised in the tests.

## 4. `audit.py` — implemented

`audit(expr_or_string, box=None, limit=1.0, gauge="oscillation")` → `AuditReport` (`passed`, `domain`, `magnitude`,
`to_json()`, `summary()`); `domain_safety`, `magnitude_bound`, `enclose`, `audit_trajectory`, `audit_box(x_floor=)`.
A template result is translated first (`translate.parse_template_string`), then audited like any tree.

**(a) Domain safety, sound.** Outward-rounded float64 interval arithmetic on every node (one ulp for + − * / sqrt, 64
ulp + one for library functions), intersected with each operator's range; forbidden sets: 0 for `/`, `inv`, negative
integer powers; (−∞, 0] for `log`, `sqrt`, real-power bases; (−∞, −1] for `log1p`; every node below
`OVERFLOW_CAP = 1e30`. Forbidden sets are dilated by `τ = 1e-5·max(1, largest magnitude in the argument's subtree)`
(float32 allowance; heuristic, more rejections only). Flagged boxes are bisected along the variables the offending
node depends on, to 2⁻¹⁰ of a side or 200k boxes. Status `safe` / `unsafe_proven` (a point inside the dilated
forbidden set or with an overflow; or a sign change of a denominator between corners of a box on which the
denominator is continuous) / `not_proven_safe`. **Only `safe` passes.**
- Verified: enclosures contain 600 random samples per case on several hundred random trees (700 drawn) over the
  *whole* vocabulary (Stage 1, trig, raw arm, audit-only operators) on random sub-boxes over three decades of size, and the derivative enclosures
  contain the autograd derivatives; each elementary enclosure is tight to 0.1 %; the four equations of the earlier
  study (`c/(r − 0.017)` and its three descendants) are rejected as `unsafe_proven` within milliseconds; a pole whose
  effect a 100001-point grid cannot see is rejected; 150 random poles inside the box are all rejected, 60 poles
  outside are all accepted; box-edge pole, removable singularity, overflow, `log`/`sqrt`/power cases.
- **False-rejection rate, measured:** of 307 random Stage-1 trees (depth ≤ 4, constants N(0, 1.5)), 302 stayed below
  the cap in 4000 random evaluations each and the audit accepted all 302 (0 false rejections); the 5 rejections are
  real overflows. This is for random trees, not for search output.
- Limits: real arithmetic with the constants as parsed; library accuracy ≤ 64 ulp is assumed, not proven; `sqrt` at
  exactly 0 is rejected by design (infinite slope) — this concerns raw `sqrt` only, `sqa` is total.

**(b) Magnitude bound, tight.** Coarse grid → per box the natural enclosure ∩ centred form (forward-mode interval
derivatives) → evaluation at centre and corners (a value above the limit **fails**, witness kept) → bisection along
the side with the largest `sup|G_i|·w_i` → at 2⁻¹⁰ of a side or at the budget (60k boxes) a dense grid of n^d
sub-cell centres with margin `Σ sup|G_i|·w_i/(2n)`. Status `certified` / `not_refuted` (no evaluation above the
limit, but not certified either; `[sup_lower, sup_upper]` reported) / `failed`. As the plan says, only `failed`
fails. Comparisons use `limit·(1 + 1e-9)`. The oscillation gauge (§8, D1) cuts the x-range into slabs adaptively
and encloses the whole range of f on a slab in a window of width 2·limit; it fails only on a pair of evaluations
at one x whose difference exceeds 2·limit. Top-level terms that depend on x alone are dropped first (exact).
- Verified: known suprema (`0.9·tanh`, exact saturation at 1.0, oscillation on the asymmetric box); the dependency
  case `0.9·(sin²(3ζ) + cos²(3ζ))·tanh(ϱ − 5)` (naive enclosure ±1.8, truth 0.9) is certified with upper bound
  0.9971 after 2459 boxes in 20 ms; a bump of height 1.05 and width 0.025 × 0.25 that the coarse grid misses is found
  and fails with a real evaluation, the same bump at 0.95 passes; brute-force agreement on random bounded trees;
  the witness of an oscillation failure is a pair of evaluations at one x.
- Trajectories: `audit_trajectory` (domain safety, rigorous range enclosure on 1024 slices, evaluated range on the
  full interval and on the training range, their ratio, optional `value_range` check with the tight machinery).

## 5. `targets.py` — implemented (definitions as designed in the first session)

**Target (iii)** `build_residual_target(log_rho_teacher (M,S,NZ,NR), log_rho_base (S,NZ,NR), slice_p, energy_w,
n_hits, z_mm, r_mm, response=, a_bar=, n_per_slice=650, gamma=0.5, halves=, seed=)`:
`y_s(c′) = ln(R_s ρ_T)(c′) − ln(R_s ρ_B)(c′)`, teacher = probability mean of the members; abscissae = centroid of
the tilted response `S = R ρ_B / (R ρ_B)`; `x = x(p_s)` identical within a slice; masks z′ < 1150 mm, first z row,
last ring, `Ā > 0.05`, ≥ 30 training hits, teacher SE ≤ 0.05 nat (skipped and flagged for one teacher);
`p_s ≥ 0.0895 GeV` enforced (raises); fixed-size systematic sampling with `π ∝ ω^γ`, HT weight `ω/π`; `y` has
HT-weighted mean zero per slice; rows sorted by slice (contiguous). Seed-null `y_null = κ[ln R ρ̄_A − ln R ρ̄_B]`,
`κ = sqrt((1/M)/(1/a + 1/b))`, same rows and weights. `event_slices(p, 12)`; `estimate_response(train, slice_of_event,
S, K)` (sparse `R_s`, energy and hit counts per cell; torch, training split); `null_loss`; `save_target` (refuses
non-contiguous slices) / `load_target_arrays`.
- Verified on synthetic maps: fixed sample size; HT estimator unbiased (600 draws) and the unweighted mean biased;
  every mask; weights = ω/π; the true δ has gauge loss < 1e-25 and stays there under `f + c(x)` while the plain L2
  does not; the stored `y` is unchanged when a constant per slice is added to the teacher; the seed-null has the
  noise sd of the full teacher (within 25 %) and no signal; response smoothing and centroid against a direct
  computation; the file is read by `search.load_target`. On synthetic showers: rows of `R_s` sum to 1, energy per
  slice = number of events (85 % of the response mass leaves the centre cell there — synthetic hits, not a
  detector number).
- **Not done:** no real target has been built — the final baseline (Track-A finalist) and the full teacher ensemble
  do not exist yet. Not measured: the Jensen difference to the hit-level mean, the size of the centroid
  approximation `Φ(f) − f(z̄, r̄)`, the number of cells that survive the masks per slice.
- Caveat kept from the first session: slices at event quantiles are uniform in p, so the lowest slice spans roughly
  x ∈ [−1, 0.1].

**Target (i)** `fisher_scores(model, train, columns, provider_attr="provider")` → per-event
`F_e,j = Σ_i w_i (∂_j ℓ_i)²` by central differences of the per-hit folded log-likelihood on common fold points
(`hit_loglik` equals `common.event_nll` to 2e-5); `trajectory_weights` (points at event quantiles, `n_g·F_jj(x_g)`);
`build_trajectory_target` → `X (G,1)`, `y (G,n_out)`, `weights (G,n_out)` (mean 1 per output, `nat_factor` back to
nats), `guesses_json` = `list[list[str]]` of Horner cubics; points outside x ∈ [−1, 1] and outputs without Fisher
information are refused.
- Verified: **the weights are the expected Hessian, not the score covariance** — on Gaussian hits the estimator
  equals the numerical Hessian of the loss (2 %) while the squared event score is smaller by `Σ w_i²`; on synthetic
  showers drawn from a toy two-parameter model (300 events × 200 hits), `fisher_scores` gives 0.796 and 1.329
  against second differences of the mean NLL of 0.802 and 1.341 (within 1 %; the test allows 15 %), while the
  squared event score is 0.0045 and 0.0067 — smaller by the number of hits.
- **Not verified:** `provider_attr` on a real Track-A model (`LastRing.provider` is a property there — the swap may
  need a setter); the neglected off-diagonal Fisher terms; `E[score | x] = 0` at the finalist.

**`.npz` schema** (`schema_version = "sfm.sr.target/1"`): `X`, `y`, `weights`, `variable_names`, `kind`, `loss`,
`meta_json`; residual: `slice_id`, `slice_p`, `slice_x`, `cell_iz`, `cell_ir`, `pi`, `omega`, `seed_se`, `y_null`;
trajectory: `output_names`, `nat_factor`, `event_fraction`, `fisher_diag`, `guesses_json`.

## 6. `refit.py` — implemented, registration is a stub

- `CorrectedModel(base, correction, freeze_base=False)`: `log_softmax(log P_base + f(zt, rh, x))` at the cell
  centres, f in float64 with all free constants trainable, no clamp, no `nan_to_num`; a non-finite f raises
  `FloatingPointError`. `expression_correction(expr)`; `check_gradients(model, data)`.
- `fit_candidate(model, train, val, FitConfig, label=)` = `sfm.train.fit` + **the audit of the refitted equation**
  (`audit_refit`, `audit_passed` in the result). It refuses to run unless `SFM_SR_REFIT_ALLOWED=1` (to be set in
  the H200 run-book) or `synthetic=True`.
- Controls: `control_c1(base)` (baseline alone, all constants trainable, same `FitConfig`); `control_c2(base, k)`
  (polynomial in rescaled `zt`, `sqrt(rh)` and `x`, first k monomials in graded order without the constant and the
  pure powers of x, zero start); C3 is `targets.build_residual_target(halves=...)`.
- `SubstitutedProvider(provider, {name: expr})` for trajectories, `k = provider − replaced rows + constants`.
- `register_refit` raises `NotImplementedError`; its docstring lists what the H200 implementation must do.
- Verified on synthetic showers (hits drawn from baseline × exp(0.8·tanh(1.5 zt − 1.5)), 500 events × 300 hits, CPU,
  300 steps): from the start (0.5, 1.0, −1.0) the fit reaches (0.798, 1.499, −1.517), truth (0.8, 1.5, −1.5); NLL on
  the synthetic validation set 16.39404 at the start, 16.36762 after the fit, 16.36755 for the generating model;
  no skipped batch; the refitted tree passes the audit; C2 with 3 constants starts at the baseline and ends at
  16.37578, worse than the equation; strict mode raises on `log(zt)`. (Synthetic numbers; one seed.)
- **Not run:** any refit on real data, any validation likelihood of a candidate, any registry call.
- Requires `SFM_ALLOW_CPU=1` for CPU tests since `sfm.train.fit` now refuses to run without CUDA.

## 7. `checks.py` — implemented

`sign_changes(v, min_peak)`, `count_extrema(v, prominence=0.01)`, `containment(P, z_edges, r_edges)` (r50, r90, z50,
z90), `monotone_sequence(values, tol=0.5 mm)`, `slope_sign_changes(log_rho, coord, start)`, `modulation_expr`
(f minus f with sin = cos = 0), `template_modulation`, `modulation_amplitude` (≤ 0.3 nat on the extended box with the
audit's tight machinery; sign changes inside / outside the data box, lobes ≥ 0.003 nat, ≤ 4 inside and 0 outside),
`trig_periods` (local period range of every sin / cos along z′ and r from interval derivatives of the phase),
`period_check` (1650/k mm, k = 1…8, 3 %, > 570 mm = trend), `period_band_coverage`, `seed_null_comparison`.
- Measured: the match bands of ±3 % cover between 25 % and 40 % of the testable period interval — the check has
  weak specificity (a match is not proof of copying, a non-match is weak evidence).
- **Not written:** the driver that evaluates a refitted model at 0.05, 0.1, 1, 5, 20, 30, 50 GeV and feeds these
  functions (it needs the refitted model, i.e. the H200); the tail-slope check against the baseline; "constants
  stable across seeds" (to be done as a comparison of the fitted functions on a common grid, §8.7).

## 8. Decisions taken in this session (mathematics) and plan rules they touch

**D1. Gauge of the magnitude bound.** Taken: `gauge="oscillation"`, limit 1 nat —
`H = sup_x ½[sup_{z′,r} f − inf_{z′,r} f] ≤ 1`, i.e. the plan's `sup|δ| ≤ 1` in the most favourable gauge
(`min_c sup|f − c(x)|`). Why not the earlier proposal `sup|f − f(ref, x)| ≤ 1`: it depends on the reference point by
up to a factor 2 in the tested quantity and contradicts the templates — `0.99·tanh(ζϱx)` satisfies the plan's rule
as written (|f| ≤ 0.99) and "fails" the reference rule at 1.98 with the reference (260 mm, 21.9 mm); it passes with
the reference at the origin. The oscillation is reference-free, exactly gauge-invariant, and implied by both other
readings, so the tanh templates pass by form. In every reading the guarantee on the normalised correction
`δ_eff = f − ln E_base[e^f]` is the same: `|δ_eff| ≤ 2H ≤ 2 nat`; a guaranteed `|δ_eff| ≤ 1` needs `limit=0.5`.
All three gauges are implemented (`gauge=`), the audit report records which one was used.

**D2. Stage-2 template.** Taken: the registered Stage-2 form stays the literal `tanh(f) + 0.3·tanh(a)·env·sin(ph)`.
It contains the Stage-1 form `tanh(f)` as the special case a → 0, so the ablation is nested and a gain can only come
from the modulation; the `0.7·tanh(f)` variant caps the trend at 0.7 and would bias the comparison against Stage 2.
`|δ| ≤ 1 nat` is therefore not guaranteed by the form (bound 1.3) and is decided per equation by the audit, as the
plan's gate already lists it. `s2_modulation_unit` stays in the code as a labelled extra arm, not run unless the
user asks.

**D3. `ppow` exponent.** Confirmed with Julia that `constraints["ppow"] = (7, 1)` admits `ppow(zt, rh)`. Taken:
`complexity_of_variables` stays 1 (the caps 18 / 30 and "a cubic is C = 13" depend on it); members with a
non-constant exponent are removed by `operators.violations` before the audit, so no such equation is audited,
refitted or counted. Consequence to report at Gate 2-pilot: the fraction of hall-of-fame members removed, and that a
removed member can leave a hole in the front at its complexity.

**D4.** The audit is repeated on the refitted equation (`refit.fit_candidate`); already in A6 point 7.
**D5.** Fisher weights = expected Hessian of the energy-weighted pseudo-likelihood (§5), tested against the score
covariance.

### Plan rules touched — items 1–8 accepted by the user on 25 Nov 2026 (with the four confirmations of §12)

1. **`sup|δ| ≤ 1 nat` (§7) is evaluated as the half-oscillation over (z′, r) at fixed x** (D1). Alternatives ready:
   reference gauge at limit 1; `limit=0.5` for a guaranteed `|δ_eff| ≤ 1 nat`.
2. **"An equation fails only if an actual evaluation exceeds the bound" is applied literally:** a result that is
   neither certified nor refuted (`not_refuted`) passes, with its interval reported. Stricter alternative: require
   `certified`.
3. **Additions to the domain check that only reject more:** the overflow cap 1e30 on every node (applies to Stage 1,
   which is pole-free but can overflow), the float32 allowance τ around forbidden sets, raw `sqrt` rejected at 0.
4. **"Exponent of `ppow` a constant" is enforced after the search** (D3), not inside it.
5. **"Probability-proportional sampling" uses the size measure ω^0.5** (`gamma=0.5`; ω = energy share), with exact
   Horvitz–Thompson weights; `gamma=1` is the strict reading and is one argument away.
6. **"Refit: all constants trainable" is read as baseline and equation together** (`freeze_base=False`), which is
   what makes control C1 meaningful; k of the equation and k of the whole model are reported separately.
7. **C2** is a polynomial in (ζ, √ϱ, x) with the equation's k, without the monomials the softmax cannot see — the
   plan says only "smooth tensor polynomial with the same number of constants".
8. **Complexity weights** (first session, still open): `sqa` and `linv` cost 2, not 1.

**New choices made while building in the four confirmations — not yet seen by the user:**

9. **Raw magnitude limit:** the raw, unshifted |f| (top-level gauge terms stripped) must stay below 16 nat on the
   extended box (`RAW_LIMIT_NAT`). A new rejection criterion; it only rejects more.
10. **Rounding tolerance:** the first-order float64 rounding-error bound of f must stay below 1e-6 nat
    (`ROUNDING_TOL_NAT`); rejects only expressions with catastrophic cancellation.
11. **Gauge constants are not counted in k and not trained:** `refit.expression_correction` strips top-level terms
    in x alone before translation.
12. **Finalist rule:** a finalist needs `certified` (magnitude bound proven, not merely unrefuted) plus the tail
    check with thresholds −2 (radial log-log slope) and 0 (depth slope), on the refitted constants; the baseline is
    assumed to continue beyond the grid as the power law / exponential of its last four interior rings / rows.
13. **`sqa` derivative at exactly 0 is defined as 0** in the torch implementation (the value is untouched), and
    `softplus` uses its exact derivative.
14. **C2:** the code default stays the unbounded polynomial with its audit verdict reported; `bounded=True`
    (tanh of the polynomial) is the labelled alternative that this author recommends as the registered control (§12.4).

### Ill-posed points that remain (not decided here)

- "Integrable tails" follows from the baseline's tail once the correction is bounded; the slope criterion is a
  regularity condition that the baseline itself may fail. Checks at 0.05, 30 and 50 GeV evaluate the baseline
  outside its training range: only "not worse than the baseline" is well defined there.
- "Modulation" for a free-form Stage-2 expression is defined here as f minus f with sin = cos = 0 (`checks`).
- "Constants stable across seeds" has no meaning across structurally different expressions.
- The gauge loss makes pure functions of x free: the confirmation run produced `… − x` and `… + 0.156` members with
  unchanged loss at higher complexity. The hall of fame should be de-duplicated modulo gauge terms
  (`audit.drop_gauge_terms`) before counting front members.

## 9. Confirmed with Julia (25 Nov 2026; Julia 1.13.4, PySR 2.7.3, SymbolicRegression.jl 2.7.2, 4 threads, seconds each)

Scripts and raw output: scratch directory `sr_math/` of the session (`confirm_julia.py`, `fixture_julia.py`,
`ppow_julia.py`; not kept in the repository — the durable evidence is the fixture inside `tests/test_sr_translate.py`).
Every item of the first session's list is confirmed (`operators.CONFIRMED_WITH_JULIA`):

| Item | Result |
|---|---|
| custom operators, three dictionaries keyed by custom names | accepted; Float64 and Float32 |
| argument-size limits | counted with operator weights (`exp(tanh ζ + tanh ϱ + tanh x)` rejected at weighted 11 > 9, unit count 8) |
| `ppow(·, variable)` | accepted by `(7, 1)` → scan needed (D3) |
| nesting semantics, complexity | Julia's `check_constraints` / `compute_complexity` agree with `violations` / `tree_complexity` on 14 hand-made trees |
| weighted elementwise loss | normalised by Σw (1e-15) |
| gauge loss, tree and template expression, `batching=False` | equals `gauge_loss_numpy` (1e-15) |
| template combine with literals, `^2`, `sin`, `0.3f0` | works; template complexity = sum over sub-expressions |
| string formats | free form `tanh(zt) * 0.53`, `erf(zt + -0.25) * 0.53`; template `u = #1 * 0.6; v = 0.3; w = square(#1)`; 17 digits |
| multi-output | `weights (N, n_out)`, `guesses list[list[str]]`, `hall_of_fame_output{i}.csv` |

Not confirmed here (runner's items): kill-and-resume, `from_file` for templates, thread scaling, convergence.
`search.py` defaults to `print_precision = 15`; `operators.search_config` asks for 17 — 15 digits are inside the
1e-5 rule but not bit-exact.

## 10. What the runner worker needs to know (interfaces)

- Targets: `targets.save_target(path, target)` writes the `.npz` that `search.load_target` reads. Residual target:
  `loss_function = operators.gauge_loss_julia(3)`, `batching=False`, rows already slice-contiguous. Trajectories:
  `elementwise_loss = operators.WL2_ELEMENTWISE`, `guesses = json.loads(str(npz["guesses_json"]))`. The seed-null
  search uses the same file with `y_null` in place of `y` (write a second file with `y := y_null`).
- Hall-of-fame scan: `operators.violations(translate.parse(eq), vocabulary)`; for templates on every sub-expression
  (`translate.split_template_string`, variables `#k`).
- Translation: free form `translate.parse(eq, names)`; template `translate.parse_template_string(name, eq)` or
  `parse_template(name, {sub: string})` from `hall_of_fame.json["subexpressions"]`.
- Round trip (Gate 2-pilot): export `model.predict(X, index=i)` per member from the search process (NumPy), then
  in a torch process `translate.round_trip(expr, X, names, reference=pred)`.
- Audit of the front (laptop, no data): `audit.audit(expr).to_json()`; send only `passed` members to the H200 with
  `translate.to_dict(expr)` (lossless, keeps the template's fixed literals) and `translate.expr_hash(expr)`.
- H200: `refit.CorrectedModel(base, refit.expression_correction(translate.from_dict(d)))`,
  `refit.fit_candidate(...)` with `SFM_SR_REFIT_ALLOWED=1`; controls `control_c1`, `control_c2(base, k)` with the same
  `FitConfig`; registration to be written there (`register_refit` docstring).

## 11. Not verified

- Anything on real data: target construction end to end, mask survival, Fisher weights of a real Track-A model.
- The audit on real search output (only on random trees, hand-made cases and the earlier study's equations).
- Library accuracy ≤ 64 ulp (assumed for the interval margins); the τ allowance is heuristic.
- The budget behaviour of the magnitude bound for size-30 expressions with high-frequency structure (60k boxes,
  then `not_refuted`); the cases tried here took tens of milliseconds.
- Template parameters `p[i]`; a real refit; the checks driver at the seven momenta.

## 12. The user's four confirmations (25 Nov 2026) — what was true, what was changed, what the tests show

### 12.1 Float32 and gradient hygiene (rules 1 and 3)

**Half-swing.** Kept: `H = sup_x ½[sup − inf over (z′, r)] ≤ 1 nat`. The report states the consequence explicitly:
`magnitude["delta_eff_bound"] = 2H` and the summary line prints `|delta_eff| <= 2H = …` — up to 2 nat after
normalisation.

**(a) "The overflow cap bounds the raw unshifted |δ| and prevents float32 cancellation" — this was FALSE.**
`OVERFLOW_CAP = 1e30` only keeps nodes finite. Measured before the change: `0.5·tanh(ζ − 2) + 1e4·x` passed the
audit (domain safe, H certified at 0.499) while a float32 evaluation damaged its ζ-dependence by 2.5e-4 nat (at
x = 0.731; the test asserts > 1e-4). Now four mechanisms, all re-applied to the refitted constants:

1. `refit.expression_correction` strips top-level additive terms in x alone (`audit.drop_gauge_terms`) before
   translation; their constants are neither trained nor counted in k; a pure gauge equation is refused.
2. `audit.numerics`: the **raw, unshifted |f|** of what remains must stay ≤ 16 nat on the extended box (tight
   machinery, gauge "none"). This catches gauge-like terms that are not top-level summands:
   `0.5·tanh(ζ − 2) + 1e4·x·linv(0·ϱ)` is now rejected (raw 4.1e4), `… + 12·linv(0·ϱ)` passes, `… + 20·linv(0·ϱ)`
   is rejected although its H is certified.
3. `refit.CorrectedModel` evaluates the correction, its sum with log P_base and the softmax in **float64** and
   raises `TypeError` for a float32 correction.
4. `audit.rounding_error_bound`: a first-order forward bound of the rounding error of f on the extended box (interval
   enclosures of values and derivatives on 16^d sub-boxes; one unit roundoff per arithmetic operation and per
   constant, 4 per library call). Required: float64 bound ≤ 1e-6 nat. Reported: the float32 bound and
   `float32_safe` (≤ 1e-5 nat).

**Guaranteed error of the correction after these mechanisms** (for an equation that passes the audit, before or
after the refit): rounding error of f in the refit ≤ its float64 bound ≤ 1e-6 nat by requirement (measured bounds:
4.0e-16 nat for `0.5·tanh(ζ − 2)`, 3.7e-14 for `0.99·tanh(ζϱx)`, 3.9e-9 for an expression with an `sqa` cusp);
adding f to log P_base costs ≤ 2⁻⁵³·(16 + |log P|) ≈ 1e-14 nat. The only float32 step left is the cast of the
normalised log P, one rounding of |log P| (1.2e-6 nat at |log P| = 20), the same for every model of the campaign.
If the correction were evaluated in float32 instead: ≤ the float32 bound of the equation (2.1e-7 nat for
`0.5·tanh(ζ − 2)`; 7.4e-3 nat for the same equation with the unstripped `+ 1e4·x`) plus 2⁻²⁴·(16 + |log P|) < 4e-6 nat for the sum.
- Tests: the premise test above; stripping and the raw limit; `(exp(6ζ) + 0.5·tanh ϱ) − exp(6ζ)` (1e20 + small −
  1e20) is rejected with a float64 bound of 2.4e6 nat although it has no pole and stays below the overflow cap;
  on 60+ random Stage-1 trees the measured float32-against-float64 difference at 3000 points never exceeds the
  float32 bound; in the refit suite the model with and without the gauge terms `+ 1e4·x + 7` is identical and k = 2.
- **Not guaranteed:** the rounding bound is first order and assumes 4 ulp per library call; it is pessimistic for
  float32 (`0.99·tanh(ζϱx)` gets 2.0e-5 and is labelled not float32-safe, from the size of the sub-boxes). A
  gauge-like constant below 16 nat hidden inside a product is not stripped (harmless in float64). The searches
  themselves run in Julia Float64 and are not affected.

**(b) "`sqa` cannot produce Inf/NaN gradients at 0" — this was FALSE.** Measured before the change: the gradient of
the torch `sqa` at exactly +0.0 and −0.0 was NaN in float32 and float64 (inf·0). `softplus` gave 0 instead of ½ at
exactly 0 (finite, wrong at one point). `abs`, `l1p`, `ppow` give 0 there (finite). Changed in
`operators.torch_impl`: `sqa` keeps its forward value bit for bit (NaN and Inf inputs still propagate) and has its
backward pass defined as 0 at exactly 0; `softplus` is the Julia formula forward and `sigmoid(x)` backward.
- Tests: every total operator is backpropagated at +0.0 and −0.0, in float32 and float64, in every argument
  position — all gradients finite; a `CorrectedModel` whose `sqa` argument is exactly 0 on 128 grid cells has a
  finite loss and finite gradients; the round trip against PySR's own predictions is unchanged (91 tests).
- **Not guaranteed:** next to the cusp the slope is 1/(2√|u|) — finite for every non-zero float, but as large as
  5e149 at |u| = 1e-300. The harness clips the gradient norm at 5 and counts skipped batches; a constant that
  sits on a cusp can stall. A cusp of `sqa` inside the tail region makes the tail slope unbounded, and such a
  finalist cannot be certified (§12.2).

### 12.2 Post-refit certification (rules 2 and 6)

**Before:** the screening audit was already repeated on the refitted equation, but `passed` accepted an undecided
magnitude bound and there was no tail check. **Now:**
- `AuditReport.certified` = domain safe ∧ magnitude **certified** ∧ numerics certified; `audit(…,
  require_certified=True)` makes `passed` equal to it. Screening keeps the plan's literal rule (undecided passes,
  interval reported).
- `audit.certify_finalist(expr, baseline_tail=…)` = that certificate plus the tail check: `audit.tail_slopes`
  (rigorous enclosures of ∂f/∂ln r beyond 700 mm and ∂f/∂z′ beyond 1450 mm on the extended box) added to the tail
  slopes of the refitted baseline (`checks.baseline_tail_slopes`); required: radial log-log slope < −2, depth
  slope < 0; "not worse than the baseline" is reported.
- `refit.fit_candidate(…, finalist=True, baseline_tail=…)` runs it **on the constants after the joint refit** and
  returns `finalist_certified`; `register_refit`'s specification refuses a finalist without it.
- Tests: `0.9995·(sin² + cos²)(3ζ)·tanh(ϱ − 5)` at a 2000-box budget is undecided (H in [0.99945, 1.072]) — it passes
  screening and is refused as a finalist, before the "refit" and after it (constants moved to 0.9993, 3.01, 5.02);
  the same form at 0.9 is certified and loses the certificate when the refit moves it to 1.3 (`failed`); a refit
  that moves a pole into the box gives the reason `domain: unsafe_proven`; tail cases (rise beating the baseline,
  baseline itself not integrable, missing baseline slopes) each refuse with their reason; in the synthetic refit the
  certificate is computed for the refitted constants (0.798, 1.499, −1.517), not the starting ones.
- **Not guaranteed:** the tail check needs the refitted baseline's slopes, i.e. the H200; without them
  `certified` is False by design. The continuation of the baseline beyond the grid is an assumption. The thresholds
  −2 and 0 are this module's (first session, §7), not the plan's wording. The tail enclosure refines adaptively
  but can stay loose (reported `converged`), which costs false refusals only. An undecided magnitude bound may be
  decidable with a larger `max_boxes`; nothing raises the budget automatically. Registration is still a stub, so
  nothing enforces the rule yet outside `fit_candidate`'s return value.

### 12.3 `ppow` constraint (rule 4)

**Already true for the stated purpose, and kept as it is:** `constraints["ppow"] = (7, 1)` in Stage 1, the raw arm,
Stage 2 and every template search — exponent size 1, so no compound exponent can be built (confirmed with Julia:
`ppow(zt, rh + x)` is rejected by `check_constraints`). `(−1, 1)` has the same exponent limit and would only remove
the cap of 7 on the base, which is one of the plan's argument-size limits; not adopted. The post-search scan stays,
because a size-1 exponent can still be a variable (`ppow(zt, rh)` is accepted by Julia).
- Test: `test_ppow_exponent_limit_is_one_in_every_vocabulary`.
- **Not guaranteed:** the search can still spend evaluations on `ppow(u, variable)`; only a change of
  `complexity_of_variables` or of the vocabulary would stop that, and both alter pre-registered sizes. The pilot
  reports the fraction of such members.

### 12.4 Pilot logging and C2 fairness (rules 5 and 7)

**(a) n_eff.** New: `targets.build_residual_target` stores per momentum slice `slice_n_eff` = (Σw)²/Σw² of the
Horvitz–Thompson weights under the size measure ω^0.5, `slice_n_eff_gamma1` (an equal-size comparison sample under
strict proportionality, drawn with its own generator so the target does not depend on it), the effective number of
cells of the population 1/Σω², and the counts; also in `meta_json`. `targets.n_eff_report(target)` and
`targets.format_n_eff(target)` give the rows and the table for the Gate 2-pilot log. Wording corrected everywhere:
the HT weights make *linear totals* unbiased; the within-slice weighted variance is a ratio of weighted sums and is
**design-consistent (asymptotically unbiased), not unbiased**.
- Tests: stored values equal the weights' n_eff; the comparison draw does not change the target; a Monte Carlo over
  designs shows the loss estimate biased low at n = 25 (beyond 4 standard errors), within 2 % at n = 400, and a
  bias of relative order 1/n_eff. Illustration on the synthetic maps (120 of about 280 cells): n_eff 91–96 under
  ω^0.5, 97–101 under ω^1, 104–112 for the population — synthetic, no statement about the real target.
- **Not known:** the real n_eff, until the target is built.

**(b) C2.** New: the control polynomial is an expression tree (`refit.c2_tree`; the same function as before, with
`sqa(ϱ) = √ϱ`), so `fit_candidate(…, role="control")` and `refit.audit_control(model)` run the **same extended-box
audit** (domain, magnitude with the same gauge and limit, numerics) on the fitted coefficients and return
`audit_verdict` = {passed, certified, domain, magnitude, H_lower, H_upper, violates_extended_box_bound}, to be
printed beside Δ(SR) − Δ(C2). It is a report, not a gate. `control_c2(base, k, bounded=True)` is the labelled
alternative tanh(polynomial), bounded by form.
- Tests show the issue is real: on the synthetic problem the fitted three-constant polynomial
  (0.996, −0.008, −1.292) **violates the bound** (H ≥ 1.48 by actual evaluation) while reaching NLL 16.3758; the
  bounded variant is certified with H ≤ 1 for arbitrary coefficients and starts at the baseline.
- **Recommendation:** register tanh(polynomial) as C2. A polynomial is free outside the data, where an SR candidate
  is constrained, so the unbounded control answers an easier problem and Δ(SR) − Δ(C2) would be tilted against SR;
  the bounded one obeys the same constraint with the same k, and for corrections well below 1 nat it differs from
  the polynomial only at third order. The unbounded polynomial can be kept as a labelled second control with its
  verdict. The code default is still the unbounded one (the reading the user accepted): the switch is a decision.
- **Not verified:** how much the two controls differ in likelihood on real data.

### 12.5 Interface changes (additions only)

- `audit`: `numerics`, `rounding_error_bound`, `tail_slopes`, `certify_finalist`, `canonical_gauge_form`;
  `audit(…, require_certified=, raw_limit=)`; `AuditReport` gains `numerics`, `certified`, `require_certified`
  (`passed` now also requires the numerics check). `drop_gauge_terms` returns the tree itself when nothing is
  dropped. Note for `scripts/sr_post.py`: hashing `canonical_gauge_form(tree)` for every member de-duplicates also
  members that differ by the bracketing of their top-level sum.
- `refit`: `expression_correction(…, strip_gauge=True)`, `c2_tree`, `PolyControl(k, bounded=)` (parameter name is now
  `correction.poly.c`), `control_c2(…, bounded=)`, `audit_control`, `audit_verdict`,
  `fit_candidate(…, role=, finalist=, baseline_tail=)` with new result keys.
- `targets`: `effective_sample_size`, `n_eff_report`, `format_n_eff`; new arrays `slice_n_eff*`, `slice_n_sampled`,
  `slice_n_masked`, `sampling_gamma`.
- `checks`: `baseline_tail_slopes`. `operators`: only `torch_impl` changed (nothing the search uses).
