# `sfm.sr` — status of the math-author modules (updated 26 Nov 2026 after the user's decisions of PLAN §20, items 4–7)

Written for whoever continues. All six modules are implemented; each section says what the code does, what was
verified and what was not. Section 8 holds the decisions and the plan rules they touch (**all eight accepted by
the user on 25 Nov 2026**); section 12 answers the user's four confirmations; **section 13 builds in the user's
decisions S1–S5 (PLAN §17.11) and holds the tail-criterion preview for the legacy formula**; section 14 explains
why the legacy formula fails the absolute depth rule; section 15 implements the user's decisions on it (inheriting
depth rule anchored to the fixed reference, Stage-2 check, cusp-stall flag); **section 16 implements PLAN §20, items
4–7 (anchor confirmed, C1 test, rising stretches enforced instead of sign changes, cusp flag as a review trigger) and
is in force where it differs from sections 12–15**; section 10 holds
the interfaces for the runner. Where sections 12 and 13 differ, 13 is in force. **Numbers are measured only where they say so.**

## 1. Status per module

| Module | State | Tests | Notes |
|---|---|---|---|
| `operators.py` | done (first session); `UNCONFIRMED` emptied, `CONFIRMED_WITH_JULIA` added; torch `sqa` / `softplus` gradients fixed (§12.1) | 70 pass | every item of `UNCONFIRMED` is now confirmed against Julia, see §9 |
| `translate.py` | **done** | 91 pass (`tests/test_sr_translate.py`, 1.3 s) | torch only on demand; checked against real PySR strings, predictions and losses |
| `audit.py` | **done** | 83 pass (`tests/test_sr_audit.py`, 18 s) | NumPy only; (a) domain, (b) magnitude, (c) numerics; stripped-term log; `tail_check`, `certify_finalist` |
| `checks.py` | **done** | 15 pass (`tests/test_sr_checks.py`) | NumPy only; tolerances are arguments with stated defaults |
| `targets.py` | **done**, not yet run on real data | 18 pass (`tests/test_sr_targets.py`) | constructions tested on synthetic maps; event-level estimators on synthetic showers (CPU child process) |
| `refit.py` | **done except registration** (stub, by instruction) | 20 pass (`tests/test_sr_refit.py`, CPU child process) | constants recovered on synthetic showers; no real refit run |
| `__init__.py` | empty, as `test_sr_package_init_is_empty` requires | | |

Run: `uv run pytest shape-form-martin/tests -q -k "sr_ and not search"` → 301 passed (about 37 s): 297 in the six
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

**Choices made while building in the four confirmations — decided by the user as S1–S5 (PLAN §17.11), see §13:**

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

### 12.2 Post-refit certification (rules 2 and 6) — the tail part is superseded by §13.4

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

## 13. The user's decisions S1–S5 (PLAN §17.11) — built in

### 13.1 S1 — tanh(polynomial) is the registered C2

- `PolyControl(k)`, `c2_tree(k)` and `control_c2(base, k)` now default to `bounded=True`: f = tanh(Σ c_j u^a v^b x^c).
  This is the only C2 that `fit_candidate(…, role="control")` accepts, and the only one `c2_report` labels as used
  for Δ(SR) − Δ(C2).
- The plain polynomial is `control_c2(base, k, bounded=False)` with `fit_candidate(…, role="diagnostic")`: a
  non-governing row carrying its audit verdict. `role="control"` refuses it; a `PolyControl` is refused as
  `role="candidate"`; `c2_report(governing, diagnostic)` raises if the two records are swapped.
- Tests (synthetic showers, 120 steps, one seed): the governing control is certified by form (H ≤ 0.979) and its
  refitted tree starts with `tanh(`; the plain polynomial is refused as governing, runs as diagnostic and its row
  prints `audit: FAIL` (H ≥ 1.48) beside it. NLL on the synthetic validation set: start 16.49732, tanh control
  16.38709, plain polynomial 16.37578, equation 16.36762.
- **Not guaranteed / to keep in mind:** on that synthetic problem the true correction is 0.8 nat, and there the
  bounded control fits 0.011 nat worse than the plain polynomial: the two are close only for corrections well
  below 1 nat. How far they differ on real data is not known.

### 13.2 S2 — 16 nat raw limit, 1e-6 nat rounding tolerance, float64

Nothing to change: `RAW_LIMIT_NAT = 16`, `ROUNDING_TOL_NAT = 1e-6` and the float64 evaluation enforced by
`CorrectedModel` are as accepted (§12.1; tests unchanged and passing).

### 13.3 S3 — stripped gauge terms: not trained, not counted, always logged

- `audit.gauge_term_records(expr, gauge_vars, stage)`: one record per stripped term — signed text, the values of
  its constants, its variables, and the stage.
- `AuditReport` gains `stripped_gauge_terms`, `k_as_written`, `k_stripped`, `stage`; **`k` is now the count without
  the stripped constants** (it used to include them). `audit(…, stage=, carried_stripped=)`.
- `refit.expression_correction` records what it strips (stage "before translation") on the module;
  `fit_candidate(…, pre_refit_audit=)` carries the pre-refit audit's records (stage "before refit") and its own
  into the post-refit audit record (stage "after refit" for anything stripped there) and into its return value
  (`stripped_gauge_terms`, `k_equation`, `k_as_written`); it warns if the two lists of stripped terms differ.
- Test, end to end: `0.5·tanh(ζ − 1) + 3.0·x + 7.0` → pre-refit record k = 3 of 5 with `+ 3.0 * x` [3.0] and
  `+ 7.0` [7.0]; the model has three trainable numbers, none of them 3.0 or 7.0; the refit record lists the two
  terms twice (before refit, before translation), `k_equation = 3`; the post-refit audit carries the same list,
  has k = 3 and an equation without x-only terms.
- **Not guaranteed:** only top-level summands are recognised as gauge terms (a hidden one is caught by the raw
  limit, not stripped).

### 13.4 S4 — tail criterion: decided once per baseline, from the baseline alone (depth rule and record format superseded by §15.1)

**(a) Which quantity.** Every slope is of the **3-D density ρ(z′, r)** — probability per volume, `log P − log(cell
volume)` or the density functional — never of the ring mass 2πrρ (whose log-log slope is larger by one; a test
shows that feeding the ring mass flips the decision for ρ ∝ r^−2.5). The correction adds `∂f/∂ln r` and `∂f/∂z′`
because `ln ρ_model = ln ρ_base + f − ln Z(x)`. **Where:** the baseline is evaluated at the seven pre-registered
momenta 0.05, 0.1, 1, 5, 20, 30, 50 GeV, at every depth row given (128 rows, z′ from −194 to 1444 mm, for a model on
the grid) and at radii covering 200–1500 mm; depth slopes at every radius up to 1500 mm for z′ > z_90(p), with
z_90 from the baseline's own map. The equation's part is enclosed on the same boxes at x(p) of those momenta; under
the alternative radial rule it is enclosed for *all* z′ and x of the audit box.

**(b) The decision** — `checks.decide_tail_criterion(log_rho, z, r, momenta, z90, baseline_hash=, label=)`:
radial log-log slopes by finite differences over r ∈ [200, 1500] mm; if the largest is ≤ −2 the rule is
`loglog_slope_le_limit`, otherwise `monotone_and_no_flattening` (∂_r ln ρ < 0 and ∂_r δ ≤ 0 for r > 200 mm). The
record stores the measured slopes (maxima per momentum × 8 depth bands × radial interval; per momentum × 4 depth
bands beyond z_90 × radius), whether the baseline satisfies each rule, the rule, the baseline hash and its own
hash. `verify_tail_criterion` refuses a record that was edited, whose rule does not follow from its slopes, or
that belongs to another baseline; `certify_finalist(…, tail_criterion=, baseline_hash=)` calls it and refuses to
run without the hash. **The rule cannot be chosen per equation.** For a closed-form baseline
`refit.tail_criterion_for_baseline(model_std, model_ext, baseline_hash=, label=)` continues the density to 1500 mm
(`extended_ring_grid`, `continued_log_density`: the model on a grid with 24 extra rings, each component matched to
the baseline on the shared rings) and makes the record.

**(c) Both rules, soundly** — `audit.tail_check`: interval enclosures of the correction's slopes, refined where
the baseline leaves little room. Tests: an equation that only inherits the halo (`−0.1·l1p(ϱ)`) is certified
under the alternative on a halo baseline and would fail "≤ −2" there (shown with a forged record, which
`certify_finalist` rejects); `+0.1·l1p(ϱ)` fails the alternative ("flattens") and passes "≤ −2" on a steep
baseline; borrowing the steep baseline's record for the halo baseline raises; a baseline that is not monotone
itself gives that reason; `measured_tail=` uses the slopes of the jointly refitted baseline while the rule stays
that of the record. After the synthetic refit the certificate is granted on a baseline with depth slope
−0.01/mm and refused on one with −0.004/mm, because the refitted equation rises by 0.0046/mm beyond z_90.

**(d) PREVIEW — legacy component formula, NOT the refitted baseline of record** (laptop, CPU, float64, read-only
checkpoint `new3_std_n3.pt`; density before the acceptance template; continuation mismatch on the shared rings
7e-15; radii 183–1477 mm, 128 depth rows):

| p / GeV | 0.05 | 0.1 | 1 | 5 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|
| largest d ln ρ / d ln r, r ≥ 200 mm | −0.95 | −0.95 | −0.89 | −1.21 | −1.27 | −1.13 | −0.92 |
| same, restricted to the grid (r < 700 mm) | −0.96 | −0.96 | −0.90 | −1.92 | −1.91 | −1.78 | −2.02 |
| z_90 / mm | 124 | 98 | 385 | 663 | 857 | 908 | 950 |
| largest d ln ρ / d z′ beyond z_90, per 100 mm | −0.08 | −0.08 | **+0.33** | **+0.55** | +0.02 | +0.04 | +0.02 |
| share of (z′, r) points beyond z_90 where ρ rises | 0 | 0 | 0.5 % | 0.4 % | 7.0 % | 18.5 % | 9.7 % |
| radii where it rises / mm | — | — | 1–227 | 74–329 | 711–1182 | 610–1260 | 610–917 |
| r-integrated depth profile beyond z_90, largest slope per 100 mm | −0.08 | −0.08 | +0.02 | +0.13 | −0.24 | −0.22 | −0.21 |

- **Radial: the legacy formula is flatter than −2 at every momentum** (largest slope −0.89 at 1 GeV, z′ = 1173 mm,
  r ≈ 193 mm; 41 % of the evaluated points are above −2) and it is monotone (all slopes negative). **Expected
  branch: `monotone_and_no_flattening`.** Within the grid only 50 GeV meets −2.
- **Depth: the legacy formula itself violates "∂_z′ ln ρ ≤ 0 beyond z_90"** at 1, 5, 20, 30 and 50 GeV — from the
  axis out to 227 mm at 1 GeV (on the axis +0.15 per 100 mm), at 74–329 mm at 5 GeV, and at 610–1260 mm at 20–50
  GeV. The r-integrated depth profile also rises at 1 and 5 GeV. Under the accepted rule no correction could then
  be certified, δ = 0 included. The cause was not investigated (candidates: the range end of the track component
  beyond z_90 at low momentum, the widening of the profile with depth at high momentum).

**Open, for the user (S4 names an alternative for the radial rule only).** If the refitted Track A finalist behaves
like the legacy formula in depth, the depth rule needs the same kind of decision. Built in, **off by default**:
`decide_tail_criterion(…, allow_depth_alternative=True)` makes the rule "no depth flattening" (∂_z′ δ ≤ 0 beyond
z_90 at the registered momenta) when — and only when — the baseline itself fails. Until that is decided the
accepted rule governs, the record states that the baseline fails it, and `tail_check` reports both verdicts
(`literal_rule_passed`, `no_flattening_passed`). Other readings the user may prefer: the r-integrated profile, or
the axis only.

**Not guaranteed.** The preview is for the legacy formula; the decision of record is taken on the H200 for the
refitted Track A finalist, which does not exist yet. The continuation needs a model with `components(lnp)` and a
`grid` (the component and mixture models of `sfm.families`); a grid-only baseline (an MLP map) is evaluated to
700 mm and the record says `covers_r_max: False`. Depth is covered to the last grid row (1444 mm), not to the
2000 mm of the audit box. The baseline slopes are finite differences at the registered momenta, not a continuum
bound. The measured slopes of the *jointly refitted* baseline have to be supplied (`measured_tail`) for the "≤ −2"
rule to be exact after a joint refit; under the alternative only its monotonicity enters. Enclosures that stay
loose cost false refusals.

### 13.5 S5 — gradient-clip log (the flag now also includes the clip-independent part, §15.3)

- `refit.ClipMonitor` (tensor hooks; no change to `sfm.train`) records at every step the gradient norm **before**
  clipping over all trainable parameters and the raw gradient and value of every constant of the correction;
  `fit_candidate` attaches it and returns `clip_log` = `refit.clip_statistics(…)` and `cusp_stalled`.
- **Definitions** (`refit.CLIP_DEFINITIONS`, repeated in every log):
  - *clipped step*: pre-clip norm > `FitConfig.clip`;
  - *driven by constant i*: a clipped step with g_i² ≥ ½ of the squared norm;
  - ***persistent***: constant i drives ≥ 20 consecutive steps, or more than half of the steps of the final quarter
    of the run (if that quarter has ≥ 8 steps);
  - *stalled*: persistent, and over that window |net change| ≤ 0.25 × Σ|step-to-step change|;
  - ***cusp-stalled (run)***: a stalled constant lies inside the argument of an `sqa` node.
  The log holds counts per constant (driven steps, longest run, share of the final quarter, net/path, sign flips)
  and per step (clipped or not, driven by which constants), plus steps without a backward pass and with a
  non-finite gradient.
- Tests: the definitions on synthetic gradient histories (bouncing on a cusp → flagged; the same outside an `sqa`
  argument → stalled, not cusp-stalled; clipped but progressing → persistent only; ten clipped steps → not
  persistent; another parameter driving → not flagged). With the real harness, a fit whose optimum lies on a kink
  of the gridded likelihood (truth `0.6·sqa(ζ − ζ_k)`, ζ_k a grid row): at clip norm 1e-4 all 160 steps are
  clipped, the constant inside the `sqa` drives 127 of them (longest run 20, 78 % of the final quarter, net/path
  0.011) → **cusp-stalled**.
- **Not guaranteed — the flag as specified does not fire at the harness's default clip norm.** The same fit with
  `clip = 5` has **0 clipped steps of 160** (largest pre-clip norm 0.019): at a V-shaped kink the gradient is
  bounded, far below 5, so the constant hovers at the cusp without ever being clipped. Clip counts flag a cusp
  stall only if the clip norm is within the range of the gradients.
- **Supplementary indicator, mine, reported and not part of the flag:** `refit.cusp_hover_statistics` /
  `cusp_hover_suspect` — for every `sqa` node, whether its argument changed sign in a grid cell during the final
  quarter of the run, or came within 3 of its own excursions of zero. It flags the default-clip run above (distance
  1.1 excursions) and not a control whose `sqa` argument has no zero on the grid (154 excursions). For a constant
  unrelated to any cusp it fires by chance at a rate of roughly 6 excursions per grid spacing of the argument.
  Whether it should become part of the flag is a decision.
- The monitor's state is not in the harness's checkpoint: after a resume the log covers the steps of that process.

### 13.6 Interface changes of this round

- `audit`: `gauge_term_records`, `tail_check`, `FLAT_SLACK`; `AuditReport.k` excludes stripped constants;
  `certify_finalist(…, tail_criterion=, baseline_hash=, measured_tail=, carried_stripped=)` — the two-number
  `baseline_tail=` of §12.2 is refused with a pointer to the new arguments.
- `checks`: `decide_tail_criterion`, `verify_tail_criterion`, `format_tail_criterion`.
- `refit`: defaults `bounded=True`; `fit_candidate(…, role="control" | "diagnostic", tail_criterion=,
  baseline_hash=, measured_tail=, pre_refit_audit=)` with new result keys (`governing`, `control_label`,
  `stripped_gauge_terms`, `k_as_written`, `clip_log`, `cusp_stalled`, `cusp_hover`, `warnings`); `c2_report`,
  `ClipMonitor`, `clip_statistics`, `cusp_hover_statistics`, `constants_inside_sqa`, `extended_ring_grid`,
  `continued_log_density`, `tail_criterion_for_baseline`.
- Nothing that `scripts/sr_post.py` or the search runner uses was renamed or removed.

## 14. Why the legacy formula violates "depth slope ≤ 0 beyond z_90" (investigation, read-only; no code changed)

Legacy component formula, **not the baseline of record**. Scripts and raw output: scratch directory `sr_math/`
(`depth_cause_model.py`, `depth_cause_data.py`). Model part: CPU, float64, density before the acceptance template,
continued to 1500 mm. Data part: **training events only** (69,783 events), hit centres, energy-weighted, 200
bootstrap resamples by event.

**14.1 Three different causes, by momentum** (each confirmed by removing one component at a time):

| p / GeV | component | where | size | total rise | energy in the rising cells |
|---|---|---|---|---|---|
| 0.05, 0.1 | — | no rise | −0.08 nat / 100 mm | — | — |
| 1 | **track**: its Bragg factor `(1 − z′/R + 0.03)^(−q)` just before the range end R(T) = 755 mm | z′ 722–760 mm, r 1–227 mm | +0.33 nat / 100 mm (track alone +0.79) | +8 % over 38 mm | 1.1e-3 |
| 5 | **end blob**: Gaussian centred at R(T) = 7035 mm, weight < 5e-5, normalised onto the grid so its flank piles up at the boundary | last depth row only, z′ 1431–1444 mm, r 74–329 mm | +0.55 nat / 100 mm | +7 % | 1.9e-4 |
| 20 | **shower**: its lateral profile widens with depth (own slope +0.09 at r ≈ 900 mm), against the far halo's −0.08 | z′ 863–1108 mm, r 711–1182 mm — entirely beyond the grid | +0.016 nat / 100 mm | +2 % | 0 inside the grid (1.4e-3 in the continuation) |
| 30, 50 | same as 20 GeV; momenta outside the training range | z′ 915–1328 mm, r 610–1260 mm | +0.04, +0.02 nat / 100 mm | +6 %, +3 % | 3.9e-4 inside the grid |

Without the track the 1 GeV rise is gone (largest slope −0.08); without the end blob the 5 GeV rise is gone (−0.05);
the r-integrated profile rises for the same two reasons (+0.02 at 735–747 mm at 1 GeV, +0.13 in the last row at
5 GeV) and not at all at 20–50 GeV.

**14.2 z_90 is not the issue.** It is computed from the baseline's own map before the acceptance template, on the
truncated grid: 125, 98, 385, 663, 856, 908, 950 mm — 500 to 1350 mm before the grid end. With the acceptance
template: 153, 118, 420, 678, 839, 880, 915 mm; continued to 1500 mm: 125, 99, 396, 680, 874, 921, 956 mm; z_95:
161, 140, 507, 841, 1049, 1098, 1135 mm. With any of these the 1 and 5 GeV results are identical (+0.33, +0.55);
with z_95 the high-momentum ones shrink (+0.007, +0.026, +0.009) and stay positive.

**14.3 The data.** Momentum bins [0.8, 1.25), [4.5, 5.5), [18, 20) GeV (1683, 3537, 6998 training events); depth
bins of 52 mm; bands all r, r < 30, 30–200, 200–700 mm. Models as the hits can show them (map truncated at each
event's back face, averaged over the bin's events).
- **No depth bin beyond z_90 rises by more than two bootstrap standard errors, in any band, in any of the three
  momentum bins** (12 profiles). The formula's bin-averaged profile has no rising bin either (largest slope −0.16
  to −0.74 nat / 100 mm); the MLP has two, at 1166 and 1217 mm for r < 30 mm at 1 GeV.
- 1 GeV, range end: in u = z′/R(T) (R from the formula's range law, 537–1051 mm across the bin) the data near the
  axis fall steadily and stop beyond u ≈ 1.1; the one upward step (u 0.92 → 0.97, r < 50 mm) is +0.30 ± 0.28. The
  formula has a shoulder there (−0.14 against −0.25 around it) and the edge after u = 1. The data are compatible
  with a Bragg shoulder of the formula's size and cannot resolve a 38 mm rise.
- 5 GeV, last row: 23 % of the training events have their back face beyond 1431 mm (median 1349 mm); the measured
  energy beyond 1400 mm is 1e-5 per event (formula 1e-4, MLP 5e-5). Nothing constrains the formula there.
- 20 GeV: the rise lies beyond 700 mm, where there is no data; in the outermost band 200–700 mm the data fall
  (largest slope −0.09 nat / 100 mm).

**Verdict.** 1 GeV: physics that the formula contains by construction (the Bragg rise of stopping pions), real but
below what the data resolve. 5 GeV: an artefact of the formula's construction in the last grid row, with no data.
20–50 GeV: an extrapolation feature beyond the grid radius (and, at 30 and 50 GeV, beyond the training momenta).
None is an artefact of how the check is evaluated — but the check is pointwise (single momenta, single rows), so
it sees features far below the resolution of the data.

**14.4 Recommendation.** Use the *inheriting* form, point by point: beyond z_90 the model may not rise where the
baseline falls, and may not rise faster where the baseline rises —
`∂_z′ ln ρ_model ≤ max(0, ∂_z′ ln ρ_base)` at every registered momentum, radius and depth band. It is the user's
radial logic applied locally: δ = 0 always passes, a correction is never rejected for the baseline's Bragg bump
or boundary row, and everywhere else the accepted absolute rule stays in force (a correction may rise as long as
the density still falls). Compared with the two forms that exist in the code: the absolute rule fails δ = 0 on this
baseline, and with it every correction that does not happen to cancel the baseline's own rises; the global relative form (`allow_depth_alternative=True`: ∂_z′ δ ≤ 0 everywhere
beyond z_90) passes δ = 0 but forbids every correction that rises with depth, even where the baseline falls by
0.2–0.5 nat / 100 mm. Not recommended: restricting the region (the 1 GeV rise is inside the data region) or using
the r-integrated profile only (it rises too). (Accepted by the user and implemented since: §15.1, anchored to the
fixed reference baseline.)
- **Stage-2 sin/cos check** (PLAN §7, "no sign change of ∂_z′ ln ρ beyond z_90" — the same statement): the baseline
  itself has sign changes (two at 1 GeV, one at 5 GeV), so in absolute form no Stage-2 result could pass either. The
  same inheriting rule applies unchanged — the modulation may not create a rise the baseline does not have — with
  the sign-change counts of model and baseline reported beside it as a diagnostic ("not more than the baseline",
  as §7 of this file already proposed).
- **Not verified:** the Track A finalist may differ (it is refitted, and several rungs change the track and halo
  forms); the data comparison is unfolded (hit centres against cell probabilities) and bin-averaged; the three
  momentum bins are the only ones looked at.

## 15. The user's decisions on the depth rule and the cusp flag (25 Nov 2026, late) — implemented

### 15.1 Inheriting depth rule, anchored to the fixed non-SR baseline

**Rule.** Beyond z_90: `∂_z′ ln ρ_model ≤ max(0, ∂_z′ ln ρ_reference)`, point by point. *Model* = the full refitted
density (jointly refitted baseline plus correction). *Reference* = a fixed non-SR baseline, an explicit, hashed
argument: `checks.decide_tail_criterion(…, reference_kind="track_a_finalist")` (default: the Track A finalist as it
is before any PySR refit) or `"c1_refit_only"`. The record (`sfm.sr.tail_criterion/2`, about 1.3 MB of JSON for
the legacy preview) now holds the reference's **full slope tables** — depth slopes at every momentum × depth
interval × radius, radial log-log slopes at every momentum × depth row × radial interval — its z_90, its
sign-change counts per profile, and:
- `depth.rising_cells`: every cell beyond z_90 in which the reference itself rises (momentum index, depth interval,
  radius index, slope) — the cells where the bound is above zero;
- `depth.rising_regions`: per momentum the depth and radius range, the number and share of cells, the steepest
  slope and where, the total rise at that radius; and, from `refit.tail_criterion_for_baseline`, the
  **attribution**: each component's share of the density in the rising cells, each component's own slope at the
  steepest cell, the largest slope left in those cells when one component is removed, `removed_by`, the energy
  in the rising cells inside the grid and in the continuation, and the track range.
`verify_tail_criterion` refuses an edited record (changing one stored slope changes the hash) and **refuses a table
of a jointly refitted baseline as the reference** (`reference_kind="refitted_baseline"`): that table can only be
passed as `measured_tail`, where it supplies the model side. z_90 is the reference's.

**How it is checked** (`audit.tail_check`). On each segment between neighbouring nodes of the record's grid the
model's finite-difference slope equals the refitted baseline's slope (its table) plus the mean of the
correction's derivative on the segment, which is bounded by an interval enclosure of `∂f/∂z′` (or `∂f/∂ln r`),
refined where the bound leaves little room.
- **Certified:** for the legacy record 26,432 depth points (7 momenta × depth intervals beyond z_90 × 56 radii) and
  36,736 radial points (7 momenta × 128 depth rows × 41 radial intervals), 0.2–0.6 s per equation — the
  correction's contribution on every segment.
- **Evaluated, not enclosed:** the baselines' own slopes (finite differences of their numerical maps and of the
  continuation beyond the grid), and the sign-change counts.
- **Not covered:** momenta between the seven registered ones; depths beyond the last grid row (1444 mm).
- A jointly refitted baseline without its table is not certified (`baseline_frozen=False`, no `measured_tail`);
  `fit_candidate(…, tail_table_fn=)` obtains the table from the refitted baseline model.

**Radial rule, anchored the same way.** A joint refit could bend the baseline radially too (flatten its halo while
the correction itself has `∂_r δ ≤ 0`). Under the alternative of S4 the check is therefore
`∂ ln ρ_model/∂ ln r ≤ ∂ ln ρ_reference/∂ ln r` at every point, with the reference monotone; with a frozen
baseline the continuum statement of S4 (`∂f/∂ln r ≤ 0` for all z′ and x beyond 200 mm) is required as before.
Under "≤ −2" the refitted baseline's own slopes enter the absolute bound.

**A numerical tolerance I had to add (new choice, needs the user's view).** The two comparisons *with the
reference* allow 1e-5 /mm in depth (0.001 nat per 100 mm) and 1e-3 in the log-log slope — the tolerances of the
sign-change count. Without them every correction with an exponentially small rising tail (tanh, erf) inside a
cell where the reference rises fails for a slope of 1e-8 /mm. The absolute rule "≤ −2" and the continuum
statement keep the rounding slack 1e-9.

**Tests.** A correction may rise where the reference falls as long as the model does not; on a reference that
rises itself, δ without a z-part and a falling δ are certified (the absolute rule could not pass them), a rising δ
fails with the worst point inside the reference's rising stretch; a refitted baseline that has learnt to rise
fails with a harmless correction, its table is refused as a reference, and without the table there is no
certificate; a refitted baseline that became 0.3 flatter in r fails "not flatter than the reference", one that
became 0.3 steeper leaves room for a correction that flattens by 0.1; tables on different grids are refused.
Legacy preview record (840 rising cells): `removed_by` = track at 1 GeV (z′ 722–760 mm, range 755 mm), end blob at
5 GeV (last row), shower at 20, 30, 50 GeV (r > 610 mm); `−0.3·tanh(ζ − 2) − 0.1·l1p(ϱ)` is certified on it.

**Reference of record — recommendation: the Track A finalist before any PySR refit.**
1. It exists before any equation is seen and is one object with one hash, so "decided once per baseline" stays
   single-valued; C1 is one run per schedule and seed and exists only after a refit.
2. The purpose is that a refit cannot move the bound. C1 is itself a refit: if more training alone bends the
   baseline upward, anchoring to C1 would let the SR model inherit that bend.
3. It is the baseline S4 named for the radial decision.
C1 should instead be *tested* against the same record (its table as `measured_tail` with a zero correction): if C1
itself violates the rule, the schedule is responsible, not the equation, and that belongs beside the SR verdict.
If the user prefers C1 as the reference, it is `reference_kind="c1_refit_only"` and nothing else changes.

### 15.2 Stage-2 sign-change condition (superseded by §16.2: rising stretches are enforced, sign changes are a diagnostic)

- **Quantities:** `∂_z′ ln ρ` beyond z_90 along depth, one profile per (momentum, radius); `∂ ln ρ/∂ ln r` for
  r > 200 mm along r, one profile per (momentum, depth row); the seven registered momenta.
- **Grid:** the record's own (depth rows, radius nodes) — fixed with the reference, not an argument.
- **Counting** (`checks.profile_sign_changes`): finite-difference slopes; non-finite and zero entries skipped; a lobe
  is a maximal run of one sign; lobes whose largest |slope| is below the tolerance (1e-5 /mm in depth, 1e-3 in the
  log-log slope) are dropped; the count is the number of sign alternations between the remaining lobes. The model's
  slopes are the refitted baseline's table plus the correction evaluated at the nodes.
- **Condition:** `sign_changes(model) ≤ sign_changes(reference)` **per profile**, against the fixed reference;
  enforced with `tail_check(…, enforce_sign_changes=True)` / `certify_finalist(…, stage2=True)` /
  `fit_candidate(…, stage2=True)`, reported (side-by-side text, totals and per momentum) in every tail check.
  The modulation cap of `checks.modulation_amplitude` (≤ 4 inside the data range, 0 outside) is unchanged.
- **Tests:** `−0.78·ζ − 0.0195·sin(40ζ)` (its `∂_ζ` never positive) passes the certified slope rule on a reference
  with one rising stretch and fails the sign-change condition; a ripple of 2e-6 /mm is not counted; a radial
  oscillation on a monotone reference fails.
- **A consequence to know (legacy preview).** `−0.3·tanh(ζ − 2) − 0.1·l1p(ϱ)`, which only falls, has *fewer* sign
  changes than the reference in total (80 against 102) and still more in 6 of 392 depth profiles (2 against 1 at
  20 GeV, r = 947 mm): where a profile begins inside a rising stretch of the reference, a falling correction turns
  `+ −` into `− + −`. Per profile, as implemented, this fails Stage 2. The number of *rising stretches* per profile
  (`checks.profile_rising_lobes`, reported beside the counts, not enforced) has no such edge effect — 54 against
  74, no profile exceeding — and still catches an oscillation inside a rising stretch. Whether the Stage-2
  condition should count rising stretches instead is a question for the user.

### 15.3 Cusp-stall flag

`cusp_stalled` = `cusp_stalled_by_clipping` (§13.5) **or** `cusp_stalled_by_hover`; the second comes from
`refit.cusp_hover_statistics`, which for every `sqa(u)` node evaluates u in all grid cells (× five x values) with
the constants of each step of the final quarter of the run and reports:
- `fraction_exact_zero` (u == 0) and `fraction_near_zero` (0 < |u| < eps), separately; **eps = 1e-3 × the 90th
  percentile of |u| over the grid**; also the smallest |u|, in units of the gap between neighbouring cell values;
- `zero_locus`: none / exact zeros / contour;
- `classification`:
  - **structural-spatial** — a fixed spatial locus given the constants, no constant stalling there: `sqa(ϱ)`
    towards r = 0 (no constant inside the argument at all), or the zero contour of `sqa(ζ − c)` between cells.
    Reported, never flagged.
  - **optimisation-stalling** (raises the flag) — (i) late-step sign-flipping: u changes sign at least twice in
    one cell within the window; (ii) stagnation near u = 0: a constant inside the argument has
    |net change| ≤ 0.5 × path length over the window while min |u| < eps in at least half of its steps;
    (iii) parked exactly on u = 0 in a cell whose u depends on a constant (which then gets no gradient from it).
- **Tests.** Real harness, default clip norm 5: a constant parked 0.04 row spacings above a grid row (truth on the
  row) — 0 clipped steps, near zero in 98 % of the window, net/path 0.30 → **flagged**; the same model with the
  optimum mid-way between rows → structural-spatial contour, not flagged; `sqa(ζ + 2)` → no zero; a fit with
  `sqa(ϱ)` → not flagged. Constructed histories: `sqa(ϱ)` evaluated at r = 0 (exact-zero fraction 1/32,
  structural); back-and-forth across a row (i); exactly on a row (iii); a contour sweeping steadily through cells
  (structural, every cell crossed once); jitter 0.05 spacings from a row (ii) against the same jitter between rows
  (structural); `sqa(0·ζ)` (u ≡ 0, dead constant, flagged).
- **What the synthetic runs showed.** The parked constant sits at a *stationary point* about 0.003 (in ζ) above a
  grid row, with a mean gradient compatible with zero: a local minimum that the kink creates, not a pull that
  Adam fails to follow. One run started far away and stopped next to a row although the true optimum lay mid-way
  between rows.
- **Not guaranteed.** (ii) cannot tell a constant trapped next to a cusp from one whose optimum happens to lie
  that close: for a depth contour on the 128-row grid eps is 0.074 row spacings, so a constant unrelated to any
  cusp is flagged in roughly 15 % of cases. The flag asks for a look, it does not prove a stall. The window is
  the final quarter; thresholds (1e-3, 0.5, half of the steps, two flips) are mine. After a resume the monitor
  covers the steps of that process only.

### 15.4 Interface changes of this round

- `checks`: record schema `/2`; `decide_tail_criterion(…, reference_kind=, rising_annotations=)` (the
  `allow_depth_alternative` switch of §13.4 is gone: the depth rule is always `inherit_reference`);
  `verify_tail_criterion(…, as_reference=)`; `profile_sign_changes`, `profile_rising_lobes`.
- `audit`: `tail_check(…, measured=, baseline_frozen=, enforce_sign_changes=)`;
  `certify_finalist(…, baseline_frozen=, stage2=)`; the certificate names the reference (kind, label, hashes).
- `refit`: `tail_criterion_for_baseline(…, reference_kind=)` with attribution; `fit_candidate(…, tail_table_fn=,
  stage2=)`; result keys `cusp_stalled_by_clipping`, `cusp_stalled_by_hover`; `cusp_hover` has the new fields.
- Nothing used by `scripts/sr_post.py` or the search runner changed.

## 16. PLAN §20, items 4–7 (user decisions of 26 Nov 2026) — implemented

### 16.1 Item 4 — the anchor, and C1 against it

Nothing had to change in the rules: `decide_tail_criterion` / `tail_criterion_for_baseline` default to
`reference_kind="track_a_finalist"` (the frozen finalist before any PySR refit: one object, one hash), both the
inheriting depth rule and the radial alternative take every bound from that record, and a table of a refitted
baseline is refused as a reference (§15.1).

**The one call for C1:** `refit.c1_against_reference(c1_std, c1_ext, tail_criterion, baseline_hash=, c1_hash=,
stage2=False)` — `c1_std` is the control C1 on its grid, `c1_ext` the same model on
`refit.extended_ring_grid(c1_std.grid)`. It makes C1's slope table (kind `refitted_baseline`) on the reference's
momenta and applies the depth rule, the radial rule and (with `stage2`) the rising-stretch condition to C1 alone
with δ = 0; the result has `passed`, `reasons`, the table and a three-line `text`. For a baseline that only exists
as slope tables the NumPy part is `audit.baseline_against_reference(criterion, measured, baseline_hash=)`.
- Tests: the legacy preview model against its own record passes (840 rising cells inherited, 0 of 26,432 depth and
  0 of 36,736 radial points above the bound); synthetic: an identical table passes, a baseline that has learnt to
  rise fails with zero correction; a refitted table cannot be the reference; a wrong baseline hash is refused.
- **A consequence to know.** The legacy preview model with all polynomial coefficients scaled by 1.02 — an
  arbitrary perturbation, not a refit — fails as "C1": 869 depth points and 12,944 of 36,736 radial points, the
  worst radial one at p = 0.05 GeV, z′ = −129 mm, r ≈ 700 mm, where the slope went from −30.5 to −25.4 in a region
  that carries no energy. "Not flatter than the fixed reference at every point" has no notion of where the density
  matters, so with a *jointly refitted* baseline the radial alternative is likely to fail for C1 itself, and then
  for every candidate of that schedule. The real C1 test will show it; if it does, the options are to freeze the
  baseline in the PySR refit (`freeze_base=True`: then only the equation enters) or to ask the user for a
  restriction of the radial comparison. Nothing was changed here.

### 16.2 Item 6 — Stage 2: rising stretches are enforced, sign changes are a diagnostic

- **Enforced** (`tail_check(…, enforce_stage2=True)`, `certify_finalist(…, stage2=True)`, `fit_candidate(…,
  stage2=True)`): `rising_stretches(model) ≤ rising_stretches(reference)` for every depth profile beyond z_90 —
  one profile per (registered momentum, radius of the record).
- **Definition of a rising stretch** (`checks.profile_rising_lobes`), on the profile of finite-difference slopes
  `∂_z′ ln ρ` on the record's grid: (1) non-finite and exactly-zero slopes are skipped; (2) a lobe is a maximal run
  of consecutive slopes of one sign; (3) lobes whose largest |slope| is below 1e-5 /mm are dropped; (4) neighbouring
  surviving lobes of equal sign are merged; (5) a rising stretch is a surviving stretch of positive slope. Model
  slopes = the refitted baseline's table + the correction evaluated at the nodes. Grid and tolerance belong to the
  reference's record, which now also stores the reference's counts (`depth.rising_stretches`).
- **Radial profiles** (`∂ ln ρ/∂ ln r` for r > 200 mm, one profile per momentum and depth row, tolerance 1e-3): the
  analogous quantity is the number of stretches in which the density *increases with r*. It is counted and
  reported, **not separately enforced** — the user's decision is worded for the depth check, and both certified
  radial rules already forbid any radial rise of the model on these segments wherever the reference is monotone (a
  non-monotone reference cannot be certified against at all). *My reading, for confirmation.*
- **Diagnostic that decides nothing:** the raw sign-change counts of the same profiles, depth and radial, total, per
  momentum and per profile, printed side by side (`sign_changes`, `enforced: False`). The old switch
  `enforce_sign_changes=True` now raises with a pointer to `enforce_stage2`.
- The modulation cap of `checks.modulation_amplitude` (≤ 4 sign changes of the modulation inside the data range, 0
  outside) is unchanged.
- **Tests.** Legacy preview record (rising stretches of the reference per momentum 0, 0, 12, 13, 16, 21, 12 = 74;
  sign changes 102): `−0.3·tanh(ζ − 2) − 0.1·l1p(ϱ)`, which had more sign changes than the reference in 6 of 392
  profiles, **is now certified with `stage2=True`** (stretches 54 against 74, no profile exceeding; the diagnostic
  still shows the 6). An oscillation inside the rising shoulders whose slope is never positive,
  `−0.104·ζ − 0.0064·sin(16.3ζ) − 0.1·l1p(ϱ)`, passes the audit and the slope rules and **still fails** (33
  profiles with more stretches). Synthetic: a falling correction on profiles that begin inside the reference's
  bump adds a sign change in all 287 profiles and no stretch, and passes; the oscillation fails; a ripple of
  2e-6 /mm is not a stretch.

### 16.3 Item 5 — tolerances

Accepted as they are (1e-5 /mm in depth, 1e-3 in the radial log-log slope); nothing changed.

### 16.4 Item 7 — the cusp-stall flag is a review trigger, not a rejection

- Checked: nothing in `audit`, `checks` or `refit` reads the flag to decide anything (a test asserts that the two
  audit modules do not mention it); `audit_passed`, `audit_certified` and `finalist_certified` are computed without
  it, and no function drops or demotes a candidate.
- `fit_candidate` now returns **`requires_fable_review`** (True for a flagged run) with
  **`fable_review_reasons`** (which constant, which `sqa` node, clipping or clip-independent), and
  **`review_material`**: the constants, their gradients and the gradient norm along the run (at most 512 evenly
  spaced steps) and the harness's loss history. `refit_summary` prints `REQUIRES_FABLE_REVIEW (diagnostic trigger,
  not a rejection)`; the specification in `register_refit` says to register the row as usual and store the flag.
- **Rule for the pipeline:** every promoted or Pareto-front candidate that carries `requires_fable_review` needs a
  Fable look at its parameter traces and loss curves before it is used. The flag has a chance rate of roughly
  15 % for a depth contour (§15.3), so it is expected to fire on sound fits too.
- Tests: the parked run (flagged at the default clip norm) and the tight-clip run keep `audit_passed = True` and
  `audit_certified = True` and carry the field, the reasons and 160 trace points; the three unflagged runs have
  `requires_fable_review = False`.

### 16.5 Fallback: the legacy formula as the Track A baseline for PySR

Checked on the laptop with the read-only checkpoint (tests skip if it is absent):
- **Works through `sfm.families.build_variant("legacy_centre", core, logA)`** — the component model that reproduces
  the legacy map and has `components`, `core_map`, `grid` and a swappable `provider`:
  `refit.tail_criterion_for_baseline` and `c1_against_reference` (this is what the preview used);
  `targets.fisher_scores` (finite, positive on 24 synthetic events — the provider swap works on a
  `ComponentModel`); `refit.SubstitutedProvider` (k 101 → 99 when a cubic row is replaced by a two-constant
  equation); `refit.CorrectedModel` (k = 101 + k of the equation).
- **Does not work with the object `sfm.legacy` builds** (`WithAcceptance` around the fitted shape): it has neither
  `components` nor `provider`. It can only be the `base` of a `CorrectedModel`; for the tail record and for the
  trajectory targets the `legacy_centre` representation has to be used.
- Model-agnostic, no assumption: `targets.build_residual_target` and `estimate_response` (arrays and hits),
  the audit, the controls C1 / C2, the cusp monitor.
- Assumptions that remain: the baseline must be re-buildable on the extended ring grid with the same parameters
  (true for `build_variant(..., grid=)`); the acceptance template exists on the standard grid only, so the tail
  record is of the density before it; the attribution names (`track`, `end`, `shower`, …) are those of the component
  model; `LastRing.provider` of a Track A wrapper is still unverified.
