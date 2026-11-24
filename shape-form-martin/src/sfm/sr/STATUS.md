# `sfm.sr` — status of the math-author modules at the checkpoint (25 Nov 2026)

Written for whoever continues. Only `operators.py` exists and is tested. Everything else below is **design, not
code**: nothing in it has been run, and no number is measured unless it says so.

## 1. Status per module

| Module | State | Notes |
|---|---|---|
| `operators.py` | **done**, 47 tests pass (`tests/test_sr_operators.py`, CPU, 1.5 s) | torch-free and pysr-free at import (checked in a subprocess) |
| `translate.py` | not started | design in §3 |
| `audit.py` | not started | algorithm and soundness argument in §4 |
| `checks.py` | not started | check list and tolerances in §7 |
| `targets.py` | not started | target (iii), Fisher weights, seed-null in §5 |
| `refit.py` | not started | design in §6 |
| `__init__.py` | empty, as required | |

No test files exist for the five unwritten modules. Nothing outside `src/sfm/sr/` and `tests/test_sr_operators.py`
was touched; no git state changed; the test split was not opened; the GPU was not used.

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
| `sqa` | `sqa(x) = sqrt(abs(x))` | 2 | [0, ∞) | infinite slope at 0; torch gradient there is NaN (inf·0) |
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

## 3. `translate.py` — design (not written)

- **Tree.** Immutable node with `kind`, `name`, `args`, `value`, `fixed`. `fixed=True` marks structural literals
  (template constants such as 0.3, integer exponents): not trainable, not counted in k.
- **Parser.** Precedence climbing over `+ − * / ^ **`, unary sign, calls, numbers (`1e-05`, `1.5f0`, `Inf`, `NaN`),
  identifiers, `#1` placeholders, `p[1]` template parameters. It must also read sympy-style strings, because the
  known-bad equations in `sr_residual.json` are stored that way (`-1.943752e-5/(r - 1*0.017030673)`, `(...)**2`).
  A sign in front of a literal folds into the constant; otherwise it is a `neg` node. `u^2` with a literal 2 →
  `square(u)`; another literal integer exponent → `pow` with a fixed exponent (integer-power semantics); any other
  exponent → real power (positive base required).
- **Formats, from the installed package and from memory of SymbolicRegression.jl — none confirmed by a run.**
  Ordinary: infix for `+ − * / ^` with parentheses around every binary sub-expression, call syntax otherwise,
  outer parentheses stripped (PySR's own test files also show the older `"(f0 + 1.4724599)"`); constants printed
  with `%.{print_precision}g`, so `1.2346e-05` occurs. CSV header `Complexity,Loss,Equation`; multi-output files
  `hall_of_fame_output{i}.csv`; a `.bak` copy beside each. Template: `f = #1 + #2; g = exp(#1); p = [1.0, 2.0]`
  (confirmed only in shape, by `pysr/test/test_main.py::test_equations_with_quoted_constants_are_read`); `#k` is the
  k-th argument of that sub-expression, so translation needs `TemplateSpec.arguments`.
- **Template translation.** Parse `combine` (same grammar plus `name = expr;` statements), substitute each parsed
  sub-expression with `#k` → its argument.
- **Outputs.** NumPy float64 evaluation (optionally recording the largest |intermediate|); torch `nn.Module` built
  on demand with the k free constants as `nn.Parameter`; `complexity` through `operators.tree_complexity`; for a
  template both the search's count (sum over sub-expressions) and the count of the substituted tree; printer using
  `repr(float)`; round trip parse → print → parse and NumPy vs torch ≤ 1e-12 in float64.
- Because the default CSV prints 5 digits, a translated equation reproduces the search loss only if
  `print_precision=17` was set.

## 4. `audit.py` — algorithm (not written, not measured)

Vectorised over boxes with NumPy; every node carries `(lo, hi)`, optionally gradient intervals `(N, d)`, and the
largest magnitude in its subtree.

**Enclosures.** Float64 endpoints, rounded outward after every operation: one `nextafter` step for `+ − * /` and
`sqrt` (correctly rounded), a relative margin of 64 ε ≈ 1.4e-14 plus one step for libm functions (not correctly
rounded, not guaranteed monotone). Then intersected with the operator's mathematical range from `OPS` — needed so
that `tanh` can be certified ≤ 1. Products use the convention 0·∞ = 0 (IEEE 1788), ∞ − ∞ → (−∞, ∞). `ppow` and real
`pow` are monotone in each argument separately, so the four corners give the exact range. `sin`/`cos`: test for an
interior extremum with a slightly enlarged interval. Overflow to ±∞ is a valid outward bound.

**Domain safety (sound).** Forbidden sets: `/`, `inv`: 0; `log`, `sqrt`, real-power base: (−∞, 0] (0 excluded for
`sqrt` because of the infinite slope); `log1p`: (−∞, −1]; negative integer power: 0. Plus a rule that applies to
Stage 1 too: every node must stay below `OVERFLOW_CAP = 1e30` in magnitude. A box is certified when no enclosure
touches a forbidden set dilated by `τ = 1e-5 · max(1, largest magnitude in the denominator's subtree)`. Flagged
boxes are bisected (widest relative side) to a minimum size or a box budget; whatever is left **rejects**.
Reported separately: *proven* (sign change of the denominator between corners of a box on which its own subtree is
safe → a zero by the intermediate value theorem; or |denominator| ≤ τ at a sample point) and *not proven safe*.
Two margins, to be stated as such: outward rounding makes the float64 enclosures rigorous; τ is an explicit,
heuristic allowance for the float32 evaluation in the refit.

*Soundness argument.* Each elementary enclosure contains the exact range (rounding margin ≥ library error), so by
induction the enclosure of every node contains its range on the box. A restricted operator is certified only if
that enclosure is disjoint from its forbidden set. Bisection covers the parent (closed boxes sharing a face), so
the certified leaves cover the box. Every other outcome rejects: over-estimation costs false rejections only.
Limits: the statement is about the expression with the constants as parsed, in real arithmetic — **the refit moves
the constants, so the audit has to be repeated on the refitted equation**; the plan only says "before any refit".

**Magnitude bound (tight).** (0) Coarse grid over the whole box: an evaluation above the limit or non-finite fails
at once. (1) For each box the natural enclosure intersected with the centred form
`f(c) + Σ_i G_i(B)·[−h_i, h_i]` (`f(c)` as a thin interval, `G_i` from forward-mode interval derivatives). Certified
if `sup|F| ≤ limit·(1 + 1e-9)`. (2) Otherwise evaluate at centre and corners: a value above the limit **fails**
(witness kept). (3) Otherwise bisect the side with the largest `sup|G_i|·w_i`. (4) At the minimum size
(2⁻¹⁰ of each side: 2.2 mm, 1.5 mm, 0.005 in x) or at the budget: `n^d` evaluations at sub-cell centres, margin
`Σ_i sup|G_i|·w_i/(2n)`; a value above the limit fails, `max + margin ≤ limit` certifies, the rest is **undecided**
with its location. Derivative ranges needed: `tanh′`, `atan′ = linv`, `erf′` even and decreasing in |x|;
`softplus′` = sigmoid; `l1p′ = sign(x)/(1+|x|)`, `[−1, 1]` across 0; `sqa′` unbounded across 0 (centred form then
unavailable, natural enclosure still valid); `linv′ = −2x/(1+x²)²` with extrema ∓3√3/8 at ±1/√3;
`∂_x ppow = y·ppow·l1p′(x)`, `∂_y ppow = ppow·l1p(x)`.

**Gauge.** `sup|δ| ≤ 1 nat` is not defined for a search with the gauge-invariant loss: `f` and `f + c(x)` are the
same model. Planned: drop top-level additive terms that depend on `x` alone, then bound
`g = f − f(zt_ref, rh_ref, x)` with a reference point (default `(1, 1)` = 260 mm, 21.9 mm). This is exactly
gauge-invariant. For the normalised correction `δ_eff = f − ln E_base[e^f]`: `|δ_eff| ≤ osc_x(f) ≤ 2 sup|g|`, so
limit 1 on `g` guarantees `|δ_eff| ≤ 2`, limit 0.5 guarantees 1. `δ_eff` itself needs the baseline and belongs in
`refit.py` / `checks.py`. Templates have a fixed form and use `gauge="none"`.

**Trajectories.** Same machinery on `x ∈ [−4.2, 1.4]`: finiteness, numerical and enclosed value range on the full
interval against `[−1, 1]`.

**Required tests.** Reject `c/(r − 0.017)` in metres and `c/(rh − 0.7763)`; the four equations of
`sr_residual.json`; pole between sample points, on a box edge (`1/rh`), removable singularity (`x/x` rejects by
design), `ppow(zt, 400)` (overflow), `exp(square(zt)*rh)`; accept smooth Stage-1 expressions. **The
false-rejection rate on random valid expressions was requested and has not been measured.**

## 5. `targets.py` — definitions (not written)

**Target (iii).** For slice `s` with momentum `p_s`:
- (a) Teacher: probability mean of the ensemble, `ln P̄_T = logsumexp_m ln P_m − ln M`; `ρ = P/V_cell`.
- (b) Smoothing by a train-estimated response matrix, energy-weighted, per slice:
  `R_s[c′, c] = Σ_{i: cell(h_i)=c′} w_i · (1/K) Σ_k 1{cell(h_i + R_e u_ik) = c} / W_s(c′)`, with
  `W_s(c′) = Σ w_i` over training hits of the slice's events centred in `c′`.
  `y_s(c′) = ln (R_s ρ^T_s)(c′) − ln (R_s ρ^B_s)(c′)`, both maps evaluated exactly at `p_s`.
  Chosen over the hit-level mean of `ln g_T − ln g_B` because it needs the maps at 12 momenta only, is
  deterministic given `R`, and is reused unchanged for the seed-null. The two differ by a Jensen term (mean of logs
  against log of mean within a cell) that cancels to first order in the difference; to be measured on a subsample.
- Abscissae: not the cell centre but the centroid of the tilted response,
  `S_s[c′, c] = R_s[c′, c] ρ^B_c / (R_s ρ^B)(c′)`, `(z̄, r̄)(c′) = Σ_c S_s[c′, c]·(z_c, r_c)`. `S` is the exact
  derivative of `ln R(ρ_B e^f)` at `f = 0`, so this removes the first-order error; near the axis the box average
  of r for a hit at r ≈ 0 is about 18 mm, not 0.
- Weights: `ω_s(c′) = W_s(c′)/N_s` (energy share). Then the within-slice ω-weighted variance of the error is
  ≈ 2·KL between the corrected baseline and the teacher, both folded (second order).
- (c) Masks: z′ < 1150 mm, not the first z row, not the last ring (ring 0 kept), `Ā > 0.05`, a minimum hit count
  per cell (about 30), standard error of the teacher mean `std_m(ln R ρ_m)/√M ≤ 0.05 nat` (skipped and flagged for
  a single teacher), `p_s ≥ 0.0895 GeV` enforced.
- (d) Twelve slices: events in quantile bin s, `p_s` = the bin median. About 650 cells per slice, fixed-size
  systematic sampling with inclusion probability `π_c ∝ ω_c^γ` (γ = 0.5 proposed, so tail cells are represented;
  units with π ≥ 1 taken with certainty), Horvitz–Thompson weight `ω_c/π_c`. The residual itself is not used in
  the design.
- (e) Gauge: the stored `y` has HT-weighted mean zero per slice, so the plain weighted L2 of `f ≡ 0` is the null
  loss.
- Approximation when the fitted correction is applied to the unsmoothed log-density: `Φ(f) := ln R(ρ_B e^f) −
  ln R ρ_B` is replaced by `f` at the centroid — exact to first order, wrong by the curvature of `f` across the
  tilted response. Measured as the weighted within-slice rms of `Φ(f)(c′) − f(z̄, r̄, x_s)` against the rms of `y`
  and of the fit error. The likelihood refit uses the true fold, so the approximation affects the search target
  only.
- Caveat to raise: slices at event quantiles are uniform in p (the sample is balanced in p), so the first slice
  spans roughly x ∈ [−1, 0.1], where AUDIT flags 12–13 put the headroom.

**Seed-null (C3).** `y_null = κ·[ln R ρ̄_A − ln R ρ̄_B]` on the same rows and weights, `κ = sqrt((1/M)/(1/a + 1/b))`
(= ½ for equal halves), so its noise variance equals that of the full-ensemble teacher.

**Target (i), Fisher weights.** `ΔNLL ≈ ½ ∫ n(x) Σ_jk Δ_j F_jk Δ_k dx`. The curvature is the *expected Hessian*,
which for the cross-entropy `−Σ_i w_i ℓ_i` (`ℓ_i = ln g_i − ln M_e`) is `F_jk(x) = E[Σ_i w_i ∂_jℓ_i ∂_kℓ_i | x]`;
the covariance of per-event scores is **not** the same thing, because the event NLL is an energy-weighted
pseudo-likelihood. Diagonal weights `w_gj = n_g·F_jj(x_g)`, `n_g` = event fraction at point g, so that
`Σ_g w_gj Δ_j(x_g)² ≈ 2·ΔNLL_j`. Estimator from training batches: `F̂_jj = mean_e Σ_i w_i (∂_jℓ_i)²`, `∂_jℓ_i` by
central differences of the per-hit folded log-likelihood with common fold points (a provider wrapper that adds a
shift to column j). Assumes E[score | x] = 0 and neglects off-diagonal terms. Points at event quantiles of x;
layout `X (N, 1)`, `y (N, n_out)`, `weights (N, n_out)` normalised to mean 1 per output with the factor to nats
stored; guesses = weighted-least-squares cubics in Horner form, `list[list[str]]`.

**`.npz` schema** (planned): `X`, `y`, `weights`, `variable_names`, `slice_id`, `slice_p`, `slice_x`, `cell_iz`,
`cell_ir`, `pi`, `omega`, `seed_se`, `kind`, `loss`, `schema_version`, `meta_json`; rows sorted by slice.

## 6. `refit.py` — design (not written)

- `log_softmax(log P_base + f(zt, rh, x))` at the cell centres; all free constants trainable; a non-finite value
  raises; also a gradient-finiteness check (`sqa` at exactly 0 gives a NaN gradient).
- Trajectory substitution: replace chosen columns of a provider by translated expressions of x; k = provider count
  minus the replaced rows plus the expressions' constants.
- C1: the baseline alone on the same schedule. C2: tensor polynomial in `(zt, sqrt(rh), x)`, first k monomials in
  graded order excluding pure powers of x (invisible to the softmax), inputs rescaled to [−1, 1], zero start.

## 7. `checks.py` — checks and proposed tolerances (not written)

NumPy only; a grid baseline is extended by a power law in r and an exponential in z′ fitted to its last interior
rings and rows.

| Check | Number | Proposed tolerance and reason |
|---|---|---|
| tails | largest `d ln ρ/d ln r` beyond 700 mm; largest `∂_z′ ln ρ` beyond 1450 mm | < −2 and < 0; also "not worse than the baseline" |
| extrema along r and z′ | count per profile, corrected against baseline | turning point counted at prominence ≥ 0.01 nat (below the 0.02–0.05 nat reproducible ringing of AUDIT §3.5) |
| sign changes of `∂_r ln ρ` (r > 200 mm), `∂_z′ ln ρ` (beyond z₉₀) at 0.05, 0.1, 1, 5, 20, 30, 50 GeV | count | zero, and not more than the baseline |
| r50, r90, z50, z90 at 20 → 30 → 50 GeV | values | no reversal beyond 0.5 mm; direction compared with 10 → 20 GeV |
| modulation amplitude | sup |m| on the extended box | ≤ 0.3 nat |
| modulation sign changes | per z′ and r profile | lobes with peak ≥ 0.003 nat; ≤ 4 inside the data range, 0 outside |
| period | nearest 1650/k, relative distance | match within 3 %; > 570 mm = trend; only k = 3…8 can match |

## 8. Points where the plan is ill-posed (to raise)

1. `sup|δ| ≤ 1 nat` is gauge-dependent (§4).
2. The literal Stage-2 template is bounded by 1.3, not 1.
3. "Integrable tails" follows from the baseline's tail once `|δ| ≤ 1`; the slope criterion is a regularity
   condition and the baseline itself may fail it.
4. Checks at 0.05, 30 and 50 GeV evaluate the baseline outside its training range; only "no worse than the
   baseline" is well defined there.
5. Period check: bands of ±5 % around 1650/k cover 53 % of 190–570 mm, ±3 % about 32 % — weak specificity.
6. "Modulation" is undefined for a free-form Stage-2 expression; proposed: f minus f with every sin/cos set to 0.
7. "Constants stable across seeds" has no meaning across structurally different expressions; compare the fitted
   functions on a common grid.
8. Fisher weights: expected Hessian, not score covariance (§5).
9. `ppow` exponent cannot be forced constant by `constraints` at unit variable complexity (§2).

## 9. To confirm with Julia (Julia was not installed in this session)

`operators.UNCONFIRMED` lists them. In short: custom operator strings accepted and type-stable; the three
dictionaries accepted with custom names; argument-size limits counted with operator weights; `ppow(·, variable)`
admitted by `(7, 1)`; elementwise loss normalised by `Σw`; the gauge loss (fields `X`, `y`, `weights`, `n`,
`eval_tree_array` on a tree and on a template expression) equal to `gauge_loss_numpy` on exported rows;
`batching=False`; template combine arithmetic with literals and `^2`; template complexity; `print_precision` in the
CSV; multi-output `weights (N, n_out)` and `guesses` as `list[list[str]]`; the string formats of §3.
