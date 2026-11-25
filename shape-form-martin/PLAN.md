# Pion shower-shape campaign in `shape-form-martin/` — Research & Engineering Plan

*Approved on 24 Nov 2026; Phase 2 is in progress. Section 13 records amendments made after Gate 0 — where it differs
from the sections above, section 13 applies. Gate reports live in `results/gates/`.*

## 1. Context

The pion average shower density `ρ(z', r | p)` is best described by a 4×32 MLP (test NLL 18.1208) and, in closed form, by
the 101-parameter component formula (18.1235). Under the **original report metric** on the pion sample the campaign asks:
(1) can statistical distributions, with parameters predicted by a small network, model the density more accurately or
more cleanly; (2) can PySR with a larger, physically audited operator set close the remaining gap; (3) can any
leakage-free improvement beat the benchmarks. All new work lives in `shape-form-martin/`.

**Expected outcome, stated up front.** θ-blind gains over the MLP of 0.0005–0.0015 nat are plausible, mostly from
ensembling and from training with the evaluation objective; a gain ≥ 0.003 is very unlikely. Track A will most likely end
at a *cleaner* formula (fewer constants at equal NLL, physically literal parameters), not a better one than the MLP.
Residual PySR is likely worth ≤ 0.0005; the sin/cos stage is expected to fail its gates. Nulls are reported as such.

## 2. What the inspection established

Rows marked † come from data-free computations on stored checkpoints by the design review; they are re-measured in M1.

| # | Finding | Consequence |
|---|---|---|
| 1 | Metric of record: `common.event_nll` / `test_nll` (`shape-form-original/common.py:293-314`). Piecewise-constant lookup on the 128×32 grid, K random points per hit (K=16 in training, K=64 in evaluation, CUDA generator seed 12345), `M_e` = model mass with `z' < z_b` (flat cut). | Every headline number uses this code unmodified. |
| 2 | Split (`common.py:216-232`): 69,783 / 14,950 / 14,956 events, per momentum bin, `default_rng(1)`. Per-event test NLLs of all benchmarks are stored in `shape-form-proton/results/particles/pion/test_v3_nll.npz`. | Parity is checked per event. |
| 3 | Validation: fine-tuned MLP 18.118084; component formula by degree 18.12278, 18.12119, **18.12041**, 18.12040, 18.12037, 18.12031. | Polynomial-in-`ln T` bias beyond degree 3 is ≤ 1e-4. The gap (0.0023 val, 0.0026 ± 0.0003 test) is functional-form bias in (z', r). |
| 4 | † At map level KL(MLP‖formula) = 0.0098, of which 0.0012 is the longitudinal marginal and 0.0086 lateral (mostly r < 30 mm). After folding with the detector cell only 0.0026 remains. Two independent MLP fits differ by a map-level KL of 0.003–0.006. | A longitudinal family swap can gain at most ~0.001. About 73 % of the map difference is invisible to the metric; single-network maps carry seed noise of the same order as the formula–MLP difference. |
| 5 | The Monte-Carlo cell average has a Jensen bias of ≈ 0.16–0.24/K nat († synthetic; logs of the same recipe read ≈18.127 at K=16 and ≈18.120 at K=64). Training and checkpoint selection run at K=16, evaluation at K=64. | The training objective penalises sharp structure more than the metric does: a real lever for the networks. The bias is model-dependent (≈1e-4 between formula and MLP at K=64). |
| 6 | Earlier PySR (`scripts/sr_residual.py`): operators `{+,−,*,/}` ∪ `{exp, square}`, 10 minutes, maxsize 18, 6,000 unweighted points, target = an older 119-parameter formula, no control for pions. Its equations contain `c/(r − 0.017)`: the pole sits 0.1 mm from the centre of ring 3 (16.93 mm), i.e. a one-ring spike that point sampling at 32 radii cannot see. Gain −0.0003 ± 0.0001. | Stage 1 must be pole-free by construction; sampling and controls are redesigned. |
| 7 | Lane `F1_lateral` (proton, slab metric) already surveyed 24 lateral forms incl. K0, Student-t with free ν, generalised gamma, stretched exponential; Gauss + Cauchy + exp/r won and is in the component formula. Lanes F2/F3 and the Bock variant tried exponential-start ⊛ gamma (worse for pions by 0.004–0.011). Lane G2 tried MLP × `Ā` on the proton: +0.00005 ± 0.00046. | Much of the requested family list is a replication on pions. New: EMG, generalised gamma / inverse Gaussian / log-normal in depth, Burr XII in r, exact cell integration, a detector-frame halo. |
| 8 | `abar.pt` was built from training events with front **and** back faces, while `M_e` already cuts at the back face. | The acceptance template is varied (front only, front+back, back-corrected). |
| 9 | For a θ-blind model the metric is proper in the bulk (z' < 1220 mm). A model that knows θ can move mass beyond `z_b(θ)`: it is not counted in `M_e` but raises the density for hits behind the flat cut. † Up to ~0.01 nat of fake gain — the same mechanism that made the slab pion MLP collapse (99 % of its mass in one cell). | The θ track needs constrained architectures and a bulk-restricted headline (§6). Deep-tail and border-cell gains are not validated by NLL (§6 guards). |
| 10 | Only `05_mlp_pion_copy.ipynb` imports `common.py`; `summary.md` documents the old path (lines 45, 209–219, 311–313). `shape-form-proton` never references `shape-form-original`. | The move breaks only documentation. |
| 11 | `shape-form-proton` scripts hold another machine's absolute paths, but the modules (`shapelab`, `particle_shape`, `proton_shape`, `families`) import cleanly with `C = vars(common)`; `shapelab.build_final` works that way. `setup_particle` would write a cache there. | Read-only adapter with bytecode writing disabled; `setup_particle` is never called. |
| 12 | One RTX 4060 Laptop GPU (8 GB), 24 threads, RAM tight (`common.load_data` through pandas peaks near 16 GB). PySR 2.7.3 installed, Julia not yet downloaded. `Showers` does not keep `event_number`. | Lean pyarrow cache that stores event numbers; serial GPU queue; one deliberate Julia bootstrap. |
| 13 | Data: 49.1 M hits, 99,689 events after cuts, p down to 0.0008 GeV (65 events < 0.1 GeV), θ = 0.052–0.628 rad, 16.7 % of events have `z_b` beyond the grid. | Used for the units audit and all extrapolation checks. |

## 3. Milestone 0 — workspace and parity

```
shape-form-martin/
  common.py                  git mv, logic unchanged (metric of record)
  05_mlp_pion_copy.ipynb     git mv; settings cell reads sfm.paths and the Parquet pattern
  src/sfm/
    paths.py                 repo root from __file__; env overrides SFM_DATA_DIR / SFM_RESULTS_DIR; no absolute paths
    data.py                  pyarrow column reader, compact cache with event numbers, θ, φ, split; test split in its own file
    metric.py                generalised NLL (K, fold scheme, bulk restriction); equals common.test_nll at defaults
    legacy.py                read-only adapter: component formula, mlp_std.pt, abar.pt
    stats.py                 paired Δ ± SE, 1-SE rule, bootstrap by event, winner's-curse correction
    families/                longitudinal.py, lateral.py, cellint.py, mixture.py, halo_det.py
    models/                  paramnet.py, field.py, acceptance.py, ensemble.py
    train.py                 resumable fit loop, deterministic seeding
    registry.py              evaluate(): the only writer of results/registry.jsonl
    sr/                      targets.py (torch), search.py (no torch import), operators.py, audit.py, translate.py, refit.py
  configs/  scripts/  tests/
  results/{m0_parity,m1_audit,m2a_dist,m2b_mlp,m2c_theta,m3_sr,final,gates}/
  cache/ (git-ignored)   .gitignore   README.md
```

Steps: create branch `shape-form-martin`; `uv add --dev pytest nbclient`; record SHA-256 of `shape-form-original/` and
`shape-form-proton/`; move the two files and fix the three path references in `summary.md`; build the cache from the
Parquet file with pyarrow (same cuts, ordering and dtypes as `Showers.from_dataframe`); run the parity script.

**File move and staging rules.** Check `git ls-files --error-unmatch <path>` for each file first: tracked → `git mv`,
untracked → plain `mv`. (Checked on 24 Nov: both files, `summary.md`, `pyproject.toml` and `uv.lock` are tracked.)
Never `git add -A` or `git add .`. Stage by explicit path only: files inside `shape-form-martin/`, `pyproject.toml`,
`uv.lock`, `summary.md` (if tracked), and — only as the removal side of the rename — the two old paths in
`shape-form-original/`. Before every commit `git diff --cached --name-only` must list nothing under
`shape-form-proton/`, no data file and no other `shape-form-original/` path.

**Gate 0 (Fable)**
- Event-number sets of the three splits disjoint; counts 69,783 / 14,950 / 14,956; 70/15/15 per bin within rounding.
- `mlp_std.pt`: test 18.120842, val 18.118084; component formula (`new3_std_n3.pt` × `abar.pt`): test 18.123459,
  val 18.120407; means within 5e-5 and per-event test arrays within 1e-4 of `test_v3_nll.npz` (`std:mlp_ft`, `std:new`).
  Fallback if the CUDA random stream differs from the original machine: correlation > 0.9999 and means within
  Monte-Carlo noise count as statistical parity; all deltas are then paired against locally recomputed arrays only, and
  the report MLP / CNN numbers (weights not in the repo) are quoted unpaired.
- `sfm.metric` at defaults equals `common.test_nll` exactly on validation; if RAM allows, cache tensors equal one
  `common.load_data` run bit for bit.
- Units (GeV, mm, rad, ns; m_π = 0.13957 GeV); `Ā` recomputed from training events only and compared with `abar.pt`.
- **Low-momentum boundary.** Exact `p_min`, `T_min`, `x_min` and the number of events below 0.1 GeV per split *after
  cuts* (the raw file reaches 0.0008 GeV with 65 events below 0.1 GeV; the stored `lnp0 ± lnp_s = 0.2912 ± 2.7045`
  implies a training minimum near 0.09 GeV, so the cuts appear to remove most of them — to be confirmed). Code check of
  flooring: on inspection `particle_shape.kinetic` and `x = (ln T − v0)/vs` have **no** floor (only `range_mm` has a
  softplus lower bound of 1 mm), and the MLP input `ln p / 3` is unbounded; Gate 0 confirms this and evaluates both
  benchmarks at the lowest validation momenta and at 0.0005 GeV for finiteness.
- Protected directories unchanged (checksums). A 200-step timing probe calibrates the GPU budget.
No new model is trained before this gate passes. These are the only test evaluations before Gate 3.

## 4. Milestone 1 — diagnostic audit (validation and training data only)

Both models are scored on identical fold points, so every decomposition is exact and paired.

| ID | Diagnostic | Answers |
|---|---|---|
| D1 | `d_e = NLL_formula − NLL_MLP` per event: mean ± SE, tails, by momentum (20 bins, fine bins below 2 GeV), θ quintile, `z_b`. The sub-100 MeV events are inspected one by one: leverage on the mean NLL and on paired differences (leave-them-out deltas), hit multiplicity, visible energy, depth reach, and whether they look like stopped or decayed pions. | Where in (p, θ) the gap lives; whether a handful of extreme low-momentum events drive any result. |
| D2 | Exact split `d_e = Σ_i w_i[ln g_i^MLP − ln g_i^F] + [ln M_F − ln M_MLP]`, hit term on the native grid per momentum bin. | Where in (z', r). |
| D3 | Marginal/conditional swap without training: `L_MLP·T_F` and `L_F·T_MLP` scored on validation; map-level KL split re-measured. | Longitudinal vs lateral share (confirms row 4). |
| D4 | Data vs each model: energy-share pulls in (z', r) per momentum and θ bin, bootstrap SE by event. | Where both models fail (θ-blindness, backward cone at z' < −20 mm, r > 200 mm). |
| D5 | Per-event model mass in front of the tilted entrance plane and beyond `z_b`; NLL by θ. | Size of the θ-blind penalty; exposure to the back-face artefact. |
| D6 | (a) K ∈ {16, 64, 256, 1024}, three fold seeds: NLL(K), 1/K extrapolation, K-dependence of the gap; (b) formula at cell centres vs exact cell integrals; (c) lookup vs continuous evaluation for the fixed formula; (d) energy share of points clamped at the grid border. | Jensen bias and metric noise; whether continuous evaluation qualifies (rule: ≥ 3e-4 nat). |
| D7 | θ-blind ceiling without a histogram (a hit histogram is already folded): cross-validated score test on per-cell logit offsets, MLP + zero-initialised residual table, learning curve at 25/50/100 % of train. | How much any 128×32 map could still gain. |
| D8 | Zero-training ensemble probe: probability average of `mlp_std.pt` and the three local `mlp_L3xH*.pt`. | First estimate of the ensemble gain. |
| D9 | Roughness of `ln P_MLP`; periods compared with its sin/cos feature periods (3300/1650/1100/825 mm in z'). | Ringing a formula should not copy. |

## 5. Milestone 2A — statistical distributions (Track A)

**Model.** `ρ = Σ_c π_c · L_c(z') · T_c(r | z')`. Cell masses by exact ring integration: 5 Gauss–Legendre nodes per z-cell
times exact differences of the radial survival function `S(r) = 1 − F(r)` (the 2πr Jacobian is exact; the 1 mm axis
regulariser disappears), `ln ΔS` via `log1mexp`; each component normalised on the truncated grid; then
`log P = log_softmax(logsumexp_c(ln π_c + ln P_c) + ln A)`. Parameters: the 22 momentum-dependent ones from `ParamNet`
(`x = (ln T − v0)/vs` → tanh MLP, width 16, weight decay 1e-4, warm-started in function space from the cubic polynomials,
≈ 700 weights), the 21 constants stay scalars. Links: softplus + floor for scales and shapes, sigmoid / stick-breaking for
fractions, cumulative ordering of widths (σ < R < λ) against label switching.

**Screening against the MLP field first (no data, minutes):** every L × T combination and K = 1…5 mixtures by KL to the
ensemble-mean map at event-quantile momenta, restricted to data-constrained depths; 1-D KL and PP plots for the
longitudinal marginal and radial conditionals. Pre-filter only; ranking is by likelihood on hits.

**Ladder** (training K=16, validation K=64, last-iterate weights)

| Rung | Change | Isolates | Prior status / expectation |
|---|---|---|---|
| A0 | existing formula refitted in the new harness | parity (val 18.12041) | — |
| A1 | Θ(ln T) from `ParamNet` | polynomial-in-`ln T` bias | 0 … −2e-4 |
| A2 | exact cell integration (base for all later rungs) | binning error (entry peak σ_z 3–5 mm vs 12.9 mm bins) | ±2e-4; makes parameters literal |
| A3 | acceptance template: front only / front+back / back-corrected | back-face double counting | ≤ 3e-4 |
| A-L | diagnostic: formula × free row factor `c(z', ln T)` | longitudinal ceiling | −0.0005 … −0.0012 |
| A-T | diagnostic: formula's z-marginal × free `T(r | z', ln T)` | lateral ceiling | ≤ −0.0015 |
| A-Z | diagnostic: current lateral family, parameters from a network in `(z', ln T)` | depth dependence vs radial family | — |
| A4 | **EMG** = Exp(λ) ⊛ N(μ, σ) as shower depth profile, λ tied to the track attenuation and untied; `log_ndtr` form, λσ ≤ 30 | named candidate; 3 instead of 5 parameters | not tried by name, near-nested in the plateau: 0 … +0.001 worse |
| A5 | **generalised gamma** `s^(d−1) e^(−(s/a)^p)` (mode, d/p, p) | stretch exponent | new; P(beats plateau) ≈ 15 % |
| A6 | **inverse Gaussian**, **log-normal**, **Exp ⊛ Erlang**, **Burr XII** in depth | remaining named families | new except Exp ⊛ gamma (tried, worse); expected null or worse |
| A7 | **Burr XII** radial body `ln S = −k·log1p((r/R)^c)` (Student-t / Lomax at c = 2 / 1) | free tail and axis exponent | new; Student-t alone was inferior in F1; ±0.0005 |
| A8 | **Weibull / stretched-exponential** halo `ln S = −(r/λ)^b` | halo shape | F1 found b ≈ 0.45–0.65 useful; 0 … −5e-4 |
| A9 | **detector-frame halo**: `E_{θ,φ}[1{in slab}·e^(−z_trans/L)]·g(r; R)`, two parameters and one weight | backward cone, the largest coherent residual | new; −0.0003 … −0.0010 |
| A10 | pure mixture: K = 2…5 components of [best L] × [best T], no hand-built physics terms | is a generic neural mixture competitive? | exploratory |
| A11 | best combination, Fisher-information pruning of dead parameters, K=64 training, 3 seeds | finalist | 101 → ~85 constants at equal NLL |

K0, generalised gamma in r and Beta-prime stay in the KL screen only (F1 negatives; no differentiable CDF in torch).
Pre-registered primary comparisons: A4, A5, A7, A8, A9, A10; the rest is exploratory. A rung is *kept* at Δval ≤ −3e-4 and
≤ −2 SE against A2, otherwise recorded as null. The finalist is the simplest model within 1 paired SE of the best; it
counts as "cleaner" only if it is ≥ 3e-4 better than 18.12041 or has ≥ 10 % fewer constants at equal NLL.

## 6. Milestone 2B — beating the MLP (θ-blind) and the θ track

Evaluation is always the unmodified `common.test_nll`. Every variant is compared with **B0**, not only with `mlp_std.pt`.

| ID | Hypothesis (in order of expected gain) | Experiment | Expectation |
|---|---|---|---|
| B0 | control | continue `mlp_std` with the identical schedule | — |
| B1 | ensemble variance | probability average of 5 seeds (reported as 5 × 3,841 weights); then distilled into one 3,841-weight network | 0.0003–0.0012 |
| B2 | objective mismatch | paired fine-tunes from `mlp_std`, K=16 vs K=64 i.i.d. vs stratified points; last-iterate | 1e-4 – 6e-4 |
| B3 | optimisation | 12k-step scratch run with EMA | ≤ 1e-4 |
| B4 | analytic acceptance | logits + `ln A`, function-space warm start, three templates | −1e-4 … +3e-4 (null on proton) |
| B5 | smoothness penalty | one scratch pair, penalty 0 vs 1e-4 | ≤ 1e-4 (penalty at optimum ≈ 2e-5) |
| B6 | capacity under the B2 recipe | 4×32, 5×32, 4×64; 1-SE rule | flat |
| B7 | resolution | only if D6(c) ≥ 3e-4; separate labelled table, and projected to the grid for the strict number. Under the strict metric it cannot exceed the free 128×32 map. | 0 by construction |

Promotion: Δval ≤ −5e-4 and ≤ −3 SE against `mlp_std`, better than B0 by > 1 SE, reproduced by a second seed, guards passed.

**Guards against metric artefacts** (own tables; the main table holds the report metric only): **G-K** re-evaluation at
K=1024 (sign holds, ≥ 50 % of the gain persists); **G-bulk** hits and normalisation restricted to z' < 1150 mm without
the last ring (≥ 50 % persists); **G-region** ≤ 50 % of the gain from deep, border or `Ā ≤ 0.3` cells; **G-mass** mass
beyond `z_b` and in `Ā ≤ 0.01` cells; **G-seed** replication.

**θ-conditioned track (approved; `results/m2c_theta/`, starts after the θ-blind finalists are frozen).** θ is the true
incidence angle of the same event — an input condition like p. Because of finding 9 the arms are constrained:
Θ1 logits + ln of the event's exact front-face acceptance (no new parameters); Θ2 adds `u = 1/cos θ − 1.1` as an input,
gated to the entrance region `σ((300 mm − z')/50 mm)`; Θ3 unrestricted, run once as a diagnostic of the back-face exploit.
Headline for this track: G-bulk; the report metric is quoted beside it with the mass-beyond-`z_b` monitor. The θ-blind
harness accepts only `forward(lnp)`; nothing from this track enters a θ-blind table or delta. Legitimate gain expected:
0.006–0.012 nat.

**Gate 1 (Fable)** — densities/CDFs against `scipy.stats` and high-order quadrature; masses sum to 1 within 1e-6; finite
gradients at parameter extremes; no `log(softmax)`; identifiability of fitted trajectories; training curves;
loss–metric alignment (K, checkpoint policy); every paired SE recomputed from stored per-event arrays; selection
reproduced independently; guards.

## 7. Milestone 3 — PySR

Prerequisites: Gate 1 passed; Julia bootstrapped once; search processes never import torch (targets are exported to
`.npz` first; the old log shows the juliacall-after-torch warning).

**Targets**
- (i) Trajectories: the momentum-dependent parameters of the Track-A finalist as functions of `x`, one multi-output search
  per seed, weights = momentum density × diagonal Fisher information (loss ≈ 2·ΔNLL), cubic polynomials passed as `guesses`.
- (ii) Bounded log-factors learned in Track A (row factor, depth dependence of lateral parameters) — never raw densities.
- (iii) Residual `δ = ln P̄_teacher − ln P_baseline` on the **final** baseline, teacher = ensemble-mean map, both maps
  smoothed with the cell-box response (the metric cannot see sub-cell structure), masked to the bulk, interior rings,
  `Ā > 0.05` and low seed variance. Twelve momentum slices at event quantiles, ~650 cells per slice by
  probability-proportional sampling with Horvitz–Thompson weights. Dimensionless inputs `ζ = z'/260 mm`, `ϱ = r/21.9 mm`, `x`.
  Loss: within-slice weighted variance of the residual error (invariant to the additive gauge the softmax ignores, ≈ 2·KL).

**Stage 1 vocabulary — the requested operators in domain-safe form**

| Requested | Realisation | Why |
|---|---|---|
| `+`, `−`, `*`, `exp`, `square`, `tanh`, `atan`, `erf`, `erfc`, `softplus` | as is (`softplus(x) = max(x,0) + log1p(exp(−|x|))`) | entire functions |
| `/`, `inv` | `linv(x) = 1/(1+x²)` and products with it | sign-definite denominators: no poles possible |
| protected `pow` | `ppow(x, y) = (1+|x|)^y`, exponent a constant | positive base |
| `log` / `log1p`, `sqrt` | `l1p(x) = log1p(|x|)`, `sqa(x) = sqrt(|x|)` | defined everywhere |

- `nested_constraints`: saturating functions (`tanh`, `atan`, `erf`, `erfc`, `softplus`, `linv`) never inside each other
  or wrapping `exp` / `ppow`; no `exp` in `exp`; `ppow`, `l1p`, `sqa`, `square` not self-nested. Argument-size limits via
  `constraints`; `complexity_of_operators`: transcendental 2, `erf`/`erfc` 3.
- Templates next to free-form: `tanh(u(z,x) + v(r,x) + w(z,r))` and `a(x)·f(z,r)`; a string → torch translator with a
  round-trip check ≤ 1e-5 (template results have no sympy export).
- Audit before any refit (`sr/audit.py`), over z' −300…2000 mm, r 0…1500 mm and momentum from **0.0005 GeV** to 50 GeV
  (x ≈ −4.1 … +1.3; if a model floors `x`, down to its exact floored `x_min` instead), so that refits without clamps
  cannot blow up on the lowest-momentum events. Two separate checks:
  - *Domain safety (sound):* interval arithmetic on every denominator, `log`/`sqrt` argument and power base — an
    interval containing zero or a forbidden sign rejects the equation. Over-estimation here only costs false rejections,
    never a missed pole.
  - *Magnitude bound `sup|δ| ≤ 1 nat` (tight):* naive interval arithmetic over-estimates because of the dependency
    problem, so the bound uses adaptive subdivision (split only boxes whose interval bound exceeds 1 nat, to a minimum box
    size) and then dense-grid verification with a gradient-based margin on the remaining boxes. An equation fails only
    if an actual evaluation exceeds the bound.
  Raw `/` and `inv` are available only as a labelled extra arm behind the same audit.
- Refit: all constants trainable, likelihood on training hits, no `nan_to_num`, no clamps; ranking by validation NLL.
- Controls: C1 refit only, same schedule; C2 smooth tensor polynomial with the same number of constants (SR is credited
  with Δ(SR) − Δ(C2)); C3 seed-null: the same search on the difference of two ensemble halves.
- Counting: free constants `k` and operator-weighted size `C` reported separately (formula k = 101, MLP k = 3,841,
  `ParamNet` model k ≈ 700; a cubic is k = 4, C ≈ 13).

**Staged budget** (20 Julia threads per run, seeds one after another; ≈ 5 h before any extension)

| Step | What | Size cap | Seeds × time | Gate |
|---|---|---|---|---|
| P | pilot on target (iii), Stage-1 vocabulary | 30 | 1 × 15–20 min | **Gate 2-pilot (Fable)** before further CPU time |
| T | 1-D trajectories | **18** | 3 × 10–15 min | — |
| S1 | 2-D / 3-D density and log-residual regressions | **30** | 3 × 30–45 min | Gate 2a |
| S2 | same target, Stage 1 ∪ {sin, cos} | 30 | 3 × 30–45 min | Gate 2 |

- S1 + S2 stay within ~3–4 h; Pareto front reported at complexity ≤ 12, ≤ 20, ≤ 30.
- One T run = one seed covering all momentum-dependent parameters in a single multi-output search, 10–15 min per seed
  in total (confirmed at approval). Target (ii) takes its time from the S1 allowance.
- **Gate 2-pilot:** target scaling (O(1) inputs, gauge, weights, loss scale); nesting constraints enforced (hall of fame
  scanned); custom operators and the gauge-invariant loss accepted by this PySR version; translator round trip;
  convergence per complexity tier (fixes the per-seed time inside 30–45 min); thread use; kill-and-resume.
  For target (i) the pilot also runs a short multi-output smoke test: `weights` of shape `(N, n_out)` and `guesses` as
  `list[list[str]]` must be accepted by PySR 2.7.3 and give per-output halls of fame.
- **Trajectory convergence check.** After the first T seed, each output's best loss against time is inspected. If any
  of the 3–5 parameters with the highest Fisher information has not converged within the 10–15 min, a brief targeted
  single-output pass (size 18, same vocabulary and seeds) is run for those parameters only and reported as such.
- **Per-seed checkpoints.** PySR writes its resumable state (`checkpoint.pkl`) only at the start and end of `fit`, so
  each seed runs as a loop of ~5-minute chunks with `warm_start=True` in `results/m3_sr/<step>/<target>/seed<k>/`; after
  each chunk the state, a timestamped `hall_of_fame.csv` and `progress.json` are on disk. Resume or extension continues
  the same populations (`PySRRegressor.from_file`); fallback: restart seeded with the saved hall of fame via `guesses`.
- **Overnight extension only without a plateau.** Plateau = over the last three chunks the best loss at each tier
  improved by < 1 % and no new front member improves the refitted validation NLL by > 1 SE. Extend (Stage 1 and Stage 2
  equally) only if at least two of three seeds have not plateaued.

**Stage 2 — gated sin/cos ablation.** Same data, seeds, budget and cap. Trig appears only in a bounded additive
log-density modulation (`tanh(f) + 0.3·tanh(a)·envelope·sin(phase)`), never nested. Pass requires all of: gain over the
Stage-1 point of equal constant count ≤ −3e-4 and ≤ −2 SE; modulation ≤ 0.3 nat and `|δ| ≤ 1 nat` on the extended box;
no sign change of `∂_r ln ρ` for r > 200 mm or of `∂_z' ln ρ` beyond z_90 at p = 0.05, 0.1, 1, 5, 20, 30, 50 GeV;
r50, r90, z50, z90 monotone from 20 → 30 → 50 GeV; ≤ 4 sign changes of the modulation inside the data range, none
outside; constants stable across seeds; fitted periods not matching the MLP's feature periods; smaller loss reduction on
the seed-null target. Otherwise a negative result with its SE.

**Gate 2 (Fable)** — domain and tail audit, extrapolation, constant-count fairness against the formula and the MLP,
controls C1–C3, plateau verdict.

## 8. Milestone 4 — locked test evaluation

Order: θ-blind finalists frozen → M2C (θ track) → `results/final/LOCK.json` written (checkpoint hashes, validation
numbers, selection trail) → **Gate 2c** → `final_test.py` → Gate 3. Finalists: at most three θ-blind (best network or
ensemble, best closed form, simplest within 1 SE) and one θ-conditioned in its own table.

**Gate 2c (Fable, fresh context) — must pass before `final_test.py` may execute.**
- Θ1–Θ3: the per-event front-face acceptance math re-derived and checked numerically against a brute-force azimuthal
  integration; bulk-guard compliance (headline on z' < 1150 mm without the last ring, hits and normalisation alike);
  mass beyond `z_b` per arm, with Θ3 confirmed as diagnostic only; θ-blind harness untouched by this track.
- `LOCK.json`: hashes match the files on disk; the θ-blind finalists are exactly those frozen before M2C; one θ finalist;
  every selection step traceable to validation arrays.
- Zero test leakage: access log of the test cache shows only the Gate 0 parity run; no test number in any results file,
  registry entry, plot or script default.

`scripts/final_test.py` then runs once and refuses a second run: official `common.test_nll` (K=64, seed 12345) for the
headline, plus the pre-registered guard columns (K=1024, bulk) in the same run.

Reported: test NLL, paired Δ ± SE against `mlp_std.pt` (18.1208) and the component formula (18.1235), constants and
complexity, validation-to-test shrinkage, Holm correction across the θ-blind finalists. A win is claimed only if the test
Δ is negative by ≥ 3 SE and consistent with validation within 2 SE; differences below 5e-4 are stated to lie within the
metric's own systematics (K bias, border and back-face conventions). **Gate 3 (Fable)** reviews the lock file, the single
run and every sentence of the conclusions.

## 9. Subagent execution graph

Main session = orchestrator: sequencing, GPU queue, gate decisions, commits. Workers via the Agent tool with
`model: "opus"` or `model: "fable"`.

| Step | Agent | Task | Writes only to | GPU |
|---|---|---|---|---|
| M0-a | Opus | branch, checksums, `git mv`, `summary.md` paths, layout, `paths.py`, `data.py`, cache, notebook cell, parity script | scaffolding, `cache/`, `results/m0_parity/` | yes |
| M0-b | Fable | `stats.py`, `metric.py` + equality test | `src/sfm/{stats,metric}.py`, `tests/test_metric.py` | no |
| **Gate 0** | Fable (fresh context) | parity, leakage, units | `results/gates/gate0.md` | yes |
| M1-a | Opus | audit D1–D9, plots | `scripts/audit_*`, `results/m1_audit/` | yes |
| M1-b | Fable | `families/` and `cellint.py` with tests (parallel) | `src/sfm/families/`, `tests/test_families.py` | no |
| M1-c | Opus | `train.py`, `models/`, `registry.py`, configs (parallel) | those files, `configs/` | no |
| M1-d | Fable | interpret the audit, confirm or prune the ladders | `results/m1_audit/AUDIT.md` | no |
| M2A | Opus | KL screen, ladder A0–A11 | `scripts/{screen,fit}_dist*`, `results/m2a_dist/` | queue |
| M2B | Opus | B0–B7 | `scripts/fit_mlp*`, `results/m2b_mlp/` | queue, after M2A |
| M3-prep | Fable + Opus | `sr/{operators,audit,translate}.py` (Fable); Julia bootstrap (Opus) — during M2 | disjoint files in `src/sfm/sr/` | no |
| **Gate 1** | Fable | math, stability, paired SE, selection, guards | `results/gates/gate1.md` | no |
| M3-pilot | Opus | 1-seed pilot, kill-and-resume | `results/m3_sr/pilot/` | no |
| **Gate 2-pilot** | Fable | scaling, nesting, convergence | `results/gates/gate2_pilot.md` | no |
| M3 | Opus | trajectories and Stage 1, 3 seeds; refits | `results/m3_sr/{traj,stage1}/` | refits |
| **Gate 2a** | Fable | Stage-1 audit, plateau verdict | `results/gates/gate2a.md` | no |
| M3-s2 | Opus | sin/cos ablation; extension only without plateau | `results/m3_sr/stage2/` | refits |
| **Gate 2** | Fable | equations, tails, fairness | `results/gates/gate2.md` | no |
| M2C | Opus | θ track Θ1–Θ3, after the θ-blind finalists are frozen | `scripts/fit_theta*`, `results/m2c_theta/` | queue |
| M4-lock | Opus | write `LOCK.json` (hashes, finalists, selection trail); no test access | `results/final/LOCK.json` | no |
| **Gate 2c** | Fable (fresh context) | audit Θ1–Θ3 (acceptance math, bulk guard, mass beyond `z_b`); verify `LOCK.json`; zero test leakage | `results/gates/gate2c.md` | no |
| M4-test | Opus | single run of `final_test.py`, tables, `REPORT.md` — only after Gate 2c passes | `results/final/`, `REPORT.md` | yes |
| **Gate 3** | Fable | final review | `results/gates/gate3.md` | no |

No two concurrent agents share a writable file or output directory. Reviewers get code and result files, not conclusions.
A failed gate returns the work to its owner; the next milestone does not start. At every reached milestone (passed gate)
the orchestrator commits and pushes branch `shape-form-martin` to `origin` (user instruction of 24 Nov, replacing "no push").

## 10. Guardrails (every agent)

- Data only from `shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet`; the CSV is never opened.
- `shape-form-proton/` and the original notebooks are read-only (imports with `-B`; `setup_particle` never called).
- Outside `shape-form-martin/` only: the two moved files' old paths, the path references in `summary.md`,
  `pyproject.toml` / `uv.lock` for `pytest` and `nbclient`, and the one-time Julia download.
- Git: never `git add -A` or `git add .`; stage explicit paths only; raw data and `shape-form-proton/` are never staged.
- Test split: its cache file is opened only by the parity script and by `final_test.py`. No test number informs a choice.
- Report-metric, slab-metric, bulk-restricted and continuous numbers never share a column or a delta.
- `torch.logsumexp` / `F.log_softmax`; guarded logs and divisions; explicit `train()` / `eval()` and
  `torch.inference_mode()`; seeds for `random`, `numpy`, `torch`; one GPU job at a time; lean column reads.
- Selection on validation at K=64 with paired SE and the 1-SE rule; last-iterate weights for new fits; at most six
  pre-registered primary comparisons per stage; second-seed replication before any promotion.
- Every long job checkpoints and resumes. Python through `uv run`. Failed approaches are reported as measured.

## 11. Decisions taken (24 Nov 2026)

| Topic | Decision |
|---|---|
| θ-conditioned models | Separate track after the θ-blind lock; own table. |
| Evaluation | Unmodified grid metric for every headline; continuous / refined evaluation only if D6(c) ≥ 3e-4 nat, separately labelled and also projected to the grid. |
| PySR | Staged: 15–20 min 1-seed pilot verified by Fable; 1-D trajectories at maxsize 18 (~10–15 min per run); 2-D/3-D at maxsize 30, ~30–45 min per seed, ~3–4 h across Stage 1 and the sin/cos ablation; 3 seeds on 20 threads; Pareto at ≤ 12, ≤ 20, ≤ 30; per-seed checkpoints; overnight extension only without a plateau. |
| Slab pion MLP | Left untouched; one paragraph in the report. |
| File move | `git mv` if tracked, plain `mv` if not; fix the path references in `summary.md`. |
| Outside changes | Julia bootstrap; `uv add --dev pytest nbclient`. |
| Git | Branch `shape-form-martin`; commit **and push** at every reached milestone (passed gate); explicit-path staging only. |

**Confirmed at approval:** (1) pole-free Stage 1, raw `/` and `inv` isolated as a labelled, audited arm; (2) G-bulk is the
headline of the θ track, with the report metric and mass beyond `z_b` reported beside it; (3) 10–15 min per seed covers
all trajectories in one multi-output search, subject to the convergence check in §7.

**Modifications added at approval:** Gate 2c between M2C and the test run (§8, §9); safe move and explicit staging (§3,
§10); low-momentum boundary checks in Gate 0, D1 and the PySR audit range down to 0.0005 GeV (§3, §4, §7); audit split
into sound interval-arithmetic domain safety plus adaptive-subdivision / dense-grid magnitude bound, and multi-output
verification with a targeted single-output fallback (§7).

## 12. Verification

- `uv run python shape-form-martin/scripts/parity.py` reproduces 18.120842 / 18.118084 / 18.123459 / 18.120407.
- `uv run pytest shape-form-martin/tests`: metric equality, family CDFs vs `scipy.stats`, mass normalisation, the audit
  rejecting `c/(r − 0.017)`, translator round trip, PySR kill-and-resume, `final_test.py` refusing a second run.
- Each gate file lists its criteria with measured values and PASS/FAIL.
- `REPORT.md` tables regenerate from `results/registry.jsonl` and the stored per-event arrays.

## 13. Amendments after Gate 0 (24 Nov 2026)

Gate 0 verdict: **PASS WITH AMENDMENTS** (`results/gates/gate0.md`). The parity criteria of §3 failed as written; the
cause was identified and the stronger property shown exactly on validation.

**Why the stored numbers did not reproduce.** `common.test_nll` draws its fold points with a seeded CUDA generator
whose stream for large tensors depends on the GPU's number of streaming multiprocessors. A CPU emulation of that kernel
(`results/gates/gate0_checks/{philox_emul.c,emul.py}`) reproduces this GPU bit for bit, and with the launch grid of
the reference machine (1056 blocks) it reproduces all 13 stored validation means within 1.9e-6 and all 10 stored paired
differences within 7.2e-8.

| ID | Replaces | Amended criterion |
|---|---|---|
| A1 | means within 5e-5, per-event test arrays within 1e-4 (§3) | (a) validation means within 4e-6 and paired differences within 1e-6 on the emulated reference stream — met; (b) stream-independent identities exact (`Ā`, `lnp0/lnp_s`, `v0/vs`, stored formula values) — met; (c) test event sequence identical — met; (d) test under the local stream statistically consistent with a change of fold points — met; (e) per-event test arrays on the reference stream within 1e-4 — **open (C1)** |
| A2 | "`sfm.metric` equals `common.test_nll` exactly" (§3, §12) | bit-identical inside `sfm.metric.deterministic()`; otherwise ≤ 1e-4 per event and ≤ 1e-6 on the mean (`common.test_nll` differs from itself by up to 2.9e-5 per event between runs) |
| A3 | fallback correlation > 0.9999 (§3) | r ≥ 0.9995 and rms of the per-event difference within ±15 % of the validation seed-to-seed analogue |
| A4 | G-bulk on z′ < 1150 mm for all events (§6) | headline on the stratum `zb ≥ 1221 mm` with `Bulk(short="raise")`; events with `zb < 1221 mm` (about 1 %, late shower starts) are reported as their own row and never removed from the report metric |
| A5 | D1 "sub-100 MeV events inspected one by one" (§4) | validation has no event below 0.1747 GeV: inspect the 84 validation events below 0.3 GeV and the two training events below 0.1 GeV; every test delta is also reported without the single test event below the training momentum range (0.0734 GeV), pre-registered |
| A6 | reference values 18.120842 / 18.118084 / 18.123459 / 18.120407 (§3, §5 A0, §12) | **numbers of record on this machine:** MLP val 18.117857, test 18.120806; formula val 18.120061, test 18.123333; formula − MLP +0.002204 ± 0.000293 (val), +0.002527 ± 0.000304 (test). Stored values are quoted as "reference machine, other fold points". All campaign deltas are paired against locally recomputed arrays (`results/m0_parity/*_nll.npy`). |

**A6 revised (25 Nov 2026 — change of machine; user decision).** The H200 becomes the machine of record. Points 1–5 and
7–10 follow from the user's decisions of 25 Nov 2026 (machine of record; no test access on the H200 before the final
run; all three controls trained on the H200; PySR refits and scores on the H200). Point 6 and the per-event clause of point 8 were
proposals of the review; the user decided them on 25 Nov 2026 (P2 and P3 of §16.0) and they are recorded here as
decided, with the user's guardrails. The gate additions of point 10 are the reviewer's consequences of the move.
Section 16 holds the transfer package and the run-book.

> **A6 (revised 25 Nov 2026 — change of machine).** Replaces row A6 for everything computed from this amendment on.
>
> **1. Machine of record.** The H200 produces every number that enters a table, a delta, a pre-registered rule or
> `LOCK.json`: the per-event arrays of the two benchmarks, all Track A fits, all Track B training still to run, the
> re-scoring of the eight networks trained on the laptop, the ensembles, the θ arms, every likelihood refit and
> every registered score of a PySR equation (controls C1 and C2 included), the zero-training scores of the geometry
> arm, all guards, and the single final test evaluation.
>
> **2. What stays on the laptop** and produces no number of record: the PySR searches (CPU; they see only exported
> targets and their own loss, and hand over equations as text), the equation audits (exact mathematics, no data),
> the geometry audit (training and validation data only; its gains are audit numbers on the laptop's sample points
> and decide only whether and in which form the arm is run), and the gates. A gate may recompute an H200 array on
> the laptop with the emulated H200 stream (`sfm.refstream` at the H200's launch grid) as a check; such an array is
> never a pairing partner.
>
> **3. Numbers of record** are the benchmark arrays computed on the H200 with the unmodified `common.test_nll`
> inside `sfm.metric.deterministic()`. The H200's launch grid is read with `sfm.refstream.local_grid()` before
> anything else is computed. [If it is 1056, the H200 draws the earlier study's fold points: validation 18.118083
> (MLP) and 18.120407 (formula), formula − MLP = +0.002324 ± 0.000294, to be met within 4e-6 on the float32 means
> and within 1e-4 per event against the laptop's emulated arrays. The stored test values 18.120842 / 18.123459 and
> +0.002616 ± 0.000292 are then the expected test numbers of record; they are confirmed by the first step of the
> final run and not assumed before it. If it is not 1056, the H200's own values are the numbers of record and both
> the stored and the laptop's values are other sample points.] The laptop's numbers — validation 18.117857 /
> 18.120061, test 18.120806 / 18.123333, formula − MLP +0.002204 ± 0.000293 (validation) and +0.002527 ± 0.000304
> (test), the Milestone 1 audit, the eight Track B rows of 24 Nov — are quoted as "laptop (RTX 4060, launch grid
> 144), other sample points". They are records: kept, never overwritten, never part of a delta with an H200 array.
>
> **4. Pairing rule.** A delta of record is formed only between two per-event arrays computed on the H200 on
> identical fold points (launch grid, seed, K, batch size, fold scheme and split all equal). Each pre-registered
> rule (keep, promotion, replication, equal NLL, 1-SE, convergence, guards, plateau) is evaluated once, on such
> arrays, and never again with another machine's numbers. Criteria written as laptop values take their paired
> form: "A0 at step 0 = 18.120061" becomes "A0 at step 0 equals the H200 formula array, within 1e-4 per event and
> 1e-6 on the mean"; "at least 3e-4 better than 18.12041, or than the A0 value" is a paired delta against the
> formula array of the same stream.
>
> **5. Provenance.** Weights are files identified by SHA-256 and labelled with the machine that trained them. The
> five-member ensemble consists of seeds 11 and 12 (fine-tuned on the laptop) and seeds 13–15 (fine-tuned on the
> H200 from the laptop's stage-1 networks); the two existing fine-tunes are not repeated. The control B0 is trained
> on the H200 for all three seeds, under the run names `b0h_seed0`, `b0h_seed1`, `b0h_seed2` (user decision P5); its
> seed scatter and the B0 mean use these three runs only.² The laptop's `b0_seed0` stays untouched as the archived
> laptop record; it is re-scored on the H200 and shown under that label, and it enters no scatter, mean or rule.
> This is fixed before any control has been scored on the H200, and no run is repeated in order to choose between
> outcomes. A comparator is trained on the machine of the variants paired with it. No interrupted fit is continued
> on another machine.
>
> **6. Budgets (approved by the user, 25 Nov 2026, P2).** The run lists of `AUDIT.md` §4.1 and §4.2 are run in full on the H200. The wall-clock rule
> (4 GPU-hours, drop orders) was written for the laptop and is not applied: which runs exist depends only on the
> lists and on validation outcomes, not on machine speed or on how many jobs run side by side.⁴ The pre-registered
> sets of primary comparisons (six per track) are unchanged and remain the basis of the multiple-testing accounting;
> every other run of the lists stays exploratory.
>
> **7. PySR.** For each search the complete audited front (every seed, every pre-registered complexity tier) goes to
> the H200. No likelihood computed on the laptop selects, ranks or screens an equation.³ Refits, controls and
> scores are made on the H200; the audit is repeated on the refitted constants; the "refitted validation NLL" of
> the plateau rule is the H200's. The number of candidates entering the winner's-curse correction is the number of
> audited front members sent to the H200.
>
> **8. Test split.** The test cache is opened only on the H200 and only by `final_test.py`, after Gate 2c. There is
> no test access of any kind before that run; benchmark parity on the H200 is shown on validation only; the laptop
> does not open the test cache again.¹ `final_test.py` first recomputes the per-event test arrays of the two
> benchmarks and stops before any finalist is loaded unless (a) the SHA-256 of the test cache equals the value
> recorded by the Gate 0 run in `results/m0_parity/parity.json`, (b) the test event numbers equal
> `results/m0_parity/test_event_number.npy` in order, and (c) for both benchmarks the arrays agree with the earlier
> study's stored arrays (`test_v3_nll.npz`, keys `std:mlp_ft` and `std:new`) within 1e-4 for every one of the
> 14,956 events and within 4e-6 on the mean — the criteria of C1. **Per-event clause (user decision P3).** TF32 is disabled explicitly on the
> H200 (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`) and both flags
> are part of the machine tag. The mean-parity check stays active in every case (the user's bound is ≤ 5e-5; the
> clause in force is the tighter C1 criterion of 4e-6, which implies it). If the validation parity on the H200 shows
> any event above 5e-5, the relaxed per-event clause — at most three events above 1e-4, none above 1e-3, rms at most
> 1e-5 — is activated only after Fable has confirmed, on validation, that the outliers are 1-ULP bin-edge flips (a
> fold point changing its grid cell through a last-bit difference in its coordinates). Without that confirmation the
> strict clause, every event within 1e-4, applies. Gate 2c records which clause is in force before the run.
> **Test-derived files (user decision P4).** The seven test-derived files — the test cache, `test_v3_nll.npz`,
> `test_event_number.npy` and the four laptop test arrays — are NOT on the H200 before Gate 2c; their physical
> absence, checked by `make_transfer_package.py verify`, proves that no test access was possible there. They are
> copied only after Gate 2c has passed and then checked with `verify --after-gate2c`. [If the launch grid is not 1056, (c) is evaluated on the emulated reference stream,
> and the native arrays must in addition satisfy A3 against the emulated ones.] A stop is not the final run: only
> the two benchmarks have been evaluated, the access stays in the log, and a new start needs the cause, the user's
> go-ahead and an unchanged `LOCK.json`. Finalists are paired against the benchmark test arrays of the same run.
>
> **9. Test-access log.** `results/test_access.log` is one tracked file. Its two lines of 24 Nov 2026 (Gate 0
> parity, C1) were written on the laptop; the file with exactly these two lines has the SHA-256
> `effa9734743f51959806d7d8b87e2d262c10773a9be25f2104ceeca9557f0c60`. From this amendment on only the H200 adds
> lines, each with host, GPU and launch grid, and the laptop refuses `allow_test`. Until the final run both working
> copies hold the file unchanged: two laptop lines, no H200 line. After the final run the H200's file is committed
> and pulled, so there is one log again: the two laptop lines and one H200 line per start of `final_test.py`. Any
> other line, on either machine, is reported to the user before anything else is done.
>
> **10. Gates.**
> *Gate 1* also checks: the identity record of the H200 (GPU, launch grid, torch build, generator test) and the
> validation parity of the two benchmarks against their targets; that the laptop-trained checkpoints have the
> SHA-256 of their laptop registry rows and that their re-scored values meet the predicted table [grid 1056]; that
> every array behind a delta, a rule or a table was computed on the H200 against the reference arrays created
> there, and that no table reads a laptop array or a laptop row; that the control consists of three H200-trained
> seeds and the H200's `b0_seed0` is not the laptop file; that no fit was continued across machines and every
> comparator was trained on the machine of its variants; by recomputation on the laptop with the emulated H200
> stream, that both benchmarks and every kept or promoted candidate agree with the H200 arrays within 1e-4 per
> event and 1e-6 on the mean; and, if fits ran side by side, the equivalence checks and that the set of runs
> equals the pre-registered lists.
> *Gate 2* also checks that every refit and score of an equation is an H200 registry row and that the search
> record contains no likelihood.
> *Gate 2c* also checks: `LOCK.json` names the machine of record and holds, for every finalist, the hash of the
> checkpoint or equation, the training machine, the H200 validation numbers and the hashes of the benchmark arrays
> they were paired with; every selection step is traceable to H200 arrays; the scores of the geometry arm and of
> the PySR finalists are H200 rows; the test-access log is as in point 9; the test cache on the H200 has the
> seven test-derived files are absent from the H200 when the gate is taken (`make_transfer_package.py verify`
> reports "test-derived files absent: yes") and are copied only after it has passed, the test cache then having the
> recorded SHA-256 (`verify --after-gate2c`);
> `final_test.py` refuses any machine other than the one in `LOCK.json`, refuses path overrides, and implements
> the stop rule of point 8 with the tolerances fixed by this gate.
> *Gate 3* also checks: the final run took place once, on the machine of record; its first step passed, with the
> measured differences on file; every test delta is paired against benchmark arrays of the same run; the log holds
> the two laptop lines and exactly one H200 line (or stop lines, each explained, with `LOCK.json` unchanged in
> between); laptop numbers appear in the report only under their label; the validation-to-test shrinkage uses H200
> validation numbers.
>
> ¹ Not chosen: a third Gate-0-type parity access on the H200 before the final run.
> ² Not chosen: a scatter from the laptop's `b0_seed0` and two H200 runs.
> ³ Not chosen: refitting or ranking candidates by likelihood on the laptop and re-scoring only the selected ones
> on the H200.
> ⁴ Not chosen: a budget in nominal units (the cost column of `AUDIT.md` §4.1, cap 240 nominal minutes).

**Item C1 — closed, PASS (24 Nov 2026, user decision: run now).** `results/m0_parity/c1_reference_stream.json`: on the
emulated reference stream the test arrays of both benchmarks agree with the stored ones for all 14,956 events (max |Δ|
2.9e-5 and 3.1e-5; means 18.120842 and 18.123458; formula − MLP 0.0026163 ± 0.000292 against the stored 0.0026164).
The test-access log now holds two Gate 0 parity lines; Gate 2c expects exactly these two before the final run.
*Original text of the item:* **Open item C1.** Exact per-event parity of the two existing benchmarks on *test* on the reference stream (pass: max
|Δ| ≤ 1e-4 per event, |Δ mean| ≤ 4e-6). It is the only direct check of the tensors in the test cache. Not blocking
Milestone 1; must be closed before any finalist touches test. How it is closed (now, as a second logged Gate 0 parity
access, or as an abort-on-fail first step of `final_test.py`) is the user's decision.

**Guard definitions refined** (from the metric author's review):
- "≥ 50 % of the gain persists" (G-K, G-bulk) is evaluated as the paired margin `guard − 0.5·main` with its SE, and is
  called resolved only at 2 SE (`sfm.stats.guard_persistence`).
- G-K measures a difference of Jensen biases; it is read as such (smoothing gains shrink at high K, sharpening gains grow).
- G-region localises the hit term only; the normalisation term is reported as its own row.
- Final claim: "consistent with validation within 2 SE" uses the validation gain corrected for selection
  (`sfm.stats.winners_curse_correction`), since an uncorrected validation gain is tilted against true winners.
- Short events (`zb < 1221 mm`) are a stratum in D1 and D5 with leave-out deltas: 0.6 % of the validation events raise
  the mean NLL by 0.0107.

**Other changes**
- Registry evaluations call the unmodified `common.test_nll` inside `sfm.metric.deterministic()` so stored arrays are
  bit-reproducible on this machine.
- The benchmark maps are not used as teachers or KL targets below the training minimum 0.0895 GeV.
- The reference-stream emulator moves to `src/sfm/refstream.py` with a local-GPU test; evaluating finalists on the
  reference stream is an optional, pre-registered secondary column (it allows paired deltas against the stored report
  MLP and baseline arrays), never the headline.
- The training-based parts of D7 (residual table, learning curve) run in M2B with the training harness; M1 keeps the
  score test. The math-sensitive diagnostics D6(b), D6(c) and D7 are computed by the Fable step M1-d.
- Git: commit and push at every reached milestone.

**Correction (M1 audit).** The periods of the MLP's sin/cos depth features are 1650/k mm = 1650, 825, 550, 412.5 mm, not
3300/1650/1100/825 mm as written in §4 (D9) and §7 (Stage 2 checks); the corrected values apply.

## 14. Milestone 1 outcome and the pre-registered run lists (24 Nov 2026)

Milestone 1 is closed (`results/m1_audit/SUMMARY.md`, `results/m1_audit/AUDIT.md`). All numbers are validation, in
10⁻⁴ nat per event, paired.

- **Where the gap is.** Formula − MLP = +22.0 ± 2.9. The formula's radial conditionals cost +18.0 ± 2.5, its depth
  marginal +4.6 ± 1.5, so no depth-profile family can gain more than about 5. By hit position the MLP wins at
  30 ≤ r < 663 mm (+50.3) and the formula wins in the core (−20.1) and at the entrance (−18.4). Cells where the metric
  is only approximately a proper score carry +10.2 (46 %); the last ring alone +7.3. Bulk gap: +18.6 ± 2.7.
- **K is not a lever.** Both models have the same Jensen coefficient (0.160/K); the gap is the same at K = 16 and
  K = 1024. B2 is kept as pre-registered but is an expected null.
- **Ensembles.** Three equal-quality networks beat their best member by −8.6 ± 2.5; adding weaker members to `mlp_std`
  hurts. Predicted gain of five `mlp_std`-quality networks over their mean member: −5.9 (−3.5 … −8.5).
- **Headroom for any 128×32 map** (cross-validated score test): at least 7.2 ± 3.1 in the population limit, mostly
  below 4 GeV; a free table fitted on the available 70k events would realise about 1.7.
- **Binning.** Exact cell integration at the legacy coefficients is +4.07 ± 0.56 worse (the coefficients were fitted
  through the centre rule); rung A2 therefore means "refit in exact mode".
- **Refined lookup (D6c): −0.5 ± 0.8 at the K of record, below the 3e-4 rule → sub-track B7 is closed.** One labelled
  diagnostic column (4× refined lookup at K = 1024) for the closed-form finalist only.
- **New: lateral detector faces.** In 25.8 % of validation events the shower axis leaves through a side face
  (|X|, |Y| ≈ 2.62 m) before z′ = 1450 mm; the MLP puts 4.9 % of its mass outside the side faces against 1.5 % in
  front of the front face. First-order acceptance gain of a geometry-aware model: front face 155, all faces 933.
  This needs φ and the entry position as inputs and is outside the approved θ-only scope (decision below).
- **Ringing.** The MLP map carries harmonics of its depth features (1650/k mm) at the level of the seed-to-seed
  differences; a closed form should not copy them. PySR period check: 1650/k for k = 1…8, periods above ≈ 570 mm
  count as trend.

**The run lists of `results/m1_audit/AUDIT.md` §4 are the pre-registered lists for Milestone 2** (rules §4.0, Track A
§4.1, Track B §4.2, θ arms §4.3) and replace the ladders of §5 and §6 where they differ. Primaries — Track A: A4 (EMG),
A5 (generalised gamma, free soft start), A7 (Burr radial body), A8 (Weibull halo), A9 (detector-frame halo), A10 (pure
mixture, K = 3); Track B: B1 (five-seed ensemble), B1d (its distillation into one network), B2b (K = 64 training),
B2c (stratified fold points), B4f (front acceptance on the logits), B6c (width). Every other family the user named
(inverse Gaussian, log-normal, Burr XII in depth, Student-t, Lomax, generalised-Gaussian core) gets one exploratory
likelihood fit. Not run: free row factor / free lateral conditional (answered by the hybrid test without training).

**User decision on the geometry-conditioned arm (25 Nov 2026).**
- The trained conditioned track stays θ-only (Θ1–Θ3) as approved. **No new network with inputs (p, θ, φ, x0, y0) is trained.**
- Added: a **zero-training 3-D geometry acceptance arm**, run after the θ-blind finalists are frozen and applied to
  both the frozen θ-blind MLP and the best closed-form formula. Own table, labelled with its inputs
  (p, θ, φ, entry position); never mixed with θ-blind deltas.
- **Before it runs, Fable must** (a) find out, from training-split hits only, why 1.35 % of events do not fit a simple
  box (stepped tower/module outline, beam-pipe cut-out, coordinate edge, …), and (b) verify on validation that the gain
  persists under G-bulk and with a front + side-only acceptance, so that it is not inflated by the flat `z_b`
  back-face cut. Report: `results/m2c_theta/geometry/GEOMETRY.md`.

## 15. Checkpoint — 25 Nov 2026, 00:20 CET (24 Nov 23:20 UTC; campaign paused at the user's request)

The running batch needed about five more hours, so every worker was stopped at a clean point. Nothing is running; all
GPU jobs are resumable from their checkpoints. **To resume:** read §13, §14 and this section, then the STATUS files
below; check `git status`, `uv run pytest shape-form-martin/tests -q`, and that `results/test_access.log` still has
exactly two lines; continue with the first unfinished item of "Next actions".

**Finished and pushed**

| Item | Commit | Evidence |
|---|---|---|
| Milestone 0: workspace, lean cache, parity; Gate 0 PASS WITH AMENDMENTS | `24305cb` | `results/gates/gate0.md` |
| C1: exact per-event test parity on the reference stream, PASS | in `bcb76fa` | `results/m0_parity/c1_reference_stream.json` |
| Milestone 1: audit D1–D9, interpretation, run lists; families, harness, registry | `bcb76fa` | `results/m1_audit/{SUMMARY,AUDIT}.md` |

**In progress when paused** (each STATUS file lists what is done, what is not, and the resume command)

| Work item | Agent type | State | STATUS file |
|---|---|---|---|
| Track B seeds: five 4×32 networks (report recipe + fine-tune, seeds 11–15), control B0 (seeds 0–2), ensembles | Opus | stage 1 done and registered for all five seeds; fine-tunes done for seeds 11 and 12 (seed 13 restarts from step 0; 14, 15 not started); B0 seed 0 done, seeds 1–2 not started; ensembles, size curve and report not done. About 23 min of fits plus a few minutes remain; the three resume commands are in the STATUS file | `results/m2b_mlp/STATUS.md` |
| Track A: pre-registered fits of `results/m1_audit/AUDIT.md` §4.1 | Opus | factory, tests (40 pass) and driver for phases templates/parity/plr/stage0/stage1p/stage1x written; **no fit run yet**; report/guards/Stage 2 script not written. Known bug to work around or fix: `ParamNet.fit_to` over-weights constant columns (variance ≈ 1e-15) | `results/m2a_dist/STATUS.md` |
| PySR math modules (`src/sfm/sr/`) | Fable | `operators.py` done (47 tests); `translate.py`, `audit.py`, `checks.py`, `targets.py`, `refit.py` **not started**, designs and derivations written down; list of points to confirm with Julia in `operators.UNCONFIRMED` | `src/sfm/sr/STATUS.md` |
| Julia + search runner | Opus | Julia 1.13.4 installed (`.venv/julia_env`, `~/.julia`), trivial fit works; `sr/search.py`, `scripts/sr_run.py` written, 15 Julia-free tests pass; **runner never run against Julia; smoke tests a–f (incl. kill-and-resume) not started**; interface between `operators.search_config` and `search.py` unchecked | `results/m3_sr/smoke/STATUS.md` |
| Detector-geometry audit (required by the user before the zero-training geometry arm) | Fable | origin identity verified on training events; the instrumented volume is a **stepped octagon with an off-centre beam hole**, not a square box; the 1.35 % misfit comes from the straight-line origin (by exclusion, not yet reproduced). Acceptance, gains and guards **not computed: both of the user's questions are open**; the earlier estimates (155 front / 933 all faces) must not be quoted | `results/m2c_theta/geometry/STATUS.md` |

**Next actions, in order** (routing: Fable for mathematics, physics, statistics and every gate; Opus for engineering
and running jobs; never two agents with overlapping write access; one GPU job at a time via `sfm.gpulock`)

1. Opus: finish the Track B seeds (remaining fine-tunes, B0 seeds 1–2, guards, ensembles E5 / E6 / size curve).
2. Opus: run Track A from `run_m2a.py templates parity plr stage0 stage1p stage1x`, write the report script, apply the
   pre-registered rules, Stage 2 (A11, A11k, A11n). About 4 GPU-hours.
3. Opus: the remaining Track B list of `AUDIT.md` §4.2 (B1d, B1f, B2a/b/c, B3s, B4f/s/x, B5f, B6a/b/c, B-ceil; needs a
   `train_fraction` option). About 2 GPU-hours. Fix the `ParamNet.fit_to` weighting and the `sfm.train` weight-decay
   docstring with tests.
4. Fable: A11p (Fisher pruning) if a Track A finalist exists; then **Gate 1** (fresh context). On a pass: commit and push Milestone 2.
5. Fable: finish `src/sfm/sr/` (translate, audit, checks, targets, refit). Opus: live runner test and smoke tests a–f.
   Then the PySR pilot (one seed, 15–20 min) → **Gate 2-pilot** → trajectories (size 18) and Stage 1 (size 30), three
   seeds on 20 threads → **Gate 2a** → sin/cos ablation → **Gate 2**. About 5 h of CPU before any extension.
6. Fable: finish the geometry audit → `results/m2c_theta/geometry/GEOMETRY.md` with a recommendation.
7. After the θ-blind finalists are frozen: Opus runs the θ track Θ1–Θ3 and, if the geometry audit recommends it, the
   zero-training geometry arm on the frozen MLP and the best closed form.
8. Opus writes `results/final/LOCK.json` → **Gate 2c** (Fable, fresh context) → single run of `final_test.py` →
   **Gate 3** → `REPORT.md`; commit and push.

**First Track B numbers (validation, 10⁻⁴ nat against `mlp_std`, paired; not yet a result — the ensembles are missing)**
- Stage-1 networks alone: +12.6 … +32.9 (seed-to-seed sd of the mean 7.4). After the fine-tune: seed 11 +1.8 ± 1.7,
  seed 12 +3.3 ± 1.7. The fine-tune is worth −16 … −20 per network.
- Control B0 (seed 0), `mlp_std` continued with the identical schedule and best-validation selection at K = 16:
  +1.19 ± 0.42, i.e. slightly worse than `mlp_std` at K = 64.

**Operational note.** Twice a Track B job that held the GPU lock ran out of memory because another process held
about 7 GB without holding the lock. Every GPU-using script must keep its tensors off the GPU until it holds
`gpu_lock` (the Track B scripts now check free memory after taking the lock and back off).

**Open points raised by workers, to settle when resuming**
- PySR magnitude bound `sup|δ| ≤ 1 nat` depends on the additive gauge; proposal: bound `f − f(reference point, x)`.
- The literal Stage-2 template is bounded by 1.3, not 1; a `0.7·tanh(f)` variant exists.
- The audit must be repeated on the refitted equation (constants move in the refit), not only before the refit.
- "Exponent a constant" for `ppow` is enforced by the hall-of-fame scan, not by the search itself.
- Fisher weights for trajectory targets must use the expected Hessian of the energy-weighted pseudo-likelihood.
- The moved notebook writes to `shape-form-martin/results`; a full run would retrain rather than reuse old checkpoints.

### 15.1 Second checkpoint — 25 Nov 2026, 01:00 CET: the H200 transfer task was started and paused

**Resume order set by the user:** first finish the H200 transfer task below (its two workers), write section 16,
commit and push it as a plan amendment, give the manifest summary, and then **stop and wait for the user's go-ahead**.
Only after that go-ahead continue with the five work items of the table above.

**Decisions taken by the user (25 Nov 2026)**
- An **H200 machine becomes the machine of record** for everything moved there and for the final test evaluation.
  This laptop's numbers stay in the plan as "laptop, other sample points". To be recorded as an amendment to §13 (A6)
  together with section 16.
- **Moved to the H200:** the Track A distribution fits (`AUDIT.md` §4.1); the remaining Track B seeds (three
  fine-tunes, two controls) and the ensembles; the remaining Track B runs (`AUDIT.md` §4.2); the θ track Θ1–Θ3.
- **Not moved:** the PySR searches, the geometry audit, the Fable gates.

**The task: new section "16. H200 transfer package" plus the file set needed to run the moved steps there.** It is a
planning and packaging task: no training, no campaign batch. Mathematical or statistical judgement goes to Fable,
packaging work to Opus. Section 16 must contain:
1. A manifest of every file the moved steps read, in three groups: (a) already in the pushed branch; (b) git-ignored
   and to be copied by hand, each with size and SHA-256; (c) regenerated on the H200, with command and expected time.
   Derived from the code (what `sfm.paths`, `sfm.data`, `sfm.legacy`, `sfm.registry`, `sfm.models`, the `run_m2a` /
   `run_m2b` scripts and the configs actually open), each file's tracked/ignored status confirmed with git. Group (b)
   is expected to include at least the Parquet data file, the `shape-form-proton` pion result files the legacy adapter
   loads, the checkpoints of the 8 registered Track B networks and the per-event pairing arrays — to be verified, and
   anything missed listed.
2. `shape-form-martin/scripts/make_transfer_package.py`: writes the group (b) list as an rsync/tar file list plus a
   checksum file, with a matching verify mode for the H200. It must not read the 15 GB CSV and must not open the test
   cache beyond a plain file hash. State whether the test cache is copied or rebuilt, and that it stays unopened until
   the final evaluation after Gate 2c.
3. The ordered run-book for the H200: environment setup with uv and how to confirm the CUDA build of torch; path
   settings (`SFM_DATA_DIR` etc.); cache build or verification; the parity check of the two benchmarks there, including
   whether the H200 reproduces the stored 18.1208 / 18.1235 directly; recomputing the validation pairing arrays;
   re-scoring the 8 registered networks; then each moved step with its exact resume command and expected duration.
4. What must be recomputed there versus what carries over unchanged (Gate 0 conclusions, Milestone 1 audit, code,
   tests), one line of justification each.
5. A proposal (for the user to approve, not a change already made) for running fits side by side on the H200: how
   many at once, how the GPU lock is relaxed or replaced, and the memory check that keeps it safe.
6. How results come back: which files the H200 run produces that are git-ignored and must be copied back for the
   gates, and which are committed.
Rules: every guardrail stays in force (test split locked, explicit-path staging, nothing under `shape-form-proton/` or
`shape-form-original/` modified, no data files committed). When done: copy `PLAN.md` to the saved plan file, commit
and push the plan change and the script (labelled as a plan amendment, not a milestone), and report the manifest
summary — file count and total size per group — and anything that could not be verified.

**State when paused** (both workers were stopped a few minutes after starting; section 16 is NOT written)

| Work item | Agent type | STATUS file |
|---|---|---|
| Trace of file reads, group a/b/c manifest, `make_transfer_package.py` with `list` and `verify`, its test, table of run-book commands that exist versus are missing, list of machine-specific constants | Opus | `transfer/STATUS.md` |
| Review: what is machine-dependent and must be recomputed, parity criteria on the H200, mixed laptop/H200 provenance of networks, safeguard against pairing arrays from different GPUs, side-by-side proposal, wording of the A6 amendment | Fable (read-only; its one allowed file is the status file) | `transfer/REVIEW_STATUS.md` |

**Already known and to be stated in section 16**
- Run-book steps without a command today: validation-only recomputation of the pairing arrays (`scripts/parity.py` in
  its default mode opens the test split), re-scoring of already registered runs, the Track A report / guards / Stage 2,
  the remaining Track B runs, the θ track.
- Laptop-specific constants in the code: `scripts/run_m2a.py` (`FORMULA_OF_RECORD = 18.120061`, stops the track if
  step-0 parity differs), `tests/test_refstream.py` (asserts a launch grid of 144), `scripts/audit_lib.py` (`RECORD`).
- `sfm.registry` pairs new scores against whatever arrays sit in `results/m0_parity/` and does not know on which GPU
  they were computed: laptop arrays copied into place would be silently mis-paired on the H200.
- If the H200 has the reference machine's launch grid (1056, readable with `sfm.refstream.local_grid()`), its
  validation numbers should equal the stored 18.118084 / 18.120407 and the test numbers follow by inference from C1.
  Evaluating the two benchmarks on test there would be a third test access: a decision for the user, not taken.

**Where the two workers stopped (25 Nov 2026, 01:15 CET)**
- *Opus, packaging* (`transfer/STATUS.md`): the trace of file reads is mostly done (audit hook plus `strace`, on CPU,
  with opens of the test cache and of any `*.csv` refused); groups (a), (b), (c) are drafted; 20 machine-specific
  constants are listed; the table of existing versus missing run-book commands is drafted. **Not done:**
  `make_transfer_package.py` and its test are not written, `list` never ran (no `files_b.txt`, `SHA256SUMS`,
  `MANIFEST.json`), SHA-256 of most group (b) files not computed, per-file `git ls-files` / `git check-ignore` not yet run.
  Its recommendation so far: copy BOTH cache files and check them by SHA-256 (a rebuild on another CPU is not
  guaranteed bit-identical, and only a byte copy carries Gate 0 and C1 over); laptop per-event arrays go to a
  quarantine folder on the H200, never to the live path.
- *Fable, review* (`transfer/REVIEW_STATUS.md`): questions 1–4 answered, 5–7 partly. Measured on validation: pairing
  the same model across the two random streams gives +2.26 ± 2.71 (×10⁻⁴) where the truth is zero, so a cross-machine
  pairing fakes about 2.3 against the MLP and 3.5 against the formula; no decision of `AUDIT.md` §4 is within reach
  of a stream change (the D6(c) rule would be crossed with probability 0.2 %). It contains predicted H200 validation
  numbers for the two benchmarks and the eight registered networks (valid if the H200's launch grid is 1056), a
  nine-point specification of the mis-pairing safeguard, and a draft of the A6 amendment.

**Decisions the user still has to take for section 16**
1. Test parity on the H200: (A) no test access before the final run, which recomputes the benchmark test arrays as its
   abort-on-fail first step; or (B) a third logged Gate-0-type parity access on the H200 now.
2. Whether the control `b0_seed0` is retrained on the H200 so that the control's seed scatter comes from one machine.

**Gap in the move as stated:** PySR refits and the geometry arm's scores enter paired deltas, so they must be produced
on the H200 even though the searches and the geometry audit stay on the laptop.

**Update, 25 Nov 2026 (midday): the H200 transfer task is complete — see section 16 and §13 (A6 revised).** The
five work items of §15 remain paused until the user gives the go-ahead; the steps listed in §16.0 then run on the H200.

## 16. H200 transfer package (25 Nov 2026)

*Plan amendment, not a milestone. Nothing was trained for this section and nothing has been run on the H200. Sources:
`transfer/MANIFEST.json` (the manifest in machine-readable form, generated by the script below at commit `c9b7f72`),
`transfer/STATUS.md` (how the file reads were traced: an audit hook plus `strace` on CPU, with every open of the test
cache or of a `*.csv` refused before it happened), and `transfer/REVIEW_STATUS.md` (the Fable review of what depends on
the machine). Where this section and those files disagree, the files hold the detail and this section the decisions.*

### 16.0 Decisions taken

**Decided by the user (25 Nov 2026)**
- The H200 is the machine of record for everything moved there and for the final test evaluation; this laptop's
  numbers are quoted as "laptop, other sample points" (§13, A6 revised).
- Moved: the Track A fits (`AUDIT.md` §4.1); the remaining Track B seeds and the ensembles; the remaining Track B runs
  (`AUDIT.md` §4.2); the θ track Θ1–Θ3. Not moved: the PySR searches, the geometry audit, the Fable gates.
- (i) No test access on the H200 before the final run. Parity there is shown on validation; `final_test.py` recomputes
  the two benchmarks' test arrays as its first step and stops before any finalist is loaded if they do not match.
- (ii) The control B0 is trained on the H200 for all three seeds; the laptop's `b0_seed0` is a labelled record.
- (iii) PySR searches stay on the laptop CPU and equations travel as text; every likelihood refit and every registered
  score is produced on the H200. The geometry arm's zero-training scores are produced on the H200 as well.

**Decided by the user on the six proposals of the review (25 Nov 2026). None of the code is written yet (§16.7).**

| # | Decision | Guardrails set by the user | Where it lands |
|---|---|---|---|
| P1 | **Accepted with guardrails.** Fits may run side by side on the H200 only if the throughput benchmark there shows a real gain | `torch.cuda.set_per_process_memory_fraction` set per worker; strictly disjoint output and checkpoint directories; serialized, locked writes to `registry.jsonl`; an exclusive single-job lock kept for parity checks, benchmarks and `final_test.py` | §16.5; enabling change E9 |
| P2 | **Accepted.** The 4-GPU-hour wall-clock cap is dropped; the fixed lists of `AUDIT.md` §4.1 and §4.2 run in full | the pre-registered sets of primary comparisons stay unchanged for the multiple-testing accounting | §13, A6 revised, point 6; E4 |
| P3 | **Accepted with guardrails.** A relaxed per-event clause of the final run's stop rule is possible (≤ 3 events > 1e-4, none > 1e-3, rms ≤ 1e-5) | TF32 explicitly disabled on the H200 (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`); the mean-parity check stays active; if validation parity shows any event > 5e-5, Fable must confirm that the outliers are 1-ULP bin-edge flips before the relaxed clause is activated | §13, A6 revised, point 8; E8, E10 |
| P4 | **Proposal rejected, alternative taken.** The seven test-derived files are NOT transferred now. The file list is split and they are copied to the H200 only after Gate 2c passes, so that their physical absence proves zero test access | — | §16.1, §16.2; the packaging script |
| P5 | **Accepted.** The H200 controls are `b0h_seed0`, `b0h_seed1`, `b0h_seed2`; `b0_seed0` stays untouched as the archived laptop record | — | §13, A6 revised, point 5; E6 |
| P6 | **Modified.** Authoritative H200 results are never placed inside the git-ignored `cache/` | either the machine tag is added to the array and registry schema before anything is synced back, or results are staged under `shape-form-martin/results/h200_records/` (tracked) and promoted only with a machine tag before Gate 1 / Gate 2c | §16.6; E1 |

Reading of two points, to be corrected by the user if wrong:
- **P3, mean-parity bound.** The user wrote "≤ 5e-5". The mean clause already in the stop rule is the C1 criterion of
  4e-6, which is tighter and therefore satisfies that bound; it is kept. (C1 measured 3e-8.)
- **P6, which of the two routes.** The first is taken: the machine tag goes into the schema before anything is
  registered on the H200 (E1 is needed there in any case), so returned arrays carry their tag. The second route is the
  fallback if results exist before E1 is finished.

### 16.1 Manifest of every file the moved steps read

| Group | Files | Bytes | Copied | File list |
|---|---|---|---|---|
| (a) tracked in the pushed branch | 129 | 2,610,270 | arrives with `git clone` / `git pull` | — |
| (b1) git-ignored inputs, to the SAME path | 20 | 2,993,885,926 (2.79 GiB) | **now** | `transfer/files_b_inputs.txt` |
| (b2) git-ignored laptop records, into the QUARANTINE folder | 51 | 5,749,702 (5.48 MiB) | **now** | `transfer/files_b_laptop_records.txt` |
| (b1) test-derived inputs, to the same path | 3 | 69,197,473 (65.99 MiB) | **only after Gate 2c** | `transfer/files_b_test_derived_inputs.txt` |
| (b2) test-derived laptop records, into the quarantine folder | 4 | 239,808 | **only after Gate 2c** | `transfer/files_b_test_derived_records.txt` |
| (c) regenerated on the H200 | 17 entries | — | — | table below |

Group (b) in total: 78 files, 3,069,072,909 bytes
(b1 23 files, 3,063,083,399; b2 55 files, 5,989,510). Copied now: 71 files,
2,999,635,628 bytes. Copied after Gate 2c: 7 files, 69,437,281 bytes.

Every (a) file was confirmed with `git ls-files --error-unmatch` and every (b) file with `git check-ignore -v` (the
ignoring rule is recorded per file in `MANIFEST.json`); no file is untracked and not ignored. Sizes and SHA-256 were
computed by a plain byte read. Cross-checks that agree: both caches against `cache/pion_s1_meta.json` and against the
hashes the Gate 0 run recorded in the tracked `results/m0_parity/parity.json`; the seven pion files against the tracked
`checksums.json`; the eight checkpoints against `checkpoint_sha256` of their registry rows; the eight registered
`val_nll.npy` against the registry's array hashes.

**Test-derived files (user decision P4).** Seven files carry per-event or per-hit information about the test events:
the test cache, the earlier study's stored per-event test arrays `test_v3_nll.npz`, the test event order
`test_event_number.npy`, and the four laptop test arrays. They are not copied with the rest. `verify` on the H200
**fails before Gate 2c if any of them is on the machine** — at its live path, in the quarantine folder, or as a copy of
the same name and size anywhere in the working tree — and reports "test-derived files absent: yes (7 checked)"
otherwise. No moved step, import or test opens any of them (checked on the laptop by making all seven invisible and
exercising the imports, loaders, registry read paths, model factories, script start-ups and test collection).
Two files that travel now mention test only as summary metadata and are needed by moved steps: `PION/test_v3.json`
(the legacy adapter reads the formula's structure from it; it also holds the earlier study's summary test numbers,
which are printed in this plan) and `SFM/cache/pion_s1_meta.json` (hashes and per-split summary statistics).

**Limit of that proof: the Parquet file.** It is not test-derived, but it holds the raw hits of all 99,689 events,
including the 14,956 test events; the split is made only when the cache is built. With the Parquet file on the H200,
the absence of the seven files shows that no stored test file was used; it does not make the test events unreachable.
No moved step reads the Parquet file while the caches are copied. Rules until the user decides otherwise:
`scripts/build_cache.py` is never run on the H200 before Gate 2c (it would write a test cache), and any code that
reads the Parquet file keeps to the training and validation event numbers. Open decision for the user: copy the
Parquet file only after Gate 2c as well.

Abbreviations: `SFM/` = `shape-form-martin/`, `PION/` = `shape-form-proton/results/particles/pion/`,
`DATA/` = `shape-form-original/data/`.

**(a) Tracked (129 files)**

| Kind | Files | Bytes |
|---|---|---|
| code | 48 | 849,275 |
| config | 11 | 8,642 |
| doc | 9 | 463,184 |
| env | 4 | 528,482 |
| package | 6 | 0 |
| record | 34 | 463,764 |
| test | 17 | 296,923 |

Tracked files that are laptop RECORDS and that the code trusts — they arrive with `git pull` whether wanted or not:

- `SFM/results/registry.jsonl` (1 file(s)): Read by registered_row (run_m2b_lib.py:51, run_m2a_lib.py:159; match by checkpoint SHA-256 only) and latest() (select_1se registry.py:521, run_m2b_ensembles._arr :162, run_m2b_report.py:42, run_m2a_lib.val_array :174). On the H200, re-running a registered run's script reuses the laptop row ('already registered; not evaluated again'); until H200 rows are appended, every table/ensemble/1-SE step reads the laptop rows' val_nll_path: FileNotFoundError while the laptop arrays are quarantined (correct), silent …
- `SFM/results/test_access.log` (1 file(s)): Appended by every load_sets(..., allow_test=True) (data.py:253-258). The line holds UTC time, script and reason but no host/GPU: on the H200 the file starts with the two laptop lines and a test access there appends a third line to the H200 copy; Gate 2c's 'exactly two lines' must be restated per machine and the two working copies' logs must be merged by hand.
- `SFM/results/registry_refs/guard_spec.json` (1 file(s)): _bench_guard_arrays (registry.py:323-338) reuses results/registry_refs/*.npy whenever this spec matches and the four arrays exist: laptop arrays at the live path would be reused silently. With them quarantined, the first guarded evaluate() on the H200 recomputes them and rewrites this file with identical content (no git diff).
- `SFM/results/m2b_mlp/*/done.json` (8 file(s)): Skip markers: run_m2b_fit.py:40, fit.py:43, fit_queue.py:68 and run_m2b_stage1.py:233 skip these runs. The laptop networks are kept (intended), but they can be re-scored only by a new script (registry.evaluate directly), never through the run scripts. run_m2b_report.py:56 reads them for training details.
- `SFM/results/m2b_mlp/*/fit_done.json` (3 file(s)): train.fit (train.py:407-416) returns the stored model.pt instead of training whenever this file exists (after checking model.pt's SHA-256). Retraining b0_seed0 on the H200 (user decision, 25 Nov 2026) therefore needs another run name/out_dir; deleting done.json alone would silently reload the laptop network.
- `SFM/results/m2b_mlp/*/resources.json` (8 file(s)): Read-modify-write by run_m2b_fit.py:79-85 only for a run being executed (none of these). Laptop timings.
- `SFM/results/m2b_mlp/s1_seed*/stage1_monitor.json` (5 file(s)): Read by run_m2b_stage1.py:75-76 only when stage 1 resumes; done.json prevents that. Laptop record.
- `SFM/results/m2b_mlp/summary.json` (1 file(s)): Not read by code; rewritten (git diff) by run_m2b_report.py:133 on the H200. Until then it shows laptop numbers (other sample points).
- `SFM/results/m0_parity/parity.json` (1 file(s)): Its existence makes parity.py refuse to run without --overwrite (parity.py:342): keep it that way.
- `SFM/results/m0_parity/c1_reference_stream.json` (1 file(s)): Existence check for parity.py --reference-stream.
- `SFM/results/m0_parity/cache_build.json` (1 file(s)): Rewritten by ANY invocation of build_cache.py (build_cache.py:64-66) unless SFM_RESULTS_DIR points elsewhere.
- `SFM/results/m0_parity/mc_noise_val.json` (1 file(s)): Rewritten by mc_noise_val.py (no argparse).
- `SFM/results/m0_parity/checksums.json` (1 file(s)): check_protected.py verify on the H200 reports every protected file that was not copied as missing (exit 1).
- `SFM/results/m0_parity/checksums_verify.json` (1 file(s)): Not read by code.

**(b1) Inputs — copied to the same path (23 files, 3,063,083,399 bytes; 20 now, 3 after Gate 2c)**

| File | Bytes | SHA-256 | Role | Sample points | Copied |
|---|---|---|---|---|---|
| `DATA/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet` | 2,604,833,654 | `92c060f0b218ffe2fed28609f9dab270c1c3458b9a7ebb482aa9efdece743eb1` | raw hit table (Parquet copy of the CSV) | none | now |
| `SFM/cache/pion_s1_trainval.pt` | 385,953,467 | `9984886fb90214b9c556489bfb3a913aba89509239c9b4a190797a540a6ab4b4` | train/val cache (69,783 + 14,950 events) | none | now |
| `SFM/cache/pion_s1_test.pt` | 68,476,893 | `14955929e0a6cfad42e43aa6fa81787a072d9f26d37c3054ef627faac4c9ec88` | TEST cache (14,956 events) - OPAQUE BYTE STREAM, hash and size only | none | **after Gate 2c** |
| `SFM/cache/pion_s1_meta.json` | 4,051 | `d73b2eb06e3f07a484d92b5391e559cea24c7fd53bf641390bfd0c541313e75e` | cache metadata (geometry, counts, per-file SHA-256 of both caches) | none | now |
| `PION/mlp_std.pt` | 314,242 | `ee1978a4b4cb3b3c1ea16d19e280327b4df433e24506f5a3f782f6d1fc120469` | benchmark MLP checkpoint (fine-tuned 4x32) | none | now |
| `PION/new3_std_n3.pt` | 71,174 | `7c86e2bf39154999ac9091e95757d27332f32dad8d20e8d2ec98ed83a93d3995` | benchmark component formula checkpoint (101 parameters) | none | now |
| `PION/abar.pt` | 17,876 | `32eade22cc46307f0ba7848a65a638b84de334b2d6ac8f36609b66b6c00d3458` | stored acceptance template A-bar (128 x 32) | none | now |
| `PION/test_v3.json` | 3,282 | `a98f38e79de10202ad056700d7f83d0bd2271f999bb5815f57f9b900040a6324` | final-formula definition of the earlier study (degree, variant); stored RESULTS, not test data | none | now |
| `PION/mlp_std.json` | 72 | `a079bfdbfae28e9fffa4963c4f88a0df3618acff66e733bfa77b017d5d383a76` | stored validation mean of mlp_std (reference machine) | reference machine (grid 1056) | now |
| `PION/new3_std.json` | 781 | `32f82c37d15137c0d1b3c8082780cc24aa64a548235fac2113784e7d91e70dbe` | stored validation means of the formula rungs (reference machine) | reference machine (grid 1056) | now |
| `PION/test_v3_nll.npz` | 600,804 | `174524ef734a50ed96a2e7a2a1b8be71349d2efed2da68a6f1702519d32fd81b` | stored per-event TEST arrays of the earlier study (not a laptop record) | reference machine (grid 1056) | **after Gate 2c** |
| `SFM/results/m0_parity/val_event_number.npy` | 119,728 | `3fc56682836e9f62f5a8351ed2f604ae5f29a0727875824cc09813b09b8c7392` | validation event numbers in pairing order | none | now |
| `SFM/results/m0_parity/test_event_number.npy` | 119,776 | `ab5dcf3828d0fd1968d64eae28250a99c835851c9bd7ef60e0c0dc6f595263a4` | test event numbers in test pairing order (order check of the final test) | none | **after Gate 2c** |
| `SFM/results/m0_parity/abar_front.pt` | 17,982 | `e6650b9ba8220e5033d44a7ab9fc3ea6eda5dc6c9da60d603682f5f1e986cc2e` | acceptance template, front face only (A3f, B4f) | none | now |
| `SFM/results/m0_parity/abar_front_back.pt` | 18,017 | `292166d95e08ee4fec677ff000600a0bcd0d383f906289541b2d64af20980c28` | acceptance template, front + back (recomputed) | none | now |
| `SFM/results/m2b_mlp/s1_seed11/model.pt` | 315,778 | `d8f6e94d04b89e5bca1d9473b547ab61f57cef962e9e56884363a15fc84d907c` | checkpoint of the registered Track B network s1_seed11 | none | now |
| `SFM/results/m2b_mlp/s1_seed12/model.pt` | 315,906 | `be1a67e11dc8171c4a620d167e5b84ac4eba13e49fa414417b0e1142955714f6` | checkpoint of the registered Track B network s1_seed12 | none | now |
| `SFM/results/m2b_mlp/s1_seed13/model.pt` | 316,098 | `5e8e7ef9f434550cf60e65e5a7363b8c198a97be2d4893977e8bbcc58a2a22f6` | checkpoint of the registered Track B network s1_seed13 | none | now |
| `SFM/results/m2b_mlp/s1_seed14/model.pt` | 316,290 | `c0d23ad25b8a555bc9b1bef3429fef0e2704d7865af737e1207a19e6fbe3c1f2` | checkpoint of the registered Track B network s1_seed14 | none | now |
| `SFM/results/m2b_mlp/s1_seed15/model.pt` | 315,650 | `95cb4fa4fde34e2dd8a17dd67255fcb7af00bb8b6fc7a458653fd50704752613` | checkpoint of the registered Track B network s1_seed15 | none | now |
| `SFM/results/m2b_mlp/s2_seed11/model.pt` | 317,314 | `0a1c0f02c39349aa43459d10713db8e8b0f12d83a814c04133f4c652238cdf6d` | checkpoint of the registered Track B network s2_seed11 | none | now |
| `SFM/results/m2b_mlp/s2_seed12/model.pt` | 317,314 | `165123ad4ed76680c1bf996b992d8487307369a3f5e2a056d063b1b93790419a` | checkpoint of the registered Track B network s2_seed12 | none | now |
| `SFM/results/m2b_mlp/b0_seed0/model.pt` | 317,250 | `75de31f2540769f68f652979163684213f43e92066f0a6a03f926595ecf6b700` | checkpoint of the registered Track B network b0_seed0 | none | now |

Expected and confirmed: the Parquet file, the pion result files the legacy adapter and the parity checks read, the
eight checkpoints. Not in the user's list but required: both cache files with their metadata file, the two front-face
acceptance templates and the two event-number arrays — the templates and event-number arrays can only be regenerated by
`scripts/parity.py`, whose default mode opens the test split.

**(b2) Laptop records — copied ONLY under `SFM/cache/laptop_records/<same repo-relative path>` (55 files, 5,989,510 bytes; 51 now, 4 after Gate 2c)**

These are per-event arrays computed on this laptop's Monte-Carlo sample points (the pairing arrays the user expected in
group (b) are here). At their live paths the registry would pair every H200 score against them without any error; the
review measured the cost of such a mis-pairing at about 2.3 × 10⁻⁴ against the MLP and 3.5 × 10⁻⁴ against the formula,
against thresholds of 3 and 5. In quarantine they serve as comparison targets only.

| File | Bytes | SHA-256 | Role | Sample points | Copied |
|---|---|---|---|---|---|
| `SFM/results/m0_parity/mlp_val_nll.npy` | 59,928 | `bbc915a1021e2c688c9a82e24baf17cf8b3aae70e5b0ddd8661b9326e008f6af` | Gate 0 benchmark pairing array (mlp, validation) | laptop native (grid 144) | now |
| `SFM/results/m0_parity/formula_val_nll.npy` | 59,928 | `1234970e1160cd8ec3caf4a63c8da92caa2bcc385666ffa53ee572c0d35c0644` | Gate 0 benchmark pairing array (formula, validation) | laptop native (grid 144) | now |
| `SFM/results/m0_parity/mlp_test_nll.npy` | 59,952 | `082a20e522830a371d1537f88b641808e67dc3321c5626fc792f1b5e79bb808d` | Gate 0 benchmark TEST array (mlp), laptop | laptop native (grid 144) | **after Gate 2c** |
| `SFM/results/m0_parity/formula_test_nll.npy` | 59,952 | `ffb37454a3de4523c2b71324e1e2585b278c9662cb1d7d9e6e78f6cd36f811f7` | Gate 0 benchmark TEST array (formula), laptop | laptop native (grid 144) | **after Gate 2c** |
| `SFM/results/m0_parity/mlp_val_nll_ref1056.npy` | 59,928 | `da6248d9985bf4ec41bbdc2ddaaf777577f7adeacec811b1f79a840cc25bc463` | laptop emulation of the reference stream (mlp, val) | laptop, emulated grid 1056 | now |
| `SFM/results/m0_parity/formula_val_nll_ref1056.npy` | 59,928 | `b7f64680b7f6222de3469fe2c4b9fa0e6e1799dcc3d4ad81bcedd3a14330c39d` | laptop emulation of the reference stream (formula, val) | laptop, emulated grid 1056 | now |
| `SFM/results/m0_parity/mlp_test_nll_ref1056.npy` | 59,952 | `eae19ea081fed9910e846b0fe574b98dfa3c4a11e812ee49a45618700632b2ec` | laptop emulation of the reference stream (mlp, test) | laptop, emulated grid 1056 | **after Gate 2c** |
| `SFM/results/m0_parity/formula_test_nll_ref1056.npy` | 59,952 | `3ecc78bb047e2190f95934be9310183f2fd6ff73ac240aebfa31b04ac90c3f3e` | laptop emulation of the reference stream (formula, test) | laptop, emulated grid 1056 | **after Gate 2c** |
| `SFM/results/registry_refs/mlp_val_k1024.npy` | 59,928 | `5c3b790040b51db0af81e9a58593b70051ff7e3f3f79c918362f55b76a7f83a7` | benchmark guard array (mlp, k1024) | laptop native (grid 144) | now |
| `SFM/results/registry_refs/mlp_val_bulk_stratum.npy` | 59,264 | `3cd7d62e9dfce4eeddc1d6a3afc5eee13ef6ea1bd5e6eed01a8e820aedaf5fc2` | benchmark guard array (mlp, bulk_stratum) | laptop native (grid 144) | now |
| `SFM/results/registry_refs/formula_val_k1024.npy` | 59,928 | `af5fe5044f998312808b2dba58111727efedf15581f4b3e3e3497b12114f5540` | benchmark guard array (formula, k1024) | laptop native (grid 144) | now |
| `SFM/results/registry_refs/formula_val_bulk_stratum.npy` | 59,264 | `da1f2a53ee181e449700be58d0711e943c7966d375288d032d0216d88c75cf3f` | benchmark guard array (formula, bulk_stratum) | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s1_seed11/val_nll.npy` | 59,928 | `46017b6548430e2b28f5ee8b3a78a017b59c625eeedee51fc8362ecd4c4873bd` | per-event validation NLL of the registered run s1_seed11 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s1_seed12/val_nll.npy` | 59,928 | `fc817d7e41643f6d5b67bf487575c806274967639844290df2dafe2dd1a57bf8` | per-event validation NLL of the registered run s1_seed12 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s1_seed13/val_nll.npy` | 59,928 | `39f0d958d37a9d0780ad077ccb459a99b60dd585ae78025d50d02f3aadf96e87` | per-event validation NLL of the registered run s1_seed13 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s1_seed14/val_nll.npy` | 59,928 | `b7591dbcf16916f8e46db3d0e49a377be0f3c54374da5cf5b9117b675fe9cf48` | per-event validation NLL of the registered run s1_seed14 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s1_seed15/val_nll.npy` | 59,928 | `efcdb1c96466b4da764fc748ab14e0b6d83f50a250831615c543b15e8afb84e3` | per-event validation NLL of the registered run s1_seed15 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s2_seed11/val_nll.npy` | 59,928 | `eb9d202700788bc94dac05f947f4913073f040a45b278b66fc1a1a9a5a528d31` | per-event validation NLL of the registered run s2_seed11 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s2_seed12/val_nll.npy` | 59,928 | `cf281d03547af30c62f43f0f5c971ff90aab4dc29581cb58ecf8ac7f767da76d` | per-event validation NLL of the registered run s2_seed12 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/b0_seed0/val_nll.npy` | 59,928 | `e1372168c018e911d3d1f7bd66c0dd28eab602114dbabf6f9c986788b935c29d` | per-event validation NLL of the registered run b0_seed0 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/b0_seed0/val_nll_k1024.npy` | 59,928 | `dfeff1095d0e4d43689ba08144aaba87faf1b005f4abfa973f3a04b2ac1c2aaa` | guard array K=1024 of b0_seed0 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s2_seed11/val_nll_k1024.npy` | 59,928 | `eaaeacbd251d6e5315aff19c4b02667f500bc8bb8aa4cacddc4a8ff14157acb9` | guard array K=1024 of s2_seed11 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s2_seed12/val_nll_k1024.npy` | 59,928 | `8addbc79724658071895f10cc0d807cb5b53dbf4e247478fba14c9067ca42869` | guard array K=1024 of s2_seed12 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/b0_seed0/val_bulk_zb1221.npy` | 59,264 | `e4e4fc80e29fe56f208ee7e1fe653ee16d17be54af24c379294cda6746770606` | guard array bulk/stratum of b0_seed0 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s2_seed11/val_bulk_zb1221.npy` | 59,264 | `18c863192d1a6f0c0b869f9e5dc88c052f1897d74c5ca1ffe613911b951edba7` | guard array bulk/stratum of s2_seed11 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/s2_seed12/val_bulk_zb1221.npy` | 59,264 | `2c545806670135abb631e26371cbde743df1421ed3975113f362e8aeedba1ac5` | guard array bulk/stratum of s2_seed12 | laptop native (grid 144) | now |
| `SFM/results/m2b_mlp/b0_seed0/val_mass_monitors.npz` | 958,826 | `e48c1d8ede3d47d2db92bb52ffa13f65b1188417c8d1d83c62eef47770f53901` | mass monitors of b0_seed0 (deterministic map functionals) | none | now |
| `SFM/results/m2b_mlp/s2_seed11/val_mass_monitors.npz` | 958,826 | `d997e6f75ec08984c68d3392520a247ade1744301bbf599db99abc75821fea41` | mass monitors of s2_seed11 (deterministic map functionals) | none | now |
| `SFM/results/m2b_mlp/s2_seed12/val_mass_monitors.npz` | 958,826 | `49305f68f609050bf2bbbe757d3062ec2c4851610a564ec8ce8b2b4074abb9e8` | mass monitors of s2_seed12 (deterministic map functionals) | none | now |
| `SFM/results/gates/gate0_checks/val_grid1056_base_std_n1.npy` | 59,928 | `bcc404dbb3983403a4dcc629dc038dcd547bcd92875bb0a739648c54841c812e` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_base_std_n2.npy` | 59,928 | `37989e8cd13803dad694be5267db609388c58a7e1780066c9b50bcce3af3c3e0` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_base_std_n3.npy` | 59,928 | `76e530fb8dee483828387661ddafa2a39ba35b8f2c0c80d6555ee818227c0201` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_base_std_n4.npy` | 59,928 | `f59cfd19dee69392a20b15c829bcf4100804dfc2a1447d6e669c4e75763503cb` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_base_std_n5.npy` | 59,928 | `a051b71665c90e92a826d9d1b57d951eec2bf9a7e573383bcb86bedc16ebb06b` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_base_std_n6.npy` | 59,928 | `5323822a15f1c286739ef2cb8f7ed71043d03ec40015a54e4404d96968548977` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_mlp_std.npy` | 59,928 | `5f1af0753a0bbdc6a7177f4ed7f961dacbddc1067d249b4a647be3af1cbab226` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_new3_std_n1.npy` | 59,928 | `b121d88780760d1970ab6c4928e87164ad314790849e21a174150ce908309f40` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_new3_std_n2.npy` | 59,928 | `f531c751c820f97b6276b43d07d563c576174bf3e9bee6908dfae3afcfa7d9bb` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_new3_std_n3.npy` | 59,928 | `6982a82292f3aa0fb2cf94ca59a87142a6ede5e75cd9fc4faa251b7d5089794c` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_new3_std_n4.npy` | 59,928 | `aa4ce9b125a1e4f5ca9d1ba22f04cea518eb4bda2cae9417ad4a4037a7db3d51` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_new3_std_n5.npy` | 59,928 | `ed004e1c77eff316dd9992a6a58dfac15363af2a9290e8d26ec241cd0bb2ff30` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid1056_new3_std_n6.npy` | 59,928 | `c6e7fc89a23376c1a76b19df216f477275ed1d6c69543a02013ac11fb79f20d4` | Gate 0 validation arrays on the emulated reference stream (13 models) | laptop, emulated grid 1056 | now |
| `SFM/results/gates/gate0_checks/val_grid144_base_std_n1.npy` | 59,928 | `a62dcbbaa550520ba5966ab6de2ede3630ca41c1af0a01afff4f0090fc80cebc` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_base_std_n2.npy` | 59,928 | `0ba41668005e1d7d2a7bfc5a90310f941968221b4851e9fa9be4298c65f1c042` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_base_std_n3.npy` | 59,928 | `3d2701c9816c6ae1d0d456dd8f9c01a7abe4a931cf1d7a6e2e1298aae4091a4c` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_base_std_n4.npy` | 59,928 | `d6b80c271f01194d387780040f2a57f376a540e591b0d56897a84a0ee1f1747f` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_base_std_n5.npy` | 59,928 | `0b86c92cabe977b3bf16f9bdb5070ab7c0dc7cf4766def0b9361ecb772f64c6d` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_base_std_n6.npy` | 59,928 | `2a6f7abc7258bc33d8bac2b92d03f3913bf773ea27181193f720e0e8e2b092a8` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_mlp_std.npy` | 59,928 | `de0b2731651c58757d4b4037b5611abea8a0996d5828805fb3b6526a23eebf2d` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_new3_std_n1.npy` | 59,928 | `03dea4ea111cc649340774114055911b898d25cd43d1ed579299d03de55111ee` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_new3_std_n2.npy` | 59,928 | `e35f0f74662b4c7a7b629c108fcf5e2d2eb85b51a7f7c85651a5d972c8d067e1` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_new3_std_n3.npy` | 59,928 | `078d0191b4978c0357deb01028764d80e93a03ab8a5b2e78fa456464423c2205` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_new3_std_n4.npy` | 59,928 | `62807dd0ae4c7c0293f57af85e743a3b9d54158761461938d1a8f2608a765973` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_new3_std_n5.npy` | 59,928 | `3c753b679e2805c0c44659c52c966e0b0c3f25be5631fcfee30ae40d0c89d9c6` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |
| `SFM/results/gates/gate0_checks/val_grid144_new3_std_n6.npy` | 59,928 | `0fa899fc22274370fde38c0bde36d9d1ea646619112c391997a8f42100290e7c` | Gate 0 validation arrays on the emulated laptop stream (13 models) | laptop, emulated grid 144 | now |

**(c) Regenerated on the H200**

| Regenerated on the H200 | How | Output exists on the laptop | Laptop time | Reproducibility |
|---|---|---|---|---|
| .venv/ | uv sync --python 3.12   (then confirm torch.version.cuda and torch.cuda.is_available()) | yes | not measured | machine-dependent install (uv.lock pins torch 2.14.0; laptop wheel +cu130, needs a CUDA-13 capable driver) |
| SFM/cache/philox_emul_622ce7d283034712.so | automatic at the first emulator use (sfm.refstream._load: gcc -O2 -fopenmp -shared -fPIC -ffp-contract=off); name = hash of the C source + flags | yes | seconds (not measured) | binary is machine-dependent; its output bits are machine-independent (integer Philox) |
| SFM/cache/pion_s1_val_fingerprint.json | automatic in sfm.registry.validation_reference (registry.py:152-160) when absent or keyed to another trainval SHA-256 | yes | seconds | bit-identical with the copied cache: fingerprint must equal 731a199763cc31c498986574083e95c9c6c61e09d83f69b294cee13cf07858a7 (tracked in … |
| SFM/cache/gpu.lock | created by the first sfm.gpulock.gpu_lock (or SFM_GPU_LOCK) | yes | - | - |
| SFM/results/m0_parity/mlp_val_nll.npy (H200 version) | no command exists yet; needs a validation-only script: load_sets(('train','val')), legacy.load_mlp_std(), common.test_nll inside sfm.metric.deterministic(), np.save; never opens test; records GPU name and refstream.local_grid() | no | 1.48 s (parity.json) | machine-dependent (stream); bit-reproducible on one machine inside deterministic() |
| SFM/results/m0_parity/formula_val_nll.npy (H200 version) | no command exists yet (same script, legacy.load_component_formula(train)) | no | 0.90 s (parity.json) | machine-dependent (stream) |
| SFM/results/registry_refs/{mlp,formula}_val_{k1024,bulk_stratum}.npy (H200 versions) | automatic at the first registry.evaluate(..., guards=True) when the files are absent | no | about 27 s | machine-dependent |
| SFM/results/m2b_mlp/<8 registered runs>/{val_nll.npy,val_nll_k1024.npy,val_bulk_zb1221.npy,val_mass_monitors.npz} + 8 H200 registry rows | no command exists yet; needs a re-scoring script: check model.pt SHA-256 against the laptop row, models.load_run, registry.evaluate(guards=True for b0_seed0/s2_seed11/s2_seed12, False for s1_*) (run_m2b_fit/run_m2b_stage1 skip on done.json and registered_row reuses laptop rows) | no | 5 x 1.5 s + 3 x 16.1 s + ~27 s once = about 1.5 min | machine-dependent; mass monitors to rounding |
| SFM/results/m2a_dist/templates/{back_corrected.pt,full_box.pt,templates_all.npz,templates.json,train_theta.npy} | uv run python -B shape-form-martin/scripts/run_m2a.py templates | no | never run | train_theta.npy bit-identical (float64 copy of cache theta); templates GPU float32 sums: identical up to rounding |
| SFM/results/m2a_dist/{plr.json,decisions.json,gpu_time.jsonl} and <run>/{model.pt,checkpoint.pt,fit_done.json,done.json,init.json,val_nll.npy,...} | uv run python -B shape-form-martin/scripts/run_m2a.py parity plr stage0 stage1p stage1x   (BLOCKED on a grid-1056 GPU by FORMULA_OF_RECORD, run_m2a.py:43/260/266) | no | estimate only: AUDIT 4.1 about 4 GPU-hours; _est_seconds defaults 0.055 s/step (poly), 0.07 s/step … | machine-dependent |
| Track A report / guards / Stage 2 (A11, A11k, A11n) | no command exists yet; phases guards/stage2 import the missing scripts/run_m2a_report.py (ImportError) | no | - | machine-dependent |
| SFM/results/m2b_mlp/{s2_seed13,s2_seed14,s2_seed15,b0_seed1,b0_seed2}/* | uv run python -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/s2_seed13.toml shape-form-martin/configs/m2b/s2_seed14.toml shape-form-martin/configs/m2b/s2_seed15.toml shape-form-martin/configs/m2b/b0_seed1.toml shape-form-martin/configs/m2b/b0_seed2.toml   (precondition: … | no | 238 s fit + 16.1 s guarded evaluation each, about 21 min for five | machine-dependent (training fold points; not bit-reproducible) |
| SFM/results/m2b_mlp/{E5_s2,E6_s2_mlpstd,E2_s2_seeds11-12,E3_s2_seeds11-13,E4_s2_seeds11-14,E5_s1}/*, ensemble_subsets/*.npy, … | uv run python -B shape-form-martin/scripts/run_m2b_ensembles.py   (--table-only: table from saved arrays; precondition: the 8 runs re-scored) | no | never run on the laptop (no measurement) | machine-dependent |
| SFM/results/m2b_mlp/summary.json (tracked, rewritten) | uv run python -B shape-form-martin/scripts/run_m2b_report.py (no argparse; CPU only) | no | seconds (not measured) | deterministic given the arrays |
| remaining Track B runs (AUDIT 4.2) | configs + scripts/fit.py or fit_queue.py work for B2a/b/c, B5f (field_mlp_std), B3s (field), B4f/B4s (field_acc front/stored); no command exists yet for B4x (field_acc rejects full_box), B6a/b/c (no factory), B1d/B1f (distillation), B-ceil (train_fraction) | no | estimate only: about 2 GPU-hours (PLAN 15) | machine-dependent |
| theta track Theta-1..Theta-3 | no command exists yet (own harness forward(lnp, aux); Theta-1b needs the absolute origin in the cache; user decision of 25 Nov 2026: zero-training geometry arm only, after the geometry audit) | no | - | machine-dependent |
| SFM/results/final/LOCK.json and the final test arrays | no command exists yet (final_test.py and LOCK.json missing; after Gate 2c) | no | - | machine-dependent |

**Caches: copied, not rebuilt.** `scripts/build_cache.py` has no argument parser (any invocation rebuilds), always
writes both cache files and the metadata file, and rewrites a tracked file. A rebuild on another CPU is not guaranteed
bit-identical (float64 `cos`/`sin` and the CPU logarithm may take other vector code paths; argued, not tested), and the
code would not notice the difference. Gate 0 and C1 certified exactly these bytes, so only a byte copy carries them
over. The train/validation cache is copied now. **The test cache is copied only after Gate 2c, is checked by a plain
SHA-256 of its bytes, and stays unopened until the single final evaluation.** The validation fingerprint that
`sfm.registry` recomputes on first use must equal `731a1997…58a7`, which is a free check of the copied
train/validation cache.

### 16.2 The packaging script

`shape-form-martin/scripts/make_transfer_package.py` (standard library only; never imports torch or numpy; 22 tests in
`tests/test_transfer_package.py`).
- `list` (laptop): builds groups (b1) and (b2) from an explicit table in the script (each entry names the code that
  reads it; no directory walk), checks each file with git, hashes it once, and writes four file lists, `SHA256SUMS`
  (`sha256sum -c` format, one section per list with its destination and phase) and `MANIFEST.json` into `transfer/`.
  It refuses a file with "test" in its name that is not explicitly classified.
- `verify` (H200, from the repository root), **before Gate 2c**: presence, size and SHA-256 of every "now" file (inputs
  at their live paths, records under the quarantine root); presence of the (a) files and that the checkout descends
  from the manifest's commit; **fails if a stream-dependent laptop record sits at a live path** (it prints the `mv`
  that would quarantine it and executes nothing); **fails if any of the seven test-derived files is on the machine**;
  lists the (c) entries and the tracked-record hazards. Options `--skip-large-hash`, `--json`.
- `verify --after-gate2c`: additionally requires the seven test-derived files, present and correct.
- `verify --source-machine`: for the laptop, where records and test-derived files are at their live paths by definition.
- It refuses any `*.csv` (also while searching the tree) and treats the test cache as an opaque byte stream: the only
  access is a chunked binary read for the hash, and only when a file of that name and size exists. A byte hash is not
  a test access and is not logged.

```
# on the laptop, from the repository root
python3 -B shape-form-martin/scripts/make_transfer_package.py list
rsync -a --files-from=shape-form-martin/transfer/files_b_inputs.txt ./ <user>@<h200>:<repo>/
rsync -a --files-from=shape-form-martin/transfer/files_b_laptop_records.txt ./ <user>@<h200>:<repo>/shape-form-martin/cache/laptop_records/
# on the H200, from the repository root
python3 -B shape-form-martin/scripts/make_transfer_package.py verify

# ONLY AFTER GATE 2c HAS PASSED — on the laptop:
rsync -a --files-from=shape-form-martin/transfer/files_b_test_derived_inputs.txt ./ <user>@<h200>:<repo>/
rsync -a --files-from=shape-form-martin/transfer/files_b_test_derived_records.txt ./ <user>@<h200>:<repo>/shape-form-martin/cache/laptop_records/
# then on the H200:
python3 -B shape-form-martin/scripts/make_transfer_package.py verify --after-gate2c
```

State on the laptop: `list` ran at commit `fcd223a`; `verify --source-machine` passes (80 checks, 0 failures).

### 16.3 Run-book for the H200

All commands from the repository root. "Missing" means no command exists today (§16.7). Durations are LAPTOP
measurements; the H200's are unknown until the calibration of step 4.

| # | Step | Command | Acceptance | Laptop time |
|---|---|---|---|---|
| 1 | Checkout | `git clone` / `git pull`, branch `shape-form-martin`, at or after the commit of this amendment | `git log -1` | — |
| 2 | Environment | `uv sync --python 3.12` then `uv run python -B -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.device_count())"` | `2.14.0+cu130`, `13.0`, `True`, the H200's name. **If `is_available()` is false, stop:** `common.py` would fall back to the CPU without an error. `nvidia-smi` must report a driver CUDA version ≥ 13.0. `gcc` with OpenMP must exist (the stream emulator compiles on first use). One full GPU visible (`CUDA_VISIBLE_DEVICES`), not a MIG slice | not measured |
| 3 | Paths | leave `SFM_DATA_DIR`, `SFM_RESULTS_DIR`, `SFM_CACHE_DIR` **unset**. `SFM_RESULTS_DIR` would hide the pairing arrays, the registry and the done-markers and move the test-access log; `SFM_CACHE_DIR` would move the caches away from where `verify` looks. If the repository is on a network file system, point `SFM_GPU_LOCK` at a local disk | — | — |
| 4 | Copy and verify | the first two `rsync` lines of §16.2 (the "now" lists only), then `python3 -B shape-form-martin/scripts/make_transfer_package.py verify` | PASS, including "test-derived files absent: yes (7 checked)". This is the cache verification for the train/validation cache (SHA-256). The test cache is not on the machine. Never run `scripts/build_cache.py` on the H200 before Gate 2c (it would write a test cache from the Parquet file) | list 2 s, verify 2 s |
| 5 | Machine identity (R0) | `uv run python -B -c "import sys; sys.path.insert(0,'shape-form-martin/src'); from sfm import refstream; print(refstream.local_grid())"` and `uv run pytest shape-form-martin/tests/test_refstream.py -q -k emulation_reproduces_torch_rand_bitwise` | launch grid recorded (expected 1056); the emulator equals `torch.rand` bit for bit. Two other tests of that file are expected to fail on any GPU but the laptop's (§16.7) | seconds |
| 6 | Benchmark parity on validation and the H200 pairing arrays (R1, R1b, R3) | **missing** — a validation-only command; `scripts/parity.py` must not be used (both of its modes open the test split) | if the grid is 1056: float32 validation means 18.118084 (MLP) and 18.120407 (formula) within 4e-6; every event within 1e-4 of the laptop's emulated `*_val_nll_ref1056.npy` in quarantine; formula − MLP = +0.002324 ± 0.000294; a second run bit-identical. Fallbacks in §16.4 | 1.5 s + 0.9 s |
| 7 | Guard reference arrays (R2), once, by one process, before any registration | automatic at the first guarded `registry.evaluate`; to be run deliberately as its own step | grid 1056: K = 1024 means 18.1155556 / 18.1177364, bulk means 18.0429497 / 18.0448778, within 1e-6; stratum of 14,784 events | ≈ 27 s |
| 8 | Re-score the eight laptop-trained networks (R4) | **missing** — the run scripts skip them (tracked `done.json`) and would reuse the laptop rows | checkpoint SHA-256 equal to the laptop rows; values of the table below within 1e-6 on the mean | ≈ 1.5 min |
| 9 | Track B, three fine-tunes | `uv run python -B shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/s2_seed13.toml shape-form-martin/configs/m2b/s2_seed14.toml shape-form-martin/configs/m2b/s2_seed15.toml` | three registered rows; no laptop checkpoint resumed (`s2_seed13` has none) | 238 s + 16 s each ≈ 13 min |
| 10 | Track B, controls | `b0h_seed0`, `b0h_seed1`, `b0h_seed2` (P5): **configs missing**; same script as step 9. The existing `configs/m2b/b0_seed1.toml` and `b0_seed2.toml` are not run. `run_m2b_report.py` must take the control from the `b0h_*` names (today it reads `b0_seed{0,1,2}`) | three H200-trained controls; none has the SHA-256 `75de31f2…` of the laptop's `b0_seed0` | 3 × (238 s + 16 s) ≈ 13 min |
| 11 | Ensembles and report | `uv run python -B shape-form-martin/scripts/run_m2b_ensembles.py` then `uv run python -B shape-form-martin/scripts/run_m2b_report.py` | E5, E6, the size curve; no laptop array read | never run; minutes |
| 12 | Track A fits | `uv run python -B shape-form-martin/scripts/run_m2a.py templates parity plr stage0 stage1p stage1x` — **blocked today**: the parity phase compares with the laptop value 18.120061 and stops the track; the drift and budget rules are laptop-specific (§16.7) | A0 at step 0 equals the H200 formula array (every event within 1e-4, mean within 1e-6) | never run; ≈ 4.0 h estimated, up to 1.3 h more if the lists run in full |
| 13 | Track A rules, guards, Stage 2 | **missing** (`scripts/run_m2a_report.py`) | `results/m2a_dist/RESULTS.md` | ≈ 38 min, inside the 4 h |
| 14 | Remaining Track B (`AUDIT.md` §4.2) | `scripts/fit.py <config>` / `scripts/fit_queue.py <configs>` work for B2a/b/c, B5f, B3s, B4f/B4s once their configs exist; **missing**: the configs, B4x (`full_box` template), B6 (factory), B1d/B1f (distillation), B-ceil (`train_fraction`) | comparators trained on the H200 | ≈ 2.0 h estimated |
| 15 | PySR refits and scores (decision iii) | **missing** (`src/sfm/sr/` translate, audit, targets, refit not written) | every refit and score an H200 registry row | — |
| 16 | θ track Θ1–Θ3 and the geometry arm's scores | **missing** (no code); after the θ-blind finalists are frozen and after the geometry audit | G-bulk headline | — |
| 17 | `LOCK.json` → Gate 2c (takes `verify` first: the seven test-derived files must be absent) → copy the two test-derived lists → `python3 -B shape-form-martin/scripts/make_transfer_package.py verify --after-gate2c` → single final run | the copy and verify commands exist; **missing**: `scripts/final_test.py` | `verify --after-gate2c` PASS; stop rule of A6 revised, point 8 | C1 took 7 s |
| 18 | Results back to the laptop | §16.6 | — | — |

**Does the H200 reproduce the stored 18.1208 / 18.1235 directly?** These are test numbers, and by decision (i) the test
split is not opened on the H200 before the final run. What can be said earlier: if steps 5 and 6 pass with launch grid
1056, the H200 draws the earlier study's sample points, its validation numbers are the stored 18.118084 / 18.120407,
and its test arrays are predicted to equal the stored ones to about 3e-5 per event — every link verified except the
evaluation on the test tensors themselves. That is a strong inference, confirmed or refuted by the first step of
`final_test.py`. That the grid is 1056 is itself an inference (an H200 has the reference machine's GPU layout) until
step 5 reads it.

Targets for step 8 if the launch grid is 1056 (validation, K = 64; Δ in 10⁻⁴ nat against `mlp_std`, paired):

| Network | Laptop Δ | Predicted H200 mean | Predicted H200 Δ |
|---|---|---|---|
| `mlp_std` | — | 18.1180829 | — |
| component formula | +22.04 ± 2.93 | 18.1204074 | +23.24 ± 2.94 |
| `s1_seed11` | +21.81 ± 3.17 | 18.1202702 | +21.87 ± 3.17 |
| `s1_seed12` | +19.25 ± 2.94 | 18.1200090 | +19.26 ± 2.94 |
| `s1_seed13` | +19.00 ± 2.60 | 18.1199865 | +19.04 ± 2.59 |
| `s1_seed14` | +12.57 ± 2.32 | 18.1193732 | +12.90 ± 2.31 |
| `s1_seed15` | +32.94 ± 4.19 | 18.1214207 | +33.38 ± 4.20 |
| `s2_seed11` | +1.82 ± 1.73 | 18.1182652 | +1.82 ± 1.74 |
| `s2_seed12` | +3.29 ± 1.70 | 18.1184227 | +3.40 ± 1.73 |
| `b0_seed0` (laptop record) | +1.19 ± 0.42 | 18.1181833 | +1.00 ± 0.41 |

### 16.4 What carries over and what is recomputed (Fable review)

*With decision P4 the test cache and the other six test-derived files reach the H200 only after Gate 2c; the rows below
that mention them (the test cache's hash, `test_event_number.npy`, C1) apply from that moment on.*

**Carries over unchanged**

| Item | Why it carries over | What shows it on the H200 |
|---|---|---|
| Code and tests | identified by the git commit | checkout descends from `c9b7f72`; the test suite passes except the two launch-grid tests of `tests/test_refstream.py` (§16.7); `test_emulation_reproduces_torch_rand_bitwise` must pass |
| Split, cuts, event order | made on the CPU with `default_rng(1)`; stored in the cache | follows from the cache hashes |
| Both cache files (byte copies, not rebuilt) | Gate 0 and C1 certified these bytes; a rebuild on another CPU may differ in the last bit | SHA-256 equal to the tracked `parity.json` (`9984886f…`, `14955929…`; test file by byte hash only); validation fingerprint recomputed there equals `731a1997…58a7` |
| Benchmark files `mlp_std.pt`, `new3_std_n3.pt`, `abar.pt`, `test_v3.json` | weights and the formula's structure are bytes | SHA-256 of `transfer/SHA256SUMS`; functionally by row R1 below |
| Weights of the eight laptop-trained networks | a network is a fixed function wherever it was trained | SHA-256 equal to `checkpoint_sha256` of the laptop registry rows |
| `abar_front.pt`, `abar_front_back.pt`, the two event-number arrays | made from training events, respectively from the split; no fold points | SHA-256 |
| Gate 0 conclusions on data, code and checkpoints | none uses fold points; they are statements about copied bytes | nothing to redo |
| C1 (exact test parity on the reference stream) | it certified bytes × code × stream 1056; bytes are identical by hash | the two new links (generator, arithmetic) by R0 and R1; the test evaluation itself by the first step of the final run |
| Milestone 1 audit and the run lists of AUDIT §4 | no decision is within reach of a change of stream | nothing to redo; numbers quoted as "laptop, other sample points" |
| Laptop per-event arrays, the eight registry rows, training histories | they are records | kept in quarantine; never at a live path, never in a delta |

**Not carried over** (laptop statements): Gate 0's A1(d) and A3 (statistical parity of the laptop stream), the A2
run-to-run floor, the numbers of record of Gate 0 §2.5, the timing probe, every cost and budget figure.

**Recomputed on the H200, in this order.** Targets marked ◆ hold only if the launch grid is 1056; the fallback
follows the table.

| # | Step | Why | Acceptance |
|---|---|---|---|
| R0 | Identity of the machine: GPU name, `refstream.local_grid()`, torch and CUDA build, emulator against `torch.rand` | the stream is a property of the GPU and of the torch build | grid read and recorded (expected 1056); torch 2.14.0+cu130; `EmulGen(seed, local_grid())` equals `torch.rand` bit for bit |
| R1 | Benchmark pairing arrays on validation: `common.test_nll` inside `deterministic()`, K = 64, both benchmarks | every delta is paired against them | ◆ float32 means 18.118083953857422 and 18.120407104492188 within 4e-6; ◆ against the laptop's `*_val_nll_ref1056.npy`: every event within 1e-4 (expected ≤ 3e-5), means within 1e-6; ◆ formula − MLP = +23.24 ± 2.94; a second run is bit-identical |
| R1b | The laptop stream emulated on the H200 (grid 144), eight networks | separates arithmetic from fold points, whatever the grid | every event within 1e-4 of the laptop's registered `val_nll.npy`, means within 1e-6 |
| R2 | Guard reference arrays (K = 1024; bulk on the stratum), once, by one process, before any registration | guards are paired against them | ◆ means 18.1155556 / 18.1177364 (K = 1024) and 18.0429497 / 18.0448778 (bulk), within 1e-6; ◆ formula − MLP +21.81 ± 2.83 and +19.28 ± 2.75; 14,784 events in the stratum, 166 short, none undefined |
| R3 | Run-to-run floor outside `deterministic()` | Gate 0's 2.9e-5 is a laptop measurement | two runs within 1e-4 per event and 1e-6 on the mean (amendment A2) |
| R4 | Re-scoring of the eight laptop-trained networks, guards for b0_seed0, s2_seed11, s2_seed12 | their laptop arrays are on other fold points | ◆ K = 64 table of session 1 and guard table D: means within 1e-6, deltas within 0.01; mass monitors within 1e-4 relative per event of the laptop's files |
| R5 | Remaining Track B training: fine-tunes of seeds 13–15; the control for seeds 0, 1 and 2 | not yet trained, or (control seed 0) decision (ii) | no numeric target; configuration equal to the pre-registered files; no checkpoint from the laptop resumed; the H200 control for seed 0 has another SHA-256 than `75de31f2…` |
| R6 | Ensembles and their subsets | functions of weights, scored on H200 points | the three representative subsets are bit-equal to their registered arrays; no laptop array is read |
| R7 | Track A: templates, then A0 at step 0, then the list of AUDIT §4.1 | never run | A0 at step 0 against the H200 formula array: every event within 1e-4, mean within 1e-6 (◆ mean 18.120407) |
| R8 | Remaining Track B list (AUDIT §4.2), θ arms | never run | comparators trained on the H200 |
| R9 | Refits, controls and scores of PySR equations; scores of the geometry arm | decision (iii) | each is an H200 registry row; translator round trip within 1e-5 against reference values shipped with the equation |
| R10 | Final test evaluation | single run on the machine of record | stop rule of §13, A6 revised, point 8 |
| R11 | Timing: throughput calibration | costs and slot count are machine properties | §16.5 |

**Fallback when the grid is not 1056** (or R0's generator check fails): the ◆ targets do not apply to the native
arrays. Then (a) R1, R2 and R4 are repeated on the *emulated* grid 1056 on the H200 and must meet the same ◆ targets;
this needs the emulated draw for `sfm.metric.draw`, which exists only for `common.fold` today. (b) R1b must pass.
(c) The native arrays are the new numbers of record and are accepted on A3: correlation with the emulated arrays
≥ 0.9995 and rms of the per-event difference within ±15 % of the laptop's validation values (0.033 for the MLP,
0.036 for the formula; Gate 0 §2.4). (d) Stored arrays are never pairing partners.

**Fallback when a per-event limit is exceeded but the means agree** (another arithmetic, as in measurement E): the
step passes if at most three events exceed 1e-4, none exceeds 1e-3, the rms is ≤ 1e-5 and the means agree within
1e-6. The finding is recorded, and the per-event clause of the final run's stop rule is then set as in §13, A6 revised, point 8.

### 16.5 Running fits side by side on the H200 — approved with guardrails (P1); not implemented yet

Measured on the laptop: 104 deterministic evaluations run two, three or four at a time were bit-identical to the
registered arrays, so concurrency does not change evaluation results; but total throughput did not rise (28.2 → 26.3
training-like steps per second for one → four processes). On one GPU without the multi-process service the driver runs
one process at a time, so the gain is bounded by the idle time of a single job. Whether an H200 behaves differently is
not known.

**The user's guardrails (25 Nov 2026).** Concurrent fits are allowed only if the throughput benchmark on the H200
shows a real gain (the rule below: adopt only if two jobs give at least 1.15 times the throughput of one). Each worker
sets `torch.cuda.set_per_process_memory_fraction` to its declared footprint before its first allocation. Output and
checkpoint directories are strictly disjoint (one run directory, one process, enforced by a per-run lock). Writes to
`registry.jsonl` are serialized under a file lock. The exclusive single-job lock is kept for the parity checks, the
benchmarks (throughput calibration, reference arrays, re-scoring) and `final_test.py`. The table below is the review's
design that implements these rules; where it is weaker than a guardrail, the guardrail applies.

| Point | Proposal |
|---|---|
| Unit | One operating-system process per job (one fit with its registered evaluation, or one evaluation). Never threads: `deterministic()` flips a process-wide switch, `refstream.patched_fold` swaps a module attribute, seeding is process-wide. |
| How many | N is measured on the H200 (below), not chosen in advance. Expected 1–4 without the multi-process service, more only with it (INFERENCE). Memory would allow more than 16. First use: N ≤ 8. |
| Adopt or drop | Adopt only if two jobs give at least 1.15 times the throughput of one. Otherwise N = 1, the exclusive lock stays, and nothing else in this table is built. |
| Lock | N slot files `cache/gpu.slot.<i>`, each an exclusive `flock` taken with the present `gpu_lock` code; a job takes the first free slot. `SFM_GPU_SLOTS` defaults to 1, which is today's behaviour, so the laptop and the tests are unchanged. |
| Exclusive steps | A second file `cache/gpu.excl`: ordinary jobs hold it shared for their lifetime, exclusive steps hold it exclusively. Exclusive steps are started when the queue is empty. |
| One run, one process | An exclusive, non-blocking `flock` on `<run directory>/.run.lock`, taken first and held to the end; the done-marker is read again after it is held. A second process on the same run stops at once and names the holder. One such lock per driver as well. |
| Shared files | Only the single driver process writes `decisions.json`, `plr.json` and the report files. Worker processes write only inside their run directory and append to the registry (already under `flock`). |
| Memory, static | N × P ≤ half of the device memory, where P is the largest device footprint of any job class measured on the H200 (laptop: about 4.6 GB for a fit with a guarded evaluation). |
| Memory, at each start | After taking a slot and before allocating: (a) the sum of the footprints declared by the other slot holders, plus the job's own, plus 4 GB, is at most 80 % of the device; (b) the free memory reported by the device is at least the job's own footprint plus 4 GB. Each slot file records its holder's declared footprint. If either fails: release the slot, wait 30 s, retry. |
| Waiting processes | A process that waits holds only its CUDA context (0.14 GB on the laptop), because `common.py` creates its grid tensors at import. This is harmless on the H200 and is counted by (b). The rules of §15 stay: no tensors on the device before the slot is held, `empty_cache()` before it is released, never wait for memory while holding a slot. |
| CPU | Threads per process limited to cores ÷ N (at most 8); N ≤ physical cores − 2. |
| File system | `flock` must be honoured where the lock files and `registry.jsonl` live. Put the lock files on a local disk (`SFM_GPU_LOCK` already relocates them) and run check E5. |

Why condition (a) is needed: the present check reads the free memory at one instant. With N slots, several jobs can
start together and all see the memory free before any of them has allocated. The declared sum removes that race.
On a dedicated H200 with N ≤ 8 both conditions always hold (8 × 5 GB against 141 GB, the latter INFERENCE); they
matter if the GPU is shared with other users or N is raised.

**How N is determined on the H200**

Nothing is registered and nothing is written into the campaign results (scratch result directory).

- **Workloads.** P1: the Track B training step (`mlp_std` architecture, batch 256, K = 16). P2: a Track A training
  step (`survival_exact` component model, batch 256, K = 16). P3: a guarded evaluation. The timing probe of
  `scripts/parity.py` cannot be used: that script opens the test split.
- **Measurement.** For n = 1, 2, 3, 4, 6, 8, 12, 16: start n identical processes behind a common start signal; each
  runs 20 warm-up steps and then a timed window of at least 20 s, with `torch.cuda.synchronize()` at both ends.
  Record steps per second per process, the total T(n), the lowest free device memory, GPU utilisation and CPU load.
  Three repeats per n; use the median. Do this without the multi-process service and, if the user allows it, with it.
- **Rule.** N is the smallest n with T(n) ≥ 0.9 × the largest T measured, taken separately for P1 and P2; use the
  smaller of the two. Stop measuring once T has stayed below 0.9 × the running maximum for two consecutive n.
  Caps: the static memory rule, the CPU rule, and a per-job slow-down of at most 4. If T(2) < 1.15 × T(1): N = 1.
  P3 is run once at the chosen N, mixed with N − 1 fits, to confirm memory and throughput.
- **Applied to the laptop numbers:** T(2)/T(1) = 0.94 for steps and 1.00 for evaluations, so N = 1.
- **If the multi-process service is used:** read `local_grid()` inside it (must equal the value of R0), set no
  thread limit, and pass check E1 against a reference computed without it. One faulting client can end all
  clients; the fits resume from their checkpoints.

**What stays strictly one at a time**

| Step | Why |
|---|---|
| R0–R4 of §16.4 (identity, reference arrays, re-scoring) | every later delta rests on these files; `_bench_guard_arrays` checks and then writes without a lock; the re-scoring also provides the sequential reference of check E1 |
| The throughput calibration | it measures time |
| Every decision of a driver: learning-rate pilot, A0 parity, A0 drift rule, convergence reruns, keep and equal-NLL rules that trigger second seeds, the content of A11, 1-SE selection, promotion | each is a function of registered arrays and must be evaluated once, by one process, in the pre-registered order, after all its inputs exist. The fits between two decision points can run side by side |
| Report scripts and `LOCK.json` | they rewrite shared files |
| The final test run | single test access; alone on the GPU, no other campaign process alive |
| Commits | taken when no job runs, so that no tracked file is caught half-way |

Dependencies inside Track A that limit what can run together (from `scripts/run_m2a.py`): templates → A0 at step 0
→ three polynomial pilot fits → frozen learning rate → A2s seed 0 → three network pilot fits → frozen learning
rate → A0 (two seeds) → drift rule → A1, A2e, A2s → A2n (needs A2s of the same seed) → twelve polynomial primaries
(need A2s of the same seed) → six network primaries (each needs its polynomial run) and the convergence reruns →
eighteen exploratory fits → second seeds of those that pass → guards → Stage 2.

**The budget rule must not depend on time (approved, P2: recommendation R1 below is the one in force)**

**Found (FINAL).** `scripts/run_m2a.py` line 44 sets `BUDGET_S` to 4 h. Lines 358–405 drop exploratory runs in the
pre-registered order while "seconds used so far + expected seconds of what is pending" exceeds it. Both terms are
wall time: the sum of `gpu_time.jsonl`, and the median seconds per step of finished runs (defaults 0.055 and 0.07 s,
laptop estimates). A faster machine drops fewer runs; jobs side by side take longer each and drop more. The rule of
AUDIT §4.0 ("if the budget runs out, runs are dropped in the order given") is the same quantity. Also: the last
entry of the drop order in AUDIT §4.1, the network runs of primaries that failed on both seeds, cannot be dropped
by the code, because those runs happen before the exploratory ones.

**Recommended (R1): run the lists in full on the H200.** All eighteen exploratory rungs at seed 0, seed 1 whenever
seed 0 passes the keep or the equal-NLL rule, all network primaries; the same for the Track B list of AUDIT §4.2. The
drop logic is switched off and `decisions.json` must show no dropped run. The set of runs then depends only on the
lists and on validation outcomes. Cost at laptop speed: about 4.0 h plus at most 1.3 h of second seeds for Track A.
Statistically nothing is added that was not pre-registered: the full list *is* the list; the drop order was a
fallback for lack of time; the six primaries are unchanged; a rung that is not run cannot enter A11, so dropping
could only remove candidates for a reason unrelated to their merit. It must be recorded before the first Track A
fit; none exists today (measurement F).

**Alternative (R2), if the cap is to be kept:** a budget in nominal units. Each run costs its "Cost" entry of AUDIT
AUDIT §4.1 (laptop minutes, fixed in advance); the cap is 240 nominal minutes; the twelve exploratory rungs that are not
on the drop list run first; the number of second seeds they trigger then fixes, by arithmetic, which rungs of the
drop list run. This is independent of speed and of concurrency, but it drops runs on a machine that has the time.

PySR budgets are wall time by design (minutes per seed on 20 threads) and stay on the laptop, so they do not change.

**Checks that show, on the H200, that side by side equals one after another**

All in a scratch result directory, with the H200 reference arrays passed in explicitly; nothing reaches the
campaign registry.

| # | Check | Criterion |
|---|---|---|
| E0 | `local_grid()` and the generator test inside the configuration used for concurrent jobs | equal to R0 |
| E1 | K = 64 arrays of the eight networks: N processes at once (different networks, then all the same network), other slots busy with the fit probe | array SHA-256 equal to the sequential re-scoring R4, 8 of 8, no tolerance |
| E2 | Guard arrays (K = 1024, bulk) and mass monitors of the three guarded networks, same arrangement | equal hashes |
| E3 | One short fit per model class (Track B network; Track A component model) with `deterministic = True`: alone, beside N − 1 other jobs, alone again | hash over the weight tensors (not the `model.pt` file; `transfer/REVIEW_STATUS.md` §4, point 13), validation history without `seconds`, and registered array: all three runs identical. If the two solitary runs differ, the class has an operation without a deterministic kernel and E3 cannot decide for it |
| E4 | One short fit in the default mode, four times alone and four times under full load, all scored on the same points | mean of "loaded − alone" within 2 SE of zero, SE taken from the alone–alone pairs; rms of per-event differences within a factor 2. Weak with four repeats; E3 is the decisive check. The alone–alone pairs also give the run-to-run scatter that the review could not measure on the laptop |
| E5 | Locks and registry: N processes appending 200 rows each to a scratch registry; two processes on one run directory; N + 1 jobs on N slots | N × 200 valid lines; the second process stops with the holder's name; never more than N at once |
| E6 | After the first real batch: two registered runs re-evaluated alone from their `model.pt` | array hash equal to the row's `val_nll_sha256` |

If E1 or E2 fails: evaluations go back to the exclusive lock. The first thing to try is
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the campaign does not set today; if it is introduced, it is part of the
arithmetic tag and the reference arrays are recomputed under it.

**Option for the user (not recommended as default):** run the Track A fits with `deterministic = True`. Every fit
is then bit-reproducible and a gate can repeat any of them exactly. It costs speed (to be measured in the
calibration) and changes the training configuration T_A before its first use. It must not be applied to the Track B
seeds, whose configurations have to equal the laptop's.

### 16.6 How results come back

Committed on the H200 and pulled on the laptop (tracked): `results/registry.jsonl` (new H200 rows, appended);
per run `done.json`, `fit_done.json`, `resources.json`, `init.json`; `results/m2a_dist/` `plr.json`, `decisions.json`,
`gpu_time.jsonl`, `RESULTS.md`; `results/m2b_mlp/summary.json` and `ensemble_scaling.json`;
`results/registry_refs/guard_spec.json`; new configs and scripts; after the final run `results/final/` JSON files and
`results/test_access.log`. Staging by explicit path, as everywhere.

Git-ignored and to be copied back by hand, because the gates recompute paired differences, selection and hashes from
them and the PySR target export needs the networks:
- per-event arrays: the H200 pairing arrays `results/m0_parity/{mlp,formula}_val_nll.npy`, the guard references
  `results/registry_refs/*.npy`, every run's `val_nll.npy`, `val_nll_k1024.npy`, `val_bulk_zb1221.npy`,
  `val_mass_monitors.npz`, the ensemble subset arrays;
- checkpoints `results/<stage>/<run>/model.pt` of every registered run (not `checkpoint.pt`, not `stage1_fit.pt`);
- the Track A templates (`results/m2a_dist/templates/*.pt`, `*.npz`);
- `fit.log` files if a gate wants the training curves (ignored by the repository-wide `*.log` rule);
- after the final run: the per-event test arrays under `results/final/`.

They must not land on the laptop's live paths, where laptop arrays of the same name exist (the pairing arrays, the
guard references, the eight re-scored runs), and — user decision P6 — authoritative H200 results are never placed
inside the git-ignored `cache/`. Route taken: the machine tag (GPU, launch grid, torch build, TF32 flags, K, seed) is
added to the per-event array and registry schema before anything is registered on the H200 (§16.7, E1), so every
returned array carries its tag, lives in a tag-named location under `results/`, and cannot be paired with an array of
another tag. Its SHA-256 stands in the tracked registry row. Fallback, if results exist before E1 is finished: they are
staged under `shape-form-martin/results/h200_records/` (tracked) and enter no table, delta or rule until a
machine-tagged promotion step has moved them; Gate 1 and Gate 2c check that nothing unpromoted was used. A return-list
mode of the packaging script does not exist yet (E12).

### 16.7 What has to exist before each step can run (none of it is written; all of it needs the go-ahead)

| # | Change | Unblocks | Routing |
|---|---|---|---|
| E1 | Safeguard against mis-pairing: a stream and machine tag on every per-event array and registry row, tag-named reference directories with no fallback, one delta function that refuses mixed tags, `trained_on` / `scored_on` per row, no cross-machine resume (specification: `transfer/REVIEW_STATUS.md` §4, 15 points) | every registration on the H200, and every sync back to the laptop (P6) | Fable specifies and reviews, Opus implements |
| E2 | Validation-only parity and pairing-array command (never opens test) | steps 6, 7 | Opus, Fable reviews |
| E3 | Re-scoring command for already registered runs | step 8 | Opus |
| E4 | Track A driver: paired step-0 parity against the same-stream formula array instead of the constant 18.120061; drift rule on H200 arrays; the budget and drop logic removed so that the lists run in full (P2, approved); the report / guards / Stage 2 script | steps 12, 13 | Opus; Fable for the rule changes |
| E5 | `sfm.refstream` and its tests without the laptop's launch grid as a constant | a clean test suite on the H200 | Opus |
| E6 | Track B §4.2 configs and missing model code; configs `b0h_seed0`, `b0h_seed1`, `b0h_seed2` for the H200 controls and the report script reading the control from these names (P5, approved) | steps 10, 14 | Opus |
| E7 | θ harness `forward(lnp, aux)`; geometry-arm scoring | step 16 | Opus; Fable verifies the acceptance mathematics (Gate 2c) |
| E8 | `scripts/final_test.py` with the stop rule of A6 revised, point 8; `LOCK.json`; refusal of any other machine and of path overrides | step 17 | Opus; Fable at Gate 2c |
| E9 | Slot locks, the per-run lock, `torch.cuda.set_per_process_memory_fraction` per worker, and the exclusive lock for parity checks, benchmarks and `final_test.py` (P1, approved with guardrails) | §16.5 | Opus |
| E10 | A hard failure instead of the silent CPU fallback; TF32 disabled explicitly at start-up and recorded in the machine tag (P3); host, GPU and launch grid in every test-access log line | steps 2, 17 | Opus |
| E11 | `src/sfm/sr/` translate, audit, checks, targets, refit (already on the list of §15) | step 15 | Fable |
| E12 | Return-list mode of the packaging script (the split list for the test-derived files, P4, is done) | step 18 | Opus |

Machine-specific constants found in the code (29 entries; each cited line is re-checked by `list`):

| Where | What | Consequence on the H200 if unchanged |
|---|---|---|
| `SFM/common.py:313` | common.test_nll draws its fold points from the CUDA Philox generator; the stream depends on the launch grid SMs x … | H200: 132 x 8 = 1056 = REFERENCE_GRID (prediction; read refstream.local_grid()). Its validation numbers would be the earlier study's 18.118084 / 18.120407, not the laptop's 18.117857 / 18.120061. A MIG slice differs |
| `SFM/common.py:64` | silent CPU fallback | if the CUDA build/driver fails, every script runs on the CPU (another fold stream, very slow) and run_on_gpu/on_gpu skip the memory check (free = inf, run_m2b_lib.py:121, run_m2a_lib.py:128) |
| `SFM/scripts/run_m2a.py:43` | laptop formula validation mean as the Track A parity target (checked at :260 to 1e-5, stop at :266) | with grid 1056 the value is ~18.120407 (diff ~3.5e-4): the parity phase registers A0_init (registry write), records passed:false in decisions.json and raises SystemExit; Track A cannot proceed until the criterion is restated machine-independently |
| `SFM/scripts/run_m2a.py:172` | A0 drift rule (/A0 - formula/ > 2e-4 -> stop, :313) pairs with the benchmark array | laptop array at the live path: false STOP (+3.5e-4 stream difference); absent: FileNotFoundError; needs the H200 array |
| `SFM/scripts/run_m2a.py:44` | 4 GPU-hour budget; with _est_seconds defaults 0.07 / 0.055 s per step (:371, :374) and 180 s per guard evaluation (:381) | the pre-registered drop order depends on machine speed (and on concurrency, since gpu_time.jsonl sums wall seconds): a fast H200 drops fewer runs |
| `SFM/src/sfm/registry.py:94` | benchmark pairing arrays = 'this machine's stream'; rows record only device 'cuda' and the torch version (:446) | laptop arrays at the live path: every H200 row pairs silently against laptop points (fake +-2.3e-4 vs MLP, 3.5e-4 vs formula); absent: loud error |
| `SFM/src/sfm/registry.py:328` | guard spec has no GPU/stream identity (_guard_spec :296) | laptop registry_refs/*.npy at the live path are reused silently; absent: recomputed |
| `SFM/scripts/run_m2b_lib.py:62` | registered_row matches by checkpoint SHA-256 only (also run_m2a_lib.py:166) | re-running a registered run's script reuses the laptop row; re-scoring is impossible through the existing scripts |
| `SFM/scripts/run_m2b_ensembles.py:177` | scaling table pairs with the benchmark array and the latest rows' arrays (:166; also run_m2b_report.py:42-49, registry.py:542) | before re-scoring: laptop arrays of seeds 11/12 mixed with H200 arrays (if copied) or FileNotFoundError |
| `SFM/src/sfm/refstream.py:51` | launch grid of the laptop GPU | tests that assume it fail (see tests_expected_to_fail_on_h200) |
| `SFM/tests/test_refstream.py:52` | asserts the local grid is 144 | fails on the H200 (1056) |
| `SFM/tests/test_refstream.py:76` | assumes the local stream differs from the grid-1056 stream | fails on a grid-1056 GPU (the streams are identical) |
| `SFM/tests/test_refstream.py:113` | uses a Gate 0 laptop record if present at its live path | with the record quarantined this sub-check is skipped silently (the test still checks the stored mean to 4e-6) |
| `SFM/scripts/audit_lib.py:48` | laptop validation numbers of record | M1 audit re-runs on the H200 would compare against laptop numbers (not a moved step) |
| `SFM/src/sfm/registry.py:440` | absolute laptop paths in rows (also train.py:661 out_dir, config_path in model.pt meta) | informational only; no loader uses them |
| `SFM/src/sfm/train.py:235` | checkpoint.pt stores CUDA RNG states; the fold generator lives on the device (:427); resume checks only config_hash (:438) | resuming a laptop checkpoint.pt on the H200 continues on another stream (hybrid fit); no partial laptop checkpoint is needed |
| `SFM/src/sfm/refstream.py:64` | always cuda:0 (with common.DEVICE) | on a multi-GPU node every job runs on GPU 0 (cache/gpu.lock serialises them); MIG changes the grid |
| `SFM/src/sfm/refstream.py:89` | gcc with -fopenmp must exist (CFLAGS :55) | without it the first emulator use fails (test_refstream, reference-stream checks) |
| `SFM/scripts/run_m2b_lib.py:110` | free-memory threshold tuned for an 8 GB GPU (run_m2a_lib.py:121: 2500) | harmless on 141 GB |
| `SFM/tests/test_families_components.py:532` | cost budget written for the RTX 4060 | passes, or skips on a busy GPU (not verified on an H200) |
| `SFM/scripts/check_protected.py:49` | protected-file manifest incl. Parquet size and mtime | verify on the H200 reports every protected file that was not copied as missing (exit 1); keep the Parquet mtime (rsync -a) |
| `SFM/src/sfm/data.py:135` | float64 numpy transcendental functions in the cache build (also common.py:144 rotation) | numpy uses AVX-512 SVML kernels where available (the laptop i7-14650HX has no AVX-512): a rebuild on the H200 host is not guaranteed bit-identical -> copy the caches |
| `SFM/src/sfm/metric.py:79` | bit-reproducibility per machine only | H200 arrays are reproducible on the H200 and differ from the laptop's |
| `SFM/scripts/parity.py:412` | laptop pairing arrays were computed outside deterministic(); registry arrays inside (registry.py:418) | the H200 recomputation should use deterministic() (differs from an outside run by up to ~2.7e-5 per event) |
| `SFM/src/sfm/data.py:258` | test-access log line has no host or GPU | lines written on the two machines cannot be told apart; Gate 2c's 'exactly two lines' must be restated per machine |
| `SFM/src/sfm/gpulock.py:32` | one flock file per working copy (or SFM_GPU_LOCK) | two checkouts on one node do not share the lock unless SFM_GPU_LOCK points to one file; flock may not work on a network filesystem |
| `SFM/src/sfm/paths.py:26` | SFM_DATA_DIR / SFM_RESULTS_DIR / SFM_CACHE_DIR overrides (:31-33) | the manifest assumes the defaults: setting any of them moves where the code looks for the b1 files |
| `pyproject.toml:5` | no .python-version; the laptop venv is CPython 3.12 | uv may choose another interpreter on the H200: use uv sync --python 3.12 |
| `uv.lock:2450` | torch 2.14.0 from PyPI (laptop: 2.14.0+cu130, CUDA 13.0) | the H200 driver must support CUDA 13.0, otherwise torch.cuda.is_available() is False (silent CPU fallback) |

Tests of the campaign suite on an H200:

| Test | Expected | Why |
|---|---|---|
| `tests/test_refstream.py::test_local_grid_constant_matches_this_gpu` | FAIL | asserts local_grid() == LOCAL_GRID == 144; the H200 grid is 1056 (any GPU but the RTX 4060 Laptop fails) |
| `tests/test_refstream.py::test_other_grids_differ_only_beyond_their_thread_count` | FAIL if the H200 grid is 1056 | requires the local stream to differ from the grid-1056 stream |
| `tests/test_refstream.py (library, bitwise emulation, local/reference stream tests)` | PASS if gcc with OpenMP exists, else FAIL | the .so is compiled on first use; test_emulation_reproduces_torch_rand_bitwise is the check that the H200 torch build has the same Philox layout |
| `tests/test_families_components.py::test_forward_backward_cost_at_batch_256` | PASS or SKIP | budget written for the RTX 4060; skipped when the GPU is busy |
| `tests/test_data.py, test_metric.py, test_registry.py real-data tests` | PASS with the b1 files; skip (cache absent) or FAIL (test_registry needs results/m0_parity/val_event_number.npy) without them | they read the copied cache, meta and event numbers |
| `tests/test_dist_factory.py template test` | SKIP until run_m2a.py templates | HAVE_TEMPLATES |

### 16.8 Not verified

- **Anything about the H200 itself:** its launch grid (1056 is a prediction), driver support for the CUDA 13 build,
  `gcc` with OpenMP, Python 3.12, whether its torch build has the same random-number kernel layout, whether `flock`
  works on its file system, MIG or multi-GPU layout, disk space, whether `rsync -a` preserves modification times
  there, its speed, and whether the predicted values of §16.3 are met.
- **Reads that only happen inside code that was not run:** fits, registry evaluations, guard recomputation and resumes
  are known from reading the code; model builds that need a finished Track A run were not exercised; the ensemble,
  report and Track A scripts were not run because they write or compute.
- **Code that does not exist yet** cannot be traced: the commands marked missing in §16.3.
- **Laptop durations** for the ensembles, the report and all of Track A are estimates; none of them has run.
- **The cache rebuild** being non-identical on another CPU is argued from the code, not observed.
- **Absence of the test-derived files** was simulated on the laptop (the seven files made invisible to Python's `open`
  and `os.stat`); a read that bypasses those would not have been caught, and the campaign's test suite was collected
  but not run in that state. The Parquet file still holds the test events' raw hits (§16.1).
- **Concurrency on the H200:** nothing measured there; the laptop result (no gain) may not transfer.
