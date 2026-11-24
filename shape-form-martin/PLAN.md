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

## 15. Checkpoint — 25 Nov 2026, 23:20 UTC (campaign paused at the user's request)

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
