# Review of the Track B list driver and report (commit `43ccfd6`)

26 Nov 2026. Independent code review (mathematics, statistics, verification). Reviewed: the committed state of `43ccfd6`
(`git archive` export in a scratch directory; the working tree holds another worker's uncommitted changes to
`scripts/run_m2a*.py`, `src/sfm/models/dist_factory.py`, `tests/test_run_m2a.py`, which were not used). Between `43ccfd6`
and `HEAD` (`d5060c6`) only `PLAN.md` and `results/m2a_dist/REVIEW.md` changed, so the code a pull of `HEAD` would run is
the reviewed code. The worker's account (`results/m2b_mlp/STATUS.md`) was read as a list of claims. Nothing under review
was edited; nothing ran on the H200. `results/registry.jsonl`, `results/test_access.log` and `common.py` were hashed
before and after (section 8).

Units: 1e-4 nat per event, paired, ± SE over the 14,950 validation events, unless stated.

**STATUS OF THIS FILE: complete. Verdict: NO-GO at `43ccfd6`** (A1: B1d / B1f starts chosen by validation rank,
excluded by the user's decision; A2: conformance with the hard-stop rule). **GO after a commit with exactly A1 (+ A2 as
decided) and a narrow re-check** (section 7). Report findings B1-B4 (among them an inverted G-K / G-bulk pass test) must
be fixed before the table is read; the report can be re-run from the registered arrays. Judgement items for the
separate reviewer: I11, I12, I13, I17, J-a, J-b (I1, I2, I5 decided by the user).

Since this review began the working tree has also gained uncommitted changes to `scripts/run_m2b_list.py`,
`run_m2b_list_report.py`, `configs/m2b/list_recipes.json` and `tests/test_run_m2b_list.py` (the author's A1 fix in
progress); they were not read.

---

## 1. The run list against AUDIT 4.2 (question 1)

Produced with `run_m2b_list.py --list` from the export (scratch output root; the batch-1 runs show "MISSING" on the
laptop because they exist on the H200 only - the read-only recomputation of section 5.3 recognises all 13 of them on the
H200 records in the repository). 31 runs, 159,000 training steps, plus KL steps (B1d 2 × 6000, B6 6 × 3000, B4 4 × 500).
T_B in the code (`Spec` defaults + `fit_config`): 6000 steps, cosine 1e-3 → 1e-4, batch 256, `k_val` 64, `eval_every`
250, selection last, smooth 0, weight decay 0, `fold_rng` dedicated, EMA 0.999, clip 5 (harness default, as for the
b0h controls), validation seed 12345, batch 128 - equal to AUDIT 4.2 "Fine-tune configuration T_B".

| AUDIT 4.2 row | AUDIT definition (verbatim essentials) | code (`build_list`, factory) | seeds AUDIT / code | comparator AUDIT / code | class AUDIT / code | match |
|---|---|---|---|---|---|---|
| S2-11…15, B0 | taken as given; B0 = `b0_seed0..2` | given, never re-run; B0 = `b0h_seed0..2` (A6 revised point 5, P5 supersedes the AUDIT names) | - | - | given | yes (amended name) |
| B1 | `ensemble(S2-11…15)`, uniform | given (`E5_s2`, batch 1) | - | mlp_std | primary / primary (done) | yes |
| B1c | 31 subsets; E5 + `mlp_std` | given (subset arrays, `E6_s2_mlpstd`) | - | - | exploratory | yes |
| B1d | `FieldNet(32, 4)` from the best member; `kl_warm_start(student, target=E5, 256 training-momentum quantiles, steps=6000, lr=1e-3)`, no data; second run from the second-best member | `field_distill(start=<best|second member>, target=E5_s2, grid=256, kl_steps=6000, kl_lr=1e-3)`, `steps=0`, `data_free`; members ranked by registered val mean (I2) | "2 starts" / `B1d_best`, `B1d_second` | "≤ −3 vs `mlp_std`" / mlp_std | primary / primary | yes (readings I1-I3) |
| B1f | B1d + 3000 steps, 3e-4 → 3e-5, K = 16, T_B otherwise | `from_run(B1d_best)`, 3000 steps, 3e-4 → 3e-5, K 16, EMA 0.999, eval 250 | 0 / 0 | "drifts back by 1-3" / `B1d_best` | exploratory / exploratory | yes (I5) |
| B2a | `field_mlp_std`, T_B, `k_train` 16, iid | same | 0, 1 / 0, 1 | "B2a − B0 within ±1" / mlp_std, Δ B0 shown | comparator / comparator | yes |
| B2b | as B2a, `k_train` 64 | same | 0, 1 / 0, 1 | B2b − B2a / `B2a_s<seed>` | primary / primary | yes |
| B2c | as B2a, `fold = "stratified"` | same | 0, 1 / 0, 1 | B2c − B2a / `B2a_s<seed>` | primary / primary | yes |
| B3 | EMA streams of B2a, B2b, B2c | `<B2x>_ema` rows, EMA − last iterate | - | own last iterate | exploratory / exploratory | yes (I6) |
| B3s | `field(32, 4)` fresh; 12,000 steps, 3e-3 → 1e-4, batch 256, K 16, `smooth` 1e-4, EMA 0.999 | same; eval 250, `k_val` 64, selection last, wd 0 from T_B; **plus** an EMA row `B3s_s<seed>_ema` | 0, 1 / 0, 1 | "+5 … +20 vs mlp_std" / mlp_std | exploratory / exploratory | yes, one extra labelled row (I6, I7) |
| B4f | `field_acc(template="front", base="mlp_std", warm_steps=500)`, T_B, K 16 | same (factory defaults: warm-start grid 64 points uniform in ln p, warm lr 1e-3) | 0, 1 / 0, 1 | B4f − B2a / `B2a_s<seed>` | primary / primary | yes |
| B4s / B4x | `template="stored"` / `"full_box"` | same; front/stored files checked against `transfer/SHA256SUMS`, full_box against the H200's Track A `templates.json` | 0 / 0 | (as B4f) / `B2a_s0` | exploratory / exploratory | yes (I16) |
| B5f | `field_mlp_std`, T_B, K 16, `smooth` 1e-4 (against B2a) | same | 0, 1 / 0, 1 | B2a / `B2a_s<seed>` | exploratory / exploratory | yes |
| B6a | `field(32, 4)` fresh, `kl_warm_start(target=mlp_std, 256 quantile momenta, steps=3000, lr=1e-3)`, T_B K 16 | `field_kl(32, 4, target=mlp_std, grid=256, kl_steps=3000, kl_lr=1e-3)` | 0, 1 / 0, 1 | "B6a − mlp_std within ±2" / mlp_std | comparator / comparator | yes |
| B6b | `field(32, 5)`, same route | `field_kl(32, 5, …)` | 0, 1 / 0, 1 | B6a / `B6a_s<seed>` | exploratory / exploratory | yes |
| B6c | `field(64, 4)`, same route; 1-SE rule | `field_kl(64, 4, …)`; 1-SE per seed in the report | 0, 1 / 0, 1 | B6c − B6a / `B6a_s<seed>` | primary / primary | yes (I17) |
| B-ceil | `residual_table(base="mlp_std", knots="uniform", freeze_base=True)`: (1 knot; wd 0, 1e-4; 100 %), (6 knots; wd 0, 1e-4; 25, 50, 100 %); 3000 steps, 3e-3 → 3e-4, K 16, `eval_every` 100; last iterate and trajectory minimum | 8 runs exactly so; knots at 0.0895, 0.264, 0.78, 2.3, 6.78, 20 GeV (one knot: 0.0895 GeV, momentum-independent table); subsets by `Showers.subset` of a permutation seeded 0 (nested 25 % ⊂ 50 %); batch 256, `k_val` 64, selection last from T_B; EMA computed but not registered | 0 / 0 | (base) / mlp_std | diagnostic / diagnostic | yes (I7-I9) |
| B7 | closed | listed as closed | - | - | closed | yes |
| primaries | B1, B1d, B2b, B2c, B4f, B6c | `PRIMARY_RUNGS` identical | | | | yes |
| convergence (4.0) | moved < 0.5 over the last 1000 steps; non-converged primary rerun once with doubled steps with its comparator | `|val(last) − val(last − 1000)| < 0.5e-4`; `<name>_x2` with `2 × steps`, its comparator `<comp>_x2` once (pinned in `x2_recipes`) | | | | yes in the driver; **not used by the report** (finding B2) |
| secondary columns (4.0) | p < 2 / ≥ 2 GeV; contained / side-leaking | same definitions as the Track A report (`dist_factory.exit_depth`, 1450 mm) | | | | yes |

Nothing of the list is missing, nothing is changed. Additions beyond the text, all labelled and outside every rule: the
EMA rows of B3s, the unregistered EMA stream of B-ceil (wall time only), the G-region reference arrays of `mlp_std`, and
in the smoke mode the stand-in ensemble `E5_smoke`. Constants: `count_params` gives 3,841 / 4,897 / 13,825 for
(32, 4) / (32, 5) / (64, 4) and 3,841 + 4,096 per knot for B-ceil, as the AUDIT's Constants column.

## 2. The 17 readings of STATUS.md section 3 (question 2)

"Literal" = is the author's reading the most literal reading of AUDIT 4.2 / 4.0 (with the quoted text). "Class" =
**mechanical** (follows from the text or the campaign's own conventions; no ruling needed) or **judgement** (a
mathematical or statistical question a separate reviewer must rule on; the question is stated). "Trains?" = whether a
different ruling would change what is trained on the H200 (such readings must be settled BEFORE the run: A6 revised
point 5 forbids repeating a run to choose between outcomes, and run names cannot be reused).

What the list says about B1d's starts (AUDIT 4.2, row B1d, verbatim): Model and initialisation "`FieldNet(32, 4)` from
the best member; `kl_warm_start(student, target=E5, lnp_grid = 256 training-momentum quantiles, steps=6000, lr=1e-3)`,
no data. Second run from the second-best member"; Training "data-free"; Seeds "2 starts"; Cost "2 × 3 min"; Class
"**primary**"; Expected "≤ −3 vs `mlp_std`: 35 %; retains 30–70 % of B1". The criterion for "best" is not stated in the
row. The AUDIT's own usage elsewhere is by validation score: section 1, row "D8: three local members, − `mlp_std` / −
best member", where "best member" is the member with the lowest validation NLL.

**I1, I2 and I5 are decided (orchestrator's note during this review, relaying a user decision):** the two B1d starts
must be chosen by a strictly validation-blind rule - the fixed seed order `s2_seed11` and `s2_seed12` - and the same
principle applies to B1f's start. The author is making that change in the working tree (a later commit, not reviewed
here). These three rows are therefore recorded as **finding A1 (fixed by user decision, to be verified in the
re-check)**, not as open questions; the analysis below is kept because it is what the re-check has to confirm.

| # | reading taken | literal? (quoted text) | class | trains? |
|---|---|---|---|---|
| I1 | B1d's two starts are the two replicates; part (d) of B1d compares `B1d_second` with `B1d_best` | **yes** - Seeds column "2 starts" puts the starts where the other rows have seeds; AUDIT 4.0 "a result counts only if the second seed gives …" | **judgement**: does a second data-free distillation from a different starting member, to the *same* fixed teacher E5 on the *same* 256 grid points, satisfy "reproduced by a second seed" (PLAN 6) / the AUDIT 4.0 replication rule? The two runs share every source of randomness except the start (no data order, no fold points, one realisation of E5), so they test only the dependence on the start, not the sampling variability of the teacher; and the second start is itself selected (I2). Alternatives: accept as is; or treat B1d's (d) as "not applicable - single teacher" and require something else (e.g. replication through B1f, or a distillation of a different ensemble realisation). **Status: decided with I2 (finding A1).** With the starts fixed as `s2_seed11` / `s2_seed12` the "second start = replicate" structure remains; the re-check must confirm that the report's part (d) for B1d pairs the two new runs and that no remaining code path ranks them. The statistical point above (one teacher, so the replicate tests only the start) is unaffected by the decision and should be stated beside B1d's (d) in the table. | no (report only) |
| I2 | "best" / "second-best" member = lowest / second-lowest registered K = 64 validation mean of `s2_seed11..15` on the live stream (H200: `s2_seed14`, then `s2_seed13`) | **yes** - the row gives no criterion; the only registered score is the validation mean, and the AUDIT uses "best member" in that sense (D8 row quoted above) | **judgement**: the starts are selected on the validation events on which B1d is then scored. On the H200 arrays (recomputed, section 5.3): best − second = −1.19 ± 1.28, second − third (`s2_seed13` − `s2_seed11`) = −0.09 ± 1.51, i.e. the order is not resolved and the second start is decided by noise. The members' spread (+0.54 … +6.54 vs `mlp_std`) is of the size of the promotion threshold (−5), so whatever part of the start's validation advantage survives 6000 KL steps enters B1d's Δ as a selection effect. Question (as it stood): is selecting the B1d starts by validation rank acceptable, or must the starts be fixed by a validation-free rule? **Status: decided - validation-blind, fixed seed order `s2_seed11`, `s2_seed12` (finding A1).** The reading taken at `43ccfd6` (literal by the AUDIT's own usage) is thereby rejected; the code at `43ccfd6` must not run (`run_m2b_list.py` lines 350-362 `member_ranking`, 372-377 `resolve`). Note for the re-check: `s2_seed11` and `s2_seed12` are laptop-trained records re-scored on the H200; `lib.member` resolves them to `repo:m2b_mlp/s2_seed1x` (the clone's record files), and `verify_sources` then needs the H200 row of each whose `checkpoint_sha256` equals that file (the H200 rows exist: `0a1c0f02c3…`, `165123ad4e…`). | **yes - settled before the run** |
| I3 | the 256 quantiles at probabilities (i + ½)/256 (midpoints) | yes - "256 training-momentum quantiles" does not fix the probability levels; midpoints are the standard quadrature of E_p[KL] over the training momentum distribution and keep the grid inside the range `kl_warm_start` enforces. Note: the code's own precedent `_knots(..., "quantile")` uses `linspace(0, 1, n)` (end points included), which would also pass the range check; the readings differ only at the two extreme points | mechanical | yes, negligibly; author's choice acceptable |
| I4 | B1d: no validation trajectory, so the convergence rule is not applied | yes - AUDIT 4.0 defines convergence on "the validation mean … over its last 1000 steps"; B1d has none | mechanical (the KL trajectory is in `init.json` and can be read at the gate) | no |
| I5 | B1f from `B1d_best`, one run (seed 0) | yes at `43ccfd6` - "B1d + 3000 steps", Seeds "0", Cost "2 min" | **decided (finding A1)**: B1f's start must follow the same validation-blind rule (the B1d run from the first fixed start, not a run named or chosen by rank); `run_m2b_list.py` lines 170-171 and 378-379 at `43ccfd6` | yes - settled before the run |
| I6 | each EMA state registered as `<run>_ema` (exploratory), compared with its own last iterate; B3s also gets an EMA row | yes for B3 ("EMA streams of B2a, B2b, B2c"; T_B "the EMA stream is evaluated separately"). The B3s EMA row is an addition; B3s's own row is the last iterate, as AUDIT 4.0 "Selection of every new fit: last iterate" requires | mechanical | no |
| I7 | unspecified settings of B3s and B-ceil from T_B (`k_val` 64, selection last, `eval_every` 250 for B3s, dedicated folds, EMA 0.999; no EMA row for B-ceil) | yes - nothing else is defined; selection last is AUDIT 4.0; B-ceil's EMA does not change its weights (wall time only, finding C2) | mechanical | no effect on the weights |
| I8 | B-ceil `train_fraction`: first round(f N) of a permutation of the training split seeded with the run seed (0), storage order kept; index SHA-256 recorded | yes - AUDIT note "`Showers.subset` on a seeded permutation". Consequence (good for a learning curve): the 25 % subset is contained in the 50 % one | mechanical | yes; unambiguous |
| I9 | B-ceil "minimum over the trajectory" = minimum of the K = 64 validation means of the history (every 100 steps), printed, not paired | yes - "report the last iterate and the minimum over the trajectory". The printed difference to the `mlp_std` mean is exact (mean of differences = difference of means on the same points); only its SE is missing; optimistic by construction, labelled so | mechanical | no |
| I10 | comparators: B2a, B3s, B6a, B-ceil against `mlp_std` (B2a also against B0); B1f against `B1d_best` | yes - each follows the row's Expected column: B2a "B2a − B0 within ±1", B3s "+5 … +20 vs `mlp_std`", B6a "B6a − `mlp_std` within ±2", B-ceil's base is `mlp_std`, B1f "drifts back by 1–3" (relative to B1d) | mechanical | no |
| I11 | the Track B thresholds (−5 and −3 SE) and the AUDIT 4.0 replication rule applied to the primary comparisons (B2b − B2a etc.); promotion itself against `mlp_std` | **not uniquely**: AUDIT 4.0 states for Track B only the promotion rule "Δ ≤ −5 and ≤ −3 SE against `mlp_std` …"; the rows say "**primary** (B2b − B2a)" without a threshold, and their Expected columns quote "≤ −3: 5 %" | **judgement**: which decision rule governs the primary comparisons B2b − B2a, B2c − B2a, B4f − B2a, B6c − B6a: (i) the promotion thresholds and replication applied to the comparison (author); (ii) only the promotion rule against `mlp_std`, the comparison being the pre-registered *descriptive* contrast; (iii) another threshold (e.g. −3, as the Expected columns quote)? And how does a "pass" of the comparison combine with promotion? | no (report only) |
| I12 | "better than B0 by > 1 SE" = z of the paired Δ against the per-event mean of `b0h_seed0..2` below −1 | yes - PLAN 6 "better than B0 by > 1 SE"; A6 revised point 5 "the B0 mean use these three runs only"; guardrail "paired SE everywhere" | **judgement**: the per-event paired SE against the B0 mean treats B0 as fixed. The three controls scatter by sd ≈ 0.41 between seeds (+1.01, +1.04, +0.31 vs `mlp_std`), the same size as the paired SE of a variant against the B0 mean (≈ 0.4). Recomputed on the H200 arrays: the control `b0h_seed2` itself is "better than B0" at z = −1.21, so part (c) is passed by a plain control. Question: should (c) use (i) the paired SE against the per-event B0 mean (author), (ii) an SE that includes B0's between-seed variance (e.g. √(SE²_paired + s²_seeds/3)), or (iii) a comparison with each control (all three, or their seed-matched one)? | no (report only) |
| I13 | replication: seed-1 Δ ≤ −2.5 with the same sign, and the two-seed mean (per-event mean of the two runs against the reference) meets (a) **and** (b) | **not uniquely**, and **inconsistent with Track A**: AUDIT 4.0 "the second seed gives Δ ≤ half the threshold with the same sign and the mean of the two seeds meets the threshold". The Track A report (three reviews) reads "the threshold" as the Δ part only for both clauses and prints the SE version as a labelled alternative; this report uses the Δ part for the first clause and Δ **and** SE for the second | **judgement**: for Track B, is "the threshold" in the replication rule the Δ part only (−5; half −2.5) or the full promotion threshold (−5 and −3 SE; half −2.5 and −1.5 SE)? One ruling for both clauses and both tracks. (Also implied by the code: seed 0 is the result, seed 1 the replicate - the literal "second seed".) | no (report only) |
| I14 | guards only for runs meeting (a)-(c) on a seed, plus the other seed of the rung; G-region with `run_m2a.g_region_compute`, `mlp_std` as reference | yes - AUDIT 4.2 note "Guards for every candidate that meets the promotion rule": (d) needs the other seed and (e) is the guard itself, so (a)-(c) is the non-circular reading; the reference of promotion is `mlp_std` | mechanical (but the x2 reruns are not candidates: finding B2) | H200 computation, additive (can be added later) |
| I15 | a comparator shared by two primaries is rerun once (`B2a_sX_x2`) | yes - "rerun once with doubled steps together with its comparator; nothing else changes" | mechanical | yes; unambiguous |
| I16 | templates: front = `abar_front.pt`, stored = `abar.pt` (both against `transfer/SHA256SUMS`), full_box = the H200's Track A template (against its `templates.json`) | yes - AUDIT "Template definitions for A3 (and B4)": front and stored "(exists)", full_box the new kind of `acceptance_template`, computed from the training events on the machine of record (A6 revised point 1) | mechanical | yes; unambiguous |
| I17 | 1-SE for B6 per seed among B6a, B6b, B6c (`registry.select_1se`) | not uniquely - B6c "capacity; 1-SE rule" does not say per seed or pooled | **judgement** (narrow, low stakes): per seed (two selections that may disagree) or once on the per-event two-seed means of B6a, B6b, B6c; and what is reported if the per-seed selections differ? | no (report only) |

Two further points need a ruling and are not in the author's list:
- **J-a (guards, part (e)).** PLAN 13 "Guard definitions refined": the persistence margin is "called resolved only at
  2 SE". The report counts a G-K / G-bulk guard as passed only if it passes **and** is resolved; the Track A report
  prints "passed" and "resolved" side by side without combining them. Question: does an unresolved pass (|z| < 2)
  count as "guards passed" for promotion, as a fail, or as "undecided"? (Independent of finding B1, which is a sign
  error.)
- **J-b (G-region).** The report passes G-region on the point estimate (margin flagged − ½ main ≥ 0) without a
  resolution flag; the Track A report adds "resolved at 2 SE". Same question as J-a for G-region.

### 2.1 Sweep: every place where a validation number decides what is TRAINED (orchestrator's point 2)

Line numbers are those of `43ccfd6`. "Allowed" = a pre-registered rule deciding what is reported or re-run (the user's
distinction); "leak" = a validation number choosing a starting point, replicate, variant or schedule.

| # | file:line | what the validation number decides | verdict |
|---|---|---|---|
| V1 | `scripts/run_m2b_list.py:350-362` (`member_ranking`: `ranked.append((r["val_mean"], n))`, sorted), used at `:372-377` (`resolve`: `kw["start"] = lib.member(rank[0] if … "<best member>" else rank[1])`) | the two starting networks of B1d | **leak - finding A1 (decided by the user: fixed seed order `s2_seed11`, `s2_seed12`)** |
| V2 | `scripts/run_m2b_list.py:164-169` (specs `B1d_best` / `B1d_second`, placeholders `<best member>` / `<second member>`), `:170-171` and `:378-379` (B1f `from_run` `<B1d_best>`) | B1f's starting network (the B1d run whose start ranked first on validation) | **leak - finding A1 (decided)**; the names `B1d_best` / `B1d_second` / `<B1d_best>` and the pinned recipe hashes `B1d_best`, `B1d_second`, `B1f_s0` in `configs/m2b/list_recipes.json` carry the ranking and must change with it |
| V3 | `scripts/run_m2b_list.py:721-731` (`smoke_prep` calls `member_ranking`) | the members of the smoke stand-in ensemble `E5_smoke` | smoke only, not of record; the order does not change a uniform ensemble. Harmless, but the fix should not keep `member_ranking` alive for any real-mode path |
| V4 | `scripts/run_m2b_list.py:760-790` (`phase_converge`: `move = abs(last["val"] − prev["val"])`, `:778`; reruns `:781-788`) | whether a primary and its comparator are re-trained as `<name>_x2` with doubled steps | **allowed** - the pre-registered convergence rule of AUDIT 4.0 deciding a re-run (it does not choose a start, variant or schedule other than the registered doubled one) |
| V5 | `scripts/run_m2b_list.py:480-500` (`divergence_check`: non-finite last validation mean; last mean > min(history ∪ comparator's registered mean) + 1 nat), `:562-566` (`_comparator_mean` reads the comparator's registered `val_mean`), `:587-594` (a diverged source blocks the runs started from it) | whether a run is "diverged", and thereby whether a dependent run is trained (only B1f depends on another run of the list, B1d; B1d is data-free, so it can diverge only through a non-finite KL, `:612-613`, or a non-finite registry score, `:662-664`) | **allowed** - PLAN 20 item 2 (user decision) defines divergence; blocking a dependent of a diverged run is a stop, not a choice between outcomes. (The 1-nat blow-up threshold is the implementation's, the same as Track A's reviewed `BLOWUP_NAT`) |
| V6 | `scripts/run_m2b_list.py:793-806` with `scripts/run_m2b_list_report.py:169-184` (`guard_candidates`: parts (a)-(c) on validation) | which runs get guard EVALUATIONS (K = 1024, bulk, mass, G-region) - no training | **allowed** (AUDIT 4.2 note); the forced smoke candidate `:797-798` is smoke only |
| V7 | `scripts/run_m2b_list.py:134` (`selection = "last"` for every spec), `:556` (passed to `FitConfig`); `src/sfm/train.py:563-575, 590-599` (best states are kept only under `selection == "best_val"`, never used here; the EMA state is selected by the same "last" rule) | nothing: no checkpoint, early stop or schedule is chosen on validation | clean |
| V8 | factories `field_distill`, `field_kl`, `field_acc` (`src/sfm/models/__init__.py:139-158, 199-231`), B-ceil subset (`run_m2b_list.py:461-469`) | nothing: KL warm starts are data-free on the training-momentum grid (B4: uniform ln p grid); the subset is a seeded permutation | clean |

No further leak was found. The report's 1-SE selection, B-ceil trajectory minimum and retention fraction are reporting
only.

### 2.2 Hard stops on the H200 (orchestrator's point 3: "any failed parity, reference, scoring, guard or file-access check is a hard stop")

Every check below raises an uncaught exception or `SystemExit(message)`; the driver has no broad `except` (the only
handlers catch `FloatingPointError`, `run_m2b_list.py:612` and `:662`), so the process ends with exit status 1, no later
phase runs, and the launcher's `.exit` file records it. Resuming is the same command.

| check | where | behaviour at `43ccfd6` |
|---|---|---|
| start-up (CUDA, TF32, pristine output root, file-access guard active off the laptop) | `lib.start` → `machine.startup`; again in `train.fit` and `registry.evaluate` | **stops** |
| reference set: missing, parity verdict not PASS, disputed, wrong machine/points, array hash | `registry.reference_set` (`phase_verify`, and inside every `registry.evaluate`) | **stops** (with `verify` skipped, at the first registration, i.e. after one fit) |
| parity itself | not run by the driver; relied on through the reference set's parity record | stops via the reference set |
| validation-set identity | `registry.check_validation` in every `evaluate` | **stops** |
| batch-1 rows, starting files, `mlp_std.pt`, templates, recipe pins | `phase_verify`, `verify_sources`, `verify_template`, `check_pins` (`SystemExit`) | **stops** |
| tagged array would be overwritten / mixed tags / marker of another machine | `provenance.save_tagged`, `check_pairable`, `done_marker`, `diverged_marker`, `train._check_machine` | **stops** |
| file-access guard refusal raised in campaign code | `sfm.guard` (`GuardRefused`, a `PermissionError`) | **stops** |
| file-access guard refusal swallowed by third-party code (the tolerated `pyarrow` imports of pandas) | outside the driver | **continues, exit 0** - visible only in the guard log; the operator must read it (STATUS's acceptance line). Also `registry._trained_on_from_checkpoint` (base module) has `except Exception: return None`, which would swallow a refusal there |
| non-finite per-event validation NLL at registration (a scoring failure of one run) | `run_m2b_list.py:660-664` | **continues**: recorded as a divergence (exploratory: "diverged"; primary/comparator/diagnostic: "diverged - awaiting diagnosis"), as PLAN 20 item 2 decided. Under the new standing rule this is a conflict the orchestrator must resolve: either "non-finite score of a run = divergence (continue)" stays, or it becomes a hard stop |
| non-finite KL in a factory's warm start | `run_m2b_list.py:610-613` | continues, recorded as a divergence (same conflict) |
| guard COMPUTATION errors (missing guard arrays in the reference set, I/O) | `registry._guards`, `reference_set(need_guards=True)` | **stops** |
| guard RESULTS (a candidate failing G-K / G-bulk / G-region) | report | continue - these are results, not failed checks |
| G-region decomposition identity (parts vs the registered array) | **not computed at all** in Track B (Track A's `g_region_arrays` does it) | cannot stop anything - finding C6 |
| clone no longer clean at the end of the job | `lib.finish` (`run_m2b_lib.py:81-88`) | **continues: prints a WARNING, exit 0** |
| report with fewer than three B0 controls | `run_m2b_list_report.py:296-298` | continues, exit 0 (the batch-1 report exits 2 in that case); in the real mode `phase_verify` already stops if a b0h row is missing |
| GPU memory below cap + 4 GB at a job start | `lib.run_on_gpu` → `machine.check_free_memory` | stops |

## 3. Models and training code against their definitions (question 3)

- **Distillation (`field_distill`, B1d).** Student = `field_model(src.hidden, src.n_layers, src.n_freq)` carrying the
  start's selected weights (`load_run(start)` → `load_state_dict`); the members are `FieldNet(32, 4)` (3,841), so the
  student is the AUDIT's `FieldNet(32, 4)`. Objective (`models/field.py:97-143`): mean over the grid of the forward KL
  Σ_cells P_E5 (ln P_E5 − ln P_student), P_E5 = exp(log P) of the teacher evaluated once in `eval()` under `no_grad`;
  Adam, constant lr 1e-3, 6000 full-batch steps on the 256 grid points; a non-finite KL raises `FloatingPointError`,
  which the driver records as a divergence. The teacher `Ensemble.forward` is `logsumexp_m(log_softmax(w)_m + log P_m)`.
  With uniform weights over midpoint quantiles the loss is the quadrature of E_{p ~ train}[KL(P_E5 ‖ P_student)] - the
  right objective for "distil the ensemble's map". No data, no validation. On a rebuild (`skip_init`) the KL fit is
  skipped and the stored weights are loaded (`run_m2b_list.py:629-637`). Correct.
- **KL warm start for the capacity variants (`field_kl`, B6).** Fresh `field_model(h, L)` initialised after
  `seed_everything(seed)` (`run_m2b_list.py:607-611`, so B6a/b/c of one seed are reproducible), KL to `mlp_std` on the
  256 training quantiles, 3000 steps, lr 1e-3; any target other than `mlp_std` is refused. Correct.
- **Acceptance on the logits (`field_acc`, B4).** `FieldAcc`: log P = log_softmax(logits + ln max(A, 1e-6)) (ln of the
  clamped template, never of a softmax); base = `mlp_std`'s weights, warm-started by KL to `mlp_std`'s own map for 500
  steps on the factory's default grid (64 points uniform in ln p over [0.0895, 20] GeV - the literal call of AUDIT 4.2;
  note it weights low momenta more than the training distribution does). Templates: `front` = `abar_front.pt`,
  `stored` = `abar.pt` (both SHA-256 against `transfer/SHA256SUMS`), `full_box` read by
  `dist_factory.acceptance_template` from the H200's Track A template directory after its SHA-256 and machine id were
  checked against `templates.json`. The factory does not store its warm-start KL in `init_info` (finding C3).
- **Stratified fold points (B2c).** `metric.draw(fold="stratified")`: K = 16 → strata (4, 4, 1), one uniform point per
  equal-volume sub-box; unbiased, variance never above i.i.d. (existing, reviewed code); training only - validation stays
  i.i.d. K = 64, seed 12345, the metric of record.
- **EMA rows (B3).** `_EMA`: shadow initialised at the starting weights, `lerp_(p, 1 − 0.999)` after every ACCEPTED
  optimiser step only (skipped batches do not update it), buffers from the live model; selected by the same "last" rule,
  saved as `model.pt["ema_state"]`, evaluated by the registry as its own row `<name>_ema` with `rung = "B3"`, compared
  with its own last iterate (`run_m2b_list.py:665-677`). Correct; no bias correction is needed (warm starts; for B3s
  the random initialisation's weight after 12,000 steps is 0.999^12000 ≈ 6e-6).
- **Smoothness penalty (B3s, B5f).** `smooth · mean((log P[:, 2:] − 2 log P[:, 1:-1] + log P[:, :-2])²)` along depth,
  added to the mean per-event NLL (`train.py:542-543`) - character for character the expression of `common.fit`
  (`common.py:356-357`). Correct.
- **`train_fraction` (B-ceil).** `torch.randperm(N, Generator().manual_seed(seed))`, first round(f N), sorted, then
  `Showers.subset`; index SHA-256 recorded in `init.json`, `done.json` and the registry meta. The fit's shuffle
  generator then runs on the subset. Nested subsets (25 % ⊂ 50 %). The quantile grid of `field_kl` / `field_distill`
  always uses the full training split (B-ceil uses no grid). Correct.
- **Numerical hygiene.** No `log(softmax)` anywhere in the new code: `FieldModel`, `FieldAcc`, `ResidualTable` use
  `log_softmax`, `Ensemble` uses `logsumexp`; the KL uses log-probabilities and `0 · finite = 0` where P underflows;
  `ln A` from a clamped A; the metric's log mass is clamped at 1e-8 (base code). Divisions: frozen-window `n = max(1,
  …)`, retention only for a negative B1 gain, G-region share only for a non-zero gain. Modes: the fit trains in
  `train()` and validates in `eval()` under `inference_mode()` inside `deterministic()`; registry scoring in `eval()`
  inside `deterministic()`; `kl_warm_start` puts the teacher in `eval()` and restores it; `g_region_compute` calls
  `eval()`. Seeds: `seed_everything(seed)` before every build, again inside `fit` on a fresh start; dedicated shuffle
  (1000 + seed) and fold (20000 + seed) generators; the subset generator is its own. Recipes: B2a/B2b/B2c/B5f of one
  seed share the shuffle order, as the pairing by seed intends.
- **Measured on the laptop GPU (under the GPU lock, real training events, batch 256, 20 steps each):** peak allocated
  memory 1,555 MiB (B2a, K = 16), 1,755 MiB (B2b, K = 64), 2,707 MiB (B6c, width 64), step time 1 : 1.26 : 1.53. All far
  below the H200's 10 GB cap, so the K = 64 and width-64 runs carry no memory risk.

## 4. Provenance and safety (question 4)

- **Starting points.** Every `start` / `target` / `run` reference must have a row on the live stream whose
  `checkpoint_sha256` is the file's SHA-256 and whose `weights_sha256` equals the stored state's (`verify_sources`,
  `run_m2b_list.py:393-424`, before the factory reads weights); `mlp_std.pt` and the front / stored templates against
  `transfer/SHA256SUMS`; `full_box` against the H200's `templates.json` with its machine id. Recomputed: on the H200
  records in the repository, the batch-1 rows of the members and of `E5_s2` are found and verify (section 5.3; the
  author's test `test_batch1_runs_recognised_and_sources_verified_on_the_h200_records` passes). For the decided starts
  `s2_seed11` / `s2_seed12`, `lib.member` gives `repo:m2b_mlp/s2_seed1x` (as for `E5_s2` on the H200, whose members
  were referenced the same way), and the H200 rows of both exist with the files' SHA-256 - the re-check should run that
  test with the new starts.
- **Recipes pinned.** `Spec.recipe_hash()` (all fields but notes) must equal `configs/m2b/list_recipes.json`
  (`check_pins`, in `verify` or at start-up); the x2 reruns have their own pins; names may not collide with batch-1
  names. `--pin` refuses off the laptop of record and in pristine mode. The pin file and the spec come from the same
  commit, so the pin protects against an edited working copy (which pristine mode already refuses) - fine as defence in
  depth. A1 changes three recipes; the pin file must be regenerated in the same commit.
- **Outputs.** Every write goes through `paths.check_output_path` into `machines/<id>/runs/<stage>/` (markers, models,
  decisions, G-region arrays, `LIST_RESULTS.md`) or the registry's tagged score directories; JSON is passed through
  `paths.scrub_str`. The fit's own `fit.log` contains absolute paths (base code, as in batch 1); it is not copied back.
- **Nothing registered is changed.** Run names are disjoint from `BATCH1_NAMES`; batch-1 rows and files are only read;
  guard rows are appended only for runs of the list; tagged arrays are never overwritten (`save_tagged` keeps a
  bit-identical array, else raises).
- **`fit.py` / `fit_queue.py`** now refuse off the laptop of record right after `machine.startup` (identity from the
  hardware fields in `sfm.policy`, no environment override). `run_m2b_fit.py` additionally refuses a config outside
  `configs/` or with an unpinned hash for the eight batch-1 names.
- **Escape hatches.** None added. `SFM_M2B_INJECT_DIVERGENCE` is refused in the real mode; the smoke mode needs
  `SFM_M2B_SMOKE=1` and writes under its own stage - but it is not refused off the laptop (finding C7).

## 5. The report (question 5)

### 5.1 Rule by rule

| rule | implementation (`run_m2b_list_report.py`) | verdict |
|---|---|---|
| (a) Δ vs `mlp_std` ≤ −5 | `paired_delta(a, refset mlp)`, `mean <= -5e-4` | correct |
| (b) ≤ −3 SE | `z <= -3` | correct |
| (c) better than B0 by > 1 SE | per-event mean of exactly `b0h_seed0..2` (stage `m2b_mlp`, live stream, `check_pairable`), `z < -1`; "B0 incomplete" otherwise | correct as read (I12 is a judgement; a plain control passes it, section 5.3) |
| (d) replication | seed 1 Δ ≤ −2.5, same sign, two-seed per-event mean meets (a) **and** (b); evaluated on the seed-0 row, the seed-1 row shows "see …" | computes what it says; the reading differs from Track A's (finding B4, judgement I13) |
| (e) guards | G-K / G-bulk: `margin >= 0 and resolved`; G-region: flagged hit-term difference − ½ main ≥ 0 on the point estimate | **G-K / G-bulk inverted - finding B1**; G-region sign correct (share ≤ ½ ⟺ flagged − ½ main ≥ 0 when main < 0), no resolution flag (J-b) |
| promoted | (a) ∧ (b) ∧ (c) ∧ (d) ∧ (e) for EVERY run, including comparators, exploratory runs and B-ceil diagnostics | finding B3 |
| primary comparisons | B1d vs `mlp_std`; B2b − B2a, B2c − B2a, B4f − B2a, B6c − B6a same seed; thresholds and replication on the comparison | computes what it says (I11 judgement); **original runs only, never the x2 rerun pair - finding B2** |
| convergence reruns | listed in a "Convergence" section (move, rerun names); their arrays are never paired, never enter (a)-(e), the primary comparisons, replication or the 1-SE rule; `guard_candidates` does not consider them | **finding B2** (the Track A report was required to use the rerun pair - its review B1) |
| 1-SE (B6) | per seed, `registry.select_1se` over the rows of B6a, B6b, B6c (k = 3,841 / 4,897 / 13,825) | correct as read (I17 judgement); only where registered |
| B3 | `paired_delta(<name>_ema, <name>)` | correct |
| B-ceil | registered last iterate; trajectory minimum from the fit history, printed as mean − `mlp_std` mean (unpaired, labelled) | correct |
| secondary columns | p < 2 / ≥ 2 GeV, contained / side-leaking (same definition as Track A), masks through `paired_delta(mask=…)` | correct; never used by a rule |
| separate tables | report metric (K = 64, all events) in the main and primary tables; G-K, G-bulk, the report metric restricted to the stratum (own column, needed for the persistence margin), G-region and G-mass in the guard table; never one column or one delta | correct |
| B1 retention | `B1d Δ / E5 Δ` whenever E5's Δ < 0 | finding C5 (on the H200 E5 is −0.83 ± 1.35: a ratio with an unresolved denominator) |
| nulls | every row printed as measured; diverged / blocked runs listed with status and diagnosis | correct |

### 5.2 Finding B1 reproduced (G-K / G-bulk pass test inverted)

`sfm.stats.guard_persistence(main, guard)` returns `margin = mean(guard − 0.5 main)` and passes the "fraction" check iff
`margin <= 0` (with `main < 0` and `guard < 0`). The report's `guards()` (lines 136-137) instead calls a guard passed iff
`margin >= 0` and resolved, and ignores the decision's `passed` and its sign checks. Reproducer
(`scratchpad/m2b_review/repro_guard_sign.py`, synthetic arrays, CPU, the report's own function with a registry row built
as `registry._guards` builds it):

```
P: guard keeps 100 % of the gain: stats.guard_persistence passed=True resolved=True margin=-5.12e-4 (z -41.1) | report gk_ok=False bulk_ok=False passed=False
F: guard keeps 0 % of the gain:   stats.guard_persistence passed=False resolved=True margin=+5.07e-4 (z +40.6) | report gk_ok=True bulk_ok=True passed=True
```

A gain that survives the guard fails it, and a pure metric artefact passes. Fix: take `g[...]["passed"]` (and
`flags["resolved"]` as its own column, as the Track A report does; how an unresolved pass counts is J-a). Not caught by
the tests: `tests/test_run_m2b_list.py` has no test of `guards()`.

### 5.3 Independent recomputation (numpy) on the real H200 arrays

A smoke run end to end could not be repeated (section 8). Instead the report's functions were called on the REAL batch-1
arrays of the H200 that are in the repository (validation only; H200 identity patched in exactly as the author's test
does; read-only) and compared with plain-numpy recomputations from the `.npy` files named in the latest registry rows
(`scratchpad/m2b_review/recompute_h200.py`):

| quantity | report function | numpy |
|---|---|---|
| `mlp_std` reference array (refset) | mean 18.118082943664348 | identical array |
| B0 per-event mean of `b0h_seed0..2` | `b0_mean()` | bit-identical |
| `E5_s2` vs `mlp_std` | −0.8258 ± 1.3548; (a) no, (b) no | −0.8258 ± 1.3548 |
| `E5_s2` vs B0 | −1.6155, z −1.140 → (c) yes | −1.6155, z −1.140 |
| `s2_seed14` vs `mlp_std` / vs B0 | +0.5416 ± 1.6449 / −0.2481, z −0.161 | identical |
| **`b0h_seed2` vs B0 (a control)** | **−0.4761, z −1.209 → (c) yes** | identical |
| two-seed mean (`s2_seed14`, `s2_seed13`) vs `mlp_std` | +1.1350 ± 1.3963 | +1.1350 ± 1.3963 |
| replication record on that pair | seed-1 half threshold no, same sign yes, replicated no | as defined |
| 1-SE over (`s2_seed14`, `s2_seed13`, `E5_s2`) | selected `E5_s2`, best `E5_s2` | best `E5_s2`, eligible {`E5_s2`}, selected `E5_s2` |
| member ranking | `s2_seed14, s2_seed13, s2_seed11, s2_seed12, s2_seed15` | same; best − second −1.19 ± 1.28; second − third −0.09 ± 1.51 |

All numbers agree to the printed digits. The B0 line shows the point of I12: against the per-event B0 mean, whose
paired SE (≈ 0.39) is as large as the seed scatter of the controls (sd 0.41), one of the three controls is itself
"better than B0 by more than 1 SE".

## 6. Robustness of an unattended run (question 6)

By code reading (the author's smoke log shows two kills and a mid-fit resume; I could not repeat it, section 8):

| kill point | what the next start does | double registration? |
|---|---|---|
| during B1d's KL fit (data-free, no checkpoint) | rebuilds from the start, KL again; nothing was written before the build finished | no |
| after `init.json`, before the first checkpoint | rebuilds (KL / warm start again), keeps the first `init.json`, writes `init_rebuild<k>.json`; on CUDA the rebuilt start may differ in the last bits (non-deterministic training mode), and that is recorded | no |
| mid-fit, after a checkpoint | factory with `skip_init` (no KL / warm start), `fit` resumes from `checkpoint.pt` after checking its config hash and machine tag | no |
| after `fit_done.json`, before registration | `fit` returns the finished run (config, machine, model SHA-256 checked); the divergence check is repeated on the stored history; `row(name, ckpt)` is empty → registered once | no |
| after the main row, before the EMA row | main row found by checkpoint SHA-256; EMA row registered | no |
| after both rows, before `done.json` | both found by checkpoint SHA-256; `done.json` written | no |
| in the guards phase | the guard row is found with `need_guards`; G-region arrays are kept if bit-identical, the marker written last | no (the guard row is a deliberate second row, as in Track A) |
| in `converge` | reruns go through the same `run_spec` | no |

- **Divergence handling** matches PLAN 17.11 / 20 item 2: exploratory → "diverged", job continues; primary,
  comparator, diagnostic and rerun → "diverged - awaiting diagnosis", job continues; "failed" is never written by code,
  a diagnosis counts only with the SHA-256 of the run's `diverged.json`; runs started from a diverged run are blocked
  (only B1f). A recovered mid-run non-finite value is a warning counted in `done.json`. The conflict with the new
  hard-stop rule is section 2.2 / finding A2.
- **A failure in one phase** that is not a divergence (an exception, `SystemExit`) ends the job; later phases do not
  run; the same command resumes. A diverged comparator leaves its comparisons "not available"; a diverged primary has
  no convergence entry; the report still runs.
- **Convergence reruns** are triggered and pinned correctly (I15); they are not used by the report (B2).
- **`run_m2a.g_region_compute` for an MLP field: correct.** It decomposes the metric itself, nll_e = −Σ_{i∈e} w_i ln g_i
  + W_e ln M_e, from `metric.test_nll(return_parts=True)` on the registry's points, and sums the hit term by
  hit-centre class (deep z′ ≥ 1221 mm, last ring, first / last z-bin, Ā ≤ 0.3 of the stored template). Nothing in it is
  model-specific; it works for `FieldNet`, `FieldAcc`, `ResidualTable` and the B1d student. The masks cached by
  `_g_region_static` (commit `dde1764`) depend only on the validation hits and the stored template; the key is the id of
  the CPU validation `Showers`, which the driver keeps for the whole process (`_DATA["tv"]`), so they are computed once
  and are valid for every model, `mlp_std`'s reference included. Two caveats: the Track A identity check (parts against
  the registered array) is not repeated here (C6); and the import executes `run_m2a.py` / `run_m2a_lib.py` of the pulled
  commit - the Track A worker has uncommitted changes to both, so the re-check must confirm that `g_region_compute`,
  `_g_region_static` and `_g_region_masks` are unchanged at the commit the H200 pulls.
- **Expected wall time** is about an hour, not 25 minutes (C1): every T_B fit validates at K = 64 every 250 steps,
  main and EMA stream (48 passes per 6000-step fit, 96 for B3s, 60 per B-ceil fit: ≈ 1,580 passes at the measured
  1.6 s of a registry evaluation ≈ 40 min) on top of ≈ 20 min of training steps; convergence reruns come on top.

## 7. Verdict, findings, step list (question 7)

**Verdict: NO-GO for `run_m2b_list.py` on the H200 at `43ccfd6`.** The committed driver chooses B1d's two starts, and
through `B1d_best` B1f's start, by validation rank, which the user has excluded (A1); run names and pinned recipes would
be spent on the rejected starts and A6 revised point 5 forbids repeating them. Everything else that the job trains and
registers is, as far as this review can see, correct and safe; the report's defects (B) do not touch what the job
trains and can be fixed afterwards, because the report re-reads registered arrays.

**GO after:** one commit that does exactly A1 (and A2 as decided), with the pin file regenerated, and a narrow
re-check of that diff (section 7.2, step 1).

### 7.1 Findings

(A) must be fixed before the run
- **A1 (fixed by user decision, to be verified in the re-check).** B1d starts by validation rank (`run_m2b_list.py:350-362,
  372-377`) and B1f from the rank-named `B1d_best` (`:164-171, 378-379`). Required: the fixed, validation-blind starts
  `s2_seed11`, `s2_seed12`; B1f from the B1d run of the first fixed start; names and the three pinned recipes changed
  with it; no other code path ranking members in the real mode (section 2.1: none found besides V1-V3).
- **A2 (decision + small change).** Conformance with the standing hard-stop rule (section 2.2): (a) a non-finite
  per-event validation NLL at registration and a non-finite KL in a factory are recorded as divergences and the job
  continues (PLAN 20 item 2) - the orchestrator must say whether that stays or becomes a hard stop; (b) a clone that is
  no longer clean at the end of the job only prints a WARNING and exits 0 (`run_m2b_lib.finish`, base code) - under the
  rule it should exit non-zero; (c) guard refusals swallowed by third-party imports exit 0 - keep as an operator check
  of the guard log (as accepted in batch 1) or decide otherwise.

(B) must be fixed before the table is read (report only; re-run the report on the H200 as its own job afterwards)
- **B1.** G-K / G-bulk pass test inverted (`run_m2b_list_report.py:136-137`, `margin >= 0`): artefacts pass, real gains
  fail (section 5.2). Use the decision's `passed`; show `resolved` separately; rule J-a.
- **B2.** Convergence reruns ignored: when `<name>_x2` / `<comp>_x2` exist they must be the pair of record for (a)-(e),
  the primary comparisons, replication and the 1-SE rule, with the original printed as a labelled row no rule uses (the
  Track A report's review B1); `guard_candidates` must consider the x2 runs (then a `guards report` job on the H200).
- **B3.** "promoted" is computed and printed for comparators, exploratory runs and B-ceil diagnostics; AUDIT 4.0 says
  everything outside the six primaries is "exploratory and is reported as such" - print "meets the rule (exploratory)"
  or leave the column empty for non-primaries.
- **B4.** The replication rule's second clause (two-seed mean meets (a) **and** (b)) differs from the Track A report's
  reading (Δ only, SE as a labelled alternative). Needs the ruling on I13, then the same code in both reports.
- Before the table is read, the judgement items must also be ruled: I11, I12, I13, I17, J-a, J-b.

(C) not blocking
- **C1.** Run-time estimate ≈ 2× low (section 6); tell the operator to expect about an hour plus reruns.
- **C2.** B-ceil carries EMA 0.999 from T_B: 30 extra validation passes per fit, weights unaffected (changing it now
  changes eight pinned recipes; leaving it is fine).
- **C3.** `field_acc` does not store its warm-start KL in `init_info` (B4's `init.json` has `init_info: null`); the
  gate cannot see whether B4 starts from `mlp_std`'s function (Track A flags warm starts with KL > 1e-4).
- **C4.** The EMA-row registration (`run_m2b_list.py:675`) is not protected against `FloatingPointError`; a non-finite
  EMA score would stop the job and block resumption at that run (practically unreachable: the EMA averages accepted
  finite iterates).
- **C5.** B1 retention fraction printed with an unresolved denominator (E5 −0.83 ± 1.35 on the H200); print only with
  its SE or when |z(E5)| ≥ 2.
- **C6.** G-region: no identity check of the parts against the registered array, no resolution flag, normalisation
  term printed only in the JSON - the Track A implementation has all three.
- **C7.** The smoke mode is not refused off the laptop of record (it would put `m2b_smoke` rows into the machine of
  record's registry).
- **C8.** No post-build weight check for the `field_mlp_std` starts (`run_m2b_fit`'s `check_built_weights`); the file
  hash is checked, the factory reads the same path - low risk.
- **C9.** Tests: nothing covers the report's guard pass test, its x2 handling or (c) on real arrays; B1 would have
  been caught by a two-line test. STATUS's "31, plus 8 + 2 `_ema` rows" should read 31 + 8 EMA rows (6 of B2, 2 of B3s).

### 7.2 Step list I endorse (after A1/A2 are committed and re-checked)

1. Laptop, re-check: `git diff 43ccfd6 <fix commit> --stat` shows only `run_m2b_list.py`, `configs/m2b/list_recipes.json`,
   the tests (and whatever A2 needs); `git diff 43ccfd6 <fix commit> -- scripts/run_m2a.py scripts/run_m2a_lib.py
   src/sfm/models/dist_factory.py` shows no change to `g_region_compute`, `_g_region_static`, `_g_region_masks`,
   `template_dir`, `acceptance_template`; `uv run pytest shape-form-martin/tests/test_run_m2b_list.py` passes, with the
   H200-records test verifying `s2_seed11` / `s2_seed12` and `E5_s2` by SHA-256; `run_m2b_list.py --list` prints the B1d
   starts as `s2_seed11`, `s2_seed12`.
2. H200: pull only; `git status --porcelain` empty; `git rev-parse HEAD` = `origin/shape-form-martin`; `nvidia-smi` shows
   ≥ 14 GB free.
3. `$L m2b_list_ls $P -B shape-form-martin/scripts/run_m2b_list.py --list` → read: the 13 given runs "registered",
   31 runs "pending", the B1d starts the fixed seeds.
4. `$L m2b_list $P -B shape-form-martin/scripts/run_m2b_list.py` (one detached job; stop only with `stop_job.sh`; resume =
   the same command). Read in the log:
   - first lines: `outputs under <out>/…`, `file-access guard ACTIVE`, the git commit of step 1, clean;
   - `verify: reference set {… 'parity': 'PASS'}; batch-1 runs present: 13/13; templates ['front', 'full_box',
     'stored']; recipes pinned`;
   - one `registered:` line per run (31) plus `EMA` on the 8 runs with EMA rows; the `field_distill … KL a -> b` lines
     of the two B1d runs (b ≪ a) and `field_kl` / `field_acc warm start` lines (read the final KL);
   - any `DIVERGED` line: read its `diverged.json` (exploratory: continues; otherwise "awaiting diagnosis");
   - `=== phase converge`: the move of every primary; `_x2` reruns only where a primary moved ≥ 0.5;
   - `=== phase guards`: either nothing (no run met (a)-(c)) or one guard row and G-region arrays per candidate;
   - `=== finished`, `.exit` = 0, no "clone is no longer clean" WARNING, the guard log only the tolerated `pyarrow` lines.
   - The `LIST_RESULTS.md` this job writes is **not** the table of record while B1-B4 are open.
5. Copy back as STATUS section 5 (registry, markers, `init*.json`, `diverged.json`, tag sidecars, `g_region` markers and
   arrays, decisions; model files and per-event arrays by hand).
6. After B1-B4 are fixed and reviewed and the judgement items are ruled: on the H200, `$L m2b_list_rep $P -B
   shape-form-martin/scripts/run_m2b_list.py guards report` (adds guard rows for x2 candidates if any; rewrites the
   report); that table goes to the gate.

## 8. What could not be verified, and integrity of the campaign files

- **No smoke run end to end.** Creating a scratch reference set needs the formula benchmark, whose builder reads a file
  of the earlier study (`shape-form-proton/results/particles/pion/test.json`, its stored configuration); copying it into
  the scratch export was refused by the permission system, and I did not work around that. Without a reference set no
  registry evaluation is possible, so the fits, the resume points, the guards phase and the report were not executed
  on the laptop; they were reviewed by reading, and the report's arithmetic was recomputed on the real H200 batch-1
  arrays instead (section 5.3). The author's smoke evidence (STATUS section 6) is therefore unconfirmed by me.
- Not verified: the H200 run itself, its step costs and convergence; the warm-start quality of B4; the A1 fix (moving
  files, not reviewed).
- Executed: `run_m2b_list.py --list` from the export; `tests/test_run_m2b_list.py` from the export (9 passed - this
  suite touches CUDA for about two seconds and ran without the GPU lock, a slip on my side; nothing else used the GPU
  outside `sfm.gpulock.gpu_lock`); the guard-sign reproducer (CPU); the read-only recomputation on the H200 arrays; the
  memory probe (under the GPU lock, about one minute). Scratch outputs only under the session's scratchpad.
- Campaign files, SHA-256 before and after this review (identical): `results/registry.jsonl` `017edeba030c…81de`,
  `results/test_access.log` `effa9734743f…0c60`, `common.py` `ece3c67f89cd…bb3`. No test array, test cache or CSV file
  was opened; nothing under `shape-form-proton/` or `shape-form-original/` was modified; no git command that writes was
  used.
