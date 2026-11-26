# Track A (M2A) - results on the machine of record

Stage `m2a_dist`, machine `h200-g1056-97b197a7`, stream `g1056-s12345-b128-iid-val731a1997@h200-g1056-97b197a7`, report code at commit `732039b80157ed9e8d788e660ffe42f1d8aa5dbb` (clean clone: True). Reference set `<out>/shape-form-martin/results/machines/h200-g1056-97b197a7/refsets/g1056-s12345-b128-iid-val731a1997` (refset.json SHA-256 `3096aec62a128199...`). Validation only. Every delta: mean ± paired SE (n events), unit 1e-4 nat per event, through `sfm.provenance.paired_delta`. Report-metric tables (K = 64) and guard tables are separate. Nulls and negative results as measured.

## Harness

- A0 at step 0 vs this machine's formula array: +9.888e-09 ± 1.1e-08 nat (n 14950), max |event| 1.14e-05 -> **PASS** (criterion: 1e-4 per event, 1e-6 on the mean)
- A0 drift rule (stop if |A0 - formula| > 2): A0_s0 +0.64 ± 0.74 (n 14950), A0_s1 +0.65 ± 0.74 (n 14950) -> continue
- P-lr: lr_poly = 0.003, lr_net = 0.001; poly lr 0.003: +0.39 ± 0.32 (n 14950) vs lowest; poly lr 0.01: +0.00 ± 0.00 (n 14950) vs lowest; poly lr 0.03: +0.10 ± 0.86 (n 14950) vs lowest; net lr 0.001: +0.00 ± 0.00 (n 14950) vs lowest; net lr 0.003: +5.05 ± 0.96 (n 14950) vs lowest; net lr 0.01: +7.24 ± 1.97 (n 14950) vs lowest
- Dropped runs: [] (not applied: the lists of AUDIT 4.1 run in full on the machine of record (PLAN 13, A6 revised point 6; PLAN 16.5 recommendation R1). No run was dropped.)

## Stage 0 (report metric)

| comparison | Δ |
|---|---|
| A0_s0 - formula | +0.64 ± 0.74 (n 14950) |
| A0_s1 - formula | +0.65 ± 0.74 (n 14950) |
| A1_s0 - A0_s0 | +2.93 ± 1.23 (n 14950) |
| A1_s1 - A0_s1 | +2.56 ± 1.29 (n 14950) |
| A2e_s0 - A0_s0 | +0.72 ± 0.12 (n 14950) |
| A2e_s1 - A0_s1 | +0.62 ± 0.11 (n 14950) |
| A2s_s0 - A2e_s0 | +1.13 ± 0.28 (n 14950) |
| A2s_s1 - A2e_s1 | +1.12 ± 0.28 (n 14950) |
| A2n_s0 - A2s_s0 | +2.11 ± 1.14 (n 14950) |
| A2n_s1 - A2s_s1 | +2.07 ± 1.21 (n 14950) |
| A2n_s2 - A2s_s2 | +2.16 ± 1.20 (n 14950) |
| A2s_s1 - A2s_s0 | -0.10 ± 0.21 (n 14950) |
| A2s_s2 - A2s_s0 | +0.12 ± 0.31 (n 14950) |
| A2s_s2 - A2s_s1 | +0.22 ± 0.23 (n 14950) |

A2s seed scatter (sd of the validation means of the seeds): 0.11e-4

A9 at step 0 (logit_dh = -5): A9_s0_init - A2s_s0 -0.58 ± 0.60 (n 14950); A9_s1_init - A2s_s1 -0.64 ± 0.60 (n 14950)

## A2c - control (added by the user on 25 Nov 2026, exploratory, outside the six primaries)

A2s continued unchanged from the fitted A2s of the same seed, with the schedule of the rungs that start from it (T_A, 4000 steps, the frozen lr_poly, the same seed). No pre-registered rule consumes it: keep, replication and equal NLL stay against A2s as registered. The column 'control: Δ vs A2c' below decides nothing.

- A2c_s0 - A2s_s0: -0.16 ± 0.12 (n 14950)
- A2c_s1 - A2s_s1: -0.13 ± 0.12 (n 14950)
- Matched step-budget groups: **matched (A2c)**: component rungs that START FROM the fitted A2s of the same seed (copy by name, or init_shower_long on it): A4, A5, A7, A8, A9, A4t, A5h, A6i, A6l, A6b, A7s, A7l, A8c, A78, A9r, A3f, A3c, A3x, A12; T_A with 4000 steps, the frozen lr_poly, train seed = the rung's seed, fresh optimiser and cosine cycle; **not matched: mixtures**: A10, A10-2, A10-4, A10-5, A10g: 6000 steps and initialised by init_from_target on the MAP of A2s (function space, a new structure), not continued from its weights. Proposal (awaiting the user): 'A2c6k' = A2c with 6000 steps, seeds 0 and 1; **not matched: convergence reruns**: <rung>_s<seed>_x2: 8000 / 12000 steps from the original A2s; their comparator A2s_s<seed>_x2 is itself a rerun. Proposal (awaiting the user): 'A2c_x2' = A2c with doubled steps, only for seeds with a rerun; **not applicable**: network runs (fitted to their own polynomial run) and Stage 0

## Stage 1 - primaries, poly (pre-registered comparisons, vs A2s of the same seed; report metric)

Multiplicity: these six rungs are the pre-registered primary comparisons of Track A; every other row of this report is exploratory and is not a test.

| rung | k | pairs used by the rules (s0 / s1) | Δ seed 0 | keep s0 | Δ seed 1 | keep s1 | replicated (two-seed mean) | replicated under the alternative reading (literal conditions AND two-seed mean <= -2 SE; alternative - decides nothing) | equal NLL (both) | Δ vs formula s0 | converged s0/s1 | control: Δ vs A2c s0 / s1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A4 | 93 | original / rerun | +1.77 ± 0.57 (n 14950) | no | +1.44 ± 0.58 (n 14950) | no | no (+1.61 ± 0.57 (n 14950)) | no | no | +4.26 ± 0.97 (n 14950) | yes/yes | +1.93 ± 0.57 (n 14950) / not matched (doubled-step rerun) |
| A4 s1 original - not converged, NOT used by the rules | 93 | A4_s1 vs A2s_s1 |  |  | +1.76 ± 0.57 (n 14950) | no | | | | | no | - |
| A5 | 101 | original / rerun | +6.32 ± 1.24 (n 14950) | no | +5.93 ± 1.18 (n 14950) | no | no (+6.13 ± 1.21 (n 14950)) | no | no | +8.81 ± 1.50 (n 14950) | yes/yes | +6.48 ± 1.23 (n 14950) / not matched (doubled-step rerun) |
| A5 s1 original - not converged, NOT used by the rules | 101 | A5_s1 vs A2s_s1 |  |  | +6.26 ± 1.25 (n 14950) | no | | | | | no | - |
| A7 | 103 | original / original | -0.51 ± 0.27 (n 14950) | no | -0.46 ± 0.28 (n 14950) | no | no (-0.48 ± 0.27 (n 14950)) | no | yes | +1.98 ± 0.88 (n 14950) | yes/yes | -0.35 ± 0.25 (n 14950) / -0.32 ± 0.26 (n 14950) |
| A8 | 102 | original / original | -0.06 ± 0.29 (n 14950) | no | -0.04 ± 0.29 (n 14950) | no | no (-0.05 ± 0.29 (n 14950)) | no | yes | +2.42 ± 0.84 (n 14950) | yes/yes | +0.10 ± 0.28 (n 14950) / +0.09 ± 0.28 (n 14950) |
| A9 | 104 | original / original | -4.67 ± 0.88 (n 14950) | yes | -4.66 ± 0.89 (n 14950) | yes | yes (-4.66 ± 0.89 (n 14950)) | yes | yes | -2.18 ± 1.26 (n 14950) | yes/yes | -4.51 ± 0.88 (n 14950) / -4.52 ± 0.89 (n 14950) |
| A10 | 74 | rerun / original | +78.79 ± 4.79 (n 14950) | no | +87.49 ± 4.99 (n 14950) | no | no (+83.14 ± 4.85 (n 14950)) | no | no | +81.48 ± 4.92 (n 14950) | no/yes | not matched (6000 steps, init_from_target) / not matched (6000 steps, init_from_target) |
| A10 s0 original - not converged, NOT used by the rules | 74 | A10_s0 vs A2s_s0 | +87.31 ± 4.99 (n 14950) | no |  |  | | | | | no | - |

Rules use the rerun pair (`<rung>_s<seed>_x2` vs `A2s_s<seed>_x2`) of a seed whenever it exists (AUDIT 4.0, convergence rule); seeds may then mix an original and a rerun, each paired with its own comparator. Kept = keep rule on seed 0 and replication by seed 1 (rule applied in seed order): a rung that passes keep on seed 1 only is not kept; the table shows both keep flags. A primary run that diverged is shown as 'diverged - awaiting diagnosis'; no rule marks it failed before an independent diagnosis (PLAN 17.11).
Reading aid (review B6; decides nothing): every rung that starts at the fitted A2s also carries a common 'continued training' term, measured by the A2c column; convergence movement of the base over its last window: A2s_s0 0.12e-4, A2s_s1 0.41e-4. Differences between rungs that share the start (paired) are free of that term.


## Stage 1 - net column (exploratory; vs A2n seed 0, or its warm-start re-run)

| run | k | Δ | keep | equal NLL (seed 0) | converged |
|---|---|---|---|---|---|
| A4n_s0 vs A2n_s0 | 665 | +0.55 ± 0.98 (n 14950) | no | yes | no |
| A5n_s0 vs A2n_s0 | 699 | +3.10 ± 1.13 (n 14950) | no | no | no |
| A7n_s0 vs A2n_s0 | 701 | +0.12 ± 0.35 (n 14950) | no | yes | no |
| A8n_s0 vs A2n_s0 | 700 | -0.34 ± 0.29 (n 14950) | no | yes | no |
| A9n_s0 vs A2n_s0 | 718 | -5.15 ± 1.00 (n 14950) | yes | yes | no |
| A10n_s0 vs A2n_s0 | 599 | +52.76 ± 4.16 (n 14950) | no | no | no |

## Stage 1 - exploratory (vs A2s of the same seed)

| rung | k | Δ seed 0 | keep s0 | equal s0 | Δ seed 1 | replicated | replicated under the alternative reading (literal conditions AND two-seed mean <= -2 SE; alternative - decides nothing) | equal NLL (both) | cleaner (gain / constants) | control: Δ vs A2c s0 / s1 |
|---|---|---|---|---|---|---|---|---|---|---|
| A4t | 89 | +19.54 ± 4.13 (n 14950) | no | no | - | - | - | - | no / no | +19.70 ± 4.12 (n 14950) / - |
| A5h | 97 | +5.04 ± 1.87 (n 14950) | no | no | - | - | - | - | no / no | +5.20 ± 1.86 (n 14950) / - |
| A6i | 97 | +6.42 ± 1.31 (n 14950) | no | no | - | - | - | - | no / no | +6.58 ± 1.30 (n 14950) / - |
| A6l | 97 | +9.66 ± 1.49 (n 14950) | no | no | - | - | - | - | no / no | +9.82 ± 1.47 (n 14950) / - |
| A6b | 98 | +1.07 ± 2.21 (n 14950) | no | no | - | - | - | - | no / no | +1.23 ± 2.21 (n 14950) / - |
| A7s | 102 | -0.52 ± 0.27 (n 14950) | no | yes | -0.47 ± 0.27 (n 14950) | no | no | yes | no / no | -0.36 ± 0.25 (n 14950) / -0.34 ± 0.25 (n 14950) |
| A7l | 102 | -0.20 ± 0.14 (n 14950) | no | yes | -0.18 ± 0.14 (n 14950) | no | no | yes | no / no | -0.04 ± 0.06 (n 14950) / -0.04 ± 0.06 (n 14950) |
| A8c | 102 | -0.23 ± 0.14 (n 14950) | no | yes | -0.20 ± 0.14 (n 14950) | no | no | yes | no / no | -0.07 ± 0.07 (n 14950) / -0.07 ± 0.06 (n 14950) |
| A78 | 104 | -0.50 ± 0.47 (n 14950) | no | yes | -0.47 ± 0.47 (n 14950) | no | no | yes | no / no | -0.34 ± 0.47 (n 14950) / -0.33 ± 0.48 (n 14950) |
| A9r | 98 | -4.63 ± 1.24 (n 14950) | yes | yes | -4.64 ± 1.24 (n 14950) | yes | yes | yes | no / no | -4.47 ± 1.25 (n 14950) / -4.51 ± 1.25 (n 14950) |
| A3f | 101 | +8.04 ± 1.04 (n 14950) | no | no | - | - | - | - | no / no | +8.20 ± 1.04 (n 14950) / - |
| A3c (template notes Q1, Q2) | 101 | +6.23 ± 0.95 (n 14950) | no | no | - | - | - | - | no / no | +6.39 ± 0.95 (n 14950) / - |
| A3x (template note Q1) | 101 | +0.37 ± 0.47 (n 14950) | no | yes | +0.39 ± 0.47 (n 14950) | no | no | yes | no / no | +0.53 ± 0.46 (n 14950) / +0.53 ± 0.46 (n 14950) |
| A10-2 | 48 | +115.67 ± 5.26 (n 14950) | no | no | - | - | - | - | no / no | not matched (6000 steps, init_from_target) / - |
| A10-4 | 100 | +44.49 ± 4.02 (n 14950) | no | no | - | - | - | - | no / no | not matched (6000 steps, init_from_target) / - |
| A10-5 | 126 | +20.27 ± 3.06 (n 14950) | no | no | - | - | - | - | no / no | not matched (6000 steps, init_from_target) / - |
| A10g | 98 | +32.45 ± 3.93 (n 14950) | no | no | - | - | - | - | no / no | not matched (6000 steps, init_from_target) / - |
| A12 | 102 | -1.15 ± 0.82 (n 14950) | no | yes | -1.13 ± 0.83 (n 14950) | no | no | yes | no / no | -0.99 ± 0.81 (n 14950) / -0.99 ± 0.81 (n 14950) |

**Template limitations (A3c / A3x; review rulings Q1 and Q2, runs as registered):**

- Q1 (A3c, A3x): numerator by the 2 x 2-point rule of `c_side.py`; in the z-bin of the front face it overestimates the innermost rings (0.50 against 0.35); effect on the validation NLL at the legacy coefficients -0.1 to -0.2 (review: stored template with the 2-point bin 14 -0.10 ± 0.01; A3c as coded - A3c with the stored numerator -0.16 ± 0.02; A3x as coded - A3x with 4 sub-points -0.14 ± 0.02; laptop diagnostic, not numbers of record).
  This machine: computed front+back vs stored abar max 0.146, mean 7.6e-04.
- Q2 (A3c): the back_corrected template exceeds 1 by more than 1 % in 79 cells at z' >= 1379 mm (maximum 1.81; smallest denominator there 0.186): it adds density behind the flat cut, where the metric's normaliser does not count mass. A gain of A3c is to be read with the G-region class `deep` and the `beyond_zb` mass monitor. (cells above 1: 906, above 1.1: 56.)

## Secondary columns (pre-registered, never selection criteria; report metric on strata)

| run vs comparator | p<2GeV (n 1462) | p>=2GeV (n 13488) | contained (n 11096) | side-leaking (n 3854) |
|---|---|---|---|---|
| A4_s0 vs A2s_s0 | +3.94 ± 2.54 (n 1462) | +1.54 ± 0.56 (n 13488) | +1.34 ± 0.67 (n 11096) | +3.02 ± 1.07 (n 3854) |
| A4_s1_x2 vs A2s_s1_x2 | +1.79 ± 2.64 (n 1462) | +1.40 ± 0.57 (n 13488) | +0.63 ± 0.68 (n 11096) | +3.76 ± 1.09 (n 3854) |
| A4_s1 vs A2s_s1 (original, not used by the rules) | +4.28 ± 2.59 (n 1462) | +1.49 ± 0.57 (n 13488) | +1.12 ± 0.67 (n 11096) | +3.60 ± 1.09 (n 3854) |
| A5_s0 vs A2s_s0 | +4.27 ± 3.59 (n 1462) | +6.55 ± 1.32 (n 13488) | +5.87 ± 1.41 (n 11096) | +7.64 ± 2.59 (n 3854) |
| A5_s1_x2 vs A2s_s1_x2 | +5.10 ± 3.65 (n 1462) | +6.02 ± 1.25 (n 13488) | +4.33 ± 1.32 (n 11096) | +10.56 ± 2.54 (n 3854) |
| A5_s1 vs A2s_s1 (original, not used by the rules) | +4.44 ± 3.56 (n 1462) | +6.46 ± 1.33 (n 13488) | +5.56 ± 1.42 (n 11096) | +8.28 ± 2.61 (n 3854) |
| A7_s0 vs A2s_s0 | -1.48 ± 1.44 (n 1462) | -0.40 ± 0.25 (n 13488) | -0.35 ± 0.31 (n 11096) | -0.95 ± 0.51 (n 3854) |
| A7_s1 vs A2s_s1 | -1.19 ± 1.46 (n 1462) | -0.38 ± 0.26 (n 13488) | -0.31 ± 0.33 (n 11096) | -0.90 ± 0.53 (n 3854) |
| A8_s0 vs A2s_s0 | -1.36 ± 1.00 (n 1462) | +0.08 ± 0.30 (n 13488) | +0.21 ± 0.36 (n 11096) | -0.83 ± 0.45 (n 3854) |
| A8_s1 vs A2s_s1 | -1.22 ± 0.98 (n 1462) | +0.08 ± 0.30 (n 13488) | +0.20 ± 0.35 (n 11096) | -0.76 ± 0.45 (n 3854) |
| A9_s0 vs A2s_s0 | -2.55 ± 3.69 (n 1462) | -4.90 ± 0.89 (n 13488) | -3.41 ± 0.89 (n 11096) | -8.27 ± 2.28 (n 3854) |
| A9_s1 vs A2s_s1 | -2.47 ± 3.72 (n 1462) | -4.89 ± 0.90 (n 13488) | -3.38 ± 0.90 (n 11096) | -8.33 ± 2.30 (n 3854) |
| A10_s0_x2 vs A2s_s0_x2 | +345.37 ± 38.73 (n 1462) | +49.90 ± 3.15 (n 13488) | +78.68 ± 5.69 (n 11096) | +79.11 ± 8.74 (n 3854) |
| A10_s1 vs A2s_s1 | +385.39 ± 40.07 (n 1462) | +55.21 ± 3.32 (n 13488) | +86.70 ± 5.86 (n 11096) | +89.77 ± 9.53 (n 3854) |
| A10_s0 vs A2s_s0 (original, not used by the rules) | +377.32 ± 39.86 (n 1462) | +55.88 ± 3.35 (n 13488) | +83.36 ± 5.86 (n 11096) | +98.70 ± 9.49 (n 3854) |
| A4t_s0 vs A2s_s0 | +88.14 ± 34.91 (n 1462) | +12.11 ± 2.56 (n 13488) | +17.88 ± 5.21 (n 11096) | +24.35 ± 5.56 (n 3854) |
| A5h_s0 vs A2s_s0 | +9.96 ± 5.06 (n 1462) | +4.50 ± 2.00 (n 13488) | +5.41 ± 2.07 (n 11096) | +3.96 ± 4.12 (n 3854) |
| A6i_s0 vs A2s_s0 | +4.06 ± 3.15 (n 1462) | +6.67 ± 1.42 (n 13488) | +5.81 ± 1.55 (n 11096) | +8.17 ± 2.45 (n 3854) |
| A6l_s0 vs A2s_s0 | +2.49 ± 3.17 (n 1462) | +10.44 ± 1.61 (n 13488) | +9.43 ± 1.74 (n 11096) | +10.32 ± 2.86 (n 3854) |
| A6b_s0 vs A2s_s0 | +7.22 ± 16.20 (n 1462) | +0.40 ± 1.71 (n 13488) | -0.70 ± 2.71 (n 11096) | +6.18 ± 3.57 (n 3854) |
| A7s_s0 vs A2s_s0 | -1.58 ± 1.43 (n 1462) | -0.40 ± 0.25 (n 13488) | -0.35 ± 0.31 (n 11096) | -0.99 ± 0.50 (n 3854) |
| A7s_s1 vs A2s_s1 | -1.32 ± 1.45 (n 1462) | -0.38 ± 0.26 (n 13488) | -0.35 ± 0.32 (n 11096) | -0.84 ± 0.51 (n 3854) |
| A7l_s0 vs A2s_s0 | +0.43 ± 0.90 (n 1462) | -0.27 ± 0.12 (n 13488) | -0.22 ± 0.16 (n 11096) | -0.15 ± 0.25 (n 3854) |
| A7l_s1 vs A2s_s1 | +0.51 ± 0.89 (n 1462) | -0.25 ± 0.12 (n 13488) | -0.20 ± 0.16 (n 11096) | -0.12 ± 0.25 (n 3854) |
| A8c_s0 vs A2s_s0 | -0.30 ± 1.16 (n 1462) | -0.22 ± 0.10 (n 13488) | -0.20 ± 0.17 (n 11096) | -0.31 ± 0.29 (n 3854) |
| A8c_s1 vs A2s_s1 | -0.19 ± 1.11 (n 1462) | -0.20 ± 0.10 (n 13488) | -0.19 ± 0.16 (n 11096) | -0.25 ± 0.28 (n 3854) |
| A78_s0 vs A2s_s0 | -3.87 ± 1.78 (n 1462) | -0.14 ± 0.48 (n 13488) | +0.04 ± 0.57 (n 11096) | -2.06 ± 0.73 (n 3854) |
| A78_s1 vs A2s_s1 | -3.57 ± 1.79 (n 1462) | -0.13 ± 0.49 (n 13488) | +0.06 ± 0.58 (n 11096) | -1.97 ± 0.74 (n 3854) |
| A9r_s0 vs A2s_s0 | +0.19 ± 9.14 (n 1462) | -5.16 ± 0.95 (n 13488) | -3.06 ± 1.46 (n 11096) | -9.17 ± 2.37 (n 3854) |
| A9r_s1 vs A2s_s1 | +0.23 ± 9.11 (n 1462) | -5.17 ± 0.96 (n 13488) | -3.01 ± 1.46 (n 11096) | -9.34 ± 2.39 (n 3854) |
| A3f_s0 vs A2s_s0 | +4.17 ± 2.96 (n 1462) | +8.46 ± 1.11 (n 13488) | +5.29 ± 1.33 (n 11096) | +15.96 ± 1.27 (n 3854) |
| A3c_s0 vs A2s_s0 | +1.65 ± 2.12 (n 1462) | +6.72 ± 1.03 (n 13488) | +2.92 ± 1.21 (n 11096) | +15.75 ± 1.19 (n 3854) |
| A3x_s0 vs A2s_s0 | -0.00 ± 1.14 (n 1462) | +0.41 ± 0.50 (n 13488) | -1.24 ± 0.59 (n 11096) | +5.00 ± 0.60 (n 3854) |
| A3x_s1 vs A2s_s1 | +0.17 ± 1.14 (n 1462) | +0.42 ± 0.50 (n 13488) | -1.18 ± 0.59 (n 11096) | +4.92 ± 0.60 (n 3854) |
| A10-2_s0 vs A2s_s0 | +432.92 ± 38.97 (n 1462) | +81.29 ± 3.90 (n 13488) | +117.43 ± 6.15 (n 11096) | +110.61 ± 10.13 (n 3854) |
| A10-4_s0 vs A2s_s0 | +171.57 ± 29.32 (n 1462) | +30.72 ± 3.10 (n 13488) | +36.49 ± 4.38 (n 11096) | +67.54 ± 9.16 (n 3854) |
| A10-5_s0 vs A2s_s0 | +92.87 ± 23.21 (n 1462) | +12.40 ± 2.26 (n 13488) | +16.88 ± 3.48 (n 11096) | +30.04 ± 6.32 (n 3854) |
| A10g_s0 vs A2s_s0 | +134.39 ± 30.17 (n 1462) | +21.40 ± 2.86 (n 13488) | +29.84 ± 4.67 (n 11096) | +39.97 ± 7.20 (n 3854) |
| A12_s0 vs A2s_s0 | +4.24 ± 2.28 (n 1462) | -1.74 ± 0.88 (n 13488) | -1.66 ± 1.08 (n 11096) | +0.30 ± 0.72 (n 3854) |
| A12_s1 vs A2s_s1 | +4.34 ± 2.30 (n 1462) | -1.72 ± 0.88 (n 13488) | -1.63 ± 1.09 (n 11096) | +0.31 ± 0.72 (n 3854) |
| A4n_s0 vs A2n_s0 | +3.32 ± 4.08 (n 1462) | +0.25 ± 0.99 (n 13488) | -0.49 ± 1.14 (n 11096) | +3.54 ± 1.90 (n 3854) |
| A5n_s0 vs A2n_s0 | +8.22 ± 4.17 (n 1462) | +2.54 ± 1.16 (n 13488) | +0.03 ± 1.27 (n 11096) | +11.93 ± 2.40 (n 3854) |
| A7n_s0 vs A2n_s0 | -1.79 ± 1.74 (n 1462) | +0.33 ± 0.34 (n 13488) | -0.02 ± 0.41 (n 11096) | +0.52 ± 0.67 (n 3854) |
| A8n_s0 vs A2n_s0 | -2.50 ± 1.00 (n 1462) | -0.11 ± 0.31 (n 13488) | +0.03 ± 0.36 (n 11096) | -1.40 ± 0.44 (n 3854) |
| A9n_s0 vs A2n_s0 | -1.62 ± 3.14 (n 1462) | -5.54 ± 1.05 (n 13488) | -4.14 ± 0.97 (n 11096) | -8.06 ± 2.69 (n 3854) |
| A10n_s0 vs A2n_s0 | +284.73 ± 35.66 (n 1462) | +27.61 ± 2.41 (n 13488) | +45.83 ± 5.03 (n 11096) | +72.69 ± 7.11 (n 3854) |

## Guards (own tables; K = 1024, bulk stratum, mass monitors, G-region)

### A7_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.43 ± 0.27 (n 14950) | -0.18 ± 0.13 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.47 ± 0.26 (n 14784) | -0.16 ± 0.13 | yes | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.63 ± 0.26 (n 14784); zb<1221 (short events) +10.48 ± 6.61 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.40 ± 0.02 (n 14950) |
| hit term: border_ring | +0.10 ± 0.05 (n 14950) |
| hit term: border_zbin | -0.02 ± 0.01 (n 14950) |
| hit term: low_abar | -0.30 ± 0.04 (n 14950) |
| hit term: flagged | +0.21 ± 0.07 (n 14950) |
| hit term: inner | -0.94 ± 0.25 (n 14950) |
| hit term: total | -0.73 ± 0.27 (n 14950) |
| normalisation term (own row) | +0.22 ± 0.01 (n 14950) |

flagged share of the gain -0.407; margin flagged - 0.5 main +0.46 ± 0.12 (n 14950) -> passed yes, resolved yes; decomposition residual 1.5e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.282; margin flagged - 0.5 hit term +0.57 ± 0.12 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4448e-03 | 3.4674e-03 | -0.23 ± 0.00 (n 14950) |
| first_z | 1.0409e-05 | 9.6426e-06 | +0.01 ± 0.00 (n 14950) |
| inside_zb | 9.9656e-01 | 9.9653e-01 | +0.23 ± 0.00 (n 14950) |
| last_ring | 1.9272e-03 | 1.9366e-03 | -0.09 ± 0.00 (n 14950) |
| last_z | 1.1833e-04 | 1.1992e-04 | -0.02 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7122e-05 | 1.6899e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.6971e-02 | 1.7075e-02 | -1.04 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A8_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | +0.08 ± 0.29 (n 14950) | +0.11 ± 0.15 | no | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.30 ± 0.26 (n 14784) | -0.20 ± 0.14 | yes | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.20 ± 0.27 (n 14784); zb<1221 (short events) +12.05 ± 9.65 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.12 ± 0.05 (n 14950) |
| hit term: border_ring | +0.14 ± 0.09 (n 14950) |
| hit term: border_zbin | +0.00 ± 0.00 (n 14950) |
| hit term: low_abar | -0.02 ± 0.01 (n 14950) |
| hit term: flagged | -0.01 ± 0.10 (n 14950) |
| hit term: inner | -0.02 ± 0.26 (n 14950) |
| hit term: total | -0.03 ± 0.29 (n 14950) |
| normalisation term (own row) | -0.04 ± 0.00 (n 14950) |

flagged share of the gain 0.142; margin flagged - 0.5 main +0.02 ± 0.13 (n 14950) -> passed yes, resolved no; decomposition residual 1.3e-05
*alternative - decides nothing:* flagged share of the hit-term total 0.344; margin flagged - 0.5 hit term +0.00 ± 0.13 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4712e-03 | 3.4674e-03 | +0.04 ± 0.00 (n 14950) |
| first_z | 9.6661e-06 | 9.6426e-06 | +0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9653e-01 | 9.9653e-01 | -0.04 ± 0.00 (n 14950) |
| last_ring | 1.9447e-03 | 1.9366e-03 | +0.08 ± 0.00 (n 14950) |
| last_z | 1.1981e-04 | 1.1992e-04 | -0.00 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6986e-05 | 1.6899e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7095e-02 | 1.7075e-02 | +0.20 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A9_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -4.65 ± 0.88 (n 14950) | -2.31 ± 0.44 | yes | yes |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -4.43 ± 0.87 (n 14784) | -2.03 ± 0.44 | yes | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) -4.81 ± 0.89 (n 14784); zb<1221 (short events) +8.04 ± 7.24 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.12 ± 0.05 (n 14950) |
| hit term: border_ring | -0.32 ± 0.13 (n 14950) |
| hit term: border_zbin | -0.36 ± 0.08 (n 14950) |
| hit term: low_abar | -2.97 ± 0.53 (n 14950) |
| hit term: flagged | -3.31 ± 0.55 (n 14950) |
| hit term: inner | -1.32 ± 0.61 (n 14950) |
| hit term: total | -4.63 ± 0.88 (n 14950) |
| normalisation term (own row) | -0.04 ± 0.02 (n 14950) |

flagged share of the gain 0.709; margin flagged - 0.5 main -0.98 ± 0.37 (n 14950) -> passed no, resolved yes; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total 0.715; margin flagged - 0.5 hit term -0.99 ± 0.37 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4683e-03 | 3.4674e-03 | +0.01 ± 0.00 (n 14950) |
| first_z | 3.2668e-05 | 9.6426e-06 | +0.23 ± 0.00 (n 14950) |
| inside_zb | 9.9653e-01 | 9.9653e-01 | -0.01 ± 0.00 (n 14950) |
| last_ring | 1.9402e-03 | 1.9366e-03 | +0.04 ± 0.00 (n 14950) |
| last_z | 1.1941e-04 | 1.1992e-04 | -0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6268e-05 | 1.6899e-05 | -0.01 ± 0.00 (n 14950) |
| outside_bulk | 1.7114e-02 | 1.7075e-02 | +0.39 ± 0.00 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A7_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.38 ± 0.28 (n 14950) | -0.16 ± 0.14 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.41 ± 0.27 (n 14784) | -0.12 ± 0.14 | yes | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.58 ± 0.27 (n 14784); zb<1221 (short events) +10.04 ± 6.98 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.41 ± 0.02 (n 14950) |
| hit term: border_ring | +0.09 ± 0.05 (n 14950) |
| hit term: border_zbin | -0.03 ± 0.01 (n 14950) |
| hit term: low_abar | -0.32 ± 0.05 (n 14950) |
| hit term: flagged | +0.18 ± 0.08 (n 14950) |
| hit term: inner | -0.87 ± 0.26 (n 14950) |
| hit term: total | -0.69 ± 0.28 (n 14950) |
| normalisation term (own row) | +0.23 ± 0.01 (n 14950) |

flagged share of the gain -0.395; margin flagged - 0.5 main +0.41 ± 0.13 (n 14950) -> passed yes, resolved yes; decomposition residual 1.6e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.264; margin flagged - 0.5 hit term +0.53 ± 0.13 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4690e-03 | 3.4922e-03 | -0.23 ± 0.00 (n 14950) |
| first_z | 1.0402e-05 | 9.5894e-06 | +0.01 ± 0.00 (n 14950) |
| inside_zb | 9.9653e-01 | 9.9651e-01 | +0.23 ± 0.00 (n 14950) |
| last_ring | 1.9292e-03 | 1.9375e-03 | -0.08 ± 0.00 (n 14950) |
| last_z | 1.1951e-04 | 1.2113e-04 | -0.02 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7087e-05 | 1.6859e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7093e-02 | 1.7198e-02 | -1.05 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A8_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | +0.10 ± 0.28 (n 14950) | +0.12 ± 0.14 | no | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.31 ± 0.26 (n 14784) | -0.21 ± 0.14 | yes | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.20 ± 0.27 (n 14784); zb<1221 (short events) +13.78 ± 9.70 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.12 ± 0.04 (n 14950) |
| hit term: border_ring | +0.15 ± 0.09 (n 14950) |
| hit term: border_zbin | +0.00 ± 0.00 (n 14950) |
| hit term: low_abar | -0.02 ± 0.02 (n 14950) |
| hit term: flagged | +0.00 ± 0.10 (n 14950) |
| hit term: inner | -0.01 ± 0.26 (n 14950) |
| hit term: total | -0.00 ± 0.29 (n 14950) |
| normalisation term (own row) | -0.04 ± 0.00 (n 14950) |

flagged share of the gain -0.11; margin flagged - 0.5 main +0.03 ± 0.13 (n 14950) -> passed yes, resolved no; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total -1.042; margin flagged - 0.5 hit term +0.01 ± 0.13 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4962e-03 | 3.4922e-03 | +0.04 ± 0.00 (n 14950) |
| first_z | 9.6125e-06 | 9.5894e-06 | +0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9650e-01 | 9.9651e-01 | -0.04 ± 0.00 (n 14950) |
| last_ring | 1.9446e-03 | 1.9375e-03 | +0.07 ± 0.00 (n 14950) |
| last_z | 1.2107e-04 | 1.2113e-04 | -0.00 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6957e-05 | 1.6859e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7217e-02 | 1.7198e-02 | +0.19 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A9_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -4.63 ± 0.89 (n 14950) | -2.30 ± 0.44 | yes | yes |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -4.45 ± 0.88 (n 14784) | -2.04 ± 0.44 | yes | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) -4.82 ± 0.90 (n 14784); zb<1221 (short events) +10.25 ± 7.31 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.12 ± 0.05 (n 14950) |
| hit term: border_ring | -0.29 ± 0.13 (n 14950) |
| hit term: border_zbin | -0.36 ± 0.08 (n 14950) |
| hit term: low_abar | -3.00 ± 0.53 (n 14950) |
| hit term: flagged | -3.33 ± 0.55 (n 14950) |
| hit term: inner | -1.29 ± 0.61 (n 14950) |
| hit term: total | -4.61 ± 0.89 (n 14950) |
| normalisation term (own row) | -0.04 ± 0.02 (n 14950) |

flagged share of the gain 0.715; margin flagged - 0.5 main -1.00 ± 0.38 (n 14950) -> passed no, resolved yes; decomposition residual 1.5e-05
*alternative - decides nothing:* flagged share of the hit-term total 0.721; margin flagged - 0.5 hit term -1.02 ± 0.38 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4935e-03 | 3.4922e-03 | +0.01 ± 0.00 (n 14950) |
| first_z | 3.2588e-05 | 9.5894e-06 | +0.23 ± 0.00 (n 14950) |
| inside_zb | 9.9651e-01 | 9.9651e-01 | -0.01 ± 0.00 (n 14950) |
| last_ring | 1.9390e-03 | 1.9375e-03 | +0.02 ± 0.00 (n 14950) |
| last_z | 1.2064e-04 | 1.2113e-04 | -0.00 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6240e-05 | 1.6859e-05 | -0.01 ± 0.00 (n 14950) |
| outside_bulk | 1.7237e-02 | 1.7198e-02 | +0.39 ± 0.00 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | -0.00 ± 0.00 (n 14950) |

### A4n_s0 vs A2n_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | +0.48 ± 0.97 (n 14950) | +0.20 ± 0.49 | no | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | +0.76 ± 0.95 (n 14784) | +0.61 ± 0.48 | no | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) +0.30 ± 0.97 (n 14784); zb<1221 (short events) +22.64 ± 13.68 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.40 ± 0.11 (n 14950) |
| hit term: border_ring | +0.17 ± 0.04 (n 14950) |
| hit term: border_zbin | +0.08 ± 0.03 (n 14950) |
| hit term: low_abar | +0.39 ± 0.11 (n 14950) |
| hit term: flagged | +0.15 ± 0.16 (n 14950) |
| hit term: inner | +0.77 ± 0.95 (n 14950) |
| hit term: total | +0.92 ± 0.98 (n 14950) |
| normalisation term (own row) | -0.37 ± 0.05 (n 14950) |

flagged share of the gain 0.276; margin flagged - 0.5 main -0.12 ± 0.47 (n 14950) -> passed no, resolved no; decomposition residual 1.6e-05
*alternative - decides nothing:* flagged share of the hit-term total 0.166; margin flagged - 0.5 hit term -0.31 ± 0.47 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.5314e-03 | 3.5003e-03 | +0.31 ± 0.02 (n 14950) |
| first_z | 6.8176e-06 | 9.2933e-06 | -0.02 ± 0.00 (n 14950) |
| inside_zb | 9.9647e-01 | 9.9650e-01 | -0.31 ± 0.02 (n 14950) |
| last_ring | 1.9185e-03 | 1.9443e-03 | -0.26 ± 0.00 (n 14950) |
| last_z | 1.2610e-04 | 1.2518e-04 | +0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6963e-05 | 1.7092e-05 | -0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7346e-02 | 1.7272e-02 | +0.74 ± 0.05 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A7n_s0 vs A2n_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | +0.18 ± 0.35 (n 14950) | +0.12 ± 0.17 | no | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.06 ± 0.33 (n 14784) | +0.03 ± 0.17 | no | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.20 ± 0.34 (n 14784); zb<1221 (short events) +28.35 ± 8.98 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.16 ± 0.02 (n 14950) |
| hit term: border_ring | +0.21 ± 0.08 (n 14950) |
| hit term: border_zbin | -0.02 ± 0.01 (n 14950) |
| hit term: low_abar | -0.04 ± 0.06 (n 14950) |
| hit term: flagged | +0.32 ± 0.10 (n 14950) |
| hit term: inner | -0.34 ± 0.32 (n 14950) |
| hit term: total | -0.02 ± 0.35 (n 14950) |
| normalisation term (own row) | +0.14 ± 0.01 (n 14950) |

flagged share of the gain 2.679; margin flagged - 0.5 main +0.26 ± 0.16 (n 14950) -> passed no, resolved no; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total -19.542; margin flagged - 0.5 hit term +0.33 ± 0.16 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4866e-03 | 3.5003e-03 | -0.14 ± 0.00 (n 14950) |
| first_z | 9.9065e-06 | 9.2933e-06 | +0.01 ± 0.00 (n 14950) |
| inside_zb | 9.9651e-01 | 9.9650e-01 | +0.14 ± 0.00 (n 14950) |
| last_ring | 1.9292e-03 | 1.9443e-03 | -0.15 ± 0.00 (n 14950) |
| last_z | 1.2230e-04 | 1.2518e-04 | -0.03 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7257e-05 | 1.7092e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7216e-02 | 1.7272e-02 | -0.56 ± 0.00 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | -0.00 ± 0.00 (n 14950) |

### A8n_s0 vs A2n_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.28 ± 0.29 (n 14950) | -0.11 ± 0.15 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.96 ± 0.25 (n 14784) | -0.51 ± 0.13 | yes | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.89 ± 0.26 (n 14784); zb<1221 (short events) +48.75 ± 11.35 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.25 ± 0.05 (n 14950) |
| hit term: border_ring | +0.34 ± 0.12 (n 14950) |
| hit term: border_zbin | -0.00 ± 0.00 (n 14950) |
| hit term: low_abar | +0.01 ± 0.03 (n 14950) |
| hit term: flagged | +0.10 ± 0.13 (n 14950) |
| hit term: inner | -0.31 ± 0.25 (n 14950) |
| hit term: total | -0.21 ± 0.29 (n 14950) |
| normalisation term (own row) | -0.13 ± 0.01 (n 14950) |

flagged share of the gain -0.289; margin flagged - 0.5 main +0.27 ± 0.13 (n 14950) -> passed yes, resolved yes; decomposition residual 1.6e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.471; margin flagged - 0.5 hit term +0.20 ± 0.13 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.5135e-03 | 3.5003e-03 | +0.13 ± 0.00 (n 14950) |
| first_z | 9.2390e-06 | 9.2933e-06 | -0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9649e-01 | 9.9650e-01 | -0.13 ± 0.00 (n 14950) |
| last_ring | 1.9432e-03 | 1.9443e-03 | -0.01 ± 0.00 (n 14950) |
| last_z | 1.2535e-04 | 1.2518e-04 | +0.00 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7053e-05 | 1.7092e-05 | -0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7316e-02 | 1.7272e-02 | +0.44 ± 0.00 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A9n_s0 vs A2n_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -5.18 ± 1.00 (n 14950) | -2.60 ± 0.50 | yes | yes |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -4.91 ± 0.98 (n 14784) | -2.19 ± 0.49 | yes | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) -5.43 ± 1.00 (n 14784); zb<1221 (short events) +19.38 ± 12.37 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.01 ± 0.05 (n 14950) |
| hit term: border_ring | -0.31 ± 0.15 (n 14950) |
| hit term: border_zbin | -0.40 ± 0.09 (n 14950) |
| hit term: low_abar | -2.83 ± 0.63 (n 14950) |
| hit term: flagged | -3.05 ± 0.65 (n 14950) |
| hit term: inner | -2.14 ± 0.64 (n 14950) |
| hit term: total | -5.19 ± 1.00 (n 14950) |
| normalisation term (own row) | +0.04 ± 0.02 (n 14950) |

flagged share of the gain 0.591; margin flagged - 0.5 main -0.47 ± 0.41 (n 14950) -> passed no, resolved no; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total 0.587; margin flagged - 0.5 hit term -0.45 ± 0.41 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4925e-03 | 3.5003e-03 | -0.08 ± 0.00 (n 14950) |
| first_z | 3.6426e-05 | 9.2933e-06 | +0.27 ± 0.00 (n 14950) |
| inside_zb | 9.9651e-01 | 9.9650e-01 | +0.08 ± 0.00 (n 14950) |
| last_ring | 1.9441e-03 | 1.9443e-03 | -0.00 ± 0.00 (n 14950) |
| last_z | 1.2446e-04 | 1.2518e-04 | -0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.5928e-05 | 1.7092e-05 | -0.01 ± 0.00 (n 14950) |
| outside_bulk | 1.7281e-02 | 1.7272e-02 | +0.08 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A7s_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.45 ± 0.27 (n 14950) | -0.19 ± 0.13 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.49 ± 0.25 (n 14784) | -0.17 ± 0.13 | yes | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.64 ± 0.26 (n 14784); zb<1221 (short events) +10.17 ± 6.76 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.40 ± 0.02 (n 14950) |
| hit term: border_ring | +0.08 ± 0.05 (n 14950) |
| hit term: border_zbin | -0.02 ± 0.01 (n 14950) |
| hit term: low_abar | -0.27 ± 0.04 (n 14950) |
| hit term: flagged | +0.21 ± 0.07 (n 14950) |
| hit term: inner | -0.95 ± 0.24 (n 14950) |
| hit term: total | -0.74 ± 0.27 (n 14950) |
| normalisation term (own row) | +0.23 ± 0.01 (n 14950) |

flagged share of the gain -0.396; margin flagged - 0.5 main +0.46 ± 0.12 (n 14950) -> passed yes, resolved yes; decomposition residual 1.6e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.276; margin flagged - 0.5 hit term +0.58 ± 0.12 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4446e-03 | 3.4674e-03 | -0.23 ± 0.00 (n 14950) |
| first_z | 1.0430e-05 | 9.6426e-06 | +0.01 ± 0.00 (n 14950) |
| inside_zb | 9.9656e-01 | 9.9653e-01 | +0.23 ± 0.00 (n 14950) |
| last_ring | 1.9297e-03 | 1.9366e-03 | -0.07 ± 0.00 (n 14950) |
| last_z | 1.1830e-04 | 1.1992e-04 | -0.02 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7064e-05 | 1.6899e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.6974e-02 | 1.7075e-02 | -1.01 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A7s_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.39 ± 0.27 (n 14950) | -0.16 ± 0.14 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.43 ± 0.26 (n 14784) | -0.13 ± 0.13 | yes | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.60 ± 0.26 (n 14784); zb<1221 (short events) +10.38 ± 7.04 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.40 ± 0.02 (n 14950) |
| hit term: border_ring | +0.09 ± 0.06 (n 14950) |
| hit term: border_zbin | -0.03 ± 0.01 (n 14950) |
| hit term: low_abar | -0.29 ± 0.05 (n 14950) |
| hit term: flagged | +0.19 ± 0.08 (n 14950) |
| hit term: inner | -0.90 ± 0.25 (n 14950) |
| hit term: total | -0.70 ± 0.27 (n 14950) |
| normalisation term (own row) | +0.23 ± 0.01 (n 14950) |

flagged share of the gain -0.41; margin flagged - 0.5 main +0.43 ± 0.12 (n 14950) -> passed yes, resolved yes; decomposition residual 1.6e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.276; margin flagged - 0.5 hit term +0.55 ± 0.12 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4690e-03 | 3.4922e-03 | -0.23 ± 0.00 (n 14950) |
| first_z | 1.0407e-05 | 9.5894e-06 | +0.01 ± 0.00 (n 14950) |
| inside_zb | 9.9653e-01 | 9.9651e-01 | +0.23 ± 0.00 (n 14950) |
| last_ring | 1.9305e-03 | 1.9375e-03 | -0.07 ± 0.00 (n 14950) |
| last_z | 1.1949e-04 | 1.2113e-04 | -0.02 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7035e-05 | 1.6859e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7096e-02 | 1.7198e-02 | -1.02 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | -0.00 ± 0.00 (n 14950) |

### A7l_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.14 ± 0.14 (n 14950) | -0.04 ± 0.07 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.00 ± 0.13 (n 14784) | +0.04 ± 0.07 | no | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.08 ± 0.13 (n 14784); zb<1221 (short events) -11.16 ± 3.25 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.29 ± 0.03 (n 14950) |
| hit term: border_ring | -0.09 ± 0.04 (n 14950) |
| hit term: border_zbin | +0.00 ± 0.00 (n 14950) |
| hit term: low_abar | -0.07 ± 0.02 (n 14950) |
| hit term: flagged | +0.12 ± 0.05 (n 14950) |
| hit term: inner | -0.49 ± 0.12 (n 14950) |
| hit term: total | -0.37 ± 0.14 (n 14950) |
| normalisation term (own row) | +0.17 ± 0.00 (n 14950) |

flagged share of the gain -0.593; margin flagged - 0.5 main +0.22 ± 0.06 (n 14950) -> passed yes, resolved yes; decomposition residual 1.6e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.321; margin flagged - 0.5 hit term +0.31 ± 0.06 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4504e-03 | 3.4674e-03 | -0.17 ± 0.00 (n 14950) |
| first_z | 9.6259e-06 | 9.6426e-06 | -0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9655e-01 | 9.9653e-01 | +0.17 ± 0.00 (n 14950) |
| last_ring | 1.9358e-03 | 1.9366e-03 | -0.01 ± 0.00 (n 14950) |
| last_z | 1.1872e-04 | 1.1992e-04 | -0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6987e-05 | 1.6899e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7008e-02 | 1.7075e-02 | -0.67 ± 0.00 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A7l_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.11 ± 0.14 (n 14950) | -0.03 ± 0.07 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | +0.02 ± 0.13 (n 14784) | +0.05 ± 0.07 | no | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.06 ± 0.13 (n 14784); zb<1221 (short events) -10.72 ± 3.20 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.28 ± 0.03 (n 14950) |
| hit term: border_ring | -0.08 ± 0.04 (n 14950) |
| hit term: border_zbin | +0.00 ± 0.00 (n 14950) |
| hit term: low_abar | -0.08 ± 0.02 (n 14950) |
| hit term: flagged | +0.12 ± 0.05 (n 14950) |
| hit term: inner | -0.46 ± 0.12 (n 14950) |
| hit term: total | -0.35 ± 0.14 (n 14950) |
| normalisation term (own row) | +0.17 ± 0.00 (n 14950) |

flagged share of the gain -0.661; margin flagged - 0.5 main +0.20 ± 0.06 (n 14950) -> passed yes, resolved yes; decomposition residual 1.3e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.338; margin flagged - 0.5 hit term +0.29 ± 0.06 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4752e-03 | 3.4922e-03 | -0.17 ± 0.00 (n 14950) |
| first_z | 9.5798e-06 | 9.5894e-06 | -0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9652e-01 | 9.9651e-01 | +0.17 ± 0.00 (n 14950) |
| last_ring | 1.9362e-03 | 1.9375e-03 | -0.01 ± 0.00 (n 14950) |
| last_z | 1.1995e-04 | 1.2113e-04 | -0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6959e-05 | 1.6859e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7131e-02 | 1.7198e-02 | -0.67 ± 0.00 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | -0.00 ± 0.00 (n 14950) |

### A8c_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.17 ± 0.14 (n 14950) | -0.06 ± 0.07 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.03 ± 0.14 (n 14784) | +0.01 ± 0.07 | no | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.07 ± 0.14 (n 14784); zb<1221 (short events) -14.62 ± 2.94 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.22 ± 0.02 (n 14950) |
| hit term: border_ring | -0.05 ± 0.02 (n 14950) |
| hit term: border_zbin | +0.00 ± 0.00 (n 14950) |
| hit term: low_abar | -0.07 ± 0.02 (n 14950) |
| hit term: flagged | +0.09 ± 0.04 (n 14950) |
| hit term: inner | -0.45 ± 0.14 (n 14950) |
| hit term: total | -0.36 ± 0.14 (n 14950) |
| normalisation term (own row) | +0.13 ± 0.00 (n 14950) |

flagged share of the gain -0.395; margin flagged - 0.5 main +0.21 ± 0.07 (n 14950) -> passed yes, resolved yes; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.249; margin flagged - 0.5 hit term +0.27 ± 0.07 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4541e-03 | 3.4674e-03 | -0.13 ± 0.00 (n 14950) |
| first_z | 9.6379e-06 | 9.6426e-06 | -0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9655e-01 | 9.9653e-01 | +0.13 ± 0.00 (n 14950) |
| last_ring | 1.9376e-03 | 1.9366e-03 | +0.01 ± 0.00 (n 14950) |
| last_z | 1.1889e-04 | 1.1992e-04 | -0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7007e-05 | 1.6899e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7024e-02 | 1.7075e-02 | -0.51 ± 0.00 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A8c_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.14 ± 0.14 (n 14950) | -0.04 ± 0.07 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -0.00 ± 0.13 (n 14784) | +0.02 ± 0.07 | no | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.05 ± 0.14 (n 14784); zb<1221 (short events) -13.90 ± 2.85 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.22 ± 0.02 (n 14950) |
| hit term: border_ring | -0.05 ± 0.02 (n 14950) |
| hit term: border_zbin | +0.00 ± 0.00 (n 14950) |
| hit term: low_abar | -0.08 ± 0.02 (n 14950) |
| hit term: flagged | +0.09 ± 0.04 (n 14950) |
| hit term: inner | -0.42 ± 0.13 (n 14950) |
| hit term: total | -0.33 ± 0.14 (n 14950) |
| normalisation term (own row) | +0.13 ± 0.00 (n 14950) |

flagged share of the gain -0.449; margin flagged - 0.5 main +0.19 ± 0.07 (n 14950) -> passed yes, resolved yes; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.271; margin flagged - 0.5 hit term +0.26 ± 0.07 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4789e-03 | 3.4922e-03 | -0.13 ± 0.00 (n 14950) |
| first_z | 9.5884e-06 | 9.5894e-06 | -0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9652e-01 | 9.9651e-01 | +0.13 ± 0.00 (n 14950) |
| last_ring | 1.9379e-03 | 1.9375e-03 | +0.00 ± 0.00 (n 14950) |
| last_z | 1.2012e-04 | 1.2113e-04 | -0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6976e-05 | 1.6859e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7147e-02 | 1.7198e-02 | -0.51 ± 0.00 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A78_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.33 ± 0.46 (n 14950) | -0.08 ± 0.23 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -1.08 ± 0.41 (n 14784) | -0.52 ± 0.22 | yes | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) -1.11 ± 0.43 (n 14784); zb<1221 (short events) +53.41 ± 17.21 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.03 ± 0.06 (n 14950) |
| hit term: border_ring | +0.44 ± 0.16 (n 14950) |
| hit term: border_zbin | -0.03 ± 0.01 (n 14950) |
| hit term: low_abar | -0.36 ± 0.06 (n 14950) |
| hit term: flagged | +0.05 ± 0.19 (n 14950) |
| hit term: inner | -0.56 ± 0.40 (n 14950) |
| hit term: total | -0.51 ± 0.47 (n 14950) |
| normalisation term (own row) | +0.01 ± 0.01 (n 14950) |

flagged share of the gain -0.101; margin flagged - 0.5 main +0.30 ± 0.21 (n 14950) -> passed yes, resolved no; decomposition residual 1.5e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.098; margin flagged - 0.5 hit term +0.31 ± 0.21 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4658e-03 | 3.4674e-03 | -0.02 ± 0.00 (n 14950) |
| first_z | 1.0800e-05 | 9.6426e-06 | +0.01 ± 0.00 (n 14950) |
| inside_zb | 9.9653e-01 | 9.9653e-01 | +0.02 ± 0.00 (n 14950) |
| last_ring | 1.9288e-03 | 1.9366e-03 | -0.08 ± 0.00 (n 14950) |
| last_z | 1.1935e-04 | 1.1992e-04 | -0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7152e-05 | 1.6899e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7049e-02 | 1.7075e-02 | -0.26 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A78_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -0.30 ± 0.47 (n 14950) | -0.06 ± 0.24 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -1.06 ± 0.41 (n 14784) | -0.52 ± 0.22 | yes | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) -1.09 ± 0.43 (n 14784); zb<1221 (short events) +55.33 ± 17.62 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -0.01 ± 0.06 (n 14950) |
| hit term: border_ring | +0.45 ± 0.17 (n 14950) |
| hit term: border_zbin | -0.03 ± 0.01 (n 14950) |
| hit term: low_abar | -0.38 ± 0.06 (n 14950) |
| hit term: flagged | +0.05 ± 0.19 (n 14950) |
| hit term: inner | -0.54 ± 0.40 (n 14950) |
| hit term: total | -0.48 ± 0.47 (n 14950) |
| normalisation term (own row) | +0.02 ± 0.01 (n 14950) |

flagged share of the gain -0.114; margin flagged - 0.5 main +0.29 ± 0.21 (n 14950) -> passed yes, resolved no; decomposition residual 1.5e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.11; margin flagged - 0.5 hit term +0.30 ± 0.21 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4898e-03 | 3.4922e-03 | -0.02 ± 0.00 (n 14950) |
| first_z | 1.0784e-05 | 9.5894e-06 | +0.01 ± 0.00 (n 14950) |
| inside_zb | 9.9651e-01 | 9.9651e-01 | +0.02 ± 0.00 (n 14950) |
| last_ring | 1.9296e-03 | 1.9375e-03 | -0.08 ± 0.00 (n 14950) |
| last_z | 1.2055e-04 | 1.2113e-04 | -0.01 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.7116e-05 | 1.6859e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7169e-02 | 1.7198e-02 | -0.29 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | -0.00 ± 0.00 (n 14950) |

### A9r_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -4.58 ± 1.24 (n 14950) | -2.26 ± 0.62 | yes | yes |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -4.66 ± 0.98 (n 14784) | -1.80 ± 0.49 | yes | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) -5.72 ± 1.00 (n 14784); zb<1221 (short events) +91.90 ± 67.16 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.54 ± 0.07 (n 14950) |
| hit term: border_ring | +0.59 ± 0.18 (n 14950) |
| hit term: border_zbin | -0.37 ± 0.09 (n 14950) |
| hit term: low_abar | -2.84 ± 0.54 (n 14950) |
| hit term: flagged | -1.64 ± 0.57 (n 14950) |
| hit term: inner | -3.45 ± 1.04 (n 14950) |
| hit term: total | -5.09 ± 1.24 (n 14950) |
| normalisation term (own row) | +0.46 ± 0.02 (n 14950) |

flagged share of the gain 0.354; margin flagged - 0.5 main +0.68 ± 0.57 (n 14950) -> passed yes, resolved no; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total 0.322; margin flagged - 0.5 hit term +0.91 ± 0.57 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4186e-03 | 3.4674e-03 | -0.49 ± 0.01 (n 14950) |
| first_z | 3.4042e-05 | 9.6426e-06 | +0.24 ± 0.00 (n 14950) |
| inside_zb | 9.9658e-01 | 9.9653e-01 | +0.49 ± 0.01 (n 14950) |
| last_ring | 1.8378e-03 | 1.9366e-03 | -0.99 ± 0.01 (n 14950) |
| last_z | 1.1537e-04 | 1.1992e-04 | -0.05 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.5986e-05 | 1.6899e-05 | -0.01 ± 0.00 (n 14950) |
| outside_bulk | 1.6828e-02 | 1.7075e-02 | -2.47 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A9r_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -4.58 ± 1.24 (n 14950) | -2.26 ± 0.62 | yes | yes |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -4.67 ± 0.98 (n 14784) | -1.80 ± 0.49 | yes | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) -5.75 ± 1.01 (n 14784); zb<1221 (short events) +93.62 ± 66.89 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.54 ± 0.07 (n 14950) |
| hit term: border_ring | +0.61 ± 0.18 (n 14950) |
| hit term: border_zbin | -0.37 ± 0.09 (n 14950) |
| hit term: low_abar | -2.88 ± 0.54 (n 14950) |
| hit term: flagged | -1.66 ± 0.58 (n 14950) |
| hit term: inner | -3.44 ± 1.04 (n 14950) |
| hit term: total | -5.10 ± 1.24 (n 14950) |
| normalisation term (own row) | +0.46 ± 0.02 (n 14950) |

flagged share of the gain 0.357; margin flagged - 0.5 main +0.66 ± 0.57 (n 14950) -> passed yes, resolved no; decomposition residual 1.5e-05
*alternative - decides nothing:* flagged share of the hit-term total 0.325; margin flagged - 0.5 hit term +0.89 ± 0.57 (n 14950) -> would pass yes (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4433e-03 | 3.4922e-03 | -0.49 ± 0.01 (n 14950) |
| first_z | 3.4073e-05 | 9.5894e-06 | +0.24 ± 0.00 (n 14950) |
| inside_zb | 9.9656e-01 | 9.9651e-01 | +0.49 ± 0.01 (n 14950) |
| last_ring | 1.8367e-03 | 1.9375e-03 | -1.01 ± 0.01 (n 14950) |
| last_z | 1.1657e-04 | 1.2113e-04 | -0.05 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.5954e-05 | 1.6859e-05 | -0.01 ± 0.00 (n 14950) |
| outside_bulk | 1.6949e-02 | 1.7198e-02 | -2.49 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | -0.00 ± 0.00 (n 14950) |

### A3x_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | +0.43 ± 0.47 (n 14950) | +0.24 ± 0.23 | no | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | +0.52 ± 0.27 (n 14784) | +0.53 ± 0.28 | no | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) -0.02 ± 0.46 (n 14784); zb<1221 (short events) +34.40 ± 8.52 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -5.72 ± 0.42 (n 14950) |
| hit term: border_ring | +0.36 ± 0.07 (n 14950) |
| hit term: border_zbin | -0.07 ± 0.03 (n 14950) |
| hit term: low_abar | -0.27 ± 0.06 (n 14950) |
| hit term: flagged | -5.41 ± 0.42 (n 14950) |
| hit term: inner | +12.72 ± 0.29 (n 14950) |
| hit term: total | +7.31 ± 0.47 (n 14950) |
| normalisation term (own row) | -6.94 ± 0.05 (n 14950) |

flagged share of the gain -14.757; margin flagged - 0.5 main -5.60 ± 0.28 (n 14950) -> passed no, resolved yes; decomposition residual 1.7e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.741; margin flagged - 0.5 hit term -9.07 ± 0.28 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 4.1569e-03 | 3.4674e-03 | +6.89 ± 0.05 (n 14950) |
| first_z | 9.6493e-06 | 9.6426e-06 | +0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9584e-01 | 9.9653e-01 | -6.89 ± 0.05 (n 14950) |
| last_ring | 1.9222e-03 | 1.9366e-03 | -0.14 ± 0.00 (n 14950) |
| last_z | 2.0989e-04 | 1.1992e-04 | +0.90 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.4297e-05 | 1.6899e-05 | -0.03 ± 0.00 (n 14950) |
| outside_bulk | 1.8139e-02 | 1.7075e-02 | +10.64 ± 0.04 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A3x_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | +0.46 ± 0.47 (n 14950) | +0.26 ± 0.23 | no | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | +0.51 ± 0.27 (n 14784) | +0.51 ± 0.28 | no | no |

Report metric on the strata (own rows): zb>=1221 (bulk events) +0.01 ± 0.46 (n 14784); zb<1221 (short events) +34.78 ± 8.58 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | -5.71 ± 0.42 (n 14950) |
| hit term: border_ring | +0.37 ± 0.07 (n 14950) |
| hit term: border_zbin | -0.07 ± 0.03 (n 14950) |
| hit term: low_abar | -0.28 ± 0.06 (n 14950) |
| hit term: flagged | -5.40 ± 0.42 (n 14950) |
| hit term: inner | +12.79 ± 0.29 (n 14950) |
| hit term: total | +7.39 ± 0.47 (n 14950) |
| normalisation term (own row) | -7.00 ± 0.05 (n 14950) |

flagged share of the gain -13.726; margin flagged - 0.5 main -5.59 ± 0.28 (n 14950) -> passed no, resolved yes; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total -0.73; margin flagged - 0.5 hit term -9.09 ± 0.28 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 4.1873e-03 | 3.4922e-03 | +6.95 ± 0.05 (n 14950) |
| first_z | 9.5975e-06 | 9.5894e-06 | +0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9581e-01 | 9.9651e-01 | -6.95 ± 0.05 (n 14950) |
| last_ring | 1.9229e-03 | 1.9375e-03 | -0.15 ± 0.00 (n 14950) |
| last_z | 2.1195e-04 | 1.2113e-04 | +0.91 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.4270e-05 | 1.6859e-05 | -0.03 ± 0.00 (n 14950) |
| outside_bulk | 1.8268e-02 | 1.7198e-02 | +10.71 ± 0.04 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | -0.00 ± 0.00 (n 14950) |

### A12_s0 vs A2s_s0

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -1.16 ± 0.83 (n 14950) | -0.58 ± 0.41 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -1.16 ± 0.34 (n 14784) | -1.39 ± 0.45 | no | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) +0.44 ± 0.59 (n 14784); zb<1221 (short events) -143.35 ± 51.47 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.51 ± 0.04 (n 14950) |
| hit term: border_ring | -7.58 ± 0.78 (n 14950) |
| hit term: border_zbin | -0.01 ± 0.01 (n 14950) |
| hit term: low_abar | -0.06 ± 0.02 (n 14950) |
| hit term: flagged | -7.05 ± 0.78 (n 14950) |
| hit term: inner | +5.59 ± 0.34 (n 14950) |
| hit term: total | -1.46 ± 0.82 (n 14950) |
| normalisation term (own row) | +0.31 ± 0.00 (n 14950) |

flagged share of the gain 6.118; margin flagged - 0.5 main -6.48 ± 0.44 (n 14950) -> passed no, resolved yes; decomposition residual 1.4e-05
*alternative - decides nothing:* flagged share of the hit-term total 4.835; margin flagged - 0.5 hit term -6.32 ± 0.44 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4369e-03 | 3.4674e-03 | -0.31 ± 0.00 (n 14950) |
| first_z | 9.9185e-06 | 9.6426e-06 | +0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9656e-01 | 9.9653e-01 | +0.31 ± 0.00 (n 14950) |
| last_ring | 2.7942e-03 | 1.9366e-03 | +8.58 ± 0.01 (n 14950) |
| last_z | 1.1750e-04 | 1.1992e-04 | -0.02 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6927e-05 | 1.6899e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7768e-02 | 1.7075e-02 | +6.93 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

### A12_s1 vs A2s_s1

| guard | Δ (guard metric) | persistence margin guard - 0.5 main | passed | resolved (2 SE) |
|---|---|---|---|---|
| G-K (K = 1024) | -1.13 ± 0.83 (n 14950) | -0.57 ± 0.42 | yes | no |
| G-bulk (zb >= 1221 mm, n 14784, excluded 0) | -1.15 ± 0.34 (n 14784) | -1.38 ± 0.45 | no | yes |

Report metric on the strata (own rows): zb>=1221 (bulk events) +0.46 ± 0.59 (n 14784); zb<1221 (short events) -142.88 ± 51.77 (n 166)

| G-region (hit term by hit centre) | Δ |
|---|---|
| hit term: deep | +0.51 ± 0.04 (n 14950) |
| hit term: border_ring | -7.61 ± 0.78 (n 14950) |
| hit term: border_zbin | -0.01 ± 0.01 (n 14950) |
| hit term: low_abar | -0.07 ± 0.02 (n 14950) |
| hit term: flagged | -7.10 ± 0.78 (n 14950) |
| hit term: inner | +5.66 ± 0.34 (n 14950) |
| hit term: total | -1.44 ± 0.83 (n 14950) |
| normalisation term (own row) | +0.31 ± 0.00 (n 14950) |

flagged share of the gain 6.286; margin flagged - 0.5 main -6.53 ± 0.44 (n 14950) -> passed no, resolved yes; decomposition residual 1.6e-05
*alternative - decides nothing:* flagged share of the hit-term total 4.942; margin flagged - 0.5 hit term -6.38 ± 0.44 (n 14950) -> would pass no (requires a gain in the total as well)

| G-mass monitor | run mean | comparator mean | Δ (paired, ×1e-4) |
|---|---|---|---|
| beyond_zb | 3.4614e-03 | 3.4922e-03 | -0.31 ± 0.00 (n 14950) |
| first_z | 9.8750e-06 | 9.5894e-06 | +0.00 ± 0.00 (n 14950) |
| inside_zb | 9.9654e-01 | 9.9651e-01 | +0.31 ± 0.00 (n 14950) |
| last_ring | 2.8004e-03 | 1.9375e-03 | +8.63 ± 0.01 (n 14950) |
| last_z | 1.1871e-04 | 1.2113e-04 | -0.02 ± 0.00 (n 14950) |
| mask_abar_le_0.01 | 1.6898e-05 | 1.6859e-05 | +0.00 ± 0.00 (n 14950) |
| outside_bulk | 1.7894e-02 | 1.7198e-02 | +6.97 ± 0.01 (n 14950) |
| total | 1.0000e+00 | 1.0000e+00 | +0.00 ± 0.00 (n 14950) |

## A6b: Fisher information of k

Observed information of the raw coefficient of `sh_rk` on the training events: 1343 (SE of the coefficient 0.02728552554283298).

## Diverged and blocked runs (user decision of 25 Nov 2026, PLAN 17.11)

Definition: A run is 'diverged' iff (i) a validation mean in its fit history is not finite, or (ii) the registry evaluation of the finished fit finds a non-finite per-event NLL (FloatingPointError), or (iii) 'frozen': in the last evaluation window at least 50 % of the training batches were skipped by sfm.train.fit for a non-finite loss or gradient, or (iv) 'blow-up': the last validation mean exceeds the reference by more than 1 nat per event (10,000 in table units), the reference being the lowest of: the validation means of the fit history, the step-0 validation mean of a network warm start, and the registered validation mean of the run's comparator. Isolated skipped batches of a fit that ends finite are NOT a divergence (they are reported with the run).

None.


## Convergence (|val(last) - val(last - window)| < 0.5e-4)

| run | converged | moved |
|---|---|---|
| A0_s0 | yes | 0.12e-4 |
| A0_s1 | yes | 0.32e-4 |
| A10-2_s0 | no | 1.13e-4 |
| A10-4_s0 | no | 2.16e-4 |
| A10-5_s0 | no | 0.99e-4 |
| A10_s0 | no | 0.76e-4 |
| A10_s0_x2 | no | 0.66e-4 |
| A10_s1 | yes | 0.30e-4 |
| A10g_s0 | no | 2.00e-4 |
| A10n_s0 | no | 1.02e-4 |
| A12_s0 | yes | 0.12e-4 |
| A12_s1 | yes | 0.41e-4 |
| A1_s0 | no | 1.27e-4 |
| A1_s1 | no | 0.61e-4 |
| A2c_s0 | yes | 0.13e-4 |
| A2c_s1 | yes | 0.40e-4 |
| A2e_s0 | yes | 0.11e-4 |
| A2e_s1 | yes | 0.40e-4 |
| A2n_s0 | no | 1.40e-4 |
| A2n_s1 | no | 0.56e-4 |
| A2n_s2 | yes | 0.09e-4 |
| A2s_s0 | yes | 0.12e-4 |
| A2s_s0_x2 | yes | 0.19e-4 |
| A2s_s1 | yes | 0.41e-4 |
| A2s_s1_x2 | yes | 0.06e-4 |
| A2s_s2 | yes | 0.02e-4 |
| A3c_s0 | yes | 0.11e-4 |
| A3f_s0 | yes | 0.10e-4 |
| A3x_s0 | yes | 0.13e-4 |
| A3x_s1 | yes | 0.40e-4 |
| A4_s0 | yes | 0.18e-4 |
| A4_s1 | no | 0.57e-4 |
| A4_s1_x2 | yes | 0.11e-4 |
| A4n_s0 | no | 1.85e-4 |
| A4t_s0 | yes | 0.07e-4 |
| A5_s0 | yes | 0.33e-4 |
| A5_s1 | no | 0.67e-4 |
| A5_s1_x2 | yes | 0.23e-4 |
| A5h_s0 | no | 1.58e-4 |
| A5n_s0 | no | 2.79e-4 |
| A6b_s0 | yes | 0.44e-4 |
| A6i_s0 | yes | 0.44e-4 |
| A6l_s0 | yes | 0.09e-4 |
| A78_s0 | yes | 0.11e-4 |
| A78_s1 | yes | 0.35e-4 |
| A7_s0 | yes | 0.10e-4 |
| A7_s1 | yes | 0.38e-4 |
| A7l_s0 | yes | 0.12e-4 |
| A7l_s1 | yes | 0.40e-4 |
| A7n_s0 | no | 1.41e-4 |
| A7s_s0 | yes | 0.10e-4 |
| A7s_s1 | yes | 0.41e-4 |
| A8_s0 | yes | 0.14e-4 |
| A8_s1 | yes | 0.37e-4 |
| A8c_s0 | yes | 0.12e-4 |
| A8c_s1 | yes | 0.40e-4 |
| A8n_s0 | no | 1.30e-4 |
| A9_s0 | yes | 0.09e-4 |
| A9_s1 | yes | 0.37e-4 |
| A9n_s0 | no | 1.54e-4 |
| A9r_s0 | yes | 0.11e-4 |
| A9r_s1 | yes | 0.33e-4 |
| plr_A2n_lr0.001 | no | 0.92e-4 |
| plr_A2n_lr0.003 | no | 1.19e-4 |
| plr_A2n_lr0.01 | no | 1.55e-4 |
| plr_A2s_lr0.003 | no | 1.22e-4 |
| plr_A2s_lr0.01 | no | 2.72e-4 |
| plr_A2s_lr0.03 | no | 3.57e-4 |

## Initialisation (review B2; kl_map in nat against the start / initialisation / fit_to target)

Net runs with kl_map > 0.0001 nat are flagged: warm start degraded - net column not interpretable for this run (reference: the pre-registered warm start of A1 gives 5.5e-6 nat). 'step-0 parity' = paired difference on validation between the network at step 0 and its source polynomial run (user decision of 25 Nov 2026; for runs made before fit_to version 3 it refers to the warm start rebuilt by warmstart-check). 'rms_max / std' is the worst column's rms error divided by that column's spread (review R5). Target columns with a relative spread between the 2^-20 floor and 0.01 and columns below the scale floor (std < 1e-2, fit_to version 3) are listed.

| run | provider | kl_map | reference | claim | net: fit_to version / rms_max / std (column) / L-BFGS stopped | step-0 parity Δ(step 0 - source) | flag | nearly-constant / below-floor target columns |
|---|---|---|---|---|---|---|---|---|
| A0_s0 | poly | 0.00e+00 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A0_s1 | poly | 0.00e+00 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A10-2_s0 | poly | 6.21e-02 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A10-4_s0 | poly | 3.22e-02 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A10-5_s0 | poly | 1.39e-02 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A10_s0 | poly | 3.17e-02 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A10_s0_x2 | poly | 3.17e-02 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A10_s1 | poly | 3.19e-02 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A10g_s0 | poly | 2.28e-02 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A10n_s0 | net | 2.26e-05 | m2a_dist/A10_s0 |  | v2 / 0.27 % (c0_rsig) / no | -0.17 ± 0.04 (n 14950) vs A10_s0, max 1.1e-02 (rebuilt) | ok | none |
| A12_s0 | poly | -2.22e-09 | m2a_dist/A2s_s0 | exactly A2s: yes |  |  |  |  |
| A12_s1 | poly | 1.60e-09 | m2a_dist/A2s_s1 | exactly A2s: yes |  |  |  |  |
| A1_s0 | net | 7.36e-06 | legacy formula (legacy_centre, legacy coefficients, stored template) |  | v2 / 1.40 % (zs) / no | +0.18 ± 0.08 (n 14950) vs A0_init, max 3.3e-02 (rebuilt) | ok | none |
| A1_s1 | net | 3.03e-06 | legacy formula (legacy_centre, legacy coefficients, stored template) |  | v2 / 0.85 % (zs) / no | +0.10 ± 0.06 (n 14950) vs A0_init, max 1.8e-02 (rebuilt) | ok | none |
| A2c_s0 | poly | 0.00e+00 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A2c_s1 | poly | 0.00e+00 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A2e_s0 | poly | 6.75e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A2e_s1 | poly | 6.75e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A2n_s0 | net | 6.99e-06 | m2a_dist/A2s_s0 |  | v2 / 1.04 % (zs) / no | +0.07 ± 0.06 (n 14950) vs A2s_s0, max 2.4e-02 (rebuilt) | ok | none |
| A2n_s1 | net | 2.71e-06 | m2a_dist/A2s_s1 |  | v2 / 1.08 % (zs) / no | +0.02 ± 0.06 (n 14950) vs A2s_s1, max 1.0e-02 (rebuilt) | ok | none |
| A2n_s2 | net | 2.36e-06 | m2a_dist/A2s_s2 |  | v2 / 1.08 % (zs) / no | +0.02 ± 0.06 (n 14950) vs A2s_s2, max 1.1e-02 (rebuilt) | ok | none |
| A2s_s0 | poly | 7.20e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A2s_s0_x2 | poly | 7.20e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A2s_s1 | poly | 7.20e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A2s_s1_x2 | poly | 7.20e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A2s_s2 | poly | 7.20e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| A3c_s0 | poly | 1.39e-03 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A3f_s0 | poly | 1.51e-03 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A3x_s0 | poly | 9.13e-04 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A3x_s1 | poly | 9.16e-04 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A4_s0 | poly | 1.24e-04 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A4_s1 | poly | 1.25e-04 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A4_s1_x2 | poly | 1.25e-04 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A4n_s0 | net | 1.70e-06 | m2a_dist/A4_s0 |  | v2 / 0.19 % (logit_end) / no | +0.05 ± 0.03 (n 14950) vs A4_s0, max 8.8e-03 (rebuilt) | ok | none |
| A4t_s0 | poly | 9.82e-03 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A5_s0 | poly | 6.22e-04 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A5_s1 | poly | 6.19e-04 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A5_s1_x2 | poly | 6.19e-04 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A5h_s0 | poly | 2.65e-03 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A5n_s0 | net | 3.96e-06 | m2a_dist/A5_s0 |  | v2 / 0.23 % (sh_rdelta) / no | +0.02 ± 0.04 (n 14950) vs A5_s0, max 7.9e-03 (rebuilt) | ok | none |
| A6b_s0 | poly | 1.82e-03 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A6i_s0 | poly | 6.06e-04 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A6l_s0 | poly | 9.82e-04 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A78_s0 | poly | -2.63e-08 | m2a_dist/A2s_s0 | exactly A2s: yes |  |  |  |  |
| A78_s1 | poly | -2.12e-08 | m2a_dist/A2s_s1 | exactly A2s: yes |  |  |  |  |
| A7_s0 | poly | -2.89e-08 | m2a_dist/A2s_s0 | exactly A2s: yes |  |  |  |  |
| A7_s1 | poly | -1.17e-08 | m2a_dist/A2s_s1 | exactly A2s: yes |  |  |  |  |
| A7l_s0 | poly | 5.30e-06 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A7l_s1 | poly | 5.32e-06 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A7n_s0 | net | 1.11e-05 | m2a_dist/A7_s0 |  | v2 / 1.19 % (zs) / no | +0.00 ± 0.07 (n 14950) vs A7_s0, max 2.1e-02 (rebuilt) | ok | none |
| A7s_s0 | poly | -2.89e-08 | m2a_dist/A2s_s0 | exactly A2s: yes |  |  |  |  |
| A7s_s1 | poly | -1.17e-08 | m2a_dist/A2s_s1 | exactly A2s: yes |  |  |  |  |
| A8_s0 | poly | -4.52e-09 | m2a_dist/A2s_s0 | exactly A2s: yes |  |  |  |  |
| A8_s1 | poly | -1.75e-08 | m2a_dist/A2s_s1 | exactly A2s: yes |  |  |  |  |
| A8c_s0 | poly | -3.29e-08 | m2a_dist/A2s_s0 | exactly A2s: yes |  |  |  |  |
| A8c_s1 | poly | 1.87e-08 | m2a_dist/A2s_s1 | exactly A2s: yes |  |  |  |  |
| A8n_s0 | net | 6.45e-06 | m2a_dist/A8_s0 |  | v2 / 1.25 % (zs) / no | +0.02 ± 0.06 (n 14950) vs A8_s0, max 1.3e-02 (rebuilt) | ok | none |
| A9_s0 | poly | 2.15e-04 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A9_s1 | poly | 2.15e-04 | m2a_dist/A2s_s1 |  |  |  |  |  |
| A9n_s0 | net | 1.42e-05 | m2a_dist/A9_s0 |  | v2 / 1.42 % (zs) / no | +0.06 ± 0.09 (n 14950) vs A9_s0, max 3.6e-02 (rebuilt) | ok | none |
| A9r_s0 | poly | 7.93e-04 | m2a_dist/A2s_s0 |  |  |  |  |  |
| A9r_s1 | poly | 7.94e-04 | m2a_dist/A2s_s1 |  |  |  |  |  |
| plr_A2n_lr0.001 | net | 6.99e-06 | m2a_dist/A2s_s0 |  | v2 / 1.04 % (zs) / no | +0.07 ± 0.06 (n 14950) vs A2s_s0, max 2.4e-02 (rebuilt) | ok | none |
| plr_A2n_lr0.003 | net | 6.99e-06 | m2a_dist/A2s_s0 |  | v2 / 1.04 % (zs) / no | +0.07 ± 0.06 (n 14950) vs A2s_s0, max 2.4e-02 (rebuilt) | ok | none |
| plr_A2n_lr0.01 | net | 6.99e-06 | m2a_dist/A2s_s0 |  | v2 / 1.04 % (zs) / no | +0.07 ± 0.06 (n 14950) vs A2s_s0, max 2.4e-02 (rebuilt) | ok | none |
| plr_A2s_lr0.003 | poly | 7.20e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| plr_A2s_lr0.01 | poly | 7.20e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |
| plr_A2s_lr0.03 | poly | 7.20e-03 | legacy formula (legacy_centre, legacy coefficients, stored template) |  |  |  |  |  |

Nearly-constant target columns in a net warm start: none.


## Kept and replicated (the input of Stage 2; no combination formed here)

- primaries: A4: no, A5: no, A7: no, A8: no, A9: yes, A10: no
- exploratory: A4t: no, A5h: no, A6i: no, A6l: no, A6b: no, A7s: no, A7l: no, A8c: no, A78: no, A9r: yes, A3f: no, A3c: no, A3x: no, A10-2: no, A10-4: no, A10-5: no, A10g: no, A12: no
- *alternative - decides nothing* (literal conditions AND two-seed mean <= -2 SE): A4: no, A5: no, A7: no, A8: no, A9: yes, A10: no, A4t: no, A5h: no, A6i: no, A6l: no, A6b: no, A7s: no, A7l: no, A8c: no, A78: no, A9r: yes, A3f: no, A3c: no, A3x: no, A10-2: no, A10-4: no, A10-5: no, A10g: no, A12: no

