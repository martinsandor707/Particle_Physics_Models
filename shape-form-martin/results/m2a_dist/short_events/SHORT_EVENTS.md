# Short-event diagnostic: where the +21 of A9 against A2c sits, and whether it is a defect

This is a diagnostic on existing results of the machine of record (H200, launch grid 1056), as copied to
`results/machines/h200-g1056-97b197a7/`. Nothing was trained, registered or written outside this directory. The
laptop GPU was not used: every recomputation ran on the CPU with CUDA hidden. Only the validation split was used; the
test cache was never opened.

Units: 1e-4 nat per event. "a ± b (n N)" means the mean of the per-event difference ± its event-paired SE over N
events. Differences between two registered arrays go through `sfm.provenance.paired_delta`, which checks the tags.
Recomputed parts go through `sfm.stats.paired`.

- **Short events:** the 166 validation events with zb < 1221 mm. The mask is `~(zb >= registry.ZB_STRATUM_MM)`, as in
  the row "zb<1221 (short events)" of `run_m2a_report.guard_block`.
- **Others:** the other 14,784 events.

Scripts in this directory:

| script | what it does |
|---|---|
| `se_common.py` | Read-only loaders. They check the tag sidecars and the registry row's SHA-256. |
| `q_stored.py` | Questions 1, 2, 3a and the stored part of question 4, from the registered K = 64 arrays. |
| `decomp.py` | Per-hit recomputation on the H200's fold points (emulated stream, see 3b). |
| `q3b.py` | Questions 3b, 3c and 5. |
| `q_deepwide.py` | Deep-wide hit term and the share of each component. |
| `q4_params.py`, `q4_region.py` | Parameter moves, component weights, training histories, regional masses. |

JSON tables: `q_stored.json`, `q3b.json`, `q_deepwide.json`, `q4_params.json`, `q4_region.json`. The per-hit arrays
(npz, about 0.3 GB) stayed in the scratch folder.

## Summary

- **No defect found.** The checks are listed in section 5.
- **The shift is real, not noise.** The +21 reproduces exactly and does not depend on K (K = 1024 gives the same).
  The two independent seed pairs agree: +20.18 and +21.80. Their per-event correlation is 0.999, and the
  difference between the seeds is +1.62 ± 0.29.
- **Which term and which region carry it:**
  - The whole +21 is in the **hit term**: +23.35 ± 7.70 (seed 0) and +24.93 ± 7.73 (seed 1).
  - The **normalisation term** is -3.17 ± 1.49, which favours A9.
  - Inside the hit term, the shift sits in **deep, wide hits**: hit centre at z' >= 300 mm and r >= 200 mm.
    These hits contribute +25.4 ± 6.2 (seed 0) and +27.1 ± 6.2 (seed 1). All other hits contribute -2.0 ± 4.5.
- **Nothing in it is specific to the halo.** A9 behaves like the null rungs A7, A7s and A8 on the same events:
  - A9 - A7 = -2.4 ± 6.8 (seed 0) and +0.2 ± 6.8 (seed 1).
  - A9 - A7s = -2.1 ± 6.9 and -0.1 ± 6.9.
  - A9 - A8 = -4.0 ± 14.2 and -3.5 ± 14.2.

  Every change of structure started from A2s shows the same shift against A2c, and so does the network column.
- **A2c's advantage reproduces but is not a stable reference.**
  - It reproduces across seeds: -12.14 and -11.55, difference between seeds +0.59 ± 0.24.
  - The stratum mean of one model moves by 7.3 units from seed to seed.
  - It moves by 13 units across structures that are null elsewhere.
  - It moves by up to 44 units for the A2s structure itself when trained with a doubled schedule.

  The short-event stratum measures the model's deep, wide tail. The other 14,784 events barely constrain that tail.

## 1. The finding, reproduced from the stored arrays

| pair | short events | other events | all events | short: median | short: worse by > 1 |
|---|---|---|---|---|---|
| A9_s0 - A2c_s0 | **+20.18 ± 7.73** (n 166) | -4.78 ± 0.89 (n 14784) | -4.51 ± 0.88 | +14.2 | 106 |
| A9_s1 - A2c_s1 | **+21.80 ± 7.77** | -4.82 ± 0.90 | -4.52 ± 0.89 | +17.5 | 107 |
| A9_s0 - A2s_s0 | +8.04 ± 7.24 | -4.81 ± 0.89 | -4.67 ± 0.88 | +5.1 | 91 |
| A9_s1 - A2s_s1 | +10.25 ± 7.31 | -4.82 ± 0.90 | -4.66 ± 0.89 | +7.5 | 93 |
| A2c_s0 - A2s_s0 | -12.14 ± 2.40 | -0.03 ± 0.12 | -0.16 ± 0.12 | -7.4 | 53 |
| A2c_s1 - A2s_s1 | -11.55 ± 2.33 | -0.01 ± 0.11 | -0.13 ± 0.12 | -7.1 | 52 |
| A7_s0 - A2c_s0 | +22.62 ± 7.13 | -0.60 ± 0.24 | -0.35 ± 0.25 | +5.9 | 92 |
| A7_s1 - A2c_s1 | +21.59 ± 7.42 | -0.57 ± 0.25 | -0.32 ± 0.26 | +4.5 | 90 |
| A8_s0 - A2c_s0 | +24.20 ± 10.43 | -0.17 ± 0.26 | +0.10 ± 0.28 | -1.7 | 74 |
| A8_s1 - A2c_s1 | +25.33 ± 10.50 | -0.19 ± 0.26 | +0.09 ± 0.28 | -2.5 | 76 |
| A7s_s0 - A2c_s0 | +22.32 ± 7.40 | -0.61 ± 0.24 | -0.36 ± 0.25 | +5.5 | 92 |
| A7s_s1 - A2c_s1 | +21.93 ± 7.59 | -0.59 ± 0.24 | -0.34 ± 0.25 | +5.3 | 91 |
| A9r_s0 - A9_s0 | +83.86 ± 62.94 | -0.91 ± 0.35 | +0.03 ± 0.78 | -1.3 | 74 |
| A9r_s1 - A9_s1 | +83.37 ± 62.52 | -0.92 ± 0.34 | +0.02 ± 0.77 | -0.8 | 75 |
| A9r_s0 - A2c_s0 | +104.04 ± 68.00 | -5.69 ± 1.00 | -4.47 ± 1.25 | +9.0 | 92 |
| A9r_s1 - A2c_s1 | +105.17 ± 67.59 | -5.74 ± 1.01 | -4.51 ± 1.25 | +9.3 | 93 |

The review numbers reproduce exactly.

- On the other events A9 gains the same 4.8 units against A2c as against A2s. A2c and A2s differ by only -0.03 and
  -0.01 there, so the choice of control matters only on the short events.
- The shift does not depend on K. On the short events the K = 1024 guard arrays give the same numbers as K = 64:
  - A9 - A2s: K = 64 +8.04 ± 7.24, K = 1024 +7.95 ± 7.25 (seed 0); K = 64 +10.25, K = 1024 +10.19 (seed 1).
  - A9 - A7: K = 1024 -2.64 ± 6.81 (seed 0) and +0.03 ± 6.86 (seed 1).
  - A2c has no K = 1024 array.

  So the shift is not fold noise or a Jensen effect.

## 2. Is A2c's advantage on the short events stable?

### 2.1 Seed-to-seed differences of nominally identical runs

All rows are later seed minus earlier seed.

| pair | short events | other events |
|---|---|---|
| A2s_s1 - A2s_s0 | -11.21 ± 2.32 | +0.03 ± 0.21 |
| A2s_s2 - A2s_s0 | +6.78 ± 3.82 | +0.05 ± 0.31 |
| A2s_s2 - A2s_s1 | +17.99 ± 2.93 | +0.02 ± 0.23 |
| A2c_s1 - A2c_s0 | -10.62 ± 2.39 | +0.05 ± 0.21 |
| A0_s1 - A0_s0 | -8.83 ± 2.89 | +0.11 ± 0.24 |
| A9_s1 - A9_s0 | -9.00 ± 2.47 | +0.02 ± 0.21 |
| A7_s1 - A7_s0 | -11.65 ± 2.36 | +0.08 ± 0.21 |
| A8_s1 - A8_s0 | -9.49 ± 2.37 | +0.03 ± 0.21 |
| A9r_s1 - A9r_s0 | -9.49 ± 2.48 | +0.00 ± 0.21 |
| A7s / A7l / A8c / A78 / A12 / A3x (s1 - s0) | -11.00 / -10.77 / -10.49 / -9.29 / -10.74 / -10.83 (SE 2.4 each) | +0.04 to +0.07 |
| A2e / A4 / A5 / A10 (s1 - s0) | -10.63 / -12.27 / -13.25 / -11.77 | -0.01 to +0.22 |
| A2n_s1 - A2n_s0 / A2n_s2 - A2n_s0 / A2n_s2 - A2n_s1 | +2.93 ± 5.11 / +8.40 ± 5.56 / +5.47 ± 3.76 | -0.17 / +0.07 / +0.25 |
| A1_s1 - A1_s0 | +0.41 ± 4.19 | -0.37 ± 0.29 |

**Run-to-run standard deviation of a stratum mean.** Pooled over the groups (21 degrees of freedom), it is
**7.3 units on the short events** and 0.08 on the others. The rungs started from A2s alone give 7.3 (11 degrees of
freedom).

This spread is almost entirely a **seed offset shared by every polynomial structure**. Seed 1 is 9 to 13 units
better than seed 0 on the short events for every rung, from A0 to A10. The offset comes from the training seed, not
from the structure. In the decomposition it is about -4.4 in the normalisation term and -4.6 to -6.8 in the hit term:

| pair | normalisation term | hit term |
|---|---|---|
| A2s_s1 - A2s_s0 | -4.44 ± 0.61 | -6.77 ± 2.39 |
| A2c_s1 - A2c_s0 | -4.49 ± 0.64 | -6.13 ± 2.48 |
| A9_s1 - A9_s0 | -4.45 ± 0.62 | -4.56 ± 2.55 |

### 2.2 The comparisons themselves, across the two independent seed pairs

A9_s0 - A2c_s0 uses other runs than A9_s1 - A2c_s1.

| difference | seed 0 | seed 1 | s1 - s0 (paired) | per-event correlation s0 vs s1 |
|---|---|---|---|---|
| A9 - A2c | +20.18 | +21.80 | +1.62 ± 0.29 | 0.999 |
| A2c - A2s | -12.14 | -11.55 | +0.59 ± 0.24 | 0.995 |
| A9 - A2s | +8.04 | +10.25 | +2.21 ± 0.39 | 0.999 |
| A7 - A2c | +22.62 | +21.59 | -1.04 ± 0.43 | 0.999 |
| A8 - A2c | +24.20 | +25.33 | +1.13 ± 0.25 | 1.000 |
| A7s - A2c | +22.32 | +21.93 | -0.39 ± 0.31 | 0.999 |
| A9 - A7 | -2.44 | +0.21 | +2.65 ± 0.55 | 0.997 |
| A12 - A2c | -131.21 | -131.33 | -0.13 ± 0.37 | 1.000 |

**Sign and size agree across the independent seed pairs, event by event.** The seed offset cancels in every
same-seed comparison. So the +21 is not seed noise: it is a reproducible consequence of the structure and the
schedule.

### 2.3 How many run-to-run standard deviations is +21?

The answer depends on the yardstick:

| yardstick | standard deviation | +21 in units of it |
|---|---|---|
| Difference of two independently trained runs (√2 × 7.3) | 10.3 | about **2.0** (1.96 and 2.12) |
| Seed-to-seed spread of the A9 - A2c comparison itself (from s1 - s0 = 1.6) | about 1.1 | about 19 |
| Null structures started from the same A2s and trained on with the same schedule (A2c 0, A8c -2.5, A7l +1.0, A7 +22.6, A7s +22.3, A8 +24.2 against A2c) | 13.0 | 1.6. A9 sits inside this spread, next to A7, A7s and A8. |
| A2s-structure runs (A2s_s0/s1/s2, A2c_s0/s1, A2s_s0_x2, A2s_s1_x2); their mean on the others agrees within 0.7 | 16.5 | 1.3 |

The most telling single number in the last yardstick is the doubled-schedule rerun. **A2s_s0_x2 - A2s_s0 = -44.3 ± 5.6
on the short events but +0.70 ± 0.22 on the others**, and A2s_s0_x2 - A2c_s0 = -32.2 ± 4.1. For seed 1 the doubled
schedule changes nothing: A2s_s1_x2 - A2s_s1 = -0.7 ± 4.1.

The rungs that start from A2s fall into two clusters on the short events (against A2c):

| cluster | rungs | short events against A2c |
|---|---|---|
| Near A2c | A2c, A8c, A7l | -2.5 to +1 |
| Shifted | A7, A7s, A8, A9, A4 | +20 to +25 |
| Outliers | A12 | -131 |
|  | A9r | +104 |
|  | A78 | +66 |
|  | A3x | +47 |

### 2.4 Bootstrap by event and influence (short events)

| difference | bootstrap SD | 95 % interval | share of resamples <= 0 | without top 1 / 3 / 5 (largest positive) | without top 1 / 3 / 5 (largest absolute) |
|---|---|---|---|---|---|
| A9_s0 - A2c_s0 | 7.70 | [+5.5, +36.0] | 0.003 | +15.41 ± 6.12 / +13.20 ± 5.99 / +11.24 ± 5.90 | +15.41 ± 6.12 / +17.33 ± 5.20 / +17.57 ± 5.02 |
| A9_s1 - A2c_s1 | 7.81 | [+7.0, +37.7] | 0.002 | +17.04 ± 6.18 / +14.80 ± 6.04 / +12.84 ± 5.96 | +17.04 ± 6.18 / +18.96 ± 5.26 / +19.19 ± 5.07 |
| A2c_s0 - A2s_s0 | 2.38 | [-16.9, -7.5] | 1.000 | -12.65 / -13.37 / -14.07 | -11.33 ± 2.27 / -10.27 ± 2.17 / -9.23 ± 2.07 |
| A9r_s0 - A9_s0 | 62.4 | [+6.9, +220.8] | 0.005 | +21.80 ± 10.55 / +12.16 ± 8.17 / +6.26 ± 7.10 | (same) |

A9_s0 - A2c_s0 has median +14.2, 10 %-trimmed mean +16.7, and 109 of 166 events positive.

Largest contributions to A9_s0 - A2c_s0:

| validation index | per-event difference | share of sum | p (GeV) | zb (mm) | theta | visible energy | hits |
|---|---|---|---|---|---|---|---|
| 7247 | +807 | 24 % | 0.55 | 1059 | 4.6° | 0.015 | 12 |
| 480 | +212 | 6 % | 10.7 | 1182 | 4.2° | 0.011 | 50 |
| 10959 | +180 | 5 % | 10.2 | 1164 | 5.7° | 0.031 | 49 |
| 4030 | +173 | 5 % | 0.32 | 1203 | 4.5° | 0.0003 | 7 |
| 9694 | +169 | 5 % | 16.6 | 1103 | 6.0° | 0.23 | 91 |
| 13231 | +164 | 5 % | 8.4 | 1104 | 6.5° | 0.17 | 113 |

Event 7247 is the same event that dominates A9r - A9: +10,324, or 74 % of that sum. Almost all of its energy (98 %)
is in two hits at z' ≈ 942 mm, r ≈ 490-510 mm, where ln g ≈ -27.

- A9 lowers ln g there by 0.08 nat against A2c.
- A7 lowers it by 0.035.
- A9r, which has no far halo, lowers it by 1.05.

The shift is broad, not driven by a few events. Without its top 5 events it is still +11 to +13 (about 2 SE). Without
the 5 largest |d| it is +17.6 ± 5.0 and +19.2 ± 5.1.

**Stability verdict (question 2).**

- **What reproduces.** A2c's short-event advantage over A2s (-12) is reproducible across the two seeds, with the same
  per-event pattern. So is A9's +21 against A2c.
- **What is not stable.** The short-event stratum mean is not a stable property of a model structure:
  - it moves by 7.3 units from seed to seed;
  - it moves by 13 units across structures that are null on the other events;
  - it moves by up to 44 units for the base structure with a longer schedule;
  - meanwhile the other events move by less than one unit.
- **Where +21 sits on that scale.** About two run-to-run SDs for independent runs, and inside the spread of the null
  rungs.
- **What A2c is.** A2c is not a stable reference level on this stratum. It is one end of a reproducible but
  structure- and schedule-dependent spread.

## 3. Where does it sit?

### 3a. Stratified differences from the stored arrays

What the short events look like, compared with the others:

| quantity | short events | other events |
|---|---|---|
| p, median (quartiles) | 10.4 GeV (4.5-14.9) | 10.0 GeV |
| theta, median (quartiles) | 5.9° (4.7°-8.3°) | 25.4° |
| zb, median (quartiles) | 1144 mm (894-1199); minimum 10 mm | — |
| hits, median (quartiles) | 70 (18-198) | 260 |
| visible energy, median (quartiles) | 0.048 (0.009-0.24) | — |

The short events are nearly head-on (small theta) late shower starts.

Differences by stratum, short events (n in each cell):

| stratum | A9_s0 - A2c_s0 | A9_s1 - A2c_s1 | A2c_s0 - A2s_s0 | A9_s0 - A2s_s0 | A7_s0 - A2c_s0 | A8_s0 - A2c_s0 |
|---|---|---|---|---|---|---|
| p < 2 GeV (24) | +16.0 ± 43.4 | +16.1 ± 43.4 | -2.9 ± 7.8 | +13.1 ± 39.3 | +3.2 ± 25.2 | -10.2 ± 12.7 |
| 2-5 GeV (22) | +7.5 ± 13.9 | +8.7 ± 14.2 | -11.8 ± 4.6 | -4.4 ± 14.0 | -20.1 ± 19.1 | -3.1 ± 12.6 |
| 5-10 GeV (33) | +28.4 ± 9.7 | +29.9 ± 10.0 | -10.1 ± 4.3 | +18.3 ± 8.0 | -0.8 ± 12.5 | -14.4 ± 12.0 |
| 10-21 GeV (87) | +21.4 ± 7.3 | +23.6 ± 7.4 | -15.6 ± 3.5 | +5.9 ± 7.5 | +47.6 ± 8.9 | +55.2 ± 18.2 |
| zb < 600 mm (22) | **-27.1 ± 9.5** | -26.1 ± 9.7 | -0.1 ± 3.8 | -27.2 ± 10.5 | +12.1 ± 12.7 | -12.4 ± 4.0 |
| zb 600-1000 (36) | +16.1 ± 13.1 | +17.7 ± 13.2 | -12.7 ± 6.4 | +3.3 ± 13.7 | +23.9 ± 12.1 | +46.6 ± 29.1 |
| zb 1000-1150 (31) | +50.1 ± 27.3 | +53.5 ± 27.1 | -24.2 ± 7.4 | +25.8 ± 23.5 | +49.3 ± 22.7 | +34.6 ± 36.7 |
| zb 1150-1221 (77) | +23.6 ± 10.2 | +24.7 ± 10.3 | -10.5 ± 2.7 | +13.1 ± 10.1 | +14.3 ± 10.3 | +20.0 ± 10.2 |
| theta < 15° (159) | +21.2 ± 8.0 | +22.9 ± 8.1 | -12.7 ± 2.5 | +8.5 ± 7.5 | +24.9 ± 7.2 | +25.5 ± 10.9 |
| theta 15-30° (5) / > 30° (2) | -17.4 ± 22.6 / +30.4 ± 19.9 | | | | | |
| hits < 20 (44) | +8.9 ± 24.6 | +9.9 ± 24.6 | -9.0 ± 5.3 | -0.1 ± 22.1 | +15.6 ± 16.5 | -8.1 ± 11.2 |
| hits 20-60 (34) | +21.3 ± 12.5 | +23.2 ± 12.5 | -8.0 ± 5.7 | +13.3 ± 13.4 | +19.3 ± 16.6 | +23.9 ± 21.4 |
| hits >= 60 (88) | +25.4 ± 6.4 | +27.2 ± 6.5 | -15.3 ± 2.9 | +10.1 ± 6.3 | +27.4 ± 8.6 | +40.4 ± 16.8 |

For comparison, on the others in the same momentum bins A9 - A2c is -3.2, -3.1, -7.1 and -4.4. The shift sits in the
moderately short events (zb 600-1221 mm), at p >= 5 GeV, and in events with many hits. On the shortest events
(zb < 600 mm) A9 is **better** than A2c by 27 ± 9.5, mostly through the normalisation term (section 3b). Events with
one to three hits are not the source: the 13 events with at most 3 hits contribute -4 % of the sum, with mean -11.

### 3b. Per-hit decomposition in (z', r)

**Method.** The runs were rebuilt read-only on the CPU:

- `sfm.models.load_run`, with a `BuildContext` pointing at the copied machine directory (`skip_init`);
- the Track A templates read from that directory (`dist_factory.set_template_dir`);
- `sfm.metric.event_nll_multi(return_parts=True)` on all 14,950 validation events;
- the fold points drawn by `sfm.refstream.patched_draw(1056)`, the emulated CUDA stream of the H200 (seed 12345,
  batch 128, K = 64).

The full pass for 14 runs took 4 minutes.

**Agreement with the registered val_nll.npy** (a tolerance check, not a delta):

| check | result |
|---|---|
| rms per event, each of the 14 runs | 4.6e-6 |
| mean difference | at most 1.1e-7 |
| events above 1e-4 | exactly one (event 9207; not a short event; 463 hits; NLL 20.58) |
| largest difference, at event 9207 | 1.35e-4 to 1.60e-4 |
| paired difference A9_s0 - A2c_s0 at event 9207 | agrees with the registered one to 5.7e-6 |
| all 166 short events | within 2.1e-5 |
| decomposition identity: weight_sum × ln M - Σ hit = nll | holds to 2.9e-5 |

At event 9207 the deviation is the same in all 14 runs, so it is a laptop-CPU against H200-GPU arithmetic difference
on one hit, not a model difference.

**Hit term and normalisation term** (short events, recomputed; registered totals in brackets):

| comparison | total | normalisation term | hit term | deep-wide hits (z' >= 300, r >= 200) | other hits |
|---|---|---|---|---|---|
| A9_s0 - A2c_s0 | +20.19 (+20.18) | **-3.17 ± 1.49** | **+23.35 ± 7.70** | **+25.36 ± 6.17** | -2.01 ± 4.52 |
| A9_s1 - A2c_s1 | +21.79 (+21.80) | -3.13 ± 1.49 | +24.93 ± 7.73 | +27.06 ± 6.21 | -2.13 ± 4.53 |
| A9_s0 - A2s_s0 | +8.04 | -2.80 ± 1.49 | +10.84 ± 7.20 | | |
| A2c_s0 - A2s_s0 | -12.15 | +0.37 ± 0.18 | -12.51 ± 2.43 | -11.10 ± 2.27 (A2s - A2c: +11.10) | -1.42 ± 0.98 |
| A7_s0 - A2c_s0 | +22.63 | -0.05 ± 0.45 | +22.67 ± 7.11 | +29.64 ± 5.54 | -6.96 ± 4.04 |
| A7s_s0 - A2c_s0 | +22.32 | +0.00 ± 0.38 | +22.31 ± 7.39 | +30.02 ± 5.64 | -7.71 ± 4.31 |
| A8_s0 - A2c_s0 | +24.20 | -0.36 ± 0.15 | +24.55 ± 10.44 | +23.49 ± 10.28 | +1.07 ± 2.02 |
| A9_s0 - A7_s0 | -2.44 | -3.12 ± 1.31 | +0.68 ± 6.76 | -4.28 ± 5.51 | +4.95 ± 3.73 |
| A9_s0 - A7s_s0 | -2.13 | -3.17 ± 1.32 | +1.04 ± 6.88 | -4.66 ± 5.60 | +5.70 ± 3.78 |
| A12_s0 - A2c_s0 | -131.21 | +0.40 ± 0.08 | -131.62 ± 50.83 | -101.31 ± 48.53 | -30.30 ± 15.40 |
| A9r_s0 - A9_s0 | +83.86 | +1.32 ± 0.38 | +82.54 ± 62.92 | | |
| A2s_s0_x2 - A2c_s0 | -32.17 | -1.65 ± 0.21 | -30.52 ± 4.15 | -34.60 ± 3.39 | +4.08 ± 1.67 |

On the others, A9_s0 - A2c_s0 = normalisation -0.14 ± 0.00, hit -4.64 ± 0.89; deep-wide hits there give
+0.33 ± 0.13.

**Hit term by region of the hit centre** (short events). z' regions are taken relative to each event's zb:

- front: z' < 0;
- entrance: 0 <= z' < 150 mm;
- last 200 mm: zb - 200 <= z' < zb;
- beyond zb: z' >= zb;
- bulk: the rest.

The last ring starts at r = 663 mm.

| region | short: energy share | A9_s0 - A2c_s0 | A9_s1 - A2c_s1 | A2c_s0 - A2s_s0 | A7_s0 - A2c_s0 | A9_s0 - A7_s0 | others: A9_s0 - A2c_s0 |
|---|---|---|---|---|---|---|---|
| front z' < 0 | 0.014 | -3.23 ± 3.05 | -3.24 ± 3.05 | -0.13 ± 0.15 | -2.66 ± 2.77 | -0.57 ± 0.36 | -3.41 ± 0.71 |
| entrance 0-150 mm | 0.168 | -0.04 ± 1.99 | -0.07 ± 2.00 | -0.08 ± 0.49 | -0.43 ± 1.64 | +0.39 ± 1.79 | +9.00 ± 0.28 |
| bulk | 0.551 | **+11.40 ± 3.72** | +12.79 ± 3.78 | -7.73 ± 1.50 | +11.32 ± 4.58 | +0.09 ± 4.95 | -9.30 ± 0.38 |
| last 200 mm before zb | 0.233 (others 0.016) | **+10.44 ± 5.44** | +10.52 ± 5.44 | -3.13 ± 1.44 | +9.03 ± 2.72 | +1.41 ± 3.52 | -0.82 ± 0.06 |
| beyond zb | 0.034 (others 0.001) | +4.78 ± 1.52 | +4.92 ± 1.56 | -1.43 ± 0.80 | +5.42 ± 1.86 | -0.64 ± 1.05 | -0.11 ± 0.01 |
| core r < 30 | 0.228 | -0.41 ± 1.15 | -0.55 ± 1.16 | -0.69 ± 0.60 | -0.50 ± 0.73 | +0.09 ± 0.98 | -2.08 ± 0.11 |
| 30-200 mm | 0.359 | +6.53 ± 2.70 | +6.42 ± 2.71 | -0.86 ± 0.61 | +2.96 ± 1.36 | +3.57 ± 2.70 | +7.09 ± 0.43 |
| 200-663 mm | 0.343 (others 0.117) | **+13.00 ± 5.81** | +14.47 ± 5.82 | -7.05 ± 1.91 | +11.56 ± 5.21 | +1.44 ± 5.24 | -9.33 ± 0.73 |
| last ring | 0.070 (others 0.002) | +4.24 ± 4.09 | +4.58 ± 4.11 | -3.91 ± 1.23 | +8.66 ± 4.02 | -4.42 ± 2.30 | -0.32 ± 0.12 |

In absolute z' (A9_s0 - A2c_s0, short events):

| z' range (mm) | contribution |
|---|---|
| -200 to 0 | -3.2 ± 3.1 |
| 0 to 100 | +5.5 ± 1.9 |
| 100 to 300 | -14.0 ± 2.6 |
| 300 to 600 | +10.3 ± 2.0 |
| 600 to 900 | +9.7 ± 2.5 |
| 900 to 1100 | +10.7 ± 5.1 |
| 1100 to 1221 | +4.3 ± 1.6 |
| 1221 to 1450 | +0.1 ± 0.1 |

The pattern from 0 to 600 mm (+5.5, -14.0, +10.3) is A9's general redistribution: the same signs and sizes appear on
the others (+17.1, -28.4, +10.2). The part specific to the short events is z' >= 600 mm.

Combined (z' region × r region), A9_s0 - A2c_s0 on the short events. Cells above 0.5 in absolute value:

| region | contribution |
|---|---|
| bulk × 30-200 | +5.5 ± 1.7 |
| bulk × 200-663 | +3.7 ± 2.2 |
| bulk × last ring | +2.9 ± 1.9 |
| last 200 mm × 200-663 | +7.1 ± 5.1 |
| last 200 mm × last ring | +3.4 ± 1.5 |
| beyond zb × 200-663 | +3.9 ± 1.4 |
| beyond zb × last ring | +0.9 ± 0.5 |
| front × last ring | -3.0 ± 3.0 |
| entrance × 200-663 | -1.3 ± 0.7 |

**By zb.** The normalisation term and the hit term per event, A9_s0 - A2c_s0:

| zb range | n | normalisation | hit |
|---|---|---|---|
| zb < 600 | 22 | -18.3 | -8.8 |
| 600-1000 | 36 | -1.2 | +17.3 |
| 1000-1150 | 31 | -1.1 | +51.2 |
| 1150-1221 | 77 | -0.6 | +24.2 |

A9's normalisation advantage comes almost entirely from the shortest events.

### 3c. Model masses and depth profiles

Mean mass in front of zb, M_e, recomputed. The guarded runs' stored `inside_zb` monitors agree with this to at most
1.3e-6 relative.

| run | short: mean M_e | short: mean ln M_e | short: min M_e | others: mean M_e | others: mean ln M_e |
|---|---|---|---|---|---|
| A9_s0 | 0.89857 | -0.19506 | 0.0267 | 0.997632 | -0.002375 |
| A2c_s0 | 0.89863 | -0.19475 | 0.0268 | 0.997646 | -0.002361 |
| A2s_s0 | 0.89859 | -0.19478 | 0.0268 | 0.997632 | -0.002374 |
| A7_s0 | 0.89866 | -0.19475 | 0.0269 | 0.997654 | -0.002352 |
| A9_s1 | 0.89833 | -0.19551 | 0.0266 | 0.997609 | -0.002398 |
| A2c_s1 | 0.89839 | -0.19519 | 0.0266 | 0.997623 | -0.002383 |
| A8_s0 / A7s_s0 / A12_s0 / A9r_s0 | 0.89858 / 0.89866 / 0.89868 / 0.89873 | -0.19478 / -0.19474 / -0.19470 / -0.19493 | 0.027 | 0.99763 / 0.99766 / 0.99766 / 0.99768 | |

The short events have on average 10.1 % of the model mass beyond zb, against 0.24 % for the others (A9_s0's mass
monitor). The masses of the models differ by a few 1e-4 relative. That is why the normalisation term is small, and it
favours A9.

Depth profile around the short events' zb: mean model mass in windows relative to each event's zb, at that event's
momentum. Relative difference from A2c:

| window | A9_s0 | A7_s0 | A2c_s0 against A2s_s0 |
|---|---|---|---|
| zb - 600 to zb - 400 | -0.14 % | +0.02 % | +0.04 % |
| zb - 400 to zb - 200 | -0.14 % | -0.03 % | +0.01 % |
| zb - 200 to zb - 100 | +0.08 % | -0.05 % | -0.02 % |
| zb - 100 to zb | +0.11 % | -0.06 % | -0.05 % |
| zb to zb + 100 | +0.12 % | -0.06 % | -0.07 % |
| zb + 100 to zb + 300 | +0.19 % | +0.02 % | -0.04 % |

The longitudinal profile is essentially the same for all models, to 0.2 %. The shift is not a depth-profile or
normalisation effect. It is a lateral-tail effect at depth.

Model mass in the regions that carry the hit term, at the short events' momenta, relative to A2c of the same seed
(`q4_region.py`):

| run | deep-wide (z' 300-1221, r >= 200) | deep last ring (z' 300-1221) | short events against A2c (stored) |
|---|---|---|---|
| A12_s0 | -0.15 % | **+44.5 %** | -131 |
| A2s_s0_x2 | +0.81 % | +1.0 % | -32 |
| A8c_s0 / A7l_s0 | -0.04 % | +0.01 % / +0.20 % | -2.5 / +1.0 |
| A2c_s0 | 0 | 0 | 0 |
| A2s_s0 | +0.07 % | -0.32 % | +12.1 |
| A9_s0 | -0.04 % | **-1.30 %** | +20.2 |
| A8_s0 | +0.30 % | -1.03 % | +24.2 |
| A7_s0 / A7s_s0 | +0.29 % / +0.30 % | **-1.65 % / -1.62 %** | +22.6 / +22.3 |
| A9r_s0 | +0.19 % | **-5.6 %** | +104 |

The ordering on the short events follows the deep last-ring and wide-tail density, almost monotonically, across all
these runs. Seed 1 gives the same pattern, within 0.3 percentage points.

**Share of each component in the deep-wide region** (short events' momenta, after the acceptance; `q_deepwide.py`):

| run | shower | far halo | detector halo | track | end blob |
|---|---|---|---|---|---|
| A2c_s0 | 89.2 % | 8.7 % | — | 1.6 % | 0.5 % |
| A7_s0 | 90.5 % | 7.5 % | — | 1.6 % | 0.5 % |
| A9_s0 | 88.2 % | **6.1 %** | 3.8 % | 1.6 % | 0.3 % |
| A9r_s0 | 91.1 % | — | 7.2 % | 1.5 % | 0.2 % |

In A9 the detector halo takes over part of the far halo's weight:

- The far-halo weight falls from 0.0031-0.0096 (A2c) to 0.0025-0.0075 over momentum.
- The detector halo has weight 0.013 and decays with detector depth, with L ≈ 172 mm. It therefore sits at z' ≲ 500 mm
  and cannot replace the far halo at the depth of the short events' deep hits.

**Where the +21 sits.**

- **The term:** all of it is in the **hit term**. The normalisation term goes the other way, -3.2 ± 1.5 (seed 0).
- **Depth:** the hits of the short events that lie **deep**: in the bulk (+11.4 ± 3.7), in the last 200 mm in front
  of zb (+10.4 ± 5.4) and beyond zb (+4.8 ± 1.5).
- **Radius:** **wide**, mostly r = 200-663 mm (+13.0 ± 5.8), then 30-200 mm (+6.5 ± 2.7) and the last ring
  (+4.2 ± 4.1).
- **Combined:** hits with centre z' >= 300 mm and r >= 200 mm carry +25.4 ± 6.2 (seed 0) and +27.1 ± 6.2 (seed 1).
- **What does not carry it:** the entrance region and the core carry nothing. The region in front of the origin, where
  the halo acts, favours A9 (-3.2 ± 3.1).
- **Why the short events see it:** they put 41 % of their energy at r >= 200 mm (others 12 %) and 23 % in the last
  200 mm in front of zb (others 1.6 %). They are the events that see the model's deep, wide tail.
- **A2c's advantage over A2s sits in the same region,** in A2c's favour (deep-wide hits -11.1 ± 2.3).
- **A7 and A7s show the same signature without any halo** (deep-wide +29.6 ± 5.5 and +30.0 ± 5.6).

## 4. Is anything specific to the halo?

Paired, on the short events (stored K = 64 arrays):

| pair | short events | other events |
|---|---|---|
| A9_s0 - A7_s0 / A9_s1 - A7_s1 | -2.44 ± 6.80 / +0.21 ± 6.84 | -4.18 ± 0.80 / -4.25 ± 0.80 |
| A9_s0 - A8_s0 / A9_s1 - A8_s1 | -4.02 ± 14.20 / -3.53 ± 14.23 | -4.61 ± 0.94 / -4.62 ± 0.94 |
| A9_s0 - A7s_s0 / A9_s1 - A7s_s1 | -2.14 ± 6.91 / -0.13 ± 6.92 | -4.17 ± 0.79 / -4.23 ± 0.80 |
| A7_s0 - A8_s0 / A7_s0 - A7s_s0 | -1.57 ± 11.20 / +0.31 ± 1.22 | -0.43 ± 0.38 / +0.01 ± 0.03 |
| A9n_s0 - A2n_s0 | +19.38 ± 12.37 | -5.43 ± 1.00 |
| A7n_s0 - A2n_s0 / A8n_s0 - A2n_s0 / A4n_s0 - A2n_s0 | +28.35 ± 8.98 / +48.75 ± 11.35 / +22.64 ± 13.68 | -0.20 / -0.89 / +0.30 |
| A2n_s0 - A2s_s0 | -43.31 ± 16.43 | +2.62 ± 1.14 |
| A12_s0 - A2c_s0 / A12_s1 - A2c_s1 | -131.21 ± 50.83 / -131.33 ± 51.16 | +0.47 ± 0.57 / +0.47 ± 0.58 |
| A12_s0 - A2s_s0 | -143.35 ± 51.47 | +0.44 ± 0.59 |
| A9_s0_init - A2s_s0 (A9 at step 0) / seed 1 | +6.43 ± 4.70 / +6.52 ± 4.70 | -0.66 ± 0.61 / -0.72 ± 0.61 |
| A4_s0 - A2c_s0 / A4_s1 - A2c_s1 | +21.83 ± 8.65 / +20.17 ± 8.78 | +1.71 / +1.69 |
| A2s_s0_x2 - A2c_s0 / A2s_s1_x2 - A2c_s1 | -32.17 ± 4.11 / +10.84 ± 2.63 | +0.73 / +0.07 |

**A9 behaves like the null rungs A7, A7s and A8 on the short events.** It is within 0 to 4 units of each of them,
with SE 7 to 14. In the decomposition against A7 and A7s, the only parts that are specific to the halo are both in
A9's favour:

- the normalisation term: -3.12 ± 1.31 against A7, -3.17 ± 1.32 against A7s;
- the deep-wide hits: -4.3 ± 5.5 against A7, -4.7 ± 5.6 against A7s.

The network column repeats the pattern: every rung is worse than A2n on the short events, and A9n is the least bad of
the four.

A12 is the mirror image: one free last-ring factor of exp(0.45) = 1.57 raises the deep last-ring mass by 44 %.

- On the short events that gives -131 ± 51: last-ring hits -217.5 ± 46.6, and hits at r = 200-663 mm +87.7 ± 17.3.
- On the others it is null: +0.47 ± 0.57.

This supports the reading that the short-event mean is driven by the last-ring and wide-tail density.

**The reading "training on from A2s with a changed structure", tested.**

1. **Parameter moves** (`q4_params.py`; raw provider outputs by name, averaged over the short events' momenta, each
   relative to A2s of the same seed).
   - A2c moves sloppy parameters by 0.1-0.3 raw units while the all-event NLL stays flat: zs +0.33, s_1 +0.18,
     s_2 -0.18, logit_th -0.09, s_sig -0.07.
   - The cosine of each rung's parameter move with A2c's:

     | rung | cosine with A2c's move |
     |---|---|
     | A8c | 0.99 |
     | A12 | 0.98 |
     | A7s | 0.91 |
     | A8 | 0.87 |
     | A7 | 0.84 |
     | A9 | 0.51 (0.80 with A7's move) |
     | A9r | 0.24 |

   - Parameter space does **not** separate the two clusters cleanly. A8 moves much like A2c but lands at +24.
   - The function-space quantity of 3c, the deep last-ring and wide-tail density, does separate them.
   - A9's move differs from the other rungs in zs -0.19, zt0 -0.24, log_w1 -0.12, log_sig +0.13, s_R +0.10 and
     logit_end -0.05.
   - Its far-halo weight is lower: 0.0025-0.0075 against 0.0031-0.0096 for A2c over momentum. The far-halo
     parameters themselves are not in the comparison, because A9r has no far block.
2. **Training histories.**
   - All 17 checked runs: selection "last" at step 4000 (8000 for the x2 runs), no skipped batches (0 / 0).
   - The validation curves of A2c, A7, A7s, A8, A8c, A7l, A12 and A9 move in parallel step by step; they share the
     seed and therefore the batches and fold points.
   - The curves differ by near-constant offsets. A9 - A2c (all events) grows from -2.5 at step 500 to -4.5 at step
     4000. A7 - A2c goes from +0.1 to -0.3.
   - There is no late drift on the all-event curve. A per-stratum curve cannot be formed, because only the selected
     model.pt is available, with no intermediate checkpoints.
3. **Conclusion.**
   - The shift belongs to "a change of structure started from A2s and trained with the T_A schedule", together with
     the sensitivity of the short events to the deep, wide tail. It is not the halo as such.
   - In A9's case there is a physical mechanism: the depth-limited detector halo takes over part of the far halo,
     which also served the deep, wide hits of late-starting showers.
   - That mechanism is a modelling trade-off of the same size as A7's lateral change, not a bug.
   - A9r makes the same trade-off fully (no far halo) and pays +104, carried by a single event.

## 5. Look for an actual defect

**No defect found** in the following checks:

1. **Recomputation against stored arrays.**
   - 14 runs, all 14,950 validation events, on the H200's emulated fold points.
   - rms 4.6e-6 per event, mean difference at most 1.1e-7. 14,949 events within 1e-4; one event (9207, not short) at
     1.4-1.6e-4, identical in all 14 runs (paired difference within 5.7e-6). All 166 short events within 2.1e-5.
   - Every registered short-stratum paired difference is reproduced to 0.01 units by the recomputed parts.
   - Decomposition identity: at most 2.9e-5.
2. **Non-finite values.**
   - None in 206 registered arrays (K = 64, K = 1024 and bulk arrays of every registered run).
   - None in the recomputed per-event NLL, per-hit terms or ln M_e of 14 runs.
3. **Normalisation term in (0, 1].**
   - Smallest M_e = 0.0266, for the event with zb = 10 mm. No event has M_e < 1e-3; the 1e-8 clamp is never active.
   - Largest ln M_e = +1.1e-6: M_e exceeds 1 by float32 rounding of the cumulative sum, an effect of order 1e-6 nat.
     It is not a defect.
   - weight_sum lies in [0.999999, 1.000001].
4. **Mass in front of zb monotone in zb.** The cumulative longitudinal profile is monotone for all 14 runs at every
   short event's momentum. The total mass is 1 ± 1e-6, and the stored `total` monitor of A9_s0 lies in
   [0.9999998, 1.0000005].
5. **Stored mass monitors.** `inside_zb` agrees with the recomputed M_e to at most 1.3e-6 relative, for the 11
   guarded runs that were recomputed.
6. **Event classes.**
   - **zb at a z-bin edge** (within 1e-3 of a bin width): 32 events, none of them short. The interpolation of M_e is
     continuous across bin edges.
   - **zb beyond the grid** (>= 1450 mm): 2,441 events, all tilted full-length events and none short. For these
     M_e = the total mass; this is the known flat cut, not specific to the short events.
   - **Minimum zb:** 10.3 mm.
   - **Few hits:** among the short events, 8 have one hit and 13 have at most three. Together they contribute -4 % of
     the A9 - A2c sum (A9 is better on them).
   - **Hits behind zb:** the short events have 255 hits at z' >= zb (3.4 % of their energy). They are scored with the
     density beyond the flat cut, as for every event. Their share of A9 - A2c is +4.8 ± 1.5 (seed 0); the same for A7
     (+5.4 ± 1.9). This is the documented flat-cut behaviour that guard G-mass monitors.
   - No short event has visible energy 0; the minimum is 3.4e-5.
7. **Component weights.**
   - Every weight is a log-sigmoid product, so in (0, 1) by construction. All are finite on a 0.2-20 GeV grid for
     the 20 models checked.
   - The detector-halo weight is 0.0132, constant in momentum as specified.
   - Over momentum the far-halo weight ranges 0.0025-0.0075 in A9, against 0.0031-0.0096 in A2c.
   - The end-blob weight falls to 2.6e-6 at the lowest momenta in every run, A2s included. That is inherited legacy
     behaviour, not specific to the halo.
   - No weight is negative or exploding.
8. **K dependence.** The short-event differences at K = 1024 equal those at K = 64 to within 0.3 units, for A9 - A2s,
   A7 - A2s, A8 - A2s and A9 - A7.
9. **Training.** No skipped steps, no divergence markers. Selection "last" as registered. No late drift on the
   validation curve.

Not checked, or only partly:

- The short-event NLL along the training trajectory (no intermediate checkpoints on the laptop).
- A third seed of A2c, so whether A2c's -12 holds beyond two seeds.
- The short training events (n 678) themselves.
- The recomputation is not bit-identical to the H200 (CPU arithmetic), only within the tolerances above.
- The (z', r) classification uses the hit centre, as G-region does. A hit near r = 700 mm whose fold points are
  clamped into the last ring is counted under its centre's ring.
