# Lane F1_lateral - lateral profile of the proton shower (progress notes)

All numbers: 3D slab likelihood (`shapelab.setup(batch=256, slab=True)`), validation set, K = 64 fixed seed (`C["test_nll"]`).
`d_ref` = paired difference to the reference `families.GammaCoreTail` degree 2 (21 parameters, val 18.21153, test 18.20566),
with its standard error.  Model = baseline gamma (families.long_gamma) x lateral mixture `fam.LatMix`.
Fits: `shapelab.fit_fast`, 3000 steps, seed 0, longitudinal coefficients warm-started from the reference of the same degree.

Files: `fam.py` (model classes), `lab.py` (helpers), `survey.py` (forms at equal degree), `verify.py` (re-evaluation of stored
checkpoints), `hatprobe.py` (non-parametric depth dependence), `latprof.py` (data-vs-model lateral profiles; was `profile.py`,
renamed because that name shadows the standard-library module `profile` and broke the IPython import of the harness).
Raw log of every fit: `results/log.jsonl`; checkpoints of exploratory fits: `results/explore/<name>.pt`.

## Batch 0 - survey of lateral forms, degree 2, linear depth dependence (run by the interrupted attempt, RE-VERIFIED 20:03)

`verify.py 2` rebuilt every model from its logged constructor + checkpoint and recomputed the validation NLL: all 24 values agree
with the log to 1e-5 (log: `logs/verify_n2.log`).  Widths `w` [mm] and fractions `f` quoted at z' = 200 mm.

| task | form (components) | n_par | val NLL | d_ref +- SE | fitted (p = 1 / 5 / 15 GeV) |
|---|---|---|---|---|---|
| a | cauchy + cauchy (LatMix re-implementation of the baseline) | 21 | 18.21109 | -0.00044 +- 0.00013 | w 19/173, 33/121, 28/98 |
| b | cauchy + expr (halo e^{-r/lam}/(2 pi lam r)) | 21 | 18.20732 | -0.00420 +- 0.00063 | core 27/47/47, lam 188/131/108, f_halo 0.18/0.50/0.62 |
| b | gauss + expr | 21 | 18.20711 | -0.00442 +- 0.00077 | sigma 24/35/37, lam 134/118/103, f_halo 0.41/0.77/0.83 |
| c | cauchy + exp (e^{-r/lam}/(2 pi lam^2)) | 21 | 18.21661 | +0.00508 +- 0.00042 | lam 95/84/86 |
| c | gauss + exp | 21 | 18.22590 | +0.01437 +- 0.00082 | |
| d | cauchy + K0 | 21 | 18.21419 | +0.00266 +- 0.00033 | lam 125/105/95 |
| d | gauss + K0 | 21 | 18.21910 | +0.00757 +- 0.00070 | |
| e | single Student-t (free exponent) | 16 | 18.22119 | +0.00966 +- 0.00060 | nu = 1.57/1.48/1.50 (density ~ r^-3) |
| e | cauchy + student | 24 | 18.20734 | -0.00418 +- 0.00052 | nu_tail 1.79/1.73/1.58 |
| e | student + student | 27 | 18.20587 | -0.00566 +- 0.00056 | |
| f | cauchy + cauchy + expr | 29 | 18.20380 | -0.00773 +- 0.00066 | |
| f | gauss + cauchy + expr | 29 | 18.20018 | -0.01135 +- 0.00068 | sigma 17/22/20, R 81/89/81, lam 125/102/92 |
| f | cauchy + exp + expr | 29 | 18.20296 | -0.00857 +- 0.00066 | |
| f | cauchy + cauchy + cauchy | 29 | 18.20181 | -0.00972 +- 0.00060 | |
| f | gauss + exp + expr | 29 | 18.20087 | -0.01066 +- 0.00064 | sigma 22/27/25, lam 41/40/34 and 152/166/143 |
| g | cauchy + ggamma (dE/dr ~ r^{k-1} e^{-r/lam}) | 24 | 18.20700 | -0.00453 +- 0.00060 | k = 1.55/1.12/0.99 (k = 1 is the pure exponential dE/dr) |
| g | cauchy + stretched exp (e^{-(r/w)^b}) | 24 | 18.20351 | -0.00802 +- 0.00063 | b = 0.65/0.55/0.44 |
| g | cauchy + stretched exp / r | 24 | 18.20673 | -0.00479 +- 0.00062 | b = 0.87/0.92/0.87 |
| g | expr + expr | 21 | 18.21073 | -0.00080 +- 0.00072 | |
| g | exp + exp | 21 | 18.21721 | +0.00568 +- 0.00050 | |
| g | single ggamma | 16 | 18.23344 | +0.02191 +- 0.00109 | |
| g | single expr | 13 | 18.24571 | +0.03418 +- 0.00174 | |
| g | ggamma + ggamma | 27 | 18.20621 | -0.00531 +- 0.00075 | |
| g | student + expr | 24 | 18.20340 | -0.00813 +- 0.00063 | nu_core 3.3/1.6/1.45, lam 155/105/76 |

Conclusions so far (degree 2, linear depth):
- Replacing the wide Cauchy by the exponential-dE/dr halo (b) gains 0.0042 +- 0.0006 at the same 21 parameters; with the K0 (d) or
  the e^{-r/lam}/lam^2 halo (c) the fit is WORSE than the two-Cauchy baseline: these halos are too flat at small r (they need the
  1/r rise towards the axis that `expr` has).
- The core kind (Cauchy vs Gauss) does not matter when the halo is `expr` (-0.0042 vs -0.0044).
- A free tail exponent (e) is worth the same as the expr halo but costs 3 more parameters; nu_tail ~ 1.6-1.8 (density ~ r^-3.4).
- Three components gain ~0.010-0.011 with 29 parameters (best: gauss + cauchy + expr).
