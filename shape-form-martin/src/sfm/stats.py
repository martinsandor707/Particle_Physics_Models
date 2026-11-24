"""sfm.stats - paired statistics and the campaign's decision rules (NumPy only).

Every comparison in the campaign is *paired by event*: two models are scored on the same events with the same
fold points, and the quantity of interest is the mean of the per-event difference ``d_e = a_e - b_e`` of their
NLLs (negative = ``a`` is better). All functions work in float64 and refuse non-finite input instead of
propagating it.

Contents
    paired                     mean, standard error (ddof=1) and n of ``a - b``
    one_se_select              the 1-SE rule (simplest candidate within one paired SE of the best)
    bootstrap_paired           event bootstrap of the paired mean difference, optionally stratified
    winners_curse_correction   optimism of "best of m candidates on validation"
    holm, z_from_paired, p_from_z
    track_b_promotion, track_a_kept, final_claim, guard_persistence   the plan's decision rules as functions

Sign convention throughout: NLL differences, lower is better, a *gain* is a negative number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np

__all__ = [
    "Paired", "paired", "Candidate", "OneSE", "one_se_select", "Bootstrap", "bootstrap_paired",
    "WinnersCurse", "winners_curse_correction", "holm", "z_from_paired", "p_from_z",
    "Decision", "track_b_promotion", "track_a_kept", "final_claim", "guard_persistence",
    "TRACK_B_MIN_GAIN", "TRACK_B_Z", "TRACK_B_CONTROL_Z", "TRACK_A_MIN_GAIN", "TRACK_A_Z",
    "CLAIM_Z", "CLAIM_CONSISTENCY_Z", "METRIC_SYSTEMATICS", "GUARD_MIN_FRACTION",
]

# Thresholds of the approved plan (sections 5, 6 and 8), in nat per event.
TRACK_B_MIN_GAIN = 5e-4      # Track B promotion: validation gain against the reference of at least this size ...
TRACK_B_Z = 3.0              # ... and at least this many paired SE,
TRACK_B_CONTROL_Z = 1.0      # ... and better than the control B0 by more than this many paired SE
TRACK_A_MIN_GAIN = 3e-4      # Track A "kept": validation gain against rung A2 of at least this size ...
TRACK_A_Z = 2.0              # ... and at least this many paired SE
CLAIM_Z = 3.0                # final claim: test difference negative by at least this many SE ...
CLAIM_CONSISTENCY_Z = 2.0    # ... and equal to the validation difference within this many SE of their difference
METRIC_SYSTEMATICS = 5e-4    # |difference| below this is stated to lie within the metric's own systematics
GUARD_MIN_FRACTION = 0.5     # guards G-K and G-bulk: at least this fraction of the gain persists under the guard metric

_MAX_CELLS = 2_000_000       # bootstrap work array: at most this many (replicate, event) cells at a time


# ----------------------------------------------------------------------------------------------------------------------
# Paired difference
# ----------------------------------------------------------------------------------------------------------------------
def _events(x, name: str) -> np.ndarray:
    """1-D float64 copy of a per-event array; non-finite values are an error, not something to average over."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D per-event array, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: {int((~np.isfinite(a)).sum())} non-finite value(s); filter them explicitly")
    return a


def z_from_paired(mean: float, se: float) -> float:
    """z = mean / se. For se == 0 (all per-event differences identical): 0 if mean == 0, else +-inf."""
    mean, se = float(mean), float(se)
    if not (math.isfinite(mean) and se >= 0.0):
        raise ValueError(f"z_from_paired: need a finite mean and se >= 0, got mean={mean}, se={se}")
    if se == 0.0:
        return 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    return mean / se


def p_from_z(z: float, alternative: str = "less") -> float:
    """Normal-theory p-value of a z-score. 'less' (default) tests "the difference is negative" = an improvement,
    'greater' the opposite, 'two-sided' both. Phi(x) = erfc(-x / sqrt 2) / 2."""
    z = float(z)
    if math.isnan(z):
        raise ValueError("p_from_z: z is NaN")
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))                      # noqa: E731
    if alternative == "less":
        return phi(z)
    if alternative == "greater":
        return phi(-z)
    if alternative == "two-sided":
        return min(1.0, 2.0 * phi(-abs(z)))
    raise ValueError(f"unknown alternative {alternative!r}")


class Paired(NamedTuple):
    """Mean and standard error of a per-event difference, and the number of events."""
    mean: float
    se: float
    n: int

    @property
    def z(self) -> float:
        return z_from_paired(self.mean, self.se)


def paired(a, b) -> Paired:
    """Mean and standard error of the per-event difference ``a - b``.

    d = float64(a) - float64(b);  mean = d.mean();  se = d.std(ddof=1) / sqrt(n).
    The same definition as ``shapelab.paired`` (shape-form-proton/shapelab.py:253-256), plus n.
    The SE assumes independent events; it is the SE of the difference, so the correlation between the two
    models' per-event NLLs is fully used.
    """
    a, b = _events(a, "a"), _events(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"paired: different numbers of events ({len(a)} and {len(b)})")
    if len(a) < 2:
        raise ValueError("paired: at least two events are needed for a standard error")
    d = a - b
    return Paired(float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int(len(d)))


# ----------------------------------------------------------------------------------------------------------------------
# 1-SE rule
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One candidate of a selection: per-event validation NLL and its complexity key (k, then C)."""
    name: str
    nll: np.ndarray = field(repr=False)     # per-event validation NLL, same events and order for every candidate
    k: int = 0                              # number of free constants
    C: float = 0.0                          # operator-weighted size (second key)


@dataclass(frozen=True)
class OneSE:
    selected: str                           # the simplest candidate within one paired SE of the best
    best: str                               # the candidate with the lowest mean validation NLL
    table: list                             # one dict per candidate, in input order (pandas.DataFrame(table) works)


def one_se_select(candidates: Sequence[Candidate]) -> OneSE:
    """The campaign's 1-SE rule on per-event validation NLLs.

    1. best c* = lowest mean validation NLL (float64). Exact ties in the mean go to the simpler candidate
       (k, then C), then to the earlier one.
    2. For every other candidate c: Delta_c = mean(nll_c - nll_c*), SE_c = sd(nll_c - nll_c*, ddof=1)/sqrt(n).
       c is *eligible* iff Delta_c < SE_c (strict). c* is always eligible.
    3. selected = the eligible candidate with the smallest (k, C); ties go to the lower mean, then to input order.

    Steps 1-2 are the rule of ``common.select_architecture`` (c* = idxmin; ``Delta < SE``; sort by
    ``["params", "val NLL"]``) with the complexity key (k, C) in place of the parameter count. Two differences,
    both confined to degenerate input: means are float64 instead of float32 (c* can differ only if two means
    agree to ~1e-6 nat), and an exact tie of the best mean is resolved towards the simpler candidate instead
    of the first listed (in ``common`` a simpler exact duplicate of c* has Delta = SE = 0 and fails the strict
    inequality).

    Table columns: name, k, C, mean, delta, se, z (= delta/se; 0 for c*), within_1se, best, selected.
    """
    cands = list(candidates)
    if not cands:
        raise ValueError("one_se_select: no candidates")
    names = [c.name for c in cands]
    if len(set(names)) != len(names):
        raise ValueError("one_se_select: candidate names must be unique")
    arrs = [_events(c.nll, f"{c.name}.nll") for c in cands]
    if len({len(a) for a in arrs}) != 1:
        raise ValueError("one_se_select: candidates were scored on different numbers of events")
    means = [float(a.mean()) for a in arrs]
    idx = range(len(cands))
    i_best = min(idx, key=lambda i: (means[i], cands[i].k, cands[i].C, i))
    stats = [Paired(0.0, 0.0, len(arrs[i])) if i == i_best else paired(arrs[i], arrs[i_best]) for i in idx]
    within = [i == i_best or stats[i].mean < stats[i].se for i in idx]
    i_sel = min((i for i in idx if within[i]), key=lambda i: (cands[i].k, cands[i].C, means[i], i))
    table = [{"name": cands[i].name, "k": cands[i].k, "C": cands[i].C, "mean": means[i],
              "delta": stats[i].mean, "se": stats[i].se,
              "z": 0.0 if i == i_best else z_from_paired(stats[i].mean, stats[i].se),
              "within_1se": bool(within[i]), "best": i == i_best, "selected": i == i_sel} for i in idx]
    return OneSE(selected=cands[i_sel].name, best=cands[i_best].name, table=table)


# ----------------------------------------------------------------------------------------------------------------------
# Event bootstrap
# ----------------------------------------------------------------------------------------------------------------------
def _groups(strata, n: int):
    """Index arrays of the strata (None -> no stratification)."""
    if strata is None:
        return None
    lab = np.asarray(strata)
    if lab.shape != (n,):
        raise ValueError(f"strata: expected one label per event ({n}), got shape {lab.shape}")
    return [np.flatnonzero(lab == v) for v in np.unique(lab)]


def _counts(rng: np.random.Generator, n: int, reps: int, groups=None) -> np.ndarray:
    """(reps, n) float64: how often each event enters each bootstrap replicate (rows sum to n).
    Without groups: n draws with replacement from all events. With groups: every stratum is resampled within
    itself, keeping its size."""
    if groups is None:
        idx = rng.integers(0, n, size=(reps, n))
    else:
        idx = np.empty((reps, n), dtype=np.int64)
        col = 0
        for g in groups:
            idx[:, col:col + len(g)] = g[rng.integers(0, len(g), size=(reps, len(g)))]
            col += len(g)
    idx += n * np.arange(reps, dtype=np.int64)[:, None]
    return np.bincount(idx.ravel(), minlength=reps * n).reshape(reps, n).astype(np.float64)


def _chunks(n_boot: int, n: int):
    step = max(1, min(n_boot, _MAX_CELLS // max(n, 1)))
    for s in range(0, n_boot, step):
        yield min(step, n_boot - s)


@dataclass(frozen=True)
class Bootstrap:
    mean: float                             # observed mean of a - b
    se: float                               # bootstrap standard error (sd of the replicate means, ddof=1)
    lo: float                               # confidence interval
    hi: float
    level: float
    method: str
    n_boot: int
    n: int
    samples: np.ndarray = field(repr=False)  # the n_boot replicate means


def bootstrap_paired(a, b, n_boot: int = 2000, seed: int = 0, *, strata=None, level: float = 0.95,
                     method: str = "percentile") -> Bootstrap:
    """Event bootstrap of the paired mean difference ``mean(a - b)``.

    Events (not hits) are resampled with replacement, n events per replicate; the statistic is the mean of the
    resampled differences. With ``strata`` (one label per event, e.g. the momentum bin) every stratum is
    resampled within itself, so the stratum sizes of the original sample are kept; the statistic is still the
    plain mean over all events. Stratification removes the between-strata component of the variance, which is
    the right thing when the sample was *drawn* stratified (the validation split is 15 % of every momentum bin).

    ``method``: "percentile" = quantiles (1-level)/2 and (1+level)/2 of the replicate means;
                "basic"      = 2*mean - those quantiles, reflected.
    Deterministic for a given ``seed`` (``numpy.random.default_rng``).
    """
    a, b = _events(a, "a"), _events(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"bootstrap_paired: different numbers of events ({len(a)} and {len(b)})")
    d = a - b
    n = len(d)
    if n < 2 or n_boot < 2:
        raise ValueError("bootstrap_paired: need at least two events and two replicates")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    groups = _groups(strata, n)
    rng = np.random.default_rng(seed)
    reps = np.concatenate([_counts(rng, n, r, groups) @ d / n for r in _chunks(n_boot, n)])
    q_lo, q_hi = np.quantile(reps, [(1 - level) / 2, (1 + level) / 2])
    mean = float(d.mean())
    if method == "percentile":
        lo, hi = float(q_lo), float(q_hi)
    elif method == "basic":
        lo, hi = float(2 * mean - q_hi), float(2 * mean - q_lo)
    else:
        raise ValueError(f"unknown method {method!r}")
    return Bootstrap(mean=mean, se=float(reps.std(ddof=1)), lo=lo, hi=hi, level=level, method=method,
                     n_boot=int(n_boot), n=n, samples=reps)


# ----------------------------------------------------------------------------------------------------------------------
# Winner's curse
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WinnersCurse:
    selected: int                           # row of the candidate with the best mean on the full sample
    apparent_gain: float                    # its mean paired gain against the reference (negative = better)
    apparent_se: float                      # paired SE of that gain (no selection correction)
    bias: float                             # estimated selection bias of apparent_gain (<= 0 up to MC noise)
    bias_mc_se: float                       # Monte-Carlo SE of `bias` from the finite number of replicates
    corrected_gain: float                   # apparent_gain - bias
    optimism_half_sample: float             # raw in-bag minus out-of-bag optimism (bias at effective size n/2)
    null_bound: float                       # bias if all candidates were equivalent: the largest possible |bias|
    selection_freq: np.ndarray = field(repr=False)   # how often each candidate won a bootstrap replicate
    n_boot: int = 0
    n: int = 0
    m: int = 0

    @property
    def bias_capped(self) -> float:
        """``bias`` projected onto [null_bound, 0], the interval that contains the true bias. In simulations this
        lowers the RMSE of the estimate by 20-25 %, at the price of being 8-12 % too small in magnitude when the
        candidates are exactly equivalent (``bias`` itself is unbiased there)."""
        return min(0.0, max(self.bias, self.null_bound))


def winners_curse_correction(val_matrix, reference=None, n_boot: int = 2000, seed: int = 0) -> WinnersCurse:
    """Bias of the validation gain of the best of m candidates, estimated by bootstrapping the selection.

    Setting. ``val_matrix[j, e]`` is the validation NLL of candidate j on event e, ``reference[e]`` that of
    the reference model; G = val_matrix - reference are the per-event paired gains with means Gbar_j and true
    (population) means delta_j. The candidate jhat = argmin_j Gbar_j is reported with the apparent gain
    Gbar_jhat. Because jhat was chosen for having a favourable fluctuation,

        B = E[ Gbar_jhat - delta_jhat ] <= 0                                   (winner's curse)

    and the gain an independent sample (the test split) will show is expected to be Gbar_jhat - B.
    (The reference cancels in jhat and in the expectation B; it enters only as a control variate that removes
    the common event-to-event fluctuation from the estimate. ``reference=None`` uses the mean over candidates.)

    Estimator (Tibshirani & Tibshirani 2009 in spirit, with out-of-bag events as the independent sample).
    For b = 1..n_boot draw bootstrap counts c_e (n events with replacement) and compute, for every j,

        I_j = sum_e c_e G_je / n                       in-bag mean
        O_j = mean of G_je over events with c_e = 0    out-of-bag mean (about 36.8 % of the events)
        j_b = argmin_j I_j                             selection repeated on the in-bag sample
        opt_b = (I_jb - O_jb) - (I_jhat - O_jhat)      optimism; the second bracket has expectation exactly 0
                                                       and only reduces Monte-Carlo noise

        optimism_half_sample = mean_b opt_b,     bias = optimism_half_sample / sqrt(2 - 1/n),
        corrected_gain = apparent_gain - bias.

    Why the factor. The out-of-bag events are independent of the in-bag events, so O_jb is an unbiased
    estimate of delta_jb and E[I_jb - O_jb] is *exactly* the selection bias of an argmin rule applied to the
    in-bag means. An in-bag mean is an unbiased estimate of delta_j with variance (2 - 1/n) sigma_j^2 / n
    (E c_e^2 = 2 - 1/n), i.e. it behaves like a sample of n/(2 - 1/n) ~ n/2 events. Writing
    B(V; delta) = E[eps_jhat] for eps ~ N(0, V), the scaling B(c^2 V; delta) = c B(V; delta/c) gives

        E[bias] = B(Sigma/n; delta / sqrt(2 - 1/n))        against the truth        B(Sigma/n; delta).

    Properties (Gaussian means, i.e. n large enough for the CLT):
      * exact when all candidates are equivalent (delta_j equal): then B = E[min_j eps_j],
        e.g. -sigma mu_m sqrt((1 - rho)/n) for m equicorrelated candidates, mu_m = E[max of m N(0,1)];
      * conservative otherwise: h(delta) = E[min_j (delta_j + eps_j)] is concave with gradient = selection
        probabilities, so B(delta) = h(delta) - delta . grad h(delta). Along a ray, with f(t) = h(t delta):
        B(t delta) = f(t) - t f'(t) and dB/dt = -t f''(t) >= 0. The bias therefore shrinks in magnitude along
        every ray away from the null (for any noise distribution and any m), and the estimator evaluates it at
        t = 1/sqrt 2 < 1: it over-corrects, never under-corrects, in expectation. For m = 2 with true gap g
        and s = sd(Gbar_1 - Gbar_2): truth -s phi(g/s), estimate -s phi(g/(sqrt2 s));
      * it tends to 0 with the truth when one candidate is separated by many SE;
      * ``null_bound`` = mean_b min_j (I_j - Gbar_j) is the plain bootstrap estimate of E[min_j eps_j]. Since
        eps_jhat >= min_j eps_j holds for *any* data-dependent selection (including the 1-SE rule),
        |B| <= |null_bound|: it is the worst case and does not depend on the unknown true gaps. A ``bias``
        beyond it is sampling noise of the estimate (see ``bias_capped``).

    Alternatives that were considered and measured on synthetic candidates: the classical bootstrap bias
    (in-bag minus full-sample mean) is half of ``optimism_half_sample``, i.e. 1/sqrt 2 of the truth at the null
    (it under-corrects); the unscaled in-bag minus out-of-bag optimism is sqrt 2 times the truth at the null;
    larger resamples (r n draws, scale sqrt(1 + 1/r)) are equally calibrated and less conservative away from the
    null but have a larger RMSE for r = 2, 3, 4 because few out-of-bag events remain.

    Assumptions and limits: events are independent and exchangeable (plain event bootstrap, no strata); the
    selection being corrected is "lowest mean validation NLL among exactly these m candidates" - candidates
    discarded earlier on the same validation events, early stopping on them, or hyper-parameters tuned on
    them are not covered; the bias of selection is a non-regular functional, so no estimator is unbiased for
    all configurations of the true gaps - this one is calibrated at the null and errs on the cautious side.
    Like any bootstrap quantity, ``bias`` itself has sampling error of the order of its own size for small m
    (it depends on the observed gaps); ``bias_mc_se`` covers only the Monte-Carlo part.

    With m = 1 there is no selection: bias = 0 exactly.
    """
    V = np.asarray(val_matrix, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"val_matrix: expected shape (candidates, events), got {V.shape}")
    if not np.isfinite(V).all():
        raise ValueError("val_matrix: non-finite values")
    m, n = V.shape
    if n < 10 or n_boot < 2:
        raise ValueError("winners_curse_correction: need at least 10 events and two replicates")
    ref = V.mean(0) if reference is None else _events(reference, "reference")
    if ref.shape != (n,):
        raise ValueError(f"reference: expected {n} events, got shape {ref.shape}")
    G = V - ref[None, :]
    gbar = G.mean(1)
    j_hat = int(np.argmin(gbar))
    apparent = Paired(float(gbar[j_hat]), float(G[j_hat].std(ddof=1) / np.sqrt(n)), n)
    if m == 1:
        return WinnersCurse(selected=0, apparent_gain=apparent.mean, apparent_se=apparent.se, bias=0.0,
                            bias_mc_se=0.0, corrected_gain=apparent.mean, optimism_half_sample=0.0, null_bound=0.0,
                            selection_freq=np.ones(1), n_boot=int(n_boot), n=n, m=1)
    rng = np.random.default_rng(seed)
    opt, null, wins = [], [], np.zeros(m, dtype=np.int64)
    for r in _chunks(n_boot, n):
        cnt = _counts(rng, n, r)
        out = cnt == 0
        n_out = out.sum(1)
        ok = n_out > 0                                  # a replicate without out-of-bag events is unusable (n!/n^n)
        cnt, out, n_out = cnt[ok], out[ok], n_out[ok]
        inbag = cnt @ G.T / n                           # (replicates, m)
        oob = (out.astype(np.float64) @ G.T) / n_out[:, None]
        jb = inbag.argmin(1)
        rows = np.arange(len(jb))
        opt.append((inbag[rows, jb] - oob[rows, jb]) - (inbag[:, j_hat] - oob[:, j_hat]))
        null.append((inbag - gbar[None, :]).min(1))
        wins += np.bincount(jb, minlength=m)
    opt, null = np.concatenate(opt), np.concatenate(null)
    if len(opt) < 2:
        raise ValueError("winners_curse_correction: no usable bootstrap replicate")
    scale = math.sqrt(2.0 - 1.0 / n)
    half = float(opt.mean())
    bias = half / scale
    return WinnersCurse(selected=j_hat, apparent_gain=apparent.mean, apparent_se=apparent.se, bias=bias,
                        bias_mc_se=float(opt.std(ddof=1) / np.sqrt(len(opt)) / scale),
                        corrected_gain=apparent.mean - bias, optimism_half_sample=half,
                        null_bound=float(null.mean()), selection_freq=wins / wins.sum(), n_boot=int(len(opt)),
                        n=n, m=m)


# ----------------------------------------------------------------------------------------------------------------------
# Multiplicity
# ----------------------------------------------------------------------------------------------------------------------
def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order.

    With p_(1) <= ... <= p_(m): adj_(i) = min(1, max_{j <= i} (m - j + 1) p_(j)). Rejecting where adj <= alpha
    controls the family-wise error rate at alpha under any dependence between the tests.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("holm: expected a 1-D array of p-values")
    if p.size == 0:
        return p.copy()
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ValueError("holm: p-values must be finite and in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Decision rules of the plan
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Decision:
    """Outcome of a rule: the verdict, every individual check, the numbers used, and informational flags."""
    passed: bool
    checks: dict
    numbers: dict
    flags: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


def track_b_promotion(vs_reference: Paired, vs_control: Paired, *, min_gain: float = TRACK_B_MIN_GAIN,
                      z_reference: float = TRACK_B_Z, z_control: float = TRACK_B_CONTROL_Z) -> Decision:
    """Track B promotion (plan section 6), the part that is a function of validation numbers:

        Delta_ref <= -min_gain   and   Delta_ref <= -z_reference * SE_ref   and   Delta_ctl < -z_control * SE_ctl

    with ``vs_reference = paired(candidate, mlp_std)`` and ``vs_control = paired(candidate, B0)`` on validation
    at K = 64. Replication by a second seed and the guards (G-K, G-bulk, G-region, G-mass) are further
    conditions of the plan that this function cannot see.
    """
    r, c = vs_reference, vs_control
    checks = {"gain": r.mean <= -min_gain, "significance": r.mean <= -z_reference * r.se,
              "beats_control": c.mean < -z_control * c.se}
    numbers = {"delta_reference": r.mean, "se_reference": r.se, "z_reference": z_from_paired(r.mean, r.se),
               "delta_control": c.mean, "se_control": c.se, "z_control": z_from_paired(c.mean, c.se),
               "min_gain": min_gain, "z_reference_required": z_reference, "z_control_required": z_control}
    return Decision(all(checks.values()), checks, numbers)


def track_a_kept(vs_base: Paired, *, min_gain: float = TRACK_A_MIN_GAIN, z: float = TRACK_A_Z) -> Decision:
    """Track A "kept" (plan section 5): Delta <= -min_gain and Delta <= -z * SE, ``vs_base = paired(rung, A2)``
    on validation. Anything else is recorded as a null."""
    checks = {"gain": vs_base.mean <= -min_gain, "significance": vs_base.mean <= -z * vs_base.se}
    numbers = {"delta": vs_base.mean, "se": vs_base.se, "z": z_from_paired(vs_base.mean, vs_base.se),
               "min_gain": min_gain, "z_required": z}
    return Decision(all(checks.values()), checks, numbers)


def final_claim(test: Paired, val: Paired, *, z: float = CLAIM_Z, z_consistency: float = CLAIM_CONSISTENCY_Z,
                systematics: float = METRIC_SYSTEMATICS, val_selection_bias: float = 0.0) -> Decision:
    """Final claim of a win (plan section 8), for ``test = paired(finalist, benchmark)`` on the test split and
    ``val`` the same pair on validation:

        significant:  Delta_test <= -z * SE_test          (and Delta_test < 0)
        consistent:   |Delta_test - Delta_val'| <= z_consistency * sqrt(SE_test^2 + SE_val^2)

    Validation and test events are disjoint, so the two means are independent and the SE of their difference
    is the quadrature sum. Delta_val' = Delta_val - val_selection_bias; the default 0 is the plan's rule as
    written, and passing ``winners_curse_correction(...).bias`` compares the test gain with the validation
    gain after removing the optimism of having selected the finalist on validation.

    Flag ``within_metric_systematics``: |Delta_test| < systematics (5e-4), to be stated next to the claim; it
    does not change ``passed``. ``numbers["shrinkage"]`` = Delta_test / Delta_val (NaN if Delta_val == 0).
    """
    val_c = val.mean - val_selection_bias
    se_diff = math.hypot(test.se, val.se)
    checks = {"significant": test.mean < 0.0 and test.mean <= -z * test.se,
              "consistent_with_validation": abs(test.mean - val_c) <= z_consistency * se_diff}
    numbers = {"delta_test": test.mean, "se_test": test.se, "z_test": z_from_paired(test.mean, test.se),
               "delta_val": val.mean, "se_val": val.se, "val_selection_bias": val_selection_bias,
               "delta_val_corrected": val_c, "test_minus_val": test.mean - val_c, "se_test_minus_val": se_diff,
               "shrinkage": test.mean / val.mean if val.mean != 0.0 else math.nan,
               "z_required": z, "z_consistency": z_consistency, "systematics": systematics}
    flags = {"within_metric_systematics": abs(test.mean) < systematics}
    return Decision(all(checks.values()), checks, numbers, flags)


def guard_persistence(main_gain, guard_gain, *, min_fraction: float = GUARD_MIN_FRACTION) -> Decision:
    """Guards G-K and G-bulk (plan section 6): "the sign holds and at least ``min_fraction`` of the gain persists".

    ``main_gain[e]`` = candidate - reference under the report metric, ``guard_gain[e]`` = the same difference under
    the guard metric (K = 1024, or bulk-restricted), on the *same events* (mask events without a bulk NLL in both).

    Checks, on point estimates as the plan states the rule:
        main_is_gain:  mean(main) < 0
        sign_holds:    mean(guard) < 0
        fraction:      mean(guard) <= min_fraction * mean(main)
    The retained fraction mean(guard)/mean(main) is a ratio of two noisy, strongly correlated means and has no
    useful SE of its own. The same condition is linear in the per-event values, margin_e = guard_e -
    min_fraction * main_e with "fraction" <=> mean(margin) <= 0, so ``numbers`` also holds mean(margin), its
    paired SE and z, and the flag ``resolved`` says whether the verdict stands at two SE (|z| >= 2). A pass or
    a fail with ``resolved`` False is a statement about noise, not about the candidate.
    """
    main, guard = _events(main_gain, "main_gain"), _events(guard_gain, "guard_gain")
    if main.shape != guard.shape:
        raise ValueError(f"guard_persistence: different numbers of events ({len(main)} and {len(guard)})")
    pm, pg = paired(main, np.zeros_like(main)), paired(guard, np.zeros_like(guard))
    margin = paired(guard, min_fraction * main)
    checks = {"main_is_gain": pm.mean < 0.0, "sign_holds": pg.mean < 0.0, "fraction": margin.mean <= 0.0}
    z = z_from_paired(margin.mean, margin.se)
    numbers = {"gain_main": pm.mean, "se_main": pm.se, "gain_guard": pg.mean, "se_guard": pg.se,
               "fraction": pg.mean / pm.mean if pm.mean != 0.0 else math.nan, "min_fraction": min_fraction,
               "margin": margin.mean, "se_margin": margin.se, "z_margin": z}
    return Decision(all(checks.values()), checks, numbers, {"resolved": abs(z) >= 2.0})
