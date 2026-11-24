"""Tests of sfm.stats: known answers, analytic standard errors, the 1-SE rule, bootstrap coverage and the
calibration of the winner's-curse estimator. NumPy only; every random number comes from a fixed seed."""
import math

import numpy as np
import pytest

from sfm import stats as S


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def expected_max_normal(m):
    """mu_m = E[max of m iid N(0,1)] = int x m phi(x) Phi(x)^(m-1) dx (trapezoid rule on [-10, 10])."""
    x = np.linspace(-10.0, 10.0, 40001)
    dens = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    cdf = np.array([0.5 * math.erfc(-v / math.sqrt(2)) for v in x])
    return float(np.trapezoid(x * m * dens * cdf ** (m - 1), x))


# ----------------------------------------------------------------------------------------------------------------------
# paired, z, p
# ----------------------------------------------------------------------------------------------------------------------
def test_paired_known_answer():
    r = S.paired([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0])
    assert r.n == 4
    assert r.mean == pytest.approx(2.5, abs=1e-15)
    assert r.se == pytest.approx(math.sqrt(5.0 / 3.0) / 2.0, rel=1e-14)      # sd = sqrt(5/3), se = sd / sqrt(4)
    assert r.z == pytest.approx(2.5 / (math.sqrt(5.0 / 3.0) / 2.0), rel=1e-14)
    mean, se, n = r                                                           # unpacks as (mean, se, n)
    assert (mean, se, n) == (r.mean, r.se, 4)


def test_paired_is_the_shapelab_definition_and_works_in_float64():
    rng = np.random.default_rng(0)
    a = (18.12 + rng.standard_normal(5000)).astype(np.float32)
    b = (a + 1e-3 * rng.standard_normal(5000).astype(np.float32)).astype(np.float32)
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)     # shapelab.py:255-256, verbatim
    ref = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))
    r = S.paired(a, b)
    assert (r.mean, r.se) == ref and r.n == 5000
    assert S.paired(b, a).mean == -r.mean and S.paired(b, a).se == r.se       # antisymmetric


def test_paired_se_analytic_cases():
    # d = +1, -1, +1, ... (n even): mean 0, sum of squares n, so sd = sqrt(n/(n-1)) and se = 1/sqrt(n-1)
    n = 1000
    d = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    r = S.paired(d + 7.0, np.full(n, 7.0))
    assert r.mean == pytest.approx(0.0, abs=1e-15) and r.se == pytest.approx(1.0 / math.sqrt(n - 1), rel=1e-12)
    # Bernoulli(q) differences: se = sqrt(q(1-q) n/(n-1)) / sqrt(n) exactly
    d = np.r_[np.ones(300), np.zeros(700)]
    r = S.paired(d, np.zeros(1000))
    assert r.mean == pytest.approx(0.3) and r.se == pytest.approx(math.sqrt(0.3 * 0.7 / 999.0), rel=1e-12)
    # the SE is that of the difference: a huge common-mode fluctuation does not enter
    rng = np.random.default_rng(1)
    common, eps = 5.0 * rng.standard_normal(20000), 0.01 * rng.standard_normal(20000)
    r = S.paired(common + eps, common)
    assert r.se == pytest.approx(0.01 / math.sqrt(20000), rel=0.03)
    # constant difference: se = 0, z = -inf
    r = S.paired(np.arange(5.0) - 0.5, np.arange(5.0))
    assert r.mean == -0.5 and r.se == 0.0 and r.z == -math.inf


def test_paired_rejects_bad_input():
    with pytest.raises(ValueError):
        S.paired([1.0, np.nan, 2.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        S.paired([1.0, np.inf], [0.0, 0.0])
    with pytest.raises(ValueError):
        S.paired([1.0, 2.0, 3.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        S.paired([1.0], [0.0])
    with pytest.raises(ValueError):
        S.paired(np.zeros((2, 3)), np.zeros((2, 3)))


def test_z_and_p():
    assert S.z_from_paired(-6e-4, 2e-4) == pytest.approx(-3.0)
    assert S.z_from_paired(0.0, 0.0) == 0.0
    assert S.z_from_paired(1.0, 0.0) == math.inf and S.z_from_paired(-1.0, 0.0) == -math.inf
    with pytest.raises(ValueError):
        S.z_from_paired(1.0, -1.0)
    with pytest.raises(ValueError):
        S.z_from_paired(math.nan, 1.0)
    assert S.p_from_z(0.0) == pytest.approx(0.5)
    assert S.p_from_z(-1.959963984540054) == pytest.approx(0.025, rel=1e-9)
    assert S.p_from_z(-3.0) == pytest.approx(1.3498980316300946e-3, rel=1e-9)
    assert S.p_from_z(3.0, "greater") == pytest.approx(S.p_from_z(-3.0, "less"))
    assert S.p_from_z(-1.959963984540054, "two-sided") == pytest.approx(0.05, rel=1e-9)
    assert S.p_from_z(-math.inf) == 0.0 and S.p_from_z(math.inf) == 1.0
    with pytest.raises(ValueError):
        S.p_from_z(0.0, "sideways")


# ----------------------------------------------------------------------------------------------------------------------
# 1-SE rule
# ----------------------------------------------------------------------------------------------------------------------
N_SEL = 4000
SE_SEL = 1e-3                                           # paired SE against the best of every constructed candidate


def _base(n=N_SEL, seed=42):
    return 18.0 + 2.0 * np.random.default_rng(seed).standard_normal(n)


def _cand(name, base, delta, k, C=0.0, period=2):
    """base + delta + s * (square wave of the given period, mean exactly 0): against `base` the paired mean is
    delta and the paired SE is s * sqrt(n/(n-1)) / sqrt(n) = SE_SEL, both exact up to rounding (no sampling)."""
    n = len(base)
    assert n % period == 0
    wave = np.where((np.arange(n) % period) < period // 2, 1.0, -1.0)
    return S.Candidate(name, base + delta + SE_SEL * math.sqrt(n - 1) * wave, k, C)


def test_one_se_picks_the_simpler_candidate_within_one_se():
    base = _base()
    cands = [_cand("small", base, 0.5 * SE_SEL, k=10),                 # 0.5 SE worse than "big": eligible
             _cand("mid", base, 5.0 * SE_SEL, k=20, period=4),         # 5 SE worse: not eligible
             S.Candidate("big", base, 40)]
    out = S.one_se_select(cands)
    rows = {r["name"]: r for r in out.table}
    assert out.best == "big" and out.selected == "small"
    assert rows["small"]["within_1se"] and not rows["mid"]["within_1se"] and rows["big"]["within_1se"]
    assert rows["small"]["delta"] == pytest.approx(0.5 * SE_SEL, rel=1e-9) and rows["small"]["se"] == pytest.approx(SE_SEL, rel=1e-9)
    assert rows["mid"]["z"] == pytest.approx(5.0, rel=1e-9) and rows["big"]["z"] == 0.0
    assert rows["big"]["delta"] == 0.0 and rows["big"]["se"] == 0.0 and rows["big"]["best"]
    assert [r["selected"] for r in out.table] == [True, False, False]
    # the table reports exactly paired(candidate, best)
    p = S.paired(cands[1].nll, cands[2].nll)
    assert (rows["mid"]["delta"], rows["mid"]["se"]) == (p.mean, p.se)
    assert [r["name"] for r in out.table] == ["small", "mid", "big"]            # input order kept


def test_one_se_keeps_the_best_when_nothing_simpler_is_within_one_se():
    base = _base()
    out = S.one_se_select([_cand("small", base, 1.5 * SE_SEL, k=10), _cand("smaller", base, 4.0 * SE_SEL, k=5, period=4),
                           S.Candidate("big", base, 40)])
    assert out.best == out.selected == "big"
    assert [r["within_1se"] for r in out.table] == [False, False, True]


def test_one_se_best_is_also_the_simplest():
    base = _base()
    cands = [S.Candidate("small", base, 10), _cand("big", base, 0.2 * SE_SEL, k=40),
             _cand("huge", base, 0.3 * SE_SEL, k=80, period=4)]
    out = S.one_se_select(cands)
    assert out.best == out.selected == "small"
    assert sum(r["selected"] for r in out.table) == 1 and sum(r["best"] for r in out.table) == 1
    # more complex candidates within 1 SE are eligible but never chosen
    assert all(r["within_1se"] for r in out.table)


def test_one_se_threshold_is_strict_and_uses_the_paired_se():
    # Construct Delta and SE exactly: d = c + s * (+1, -1, ...) has mean c and SE s / sqrt(n - 1).
    n = 101 * 2
    pm = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    best = np.zeros(n)
    se = 1.0 / math.sqrt(n - 1)
    for delta, eligible in ((0.5 * se, True), (0.999 * se, True), (1.001 * se, False), (3.0 * se, False)):
        out = S.one_se_select([S.Candidate("simple", best + delta + pm, 1), S.Candidate("best", best, 9)])
        assert out.best == "best"
        assert (out.selected == "simple") is eligible, delta / se
    # a candidate that is much better on most events but not on average is judged on the paired mean only
    out = S.one_se_select([S.Candidate("simple", best + 2.0 * se + pm, 1), S.Candidate("best", best, 9)])
    assert out.selected == "best"


def test_one_se_ties():
    base = _base()
    best = S.Candidate("best", base, 40)
    a = _cand("a", base, 0.2 * SE_SEL, k=10, C=30.0)
    b = _cand("b", base, 0.3 * SE_SEL, k=10, C=20.0, period=4)   # same k, smaller C, slightly worse mean
    c = _cand("c", base, 0.1 * SE_SEL, k=10, C=20.0, period=8)   # same (k, C) as b, better mean
    assert S.one_se_select([best, a, b]).selected == "b"         # k ties -> C decides
    assert S.one_se_select([best, a, b, c]).selected == "c"      # (k, C) ties -> lower mean decides
    assert S.one_se_select([c, b, a, best]).selected == "c"      # independent of the input order
    assert S.one_se_select([c, b, a, best]).best == "best"
    # identical (k, C, mean): the earlier candidate
    d1, d2 = S.Candidate("d1", c.nll.copy(), 10, 20.0), S.Candidate("d2", c.nll.copy(), 10, 20.0)
    assert S.one_se_select([best, d1, d2]).selected == "d1" and S.one_se_select([best, d2, d1]).selected == "d2"
    # exact duplicate of the best mean with fewer constants: it *is* the best (common would depend on the order)
    dup = S.Candidate("dup", best.nll.copy(), 5)
    for order in ([best, dup], [dup, best]):
        out = S.one_se_select(order)
        assert out.best == "dup" and out.selected == "dup"
    # equal mean, different events (Delta = 0 < SE): the simpler one is eligible and chosen
    n = 200
    pm = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    out = S.one_se_select([S.Candidate("complex", np.zeros(n), 9), S.Candidate("simple", pm, 1)])
    assert out.selected == "simple"


def test_one_se_matches_the_rule_in_common_select_architecture():
    """Transcription of common.py:846-851 (k_star = idxmin; ok = k_star or Delta < SE; sort by params, val NLL)
    with float64 means, compared on random candidate sets with distinct parameter counts."""
    rng = np.random.default_rng(7)
    for trial in range(200):
        m, n = int(rng.integers(2, 7)), 500
        base = 18.0 + rng.standard_normal(n)
        val = {f"c{j}": base + rng.normal(0, 2e-3) + 0.03 * rng.standard_normal(n) for j in range(m)}
        params = {k: int(p) for k, p in zip(val, rng.permutation(m) * 100 + 50)}
        mean = {k: float(np.asarray(v, dtype=np.float64).mean()) for k, v in val.items()}
        k_star = min(val, key=lambda k: mean[k])
        d = {k: val[k] - val[k_star] for k in val}
        ok = [k for k in val if k == k_star or float(d[k].mean()) < float(d[k].std(ddof=1) / np.sqrt(n))]
        k_sel = sorted(ok, key=lambda k: (params[k], mean[k]))[0]
        out = S.one_se_select([S.Candidate(k, val[k], params[k]) for k in val])
        assert (out.best, out.selected) == (k_star, k_sel), trial


def test_one_se_input_validation():
    with pytest.raises(ValueError):
        S.one_se_select([])
    with pytest.raises(ValueError):
        S.one_se_select([S.Candidate("a", np.zeros(5), 1), S.Candidate("a", np.ones(5), 2)])
    with pytest.raises(ValueError):
        S.one_se_select([S.Candidate("a", np.zeros(5), 1), S.Candidate("b", np.ones(6), 2)])
    with pytest.raises(ValueError):
        S.one_se_select([S.Candidate("a", np.array([0.0, np.nan, 1.0]), 1), S.Candidate("b", np.ones(3), 2)])
    one = S.one_se_select([S.Candidate("only", np.arange(4.0), 3)])
    assert one.best == one.selected == "only"


# ----------------------------------------------------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------------------------------------------------
def test_bootstrap_se_matches_the_analytic_se_and_is_deterministic():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(3000) + 0.02, rng.standard_normal(3000)
    p = S.paired(a, b)
    bs = S.bootstrap_paired(a, b, n_boot=4000, seed=11)
    assert bs.mean == p.mean and bs.n == 3000 and bs.n_boot == 4000 and bs.samples.shape == (4000,)
    # bootstrap sd of a mean = sqrt((n-1)/n) * sd/sqrt(n); its own Monte-Carlo error is 1/sqrt(2*4000) = 1.1 %
    assert bs.se == pytest.approx(p.se * math.sqrt(2999 / 3000), rel=0.05)
    assert bs.lo < bs.mean < bs.hi
    assert (bs.lo, bs.hi) == pytest.approx((p.mean - 1.96 * p.se, p.mean + 1.96 * p.se), abs=0.25 * p.se)
    again = S.bootstrap_paired(a, b, n_boot=4000, seed=11)
    assert np.array_equal(bs.samples, again.samples) and (bs.lo, bs.hi) == (again.lo, again.hi)
    other = S.bootstrap_paired(a, b, n_boot=4000, seed=12)
    assert not np.array_equal(bs.samples, other.samples)
    basic = S.bootstrap_paired(a, b, n_boot=4000, seed=11, method="basic")
    assert basic.lo == pytest.approx(2 * bs.mean - bs.hi) and basic.hi == pytest.approx(2 * bs.mean - bs.lo)


def test_bootstrap_coverage():
    """Coverage of the nominal 95 % percentile interval on synthetic paired data with a known mean difference.

    300 independent data sets (n = 200 events; skewed differences: exponential, sd 1, true mean 0.3).
    The observed coverage is binomial with sd sqrt(0.95*0.05/300) = 1.3 %; the percentile interval for a skewed
    mean at n = 200 is known to under-cover by about 1 %. Accepted range 0.90 ... 0.985 (about -3.5 / +2.7 sd)."""
    rng = np.random.default_rng(2024)
    true, hits = 0.3, 0
    for r in range(300):
        common = 4.0 * rng.standard_normal(200)                          # shared event fluctuation, cancels
        d = rng.exponential(1.0, 200) - 1.0 + true
        bs = S.bootstrap_paired(common + d, common, n_boot=400, seed=r)
        hits += bs.lo <= true <= bs.hi
    assert 0.90 <= hits / 300 <= 0.985, hits / 300


def test_bootstrap_stratified():
    """Strata with very different means: resampling within strata removes the between-strata variance.
    Analytic SEs: unstratified sqrt(var_total/n); stratified sqrt(sum_s (n_s/n)^2 var_s/n_s)."""
    rng = np.random.default_rng(5)
    sizes, means = [500, 1500, 1000], [-1.0, 0.0, 2.0]
    lab = np.repeat([10, 20, 30], sizes)
    d = np.concatenate([m + 0.2 * rng.standard_normal(s) for m, s in zip(means, sizes)])
    perm = rng.permutation(len(d))                                       # labels need not be contiguous
    d, lab = d[perm], lab[perm]
    n = len(d)
    se_plain = math.sqrt(d.var() / n)
    se_strat = math.sqrt(sum((np.sum(lab == v) / n) ** 2 * d[lab == v].var() / np.sum(lab == v) for v in (10, 20, 30)))
    plain = S.bootstrap_paired(d, np.zeros(n), n_boot=3000, seed=1)
    strat = S.bootstrap_paired(d, np.zeros(n), n_boot=3000, seed=1, strata=lab)
    assert plain.se == pytest.approx(se_plain, rel=0.06) and strat.se == pytest.approx(se_strat, rel=0.06)
    assert strat.se < 0.3 * plain.se                                     # here the between-strata part dominates
    assert strat.mean == plain.mean == float(d.mean())
    # every replicate keeps the stratum sizes: with constant strata the replicate means do not vary at all
    const = np.repeat([1.0, 5.0], [40, 60])
    flat = S.bootstrap_paired(const, np.zeros(100), n_boot=50, seed=0, strata=np.repeat([0, 1], [40, 60]))
    assert np.allclose(flat.samples, const.mean(), rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        S.bootstrap_paired(d, np.zeros(n), n_boot=10, seed=0, strata=lab[:-1])


def test_bootstrap_counts_are_proper_resamples():
    rng = np.random.default_rng(0)
    c = S._counts(rng, 50, 200)
    assert c.shape == (200, 50) and np.all(c.sum(1) == 50) and c.min() >= 0
    assert abs(c.mean() - 1.0) < 1e-12 and abs(c.var() - (1 - 1 / 50)) < 0.05      # multinomial(n, 1/n) marginals
    assert abs((c == 0).mean() - (1 - 1 / 50) ** 50) < 0.02                         # out-of-bag fraction ~ e^-1
    groups = [np.arange(0, 20), np.arange(20, 50)]
    c = S._counts(rng, 50, 200, groups)
    assert np.all(c[:, :20].sum(1) == 20) and np.all(c[:, 20:].sum(1) == 30)
    with pytest.raises(ValueError):
        S.bootstrap_paired([1.0, 2.0], [0.0, 0.0], n_boot=10, seed=0, method="nope")


# ----------------------------------------------------------------------------------------------------------------------
# Winner's curse
# ----------------------------------------------------------------------------------------------------------------------
def _null_candidates(rng, m, n, rho=0.0, gaps=None):
    """m candidates whose per-event gains have unit variance, pairwise correlation rho and true means `gaps`
    (default all 0), on top of a large event fluctuation shared with the reference."""
    shared = rng.standard_normal(n)
    G = math.sqrt(rho) * shared + math.sqrt(1 - rho) * rng.standard_normal((m, n))
    if gaps is not None:
        G = G + np.asarray(gaps, dtype=float)[:, None]
    ref = 3.0 * rng.standard_normal(n)
    return G + ref, ref, G


@pytest.mark.parametrize("m,n,rho", [(8, 400, 0.0), (8, 400, 0.6), (4, 1000, 0.0)])
def test_winners_curse_recovers_the_null_optimism(m, n, rho):
    """m equivalent candidates (true gains all 0), gains N(0,1) per event with pairwise correlation rho.

    Known optimism: the mean gains are equicorrelated normals with sd 1/sqrt(n), so
        E[min_j Gbar_j] = -mu_m * sqrt((1 - rho)/n),   mu_m = E[max of m iid N(0,1)]  (mu_8 = 1.4236, mu_4 = 1.0294),
    because the shared component shifts all candidates alike and drops out of the minimum's excess.

    Tolerance. One data set gives the estimate with a relative sd of about 19 % (it depends on the observed
    gaps; measured by simulation, with 300 replicates the Monte-Carlo part is 6 %). The mean over 60 data sets
    therefore has a relative sd of 2.5 %; 10 % is four of those. The un-rescaled in-bag minus out-of-bag
    optimism must instead come out sqrt(2) = 1.414 times larger (effective selection sample n/2)."""
    rng = np.random.default_rng(100 + m + n)
    expected = -expected_max_normal(m) * math.sqrt((1 - rho) / n)
    bias, raw, null = [], [], []
    for r in range(60):
        V, ref, _ = _null_candidates(rng, m, n, rho)
        w = S.winners_curse_correction(V, ref, n_boot=300, seed=r)
        bias.append(w.bias), raw.append(w.optimism_half_sample), null.append(w.null_bound)
        assert w.m == m and w.n == n and w.selection_freq.shape == (m,) and w.selection_freq.sum() == pytest.approx(1)
        assert w.corrected_gain == pytest.approx(w.apparent_gain - w.bias)
    assert np.mean(bias) == pytest.approx(expected, rel=0.10)
    assert np.mean(raw) == pytest.approx(math.sqrt(2) * expected, rel=0.10)
    assert np.mean(null) == pytest.approx(expected, rel=0.10)             # the worst-case bound is attained at the null
    assert np.mean(bias) < 0


def test_winners_curse_corrected_gain_is_unbiased_at_the_null():
    """Direct check of the purpose: over many data sets with equivalent candidates the apparent gain of the
    winner is negative on average (true gain 0) and the corrected gain is compatible with 0.
    sd of the mean corrected gain over 150 data sets ~ 0.6 * (1/sqrt(n)) / sqrt(150) = 2.4e-3 at n = 400;
    the bound 0.01 is four of those, while the uncorrected mean is -0.071."""
    rng = np.random.default_rng(9)
    app, cor = [], []
    for r in range(150):
        V, ref, _ = _null_candidates(rng, 8, 400)
        w = S.winners_curse_correction(V, ref, n_boot=200, seed=r)
        app.append(w.apparent_gain), cor.append(w.corrected_gain)
    assert np.mean(app) == pytest.approx(-expected_max_normal(8) / 20.0, rel=0.10)
    assert abs(np.mean(cor)) < 0.01


def test_winners_curse_two_candidates_with_a_gap_is_conservative():
    """m = 2, true gap g, s = sd(Gbar_1 - Gbar_2) = sqrt(2/n). Truth: -s phi(g/s). The estimator evaluates the
    bias with the gap shrunk by sqrt 2: expectation -s phi(g/(sqrt2 s)) - larger in magnitude, never smaller.
    At g = 1.5 s: truth -0.1295 s, estimator -0.2273 s. Relative sd of the mean over 120 data sets ~ 5 %
    (per data set ~ 55 %, the estimate hinges on one observed gap); tolerance 20 %."""
    n = 400
    s = math.sqrt(2.0 / n)
    g = 1.5 * s
    rng = np.random.default_rng(31)
    bias = []
    for r in range(120):
        V, ref, _ = _null_candidates(rng, 2, n, gaps=[0.0, g])
        bias.append(S.winners_curse_correction(V, ref, n_boot=300, seed=r).bias)
    truth, predicted = -s * phi(g / s), -s * phi(g / (math.sqrt(2) * s))
    assert np.mean(bias) == pytest.approx(predicted, rel=0.20)
    assert np.mean(bias) < truth < 0                                       # over-corrects, does not under-correct


def test_winners_curse_vanishes_for_a_clear_winner_and_for_one_candidate():
    rng = np.random.default_rng(4)
    n = 2000
    V, ref, _ = _null_candidates(rng, 5, n, gaps=[0.0, 0.0, -1.0, 0.0, 0.0])    # candidate 2 leads by ~32 SE
    w = S.winners_curse_correction(V, ref, n_boot=300, seed=0)
    assert w.selected == 2 and w.selection_freq[2] == 1.0
    assert w.bias == 0.0 and w.bias_mc_se == 0.0                           # the control variate makes this exact
    assert w.corrected_gain == w.apparent_gain == pytest.approx(-1.0, abs=0.1)
    assert w.null_bound < -0.5 * expected_max_normal(5) / math.sqrt(n)     # the worst case is still reported
    one = S.winners_curse_correction(V[:1], ref, n_boot=50, seed=0)
    assert one.bias == 0.0 and one.corrected_gain == one.apparent_gain and one.m == 1


def test_winners_curse_reference_and_determinism():
    rng = np.random.default_rng(8)
    V, ref, _ = _null_candidates(rng, 6, 500)
    a = S.winners_curse_correction(V, ref, n_boot=400, seed=3)
    b = S.winners_curse_correction(V, ref, n_boot=400, seed=3)
    assert (a.bias, a.null_bound, a.selected) == (b.bias, b.null_bound, b.selected)
    p = S.paired(V[a.selected], ref)
    assert a.apparent_gain == pytest.approx(p.mean, abs=1e-14) and a.apparent_se == pytest.approx(p.se, rel=1e-12)
    # the reference changes neither the selection nor (beyond Monte-Carlo noise) the bias; None = candidate mean
    c = S.winners_curse_correction(V, None, n_boot=400, seed=3)
    assert c.selected == a.selected
    assert abs(c.bias - a.bias) < 5 * math.hypot(a.bias_mc_se, c.bias_mc_se)
    assert a.null_bound <= a.bias_capped <= 0.0 and a.bias_capped == min(0.0, max(a.bias, a.null_bound))
    with pytest.raises(ValueError):
        S.winners_curse_correction(V[0], ref)
    with pytest.raises(ValueError):
        S.winners_curse_correction(V, ref[:-1])
    with pytest.raises(ValueError):
        S.winners_curse_correction(np.where(V > 100, np.nan, V) * np.nan, ref)


# ----------------------------------------------------------------------------------------------------------------------
# Holm
# ----------------------------------------------------------------------------------------------------------------------
def test_holm_known_answers():
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    adj = S.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02], abs=1e-15)
    assert S.holm([0.2]) == pytest.approx([0.2])
    assert S.holm([0.5, 0.6, 0.9]) == pytest.approx([1.0, 1.0, 1.0])       # clipped at 1
    assert S.holm([0.01, 0.01, 0.01]) == pytest.approx([0.03, 0.03, 0.03])  # ties
    assert S.holm([]).shape == (0,)
    with pytest.raises(ValueError):
        S.holm([0.1, 1.2])
    with pytest.raises(ValueError):
        S.holm([0.1, np.nan])


def test_holm_properties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(0, 0.2, int(rng.integers(1, 9)))
        adj = S.holm(p)
        m = len(p)
        assert np.all(adj >= p - 1e-15) and np.all(adj <= np.minimum(1.0, m * p) + 1e-15)   # between raw and Bonferroni
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= -1e-15)                                         # monotone in p
        assert adj[order[0]] == pytest.approx(min(1.0, m * p[order[0]]))                     # smallest gets Bonferroni
        perm = rng.permutation(m)
        assert S.holm(p[perm]) == pytest.approx(adj[perm])                                   # order-equivariant


# ----------------------------------------------------------------------------------------------------------------------
# Decision rules
# ----------------------------------------------------------------------------------------------------------------------
def test_track_b_promotion():
    P = S.Paired
    ok = S.track_b_promotion(P(-8e-4, 2e-4, 14950), P(-5e-4, 2e-4, 14950))
    assert ok.passed and bool(ok) and all(ok.checks.values())
    assert ok.numbers["z_reference"] == pytest.approx(-4.0) and ok.numbers["z_control"] == pytest.approx(-2.5)
    assert ok.numbers["min_gain"] == 5e-4 and ok.numbers["z_reference_required"] == 3.0
    # each condition fails on its own
    small = S.track_b_promotion(P(-4e-4, 1e-4, 100), P(-4e-4, 1e-4, 100))            # 4 SE but below 5e-4
    assert not small.passed and small.checks == {"gain": False, "significance": True, "beats_control": True}
    noisy = S.track_b_promotion(P(-8e-4, 3e-4, 100), P(-8e-4, 3e-4, 100))            # 2.7 SE only
    assert not noisy.passed and noisy.checks == {"gain": True, "significance": False, "beats_control": True}
    ctrl = S.track_b_promotion(P(-8e-4, 2e-4, 100), P(-1e-4, 2e-4, 100))             # only 0.5 SE better than B0
    assert not ctrl.passed and ctrl.checks == {"gain": True, "significance": True, "beats_control": False}
    # boundaries: "<= -5e-4" and "<= -3 SE" are inclusive, "better than the control by > 1 SE" is strict
    edge = S.track_b_promotion(P(-5e-4, 5e-4 / 3.0, 100), P(-2e-4, 1e-4, 100))
    assert edge.checks["gain"] and edge.checks["significance"] and edge.passed
    assert not S.track_b_promotion(P(-8e-4, 2e-4, 100), P(-2e-4, 2e-4, 100)).checks["beats_control"]
    assert not S.track_b_promotion(P(+8e-4, 2e-4, 100), P(-5e-4, 2e-4, 100)).passed  # a loss never passes


def test_track_a_kept():
    P = S.Paired
    assert S.track_a_kept(P(-3e-4, 1.5e-4, 100)).passed                              # exactly -3e-4 and -2 SE
    assert S.track_a_kept(P(-4e-4, 1e-4, 100)).numbers["z"] == pytest.approx(-4.0)
    assert not S.track_a_kept(P(-2.9e-4, 1e-4, 100)).passed                          # too small
    assert not S.track_a_kept(P(-4e-4, 2.1e-4, 100)).passed                          # 1.9 SE
    assert not S.track_a_kept(P(3e-4, 1e-4, 100)).passed


def test_final_claim():
    P = S.Paired
    win = S.final_claim(test=P(-9e-4, 2e-4, 14956), val=P(-1.0e-3, 2e-4, 14950))
    assert win.passed and not win.flags["within_metric_systematics"]
    assert win.numbers["se_test_minus_val"] == pytest.approx(math.hypot(2e-4, 2e-4))
    assert win.numbers["shrinkage"] == pytest.approx(0.9) and win.numbers["z_test"] == pytest.approx(-4.5)
    # significant but small: the claim passes and carries the systematics flag
    small = S.final_claim(P(-4e-4, 1e-4, 100), P(-5e-4, 1e-4, 100))
    assert small.passed and small.flags["within_metric_systematics"]
    # not 3 SE
    weak = S.final_claim(P(-5e-4, 2e-4, 100), P(-6e-4, 2e-4, 100))
    assert not weak.passed and weak.checks == {"significant": False, "consistent_with_validation": True}
    # inconsistent with validation: |(-6e-4) - (-2e-3)| = 1.4e-3 > 2 * hypot(1e-4, 2e-4) = 4.5e-4
    shrunk = S.final_claim(P(-6e-4, 1e-4, 100), P(-2e-3, 2e-4, 100))
    assert not shrunk.passed and shrunk.checks == {"significant": True, "consistent_with_validation": False}
    # ... unless the validation gain is first corrected for its selection optimism
    fixed = S.final_claim(P(-6e-4, 1e-4, 100), P(-2e-3, 2e-4, 100), val_selection_bias=-1.2e-3)
    assert fixed.passed and fixed.numbers["delta_val_corrected"] == pytest.approx(-8e-4)
    # a loss, however significant, is not a win; boundary of the consistency window is inclusive
    assert not S.final_claim(P(9e-4, 2e-4, 100), P(9e-4, 2e-4, 100)).passed
    edge = S.final_claim(P(-1e-3, 3e-4, 100), P(-1e-3 - 2 * 5e-4, 4e-4, 100))
    assert edge.checks["consistent_with_validation"]
    assert math.isnan(S.final_claim(P(-1e-3, 1e-4, 100), P(0.0, 1e-4, 100)).numbers["shrinkage"])


def test_rules_work_on_per_event_arrays_end_to_end():
    rng = np.random.default_rng(12)
    n = 15000
    base = 18.1 + 3.0 * rng.standard_normal(n)
    ref = base + 0.02 * rng.standard_normal(n)
    ctrl = ref - 2e-4 + 0.01 * rng.standard_normal(n)
    cand = ref - 1.2e-3 + 0.02 * rng.standard_normal(n)
    d = S.track_b_promotion(S.paired(cand, ref), S.paired(cand, ctrl))
    assert d.passed and d.numbers["delta_reference"] == pytest.approx(-1.2e-3, abs=6e-4)
    assert S.track_a_kept(S.paired(cand, ref)).passed and not S.track_a_kept(S.paired(ref, cand)).passed
    adj = S.holm([S.p_from_z(S.paired(c, ref).z) for c in (cand, ctrl, ref + 0.01 * rng.standard_normal(n))])
    assert adj[0] < 0.01 and adj[2] > 0.05


def test_guard_persistence():
    """G-K / G-bulk: sign holds and >= 50 % of the gain persists; the verdict is also given with its SE."""
    n = 20000
    pm = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)                      # mean exactly 0
    pm2 = np.where((np.arange(n) % 4) < 2, 1.0, -1.0)                    # orthogonal to pm, mean exactly 0
    main = -1.0e-3 + 0.02 * pm                                           # gain -1.0e-3, SE 0.02/sqrt(n-1) = 1.4e-4
    keep = S.guard_persistence(main, 0.8 * main + 0.002 * pm2)           # 80 % persists
    assert keep.passed and keep.numbers["fraction"] == pytest.approx(0.8) and keep.flags["resolved"]
    assert keep.numbers["margin"] == pytest.approx(-3e-4) and keep.numbers["z_margin"] < -2
    lost = S.guard_persistence(main, 0.2 * main + 0.002 * pm2)           # only 20 % persists
    assert not lost.passed and lost.checks == {"main_is_gain": True, "sign_holds": True, "fraction": False}
    assert lost.numbers["fraction"] == pytest.approx(0.2) and lost.flags["resolved"]
    flip = S.guard_persistence(main, -0.5 * main)                        # the sign flips under the guard metric
    assert not flip.passed and not flip.checks["sign_holds"] and not flip.checks["fraction"]
    none = S.guard_persistence(-main, -0.9 * main)                       # no gain to persist
    assert not none.passed and not none.checks["main_is_gain"]
    # a nominal pass that is inside the noise: fraction 0.55 with a guard difference much noisier than the margin
    weak = S.guard_persistence(main, 0.55 * main + 0.5 * pm2)
    assert weak.passed and not weak.flags["resolved"] and abs(weak.numbers["z_margin"]) < 2
    assert weak.numbers["se_margin"] == pytest.approx(S.paired(0.55 * main + 0.5 * pm2, 0.5 * main).se)
    assert S.guard_persistence(main, 0.5 * main).checks["fraction"]     # exactly the minimum fraction passes
    with pytest.raises(ValueError):
        S.guard_persistence(main, main[:-1])
    with pytest.raises(ValueError):
        S.guard_persistence(main, np.where(np.arange(n) == 3, np.nan, main))
