"""sfm.sr.checks: sign changes, extrema, containment, monotonicity, modulation, period check, seed-null."""
import math

import numpy as np
import pytest

from sfm.sr import checks as K
from sfm.sr import translate as T


def test_sign_changes_ignore_small_lobes():
    x = np.linspace(0, 4 * np.pi, 2001)
    assert K.sign_changes(np.sin(x)[1:-1]) == 3
    v = np.sin(x) * np.where(x > 2 * np.pi, 0.001, 1.0)     # the second period is ripple of 0.001
    assert K.sign_changes(v[1:-1]) == 3 and K.sign_changes(v[1:-1], min_peak=0.003) == 1
    assert K.sign_changes([1, 2, 3]) == 0 and K.sign_changes([]) == 0 and K.sign_changes([0, 0, -1, 0, 1]) == 1


def test_count_extrema_with_prominence():
    x = np.linspace(0, 1, 5001)
    assert K.count_extrema(x) == 0 and K.count_extrema(-(x - 0.4) ** 2) == 1
    assert K.count_extrema(np.sin(6 * np.pi * x)) == 6                       # three periods: six turning points
    ripple = x + 0.004 * np.sin(800 * x)                    # non-monotone, but each dip is only 0.004 deep
    assert K.count_extrema(ripple, 0.01) == 0 and K.count_extrema(ripple, 0.0005) > 100
    assert K.count_extrema(np.sin(6 * np.pi * x) + 0.004 * np.sin(800 * x), 0.01) == 6


def test_containment_and_monotone_sequence():
    z_edges, r_edges = np.linspace(0, 1000, 101), np.linspace(0, 500, 51)
    zc, rc = 0.5 * (z_edges[1:] + z_edges[:-1]), 0.5 * (r_edges[1:] + r_edges[:-1])
    out = []
    for lam in (100.0, 130.0, 170.0):
        P = np.exp(-zc[:, None] / lam) * np.exp(-rc[None, :] / 40.0)
        c = K.containment(P, z_edges, r_edges)
        trunc = 1 - math.exp(-1000 / lam)
        assert c["z50"] == pytest.approx(-lam * math.log(1 - 0.5 * trunc), rel=0.01)
        assert c["r90"] == pytest.approx(-40 * math.log(1 - 0.9 * (1 - math.exp(-12.5))), rel=0.02)
        out.append(c["z90"])
    assert K.monotone_sequence(out)["monotone"] and K.monotone_sequence(out)["direction"] == "up"
    m = K.monotone_sequence([100.0, 103.0, 101.0])
    assert not m["monotone"] and m["direction"] == "reversal" and m["largest_reversal"] == pytest.approx(2.0)
    assert K.monotone_sequence([100.0, 103.0, 102.7])["monotone"]          # a reversal below 0.5 mm is not one
    assert K.monotone_sequence([5.0, 5.2, 5.1], tol=0.5)["direction"] == "flat"


def test_slope_sign_changes():
    r = np.linspace(0, 700, 141)
    assert K.slope_sign_changes(-r / 50.0, r, start=200.0)["n"] == 0
    bump = -r / 50.0 + 1.5 * np.exp(-0.5 * ((r - 450) / 30) ** 2)
    s = K.slope_sign_changes(bump, r, start=200.0)
    assert s["n"] == 2 and s["max_slope"] > 0
    assert K.slope_sign_changes(bump, r, start=520.0)["n"] == 0


def test_period_check_uses_the_corrected_periods():
    assert K.MLP_BASE_PERIOD_MM == 1650.0
    assert K.period_check(1650.0)["verdict"] == "trend" and K.period_check(825.0)["verdict"] == "trend"
    assert K.period_check(600.0)["verdict"] == "trend"                       # above ~570 mm: a trend, not a match
    c = K.period_check(550.0)
    assert c["verdict"] == "match" and c["nearest_k"] == 3 and c["relative_distance"] == 0.0
    assert K.period_check(412.5)["nearest_k"] == 4 and K.period_check(412.5 * 1.02)["verdict"] == "match"
    assert K.period_check(206.25)["nearest_k"] == 8 and K.period_check(206.25)["verdict"] == "match"
    assert K.period_check(480.0)["verdict"] == "no match" and K.period_check(1100.0)["verdict"] == "trend"
    assert K.period_check(150.0)["verdict"] == "no match"                    # below 1650/8
    cov = K.period_band_coverage()
    assert 0.25 < cov < 0.40                                                 # weak specificity: about a third
    assert K.period_band_coverage(rel_tol=0.05) > cov


def test_trig_periods_from_the_phase():
    w = 2 * math.pi * 260.0 / 550.0                         # phase = w * zt: period 550 mm in z'
    e = T.parse(f"tanh(zt) + 0.2 * sin(zt * {w!r} + x)")
    (p,) = K.trig_periods(e)
    lo, hi = p["periods_mm"]["zt"]
    assert lo == pytest.approx(550.0, rel=1e-9) and hi == pytest.approx(550.0, rel=1e-9) and "rh" not in p["periods_mm"]
    assert K.period_check(lo)["verdict"] == "match"
    e = T.parse("0.1 * cos(rh * 0.5 + zt * 0.0)")
    (p,) = K.trig_periods(e)
    assert p["periods_mm"]["rh"][0] == pytest.approx(2 * math.pi * 21.9 / 0.5) and p["periods_mm"]["zt"] == (math.inf, math.inf)
    e = T.parse("sin(l1p(zt) * 6.0)")                       # a chirp: the period varies, the enclosure is a range
    (p,) = K.trig_periods(e)
    assert p["periods_mm"]["zt"][0] < p["periods_mm"]["zt"][1]


def test_modulation_of_free_form_and_template():
    e = T.parse("0.6 * tanh(zt - 1.0) + 0.2 * sin(zt * 3.0) * linv(rh)")
    m = K.modulation_expr(e)
    zt, rh = np.linspace(-1, 7, 50), np.linspace(0, 60, 50)
    assert np.allclose(T.evaluate(m, zt=zt, rh=rh), 0.2 * np.sin(3 * zt) / (1 + rh ** 2), atol=1e-15)
    r = K.modulation_amplitude(m)
    assert r["passed_amplitude"] and 0.19 < r["amplitude"]["sup_lower"] <= 0.2 and r["amplitude"]["sup_upper"] <= 0.3 + 1e-9
    assert r["sign_changes"]["zt"]["inside"] > 4 and not r["passed_sign_changes"]     # six periods inside the data
    big = K.modulation_amplitude(K.modulation_expr(T.parse("0.6 * tanh(zt) + 0.4 * sin(zt * 0.5)")))
    assert not big["passed_amplitude"] and big["amplitude"]["status"] == "failed"
    sub = {"f": "(#1 * 0.5) - 1.0", "a": "#1 * 50.0", "e": "(#1 - 2.0) * 3.0", "ph": "#1 * 1.2"}
    tm = K.template_modulation(sub)
    assert "sin" in T.to_string(tm) and "tanh(tanh" not in T.to_string(tm)
    r = K.modulation_amplitude(tm)
    assert r["passed_amplitude"] and r["amplitude"]["status"] == "certified" and r["amplitude"]["sup_upper"] <= 0.3 + 1e-9
    assert r["sign_changes"]["zt"]["outside"] == 0          # the envelope 1/(1+e^2) kills it outside


def test_seed_null_comparison():
    ok = K.seed_null_comparison(0.040, 0.010, 0.0020, 0.0018)
    assert ok["passed"] and ok["relative_reduction_target"] == pytest.approx(0.75)
    assert ok["noise_share_of_target_reduction"] == pytest.approx(0.0002 / 0.030)
    bad = K.seed_null_comparison(0.040, 0.039, 0.0020, 0.0005)
    assert not bad["passed"]
    assert K.CHECK_MOMENTA_GEV == (0.05, 0.1, 1.0, 5.0, 20.0, 30.0, 50.0)
