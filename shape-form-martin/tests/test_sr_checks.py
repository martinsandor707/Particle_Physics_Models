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


def test_baseline_tail_slopes_summary():
    z = np.linspace(-193.5, 1443.5, 128)
    r = 700.0 * ((np.arange(32) + 0.5) / 32) ** 1.7
    lr = -3.2 * np.log(r)[None, :] - z[:, None] / 250.0
    t = K.baseline_tail_slopes(lr, z, r)
    assert t["dlnr"] == pytest.approx(-3.2) and t["dz_per_mm"] == pytest.approx(-1 / 250.0)
    assert t["r_fit_mm"][1] == pytest.approx(r[-2]) and t["n_fit"] == 4
    both = np.stack([lr, -2.4 * np.log(r)[None, :] - z[:, None] / 400.0])      # the worse of two momenta counts
    t2 = K.baseline_tail_slopes(both, z, r)
    assert t2["dlnr"] == pytest.approx(-2.4) and t2["dz_per_mm"] == pytest.approx(-1 / 400.0)
    with pytest.raises(ValueError):
        K.baseline_tail_slopes(lr[:, :3], z, r[:3])


# ----------------------------------------------------------------------------------------------------------------------
# the tail criterion of a baseline, decided once from the baseline alone (PLAN 17.11, S4)
# ----------------------------------------------------------------------------------------------------------------------
Z = np.linspace(-193.5, 1443.5, 128)
RN = np.exp(np.linspace(np.log(150.0), np.log(1500.0), 41))
MOM = np.array(K.CHECK_MOMENTA_GEV)
Z90 = np.array([120.0, 100.0, 380.0, 660.0, 850.0, 900.0, 950.0])


def baseline(kind):
    """ln rho (P, NZ, NR) of three synthetic baselines: "steep" r^-3.2; "halo" with a far halo exp(-r/3000)/(r + 10 mm)
    like the legacy formula; "rising" = halo with a density that rises with depth beyond z_90; "bump" = halo with
    a radial bump at 600 mm (not monotone in r)."""
    zz, rr = np.meshgrid(Z, RN, indexing="ij")
    if kind == "steep":
        lr = -3.2 * np.log(rr) - zz / 250.0
    else:
        lr = np.log(1e7 * rr ** -4.5 + np.exp(-rr / 3000.0) / (rr + 10.0)) - zz / 250.0
        if kind == "rising":
            lr = lr + 1.5 / (1.0 + np.exp(-(zz - 1200.0) / 60.0))
        if kind == "bump":
            lr = lr + 0.8 * np.exp(-((np.log(rr) - np.log(600.0)) / 0.1) ** 2)
    return np.stack([lr + 0.01 * j for j in range(len(MOM))])


def record(kind, **kw):
    kw.setdefault("reference_kind", "synthetic")
    return K.decide_tail_criterion(baseline(kind), Z, RN, MOM, Z90, baseline_hash=f"hash-of-{kind}",
                                   label=f"synthetic {kind} baseline", **kw)


def test_tail_criterion_is_decided_from_the_baseline_alone():
    steep, halo = record("steep"), record("halo")
    assert steep["criterion"]["radial"] == "loglog_slope_le_limit" and steep["radial"]["baseline_satisfies_limit"]
    assert steep["radial"]["baseline_max"] == pytest.approx(-3.2) and steep["radial"]["baseline_monotone"]
    assert halo["criterion"]["radial"] == "monotone_and_no_flattening"
    assert not halo["radial"]["baseline_satisfies_limit"] and halo["radial"]["baseline_monotone"]
    assert -1.6 < halo["radial"]["baseline_max"] < -1.0                    # the halo exp(-r/R)/(r + 1 cm) is flatter
    assert 0.3 < halo["radial"]["fraction_above_limit"] <= 1.0
    for rec in (steep, halo):
        assert rec["schema"] == "sfm.sr.tail_criterion/2" and rec["reference_kind"] == "synthetic"
        assert rec["criterion"]["depth"] == "inherit_reference" and rec["depth"]["baseline_falls_everywhere"]
        assert rec["depth"]["baseline_max"] == pytest.approx(-1 / 250.0) and rec["depth"]["n_rising_cells"] == 0
        assert rec["depth"]["rising_cells"] == [] and all(g["n_cells"] == 0 for g in rec["depth"]["rising_regions"])
        assert rec["momenta_GeV"] == list(K.CHECK_MOMENTA_GEV) and rec["slope_limit"] == -2.0
        assert rec["r_nodes_mm"][0] < 200.0 < rec["r_nodes_mm"][1] and rec["r_nodes_mm"][-1] == pytest.approx(1500.0)
        assert rec["covers_r_max"] and rec["n_depths"] == 128 and "not of the ring mass" in rec["density"]
        assert np.array(rec["radial"]["loglog_slope"]).shape == (7, 128, len(rec["r_nodes_mm"]) - 1)   # full tables
        assert np.array(rec["depth"]["dz_per_mm"]).shape == (7, 127, 41) and len(rec["z_nodes_mm"]) == 128
        assert [Z[k] > z90 >= Z[k - 1] for k, z90 in zip(rec["depth"]["k0"], Z90)] == [True] * 7
        assert np.array(rec["depth"]["sign_changes"]).shape == (7, 41) and np.array(rec["radial"]["sign_changes"]).shape == (7, 128)
        assert np.sum(rec["depth"]["sign_changes"]) == 0 and np.sum(rec["radial"]["sign_changes"]) == 0
        assert np.array(rec["depth"]["rising_stretches"]).shape == (7, 41) and np.sum(rec["depth"]["rising_stretches"]) == 0
        assert np.array(rec["radial"]["rising_stretches"]).shape == (7, 128) and "rising_stretches(model) <=" in rec["criterion"]["stage2"]
        assert "not selectable per equation" in rec["criterion"]["decided_from"]
        assert len(__import__("json").dumps(rec)) < 2_000_000
    txt = K.format_tail_criterion(halo)
    assert "3-D density" in txt and "monotone_and_no_flattening" in txt and "hash-of-halo"[:16] in txt
    short = K.decide_tail_criterion(baseline("steep")[:, :, :20], Z, RN[:20], MOM, Z90, baseline_hash="h", label="map")
    assert not short["covers_r_max"] and short["reference_kind"] == "track_a_finalist"     # the default reference
    with pytest.raises(ValueError, match="reference_kind"):
        record("steep", reference_kind="whatever")


def test_density_not_ring_mass():
    """rho ~ r^-2.5 is integrable in r dr (slope -2.5 <= -2); its ring mass 2 pi r rho has slope -1.5. The rule is
    stated for rho: feeding the ring mass by mistake flips the decision."""
    zz, rr = np.meshgrid(Z, RN, indexing="ij")
    lr = np.stack([-2.5 * np.log(rr) - zz / 250.0] * len(MOM))
    a = K.decide_tail_criterion(lr, Z, RN, MOM, Z90, baseline_hash="h", label="rho")
    b = K.decide_tail_criterion(lr + np.log(2 * np.pi * rr)[None], Z, RN, MOM, Z90, baseline_hash="h", label="ring mass")
    assert a["radial"]["baseline_max"] == pytest.approx(-2.5) and a["criterion"]["radial"] == "loglog_slope_le_limit"
    assert b["radial"]["baseline_max"] == pytest.approx(-1.5) and b["criterion"]["radial"] == "monotone_and_no_flattening"
    c = K.decide_tail_criterion(lr + np.arange(len(MOM))[:, None, None] * 3.0, Z, RN, MOM, Z90, baseline_hash="h", label="rho")
    assert c["criterion"] == a["criterion"]                                 # a normalisation per momentum is irrelevant


def test_tail_record_is_bound_to_its_baseline_and_cannot_be_edited():
    import copy
    halo = record("halo")
    K.verify_tail_criterion(halo)
    K.verify_tail_criterion(halo, "hash-of-halo")
    with pytest.raises(ValueError, match="belongs to baseline"):
        K.verify_tail_criterion(halo, "hash-of-steep")
    forged = copy.deepcopy(halo)
    forged["criterion"]["radial"] = "loglog_slope_le_limit"
    with pytest.raises(ValueError, match="altered"):
        K.verify_tail_criterion(forged)
    forged["record_hash"] = K._tail_record_hash(forged)                     # even with a fresh hash
    with pytest.raises(ValueError, match="does not follow"):
        K.verify_tail_criterion(forged)
    bent = copy.deepcopy(halo)                                               # raising one stored slope: the bound moves
    bent["depth"]["dz_per_mm"][3][100][5] = 0.01
    with pytest.raises(ValueError, match="altered"):
        K.verify_tail_criterion(bent)
    moved = copy.deepcopy(halo)
    moved["baseline_hash"] = "hash-of-steep"
    with pytest.raises(ValueError, match="altered"):
        K.verify_tail_criterion(moved, "hash-of-steep")
    with pytest.raises(ValueError, match="baseline_hash is required"):
        K.decide_tail_criterion(baseline("steep"), Z, RN, MOM, Z90, baseline_hash="", label="x")
    with pytest.raises(ValueError):
        K.decide_tail_criterion(baseline("steep"), Z, RN, MOM, np.full(7, 5000.0), baseline_hash="h", label="x")


def test_a_refitted_baseline_can_never_be_the_reference():
    """The user's anchoring rule: the bound comes from the fixed non-SR baseline, so a joint refit cannot bend it."""
    refit_table = record("rising", reference_kind=K.MEASURED_KIND)
    assert refit_table["reference_kind"] == "refitted_baseline"
    K.verify_tail_criterion(refit_table, as_reference=False)                # fine on the model side
    with pytest.raises(ValueError, match="cannot be the reference"):
        K.verify_tail_criterion(refit_table)
    with pytest.raises(ValueError, match="cannot be the reference"):
        K.verify_tail_criterion(refit_table, "hash-of-rising")
    for kind in K.REFERENCE_KINDS:
        K.verify_tail_criterion(record("halo", reference_kind=kind))
    assert K.REFERENCE_KINDS[:2] == ("track_a_finalist", "c1_refit_only")


def test_where_the_reference_rises_is_logged():
    """Every cell in which the reference itself rises beyond z_90 is in the record, with its size, and summarised
    per momentum; annotations (which component) are stored with the region."""
    rec = record("rising", rising_annotations={2: {"removed_by": ["track"], "comment": "Bragg"}})
    de = rec["depth"]
    assert not de["baseline_falls_everywhere"] and de["n_rising_cells"] == len(de["rising_cells"]) > 0
    assert de["baseline_max"] == pytest.approx(1.5 / 240.0 - 1 / 250.0, rel=0.02)   # +0.0022 / mm
    dz = np.array(de["dz_per_mm"], dtype=float)
    for j, k, q, v in de["rising_cells"][:50] + de["rising_cells"][-50:]:
        assert v > 0 and v == pytest.approx(dz[j, k, q], rel=1e-8) and k >= de["k0"][j]
    beyond = np.zeros(dz.shape, bool)
    for j in range(7):
        beyond[j, de["k0"][j]:] = True
    assert de["n_rising_cells"] == int(np.sum(beyond & (dz > 0)))           # complete
    for reg in de["rising_regions"]:
        assert reg["n_cells"] > 0 and 1100.0 < reg["z_range_mm"][0] < 1200.0 < reg["z_range_mm"][1] < 1300.0
        assert reg["r_range_mm"] == [pytest.approx(RN[0]), pytest.approx(RN[-1])] and 0.2 < reg["max_slope_per_100mm"] < 0.23
        assert 1100.0 < reg["max_at"]["z_mm"][0] < 1300.0 and reg["total_rise_at_that_radius_nat"] > 0.1
    assert de["rising_regions"][2]["attribution"] == {"removed_by": ["track"], "comment": "Bragg"}
    assert "attribution" not in de["rising_regions"][0]
    assert np.all(np.array(de["sign_changes"]) == 2)                         # one bump per depth profile: - + -
    assert np.all(np.array(de["rising_stretches"]) == 1)                     # ... which is one rising stretch
    assert np.all(np.array(record("bump")["radial"]["rising_stretches"]) == 1)
    assert rec["radial"]["baseline_monotone"] and not record("bump")["radial"]["baseline_monotone"]
    assert np.all(np.array(record("bump")["radial"]["sign_changes"]) == 2)
    txt = K.format_tail_criterion(rec)
    assert "the reference itself rises in" in txt and "removed by leaving out: ['track']" in txt
    K.verify_tail_criterion(rec)


def test_profile_sign_changes_are_reproducible():
    """Counting rule of the Stage-2 condition: fixed grid, lobes below the tolerance dropped, zeros skipped."""
    x = np.linspace(0, 4 * np.pi, 400)
    assert K.profile_sign_changes(np.sin(x)[1:-1], 1e-3) == 3
    assert K.profile_sign_changes(-np.ones(50), 1e-3) == 0
    ripple = -1.0 + 1e-4 * np.sin(40 * x)                                   # never changes sign
    assert K.profile_sign_changes(ripple, 1e-3) == 0
    tiny = 1e-4 * np.sin(x)                                                 # all lobes below the tolerance
    assert K.profile_sign_changes(tiny, 1e-3) == 0 and K.profile_sign_changes(tiny, 1e-5) == 3
    v = np.array([-1.0, 0.0, np.nan, 2e-4, -1.0, 1.0, 0.0, 1.0, -1.0])      # the 2e-4 lobe is dropped at tol 1e-3
    assert K.profile_sign_changes(v, 1e-3) == 2 and K.profile_sign_changes(v, 1e-5) == 4
    arr = np.stack([np.sin(x), -np.ones_like(x), np.cos(x)])
    assert K.profile_sign_changes(arr, 1e-3).tolist() == [3, 0, 4]
    assert K.SIGN_TOL_DEPTH_PER_MM == 1e-5 and K.SIGN_TOL_RADIAL == 1e-3
    # rising stretches: the same lobes, counting the positive ones - no edge effect at the start of a profile
    assert K.profile_rising_lobes(np.sin(x)[1:-1], 1e-3) == 2 and K.profile_rising_lobes(-np.ones(50), 1e-3) == 0
    assert K.profile_rising_lobes(np.array([1.0, 1.0, -1.0, -1.0]), 1e-3) == 1       # + -   : one sign change
    assert K.profile_rising_lobes(np.array([-1.0, 1.0, -1.0, -1.0]), 1e-3) == 1      # - + - : two sign changes
    assert K.profile_sign_changes(np.array([1.0, 1.0, -1.0, -1.0]), 1e-3) == 1
    assert K.profile_sign_changes(np.array([-1.0, 1.0, -1.0, -1.0]), 1e-3) == 2
    assert K.profile_rising_lobes(tiny, 1e-3) == 0 and K.profile_rising_lobes(arr, 1e-3).tolist() == [2, 0, 3]
