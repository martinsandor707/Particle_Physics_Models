"""sfm.families.lateral: survival functions, radial densities, nesting, mixtures (CPU, no data)."""
import math

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from sfm.families import cellint as ci
from sfm.families import lateral as lat
from sfm.families.longitudinal import raw_of

F64 = torch.float64
R = np.array([1e-3, 0.4, 1.37, 5.0, 16.9, 60.0, 180.0, 420.0, 700.0])


def T(v, dt=F64):
    return torch.tensor(v, dtype=dt)


def _cases():
    """(name, family, natural shapes, scale, scipy distribution of the radial distance)."""
    return [
        ("gauss", lat.GaussLat(), {}, 34.0, stats.rayleigh(scale=34.0)),
        ("cauchy", lat.CauchyLat(), {}, 79.0, stats.burr12(c=2.0, d=1.0, scale=79.0)),
        ("student k=1.7", lat.BurrLat(c=2.0), {"c": 2.0, "k": 1.7}, 60.0, stats.burr12(c=2.0, d=1.7, scale=60.0)),
        ("burr c=1.3 k=2.2", lat.BurrLat(), {"c": 1.3, "k": 2.2}, 45.0, stats.burr12(c=1.3, d=2.2, scale=45.0)),
        ("burr c=3 k=0.4", lat.BurrLat(), {"c": 3.0, "k": 0.4}, 120.0, stats.burr12(c=3.0, d=0.4, scale=120.0)),
        ("exp", lat.ExpLat(), {}, 80.0, stats.expon(scale=80.0)),
        ("weibull b=0.55", lat.WeibullLat(1.0), {"b": 0.55}, 150.0, stats.weibull_min(c=0.55, scale=150.0)),
        ("weibull b=1.6", lat.WeibullLat(1.0), {"b": 1.6}, 90.0, stats.weibull_min(c=1.6, scale=90.0)),
        ("weibull_core b=2.6", lat.WeibullLat(0.5), {"b": 2.6}, 30.0, stats.weibull_min(c=2.6, scale=30.0 * 2 ** (1 / 2.6))),
    ]


@pytest.mark.parametrize("name,fam,sh,scale,dist", _cases(), ids=[c[0] for c in _cases()])
def test_survival_density_and_median_against_scipy(name, fam, sh, scale, dist):
    shp = {k: T(v) for k, v in sh.items()}
    sc = T([scale])
    assert np.allclose(fam.log_sf(T(R), sc, shp).numpy(), dist.logsf(R), rtol=1e-11, atol=1e-13)
    assert np.allclose(fam.log_dEdr(T(R), sc, shp).numpy(), dist.logpdf(R), rtol=1e-11, atol=1e-11)
    area = fam.log_pdf_area(T(R), sc, shp).numpy()                            # T = (dE/dr) / (2 pi r)
    assert np.allclose(area, dist.logpdf(R) - np.log(2 * np.pi * R), rtol=1e-10, atol=1e-10)
    med = scale * math.exp(float(fam.log_median(shp)))
    assert abs(dist.sf(med) - 0.5) < 1e-12
    assert float(fam.log_sf(T([0.0]), sc, shp)) == 0.0                        # S(0) = 1 exactly
    # -dS/dr = dE/dr by automatic differentiation
    r = T(R[1:]).requires_grad_(True)
    (g,) = torch.autograd.grad(fam.log_sf(r, sc, shp).exp().sum(), r)
    assert np.allclose(-g.numpy(), dist.pdf(R[1:]), rtol=1e-9, atol=1e-300)


def test_nesting_of_the_radial_families():
    r, sc = T(R), T([55.0])
    burr, cau = lat.BurrLat(), lat.CauchyLat()
    assert torch.allclose(burr.log_sf(r, sc, {"c": T(2.0), "k": T(1.0)}), cau.log_sf(r, sc, {}), rtol=1e-13, atol=1e-15)
    assert torch.allclose(burr.log_pdf_area(r, sc, {"c": T(2.0), "k": T(1.0)}), cau.log_pdf_area(r, sc, {}), rtol=1e-12, atol=1e-12)
    wb, ex = lat.WeibullLat(1.0), lat.ExpLat()
    assert torch.allclose(wb.log_sf(r, sc, {"b": T(1.0)}), ex.log_sf(r, sc, {}), rtol=1e-13)
    assert torch.allclose(wb.log_pdf_area(r, sc, {"b": T(1.0)}), ex.log_pdf_area(r, sc, {}), rtol=1e-12, atol=1e-12)
    wc, ga = lat.WeibullLat(0.5), lat.GaussLat()
    assert torch.allclose(wc.log_sf(r, sc, {"b": T(2.0)}), ga.log_sf(r, sc, {}), rtol=1e-13)
    assert torch.allclose(wc.log_pdf_area(r, sc, {"b": T(2.0)}), ga.log_pdf_area(r, sc, {}), rtol=1e-12, atol=1e-12)
    # the default raw shapes are the nested points: burr -> cauchy, weibull -> exp, weibull_core -> gauss
    for fam, want in ((lat.make_lat("burr"), {"c": 2.0, "k": 1.0}), (lat.make_lat("student"), {"c": 2.0, "k": 1.0}),
                      (lat.make_lat("weibull"), {"b": 1.0}), (lat.make_lat("weibull_core"), {"b": 2.0})):
        sh = fam.shapes({n: T(v) for n, v in fam.SHAPES})
        for k, v in want.items():
            assert abs(float(sh[k]) - v) < 1e-12, (fam.name, k)


def test_legacy_area_densities_are_reproduced():
    """lat_gauss, lat_cauchy and lat_exp_over_r of shape-form-proton/families.py (formulas restated here)."""
    r = T(R)
    s = T([34.0])
    lg = -r * r / (2 * s * s) - math.log(2 * math.pi) - 2 * torch.log(s)
    lc = 2 * torch.log(s) - math.log(math.pi) - 2 * torch.log(r * r + s * s)
    for eps in (1.0, 10.0):
        le = -r / s - math.log(2 * math.pi) - torch.log(s) - torch.log(r + eps)
        assert torch.allclose(lat.ExpRegLat(eps).log_pdf_area(r, s, {}), le, rtol=1e-14, atol=1e-14)
        f = lambda x: 2 * math.pi * x * math.exp(float(lat.ExpRegLat(eps).log_pdf_area(T([x]), s, {})))   # noqa: E731
        assert abs(math.exp(float(lat.ExpRegLat(eps).log_dEdr(T([7.0]), s, {}))) - f(7.0)) < 1e-15
    assert torch.allclose(lat.GaussLat().log_pdf_area(r, s, {}), lg, rtol=1e-14, atol=1e-13)
    assert torch.allclose(lat.CauchyLat().log_pdf_area(r, s, {}), lc, rtol=1e-14, atol=1e-13)
    with pytest.raises(NotImplementedError):
        lat.ExpRegLat().log_sf(r, s, {})


@pytest.mark.parametrize("dt", [torch.float32, F64])
def test_finite_values_and_gradients_at_extreme_scales_and_shapes(dt):
    """Scales from 6e-6 mm to 1.6e15 mm (the clamp of the scale link) and saturated shape links."""
    g = ci.Grid(dtype=dt)
    for name in lat.LAT_FAMILIES:
        fam = lat.make_lat(name)
        raws = [-1e3, -12.0, -3.0, 0.0, 4.0, 12.0, 35.0, 1e3]
        log_scale = torch.tensor(raws, dtype=dt).reshape(-1, 1).requires_grad_(True)
        for shape_raw in (-80.0, 0.0, 80.0):
            sraw = {n: torch.full((len(raws), 1), shape_raw, dtype=dt, requires_grad=True) for n, _ in fam.SHAPES}
            sc, sh = lat._scale(log_scale), fam.shapes(sraw)
            ring = fam.ring_logmass(g, sc, sh)
            disc = fam.log_disc_mass(g, sc, sh, ring)
            cen = fam.log_pdf_area(g.rc, sc, sh)
            for out in (ring, disc, cen):
                assert torch.isfinite(out).all(), (name, shape_raw)
            w = torch.softmax(torch.randn(len(raws), 32, generator=torch.Generator().manual_seed(1)).to(dt), 1)
            loss = (torch.log_softmax(ring, 1) * w).sum() + (torch.log_softmax(cen + g.log_area, 1) * w).sum() + disc.sum()
            grads = torch.autograd.grad(loss, [log_scale] + list(sraw.values()), allow_unused=True)
            assert all(x is None or torch.isfinite(x).all() for x in grads), (name, shape_raw)


# ----------------------------------------------------------------------------------------------------------------------
# mixtures
# ----------------------------------------------------------------------------------------------------------------------
def _th(mix, B=3, seed=0, **over):
    gen = torch.Generator().manual_seed(seed)
    th = {n: torch.full((B,), float(v), dtype=F64) + 0.3 * torch.randn(B, dtype=F64, generator=gen) for n, v in mix.defaults.items()}
    for k, v in over.items():
        th[k] = torch.full((B,), float(v), dtype=F64)
    return th


@pytest.mark.parametrize("families", [("gauss", "cauchy", "exp_reg"), ("gauss", "cauchy", "exp"), ("gauss", "burr", "weibull"),
                                      ("weibull_core", "student", "weibull")])
@pytest.mark.parametrize("disc_norm", ["mixture", "term", "none"])
def test_mixture_weights_and_disc_normalisation(families, disc_norm):
    g = ci.Grid(n_r=8, dtype=F64)
    mix = lat.legacy_shower_lateral(families, curv=True, disc_norm=disc_norm)
    th = _th(mix, s_sig=2.9, s_R=-1.9, s_lam=2.4, s_1=-1.9, s_2=-1.2, q_sig=-1.9, q_1=-1.6, q_2=1.7)
    zk = g.z_nodes[::40] / 1000
    lw = mix.log_weights(th, zk)
    assert torch.allclose(torch.logsumexp(torch.stack([w.expand(3, len(zk), 1) for w in lw]), 0).exp(),
                          torch.ones(3, len(zk), 1, dtype=F64), atol=1e-13)
    ring = mix.ring_logmass(th, zk, g)
    assert ring.shape == (3, len(zk), 32) and torch.isfinite(ring).all()
    tot = ring.exp().sum(-1)
    if disc_norm == "none":                                                    # plane-normalised: mass inside 700 mm
        want = sum(w.exp() * t.family.log_disc_mass(g, sc, sh).exp() for w, t, (sc, sh) in zip(lw, mix.terms, mix.scales(th, zk)))
        assert torch.allclose(tot, want[..., 0].expand_as(tot), rtol=1e-9)
        assert (tot < 1).all()
    else:
        assert torch.allclose(tot, torch.ones_like(tot), atol=1e-12)
    cen = mix.log_area_density(th, zk, g)
    if disc_norm != "none":                                                    # centre rule normalised with the ring areas
        assert torch.allclose((cen + g.log_area).exp().sum(-1), torch.ones(3, len(zk), dtype=F64), atol=1e-12)
    # depth dependence of the raw parameters: exp(log_sig + s_sig zk + q_sig zk^2)
    sig = mix.scales(th, zk)[0][0]
    assert torch.allclose(sig[:, :, 0], torch.exp(th["log_sig"][:, None] + 2.9 * zk[None] - 1.9 * zk[None] ** 2), rtol=1e-13)


def test_term_normalisation_makes_the_weights_literal_disc_fractions():
    g = ci.Grid(dtype=F64)
    mix = lat.legacy_shower_lateral(("gauss", "cauchy", "exp"), curv=False, disc_norm="term")
    th = _th(mix, log_sig=math.log(40.0), log_R=math.log(300.0), log_lam=math.log(400.0), s_sig=0, s_R=0, s_lam=0, s_1=0, s_2=0,
             logit_1=0.3, logit_2=-0.4)
    zk = T([0.1])
    f1 = 1 / (1 + math.exp(-0.3))
    # with one term switched off in turn the ring masses are those of the renormalised remaining terms
    ring = mix.ring_logmass(th, zk, g).exp()[0, 0]
    parts = []
    for t, (sc, sh) in zip(mix.terms, mix.scales(th, zk)):
        m = t.family.ring_logmass(g, sc, sh).exp()[0, 0]
        parts.append(m / m.sum())
    f2 = (1 - f1) / (1 + math.exp(0.4))
    want = f1 * parts[0] + f2 * parts[1] + (1 - f1 - f2) * parts[2]
    assert torch.allclose(ring, want, rtol=1e-12)


def test_ordered_mixture_has_increasing_median_radii_at_every_depth():
    g = ci.Grid(dtype=F64)
    for fams in (("gauss", "cauchy", "exp"), ("burr", "burr", "burr"), ("weibull_core", "student", "weibull")):
        mix = lat.legacy_shower_lateral(fams, curv=True, disc_norm="term", ordered=True)
        gen = torch.Generator().manual_seed(3)
        th = {n: 3.0 * torch.randn(40, dtype=F64, generator=gen) for n in mix.names}     # arbitrary raw values
        zk = g.zc / 1000
        med = [m.expand(40, len(zk), 1) for m in mix.median_radii(th, zk)]
        # exact wherever no scale sits on the clamp of the link (6e-6 mm ... 1.6e15 mm)
        free = torch.stack([(sc > 1e-5) & (sc < 1e14) for sc, _ in mix.scales(th, zk)]).all(0).expand(40, len(zk), 1)
        assert free.double().mean() > 0.5
        assert (med[1] >= med[0])[free].all() and (med[2] >= med[1])[free].all()
        assert ((med[1] > med[0])[free].double().mean() > 0.99) and ((med[2] > med[1])[free].double().mean() > 0.99)
        # the medians are medians: S(r50) = 1/2 for every term
        for t, (sc, sh), m in zip(mix.terms, mix.scales(th, zk), med):
            if t.family.closed_form:
                ok = (sc > 1e-5) & (sc < 1e14)                                 # away from the clamp of the scale link
                s_at = t.family.log_sf(m, sc.expand(40, len(zk), 1), sh).exp()
                assert torch.allclose(s_at[ok.expand_as(s_at)], torch.full_like(s_at, 0.5)[ok.expand_as(s_at)], rtol=1e-9)
        ring = mix.ring_logmass(th, zk, g)
        assert torch.isfinite(ring).all() and torch.allclose(ring.exp().sum(-1), torch.ones(40, len(zk), dtype=F64), atol=1e-11)


def test_unordered_legacy_fit_is_not_ordered():
    """Documented limitation: in the legacy fit the Gaussian 'core' is the widest term at depth (z' = 600 mm, 5 GeV:
    sigma = 303 mm -> r50 = 357 mm against R = 111 mm), so the ordered parametrisation does not contain that fit."""
    r50_gauss = 303.0 * math.exp(lat.GaussLat().log_median({}))
    assert abs(r50_gauss - 303.0 * math.sqrt(2 * math.log(2))) < 1e-9 and r50_gauss > 111.0


def test_generic_names_and_single_term_mixture():
    term = lat.LatTerm("burr", ("c0_log_r", "c0_s_r", None), "c0_lat_", math.log(25.0))
    mix = lat.LateralMixture([term], [], disc_norm="mixture")
    assert mix.names == ["c0_log_r", "c0_s_r", "c0_lat_rc", "c0_lat_rk"]
    assert abs(mix.defaults["c0_lat_rc"] - raw_of(2.0, 0.3)) < 1e-12 and mix.defaults["c0_s_r"] == 0.0
    g = ci.Grid(dtype=F64)
    th = {n: torch.full((2,), float(v), dtype=F64) for n, v in mix.defaults.items()}
    ring = mix.ring_logmass(th, g.zc[:5] / 1000, g)
    assert ring.shape == (2, 5, 32) and torch.allclose(ring.exp().sum(-1), torch.ones(2, 5, dtype=F64), atol=1e-12)
    with pytest.raises(ValueError):
        lat.LateralMixture([term, term], [], "mixture")
    with pytest.raises(KeyError):
        lat.make_lat("k0")


def test_centre_rule_error_of_singular_axis_densities():
    """The hazard stated in the module docstring: for T ~ 1/r the centre rule misses the first ring by tens of per
    cent, the survival difference is exact."""
    g = ci.Grid(dtype=F64)
    lam = T([80.0])
    exact = lat.ExpLat().ring_logmass(g, lam, {}).exp()
    centre = (lat.ExpLat().log_pdf_area(g.rc, lam, {}) + g.log_area).exp()
    ref0 = integrate.quad(lambda r: math.exp(-r / 80.0) / 80.0, 0, float(g.r_edges[1]))[0]
    assert abs(float(exact[0]) / ref0 - 1) < 1e-12
    assert abs(float(centre[0]) / ref0 - 1) > 0.25 and abs(float(centre[20]) / float(exact[20]) - 1) < 2e-2
