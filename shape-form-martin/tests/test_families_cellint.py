"""sfm.families.cellint: careful primitives and the accuracy of the cell quadratures (CPU, no data, no GPU).

The thresholds are the accuracy statements of the module docstring; each test measures the quantity it bounds.
"""
import math

import mpmath as mp
import numpy as np
import pytest
import torch
from scipy import integrate, special

from sfm.families import cellint as ci
from sfm.families import lateral as lat

mp.mp.dps = 60
F64 = torch.float64


# ----------------------------------------------------------------------------------------------------------------------
# log1mexp / logdiffexp
# ----------------------------------------------------------------------------------------------------------------------
X_EDGE = [-1e-300, -1e-30, -1e-12, -1e-6, -1e-3, -0.5, -math.log(2.0) + 1e-12, -math.log(2.0), -math.log(2.0) - 1e-12,
          -1.0, -5.0, -36.0, -40.0, -700.0, -745.0, -1e5]


def test_log1mexp_matches_multiprecision_over_the_whole_range():
    x = torch.tensor(X_EDGE, dtype=F64)
    got = ci.log1mexp(x).tolist()
    for xi, g in zip(X_EDGE, got):
        ref = mp.log(-mp.expm1(mp.mpf(xi)))
        assert abs(g - float(ref)) <= 4e-16 * max(1.0, abs(float(ref))), (xi, g, float(ref))


def test_log1mexp_float32_relative_accuracy():
    xs = [-1e-30, -1e-10, -1e-4, -0.3, -0.7, -3.0, -20.0, -80.0]
    got = ci.log1mexp(torch.tensor(xs, dtype=torch.float32)).double().tolist()
    for xi, g in zip(xs, got):
        ref = float(mp.log(-mp.expm1(mp.mpf(float(np.float32(xi))))))
        assert abs(g - ref) <= 3e-7 * max(abs(ref), 1e-30), (xi, g, ref)       # relative, also where the result is ~1e-35


def test_log1mexp_edges_and_gradient():
    x = torch.tensor([0.0, -0.0, float("-inf")], dtype=F64)
    y = ci.log1mexp(x)
    assert y[0] == float("-inf") and y[1] == float("-inf") and y[2] == 0.0
    assert torch.isnan(ci.log1mexp(torch.tensor([1e-3], dtype=F64))).all()      # outside the domain
    x = torch.tensor(X_EDGE[1:], dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(ci.log1mexp(x).sum(), x)
    ref = [float(-1 / mp.expm1(-mp.mpf(v))) for v in X_EDGE[1:]]
    assert np.allclose(g.numpy(), ref, rtol=1e-12, atol=1e-300)
    assert torch.isfinite(g).all() and float(g[-1]) == 0.0                      # -> -0 for x -> -inf, never NaN
    xs = torch.tensor([-1e-3, -0.4, -0.9, -6.0], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(ci.log1mexp, (xs,), eps=1e-7, atol=1e-7)
    assert torch.autograd.gradgradcheck(ci.log1mexp, (xs,), eps=1e-7, atol=1e-6)


def test_logdiffexp():
    a = torch.tensor([0.0, -3.0, 500.0, -1e4, 2.0], dtype=F64)
    b = torch.tensor([-1e-9, -3.5, 499.0, -1e4 - 1e-7, -800.0], dtype=F64)
    got = ci.logdiffexp(a, b).tolist()
    for ai, bi, g in zip(a.tolist(), b.tolist(), got):
        ref = float(mp.mpf(ai) + mp.log(-mp.expm1(mp.mpf(bi) - mp.mpf(ai))))
        assert abs(g - ref) <= 1e-9 * max(1.0, abs(ref)), (ai, bi, g, ref)


# ----------------------------------------------------------------------------------------------------------------------
# log Phi and log erfcx: values and tail-stable derivatives
# ----------------------------------------------------------------------------------------------------------------------
def test_log_ndtr_gradient_is_the_inverse_mills_ratio_in_the_far_tails():
    xs = [-2e4, -1000.0, -100.0, -13.5, -5.0, -1.0, 0.0, 1.0, 5.0, 13.5, 40.0]
    mills = [float(mp.npdf(v) / mp.ncdf(v)) for v in xs]
    for dt, tol in ((torch.float32, 3e-6), (F64, 1e-12)):
        x = torch.tensor(xs, dtype=dt, requires_grad=True)
        y = ci.log_ndtr(x)
        (g,) = torch.autograd.grad(y.sum(), x)
        assert np.allclose(g.double().numpy(), mills, rtol=tol, atol=1e-30)
        assert np.allclose(y.detach().double().numpy(), special.log_ndtr(np.array(xs)), rtol=5e-7 if dt == torch.float32 else 1e-13)
    # torch's own float32 derivative fails there (the reason for the custom function)
    x = torch.tensor([-2e4, -1000.0], dtype=torch.float32, requires_grad=True)
    (g,) = torch.autograd.grad(torch.special.log_ndtr(x).sum(), x)
    assert abs(float(g[0]) / 2e4 - 1) > 1.0 and abs(float(g[1]) / 1000.0 - 1) > 0.01


def test_log_erfcx_value_and_gradient():
    ys = [0.0, 0.3, 2.0, 9.9, 10.1, 50.0, 1e3, 2e4]
    val = [float(mp.log(mp.exp(mp.mpf(v) ** 2) * mp.erfc(v))) for v in ys]
    der = [float(2 * mp.mpf(v) - 2 / (mp.sqrt(mp.pi) * mp.exp(mp.mpf(v) ** 2) * mp.erfc(v))) for v in ys]
    for dt, tv, tg in ((torch.float32, 2e-6, 2e-5), (F64, 1e-13, 1e-9)):
        y = torch.tensor(ys, dtype=dt, requires_grad=True)
        out = ci.log_erfcx(y)
        (g,) = torch.autograd.grad(out.sum(), y)
        assert np.allclose(out.detach().double().numpy(), val, rtol=tv, atol=tv)
        assert np.allclose(g.double().numpy(), der, rtol=tg, atol=1e-12), (dt, g.tolist(), der)


# ----------------------------------------------------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------------------------------------------------
def test_grid_matches_common_and_weights_integrate_constants():
    import common as C
    g = ci.Grid()
    for mine, ref in ((g.z_edges, C.Z_EDGES), (g.r_edges, C.R_EDGES), (g.zc, C.ZC), (g.rc, C.RC), (g.log_vol, C.LOG_VOL)):
        assert torch.allclose(mine, ref.cpu(), rtol=3e-7, atol=3e-6)
    gc = ci.Grid.from_common(C)
    for mine, ref in ((gc.z_edges, C.Z_EDGES), (gc.r_edges, C.R_EDGES), (gc.zc, C.ZC), (gc.rc, C.RC), (gc.log_vol, C.LOG_VOL)):
        assert torch.equal(mine, ref.cpu())
    g = ci.Grid(dtype=F64)
    assert g.shape == (128, 32) and abs(g.dz - 12.890625) < 1e-12
    one = torch.zeros(1, 128 * g.n_z, dtype=F64)
    assert torch.allclose(g.integrate_z(one).exp(), torch.full((1, 128), g.dz, dtype=F64), rtol=1e-14)
    assert torch.allclose(g.integrate_zf(torch.zeros(1, 128 * g.n_zf, dtype=F64)).exp(), torch.full((1, 128), g.dz, dtype=F64), rtol=1e-13)
    # radial rule: int_ring dr = ring width, and int 2 pi r dr = ring area
    w = g.ring_logmass_quad(lambda r: torch.zeros_like(r)).exp()
    assert torch.allclose(w, g.r_edges[1:] - g.r_edges[:-1], rtol=1e-6)
    area = g.ring_logmass_quad(lambda r: torch.log(2 * math.pi * r)).exp()
    assert torch.allclose(area, g.log_area.exp(), rtol=1e-5)
    assert torch.allclose(g.lag.sum(1), torch.ones(g.n_zf, dtype=F64), atol=1e-12)    # Lagrange rows sum to 1


def _cell_ref(f, ze):
    return np.array([integrate.quad(f, ze[i], ze[i + 1], epsabs=0, epsrel=1e-13, limit=200)[0] for i in range(len(ze) - 1)])


def _z_err(f, n_z):
    g = ci.Grid(n_z=n_z, dtype=F64)
    got = g.integrate_z(torch.log(torch.tensor(f(g.z_nodes.numpy())))[None])[0].exp().numpy()
    ref = _cell_ref(f, g.z_edges.numpy())
    m = ref > 1e-12 * ref.max()
    return np.abs(got[m] / ref[m] - 1).max(), np.abs(got - ref).sum() / ref.sum()


# (integrand, n_z, bound on the largest relative cell error, bound on the L1 error of the profile)
Z_CASES = [
    ("logistic edge w=4 mm", lambda z: special.expit((z - 21.37) / 4.0) * special.expit((500 - z) / 30.0), 5, 2e-6, 1e-7),
    ("logistic edge w=2 mm", lambda z: special.expit((z - 21.37) / 2.0) * special.expit((500 - z) / 30.0), 5, 2e-4, 1e-5),
    ("logistic edge w=2 mm, 9 nodes", lambda z: special.expit((z - 21.37) / 2.0) * special.expit((500 - z) / 30.0), 9, 2e-7, 1e-8),
    ("plateau w1=60 mm", lambda z: special.expit((z - 15.0) / 60.0) * special.expit((150 - z) / 500.0) * np.exp(-2.2 * z / 1000), 5, 1e-12, 1e-13),
    ("gauss sigma=12.9 mm", lambda z: np.exp(-(z - 5.4) ** 2 / (2 * 12.9 ** 2)), 5, 5e-5, 5e-10),
    ("gauss sigma=6.4 mm", lambda z: np.exp(-(z - 5.4) ** 2 / (2 * 6.4 ** 2)), 5, 3e-3, 1e-6),
    ("gauss sigma=3.2 mm", lambda z: np.exp(-(z - 5.4) ** 2 / (2 * 3.2 ** 2)), 5, 2e-2, 3e-4),
]


@pytest.mark.parametrize("name,f,n_z,tol_max,tol_l1", Z_CASES, ids=[c[0] for c in Z_CASES])
def test_z_quadrature_accuracy_statements(name, f, n_z, tol_max, tol_l1):
    emax, el1 = _z_err(f, n_z)
    assert emax < tol_max and el1 < tol_l1, (name, emax, el1)


def test_z_quadrature_kink_converges_only_algebraically():
    """A clamp (legacy attenuation exp(-max(z, 0)/lam)) inside a cell: error ~ n^-2 in that cell, not geometric."""
    f = lambda z: np.exp(-np.clip(z, 0, None) / 40.0)                         # noqa: E731
    e = [_z_err(f, n)[0] for n in (5, 10, 20, 40)]
    assert 1e-6 < e[0] < 2e-3 and e[3] < e[0]
    assert e[0] / e[3] < 64 * 8                                               # far from geometric convergence


@pytest.mark.parametrize("sigma", [1e-3, 0.5, 3.2, 12.9, 99.0, 1e4])
@pytest.mark.parametrize("dt", [torch.float32, F64])
def test_gauss_cell_logmass_is_exact_for_every_width(sigma, dt):
    g = ci.Grid(dtype=dt)
    mu = 5.4
    got = g.gauss_z(torch.tensor([[mu]], dtype=dt), torch.tensor([[sigma]], dtype=dt))[0].double().numpy()
    ze = ci.Grid(dtype=F64).z_edges.numpy()

    def cell(i):                                                              # on the tail where both CDFs are small
        a, b = (mp.mpf(float(ze[i])) - mu) / sigma, (mp.mpf(float(ze[i + 1])) - mu) / sigma
        return float(mp.log(mp.ncdf(-a) - mp.ncdf(-b))) if a + b > 0 else float(mp.log(mp.ncdf(b) - mp.ncdf(a)))

    ok = np.array([min(abs(ze[i] - mu), abs(ze[i + 1] - mu)) / sigma < 3e4 for i in range(128)])
    ref = np.array([cell(i) if ok[i] else np.nan for i in range(128)])
    assert ok.sum() >= 2 and np.isfinite(got).all()
    tol = 1e-6 if dt == torch.float32 else 1e-11                              # relative accuracy of the ln mass
    assert np.allclose(got[ok], ref[ok], rtol=tol, atol=tol)
    if sigma >= 0.5:
        assert abs(np.exp(got).sum() - float(mp.ncdf((1450 - mu) / sigma) - mp.ncdf((-200 - mu) / sigma))) < 1e-6


def test_gauss_cell_logmass_gradient():
    g = ci.Grid(dtype=F64)
    mu = torch.tensor([[5.4], [800.0]], dtype=F64, requires_grad=True)
    sg = torch.tensor([[6.0], [99.0]], dtype=F64, requires_grad=True)
    w = torch.softmax(torch.randn(2, 128, dtype=F64, generator=torch.Generator().manual_seed(0)), 1)
    f = lambda m, s: (torch.log_softmax(g.gauss_z(m, s), 1).exp() * w).sum()  # noqa: E731
    assert torch.autograd.gradcheck(f, (mu, sg), eps=1e-6, atol=1e-9, rtol=1e-5)
    mu32 = torch.tensor([[5.4]], requires_grad=True)
    sg32 = torch.tensor([[1e-3]], requires_grad=True)                         # 1e5 sigma away: finite gradients
    out = ci.Grid().gauss_z(mu32, sg32)
    gm, gs = torch.autograd.grad(out.sum(), (mu32, sg32))
    assert torch.isfinite(out).all() and torch.isfinite(gm).all() and torch.isfinite(gs).all()


# ----------------------------------------------------------------------------------------------------------------------
# radial ring masses
# ----------------------------------------------------------------------------------------------------------------------
def _sf_cases():
    return [("gauss", lat.GaussLat(), {}, lambda r, s: np.exp(-r * r / (2 * s * s))),
            ("cauchy", lat.CauchyLat(), {}, lambda r, s: 1 / (1 + (r / s) ** 2)),
            ("exp", lat.ExpLat(), {}, lambda r, s: np.exp(-r / s)),
            ("burr c=1.4 k=2.5", lat.BurrLat(), {"c": 1.4, "k": 2.5}, lambda r, s: (1 + (r / s) ** 1.4) ** -2.5),
            ("weibull b=0.55", lat.WeibullLat(1.0), {"b": 0.55}, lambda r, s: np.exp(-(r / s) ** 0.55)),
            ("weibull b=0.3", lat.WeibullLat(1.0), {"b": 0.3}, lambda r, s: np.exp(-(r / s) ** 0.3))]


@pytest.mark.parametrize("scale", [0.5, 5.0, 60.0, 300.0, 5e3, 1e6])
def test_ring_masses_from_survival_differences(scale):
    """float64: rounding only. float32: <= 3e-5 relative for realistic widths, and never worse than 2e-4 for any
    width between 0.5 mm and 1e6 mm (every ring that holds more than 1e-12 of the disc mass)."""
    g64, g32 = ci.Grid(dtype=F64), ci.Grid()
    re = np.array([mp.mpf(float(v)) for v in g64.r_edges.tolist()])
    for name, fam, sh, _ in _sf_cases():
        sh64 = {k: torch.tensor(v, dtype=F64) for k, v in sh.items()}
        sh32 = {k: torch.tensor(v) for k, v in sh.items()}
        S = {"gauss": lambda r: mp.exp(-r * r / (2 * scale ** 2)), "cauchy": lambda r: 1 / (1 + (r / scale) ** 2),
             "exp": lambda r: mp.exp(-r / scale), "burr": lambda r: (1 + (r / scale) ** mp.mpf(sh.get("c", 1))) ** (-mp.mpf(sh.get("k", 1))),
             "weibull": lambda r: mp.exp(-(r / scale) ** mp.mpf(sh.get("b", 1)))}[name.split()[0]]
        ref = np.array([float(mp.log(S(re[j]) - S(re[j + 1]))) if S(re[j]) - S(re[j + 1]) > 0 else -np.inf for j in range(32)])
        m64 = fam.ring_logmass(g64, torch.tensor([scale], dtype=F64), sh64).numpy()
        m32 = fam.ring_logmass(g32, torch.tensor([scale]), sh32).double().numpy()
        keep = ref > ref.max() - 27.6                                         # rings above 1e-12 of the largest
        assert np.isfinite(m64).all() and np.isfinite(m32).all()
        assert np.abs(m64[keep] - ref[keep]).max() < 2e-11, (name, scale)
        e32 = np.abs(m32[keep] - ref[keep]).max()
        assert e32 < (3e-5 if 5.0 <= scale <= 300.0 and "0.3" not in name else 2e-4), (name, scale, e32)
        # disc mass 1 - S(700)
        d = fam.log_disc_mass(g64, torch.tensor([scale], dtype=F64), sh64)
        assert abs(float(d) - float(mp.log(1 - S(re[-1])))) < 1e-11
        assert abs(float(torch.logsumexp(torch.as_tensor(m64), 0)) - float(d)) < 1e-11


def test_ring_logmass_degenerate_neighbours_stay_finite():
    ls = torch.tensor([[0.0, 0.0, -1e-9, -1e-9, -5.0, -1e30, -1e30]], dtype=torch.float32, requires_grad=True)
    out = ci.ring_logmass(ls)
    (g,) = torch.autograd.grad(out.sum(), ls)
    assert torch.isfinite(out).all() and torch.isfinite(g).all()
    assert float(out[0, 0].detach()) < -60.0                                           # relative mass 1e-30, not -inf


EXPREG = [(1.0, 5.0), (1.0, 20.0), (1.0, 80.0), (1.0, 330.0), (10.0, 300.0), (10.0, 2.5e5)]


@pytest.mark.parametrize("eps,lam", EXPREG)
def test_radial_quadrature_fallback_for_the_regularised_halo(eps, lam):
    """Gauss-Legendre in ln(r + 1 mm). n = 4: <= 3e-7 (lam >= 20 mm), <= 2e-5 (lam = 5 mm); n = 8: <= 1e-9."""
    fam = lat.ExpRegLat(eps)
    f = lambda r: r / (r + eps) * np.exp(-r / lam) / lam                      # noqa: E731
    re = ci.Grid(dtype=F64).r_edges.numpy()
    ref = np.array([integrate.quad(f, re[j], re[j + 1], epsabs=0, epsrel=1e-13)[0] for j in range(32)])
    for n_r, tol in ((4, 3e-7 if lam >= 20 else 2e-5), (8, 1e-9)):
        g = ci.Grid(n_r=n_r, dtype=F64)
        got = fam.ring_logmass(g, torch.tensor([lam], dtype=F64), {}).exp().numpy()
        keep = ref > 1e-12 * ref.max()
        assert np.abs(got[keep] / ref[keep] - 1).max() < tol, (n_r, np.abs(got[keep] / ref[keep] - 1).max())
    # float32, default nodes
    got32 = fam.ring_logmass(ci.Grid(), torch.tensor([lam]), {}).double().exp().numpy()
    assert np.abs(got32[keep] / ref[keep] - 1).max() < 2e-5
    # plane mass of the regularised form: 1 - (eps/lam) e^(eps/lam) E1(eps/lam) < 1
    plane = 1 - eps / lam * math.exp(eps / lam) * special.exp1(eps / lam)
    assert plane < 1
    if lam <= 1e3:
        assert abs(integrate.quad(f, 0, np.inf, epsabs=0, epsrel=1e-12)[0] - plane) < 1e-10


def test_two_level_z_rule_for_a_sharp_profile_times_a_smooth_cross_section():
    """integrate_z_fine: sharp f (4 mm entry edge, range edge, Bragg cusp) on the fine nodes, ln T_j carried from the
    coarse nodes by polynomial interpolation. Here T_j is the track Gaussian with a strongly depth-dependent width."""
    f = lambda z: torch.sigmoid((z - 21.37) / 4.0) * torch.sigmoid((247.3 - z) / 26.0) * (1 - (z / 247.3).clamp(0, 1) + 0.03) ** -0.52   # noqa: E731
    sig = lambda z: torch.sqrt(20.0 ** 2 + (117.0 * (z.clamp_min(1.0) / 1000) ** 1.6 / 0.4) ** 2)                                       # noqa: E731
    gref = ci.Grid(n_z=5, n_zf=600, dtype=F64)
    zr = gref.zf_nodes
    ref = torch.logsumexp((torch.log(f(zr))[:, None] + lat.GaussLat().ring_logmass(gref, sig(zr)[:, None], {})).reshape(128, 600, 32)
                          + gref.log_wzf[None, :, None], 1)
    keep = ref > ref.max() - 20.7                                             # cells above 1e-9 of the largest
    err = {}
    for n_zf in (16, 32):
        g = ci.Grid(n_z=5, n_zf=n_zf, dtype=F64)
        lat_c = lat.GaussLat().ring_logmass(g, sig(g.z_nodes)[None, :, None], {})
        got = g.integrate_z_fine(torch.log(f(g.zf_nodes))[None], lat_c)[0]
        err[n_zf] = float((got - ref)[keep].abs().max())
    plain = g.integrate_z(torch.log(f(g.z_nodes))[None, :, None] + lat_c)[0]  # the 5-node rule alone
    err_plain = float((plain - ref)[keep].abs().max())
    # the rows holding the kinks at z = 1 mm and z = R limit both rules; elsewhere the fine rule is at 1e-7
    assert err[16] < 3e-4 and err[32] < 1e-4 and err_plain > 4 * err[16], (err, err_plain)
    # z-independent cross-section: reduces to the fine rule in z
    flat = g.integrate_z_fine(torch.log(f(g.zf_nodes))[None], lat_c[:, :1])
    assert torch.allclose(flat, g.integrate_zf(torch.log(f(g.zf_nodes))[None])[:, :, None] + lat_c[:, :1], atol=1e-12)
