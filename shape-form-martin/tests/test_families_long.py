"""sfm.families.longitudinal: every depth density against scipy.stats / multiprecision, limits, extremes (CPU)."""
import math

import mpmath as mp
import numpy as np
import pytest
import torch
from scipy import integrate, stats

from sfm.families import cellint as ci
from sfm.families import longitudinal as lg

mp.mp.dps = 50
F64 = torch.float64
S = np.array([1e-3, 0.05, 1.0, 7.0, 40.0, 150.0, 400.0, 900.0, 1650.0])


def T(v, dt=F64):
    return torch.tensor(v, dtype=dt)


def nat_of(fam, raw, dt=F64):
    return fam.natural({k: T([[v]], dt) for k, v in raw.items()})


def raw_defaults(fam, **over):
    d = {n: v for n, v in fam.params()}
    d.update(over)
    return d


# ----------------------------------------------------------------------------------------------------------------------
# densities in s against scipy.stats
# ----------------------------------------------------------------------------------------------------------------------
def test_gamma_legacy_form_is_the_gamma_density():
    fam = lg.GammaLegacy()
    for tmax, am1 in ((100.0, 3.0), (20.0, 0.3), (600.0, 12.0)):
        nat = nat_of(fam, {"z0": -20.0, "log_tmax": math.log(tmax), "log_am1": math.log(am1)})
        got = fam.log_pdf_s(T(S)[None], nat)[0].numpy()
        assert np.allclose(got, stats.gamma(a=am1 + 1, scale=tmax / am1).logpdf(S), rtol=1e-12, atol=1e-10)
    # the unnormalised form is the legacy expression (shape-form-proton/families.py: long_gamma), at its soft start
    z = torch.linspace(-200, 1450, 300, dtype=F64)[None]
    nat = nat_of(fam, {"z0": -20.0, "log_tmax": math.log(100.0), "log_am1": math.log(3.0)})
    s = torch.nn.functional.softplus((z - (-20.0)) / 2) * 2 + 1e-3
    assert torch.allclose(fam.log_pdf(z, nat), 3.0 * torch.log(s) - 3.0 / 100.0 * s, rtol=1e-13, atol=1e-12)


GG = [(150.0, 0.8, 1.0), (80.0, 0.35, 0.4), (300.0, 1.5, 2.5), (40.0, 0.6, 0.12), (500.0, 0.25, 5.5)]


@pytest.mark.parametrize("m,w,p", GG)
def test_gengamma_matches_scipy_and_its_mode_width_parametrisation(m, w, p):
    fam = lg.GenGamma()
    f = (p - fam.p_lo) / (fam.p_hi - fam.p_lo)
    nat = nat_of(fam, {"z0": 0.0, "rm": lg.raw_of(m, 1.0, lg.U_MM), "rw": lg.raw_of(w, 0.02), "rp": math.log(f / (1 - f))})
    assert abs(float(nat["m"]) - m) < 1e-9 and abs(float(nat["w"]) - w) < 1e-12 and abs(float(nat["p"]) - p) < 1e-12
    a, d, pp = (float(v) for v in fam.stacy(nat))
    got = fam.log_pdf_s(T(S)[None], nat)[0].numpy()
    ref = stats.gengamma(a=d / pp, c=pp, scale=a).logpdf(S)
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-8)
    assert abs(a * ((d - 1) / pp) ** (1 / pp) - m) < 1e-8 * m                 # the mode is m
    # curvature of ln f in ln s at the mode is -1/w^2
    h = 1e-4
    ls = T([[math.log(m) - h, math.log(m), math.log(m) + h]])
    v = fam.log_pdf_s(torch.exp(ls), nat, normalised=False)[0]
    assert abs(float((v[0] - 2 * v[1] + v[2]) / h ** 2) + 1 / w ** 2) < 1e-4 / w ** 2 + 1e-5


def test_gengamma_limits_gamma_at_p1_and_lognormal_at_p0_and_series_branch():
    gg, ga, ln = lg.GenGamma(p_range=(1e-7, 6.0)), lg.GammaLegacy(), lg.LogNormal()
    s = T(S)[None]
    m, w = 120.0, 0.7
    one = lambda p: {"z0": T([[0.0]]), "delta": 2.0, "m": T([[m]]), "w": T([[w]]), "p": T([[p]])}   # noqa: E731
    g1 = gg.log_pdf_s(s, one(1.0))
    gam = ga.log_pdf_s(s, {"am1": T([[1 / w ** 2]]), "tmax": T([[m]])})
    assert torch.allclose(g1, gam, rtol=1e-11, atol=1e-10)
    g0 = gg.log_pdf_s(s, one(1e-7), normalised=False)
    lnn = ln.log_pdf_s(s, {"med": T([[m * math.exp(w ** 2)]]), "sig": T([[w]])}, normalised=False)
    assert torch.allclose(g0 - g0[0, 3], lnn - lnn[0, 3], rtol=1e-5, atol=1e-5)          # same shape (constants differ)
    # the |q| < 0.5 series and the closed form agree across the switch, in both precisions
    for dt, tol in ((F64, 1e-13), (torch.float32, 3e-6)):
        for p in (0.3, 1.0, 3.0):
            sq = torch.exp(torch.tensor([-0.5000001, -0.4999999, 0.4999999, 0.5000001], dtype=dt) / p) * m
            nat = {"m": T([[m]], dt), "w": T([[w]], dt), "p": T([[p]], dt)}
            v = gg.log_pdf_s(sq[None], nat, normalised=False)[0].double()
            ref = [float((q - mp.expm1(q)) / (w * p) ** 2) for q in (-mp.mpf("0.5"), mp.mpf("0.5"))]
            assert abs(float(v[0]) - ref[0]) < tol * 50 + 3e-6 and abs(float(v[1]) - ref[0]) < tol * 50 + 3e-6
            assert abs(float(v[2]) - ref[1]) < tol * 50 + 3e-6 and abs(float(v[3]) - ref[1]) < tol * 50 + 3e-6
    # small |q| in float32: the series keeps relative accuracy where q - expm1(q) cancels
    nat = {"m": T([[m]], torch.float32), "w": T([[w]], torch.float32), "p": T([[1.0]], torch.float32)}
    sq = torch.tensor([[m * math.exp(1e-3), m * math.exp(-2e-2)]], dtype=torch.float32)
    v = gg.log_pdf_s(sq, nat, normalised=False)[0].double().numpy()
    q = np.log(sq[0].double().numpy() / m)
    ref = np.array([float(mp.mpf(x) - mp.expm1(mp.mpf(x))) for x in q]) / w ** 2
    assert np.allclose(v, ref, rtol=2e-4)                                     # limited by ln(s/m) in float32, not by the series


@pytest.mark.parametrize("mu,cv", [(250.0, 0.9), (60.0, 0.15), (900.0, 2.5)])
def test_invgauss_matches_scipy(mu, cv):
    fam = lg.InvGauss()
    nat = nat_of(fam, {"z0": 0.0, "rmu": lg.raw_of(mu, 1.0, lg.U_MM), "rcv": lg.raw_of(cv, 0.02)})
    lam = mu / cv ** 2
    got = fam.log_pdf_s(T(S)[None], nat)[0].numpy()
    assert np.allclose(got, stats.invgauss(mu=mu / lam, scale=lam).logpdf(S), rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("med,sig", [(200.0, 0.8), (30.0, 0.1), (700.0, 2.0)])
def test_lognormal_matches_scipy(med, sig):
    fam = lg.LogNormal()
    nat = nat_of(fam, {"z0": 0.0, "rmed": lg.raw_of(med, 1.0, lg.U_MM), "rsig": lg.raw_of(sig, 0.02)})
    got = fam.log_pdf_s(T(S)[None], nat)[0].numpy()
    assert np.allclose(got, stats.lognorm(s=sig, scale=med).logpdf(S), rtol=1e-11, atol=1e-9)


@pytest.mark.parametrize("lam,c,k", [(300.0, 1.8, 3.0), (50.0, 1.05, 0.2), (800.0, 6.0, 0.7)])
def test_burr_matches_scipy(lam, c, k):
    fam = lg.BurrXII()
    nat = nat_of(fam, {"z0": 0.0, "rscale": lg.raw_of(lam, 1.0, lg.U_MM), "rc": lg.raw_of(c, 1.0), "rk": lg.raw_of(k, 0.05)})
    got = fam.log_pdf_s(T(S)[None], nat)[0].numpy()
    assert np.allclose(got, stats.burr12(c=c, d=k, scale=lam).logpdf(S), rtol=1e-10, atol=1e-8)


# ----------------------------------------------------------------------------------------------------------------------
# Exp (*) Erlang
# ----------------------------------------------------------------------------------------------------------------------
def _hn_ref(x, n):
    x = mp.mpf(x)
    if x == 0:
        return mp.log(1 / mp.factorial(n))
    return mp.log((1 - mp.exp(-x) * sum(x ** k / mp.factorial(k) for k in range(n))) / x ** n)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_log_hn_all_three_regions_and_the_removable_singularity(n):
    xs = [-3300.0, -200.0, -50.0, -2.0001, -2.0, -1.9999, -1.0, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 1.0, 1.9999, 2.0, 2.0001,
          50.0, 3300.0]
    ref = np.array([float(_hn_ref(x, n)) for x in xs])
    got = lg.log_hn(T(xs), n).numpy()
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)
    got32 = lg.log_hn(T(xs, torch.float32), n).double().numpy()
    assert np.isfinite(got32).all() and np.allclose(got32, ref, rtol=3e-6, atol=5e-6)
    x = T([-30.0, -2.5, -1.5, 0.0, 0.7, 1.99, 2.01, 9.0], F64).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda v: lg.log_hn(v, n), (x,), eps=1e-6, atol=1e-7)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("l1,l2", [(250.0, 60.0), (60.0, 250.0), (100.0, 100.0), (100.0, 100.0 * (1 + 1e-9)), (5.0, 900.0)])
def test_experlang_is_the_convolution_of_an_exponential_and_an_erlang(n, l1, l2):
    fam = lg.ExpErlang(n=n, ordered=False)
    nat = {"l1": T([[l1]]), "l2": T([[l2]])}
    s = np.array([0.5, 7.0, 40.0, 150.0, 400.0, 900.0, 1650.0])
    got = fam.log_pdf_s(T(s)[None], nat)[0].numpy()
    nu, be = mp.mpf(1) / l1, mp.mpf(1) / l2
    ref = [float(mp.log(mp.quad(lambda t: nu * mp.exp(-nu * (si - t)) * be ** n * t ** (n - 1) * mp.exp(-be * t) / mp.factorial(n - 1),
                                [0, si / 2, si]))) for si in s]
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)
    if l1 == l2:                                                              # equal rates: Gamma(n + 1, l1)
        assert np.allclose(got, stats.gamma(a=n + 1, scale=l1).logpdf(s), rtol=1e-11, atol=1e-10)


def test_experlang_ordering_link():
    fam = lg.ExpErlang(n=1)                                                   # ordered by default for n = 1
    assert fam.ordered and not lg.ExpErlang(n=2).ordered
    raw = {k: torch.randn(50, 1, dtype=F64, generator=torch.Generator().manual_seed(0)) * 3 for k in ("z0", "rl1", "rl2")}
    nat = fam.natural(raw)
    assert (nat["l1"] > nat["l2"]).all()


# ----------------------------------------------------------------------------------------------------------------------
# EMG
# ----------------------------------------------------------------------------------------------------------------------
def _emg_ref(z, mu, sigma, tau):
    z, mu, sigma, tau = (mp.mpf(float(v)) for v in (z, mu, sigma, tau))
    t, K = (z - mu) / sigma, sigma / tau
    return float(-mp.log(tau) - K * t + K * K / 2 + mp.log(mp.ncdf(t - K)))


EMG_CASES = [(30.0, 60.0, 300.0), (5.0, 4.0, 297.0), (0.0, 0.5, 50.0), (100.0, 300.0, 10.0), (-50.0, 30.0, 1.0),
             (10.0, 1000.0, 1.0), (10.0, 100.0, 5000.0)]


@pytest.mark.parametrize("mu,sigma,tau", EMG_CASES)
def test_emg_density_is_stable_for_every_ratio_sigma_over_tau(mu, sigma, tau):
    fam = lg.EMG()
    z = np.array([-200.0, -150.0, -20.0, 0.0, 5.0, 30.0, 100.0, 400.0, 1000.0, 1450.0])
    ref = np.array([_emg_ref(v, mu, sigma, tau) for v in z])
    for dt, tol in ((F64, 1e-11), (torch.float32, 4e-6)):
        nat = {"mu": T([[mu]], dt), "sigma": T([[sigma]], dt), "tau": T([[tau]], dt)}
        got = fam.log_pdf(T(z, dt)[None], nat)[0].double().numpy()
        assert np.isfinite(got).all()
        assert np.allclose(got, ref, rtol=tol, atol=tol * 10), (dt, np.abs(got - ref).max())
    if sigma / tau < 20 and sigma >= 4:                                       # scipy agrees where it is itself accurate
        sc = stats.exponnorm(K=tau / sigma, loc=mu, scale=sigma).logpdf(z)
        assert np.allclose(sc, ref, rtol=1e-6, atol=1e-6)


def test_emg_normalised_and_moments():
    fam = lg.EMG()
    mu, sigma, tau = 30.0, 60.0, 300.0
    nat = {"mu": T([[mu]]), "sigma": T([[sigma]]), "tau": T([[tau]])}
    f = lambda z: float(fam.log_pdf(T([[z]]), nat).exp())                     # noqa: E731
    n0 = integrate.quad(f, -2000, 20000, epsabs=0, epsrel=1e-11, limit=400)[0]
    n1 = integrate.quad(lambda z: z * f(z), -2000, 20000, epsabs=0, epsrel=1e-11, limit=400)[0]
    n2 = integrate.quad(lambda z: (z - mu - tau) ** 2 * f(z), -2000, 20000, epsabs=0, epsrel=1e-11, limit=400)[0]
    assert abs(n0 - 1) < 1e-9 and abs(n1 - (mu + tau)) < 1e-6 and abs(n2 - (sigma ** 2 + tau ** 2)) < 1e-3


def test_emg_single_branch_forms_fail_where_the_two_branch_form_does_not():
    """The reasons for the two branches (float32): the log_ndtr-only form cancels K^2/2, a pure erfcx form overflows."""
    fam = lg.EMG()
    dt = torch.float32
    # (a) large K = sigma/tau: the naive form adds K^2/2 = 5e5 to a number near -5e5
    mu, sigma, tau = 10.0, 1000.0, 1.0
    nat = {"mu": T([[mu]], dt), "sigma": T([[sigma]], dt), "tau": T([[tau]], dt)}
    z = T([[-200.0, 0.0, 500.0, 1011.0, 1450.0]], dt)
    ref = np.array([_emg_ref(v, mu, sigma, tau) for v in z[0].tolist()])
    stable = fam.log_pdf(z, nat)[0].double().numpy()
    naive = fam.log_pdf_naive(z, nat)[0].double().numpy()
    assert np.abs(stable - ref).max() < 1e-5 and np.abs(naive - ref).max() > 1e-3
    # at K = 30 (the bound of PLAN.md A4) the naive form is still fine: error ~ 6e-8 K^2 / 2
    nat30 = {"mu": T([[mu]], dt), "sigma": T([[60.0]], dt), "tau": T([[2.0]], dt)}
    ref30 = np.array([_emg_ref(v, mu, 60.0, 2.0) for v in z[0].tolist()])
    assert np.abs(fam.log_pdf_naive(z, nat30)[0].double().numpy() - ref30).max() < 5e-4
    # (b) erfcx(-u / sqrt 2) overflows for u = (z - mu)/sigma - K > 13.3 in float32, 37.7 in float64
    assert torch.isinf(torch.special.erfcx(T(-13.4 / math.sqrt(2), dt))) and torch.isfinite(torch.special.erfcx(T(-13.2 / math.sqrt(2), dt)))
    assert torch.isinf(torch.special.erfcx(T(-37.8 / math.sqrt(2)))) and torch.isfinite(torch.special.erfcx(T(-37.6 / math.sqrt(2))))
    nat = {"mu": T([[5.0]], dt), "sigma": T([[3.0]], dt), "tau": T([[297.0]], dt)}
    far = fam.log_pdf(T([[5.0 + 3.0 * 14, 1450.0]], dt), nat)                 # u = 14 and u = 480: fine
    assert torch.isfinite(far).all()
    assert abs(float(far[0, 1]) - _emg_ref(1450.0, 5.0, 3.0, 297.0)) < 1e-5


def test_emg_gradients_stay_accurate_in_the_far_tails():
    """d ln f / d(mu, sigma, tau) in float32 against float64, including cells 400 sigma in front of the edge."""
    fam = lg.EMG()
    z = [-200.0, 0.0, 30.0, 700.0, 1450.0]
    for mu, sigma, tau in ((5.0, 0.5, 297.0), (30.0, 60.0, 300.0), (10.0, 300.0, 1.0)):
        grads = {}
        for dt in (F64, torch.float32):
            p = [T([[v]], dt).requires_grad_(True) for v in (mu, sigma, tau)]
            out = fam.log_pdf(T(z, dt)[None], {"mu": p[0], "sigma": p[1], "tau": p[2]})
            grads[dt] = torch.stack([torch.stack([torch.autograd.grad(out[0, i], q, retain_graph=True)[0][0, 0] for q in p])
                                     for i in range(len(z))]).double()
        assert torch.isfinite(grads[torch.float32]).all()
        scale = grads[F64].abs().amax(1, keepdim=True)
        assert float(((grads[torch.float32] - grads[F64]).abs() / scale).max()) < 2e-4


def test_emg_external_tau_and_cap():
    fam = lg.EMG("sh_", tau="external")
    assert fam.names == ["sh_mu", "sh_rsig"]
    raw = {"mu": T([[0.3]]), "rsig": T([[0.1]])}
    with pytest.raises(ValueError):
        fam.natural(raw)
    nat = fam.natural(raw, tau=T([[297.0]]))
    assert float(nat["tau"]) == 297.0 and abs(float(nat["mu"]) - 30.0) < 1e-12
    capped = lg.EMG(k_max=30.0)
    nat = {"mu": T([[0.0]]), "sigma": T([[3000.0]]), "tau": T([[1.0]])}
    assert torch.isfinite(capped.log_pdf(T([[0.0, 100.0]]), nat)).all()


# ----------------------------------------------------------------------------------------------------------------------
# the soft start and the plateau
# ----------------------------------------------------------------------------------------------------------------------
def test_soft_start_is_literal_behind_the_start_and_finite_in_front():
    for name in ("gengamma", "invgauss", "lognormal", "experlang2", "burr"):
        fam = lg.make_long(name)
        nat = fam.natural({k: T([[v]]) for k, v in raw_defaults(fam, z0=0.1).items()})
        z = T([[80.0, 200.0, 900.0]])
        direct = fam.log_pdf_s(z - 10.0 + lg.S_MIN, nat, normalised=False)
        assert torch.allclose(fam.log_pdf(z, nat), direct, rtol=1e-10, atol=1e-10)      # z - z0 >= 70 mm = 35 delta
        front = fam.log_pdf(T([[-200.0, -50.0, 9.0]]), nat)
        assert torch.isfinite(front).all() and float(front[0, 0]) <= float(front[0, 2])
    free = lg.make_long("gengamma", soft="free")
    assert free.local_names[-1] == "rdelta"
    nat = free.natural({k: T([[v]]) for k, v in raw_defaults(free).items()})
    assert abs(float(nat["delta"]) - 2.0) < 1e-9
    s = lg.soft_start(T([-1e4, -10.0, 0.0, 10.0, 1e4]), 0.0, 2.0)
    assert float(s[0]) == lg.S_MIN and abs(float(s[2]) - (2 * math.log(2) + lg.S_MIN)) < 1e-12 and abs(float(s[4]) - (1e4 + lg.S_MIN)) < 1e-9


def test_plateau_form_and_names():
    fam = lg.Plateau()
    assert fam.names == ["zs", "log_w1", "log_ell", "log_w2", "kappa"]
    raw = {"zs": 17.4, "log_w1": math.log(117.0), "log_ell": math.log(36.2), "log_w2": math.log(175.0), "kappa": -2.04}
    nat = nat_of(fam, raw)
    z = np.linspace(-200, 1450, 50)
    ref = (np.log(1 / (1 + np.exp(-(z - 17.4) / 117.0))) + np.log(1 / (1 + np.exp(-(36.2 - z) / 175.0))) - 2.04 * z / 1000)
    assert np.allclose(fam.log_pdf(T(z)[None], nat)[0].numpy(), ref, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------------------------------------------------------
# finite values and gradients at parameter extremes, float32, on the cell nodes
# ----------------------------------------------------------------------------------------------------------------------
ALL = ["plateau", "gamma", "emg", "gengamma", "invgauss", "lognormal", "experlang1", "experlang2", "experlang3", "burr"]


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("soft", [2.0, "free"])
def test_finite_values_and_gradients_at_raw_extremes(name, soft):
    kw = {"soft": soft} if name not in ("plateau", "gamma", "emg") else {}
    if soft == "free" and not kw:
        pytest.skip("no soft start")
    fam = lg.make_long(name, **kw)
    g = ci.Grid()
    z = torch.cat([g.z_nodes, g.z_edges])[None]
    gen = torch.Generator().manual_seed(0)
    n = len(fam.local_names)
    # corners of the raw box [-60, 60]^n (softplus / sigmoid / exp links all saturated) plus random points
    corners = torch.tensor([[60.0 if (i >> j) & 1 else -60.0 for j in range(n)] for i in range(2 ** n)])
    pts = torch.cat([corners, (torch.rand(200, n, generator=gen) - 0.5) * 120, torch.randn(200, n, generator=gen) * 3])
    raw = {k: pts[:, j:j + 1].clone().requires_grad_(True) for j, k in enumerate(fam.local_names)}
    out = fam.log_pdf(z, fam.natural(raw))
    assert out.shape == (len(pts), z.shape[1]) and torch.isfinite(out).all()
    w = torch.softmax(torch.randn(out.shape, generator=gen), 1)
    grads = torch.autograd.grad((torch.log_softmax(out, 1) * w).sum(), list(raw.values()))
    assert all(torch.isfinite(x).all() for x in grads)


@pytest.mark.parametrize("name", ALL)
def test_defaults_names_and_order_keys(name):
    fam = lg.make_long(name, prefix="c1_")
    assert all(n.startswith("c1_") for n in fam.names) and list(fam.defaults) == fam.names
    assert not any("." in n for n in fam.names)
    key, raw_name, kind = fam.ORDER_KEY
    assert raw_name in fam.local_names and kind in ("real", "positive")
    raw = {k: T([[v], [v]]) for k, v in fam.params()}
    first = fam.natural(raw)
    second = fam.natural(raw, key_prev=first[key])
    assert key in first and (second[key] > first[key]).all()                  # cumulative ordering is strict
