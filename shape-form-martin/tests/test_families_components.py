"""sfm.families.components: legacy parity, exact integration, variants, swapped families, mixture, initialisation.

Data-free: the legacy formula is loaded from its checkpoint (``load_legacy_core``), all comparisons are at map level
on the CPU. Only the last test touches the GPU (one forward + backward at batch 256), and only if it is idle.

Measured values quoted in the assertions' comments are from this machine (torch 2.14, float64 unless stated).
"""
import copy
import time

import numpy as np
import pytest
import torch

from sfm import families as fam
from sfm.families import cellint as ci
from sfm.families import components as cp
from sfm.families.params import ConcatProvider, ConstProvider, PolyProvider, lnT, x_of_lnp

F64 = torch.float64
P = [0.0895, 0.12, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]      # 15 momenta, GeV


@pytest.fixture(scope="module")
def legacy():
    core, logA = cp.load_legacy_core("cpu")
    import shapelab                                                           # shape-form-proton (read-only)
    formula = shapelab.WithAcceptance(core, torch.exp(logA)).eval()
    with torch.no_grad():
        formula.logA.copy_(logA)
    return core, logA, formula


def lnp_of(p=P, dt=torch.float32):
    return torch.log(torch.tensor(p, dtype=dt))


def mk(legacy, variant, dt=F64, grid=None, **kw):
    core, logA, _ = legacy
    grid = grid if grid is not None else (ci.Grid(dtype=F64) if dt == F64 else None)
    return cp.build_variant(variant, core, logA, grid=grid, dtype=dt if dt == F64 else None, **kw)


def massive(logp, thr=1e-9):
    return logp.exp() > thr


# ----------------------------------------------------------------------------------------------------------------------
# providers
# ----------------------------------------------------------------------------------------------------------------------
def test_poly_provider_returns_exactly_the_legacy_raw_parameters(legacy):
    core, _, _ = legacy
    prov = PolyProvider.from_legacy(core)
    lnp = lnp_of([0.0005, 0.0734, 0.0895, 0.3, 1.0, 7.0, 20.0, 50.0])
    with torch.no_grad():
        assert torch.equal(prov(lnp), core.params(lnp))                       # bit for bit
        assert torch.equal(prov.x(lnp), (core.lnT(lnp) - core.v0) / core.vs)
        x = x_of_lnp(lnp)
    assert prov.names == list(core.names) and prov.n_params() == core.n_params() == 101
    assert abs(float(prov.v0) + 0.325860) < 1e-6 and abs(float(prov.vs) - 3.314613) < 1e-6
    assert abs(float(x[2]) + 1.0) < 2e-4 and abs(float(x[6]) - 1.0) < 1e-6 and abs(float(x[0]) + 4.103) < 1e-3
    assert torch.isfinite(lnT(lnp_of([1e-15]))).all()                          # no floor, still finite far below 0.0005 GeV


def test_provider_helpers():
    a = PolyProvider(["u", "v"], {"u": 2, "v": 0}, {"u": [1.0, 2.0, 3.0], "v": 4.0}, v0=0.0, vs=1.0)
    c = ConstProvider(["w"], {"w": -1.5})
    cat = ConcatProvider([a, c])
    lnp = lnp_of([0.5, 2.0])
    out = cat(lnp)
    x = a.x(lnp)
    assert cat.names == ["u", "v", "w"] and out.shape == (2, 3) and cat.n_params() == 3 + 1 + 1
    assert torch.allclose(out[:, 0], 1 + 2 * x + 3 * x * x) and torch.allclose(out[:, 1], torch.full((2,), 4.0))
    assert torch.allclose(out[:, 2], torch.full((2,), -1.5))
    assert fam.params.is_param_provider(cat) and not fam.params.is_param_provider(torch.nn.Linear(1, 1))
    # function-space initialisation of one row
    l = torch.linspace(-2, 3, 40)
    a.set_rows_lstsq(["u"], l, (0.5 - 0.25 * a.x(l) + 2.0 * a.x(l) ** 2)[:, None])
    assert torch.allclose(a.coef[0].detach(), torch.tensor([0.5, -0.25, 2.0]), atol=1e-5)
    assert float(a.coef[1, 0].detach()) == 4.0 and float(a.row_mask(["v"]).sum()) == 1.0
    with pytest.raises(ValueError):
        PolyProvider(["u", "u"])
    with pytest.raises(KeyError):
        cp.ComponentModel(ConstProvider(["zs"]))                              # provider lacks required names


def test_names_counting_and_split(legacy):
    core, logA, _ = legacy
    m = mk(legacy, "legacy_centre", torch.float32)
    assert sorted(m.names) == sorted(core.names) and m.new_names == []
    assert m.n_params() == 101
    assert len(m.momentum_dependent) == 22 and len(m.legacy_constant) == 21
    deg = {n: int(core.mask[i].sum()) - 1 for i, n in enumerate(core.names)}
    assert all(deg[n] == 0 for n in m.legacy_constant) and all(deg[n] > 0 for n in m.momentum_dependent)
    # the acceptance is a buffer, not a parameter; a swapped family adds its own constants
    assert not any(p.shape == (128, 32) for p in m.parameters())
    e = mk(legacy, "survival_exact", torch.float32, shower_long="emg")
    assert e.new_names == ["sh_mu", "sh_rsig", "sh_rtau"] and e.n_params() == 101 - 5 * 4 + 3 * 4
    b = mk(legacy, "survival_exact", torch.float32, shower_lat=("gauss", "burr", "weibull"))
    assert b.new_names == ["lat1_rc", "lat1_rk", "lat2_rb"] and b.n_params() == 104


# ----------------------------------------------------------------------------------------------------------------------
# legacy parity (integration="centre")
# ----------------------------------------------------------------------------------------------------------------------
def test_centre_mode_reproduces_the_legacy_formula_float32(legacy):
    """15 momenta across [0.0895, 20] GeV. Measured: max |d ln P| over cells with P > 1e-9: 6.0e-5; |KL| <= 3e-7.
    The residual is float32 rounding of the legacy code itself (next test)."""
    core, logA, formula = legacy
    m = mk(legacy, "legacy_centre", torch.float32)
    lnp = lnp_of()
    with torch.no_grad():
        ref, out = formula(lnp), m(lnp)
        assert out.shape == (15, 128, 32) and torch.isfinite(out).all()
        d = (out - ref).abs()
        assert float(d[massive(ref)].max()) < 2e-4
        assert float(d.max()) < 2e-3                                          # every cell, down to P ~ 1e-24
        assert float(cp.kl_div(ref.double(), out.double()).abs().max()) < 2e-6
        # the mixture before the acceptance, and component by component
        base = core(lnp)
        assert float((m.core_map(lnp) - base).abs()[massive(base)].max()) < 2e-4
        lw_l, comp_l, _ = core.components(lnp)
        lw_m, comp_m = m.components(lnp)
        for k, name in enumerate(m.COMPONENTS):
            assert torch.allclose(lw_m[name], lw_l[k].reshape(-1), atol=2e-6), name
            # the end blob sits far outside the grid (R(T) up to 4e4 mm): its ln P reaches -1e4, float32 ulp 1e-3
            assert float((comp_m[name] - comp_l[k]).abs()[massive(comp_l[k])].max()) < (3e-4 if name != "end" else 2e-3), name
            assert abs(float(torch.logsumexp(comp_m[name].reshape(15, -1), 1).abs().max())) < 1e-5


def test_centre_mode_parity_residual_is_legacy_float32_rounding(legacy):
    """In float64 the two implementations agree to 1e-12; the float32 legacy map is 8e-5 away from its own float64
    evaluation (it forms (L_gamma + lat) - L_gamma with |L_gamma| ~ 1e3)."""
    core, logA, formula = legacy
    f64 = copy.deepcopy(formula).double()
    grid = ci.Grid.from_common().double()                                     # the float32 grid numbers, upcast like the legacy buffers
    m64 = mk(legacy, "legacy_centre", F64, grid=grid)
    lnp = lnp_of(dt=F64)
    with torch.no_grad():
        r64, o64, r32 = f64(lnp), m64(lnp), formula(lnp.float())
        assert float((o64 - r64).abs().max()) < 1e-10                         # measured 1.1e-13
        assert float(cp.kl_div(r64, o64).abs().max()) < 1e-13
        legacy_noise = float((r32.double() - r64).abs()[massive(r64)].max())
        assert 1e-5 < legacy_noise < 3e-4                                     # measured 8.5e-5
        # an independently built float64 grid (edges rounded once instead of float32 arithmetic) changes nothing visible
        o64b = mk(legacy, "legacy_centre", F64)(lnp)
        assert float((o64b - r64).abs()[massive(r64)].max()) < 5e-6


def test_centre_mode_agrees_with_legacy_far_outside_the_training_range(legacy):
    """p = 0.0005 GeV (x = -4.10) ... 50 GeV: both finite and equal; the clamps of the log links are not active in
    any way that changes the map."""
    _, _, formula = legacy
    m = mk(legacy, "legacy_centre", torch.float32)
    lnp = lnp_of([0.0005, 0.002, 0.01, 0.05, 30.0, 50.0])
    with torch.no_grad():
        ref, out = formula(lnp), m(lnp)
    assert torch.isfinite(ref).all() and torch.isfinite(out).all()
    assert float((out - ref).abs()[massive(ref)].max()) < 3e-4


# ----------------------------------------------------------------------------------------------------------------------
# exact integration
# ----------------------------------------------------------------------------------------------------------------------
VARIANT_KW = [("legacy_centre", {}), ("legacy_exact", {}), ("survival_exact", {}), ("survival_centre", {}),
              ("survival_exact", dict(shower_lat=("gauss", "burr", "weibull"))),
              ("survival_exact", dict(shower_long="emg")), ("survival_exact", dict(shower_long="emg_tied")),
              ("survival_exact", dict(shower_long="gamma")), ("survival_exact", dict(shower_long="gengamma")),
              ("survival_exact", dict(shower_long="invgauss")), ("survival_exact", dict(shower_long="lognormal")),
              ("survival_exact", dict(shower_long="experlang1")), ("survival_exact", dict(shower_long="experlang2")),
              ("survival_exact", dict(shower_long="experlang3")), ("survival_exact", dict(shower_long="burr")),
              ("survival_exact", dict(shower_long="gengamma", shower_long_kw=dict(soft="free"))),
              ("survival_exact", dict(lat_norm="term", lat_ordered=True)),
              ("legacy_exact", dict(shower_lat=("weibull_core", "student", "exp_reg")))]
VARIANT_IDS = [v + ("" if not kw else ":" + "/".join(str(x) for x in kw.values())) for v, kw in VARIANT_KW]


@pytest.mark.parametrize("variant,kw", VARIANT_KW, ids=VARIANT_IDS)
def test_cell_masses_sum_to_one_and_stay_finite_down_to_half_an_MeV(legacy, variant, kw):
    """float32, p from 0.0005 GeV (x = -4.10) to 50 GeV: finite values, finite gradients of all coefficients, and
    sum of the cell masses = 1 within 1e-6 for p >= 0.01 GeV. At 0.0005 and 0.002 GeV the float32 ``log_softmax``
    of a map that sits in one or two cells is normalised to 7e-6 only; the legacy formula has the same 6e-6 there
    (float64: 1e-12 everywhere, next test)."""
    m = mk(legacy, variant, torch.float32, **kw)
    lnp = lnp_of([0.0005, 0.002, 0.01, 0.05, 0.0895, 0.3, 1.0, 5.0, 20.0, 50.0])
    for q in m.parameters():
        q.requires_grad_(True)
    out = m(lnp)
    assert torch.isfinite(out).all()
    tot = (out.detach().double().exp().sum((1, 2)) - 1).abs()
    assert float(tot[2:].max()) < 1e-6 and float(tot[:2].max()) < 2e-5, tot.tolist()
    w = torch.softmax(torch.randn(len(lnp), 4096, generator=torch.Generator().manual_seed(0)), 1).reshape(out.shape)
    (-(w * out).sum()).backward()
    assert torch.isfinite(m.provider.coef.grad).all()
    lw, comp = m.components(lnp)
    for k, c in comp.items():
        assert torch.isfinite(c).all() and torch.isfinite(lw[k]).all(), k
    assert float((torch.logsumexp(torch.stack(list(lw.values())), 0)).detach().abs().max()) < 1e-5      # weights sum to 1


@pytest.mark.parametrize("variant", ["legacy_exact", "survival_exact", "legacy_centre"])
def test_cell_masses_sum_to_one_float64(legacy, variant):
    with torch.no_grad():
        out = mk(legacy, variant)(lnp_of(dt=F64))
        lw, comp = mk(legacy, variant).components(lnp_of(dt=F64))
    assert float((out.exp().sum((1, 2)) - 1).abs().max()) < 1e-12
    for c in comp.values():
        assert float((c.exp().sum((1, 2)) - 1).abs().max()) < 1e-12


def test_binning_error_of_the_centre_rule_per_momentum_and_where_it_sits(legacy):
    """Same forms, same coefficients: KL(centre || exact). Measured (nat):
        p [GeV]  0.0895  0.12    0.2     0.35    0.5     0.75    1       1.5     2       3       5       7.5     10      15      20
        KL       9.2e-2  8.1e-2  3.1e-2  5.0e-3  2.3e-3  1.1e-3  7.2e-4  4.5e-4  3.4e-4  2.5e-4  2.0e-4  1.8e-4  1.7e-4  1.6e-4  1.5e-4
    It sits at the entrance: the z rows [-6.6, 6.3] and [6.3, 19.1] mm carry the two largest contributions (opposite
    signs: the centre rule misplaces the narrow spike, sigma_z = 3-5 mm, and the 4 mm track edge between them); the
    cells with z' < 45 mm hold 100 % (p <= 0.2), 69 % (1 GeV), 36 % (20 GeV) of the absolute contributions, the rings
    r < 12.5 mm 86 % -> 23 %. By component: spike 6e-3 ... 0.2, track 4e-4 ... 2e-2, shower 7e-6 ... 3e-5."""
    lnp = lnp_of(dt=F64)
    mc, me = mk(legacy, "legacy_centre"), mk(legacy, "legacy_exact")
    with torch.no_grad():
        c, e = mc(lnp), me(lnp)
        kl = cp.kl_div(c, e)
        want = [9.17e-2, 8.14e-2, 3.07e-2, 5.03e-3, 2.28e-3, 1.08e-3, 7.18e-4, 4.48e-4, 3.41e-4, 2.51e-4, 1.97e-4, 1.79e-4,
                1.71e-4, 1.58e-4, 1.47e-4]
        assert np.allclose(kl.numpy(), want, rtol=0.03)
        contrib = c.exp() * (c - e)
        for i, front, inner in ((0, 0.99, 0.80), (6, 0.6, 0.28), (14, 0.3, 0.18)):
            a = contrib[i].abs()
            assert float(a[:19].sum() / a.sum()) > front and float(a[:, :3].sum() / a.sum()) > inner
            rows = a.sum(1)
            assert set(torch.topk(rows, 2).indices.tolist()) == {15, 16}       # z' in [-6.6, 19.1] mm
        _, cc = mc.components(lnp)
        _, ce = me.components(lnp)
        klc = {k: cp.kl_div(cc[k], ce[k]) for k in cc}
        assert float(klc["spike"].min()) > 5e-3 and float(klc["shower"].max()) < 5e-5
        assert float(klc["track"].min()) > 3e-4 and float(klc["end"].max()) < 2e-5 and float(klc["far"].max()) < 2e-5


def test_exact_mode_converges_with_the_number_of_nodes(legacy):
    """Against a reference with n_z = 17, n_zf = 192, n_r = 12 (float64). The shower (smooth plateau profile) is
    converged to 5e-9 at n_z = 5; what remains comes from the kinks of the legacy track profile (clamps, Bragg cusp
    at z' = R(T)) in the row that contains them and falls with the number of fine nodes like n_zf^-2.
    Measured max |d ln P| over cells with P > 1e-9, n_z = 5, n_zf = 16:
        p < 0.3 GeV: 3e-3;  0.3-1.5 GeV: 3e-4;  p >= 2 GeV: 1e-6.     KL(ref || model) <= 2e-7 everywhere."""
    lnp = lnp_of(dt=F64)
    ref = mk(legacy, "legacy_exact", grid=ci.Grid(n_z=17, n_zf=192, n_r=12, dtype=F64))
    with torch.no_grad():
        E = ref(lnp)
        _, comp_ref = ref.components(lnp)
        err, klv = {}, {}
        for n_z, n_zf in ((3, 16), (5, 16), (5, 64)):
            mdl = mk(legacy, "legacy_exact", grid=ci.Grid(n_z=n_z, n_zf=n_zf, dtype=F64))
            e = mdl(lnp)
            err[(n_z, n_zf)] = ((e - E).abs() * massive(E)).amax((1, 2))
            klv[(n_z, n_zf)] = cp.kl_div(E, e)
            if (n_z, n_zf) == (5, 16):
                sh = mdl.components(lnp)[1]["shower"]
                assert float(((sh - comp_ref["shower"]).abs() * massive(comp_ref["shower"])).max()) < 5e-8
        d = err[(5, 16)]
        assert float(d[8:].max()) < 5e-6 and float(d[3:8].max()) < 1e-3 and float(d[:3].max()) < 1e-2
        assert float(klv[(5, 16)].max()) < 1e-6 and float(klv[(3, 16)].max()) < 1e-6
        assert float(err[(5, 64)].max()) < 0.35 * float(d.max())               # more fine nodes: the kink rows improve
        assert float(err[(5, 64)][8:].max()) < 1e-6
        # the cell-centre rule is not a member of this sequence: it is 3 to 6 orders of magnitude further away
        c = mk(legacy, "legacy_centre")(lnp)
        assert float((cp.kl_div(c, E) / klv[(5, 16)].clamp_min(1e-12)).min()) > 1e3


def test_float32_exact_mode_equals_float64(legacy):
    lnp = lnp_of()
    with torch.no_grad():
        for v in ("legacy_exact", "survival_exact"):
            a, b = mk(legacy, v, torch.float32)(lnp).double(), mk(legacy, v)(lnp.double())
            assert float(((a - b).abs() * massive(b)).max()) < 3e-4            # measured 1e-4 (far cells, P ~ 1e-9)
            assert float(((a - b).abs() * massive(b, 1e-5)).max()) < 5e-5
            assert float(cp.kl_div(b, a).abs().max()) < 2e-6                   # measured 3e-7


def test_survival_variant_is_a_different_named_model_and_nested_swaps_are_identities(legacy):
    """Dropping the 1 mm / 10 mm regularisers changes the map by KL 9e-6 ... 6.9e-4 (same coefficients), about half
    of it in the three innermost rings. Burr (c, k) = (2, 1) is the Cauchy term and Weibull b = 1 the exponential
    halo, so those swaps start exactly on the survival variant."""
    lnp = lnp_of(dt=F64)
    with torch.no_grad():
        e, s = mk(legacy, "legacy_exact")(lnp), mk(legacy, "survival_exact")(lnp)
        kl = cp.kl_div(e, s)
        assert 5e-6 < float(kl.min()) and float(kl.max()) < 1e-3 and float(kl[6]) > 4e-4
        contrib = (e.exp() * (e - s)).abs()
        assert 0.3 < float(contrib[6][:, :3].sum() / contrib[6].sum()) < 0.6
        for latf in (("gauss", "burr", "weibull"), ("weibull_core", "student", "weibull")):
            n = mk(legacy, "survival_exact", shower_lat=latf)(lnp)
            assert float((n - s).abs().max()) < 1e-6                           # 3e-8: float32 storage of the raw shapes
        t = mk(legacy, "survival_exact", lat_norm="term")(lnp)                 # same numbers, other meaning of the weights
        assert 1e-12 < float(cp.kl_div(s, t).max()) < 1e-3


def test_shower_depth_profile_is_the_depth_marginal_of_the_shower_component(legacy):
    lnp = lnp_of([0.3, 1.0, 8.0], F64)
    with torch.no_grad():
        for v, kw in (("legacy_exact", {}), ("survival_exact", dict(shower_long="emg")), ("legacy_centre", {})):
            m = mk(legacy, v, **kw)
            marg = torch.logsumexp(m.component_map(lnp, "shower"), 2)
            assert torch.allclose(m.shower_depth_profile(lnp), marg, atol=1e-10), v


# ----------------------------------------------------------------------------------------------------------------------
# initialisation of swapped families from the legacy fit
# ----------------------------------------------------------------------------------------------------------------------
def test_init_shower_long_moves_an_emg_profile_onto_the_legacy_shower(legacy):
    """Full schedule (steps (1000, 1500), 400 candidates): depth-marginal KL 0.92 -> 2.1e-3, map KL 1.2e-4 (6 s on
    the CPU). Here a short schedule; the thresholds are loose accordingly."""
    torch.manual_seed(0)
    ref = mk(legacy, "survival_exact", torch.float32)
    m = mk(legacy, "survival_exact", torch.float32, shower_long="emg")
    before = {n: p.detach().clone() for n, p in m.provider.named_parameters()}
    lnp = cp.default_lnp(16)
    out = cp.init_shower_long(m, ref, lnp=lnp, steps=(250, 250), n_search=100)
    assert out["kl_start"] > 0.5 and out["kl_final"] < 0.01 and out["kl_final"] < out["kl_search"] < out["kl_start"]
    assert out["kl_map"] < 1e-3 and abs(out["kl_shower"] - out["kl_final"]) < 1e-4
    # only the rows of the new family moved, and the provider is back in place
    rows = [m.provider.index(n) for n in m.new_names]
    keep = torch.ones(len(m.provider.names), dtype=torch.bool)
    keep[rows] = False
    assert torch.equal(m.provider.coef.detach()[keep], before["coef"][keep])
    assert not torch.equal(m.provider.coef.detach()[~keep], before["coef"][~keep])
    assert isinstance(m.provider, PolyProvider) and m.provider.coef.requires_grad


def test_init_from_target_recovers_perturbed_rows_at_map_level(legacy):
    torch.manual_seed(0)
    ref = mk(legacy, "legacy_centre", torch.float32)
    m = mk(legacy, "legacy_centre", torch.float32)
    names = ["log_sp_sz", "log_latt"]
    with torch.no_grad():
        m.provider.coef[m.provider.index("log_sp_sz"), 0] += 0.5
        m.provider.coef[m.provider.index("log_latt"), 0] -= 0.4
    out = cp.init_from_target(m, ref, names, lnp=cp.default_lnp(12), steps=(120, 60), lr=(0.05, 0.01))
    assert out["kl_start"] > 1e-3 and out["kl_final"] < 0.02 * out["kl_start"]
    c_new, c_ref = m.provider.coef.detach(), ref.provider.coef.detach()
    assert abs(float(c_new[m.provider.index("log_latt"), 0]) - float(c_ref[ref.provider.index("log_latt"), 0])) < 0.05
    with pytest.raises(TypeError):
        m.set_provider(ConcatProvider([m.provider]))
        cp.init_from_target(m, ref, names)


# ----------------------------------------------------------------------------------------------------------------------
# pure mixture
# ----------------------------------------------------------------------------------------------------------------------
def _mixture(K, lf, lt, logA=None, dt=torch.float32, seed=1, spread=0.3, **kw):
    mm = cp.MixtureModel(None, K=K, long=lf, lat=lt, logA=logA, grid=ci.Grid(dtype=dt), **kw)
    prov = PolyProvider(mm.names, 3, mm.defaults)
    with torch.no_grad():
        prov.coef[:, 1:].normal_(0, spread, generator=torch.Generator().manual_seed(seed))
    mm.set_provider(prov)
    return mm.to(dt)


MIX = [("gengamma", "burr"), ("emg", "weibull"), ("plateau", "gauss"), ("gamma", "cauchy"), ("invgauss", "exp"),
       ("lognormal", "student"), ("experlang2", "exp_reg"), ("burr", "weibull_core")]


@pytest.mark.parametrize("lf,lt", MIX, ids=[a + "x" + b for a, b in MIX])
@pytest.mark.parametrize("integration", ["exact", "centre"])
def test_mixture_model_is_normalised_finite_and_ordered(legacy, lf, lt, integration):
    _, logA, _ = legacy
    mm = _mixture(3, lf, lt, logA, integration=integration)
    lnp = lnp_of([0.0005, 0.01, 0.0895, 0.5, 2.0, 20.0, 50.0])                # random cubic trajectories, x down to -4.1
    for q in mm.parameters():
        q.requires_grad_(True)
    out = mm(lnp)
    assert out.shape == (7, 128, 32) and torch.isfinite(out).all()
    assert float((out.detach().double().exp().sum((1, 2)) - 1).abs().max()) < 3e-6      # float32; random extreme trajectories
    w = torch.softmax(torch.randn(7, 4096, generator=torch.Generator().manual_seed(0)), 1).reshape(out.shape)
    (-(w * out).sum()).backward()
    assert torch.isfinite(mm.provider.coef.grad).all()
    # label ordering: the ordering key increases with the component index at every momentum
    key = mm.longs[0].ORDER_KEY[0]
    keys = torch.stack([nat[key].reshape(-1) for nat in mm.naturals(mm.theta(lnp))])
    assert (keys[1:] >= keys[:-1]).all() and (keys[1:] > keys[:-1]).double().mean() > 0.8


def test_mixture_model_conventions(legacy):
    mm = _mixture(4, "gengamma", "burr", None, F64, spread=0.0)
    lnp = lnp_of([0.5, 5.0], F64)
    with torch.no_grad():
        lw, comp = mm.components(lnp)
        assert torch.allclose(torch.stack(lw).exp(), torch.full((4, 2), 0.25, dtype=F64), atol=1e-12)   # zero logits: 1/K
        for c in comp:
            assert float((c.exp().sum((1, 2)) - 1).abs().max()) < 1e-12
        modes = torch.stack([n["m"].reshape(-1) for n in mm.naturals(mm.theta(lnp))])
        assert torch.allclose(modes[1:] - modes[:-1], torch.full((3, 2), 100.0, dtype=F64), atol=1.0)   # default steps of 100 mm
    per = len(mm.longs[0].names) + 2 + 2                                      # depth family + (log_r, s_r) + burr (c, k)
    assert len(mm.names) == 4 * per + 3 and mm.n_params() == 4 * len(mm.names)
    assert not any("." in n for n in mm.names)
    free = cp.MixtureModel(None, K=2, long="emg", lat="exp", order=False, depth="quadratic")
    assert "c1_q_r" in free.names and free.defaults["c1_mu"] == free.defaults["c0_mu"]
    with torch.no_grad():                                                     # fine z nodes: same map for a smooth profile
        a = _mixture(2, "emg", "gauss", None, F64, spread=0.0)(lnp)
        b = _mixture(2, "emg", "gauss", None, F64, spread=0.0, fine=True)(lnp)
    assert float(((a - b).abs() * massive(a)).max()) < 1e-6


def test_permuting_unordered_components_leaves_the_map_unchanged_but_not_ordered_ones():
    """The label symmetry itself, and that ``order=True`` breaks it by construction."""
    lnp = lnp_of([0.7, 6.0], F64)
    mm = _mixture(3, "emg", "gauss", None, F64, order=False)
    with torch.no_grad():
        mm.provider.coef[:, 0] += torch.randn(len(mm.names), generator=torch.Generator().manual_seed(5)).double() * 0.5
        perm = copy.deepcopy(mm)
        c = perm.provider.coef
        i0 = [perm.provider.index(n) for n in perm.names if n.startswith("c0_")]
        i2 = [perm.provider.index(n) for n in perm.names if n.startswith("c2_")]
        tmp = c[i0].clone()
        c[i0] = c[i2]
        c[i2] = tmp
        # with all-equal weights (logits at their defaults) swapping two components is a pure relabelling
        for k in (1, 2):
            mm.provider.coef[mm.provider.index(f"mix_logit{k}")] = 0.0
            perm.provider.coef[perm.provider.index(f"mix_logit{k}")] = 0.0
        assert torch.allclose(mm(lnp), perm(lnp), atol=1e-10)
        comp_a, comp_b = mm.components(lnp)[1], perm.components(lnp)[1]
        assert torch.allclose(comp_a[0], comp_b[2], atol=1e-10) and not torch.allclose(comp_a[0], comp_b[0], atol=1e-3)
        # ordered model: the same exchange of raw rows gives a different map (the rows mean different things)
        om = _mixture(3, "emg", "gauss", None, F64, order=True)
        op = copy.deepcopy(om)
        i0 = [op.provider.index(n) for n in op.names if n.startswith("c0_")]
        i2 = [op.provider.index(n) for n in op.names if n.startswith("c2_")]
        tmp = op.provider.coef[i0].clone()
        op.provider.coef[i0] = op.provider.coef[i2]
        op.provider.coef[i2] = tmp
        assert float(cp.kl_div(om(lnp), op(lnp)).min()) > 1e-4


def test_component_model_accepts_the_neural_provider_of_sfm_models(legacy):
    """The provider contract in practice: a ParamNet (sfm.models, written separately) warm-started in function space
    from the polynomial provider drives the same ComponentModel. Skipped if that module is absent."""
    paramnet = pytest.importorskip("sfm.models.paramnet")
    core, logA, _ = legacy
    torch.manual_seed(0)
    m = mk(legacy, "survival_exact", torch.float32)
    poly = m.provider
    net = paramnet.ParamNet(m.momentum_dependent, m.legacy_constant, width=16, depth=2, v0=float(poly.v0), vs=float(poly.vs))
    assert fam.params.is_param_provider(net) and sorted(net.names) == sorted(m.names)
    net.fit_to(poly, cp.default_lnp(200), steps=1500, lbfgs_iter=300)
    m2 = cp.build_variant("survival_exact", core, logA, provider=net)
    lnp = cp.default_lnp(24)
    with torch.no_grad():
        kl = cp.kl_div(m(lnp), m2(lnp))
    assert float(kl.median()) < 1e-3 and torch.isfinite(kl).all()             # measured: mean 7e-6, max 1e-4 (the net's fit)
    assert m2.n_params() == sum(p.numel() for p in net.parameters())
    (-m2(lnp).mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


# ----------------------------------------------------------------------------------------------------------------------
# detector-frame halo inside the component model
# ----------------------------------------------------------------------------------------------------------------------
def test_detector_halo_is_mixed_in_after_the_acceptance(legacy):
    core, logA, _ = legacy
    th = np.random.default_rng(0).uniform(0.0524, 0.6283, 5000)
    halo = fam.DetHalo.from_samples(th, n_theta=8, n_phi=8)
    m = cp.build_variant("survival_exact", core, logA, det_halo=halo)
    assert "logit_dh" in m.names and m.new_names == ["logit_dh"]
    assert m.n_params() == 101 + 1 + 2                                        # weight + (L, R) of the template
    lnp = lnp_of([0.5, 5.0])
    base = cp.build_variant("survival_exact", core, logA)
    out = m(lnp)
    with torch.no_grad():
        w = torch.sigmoid(torch.tensor(-5.0))
        want = torch.log((1 - w) * base(lnp).exp() + w * halo().exp()[None])
        assert torch.allclose(out, want, atol=2e-5)
        assert float((out.double().exp().sum((1, 2)) - 1).abs().max()) < 1e-5
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in halo.parameters())


# ----------------------------------------------------------------------------------------------------------------------
# cost of one forward + backward at batch 256 (GPU, only if idle)
# ----------------------------------------------------------------------------------------------------------------------
def _gpu_idle():
    if not torch.cuda.is_available():
        return False
    free, total = torch.cuda.mem_get_info()
    try:
        util = torch.cuda.utilization()
    except Exception:
        util = 0
    return (total - free) < 1.0 * 2 ** 30 and util < 20


def test_forward_backward_cost_at_batch_256(legacy):
    """Budget on the RTX 4060 Laptop: <= 1.5 GB and <= 0.15 s per forward + backward of a full model at 256 momenta.
    Measured: legacy_centre 168 MiB / 6 ms; legacy_exact 766 MiB / 51 ms; survival_exact 706 MiB / 42 ms;
    survival_exact + (gauss, burr, weibull) 788 MiB / 44 ms; + emg 761 MiB / 48 ms; mixture K = 5 914 MiB / 52 ms."""
    if not _gpu_idle():
        pytest.skip("GPU busy or absent: the cost assertion is not meaningful")
    core, logA, _ = legacy
    dev = torch.device("cuda")
    torch.manual_seed(0)
    lnp = torch.log(torch.rand(256, device=dev) * 19.9 + 0.09)
    tgt = torch.softmax(torch.randn(256, 4096, device=dev), 1).reshape(256, 128, 32)
    report = {}
    cases = [("legacy_exact", {}), ("survival_exact", {}), ("survival_exact", dict(shower_lat=("gauss", "burr", "weibull"))),
             ("survival_exact", dict(shower_long="emg"))]
    models = [(v + str(kw), cp.build_variant(v, core, logA, **kw).to(dev)) for v, kw in cases]
    mm = cp.MixtureModel(None, K=5, long="gengamma", lat="burr", logA=logA)
    mm.set_provider(PolyProvider(mm.names, 3, mm.defaults))
    models.append(("mixture K=5", mm.to(dev)))
    for name, model in models:
        model.train()
        for q in model.parameters():
            q.requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            loss = -(tgt * model(lnp)).sum() / 256
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        report[name] = ((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1e3 * float(np.median(ts[2:])))
        assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
        del loss
        model.cpu()
        torch.cuda.empty_cache()
    print("\nforward+backward at batch 256 (peak MiB, ms): " + "; ".join(f"{k}: {a:.0f}, {b:.0f}" for k, (a, b) in report.items()))
    for name, (mem, ms) in report.items():
        assert mem < 1500 and ms < 150, (name, mem, ms)
