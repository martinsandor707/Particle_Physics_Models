"""sfm.models.dist_factory (Track A, AUDIT.md section 4.1): every model of the pre-registered list is built by the
``component`` / ``mixture`` factories on the CPU, is normalised, has finite gradients and the pre-registered number of
free constants; ``legacy_centre`` with the legacy coefficients is the legacy formula's map; the nested swaps start on
A2s; the two new acceptance templates are finite and in their expected ranges. Data-free except for the cached
templates (read from results/m2a_dist/templates/ if present)."""
import math

import numpy as np
import pytest
import torch

from sfm import models
from sfm.models import dist_factory as dfac
from sfm.families import components as cp

P = [0.0895, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
LN20 = math.log(20.0)
HAVE_TEMPLATES = all((dfac.TEMPLATE_DIR / f"{k}.pt").exists() for k in dfac.NEW_TEMPLATES)

# (id, factory, kwargs, free constants of the poly version per AUDIT 4.1)
SPECS = [
    ("A0", "component", dict(variant="legacy_centre"), 101),
    ("A1", "component", dict(variant="legacy_centre", provider="net"), 699),
    ("A2e", "component", dict(variant="legacy_exact"), 101),
    ("A2s", "component", dict(variant="survival_exact"), 101),
    ("A2n", "component", dict(variant="survival_exact", provider="net"), 699),
    ("A4", "component", dict(shower_long="emg"), 93),
    ("A5", "component", dict(shower_long="gengamma", shower_long_kw={"soft": "free"}), 101),
    ("A7", "component", dict(shower_lat=["gauss", "burr", "exp"]), 103),
    ("A8", "component", dict(shower_lat=["gauss", "cauchy", "weibull"]), 102),
    ("A9", "component", dict(det_halo=True, far=True), 104),
    ("A10", "mixture", dict(K=3, long="emg", lat="burr", init="none"), 74),
    ("A4t", "component", dict(shower_long="emg_tied"), 89),
    ("A5h", "component", dict(shower_long="gengamma", shower_long_kw={"soft": 2.0}), 97),
    ("A6i", "component", dict(shower_long="invgauss", shower_long_kw={"soft": "free"}), 97),
    ("A6l", "component", dict(shower_long="lognormal", shower_long_kw={"soft": "free"}), 97),
    ("A6b", "component", dict(shower_long="burr", shower_long_kw={"soft": "free"}, degrees={"sh_rk": 0}), 98),
    ("A7s", "component", dict(shower_lat=["gauss", "student", "exp"]), 102),
    ("A7l", "component", dict(shower_lat=["gauss", "cauchy", "burr"],
                              lat_family_kw=[{}, {}, {"c": 1.0, "init": [1.0, 20.0]}], coef_shift={"log_lam": LN20}), 102),
    ("A8c", "component", dict(shower_lat=["weibull_core", "cauchy", "exp"]), 102),
    ("A78", "component", dict(shower_lat=["gauss", "burr", "weibull"]), 104),
    ("A9r", "component", dict(det_halo=True, far=False), 98),
    ("A3f", "component", dict(template="front"), 101),
    ("A3c", "component", dict(template="back_corrected"), 101),
    ("A3x", "component", dict(template="full_box"), 101),
    ("A10-2", "mixture", dict(K=2, long="emg", lat="burr", init="none"), 48),
    ("A10-4", "mixture", dict(K=4, long="emg", lat="burr", init="none"), 100),
    ("A10-5", "mixture", dict(K=5, long="emg", lat="burr", init="none"), 126),
    ("A10g", "mixture", dict(K=3, long="gengamma", long_kw={"soft": "free"}, lat="burr", init="none"), 98),
    ("A12", "component", dict(last_ring=True), 102),
]


@pytest.fixture(autouse=True)
def _synthetic_theta(monkeypatch):
    """The detector-frame halo needs the training angles; a synthetic sample in the training range is enough here."""
    th = np.random.default_rng(0).uniform(0.0524, 0.6283, 4000)
    monkeypatch.setattr(dfac, "_theta_train", lambda: th)


def ctx(skip=True):
    return models.BuildContext(device="cpu", train=None, seed=0, log=lambda *a: None, skip_init=skip)


def lnp_of(p=P):
    return torch.log(torch.tensor(p, dtype=torch.float32))


def build(fac, kw, skip=True):
    torch.manual_seed(0)
    return models.build(fac, ctx(skip), **kw)


def test_factories_registered():
    assert "component" in models.FACTORIES and "mixture" in models.FACTORIES


@pytest.mark.parametrize("rid,fac,kw,k", SPECS, ids=[s[0] for s in SPECS])
def test_every_model_builds_normalised_with_finite_gradients(rid, fac, kw, k):
    if kw.get("template") in dfac.NEW_TEMPLATES and not HAVE_TEMPLATES:
        pytest.skip("new templates not computed yet (scripts/run_m2a.py templates)")
    m = build(fac, kw)
    assert m.n_params() == k, (rid, m.n_params())
    lnp = lnp_of()
    out = m(lnp)
    assert out.shape == (len(P), 128, 32) and torch.isfinite(out).all()
    lse = torch.logsumexp(out.detach().double().reshape(len(P), -1), 1)
    assert float(lse.abs().max()) < 1e-5, lse.tolist()
    w = torch.softmax(torch.randn(len(P), 4096, generator=torch.Generator().manual_seed(1)), 1).reshape(out.shape)
    (-(w * out).sum()).backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


@pytest.mark.parametrize("rid,fac,kw", [(s[0], s[1], s[2]) for s in SPECS if s[0] in ("A4", "A7", "A9", "A10")],
                         ids=["A4n", "A7n", "A9n", "A10n"])
def test_net_versions_of_the_primaries(rid, fac, kw):
    """The net provider of a primary: ParamNet with names_net / names_const of ``net_name_split``, fitted to the poly
    provider; the map stays close, normalised, with finite gradients; constants 304 + 17 n_net + n_const."""
    m = build(fac, kw)
    poly = (m.base if hasattr(m, "base") else m).provider
    ref = build(fac, kw)
    net_names, const_names = dfac.net_name_split(m)
    assert not set(net_names) & set(const_names)
    assert all(n not in net_names for n in dfac.radial_shape_names(m))
    torch.manual_seed(0)
    net, rep = dfac.make_paramnet(m, poly)                     # the specified fit: 2000 Adam + 500 L-BFGS
    m.set_provider(net)
    assert m.n_params() == 304 + 17 * len(net_names) + len(const_names) + (2 if kw.get("det_halo") else 0)
    lnp = lnp_of()
    out = m(lnp)
    lse = torch.logsumexp(out.detach().double().reshape(len(P), -1), 1)
    assert float(lse.abs().max()) < 1e-5
    with torch.no_grad():
        kl = cp.kl_div(ref(lnp), out.detach())
    print(rid, "KL(poly || net) per momentum", [f"{v:.2e}" for v in kl.tolist()], "fit", rep["loss"])
    assert torch.isfinite(kl).all() and float(kl.max()) < 0.05, kl.tolist()
    (-out.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_legacy_centre_reproduces_the_legacy_formula():
    """factory ``component(variant="legacy_centre", template="stored")`` with the legacy coefficients against
    ``shapelab.WithAcceptance(core, abar)`` (float32, the families test's tolerances)."""
    import shapelab
    core, logA = dfac.legacy_core()
    formula = shapelab.WithAcceptance(core, torch.exp(logA)).eval()
    with torch.no_grad():
        formula.logA.copy_(logA)
    m = build("component", dict(variant="legacy_centre"))
    lnp = lnp_of()
    with torch.no_grad():
        ref, out = formula(lnp), m(lnp)
    d = (out - ref).abs()
    assert float(d[ref.exp() > 1e-9].max()) < 2e-4
    assert float(cp.kl_div(ref.double(), out.double()).abs().max()) < 2e-6


def test_stored_template_is_the_legacy_acceptance():
    """ln of abar.pt (floored at 1e-6) equals the legacy checkpoint's logA up to 1 ulp of the log (49 cells differ
    by 2.4e-7: the checkpoint's buffer was computed on another device); the factory uses the checkpoint's buffer
    for template="stored", so legacy_centre is the legacy map bit for bit in the acceptance."""
    core, logA = dfac.legacy_core()
    A = dfac.acceptance_template("stored", "cpu")
    assert float((torch.log(A.clamp_min(1e-6)) - logA).abs().max()) < 5e-7
    m = build("component", dict(variant="legacy_centre"))
    assert torch.equal(m.logA, logA)


def test_nested_swaps_start_exactly_on_A2s():
    """A7, A8, A78, A7s, A8c reduce to A2s at their default shapes (KL ~ float32 noise); A7l (Lomax with k = 20 and
    log_lam + ln 20, applied only when the initialisation runs) is close but not exact."""
    lnp = lnp_of()
    with torch.no_grad():
        base = build("component", dict(variant="survival_exact"))(lnp)
        for rid in ("A7", "A8", "A78", "A7s", "A8c"):
            kw = next(s[2] for s in SPECS if s[0] == rid)
            kl = cp.kl_div(base.double(), build("component", kw)(lnp).double())
            assert float(kl.abs().max()) < 1e-6, (rid, kl.tolist())
    kw = next(s[2] for s in SPECS if s[0] == "A7l")
    m = build("component", kw, skip=False)
    with torch.no_grad():
        kl = cp.kl_div(base.double(), m(lnp).double())
    print("A7l start: KL(A2s-at-legacy-coefficients || A7l) per momentum", [f"{v:.2e}" for v in kl.tolist()],
          "kl_map vs legacy formula", m.init_info["kl_map"])
    assert float(kl.max()) < 0.05, kl.tolist()


def test_last_ring_wrapper_is_the_base_at_zero_and_moves_only_the_last_ring():
    lnp = lnp_of()
    m = build("component", dict(last_ring=True))
    with torch.no_grad():
        assert torch.allclose(m(lnp), m.base(lnp), atol=1e-6)
        m.log_f_last.fill_(0.5)
        out, b = m(lnp), m.base(lnp)
        shift = (out - b)
        # inner rings move by the common normalisation only, the last ring by 0.5 more
        assert torch.allclose(shift[:, :, :-1], shift[:, :1, :1].expand_as(shift[:, :, :-1]), atol=1e-5)
        assert torch.allclose(shift[:, :, -1] - shift[:, :, 0], torch.full_like(shift[:, :, -1], 0.5), atol=1e-5)


def test_copy_coefficients_by_name():
    a = build("component", dict(variant="survival_exact"))
    b = build("component", dict(shower_lat=["gauss", "burr", "exp"]))
    with torch.no_grad():
        a.provider.coef.add_(0.01 * torch.randn_like(a.provider.coef) * a.provider.mask)
    done = dfac.copy_coefficients_by_name(b.provider, a.provider)
    assert set(done) == set(a.provider.names) and "lat1_rc" not in done
    for n in done:
        i, j = b.provider.index(n), a.provider.index(n)
        assert torch.equal(b.provider.coef[i] * b.provider.mask[i], a.provider.coef[j] * a.provider.mask[j])


def test_degree_override_only_for_new_rows():
    m = build("component", dict(shower_long="burr", shower_long_kw={"soft": "free"}, degrees={"sh_rk": 0}))
    assert m.provider.deg["sh_rk"] == 0 and m.provider.deg["sh_rscale"] == 3
    with pytest.raises(KeyError):
        build("component", dict(degrees={"log_sig": 0}))


def _synthetic_batch(B=3, zb=(600.0, 1300.0, 2000.0)):
    from types import SimpleNamespace
    R = torch.eye(3).expand(B, 3, 3).clone()
    return SimpleNamespace(R=R, lnp=torch.zeros(B), zb=torch.tensor(zb[:B]))


def test_acceptances_port_normal_incidence():
    """theta = 0 (shower frame = global frame), origin on the axis: the side faces are 2.6 m away (never reached by
    r < 700 mm); front = 1 behind -layer/2; front-and-back = 1[-layer/2 <= z' <= zb] at the sub-points; hence
    back_corrected = front wherever some sub-point lies before zb."""
    b = _synthetic_batch()
    zeros = torch.zeros(3)
    A = dfac.acceptances(b, zeros, zeros, b.zb)
    assert torch.equal(A["side"], torch.ones_like(A["side"]))
    assert torch.equal(A["all"], A["fb"])
    ze = torch.linspace(-200.0, 1450.0, 129)
    sub = (ze[:-1, None] + (ze[1] - ze[0]) * torch.tensor([0.25, 0.75])).reshape(-1)
    want_front = (sub >= -10.25).float().reshape(128, 2).mean(1)
    assert torch.allclose(A["front"][0, :, 0], want_front)
    for e in range(3):
        want_fb = ((sub >= -10.25) & (sub <= b.zb[e])).float().reshape(128, 2).mean(1)
        assert torch.allclose(A["fb"][e, :, 5], want_fb)
        below = (sub < b.zb[e]).float().reshape(128, 2).mean(1)
        assert torch.allclose(A["below_zb"][e, :, 7], below)


def test_exit_depth_port():
    zex, X0, Y0 = dfac.exit_depth(np.array([0.0, 0.5]), np.array([0.0, 0.0]), np.array([3662.4, 3662.4]))
    assert np.isinf(zex[0]) and X0[0] == 0.0
    rho0 = 3662.4 * math.tan(0.5)
    assert abs(zex[1] - (dfac.X_E - rho0) / math.sin(0.5)) < 1e-9 and abs(X0[1] - rho0) < 1e-9


@pytest.mark.skipif(not HAVE_TEMPLATES, reason="new templates not computed yet (scripts/run_m2a.py templates)")
def test_new_templates_are_finite_and_in_range():
    """back_corrected = mean A_fb / mean 1[z' < zb], full_box = mean A_fbs / mean 1[z' < zb] (training events):
    finite, >= 0, at most slightly above 1 (the tilted back face lets large-r points behind zb count while 1[z' < zb]
    does not), full_box <= back_corrected, and the recomputed front+back template reproduces the stored abar.pt."""
    bc, fx = (dfac.acceptance_template(k, "cpu") for k in dfac.NEW_TEMPLATES)
    for A in (bc, fx):
        assert torch.isfinite(A).all() and float(A.min()) >= 0.0 and float(A.max()) <= 1.1
    assert bool((fx <= bc + 1e-6).all())
    import json
    meta = json.loads((dfac.TEMPLATE_DIR / "templates.json").read_text())
    assert meta["fb_vs_stored_abar"]["max_abs"] < 0.02
