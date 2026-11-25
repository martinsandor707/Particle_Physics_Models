"""Tests of sfm.models: shapes, normalisation, warm starts, ensembles, residual tables, the factory registry.

All on the CPU (models are moved there; ``common.FieldModel`` builds its feature buffer on ``common.DEVICE`` first).
Real files are only read: ``mlp_std.pt`` and the formula checkpoint (read-only, shape-form-proton/), the cache
metadata. No data split is loaded.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

import common as C
from sfm import models, paths
from sfm.models import (Ensemble, FieldAcc, FieldNet, ParamNet, ResidualTable, centring, kinetic, kl_warm_start)
from sfm.models.paramnet import ParamProvider

CPU = torch.device("cpu")
LNP = torch.log(torch.tensor([0.2, 0.9, 3.0, 11.0, 19.5]))


def normalised(logP, atol=2e-5):
    s = torch.logsumexp(logP.reshape(len(logP), -1).double(), 1)
    return bool(torch.isfinite(logP).all()) and float(s.abs().max()) < atol


def small_field(seed, hidden=8, n_layers=2):
    torch.manual_seed(seed)
    net = FieldNet(hidden, n_layers, n_freq=2).to(CPU)
    with torch.no_grad():
        net.mlp[-1].weight.mul_(5.0)                                 # visible structure
    return net


class Poly(ParamProvider):
    """A cubic-in-x provider (the shape of the families' polynomial provider), used as a warm-start target."""

    def __init__(self, names, v0, vs):
        super().__init__()
        self.names = list(names)
        g = torch.Generator().manual_seed(0)
        self.coef = nn.Parameter(torch.randn(len(names), 4, generator=g) * torch.tensor([1.0, 0.5, 0.3, 0.2]))
        self.v0, self.vs = v0, vs

    def forward(self, lnp):
        x = (torch.log(kinetic(lnp)) - self.v0) / self.vs
        return torch.stack([x ** k for k in range(4)], 1) @ self.coef.T


# ----------------------------------------------------------------------------------------------------------------------
# ParamNet
# ----------------------------------------------------------------------------------------------------------------------
def test_paramnet_contract_and_shapes():
    net = ParamNet(["a", "b", "c"], ["k1", "k2"], width=16, depth=2, v0=-0.3, vs=3.3, init={"b": 2.0, "k2": -1.5})
    assert isinstance(net, ParamProvider) and models.is_param_provider(net)
    out = net(LNP)
    assert out.shape == (len(LNP), 5) and net.names == ["a", "b", "c", "k1", "k2"]
    # fresh net = the constant function init (last layer starts at zero weights)
    assert torch.equal(out, torch.tensor([0.0, 2.0, 0.0, 0.0, -1.5]).expand(len(LNP), 5))
    assert net.n_params() == (1 * 16 + 16) + (16 * 16 + 16) + (16 * 3 + 3) + 2
    # x = (ln T - v0)/vs with T = p^2/(sqrt(p^2+m^2)+m), unclamped (also far below the training range)
    lnp = torch.log(torch.tensor([0.0005, 0.0734, 1.0, 20.0]))
    p = lnp.exp().double()
    T = p * p / (torch.sqrt(p * p + 0.139570 ** 2) + 0.139570)
    assert torch.allclose(net.x(lnp).double(), (torch.log(T) + 0.3) / 3.3, rtol=1e-5)
    # order= permutes the columns; constants do not depend on ln p once trained values differ
    perm = ParamNet(["a", "b", "c"], ["k1", "k2"], v0=-0.3, vs=3.3, order=["k2", "a", "k1", "c", "b"])
    perm.load_state_dict(net.state_dict())
    assert torch.equal(perm(LNP), net(LNP)[:, [4, 0, 3, 2, 1]])
    with torch.no_grad():
        net.mlp[-1].weight.normal_()
    out = net(LNP)
    assert not torch.allclose(out[0, :3], out[-1, :3]) and torch.equal(out[0, 3:], out[-1, 3:])
    for bad in ({"names_net": ["a"], "names_const": ["a"]}, {"names_net": [], "names_const": []},
                {"names_net": ["a"], "names_const": [], "order": ["b"]}, {"names_net": ["a"], "activation": "relu6"},
                {"names_net": ["a"], "init": {"zz": 1.0}}):
        with pytest.raises(ValueError):
            ParamNet(**bad)


def test_paramnet_constants_only_and_eval_inference():
    net = ParamNet([], ["k"], init={"k": 0.7})
    net.eval()
    with torch.inference_mode():
        out = net(LNP)
    assert out.shape == (len(LNP), 1) and float(out.max()) == float(out.min()) == pytest.approx(0.7)


def test_paramnet_fit_to_polynomial_trajectories():
    v0, vs = -0.32585955, 3.31461334
    target = Poly(["mu", "log_s", "frac", "const"], v0, vs)
    with torch.no_grad():
        target.coef[3, 1:] = 0.0                                     # "const" is momentum independent
    torch.manual_seed(0)
    net = ParamNet(["mu", "log_s", "frac"], ["const"], width=16, depth=2, v0=v0, vs=vs)
    grid = torch.linspace(math.log(0.0895), math.log(20.0), 200)
    res = net.fit_to(target, grid, steps=1000, lr=1e-2, lbfgs_iter=500)     # measured: worst rms ~1e-3 of the spread
    with torch.no_grad():
        err = (net(grid) - target(grid)).abs()
        spread = target(grid).std(0)
    assert float(err[:, 3].max()) < 1e-6                              # constant: closed-form weighted mean
    assert all(res["rms"][n] < 0.005 * float(spread[j]) for j, n in enumerate(["mu", "log_s", "frac"])), res
    # per-point weights (G,) and per-point-per-name weights (G, P)
    w = torch.exp(-((grid - 1.0) ** 2))
    net.fit_to(target, grid, weights=w, steps=10, lbfgs_iter=10)
    net.fit_to(target, grid, weights=w[:, None].expand(-1, 4), steps=10, lbfgs_iter=0)
    with pytest.raises(KeyError):
        ParamNet(["unknown"], v0=v0, vs=vs).fit_to(target, grid, steps=1)
    with pytest.raises(ValueError):
        net.fit_to(target, grid, weights=torch.ones(3), steps=1)


def _fit_to_before_fix(net, provider, lnp_grid, steps=2000, lr=1e-2, lbfgs_iter=500):
    """ParamNet.fit_to as it was before the scale-floor fix (weights=None path), verbatim, for the bit-identity test."""
    dev = net.v0.device
    lnp_grid = torch.as_tensor(lnp_grid, dtype=torch.float32, device=dev).reshape(-1)
    G = lnp_grid.numel()
    provider.eval()
    with torch.no_grad():
        full = provider(lnp_grid).to(dev, torch.float32)
    idx = torch.tensor([provider.names.index(n) for n in net.names], device=dev)
    target = full.index_select(1, idx)
    P = len(net.names)
    w = torch.ones(G, P, device=dev)
    w = w / w.sum(0, keepdim=True)
    mean = (w * target).sum(0)
    var = (w * (target - mean) ** 2).sum(0)
    w = w / torch.where(var > 0, var, torch.ones_like(var))
    col = {n: j for j, n in enumerate(net.names)}
    with torch.no_grad():
        for n in net.names_const:
            net.const[n].copy_(mean[col[n]])
        last = net.mlp[-1]
        last.weight.zero_()
        last.bias.copy_(mean[[col[n] for n in net.names_net]])
    net_cols = torch.tensor([col[n] for n in net.names_net], device=dev, dtype=torch.long)
    xg = net.x(lnp_grid)[:, None]
    tw, tt = w.index_select(1, net_cols), target.index_select(1, net_cols)

    def objective():
        return (tw * (net.mlp(xg) - tt) ** 2).sum()

    net.train()
    opt = torch.optim.Adam(net.mlp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps), eta_min=lr * 0.05)
    for _ in range(int(steps)):
        loss = objective()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    saved = {k: v.detach().clone() for k, v in net.mlp.state_dict().items()}
    with torch.no_grad():
        before = float(objective())
    opt2 = torch.optim.LBFGS(net.mlp.parameters(), lr=1.0, max_iter=min(100, int(lbfgs_iter)), history_size=50,
                             line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        opt2.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    for _ in range(math.ceil(int(lbfgs_iter) / 100)):
        opt2.step(closure)
    with torch.no_grad():
        after = float(objective())
    if not (math.isfinite(after) and after <= before):
        net.mlp.load_state_dict(saved)
    with torch.no_grad():
        err = net(lnp_grid) - target
        wn = (w / w.sum(0, keepdim=True))
        rms = torch.sqrt((wn * err ** 2).sum(0))
    return {"rms": {n: float(rms[j]) for j, n in enumerate(net.names)}, "loss": float((w * err ** 2).sum())}


V0, VS = -0.32585955, 3.31461334
GRID = torch.linspace(math.log(0.0895), math.log(20.0), 120)


def test_fit_to_without_constant_columns_is_bit_identical_to_the_code_before_the_fix():
    """(iii) No column at the scale floor: the fix changes nothing, bit for bit (weights, rms, loss)."""
    target = Poly(["mu", "log_s", "frac", "c2"], V0, VS)                       # four genuinely varying columns
    torch.manual_seed(3)
    a = ParamNet(["mu", "log_s", "frac"], ["c2"], v0=V0, vs=VS)
    torch.manual_seed(3)
    b = ParamNet(["mu", "log_s", "frac"], ["c2"], v0=V0, vs=VS)
    ra = a.fit_to(target, GRID, steps=400, lbfgs_iter=200)
    rb = _fit_to_before_fix(b, target, GRID, steps=400, lbfgs_iter=200)
    assert ra["constant_columns"] == [] and ra["near_floor_columns"] == {}
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert ra["rms"] == rb["rms"] and ra["loss"] == rb["loss"]


def test_fit_to_constant_net_column_is_reproduced_and_does_not_spoil_the_others():
    """(i) + (ii) A net output whose target is constant in momentum (the A9n case: logit_dh, a degree-0 row among the
    net outputs). Before the fix its normalisation weight was 1/(float rounding) ~ 1e14 and the other columns went
    unfitted; now it is reproduced (to its own rounding) and the varying columns reach the accuracy of a fit without
    it. A bitwise-constant target is reproduced bit for bit."""
    poly = Poly(["mu", "log_s", "frac", "dh"], V0, VS)
    c = torch.tensor(-4.3)

    class Wobble(ParamProvider):                    # dh = c, every second grid point one float32 ulp higher
        names = ["mu", "log_s", "frac", "dh"]

        def forward(self, lnp):
            out = poly(lnp).clone()
            up = torch.nextafter(c, torch.tensor(math.inf))
            out[:, 3] = torch.where(torch.arange(len(lnp)) % 2 == 0, c, up)
            return out

    target = Wobble()
    with torch.no_grad():
        t = target(GRID)
    w = torch.ones(len(GRID), 4)
    w = w / w.sum(0, keepdim=True)
    mean = (w * t).sum(0)
    var = (w * (t - mean) ** 2).sum(0)
    assert 0 < float(var[3]) < 1e-12 and float(1 / var[3]) > 1e12             # the pathology: weight 1/rounding
    torch.manual_seed(0)
    net = ParamNet(["mu", "log_s", "frac", "dh"], [], v0=V0, vs=VS)
    rep = net.fit_to(target, GRID, steps=1000, lbfgs_iter=500)
    assert rep["constant_columns"] == ["dh"]
    ulp = float(torch.nextafter(c, torch.tensor(math.inf)) - c)
    with torch.no_grad():
        out = net(GRID)
        far = net(torch.log(torch.tensor([0.001, 0.05, 50.0, 500.0])))        # also outside the grid (row 0)
    assert float((out[:, 3] - t[:, 3]).abs().max()) <= ulp and bool((far[:, 3] == out[0, 3]).all())
    assert torch.equal(net.mlp[-1].weight[3], torch.zeros(16))
    # bitwise-constant target: reproduced bit for bit
    exact = Poly(["mu", "log_s", "frac", "dh"], V0, VS)
    with torch.no_grad():
        exact.coef.copy_(poly.coef)
        exact.coef[3] = torch.tensor([float(c), 0.0, 0.0, 0.0])
    torch.manual_seed(0)
    net_e = ParamNet(["mu", "log_s", "frac", "dh"], [], v0=V0, vs=VS)
    net_e.fit_to(exact, GRID, steps=50, lbfgs_iter=0)
    with torch.no_grad():
        assert torch.equal(net_e(GRID)[:, 3], exact(GRID)[:, 3])
    spread = t.std(0)
    assert all(rep["rms"][n] < 0.005 * float(spread[j]) for j, n in enumerate(["mu", "log_s", "frac"])), rep["rms"]
    # the same varying columns fitted without the constant column (same hidden-layer initialisation)
    sub = Poly(["mu", "log_s", "frac"], V0, VS)
    with torch.no_grad():
        sub.coef.copy_(poly.coef[:3])
    torch.manual_seed(0)
    ref = ParamNet(["mu", "log_s", "frac"], [], v0=V0, vs=VS)
    rr = ref.fit_to(sub, GRID, steps=1000, lbfgs_iter=500)
    # Same order of accuracy (the two optimisations are not bitwise equal: summation order in the shared layers
    # differs and the fit is chaotic at the 1e-3-of-spread level of the net, see fit_to's docstring). Measured on
    # the laptop CPU: mu 1.99e-4 vs 2.06e-4, log_s 9.6e-5 vs 2.9e-5, frac 2.09e-4 vs 1.82e-4.
    for n in ("mu", "log_s", "frac"):
        assert rep["rms"][n] <= 5.0 * rr["rms"][n], (n, rep["rms"][n], rr["rms"][n])
        assert rr["rms"][n] < 0.005 * float(spread[["mu", "log_s", "frac"].index(n)])
    assert math.isfinite(rep["loss"]) and rep["lbfgs"]["stopped"] is None
    # gradients of the fitted net are finite (the old failure left NaN model gradients)
    (net(GRID) ** 2).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_fit_to_constant_scalar_column_and_floor_rule():
    """A constant names_const column takes the float64 mean rounded to float32 = the target bit for bit; a column
    that varies by a few ulps more than the floor is NOT constant; with per-(point, name) weights nothing changes."""
    target = Poly(["mu", "k", "tiny"], V0, VS)
    with torch.no_grad():
        target.coef[1, 1:] = 0.0
        target.coef[1, 0] = 0.123456789
        target.coef[2] = torch.tensor([3.0, 1e-3, 0.0, 0.0])                 # varies by ~2e-3 over the grid
    torch.manual_seed(1)
    net = ParamNet(["mu", "tiny"], ["k"], v0=V0, vs=VS)
    rep = net.fit_to(target, GRID, steps=200, lbfgs_iter=100)
    with torch.no_grad():
        t = target(GRID)
        out = net(GRID)
    assert net.names == ["mu", "tiny", "k"]
    assert rep["constant_columns"] == ["k"] and torch.equal(out[:, 2], t[:, 1])
    assert "tiny" not in rep["constant_columns"]
    assert rep["near_floor_columns"].get("tiny") is None or rep["near_floor_columns"]["tiny"] > 1.0
    # explicit (G, P) weights: used as given, no floor, no constant handling
    torch.manual_seed(1)
    net2 = ParamNet(["mu", "tiny"], ["k"], v0=V0, vs=VS)
    rep2 = net2.fit_to(target, GRID, weights=torch.ones(len(GRID), 3), steps=10, lbfgs_iter=0)
    assert rep2["constant_columns"] == []


def test_fit_to_lbfgs_guard_restores_the_best_state(monkeypatch):
    """An L-BFGS step that raises (the earlier overflow) restores the best state seen and stops the polish."""
    target = Poly(["mu", "log_s"], V0, VS)
    torch.manual_seed(2)
    net = ParamNet(["mu", "log_s"], [], v0=V0, vs=VS)
    calls = {"n": 0}
    orig = torch.optim.LBFGS.step

    def flaky(self, closure):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("value cannot be converted to type float without overflow")
        return orig(self, closure)

    monkeypatch.setattr(torch.optim.LBFGS, "step", flaky)
    rep = net.fit_to(target, GRID, steps=100, lbfgs_iter=300)
    assert rep["lbfgs"]["stopped"].startswith("RuntimeError") and len(rep["lbfgs"]["chunk_losses"]) == 1
    assert rep["lbfgs"]["after"] <= rep["lbfgs"]["before"] and math.isfinite(rep["loss"])
    assert rep["lbfgs"]["after"] == pytest.approx(rep["lbfgs"]["chunk_losses"][0])


def test_centring_reproduces_the_formula_buffers():
    """v0, vs from the training ln p range, computed as the component formula does, equal its checkpoint buffers."""
    from sfm import data
    try:
        tr = data.load_meta()["splits"]["train"]
    except FileNotFoundError:
        pytest.skip("cache metadata not built")
    lnp = torch.log(torch.tensor([tr["p_min_GeV"], tr["p_max_GeV"]], dtype=torch.float32))
    c = centring(lnp)
    st = torch.load(paths.PION_RESULTS_DIR / "new3_std_n3.pt", map_location="cpu", weights_only=False)["state"]
    for k in ("lnp0", "lnp_s", "v0", "vs"):
        assert np.float32(c[k]) == np.float32(float(st[f"base.{k}"])), k


# ----------------------------------------------------------------------------------------------------------------------
# Field models
# ----------------------------------------------------------------------------------------------------------------------
def test_fieldnet_matches_fieldmodel():
    net = small_field(0)
    ref = C.FieldModel(8, 2, n_freq=2).to(CPU)
    ref.load_state_dict(net.state_dict())
    with torch.no_grad():
        assert torch.equal(net(LNP), ref(LNP))
        assert torch.allclose(torch.log_softmax(net.logits(LNP).reshape(len(LNP), -1), 1).reshape(net(LNP).shape),
                              net(LNP), atol=1e-6)
    assert net.n_params() == sum(p.numel() for p in ref.parameters())
    assert models.FieldNet(32, 4).n_params() == 3841                 # the benchmark architecture


def test_mlp_std_start_equals_the_benchmark():
    from sfm import legacy
    net = models.mlp_std_start(CPU)
    ref = legacy.load_mlp_std(CPU)
    assert net.training and net.n_params() == 3841
    with torch.inference_mode():
        assert torch.equal(net.eval()(LNP), ref(LNP))


def test_field_acc_normalisation_floor_and_identity():
    base = small_field(1)
    A = torch.rand(C.N_Z, C.N_R, generator=torch.Generator().manual_seed(0))
    A[:5, :] = 0.0                                                    # floored at 1e-6, not log(0)
    fa = FieldAcc(base, A=A).eval()
    with torch.inference_mode():
        lp = fa(LNP)
        assert lp.shape == (len(LNP), C.N_Z, C.N_R) and normalised(lp)
        ref = torch.log_softmax((base.logits(LNP) + torch.log(A.clamp_min(1e-6))).reshape(len(LNP), -1), 1)
        assert torch.allclose(lp.reshape(len(LNP), -1), ref, atol=1e-5)
        same = FieldAcc(base, torch.log(A.clamp_min(1e-6)))           # positional logA
        assert torch.allclose(same(LNP), lp, atol=1e-6)
        ones = FieldAcc(base, A=torch.ones(C.N_Z, C.N_R))              # A = 1 is the base model
        assert torch.allclose(ones(LNP), base(LNP), atol=1e-5)
    assert fa.n_params() == base.n_params() and "logA" in dict(fa.named_buffers())
    with pytest.raises(ValueError):
        FieldAcc(base)
    with pytest.raises(ValueError):
        FieldAcc(base, A=torch.ones(3, 3))


def test_field_acc_warm_start_reduces_kl():
    """Smooth template (a tilted entrance cut, like the real Abar): the wrapper of the target's own weights starts
    at the KL of the template (~0.14) and the warm start removes most of it (measured 0.002 after 60 steps);
    a wrapper of different weights also gets close."""
    target = small_field(2).eval()
    A = 0.02 + 0.98 * torch.sigmoid((C.ZZ.cpu() + 0.4 * C.RR.cpu() - 100.0) / 60.0)
    grid = torch.linspace(math.log(0.2), math.log(19.0), 6)
    fa = FieldAcc(small_field(2), A=A)
    res = fa.warm_start_from(target, grid, steps=60, lr=1e-2)
    assert res["kl_initial"] > 0.05 and res["kl_final"] < 0.05 * res["kl_initial"], res
    other = FieldAcc(small_field(3), A=A).warm_start_from(target, grid, steps=60, lr=1e-2)
    assert other["kl_final"] < 0.1 * other["kl_initial"], other
    # the same function from the same weights: a wrapper of a copy of the target with A = 1 has KL ~ 0
    twin = FieldAcc(small_field(2), A=torch.ones(C.N_Z, C.N_R))
    assert kl_warm_start(twin, target, grid, steps=0)["kl_initial"] < 1e-8
    with pytest.raises(ValueError, match="training momentum range"):
        fa.warm_start_from(target, torch.log(torch.tensor([0.05, 1.0])), steps=1)
    fa.warm_start_from(target, torch.log(torch.tensor([0.05, 1.0])), steps=1, allow_below_train_range=True)


def test_acceptance_templates():
    for kind in ("stored", "front", "front_back"):
        path = {"stored": paths.PION_RESULTS_DIR / "abar.pt", "front": paths.M0_DIR / "abar_front.pt",
                "front_back": paths.M0_DIR / "abar_front_back.pt"}[kind]
        if not path.exists():
            pytest.skip(f"{path} missing")
        A = models.acceptance_template(kind, CPU)
        assert A.shape == (C.N_Z, C.N_R) and float(A.min()) >= 0 and float(A.max()) <= 1 + 1e-6
    assert torch.equal(models.acceptance_template("stored", CPU), models.acceptance_template("front_back", CPU))
    with pytest.raises(ValueError):
        models.acceptance_template("back")


# ----------------------------------------------------------------------------------------------------------------------
# Ensemble
# ----------------------------------------------------------------------------------------------------------------------
def test_ensemble_of_identical_members_equals_the_member():
    m = small_field(4).eval()
    ens = Ensemble([m, m, m]).eval()
    with torch.inference_mode():
        assert torch.allclose(ens(LNP), m(LNP), atol=1e-5)
        weighted = Ensemble([m, m], log_weights=[0.3, -2.0])
        assert torch.allclose(weighted(LNP), m(LNP), atol=1e-5)
    assert ens.n_params() == 3 * m.n_params()


def test_ensemble_mixture_and_weights():
    a, b = small_field(5).eval(), small_field(6).eval()
    ens = Ensemble([a, b], log_weights=[math.log(0.25), math.log(0.75)]).eval()
    with torch.inference_mode():
        lp = ens(LNP)
        ref = torch.log(0.25 * a(LNP).double().exp() + 0.75 * b(LNP).double().exp())
    assert normalised(lp) and torch.allclose(lp.double(), ref, atol=1e-5)
    assert torch.allclose(ens.weights(), torch.tensor([0.25, 0.75]))
    tw = Ensemble([a, b], trainable_weights=True)
    assert isinstance(tw.log_weights, nn.Parameter) and tw.n_params() == a.n_params() + b.n_params() + 1
    with pytest.raises(ValueError):
        Ensemble([])
    with pytest.raises(ValueError):
        Ensemble([a, b], log_weights=[0.0])


# ----------------------------------------------------------------------------------------------------------------------
# Residual table
# ----------------------------------------------------------------------------------------------------------------------
def test_residual_table_at_zero_equals_base():
    base = small_field(7)
    knots = torch.log(torch.tensor([0.1, 1.0, 5.0, 20.0]))
    rt = ResidualTable(base, knots).eval()
    with torch.inference_mode():
        assert torch.allclose(rt(LNP), base(LNP), atol=1e-5) and normalised(rt(LNP))
    assert all(not p.requires_grad for p in base.parameters()) and rt.delta.requires_grad
    assert rt.n_params() == base.n_params() + 4 * C.N_Z * C.N_R and rt.n_trainable() == 4 * C.N_Z * C.N_R
    assert ResidualTable(small_field(7), knots, freeze_base=False).n_trainable() == base.n_params() + 4 * C.N_Z * C.N_R


def test_residual_table_interpolates_linearly_in_lnp():
    base = small_field(8).eval()
    knots = torch.tensor([0.0, 1.0, 2.0])
    rt = ResidualTable(base, knots).eval()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        rt.delta.copy_(torch.randn(3, C.N_Z, C.N_R, generator=g))
    lnp = torch.tensor([-1.0, 0.0, 0.25, 1.0, 1.5, 2.0, 3.0])
    with torch.inference_mode():
        f = rt.factor(lnp)
        d = rt.delta
        expect = torch.stack([d[0], d[0], 0.75 * d[0] + 0.25 * d[1], d[1], 0.5 * d[1] + 0.5 * d[2], d[2], d[2]])
        assert torch.allclose(f, expect, atol=1e-6)                    # constant beyond the end knots
        lp = rt(lnp)
        assert normalised(lp)
        ref = torch.log_softmax((base(lnp) + expect).reshape(len(lnp), -1), 1).reshape(lp.shape)
        assert torch.allclose(lp, ref, atol=1e-5)
    with pytest.raises(ValueError):
        ResidualTable(base, [1.0, 0.5])


def test_residual_table_trains_only_the_table():
    base = small_field(9)
    rt = ResidualTable(base, torch.tensor([0.0, 2.0]))
    loss = -rt(LNP)[:, 10, 3].mean()
    loss.backward()
    assert rt.delta.grad is not None and float(rt.delta.grad.abs().sum()) > 0
    assert all(p.grad is None for p in base.parameters())


# ----------------------------------------------------------------------------------------------------------------------
# Every model: eval() + inference_mode, normalised
# ----------------------------------------------------------------------------------------------------------------------
def test_all_models_eval_inference_mode():
    base = small_field(10)
    ms = {"field": base, "field_acc": FieldAcc(small_field(11), A=torch.full((C.N_Z, C.N_R), 0.5)),
          "ensemble": Ensemble([small_field(12), small_field(13)]),
          "table": ResidualTable(small_field(14), torch.tensor([0.0, 1.0]))}
    for name, m in ms.items():
        m.eval()
        with torch.inference_mode():
            lp = m(LNP)
        assert lp.shape == (len(LNP), C.N_Z, C.N_R) and normalised(lp), name
        assert isinstance(m.n_params(), int) and m.n_params() > 0


# ----------------------------------------------------------------------------------------------------------------------
# Factory registry
# ----------------------------------------------------------------------------------------------------------------------
def test_factory_registry_and_load_run(tmp_path):
    ctx = models.BuildContext(device=CPU, results_dir=tmp_path)
    assert {"field", "field_mlp_std", "field_acc", "residual_table", "ensemble", "from_run"} <= set(models.available())
    torch.manual_seed(0)
    m = models.build("field", ctx, hidden=4, n_layers=1, n_freq=1)
    assert isinstance(m, FieldNet) and m.hidden == 4
    with pytest.raises(KeyError):
        models.build("nope", ctx)
    with pytest.raises(ValueError):
        models.register("field")(lambda ctx: None)
    # a finished run is rebuilt from its stored spec and weights
    run = tmp_path / "stage" / "r1"
    run.mkdir(parents=True)
    with torch.no_grad():
        m.mlp[-1].bias.fill_(0.3)
    torch.save({"state": m.state_dict(), "meta": {"model": {"factory": "field",
                                                            "kwargs": {"hidden": 4, "n_layers": 1, "n_freq": 1}}}},
               run / "model.pt")
    back = models.load_run("stage/r1", ctx)
    with torch.inference_mode():
        assert torch.equal(back(LNP), m.eval()(LNP))
    ens = models.build("ensemble", ctx, members=["stage/r1", "stage/r1"])
    assert isinstance(ens, Ensemble) and ens.n_params() == 2 * m.n_params()
    rt = models.build("residual_table", ctx, base="stage/r1", n_knots=3, knots="uniform")
    assert isinstance(rt, ResidualTable) and rt.knots.numel() == 3
    with pytest.raises(ValueError):
        models.build("residual_table", ctx, base="stage/r1", knots="quantile")      # needs the training set
    fa = models.build("field_acc", ctx, template="stored", base="fresh", hidden=4, n_layers=1, n_freq=1)
    assert isinstance(fa, FieldAcc)


# ----------------------------------------------------------------------------------------------------------------------
# Cross-check with the families package (written in parallel; skipped if it is absent or does not import)
# ----------------------------------------------------------------------------------------------------------------------
def test_paramnet_satisfies_the_families_contract():
    try:
        from sfm.families import params as FP
    except Exception as err:                                           # not there yet, or mid-edit
        pytest.skip(f"sfm.families.params not importable ({type(err).__name__})")
    net = ParamNet(["a", "b"], ["k"], v0=FP.V0_PION, vs=FP.VS_PION)
    assert FP.is_param_provider(net)
    raw = FP.raw_dict(net, LNP)
    assert set(raw) == {"a", "b", "k"} and all(v.shape == (len(LNP),) for v in raw.values())
    # same kinetic variable as the families' providers
    assert torch.allclose(net.x(LNP), FP.x_of_lnp(LNP, FP.V0_PION, FP.VS_PION), atol=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# fit_to version 3: scale floor max(std_j, 1e-2) (user decision of 25 Nov 2026, PLAN.md section 17.11)
# ---------------------------------------------------------------------------------------------------------------------
def test_fit_to_scale_floor_changes_nothing_when_every_varying_column_is_above_it():
    target = Poly(["mu", "log_s", "frac", "c2"], V0, VS)
    with torch.no_grad():
        spread = target(GRID).double().std(0, unbiased=False)
    assert float(spread.min()) > 1e-2
    out = []
    for floor in (None, 1e-2):
        torch.manual_seed(5)
        net = ParamNet(["mu", "log_s", "frac"], ["c2"], v0=V0, vs=VS)
        rep = net.fit_to(target, GRID, steps=300, lbfgs_iter=200, scale_floor=floor)
        out.append((net.state_dict(), rep))
    (sa, ra), (sb, rb) = out
    assert all(torch.equal(sa[k], sb[k]) for k in sa) and ra["rms"] == rb["rms"] and ra["loss"] == rb["loss"]
    assert ra["fit_to_version"] == 2 and rb["fit_to_version"] == 3 and rb["below_scale_floor"] == []
    assert rb["scale_floor_abs"] == 1e-2 and set(rb["column_std"]) == set(net.names)


def test_fit_to_scale_floor_cures_a_nearly_constant_column():
    """The review's degraded case (section 3): six ordinary cubic columns plus one column of relative spread 1e-4.
    Measured on the laptop CPU with the pre-registered call (48 points, 2000 Adam, 500 L-BFGS): worst ordinary
    column 56.5 % of its spread without the floor, 1.3 % with it (0.19 % without the extra column)."""
    from sfm.families.components import default_lnp
    g = torch.Generator().manual_seed(0)
    base = torch.randn(6, 4, generator=g) * torch.tensor([1.0, 0.5, 0.3, 0.2])
    grid = default_lnp(48)
    xg = (torch.log(kinetic(grid)) - V0) / VS
    shape = torch.tensor([[0.0, 1.0, 0.5, 0.3]])
    sd = float((torch.stack([xg ** k for k in range(4)], 1) @ shape.T).std(unbiased=False))
    coef = torch.cat([base, torch.tensor([[2.0, 0, 0, 0]]) + shape * (1e-4 * 2.0 / sd)])

    class Prov(ParamProvider):
        names = [f"c{i}" for i in range(6)] + ["x"]

        def forward(self, lnp):
            x = (torch.log(kinetic(lnp)) - V0) / VS
            return torch.stack([x ** k for k in range(4)], 1) @ coef.T

    prov = Prov()
    worst = {}
    for floor in (None, 1e-2):
        torch.manual_seed(0)
        net = ParamNet(prov.names, [], v0=V0, vs=VS)
        rep = net.fit_to(prov, grid, steps=2000, lbfgs_iter=500, scale_floor=floor)
        with torch.no_grad():
            t = prov(grid)
            e = net(grid) - t
        worst[floor] = max(float(e[:, j].pow(2).mean().sqrt() / t[:, j].std(unbiased=False)) for j in range(6))
        if floor:
            assert rep["below_scale_floor"] == ["x"] and rep["column_weight"]["x"] == pytest.approx(1e4)
            assert rep["column_std"]["x"] == pytest.approx(2e-4, rel=1e-3) and rep["constant_columns"] == []
    assert worst[None] > 0.10 and worst[1e-2] < 0.02, worst
