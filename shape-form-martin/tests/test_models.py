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
