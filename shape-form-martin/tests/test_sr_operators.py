"""sfm.sr.operators: vocabularies, constraint dictionaries, tree scans, losses. Needs neither Julia nor a GPU."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sfm.sr import operators as O

SRC = Path(__file__).resolve().parents[1] / "src"


def test_import_loads_neither_torch_nor_pysr():
    code = ("import sys; sys.path.insert(0, %r); import sfm.sr.operators as O; O.search_config('stage2', 3, 'gauge'); "
            "O.search_config('stage1', 1, 'wl2'); "
            "bad = [m for m in ('torch', 'pysr', 'juliacall') if m in sys.modules]; assert not bad, bad" % str(SRC))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sr_package_init_is_empty():
    assert (SRC / "sfm" / "sr" / "__init__.py").read_text().strip() == ""


# ----------------------------------------------------------------------------------------------------------------------
# vocabularies and the form PySR expects
# ----------------------------------------------------------------------------------------------------------------------
def test_stage1_vocabulary_is_the_plan():
    assert set(O.STAGE1.binary) == {"+", "-", "*", "ppow"}
    assert set(O.STAGE1.unary) == {"exp", "square", "tanh", "atan", "erf", "erfc", "softplus", "l1p", "sqa", "linv"}
    assert all(O.OPS[n].total for n in O.STAGE1.names) and O.STAGE1.pole_free
    assert set(O.STAGE2.unary) - set(O.STAGE1.unary) == {"sin", "cos"} and O.STAGE2.binary == O.STAGE1.binary
    assert set(O.STAGE1_RAW.names) - set(O.STAGE1.names) == {"/", "inv"} and not O.STAGE1_RAW.pole_free
    assert "EXTRA ARM" in O.STAGE1_RAW.label
    assert not O.OPS["/"].total and not O.OPS["inv"].total


def test_julia_strings():
    un = O.STAGE1.unary_operators()
    assert "softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))" in un
    assert "l1p(x) = log1p(abs(x))" in un and "sqa(x) = sqrt(abs(x))" in un and "linv(x) = 1 / (1 + x * x)" in un
    assert "ppow(x, y) = (1 + abs(x))^y" in O.STAGE1.binary_operators()
    for n, o in O.OPS.items():
        if not o.builtin:                                  # PySR takes the function name from the definition
            assert o.julia.startswith(n + "(") and "=" in o.julia
        else:
            assert "(" not in o.julia


def test_pysr_dict_forms():
    for v in O.VOCABULARIES.values():
        cfg = O.search_config(v.name, 3, "gauge")
        for op, lim in cfg["constraints"].items():
            assert op in v.names
            if O.OPS[op].arity == 1:
                assert isinstance(lim, int)
            else:
                assert isinstance(lim, tuple) and len(lim) == 2 and all(isinstance(a, int) for a in lim)
        for outer, inner in cfg["nested_constraints"].items():
            assert outer in v.names and all(i in v.names and isinstance(n, int) for i, n in inner.items())
        assert all(k in v.names and isinstance(c, int) for k, c in cfg["complexity_of_operators"].items())
        assert set(cfg["extra_sympy_mappings"]) == {n for n in v.names if not O.OPS[n].builtin}
    assert O.STAGE1.constraints["ppow"][1] == 1
    cw = O.STAGE2.complexity_of_operators()
    assert cw["erf"] == cw["erfc"] == 3
    assert all(cw[n] == 2 for n in ("exp", "tanh", "atan", "softplus", "l1p", "ppow", "sin", "cos"))
    assert "square" not in cw and O.OPS["square"].complexity == 1


def test_nested_constraints_are_the_plan():
    nc = O.STAGE1.nested_constraints
    for o in O.SATURATING:
        assert all(nc[o][i] == 0 for i in O.SATURATING) and nc[o]["exp"] == 0 and nc[o]["ppow"] == 0
    assert nc["exp"] == {"exp": 0}
    for o in ("ppow", "l1p", "sqa", "square"):
        assert nc[o] == {o: 0}
    n2 = O.STAGE2.nested_constraints
    for t in ("sin", "cos"):
        assert n2[t]["sin"] == 0 and n2[t]["cos"] == 0
    for o in O.STAGE1.unary + ("ppow",):
        assert n2[o]["sin"] == 0 and n2[o]["cos"] == 0
    assert O.STAGE2.constraints["sin"] == O.STAGE2.constraints["cos"] == 9


def test_caps_and_tiers():
    assert (O.MAXSIZE_1D, O.MAXSIZE_ND, O.PARETO_TIERS) == (18, 30, (12, 20, 30))
    assert O.search_config("stage1", 1, "wl2")["maxsize"] == 18 and O.search_config("stage1", 3)["maxsize"] == 30
    assert [O.pareto_tier(c) for c in (1, 12, 13, 20, 21, 30, 31)] == [12, 12, 20, 20, 30, 30, None]


def test_variable_names_are_legal_for_pysr():
    sympy = pytest.importorskip("sympy")
    assert not any(n in sympy.__dict__ for n in O.VARIABLE_NAMES)
    assert "zeta" in sympy.__dict__                        # the reason the variable is called zt


def test_box_and_kinematics():
    assert O.x_of_p(0.0005) == pytest.approx(-4.10301, abs=1e-4)
    assert O.x_of_p(50.0) == pytest.approx(1.27770, abs=1e-4)
    assert abs(O.x_of_p(O.TRAIN_P_MIN) + 1) < 2e-4 and abs(O.x_of_p(O.TRAIN_P_MAX) - 1) < 2e-4
    b = O.extended_box()
    assert b["zt"] == pytest.approx((-300 / 260, 2000 / 260)) and b["rh"] == pytest.approx((0, 1500 / 21.9))
    assert b["x"][0] <= O.x_of_p(0.0005) and b["x"][1] >= O.x_of_p(50.0)


# ----------------------------------------------------------------------------------------------------------------------
# implementations
# ----------------------------------------------------------------------------------------------------------------------
_EXTREME = np.array([0.0, 1e-300, 1e-30, 1e-8, 0.3, 1.0, 5.0, 19.0, 30.0, 88.0, 89.0, 700.0, 709.0, 710.0, 1e5,
                     1e19, 1e20, 1e154, 1e155, 1e300])
_EXTREME = np.concatenate([-_EXTREME[::-1], _EXTREME])


@pytest.mark.parametrize("name", [n for n, o in O.OPS.items() if o.arity == 1 and o.total])
def test_numpy_and_torch_agree_unary(name):
    torch = pytest.importorskip("torch")
    with np.errstate(all="ignore"):
        a = O.numpy_impl(name)(_EXTREME)
    b = O.torch_impl(name)(torch.tensor(_EXTREME, dtype=torch.float64)).numpy()
    assert not np.isnan(a).any() and not np.isnan(b).any()             # total operators: Inf at worst, never NaN
    assert np.array_equal(np.isinf(a), np.isinf(b))
    fin = np.isfinite(a)
    np.testing.assert_allclose(b[fin], a[fin], rtol=1e-13, atol=1e-300)
    lo, hi = O.OPS[name].range
    assert (a[fin] >= lo).all() and (a[fin] <= hi).all()


def test_numpy_and_torch_agree_ppow():
    torch = pytest.importorskip("torch")
    x, y = np.meshgrid(_EXTREME, np.array([-400.0, -3.0, -0.5, 0.0, 0.5, 2.0, 7.3, 400.0]))
    with np.errstate(all="ignore"):
        a = O.numpy_impl("ppow")(x, y)
    b = O.torch_impl("ppow")(torch.tensor(x), torch.tensor(y)).numpy()
    assert not np.isnan(a).any() and (a >= 0).all() and np.array_equal(np.isinf(a), np.isinf(b))
    fin = np.isfinite(a)
    np.testing.assert_allclose(b[fin], a[fin], rtol=1e-13, atol=1e-300)
    assert np.isinf(O.numpy_impl("ppow")(np.array(7.7), np.array(400.0)))          # (1 + 7.7)^400 overflows


def test_where_exp_and_square_overflow():
    with np.errstate(all="ignore"):
        assert np.isfinite(np.exp(np.float64(709.78))) and np.isinf(np.exp(np.float64(709.79)))
        assert np.isfinite(np.exp(np.float32(88.72))) and np.isinf(np.exp(np.float32(88.73)))
        assert np.isfinite(np.square(np.float64(1.34e154))) and np.isinf(np.square(np.float64(1.35e154)))
        assert np.isfinite(np.square(np.float32(1.84e19))) and np.isinf(np.square(np.float32(1.85e19)))


def test_softplus_is_the_julia_formula_not_the_shortcut():
    x = np.array([-800.0, -30.0, -1.0, 0.0, 1.0, 30.0, 800.0])
    sp = O.numpy_impl("softplus")(x)
    assert sp[0] == pytest.approx(np.exp(-800.0), abs=1e-300) and sp[-1] == 800.0 and np.isfinite(sp).all()
    np.testing.assert_allclose(sp[1:6], np.log1p(np.exp(x[1:6])), rtol=1e-14)


def test_sympy_mappings_match_numpy():
    sympy = pytest.importorskip("sympy")
    m = O.sympy_mappings()
    assert set(m) == {n for n, o in O.OPS.items() if not o.builtin}
    xs, ys = sympy.symbols("xs ys")
    for xv in (-2.3, 0.4, 3.1):
        for n in ("softplus", "l1p", "sqa", "linv"):
            assert float(m[n](xs).subs(xs, xv)) == pytest.approx(float(O.numpy_impl(n)(xv)), rel=1e-12)
        assert float(m["ppow"](xs, ys).subs({xs: xv, ys: -1.7})) == pytest.approx(
            float(O.numpy_impl("ppow")(xv, -1.7)), rel=1e-12)


# ----------------------------------------------------------------------------------------------------------------------
# tree scans (nested tuples stand in for sfm.sr.translate.Expr)
# ----------------------------------------------------------------------------------------------------------------------
def test_tree_complexity_of_a_cubic_is_13():
    horner = ("+", 1.0, ("*", "x", ("+", 1.0, ("*", "x", ("+", 1.0, ("*", "x", 1.0))))))
    assert O.tree_complexity(horner) == 13
    assert O.tree_complexity(("erf", ("exp", "x"))) == 3 + 2 + 1
    assert O.tree_complexity(("tanh", "x"), {"tanh": 2.4}) == 3                     # rounded like SymbolicRegression.jl


@pytest.mark.parametrize("tree, bad", [
    (("tanh", ("exp", "zt")), True),                       # saturating wrapping exp
    (("exp", ("tanh", "zt")), False),                      # the other way round is allowed
    (("tanh", ("+", "zt", ("atan", "x"))), True),          # saturating inside saturating, through a +
    (("erf", ("ppow", "zt", 2.0)), True),
    (("exp", ("*", 2.0, ("exp", "x"))), True),             # exp in exp
    (("square", ("+", ("square", "x"), 1.0)), True),       # self-nested
    (("l1p", ("sqa", ("l1p", "x"))), True),
    (("ppow", ("ppow", "x", 2.0), 3.0), True),
    (("*", ("tanh", "zt"), ("tanh", "x")), False),         # siblings are not nested
    (("exp", ("*", -1.0, ("square", ("-", "zt", 0.5)))), False),
    (("+", ("tanh", ("+", "zt", "x")), ("linv", "rh")), False),
])
def test_violates_nesting_stage1(tree, bad):
    assert bool(O.violates_nesting(tree)) is bad


def test_nestedness_counts_like_symbolicregression():
    nc = {"cos": {"cos": 2}}                               # PySR docs: cos(cos(cos(x))) allowed, four deep is not
    three = ("cos", ("cos", ("cos", "x")))
    assert O.violates_nesting(three, nc) == []
    assert O.violates_nesting(("cos", ("+", three, 1.0)), nc) != []


def test_stage2_trig_rules():
    v = O.STAGE2
    assert v.violations(("+", ("tanh", "zt"), ("*", ("tanh", "x"), ("sin", ("+", ("*", 3.0, "zt"), "x"))))) == []
    assert v.violations(("sin", ("cos", "zt")))
    assert v.violations(("tanh", ("sin", "zt")))           # trig inside another operator
    assert v.violations(("exp", ("*", 0.1, ("cos", "zt"))))
    assert v.violations(("sin", ("square", "zt")))         # chirp
    assert O.STAGE1.violations(("sin", "zt"))              # not in the Stage-1 vocabulary at all


def test_violates_constraints():
    affine3 = ("+", ("+", ("+", ("*", 1.0, "zt"), ("*", 1.0, "rh")), ("*", 1.0, "x")), 1.0)
    assert O.tree_complexity(affine3) == 13
    assert O.violates_constraints(("tanh", affine3)) == []
    assert O.violates_constraints(("tanh", ("+", affine3, "x")))
    assert O.violates_constraints(("exp", affine3))        # exp limit is 9
    assert O.violates_constraints(("ppow", "zt", "x")) == ["ppow: exponent is not a constant"]
    assert O.violates_constraints(("ppow", "zt", -1.5)) == []
    assert O.violates_constraints(("ppow", "zt", ("+", 1.0, 2.0)))
    assert O.STAGE1_RAW.violations(("/", "zt", ("+", "rh", 5.0))) == []
    assert O.STAGE1.violations(("/", "zt", ("+", "rh", 5.0)))


# ----------------------------------------------------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------------------------------------------------
def _slices(rng, n_slices=5):
    sizes = rng.integers(3, 40, n_slices)
    key = np.repeat(np.sort(rng.normal(size=n_slices)), sizes)
    n = len(key)
    return key, rng.normal(size=n), rng.normal(size=n), rng.uniform(0.1, 3.0, n)


def test_gauge_loss_is_invariant_and_equals_the_weighted_variance():
    rng = np.random.default_rng(0)
    key, pred, y, w = _slices(rng)
    L = O.gauge_loss_numpy(pred, y, w, key)
    assert O.gauge_loss_numpy(pred + 3.0 * key ** 2 - 100.0 * key + 7.0, y, w, key) == pytest.approx(L, rel=1e-10)
    assert O.gauge_loss_numpy(pred, y + np.sin(key), w, key) == pytest.approx(L, rel=1e-12)
    ref = 0.0
    for k in np.unique(key):
        m = key == k
        e = pred[m] - y[m]
        ref += np.sum(w[m] * (e - np.average(e, weights=w[m])) ** 2)
    assert L == pytest.approx(ref / w.sum(), rel=1e-12)
    assert L <= O.wl2_loss_numpy(pred, y, w) + 1e-12       # the variance never exceeds the plain weighted L2
    assert O.gauge_loss_numpy(y + key, y, w, key) == pytest.approx(0.0, abs=1e-25)
    assert O.gauge_loss_numpy(np.where(np.arange(len(y)) == 2, np.inf, pred), y, w, key) == math.inf


def test_wl2_reference():
    assert O.wl2_loss_numpy([1.0, 3.0], [0.0, 0.0], [1.0, 3.0]) == pytest.approx((1 + 27) / 4)


def test_gauge_loss_julia_source():
    src = O.gauge_loss_julia(3)
    assert src.startswith("function sfm_gauge_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}")
    assert "eval_tree_array(tree, dataset.X, options)" in src and "X[3, i]" in src and src.rstrip().endswith("end")
    assert src.count("function") == 1 and "{xi}" not in src
    assert "X[1, i]" in O.gauge_loss_julia(1)
    with pytest.raises(ValueError):
        O.gauge_loss_julia(0)
    cfg = O.search_config("stage1", 3, "gauge")
    assert cfg["loss_function"] == src and cfg["batching"] is False and "elementwise_loss" not in cfg
    cw = O.search_config("stage1", 1, "wl2")
    assert cw["elementwise_loss"] == O.WL2_ELEMENTWISE and "loss_function" not in cw
    assert cfg["precision"] == 64 and cfg["print_precision"] == 17


# ----------------------------------------------------------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------------------------------------------------------
def test_templates():
    t = O.TEMPLATES["s1_tanh_sum"]
    assert t.combine() == "tanh(u(zt, x) + v(rh, x) + w(zt, rh))" and t.bound == 1.0
    assert O.TEMPLATES["s1_amp_shape"].combine() == "a(x) * f(zt, rh)"
    for t in O.TEMPLATES.values():
        assert set(t.arguments) == set(t.expressions)
        for name, args in t.arguments.items():
            assert f"{name}({', '.join(args)})" in t.combine()
            assert set(args) <= set(O.VARIABLE_NAMES)
        assert t.pysr_kwargs()["variable_names"] == list(O.VARIABLE_NAMES)
    s2 = O.TEMPLATES["s2_modulation"]
    assert "0.3 * tanh(a(x))" in s2.combine(64) and "0.3f0 * tanh(a(x))" in s2.combine(32)
    assert "e(zt, rh)^2" in s2.combine(32) and "1 / (1 + " in s2.combine(32)      # integer literals stay integers
    assert s2.modulation in s2.combine() and s2.bound == 1.3 and O.TEMPLATES["s2_modulation_unit"].bound == 1.0
    cfg = O.search_config("stage1", 3, "gauge", template="s2_modulation")
    assert "loss_function_expression" in cfg and "loss_function" not in cfg and "extra_sympy_mappings" not in cfg
    assert "sin" not in O.VOCABULARIES[s2.vocabulary].names            # the template's sin cannot be nested
    with pytest.raises(ValueError):
        O.search_config("stage2", 3, "gauge", template="s2_modulation")


def test_julia_confirmation_list():
    """The first session's open list is closed: every item was confirmed with tiny fits (STATUS.md section 9)."""
    assert O.UNCONFIRMED == () and len(O.CONFIRMED_WITH_JULIA) == 12
    assert all(isinstance(s, str) and s for s in O.CONFIRMED_WITH_JULIA)


# ----------------------------------------------------------------------------------------------------------------------
# gradient hygiene at exactly 0 (user confirmation 1: no Inf / NaN gradient in a PyTorch refit)
# ----------------------------------------------------------------------------------------------------------------------
TOTAL_OPS = sorted(n for n, o in O.OPS.items() if o.total)


@pytest.mark.parametrize("name", TOTAL_OPS)
def test_gradients_are_finite_at_exactly_zero(name):
    """Backpropagate through every total operator at x = +0.0 and -0.0 (r = 0 on the axis and x = 0 are reachable),
    in float32 and float64, for every argument of a binary operator."""
    import torch
    f = O.torch_impl(name)
    for dt in (torch.float64, torch.float32):
        for z in (0.0, -0.0):
            if O.OPS[name].arity == 1:
                x = torch.tensor([z, z], dtype=dt, requires_grad=True)
                y = f(x)
                grads = torch.autograd.grad(y.sum(), x)
            else:
                for other in (z, 1.5, -2.0):
                    for first in (True, False):
                        a = torch.tensor([z], dtype=dt, requires_grad=True)
                        b = torch.tensor([other], dtype=dt, requires_grad=True)
                        y = f(a, b) if first else f(b, a)
                        grads = torch.autograd.grad(y.sum(), (a, b))
                        assert all(bool(torch.isfinite(g).all()) for g in grads), (name, z, other, first, grads)
            assert bool(torch.isfinite(y).all()) and all(bool(torch.isfinite(g).all()) for g in grads), (name, dt, z)


def test_sqa_backward_is_guarded_and_forward_is_the_formula():
    import torch
    f = O.torch_impl("sqa")
    x = torch.tensor([0.0, -0.0, 1e-300, -4.0, 9.0, 5e-324], dtype=torch.float64, requires_grad=True)
    y = f(x)
    assert torch.equal(y.detach(), torch.sqrt(torch.abs(x.detach())))              # bit for bit
    (g,) = torch.autograd.grad(y.sum(), x)
    assert g[0] == 0.0 and g[1] == 0.0                                              # was NaN (inf * 0)
    assert g[3] == pytest.approx(-0.25) and g[4] == pytest.approx(1 / 6) and torch.isfinite(g).all()
    assert g[2] == pytest.approx(0.5e150) and g[5] > 1e160                         # huge next to the cusp, finite
    plain = torch.sqrt(torch.abs(x))                                                # what autograd does unguarded
    assert torch.isnan(torch.autograd.grad(plain.sum(), x)[0][0])
    bad = f(torch.tensor([float("nan"), float("inf"), -float("inf")]))             # nothing is hidden
    assert torch.isnan(bad[0]) and torch.isinf(bad[1]) and torch.isinf(bad[2])
    for dt in (torch.float32, torch.float64):                                       # gradient w.r.t. constants
        c = torch.tensor([2.0, 0.0], dtype=dt, requires_grad=True)
        r = torch.tensor([0.0, 0.0, 3.0], dtype=dt)
        out = f(r * c[0] + c[1]).sum()
        (gc,) = torch.autograd.grad(out, c)
        assert torch.isfinite(gc).all()


def test_softplus_gradient_is_the_sigmoid_everywhere():
    import torch
    f = O.torch_impl("softplus")
    x = torch.tensor([-800.0, -30.0, -1.0, -0.0, 0.0, 1.0, 30.0, 800.0], dtype=torch.float64, requires_grad=True)
    y = f(x)
    ref = np.maximum(x.detach().numpy(), 0.0) + np.log1p(np.exp(-np.abs(x.detach().numpy())))
    assert np.array_equal(y.detach().numpy(), ref)                                  # the Julia formula, unchanged
    (g,) = torch.autograd.grad(y.sum(), x)
    assert torch.allclose(g, torch.sigmoid(x.detach()), rtol=0, atol=0) and g[3] == 0.5 and g[4] == 0.5
    xx = torch.linspace(-3, 3, 7, dtype=torch.float64, requires_grad=True)          # second derivative works too
    (g1,) = torch.autograd.grad(f(xx).sum(), xx, create_graph=True)
    (g2,) = torch.autograd.grad(g1.sum(), xx)
    s = torch.sigmoid(xx.detach())
    assert torch.allclose(g2, s * (1 - s))
    with torch.inference_mode():
        assert float(f(torch.tensor(0.0))) == pytest.approx(math.log(2.0))


def test_ppow_exponent_limit_is_one_in_every_vocabulary():
    """User confirmation 3: no compound exponents in the search. (7, 1) = base size <= 7, exponent size 1; the
    exponent limit is the same as in (-1, 1), the base limit is the plan's argument-size cap. A size-1 exponent can
    still be a variable (confirmed with Julia), so the post-search scan stays."""
    for v in O.VOCABULARIES.values():
        assert v.constraints["ppow"] == (7, 1)
        cfg = O.search_config(v.name, 3, "gauge")
        assert cfg["constraints"]["ppow"] == (7, 1)
        assert "ppow: argument 2 has complexity 3 (limit 1)" in O.violations(("ppow", "zt", ("+", "rh", "x")), v)
        assert O.violations(("ppow", "zt", "rh"), v) == ["ppow: exponent is not a constant"]
        assert O.violations(("ppow", "zt", -1.5), v) == []
    for name, t in O.TEMPLATES.items():                    # templates search their parts with the same constraints
        assert O.search_config(t.vocabulary, 3, "gauge", template=name)["constraints"]["ppow"] == (7, 1)
    assert O.search_config("stage1", 1, "wl2")["constraints"]["ppow"] == (7, 1)
