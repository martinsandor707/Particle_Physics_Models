"""sfm.sr.audit: soundness of the interval arithmetic, domain safety, the tight magnitude bound and its gauges.
Exact mathematics on synthetic expressions: no data, no Julia, no GPU."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sfm.sr import audit as A
from sfm.sr import operators as O
from sfm.sr import translate as T

SRC = Path(__file__).resolve().parents[1] / "src"
BOX = O.extended_box()
VN = ("zt", "rh", "x")
UNARY_S1 = list(O.STAGE1.unary)
UNARY_ALL = UNARY_S1 + ["sin", "cos", "neg", "abs", "cube", "inv", "log", "log1p", "sqrt"]
BINARY_ALL = ["+", "-", "*", "ppow", "/", "pow"]


def test_import_loads_no_torch():
    code = ("import sys; sys.path.insert(0, %r); import sfm.sr.audit as A; r = A.audit('tanh(zt) * 0.5'); "
            "assert r.passed; bad = [m for m in ('torch', 'pysr', 'juliacall') if m in sys.modules]; "
            "assert not bad, bad" % str(SRC))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_default_box_is_the_users_box():
    assert BOX["zt"] == (-300.0 / 260.0, 2000.0 / 260.0) and BOX["rh"] == (0.0, 1500.0 / 21.9)
    assert BOX["x"][0] <= float(O.x_of_p(0.0005)) and BOX["x"][1] >= float(O.x_of_p(50.0))
    b = A.audit_box(x_floor=-2.5)
    assert b["x"] == (-2.5, BOX["x"][1]) and b["zt"] == BOX["zt"]
    with pytest.raises(ValueError):
        A.audit_box(x_floor=5.0)


# ----------------------------------------------------------------------------------------------------------------------
# random expressions
# ----------------------------------------------------------------------------------------------------------------------
def random_expr(rng, depth, unary, binary, p_leaf=0.25):
    if depth == 0 or rng.random() < p_leaf:
        if rng.random() < 0.6:
            return T.var(VN[rng.integers(3)])
        return T.const(float(np.round(rng.normal(0, 1.5), 3)))
    if rng.random() < 0.5:
        return T.op(unary[rng.integers(len(unary))], random_expr(rng, depth - 1, unary, binary))
    name = binary[rng.integers(len(binary))]
    a = random_expr(rng, depth - 1, unary, binary)
    if name == "ppow":
        return T.op("ppow", a, T.const(float(np.round(rng.uniform(-3, 3), 2))))
    if name == "pow":
        if rng.random() < 0.6:
            return T.op("pow", a, T.const(float(rng.integers(-3, 5)), fixed=True))
        return T.op("pow", a, T.const(float(np.round(rng.uniform(-2, 2), 2))))
    return T.op(name, a, random_expr(rng, depth - 1, unary, binary))


def random_box(rng):
    box = {}
    for n in VN:
        lo, hi = BOX[n]
        w = (hi - lo) * 10.0 ** rng.uniform(-3, 0)
        a = rng.uniform(lo, hi - w)
        box[n] = (float(a), float(a + w))
    return box


def sample(rng, box, n):
    pts = {k: rng.uniform(v[0], v[1], n) for k, v in box.items()}
    for k, v in box.items():                               # include the corners of every side
        pts[k][: 2] = v
    return pts


def test_interval_soundness_against_brute_force():
    """Every sampled value lies inside the enclosure; a box certified as domain-safe has only finite samples; the
    derivative enclosures contain the autograd derivatives. Random trees over the whole vocabulary (Stage 1,
    trig, the raw arm and the audit-only operators), random sub-boxes over three decades of size."""
    rng = np.random.default_rng(20261125)
    n_expr = n_cert = n_grad = 0
    for trial in range(700):
        e = random_expr(rng, int(rng.integers(1, 5)), UNARY_ALL, BINARY_ALL)
        if not T.variables(e):
            continue
        box = random_box(rng)
        enc = A.enclose(e, box, grad=True)
        pts = sample(rng, box, 600)
        rec = {}
        v = np.asarray(T.evaluate(e, pts, record=rec)) + np.zeros(600)
        fin = np.isfinite(v)
        n_expr += 1
        assert enc["lo"] <= enc["hi"]
        if fin.any():
            assert enc["lo"] <= v[fin].min() and v[fin].max() <= enc["hi"], (T.to_string(e), box, enc)
        if enc["certified"]:
            n_cert += 1
            assert fin.all() and not rec["nonfinite"], (T.to_string(e), box)
            assert rec["max_abs"] <= A.OVERFLOW_CAP
            m = T.TorchExpr(e, VN)
            cols = [torch.tensor(pts[n], requires_grad=True) for n in VN]
            out = m(*cols)
            grads = torch.autograd.grad(out.sum(), [c for c, n in zip(cols, VN) if n in enc["variables"]],
                                        allow_unused=True)
            for g, n in zip(grads, [n for n in VN if n in enc["variables"]]):
                if g is None:
                    continue
                g = g.numpy()
                ok = np.isfinite(g)
                tol = 1e-9 * (1.0 + np.abs(g[ok]))
                assert np.all(g[ok] >= enc["grad_lo"][n] - tol) and np.all(g[ok] <= enc["grad_hi"][n] + tol), \
                    (T.to_string(e), n, box, enc["grad_lo"][n], enc["grad_hi"][n], g[ok].min(), g[ok].max())
                n_grad += 1
    assert n_expr > 300 and n_cert > 100 and n_grad > 100


@pytest.mark.parametrize("name", sorted(A._UNARY_VAL))
def test_elementary_enclosures_contain_and_are_tight(name):
    rng = np.random.default_rng(sorted(A._UNARY_VAL).index(name))
    e = T.op(name, T.var("u"))
    for _ in range(200):
        c, w = rng.uniform(-8, 8), 10.0 ** rng.uniform(-6, 1)
        box = {"u": (c - w, c + w)}
        enc = A.enclose(e, box, grad=True)
        u = np.concatenate([np.linspace(c - w, c + w, 2001), [c - w, c + w]])
        if c - w < 0 < c + w:
            u[1000] = 0.0                                   # the kink of l1p / sqa / abs, the minimum of square / linv
        v = np.asarray(T.evaluate(e, u=u))
        assert enc["certified"] and enc["lo"] <= v.min() and v.max() <= enc["hi"]
        span = v.max() - v.min()
        assert enc["hi"] - enc["lo"] <= span * (1 + 1e-3) + 1e-12 * (1 + abs(v).max()) + 2e-6 * (name in ("sin", "cos"))
        if name == "sqa" and c - w <= 0 <= c + w:
            assert enc["grad_hi"]["u"] == math.inf and enc["grad_lo"]["u"] == -math.inf
            continue
        g = np.gradient(v[:-2], np.linspace(c - w, c + w, 2001))[1:-1] if not (c - w < 0 < c + w) else \
            np.diff(v[:-2]) / np.diff(u[:-2])              # secant slopes: inside the derivative range by the MVT
        tol = 1e-3 * (1 + np.abs(g).max())
        assert enc["grad_lo"]["u"] <= g.min() + tol and g.max() - tol <= enc["grad_hi"]["u"]


def test_ppow_and_pow_enclosures():
    rng = np.random.default_rng(5)
    for s in ["ppow(u, v)", "(u + 9.0) ^ v", "u ^ 3", "u ^ 4", "u ^ -2", "(u + 9.0) ^ -3"]:
        e = T.parse(s)
        for _ in range(100):
            cu, wu, cv, wv = rng.uniform(-6, 6), 10.0 ** rng.uniform(-3, 0.3), rng.uniform(-3, 3), 10.0 ** rng.uniform(-3, 0)
            box = {"u": (cu - wu, cu + wu), "v": (cv - wv, cv + wv)}
            enc = A.enclose(e, box)
            uu, vv = np.meshgrid(np.linspace(*box["u"], 81), np.linspace(*box["v"], 81))
            with np.errstate(all="ignore"):
                val = np.asarray(T.evaluate(e, u=uu, v=vv)) + np.zeros_like(uu)
            fin = np.isfinite(val)
            if s == "u ^ -2" and box["u"][0] <= 0 <= box["u"][1]:
                assert not enc["certified"]
                continue
            assert enc["certified"] and fin.all()
            assert enc["lo"] <= val.min() and val.max() <= enc["hi"]
            assert enc["hi"] - enc["lo"] <= (val.max() - val.min()) * 1.05 + 1e-9 * (1 + np.abs(val).max())


def test_outward_rounding_is_strict():
    enc = A.enclose("u + v", {"u": (0.1, 0.1), "v": (0.2, 0.2)})
    assert enc["lo"] < 0.1 + 0.2 < enc["hi"] and enc["hi"] - enc["lo"] < 1e-15
    enc = A.enclose("tanh(u)", {"u": (30.0, 40.0)})
    assert enc["hi"] == 1.0 and enc["lo"] < 1.0           # intersected with the mathematical range
    enc = A.enclose("exp(u)", {"u": (700.0, 800.0)})
    assert enc["hi"] == math.inf and not enc["certified"]  # overflow is an outward bound and a flag


# ----------------------------------------------------------------------------------------------------------------------
# (a) domain safety
# ----------------------------------------------------------------------------------------------------------------------
EARLIER_STUDY = [                                           # shape-form-proton/results/particles/pion/sr_residual.json
    "-1.943752e-5/(r - 1*0.017030673)",
    "(x - 1*1.2091627)**2*(-1.7122895e-5)/(r - 1*0.017030673)",
    "-1.6204527e-5*(x - 1*1.2589178)**2*exp(-2*z)/(r - 1*0.017030673)",
    "-1.6204527e-5*(1.2589178 - (-r + x + z))**2*exp(-2*z)/(r - 1*0.017030673)",
]
BOX_METRES = {"r": (0.0, 1.5), "z": (-0.3, 2.0), "x": BOX["x"]}


@pytest.mark.parametrize("eq", EARLIER_STUDY)
def test_the_earlier_studys_equations_are_rejected(eq):
    """c / (r - 0.017): a pole 0.1 mm from a ring centre that no data row sees."""
    rep = A.audit(eq, BOX_METRES)
    assert not rep.passed and rep.domain["status"] == "unsafe_proven" and rep.magnitude["status"] == "skipped"
    w = rep.domain["witness"]
    assert "r - (1.0 * 0.017030673)" in w["denominator"] and w["box"]["r"][0] < 0.017030673 < w["box"]["r"][1]
    json.dumps(rep.to_json())
    assert "REJECT" in rep.summary()


def test_pole_in_zt_rh_x_variables_invisible_to_a_grid():
    s = "1.0e-9 / (rh - 0.7763)"                           # |value| < 1e-6 unless rh is within 1e-3 of the pole
    rh = np.linspace(0.0, 68.49, 100001)
    assert np.max(np.abs(T.evaluate(T.parse(s), rh=rh))) < 1e-3     # a dense grid sees nothing
    rep = A.audit(s)
    assert not rep.passed and rep.domain["status"] == "unsafe_proven"


@pytest.mark.parametrize("eq,status", [
    ("1 / rh", "unsafe_proven"),                           # pole on the box edge r = 0
    ("x / x", "unsafe_proven"),                            # removable singularity: 0/0 is NaN in the refit
    ("inv(zt - 3.3)", "unsafe_proven"),
    ("ppow(zt, 400.0)", "unsafe_proven"),                  # overflow, no pole
    ("exp(square(zt) * rh)", "unsafe_proven"),
    ("log(rh)", "unsafe_proven"), ("sqrt(rh)", "unsafe_proven"), ("log1p(zt - 0.5)", "unsafe_proven"),
    ("rh ^ 1.5", "unsafe_proven"), ("rh ^ -2", "unsafe_proven"),
    ("1 / (square(rh) + 1.0e-9)", "unsafe_proven"),        # inside the float32 allowance tau
    ("1 / (square(x - 0.0123) + 0.0)", "unsafe_proven"),   # a zero touched without a sign change
    ("zt / (rh * x - 1.0)", "unsafe_proven"),
    ("zt * NaN", "unsafe_proven"),
    ("1 / (rh + 0.5)", "safe"), ("log(rh + 1.0)", "safe"), ("sqrt(rh + 0.01)", "safe"), ("log1p(rh)", "safe"),
    ("(rh + 0.5) ^ 1.5", "safe"), ("(rh + 1.0) ^ -2", "safe"), ("zt ^ 3", "safe"),
    ("1 / (square(zt) + square(rh) + 0.01)", "safe"),
    ("1 / (1.0 + square(zt * rh - x))", "safe"),
    ("zt / (exp(x) + 0.001)", "safe"),
    ("tanh((zt * 1.3) + -0.4) * 0.5", "safe"),
    ("exp(zt * 3.0) * linv(rh)", "safe"),
    ("exp(-1.0 * square(zt * rh))", "safe"),               # underflow is harmless
])
def test_domain_cases(eq, status):
    r = A.domain_safety(eq)
    assert r["status"] == status and r["safe"] == (status == "safe"), r
    if status != "safe":
        assert r["witness"] is not None


def test_no_pole_is_missed_and_distant_poles_are_accepted():
    rng = np.random.default_rng(3)
    for _ in range(150):
        name = VN[rng.integers(3)]
        lo, hi = BOX[name]
        c = rng.uniform(lo, hi)
        inner = [f"{name} - {c!r}", f"({name} * 2.0) - {2 * c!r}", f"tanh({name} - {c!r})",
                 f"square({name}) - {c * c!r}" if c > 0 else f"{name} - {c!r}"][rng.integers(4)]
        outer = ["0.001 / ({})", "inv({}) * 1.0e-6", "zt / ({})", "({}) ^ -1", "1.0e-12 / (({}) * (rh + 1.0))"][rng.integers(5)]
        assert not A.domain_safety(outer.format(inner))["safe"], (outer, inner)
    for _ in range(60):
        name = VN[rng.integers(3)]
        lo, hi = BOX[name]
        c = hi + rng.uniform(0.05, 3.0) if rng.random() < 0.5 else lo - rng.uniform(0.05, 3.0)
        assert A.domain_safety(f"0.001 / ({name} - {c!r})")["safe"]


def test_bisection_resolves_overestimation():
    # (zt - zt) + 1 is 1, but the natural enclosure of zt - zt on the full box contains 0: bisection certifies
    r = A.domain_safety("1 / ((zt - zt) + 0.5)")
    assert r["safe"] and r["n_boxes"] > 1
    # a false rejection that the budget cannot resolve is still a rejection, labelled as not proven
    r = A.domain_safety("1 / ((zt * rh - rh * zt) + 0.5)", max_boxes=200)
    assert not r["safe"] and r["status"] == "not_proven_safe" and r.get("budget_exhausted")
    assert A.domain_safety("1 / ((zt * rh - rh * zt) + 0.5)", {"zt": (0.0, 1.0), "rh": (0.0, 1.0)})["safe"]
    # the float32 allowance: a denominator of 1e-3 built from terms of size 500 is inside tau = 1e-5 * 527
    r = A.domain_safety("1 / ((zt * rh - rh * zt) + 1.0e-3)")
    assert r["status"] == "unsafe_proven" and r["witness"]["kind"] == "point"


def test_stage1_is_pole_free_and_false_rejection_rate():
    """Stage-1 trees have no restricted node, so only overflow can reject. Measured here: how often the audit
    rejects a random Stage-1 tree whose 4000 random evaluations stay below the overflow cap (a false rejection as
    far as sampling can tell)."""
    rng = np.random.default_rng(11)
    n = n_ok = n_false = n_reject = 0
    for _ in range(420):
        e = random_expr(rng, int(rng.integers(2, 5)), UNARY_S1, ["+", "-", "*", "ppow"])
        if not T.variables(e):
            continue
        assert O.violations(e, O.STAGE1) is not None
        r = A.domain_safety(e, max_boxes=20000)
        assert not r["restricted_nodes"]
        n += 1
        n_reject += not r["safe"]
        rec = {}
        pts = sample(rng, BOX, 4000)
        T.evaluate(e, pts, record=rec)
        truly_ok = (not rec["nonfinite"]) and rec["max_abs"] <= A.OVERFLOW_CAP
        if r["safe"]:
            assert truly_ok, T.to_string(e)                 # soundness again
        if truly_ok:
            n_ok += 1
            n_false += not r["safe"]
    print(f"\nstage-1 random trees: {n}, sampling-valid {n_ok}, rejected {n_reject}, false rejections {n_false}")
    assert n > 250 and n_ok > 100
    assert n_false / n_ok < 0.05, (n_false, n_ok, n_reject, n)


# ----------------------------------------------------------------------------------------------------------------------
# (b) magnitude bound
# ----------------------------------------------------------------------------------------------------------------------
def test_known_suprema():
    top = math.tanh(BOX["zt"][1])
    r = A.magnitude_bound("0.9 * tanh(zt)", gauge="none")
    assert r["status"] == "certified" and r["sup_lower"] == pytest.approx(0.9 * top, rel=1e-12)
    assert r["sup_lower"] <= r["sup_upper"] <= 0.9 * top * (1 + 1e-9)
    r = A.magnitude_bound("1.01 * tanh(zt)", gauge="none")
    assert r["status"] == "failed" and not r["passed"] and abs(r["witness"]["value"]) > 1.0
    # exactly at the limit: tanh saturates to 1.0 in floating point; not a failure
    r = A.magnitude_bound("tanh(zt * 30.0)", gauge="none")
    assert r["status"] == "certified" and r["sup_lower"] == 1.0
    # oscillation of an odd function of zt on an asymmetric box
    r = A.magnitude_bound("0.9 * tanh(zt)")
    want = 0.45 * (top - math.tanh(BOX["zt"][0]))
    assert r["status"] == "certified" and r["sup_lower"] == pytest.approx(want, rel=1e-9) and r["sup_upper"] < 1
    assert r["delta_eff_bound"] == pytest.approx(2 * r["sup_upper"])


def test_dependency_problem_true_sup_below_one():
    """sin^2 + cos^2 = 1: the naive enclosure is [0, 2], the truth is 0.9 * tanh(...) < 1."""
    s = "0.9 * (square(sin(zt * 3.0)) + square(cos(zt * 3.0))) * tanh(rh - 5.0)"
    naive = A.enclose(s)
    assert naive["hi"] > 1.7 and naive["lo"] < -1.7        # naive intervals would reject
    for gauge in ("none", "oscillation"):
        r = A.magnitude_bound(s, gauge=gauge)
        assert r["status"] == "certified" and r["passed"], r
        assert 0.8999 < r["sup_lower"] <= 0.9 + 1e-12 and r["sup_upper"] <= 1.0 + 1e-9
        assert r["n_boxes"] > 100                          # subdivision was needed, and it was enough
    r = A.magnitude_bound("(zt - zt) + (rh * x - x * rh) + 0.99", gauge="none")
    assert r["status"] == "certified" and r["sup_lower"] == pytest.approx(0.99)


def test_failure_needs_an_actual_evaluation_and_finds_a_narrow_peak():
    peak = "{} * exp(-1.0 * square((zt - 3.1234) * 40.0)) * exp(-1.0 * square((rh - 31.7) * 4.0))"
    # the coarse grid (33 points per side) does not see a bump of width 0.025 x 0.25
    zt, rh = np.meshgrid(np.linspace(*BOX["zt"], 33), np.linspace(*BOX["rh"], 33))
    assert np.max(np.abs(T.evaluate(T.parse(peak.format(1.05)), zt=zt, rh=rh))) < 0.5
    r = A.magnitude_bound(peak.format(1.05), gauge="none")
    assert r["status"] == "failed" and r["witness"]["value"] > 1.0
    v = T.evaluate(T.parse(peak.format(1.05)), r["witness"]["point"])
    assert v == pytest.approx(r["witness"]["value"]) and v > 1.0          # the witness is a real evaluation
    r = A.magnitude_bound(peak.format(0.95), gauge="none")
    assert r["passed"] and r["status"] in ("certified", "not_refuted")
    assert r["sup_lower"] <= 0.95 <= r["sup_upper"] * (1 + 1e-12) and r["sup_upper"] <= 1.0 + 1e-9
    # the half-oscillation of the same bump is half its height
    r = A.magnitude_bound(peak.format(1.9))
    assert r["passed"] and r["sup_lower"] == pytest.approx(0.95, abs=0.01)
    r = A.magnitude_bound(peak.format(2.1))
    assert r["status"] == "failed" and r["witness"]["half_oscillation"] > 1.0


def test_oscillation_gauge_is_gauge_invariant():
    f = "0.45 * tanh(zt - 2.0) + 0.5 * tanh(rh - 3.0) * tanh(x)"
    base = A.magnitude_bound(f)
    for c in (" + 3.0 * square(x)", " - exp(-1.0 * x)", " + 17.5", " + (sqa(x) - 2.0 * x)"):
        r = A.magnitude_bound(f + c)
        assert r["status"] == base["status"] == "certified"
        assert r["sup_lower"] == pytest.approx(base["sup_lower"], rel=1e-12)
        assert r["sup_upper"] == pytest.approx(base["sup_upper"], rel=1e-9)
        assert len(r["dropped_gauge_terms"]) >= 1
    assert A.magnitude_bound(f + " + 3.0 * square(x)", gauge="none")["status"] == "failed"
    ref = [A.magnitude_bound(f + c, gauge="reference") for c in ("", " + 3.0 * square(x)")]
    assert ref[0]["status"] == ref[1]["status"] and ref[0]["sup_lower"] == pytest.approx(ref[1]["sup_lower"])
    # a pure gauge term changes nothing in the model
    r = A.magnitude_bound("3.0 * x + exp(x)")
    assert r["status"] == "certified" and r["sup_upper"] == 0.0 and "does not change the model" in r["note"]


def test_gauge_term_that_is_not_a_top_level_summand_needs_slabs():
    # 5 x enters through linv(0 * rh) = 1: syntactically not a pure function of x, numerically a gauge term
    s = "0.9 * tanh(zt) + 5.0 * x * linv(rh * 0.0)"
    r = A.magnitude_bound(s)
    want = 0.45 * (math.tanh(BOX["zt"][1]) - math.tanh(BOX["zt"][0]))
    assert r["status"] == "certified" and r["n_slabs"] > 16
    assert r["sup_lower"] == pytest.approx(want, rel=1e-6) and r["sup_upper"] <= 1.0 + 1e-9


def test_three_gauges_are_ordered():
    """oscillation <= both other readings: whatever passes the literal rule or the reference rule passes it."""
    rng = np.random.default_rng(8)
    seen = 0
    for _ in range(60):
        e = random_expr(rng, 3, ["tanh", "atan", "erf", "linv", "square", "sin"], ["+", "-", "*"])
        e = T.op("*", T.const(0.3), e)
        if not [v for v in T.variables(e) if v != "x"] or not A.domain_safety(e)["safe"]:
            continue
        r = {g: A.magnitude_bound(e, gauge=g, limit=1e6, max_boxes=2000) for g in ("oscillation", "reference", "none")}
        seen += 1
        assert r["oscillation"]["sup_lower"] <= r["reference"]["sup_upper"] * (1 + 1e-9) + 1e-12
        assert r["oscillation"]["sup_lower"] <= r["none"]["sup_upper"] * (1 + 1e-9) + 1e-12
    assert seen > 20
    # the reference reading depends on the reference point; the oscillation does not
    s = "0.99 * tanh(zt * rh * x)"
    assert A.magnitude_bound(s)["status"] == "certified" and A.magnitude_bound(s, gauge="none")["status"] == "certified"
    assert A.magnitude_bound(s, gauge="reference")["status"] == "failed"
    assert A.magnitude_bound(s, gauge="reference", reference={"zt": 0.0, "rh": 0.0})["status"] == "certified"


def test_oscillation_failure_is_a_pair_at_one_gauge_point():
    s = "tanh(zt) * (0.3 * x)"                             # half-oscillation 0.15 |x| (1 + 0.818) at x = -4.11
    r = A.magnitude_bound(s)
    assert r["status"] == "failed"
    w = r["witness"]
    e = T.parse(s)
    hi = T.evaluate(e, {**w["max_at"], **w["gauge_point"]})
    lo = T.evaluate(e, {**w["min_at"], **w["gauge_point"]})
    assert hi - lo > 2.0 and 0.5 * (hi - lo) == pytest.approx(w["half_oscillation"])
    box = dict(BOX, x=(-1.0, 1.0))                         # on the training range the same equation passes
    assert A.magnitude_bound(s, box)["status"] == "certified"


def test_brute_force_agreement_on_random_expressions():
    rng = np.random.default_rng(21)
    done = 0
    for _ in range(40):
        e = T.op("*", T.const(float(rng.uniform(0.2, 1.2))),
                 random_expr(rng, 3, ["tanh", "atan", "erf", "linv", "sin", "cos"], ["+", "-", "*"]))
        if len([v for v in T.variables(e) if v != "x"]) == 0:
            continue
        r = A.magnitude_bound(e, max_boxes=20000)
        pts = sample(rng, BOX, 3)
        xs = np.linspace(*BOX["x"], 41) if "x" in T.variables(e) else np.array([0.0])
        zt, rh = np.meshgrid(np.linspace(*BOX["zt"], 201), np.linspace(*BOX["rh"], 201), indexing="ij")
        brute = 0.0
        for xv in xs:
            v = np.asarray(T.evaluate(e, zt=zt, rh=rh, x=np.full_like(zt, xv))) + np.zeros_like(zt)
            brute = max(brute, 0.5 * (v.max() - v.min()))
        done += 1
        if r["status"] == "failed":
            assert r["sup_lower"] > 1.0
        else:
            assert brute <= r["sup_upper"] * (1 + 1e-9) + 1e-12, (T.to_string(e), brute, r["sup_upper"])
            assert brute <= 1.0 + 1e-9
        if brute > 1.0 + 1e-6:
            assert r["status"] == "failed", (T.to_string(e), brute)
        del pts
    assert done > 20


# ----------------------------------------------------------------------------------------------------------------------
# audit(), templates, trajectories
# ----------------------------------------------------------------------------------------------------------------------
def test_audit_report_and_refit_repetition():
    rep = A.audit("tanh((zt * 1.3) + -0.4) * 0.5")
    assert rep.passed and rep.k == 3 and rep.complexity == 9 and rep.gauge == "oscillation" and rep.limit == 1.0
    assert rep.domain["status"] == "safe" and rep.magnitude["status"] == "certified" and "PASS" in rep.summary()
    json.dumps(rep.to_json())
    # constants move in a refit: the same structure with other constants can fail - the audit must be repeated
    e = T.parse("0.3 * linv((rh * 0.05) - 4.0) / (zt + 1.5)")
    assert A.audit(e).passed
    moved = T.with_constants(e, [0.3, 0.05, 4.0, 1.1])    # the pole z' = -286 mm has moved into the box
    rep2 = A.audit(moved)
    assert not rep2.passed and rep2.domain["status"] == "unsafe_proven"


def test_templates_bounded_by_form():
    sub = {"u": "(#1 * 3.0) - #2", "v": "square(#1) * #2", "w": "(#1 - 2.0) * #2"}
    rep = A.audit(T.parse_template("s1_tanh_sum", sub))
    assert rep.passed and rep.magnitude["status"] == "certified" and rep.magnitude["n_boxes"] <= 2
    assert A.audit(T.parse_template("s1_tanh_sum", sub), gauge="none").magnitude["status"] == "certified"
    s2 = {"f": "(#1 * 3.0) - #2", "a": "#1 * 50.0", "e": "(#1 - 2.0) * 0.01", "ph": "#1 * 7.0 + #3"}
    lit = A.audit(T.parse_template("s2_modulation", s2), gauge="none")       # |value| reaches 1.3: fails as written
    assert lit.magnitude["status"] == "failed" and 1.0 < lit.magnitude["sup_lower"] <= 1.3
    unit = A.audit(T.parse_template("s2_modulation_unit", s2), gauge="none")
    assert unit.magnitude["status"] == "certified" and unit.magnitude["sup_upper"] <= 1.0 + 1e-9
    # a template with an unbounded part is not protected by its form
    rep = A.audit(T.parse_template("s1_amp_shape", {"a": "#1 * 2.0", "f": "tanh(#1) + linv(#2)"}))
    assert not rep.passed and rep.magnitude["status"] == "failed"


def test_trajectory_audit():
    cubic = T.horner_string([0.3, 0.5, -0.2, 0.1])
    r = A.audit_trajectory(cubic)
    assert r["passed"] and r["domain"]["status"] == "safe" and r["k"] == 4 and r["complexity"] == 13
    xs = np.linspace(-4.2, 1.4, 20001)
    v = 0.3 + 0.5 * xs - 0.2 * xs ** 2 + 0.1 * xs ** 3
    assert r["range_enclosure"][0] <= v.min() and v.max() <= r["range_enclosure"][1]
    assert r["range_evaluated"][0] == pytest.approx(v.min(), abs=1e-6)
    assert r["extrapolation_ratio"] > 5                    # a cubic leaves its training band quickly
    assert not A.audit_trajectory("1 / (x + 2.0)")["passed"]
    assert A.audit_trajectory("0.5 * tanh(x * 2.0) + 0.1")["extrapolation_ratio"] < 1.1
    ok = A.audit_trajectory("0.5 * tanh(x * 2.0) + 0.1", value_range=(-0.5, 0.7))
    assert ok["passed"] and ok["value_range_check"]["status"] == "certified"
    assert not A.audit_trajectory(cubic, value_range=(-1.0, 1.0))["passed"]
    with pytest.raises(ValueError):
        A.audit_trajectory("zt + x")


def test_floored_x_box():
    s = "0.2 * exp(-1.0 * x)"                              # grows towards low momentum
    assert A.audit(s + " * tanh(zt)").magnitude["status"] == "failed"
    assert A.audit(s + " * tanh(zt)", box=A.audit_box(x_floor=-1.2)).passed


def test_unknown_variable_and_unresolved_template():
    with pytest.raises(KeyError):
        A.audit("tanh(q)")
    with pytest.raises(ValueError):
        A.domain_safety(T.parse("#1 + 1.0"), {"#1": (0.0, 1.0)}) if False else A.enclose(
            T.Expr("sub", name="u", args=(T.var("zt"),)))


# ----------------------------------------------------------------------------------------------------------------------
# (c) numerical hygiene: raw magnitude, float32 cancellation (user confirmation 1)
# ----------------------------------------------------------------------------------------------------------------------
def test_overflow_cap_alone_does_not_prevent_cancellation():
    """The premise that had to be checked: a node cap of 1e30 says nothing about a gauge term of 1e4."""
    e = T.parse("0.5 * tanh(zt - 2.0) + 1.0e4 * x")
    assert A.domain_safety(e)["safe"]                                       # far below the overflow cap
    assert A.magnitude_bound(e)["status"] == "certified"                    # H does not see the gauge term
    rng = np.random.default_rng(0)
    zt = rng.uniform(*BOX["zt"], 5000).astype(np.float32)
    x = np.full(5000, 0.731, dtype=np.float32)
    with torch.inference_mode():
        f64 = T.TorchExpr(e)(torch.tensor(zt), torch.zeros(5000), torch.tensor(x)).numpy()
        f32 = T.TorchExpr(e, dtype=torch.float32)(torch.tensor(zt), torch.zeros(5000), torch.tensor(x)).double().numpy()
    err = np.abs((f32 - f32.mean()) - (f64 - f64.mean())).max()
    assert err > 1e-4                                                        # measured: the (z) variation is damaged
    assert A.rounding_error_bound(e)["float32"] > 1e-3 and A.rounding_error_bound(e)["float64"] < 1e-10


def test_gauge_terms_are_stripped_and_the_raw_value_is_bounded():
    r = A.audit("0.5 * tanh(zt - 2.0) + 1.0e4 * x")
    n = r.numerics
    assert r.passed and r.certified and n["dropped_gauge_terms"] == ["+ 10000.0 * x"]
    assert n["expression"] == "0.5 * tanh(zt - 2.0)" and n["raw"]["status"] == "certified"
    assert n["raw"]["sup_upper"] <= 0.5 + 1e-9 and n["rounding"]["float32"] < 1e-6 and n["float32_safe"]
    # the same gauge term hidden inside a product is not a top-level summand: the raw bound catches it
    hidden = A.audit("0.5 * tanh(zt - 2.0) + 1.0e4 * x * linv(rh * 0.0)")
    assert not hidden.passed and hidden.numerics["raw"]["status"] == "failed" and hidden.domain["safe"]
    assert abs(hidden.numerics["raw"]["witness"]["value"]) > A.RAW_LIMIT_NAT
    assert hidden.numerics["rounding"]["float32"] > 1e-3 and "numerical hygiene" in " ".join(hidden.notes)
    # the limit itself: 16 nat on the raw value, whatever the gauge-invariant H says
    ok, bad = A.audit("0.5 * tanh(zt - 2.0) + 12.0 * linv(rh * 0.0)"), A.audit("0.5 * tanh(zt - 2.0) + 20.0 * linv(rh * 0.0)")
    assert ok.passed and bad.magnitude["status"] == "certified" and not bad.passed
    assert ok.magnitude["sup_upper"] == pytest.approx(bad.magnitude["sup_upper"], rel=1e-6)
    assert A.RAW_LIMIT_NAT == 16.0 and A.U32 * (A.RAW_LIMIT_NAT + 40.0) < 4e-6
    # a pure gauge expression has nothing left to evaluate
    g = A.numerics("3.0 * x + exp(x)")
    assert g["passed"] and g["certified"] and g["expression"] is None


def test_catastrophic_cancellation_is_rejected_even_in_float64():
    s = "(exp(zt * 6.0) + 0.5 * tanh(rh)) - exp(zt * 6.0)"                 # 1e20 + small - 1e20
    assert A.domain_safety(s)["safe"]                                       # no pole, below the overflow cap
    r = A.audit(s)
    assert not r.passed and r.numerics["rounding"]["float64"] > 1.0


def test_rounding_bound_covers_the_measured_float32_error():
    rng = np.random.default_rng(4)
    names = ("zt", "rh", "x")
    checked = 0
    for _ in range(120):
        e = random_expr(rng, int(rng.integers(2, 5)), UNARY_S1, ["+", "-", "*", "ppow"])
        if not T.variables(e) or not A.domain_safety(e, max_boxes=2000)["safe"]:
            continue
        bound = A.rounding_error_bound(e)
        pts = {n: rng.uniform(*BOX[n], 3000).astype(np.float32) for n in names}
        with torch.inference_mode():
            cols = [torch.tensor(pts[n]) for n in names]
            f64 = (T.TorchExpr(e)(*cols) + torch.zeros(3000, dtype=torch.float64)).numpy()
            f32 = (T.TorchExpr(e, dtype=torch.float32)(*cols) + torch.zeros(3000)).double().numpy()
        ok = np.isfinite(f32) & np.isfinite(f64)
        err = np.abs(f32 - f64)[ok].max()
        assert err <= bound["float32"] * 1.0001 + 1e-30, (T.to_string(e), err, bound)
        assert bound["float64"] <= bound["float32"]
        checked += 1
    assert checked > 60


# ----------------------------------------------------------------------------------------------------------------------
# screening against finalists (user confirmation 2)
# ----------------------------------------------------------------------------------------------------------------------
UNDECIDED = "0.9995 * (square(sin(zt * 3.0)) + square(cos(zt * 3.0))) * tanh(rh - 5.0)"   # true H = 0.99945
SMALL_BUDGET = {"max_boxes": 2000}


def test_undecided_passes_screening_but_not_the_finalist_rule_before_and_after_a_refit():
    e = T.parse(UNDECIDED)
    before = A.audit(e, magnitude_kw=SMALL_BUDGET)
    assert before.magnitude["status"] == "not_refuted" and before.passed and not before.certified
    assert before.magnitude["sup_lower"] <= 1.0 < before.magnitude["sup_upper"]       # neither proven nor refuted
    strict = A.audit(e, magnitude_kw=SMALL_BUDGET, require_certified=True)
    assert not strict.passed and strict.require_certified and "finalist needs a proven" in " ".join(strict.notes)
    after = T.with_constants(e, [0.9993, 3.01, 3.01, 5.02])                           # the refit moved the constants
    a2 = A.audit(after, magnitude_kw=SMALL_BUDGET)
    assert a2.magnitude["status"] == "not_refuted" and a2.passed and not a2.certified
    c = A.certify_finalist(after, magnitude_kw=SMALL_BUDGET, baseline_tail={"dlnr": -3.0, "dz_per_mm": -0.004})
    assert not c["certified"] and not c["audit_certified"]
    assert any("needs it certified" in r for r in c["reasons"])


def test_refit_that_moves_the_equation_outside_the_bound_loses_its_certificate():
    e = T.with_constants(T.parse(UNDECIDED), [0.9, 3.0, 3.0, 5.0])
    tail = {"dlnr": -3.0, "dz_per_mm": -0.004}
    c0 = A.certify_finalist(e, baseline_tail=tail)
    assert c0["certified"] and c0["audit_certified"] and c0["reasons"] == [] and c0["tail"]["evaluated"]
    assert c0["audit"]["magnitude"]["status"] == "certified" and c0["audit"]["certified"] is True
    moved = T.with_constants(e, [1.3, 3.0, 3.0, 5.0])                                # after the refit
    c1 = A.certify_finalist(moved, baseline_tail=tail)
    assert not c1["certified"] and c1["audit"]["magnitude"]["status"] == "failed"
    assert c1["hash"] != c0["hash"] and len(c1["reasons"]) >= 1
    pole = T.with_constants(T.parse("0.3 * linv((rh * 0.05) - 4.0) / (zt + 1.5)"), [0.3, 0.05, 4.0, 1.1])
    c2 = A.certify_finalist(pole, baseline_tail=tail)
    assert not c2["certified"] and c2["reasons"] == ["domain: unsafe_proven"] and c2["tail"]["slopes"] is None


def test_tail_slopes_and_the_tail_part_of_the_certificate():
    s = "-0.5 * l1p(rh) * 0.2 + 0.3 * tanh(zt - 6.0)"
    t = A.tail_slopes(s)
    r0, r1 = 700.0 / 21.9, BOX["rh"][1]
    lo, hi = -0.1 * r1 / (1 + r1), -0.1 * r0 / (1 + r0)                    # d f / d ln r = -0.1 rh / (1 + rh)
    assert t["dlnr"]["inf"] <= lo <= t["dlnr"]["inf_sampled"] and t["dlnr"]["sup_sampled"] <= hi <= t["dlnr"]["sup"]
    assert t["dlnr"]["sup"] - hi < 0.01
    assert t["dz_per_mm"]["sup"] == pytest.approx(0.3 / 260.0, rel=1e-9)   # the steepest point zt = 6 is in the tail
    assert t["dz_per_mm"]["inf"] >= 0.0 and t["dz_per_mm"]["sup_sampled"] <= t["dz_per_mm"]["sup"]
    ok = A.certify_finalist(s, baseline_tail={"dlnr": -3.1, "dz_per_mm": -0.004})
    assert ok["certified"] and ok["tail"]["r_ok"] and ok["tail"]["z_ok"] and not ok["tail"]["not_worse_than_baseline"]
    assert ok["tail"]["model_dz_per_mm_max"] == pytest.approx(-0.004 + 0.3 / 260.0)
    flat = A.certify_finalist(s, baseline_tail={"dlnr": -3.1, "dz_per_mm": -0.001})   # the rise beats the baseline
    assert not flat["certified"] and flat["audit_certified"] and "depth tail" in flat["reasons"][0]
    shallow = A.certify_finalist("0.2 * l1p(rh) * 0.5", baseline_tail={"dlnr": -2.05, "dz_per_mm": -0.004})
    assert not shallow["certified"] and "radial tail" in shallow["reasons"][0]
    missing = A.certify_finalist(s)                                          # no baseline slopes: not a certificate
    assert not missing["certified"] and missing["audit_certified"] and not missing["tail"]["evaluated"]
    assert "tail check not evaluated" in missing["reasons"][0]
    assert A.tail_slopes("0.3 * tanh(x)")["dlnr"]["sup"] == 0.0


def test_report_states_the_bound_after_normalisation():
    r = A.audit("0.9 * tanh(zt)")
    assert r.magnitude["delta_eff_bound"] == pytest.approx(2 * r.magnitude["sup_upper"])
    assert "|delta_eff| <= 2H" in r.summary() and "certified=True" in r.summary()
    j = r.to_json()
    assert j["certified"] is True and j["require_certified"] is False and "raw" in j["numerics"]


def test_canonical_gauge_form_for_de_duplication():
    a = T.parse("tanh(zt) + (linv(rh) + (square(zt) * 0.1))")
    b = T.parse("((tanh(zt) + x) + linv(rh)) + ((square(zt) * 0.1) - 3.0)")
    ca, cb = A.canonical_gauge_form(a), A.canonical_gauge_form(b)
    assert T.expr_hash(ca) == T.expr_hash(cb) and T.to_string(ca) == "(tanh(zt) + linv(rh)) + (square(zt) * 0.1)"
    kept, dropped = A.drop_gauge_terms(a)
    assert kept is a and dropped == []                     # nothing to strip: the tree itself
    assert T.expr_hash(kept) != T.expr_hash(A.drop_gauge_terms(b)[0])     # which is why the canonical form exists
    assert A.canonical_gauge_form(T.parse("3.0 * x - exp(x)")) is None
    assert T.to_string(A.canonical_gauge_form(T.parse("x - tanh(zt)"))) == "neg(tanh(zt))"
