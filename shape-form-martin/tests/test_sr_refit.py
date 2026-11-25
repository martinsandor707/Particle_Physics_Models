"""sfm.sr.refit on synthetic showers: model construction, recovery of known constants through ``sfm.train.fit``, the
audit of the refitted equation, controls C1 / C2, trajectory substitution, and the guards.

``common.py`` builds its grid on ``common.DEVICE`` (CUDA when available), so everything that evaluates a model runs
once in a child process with ``CUDA_VISIBLE_DEVICES=""`` (``_cpu_suite``), as in ``tests/test_train.py``. No real
data, no registry, no likelihood of a real candidate equation.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parent
TRUE_EQ = "0.8 * tanh((zt * 1.5) + -1.5)"
START_EQ = "0.5 * tanh((zt * 1.0) + -1.0)"


class BaseMap(nn.Module):
    """A fixed smooth baseline: Gaussian in depth (moving with ln p), exponential in r. No parameters."""

    def forward(self, lnp):
        import common as C
        z0 = 250.0 + 60.0 * lnp[:, None, None]
        logit = -0.5 * ((C.ZZ[None] - z0) / 260.0) ** 2 - C.RR[None] / 80.0
        return F.log_softmax(logit.reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R)

    def n_params(self):
        return 0


def sample_showers(model, n_ev, hits, seed, box=1e-3):
    """Events whose hits are drawn from ``model``'s own map (uniform inside the cell), axis along z, no truncation,
    equal hit weights, and a cell box so small that the fold does nothing: the NLL is then a proper likelihood whose
    maximum is at the generating constants."""
    import common as C
    g = torch.Generator().manual_seed(seed)
    lnp = torch.log(torch.tensor(0.3)) + torch.rand(n_ev, generator=g) * (math.log(15.0) - math.log(0.3))
    with torch.no_grad():
        P = model(lnp).exp().reshape(n_ev, -1).double()
    cell = torch.multinomial(P, hits, replacement=True, generator=g)                  # (E, hits)
    iz, ir = cell // C.N_R, cell % C.N_R
    z = C.Z_EDGES[iz] + torch.rand(iz.shape, generator=g) * C.DZ
    r1, r2 = C.R_EDGES[ir], C.R_EDGES[ir + 1]
    r = torch.sqrt(r1 ** 2 + torch.rand(ir.shape, generator=g) * (r2 ** 2 - r1 ** 2))
    a = torch.rand(ir.shape, generator=g) * 2 * math.pi
    xyz = torch.stack([r * torch.cos(a), r * torch.sin(a), z], -1).reshape(-1, 3).float()
    p = torch.exp(lnp)
    ev = {"p": p, "bin": p.floor().long(), "zb": torch.full((n_ev,), 5000.0),
          "R": torch.tensor(C.rotation(np.zeros(n_ev), np.zeros(n_ev))), "evis": torch.ones(n_ev), "lnp": lnp}
    return C.Showers(ev, torch.arange(n_ev + 1) * hits, xyz, torch.full((n_ev * hits,), 1.0 / hits),
                     torch.tensor([box, box, box], dtype=torch.float32))


def _cpu_suite(out_json: str, tmp: str) -> None:
    import common as C
    from sfm import train
    from sfm.families.params import PolyProvider
    from sfm.sr import audit as A
    from sfm.sr import refit as R
    from sfm.sr import translate as T

    assert C.DEVICE.type == "cpu", "the CPU suite must run without CUDA"
    tmp = Path(tmp)
    rep = {}
    base = BaseMap()
    truth = R.CorrectedModel(base, R.expression_correction(TRUE_EQ))
    rep["freeze_default"] = truth.freeze_base
    tr, va = sample_showers(truth, 500, 300, 1), sample_showers(truth, 120, 300, 2)

    # construction ---------------------------------------------------------------------------------------------------
    m = R.CorrectedModel(BaseMap(), R.expression_correction(START_EQ))
    lnp = torch.tensor([0.0, 1.0, 2.5])
    with torch.no_grad():
        logP = m(lnp)
        f = m.f(lnp)
    want = torch.log_softmax((base(lnp).double() + f).reshape(3, -1), 1).reshape(3, C.N_Z, C.N_R)
    rep["construct"] = {"shape": list(logP.shape), "dtype": str(logP.dtype), "sum": logP.exp().sum((1, 2)).tolist(),
                        "max_diff": float((logP.double() - want).abs().max()), "k": m.n_params(),
                        "params": [n for n, p in m.named_parameters() if p.requires_grad],
                        "c_dtype": str(m.correction.c.dtype),
                        "x": m.x(lnp).tolist(), "x_ref": [float(v) for v in __import__("sfm.sr.operators", fromlist=["x"]).x_of_p(np.exp(lnp.numpy()))],
                        "f_ref_diff": float((f[1] - torch.tensor(T.evaluate(T.parse(START_EQ), zt=(C.ZZ.double() / 260.0).numpy(),
                                                                            rh=(C.RR.double() / 21.9).numpy(), x=0.0))).abs().max())}

    # recovery of the constants -----------------------------------------------------------------------------------
    nll0 = float(np.mean(train.val_nll(m, va, 1)))
    nll_true = float(np.mean(train.val_nll(truth, va, 1)))
    cfg = train.FitConfig(steps=300, lr=0.05, lr_end=0.003, batch=64, k_train=1, k_val=1, eval_every=150,
                          name="sr_synth", out_dir=str(tmp / "sr"))
    refused = None
    try:
        R.fit_candidate(m, tr, va, cfg, label="x")
    except R.RefitNotAllowed as err:
        refused = str(err)
    out = R.fit_candidate(m, tr, va, cfg, label="synthetic recovery", synthetic=True, log=None, finalist=True,
                          baseline_tail={"dlnr": -3.0, "dz_per_mm": -0.004})
    rep["recovery"] = {"refused_without_flag": refused, "c": m.correction.c.tolist(), "nll0": nll0,
                       "nll_true": nll_true, "nll_fit": float(np.mean(train.val_nll(m, va, 1))),
                       "out": {k: v for k, v in out.items() if k not in ("audit_refit", "finalist")},
                       "finalist": {"certified": out["finalist"]["certified"], "reasons": out["finalist"]["reasons"],
                                    "tail": {k: v for k, v in out["finalist"]["tail"].items() if k != "slopes"},
                                    "status": out["finalist"]["audit"]["magnitude"]["status"],
                                    "require_certified": out["finalist"]["audit"]["require_certified"]},
                       "audit_status": [out["audit_refit"]["domain"]["status"], out["audit_refit"]["magnitude"]["status"]],
                       "summary": R.refit_summary(out), "grad_check": R.check_gradients(m, tr)}
    json.dumps(out)

    # controls -------------------------------------------------------------------------------------------------------
    c2 = R.control_c2(base, 3)
    with torch.no_grad():
        d0 = float((c2(lnp) - base(lnp)).abs().max())
        zt, rh, xx = c2.zt, c2.rh, torch.tensor([-0.7, 0.1, 0.9], dtype=torch.float64)[:, None, None]
        box = __import__("sfm.sr.operators", fromlist=["x"]).data_box()
        u = 2.0 * (zt - box["zt"][0]) / (box["zt"][1] - box["zt"][0]) - 1.0
        v = 2.0 * torch.sqrt(rh) / math.sqrt(box["rh"][1]) - 1.0
        probe = R.PolyControl(7)
        probe.c.copy_(torch.tensor([0.3, -0.2, 0.1, 0.05, -0.4, 0.25, 0.15], dtype=torch.float64))
        direct = sum(cj * u ** a * v ** b * xx ** c for cj, (a, b, c) in zip(probe.c, probe.exponents))
        tree_diff = float((probe(zt, rh, xx) - direct).abs().max())
        uv_range = [float(u.min()), float(u.max()), float(v.min()), float(v.max())]
    c2_nll_start = float(np.mean(train.val_nll(c2, va, 1)))
    out2 = R.fit_candidate(c2, tr, va, train.FitConfig(steps=120, lr=0.05, lr_end=0.003, batch=64, k_train=1, k_val=1,
                                                       eval_every=60, name="c2", out_dir=str(tmp / "c2")),
                           label="C2 polynomial", synthetic=True, log=None, role="control")
    json.dumps(out2)
    after = R.audit_control(c2)
    wild = R.PolyControl(3)                                 # a polynomial that fits the data box and leaves the bound
    with torch.no_grad():
        wild.c.copy_(torch.tensor([0.9, 0.0, 0.6], dtype=torch.float64))
    wild_rep = R.audit_control(wild)
    bounded = R.control_c2(base, 5, bounded=True)
    with torch.no_grad():
        d0b = float((bounded(lnp) - base(lnp)).abs().max())
        bounded.correction.c.copy_(torch.tensor([3.0, -2.0, 1.5, 4.0, -2.5], dtype=torch.float64))
    bounded_rep = R.audit_control(bounded)
    rep["c2"] = {"tree_diff": tree_diff, "uv_range": uv_range, "nll_start": c2_nll_start,
                 "out": {k: v for k, v in out2.items() if k not in ("audit_refit",)},
                 "after": {k: v for k, v in after.items() if k != "audit"}, "c_fit": c2.correction.c.tolist(),
                 "wild": wild_rep["verdict"], "wild_eq": wild_rep["equation"],
                 "bounded_zero_start_diff": d0b, "bounded": bounded_rep["verdict"], "bounded_label": bounded_rep["label"],
                 "bounded_k": bounded.n_correction(), "label": after["label"]}

    class Trainable(nn.Module):
        def __init__(self):
            super().__init__()
            self.s = nn.Parameter(torch.tensor(80.0), requires_grad=False)

        def forward(self, lnp):
            z0 = 250.0 + 60.0 * lnp[:, None, None]
            logit = -0.5 * ((C.ZZ[None] - z0) / 260.0) ** 2 - C.RR[None] / self.s
            return F.log_softmax(logit.reshape(len(lnp), -1), 1).reshape(len(lnp), C.N_Z, C.N_R)

    b1 = Trainable()
    c1 = R.control_c1(b1)
    rep["controls"] = {"c2_zero_start_diff": d0, "c2_k": c2.n_params(), "c2_exponents": c2.correction.exponents,
                       "c2_nll_fit": float(np.mean(train.val_nll(c2, va, 1))),
                       "c2_expr": T.to_string(c2.current_expr()), "c1_trainable": [p.requires_grad for p in c1.parameters()],
                       "c1_is_copy": c1 is not b1 and not b1.s.requires_grad}

    # strict mode ------------------------------------------------------------------------------------------------------
    bad = R.CorrectedModel(BaseMap(), R.expression_correction("0.1 * log(zt)"))
    try:
        bad(lnp)
        rep["strict"] = "no error"
    except FloatingPointError as err:
        rep["strict"] = str(err)
    loose = R.CorrectedModel(BaseMap(), R.expression_correction("0.1 * log(zt)"), strict=False)
    rep["loose_has_nan"] = bool(torch.isnan(loose(lnp)).any())
    rep["sqa_grad"] = R.check_gradients(R.CorrectedModel(BaseMap(), R.expression_correction("0.3 * sqa(rh - 0.2)")), tr)
    rep["pole_audit"] = A.audit("0.1 * log(zt)").domain["status"]

    # numerical hygiene (user confirmation 1) --------------------------------------------------------------------------
    g = R.expression_correction("0.5 * tanh(zt - 2.0) + 1.0e4 * x + 7.0")
    plain = R.CorrectedModel(BaseMap(), R.expression_correction("0.5 * tanh(zt - 2.0)"))
    gauge = R.CorrectedModel(BaseMap(), g)
    raw = R.CorrectedModel(BaseMap(), R.expression_correction("0.5 * tanh(zt - 2.0) + 1.0e4 * x + 7.0",
                                                              strip_gauge=False))
    with torch.no_grad():
        same = float((gauge(lnp) - plain(lnp)).abs().max())
        unstripped = float((raw(lnp) - plain(lnp)).abs().max())      # float64 inside: still the same model
    pure = None
    try:
        R.expression_correction("3.0 * x + exp(x)")
    except ValueError as err:
        pure = str(err)
    f32 = None
    try:
        R.CorrectedModel(BaseMap(), R.expression_correction("0.5 * tanh(zt)", dtype=torch.float32))(lnp)
    except TypeError as err:
        f32 = str(err)
    r5 = float(C.RR.double()[0, 5] / 21.9)                  # sqa(rh - r5) is exactly 0 on ring 5: a cusp on the grid
    cusp = R.CorrectedModel(BaseMap(), R.expression_correction(f"0.3 * sqa(rh - {r5!r}) + 0.2 * sqa(zt * 0.0)"))
    with torch.no_grad():
        zero_cells = int((cusp.rh - r5 == 0).sum())
    rep["hygiene"] = {"k": gauge.n_correction(), "dropped": g.dropped_gauge_terms, "same": same,
                      "unstripped": unstripped, "pure": pure, "float32": f32, "zero_cells": zero_cells,
                      "cusp_grad": R.check_gradients(cusp, tr)}

    # trajectory substitution -------------------------------------------------------------------------------------
    prov = PolyProvider(["a", "b"], {"a": 3, "b": 1}, init={"a": [0.1, 0.2, 0.3, 0.4], "b": [1.0, -0.5]})
    sub = R.SubstitutedProvider(prov, {"a": "0.3 * tanh(x * 2.0)"})
    raw0, raw1 = prov(lnp), sub(lnp)
    x = prov.x(lnp)
    rep["subst"] = {"k_before": prov.n_params(), "k_after": sub.n_params(),
                    "a_diff": float((raw1[:, 0] - 0.3 * torch.tanh(2.0 * x)).abs().max()),
                    "b_diff": float((raw1[:, 1] - raw0[:, 1]).abs().max()),
                    "exprs": {k: T.to_string(v) for k, v in sub.current_exprs().items()}}
    try:
        R.register_refit(out)
        rep["register"] = "no error"
    except NotImplementedError as err:
        rep["register"] = str(err)
    rep["registry_imported"] = "sfm.registry" in sys.modules
    rep["monomials"] = {str(k): R.c2_monomials(k) for k in (1, 3, 5, 7, 20, 40)}
    try:
        R.PolyControl(0)
        rep["poly0"] = "no error"
    except ValueError as err:
        rep["poly0"] = str(err)
    Path(out_json).write_text(json.dumps(rep))


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sr_refit")
    out = tmp / "report.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", SFM_ALLOW_CPU="1")   # CPU unit test
    env.pop("SFM_SR_REFIT_ALLOWED", None)
    code = (f"import sys; sys.dont_write_bytecode = True; sys.path[:0] = [{str(MARTIN / 'src')!r}, {str(MARTIN)!r}, "
            f"{str(HERE)!r}]; import test_sr_refit as M; M._cpu_suite({str(out)!r}, {str(tmp)!r})")
    proc = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    return json.loads(out.read_text())


def test_model_construction(suite):
    r = suite["construct"]
    assert suite["freeze_default"] is False                           # PLAN: all constants trainable
    assert r["shape"] == [3, 128, 32] and r["dtype"] == "torch.float32" and r["c_dtype"] == "torch.float64"
    assert all(abs(s - 1.0) < 1e-5 for s in r["sum"]) and r["max_diff"] < 1e-5
    assert r["k"] == 3 and r["params"] == ["correction.c"]            # all constants of the equation, nothing else
    assert np.allclose(r["x"], r["x_ref"], atol=1e-6) and r["f_ref_diff"] < 1e-12


def test_constants_are_recovered(suite):
    r = suite["recovery"]
    assert "H200" in r["refused_without_flag"]                         # no refit on the laptop without the flag
    c = r["c"]
    assert c[0] == pytest.approx(0.8, abs=0.06) and c[1] == pytest.approx(1.5, abs=0.15) \
        and c[2] == pytest.approx(-1.5, abs=0.15), c
    assert r["nll_fit"] < r["nll0"] - 1e-3 and abs(r["nll_fit"] - r["nll_true"]) < 2e-3, r
    assert r["out"]["fit"]["skipped_loss"] == 0 and r["out"]["fit"]["skipped_grad"] == 0
    assert r["out"]["k"] == 3 and r["out"]["registered"] is False and r["out"]["synthetic"] is True
    assert r["grad_check"] == {"loss_finite": True, "gradients_finite": True, "unused_parameters": 0,
                               "n_parameters": 1}


def test_refitted_equation_is_audited_again(suite):
    r = suite["recovery"]
    assert r["out"]["audit_passed"] is True and r["audit_status"] == ["safe", "certified"]
    assert r["out"]["expr_start"] == "0.5 * tanh((zt * 1.0) + -1.0)"
    from sfm.sr import translate as T
    e = T.from_dict(r["out"]["expr_refit_dict"])
    assert T.constants(e) == pytest.approx(r["c"]) and T.to_string(e) == r["out"]["expr_refit"]
    assert "audit_refit=PASS" in r["summary"]


def test_controls(suite):
    r = suite["controls"]
    assert r["c2_zero_start_diff"] < 5e-6 and r["c2_k"] == 3            # zero start: the baseline (float32 rounding)
    assert r["c2_exponents"] == [[1, 0, 0], [0, 1, 0], [2, 0, 0]]
    assert r["c2_nll_fit"] < suite["c2"]["nll_start"] - 1e-3            # the control does fit something
    assert r["c2_nll_fit"] > suite["recovery"]["nll_fit"]              # 3 polynomial constants do not reach the truth
    assert "sqa(rh)" in r["c2_expr"] and "zt" in r["c2_expr"]
    assert r["c1_trainable"] == [True] and r["c1_is_copy"]


def test_c2_is_the_polynomial_and_gets_the_same_audit(suite):
    """User confirmation 4: C2 in (zeta, sqrt(rho), x) is audited on the extended box like an equation, after its
    fit, and the verdict is returned so that it can be printed beside Delta(SR) - Delta(C2)."""
    r = suite["c2"]
    assert r["tree_diff"] < 1e-12                                       # the tree is u^a v^b x^c, as before
    u0, u1, v0, v1 = r["uv_range"]
    assert -1.0 < u0 < -0.98 and 0.98 < u1 < 1.0 and -1.0 < v0 < v1 < 1.0   # rescaled to [-1, 1] on the data box
    out = r["out"]
    assert out["role"] == "control" and out["k"] == 3 and out["k_equation"] == 3 and out["registered"] is False
    v = out["audit_verdict"]
    assert set(v) >= {"passed", "certified", "domain", "magnitude", "H_lower", "H_upper", "violates_extended_box_bound"}
    assert v["domain"] == "safe" and v["magnitude"] in ("certified", "not_refuted", "failed")
    assert r["after"]["verdict"] == v and r["after"]["k"] == 3 and r["label"] == "C2 polynomial (unbounded)"
    assert out["expr_refit"] == r["after"]["equation"] and out["finalist_certified"] is None
    # a polynomial that is harmless on the data box is not bounded outside it: reported, not hidden
    w = r["wild"]
    assert w["domain"] == "safe" and w["magnitude"] == "failed" and w["violates_extended_box_bound"] and not w["passed"]
    assert w["H_lower"] > 1.0
    # the labelled alternative tanh(polynomial) passes by construction, whatever the coefficients
    b = r["bounded"]
    assert b["passed"] and b["certified"] and b["magnitude"] == "certified" and b["H_upper"] <= 1.0 + 1e-9
    assert not b["violates_extended_box_bound"] and r["bounded_k"] == 5 and r["bounded_zero_start_diff"] < 5e-6
    assert r["bounded_label"] == "C2 tanh(polynomial), bounded by form"


def test_finalist_certificate_is_computed_on_the_refitted_constants(suite):
    """User confirmation 2: the proven bound, the domain check and the tail check are re-run after the refit."""
    r = suite["recovery"]
    f = r["finalist"]
    assert r["out"]["finalist_certified"] is True and f["certified"] and f["reasons"] == []
    assert f["status"] == "certified" and f["require_certified"] is True
    assert f["tail"]["evaluated"] and f["tail"]["r_ok"] and f["tail"]["z_ok"]
    assert r["out"]["audit_certified"] is True and r["out"]["audit_verdict"]["certified"] is True
    from sfm.sr import audit as A, translate as T
    e = T.from_dict(r["out"]["expr_refit_dict"])                        # the certificate is of THESE constants
    assert T.constants(e) == pytest.approx(r["c"]) and T.constants(e) != [0.5, 1.0, -1.0]
    assert A.certify_finalist(e, baseline_tail={"dlnr": -3.0, "dz_per_mm": -0.004})["certified"]
    assert "finalist_certified=True" in r["summary"] and "certified=True" in r["summary"]


def test_gauge_terms_are_stripped_and_float64_is_enforced(suite):
    """User confirmation 1: no float32 cancellation from a pure-x gauge term in the refit."""
    r = suite["hygiene"]
    assert r["k"] == 2 and r["dropped"] == ["+ 10000.0 * x", "+ 7.0"]   # k = 2 of 4: gauge constants not counted
    assert r["same"] == 0.0 and r["unstripped"] < 1e-5                  # the model is the same; float64 inside
    assert "does not change the model" in r["pure"] and "float64" in r["float32"]
    assert r["zero_cells"] == 128                                       # an exact zero of the sqa argument on the grid
    assert r["cusp_grad"]["loss_finite"] and r["cusp_grad"]["gradients_finite"]


def test_c2_monomials_exclude_what_the_softmax_cannot_see(suite):
    for k, mono in suite["monomials"].items():
        mono = [tuple(m) for m in mono]
        assert len(mono) == int(k) == len(set(mono))
        assert all(a + b >= 1 for a, b, c in mono)                     # no constant, no pure power of x
        deg = [a + b + c for a, b, c in mono]
        assert deg == sorted(deg)                                      # graded
    assert suite["monomials"]["5"] == [[1, 0, 0], [0, 1, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1]]
    assert "k must be" in suite["poly0"]


def test_no_clamps_and_strict_mode(suite):
    assert "not finite" in suite["strict"] and suite["loose_has_nan"] is True
    assert suite["pole_audit"] == "unsafe_proven"                      # and the audit would have stopped it before
    assert suite["sqa_grad"]["gradients_finite"] is True
    src = (MARTIN / "src" / "sfm" / "sr" / "refit.py").read_text()
    code = "\n".join(line.split("#")[0] for line in src.split('"""')[2::2] for line in line.split("\n"))
    assert "nan_to_num(" not in code and ".clamp(" not in code and "clamp_min(" not in code


def test_trajectory_substitution(suite):
    r = suite["subst"]
    assert r["k_before"] == 6 and r["k_after"] == 6 - 4 + 2
    assert r["a_diff"] < 1e-6 and r["b_diff"] == 0.0 and list(r["exprs"]) == ["a"]


def test_registration_is_a_stub_and_registry_is_not_imported(suite):
    assert "stub" in suite["register"] and suite["registry_imported"] is False
    src = (MARTIN / "src" / "sfm" / "sr" / "refit.py").read_text()
    assert "import registry" not in src and "from .. import registry" not in src and "sfm.registry import" not in src
