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
    import test_sr_checks as TC
    # tail-criterion record of a synthetic baseline r^-3.2 exp(-z/100 mm): rule "<= -2", depth slope -0.01 / mm
    zz, rr = np.meshgrid(TC.Z, TC.RN, indexing="ij")
    steep = TC.K.decide_tail_criterion(np.stack([-3.2 * np.log(rr) - zz / 100.0] * 7), TC.Z, TC.RN, TC.MOM, TC.Z90,
                                       baseline_hash="hash-of-steep", label="synthetic baseline, fast depth decay")
    out = R.fit_candidate(m, tr, va, cfg, label="synthetic recovery", synthetic=True, log=None, finalist=True,
                          tail_criterion=steep, baseline_hash="hash-of-steep")
    # the same refitted equation on a baseline that decays more slowly in depth (-0.004 / mm): its rise wins
    slow = A.certify_finalist(m.current_expr(), tail_criterion=TC.record("steep"), baseline_hash="hash-of-steep")
    old_arg = None
    try:
        R.fit_candidate(m, tr, va, cfg, label="x", synthetic=True, log=None, baseline_tail={"dlnr": -3.0})
    except ValueError as err:
        old_arg = str(err)
    rep["recovery"] = {"refused_without_flag": refused, "c": m.correction.c.tolist(), "nll0": nll0,
                       "nll_true": nll_true, "nll_fit": float(np.mean(train.val_nll(m, va, 1))),
                       "out": {k: v for k, v in out.items() if k not in ("audit_refit", "finalist")},
                       "old_arg": old_arg, "steep_hash": steep["record_hash"],
                       "slow": {"certified": slow["certified"], "audit_certified": slow["audit_certified"],
                                "reasons": slow["reasons"], "depth": slow["tail"]["depth"]},
                       "finalist": {"certified": out["finalist"]["certified"], "reasons": out["finalist"]["reasons"],
                                    "tail": out["finalist"]["tail"], "baseline_hash": out["finalist"]["baseline_hash"],
                                    "tail_criterion_hash": out["finalist"]["tail_criterion_hash"],
                                    "status": out["finalist"]["audit"]["magnitude"]["status"],
                                    "stage": out["finalist"]["audit"]["stage"],
                                    "require_certified": out["finalist"]["audit"]["require_certified"]},
                       "audit_status": [out["audit_refit"]["domain"]["status"], out["audit_refit"]["magnitude"]["status"]],
                       "summary": R.refit_summary(out), "grad_check": R.check_gradients(m, tr)}
    json.dumps(out)

    # controls (S1: tanh(polynomial) governs; the plain polynomial is a diagnostic row) ------------------------------
    c2 = R.control_c2(base, 3)                              # the default IS the registered, bounded control
    with torch.no_grad():
        d0 = float((c2(lnp) - base(lnp)).abs().max())
        zt, rh, xx = c2.zt, c2.rh, torch.tensor([-0.7, 0.1, 0.9], dtype=torch.float64)[:, None, None]
        box = __import__("sfm.sr.operators", fromlist=["x"]).data_box()
        u = 2.0 * (zt - box["zt"][0]) / (box["zt"][1] - box["zt"][0]) - 1.0
        v = 2.0 * torch.sqrt(rh) / math.sqrt(box["rh"][1]) - 1.0
        coef = torch.tensor([0.3, -0.2, 0.1, 0.05, -0.4, 0.25, 0.15], dtype=torch.float64)
        probe_plain, probe = R.PolyControl(7, bounded=False), R.PolyControl(7)
        probe_plain.c.copy_(coef)
        probe.c.copy_(coef)
        direct = sum(cj * u ** a * v ** b * xx ** c for cj, (a, b, c) in zip(coef, probe.exponents))
        tree_diff = float((probe_plain(zt, rh, xx) - direct).abs().max())
        tanh_diff = float((probe(zt, rh, xx) - torch.tanh(direct)).abs().max())
        uv_range = [float(u.min()), float(u.max()), float(v.min()), float(v.max())]
    c2_nll_start = float(np.mean(train.val_nll(c2, va, 1)))
    c2cfg = dict(steps=120, lr=0.05, lr_end=0.003, batch=64, k_train=1, k_val=1, eval_every=60)
    out2 = R.fit_candidate(c2, tr, va, train.FitConfig(**c2cfg, name="c2", out_dir=str(tmp / "c2")),
                           label="C2", synthetic=True, log=None, role="control")
    plain = R.control_c2(base, 3, bounded=False)
    refused_plain = None
    try:
        R.fit_candidate(plain, tr, va, train.FitConfig(**c2cfg, name="c2p", out_dir=str(tmp / "c2p_refused")),
                        label="C2 plain", synthetic=True, log=None, role="control")
    except ValueError as err:
        refused_plain = str(err)
    refused_candidate = None
    try:
        R.fit_candidate(R.control_c2(base, 3), tr, va, train.FitConfig(**c2cfg, name="c2x", out_dir=str(tmp / "c2x")),
                        label="C2 as candidate", synthetic=True, log=None)
    except ValueError as err:
        refused_candidate = str(err)
    out2p = R.fit_candidate(plain, tr, va, train.FitConfig(**c2cfg, name="c2p", out_dir=str(tmp / "c2p")),
                            label="C2 plain", synthetic=True, log=None, role="diagnostic")
    json.dumps(out2), json.dumps(out2p)
    report = R.c2_report(out2, out2p)
    mixups = []
    for g_, d_ in ((out2p, None), (out2, out2), (out, None)):
        try:
            R.c2_report(g_, d_)
            mixups.append("no error")
        except ValueError as err:
            mixups.append(str(err))
    after = R.audit_control(c2)
    wild = R.PolyControl(3, bounded=False)                  # a polynomial that fits the data box and leaves the bound
    with torch.no_grad():
        wild.c.copy_(torch.tensor([0.9, 0.0, 0.6], dtype=torch.float64))
    wild_rep = R.audit_control(wild)
    bounded = R.control_c2(base, 5)
    with torch.no_grad():
        d0b = float((bounded(lnp) - base(lnp)).abs().max())
        bounded.correction.c.copy_(torch.tensor([3.0, -2.0, 1.5, 4.0, -2.5], dtype=torch.float64))
    bounded_rep = R.audit_control(bounded)
    rep["c2"] = {"tree_diff": tree_diff, "tanh_diff": tanh_diff, "uv_range": uv_range, "nll_start": c2_nll_start,
                 "default_bounded": [c2.correction.bounded, R.PolyControl(2).bounded, c2.correction.governing],
                 "out": {k: v for k, v in out2.items() if k not in ("audit_refit", "clip_log")},
                 "out_plain": {k: v for k, v in out2p.items() if k not in ("audit_refit", "clip_log")},
                 "nll_plain": float(np.mean(train.val_nll(plain, va, 1))),
                 "refused_plain": refused_plain, "refused_candidate": refused_candidate, "report": report,
                 "mixups": mixups, "after": {k: v for k, v in after.items() if k != "audit"},
                 "c_fit": c2.correction.c.tolist(), "wild": wild_rep["verdict"], "wild_label": wild_rep["label"],
                 "bounded_zero_start_diff": d0b, "bounded": bounded_rep["verdict"], "bounded_label": bounded_rep["label"],
                 "bounded_k": bounded.n_correction(), "label": after["label"],
                 "labels": [R.C2_GOVERNING_LABEL, R.C2_DIAGNOSTIC_LABEL]}

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

    # S3: a stripped gauge term is traceable from the pre-refit audit to the post-refit record, and never in k ----------
    gauge_eq = "0.5 * tanh((zt * 1.0) + -1.0) + 3.0 * x + 7.0"
    pre = A.audit(gauge_eq, stage=R.STAGE_BEFORE).to_json()
    gm = R.CorrectedModel(BaseMap(), R.expression_correction(gauge_eq))
    gout = R.fit_candidate(gm, tr, va, train.FitConfig(steps=8, lr=0.02, lr_end=0.01, batch=64, k_train=1, k_val=1,
                                                       eval_every=8, name="gauge", out_dir=str(tmp / "gauge")),
                           label="gauge trace", synthetic=True, log=None, pre_refit_audit=pre)
    other = A.audit("0.5 * tanh((zt * 1.0) + -1.0) + 2.0 * x", stage=R.STAGE_BEFORE).to_json()
    gm2 = R.CorrectedModel(BaseMap(), R.expression_correction(gauge_eq))
    gout2 = R.fit_candidate(gm2, tr, va, train.FitConfig(steps=2, lr=0.02, lr_end=0.01, batch=64, k_train=1, k_val=1,
                                                         eval_every=2, name="gauge2", out_dir=str(tmp / "gauge2")),
                            label="gauge mismatch", synthetic=True, log=None, pre_refit_audit=other)
    rep["trace"] = {"pre": {k: pre[k] for k in ("k", "k_as_written", "k_stripped", "stripped_gauge_terms", "stage")},
                    "n_param_elements": int(sum(p.numel() for p in gm.parameters() if p.requires_grad)),
                    "c": gm.correction.c.tolist(), "module_records": gm.correction.stripped_gauge_terms,
                    "out": {k: gout[k] for k in ("k", "k_equation", "k_as_written", "stripped_gauge_terms",
                                                  "dropped_gauge_terms", "expr_refit", "warnings")},
                    "audit_refit": {k: gout["audit_refit"][k] for k in ("k", "k_as_written", "k_stripped", "stage",
                                                                         "stripped_gauge_terms", "equation")},
                    "refit_constants": T.constants(T.from_dict(gout["expr_refit_dict"])),
                    "mismatch_warnings": gout2["warnings"], "summary": R.refit_summary(gout)}

    # S5: clip counts per constant and per step; a constant sitting on a cusp ---------------------------------------
    n = 200
    alt = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    seq = {}
    grads = np.stack([1e3 * alt, np.full(n, 1e-2)], 1)                      # constant 0 bounces with huge gradients
    vals = np.stack([1.0 + 1e-3 * alt, 2.0 + 1e-4 * np.arange(n)], 1)
    norm = np.sqrt((grads ** 2).sum(1))
    seq["bouncing_on_cusp"] = R.clip_statistics(norm, grads, vals, 5.0, in_cusp=[True, False], start_values=[1.0, 2.0])
    seq["bouncing_no_cusp"] = R.clip_statistics(norm, grads, vals, 5.0, in_cusp=[False, False], start_values=[1.0, 2.0])
    g2 = np.stack([np.full(n, 50.0), np.full(n, 1e-2)], 1)                  # large gradient, steady progress
    v2 = np.stack([1.0 - 1e-3 * np.arange(1, n + 1), np.full(n, 2.0)], 1)
    seq["moving"] = R.clip_statistics(np.sqrt((g2 ** 2).sum(1)), g2, v2, 5.0, in_cusp=[True, False], start_values=[1.0, 2.0])
    seq["quiet"] = R.clip_statistics(np.full(n, 0.3), 0.2 * grads / 1e3, vals, 5.0, in_cusp=[True, False])
    g3 = grads.copy()
    g3[10:] = 1e-3 * alt[10:, None]
    seq["short_burst"] = R.clip_statistics(np.sqrt((g3 ** 2).sum(1)), g3, vals, 5.0, in_cusp=[True, False])
    n3 = norm.copy()
    n3[5], n3[6] = np.nan, np.inf
    seq["nonfinite"] = R.clip_statistics(n3, grads, vals, 5.0, in_cusp=[True, False])
    g4 = np.stack([1e3 * alt, 2e3 * np.ones(n)], 1)                         # clipped, but another parameter drives
    seq["other_drives"] = R.clip_statistics(np.sqrt((g4 ** 2).sum(1)), g4, vals, 5.0, in_cusp=[True, False])
    rep["clip_unit"] = {k: {"flags": v["flags"], "n_clipped": v["n_clipped"], "n_steps": v["n_steps"],
                            "n_no_backward": v["n_no_backward"], "n_nonfinite_gradient": v["n_nonfinite_gradient"],
                            "per_constant": v["per_constant"], "per_step_len": len(v["per_step"]["clipped"]),
                            "driven_first": v["per_step"]["driven_by"][:3], "definitions": sorted(v["definitions"])}
                        for k, v in seq.items()}
    rep["inside_sqa"] = [R.constants_inside_sqa(T.parse(s)) for s in
                         ("0.6 * sqa(zt - 1.2)", "0.5 * tanh((zt * 1.0) + -1.0)", "sqa((rh * 0.3) + x) * 2.0 + 1.5 * zt")]
    zk = float(C.ZZ.double()[40, 0] / 260.0)                 # a grid row: the likelihood has a kink at c = zk
    cusp_truth = R.CorrectedModel(BaseMap(), R.expression_correction(f"0.6 * sqa(zt - {zk!r})"))
    ctr, cva = sample_showers(cusp_truth, 400, 300, 11), sample_showers(cusp_truth, 80, 300, 12)
    cusp = {}
    for name, clip, eq in (("tight_clip", 1e-4, f"0.6 * sqa(zt - {zk + 0.004!r})"),
                           ("default_clip", 5.0, f"0.6 * sqa(zt - {zk + 0.004!r})"),
                           ("no_cusp_in_grid", 5.0, "0.5 * sqa(zt + 2.0)")):
        cm = R.CorrectedModel(BaseMap(), R.expression_correction(eq))
        co = R.fit_candidate(cm, ctr, cva, train.FitConfig(steps=160, lr=0.004, lr_end=0.0004, batch=64, k_train=1,
                                                           k_val=1, eval_every=80, clip=clip, name=name,
                                                           out_dir=str(tmp / name)),
                             label=name, synthetic=True, log=None)
        json.dumps(co)
        cusp[name] = {"flags": co["clip_log"]["flags"], "n_clipped": co["clip_log"]["n_clipped"],
                      "n_steps": co["clip_log"]["n_steps"], "max_norm": co["clip_log"]["max_norm"],
                      "per_constant": co["clip_log"]["per_constant"], "cusp_stalled": co["cusp_stalled"],
                      "hover": co["cusp_hover"], "hover_suspect": co["cusp_hover_suspect"], "warnings": co["warnings"],
                      "c": cm.correction.c.tolist(), "zk": zk, "summary": R.refit_summary(co),
                      "per_step_clipped_sum": int(sum(co["clip_log"]["per_step"]["clipped"]))}
    rep["cusp"] = cusp
    rep["recovery_clip"] = {"flags": out["clip_log"]["flags"], "n_clipped": out["clip_log"]["n_clipped"],
                            "n_steps": out["clip_log"]["n_steps"], "hover": out["cusp_hover"],
                            "cusp_stalled": out["cusp_stalled"]}

    # S4: the tail criterion of a closed-form baseline; PREVIEW on the legacy formula (not the baseline of record) ------
    rep["legacy"] = None
    ck = __import__("sfm.paths", fromlist=["x"]).PION_RESULTS_DIR / "new3_std_n3.pt"
    if ck.exists():
        import hashlib
        from sfm.families import build_variant, load_legacy_core
        core, _ = load_legacy_core("cpu")
        std = build_variant("legacy_centre", core, None, device="cpu", dtype=torch.float64)
        ext = build_variant("legacy_centre", core, None, device="cpu", dtype=torch.float64,
                            grid=R.extended_ring_grid(std.grid))
        h = hashlib.sha256(ck.read_bytes()).hexdigest()
        label = "LEGACY component formula (preview) - not the refitted baseline of record"
        rec = R.tail_criterion_for_baseline(std, ext, baseline_hash=h, label=label)
        cd = R.continued_log_density(std, ext, torch.log(torch.tensor([1.0, 20.0], dtype=torch.float64)))
        wrong = None
        try:
            other_model = build_variant("legacy_centre", core, None, device="cpu", dtype=torch.float64,
                                        grid=R.extended_ring_grid(std.grid))
            with torch.no_grad():
                other_model.provider.coef.mul_(1.05)
            R.tail_criterion_for_baseline(std, other_model, baseline_hash=h, label=label)
        except ValueError as err:
            wrong = str(err)
        Pz = np.exp(cd["log_p_std"]).sum(2)                                  # r-integrated depth profile, 1 and 20 GeV
        (tmp / "legacy_tail_criterion.json").write_text(json.dumps(rec))
        import os as _os
        if _os.environ.get("SFM_SR_PREVIEW_OUT"):
            Path(_os.environ["SFM_SR_PREVIEW_OUT"]).write_text(json.dumps(rec))
        from sfm.sr import checks as K
        rep["legacy"] = {"record": {k: rec[k] for k in ("label", "baseline_hash", "criterion", "record_hash",
                                                         "covers_r_max", "note", "momenta_GeV", "density", "n_depths",
                                                         "z_range_mm")},
                         "r_range": [rec["r_nodes_mm"][0], rec["r_nodes_mm"][-1]], "n_r": len(rec["r_nodes_mm"]),
                         "radial": {k: rec["radial"][k] for k in ("baseline_max", "baseline_max_at",
                                                                  "baseline_max_per_momentum", "fraction_above_limit",
                                                                  "baseline_satisfies_limit", "baseline_monotone")},
                         "depth": {k: rec["depth"][k] for k in ("z90_mm", "baseline_max", "baseline_max_at",
                                                                 "baseline_satisfies", "baseline_max_per_momentum")},
                         "match_error": cd["match_error"], "shape": list(cd["log_rho"].shape),
                         "r_max": float(cd["r_mm"][-1]), "wrong_model": wrong, "text": K.format_tail_criterion(rec),
                         "integrated_profile_max_dz_beyond_z90": [
                             float(np.max(np.diff(np.log(Pz[i]))[cd["z_mm"][:-1] > z90] / np.diff(cd["z_mm"])[cd["z_mm"][:-1] > z90]))
                             for i, z90 in ((0, rec["depth"]["z90_mm"][2]), (1, rec["depth"]["z90_mm"][4]))],
                         "certify": {eq: {"certified": c["certified"], "reasons": c["reasons"],
                                          "radial": c["tail"]["radial"]["passed"], "depth": c["tail"]["depth"]["passed"],
                                          "depth_no_flattening": c["tail"]["depth"]["no_flattening_passed"]}
                                     for eq, c in ((eq, A.certify_finalist(eq, tail_criterion=rec, baseline_hash=h))
                                                   for eq in ("-0.3 * tanh(zt - 2.0) - 0.1 * l1p(rh)",
                                                              "-0.3 * tanh(zt - 2.0) + 0.1 * l1p(rh)"))}}

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
    assert r["c2_expr"].startswith("tanh(") and "sqa(rh)" in r["c2_expr"] and "zt" in r["c2_expr"]
    assert r["c1_trainable"] == [True] and r["c1_is_copy"]


def test_registered_c2_is_tanh_of_the_polynomial(suite):
    """User decision S1 (PLAN 17.11): tanh(polynomial) is the governing C2 everywhere; the plain polynomial is an
    optional diagnostic row with its audit verdict printed beside it."""
    r = suite["c2"]
    assert r["default_bounded"] == [True, True, True]                   # control_c2 and PolyControl default
    assert r["tree_diff"] < 1e-12 and r["tanh_diff"] < 1e-12            # f = tanh(sum c u^a v^b x^c)
    u0, u1, v0, v1 = r["uv_range"]
    assert -1.0 < u0 < -0.98 and 0.98 < u1 < 1.0 and -1.0 < v0 < v1 < 1.0   # rescaled to [-1, 1] on the data box
    gov, dia = r["out"], r["out_plain"]
    assert gov["role"] == "control" and gov["governing"] is True and gov["control_label"] == r["labels"][0]
    assert gov["k"] == 3 and gov["k_equation"] == 3 and gov["registered"] is False
    v = gov["audit_verdict"]
    assert v["passed"] and v["certified"] and v["magnitude"] == "certified" and v["H_upper"] <= 1.0 + 1e-9
    assert not v["violates_extended_box_bound"] and gov["audit_passed"] is True and gov["expr_refit"].startswith("tanh(")
    assert r["after"]["verdict"] == v and r["after"]["k"] == 3 and r["label"] == r["labels"][0]
    # the plain polynomial: refused as the governing control, accepted as a diagnostic row, verdict beside it
    assert "role='diagnostic'" in r["refused_plain"] and "not a candidate" in r["refused_candidate"]
    assert dia["role"] == "diagnostic" and dia["governing"] is False and dia["control_label"] == r["labels"][1]
    assert dia["audit_verdict"]["domain"] == "safe" and not dia["expr_refit"].startswith("tanh(")
    assert dia["audit_verdict"]["violates_extended_box_bound"] is True  # on this problem it leaves the bound
    rep = r["report"]
    assert rep["governing"]["used_for"] == "Delta(SR) - Delta(C2)" and rep["governing"]["audit_passed"]
    assert rep["diagnostic"]["used_for"].startswith("nothing") and rep["diagnostic"]["violates_extended_box_bound"]
    assert rep["warnings"] == [] and len(rep["text"].splitlines()) == 2
    assert "governing" in rep["text"].splitlines()[0] and "diagnostic" in rep["text"].splitlines()[1]
    assert "audit: FAIL" in rep["text"].splitlines()[1] and "audit: pass" in rep["text"].splitlines()[0]
    assert all("must come from" in m for m in r["mixups"])              # roles cannot be swapped in the report
    # by form: any coefficients pass; an unbounded polynomial with moderate coefficients does not
    b, w = r["bounded"], r["wild"]
    assert b["passed"] and b["certified"] and b["H_upper"] <= 1.0 + 1e-9 and r["bounded_k"] == 5
    assert r["bounded_zero_start_diff"] < 5e-6 and r["bounded_label"] == r["labels"][0]
    assert w["magnitude"] == "failed" and w["violates_extended_box_bound"] and r["wild_label"] == r["labels"][1]
    assert "registered control" in r["labels"][0] and "not used for Delta(SR) - Delta(C2)" in r["labels"][1]


def test_finalist_certificate_is_computed_on_the_refitted_constants(suite):
    """User confirmation 2 and decision S4: the proven bound, the domain check and the tail check (under the
    criterion record of the baseline, bound to its hash) are re-run after the refit."""
    r = suite["recovery"]
    f = r["finalist"]
    assert r["out"]["finalist_certified"] is True and f["certified"] and f["reasons"] == []
    assert f["status"] == "certified" and f["require_certified"] is True and f["stage"] == "after refit"
    assert f["tail"]["evaluated"] and f["tail"]["passed"] and f["tail"]["radial"]["passed"] and f["tail"]["depth"]["passed"]
    assert f["tail"]["radial"]["rule"] == "loglog_slope_le_limit" and f["baseline_hash"] == "hash-of-steep"
    assert f["tail_criterion_hash"] == r["steep_hash"] and "superseded" in r["old_arg"]
    assert r["out"]["audit_certified"] is True and r["out"]["audit_verdict"]["certified"] is True
    from sfm.sr import translate as T
    e = T.from_dict(r["out"]["expr_refit_dict"])                        # the certificate is of THESE constants
    assert T.constants(e) == pytest.approx(r["c"]) and T.constants(e) != [0.5, 1.0, -1.0]
    assert "finalist_certified=True" in r["summary"] and "certified=True" in r["summary"]
    # the same refitted constants against a baseline that falls more slowly in depth: the correction's rise
    # (0.8 * 1.5 / 260 = 0.0046 / mm at z' = 263 mm, beyond z_90 at low momentum) beats -0.004 / mm -> no certificate
    s = r["slow"]
    assert not s["certified"] and s["audit_certified"] and "depth tail" in s["reasons"][0]
    assert s["depth"]["model_max_dz_per_mm"] == pytest.approx(-0.004 + r["c"][0] * r["c"][1] / 260.0, abs=2e-4)


def test_stripped_gauge_term_is_traceable_end_to_end_and_never_in_k(suite):
    """User decision S3: constants of stripped pure-x terms are neither trained nor counted; every stripped term is
    logged with its text, constants and stage - before the refit, before translation, after the refit."""
    r = suite["trace"]
    pre = r["pre"]
    assert pre["stage"] == "before refit" and pre["k"] == 3 and pre["k_as_written"] == 5 and pre["k_stripped"] == 2
    assert [t["term"] for t in pre["stripped_gauge_terms"]] == ["+ 3.0 * x", "+ 7.0"]
    assert [t["constants"] for t in pre["stripped_gauge_terms"]] == [[3.0], [7.0]]
    # the model: three trainable numbers, none of them 3.0 or 7.0
    assert r["n_param_elements"] == 3 and len(r["c"]) == 3 and 3.0 not in r["c"] and 7.0 not in r["c"]
    assert [m["stage"] for m in r["module_records"]] == ["before translation"] * 2
    out = r["out"]
    assert out["k"] == 3 and out["k_equation"] == 3 and out["k_as_written"] == 5 and out["warnings"] == []
    assert out["dropped_gauge_terms"] == ["+ 3.0 * x", "+ 7.0"]
    stages = [t["stage"] for t in out["stripped_gauge_terms"]]
    assert stages == ["before refit", "before refit", "before translation", "before translation"]
    assert [t["term"] for t in out["stripped_gauge_terms"]] == ["+ 3.0 * x", "+ 7.0"] * 2
    assert all(t["constants"] in ([3.0], [7.0]) for t in out["stripped_gauge_terms"])
    # the post-refit audit record carries the same trail, counts k without them, and finds nothing new to strip
    a = r["audit_refit"]
    assert a["stage"] == "after refit" and a["k"] == 3 and a["k_stripped"] == 0 and a["k_as_written"] == 3
    assert a["stripped_gauge_terms"] == out["stripped_gauge_terms"]
    assert " x" not in a["equation"] and "7.0" not in a["equation"] and a["equation"] == out["expr_refit"]
    assert len(r["refit_constants"]) == 3 and r["refit_constants"] == pytest.approx(r["c"])
    assert "stripped_gauge_terms=4" in r["summary"]
    assert any("differ from those of the pre-refit audit" in w for w in r["mismatch_warnings"])


def test_clip_counts_per_constant_and_per_step(suite):
    """User decision S5: the definitions, on synthetic gradient histories."""
    u = suite["clip_unit"]
    assert u["bouncing_on_cusp"]["definitions"] == ["clipped step", "cusp-stalled (run)", "driven by constant i",
                                                    "persistent (constant i)", "stalled (constant i)"]
    b = u["bouncing_on_cusp"]
    assert b["n_clipped"] == b["n_steps"] == b["per_step_len"] == 200 and b["driven_first"] == [[0], [0], [0]]
    c0, c1 = b["per_constant"]
    assert c0["n_driven"] == 200 and c0["longest_driven_run"] == 200 and c0["persistent"] and c0["stalled"]
    assert c0["net_over_path"] < 0.02 and c0["gradient_sign_flip_fraction_when_driven"] == 1.0
    assert c1["n_driven"] == 0 and not c1["persistent"] and not c1["stalled"]
    assert b["flags"] == {"persistent_clipping": True, "stalled": True, "cusp_stalled": True}
    assert u["bouncing_no_cusp"]["flags"] == {"persistent_clipping": True, "stalled": True, "cusp_stalled": False}
    m = u["moving"]                                                      # clipped all the way, but making progress
    assert m["flags"] == {"persistent_clipping": True, "stalled": False, "cusp_stalled": False}
    assert m["per_constant"][0]["net_over_path"] == pytest.approx(1.0)
    assert u["quiet"]["flags"] == {"persistent_clipping": False, "stalled": False, "cusp_stalled": False}
    assert u["quiet"]["n_clipped"] == 0
    s = u["short_burst"]                                                 # ten clipped steps are not persistent
    assert s["n_clipped"] == 10 and s["per_constant"][0]["longest_driven_run"] == 10 and not s["flags"]["persistent_clipping"]
    nf = u["nonfinite"]
    assert nf["n_no_backward"] == 1 and nf["n_nonfinite_gradient"] == 1 and nf["n_clipped"] == 198
    o = u["other_drives"]                                                # clipped, but not because of the cusp constant
    assert o["n_clipped"] == 200 and o["per_constant"][0]["n_driven"] == 0 and o["per_constant"][1]["n_driven"] == 200
    assert not o["flags"]["cusp_stalled"]
    assert suite["inside_sqa"] == [[False, True], [False, False, False], [True, False, False]]


def test_constant_sitting_on_a_cusp_is_flagged(suite):
    """A fit whose optimum lies on a kink of the gridded likelihood (truth 0.6 sqa(zt - z_k), z_k a grid row)."""
    c = suite["cusp"]
    t = c["tight_clip"]                                                  # clip norm inside the range of the gradients
    assert t["n_clipped"] == t["n_steps"] == t["per_step_clipped_sum"] == 160
    a_, c_ = t["per_constant"]
    assert c_["inside_sqa_argument"] and not a_["inside_sqa_argument"]
    assert c_["persistent"] and c_["stalled"] and c_["n_driven"] > 80 and c_["net_over_path"] < 0.25
    assert t["flags"]["cusp_stalled"] and t["cusp_stalled"] is True and "cusp-stalled" in t["warnings"][0]
    assert abs(t["c"][1] - t["zk"]) < 0.01 and "cusp_stalled=True" in t["summary"]
    # with the harness's default clip norm (5) the same run is never clipped: the clip count alone cannot flag it
    d = c["default_clip"]
    assert d["n_clipped"] == 0 and d["max_norm"] < 1.0 and d["cusp_stalled"] is False and abs(d["c"][1] - d["zk"]) < 0.01
    assert d["flags"] == {"persistent_clipping": False, "stalled": False, "cusp_stalled": False}
    # the supplementary, clip-independent indicator does
    assert d["hover_suspect"] is True and d["hover"]["nodes"][0]["min_distance_in_excursions"] <= 3.0
    assert "supplementary indicator" in d["warnings"][0] and t["hover_suspect"] is True
    n = c["no_cusp_in_grid"]                                             # sqa(zt + 2): no zero of the argument in the grid
    assert n["hover_suspect"] is False and n["cusp_stalled"] is False and n["n_clipped"] == 0 and n["warnings"] == []
    assert n["hover"]["nodes"][0]["min_abs_argument"] > 1.0
    r = suite["recovery_clip"]                                           # the smooth recovery run: nothing to flag
    assert r["n_steps"] == 300 and r["n_clipped"] == 0 and r["cusp_stalled"] is False and r["hover"]["nodes"] == []
    assert r["flags"] == {"persistent_clipping": False, "stalled": False, "cusp_stalled": False}


def test_tail_criterion_of_the_legacy_formula_preview(suite):
    """PREVIEW ONLY (decision S4, part d): the criterion is decided of record on the H200 for the refitted Track A
    finalist. Here the machinery is run on the legacy component formula, read-only, to see which branch to expect."""
    r = suite["legacy"]
    if r is None:
        pytest.skip("legacy formula checkpoint not present")
    rec = r["record"]
    assert "LEGACY" in rec["label"] and "not the refitted baseline of record" in rec["label"]
    assert len(rec["baseline_hash"]) == 64 and rec["covers_r_max"] and rec["momenta_GeV"] == [0.05, 0.1, 1.0, 5.0, 20.0, 30.0, 50.0]
    assert r["match_error"] < 1e-9 and "before the acceptance template" in rec["note"]   # exact continuation
    assert r["shape"] == [2, 128, 56] and 1400.0 < r["r_max"] < 1500.0 and r["r_range"][0] < 200.0 < r["r_range"][1]
    assert "does not continue the baseline" in r["wrong_model"]         # another model is not accepted as continuation
    ra = r["radial"]
    assert not ra["baseline_satisfies_limit"] and ra["baseline_monotone"] and -2.0 < ra["baseline_max"] < 0.0
    assert rec["criterion"]["radial"] == "monotone_and_no_flattening"   # the far halo: flatter than -2
    assert all(-2.0 < v < 0.0 for v in ra["baseline_max_per_momentum"])
    assert rec["criterion"]["depth"] == "dz_le_0_beyond_z90" and len(r["depth"]["z90_mm"]) == 7
    assert "3-D density" in r["text"]
    cert = r["certify"]
    falls, flat = cert["-0.3 * tanh(zt - 2.0) - 0.1 * l1p(rh)"], cert["-0.3 * tanh(zt - 2.0) + 0.1 * l1p(rh)"]
    assert falls["radial"] and falls["depth_no_flattening"] and not flat["radial"]


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
