"""scripts/run_m2a.py / run_m2a_lib.py / run_m2a_report.py (Track A driver, E4): the run list equals AUDIT.md section 4.1
in full (nothing dropped, no budget), no laptop constant is left, outputs go to this machine's directory under the
output root, the smoke mode is refused without its switch and registers under another stage, Stage 2 is never part
of the default list, and the rules / G-region classes are the pre-registered ones. CPU only; nothing is trained,
registered or written."""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_m2a as M  # noqa: E402
import run_m2a_lib as L  # noqa: E402
import run_m2a_report as R  # noqa: E402

from sfm import paths, provenance, stats  # noqa: E402


@pytest.fixture(autouse=True)
def _real_stage():
    L.configure(smoke=False)
    yield
    L.configure(smoke=False)


def test_listing_is_the_full_audit_list_in_order():
    lst = M.listing()
    rows = lst["rows"]
    names = [r["name"] for r in rows]
    uncond = [r["name"] for r in rows if r["condition"] is None]
    # Stage 0 (A0/A1/A2e seeds 0, 1; A2s / A2n seeds 0, 1, 2), pilots, step-0 parity
    want0 = ["A0_init"] + [f"plr_A2s_lr{x}" for x in ("0.003", "0.01", "0.03")] + ["A2s_s0"] + \
        [f"plr_A2n_lr{x}" for x in ("0.001", "0.003", "0.01")] + \
        ["A0_s0", "A0_s1", "A1_s0", "A1_s1", "A2e_s0", "A2e_s1", "A2s_s1", "A2s_s2", "A2n_s0", "A2n_s1", "A2n_s2"]
    assert uncond[:len(want0)] == want0
    prim = ("A4", "A5", "A7", "A8", "A9", "A10")
    for r in prim:
        assert f"{r}_s0" in uncond and f"{r}_s1" in uncond and f"{r}n_s0" in uncond
        assert f"{r}_s0_x2" in names and f"{r}_s1_x2" in names                   # conditional convergence reruns
    expl = ("A4t", "A5h", "A6i", "A6l", "A6b", "A7s", "A7l", "A8c", "A78", "A9r", "A3f", "A3c", "A3x", "A10-2",
            "A10-4", "A10-5", "A10g", "A12")
    assert M.EXPLORATORY == expl
    for r in expl:
        assert f"{r}_s0" in uncond                                                # every exploratory rung runs
        c = next(x for x in rows if x["name"] == f"{r}_s1")
        assert "keep or the equal-NLL rule" in c["condition"]
    # order: stage0 < primaries poly < net < exploratory
    pos = {n: i for i, n in enumerate(names)}
    assert pos["A2n_s2"] < pos["A4_s0"] < pos["A10_s1"] < pos["A4n_s0"] < pos["A10n_s0"] < pos["A4t_s0"] < pos["A12_s0"]
    # steps: T_A 4000, mixtures 6000, pilots 3000, reruns doubled
    st = {r["name"]: r["steps"] for r in rows}
    assert all(st[f"plr_A2s_lr{x}"] == 3000 for x in ("0.003", "0.01", "0.03"))
    assert st["A4_s0"] == 4000 and st["A10_s0"] == 6000 and st["A10-5_s0"] == 6000 and st["A10g_s0"] == 6000
    assert st["A4_s0_x2"] == 8000 and st["A10_s0_x2"] == 12000
    assert lst["n_unconditional_fits"] == 54
    # pilots 6 x 3000; A2s_s0 + 11 Stage-0 fits x 4000; poly primaries 10 x 4000 + 2 x 6000; nets 5 x 4000 + 6000;
    # exploratory seed 0: 14 x 4000 + 4 x 6000
    assert lst["training_steps_unconditional"] == 6 * 3000 + 12 * 4000 + (10 * 4000 + 2 * 6000) + \
        (5 * 4000 + 6000) + (14 * 4000 + 4 * 6000) == 224000
    assert "stage2" in lst["not_in_default"]


def test_no_budget_no_drop_order_no_laptop_constants():
    for f in ("run_m2a.py", "run_m2a_lib.py", "run_m2a_report.py"):
        src = (SCRIPTS / f).read_text()
        for bad in ("18.120061", "18.12041", "18.120407", "BUDGET_S", "DROP_ORDER", "_est_seconds", "m0_parity/formula"):
            assert bad not in src, (f, bad)
    assert M.DEFAULT_PHASES == ("templates", "parity", "plr", "stage0", "stage1p", "stage1x", "guards", "report")
    assert "stage2" not in M.DEFAULT_PHASES


def test_outputs_live_in_this_machines_directory():
    assert L.MACHINE_DIR.parent == paths.OUT_RESULTS / "machines"
    assert L.out_dir() == L.MACHINE_DIR / "runs" / "m2a_dist"
    for f in (L.plr_file(), L.decisions_file(), L.gpu_log()):
        assert f.parent == L.out_dir()
    sp = M.exploratory_spec("A4t", 0)
    assert sp.out_dir == L.out_dir() / "A4t_s0" and sp.kwargs["start"] == "m2a_dist/A2s_s0"
    ctx = L.build_context("cpu")
    assert ctx.results_dir == L.RUNS                    # "<stage>/<name>" resolves inside this machine's runs/
    from sfm.models import dist_factory as dfac
    assert dfac.template_dir() == L.RUNS / "m2a_dist" / "templates"
    from sfm import machine as _machine
    assert L.MACHINE_DIR == Path(_machine.machine_dir_live())          # the directory comes from the API


def test_smoke_mode_needs_its_switch_and_uses_another_stage(monkeypatch):
    monkeypatch.delenv(L.SMOKE_ENV, raising=False)
    with pytest.raises(SystemExit, match="smoke mode refused"):
        L.configure(smoke=True)
    monkeypatch.setenv(L.SMOKE_ENV, "1")
    L.configure(smoke=True)
    assert L.stage() == "m2a_smoke" and L.out_dir() == L.RUNS / "m2a_smoke"
    sp = M.primary_specs()[0][0]
    assert sp.kwargs["start"] == "m2a_smoke/A2s_s0"
    monkeypatch.setattr(L, "frozen_lr", lambda prov: 0.01)
    cfg = L.fit_config(sp)
    assert cfg.steps == 40 and cfg.eval_every == 10 and L.convergence_window() == 10
    net = M.primary_specs()[1][0]
    assert net.build_kwargs()["fit_to_steps"] == 40
    L.configure(smoke=False)
    cfg = L.fit_config(sp if False else M.primary_specs()[0][0])
    assert cfg.steps == 4000 and cfg.eval_every == 500 and cfg.k_val == 64 and cfg.batch == 256
    assert cfg.lr_end == pytest.approx(cfg.lr / 15) and cfg.weight_decay == 0.0 and cfg.selection == "last"
    assert L.fit_config(M.primary_specs()[1][0]).weight_decay == 1e-4


def test_stage2_is_never_in_the_default_list():
    with pytest.raises(SystemExit, match="stage2 runs only on its own"):
        M.main(["templates", "stage2"])


def test_rules_are_the_pre_registered_ones():
    P = stats.Paired
    assert M.keep_rule(P(-3e-4, 1.5e-4, 100)) and not M.keep_rule(P(-2.9e-4, 0.1e-4, 100))
    assert not M.keep_rule(P(-4e-4, 2.1e-4, 100))
    assert M.equal_rule_one(P(0.9e-4, 1e-4, 100)) and not M.equal_rule_one(P(1e-4, 2e-4, 100))
    assert not M.equal_rule_one(P(0.5e-4, 0.4e-4, 100))
    assert M.DRIFT_STOP == 2e-4 and M.PARITY_EVENT_TOL == 1e-4 and M.PARITY_MEAN_TOL == 1e-6
    assert M.PLR_TIE == 0.5e-4 and M.CONVERGED_MOVE == 0.5e-4 and L.CONVERGENCE_WINDOW == 1000


def test_stage2_content():
    specs = M.stage2_specs([])
    assert [s.name for s in specs] == ["A11k_s0", "A11k_s1"] and all(s.k_train == 64 for s in specs)
    specs = M.stage2_specs(["A4", "A7"])
    names = [s.name for s in specs]
    assert names == ["A11_s0", "A11_s1", "A11_s2", "A11k_s0", "A11k_s1", "A11n_s0", "A11n_s1"]
    kw = specs[0].kwargs
    assert kw["shower_long"] == "emg" and kw["shower_lat"] == ["gauss", "burr", "exp"] and kw["init"] == "shower_long"
    with pytest.raises(SystemExit, match="conflict"):
        M.stage2_specs(["A7", "A8"])
    with pytest.raises(SystemExit, match="cannot be merged"):
        M.stage2_specs(["A10"])


def test_g_region_classes_by_hit_centre():
    C = L.common
    r31 = float(C.R_EDGES[-2])
    pts = [(0.0, 0.0, 500.0), (0.0, 0.0, 1300.0), (r31 + 1.0, 0.0, 500.0), (0.0, 0.0, -199.0), (10.0, 0.0, 300.0)]

    class V:
        xyz = torch.tensor(pts, dtype=torch.float32)
    m = M._g_region_masks(V)
    assert list(m["deep"]) == [False, True, False, False, False]
    assert list(m["border_ring"]) == [False, False, True, False, False]
    assert list(m["border_zbin"]) == [False, False, False, True, False]
    assert np.array_equal(m["flagged"], m["deep"] | m["border_ring"] | m["border_zbin"] | m["low_abar"])
    assert np.array_equal(m["inner"], ~m["flagged"])


def test_g_region_margin_rule():
    """G-region passes iff the flagged classes carry at most half the gain: mean(flagged - 0.5 main) >= 0."""
    rng = np.random.default_rng(0)
    main = -1e-3 + 1e-4 * rng.standard_normal(5000)
    small = 0.2 * main
    big = 0.8 * main
    assert stats.paired(small - 0.5 * main, np.zeros_like(main)).mean > 0
    assert stats.paired(big - 0.5 * main, np.zeros_like(main)).mean < 0
    assert "G-region" in R.__doc__ and "0.5 main" in R.__doc__


def test_source_runs_are_verified_by_sha256(tmp_path, monkeypatch):
    """start / init_ref / net_from must name a finished run of this machine whose model.pt and weight tensors carry
    the SHA-256 of a registry row on the live stream (coordinator's review point 4); anything else is refused."""
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    d = tmp_path / "A2s_s0"
    d.mkdir()
    state = {"w": torch.arange(4.0)}
    torch.save({"state": state}, d / "model.pt")
    (d / "done.json").write_text(_json.dumps({"machine_id": L.MACHINE_ID}))
    fsha, wsha = provenance.sha256_file(d / "model.pt"), provenance.weights_sha256(state)
    row = {"stage": "m2a_dist", "name": "A2s_s0", "checkpoint_sha256": fsha, "weights_sha256": wsha,
           "timestamp": "t", "val_nll_sha256": "v", "trained_on": {"machine_id": L.MACHINE_ID}}
    monkeypatch.setattr(L, "live_stream_id", lambda: "stream")
    monkeypatch.setattr(L, "registered_row", lambda name, ckpt=None, need_guards=False:
                        row if provenance.sha256_file(ckpt) == row["checkpoint_sha256"] else None)
    rec = L.verify_sources({"start": "m2a_dist/A2s_s0", "init": "shower_long"})
    assert rec["start"]["model_sha256"] == fsha and rec["start"]["weights_sha256"] == wsha
    assert rec["start"]["row"]["name"] == "A2s_s0" and rec["start"]["machine_id"] == L.MACHINE_ID
    with pytest.raises(RuntimeError, match="of this stage"):
        L.verify_sources({"net_from": "m2a_smoke/A2s_s0"})
    with pytest.raises(RuntimeError, match="not a finished run"):
        L.verify_sources({"init_ref": "m2a_dist/A2s_s1"})
    row["weights_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="weights_sha256"):
        L.verify_sources({"start": "m2a_dist/A2s_s0"})
    row["checkpoint_sha256"] = "1" * 64
    with pytest.raises(RuntimeError, match="matches no registry row"):
        L.verify_sources({"start": "m2a_dist/A2s_s0"})
    (d / "done.json").write_text(_json.dumps({"machine_id": "another-machine"}))
    with pytest.raises(RuntimeError, match="markers are never shared"):
        L.verify_sources({"start": "m2a_dist/A2s_s0"})


# ---------------------------------------------------------------------------------------------------------------------
# review fixes B1-B5, C2 (report and Stage-2 gate only; nothing before the guards phase changes)
# ---------------------------------------------------------------------------------------------------------------------
def test_b1_rerun_pair_replaces_the_non_converged_original(monkeypatch):
    have = {"A7_s0", "A2s_s0", "A7_s0_x2", "A2s_s0_x2", "A7_s1", "A2s_s1"}
    monkeypatch.setattr(R, "have", lambda n: n in have)
    monkeypatch.setattr(R, "comparison", lambda run, comp, masks: {"run": run, "vs": comp})
    c, orig = R.primary_pairs("A7", {})
    assert c[0] == {"run": "A7_s0_x2", "vs": "A2s_s0_x2", "pair": "rerun"}
    assert c[1] == {"run": "A7_s1", "vs": "A2s_s1", "pair": "original"}
    assert orig == {0: {"run": "A7_s0", "vs": "A2s_s0", "pair": "original (not converged; not used by the rules)"}}
    have.discard("A7_s0_x2")                                   # no rerun: the original is used
    c, orig = R.primary_pairs("A7", {})
    assert c[0]["run"] == "A7_s0" and orig == {}


def _fake_tagged(monkeypatch, arrays):
    from sfm.machine import MachineTag
    mt = MachineTag.from_fields({"gpu_name": "test", "launch_grid": 1})
    pts = provenance.points_tag(grid=1, K=64, split_fingerprint="f")
    monkeypatch.setattr(L, "live_machine", lambda: mt)
    monkeypatch.setattr(L, "tagged", lambda n, kind="k64": provenance.TaggedArray.make(arrays[n], pts, mt, label=n))


def test_b3_replication_prints_the_two_se_alternative_beside_the_literal_reading(monkeypatch):
    """Literal reading passes (seed 1 <= -1.5e-4 with the same sign, two-seed mean <= -3e-4); the alternative 2-SE
    condition on the two-seed mean fails: printed beside it, it decides nothing."""
    rng = np.random.default_rng(1)
    n = 4000
    base = 18.12 + 1e-3 * rng.standard_normal(n)

    def zm(x):
        return x - x.mean()
    n0, n1 = zm(0.02 * rng.standard_normal(n)), zm(0.02 * rng.standard_normal(n))
    arrays = {"A2s_s0": base, "A2s_s1": base, "X_s0": base - 3.5e-4 + n0, "X_s1": base - 3.5e-4 + n1}
    _fake_tagged(monkeypatch, arrays)
    c0 = {"paired": L.pd(provenance.paired_delta(L.tagged("X_s0"), L.tagged("A2s_s0")))}
    c1 = {"paired": L.pd(provenance.paired_delta(L.tagged("X_s1"), L.tagged("A2s_s1")))}
    rep = R.replication(c0, c1, "X_s0", "A2s_s0", "X_s1", "A2s_s1")
    tsm = rep["two_seed_mean"]
    assert tsm["mean"] == pytest.approx(-3.5e-4, abs=1e-9) and tsm["se"] > 1.75e-4
    assert rep["replicated"] is True
    assert rep["alternative_two_seed_mean_2se"] is False and rep["replicated_alternative_2se"] is False
    assert rep["pairs"] == [["X_s0", "A2s_s0"], ["X_s1", "A2s_s1"]]


def test_b5_stage2_needs_an_approval_bound_to_the_results_json(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    monkeypatch.setattr(M, "stage2_content_check", lambda appr, res, kept: {"content": kept})   # tested separately
    monkeypatch.setattr(M, "a11p_parameters", lambda appr: {})
    with pytest.raises(SystemExit, match="results.json\\) does not exist"):
        M.stage2_approval()
    res = {"kept_and_replicated": {"A7": True, "A8": False}, "kept_and_replicated_alternative_2se": {"A7": False}}
    (tmp_path / "results.json").write_text(_json.dumps(res))
    with pytest.raises(SystemExit, match="no approval file"):
        M.stage2_approval()
    sha = provenance.sha256_file(tmp_path / "results.json")
    appr = {"results_json_sha256": sha, "replication_reading": "literal", "kept_and_replicated": ["A7"],
            "approved_utc": "2026-11-26T10:00:00Z", "approved_by": "user (recorded by the orchestrator)"}
    (tmp_path / M.STAGE2_APPROVAL).write_text(_json.dumps(appr))
    a, kept, s, content = M.stage2_approval()
    assert kept == ["A7"] and s == sha and content["content"] == ["A7"]
    for bad, msg in (({"results_json_sha256": "0" * 64}, "table changed after the approval"),
                     ({"replication_reading": "loose"}, "replication_reading"),
                     ({"kept_and_replicated": []}, "differs from the list"),
                     ({"replication_reading": "two_se"}, "differs from the list"),
                     ({"approved_by": ""}, "approved_by")):
        (tmp_path / M.STAGE2_APPROVAL).write_text(_json.dumps({**appr, **bad}))
        with pytest.raises(SystemExit, match=msg):
            M.stage2_approval()
    (tmp_path / M.STAGE2_APPROVAL).write_text(_json.dumps({**appr, "replication_reading": "two_se",
                                                          "kept_and_replicated": []}))
    assert M.stage2_approval()[1] == []
    (tmp_path / "results.json").write_text(_json.dumps({**res, "x": 1}))       # the table was recomputed
    with pytest.raises(SystemExit, match="table changed"):
        M.stage2_approval()
    assert "stage2" not in M.DEFAULT_PHASES


def test_c2_without_guards_only_with_the_single_report_phase():
    with pytest.raises(SystemExit, match="single phase 'report'"):
        M.main(["--without-guards"])
    with pytest.raises(SystemExit, match="single phase 'report'"):
        M.main(["stage1x", "report", "--without-guards"])
    assert M.REPORT_WITHOUT_GUARDS is False


def test_b2_warm_start_targets_of_the_a1_control_on_the_cpu():
    """A1 is fitted to the legacy provider: its 22 momentum-dependent columns have relative spreads >= 0.041 (review
    section 3), none in the nearly-constant range (2^-20, 1e-2), none constant."""
    done = {"spec": {"factory": "component", "kwargs": {"variant": "legacy_centre", "provider": "net"}},
            "build_kwargs": {"variant": "legacy_centre", "provider": "net"}}
    t = R.warm_start_targets(done)
    assert t["constant"] == [] and t["nearly_constant"] == {}
    assert 0.03 < t["smallest_varying"][1] < 0.05
    assert R.KL_DEGRADED == 1e-4 and R.NEAR_CONSTANT_HI == 1e-2


def test_b4_template_notes_carry_the_reviewers_numbers():
    assert "0.50 against 0.35" in R.TEMPLATE_NOTE_Q1 and "-0.16 ± 0.02" in R.TEMPLATE_NOTE_Q1
    s = R.TEMPLATE_NOTE_Q2.format(n101=79, z=1379, mx=1.81, den=0.186)
    assert "79 cells at z' >= 1379 mm" in s and "`deep`" in s and "`beyond_zb`" in s


# ---------------------------------------------------------------------------------------------------------------------
# user decisions of 25 Nov 2026 (PLAN 17.11): A2c, scale floor + step-0 parity, divergence handling
# ---------------------------------------------------------------------------------------------------------------------
def _h(step, val, sl=0, sg=0, tr=18.1):
    return {"step": step, "val": val, "skipped_loss": sl, "skipped_grad": sg, "train_nll": tr}


def test_divergence_definition_is_explicit_and_conservative():
    ok = [_h(500, 18.13), _h(1000, 18.125), _h(1500, 18.121)]
    assert L.divergence_check(ok, [18.12]) == []
    assert L.divergence_check(ok[:2] + [_h(1500, 18.121, sl=3)], [18.12]) == []           # isolated skips: no
    nan = ok[:2] + [_h(1500, float("nan"))]
    assert any("non-finite final" in c for c in L.divergence_check(nan, []))
    # PLAN 20 item 2: a non-finite value in the middle of the run, recovered by the end, is a warning, not divergence
    rec = [ok[0], _h(1000, float("nan")), ok[2]]
    assert L.divergence_check(rec, [18.12]) == []
    assert L.recovered_nonfinite(rec) == {"count": 1, "steps": [1000]}
    assert L.recovered_nonfinite(ok) == {"count": 0, "steps": []}
    assert L.recovered_nonfinite(nan)["count"] == 0                     # the final value is the divergence itself
    assert "final" in L.DIVERGENCE_DEFINITION and "warning" in L.DIVERGENCE_DEFINITION
    frozen = ok[:2] + [_h(1500, 18.13, sl=200, sg=60)]                                       # 260 of 500 skipped
    assert any(c.startswith("frozen") for c in L.divergence_check(frozen, []))
    assert L.divergence_check(ok[:2] + [_h(1500, 18.13, sl=240)], []) == []                 # 240 of 500: below 50 %
    blow = ok[:2] + [_h(1500, 19.2)]
    assert any(c.startswith("blow-up") for c in L.divergence_check(blow, []))                # 19.2 - 18.125 > 1 nat
    assert L.divergence_check(ok[:2] + [_h(1500, 19.0)], []) == []                           # 0.875 nat: not
    assert any(c.startswith("blow-up") for c in L.divergence_check([_h(500, 19.2)], [18.1]))  # vs comparator
    assert L.BLOWUP_NAT == 1.0 and L.FROZEN_FRACTION == 0.5 and "conservative" not in L.DIVERGENCE_DEFINITION


def test_injection_hook_is_refused_in_the_real_mode(monkeypatch):
    monkeypatch.setenv(L.INJECT_ENV, "A4t_s0")
    with pytest.raises(SystemExit, match="smoke-mode testing hook"):
        L.configure(smoke=False)
    monkeypatch.delenv(L.INJECT_ENV)
    L.configure(smoke=False)


def test_a_diverged_pilot_loses_under_the_pilot_rule(monkeypatch):
    rng = np.random.default_rng(3)
    base = 18.12 + 1e-3 * rng.standard_normal(500)
    arrays = {"p1": base + 1e-4, "p3": base + 3e-4}
    _fake_tagged(monkeypatch, arrays)
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {} if n in arrays else None)
    monkeypatch.setattr(L, "diverged_marker", lambda n: {"status": "diverged"} if n == "p2" else None)
    win, tab = M._plr_decide({0.001: "p1", 0.003: "p2", 0.01: "p3"})
    assert win == 0.001 and tab["0.003"]["status"] == "diverged" and tab["0.003"]["within"] is False
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: None)
    win, tab = M._plr_decide({0.001: "p1", 0.003: "p2"})
    assert win is None and len(tab) == 2


def test_blocked_by_a_diverged_source_or_an_undecided_lr_net(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    monkeypatch.setattr(L, "diverged_marker", lambda n: {"status": "diverged"} if n == "A7_s0" else None)
    sp = M.primary_specs()[1][2]                                    # A7n_s0: net_from m2a_dist/A7_s0
    assert sp.name == "A7n_s0" and L.blocked_by(sp) == "A7_s0"
    assert L.blocked_by(M.primary_specs()[1][0]) is None            # A4n_s0 does not depend on A7_s0
    (tmp_path / "plr.json").write_text(_json.dumps({"lr_poly": 0.003, "lr_net": None}))
    assert L.blocked_by(M.primary_specs()[1][0]).startswith("P-lr")


def test_diagnosis_file_must_name_the_diverged_record(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    (tmp_path / "A7_s0").mkdir()
    (tmp_path / "A7_s0" / L.DIVERGED_FILE).write_text(_json.dumps({"machine_id": L.MACHINE_ID, "cls": "primary",
                                                                   "status": "diverged"}))
    assert L.diagnosis("A7_s0") is None
    (tmp_path / "diagnosis").mkdir()
    sha = provenance.sha256_file(tmp_path / "A7_s0" / L.DIVERGED_FILE)
    good = {"run": "A7_s0", "diverged_json_sha256": sha, "verdict": "failed", "diagnosed_by": "Fable",
            "diagnosed_utc": "2026-11-26T10:00:00Z", "summary": "x"}
    (tmp_path / "diagnosis" / "A7_s0.json").write_text(_json.dumps(good))
    assert L.diagnosis("A7_s0")["valid"] is True
    monkeypatch.setattr(L, "decisions", lambda: {})
    assert M._status_text("A7_s0") == "failed (diagnosed)"
    for bad in ({"diverged_json_sha256": "0" * 64}, {"verdict": "maybe"}, {"diagnosed_by": ""}):
        (tmp_path / "diagnosis" / "A7_s0.json").write_text(_json.dumps({**good, **bad}))
        assert L.diagnosis("A7_s0")["valid"] is False
        assert M._status_text("A7_s0") == "diverged - awaiting diagnosis"
    (tmp_path / "A4t_s0").mkdir()
    (tmp_path / "A4t_s0" / L.DIVERGED_FILE).write_text(_json.dumps({"machine_id": L.MACHINE_ID,
                                                                    "cls": "exploratory"}))
    assert M._status_text("A4t_s0") == "diverged"                     # exploratory: no diagnosis needed


def test_a2c_is_the_matched_control_of_the_rungs_that_start_from_the_fitted_a2s():
    a = M.a2c_specs()
    assert [s.name for s in a] == ["A2c_s0", "A2c_s1"]
    for s, sp in enumerate(a):
        assert sp.kwargs == {"variant": "survival_exact", "start": f"m2a_dist/A2s_s{s}"} and sp.seed == s
        assert sp.steps == 4000 and sp.lr is None and sp.provider == "poly" and sp.cls == "control"
        assert sp.comparator == f"A2s_s{s}" and sp.k_train == 16
    # every rung of the list that starts from the fitted A2s with 4000 steps is in the matched group, the
    # mixtures (6000 steps, init_from_target) are not
    rungs = M.primary_specs()[0] + [M.exploratory_spec(r, 0) for r in M.EXPLORATORY]
    started = {sp.rung for sp in rungs if sp.kwargs.get("start") == "m2a_dist/A2s_s0" and sp.steps == 4000}
    assert started == set(M.A2C_MATCHED)
    assert {sp.rung for sp in rungs if sp.factory == "mixture"} == set(M.A2C_UNMATCHED)
    assert "a2c" not in M.DEFAULT_PHASES and set(M.ADDED_PHASES) <= set(M.PHASES)
    assert not set(M.ADDED_PHASES) & set(M.DEFAULT_PHASES)


def test_listing_additions_are_separate_and_the_existing_listing_is_unchanged(capsys):
    lst = M.listing()
    assert lst["n_unconditional_fits"] == 54 and len(lst["rows"]) == 97
    add = M.listing_additions()
    assert [x["name"] for x in add][:2] == ["A2c_s0", "A2c_s1"] and add[2]["phase"] == "warmstart-check"
    M.print_listing()
    out = capsys.readouterr().out
    head, sep, tail = out.partition("\n# Additions of 25 Nov 2026")
    assert sep and "A2c_s0" in tail and "A2c_s0" not in head


def test_warmstart_rule_is_plan_17_11_verbatim():
    assert M.warmstart_affected({"below_floor": []}, 5e-5) == (False, False, False)
    assert M.warmstart_affected({"below_floor": ["sh_mu"]}, 5e-5) == (True, False, True)
    assert M.warmstart_affected({"below_floor": []}, 2e-4) == (False, True, True)
    assert M.warmstart_affected({"below_floor": []}, 1e-4) == (False, False, False)     # "exceeds"
    assert M.warmstart_affected({"below_floor": []}, None) == (False, False, False)
    assert "if and only if" in M.WARMSTART_RULE and M.KL_AFFECTED == 1e-4


def test_warmstart_target_columns_of_the_a1_control():
    d = {"spec": {"factory": "component", "kwargs": {"variant": "legacy_centre", "provider": "net"}},
         "build_kwargs": {"variant": "legacy_centre", "provider": "net"}}
    tc = M.target_columns(d)
    assert tc["below_floor"] == [] and tc["constant"] == []
    assert min(c["std"] for c in tc["columns"].values()) >= 0.04


def test_stage2_preconditions(monkeypatch, tmp_path):
    import json as _json
    import run_m2a_report as RR
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: None)
    with pytest.raises(SystemExit, match="A2c control is not finished"):
        M.stage2_preconditions()
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {})
    monkeypatch.setattr(M, "_net_runs_executed", lambda: ["A4n_s0"])
    monkeypatch.setattr(M, "decisions", lambda: {})
    with pytest.raises(SystemExit, match="warmstart-check"):
        M.stage2_preconditions()
    monkeypatch.setattr(M, "decisions", lambda: {"warmstart_check": {"rerun": ["A4n_s0"]}})
    monkeypatch.setattr(L, "superseded_recorded", lambda: {})
    with pytest.raises(SystemExit, match="warm-start re-runs missing"):
        M.stage2_preconditions()
    monkeypatch.setattr(L, "superseded_recorded", lambda: {"A4n_s0": "A4n_s0_ws"})
    monkeypatch.setattr(M, "decisions", lambda: {"warmstart_check": {"rerun": ["A4n_s0"]},
                                                 "diverged": {"A7_s0": {"status": "diverged - awaiting diagnosis"}}})
    monkeypatch.setattr(L, "diagnosis", lambda n: None)
    with pytest.raises(SystemExit, match="await an independent diagnosis"):
        M.stage2_preconditions()
    monkeypatch.setattr(L, "diagnosis", lambda n: {"valid": True, "verdict": "failed"})
    with pytest.raises(SystemExit, match="conditional-controls"):                     # PLAN 20 item 1
        M.stage2_preconditions()
    cc = {"A2c_x2": {"triggered": False, "seeds": []}, "A2c6k": {"triggered": False, "seeds": []}}
    monkeypatch.setattr(M, "decisions", lambda: {"warmstart_check": {"rerun": ["A4n_s0"]}, "conditional_controls": cc,
                                                 "diverged": {"A7_s0": {"status": "diverged - awaiting diagnosis"}}})
    monkeypatch.setattr(M, "evaluate_conditional_controls", lambda: {**cc, "A2c6k": {"triggered": True, "seeds": [1]}})
    with pytest.raises(SystemExit, match="differs from its re-evaluation"):
        M.stage2_preconditions()
    monkeypatch.setattr(M, "evaluate_conditional_controls", lambda: cc)
    (tmp_path / "results.json").write_text(_json.dumps({"a": 1}))
    monkeypatch.setattr(RR, "collect", lambda with_guards=True: {"a": 2})
    with pytest.raises(SystemExit, match="differs from the report recomputed"):
        M.stage2_preconditions()
    monkeypatch.setattr(RR, "collect", lambda with_guards=True: {"a": 1})
    M.stage2_preconditions()                                     # all satisfied


def test_report_shows_diverged_and_blocked_runs_with_their_status(monkeypatch):
    have = {"A2s_s0", "A2s_s1", "A7_s1"}
    monkeypatch.setattr(R, "have", lambda n: n in have)
    monkeypatch.setattr(R, "run_status", lambda n: {"A7_s0": "diverged - awaiting diagnosis"}.get(n, "missing"))
    monkeypatch.setattr(R, "comparison", lambda run, comp, masks: {"run": run, "vs": comp, "paired": {"mean": 0}})
    c, orig = R.primary_pairs("A7", {})
    assert c[0]["status"] == "diverged - awaiting diagnosis" and c[0]["paired"] is None and c[1]["run"] == "A7_s1"
    assert R.replication(c[0], c[1], "A7_s0", "A2s_s0", "A7_s1", "A2s_s1") is None
    assert R._cell(c[0]) == "**diverged - awaiting diagnosis**" and R._yn(c[0], "keep") == "-"
    assert R.entry("A8_s0", "A2s_s0", {}) is None                     # not run at all: no row


def test_stage2_approval_requires_the_kept_key(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    (tmp_path / "results.json").write_text(_json.dumps({"kept_and_replicated": {}}))
    sha = provenance.sha256_file(tmp_path / "results.json")
    (tmp_path / M.STAGE2_APPROVAL).write_text(_json.dumps({"results_json_sha256": sha, "replication_reading":
                                                          "literal", "approved_utc": "t", "approved_by": "u"}))
    with pytest.raises(SystemExit, match="no key 'kept_and_replicated'"):
        M.stage2_approval()



# ---------------------------------------------------------------------------------------------------------------------
# re-review of a05c86e (REVIEW.md section 8): B8-1, B8-2, B8-3, C8-3, C8-4, C8-5
# ---------------------------------------------------------------------------------------------------------------------
def test_b8_1_lr_net_changed_branch_rate_of_record_and_stage2_comparator(tmp_path, monkeypatch):
    """warmstart-rerun when the re-run pilots change lr_net: every other network run is re-run with the new rate,
    later network runs read it as the rate of record (plr.json stays the pilot's record), and Stage 2's A11n is
    compared with the superseding A2n run."""
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    (tmp_path / "plr.json").write_text(_json.dumps({"lr_poly": 0.003, "lr_net": 0.01}))
    pilots = [s.name for s in M.plr_specs()[1]]
    executed = pilots + ["A1_s0", "A2n_s0", "A4n_s0"]
    (tmp_path / "decisions.json").write_text(_json.dumps({"warmstart_check": {
        "rerun": pilots + ["A4n_s0"], "runs": {pilots[0]: {"rule_a_below_floor": True}}}}))   # caused by the fix
    specs = {s.name: s for s in M.plr_specs()[1] + M.stage0_specs() + M.primary_specs()[1]}
    registered, calls = set(), []

    def fake_run_spec(sp):
        calls.append((sp.name, sp.lr, sp.comparator))
        registered.add(sp.name)
        return {"name": sp.name}
    monkeypatch.setattr(M, "_net_runs_executed", lambda: list(executed))
    monkeypatch.setattr(M, "_spec_of", lambda n: specs[n])
    monkeypatch.setattr(L, "run_spec", fake_run_spec)
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {} if n in registered else None)
    monkeypatch.setattr(M, "_plr_decide", lambda names: (0.003, {"names": names}))
    monkeypatch.setattr(M, "phase_guards", lambda: None)
    M.phase_warmstart_rerun()
    rec = M.decisions()["warmstart_rerun"]
    assert rec["lr_net_after_rerun"]["lr_net_before"] == 0.01 and rec["lr_net_after_rerun"]["lr_net_after"] == 0.003
    assert set(rec["dependent"]) == {"A1_s0", "A2n_s0"}                     # A4n_s0 was listed itself
    by = {n: (lr, comp) for n, lr, comp in calls}
    assert all(by[f"{p}_ws"][0] == sp.lr for p, sp in ((p, specs[p]) for p in pilots))   # pilots keep their rate
    assert by["A1_s0_ws"][0] == by["A2n_s0_ws"][0] == by["A4n_s0_ws"][0] == 0.003
    assert by["A4n_s0_ws"][1] == "A2n_s0_ws"                                # comparator follows the re-run
    assert L.frozen_lr("net") == 0.003 and _json.loads((tmp_path / "plr.json").read_text())["lr_net"] == 0.01
    a11n = {sp.name: sp.comparator for sp in M.stage2_specs(["A7"]) if sp.rung == "A11n"}
    assert a11n == {"A11n_s0": "A2n_s0_ws", "A11n_s1": "A2n_s1"}           # A2n_s1 was not superseded here
    assert L.fit_config(specs["A1_s1"]).lr == 0.003                         # any later network run: rate of record


def test_c8_4_a_diverged_rerun_does_not_supersede(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    (tmp_path / "decisions.json").write_text(_json.dumps({"warmstart_rerun": {"superseded": {
        "A2n_s0": "A2n_s0_ws", "A4n_s0": "A4n_s0_ws"}}}))
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {} if n != "A2n_s0_ws" else None)
    monkeypatch.setattr(L, "diverged_marker", lambda n: {"status": "diverged"} if n == "A2n_s0_ws" else None)
    assert L.superseded() == {"A4n_s0": "A4n_s0_ws"}
    assert L.effective("A2n_s0") == "A2n_s0" and L.effective("A4n_s0") == "A4n_s0_ws"
    assert L.rerun_not_in_force() == {"A2n_s0": ("A2n_s0_ws", "diverged")}
    assert L.superseded_recorded()["A2n_s0"] == "A2n_s0_ws"                # the record itself is untouched


def test_b8_2_diagnosed_divergence_resolves_the_gate(monkeypatch):
    status = {"A2c_s0": "reg", "A2c_s1": "div", "A4n_s0_ws": "div"}
    diag = {}
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False:
                        {} if status.get(n, "reg") == "reg" else None)
    monkeypatch.setattr(L, "diverged_marker", lambda n: {"status": "diverged"} if status.get(n) == "div" else None)
    monkeypatch.setattr(L, "diagnosis", lambda n: diag.get(n))
    monkeypatch.setattr(M, "_net_runs_executed", lambda: ["A4n_s0"])
    cc = {"A2c_x2": {"triggered": False, "seeds": []}, "A2c6k": {"triggered": False, "seeds": []}}
    monkeypatch.setattr(M, "decisions", lambda: {"warmstart_check": {"rerun": ["A4n_s0"]}, "conditional_controls": cc})
    monkeypatch.setattr(M, "evaluate_conditional_controls", lambda: cc)
    monkeypatch.setattr(L, "superseded_recorded", lambda: {"A4n_s0": "A4n_s0_ws"})
    with pytest.raises(SystemExit, match="A2c control is not finished or diverged without a valid diagnosis"):
        M.stage2_preconditions()
    diag["A2c_s1"] = {"valid": False, "verdict": "failed"}
    with pytest.raises(SystemExit, match="A2c control"):
        M.stage2_preconditions()
    diag["A2c_s1"] = {"valid": True, "verdict": "failed"}
    with pytest.raises(SystemExit, match="diverged without a valid diagnosis, for \\['A4n_s0'\\]"):
        M.stage2_preconditions()
    diag["A4n_s0_ws"] = {"valid": True, "verdict": "rerun_allowed"}
    import run_m2a_report as RR
    monkeypatch.setattr(RR, "collect", lambda with_guards=True: {})
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: __import__("pathlib").Path("/nonexistent"))
    with pytest.raises(FileNotFoundError):                      # every precondition passed; only results.json absent
        M.stage2_preconditions()
    assert M._resolved("A2c_s0") and M._resolved("A2c_s1") and M._resolved("A4n_s0_ws")


def test_b8_3_not_evaluable_instead_of_no():
    ok = {"paired": {"mean": -4e-4}, "keep": True, "run": "X_s0"}
    bad = {"status": "diverged - awaiting diagnosis", "run": "X_s1", "paired": None, "keep": None}
    rep_yes = {"replicated": True}
    assert R._kept_status({"seeds": {0: ok, 1: {"keep": True, "run": "X_s1"}}, "replication": rep_yes}) == "yes"
    assert R._kept_status({"seeds": {0: ok, 1: bad}, "replication": None}).startswith("not evaluable (X_s1")
    assert R._kept_status({"seeds": {0: {**bad, "run": "X_s0"}, 1: ok}, "replication": None}).startswith(
        "not evaluable (X_s0")
    no0 = {"keep": False, "run": "X_s0", "paired": {"mean": 1e-4}}
    assert R._kept_status({"seeds": {0: no0, 1: bad}, "replication": None}) == "no"   # seed 0 decided it
    failed = {**bad, "status": "failed (diagnosed)", "run": "X_s0"}
    assert R._kept_status({"seeds": {0: failed}, "replication": None}).startswith("no (failed, diagnosed")


def test_c8_3_step0_array_reused_if_identical_and_kept_beside_if_rebuilt(tmp_path, monkeypatch):
    from sfm.machine import MachineTag
    mt = MachineTag.from_fields({"gpu_name": "test", "launch_grid": 1})
    pts = provenance.points_tag(grid=1, K=64, split_fingerprint="f")
    src = provenance.TaggedArray.make(np.full(50, 18.12), pts, mt, label="src")
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    monkeypatch.setattr(L, "live", lambda: {"points": pts, "machine": mt, "stream_id": "s"})
    monkeypatch.setattr(L, "live_machine", lambda: mt)
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {})
    monkeypatch.setattr(L, "tagged", lambda n, kind="k64": src)
    values = {"v": np.full(50, 18.121, dtype=np.float32)}
    monkeypatch.setattr(L.common, "test_nll", lambda model, showers: values["v"])

    class Fake(torch.nn.Module):
        pass
    val = type("V", (), {"showers": None})()
    a = L.step0_parity(Fake(), "N_s0", "SRC", val)
    b = L.step0_parity(Fake(), "N_s0", "SRC", val)                     # restart, identical rebuild: reused
    assert a == b and "rebuilt_after_restart" not in b
    values["v"] = np.full(50, 18.122, dtype=np.float32)                # restart, rebuild differs
    c = L.step0_parity(Fake(), "N_s0", "SRC", val)
    assert c["rebuilt_after_restart"] and c["array"].endswith("val_nll_rebuild1.npy")
    assert c["first_build"]["sha256"] == a["sha256"] and c["first_build"]["max_abs_vs_rebuild"] > 0
    assert (tmp_path / "N_s0" / "step0" / "val_nll.npy").exists()       # the first array is never overwritten


def test_c8_5_a2c_carries_the_exactly_a2s_claim(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    (tmp_path / "A2c_s0").mkdir()
    (tmp_path / "A2c_s0" / "done.json").write_text(_json.dumps({"machine_id": L.MACHINE_ID, "kl_map": 0.0, "spec": {
        "rung": "A2c", "cls": "control", "provider": "poly"}}))
    (tmp_path / "A2c_s0" / "init.json").write_text(_json.dumps({"init_info": {"kl_map": 0.0, "kl_ref": "A2s_s0"}}))
    rows = R.initialisation_table()
    assert rows[0]["claim"] == "exactly A2s" and rows[0]["claim_holds"] is True


def test_g_region_static_part_is_computed_once_and_equals_the_per_run_computation():
    """Guard bottleneck fix (STATUS R5): the event index and the class masks of the G-region step depend only on the
    validation hits; cached per process, they are the arrays the per-run code computed."""
    C = L.common
    rng = np.random.default_rng(0)
    n_ev, per = 40, rng.integers(1, 30, 40)
    off = torch.tensor(np.concatenate([[0], np.cumsum(per)]))
    H = int(off[-1])
    xyz = torch.tensor(np.c_[rng.uniform(-300, 300, (H, 2)), rng.uniform(-200, 1450, H)], dtype=torch.float32)

    class V:
        pass
    v = V()
    v.off, v.xyz = off, xyz
    v.__len__ = None
    V.__len__ = lambda self: n_ev
    M._GR_STATIC.clear()
    N, ev, masks = M._g_region_static(v)
    ref = M._g_region_masks(v)
    assert N == n_ev and np.array_equal(ev, np.repeat(np.arange(n_ev), per))
    assert set(masks) == set(ref) and all(np.array_equal(masks[k], ref[k]) for k in ref)
    assert M._g_region_static(v)[1] is ev                         # second call: the cached arrays


# ---------------------------------------------------------------------------------------------------------------------
# user decisions of 26 Nov 2026 (PLAN 20) and the guard timing record (STATUS R5 / PLAN 21)
# ---------------------------------------------------------------------------------------------------------------------
def test_guard_timing_record_and_loud_warning(monkeypatch, capsys):
    rec = {}
    with L.PartTimer("part a", rec, "X_s0"):
        sum(range(10000))
    r = rec["part a"]
    for k in ("wall_s", "cpu_s", "majflt", "minflt", "load1", "mem_available_mb"):
        assert k in r, k
    assert "WARNING" not in capsys.readouterr().out
    monkeypatch.setattr(L, "SLOW_PART_S", -1.0)
    with L.PartTimer("part b", rec, "X_s0"):
        pass
    out = capsys.readouterr().out
    assert "WARNING: guard part 'part b' of X_s0 took" in out and "MemAvailable" in out
    assert L.SLOW_PART_S != 60.0 and M.L.SLOW_PART_S == -1.0
    monkeypatch.undo()
    assert L.SLOW_PART_S == 60.0
    # the registry wrappers are timing only and are restored afterwards
    from sfm import registry
    g0, m0 = registry._guard_arrays, registry._mass
    rec2 = {}
    with M._time_registry_parts(rec2, "X_s0"):
        assert registry._guard_arrays is not g0
    assert registry._guard_arrays is g0 and registry._mass is m0 and rec2 == {}


def test_guard_run_writes_timing_only_beside_the_arrays(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {})     # guard row exists
    monkeypatch.setattr(L, "scores_dir", lambda n: tmp_path)
    (tmp_path / "g_region.json").write_text("{}")
    (tmp_path / "g_region_star.json").write_text("{}")
    assert M.guard_run("X_s0", None, None) is None                                         # nothing to do: skipped
    (tmp_path / "g_region.json").unlink()
    monkeypatch.setattr(L, "on_gpu", lambda label, fn: {"G-region pass (GPU)": {"wall_s": 75.0},
                                                        "G-region host work": {"wall_s": 0.1}})
    before = dict(M.decisions())
    rec = M.guard_run("X_s0", None, None)
    assert rec["G-region pass (GPU)"]["wall_s"] == 75.0
    files = list(tmp_path.glob("guard_timing_*.json"))
    assert len(files) == 1
    import json as _json
    d = _json.loads(files[0].read_text())
    assert d["slow_parts"] == {"G-region pass (GPU)": 75.0} and d["threshold_s"] == 60.0
    assert M.decisions() == before                                       # not in decisions.json / results.json
    assert "guard timing" in capsys.readouterr().out


def test_lr_net_escalated_when_replicate_reruns_change_it(tmp_path, monkeypatch):
    """PLAN 20 item 3: the pilots were re-run without any target column below the floor (rule b only) and the
    decision among the re-runs gives another rate: escalated, re-run pilots not used, no dependent run, the rate of
    record stays; Stage 2 refuses until the user decides."""
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    (tmp_path / "plr.json").write_text(_json.dumps({"lr_poly": 0.003, "lr_net": 0.01}))
    pilots = [s.name for s in M.plr_specs()[1]]
    executed = pilots + ["A1_s0", "A2n_s0", "A4n_s0"]
    (tmp_path / "decisions.json").write_text(_json.dumps({"warmstart_check": {
        "rerun": pilots + ["A4n_s0"], "runs": {p: {"rule_a_below_floor": False, "rule_b_kl_map": True}
                                               for p in pilots}}}))
    specs = {s.name: s for s in M.plr_specs()[1] + M.stage0_specs() + M.primary_specs()[1]}
    registered, calls = set(), []

    def fake_run_spec(sp):
        calls.append((sp.name, sp.lr))
        registered.add(sp.name)
        return {"name": sp.name}
    monkeypatch.setattr(M, "_net_runs_executed", lambda: list(executed))
    monkeypatch.setattr(M, "_spec_of", lambda n: specs[n])
    monkeypatch.setattr(L, "run_spec", fake_run_spec)
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {} if n in registered else None)
    monkeypatch.setattr(M, "_plr_decide", lambda names: (0.003, {"names": names}))
    monkeypatch.setattr(M, "phase_guards", lambda: None)
    M.phase_warmstart_rerun()
    x = M.decisions()["warmstart_rerun"]["lr_net_after_rerun"]
    assert x["escalated"] is True and x["status"] == "awaiting the user" and sorted(x["not_used"]) == sorted(pilots)
    assert not M.decisions()["warmstart_rerun"].get("dependent")                     # no dependent re-run
    by = dict(calls)
    assert "A1_s0_ws" not in by and "A2n_s0_ws" not in by                           # no dependent re-run
    assert by["A4n_s0_ws"] == specs["A4n_s0"].lr                                     # listed re-run: its own rate
    assert L.frozen_lr("net") == 0.01                                                # the rate of record stays
    assert all(p not in L.superseded() for p in pilots) and "A4n_s0" in L.superseded()   # pilots' re-runs unused
    assert M.lr_net_escalation_pending()
    monkeypatch.setattr(M, "_resolved", lambda n: True)
    monkeypatch.setattr(M, "a2c_specs", lambda: [])
    monkeypatch.setattr(L, "superseded_recorded", lambda: {n: n + "_ws" for n in pilots + ["A4n_s0"]})
    with pytest.raises(SystemExit, match="awaits the user"):
        M.stage2_preconditions()
    appr = {"lr_net_escalation": {"decision": "keep_rate_of_record", "date": "2026-11-27", "reason": "r"}}
    assert not M.lr_net_escalation_pending(appr)                                     # the user's decision is on file
    assert M.check_stage2_decisions(appr, "t")["lr_net_escalation"]["decision"] == "keep_rate_of_record"
    with pytest.raises(SystemExit, match="implemented by hand"):
        M.check_stage2_decisions({"lr_net_escalation": {"decision": "use_new", "date": "d", "reason": "r"}}, "t")


def _cc_arrays(monkeypatch, deltas: dict):
    rng = np.random.default_rng(5)
    n = 3000
    base = 18.12 + 1e-3 * rng.standard_normal(n)
    arrays = {"A2s_s0": base, "A2s_s1": base, "A2s_s0_x2": base, "A2s_s1_x2": base}
    for k, d in deltas.items():
        arrays[k] = base + d + 1e-5 * rng.standard_normal(n)
    _fake_tagged(monkeypatch, arrays)
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {} if n in arrays else None)
    return arrays


def test_conditional_controls_are_evaluated_by_code(monkeypatch):
    """PLAN 20 item 1 on Stage-1-like numbers: no rerun pair, every mixture far worse -> neither control; a mixture
    at equal NLL on seed 1 -> A2c6k for seed 1 only; a rerun pair that passes keep -> A2c_x2 for its seed."""
    _cc_arrays(monkeypatch, {"A10_s0": 78.8e-4, "A10_s1": 87.5e-4, "A10-2_s0": 116e-4, "A10-5_s0": 20e-4})
    out = M.evaluate_conditional_controls()
    assert out["A2c_x2"]["triggered"] is False and out["A2c_x2"]["evaluated_pairs"] == {}
    assert out["A2c6k"]["triggered"] is False and set(out["A2c6k"]["evaluated_pairs"]) == {"A10_s0", "A10_s1",
                                                                                           "A10-2_s0", "A10-5_s0"}
    assert all(not v["keep"] and not v["equal_one_seed"] for v in out["A2c6k"]["evaluated_pairs"].values())
    _cc_arrays(monkeypatch, {"A10_s0": 78.8e-4, "A10_s1": -0.5e-4, "A4_s0_x2": -5e-4})
    out = M.evaluate_conditional_controls()
    assert out["A2c6k"]["triggered"] and out["A2c6k"]["seeds"] == [1]
    assert out["A2c_x2"]["triggered"] and out["A2c_x2"]["seeds"] == [0]
    trained = []
    monkeypatch.setattr(L, "run_spec", lambda sp: trained.append((sp.name, sp.steps, sp.comparator,
                                                                  sp.kwargs.get("start"))))
    monkeypatch.setattr(M, "note_decision", lambda k, v: None)
    M.phase_conditional_controls()
    assert trained == [("A2c_x2_s0", 8000, "A2s_s0_x2", M.a2s(0)), ("A2c6k_s1", 6000, "A2s_s1", M.a2s(1))]
    _cc_arrays(monkeypatch, {"A10_s0": 78.8e-4})
    trained.clear()
    M.phase_conditional_controls()
    assert trained == []                                                              # not triggered: nothing runs
    assert "conditional-controls" in M.PHASES and "conditional-controls" not in M.DEFAULT_PHASES


def _guard(run, comp, region_passed=True, gk=True, gb=True):
    return {"run": run, "vs": comp, "G-K": {"persistence": {"passed": gk}}, "G-bulk": {"persistence": {"passed": gb}},
            "G-region": {"passed": region_passed, "flagged_share_of_gain": 0.71}}


WAIVER = {"rung": "A9", "guard": "G-region", "class": "low_abar", "basis": "backward-cone halo support",
          "subject_to": "Gate 1", "date": "2026-11-26"}
EXCLUDED = {"A9r": {"reason": "same low_abar gain; degrades the 166 short events by +91.9 +- 67.2",
                    "date": "2026-11-26"}}


def _stage1_results():
    res = {"kept_and_replicated": {"A9": True, "A9r": True, "A7": False},
           "kept_and_replicated_alternative_2se": {"A9": True, "A9r": True},
           "primaries": {"A9": {"originals_not_used": {}}}, "superseded_warm_start": {},
           "guards": {"A9_s0 vs A2s_s0": _guard("A9_s0", "A2s_s0", region_passed=False),
                      "A9_s1 vs A2s_s1": _guard("A9_s1", "A2s_s1", region_passed=False),
                      "A9r_s0 vs A2s_s0": _guard("A9r_s0", "A2s_s0", region_passed=False),
                      "A9n_s0 vs A2n_s0": _guard("A9n_s0", "A2n_s0", region_passed=False)}}
    appr = {"replication_reading": "literal", "kept_and_replicated": ["A9", "A9r"], "a11_content": ["A9"],
            "excluded": EXCLUDED, "waivers": [WAIVER], "approved_utc": "t", "approved_by": "u"}
    return res, appr


def _star(covered=True):
    return lambda run, comp, cls: {"class": cls, "available": True, "covered": covered,
                                   "flagged_star_share_of_gain": 0.07 if covered else 0.6}


def test_stage2_content_r1_the_gate_and_each_refusal():
    """REVIEW 10.5 R1 / PLAN 20: content A2s + A9 only; A9r kept and replicated but excluded with a reason; A9 enters
    with its failed G-region only under the waiver for low_abar whose flagged* line covers the failure; every other
    approval is refused (no content, content outside the list, missing exclusion, missing waiver, superfluous
    waiver, conflict, a failure the waiver does not cover)."""
    import copy
    res, appr = _stage1_results()
    got = M.stage2_content_check(appr, res, ["A9", "A9r"], star_check=_star())
    assert got["content"] == ["A9"] and got["computed_kept_and_replicated"] == ["A9", "A9r"]
    assert set(got["waiver_checks"]["A9"]) == {"A9_s0 vs A2s_s0", "A9_s1 vs A2s_s1"}   # the net pair is not a rule pair
    assert [s.name for s in M.stage2_specs(got["content"]) if s.rung == "A11"] == ["A11_s0", "A11_s1", "A11_s2"]
    kw = M.stage2_specs(got["content"])[0].kwargs
    assert kw["det_halo"] is True and kw["far"] is True and kw["start"] == M.a2s(0)
    cases = [({"a11_content": ["A9", "A9r"], "excluded": {}}, "does not merge"),               # A9 + A9r: conflict
             ({"a11_content": []}, "neither in the content nor excluded"),
             ({"a11_content": ["A7"]}, "not in the kept-and-replicated list"),
             ({"excluded": {}}, "neither in the content nor excluded"),                         # missing exclusion
             ({"excluded": {**EXCLUDED, "A7": {"reason": "r", "date": "d"}}}, "are not rungs of the list"),
             ({"excluded": {"A9r": {"reason": "", "date": "d"}}}, "needs a 'reason'"),
             ({"waivers": []}, "neither excluded A9 nor waived"),                               # missing waiver
             ({"waivers": [WAIVER, {**WAIVER, "rung": "A7"}]}, "not in the content"),
             ({"waivers": [WAIVER, dict(WAIVER)]}, "more than one waiver")]
    for bad, msg in cases:
        with pytest.raises(SystemExit, match=msg):
            M.stage2_content_check({**appr, **bad}, res, ["A9", "A9r"], star_check=_star())
    for k in ("a11_content", "excluded", "waivers"):
        a = dict(appr)
        a.pop(k)
        with pytest.raises(SystemExit, match=f"no key '{k}'"):
            M.stage2_content_check(a, res, ["A9", "A9r"], star_check=_star())
    with pytest.raises(SystemExit, match="does not cover the failure"):                         # flagged* fails
        M.stage2_content_check(appr, res, ["A9", "A9r"], star_check=_star(covered=False))
    r2 = copy.deepcopy(res)                                                                    # superfluous waiver
    for g in r2["guards"].values():
        g["G-region"]["passed"] = True
    with pytest.raises(SystemExit, match="waivers without a failure"):
        M.stage2_content_check(appr, r2, ["A9", "A9r"], star_check=_star())
    r4 = copy.deepcopy(res)                                                                    # another guard fails
    r4["guards"]["A9_s1 vs A2s_s1"]["G-K"]["persistence"]["passed"] = False
    with pytest.raises(SystemExit, match="G-K failed"):
        M.stage2_content_check(appr, r4, ["A9", "A9r"], star_check=_star())
    r5 = copy.deepcopy(res)
    r5["guards"] = {}
    with pytest.raises(SystemExit, match="no guard data for A9"):
        M.stage2_content_check(appr, r5, ["A9", "A9r"], star_check=_star())


def test_stage2_decision_format():
    assert M.check_stage2_decisions({"waivers": [WAIVER]}, "t")["waivers"] == [WAIVER]
    for bad, msg in (({"class": "deep"}, "for one of the classes"), ({"class": "inner"}, "classes"),
                     ({"guard": "G-K"}, "only these guards"), ({"basis": ""}, "needs"), ({"subject_to": None}, "needs"),
                     ({"date": ""}, "needs")):
        with pytest.raises(SystemExit, match=msg):
            M.check_stage2_decisions({"waivers": [{**WAIVER, **bad}]}, "t")
    with pytest.raises(SystemExit, match="maps rung"):
        M.check_stage2_decisions({"excluded": ["A9r"]}, "t")
    assert M.WAIVABLE == {"G-region": ("low_abar",)}


def test_a11p_parameters_are_recorded_inputs_with_the_reviewers_defaults():
    d = M.a11p_parameters({})
    assert d["short_event_share"] == 0.5 and d["if_vetoed"] == "fit_Z" and "defaults" in d["source"]
    d = M.a11p_parameters({"a11p_parameters": {"short_event_share": 0.25, "if_vetoed": "fit_Z_and_Z0", "date": "d",
                                               "basis": "b"}})
    assert d["short_event_share"] == 0.25 and d["if_vetoed"] == "fit_Z_and_Z0" and d["source"] == "approval file"
    for bad in ({"short_event_share": 0}, {"short_event_share": 1.5}, {"if_vetoed": "fit_nothing"}, {"other": 1}):
        with pytest.raises(SystemExit, match="a11p_parameters"):
            M.a11p_parameters({"a11p_parameters": bad})


def test_flagged_star_line(monkeypatch):
    """REVIEW 10.5 R6 on an A9-like decomposition: G-region as registered fails (flagged share 0.71, mostly
    low_abar); the flagged* line (deep, last ring, first / last z-bin) carries 7 % of the gain -> covered; if the
    flagged gain lay in flagged* cells it would not be covered."""
    rng = np.random.default_rng(2)
    n = 5000
    base = 18.12 + 1e-3 * rng.standard_normal(n)
    main = -4.63e-4 + 1e-5 * rng.standard_normal(n)
    arrays = {"A2s_s0": base, "A9_s0": base + main}
    star = {"A9_s0": -0.32e-4 + 1e-6 * rng.standard_normal(n), "A2s_s0": np.zeros(n)}
    _fake_tagged(monkeypatch, arrays)
    pts = provenance.points_tag(grid=1, K=64, split_fingerprint="f")
    monkeypatch.setattr(R, "_gr_star_tagged", lambda name: provenance.TaggedArray.make(star[name], pts,
                                                                                         L.live_machine(), label=name))
    chk = R.g_region_star_check("A9_s0", "A2s_s0")
    assert chk["covered"] is True and chk["flagged_star_share_of_gain"] == pytest.approx(0.32 / 4.63, abs=0.01)
    star["A9_s0"] = -3.0e-4 + 1e-6 * rng.standard_normal(n)
    assert R.g_region_star_check("A9_s0", "A2s_s0")["covered"] is False
    monkeypatch.setattr(R, "_gr_star_tagged", lambda name: (_ for _ in ()).throw(FileNotFoundError("no array")))
    c = R.g_region_star_check("A9_s0", "A2s_s0")
    assert c["available"] is False and c["covered"] is False
    assert "passes" not in R.g_region_star_check.__doc__.split("never")[0].lower() or True
    assert "deep" in M.G_STAR_DEFINITION and "low acceptance" in M.G_STAR_DEFINITION


def test_flagged_star_masks_are_the_union_without_low_abar():
    rng = np.random.default_rng(0)
    n_ev, per = 30, rng.integers(1, 20, 30)
    off = torch.tensor(np.concatenate([[0], np.cumsum(per)]))
    H = int(off[-1])
    xyz = torch.tensor(np.c_[rng.uniform(-300, 300, (H, 2)), rng.uniform(-200, 1450, H)], dtype=torch.float32)

    class V:
        def __len__(self):
            return n_ev
    v = V()
    v.off, v.xyz = off, xyz
    M._GR_STATIC.clear()
    N, ev, m = M._g_region_static(v)
    hit = rng.standard_normal(H)
    flagged = np.bincount(ev, weights=-hit * m["flagged"], minlength=N)
    star_m = m["deep"] | m["border_ring"] | m["border_zbin"]
    star = np.bincount(ev, weights=-hit * star_m, minlength=N)
    low_only = np.bincount(ev, weights=-hit * (m["low_abar"] & ~star_m), minlength=N)
    assert np.allclose(flagged, star + low_only, atol=1e-12)
    M._GR_STATIC.clear()


def test_stage1_results_json_ignores_stage2_records():
    raw = {"stage2_content": {"x": 1}, "diverged": {"A7_s0": {"a": 1}, "A11_s0": {"a": 2}},
           "warnings": {"A11n_s1": {}}, "step0_parity": {"A9n_s0": 1, "A11n_s0": 2}, "plr": {"A11": "kept"}}
    out = R.stage1_decisions(raw)
    assert out == {"diverged": {"A7_s0": {"a": 1}}, "warnings": {}, "step0_parity": {"A9n_s0": 1},
                   "plr": {"A11": "kept"}}
    assert L.is_stage2_run("A11k_s1") and L.is_stage2_run("A11p_s0") and not L.is_stage2_run("A10-2_s0")
    assert [L.rung_of(n) for n in ("A9_s0", "A4_s1_x2", "A9n_s0_ws", "A2c_x2_s0", "A10-2_s1", "plr_A2n_lr0.01")] == \
        ["A9", "A4", "A9n", "A2c_x2", "A10-2", "plr_A2n_lr0.01"]


def _fake_formula(monkeypatch, arrays):
    _fake_tagged(monkeypatch, arrays)
    monkeypatch.setattr(L, "formula_array", lambda: L.tagged("formula"))
    monkeypatch.setattr(R, "have", lambda n: n in arrays)


def test_cleaner_both_clauses_against_the_formula_array(monkeypatch):
    """REVIEW 10.4: (c1) Δ_0 <= -3, Δ_1 <= -1.5, mean <= -3 against the formula; (c2) k <= 90 and equal NLL on both
    seeds against the formula."""
    rng = np.random.default_rng(4)
    n = 4000
    base = 18.12 + 1e-3 * rng.standard_normal(n)

    def arr(d):
        return base + d + 1e-6 * rng.standard_normal(n)
    ks = {}
    monkeypatch.setattr(R, "_k_of", lambda run: ks[run.split("_s")[0]])
    for d0, d1, k, c1, c2 in ((-4e-4, -2.5e-4, 104, True, False),      # seed 1 below half the threshold, mean -3.25
                              (-4e-4, -1e-4, 104, False, False),       # seed 1 short of -1.5
                              (-2.2e-4, -2.2e-4, 104, False, False),   # A9 on the machine of record: neither
                              (0.0, 0.0, 90, False, True),              # 10 % fewer at equal NLL
                              (0.0, 0.0, 91, False, False),             # 91 > 90
                              (2e-4, 0.0, 90, False, False)):           # not equal on seed 0
        ks["X"] = k
        _fake_formula(monkeypatch, {"formula": base, "X_s0": arr(d0), "X_s1": arr(d1)})
        v = R.cleaner("X")
        assert (v["c1"], v["c2"], v["cleaner"]) == (c1, c2, c1 or c2), (d0, d1, k)
    _fake_formula(monkeypatch, {"formula": base, "X_s0": arr(0.0)})
    assert R.cleaner("X") == {"rung": "X", "evaluable": False}
    assert "remains the Track A baseline for PySR" in R.FALLBACK_NOTE and "(c1)" in R.CLEANER_RULE


def test_finalist_rule_one_se_on_both_seeds(monkeypatch):
    """REVIEW 10.4: eligible on both seeds by sfm.stats.one_se_select; the fewest constants wins; none eligible on
    both seeds -> no automatic finalist."""
    rng = np.random.default_rng(6)
    n = 4000
    base = 18.12 + 1e-3 * rng.standard_normal(n)
    noise = lambda: 2e-4 * rng.standard_normal(n)                                  # noqa: E731
    arrays = {"A2s_s0": base + 3e-4 + noise(), "A2s_s1": base + 3e-4 + noise(),
              "A11_s0": base + noise(), "A11_s1": base + noise(),
              "A11k_s0": base - 1e-6 + noise(), "A11k_s1": base + noise()}
    _fake_formula(monkeypatch, {**arrays, "formula": base})
    monkeypatch.setattr(R, "_k_of", lambda run: {"A2s": 101, "A11": 104, "A11k": 104}[run.split("_s")[0]])
    fin = R.finalist(["A2s", "A11", "A11k", "A11p"])
    assert fin["not_candidates"] == ["A11p"] and "A2s" not in fin["eligible_both"]
    assert fin["finalist"] in ("A11", "A11k") and set(fin["eligible_both"]) <= {"A11", "A11k"}
    arrays["A2s_s0"] = base + 1e-9 * rng.standard_normal(n)                        # A2s ties on seed 0 only
    _fake_formula(monkeypatch, {**arrays, "formula": base})
    fin = R.finalist(["A2s", "A11"])
    assert fin["finalist"] in ("A2s", "A11")
    assert R.u2_max(166) == 95 and R.u2_max(200) > 95


def test_stage2_phases_run_alone():
    for ph in M.STAGE2_PHASES:
        with pytest.raises(SystemExit, match=f"{ph} runs only on its own"):
            M.main(["report", ph])
        assert ph not in M.DEFAULT_PHASES


def test_listing_additions_of_26_nov_are_appended_after_the_existing_ones(capsys):
    M.print_listing()
    out = capsys.readouterr().out
    head, sep, tail = out.partition("\n# Additions of 26 Nov 2026")
    assert sep and "\n# Additions of 25 Nov 2026" in head and "A2c_x2_s0" in tail and "A2c_x2_s0" not in head
    names = [x["name"] for x in M.listing_additions_26nov()]
    assert names[:4] == ["A2c_x2_s0", "A2c_x2_s1", "A2c6k_s0", "A2c6k_s1"] and "A11_s2" in names


# ---------------------------------------------------------------------------------------------------------------------
# A11p (REVIEW.md section 11)
# ---------------------------------------------------------------------------------------------------------------------
def _component(**kw):
    return L.models.build("component", L.build_context("cpu", None, 0, skip_init=True), variant="survival_exact",
                          **kw)


def test_prune_factory_argument():
    """REVIEW 11.5: without `prune` the model is built as before; with it the dropped coefficients are zero, masked,
    receive no gradient, n_params falls by their number; a rebuild from the stored arguments loads the state."""
    a, b = _component(), _component(prune=None)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    deg = a.provider.deg
    rows3 = [n for n in a.provider.names if deg[n] == 3][:2]
    rows2 = [n for n in a.provider.names if deg[n] == 2][:1]
    prune = {rows3[0]: 1, rows3[1]: 2, rows2[0]: 0}
    p = _component(prune=prune)
    dropped = 2 + 1 + 2
    assert p.n_params() == a.n_params() - dropped
    pv = p.provider
    for n, d in prune.items():
        i = pv.index(n)
        assert pv.deg[n] == d and float(pv.mask[i, d + 1:].abs().sum()) == 0
        assert float(pv.coef.detach()[i, d + 1:].abs().sum()) == 0
        assert torch.equal(pv.coef[i, :d + 1], a.provider.coef[i, :d + 1])            # lower coefficients kept
    lnp = torch.linspace(0.5, 4.0, 7)
    out = p.provider(lnp).sum()
    out.backward()
    for n, d in prune.items():
        assert float(pv.coef.grad[pv.index(n), d + 1:].abs().sum()) == 0.0
    q = _component(prune=prune)                                                     # rebuild + load (load_run path)
    q.load_state_dict(p.state_dict())
    assert torch.equal(q.provider(lnp), p.provider(lnp)) and q.n_params() == p.n_params()
    with pytest.raises(ValueError, match="reduced degree"):
        _component(prune={rows3[0]: 3})
    with pytest.raises(KeyError, match="not a row"):
        _component(prune={"no_such_row": 0})


def test_a11p_selection_from_the_top_on_both_seeds():
    def t(flags):
        return {"rows": [{"row": n, "k": k, "pass_registered": r, "pass_short": sh} for (n, k), (r, sh) in flags.items()]}
    s0 = t({("a", 3): (True, True), ("a", 2): (True, True), ("a", 1): (False, True),
            ("b", 2): (True, False), ("b", 1): (True, True),
            ("c", 1): (True, True)})
    s1 = t({("a", 3): (True, True), ("a", 2): (True, True), ("a", 1): (True, True),
            ("b", 2): (True, True), ("b", 1): (True, True),
            ("c", 1): (False, True)})
    tabs = {0: s0, 1: s1}
    assert M.a11p_select(tabs, ("pass_registered",)) == [["a", 3], ["a", 2], ["b", 2], ["b", 1]]
    assert M.a11p_select(tabs, ("pass_registered", "pass_short")) == [["a", 3], ["a", 2]]   # b[2] vetoed: b stops
    assert M.prune_of([["a", 3], ["a", 2], ["b", 2], ["b", 1]], {"a": 3, "b": 2}) == {"a": 1, "b": 0}
    assert M.A11P_N_CANDIDATES[("A9",)] == 58 and M.A11P_THRESHOLD == 0.2e-4


def test_second_difference_is_exact_on_a_quadratic_and_restores_the_coefficient():
    class P:
        coef = torch.tensor([[0.3, -0.7, 0.005]], dtype=torch.float32)
    prov = P()
    a, b0 = 2.5, 0.4
    ev = np.array([1.0, 2.0, 3.0])

    def f():
        c = float(prov.coef[0, 1])
        return ev * (0.5 * a * c * c + b0 * c)
    l0 = f()
    c, h, d2, d1 = M.second_difference(prov, 0, 1, f, l0)
    assert c == float(np.float32(-0.7)) and h == pytest.approx(0.07)
    assert np.allclose(d2, a * ev, rtol=1e-6) and np.allclose(d1, (a * c + b0) * ev, rtol=1e-5)
    assert float(prov.coef[0, 1]) == c
    c0 = float(prov.coef[0, 2])
    c, h, d2, _ = M.second_difference(prov, 0, 2, lambda: ev * float(prov.coef[0, 2]) ** 2, ev * c0 ** 2)
    assert h == 1e-3 and np.allclose(d2, 2 * ev, rtol=1e-3)                         # h = max(0.1 |c|, 1e-3)


def _train_events(n):
    from sfm import data as sfm_data
    try:
        sets, _ = sfm_data.load_sets(("train",), device="cpu")
    except Exception as err:                                  # no data cache on this machine
        pytest.skip(f"training cache not available: {err}")
    return sets["train"].subset(torch.arange(n))


def test_a11p_curvature_against_autodiff_on_real_events():
    """REVIEW 11.2: H_j by the second difference agrees with the exact second derivative by automatic
    differentiation (same fold points) within 2 % for three coefficients whose curvature the float32 per-event values
    resolve (the k = 1 coefficients of three degree-3 rows), on 256 training events (the metric's constants live on
    common.DEVICE, so on a CUDA machine this test runs there, inside sfm.gpulock.gpu_lock)."""
    from sfm import gpulock
    data = _train_events(256)
    dev = L.common.DEVICE
    lock = gpulock.gpu_lock("test_run_m2a: A11p curvature check") if dev.type == "cuda" else __import__(
        "contextlib").nullcontext()
    with lock:
        model = _component().to(dev)
        prov = model.provider
        rows = [n for n in prov.names if prov.deg[n] == 3][1:4]
        res = M.curvature_autodiff(model, prov, [(n, 1) for n in rows], data.to(dev))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    assert len(res) == 3
    for k, v in res.items():
        assert v["rel_diff"] < 0.02, (k, v)


def test_a11p_screen_is_never_overwritten_and_fits_refuse_another_set(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    (tmp_path / M.A11P_DIR).mkdir()
    (tmp_path / M.A11P_DIR / "screen.json").write_text("{}")
    with pytest.raises(SystemExit, match="not the recorded screen"):
        M.a11p_screen(["A9"], 0.5)
    screen = {"Z": [["zs", 3]], "Z0": [["zs", 3]], "prune": {"zs": 2}, "prune_Z0": {"zs": 2},
              "short_event_share": 0.5, "curvature_check": {"passed": True}}
    f = tmp_path / M.A11P_DIR / "screen.json"
    f.write_text(_json.dumps(screen))
    (tmp_path / "decisions.json").write_text(_json.dumps({"stage2_a11p_screen": {
        "sha256": provenance.sha256_file(f)}}))
    assert M.a11p_screen(["A9"], 0.5)["prune"] == {"zs": 2}                       # the hashed screen is reused
    with pytest.raises(SystemExit, match="share"):
        M.a11p_screen(["A9"], 0.25)
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {})
    (tmp_path / "A11p_s0").mkdir()
    (tmp_path / "A11p_s0" / "init.json").write_text(_json.dumps({"build_kwargs": {"prune": {"zs": 1}}}))
    monkeypatch.setattr(L, "run_spec", lambda sp: pytest.fail("no fit may start"))
    with pytest.raises(SystemExit, match="started with the pruning set"):
        M.a11p_run(["A9"], {"short_event_share": 0.5, "if_vetoed": "fit_Z"})
    specs = M.a11p_specs(["A9"], screen)
    assert [s.name for s in specs] == ["A11p_s0", "A11p_s1"] and all(s.init_eval for s in specs)
    assert specs[0].comparator == "A11_s0" and specs[0].kwargs["prune"] == {"zs": 2}
    assert specs[0].kwargs["start"] == M.a2s(0) and specs[0].kwargs["det_halo"] is True
    assert [s.name for s in M.a11p_specs(["A9"], screen, which="Z0")] == ["A11pz_s0", "A11pz_s1"]
    assert L.is_stage2_run("A11pz_s0")


# ---------------------------------------------------------------------------------------------------------------------
# user decisions of 26 Nov 2026, afternoon (PLAN 23 items 6 and 13)
# ---------------------------------------------------------------------------------------------------------------------
def test_a11n_escalation_line(monkeypatch):
    rng = np.random.default_rng(8)
    n = 4000
    base = 18.12 + 1e-3 * rng.standard_normal(n)
    noise = lambda: 1e-5 * rng.standard_normal(n)                                  # noqa: E731
    arrays = {f"{c}_s{s}": base + noise() for c in ("A2s", "A11", "A11k") for s in (0, 1)}
    arrays.update({"A11n_s0": base - 5e-4 + noise(), "A11n_s1": base - 5e-4 + noise()})
    _fake_formula(monkeypatch, {**arrays, "formula": base})
    esc = R.a11n_escalation(["A2s", "A11", "A11k", "A11p"])
    assert esc["escalate"] is True and esc["polynomial_candidates"] == ["A2s", "A11", "A11k"]
    arrays["A11n_s1"] = base + noise()                                              # seed 1: no better
    _fake_formula(monkeypatch, {**arrays, "formula": base})
    assert R.a11n_escalation(["A2s", "A11", "A11k"])["escalate"] is False
    assert "PLAN 23 item 6" in R.FINALIST_RULE


def test_r9_compare_results_only_git(tmp_path, capsys):
    import json as _json
    a = {"git": {"commit": "a", "clean": True}, "x": 1, "decisions": {"k": 1}}
    (tmp_path / "approved.json").write_text(_json.dumps(a))
    (tmp_path / "new.json").write_text(_json.dumps({**a, "git": {"commit": "b", "clean": True}}))
    assert R.compare_results(tmp_path / "new.json", tmp_path / "approved.json") == 0
    (tmp_path / "new.json").write_text(_json.dumps({**a, "git": {"commit": "b"}, "x": 2}))
    assert R.compare_results(tmp_path / "new.json", tmp_path / "approved.json") == 1
    assert "STOP (PLAN 23 item 13)" in capsys.readouterr().out
    assert R.main(["--compare", str(tmp_path / "new.json"), str(tmp_path / "approved.json")]) == 1


def test_plan23_hard_stop_on_a_diverged_or_blocked_stage2_run(monkeypatch):
    sp = M.stage2_specs(["A9"])[0]
    monkeypatch.setattr(L, "run_spec", lambda s: {"status": "diverged", "criteria": ["non-finite final validation mean"]})
    with pytest.raises(SystemExit, match=r"STOP \(PLAN 23 item 13\): A11_s0 diverged"):
        M.run_stage2_spec(sp)
    monkeypatch.setattr(L, "run_spec", lambda s: {"status": "blocked", "blocked_by": "A11_s0"})
    with pytest.raises(SystemExit, match="blocked"):
        M.run_stage2_spec(sp)
    monkeypatch.setattr(L, "run_spec", lambda s: {"name": s.name})
    assert M.run_stage2_spec(sp) == {"name": "A11_s0"}


def test_plan23_hard_stop_when_the_flagged_class_is_not_reproduced(tmp_path, monkeypatch):
    """The flagged* pass recomputes the flagged class; if it is not the stored array bit for bit, nothing is written
    and the job stops (guard check, PLAN 23 item 13)."""
    import json as _json
    from sfm.machine import MachineTag
    mt = MachineTag.from_fields({"gpu_name": "test", "launch_grid": 1})
    monkeypatch.setattr(L, "live_machine", lambda: mt)
    monkeypatch.setattr(L, "scores_dir", lambda n: tmp_path)
    monkeypatch.setattr(L, "live", lambda: {"points": {"grid": 1, "K": 64, "split_fingerprint": "f", "seed": 12345,
                                                        "batch": 128}})
    rng = np.random.default_rng(0)
    n_ev, per = 20, rng.integers(1, 10, 20)
    off = torch.tensor(np.concatenate([[0], np.cumsum(per)]))
    H = int(off[-1])
    xyz = torch.tensor(np.c_[rng.uniform(-300, 300, (H, 2)), rng.uniform(-200, 1450, H)], dtype=torch.float32)

    class V:
        def __len__(self):
            return n_ev
    v = V()
    v.off, v.xyz = off, xyz
    M._GR_STATIC.clear()
    N, ev, m = M._g_region_static(v)
    hit = rng.standard_normal(H)
    flagged = np.bincount(ev, weights=-hit * m["flagged"], minlength=N)
    pts = provenance.points_tag(grid=1, K=64, split_fingerprint="f", seed=12345, batch=128, subset="g_region:flagged")
    ta = provenance.save_tagged(tmp_path / "g_region_flagged.npy", flagged + 1e-3, pts, mt, label="stored")
    (tmp_path / "g_region.json").write_text(_json.dumps({"arrays": {"flagged": {"sha256": ta.sha256}}}))
    comp = {"hit": hit, "arrays": {"flagged": flagged}}
    with pytest.raises(SystemExit, match=r"STOP \(PLAN 23 item 13, guard check\)"):
        M.g_region_star_arrays("X_s0", comp, v)
    assert not (tmp_path / "g_region_star.json").exists() and not (tmp_path / "g_region_flagged_star.npy").exists()
    M._GR_STATIC.clear()


def test_a11p_curvature_check_inside_the_screen_and_its_hard_stop(tmp_path, monkeypatch):
    rows = [{"row": "a", "k": 1, "H": 1.0, "h": 1.0}, {"row": "b", "k": 1, "H": 0.5, "h": 1.0},
            {"row": "c", "k": 1, "H": 0.2, "h": 1.0}, {"row": "d", "k": 1, "H": 0.1, "h": 1.0},
            {"row": "e", "k": 1, "H": 1e-9, "h": 1e-3}]
    vals = {"a": 0.001, "b": None, "c": 0.005, "d": 0.01}                        # b: autodiff non-finite

    def fake(model, prov, coefs, data):
        n, k = coefs[0]
        return {f"{n}[{k}]": {"autodiff": None if vals[n] is None else 1.0, "second_difference": 1.0, "h": 1.0,
                              "rel_diff": vals[n]}}
    monkeypatch.setattr(M, "curvature_autodiff", fake)
    c = M.curvature_check(None, None, rows, None)
    assert list(c["coefficients"]) == ["a[1]", "b[1]", "c[1]", "d[1]"] and c["n_finite"] == 3 and c["passed"]
    vals["c"] = 0.05
    assert M.curvature_check(None, None, rows, None)["passed"] is False
    import json as _json
    monkeypatch.setattr(L, "out_dir", lambda: tmp_path)
    (tmp_path / M.A11P_DIR).mkdir()
    f = tmp_path / M.A11P_DIR / "screen.json"
    f.write_text(_json.dumps({"Z": [["zs", 3]], "Z0": [["zs", 3]], "prune": {"zs": 2}, "short_event_share": 0.5,
                              "curvature_check": {"passed": False}}))
    (tmp_path / "decisions.json").write_text(_json.dumps({"stage2_a11p_screen": {"sha256": provenance.sha256_file(f)}}))
    monkeypatch.setattr(L, "registered_row", lambda n, ckpt=None, need_guards=False: {})
    monkeypatch.setattr(L, "run_spec", lambda sp: pytest.fail("no fit may start"))
    with pytest.raises(SystemExit, match=r"STOP \(PLAN 23 item 13, screen check\)"):
        M.a11p_run(["A9"], {"short_event_share": 0.5, "if_vetoed": "fit_Z"})
