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
    a, kept, s = M.stage2_approval()
    assert kept == ["A7"] and s == sha
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
    assert any("non-finite" in c for c in L.divergence_check(nan, []))
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
    (tmp_path / "decisions.json").write_text(_json.dumps({"warmstart_check": {"rerun": pilots + ["A4n_s0"]}}))
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
    monkeypatch.setattr(M, "decisions", lambda: {"warmstart_check": {"rerun": ["A4n_s0"]}})
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
