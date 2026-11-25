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
