"""Tests of the Track B list driver (scripts/run_m2b_list.py), its report (run_m2b_list_report.py), the new model
factories (field_kl, field_distill, field_acc with Track A templates) and the N5 pins. No real fit; the test split is
never opened; the campaign registry and the test-access log stay unchanged."""
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import common as C
from sfm import machine, models, paths, policy, provenance, registry

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parent
sys.path.insert(0, str(MARTIN / "scripts"))
import run_m2b_lib as lib  # noqa: E402
import run_m2b_list as D  # noqa: E402
import run_m2b_list_report as R  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
H200 = MARTIN / "results" / "machines" / "h200-g1056-97b197a7"
H200_IDENTITY = {"gpu_name": "NVIDIA H200", "sm_count": 132, "max_threads_per_sm": 2048, "launch_grid": 1056,
                 "torch": "2.14.0+cu130", "cuda": "13.0", "cudnn": 92400, "python": "3.13.12", "tf32_matmul": False,
                 "tf32_cudnn": False}


def _sha(p):
    return provenance.sha256_file(p) if Path(p).exists() else None


@pytest.fixture(scope="module", autouse=True)
def campaign_files_unchanged():
    files = [paths.REPO_RESULTS / "registry.jsonl", paths.REPO_RESULTS / "test_access.log"]
    before = {f: _sha(f) for f in files}
    yield
    assert {f: _sha(f) for f in files} == before


# ---- the list against AUDIT 4.2 ----------------------------------------------------------------------------------
def test_list_matches_audit_4_2():
    L = D.build_list()
    names = [s.name for s in L]
    assert len(names) == len(set(names)) == 31
    rungs = {}
    for s in L:
        rungs.setdefault(s.rung, []).append(s)
    assert set(rungs) == {"B1d", "B1f", "B2a", "B2b", "B2c", "B3s", "B4f", "B4s", "B4x", "B5f", "B6a", "B6b", "B6c",
                          "B-ceil"}
    seeds = {r: sorted(s.seed for s in v) for r, v in rungs.items()}
    for r in ("B2a", "B2b", "B2c", "B3s", "B4f", "B5f", "B6a", "B6b", "B6c"):
        assert seeds[r] == [0, 1], r
    assert seeds["B4s"] == seeds["B4x"] == seeds["B1f"] == [0] and len(rungs["B1d"]) == 2
    prim = sorted({s.rung for s in L if s.cls == "primary"})
    assert prim == ["B1d", "B2b", "B2c", "B4f", "B6c"] and set(D.PRIMARY_RUNGS) == set(prim) | {"B1"}
    assert {s.rung for s in L if s.cls == "comparator"} == {"B2a", "B6a"}
    tb = rungs["B2a"][0]                                               # T_B
    assert (tb.steps, tb.lr, tb.lr_end, tb.batch, tb.eval_every, tb.selection, tb.smooth, tb.weight_decay,
            tb.ema_decay) == (6000, 1e-3, 1e-4, 256, 250, "last", 0.0, 0.0, 0.999)
    assert rungs["B2b"][0].k_train == 64 and rungs["B2c"][0].fold == "stratified"
    b3s = rungs["B3s"][0]
    assert (b3s.steps, b3s.lr, b3s.lr_end, b3s.smooth, b3s.kwargs) == (12000, 3e-3, 1e-4, 1e-4,
                                                                       {"hidden": 32, "n_layers": 4})
    assert [s.kwargs["template"] for s in rungs["B4f"] + rungs["B4s"] + rungs["B4x"]] == ["front", "front", "stored",
                                                                                          "full_box"]
    assert all(s.kwargs["warm_steps"] == 500 and s.kwargs["base"] == "mlp_std" for s in rungs["B4f"])
    assert [(s.kwargs["hidden"], s.kwargs["n_layers"]) for r in ("B6a", "B6b", "B6c") for s in rungs[r][:1]] == \
        [(32, 4), (32, 5), (64, 4)]
    assert all(s.kwargs["kl_steps"] == 3000 and s.kwargs["grid"] == 256 for r in ("B6a", "B6b", "B6c")
               for s in rungs[r])
    bc = rungs["B-ceil"]
    assert sorted((s.kwargs["n_knots"], s.weight_decay, s.train_fraction) for s in bc) == sorted(
        [(1, 0.0, 1.0), (1, 1e-4, 1.0), (6, 0.0, 0.25), (6, 0.0, 0.5), (6, 0.0, 1.0), (6, 1e-4, 0.25),
         (6, 1e-4, 0.5), (6, 1e-4, 1.0)])
    assert all((s.steps, s.lr, s.lr_end, s.eval_every, s.kwargs["knots"]) == (3000, 3e-3, 3e-4, 100, "uniform")
               for s in bc)
    b1d = rungs["B1d"][0]
    assert b1d.data_free and b1d.kwargs["kl_steps"] == 6000 and b1d.kwargs["grid"] == 256
    # user decision (PLAN section 23 item 7): fixed, validation-blind starts; B1f from the first replicate
    assert [(s.name, s.kwargs["start"], s.replicate_of) for s in rungs["B1d"]] == [
        ("B1d_m11", "s2_seed11", None), ("B1d_m12", "s2_seed12", "B1d_m11")]
    b1f = rungs["B1f"][0]
    assert (b1f.steps, b1f.lr, b1f.lr_end, b1f.k_train) == (3000, 3e-4, 3e-5, 16)
    assert b1f.kwargs == {"run": "<B1d_m11>"} and b1f.comparator == "B1d_m11"
    assert not (set(names) & D.BATCH1_NAMES)
    assert {s.name for s in L if s.ema_row} == {f"{r}_s{x}" for r in ("B2a", "B2b", "B2c", "B3s") for x in (0, 1)}


def test_pins_are_current_and_enforced(monkeypatch):
    D.check_pins(D.build_list())                                      # the committed pin file is up to date
    changed = [replace(s, steps=s.steps + 1) if s.name == "B2b_s0" else s for s in D.build_list()]
    with pytest.raises(D.HardStop):                                    # a hard stop (PLAN 23 item 13)
        D.check_pins(changed)
    x = D.x2_spec(D.spec_map()["B2b_s0"])
    assert x.steps == 12000 and x.comparator == "B2a_s0_x2" and x.cls == "rerun"
    assert D.pinned()["x2_recipes"][x.name] == x.recipe_hash()


def test_smoke_and_injection_switches(monkeypatch):
    monkeypatch.delenv(D.SMOKE_ENV, raising=False)
    with pytest.raises(SystemExit):
        D.configure(True)
    monkeypatch.setenv("SFM_M2B_INJECT_DIVERGENCE", "B2a_s0")
    with pytest.raises(SystemExit):
        D.configure(False)
    monkeypatch.delenv("SFM_M2B_INJECT_DIVERGENCE")
    monkeypatch.setenv(D.SMOKE_ENV, "1")
    D.configure(True)
    assert D.CTX.stage == "m2b_smoke" and D.fit_config(D.spec_map()["B3s_s0"]).steps == 120
    D.configure(False)
    assert D.CTX.stage == "m2b_mlp" and D.fit_config(D.spec_map()["B3s_s0"]).steps == 12000


# ---- divergence (PLAN section 20, item 2) ---------------------------------------------------------------------------
def _h(vals, skips=None):
    skips = skips or [0] * len(vals)
    return [{"step": 250 * (i + 1), "val": v, "skipped_loss": s, "skipped_grad": 0} for i, (v, s) in
            enumerate(zip(vals, skips))]


def test_divergence_rules():
    assert D.divergence_check(_h([18.2, 18.15, 18.13]), []) == ([], [])
    crit, warn = D.divergence_check(_h([18.2, float("nan"), 18.13]), [])
    assert crit == [] and warn and "recovered" in warn[0]                 # mid-run NaN that recovers: a warning
    crit, _ = D.divergence_check(_h([18.2, 18.15, float("nan")]), [])
    assert crit and "last validation" in crit[0]
    crit, _ = D.divergence_check(_h([18.2, 18.15, 18.14], [0, 0, 200]), [])
    assert any("frozen" in c for c in crit)
    crit, _ = D.divergence_check(_h([18.2, 18.15, 19.5]), [18.12])
    assert any("blow-up" in c for c in crit)
    sm = D.spec_map()
    assert D.status_label(sm["B3s_s0"]) == "diverged"
    for n in ("B2b_s0", "B2a_s0", "Bceil_k6_wd0_f25", "B1d_m11"):
        assert D.status_label(sm[n]) == "diverged - awaiting diagnosis"


# ---- report rules ---------------------------------------------------------------------------------------------------
def _ta(vals, tag=None):
    pts = provenance.points_tag(grid=144, K=64, split_fingerprint="f" * 64)
    return provenance.TaggedArray.make(np.asarray(vals, np.float32), pts, tag or machine.machine_tag())


def test_promotion_parts_and_replication():
    rng = np.random.default_rng(1)
    ref = _ta(18 + rng.normal(0, 0.1, 4000))
    good = _ta(ref.values - 8e-4 + rng.normal(0, 1e-4, 4000))
    b0 = _ta(ref.values + 1e-4)
    p = R.parts(good, ref, b0)
    assert p["a_gain"] and p["b_z"] and p["c_b0"]
    p0 = R.parts(good, ref, None)
    assert p0["c_b0"] is None and p0["vs_b0"] is None                    # B0 incomplete: not evaluated
    weak = _ta(ref.values - 1e-4 + rng.normal(0, 1e-4, 4000))
    both = R.two_seed(good, weak, ref)
    rep = R.replication(p["vs_mlp"], R.parts(weak, ref, b0)["vs_mlp"], both)
    assert rep["same_sign"] and not rep["seed1_half_threshold"] and not rep["replicated"]
    good2 = _ta(ref.values - 7e-4 + rng.normal(0, 1e-4, 4000))
    rep = R.replication(p["vs_mlp"], R.parts(good2, ref, b0)["vs_mlp"], R.two_seed(good, good2, ref))
    assert rep["replicated"]
    other = _ta(good.values, machine.machine_tag().with_(draw="emulated"))
    with pytest.raises(provenance.MixedTags):
        R.parts(other, ref, b0)


# ---- N5 ---------------------------------------------------------------------------------------------------------------
def test_n5_pinned_configs(tmp_path):
    from sfm import train
    rc = train.load_run_config(MARTIN / "configs" / "m2b" / "s2_seed13.toml")
    lib.check_pinned_config(rc)
    assert rc.hash() == "0354cc665e06bf4b"
    copy = tmp_path / "s2_seed13.toml"
    copy.write_text((MARTIN / "configs" / "m2b" / "s2_seed13.toml").read_text().replace("steps = 6000", "steps = 500"))
    with pytest.raises(SystemExit):                                      # outside the clone and another recipe
        lib.check_pinned_config(train.load_run_config(copy))


@cuda
def test_n5_generic_fit_scripts_refuse_off_the_laptop(monkeypatch):
    import fit
    monkeypatch.setattr(policy, "LAPTOP_MACHINE_IDS", ())
    monkeypatch.setattr(machine, "require_guarded_start", lambda what="": {})     # the subject is the N5 refusal
    with pytest.raises(SystemExit, match="no starting-point"):
        fit.main([str(MARTIN / "configs" / "m2b" / "b0h_seed0.toml"), "--dry-run"])


# ---- factories --------------------------------------------------------------------------------------------------------
@cuda
def test_field_kl_and_distill_factories(tmp_path):
    from test_train import make_showers
    tr = make_showers(400, 3, C.DEVICE)
    ctx = models.BuildContext(device=C.DEVICE, train=tr, seed=0, log=lambda *a: None)
    torch.manual_seed(0)
    m = models.build("field_kl", ctx, hidden=8, n_layers=2, kl_steps=20)
    assert m.init_info["kl_warm_start"]["kl_final"] < m.init_info["kl_warm_start"]["kl_initial"]
    grid = models.train_quantile_grid(ctx, 256)
    lnp = tr.ev["lnp"].cpu()
    assert len(grid) == 256 and float(grid.min()) > float(lnp.min()) and float(grid.max()) < float(lnp.max())
    with pytest.raises(ValueError):
        models.build("field_kl", ctx, target="formula")
    # field_distill: start and target are finished runs
    d = tmp_path / "runs"
    for name, mod in (("member", models.build("field", ctx, hidden=8, n_layers=2)),):
        (d / name).mkdir(parents=True)
        torch.save({"state": mod.state_dict(), "meta": {"model": {"factory": "field",
                                                                  "kwargs": {"hidden": 8, "n_layers": 2}}}},
                   d / name / "model.pt")
    (d / "ens").mkdir()
    ens = models.build("ensemble", models.BuildContext(device=C.DEVICE, results_dir=d), members=["member"])
    torch.save({"state": ens.state_dict(), "meta": {"model": {"factory": "ensemble", "kwargs": {"members": ["member"]}}}},
               d / "ens" / "model.pt")
    ctx2 = models.BuildContext(device=C.DEVICE, train=tr, seed=0, log=lambda *a: None, results_dir=d)
    st = models.build("field_distill", ctx2, start="member", target="ens", kl_steps=5)
    assert isinstance(st, models.FieldNet) and st.init_info["kl_warm_start"]["steps"] == 5
    rebuilt = models.build("field_distill", models.BuildContext(device=C.DEVICE, train=tr, skip_init=True,
                                                                results_dir=d), start="member", target="ens")
    assert not hasattr(rebuilt, "init_info")                              # no KL fit when rebuilding


# ---- batch-1 recognition on the real H200 records (read-only; the H200 identity patched in) -------------------------
@cuda
@pytest.mark.skipif(not (H200 / "registry.jsonl").exists(), reason="H200 machine directory not present")
def test_batch1_runs_recognised_and_sources_verified_on_the_h200_records(monkeypatch):
    real = machine.machine_tag

    def live(deterministic=True, draw="native", device=0):
        return real(deterministic=deterministic, draw=draw, device=device).with_(**H200_IDENTITY)

    monkeypatch.setattr(machine, "machine_tag", live)
    assert live().id == "h200-g1056-97b197a7"
    monkeypatch.setattr(paths, "OUT_RESULTS", paths.REPO_RESULTS)
    monkeypatch.setattr(lib, "MACHINE", live())
    monkeypatch.setattr(lib, "MACHINE_DIR", H200)
    monkeypatch.setattr(lib, "OUT", H200 / "runs" / "m2b_mlp")
    lib.live_stream_id.cache_clear()
    try:
        D.configure(False)
        for names in D.GIVEN.values():
            for n in names:
                assert D.given_row(n) is not None, n                     # batch-1 rows on the H200 stream
        assert D.registered_members() == [f"s2_seed{s}" for s in lib.S1_SEEDS]    # seed order, no ranking
        for name, member in (("B1d_m11", "s2_seed11"), ("B1d_m12", "s2_seed12")):
            s = D.spec_map()[name]
            kw = D.resolve(s)
            assert kw["start"] == lib.member(member) and kw["target"].endswith("runs/m2b_mlp/E5_s2")
            if (H200 / "runs" / "m2b_mlp" / "E5_s2" / "model.pt").exists():
                src = D.verify_sources(s, kw)                            # SHA-256 against the H200 rows
                assert src["start"]["name"] == member and src["target"]["name"] == "E5_s2"
        kw = D.resolve(D.spec_map()["B1f_s0"])
        assert kw["run"].endswith("runs/m2b_mlp/B1d_m11")
        assert D.status(D.spec_map()["B2a_s0"]) == "pending"            # nothing of the list ran there yet
        if (H200 / "runs" / "m2a_dist" / "templates" / "full_box.pt").exists():
            t = D.verify_template("full_box")                           # the Track A record of the H200
            assert t["track_a_record"] and len(t["file_sha256"]) == 64
        if list((H200 / "refsets").glob("*/mlp_val_nll.npy")):        # per-event arrays: laptop repository only
            rs = registry.reference_set(need_guards=True)               # licensed by a passing parity record
            assert rs.manifest["parity"]["verdict"] == "PASS"
    finally:
        lib.live_stream_id.cache_clear()


def test_no_validation_number_decides_what_is_trained(monkeypatch):
    """The B1d / B1f starts do not depend on any registered validation value (user decision, PLAN 23 item 7):
    resolve() gives the same starts whatever the members' rows say."""
    calls = []
    monkeypatch.setattr(D, "given_row", lambda n: calls.append(n) or {"val_mean": {"s2_seed11": 9.0}.get(n, 1.0)})
    monkeypatch.setattr(D, "ensemble_ref", lambda: "out:machines/x/runs/m2b_mlp/E5_s2")
    D.configure(False)
    a = D.resolve(D.spec_map()["B1d_m11"])["start"]
    monkeypatch.setattr(D, "given_row", lambda n: {"val_mean": {"s2_seed11": 0.0}.get(n, 1.0)})
    b = D.resolve(D.spec_map()["B1d_m11"])["start"]
    assert a == b == lib.member("s2_seed11") and not calls              # no row consulted for the choice
    assert not hasattr(D, "member_ranking")


def test_hard_stop_exit_code_and_last_line(monkeypatch, capsys):
    """PLAN section 23 item 13: a failed check ends the job with exit code 3 and a clear last line."""
    monkeypatch.setattr(D, "_main", lambda argv=None: D.hard_stop("reference", "unit test: source SHA-256 differs"))
    monkeypatch.setattr(D, "record_decision", lambda *a, **k: None)
    assert D.main([]) == D.HARD_STOP_EXIT
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("HARD STOP (PLAN.md section 23, item 13) - reference: unit test") and "waiting" in last
    monkeypatch.setattr(D, "_main", lambda argv=None: (_ for _ in ()).throw(registry.ReferenceSetDisputed("x")))
    assert D.main([]) == D.HARD_STOP_EXIT
    assert "reference check failed" in capsys.readouterr().out.strip().splitlines()[-1]
    monkeypatch.setattr(D, "_main", lambda argv=None: (_ for _ in ()).throw(machine.InsufficientGPUMemory("full")))
    assert D.main([]) == D.RESOURCE_EXIT


def test_guard_log_refusal_is_a_hard_stop(monkeypatch, tmp_path):
    from sfm import guard
    f = tmp_path / "guard.log"
    monkeypatch.setattr(guard, "guard_log_path", lambda: f)
    pid = os.getpid()
    f.write_text(f"t\tpid {pid}\tx.py\tpyarrow\t{D.TOLERATED_GUARD_REASON}\n")
    D.check_guard_log()                                                # the tolerated pandas probe
    f.write_text(f.read_text() + f"t\tpid {pid + 1}\tx.py\t<area>/other\topen: refused\n")
    D.check_guard_log()                                                # another process: not this job's
    f.write_text(f.read_text() + f"t\tpid {pid}\tx.py\t<area>/other\topen: inside the guarded area\n")
    with pytest.raises(D.HardStop):
        D.check_guard_log()


# ---- review of 43ccfd6 and the Fable rulings (REVIEW.md 9.1): B1, J-a, J-b, B2, B3, I11, I12, I13, I17, B1d, I4, A2 ----
def _row(d):
    return {"passed": d.passed, "checks": d.checks, "numbers": d.numbers, "flags": d.flags}


def test_b1_guard_pass_test_follows_the_definition():
    """persistence margin = mean(guard - 0.5 main), gains negative: passes when the margin is <= 0 (the guard keeps at
    least half the gain), resolved only at 2 SE (review 5.2, the two synthetic cases). J-a: three values."""
    from sfm import stats
    rng = np.random.default_rng(3)
    main = -6e-4 + rng.normal(0, 2e-4, 20000)
    survives = main + rng.normal(0, 5e-5, 20000)                       # P: the guard keeps 100 % of the gain
    artefact = rng.normal(0, 5e-5, 20000)                              # F: the guard keeps 0 %
    d1, d0 = stats.guard_persistence(main, survives), stats.guard_persistence(main, artefact)
    assert d1.numbers["margin"] <= 0 and d0.numbers["margin"] > 0
    g1, g0 = R.guard_decision(_row(d1)), R.guard_decision(_row(d0))
    assert (g1["passed"], g1["resolved"], g1["verdict"]) == (True, True, "passed")
    assert (g0["passed"], g0["resolved"], g0["verdict"]) == (False, True, "failed")
    noise = rng.normal(0, 2e-4, 30)
    noise = noise - noise.mean() - 1e-5                                # margin -1e-5 with SE ~ 3.6e-5
    weak = stats.guard_persistence(main[:30], 0.5 * main[:30] + noise)   # a pass that is not resolved
    gw = R.guard_decision(_row(weak))
    assert gw["passed"] and not gw["resolved"] and gw["verdict"] == "undecided"


def test_j_a_j_b_three_values_and_the_g_region_sign():
    from sfm import stats
    assert [R.rule_guard_part(p, r) for p, r in ((True, True), (False, True), (True, False), (False, False))] == \
        ["passed", "failed", "undecided", "undecided"]
    rng = np.random.default_rng(5)
    main = -6e-4 + rng.normal(0, 1e-5, 20000)                          # a resolved gain
    for share, want in ((0.2, "passed"), (0.8, "failed"), (0.5, "undecided")):
        flagged = share * main + rng.normal(0, 1e-6, 20000)
        margin = stats.paired(flagged - 0.5 * main, np.zeros_like(main))  # >= 0 iff share <= 1/2 (gains negative)
        v = R.rule_g_region(float(main.mean()), margin)
        assert v["verdict"] == want, (share, v)
    assert R.rule_g_region(+1e-4, stats.paired(np.full(10, 1.0), np.zeros(10)))["passed"] is False   # no gain
    assert R.combine_guards(["passed", "passed", "passed"]) == "passed"
    assert R.combine_guards(["passed", "undecided", "passed"]) == "undecided"
    assert R.combine_guards(["undecided", "failed", None]) == "failed"
    assert R.combine_guards(["passed", "passed", None]) == "not evaluated"


@pytest.mark.skipif(not (H200 / "registry.jsonl").exists(), reason="H200 machine directory not present")
def test_guard_verdicts_on_the_real_h200_rows():
    """REVIEW.md 9.2: E5_s2 (passed, unresolved) is undecided; b0h_seed0 and s2_seed15 (failed, resolved) fail."""
    rows = {}
    for ln in (H200 / "registry.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("stage") == "m2b_mlp" and r.get("guards"):
            rows[r["name"]] = r
    for name, want in (("E5_s2", "undecided"), ("b0h_seed0", "failed"), ("s2_seed15", "failed")):
        g = rows[name]["guards"]["vs_mlp"]
        assert R.guard_decision(g["gk_persistence"])["verdict"] == want, name
        assert R.guard_decision(g["bulk_persistence"])["verdict"] == want, name


def test_i13_replication_literal_with_the_alternative_column():
    p0 = {"mean": -6e-4, "z": -4.0}
    rep = R.rule_replication(p0, {"mean": -3e-4, "z": -2.0}, {"mean": -5.5e-4, "z": -2.5})
    assert rep["replicated"] and not rep["replicated_alternative_se"] and rep["disagree"]   # -> to the user
    rep = R.rule_replication(p0, {"mean": -3e-4, "z": -2.0}, {"mean": -5.5e-4, "z": -3.1})
    assert rep["replicated"] and rep["replicated_alternative_se"] and not rep["disagree"]
    assert not R.rule_replication(p0, {"mean": -2.4e-4, "z": -2.0}, {"mean": -6e-4, "z": -4})["replicated"]  # half
    assert not R.rule_replication(p0, {"mean": -3e-4, "z": -2.0}, {"mean": -4.9e-4, "z": -4})["replicated"]  # mean
    assert not R.rule_replication({"mean": +1e-4, "z": 1}, {"mean": -6e-4, "z": -5},
                                  {"mean": -6e-4, "z": -5})["same_sign"]
    assert rep["constants"] == {"half": -2.5e-4, "mean": -5e-4, "alternative_z": -3.0}
    assert R.rule_replication(p0, None, None) is None


def test_i11_primary_comparison_option_i_with_the_other_readings_beside_it():
    assert R.I11_READING == "i"
    v = R.rule_primary_comparison({"mean": -4e-4, "z": -2.5}, {"mean": -2e-4, "z": -1.5}, {"mean": -3.5e-4, "z": -2.3})
    assert v["option_i"] is False and v["option_iii"] is True and v["counts"] is False      # (i) decides
    assert R.between_note(-4e-4) == "between −3 and −5" and R.between_note(-6e-4) is None
    assert R.between_note(-2e-4) is None
    v = R.rule_primary_comparison({"mean": -6e-4, "z": -3.5}, {"mean": -3e-4, "z": -2}, {"mean": -5.5e-4, "z": -3.2})
    assert v["option_i"] and v["option_iii"] and v["counts"] and not v["replication_disagree"]
    assert v["option_ii"].startswith("descriptive")


def test_i12_part_c_against_numpy():
    rng = np.random.default_rng(7)
    ref = _ta(18 + rng.normal(0, 0.1, 5000))
    ctrls = {n: _ta(ref.values + off + rng.normal(0, 3e-4, 5000)) for n, off in
             zip(lib.CONTROL_NAMES, (1.0e-4, 1.04e-4, 0.31e-4))}
    run = _ta(ref.values - 0.5e-4 + rng.normal(0, 3e-4, 5000))
    b0 = R.b0_mean(ctrls)
    p = R.parts(run, ref, b0, ctrls)
    d = run.values.astype(np.float64) - np.mean([c.values.astype(np.float64) for c in ctrls.values()], axis=0)
    z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(p["vs_b0"]["mean"] - d.mean()) < 1e-12 and abs(p["vs_b0"]["z"] - z) < 1e-9
    assert p["c_b0"] == bool(z < -1)
    assert set(p["vs_controls"]) == set(lib.CONTROL_NAMES)
    sd = np.std([(c.values.astype(np.float64) - ref.values).mean() for c in ctrls.values()], ddof=1)
    assert abs(R.controls_seed_sd(ctrls, ref) - sd) < 1e-12


def test_i17_one_se_eligibility_on_both_seeds():
    rng = np.random.default_rng(11)
    base = 18 + rng.normal(0, 0.1, 4000)
    def near():                                                        # Delta 1e-5 < SE ~ 1.6e-5: eligible
        n = rng.normal(0, 1e-3, 4000)
        return base + 1e-5 + (n - n.mean())
    k = {"B6a": 3841, "B6b": 4897, "B6c": 13825}
    mk = lambda arrs: [(c, f"{c}_s", k[c], a) for c, a in zip(("B6a", "B6b", "B6c"), arrs)]  # noqa: E731
    o = R.rule_one_se({0: mk([base, base + 0.01, near()]), 1: mk([base + 0.01, near(), base])})
    assert o["per_seed"][0]["eligible"] == ["B6a", "B6c"] and o["per_seed"][1]["eligible"] == ["B6b", "B6c"]
    assert o["per_seed"][0]["selected"] == "B6a" and o["per_seed"][1]["selected"] == "B6b"
    assert o["eligible_on_both_seeds"] == ["B6c"] and o["combined"] == "B6c" and o["per_seed_selections_differ"]
    o = R.rule_one_se({0: mk([base, base + 0.01, base + 0.01]), 1: mk([base + 0.01, base + 0.01, base])})
    assert o["combined"] is None and o["statement"].startswith("no automatic selection - to Gate 1")
    o = R.rule_one_se({0: mk([base, None, base]), 1: mk([base, base, base])})
    assert o["statement"].startswith("not evaluated")


def test_promotion_values_and_the_b1d_anomaly():
    ok = {"a_gain": True, "b_z": True, "c_b0": True}
    rep = {"replicated": True}
    assert R.promotion(ok, rep, "passed", False) == "promoted"
    assert R.promotion(ok, rep, "undecided", False).startswith("undecided - Gate 1")      # J-a
    assert R.promotion(ok, rep, None, False).startswith("undecided - Gate 1")
    assert R.promotion(ok, rep, "failed", False).startswith("not promoted")
    assert R.promotion(ok, {"replicated": False}, "passed", False) == "not promoted"
    assert R.promotion({**ok, "c_b0": None}, rep, "passed", False) == "not promoted"     # B0 incomplete
    assert R.promotion(ok, rep, "passed", True).startswith("anomaly - Gate 1")           # ruling B1d
    assert R.b1d_anomaly(ok, {"a_gain": False, "b_z": False, "c_b0": True})
    assert not R.b1d_anomaly(ok, ok) and not R.b1d_anomaly({**ok, "a_gain": False}, None)
    assert R.B1D_REPLICATION_LABEL == "replicated over starting networks (one teacher)"


def test_rulings_listed_with_their_status():
    for k in ("I11", "I12", "I13", "I17", "J-a", "J-b", "B1d", "I4"):
        assert k in R.RULINGS and R.RULINGS[k]["function"]
    assert "confirm" in R.RULINGS["I11"]["status"] and "confirm" in R.RULINGS["I4"]["status"]
    assert R.I4_OPTION == "alpha"


def test_b2_rerun_pair_of_record_and_b3_scope():
    sm = dict(D.spec_map())
    for n in ("B2b_s0", "B2a_s0"):
        x = D.x2_spec(sm[n])
        sm[x.name] = x
    assert R.pair_of_record("B2b_s0", "B2a_s0", sm) == ("B2b_s0_x2", "B2a_s0_x2", True)
    assert R.pair_of_record("B2c_s0", "B2a_s0", sm) == ("B2c_s0", "B2a_s0", False)   # converged: originals
    assert R.pair_of_record("B2b_s1", "B2a_s1", sm) == ("B2b_s1", "B2a_s1", False)
    rec = R.records(sm)
    assert rec["B2b"] == ["B2b_s0_x2", "B2b_s1"] and rec["B2c"] == ["B2c_s0", "B2c_s1"]   # mixed for B2b
    assert rec["B1d"] == ["B1d_m11", "B1d_m12"]
    pairs = R.comparison_pairs(sm, rec)
    assert [(r, c) for _, r, c, _ in pairs["B2b"]] == [("B2b_s0_x2", "B2a_s0_x2"), ("B2b_s1", "B2a_s1")]
    lab = R.labels(sm, rec, pairs)
    assert lab["B2b_s0"].startswith("not converged, NOT used by the rules")
    assert lab["B2a_s0"].startswith("comparator of record for") and "B2c_s0" in lab["B2a_s0"] \
        and "B4f_s0" in lab["B2a_s0"]                                   # the converged primaries keep the original
    assert "B2b_s1" not in lab and "B2a_s1" not in lab
    assert R.record_name("B2b_s0", sm) == "B2b_s0_x2" and R.record_name("B2b_s1", sm) == "B2b_s1"
    assert sm["B2b_s0_x2"].rung == "B2b" and sm["B2b_s0_x2"].cls == "rerun"


def test_a2_i_scoring_check_is_a_hard_stop(monkeypatch):
    """Fable ruling 9.1, A2 (i) boundary case: the fit's last validation mean and the registry's mean of the same
    weights must agree to 1e-6; a finite fit value with a non-finite registry score is a failed scoring check."""
    ok = D.scoring_check("x", 18.1180805, 18.1180805)
    assert ok["consistent"] and ok["abs_diff"] == 0.0
    D.scoring_check("x", 18.1180805, 18.1180805 + 9e-7)
    with pytest.raises(D.HardStop, match="differ by"):
        D.scoring_check("x", 18.1180805, 18.1180805 + 2e-6)
    with pytest.raises(D.HardStop, match="non-finite"):
        D.scoring_check("x", 18.1180805, reg_error="3 non-finite per-event validation NLL(s)")
    with pytest.raises(D.HardStop):
        D.scoring_check("x_ema", float("nan"), 18.1)
    both = D.scoring_check("x_ema", float("nan"), reg_error="non-finite")
    assert both["consistent"] is None                                  # the stream's own outcome: a divergence
    monkeypatch.setattr(D.registry, "live_stream", lambda: {"points": {"K": 64, "seed": 12345, "batch": 128,
                                                                         "fold": "iid"}})
    D.configure(False)
    D.check_scoring_settings(D.fit_config(D.spec_map()["B2a_s0"]))     # the fit validates on the registry's points
    with pytest.raises(D.HardStop, match="cannot be made"):
        D.check_scoring_settings(replace(D.fit_config(D.spec_map()["B2a_s0"]), k_val=16))


def test_i4_alpha_kl_record_and_beta_prepared_outside_the_default_list():
    lines = ["field_distill from a to b: KL 1.2000e-01 -> 3.0000e-03", "  warm start step     0  KL 1.200e-01",
             "  warm start step  1200  KL 9.000e-03"]
    k = D.kl_record(lines, {"kl_warm_start": {"kl_initial": 0.12, "kl_final": 0.003, "steps": 6000}})
    assert k["trajectory"] == [{"step": 0, "kl": 0.12}, {"step": 1200, "kl": 0.009}] and k["final"] == 0.003
    assert len(k["log_lines"]) == 3
    comp = D.companion_specs()
    assert [(s.name, s.kwargs["start"], s.kwargs["kl_steps"], s.cls, s.data_free) for s in comp] == [
        ("B1d_m11_kl12000", "s2_seed11", 12000, "companion", True),
        ("B1d_m12_kl12000", "s2_seed12", 12000, "companion", True)]
    assert "b1d_companions" not in D.DEFAULT_PHASES and "b1d_companions" in D.PHASES
    assert D.PHASE_ORDER.index("b1d_companions") == D.PHASE_ORDER.index("b1d") + 1
    assert set(D.DEFAULT_PHASES) == set(D.PHASE_ORDER) - {"b1d_companions"}
    D.check_companion_pins()                                           # pinned in the committed file
    assert not (set(D.companion_map()) & set(D.spec_map()))            # never part of the list
    rng = np.random.default_rng(2)
    a6 = _ta(18 + rng.normal(0, 0.1, 3000))
    assert R.rule_b1d_convergence(a6, _ta(a6.values + 0.3e-4))["of_record"] == "6000"
    assert R.rule_b1d_convergence(a6, _ta(a6.values - 0.7e-4))["of_record"] == "12000"


@pytest.mark.skipif(not (H200 / "registry.jsonl").exists(), reason="H200 machine directory not present")
def test_report_functions_against_numpy_on_the_real_h200_arrays():
    """REVIEW.md 5.3 / 9.2 numbers from the real H200 batch-1 arrays (validation only; CPU; read-only)."""
    rows = {}
    for ln in (H200 / "registry.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("stage") == "m2b_mlp" and r.get("stream"):
            rows[r["name"]] = r
    arr = lambda n: registry.row_array(rows[n], results_dir=paths.REPO_RESULTS)  # noqa: E731
    need = ["E5_s2", "s2_seed13", "s2_seed14", *lib.CONTROL_NAMES]
    if not all((paths.REPO_RESULTS / rows[n]["val_nll_path"]).exists() for n in need if n in rows):
        pytest.skip("H200 arrays not present")
    rs = sorted((H200 / "refsets").glob("*/mlp_val_nll.npy"))
    ref = provenance.load_tagged(rs[0], expect_machine_id=H200.name)
    ctrls = {n: arr(n) for n in lib.CONTROL_NAMES}
    b0 = R.b0_mean(ctrls)
    e5 = R.parts(arr("E5_s2"), ref, b0, ctrls)
    assert round(e5["vs_mlp"]["mean"] * 1e4, 4) == -0.8258 and round(e5["vs_mlp"]["se"] * 1e4, 4) == 1.3548
    assert round(e5["vs_b0"]["mean"] * 1e4, 4) == -1.6155 and round(e5["vs_b0"]["z"], 3) == -1.140
    c2 = R.parts(arr("b0h_seed2"), ref, b0, ctrls)
    assert round(c2["vs_b0"]["mean"] * 1e4, 4) == -0.4761 and round(c2["vs_b0"]["z"], 3) == -1.209
    assert round(R.controls_seed_sd(ctrls, ref) * 1e4, 2) == 0.41
    both = R.two_seed(arr("s2_seed14"), arr("s2_seed13"), ref)
    assert round(both["mean"] * 1e4, 4) == 1.1350 and round(both["se"] * 1e4, 4) == 1.3963
    m = np.mean([arr("s2_seed14").values.astype(np.float64), arr("s2_seed13").values], axis=0) - ref.values
    assert abs(both["mean"] - m.mean()) < 1e-12


def test_driver_run_as_a_script_is_one_module_instance(tmp_path):
    """Run as a script the driver is __main__; the report imports it as run_m2b_list. Both must be the same module
    (one CTX, one HardStop class), else the guards phase reads the wrong stage and a hard stop raised in the report
    is not exit 3 (found by the smoke on the H200 test copy)."""
    import subprocess
    code = (f"import sys, runpy\nsys.path.insert(0, {str(MARTIN / 'scripts')!r})\n"
            "sys.argv = ['run_m2b_list.py', '--help']\n"
            f"try:\n    runpy.run_path({str(MARTIN / 'scripts' / 'run_m2b_list.py')!r}, run_name='__main__')\n"
            "except SystemExit:\n    pass\n"
            "import run_m2b_list_report as R\n"
            "m = sys.modules['run_m2b_list']\n"
            "print('SAME', R.D is m, m.__name__, R.D.HardStop is m.HardStop)\n")
    env = {**os.environ, "SFM_OUT_ROOT": str(tmp_path / "out")}
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert "SAME True __main__ True" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]


def test_a2_dirty_clone_at_the_end_is_a_hard_stop(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(paths, "OUT_ROOT", tmp_path / "out")             # pristine mode
    monkeypatch.setattr(machine, "git_state", lambda: {"commit": "x", "clean": False, "n_changed": 2})
    monkeypatch.delenv("SFM_ALLOW_DIRTY_CLONE", raising=False)
    with pytest.raises(SystemExit) as ex:
        lib.finish({})
    assert ex.value.code == lib.CLONE_DIRTY_EXIT
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("HARD STOP (PLAN.md section 23, item 13) - clone")
    monkeypatch.setenv("SFM_ALLOW_DIRTY_CLONE", "1")                     # laptop dry runs only
    assert lib.finish({})["git_at_end"]["clean"] is False
