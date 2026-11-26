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
        rs = registry.reference_set(need_guards=True)                   # licensed by a passing parity record
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


# ---- review of 43ccfd6: B1 (guard decision), B2 (rerun pairs), B3 (promotion scope), A2 (ii) ---------------------------
def test_b1_guard_pass_test_follows_the_definition():
    """persistence margin = mean(guard - 0.5 main), gains negative: passes when the margin is <= 0 (the guard keeps at
    least half the gain), resolved only at 2 SE. A gain that fully survives passes; a pure artefact fails."""
    from sfm import stats
    rng = np.random.default_rng(3)
    main = -6e-4 + rng.normal(0, 2e-4, 20000)
    survives = main + rng.normal(0, 5e-5, 20000)                       # the guard keeps 100 % of the gain
    artefact = rng.normal(0, 5e-5, 20000)                              # the guard keeps 0 %
    d1 = stats.guard_persistence(main, survives)
    d0 = stats.guard_persistence(main, artefact)
    assert d1.numbers["margin"] <= 0 and d0.numbers["margin"] > 0
    as_row = lambda d: {"passed": d.passed, "checks": d.checks, "numbers": d.numbers, "flags": d.flags}  # noqa: E731
    g1, g0 = R.guard_decision(as_row(d1)), R.guard_decision(as_row(d0))
    assert g1["passed"] and g1["resolved"] and g1["ok"]
    assert not g0["passed"] and not g0["ok"]
    weak = stats.guard_persistence(main[:30], main[:30] * 0.55)          # a pass that is not resolved
    gw = R.guard_decision(as_row(weak))
    assert gw["ok"] == R.rule_guard_part(gw["passed"], gw["resolved"])   # J-a: one place decides


def test_b2_rerun_pair_of_record_and_b3_scope(monkeypatch):
    sm = dict(D.spec_map())
    for n in ("B2b_s0", "B2a_s0"):
        x = D.x2_spec(sm[n])
        sm[x.name] = x
    assert R.pair_of_record("B2b_s0", "B2a_s0", sm) == ("B2b_s0_x2", "B2a_s0_x2", True)
    assert R.pair_of_record("B2c_s0", "B2a_s0", sm) == ("B2c_s0", "B2a_s0", False)   # converged: originals
    assert R.pair_of_record("B2b_s1", "B2a_s1", sm) == ("B2b_s1", "B2a_s1", False)
    assert R.superseded(sm) == {"B2b_s0", "B2a_s0"}
    assert R.record_name("B2b_s0", sm) == "B2b_s0_x2" and R.record_name("B2b_s1", sm) == "B2b_s1"
    assert sm["B2b_s0_x2"].rung == "B2b" and sm["B2b_s0_x2"].cls == "rerun"
    for k in ("I11", "I12", "I13", "I17", "J-a", "J-b"):
        assert k in R.PENDING_RULINGS                                    # every judgement item in one place


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
