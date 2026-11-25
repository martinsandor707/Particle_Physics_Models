"""Regression tests for the findings of transfer/ENABLING_REVIEW.md (A1-A5, B1-B3, C1-C5). Each test reproduces the
reviewer's failure mode and checks that the code now fails closed. Synthetic data and temporary directories only;
the real test split is never opened; results/registry.jsonl and results/test_access.log stay unchanged."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import common as C
from sfm import guard, machine, paths, policy, provenance, registry
from sfm.registry import RegistryEnv, ValidationReference, ValidationSet
from test_train import Tiny, make_showers

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parent
SCRIPTS = MARTIN / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
PAR = {"verdict": "PASS", "source": "unit test"}


def _sha(p):
    return provenance.sha256_file(p) if Path(p).exists() else None


@pytest.fixture(scope="module", autouse=True)
def campaign_files_unchanged():
    files = [paths.REPO_RESULTS / "registry.jsonl", paths.REPO_RESULTS / "test_access.log"]
    before = {f: _sha(f) for f in files}
    yield
    assert {f: _sha(f) for f in files} == before


def tiny(seed, scale=0.5):
    torch.manual_seed(seed)
    m = Tiny().to(C.DEVICE).eval()
    with torch.no_grad():
        m.a.normal_(0, scale)
        m.b.normal_(0, scale)
    return m


@pytest.fixture()
def reg(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("registry rows need CUDA")
    val = make_showers(40, 31, C.DEVICE)
    val.ev["zb"][:4] = 800.0
    en = np.arange(10, 10 + len(val), dtype=np.int64)
    ref = ValidationReference(event_number=en, fingerprint=registry.fingerprint(val), n_events=len(val), source="t")
    return {"vs": ValidationSet(val, en), "ref": ref, "bench": {"mlp": tiny(1), "formula": tiny(2)},
            "env": RegistryEnv(results_dir=tmp_path / "results")}


def _unseal(d: Path):
    for p in list(d.rglob("*")) + [d]:
        p.chmod(0o755 if p.is_dir() else 0o644)


# ---- A1: test_guard no longer depends on whether the pytest process is guarded --------------------------------------
def test_A1_guard_tests_pass_inside_a_guarded_pytest_process(tmp_path):
    area = tmp_path / "unrelated_area"
    area.mkdir()
    code = (f"import sys; sys.path[:0] = [{str(MARTIN / 'src')!r}, {str(MARTIN)!r}, {str(HERE)!r}]\n"
            "import sfm.paths, sfm.guard\nassert sfm.guard.active()\nimport pytest\n"
            f"sys.exit(pytest.main([{str(HERE / 'test_guard.py')!r}, '-q', '-p', 'no:cacheprovider']))")
    env = dict(os.environ, SFM_GUARD_AREA=str(area), SFM_OUT_ROOT=str(tmp_path / "out"), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]


# ---- A2: every computed check is part of the val_parity verdict ------------------------------------------------------
def _report(grid=144, status="created", a3=True, r1b=True, verified=True):
    rec = {"rel": "results/m0_parity/mlp_val_nll_ref1056.npy", "verified": verified, "where": "quarantine"}
    r1 = {"second_run_bit_identical": {"mlp": True, "formula": True},
          "targets": {"mlp_float32_mean": {"pass": True}, "formula_float32_mean": {"pass": True}},
          "records": {"mlp": rec, "formula": dict(rec)},
          "A3_native_vs_emulated": {"mlp": {"pass": a3}, "formula": {"pass": True}}}
    r1b_ = {b: {"strict_pass": r1b if b == "mlp" else True, "max_abs": 1e-5,
                "record": {"rel": f"results/m0_parity/{b}_val_nll.npy", "verified": True}} for b in ("mlp", "formula")}
    checks = {"R0": {"generator": {"pass": True}}, "R1": r1, "R1b_benchmarks": r1b_,
              "R3": {b: {"strict_pass": True, "max_abs": 2e-5} for b in ("mlp", "formula")},
              "R2": {"refset": {"status": status}, "targets": {"n_stratum": {"pass": True}}}}
    return {"checks": checks}


def test_A2_verdict_includes_refset_status_A3_R1b_and_record_identity():
    import val_parity as vp
    ok = lambda rep, grid=144: all(x[1] for x in vp.verdict_items(rep, grid, create_requested=True))  # noqa: E731
    assert ok(_report())
    assert not ok(_report(status="exists_DIFFERENT"))                 # (a)
    assert not ok(_report(a3=False))                                  # (b) fallback acceptance
    assert ok(_report(a3=False), grid=1056)                           # A3 only applies off grid 1056
    assert not ok(_report(r1b=False))                                 # (c)
    assert not ok(_report(verified=False))                            # (d) unverified laptop record
    rep = _report()
    del rep["checks"]["R2"]
    assert not ok(rep)                                                # requested but not created


# ---- A3: no reference set without a passing parity record; a differing recomputation blocks the set -----------------
@cuda
def test_A3_reference_set_needs_parity_and_disputes_block(reg):
    env, vs, ref = reg["env"], reg["vs"], reg["ref"]
    with pytest.raises(registry.ReferenceSetNotLicensed):
        registry.create_reference_set(vs, env=env, reference=ref, bench_models=lambda: reg["bench"], guards=False,
                                      log=lambda *a: None)
    with pytest.raises(registry.ReferenceSetNotLicensed):
        registry.create_reference_set(vs, env=env, reference=ref, bench_models=lambda: reg["bench"], guards=False,
                                      log=lambda *a: None, parity={"verdict": "FAIL"})
    res = registry.create_reference_set(vs, env=env, reference=ref, bench_models=lambda: reg["bench"], guards=False,
                                        log=lambda *a: None, parity=PAR)
    d = Path(res["dir"])
    assert json.loads((d / "refset.json").read_text())["parity"]["verdict"] == "PASS"
    registry.evaluate(tiny(3), "ok_before", "s", vs, env=env, reference=ref)
    other = {"mlp": tiny(7), "formula": reg["bench"]["formula"]}      # the recomputation now differs
    again = registry.create_reference_set(vs, env=env, reference=ref, bench_models=lambda: other, guards=False,
                                          log=lambda *a: None, parity=PAR)
    assert again["status"] == "exists_DIFFERENT"
    assert list(d.parent.glob(f"{d.name}.DIFFERENT*.json"))
    with pytest.raises(registry.ReferenceSetDisputed):
        registry.evaluate(tiny(3), "blocked", "s", vs, env=env, reference=ref)
    with pytest.raises(registry.ReferenceSetDisputed):
        registry.reference_set(env, ref)
    _unseal(d.parent)


@cuda
def test_A3_set_without_parity_record_is_refused(reg):
    env, vs, ref = reg["env"], reg["vs"], reg["ref"]
    res = registry.create_reference_set(vs, env=env, reference=ref, bench_models=lambda: reg["bench"], guards=False,
                                        log=lambda *a: None, parity=PAR, seal=False)
    mf = Path(res["dir"]) / "refset.json"
    m = json.loads(mf.read_text())
    m.pop("parity")
    mf.write_text(json.dumps(m))
    with pytest.raises(registry.ReferenceSetNotLicensed):
        registry.evaluate(tiny(3), "c", "s", vs, env=env, reference=ref)


# ---- A4: a check that cannot be made makes the re-score INCOMPLETE, never PASS --------------------------------------
def test_A4_rescore_missing_checks_are_incomplete():
    import rescore_registered as rs
    req = rs.required_checks("s1_seed11", 144, guarded=False, emulate=True)
    assert set(req) == {"laptop_registered_array", "R1b_emulated_144_vs_laptop"}
    assert rs.network_verdict({}, req)[0] == "INCOMPLETE"                              # records not found
    chk = {"laptop_registered_array": {"made": False, "reason": "laptop record not found"},
           "R1b_emulated_144_vs_laptop": {"made": False, "reason": "laptop record not found"}}
    v, not_made, failed = rs.network_verdict(chk, req)
    assert v == "INCOMPLETE" and set(not_made) == set(req) and not failed
    chk = {k: {"pass": True} for k in req}
    assert rs.network_verdict(chk, req)[0] == "PASS"
    chk["R1b_emulated_144_vs_laptop"] = {"pass": False}
    assert rs.network_verdict(chk, req)[0] == "FAIL"
    req3 = rs.required_checks("b0_seed0", 912, guarded=True, emulate=True)            # a third grid
    assert "R4_on_emulated_grid_1056" in req3
    assert rs.network_verdict({k: {"pass": True} for k in req3 if k != "R4_on_emulated_grid_1056"}, req3)[0] == \
        "INCOMPLETE"
    req1056 = rs.required_checks("b0_seed0", 1056, guarded=True, emulate=False)
    assert {"predicted_mean", "predicted_k1024_mean", "predicted_bulk_mean", "mass_monitors"} <= set(req1056)


# ---- A5: fail closed off the laptop of record ---------------------------------------------------------------------
@cuda
def test_A5_other_machine_needs_output_root_and_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "LAPTOP_MACHINE_IDS", ())                 # pretend: not the laptop of record
    monkeypatch.setattr(paths, "OUT_ROOT", paths.REPO_ROOT)
    monkeypatch.setattr(guard, "active", lambda: False)
    with pytest.raises(machine.UnguardedStart):
        machine.startup("t")
    with pytest.raises(machine.UnguardedStart):
        machine.require_guarded_start("sfm.train.fit")
    monkeypatch.setattr(paths, "OUT_ROOT", tmp_path / "out")             # output root outside, guard still off
    with pytest.raises(machine.UnguardedStart):
        machine.startup("t")
    monkeypatch.setattr(guard, "active", lambda: True)                    # both: allowed
    monkeypatch.setenv("SFM_ALLOW_DIRTY_CLONE", "1")
    assert machine.startup("t").id
    monkeypatch.setattr(policy, "LAPTOP_MACHINE_IDS", (machine.machine_tag().id,))
    monkeypatch.setattr(paths, "OUT_ROOT", paths.REPO_ROOT)
    monkeypatch.setattr(guard, "active", lambda: False)
    assert machine.require_guarded_start("t")["laptop_of_record"] is True


def test_A5_policy_names_this_laptop_by_machine_id_only():
    import socket
    text = (MARTIN / "src" / "sfm" / "policy.py").read_text()
    assert socket.gethostname() not in text
    assert all("-g" in i for i in policy.LAPTOP_MACHINE_IDS)


# ---- B1: the machine id ignores per-call settings; pairing still requires them equal ------------------------------
def test_B1_machine_id_vs_settings():
    t = machine.machine_tag()
    for kw in ({"deterministic": False}, {"draw": "emulated"}, {"cublas_workspace_config": ":4096:8"}):
        u = t.with_(**kw)
        assert u.id == t.id
        pts = provenance.points_tag(grid=144, K=64, split_fingerprint="f" * 64)
        a = provenance.TaggedArray.make(np.zeros(5, np.float32), pts, t)
        b = provenance.TaggedArray.make(np.zeros(5, np.float32), pts, u)
        with pytest.raises(provenance.MixedTags):
            provenance.paired_delta(a, b)
    assert set(t.identity) == set(machine.IDENTITY_KEYS) and "deterministic" in t.settings


# ---- B2: starting points verified and recorded ----------------------------------------------------------------------
def test_B2_verify_init(tmp_path):
    import run_m2b_lib as lib
    st = Tiny().state_dict()
    rr = tmp_path / "results"
    (rr / "m2b_mlp" / "s1_seed13").mkdir(parents=True)
    f = rr / "m2b_mlp" / "s1_seed13" / "model.pt"
    torch.save({"state": st, "meta": {"model": {"factory": "field"}}}, f)
    reg_f = tmp_path / "registry.jsonl"
    row = {"name": "s1_seed13", "stage": "m2b_mlp", "checkpoint_sha256": provenance.sha256_file(f),
           "timestamp": "2026-11-24T23:47:00+00:00", "val_nll_sha256": "x"}
    reg_f.write_text(json.dumps(row) + "\n")
    rec = lib.verify_init("s2_seed13", "from_run", {"run": "m2b_mlp/s1_seed13"}, repo_results=rr, legacy_registry=reg_f)
    assert rec["file_sha256"] == row["checkpoint_sha256"] and rec["weights_sha256"] == provenance.weights_sha256(st)
    assert rec["source_row"]["timestamp"] == row["timestamp"] and "laptop" in rec["trained_on"]
    with pytest.raises(SystemExit):                                     # not the pre-registered starting point
        lib.verify_init("s2_seed13", "from_run", {"run": "m2b_mlp/s1_seed14"}, repo_results=rr, legacy_registry=reg_f)
    with pytest.raises(SystemExit):
        lib.verify_init("s2_seed99", "from_run", {"run": "m2b_mlp/s1_seed13"}, repo_results=rr, legacy_registry=reg_f)
    torch.save({"state": {k: v + 1 for k, v in st.items()}}, f)       # the file was replaced
    with pytest.raises(SystemExit):
        lib.verify_init("s2_seed13", "from_run", {"run": "m2b_mlp/s1_seed13"}, repo_results=rr, legacy_registry=reg_f)
    m = Tiny()
    with pytest.raises(SystemExit):
        lib.check_built_weights(m, {"weights_sha256": "0" * 64})
    lib.check_built_weights(m, {"weights_sha256": provenance.weights_sha256(m.state_dict())})


def test_B2_mlp_std_start_is_checked_against_sha256sums():
    import run_m2b_lib as lib
    if not (paths.PION_RESULTS_DIR / "mlp_std.pt").exists():
        pytest.skip("mlp_std.pt not present")
    rec = lib.verify_init("b0h_seed0", "field_mlp_std", {})
    assert rec["file_sha256"] == lib.input_hashes()[lib.MLP_STD_REL]
    with pytest.raises(SystemExit):
        lib.verify_init("b0h_seed0", "from_run", {"run": "m2b_mlp/s1_seed11"})


# ---- B3: subset arrays bound to name and content; representatives and the three controls enforced ------------------
def test_B3_renamed_tagged_pair_is_refused(tmp_path):
    t = machine.machine_tag()
    pts = provenance.points_tag(grid=144, K=64, split_fingerprint="f" * 64)
    a = tmp_path / "s2_11-12.npy"
    provenance.save_tagged(a, np.arange(4, dtype=np.float32), pts, t)
    b = tmp_path / "s2_11-13.npy"
    b.write_bytes(a.read_bytes())
    provenance.sidecar_path(b).write_text(provenance.sidecar_path(a).read_text())
    with pytest.raises(provenance.UntaggedArray):
        provenance.load_tagged(b)


def test_B3_representatives_and_controls():
    import run_m2b_ensembles as ens
    import run_m2b_report as rep
    good = {"E2_s2_seeds11-12": True, "E3_s2_seeds11-13": True, "E4_s2_seeds11-14": True}
    assert ens.representatives_verdict(good) == (True, [])
    assert ens.representatives_verdict({**good, "E3_s2_seeds11-13": False})[0] is False
    assert ens.representatives_verdict({"E2_s2_seeds11-12": True})[0] is False
    t = machine.machine_tag()
    pts = provenance.points_tag(grid=144, K=64, split_fingerprint="f" * 64)
    arr = {n: provenance.TaggedArray.make(np.full(3, i, np.float32), pts, t) for i, n in enumerate(("b0h_seed0",
                                                                                                    "b0h_seed1"))}
    names, mean, complete, present = rep.b0_controls(arr)
    assert not complete and mean is None and names == [] and present == ["b0h_seed0", "b0h_seed1"]
    arr["b0h_seed2"] = provenance.TaggedArray.make(np.full(3, 2, np.float32), pts, t)
    arr["b0_seed0"] = provenance.TaggedArray.make(np.full(3, 9, np.float32), pts, t)    # laptop record: never in
    names, mean, complete, _ = rep.b0_controls(arr)
    assert complete and names == ["b0h_seed0", "b0h_seed1", "b0h_seed2"] and np.allclose(mean.values, 1.0)


# ---- C1: the deciding P3 test ----------------------------------------------------------------------------------------
def test_C1_cell_attribution():
    import p3_compare as p3
    rng = np.random.default_rng(0)
    H, K = 5, 8
    logrho = np.log(rng.dirichlet(np.ones(C.N_Z * C.N_R)).reshape(C.N_Z, C.N_R))
    iz, ir = rng.integers(0, C.N_Z, (H, K)), rng.integers(0, C.N_R, (H, K))
    w = np.full(H, 1 / H)
    lap = {"iz": iz, "ir": ir, "w": w, "logrho": logrho}
    same = p3.attribute_cells({"iz": iz.copy(), "ir": ir.copy()}, lap, recorded_diff=6e-5, K=K)
    assert same["n_points_cell_differs"] == 0 and same["verdict"].startswith("no differing cell")
    iz2 = iz.copy()
    iz2[2, 3] = (iz2[2, 3] + 1) % C.N_Z                                 # one point flips its z-bin
    g = lambda a, b: np.log(np.mean(np.exp(logrho[a, b]), 1))           # noqa: E731
    true_d = float(-np.sum(w * (g(iz2, ir) - g(iz, ir))))
    flip = p3.attribute_cells({"iz": iz2, "ir": ir}, lap, recorded_diff=true_d + 1e-6, K=K)
    assert flip["n_points_cell_differs"] == 1 and flip["verdict"] == "flip: explained"
    assert flip["differing_points"][0][:2] == [2, 3]
    wrong = p3.attribute_cells({"iz": iz2, "ir": ir}, lap, recorded_diff=true_d + 1e-3, K=K)
    assert wrong["verdict"] == "differing cells, NOT explained"


# ---- C3: launcher reports immediate failures; stop helper checks the process ----------------------------------------
def _launch(tmp_path, *cmd, grace="1"):
    env = dict(os.environ, SFM_OUT_ROOT=str(tmp_path / "out"), SFM_USER_CACHE=str(tmp_path / "cache"),
               SFM_GUARD_AREA="off", SFM_LAUNCH_GRACE=grace)
    return subprocess.run(["bash", str(SCRIPTS / "launch.sh"), *cmd], env=env, capture_output=True, text=True,
                          timeout=60, cwd=str(paths.REPO_ROOT))


def test_C3_launcher_and_stop_helper(tmp_path):
    p = _launch(tmp_path, "bad", str(tmp_path / "no_such_interpreter"), "-c", "1")
    assert p.returncode == 1 and "FAILED" in p.stderr
    p = _launch(tmp_path, "uvjob", "uv", "run", "python", "-c", "1")
    assert p.returncode == 2 and "refusing" in p.stderr
    p = _launch(tmp_path, "sleeper", sys.executable, "-c", "import time; time.sleep(30)")
    assert p.returncode == 0 and "started job" in p.stdout, p.stderr
    pidf = next((tmp_path / "out" / "jobs").glob("sleeper_*.pid"))
    jobf = pidf.with_suffix(".job")
    pid = int(pidf.read_text())
    assert f"pid={pid}" in jobf.read_text()
    good = jobf.read_text()
    jobf.write_text(good.replace("starttime=", "starttime=1"))         # as if the PID had been reused
    q = subprocess.run(["bash", str(SCRIPTS / "stop_job.sh"), str(pidf)], capture_output=True, text=True, timeout=30)
    assert q.returncode == 1 and "refusing" in q.stderr and Path(f"/proc/{pid}").exists()
    jobf.write_text(good)
    q = subprocess.run(["bash", str(SCRIPTS / "stop_job.sh"), str(pidf)], capture_output=True, text=True, timeout=30)
    assert q.returncode == 0 and "sent SIGTERM" in q.stdout
    exitf = pidf.with_suffix(".exit")
    for _ in range(100):
        if exitf.exists():
            break
        import time
        time.sleep(0.1)
    assert exitf.exists() and "# exit status" in pidf.with_suffix(".log").read_text()


# ---- C4 / C5 --------------------------------------------------------------------------------------------------------
def test_C4_git_state_and_env_record(monkeypatch):
    g = machine.git_state()
    assert {"commit", "clean", "upstream", "head_equals_upstream"} <= set(g)
    monkeypatch.setenv("SFM_ALLOW_DIRTY_CLONE", "1")
    monkeypatch.setenv("SFM_CPU_THREADS", "3")
    rec = machine.env_record()
    assert rec["SFM_ALLOW_DIRTY_CLONE"] == "1" and rec["SFM_CPU_THREADS"] == "3"


def test_C5_registered_row_refuses_untagged_lookup():
    with pytest.raises(ValueError):
        registry.registered_row("m2b_mlp", "b0_seed0", stream_id=None, path=registry.LEGACY_REGISTRY)
