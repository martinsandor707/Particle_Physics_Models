"""Tests of the machine / stream tags (sfm.machine, sfm.provenance) and of their use in sfm.registry, sfm.train and
sfm.data (transfer/REVIEW_STATUS.md section 4; PLAN.md section 13, A6 revised; the H200 constraints of 25 Nov 2026).

Synthetic data and temporary result directories only. The real test cache is never opened: the test-access checks
are exercised only on their refusal paths (no LOCK.json naming this machine exists, or a path override is set), and
the real results/test_access.log and results/registry.jsonl are checked to be unchanged.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import common as C
from sfm import data, machine, metric, paths, policy, provenance, registry, stats, train
from sfm.registry import RegistryEnv, ValidationReference, ValidationSet
from test_train import Tiny, interrupt_at, Interrupt, make_showers

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parent
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _sha(p):
    return provenance.sha256_file(p) if Path(p).exists() else None


@pytest.fixture(scope="module", autouse=True)
def campaign_files_unchanged():
    """The tracked laptop records must stay byte-identical whatever these tests do."""
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


def fake_tag(**kw):
    """The live tag with some fields replaced (settings only in the tests below; identity changes use foreign_tag)."""
    return machine.machine_tag().with_(**kw)


def foreign_tag(deterministic: bool = True, **extra):
    """A machine tag guaranteed to differ from the LIVE machine on any machine (H200, laptop, CPU): derived from the
    live tag, with the GPU name, launch grid and SM count changed relative to their live values - never a literal
    that a real machine could have (the H200 run of 5a showed why)."""
    t = machine.machine_tag(deterministic=deterministic)
    f = t.fields
    changed = {"gpu_name": f"{f['gpu_name']} [foreign test machine]",
               "launch_grid": (f.get("launch_grid") or 0) + 8, "sm_count": (f.get("sm_count") or 0) + 1, **extra}
    u = t.with_(**changed)
    assert u.id != t.id and u.fields["launch_grid"] != f.get("launch_grid") and u.fields != t.fields
    return u


# The identities of the two machines of this campaign (reported by the user / read on the laptop), used only to show
# that the scenarios below do not depend on which machine runs them (test_scenarios_are_machine_independent).
H200_IDENTITY = {"gpu_name": "NVIDIA H200", "sm_count": 132, "max_threads_per_sm": 2048, "launch_grid": 1056,
                 "torch": "2.14.0+cu130", "cuda": "13.0", "cudnn": 92400, "python": "3.13.12", "tf32_matmul": False,
                 "tf32_cudnn": False}
LAPTOP_IDENTITY = {"gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU", "sm_count": 24, "max_threads_per_sm": 1536,
                   "launch_grid": 144, "torch": "2.14.0+cu130", "cuda": "13.0", "cudnn": 92400, "python": "3.12.13",
                   "tf32_matmul": False, "tf32_cudnn": False}


# ----------------------------------------------------------------------------------------------------------------------
# sfm.machine
# ----------------------------------------------------------------------------------------------------------------------
def _scenario_tag_id():
    t = machine.machine_tag()
    f = t.fields
    for k in ("gpu_name", "launch_grid", "torch", "cuda", "python", "tf32_matmul", "tf32_cudnn", "deterministic",
              "draw"):
        assert k in f
    assert f["tf32_matmul"] is False and f["tf32_cudnn"] is False          # switched off at import (E10)
    assert t.id == machine.machine_tag().id                                 # stable
    # every identity field changes the id - each change derived from the LIVE value, so it is a change on any machine
    for change in ({"launch_grid": (f.get("launch_grid") or 0) + 8}, {"python": f"{f['python']}.9"},
                   {"tf32_cudnn": not f["tf32_cudnn"]}, {"gpu_name": f"{f['gpu_name']} [other]"},
                   {"sm_count": (f.get("sm_count") or 0) + 1}, {"torch": f"{f['torch']}+other"},
                   {"cudnn": (f.get("cudnn") or 0) + 1}):
        assert t.with_(**change).id != t.id, change
    # review B1: per-computation settings stay in the tag (pairing) but do not change the machine id
    for change in ({"deterministic": False}, {"draw": "emulated"}, {"cublas_workspace_config": ":4096:8"}):
        assert t.with_(**change).id == t.id and t.with_(**change).fields != t.fields, change
    assert machine.machine_tag(deterministic=False).id == t.id
    assert "/" not in t.id and " " not in t.id


def test_machine_tag_fields_and_id():
    _scenario_tag_id()


def test_host_id_is_not_the_host_name():
    import socket
    h = machine.host_id()
    assert len(h) == 12 and socket.gethostname() not in h
    assert "host_id=" in machine.short_label() and socket.gethostname() not in machine.short_label()


def test_no_cuda_is_a_hard_failure_and_cpu_needs_explicit_opt_in():
    code = ("import sys; sys.path[:0] = [%r, %r]\n"
            "from sfm import machine\n"
            "try:\n    machine.startup('x'); print('STARTED')\nexcept machine.NoCUDA:\n    print('REFUSED')\n"
            "import os; os.environ['SFM_ALLOW_CPU'] = '1'\n"
            "machine.require_cuda('fit', allow_cpu_env=True); print('CPU_OPT_IN_OK')\n"
            "try:\n    machine.startup('script'); print('STARTED2')\nexcept machine.NoCUDA:\n    print('REFUSED2')\n"
            % (str(MARTIN / "src"), str(MARTIN)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("SFM_ALLOW_CPU", None)
    out = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split() == ["REFUSED", "CPU_OPT_IN_OK", "REFUSED2"]


def test_caps_defaults_follow_the_device_size(monkeypatch):
    monkeypatch.setattr(machine, "_caps_applied", {})
    monkeypatch.delenv("SFM_GPU_MEM_GB", raising=False)                     # the launcher exports these (A1)
    monkeypatch.delenv("SFM_CPU_THREADS", raising=False)
    c = machine.apply_caps()
    if c["device_total_gb"] and c["device_total_gb"] <= machine.LARGE_DEVICE_GB:
        assert c["gpu_mem_cap_gb"] is None and c["cpu_threads"] is None     # laptop: no cap unless asked
    elif c["device_total_gb"]:
        assert c["gpu_mem_cap_gb"] == 10.0 and c["cpu_threads"] == 4


def test_pristine_mode_refuses_writes_inside_the_clone(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(paths, "OUT_RESULTS", tmp_path / "out" / "shape-form-martin" / "results")
    assert paths.pristine_mode()
    with pytest.raises(PermissionError):
        paths.check_output_path(paths.REPO_RESULTS / "registry.jsonl")
    with pytest.raises(PermissionError):
        train.atomic_write_json({"x": 1}, paths.REPO_RESULTS / "m2b_mlp" / "never.json")
    assert not (paths.REPO_RESULTS / "m2b_mlp" / "never.json").exists()
    paths.check_output_path(paths.MARTIN_DIR / "cache" / "x.so")                # the ignored cache is allowed
    paths.check_output_path(tmp_path / "out" / "a.json")
    assert paths.portable(tmp_path / "out" / "a" / "b.json") == "<out>/a/b.json"
    assert paths.portable(paths.MARTIN_DIR / "x") == "<repo>/shape-form-martin/x"
    s = paths.scrub_str(json.dumps({"p": str(tmp_path / "out" / "q"), "r": str(paths.MARTIN_DIR / "y")}))
    assert str(tmp_path) not in s and str(paths.REPO_ROOT) not in s and "<out>/q" in s
    monkeypatch.setenv("SFM_USER_CACHE", str(tmp_path / "ucache"))          # outside clone and output root
    monkeypatch.setenv("SFM_GUARD_AREA", str(tmp_path))
    s = paths.scrub_str(json.dumps({"c": str(tmp_path / "ucache" / "tmp"), "a": str(tmp_path / "other")}))
    assert str(tmp_path) not in s and "<cache>/tmp" in s and "<area>/other" in s
    assert paths.portable(tmp_path / "ucache" / "x") == "<cache>/x"


def test_dirty_clone_is_refused_in_pristine_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(machine, "git_state", lambda: {"commit": "x", "clean": False, "n_changed": 3})
    monkeypatch.delenv("SFM_ALLOW_DIRTY_CLONE", raising=False)
    with pytest.raises(machine.DirtyClone):
        machine.require_clean_clone("t")
    monkeypatch.setattr(machine, "git_state", lambda: {"commit": "x", "clean": True, "n_changed": 0})
    assert machine.require_clean_clone("t")["clean"]


# ----------------------------------------------------------------------------------------------------------------------
# sfm.provenance
# ----------------------------------------------------------------------------------------------------------------------
def _pts(K=64, grid=1056, fp="f" * 64, subset="all"):
    return provenance.points_tag(grid=grid, K=K, split_fingerprint=fp, subset=subset)


def test_tagged_arrays_save_load_and_never_overwrite(tmp_path):
    t = machine.machine_tag()
    a = np.arange(10, dtype=np.float32)
    p = tmp_path / "x" / "val_nll.npy"
    ta = provenance.save_tagged(p, a, _pts(), t, label="x")
    assert provenance.sidecar_path(p).name == "val_nll.tag.json"
    back = provenance.load_tagged(p)
    assert np.array_equal(back.values, a) and back.machine_id == t.id and back.sha256 == ta.sha256
    provenance.save_tagged(p, a, _pts(), t)                                     # identical: kept
    with pytest.raises(FileExistsError):
        provenance.save_tagged(p, a + 1, _pts(), t)
    with pytest.raises(provenance.MixedTags):
        provenance.load_tagged(p, expect_machine_id="other")
    q = tmp_path / "y" / "val_nll.npy"                                          # untagged (a laptop record copied)
    q.parent.mkdir()
    np.save(q, a)
    with pytest.raises(provenance.UntaggedArray):
        provenance.load_tagged(q)
    np.save(p, a + 2)                                                           # replaced behind the sidecar's back
    with pytest.raises(provenance.UntaggedArray):
        provenance.load_tagged(p)


def _scenario_delta_refuses():
    t = machine.machine_tag()
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=100).astype(np.float32), rng.normal(size=100).astype(np.float32)
    A = provenance.TaggedArray.make(x, _pts(), t)
    B = provenance.TaggedArray.make(y, _pts(), t)
    p = provenance.paired_delta(A, B)
    q = stats.paired(x, y)
    assert (p.mean, p.se, p.n) == (q.mean, q.se, q.n)
    for other in (provenance.TaggedArray.make(y, _pts(grid=144), t),               # other fold points (synthetic
                  provenance.TaggedArray.make(y, _pts(K=1024), t),                 #  points, both sides non-live)
                  provenance.TaggedArray.make(y, _pts(fp="e" * 64), t),            # other split
                  provenance.TaggedArray.make(y, _pts(), foreign_tag()),           # another machine
                  provenance.TaggedArray.make(y, _pts(), fake_tag(draw="emulated")),          # other draw
                  provenance.TaggedArray.make(y, _pts(), fake_tag(deterministic=False))):     # other setting
        with pytest.raises(provenance.MixedTags):
            provenance.paired_delta(A, other)
        with pytest.raises(provenance.MixedTags):
            provenance.paired_difference(A, other)
    with pytest.raises(provenance.MixedTags):
        provenance.check_pairable(A, y)                                         # a bare array is never paired


def test_the_delta_function_refuses_mixed_tags():
    _scenario_delta_refuses()


def test_tolerance_check_reports_and_flags():
    a = np.zeros(1000, dtype=np.float32) + 18.0
    b = a.copy()
    b[[3, 7]] += np.float32(6e-5)
    tc = provenance.tolerance_check(a, b)
    assert tc["strict_pass"] and tc["p3_flag_events_above_5e-5"] == 2 and not tc["bit_identical"]
    assert {e["index"] for e in tc["largest"]} == {3, 7} and "se" not in tc
    b[11] += np.float32(2e-3)
    tc = provenance.tolerance_check(a, b)
    assert not tc["strict_pass"] and not tc["relaxed_pass"]


def test_weights_hash_ignores_file_metadata():
    m1, m2 = tiny(1), tiny(1)
    assert provenance.weights_sha256(m1.state_dict()) == provenance.weights_sha256(m2.state_dict())
    with torch.no_grad():
        m2.a[0, 0] += 1e-6
    assert provenance.weights_sha256(m1.state_dict()) != provenance.weights_sha256(m2.state_dict())


def test_laptop_record_lookup_verifies_hashes(tmp_path):
    rel = "results/m0_parity/mlp_val_nll_ref1056.npy"
    q = tmp_path / "laptop_records" / "shape-form-martin" / rel
    q.parent.mkdir(parents=True)
    q.write_bytes(b"not the record")
    r = provenance.laptop_record(rel, quarantine=tmp_path / "laptop_records", allow_original=False)
    assert r["where"] == "quarantine" and r["verified"] is False and r["expected_sha256"]
    with pytest.raises(PermissionError):
        provenance.laptop_record("results/m0_parity/mlp_test_nll.npy")         # test-derived: never opened
    r = provenance.laptop_record(rel, quarantine=tmp_path / "none", allow_original=False)
    assert r["path"] is None


# ----------------------------------------------------------------------------------------------------------------------
# registry with tags
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def reg(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("registry rows need CUDA")
    val = make_showers(50, 21, C.DEVICE)
    val.ev["zb"][:5] = 800.0
    en = np.arange(100, 100 + len(val), dtype=np.int64)
    ref = ValidationReference(event_number=en, fingerprint=registry.fingerprint(val), n_events=len(val), source="t")
    bench = {"mlp": tiny(1), "formula": tiny(2)}
    return {"val": val, "en": en, "ref": ref, "bench": bench, "tmp": tmp_path}


@cuda
def test_no_reference_set_no_fallback(reg):
    env = RegistryEnv(results_dir=reg["tmp"] / "results")
    m0 = reg["tmp"] / "results" / "m0_parity"                                   # untagged arrays at the old paths
    m0.mkdir(parents=True)
    with metric.deterministic():
        for b, mdl in reg["bench"].items():
            np.save(m0 / f"{b}_val_nll.npy", C.test_nll(mdl, reg["val"]))
    with pytest.raises(registry.NoReferenceSet):
        registry.evaluate(tiny(3), "c", "s", ValidationSet(reg["val"], reg["en"]), env=env, reference=reg["ref"])
    assert not env.registry_path.exists()


def _scenario_refset_of_another_machine(reg):
    other = foreign_tag()
    env_o = RegistryEnv(results_dir=reg["tmp"] / "results", machine=other)
    res = registry.create_reference_set(ValidationSet(reg["val"], reg["en"]), env=env_o, reference=reg["ref"],
                                        bench_models=lambda: reg["bench"], guards=False, log=lambda *a: None, parity={"verdict": "PASS", "source": "unit test"})
    assert res["status"] == "created" and other.id in res["dir"]
    env = RegistryEnv(results_dir=reg["tmp"] / "results")
    with pytest.raises(registry.NoReferenceSet):
        registry.evaluate(tiny(3), "c", "s", ValidationSet(reg["val"], reg["en"]), env=env, reference=reg["ref"])
    # the set is read-only and a second creation compares instead of overwriting
    d = Path(res["dir"])
    assert not os.access(d / "mlp_val_nll.npy", os.W_OK) and not os.access(d, os.W_OK)
    again = registry.create_reference_set(ValidationSet(reg["val"], reg["en"]), env=env_o, reference=reg["ref"],
                                          bench_models=lambda: reg["bench"], guards=False, log=lambda *a: None, parity={"verdict": "PASS", "source": "unit test"})
    assert again["status"] == "exists_identical"
    for p in list(d.iterdir()) + [d]:                                           # let pytest clean up
        p.chmod(0o755 if p.is_dir() else 0o644)


@cuda
def test_reference_set_of_another_machine_is_not_used(reg):
    _scenario_refset_of_another_machine(reg)


def _scenario_mixed_streams(reg):
    vs = ValidationSet(reg["val"], reg["en"])
    root = reg["tmp"] / "results"
    env = RegistryEnv(results_dir=root)
    registry.create_reference_set(vs, env=env, reference=reg["ref"], bench_models=lambda: reg["bench"], guards=False,
                                  log=lambda *a: None, parity={"verdict": "PASS", "source": "unit test"})
    rows = [registry.evaluate(tiny(s), f"c{s}", "s", vs, env=env, reference=reg["ref"]) for s in (3, 4)]
    live = registry.live_stream(env, reg["ref"])["stream_id"]
    assert registry.registered_row("s", "c3", stream_id=live, path=env.registry_path)["val_nll_sha256"] == \
        rows[0]["val_nll_sha256"]
    # a row on another machine's stream, written into the same file
    other = foreign_tag()
    env_o = RegistryEnv(results_dir=root, machine=other, registry_path=env.registry_path)
    registry.create_reference_set(vs, env=env_o, reference=reg["ref"], bench_models=lambda: reg["bench"],
                                  guards=False, log=lambda *a: None, parity={"verdict": "PASS", "source": "unit test"})
    registry.evaluate(tiny(5), "c5", "s", vs, env=env_o, reference=reg["ref"])
    # an untagged (laptop, schema 1) row appended by hand
    legacy = {"name": "c3", "stage": "s", "k": 1, "C": None, "val_mean": 1.0, "val_nll_path": "s/c3/val_nll.npy",
              "val_nll_sha256": "x", "checkpoint_sha256": None, "timestamp": "2026-11-24T00:00:00+00:00"}
    with open(env.registry_path, "a") as f:
        f.write(json.dumps(legacy) + "\n")
    allrows = registry.read_registry(env.registry_path)
    keys = {(r["name"], registry.row_stream_id(r)) for r in registry.latest(allrows)}
    assert ("c3", None) in keys and ("c3", live) in keys                       # keyed by stream: no shadowing
    got = registry.registered_row("s", "c3", stream_id=live, path=env.registry_path)
    assert got["timestamp"] == rows[0]["timestamp"] and registry.row_stream_id(got) == live
    with pytest.raises(registry.UntaggedRow):
        registry.row_array(legacy, root)
    with pytest.raises(provenance.MixedTags):                                  # two streams selected together
        registry.select_1se(stage="s", path=env.registry_path, results_dir=root)
    sel = registry.select_1se(stage="s", path=env.registry_path, results_dir=root, stream=live)
    assert {r["name"] for r in sel.table} == {"c3", "c4"}
    a3 = registry.row_array(rows[0], root)
    a5 = registry.row_array([r for r in allrows if r["name"] == "c5"][0], root)
    with pytest.raises(provenance.MixedTags):                                  # the delta function refuses
        provenance.paired_delta(a3, a5)
    # a laptop-style untagged array placed at the live tagged path is not picked up
    live_path = root / rows[1]["val_nll_path"]
    live_path.unlink()
    np.save(live_path, np.zeros(len(reg["val"]), dtype=np.float32))
    with pytest.raises(provenance.UntaggedArray):
        registry.row_array(rows[1], root)
    with pytest.raises(provenance.UntaggedArray):
        registry.select_1se(stage="s", path=env.registry_path, results_dir=root, stream=live)


@cuda
def test_mixed_streams_untagged_rows_and_live_paths(reg):
    _scenario_mixed_streams(reg)


@cuda
def test_evaluate_refuses_replace_and_the_legacy_registry(reg):
    vs = ValidationSet(reg["val"], reg["en"])
    env = RegistryEnv(results_dir=reg["tmp"] / "results")
    registry.create_reference_set(vs, env=env, reference=reg["ref"], bench_models=lambda: reg["bench"], guards=False,
                                  log=lambda *a: None, parity={"verdict": "PASS", "source": "unit test"})
    with pytest.raises(ValueError):
        registry.evaluate(tiny(3), "c", "s", vs, env=env, reference=reg["ref"], replace=True)
    with pytest.raises(TypeError):
        registry.evaluate(tiny(3), "c", "s", vs, env=env, reference=reg["ref"], bench_models=lambda: reg["bench"])
    env_l = RegistryEnv(results_dir=reg["tmp"] / "results", registry_path=registry.LEGACY_REGISTRY)
    with pytest.raises(PermissionError):
        registry.evaluate(tiny(3), "c", "s", vs, env=env_l, reference=reg["ref"])


# ----------------------------------------------------------------------------------------------------------------------
# train: no fit continued or returned across machines
# ----------------------------------------------------------------------------------------------------------------------
def _scenario_fit_refuses(tmp_path):
    tr, va = make_showers(120, 1, C.DEVICE), make_showers(40, 2, C.DEVICE)
    cfg = train.FitConfig(steps=20, lr=0.05, lr_end=0.005, batch=32, eval_every=10, ckpt_every=10, k_val=16,
                          name="x", out_dir=str(tmp_path / "x"))
    with pytest.raises(Interrupt):
        train.fit(Tiny().to(C.DEVICE), tr, va, cfg, log=None, callback=interrupt_at(13))
    ck = tmp_path / "x" / "checkpoint.pt"
    obj = torch.load(ck, weights_only=False)
    assert obj["machine_tag"]["id"] == machine.machine_tag(deterministic=False).id
    obj["machine_tag"] = foreign_tag(deterministic=False).to_dict()            # another machine, same settings
    torch.save(obj, ck)
    with pytest.raises(train.MachineMismatch):
        train.fit(Tiny().to(C.DEVICE), tr, va, cfg, log=None)
    obj.pop("machine_tag")
    torch.save(obj, ck)
    with pytest.raises(train.MachineMismatch):                                 # untagged (laptop) checkpoint
        train.fit(Tiny().to(C.DEVICE), tr, va, cfg, log=None)
    cfg2 = train.FitConfig(steps=10, lr=0.05, batch=32, eval_every=10, k_val=16, name="y", out_dir=str(tmp_path / "y"))
    train.fit(Tiny().to(C.DEVICE), tr, va, cfg2, log=None)
    done = json.loads((tmp_path / "y" / "fit_done.json").read_text())
    assert done["machine_tag"]["id"] == machine.machine_tag(deterministic=False).id and done["weights_sha256"]
    train.fit(Tiny().to(C.DEVICE), tr, va, cfg2, log=None)                     # same machine: loads
    done.pop("machine_tag")
    (tmp_path / "y" / "fit_done.json").write_text(json.dumps(done))
    with pytest.raises(train.MachineMismatch):                                 # a finished run of unknown origin
        train.fit(Tiny().to(C.DEVICE), tr, va, cfg2, log=None)


@cuda
def test_fit_refuses_checkpoints_and_done_markers_of_another_machine(tmp_path):
    _scenario_fit_refuses(tmp_path)


# ----------------------------------------------------------------------------------------------------------------------
# the five scenarios above, run as if the live machine were the H200 and as if it were the laptop
# ----------------------------------------------------------------------------------------------------------------------
@cuda
@pytest.mark.parametrize("who", ["h200", "laptop"])
def test_scenarios_are_machine_independent(who, monkeypatch, reg, tmp_path):
    """The live tag is replaced by the H200's (id h200-g1056-97b197a7, reported from step 5a) or the laptop's identity
    (settings and the computation itself stay those of this process); every scenario must still refuse what it
    refuses. Shows on the laptop that the suite does not depend on the machine it runs on."""
    identity = H200_IDENTITY if who == "h200" else LAPTOP_IDENTITY
    real = machine.machine_tag

    def live(deterministic=True, draw="native", device=0):
        return real(deterministic=deterministic, draw=draw, device=device).with_(**identity)

    monkeypatch.setattr(machine, "machine_tag", live)
    monkeypatch.setattr(policy, "LAPTOP_MACHINE_IDS", (live().id,))          # the A5 start check is not the subject
    assert live().identity == identity
    assert live().id == ("h200-g1056-97b197a7" if who == "h200" else "rtx-4060-laptop-gpu-g144-76c0f59a")
    assert foreign_tag().id != live().id
    _scenario_tag_id()
    _scenario_delta_refuses()
    sub = dict(reg, tmp=tmp_path / "refset_case")
    _scenario_refset_of_another_machine(sub)
    sub = dict(reg, tmp=tmp_path / "mixed_case")
    _scenario_mixed_streams(sub)
    _scenario_fit_refuses(tmp_path / "fit_case")


# ----------------------------------------------------------------------------------------------------------------------
# test access (spec points 6 and 14): only refusal paths are exercised
# ----------------------------------------------------------------------------------------------------------------------
def test_test_access_refused_without_lock_and_with_overrides(monkeypatch, tmp_path):
    log = paths.TEST_ACCESS_LOG
    before = _sha(log)
    assert not data.lock_file().exists()                                       # no LOCK.json on this machine yet
    with pytest.raises(PermissionError):
        data.load_sets(("test",), device="cpu", allow_test=True, reason="unit test: must be refused")
    monkeypatch.setenv("SFM_RESULTS_DIR", str(tmp_path))
    with pytest.raises(PermissionError, match="override"):
        data.check_test_access_allowed()
    monkeypatch.delenv("SFM_RESULTS_DIR")
    monkeypatch.setenv("SFM_CACHE_DIR", str(tmp_path))
    with pytest.raises(PermissionError, match="override"):
        data.check_test_access_allowed()
    monkeypatch.delenv("SFM_CACHE_DIR")
    lock = tmp_path / "LOCK.json"
    lock.write_text(json.dumps({"machine_of_record": {"machine_id": foreign_tag().id}}))   # never the live id
    if torch.cuda.is_available():
        with pytest.raises(PermissionError, match="names machine"):
            data.check_test_access_allowed(lock)
    assert _sha(log) == before


def test_test_access_line_has_host_id_gpu_and_grid():
    line = data.test_access_line("why", script="final_test.py")
    parts = line.split("\t")
    assert parts[1:3] == ["final_test.py", "why"]
    assert any(p.startswith("host_id=") for p in parts) and any(p.startswith("gpu=") for p in parts)
    assert any(p.startswith("grid=") for p in parts) and any(p.startswith("machine=") for p in parts)
