"""Tests of sfm.sr.search (chunked, resumable PySR runner).

The fast tests need no Julia: spec parsing, hall-of-fame CSV parsing, tier bookkeeping, the loss-plateau rule on
synthetic histories, atomic progress writes, status without Julia, and the no-torch import check (run in a fresh
subprocess so pytest's own imports do not interfere).  Julia-dependent tests are skipped unless SFM_RUN_JULIA_TESTS=1
(they start Julia with at most 8 threads and run for about a minute).
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sfm.sr import search as S

MARTIN = Path(__file__).resolve().parents[1]
RUN_JULIA = os.environ.get("SFM_RUN_JULIA_TESTS") == "1"
julia = pytest.mark.skipif(not RUN_JULIA, reason="Julia tests run only with SFM_RUN_JULIA_TESTS=1")


def _spec_dict(tmp_path, **kw):
    X = np.random.default_rng(0).normal(size=(50, 2))
    np.savez(tmp_path / "t.npz", X=X, y=X[:, 0] ** 2, variable_names=np.array(["a", "b"]))
    d = {"step": "s", "target": "t", "data": "t.npz", "output_root": "runs", "budget_minutes": 2.0,
         "chunk_minutes": 0.5, "maxsize": 30, "threads": 4,
         "operators": {"binary_operators": ["+", "*", "ppow(x, y) = (1 + abs(x))^y"],
                       "unary_operators": ["exp", "linv(x) = 1 / (1 + x * x)"],
                       "constraints": {"ppow": [-1, 1]}, "nested_constraints": {"exp": {"exp": 0}},
                       "complexity_of_operators": {"exp": 2},
                       "sympy_mappings": {"ppow": "(1 + Abs(x))**y", "linv": "1/(1 + x**2)"}}}
    d.update(kw)
    return d


def test_import_does_not_pull_torch_or_julia():
    code = ("import sys; sys.path.insert(0, %r); import sfm.sr.search; "
            "print(int('torch' in sys.modules), int('juliacall' in sys.modules), int('pysr' in sys.modules))"
            % str(MARTIN / "src"))
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["0", "0", "0"]


def test_parse_spec_defaults_and_paths(tmp_path):
    spec = S.parse_spec(_spec_dict(tmp_path), base_dir=tmp_path)
    assert spec.data == (tmp_path / "t.npz").resolve()
    assert spec.seed_dir(2) == (tmp_path / "runs" / "s" / "t" / "seed2").resolve()
    assert spec.budget_s == 120.0 and spec.chunk_s == 30.0
    assert spec.tiers == (12, 20, 30)
    assert spec.resolved_populations() == 12            # 3 x threads
    assert spec.resolved_populations(20) == 60
    assert spec.pysr["batching"] is False and spec.pysr["precision"] == 64
    assert spec.operators["constraints"]["ppow"] == (-1, 1)
    assert S.custom_operator_definitions(spec.operators) == ["ppow(x, y) = (1 + abs(x))^y",
                                                             "linv(x) = 1 / (1 + x * x)"]
    assert S.operator_name("linv(x) = 1 / (1 + x * x)") == "linv"
    assert S.operator_name("function f(x) x end") == "f"


def test_parse_spec_rejects_bad_input(tmp_path):
    with pytest.raises(S.SpecError):
        S.parse_spec(_spec_dict(tmp_path, pysr={"warm_start": False}), base_dir=tmp_path)
    with pytest.raises(S.SpecError):
        S.parse_spec(_spec_dict(tmp_path, operators={"binary_operators": ["+"], "bogus": 1}), base_dir=tmp_path)
    with pytest.raises(S.SpecError):
        S.parse_spec(_spec_dict(tmp_path, template={"combine": "f(a, b)", "expressions": ["f"]},
                                loss={"loss_function": "function l(t, d, o) 0.0 end"}), base_dir=tmp_path)
    with pytest.raises(S.SpecError):
        S.parse_spec(_spec_dict(tmp_path, loss={"loss_function": "a", "elementwise_loss": "b"}), base_dir=tmp_path)


def test_module_vocabulary_is_deferred(tmp_path):
    spec = S.parse_spec(_spec_dict(tmp_path, operators={"module": "sfm.sr.operators", "attr": "STAGE1",
                                                        "constraints": {"exp": 9}}), base_dir=tmp_path)
    assert spec.operators == {} and spec.operators_source["module"] == "sfm.sr.operators"
    assert spec.operators_source["overrides"] == {"constraints": {"exp": 9}}
    with pytest.raises(S.SpecError):
        S.SeedRun(spec, 0)
    filled = spec.with_operators({"binary": ["+"], "unary": ["exp"]})
    assert filled.operators["binary_operators"] == ["+"]


def test_toml_and_overrides(tmp_path):
    _spec_dict(tmp_path)  # writes t.npz
    (tmp_path / "s.toml").write_text('step = "s"\ntarget = "t"\ndata = "t.npz"\nbudget_minutes = 1\nmaxsize = 12\ntiers = [6, 12]\n'
                                     '[operators]\nbinary = ["+", "*"]\n')
    raw = S.apply_overrides(S.load_spec_file(tmp_path / "s.toml"), threads=8, budget_min=3, seeds=[1, 2])
    spec = S.parse_spec(raw)
    assert spec.threads == 8 and spec.budget_s == 180 and spec.seeds == [1, 2]
    assert spec.data == (tmp_path / "t.npz").resolve()


def test_identity_ignores_soft_keys(tmp_path):
    a = S.parse_spec(_spec_dict(tmp_path), base_dir=tmp_path)
    b = S.parse_spec(_spec_dict(tmp_path, budget_minutes=9, threads=20, chunk_minutes=4, tiers=[10, 20]),
                     base_dir=tmp_path)
    c = S.parse_spec(_spec_dict(tmp_path, maxsize=35), base_dir=tmp_path)
    ia, ib, ic = (S._canon(x.identity(12, "h")) for x in (a, b, c))
    assert ia == ib and ia != ic


def test_hof_csv_parsing_and_tiers(tmp_path):
    p = tmp_path / "hall_of_fame.csv"
    p.write_text('Complexity,Loss,Equation\n1,0.5f0,"1.2"\n3,0.25,"a * 2.0"\n7,1.0e-3,"exp(a) + ""q"""\n'
                 '15,Inf,"b"\n')
    rows = S.read_hof_csv(p)
    assert [r["complexity"] for r in rows] == [1, 3, 7, 15]
    assert rows[0]["loss"] == 0.5 and rows[2]["loss"] == 1e-3 and math.isinf(rows[3]["loss"])
    assert rows[2]["equation"] == 'exp(a) + "q"'
    assert S.tier_bests(rows, (2, 5, 12)) == {"2": 0.5, "5": 0.25, "12": 1e-3}
    assert S.tier_bests([], (12,)) == {"12": None}
    assert S.parse_julia_float("1.5f-3") == pytest.approx(1.5e-3)


def test_hof_regressions_staircase():
    prev = [{"complexity": 1, "loss": 1.0}, {"complexity": 5, "loss": 0.5}]
    # a simpler member now beats the complexity-5 entry, which is no longer listed: not a regression
    new = [{"complexity": 1, "loss": 1.0}, {"complexity": 3, "loss": 0.4}]
    assert S.hof_regressions(prev, new) == []
    worse = [{"complexity": 1, "loss": 1.0}, {"complexity": 5, "loss": 0.6}]
    assert S.hof_regressions(prev, worse) == [{"complexity": 5, "prev": 0.5, "new": 0.6}]


def _hist(values):
    """values: list over chunks of {tier: loss} for one output."""
    return [{"index": i, "tier_best": [v]} for i, v in enumerate(values)]


def test_plateau_rule():
    tiers = (12, 20, 30)
    flat = {"12": 1.0, "20": 0.5, "30": 0.4}
    r = S.plateau_check(_hist([flat] * 3), tiers)
    assert r["plateau"] is None                                    # needs window + 1 = 4 chunks
    r = S.plateau_check(_hist([{"12": 2.0, "20": 1.0, "30": 0.9}] + [flat] * 3), tiers)
    assert r["plateau"] is False                                   # chunk 0 -> 3 improved a lot
    r = S.plateau_check(_hist([{"12": 2.0, "20": 1.0, "30": 0.9}] + [flat] * 4), tiers)
    assert r["plateau"] is True and r["ref_chunk"] == 1 and r["cur_chunk"] == 4
    almost = {"12": 0.995, "20": 0.5, "30": 0.4}                   # 0.5 % at tier 12
    assert S.plateau_check(_hist([flat, flat, flat, almost]), tiers)["plateau"] is True
    more = {"12": 1.0, "20": 0.5, "30": 0.39}                      # 2.5 % at tier 30
    assert S.plateau_check(_hist([flat, flat, flat, more]), tiers)["plateau"] is False
    # an expression appears at a tier that had none -> improvement
    none12 = {"12": None, "20": 0.5, "30": 0.4}
    assert S.plateau_check(_hist([none12, none12, none12, flat]), tiers)["plateau"] is False
    # multi-output: every output must have plateaued
    h = [{"index": i, "tier_best": [flat, {"12": 1.0 - 0.05 * i, "20": 0.5, "30": 0.4}]} for i in range(4)]
    assert S.plateau_check(h, tiers)["plateau"] is False


def test_atomic_progress_write(tmp_path):
    p = tmp_path / "progress.json"
    S.atomic_write_json(p, {"a": 1.0, "b": float("inf"), "c": np.float32(2.5), "d": (1, 2)})
    assert json.loads(p.read_text()) == {"a": 1.0, "b": None, "c": 2.5, "d": [1, 2]}
    S.atomic_write_json(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2}
    assert [q.name for q in tmp_path.iterdir()] == ["progress.json"]   # no temp files left


def test_val_nll_hook_and_status_without_julia(tmp_path):
    spec = S.parse_spec(_spec_dict(tmp_path), base_dir=tmp_path)
    d = spec.seed_dir(0)
    d.mkdir(parents=True)
    prog = S.new_progress(spec, 0, 1)
    prog.update(status="running", chunks_completed=4, elapsed_search_s=100.0,
                plateau={"plateau": True}, best_by_tier=[{"12": 0.1, "20": 0.05, "30": None}])
    S.atomic_write_json(d / S.PROGRESS_FILE, prog)
    st = S.plateau_status(d)
    assert (st["loss_rule"], st["val_nll_rule"], st["plateau"]) == (True, None, None)
    assert st["loss_rule_source"].startswith("runner")
    S.record_val_nll_check(d, front_improves_by_more_than_1se=False, detail={"se": 1e-4})
    assert S.plateau_status(d)["plateau"] is True
    # a current sr_post result (front after the scan and the gauge de-duplication) takes precedence
    S.atomic_write_json(d / S.POST_DIR / S.POST_PLATEAU_FILE, {"chunks_completed": 4, "plateau": {"plateau": False}})
    st = S.plateau_status(d)
    assert st["loss_rule"] is False and st["loss_rule_raw"] is True and st["loss_rule_source"].startswith("sr_post")
    S.atomic_write_json(d / S.POST_DIR / S.POST_PLATEAU_FILE, {"chunks_completed": 3, "plateau": {"plateau": False}})
    st = S.plateau_status(d)
    assert st["loss_rule"] is True and "stale" in st["loss_rule_source"]
    S.record_val_nll_check(d, front_improves_by_more_than_1se=True)
    assert S.plateau_status(d)["plateau"] is False
    txt = S.format_status(spec, [0, 1])
    assert "seed0" in txt and "seed1: not started" in txt


def test_status_cli_does_not_start_julia(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps(_spec_dict(tmp_path)))
    code = ("import sys, runpy; sys.argv = ['sr_run.py', %r, '--status']; "
            "runpy.run_path(%r, run_name='__main__')" % (str(tmp_path / "s.json"),
                                                          str(MARTIN / "scripts" / "sr_run.py")))
    wrapper = code + "\n"
    r = subprocess.run([sys.executable, "-B", "-c",
                        "import sys, atexit\natexit.register(lambda: print('JULIA', int('juliacall' in sys.modules)))\n"
                        + wrapper], capture_output=True, text=True)
    assert "not started" in r.stdout and "JULIA 0" in r.stdout


def test_npz_placeholders():
    src = "ends = {{npz:slice_ends}}; w = {{ npz:w }}"
    out = S.fill_npz_placeholders(src, {"slice_ends": np.array([3, 6]), "w": np.array([0.5])})
    assert out == "ends = (3, 6); w = (0.5,)"
    with pytest.raises(S.SpecError):
        S.fill_npz_placeholders("{{npz:missing}}", {})


def test_template_equation_split_and_merge_guesses():
    assert S.parse_template_equation("u = #1 * 2.0; v = cos(#2)", ["u", "v"]) == {"u": "#1 * 2.0", "v": "cos(#2)"}
    assert S.parse_template_equation("u = #1", ["u", "v"]) is None
    assert S.merge_guesses(["a"], ["b"], 1) == ["a", "b"]
    assert S.merge_guesses([["a"], ["b"]], [["c"], []], 2) == [["a", "c"], ["b"]]
    assert S.merge_guesses(None, None, 1) is None


def _sc_raw(tmp_path, **kw):
    d = {"step": "s", "target": "t", "data": "t.npz", "budget_minutes": 1.0, "search_config": {"kwargs": kw}}
    return d


@pytest.mark.parametrize("kw,loss_key,maxsize", [
    ({"stage": "stage1", "ndim": 3, "loss": "gauge"}, "loss_function", 30),
    ({"stage": "stage1", "ndim": 3, "loss": "gauge", "template": "s1_tanh_sum"}, "loss_function_expression", 30),
    ({"stage": "stage2", "ndim": 1, "loss": "wl2"}, "elementwise_loss", 18),
])
def test_search_config_interface_end_to_end(tmp_path, kw, loss_key, maxsize):
    """Every key of operators.search_config() reaches the place PySR needs it (no key dropped)."""
    from sfm.sr import operators as O
    cfg = O.search_config(**kw)
    raw = S.resolve_search_config(_sc_raw(tmp_path, **kw))
    spec = S.parse_spec(raw, base_dir=tmp_path)
    assert spec.maxsize == maxsize == cfg["maxsize"]
    assert list(spec.loss) == [loss_key] and spec.loss[loss_key] == cfg[loss_key]
    for k in ("binary_operators", "unary_operators", "constraints", "nested_constraints", "complexity_of_operators"):
        assert k in spec.operators
    assert spec.operators["binary_operators"] == cfg["binary_operators"]
    assert spec.operators["nested_constraints"] == cfg["nested_constraints"]
    for k in ("precision", "print_precision", "complexity_of_constants", "complexity_of_variables"):
        assert spec.pysr[k] == cfg[k]
    assert spec.pysr["print_precision"] == 17                       # not the runner's old 15
    if "template" in kw:
        assert spec.template["combine"] == cfg["template"]["combine"] and spec.template["name"] == kw["template"]
        assert "extra_sympy_mappings" not in spec.operators
        assert spec.pysr["batching"] is False
    else:
        assert spec.template is None
        names = {S.operator_name(o) for o in spec.operators["binary_operators"] + spec.operators["unary_operators"]}
        custom = {S.operator_name(o) for o in S.custom_operator_definitions(spec.operators)}
        assert custom <= set(spec.operators["extra_sympy_mappings"]) and custom <= names
    routed = (set(spec.operators) | set(spec.loss) | {"maxsize"} | set(spec.pysr)
              | ({"template"} if spec.template else set()))
    assert set(cfg) <= routed
    assert raw["search_config_resolved"]["keys"] == sorted(cfg)


def test_search_config_refuses_unknown_keys_and_conflicts(tmp_path, monkeypatch):
    from sfm.sr import operators as O
    real = O.search_config
    monkeypatch.setattr(O, "search_config", lambda **kw: {**real(**kw), "new_key": 1})
    with pytest.raises(S.SpecError, match="new_key"):
        S.resolve_search_config(_sc_raw(tmp_path, stage="stage1"))
    monkeypatch.setattr(O, "search_config", real)
    with pytest.raises(S.SpecError, match="maxsize"):
        S.resolve_search_config({**_sc_raw(tmp_path, stage="stage1"), "maxsize": 25})
    with pytest.raises(S.SpecError, match="operators"):
        S.resolve_search_config({**_sc_raw(tmp_path, stage="stage1"), "operators": {"binary": ["+"]}})
    raw = S.resolve_search_config({**_sc_raw(tmp_path, stage="stage1"), "pysr": {"print_precision": 12}})
    assert raw["pysr"]["print_precision"] == 12
    assert raw["search_config_resolved"]["pysr_overridden_by_spec"] == ["print_precision"]


def test_print_precision_default_follows_precision(tmp_path):
    assert S.parse_spec(_spec_dict(tmp_path), base_dir=tmp_path).pysr["print_precision"] == 17
    assert S.parse_spec(_spec_dict(tmp_path, pysr={"precision": 32}), base_dir=tmp_path).pysr["print_precision"] == 9


def test_module_vocabulary_refuses_full_search_config():
    # fresh process: load_vocabulary refuses to run once torch is imported (other tests import it here)
    code = f"""
import sys
sys.path.insert(0, {str(MARTIN / 'scripts')!r}); sys.path.insert(0, {str(MARTIN / 'src')!r})
import sr_run
try:
    sr_run.load_vocabulary({{"module": "sfm.sr.operators", "attr": "search_config", "kwargs": {{}}}})
except sr_run.search.SpecError as e:
    print("REFUSED", "search_config" in str(e))
"""
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True).stdout
    assert "REFUSED True" in out


def test_search_process_imports_stay_torch_free(tmp_path):
    """Everything the search process imports before and while starting Julia (runner, vocabulary resolution through
    operators.search_config, spec parsing, target loading) leaves torch out of sys.modules."""
    X = np.zeros((4, 3))
    X[:, 2] = [0.1, 0.1, 0.2, 0.2]
    np.savez(tmp_path / "t.npz", X=X, y=np.zeros(4), weights=np.ones(4), variable_names=np.array(["zt", "rh", "x"]))
    code = f"""
import sys, runpy
sys.path.insert(0, {str(MARTIN / 'scripts')!r}); sys.path.insert(0, {str(MARTIN / 'src')!r})
import sr_run
from sfm.sr import search as S
raw = S.resolve_search_config({{"step": "s", "target": "t", "data": {str(tmp_path / 't.npz')!r},
    "budget_minutes": 1, "search_config": {{"kwargs": {{"stage": "stage1", "ndim": 3, "loss": "gauge"}}}}}})
spec = S.parse_spec(raw)
S.load_target(spec.data)
import sfm.sr.translate, sfm.sr.audit   # numpy-only modules the post-processing uses
print("TORCH", int("torch" in sys.modules), "JULIA", int("juliacall" in sys.modules))
"""
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True).stdout
    assert "TORCH 0 JULIA 0" in out


def test_output_redirect_is_noop_without_log():
    had_julia = "juliacall" in sys.modules
    with S.julia_output_to(None):
        pass
    assert ("juliacall" in sys.modules) == had_julia          # no log file -> Julia is not started


def test_load_target_shapes(tmp_path):
    X = np.zeros((10, 2))
    np.savez(tmp_path / "m.npz", X=X, y=np.zeros((10, 3)), weights=np.ones((10, 3)))
    t = S.load_target(tmp_path / "m.npz")
    assert t.nout == 3 and t.variable_names == ["x0", "x1"]
    np.savez(tmp_path / "bad.npz", X=X, y=np.zeros((10, 3)), weights=np.ones(10))
    with pytest.raises(S.SpecError):
        S.load_target(tmp_path / "bad.npz")


# ----------------------------------------------------------------------------------------------------------------
# Julia-dependent (skipped by default)
# ----------------------------------------------------------------------------------------------------------------
@julia
def test_julia_short_search_resume_and_extend(tmp_path):
    """Two search processes (a 1-chunk search, then an extension that must restore the checkpoint with the identical
    populations), then the post-processing in a third process.  The search process records `"torch" in
    sys.modules` itself (progress.json), which must stay False.  About 75 s at 4 threads."""
    d = _spec_dict(tmp_path, budget_minutes=0.4, chunk_minutes=0.4, min_chunk_seconds=5, maxsize=12, threads=4,
                   tiers=[5, 12], pysr={"verbosity": 1})
    (tmp_path / "s.json").write_text(json.dumps(d))
    cmd = [sys.executable, "-B", str(MARTIN / "scripts" / "sr_run.py"), str(tmp_path / "s.json")]
    subprocess.run(cmd, check=True, timeout=600)
    seed = tmp_path / "runs" / "s" / "t" / "seed0"
    p1 = json.loads((seed / "progress.json").read_text())
    assert p1["done"] and p1["chunks_completed"] >= 1 and (seed / "DONE.json").exists()
    assert p1["julia"]["nthreads"] == 4
    assert p1["torch_in_process"] is False and all(h["torch_in_process"] is False for h in p1["history"])
    assert p1["history"][0]["evals_per_s_mean"]                        # Julia's log reaches search.log
    subprocess.run(cmd + ["--budget-min", "0.8"], check=True, timeout=600)
    p2 = json.loads((seed / "progress.json").read_text())
    assert p2["chunks_completed"] > p1["chunks_completed"]
    r = p2["restores"][-1]
    assert r["method"] == "from_file" and not p2["fallback_used"] and r["same_as_chunk_end"] is True
    assert p2["history"][p1["chunks_completed"]]["hof_not_worse_than_prev_chunk"] is True
    assert p2["torch_in_process"] is False
    hof = json.loads((seed / "hall_of_fame.json").read_text())
    assert hof["done"] and hof["outputs"][0]["equations"] and hof["predictions"]
    post = subprocess.run([sys.executable, "-B", str(MARTIN / "scripts" / "sr_post.py"), "--seed-dir", str(seed)],
                          capture_output=True, text=True, timeout=300)
    assert post.returncode == 0, post.stdout + post.stderr
    doc = json.loads(sorted((seed / "post").glob("post_chunk*.json"))[-1].read_text())
    assert doc["outputs"][0]["summary"]["round_trip_all_passed"] is True
