"""Tests of scripts/sr_post.py (post-processing of a chunked PySR search) without Julia.

A fake seed directory is built the way the runner writes it (spec_snapshot.json, hall_of_fame.json, predictions npz,
hof snapshots, progress.json); the "search's own predictions" are NumPy evaluations of the same strings.  torch is
imported by the round trip (this process is not a search process).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sfm.sr import operators as O
from sfm.sr import search as S
from sfm.sr import translate as T

MARTIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MARTIN / "scripts"))
import sr_post  # noqa: E402

EQS = [
    "tanh(zt)",
    "tanh(zt * 1.2) * 0.5",
    "(tanh(zt * 1.2) * 0.5) - x",                 # gauge duplicate of the previous member (same loss, larger C)
    "(ppow(zt, rh) * 0.1) + tanh(zt)",            # ppow exponent not a constant -> dropped
    "tanh(tanh(zt * 1.2) * 0.5)",                 # tanh inside tanh -> dropped
    "(tanh((zt * 1.2) + -1.0) * 0.5) + (linv(rh * 0.15) * -0.4)",
]


def _seed_dir(tmp_path, *, corrupt_pred=False, template=False):
    rng = np.random.default_rng(1)
    S_, n = 4, 50
    zt = rng.uniform(-0.7, 5.5, S_ * n)
    rh = rng.uniform(0, 30, S_ * n)
    x = np.repeat(np.linspace(-0.9, 0.9, S_), n)
    X = np.stack([zt, rh, x], axis=1)
    y = 0.5 * np.tanh(1.2 * zt - 1.0) - 0.4 / (1 + (0.15 * rh) ** 2) + 0.3 * x + rng.normal(0, 0.01, len(x))
    w = rng.uniform(0.5, 1.5, len(x))
    np.savez(tmp_path / "t.npz", X=X, y=y, weights=w, variable_names=np.array(["zt", "rh", "x"]))
    names = ["zt", "rh", "x"]
    raw = S.resolve_search_config({"step": "s", "target": "t", "data": "t.npz", "output_root": "runs",
                                   "budget_minutes": 1, "tiers": [5, 12, 30],
                                   "search_config": {"kwargs": {"stage": "stage1", "ndim": 3, "loss": "gauge",
                                                                **({"template": "s1_tanh_sum"} if template else {})}}})
    spec = S.parse_spec(raw, base_dir=tmp_path)
    d = spec.seed_dir(0)
    (d / S.SNAPSHOT_DIR).mkdir(parents=True)
    (d / S.PRED_DIR).mkdir()
    sha = S.sha256_file(tmp_path / "t.npz")
    S.atomic_write_json(d / S.SPEC_SNAPSHOT, {"identity": spec.identity(12, sha), "spec": spec.raw,
                                               "spec_path": str(tmp_path / "s.toml")})
    if template:
        subs = [{"u": "#1 * 1.2", "v": "#2 * 0.1", "w": "#2 * -0.05"},
                {"u": "(#1 * 1.2) + -1.0", "v": "#2 * 0.1", "w": "linv(#2 * 0.15) * -0.8"}]
        eqs = ["; ".join(f"{k} = {v}" for k, v in s.items()) for s in subs]
        trees = [T.parse_template("s1_tanh_sum", s) for s in subs]
        cs = [sum(O.tree_complexity(T.parse(v)) for v in s.values()) for s in subs]
    else:
        subs = [None] * len(EQS)
        eqs = EQS
        trees = [T.parse(e) for e in eqs]
        cs = [O.tree_complexity(t) for t in trees]
    env = {k: X[:, j] for j, k in enumerate(names)}
    preds = np.stack([np.asarray(T.evaluate(t, env), dtype=np.float64) + np.zeros(len(x)) for t in trees])
    losses = [O.gauge_loss_numpy(p, y, w, x) for p in preds]
    rows = [{"complexity": c, "loss": lo, "loss_str": repr(lo), "equation": e,
             **({"subexpressions": s} if s else {})} for c, lo, e, s in zip(cs, losses, eqs, subs)]
    if corrupt_pred:
        preds[1] += 1e-3
    np.savez(d / S.PRED_DIR / "chunk003_x.npz", rows=np.arange(len(x)), pred_o0=preds, ok_o0=np.ones(len(eqs), bool),
             complexity_o0=np.array(cs), loss_state_o0=np.array(losses), loss_csv_o0=np.array(losses),
             equation_o0=np.array(eqs, dtype=str))
    S.atomic_write_json(d / S.HOF_JSON, {"chunks_completed": 4, "predictions": "predictions/chunk003_x.npz",
                                         "outputs": [{"output": 0, "equations": rows}]})
    hist = []
    for k in range(4):
        with open(d / S.SNAPSHOT_DIR / f"chunk{k:03d}_2026_hall_of_fame.csv", "w") as f:
            f.write("Complexity,Loss,Equation\n")
            for r in rows[: len(rows) - 3 + k]:         # the last snapshot adds the best member
                f.write(f'{r["complexity"]},{r["loss"]!r},"{r["equation"]}"\n')
        hist.append({"index": k})
    S.atomic_write_json(d / S.PROGRESS_FILE, {"chunks_completed": 4, "tiers": [5, 12, 30], "history": hist})
    return d


def test_post_scan_dedup_round_trip_and_plateau(tmp_path):
    d = _seed_dir(tmp_path)
    doc = sr_post.post_seed(d, log=lambda *_: None)
    s = doc["outputs"][0]["summary"]
    assert doc["passed"], doc["failures"]
    assert s["n_members"] == len(EQS)
    assert s["n_violating"] == 2
    msgs = " | ".join(m for v in s["violating"] for m in v["violations"])
    assert "exponent is not a constant" in msgs and "tanh nested" in msgs
    assert s["n_gauge_duplicates"] == 1 and s["members_with_gauge_terms"] == 1
    # the duplicate does not appear on the de-duplicated front; violating members neither
    front_eqs = [m["equation"] for m in s["front_after_scan_and_dedup"]]
    assert EQS[2] not in front_eqs and EQS[3] not in front_eqs and EQS[4] not in front_eqs
    assert s["round_trip_all_passed"] and s["round_trip_max_diff"] <= 1e-12
    assert s["loss_max_rel_diff"] <= 1e-12 and s["complexity_all_ok"]
    dup = doc["outputs"][0]["members"][2]
    assert dup["dropped_gauge_terms"] == ["- x"] and dup["effective_complexity"] == dup["complexity"] - 2
    assert (d / "post" / "post_chunk003.json").exists()
    pl = json.loads((d / "post" / S.POST_PLATEAU_FILE).read_text())
    assert pl["chunks_completed"] == 4 and len(pl["history"]) == 4
    assert pl["plateau"]["plateau"] is False          # the last snapshot adds a better member at tier 30
    st = S.plateau_status(d)
    assert st["loss_rule_source"].startswith("sr_post") and st["loss_rule"] is False


def test_post_round_trip_detects_a_wrong_prediction(tmp_path):
    d = _seed_dir(tmp_path, corrupt_pred=True)
    doc = sr_post.post_seed(d, log=lambda *_: None)
    assert not doc["passed"]
    assert any("round trip" in f for f in doc["failures"])
    assert doc["outputs"][0]["members"][1]["round_trip"]["reference_max_diff"] == pytest.approx(1e-3, rel=1e-6)


def test_post_template(tmp_path):
    d = _seed_dir(tmp_path, template=True)
    doc = sr_post.post_seed(d, log=lambda *_: None)
    s = doc["outputs"][0]["summary"]
    assert doc["passed"], doc["failures"]
    assert doc["template"] == "s1_tanh_sum" and s["n_violating"] == 0 and s["complexity_all_ok"]
    assert s["round_trip_all_passed"]


def test_loss_kind():
    assert sr_post.loss_kind({}) == ("l2", None)
    assert sr_post.loss_kind({"loss_function": O.gauge_loss_julia(2)}) == ("gauge", 2)
    assert sr_post.loss_kind({"elementwise_loss": O.WL2_ELEMENTWISE}) == ("wl2", None)
