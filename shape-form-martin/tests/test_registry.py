"""Tests of sfm.registry: refusal of anything but the validation split, append-only writing, paired numbers equal to
sfm.stats.paired, guard columns, reading helpers and the 1-SE selection.

Synthetic tests use a small synthetic "validation" set on common.DEVICE with its own ValidationReference and a
temporary results directory (RegistryEnv), so the campaign's registry is never touched; GPU use is tiny. The
real-data test only fingerprints the cached validation split on the CPU (no model is evaluated, no test file).
"""
import json

import numpy as np
import pytest
import torch

import common as C
from sfm import metric, registry, stats
from sfm.registry import NotValidationSet, RegistryEnv, ValidationReference, ValidationSet
from test_train import Tiny, make_showers

DEV = C.DEVICE


def tiny(seed, scale=0.5):
    torch.manual_seed(seed)
    m = Tiny().to(DEV).eval()
    with torch.no_grad():
        m.a.normal_(0, scale)
        m.b.normal_(0, scale)
    return m


@pytest.fixture()
def setup(tmp_path):
    val = make_showers(60, 11, DEV)
    val.ev["zb"][:6] = 800.0                                           # a few late-start ("short") events
    en = np.arange(5000, 5000 + len(val), dtype=np.int64)
    ref = ValidationReference(event_number=en, fingerprint=registry.fingerprint(val), n_events=len(val), source="test")
    bench_models = {"mlp": tiny(1), "formula": tiny(2)}
    (tmp_path / "m0").mkdir()
    files = {}
    with metric.deterministic():
        for b, m in bench_models.items():
            files[b] = tmp_path / "m0" / f"{b}_val_nll.npy"
            np.save(files[b], C.test_nll(m, val))
    env = RegistryEnv(results_dir=tmp_path / "results", bench_files=files)
    return {"val": val, "en": en, "ref": ref, "env": env, "bench_models": bench_models,
            "bench": {b: np.load(f) for b, f in files.items()}}


def run(s, model, name, stage="stage_a", **kw):
    return registry.evaluate(model, name, stage, ValidationSet(s["val"], s["en"]), meta=kw.pop("meta", None),
                             env=s["env"], reference=s["ref"], **kw)


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def test_evaluate_row_and_paired_numbers_equal_stats(setup):
    s = setup
    m = tiny(3)
    row = run(s, m, "cand", meta={"seed": 7, "config": {"lr": 1e-3}, "notes": "x", "something": 1})
    with metric.deterministic():
        ref = C.test_nll(m, s["val"])
    saved = np.load(s["env"].results_dir / "stage_a" / "cand" / "val_nll.npy")
    assert np.array_equal(saved, ref)                                  # the unmodified metric, bit-reproducible
    for b in ("mlp", "formula"):
        p = stats.paired(ref, s["bench"][b])
        assert row[f"vs_{b}"] == {"mean": p.mean, "se": p.se, "z": p.z, "n": p.n}
    assert row["val_mean"] == float(ref.astype(np.float64).mean()) and row["n_events"] == len(ref)
    assert row["k"] == m.n_params() and row["C"] is None and row["seed"] == 7
    assert row["config_hash"] == registry._config_hash({"lr": 1e-3}) and row["extra"] == {"something": 1}
    assert row["val_nll_path"] == "stage_a/cand/val_nll.npy" and row["guards"] is None
    assert row["git"]["commit"] is None or len(row["git"]["commit"]) == 40
    (only,) = lines(s["env"].registry_path)
    assert json.loads(only) == json.loads(json.dumps(row))


def test_registry_is_append_only(setup):
    s = setup
    path = s["env"].registry_path
    run(s, tiny(3), "cand")
    first = path.read_bytes()
    run(s, tiny(4), "cand2", meta={"k": 5, "C": 13})
    assert path.read_bytes().startswith(first) and len(lines(path)) == 2
    run(s, tiny(3), "cand")                                            # same model again: same array, new line
    assert len(lines(path)) == 3 and path.read_bytes().startswith(first)
    with pytest.raises(FileExistsError):                               # different array under an existing name
        run(s, tiny(5), "cand")
    assert len(lines(path)) == 3
    run(s, tiny(5), "cand", replace=True)
    rows = registry.read_registry(path)
    assert len(rows) == 4 and [r["name"] for r in rows] == ["cand", "cand2", "cand", "cand"]
    assert rows[0]["val_nll_sha256"] == rows[2]["val_nll_sha256"] != rows[3]["val_nll_sha256"]
    lat = registry.latest(rows)
    assert [r["name"] for r in lat] == ["cand", "cand2"] and lat[0] is rows[3]
    tab = registry.table(path)
    assert list(tab["name"]) == ["cand", "cand2"] and tab.loc[1, "k"] == 5 and tab.loc[1, "C"] == 13
    # a partial line left by a crashed writer does not merge with the next row
    with open(path, "ab") as f:
        f.write(b'{"broken": ')
    run(s, tiny(6), "cand3")
    with pytest.raises(ValueError):
        registry.read_registry(path)
    assert [r["name"] for r in registry.read_registry(path, strict=False)][-1] == "cand3"


def test_refuses_anything_but_the_validation_set(setup):
    s = setup
    val, en, path = s["val"], s["en"], s["env"].registry_path
    m = tiny(3)
    other = make_showers(60, 99, DEV)                                  # e.g. a test split of the same size
    perturbed = val.subset(torch.arange(len(val), device=DEV))
    perturbed.w[0] += 1e-3
    cases = {"other events, same count": ValidationSet(other, en),
             "subset": ValidationSet(val.subset(torch.arange(len(val) - 1, device=DEV)), en[:-1]),
             "wrong event numbers": ValidationSet(val, en + 1),
             "reordered event numbers": ValidationSet(val, en[::-1].copy()),
             "modified tensors": ValidationSet(perturbed, en),
             "reordered events": ValidationSet(val.subset(torch.arange(len(val) - 1, -1, -1, device=DEV)), en)}
    for label, vs in cases.items():
        with pytest.raises(NotValidationSet):
            registry.evaluate(m, "x", "stage_a", vs, env=s["env"], reference=s["ref"])
    with pytest.raises(NotValidationSet):
        registry.evaluate(m, "x", "stage_a", val, env=s["env"], reference=s["ref"])     # no event numbers at all
    assert not path.exists() and not (s["env"].results_dir / "stage_a").exists()       # nothing written
    registry.evaluate(m, "x", "stage_a", (val, {"event_number": en}), env=s["env"], reference=s["ref"])
    for bad in ("../x", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            run(s, m, bad)


def test_select_1se_matches_stats(setup):
    s = setup
    models = {"small": (tiny(3), 10, 1.0), "big": (tiny(4, 0.3), 100, 0.0), "mid": (tiny(5, 0.2), 50, 0.0)}
    for name, (m, k, Cc) in models.items():
        run(s, m, name, meta={"k": k, "C": Cc})
    arrs = {n: np.load(s["env"].results_dir / "stage_a" / n / "val_nll.npy") for n in models}
    expect = stats.one_se_select([stats.Candidate(n, arrs[n], k, Cc) for n, (_, k, Cc) in models.items()])
    got = registry.select_1se(stage="stage_a", path=s["env"].registry_path)
    assert got.selected == expect.selected and got.best == expect.best and got.table == expect.table
    sub = registry.select_1se(names=["small", "stage_a/mid"], path=s["env"].registry_path)
    assert {r["name"] for r in sub.table} == {"small", "mid"}
    with pytest.raises(KeyError):
        registry.select_1se(names=["nope"], path=s["env"].registry_path)
    np.save(s["env"].results_dir / "stage_a" / "mid" / "val_nll.npy", arrs["mid"] + 1)   # tampered array
    with pytest.raises(RuntimeError):
        registry.select_1se(stage="stage_a", path=s["env"].registry_path)


def test_guard_columns(setup):
    s = setup
    m = tiny(3)
    abar = torch.ones(C.N_Z, C.N_R, device=DEV)
    abar[:, -3:] = 0.0
    row = run(s, m, "guarded", guards=True, bench_models=lambda: s["bench_models"], abar=abar)
    g = row["guards"]
    val = s["val"]
    zb = val.ev["zb"].cpu().numpy()
    idx = np.flatnonzero(zb >= registry.ZB_STRATUM_MM)
    assert g["n_stratum"] == len(idx) == len(val) - 6 and g["n_short"] == 6
    with torch.inference_mode():
        k1024 = metric.test_nll(m, val, K=1024, chunk=64, deterministic_sum=True)
        sub = val.subset(torch.as_tensor(idx, device=DEV))
        bulk = metric.test_nll(m, sub, bulk=metric.Bulk(short="raise"), deterministic_sum=True)
        ref = {b: (metric.test_nll(bm, val, K=1024, chunk=64, deterministic_sum=True),
                   metric.test_nll(bm, sub, bulk=metric.Bulk(short="raise"), deterministic_sum=True))
               for b, bm in s["bench_models"].items()}
    with metric.deterministic():
        main = C.test_nll(m, val).astype(np.float64)
    for b in ("mlp", "formula"):
        gb = g[f"vs_{b}"]
        p = stats.paired(k1024, ref[b][0])
        assert gb["gk"]["mean"] == p.mean and gb["gk"]["se"] == p.se
        pb = stats.paired(bulk.astype(np.float64) - ref[b][1], np.zeros(len(idx)))
        assert gb["bulk"]["mean"] == pytest.approx(pb.mean, abs=1e-12)
        d = main - s["bench"][b]
        assert gb["report_on_stratum"]["mean"] == pytest.approx(d[idx].mean(), abs=1e-12)
        assert gb["report_on_short"]["n"] == 6
        assert gb["report_on_short"]["mean"] == pytest.approx(np.delete(d, idx).mean(), abs=1e-12)
        pers = stats.guard_persistence(d, k1024.astype(np.float64) - ref[b][0])
        assert gb["gk_persistence"]["numbers"]["margin"] == pytest.approx(pers.numbers["margin"], abs=1e-12)
    assert set(g["mass"]) >= {"total", "beyond_zb", "last_ring", "outside_bulk", "mask:abar_le_0.01"}
    assert abs(g["mass"]["total"]["mean"] - 1) < 1e-5
    rd = s["env"].results_dir / "stage_a" / "guarded"
    assert np.array_equal(np.load(rd / "val_nll_k1024.npy"), k1024)
    assert (s["env"].refs_dir / "guard_spec.json").exists()
    # the cached benchmark guard arrays are reused (no benchmark models needed the second time)
    row2 = run(s, tiny(4), "guarded2", guards=True, bench_models=lambda: pytest.fail("recomputed"), abar=abar)
    assert row2["guards"]["n_stratum"] == len(idx)


TRAINVAL = registry.paths.CACHE_DIR / "pion_s1_trainval.pt"


@pytest.mark.skipif(not TRAINVAL.exists(), reason="validation cache not built")
def test_real_validation_split_passes_and_variants_are_refused():
    """The cached validation split (on the CPU; nothing is evaluated) passes the identity check; a subset, a reordering
    and a shifted event-number list do not."""
    ref = registry.validation_reference()
    assert ref.n_events == registry.EXPECTED_VAL_EVENTS and len(ref.fingerprint) == 64
    val = registry.load_validation(device="cpu")
    assert registry.check_validation(val, ref) is val.showers
    n = len(val.showers)
    with pytest.raises(NotValidationSet):
        registry.check_validation(ValidationSet(val.showers.subset(torch.arange(n - 1)), val.event_number[:-1]), ref)
    perm = torch.arange(n)
    perm[[0, 1]] = perm[[1, 0]]
    with pytest.raises(NotValidationSet):
        registry.check_validation(ValidationSet(val.showers.subset(perm), val.event_number[perm.numpy()]), ref)
    with pytest.raises(NotValidationSet):
        registry.check_validation(ValidationSet(val.showers, val.event_number + 1), ref)
