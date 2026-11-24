"""Cheap checks of the lean cache and its access rules (no model evaluation, never opens the test cache)."""
import re

import numpy as np
import pytest

from sfm import data, paths


def test_no_absolute_paths_in_code():
    for f in list((paths.MARTIN_DIR / "src" / "sfm").glob("*.py")) + list((paths.MARTIN_DIR / "scripts").glob("*.py")):
        assert not re.search(r"['\"]/(home|tmp|Users)/", f.read_text()), f


def test_paths_layout():
    assert paths.MARTIN_DIR.name == "shape-form-martin"
    assert paths.parquet_path().suffix == ".parquet"
    assert paths.PION_RESULTS_DIR.parts[-4:] == ("shape-form-proton", "results", "particles", "pion")


@pytest.mark.skipif(not data.cache_files()["meta"].exists(), reason="cache not built")
def test_meta_counts_and_geometry():
    meta = data.load_meta()
    assert meta["counts"] == data.EXPECTED_COUNTS
    assert meta["events_total"] == 99689
    for k, v in data.EXPECTED_GEOMETRY.items():
        assert abs(meta["geometry"][k] - v) < 1e-9 * v


def test_test_split_is_locked():
    log = paths.TEST_ACCESS_LOG
    before = log.read_text() if log.exists() else None
    with pytest.raises(PermissionError):
        data.load_sets(("test",), device="cpu")
    with pytest.raises(ValueError):
        data.load_sets(("test",), device="cpu", allow_test=True, reason="  ")
    assert (log.read_text() if log.exists() else None) == before


@pytest.mark.skipif(not data.cache_files()["trainval"].exists(), reason="cache not built")
def test_trainval_disjoint_and_sorted():
    sets, ex = data.load_sets(("train", "val"), device="cpu")
    assert len(sets["train"]) == 69783 and len(sets["val"]) == 14950
    tr, va = ex["train"]["event_number"], ex["val"]["event_number"]
    assert not set(tr.tolist()) & set(va.tolist())
    assert np.all(np.diff(tr) > 0) and np.all(np.diff(va) > 0)
    for s in sets.values():
        assert int(s.off[-1]) == len(s.w) == len(s.xyz)
