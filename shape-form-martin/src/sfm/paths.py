"""Locations used by the campaign, derived from this file's position (no absolute paths in code).

Layout assumed (repo root = three levels above this file):
    <repo>/shape-form-martin/src/sfm/paths.py
    <repo>/shape-form-original/data/                      raw data (read-only; only the Parquet file is used)
    <repo>/shape-form-proton/results/particles/pion/      benchmark checkpoints (read-only)

Environment overrides: SFM_DATA_DIR, SFM_RESULTS_DIR, SFM_CACHE_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

SFM_DIR = Path(__file__).resolve().parent          # shape-form-martin/src/sfm
MARTIN_DIR = SFM_DIR.parents[1]                    # shape-form-martin
REPO_ROOT = MARTIN_DIR.parent
ORIGINAL_DIR = REPO_ROOT / "shape-form-original"   # read-only
PROTON_DIR = REPO_ROOT / "shape-form-proton"       # read-only

PARTICLE = "pion"
# Same naming as common.FILE_PATTERN, but the Parquet copy (the CSV is never opened).
FILE_PATTERN = "{particle}_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet"


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


DATA_DIR = _env_path("SFM_DATA_DIR", ORIGINAL_DIR / "data")
RESULTS_DIR = _env_path("SFM_RESULTS_DIR", MARTIN_DIR / "results")
CACHE_DIR = _env_path("SFM_CACHE_DIR", MARTIN_DIR / "cache")
PION_RESULTS_DIR = PROTON_DIR / "results" / "particles" / PARTICLE   # read-only benchmark files

M0_DIR = RESULTS_DIR / "m0_parity"
GATES_DIR = RESULTS_DIR / "gates"
TEST_ACCESS_LOG = RESULTS_DIR / "test_access.log"


def parquet_path(particle: str = PARTICLE) -> Path:
    """The Parquet hit table for `particle`."""
    return DATA_DIR / FILE_PATTERN.format(particle=particle)


def rel_to(path: Path, start: Path) -> str:
    """`path` relative to `start` as a POSIX string (for configuring common.py from a notebook cwd)."""
    return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()
