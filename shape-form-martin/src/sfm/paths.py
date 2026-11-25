"""Locations used by the campaign, derived from this file's position (no absolute paths in code).

Layout assumed (repo root = three levels above this file):
    <repo>/shape-form-martin/src/sfm/paths.py
    <repo>/shape-form-original/data/                      raw data (read-only; only the Parquet file is used)
    <repo>/shape-form-proton/results/particles/pion/      benchmark checkpoints (read-only)

Environment overrides: SFM_DATA_DIR, SFM_RESULTS_DIR, SFM_CACHE_DIR, SFM_OUT_ROOT.

Output root (H200 constraint of 25 Nov 2026: the clone there is pull-only and must stay pristine)
    Everything a machine computes from now on is written under ``OUT_RESULTS / "machines" / <machine id>``, where
    ``OUT_RESULTS = OUT_ROOT / "shape-form-martin" / "results"`` and ``OUT_ROOT`` is ``$SFM_OUT_ROOT`` or, by default,
    the repository root. On the H200 ``SFM_OUT_ROOT`` points OUTSIDE the clone ("pristine mode"); the layout below it
    is the tracked layout of the laptop repository, so a machine directory copied back lands at
    ``shape-form-martin/results/machines/<machine id>/`` unchanged. In pristine mode ``check_output_path`` refuses
    every write inside the clone except under the git-ignored ``shape-form-martin/cache/``. Records of other
    machines (laptop rows, done-markers, model.pt, quarantined arrays) are read-only inputs from the clone.
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

OUT_ROOT = _env_path("SFM_OUT_ROOT", REPO_ROOT)
OUT_RESULTS = OUT_ROOT / MARTIN_DIR.name / "results"                 # == MARTIN_DIR / "results" by default
REPO_RESULTS = MARTIN_DIR / "results"                                 # inputs and records from the clone (read-only)

M0_DIR = RESULTS_DIR / "m0_parity"
GATES_DIR = RESULTS_DIR / "gates"
TEST_ACCESS_LOG = RESULTS_DIR / "test_access.log"


def parquet_path(particle: str = PARTICLE) -> Path:
    """The Parquet hit table for `particle`."""
    return DATA_DIR / FILE_PATTERN.format(particle=particle)


def rel_to(path: Path, start: Path) -> str:
    """`path` relative to `start` as a POSIX string (for configuring common.py from a notebook cwd)."""
    return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()


def _inside(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def pristine_mode() -> bool:
    """True when the output root lies outside the repository (the H200 setting): the clone must not be written."""
    return not _inside(OUT_ROOT, REPO_ROOT)


def check_output_path(p) -> Path:
    """Refuse (PermissionError) a write inside the clone in pristine mode, except under the git-ignored cache
    directory of shape-form-martin. Returns the path."""
    p = Path(p)
    if pristine_mode() and _inside(p, REPO_ROOT) and not _inside(p, MARTIN_DIR / "cache"):
        raise PermissionError(f"pristine clone: refusing to write {portable(p)} inside the repository; outputs go "
                              f"under SFM_OUT_ROOT ({portable(OUT_RESULTS)})")
    return p


def portable(p) -> str:
    """A path string without the local directory layout: '<out>/...' under the output root (pristine mode),
    '<repo>/...' inside the repository, otherwise the absolute path passed through ``scrub_str`` (placeholders for
    SFM_USER_CACHE, the guarded area, TMPDIR and the home directory)."""
    p = Path(p)
    if pristine_mode() and _inside(p, OUT_ROOT):
        return "<out>/" + Path(os.path.relpath(p.resolve(), OUT_ROOT.resolve())).as_posix()
    if _inside(p, REPO_ROOT):
        return "<repo>/" + Path(os.path.relpath(p.resolve(), REPO_ROOT.resolve())).as_posix()
    return scrub_str(str(p))


def scrub_str(text: str) -> str:
    """Replace absolute local roots in a string (a JSON line, a report) by placeholders: the output root by '<out>'
    (pristine mode), the repository by '<repo>', SFM_USER_CACHE by '<cache>', the guarded area by '<area>', TMPDIR
    by '<tmp>', the home directory by '<home>' (longest prefix first). Nothing that names the local
    directory layout (and with it a user name) is written to a file that may be committed."""
    reps = []
    if pristine_mode():
        reps.append((str(OUT_ROOT.resolve()), "<out>"))
    reps.append((str(REPO_ROOT.resolve()), "<repo>"))
    for var, label in (("SFM_USER_CACHE", "<cache>"), ("SFM_GUARD_AREA", "<area>"), ("TMPDIR", "<tmp>")):
        v = os.environ.get(var)
        if v and v not in ("off", "/"):
            reps.append((str(Path(v).expanduser().resolve()), label))
    home = os.path.expanduser("~")
    if home and home != "/":
        reps.append((str(Path(home).resolve()), "<home>"))
    import re
    for a, b in sorted(reps, key=lambda x: -len(x[0])):
        text = re.sub(re.escape(a) + r"(?=[/\\\"'\s,;:)\]]|$)", b, text)       # whole path components only
    return text


# File-access guard of the H200 (sfm.guard): installed here, at the first import of the campaign code, so that it is
# in place before common.py / pandas are imported. Inert unless SFM_GUARD_AREA is set.
if os.environ.get("SFM_GUARD_AREA"):
    from . import guard as _guard
    _guard.install()
