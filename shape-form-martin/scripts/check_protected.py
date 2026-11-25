"""Checksum guard for the read-only directories shape-form-proton/ and shape-form-original/.

Usage (from anywhere):
    uv run python -B shape-form-martin/scripts/check_protected.py record
    uv run python -B shape-form-martin/scripts/check_protected.py verify

record  hashes (SHA-256) every file under shape-form-proton/ and shape-form-original/ and writes
        results/m0_parity/checksums.json. Exceptions:
          - the two files that M0 moves into shape-form-martin/ (common.py, 05_mlp_pion_copy.ipynb):
            hashed into a separate "moved" section so the destination can be compared;
          - everything under shape-form-original/data/: only size and mtime are recorded, the files
            are never opened (the CSV is 15 GB and must not be read);
          - __pycache__/ contents: recorded, but changes there never fail the check (reported only).
verify  re-scans and compares with the manifest. Exit code 0 = unchanged, 1 = a protected file was
        changed, added or removed (or common.py's moved copy differs from the original).
        A verification report is written next to the manifest (checksums_verify.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

MARTIN = Path(__file__).resolve().parents[1]
REPO = MARTIN.parent
ROOTS = ("shape-form-proton", "shape-form-original")
STAT_ONLY_DIRS = ("shape-form-original/data",)
MOVED = {
    "shape-form-original/common.py": {"dest": "shape-form-martin/common.py", "must_match": True},
    "shape-form-original/05_mlp_pion_copy.ipynb": {
        "dest": "shape-form-martin/05_mlp_pion_copy.ipynb",
        "must_match": False,  # settings cell is edited on purpose
    },
}


def results_dir() -> Path:
    base = os.environ.get("SFM_RESULTS_DIR")
    return Path(base) if base else MARTIN / "results"


def manifest_path() -> Path:
    return results_dir() / "m0_parity" / "checksums.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_pycache(rel: str) -> bool:
    return "__pycache__" in Path(rel).parts or rel.endswith((".pyc", ".pyo"))


def is_stat_only(rel: str) -> bool:
    return any(rel == d or rel.startswith(d + "/") for d in STAT_ONLY_DIRS)


def scan() -> dict:
    files, stat_only, pycache, links = {}, {}, {}, {}
    for root in ROOTS:
        for dirpath, dirnames, filenames in os.walk(REPO / root, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames) + sorted(d for d in dirnames if (Path(dirpath) / d).is_symlink()):
                p = Path(dirpath) / name
                rel = p.relative_to(REPO).as_posix()
                if rel in MOVED:
                    continue
                if p.is_symlink():
                    links[rel] = os.readlink(p)
                    continue
                if is_stat_only(rel):
                    st = p.stat()  # never opened
                    stat_only[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
                elif is_pycache(rel):
                    pycache[rel] = sha256(p)
                else:
                    files[rel] = {"sha256": sha256(p), "size": p.stat().st_size}
    return {"files": files, "stat_only": stat_only, "pycache": pycache, "symlinks": links}


def record() -> int:
    snap = scan()
    moved = {}
    for src, info in MOVED.items():
        p = REPO / src
        if not p.exists():
            print(f"ERROR: {src} does not exist; record must run before the move.", file=sys.stderr)
            return 1
        moved[src] = {"sha256": sha256(p), "size": p.stat().st_size, **info}
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "roots": list(ROOTS),
        "stat_only_dirs": list(STAT_ONLY_DIRS),
        "moved": moved,
        **snap,
    }
    mp = manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(snap['files'])} hashed files, {len(snap['stat_only'])} stat-only data files, "
          f"{len(snap['pycache'])} __pycache__ files, {len(snap['symlinks'])} symlinks, "
          f"{len(moved)} to-be-moved files -> {mp.relative_to(REPO) if mp.is_relative_to(REPO) else mp}")
    return 0


def _diff(old: dict, new: dict, key=None):
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(k for k in set(old) & set(new) if (old[k] if key is None else old[k][key])
                     != (new[k] if key is None else new[k][key]))
    return added, removed, changed


def verify() -> int:
    mp = manifest_path()
    man = json.loads(mp.read_text())
    snap = scan()
    rep = {"verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "manifest": str(mp.name)}
    a, r, c = _diff(man["files"], snap["files"], "sha256")
    rep["files"] = {"n_recorded": len(man["files"]), "n_now": len(snap["files"]),
                    "added": a, "removed": r, "changed": c}
    sa, sr, sc = _diff(man["stat_only"], snap["stat_only"])
    rep["stat_only"] = {"n_recorded": len(man["stat_only"]), "added": sa, "removed": sr, "changed": sc}
    la, lr, lc = _diff(man.get("symlinks", {}), snap["symlinks"])
    rep["symlinks"] = {"added": la, "removed": lr, "changed": lc}
    pa, pr, pc = _diff(man["pycache"], snap["pycache"])
    rep["pycache_info_only"] = {"added": pa, "removed": pr, "changed": pc}
    moved = {}
    moved_fail = False
    for src, info in man["moved"].items():
        dest = REPO / info["dest"]
        st = {"old_path_exists": (REPO / src).exists(), "dest_exists": dest.exists()}
        if dest.exists():
            st["dest_sha256_equal"] = sha256(dest) == info["sha256"]
        if info["must_match"] and not st.get("dest_sha256_equal", False):
            moved_fail = True
        moved[src] = st
    rep["moved"] = moved
    fail = bool(a or r or c or sa or sr or sc or la or lr or lc or moved_fail)
    rep["status"] = "FAIL" if fail else "PASS"
    out = mp.with_name("checksums_verify.json")
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=1, sort_keys=True))
    if pa or pc:
        print("NOTE: new/changed __pycache__ files under a protected directory (not a failure):", pa + pc)
    print("STATUS:", rep["status"])
    return 1 if fail else 0



def _refuse_record_writer(what: str) -> None:
    """transfer/REVIEW_STATUS.md section 4, point 15: this script overwrites a tracked laptop record (or opens test).
    Blocked unless SFM_ALLOW_RECORD_WRITER=1 is set explicitly, and always blocked with an output root
    (SFM_OUT_ROOT: the H200 / pristine-clone setting)."""
    import os as _os
    if _os.environ.get("SFM_OUT_ROOT") or _os.environ.get("SFM_ALLOW_RECORD_WRITER") != "1":
        raise SystemExit(f"{what}: blocked - it overwrites tracked laptop records (spec point 15). Never run it on "
                         "the H200; on the laptop only with SFM_ALLOW_RECORD_WRITER=1 and the user's go-ahead.")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "verify"])
    args = ap.parse_args()
    if args.mode == "record":
        _refuse_record_writer("check_protected.py record (rewrites checksums.json)")
    return record() if args.mode == "record" else verify()


if __name__ == "__main__":
    raise SystemExit(main())
