"""Tests of scripts/make_transfer_package.py on synthetic git repositories in temporary directories.

Nothing here touches the real repository's data: the script's tables are replaced by a small synthetic table
(same entry format, same test-cache path, a glob pattern with an expected count, three test-derived files and a raw
table that contains test events), so the tests also run on a machine where the real group-b files are absent or quarantined. Standard library + pytest only.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_transfer_package.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_transfer_package_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()
TEST_CACHE = M.TEST_CACHE_REL
META = M.META_REL
OUT = M.TRANSFER_REL
Q = M.QUARANTINE_REL
TD_INPUTS = [TEST_CACHE, "results/m0/test_event_number.npy"]          # test-derived, b1
TD_RECORDS = ["results/m0/bench_test_nll.npy"]                          # test-derived, b2 (stream-dependent)
TD_ALL = TD_INPUTS + TD_RECORDS
RAW = "data/raw.parquet"                                                # contains_test_events (deferred, not test-derived)
DEFERRED_INPUTS = [RAW] + TD_INPUTS
DEFERRED_ALL = DEFERRED_INPUTS + TD_RECORDS
N_DEFERRED = "4 checked: 3 test-derived + the raw Parquet table"
GITIGNORE = "cache/\n*.npy\n*.npz\n*.parquet\n*.csv\n"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def synthetic_tables() -> dict:
    a = [M._a("code/tool.py", role="code"),
         M._a("results/registry.jsonl", kind="record", laptop_record=True, role="rows", hazard="trusted by the code")]
    b1 = [M._b(RAW, "b1", "raw table", ["reader"], False, "none", M.AFTER_G2C_RAW, {"command": "-"},
               test_note="contains the raw hits of all events", contains_test_events=True),
          M._b(TEST_CACHE, "b1", "test cache", ["load_sets(allow_test=True)"], False, "none", M.AFTER_G2C,
               {"command": "-"}, cross_check={"kind": "meta", "key": "test"}, opaque=True, test_derived=True),
          M._b(META, "b1", "meta", ["load_meta"], False, "none", M.COPY_LIVE, {"command": "-"},
               test_note="summary metadata only"),
          M._b("results/m0/test_event_number.npy", "b1", "test order", ["final test"], False, "none", M.AFTER_G2C,
               {"command": "-"}, test_derived=True)]
    b2 = [M._b("results/m0/bench_val_nll.npy", "b2", "pairing array", ["evaluate"], True, "laptop grid 144", M.COPY_Q,
               {"command": "-"}, hazard="pairs silently"),
          M._b("results/m0/mon.npz", "b2", "monitors", ["none"], False, "none", M.COPY_Q, {"command": "-"}),
          M._b("results/grid/g*.npy", "b2", "grid arrays", ["tests"], True, "emulated grid 1056", M.COPY_Q,
               {"command": "-"}, count=2),
          M._b("results/m0/bench_test_nll.npy", "b2", "test array", ["none"], True, "laptop grid 144", M.AFTER_G2C,
               {"command": "-"}, test_derived=True)]
    c = [M._c("results/new/out.npy", "no command exists yet", "-", "machine-dependent")]
    k = [M._k("code/tool.py", 2, "SEED = 7", "a constant", "none")]
    x = [M._x("code/tool.py", 1, "x = 1", "a reader", "fails before Gate 2c")]
    return {"GROUP_A": a, "GROUP_B1": b1, "GROUP_B2": b2, "GROUP_C": c, "CONSTANTS": k, "ABSENCE": x}


TEST_BYTES = b"\x80\x04PK opaque test-cache bytes (never unpickled) " * 4096
SIZES = {"data/raw.parquet": 5000, "results/m0/test_event_number.npy": 400, "results/m0/bench_val_nll.npy": 800,
         "results/m0/mon.npz": 300, "results/m0/bench_test_nll.npy": 700}


def make_src(root: Path) -> Path:
    repo = root / "src"
    repo.mkdir()
    git(repo, "init", "-q")
    files = {".gitignore": GITIGNORE.encode(), "code/tool.py": b"x = 1\nSEED = 7\n", "results/registry.jsonl": b"{}\n",
             TEST_CACHE: TEST_BYTES, "results/grid/g1.npy": b"g1" * 50, "results/grid/g2.npy": b"g2" * 60}
    files.update({rel: os.urandom(n) for rel, n in SIZES.items()})
    files[META] = json.dumps({"files": {"test": {"name": "pion_s1_test.pt", "sha256": sha(TEST_BYTES),
                                                 "bytes": len(TEST_BYTES)}}}).encode()
    for rel, data in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    git(repo, "add", ".gitignore", "code/tool.py", "results/registry.jsonl")
    git(repo, "commit", "-q", "-m", "synthetic")
    return repo


@pytest.fixture
def tables(monkeypatch):
    t = synthetic_tables()
    for k, v in t.items():
        monkeypatch.setattr(M, k, v)
    return t


@pytest.fixture
def src(tmp_path, tables):
    return make_src(tmp_path)


def copy_list(src: Path, dst_root: Path, list_file: Path) -> None:
    """What `rsync -a --files-from=<list> <src>/ <dst_root>/` does."""
    for rel in list_file.read_text().splitlines():
        d = dst_root / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, d)


def round_trip(src: Path, tmp: Path) -> Path:
    """list on the laptop, commit, clone to 'h200', copy the two lists of phase 'now'."""
    assert M.main(["list", "--repo", str(src)]) == 0
    git(src, "add", *[f"{OUT}/{n}" for n in M.OUTPUT_NAMES])
    git(src, "commit", "-q", "-m", "package")
    dst = tmp / "h200"
    subprocess.run(["git", "clone", "-q", str(src), str(dst)], check=True, capture_output=True)
    copy_list(src, dst, src / OUT / "files_b_inputs.txt")
    copy_list(src, dst / Q, src / OUT / "files_b_laptop_records.txt")
    return dst


def copy_after_gate2c(src: Path, dst: Path) -> None:
    copy_list(src, dst, src / OUT / "files_b_test_derived_inputs.txt")
    copy_list(src, dst / Q, src / OUT / "files_b_test_derived_records.txt")


def verify_json(repo: Path, *extra, capsys) -> tuple:
    capsys.readouterr()
    rc = M.main(["verify", "--repo", str(repo), "--json", *extra])
    return rc, json.loads(capsys.readouterr().out)


# ----------------------------------------------------------------------------------------------------------------------
def test_four_lists_and_manifest_fields(src):
    assert M.main(["list", "--repo", str(src)]) == 0
    rd = lambda n: (src / OUT / n).read_text().splitlines()                                   # noqa: E731
    assert rd("files_b_inputs.txt") == [META]
    assert rd("files_b_laptop_records.txt") == ["results/m0/bench_val_nll.npy", "results/m0/mon.npz",
                                                "results/grid/g1.npy", "results/grid/g2.npy"]
    assert rd("files_b_test_derived_inputs.txt") == DEFERRED_INPUTS       # 2 test-derived inputs + the raw table
    assert rd("files_b_test_derived_records.txt") == TD_RECORDS
    man = json.loads((src / OUT / "MANIFEST.json").read_text())
    assert man["schema"] == 2
    for g in ("b1", "b2"):
        for x in man["groups"][g]["files"]:
            assert x["test_derived"] == (x["path"] in TD_ALL), x["path"]
            assert x["contains_test_events"] == (x["path"] == RAW), x["path"]
            assert x["transfer_phase"] == ("after_gate2c" if x["path"] in DEFERRED_ALL else "now")
            assert x["list"] == ("files_b_test_derived_inputs.txt" if x["path"] in DEFERRED_INPUTS else
                                 "files_b_test_derived_records.txt" if x["path"] in TD_RECORDS else
                                 "files_b_inputs.txt" if g == "b1" else "files_b_laptop_records.txt")
            assert x["destination"] == (x["path"] if g == "b1" else f"{Q}/{x['path']}")
    g = man["groups"]
    n_tc, n_meta = len(TEST_BYTES), len((src / META).read_bytes())
    assert g["b1"]["totals"] == {"files": 4, "bytes": 5000 + n_tc + n_meta + 400}
    assert g["b1"]["totals_by_phase"] == {"now": {"files": 1, "bytes": n_meta},
                                          "after_gate2c": {"files": 3, "bytes": 5000 + n_tc + 400}}
    assert g["b2"]["totals"] == {"files": 5, "bytes": 800 + 300 + 100 + 120 + 700}
    assert g["b2"]["totals_by_phase"] == {"now": {"files": 4, "bytes": 1320}, "after_gate2c": {"files": 1, "bytes": 700}}
    assert man["totals_by_phase"] == {"now": {"files": 5, "bytes": n_meta + 1320},
                                      "after_gate2c": {"files": 4, "bytes": 5000 + n_tc + 400 + 700}}
    lists = man["lists"]
    assert {k: v["totals"]["files"] for k, v in lists.items()} == {
        "inputs": 1, "laptop_records": 4, "test_derived_inputs": 3, "test_derived_records": 1}
    tdi = lists["test_derived_inputs"]
    assert tdi["test_derived_files"] == 2 and tdi["contains_test_events_files"] == 1 and tdi["deferred"]
    assert "raw Parquet table" in tdi["holds"] and "three test-derived inputs" in tdi["holds"]
    tcmd = man["transfer_commands"]
    assert tcmd["phases"] == {"now": ["inputs", "laptop_records"],
                              "after_gate2c": ["test_derived_inputs", "test_derived_records"]}
    for key, f, root in (("inputs", "files_b_inputs.txt", "<repo>/"), ("laptop_records", "files_b_laptop_records.txt",
                         f"<repo>/{Q}/"), ("test_derived_inputs", "files_b_test_derived_inputs.txt", "<repo>/"),
                         ("test_derived_records", "files_b_test_derived_records.txt", f"<repo>/{Q}/")):
        assert tcmd[key] == f"rsync -a --files-from={OUT}/{f} <repo>/ <user>@<h200>:{root}"
    pol = man["policies"]
    assert "Gate 2c" in pol["test_derived"] and "absence" in pol["test_derived"] and "\n" not in pol["test_derived"]
    raw_pol = pol["raw_table"]
    assert "\n" not in raw_pol and "25 Nov 2026" in raw_pol and "contains_test_events" in raw_pol
    assert "NOT test-derived" in raw_pol and "after Gate 2c" in raw_pol
    assert "DEFERRED" in pol["test_events_in_raw_parquet"] and "geometry arm" in pol["test_events_in_raw_parquet"]
    assert [x["path"] for x in pol["test_derived_files"]] == TD_ALL
    assert [x["path"] for x in pol["deferred_files"]] == DEFERRED_ALL
    assert {x["path"] for x in pol["test_summary_metadata_now"]} >= {META}
    assert "Gate 2c" in pol["test_cache"] and "opaque" in pol["test_cache"]
    tc = next(x for x in g["b1"]["files"] if x["path"] == TEST_CACHE)
    assert tc["opaque_byte_stream"] and tc["sha256"] == sha(TEST_BYTES) and tc["cross_check"]["equal"]
    assert all(x["git"]["ignored"] and x["git"]["rule"] for k in ("b1", "b2") for x in g[k]["files"])
    assert man["absence_before_gate2c"]["consequences"][0]["line_checked"]
    assert {f"{OUT}/{n}" for n in M.OUTPUT_NAMES} <= {e["pattern"] for e in REAL_GROUP_A if e["package"]}
    assert len(M.OUTPUT_NAMES) == 6


REAL = {k: getattr(M, k) for k in ("GROUP_B1", "GROUP_B2")}
REAL_GROUP_A = M.GROUP_A
REAL_GROUP_C = M.GROUP_C
REAL_PARQUET = "shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet"


def test_real_tables_mark_the_eight_deferred_files():
    entries = [e for g in REAL.values() for e in g]
    td = sorted(e["pattern"] for e in entries if e["test_derived"])
    assert td == sorted([TEST_CACHE, f"{M.PION}/test_v3_nll.npz", f"{M.MARTIN}/results/m0_parity/test_event_number.npy"]
                        + [f"{M.MARTIN}/results/m0_parity/{m}_test_nll{s}.npy" for m in ("mlp", "formula")
                           for s in ("", "_ref1056")])
    raw = [e for e in entries if e["contains_test_events"]]
    assert [e["pattern"] for e in raw] == [REAL_PARQUET] and not raw[0]["test_derived"] and raw[0]["group"] == "b1"
    assert sorted(e["pattern"] for e in entries if M.is_deferred(e)) == sorted(td + [REAL_PARQUET])
    assert all(e["count"] == 1 for e in entries if M.is_deferred(e))
    now_with_note = {e["pattern"] for e in entries if e["test_note"] and not M.is_deferred(e)}
    assert {f"{M.PION}/test_v3.json", META} <= now_with_note
    origin = [c for c in REAL_GROUP_C if "absolute shower origin" in c["path"]]
    assert len(origin) == 1 and "NOT YET EXISTING" in origin[0]["path"] and "event_number" in origin[0]["command"]


def test_round_trip_before_gate2c_passes_with_deferred_absent(src, tmp_path, capsys):
    dst = round_trip(src, tmp_path)
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 0 and res["ok"], res["failures"]
    assert res["mode"] == "target machine, before Gate 2c" and res["phase"] == "before_gate2c"
    d = res["deferred"]
    assert d["absent"] is True and d["offending_paths"] == [] and d["checked"] == 4
    assert d["test_derived"] == 3 and d["contains_test_events"] == 1
    assert d["summary"] == f"deferred files absent: yes ({N_DEFERRED})"
    assert sum(1 for l in res["lines"] if l[1] == "deferred-absent" and l[0] == "OK") == 4
    assert "ABSENT" in res["test_cache"] and "Gate 2c" in res["test_cache"] and "byte hash" in res["test_cache"]
    assert RAW in res["raw_table"] and "ABSENT" in res["raw_table"]
    assert res["tracked_laptop_record_hazards"][0]["pattern"] == "results/registry.jsonl"
    assert any(l[1] == "c" and "to be regenerated" in l[2] for l in res["lines"])
    capsys.readouterr()
    assert M.main(["verify", "--repo", str(dst)]) == 0                     # human-readable mode
    out = capsys.readouterr().out
    assert "SUMMARY:" in out and "PASS" in out and "TEST CACHE:" in out and "trusted by the code" in out
    assert f"SUMMARY: deferred files absent: yes ({N_DEFERRED})" in out and "RAW TABLE:" in out


@pytest.mark.parametrize("rel", DEFERRED_ALL)
@pytest.mark.parametrize("where", ["live", "quarantine", "elsewhere"])
def test_default_verify_fails_when_a_deferred_file_is_present(src, tmp_path, capsys, rel, where):
    dst = round_trip(src, tmp_path)
    target = {"live": dst / rel, "quarantine": dst / Q / rel,
              "elsewhere": dst / "old" / "backup copy" / Path(rel).name}[where]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / rel, target)
    prel = target.relative_to(dst).as_posix()
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1 and not res["ok"]
    d = res["deferred"]
    assert d["absent"] is False and d["offending_paths"] == [prel]
    assert d["summary"] == f"deferred files absent: no ({N_DEFERRED})"
    msg = next(f for f in res["failures"] if prel in f)
    assert "PRESENT BEFORE GATE 2c" in msg
    assert ("RAW TABLE (contains test events)" if rel == RAW else "TEST-DERIVED FILE") in msg
    capsys.readouterr()
    assert M.main(["verify", "--repo", str(dst)]) == 1
    out = capsys.readouterr().out
    assert f"DEFERRED FILE ON THIS MACHINE: {prel}" in out and f"deferred files absent: no ({N_DEFERRED})" in out
    target.unlink()
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 0 and res["deferred"]["absent"], res["failures"]


def test_absence_search_by_name_and_size(src, tmp_path, capsys):
    dst = round_trip(src, tmp_path)
    other = dst / "elsewhere" / "pion_s1_test.pt"                       # same name, different size: not the file
    other.parent.mkdir(parents=True)
    other.write_bytes(TEST_BYTES[:-1])
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 0 and any(l[0] == "INFO" and "different size" in l[2] for l in res["lines"]), res["failures"]
    other.write_bytes(b"\x00" + TEST_BYTES[1:])                         # same name and size, other bytes: warning
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 0 and any("different SHA-256" in w and "elsewhere/pion_s1_test.pt" in w for w in res["warnings"])
    other.write_bytes(TEST_BYTES)                                       # identical: failure
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1 and res["deferred"]["offending_paths"] == ["elsewhere/pion_s1_test.pt"]
    rc, res = verify_json(dst, "--skip-large-hash", "0", capsys=capsys)  # not hashed: treated as present
    assert rc == 1 and any("not checked: treated as present" in f for f in res["failures"])
    other.unlink()
    lr = dst / Q / TEST_CACHE                                          # any file at a destination fails
    lr.parent.mkdir(parents=True, exist_ok=True)
    lr.write_bytes(b"rebuilt elsewhere")
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1 and any("quarantine root" in f and "different size" in f for f in res["failures"])
    lr.unlink()
    (dst / ".git" / "pion_s1_test.pt").write_bytes(TEST_BYTES)          # .git is not the working tree
    assert verify_json(dst, capsys=capsys)[0] == 0


def test_raw_table_absence_rules(src, tmp_path, capsys, monkeypatch):
    dst = round_trip(src, tmp_path)
    raw_bytes = (src / RAW).read_bytes()
    extract = dst / "scratch" / "raw.parquet"                          # same name, another size: information only
    extract.parent.mkdir(parents=True)
    extract.write_bytes(raw_bytes[:1234])
    (dst / "scratch" / "trainval_extract.parquet").write_bytes(b"PAR1 train/val rows only")  # another name: info
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 0 and res["deferred"]["absent"], res["failures"]
    infos = [l[2] for l in res["lines"] if l[0] == "INFO" and l[1] == "deferred-absent"]
    assert any("scratch/raw.parquet" in i and "train/val-only extract" in i for i in infos)
    assert any("scratch/trainval_extract.parquet" in i and "information only" in i for i in infos)
    (dst / RAW).parent.mkdir(parents=True, exist_ok=True)
    (dst / RAW).write_bytes(b"any content at the live path")           # live path: any content fails
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1 and res["deferred"]["offending_paths"] == [RAW]
    assert any("RAW TABLE (contains test events) PRESENT BEFORE GATE 2c at its live path" in f and
               "different size" in f for f in res["failures"])
    (dst / RAW).unlink()
    calls = []                                                          # identical copies: one SHA-256 per file
    real_sha = M.sha256_file
    monkeypatch.setattr(M, "sha256_file", lambda p: calls.append(os.path.realpath(p)) or real_sha(p))
    monkeypatch.setattr(M, "LARGE_HASH_NOTE", 1000)                     # the synthetic table counts as large
    shutil.copy2(src / RAW, dst / RAW)
    extract.write_bytes(raw_bytes)
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1 and sorted(res["deferred"]["offending_paths"]) == sorted([RAW, "scratch/raw.parquet"])
    raw_calls = [c for c in calls if c.endswith("raw.parquet")]
    assert sorted(raw_calls) == sorted({os.path.realpath(dst / RAW), os.path.realpath(extract)})  # once each
    assert sum("SHA-256 of 5000 B computed once" in f for f in res["failures"]) == 2


def test_after_gate2c_requires_all_deferred_files(src, tmp_path, capsys):
    dst = round_trip(src, tmp_path)
    rc, res = verify_json(dst, "--after-gate2c", capsys=capsys)
    assert rc == 1 and res["phase"] == "after_gate2c"
    assert sorted(f for f in res["failures"] if "missing" in f) == sorted(
        [f"[b1-deferred] missing: {r}" for r in DEFERRED_INPUTS] + [f"[b2-deferred] missing: {Q}/{r}" for r in TD_RECORDS])
    assert res["deferred"]["verified"] is False
    assert res["deferred"]["summary"] == f"deferred files present and verified (after Gate 2c): no ({N_DEFERRED})"
    copy_after_gate2c(src, dst)
    rc, res = verify_json(dst, "--after-gate2c", capsys=capsys)
    assert rc == 0 and res["ok"], res["failures"]
    assert res["deferred"]["summary"] == f"deferred files present and verified (after Gate 2c): yes ({N_DEFERRED})"
    assert "byte hash only" in res["test_cache"] and "Gate 2c" in res["test_cache"]
    assert "one SHA-256" in res["raw_table"]
    assert any(l[1] == "b1-deferred" and "[opaque byte hash]" in l[2] for l in res["lines"])
    assert any(l[1] == "b1-deferred" and l[2].endswith(RAW) for l in res["lines"])
    rc, res = verify_json(dst, capsys=capsys)                           # before-Gate-2c mode now fails
    assert rc == 1 and sorted(res["deferred"]["offending_paths"]) == sorted(
        DEFERRED_INPUTS + [f"{Q}/{r}" for r in TD_RECORDS])
    f = dst / TEST_CACHE                                                # corrupted after the copy
    f.write_bytes(f.read_bytes()[:-1] + b"X")
    rc, res = verify_json(dst, "--after-gate2c", capsys=capsys)
    assert rc == 1 and any("SHA-256 mismatch" in x and TEST_CACHE in x for x in res["failures"])
    shutil.copy2(src / TEST_CACHE, f)
    (dst / RAW).unlink()                                                # the raw table is required too
    rc, res = verify_json(dst, "--after-gate2c", capsys=capsys)
    assert rc == 1 and res["failures"] == [f"[b1-deferred] missing: {RAW}"]
    shutil.copy2(src / RAW, dst / RAW)
    rel = TD_RECORDS[0]                                                 # a test record at its LIVE path
    shutil.copy2(src / rel, dst / rel)
    rc, res = verify_json(dst, "--after-gate2c", capsys=capsys)
    assert rc == 1 and any("LIVE PATH" in x and rel in x for x in res["failures"])
    (dst / rel).unlink()
    (dst / Q / rel).unlink()
    rc, res = verify_json(dst, "--after-gate2c", capsys=capsys)
    assert rc == 1 and res["failures"] == [f"[b2-deferred] missing: {Q}/{rel}"]


def test_sha256sums_sections_and_commands(src, tmp_path):
    if not (shutil.which("sha256sum") and shutil.which("sed") and shutil.which("sh")):
        pytest.skip("coreutils not available")
    dst = round_trip(src, tmp_path)
    sums = (src / OUT / "SHA256SUMS").read_text()
    for key in ("inputs", "laptop_records", "test_derived_inputs", "test_derived_records"):
        assert sums.count(f"\n# BEGIN {key}\n") == 1 and sums.count(f"\n# END {key}\n") == 1
    assert sums.count("#   phase: now") == 2 and sums.count("#   phase: AFTER GATE 2c") == 2
    assert sums.count(f"#   destination root: <repo>/{Q}/") == 2 and sums.count("#   destination root: <repo>/\n") == 2
    holds = [l for l in sums.splitlines() if l.startswith("#   holds: ")]
    assert len(holds) == 4 and "three test-derived inputs" in holds[2] and "raw Parquet table" in holds[2]
    section3 = sums.split("# BEGIN test_derived_inputs\n")[1].split("# END test_derived_inputs")[0].splitlines()
    assert [l.split("  ", 1)[1] for l in section3] == DEFERRED_INPUTS
    checks = [l.split("#   check: ", 1)[1] for l in sums.splitlines() if l.startswith("#   check: ")]
    assert len(checks) == 4

    def run(cmd, cwd):
        return subprocess.run(["sh", "-c", cmd], cwd=cwd, capture_output=True, text=True)
    for c in checks[:2]:                                               # phase now: pass on the H200 copy
        p = run(c, dst)
        assert p.returncode == 0, c + p.stdout + p.stderr
    for c in checks[2:]:                                               # after Gate 2c: not there yet
        assert run(c, dst).returncode != 0, c
    copy_after_gate2c(src, dst)
    for c in checks:
        p = run(c, dst)
        assert p.returncode == 0, c + p.stdout + p.stderr
    p = subprocess.run(["sha256sum", "-c", f"{OUT}/SHA256SUMS"], cwd=src, capture_output=True, text=True)  # laptop
    assert p.returncode == 0, p.stdout + p.stderr
    assert sum(1 for l in p.stdout.splitlines() if l.endswith(": OK")) == 4 + 5


def test_corrupted_file_is_detected(src, tmp_path, capsys):
    dst = round_trip(src, tmp_path)
    f = dst / META
    b = bytearray(f.read_bytes())
    b[10] ^= 0xFF
    f.write_bytes(bytes(b))
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1 and any("SHA-256 mismatch" in x and META in x for x in res["failures"])
    q = dst / Q / "results/grid/g2.npy"                                # a record in the quarantine root
    q.write_bytes(q.read_bytes() + b"x")
    rc, res = verify_json(dst, capsys=capsys)
    assert any("size" in x and "g2.npy" in x for x in res["failures"])
    (dst / META).unlink()
    rc, res = verify_json(dst, capsys=capsys)
    assert any("missing" in x and META in x for x in res["failures"])
    rc, res = verify_json(dst, "--skip-large-hash", "50", capsys=capsys)  # size-only pass still sees size/missing
    assert rc == 1 and any("NOT checked" in w for w in res["warnings"])


def test_laptop_record_at_live_path_fails(src, tmp_path, capsys):
    dst = round_trip(src, tmp_path)
    rel = "results/m0/bench_val_nll.npy"
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / rel, dst / rel)
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1
    msg = next(x for x in res["failures"] if rel in x)
    assert "LIVE PATH" in msg and f"mv {rel} {Q}/{rel}" in msg
    assert (dst / rel).exists()                                            # printed, never executed
    (dst / rel).write_bytes(b"produced on this machine")                   # an H200 product at the live path: fine
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 0 and any(l[1] == "b2-live" and "different file" in l[2] for l in res["lines"])
    (dst / rel).unlink()
    shutil.copy2(src / "results/m0/mon.npz", dst / "results/m0/mon.npz")   # not stream-dependent: warning only
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 0 and any("mon.npz" in w for w in res["warnings"])


def test_source_machine_mode(src, capsys):
    assert M.main(["list", "--repo", str(src)]) == 0
    rc, res = verify_json(src, "--source-machine", capsys=capsys)
    assert rc == 0 and res["mode"] == "source machine", res["failures"]
    d = res["deferred"]
    assert d["verified"] is True and d["absent"] is None and "not a failure" in d["summary"]
    assert f"({N_DEFERRED})" in d["summary"]
    assert sum(1 for l in res["lines"] if l[1] in ("b1-deferred", "b2-deferred") and l[0] == "OK") == 4
    assert not [l for l in res["lines"] if l[1] == "deferred-absent"]
    rc, res = verify_json(src, capsys=capsys)                              # without the flag: records at live paths
    assert rc == 1 and any("LIVE PATH" in x for x in res["failures"])
    assert res["deferred"]["absent"] is False and len(res["deferred"]["offending_paths"]) == 4


def test_csv_is_refused(src, tables, monkeypatch, capsys, tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n")
    with pytest.raises(M.RefusedFile):
        M.sha256_file(p)
    with pytest.raises(M.RefusedFile):
        M.sha256_file(tmp_path / "Y.CSV")                                   # refused before any open
    (src / "data/raw.csv").write_text("never read\n")
    assert M.find_by_name(src, {"raw.csv"}) == []                           # the absence search skips *.csv names
    monkeypatch.setattr(M, "GROUP_B1", tables["GROUP_B1"] + [M._b("data/raw.csv", "b1", "csv", ["-"], False, "none",
                                                                     M.COPY_LIVE, {"command": "-"})])
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "refused" in capsys.readouterr().err
    assert not (src / OUT / "MANIFEST.json").exists()                      # nothing written
    monkeypatch.setattr(M, "GROUP_B1", tables["GROUP_B1"])
    assert M.main(["list", "--repo", str(src)]) == 0
    man_p = src / OUT / "MANIFEST.json"
    man = json.loads(man_p.read_text())
    man["groups"]["b1"]["files"].append({**man["groups"]["b1"]["files"][0], "path": "data/raw.csv"})
    man_p.write_text(json.dumps(man))
    assert M.main(["verify", "--repo", str(src), "--source-machine"]) == 1
    assert M.main(["verify", "--repo", str(src)]) == 1
    man["groups"]["b1"]["files"][-1].update(test_derived=True, transfer_phase="after_gate2c")
    man_p.write_text(json.dumps(man))
    assert M.main(["verify", "--repo", str(src)]) == 1
    assert "refused" in capsys.readouterr().err


def test_list_refuses_tracked_unignored_and_mismatched_files(src, tables, monkeypatch, capsys):
    (src / "code/extra.bin").write_bytes(b"neither tracked nor ignored")
    monkeypatch.setattr(M, "GROUP_B2", tables["GROUP_B2"] + [M._b("code/extra.bin", "b2", "x", ["-"], True, "s",
                                                                     M.COPY_Q, {"command": "-"})])
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "NOT ignored" in capsys.readouterr().err
    monkeypatch.setattr(M, "GROUP_B2", tables["GROUP_B2"] + [M._b("code/tool.py", "b2", "x", ["-"], True, "s",
                                                                     M.COPY_Q, {"command": "-"})])
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "tracked by git" in capsys.readouterr().err
    (src / "results/m0/latest_test_scores.npy").write_bytes(b"t" * 10)    # 'test' in the name, no decision
    monkeypatch.setattr(M, "GROUP_B2", tables["GROUP_B2"] + [M._b("results/m0/latest_test_scores.npy", "b2", "x",
                                                                     ["-"], True, "s", M.COPY_Q, {"command": "-"})])
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "neither test_derived nor explained" in capsys.readouterr().err
    monkeypatch.setattr(M, "GROUP_B2", tables["GROUP_B2"] + [M._b("results/m0/latest_test_scores.npy", "b2", "x",
                                                                     ["-"], True, "s", M.COPY_Q, {"command": "-"},
                                                                     test_note="n", contains_test_events=True)])
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "only defined for group b1" in capsys.readouterr().err
    monkeypatch.setattr(M, "GROUP_B1", [dict(e, test_derived=True) if e["pattern"] == RAW else e
                                        for e in tables["GROUP_B1"]])
    monkeypatch.setattr(M, "GROUP_B2", tables["GROUP_B2"])
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "test_derived and contains_test_events are exclusive" in capsys.readouterr().err
    monkeypatch.setattr(M, "GROUP_B2", tables["GROUP_B2"])
    monkeypatch.setattr(M, "GROUP_B1", [dict(e, test_derived=False) if e["pattern"] == TEST_CACHE else e
                                        for e in tables["GROUP_B1"]])
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "the test cache must be test_derived" in capsys.readouterr().err
    monkeypatch.setattr(M, "GROUP_B1", tables["GROUP_B1"])
    meta = json.loads((src / META).read_text())
    meta["files"]["test"]["sha256"] = "0" * 64
    (src / META).write_text(json.dumps(meta))
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "differs from" in capsys.readouterr().err
    monkeypatch.setattr(M, "GROUP_B2", [dict(tables["GROUP_B2"][2], count=3)])
    assert M.main(["list", "--repo", str(src)]) == 1
    assert "expected 3 file(s), found 2" in capsys.readouterr().err


DRIVER = r'''
import importlib.util, json, os, sys
script, tables_json, cmds_json, caches_json = sys.argv[1:5]
test_caches = set(json.loads(caches_json))
events = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            p = os.path.realpath(os.fsdecode(args[0]))
        except Exception:
            return
        if p in test_caches:
            f = sys._getframe(1)
            events.append({"open": p, "mode": args[1], "flags": args[2], "func": f.f_code.co_name,
                           "file": f.f_code.co_filename, "lineno": f.f_lineno})
    elif event == "import" and args[0].split(".")[0] in ("torch", "numpy"):
        events.append({"import": args[0]})
    elif event == "pickle.find_class":
        events.append({"pickle": str(args)})
sys.addaudithook(hook)
spec = importlib.util.spec_from_file_location("mtp", script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for k, v in json.load(open(tables_json)).items():
    setattr(m, k, v)
rc = [m.main(c) for c in json.loads(cmds_json)]
import linecache
for e in events:
    if "lineno" in e:
        e["source_line"] = linecache.getline(e["file"], e["lineno"]).strip()
mods = sorted(x for x in sys.modules if x.split(".")[0] in ("torch", "numpy"))
sys.stderr.write("RESULT " + json.dumps({"rc": rc, "events": events, "modules": mods}) + "\n")
'''


def _drive(tmp_path, tables, cmds, caches) -> dict:
    tj = tmp_path / "tables.json"
    tj.write_text(json.dumps(tables))
    drv = tmp_path / "driver.py"
    drv.write_text(DRIVER)
    p = subprocess.run([sys.executable, "-B", "-I", str(drv), str(SCRIPT), str(tj), json.dumps(cmds),
                        json.dumps([os.path.realpath(c) for c in caches])], capture_output=True, text=True)
    line = next(l for l in p.stderr.splitlines() if l.startswith("RESULT "))
    res = json.loads(line[len("RESULT "):])
    res["stdout"] = p.stdout
    return res


def _assert_only_byte_hash_opens(res, n):
    opens = [e for e in res["events"] if "open" in e]
    assert len(opens) == n, opens
    write_bits = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    for e in opens:                   # the audit event carries FileIO's mode ('r') and the os.open flags
        assert e["mode"] in ("r", "rb") and not (e["flags"] & write_bits), e
        assert e["func"] == "sha256_file" and e["source_line"] == 'with open(path, "rb") as f:', e
    assert not [e for e in res["events"] if "import" in e or "pickle" in e] and res["modules"] == []
    return opens


def test_test_cache_is_only_opened_for_a_byte_hash(src, tables, tmp_path):
    dst = round_trip(src, tmp_path)
    s, d = str(src), str(dst)
    elsewhere = dst / "elsewhere" / "pion_s1_test.pt"
    caches = [src / TEST_CACHE, dst / TEST_CACHE, elsewhere]
    # before Gate 2c on the H200 copy (absent: never opened), list and source-machine verify on the laptop (one each)
    res = _drive(tmp_path, tables, [["verify", "--repo", d, "--json"], ["list", "--repo", s],
                                    ["verify", "--repo", s, "--source-machine", "--json"]], caches)
    assert res["rc"] == [0, 0, 0], res["stdout"][-2000:]
    opens = _assert_only_byte_hash_opens(res, 2)
    assert {e["open"] for e in opens} == {os.path.realpath(src / TEST_CACHE)}
    # after Gate 2c (byte hash), then the before-Gate-2c check finding it (size equal -> byte hash); a same-name file
    # of another size elsewhere is compared by size only and never opened
    copy_after_gate2c(src, dst)
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_bytes(TEST_BYTES[:100])
    res = _drive(tmp_path, tables, [["verify", "--repo", d, "--after-gate2c", "--json"],
                                    ["verify", "--repo", d, "--json"]], caches)
    assert res["rc"] == [0, 1], res["stdout"][-2000:]
    opens = _assert_only_byte_hash_opens(res, 2)
    assert {e["open"] for e in opens} == {os.path.realpath(dst / TEST_CACHE)}


def test_script_imports_neither_torch_nor_numpy():
    tree = ast.parse(SCRIPT.read_text())
    mods = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not mods & {"torch", "numpy", "pandas", "pyarrow", "pickle"}, mods
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods - set(sys.stdlib_module_names)
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('m', %r); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            "try:\n    m.main(['--help'])\nexcept SystemExit:\n    pass\n"
            "print(sorted(x for x in sys.modules if x.split('.')[0] in ('torch', 'numpy')))") % str(SCRIPT)
    p = subprocess.run([sys.executable, "-B", "-I", "-c", code], capture_output=True, text=True, check=True)
    assert p.stdout.strip().splitlines()[-1] == "[]"
