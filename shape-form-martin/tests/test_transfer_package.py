"""Tests of scripts/make_transfer_package.py on synthetic git repositories in temporary directories.

Nothing here touches the real repository's data: the script's tables are replaced by a small synthetic table
(same entry format, same test-cache path, a glob pattern with an expected count), so the tests also run on a
machine where the real group-b files are absent or quarantined. Standard library + pytest only.
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
    b1 = [M._b("data/raw.parquet", "b1", "raw table", ["reader"], False, "none", M.COPY_LIVE, {"command": "-"}),
          M._b(TEST_CACHE, "b1", "test cache", ["load_sets(allow_test=True)"], False, "none", M.COPY_LIVE,
               {"command": "-"}, cross_check={"kind": "meta", "key": "test"}, opaque=True),
          M._b(META, "b1", "meta", ["load_meta"], False, "none", M.COPY_LIVE, {"command": "-"})]
    b2 = [M._b("results/m0/bench_val_nll.npy", "b2", "pairing array", ["evaluate"], True, "laptop grid 144", M.COPY_Q,
               {"command": "-"}, hazard="pairs silently"),
          M._b("results/m0/mon.npz", "b2", "monitors", ["none"], False, "none", M.COPY_Q, {"command": "-"}),
          M._b("results/grid/g*.npy", "b2", "grid arrays", ["tests"], True, "emulated grid 1056", M.COPY_Q,
               {"command": "-"}, count=2)]
    c = [M._c("results/new/out.npy", "no command exists yet", "-", "machine-dependent")]
    k = [M._k("code/tool.py", 2, "SEED = 7", "a constant", "none")]
    return {"GROUP_A": a, "GROUP_B1": b1, "GROUP_B2": b2, "GROUP_C": c, "CONSTANTS": k}


TEST_BYTES = b"\x80\x04PK opaque test-cache bytes (never unpickled) " * 4096


def make_src(root: Path) -> Path:
    repo = root / "src"
    repo.mkdir()
    git(repo, "init", "-q")
    files = {".gitignore": GITIGNORE.encode(), "code/tool.py": b"x = 1\nSEED = 7\n", "results/registry.jsonl": b"{}\n",
             "data/raw.parquet": os.urandom(5000), TEST_CACHE: TEST_BYTES,
             "results/m0/bench_val_nll.npy": os.urandom(800), "results/m0/mon.npz": os.urandom(300),
             "results/grid/g1.npy": b"g1" * 50, "results/grid/g2.npy": b"g2" * 60}
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
    assert M.main(["list", "--repo", str(src)]) == 0
    git(src, "add", *[f"{OUT}/{n}" for n in M.OUTPUT_NAMES])
    git(src, "commit", "-q", "-m", "package")
    dst = tmp / "h200"
    subprocess.run(["git", "clone", "-q", str(src), str(dst)], check=True, capture_output=True)
    copy_list(src, dst, src / OUT / "files_b_inputs.txt")
    copy_list(src, dst / M.QUARANTINE_REL, src / OUT / "files_b_laptop_records.txt")
    return dst


def verify_json(repo: Path, *extra, capsys) -> tuple:
    capsys.readouterr()
    rc = M.main(["verify", "--repo", str(repo), "--json", *extra])
    return rc, json.loads(capsys.readouterr().out)


# ----------------------------------------------------------------------------------------------------------------------
def test_round_trip_list_copy_verify(src, tmp_path, capsys):
    dst = round_trip(src, tmp_path)
    man = json.loads((src / OUT / "MANIFEST.json").read_text())
    assert man["groups"]["b1"]["totals"]["files"] == 3 and man["groups"]["b2"]["totals"]["files"] == 4
    assert man["groups"]["b2"]["totals"]["bytes"] == 800 + 300 + 100 + 120
    tc = next(x for x in man["groups"]["b1"]["files"] if x["path"] == TEST_CACHE)
    assert tc["opaque_byte_stream"] and tc["sha256"] == sha(TEST_BYTES) and tc["cross_check"]["equal"]
    assert all(x["git"]["ignored"] and x["git"]["rule"] for g in ("b1", "b2") for x in man["groups"][g]["files"])
    assert "opaque" in man["policies"]["test_cache"] and "Gate 2c" in man["policies"]["test_cache"]
    assert (src / OUT / "files_b_inputs.txt").read_text().splitlines() == ["data/raw.parquet", TEST_CACHE, META]
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 0 and res["ok"], res["failures"]
    assert res["tracked_laptop_record_hazards"][0]["pattern"] == "results/registry.jsonl"
    assert "byte hash only" in res["test_cache"] and "Gate 2c" in res["test_cache"]
    assert any(l[1] == "c" and "to be regenerated" in l[2] for l in res["lines"])
    capsys.readouterr()
    assert M.main(["verify", "--repo", str(dst)]) == 0                     # human-readable mode
    out = capsys.readouterr().out
    assert "SUMMARY:" in out and "PASS" in out and "TEST CACHE:" in out and "trusted by the code" in out


def test_sha256sums_header_commands_work(src, tmp_path):
    if not (shutil.which("sha256sum") and shutil.which("sed") and shutil.which("sh")):
        pytest.skip("coreutils not available")
    dst = round_trip(src, tmp_path)
    sums = f"{OUT}/SHA256SUMS"
    p = subprocess.run(["sh", "-c", f"sed -n '/^# BEGIN inputs/,/^# END inputs/p' {sums} | sha256sum -c"], cwd=dst,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    p = subprocess.run(["sh", "-c", f"cd {M.QUARANTINE_REL} && sed -n '/^# BEGIN laptop_records/,/^# END "
                        "laptop_records/p' ../../transfer/SHA256SUMS | sha256sum -c"], cwd=dst, capture_output=True,
                       text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    p = subprocess.run(["sha256sum", "-c", sums], cwd=src, capture_output=True, text=True)    # source machine
    assert p.returncode == 0, p.stdout + p.stderr


def test_corrupted_file_is_detected(src, tmp_path, capsys):
    dst = round_trip(src, tmp_path)
    f = dst / "data/raw.parquet"
    b = bytearray(f.read_bytes())
    b[100] ^= 0xFF
    f.write_bytes(bytes(b))
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1 and any("SHA-256 mismatch" in x and "data/raw.parquet" in x for x in res["failures"])
    q = dst / M.QUARANTINE_REL / "results/grid/g2.npy"                     # a record in the quarantine root
    q.write_bytes(q.read_bytes() + b"x")
    rc, res = verify_json(dst, capsys=capsys)
    assert any("size" in x and "g2.npy" in x for x in res["failures"])
    (dst / TEST_CACHE).unlink()
    rc, res = verify_json(dst, capsys=capsys)
    assert any("missing" in x and TEST_CACHE in x for x in res["failures"])
    rc, res = verify_json(dst, "--skip-large-hash", "1000", capsys=capsys)  # size-only pass still sees size/missing
    assert rc == 1 and any("NOT checked" in w for w in res["warnings"])


def test_laptop_record_at_live_path_fails(src, tmp_path, capsys):
    dst = round_trip(src, tmp_path)
    rel = "results/m0/bench_val_nll.npy"
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / rel, dst / rel)
    rc, res = verify_json(dst, capsys=capsys)
    assert rc == 1
    msg = next(x for x in res["failures"] if rel in x)
    assert "LIVE PATH" in msg and f"mv {rel} {M.QUARANTINE_REL}/{rel}" in msg
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
    rc, res = verify_json(src, capsys=capsys)                              # without the flag: records at live paths
    assert rc == 1 and any("LIVE PATH" in x for x in res["failures"])


def test_csv_is_refused(src, tables, monkeypatch, capsys, tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n")
    with pytest.raises(M.RefusedFile):
        M.sha256_file(p)
    with pytest.raises(M.RefusedFile):
        M.sha256_file(tmp_path / "Y.CSV")                                   # refused before any open
    (src / "data/raw.csv").write_text("never read\n")
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
    monkeypatch.setattr(M, "GROUP_B2", tables["GROUP_B2"])
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
script, tables_json, src, dst = sys.argv[1:5]
test_caches = set(json.loads(sys.argv[5]))
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
rc = [m.main(["verify", "--repo", dst, "--json"]), m.main(["list", "--repo", src]),
      m.main(["verify", "--repo", src, "--source-machine", "--json"])]
import linecache
for e in events:
    if "lineno" in e:
        e["source_line"] = linecache.getline(e["file"], e["lineno"]).strip()
mods = sorted(x for x in sys.modules if x.split(".")[0] in ("torch", "numpy"))
sys.stderr.write("RESULT " + json.dumps({"rc": rc, "events": events, "modules": mods}) + "\n")
'''


def test_test_cache_is_only_opened_for_a_byte_hash(src, tables, tmp_path):
    dst = round_trip(src, tmp_path)
    tj = tmp_path / "tables.json"
    tj.write_text(json.dumps(tables))
    drv = tmp_path / "driver.py"
    drv.write_text(DRIVER)
    tcs = json.dumps([os.path.realpath(src / TEST_CACHE), os.path.realpath(dst / TEST_CACHE)])
    p = subprocess.run([sys.executable, "-B", "-I", str(drv), str(SCRIPT), str(tj), str(src), str(dst), tcs],
                       capture_output=True, text=True)
    line = next(l for l in p.stderr.splitlines() if l.startswith("RESULT "))
    res = json.loads(line[len("RESULT "):])
    assert res["rc"] == [0, 0, 0], p.stdout[-2000:]
    opens = [e for e in res["events"] if "open" in e]
    assert len(opens) == 3                       # verify on the H200 copy, list and source-machine verify (laptop)
    write_bits = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    for e in opens:                   # the audit event carries FileIO's mode ('r') and the os.open flags
        assert e["mode"] in ("r", "rb") and not (e["flags"] & write_bits), e
        assert e["func"] == "sha256_file" and e["source_line"] == 'with open(path, "rb") as f:', e
    assert not [e for e in res["events"] if "import" in e or "pickle" in e] and res["modules"] == []


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
