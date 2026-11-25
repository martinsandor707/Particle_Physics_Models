"""Tests of sfm.guard (the H200 file-access guard). Each scenario runs in a child process, because an audit hook
cannot be removed once installed. Temporary directories only: a fake "area" holding an output root, a cache directory
and another project's folder; the clone is the real repository (outside the fake area), read-only."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MARTIN = HERE.parent

CHILD = textwrap.dedent('''
    import json, os, subprocess, sys
    sys.path[:0] = [{src!r}, {martin!r}]
    from sfm import paths, guard                      # importing sfm installs the guard when SFM_GUARD_AREA is set
    res = {{"active": guard.active()}}

    def attempt(label, fn):
        try:
            fn()
            res[label] = "ok"
        except guard.GuardRefusedImport:
            res[label] = "refused-import"
        except guard.GuardRefused:
            res[label] = "refused"
        except ImportError as e:
            res[label] = "importerror:" + type(e).__name__
        except Exception as e:
            res[label] = "error:" + type(e).__name__
    area = os.environ.get("T_AREA")
    attempt("clone_file", lambda: open(os.path.join({martin!r}, "common.py")).read(10))
    attempt("out_file", lambda: open(os.path.join(area, "out", "a.txt"), "w").write("x"))
    attempt("cache_file", lambda: open(os.path.join(area, "cache", "b.txt"), "w").write("x"))
    attempt("other_project_file", lambda: open(os.path.join(area, "other", "secret.txt")).read())
    attempt("other_project_listdir", lambda: os.listdir(os.path.join(area, "other")))
    attempt("symlink_into_other", lambda: open(os.path.join(area, "out", "link.txt")).read())
    attempt("deferred_name_outside_area", lambda: open(os.path.join(os.environ["T_ELSEWHERE"], "pion_s1_test.pt"), "rb"))
    attempt("deferred_name_in_out", lambda: open(os.path.join(area, "out", "test_event_number.npy"), "rb"))
    attempt("foreign_parquet", lambda: open(os.path.join(os.environ["T_ELSEWHERE"], "pion_hits_with_model_outputs.parquet"), "rb"))
    attempt("import_pyarrow_parquet", lambda: __import__("pyarrow.parquet"))
    attempt("subprocess_reads_other", lambda: subprocess.run(["cat", os.path.join(area, "other", "secret.txt")],
                                                            capture_output=True, check=True))
    attempt("outside_area_file", lambda: open(os.path.join(os.environ["T_ELSEWHERE"], "plain.txt")).read())
    attempt("final_test_switch_refused", lambda: guard.enable_final_test_access())
    print("RESULT " + json.dumps(res))
''')


def _run(tmp_path, guard_on: bool):
    area = tmp_path / "area"
    for d in ("out", "cache", "other"):
        (area / d).mkdir(parents=True)
    (area / "other" / "secret.txt").write_text("other project")
    os.symlink(area / "other" / "secret.txt", area / "out" / "link.txt")
    (area / "out" / "test_event_number.npy").write_bytes(b"x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for n in ("pion_s1_test.pt", "pion_hits_with_model_outputs.parquet", "plain.txt"):
        (elsewhere / n).write_text("x")
    env = dict(os.environ, T_AREA=str(area), T_ELSEWHERE=str(elsewhere), SFM_OUT_ROOT=str(area / "out"),
               SFM_USER_CACHE=str(area / "cache"), PYTHONDONTWRITEBYTECODE="1")
    env.pop("SFM_GUARD_AREA", None)
    if guard_on:
        env["SFM_GUARD_AREA"] = str(area)
    code = CHILD.format(src=str(MARTIN / "src"), martin=str(MARTIN))
    p = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    line = next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):]), area


def test_guard_refuses_and_allows(tmp_path):
    r, area = _run(tmp_path, guard_on=True)
    assert r["active"] is True
    for k in ("clone_file", "out_file", "cache_file", "outside_area_file"):
        assert r[k] == "ok", (k, r)
    for k in ("other_project_file", "other_project_listdir", "symlink_into_other", "deferred_name_outside_area",
              "deferred_name_in_out", "foreign_parquet", "subprocess_reads_other"):
        assert r[k] == "refused", (k, r)
    assert r["import_pyarrow_parquet"] == "refused-import"
    assert r["final_test_switch_refused"].startswith(("error:", "refused")) and r["final_test_switch_refused"] != "ok"
    log = (area / "out" / "guard" / "guard.log").read_text().splitlines()
    assert len(log) >= 8
    fields = log[0].split("\t")
    assert len(fields) == 5 and fields[1].startswith("pid ") and fields[3].startswith(("<area>", "/", "pyarrow"))
    assert all(str(area) not in l for l in log)                        # the area prefix is written as <area>


def test_guard_is_inert_when_unset(tmp_path):
    r, area = _run(tmp_path, guard_on=False)
    assert r["active"] is False
    for k in ("clone_file", "out_file", "cache_file", "other_project_file", "other_project_listdir",
              "symlink_into_other", "deferred_name_outside_area", "deferred_name_in_out", "foreign_parquet",
              "subprocess_reads_other", "outside_area_file"):
        assert r[k] == "ok", (k, r)
    assert not (area / "out" / "guard").exists()


def test_sfm_data_imports_pyarrow_lazily():
    code = (f"import sys; sys.path[:0] = [{str(MARTIN / 'src')!r}, {str(MARTIN)!r}]\n"
            "import sfm.data, sfm.registry, sfm.legacy\n"
            "print(sorted(m for m in sys.modules if m.startswith('pyarrow.parquet') or m.startswith('pyarrow._parquet')))")
    p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip().splitlines()[-1] == "[]"
