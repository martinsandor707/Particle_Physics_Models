"""Smoke test a: kill (SIGKILL mid-chunk) and resume, SIGINT stop and resume, extension, fallback restart.

    uv run python -B shape-form-martin/results/m3_sr/smoke/smoke_a.py

Drives scripts/sr_run.py as separate processes on specs/a_resume.toml (toy 2-D target, 8 Julia threads, 20 s chunks)
and writes evidence/a.json.  Steps:
  1  fresh start; when chunks 0 and 1 are complete and chunk 2 has run for ~8 s: SIGKILL the search process
  2  resume: must restore checkpoint.pkl = end of chunk 1 (state hashes equal), salvage the killed chunk's in-flight
     front as guesses, and the first resumed chunk's front must not be worse than chunk 1's at any complexity;
     during its second chunk: SIGINT (PySR stops the chunk early, checkpoints; the runner stops, resumable)
  3  resume to the end of the budget (DONE)
  4  extension with --budget-min 2.5 (same seed continues, DONE retired and written again)
  5  fallback: checkpoint.pkl truncated, extension to 2.9 min -> restore fails -> restart seeded with the saved
     hall of fame (guesses); its first chunk's front must not be worse than the last chunk before
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SR_RUN = REPO / "shape-form-martin" / "scripts" / "sr_run.py"
SPEC = HERE / "specs" / "a_resume.toml"
SEED = HERE / "runs" / "smoke" / "a_resume" / "seed0"
EVID = HERE / "evidence"


def prog():
    try:
        return json.loads((SEED / "progress.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def start(extra=(), log_name="a_console.txt"):
    log = open(EVID / log_name, "a")
    log.write(f"\n===== {time.strftime('%H:%M:%S')} sr_run {' '.join(extra)} =====\n")
    log.flush()
    return subprocess.Popen([sys.executable, "-B", str(SR_RUN), str(SPEC), *extra], stdout=log,
                            stderr=subprocess.STDOUT, cwd=REPO), time.time()


def wait_for(cond, timeout, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = prog()
        if p is not None and cond(p):
            return p
        if proc is not None and proc.poll() is not None:
            return prog()
        time.sleep(0.25)
    raise TimeoutError("condition not reached")


def in_flight_for(p, k, secs):
    inf = p.get("in_flight")
    if not inf or inf["chunk"] != k:
        return False
    import datetime as dt
    t = dt.datetime.fromisoformat(inf["started_at"].replace("Z", "+00:00")).timestamp()
    return time.time() - t >= secs


def snap_rows(idx):
    sys.path.insert(0, str(REPO / "shape-form-martin" / "src"))
    from sfm.sr import search as S
    c = sorted((SEED / "hof_snapshots").glob(f"chunk{idx:03d}_*_hall_of_fame.csv"))
    return S.read_hof_csv(c[-1]) if c else None


def main():
    EVID.mkdir(exist_ok=True)
    if SEED.exists():
        shutil.rmtree(SEED)
    (EVID / "a_console.txt").write_text("")
    ev = {"spec": str(SPEC.relative_to(REPO)), "steps": []}

    # 1 -- fresh start, SIGKILL in the middle of chunk 2
    proc, t0 = start()
    p = wait_for(lambda q: q["chunks_completed"] >= 2 and in_flight_for(q, 2, 8.0), 240, proc)
    ck = SEED / "pysr" / "checkpoint.pkl"
    ck_mtime = ck.stat().st_mtime
    csv_mtime = (SEED / "pysr" / "hall_of_fame.csv").stat().st_mtime
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    t_kill = time.time()
    h1 = p["history"][1]
    ev["steps"].append({
        "step": "1 fresh start, SIGKILL during chunk 2", "returncode": proc.returncode, "wall_s": t_kill - t0,
        "chunks_completed_at_kill": p["chunks_completed"], "in_flight_at_kill": p["in_flight"],
        "checkpoint_mtime_minus_kill_s": ck_mtime - t_kill,
        "checkpoint_written_before_chunk2_start": ck_mtime <= __import__("datetime").datetime.fromisoformat(
            p["in_flight"]["started_at"].replace("Z", "+00:00")).timestamp(),
        "run_dir_csv_newer_than_checkpoint_s": csv_mtime - ck_mtime,
        "chunk1_state_summary": h1.get("state_summary"),
        "history": [{k: h.get(k) for k in ("index", "timeout_s", "fit_wall_s", "fit_overhead_s", "export_s",
                                           "evals_per_s_mean", "tier_best", "torch_in_process")}
                    for h in p["history"]]})

    # 2 -- resume; SIGINT during its second chunk
    proc, t0 = start()
    p = wait_for(lambda q: q["chunks_completed"] >= 3 and in_flight_for(q, 3, 5.0), 240, proc)
    proc.send_signal(signal.SIGINT)
    rc = proc.wait(timeout=120)
    p = prog()
    rest = p["restores"][-1]
    ab = p["aborted_chunks"][-1] if p["aborted_chunks"] else None
    h2 = p["history"][2]
    h3 = p["history"][3] if len(p["history"]) > 3 else None
    ev["steps"].append({
        "step": "2 resume after SIGKILL, then SIGINT during the next chunk", "returncode": rc,
        "wall_s": time.time() - t0, "restore": rest, "aborted_chunk": ab,
        "first_resumed_chunk": {k: h2.get(k) for k in ("index", "with_guesses", "hof_not_worse_than_prev_chunk",
                                                        "hof_regressions", "fit_wall_s", "fit_overhead_s",
                                                        "tier_best")},
        "sigint_chunk": None if h3 is None else {k: h3.get(k) for k in ("index", "interrupted", "timeout_s",
                                                                         "fit_wall_s", "hof_not_worse_than_prev_chunk")},
        "status_after": p["status"], "stop_reason": p.get("stop_reason"),
        "chunks_completed": p["chunks_completed"]})

    # 3 -- resume to the end of the budget
    proc, t0 = start()
    rc = proc.wait(timeout=300)
    p = prog()
    ev["steps"].append({"step": "3 resume to DONE", "returncode": rc, "wall_s": time.time() - t0,
                        "restore": p["restores"][-1], "done": p["done"], "done_file": (SEED / "DONE.json").exists(),
                        "elapsed_search_s": p["elapsed_search_s"], "budget_s": p["budget_s"],
                        "chunks_completed": p["chunks_completed"],
                        "new_chunks": [{k: h.get(k) for k in ("index", "hof_not_worse_than_prev_chunk", "fit_wall_s")}
                                       for h in p["history"][ev["steps"][-1]["chunks_completed"]:]]})

    # 4 -- extension
    n_before = p["chunks_completed"]
    proc, t0 = start(["--budget-min", "2.5"])
    rc = proc.wait(timeout=300)
    p = prog()
    ev["steps"].append({"step": "4 extension --budget-min 2.5", "returncode": rc, "wall_s": time.time() - t0,
                        "restore": p["restores"][-1], "done": p["done"],
                        "superseded_done_files": sorted(x.name for x in SEED.glob("DONE.superseded-*.json")),
                        "budget_history": p["budget_history"], "chunks_completed": p["chunks_completed"],
                        "new_chunks": [{k: h.get(k) for k in ("index", "hof_not_worse_than_prev_chunk", "fit_wall_s")}
                                       for h in p["history"][n_before:]]})

    # 5 -- fallback: truncate the checkpoint
    n_before = p["chunks_completed"]
    data = ck.read_bytes()
    ck.write_bytes(data[: len(data) // 3])
    proc, t0 = start(["--budget-min", "2.9"])
    rc = proc.wait(timeout=300)
    p = prog()
    new = p["history"][n_before:]
    ev["steps"].append({"step": "5 truncated checkpoint -> fallback restart from the saved hall of fame",
                        "returncode": rc, "wall_s": time.time() - t0, "restore": p["restores"][-1],
                        "fallback_used": p["fallback_used"], "done": p["done"],
                        "broken_checkpoints": sorted(x.name for x in (SEED / "pysr").glob("checkpoint.pkl.broken-*")),
                        "new_chunks": [{k: h.get(k) for k in ("index", "with_guesses", "hof_not_worse_than_prev_chunk",
                                                              "hof_regressions", "fit_wall_s", "fit_overhead_s")}
                                       for h in new]})

    # verdicts
    s1, s2, s3, s4, s5 = ev["steps"]
    checks = {
        "kill_mid_chunk": s1["returncode"] == -9 and s1["chunks_completed_at_kill"] == 2,
        "checkpoint_is_end_of_chunk1": bool(s1["checkpoint_written_before_chunk2_start"]),
        "restored_from_file_no_fallback": s2["restore"]["method"] == "from_file",
        "restored_state_identical_to_chunk1_end": s2["restore"].get("same_as_chunk_end") is True,
        "aborted_chunk_detected_and_salvaged": bool(s2["aborted_chunk"]) and bool(s2["restore"].get("salvaged_guesses")),
        "first_resumed_chunk_front_not_worse": s2["first_resumed_chunk"]["hof_not_worse_than_prev_chunk"] is True,
        "sigint_stops_chunk_and_runner": (s2["sigint_chunk"] is not None and s2["sigint_chunk"]["interrupted"] is True
                                          and s2["stop_reason"] == "sigint" and s2["returncode"] == 0),
        "resume_to_done": s3["done"] and s3["done_file"] and s3["restore"].get("same_as_chunk_end") is True,
        "extension_continues_same_seed": (s4["done"] and s4["restore"]["method"] == "from_file"
                                          and s4["restore"].get("same_as_chunk_end") is True
                                          and len(s4["superseded_done_files"]) >= 1
                                          and all(c["hof_not_worse_than_prev_chunk"] for c in s4["new_chunks"])),
        "fallback_restart_from_hof": (s5["fallback_used"] and s5["restore"]["method"] == "fallback_guesses"
                                      and s5["restore"]["n_guesses"] > 0 and s5["done"]
                                      and len(s5["broken_checkpoints"]) == 1),
        "fallback_first_chunk_front_not_worse": bool(s5["new_chunks"])
        and s5["new_chunks"][0]["hof_not_worse_than_prev_chunk"] is True,
    }
    ev["checks"] = checks
    ev["passed"] = all(checks.values())
    (EVID / "a.json").write_text(json.dumps(ev, indent=1, default=str) + "\n")
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("smoke a:", "PASS" if ev["passed"] else "FAIL")
    return 0 if ev["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
