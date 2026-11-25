"""Smoke tests d (template), e (multi-output), f (thread scaling), each run fresh, then post-processed.

    uv run python -B shape-form-martin/results/m3_sr/smoke/smoke_def.py [d] [e] [f]

d  specs/d_template.toml: one chunk, then an extension in a NEW process (PySRRegressor.from_file on a template
   checkpoint), then scripts/sr_post.py (template translation, per-sub-expression scan, round trip)
e  specs/e_multiout.toml: weights (N, 3), guesses list[list[str]]; then an extension in a new process; sr_post
f  specs/f_threads8.toml and specs/f_threads20.toml: one 60 s chunk each, same spec otherwise (populations = 24),
   evaluation rates from Julia's log
Console output: evidence/<target>_console.txt; condensed records: evidence/<target>.json (collect.py).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCRIPTS = REPO / "shape-form-martin" / "scripts"
RUNS = HERE / "runs" / "smoke"
EVID = HERE / "evidence"
sys.path.insert(0, str(HERE))
import collect  # noqa: E402


def run(args, log_name) -> tuple[int, float]:
    """Run `python -B <args>` from the repository root, console appended to evidence/<log_name>."""
    t0 = time.time()
    with open(EVID / log_name, "a") as log:
        log.write(f"\n===== {time.strftime('%H:%M:%S')} {' '.join(str(c) for c in args)} =====\n")
        log.flush()
        rc = subprocess.run([sys.executable, "-B", *map(str, args)], stdout=log, stderr=subprocess.STDOUT,
                            cwd=REPO).returncode
    return rc, time.time() - t0


def fresh(target):
    d = RUNS / target / "seed0"
    if d.exists():
        shutil.rmtree(d)
    (EVID / f"{target}_console.txt").write_text("")


def sr_run(spec, target, *extra):
    return run([SCRIPTS / "sr_run.py", HERE / "specs" / spec, *extra], f"{target}_console.txt")


def sr_post(spec, target, *extra):
    return run([SCRIPTS / "sr_post.py", HERE / "specs" / spec, *extra], f"{target}_console.txt")


def smoke_d():
    t = "d_template"
    fresh(t)
    steps = [("run", *sr_run("d_template.toml", t)), ("extend 1.6 min", *sr_run("d_template.toml", t,
                                                                                    "--budget-min", "1.6")),
             ("post", *sr_post("d_template.toml", t, "--float32"))]
    return t, steps


def smoke_e():
    t = "e_multiout"
    fresh(t)
    steps = [("run", *sr_run("e_multiout.toml", t)), ("extend 1.2 min", *sr_run("e_multiout.toml", t,
                                                                                   "--budget-min", "1.2")),
             ("post", *sr_post("e_multiout.toml", t, "--float32"))]
    return t, steps


def smoke_f():
    out = []
    for th in (8, 20):
        t = f"f_threads{th}"
        fresh(t)
        out.append((t, [("run", *sr_run(f"f_threads{th}.toml", t)), ("post", *sr_post(f"f_threads{th}.toml", t))]))
    return out


def main(argv):
    which = argv or ["d", "e", "f"]
    results = {}
    for w in which:
        if w == "f":
            pairs = smoke_f()
        else:
            pairs = [{"d": smoke_d, "e": smoke_e}[w]()]
        for target, steps in pairs:
            ev = collect.collect(target)
            rec = {"steps": [{"step": s, "returncode": rc, "wall_s": round(wall, 1)} for s, rc, wall in steps]}
            path = EVID / f"{target}.json"
            doc = json.loads(path.read_text())
            doc["driver_steps"] = rec["steps"]
            path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
            results[target] = rec
            print(target, rec["steps"], "post passed:", None if ev["post"] is None else ev["post"]["passed"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
