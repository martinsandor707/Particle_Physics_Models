"""Condense smoke-test run directories into small evidence records (evidence/<target>.json).

    uv run python -B shape-form-martin/results/m3_sr/smoke/collect.py [target ...]

Reads runs/smoke/<target>/seed0/{progress.json, hall_of_fame.json, post/*.json, search.log, spec_snapshot.json} and
keeps what the STATUS file quotes: per-chunk timings, evaluation rates, state hashes, restores, post-processing
summaries, warnings from Julia's log, the file listing with sizes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "smoke"
EVID = HERE / "evidence"

CHUNK_KEYS = ("index", "timeout_s", "fit_wall_s", "fit_overhead_s", "export_s", "interrupted", "with_guesses",
              "torch_in_process", "evals_per_s_mean", "evals_per_s_last", "n_front", "tier_best",
              "hof_not_worse_than_prev_chunk", "hof_regressions", "threads", "state_summary")


def listing(d: Path) -> list[str]:
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file():
            out.append(f"{p.relative_to(d)}  {p.stat().st_size}")
    return out


def collect(target: str) -> dict:
    d = RUNS / target / "seed0"
    prog = json.loads((d / "progress.json").read_text())
    hof = json.loads((d / "hall_of_fame.json").read_text())
    log = (d / "search.log").read_text(errors="replace") if (d / "search.log").exists() else ""
    rates = [float(x) for x in re.findall(r"Expressions evaluated per second:\s*([0-9.eE+-]+)", log)]
    posts = sorted((d / "post").glob("post_chunk*.json")) if (d / "post").exists() else []
    post = json.loads(posts[-1].read_text()) if posts else None
    ev = {
        "target": target, "status": prog["status"], "done": prog["done"], "budget_s": prog["budget_s"],
        "elapsed_search_s": prog["elapsed_search_s"], "chunks_completed": prog["chunks_completed"],
        "julia": prog.get("julia"), "startups": prog.get("startups"), "torch_in_process": prog.get("torch_in_process"),
        "restores": prog.get("restores"), "aborted_chunks": prog.get("aborted_chunks"),
        "fallback_used": prog.get("fallback_used"), "errors": prog.get("errors"), "plateau_raw": prog.get("plateau"),
        "chunks": [{k: h.get(k) for k in CHUNK_KEYS} for h in prog["history"]],
        "evals_per_s_all_prints": rates,
        "julia_log_warnings": sorted(set(re.findall(r"Warning: ([^\n]+)", log))),
        "recreated_populations_in_log": log.count("Recreating population"),
        "hof": {"nout": hof["nout"], "print_precision": hof.get("print_precision"), "precision": hof.get("precision"),
                "predictions": hof.get("predictions"),
                "outputs": [{"n": len(o["equations"]),
                             "front": [(e["complexity"], e["loss"], e["equation"]) for e in o["equations"]],
                             **({"subexpressions": [e.get("subexpressions") for e in o["equations"]]}
                                if hof.get("template") else {})} for o in hof["outputs"]]},
        "post": None if post is None else {"passed": post["passed"], "failures": post["failures"],
                                           "loss_kind": post["loss_kind"], "vocabulary": post["vocabulary"],
                                           "template": post["template"], "plateau": post["plateau"],
                                           "summaries": [o["summary"] for o in post["outputs"]]},
        "files": listing(d),
    }
    EVID.mkdir(exist_ok=True)
    (EVID / f"{target}.json").write_text(json.dumps(ev, indent=1, default=str) + "\n")
    return ev


def sanitize_evidence() -> None:
    """Replace the absolute repository path by <repo> in every evidence file (the repository is public)."""
    repo = str(HERE.parents[3])
    for p in EVID.iterdir():
        if p.suffix in (".json", ".txt"):
            s = p.read_text(errors="replace")
            if repo in s:
                p.write_text(s.replace(repo, "<repo>"))


def main(argv):
    targets = argv or sorted(p.name for p in RUNS.iterdir() if (p / "seed0" / "progress.json").exists())
    for t in targets:
        ev = collect(t)
        print(t, ev["status"], ev["chunks_completed"], "chunks,", round(ev["elapsed_search_s"]), "s; post:",
              None if ev["post"] is None else ev["post"]["passed"])
    sanitize_evidence()


if __name__ == "__main__":
    main(sys.argv[1:])
