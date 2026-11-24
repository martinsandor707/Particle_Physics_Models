"""Run several fit configs one after another (serial GPU queue), skipping finished runs.

    uv run python -B shape-form-martin/scripts/fit_queue.py shape-form-martin/configs/m2b_mlp/*.toml
    uv run python -B shape-form-martin/scripts/fit_queue.py shape-form-martin/configs/m2b_mlp      # every *.toml, sorted
    uv run python -B shape-form-martin/scripts/fit_queue.py @my_queue.txt                         # one path per line
    uv run python -B shape-form-martin/scripts/fit_queue.py ... --dry-run                         # show what would run

Every config is loaded first (a broken file stops the queue before anything runs). A config whose run directory
holds ``done.json`` (written by ``scripts/fit.py`` after the fit and the registry evaluation) is skipped. The others
run as separate ``scripts/fit.py`` processes, in order (GPU memory is released between runs; each takes the GPU
lock itself, so the queue also waits for unrelated jobs holding it). An interrupted run resumes from its checkpoint
the next time the queue reaches it. Failures are reported at the end (exit status 1); ``--stop-on-error`` stops at
the first one.

Name: not ``queue.py`` on purpose - a ``scripts/queue.py`` would shadow the standard-library module ``queue`` (which
torch imports) for every script started from this directory.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfm import train  # noqa: E402

FIT = Path(__file__).resolve().parent / "fit.py"


def expand(items) -> list:
    out = []
    for it in items:
        if it.startswith("@"):
            lines = Path(it[1:]).read_text().splitlines()
            out += expand([ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")])
            continue
        p = Path(it)
        if p.is_dir():
            out += sorted(str(q) for q in p.rglob("*.toml"))
        else:
            out.append(str(p))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="config files, directories, or @list files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args(argv)

    files = expand(args.configs)
    if not files:
        print("no config files")
        return 1
    rcs = [train.load_run_config(f) for f in files]                  # fail early on a broken file
    seen = {}
    for rc in rcs:
        key = (rc.stage, rc.name)
        if key in seen:
            raise SystemExit(f"{rc.source} and {seen[key]} both define run {rc.stage}/{rc.name}")
        seen[key] = rc.source
    status = []
    for i, rc in enumerate(rcs, 1):
        tag = f"[{i}/{len(rcs)}] {rc.stage}/{rc.name}"
        if rc.done_file.exists():
            print(f"{tag}: done marker exists, skipped")
            status.append((rc, "skipped"))
            continue
        if args.dry_run:
            print(f"{tag}: would run ({rc.source})")
            status.append((rc, "pending"))
            continue
        print(f"{tag}: running {rc.source}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-B", str(FIT), str(rc.source)])
        ok = proc.returncode == 0 and rc.done_file.exists()
        status.append((rc, "ok" if ok else f"FAILED (exit {proc.returncode})"))
        print(f"{tag}: {'ok' if ok else 'FAILED'} after {time.time() - t0:.0f} s", flush=True)
        if not ok and args.stop_on_error:
            break
    print("\nqueue summary")
    for rc, s in status:
        print(f"  {rc.stage}/{rc.name}: {s}")
    return 1 if any(s.startswith("FAILED") for _, s in status) else 0


if __name__ == "__main__":
    sys.exit(main())
