"""Run, resume, extend or inspect chunked PySR searches (PLAN.md §7).

    uv run python shape-form-martin/scripts/sr_run.py SPEC [--seeds 0 1 2] [--threads N] [--budget-min M]
                                                          [--chunk-min C] [--status] [--allow-spec-change]

* Seeds run one after another in this process (one Julia start).  Output: results/m3_sr/<step>/<target>/seed<k>/.
* Running the same command again resumes every unfinished seed from its last completed chunk; raising the budget
  (in the spec or with --budget-min) extends finished seeds.
* --status prints progress and the plateau flags per seed and never starts Julia.
* Stopping: Ctrl-C / SIGINT stops the running chunk early (PySR returns partial results and checkpoints); SIGTERM or
  `touch <seed_dir>/STOP` stops after the current chunk.  SIGKILL loses only the chunk in flight.

Recommended: a [search_config] table, `kwargs = {stage = "stage1", ndim = 3, loss = "gauge"}` (and `template = ...`),
which takes operators, constraints, loss, maxsize, precision and print_precision from
`sfm.sr.operators.search_config(**kwargs)` (search.resolve_search_config; unknown keys are refused).
Alternatively the [operators] table: explicit PySR settings, or `module = "sfm.sr.operators"` with `attr = "<name>"`
(a dict, an object with `pysr_kwargs()`, or a callable taking `kwargs`) returning operator settings only, optionally
followed by explicit keys that override the module's.  This process must never import torch: the module import is
checked, progress.json records `torch_in_process` per chunk, and Julia refuses to start after torch.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Mapping
from pathlib import Path

sys.dont_write_bytecode = True
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sfm.sr import search  # noqa: E402  (standard library + numpy only)


def load_vocabulary(source: Mapping) -> dict:
    """Operator settings from a module (e.g. sfm.sr.operators), plus explicit overrides from the spec."""
    mod_name = source["module"]
    mod = importlib.import_module(mod_name)
    if "torch" in sys.modules:
        raise RuntimeError(f"importing {mod_name} pulled in torch; the search process must stay torch-free")
    attr = source.get("attr")
    obj = getattr(mod, attr) if attr else mod
    kwargs = dict(source.get("kwargs") or {})
    if hasattr(obj, "pysr_kwargs") and callable(obj.pysr_kwargs):
        vocab = obj.pysr_kwargs(**kwargs)
    elif callable(obj) and not isinstance(obj, Mapping):
        vocab = obj(**kwargs)
    elif isinstance(obj, Mapping):
        vocab = obj
    else:
        raise TypeError(f"{mod_name}.{attr}: expected a mapping, a callable or an object with pysr_kwargs()")
    vocab = dict(vocab)
    ignored = sorted(set(vocab) - set(search.OPERATOR_KEYS) - {"binary", "unary"})
    if ignored:
        # e.g. operators.search_config also returns the loss, maxsize, precision: dropping them would silently
        # change the search, so refuse and point to the table that routes every key
        raise search.SpecError(f"{mod_name}.{attr} returned non-operator keys {ignored}; use a [search_config] "
                               f"table for a full search configuration")
    vocab.update(source.get("overrides") or {})
    if "torch" in sys.modules:
        raise RuntimeError("torch got imported while building the vocabulary")
    return vocab


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="run spec (.toml or .json)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="seeds to run (default: the spec's)")
    ap.add_argument("--threads", type=int, default=None, help="Julia threads (default: the spec's, 20)")
    ap.add_argument("--budget-min", type=float, default=None, help="total search minutes per seed (overrides spec)")
    ap.add_argument("--chunk-min", type=float, default=None, help="chunk length in minutes (default 5)")
    ap.add_argument("--status", action="store_true", help="print progress and plateau flags; do not start Julia")
    ap.add_argument("--allow-spec-change", action="store_true",
                    help="resume although search-defining settings differ from the seed's snapshot")
    a = ap.parse_args(argv)

    raw = search.apply_overrides(search.load_spec_file(a.spec), threads=a.threads, budget_min=a.budget_min,
                                 chunk_min=a.chunk_min, seeds=a.seeds)
    spec = search.parse_spec(search.resolve_search_config(raw))
    if a.status:
        print(search.format_status(spec, spec.seeds))
        return 0

    if spec.operators_source is not None:
        spec = spec.with_operators(load_vocabulary(spec.operators_source))
    if not search.configure_julia_env(spec.threads):
        print("[sr_run] warning: Julia already running; thread setting ignored", file=sys.stderr)
    search.install_sigterm_handler()

    rc = 0
    for seed in spec.seeds:
        try:
            search.SeedRun(spec, seed, allow_spec_change=a.allow_spec_change).run()
        except search.RunnerBusy as e:
            print(f"[sr_run] seed {seed}: {e}; skipped", file=sys.stderr)
            rc = 3
        except KeyboardInterrupt:
            print(f"[sr_run] interrupted during seed {seed}; state is resumable", file=sys.stderr)
            return 130
        if search._STOP_REQUESTED:
            print("[sr_run] SIGTERM: stopping after this seed's current chunk", file=sys.stderr)
            return 143
        st = search.seed_status(spec, seed)
        if st.get("status") == "stopped":
            print(f"[sr_run] seed {seed} stopped before its budget was used; not starting later seeds",
                  file=sys.stderr)
            return 0
    print(search.format_status(spec, spec.seeds))
    return rc


if __name__ == "__main__":
    sys.exit(main())
