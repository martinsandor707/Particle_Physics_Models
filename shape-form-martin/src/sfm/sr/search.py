"""Chunked, resumable PySR search runner (Milestone 3, PLAN.md §7 "Per-seed checkpoints").

The search process must never import torch (juliacall after torch can segfault), so this module imports only the
standard library and numpy at import time, and imports PySR lazily (``_pysr()``) after the Julia thread count has been
placed in the environment (``configure_julia_env``).  ``sfm.paths`` is used only for the default results directory
(standard library only).

One seed = one ``PySRRegressor`` with a fixed ``output_directory``/``run_id``, driven through a loop of ``fit`` calls
("chunks", ≈ 5 min each, ``warm_start=True``) that continue the same populations.  PySR 2.7.3 writes
``checkpoint.pkl`` (with the serialised Julia search state) only at the start of a fresh ``fit`` and at the end of every
``fit``, so after each chunk the following are on disk in ``results/m3_sr/<step>/<target>/seed<k>/``::

    pysr/checkpoint.pkl              PySR's resumable state (written by PySR, atomically)
    pysr/hall_of_fame*.csv           PySR's Pareto front (rewritten by Julia during the search)
    hof_snapshots/chunkNNN_<UTC>_*.csv   timestamped copies of the hall of fame after every chunk
    progress.json                    atomically written bookkeeping (elapsed search seconds, chunk index, best loss
                                     per complexity tier, per-chunk history, loss-plateau flag)
    hall_of_fame.json                full hall of fame (complexity, loss, equation strings exactly as PySR writes
                                     them; per-sub-expression strings for template expressions)
    DONE.json                        done marker once the budget is used up
    search.log                       Julia's own console output (one file per seed, appended)
    events.jsonl                     one line per start / chunk / resume / fallback / stop / done

Resume: started again with the same spec, the runner restores ``pysr/checkpoint.pkl`` with
``PySRRegressor.from_file(run_directory=...)`` and continues with ``warm_start=True``.  If that fails, the broken
checkpoint is moved aside and the search restarts seeded with the saved hall of fame through ``guesses``; the fallback
is recorded in ``progress.json``.  A chunk killed in flight is lost (the state is the end of the previous chunk), but its
in-flight hall of fame is snapshotted and, by default, fed back as ``guesses`` into the first resumed chunk.

Extend: raising the total budget (spec or ``--budget-min``) continues the same seed; the done marker is retired.

Plateau (loss part of PLAN §7): over the last ``window`` (3) chunks the best loss at every tier improved by less than
``rel_tol`` (1 %).  The second part of the rule (no new front member improves the refitted validation NLL by > 1 SE) is
computed elsewhere and recorded with ``record_val_nll_check``; ``plateau_status`` combines both.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import datetime as _dt
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import sys
import time
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------------------------------------
SCHEMA_PROGRESS = "sfm.sr.progress/1"
SCHEMA_HOF = "sfm.sr.hof/1"
RUN_ID = "pysr"                       # PySR run_id: the run directory is <seed_dir>/pysr
PROGRESS_FILE = "progress.json"
DONE_FILE = "DONE.json"
HOF_JSON = "hall_of_fame.json"
SPEC_SNAPSHOT = "spec_snapshot.json"
SNAPSHOT_DIR = "hof_snapshots"
EVENTS_FILE = "events.jsonl"
LOG_FILE = "search.log"
STOP_FILE = "STOP"                     # touch <seed_dir>/STOP to stop after the current chunk
LOCK_FILE = ".lock"
VAL_NLL_FILE = "plateau_val_nll.json"  # written by the refit side (record_val_nll_check)

DEFAULT_TIERS = (12, 20, 30)
DEFAULT_CHUNK_MIN = 5.0
DEFAULT_MIN_CHUNK_S = 30.0
DEFAULT_THREADS = 20
NITERATIONS_PER_CHUNK = 100_000_000    # effectively unbounded: every chunk is ended by timeout_in_seconds

# PySR keyword arguments that the runner owns; a spec may not set them in [pysr].
RUNNER_OWNED_KW = frozenset({
    "warm_start", "output_directory", "run_id", "timeout_in_seconds", "niterations", "random_state",
    "temp_equation_file", "guesses", "expression_spec", "binary_operators", "unary_operators", "operators",
    "constraints", "nested_constraints", "complexity_of_operators", "extra_sympy_mappings", "elementwise_loss",
    "loss_function", "loss_function_expression", "maxsize", "populations", "procs", "cluster_manager",
    "input_stream", "deterministic", "delete_tempfiles", "tempdir",
})
# Runner defaults for PySR keywords (overridable from [pysr]).
PYSR_DEFAULTS: dict[str, Any] = {
    "parallelism": "multithreading",
    "precision": 64,
    "print_precision": 15,     # PySR's default (5) makes the exported equation strings lossy
    "batching": False,         # a row-order dependent custom loss needs the full, ordered data set
    "verbosity": 1,            # Julia prints its search state (and evaluation rate) to search.log
    "progress": False,
    "model_selection": "best",
}
# Spec keys that may change between runs of the same seed without breaking resume (bookkeeping only).
SOFT_KEYS = frozenset({"seeds", "threads", "budget_minutes", "chunk_minutes", "min_chunk_seconds", "tiers",
                       "plateau", "salvage_aborted", "output_root", "description", "log_julia_output"})
SOFT_PYSR_KEYS = frozenset({"verbosity", "progress", "print_precision", "update_verbosity", "model_selection"})
OPERATOR_KEYS = ("binary_operators", "unary_operators", "operators", "constraints", "nested_constraints",
                 "complexity_of_operators", "extra_sympy_mappings", "sympy_mappings")
LOSS_KEYS = ("elementwise_loss", "loss_function", "loss_function_expression")

_NPZ_PLACEHOLDER = re.compile(r"\{\{\s*npz:([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_EVALS_RE = re.compile(r"Expressions evaluated per second:\s*([0-9.]+(?:[eE][+-]?\d+)?)")
_JULIA_F32 = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))f([+-]?\d+)$")

_STOP_REQUESTED = False                # set by the SIGTERM handler: stop after the current chunk


class SpecError(ValueError):
    """Invalid or inconsistent run spec."""


class RunnerBusy(RuntimeError):
    """Another runner holds the seed directory's lock."""


# --------------------------------------------------------------------------------------------------------------------
# small utilities (no Julia)
# --------------------------------------------------------------------------------------------------------------------
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def json_safe(obj: Any) -> Any:
    """Recursively convert to JSON-serialisable values; non-finite floats become None."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if callable(obj):
        return f"<callable {getattr(obj, '__name__', type(obj).__name__)}>"
    return repr(obj)


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (temp file in the same directory, fsync, os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    with contextlib.suppress(OSError):
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(json_safe(obj), indent=1, allow_nan=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def append_event(seed_dir: Path, kind: str, **data: Any) -> None:
    rec = {"at": utc_now(), "event": kind, **data}
    with open(Path(seed_dir) / EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(rec)) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canon(obj: Any) -> str:
    return json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------------------------------------------------
# hall-of-fame CSV parsing, tiers, monotonicity, plateau (no Julia)
# --------------------------------------------------------------------------------------------------------------------
def parse_julia_float(s: str) -> float:
    """Parse a loss as Julia prints it (Float64 `0.25`, Float32 `0.25f0`, `Inf`, `NaN`)."""
    t = s.strip()
    m = _JULIA_F32.match(t)
    if m:
        t = f"{m.group(1)}e{m.group(2)}"
    low = t.lower()
    if low in ("inf", "+inf", "inf32", "inf16"):
        return math.inf
    if low in ("-inf", "-inf32"):
        return -math.inf
    if low.startswith("nan"):
        return math.nan
    return float(t)


def read_hof_csv(path: Path) -> list[dict]:
    """Rows of a PySR hall-of-fame CSV: {complexity:int, loss:float, loss_str:str, equation:str (verbatim)}."""
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        if not r or r.get("Complexity") in (None, ""):
            continue
        rows.append({"complexity": int(r["Complexity"]), "loss": parse_julia_float(r["Loss"]),
                     "loss_str": r["Loss"].strip(), "equation": r["Equation"]})
    rows.sort(key=lambda d: d["complexity"])
    return rows


def hof_csv_names(nout: int) -> list[str]:
    """PySR's file names: hall_of_fame.csv, or hall_of_fame_output{1..n}.csv for multi-output searches."""
    return ["hall_of_fame.csv"] if nout == 1 else [f"hall_of_fame_output{j}.csv" for j in range(1, nout + 1)]


def read_run_hofs(run_dir: Path, nout: int) -> list[list[dict]] | None:
    """Per-output rows from the run directory (``.csv``, falling back to ``.csv.bak``); None if missing."""
    out = []
    for name in hof_csv_names(nout):
        p = Path(run_dir) / name
        rows = None
        for cand in (p, p.with_name(p.name + ".bak")):
            if cand.exists():
                try:
                    rows = read_hof_csv(cand)
                    break
                except Exception:  # noqa: BLE001  half-written file: try the .bak twin
                    continue
        if rows is None:
            return None
        out.append(rows)
    return out


def staircase(rows: Sequence[Mapping], c: int) -> float:
    """Best loss among rows with complexity <= c (inf if none)."""
    best = math.inf
    for r in rows:
        if r["complexity"] <= c and not math.isnan(r["loss"]):
            best = min(best, r["loss"])
    return best


def tier_bests(rows: Sequence[Mapping], tiers: Iterable[int]) -> dict[str, float | None]:
    out = {}
    for t in tiers:
        v = staircase(rows, int(t))
        out[str(int(t))] = v if math.isfinite(v) else None
    return out


def hof_regressions(prev: Sequence[Mapping], new: Sequence[Mapping], rtol: float = 1e-9,
                    atol: float = 1e-15) -> list[dict]:
    """Complexities at which the new front is worse than the previous one.

    The CSV lists only the dominating (Pareto) members, so the comparison is between staircases: for each complexity
    c of the previous front, best_new(<= c) must not exceed best_prev(c) (up to rounding).  Empty list = not worse."""
    bad = []
    for r in prev:
        c, lp = r["complexity"], r["loss"]
        if not math.isfinite(lp):
            continue
        ln = staircase(new, c)
        if not (ln <= lp + rtol * abs(lp) + atol):
            bad.append({"complexity": c, "prev": lp, "new": ln})
    return bad


def _rel_improvement(ref: float | None, cur: float | None) -> float:
    if ref is None and cur is None:
        return 0.0
    if ref is None:
        return math.inf
    if cur is None:
        return -math.inf
    if ref == cur:
        return 0.0
    return (ref - cur) / max(abs(ref), 1e-300)


def plateau_check(history: Sequence[Mapping], tiers: Sequence[int], window: int = 3,
                  rel_tol: float = 0.01) -> dict:
    """Loss part of the plateau rule: over the last `window` chunks the best loss at each tier (and each output)
    improved by less than `rel_tol` (relative).  Compares the record after the last chunk with the record `window`
    chunks earlier, so it needs `window + 1` completed chunks; with fewer, ``plateau`` is None."""
    n = len(history)
    res: dict[str, Any] = {"rule": f"best loss at every tier improved < {rel_tol:.3g} (relative) over the last "
                                   f"{window} chunks", "window": window, "rel_tol": rel_tol}
    if n < window + 1:
        res.update(plateau=None, reason=f"needs {window + 1} completed chunks, have {n}")
        return res
    ref, cur = history[n - 1 - window], history[n - 1]
    rel: list[dict[str, float]] = []
    ok = True
    for o, (tb_ref, tb_cur) in enumerate(zip(ref["tier_best"], cur["tier_best"])):
        d = {}
        for t in tiers:
            k = str(int(t))
            r = _rel_improvement(tb_ref.get(k), tb_cur.get(k))
            d[k] = r
            if not (r < rel_tol):
                ok = False
        rel.append(d)
    res.update(plateau=bool(ok), ref_chunk=ref["index"], cur_chunk=cur["index"], rel_improvement=rel)
    return res


# --------------------------------------------------------------------------------------------------------------------
# run spec
# --------------------------------------------------------------------------------------------------------------------
def load_spec_file(path: str | os.PathLike) -> dict:
    """Read a JSON (``.json``) or TOML (``.toml``) run spec into a plain dict; records its directory."""
    p = Path(path).resolve()
    if p.suffix.lower() == ".toml":
        with open(p, "rb") as f:
            raw = tomllib.load(f)
    elif p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise SpecError(f"spec must be .json or .toml: {p}")
    raw.setdefault("_spec_path", str(p))
    return raw


def _default_output_root() -> Path:
    try:
        from sfm.paths import RESULTS_DIR  # standard library only
        return Path(RESULTS_DIR) / "m3_sr"
    except Exception:  # noqa: BLE001  module used outside the package layout
        return Path(__file__).resolve().parents[3] / "results" / "m3_sr"


def _int_keys(d: Mapping) -> dict:
    return {int(k): list(v) for k, v in d.items()}


@dataclass
class RunSpec:
    """A parsed run spec.  `operators` holds plain PySR operator settings (lists/dicts of strings and numbers);
    `operators_source` is set instead when the vocabulary is to be loaded from a module (see scripts/sr_run.py)."""
    step: str
    target: str
    data: Path
    seeds: list[int]
    threads: int
    budget_s: float
    chunk_s: float
    min_chunk_s: float
    tiers: tuple[int, ...]
    plateau_window: int
    plateau_rel_tol: float
    maxsize: int
    populations: int | None
    operators: dict
    operators_source: dict | None
    loss: dict
    template: dict | None
    guesses: list | None
    pysr: dict
    output_root: Path
    salvage_aborted: bool = True
    log_julia_output: bool = True
    spec_path: Path | None = None
    raw: dict = field(default_factory=dict)

    # -- derived -------------------------------------------------------------------------------------------------
    def seed_dir(self, seed: int) -> Path:
        return Path(self.output_root) / self.step / self.target / f"seed{int(seed)}"

    def resolved_populations(self, threads: int | None = None) -> int:
        """`populations` or, by default, 3 x threads (PLAN: about 3 x threads)."""
        if self.populations is not None:
            return int(self.populations)
        return 3 * int(threads or self.threads)

    def with_operators(self, ops: Mapping) -> "RunSpec":
        """A copy with the operator vocabulary filled in (e.g. from sfm.sr.operators)."""
        new = copy.copy(self)
        new.operators = normalize_operators(ops)
        return new

    def identity(self, populations: int, data_sha256: str) -> dict:
        """Everything that must stay fixed for a seed to be resumed (not budget, threads, chunk length, tiers)."""
        pysr_hard = {k: v for k, v in sorted(self.pysr.items()) if k not in SOFT_PYSR_KEYS}
        ops = {k: (sorted(v) if k in ("extra_sympy_mappings", "sympy_mappings") and isinstance(v, Mapping) else v)
               for k, v in self.operators.items()}
        return {"step": self.step, "target": self.target, "data_sha256": data_sha256, "maxsize": self.maxsize,
                "populations": populations, "operators": ops, "loss": self.loss, "template": self.template,
                "pysr": pysr_hard}


def normalize_operators(ops: Mapping) -> dict:
    """Validate and normalise an operator vocabulary into PySR keyword arguments.

    Accepted keys: binary_operators / unary_operators (or the aliases binary / unary), operators (arity -> list),
    constraints, nested_constraints, complexity_of_operators, and sympy_mappings (name -> sympy expression string in
    x, y) or extra_sympy_mappings (name -> callable).  Unknown keys raise."""
    ops = dict(ops)
    if "binary" in ops:
        ops["binary_operators"] = ops.pop("binary")
    if "unary" in ops:
        ops["unary_operators"] = ops.pop("unary")
    for k in ("module", "attr", "kwargs"):
        ops.pop(k, None)
    unknown = sorted(set(ops) - set(OPERATOR_KEYS))
    if unknown:
        raise SpecError(f"unknown operator settings {unknown}; allowed: {list(OPERATOR_KEYS)}")
    out: dict[str, Any] = {}
    if "operators" in ops and ops["operators"] is not None:
        if ops.get("binary_operators") or ops.get("unary_operators"):
            raise SpecError("use either `operators` (arity -> list) or binary_operators/unary_operators")
        out["operators"] = _int_keys(ops["operators"])
    for k in ("binary_operators", "unary_operators"):
        if ops.get(k) is not None:
            out[k] = [str(s) for s in ops[k]]
    if not any(k in out for k in ("operators", "binary_operators", "unary_operators")):
        raise SpecError("no operators given")
    if ops.get("constraints"):
        out["constraints"] = {str(k): (tuple(v) if isinstance(v, (list, tuple)) else int(v))
                              for k, v in ops["constraints"].items()}
    if ops.get("nested_constraints"):
        out["nested_constraints"] = {str(k): {str(k2): int(v2) for k2, v2 in v.items()}
                                     for k, v in ops["nested_constraints"].items()}
    if ops.get("complexity_of_operators"):
        out["complexity_of_operators"] = {str(k): v for k, v in ops["complexity_of_operators"].items()}
    if ops.get("sympy_mappings"):
        out["sympy_mappings"] = {str(k): str(v) for k, v in ops["sympy_mappings"].items()}
    if ops.get("extra_sympy_mappings"):
        out["extra_sympy_mappings"] = dict(ops["extra_sympy_mappings"])
    return out


def custom_operator_definitions(ops: Mapping) -> list[str]:
    """The Julia definitions among the operators (strings containing '('), e.g. 'linv(x) = 1 / (1 + x * x)'."""
    lists: list[str] = []
    if "operators" in ops:
        for v in ops["operators"].values():
            lists.extend(v)
    lists.extend(ops.get("binary_operators", []) or [])
    lists.extend(ops.get("unary_operators", []) or [])
    return [s for s in lists if "(" in s]


def operator_name(op: str) -> str:
    """'linv(x) = ...' -> 'linv'; plain names are returned unchanged."""
    if "(" not in op:
        return op.strip()
    head = op.split("(", 1)[0].strip()
    if head.startswith("function "):
        head = head[len("function "):].strip()
    return head


def parse_spec(raw: Mapping, base_dir: str | os.PathLike | None = None) -> RunSpec:
    """Parse a spec dict.  Relative paths (`data`, `output_root`) are resolved against `base_dir` (default: the spec
    file's directory if known, else the current directory)."""
    raw = dict(raw)
    spec_path = Path(raw["_spec_path"]) if raw.get("_spec_path") else None
    base = Path(base_dir) if base_dir is not None else (spec_path.parent if spec_path else Path.cwd())

    def need(k):
        if k not in raw:
            raise SpecError(f"spec is missing `{k}`")
        return raw[k]

    def rel(p):
        q = Path(p).expanduser()
        return q if q.is_absolute() else (base / q).resolve()

    step, target = str(need("step")), str(need("target"))
    for name, v in (("step", step), ("target", target)):
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", v):
            raise SpecError(f"`{name}` must be a simple name: {v!r}")
    data = rel(need("data"))
    seeds = [int(s) for s in raw.get("seeds", [0])]
    threads = int(raw.get("threads", DEFAULT_THREADS))
    if threads < 1:
        raise SpecError("threads must be >= 1")
    budget_min = float(need("budget_minutes"))
    chunk_min = float(raw.get("chunk_minutes", DEFAULT_CHUNK_MIN))
    if budget_min <= 0 or chunk_min <= 0:
        raise SpecError("budget_minutes and chunk_minutes must be positive")
    tiers = tuple(sorted(int(t) for t in raw.get("tiers", DEFAULT_TIERS)))
    plateau = dict(raw.get("plateau", {}))
    maxsize = int(need("maxsize"))
    tiers_over = [t for t in tiers if t > maxsize]
    if tiers_over:
        warnings.warn(f"tiers {tiers_over} exceed maxsize {maxsize}; they equal the full front", stacklevel=2)

    ops_raw = dict(raw.get("operators", {}))
    operators_source = None
    if "module" in ops_raw:
        operators_source = {"module": ops_raw["module"], "attr": ops_raw.get("attr"),
                            "kwargs": dict(ops_raw.get("kwargs", {}))}
        overrides = {k: v for k, v in ops_raw.items() if k not in ("module", "attr", "kwargs")}
        operators_source["overrides"] = overrides
        operators: dict = {}
    else:
        operators = normalize_operators(ops_raw)

    loss = {k: str(v) for k, v in dict(raw.get("loss", {})).items() if v is not None}
    unknown_loss = sorted(set(loss) - set(LOSS_KEYS))
    if unknown_loss:
        raise SpecError(f"unknown loss settings {unknown_loss}; allowed: {list(LOSS_KEYS)}")
    if len(loss) > 1:
        raise SpecError(f"give at most one of {list(LOSS_KEYS)}")

    template = raw.get("template")
    if template is not None:
        template = dict(template)
        for k in ("combine", "expressions"):
            if k not in template:
                raise SpecError(f"template is missing `{k}`")
        template["expressions"] = [str(e) for e in template["expressions"]]
        if template.get("variable_names") is not None:
            template["variable_names"] = [str(v) for v in template["variable_names"]]
        if template.get("parameters"):
            template["parameters"] = {str(k): int(v) for k, v in template["parameters"].items()}
        else:
            template.pop("parameters", None)
        if "loss_function" in loss:
            raise SpecError("with a template expression use `loss_function_expression` (PySR passes only the inner "
                            "tree to `loss_function`)")

    pysr = dict(raw.get("pysr", {}))
    owned = sorted(set(pysr) & RUNNER_OWNED_KW)
    if owned:
        raise SpecError(f"[pysr] may not set runner-owned keywords {owned}")
    merged = dict(PYSR_DEFAULTS)
    merged.update(pysr)
    if merged.get("batching") and any(k in loss for k in ("loss_function", "loss_function_expression")):
        warnings.warn("batching with a full-objective loss: the loss sees a random row subset (row order lost) "
                      "unless it defines the 4-argument (tree, dataset, options, idx) method", stacklevel=2)

    guesses = raw.get("guesses")
    if guesses is not None and not isinstance(guesses, list):
        raise SpecError("guesses must be a list (single output) or a list of lists (multi-output)")

    populations = raw.get("populations")
    return RunSpec(
        step=step, target=target, data=data, seeds=seeds, threads=threads, budget_s=60.0 * budget_min,
        chunk_s=60.0 * chunk_min, min_chunk_s=float(raw.get("min_chunk_seconds", DEFAULT_MIN_CHUNK_S)),
        tiers=tiers, plateau_window=int(plateau.get("window", 3)), plateau_rel_tol=float(plateau.get("rel_tol", 0.01)),
        maxsize=maxsize, populations=(int(populations) if populations is not None else None), operators=operators,
        operators_source=operators_source, loss=loss, template=template, guesses=guesses, pysr=merged,
        output_root=rel(raw["output_root"]) if raw.get("output_root") else _default_output_root(),
        salvage_aborted=bool(raw.get("salvage_aborted", True)), log_julia_output=bool(raw.get("log_julia_output", True)),
        spec_path=spec_path, raw={k: v for k, v in raw.items() if not k.startswith("_")},
    )


def apply_overrides(raw: Mapping, *, threads: int | None = None, budget_min: float | None = None,
                    chunk_min: float | None = None, seeds: Sequence[int] | None = None) -> dict:
    """CLI overrides on a raw spec dict."""
    raw = dict(raw)
    if threads is not None:
        raw["threads"] = int(threads)
    if budget_min is not None:
        raw["budget_minutes"] = float(budget_min)
    if chunk_min is not None:
        raw["chunk_minutes"] = float(chunk_min)
    if seeds is not None:
        raw["seeds"] = [int(s) for s in seeds]
    return raw


# --------------------------------------------------------------------------------------------------------------------
# target data
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Target:
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray | None
    variable_names: list[str]
    sha256: str
    arrays: dict            # every array in the file (for {{npz:key}} placeholders)

    @property
    def nout(self) -> int:
        return 1 if self.y.ndim == 1 else int(self.y.shape[1])


def load_target(path: Path) -> Target:
    """Read a target `.npz` with X (N, d), y (N,) or (N, n_out), optional weights (same shape as y) and
    variable_names (d strings)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for k in ("X", "y"):
        if k not in arrays:
            raise SpecError(f"{path}: missing array `{k}`")
    X = np.asarray(arrays["X"], dtype=np.float64)
    y = np.asarray(arrays["y"], dtype=np.float64)
    if X.ndim != 2 or y.ndim not in (1, 2) or len(y) != len(X):
        raise SpecError(f"{path}: X must be (N, d) and y (N,) or (N, n_out); got {X.shape}, {y.shape}")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    w = arrays.get("weights")
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != y.shape:
            if w.ndim == 2 and w.shape[1] == 1 and y.ndim == 1:
                w = w[:, 0]
            else:
                raise SpecError(f"{path}: weights {w.shape} must match y {y.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise SpecError(f"{path}: weights must be finite and non-negative")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise SpecError(f"{path}: X and y must be finite")
    if "variable_names" in arrays:
        names = [str(s) for s in np.asarray(arrays["variable_names"]).tolist()]
    else:
        names = [f"x{i}" for i in range(X.shape[1])]
    if len(names) != X.shape[1]:
        raise SpecError(f"{path}: {len(names)} variable_names for {X.shape[1]} columns")
    return Target(X=X, y=y, weights=w, variable_names=names, sha256=sha256_file(path), arrays=arrays)


def fill_npz_placeholders(src: str, arrays: Mapping[str, np.ndarray]) -> str:
    """Replace ``{{npz:key}}`` in a Julia loss string with a Julia tuple literal of that array (e.g. slice ends)."""
    def lit(m):
        k = m.group(1)
        if k not in arrays:
            raise SpecError(f"loss placeholder {{{{npz:{k}}}}}: array `{k}` not in the target file")
        a = np.asarray(arrays[k]).ravel()
        if np.issubdtype(a.dtype, np.integer):
            items = [str(int(v)) for v in a]
        else:
            items = [repr(float(v)) for v in a]
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    return _NPZ_PLACEHOLDER.sub(lit, src)


# --------------------------------------------------------------------------------------------------------------------
# Julia start-up (threads before juliacall starts; never after torch)
# --------------------------------------------------------------------------------------------------------------------
def configure_julia_env(threads: int) -> bool:
    """Put the Julia thread count and signal handling into the environment.  Must run before juliacall starts
    (i.e. before ``import pysr``).  Returns False if Julia is already running (the setting then has no effect).

    juliacall 0.9.36 reads PYTHON_JULIACALL_THREADS (-> julia --threads) and PYTHON_JULIACALL_HANDLE_SIGNALS
    (must be "yes" for multithreaded searches and for PySR's cooperative Ctrl-C).  PySR 2.7.3 then sets
    JULIA_NUM_GC_THREADS to "max(1, threads // 2),1" unless it is already set."""
    if "juliacall" in sys.modules:
        return False
    os.environ["PYTHON_JULIACALL_THREADS"] = str(int(threads))
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no")
    return True


def _pysr():
    """Import PySR (starting Julia on first use).  Refuses to start Julia after torch has been imported."""
    if "pysr" not in sys.modules:
        if "torch" in sys.modules:
            raise RuntimeError("torch is imported in this process; refusing to start Julia (juliacall after torch "
                               "can segfault). Run the search in a process that never imports torch.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*PYTHON_JULIACALL_THREADS.*")
            __import__("pysr")
    return sys.modules["pysr"]


def julia_thread_info() -> dict:
    jl = _pysr().jl
    return {"nthreads": int(jl.seval("Threads.nthreads()")),
            "nthreads_interactive": int(jl.seval("Threads.nthreads(:interactive)")),
            "ngcthreads": int(jl.seval("Threads.ngcthreads()")),
            "julia_version": str(jl.seval("string(VERSION)")),
            "symbolic_regression": str(jl.seval("string(pkgversion(SymbolicRegression))"))}


def _sympy_callables(mappings: Mapping[str, str]) -> dict:
    """{name: 'sympy expression in x[, y]'} -> {name: callable} for PySR's extra_sympy_mappings.  sympy is a hard
    dependency of PySR and already imported by ``import pysr`` (it never imports torch)."""
    _pysr()
    sympy = sys.modules["sympy"]
    out = {}
    for name, expr_src in mappings.items():
        xs, ys = sympy.Symbol("x"), sympy.Symbol("y")
        expr = sympy.sympify(expr_src, locals={"x": xs, "y": ys})
        if ys in expr.free_symbols:
            out[name] = (lambda e: (lambda a, b: e.subs({xs: a, ys: b}, simultaneous=True)))(expr)
        else:
            out[name] = (lambda e: (lambda a: e.subs(xs, a)))(expr)
    return out


_JULIA_HELPERS = r"""
function sfm_template_parts(hof, options, cs)
    res = Vector{Vector{String}}()
    for c in cs
        parts = String[]
        if 1 <= c <= length(hof.members) && hof.exists[c]
            ex = hof.members[c].tree
            contents = SymbolicRegression.DynamicExpressions.get_contents(ex)
            for (k, v) in pairs(contents)
                push!(parts, String(k))
                push!(parts, SymbolicRegression.string_tree(v, options; pretty=false))
            end
            md = SymbolicRegression.DynamicExpressions.get_metadata(ex)
            if hasproperty(md, :_data) && hasproperty(md._data, :parameters)
                for (k, v) in pairs(md._data.parameters)
                    push!(parts, "param:" * String(k))
                    push!(parts, join(string.(collect(v)), ","))
                end
            end
        end
        push!(res, parts)
    end
    return res
end
"""


def _ensure_julia_helpers() -> None:
    jl = _pysr().jl
    if not bool(jl.seval("isdefined(Main, :sfm_template_parts)")):
        jl.seval(_JULIA_HELPERS)


# --------------------------------------------------------------------------------------------------------------------
# locking and fd redirection
# --------------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def seed_lock(seed_dir: Path):
    seed_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(seed_dir / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RunnerBusy(f"another runner holds {seed_dir / LOCK_FILE}") from e
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def lock_is_held(seed_dir: Path) -> bool:
    p = Path(seed_dir) / LOCK_FILE
    if not p.exists():
        return False
    fd = os.open(p, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


@contextlib.contextmanager
def redirect_fds_to(log_path: Path | None):
    """Send fd 1 and 2 (Julia writes there directly) to `log_path` for the duration; no-op if None."""
    if log_path is None:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            if "pysr" in sys.modules:
                sys.modules["pysr"].jl.seval("flush(stdout); flush(stderr)")
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def parse_eval_rates(text: str) -> list[float]:
    """'Expressions evaluated per second: 1.23e+05' values printed by SymbolicRegression.jl (verbosity > 0)."""
    return [float(m.group(1)) for m in _EVALS_RE.finditer(text)]


def _sigterm_handler(signum, frame):  # noqa: ARG001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def install_sigterm_handler() -> None:
    """SIGTERM -> stop after the current chunk (SIGINT stops the running chunk early, via PySR)."""
    if hasattr(signal, "SIGTERM"):
        with contextlib.suppress(ValueError):  # not the main thread
            signal.signal(signal.SIGTERM, _sigterm_handler)


# --------------------------------------------------------------------------------------------------------------------
# the per-seed runner
# --------------------------------------------------------------------------------------------------------------------
def new_progress(spec: RunSpec, seed: int, nout: int) -> dict:
    return {"schema": SCHEMA_PROGRESS, "step": spec.step, "target": spec.target, "seed": int(seed),
            "status": "new", "created_at": utc_now(), "updated_at": utc_now(), "budget_s": spec.budget_s,
            "budget_history": [], "chunk_s": spec.chunk_s, "elapsed_search_s": 0.0, "chunks_completed": 0,
            "tiers": list(spec.tiers), "nout": nout, "best_by_tier": None, "history": [], "plateau": None,
            "val_nll_check": None, "in_flight": None, "aborted_chunks": [], "restores": [], "fallback_used": False,
            "errors": [], "threads_history": [], "julia": None, "done": False}


class SeedRun:
    """One seed of one spec: chunked search, resume, extension, exports."""

    def __init__(self, spec: RunSpec, seed: int, *, log: Callable[[str], None] = print,
                 allow_spec_change: bool = False):
        if not spec.operators:
            raise SpecError("operator vocabulary not resolved (operators.module given?): use RunSpec.with_operators "
                            "or scripts/sr_run.py")
        self.spec = spec
        self.seed = int(seed)
        self.log = log
        self.allow_spec_change = allow_spec_change
        self.seed_dir = spec.seed_dir(seed)
        self.run_dir = self.seed_dir / RUN_ID
        self.snap_dir = self.seed_dir / SNAPSHOT_DIR
        self.target: Target | None = None
        self.model = None
        self.populations: int | None = None
        self.progress: dict = {}

    # -- paths / small helpers ----------------------------------------------------------------------------------
    @property
    def progress_path(self) -> Path:
        return self.seed_dir / PROGRESS_FILE

    def _write_progress(self) -> None:
        self.progress["updated_at"] = utc_now()
        atomic_write_json(self.progress_path, self.progress)

    def _say(self, msg: str) -> None:
        self.log(f"[{self.spec.step}/{self.spec.target}/seed{self.seed}] {msg}")

    # -- spec snapshot / identity ---------------------------------------------------------------------------------
    def _check_or_write_snapshot(self) -> None:
        snap_path = self.seed_dir / SPEC_SNAPSHOT
        snap = read_json(snap_path)
        if snap is None:
            self.populations = self.spec.resolved_populations()
            ident = self.spec.identity(self.populations, self.target.sha256)
            atomic_write_json(snap_path, {"written_at": utc_now(), "identity": ident,
                                          "identity_sha256": hashlib.sha256(_canon(ident).encode()).hexdigest(),
                                          "spec": self.spec.raw, "spec_path": self.spec.spec_path,
                                          "threads_at_start": self.spec.threads})
            return
        self.populations = int(snap["identity"]["populations"])
        ident = self.spec.identity(self.populations, self.target.sha256)
        old = snap["identity"]
        if _canon(ident) != _canon(old):
            diff = sorted(k for k in set(ident) | set(old) if _canon(ident.get(k)) != _canon(old.get(k)))
            msg = f"spec differs from the snapshot in {diff} (only budget/threads/chunk/tiers may change on resume)"
            if not self.allow_spec_change:
                raise SpecError(msg + "; pass allow_spec_change=True (--allow-spec-change) to override")
            self._say("WARNING: " + msg)
            append_event(self.seed_dir, "spec_change_allowed", fields=diff)

    # -- PySR model construction ---------------------------------------------------------------------------------
    def _loss_kwargs(self) -> dict:
        return {k: fill_npz_placeholders(v, self.target.arrays) for k, v in self.spec.loss.items()}

    def _operator_kwargs(self) -> dict:
        ops = dict(self.spec.operators)
        kw: dict[str, Any] = {}
        for k in ("binary_operators", "unary_operators", "operators", "constraints", "nested_constraints",
                  "complexity_of_operators"):
            if k in ops:
                kw[k] = copy.deepcopy(ops[k])
        if self.spec.template is None:
            maps = dict(ops.get("extra_sympy_mappings") or {})
            if ops.get("sympy_mappings"):
                maps.update(_sympy_callables(ops["sympy_mappings"]))
            missing = [operator_name(d) for d in custom_operator_definitions(ops) if operator_name(d) not in maps]
            if missing:
                raise SpecError(f"custom operators {missing} need sympy mappings for free-form searches "
                                f"(operators.sympy_mappings = {{name = 'expression in x, y'}})")
            if maps:
                kw["extra_sympy_mappings"] = maps
        return kw

    def _expression_spec(self):
        if self.spec.template is None:
            return None
        t = self.spec.template
        names = t.get("variable_names") or self.target.variable_names
        if len(names) != self.target.X.shape[1]:
            raise SpecError(f"template variable_names {names} do not match the {self.target.X.shape[1]} data columns")
        return _pysr().TemplateExpressionSpec(combine=t["combine"], expressions=list(t["expressions"]),
                                              variable_names=list(names), parameters=t.get("parameters"))

    def _runtime_kwargs(self) -> dict:
        """Keyword arguments that are not pickled with the model (callables) or must be (re)asserted."""
        kw = {"warm_start": True}
        if self.spec.template is None:
            ok = self._operator_kwargs()
            if "extra_sympy_mappings" in ok:
                kw["extra_sympy_mappings"] = ok["extra_sympy_mappings"]
        for k in SOFT_PYSR_KEYS:
            if k in self.spec.pysr:
                kw[k] = self.spec.pysr[k]
        return kw

    def _new_model(self, guesses: list | None):
        pysr = _pysr()
        kw = dict(self.spec.pysr)
        kw.update(self._operator_kwargs())
        kw.update(self._loss_kwargs())
        es = self._expression_spec()
        if es is not None:
            kw["expression_spec"] = es
        kw.update(maxsize=self.spec.maxsize, populations=self.populations, warm_start=True,
                  output_directory=str(self.seed_dir), run_id=RUN_ID, temp_equation_file=False,
                  niterations=NITERATIONS_PER_CHUNK, timeout_in_seconds=self.spec.chunk_s, input_stream="devnull",
                  guesses=guesses, random_state=self._chunk_random_state(0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pysr.PySRRegressor(**kw)

    def _define_custom_julia_objects(self) -> None:
        """Define custom operators and loss functions in Julia's Main before a checkpoint is deserialised (template
        checkpoints rebuild expressions from the Julia state, which references these functions by name)."""
        jl = _pysr().jl
        for d in custom_operator_definitions(self.spec.operators):
            jl.seval(d)
        for src in self._loss_kwargs().values():
            jl.seval(src)
        es = self._expression_spec()
        if es is not None:
            es.julia_expression_spec()

    def _chunk_random_state(self, chunk_index: int) -> int:
        return (1_000_003 * (self.seed + 1) + 7919 * int(chunk_index)) % (2**31 - 1)

    # -- hall-of-fame snapshots and export -------------------------------------------------------------------------
    def _snapshot_hofs(self, label: str) -> list[str]:
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_stamp()
        out = []
        for name in hof_csv_names(self.target.nout):
            src = self.run_dir / name
            if not src.exists():
                src = self.run_dir / (name + ".bak")
            if not src.exists():
                continue
            dst = self.snap_dir / f"{label}_{stamp}_{name}"
            shutil.copy2(src, dst)
            out.append(str(dst.relative_to(self.seed_dir)))
        return out

    def _template_parts(self, rows_per_output: list[list[dict]]) -> list[list[dict | None]] | None:
        if self.spec.template is None or self.model is None:
            return None
        try:
            _ensure_julia_helpers()
            jl = _pysr().jl
            state = self.model.julia_state_
            options = self.model.julia_options_
            hofs = state[1]
            res = []
            for o, rows in enumerate(rows_per_output):
                hof = hofs[o] if self.target.nout > 1 else hofs
                cs = jl.seval("Vector{Int}")([int(r["complexity"]) for r in rows])
                parts = jl.sfm_template_parts(hof, options, cs)
                per = []
                for p in parts:
                    lst = [str(s) for s in p]
                    if not lst:
                        per.append(None)
                        continue
                    sub, params = {}, {}
                    for k, v in zip(lst[0::2], lst[1::2]):
                        if k.startswith("param:"):
                            params[k[6:]] = [float(x) for x in v.split(",") if x]
                        else:
                            sub[k] = v
                    per.append({"subexpressions": sub, **({"parameters": params} if params else {})})
                res.append(per)
            return res
        except Exception as e:  # noqa: BLE001  export must not kill the search
            self._say(f"WARNING: template sub-expression export failed: {e!r}")
            return None

    def _export_hof(self, rows_per_output: list[list[dict]], done: bool) -> None:
        parts = self._template_parts(rows_per_output)
        outputs = []
        for o, rows in enumerate(rows_per_output):
            eqs = []
            for i, r in enumerate(rows):
                e = {"complexity": r["complexity"], "loss": r["loss"], "loss_str": r["loss_str"],
                     "equation": r["equation"]}
                if parts is not None and parts[o][i] is not None:
                    e.update(parts[o][i])
                eqs.append(e)
            outputs.append({"output": o, "csv": f"{RUN_ID}/{hof_csv_names(self.target.nout)[o]}", "equations": eqs})
        doc = {"schema": SCHEMA_HOF, "step": self.spec.step, "target": self.spec.target, "seed": self.seed,
               "written_at": utc_now(), "done": done, "chunks_completed": self.progress["chunks_completed"],
               "elapsed_search_s": self.progress["elapsed_search_s"], "variable_names": self.target.variable_names,
               "nout": self.target.nout, "maxsize": self.spec.maxsize, "template": self.spec.template,
               "print_precision": self.spec.pysr.get("print_precision"), "precision": self.spec.pysr.get("precision"),
               "loss": self.spec.loss or "PySR default (weighted) L2", "outputs": outputs}
        atomic_write_json(self.seed_dir / HOF_JSON, doc)

    def _guesses_from_export(self) -> list | None:
        """Hall of fame as `guesses` (for the fallback restart): equation strings, or sub-expression dicts for
        template expressions; list[list] for multi-output searches."""
        doc = read_json(self.seed_dir / HOF_JSON)
        per_output: list[list] = []
        if doc is not None:
            for out in doc["outputs"]:
                g = []
                for e in out["equations"]:
                    if self.spec.template is not None:
                        if e.get("subexpressions"):
                            g.append(dict(e["subexpressions"]))
                    else:
                        g.append(e["equation"])
                per_output.append(g)
        else:
            rows = read_run_hofs(self.run_dir, self.target.nout) or self._latest_snapshot_rows()
            if rows is None or self.spec.template is not None:
                return None
            per_output = [[r["equation"] for r in rs] for rs in rows]
        if not any(per_output):
            return None
        return per_output[0] if self.target.nout == 1 else per_output

    def _latest_snapshot_rows(self) -> list[list[dict]] | None:
        if not self.snap_dir.exists():
            return None
        names = hof_csv_names(self.target.nout)
        out = []
        for name in names:
            cands = sorted(p for p in self.snap_dir.glob(f"chunk*_{name}"))
            if not cands:
                return None
            out.append(read_hof_csv(cands[-1]))
        return out

    def _salvage_guesses(self, rows_per_output: list[list[dict]] | None) -> list | None:
        """Equations of a killed chunk's in-flight hall of fame, as guesses for the first resumed chunk."""
        if rows_per_output is None:
            return None
        per_output = []
        for rows in rows_per_output:
            g = []
            for r in rows:
                if self.spec.template is None:
                    g.append(r["equation"])
                else:
                    d = parse_template_equation(r["equation"], self.spec.template["expressions"])
                    if d is not None:
                        g.append(d)
            per_output.append(g)
        if not any(per_output):
            return None
        return per_output[0] if self.target.nout == 1 else per_output

    # -- restore -------------------------------------------------------------------------------------------------
    def _restore_from_checkpoint(self):
        pysr = _pysr()
        self._define_custom_julia_objects()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = pysr.PySRRegressor.from_file(run_directory=str(self.run_dir), **self._runtime_kwargs())
        if getattr(model, "julia_state_stream_", None) is None:
            raise RuntimeError("checkpoint has no Julia search state")
        if int(getattr(model, "nout_", 1)) != self.target.nout:
            raise RuntimeError(f"checkpoint nout {model.nout_} != data nout {self.target.nout}")
        if int(getattr(model, "n_features_in_", -1)) != self.target.X.shape[1]:
            raise RuntimeError("checkpoint feature count does not match the data")
        return model

    def _fallback_model(self, reason: str):
        ck = self.run_dir / "checkpoint.pkl"
        moved = None
        if ck.exists():
            moved = ck.with_name(f"checkpoint.pkl.broken-{utc_stamp()}")
            os.replace(ck, moved)
        guesses = self._guesses_from_export()
        rec = {"at": utc_now(), "method": "fallback_guesses" if guesses else "fresh_no_guesses",
               "reason": reason, "moved_checkpoint": str(moved) if moved else None,
               "n_guesses": (sum(len(g) for g in guesses) if guesses and isinstance(guesses[0], list)
                             else (len(guesses) if guesses else 0)),
               "after_chunk": self.progress["chunks_completed"] - 1}
        self.progress["restores"].append(rec)
        self.progress["fallback_used"] = True
        append_event(self.seed_dir, "fallback", **rec)
        self._say(f"FALLBACK restart seeded with the saved hall of fame ({rec['n_guesses']} guesses): {reason}")
        return self._new_model(guesses)

    # -- main loop -------------------------------------------------------------------------------------------------
    def run(self) -> dict:
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        with seed_lock(self.seed_dir):
            self.target = load_target(self.spec.data)
            self._check_or_write_snapshot()
            prog = read_json(self.progress_path)
            self.progress = prog if prog is not None else new_progress(self.spec, self.seed, self.target.nout)
            p = self.progress
            if p["nout"] != self.target.nout:
                raise SpecError(f"progress nout {p['nout']} != data nout {self.target.nout}")
            p["tiers"] = list(self.spec.tiers)
            p["chunk_s"] = self.spec.chunk_s
            if p["budget_s"] != self.spec.budget_s or not p["budget_history"]:
                p["budget_history"].append({"at": utc_now(), "budget_s": self.spec.budget_s})
            p["budget_s"] = self.spec.budget_s
            p["threads_history"].append({"at": utc_now(), "threads": self.spec.threads})

            done_path = self.seed_dir / DONE_FILE
            remaining = p["budget_s"] - p["elapsed_search_s"]
            if done_path.exists():
                if remaining < self.spec.min_chunk_s:
                    self._say(f"already done ({p['elapsed_search_s']:.0f} s of {p['budget_s']:.0f} s)")
                    self._write_progress()
                    return p
                os.replace(done_path, self.seed_dir / f"DONE.superseded-{utc_stamp()}.json")
                p["done"] = False
                append_event(self.seed_dir, "extended", budget_s=p["budget_s"], elapsed_s=p["elapsed_search_s"])
                self._say(f"budget extended to {p['budget_s']:.0f} s; continuing the same seed")

            # a chunk that was in flight when the process died
            salvage_rows = None
            if p.get("in_flight"):
                inf = p["in_flight"]
                rows = read_run_hofs(self.run_dir, self.target.nout)
                started = _dt.datetime.fromisoformat(inf["started_at"].replace("Z", "+00:00")).timestamp()
                mt = max((f.stat().st_mtime for f in self.run_dir.glob("hall_of_fame*.csv")), default=started)
                snap = self._snapshot_hofs(f"aborted{int(inf['chunk']):03d}") if rows is not None else []
                rec = {**inf, "detected_at": utc_now(), "lost_s_estimate": max(0.0, mt - started),
                       "orphan_snapshots": snap,
                       "orphan_tier_best": [tier_bests(r, self.spec.tiers) for r in rows] if rows else None}
                p["aborted_chunks"].append(rec)
                p["in_flight"] = None
                append_event(self.seed_dir, "aborted_chunk_detected", **rec)
                self._say(f"chunk {inf['chunk']} was aborted in flight (~{rec['lost_s_estimate']:.0f} s lost); "
                          f"resuming from the end of chunk {p['chunks_completed'] - 1}")
                if self.spec.salvage_aborted:
                    salvage_rows = rows
            self._write_progress()

            # Julia
            _pysr()
            p["julia"] = julia_thread_info()
            if p["julia"]["nthreads"] != self.spec.threads:
                self._say(f"WARNING: Julia runs {p['julia']['nthreads']} threads, spec asks for {self.spec.threads} "
                          f"(set the thread count before Julia starts)")

            # model: restore, fallback, or new
            pending_guesses = None
            restored_ok = None
            if p["chunks_completed"] > 0:
                try:
                    self.model = self._restore_from_checkpoint()
                    restored_ok = False   # confirmed after the first warm-start fit
                    rec = {"at": utc_now(), "method": "from_file", "after_chunk": p["chunks_completed"] - 1}
                    p["restores"].append(rec)
                    append_event(self.seed_dir, "restored", **rec)
                    self._say(f"restored checkpoint after chunk {p['chunks_completed'] - 1} (warm start)")
                    pending_guesses = self._salvage_guesses(salvage_rows)
                    if pending_guesses:
                        rec["salvaged_guesses"] = (sum(map(len, pending_guesses)) if self.target.nout > 1
                                                   else len(pending_guesses))
                except Exception as e:  # noqa: BLE001
                    self.model = self._fallback_model(f"restore failed: {type(e).__name__}: {e}")
            else:
                if (self.run_dir / "checkpoint.pkl").exists():
                    stale = self.run_dir.with_name(f"{RUN_ID}.stale-{utc_stamp()}")
                    os.replace(self.run_dir, stale)
                    append_event(self.seed_dir, "stale_run_dir_moved", to=str(stale))
                pending_guesses = merge_guesses(self.spec.guesses, self._salvage_guesses(salvage_rows),
                                                self.target.nout)
                self.model = self._new_model(pending_guesses)
            self._write_progress()
            append_event(self.seed_dir, "start", budget_s=p["budget_s"], elapsed_s=p["elapsed_search_s"],
                         threads=self.spec.threads, populations=self.populations, julia=p["julia"])

            p["status"] = "running"
            stop_reason = None
            try:
                while True:
                    remaining = p["budget_s"] - p["elapsed_search_s"]
                    if remaining < self.spec.min_chunk_s:
                        stop_reason = "budget"
                        break
                    if _STOP_REQUESTED:
                        stop_reason = "sigterm"
                        break
                    if (self.seed_dir / STOP_FILE).exists():
                        os.replace(self.seed_dir / STOP_FILE, self.seed_dir / f"STOP.consumed-{utc_stamp()}")
                        stop_reason = "stop_file"
                        break
                    chunk_s = min(self.spec.chunk_s, remaining)
                    if remaining - chunk_s < self.spec.min_chunk_s:
                        chunk_s = remaining          # absorb a too-short tail into this chunk
                    idx = p["chunks_completed"]
                    try:
                        rec = self._run_chunk(idx, chunk_s, pending_guesses)
                    except Exception as e:  # noqa: BLE001
                        if restored_ok is False:
                            # the first warm start after a restore failed: state unusable -> fallback, once
                            restored_ok = None
                            self.model = self._fallback_model(f"warm start after restore failed: "
                                                              f"{type(e).__name__}: {e}")
                            pending_guesses = self.model.guesses
                            self._write_progress()
                            continue
                        raise
                    restored_ok = True if restored_ok is False else restored_ok
                    pending_guesses = None
                    if rec.get("interrupted"):
                        stop_reason = "sigint"
                        break
            except KeyboardInterrupt:
                # keep `in_flight`: the next start treats the chunk as aborted (its checkpoint may or may not have
                # been written), which is the conservative reading
                stop_reason = "keyboard_interrupt"
                raise
            except BaseException as e:
                p["status"] = "error"
                p["errors"].append({"at": utc_now(), "error": f"{type(e).__name__}: {e}"})
                p["in_flight"] = None
                self._write_progress()
                append_event(self.seed_dir, "error", error=f"{type(e).__name__}: {e}")
                raise
            finally:
                if stop_reason is not None:
                    self._finish(stop_reason)
            return p

    def _finish(self, reason: str) -> None:
        p = self.progress
        if reason == "budget":
            p["status"] = "done"
            p["done"] = True
            rows = read_run_hofs(self.run_dir, self.target.nout)
            if rows is not None and self.model is not None:
                self._export_hof(rows, done=True)
            hof_sha = sha256_file(self.seed_dir / HOF_JSON) if (self.seed_dir / HOF_JSON).exists() else None
            atomic_write_json(self.seed_dir / DONE_FILE, {
                "done_at": utc_now(), "elapsed_search_s": p["elapsed_search_s"], "budget_s": p["budget_s"],
                "chunks_completed": p["chunks_completed"], "hall_of_fame_json_sha256": hof_sha,
                "plateau": p.get("plateau"), "fallback_used": p["fallback_used"],
                "aborted_chunks": len(p["aborted_chunks"])})
            self._say(f"DONE: {p['chunks_completed']} chunks, {p['elapsed_search_s']:.0f} s of {p['budget_s']:.0f} s")
        else:
            p["status"] = "stopped"
            self._say(f"stopped ({reason}) after chunk {p['chunks_completed'] - 1}; resumable")
        p["stop_reason"] = reason
        self._write_progress()
        append_event(self.seed_dir, "finish", reason=reason, elapsed_s=p["elapsed_search_s"],
                     chunks=p["chunks_completed"])

    def _run_chunk(self, idx: int, chunk_s: float, guesses: list | None) -> dict:
        p = self.progress
        t = self.target
        self.model.set_params(timeout_in_seconds=float(chunk_s), niterations=NITERATIONS_PER_CHUNK, warm_start=True,
                              random_state=self._chunk_random_state(idx), guesses=guesses)
        p["in_flight"] = {"chunk": idx, "started_at": utc_now(), "timeout_s": chunk_s,
                          "with_guesses": guesses is not None}
        self._write_progress()
        self._say(f"chunk {idx}: {chunk_s:.0f} s (elapsed {p['elapsed_search_s']:.0f} / {p['budget_s']:.0f} s)")
        log_path = self.seed_dir / LOG_FILE if self.spec.log_julia_output else None
        off = log_path.stat().st_size if log_path is not None and log_path.exists() else 0
        # the previous *completed* chunk's snapshot (the run directory may hold a killed chunk's newer front)
        prev_rows = self._latest_snapshot_rows() if idx > 0 else None
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== {utc_now()} chunk {idx} timeout {chunk_s:.1f} s =====\n")
        t0 = time.monotonic()
        with redirect_fds_to(log_path), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*random_state.*")
            warnings.filterwarnings("ignore", message=".*maxsize.*")
            self.model.fit(t.X, t.y, weights=t.weights, variable_names=list(t.variable_names))
        dt = time.monotonic() - t0
        interrupted = bool(getattr(self.model, "interrupted_", False))
        rates = []
        if log_path is not None and log_path.exists():
            with open(log_path, encoding="utf-8", errors="replace") as f:
                f.seek(off)
                rates = parse_eval_rates(f.read())
        rows = read_run_hofs(self.run_dir, t.nout)
        if rows is None:
            raise RuntimeError("no hall_of_fame CSV after the chunk")
        snaps = self._snapshot_hofs(f"chunk{idx:03d}")
        regress = None
        if prev_rows is not None:
            regress = [hof_regressions(a, b) for a, b in zip(prev_rows, rows)]
        rec = {"index": idx, "started_at": p["in_flight"]["started_at"], "finished_at": utc_now(),
               "timeout_s": chunk_s, "fit_wall_s": dt, "elapsed_after_s": p["elapsed_search_s"] + dt,
               "interrupted": interrupted, "with_guesses": guesses is not None,
               "tier_best": [tier_bests(r, self.spec.tiers) for r in rows],
               "n_front": [len(r) for r in rows], "snapshots": snaps,
               "hof_not_worse_than_prev_chunk": (None if regress is None else all(len(x) == 0 for x in regress)),
               "hof_regressions": regress if regress and any(regress) else None,
               "evals_per_s_last": rates[-1] if rates else None,
               "evals_per_s_mean": (float(np.mean(rates)) if rates else None), "threads": self.spec.threads}
        p["history"].append(rec)
        p["elapsed_search_s"] += dt
        p["chunks_completed"] = idx + 1
        p["last_chunk_index"] = idx
        p["best_by_tier"] = rec["tier_best"]
        p["plateau"] = plateau_check(p["history"], self.spec.tiers, self.spec.plateau_window,
                                     self.spec.plateau_rel_tol)
        p["in_flight"] = None
        self._export_hof(rows, done=False)
        self._write_progress()
        append_event(self.seed_dir, "chunk", index=idx, fit_wall_s=dt, tier_best=rec["tier_best"],
                     interrupted=interrupted, evals_per_s=rec["evals_per_s_last"])
        tb = "; ".join(" ".join(f"<={k}:{v:.4g}" if v is not None else f"<={k}:-" for k, v in d.items())
                       for d in rec["tier_best"])
        self._say(f"chunk {idx} done in {dt:.0f} s | {tb} | plateau={p['plateau'].get('plateau')}"
                  + (f" | {rec['evals_per_s_last']:.3g} evals/s" if rates else ""))
        return rec


def merge_guesses(a: list | None, b: list | None, nout: int) -> list | None:
    """Concatenate two `guesses` lists (flat for one output, list of lists for several)."""
    if not a:
        return b or None
    if not b:
        return a or None
    if nout == 1:
        return list(a) + list(b)
    return [list(x) + list(y) for x, y in zip(a, b)]


def parse_template_equation(eq: str, expressions: Sequence[str]) -> dict | None:
    """Split a template equation string as PySR prints it ('u = ...; v = ...') into {name: sub-expression}; None if
    the format does not match."""
    parts = {}
    for chunk in eq.split(";"):
        if "=" not in chunk:
            return None
        k, v = chunk.split("=", 1)
        parts[k.strip()] = v.strip()
    if sorted(parts) != sorted(expressions):
        return None
    return parts


# --------------------------------------------------------------------------------------------------------------------
# status (no Julia) and the validation-NLL hook
# --------------------------------------------------------------------------------------------------------------------
def record_val_nll_check(seed_dir: str | os.PathLike, front_improves_by_more_than_1se: bool,
                         detail: Mapping | None = None) -> None:
    """Hook for the second part of the plateau rule (computed by the refit, not here): record whether any new front
    member improves the refitted validation NLL by more than 1 SE.  Written to its own file so the refit process
    never races the runner on progress.json."""
    atomic_write_json(Path(seed_dir) / VAL_NLL_FILE, {"at": utc_now(),
                                                      "front_improves_by_more_than_1se": bool(
                                                          front_improves_by_more_than_1se),
                                                      "detail": dict(detail or {})})


def plateau_status(seed_dir: str | os.PathLike) -> dict:
    """Combined plateau verdict: loss rule from progress.json, validation-NLL rule from record_val_nll_check.
    ``plateau`` is True/False when both parts are known, None otherwise (``loss_rule`` alone is still reported)."""
    seed_dir = Path(seed_dir)
    prog = read_json(seed_dir / PROGRESS_FILE) or {}
    loss_rule = (prog.get("plateau") or {}).get("plateau")
    val = read_json(seed_dir / VAL_NLL_FILE)
    val_ok = None if val is None else (not val["front_improves_by_more_than_1se"])
    if loss_rule is None or val_ok is None:
        combined = None
    else:
        combined = bool(loss_rule and val_ok)
    return {"loss_rule": loss_rule, "val_nll_rule": val_ok, "plateau": combined}


def seed_status(spec: RunSpec, seed: int) -> dict:
    d = spec.seed_dir(seed)
    prog = read_json(d / PROGRESS_FILE)
    st = {"seed": int(seed), "dir": str(d), "exists": d.exists(), "running": lock_is_held(d) if d.exists() else False,
          "done": (d / DONE_FILE).exists()}
    if prog is None:
        st["status"] = "not started"
        return st
    st.update(status=prog.get("status"), chunks=prog.get("chunks_completed"), elapsed_s=prog.get("elapsed_search_s"),
              budget_s=spec.budget_s, recorded_budget_s=prog.get("budget_s"), best_by_tier=prog.get("best_by_tier"),
              plateau=plateau_status(d), plateau_detail=prog.get("plateau"), fallback_used=prog.get("fallback_used"),
              aborted=len(prog.get("aborted_chunks", [])), in_flight=prog.get("in_flight"),
              updated_at=prog.get("updated_at"))
    hist = prog.get("history") or []
    if hist:
        st["last_evals_per_s"] = hist[-1].get("evals_per_s_last")
    return st


def format_status(spec: RunSpec, seeds: Sequence[int]) -> str:
    lines = [f"{spec.step}/{spec.target}  (budget {spec.budget_s / 60:.1f} min per seed, chunk "
             f"{spec.chunk_s / 60:.1f} min, tiers {list(spec.tiers)})"]
    not_plateaued = 0
    known = 0
    for s in seeds:
        st = seed_status(spec, s)
        if st.get("status") == "not started":
            lines.append(f"  seed{s}: not started")
            continue
        pl = st["plateau"]
        lr = pl["loss_rule"]
        if lr is not None:
            known += 1
            not_plateaued += (lr is False) if pl["plateau"] is None else (pl["plateau"] is False)
        tb = st.get("best_by_tier") or []
        tbs = " | ".join(" ".join(f"<={k}: {v:.4g}" if v is not None else f"<={k}: -" for k, v in d.items())
                         for d in tb)
        flag = {True: "yes", False: "no", None: "n/a"}
        state = "RUNNING" if st["running"] else st["status"]
        if st["in_flight"] and not st["running"]:
            state += " (chunk in flight was killed)"
        lines.append(
            f"  seed{s}: {state}{' DONE' if st['done'] else ''} | chunks {st['chunks']} | "
            f"{st['elapsed_s']:.0f}/{spec.budget_s:.0f} s | {tbs} | plateau(loss) {flag[lr]}, "
            f"val-NLL {flag[pl['val_nll_rule']]}, combined {flag[pl['plateau']]} | "
            f"fallback {'yes' if st['fallback_used'] else 'no'} | aborted {st['aborted']}"
            + (f" | {st['last_evals_per_s']:.3g} evals/s" if st.get("last_evals_per_s") else ""))
    if known:
        lines.append(f"  extension rule (PLAN §7): {not_plateaued} of {len(seeds)} seeds not plateaued -> "
                     f"{'extend' if not_plateaued >= 2 else 'no extension'} (needs >= 2)")
    return "\n".join(lines)
