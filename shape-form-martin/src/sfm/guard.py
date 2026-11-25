"""sfm.guard - a file-access guard for the H200 (user constraint of 25 Nov 2026).

The user's folder on the H200 (the parent of the clone, ``<area>``) also holds files of another project that the
campaign must never open, among them Parquet hit tables ``<particle>_hits_with_model_outputs.parquet``. The plan's
guarantee is: no campaign process opens a path in ``<area>`` outside the clone, its output root and its cache / temp
directory. This module enforces it inside every Python process of the campaign.

Activation
    Only when the environment variable ``SFM_GUARD_AREA`` names the guarded area (the launcher sets it on the H200).
    Unset (the laptop): the guard is inert and nothing is installed. When set, the guard is installed at the import
    of ``sfm.paths`` (the first module every campaign entry point imports, before ``common`` / pandas) and again,
    idempotently, by ``sfm.machine.startup``.

Rules (a Python audit hook, ``sys.addaudithook``)
    1. Area: a path that resolves (``os.path.realpath``: symbolic links followed) under ``SFM_GUARD_AREA`` but not
       under the clone (``paths.REPO_ROOT``), the output root (``paths.OUT_ROOT``) or the launcher's cache / temp
       directory (``SFM_USER_CACHE``) is refused. Checked events: ``open`` (builtins/io/os.open, so numpy.load,
       torch.load with a path, json reads ...), ``os.listdir``, ``os.scandir``, ``shutil.copyfile``,
       ``shutil.copytree``, ``ctypes.dlopen``, ``subprocess.Popen``, ``os.exec``, ``os.posix_spawn``, ``os.spawn``
       (path-like arguments of a child process). Paths outside the area (interpreter, system libraries) pass.
    2. Names, wherever they are: the eight deferred files (``DEFERRED_NAMES``: the test cache, ``test_v3_nll.npz``,
       ``test_event_number.npy``, the four ``*_test_nll*.npy`` arrays, the raw Parquet table) and every
       ``*_hits_with_model_outputs.parquet`` are refused. The deferred names (only those) can be released by
       ``enable_final_test_access()``, which requires ``sfm.data.check_test_access_allowed()`` to pass (LOCK.json
       naming this machine, no path overrides). It is meant for the later ``final_test.py`` only; nothing calls it
       today and it is off by default.
    3. Parquet readers: while the guard is active, importing ``pyarrow`` (any submodule; in particular
       ``pyarrow.parquet`` / ``pyarrow.dataset``, which pandas loads only when it reads Parquet), ``fastparquet``,
       ``polars`` or ``duckdb`` is refused with an ``ImportError`` (pandas treats pyarrow as optional and keeps working;
       ``sfm.data`` imports pyarrow only inside the cache-build functions). Reason: Parquet reads go through C++ and
       raise no Python ``open`` event.
    A refusal raises (``GuardRefused``, a PermissionError; ``GuardRefusedImport``, an ImportError) and appends one
    line to ``<output root>/guard/guard.log``: UTC time, PID, script, refused path (the area prefix written as
    ``<area>``), reason.

Limits (stated honestly)
    * C extensions that open files through the C library directly (``fopen``/``open(2)`` in C/C++, e.g. a Parquet
      reader already imported, CUDA / cuDNN loading kernels, ``torch.load`` given an open file object created
      elsewhere) raise no audit event and are not seen. Rule 3 closes the Parquet route only for readers imported
      after the guard; ``identity_record()`` reports whether a pyarrow module was loaded before the guard.
    * A subprocess runs without the hook; only its command line is checked (path-like arguments). A program it starts
      can open anything its own code decides.
    * ``os.stat`` / ``os.path.exists`` / ``os.readlink`` raise no audit event: metadata of a refused path (size,
      mtime, existence) can be read; file contents cannot through Python.
    * Audit hooks cannot be removed and are process-wide; threads are covered, other processes are not.
"""
from __future__ import annotations

import fnmatch
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import paths

__all__ = ["GuardRefused", "GuardRefusedImport", "GUARD_ENV", "DEFERRED_NAMES", "FOREIGN_PATTERNS", "install",
           "active", "state", "check_path", "enable_final_test_access", "guard_log_path"]

GUARD_ENV = "SFM_GUARD_AREA"
DEFERRED_NAMES = frozenset({
    "pion_s1_test.pt", "test_v3_nll.npz", "test_event_number.npy",
    "mlp_test_nll.npy", "formula_test_nll.npy", "mlp_test_nll_ref1056.npy", "formula_test_nll_ref1056.npy",
    "pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet"})
FOREIGN_PATTERNS = ("*_hits_with_model_outputs.parquet",)
REFUSED_IMPORTS = ("pyarrow", "fastparquet", "polars", "duckdb")
PATH_EVENTS = {"open": (0,), "os.listdir": (0,), "os.scandir": (0,), "shutil.copyfile": (0, 1),
               "shutil.copytree": (0, 1), "ctypes.dlopen": (0,)}
PROC_EVENTS = ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn")

_STATE = {"active": False, "area": None, "allowed": [], "allow_deferred": False, "installed_hook": False,
          "pyarrow_loaded_before_guard": None, "refusals": 0}
_TLS = threading.local()


class GuardRefused(PermissionError):
    """An access refused by the file-access guard."""


class GuardRefusedImport(ImportError):
    """An import of a Parquet reader refused by the file-access guard."""


def _real(p) -> Optional[str]:
    if isinstance(p, int) or p is None:
        return None
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    if not isinstance(p, (str, os.PathLike)):
        return None
    p = os.fspath(p)
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    if not p:
        return None
    return os.path.realpath(os.path.abspath(p))


def _under(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip(os.sep) + os.sep)


def guard_log_path() -> Path:
    return Path(paths.OUT_ROOT) / "guard" / "guard.log"


def state() -> dict:
    return {k: (list(v) if isinstance(v, list) else v) for k, v in _STATE.items()}


def active() -> bool:
    return bool(_STATE["active"])


def check_path(raw) -> Optional[str]:
    """The reason a path would be refused, or None (pure function of the guard's state; usable while inert)."""
    rp = _real(raw)
    names = set()
    if rp is not None:
        names.add(os.path.basename(rp))
    if isinstance(raw, (str, bytes, os.PathLike)):
        try:
            names.add(os.path.basename(os.fsdecode(os.fspath(raw))))
        except TypeError:
            pass
    for n in names:
        if any(fnmatch.fnmatch(n, pat) for pat in FOREIGN_PATTERNS):
            return f"file of another project ({n})"
        if n in DEFERRED_NAMES and not _STATE["allow_deferred"]:
            return f"deferred file ({n}): opened only by the final test after Gate 2c"
    area = _STATE["area"]
    if rp is not None and area and _under(rp, area) and not any(_under(rp, a) for a in _STATE["allowed"]):
        return "inside the guarded area but outside the clone, the output root and the cache directory"
    return None


def _portable(p: str) -> str:
    area = _STATE["area"]
    if area and isinstance(p, str) and _under(p, area):
        return "<area>" + p[len(area.rstrip(os.sep)):]
    return str(p)


def _log(path_text: str, reason: str) -> None:
    try:
        lp = guard_log_path()
        lp.parent.mkdir(parents=True, exist_ok=True)
        script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "<interactive>"
        with open(lp, "a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\tpid {os.getpid()}\t{script}\t"
                    f"{_portable(path_text)}\t{reason}\n")
    except Exception:                                                      # never mask the refusal itself
        pass


def _refuse(path_text: str, reason: str, exc=GuardRefused):
    _TLS.busy = True
    try:
        _STATE["refusals"] += 1
        _log(path_text, reason)
    finally:
        _TLS.busy = False
    raise exc(f"[sfm.guard] refused: {_portable(path_text)} - {reason}")


def _check_proc_args(args) -> None:
    items = []
    for a in args:
        if isinstance(a, (list, tuple)):
            items.extend(a)
        else:
            items.append(a)
    for a in items:
        if isinstance(a, (str, bytes, os.PathLike)):
            s = os.fsdecode(os.fspath(a))
            if os.sep in s or s.endswith(".parquet") or s in DEFERRED_NAMES:
                reason = check_path(s)
                if reason:
                    _refuse(os.path.realpath(os.path.abspath(s)), f"subprocess argument: {reason}")


def _hook(event: str, args) -> None:
    if not _STATE["active"] or getattr(_TLS, "busy", False):
        return
    if event == "import":
        name = args[0] if args else ""
        if isinstance(name, str) and any(name == m or name.startswith(m + ".") for m in REFUSED_IMPORTS):
            _refuse(name, "import of a Parquet reader while the guard is active", GuardRefusedImport)
        return
    idx = PATH_EVENTS.get(event)
    if idx is not None:
        for i in idx:
            if i < len(args):
                reason = check_path(args[i])
                if reason:
                    _refuse(_real(args[i]) or str(args[i]), f"{event}: {reason}")
        return
    if event in PROC_EVENTS:
        _check_proc_args(args)


def install(area: Optional[str] = None) -> bool:
    """Activate the guard if ``area`` (default: ``$SFM_GUARD_AREA``) is set; idempotent. Returns whether active."""
    area = area or os.environ.get(GUARD_ENV)
    if not area:
        return _STATE["active"]
    area_r = os.path.realpath(os.path.abspath(os.path.expanduser(area)))
    allowed = [os.path.realpath(str(paths.REPO_ROOT)), os.path.realpath(str(paths.OUT_ROOT))]
    uc = os.environ.get("SFM_USER_CACHE")
    if uc:
        allowed.append(os.path.realpath(os.path.abspath(os.path.expanduser(uc))))
    if _STATE["pyarrow_loaded_before_guard"] is None:
        _STATE["pyarrow_loaded_before_guard"] = sorted(m for m in sys.modules if m.split(".")[0] in REFUSED_IMPORTS)
    _STATE.update(area=area_r, allowed=allowed, active=True)
    if not _STATE["installed_hook"]:
        sys.addaudithook(_hook)
        _STATE["installed_hook"] = True
    return True


def enable_final_test_access() -> None:
    """Release the DEFERRED names (not the other project's files, not the area rule) for the final test run only.
    Requires LOCK.json naming this machine and no path override (``sfm.data.check_test_access_allowed``). Nothing in
    the code calls it today; ``final_test.py`` (not written) is its only intended caller."""
    from . import data
    data.check_test_access_allowed()
    _STATE["allow_deferred"] = True
    _TLS.busy = True
    try:
        _log("<deferred names>", "released for the final test (enable_final_test_access)")
    finally:
        _TLS.busy = False
