"""sfm.machine - the identity of the machine that computes a number, and the start-up checks of every GPU entry point.

    tag = machine.startup("run_m2b_fit")      # hard failure without CUDA; TF32 off; returns the live MachineTag
    tag.id                                    # e.g. "nvidia-h200-g1056-3f2a9c1d" (names result directories)
    machine.identity_record()                 # R0: the tag plus provenance (host id, devices, versions, git state)

Why (PLAN.md section 13, A6 revised; ``transfer/REVIEW_STATUS.md`` section 4). ``common.test_nll`` draws its fold
points from the CUDA Philox generator, whose stream depends on the GPU's launch grid, and the arithmetic depends on
the GPU, the torch / CUDA build and a few switches. Every per-event array and every registry row therefore carries a
*machine tag* (this module) and a *points tag* (``sfm.provenance``); a delta of record needs both equal.

Machine tag (``MachineTag.fields``)
    gpu_name, sm_count, max_threads_per_sm, launch_grid (``sfm.refstream.local_grid()``), torch, cuda (build),
    cudnn, python, tf32_matmul, tf32_cudnn, deterministic, draw ("native" or "emulated"), cublas_workspace_config.
    ``id`` = readable prefix + the first 8 hex digits of the SHA-256 of the canonical JSON of all fields, so two
    machines (or two builds, two Python versions, TF32 on/off, ...) can never share an id. The host is provenance
    only, and only as ``host_id()`` (a hash): no host name, address or user name is ever written to a file (the
    repository is public).

E10 (PLAN.md section 16.7)
    ``common.py`` silently falls back to the CPU when CUDA is unavailable (``common.DEVICE``). ``startup`` /
    ``require_cuda`` turn that into a hard failure, and ``disable_tf32`` sets
    ``torch.backends.cuda.matmul.allow_tf32 = False`` and ``torch.backends.cudnn.allow_tf32 = False`` explicitly
    (user decision P3). Both are called from every campaign script entry point, from ``sfm.registry.evaluate`` and
    from ``sfm.train.fit``. Only ``sfm.train.fit`` accepts an explicit opt-in to the CPU (environment variable
    ``SFM_ALLOW_CPU=1``; used by the CPU unit tests); registry rows and scripts never run on the CPU. TF32 is also
    switched off when this module is imported, so every machine tag computed in a campaign process reads it off.

Shared GPU (H200 constraints of 25 Nov 2026)
    ``startup`` also applies the per-job caps: ``torch.cuda.set_per_process_memory_fraction`` from
    ``SFM_GPU_MEM_GB`` (default 10 GB on a device larger than 24 GB, no cap on smaller devices such as the laptop;
    "0" = no cap) and the CPU threads from ``SFM_CPU_THREADS`` (default 4 on a device larger than 24 GB;
    ``torch.set_num_threads`` and OMP/MKL/OpenBLAS/NumExpr variables). ``check_free_memory`` (called after the GPU
    lock is taken) refuses to start when less than the job's cap + 4 GB is free; it never waits.
    In pristine mode (``paths.pristine_mode()``: the output root lies outside the clone) ``startup`` refuses to run
    in a clone whose ``git status --porcelain`` is not empty; the git state is recorded by every batch command.
    (``SFM_ALLOW_DIRTY_CLONE=1`` lifts the refusal for dry runs on the laptop; it is recorded in the git state and
    must never be set on the H200.)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import torch

from . import paths

if str(paths.MARTIN_DIR) not in sys.path:
    sys.path.insert(0, str(paths.MARTIN_DIR))
import common  # noqa: E402  (shape-form-martin/common.py, the metric of record; not modified)

__all__ = ["NoCUDA", "DirtyClone", "InsufficientGPUMemory", "MachineTag", "ALLOW_CPU_ENV", "LAPTOP_RECORD_LABEL",
           "LAPTOP_RECORD", "disable_tf32", "tf32_flags", "require_cuda", "startup", "machine_tag", "host_id",
           "identity_record", "short_label", "apply_caps", "caps", "check_free_memory", "git_state",
           "require_clean_clone", "machine_dir_live"]

ALLOW_CPU_ENV = "SFM_ALLOW_CPU"
# The laptop that produced every array and row of 24 Nov 2026. Its rows carry no tag (schema 1); they are records.
LAPTOP_RECORD = {"gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU", "launch_grid": 144, "torch": "2.14.0+cu130",
                 "python": "3.12", "note": "rows and arrays of 24 Nov 2026 written before machine tags existed"}
LAPTOP_RECORD_LABEL = "laptop-rtx4060-g144 (untagged laptop record, 24 Nov 2026)"


class NoCUDA(RuntimeError):
    """CUDA is not available: campaign numbers are never computed on the CPU (common.py would fall back silently)."""


class DirtyClone(RuntimeError):
    """Pristine mode and the clone has local changes: a later ``git pull`` could abort."""


class InsufficientGPUMemory(RuntimeError):
    """Less than the job's memory cap + 4 GB is free on the (shared) device."""


LARGE_DEVICE_GB = 24.0
DEFAULT_CAP_GB_LARGE = 10.0
DEFAULT_THREADS_LARGE = 4
FREE_MARGIN_GB = 4.0
_caps_applied: dict = {}


def disable_tf32() -> dict:
    """Switch TF32 off for matmul and cuDNN (P3) and return the two flags as read back."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flags = tf32_flags()
    if flags["tf32_matmul"] or flags["tf32_cudnn"]:
        raise RuntimeError(f"TF32 could not be disabled: {flags}")
    return flags


def tf32_flags() -> dict:
    return {"tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}


def require_cuda(what: str = "this step", allow_cpu_env: bool = False) -> None:
    """Raise ``NoCUDA`` unless CUDA is available and ``common.DEVICE`` is a CUDA device. With ``allow_cpu_env`` the
    environment variable ``SFM_ALLOW_CPU=1`` is an explicit opt-in to the CPU (only ``sfm.train.fit`` passes it)."""
    if torch.cuda.is_available() and torch.device(common.DEVICE).type == "cuda":
        return
    if allow_cpu_env and os.environ.get(ALLOW_CPU_ENV) == "1":
        return
    raise NoCUDA(f"{what}: CUDA is not available (torch {torch.__version__}, torch.version.cuda "
                 f"{torch.version.cuda}, common.DEVICE {common.DEVICE}). common.py would silently run on the CPU, "
                 "with another fold-point stream; refusing. Check the driver (nvidia-smi: CUDA >= the torch build), "
                 "CUDA_VISIBLE_DEVICES and the torch build." + (f" (Set {ALLOW_CPU_ENV}=1 only for CPU unit tests.)"
                                                                 if allow_cpu_env else ""))


def startup(what: str = "script", *, allow_cpu_env: bool = False, deterministic: bool = True) -> "MachineTag":
    """Entry-point check (E10): hard failure without CUDA, TF32 off, the per-job caps, the clean-clone check in
    pristine mode, then the live machine tag."""
    from . import guard
    guard.install()                                      # file-access guard (inert unless SFM_GUARD_AREA is set)
    require_cuda(what, allow_cpu_env=allow_cpu_env)
    disable_tf32()
    apply_caps()
    require_clean_clone(what)
    return machine_tag(deterministic=deterministic)


def host_id() -> str:
    """A stable, non-reversible identifier of this host (no host name is ever written to a file)."""
    return hashlib.sha256(b"sfm-host|" + socket.gethostname().encode()).hexdigest()[:12]


def _env_float(name: str) -> Optional[float]:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    if v.strip().lower() in ("0", "none", "off"):
        return 0.0
    return float(v)


def caps() -> dict:
    """The caps in force for this process (after ``apply_caps``)."""
    return dict(_caps_applied)


def apply_caps() -> dict:
    """Per-job caps on a shared device (idempotent): GPU memory fraction and CPU threads (module docstring)."""
    if _caps_applied:
        return dict(_caps_applied)
    out = {"gpu_mem_cap_gb": None, "cpu_threads": None, "device_total_gb": None}
    total_gb = None
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        out["device_total_gb"] = total_gb
    large = total_gb is not None and total_gb > LARGE_DEVICE_GB
    cap = _env_float("SFM_GPU_MEM_GB")
    cap = (DEFAULT_CAP_GB_LARGE if large else None) if cap is None else (cap or None)
    if cap and total_gb:
        frac = min(1.0, cap / total_gb)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        out.update(gpu_mem_cap_gb=float(cap), gpu_mem_fraction=frac)
    th = _env_float("SFM_CPU_THREADS")
    th = (DEFAULT_THREADS_LARGE if large else None) if th is None else (int(th) or None)
    if th:
        torch.set_num_threads(int(th))
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[k] = str(int(th))
        out["cpu_threads"] = int(th)
    _caps_applied.update(out)
    return dict(out)


def check_free_memory(what: str = "job") -> dict:
    """After the GPU lock is taken: refuse (``InsufficientGPUMemory``) if less than cap + 4 GB is free. Without a
    cap (laptop) nothing is checked here (the caller keeps its own rule). Never waits."""
    c = apply_caps()
    free, total = torch.cuda.mem_get_info()
    out = {"free_gb": free / 2 ** 30, "total_gb": total / 2 ** 30, "cap_gb": c.get("gpu_mem_cap_gb")}
    if c.get("gpu_mem_cap_gb"):
        need = c["gpu_mem_cap_gb"] + FREE_MARGIN_GB
        out["need_gb"] = need
        if out["free_gb"] < need:
            raise InsufficientGPUMemory(f"{what}: only {out['free_gb']:.1f} GB free on the shared device, need the "
                                        f"job's cap {c['gpu_mem_cap_gb']:.1f} GB + {FREE_MARGIN_GB:.0f} GB = "
                                        f"{need:.1f} GB. Not started (no waiting: the device is shared). Lower "
                                        "SFM_GPU_MEM_GB if the job fits, or start it later.")
    return out


def git_state() -> dict:
    """``git rev-parse HEAD`` and whether ``git status --porcelain`` is empty (the clone of the code)."""
    def run(*args):
        return subprocess.run(["git", "-C", str(paths.REPO_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout
    try:
        commit = run("rev-parse", "HEAD").strip()
        porcelain = run("status", "--porcelain")
        return {"commit": commit, "clean": porcelain.strip() == "", "n_changed": len(porcelain.splitlines())}
    except Exception as err:                                                     # pragma: no cover
        return {"commit": None, "clean": None, "error": type(err).__name__}


def require_clean_clone(what: str = "script") -> dict:
    """In pristine mode, refuse to start unless the clone is clean (``DirtyClone``)."""
    gs = git_state()
    if paths.pristine_mode() and gs.get("clean") is not True and os.environ.get("SFM_ALLOW_DIRTY_CLONE") == "1":
        gs["dirty_clone_allowed_by_env"] = True        # laptop dry runs only (recorded); never on the H200
        return gs
    if paths.pristine_mode() and gs.get("clean") is not True:
        raise DirtyClone(f"{what}: SFM_OUT_ROOT is outside the clone (pristine mode) but `git status --porcelain` "
                         f"is not empty ({gs.get('n_changed')} entries) or git failed; refusing to start. The clone "
                         "must stay exactly as pulled.")
    return gs


def machine_dir_live() -> "os.PathLike":
    """This machine's output directory: OUT_RESULTS / machines / <machine id>."""
    return paths.OUT_RESULTS / "machines" / machine_tag().id


def _slug(s: str) -> str:
    s = re.sub(r"^nvidia\s+", "", str(s).strip(), flags=re.I)
    s = re.sub(r"geforce\s+", "", s, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:32] or "unknown"


@dataclass(frozen=True)
class MachineTag:
    """Arithmetic part of the provenance of an array (see the module docstring)."""
    fields_json: str                                   # canonical JSON of the fields (frozen, hashable)

    @classmethod
    def from_fields(cls, fields: dict) -> "MachineTag":
        return cls(json.dumps(dict(fields), sort_keys=True, default=str))

    @property
    def fields(self) -> dict:
        return json.loads(self.fields_json)

    @property
    def sha8(self) -> str:
        return hashlib.sha256(self.fields_json.encode()).hexdigest()[:8]

    @property
    def id(self) -> str:
        f = self.fields
        grid = f.get("launch_grid")
        return f"{_slug(f.get('gpu_name', 'unknown'))}-g{grid if grid is not None else 'na'}-{self.sha8}"

    def to_dict(self) -> dict:
        return {"id": self.id, "fields": self.fields}

    def with_(self, **kw) -> "MachineTag":
        f = self.fields
        f.update(kw)
        return MachineTag.from_fields(f)


def machine_tag(deterministic: bool = True, draw: str = "native", device: int = 0) -> MachineTag:
    """The live machine tag. ``deterministic``: the PyTorch deterministic-algorithms flag under which the array is
    computed (every registry evaluation: True). ``draw``: "native" (the GPU's own generator) or "emulated"
    (``sfm.refstream``). On a machine without CUDA (only reachable through the explicit CPU opt-in of the unit tests)
    the GPU fields are "cpu" / None."""
    if draw not in ("native", "emulated"):
        raise ValueError("draw must be 'native' or 'emulated'")
    if torch.cuda.is_available() and torch.device(common.DEVICE).type == "cuda":
        from . import refstream
        p = torch.cuda.get_device_properties(device)
        gpu = {"gpu_name": torch.cuda.get_device_name(device), "sm_count": int(p.multi_processor_count),
               "max_threads_per_sm": int(p.max_threads_per_multi_processor),
               "launch_grid": int(refstream.local_grid(device))}
    else:
        gpu = {"gpu_name": "cpu", "sm_count": None, "max_threads_per_sm": None, "launch_grid": None}
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return MachineTag.from_fields({
        **gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": cudnn,
        "python": platform.python_version(), **tf32_flags(), "deterministic": bool(deterministic), "draw": draw,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})


def short_label(tag: Optional[MachineTag] = None) -> str:
    """'host_id=<hash> gpu=<name> grid=<g> machine=<id>' (for log lines; no host name)."""
    tag = tag or machine_tag()
    f = tag.fields
    return f"host_id={host_id()}\tgpu={f.get('gpu_name')}\tgrid={f.get('launch_grid')}\tmachine={tag.id}"


def identity_record(generator_check: Optional[dict] = None) -> dict:
    """Row R0 of PLAN.md section 16.4: the machine tag plus provenance that is not part of the tag."""
    tag = machine_tag()
    rec = {"machine": tag.to_dict(), "host_id": host_id(), "platform": platform.platform(),
           "python_executable": paths.portable(sys.executable), "cuda_available": torch.cuda.is_available(),
           "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "common_device": str(common.DEVICE)}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec["device0"] = {"name": p.name, "total_memory_mb": p.total_memory / 2 ** 20,
                          "multi_processor_count": p.multi_processor_count,
                          "max_threads_per_multi_processor": p.max_threads_per_multi_processor,
                          "capability": f"{p.major}.{p.minor}"}
        try:
            rec["driver"] = {"cuda_driver_version": torch.cuda.driver_version()
                             if hasattr(torch.cuda, "driver_version") else None}
        except Exception:                                                       # pragma: no cover
            rec["driver"] = None
        rec["mig_suspected"] = "MIG" in p.name
    import numpy as np
    rec["versions"] = {"numpy": np.__version__, "torch": torch.__version__}
    for mod in ("scipy", "pyarrow"):
        try:
            rec["versions"][mod] = __import__(mod).__version__
        except Exception:
            rec["versions"][mod] = None
    rec["env_overrides"] = {k: (paths.portable(os.environ[k]) if os.environ.get(k) else None)
                            for k in ("SFM_DATA_DIR", "SFM_RESULTS_DIR", "SFM_CACHE_DIR", "SFM_GPU_LOCK", "SFM_OUT_ROOT")}
    rec["env_overrides"]["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    rec["caps"] = apply_caps()
    rec["pristine_mode"] = paths.pristine_mode()
    from . import guard
    gst = guard.state()
    rec["file_access_guard"] = {"active": gst["active"], "area_set": bool(gst["area"]),
                                "allowed_roots": [paths.portable(a) for a in gst["allowed"]] if gst["active"] else [],
                                "deferred_names_released": gst["allow_deferred"],
                                "parquet_reader_modules_loaded_before_guard": gst["pyarrow_loaded_before_guard"],
                                "refusals_so_far": gst["refusals"],
                                "log": paths.portable(guard.guard_log_path()) if gst["active"] else None}
    rec["git"] = git_state()
    if generator_check is not None:
        rec["generator_check"] = generator_check
    return rec


disable_tf32()                       # E10: TF32 off in every process that imports the campaign code
