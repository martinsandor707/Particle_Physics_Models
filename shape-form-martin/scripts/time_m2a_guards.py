"""Timing only: each part of a guarded Track A evaluation, for registered runs of THIS machine, loaded read-only.
Writes nothing but its log (stdout). Through the launcher like every other job:

    $L time_guards $P -B shape-form-martin/scripts/time_m2a_guards.py A7_s0 A2s_s0 A9n_s0 --repeat 2

Why (STATUS.md, section R5): on the H200 the guards phase took 15-37 minutes per run, but the file times show that
the registry part (K = 1024, bulk stratum, mass monitors) took about 3 s per run; the time went into the G-region step
of run_m2a.py, which on the laptop takes about 2 s and on the H200 took 3 s for five runs in a row and 15-37 minutes
otherwise. This command measures, per part and per repeat:
    wall and process CPU seconds, major / minor page faults, resident set size, MemAvailable and swap in use, the
    Linux pressure-stall information (/proc/pressure: cpu, memory, io; avg10 and the stall time during the part),
    the load average, free device memory, peak allocated device memory and allocator retries.
Parts: load_run; the K = 64 metric of record (common.test_nll in deterministic()); the K = 1024 and bulk guard arrays
(registry._guard_arrays); the mass monitors (registry._mass); the G-region validation pass with return_parts (GPU);
the G-region host work as before the fix (event index and class masks recomputed) and as after it (cached).
A stall time that grows with the wall time of a host part identifies the contended resource.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_m2a as M  # noqa: E402
import run_m2a_lib as L  # noqa: E402

from sfm import data as sfm_data, metric, models, registry  # noqa: E402
from sfm.models import dist_factory as dfac  # noqa: E402


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _psi() -> dict:
    out = {}
    for res in ("cpu", "memory", "io"):
        txt = _read(f"/proc/pressure/{res}")
        for line in txt.splitlines():
            f = dict(x.split("=") for x in line.split()[1:])
            out[f"{res}_{line.split()[0]}_avg10"] = float(f.get("avg10", "nan"))
            out[f"{res}_{line.split()[0]}_total_us"] = int(f.get("total", 0))
    return out


def _mem() -> dict:
    mi = {ln.split(":")[0]: int(ln.split()[1]) for ln in _read("/proc/meminfo").splitlines() if ln.split()[1:]}
    st = {ln.split(":")[0]: ln.split()[1] for ln in _read("/proc/self/status").splitlines() if ln.split()[1:]}
    return {"mem_available_mb": mi.get("MemAvailable", 0) // 1024,
            "swap_used_mb": (mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) // 1024,
            "rss_mb": int(st.get("VmRSS", 0)) // 1024}


def _snap() -> dict:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {"wall": time.perf_counter(), "cpu": time.process_time(), "majflt": ru.ru_majflt, "minflt": ru.ru_minflt,
            "psi": _psi(), "load1": float((_read("/proc/loadavg").split() or ["nan"])[0])}


def timed(label: str, fn, rec: dict):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    r0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    a = _snap()
    out = fn()
    torch.cuda.synchronize()
    b = _snap()
    free, total = torch.cuda.mem_get_info()
    stall = {k.replace("_total_us", "_stall_s"): round((b["psi"][k] - a["psi"].get(k, 0)) / 1e6, 3)
             for k in b["psi"] if k.endswith("_total_us")}
    rec[label] = {"wall_s": round(b["wall"] - a["wall"], 3), "cpu_s": round(b["cpu"] - a["cpu"], 3),
                  "majflt": b["majflt"] - a["majflt"], "minflt": b["minflt"] - a["minflt"],
                  **_mem(), "load1": b["load1"],
                  **{k: v for k, v in b["psi"].items() if k.endswith("avg10")}, **stall,
                  "gpu_free_gb": round(free / 2 ** 30, 1), "gpu_total_gb": round(total / 2 ** 30, 1),
                  "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20),
                  "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0) - r0}
    print(f"  {label:30s} {json.dumps(rec[label])}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    L.configure(smoke=args.smoke)
    L.start("time_m2a_guards (timing only; writes nothing)")
    C = L.common
    print(f"host: {os.cpu_count()} CPUs; torch threads {torch.get_num_threads()}; cgroup cpu.max "
          f"{_read('/sys/fs/cgroup/cpu.max').strip() or '-'}; memory.max {_read('/sys/fs/cgroup/memory.max').strip() or '-'}"
          f"; {_mem()}; caps {L.CTX.batch.get('caps')}", flush=True)
    sets, extras = sfm_data.load_sets(("val",), device="cpu")
    val_cpu = sets["val"]

    def job():
        dev = C.DEVICE
        va = val_cpu.to(dev)
        idx = np.flatnonzero(va.ev["zb"].cpu().numpy() >= registry.ZB_STRATUM_MM)
        abar = dfac.acceptance_template("stored", dev)
        for rep in range(args.repeat):
            for n in args.runs:
                rec = {}
                print(f"== {n} (repeat {rep + 1} of {args.repeat})", flush=True)
                model = timed("load_run", lambda: L.rebuild(n, dev), rec)

                def k64():
                    with metric.deterministic():
                        return C.test_nll(model, va)
                timed("K=64 metric of record", k64, rec)
                timed("guard K=1024 + bulk", lambda: registry._guard_arrays(model, va, idx), rec)
                timed("guard mass monitors", lambda: registry._mass(model, va, abar), rec)
                parts = timed("G-region pass (GPU)", lambda: metric.test_nll(model, va, return_parts=True,
                                                                             deterministic_sum=True), rec)

                def host_old():
                    hit = parts["hit"][0].astype(np.float64)
                    N = len(val_cpu)
                    ev = np.repeat(np.arange(N), np.diff(val_cpu.off.numpy()))
                    masks = M._g_region_masks(val_cpu)
                    return [np.bincount(ev, weights=-hit * m, minlength=N) for m in masks.values()]

                def host_new():
                    hit = parts["hit"][0].astype(np.float64)
                    N, ev, masks = M._g_region_static(val_cpu)
                    return [np.bincount(ev, weights=-hit * m, minlength=N) for m in masks.values()]
                timed("G-region host (before fix)", host_old, rec)
                timed("G-region host (after fix)", host_new, rec)
                del model, parts
                torch.cuda.empty_cache()
        del va

    L.on_gpu("time_m2a_guards", job)
    print("timing finished; nothing was written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
