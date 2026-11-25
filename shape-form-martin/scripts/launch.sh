#!/usr/bin/env bash
# Start ONE campaign command detached, with a PID file, a job file and a log file under the output root.
#
#   SFM_OUT_ROOT=<out> SFM_USER_CACHE=<cache> shape-form-martin/scripts/launch.sh <job-name> <command> [args...]
#
# Example (from the clone root; always start from there - the file-access guard resolves relative arguments
# against the working directory):
#   shape-form-martin/scripts/launch.sh val_parity .venv/bin/python -B shape-form-martin/scripts/val_parity.py
#
# Required environment:
#   SFM_OUT_ROOT     output root; on the H200 a directory OUTSIDE the clone (the clone stays pristine)
#   SFM_USER_CACHE   a directory in the user's area; caches and temporary files go to subdirectories of it
# Optional: SFM_CPU_THREADS (default 4), SFM_GPU_MEM_GB (per-job GPU memory cap, read by sfm.machine; default
# 10 GB on a device larger than 24 GB), SFM_GUARD_AREA (default: the directory containing the clone; "off" disables
# the file-access guard of sfm.guard - campaign code then refuses to start off the laptop of record),
# SFM_LAUNCH_GRACE (seconds to wait before checking that the job is alive; default 3).
#
# Writes under $SFM_OUT_ROOT/jobs/: <job>_<UTC>.log (output, then "# exit status N at <time>"), .pid (the PID of the
# command itself, not of a wrapper), .job (pid, start time from /proc, working directory, command line: what
# stop_job.sh checks before it signals), .exit (exit status, when finished). Prints the paths.
# A command that dies within the grace period is reported as FAILED (exit code 1).
# Stop a job ONLY with:   shape-form-martin/scripts/stop_job.sh <pid file>
# (shared account, PIDs are reused: never kill by name, never a bare kill of a PID that was not checked).
# Never start "uv run ..." through this launcher (the PID would be uv's, not the interpreter's): use .venv/bin/python.
# No host names, addresses or absolute paths are written into the repository by this script.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: SFM_OUT_ROOT=<out> SFM_USER_CACHE=<cache> $0 <job-name> <command> [args...]" >&2
    exit 2
fi
: "${SFM_OUT_ROOT:?set SFM_OUT_ROOT (the output root; on the H200 outside the clone)}"
: "${SFM_USER_CACHE:?set SFM_USER_CACHE (cache and temp directory in your area)}"
name=$1
shift
if [[ ! "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "job name must match [A-Za-z0-9._-]+" >&2
    exit 2
fi
case "$(basename -- "$1")" in
    uv|uvx|conda|mamba|pixi|poetry)
        echo "refusing to start '$1' through the launcher: the PID file would name the wrapper, not the interpreter;" \
             "use .venv/bin/python directly" >&2
        exit 2 ;;
esac

# File-access guard (sfm.guard): the guarded area defaults to the directory that contains the clone; inside it only
# the clone, the output root and SFM_USER_CACHE may be opened.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ "${SFM_GUARD_AREA:-}" == "off" ]]; then
    unset SFM_GUARD_AREA
else
    export SFM_GUARD_AREA="${SFM_GUARD_AREA:-$(dirname "$repo_root")}"
fi
mkdir -p "$SFM_USER_CACHE"/{pip,xdg,matplotlib,cuda,torch,tmp}
export SFM_USER_CACHE
export PIP_CACHE_DIR="$SFM_USER_CACHE/pip"
export XDG_CACHE_HOME="$SFM_USER_CACHE/xdg"
export MPLCONFIGDIR="$SFM_USER_CACHE/matplotlib"
export CUDA_CACHE_PATH="$SFM_USER_CACHE/cuda"
export TORCH_HOME="$SFM_USER_CACHE/torch"
export TMPDIR="$SFM_USER_CACHE/tmp"
export SFM_CPU_THREADS="${SFM_CPU_THREADS:-4}"
export OMP_NUM_THREADS="$SFM_CPU_THREADS" MKL_NUM_THREADS="$SFM_CPU_THREADS" OPENBLAS_NUM_THREADS="$SFM_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$SFM_CPU_THREADS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export SFM_OUT_ROOT

jobs_dir="$SFM_OUT_ROOT/jobs"
mkdir -p "$jobs_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base="$jobs_dir/${name}_${stamp}"
log="$base.log"; pidf="$base.pid"; jobf="$base.job"; exitf="$base.exit"
for f in "$log" "$pidf" "$jobf" "$exitf"; do
    if [[ -e "$f" ]]; then echo "refusing to overwrite $f" >&2; exit 1; fi
done
{
    echo "# job ${name}, started ${stamp} (UTC)"
    echo "# command: $*"
    echo "# SFM_CPU_THREADS=${SFM_CPU_THREADS} SFM_GPU_MEM_GB=${SFM_GPU_MEM_GB:-<default>} guard=${SFM_GUARD_AREA:+on}"
} > "$log"

export SFM_JOB_LOG="$log" SFM_JOB_PID="$pidf" SFM_JOB_EXIT="$exitf"
setsid_cmd=()
if command -v setsid > /dev/null 2>&1; then setsid_cmd=(setsid); fi
# The wrapper starts the command in the background (so $! is the command's own PID), records it, waits for it and
# appends the exit status. It holds no lock and does nothing else.
nohup "${setsid_cmd[@]}" bash -c '
    "$@" >> "$SFM_JOB_LOG" 2>&1 < /dev/null &
    child=$!
    echo "$child" > "$SFM_JOB_PID.tmp" && mv "$SFM_JOB_PID.tmp" "$SFM_JOB_PID"
    wait "$child"; st=$?
    echo "# exit status $st at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SFM_JOB_LOG"
    echo "$st" > "$SFM_JOB_EXIT"
' sfm-job "$@" > /dev/null 2>&1 < /dev/null &

for _ in $(seq 1 100); do [[ -s "$pidf" ]] && break; sleep 0.1; done
if [[ ! -s "$pidf" ]]; then
    echo "FAILED: no PID recorded for job ${name} (see ${log})" >&2
    exit 1
fi
pid=$(cat "$pidf")
if [[ -r "/proc/$pid/stat" ]]; then
    starttime=$(sed 's/^.*) //' "/proc/$pid/stat" | awk '{print $20}')
    cwd=$(readlink "/proc/$pid/cwd" || true)
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" || true)
    printf 'pid=%s\nstarttime=%s\ncwd=%s\ncmdline=%s\n' "$pid" "$starttime" "$cwd" "$cmdline" > "$jobf"
fi
sleep "${SFM_LAUNCH_GRACE:-3}"
if [[ -e "$exitf" ]] || ! kill -0 "$pid" 2> /dev/null; then
    st=$(cat "$exitf" 2> /dev/null || echo "?")
    echo "FAILED: job ${name} ended within ${SFM_LAUNCH_GRACE:-3} s (exit status ${st}); read ${log}" >&2
    exit 1
fi
echo "started job: ${name} (alive after ${SFM_LAUNCH_GRACE:-3} s)"
echo "pid file:    ${pidf}   (pid ${pid})"
echo "log file:    ${log}"
echo "stop with:   shape-form-martin/scripts/stop_job.sh ${pidf}"
