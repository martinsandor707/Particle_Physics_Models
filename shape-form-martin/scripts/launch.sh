#!/usr/bin/env bash
# Start ONE campaign command detached (nohup), with a PID file and a log file under the output root.
#
#   SFM_OUT_ROOT=<out> SFM_USER_CACHE=<cache> shape-form-martin/scripts/launch.sh <job-name> <command> [args...]
#
# Example (from the clone root):
#   shape-form-martin/scripts/launch.sh val_parity .venv/bin/python -B shape-form-martin/scripts/val_parity.py --create-refset
#
# Required environment:
#   SFM_OUT_ROOT     output root; on the H200 a directory OUTSIDE the clone (the clone stays pristine)
#   SFM_USER_CACHE   a directory in the user's area; caches and temporary files go to subdirectories of it
# Optional: SFM_CPU_THREADS (default 4), SFM_GPU_MEM_GB (per-job GPU memory cap, read by sfm.machine; default
# 10 GB on a device larger than 24 GB), SFM_GUARD_AREA (default: the directory containing the clone; "off" disables
# the file-access guard of sfm.guard).
#
# Writes $SFM_OUT_ROOT/jobs/<job-name>_<UTC stamp>.pid and .log and prints both paths. Stop a job ONLY by its
# recorded PID:   kill "$(cat <pid file>)"      (shared account: never pkill / killall / anything by name)
# No host names, addresses or absolute paths are written by this script; everything comes from its arguments and
# the environment.
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

# File-access guard (sfm.guard): the guarded area defaults to the directory that contains the clone; inside it only
# the clone, the output root and SFM_USER_CACHE may be opened. SFM_GUARD_AREA=off disables it (not on the H200).
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ "${SFM_GUARD_AREA:-}" == "off" ]]; then
    unset SFM_GUARD_AREA
else
    export SFM_GUARD_AREA="${SFM_GUARD_AREA:-$(dirname "$repo_root")}"
fi
mkdir -p "$SFM_USER_CACHE"/{pip,xdg,matplotlib,cuda,torch,tmp}
export PIP_CACHE_DIR="$SFM_USER_CACHE/pip"
export XDG_CACHE_HOME="$SFM_USER_CACHE/xdg"
export MPLCONFIGDIR="$SFM_USER_CACHE/matplotlib"
export CUDA_CACHE_PATH="$SFM_USER_CACHE/cuda"
export TORCH_HOME="$SFM_USER_CACHE/torch"
export TMPDIR="$SFM_USER_CACHE/tmp"
export SFM_CPU_THREADS="${SFM_CPU_THREADS:-4}"
export OMP_NUM_THREADS="$SFM_CPU_THREADS" MKL_NUM_THREADS="$SFM_CPU_THREADS" OPENBLAS_NUM_THREADS="$SFM_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$SFM_CPU_THREADS"
export PYTHONDONTWRITEBYTECODE=1
export SFM_OUT_ROOT

jobs_dir="$SFM_OUT_ROOT/jobs"
mkdir -p "$jobs_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$jobs_dir/${name}_${stamp}.log"
pidf="$jobs_dir/${name}_${stamp}.pid"
if [[ -e "$log" || -e "$pidf" ]]; then
    echo "refusing to overwrite $log / $pidf" >&2
    exit 1
fi
{
    echo "# job ${name}, started ${stamp} (UTC)"
    echo "# command: $*"
    echo "# SFM_CPU_THREADS=${SFM_CPU_THREADS} SFM_GPU_MEM_GB=${SFM_GPU_MEM_GB:-<default>} guard=${SFM_GUARD_AREA:+on}"
} > "$log"
nohup "$@" >> "$log" 2>&1 < /dev/null &
echo $! > "$pidf"
echo "started job: ${name}"
echo "pid file:    ${pidf}   (pid $(cat "$pidf"))"
echo "log file:    ${log}"
echo "stop with:   kill \"\$(cat ${pidf})\""
