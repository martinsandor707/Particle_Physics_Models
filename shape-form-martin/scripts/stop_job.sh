#!/usr/bin/env bash
# Stop ONE job started by launch.sh, after checking that the PID still belongs to it.
#
#   shape-form-martin/scripts/stop_job.sh <out>/jobs/<job>_<UTC>.pid [SIGNAL]      (default SIGTERM)
#
# Reads the .job file written by launch.sh (pid, start time from /proc/<pid>/stat, working directory, command line)
# and signals the process ONLY if all four still match /proc/<pid>. A reused PID (another user's or another job's
# process) therefore never receives the signal. Nothing is killed by name.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "usage: $0 <pid file> [SIGNAL]" >&2
    exit 2
fi
pidf=$1
sig=${2:-TERM}
jobf="${pidf%.pid}.job"
if [[ ! -s "$pidf" || ! -s "$jobf" ]]; then
    echo "refusing: $pidf or $jobf missing (only jobs started by launch.sh can be stopped with this helper)" >&2
    exit 1
fi
pid=$(cat "$pidf")
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then echo "refusing: bad pid '$pid'" >&2; exit 1; fi
want_start=$(sed -n 's/^starttime=//p' "$jobf")
want_cwd=$(sed -n 's/^cwd=//p' "$jobf")
want_cmd=$(sed -n 's/^cmdline=//p' "$jobf")
if [[ "$(sed -n 's/^pid=//p' "$jobf")" != "$pid" ]]; then
    echo "refusing: the .job file names another pid" >&2; exit 1
fi
if [[ ! -r "/proc/$pid/stat" ]]; then
    echo "job not running (pid $pid has ended); nothing signalled"
    exit 0
fi
cur_start=$(sed 's/^.*) //' "/proc/$pid/stat" | awk '{print $20}')
cur_cwd=$(readlink "/proc/$pid/cwd" || true)
cur_cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" || true)
if [[ "$cur_start" != "$want_start" || "$cur_cwd" != "$want_cwd" || "$cur_cmd" != "$want_cmd" ]]; then
    echo "refusing: pid $pid no longer belongs to this job (start time / working directory / command line differ);" \
         "nothing signalled" >&2
    exit 1
fi
kill "-$sig" "$pid"
echo "sent SIG$sig to pid $pid (start time, working directory and command line checked)"
