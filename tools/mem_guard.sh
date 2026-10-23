#!/usr/bin/env bash
# Host memory guard for test servers: kills the process group of PID when MemAvailable drops below MIN_GIB, so a
# test that overcommits host memory cannot push production services or the desktop into swap / the OOM killer.
#   tools/mem_guard.sh PID [MIN_GIB=20] [INTERVAL_S=1]
pid=$1 min=${2:-20} interval=${3:-1}
[ -n "$pid" ] || { echo "usage: $0 PID [MIN_GIB] [INTERVAL_S]" >&2; exit 2; }
pgid=$(ps -o pgid= "$pid" | tr -d ' ')
low=999999
while kill -0 "$pid" 2>/dev/null; do
  avail=$(awk '/^MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  [ "$avail" -lt "$low" ] && low=$avail
  if [ "$avail" -lt "$min" ]; then
    echo "$(date '+%F %T') mem_guard: MemAvailable $avail GiB < $min GiB, killing process group $pgid (pid $pid)"
    kill -9 -- "-$pgid" 2>/dev/null || kill -9 "$pid"
    exit 1
  fi
  sleep "$interval"
done
echo "$(date '+%F %T') mem_guard: pid $pid exited, lowest MemAvailable $low GiB"
