#!/usr/bin/env bash
# Run a CPU-heavy command next to a model server without slowing it down: in a transient user scope of CPU weight 1.
#   tools/low_cpu.sh COMMAND [ARGS...]          (blocks like the command; use nohup/setsid around it as usual)
#   LOW_CPU_CPUS=70,71,78 tools/low_cpu.sh ...   (also only on these CPUs: the ones without CPU expert threads)
#
# Every MoE layer waits for all CPU expert threads, and they take nearly every core. Linux shares CPU time between
# control groups first: a login session (session-N.scope) and the user manager that runs the server (user@UID.service)
# get half of a contended core each, whatever the nice value or scheduling class (2026-11-10 audit R1, DeepSeek
# V4.1: 37 decode steps/s idle, 3.5 with 16 busy loops at nice 19, 0.4 with them on expert cores). In a scope of
# weight 1 next to the console's server scope (CPUWeight 10000) the same loops cost decode a few percent and get the
# CPU time the server leaves. Needs a user systemd (systemd-run --user); without one the command runs as is.
set -e
[ $# -gt 0 ] || { echo "usage: $0 COMMAND [ARGS...]" >&2; exit 2; }
if [ -n "$LOW_CPU_CPUS" ]; then
  set -- taskset -c "$LOW_CPU_CPUS" "$@"
fi
if systemd-run --user --scope --quiet --collect true 2>/dev/null; then
  exec systemd-run --user --scope --quiet --collect -p CPUWeight=1 -- "$@"
fi
echo "low_cpu: no user systemd here, running at nice 19 (that does not protect the server)" >&2
exec nice -n 19 "$@"
