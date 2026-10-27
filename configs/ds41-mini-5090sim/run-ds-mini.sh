#!/usr/bin/env bash
# XTier: DeepSeek V4.1 Flash 1017 the smallest way, as on an RTX 5090 (32 GB) with 128 GB of host memory and the cold
# experts on 4 NVMe drives, simulated on this host: the systemd scope this runs in (the WebUI's xtier-dev.scope, a
# test's own scope) gets MemoryMax=XTIER_SIM_HOST_MAX (128G) without swap, so the whole run, load included, stays
# under it or is killed; tools/vram_limit.py holds the GPU memory above XTIER_SIM_VRAM_MIB (32000 MiB free, about
# what a 5090 leaves for a server) until this process exits. Then run-ds-xtier.sh with the settings of console.py.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
D=/data/fast/workspaces/xtier-dev
VRAM_MIB=${XTIER_SIM_VRAM_MIB:-32000} HOST_MAX=${XTIER_SIM_HOST_MAX:-128G}
cg=$(cut -d: -f3 /proc/self/cgroup)
unit=$(basename "$cg")
case "$unit" in
  *.scope) systemctl --user set-property --runtime "$unit" MemoryMax=$HOST_MAX MemorySwapMax=0 ;;
  *) echo "run-ds-mini: not in a systemd scope ($cg), so no host memory limit: not started" >&2; exit 1 ;;
esac
echo "run-ds-mini: $unit memory.max $(cat /sys/fs/cgroup$cg/memory.max) swap.max $(cat /sys/fs/cgroup$cg/memory.swap.max)"
ready=$(mktemp)
$D/env/bin/python $HERE/../../tools/vram_limit.py $VRAM_MIB $$ $ready &
for i in $(seq 1 120); do [ -s $ready ] && break; sleep 0.5; done
[ -s $ready ] || { echo "run-ds-mini: the GPU memory limit did not come up" >&2; exit 1; }
rm -f $ready
exec $HERE/../ds41-pro5000-nps4/run-ds-xtier.sh "$@"
