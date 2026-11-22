#!/usr/bin/env bash
# Build the CPU expert engine (csrc/) and install the libraries into xtier/engine/lib.
#   tools/build_engine.sh [CUDA_ROOT]
# CUDA_ROOT: directory with include/ and lib/ of the CUDA runtime (e.g. site-packages/nvidia/cu13); without it the
# mailbox library (libxtier_mailbox.so) is not built. Extra CMake options can be passed in XTIER_CMAKE_ARGS.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
build=${XTIER_BUILD_DIR:-$root/build/engine}
cuda=${1:-}
cmake -S "$root/csrc" -B "$build" -DCMAKE_BUILD_TYPE=Release ${cuda:+-DXTIER_CUDA_ROOT="$cuda"} ${XTIER_CMAKE_ARGS:-}
cmake --build "$build" -j "${XTIER_BUILD_JOBS:-8}"
mkdir -p "$root/xtier/engine/lib"
# Atomic replace (copy to temp on same filesystem, then rename) so rebuilding while a server is running never truncates an active .so.
trap 'rm -f "$root/xtier/engine/lib"/.*.so.tmp.$$' EXIT
for so in "$build"/libxtier_*.so; do
  [ -f "$so" ] || continue
  base=$(basename "$so")
  tmp="$root/xtier/engine/lib/.${base}.tmp.$$"
  cp "$so" "$tmp"
  mv -f "$tmp" "$root/xtier/engine/lib/$base"
done
ls -la "$root/xtier/engine/lib"
