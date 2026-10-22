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
cp "$build"/libxtier_*.so "$root/xtier/engine/lib/"
ls -la "$root/xtier/engine/lib"
