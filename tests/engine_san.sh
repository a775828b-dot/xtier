#!/usr/bin/env bash
# The CPU engine and the cold store under a sanitizer, without a GPU: csrc/ built with -fsanitize=KIND (the engine and the
# cold store; the mailbox needs CUDA and is left out) into build/san-KIND, then tests/engine_san.cpp against it, its store
# on DIR (a filesystem with O_DIRECT). Passes with ENGINE-SAN PASS and no sanitizer report (2026-11-10: thread and
# address both clean, about 30 s to build and 10 s to run with 16 threads).
#   tests/engine_san.sh thread|address DIR [THREADS=16] [ITERATIONS=300]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
kind=${1:-}; dir=${2:-}; threads=${3:-16}; iterations=${4:-300}
case $kind in
  thread) flags="-fsanitize=thread" ;;
  address) flags="-fsanitize=address,undefined" ;;
  *) echo "usage: $0 thread|address DIR [THREADS] [ITERATIONS]" >&2; exit 2 ;;
esac
[ -n "$dir" ] || { echo "usage: $0 thread|address DIR [THREADS] [ITERATIONS]" >&2; exit 2; }
flags="$flags -g -fno-omit-frame-pointer"
build=$root/build/san-$kind
mkdir -p "$build" "$dir"
cmake -S "$root/csrc" -B "$build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="$flags"       -DCMAKE_SHARED_LINKER_FLAGS="$flags" > "$build/build.log" 2>&1
cmake --build "$build" -j "${XTIER_BUILD_JOBS:-8}" >> "$build/build.log" 2>&1
g++ -std=c++17 -O1 $flags -I"$root/csrc/include" "$root/tests/engine_san.cpp" -o "$build/engine_san"     -L"$build" -lxtier_engine -Wl,-rpath,"$build" -lpthread
run=("$build/engine_san" "$threads" "$dir" "$iterations")
# ThreadSanitizer needs a layout of the address space it knows; newer kernels randomize more than it accepts
[ "$kind" = thread ] && run=(setarch "$(uname -m)" -R "${run[@]}")
"${run[@]}" > "$build/run.log" 2>&1 || true
reports=$(grep -cE "WARNING: ThreadSanitizer|ERROR: AddressSanitizer|runtime error:" "$build/run.log" || true)
grep -E "ENGINE-SAN|FAIL" "$build/run.log" || true
echo "engine_san $kind: $reports sanitizer reports (log: $build/run.log)"
[ "$reports" = 0 ] && grep -q "ENGINE-SAN PASS" "$build/run.log"
