#!/usr/bin/env bash
# XTier on DeepSeek-V4.1-Flash-1017, TP=2 across two L20 (SM89, 48 GB) cards that sit on
# different NUMA nodes. Extra server arguments are appended ("$@"); XTIER_* settings come
# from the environment. Every knob below is overridable, e.g. CHUNK=16384 bash run-ds-xtier.sh.
#
# Differences from the SM120 production config (configs/ds41-pro5000-nps4/run-ds-xtier.sh):
#   1. FLASHINFER/TVM arch lists are 8.9, not 12.0;
#   2. XTIER_PREFILL_FP4=0: Ada has no FP4 tensor cores, the experts are decoded in software
#      (xtier/arch.py);
#   3. no --enable-deepseek-v4-fp4-indexer (that indexer refuses SM89);
#   4. TP=2 with one rank per NUMA node, XTIER_TP_NODES written out explicitly (see below);
#   5. XTIER_THREADS=32 per rank: one worker per physical core of its own socket;
#   6. XTIER_SPIN_COUNT=32768 instead of 1048576, so 64 idle workers do not spin on the box.
set -eo pipefail
M=${M:?set M to the DeepSeek-V4.1-Flash-1017 checkpoint directory}
PY=${PY:-python3}                      # the interpreter that has SGLang installed
PORT=${PORT:-8001}

# ---- SM89: the architecture lists must say 8.9 ----
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export FLASHINFER_CUDA_ARCH_LIST=8.9 TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=8.9
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1
# Pin the rustup toolchain when the SGLang tree ships a rust-toolchain.toml that asks for more
# components than the local install has: rustup then downloads them in the background and the
# Rust image-processor probe stalls on it for tens of seconds to minutes. Pinned, the probe
# fails fast and falls back to PIL. Empty = leave rustup alone.
if [ -n "${RUSTUP_TOOLCHAIN_PIN:-}" ]; then export RUSTUP_TOOLCHAIN=$RUSTUP_TOOLCHAIN_PIN; fi

# ---- one card per NUMA node ----
# The C engine picks its worker CPUs with xtier_worker_cpus() (csrc/engine/engine.cpp), which has
# no rank offset: two ranks on one node get the same CPU set (29/29 identical, CPU time per layer
# 625 us -> 8812 us). XTIER_TP_NODES=auto splits the nodes over the ranks, which only lines up when
# the cards sit on different sockets; here they do, so write it out.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
export XTIER_TP_NODES=${XTIER_TP_NODES:-"0|1"}
# Different sockets means no P2P: keep the NCCL transfers on the host bridge.
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}

# ---- DSpark speculative decoding: the draft layer (mtp.*) is in the main checkpoint ----
SPEC=()
if [ "${DSPARK:-0}" = 1 ]; then
  export XTIER_DRAFT=1
  # compact = variable-length verification, rows_per_request becomes 1 and SGLang plans the
  # indexer metadata itself. static (the default) keeps 6 rows and calls
  # deep_gemm.get_paged_mqa_logits_metadata, which returns None on SM89.
  export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-compact}
  SPEC=(--speculative-draft-model-path "$M"
        --speculative-algorithm DSPARK
        --speculative-dspark-block-size 5
        --speculative-num-draft-tokens 6)
  if [ -n "${SPS_TABLE:-}" ]; then SPEC+=(--speculative-dspark-sps-table-path "$SPS_TABLE"); fi
fi

# ---- engine ----
export XTIER=1
export XTIER_THREADS=${XTIER_THREADS:-32}   # one node per rank: 32 physical cores, one worker each
# auto = "the rank's node CPUs minus the workers and the dispatcher", i.e. the Python, scheduler
# and torch threads are moved off the expert cores.
export XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
# 43 resident slots = CHUNK_SLOTS['32768'], the floor while elastic is off; elastic then grows to
# 64. 80 slots eat the last ~5 MB of VRAM, the allocator OOMs and falls back, prefill 2.8K -> 600.
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-43}
# Swap every 2 decode windows, 32 swaps per round. A swap reads experts from host memory and
# competes with the CPU engine for bandwidth, so fewer is better.
export XTIER_CACHE_EVERY=${XTIER_CACHE_EVERY:-2} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32} XTIER_CACHE_LOG=200
# Promotion threshold 3 / decay 0.9 (source defaults 1.5 / 0.7): higher bar, forgets slower, so
# fewer swaps. Measured swaps/round 111 -> 63, hit rate 0.654 -> 0.671, CPU time per layer 1023 ->
# 985 us, throughput unchanged or better.
export XTIER_CACHE_MIN_GAIN=${XTIER_CACHE_MIN_GAIN:-3} XTIER_CACHE_DECAY=${XTIER_CACHE_DECAY:-0.9}
# Agent workloads: warm the slots from prefill routes, borrow prefill VRAM for extra hot slots and
# adapt faster. 64 swaps x 8 rounds carries the same volume as 256 x 4 with smaller bursts.
export XTIER_CACHE_PROMPT=1 XTIER_CACHE_PROMPT_WINDOWS=10
export XTIER_CACHE_PROMPT_ROUNDS=${XTIER_CACHE_PROMPT_ROUNDS:-8} XTIER_CACHE_PROMPT_SWAPS=${XTIER_CACHE_PROMPT_SWAPS:-64}
# Elastic: base 0.5 keeps 21 slots resident, the ceiling is 64 = 21 resident + 43 elastic.
# 43 -> 64 measured hit rate 62% -> 74%, throughput +33%, rounds over 200 ms 49% -> 4%, and 5.0 GB
# left at full growth against a 2 GB reserve: no OOM.
export XTIER_CACHE_ELASTIC=${XTIER_CACHE_ELASTIC:-1} XTIER_CACHE_ELASTIC_RESERVE_GIB=2 XTIER_CACHE_ELASTIC_CRITICAL_GIB=1
export XTIER_CACHE_ELASTIC_BASE=${XTIER_CACHE_ELASTIC_BASE:-0.5} XTIER_CACHE_ELASTIC_MAX_SLOTS=${XTIER_CACHE_ELASTIC_MAX_SLOTS:-64}
export XTIER_CACHE_ADAPT=1
# Optional routing trace of this box's traffic: it does not use VRAM, it only decides which experts
# the hot slots start with. Leave empty to start from the source default.
if [ -n "${XTIER_CACHE_PROFILE:-}" ]; then export XTIER_CACHE_PROFILE; fi
export XTIER_THP=0                          # required, otherwise locked memory eats the huge pages
# Idle workers spin XTIER_SPIN_COUNT times before sleeping on the futex. The SM120 production value
# is 1048576; with 64 workers here that is a permanent busy wait (~80% of the box). 32768 is the
# source default: throughput unchanged, CPU load 80% -> 1.7-39%, which matters because the box runs
# other services.
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-32768}
export XTIER_DYNAMIC=2 XTIER_DOWN_BF16=1 XTIER_DOWN_GRAIN=24

# ---- Engram table (189 GB) on NVMe, read by row ----
export XTIER_ENGRAM_ROWS=nvme XTIER_ENGRAM_MODEL=$M XTIER_ENGRAM_CACHE_GIB=${XTIER_ENGRAM_CACHE_GIB:-4}

# ---- prefill: no FP4 tensor cores on Ada ----
export XTIER_PREFILL_FP4=0 XTIER_PREFILL_UNPACK=1 XTIER_PREFILL_PREFETCH=1
export XTIER_PREFILL_OVERLAP=1 XTIER_PREFILL_PREFETCH_CHUNKS=6
export XTIER_PREFILL_EXPERT_CHUNK=8 XTIER_PREFILL_ROUTE_ACCUM=1
export XTIER_CPU_PREFILL_BATCH=2048
# An agent turn is only 700-1000 tokens: the source threshold of 1024 pushed those to the CPU,
# where they competed with decode for the same cores.
export XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-64}
export XTIER_HYBRID_PREFILL_MAX=0           # no "uncached routes go to the CPU" hybrid path
export XTIER_SPLIT_PREFILL_MAX=${XTIER_SPLIT_PREFILL_MAX:-6144}
export XTIER_DOWN_PRECISION=tf32 XTIER_PREFILL_TILE_M=64
# The SM120 128-wide prefill tile does not fit in the 99 KiB of shared memory an SM89 SM has; these
# are the L20-safe tiles (TILES_PROD=0 keeps the source defaults).
if [ "${TILES_PROD:-0}" = 1 ]; then
  export XTIER_GATE_BN=${XTIER_GATE_BN:-128} XTIER_GATE_LAUNCH=${XTIER_GATE_LAUNCH:-8,3}
  export XTIER_DOWN_BN=${XTIER_DOWN_BN:-128} XTIER_DOWN_BK=${XTIER_DOWN_BK:-64} XTIER_DOWN_LAUNCH=${XTIER_DOWN_LAUNCH:-8,3}
fi

# ---- server ----
# No "#" comment lines inside the command below: bash treats a comment between two backslash-continued
# lines as "comment to the end of the logical line" and swallows every argument after it (this cost the
# config its --disable-custom-all-reduce once). That is also why the argument list is an array.
# chunk 32K: every prefill chunk re-uploads all 384 experts of all 40 layers (3.36 GiB/layer/card), so a
# larger chunk means fewer upload rounds. Cold 100K prefill measured 8K=1597, 16K=2498, 32K=3028 tok/s.
# PDI 32: how many decode rounds run before the next prefill is scheduled (0 = no interleave). With 0
# several long prefills run back to back and ready requests get no token at all (116 s max gap on 6 x
# 100K); 32 brings that to 330 ms and +27% on 8 x 4 agents at the same cold prefill, 64 pushes the TTFT
# p95 of new requests to 16 s.
# fcfs, not lpm: lpm sorts by longest prefix and keeps pushing a queue-tail request's second prefill
# back (216 s -> 117 s wall, +80% throughput on 12 parallel agents, no starvation).
# max-running-requests 8 is this box's hard ceiling: above 8 the CUDA graph cannot replay (only bs 1..8
# are captured), compact verify degrades to static, the JIT indexer planner is skipped and
# deep_gemm.get_paged_mqa_logits_metadata returns None on SM89 -> assert.
ARGS=(
  --model-path "$M"
  --served-model-name DeepSeek-V4.1-Flash
  --host 127.0.0.1 --port "$PORT"
  --tp-size ${TP_SIZE:-2}
  --trust-remote-code
  --prefill-decode-interval ${PDI:-32} --schedule-policy ${SCHEDULE:-fcfs}
  --skip-server-warmup
  --max-running-requests ${MAX_REQ:-8}
  --chunked-prefill-size ${CHUNK:-32768} --max-prefill-tokens ${PREFILL_TOKENS:-32768}
  --prefill-max-requests ${PREFILL_REQ:-8} --cuda-graph-max-bs-decode ${GRAPH_BS:-8}
  --context-length ${CONTEXT:-262144} --max-total-tokens ${KV_POOL:-1572864}
  --mem-fraction-static ${MEM:-0.88}
  --disable-shared-experts-fusion
  --disable-custom-all-reduce
  --cuda-graph-backend-prefill disabled
  --cuda-graph-backend-decode full
  --enable-decoder-swa-bounded-replay
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
  --watchdog-timeout 1200 --enable-metrics --enable-cache-report
  --enable-hierarchical-cache --hicache-ratio 2
  --weight-loader-drop-cache-after-load
)
if [ -n "${MAMBA_CACHE:-}" ]; then ARGS+=(--max-mamba-cache-size "$MAMBA_CACHE"); fi
if [ "${#SPEC[@]}" -gt 0 ]; then ARGS+=("${SPEC[@]}"); fi

if [ "${DRYRUN:-0}" = 1 ]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi

exec "$PY" -m sglang.launch_server "${ARGS[@]}" "$@"
