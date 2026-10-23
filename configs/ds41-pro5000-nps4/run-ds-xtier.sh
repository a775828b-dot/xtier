#!/usr/bin/env bash
# XTier e2e: upstream SGLang main + XTier on DeepSeek V4.1 Flash 1017. Extra server arguments are appended ("$@");
# XTIER_* settings come from the environment. Defaults = the production 1017 layout: every MoE layer on the CPU
# engine with 50 hot slots (no fully resident layer), the production profile and switches, Engram rows on NVMe
# (packs/deepseek_v4) with a 4 GiB row cache. No speculative decoding yet (DSPARK=1 adds the production DSpark).
set -eo pipefail
D=/data/fast/workspaces/xtier-dev
M=/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
P=/data/fast/workspaces/ds41-1017
export CUDA_HOME=$D/env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export FLASHINFER_CUDA_ARCH_LIST=12.0 TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0
export SGLANG_DSV41_REASONING_EFFORT=${SGLANG_DSV41_REASONING_EFFORT:-max}
export XTIER=${XTIER:-1} XTIER_THREADS=${XTIER_THREADS:-112} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_PROBE_DISPATCH_CPU=${XTIER_PROBE_DISPATCH_CPU:-95}
# engine switches as in production (lkqmoe 0.5.6 LKQMOE_*)
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576} XTIER_DYNAMIC=${XTIER_DYNAMIC:-2} XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1}
export XTIER_DOWN_GRAIN=${XTIER_DOWN_GRAIN:-24} XTIER_THP=${XTIER_THP:-0}
# Engram rows on NVMe
export XTIER_ENGRAM_ROWS=${XTIER_ENGRAM_ROWS:-nvme} XTIER_ENGRAM_MODEL=$M XTIER_ENGRAM_CACHE_GIB=${XTIER_ENGRAM_CACHE_GIB:-4}
# expert cache: 50 slots on every layer (production 22528-token chunks), production profile and adaptation
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-50} XTIER_CACHE_EVERY=${XTIER_CACHE_EVERY:-2} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32}
export XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500} XTIER_CACHE_PROFILE=${XTIER_CACHE_PROFILE:-$P/deploy/lkqstrata-ds41-seed.json}
# prefill as in production: FP4 tensor cores, 6 prefetched chunks, 8-expert chunks with route accumulation, TF32
# Down and the production tiles. Production's lkqmoe runtime {"hybrid_prefill_max": 6144, "cpu_prefill_below": 1024}
# means: below 1024 tokens on the CPU, up to 6144 the experts split between the GPU and the CPU (here
# XTIER_SPLIT_PREFILL_MAX), above on the GPU. Below 1024 XTier runs the cached routes on the GPU and the others on the
# CPU (XTIER_HYBRID_PREFILL_MAX): XTier's cache hybrid puts every uncached route on the CPU, too slow for 4K chunks of
# DeepSeek's experts.
export XTIER_PREFILL_FP4=${XTIER_PREFILL_FP4:-1} XTIER_PREFILL_UNPACK=1 XTIER_PREFILL_PREFETCH=1
export XTIER_PREFILL_OVERLAP=${XTIER_PREFILL_OVERLAP:-1} XTIER_PREFILL_PREFETCH_CHUNKS=${XTIER_PREFILL_PREFETCH_CHUNKS:-6}
export XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-8} XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-2048} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-1024}
export XTIER_HYBRID_PREFILL_MAX=${XTIER_HYBRID_PREFILL_MAX:-1024} XTIER_SPLIT_PREFILL_MAX=${XTIER_SPLIT_PREFILL_MAX:-6144}
export XTIER_DOWN_PRECISION=tf32 XTIER_PREFILL_TILE_M=64 XTIER_GATE_BN=128 XTIER_GATE_LAUNCH=8,3 XTIER_DOWN_BN=128 \
  XTIER_DOWN_BK=64 XTIER_DOWN_LAUNCH=8,3
BATCH=${BATCH:-22528}
SPEC=()
if [ "${DSPARK:-0}" = 1 ]; then
  export SGLANG_RAGGED_VERIFY_MODE=static XTIER_DRAFT=${XTIER_DRAFT:-1}
  SPEC=(--speculative-draft-model-path $P/draft --speculative-algorithm DSPARK --speculative-dspark-block-size 5
        --speculative-num-draft-tokens 6 --speculative-dspark-sps-table-path $P/overlay/sps-table.json)
fi
exec $D/env/bin/python -m sglang.launch_server --model-path $M --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 127.0.0.1 --port ${PORT:-18190} --tp-size 1 --trust-remote-code --skip-server-warmup \
  --max-running-requests 2 --chunked-prefill-size $BATCH --max-prefill-tokens $BATCH \
  --context-length ${CONTEXT:-524288} --max-total-tokens ${CONTEXT:-524288} --mem-fraction-static ${MEM:-0.85} \
  --disable-shared-experts-fusion --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --enable-decoder-swa-bounded-replay --enable-deepseek-v4-fp4-indexer \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 --watchdog-timeout 1200 --enable-metrics \
  --weight-loader-drop-cache-after-load "${SPEC[@]}" "$@"
# --weight-loader-drop-cache-after-load (as in production): the page cache of the 300 GB checkpoint would otherwise
# push the engine's freshly built shards into swap on this 377 GB host
