#!/usr/bin/env bash
# XTier e2e: upstream SGLang main + XTier, DeepSeek V4.1 Flash 1017 on 2x RTX 3090 (SM86, TP2) with
# 2x EPYC 7532 (Zen2, NPS4, 64 cores / 128 threads, 503 GB host). Extra server arguments are appended ("$@");
# every setting can be overridden from the environment.
#
# Base: configs/ds41-pro5000-nps4/run-ds-xtier.sh (2026-10-29, HEAD 788746a). Differences for this box:
#   * TP2 instead of TP1; no NVLink, so custom all-reduce stays off and PCIe-IPC carries the TP collectives
#     (SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1: 67.4 us NCCL-proxied all-reduce -> 2.9 us p50)
#   * SM86: no FP4/FP8 tensor cores -> software FP4 unpack + BF16 MMA on the GPU prefill, BF16 Down,
#     FlashInfer/TVM arch lists pinned to 8.6, the CPU engine built and loaded for AVX2 (Zen2 has no AVX-512)
#   * 24 GB cards: 26 hot slots per layer instead of 50, 262144 context instead of 524288
#   * Engram rows stay on NVMe (XTIER_ENGRAM_ROWS=nvme): 189 GB of table never enters RAM
#   * no speculative decoding (DSPARK=1 enables DSpark; it needs >= 8 GB of free VRAM at 262144, which these
#     two cards do not have once 26 slots + the KV pool are resident)
set -eo pipefail

R=${XTIER_ROOT:-/home/hpy/sglang-1030}
V=$R/env
M=${MODEL:-/home/hpy/Models/DeepSeek-V4.1-Flash-1017}
P=${PROFILE:-/home/hpy/lsglang_src/lkqmoe/lkqstrata-ds41-1017.json}

export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$V/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-$CUDA_HOME/lib:}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# SM86: Ampere. FlashInfer/TVM must never be told to build for Blackwell.
export FLASHINFER_CUDA_ARCH_LIST=8.6 TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=8.6
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export SGLANG_SKIP_P2P_CHECK=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_DSV41_REASONING_EFFORT=${SGLANG_DSV41_REASONING_EFFORT:-max}
# 30 x 32 block FP8 indexer scores: one 1024 MiB chunk cannot be allocated once the KV pool takes 262144 tokens.
# 256 MiB moves the allocator out of scoring.py's failure path on a 24 GB card.
export SGLANG_TORCH_SCORE_BUDGET_MIB=${SGLANG_TORCH_SCORE_BUDGET_MIB:-256}
# Pad the paged MQA logits with torch instead of FlashInfer: the FlashInfer path allocates per call.
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=${SGLANG_FP8_PAGED_MQA_LOGITS_TORCH:-1}
# TP2 over PCIe-IPC: 67.4 us (NCCL proxy) -> 2.9 us p50 for a decode all-reduce.
export SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=${SGLANG_ENABLE_PCIE_IPC_ALLREDUCE:-1}

# ---- XTier core ---------------------------------------------------------------------------------------------
export XTIER=${XTIER:-1}
export XTIER_ISA=${XTIER_ISA:-avx2}                 # Zen2: AVX2 engine library only
export XTIER_PACKS=${XTIER_PACKS:-deepseek_v4,speculative}
export XTIER_TP_NODES=${XTIER_TP_NODES:-auto} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_THREADS=${XTIER_THREADS:-60}           # 4 nodes x (15 workers + 1 core left for the dispatcher)
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576} XTIER_DYNAMIC=${XTIER_DYNAMIC:-2}
export XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1} XTIER_DOWN_GRAIN=${XTIER_DOWN_GRAIN:-24} XTIER_THP=${XTIER_THP:-0}
export XTIER_DECODE_MISS_GPU=${XTIER_DECODE_MISS_GPU:-0}

# ---- Engram on NVMe -----------------------------------------------------------------------------------------
export ENGRAM=${ENGRAM:-nvme}
ENGRAM_ARG=()
if [ "$ENGRAM" = "nvme" ]; then
  export XTIER_ENGRAM_ROWS=nvme XTIER_ENGRAM_MODEL=$M XTIER_ENGRAM_CACHE_GIB=${XTIER_ENGRAM_CACHE_GIB:-32}
else
  ENGRAM_ARG=(--engram-config '{"placement": "cpu", "replication": "none", "lookup": "auto"}')
fi

# ---- expert cache -------------------------------------------------------------------------------------------
# 26 base slots (0.35 GiB each = 9.1 GiB/card) + elastic growth to 41 while decode owns the memory, released
# before a long prefill. RESERVE 2.5 GiB keeps the grower from pressing free memory under 1 GiB, where the
# protective all-release + 600 s cooldown starts.
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-26} XTIER_CACHE_EVERY=${XTIER_CACHE_EVERY:-2}
export XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32} XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
if [ -f "$P" ]; then export XTIER_CACHE_PROFILE=${XTIER_CACHE_PROFILE:-$P}; fi
export XTIER_CACHE_PROMPT=${XTIER_CACHE_PROMPT:-1}
export XTIER_CACHE_ELASTIC=${XTIER_CACHE_ELASTIC:-1}
export XTIER_CACHE_ELASTIC_MAX_SLOTS=${XTIER_CACHE_ELASTIC_MAX_SLOTS:-41}
export XTIER_CACHE_ELASTIC_RESERVE_GIB=${XTIER_CACHE_ELASTIC_RESERVE_GIB:-2.5}

# ---- prefill: SM86 (no FP4 tensor cores) --------------------------------------------------------------------
export XTIER_PREFILL_FP4=${XTIER_PREFILL_FP4:-0} XTIER_PREFILL_UNPACK=${XTIER_PREFILL_UNPACK:-1}
export XTIER_PREFILL_PREFETCH=${XTIER_PREFILL_PREFETCH:-1} XTIER_PREFILL_OVERLAP=${XTIER_PREFILL_OVERLAP:-0}
export XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-8} XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-2048} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-512}
export XTIER_HYBRID_PREFILL_MAX=${XTIER_HYBRID_PREFILL_MAX:-0} XTIER_SPLIT_PREFILL_MAX=${XTIER_SPLIT_PREFILL_MAX:-0}
export XTIER_GPU_PREFILL_CHUNK=${XTIER_GPU_PREFILL_CHUNK:-8192}
export XTIER_DOWN_PRECISION=${XTIER_DOWN_PRECISION:-bf16} XTIER_PREFILL_TILE_M=${XTIER_PREFILL_TILE_M:-64}
export XTIER_GATE_BN=${XTIER_GATE_BN:-128} XTIER_GATE_LAUNCH=${XTIER_GATE_LAUNCH:-8,2}
export XTIER_DOWN_BN=${XTIER_DOWN_BN:-128} XTIER_DOWN_BK=${XTIER_DOWN_BK:-64} XTIER_DOWN_LAUNCH=${XTIER_DOWN_LAUNCH:-8,2}
# wo_a (the FP8 attention output projection): 4 rotating weight copies of 512 rows stay in L2, and the E4M3
# decode is done by bit surgery (packs/deepseek_v4/wo_a.py `_e4m3_fast`, bit-exact, 1.33x on sm_86).
export XTIER_DS_WO_A_GB_R=${XTIER_DS_WO_A_GB_R:-4} XTIER_DS_WO_A_GB_H=${XTIER_DS_WO_A_GB_H:-512}
export XTIER_DS_WO_A_FASTDEC=${XTIER_DS_WO_A_FASTDEC:-1}

# ---- context and memory -------------------------------------------------------------------------------------
# 1.2: swa_tokens = round(1.2 x 16384) + 3072 = 22784 instead of 35840, about 0.33 GB back. Below 1.0 the pool
# cannot hold one 16384-token chunk.
export SGLANG_DSV4_SWA_CHUNKS_IN_FLIGHT=${SGLANG_DSV4_SWA_CHUNKS_IN_FLIGHT:-1.2}

BATCH=${BATCH:-16384}
DRAFT=${DRAFT:-/home/hpy/Models/DeepSeek-V4.1-Flash-1017-dspark}
CG_DECODE=${CG_DECODE:-full}
SPEC=()
if [ "${DSPARK:-0}" = 1 ]; then
  export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static} XTIER_DRAFT=${XTIER_DRAFT:-1}
  SPEC=(--speculative-draft-model-path "$DRAFT" --speculative-algorithm DSPARK
        --speculative-dspark-block-size ${DSPARK_BLOCK:-5}
        --speculative-num-draft-tokens ${DSPARK_DRAFT:-6})
fi

exec $V/bin/python -m sglang.launch_server --model-path "$M" --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host ${HOST:-0.0.0.0} --port ${PORT:-39504} --tp-size ${TP:-2} --trust-remote-code \
  --max-running-requests ${MAX_REQUESTS:-1} --chunked-prefill-size $BATCH --max-prefill-tokens $BATCH \
  --context-length ${CONTEXT:-262144} --max-total-tokens ${CONTEXT:-262144} \
  --mem-fraction-static ${MEM:-0.90} \
  --moe-runner-backend marlin --disable-shared-experts-fusion $([ "${CUSTOM_AR:-0}" = 1 ] || echo --disable-custom-all-reduce) \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode $CG_DECODE --cuda-graph-max-bs-decode ${CG_BS:-8} \
  --enable-decoder-swa-bounded-replay --weight-loader-drop-cache-after-load \
  --model-loader-extra-config '{"enable_multithread_load":"true"}' \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
  --watchdog-timeout 1200 --enable-metrics --enable-cache-report \
  "${ENGRAM_ARG[@]}" "${SPEC[@]}" \
  ${SPS_TABLE:+--speculative-dspark-sps-table-path "$SPS_TABLE"} \
  ${STS_TABLE:+--speculative-dspark-confidence-sts-path "$STS_TABLE"} \
  ${ALIGN_TIER:+--speculative-dspark-align-verify-tokens-to-graph-tier} "$@"
# --moe-runner-backend marlin: on SM86 SGLang routes these MXFP4 experts to Mxfp4MarlinMoEMethod, which XTier's
# backend/formats.py maps to Fp8Fp4Experts (the parameters keep the checkpoint layout: Marlin's repack lives in
# process_weights_after_loading, which XTier replaces). Without that mapping the CPU engine refuses to start.
# --weight-loader-drop-cache-after-load: the page cache of the 300 GB checkpoint would otherwise push the engine's
# freshly built shards into swap. --disable-custom-all-reduce: two cards without NVLink; PCIe-IPC is the faster path.
