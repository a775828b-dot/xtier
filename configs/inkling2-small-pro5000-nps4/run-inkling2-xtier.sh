#!/usr/bin/env bash
# XTier e2e: upstream SGLang main + XTier on Thinking Machines Inkling-2-Small (594B / 17B active, 40 NVFP4 + 8 MXFP8
# MoE layers, 40 SWA + 8 global attention, native 5-layer DSpark MTP, multimodal ViT).
# Default configuration: Plan A+ Squeezed (2-conc, 2% NVMe cold tail [376/384 experts in RAM, >=12 GiB avail RAM] + real
# seed profile, elastic base=1 reserve=0.8G critical=0.5G -> 45 hot slots/layer at 70.65/73.4 GiB VRAM, A1 NVFP4 4over6
# MAE + MXFP8 cuDNN prefill, A2 NVFP4 target KV, A3 116-layer FP8 + W8A8 prefill, Recall + DSpark
# OnlineAcceptanceEstimator truncate=0.20, 64K single-chunk BATCH=65664).
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=/data/fast/workspaces/xtier-dev
M=${XTIER_MODEL_PATH:-/data/fast/cold1/models/Inkling-2-Small}
export CUDA_HOME=$D/env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export CUDNN_FRONTEND_CUDART_LIB_NAME=${CUDNN_FRONTEND_CUDART_LIB_NAME:-libcudart.so.13}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export FLASHINFER_CUDA_ARCH_LIST=12.0 TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0
export XTIER=${XTIER:-1} XTIER_THREADS=${XTIER_THREADS:-112} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_PROBE_DISPATCH_CPU=${XTIER_PROBE_DISPATCH_CPU:-95}
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576} XTIER_DYNAMIC=${XTIER_DYNAMIC:-2} XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1}
export XTIER_DOWN_GRAIN=${XTIER_DOWN_GRAIN:-24} XTIER_THP=${XTIER_THP:-0}
export XTIER_COLD=${XTIER_COLD:-1} XTIER_COLD_FRACTION=${XTIER_COLD_FRACTION:-0.02}
export XTIER_COLD_PATHS=${XTIER_COLD_PATHS:-/data/fast/cold1/xtier-cold:/data/fast/cold2/xtier-cold:/data/fast/cold3/xtier-cold:/data/fast/cold4/xtier-cold}
export XTIER_FAST_LOAD=${XTIER_FAST_LOAD:-1}
if [ -z "${XTIER_CACHE_PROFILE+x}" ] && [ -f "$HERE/inkling2-seed-profile.json" ]; then
  export XTIER_CACHE_PROFILE="$HERE/inkling2-seed-profile.json"
fi
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-32} XTIER_CACHE_EVERY=${XTIER_CACHE_EVERY:-1} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-64}
export XTIER_CACHE_ELASTIC=${XTIER_CACHE_ELASTIC:-1} XTIER_CACHE_ELASTIC_BASE=${XTIER_CACHE_ELASTIC_BASE:-0.03125}
export XTIER_CACHE_ELASTIC_MAX_SLOTS=${XTIER_CACHE_ELASTIC_MAX_SLOTS:-96}
export XTIER_CACHE_ELASTIC_RESERVE_GIB=${XTIER_CACHE_ELASTIC_RESERVE_GIB:-0.8}
export XTIER_CACHE_ELASTIC_CRITICAL_GIB=${XTIER_CACHE_ELASTIC_CRITICAL_GIB:-0.5}
export XTIER_CACHE_ELASTIC_IDLE=${XTIER_CACHE_ELASTIC_IDLE:-4}
export XTIER_CACHE_ELASTIC_IDLE_CHUNK=${XTIER_CACHE_ELASTIC_IDLE_CHUNK:-4}
export XTIER_CACHE_ELASTIC_CHECK=${XTIER_CACHE_ELASTIC_CHECK:-4}
export XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
export XTIER_PREFILL_UNPACK=1 XTIER_PREFILL_PREFETCH=${XTIER_PREFILL_PREFETCH:-1}
export XTIER_PREFILL_OVERLAP=${XTIER_PREFILL_OVERLAP:-1} XTIER_PREFILL_PREFETCH_CHUNKS=${XTIER_PREFILL_PREFETCH_CHUNKS:-6}
export XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-8} XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-2048} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-1024}
export XTIER_HYBRID_PREFILL_MAX=${XTIER_HYBRID_PREFILL_MAX:-1024}
export XTIER_DOWN_PRECISION=${XTIER_DOWN_PRECISION:-tf32} XTIER_PREFILL_TILE_M=${XTIER_PREFILL_TILE_M:-64} \
  XTIER_GATE_BN=${XTIER_GATE_BN:-128} XTIER_GATE_LAUNCH=${XTIER_GATE_LAUNCH:-8,3} \
  XTIER_DOWN_BN=${XTIER_DOWN_BN:-128} XTIER_DOWN_BK=${XTIER_DOWN_BK:-64} XTIER_DOWN_LAUNCH=${XTIER_DOWN_LAUNCH:-8,3}
export XTIER_PREFILL_FP4=${XTIER_PREFILL_FP4:-1} XTIER_PREFILL_NVFP4=${XTIER_PREFILL_NVFP4:-1} \
  XTIER_PREFILL_NVFP4_4OVER6=${XTIER_PREFILL_NVFP4_4OVER6:-1} XTIER_PREFILL_MXFP8=${XTIER_PREFILL_MXFP8:-1} \
  XTIER_PREFILL_FP4_SLOTS=${XTIER_PREFILL_FP4_SLOTS:-6} XTIER_PREFILL_MXFP8_SLOTS=${XTIER_PREFILL_MXFP8_SLOTS:-4}
export XTIER_INKLING2_KV_FP4=${XTIER_INKLING2_KV_FP4:-1}
export XTIER_FP8_WEIGHTS=${XTIER_FP8_WEIGHTS:-1}
export XTIER_FP8_INCLUDE=${XTIER_FP8_INCLUDE:-qkvr|wo_ud|qkv_proj|o_proj|gate_up_proj|down_proj}
export XTIER_FP8_W8A8_PREFILL=${XTIER_FP8_W8A8_PREFILL:-.*}
export XTIER_RECALL=${XTIER_RECALL:-1}
export XTIER_DSPARK_TRUNCATE=${XTIER_DSPARK_TRUNCATE:-0.20}
export XTIER_DRAFT_LMHEAD_FP8=${XTIER_DRAFT_LMHEAD_FP8:-1}
export XTIER_INKLING2_MARKOV_PAD2=${XTIER_INKLING2_MARKOV_PAD2:-1}
BATCH=${BATCH:-65664}
SWA_RATIO=${SWA_RATIO:-0.14}
EXTRA=()
if [ "${MULTIMODAL:-1}" = 1 ]; then
  EXTRA+=(--enable-multimodal)
fi
if [ "${DSPARK:-1}" = 1 ]; then
  EXTRA+=(--speculative-algorithm DSPARK)
fi
exec $D/env/bin/python -m sglang.launch_server --model-path "$M" --served-model-name "${XTIER_SERVED_MODEL:-Inkling-2-Small}" \
  --host 127.0.0.1 --port ${PORT:-18195} --tp-size 1 --trust-remote-code --skip-server-warmup \
  --max-running-requests ${MAX_RUNNING_REQUESTS:-2} --chunked-prefill-size "$BATCH" --max-prefill-tokens "$BATCH" \
  --context-length ${CONTEXT:-524288} --max-total-tokens ${KV_POOL:-${CONTEXT:-524288}} --mem-fraction-static ${MEM:-0.86} \
  --swa-full-tokens-ratio "$SWA_RATIO" --attention-backend fa4 --dtype bfloat16 --page-size 128 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --reasoning-parser inkling --tool-call-parser inkling2 \
  --watchdog-timeout 1200 --enable-metrics --enable-cache-report \
  --weight-loader-drop-cache-after-load "${EXTRA[@]}" "$@"
