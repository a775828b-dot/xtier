#!/usr/bin/env bash
# XTier e2e (step 2): upstream SGLang main + XTier on Qwen3.8 Flash Next. Extra server arguments are appended ("$@");
# XTIER_* settings come from the environment (defaults below). LAYOUT=prod: the expert cache laid out like production
# (its 25 GPU-resident layers fully cached, 155 hot slots on the other 23, the production profile).
set -eo pipefail
D=/data/fast/workspaces/xtier-dev
M=/data/fast/nvme2/models/huggingface/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4
# transformers 5.19 cannot read this checkpoint's tokenizer (no tokenizer.json): a converted copy, same encodings
T=$D/tokenizers/qwen38-flash-next
export CUDA_HOME=$D/env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1
export XTIER=${XTIER:-1} XTIER_THREADS=${XTIER_THREADS:-112} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576}
# GPU prefill as in production (lkqmoe 0.5.6 LKQMOE_PREFILL_*)
export XTIER_GPU_PREFILL_CHUNK=${XTIER_GPU_PREFILL_CHUNK:-8192} XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-32}
export XTIER_PREFILL_PREFETCH=${XTIER_PREFILL_PREFETCH:-1} XTIER_PREFILL_RELEASE_WORKSPACE=${XTIER_PREFILL_RELEASE_WORKSPACE:-1}
export XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1} XTIER_PREFILL_UNPACK=${XTIER_PREFILL_UNPACK:-1}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-1024}
if [ "${LAYOUT:-}" = prod ]; then
  export XTIER_CACHE_FULL=${XTIER_CACHE_FULL:-0,1,2,5,6,7,8,10,11,12,13,14,17,18,19,20,21,22,26,28,29,33,34,37,38}
  export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-155} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32} XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
  export XTIER_CACHE_PROFILE=${XTIER_CACHE_PROFILE:-/data/fast/workspaces/lkqmoe-20260928/deploy-0.3.2/lkqstrata-qwen38.json}
fi
exec $D/env/bin/python -m sglang.launch_server --model-path $M --tokenizer-path $T --served-model-name Qwen3.8-Flash-Next \
  --host 127.0.0.1 --port ${PORT:-18180} --tp-size 1 --trust-remote-code --skip-server-warmup --cuda-graph-backend-prefill disabled \
  --max-running-requests 2 --page-size 64 --reasoning-parser qwen3 --enable-metrics --weight-loader-drop-cache-after-load "$@"
# --weight-loader-drop-cache-after-load: evicts each checkpoint file once loaded (XTier makes it work for mapped files,
# xtier/platform.py); without it the page cache piles up on a few NUMA nodes, which then swap (4.3 GB once).
