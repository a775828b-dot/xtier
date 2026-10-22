#!/usr/bin/env bash
# XTier e2e: upstream SGLang main + XTier on GLM-5.3 Flash NVFP4 (42 MoE layers, 288 experts). Extra server arguments
# are appended ("$@"); XTIER_* settings come from the environment (defaults: like the production pipeline, 60 hot
# slots per layer from its ranking profile, its GPU prefill switches).
set -eo pipefail
D=/data/fast/workspaces/xtier-dev
M=/data/fast/nvme1/models/huggingface/LibertAIDAI/GLM-5.3-Flash-NVFP4
export CUDA_HOME=$D/env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1
export XTIER=${XTIER:-1} XTIER_THREADS=${XTIER_THREADS:-112} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576} XTIER_DYNAMIC=${XTIER_DYNAMIC:-2} XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1}
# GPU prefill as in production (lkqmoe 0.5.6 LKQMOE_PREFILL_* of the GLM-5.3 pipeline)
export XTIER_GPU_PREFILL_CHUNK=${XTIER_GPU_PREFILL_CHUNK:-8192} XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-16}
export XTIER_PREFILL_OVERLAP=${XTIER_PREFILL_OVERLAP:-1} XTIER_PREFILL_PREFETCH=${XTIER_PREFILL_PREFETCH:-1}
export XTIER_PREFILL_PREFETCH_CHUNKS=${XTIER_PREFILL_PREFETCH_CHUNKS:-12} XTIER_PREFILL_RELEASE_WORKSPACE=${XTIER_PREFILL_RELEASE_WORKSPACE:-1}
export XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1} XTIER_PREFILL_UNPACK=${XTIER_PREFILL_UNPACK:-1}
export XTIER_PREFILL_TILE_M=${XTIER_PREFILL_TILE_M:-64} XTIER_GATE_BN=${XTIER_GATE_BN:-128} XTIER_GATE_LAUNCH=${XTIER_GATE_LAUNCH:-8,3}
export XTIER_DOWN_BN=${XTIER_DOWN_BN:-128} XTIER_DOWN_BK=${XTIER_DOWN_BK:-64} XTIER_DOWN_LAUNCH=${XTIER_DOWN_LAUNCH:-8,3}
export XTIER_DOWN_PRECISION=${XTIER_DOWN_PRECISION:-bf16}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-1024}
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-60} XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
export XTIER_CACHE_PROFILE=${XTIER_CACHE_PROFILE:-/data/fast/workspaces/gh-ref/lsglang_glm5.3/lkqmoe/lkqstrata-glm53.json}
exec $D/env/bin/python -m sglang.launch_server --model-path $M --served-model-name GLM-5.3-Flash-NVFP4 \
  --host 127.0.0.1 --port ${PORT:-18185} --tp-size 1 --trust-remote-code --skip-server-warmup --cuda-graph-backend-prefill disabled \
  --max-running-requests 2 --page-size 64 --reasoning-parser glm45 --enable-metrics --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 --dsa-decode-backend flashinfer_sparse_mla --dsa-prefill-backend flashinfer_sparse_mla \
  --disable-shared-experts-fusion "$@"
