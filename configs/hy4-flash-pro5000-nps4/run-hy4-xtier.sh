#!/usr/bin/env bash
# XTier e2e: upstream SGLang main + XTier on Hunyuan-4-Flash NVFP4 (42 layers, 41 MoE x 512 experts
# top-16 on 1536 latent dim, 4 FP8 Engram tables on NVMe, 12 DSA + 30 SWA MLA layers with sinks).
# Target configuration: 1M KV pool, 1M context, 4 concurrent requests on one RTX PRO 5000 72 GB.
set -eo pipefail
D=/data/fast/workspaces/xtier-dev
M=${XTIER_MODEL_PATH:-/data/fast/nvme1/models/huggingface/atakans/Hy4-flash-NVFP4}
export CUDA_HOME=$D/env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export FLASHINFER_CUDA_ARCH_LIST=12.0 TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0
export XTIER=${XTIER:-1} XTIER_THREADS=${XTIER_THREADS:-112} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576} XTIER_DYNAMIC=${XTIER_DYNAMIC:-2} XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1}
# Hy4-Flash packs: Hybrid SWA+DSA pool (1M full + 128K window), 416 B/tok NVFP4 KV (7.95 GB at 1M),
# native FP8 IndexPool logits, and FP8 storage + W8A8 prefill for 208 BF16 attention/latent linears.
export XTIER_HY4_HYBRID_SWA=${XTIER_HY4_HYBRID_SWA:-1} XTIER_HY4_KV_FP4=${XTIER_HY4_KV_FP4:-1}
export XTIER_HY4_INDEXPOOL_FP8=${XTIER_HY4_INDEXPOOL_FP8:-1}
export XTIER_FP8_WEIGHTS=${XTIER_FP8_WEIGHTS:-1}
export XTIER_FP8_W8A8_PREFILL=${XTIER_FP8_W8A8_PREFILL:-'[.](self_attn|mlp)[.]'}
export SGLANG_OPT_RELEASE_PREFILL_SWA=${SGLANG_OPT_RELEASE_PREFILL_SWA:-1}
# Engram rows on NVMe (4 tables, 64.7 GB -> 1.32 GB scales in RAM + L1-weighted row cache)
export XTIER_ENGRAM_ROWS=${XTIER_ENGRAM_ROWS:-nvme} XTIER_ENGRAM_MODEL=$M XTIER_ENGRAM_CACHE_GIB=${XTIER_ENGRAM_CACHE_GIB:-32}
# Hot expert cache (0.152 GiB/slot across 41 MoE layers; elastic expands up to 288 during decode)
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-152} XTIER_CACHE_EVERY=${XTIER_CACHE_EVERY:-2} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32}
export XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
export XTIER_CACHE_ELASTIC=${XTIER_CACHE_ELASTIC:-1} XTIER_CACHE_ELASTIC_MAX_SLOTS=${XTIER_CACHE_ELASTIC_MAX_SLOTS:-288}
# GPU prefill switches (NVFP4 experts, 1536 x 1536) + CPU prefill share
export XTIER_PREFILL_FP4=${XTIER_PREFILL_FP4:-1} XTIER_PREFILL_UNPACK=1 XTIER_PREFILL_PREFETCH=1
export XTIER_PREFILL_OVERLAP=${XTIER_PREFILL_OVERLAP:-1} XTIER_PREFILL_PREFETCH_CHUNKS=${XTIER_PREFILL_PREFETCH_CHUNKS:-8}
export XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-16} XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1}
export XTIER_PREFILL_RELEASE_WORKSPACE=${XTIER_PREFILL_RELEASE_WORKSPACE:-1}
export XTIER_PREFILL_CPU_SHARE=${XTIER_PREFILL_CPU_SHARE:-1}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-640}
export XTIER_HYBRID_PREFILL_MAX=${XTIER_HYBRID_PREFILL_MAX:-640} XTIER_SPLIT_PREFILL_MAX=${XTIER_SPLIT_PREFILL_MAX:-4096}
export XTIER_DOWN_PRECISION=${XTIER_DOWN_PRECISION:-bf16} XTIER_PREFILL_TILE_M=${XTIER_PREFILL_TILE_M:-64}
export XTIER_GATE_BN=${XTIER_GATE_BN:-128} XTIER_GATE_LAUNCH=${XTIER_GATE_LAUNCH:-8,3}
export XTIER_DOWN_BN=${XTIER_DOWN_BN:-128} XTIER_DOWN_BK=${XTIER_DOWN_BK:-64} XTIER_DOWN_LAUNCH=${XTIER_DOWN_LAUNCH:-8,3}
BATCH=${BATCH:-49152}
DFLY=${DFLY:-1}
EXTRA=()
if [ "${XTIER_HY4_HYBRID_SWA}" != "0" ]; then
  EXTRA+=(--swa-full-tokens-ratio "${XTIER_HY4_SWA_RATIO:-0.125}")
fi
if [ "${DFLY}" != "0" ]; then
  export XTIER_DRAFT_LMHEAD_FP8=${XTIER_DRAFT_LMHEAD_FP8:-1}
  export XTIER_DSPARK_TRUNCATE=${XTIER_DSPARK_TRUNCATE:-0.20}
  EXTRA+=(--speculative-algorithm DFLY --speculative-num-draft-tokens "${DFLY_BLOCK_SIZE:-6}")
fi
exec $D/env/bin/python -m sglang.launch_server --model-path "$M" --served-model-name "${XTIER_SERVED_MODEL:-Hy4-flash-NVFP4}"   --host 127.0.0.1 --port ${PORT:-18192} --tp-size 1 --trust-remote-code --skip-server-warmup   --max-running-requests ${MAX_RUNNING_REQUESTS:-4} --chunked-prefill-size "$BATCH" --max-prefill-tokens "$BATCH"   --context-length ${CONTEXT:-1048576} --max-total-tokens ${KV_POOL:-1048576} --mem-fraction-static ${MEM:-0.88}   --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --page-size 64   --dsa-decode-backend flashinfer_sparse_mla --dsa-prefill-backend flashinfer_sparse_mla   --disable-shared-experts-fusion --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full   --reasoning-parser hunyuan --tool-call-parser hunyuan --watchdog-timeout 1200   --enable-metrics --enable-cache-report --weight-loader-drop-cache-after-load "${EXTRA[@]}" "$@"
