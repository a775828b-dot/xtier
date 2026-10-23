#!/usr/bin/env bash
# XTier e2e: SGLang with MiniMax M3.1 support (PR #45925 of sgl-project/sglang, not in main yet: checkout sglang-m31,
# venv env-m31 with its new dependencies and MiniMax's MSA kernels) + XTier on MiniMax-M3.1-Flash: 60 layers, the
# first 3 dense, 57 MoE of 128 routed experts (NVFP4 W4A4, top 4, GPT-OSS SwiGLU) and a shared MXFP8 expert; MSA
# sparse attention throughout; a DSpark draft of 5 dense layers (served by the vision-language wrapper only).
# Extra server arguments are appended ("$@"); XTIER_* settings come from the environment. Defaults: every routed
# expert on the CPU engine, 16 hot experts per layer on the GPU (XTIER_CACHE_SLOTS), a 128K KV pool (FP8: 16 slots
# and a 256K pool do not fit together; CONTEXT=262144 with about 8 slots), DSpark.
set -eo pipefail
D=/data/fast/workspaces/xtier-dev
M=/data/fast/nvme2/models/modelscope/MiniMax/MiniMax-M3.1-Flash
export CUDA_HOME=$D/env-m31/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export FMHA_SM100_ALLOW_JIT=1   # the MSA kernels compile at runtime on Blackwell (SGLang cookbook, MiniMax-M3.1)
export XTIER=${XTIER:-1} XTIER_THREADS=${XTIER_THREADS:-112} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
# engine and GPU prefill switches as the GLM-5.3 config (production's lkqmoe 0.5.6 settings for its NVFP4 experts)
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576} XTIER_DYNAMIC=${XTIER_DYNAMIC:-2} XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1}
export XTIER_GPU_PREFILL_CHUNK=${XTIER_GPU_PREFILL_CHUNK:-8192} XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-16}
export XTIER_PREFILL_OVERLAP=${XTIER_PREFILL_OVERLAP:-1} XTIER_PREFILL_PREFETCH=${XTIER_PREFILL_PREFETCH:-1}
export XTIER_PREFILL_PREFETCH_CHUNKS=${XTIER_PREFILL_PREFETCH_CHUNKS:-12} XTIER_PREFILL_RELEASE_WORKSPACE=${XTIER_PREFILL_RELEASE_WORKSPACE:-1}
export XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1} XTIER_PREFILL_UNPACK=${XTIER_PREFILL_UNPACK:-1}
export XTIER_PREFILL_TILE_M=${XTIER_PREFILL_TILE_M:-64} XTIER_GATE_BN=${XTIER_GATE_BN:-128} XTIER_GATE_LAUNCH=${XTIER_GATE_LAUNCH:-8,3}
export XTIER_DOWN_BN=${XTIER_DOWN_BN:-128} XTIER_DOWN_BK=${XTIER_DOWN_BK:-64} XTIER_DOWN_LAUNCH=${XTIER_DOWN_LAUNCH:-8,3}
export XTIER_DOWN_PRECISION=${XTIER_DOWN_PRECISION:-bf16}
# prefill tiers: MiniMax's experts (6144 x 3072) cost the CPU engine ~3x a GLM expert per token, so the experts split
# between the GPU and the CPU from 512 tokens (XTIER_SPLIT_PREFILL_MAX: up to 4K), above on the GPU
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-512}
export XTIER_HYBRID_PREFILL_MAX=${XTIER_HYBRID_PREFILL_MAX:-512} XTIER_SPLIT_PREFILL_MAX=${XTIER_SPLIT_PREFILL_MAX:-4096}
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-16} XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
SPEC=()
if [ "${DSPARK:-1}" = 1 ]; then
  export SGLANG_RAGGED_VERIFY_MODE=static   # M3.1 NVFP4 DSpark requires it (SGLang's MiniMax M3 argument checks)
  SPEC=(--speculative-algorithm DSPARK)
  [ -n "${GAMMA:-}" ] && SPEC+=(--speculative-dspark-block-size $GAMMA)   # draft tokens per step (checkpoint: 7)
fi
exec $D/env-m31/bin/python -m sglang.launch_server --model-path $M --served-model-name MiniMax-M3.1-Flash \
  --host 127.0.0.1 --port ${PORT:-18195} --tp-size 1 --trust-remote-code --skip-server-warmup \
  --max-running-requests 2 --cuda-graph-max-bs-decode 2 --context-length ${CONTEXT:-131072} \
  --max-total-tokens ${CONTEXT:-131072} --chunked-prefill-size ${BATCH:-16384} --mem-fraction-static ${MEM:-0.88} \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --reasoning-parser minimax-m3 --tool-call-parser minimax-m3 --enable-metrics --enable-cache-report \
  --weight-loader-drop-cache-after-load "${SPEC[@]}" "$@"
# --reasoning-parser / --tool-call-parser minimax-m3, --speculative-algorithm DSPARK (the draft loads from dspark/)
# and FMHA_SM100_ALLOW_JIT as in SGLang's MiniMax-M3.1 cookbook; --enable-cache-report and the CUDA graph backends
# (prefill eager: its MoE calls cannot go through XTier's capturable decode path) as the other XTier configs.
