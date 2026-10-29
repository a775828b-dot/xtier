#!/usr/bin/env bash
# Qwen3.8 Flash Next on one 24 GB GeForce GPU (RTX 4090 / RTX 3090) and ~120 GB of host memory: upstream SGLang main +
# XTier. The GPU holds the dense weights, an FP8 KV cache and N hot experts per layer; all 48 layers' experts sit in
# host memory for the CPU engine (63 GiB); the PLE n-gram table (48 GiB) stays in the checkpoint on the data disk and
# is read by rows. Extra server arguments are appended ("$@"); settings come from the environment:
#   DATA (data disk), XTIER_ENV (Python environment), MODEL (checkpoint), TOKENIZER (converted tokenizer, see README),
#   XTIER_FILES (cache ranking profile, MTP draft token map), PORT, CONTEXT (tokens), MTP=1 (speculative decoding)
set -eo pipefail
DATA=${DATA:-/root/autodl-tmp}
E=${XTIER_ENV:-$DATA/xtier-env}
M=${MODEL:-$DATA/models/qwen38}
T=${TOKENIZER:-$DATA/models/qwen38-tokenizer}
F=${XTIER_FILES:-$DATA/xtier-files}
export CUDA_HOME=${CUDA_HOME:-$E/lib/python3.12/site-packages/nvidia/cu13}
export PATH=$CUDA_HOME/bin:$E/bin:$PATH PYTHONUNBUFFERED=1   # $E/bin: ninja for SGLang's JIT kernels
# CPU engine threads: the CPUs this container may use (affinity, cgroup v2 / v1 quota) less one for the scheduler, a
# multiple of the NUMA nodes the engine spreads them over (the nodes with CPUs of ours)
if [ -z "${XTIER_THREADS:-}" ]; then
  XTIER_THREADS=$($E/bin/python "$(dirname "$0")/../../tools/cpu_budget.py")
fi
export XTIER=${XTIER:-1} XTIER_THREADS XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576}
# GPU prefill as on the RTX PRO 5000 (configs/qwen38-pro5000-nps4)
export XTIER_GPU_PREFILL_CHUNK=${XTIER_GPU_PREFILL_CHUNK:-8192} XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-32}
export XTIER_PREFILL_PREFETCH=${XTIER_PREFILL_PREFETCH:-1} XTIER_PREFILL_RELEASE_WORKSPACE=${XTIER_PREFILL_RELEASE_WORKSPACE:-1}
export XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1} XTIER_PREFILL_UNPACK=${XTIER_PREFILL_UNPACK:-1}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-1024}
# PLE rows from the checkpoint shards (no 48 GiB table in memory), a row cache in pinned host memory
export XTIER_PLE_ROWS=${XTIER_PLE_ROWS:-nvme} XTIER_PLE_CACHE_GIB=${XTIER_PLE_CACHE_GIB:-4}
# hot experts: N slots per layer, filled from the production ranking profile, swapped by use; 48 leave ~3.6 GB of GPU
# memory free after start (a 4K-token prompt-logprob request needs 1.9 GiB of logits alone), with MTP 40 (its
# draft holds all 512 experts of its MoE layer on the GPU)
SLOTS=48; if [ "${MTP:-0}" = 1 ]; then SLOTS=40; fi
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-$SLOTS} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32} XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
if [ -f "$F/lkqstrata-qwen38.json" ]; then export XTIER_CACHE_PROFILE=${XTIER_CACHE_PROFILE:-$F/lkqstrata-qwen38.json}; fi
CONTEXT=${CONTEXT:-131072}
SPEC=()
if [ "${MTP:-0}" = 1 ]; then   # as production: NEXTN 3 steps, 4 draft tokens, the draft's MoE on XTier (all on the GPU)
  export XTIER_DRAFT=${XTIER_DRAFT:-1}
  SPEC=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
        --speculative-draft-model-quantization modelopt_mixed --speculative-draft-kv-cache-dtype fp8_e4m3)
  if [ -f "$F/draft-token-map-49152.pt" ]; then SPEC+=(--speculative-token-map "$F/draft-token-map-49152.pt"); fi
fi
exec $E/bin/python -m sglang.launch_server --model-path $M --tokenizer-path $T --served-model-name Qwen3.8-Flash-Next \
  --host 127.0.0.1 --port ${PORT:-30000} --tp-size 1 --trust-remote-code --skip-server-warmup --cuda-graph-backend-prefill disabled \
  --max-running-requests 2 --cuda-graph-max-bs-decode 2 --page-size 64 --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --enable-cache-report --weight-loader-drop-cache-after-load \
  --context-length $CONTEXT --max-total-tokens $CONTEXT --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 8192 \
  --mamba-ssm-dtype bfloat16 --mem-fraction-static ${MEM_FRACTION:-0.88} "${SPEC[@]}" "$@"
# --kv-cache-dtype fp8_e4m3: 12 KiB per token (BF16 24 KiB); RTX 30 series read it through the pack's software E4M3
# decode (packs/qwen4_exp/qsa_fp4.py), RTX 40 series through SGLang's kernels. --max-total-tokens: one full context.
# --mamba-ssm-dtype bfloat16: as production (half the SSM state memory of the FP32 default).
