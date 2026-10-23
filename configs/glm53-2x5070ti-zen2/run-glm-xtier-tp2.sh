#!/usr/bin/env bash
# XTier on GLM-5.3 Flash NVFP4 with 2x RTX 5070 Ti 16GB (SM120, TP=2) + 2x EPYC 7642 (Zen2: AVX2 kernels, NPS4 = 8
# NUMA nodes, GPU0 on node 2 and GPU1 on node 7). PREPARED, NOT RUN on this hardware yet: written 2026-10-23 from the
# lkqmoe 0.5.3 deployment of 2026-10-17 (README.md). Extra server arguments are appended ("$@").
#
# One process per TP rank: each holds the I/2 slice of every expert on the NUMA nodes of its GPU's socket
# (XTIER_TP_NODES=auto) and SGLang all-reduces the halves. Per GPU (GiB, estimated in 10-17): weights ~8, 256K KV
# ~2.35 (FP8 MLA, on both GPUs), mamba ~0.85, graphs / activations 2-2.5, so ~1.6-2.1 for hot experts (~0.28 per slot
# over 42 layers): start at 4 slots and raise after reading "avail mem" in the log. The NextN draft layer's experts
# stay on the CPU engine (XTIER_DRAFT_SLOTS=0; every expert on the GPU would cost 1.9 GiB per GPU).
set -eo pipefail
D=${D:-$HOME/xtier-dev}                       # upstream SGLang main (sglang/), its venv (env/) with XTier installed
M=${M:-$HOME/Models/GLM-5.3-Flash-NVFP4}      # LibertAIDAI/GLM-5.3-Flash-NVFP4 (nvidia's has a BF16 NextN layer)
PROFILE=${PROFILE:-$D/profiles/lkqstrata-glm53.json}
CTX=${CTX:-262144}                            # 256K, KV on both GPUs
CHUNK=${CHUNK:-5120}                          # prefill is PCIe 4.0 bound: every chunk uploads all experts (~80 GiB
                                              # per GPU), so its speed scales with the chunk (10-17: 5120 -> 1083 tok/s)
cuda=$D/env/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$cuda PATH=$cuda/bin:$PATH LD_LIBRARY_PATH=$cuda/lib:${LD_LIBRARY_PATH:-} PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo   # the wired NICs have no address on that host
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLASHINFER_CUDA_ARCH_LIST=12.0
export TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
# XTier per rank: 44 expert workers (4 nodes x 11 of 12 cores), the dispatcher and the other threads on the rank's
# remaining CPUs; the kernels pick AVX2 by themselves (XTIER_ISA=avx2 to be explicit)
export XTIER=1 XTIER_TP_NODES=${XTIER_TP_NODES:-auto} XTIER_THREADS=${XTIER_THREADS:-44} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576} XTIER_DYNAMIC=${XTIER_DYNAMIC:-2} XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1}
# GPU prefill on 16 GB: 4096-token blocks, 2-expert upload chunks, a ring of 2 prefetched chunks
export XTIER_GPU_PREFILL_CHUNK=${XTIER_GPU_PREFILL_CHUNK:-4096} XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-2}
export XTIER_PREFILL_PREFETCH_CHUNKS=${XTIER_PREFILL_PREFETCH_CHUNKS:-2} XTIER_PREFILL_OVERLAP=1 XTIER_PREFILL_PREFETCH=1
export XTIER_PREFILL_RELEASE_WORKSPACE=1 XTIER_PREFILL_ROUTE_ACCUM=1 XTIER_PREFILL_UNPACK=1
export XTIER_PREFILL_TILE_M=64 XTIER_GATE_BN=128 XTIER_GATE_LAUNCH=8,3 XTIER_DOWN_BN=128 XTIER_DOWN_BK=64 XTIER_DOWN_LAUNCH=8,3
export XTIER_DOWN_PRECISION=${XTIER_DOWN_PRECISION:-bf16}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-512}
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-4} XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500} XTIER_CACHE_PROFILE=$PROFILE
export XTIER_DRAFT=${XTIER_DRAFT:-1} XTIER_DRAFT_SLOTS=${XTIER_DRAFT_SLOTS:-0}
exec $D/env/bin/python -m sglang.launch_server --model-path $M --served-model-name GLM-5.3-Flash-NVFP4 \
  --host 127.0.0.1 --port ${PORT:-18085} --trust-remote-code --skip-server-warmup \
  --tp-size 2 --disable-custom-all-reduce \
  --context-length $CTX --max-total-tokens $((CTX + 4096)) --max-running-requests ${MAXREQ:-1} \
  --max-mamba-cache-size ${MAMBA:-6} --chunked-prefill-size $CHUNK --max-prefill-tokens $CHUNK \
  --mem-fraction-static ${MEMFRAC:-0.90} --page-size 64 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --dsa-decode-backend tilelang --dsa-prefill-backend tilelang --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode ${MAXREQ:-1} --disable-shared-experts-fusion \
  --reasoning-parser glm45 --tool-call-parser glm47 --watchdog-timeout 3600 --enable-metrics --enable-cache-report \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  "$@"
# --tool-call-parser, --enable-cache-report: the API as production (tool calls parsed; usage reports the cached prompt
# tokens, prompt_tokens_details.cached_tokens, which clients such as dsh show as cache hits).
