#!/usr/bin/env bash
# XTier on Kimi K3.1 Pro (2.8T, 896 routed MXFP4 experts / layer, top-16, SiTU) on the RTX PRO 5000 72 GB host, an
# experiment (branch kimi-k31): SGLang kimi-k3.1-support (env-k31), every MoE layer on the CPU engine with hot slots on
# the GPU, 78% of each layer's experts only in the cold store (XTIER_COLD, fast load from the store; the checkpoint's
# expert bytes punched out, tools/cold_punch.py), the store on six drives (four PCIe 5.0 9100 PRO and two PCIe 4.0
# P5800X, every node part in three units, 5:5:5:5:2:2), the dense projections NVFP4 but the latent MoE projections
# (tools/kimi_mxfp4_dense.py: MODEL; with --latent ~4% faster for NLL +0.009), the kimi_k3 pack for SM120 (DSA on
# Triton), DFlash2 (RadixArk/Kimi-K3.1-Dflash2-Preview, block 5) and the route tail skip (XTIER_TAIL_SKIP=9: changes
# the output). Extra server arguments are appended ("$@").
# 11-05, code prompts with thinking off (tok/s): 3.73 without DFlash2 and the skip on the four-drive store; with
# DFlash2 on six drives 4.94 without the skip, 8.29 at R = 11, 9.36 at R = 10, 10.31 at R = 9; a 12K prompt about
# 91 tok/s (uploads and GPU bound, not the drives).
set -eo pipefail
D=/data/fast/workspaces/xtier-dev
M=${MODEL:-/data/fast/nvme1/models/moonshotai/Kimi-K3.1-Pro-NVD}
DRAFT=${DRAFT_MODEL:-/data/fast/nvme1/models/RadixArk/Kimi-K3.1-Dflash2-Preview}
export CUDA_HOME=$D/env-k31/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# the working directory off sys.path: started next to an SGLang checkout (the console starts its servers in xtier-dev),
# python -m makes sglang a namespace package of that directory and SGLang's __init__ never runs: no cache redirect
# (FlashInfer and Triton compile again) and, 11-05, about 1.2 GiB more GPU memory held, one elastic slot level fewer
export PYTHONSAFEPATH=1
export FLASHINFER_CUDA_ARCH_LIST=12.0 TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0
export XTIER=${XTIER:-1} XTIER_THREADS=${XTIER_THREADS:-112} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
# engine switches as for DeepSeek V4.1 (production lkqmoe 0.5.6 values)
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576} XTIER_DYNAMIC=${XTIER_DYNAMIC:-2} XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1}
export XTIER_DOWN_GRAIN=${XTIER_DOWN_GRAIN:-24} XTIER_THP=${XTIER_THP:-0}
# hot experts on the GPU: 4 fixed slots per layer, the rest of the memory to elastic slots during decode (a 2 GiB
# reserve: DFlash2's eager steps need room; no room kept for small prefills); a GPU hot expert is in RAM too, so more
# slots spare CPU work, not reads (4 -> 17 slots: +2.7%)
export XTIER_CACHE_SLOTS=${CACHE_SLOTS:-4} XTIER_CACHE_ELASTIC=${XTIER_CACHE_ELASTIC:-1}
export XTIER_CACHE_ELASTIC_MAX_SLOTS=${XTIER_CACHE_ELASTIC_MAX_SLOTS:-24} XTIER_CACHE_ELASTIC_SMALL=${XTIER_CACHE_ELASTIC_SMALL:-0}
export XTIER_CACHE_EVERY=${XTIER_CACHE_EVERY:-2} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32}
# cold experts: every expert in the store, COLD_FRACTION of each layer not in RAM; fast load from the store
export XTIER_COLD=${XTIER_COLD:-1} XTIER_COLD_FRACTION=${COLD_FRACTION:-0.78} XTIER_FAST_LOAD=${XTIER_FAST_LOAD:-1}
export XTIER_COLD_KEY=${XTIER_COLD_KEY:-Kimi-K3.1-Pro-E896-H3584-I3072-mxfp4-n8-u3}
export XTIER_COLD_PATHS=${XTIER_COLD_PATHS:-/data/fast/cold1/xtier-cold:/data/fast/cold2/xtier-cold:/data/fast/cold3/xtier-cold:/data/fast/cold4/xtier-cold:/data/fast/p5800x1/xtier-cold:/data/fast/p5800x2/xtier-cold}
export XTIER_COLD_UNITS=${XTIER_COLD_UNITS:-3} XTIER_COLD_PARTS=${XTIER_COLD_PARTS:-5:5:5:5:2:2}
# a decode step / verify window leaves out the experts outside RAM that every token ranks R-th or later (0 = the
# largest weight; xtier/cache/tail.py). 11-05, teacher-forced against no skip: R = 9 KL 0.024 (between Q5_K_M and
# Q4_K_M of llama.cpp's Llama 3 8B table), NLL +0.015; R = 11 KL 0.015, NLL +0.008; the copy test exact at 9, 10, 11
export XTIER_TAIL_SKIP=${XTIER_TAIL_SKIP:-9}
# prefill: CPU below 1024 tokens with cached routes on the GPU, GPU above (cold experts read per call)
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-2048} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-1024}
export XTIER_HYBRID_PREFILL_MAX=${XTIER_HYBRID_PREFILL_MAX:-1024} XTIER_SPLIT_PREFILL_MAX=${XTIER_SPLIT_PREFILL_MAX:-6144}
BATCH=${BATCH:-8192}
CONTEXT=${CONTEXT:-98304}
SPEC=()
if [ "${DFLASH:-1}" = 1 ]; then
  SPEC=(--speculative-algorithm DFLASH --speculative-draft-model-path $DRAFT --speculative-num-draft-tokens ${DRAFT_TOKENS:-5}
        --speculative-use-block-verification --enable-linear-replayssm-spec --speculative-draft-kv-cache-dtype fp8_e4m3
        --speculative-draft-attention-backend ${DRAFT_ATTN:-triton})
fi
exec $D/env-k31/bin/python -m sglang.launch_server --model-path $M --served-model-name ${SERVED:-Kimi-K3.1-Pro} \
  --host 127.0.0.1 --port ${PORT:-18200} --tp-size 1 --trust-remote-code --skip-server-warmup \
  --max-running-requests 2 --chunked-prefill-size $BATCH --max-prefill-tokens $BATCH \
  --context-length $CONTEXT --max-total-tokens $CONTEXT --mem-fraction-static ${MEM:-0.92} \
  --moe-runner-backend marlin --disable-shared-experts-fusion \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode ${DECODE_GRAPH:-full} \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --max-mamba-cache-size ${MAMBA_CACHE:-8} \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --watchdog-timeout 1800 --enable-metrics --enable-cache-report \
  --weight-loader-drop-cache-after-load --model-loader-extra-config '{"enable_multithread_load": false}' \
  "${SPEC[@]}" "$@"
# --moe-runner-backend marlin: fixes the unpadded MXFP4 expert layout the CPU engine takes (Mxfp4MoEMethod pads for
# some other runners). --weight-loader-drop-cache-after-load: the page cache of the 1.4 TB checkpoint would push the
# engine's shards into swap. enable_multithread_load false: SGLang reads 8 checkpoint files at once by default, i.e.
# 8 of Kimi's 15.7 GB expert layers in host memory before the first hand-off (11-05: 309 GiB used, no layer done);
# one file at a time XTier holds the RAM slots plus the layer being loaded. --mamba-ssm-dtype bfloat16 /
# --max-mamba-cache-size: KDA states (69 layers) as GLM-5.3; 8 states hold two requests and their prefixes.
# --kv-cache-dtype fp8_e4m3: SGLang's DSA pool has no NVFP4 layout. DFlash2: its draft's sliding-window MLA runs on the
# Triton backend only (DRAFT_ATTN), and its pools fit only with --mem-fraction-static 0.92 (MEM).
