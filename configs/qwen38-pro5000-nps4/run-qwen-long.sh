#!/usr/bin/env bash
# Qwen3.8 + XTier, long context: 512K tokens, YaRN x4 over the 256K native window, MTP. Extra args appended.
# NVFP4 KV cache as production (the Qwen pack reads it, packs/qwen4_exp/qsa_fp4.py; SGLang main's QSA cannot), draft
# KV FP8, a KV pool of one 512K context. Against FP8 KV this frees 1.8 GB of GPU memory; spent on layer 27 resident
# (the 26th: next by hot-slot coverage), which makes 64K / 512K cold prefill 1.4% / 0.5% faster (decode is
# GPU-bound: more GPU experts do not speed it up, 181 slots per layer did nothing; docs/reports/2026-10-24-qwen-kv-ds-agent.md).
# 24K prefill chunks: production's 32K were 3% slower at 64K here (no fused KV write + attention on an NVFP4 pool);
# they need the pack's PLE layer in blocks (packs/qwen4_exp/ple_blocks.py). Official main: --kv-cache-dtype fp8_e4m3
# --chunked-prefill-size 16384 and the production layout (XTIER_CACHE_FULL without 27).
export LAYOUT=${LAYOUT:-prod} SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export XTIER_CACHE_FULL=${XTIER_CACHE_FULL:-0,1,2,5,6,7,8,10,11,12,13,14,17,18,19,20,21,22,26,27,28,29,33,34,37,38}
exec "$(dirname "$0")/run-qwen-mtp.sh" --context-length 524288 --max-total-tokens 524288 --kv-cache-dtype nvfp4 \
  --speculative-draft-kv-cache-dtype fp8_e4m3 --chunked-prefill-size 24576 \
  --mem-fraction-static 0.90 --max-mamba-cache-size 16 --mamba-ssm-dtype bfloat16 \
  --json-model-override-args '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}' \
  "$@"
