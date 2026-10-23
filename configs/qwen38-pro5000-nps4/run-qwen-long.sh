#!/usr/bin/env bash
# Qwen3.8 + XTier, long context: 512K tokens, YaRN x4 over the 256K native window, MTP. Extra args appended.
# FP8 KV cache and 24K prefill chunks. Production runs an NVFP4 KV cache with 32K chunks; the Qwen pack reads it too
# (packs/qwen4_exp/qsa_fp4.py, SGLang main's QSA sparse attention cannot). Append
#   --kv-cache-dtype nvfp4 --speculative-draft-kv-cache-dtype fp8_e4m3 --chunked-prefill-size 32768 --max-total-tokens 524288
# for 1.7 GB more free GPU memory after the CUDA graphs: 64K cold prefill 3% slower (no fused KV write + attention on an
# NVFP4 pool), 512K prefill and decode the same (docs/reports/2026-10-23-speculative-fp4kv.md). With FP8 KV, 32K chunks
# run out of GPU memory in the GDN linear attention; 24K chunks need the pack's PLE layer in blocks
# (packs/qwen4_exp/ple_blocks.py; SGLang's whole-chunk PLE layer peaks at about 5 GB there).
export LAYOUT=${LAYOUT:-prod} SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
exec "$(dirname "$0")/run-qwen-mtp.sh" --context-length 524288 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 24576 \
  --mem-fraction-static 0.90 --max-mamba-cache-size 16 --mamba-ssm-dtype bfloat16 \
  --json-model-override-args '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}' \
  "$@"
