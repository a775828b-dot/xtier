#!/usr/bin/env bash
# Qwen3.8 + XTier, long context: 512K tokens, YaRN x4 over the 256K native window, MTP. Extra args appended.
# FP8 KV cache and 24K prefill chunks, not production's NVFP4 KV and 32K chunks: SGLang main's QSA sparse attention
# cannot read an NVFP4 KV cache (its extraction kernel hands the FP4 tensor to Triton as is), and without the ~3 GB
# that NVFP4 saves, 32K chunks run out of GPU memory in the GDN linear attention. 24K chunks need the Qwen pack's PLE
# layer in blocks (packs/qwen4_exp/ple_blocks.py; SGLang's whole-chunk PLE layer peaks at about 5 GB there).
export LAYOUT=${LAYOUT:-prod} SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
exec "$(dirname "$0")/run-qwen-mtp.sh" --context-length 524288 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 24576 \
  --mem-fraction-static 0.90 --max-mamba-cache-size 16 --mamba-ssm-dtype bfloat16 \
  --json-model-override-args '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}' \
  "$@"
