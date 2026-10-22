#!/usr/bin/env bash
# Qwen3.8 + XTier, long context as in production: 512K tokens, NVFP4 KV cache, YaRN x4 over the 256K native window,
# MTP. Extra args appended.
export LAYOUT=${LAYOUT:-prod} SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
exec /data/fast/workspaces/xtier-dev/run-qwen-mtp.sh --context-length 524288 --kv-cache-dtype nvfp4 \
  --mem-fraction-static 0.90 --max-mamba-cache-size 16 --mamba-ssm-dtype bfloat16 \
  --json-model-override-args '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}' \
  "$@"
