#!/usr/bin/env bash
# Qwen3.8 + XTier with MTP like production (NEXTN 3 steps, 4 draft tokens, draft token map). Extra args appended.
# The draft MoE runs on XTier with every expert on the GPU (W4A16 like production; XTIER_DRAFT=0 hands it to SGLang).
export XTIER_DRAFT=${XTIER_DRAFT:-1}
exec "$(dirname "$0")/run-qwen-xtier.sh" --context-length 32768 --chunked-prefill-size 32768 --mem-fraction-static 0.85   --cuda-graph-max-bs-decode 2 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1   --speculative-num-draft-tokens 4 --speculative-draft-model-quantization modelopt_mixed   --speculative-token-map /data/fast/workspaces/lkqmoe-20260928/deploy-0.3.2/draft-token-map-49152.pt "$@"
