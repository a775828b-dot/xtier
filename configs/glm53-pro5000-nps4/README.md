# GLM-5.3 Flash NVFP4 on RTX PRO 5000 72GB + 2x EPYC 9334 (NPS4)

`run-glm-xtier.sh`: upstream SGLang main + XTier, 42 MoE layers on XTier with 60 hot slots each (production profile and
GPU prefill switches), 16K prefill chunks and BF16 SSM states as in production, a KV pool of one 512K context.
Arguments of the README measurement (2026-10-23, EAGLE with the NextN draft on XTier):

    XTIER_DRAFT=1 ./run-glm-xtier.sh --context-length 524288 --mem-fraction-static 0.92 \
      --max-mamba-cache-size 10 --cuda-graph-max-bs-decode 2 --dsa-decode-backend tilelang --dsa-prefill-backend tilelang \
      --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
      --speculative-adaptive --speculative-adaptive-config <lsglang_glm5.3>/launch/adaptive-steps-1to5.json \
      --speculative-token-map <lsglang_glm5.3>/lkqmoe/draft-token-map-49152.pt --speculative-use-rejection-sampling

- TileLang DSA backends: SGLang's FlashInfer sparse MLA decode does not support GLM-5.3's kpool (index_kpool > 1).
- `--speculative-use-rejection-sampling` (as production) needs the speculative pack: SGLang alone requires the draft vocab
  to equal the target vocab (token map 49152). The pack scatters the draft distribution into the target vocab, drafts at
  0.8 x the request temperature and truncates the verify at draft confidence 0.2 (`xtier/packs/speculative/`); the
  official-main row of the README runs without the flag.
