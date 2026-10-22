# Qwen3.8 Flash Next on RTX PRO 5000 72GB + 2x EPYC 9334 (NPS4, 8 NUMA nodes)

Development launchers of step 2 (paths of the test host; the declarative configuration replaces them later).

- `run-qwen-xtier.sh`: upstream SGLang main + XTier. `LAYOUT=prod` lays the expert cache out like the production
  pipeline (its 25 GPU-resident layers fully cached, 155 hot slots on the other 23, the production ranking profile).
  Notes: `--tokenizer-path` points to a converted tokenizer (transformers 5.19 cannot read the checkpoint's
  tokenizer without `tokenizer.json`; the converted copy encodes identically), `--skip-server-warmup` (the upstream
  warm-up request fails in the Qwen VL processor), `--cuda-graph-backend-prefill disabled` (XTier's prefill paths
  synchronize; decode graphs only).
- `run-qwen-mtp.sh`: the same with MTP as in production (NEXTN, 3 steps, 4 draft tokens, draft token map). The
  draft's MoE runs either on SGLang (`--speculative-moe-runner-backend flashinfer_cutlass`) or on XTier
  (`XTIER_DRAFT=1`, every expert on the GPU).

Results of 2026-10-23: [docs/reports/2026-10-23-step2-qwen.md](../../docs/reports/2026-10-23-step2-qwen.md).
