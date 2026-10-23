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

- PLE n-gram table on NVMe instead of 47.7 GiB of pinned host memory: `XTIER_PLE_ROWS=nvme` (rows read from the
  checkpoint shards, 4 GiB row cache), `XTIER_PLE_MODEL=<dir>` for a copy of the shards elsewhere (a directory with
  `model.safetensors.index.json` and the `model-plefp8-*` files). On the test host the copy on the Optane P5800X
  (`/data/fast/p5800x1/qwen38-flash-next-ple`) matches the pinned table (verify step 14.14 vs 14.13 ms, cold prefill
  within 3%); from the DRAM-less EX900 the verify step is 7% slower and long prefills 16-48% slower.

Results of 2026-10-23: [docs/reports/2026-10-23-step2-qwen.md](../../docs/reports/2026-10-23-step2-qwen.md),
[docs/reports/2026-10-23-step3-ds-tables.md](../../docs/reports/2026-10-23-step3-ds-tables.md).
