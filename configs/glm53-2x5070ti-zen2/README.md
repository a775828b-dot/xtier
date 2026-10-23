# GLM-5.3 Flash NVFP4 on 2x RTX 5070 Ti 16GB + 2x EPYC 7642 (Zen2, NPS4)

**Prepared, not run on this hardware yet** (2026-10-23). `run-glm-xtier-tp2.sh`: upstream SGLang main + XTier, TP=2
(one process per GPU, each with the I/2 slice of every expert on the NUMA nodes of its GPU's socket), AVX2 kernels,
4 hot slots per layer per GPU, the NextN draft's experts on the CPU engine (`XTIER_DRAFT_SLOTS=0`), 256K context.

The settings come from the lkqmoe 0.5.3 deployment on the same machine (2026-10-17: 64K TTFT 60.6 s, 256K TTFT
237.8 s, decode 33-34 tok/s). Bring-up checklist, VRAM budget and the environment pitfalls of that deployment:
`docs/deploy-zen2-dual-gpu.md`.
