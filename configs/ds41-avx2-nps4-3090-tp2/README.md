# DeepSeek V4.1 Flash 1017 on 2x RTX 3090 (SM86, TP2), Zen2 dual-socket, NPS4

`run-ds41.sh`: upstream SGLang main + XTier for two RTX 3090 (SM86, 24 GB each, PCIe only, no NVLink) on
2x EPYC 7532 (Zen2, 32 cores each, NPS4 -> 8 NUMA nodes, 503 GB host). Paths and options come from the environment
(see the script's header); the defaults assume this box's layout: the Python environment at `$XTIER_ROOT/env`, the
checkpoint at `/home/hpy/Models/DeepSeek-V4.1-Flash-1017`, the cache ranking profile at
`/home/hpy/lsglang_src/lkqmoe/lkqstrata-ds41-1017.json`.

Where things live, per card (24 GB):

- 26 hot expert slots per layer (0.35 GiB each, 9.0 MB per expert x 40 layers = 8.9 GiB) that grow to 41 while
  decode owns the memory and shrink again before a long prefill.
- The dense weights plus the FP8 KV cache of one 262144-token context.
- ~5.8 GB left when the pool and the graphs are up: enough for a 16384-token prefill chunk, not enough for DSpark.

Host memory: all layers' experts for the CPU engine, the Engram row cache (32 GiB) and the runtime. Data disk: the
checkpoint, including the 189 GB Engram table, read by rows (`XTIER_ENGRAM_ROWS=nvme`).

The GPU generation is handled in the code, not here: below SM100 the expert cache decodes FP4 in software, below
SM89 FP8 as well (`xtier/arch.py`, and `packs/deepseek_v4/wo_a.py`'s `_e4m3_fast` for the wo_a projection), and
below SM90 sparse MLA decode runs on the pack's split-key Triton kernels (`packs/deepseek_v4/sparse_mla.py`).
SGLang serves these MXFP4 experts with Marlin (`--moe-runner-backend marlin`) on SM86, which XTier's
`backend/formats.py` maps to `Fp8Fp4Experts`; the FP8 attention projections go through the pack's W8A16 path.

Measured on this box (2026-11-01, `bench/` style ASCII prompt through `/v1/chat/completions`, three runs):
decode **37.3 / 35.8 / 36.4 tok/s** (27.5 ms per token), prefill 1010-3233 tok/s depending on radix-cache hits, and
a 261073-token request in 157 s without an OOM. `available_gpu_mem` after the pool: 5.84 / 5.82 GB.

Decode is bound by the CPU engine, not by the GPUs: 6.6 of 40 layers' distinct experts miss the cache per window
(hit 0.53), each missing expert costs ~140 us (64 GB/s, 31% of one socket's 204 GB/s — Zen2 has no AVX-512-VBMI, so
MXFP4 nibbles are unpacked in scalar code), which is 17.4 ms of the 27.5 ms. The three switches that matter for
decode on this box are `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` (SGLang's own switch, environment only: all-reduce
67.4 -> 2.9 us p50), `XTIER_DS_WO_A_GB_R=4 XTIER_DS_WO_A_GB_H=512` (split-K GEMV for the decode shapes, bandwidth
535 -> 567, 481 -> 607, 423 -> 532, 396 -> 506 GB/s) and `XTIER_DS_WO_A_FASTDEC=1` (E4M3 decode by bit surgery,
44.7 -> 33.6 us); together they are worth ~25% against the same layout without them.  The last two belong to the
wo_a patches — on a tree that does not have them the two variables are inert, so this script can land first.

Notes specific to this configuration:

- `XTIER_THREADS=60`, not 64: four nodes x 15 workers + one core left for the dispatcher. 64 makes the dispatcher
  share an SMT sibling with a spinning worker and decode collapses to 2.2 tok/s.
- `XTIER_SPIN_COUNT=1048576` (the engine's compiled default is 32768): a decode layer is 0.69 ms, and 32 K pauses
  are only ~0.15 ms, so the workers would enter `futex` between every pair of layers.
- `XTIER_CACHE_ELASTIC_RESERVE_GIB=2.5`: at the default 2.0 the grower takes 8 extra slots, presses free memory
  below 1 GiB and triggers the protective all-release with its 600 s cooldown; the speed then oscillates 31.7/27.9.
- `SGLANG_DSV4_SWA_CHUNKS_IN_FLIGHT=1.2` and `SGLANG_TORCH_SCORE_BUDGET_MIB=256` are what keeps a 262144-token
  context alive on a 24 GB card: the first frees 0.33 GB of SWA pool, the second moves the indexer's scoring out of
  the allocator's failure path (a 1024 MiB chunk cannot be satisfied once the KV pool is resident).
- DSpark does not fit: with 26 slots plus the 262144-token KV pool there are 5.8 GB free, and the draft path wants
  >= 8 GB at that context. With DSpark the same box ran 23-27 tok/s; without it, 30-36.
- Launch from `/home/hpy`, not from the checkout: `xtier/` at the top of the checkout has no `__init__.py`, so a
  working directory inside it turns `import xtier` into a namespace package and the plugin silently fails to load
  (Engram then falls back to GPU tables and dies asking for 45.78 GiB).
