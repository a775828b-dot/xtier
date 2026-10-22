# csrc

The CPU expert engine and the CPU-GPU mailbox, ported from lkqmoe 0.5.6 (commit `93edcc1`) by
[tools/port_engine.py](../tools/port_engine.py): identifiers, strings and comments renamed only.

| File | What |
|---|---|
| `engine/engine.cpp` | MoE experts on the CPU, generic over hidden size H and expert intermediate size I: NVFP4 (E4M3 / 16) and MXFP4 (UE8M0 / 32) weights, BF16 input, FP32 SiLU, intermediate, down projection and accumulation. AVX512 BF16 (+VBMI) and AVX2 / FMA kernels; NUMA topology from sysfs, each expert's rows split over the nodes, one worker pool per node. One translation unit on purpose: the kernels must compile to the same code as before. |
| `engine/dispatch.cpp` | `libxtier_engine.so`: exports the API and loads the AVX512 or AVX2 build next to it on first use (`XTIER_ISA=avx512\|avx2` forces one). |
| `mailbox/mailbox.cpp` | `libxtier_mailbox.so`: mapped host mailboxes polled by one dispatcher thread, stream memory operations instead of host callbacks (CUDA-graph capturable), and the expert-pool command ring with its policy thread. |
| `include/xtier_engine.h`, `include/xtier_mailbox.h` | C APIs. |
| `engram/rows.cpp` | `libxtier_engram.so`: DeepSeek V4.1 Engram rows on NVMe (O_DIRECT / AIO reads, bounded row cache, preloaded scales, TP row range), ported from the 1017 pipeline by [tools/port_engram.py](../tools/port_engram.py); its lookup calls no CUDA API. |

Build: `tools/build_engine.sh [CUDA_ROOT]` (CMake; the AVX512 build uses the one-row Down microkernel, `XTIER_AVX512_DOWN_R4=1`,
like the lkqmoe release). Check: `tests/engine_ab.py` (one real layer, output digests and timings).

Status (2026-10-23): on one real layer of Qwen3.8 Flash Next (NVFP4), GLM-5.3 Flash (NVFP4) and DeepSeek V4.1 Flash 1017 (MXFP4),
9 shapes each (1 to 2048 tokens, decode and verify routing), the outputs equal lkqmoe 0.5.6 bit for bit with both the AVX512 and
the AVX2 kernels. The AVX512 kernel library disassembles to the same instructions as the production build except for the removed
`lkqmoe_create` compatibility entry point.
