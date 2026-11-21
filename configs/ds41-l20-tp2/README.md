# DeepSeek-V4.1-Flash-1017 on 2×L20 48G (SM89)

TP=2 on **GPU0 + GPU2** — one card per NUMA node — with the XTier packs on the official SGLang main.

Measured numbers and the full tuning log: [docs/reports/2026-11-20-l20-tp2-tuning.md](../../../docs/reports/2026-11-20-l20-tp2-tuning.md).

## Hardware layout

```
GPU0 → NUMA node 0 (PCIe NODE)     GPU2 → NUMA node 1
GPU1 → node 0 (spare)              GPU3 → node 1 (Strata / Qwen)
```

**One TP rank per NUMA node — this is the hard constraint of this box.**

XTier places the CPU expert workers per NUMA node (`xtier/engine/placement.py`) and the
C engine picks their CPUs with `xtier_worker_cpus()` (`csrc/engine/engine.cpp`), which
has **no rank offset**: two ranks on one node get the *same* CPU set.

Measured with both cards on node 0 (`XTIER_TP_NODES=0|0`):

| | one card per node (0+2) | both on node 0 (0+1) |
|---|---:|---:|
| worker CPUs of the two ranks | disjoint | **29/29 identical** |
| CPU time per layer | 625–633 µs | **8812 µs** (34×) |
| single request | 36–55 tok/s | stuttering |
| model load | stable | Bus error / Segmentation fault |

`XTIER_TP_NODES` is therefore set explicitly to `0|1`; the `auto` default splits the
nodes over the ranks and only lines up when the cards sit on different sockets.

The cards are on different sockets, so **no P2P**: `--disable-custom-all-reduce` (the
custom all-reduce's IPC registration fails without P2P and kills the CUDA graph
capture) and `NCCL_P2P_LEVEL=PHB`.

## What SM89 changes

| | PRO 5000 (SM120) | L20 (SM89) |
|---|---|---|
| FP4 tensor cores | yes | **no** — the MXFP4 experts are dequantized in software |
| FP4 indexer | yes | **no** (`--enable-deepseek-v4-fp4-indexer` refuses SM89) |
| b12x / Markov / MXFP8 prefill tile | yes | **no** (hard `capability[0] == 12` gates) |
| shared memory per SM | 228 KiB | **99 KiB** — the production 128-wide prefill tile does not fit |
| `--fp8-gemm-backend` | flashinfer_cutlass | keep the default `auto` (Triton), so the L20 tuned configs apply |

The tuned FP8 GEMM configs for this card are **in the pack**
(`xtier/packs/deepseek_v4/fp8_configs`, `device_name=NVIDIA_L20`), tuned per M:
decode's M is batch × top-k and ranges from 1 to 48, and the best tile differs per M
(1.3× over the default tile at the median, up to 6.9× at M=1).

The CPU engine's kernel is built with `-mtune=emeraldrapids`, detected from the build
host (`csrc/CMakeLists.txt`); the upstream default was `-mtune=znver4`, i.e. an AMD
schedule on an Intel Xeon.

## Configuration

| | |
|---|---|
| KV pool | 1,572,864 tokens (= 6 × 256K) |
| context | 262,144 |
| prefill chunk | 32,768 |
| **prefill-decode interleave** | **32**（多请求不卡顿的关键） |
| resident expert slots | 32/layer, elastic up to 128 |
| expert profile | `ds41-profile-l20.json` (this box's routing trace) |
| CPU threads | 32/rank (one worker per physical core) |
| DSpark | on, compact verification, this box's SPS table |
| Engram rows | NVMe, 4 GiB row cache |

## Measured

The metric that matters for an agent's feel is the **gap between tokens**, not the aggregate
throughput (`tools/bench_latency.py`):

| | ITL p50 | max gap |
|---|---:|---:|
| 6 requests × 100K context | 45–67 ms | **332 ms** |
| 4 requests × 256K context (1M tokens) | 50–54 ms | **76 ms** |

With `--prefill-decode-interval 0` the same test shows a **116,000 ms** gap: several long
prefills run back to back and the requests that are already decoding get nothing.

| | |
|---|---|
| cold 100K prefill | **3488 tok/s** |
| 8 sessions × 4 turns | 55–57 tok/s aggregate, single stream 36–52 tok/s |
| 4 requests × 84K | 91.5 s |
| 6 requests × 60K | 97.8 s |
| 4 requests × 128K | 137.7 s |
| 4 requests × 256K | 409.4 s |
| 6 requests × 256K | 505.3 s, all six succeed, queue 0 |

## Running it

The systemd service `DeepSeek-V4.1-Flash.service` runs
`/opt/Xtier/start-ds41-with-console.sh` → this directory's `run-ds-xtier.sh`, with the
read-only console on 8100. The console card (`console.py`) documents the same options.

`run-ds-xtier.sh` builds its argument list in an array — **no `#` comments inside the
command**: a comment between two backslash-continued lines makes bash swallow the rest
of the logical line (it cost this config `--disable-custom-all-reduce` once).
`DRYRUN=1 bash run-ds-xtier.sh` prints the exact argument list.
