# XTier

[中文](../README.md) · [Quick start](quickstart.en.md) · [Docs index (ZH)](README.md) · [Support matrix (ZH)](support.md)

**Cross-tier CPU/GPU hybrid MoE inference for [SGLang](https://github.com/sgl-project/sglang).** XTier is a zero-patch plugin for upstream SGLang (`main`): hot MoE experts live in GPU VRAM, the rest in host DRAM with adaptive scheduling, while NVMe holds large embedding tables (DeepSeek Engram, Qwen PLE) or optional cold-expert tails.

**Status: early development** ([plan](PLAN.md)). Inkling-2-Small, Hunyuan-4-Flash NVFP4, DeepSeek V4.1 Flash 1017, Qwen3.8 Flash Next and GLM-5.3 Flash run on upstream SGLang `main`; MiniMax-M3.1-Flash is paused.

- **Zero SGLang source patches**: attaches via official plugin hooks with startup self-checks.
- **Model-agnostic core**: wraps standard `FusedMoE` quantized layers (NVFP4 / MXFP4 / MXFP8).
- **Optional packs**: model-specific kernels and fixes live in `xtier/packs/` and can be toggled in the web console.

## Layout

```
xtier/
├── backend/   Hybrid MoE backend: weight offload, CUDA-graph mailbox decode, GPU grouped GEMM / CPU prefill
├── cache/     Adaptive hot-expert cache (elastic slots), second-GPU pool (XTIER_GPU2_SLOTS), NVMe cold pool (XTIER_COLD=1)
├── planner/   Auto-layout planner from host probes & routing profiles (docs/planner.md)
├── engine/    CPU engine Python bindings & pinned memory (C++ sources in csrc/)
├── packs/     Model packs: inkling2 / hy4_flash / deepseek_v4 / qwen4_exp / glm5_next / minimax_m3 / speculative
├── tables/    NVMe-backed large embedding tables (Engram / PLE) with CUDA-graph row cache
└── console/   Stdlib-only web console for model launch, tuning & live telemetry
csrc/          AVX-512 / AVX2 CPU expert kernels, NUMA thread pool & mailbox
configs/       Per-(model × hardware) launch scripts and console cards (console.py)
tests/ tools/  Unit/regression tests, tiered release verification (tools/fast_verify.py), build & planner tools
bench/         Acceptance benchmarks (PPL, verbatim copy, agent_bench, context & concurrency speed)
```

## Quick start

See [Quick start](quickstart.en.md) for a full setup. With an existing SGLang environment:

```bash
git -C /path/to/sglang checkout 4ce41378f9      # tested upstream SGLang main (2026-11-21)
SGLANG_BUILD_RUST_EXTS=none pip install -e /path/to/sglang/python
tools/build_engine.sh /path/to/site-packages/nvidia/cu13
pip install --no-deps -e .
python tools/check_sglang.py --strict
python -m xtier.console --port 8080
```

## Supported models

| Model | Pack | Highlights |
|---|---|---|
| Inkling-2-Small (594B / 17B, multimodal) | `inkling2` | Runs on `main` (512K / 2-conc, 5-layer DSpark + 23-layer ViT): NVFP4 KV (6.75 GiB), 64K single-chunk prefill, lossless DSpark truncation, default Plan A+ (2% cold tail). [Report (ZH)](reports/2026-11-25-inkling2-small.md) |
| Hunyuan-4-Flash NVFP4 | `hy4_flash` | Runs on `main` (1M context / 4-conc, native 5-layer DFly2): hybrid SWA+DSA NVFP4 KV (7.95 GB), FP8 IndexPool, NVMe Engram, 0/4 copy errors |
| DeepSeek V4.1 Flash 1017 | `deepseek_v4` | Runs on `main` (512K / 4-conc, 1M KV pool, DSpark + Recall): NVMe Engram, 43-layer FP8 `wo_a`, 24-token weight reuse, 500K prefill >9,000 tok/s |
| Qwen3.8 Flash Next | `qwen4_exp` | Runs on `main` (512K, MTP): NVFP4 KV + auto planner layout, NVMe PLE, production-grade accuracy |
| GLM-5.3 Flash | `glm5_next` | Runs on `main` (512K, EAGLE): FP8 dense layers, hot-vocab rejection sampling, elastic slots (+22% decode) |
| MiniMax-M3.1-Flash | `minimax_m3` | **Paused** (2026-10-26, SGLang PR #45925 branch) |

## Supported GPUs & multi-GPU

Full verification matrix: [support.md](support.md) (in Chinese).

| GPU / Mode | Status |
|---|---|
| Blackwell (SM120) | Primary dev/benchmark platform (RTX PRO 5000 72 GB); RTX PRO 6000 96 GB verified on Qwen3.8 |
| Ada / RTX 40 & L4 / L20 (SM89) | Supported: RTX 4090 & L4 verified on Qwen3.8 ([config](../configs/qwen38-rtx4090-3090/README.md)); community `2×L20 TP=2` DeepSeek config ([config](../configs/ds41-l20-tp2/README.md)) |
| Ampere / RTX 30 & A100 (SM86 / SM80) | Supported: RTX 3090 & A100 40 GB verified on Qwen3.8 (bit-exact software FP8 decode) |
| Dual-GPU `TP=2` | No P2P/NVLink needed (NCCL fallback); 2× RTX 3090 +42–48% on Qwen3.8 ([report](reports/2026-10-30-tp.md)) |
| Second-GPU hot pool (`XTIER_GPU2_SLOTS=auto`) | Offloads hot experts to a slower second GPU (`TP=1`), +28–40% in 24+16 GB emulation ([report](reports/2026-10-31-second-gpu-pool.md)) |

## Speed

One RTX PRO 5000 72 GB + 2× EPYC 9334 (NPS4) + 384 GiB DDR5, in `tok/s`. Baseline = console "packs off" (upstream SGLang + XTier hybrid backend + required bring-up fixes, **not stock SGLang**). Full methodology & accuracy: [benchmarks.md](benchmarks.md).

### Inkling-2-Small (594B / 17B, multimodal)

512K context / 2 concurrent requests (40 NVFP4 + 8 MXFP8 MoE, 5-layer DSpark + 23-layer ViT), SGLang `4ce41378f9`:

| Mode | Decode bs=1 | Decode bs=2 agg | 64K cold prefill | 256K cold prefill | 500K cold prefill | 500K cached decode |
|---|---:|---:|---:|---:|---:|---:|
| **XTier + packs (default)** | **74.7** (r0 81.7) | **78.4** | **3,835** | **2,464** | **1,593** | **43.7** (peak 47.0) |
| Baseline (packs off) | 47.9 | ~43 | 1,591 | 1,217 | OOM | OOM |

Packs vs baseline: **2.41× / 2.02×** faster 64K / 256K prefill, **+56% / +82%** faster `bs=1` / `bs=2` decode, and NVFP4 KV (6.75 GiB) unlocks 500K. 2-round `agent_bench` (64K→513K): **66.2 tok/s** long replies (66.3 overall, accept 4.02). Accuracy: 13-seg `ΔNLL +0.00423` / `KL 0.00938`, copy `0/8`, 256K/500K needles `5/5`. Plan A+/B2 & trace anatomy: [report](reports/2026-11-25-inkling2-small.md) (ZH).

### Hunyuan-4-Flash NVFP4

1M KV pool / 1M context / 4-conc (hybrid SWA+DSA NVFP4 KV 7.95 GB + native 5-layer DFly2), SGLang `4ce41378f9`:

| Mode | Decode bs=1 | Decode bs=4 agg | 100K needle prefill | 4×128K prefill | 876K–950K prefill | Long cached decode |
|---|---:|---:|---:|---:|---:|---:|
| **XTier + packs + DFly2 (default)** | **92.1** (steady 124–152) | **186.8–190.4** (steady 209–226) | 3,630 (PASS) | 3,565 | 3,454 (876K) | **87.5–110.8** (876K) |
| Baseline (FP8 KV, BF16 dense) | 55–56 | 148–162 | 3,657 (PASS) | — | 3,084 (950K) | 43.8 (950K) |

Copy `0/4` (299–306 tok/s at conc=2), 100% lossless PPL, 0 OOM retries at 876K.

### DeepSeek V4.1 Flash 1017

2-round [agent_bench.py](../bench/agent_bench.py) 64K→500K coding agent (packs measured 2026-11-24 on SGLang `4ce41378f9`):

| DSpark | Round | Cold prefill 64K | Cold prefill 500K | Decode (long replies) | Acceptance | Per step |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + packs** | 1 | 5,969 | **9,038** | 156.6 | 4.64 | 29.6 ms |
|  | 2 | **7,967** | **9,023** | 154.7 | 4.65 | 30.1 ms |
| **XTier + packs, 4 concurrent agg** | — | — | — | **268** | 3.40 | 51.0 ms |
| Baseline (packs off) | 1 | 3,318 | 3,627 | 65.5 | 3.25 | 50.9 ms |

Packs vs baseline: **41%** faster per step, **+110% / +149%** faster 64K / 500K cold prefill (100% lossless PPL 1.2001 vs 1.1996, 0 body copy errors); all 127 requests 163.9 tok/s, 2-way 195.8 tok/s, 4-way rewrite/edit 332–382 tok/s. See [9K+ prefill report](reports/2026-11-19-lossless-9k-prefill.md) and [Recall report](reports/2026-11-18-lossless-prefill-recall.md).

### Qwen3.8 Flash Next

| MTP | Round | Cold prefill 64K | Cold prefill 500K | Decode (long replies) | Acceptance | Per step |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + packs** | 1 | 5,715 | 5,170 | 181.7 | 2.57 | 14.2 ms |
|  | 2 | 5,610 | 5,150 | 207.2 | 2.68 | 13.0 ms |
| Baseline (packs off) | 1 | 5,319 | 4,078 | 139.5 | 2.13 | 15.2 ms |

Packs vs baseline: 11% faster per step, 6% / 27% faster cold prefill (NVFP4 KV + auto planner, 0 copy errors).

### GLM-5.3 Flash

| EAGLE | Round | Cold prefill 64K | Cold prefill 500K | Decode (long replies) | Acceptance | Per step |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + packs** | 1 | 2,648 | 2,640 | 96.3 | 2.54 | 26.4 ms |
|  | 2 | 2,715 | 2,635 | 98.2 | 2.66 | 27.1 ms |
| Baseline (packs off) | 1 | 2,663 | 2,518 | 76.8 | 2.61 | 34.4 ms |

Packs vs baseline: 22% faster per step, 1% / 5% faster cold prefill (FP8 dense + elastic slots, PPL 3.987 → 3.998, 0 copy errors).

### MiniMax-M3.1-Flash (paused)

SGLang M3.1 branch (PR #45925), 64K session: 51.7 vs 46.1 tok/s (55.4 vs 84.1 ms/step, 34% faster); [report](reports/2026-10-24-minimax-512k.md).

## Cold experts on NVMe & minimal run

When host DRAM cannot fit all experts, `XTIER_COLD=1` offloads the coldest tail to NVMe ([design](design-cold-experts.md)), with single-copy checkpoint punching (`tools/cold_punch.py`), unequal drive weighting (`XTIER_COLD_UNITS`), and optional route tail skip (`XTIER_TAIL_SKIP`).

- **4× PCIe 5.0 NVMe (DeepSeek 64K)**: 10% / 20% / 30% / 50% cold yields `135 (−5%)` / `124 (−13%)` / `112 (−21%)` / `80 (−44%) tok/s`; [report](reports/2026-11-01-cold-pcie5.md).
- **Minimal run (32 GB VRAM + 128 GB RAM emulated, 70% cold)**: DeepSeek 32K–128K prefill `1,155–1,186 tok/s`, decode `28.1–32.7 tok/s`; [report](reports/2026-10-27-ds41-minimal.md).

## Recent changes

- **2026-11-25 (v0.1.10)**: adds zero-patch **Inkling-2-Small (594B multimodal)** pack & console card (default Plan A+: `cold=2%`, 45 hot slots, 64K single-chunk 3,835 tok/s, NVFP4 KV unlocking 500K, 2-round agent 66.3 tok/s; `cold=off` Plan B2 with `L47` full-GPU), plus the 11-24 DeepSeek 9K+ lossless speedups. [Report](reports/2026-11-25-inkling2-small.md)
- **2026-11-24**: 100% lossless DeepSeek V4.1 & core speedups (24-token weight reuse, 43-layer FP8 `wo_a`, fused writeback, SM120 eager MXFP8), pushing 500K cold prefill >9,000 tok/s and 4-way fresh decode to 268 tok/s.
- **2026-11-23 (v0.1.9)**: adds zero-patch **Hunyuan-4-Flash NVFP4** pack & native 5-layer DFly2 speculative decoding (1M NVFP4 KV 7.95 GB, bs=1 92.1 / bs=4 190 tok/s), `tools/fast_verify.py` 3-tier verification, and community SM89 L20 PRs (#18/#7/#19).
- **2026-11-22 (v0.1.8)**: tracks upstream SGLang `4ce41378f9`, shipping 9,000+ lossless prefill / 1.1 s tool turns and default CPU multimodal decoding. [Report](reports/2026-11-22-sglang-4ce4137.md) · [9K report](reports/2026-11-19-lossless-9k-prefill.md)
- **2026-11-18 (v0.1.7)**: lossless prefill fusion, cross-chunk Engram lookahead, and DSpark context Recall (180.4 tok/s across all agent requests). [Report](reports/2026-11-18-lossless-prefill-recall.md)

Earlier releases and full technical notes: [docs/changelog.md](docs/changelog.md) (ZH).

## License

Apache-2.0, see [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
