# Generic Multi-Drive NVMe KV Cache Offloading (`xtier/kvstore`)

Date: 2026-11-26  
Host: RTX PRO 5000 72 GB Blackwell (SM120) · 2× EPYC 9334 NPS4 · 4× Samsung 9100 PRO PCIe 5.0 NVMe (`/data/fast/cold{1..4}/xtier-kv/pool.dat`, 4 × 50 GiB = 200 GiB)

## 1. Architecture Summary

XTier provides a **model-agnostic** L2 NVMe KV cache offloading & reload engine integrated via SGLang's public `HiCacheStorage` v2 dynamic backend SPI (`--hicache-host-memory-mode buffer_only --hicache-storage-backend dynamic` with `"interface_v1": 1`), requiring **zero SGLang source patches** and **zero model-name branches**:

- **Native C++ engine** (`csrc/include/xtier_kvstore.h`, `csrc/kvstore/kvstore.cpp` -> `xtier/engine/lib/libxtier_kvstore.so`):
  - Per-drive `io_uring` + `O_DIRECT` + `O_NOATIME` worker threads over 4 preallocated 50 GiB `pool.dat` files (200 GiB total, volatile per server process, reboot-clear).
  - 64-shard `std::shared_mutex` hash index with 256 MiB segmented ring recycling + CLOCK hot-entry relocation (≤ 50% per recycled region) + in-flight reader refcounts + read-before-write priority.
  - Zero-copy I/O directly to/from SGLang's pinned host buffers when 4 KiB aligned (`bounce=0` for all 4 KiB-aligned pools); aligned bounce fallback only for unaligned payloads (e.g. Mamba state).
- **Python HiCache v2 backend** (`xtier/kvstore/__init__.py`, `native.py`, `backend.py`):
  - Implements `HiCacheStorage` v2 (`register_mem_host_pool_v2`, `batch_exists_v2`, `batch_set_v1`, `batch_get_v1`, `batch_set_v2`, `batch_get_v2`, `clear`, `close`).
  - Supports `ALL_PAGES`, `TRAILING_PAGES`, `BLOCK_GROUPS` hit policies and `LogicalHostPool` 0-byte anchors (DeepSeek-V4.1).
  - Enforces `tp_size == 1` (`tp_size > 1` degrades cleanly to no-op).
  - Defaults to `--hicache-ratio 1.25` (`XTIER_KV_STAGING_RATIO`), universally compatible across all models (including DeepSeek-V4.1 which disallows `--hicache-size`).
- **Generic self-retiring HiCache bridge** (`xtier/kvstore/bridge.py`, wired in `xtier/plugin.py`):
  - Registers **0 hooks when `XTIER_KV_OFFLOAD=0`** (default OFF; zero impact on SGLang weekly upgrades).
  - When `XTIER_KV_OFFLOAD=1`, registers 4 generic hooks tracked in `tools/hook_sources.json` (`check_sglang.py --strict`), each auto-retiring if upstream SGLang adds native support:
    1. Heterogeneous packed draft fallback (`BaseSpecWorker._build_hicache_draft_plan`): detects KV geometry/dtype/scale mismatch between target & draft and falls back to `HiCacheDraftMode.SIDECAR`.
    2. Packed sub-byte MHA dim normalization & `KVScaleLayout.TOKEN` deferral (`build_kv_host_pool`): aligns packed sub-byte MHA `MHABufferInfo.head_dim`/`v_head_dim` (e.g. NVFP4 `uint8` `head_dim // 2`) to physical storage width, and defers unsupported `KVScaleLayout.TOKEN` K/V scales to paged sidecars (`DeepSeekV4PagedHostPool(page_aligned_only=True)`).
    3. Missing declared `IndexKeyBufferInfo` sidecar attachment (`attach_hybrid_pool_to_unified_cache`).
    4. Exact `data_ptr()` coverage verifier (`attach_hybrid_pool_to_unified_cache` + `maybe_register_hicache_draft`): verifies every persistent device KV/indexer/scale/Mamba/draft tensor is bound by a registered host pool; disables storage under `XTIER_KV_STRICT_COVERAGE=1` on any uncovered tensor rather than corrupting KV.
  - Existing Hy4-Flash config guard (`xtier/packs/hy4_flash/dfly.py`) allows `--enable-hierarchical-cache` when `XTIER_KV_OFFLOAD=1` with zero new hooks.

## 2. Phase 0 C++ Engine & Microbenchmark Verification

- **ASan + UBSan** (`test_kvstore.cpp`, 4 unit/stress suites including concurrent ring recycle & CLOCK relocation): **PASS** (`recycled=4702, hits=85888, relocated=36972, evict_waits=210, corrupt=0`, 0 sanitizer errors).
- **TSan** (same 4 suites): **PASS** (0 data races).
- **4× 9100 PRO E2E microbenchmark** (`tools/kvstore_bench.py`, 8 GiB, 512 KiB pages, 4 drives):
  - NVMe Write (`O_DIRECT`): **28.19 GB/s**
  - NVMe Read → Pinned Host: **40.28 GB/s**
  - Pinned Host → GPU (H2D): **57.47 GB/s**
  - **E2E NVMe → Pinned → GPU**: **35.79 GB/s** (target gate > 20 GB/s)
  - Direct I/O: `direct=65536, bounce=0`; Linux Page Cache `Dirty = +0.4 MiB`.

## 3. Multi-Model Live E2E Verification (`KV_POOL=65536`, 4 Evicted Docs, Reload + 2-Way Concurrent + `copy_test.py`)

All 4 major models were verified end-to-end with 4 distinct ~27K–34K-token documents (total ~110K–123K tokens overflowing a 64K GPU KV pool), followed by single-stream evicted NVMe reload, 2-way concurrent NVMe reload, and `bench/copy_test.py` at concurrency 1 & 4:

| Model | Attention / KV / Spec | Registered Host Pools & `data_ptr()` Coverage | Pinned Staging | Cold TTFT | NVMe Reload TTFT (Speedup) | 2-Way Conc TTFT | Cache Hit | Peak Zero-Copy Read BW | `copy_test.py` (c=1 / c=4) | Page Cache `Dirty` Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DeepSeek-V4.1-Flash-1017** | C1/C2 Compressed MLA + SWA + Indexers + DSpark | 6 pools (`kv` anchor, `swa`, `c1`, `c1_idx`, `c2`, `c2_idx`); **51/51 + 3/3** | 1.77 GiB | 5.76–6.09 s | **0.576–0.630 s** (**9.3×–10.4×**) | 1.07–1.08 s | 99.2–99.4% | — | **0/8 / 0/8** | −6.2 MiB |
| **GLM-5.3-Flash-NVFP4** | MLA + DSA Indexer + Mamba + EAGLE | 3 pools (`kv`, `indexer`, `mamba`); **22/22** | 1.95 GiB | 10.48–16.39 s | **0.535–0.769 s** (**17×–21×**) | 1.12–1.13 s | 99.5–99.6% | `kv` 7.46 GB/s, `mamba` 9.66 GB/s | **0/8 / 0/8** | −1.4 MiB |
| **Hy4-Flash-NVFP4** | SWA + DSA MLA + DFly2 | 3 pools (`kv`, `swa`, `indexer` auto-attached); **54/54** | 0.50 GiB | 9.11–19.29 s | **0.549–0.699 s** (**13×–32×**) | 0.628–0.872 s | 99.8–99.9% | `kv` **11.37 GB/s** | 1/8* / 1/8* (4285/4286 chars, benign space) | −2.0 MiB |
| **Qwen3.8-Flash-Next** | NVFP4 QSA + GDN/Mamba + FP8 MTP | 7 pools (`kv` packed-dim normalized, `kv_k_scale`, `kv_v_scale`, `indexer`, `mamba`, `draft`, `draft_indexer`); **64/64 + 7/7** | 1.68 GiB | 5.93–6.89 s | **0.285–0.664 s** (**9.0×–20.9×**) | 0.441–0.602 s | 99.9–100.0% | `kv` **11.39 GB/s** | **0/8 / 0/8** (MTP accept 3.80–3.95) | −3.5 MiB |

Across all 4 models:
- Checksum verification on all evicted reload & concurrent reload passes: **100% exact match**.
- Prometheus counters: `storage_prefetch_deferred_tokens_total = 0`, `storage_prefetch_unfulfilled_tokens_total = 0`, `hicache_dropped_tokens_total = 0`, `hicache_host_used_tokens = 0` (buffer-only staging fully reclaimed).

## 4. Console UI & Usage

- Console option `kvoffload` (`NVMe KV 二级卸载`, default **OFF**) is wired across all model configs (`ds41`, `ds41mini`, `glm53`, `hy4flash`, `qwen38`, `inkling2`, `minimax31`) via `xtier/console/core.py` (`kv_offload_option` / `kv_offload_env` / `kv_offload_args` / `kv_offload_tags`).
- CLI usage with any launch script:
  ```bash
  XTIER_KV_OFFLOAD=1 configs/ds41-pro5000-nps4/run-ds-xtier.sh $(python -c 'from xtier import kvstore; print(" ".join(kvstore.sglang_args()))')
  ```
- Optional overrides: `XTIER_KV_PATHS`, `XTIER_KV_SIZE_GIB`, `XTIER_KV_STAGING_RATIO` (default `1.25`), `XTIER_KV_STAGING_GIB` (overrides ratio when > 0), `XTIER_KV_WRITE_QD`, `XTIER_KV_READ_QD`, `XTIER_KV_REGION_MIB`, `XTIER_KV_RELOCATE_PCT`, `XTIER_KV_STRICT_COVERAGE`.

## 7. 启动台 100G / 200G 选项与 DS4.1 生产默认 `dsh` 全链路实测

### 7.1 启动台配置升级（`100 GiB` / `200 GiB` 与 DS4.1 默认开启）
- `xtier/console/core.py`：`kv_offload_option` 升级为三档选项：`关`、`100 GiB（4×25 GiB）`、`200 GiB（4×50 GiB）`，通过 `XTIER_KV_SIZE_GIB=<100|200>` 控制单盘环容量（在既有 50 GiB/盘的 `pool.dat` 上零截断直接生效）；无 NVMe 池或缺失 `libxtier_kvstore.so` 的机器自动回退为 `off` 并将两档置灰。
- **开源仓库默认关闭**：`configs/*/console.py` 全部 7 个模型默认 `kvoffload='off'`（零钩子、零开销、无外部盘依赖），用户需要使用时在启动台选择 `100 GiB（4×25 GiB）` / `200 GiB（4×50 GiB）` 或设置 `XTIER_KV_OFFLOAD=1` 自行打开（DS4.1 下 100 GiB 约装 6,300 万 token / 约 118 个 512K 会话）；本机私有 WebUI 接线（`qwen-runtime/xtier_ds1017_storage.py`，同 `XTIER_FAST_LOAD=1` 机制、不入开源仓库）单独为本机 `ds41` / `ds41uncensored` 设为默认 `100 GiB`。
- WebUI（`qwen-webui.service` `:8090`）已重启生效，实测 `ds41` 与 `ds41uncensored` 默认与已保存配置均切至 `100 GiB`。

### 7.2 生产默认配置启动验证（`KV_POOL=1M` + `XTIER_KV_SIZE_GIB=100`）
使用仓库 `ds41` 启动台默认参数直接拉起 `DeepSeek-V4.1-Flash-1017`（`KV_POOL=1048576`、`BATCH=49152`、弹性热槽 `35→68`、`DSpark compact`、`Recall` 开、`--prefill-decode-interval 72`、`XTIER_KV_OFFLOAD=1`、`XTIER_KV_SIZE_GIB=100`）：
- `xtier kvstore: opened 4 drives x 25.0 GiB (total 100.0 GiB, region=256 MiB, write_qd=4, read_qd=64)`
- `staging pinned memory = 5.20 GiB`（FULL 1M 池 + SWA 固定上限 `109,568` token，× `1.25`）
- `coverage check OK (attach): 51/51 device KV tensors bound`、`coverage check OK (draft_sidecar): 3/3`
- 启动后 `MemAvailable = 71.1 GiB`，整场 >630 万 token 压测期间最低 `MemAvailable ≈ 44.8 GiB`（远高于 `20 GiB` 内存守护红线），显存 `51.9 / 73.4 GiB`。

### 7.3 真实 `dsh` 客户端三大场景实测结果

| 场景 | 测试设计 | 挤出与读回指标 | 耗时 / 加速 | 内容正确性 |
|---|---|---|---|---|
| **Test 1a：多会话 1M 池挤出 → 读回** | 真实 `dsh` 连发 4 个 ~370K 会话（A/B/C/D 合计 1.495M > 1M GPU 池），A 全部挤出后 `--session-id` 切回 A | `cached=369,920/370,132`（**99.94%**），**`storage_prefetch_hit=364,032`**（100% 挤出页全部从 NVMe 读回） | `dsh` 总耗时 `49.73s → 4.86s`（服务端读回+首字 `<0.9s`，**>50×**） | 首/中/尾暗号 `COBALT-9182` / `QUARTZ-4471` / `OBSIDIAN-6305` 全对 |
| **Test 1b：多会话乒乓切回** | A 读回将 B 挤出后，立即 `--session-id` 切回 B（375K） | `cached=349,952/375,113`（**93.3%**），**`storage_prefetch_hit=344,064`** | `dsh` 总耗时 `42.91s → 7.88s`（**5.4×**） | 首/中/尾暗号 `CRIMSON-5510` / `TOPAZ-2284` / `EMERALD-7740` 全对 |
| **Test 2a：`dsh` 原生读图 + 挤出读回** | `dsh` `read_image` 读入 `img_alpha.png` + 50K 文本 → 1.18M 流量彻底挤出 GPU → `--session-id` 切回追问 | `cached=59,392/60,042`（**98.9%**），**`storage_prefetch_hit=59,392`** | `dsh` 总耗时 `24.2s → 4.81s` | 准确回忆图内 `ALPHA-7429` / `NORDIC-88` / `991-BLUE-RECT` / 蓝色矩形及文本首尾令牌 |
| **Test 2b：同文本不同图片防碰撞** | 完全相同 40K 文本前缀分别配 `img_alpha.png` 与 `img_beta.png` → 1.18M 挤出 → 分别续聊 | 两路均 **`cached=40,704/40,865`（99.6%）**，**`storage_prefetch_hit=40,704 × 2`** | TTFT `5.91s → 1.27s` / `5.09s → 1.05s`（**4.7–4.9×**） | 零串味（alpha 仅答 `ALPHA-7429`，beta 仅答 `BETA-3851`） |
| **Test 3：`dsh` 自动压缩（含图片）+ 挤出读回** | 启用 `compaction-basic` 真实触发 2 次自动压缩（含 `img_beta.png`），压缩稳定后再用 1.20M 流量挤出 GPU 并续聊 | Turn 5 压缩后完整前缀 **`cached=18,944/19,297`（98.2%）**，**`storage_prefetch_hit=18,944`** | `dsh` 总耗时 **`3.14s`**（`first_delta=3.08s`） | 图片码 + 压缩前 PART1/2 + 压缩后 PART3 共 5 组暗号 100% 准确回忆，无旧 KV 污染 |

- **引擎关停汇总**：`xtier kvstore: closing (puts=83481, hits=13752, misses=0, recycled=0, relocated=0, direct=0, bounce=97233)` — 累计写入 **83,481 页**、读回命中 **13,752 页**、**`misses=0`**、**0 报错 / 0 Traceback**；`/proc/meminfo` `Dirty` 全程稳定在 `4–7 MiB`（`O_DIRECT` 零页缓存污染）。
