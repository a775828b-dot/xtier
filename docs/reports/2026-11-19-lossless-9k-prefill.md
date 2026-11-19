# 全双工零同步预填突破 9,000 tok/s、SM120 mHC 精确核与工具回传提速（2026-11-19）

基于 SGLang `main 084a5e9aae`（v0.1.7 环境），在**保持启动台 48K 默认分块（`BATCH=49152`、35 基础槽 + 弹性 68 槽、`--prefill-decode-interval 72`）不变且 100% 精度无损**的前提下，消除 GPU/CPU 预填管线中的隐式流同步、PCIe 空转与 SM120 算子回退，将 DeepSeek V4.1 长上下文冷 prefill 推过 **9,000 tok/s**（`agent_bench` 384K–500K 会话达 `8,894–9,085 tok/s`、458K 冷提示达 `9,180 tok/s`、稳态 48K 单块峰值 `9,999 tok/s`），同时将 500–900 token 工具回传首字延迟从 `2.07 s` 压降至 **`1.12 s`**，并消除 48K 分块下 `mhc.post` 的 **`+22.5 GiB` 瞬时显存尖峰**。[回到文档目录](../README.md)

## 1. 改动拆解：通用核心层 vs DeepSeek V4 加速包

### 1.1 XTier 通用核心层（全模型生效）

1. **`_run_fp4` 纯 DMA 双流解耦、保序单遍规约与零冗余流同步（`xtier/backend/prefill.py`）**：
   - **锁页预重排 `UE8M0` 缩放 + 自适应 24 暂存槽**：在锁页内存一次性预计算 `block_scale_interleave`，彻底清空 `_copy_stream` 上每块 `3,840` 次 GPU SM 核；暂存槽自适应扩至 `24`（`m < 2048` 用 `8`），统一在 `compute` 流分配消除 PyTorch 双流显存池割裂，支持跨层与跨分块（`L_last → L0`）纯 DMA 预取。
   - **消除每块 `280+` 次隐式 GPU→CPU 同步**：用 `2.4 MiB` L2 驻留位图核 `_mark_chunk_tokens` + `_compact_chunk_rows` 替代 `torch.unique`（省 `171 ms/块`）；定位 PyTorch CUDA `torch.bincount` 每次调用均执行 `.min().item()` / `.max().item()` 造成每块 `200` 次阻塞同步并卡死 Python 预取的瓶颈，替换为零同步 Triton `_hist_rows`；用 64-bit `_copy_pool_chunk8` 单核替代每块 `2,100+` 次逐专家 `.copy_()`。
   - **保序单遍 `_reduce_band` 与全双工 `CPU_SHARE`**：默认 `FP4_ROWS=8192`、`BAND_ROWS=65536` 单遍严格按升序专家块累加 FP32（与原路径逐位一致）；新增 `_d2h_stream` 与常驻 `_PersistentWorker`，配合 `_copy_stream` 实现 PCIe 5.0 x16 双向全双工重叠。
2. **CPU 专家引擎 `m > 16` 动态工作窃取与私有 L2 缓冲（`csrc/engine/engine.cpp`、`xtier/cache/prefill.py`）**：
   原先 `m > DECODE_M(16)` 时强制走静态切分，热点专家集中时 8 个 NUMA 节点出现长尾等待；现对 `m * H <= 16M` 放开 `dynamic` guided 窃取与线程私有 L2 `local_buffer`，并在 `hybrid_prefill` 中复用常驻线程与锁页缓冲（单测 `m=1000`：DS `139 → 49 ms`、GLM `127 → 33 ms`、Qwen `201 → 21 ms`）。
3. **HTTP 入口多线程并行 BPE 分词（`xtier/platform.py`）**：
   挂载 `TokenizerManager._tokenize_texts` 与 `init_tokenizer_and_processor`，对 `>= 64 KB` 文本按换行边界多线程并行编码并校验特殊词元边界：68.2 万 token 分词耗时从 `1,569.9 ms` 降至 **`208.1 ms`（快 `7.5x`，100% 逐 token 一致）**。
4. **弹性热专家槽 `2,048` 暂存档位边界修复（`xtier/cache/elastic.py`）**：
   将 `self.small` 与混合预填阈值联动，并禁止 `< 2,048`（8 槽档）样本向 `>= 2,048`（24 槽档）跨档外推显存需求。

### 1.2 DeepSeek V4 加速包（`xtier/packs/deepseek_v4/`）

1. **SM120 PTX 精确 `mhc.post` 与 24-bit `bf16x3` mHC 投影（`mhc_sm120.py`）**：
   - 上游 `deepseek_v4_mhc.post` 在 `T > 384` 回退 PyTorch eager 广播，在 `T = 49,152` 时物化 `[49152, 4, 4, 5120]` 临时张量（**单次 `60.09 ms` + 瞬时吃掉 `+22.50 GiB` 显存**）；新增内联 PTX `mul.rn.f32 + add.rn.f32` 精确核将其压至 **`4.36 ms`**（逐位一致，彻底消除显存尖峰）。
   - 挂载 `refresh_mhc_norm_weight_cache` 与 `_mix_stats_impl`（`4,096 <= m <= 65,536`），在不触碰非融合边界层的前提下启用 SM120 纯 Triton 3-片 24-bit 精确拆分核 `hc_mix_stats_sinkhorn_bf16x3`（`5.54 → 1.94 ms/次`，每 48K 块省约 `280 ms`，相对误差 `5.1e-07`）。
2. **SM120 CUTLASS Dense MXFP8 预填自调优与路由分流（`mhc_sm120.py`）**：
   补齐 `autotune_prefill_kernels` 对 `flashinfer_cutlass` 的调优，并将 `< 512` 行留给低延迟 `b12x`、`>= 512` 行路由至自调优 `cutlass`（40 层 `M=49,152` 从 `1,155.6 ms` 降至 **`815.3 ms`**，逐位一致）。
3. **`wo_a` L2 分组 BF16 预填核、中间块 SWA 剪枝与 640 阈值（`wo_a.py`、`__init__.py`、`configs/ds41-pro5000-nps4/`）**：
   新增 `(128, 256, 64)` L2 分组 Triton BF16 点积核替代整层 80 MiB BF16 反量化 + `bmm`（单层 `23.37 → 15.67 ms`，逐位一致，省 `768 MiB` 峰值显存），解码分组上限 `64 → 96` 行；非末尾且非 `return_logprob` 的中间 48K 分块跳过冗余 `L20–39` 128-tok SWA 重放；无冷专家时将 `XTIER_HYBRID_PREFILL_MAX` / `XTIER_GPU_PREFILL_MIN` 调至 `640`（`packs='off'` 基线保持原配置）。

### 1.3 A/B 实测证伪与排除项

- **`SLOT_TOKENS` 扩至 24（已回退）**：经 `sps_profile.py` 实测，扩至 24 会使 `pool.py` 密集槽核在 `m = 15..24` 切到 `BM = 32`，步时反慢 `21–28 ms`（`theta[18]: 33.0 → 53.9 ms`），且每层静态暂存翻倍多占 `1.6 GiB` 显存、吃掉 2 个弹性热专家槽（`+26 → +24`），故保持 `SLOT_TOKENS = 16`。
- **`FP4_ROWS = 12,288 / 16,384`（已排除）**：单层 20 层实测略慢于 `8,192` 且多占 `1.0 GiB` 显存。
- **跳过草稿词表头或目标词表头微损量化（已排除）**：违背 100% 无损红线。

---

## 2. DeepSeek V4.1：README 同口径两轮 `agent_bench` 与专项实测

启动台默认配置（48K 分块、35 槽 + 弹性至 68 槽、CPU 分担 prefill、变长验证、上下文草稿检索开、1M KV 池，`--prefill-decode-interval 72`）：

### 第 1 轮（98 个请求，可计时 98 个）

| 会话 | 上下文（起 → 止） | 冷 prefill：首字 s / tok/s | 请求数 | 输出 token | decode tok/s | 接受长度 | 每步 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 64k | 66K → 135K | 11.6 / 5,719 | 21 | 7,068 | 167.3 | 4.71 | 28.2 |
| 128k-write | 132K → 191K | 16.6 / 7,972 | 22 | 6,477 | 164.7 | 4.61 | 28.0 |
| 256k | 264K → 361K | 30.4 / 8,682 | 24 | 1,403 | 237.2 | 5.06 | 21.4 |
| 384k | 394K → 440K | 44.3 / 8,894 | 24 | 5,158 | 185.5 | 5.33 | 28.7 |
| 500k | 489K → 517K | 54.6 / 8,950 | 7 | 370 | 271.5 | 5.61 | 20.7 |
| **全部** | | | 98 | 20,476 | **175.5** | **4.86** | **27.7** |

### 第 2 轮（94 个请求，可计时 94 个）

| 会话 | 上下文（起 → 止） | 冷 prefill：首字 s / tok/s | 请求数 | 输出 token | decode tok/s | 接受长度 | 每步 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 64k | 66K → 96K | 8.6 / 7,713 | 17 | 5,786 | 165.9 | 4.66 | 28.1 |
| 128k-write | 132K → 167K | 15.5 / 8,543 | 24 | 5,214 | 175.4 | 4.70 | 26.8 |
| 256k | 264K → 378K | 30.3 / 8,726 | 24 | 1,281 | 272.5 | 5.34 | 19.6 |
| 384k | 394K → 415K | 43.4 / 9,073 | 23 | 5,215 | 180.2 | 5.17 | 28.7 |
| 500k | 489K → 519K | 54.7 / 8,930 | 6 | 317 | 240.5 | 5.20 | 21.6 |
| **全部** | | | 94 | 17,813 | **178.9** | **4.87** | **27.2** |

- **两轮汇总**：192 个请求全部成功（`cached_tokens` 192/192），全部请求 decode **177.1 tok/s**（接受长度 4.86，每步 27.4 ms）；24 个长回复（≥ 300 token）decode **159.0 tok/s**（第 1 / 2 轮 `159.5` / `158.5 tok/s`，接受长度 4.70，每步 29.5 ms）。另一组两轮 `agent_bench`（`V018_gold`）第 2 轮 256K / 384K / 500K 冷 prefill 分别达 **9,142 / 9,020 / 9,085 tok/s**。
- **专项与多轮工具回传**：458K 单次冷提示端到端 `49.89 s`（**9,179.6 tok/s**，稳态 48K 单块峰值 **9,998.6 tok/s**）；`context_speed.py` 65K / 520K 达 **7,695 / 8,874 tok/s**；工具回传 `m = 394 / 611 / 743 / 917` 首字分别压至 **0.92 / 1.16 / 1.12 / 1.17 s**（`m = 743` 自 `2.07 s` 提速 **1.86×**）。

---

## 3. 全量门禁、精度与跨模型回归

- **静态与消毒器**：`repo_check.py`（0 findings）、`console_check.py`（2,808,192 组合）、`planner_check.py`、`check_sglang.py --strict`（原生 SM120 + `sm86` + `sm100`）全过；`engine_san.sh thread` 与 `address` 均 **0 sanitizer reports**；58 项 Section 2 单测全数 PASS。
- **精度与鲁棒性**：DeepSeek 13 段 `ppl_eval.py` **NLL 1.20031**（v0.1.7 为 1.1990，差 +0.0013）；1/2/4 路 `copy_test.py` 正文 0 错；`abuse_test.py` 44/44 全过；`XTIER_DEBUG_SEQ_START=4294962296` 序号回绕正常；GLM-5.3 双路照抄 **0/8**、13 段 PPL **NLL 1.38531**（v0.1.7 为 1.3866），全程服务日志 0 错误。
