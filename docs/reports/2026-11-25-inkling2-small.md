# Inkling-2-Small 零补丁支持、极限压榨与解码/预填剖析（2026-11-25，v0.1.10）

XTier v0.1.10，官方 SGLang main `4ce41378f9`（零源码改动），单卡 RTX PRO 5000 72 GB（SM120，`73415 MiB`）+ 2× EPYC 9334（64C/128T NPS4）+ 377 GiB DDR5 + 4× Samsung 9100 PRO PCIe 5.0 NVMe。目标 2 并发（`MAX_RUNNING_REQUESTS=2`），512K 上下文。

## 1. 精度方案与核心优化

- **模型规模**：`thinkingmachines/Inkling-2-Small`（594B 总参 / 17B 激活，48 层 MoE = 40 层 NVFP4 `L0..39` + 8 层 MXFP8 `L40..47`，384 专家 top-8，`H=4608, I=2304`，`SiTU(4.0/25.0)`，40 SWA 窗口 1024 + 8 全局注意力 `L5,11,17,23,29,35,41,47`，23 层 ViT 多模态，原生 5 层 DSpark MTP `block_size=8`）。48 层 MoE 原生权重 350.3 GiB（NVFP4 256.3 + MXFP8 94.0）。
- **确认精度组合**：
  - **A1（MoE Prefill 激活）**：`L0..39` NVFP4 `W4A4 4over6 MAE` + `L40..47` MXFP8 cuDNN；Decode 保持原生 `W4A16/W8A16`。
  - **A2（KV Cache）**：Target 48 层 NVFP4 KV（`72 B/head`，3.56× 压缩，512K 仅占 `6.75 GiB`）；MTP 5 层 Draft KV 保持 BF16（`2.80 GiB`）。
  - **A3（Dense + MTP）**：116 层全量 FP8 权重 + W8A8 Prefill（96 层 Target `qkvr`/`wo_ud` + 20 层 MTP）。
- **关键工程修复与加速**：
  1. CPU 引擎与信箱新增 MXFP8 + SiTU 支持与零内存全驻 GPU 层（`XTIER_CACHE_FULL` 在无冷专家时直写 GPU 槽、免除主机内存占用）。
  2. 修复 `_copy_stream` 异步 DMA Use-After-Free（`xtier/backend/prefill.py`）、`cold.relink()` 字母序预读链（GPU Prefill NVMe 等待 `0.0 s`）、`fp8_weights.py` 大矩阵 `tl.int64` 溢出，以及 Hook `_mix_norm` 绕过上游 Triton `int32` 溢出，解锁 **64K 单块 Prefill（`BATCH=65664`）**。
  3. Global Attention 3 流切片投影（减 `2.6×` 计算、省 `3.3 GiB` 峰值显存）；NVFP4 全局注意力远区无掩码解码核（500K 8 层 `14.70 → 11.31 ms`）；FA4 `_shear_bias_workspace` 精准回收；零补丁 DSpark `OnlineAcceptanceEstimator` 在线截断（`0.20`）+ Recall；启动台新增每模型 `guard_gib`（Inkling-2 设为 `8 GiB`）。

## 2. 方案对比（基线 vs 方案 A+ / A / B2 / B1）

| 指标 | **方案 A+（默认）**<br>`cold=2%, 45 槽, 64K` | 方案 A（稳妥档）<br>`cold=8%, 44 槽, 64K` | 方案 B2（`cold=off`）<br>`仅 L47 全驻, 35 槽, 48K` | 方案 B1<br>`仅 L0 全驻, 64K` | **基线（关加速包）**<br>`W4A16/W8A16, BF16 KV, 16K, 27 槽` |
|---|---:|---:|---:|---:|---:|
| 全驻 GPU 层（显存） | 0 | 0 | 1 层 `L47`（11.75 GiB） | 1 层 `L0`（6.41 GiB） | 0 |
| 内存常驻专家 | 376/384（97.9%） | 353/384（91.9%） | 384/384（100%） | 384/384 | 353/384 |
| 启动后 / 500K 后可用内存 | **15 / 12 GiB** | 35 / 30 GiB | **20 / 13 GiB** | 7 GiB（破 8G） | ~30 GiB |
| 稳态显存 / 弹性热槽 | **70.65 / 73.4G（45 槽）** | 69.3 / 73.4G（44 槽） | 70.53 / 73.4G（35 槽） | 64K OOM（缺 3.28G） | 27 固定槽 |
| 启动耗时 | **126 s** | 120 s | 420 s | 410 s | ~390 s |
| Decode NVMe 读盘 | **0.2–0.7 / 步** | 2.3–4.6 / 步 | 0 | 0 | — |
| 短文 `bs=1` / `bs=2` 合计 | **74.7**（首轮 81.7） / **78.4** | 71.1 / 79.0–83.3 | 82.9（首轮） / 72.6 | OOM | 47.9 / ~43 |
| 64K 冷 Prefill（tok/s） | **3,835**（17.1 s） | 3,815（17.2 s） | 2,846（23.1 s） | OOM | 1,591（41.3 s） |
| 256K 冷 Prefill（tok/s） | **2,464**（106.4 s） | 2,465（106.3 s） | 2,215（118.3 s） | OOM | 1,217（215.3 s） |
| 500K 冷 Prefill（tok/s） | **1,593**（313.9 s） | 1,603（311.8 s） | 1,512（330.7 s） | OOM | OOM |
| 64K / 256K / 500K 缓存 Decode | 52.6（峰 66.7）/ 峰 49.5 / 43.7（峰 47.0） | 58.5 / 51.4 / **47.0**（峰 51.1） | 56.8 / 52.0 / 40.1 | OOM | 43.4 / 33.8 / OOM |

- **两轮 `agent_bench`（64K→513K，91/91 命中缓存，方案 A）**：长回复 **`66.2 tok/s`**（全部 33 可计时请求 **`66.3 tok/s`**，接受长度 **`4.02`**，`60.6 ms/步`）；64k / 128k-write / 256k / 384k / 500k 会话冷 prefill 分别为 `2,628–2,679` / `2,755–2,771` / `2,400–2,404` / `1,908–1,910` / `1,603 tok/s`，长回复 decode 分别为 `73.6–75.9` / `73.8–77.1` / `57.0–65.8` / `59.8–60.9` / `60.8–62.9 tok/s`。
- **精度与多模态全检**：13 段 PPL/KL（53,235 位置）`all.nll = 0.99955` 对基线 `0.99531`（`ΔNLL +0.00423`、`KL 0.00938`）；`copy_test`（`T=0` / `T=0.01`, `c=1/2`）全 `0/8`；256K 与 500K 5 深度捞针全 `5/5`；思考 / 工具调用 / 23 层 ViT 视觉冒烟全通过。

## 3. 解码步剖析（实机 GPU Chrome Trace，方案 A+ 默认）

测法：SGLang `/start_profile`（`activities=["GPU"]`）在 45 弹性热槽稳态下分别录制短文 `bs=1`（25 步）、短文 `bs=2`（25 步）、64K 缓存（25 步）、256K 缓存（20 步），逐层对齐 `_pack` → GPU 热专家（`_gate_up` + `_down` + `_combine`）→ `_unpack`（CPU 信箱完成）计算精确每步墙钟拆分：

| 组件（单步中位耗时） | 短文 `bs=1`<br>（52.04 ms） | 短文 `bs=2`<br>（58.33 ms） | 64K 缓存 `bs=1`<br>（45.58 ms） | 256K 缓存 `bs=1`<br>（43.76 ms） |
|---|---:|---:|---:|---:|
| **GPU 等 CPU（纯空转等待）** | **33.19 ms（63.8%）** | **27.53 ms（47.2%）** | **24.43 ms（53.6%）** | **14.38 ms（32.9%）** |
| ↳ NVFP4 40 层（等待层占比） | 26.33 ms（99.7%） | 22.86 ms（97.6%） | 19.39 ms（98.9%） | 12.41 ms（94.5%） |
| ↳ MXFP8 8 层（等待层占比） | 6.86 ms（96.0%） | 4.67 ms（89.0%） | 5.04 ms（96.5%） | 1.97 ms（76.9%） |
| **GPU 算热专家（与 CPU 并行）** | **5.36 ms（10.3%）** | **13.34 ms（22.9%）** | **4.84 ms（10.6%）** | **8.25 ms（18.9%）** |
| ↳ 对应 CPU MoE 总耗时（含重叠） | 38.14 ms（73.3%） | 39.89 ms（68.4%） | 28.99 ms（63.6%） | 23.06 ms（52.7%） |
| **48 层非 MoE（投影/注意力/Norm/SConv）** | 7.36 ms（14.1%） | 7.40 ms（12.7%） | 9.18 ms（20.1%） | **13.41 ms（30.6%）** |
| ↳ 其中 Attn + NVFP4 KV 解码 | 1.28 ms | 1.44 ms | 4.08 ms | **8.49 ms** |
| **目标 `lm_head` + 5 层 DSpark 草稿** | 5.56 ms（10.7%） | 5.60 ms（9.6%） | 5.84 ms（12.8%） | 6.47 ms（14.8%） |
| **`pack/unpack` + Host 调度间隙** | 0.78 ms（Host 0.07） | **5.33 ms（Host 4.33）** | 0.70 ms（Host 0.07） | 0.72 ms（Host 0.07） |
| 后台热专家换入 HtoD（完全重叠） | 17.44 ms | 16.43 ms | 15.14 ms | 17.20 ms |

**结论**：
1. **存在显著的 GPU 等 CPU**：48 层中 **95%–99.7%** 的层 GPU 算完热专家后都在空等 CPU。短文 `bs=1` 一步中 GPU 实际计算仅约 **19 ms（36%）**，纯等 CPU 占 **33.2 ms（64%）**；原因在于 Inkling-2 没有共享专家可与 CPU 重叠，且 384 专家每层仅 45 热槽（命中 38–53%），单专家体积大（NVFP4 18.4 MB / MXFP8 32.8 MB），24 通道 DDR5 读 12–16 个未命中专家需 500–970 µs/层。
2. **`bs=2` 额外暴露两处 Host / Kernel 开销**：尾端 Host 调度间隙从 `0.07 ms` 跳涨至 **`4.33 ms`**（非 CUDA 图的索引/采样/KV 元数据同步），且 8 层 Global Attention 3 流拼接 `3×16=48 > MAX_M(32)` 退回 `_w8a8_gemm` 多耗 `0.86 ms`。
3. **长上下文（256K→500K）**：GPU 等 CPU 占比随命中率上升回落至 ~33%，而 8 层 Global Attention NVFP4 KV 解码成为第二大开销（256K `8.49 ms`，500K `~11.3 ms`，占 ~20%）。

## 4. Prefill 过程剖析（64K 单块 → 500K 8 块）

结合 `prefill_64k` GPU Trace 与 Boot 18 的 500K 逐块计时：

- **64K 单块（实际墙钟 `17.1 s`，`3,835 tok/s`）**：
  - **48 层 MoE 仅占 `7.05 s`（41%）**：40 层 NVFP4 `~5.3 s`（GPU 计算 `3.54 s` + PCIe DMA 等待 `1.66 s`）+ 8 层 MXFP8 `~1.75 s`（cuDNN `0.64 s` + `_mxfp8_reduce_chunk 0.45 s` + DMA 等待 `1.03 s`）。350 GiB 专家过 PCIe 5.0 x16 理论底线已达 `~6.6 s`，MoE 已接近 PCIe 带宽上限。
  - **8 层 Global Attention FA4 = `2.83 s`（17%）**（`354 ms/层`），**40 层 SWA FA4 = `0.79 s`**（`19.7 ms/层`）。
  - **112 次 Triton `_w8a8_gemm` + `_act_quant_fp8` = `2.12 s`（12.4%）**（仅跑出 ~250–300 TFLOPS，远低于 SM120 cuBLASLt/CUTLASS FP8 算力）。
  - **杂项融合开销 = `~2.6 s`**：`ShearingBias`（48 次 `0.61 s`，其中 40 层 SWA 占 `0.51 s`）、`AttnRes`（`0.39 s`）、BF16 路由/MTP GEMM（`0.46 s`）、`RMSNorm`/`QKL2Clip`/`_gate_sink`（`0.37 s`）、`ShortConv`（`0.12 s`）。
- **256K（`106.4 s`）→ 500K（`313.9 s`）瓶颈反转**：
  - 500K 共 8 块，每块 MoE 恒定 **`7.05 s`**（8 块合计 `56.4 s`，仅占 **18%**）。
  - 非 MoE 部分随历史长度线性/二次增长：**第 1 块 `9.0 s` → 第 2 块 `16.0 s` → 第 4 块 `30.0 s` → 第 7 块 `55.9 s`**（每多 64K 历史增加 `+7.0~9.5 s`），其中 **8 层 Global Attention FA4 + NVFP4 KV 解包** 占 500K 总时间的 **65%–70%（~210 s）**。

## 5. 后续优化方向（按收益排序）

1. **Prefill（高收益、100% 无损）**：
   - **替换 Triton `_w8a8_gemm` 为 SM120 cuBLASLt / FlashInfer FP8 GEMM**：预计 `2.12 s → ~0.8 s/块`（64K 省 ~1.3 s / +8%，500K 省 ~10 s）。
   - **优化 `_reduce_band`（`671 ms`）+ `_mxfp8_reduce_chunk`（`449 ms`）+ `block_scale_interleave`（`172 ms`）**：融合归约与在冷专家模式下对 97.9% 内存专家预重排缩放，预计再省 `0.6–0.9 s/块`。
   - **解锁冷专家模式下的 `CPU_SHARE`**：对完全在内存中的分块放开 CPU 分担，预计每 64K 块减 6–10% PCIe 上传（再省 `0.4–0.6 s/块`）。
   - **长上下文（256K–500K）**：优化 8 层 Global Attention 的 NVFP4 KV 融合读取（若 FA4/FP8 KV 原生支持免 BF16 解包）或结合层间共享 `ShearingBias` 缓存（省 `0.5 s/块`）；根本上 500K 67% 耗时已在 SM120 FA4 稠密 `O(N^2)` 算力墙上，除非采用稀疏注意力（有损）。
2. **Decode（突破 CPU 内存带宽墙）**：
   - **短期无损工程收益**：①修复 `bs=2` 尾端 **`4.33 ms` Host 同步间隙** + 解码期 8 层 Global Attention 切片投影（免 `3×16=48` 退回 `_w8a8_gemm`，省 `0.86 ms`），`bs=2` 可直接再提 `+8%–10%`；②引入 `tools/plan_auto.py` 按层路由集中度与 NVFP4/MXFP8 代价非均匀分配 45 槽（把更多热槽倾斜给高 miss/高代价层，减 `1–2 ms/步`）；③在 `cold=2%` 时为 97.9% 内存专家探索有限度 `XTIER_DECODE_MISS_GPU`（仅对 MXFP8 `L40..47` 或极低 CPU 负载时需权衡 PCIe 延迟 383 µs/专家）。
   - **物理天花板与硬件路线**：单步 `bs=1` 64% 耗时受限于 24 通道 DDR5 带宽读取 18–33 MB 大专家；无损条件下只有**加大显存（如 96 GB PRO 6000 或 `TP=2` 双卡将热槽翻倍至 100+ 槽）**或**允许轻微精损的动态尾部跳过（`XTIER_TAIL_SKIP`）**才能把 `bs=1` 从 ~70 tok/s 推上 100+ tok/s。
## 6. 无损 Prefill 优化实测（v0.1.10 之后）

针对第 4 节 Trace 暴露的瓶颈，落地以下 **100% 无损** 优化（零 SGLang 补丁）：

1. **融合 `RelLogitsProj + ShearingBias` Triton 核（`xtier/packs/inkling2/attn.py`，`XTIER_INKLING2_FUSED_SHEAR=1`）**：
   - 原实现先做 `einsum("thd,de->the")` 写出 `6.45 GiB` `rel_logits`（`350 ms`，144 次 32×32 WMMA），再由 CuTe `ShearingBias` 读回并重排写出 `8.06 GiB` `bias`（`614 ms`）。
   - 改为挂钩 `RelLogitsProj.forward` + `_flash_attn_fwd`（走原生 `rel_bias_presheared=True`），在单个 Triton 核内对每个 token 直接以 `[48, 16] @ [16, 256]` Tensor Core `tl.dot` 计算并按 sheared 坐标写入 `_shear_bias_workspace`。
   - 单层 64K 实测 `21.76 → 8.30 ms`（**2.62×**，48 层直省 **645 ms/64K 块**），SWA/Global/256K paged/2 并发不规则 batch/非连续 view 5 组边界单测均为 `inf_match=True`、`max_diff=0.00e+00` 逐比特一致，且直接省去 **`6.45 GiB` 中间显存**（64K prefill 峰值需求从 `43.9 → 37.0 GiB`）。
2. **显存感知动态 1-band `_reduce_band`（`B=512`）+ `_mxfp8_reduce_runs` 活跃 run 归约（`xtier/backend/prefill.py`）**：
   - 取消 `total_gpu <= 393216` 硬阈值，改为按当前可用显存留 `4.5 GiB` 裕量自动单 band 归约（消除 64K 8 轮 FP32 `acc` 往返）并将瓦片扩宽至 `B=512`。
   - MXFP8 以 `_mxfp8_reduce_runs` + `torch.sort` 替代每子批 `_hist_rows`/`cumsum` + 全 token 扫描，消除空 CTA，FMA 精度 1-ULP 对齐。
   - 提升暂存深度 `XTIER_PREFILL_FP4_SLOTS=12`、`XTIER_PREFILL_MXFP8_SLOTS=8`；单层 64K 实测 NVFP4 `132.8 → 129.5 ms`、MXFP8 `228.4 → 222.7 ms`，均已贴近 PCIe 5.0 x16 物理线速（`52–54 GB/s`）。
3. **向量化 `_act_quant_fp8`（`ROWS_PER_CTA=8`）+ SM120 `_w8a8_gemm` `stages=4`（`xtier/packs/glm5_next/fp8_weights.py`）**：
   - 逐比特 `0` 差异，量化提速 `1.58×`、SM120 W8A8 达 `283.8 TFLOPS`（实测已优于 FlashInfer CUTLASS / DeepGEMM / `mm_mxfp8` 在 RTX PRO 5000 上约 272–275 TFLOPS 的硬件上限）。
4. **精确 slab `cudaHostRegister` `_share_buffers` + 冷专家 `CPU_SHARE` 解锁（`xtier/backend/prefill.py`）**：
   - 消除 `pin_memory=True` 扩容时 PyTorch `CachingHostAllocator` 按 2 的幂永久锁住 3–6.5 GiB 主机内存的问题；解锁冷专家模式下对 100% 驻留内存块（`0 .. C//CH - 1`）的 CPU 分担（单测 `nvme_reads_delta=0`）。
   - 实测消融确认：Inkling-2 48 层 MoE 已贴近 PCIe 5.0 物理线速且 384 专家分布较均衡，开 112 线程 CPU 分担会与 PCIe 5.0 DMA 争用 24 通道 DDR5（单块上传 `2.6 → 3.43 ms`），因此 Inkling-2 默认关闭 `XTIER_PREFILL_CPU_SHARE=0`（仍保留为启动台可选项，并惠及高稀疏路由模型）。

| 指标（Inkling-2 Plan A+ 默认） | v0.1.10 | 无损 Prefill 优化后 | 变化 |
|---|---:|---:|---:|
| 64K 冷 Prefill（首次 / 稳态块） | 3,835 tok/s（17.10 s） | **3,986 tok/s（16.46 s）** / 稳态 ~4,250 | **-0.64 s（+3.9%）** |
| 256K 冷 Prefill | 2,464 tok/s（106.4 s） | **2,497 tok/s（105.0 s）** | **-1.4 s（+1.3%）** |
| 500K 冷 Prefill | 1,593 tok/s（313.9 s） | **1,627 tok/s（307.3 s）** | **-6.6 s（+2.1%）** |
| 64K Prefill 峰值显存需求 | 43.9 GiB | **37.0 GiB** | **-6.9 GiB** |
| 实测全程峰值显存 / 最低可用内存 | 72.3 GiB / 12 GiB | **71.9 GiB / 9.2 GiB**（同机含 1.8M 语料客户端） | 0 OOM 重试 |
| 照抄（`T=0.01`，`c=1 / c=2`） | `0/4` / `0/4` | **`0/4` / `0/4`** | 100% 无损 |
| 13 段 PPL（docs/agent/zh） | `2.7171`（NLL `0.99955`） | **`2.7172`（NLL `0.99960`）** | ΔNLL `+0.00005`（零漂移） |
