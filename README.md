# XTier

**Cross-tier CPU/GPU hybrid MoE inference for [SGLang](https://github.com/sgl-project/sglang).**
XTier is a plugin for upstream SGLang (tracking `main`): MoE experts live in GPU memory and host DRAM, are placed by how often
they are used and computed where they are, while NVMe holds only the rows of large embedding tables (DeepSeek's Engram, Qwen's
PLE); when host memory cannot hold all experts, an optional cold-expert pool keeps the rarely used ones on NVMe and reads
them back when they are routed (its size and speed are bound by the drives' speed and type). It wraps SGLang's standard
`FusedMoE` quantization methods, so any MoE model
that uses the standard layer can run in hybrid mode without model-specific code; optional per-model acceleration packs add
model-specific optimizations. **Status: early development** — Qwen3.8 Flash Next, GLM-5.3 Flash and DeepSeek V4.1 Flash 1017
run on SGLang main; MiniMax-M3.1-Flash runs on SGLang's M3.1 branch (PR #45925), its development is paused (too slow here,
and not smart enough); see [docs/PLAN.md](docs/PLAN.md).

---

XTier 是官方 SGLang 的插件（跟 main 分支），做 CPU/GPU 分层 MoE 推理：专家分布在显存和内存两层，按热度放置、按需调度；NVMe 只放大嵌入表的行（DeepSeek 的 Engram、Qwen 的 PLE）。内存放不下全部专家时，可选的冷专家池把少用的专家放到硬盘、用到时读回内存，池子大小和速度受限于硬盘速度和类型。

- **不改 SGLang 源码**：只用官方插件框架的钩子（接在官方 KTransformers 混合推理的入口上），加载后自检是否接管成功。
- **与模型无关**：混合推理包在官方 `FusedMoE` 的量化方法外面，用标准 MoE 层的模型装上就能跑（NVFP4、MXFP4 专家）。
- **加速包可选**：某个模型独有的修正和内核放在 `xtier/packs/` 下。

## 结构

```
xtier/
├── backend/   混合 MoE 后端：接管官方量化方法，权重加载完即交给 CPU 引擎；解码走信箱（CUDA 图可捕获），prefill 走 GPU 分组 GEMM 或 CPU
├── cache/     热专家缓存：每层槽数可不同，全部常驻的层不经过 CPU；自适应换入换出；常驻层的 prefill 直接在槽上算；
│              冷专家池（可选，`XTIER_COLD=1`）：少用的专家放到硬盘（[设计](docs/design-cold-experts.md)）
├── planner/   规划器：本机探测的 GPU / CPU 耗时 + 路由档案 → 显存预算内每层的热专家槽数（按 CPU 和 GPU 同时算完分配），可在启动时自动算
├── engine/    CPU 引擎的 Python 绑定、绑核、锁页内存（C++ 源码在 csrc/）
├── packs/     加速包：qwen4_exp（PLE 表、PLE 行放 NVMe、hc_mix 重叠内核、QSA 读 NVFP4 KV）、glm5_next（量化排除、DSA、NextN）、
│              deepseek_v4（Engram 行放 NVMe、SM120 候选索引器、FP8 GEMM 调优配置、FP8 wo_a、recipe 分词、分页索引器）、
│              minimax_m3（SM120 通用栈、CUDA 上的 DSpark verify、草稿词表头 FP8、SM120 上读 NVFP4 KV）、
│              speculative（通用投机解码：EAGLE 的热词表配无损拒绝采样、草稿温度、按置信度截断验证；DSpark 按置信度截断验证）
├── tables/    大嵌入表的行放在 NVMe：多段多文件、任意行宽、行缓存、CUDA 图内查表
├── console/   启动台：一个网页选模型、调启动参数、启动 / 停止，跟踪加载阶段与 decode 遥测（只用标准库）
└── platform.py  与模型无关的主机修正（TileLang 目标选择）
csrc/          CPU 引擎：AVX-512 / AVX2 专家内核、NUMA 线程池、CPU 与 GPU 之间的信箱通信
configs/       每个"模型 × 硬件"的启动配置，console.py 是它在启动台里的条目（选项、默认值、启动命令）
tests/         单层测试（引擎、信箱、缓存、prefill、格式适配、hc_mix、投机截断、QSA NVFP4、GPT-OSS SwiGLU、wo_a、MiniMax NVFP4 KV、快速加载、未命中代算）、启动台与规划器检查
tools/         构建、内存看门狗、规划器（plan_probe 探测、plan_trace 路由档案、plan_auto 出方案、plan_calibrate 校准，见 docs/planner.md）
bench/         验收脚本（PPL、照抄、验证步、并发、prefill 延迟）与结果
```

## 快速上手

```bash
pip install -e /path/to/sglang/python          # 官方 SGLang main
tools/build_engine.sh /path/to/site-packages/nvidia/cu13   # CPU 引擎与信箱库 -> xtier/engine/lib
pip install --no-deps -e .                      # 注册插件入口 sglang.srt.plugins: xtier
XTIER=1 XTIER_THREADS=112 XTIER_MAIN_CPUS=auto XTIER_CACHE_SLOTS=155 \
  python -m sglang.launch_server --model-path <MoE 检查点> --cuda-graph-backend-prefill disabled ...
```

开关见 [xtier/config.py](xtier/config.py)；按模型和硬件的完整启动参数见 [configs/](configs/)，也可以用下面的启动台。

快速加载（可选，`XTIER_FAST_LOAD=1`；默认关，即 SGLang 原来的加载方式）：专家直接从模型文件（开冷专家时从冷存储）读进
CPU 引擎，全部读完、检查完才报在线，DeepSeek V4.1 Flash 1017 启动 472 → 186 s（[报告](docs/reports/2026-10-27-fast-load.md)）。

## 启动台

![XTier 启动台](docs/images/console.png)

在仓库根目录运行 `python3 -m xtier.console`（只用 Python 标准库），打开 http://127.0.0.1:8100/ ：

- **左边模型栏**：每个模型一张小卡片，写着端口和 README 实测速度，标签注明测法（AGENT = agent 速度，文本 = 文本速度），
  模型多了在栏内滚动。
- **中间启动参数**：默认就是实测配置，改过的项会标出。公共的有加速包开关（关 = 测速表里「官方 main」的配置）、投机解码、上下文
  （MiniMax 固定 512K，没有这一项）、CPU 专家线程；各模型自己的有 Qwen 的 KV 精度、分块、常驻层布局、PLE 表放内存还是 NVMe，GLM 的拒绝采样、热专家槽、分块，
  DeepSeek 的 Engram 行缓存、热专家槽、分块、冷专家卸载，MiniMax 的热专家槽、DSpark 草稿长度、分块。
- **右边运行状态**：加载阶段（权重 → 草稿 → 专家池 → CUDA 图 → 预热 → 在线），在线后显示 decode 速度、接受长度、显存和日志。

一次只跑一个模型；模型服务在自己的端口上提供 OpenAI 兼容接口，启动台退出不影响它。加模型只要在 `configs/<模型>-<硬件>/` 里
加一个 `console.py`（卡片、选项、启动命令）。细节见 [docs/console.md](docs/console.md)。

## 支持的模型

| 模型 | 官方模块 | 状态 |
|---|---|---|
| Qwen3.8 Flash Next | `qwen4_exp` | 官方 main 上跑通：精度、照抄、512K 召回与生产一致；加速包比官方 main 冷 prefill 快 5% / 24%（64K / 512K）；KV 存 NVFP4（与生产同参，加速包让官方 QSA 能读），省下的显存用来多放常驻层，启动台默认由规划器自动布局；PLE 表（48 GiB）可放 NVMe，放傲腾时速度与锁页表持平 |
| GLM-5.3 Flash | `glm5_next` | 官方 main 上跑通（含 EAGLE）：加速包（FP8 权重、热词表拒绝采样、截断验证、弹性热专家槽）比官方 main 长回复每步快 22%，PPL 3.987 → 3.998 |
| DeepSeek V4.1 Flash 1017 | `deepseek_v4` | 官方 main 上跑通（含 DSpark）：Engram 表（189 GB）的行放 NVMe（[设计](docs/design-tables-nvme.md)），照抄 0 错；加速包（含 FP8 wo_a、分页索引器、弹性热专家槽）比官方 main 长回复每步快 39%、冷 prefill 快 33% |
| MiniMax-M3.1-Flash | `minimax_m3` | **暂停开发**（2026-10-26，本机太慢、智力不强）。SGLang 的 M3.1 分支（PR #45925）上跑通（含 DSpark）：SM120 上的必需修正、GPT-OSS SwiGLU、NVFP4 KV（固定 512K）；加速包比官方代码长回复每步快 34% |

## 测速

单卡 RTX PRO 5000 72GB + 2× EPYC 9334（NPS4）+ 384 GiB DDR5，官方 SGLang main 8cd3c494（MiniMax 用 SGLang 的 M3.1 分支），单位 tok/s，表里只放最新一次的结果。

**agent 速度**：生产仓库的 agent_bench（[bench/agent_bench.py](bench/agent_bench.py)：lsglang_ds4.1 那份加两处健壮性修正，计时不变），
模拟编码代理（dsh / Claude Code 一类工具）：5 个会话从 64K 到约 500K，铺满 512K KV 池，多步调用工具，后续请求命中前缀缓存，
服务端默认采样，同一次启动连跑两轮。每张表比官方 main 和 XTier + 加速包。冷 prefill = 会话第一个请求里未命中缓存的 token / 首字；decode 只算
长回复（≥ 300 token，写代码、写总结），接受长度随内容变，比较两边先看每个验证步的耗时。表用 [bench/agent_table.py](bench/agent_table.py)
生成，每会话明细见[报告](docs/reports/2026-10-24-ds-woa-minimax.md)。

### DeepSeek V4.1 Flash 1017

| DSpark | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 3,812 | 4,768 | 141.8 | 4.50 | 31.7 ms |
|  | 2 | 5,027 | 4,844 | 140.8 | 4.30 | 30.6 ms |
| 官方 main | 1 | 3,318 | 3,627 | 65.5 | 3.25 | 50.9 ms |

- 加速包（2026-10-26 重测，启动台默认，含弹性热专家槽）对官方 main：长回复每步 31.2 对 50.9 ms（快 39%），冷 prefill 64K / 500K
  4,420 / 4,806 对 3,318 / 3,627（快 33% / 33%）。弹性热专家槽（decode 时把只有 prefill 才用的显存借给热专家，至多 68 槽）把每步
  从 32.6 降到 31.2 ms，见[报告](docs/reports/2026-10-26-elastic-cache.md)。
- 官方 main 的 DeepSeek 工具调用解析不逐 token 流式返回参数，工具调用请求量不出 decode（两轮 177 个请求里能计时 13 个），所以只比
  长回复；加速包第 1 轮可计时的长回复只有 3 个，tok/s 偏低。
- 加速包精度：无损——块 FP8 GEMM 的分块配置和内核、Engram 提前查表（取到的行不变）、草稿词表头 FP8（只影响草稿提案，输出仍由主模型
  验证）、wo_a 存检查点的 FP8（还原逐位一致，kernel 与 cuBLAS 只差求和顺序）、索引器走分页路径（logits 与非分页路径逐位相同，top-k
  只在同分时可能选到不同位置），PPL 3.319 → 3.317（分页索引器之前测），照抄 0 错。
- 和生产管线（lsglang_ds4.1）的 agent 对比见[报告](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-ds41-elastic-r{1,2}-20261026.json`（不开弹性：`agent-bench-ds41-xtier-paged-r{1,2}-20261024.json`），
  官方 main `bench/results/context-ds41-xtier-base-20261023.json`。

### Qwen3.8 Flash Next

| MTP | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 5,715 | 5,170 | 181.7 | 2.57 | 14.2 ms |
|  | 2 | 5,610 | 5,150 | 207.2 | 2.68 | 13.0 ms |
| 官方 main | 1 | 5,319 | 4,078 | 139.5 | 2.13 | 15.2 ms |

- 加速包（2026-10-26 重测，启动台默认：规划器的自动布局，29 层常驻 + 18 × 64–88 槽）对官方 main：长回复每步 13.6 对 15.2 ms
  （快 11%），冷 prefill 64K / 500K 5,663 / 5,160 对 5,319 / 4,078（快 6% / 27%，QSA 索引器和 PLE 层分段主要省在长上下文）。官方 main
  用 FP8 KV、16K 分块、生产的 25 层常驻（官方 QSA 读不了 NVFP4 KV）。工具调用逐 token 流式，所有请求都能计时。
- 弹性热专家槽默认关：Qwen 的 decode 受 GPU 限制，多放热专家会使 decode 变慢约 10%（多放两层常驻、池化层的槽全部动态补回：
  prefill 只快约 1%），见[报告](docs/reports/2026-10-26-elastic-cache.md)。
- 加速包精度：轻微——KV 存 NVFP4（与生产相同，每 16 个值一个 E4M3 缩放；读出与官方反量化逐位一致），照抄 0 错、PPL 3.481；hc_mix
  换成重叠的持久内核、PLE 层与 QSA 索引器分段与官方逐位一致。
- 和生产管线（lsglang_q3.8fn）的 agent 对比见[报告](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-qwen38-xtier-r{1,2}-20261026.json`，官方 main `bench/results/context-qwen38-xtier-base-20261023.json`。
- 自动布局（`LAYOUT=auto`，2026-10-25 起默认）对 26 层常驻：同显存，五个会话的冷 prefill +2.4–2.9%，decode 每步持平，见
  [reports/2026-10-24-planner.md](docs/reports/2026-10-24-planner.md)。

### GLM-5.3 Flash

| EAGLE | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 2,648 | 2,640 | 96.3 | 2.54 | 26.4 ms |
|  | 2 | 2,715 | 2,635 | 98.2 | 2.66 | 27.1 ms |
| 官方 main | 1 | 2,663 | 2,518 | 76.8 | 2.61 | 34.4 ms |

- 加速包（2026-10-26 重测，启动台默认：常备 58 槽 + 弹性热专家槽）对官方 main：长回复每步 26.9 对 34.4 ms（快 22%），冷 prefill
  64K / 500K 2,682 / 2,638 对 2,663 / 2,518（+1% / +5%）。常备槽 60 → 58：60 槽时 16K 分块的冷 prefill 贴着显存上限（同一晚两轮
  agent_bench 里分配器释放缓存重试 18 次，一次 25 万 token 的冷提示服务退出），58 槽只在预热时重试 1 次、每步持平（60 槽 27.6 ms），
  decode 时弹性槽把显存借回来（每层多约 12–14 槽），见[报告](docs/reports/2026-10-26-elastic-cache.md)。官方 main 用 BF16 权重、
  55 槽（60 槽时 prefill 显存不够）、不开拒绝采样。
- 加速包精度：轻微——BF16 注意力 / 稠密层存成 FP8（每行每 128 列一个 scale），PPL 3.987 → 3.998（+0.3%）；拒绝采样与截断验证无损
  （输出分布与不用投机解码相同），照抄 0 错。
- 和生产管线（lsglang_glm5.3）的 agent 对比见[报告](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-glm53-base58-r{1,2}-20261026.json`（60 槽：`agent-bench-glm53-base60-r{1,2}-20261026.json`；
  不开弹性：`agent-bench-glm53-xtier-r{1,2}-20261024.json`），官方 main `bench/results/context-glm53-xtier-base55-20261023.json`。

### MiniMax-M3.1-Flash

**暂停开发**（2026-10-26）：在本机太慢（agent 长回复 43 tok/s），智力也不强，代码和启动台卡片保留。SGLang 的 M3.1 分支（PR #45925，
还没合入 main）+ XTier，固定 512K 上下文，KV 存 NVFP4，热专家每层 8 槽；只跑 64k 会话（MiniMax 常把第一步的 4096 个输出 token
全用来思考），没有生产管线可比，对比的是官方代码加必要修正。

| DSpark | 轮 | 冷 prefill 64K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 1,854 | 53.4 | 3.04 | 56.9 ms |
|  | 2 | 1,877 | 50.0 | 2.70 | 54.0 ms |
| 官方代码 + 必要修正 | 1 | 1,824 | 46.8 | 3.94 | 84.3 ms |
|  | 2 | 1,883 | 45.5 | 3.82 | 84.0 ms |

- 加速包（草稿长度 5、DeepSeek 的换入节奏、草稿词表头 FP8、6 行验证走 b12x、DSpark 按置信度截断验证）对官方代码 + 必要修正
  （SM120 通用栈、CUDA 上的 DSpark verify、NVFP4 KV，草稿长度 7）：每步 55.4 对 84.1 ms（快 34%），tok/s 快 12%，冷 prefill 相同。
- 加速包精度：轻微——NVFP4 KV 照抄单并发偶有同一个临界 token 翻转（双并发 0 错），480K 找数 5/5；截断验证无损。PPL 不适合衡量
  MiniMax 的精度，见[报告](docs/reports/2026-10-24-minimax-512k.md)。
- 2026-10-26 用全部五个会话重测：长回复 43.4 tok/s、每步 69.2 ms，500K 冷 prefill 1,531；弹性热专家槽每步持平，默认关。
- 原始数据：`bench/results/agent-bench-minimax31-{xtier-512k,base-512k}-r{1,2}-20261024.json`、`agent-bench-minimax31-xtier-r{1,2}-20261026.json`。

### 配置

- **官方 main**（启动台「关加速包」）：XTier 后端加上跑起来必需的修正，不开任何提速项。必需的修正：Qwen 的 PLE 表锁页分配；GLM 的量化排除、DSA
  别名、NextN 草稿量化；DeepSeek 的 SM120 候选索引器、Engram 行放 NVMe、尾部重放不走 CUDA 图；
  MiniMax 的 SM120 通用栈、CUDA 上的 DSpark verify 和 NVFP4 KV（`xtier/packs/minimax_m3`；512K 时 FP8 KV 放不下）。
- **热专家布局与启动参数与生产相同**：Qwen NVFP4 KV、其余层每层 155 槽；GLM 每层 60 槽（启动台默认 58 槽 + 弹性，见 GLM 一节）、16K 分块、BF16 SSM 状态、拒绝采样；
  DeepSeek 每层 50 槽；MiniMax（没有生产管线）每层 8 槽、512K NVFP4 KV 池、16K 分块。
  不同的地方：Qwen 加速包那行 26 层常驻（比生产多第 27 层，用 NVFP4 KV 省下的显存，prefill +0.5–1.4%；10-25 起启动台默认由规划器
  在这份显存内重排成 29 层常驻 + 18 × 64–88 槽）、
  24K 分块（生产 32K：这里 32K 在 64K 冷 prefill 慢 3%），见[报告](docs/reports/2026-10-24-qwen-kv-ds-agent.md)；Qwen 官方 main
  用 FP8 KV（官方 main 的 QSA 读不了 NVFP4）、16K 分块、生产的 25 层常驻；GLM 的官方 main 用 BF16 权重，60 槽时 prefill 显存不够，
  减到 55 槽，也不开拒绝采样（官方 main 不允许热词表配拒绝采样）。
- **+ 加速包**：Qwen 是 hc_mix 重叠内核（`XTIER_HC_MIX_OVERLAP`）、PLE 层分段（`XTIER_PLE_BLOCK`）、QSA 索引器 prefill 按键分段
  （`XTIER_QSA_KEYS_PER_CTA`）和 QSA 读 NVFP4 KV（`XTIER_QSA_FP4`）；GLM 是 FP8 权重（`XTIER_FP8_WEIGHTS`，生产用的也是这一套）和通用投机解码包（热词表拒绝采样、
  草稿温度 ×0.8、按草稿置信度截断验证，截掉的 token 跳过 CPU 专家，`xtier/packs/speculative/`）；
  DeepSeek 是调好的块 FP8 GEMM 配置、MXFP8 走 FlashInfer CUTLASS（解码宽度走 b12x）、Engram 提前查表（`XTIER_ENGRAM_PREFETCH`）、
  草稿词表头 FP8（`XTIER_DRAFT_LMHEAD_FP8`）和 wo_a 存检查点的 FP8、decode 走 Triton 分组 einsum（`XTIER_DS_WO_A`）、索引器 decode 走官方的分页路径（`XTIER_DS_PAGED_INDEXER`）；
  MiniMax 是 DSpark 草稿长度 5（`GAMMA`，检查点 7）、热专家换入节奏照 DeepSeek（`XTIER_CACHE_EVERY=2`、`XTIER_CACHE_SWAPS=32`）、
  草稿词表头 FP8（与 DeepSeek 同一个钩子）、6 行验证的 MXFP8 投影走 b12x（`XTIER_MXFP8_B12X_ROWS`，DeepSeek 包的钩子）、
  DSpark 按置信度截断验证（`XTIER_DSPARK_TRUNCATE=0.3`，截掉的 token 跳过 CPU 专家，`xtier/packs/speculative/dspark_truncate.py`）。
- **平台修正**（`xtier/platform.py`，与模型无关，开不开加速包都在）：SGLang 只按 decode 的形状给 FlashInfer 的 GEMM 做自动调优，
  prefill 大小的 MXFP8 GEMM 查不到缓存，退回 SM120 上最小的 128×32 tile；256 行起改用 128×128（输出逐位相同，16384 行时快 1.5–5.7 倍）：
  MiniMax 64K 冷 prefill +22%；DeepSeek 的 MXFP8 调用被 SGLang 固定了 tactic（`pin_tactic`），不受影响。
- **接口参数与生产相同**：推理解析、工具调用解析（DeepSeek `deepseekv41`、GLM `glm47`、Qwen `qwen3_coder`、MiniMax `minimax-m3`）和缓存命中上报
  （`--enable-cache-report`：usage 里的 `prompt_tokens_details.cached_tokens`，dsh 一类客户端据此显示缓存命中）。

## 冷专家卸载到硬盘

内存放不下全部专家时，每层只把一部分专家放内存，其余放到 NVMe 上的冷专家池，路由到时读回内存再算（`XTIER_COLD=1`，默认关，
[设计](docs/design-cold-experts.md)）。冷存储存全部专家，每个专家按 NUMA 节点切成 2.35 MB 的片分到各块盘，一次缺失所有盘同时读；
池子大小和速度受限于硬盘的速度和类型。启动台的 DeepSeek 卡片有「冷专家卸载」（关 / 10% / 20% / 30%）。

测试用的冷存储是 **4 块 PCIe 4.0 NVMe**：傲腾 P5800X ×2、三星 990 PRO、HP FX700（无 DRAM），机器同上。

**四个模型都能用**，各按卸载 30% 跑过端到端回归（照抄、开 / 关冷专家逐 token 对比 logprob、15 分钟 dsh 会话：上下文填到约
20 万 token、召回、编码），全部通过：

| 模型 | 每层在内存的专家 | 冷存储（每块盘） | 照抄 / dsh 15 分钟 | 所需内存（不开冷专家） | 所需内存（卸载 30%） |
|---|---|---|---|---:|---:|
| DeepSeek V4.1 Flash 1017 | 269 / 384 | 72.2 GB | 0 错 / 18 次 0 失败 | 约 298 GiB | 约 224 GiB（−74） |
| GLM-5.3 Flash | 202 / 288 | 42.8 GB | 0 错 / 15 次 0 失败 | 约 174 GiB | 约 129 GiB（−45） |
| Qwen3.8 Flash Next | 358 / 512（CPU 上的 18 层） | 6.4 GB | 0 错 / 65 次 0 失败 | 约 121 GiB | 约 116 GiB（−5） |
| MiniMax-M3.1-Flash | 90 / 128 | 58.1 GB | 0 错 / 10 次 0 失败 | 约 227 GiB | 约 165 GiB（−62） |

所需内存 = 空闲时整机的可用内存（369 GiB）减去运行期间（含加载）可用内存的最低点（`tools/mem_guard.sh` 每秒采样
MemAvailable），不含显存；DeepSeek 不开冷专家取 6 次启动的中位数。Qwen 大部分层的专家在显存里，只有 CPU 上的 18 层卸载，所以省得少。

### DeepSeek V4.1 Flash 1017：卸载比例与速度

前六列是 4 块 PCIe 4.0 盘上的实测，后两列是换成 4 块 PCIe 5.0 盘的估算（还没有 PCIe 5.0 盘）。单位 tok/s。

| 卸载 | 放到硬盘的专家 | 每步缺失 | decode（每步 / 速度） | 64K 冷 prefill | 增量 prefill | PCIe 5.0 估算 decode | PCIe 5.0 估算 prefill（64K / 增量） |
|---|---:|---:|---|---|---|---|---|
| 不卸载 | 0 | — | 30.5 ms / 139.6 | 3,751 | 271（每次约 300 token） | 同不卸载 | 同不卸载 |
| 10% | 29 GB | 2.5 | 32.8 ms / 130（−7%） | 3,690（−1.6%） | 约 0 | 31.7 ms / 134（−4%） | 3,693（−1.5%）/ 约 0 |
| 20% | 58 GB | 8.8 | 37.8 ms / 113（−19%） | 3,658（−2.5%） | −14% | 34.9 ms / 122（−13%） | 3,676（−2%）/ −8% |
| 30% | 87 GB | 16.8 | 43.9 ms / 97（−31%） | 3,390（−9.6%） | −27% | 39.0 ms / 109（−22%） | 3,537（−6%）/ −15% |
| 40% | 116 GB | 32.4 | 约 58 ms / 73（约 −48%）* | 3,494（−6.9%） | −50% | 46.9 ms / 91（−35%） | 3,591（−4%）/ −33% |
| 50% | 144 GB | 48 | 73.8 ms / 58（−59%） | 3,302（−12%） | −61% | 54.7 ms / 78（−44%） | 3,487（−7%）/ −44% |

- 实测用 agent_bench 的 64K 会话：decode 是长回复的每步耗时，速度按每步折算（不卸载时每个 DSpark 验证步约出 4.3 个 token）；
  64K 冷 prefill 是会话第一个请求（66K token）；增量 prefill 是后续请求（每次新增约 300 token）的中位数。都是单次运行，有
  ±5–10% 波动（30% 和 40% 的长 prefill 大小颠倒就是波动）。
- \* 40% 这次只有 2 个长回复，实测 89.9 ms 偏高，表里按每次缺失约 0.86 ms 推算（30.5 + 32.4 × 0.86 ms）。
- PCIe 5.0 估算：4 块 PCIe 5.0 x4 盘，2.35 MB 大块按每块 10–12 GB/s 算，误差约 ±2 个百分点；拆分卡要支持 5.0，盘要有散热。
  冷模式每层同步一次的固定开销（12K 文本 prefill 各档都慢 6–7%）换盘也不变。
- 盘要做几分钟的持续读测试：无 DRAM 的 HP FX700 满速读约 100 秒后会塌到约 60 MB/s（停下约 7 分钟才恢复），一次缺失要等最慢的
  那块盘，一块这样的盘会拖住整个池子；几秒的短测看不出来。本节两张测速表（上表和下面的最小化运行）都没碰上塌速。
- 每档明细和精度（照抄 0 错，NLL 在两次启动之间的波动内）见[报告](docs/reports/2026-10-26-cold-experts.md)。

### 最小化运行：32 GB 显存 + 128 GB 内存跑 DeepSeek V4.1 Flash 1017

模拟一张 RTX 5090（32 GB）+ 128 GB 内存 + 4 块 NVMe（冷存储同样是上面这 4 块 PCIe 4.0 盘），启动台卡片「DeepSeek V4.1 Flash 1017 ·
最小化」：冷专家 70%（每层内存只放 384 个里的 115 个）、常驻热专家每层 4 槽 + 弹性、8K 分块、128K 上下文、DSpark。显存和内存都是
硬上限（GPU 仍是 RTX PRO 5000，只给它 32 GB）：`tools/vram_limit.py` 占住 32,000 MiB 以外的显存，服务所在的 systemd scope 设
`MemoryMax=128G` 且不许用 swap。内存带宽是本机的 24 通道；换成 8 通道（约 200 GB/s）速度也差不多（按日志估算：prefill 基本不变、
decode 约慢 10%），主要受限于硬盘速度。

| 单位 tok/s | 32K | 64K | 128K |
|---|---:|---:|---:|
| prefill（32 GB + 128 GB） | 1,186 | 1,165 | 1,155 |
| decode（32 GB + 128 GB） | 28.5 | 28.1 | 32.7 |
| prefill（正常配置：72 GB 显存，专家全在内存） | 3,886 | 4,609 | 4,655 |
| decode（正常配置） | 130.7 | 127.9 | 122.4 |

- 测法：真实文档拼成的提示，清前缀缓存后测冷 prefill（提示 token / 首字），再 decode 512 token（温度 0）；两次启动各两轮
  （prefill 取每次启动的第 2 轮）。内存峰值 110.5 GiB、显存 29.5 GB，照抄 0 错。
- 约为正常配置的 25%，瓶颈在读盘：prefill 每个 8K 分块都要把约 70% 的专家从 4 块盘重读一遍，decode 每步约 100 个缺失。详见
  [报告](docs/reports/2026-10-27-ds41-minimal.md)。

## 版本更新

- 2026-10-28：decode 时可把一部分未命中的专家交给 GPU 代算（可选，`XTIER_DECODE_MISS_GPU`，默认关；启动台三张卡可选），
  给 CPU 比 PCIe 慢的机器用，结果与命中的专家相同。详见[文档](docs/reports/2026-10-28-decode-miss-gpu.md)。
- 2026-10-27（下午）：快速加载（可选，`XTIER_FAST_LOAD=1`，默认关）：专家直接从模型文件或冷存储读进 CPU 引擎，仍是全部装好
  才报在线；DeepSeek 启动 472 → 186 s，卸载 30% 409 → 124 s。详见[文档](docs/reports/2026-10-27-fast-load.md)。
- 2026-10-27：启动台新卡片「DeepSeek V4.1 Flash 1017 · 最小化」：32 GB 显存 + 128 GB 内存（模拟 RTX 5090），70% 的专家放硬盘，
  prefill 约 1,160、decode 约 30 tok/s。详见[文档](docs/reports/2026-10-27-ds41-minimal.md)。
- 2026-10-26（晚）：冷专家卸载到硬盘（可选，`XTIER_COLD=1`），四个模型都能用；DeepSeek 卸载 30% 时 decode 每步 +44%。
  详见[文档](docs/reports/2026-10-26-cold-experts.md)。
- 2026-10-26：弹性热专家槽（DeepSeek 每步 −4.3%、GLM −1.5%）；GLM 常备槽 60 → 58；启动台报在线前先预热；MiniMax-M3.1-Flash
  暂停开发。详见[文档](docs/reports/2026-10-26-elastic-cache.md)。
- 2026-10-25：测速表改为官方 main 对加速包；Qwen 启动台默认用自动规划布局；热专家池可选用提示的路由预热（`XTIER_CACHE_PROMPT=1`，
  默认关）；修了 MiniMax 包钩子和 GC 冻结两个启动报错。详见[文档](docs/reports/2026-10-25-prompt-warmup.md)。
- 2026-10-24（晚）：自动规划器第一版：本机测出各层 GPU / CPU 耗时，在显存预算内逐层定热专家槽数（`XTIER_CACHE_PLAN=auto`）；
  Qwen 启动时自动布局冷 prefill +2.4–2.9%，其余模型保留手工布局。详见[文档](docs/reports/2026-10-24-planner.md)。
- 2026-10-24（下午）：MiniMax 固定 512K、KV 存 NVFP4（42.5 → 24 GiB），DSpark 按置信度截断验证（无损，agent decode +11%）。详见[文档](docs/reports/2026-10-24-minimax-512k.md)。
- 2026-10-24（上午，二）：SM120 上 prefill 大小的 MXFP8 GEMM 改用 128×128 tile（输出逐位相同），MiniMax 64K 冷 prefill
  1,606 → 1,963 tok/s。详见[文档](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（上午）：新模型 MiniMax-M3.1-Flash（SGLang M3.1 分支），加速包 agent 每步 70.6 → 59.5 ms；DeepSeek 索引器 decode
  走分页路径，每步 33.0 → 32.5 ms。详见[文档](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（清晨）：DeepSeek 的 wo_a 保持 FP8（每步 33.9 → 33.0 ms，显存 −1.25 GiB），提示按 deepseek-recipe 分词；修了 TP 1
  下 NCCL 临时申请显存导致的 OOM。详见[文档](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（夜）：测速统一用 agent 速度；启动参数补上生产的接口参数（缓存命中上报、工具调用解析）。详见[文档](docs/reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-24：控制台 `xtier/console/`：一个网页选模型、调参数、启动 / 停止，加模型只需加配置目录。详见[文档](docs/console.md)。
- 2026-10-24：Qwen 默认 NVFP4 KV，省下的显存多放一层常驻（冷 prefill +0.5–1.4%）；DeepSeek 按生产 agent_bench 实测冷 prefill 快
  10–25%。详见[文档](docs/reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-23（夜）：通用投机解码包：GLM 用热词表拒绝采样和置信度截断，decode 93.0 → 95.6 tok/s（生产 84.8）。详见[文档](docs/reports/2026-10-23-speculative-fp4kv.md)。
- 2026-10-23（晚）：512K prefill 追平生产：Qwen 125 → 102 s（生产 106 s），GLM 224 → 192 s（生产 201 s）。详见[文档](docs/reports/2026-10-23-prefill.md)。
- 2026-10-23（晚）：DeepSeek 移植 Engram 提前查表和草稿词表头 FP8，decode 每步 −8%。详见[文档](docs/reports/2026-10-23-ds-engram-prefetch.md)。
- 2026-10-23：README 测速表；加载不再换出，DeepSeek 启动 811 → 409 s。详见[文档](docs/reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 3 步：DeepSeek 跑通官方 main，通用 NVMe 表，GLM 的 FP8 权重，双卡准备。详见[文档](docs/reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 2 步：后端、热专家缓存、GPU prefill 在官方 main 上跑通 Qwen 与 GLM。详见[文档](docs/reports/2026-10-23-step2-qwen.md)。
- 2026-10-23：第 1 步：CPU 引擎迁入 `csrc/`，三个模型的真实层与 lkqmoe 0.5.6 逐位一致（`bench/results/engine-ab-20261023.json`）。

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
