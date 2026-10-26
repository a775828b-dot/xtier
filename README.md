# XTier

**Cross-tier CPU/GPU hybrid MoE inference for [SGLang](https://github.com/sgl-project/sglang).**
XTier is a plugin for upstream SGLang (tracking `main`): MoE experts live in GPU memory and host DRAM, are placed by how often
they are used and computed where they are, while NVMe holds only the rows of large embedding tables (DeepSeek's Engram, Qwen's
PLE). It wraps SGLang's standard `FusedMoE` quantization methods, so any MoE model
that uses the standard layer can run in hybrid mode without model-specific code; optional per-model acceleration packs add
model-specific optimizations. **Status: early development** — Qwen3.8 Flash Next, GLM-5.3 Flash and DeepSeek V4.1 Flash 1017
run on SGLang main, MiniMax-M3.1-Flash on SGLang's M3.1 branch (PR #45925); see [docs/PLAN.md](docs/PLAN.md).

---

XTier 是官方 SGLang 的插件（跟 main 分支），做 CPU/GPU 分层 MoE 推理：专家分布在显存和内存两层，按热度放置、按需调度；NVMe 只放大嵌入表的行（DeepSeek 的 Engram、Qwen 的 PLE）。

- **不改 SGLang 源码**：只用官方插件框架的钩子（接在官方 KTransformers 混合推理的入口上），加载后自检是否接管成功。
- **与模型无关**：混合推理包在官方 `FusedMoE` 的量化方法外面，用标准 MoE 层的模型装上就能跑（NVFP4、MXFP4 专家）。
- **加速包可选**：某个模型独有的修正和内核放在 `xtier/packs/` 下。

## 结构

```
xtier/
├── backend/   混合 MoE 后端：接管官方量化方法，权重加载完即交给 CPU 引擎；解码走信箱（CUDA 图可捕获），prefill 走 GPU 分组 GEMM 或 CPU
├── cache/     热专家缓存：每层槽数可不同，全部常驻的层不经过 CPU；自适应换入换出；常驻层的 prefill 直接在槽上算
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
tests/         单层测试（引擎、信箱、缓存、prefill、格式适配、hc_mix、投机截断、QSA NVFP4、GPT-OSS SwiGLU、wo_a、MiniMax NVFP4 KV）、启动台与规划器检查
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

## 启动台

![XTier 启动台](docs/images/console.png)

在仓库根目录运行 `python3 -m xtier.console`（只用 Python 标准库），打开 http://127.0.0.1:8100/ ：

- **左边模型栏**：每个模型一张小卡片，写着端口和 README 实测速度，标签注明测法（AGENT = agent 速度，文本 = 文本速度），
  模型多了在栏内滚动。
- **中间启动参数**：默认就是实测配置，改过的项会标出。公共的有加速包开关（关 = 测速表里「官方 main」的配置）、投机解码、上下文
  （MiniMax 固定 512K，没有这一项）、CPU 专家线程；各模型自己的有 Qwen 的 KV 精度、分块、常驻层布局、PLE 表放内存还是 NVMe，GLM 的拒绝采样、热专家槽、分块，
  DeepSeek 的 Engram 行缓存、热专家槽、分块，MiniMax 的热专家槽、DSpark 草稿长度、分块。
- **右边运行状态**：加载阶段（权重 → 草稿 → 专家池 → CUDA 图 → 预热 → 在线），在线后显示 decode 速度、接受长度、显存和日志。

一次只跑一个模型；模型服务在自己的端口上提供 OpenAI 兼容接口，启动台退出不影响它。加模型只要在 `configs/<模型>-<硬件>/` 里
加一个 `console.py`（卡片、选项、启动命令）。细节见 [docs/console.md](docs/console.md)。

## 支持的模型

| 模型 | 官方模块 | 状态 |
|---|---|---|
| Qwen3.8 Flash Next | `qwen4_exp` | 官方 main 上跑通：精度、照抄、512K 召回与生产一致；加速包比官方 main 冷 prefill 快 5% / 24%（64K / 512K）；KV 存 NVFP4（与生产同参，加速包让官方 QSA 能读），省下的显存用来多放常驻层，启动台默认由规划器自动布局；PLE 表（48 GiB）可放 NVMe，放傲腾时速度与锁页表持平 |
| GLM-5.3 Flash | `glm5_next` | 官方 main 上跑通（含 EAGLE）：加速包（FP8 权重、热词表拒绝采样、截断验证、弹性热专家槽）比官方 main 长回复每步快 21%，PPL 3.987 → 3.998 |
| DeepSeek V4.1 Flash 1017 | `deepseek_v4` | 官方 main 上跑通（含 DSpark）：Engram 表（189 GB）的行放 NVMe（[设计](docs/design-tables-nvme.md)），照抄 0 错；加速包（含 FP8 wo_a、分页索引器、弹性热专家槽）比官方 main 长回复每步快 39%、冷 prefill 快 33% |
| MiniMax-M3.1-Flash | `minimax_m3` | SGLang 的 M3.1 分支（PR #45925，还没合入 main）上跑通（含 DSpark）：SM120 上四处必需的修正（PR 把 SM120 当 SM100：原生栈、FP8 KV + TRT-LLM、MegaMoE、CUDA 上的 DSpark verify），CPU 引擎和 GPU 内核加了 GPT-OSS SwiGLU；KV 存 NVFP4，固定 512K 上下文（480K 找数 5/5）；按 agent_bench（64k 会话），加速包后 decode 每步快 34%（55.4 对 84.1 ms），tok/s 快 12%（含无损的 DSpark 按置信度截断验证）；没有生产管线可比 |

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
- 不开弹性热专家槽：Qwen 的 decode 受 GPU 限制；多放两层常驻、池化层的槽全部动态补回的布局 prefill 只快约 1%，decode 每步慢 10%，
  见[报告](docs/reports/2026-10-26-elastic-cache.md)。
- 加速包精度：轻微——KV 存 NVFP4（与生产相同，每 16 个值一个 E4M3 缩放；读出与官方反量化逐位一致），照抄 0 错、PPL 3.481；hc_mix
  换成重叠的持久内核、PLE 层与 QSA 索引器分段与官方逐位一致。
- 和生产管线（lsglang_q3.8fn）的 agent 对比见[报告](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-qwen38-xtier-r{1,2}-20261026.json`，官方 main `bench/results/context-qwen38-xtier-base-20261023.json`。
- 自动布局（`LAYOUT=auto`，2026-10-25 起默认）对 26 层常驻：同显存，五个会话的冷 prefill +2.4–2.9%，decode 每步持平，见
  [reports/2026-10-24-planner.md](docs/reports/2026-10-24-planner.md)。

### GLM-5.3 Flash

| EAGLE | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 2,485 | 2,674 | 95.7 | 2.52 | 26.3 ms |
|  | 2 | 2,729 | 2,662 | 101.6 | 2.82 | 27.8 ms |
| 官方 main | 1 | 2,663 | 2,518 | 76.8 | 2.61 | 34.4 ms |

- 加速包（2026-10-26 重测，启动台默认，含弹性热专家槽）对官方 main：长回复每步 27.1 对 34.4 ms（快 21%），冷 prefill 500K 2,668 对
  2,518（+6%）；64K 第 1 轮是启动后的第一个请求（2,485），第 2 轮 2,729 对 2,663。弹性热专家槽（decode 时多 11–13 槽 / 层）把每步
  从 27.5 降到 27.1 ms，见[报告](docs/reports/2026-10-26-elastic-cache.md)。官方 main 用 BF16 权重、55 槽（60 槽时 prefill 显存不够）、
  不开拒绝采样。
- 加速包精度：轻微——BF16 注意力 / 稠密层存成 FP8（每行每 128 列一个 scale），PPL 3.987 → 3.998（+0.3%）；拒绝采样与截断验证无损
  （输出分布与不用投机解码相同），照抄 0 错。
- 和生产管线（lsglang_glm5.3）的 agent 对比见[报告](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-glm53-elastic-r{1,2}-20261026.json`（不开弹性：`agent-bench-glm53-xtier-r{1,2}-20261024.json`），
  官方 main `bench/results/context-glm53-xtier-base55-20261023.json`。

### MiniMax-M3.1-Flash

SGLang 的 M3.1 分支（PR #45925，还没合入 main）+ XTier，固定 512K 上下文，KV 存 NVFP4，热专家每层 8 槽。只跑 64k 会话，
与之前 128K 池时的测法相同；更长的会话现在放得下，还没测。MiniMax 常在第一步就把 4096 个输出 token 全用来思考（没到工具调用
就到了上限），所以每轮是一个 64K 冷 prefill 加一到两个长回复，同一次启动连跑两轮。没有生产管线可比，对比的是官方代码加必要修正。

| DSpark | 轮 | 冷 prefill 64K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 1,854 | 53.4 | 3.04 | 56.9 ms |
|  | 2 | 1,877 | 50.0 | 2.70 | 54.0 ms |
| 官方代码 + 必要修正 | 1 | 1,824 | 46.8 | 3.94 | 84.3 ms |
|  | 2 | 1,883 | 45.5 | 3.82 | 84.0 ms |

- **官方代码 + 必要修正**（启动台「关加速包」）：PR 的代码加上在 SM120 上跑起来必需的修正（`xtier/packs/minimax_m3`：SM120 走 SGLang
  给 Hopper 的通用栈、CUDA 上的 DSpark verify、NVFP4 KV），检查点的草稿长度 7。**+ 加速包**：草稿长度 5、热专家换入节奏照
  DeepSeek（每 2 窗、每次最多 32 个）、草稿词表头 FP8、6 行验证的 MXFP8 投影走 b12x、DSpark 按置信度截断验证。
- 加速包每步快 34%（两轮 55.4 对 84.1 ms），tok/s 快 12%（51.6 对 45.9），接受长度短 26%（草稿 5 加截断，对比草稿 7）；
  冷 prefill 相同。
- 按置信度截断验证：草稿的置信度头预测每个草稿 token 被接受的概率，生存概率（累乘）低于 0.3 之后的草稿直接判拒绝（与 SGLang 的
  cap-accept 同一个机制，无损），这些行跳过 CPU 专家。CPU 每层从 798 降到 493–517 µs，decode +11%。
- 与之前的默认（128K FP8 KV、16 槽、不截断，今天复测：冷 prefill 1,900 / 1,953，decode 55.5 tok/s，每步 56.2 ms）相比：
  冷 prefill −3%，每步持平，tok/s −7%。512K 的 KV 从 FP8 的 42.5 GiB 降到 24 GiB（NVFP4 19 GiB，加 BF16 草稿 5 GiB），所以
  热专家从 16 槽减到 8 槽（命中率 0.43 → 0.27）。
- decode 受 CPU 专家限制：未命中的专家每个约 32 MB，读一个约 67 µs（约 470 GB/s），更多见
  [报告](docs/reports/2026-10-24-ds-woa-minimax.md)。
- 精度：NVFP4 KV 轻微。照抄（温度 0，单 / 双并发各 8 段）：单并发三次里有两次错同一个 token（前两名 logit 很接近），双并发都是
  0 错（FP8 都是 0 错）；480K 长文找数 5/5。截断验证无损（照抄 0 错）。MiniMax 的 PPL 不适合衡量精度：裸文本和聊天里用户
  文档的 NLL 都在 4–6，FP8 和 NVFP4 一样，见[报告](docs/reports/2026-10-24-minimax-512k.md)。CPU 引擎的 GPT-OSS SwiGLU
  与 FP32 参考的相对误差是 1e-6；草稿长度、换入节奏、草稿词表头 FP8 都不改变主模型的输出。
- 2026-10-26 用全部五个会话（64K–500K，两轮）重测加速包：长回复 43.4 tok/s、每步 69.2 ms（长上下文的会话更慢），500K 冷
  prefill 1,531 tok/s（`bench/results/agent-bench-minimax31-xtier-r{1,2}-20261026.json`）。弹性热专家槽在 MiniMax 上每步持平（一档
  额外槽 1.69 GiB，只加得上 3 档），启动台默认关，见[报告](docs/reports/2026-10-26-elastic-cache.md)。
- 原始数据 `bench/results/agent-bench-minimax31-{xtier-512k,base-512k,xtier-fp8kv-128k}-r{1,2}-20261024.json`、
  `needle-minimax31-512k-20261024.json`。

### 配置

- **官方 main**（启动台「关加速包」）：XTier 后端加上跑起来必需的修正，不开任何提速项。必需的修正：Qwen 的 PLE 表锁页分配；GLM 的量化排除、DSA
  别名、NextN 草稿量化；DeepSeek 的 SM120 候选索引器、Engram 行放 NVMe、尾部重放不走 CUDA 图；
  MiniMax 的 SM120 通用栈、CUDA 上的 DSpark verify 和 NVFP4 KV（`xtier/packs/minimax_m3`；512K 时 FP8 KV 放不下）。
- **热专家布局与启动参数与生产相同**：Qwen NVFP4 KV、其余层每层 155 槽；GLM 每层 60 槽、16K 分块、BF16 SSM 状态、拒绝采样；
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

## 版本更新

- 2026-10-26：弹性热专家槽：decode 时把只有 prefill 才用的显存借给热专家，prefill 前还回（DeepSeek 每步 −4.3%、GLM −1.5%，
  启动台默认开）；启动台报在线前先预热，换模型后的第一个请求 88 → 6–7 s。详见[文档](docs/reports/2026-10-26-elastic-cache.md)。
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
