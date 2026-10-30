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
and not smart enough); see [docs/PLAN.md](docs/PLAN.md). GPUs: developed and benchmarked on Blackwell
(SM120); the RTX 40 (SM89) and RTX 30 (SM86) series are supported, tested on an RTX 4090 and an RTX 3090
([RTX 40 report](docs/reports/2026-10-29-rtx40.md), [RTX 30 report](docs/reports/2026-10-29-rtx30.md)); two GPUs
run tensor parallel (TP=2), tested on two RTX 3090s ([TP report](docs/reports/2026-10-30-tp.md)); a second GPU of another
size or speed can hold only hot experts instead ([report](docs/reports/2026-10-31-second-gpu-pool.md)).

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
│              第二张卡当热专家池（可选，`XTIER_GPU2_SLOTS`）
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
git -C /path/to/sglang checkout 6162b07199      # 测试过的官方 SGLang main（2026-10-30）
SGLANG_BUILD_RUST_EXTS=none pip install -e /path/to/sglang/python   # 没装 Rust 时跳过 SGLang 可选的 Rust 扩展
tools/build_engine.sh /path/to/site-packages/nvidia/cu13   # CPU 引擎与信箱库 -> xtier/engine/lib
pip install --no-deps -e .                      # 注册插件入口 sglang.srt.plugins: xtier
python tools/check_sglang.py                    # 不用模型：核对 SGLang 的提交、XTier 的钩子目标和 import
XTIER=1 XTIER_THREADS=112 XTIER_MAIN_CPUS=auto XTIER_CACHE_SLOTS=155 \
  python -m sglang.launch_server --model-path <MoE 检查点> --cuda-graph-backend-prefill disabled ...
```

XTier 跟着官方 SGLang main 走，每次只认一个测试过的提交（[xtier/__init__.py](xtier/__init__.py) 的 `SGLANG_COMMIT`，更新时写进
版本更新）；装的 SGLang 不是这个提交时，启动日志会警告，`tools/check_sglang.py` 列出挂不上的钩子。

开关见 [xtier/config.py](xtier/config.py)；按模型和硬件的完整启动参数见 [configs/](configs/)，也可以用下面的启动台。

快速加载（可选，`XTIER_FAST_LOAD=1`；默认关，即 SGLang 原来的加载方式）：专家直接从模型文件（开冷专家时从冷存储）读进
CPU 引擎，全部读完、检查完才报在线，DeepSeek V4.1 Flash 1017 启动 472 → 186 s（[报告](docs/reports/2026-10-27-fast-load.md)）。
MoE 张量并行（`--tp-size` > 1）时每个 rank 只读自己那份（双 RTX 4090 实测 Qwen3.8 权重加载 170 → 133 s，分片与普通加载逐字节相同）。

## 启动台

![XTier 启动台](docs/images/console.png)

在仓库根目录运行 `python3 -m xtier.console`（只用 Python 标准库），打开 http://127.0.0.1:8100/ ：

- **左边模型栏**：每个模型一张小卡片，写着端口和 README 实测速度，标签注明测法（AGENT = agent 速度，文本 = 文本速度），
  模型多了在栏内滚动。
- **中间启动参数**：默认就是实测配置，改过的项会标出。公共的有加速包开关（关 = 测速表里的「官方 main」）、投机解码、上下文、
  CPU 专家线程，其余是各模型自己的（热专家槽、分块、弹性热专家、冷专家卸载、未命中专家 GPU 代算等）。
- **右边运行状态**：加载阶段（权重 → 草稿 → 专家池 → CUDA 图 → 预热 → 在线），在线后显示 decode 速度、接受长度、显存和日志。

一次只跑一个模型；模型服务在自己的端口上提供 OpenAI 兼容接口，启动台退出不影响它。加模型只要在 `configs/<模型>-<硬件>/` 里
加一个 `console.py`（卡片、选项、启动命令）。细节见 [docs/console.md](docs/console.md)。

## 支持的模型

| 模型 | 官方模块 | 状态 |
|---|---|---|
| Qwen3.8 Flash Next | `qwen4_exp` | 官方 main 上跑通：精度、照抄、512K 召回与生产一致；加速包比官方 main 冷 prefill 快 5% / 24%（64K / 512K）；KV 存 NVFP4（与生产同参，加速包让官方 QSA 能读），省下的显存用来多放常驻层，启动台默认由规划器自动布局；PLE 表（48 GiB）可放 NVMe，放傲腾时速度与锁页表持平 |
| GLM-5.3 Flash | `glm5_next` | 官方 main 上跑通（含 EAGLE）：加速包（FP8 权重、热词表拒绝采样、截断验证、弹性热专家槽）比官方 main 长回复每步快 22%，PPL 3.987 → 3.998 |
| DeepSeek V4.1 Flash 1017 | `deepseek_v4` | 官方 main 上跑通（含 DSpark）：Engram 表（189 GB）的行放 NVMe（[设计](docs/design-tables-nvme.md)），照抄 0 错；加速包（含 FP8 wo_a、分页索引器、弹性热专家槽）比官方 main 长回复每步快 39%、冷 prefill 快 33% |
| MiniMax-M3.1-Flash | `minimax_m3` | **暂停开发**（2026-10-26，本机太慢、智力不强）。SGLang 的 M3.1 分支（PR #45925）上跑通（含 DSpark），加速包比官方代码长回复每步快 34% |

## 支持的显卡

| 显卡 | 状态 |
|---|---|
| Blackwell（SM120） | 开发和测速平台（RTX PRO 5000），下面的测速都在这上面测 |
| RTX 40 系（SM89） | 支持。RTX 4090 实测：Qwen3.8 整模型在单卡 24 GB + 120 GB 内存上照抄 0 错、120K 上下文（[配置](configs/qwen38-rtx4090-3090/README.md)）；GLM-5.3 截层、DeepSeek V4.1 单层测试通过，整模型在本机模拟验证；GLM 要关投机解码。详见[报告](docs/reports/2026-10-29-rtx40.md) |
| RTX 30 系（SM86） | 支持。RTX 3090 实测：Qwen3.8 整模型在单卡 24 GB 上照抄 0 错、120K 上下文，decode 约 42 tok/s（云主机 28 个 CPU 线程）；GLM-5.3 截层、DeepSeek V4.1 单层测试通过，整模型在本机模拟验证。FP8 全部软件解码（与硬件逐位相同），GLM 用 BF16 KV、要关投机解码。详见[报告](docs/reports/2026-10-29-rtx30.md) |

## 多卡（张量并行）

两张卡用张量并行（`TP=2`）：每张卡放每个热专家的一半和一半 KV，每卡能放的热专家槽数是单卡的三倍多；CPU 引擎每个 rank 一个进程，
各用自己 GPU 那组 NUMA 节点（每颗 CPU 带一张卡时正好一卡一 CPU）。两卡之间不需要 P2P / NVLink，SGLang 会自动改走 NCCL。
双 RTX 3090 实测 Qwen3.8：decode 42.1 → 59.6–61.7 tok/s（+42–48%，开 MTP 69–72 tok/s），120K 首字 72.6 → 47.0 s，照抄 0 错。
两张卡显存不一样大时，SGLang 默认拒绝启动，加 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` 可以跑，但两张卡都只能按小卡的显存放
（24 + 16 GiB 实测每层 64 槽，decode 51 tok/s）。
用法见[配置](configs/qwen38-rtx4090-3090/README.md#two-gpus-tp2)，详见[报告](docs/reports/2026-10-30-tp.md)。

### 异构卡：快卡带慢卡，第二张卡纯当热专家池

两张卡显存或速度不一样时，TP 两边都只能按小卡、慢卡的节奏跑。这时可以让第二张卡不放模型，只放更多热专家（`XTIER_GPU2_SLOTS`，默认关）：

```
XTIER_GPU2_SLOTS=auto CUDA_VISIBLE_DEVICES=快卡,慢卡 bash configs/qwen38-rtx4090-3090/run-qwen.sh
```

- **做法**：主卡完全不变。调度线程把主卡没命中、第二张卡有的专家交给它，结果经内存加回 CPU 的输出，不需要 P2P。prefill 只用主卡，所以首字不变。
- **自动分配**：第二张卡的显存全部用来放专家（`auto`，留 1.5 GiB）。两张卡放不同的专家，按路由统计换入换出；每次调用分给第二张卡多少专家会自动调整，让它和 CPU 差不多同时算完。
- **精度**：用和主卡相同的内核。单层测试（Qwen3.8、GLM-5.3、DeepSeek V4.1）里，同样的专家放在第二张卡上和放在主卡上结果逐位相同。
- **限制**：只支持 `TP=1`，不能和冷专家一起用；第一张卡有硬件 FP4 解码时，第二张卡也要有（Blackwell 之间没问题）。

用两张 RTX 4090 模拟实测（Qwen3.8 整模型，单请求，输出 512 token，云主机 30 个 CPU 线程、AVX2）。第二张卡的 16 GiB 用占位进程限制。容器不能改频率，慢卡是让第二张卡每次调用多算一遍 `_down` 内核（每次调用约 102 → 120 µs，约慢 1.2 倍）：

| 配置 | 每层热专家槽 | 32K decode | 120K decode | 120K 首字 |
|---|---|---:|---:|---:|
| 单卡 24 GiB | 48 | 41.0 | 43.6 | 60.9 s |
| TP 24 + 16 GiB | 64 | 51.1 | 49.4 | 60.0 s |
| **池 24 + 16 GiB** | **48 + 113** | **57.3** | **55.9** | **58.9 s** |
| 池 24 + 16 GiB，第二张卡约慢 1.2 倍 | 48 + 113 | 54.5 | 54.2 | 59.0 s |
| 池 24 + 24 GiB | 48 + 178 | 58.2 | 56.2 | 59.3 s |
| TP 24 + 24 GiB（同规格） | 180 | 62.1 | 63.4 | 68.9 s |

速度单位 tok/s。TP 的槽放半个专家。显存不对等时池比 TP 快 12–13%，比单卡快 28–40%；同规格的两张卡 decode 还是 TP 更快。这组数据只说明趋势：这台机器的 CPU 偏弱，CPU 越强，第二张卡能省的时间越少；第二张卡的显存带宽也没有模拟。真实的 RTX PRO 5000 + PRO 4500、RTX 5090 + 5070 Ti 还没测过，DeepSeek V4.1 和 GLM-5.3 只测了单层和截层。

照抄：开这个功能时 9 次启动里有 3 次在同一处翻 1 个 token，其余 0 错。不开第二张卡也会翻这一处：这个 token 对提示词 KV 很敏感，单张卡反复清缓存重算，16 次里翻 1 次。详见[报告](docs/reports/2026-10-31-second-gpu-pool.md)。

## 测速

单卡 RTX PRO 5000 72GB + 2× EPYC 9334（NPS4）+ 384 GiB DDR5，官方 SGLang main 8cd3c494，单位 tok/s，只放最新一次的结果。

用 [bench/agent_bench.py](bench/agent_bench.py) 模拟编码代理：5 个会话从 64K 到约 500K，铺满 512K KV 池，多步调用工具，同一次启动
连跑两轮。冷 prefill = 会话第一个请求里未命中缓存的 token / 首字，decode 只算长回复（≥ 300 token）。官方 main = 启动台「关加速包」：
XTier 加上跑起来必需的修正，不开提速项。配置、精度和原始数据见[测速细节](docs/benchmarks.md)。

### DeepSeek V4.1 Flash 1017

| DSpark | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 3,812 | 4,768 | 141.8 | 4.50 | 31.7 ms |
|  | 2 | 5,027 | 4,844 | 140.8 | 4.30 | 30.6 ms |
| 官方 main | 1 | 3,318 | 3,627 | 65.5 | 3.25 | 50.9 ms |

加速包对官方 main：每步 31.2 对 50.9 ms（快 39%），冷 prefill 快 33%；精度无损，照抄 0 错。

### Qwen3.8 Flash Next

| MTP | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 5,715 | 5,170 | 181.7 | 2.57 | 14.2 ms |
|  | 2 | 5,610 | 5,150 | 207.2 | 2.68 | 13.0 ms |
| 官方 main | 1 | 5,319 | 4,078 | 139.5 | 2.13 | 15.2 ms |

加速包（规划器自动布局）对官方 main：每步 13.6 对 15.2 ms（快 11%），冷 prefill 64K / 500K 快 6% / 27%；精度轻微损失
（KV 存 NVFP4，与生产相同），PPL 3.481，照抄 0 错。

### GLM-5.3 Flash

| EAGLE | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 2,648 | 2,640 | 96.3 | 2.54 | 26.4 ms |
|  | 2 | 2,715 | 2,635 | 98.2 | 2.66 | 27.1 ms |
| 官方 main | 1 | 2,663 | 2,518 | 76.8 | 2.61 | 34.4 ms |

加速包（58 槽 + 弹性热专家槽）对官方 main：每步 26.9 对 34.4 ms（快 22%），冷 prefill 64K / 500K 快 1% / 5%；精度轻微损失
（注意力 / 稠密层存 FP8，PPL 3.987 → 3.998），照抄 0 错。

### MiniMax-M3.1-Flash（暂停开发）

| DSpark | 轮 | 冷 prefill 64K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 1,854 | 53.4 | 3.04 | 56.9 ms |
|  | 2 | 1,877 | 50.0 | 2.70 | 54.0 ms |
| 官方代码 + 必要修正 | 1 | 1,824 | 46.8 | 3.94 | 84.3 ms |
|  | 2 | 1,883 | 45.5 | 3.82 | 84.0 ms |

SGLang 的 M3.1 分支（PR #45925），固定 512K，只跑 64K 会话；加速包对官方代码 + 必要修正每步 55.4 对 84.1 ms（快 34%）。

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

- 2026-10-31：第二张卡可以不放模型、纯当热专家池（可选，`XTIER_GPU2_SLOTS`，默认关），给显存或速度不一样的两张卡用：双 RTX 4090 模拟 24 + 16 GiB 时 Qwen3.8 decode 比同样配置的 TP 快 12–13%，比单卡快 28–40%。详见[文档](docs/reports/2026-10-31-second-gpu-pool.md)。
- 2026-10-30（下午）：跟进官方 SGLang main 6162b07，安装步骤写明测试过的提交，装的版本不同时启动会警告，`tools/check_sglang.py`
  不用模型就能核对钩子；快速加载支持 MoE 张量并行（双 RTX 4090 实测通过）；英伟达官方 GLM NVFP4 检查点开 MTP 不再崩溃。详见[文档](docs/reports/2026-10-30-sglang-6162b07.md)。
- 2026-10-30：双卡张量并行（`TP=2`）实测通过：双 RTX 3090 上 Qwen3.8 decode +42–48%、120K 首字 −35%，照抄 0 错；修了
  `~/.cache` 是软链接时 SGLang 的 JIT 内核每次启动都重编（30 系单卡启动省约 5 分钟）。详见[文档](docs/reports/2026-10-30-tp.md)。
- 2026-10-29（晚）：支持 RTX 30 系（SM86）：没有 FP8 硬件，FP8 一律读时软件解码（与硬件逐位相同），DeepSeek 的 FP8
  投影只量化权重；RTX 3090 实测 Qwen3.8 整模型照抄 0 错、120K 上下文。详见[文档](docs/reports/2026-10-29-rtx30.md)。
- 2026-10-29：支持 RTX 40 系（SM89）：FP4 读时软件解码（与硬件逐位相同，不增加精度损失），三个模型的加速包补上 SM90
  以下的内核；RTX 4090 实测 Qwen3.8 整模型照抄 0 错、120K 上下文。详见[文档](docs/reports/2026-10-29-rtx40.md)。

更早的版本更新见 [docs/changelog.md](docs/changelog.md)。

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
