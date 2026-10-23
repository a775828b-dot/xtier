# XTier

**Cross-tier CPU/GPU hybrid MoE inference for [SGLang](https://github.com/sgl-project/sglang).**
XTier is a plugin for upstream SGLang (tracking `main`): MoE experts live across GPU memory, host DRAM and NVMe, are placed by
how often they are used and computed where they are. It wraps SGLang's standard `FusedMoE` quantization methods, so any MoE model
that uses the standard layer can run in hybrid mode without model-specific code; optional per-model acceleration packs add
model-specific optimizations. **Status: early development** — Qwen3.8 Flash Next, GLM-5.3 Flash and DeepSeek V4.1 Flash 1017
run on SGLang main; see [docs/PLAN.md](docs/PLAN.md).

---

XTier 是官方 SGLang 的插件（跟 main 分支），做 CPU/GPU 分层 MoE 推理：专家分布在显存、内存和 NVMe 三层，按热度放置、按需调度。

- **不改 SGLang 源码**：只用官方插件框架的钩子（接在官方 KTransformers 混合推理的入口上），加载后自检是否接管成功。
- **与模型无关**：混合推理包在官方 `FusedMoE` 的量化方法外面，用标准 MoE 层的模型装上就能跑（NVFP4、MXFP4 专家）。
- **加速包可选**：某个模型独有的修正和内核放在 `xtier/packs/` 下。

## 结构

```
xtier/
├── backend/   混合 MoE 后端：接管官方量化方法，权重加载完即交给 CPU 引擎；解码走信箱（CUDA 图可捕获），prefill 走 GPU 分组 GEMM 或 CPU
├── cache/     热专家缓存：每层槽数可不同，全部常驻的层不经过 CPU；自适应换入换出；常驻层的 prefill 直接在槽上算
├── planner/   规划器：决定每层由谁算、缓存多少（目前按配置，目标是 CPU 和 GPU 同时算完）
├── engine/    CPU 引擎的 Python 绑定、绑核、锁页内存（C++ 源码在 csrc/）
├── packs/     加速包：qwen4_exp（PLE 表、PLE 行放 NVMe、hc_mix 重叠内核、QSA 读 NVFP4 KV）、glm5_next（量化排除、DSA、NextN）、
│              deepseek_v4（Engram 行放 NVMe、SM120 候选索引器、FP8 GEMM 调优配置）、
│              speculative（通用 EAGLE 投机解码：热词表配无损拒绝采样、草稿温度、按置信度截断验证）
├── tables/    大嵌入表的行放在 NVMe：多段多文件、任意行宽、行缓存、CUDA 图内查表
├── console/   启动台：一个网页选模型、调启动参数、启动 / 停止，跟踪加载阶段与 decode 遥测（只用标准库）
└── platform.py  与模型无关的主机修正（TileLang 目标选择）
csrc/          CPU 引擎：AVX-512 / AVX2 专家内核、NUMA 线程池、CPU 与 GPU 之间的信箱通信
configs/       每个"模型 × 硬件"的启动配置，console.py 是它在启动台里的条目（选项、默认值、启动命令）
tests/         单层测试（引擎、信箱、缓存、prefill、格式适配、hc_mix、投机截断、QSA NVFP4）与启动台检查
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
- **中间启动参数**：默认就是实测配置，改过的项会标出。公共的有加速包开关（关 = 测速表里「官方 main」的配置）、投机解码、上下文、
  CPU 专家线程；各模型自己的有 Qwen 的 KV 精度、分块、常驻层布局、PLE 表放内存还是 NVMe，GLM 的拒绝采样、热专家槽、分块，
  DeepSeek 的 Engram 行缓存、热专家槽、分块。
- **右边运行状态**：加载阶段（权重 → 草稿 → 专家池 → CUDA 图 → 预热 → 在线），在线后显示 decode 速度、接受长度、显存和日志。

一次只跑一个模型；模型服务在自己的端口上提供 OpenAI 兼容接口，启动台退出不影响它。加模型只要在 `configs/<模型>-<硬件>/` 里
加一个 `console.py`（卡片、选项、启动命令）。细节见 [docs/console.md](docs/console.md)。

## 支持的模型

| 模型 | 官方模块 | 状态 |
|---|---|---|
| Qwen3.8 Flash Next | `qwen4_exp` | 官方 main 上跑通：精度、照抄、512K 召回与生产一致；加速包后 decode 每步、64K / 512K 冷 prefill 与生产持平或更快；KV 存 NVFP4（与生产同参，加速包让官方 QSA 能读），省下的显存多放一层常驻；PLE 表（48 GiB）可放 NVMe，放傲腾时速度与锁页表持平 |
| GLM-5.3 Flash | `glm5_next` | 官方 main 上跑通（含 EAGLE）：加速包后生产的温度 1.0 测法 decode 95.6 对 84.8 tok/s（热词表拒绝采样，与生产相同），64K / 512K 冷 prefill 比生产快，PPL 3.994（生产 4.060） |
| DeepSeek V4.1 Flash 1017 | `deepseek_v4` | 官方 main 上跑通（含 DSpark）：Engram 表（189 GB）的行放 NVMe（[设计](docs/design-tables-nvme.md)），照抄 0 错；按生产的 agent_bench（同一套编码代理会话）实测：冷 prefill 比生产快 10–25%，decode 每步慢约 6%（33.9 对 32.0 ms），原因在查 |

## 测速

单卡 RTX PRO 5000 72GB + 2× EPYC 9334（NPS4）+ 384 GiB DDR5，官方 SGLang main 8cd3c494，单位 tok/s，表里只放最新一次的结果。
两种测法，以后统一用 agent 速度：

- **agent 速度**：与生产仓库同一个 `bench/agent_bench.py`，模拟编码代理（dsh / Claude Code 一类工具）：5 个会话从 64K 到约 500K，
  铺满 512K KV 池，多步调用工具，后续请求命中前缀缓存，服务端默认采样，同一次启动连跑两轮。目前 DeepSeek 用这个。
- **文本速度**：编码代理的会话记录截成 64K / 512K token 当提示，冷 prefill（先清前缀缓存），输出 1K token（含思考），单请求、
  默认采样，每个配置先用一个 32K 的提示预热（[bench/context_speed.py](bench/context_speed.py)）。Qwen、GLM 目前是这个，之后换成
  agent 速度。

### agent 速度

| DeepSeek V4.1 Flash 1017（DSpark） | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 3,503 | 4,801 | 121.7 | 4.13 | 33.9 ms |
|  | 2 | 4,994 | 4,825 | 131.5 | 4.45 | 33.8 ms |
| 生产（lsglang_ds4.1，10-22） | 1 | 3,377 | 3,856 | 142.7 | 4.56 | 31.9 ms |
|  | 2 | 4,431 | 3,856 | 136.1 | 4.36 | 32.0 ms |

- **decode 只比长回复**（≥ 300 token，写代码、写总结）：官方 main 的 DeepSeek 工具调用解析不逐 token 流式返回参数，XTier 上工具调用
  请求量不出 decode（两轮 149 个请求里能计时 18 个）；生产所有请求都能计时，算上工具调用（输出短、接受长度高）是 158.6 / 144.6 tok/s。
- 接受长度随回复内容变，比较两边看每步耗时：XTier 慢约 6%（CPU 专家侧与生产持平，差在 GPU 侧，在查）。冷 prefill XTier 快 10–25%
  （两边第 1 轮的 64K 会话都是启动后的第一个请求，偏慢）。
- 加速包精度：无损——块 FP8 GEMM 的分块配置和内核、Engram 提前查表（取到的行不变）、草稿词表头 FP8（只影响草稿提案，输出仍由主模型
  验证），PPL 3.319 → 3.317，照抄 0 错。官方 main（不开加速包）还没按 agent 测。
- 每会话的明细（上下文起止、首字、请求数）见[报告](docs/reports/2026-10-24-qwen-kv-ds-agent.md)，原始数据
  `bench/results/agent-bench-ds41-xtier-r{1,2}-20261023.json`。

### 文本速度（Qwen、GLM）

| 模型 | 配置 | 64K prefill | 64K decode | 512K prefill | 512K decode | 加速包精度 |
|---|---|---:|---:|---:|---:|---|
| Qwen3.8 Flash Next（MTP） | 官方 main | 5319 | 140.7（14.63 ms） | 4078 | 138.3（15.75 ms） | 基准 |
|  | + 加速包 | 5587 | 161.4（14.47 ms） | 5040 | 147.5（15.62 ms） | 轻微：KV 存 NVFP4（与生产相同，每 16 个值一个 E4M3 缩放；读出与官方反量化逐位一致），照抄 0 错、PPL 3.481；hc_mix 换成重叠的持久内核、PLE 层与 QSA 索引器分段与官方逐位一致 |
| GLM-5.3 Flash（EAGLE） | 官方 main | 2663 | 78.6（34.93 ms） | 2518 | 75.0（33.83 ms） | 基准 |
|  | + 加速包 | 2808 | 100.0（29.16 ms） | 2656 | 91.9（26.78 ms） | 轻微：BF16 注意力 / 稠密层存成 FP8（每行每 128 列一个 scale），PPL 3.987 → 3.998（+0.3%）；拒绝采样与截断验证无损（输出分布与不用投机解码相同），照抄 0 错 |

- **decode**：5 次回答的中位数（1 次冷启动，加 4 次命中前缀缓存重发，上下文相同、各自采样）。括号里是每个验证步的耗时。
  投机解码每步出几个 token 取决于这段文字好不好猜，单次 tok/s 能差 30%。Qwen 的草稿长度固定，每步耗时 5 次之间
  只差 1–3%，比较两种配置看括号里的数；GLM 开了自适应草稿步数，每步耗时也随接受长度变（±5%）。例如 Qwen 两行的 decode
  tok/s 差 10%，大部分是这几次回答的接受长度不同（2.10 对 2.22、2.16 对 2.38），加速包本身每步只快 2%。
- 512K 一列的提示是 510K token，加上 1K 输出不超过 524288 的上下文。每个配置的完整数据（首 token、5 次 decode 的范围、
  接受长度）：加速包（10-23 夜重测）见 [reports/2026-10-23-speculative-fp4kv.md](docs/reports/2026-10-23-speculative-fp4kv.md#readme-重测加速包三行10-23-夜)，
  官方 main 见 [reports/2026-10-23-step3-ds-tables.md](docs/reports/2026-10-23-step3-ds-tables.md#readme-测速)。

### 配置（两种测法通用）

- **官方 main**：XTier 后端加上跑起来必需的修正，不开任何提速项。必需的修正：Qwen 的 PLE 表锁页分配；GLM 的量化排除、DSA
  别名、NextN 草稿量化；DeepSeek 的 SM120 候选索引器、Engram 行放 NVMe、尾部重放不走 CUDA 图。
- **热专家布局与启动参数与生产相同**：Qwen NVFP4 KV、其余层每层 155 槽；GLM 每层 60 槽、16K 分块、BF16 SSM 状态、拒绝采样；
  DeepSeek 每层 50 槽。不同的地方：Qwen 加速包那行 26 层常驻（比生产多第 27 层，用 NVFP4 KV 省下的显存，prefill +0.5–1.4%）、
  24K 分块（生产 32K：这里 32K 在 64K 冷 prefill 慢 3%），见[报告](docs/reports/2026-10-24-qwen-kv-ds-agent.md)；Qwen 官方 main 那行
  用 FP8 KV（官方 main 的 QSA 读不了 NVFP4）、16K 分块、生产的 25 层常驻；GLM 的官方 main 用 BF16 权重，60 槽时 prefill 显存不够，
  减到 55 槽（显存余量 10.3 GB，加速包那行 9.9 GB），也不开拒绝采样（官方 main 不允许热词表配拒绝采样）。
- **+ 加速包**：Qwen 是 hc_mix 重叠内核（`XTIER_HC_MIX_OVERLAP`）、PLE 层分段（`XTIER_PLE_BLOCK`）、QSA 索引器 prefill 按键分段
  （`XTIER_QSA_KEYS_PER_CTA`）和 QSA 读 NVFP4 KV（`XTIER_QSA_FP4`）；GLM 是 FP8 权重（`XTIER_FP8_WEIGHTS`，生产用的也是这一套）和通用投机解码包（热词表拒绝采样、
  草稿温度 ×0.8、按草稿置信度截断验证，截掉的 token 跳过 CPU 专家，`xtier/packs/speculative/`）；
  DeepSeek 是调好的块 FP8 GEMM 配置、MXFP8 走 FlashInfer CUTLASS（解码宽度走 b12x）、Engram 提前查表（`XTIER_ENGRAM_PREFETCH`）
  和草稿词表头 FP8（`XTIER_DRAFT_LMHEAD_FP8`）。
- **接口参数与生产相同**：推理解析、工具调用解析（DeepSeek `deepseekv41`、GLM `glm47`、Qwen `qwen3_coder`）和缓存命中上报
  （`--enable-cache-report`：usage 里的 `prompt_tokens_details.cached_tokens`，dsh 一类客户端据此显示缓存命中）。

## 版本更新

- 2026-10-24（夜）：README 加启动台介绍；测速分成 agent 速度和文本速度，以后统一用 agent 速度：DeepSeek 改列 agent 速度（与生产
  同一套 agent_bench，每会话明细见 [reports/2026-10-24-qwen-kv-ds-agent.md](docs/reports/2026-10-24-qwen-kv-ds-agent.md)），Qwen、GLM
  暂为文本速度。启动参数补上生产的接口参数：三个模型 `--enable-cache-report`，GLM、Qwen 的工具调用解析器（`glm47`、`qwen3_coder`）；
  按 dsh 的请求（流式 chat completions）实测三个模型：缓存命中正常上报，工具调用正常解析。
- 2026-10-24：控制台 `xtier/console/`（`python3 -m xtier.console`）：一个网页管所有模型——选模型、调启动参数（默认是实测配置）、
  启动 / 停止，跟踪加载阶段与 decode 遥测；每个模型在 `configs/*/console.py` 里描述，加模型只要加一个配置目录。
  见 [docs/console.md](docs/console.md)。
- 2026-10-24：Qwen 默认改为 NVFP4 KV（与生产同参），省下的 1.8 GB 显存实测三种用法：多放一层常驻（冷 prefill 64K +1.4%、
  512K +0.5%）、加热专家槽（无效）、32K 分块（64K 慢 3%），decode 都不变（受 GPU 限制），采用第一种；DeepSeek 按生产的 agent_bench 实测：冷 prefill 快 10–25%，decode 每步慢约 6%（CPU 专家侧不慢，差在 GPU 侧）。
  详见 [reports/2026-10-24-qwen-kv-ds-agent.md](docs/reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-23（夜）：通用投机解码包 `xtier/packs/speculative/`：GLM 用上生产的热词表无损拒绝采样、草稿温度 ×0.8 和按草稿置信度
  截断验证（截掉的验证 token 跳过 CPU 专家，核心加了逐 token 标志），生产的温度 1.0 测法 93.0 → 95.6 tok/s（生产 84.8），README 测法
  64K / 512K decode 94.7 → 100.0、86.7 → 91.9 tok/s，照抄 0 错；三个加速包按当前代码重测；
  Qwen 的 QSA 能读 NVFP4 KV（与官方反量化逐位一致，显存多 1.7 GB），测速表仍用 FP8；DeepSeek 剩余差距排查：热专家命中率、CPU
  每层耗时与生产持平，差别来自测法不同。详见 [reports/2026-10-23-speculative-fp4kv.md](docs/reports/2026-10-23-speculative-fp4kv.md)。
- 2026-10-23（晚）：512K prefill 追平生产。Qwen：QSA 索引器 prefill 的 logits 内核在长上下文时只开 11 个 CTA，加速包按键分段（逐位一致），
  PLE 层分段、分块 16K → 24K，512K 冷 prefill 125 → 102 s（生产 106 s）；GLM：启动参数对齐生产（16K 分块、BF16 SSM 状态），
  224 → 192 s（生产 201 s）。两者 PPL、照抄不变，详见 [reports/2026-10-23-prefill.md](docs/reports/2026-10-23-prefill.md)。
- 2026-10-23（晚）：DeepSeek 加速包移植生产的 Engram 提前查表和草稿词表头 FP8：decode 每步 −8%（64K 33.7 ms、512K 35.0 ms，
  生产约 31.5 ms）；冷提示的 prefill 持平，Engram 行缓存热时 64K prefill +16%；照抄 0 错、PPL 持平，详见
  [reports/2026-10-23-ds-engram-prefetch.md](docs/reports/2026-10-23-ds-engram-prefetch.md)。
- 2026-10-23：README 测速表（三个模型 64K / 512K 上下文的 prefill 与 decode，官方 main 对加速包）；加载不再换出（专家按检查点顺序
  拷贝、映射着的检查点页 page-out、提前读），DS 启动 811 → 409 s；全局预算的热专家规划器（只做了离线核对），详见
  [reports/2026-10-23-step3-ds-tables.md](docs/reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 3 步——DeepSeek V4.1 1017 上官方 main（Engram 行放 NVMe）、通用 NVMe 表（Qwen PLE 也可放）、GLM 的 FP8
  权重、Zen2 / 双卡准备，详见 [reports/2026-10-23-step3-ds-tables.md](docs/reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 2 步——后端、热专家缓存、GPU prefill 在官方 main 上跑通 Qwen3.8 与 GLM-5.3，详见
  [reports/2026-10-23-step2-qwen.md](docs/reports/2026-10-23-step2-qwen.md)。
- 2026-10-23：第 1 步——CPU 引擎迁入 `csrc/`，三个模型的真实层与 lkqmoe 0.5.6 逐位一致（`bench/results/engine-ab-20261023.json`）。

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
