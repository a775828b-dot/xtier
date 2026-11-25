# XTier

[English](docs/README.en.md) · [文档目录](docs/README.md) · [快速开始](docs/quickstart.md) · [支持情况](docs/support.md)

XTier 是官方 [SGLang](https://github.com/sgl-project/sglang) 的插件（跟 `main` 分支），做 CPU/GPU 分层 MoE 推理：热专家常驻显存、其余放内存并按热度自适应调度；NVMe 只放大嵌入表（DeepSeek Engram、Qwen PLE）或可选的冷专家尾池。

**状态：早期开发**（[工作计划](docs/PLAN.md)）。Inkling-2-Small、Hunyuan-4-Flash NVFP4、DeepSeek V4.1 Flash 1017、Qwen3.8 Flash Next、GLM-5.3 Flash 跑在官方 SGLang `main` 上；MiniMax-M3.1-Flash 已暂停开发。

- **不改 SGLang 源码**：通过官方插件钩子接管混合推理入口，加载后自检接管状态。
- **与模型无关**：包装官方 `FusedMoE` 量化方法，标准 MoE 模型（NVFP4 / MXFP4 / MXFP8）装上即可跑。
- **加速包可选**：各模型专属内核与修正集中在 `xtier/packs/`，启动台可一键开关对比基线。

## 结构

```
xtier/
├── backend/   混合 MoE 后端：接管权重加载、CUDA 图信箱解码、GPU 分组 GEMM / CPU 预填
├── cache/     热专家缓存（弹性热槽、自适应换入换出）、双卡热池（XTIER_GPU2_SLOTS）、冷专家池（XTIER_COLD=1）
├── planner/   规划器：结合本机探测耗时与路由档案自动分配每层热槽（docs/planner.md）
├── engine/    CPU 引擎 Python 绑定与锁页内存（C++ 源码在 csrc/）
├── packs/     模型加速包：inkling2 / hy4_flash / deepseek_v4 / qwen4_exp / glm5_next / minimax_m3 / speculative
├── tables/    大嵌入表放 NVMe（Engram / PLE），行缓存 + CUDA 图内查表
└── console/   零依赖 Web 启动台：选模型、调参数、启停服务与实时遥测
csrc/          AVX-512 / AVX2 CPU 专家内核、NUMA 线程池与信箱通信
configs/       各「模型 × 硬件」启动脚本与启动台卡片（console.py）
tests/ tools/  单层与回归测试、发版验证（tools/fast_verify.py）、构建与规划工具
bench/         验收脚本（PPL、照抄、agent_bench、上下文与并发测速）
```

## 快速上手

完整部署见[快速开始](docs/quickstart.md)。已有 SGLang 环境时：

```bash
git -C /path/to/sglang checkout 4ce41378f9      # 测试过的官方 SGLang main（2026-11-21）
SGLANG_BUILD_RUST_EXTS=none pip install -e /path/to/sglang/python
tools/build_engine.sh /path/to/site-packages/nvidia/cu13   # 编译 CPU 引擎与信箱库
pip install --no-deps -e .                      # 注册插件入口 sglang.srt.plugins: xtier
python tools/check_sglang.py --strict           # 核对 SGLang 钩子
python -m xtier.console --port 8080             # 打开启动台（或直接运行 configs/ 下的脚本）
```

## 启动台

`python -m xtier.console --port 8080` 提供单页 Web 控制台：选模型卡片、调参数（默认即实测推荐配置，关加速包即基线）、查看加载阶段与 decode 遥测。新增模型只需在 `configs/<模型>-<硬件>/` 添加 `console.py`，详见 [docs/console.md](docs/console.md)。

## 支持的模型

| 模型 | 加速包 | 状态与要点 |
|---|---|---|
| Inkling-2-Small（594B / 17B，多模态） | `inkling2` | 官方 `main` 跑通（512K / 2 并发，5 层 DSpark + 23 层 ViT）：NVFP4 KV（6.75 GiB）、64K 单块预填、DSpark 在线截断，默认方案 A+（2% 冷尾）[报告](docs/reports/2026-11-25-inkling2-small.md) |
| Hunyuan-4-Flash NVFP4 | `hy4_flash` | 官方 `main` 跑通（1M 上下文 / 4 并发，原生 5 层 DFly2）：混合 SWA+DSA NVFP4 KV（7.95 GB）、FP8 IndexPool、Engram 放 NVMe，照抄 0 错 |
| DeepSeek V4.1 Flash 1017 | `deepseek_v4` | 官方 `main` 跑通（512K / 4 并发，1M KV 池，DSpark + Recall）：Engram 放 NVMe、43 层 FP8 `wo_a`、24-token 权重复用、500K 预填破 9,000 tok/s |
| Qwen3.8 Flash Next | `qwen4_exp` | 官方 `main` 跑通（512K，MTP）：NVFP4 KV + 自动规划布局、PLE 放 NVMe，精度与生产一致 |
| GLM-5.3 Flash | `glm5_next` | 官方 `main` 跑通（512K，EAGLE）：FP8 权重、热词表拒绝采样、弹性热槽，长回复每步快 22% |
| MiniMax-M3.1-Flash | `minimax_m3` | **暂停开发**（2026-10-26，SGLang PR #45925 分支，代码与卡片保留） |

## 支持的显卡与多卡

完整矩阵与实测范围见[支持情况](docs/support.md)。

| 显卡 / 模式 | 状态 |
|---|---|
| Blackwell（SM120） | 开发与测速平台（RTX PRO 5000 72 GB）；RTX PRO 6000 96 GB 已验 Qwen3.8 整模型 |
| Ada / RTX 40 系 & L4 / L20（SM89） | 支持：RTX 4090 / L4 已验 Qwen3.8 整模型（[配置](configs/qwen38-rtx4090-3090/README.md)）；社区贡献 `2×L20 TP=2` DeepSeek 配置（[配置](configs/ds41-l20-tp2/README.md)） |
| Ampere / RTX 30 系 & A100（SM86 / SM80） | 支持：RTX 3090、A100 40 GB 已验 Qwen3.8 整模型（FP8 软件逐位解码）；详见 [RTX 30 报告](docs/reports/2026-10-29-rtx30.md) |
| 双卡张量并行（`TP=2`） | 无需 P2P / NVLink（走 NCCL），双 RTX 3090 跑 Qwen3.8 提速 +42–48%（[报告](docs/reports/2026-10-30-tp.md)） |
| 异构双卡热专家池（`XTIER_GPU2_SLOTS=auto`） | 第二张慢卡只存热专家并代算（`TP=1`），模拟 24+16 GB 提速 +28–40%（[报告](docs/reports/2026-10-31-second-gpu-pool.md)） |

## 测速

平台：单卡 RTX PRO 5000 72 GB + 2× EPYC 9334（NPS4）+ 384 GiB DDR5，单位 `tok/s`。基线 = 启动台「关加速包」（官方 SGLang + XTier 混合后端与跑通必需修正，**不是原版 SGLang**）。详细配置、精度与原始数据见[测速细节](docs/benchmarks.md)。

### Inkling-2-Small（594B / 17B，多模态）

512K 上下文 / 2 并发（40 NVFP4 + 8 MXFP8 MoE，5 层 DSpark + 23 层 ViT），SGLang `4ce41378f9`：

| 模式 | 单路 decode（bs=1） | 2 并发合计（bs=2） | 64K 冷 prefill | 256K 冷 prefill | 500K 冷 prefill | 500K 缓存 decode |
|---|---:|---:|---:|---:|---:|---:|
| **XTier + 加速包（默认）** | **74.7**（首轮 81.7） | **78.4** | **3,835** | **2,464** | **1,593** | **43.7**（峰 47.0） |
| 基线（关加速包） | 47.9 | ~43 | 1,591 | 1,217 | OOM | OOM |

加速包对基线：64K / 256K 冷 prefill 提速 **2.41× / 2.02×**，短文 `bs=1` / `bs=2` 提速 **+56% / +82%**，NVFP4 KV（6.75 GiB）完整解锁 500K；两轮 `agent_bench`（64K→513K）长回复 **66.2 tok/s**（全部 66.3，接受长度 4.02）。精度：13 段 `ΔNLL +0.00423` / `KL 0.00938`，照抄 `0/8`，256K/500K 捞针全 `5/5`。方案 A+/B2 对比与解/预填剖析见[报告](docs/reports/2026-11-25-inkling2-small.md)。

### Hunyuan-4-Flash NVFP4

1M KV 池 / 1M 上下文 / 4 并发（混合 SWA+DSA NVFP4 KV 7.95 GB + 原生 5 层 DFly2），SGLang `4ce41378f9`：

| 模式 | 单路 decode（bs=1） | 4 并发合计（bs=4） | 100K 召回 prefill | 4×128K 并发 prefill | 876K–950K 冷 prefill | 超长缓存 decode |
|---|---:|---:|---:|---:|---:|---:|
| **XTier + 加速包 + DFly2（默认）** | **92.1**（稳态 124–152） | **186.8–190.4**（稳态 209–226） | 3,630（PASS） | 3,565 | 3,454（876K） | **87.5–110.8**（876K） |
| 基线（全量 FP8 KV，BF16 稠密层） | 55–56 | 148–162 | 3,657（PASS） | — | 3,084（950K） | 43.8（950K） |

照抄 `0/4` 差（双路达 299–306 tok/s），PPL 100% 无损，876K 缓存 decode 零 OOM 重试。

### DeepSeek V4.1 Flash 1017

[agent_bench.py](bench/agent_bench.py) 两轮 64K→500K 编码代理（加速包行测于 2026-11-24 / SGLang `4ce41378f9`）：

| DSpark | 轮 | 冷 prefill 64K | 冷 prefill 500K | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + 加速包** | 1 | 5,969 | **9,038** | 156.6 | 4.64 | 29.6 ms |
|  | 2 | **7,967** | **9,023** | 154.7 | 4.65 | 30.1 ms |
| **XTier + 加速包，4 并发合计** | — | — | — | **268** | 3.40 | 51.0 ms |
| 基线（关加速包） | 1 | 3,318 | 3,627 | 65.5 | 3.25 | 50.9 ms |

加速包对基线：长回复每步快 **41%**，64K / 500K 冷 prefill 快 **110% / 149%**（100% 无损，PPL NLL 1.2001 对 1.1996，照抄正文 0 错）；含工具调用全部 127 请求合计 decode 163.9 tok/s，2 并发新写 195.8 tok/s，4 并发改写/编辑达 332–382 tok/s。详见 [9K+ 无损预填报告](docs/reports/2026-11-19-lossless-9k-prefill.md) 与 [Recall 报告](docs/reports/2026-11-18-lossless-prefill-recall.md)。

### Qwen3.8 Flash Next

| MTP | 轮 | 冷 prefill 64K | 冷 prefill 500K | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + 加速包** | 1 | 5,715 | 5,170 | 181.7 | 2.57 | 14.2 ms |
|  | 2 | 5,610 | 5,150 | 207.2 | 2.68 | 13.0 ms |
| 基线（关加速包） | 1 | 5,319 | 4,078 | 139.5 | 2.13 | 15.2 ms |

加速包对基线：每步快 11%，冷 prefill 快 6% / 27%（NVFP4 KV + 自动规划，照抄 0 错）。

### GLM-5.3 Flash

| EAGLE | 轮 | 冷 prefill 64K | 冷 prefill 500K | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + 加速包** | 1 | 2,648 | 2,640 | 96.3 | 2.54 | 26.4 ms |
|  | 2 | 2,715 | 2,635 | 98.2 | 2.66 | 27.1 ms |
| 基线（关加速包） | 1 | 2,663 | 2,518 | 76.8 | 2.61 | 34.4 ms |

加速包对基线：每步快 22%，冷 prefill 快 1% / 5%（FP8 稠密层 + 弹性热槽，PPL 3.987 → 3.998，照抄 0 错）。

### MiniMax-M3.1-Flash（暂停开发）

SGLang M3.1 分支（PR #45925），64K 会话加速包对比官方代码 + 必需修正：长回复 51.7 对 46.1 tok/s（每步 55.4 对 84.1 ms，快 34%），见[报告](docs/reports/2026-10-24-minimax-512k.md)。

## 冷专家卸载与最小化运行

内存放不下全部专家时开启 `XTIER_COLD=1` 将少用专家放 NVMe（[设计](docs/design-cold-experts.md)），并支持 `tools/cold_punch.py` 让大模型在硬盘只存一份、按盘速分段（`XTIER_COLD_UNITS`）与尾部跳过（`XTIER_TAIL_SKIP`）。

- **4 块 PCIe 5.0 实测（DeepSeek 64K）**：卸载 10% / 20% / 30% / 50% 对应 decode `135 (−5%)` / `124 (−13%)` / `112 (−21%)` / `80 (−44%) tok/s`，详见[报告](docs/reports/2026-11-01-cold-pcie5.md)。
- **最小化运行（模拟 32 GB 显存 + 128 GB 内存，70% 冷卸载）**：DeepSeek 32K–128K prefill `1,155–1,186 tok/s`、decode `28.1–32.7 tok/s`，详见[报告](docs/reports/2026-10-27-ds41-minimal.md)。

## 版本更新

- **2026-11-25（v0.1.10）**：新增 **Inkling-2-Small（594B 多模态）** 零补丁加速包与启动台卡片（默认方案 A+：`cold=2%`、45 热槽、64K 单块 3,835 tok/s、NVFP4 KV 解锁 500K、两轮 agent 66.3 tok/s；`cold=off` 方案 B2 仅 `L47` 全驻），并打包合入 11-24 的 DeepSeek 无损 9K+ 优化。[报告](docs/reports/2026-11-25-inkling2-small.md)
- **2026-11-24**：DeepSeek V4.1 与通用层新一轮 100% 无损提速（24-token 权重复用、43 层 FP8 `wo_a`、写回融合、SM120 Eager MXFP8 快分发），500K 冷 prefill 双破 9,000 tok/s，4 并发新写达 268 tok/s。
- **2026-11-23（v0.1.9）**：新增 **Hunyuan-4-Flash NVFP4** 零补丁加速包与原生 5 层 DFly2 投机解码（1M 混合 NVFP4 KV 7.95 GB，bs=1 92.1 / bs=4 190 tok/s），上线 `tools/fast_verify.py` 三层快速验证，合入社区 L20（SM89）等 PR（#18/#7/#19）。
- **2026-11-22（v0.1.8）**：跟进官方 SGLang `4ce41378f9`，打包 9,000+ 无损 prefill / 1.1 s 工具回传优化与默认 CPU 多模态解码限流。[报告](docs/reports/2026-11-22-sglang-4ce4137.md) · [9K 报告](docs/reports/2026-11-19-lossless-9k-prefill.md)
- **2026-11-18（v0.1.7）**：全链路无损 prefill 融合、Engram 跨块预热与 DSpark 上下文草稿检索（Recall，agent 全请求 180.4 tok/s）。[报告](docs/reports/2026-11-18-lossless-prefill-recall.md)

更早的版本更新与完整细节见 [docs/changelog.md](docs/changelog.md)。

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
