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
├── kvstore/   通用多盘 NVMe KV 二级卸载（O_DIRECT + io_uring，默认关闭，零模型分支）
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

## 测试机硬件配置

下文所有测速与回归均在以下单机平台实测：

| 部件 | 规格 |
|---|---|
| GPU | 1× NVIDIA RTX PRO 5000 72 GB Blackwell（SM120，73,415 MiB，PCIe 5.0 ×16） |
| CPU | 2× AMD EPYC 9334（共 64 核 128 线程，Zen 4，AVX-512 BF16，8-NUMA NPS4） |
| 内存 | 384 GiB DDR5（24 通道，系统可用 377 GiB） |
| 存储 | 4× Samsung 9100 PRO 1 TB（PCIe 5.0 NVMe，冷专家/模型池）· 2× Optane P5800X 400 GB U.2（大嵌入表）· 1× 990 PRO 1 TB |
| 系统与软件 | Ubuntu 24.04.4 LTS（Linux 7.0）· NVIDIA 595.99.02 · CUDA 13.0 · PyTorch 2.14.1 · SGLang main `4ce41378f9` |

## 测速

单位 `tok/s`。基线 = 启动台「关加速包」（官方 SGLang + XTier 混合后端与跑通必需修正，**不是原版 SGLang**）。详细配置、精度与原始数据见[测速细节](docs/benchmarks.md)。

### Inkling-2-Small（594B / 17B，多模态）

512K 上下文 / 2 并发（40 NVFP4 + 8 MXFP8 MoE，5 层 DSpark + 23 层 ViT），SGLang `4ce41378f9`：

| 模式 | 单路 decode（bs=1） | 2 并发合计（bs=2） | 64K 冷 prefill | 256K 冷 prefill | 500K 冷 prefill | 500K 缓存 decode |
|---|---:|---:|---:|---:|---:|---:|
| **XTier + 加速包（默认）** | **74.7**（首轮 81.7） | **81.1**（稳态 89.8） | **3,986** | **2,514** | **1,627** | **56.0** |
| 基线（关加速包） | 47.9 | ~43 | 1,591 | 1,217 | OOM | OOM |

加速包对基线：64K / 256K 冷 prefill 提速 **2.41× / 2.07×**，短文 `bs=1` / `bs=2` 提速 **+56% / +89%**，NVFP4 KV（6.75 GiB）完整解锁 500K；开启通用 Recall 后 `agent_bench`（64K→500K）热态第 2 轮长回复升至 **73.6 tok/s**（`54.0 ms/步`，384K 接受长度 4.93 达 `74.8 tok/s`，Recall 落地 `3,793` token；旧两轮均值 66.2 tok/s）。精度：13 段 `ΔNLL +0.00423` / `KL 0.00938`，照抄 `0/8`，256K/500K 捞针全 `5/5`。方案 A+/B2 对比与解/预填剖析见[报告](docs/reports/2026-11-25-inkling2-small.md)。

### Hunyuan-4-Flash NVFP4

1M KV 池 / 1M 上下文 / 4 并发（混合 SWA+DSA NVFP4 KV 7.95 GB + 原生 5 层 DFly2），SGLang `4ce41378f9`：

| 模式 | 单路 decode（bs=1） | 4 并发合计（bs=4） | 100K 召回 prefill | 4×128K 并发 prefill | 876K–950K 冷 prefill | 超长缓存 decode |
|---|---:|---:|---:|---:|---:|---:|
| **XTier + 加速包 + DFly2（默认）** | **94.0**（两轮均 89.8，稳态 124–152） | **194.4**（稳态 225.5） | 3,630（PASS） | 3,565 | 3,454（876K） | **87.5–110.8**（876K） |
| 基线（全量 FP8 KV，BF16 稠密层） | 55–56 | 148–162 | 3,657（PASS） | — | 3,084（950K） | 43.8（950K） |

DFly2 图内融合通用 Recall（落地 908 token），照抄 `0/4` 差（双路达 299–306 tok/s），PPL 100% 无损，876K 缓存 decode 零 OOM 重试。

### DeepSeek V4.1 Flash 1017

[agent_bench.py](bench/agent_bench.py) 两轮 64K→500K 编码代理（加速包行测于 2026-11-28 `v0.1.14` / SGLang `4ce41378f9`，`48K / 37 槽 + mHC 原地写回 + 1.6 GiB 分配器水位墙 + 通用 Recall`）：

| DSpark | 轮 | 冷 prefill 64K | 冷 prefill 500K | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + 加速包** | 1 | 3,126* | **8,948** | 152.3 | 4.57 | 30.0 ms |
|  | 2 | **8,024** | **9,103** | **158.0** | **4.65** | **29.4 ms** |
| **XTier + 加速包，4 并发合计** | — | — | — | **268** | 3.40 | 51.0 ms |
| 基线（关加速包） | 1 | 3,318 | 3,627 | 65.5 | 3.25 | 50.9 ms |

*\*第 1 轮 64K 含 `--skip-server-warmup` 首次内核编译，同轮 256K–500K 为 `8,889–8,973 tok/s`，第 2 轮稳态 64K 破八千（`8,024`）、256K / 384K / 500K 全线破九千（`9,082 / 9,096 / 9,103 tok/s`）。*
加速包对基线：长回复每步快 **42%**，稳态 64K / 500K 冷 prefill 快 **142% / 151%**（100% 无损，PPL 3.6304，照抄正文 0 错）；含工具调用全部 169 请求合计 decode 跃升至 **174.0 tok/s**（接受长度 4.81，每步 **27.6 ms**，旧版 163.9 tok/s / 28.9 ms），全程 500K 显存峰值稳守 **69.52 GiB**（真实可用余量 `1.60 GiB`，0 次 PyTorch OOM 告警），2 并发新写 187.2–195.8 tok/s，4 并发改写/编辑达 332–382 tok/s。详见 [9K+ 无损预填报告](docs/reports/2026-11-19-lossless-9k-prefill.md) 与 [Recall 报告](docs/reports/2026-11-18-lossless-prefill-recall.md)。

### Qwen3.8 Flash Next

| MTP | 轮 | 冷 prefill 64K | 冷 prefill 500K | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + 加速包** | 1 | 5,715 | 5,170 | 181.7 | 2.57 | 14.2 ms |
|  | 2 | 5,610 | 5,150 | 207.2 | 2.68 | 13.0 ms |
| 基线（关加速包） | 1 | 5,319 | 4,078 | 139.5 | 2.13 | 15.2 ms |

加速包对基线：每步快 11%，冷 prefill 快 6% / 27%（NVFP4 KV + 自动规划，照抄 0 错）。`v0.1.14` 接入通用 Recall 后同版本 A/B 提速 **+5.1–7.4%**，2026-11-28 `agent_bench` 首轮长回复升至 **194.2 tok/s**（`13.1 ms/步`，384K 改写步达 `213–242 tok/s`，Recall 落地 `4,078` token）。

### GLM-5.3 Flash

| EAGLE | 轮 | 冷 prefill 64K | 冷 prefill 500K | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| **XTier + 加速包** | 1 | 2,416 | 2,505 | **96.9** | 2.52 | **26.0 ms** |
|  | 2 | 2,533 | 2,507 | **102.9** | **2.91** | 28.3 ms |
| 基线（关加速包） | 1 | 2,663 | 2,518 | 76.8 | 2.61 | 34.4 ms |

加速包对基线：开启通用 Recall + EAGLE 无损拒绝采样（2026-11-28 `v0.1.14` / SGLang `4ce41378f9`），两轮长回复合计破百达 **101.0 tok/s**（接受长度 `2.78`，对比基线提速 **+31.5%**，每步快 20–24%），全部 116 请求含工具调用合计 **105.5 tok/s**（接受长度 `2.94`，工具调用达 `149–241 tok/s`），FP8 稠密层 + 弹性热槽，照抄 0 错。

### MiniMax-M3.1-Flash（暂停开发）

SGLang M3.1 分支（PR #45925），64K 会话加速包对比官方代码 + 必需修正：长回复 51.7 对 46.1 tok/s（每步 55.4 对 84.1 ms，快 34%），见[报告](docs/reports/2026-10-24-minimax-512k.md)。

## 冷专家卸载与最小化运行

内存放不下全部专家时开启 `XTIER_COLD=1` 将少用专家放 NVMe（[设计](docs/design-cold-experts.md)），并支持 `tools/cold_punch.py` 让大模型在硬盘只存一份、按盘速分段（`XTIER_COLD_UNITS`）与尾部跳过（`XTIER_TAIL_SKIP`）。

- **4 块 PCIe 5.0 实测（DeepSeek 64K）**：卸载 10% / 20% / 30% / 50% 对应 decode `135 (−5%)` / `124 (−13%)` / `112 (−21%)` / `80 (−44%) tok/s`，详见[报告](docs/reports/2026-11-01-cold-pcie5.md)。
- **最小化运行（模拟 32 GB 显存 + 128 GB 内存，70% 冷卸载）**：DeepSeek 32K–128K prefill `1,155–1,186 tok/s`、decode `28.1–32.7 tok/s`，详见[报告](docs/reports/2026-10-27-ds41-minimal.md)。

## NVMe KV 二级卸载（默认关闭，按需开启）

多会话切换或并发超出 GPU KV 池时，可开启通用多盘 NVMe 二级 KV 卸载（`XTIER_KV_OFFLOAD=1`），将挤出的 KV 页经 `O_DIRECT + io_uring`（`libxtier_kvstore.so`，绕过 OS 页缓存）异步写入多块 NVMe 环形池，切回时直接从 NVMe 读回免冷 prefill。**开源默认关闭**（零钩子、零开销、无外部盘依赖），需要使用时在启动台选择 `100 GiB（4×25 GiB）` / `200 GiB（4×50 GiB）` 或通过环境变量 `XTIER_KV_OFFLOAD=1`、`XTIER_KV_PATHS=...`、`XTIER_KV_SIZE_GIB=<100|200>` 开启（编译需 `liburing-dev`，未安装时 `tools/build_engine.sh` 自动跳过）。

- **与模型无关 & 零 SGLang 补丁**：挂载在官方 SGLang HiCache `buffer_only` 动态存储后端上，核心引擎 `csrc/kvstore/` 与 `xtier/kvstore/` 零模型分支，自动覆盖 MLA / MHA / SWA / DSA / NVFP4 KV 与草稿侧池。
- **4× Samsung 9100 PRO PCIe 5.0 底层基准（200 GiB 环池）**：写入 **28.19 GB/s**、读回 **40.28 GB/s**、端到端 NVMe→Pinned→GPU **35.79 GB/s**（`bounce=0`，`Dirty ≈ 0`）。
- **4 模型端到端挤出→读回（`KV_POOL=64K`，全员照抄 `0/8`）**：DS4.1 命中 **99.2–99.4%**（TTFT **9.3–10.4×**）、GLM-5.3 **99.5–99.6%**（**17–21×**）、Hy4-Flash **99.8–99.9%**（**13–32×**，零拷贝 `11.37 GB/s`）、Qwen3.8 **99.9–100%**（**9.0–20.9×**，零拷贝 `11.39 GB/s`）。
- **DS4.1 生产 `1M KV 池 + 100 GiB NVMe` 真实 `dsh` 全链路实测**（锁页缓冲仅 `5.20 GiB`）：多会话 1.5M 流量挤出 1M 池后切回 370K 会话，`dsh` 耗时 **`49.73s → 4.86s`**（服务端 `<0.9s`，**364,032 token 100% 从 NVMe 读回**，命中 `99.94%`）；原生 `read_image` 图片输入 + 同文本不同图防碰撞（两路 **99.6%** 读回、零串味）与 `dsh` 自动压缩（压缩后 **98.2%** 读回）全程零报错（`puts=83,481 / hits=13,752 / misses=0`）。详见[报告](docs/reports/2026-11-26-nvme-kv-offload.md)。

## 版本更新

- **2026-11-28（v0.1.14）**：① **通用上下文草稿检索（Universal Recall）与零同步 GPU 后缀续写（`propose_after`）**（`xtier/packs/speculative/recall/`）：将 Recall 从 DeepSeek 专属抽离为通用模块，覆盖全部 3 类投机框架（DSpark / EAGLE / DFly2 CUDA 图内融合），默认对 Qwen3.8 / GLM-5.3 / Hy4-Flash / Inkling-2 / MiniMax / DS4.1 开启（`XTIER_RECALL=1`），主模型逐词验证 100% 无损；② **DeepSeek V4.1 mHC Prefill 原地写回与显存双防线**：新增融合 Triton 原地内核 `mhc_post_combine_norm_prefill_inplace`（`XTIER_DS_MHC_INPLACE=1`，48K 实省 `1.56 GiB` 峰值显存），修复 `xtier/cache/elastic.py` 暖池尾块低估冷启峰值的弹性悬崖，并引入 `0.009 ms` 动态 PyTorch 分配器水位墙（`XTIER_CACHE_ELASTIC_TORCH_GUARD_GIB=1.6`，遇真正 OOM 自动放开回退），将 DS4.1 默认热槽从 `48K/35、32K/43` 提升至 `48K/37、32K/44`，全程 500K 显存峰值稳守 `69.52 GiB`（`1.60 GiB` 真实可用余量，0 次 OOM 告警）；③ **300W 全模型官方重测刷新**：DS4.1 `agent_bench` 169 请求合计 `163.9 → 174.0 tok/s`（`27.6 ms/步`，次轮长回复 `158.0`、64K/500K 冷预填 `8,024 / 9,103`），GLM-5.3 长回复破百 `97.6 → 101.0 tok/s`（全部 `105.5`），Inkling-2 `bs=2` `78.4 → 81.1`（稳态 `89.8`）、热态 agent 长回复 `73.6 tok/s`，Hy4-Flash `bs=4` `186.8–190.4 → 194.4`（稳态 `225.5`）。
- **2026-11-28（v0.1.13）**：新增三项零 SGLang 补丁稳定性与冷启动修复：① `XTIER_KV_OFFLOAD=1` 下的 HiCache 混合驻留/落盘前缀过度承诺修复（仅统计已驱逐节点，并在上游未合入前锁定 Python `TreeCore` 防止 Rust `full.rs` 绕过）；② `SWAComponent.prepare_for_caching_req` 完整 span Radix 插入修复（消除新会话部分命中旧 SWA-less 前缀后的缓存冻结死点）；③ `EagleDraftWorker._capture_cuda_graphs` 启动期 Eager 草稿核与 QSA 多步草稿图元数据预热（`XTIER_DRAFT_PREWARM=1` 默认开启，消除首步 ~500 ms `cuModuleLoadData` 卡顿与 TP>1 死锁，零修改 `eagle.py` 规避越界读）。
- **2026-11-26（v0.1.12）**：修复一些 bug。
- **2026-11-26（v0.1.11）**：新增**通用多盘 NVMe KV 二级卸载**（`XTIER_KV_OFFLOAD=1`，**默认关闭**，启动台可选 `100 GiB` / `200 GiB` 按需开启）：`O_DIRECT + io_uring`（4×9100 PRO 写入 `28.19 GB/s`、读回 `40.28 GB/s`、端到端 `35.79 GB/s`），零 SGLang 补丁、零模型分支，DS4.1 / GLM-5.3 / Hy4-Flash / Qwen3.8 四模型挤出读回命中 **99.2–100%**（TTFT **9–32×**），DS4.1 `1M 池 + 100 GiB` 真实 `dsh` 多会话/图片多模态/自动压缩全链路零报错通过；同时打包合入 11-25 Prefill 100% 无损提速与社区 PR #20（弹性缓存 `release()` 排空保护）。[报告](docs/reports/2026-11-26-nvme-kv-offload.md)
- **2026-11-25**：**Prefill 100% 无损提速**——新增零补丁融合 `RelLogitsProj + ShearingBias` Triton 核（逐比特 `max_diff=0`，64K 省 645 ms 与 6.9 GiB 峰值显存）、显存感知 1-band `_reduce_band`（`B=512`）、`_mxfp8_reduce_runs` 活跃 run 归约、向量化 `_act_quant_fp8` + SM120 `stages=4` W8A8，以及精确 slab `cudaHostRegister` `_share_buffers` 与冷专家 `CPU_SHARE` 支持；Inkling-2 64K 冷 prefill `3,835 → 3,986 tok/s`（稳态块 ~4,250）、500K `1,593 → 1,627 tok/s`（`-6.6 s`），照抄 `0/4`（`c=1/2`）、PPL `2.7172`（ΔNLL `+0.00005`）零损。[报告](docs/reports/2026-11-25-inkling2-small.md)
- **2026-11-25（v0.1.10）**：新增 **Inkling-2-Small（594B 多模态）** 零补丁加速包与启动台卡片（默认方案 A+：`cold=2%`、45 热槽、64K 单块 3,835 tok/s、NVFP4 KV 解锁 500K、两轮 agent 66.3 tok/s；`cold=off` 方案 B2 仅 `L47` 全驻），并打包合入 11-24 的 DeepSeek 无损 9K+ 优化。[报告](docs/reports/2026-11-25-inkling2-small.md)
- **2026-11-24**：DeepSeek V4.1 与通用层新一轮 100% 无损提速（24-token 权重复用、43 层 FP8 `wo_a`、写回融合、SM120 Eager MXFP8 快分发），500K 冷 prefill 双破 9,000 tok/s，4 并发新写达 268 tok/s。
- **2026-11-23（v0.1.9）**：新增 **Hunyuan-4-Flash NVFP4** 零补丁加速包与原生 5 层 DFly2 投机解码（1M 混合 NVFP4 KV 7.95 GB，bs=1 92.1 / bs=4 190 tok/s），上线 `tools/fast_verify.py` 三层快速验证，合入社区 L20（SM89）等 PR（#18/#7/#19）。
- **2026-11-22（v0.1.8）**：跟进官方 SGLang `4ce41378f9`，打包 9,000+ 无损 prefill / 1.1 s 工具回传优化与默认 CPU 多模态解码限流。[报告](docs/reports/2026-11-22-sglang-4ce4137.md) · [9K 报告](docs/reports/2026-11-19-lossless-9k-prefill.md)
- **2026-11-18（v0.1.7）**：全链路无损 prefill 融合、Engram 跨块预热与 DSpark 上下文草稿检索（Recall，agent 全请求 180.4 tok/s）。[报告](docs/reports/2026-11-18-lossless-prefill-recall.md)

更早的版本更新与完整细节见 [docs/changelog.md](docs/changelog.md)。

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
