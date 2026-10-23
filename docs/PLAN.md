# XTier 工作计划

## 目标

在官方 SGLang（跟 main 分支）上，用一个插件包实现 CPU/GPU 分层 MoE 推理：专家分布在显存、内存和 NVMe 三层，按热度放置、按需调度。
三个模型（DeepSeek V4.1 Flash 1017、Qwen3.8 Flash Next、GLM-5.3 Flash）的速度和精度对齐现在的生产管线后切换过去；以后加新的 MoE 模型不需要写混合推理代码。

## 原则

- **不改 SGLang 源码**：只用官方插件框架（`sglang.srt.plugins` 的钩子）。官方改名或删掉了钩子目标，启动时直接报错。
- **与模型无关的核心**：混合推理包在官方 `FusedMoE` 的量化方法外面，任何用标准 MoE 层的模型装上就能跑。
- **加速包可选**：某个模型独有的优化放在 `xtier/packs/<官方模型模块名>/`，不装也能正确运行；能通用的优先提交给官方。
- **生产不动**：开发用独立环境、独立端口；每个模型达标才切换，旧管线保留到稳定为止。
- **用数据验收**：每一步都有门槛，基线就是现在三条生产管线的实测。
- **自己确认生效**：SGLang 会吞掉插件里的异常，XTier 加载模型时检查后端确实接管了 MoE，没接管就报错退出，不允许悄悄退回纯 GPU。

## 架构

```
SGLang 官方 main（固定到验收通过的提交号，每周同步）
└─ xtier（本仓库，一个包、一个版本号）
   ├─ backend/   混合 MoE 后端：包装官方量化方法；权重分流、decode 调度、prefill GPU 路径、CUDA 图兼容
   ├─ cache/     全局专家缓存：所有层共用显存预算，按热度和收益放置、在线换入换出；一个 GPU 专家内核
   ├─ planner/   自动规划器：探测显存、PCIe、CPU 内存带宽和 NUMA，结合路由统计算出缓存、分块、线程，目标 CPU 和 GPU 同时算完
   ├─ engine/ + csrc/   CPU 引擎：AVX-512 / AVX2 专家内核、NUMA 线程池、CPU 与 GPU 之间的信箱通信、预取
   └─ packs/     加速包：deepseek_v4、qwen4_exp、glm5_next
```

## 阶段

### 0. 盘点与基线（2–3 天）

- 0.1 改动清单：把三条生产管线和 CPU 引擎相对官方 SGLang 的每一处改动按**功能**列出来，每项标落点：
  XTier 后端 / 加速包 / 官方已有（删）/ 临时绕法（删）。产出 `docs/inventory.md`。
- 0.2 基线固化：三条管线现在的验收数据整理成机器可读的基线文件和测法脚本（`bench/baselines/`）。
- 0.3 官方 main 摸底：新环境装官方 main，确认三个模型各自的 MoE 调用路径、量化方法、投机解码方式；Qwen 专用 decode 路径
  （`qwen4_decode_moe`）的接法。

### 1. 骨架与 CPU 引擎迁移（约 1 周）

- 构建（pyproject + C++ 编译）、插件入口、声明式配置格式（取代现在分散的环境变量）。
- CPU 引擎源码迁入 `csrc/` 并公开；去掉旧的兼容接口和环境变量。
- 门槛：单层输出与旧引擎 0.5.6 **逐位一致**，单层耗时不慢（三个模型的实权重单层对照）。

### 2. 混合 MoE 后端 + 全局专家缓存，先做 Qwen（约 2 周）

- 包装 `FusedMoE` 量化方法：CPU 专家权重直接进 NUMA 内存，缓存的专家进显存；decode 时命中缓存的走 GPU，其余走 CPU 信箱；
  prefill 走 GPU 上传流水；CUDA 图可捕获。
- 全局专家缓存取代"整层常驻 + 每层热专家池"；GPU 专家内核先做微基准，对齐 Marlin 和现在热专家池内核（0.94–0.95 TB/s）。
- qwen4_exp 加速包：hc_mix 重叠内核等（以盘点结果为准）。
- 门槛（对比现生产）：每个验证步不慢于 13.75 ms 的 2%；2 并发合计不低于 202 tok/s 的 98%；PPL 3.48 在噪声内；
  照抄测试单请求 / 两请求 0 错；512K 召回正确。

### 3. 自动规划器（约 1 周，可与第 4 步并行）

- 硬件探测 + 路由统计 → 缓存大小、分块、线程、并发下的显存预算，按"CPU 和 GPU 同时算完"分配。
- 门槛：三个模型自动配置的结果不差于现在手工调出来的生产配置。

### 4. GLM 加速包（约 1 周）

- glm5_next：EAGLE 自适应步数、无损拒绝采样、截断验证、NoPE 稀疏 MLA（sm_120）等（以盘点结果为准）。
- 门槛：8 点测速平均不低于 106.9 tok/s 的 98%；2 并发合计不低于 112.4 tok/s 的 98%；PPL 4.06 在噪声内；照抄 0 错；512K 召回。

### 5. DeepSeek 加速包（约 2 周）

- deepseek_v4：Engram NVMe（支持多卡）、DSpark 固定验证、MXFP4 适配、FP4 索引器、官方编码等（以盘点结果为准）。
- 门槛：agent_bench decode 不低于 144–157 tok/s 的 98%；冷 prefill 不慢（64K 3.4K、128K 4.2K tok/s，522K 首字 142.7 s）；
  PPL 在噪声内；照抄 0 错；512K 召回；2 并发正常。

### 6. 切换生产（每个模型达标即切）

- WebUI 加新档位，和旧管线并行一段时间；旧管线保留回滚，稳定后归档。
- 原来三个公开仓库（lsglang_ds4.1 / lsglang_glm5.3 / lsglang_q3.8fn）改为部署配方：启动配置、测速结果和报告。

### 7. 持续：每周跟 main

- 拉官方 main → 单元测试 + 三个模型的快速验收（PPL、照抄、短测速）→ 通过才更新受支持的提交号，写进版本说明。

## 时间

约 6–8 周：盘点 0.5 周，骨架与引擎 1 周，后端与 Qwen 2 周，规划器 1 周（与 GLM 并行），GLM 1 周，DeepSeek 2 周。
开发期版本号 0.x，三个模型都过门槛后发 1.0。

## 风险与对策

| 风险 | 对策 |
|---|---|
| 官方 main 变化快，钩子目标会变 | 固定提交号，每周验收后再升级；钩子挂不上时启动即报错 |
| 现生产有不少深入 SGLang 核心的优化 | 盘点逐项判断；官方已有的删掉；必须保留的做成整函数替换钩子，并尽量提交给官方 |
| 全局缓存的 GPU 内核达不到 Marlin | 第 2 步先做内核微基准，达不到就保留整层路径作为缓存的特例 |
| 插件异常被 SGLang 吞掉 | 加载模型时自检，后端未接管即报错 |
| 测试要用 GPU | 测试窗口停 DS 生产服务，按现在的做法 |

## 当前状态

- 2026-10-22：确定架构、名字（XTier）和本计划；仓库公开；开始第 0 步。
- 2026-10-22 晚：0.1 盘点 v1（[docs/inventory.md](inventory.md)）、0.2 基线（`bench/baselines/`）完成；0.3 官方 main 环境安装中。
- 2026-10-23：第 1 步 CPU 引擎迁入 `csrc/`（改名移植，删掉旧兼容入口）、Python 绑定 `xtier/engine`、构建脚本；三个模型的真实单层在 AVX512 和 AVX2 下都与 0.5.6 逐位一致（`bench/results/engine-ab-20261023.json`）。
- 2026-10-23 凌晨：第 2 步进行中。后端（`xtier/backend`）接管官方 `FusedMoE`：插件钩住官方 KTransformers 混合推理的入口（`create_kt_config_from_server_args` / `KTEPWrapperMethod`），专家权重按检查点自己的量化方法的布局建在主机内存，每层一加载完就交给 CPU 引擎并原地释放；解码走信箱（CUDA 图可捕获），prefill 移植 0.5.6 的 GPU 分组 GEMM。Qwen3.8 在官方 main 8cd3c494 上 48 层全走 CPU 引擎：PPL 3.4659（13 段 × 4K，生产 3.4779），生成正确。全局热专家缓存（`xtier/cache`，每层槽数可不同，全部常驻的层不经 CPU）单层验证通过。顺带修了官方 main 在这台机器上的三个问题：PLE 锁页表在 NPS4 上分配不了（`packs/qwen4_exp`）、TileLang 在装了 ROCm 的 CUDA 机器上选错目标（`xtier/platform.py`）、transformers 5.19 读不了这个检查点的分词器（另存带 tokenizer.json 的分词器，用 `--tokenizer-path`）。
- 2026-10-23 凌晨（续）：Qwen3.8 第 2 步基本达标：按生产布局开热专家缓存（25 层全常驻 + 23 层各 155 槽）、MTP、解码 CUDA 图，PPL 3.4769（生产 3.4779），照抄 0/8、0/8，解码单请求 139–145 tok/s、2 并发 200–207 tok/s（生产 145.6 / 202.2），4K 冷 prefill 1.13 s（生产 1.07 s）。全常驻层的 prefill 改为直接在缓存槽上算。详见 [reports/2026-10-23-step2-qwen.md](reports/2026-10-23-step2-qwen.md)。未完成：512K 召回、NVFP4 KV、每步 2–7% 的差距剖析。
- 2026-10-23 清晨：Qwen3.8 加上 hc_mix 重叠内核（`packs/qwen4_exp`）与 XTier 草稿后，验证步 13.98 ms（生产 13.78）、解码 166.6 tok/s（生产 162.2）。GLM-5.3 在官方 main 上跑通（不开投机，42 层全由 XTier 接管），PPL 3.9869（生产 4.0598）；EAGLE 与 kpool 下的 sparse MLA 待做。
- 2026-10-23 早上：Qwen3.8 长上下文 4K–500K 召回全部正确（FP8 KV；官方 QSA 不支持 NVFP4 KV）；GLM-5.3 的 EAGLE 跑通（NextN 草稿保持 NVFP4 并由 XTier 全常驻），速度约为生产的 80%；DeepSeek 1017 的 MXFP4 格式已适配，端到端受 Engram 表（189 GB）所限，下一步做 Engram NVMe 行存储包。DS 生产已恢复。
- 2026-10-23 早上（续）：部分缓存层的混合 prefill（命中路由在 GPU 槽上算、其余同时在 CPU 上算）与常驻层 prefill 调参，Qwen 冷 prefill 1000 token 0.95→0.62 s、4K 1.09 s（生产 1.07）、8K 1.58 s（生产 1.49）；验证步 13.98 ms 在门槛 2% 以内。第 2 步 Qwen 剩余：8K 以上的长提示 prefill、QSA 的 NVFP4 KV。
- 2026-10-23 早上（续）：`packs/deepseek_v4` 移入生产的 Engram NVMe 行存储（`csrc/engram`）与图内查表，两者都对照直接读文件逐位一致；下一步是 `EngramEmbedding` 的钩子和 DS 1017 端到端（见 [design-tables-nvme.md](design-tables-nvme.md)）。
