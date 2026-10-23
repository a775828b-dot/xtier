# 2026-10-23 第 3 步：DeepSeek V4.1 1017 上官方 main，大表放 NVMe，Zen2 / 双卡准备

硬件：RTX PRO 5000 72GB（SM120）+ 2× EPYC 9334（NPS4，8 个 NUMA 节点）+ 384 GiB DDR5；官方 SGLang main 8cd3c494。

## DeepSeek V4.1 Flash 1017

布局同生产：40 个 MoE 层全部由 XTier 计算，每层 50 个热专家槽（生产画像 + 自适应换入换出），Engram 两张表（189 GB）
走 NVMe 行表（每层 2 GiB 行缓存，scale 预读），512K 上下文、22528 token prefill 分块、两并发
（`configs/ds41-pro5000-nps4/run-ds-xtier.sh`）。

### 在官方 main 上跑起来要修的地方

| 问题 | 现象 | 处理 |
|---|---|---|
| Engram 表 189 GB + 专家 276 GB 超过 377 GB 内存 | 官方只有显存表和主机表 | `xtier/tables` 行表 + `packs/deepseek_v4` 钩子（`EngramEmbedding.__init__` / `_owned_rows`） |
| 官方在 SM120 上给专家选 `Mxfp4FlashinferCutlassMoEMethod` | XTier 格式检测不认识 | 认作 MXFP4（参数与 `Fp8MoEMethod` 相同，scale 直接是 E8M0 字节，W13 按 [up; gate] 加载） |
| 候选索引器 | 官方把 SM120 当成 SM100+，要 DeepGEMM 的 paged sparse MQA logits（只在 SM100 有）→ 启动报错 | SM120 走官方给 SM100 以下准备的 dense-block 候选（DeepGEMM 在 SM120 有 dense FP8/FP4 MQA logits） |
| 调度进程被绑到 GPU 的 NUMA 节点 | 主进程用 numactl 启动调度子进程（`SGLANG_NUMA_BIND_V2`），与 XTier 的主线程 CPU 相交后只剩 2 个核，且"优先节点 0"的内存策略把调度进程自己的分配全压到节点 0，第 6 次启动时节点 0 的专家分片申请不到页被 OOM 杀掉（`CONSTRAINT_MEMORY_POLICY, nodemask=0`） | XTier 管放置时在主进程设 `SGLANG_AUTO_NUMA_BIND=0`；行表的大块内存（行缓存、scale）按节点交错分配 |
| 官方 Triton 块 FP8 GEMM 没有 SM120 上这些形状的配置 | 解码一步 68 ms 中 63 ms 在 `_w8a8_block_fp8_matmul`（20 步 4200 次，每次 302 µs） | 生产调好的 11 个配置放进 `packs/deepseek_v4/fp8_configs`，钩子优先查包里的 |
| 尾部有界重放（`--enable-decoder-swa-bounded-replay`） | 官方在服务中按行数分桶捕获尾部层的 CUDA 图（最多 2048 行），MoE 调用在捕获里成了 2048 行的"解码"，超出 XTier 的信箱，2K 冷 prefill 时服务退出 | 尾部层不用图、照常 eager（重放本身不变） |
| prefill 分档 | 生产 lkqmoe 的 hybrid 是"专家平衡切分"（GPU 算前 g 个专家、CPU 同时算其余），XTier 的缓存式 hybrid 把未命中的路由全给 CPU：4K 段每段约 18 s | 移植平衡切分（`XTIER_SPLIT_PREFILL_MAX=6144`），缓存式 hybrid 只用于 1024 以下 |

### 结果

| 项目 | XTier（官方 main，无投机） | 生产（lsglang_ds4.1，DSpark） |
|---|---|---|
| 生成 | 正确（中文问答、代码） | — |
| PPL（`bench/ppl_eval.py`，13 段 × 4K） | 3.3187（docs 3.235、agent 2.983、zh 7.336）；缓存式混合路径时 3.316 | 测不了：开着尾部有界重放时生产分支拒绝返回提示词 logprob，并因此退出 |
| 照抄（8 段，温度 0） | 单并发、双并发均 0 错（单并发 1 段末尾多一个 `</src>`，生产也有的已知现象）；FP8 配置换上后复测仍 0 错 | 0 错 |
| 解码（`bench/decode_speed.py`，512 token） | 无投机：单请求 33.9 tok/s，双请求合计 46.0 tok/s；短回答约 44 tok/s（修前 14.8，68 ms/步） | DSpark：单请求 120.3 tok/s，双请求合计 114.8 tok/s |

冷 prefill（`bench/ttft_probe.py`，无前缀缓存，两次取中位）：

| 提示长度 | 500 | 1K | 2K | 4K | 8K | 16K | 22.5K |
|---|---:|---:|---:|---:|---:|---:|---:|
| XTier | 2.24 s | 2.79 s | 3.80 s | 4.28 s | 4.99 s | 5.41 s | 5.81 s |
| 生产 | 1.42 s | 2.34 s | 2.83 s | 3.34 s | 3.87 s | 4.58 s | 5.25 s |
| XTier / 生产 | 1.58 | 1.19 | 1.34 | 1.28 | 1.29 | 1.18 | 1.11 |

DSpark（静态 6-token 验证，草稿 3 层 128 专家由 XTier 全常驻）：照抄单/双并发 0 错；`bench/decode_speed.py` 提示上的平均
接受长度 XTier 3.24、生产 3.31（接受得一样多）；单请求 XTier 70.0 tok/s（缓存热起来后）对生产 120.3。热专家命中率
（0.52–0.56）、每层不同专家数（约 22）、换入次数（365 轮约 23360 次）与生产一模一样，差在 **CPU 每层调用 522 µs 对生产
384–401 µs**，以及 GPU 侧（解码轨迹里主计算流只有 36% 时间在算，其余在等 CPU 专家）。修过的问题：自适应换入换出原来只在
prefill 调用时启动，DSpark 加短提示时从不启动（命中率停在画像初值 0.3）。

解码一步约 23 ms 时的分布（20 步 GPU 轨迹）：块 FP8 GEMM 13.5 ms（按带宽算 M=1 时约 4 ms 可读完，还有空间）、
cuBLAS BF16 GEMV 约 3 ms、GPU 上的热专家约 3 ms、等 CPU 专家约 2 ms；CPU 专家每层调用 130–200 µs，热专家命中率约 0.7。

## 大表放 NVMe（`xtier/tables`）

见 [design-tables-nvme.md](../design-tables-nvme.md)。CPU 逐字节（`tests/tables_check.py`）：DS 第 1、14 层（整表、TP 分片，
scale 读盘 / 预读）与 Qwen 128 分片表（含每个分片边界）全部一致；GPU 图内逐位（`tests/table_lookup_check.py`）：DS 对
`engram_gather`、Qwen 对官方 pinned 表 Triton 内核，直接调用与 CUDA 图重放换 id 都一致。

Qwen3.8 端到端（生产布局 + MTP，同一代码同一时段；`XTIER_PLE_ROWS=nvme`，傲腾一列用 `XTIER_PLE_MODEL` 指向傲腾上的分片副本）：

| | 锁页表（47.7 GiB 主机内存） | NVMe·EX900（无 DRAM） | NVMe·傲腾 P5800X |
|---|---|---|---|
| 验证步（`bench/verify_step_bench.py`） | 14.13 ms，167.2 tok/s | 15.08 ms，155.3 tok/s | 14.14 ms，166.8 tok/s |
| 接受长度 | 2.37 | 2.35 | 2.36 |
| 照抄单 / 双并发 | 1/8、1/8 | 0/8、0/8 | 0/8、0/8 |
| PPL | 3.4813 | 3.4775 | 3.4736 |
| 冷 prefill 1K / 4K / 8K / 30K | 0.80 / 1.10 / 1.61 / 5.45 s | 0.93 / 1.33 / 2.39 / 6.81 s | 0.80 / 1.11 / 1.66 / 5.51 s |
| PLE 占主机内存 | 47.7 GiB | 4 GiB 行缓存 | 4 GiB 行缓存 |

锁页表那列的照抄差异是同一处（`PREFILL_TRUNCATION` 抄成 `PREFILL_TRUNCATE`，源文件里另有这个常见写法，两种写法几乎打平）；
关掉自适应换入换出（驻留固定）后锁页表也是 0/8、0/8：翻转来自热专家的换入历史（命中路由在 GPU W4A16 与 CPU 引擎之间的
分配不同），不是解码路径的错误。PPL 早上几次在 3.4726–3.4769，生产 3.4779，都在这个噪声范围内。

## GLM-5.3：BF16 注意力 / 稠密层存成 FP8（`packs/glm5_next/fp8_weights.py`）

从 GLM-5.3 管线的 `lkqmoe/gpu/fp8_weights.py` 移植生产实际用到的部分（`XTIER_FP8_WEIGHTS=1`，选择规则、W8A8 prefill 都照生产
的启动参数和运行时文件）：加载后 191 个 BF16 线性层存成 FP8（每行每 128 列一个 FP32 scale），13.42 → 6.92 GiB，NextN 草稿另有
5 层；解码 M ≤ 16 走单次启动的 split-K W8A16 内核（确定性求和，图内可捕获），prefill 走 W8A8。单测
`tests/fp8_weights_check.py`：GLM 的 8 种形状，解码内核对 FP64 参考 1.7e-3、CUDA 图重放逐位一致、W8A8 2.6e-2。同时把 GLM 的
prefill 分档对齐生产的运行时文件（640 以下缓存式混合、1792 以下专家平衡切分）。

EAGLE（README 参数），`bench/decode_speed.py` 512 token：

| | XTier FP8 | XTier BF16（早上） | 生产 |
|---|---|---|---|
| 单请求 | 90.9 tok/s | 64.7（当前代码 76.6） | 82.3 |
| 双请求合计 | 113.1 tok/s | 89.7（当前代码 100.2） | 112.4 |
| PPL | 3.9982 | 3.9869 | 4.0598 |
| 照抄单 / 双并发 | 1/8、0/8；驻留固定时 0/8、0/8 | 0/8、0/8 | — |

FP8 那次单并发的差异是段尾一个单独的 `(` 少缩进 4 格；关掉自适应换入换出后 FP8 也是 0/8、0/8，与 Qwen 锁页表那次一样是换入历史
造成的接近平局翻转（命中路由在 GPU W4A16 与 CPU 引擎之间分配不同，生产的 LKQstrata 也有这个性质）。BF16 当前代码比早上快
（76.6 / 100.2 对 64.7 / 89.7）来自本步的 NUMA 放置与自适应启动两处修复。

## Zen2 / 双卡准备（未在实机上跑）

见 [deploy-zen2-dual-gpu.md](../deploy-zen2-dual-gpu.md)。3.23 上能验证的部分：`tools/isa_check.sh`（AVX2 主机加载的 4 个库
0 条 AVX-512 指令）；`tests/tp_check.py`（GLM-5.3 第 20 层切成两片、各在 4 个 NUMA 节点上算，之和对整层相对误差
1.2e-7，AVX-512 与 AVX2 内核都是；AVX2 整层对 AVX-512 整层 4.9e-5）。
