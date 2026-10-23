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

### 加载时的内存：页缓存与 swap

修之前，DS 加载时页缓存一路涨到 220 GB 左右，专家分片（约 290 GB）也在同时涨。空闲内存一用完，MGLRU 就连匿名页一起换出：
12:21 那次锁页用了 553.8 s（约 20 GB 在 swap 里），14:13 那次到第 19 层时调度进程已经换出 50 GB。原因有两个：

1. 官方 DeepSeek 加载器把每次主机张量拷贝都丢进线程池（`maybe_executor_submit`），读文件的迭代器跑在拷贝前面，很多层同时在
   填，而且拷贝发生在迭代器丢掉页缓存之后，页又被读了回来。现在 XTier 的专家张量在加载线程上按检查点顺序拷贝
   （`backend/hooks.py`），一个文件（一层）拷完才走下一个。
2. `--weight-loader-drop-cache-after-load` 用 `POSIX_FADV_DONTNEED` 丢页缓存，可 safetensors 的懒张量是整个文件一个私有映射，
   只要还有张量活着（迭代器刚交出去的最后一个、线程池里排队的拷贝），页就都还映射着，DONTNEED 一页也丢不掉（单测：张量还在时
   2.52 GiB 原样留着）。现在 XTier 先对本进程里这个文件的映射做 `MADV_PAGEOUT`，干净的文件页直接离开页缓存（单测降到 0，重读
   校验 0 错），再让 SGLang 丢剩下的（`platform.py`）。另有一个后台线程把后面两个检查点文件读进页缓存（同时最多 2 个，超过
   16 GiB 的 Engram 文件不读；`POSIX_FADV_WILLNEED` 只读一个预读窗口，所以直接读文件），拷贝走页缓存，不再一页一页缺页。

| DS 1017 + DSpark | 修前（12:21） | 只改顺序拷贝（14:26） | 顺序拷贝 + page-out（14:36） | 再加提前读（15:04） |
|---|---|---|---|---|
| 加载时页缓存 | 涨到约 220 GB | 第 16 层时已 213 GB | 9–14 GB | 11–21 GB |
| 调度进程 swap | 约 20 GB（14:13 那次第 19 层 50 GB） | 空闲内存用完就开始涨，停掉 | 0 | 0 |
| 权重加载 | 150.8 s | — | 413.3 s | 324.3 s |
| NUMA 归位 + 锁页 | 锁页 553.8 s | — | 7.7 + 20.3 s | 10.5 + 20.8 s |
| 启动到就绪 | 811 s | — | 497 s | 409 s |

GLM / Qwen 的启动脚本原来没开 `--weight-loader-drop-cache-after-load`：GLM 加载时页缓存堆在加载线程所在的 0/1/4/5 号节点，这几个
节点满了就换出了 3.3 GB，而别的节点还空着 160 GB；Qwen 有一次换出 4.3 GB（锁页 1.6 → 20.5 s）。现在两个启动脚本都开了，加载完
页缓存只剩 3–7 GB，swap 为 0。代价是 Qwen 的 MTP 草稿那一遍加载 17 → 64 s（草稿要的张量得从盘上重读），就绪 316 → 342 s。
GLM 还按 `--mem-fraction-static 0.92` 开了 107.8 万 token 的 KV 池（6 GB，是 512K 的两倍），只剩 4.9 GB 给 XTier 的 prefill
缓冲，32K 提示就显存不够；启动脚本改成 `--max-total-tokens 524288`。

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

## README 测速

`bench/context_speed.py`，用 `readme_speed.sh` 每个配置起一次服务：先用 32K 提示预热，再对 64K 和 510K 的提示各测一次冷
prefill（先清前缀缓存），之后同一提示命中前缀缓存再重发 4 次，5 次回答各自采样、各出 1K token。decode 取 5 次的中位数，
每步耗时来自 SGLang 返回的投机统计（`return_spec_tokens_details` 的验证步数）。原始数据 `bench/results/context-*-20261023.json`。

| 配置 | 上下文 | 首 token | prefill | decode 中位（5 次） | decode 范围 | 每步 | 接受长度 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen-base | 64K | 12.3 s | 5319 | 140.7 | 132.8–153.1 | 14.63 ms | 2.10 |
| qwen-base | 510K | 125.1 s | 4078 | 138.3 | 118.7–148.8 | 15.75 ms | 2.16 |
| qwen-pack | 64K | 12.5 s | 5268 | 154.9 | 131.4–171.0 | 14.34 ms | 2.22 |
| qwen-pack | 510K | 125.1 s | 4076 | 154.7 | 149.1–160.6 | 15.50 ms | 2.38 |
| glm-base（55 槽） | 64K | 28.3 s | 2320 | 79.5 | 71.8–81.2 | 35.73 ms | 2.72 |
| glm-base（55 槽） | 510K | 231.6 s | 2203 | 69.6 | 62.2–71.5 | 32.39 ms | 2.23 |
| glm-base（50 槽） | 64K | 28.2 s | 2327 | 72.7 | 71.8–78.8 | 35.65 ms | 2.66 |
| glm-base（50 槽） | 510K | 229.8 s | 2220 | 68.2 | 54.3–84.9 | 36.03 ms | 2.48 |
| glm-pack（60 槽） | 64K | 27.5 s | 2389 | 90.6 | 83.1–95.0 | 28.75 ms | 2.61 |
| glm-pack（60 槽） | 510K | 224.2 s | 2276 | 87.4 | 81.3–91.6 | 27.46 ms | 2.39 |
| ds-base | 64K | 19.8 s | 3318 | 64.9 | 58.5–69.2 | 49.93 ms | 3.16 |
| ds-base | 510K | 140.6 s | 3627 | 66.1 | 60.6–69.5 | 51.83 ms | 3.34 |
| ds-pack | 64K | 16.6 s | 3962 | 89.5 | 84.4–96.5 | 36.78 ms | 3.26 |
| ds-pack | 510K | 116.9 s | 4363 | 80.2 | 76.6–84.6 | 38.20 ms | 3.07 |

- 单次 decode 的 tok/s 跟着接受长度走（同一配置 5 次能差 30%：Qwen 加速包第一轮 64K 时 135–197 tok/s）。草稿长度固定的
  Qwen、DeepSeek 每步耗时 5 次之间只差 1–3%；GLM 开了自适应草稿步数（1–5 步），每步耗时随接受长度变。比较加速包看每步耗时：
  Qwen 64K 14.63 → 14.34 ms（−2%）、510K 15.75 → 15.50 ms（−2%）；GLM 64K 35.73 → 28.75 ms（−20%）、510K 32.39 → 27.46 ms（−15%）；DeepSeek 64K 49.93 → 36.78 ms（−26%）、510K 51.83 → 38.20 ms（−26%）。
- 第一轮只测 1 次、没有预热：64K 的首 token 带着内核编译和调参（DS 加速包 20.7 s，预热后 16.6 s），Qwen 的 decode 还出现了
  "官方 main 比加速包快"（146.2 对 138.5），都是单次采样的波动，所以改成上面的口径重测了全部配置。
- GLM 第一次跑 32K 预热时显存不够：KV 池按 `--mem-fraction-static 0.92` 开到 107.8 万 token（6 GB，是 512K 的两倍），只剩
  4.9 GB 给 XTier 的 prefill 缓冲和 KDA 线性注意力。启动脚本改成 `--max-total-tokens 524288`，加速包那一行之后剩 8.0 GB。
  官方 main（BF16 权重）在生产的 60 槽时只剩 5.7 GB，同样在 32K 预热时显存不够（KDA 的分块 delta 规则）。先试了 50 槽（余量
  11.2 GB，比加速包那行还多 3 GB，所以不算同等显存，表里留作参照），README 用 55 槽（余量 8.4 GB）。FP8 权重实际省下的显存只有
  约 2.3 GB（不是按权重大小算的 6.5 GB）。
- DS 重发时首 token 1–5 s（Qwen、GLM 0.2–0.8 s），原因没有细查，不影响 decode 的测量。

## Zen2 / 双卡准备（未在实机上跑）

见 [deploy-zen2-dual-gpu.md](../deploy-zen2-dual-gpu.md)。3.23 上能验证的部分：`tools/isa_check.sh`（AVX2 主机加载的 4 个库
0 条 AVX-512 指令）；`tests/tp_check.py`（GLM-5.3 第 20 层切成两片、各在 4 个 NUMA 节点上算，之和对整层相对误差
1.2e-7，AVX-512 与 AVX2 内核都是；AVX2 整层对 AVX-512 整层 4.9e-5）。
