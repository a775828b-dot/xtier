# 部署：Zen2 + 双卡（2× EPYC 7642 + 2× RTX 5070 Ti 16GB）

状态（2026-10-23）：**已准备，未在这类机器上跑过**。下面的经验来自 2026-10-17 用 lkqmoe 0.5.3 在同一台机器上部署
GLM-5.3 TP2（Lsglang 分支）；XTier 的对应部分已移植并在 3.23（Zen4、单卡）上做了能做的验证，实机到手后按文末清单上线。

## 目标机器

| 项目 | 配置 | 影响 |
|---|---|---|
| CPU | 2× EPYC 7642（Zen2，48 核 / 96 线程 ×2），**无 AVX-512** | 引擎走 AVX2 内核；分发的库里不能混进 AVX-512 指令 |
| NUMA | NPS4，8 个节点；GPU0 在节点 2，GPU1 在节点 7（两颗 CPU 各一张） | 每个 TP rank 用自己 GPU 那颗 CPU 的 4 个节点 |
| 内存 | 16×16GB DDR4-3200，256GB，槽已满 | GLM-5.3 专家约 164 GB 放得下；DS 1017（专家 276 GB）放不下 |
| GPU | 2× RTX 5070 Ti 16GB（SM120），PCIe 4.0 x16，两卡之间 SYS（无 P2P） | TP=2，`--disable-custom-all-reduce`；prefill 受 PCIe 4.0 限制 |
| 盘 | 一块 1TB PCIe 4.0（系统盘兼模型盘） | 大表放 NVMe 时与系统共用一块盘 |
| 网络 | 有线网口无地址 | `GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo` |

## XTier 为此做了什么

| 项目 | 做法 | 3.23 上的验证 |
|---|---|---|
| AVX2 内核 | 引擎每种 ISA 编一个库（`libxtier_engine_avx512.so` / `_avx2.so`），`libxtier_engine.so` 按 CPU 选择并 dlopen（`XTIER_ISA=auto\|avx512\|avx2`）。同一个库里放两份会让 `std::to_string` 之类的内联函数被合并成 AVX-512 版，Zen2 上 SIGILL（lkqmoe 0.5.3 踩过） | `tools/isa_check.sh`：分发库 + AVX2 库 + 信箱库 + 表存储库里 0 条 AVX-512 指令，AVX-512 库有 14534 条 |
| 张量并行 | 每个 TP rank 一个进程，持有每个专家的 I/2 切片（官方加载器按 rank 切），部分输出由官方 all-reduce 相加；`xtier/engine/placement.py`：`XTIER_TP_NODES=auto` 把有 CPU 的 NUMA 节点均分成 tp 组，rank 取自己 GPU 所在的组，作为 `XTIER_NODES` 交给引擎（引擎首次调用时固定拓扑，所以 TP 时放置推迟到第一层交接），调度线程和其余线程也只用本 rank 的 CPU | `tests/tp_check.py`：两个切片进程（各占 4 个节点）的输出之和 vs 单进程整层 |
| 热专家缓存策略线程 | 在调用者的 GPU 上运行（`cudaSetDevice`；lkqmoe 0.5.3 时新线程默认 GPU0，rank 1 报 invalid argument） | 已在 `csrc/mailbox` |
| 草稿层专家 | `XTIER_DRAFT_SLOTS=0`：NextN 草稿层的专家留在 CPU 引擎（全放 GPU 每卡要 1.9 GiB） | 在 3.23 单卡上测 `XTIER_DRAFT_SLOTS` |
| 大表放 NVMe | Qwen PLE（48 GiB）可放 NVMe（`XTIER_PLE_ROWS=nvme`，`docs/design-tables-nvme.md`） | 3.23 上逐字节 / 端到端验证 |

## 显存预算（每卡，GLM-5.3 TP2，256K，10-17 估算 + 实测峰值）

权重约 8 GiB、256K FP8 KV 约 2.35 GiB（MLA，两卡各存一份）、mamba 约 0.85 GiB、图和激活 2–2.5 GiB，剩约 1.6–2.1 GiB
给热专家（42 层、半个专家，每槽约 0.28 GiB）：从每层 4 槽起，看启动日志的 `avail mem` 再加。10-17 的 256K 峰值
15770 / 16303 MiB。

## 10-17 的实测（lkqmoe 0.5.3 + Lsglang，作为 XTier 的对照）

| 输入 | 首 token | prefill | decode |
|---|---:|---:|---:|
| 64K | 60.6 s | 1083 tok/s | 33.4 tok/s |
| 256K | 237.8 s | 1089 tok/s | 34.4 tok/s |

prefill 受 PCIe 4.0 限制：每个分块把全部专家传到 GPU（每卡约 80 GiB），速度与分块大小成正比（CHUNK 2048：531 tok/s；
4096 + 每块 4 专家：940 tok/s；5120 + 每块 2 专家：1083 tok/s）。冷启动约 7 分钟。

## 环境的坑（10-17）

- venv 里 `nvidia/cu13` 缺 `lib64 -> lib` 和 `lib/libcudart.so -> libcudart.so.13` 两个链接，JIT 内核编不过；ninja 要在 PATH 上。
- GPU 预填充块（`XTIER_GPU_PREFILL_CHUNK`）只支持 4096 / 8192 / 16384；16GB 卡用 4096。
- TP 下关掉草稿热词表（`--speculative-token-map`）：草稿头按全词表 id 索引目标 lm_head，每个 rank 只有自己那份词表，越界。
- 不要用 nvidia/GLM-5.3-Flash-NVFP4：其 NextN 层整层 BF16（专家约 14.5 GB），16GB 卡放不下；用 LibertAIDAI 的。
- GPU 出过 Xid 32 + uvm fatal，只能重启；驱动坏时远程 reboot 会卡住，要现场强制重启；正常时远程重启约 7.5 分钟回来。

## 机器到手后的上线清单

1. 告诉用户、等朋友停手再登录（朋友的机器）；大文件（模型、pip 包）在那台机器上用它自己的网络下，只有小文件经中转。
2. 环境：官方 SGLang main（与 3.23 同一提交）+ venv（同版本锁定），补上面两个链接；XTier `pip install -e`。
   引擎库在 3.23 上编好拷过去（AVX2 构建只用 `-march=x86-64-v3`），拷过去后跑 `tools/isa_check.sh`。
3. 单进程先验：`XTIER_ISA=avx2 python tests/tp_check.py --preset glm53 --tp 2 --rank-threads 44`（两 rank 各 4 节点），
   `tests/mailbox_check.py`、`tests/cache_check.py`（单卡）。
4. `configs/glm53-2x5070ti-zen2/run-glm-xtier-tp2.sh`：先 32K 上下文起服务，看两个 rank 的放置日志（每个 rank 的 NUMA
   节点应是 GPU 所在那颗 CPU 的 4 个节点）和 `avail mem`，再加到 256K、调槽数。
5. 对照 10-17 的表测 64K / 256K 首 token 与 decode，PPL 与照抄（`bench/`）。
