# 常见问题

按症状分类，日志里的原文可以直接搜。[回到文档目录](README.md)

## 安装和启动

### 日志警告 `is tested with SGLang main`

XTier 每次只认一个测试过的 SGLang 提交（[xtier/__init__.py](../xtier/__init__.py) 的 `SGLANG_COMMIT`）。版本不同时服务还能起来，但可能有钩子挂不上。运行 `python tools/check_sglang.py` 看具体是哪些；最稳的是 checkout 这个提交，步骤见[快速上手](../README.md#快速上手)。

### 启动直接退出：`hooks not applied` 或 `has no run_scheduler_process`

这个 SGLang 和 XTier 对不上。继续跑的话加速项会悄悄消失，甚至退回官方路径，所以直接退出。处理方法同上。不想用 XTier 时去掉 `XTIER=1`。

### 日志有 `settings XTier does not know are ignored`

环境变量拼错了，或者这个开关已经删掉。开关的名字和默认值见 [xtier/config.py](../xtier/config.py)。

### 启动脚本说路径不存在

`configs/` 大多数脚本开头的 `D=`（SGLang 环境）、`M=`（模型目录）、`P=`（档案和草稿）是测试机的路径，要改成自己的。[qwen38-rtx4090-3090](../configs/qwen38-rtx4090-3090/README.md) 和 [glm53-2x5070ti-zen2](../configs/glm53-2x5070ti-zen2/README.md) 可以直接用环境变量指定。启动台调用的就是这些脚本，也要先改。

### 第一次启动很久，日志停在 `building FlashInfer modules`

FlashInfer 的内核在加载权重之前编译，好过在第一个请求里现编。从零编译要几分钟（DeepSeek 实测 365 秒、4,300 CPU 秒），每个编译任务约 4–6 GiB 内存，并发数按可用内存自动限制。编好后有缓存，下次不用再编。

`XTIER_PREBUILD=0` 可以关，但编译会推迟到第一个长 prefill 请求里，那时内存可能已经不够。

### 每次启动都重新编译

一般是 `ninja` 变了：FlashInfer 靠 ninja 判断要不要重编，虚拟环境里的 ninja 和系统的版本不同，切换一次就全部重编。启动服务时 `PATH` 要保持一致，不要有时激活虚拟环境、有时不激活。

## 内存和显存

### 日志有 `mem_guard: MemAvailable ... killing process group`

主机可用内存低于下限（启动台默认 20 GiB），内存守护停掉了服务，避免把机器推进 swap 或 OOM killer。这一行的 `largest` 列出了占内存最多的进程，日志在模型日志旁边的 `.guard.log`。

先看是不是别的进程占的；如果就是模型放不下，查 README 的[所需内存表](../README.md#冷专家卸载到硬盘)，开冷专家卸载。启动台的 `--mem-guard-gib` 可以改下限。

### 显存不够

按省显存多少，依次试启动台的这些选项：

- **热专家槽 / 层**（`XTIER_CACHE_SLOTS`）：每少一个槽，省“层数 × 每个专家大小”的显存。
- **Prefill 分块**：分块越小，prefill 的缓冲越小，长 prefill 会变慢。
- **上下文 / KV 池**：上下文越短，KV 占得越少。

启动前也看看 GPU 上有没有别的进程。启动台发现别的进程占着超过 2 GiB 时不会启动。

### 模型比内存还大

开冷专家：内存只放一部分专家，其余放 NVMe，用到时读回来（`XTIER_COLD=1`）。卸载比例和速度的关系见[实测表](../README.md#deepseek-v41-flash-1017卸载比例与速度)，用法和盘的要求见[设计文档](design-cold-experts.md)。

注意力、嵌入这些非专家部分必须放显存，它们比显存还大时，冷专家也救不了，要换更大的卡或用多卡。

## 冷专家和只存一份

### 冷专家池用什么盘

看大块读的带宽，不看 4K IOPS。每次缺失要读一个完整专家，并且要等最慢的那块盘，所以尽量用几块同型号的盘。4 块 PCIe 5.0 盘实测每次缺失多花约 0.5 ms。

**一定要做几分钟的持续读测试**：无 DRAM 的 HP FX700 满速读约 100 秒后塌到约 60 MB/s，几秒钟的短测看不出来。

### 启动台的冷专家选项是灰的，提示“本机没有冷专家盘目录”

卡片里的 `COLD_PATHS`（[configs/ds41-pro5000-nps4/console.py](../configs/ds41-pro5000-nps4/console.py)）是测试机的目录。改成自己每块盘上的目录并建好，再重启启动台。

### 只存一份之后，还能换卸载比例吗

能。冷存储存的是全部专家，改 `XTIER_COLD_FRACTION` 不用重写。

### 启动报 `the checkpoint no longer holds its experts`

这个模型目录打过洞（目录里有 `XTIER-COLD-ONLY`），专家只在冷存储里，而这次启动的配置没法从冷存储加载。报错是有意的，否则会读到全零的权重。检查三件事：

- `XTIER_COLD=1` 和 `XTIER_FAST_LOAD=1` 都开着；
- `XTIER_COLD_PATHS` 和打洞时一样，盘都挂上了；
- 这一层是部分缓存的层（热专家槽大于 0，且小于专家数）。

想恢复完整的模型文件：`python tools/cold_punch.py 模型目录 --restore`。

### 打洞之后有什么风险

冷存储成了唯一一份，任何一块盘坏了都要重新下载模型。没有 XTier 的官方 SGLang 不能加载打洞后的目录。TP 下不能打洞。

## 速度和精度

### 比 README 的测速表慢

- **先比硬件**：表是 RTX PRO 5000 72GB + 2× EPYC 9334（NPS4）+ 384 GiB DDR5 上测的。DeepSeek 和 GLM 的 decode 主要受 CPU 内存带宽限制，Qwen 主要受 GPU 限制。
- **再比测法**：表里的 decode 只算长回复（≥ 300 token），用 [bench/agent_bench.py](../bench/agent_bench.py) 可以同样测一遍。token/s 还受投机接受率影响，比较时看每步耗时更稳。
- **看线程数**：`XTIER_THREADS` 要是 NUMA 节点数的倍数。
- **启动后第一个长请求会慢几秒**（64K 实测多 2.6–7 s），第二个开始才是正常速度。

### 温度 0，两次输出不完全一样

不是 bug。top-k 内核（SGLang 的 `fast_topk`）输出的集合相同，但顺序取决于原子竞争，浮点累加的舍入跟着变，逐层放大。大多数时候没有影响，在两个 token 分数很接近的地方偶尔会翻一个（比如照抄时漏一个词）。

判断精度看 [bench/ppl_eval.py](../bench/ppl_eval.py) 的 PPL 均值，不要看单次输出。

### 开了冷专家，输出和不开时不一样

原因同上。冷专家的结果差在 FP32 末位，照抄 0 错、NLL 在两次启动之间的波动内，见[报告](reports/2026-11-01-cold-pcie5.md)。

### MiniMax 还能用吗

代码还在，但 2026-10-26 起暂停开发，新版本没有再测过，也需要 SGLang 的 M3.1 分支（PR #45925）。
