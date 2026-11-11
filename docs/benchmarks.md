# 测速细节

[README](../README.md)「测速」各表的测法、配置、精度和原始数据。单卡 RTX PRO 5000 72GB + 2× EPYC 9334（NPS4）
+ 384 GiB DDR5，XTier 跟官方 SGLang main 8cd3c494 时（2026-10 下旬）测的（MiniMax 用 SGLang 的 M3.1 分支），单位 tok/s。后面的版本没有重测这些表，只有 DeepSeek 加速包那两行在 2026-11-11 重测（prefill 分块改成 48K、CPU 分担 prefill）。

## 测法

**agent 速度**：生产仓库的 agent_bench（[bench/agent_bench.py](../bench/agent_bench.py)：lsglang_ds4.1 那份加两处健壮性修正，计时不变），
模拟编码代理（dsh / Claude Code 一类工具）：5 个会话从 64K 到约 500K，铺满 512K KV 池，多步调用工具，后续请求命中前缀缓存，
服务端默认采样，同一次启动连跑两轮。每张表比基线 和 XTier + 加速包。冷 prefill = 会话第一个请求里未命中缓存的 token / 首字；decode 只算
长回复（≥ 300 token，写代码、写总结），接受长度随内容变，比较两边先看每个验证步的耗时。表用 [bench/agent_table.py](../bench/agent_table.py)
生成，每会话明细见[报告](reports/2026-10-24-ds-woa-minimax.md)。

## DeepSeek V4.1 Flash 1017

- 加速包（2026-11-11 重测，启动台默认：48K 分块、每层 35 槽、弹性热专家槽、CPU 分担 prefill）对基线（2026-10 下旬测，22K 分块、
  50 槽）：长回复每步 30.4 对 50.9 ms（快 40%），冷 prefill 64K / 500K 5,738 / 6,670 对 3,318 / 3,627（快 73% / 84%）。
- CPU 分担 prefill（2026-11-11，DeepSeek 启动台默认开）：读长提示时路由少的专家块不上传，交给空闲的 CPU 同时算。同一天不开时的
  两轮：冷 prefill 64K / 500K 5,268–5,886 / 6,576–6,600，长回复每步 30.2–30.7 ms；开着是上表的 5,246–6,230 / 6,620–6,721、
  30.3–30.5 ms，128K 以上的会话快 1–3%。GLM、Qwen 上实测更慢，不开，见同一份[报告](reports/2026-11-11-prefill-chunk.md)。
- prefill 分块 48K（2026-11-11，原来 22K）：每读一块，每层都要把不在热专家槽里的专家传上显卡，约 2.5 秒，和块的大小无关；块加大后
  摊给更多 token。同一天用原默认（22K、50 槽）跑的一轮：冷 prefill 64K / 500K 4,107 / 4,893，长回复每步 30.3 ms；改成 48K 后
  5,268–5,886 / 6,576–6,600，每步 30.2–30.7 ms。大块多用的显存从热专家槽里让（50 → 35），decode 时弹性热专家槽补回到 68。
  2026-10-26 的 22K 数据是 4,420 / 4,806、每步 31.2 ms。各档分块、显存、两个请求时的表现和 GLM / Qwen 的结果见[报告](reports/2026-11-11-prefill-chunk.md)。弹性热专家槽（decode 时把只有 prefill 才用的显存借给热专家，至多 68 槽）把每步
  从 32.6 降到 31.2 ms，见[报告](reports/2026-10-26-elastic-cache.md)。
- 基线 的 DeepSeek 工具调用解析不逐 token 流式返回参数，工具调用请求量不出 decode（两轮 177 个请求里能计时 13 个），所以只比
  长回复。
- 加速包精度：无损——块 FP8 GEMM 的分块配置和内核、Engram 提前查表（取到的行不变）、草稿词表头 FP8（只影响草稿提案，输出仍由主模型
  验证）、wo_a 存检查点的 FP8（还原逐位一致，kernel 与 cuBLAS 只差求和顺序）、索引器走分页路径（logits 与非分页路径逐位相同，top-k
  只在同分时可能选到不同位置），PPL 3.319 → 3.317（分页索引器之前测），照抄 0 错。
- 和生产管线（lsglang_ds4.1）的 agent 对比见[报告](reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-ds41-cpushare-r{1,2}-20261111.json`（不开 CPU 分担：`agent-bench-ds41-chunk48k-r{1,2}-20261111.json`；22K 分块：`agent-bench-ds41-chunk22k-r1-20261111.json`、`agent-bench-ds41-elastic-r{1,2}-20261026.json`；不开弹性：`agent-bench-ds41-xtier-paged-r{1,2}-20261024.json`），
  基线 `bench/results/context-ds41-xtier-base-20261023.json`。

## Qwen3.8 Flash Next

- 加速包（2026-10-26 重测，启动台默认：规划器的自动布局，29 层常驻 + 18 × 64–88 槽）对基线：长回复每步 13.6 对 15.2 ms
  （快 11%），冷 prefill 64K / 500K 5,663 / 5,160 对 5,319 / 4,078（快 6% / 27%，QSA 索引器和 PLE 层分段主要省在长上下文）。基线
  用 FP8 KV、16K 分块、生产的 25 层常驻（官方 QSA 读不了 NVFP4 KV）。工具调用逐 token 流式，所有请求都能计时。
- 弹性热专家槽默认关：Qwen 的 decode 受 GPU 限制，多放热专家会使 decode 变慢约 10%（多放两层常驻、池化层的槽全部动态补回：
  prefill 只快约 1%），见[报告](reports/2026-10-26-elastic-cache.md)。
- 加速包精度：轻微——KV 存 NVFP4（与生产相同，每 16 个值一个 E4M3 缩放；读出与官方反量化逐位一致），照抄 0 错、PPL 3.481；hc_mix
  换成重叠的持久内核、PLE 层与 QSA 索引器分段与官方逐位一致。
- 和生产管线（lsglang_q3.8fn）的 agent 对比见[报告](reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-qwen38-xtier-r{1,2}-20261026.json`，基线 `bench/results/context-qwen38-xtier-base-20261023.json`。
- 自动布局（`LAYOUT=auto`，2026-10-25 起默认）对 26 层常驻：同显存，五个会话的冷 prefill +2.4–2.9%，decode 每步持平，见
  [reports/2026-10-24-planner.md](reports/2026-10-24-planner.md)。

## GLM-5.3 Flash

- 加速包（2026-10-26 重测，启动台默认：常备 58 槽 + 弹性热专家槽）对基线：长回复每步 26.9 对 34.4 ms（快 22%），冷 prefill
  64K / 500K 2,682 / 2,638 对 2,663 / 2,518（+1% / +5%）。常备槽 60 → 58：60 槽时 16K 分块的冷 prefill 贴着显存上限（同一晚两轮
  agent_bench 里分配器释放缓存重试 18 次，一次 25 万 token 的冷提示服务退出），58 槽只在预热时重试 1 次、每步持平（60 槽 27.6 ms），
  decode 时弹性槽把显存借回来（每层多约 12–14 槽），见[报告](reports/2026-10-26-elastic-cache.md)。基线 用 BF16 权重、
  55 槽（60 槽时 prefill 显存不够）、不开拒绝采样。
- 加速包精度：轻微——BF16 注意力 / 稠密层存成 FP8（每行每 128 列一个 scale），PPL 3.987 → 3.998（+0.3%）；拒绝采样与截断验证无损
  （输出分布与不用投机解码相同），照抄 0 错。
- 和生产管线（lsglang_glm5.3）的 agent 对比见[报告](reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-glm53-base58-r{1,2}-20261026.json`（60 槽：`agent-bench-glm53-base60-r{1,2}-20261026.json`；
  不开弹性：`agent-bench-glm53-xtier-r{1,2}-20261024.json`），基线 `bench/results/context-glm53-xtier-base55-20261023.json`。

## MiniMax-M3.1-Flash

**暂停开发**（2026-10-26）：在本机太慢（agent 长回复 43 tok/s），智力也不强，代码和启动台卡片保留。SGLang 的 M3.1 分支（PR #45925，
还没合入 main）+ XTier，固定 512K 上下文，KV 存 NVFP4，热专家每层 8 槽；只跑 64k 会话（MiniMax 常把第一步的 4096 个输出 token
全用来思考），没有生产管线可比，对比的是官方代码加必要修正。

- 加速包（草稿长度 5、DeepSeek 的换入节奏、草稿词表头 FP8、6 行验证走 b12x、DSpark 按置信度截断验证）对官方代码 + 必要修正
  （SM120 通用栈、CUDA 上的 DSpark verify、NVFP4 KV，草稿长度 7）：每步 55.4 对 84.1 ms（快 34%），tok/s 快 12%，冷 prefill 相同。
- 加速包精度：轻微——NVFP4 KV 照抄单并发偶有同一个临界 token 翻转（双并发 0 错），480K 找数 5/5；截断验证无损。PPL 不适合衡量
  MiniMax 的精度，见[报告](reports/2026-10-24-minimax-512k.md)。
- 2026-10-26 用全部五个会话重测：长回复 43.4 tok/s、每步 69.2 ms，500K 冷 prefill 1,531；弹性热专家槽每步持平，默认关。
- 原始数据：`bench/results/agent-bench-minimax31-{xtier-512k,base-512k}-r{1,2}-20261024.json`、`agent-bench-minimax31-xtier-r{1,2}-20261026.json`。

## 配置

- **基线**（启动台「关加速包」，以前的报告里叫“官方 main”）：官方 SGLang main 加 XTier 后端和跑起来必需的修正，不开任何提速项，不是原版 SGLang。必需的修正：Qwen 的 PLE 表锁页分配；GLM 的量化排除、DSA
  别名、NextN 草稿量化；DeepSeek 的 SM120 候选索引器、Engram 行放 NVMe、尾部重放不走 CUDA 图；
  MiniMax 的 SM120 通用栈、CUDA 上的 DSpark verify 和 NVFP4 KV（`xtier/packs/minimax_m3`；512K 时 FP8 KV 放不下）。
- **热专家布局与启动参数与生产相同**：Qwen NVFP4 KV、其余层每层 155 槽；GLM 每层 60 槽（启动台默认 58 槽 + 弹性，见 GLM 一节）、16K 分块、BF16 SSM 状态、拒绝采样；
  DeepSeek 每层 50 槽（启动台默认从 2026-11-11 起是 48K 分块、35 槽加弹性，见 DeepSeek 一节）；MiniMax（没有生产管线）每层 8 槽、512K NVFP4 KV 池、16K 分块。
  不同的地方：Qwen 加速包那行 26 层常驻（比生产多第 27 层，用 NVFP4 KV 省下的显存，prefill +0.5–1.4%；10-25 起启动台默认由规划器
  在这份显存内重排成 29 层常驻 + 18 × 64–88 槽）、
  24K 分块（生产 32K：这里 32K 在 64K 冷 prefill 慢 3%），见[报告](reports/2026-10-24-qwen-kv-ds-agent.md)；Qwen 基线
  用 FP8 KV（基线 的 QSA 读不了 NVFP4）、16K 分块、生产的 25 层常驻；GLM 的基线 用 BF16 权重，60 槽时 prefill 显存不够，
  减到 55 槽，也不开拒绝采样（基线 不允许热词表配拒绝采样）。
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
