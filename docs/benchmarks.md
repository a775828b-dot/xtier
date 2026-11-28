# 测速细节

[README](../README.md)「测速」各表的测法、配置、精度和原始数据。单卡 RTX PRO 5000 72GB + 2× EPYC 9334（NPS4）
+ 384 GiB DDR5，基线与 Qwen3.8、GLM-5.3 各行在 XTier 跟官方 SGLang main 8cd3c494 时（2026-10 下旬）测得（MiniMax 用 SGLang 的 M3.1 分支），单位 tok/s；DeepSeek 加速包各行在 2026-11-24 按最新默认配置重测（SGLang main 4ce41378f9）。

## 测法

**agent 速度**：生产仓库的 agent_bench（[bench/agent_bench.py](../bench/agent_bench.py)：lsglang_ds4.1 那份加两处健壮性修正，计时不变），
模拟编码代理（dsh / Claude Code 一类工具）：5 个会话从 64K 到约 500K，铺满 512K KV 池，多步调用工具，后续请求命中前缀缓存，
服务端默认采样，同一次启动连跑两轮。每张表比基线 和 XTier + 加速包。冷 prefill = 会话第一个请求里未命中缓存的 token / 首字；decode 只算
长回复（≥ 300 token，写代码、写总结），接受长度随内容变，比较两边先看每个验证步的耗时。表用 [bench/agent_table.py](../bench/agent_table.py)
生成，每会话明细见[报告](reports/2026-10-24-ds-woa-minimax.md)。

## Inkling-2-Small

- `v0.1.14` 通用 Recall 重测（2026-11-28，300W，SGLang `4ce41378f9`，默认方案 A+ + `XTIER_RECALL=1`）：`decode_speed` `bs=2` 升至 **`81.1 tok/s`**（次轮稳态 **`89.8 tok/s`**），256K 冷 prefill **`2,514 tok/s`**，500K 缓存 decode 升至 **`56.0 tok/s`**；`agent_bench` 热态第 2 轮长回复升至 **`73.6 tok/s`**（`54.0 ms/步`，384K 接受长度 `4.93` 达 `74.8 tok/s`，Recall 落地 `3,793` token），峰值 `70.21 GiB`，0 fatal、0 OOM 告警。

- **硬件与目标配置**：RTX PRO 5000 72 GB + 2× EPYC 9334 NPS4 + 377 GiB DDR5 + 4× Samsung 9100 PRO PCIe 5.0 NVMe；512K 上下文 / 2 并发（`MAX_RUNNING_REQUESTS=2`），SGLang main `4ce41378f9` 零源码改动。
- **加速包（方案 A+，启动台默认：`cold=0.02`, `376/384` 专家在内存, `base=1`, `reserve=0.8G`, 45 弹性热槽, `BATCH=65664`）对基线（`W4A16/W8A16, BF16 KV, BF16 稠密层, 16K, 27 槽`）**：
  - 启动耗时：`126 s`（4 盘快载）对 `~390 s`（快 `3.1×`）；启动后 / 500K 后可用内存 `15 / 12 GiB`（≥8 GiB 红线），稳态显存 `70.65 / 73.41 GiB`（0 次分配器 OOM 重试），Decode NVMe 读盘仅 `0.2–0.7` 专家/步（较 `cold=0.08` 降 6–10×），Prefill NVMe 等待 `0.0 s`。
  - 吞吐对比：短文 `bs=1` `74.7`（首轮 `81.7`）对 `47.9 tok/s`（`+56%`），`bs=2` 合计 `78.4` 对 `~43 tok/s`（`+82%`）；64K 冷 prefill `3,985.9`（`16.46 s` 单块，稳态块 ~4,250）对 `1,590.5 tok/s`（`2.51×`），256K `2,496.5`（`105.0 s`）对 `1,217.3 tok/s`（`2.05×`），500K `1,626.9 tok/s`（`307.31 s`，基线 OOM，较 v0.1.10 省 `6.6 s`）；64K 峰值 decode `68.1 tok/s`（`45.3 ms/步`），256K `57.3 tok/s`（`53.2 ms/步`），500K `46.0 tok/s`（峰 `55.1`，`54.7 ms/步`）；无损 Prefill 优化后 64K 峰值显存需求由 `43.9 → 37.0 GiB`（`-6.9 GiB`），13 段 PPL `2.7172`（`NLL 0.9996`，ΔNLL `+0.00005`），`T=0.01` 照抄 `c=1/2` 均为 `0/4`。
- **方案 A 稳妥档（`cold=0.08`, 44 槽, `30–35 GiB` 可用内存）与两轮 `agent_bench`（64K→513K）**：两轮共 91 请求（33 可计时，48,102 输出 token，91/91 命中缓存），长回复 **`66.2 tok/s`**（全部 **`66.3 tok/s`**，接受长度 **`4.02`**，`60.6 ms/步`）；会话冷 prefill 64K `2,628–2,679`、128K `2,755–2,771`、256K `2,400–2,404`、384K `1,908–1,910`、500K `1,603 tok/s`；非 Recall 新写（`T=0.6`）中文 `56.7` / 英文 `55.6 tok/s`。
- **方案 B2（`cold=off` 无冷专家：仅 1 个 MXFP8 层 `L47` 占 `11.75 GiB` 全驻 GPU 零内存，35 弹性热槽, `BATCH=49152`, `CPU_SHARE=1`）**：可用内存 `20 / 13 GiB`，0 次 NVMe 读盘，64K / 256K / 500K 冷 prefill `2,846.3 / 2,215.2 / 1,511.7 tok/s`（较旧 3 层全驻 `16K` 提速 `+50.4% / +44.4%`），64K / 256K / 500K 缓存 decode `56.8 / 52.0 / 40.1 tok/s`（64K 峰 `62.5`、`40.09 ms/步`），短文首轮 `82.9` / `bs=2 72.6 tok/s`，启动 `420 s`。
- **精度验收**：A1（`L0..39` NVFP4 `4over6 MAE` + `L40..47` MXFP8 cuDNN prefill，decode 保持原生 W4A16/W8A16）+ A2（Target 48 层 NVFP4 KV `6.75 GiB` + Draft 5 层 BF16 KV）+ A3（116 层全量 FP8/W8A8）+ DSpark 截断 `0.20`：13 段 PPL/KL（53,235 位置）`all.nll = 0.99955` 对基线 `0.99531`（`ΔNLL +0.00423`、`KL 0.00938`）；`copy_test`（`T=0` / `T=0.01`, `c=1/2`）全部 `0/8`；256K 与 500K 5 深度捞针全 `5/5`；思考（`410`）、工具调用（`get_weather`）与视觉多模态均通过。

## DeepSeek V4.1 Flash 1017

- `v0.1.14` 重测（2026-11-28，300W，SGLang `4ce41378f9`，48K 分块 / 37 槽 + 弹性槽、mHC 原地写回 `XTIER_DS_MHC_INPLACE=1`、`XTIER_CACHE_ELASTIC_TORCH_GUARD_GIB=1.6`、通用 Recall + GPU `propose_after`）：两轮 `agent_bench` 共 169 请求全部命中缓存上报，合计 decode 升至 **`174.0 tok/s`**（接受长度 `4.81`，每步 **`27.6 ms`**，对比 11-24 `163.9 tok/s` / `28.9 ms` 快 `+6.2%`）；第 2 轮稳态长回复 **`158.0 tok/s`**（`29.4 ms/步`，两轮长回复均值 `155.0 tok/s` / `29.7 ms/步`），第 2 轮冷 prefill 64K **`8,024`**、128K `8,726`、256K **`9,082`**、384K **`9,096`**、500K **`9,103 tok/s`**（第 1 轮首请求含冷编译 64K `3,126`，256K–500K 为 `8,889–8,973`）；`speed4` 2/4 并发每步 `39.3 / 51.3 ms`（`187.2 / 255.8 tok/s`）；全程 500K 峰值显存稳守 **`69.52 GiB`**（`1.60 GiB` 真实可用余量），0 fatal、0 次 PyTorch OOM 告警、0 次水位墙回退；PPL `3.6304`。

- 无损 24-token 权重复用 decode、写回融合、43 层 FP8 `wo_a`、Eager MXFP8 快分发与非末块跳尾（2026-11-24 重测，SGLang main `4ce41378f9`，默认 48K 分块、35 槽 + 弹性槽、CPU 分担 prefill、变长验证、上下文草稿检索 Recall 开、1M KV 池）：两轮 `agent_bench` 冷 prefill 64K 升至 `5,969 / 7,967`（均值 `6,968`，较基线快 110%）、128K `8,449 / 8,714`、256K `8,844 / 8,931`、384K `8,966 / 8,927`、500K 两轮双破九千达 `9,038 / 9,023 tok/s`（均值 `9,030`，较基线快 149%，稳态 48K 单块达 `9,806 tok/s`），长回复两轮 `156.6 / 154.7` 合计 `155.4 tok/s`（每步 `29.9 ms`，较基线快 41%；同会话 64K / 128K 长回复为 `157.8–162.2 tok/s`、每步 `28.8 ms`，第 2 轮 384K 推进至 488K 上下文写长文件使两轮长回复加权上下文为 `234K / 282K`），两轮全部 127 个请求 `163.9 tok/s`（每步 `28.9 ms`）；`speed4` 4 个 16K 提示同时从零新写 3,000 token 合计升至 `268.0 tok/s`（接受长度 `3.40`、每步 `51.0 ms`），2 个同时新写升至 `195.8 tok/s`（每步 `37.4 ms`）；160K（`160,041` tok）全量冷 prefill `20.43 → 17.86 s`（`7,833 → 8,961 tok/s`，+14.4%，首轮冷战术 `18.69 s`），首个 16.5K 预填 `4.22 → 3.81 s`（−9.7%），`286–594` token 增量预填快 `2.0–5.1%`（`594` tok `1.219 → 1.157–1.175 s`）；单路满宽验证 `recall` `28.44 → 28.20–28.27 ms`/步（`210.5 → 212.0–212.5 tok/s`），同温背靠背 `bs=1` 推理步时 `26.82 → 26.60 ms`/步，61 槽稳态 `M=16 / 18 / 24` 验证步分别从 `57.8 / 57.5 / 80.0 ms` 降至 `48.2 / 45.0 / 71.4 ms`（−16.6% / −21.8% / −10.8%）；`copy_test` 并发 1/2/4 均 `0/8` 正文差，160K 三针跨 3 个跳尾中间块 100% 召回，13 段 PPL 平均 NLL `1.2001`（基线 `1.1996`，偏差 `+0.00056`）。
- 全双工零同步 FP4 预填、SM120 mHC 精确核与工具回传提速（2026-11-19 重测，保持启动台默认 48K 分块、35 槽 + 弹性槽、CPU 分担 prefill、变长验证、上下文草稿检索 Recall 开、1M KV 池）：长回复 159.0 对基线 65.5 tok/s（每步 29.5 对 50.9 ms，快 42%），冷 prefill 64K / 500K 两轮均值 6,716 / 8,940 对 3,318 / 3,627（快 102% / 146%；第 2 轮 64K 7,713，384K–500K 会话 8,894–9,085，458K 单次冷提示 9,180，稳态 48K 单块峰值 9,999 tok/s）；两轮全部 192 个请求合计 decode 177.1 tok/s、接受长度 4.86、每步 27.4 ms，743 token 工具回传首字 2.07 → 1.12 s；13 段 PPL 的 NLL 1.2003（v0.1.7 为 1.1990）。详见[报告](reports/2026-11-19-lossless-9k-prefill.md)。
- 无损 prefill 融合、Engram 跨块预热与 DSpark 上下文草稿检索（2026-11-18 重测，启动台默认：48K 分块、35 槽 + 弹性槽、CPU 分担 prefill、变长验证、上下文草稿检索 Recall 开、1M KV 池）：长回复每步 28.8 对基线 50.9 ms（快 43%），冷 prefill 64K / 500K 两轮均值 6,363 / 7,634 对 3,318 / 3,627（快 92% / 110%）；两轮全部 159 个请求合计 decode 180.4 tok/s、接受长度 4.79、每步 26.5 ms；4 个请求同时从零新写 3,000 token 合计 258.3 tok/s（基于原文件改写 / 编辑时合计 332–382 tok/s），2 个同时新写 193–195 tok/s，单请求每步 27.1 ms。详见[报告](reports/2026-11-18-lossless-prefill-recall.md)。
- 加速包（2026-11-12 重测，v0.1.5，启动台默认：48K 分块、每层 35 槽、弹性热专家槽、CPU 分担 prefill、变长验证、1M KV 池）对基线（2026-10 下旬测，
  22K 分块、50 槽）：长回复每步 29.9 对 50.9 ms（快 41%），冷 prefill 64K / 500K 5,822 / 6,674 对 3,318 / 3,627（快 75% / 84%）。
- 草稿和两个请求的 FP8 投影走 b12x（2026-11-11）：FP8 投影的小批量快内核原来只对 1、4、6 行开，DSpark 草稿模型的 5 行、两个请求同时的
  10 和 12 行走 CUTLASS（125–159 µs 对 39–41 µs）。默认行数改成 1、4、5、6、10、12 以后草稿模型一遍 3.03 → 2.20 ms，固定验证两个请求
  同时每步 52.3 → 44.0 ms。同一天改之前的两轮（变长验证）：长回复每步 30.5 / 30.3 ms；改之后变长验证 29.9 / 29.6 ms，固定验证是
  上表的 29.6 / 29.5 ms。变长验证的成本表同时重测。一步的构成和多个请求的估算见[报告](reports/2026-11-11-decode-step.md)。
- 同时 4 个请求（2026-11-12，v0.1.5）：`--max-running-requests` 2 → 4，KV 池 512K → 1M token（单个请求的上下文仍是 512K），一次专家
  调用最多 16 个 token，启动台默认变长验证（补满档位）。同样的提示（各 16K token，写 3,000 token，温度 0）：1 个请求每步 27.7–28.2 ms、121–139 tok/s，
  2 个同时每步 39.0 ms、合计 191 tok/s，4 个同时每步 49.7 ms、合计 254 tok/s（README 的第三行）。固定验证（启动台可选）：
  28.3 ms，42.4 ms、192 tok/s，75.9 ms、212 tok/s。改之前（一次 8 个 token、固定验证、最多 2 个请求）：
  27.7 ms，45 ms、约 175 tok/s。[报告](reports/2026-11-12-ds-four-requests.md)
- CPU 分担 prefill（2026-11-11，DeepSeek 启动台默认开）：读长提示时路由少的专家块不上传，交给空闲的 CPU 同时算。同一天不开时的
  两轮：冷 prefill 64K / 500K 5,268–5,886 / 6,576–6,600，长回复每步 30.2–30.7 ms；开着是 5,246–6,230 / 6,620–6,721、
  30.3–30.5 ms，128K 以上的会话快 1–3%。GLM、Qwen 上实测更慢，不开，见同一份[报告](reports/2026-11-11-prefill-chunk.md)。
- prefill 分块 48K（2026-11-11，原来 22K）：每读一块，每层都要把不在热专家槽里的专家传上显卡，约 2.5 秒，和块的大小无关；块加大后
  摊给更多 token。同一天用原默认（22K、50 槽）跑的一轮：冷 prefill 64K / 500K 4,107 / 4,893，长回复每步 30.3 ms；改成 48K 后
  5,268–5,886 / 6,576–6,600，每步 30.2–30.7 ms。大块多用的显存从热专家槽里让（50 → 35），decode 时弹性热专家槽补回到 68。
  2026-10-26 的 22K 数据是 4,420 / 4,806、每步 31.2 ms。各档分块、显存、两个请求时的表现和 GLM / Qwen 的结果见[报告](reports/2026-11-11-prefill-chunk.md)。弹性热专家槽（decode 时把只有 prefill 才用的显存借给热专家，至多 68 槽）把每步
  从 32.6 降到 31.2 ms，见[报告](reports/2026-10-26-elastic-cache.md)。
- 基线 的 DeepSeek 工具调用解析不逐 token 流式返回参数，工具调用请求量不出 decode（两轮 177 个请求里能计时 13 个），所以只比
  长回复。
- 加速包精度：无损——块 FP8 GEMM 的分块配置和内核、Engram 提前查表（取到的行不变）、草稿词表头 FP8（只影响草稿提案，输出仍由主模型
  验证）、wo_a 存检查点的 FP8（还原逐位一致，kernel 与 cuBLAS 只差求和顺序）、FP8 投影在 b12x 和 CUTLASS 之间换内核（量化后的
  权重和激活相同，实测输出逐位相同）、索引器走分页路径（logits 与非分页路径逐位相同，top-k
  只在同分时可能选到不同位置），PPL 3.319 → 3.317（分页索引器之前测），照抄 0 错。
- 和生产管线（lsglang_ds4.1）的 agent 对比见[报告](reports/2026-10-24-ds-woa-minimax.md)。
- 原始数据：加速包 `bench/results/agent-bench-ds41-b12xrows-r{1,2}-20261111.json`（变长验证：`agent-bench-ds41-b12xrows-compact-r{1,2}-20261111.json`；改行数之前、变长验证：`agent-bench-ds41-cpushare-r{1,2}-20261111.json`；不开 CPU 分担：`agent-bench-ds41-chunk48k-r{1,2}-20261111.json`；22K 分块：`agent-bench-ds41-chunk22k-r1-20261111.json`、`agent-bench-ds41-elastic-r{1,2}-20261026.json`；不开弹性：`agent-bench-ds41-xtier-paged-r{1,2}-20261024.json`），
  基线 `bench/results/context-ds41-xtier-base-20261023.json`。

## Qwen3.8 Flash Next

- `v0.1.14` 通用 Recall 重测（2026-11-28，300W，SGLang `4ce41378f9`，`XTIER_RECALL=1`）：同版本 A/B 提速 `+5.1–7.4%`；两轮 `agent_bench` 81 请求全部命中缓存，Recall 落地 `4,078` token，首轮长回复 **`194.2 tok/s`**（`13.1 ms/步`，384K 改写步达 `213–242 tok/s`），次轮长回复 `170.3 tok/s`（73% 输出在 384K–500K 上下文），冷 prefill 64K `5,113 / 5,157`、500K `4,858 / 4,830 tok/s`，峰值 `69.71 GiB`，0 fatal。

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

- `v0.1.14` 重测（2026-11-28，300W，SGLang `4ce41378f9`，58 槽 + 弹性槽 + 通用 Recall `XTIER_RECALL=1` + 无损拒绝采样）：两轮 `agent_bench` 共 116 请求全部命中缓存上报，长回复两轮 `96.9 / 102.9` 合计破百达 **`101.0 tok/s`**（接受长度 `2.78`，每步 `27.6 ms`），全部 116 请求合计 **`105.5 tok/s`**（接受长度 `2.94`，工具调用达 `149–241 tok/s`）；冷 prefill 64K `2,416 / 2,533`、500K `2,505 / 2,507 tok/s`；峰值 `69.77 GiB`，0 fatal、0 OOM 告警。

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
