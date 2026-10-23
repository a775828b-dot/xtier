# 2026-10-23 夜：通用投机解码包（GLM 拒绝采样）、Qwen 的 NVFP4 KV、DeepSeek 剩余差距

## GLM：通用投机解码包 `packs/speculative/`

生产 GLM 管线的投机解码有三项 XTier 没有：草稿热词表配无损拒绝采样、草稿温度 ×0.8、按草稿置信度截断验证（被截掉的 token
不算 CPU 专家）。三项都与模型无关，放进新的通用包 `xtier/packs/speculative/`（EAGLE 系列都能用）；只有"截掉的 token 跳过
专家"属于混合 MoE，放在核心。

- **热词表 + 拒绝采样**：官方 main 不允许 `--speculative-use-rejection-sampling` 和 `--speculative-token-map` 同时用（源码里留着
  "FIXME：把草稿分布散射回目标词表"）。包里：建草稿内存池时用 `get_spec().override` 临时关掉开关，绕过这项检查（它是那个函数的
  最后一句，也是里面唯一读开关的地方）；草稿 CUDA 图的 q 缓冲按热词表宽度建（官方的 draft-extend 图本来就这样）；空闲批的 q
  也用热词表宽度；构造验证输入前把 q 散射回完整词表（表外为 0，草稿本来就提不出这些 token，接受判据 coin·q(x) < p(x) 和残差
  (p − q)⁺ 都不变）。
- **草稿温度 = 请求温度 × 0.8**（`XTIER_SPEC_DRAFT_TEMP_SCALE`）：只改提案分布，验证用的是采样时的同一个 q，任意系数都无损；
  贪心请求（top_k = 1）照官方代码直接取 argmax。
- **按置信度截断**（`XTIER_SPEC_TRUNCATE_Q`，默认 0.2，`packs/speculative/truncate.py`）：草稿完成后，每个采样请求只保留开头
  q(x) ≥ 0.2 的草稿（至少 1 个，贪心请求全留）。没有改官方的拒绝采样内核：把第一个被丢草稿那一行的 q 清零——官方内核对
  q(x) = 0 的提案一律拒绝，补采时 (p − 0)⁺ = p，结果与"链本来就这么长"完全等价。被截掉的验证 token 通过核心新加的逐 token
  标志（`backend/method.py` 的 `active_flags()`）跳过路由专家：专家池的打包内核本来就支持（路由写成 −1，CPU 只算其余 token、
  被跳过的行返回 0），这次接到解码路径上，信箱直通路径的打包内核也补上。
- 单测 `tests/spec_truncate_check.py`：同样的随机数下，清零一行 q 的链与真正截短的链，官方采样内核的接受数、接受的 token、
  最后补采的 token 全部一致（经典验证、块验证各 1920/1920：3 种分布锐度 × 640 例）；截断内核的保留数、标志、清零的行与参考
  一致；信箱打包按标志写 −1。

测法用生产的 `bench/spec_t1_bench.py`（4 类中文任务；温度 1.0 / top_p 0.95 各 3 次、温度 0 各 1 次，每次 512 token；长文档那项
两边用同一份语料）：

| GLM-5.3 加速包 | 温度 1.0 decode | 接受长度 | 温度 0 decode | 照抄（单 / 双并发） |
|---|---:|---:|---:|---|
| 不开拒绝采样（之前） | 93.0 tok/s | 2.55 | 98.0 tok/s | 0/8、0/8 |
| + 热词表拒绝采样、草稿温度 ×0.8 | 94.5 | 2.69 | 93.6 | 0/8、0/8 |
| + 置信度截断（现在） | 95.6 | 2.55 | 98.1 | 0/8、0/8 |
| 生产（10-22，同一脚本） | 84.8 | 2.63 | 82.8 | — |

原始数据 `bench/results/spec-t1-glm53-xtier-{nors,rs,rs-trunc}-20261023.json`；生产的是 lsglang_glm5.3 的
`results/temperature1-20261022-24ch-nps4.json`。

- 单次 decode 波动约 ±5%，12 次平均的标准误约 1.4%：温度 1.0 下 +2.8%，方向一致但幅度小。生产当年这一组合是 +8.5–10%，那时
  还没有热专家池，每丢一个 token 约省 4.9 ms CPU；现在大部分路由命中 GPU 上的热专家，CPU 每个 token 的代价小得多。
- 温度 0 那行的 93.6 是噪声：贪心请求在两种模式下走的代码一样（草稿 argmax、验证 argmax），每个提示只测 1 次，文本在不同验证
  宽度下会数值分叉，加上自适应步数的状态，接受长度单次能差 20%。
- 照抄是温度 0，贪心请求不截断，测不到标志。另做了端到端检查（`xtier-dev/glm_flags_test3.sh`）：阈值设成 1.5（q 到不了，每个
  采样请求每步都只留 1 个草稿，其余验证 token 全部跳过专家；0.999 时温度 0.01 的草稿分布太尖，几乎截不到），温度 0.01 照抄
  （走采样和截断路径，文本近似确定）：单 / 双并发 0/8、0/8，平均接受长度 1.99（≤ 2，确认每步都截了）。标志若与验证 token 的
  排布错位，保留下来的 token 会少算专家，照抄会出错。

## Qwen：QSA 读 NVFP4 KV

生产 Qwen 用 `--kv-cache-dtype nvfp4`（加 `--speculative-draft-kv-cache-dtype fp8_e4m3`、32K 分块、KV 池 524288 封顶）。官方 main
能存 NVFP4 KV（NVFP4KVCacheMethod：E2M1 成对打包、每 16 个值一个 E4M3 缩放、一个 FP32 全局缩放），但 QSA 后端按普通 K/V 读池子：
decode / verify 的取行和 chunked prefill 的前缀打包把打包后的张量直接交给 Triton，paged decode 还从它取 head 大小。
`packs/qwen4_exp/qsa_fp4.py`（`XTIER_QSA_FP4`，默认开，只在 NVFP4 池上起作用）：

- decode / verify（`_forward_paged_attention`）：照官方代码走，池缓冲换成 BF16 形状的替身（只用来定暂存区和 paged 视图的大小，
  从不读），其中的取行（`qwen_sparse_kv_extraction_compact_triton`）改为从打包行直接读、在寄存器里反量化；
- 有前缀的 prefill（`forward_extend`）：`pack_qsa_prefill_kv` 同样处理；
- 顺带绕开官方的一个 bug：混合池写 KV 时把全局缩放 1.0 当 Python 浮点数传给 `NVFP4KVQuantizeUtil.quantize`，后者把它变成一次
  主机到显卡的拷贝，target verify 的 CUDA 图因此捕获失败。包里在建 NVFP4 池时（还没开始捕获）在显卡上建好 1.0 换掉它，读取端
  用同一个 1.0。

单测 `tests/qsa_fp4_check.py`（Qwen3.8 QSA 的 2 × 256 形状）：decode 取行（紧凑、页对齐两种布局）和 prefill 打包，与官方内核在
官方反量化后的池上算出的结果逐位一致；取 3 × 2048 行 18.7 µs（官方读 BF16 池 16.8 µs）。

端到端（README 口径，`bench/results/context-qwen38-xtier-nvfp4-20261023.json`、`ppl-qwen38-xtier-nvfp4-20261023.json`）：

| Qwen3.8 加速包 | FP8 KV、24K 分块（测速表用） | NVFP4 KV、32K 分块（与生产同参） | 生产 |
|---|---:|---:|---:|
| 64K 冷 prefill | 5511 tok/s | 5348 tok/s | 5405（68K） |
| 512K 冷 prefill | 5011（101.8 s） | 5033（101.4 s） | 4706（500K，106.2 s） |
| decode 每步 64K / 512K | 14.37 / 15.57 ms | 14.47 / 15.64 ms | — |
| KV 池 548288 token（K + V） | 6.28 GB | 4.06 GB | — |
| 建图后空闲显存 | 8.11 GB | 9.79 GB | 10.24 GB |
| PPL（13 段 × 4K） | 3.4817 | 3.4884 | 3.4779 |
| 照抄（单 / 双并发） | 0/8、0/8 | 0/8、0/8 | — |

- 64K prefill 慢 3%：FP8 池时 QSA 走"写 KV + 注意力"的融合路径，NVFP4 池不满足融合条件，分开写、分开读；512K 时 32K 分块少走
  几块，抵掉了。
- PPL 的差异不是 FP4 读取造成的：每段 4K、一块就算完，不从池里读前缀；差在首块注意力用的数据——FP8 池走融合路径，读的是刚
  写进去的 FP8 K/V，NVFP4 池直接用 BF16 的当前 k/v。FP4 读取的正确性由单测（与官方反量化逐位一致）和照抄保证。
- 按比例分池时（`--mem-fraction-static`）NVFP4 只会让池子多装 token（822464 对 548288），空闲显存不变；要像生产一样用
  `--max-total-tokens` 封顶才省得出来。官方的 NVFP4 方法按 flashinfer 的访问规则还分了一块 QSA 用不到的反量化工作区（约 0.56 GB）。
- 测速表仍用 FP8 KV + 24K 分块（64K prefill 快 3%，精度不低于 NVFP4）。要多 1.7 GB 显存或更大的 KV 池时，`run-qwen-long.sh` 后面加
  `--kv-cache-dtype nvfp4 --speculative-draft-kv-cache-dtype fp8_e4m3 --chunked-prefill-size 32768 --max-total-tokens 524288`。

## README 重测（加速包三行，10-23 夜）

三个加速包用当前代码按 README 口径重测（`xtier-dev/readme_speed.sh qwen-pack glm-pack ds-pack`；GLM 加速包那行加了
`--speculative-use-rejection-sampling`，Qwen 仍是 FP8 KV + 24K 分块；官方 main 三行沿用上次的结果）：

| 配置 | 上下文 | 首 token | prefill | decode 中位（5 次） | decode 范围 | 每步 | 接受长度 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen-pack | 64K | 11.9 s | 5513 | 151.1 | 135.0–171.9 | 14.34 ms | 2.21 |
| qwen-pack | 510K | 101.7 s | 5015 | 145.5 | 128.8–154.3 | 15.49 ms | 2.23 |
| glm-pack（60 槽，拒绝采样） | 64K | 23.4 s | 2808 | 100.0 | 92.9–106.6 | 29.16 ms | 2.83 |
| glm-pack（60 槽，拒绝采样） | 510K | 192.0 s | 2656 | 91.9 | 91.0–98.7 | 26.78 ms | 2.49 |
| ds-pack | 64K | 15.7 s | 4169 | 96.2 | 92.9–107.2 | 32.59 ms | 3.23 |
| ds-pack | 510K | 110.3 s | 4623 | 85.4 | 78.7–94.2 | 35.99 ms | 3.10 |

- GLM：decode 94.7 → 100.0（64K）、86.7 → 91.9 tok/s（510K），主要来自接受长度（2.74 → 2.83、2.30 → 2.49），每步 29.27 → 29.16、
  27.07 → 26.78 ms。
- Qwen、DS 的代码路径没变（投机包的钩子在不开拒绝采样时直接放行），每步耗时与上次差 0.2–3%，属于正常波动；DS 64K prefill
  3977 → 4169 是 Engram 行缓存更热。
- 原始数据 `bench/results/context-{qwen38,glm53,ds41}-xtier-pack-20261023.json`（上一版留在 `xtier-dev/results/readme-speed/*-2147.json`）。

## DeepSeek：剩余差距排查（限时 30 分钟）

README 那一行 64K 每步 33.7 ms，生产 agent_bench 的 64K 会话 31.6 ms。用一份 GPU 轨迹（`xtier-dev/results/ds-dtrace`：8K 提示的
12 个 decode 步，profiler 下每步 37.6 ms，不开时 31.2 ms）和两边的热专家日志查了：

- 整卡空闲每步 13.5 ms，其中 10.2 ms 是各 MoE 层在等 CPU 专家（每步约 39 次、每次约 260 µs），与生产是同一结构；其余每项都在
  1.6 ms/步以下，GPU 等 Engram 取行（第 1、14 层）的空闲排不进前十几项。
- 专家池的自适应换入每步约 31 个专家（约 590 MB/步，16 GB/s），每轮都打满预算（每 2 个窗口一轮、每窗口最多 32 次）。生产
  lkqmoe 参数相同，也是每轮打满（23360 次 / 365 轮）。命中率 XTier 0.56–0.60、生产 0.52–0.54；CPU 每层调用 XTier 330–385 µs、
  生产 384–401 µs。
- 生产的每步时间来自 agent_bench（编码代理会话），本身随内容变化很大：64K 会话 31.6 ms，500K 会话 25.7 ms（DSpark 每步固定验证
  6 个 token，差别来自工具输出的专家局部性）。README 的测法是语料加审阅任务，负载不同。

结论：代码层面没找到差距，"33.7 对 31.5 ms" 比的是两种负载。按约定到此为止；要同口径的数，需要在 XTier 上跑一遍生产的 agent_bench。
