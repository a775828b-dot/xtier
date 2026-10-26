# 2026-10-26：弹性热专家槽（prefill 余量给热专家）

目标：prefill 的激活（和 SGLang 的其他临时缓冲）只在 prefill 时占显存，平时这部分是空的——decode 时的显存占用对 prefill 峰值，
GLM 约 9.6 GiB、DeepSeek 约 10.6 GiB、MiniMax 约 10 GiB、Qwen 约 7.5 GiB。decode 时把它借给热专家，prefill 要用时先还回来。
`XTIER_CACHE_ELASTIC=1`（启动台“弹性热专家”选项）。

结果（生产的 agent_bench 两轮，同一次启动；每步 = 长回复每个验证步的耗时）：

| 模型 | 每步 关 → 开 | 命中率 关 → 开 | 每层 CPU 调用 关 → 开 | 冷 prefill | 启动台默认 |
|---|---|---|---|---|---|
| DeepSeek V4.1 1017（至多 68 槽） | 32.6 → 31.2 ms（−4.3%） | 0.565 → 0.61–0.63 | 390 → 344–360 µs | 不变 | 开 |
| GLM-5.3 Flash | 27.5 → 27.1 ms（−1.5%） | 0.517 → 0.556–0.573 | 297 → 256–284 µs | 不变 | 开 |
| MiniMax-M3.1-Flash | 69.2 → 69.2 ms | 0.245 → 0.285 | 533 → 515 µs | 500K −1.9% | 关 |

- 同一配置跑两次 agent_bench，tok/s 差 2–3%（输出和接受长度不同），所以收益看路由记录（`XTIER_ROUTE_TRACE`，命中率、每次调用的
  CPU 专家、窗口时间）和服务日志里按额外槽数分组的 CPU 时间；GLM 第一次“开”的运行一个额外槽都没加上（见下面第 6 条），它和“关”
  的 tok/s 也差了 2.2%。
- DeepSeek：超过约 66 槽后 GPU 一侧（每个热专家约 16.7 µs，已接近显存带宽）比 CPU 一侧慢，规划器的耗时模型给出的拐点，所以封顶 68。
- MiniMax：一档额外槽（57 层 × 30.4 MiB）就是 1.69 GiB，它的 prefill 还要 5.8 GB 的专家暂存区，decode 时只加得上 3 档；CPU 每层
  省 3%，每步没有变化，500K 冷 prefill 慢 1.9%（每次大 prefill 前要先还槽）。选项保留，默认关。

## Qwen：多放常驻层、热专家槽全部动态（不采用）

Qwen 的 decode 受 GPU 限制，多放热专家没有用；设想是启动时多放常驻层（prefill 少上传），池化层的槽在 decode 时动态补回。规划器在
现有显存上再加 3 GiB 预算（`plan_auto.py --add-gib 3.0`，KEEP）：31 层常驻 + 16 层 × 72–112 槽，池化层只常备一成
（`XTIER_CACHE_ELASTIC_MAX=1.0 XTIER_CACHE_ELASTIC_BASE=0.1`），prefill 时的静态显存 41.2 GiB（现在 41.7），decode 时 44.7 GiB；
预测 decode 3.704 → 3.624 ms / 窗口、prefill 每块上传 452 → 402 ms。实测：

| Qwen3.8 | 冷 prefill 64K | 冷 prefill 500K | decode（长回复） | 每步 |
|---|---:|---:|---:|---:|
| 自动规划（29 层常驻） | 5,715 / 5,610 | 5,170 / 5,150 | 192.8 | 13.6 ms |
| 31 层常驻 + 动态槽 | 5,804 / 5,616 | 5,234 / 5,221 | 167.7 | 15.0 ms |

prefill 只快约 1%，decode 慢 10%：常驻层的专家全在 GPU 上算，多两层常驻是把本来与 CPU 并行的那部分也压到 GPU 上，而 Qwen 的 GPU
本来就是瓶颈；规划器的模型只算 MoE 那一段，看不到 GPU 已经满载。不采用，Qwen 保持自动规划、不开弹性。

## 端到端回归

启动台默认配置（DeepSeek、GLM 开弹性；Qwen 自动布局、不开弹性），本机 dsh（headless）经网关调用：一个会话用读文件工具逐个读
4–5K token 的笔记，把上下文涨到约 49 万（KV 用量 0.94），再只凭上下文回答开头三个文件里门的颜色；然后是简单编码任务（写脚本、
写单测并运行），每个模型至少 30 分钟。MiniMax 暂停开发，没有跑。

| 模型 | 时长 | dsh 运行 | 满 KV 会话 | 回忆 | 服务端错误 | 弹性槽 |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 1017 | 35 + 18 分钟（同一次启动） | 71 次全部成功 | 491,975 token（0.94） | 3 / 3 | 0 | 还 131 次，没关过 |
| GLM-5.3 Flash | 32 分钟 | 73 次全部成功 | 493,217 token（0.94） | 3 / 3 | 0 | 还 112 次，没关过 |
| Qwen3.8 Flash Next | 32 分钟 | 152 次全部成功 | 492,754 token（0.94） | 3 / 3 | 0 | 不开 |

- 已知限制：需求只按 forward 的 token 数学习（取见过的最大值），不分上下文长度。长上下文时同样大小的 prefill 要的显存多得多
  （GLM 49 万上下文时 4.4K token 要 6.9 GiB，DeepSeek 40 万时 4.2K 要 4.4 GiB），之后短上下文的 prefill 也按这个还槽：GLM 回归
  后半段每个新会话的首个 prefill（约 6.7K token）都把 13 档额外槽全部还回（约 0.2 s），空闲后再补回。安全但偏保守，以后可以按
  上下文长度分档学习。
- 用 dsh 打满 KV 要绕开它的两处估算（与 XTier 无关）：自动压缩和输出上限都按“4 字符一个 token”估算上下文，这些英文笔记比真实
  多 1.3–1.5 倍——默认配置下真实约 28.5 万 token 就压缩；声明 524288 的窗口时，真实 34.4 万就把 max_tokens 截成 1（服务端照做，
  只回 1 个 token）。所以满 KV 会话关了 dsh 的自动压缩、声明 1M 的窗口，真实上限由服务端把关。

## 做法

- 部分缓存层按最多的槽预留地址空间（`cuMemAddressReserve`），只给用得到的槽挂物理页（每 2 MiB 一页，单独 `cuMemCreate` /
  `cuMemMap`，任何尾部都能单独还回去）。arena 指针始终不变，decode 的 CUDA 图不用重新捕获（`csrc/mailbox` 的
  `xtier_mailbox_vmm_*`，`xtier/cache/elastic.py arena`）。
- 每层：常备槽 `base`（配置的槽数 × `XTIER_CACHE_ELASTIC_BASE`，默认 1.0）+ 额外槽（所有层同一个数，最多到地址空间
  `XTIER_CACHE_ELASTIC_MAX` × 配置的槽数，默认 2.0；`XTIER_CACHE_ELASTIC_MAX_SLOTS` 直接给槽数）。换入策略只在可用槽里选，空槽当作
  最差的受害者先被填满——补槽用的是平时换入的预算，不额外搬运。
- 加槽：见过一次真实请求的 prefill 后，连续 `XTIER_CACHE_ELASTIC_IDLE`（32）个 decode forward 没有会分配显存的 forward，后台线程每次
  挂 2 档，直到空闲显存只剩预留（`XTIER_CACHE_ELASTIC_RESERVE_GIB`，2 GiB）加“常见小 prefill 的余量”（最近 32 次不超过
  `XTIER_CACHE_ELASTIC_SMALL`（4096）token 的 forward 的需求 75 分位 × 1.25）——编码代理每轮的小 prefill 不用还槽。
- 还槽：每个会分配显存的 forward（prefill、混合、草稿 extend）之前，按预测需求 × `XTIER_CACHE_ELASTIC_MARGIN`（1.25）+ 预留算缺口，
  只还缺的那几档：暂停换入策略 → 在流上补跑一次提交内核、同步 → 等进行中的拷贝落地 → 那些槽里的专家出表（不带上传的命令，立即
  落地，提交内核不发布）→ 再跑一次提交内核、同步 → 解除映射。GLM 一次还 14 档（7.8 GiB）约 0.2 s。
- 需求：每次 forward 后记录已分配峰值、已预留峰值对 forward 前的增量，取大者；按 token 数在见过的大小之间插值（取累计最大值，
  保证单调）；比见过的都大的 forward 一律全还。只看驱动层面的空闲显存（分配器缓存里的空闲块可能太碎），不够时先清空分配器缓存。

## 保险

- 预留 2 GiB 永远不给槽（内核首次加载、各类工作区）；
- 每 16 个 decode forward 检查一次：空闲显存低于 `XTIER_CACHE_ELASTIC_CRITICAL_GIB`（1 GiB）就全部还回，
  `XTIER_CACHE_ELASTIC_COOLDOWN_S`（600 s）内不再加槽；
- forward 里真出现显存不足：全部还回、清空分配器缓存、本进程关掉弹性，然后在 except 块之外重跑一次这个 forward；
- 加 / 还槽的任何一步出错：本进程关掉弹性（换入策略恢复），槽保持原样，服务照常；
- 启动阶段（CUDA 图捕获、自适应投机的多套捕获）不加槽；
- 常备槽扣掉 2 MiB 取整的损耗（`fit_base`：前面让出的槽留下的富余先给后面的层抵取整，累计映射量始终不超过原来的槽；GLM 42 层
  共让出约 12 槽、DeepSeek 9 槽、MiniMax 5 槽），额外槽全部还回时显存不比不开弹性多。

## 冒烟测试中修掉的问题

GLM（`xtier-dev/tp-sim/elastic_glm_smoke.sh`：预热 → 长回复 → 3 万 token 提示 → 一边长回复一边 6 万 token 提示 → 照抄测试
并发 1 / 2；同样的流程不开弹性全部通过、照抄 0 / 8；修完后开弹性同样全部通过、照抄 0 / 8）：

1. 槽在 CUDA 图捕获期间就加上了（捕获前 decode 大小的 forward 被当成了 decode），之后的捕获只剩 2.1 GiB → 先要见过 prefill。
2. 取整损耗约 170 MB，让本来就只剩 0.6 GiB 的 16K 分块 prefill 显存不足 → `fit_base`。
3. 把分配器缓存里的空闲块算作可用，只还了 12 档就去分配 1 GiB → 只看驱动空闲、不够先清缓存。
4. 显存不足后在 except 块里重跑：失败那次的栈帧还被异常引用着，它们的张量没释放，全部还回后仍不足 → 块外重跑。
5. 学到的需求对没见过的大小取下一个更大的：700–1500 token 的 prefill 每 8 s 就还 4–10 档又补回 → 插值 + 加槽时留小 prefill 的余量
   （修后照抄测试期间一次都没还）。
6. 余量按全部 prefill 的 75 分位算：agent_bench 的 16K 分块（约 10 GiB）把空闲显存全留了出去，一次都没加槽 → 只按小 prefill 算。

原始数据：`bench/results/agent-bench-{ds41,glm53}-elastic-r{1,2}-20261026.json`（开）、
`agent-bench-{qwen38,minimax31}-xtier-r{1,2}-20261026.json`（启动台默认配置）；关弹性的对照与 Qwen / MiniMax 的弹性运行在
`xtier-dev/results/planner/`。
