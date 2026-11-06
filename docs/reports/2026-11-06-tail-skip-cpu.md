# 尾部跳过不再只管冷专家（2026-11-06）

同一天上午移植进 main 的尾部跳过（[报告](2026-11-06-cold-units-tail-skip.md)）只跳不在内存的冷专家，不开冷专家时不起作用。现在改成：
**decode 时 CPU 算的专家都可以跳**，不用开冷专家；写法改成「保留/总共」。**默认关，会改变输出。**

## 改了什么

| 项 | 以前 | 现在 |
|---|---|---|
| 写法 | `XTIER_TAIL_SKIP=R`（第 R 名起跳） | `XTIER_TAIL_SKIP=保留/总共`，如 DeepSeek V4.1（top-6）的 `5/6`、`4/6`、`3/6`；只认"数字/数字"，单写一个数、总数和模型的 top-k 对不上，启动时报错；不设或 `0` 是关 |
| 跳哪些 | 不在内存的冷专家 | CPU 算的专家（在内存或在盘上）；GPU 上的热专家照算，它们的权重仍参与排名 |
| 在哪跳 | 不超过 8 个 token 的 CPU 调用（prefill 最后不满 8 个 token 的一段也会跳） | 只在 decode 路径（mailbox）：decode 一步、投机验证窗口、不超过 `XTIER_DECODE_MAX` 的短 prompt；prefill 不跳 |
| 草稿模型 | — | DSpark 草稿模型的 MoE 层（top-3）不动 |
| 统计 | 冷统计第 7 项 | 新接口 `xtier_tail_stats`，日志（`XTIER_CACHE_LOG=N` 时每 N 步）一行 `[xtier] tail skip K/N: ... CPU experts per layer call left out`；冷统计第 7 项改成省下的读盘数 |

引擎加了 `xtier_forward_decode`（mailbox 调它，规则只在这里生效）和 `xtier_tail_stats`。规则本身没变：一次调用里没有任何 token
把某个专家排进前「保留」名（按该 token top-k 里的权重排，同权重按位置），这个专家就不读、不算，其余路由不重新归一化。

## 测试

测试机：2× EPYC 9334（NPS4，8 个节点）、384 GiB、RTX PRO 5000 72GB。

### 单层

| 测试 | 结果 |
|---|---|
| `tests/tail_skip_check.py`（DS 第 20 层，4/6） | 通过：全部专家在内存、30% 冷专家两种计划下，decode 调用都和"去掉那些路由的普通调用"逐位相同，跳过数、读盘数、省下的读盘数都对得上；GPU 路由（−1）只参与排名；普通调用（prefill 最后一段）和 16 个 token 的调用不跳；评估掩码按 GPU 热专家表；12 种错误写法引擎和 Python 都拒绝，`0` 是关，总数不对时拒绝且不跳 |
| `tests/engine_ab.py`（DS 预设，9 个用例） | 不开时输出摘要和改动前相同 |

### 整机

DeepSeek V4.1 Flash 1017，启动台 DS 卡的默认配置（每层 50 个热专家槽 + 弹性、DSpark、Engram 行缓存 4 GiB，**不卸载**），开快速加载。
每组：`cold_e2e.py` 三道贪心题、agent_bench（64k、500k 会话各一轮）。精度另起一次，加 `XTIER_TAIL_SKIP_EVAL=1`（每个 prompt token 单独套规则），
`cold_ppl.py` 给 600 / 3000 / 12000 token 的文档和 3000 token 的代码打分，`bench/ppl_eval.py --top 20` 加 `bench/kl_compare.py` 算 KL。

| 设置 | CPU 算的专家里跳过 | 每层 CPU 时间 | 64k 会话每步 | 照抄题 | NLL 差（对第一组，四段） | KL / 首位 token 改变 |
|---|---|---|---|---|---|---|
| 关（改动前） | — | 342 µs | 28.69 ms | 98.1 tok/s，0 错 | 参照；自己两轮之间差 0.014 / 0.002 / 0.001 / 0.006 | 同一次启动跑两遍：0.0102 / 4.3% |
| 关（改动后） | — | — | — | 99.9 tok/s，逐字相同 | +0.010 / +0.003 / −0.002 / +0.004 | 0.0101 / 4.4% |
| 5/6 | 22–23% | 272 µs | 28.02 ms（−2.3%） | 100.3 tok/s，逐字相同 | +0.020 / +0.007 / −0.007 / −0.006 | 0.0175 / 5.6% |
| 4/6 | 42–43% | 219 µs | 27.16 ms（−5.3%） | 101.4 tok/s，逐字相同 | +0.030 / +0.019 / +0.003 / −0.006 | 没测 |

3/6 没测。每层 CPU 时间取 agent 阶段最后一行日志（每层每步平均 23.4 个不同的专家，GPU 算其中约 62%）。

说明：

- **CPU 省得多，每步快得少**：每层 MoE 是 GPU 算热专家、CPU 算其余，同时进行，等慢的一边。热专家槽数（含弹性槽）已经把两边调到差不多快，
  CPU 快了以后就等 GPU，所以 5/6 时 CPU −20%、每步只快 2.3%，4/6 时 CPU −36%、每步快 5.3%。要再快得同时减 GPU 那边（少放热专家，
  或 GPU 上也跳），没做。
- **精度**：5/6 的 KL 比噪声底高 0.007；4/6 的 NLL 在 600、3000 token 两段文档上超出两轮之间的波动。评估按每个 token 单独套规则，比实际 decode
  偏严（实际验证窗口里要每个 token 都排得低才跳）。代码题和数学题每次启动都可能不同，只把照抄和 NLL 当判据（见[上午的报告](2026-11-06-cold-units-tail-skip.md)）。
- **DeepSeek 不卸载时不建议开**，默认关。专家大多在盘上、读盘是瓶颈时收益大：Kimi K3.1 Pro 的 9/16 decode 约 2.1 倍，那是只跳冷专家的旧规则测的，
  新规则还会跳内存里的，跳得更多。

## 已知限制

- 两个请求同时 decode 时，一次调用最多 8 个 token 混着两个请求，要每个 token 都排得低才跳，省得少。
- 不超过 `XTIER_DECODE_MAX`（64）个 token 的 prompt 走 decode 路径，也会跳。
- `XTIER_TAIL_SKIP_TOKENS`（默认 8）可以只让更小的调用跳；设 0 时只剩 `XTIER_TAIL_SKIP_EVAL` 的评估。

## 文件

`csrc/engine/engine.cpp`（`tail_skip`、`forward_decode`、`tail_stats`）、`csrc/engine/dispatch.cpp`、`csrc/include/xtier_engine.h`、
`csrc/mailbox/mailbox.cpp`、`xtier/cache/tail.py`、`xtier/backend/method.py`、`xtier/engine/__init__.py`、`xtier/cache/cold.py`、
`tests/tail_skip_check.py`。
