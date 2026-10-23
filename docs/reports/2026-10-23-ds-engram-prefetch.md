# 2026-10-23 晚：DeepSeek 加速包移植生产的 Engram 提前查表与草稿词表头 FP8

目标是 XTier 加载加速包后速度与生产持平或差 5% 以内。README 测速时 DS 的 decode 每步比生产慢约 20%（64K：36.8 对约 31.5 ms），
prefill 不慢。对照 lsglang_ds4.1 10-19 的优化清单，XTier 还缺三项：Engram 提前查表（生产测得 GPU 每步少等约 2.3 ms）、草稿
词表头 FP8（1.11 → 0.62 ms）、DSpark 验证免同步（约 4 ms）。前两项移进了 `packs/deepseek_v4`，第三项在官方 main 上不存在。

## 改了什么

| 项目 | 开关（默认开） | 做法 |
|---|---|---|
| Engram 提前查表 | `XTIER_ENGRAM_PREFETCH` | 每次 forward 一开头就把所有 token 的哈希算好。`EngramHasher.forward` 一返回，两层 Engram（第 1、14 层）的查表（id 拷到主机、行表的主机函数查缓存或读盘、行拷回、反量化）就各在一条侧流上开始，与前面的层重叠；到了该层只等事件。取到的行不变，CUDA 图里是标准的分叉与汇合。生产只在 decode / 验证步做，XTier 的 prefill 也做（每个 22528 token 的块每层要查 54 万行）。 |
| 草稿词表头 FP8 | `XTIER_DRAFT_LMHEAD_FP8` | DSpark 草稿每步都要读一遍主模型的 BF16 词表头（129280 × 5120，1.32 GB）。草稿挂上主模型词表头时（`attach_shared_modules`，在图捕获之前）做一份按行缩放的 FP8 E4M3 副本（多占 0.62 GiB），投影改用 `torch._scaled_mm`（激活按 token 缩放）。只影响草稿提案，每个输出 token 仍由主模型的 BF16 词表头验证。 |

单测 `tests/draft_lmhead_fp8_check.py`：相对 L2 误差 3.75%（FP8 E4M3 的正常水平）；CUDA 图重放与直接调用逐位相同；每次调用
5 / 10 行时 1.107 → 0.622 ms，与生产一致。随机权重的 logits 几乎打平，top-1 一致率只作参考，实际效果看服务里的接受长度。

## 结果

同一份测速脚本（`bench/context_speed.py`，8K / 64K 提示，1K 输出，5 次回答取中位），先在旧代码的服务上测基线，再换新代码重启：

| | 基线 | 新代码 | 变化 |
|---|---:|---:|---:|
| 8K 每步 | 32.45 ms（32.1–34.2） | 30.09 ms（29.8–32.6） | −7.3% |
| 64K 每步 | 35.24 ms（34.6–36.9） | 32.83 ms（32.4–35.7） | −6.8% |
| 64K 冷 prefill | 3999 tok/s | 4657 tok/s | +16% |
| 8K 冷 prefill | 2537 tok/s | 2501 tok/s | 持平 |
| 照抄（8 段，温度 0） | — | 单并发 0/8、双并发 0/8 | |
| PPL（13 段 × 4K） | 3.3151 / 3.3187 | 3.3171 | 持平 |
| CPU 专家每层调用 | 约 410 µs | 377 µs | |

- 和生产比（这次 A/B）：64K 每步 32.8 对 31.4–31.6 ms（agent_bench 64K 会话），+4%；8K 30.1 对约 30 ms（20 个多领域提示），持平。
- 接受长度与之前相同（2.8–3.3）：FP8 草稿词表头没有让草稿变差。

README 口径重测（干净重启、32K 预热、64K / 510K 冷提示各 5 次回答，`bench/results/context-ds41-xtier-pack-20261023.json`）：

| | 加速包（10-23 下午） | 加上两项 | 变化 |
|---|---:|---:|---:|
| 64K 每步 | 36.78 ms | 33.73 ms（33.4–35.2） | −8.3% |
| 510K 每步 | 38.20 ms | 34.98 ms（34.7–36.7） | −8.4% |
| 64K 冷 prefill | 3962 tok/s | 3977 tok/s | 持平 |
| 510K 冷 prefill | 4363 tok/s | 4435 tok/s | +1.7% |

- decode 两次测都快 7–8%。prefill 的差别来自 Engram 行缓存：A/B 那次 64K 提示前面跑过同一段语料的 8K 提示，行大多已在缓存里，
  查表能完全藏在前面层的计算后面（+16%）；README 这次是冷提示，大量行要从 NVMe 读，查得比前面层算得慢，藏不住（持平）。
- 和生产比：64K 每步 33.7 对 31.4–31.6 ms（+7%），A/B 那次 32.8 ms（+4%），两次之间差 3%，与生产的差距在 5% 上下。

## DSpark 验证免同步：官方 main 上没有这处同步

生产修的是 Lsglang 基线里 `decode_cuda_graph_runner` 为选 c128 图做的 `seq_lens.max().item()`（DSpark 验证批次没有
`seq_lens_cpu`，主机每步都要等上一步跑完，约 4 ms）。官方 main 的这段已经改写，不再按序列长度挑 c128 图。DSpark 规划器里剩下的
`.tolist()`（`dspark_planner.py` 460 行、`dspark_verify.py` 415 行）都在变长验证的布局分支上，固定验证（`SGLANG_RAGGED_VERIFY_MODE=static`）
不会走到；DP 分层的 `all_gather` 用的是 CPU 张量。所以这一项不移植。

## 测量上的两个坑

- GPU 剖析器（`/start_profile`，只开 GPU）开着时每步约 71 ms，不开 32 ms：轨迹里的空隙不能当真，这次改用 A/B 直接比每步耗时。
- 这台机器的 py-spy 需要 root（ptrace 限制），没有采到调度进程的栈。
