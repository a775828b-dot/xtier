# 冷存储分段、尾部跳过（2026-11-06）

> 同日下午尾部跳过改了规则：不再只管冷专家（CPU 算的专家都可以跳），写法改成「保留/总共」，这里的 R = 4 即 `4/6`、Kimi 的 R = 9 即 `9/16`。
> 见[新报告](2026-11-06-tail-skip-cpu.md)。下面是当时（只跳冷专家）的结果。

冷专家池的两个选项，从 Kimi K3.1 Pro 实验分支（`kimi-k31`，已归档）移植到 main，**都默认关**：不设这几个变量时行为和以前一样，
以前写的冷存储照常复用。规则和实现见[设计](../design-cold-experts.md)的「结构」和「尾部跳过」两节。

## 改了什么

| 选项 | 作用 | 什么时候用 |
|---|---|---|
| `XTIER_COLD_UNITS=3` 和 `XTIER_COLD_PARTS` | 每片再切成 3 段，一条记录 24 段，按盘速分到各盘 | 盘数不是 4 块（和每条记录的 8 片对不上），或者盘速不一样 |
| `tools/cold_relayout.py` | 已有的存储换布局（换盘、分段、按盘速分），逐条读回比对，旧存储不动 | 换盘、加盘；模型打过洞时只能这样改布局 |
| `XTIER_TAIL_SKIP=R`（**会改变输出**） | decode 一步里，每个 token 都排在第 R 名以后的冷专家不读、不算 | 大部分专家在盘上、top-k 大、读盘是瓶颈 |
| `XTIER_TAIL_SKIP_EVAL=1`、`bench/ppl_eval.py --top N`、`bench/kl_compare.py` | 测尾部跳过的精度：KL、首位 token 改变比例、NLL 差 | 选 R 之前 |

来源（kimi-k31 分支的[报告](https://github.com/a775828b-dot/xtier/blob/kimi-k31/docs/reports/2026-11-05-kimi-k31.md)，Kimi K3.1 Pro：
92 层 × 896 个专家、top-16，78% 的专家只在盘上）：

- 4 块 9100 PRO 再加 2 块傲腾、按 `5:5:5:5:2:2` 分段：decode 快 13.6%。原来只能整片分（2:2:1:1:1:1），加盘没有收益。
- R = 9：decode 4.94 → 10.31 tok/s（开 DFlash2 的代码题），KL 0.024，好比 Q5_K_M 和 Q4_K_M 之间。

## 测试

测试机：2× EPYC 9334（NPS4，8 个节点）、384 GiB、RTX PRO 5000 72GB；冷存储放在 4 块三星 9100 PRO（PCIe 5.0）上。

### 单层

| 测试 | 布局 | 结果 |
|---|---|---|
| `tests/cold_passes_check.py`（DS 第 20 层，CPU 分批） | 整份 2 盘、整份 5:3、分段 2 盘、分段 3 盘、分段 10:8:6 | 都通过；5 种布局的误差完全相同（最大 1.5e-7） |
| `tests/cold_prefill_check.py`（GPU prefill） | 整份 2 盘、分段 3 盘 | 都通过，两种布局误差相同 |
| `tests/cold_punch_check.py`（打洞、恢复） | 整份、分段 | 都通过 |
| `tests/tail_skip_check.py`（新加） | 整份 2 盘、分段 3 盘 | 8 个 token 以内的调用和完整计划（喂同样去掉那些路由的输入）逐位相同，跳过的专家数对得上；两个 token 共用专家时按"每个 token 都排得低"判断；16 个 token 不跳；评估用的掩码符合规则 |
| GLM-5.3 第 20 层（NVFP4），一次性脚本 | 分段 3 盘、整份 3 盘 | decode 大小的调用和完整计划逐位相同 |
| `tests/engine_ab.py`（DS 预设，9 个用例） | 改动前后的引擎库 | 输出摘要全部相同：默认设置下 CPU 引擎逐位不变 |
| CI 三项（`repo_check`、启动台、规划器） | — | 通过 |

### 整机

DeepSeek V4.1 Flash 1017：启动台 DS 卡的默认配置加「冷专家 30%」，开快速加载（启动台的做法）。`cold_e2e.py` 做三道贪心题（照抄 600 词、
写代码、算术），`cold_ppl.py` 给 600 / 3000 / 12000 token 的文档和 3000 token 的代码打分（两轮）。

| 组 | 代码和设置 | 冷存储 | 照抄 | NLL 差（对 A，四段） | 照抄题 decode |
|---|---|---|---|---|---|
| A | 改动前的 main | 启动时写入，整份 4 盘 | 0 错 | 参照（A 自己两轮之间差到 0.004） | 87.9 tok/s |
| B | 改动后，默认设置 | 复用 A 写的，40 层从冷存储读进内存槽 | 和 A 逐字相同 | −0.006 / +0.006 / +0.002 / −0.001 | 87.6 tok/s |
| C | 分段，3 盘 | `cold_relayout.py` 转换（268.9 GiB、116 s、逐条比对），复用 | 和 A 逐字相同 | −0.006 / +0.002 / −0.002 / +0.001 | 77.3 tok/s |
| C0 | 同 C，不开快速加载 | 复用（每条记录抽样核对） | 和 A 逐字相同 | 没测 | 77.9 tok/s |
| D1 | `XTIER_TAIL_SKIP=4` | 复用 A 写的 | 和 A 逐字相同 | — | 92.0 tok/s |
| D2 | R = 4 加 `XTIER_TAIL_SKIP_EVAL=1` | 复用 A 写的 | 0 错 | −0.001 / −0.003 / +0.001 / −0.005 | 92.4 tok/s |

GLM-5.3 Flash（NVFP4）：G0 不开冷专家；G1 冷专家 30%、分段 4 盘（6:6:6:6，启动时写入 42 层）。

- NLL 差（G1 对 G0）−0.019 / +0.004 / −0.004 / −0.006。600 token 那段 G0 自己两轮之间就差 0.021；之前 NVFP4 全局缩放出错时，这一段是 1.64 → 2.0。
- G1 的照抄题没过：模型先在推理里把 40 行全抄了一遍（逐行比对 0 处不同），再作答，到 1,400 token 上限被截断；G0 只推理了 157 个字符就作答。
  这是轨迹不同（第一句推理就选了另一种说法），不是读错：同一层的逐位检查见上面单层的表。

说明：

- **代码题和数学题每次启动都可能不同**，B、C、C0、D1 都和 A 不一样。同一份 main 以前跑的 DS e2e 里，代码题就有两种答案（41 次一种、11 次另一种），
  A 这次落在少的那一种；A 自己两轮 PPL 之间，单个 token 的 logprob 差到 6.2（top-k 原子竞争，见 [2026-11-01 报告](2026-11-01-audit-fixes.md)）。
  所以只把照抄和 NLL 当判据。
- **尾部跳过 R = 4**（DS 是 top-6，跳最后两名）：
  - 每步冷读 13.3–13.7 → 8.5 个（−37%），e2e 期间每块盘读 32.3 → 19.5 GB；
  - 照抄题 decode 快约 5%（87.6–87.9 → 92.0–92.4 tok/s），NLL 变化在两次启动的波动内；
  - DS 卸载 30% 时读盘只占每步的一小部分（约 13 次 × 0.5 ms），所以快得不多。专家大多在盘上的模型才明显（Kimi K3.1 是 2.1 倍）。
- **3 块盘分段（C）比 4 块整份慢**：每分钟 decode 等读盘 6.1 s 对 3.1–3.3 s（读的量差不多），照抄题 decode 77 对 88 tok/s。盘少了带宽本来就少，
  分段后每条记录 24 段（DS 每段约 0.8 MB），读请求也多了 3 倍。**4 块一样快的盘不要开分段**；没测同样 4 块盘分段和不分段的差。

## 已知限制

- 分段只有 1 和 3 两档；`XTIER_COLD_PARTS` 加起来要等于 节点数 × 段数（NPS4 时 24）。
- 尾部跳过只在不超过 `XTIER_TAIL_SKIP_TOKENS`（8）个 token 的 CPU 调用里生效。两个请求同时 decode 时，要每个 token 都排得低才跳，省得少。只管冷专家，内存里的照算。
- R 要按模型的 top-k 选（R ≥ top-k 等于关），先用 `XTIER_TAIL_SKIP_EVAL=1` 测 KL 再定。main 上只测了 DS 的 R = 4。

## 文件

`xtier/cache/cold.py`、`xtier/cache/tail.py`、`csrc/engine/engine.cpp`（`tail_skip`、按段读盘）、`xtier/engine/__init__.py`、`xtier/backend/method.py`、
`tools/cold_relayout.py`、`tools/cold_punch.py`、`bench/ppl_eval.py`、`bench/kl_compare.py`、`tests/tail_skip_check.py`、`tests/cold_passes_check.py`、
`tests/cold_punch_check.py`、`tests/cold_check.py`。
