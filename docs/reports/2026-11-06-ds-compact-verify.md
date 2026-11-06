# DeepSeek 变长验证（compact，2026-11-06）

DSpark 的验证有三种模式（`SGLANG_RAGGED_VERIFY_MODE`）：

| 模式 | 每步怎么验证 |
|---|---|
| `static`（默认） | 每个请求固定验证 6 个草稿 token |
| `cap-accept` | 仍验证 6 个，接受长度按规划器的预算截断，用来测量 |
| `compact`（变长验证） | 按草稿置信度和成本表，每个请求只验 1–6 个，几个请求的 token 紧挨着拼在一起 |

DeepSeek V4.1 以前不能用 compact：Engram 哈希器和低压缩比注意力层（`token_req_indices`）都假定每个请求验证一样多的 token。
[#13](https://github.com/a775828b-dot/xtier/pull/13)（[cyrus-xc](https://github.com/cyrus-xc)）补上了这两处；这个 PR 的控制台部分没有合并。

> 同日晚上跟进 SGLang main 27bc1a0（v0.1.2）以后，这两处由 SGLang 自己做（按布局的 `qo_indptr`，CUDA 图回放时会刷新），#13 的这两个钩子
> 已删，只留校准预算那一处，见[跟进报告](2026-11-06-sglang-27bc1a0.md)。「改了什么」和「合并 #13 时的测试」两节是合并 #13 时的记录。

## 更正（11-06 夜）：要配本机实测的成本表

下面「合并 #13 时的测试」没有测到真正的变长验证：DS 启动脚本当时传的成本表斜率全是 0（每步耗时和验证几个 token 无关），规划器
没有理由少验证，基本每步都验证满宽（SGLang 的文档也写了：没有实测的成本表时 compact 不起作用）。所以那组「0 错、没有收益」说明不了
compact 本身。

成本表要在本机测（`tools/sps_profile.py`，步骤见下）。这台机器测出的表在 `configs/ds41-pro5000-nps4/sps-table.json`：每步基础
24.6 ms，两个请求比一个多 1.4 ms；验证 token 在 6 个以内只多 0–4.7 ms，8 个以上要多 22–26 ms。所以单请求时规划器仍验证满 6 个，
两个请求同时才裁剪。

配上这张表重测（SGLang 27bc1a0 / v0.1.2，启动台默认配置）：

| | static | compact + 实测表 |
|---|---|---|
| 单请求照抄（8 段，温度 0） | 0 错 | 0 错（1 段结尾多抄 `</src>`，static 也有这个现象） |
| 两请求同时照抄（8 段） | 0/8 | 0/8 |
| 采样聊天 12 题，两个同时（代码、长文） | 正常 | 正常，没有乱码 |
| 两请求同时：每个请求平均接受 / 每步 | 3.92 / 55.3 ms | 2.84 / 35.6 ms |
| 两请求同时的聊天吞吐 | 135 tok/s | 158 tok/s（+17%） |
| 单请求 decode 每步 | 约 30 ms | 约 29 ms，和 static 一样 |

另外用真实任务（单请求，上下文到 16 万 token）跑了约 2.9 万步，没有问题。#13 的作者反映开了实测表以后输出会乱掉，在 v0.1.2 上
没有复现；把 #13 原来按 `extend_start_loc` 映射的钩子临时挂回去测，也没有复现。

测表：

1. 起一个测量用的服务：`SGLANG_RAGGED_VERIFY_MODE=compact SGLANG_DSPARK_ENABLE_SPS_RECORD=1 SGLANG_SIMULATE_ACC_LEN=1.0 DSPARK=1 configs/ds41-pro5000-nps4/run-ds-xtier.sh`
2. `python tools/sps_profile.py --base-url http://127.0.0.1:18190 --out my-sps-table.json`，几分钟。
3. 停掉这个服务，用 `SGLANG_RAGGED_VERIFY_MODE=compact SPS_TABLE=$PWD/my-sps-table.json DSPARK=1 configs/ds41-pro5000-nps4/run-ds-xtier.sh`
   启动。启动台的 DS 卡可以直接选「DSpark 验证：变长」（用的是这台机器的表）。

默认仍是 static：成本表跟机器有关，别的机器要先测。


## 改了什么

| 文件 | 内容 |
|---|---|
| `xtier/packs/deepseek_v4/ragged_verify.py`（新） | compact 验证时：Engram 按每个请求自己已提交的历史哈希（extend 的不等长路径），不写回历史；`token_req_indices`（`dsv41_sparse`、`deepseek_v4_backend`）按请求起点映射，不再按固定宽度重复；校准时强制的预算不再被"成本表未初始化就全量验证"覆盖 |
| `xtier/packs/deepseek_v4/__init__.py` | 注册这四个钩子；static 下走原来的代码 |
| `configs/ds41-pro5000-nps4/run-ds-xtier.sh` | `SGLANG_RAGGED_VERIFY_MODE` 可以从外面设，默认仍是 `static` |
| `tests/engram_ragged_check.py`（新） | 打包的哈希和逐个请求单独算的相同（不等长、序列起点、图片 token、填充），已提交的历史不变；CPU 和 CUDA 都测 |
| `bench/copy_test.py`、`bench/agent_bench.py` | 单个请求出错时记下来、不中断，最后非零退出；agent_bench 记录 HTTP 失败，`--repo` 是 xtier 布局时也能跑 |

## 合并 #13 时的测试（compact 没有生效，见上面的更正）

测试机：RTX PRO 5000 72GB、2× EPYC 9334（NPS4）、384 GiB。DeepSeek V4.1 Flash 1017，启动台 DS 卡的默认配置，开快速加载。
照抄：`cold_e2e.py` 的 600 词照抄题，加 `bench/copy_test.py` 的 8 段代码（单请求、两请求同时），都是温度 0。

| | static | compact |
|---|---|---|
| 单请求照抄（600 词 + 8 段） | 0 错 | 0 错 |
| 两请求同时照抄（8 段） | 0/8 | 0/8 |
| 600 词照抄题的 decode | 100.5 tok/s | 82.8 tok/s（−18%） |
| `bench/decode_speed.py` 单请求 | 119.8 tok/s | 123.3 tok/s（+2.9%） |
| `bench/decode_speed.py` 两请求合计 | 111.1 tok/s | 113.4 tok/s（+2.1%） |
| 服务日志里单请求 decode 的平均 | 接受 4.50，141.9 tok/s | 接受 4.07，129.7 tok/s（−8.6%） |

说明：

- 两组 8 段里各有 1 段和原文不同：static 结尾多抄一个 `</src>`，compact 第一行少了缩进，都不是错字。
- agent_bench 两次的 64k 会话都在第一轮就结束（模型没有调用工具），没法比较。
- 当时以为没有收益的原因是规划器按两步前的置信度定验证长度（`lag=2`）、成本表只有一档；实际是成本表斜率为 0，规划器没有理由少验证，compact 等于没开。

## 结论

默认仍是 static。compact 要配本机实测的成本表，做法见上面的更正。

10 月在旧管线（lsglang_ds4.1）上试过 compact：单请求会改错字，已放弃（当时两请求同时 decode 快 13.9%，单请求 +1–3%）。
这次在官方 SGLang main 上没再出现改错字。
