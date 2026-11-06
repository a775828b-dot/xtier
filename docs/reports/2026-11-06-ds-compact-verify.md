# DeepSeek 变长验证（compact，2026-11-06）

DSpark 的验证有三种模式（`SGLANG_RAGGED_VERIFY_MODE`）：

| 模式 | 每步怎么验证 |
|---|---|
| `static`（默认） | 每个请求固定验证 6 个草稿 token |
| `cap-accept` | 仍验证 6 个，接受长度按规划器的预算截断，用来测量 |
| `compact`（变长验证） | 按草稿置信度和成本表，每个请求只验 1–6 个，几个请求的 token 紧挨着拼在一起 |

DeepSeek V4.1 以前不能用 compact：Engram 哈希器和低压缩比注意力层（`token_req_indices`）都假定每个请求验证一样多的 token。
[#13](https://github.com/a775828b-dot/xtier/pull/13)（cyrus-xc）补上了这两处；这个 PR 的控制台部分没有合并。

> 同日晚上跟进 SGLang main 27bc1a0（v0.1.2）以后，这两处由 SGLang 自己做（按布局的 `qo_indptr`，CUDA 图回放时会刷新），#13 的这两个钩子
> 已删，只留校准预算那一处，见[跟进报告](2026-11-06-sglang-27bc1a0.md)。下面是合并 #13 时的记录。

## 改了什么

| 文件 | 内容 |
|---|---|
| `xtier/packs/deepseek_v4/ragged_verify.py`（新） | compact 验证时：Engram 按每个请求自己已提交的历史哈希（extend 的不等长路径），不写回历史；`token_req_indices`（`dsv41_sparse`、`deepseek_v4_backend`）按请求起点映射，不再按固定宽度重复；校准时强制的预算不再被"成本表未初始化就全量验证"覆盖 |
| `xtier/packs/deepseek_v4/__init__.py` | 注册这四个钩子；static 下走原来的代码 |
| `configs/ds41-pro5000-nps4/run-ds-xtier.sh` | `SGLANG_RAGGED_VERIFY_MODE` 可以从外面设，默认仍是 `static` |
| `tests/engram_ragged_check.py`（新） | 打包的哈希和逐个请求单独算的相同（不等长、序列起点、图片 token、填充），已提交的历史不变；CPU 和 CUDA 都测 |
| `bench/copy_test.py`、`bench/agent_bench.py` | 单个请求出错时记下来、不中断，最后非零退出；agent_bench 记录 HTTP 失败，`--repo` 是 xtier 布局时也能跑 |

## 测试

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
- 没有收益的原因：SGLang 的规划器按两步前的置信度定验证长度（`lag=2`）；成本表只有一档（`m_probes` 只有 0），没有长度维度。
  验证的 token 少了，接受长度跟着降（4.50 → 4.07），每步省下的时间抵不过。

## 结论

默认仍是 static。上面是这台测试机上的结果，其他机器（显卡、CPU、内存带宽不同）请自行测试：compact 现在能跑对，用
`SGLANG_RAGGED_VERIFY_MODE=compact` 启动 DS 即可。

10 月在旧管线（lsglang_ds4.1）上试过 compact：单请求会改错字，已放弃（当时两请求同时 decode 快 13.9%，单请求 +1–3%）。
这次在官方 SGLang main 上没再出现改错字。
