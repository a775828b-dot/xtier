# 代码审计和修复（2026-11-01，v0.1.0）

对 `7008fc7` 做了一次全面代码审计：`csrc/` 和 `xtier/` 逐行读（已暂停的 MiniMax 包除外），加静态检查、单元测试、采样和 A/B。这里是修了什么、查明了什么、
试过但没采用什么。冷专家搬盘和“只存一份”见[另一份报告](2026-11-01-cold-pcie5.md)。

## 修复

| 问题 | 不修的后果 | 现在 |
|---|---|---|
| DeepSeek 的 CUTLASS 内核在第一个冷 prefill 请求里才编译（换机器、升级 FlashInfer 或换 ninja 后） | 编译要约 60 GiB，内存不够时守护进程杀掉服务，重启也过不去（复现 4 次） | 加载权重之前先编译（`xtier/prebuild.py`，`XTIER_PREBUILD=0` 关），用服务器的全部 CPU、按可用内存限制并发；从零编译 365 秒，内存余量没有低于 244 GiB |
| 换槽序号到 2^31 回绕后不再摘除被换出的专家（DeepSeek 满载约 25 天） | 静默算错，然后 GPU 非法访问（复现） | `_commit` 和 `mailbox.cpp` 按回绕差值计算；`tests/pool_wrap_check.py`，`XTIER_DEBUG_SEQ_START` 可把序号直接拨到回绕前做整机测试 |
| split prefill 的 CPU 线程报错被吞 | 未初始化的结果加进输出 | 等线程结束后抛出（`tests/prefill_errors_check.py`） |
| 冷专家写盘线程报错被吞，清单照写 | 下次快速加载读到坏权重 | 写失败不写清单，启动报错 |
| 混合 prefill 的 `join` 不在 `finally` 里 | GPU 部分 OOM 后，重试撞上还在跑的线程 | 改 `try/finally` |
| 专家张量没加载全只打警告 | 用未初始化的权重服务 | 启动报错 |
| 弹性热专家 OOM 后重跑整个 forward，不是幂等的 | 故障注入：重试后提示的 logprob 差 GLM 0.3、Qwen 0.17 nats，DeepSeek 没有差别（历史 138 份日志里没有真实发生过） | 记错误日志，空闲时清前缀缓存 |
| SGLang 改名调度入口或钩子没生效时，加速项静默消失 | 只表现为变慢，甚至悄悄退回官方路径 | 启动时报错；`tools/check_sglang.py` 也核对入口（`tests/startup_checks_check.py`） |
| `noexcept` 回调里 `std::stol` 可能抛异常 | 环境变量写错时进程直接 `terminate` | 改 `std::strtol` |
| GLM 脚本默认 60 槽，启动台是 58 | 不经启动台直接跑时显存贴边 | 改成 58 |
| GLM 的位置张量按长度缓存、不淘汰（RTX 30 / 40 系路径） | 显存慢慢被吃掉 | 每张卡只留最长的一份 |
| 未知的 `XTIER_*` 没有提示 | 拼错了静默不生效 | 启动时警告（`xtier/env_names.txt`，`tools/repo_check.py --write` 生成） |
| 测试：两个脚本过期，13 个脚本打印 FAIL 时退出码还是 0，没有 CI | 回归靠人看输出 | 都修了；`tools/repo_check.py` + GitHub Actions 做不需要 GPU 的基础检查 |
| 过期接口和死代码：`xtier_mailbox.h` 和实现不一致，`libcudart.so.13` 写死，`xtier_create2`、`link_prefill_ring`、`tools/port_*` | CUDA 12 的 torch 上 prefill 加载失败；增加阅读负担 | 头文件同步并由实现 include，CUDA 库复用 `tables/lookup.py` 的查找；死代码删除 |
| `_reduce` 和 `_copy_ids` 的 Triton 内核按尺寸重新编译 | 新尺寸首字慢约 85 ms，磁盘缓存无限增长 | 尺寸改成运行时参数 |

## 查明的问题

- **同一个提示词重算，logprob 会摆动**（偶发翻 1 个 token 的来源）：不是 bug。top-k 内核（SGLang `fast_topk`）输出的集合相同，但顺序取决于原子竞争，FP32 累加的舍入跟着变，
  逐层放大。把索引排序后 GLM 8192 个 token 逐位相同；Qwen 关掉换槽后 1500 个 token 逐位相同。PPL 均值不变，判断精度看均值。
- **decode 时 GPU 几乎不空闲**：replay 级时间线上 GLM 每步约 0.5 ms（2%），DeepSeek 0.18 ms。overlap 调度下 Python 和 GPU 是并行的，不在关键路径上，优化主线程 Python 没有收益。

## 试过，没有采用

每组都是新起服务、64K 上下文、按 verify 的 token 数分组比 GPU 时间（每步受 token 数的影响比换槽设置大）。

| 改动 | 结果 |
|---|---|
| GLM 每 2 步换 32 / 64 个（现在 4 步 16 个） | 3 token：默认 22.58 / 22.36 ms，32 个 22.51 / 22.52，64 个 22.43；4 token 24.77 / 24.74，25.10 / 24.66，25.31；差别在波动内 |
| DeepSeek `XTIER_CACHE_MIN_GAIN=3` | 25.75 / 26.33 ms，默认 26.69 / 26.40 / 25.43 ms，差别在波动内 |
| DeepSeek `XTIER_CACHE_SWAPS=64` | 26.03 ms，同上 |
| 提示词预热热专家（`XTIER_CACHE_PROMPT=1`） | 短回复每步 DeepSeek 29.78 / 30.11 ms 对 29.26 / 29.32，GLM 29.18 对 28.52；首字基本不变 |
| 启动预热覆盖完整分块 | 第一个 64K 请求比第二个多 DeepSeek 5.6、GLM 7.2、Qwen 2.6 秒，只影响启动后第一次，预热只是把这段时间挪到启动里 |
| 降低弹性检查频率、透明大页 | 每步没有变化 |

## 回归

三个模型都用启动台默认配置，RTX PRO 5000 72 GB + 2×EPYC 9334。

| 模型 | 照抄（单路 / 两路） | 序号回绕后照抄 | PPL NLL（上次 → 这次） | 64K prefill | 512K prefill | dsh 15 分钟 |
|---|---|---|---|---:|---:|---|
| DeepSeek V4.1 Flash 1017 | 0 错 / 0 错 | 2^31、2^32 都 0 错 | 1.2001 → 1.2002 | 4,894 tok/s | 4,593 tok/s | 173 个请求 0 失败，召回 3/3 |
| GLM-5.3 Flash | 各 1 段不同（见下） | 2^31 0 错 | 1.3859 → 1.3855 | 2,600 tok/s | 2,495 tok/s | 203 个请求 0 失败，召回 3/3 |
| Qwen3.8 Flash Next | 0 错 / 0 错 | 2^31 0 错 | 1.2471 → 1.2485 | 5,237 tok/s | 4,794 tok/s | 404 个请求 0 失败，召回 3/3 |

GLM 的照抄：回归时单路、两路都是第 2 段差一处（漏一个词、少一层缩进）。这一段里有几个分数很接近的 token，会随 top-k 的舍入翻转（见上面“查明的问题”），
以前的版本也出现过一模一样的错误。为了确认不是这次改动造成的，新代码和 `7008fc7` 各新起一次服务、重复照抄 16 次（单路、两路各 8 次，一半在 64K 请求之后）：

| 代码 | 重复 16 次 | 算上回归和更早的记录 |
|---|---|---|
| 这个版本 | 0 次出错 | 19 次里 2 次，都在第 2 段 |
| `7008fc7` 及更早 | 1 次出错，也在第 2 段 | 21 次里 2 次，都在第 2 段 |

出错率没有变化，PPL 也一致。

单元测试全部通过（发布前要跑的清单见[这里](../release-checklist.md)），日志没有 Traceback 和 OOM。
