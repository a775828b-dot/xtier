# XTier 文档目录

第一次用 XTier：先看[支持情况](support.md)确认能不能跑，再照[快速开始](quickstart.md)跑通第一次请求。遇到问题先看[常见问题](faq.md)。

English: [README](README.en.md) · [Quick start](quickstart.en.md)

## 我想……

| 想做什么 | 看这里 |
|---|---|
| 确认我的模型和显卡能不能跑 | [支持情况](support.md)（实际验证过什么） |
| 从空机器跑通第一次请求 | [快速开始](quickstart.md) |
| 用网页选模型、调参数 | [启动台](console.md) |
| 解决启动报错、内存或显存不够、速度不对 | [常见问题](faq.md) |
| 跑内存放不下的大模型 | [冷专家卸载到硬盘](../README.md#冷专家卸载到硬盘)、[大模型只存一份](../README.md#大模型只存一份) |
| 用两张卡 | [多卡](../README.md#多卡张量并行)、[快卡带慢卡](../README.md#异构卡快卡带慢卡第二张卡纯当热专家池) |
| 看能跑多快 | [测速](../README.md#测速)、[测速细节](benchmarks.md) |
| 看每个版本改了什么 | [版本更新](changelog.md) |

## 安装

| 内容 | 文档 |
|---|---|
| 从空机器到第一次请求：环境、模型准备、启动命令、测试请求 | [快速开始](quickstart.md)（[English](quickstart.en.md)） |
| 已经有 SGLang 环境 | [快速上手](../README.md#快速上手) |
| 测试过的版本：SGLang、Python、PyTorch、驱动、模型 | [支持情况](support.md#v012-的测试环境)、[constraints.txt](../constraints.txt) |
| 不加载模型，核对 SGLang 和 XTier 对得上 | `python tools/check_sglang.py --strict` |
| 单卡 24 GB（RTX 4090 / 3090）和约 120 GB 内存，比如租用的云主机 | [Qwen3.8 配置](../configs/qwen38-rtx4090-3090/README.md) |
| Zen2 + 双卡（已准备，还没有在这类机器上跑过） | [部署说明](deploy-zen2-dual-gpu.md) |

## 配置

**启动参数有三个入口**：启动台（最简单）、`configs/` 里的启动脚本、环境变量。`configs/` 大多数脚本里的路径是测试机的，用之前要改成自己的（见[常见问题](faq.md#启动脚本说路径不存在)）。

| 内容 | 文档 |
|---|---|
| 启动台：网页上选模型和参数、看加载进度和速度、加新模型 | [启动台](console.md) |
| 全部 `XTIER_*` 开关和默认值 | [xtier/config.py](../xtier/config.py) 开头的说明；功能各自的开关在对应模块开头（如 [cache/cold.py](../xtier/cache/cold.py)、[backend/direct.py](../xtier/backend/direct.py)） |
| 每层放几个热专家、自动规划 | [规划器](planner.md) |
| 冷专家池：盘的要求、池子大小、只存一份的流程、按盘速分段、尾部跳过 | [冷专家设计](design-cold-experts.md) |
| 大嵌入表（DeepSeek Engram、Qwen PLE）的行放 NVMe | [大嵌入表设计](design-tables-nvme.md) |

**现成的启动配置**（[configs/](../configs/)）：

| 目录 | 模型和硬件 | 说明 |
|---|---|---|
| [ds41-pro5000-nps4](../configs/ds41-pro5000-nps4/) | DeepSeek V4.1 Flash 1017，RTX PRO 5000 72GB + 2× EPYC 9334 | 测速用的配置 |
| [ds41-mini-5090sim](../configs/ds41-mini-5090sim/) | 同上，模拟 32 GB 显存 + 128 GB 内存 | [最小化运行](../README.md#最小化运行32-gb-显存--128-gb-内存跑-deepseek-v41-flash-1017) |
| [glm53-pro5000-nps4](../configs/glm53-pro5000-nps4/README.md) | GLM-5.3 Flash，RTX PRO 5000 | 测速用的配置 |
| [glm53-2x5070ti-zen2](../configs/glm53-2x5070ti-zen2/README.md) | GLM-5.3 Flash，2× RTX 5070 Ti + 2× EPYC 7642 | 已准备，还没有在这类机器上跑过 |
| [qwen38-pro5000-nps4](../configs/qwen38-pro5000-nps4/README.md) | Qwen3.8 Flash Next，RTX PRO 5000 | 测速用的配置 |
| [qwen38-rtx4090-3090](../configs/qwen38-rtx4090-3090/README.md) | Qwen3.8 Flash Next，1–2 张 RTX 4090 / 3090 | 路径和选项都走环境变量，改起来最省事 |
| [minimax-m31-pro5000-nps4](../configs/minimax-m31-pro5000-nps4/) | MiniMax-M3.1-Flash，RTX PRO 5000 | 暂停开发 |

**常用工具**（[tools/](../tools/)）：

| 工具 | 做什么 |
|---|---|
| [check_sglang.py](../tools/check_sglang.py) | 不加载模型，核对 SGLang 提交、钩子和 import |
| [build_engine.sh](../tools/build_engine.sh) | 编译 CPU 引擎和信箱库 |
| [mem_guard.sh](../tools/mem_guard.sh) | 可用内存太低时停掉服务（启动台默认带） |
| [cold_punch.py](../tools/cold_punch.py) | 专家只留在冷存储，模型文件打洞释放空间，也能写回 |
| [cold_relayout.py](../tools/cold_relayout.py) | 冷存储换布局（换盘、分段、按盘速分配），逐条读回比对，旧存储不动 |
| [sps_profile.py](../tools/sps_profile.py) | 测 DSpark 变长验证（compact）要用的成本表 |
| [plan_probe.py](../tools/plan_probe.py)、[plan_trace.py](../tools/plan_trace.py)、[plan_auto.py](../tools/plan_auto.py)、[plan_calibrate.py](../tools/plan_calibrate.py) | 规划器：探测、路由档案、出方案、校准 |

## 常见问题

[常见问题](faq.md)按症状分类，日志里的原文可以直接搜：

- [安装和启动](faq.md#安装和启动)：SGLang 版本警告、钩子没生效、第一次启动慢、每次都重新编译
- [内存和显存](faq.md#内存和显存)：服务被内存守护停掉、显存不够、模型比内存还大
- [冷专家和只存一份](faq.md#冷专家和只存一份)：选什么盘、启动台选项是灰的、打洞后启动报错
- [速度和精度](faq.md#速度和精度)：比测速表慢、温度 0 两次输出不一样

## 测试报告

**先看这三个**：[支持情况](support.md)（每个模型、显卡和多卡模式验证到什么程度）、[测速细节](benchmarks.md)（README 各速度表的测法、配置和原始数据）、[版本更新](changelog.md)（按时间列出所有改动）。

下面的报告按主题分类，每组按时间从新到旧。

**版本和升级**

| 报告 | 内容 |
|---|---|
| [2026-11-06 跟进 SGLang 27bc1a0（v0.1.2）](reports/2026-11-06-sglang-27bc1a0.md) | 依赖变化；XTier 改的三处（SM12x 的 DeepSeek 索引器、Engram、变长验证）；三个模型的回归 |
| [2026-11-01 代码审计和修复](reports/2026-11-01-audit-fixes.md) | v0.1.0：修了什么、查明了什么、试过但没采用什么，三个模型的回归结果 |
| [2026-10-30 跟进 SGLang 6162b07](reports/2026-10-30-sglang-6162b07.md) | 升级 SGLang 时改了什么、怎么验证 |

**显卡和多卡**

| 报告 | 内容 |
|---|---|
| [2026-10-31 第二张卡纯当热专家池](reports/2026-10-31-second-gpu-pool.md) | 显存或速度不同的两张卡 |
| [2026-10-30 双卡张量并行](reports/2026-10-30-tp.md) | `TP=2`，双 RTX 3090 实测 |
| [2026-10-29 RTX 30 系](reports/2026-10-29-rtx30.md) | SM86，RTX 3090 实测 |
| [2026-10-29 RTX 40 系](reports/2026-10-29-rtx40.md) | SM89，RTX 4090 实测 |

**内存、硬盘和加载**

| 报告 | 内容 |
|---|---|
| [2026-11-06 尾部跳过不再只管冷专家](reports/2026-11-06-tail-skip-cpu.md) | CPU 算的专家都可以跳，写法 `5/6`；DeepSeek 不卸载时每步 −2.3% / −5.3% |
| [2026-11-06 冷存储分段和尾部跳过](reports/2026-11-06-cold-units-tail-skip.md) | 盘数不是 4 块、盘速不一样时按盘速分配；不读排名靠后的冷专家（旧规则） |
| [2026-11-01 PCIe 5.0 冷专家池和只存一份](reports/2026-11-01-cold-pcie5.md) | 卸载 10–50% 实测、打洞和恢复 |
| [2026-10-27 最小化运行](reports/2026-10-27-ds41-minimal.md) | 模拟 32 GB 显存 + 128 GB 内存跑 DeepSeek |
| [2026-10-27 快速加载](reports/2026-10-27-fast-load.md) | 专家直接从模型文件或冷存储读进引擎 |
| [2026-10-26 冷专家池](reports/2026-10-26-cold-experts.md) | 第一版，四类盘的对比 |

**热专家和调度**

| 报告 | 内容 |
|---|---|
| [2026-10-28 未命中专家 GPU 代算](reports/2026-10-28-decode-miss-gpu.md) | 可选，默认关 |
| [2026-10-26 弹性热专家槽](reports/2026-10-26-elastic-cache.md) | prefill 用不完的显存，decode 时给热专家 |
| [2026-10-25 提示词预热热专家池](reports/2026-10-25-prompt-warmup.md) | 实测没有明显收益，默认关 |
| [2026-10-24 自动规划器](reports/2026-10-24-planner.md) | 按显存预算自动分配每层的槽数 |

**各模型的优化记录**

| 报告 | 内容 |
|---|---|
| [2026-11-06 DeepSeek 变长验证（compact）](reports/2026-11-06-ds-compact-verify.md) | 来自 #13：要配本机实测的成本表；照抄 0 错，两个请求同时快 17%，默认仍是 static |
| [2026-10-24 MiniMax 512K](reports/2026-10-24-minimax-512k.md) | NVFP4 KV、DSpark 按置信度截断（已暂停开发） |
| [2026-10-24 Qwen NVFP4 KV、DeepSeek agent 实测](reports/2026-10-24-qwen-kv-ds-agent.md) | 省下的显存怎么用 |
| [2026-10-24 DeepSeek decode、分词和 MiniMax](reports/2026-10-24-ds-woa-minimax.md) | FP8 wo_a、分页索引器、deepseek-recipe 分词 |
| [2026-10-23 通用投机解码、Qwen NVFP4 KV](reports/2026-10-23-speculative-fp4kv.md) | GLM 拒绝采样、DeepSeek 的剩余差距 |
| [2026-10-23 DeepSeek Engram 提前查表](reports/2026-10-23-ds-engram-prefetch.md) | 提前查表、草稿词表头 FP8 |
| [2026-10-23 GLM / Qwen 512K prefill](reports/2026-10-23-prefill.md) | 长上下文 prefill |
| [2026-10-23 DeepSeek 上官方 main](reports/2026-10-23-step3-ds-tables.md) | 大表放 NVMe、Zen2 和双卡的准备 |
| [2026-10-23 Qwen、GLM 上官方 main](reports/2026-10-23-step2-qwen.md) | 最早的移植记录 |

## 设计和开发

想改代码再看这部分，只是使用可以跳过。

| 内容 | 文档 |
|---|---|
| 工作计划和路线 | [PLAN.md](PLAN.md) |
| 发版前要跑哪些测试 | [发布前检查清单](release-checklist.md) |
| 从生产管线移植时的盘点 | [inventory.md](inventory.md) |
| 代码结构 | [README 的结构一节](../README.md#结构) |
| 验收脚本：PPL、照抄、验证步、并发、prefill 延迟 | [bench/](../bench/README.md) |
| 单层测试 | [tests/](../tests/README.md) |
| 不需要 GPU 的检查（CI 每次推送都跑） | `python tools/repo_check.py`、`tests/console_check.py`、`tests/planner_check.py` |
