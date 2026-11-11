# 支持情况：实际验证过什么

用来判断自己的机器能不能跑。这里只写真正跑过的，没跑过的写“未验证”。[回到文档目录](README.md)

几个词的意思：

| 词 | 意思 |
|---|---|
| **整模型** | 真卡上跑完整的模型：起服务、照抄、PPL |
| **截层** | 真卡上只跑前几层，只比数值，输出没有意义 |
| **单层** | 真卡上只测一层和内核 |
| **模拟** | 不是这张卡：在 RTX PRO 5000 上模拟它的架构（`tools/arch_check.py --emulate`）或限制显存和内存。预编译库和 C++ 编译器还是本机的，真卡上可能还有问题 |
| **未验证** | 没跑过，能不能跑都不确定 |

## 单卡

| 显卡 | Qwen3.8 Flash Next | GLM-5.3 Flash | DeepSeek V4.1 Flash 1017 | 说明 |
|---|---|---|---|---|
| RTX PRO 5000 72 GB（SM120） | 整模型 | 整模型 | 整模型 | 开发和测速平台，每个版本都回归 |
| RTX 4090 24 GB（SM89） | 整模型（含 MTP） | 前 8 层截层；整模型只模拟 | 第 20 层单层；整模型只模拟 | GLM 要关投机解码；DeepSeek 还没有 24 GB 卡的配置 |
| RTX 3090 24 GB（SM86） | 整模型（含 MTP） | 前 8 层截层（BF16 KV）；整模型只模拟 | 第 20 层单层；整模型只模拟 | FP8 全部软件解码；GLM 要关投机解码；DeepSeek 同上 |
| RTX 5090 32 GB（SM120） | 未验证 | 未验证 | 模拟（32 GB 显存 + 128 GB 内存，[报告](reports/2026-10-27-ds41-minimal.md)） | 和 RTX PRO 5000 同架构，真卡没跑过 |
| 其他显卡 | 未验证 | 未验证 | 未验证 | 架构低于 SM86 不支持 |

上表 DeepSeek 在 RTX 30 / 40 系的结果都早于 SGLang 6162b07。v0.1.0 在这两代卡上第一个请求会崩（[#10](https://github.com/a775828b-dot/xtier/pull/10)），v0.1.1 已修；
修复后只在 RTX PRO 5000 上模拟 SM86 跑过整模型（照抄 0 错，NLL 1.197），真卡结果要等作者确认。

MiniMax-M3.1-Flash 只在 RTX PRO 5000 上跑过整模型，用 SGLang 的 M3.1 分支，2026-10-26 起暂停开发，新版本没有再测。

## 多卡

| 模式 | 硬件 | Qwen3.8 Flash Next | GLM-5.3 Flash | DeepSeek V4.1 Flash 1017 |
|---|---|---|---|---|
| 张量并行 `TP=2` | 2× RTX 3090 | 整模型 | 前 8 层截层 | 只做过 CPU 模拟 |
| 张量并行 `TP=2` | 2× RTX 4090 | 整模型 | 未验证 | 未验证 |
| `TP=2`，两张卡显存不同 | 模拟：2× RTX 4090，一张限制到 16 GiB | 模拟 | 未验证 | 未验证 |
| 第二张卡当热专家池 | 模拟：2× RTX 4090，限制显存、拖慢第二张卡 | 模拟 | 单层 + 8 层截层（只确认能跑） | 单层 |
| 第二张卡当热专家池 | **真实的异构卡**（如 PRO 5000 + PRO 4500、5090 + 5070 Ti） | 未验证 | 未验证 | 未验证 |
| `TP=2` | 2× RTX 5070 Ti + Zen2 | 未验证 | 配置已准备（[configs](../configs/glm53-2x5070ti-zen2/README.md)），未验证 | 未验证 |
| `TP=4` | | 未验证 | 未验证 | 未验证 |
| 流水线并行（PP） | | 不支持 | 不支持 | 不支持 |

第二张卡当热专家池只支持 `TP=1`，不能和冷专家同用。

## CPU

| CPU | CPU 引擎 | 验证 |
|---|---|---|
| 2× EPYC 9334（Zen 4，NPS4，24 通道） | AVX-512 | 开发和测速平台 |
| 租用机的 Xeon（没有 AVX-512 BF16） | AVX2 | Qwen3.8 整模型（RTX 4090、RTX 3090 那两台），decode 明显受 CPU 限制 |
| 2× EPYC 7642（Zen 2） | AVX2 | 未验证（`tools/isa_check.sh` 只查过库里没有 AVX-512 指令） |

## 功能

都在 RTX PRO 5000 上验证。

| 功能 | 验证过的范围 | 默认 |
|---|---|---|
| 冷专家卸载 30% | 四个模型的整模型，加 15 分钟 dsh（PCIe 4.0 盘） | 关 |
| 冷专家卸载 10–50% | DeepSeek 整模型（PCIe 5.0 盘，[报告](reports/2026-11-01-cold-pcie5.md)） | 关 |
| 冷专家卸载 70% | DeepSeek，模拟 32 GB 显存 + 128 GB 内存（PCIe 4.0 盘） | 关 |
| 大模型只存一份（`tools/cold_punch.py`） | DeepSeek 副本：打洞、30% 全套测试、恢复后 sha256 一致；其他模型只跑过小模型单元测试 | 手动 |
| 快速加载（`XTIER_FAST_LOAD`） | DeepSeek、GLM、Qwen 整模型；TP 下在 2× RTX 4090 上跑过 Qwen | 关 |
| 弹性热专家槽 | DeepSeek、GLM 默认开；Qwen 实测没有收益 | 启动台按模型 |
| 规划器自动布局 | Qwen 默认用；DeepSeek、GLM 用固定槽数 | 启动台按模型 |
| 未命中专家 GPU 代算 | 三个模型能跑；本机反而变慢，弱 CPU 只模拟过（8 线程 +6–7%） | 关 |
| 投机解码 | DeepSeek DSpark、GLM EAGLE、Qwen MTP，都是整模型 | 启动台按模型 |

## v0.1.5 的测试环境

和 v0.1.2 到 v0.1.4 一样（SGLang 提交和依赖都没变）。

| 项目 | 版本 |
|---|---|
| SGLang | main `27bc1a0615`（2026-11-06） |
| Python | 3.12.3 |
| PyTorch | 2.14.1+cu130（CUDA 13.0） |
| FlashInfer | 0.7.2 |
| Triton | 3.8.0 |
| transformers | 5.19.0 |
| 其他 Python 包 | [constraints.txt](../constraints.txt) |
| 驱动 | NVIDIA 595.99.02（CUDA 13.0 要 580 以上） |
| 系统 | Ubuntu 24.04，内核 7.0，g++ 13.3 |
| 硬件 | RTX PRO 5000 72GB，2× EPYC 9334（NPS4），384 GiB DDR5 |

| 模型 | Hugging Face 仓库 | 版本 |
|---|---|---|
| DeepSeek V4.1 Flash 1017 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `1bae0e108899` |
| GLM-5.3 Flash | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | `11d73216cd63` |
| Qwen3.8 Flash Next | `lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4` | `f748827507ec` |
