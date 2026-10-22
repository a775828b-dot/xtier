# XTier

**Cross-tier CPU/GPU hybrid MoE inference for [SGLang](https://github.com/sgl-project/sglang).**
XTier is a plugin for upstream SGLang (tracking `main`): MoE experts live across GPU memory, host DRAM and NVMe, are placed by
how often they are used and computed where they are. It wraps SGLang's standard `FusedMoE` quantization methods, so any MoE model
that uses the standard layer can run in hybrid mode without model-specific code; optional per-model acceleration packs add
model-specific optimizations. **Status: early development, not usable yet** — see [docs/PLAN.md](docs/PLAN.md).

---

XTier 是官方 SGLang 的插件（跟 main 分支），做 CPU/GPU 分层 MoE 推理：专家分布在显存、内存和 NVMe 三层，按热度放置、按需调度。

- **不改 SGLang 源码**：只用官方插件框架的钩子。
- **与模型无关**：混合推理包在官方 `FusedMoE` 的量化方法外面，用标准 MoE 层的模型装上就能跑。
- **加速包可选**：某个模型独有的优化放在 `xtier/packs/` 下，按模型架构自动启用。

## 结构

```
xtier/
├── backend/   混合 MoE 后端：包装官方量化方法，负责权重分流、decode 调度、prefill GPU 路径、CUDA 图兼容
├── cache/     全局专家缓存：所有层共用显存预算，按热度和收益放置，在线换入换出
├── planner/   自动规划器：根据硬件和路由统计算出缓存、分块、线程，目标是 CPU 和 GPU 同时算完
├── engine/    CPU 引擎的 Python 绑定（C++ 源码在 csrc/）
└── packs/     加速包：deepseek_v4、qwen4_exp、glm5_next
csrc/          CPU 引擎：AVX-512 / AVX2 专家内核、NUMA 线程池、CPU 与 GPU 之间的信箱通信
configs/       每个"模型 × 硬件"一份声明式配置
tests/         单元测试
bench/         回归验收（速度、PPL、照抄测试、长上下文召回）
```

## 计划支持的模型

| 模型 | 官方模块 | 状态 |
|---|---|---|
| DeepSeek V4.1 Flash 1017 | `deepseek_v4` | 计划中 |
| Qwen3.8 Flash Next | `qwen4_exp` | 开发中：官方 main 上跑通，速度与精度接近生产（[报告](docs/reports/2026-10-23-step2-qwen.md)） |
| GLM-5.3 Flash | `glm5_next` | 开发中：官方 main 上跑通（不开投机），PPL 优于生产（[报告](docs/reports/2026-10-23-step2-qwen.md)） |

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
