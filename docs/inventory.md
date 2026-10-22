# 盘点（第 0 步，v1）

对象：三条生产管线和它们共用的 CPU 引擎（lkqmoe 0.5.6，源码提交 `93edcc1`），对照官方 SGLang main `8cd3c494`（2026-10-22）。
落点：**后端**（`xtier/backend`）、**缓存**（`xtier/cache`）、**规划器**（`xtier/planner`）、**引擎**（`csrc/` + `xtier/engine`）、
**包**（`xtier/packs/<模块>`）、**官方**（官方 main 已有，删掉）、**删**（实验失败、没有收益或只为兼容旧框架）。
"官方"一栏来自按关键词检索官方 main，标"待核"的要在对应步骤里逐项确认。

## 关键发现

1. **官方 main 已支持三个模型**：`deepseek_v4`（含 DSpark、Engram）、`qwen4_exp`（含 MTP、QSA、PLE）、`glm5_next`（含 NextN、DSA）。
   三个模型都用官方 `FusedMoE`，权重创建、加载和计算都经过量化方法对象，混合推理可以包在这一层，与模型无关。
   例外：`qwen4_exp` 有一条专用 decode 路径 `qwen4_decode_moe`，要单独接。
2. **官方有插件框架**：`sglang.srt.plugins`（entry_points）+ `HookRegistry`（before / after / around / replace，按完整路径挂到任意函数、方法、类），
   另有外部模型包 `SGLANG_EXTERNAL_MODEL_PACKAGE`。插件里抛的异常会被记日志后吞掉，XTier 要自检。
3. **官方没有 CPU 专家卸载**：检索 `srt/` 只在多模态生成里有层级卸载，MoE 专家放 CPU 计算这件事要全部由 XTier 提供。
4. **生产代码散落**：lkqmoe 的 git 源码之外，三个部署目录里还各有一批不在版本控制里的模块（见下表"仅部署"），这次全部收进本仓库。
5. **旧框架分叉越来越大**：三条管线分别基于 Lsglang 的三个不同提交；Lsglang 最新版相对它自称的官方基线已经差 554 个文件，不再追它。

## CPU 引擎（lkqmoe 0.5.6）拆分

| 现在 | 内容 | 落点 |
|---|---|---|
| `src/lkqmoe.cpp` | 与模型无关的 NVFP4 / MXFP4 CPU MoE：H、I 运行时给定（常见形状模板特化），AVX-512 BF16(+VBMI) 与 AVX2/FMA 两套内核，单行 Down（R4=1），NUMA 拓扑读 sysfs，每节点工作线程池 | 引擎：拆成 `csrc/kernels/`（内核）与 `csrc/runtime/`（拓扑、线程池、计划） |
| `src/dispatch.cpp` | 一个入口库按 CPU 指令集 dlopen avx512 / avx2 实现 | 引擎 |
| `src/cuda_doorbell.cpp` | 映射内存信箱 + 调度线程（流内存操作，CUDA 图可捕获）；热专家池命令环与换入换出（PoolManager） | 引擎（信箱）+ 缓存（池管理改为全局缓存） |
| `src/cuda_bridge.cpp` | 旧的主机回调桥（只做 CPU 计算） | 删（信箱已取代），确认无调用后删除 |
| `python/lkqmoe/__init__.py` | 计划（NativePlan）、执行器、ctypes 绑定 | 引擎（`xtier/engine`） |
| `zerocopy.py` | 信箱的 GPU 侧打包 / 解包内核 | 后端 |
| `pool.py` | LKQstrata：每层 GPU 热专家槽，Triton 内核 `_pack/_gate_up/_down/_combine/_unpack`，按画像装填、衰减计数、按收益换入 | 缓存（改为全局预算，GPU 专家内核保留并泛化） |
| `prefill.py` | GPU prefill：专家分组 GEMM（0.5.6 起 MXFP4 × MXFP8 走 FP4 张量核）、上传与解包流水、跨层预取 | 后端（prefill 路径） |
| `tp.py` | 每个 TP rank 一个实例，按 rank 切专家中间维、分配 CPU | 后端 |
| `integration.py`、`standalone.py`、`sitecustomize.py` | 冒充 lk_moe 公开接口接入 Lsglang；启动钩子 | 删（改为官方插件入口 `xtier/plugin.py`） |
| `gpu/mtp_quant.py`（仅部署） | 序列化量化的 MTP 草稿保持量化 | 包（qwen4_exp / glm5_next），待核官方是否已支持 |
| `gpu/router_dependency.py`（仅部署） | 路由偏置的 PDL 提前启动读到旧值的规避 | 包，待核官方是否已修 |
| `gpu/resident_w4a16.py`、`gpu/w4a16_resident.py`（仅部署） | GPU 常驻 NVFP4 专家用 BF16 激活（W4A16） | 缓存（全局缓存的 GPU 专家内核就是 W4A16） |
| `gpu/fp8_weights.py`（仅部署） | 选定的 BF16 线性层存成 FP8 / NVFP4 | 包（glm5_next） |
| `gpu/glm53_mhc.py`（仅部署） | GLM-5.3 在 SM120 上的全融合 mHC pre | 包（glm5_next），待核官方 |
| `gpu/indexer_chunk.py`（仅部署） | DSA kpool 索引器 prefill 按查询分块 | 包（glm5_next），待核官方 |
| `gpu/glm53_cpu_image.py`（仅部署） | GLM-5.3 图片在 CPU 解码 | 包（glm5_next） |
| `gpu/ds41_dense_indexer.py`（仅部署） | 长上下文 prefill 的 FP4 稀疏索引器按行分块 | 包（deepseek_v4），待核官方 `v41_indexer` |
| `gpu/ds41_wo_a.py`（仅部署） | `wo_a` 低秩分组 einsum（Triton，可选 FP8 权重） | 包（deepseek_v4） |
| `gpu/ds41_split_zero.py`（仅部署） | SM120 FlashMLA 分页暂存区保留页清零 | 包（deepseek_v4），待核官方 `_split_kv_pages_to_64` |
| `gpu/fp8_configs.py`（仅部署） | Triton W8A8 块 FP8 GEMM 的调优配置 | 包（deepseek_v4），能通用的提交官方 |
| `gpu/ds41_c128.py`、`gpu/ds41_trace.py`（仅部署） | 测量与调试 | 删（需要时放 `bench/`） |
| `gpu/hc_mix.py`、`fp8_projection.py`、`sglang_hc.py`、`sglang_projection.py`、`startup.py` | 实验算子，完整模型质量验证没过 | 删 |
| `gpu/small_gemm.py` | 小 M BF16 GEMM，实测无收益 | 删 |
| `gpu/deterministic_marlin.py` | Marlin 稳定顺序 + FP32 split-K（可选） | 删（全局缓存后不再走 Marlin 常驻路径），确认无依赖后删除 |

## DeepSeek V4.1 Flash 1017（lsglang_ds4.1）

| 功能 | 现在的位置 | 官方 main | 落点 |
|---|---|---|---|
| CPU 专家 + 每层 50 个热专家槽（自适应） | lkqmoe + LKQstrata | 无 | 后端 + 缓存 |
| 加载时就把 CPU 层交给 CPU 引擎（不先占显存） | `adapter/early_lk_moe.py` | 无 | 后端（权重加载分流） |
| GPU prefill FP4 张量核、6 块跨层预取、分块 22528 | lkqmoe prefill + 启动参数 | 无 | 后端 + 规划器 |
| Engram 行在 NVMe（O_DIRECT、行缓存、scale 常驻、提前查表与并行读，多卡已修） | `adapter/engram_backend.py`、`row_store.cpp` | 只有主机表 | 包（deepseek_v4） |
| DSpark 固定验证 6 token | 补丁 + `dspark_worker_v2` 覆盖 | DSpark 已有 | 待核差异，余下的进包 |
| DSpark 验证选 c128 图免 D2H 同步 | 补丁（`decode_cuda_graph_runner`） | 未见 `_resolve_c128_width` | 待核（官方可能已改写） |
| DSpark 草稿词表头 FP8 | 补丁 | 无 | 包 |
| prefill 稀疏索引器走 DeepGEMM FP4、按行分块 | overlay `deepseek_v4_backend` + `gpu/ds41_dense_indexer` | 有 `fp8_fp4_mqa_logits`（`v41_indexer`） | 待核，余下的进包 |
| 官方 1017 会话编码、控制 token 防注入（10-21） | 补丁（`encoding_dsv41` 等） | 有 `encoding_dsv41` | 待核差异 |
| MXFP4 1017 检查点适配 | `adapter/ds1017_quant.py` | MXFP4 广泛支持 | 待核，余下的进包 |
| 调度器、responses、工具调用解析的修改 | overlay（整文件替换） | 已大改 | 逐项核对，余下的做成钩子进包 |
| SWA 尾部有界重放 | 启动参数 | 有 | 官方 |
| OpenAI developer 角色 | 补丁 | 有 | 官方，待核 |

## Qwen3.8 Flash Next（lsglang_q3.8fn）

| 功能 | 现在的位置 | 官方 main | 落点 |
|---|---|---|---|
| 模型、MTP、QSA、超连接、PLE（PR #36497 的内容，补丁 81 个文件的大头） | 补丁 | 有 | 官方 |
| CPU 专家 + 25 个常驻层 + 23 层 × 155 个热专家槽 | lkqmoe + LKQstrata + 启动参数 | 无 | 后端 + 缓存（全局缓存取代"常驻层 + 每层槽"）+ 规划器 |
| 常驻 NVFP4 层用 Marlin W4A16 | 补丁（`marlin_utils_fp4` 等）+ `SGLANG_W4A16_RESIDENT_MARLIN` | 无 | 缓存（GPU 专家内核） |
| hc_mix 重叠内核（10-22） | 补丁（`hc_mix_triton`） | 原内核已移到 `kernels/ops/gemm/hc_mix.py` | 包（qwen4_exp），考虑提交官方 |
| NVFP4 KV | 补丁（`fp4_kv_cache_quant_method`） | 待核 | 待核 |
| MTP 草稿保持 NVFP4、草稿热词表 | `gpu/mtp_quant` + 启动参数 | 热词表有，量化待核 | 包 / 官方 |
| validated split-K | 补丁 + 覆盖层 | 无 | 删（只在 sm100 生效），确认后删除 |
| 内存池、KV 配置、EAGLE worker、调度批次等改动 | 补丁 | 已大改 | 逐项核对 |

## GLM-5.3 Flash（lsglang_glm5.3）

| 功能 | 现在的位置 | 官方 main | 落点 |
|---|---|---|---|
| 模型（PR #36507）、NextN、DSA | 作者分支 + 补丁 | 有 | 官方 |
| CPU 专家（42 层全走 CPU 引擎）+ 每层 60 个热专家槽 | lkqmoe + LKQstrata | 无 | 后端 + 缓存 |
| EAGLE 自适应步数、无损拒绝采样、截断验证 | 补丁（eagle worker / utils / info / draft 图） | 有拒绝采样；自适应待核 | 待核，余下的进包 |
| HiCache 与 DSA 的修复（usrlocalben 分支） | 补丁（mem_cache） | 有 HiSparse | 待核 |
| 全融合 mHC pre（SM120） | `gpu/glm53_mhc` + 补丁 | 待核 | 包 |
| BF16 权重转 FP8 / NVFP4 | `gpu/fp8_weights` | 无 | 包 |
| DSA 索引器 prefill 分块、图片 CPU 解码 | `gpu/indexer_chunk`、`gpu/glm53_cpu_image` | 待核 | 包 |

## 基线

见 `bench/baselines/`：三条管线 10-22 的实测（速度、并发、PPL、照抄测试、长上下文召回）和测法。
