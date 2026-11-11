# 发布前检查清单

每次发版或跟进 SGLang 时用。[回到文档目录](README.md)

## 1. 自动检查（CI）

每次推送和 PR 由 [GitHub Actions](../.github/workflows/check.yml) 跑，不需要 GPU：

```bash
python tools/repo_check.py        # 语法、换行符、XTIER_* 名单、原生库接口、FAIL 要非零退出
python tests/console_check.py     # 启动台：所有模型的所有选项组合都能生成启动命令
python tests/planner_check.py     # 规划器（只需要 numpy）
```

新增 `XTIER_*` 开关后运行 `python tools/repo_check.py --write` 更新名单。

## 2. 单层测试（一张卡 + 模型）

测试脚本要像启动脚本一样设好 CUDA 环境，否则 FlashInfer 会用系统里旧的 nvcc：

```bash
export CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13 PATH=$CUDA_HOME/bin:$PATH
```

看结果要同时看退出码和有没有 `FAIL`。

| 内容 | 命令（都在 `tests/` 下） |
|---|---|
| 不需要模型 | `fp4_decode_check` `fp8_decode_check` `fp8_encode_check` `qsa_decode_check` `qsa_fp4_check` `qsa_fp8_check` `qsa_prefill_check` `ds_sparse_mla_check` `ds_fp8_soft_check` `hc_mix_check` `ple_blocks_check` `swiglu_oai_check` `spec_truncate_check` `glm_indexer_torch_check` `draft_bf16_check` `draft_lmhead_fp8_check` `ds_mxfp8_rows_check` `fp8_weights_check` `elastic_check` `elastic_need_check` `elastic_release_check` `elastic_chunk_check` `elastic_critical_check` `tables_check` `table_lookup_check` `ds_wo_a_check` `ds_w8a16_check` `direct_load_check` `pool_wrap_check` `prefill_errors_check` `startup_checks_check` |
| 每个模型（`--preset ds41_1017`、`glm53`、`qwen38`） | `formats_check`；`mailbox_check --threads N`；`cache_check --threads N --slots 64`；`hybrid_prefill_check --threads N --slots 64 --lengths 100,500,1000`；`resident_prefill_check`；`miss_gpu_check --threads N`；DeepSeek 另跑 `prefill_fp4_retry_check`（SM120 的 FP4 prefill 失败后重跑）、`prefill_cpu_share_check --threads N`（CPU 分担 prefill） |
| CPU 模拟多卡 | `tp_check --preset glm53 --tp 2` |
| 冷专家（往盘上写测试数据） | `cold_punch_check DIR`（再用 `XTIER_COLD_UNITS=3` 跑一次）；`cold_sole_copy_check DIR`；`cold_passes_check DIR1 DIR2`；`cold_prefill_check --store DIR1 DIR2` |
| 两张卡 | `second_gpu_check` |

路径和模型位置写在脚本开头，换机器要改。`tools/check_sglang.py --strict` 也要通过：除了钩子目标和 import 都在，还比对被钩函数的源码指纹（[tools/hook_sources.json](../tools/hook_sources.json)，不看注释、文档字符串和排版）和 XTier 带默认值读取的上游属性。跟进 SGLang 后，逐个看完它列出的改动，再用 `--write-sources` 记下新提交的指纹（`--emulate sm86` 再记一次 SM86 才挂的钩子）。

CPU 引擎改了的话，再用线程检查（TSan）和内存检查（ASan、UBSan）构建引擎跑一遍并发驱动程序，0 报告：`tests/engine_san.sh thread|address DIR`（DIR 在支持 O_DIRECT 的盘上）。

## 3. 整模型回归（每个模型）

三个模型都用启动台默认配置，每项都要通过：

| 测试 | 命令 | 通过标准 |
|---|---|---|
| 照抄，单路和两路 | `bench/copy_test.py --port P --model M --chunks 8 --concurrency 1`（再跑 `2`，DeepSeek 再跑 `4`；Qwen 加 `--ignore-first-indent`） | 0 错。温度 0 偶尔翻 1 个 token（[原因](faq.md#温度-0两次输出不完全一样)），有错要重复多次，和上个版本比出错率 |
| 精度 | `bench/ppl_eval.py --port P --tokenizer T --segments 6 --tokens 4096` | NLL 和上个版本的差在两次启动之间的波动内（约 0.01） |
| 长上下文 | `bench/context_speed.py --port P --model M --tokenizer T --contexts 65536,65536,520000 --output 1024` | 不 OOM，速度和上个版本接近 |
| 真实使用 | dsh 连续 15 分钟（填上下文、召回、写代码） | 请求全部成功，召回正确 |
| 换槽序号回绕 | `XTIER_DEBUG_SEQ_START=2147478648` 起服务，再跑一次照抄（DeepSeek 还要测 `4294962296`） | 0 错，日志没有 GPU 错误 |
| 异常客户端 | `bench/abuse_test.py --port P --model M --out F`：中途断开、超长、坏参数、并发，约 5 分钟 | 全部通过，之后照抄 0 错 |
| 日志 | `grep -E "Traceback\|OutOfMemory\|FATAL\|Scheduler hit\|does not know"` | 没有 |

改了冷专家的话，还要在 GLM（NVFP4）上跑开 / 关冷专家的 PPL：DeepSeek 是 MXFP4，全局缩放都是 1，会掩盖缩放类错误。

## 4. 多卡和其他显卡

改到相关代码才需要，要租用真卡。结果更新到[支持情况](support.md)。

- 2× RTX 3090 或 4090：`TP=2` 和第二张卡当热专家池，Qwen3.8 照抄和速度（[配置](../configs/qwen38-rtx4090-3090/README.md)）。
- RTX 4090 / 3090 单卡：`tools/arch_check.py` 查内核能不能编译，真卡跑 Qwen3.8 整模型。

跟进了 SGLang 的话，不租卡也要模拟 SM86 跑一次 DeepSeek 整模型（`tools/arch_check.py sm86 --emulate sm86 -- 启动脚本`，去掉 SM120 专用的参数），
发请求并跑照抄。SM90 以下的钩子在 RTX PRO 5000 上不注册，`check_sglang.py` 查不到；目标还在、但调用方换了入口时，只有这样才能发现（[#10](https://github.com/a775828b-dot/xtier/pull/10)）。

## 5. 发布

1. 更新版本号（`pyproject.toml`、`xtier/__init__.py`）、[README 的版本更新](../README.md#版本更新)（只留 5 条）、[changelog](changelog.md)、新报告，并把报告加进[文档目录](README.md)。
2. 跟进了 SGLang 的话，更新 `SGLANG_COMMIT`、[constraints.txt](../constraints.txt)、[支持情况](support.md)里的环境表。
3. **干净的 checkout 按[快速开始](quickstart.md)跑一遍**（新环境、新缓存），能发现漏提交的文件和写错的步骤。
4. 推送、打标签、发 GitHub Release，Release 说明里写清 XTier 和 SGLang 的提交、环境、验证范围。
