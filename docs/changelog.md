# 版本更新

按日期倒序，README 只列最近 5 条。

- 2026-10-30（下午）：跟进官方 SGLang main `6162b07199`（10-22 的 `8cd3c494` 之后 592 个提交）。XTier 跟着改了 Engram 行放
  NVMe、MoE 方法的 top-k、Qwen PLE 分段 prefill；DeepSeek 的 prompt 改由上游用 deepseek-recipe 编码，删掉 `recipe_tokens`。测试过的
  提交写进安装步骤和 `xtier.SGLANG_COMMIT`，装的版本不同时启动会警告，`tools/check_sglang.py` 核对钩子目标与 import。三个模型照抄、
  PPL 与旧版一致，512K 不 OOM，dsh 连续 15 分钟无错。另外：快速加载支持 MoE 张量并行（逐字节对过，尚未在双卡上实测）；英伟达官方
  GLM NVFP4 检查点（NextN 层是 BF16）开 MTP 不再崩溃，草稿层的专家加载时转成 NVFP4 交给 CPU 引擎；启动台在 chroot 等没有用户
  systemd 的环境里退回 setsid。详见[文档](reports/2026-10-30-sglang-6162b07.md)。
- 2026-10-30：双卡张量并行（`TP=2`）实测通过：双 RTX 3090 上 Qwen3.8 decode +42–48%、120K 首字 −35%，照抄 0 错；修了
  `~/.cache` 是软链接时 SGLang 的 JIT 内核每次启动都重编（30 系单卡启动省约 5 分钟）。详见[文档](reports/2026-10-30-tp.md)。
- 2026-10-29（晚）：支持 RTX 30 系（SM86）：没有 FP8 硬件，FP8 一律读时软件解码（与硬件逐位相同），DeepSeek 的 FP8
  投影只量化权重；RTX 3090 实测 Qwen3.8 整模型照抄 0 错、120K 上下文。详见[文档](reports/2026-10-29-rtx30.md)。
- 2026-10-29：支持 RTX 40 系（SM89）：FP4 读时软件解码（与硬件逐位相同，不增加精度损失），三个模型的加速包补上 SM90
  以下的内核；RTX 4090 实测 Qwen3.8 整模型照抄 0 错、120K 上下文。详见[文档](reports/2026-10-29-rtx40.md)。
- 2026-10-28：decode 时可把一部分未命中的专家交给 GPU 代算（可选，`XTIER_DECODE_MISS_GPU`，默认关；启动台三张卡可选），
  给 CPU 比 PCIe 慢的机器用，结果与命中的专家相同。详见[文档](reports/2026-10-28-decode-miss-gpu.md)。
- 2026-10-27（下午）：快速加载（可选，`XTIER_FAST_LOAD=1`，默认关）：专家直接从模型文件或冷存储读进 CPU 引擎，仍是全部装好
  才报在线；DeepSeek 启动 472 → 186 s，卸载 30% 409 → 124 s。详见[文档](reports/2026-10-27-fast-load.md)。
- 2026-10-27：启动台新卡片「DeepSeek V4.1 Flash 1017 · 最小化」：32 GB 显存 + 128 GB 内存（模拟 RTX 5090），70% 的专家放硬盘，
  prefill 约 1,160、decode 约 30 tok/s。详见[文档](reports/2026-10-27-ds41-minimal.md)。
- 2026-10-26（晚）：冷专家卸载到硬盘（可选，`XTIER_COLD=1`），四个模型都能用；DeepSeek 卸载 30% 时 decode 每步 +44%。
  详见[文档](reports/2026-10-26-cold-experts.md)。
- 2026-10-26：弹性热专家槽（DeepSeek 每步 −4.3%、GLM −1.5%）；GLM 常备槽 60 → 58；启动台报在线前先预热；MiniMax-M3.1-Flash
  暂停开发。详见[文档](reports/2026-10-26-elastic-cache.md)。
- 2026-10-25：测速表改为官方 main 对加速包；Qwen 启动台默认用自动规划布局；热专家池可选用提示的路由预热（`XTIER_CACHE_PROMPT=1`，
  默认关）；修了 MiniMax 包钩子和 GC 冻结两个启动报错。详见[文档](reports/2026-10-25-prompt-warmup.md)。
- 2026-10-24（晚）：自动规划器第一版：本机测出各层 GPU / CPU 耗时，在显存预算内逐层定热专家槽数（`XTIER_CACHE_PLAN=auto`）；
  Qwen 启动时自动布局冷 prefill +2.4–2.9%，其余模型保留手工布局。详见[文档](reports/2026-10-24-planner.md)。
- 2026-10-24（下午）：MiniMax 固定 512K、KV 存 NVFP4（42.5 → 24 GiB），DSpark 按置信度截断验证（无损，agent decode +11%）。详见[文档](reports/2026-10-24-minimax-512k.md)。
- 2026-10-24（上午，二）：SM120 上 prefill 大小的 MXFP8 GEMM 改用 128×128 tile（输出逐位相同），MiniMax 64K 冷 prefill
  1,606 → 1,963 tok/s。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（上午）：新模型 MiniMax-M3.1-Flash（SGLang M3.1 分支），加速包 agent 每步 70.6 → 59.5 ms；DeepSeek 索引器 decode
  走分页路径，每步 33.0 → 32.5 ms。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（清晨）：DeepSeek 的 wo_a 保持 FP8（每步 33.9 → 33.0 ms，显存 −1.25 GiB），提示按 deepseek-recipe 分词；修了 TP 1
  下 NCCL 临时申请显存导致的 OOM。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（夜）：测速统一用 agent 速度；启动参数补上生产的接口参数（缓存命中上报、工具调用解析）。详见[文档](reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-24：控制台 `xtier/console/`：一个网页选模型、调参数、启动 / 停止，加模型只需加配置目录。详见[文档](console.md)。
- 2026-10-24：Qwen 默认 NVFP4 KV，省下的显存多放一层常驻（冷 prefill +0.5–1.4%）；DeepSeek 按生产 agent_bench 实测冷 prefill 快
  10–25%。详见[文档](reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-23（夜）：通用投机解码包：GLM 用热词表拒绝采样和置信度截断，decode 93.0 → 95.6 tok/s（生产 84.8）。详见[文档](reports/2026-10-23-speculative-fp4kv.md)。
- 2026-10-23（晚）：512K prefill 追平生产：Qwen 125 → 102 s（生产 106 s），GLM 224 → 192 s（生产 201 s）。详见[文档](reports/2026-10-23-prefill.md)。
- 2026-10-23（晚）：DeepSeek 移植 Engram 提前查表和草稿词表头 FP8，decode 每步 −8%。详见[文档](reports/2026-10-23-ds-engram-prefetch.md)。
- 2026-10-23：README 测速表；加载不再换出，DeepSeek 启动 811 → 409 s。详见[文档](reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 3 步：DeepSeek 跑通官方 main，通用 NVMe 表，GLM 的 FP8 权重，双卡准备。详见[文档](reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 2 步：后端、热专家缓存、GPU prefill 在官方 main 上跑通 Qwen 与 GLM。详见[文档](reports/2026-10-23-step2-qwen.md)。
- 2026-10-23：第 1 步：CPU 引擎迁入 `csrc/`，三个模型的真实层与 lkqmoe 0.5.6 逐位一致（`bench/results/engine-ab-20261023.json`）。
