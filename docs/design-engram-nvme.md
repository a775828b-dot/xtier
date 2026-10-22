# 设计：DeepSeek V4.1 的 Engram 行放在 NVMe（`packs/deepseek_v4`）

状态（2026-10-23）：行存储已移植（`csrc/engram/rows.cpp`、`xtier/packs/deepseek_v4/rows.py`），`tests/engram_rows_check.py` 在第 1、14 层（各约 3.84 亿行）整表和 TP 分片下与直接读文件逐字节一致；钩子（方案第 2 步）未做。DeepSeek V4.1 1017 在官方 main 上跑通的前置条件。

## 为什么

1017 检查点里 Engram 表 189 GB、专家 276 GB。XTier 把专家放在主机内存（CPU 引擎的 NUMA 分片），
官方 main 的 Engram 只有两种放法：显存（按 TP 分行），或主机内存（`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`，
`_HostTable` 映射给 GPU，`engram_gather` 内核直接读）。后者要把 189 GB 整表锁在内存里，加上专家超过整机 377 GB。
生产管线（`ds41-1017/adapter/engram_backend.py` + `row_store.cpp`）把行留在 NVMe 上按需读，内存里只放缩放和行缓存。

## 官方 main 的接口（`sglang/srt/layers/engram.py`）

- `EngramEmbedding(num_embeddings, dim, layer_id, column_bounds)`：每层一张 FP8 表（`weight` [rows, dim]）
  加 E8M0 块缩放（`scale` [rows, dim/32]），按 TP 分行（`shard_range`）；`_load_rows` 是两者的加载器。
- `forward(indices)` → `_lookup` → `_owned_rows(indices)`：本 rank 持有的行反量化成 BF16，其余行置零，TP 再 all-reduce。
  `_owned_rows` 是唯一读表的地方，生产补丁替换的也是它。
- 哈希在 `EngramHasher`（GPU 上算，`engram_hash_ids*`），历史由 `ngram_embedding_manager` 在图外准备。

## 生产的做法（`row_store.cpp`，451 行）

- 打开检查点的 safetensors 文件，按行偏移 O_DIRECT 读，`row_lo`/`row_hi` 限定本 rank 的行（10-22 修过多卡两处问题）。
- 缩放启动时预读常驻（每层 rows × 8 字节量级），行缓存有预算（生产 4 GiB），统计命中/读次数。
- 查表在 CUDA 图里执行：`cudaLaunchHostFunc` 的主机回调里做 IO（回调里不调 CUDA API），前后是 ids 的 D2H 和行的 H2D，
  然后 GPU 反量化。`DS41_ENGRAM_PREFETCH=1`：哈希一出来就在旁路流上开始第 1、14 层的查表，和前面的层重叠。

## 方案

1. `packs/deepseek_v4/engram.py`：移植 `row_store.cpp` 进 `csrc/engram/`（构建进 `libxtier_engram.so`），Python 侧移植
   `engram_backend.py`。
2. 钩子：
   - AROUND `EngramEmbedding.__init__`：开关 `XTIER_ENGRAM_ROWS=nvme` 时不分配表，记下层号、行范围与 `column_bounds`，
     参数换成零长度张量，`_load_rows` 只校验来源（文件、偏移、形状），不复制。
   - REPLACE `EngramEmbedding._owned_rows`：行存储查表（主机回调，图内可用），返回 BF16，未持有的行置零（与官方语义一致，TP 仍 all-reduce）。
   - 可选：哈希之后的旁路流预取（生产的 `prefetch`）。
3. 内存预算（单机）：专家分片 276 GB + Engram 缩放约 6 GB + 行缓存 4 GB + 其余约 10 GB，约 296 GB（生产实测上限 298 GB）。
   GPU prefill 要锁页专家分片（276 GB），与生产相同。
4. 验证：
   - 单元：同一组 ids，行存储查表与 safetensors 直接读 + 官方反量化逐位一致（含 TP=2/4 的行范围，沿用 10-22 的多卡模拟）。
   - 端到端：PPL、照抄、agent_bench（生产 DS 的测速口径，五会话覆盖 512K）对照 `bench/baselines/deepseek_v4_1017.json`。
5. 其余 DS 生产改动按 `docs/inventory.md` 逐项核对：DSpark 固定验证 6 token、DSpark c128 图、草稿词表头 FP8、
   prefill 稀疏索引器、1017 会话编码与防注入、MXFP4 检查点适配。
