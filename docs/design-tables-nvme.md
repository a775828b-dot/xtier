# 设计：大嵌入表的行放在 NVMe（`xtier/tables`）

两个模型各有一张按哈希查的大嵌入表，官方 main 只会把它整张放进显存或主机内存：

| 模型 | 表 | 行 | 行宽 | 大小 | 官方的放法 |
|---|---|---:|---|---:|---|
| DeepSeek V4.1 Flash 1017 | Engram（第 1、14 层各一张） | 2 × 3.84 亿 | 256 字节 FP8 + 8 字节 E8M0 scale | 189 GB | 显存按 TP 分行，或主机表（`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`） |
| Qwen3.8 Flash Next | PLE n-gram（检查点 128 个分片张量，分在 10 个文件） | 3.2 亿 | 160 字节 FP8（整表一个 BF16 scale） | 48 GiB | 锁页主机内存，或文件映射让 GPU 经主机页表直接读（`ple_offload_backend`） |

DS 的 Engram 加专家（276 GB）超过整机 377 GB 内存，不放 NVMe 就起不来；Qwen 的表放得下，但放 NVMe 后主机内存省 48 GiB
（小内存机器、或几个模型同机时有用）。

## 结构

- `csrc/tables/rows.cpp`（`libxtier_tables.so`）：一张表 = 按全局行号排好的若干段（检查点张量），每段 (文件, 行数, 行偏移,
  scale 偏移)；行宽任意，每行可带一段 scale 字节（DS）或不带（Qwen）。由 1017 管线的 `row_store.cpp`（10-22 版，含多卡修复）
  演变而来：
  - 未命中的行用 O_DIRECT 读，一次查表的所有未命中合成若干轮（每轮最多 512 页），由 16 个常驻读线程并发读
    （`XTIER_ROWS_IO_THREADS`），回调线程自己也参与；
  - 有界的直接映射行缓存（DS 两层共 4 GiB，`XTIER_ENGRAM_CACHE_GIB`；Qwen `XTIER_PLE_CACHE_GIB`，默认 4）；
  - DS 的 scale 可以启动时预读本 rank 的行（每层约 3 GB），未命中时只读行本身；
  - 本 rank 不持有的行（TP 分片）返回全零，由官方的 all-reduce 相加；行号越界或读失败直接 abort，不让生成带着错行继续；
  - `XTIER_ROWS_MODE=ram`：整文件映射并锁定（不读盘、无缓存），对照用。
- `xtier/tables/__init__.py`：绑定（`RowStore`、`Segment`、`tensor_locations` 从索引和 safetensors 头找张量位置）。
- `xtier/tables/lookup.py`（`GraphLookup`）：CUDA 图里可用的查表——当前流上 ids 拷到锁页内存、`cudaLaunchHostFunc` 节点里
  跑查表（回调里不调 CUDA API）、行和 scale 拷回显存，反量化留给模型包。暂存按容量复用、每种 id 数一个回调参数，
  图重放时地址不变；图捕获时不分配，捕获前由包预热（DS 每层 16384 个 id，即 682 token × 24 列；Qwen 16384 个 id）。

模型包只做钩子和反量化：

- `packs/deepseek_v4`（`XTIER_ENGRAM_ROWS=nvme`）：AROUND `EngramEmbedding.__init__` 不分配表、打开行表并预读 scale，
  参数换成零行张量，加载器只核对行数（safetensors 0.8 的 `get_tensor` 是惰性映射，98 GB 的张量不会被读）；
  REPLACE `_owned_rows` = 查表 + 官方 `engram_gather` 反量化。哈希、门控、投影和 all-reduce 仍是官方的。
- `packs/qwen4_exp`（`XTIER_PLE_ROWS=nvme`，`ple_rows.py`）：AROUND `Qwen4ExpNGramEmbedding.__init__` 记下模块前缀，
  AROUND `Qwen4ExpPinnedHostEmbedding.__init__` 不分配主机表、按前缀找到检查点的 128 个分片建行表（核对分片行数 =
  ceil(词表 / 分片数)，与官方加载器的摆放一致），REPLACE `gather` = 查表 + FP8→BF16（与官方 pinned 表内核同一转换，
  整表 `weight_scale` 仍由调用方乘），AROUND `load_weights` 把 `.ngram_embedding.shard_*` 从权重迭代器里滤掉。
  官方的 PLE 预取（旁路流上提前 `gather`）照常工作：查表在哪个流上调用就排在哪个流上。

## 验证

- `tests/tables_check.py`（CPU）：随机 id（含重复，第二次查表大多走缓存）+ 表两端 + 每个分段边界，整表和 TP 分片，
  scale 读盘 / 预读，与直接读文件逐字节比较。10-23：DS 第 1、14 层 8 种组合、Qwen 128 分片 2 种组合全部逐字节一致。
- `tests/table_lookup_check.py`（GPU）：包里的查表（DS `EngramLookup`、Qwen `ple_rows.gather`）对官方内核
  （`engram_gather`；`_gather_ple_embedding_from_pinned_kernel` 读同样字节的锁页表），直接调用和 CUDA 图重放换 id，
  整表和 TP 分片，逐位比较。
- 端到端：DS 1017 在官方 main + XTier 上起服务（Engram 两层各 2 GiB 行缓存），生成正确、照抄 0 错；Qwen3.8 的 NVMe 表与
  锁页表对照见 `docs/reports/2026-10-23-step3-ds-tables.md`。

## 放在哪块盘

随机读延迟决定开销。Qwen3.8 的 128 个分片：在傲腾 P5800X 上（`XTIER_PLE_MODEL` 指向分片副本）验证步 14.14 ms 对锁页表
14.13 ms、冷 prefill 差 3% 以内；在无 DRAM 的 EX900 上验证步 +7%、8K prefill +48%（8K token × 16 头约 13 万次 160 字节
随机读，16 路并发下 EX900 约 15–20 万 IOPS）。DS 的 Engram 文件一直在傲腾上（检查点第 47、48 个文件软链过去）。

## 以后

- 预取：哈希结果出来后在旁路流上提前查后面层的表（生产 `DS41_ENGRAM_PREFETCH=1`，与前面的层重叠）。
- 冷启动时的行缓存预热（按画像或上次运行的热行）。
