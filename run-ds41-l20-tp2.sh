#!/usr/bin/env bash
# XTier 在 L20(SM89) 上的 TP=2 配置：GPU0(node0) + GPU2(node1)，256K 上下文，关掉 SM120 专用加速项。
#
# 用法： setsid bash run-ds41-l20-tp2.sh > /tmp/opencode/xtier-tp2.log 2>&1 &
#
# ★ 本脚本的默认值 = 生产配置（与 /etc/systemd/system/DeepSeek-V4.1-Flash.service 的
#   Environment= 逐项同值）。生产入口是 systemd unit，不是直接跑本脚本。
#   改配置请改 unit（权威面）或本脚本，两边必须同步；verify_prod.sh 会交叉校验。
#
# 相对作者生产配置 (configs/ds41-pro5000-nps4/run-ds-xtier.sh) 的改动：
#   1. FLASHINFER/TVM 架构 12.0 -> 8.9（L20 是 SM89）
#   2. XTIER_PREFILL_FP4=1 -> 0（Ada 没有 FP4 张量核，走 arch.py 里的软件解码）
#   3. 去掉 --enable-deepseek-v4-fp4-indexer（SM120 的 FP4 索引器）
#   4. DSpark 投机已开（DSPARK 默认 1），但必须配 SGLANG_RAGGED_VERIFY_MODE=compact：
#      static 验证在 SM89 上会调 deep_gemm.get_paged_mqa_logits_metadata（返回 None -> assert 崩）。
#   5. tp-size 1 -> 2，GPU0(node0)+GPU2(node1)（一卡一 NUMA 节点）
#   6. XTIER_TP_NODES="0|1"：必须显式写。默认 auto 会按 NUMA 节点数平均切，
#      两个 rank 会拿到 node0/node1；GPU0/1 都在 node0，rank1 就被推到 node1 跨 UPI。
#   7. XTIER_CACHE_SLOTS 50 -> 43：TP 下专家按中间维切半，每槽 8.95 MiB；
#      43 是 CHUNK_SLOTS['32768']，即「弹性关掉时」的下限。80 槽会把显存吃到只剩 ~5 MB，
#      allocator 反复 OOM 回落，prefill 从 2.8K 掉到 600 tok/s。
#      真正的提速靠 XTIER_CACHE_ELASTIC_MAX_SLOTS=64（base 21 + 弹性 43）：
#      命中率 62%->74%，4 路吞吐 +33%，8 路 >200ms 49%->4%。
#   8. XTIER_THREADS 112 -> 32：每个 rank 独占一个 socket 的 32 个物理核
#      （40/48/56 更差：只能在同一物理核塞第 2 个 worker，SMT 不翻倍，且主线程被挤到 7 CPU）
#   9. --schedule-policy lpm -> fcfs：lpm 按最长前缀插队，会把队尾请求的第 2 轮 prefill
#      一再推迟（多 agent 场景是负优化）。12 路并发实测 wall 216s->117s、吞吐 +80%、无饿死。
#  10. --max-running-requests 8 = 本机(L20/SM89 + DSpark)的真实硬顶：
#      bs>8 时 CUDA graph 无法 replay(只捕获 bs 1..8) -> compact(ragged) verify 失效 ->
#      退化成 static(rows_per_request=5) -> 跳过 JIT indexer planner ->
#      deep_gemm.get_paged_mqa_logits_metadata 在 SM89 返回 None -> assert 崩。
#      过 8 只有三条路：放弃 DSpark(不值) / 换 SM100+ 机器 / 改 SGLang 的 kernel 元数据规划器。
set -eo pipefail
M=${M:-/opt/models/unsloth/DeepSeek-V4.1-Flash-1017}
PORT=${PORT:-8001}
# SPS 投机接受率表（L20 实测）。没有它 DSpark 也能跑，但没有调优收益。
SPS_TABLE=${SPS_TABLE:-/opt/Xtier/sps-table-l20-gpu02.json}
ENV=/opt/miniforge3/envs/SGLang

# ---- SM89：架构列表必须是 8.9 ----
export CUDA_HOME=/usr/local/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export FLASHINFER_CUDA_ARCH_LIST=8.9 TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=8.9
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1
# 钉住 rustup 工具链：/opt/sglang/rust/rust-toolchain.toml 要 1.92 + clippy/rustfmt，
# 缺组件时 rustup 会在后台下载并卡死几十秒到几分钟，SGLang 的 Rust 图像处理器探测就会挂在这里。
# 钉死后 cargo --version 立即返回，探测快速失败回落 PIL。
export RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-1.92-x86_64-unknown-linux-gnu}

# ---- GPU0 + GPU2（一卡一 NUMA 节点）----
# XTier 的 TP 放置要求每个 rank 一个 CPU 引擎进程 + 自己的 NUMA 节点：worker_cpu() 无 rank 偏移，
# 两个 rank 若同在 node0 会拿到完全相同的 CPU 集合（实测 29/29 重叠，每层 CPU 从 257us 劣化到 8812us）。
# GPU0 在 node0、GPU2 在 node1，各自的专家权重与 GPU 同域，NCCL all-reduce 走 UPI。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
# NCCL: 让 NCCL 走 host bridge 而非整机内存中转（实战报告：all-reduce 3.83 -> 19.57 GB/s）
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}

# ---- XTier 基本开关 ----
# ---- DSpark 投机解码：草稿层（mtp.*）就在主 checkpoint 里，草稿路径填主模型路径 ----
SPEC=()
if [ "${DSPARK:-1}" = 1 ]; then
  export XTIER_DRAFT=1
  # compact = 变长验证，会把 rows_per_request 变成 1，从而走 SGLang 自己的 JIT indexer metadata。
  # static（默认）是固定 6 行，会调 deep_gemm 的 get_paged_mqa_logits_metadata，而它在 SM89 上返回 None。
  export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-compact}
  SPEC=(--speculative-draft-model-path $M
        --speculative-algorithm DSPARK
        --speculative-dspark-block-size 5
        --speculative-num-draft-tokens 6)
  if [ -n "${SPS_TABLE:-}" ]; then
    SPEC+=(--speculative-dspark-sps-table-path "$SPS_TABLE")
  fi
fi

export XTIER=1
export XTIER_THREADS=${XTIER_THREADS:-32}        # 每 rank 一个节点（32 物理核），一 worker 一物理核，SMT 兄弟核留给主线程
if [ "${TP_SIZE:-2}" -gt 1 ]; then
  # 一卡一节点：rank0->node0(GPU0), rank1->node1(GPU2)。默认 auto 在两卡同节点时会退化，这里显式写。
  export XTIER_TP_NODES=${XTIER_TP_NODES:-"0|1"}
else
  unset XTIER_TP_NODES                              # TP=1 不要设：per_rank 会报 2 组对 1 个 rank
fi
# auto = 「rank 的节点 CPU 减去 worker 和 dispatcher」，把 Python/scheduler/torch 线程挪开不抢专家核
export XTIER_MAIN_CPUS=auto
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-43}   # 43 = CHUNK_SLOTS['32768']，是「弹性关掉时」的下限；80 槽会把显存吃到只剩 ~5 MB，allocator 反复 OOM 回落，prefill 从 2.8K 掉到 600 tok/s
# 换页节奏：每 2 个 decode window 换一轮、每轮 32 次（= 32 次/window，原来 64 次/window）。
# 换页要从主机内存读专家、和 CPU 引擎抢内存带宽，所以宁少勿多。
export XTIER_CACHE_EVERY=${XTIER_CACHE_EVERY:-2} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32} XTIER_CACHE_LOG=200
# 提速项 #1 专家热度 profile（本机 L20 实测路由统计；不占显存，只决定热槽初始装谁）
export XTIER_CACHE_PROFILE=${XTIER_CACHE_PROFILE:-/opt/Xtier/ds41-profile-l20.json}
# 换入门槛 3 / 遗忘衰减 0.9（源码默认 1.5 / 0.7）：门槛高、忘得慢 → 少换页。
# 实测 swaps/round 111 → 63，命中率 0.654 → 0.671，每层 CPU 1023 → 985 µs，吞吐持平或更好。
export XTIER_CACHE_MIN_GAIN=${XTIER_CACHE_MIN_GAIN:-3}
export XTIER_CACHE_DECAY=${XTIER_CACHE_DECAY:-0.9}
# agent 场景三件套：用 prefill 路由预热、借 prefill 显存当额外热槽、更激进的自适应换入
# prompt 预热改成 64 次 × 8 轮（总量与 256×4 相同，但轮次多、每次少，仓库报告里这个组合最好）
export XTIER_CACHE_PROMPT=1 XTIER_CACHE_PROMPT_WINDOWS=10
export XTIER_CACHE_PROMPT_ROUNDS=${XTIER_CACHE_PROMPT_ROUNDS:-8} XTIER_CACHE_PROMPT_SWAPS=${XTIER_CACHE_PROMPT_SWAPS:-64}
export XTIER_CACHE_ELASTIC=${XTIER_CACHE_ELASTIC:-1} XTIER_CACHE_ELASTIC_RESERVE_GIB=2 XTIER_CACHE_ELASTIC_CRITICAL_GIB=1
# 常驻槽降到 21（base 0.5），上限 64：空闲时弹性扩回，prefill 时让出显存给 32K chunk 的专家上传暂存。
# 实测 available_gpu_mem 从 7.9GB 涨到 18.3GB，冷 prefill 2498 -> 3028 tok/s。
export XTIER_CACHE_ELASTIC_BASE=${XTIER_CACHE_ELASTIC_BASE:-0.5} XTIER_CACHE_ELASTIC_MAX_SLOTS=${XTIER_CACHE_ELASTIC_MAX_SLOTS:-64}   # base 0.5 -> 常驻 21 槽；上限 64 = 21 常驻 + 43 弹性槽（作者 72GB 卡用 68）。43->64：命中率 62%->74%，4 路吞吐 +33%，8 路 >200ms 49%->4%；涨满后余 5.0GB > reserve 2GB，零 OOM
export XTIER_CACHE_ADAPT=1
export XTIER_THP=${XTIER_THP:-0}                  # 必须：否则锁页内存吃掉大页
# 引擎开关（沿用生产）
# SPIN_COUNT：worker 空转时自旋多少次再 futex 休眠。生产配置用 1048576（100 万次 pause），
# 两个 rank 共 64 个 worker 会长期忙等（整机 80% us）。32768 是源码默认值，本机实测吞吐不变、
# 负载下 CPU 占用从 80% 降到 1.7-39%：机器上还有别的服务（Strata），省下来的核很值。
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-32768} XTIER_DYNAMIC=2 XTIER_DOWN_BF16=1 XTIER_DOWN_GRAIN=24

# ---- Engram 表（189 GB）走 NVMe，按行读 ----
export XTIER_ENGRAM_ROWS=nvme XTIER_ENGRAM_MODEL=$M XTIER_ENGRAM_CACHE_GIB=${XTIER_ENGRAM_CACHE_GIB:-4}

# ---- prefill：Ada 没有 FP4 张量核，关掉 FP4 路径 ----
export XTIER_PREFILL_FP4=0 XTIER_PREFILL_UNPACK=1 XTIER_PREFILL_PREFETCH=1
export XTIER_PREFILL_OVERLAP=1 XTIER_PREFILL_PREFETCH_CHUNKS=6
export XTIER_PREFILL_EXPERT_CHUNK=8 XTIER_PREFILL_ROUTE_ACCUM=1
export XTIER_CPU_PREFILL_BATCH=2048
# agent 每轮只有 700-1000 token：原来的 1024 门槛把它们全推给 CPU，和 decode 抢同一批核
export XTIER_GPU_PREFILL_MIN=64
export XTIER_HYBRID_PREFILL_MAX=0          # 关掉「未命中路由交给 CPU」的 hybrid 路径
export XTIER_SPLIT_PREFILL_MAX=6144
export XTIER_DOWN_PRECISION=tf32 XTIER_PREFILL_TILE_M=64
# 提速项 #4 GPU prefill tile（生产全套；默认空 = 源码默认 64；设 1 启用全套）
if [ "${XTIER_TILES_PROD:-0}" = 1 ]; then
  export XTIER_GATE_BN=${XTIER_GATE_BN:-128} XTIER_GATE_LAUNCH=${XTIER_GATE_LAUNCH:-8,3}
  export XTIER_DOWN_BN=${XTIER_DOWN_BN:-128} XTIER_DOWN_BK=${XTIER_DOWN_BK:-64} XTIER_DOWN_LAUNCH=${XTIER_DOWN_LAUNCH:-8,3}
fi

# ---- 服务参数 ----
# 注意：下面 exec 的命令行里绝对不能出现 # 注释行——bash 会把续行中间的注释当成
# 「注释到逻辑行结尾」，从而吞掉后面的所有参数（曾导致 --disable-custom-all-reduce
# 丢失、custom all-reduce 在无 P2P 的 GPU0+GPU2 上崩溃）。
# chunk 32K：每个 prefill chunk 都要为 40 层重传全部 384 个专家（3.36 GiB/层/卡），
# chunk 越大上传轮数越少。实测冷 100K prefill：8K=1597、16K=2498、32K=3028 tok/s。
ARGS=(
  --model-path $M
  --served-model-name DeepSeek-V4.1-Flash
  --host 127.0.0.1 --port $PORT
  --tp-size ${TP_SIZE:-2}
  --trust-remote-code
  # PDI：prefill 之后强制跑多少个 decode 轮再排下一个 prefill（0 = 不穿插）。
  # 这是「多请求卡顿」的开关：0 时几个长上下文的 prefill 会连成一段几百秒的连续 prefill，
  # 期间所有已就绪的请求一个 token 都发不出来——实测 6 路 × 100K 时最大 token 停顿 116 秒。
  # 32：停顿降到 330 ms，8×4 agent 吞吐 +27%，冷 prefill 不变；64 会把新请求的 TTFT p95 推到 16s。
  # fcfs 而非 lpm：lpm 按最长前缀插队，会把队尾请求的第 2 轮 prefill 一再推迟；12 路并发实测 wall 216s->117s、吞吐 +80%、无饿死
  --prefill-decode-interval ${PDI:-32} --schedule-policy ${SCHEDULE:-fcfs}
  --skip-server-warmup
  # 8 = 本机(L20/SM89 + DSpark)的真实硬顶：bs>8 时 compact(ragged) verify 失效，退化成 static 后 SGLang 的 indexer 元数据规划器在 SM89 返回 None → assert 崩
  --max-running-requests ${MAX_REQ:-8}
  --chunked-prefill-size ${CHUNK:-32768} --max-prefill-tokens ${PREFILL_TOKENS:-32768}
  --prefill-max-requests ${PREFILL_REQ:-8} --cuda-graph-max-bs-decode ${GRAPH_BS:-8}
  --context-length ${CONTEXT:-262144} --max-total-tokens ${KV_POOL:-1572864}
  --mem-fraction-static ${MEM:-0.88}
  --disable-shared-experts-fusion
  --disable-custom-all-reduce
  --cuda-graph-backend-prefill disabled
  --cuda-graph-backend-decode full
  --enable-decoder-swa-bounded-replay
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
  --watchdog-timeout 1200 --enable-metrics --enable-cache-report
  --enable-hierarchical-cache --hicache-ratio 2
  --weight-loader-drop-cache-after-load
)
[ -n "${MAMBA_CACHE:-16}" ] && ARGS+=(--max-mamba-cache-size ${MAMBA_CACHE:-16})
ARGS+=("${SPEC[@]}")

if [ "${DRYRUN:-0}" = 1 ]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi

exec $ENV/bin/python -m sglang.launch_server "${ARGS[@]}" "$@"
# 刻意没开：--enable-deepseek-v4-fp4-indexer（SM120 专用，L20/SM89 不支持，serving_hook 会拒绝）
