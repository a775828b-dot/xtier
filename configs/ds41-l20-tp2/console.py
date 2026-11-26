"""XTier console entry: DeepSeek V4.1 Flash 1017 on 2×L20 48G (SM89 Ada), TP=2, one card per NUMA node.

Hardware layout (measured on this box, 2× Xeon Gold 6530, 2 NUMA nodes, 32 physical cores each):

  GPU0 → NUMA node 0 (PCIe NODE)      GPU2 → NUMA node 1
  GPU1 → node 0 (spare)               GPU3 → node 1 (Strata / Qwen)

  **One TP rank per NUMA node.** XTier's expert workers are placed per NUMA node
  (xtier/engine/placement.py) and the C engine's worker CPU selection
  (xtier_worker_cpus in csrc/engine/engine.cpp) has no rank offset: two ranks on the
  same node get the *same* CPU set. Measured with both cards on node 0: 29/29 worker
  threads of the two ranks pinned to the same 29 CPUs, the per-layer CPU time went
  from 257 µs to 8812 µs (34×), single requests stuttered and the load crashed with
  Bus error / Segmentation fault. One card per node: 0 + 2 (not 0 + 1).

  The two cards are on different sockets, so no P2P: --disable-custom-all-reduce
  (custom all-reduce's IPC registration fails without P2P) and NCCL_P2P_LEVEL=PHB.

SM89 specifics (Ada, not Blackwell): no FP4 tensor cores (the experts are MXFP4 and
are dequantized in software), no FP4 indexer, no b12x, and 99 KiB of shared memory per
SM — the PRO 5000 production tile (128 wide) does not fit. The L20 tuned FP8 GEMM
configs live in the pack (xtier/packs/deepseek_v4/fp8_configs, device_name=NVIDIA_L20)
and are tuned per M (decode's M is batch × top-k, 1..48).

The measured numbers are from this box (docs/reports/2026-11-20-l20-tp2-tuning.md).

Standalone `python -m xtier.console` auto-discovers every configs/*/console.py, so this
2x-L20 TP=2 card also appears on single-GPU hosts. It needs both cards on separate NUMA
nodes and M set to the checkpoint (from XTIER_MODEL_PATH or M); use `--configs` to
allowlist the cards a host can actually run.
"""

import os
from pathlib import Path

from xtier.console.core import (common_options, elastic_option, interleave_args, interleave_option, interleave_tags,
                                share_env, share_option, share_tags)

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run-ds-xtier.sh'      # this config's launch script
SPS_TABLE = HERE / 'sps-table.json'  # DSpark cost table measured on this box

# The KV pool of all requests together (--max-total-tokens). 1,573, = 6 × 262144:
# six agents with a 256K context each stay resident. 1,673 bytes a token, so the
# pool is 2.45 GiB — the expert cache, not the KV pool, is what the VRAM goes to.
POOL = 1572864


def _common():
    options = common_options('DSpark', threads=('32', '56'), contexts=('1572864', '1048576', '524288'))
    context = next(o for o in options if o['key'] == 'context')
    context.update(label='上下文 · KV 池 1.5M', hint=(
        '单个请求的上限 256K。所有请求共用的 KV 池 1,572,864 token（= 6 × 256K）：6 个 agent 各自的 256K 上下文'
        '能同时常驻（实测 6 路 × 256K 并发，505 秒全部成功，队列为 0）。DS4.1 的 KV 每 token 1,673 字节，'
        '1.5M 池只占 2.45 GiB——真正吃显存的是热专家槽（每槽每层 8.96 MiB，40 层）'))
    threads = next(o for o in options if o['key'] == 'threads')
    threads.update(hint=(
        '每 rank 的专家 worker 数。每个 rank 独占一个 NUMA 节点（32 物理核），32 = 一个 worker 一个物理核，'
        'SMT 兄弟核留给 Python/scheduler 线程；实测每层 CPU 时间 625 µs（32）vs 796 µs（56，用了 SMT 兄弟核）'))
    return options


MODEL = {
    'id': 'ds41-l20', 'order': 50,
    'name': 'DeepSeek V4.1 Flash · L20 TP2', 'code': 'DSV4.1-L20', 'port': 8001, 'served': 'DeepSeek-V4.1-Flash',
    'spec': 'DSpark', 'arch': '40 层 + 3 层 DSpark 草稿 · 384 专家/层 · MXFP4 专家（SM89 软件解码）',
    'hardware': '2×L20 48G SM89（GPU0+2，一卡一 NUMA 节点）· 2×Xeon Gold 6530 · 1007 GiB',
    'ready_timeout': 1800,
    # measured here: 8 sessions × 4 turns agent bench (2026-11-20); cold prefill of a 100K prompt
    'bench': {'kind': 'agent', 'dec': '43', 'pre': '3028',
              'title': 'agent 速度：8 会话 × 4 轮聚合 42-46 tok/s（单流 36-55）；冷 prefill：100K 提示 3028 tok/s'},
    'options': _common() + [
        {'key': 'verify', 'label': 'DSpark 验证', 'default': 'compact',
         'choices': [['compact', '变长（compact）· 实测'], ['static', '固定（static）· SM89 不可用']],
         'hint': '变长：按本机实测的成本表（sps-table.json）决定每个请求验证几个草稿 token。固定（static）在 SM89 上不可用：'
                 '它要 deep_gemm 的 get_paged_mqa_logits_metadata，而那个在 SM89 上返回 None'},
        {'key': 'engram', 'label': 'Engram 行缓存', 'default': '4',
         'choices': [['4', '4 GiB · 实测'], ['8', '8 GiB'], ['16', '16 GiB']]},
        elastic_option('on', 'decode 时把只有 prefill 才用的显存借给热专家，prefill 前先还回。'
                             '本机常驻 32 槽、弹性上限 128 槽：空闲显存从 7.9 GiB 涨到 18.3 GiB，冷 prefill 2498 → 3028 tok/s'),
        {'key': 'slots', 'label': '常驻热专家槽 / 层', 'default': '32',
         'choices': [['32', '32 · 实测（+ 弹性到 128）'], ['48', '48'], ['64', '64 · 常驻占 22.4 GiB']],
         'hint': '读长提示时每层常备的槽数；decode 时弹性热专家补到 128 槽。常驻 64 槽要 22.4 GiB 显存，'
                 '留给 KV 池和 prefill 暂存的就少了（冷 prefill 1806 → 3028 tok/s）',
         'off_when': [{'packs': 'off'}]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '32768',
         'choices': [['32768', '32K · 实测'], ['16384', '16K'], ['8192', '8K']],
         'hint': '每个分块都要把 40 层不在热槽里的全部专家传上显卡（3.36 GiB/层/卡），块越大上传轮数越少：'
                 '实测冷 100K prefill 8K=1597、16K=2498、32K=3028 tok/s。'
                 '注意：别照抄 PRO 6000 那套的 2K——那是 all-reduce 走 PCIe switch（19.6 GB/s）的前提，'
                 '本机跨 socket 走 UPI，块越小 all-reduce 次数越多反而更慢（2K 时只有 771 tok/s）',
         'off_when': [{'packs': 'off'}]},
        share_option('off', '长提示读入时 CPU 引擎（那时闲着）替 GPU 算一部分专家。'
                            'SM89 没有 FP4 张量核，GPU 侧要先把 MXFP4 展开成 BF16，这项在本机未测出收益，默认关'),
        interleave_option([['16', '每块后穿插 16 步'], ['32', '每块后穿插 32 步 · 实测'],
                           ['64', '每块后穿插 64 步']], '32',
                          '多请求不卡顿的关键：prefill 之后强制跑 N 个 decode 轮再排下一个 prefill。'
                          '关掉（0）时几个长上下文的 prefill 会连成几百秒的连续 prefill，'
                          '期间已就绪的请求一个 token 都发不出来——实测 6 路 × 100K 时最大 token 停顿 116 秒。'
                          '32：停顿降到 330 ms、8×4 agent 吞吐 +27%、TTFT 不变；'
                          '64 会让新请求的 TTFT p95 从 9 秒涨到 16 秒'),
    ],
    'notes': [],
}


def build(o, port):
    """(environment additions, argv): every declared option wired into the launch script's env.

    run-ds-xtier.sh unconditionally passes --prefill-decode-interval ${PDI:-32}, so the
    interleave choice goes through PDI (0 = off) instead of a second, duplicate CLI flag.
    """
    base = o['packs'] == 'off'
    spec = o['spec'] == 'on'
    chunk = o['chunk']
    env = {'PORT': str(port), 'KV_POOL': o['context'], 'XTIER_THREADS': o['threads'],
           'DSPARK': '1' if spec else '0', 'XTIER_ENGRAM_CACHE_GIB': o['engram'],
           'CHUNK': chunk, 'PREFILL_TOKENS': chunk, 'XTIER_CACHE_SLOTS': o['slots'],
           'PDI': '0' if o['interleave'] == 'off' else o['interleave']}
    if base:   # the README baseline: no elastic cache and no CPU-prefill pack
        env.update(XTIER_CACHE_ELASTIC='0', XTIER_GPU_PREFILL_MIN='1024')
    else:
        env['XTIER_CACHE_ELASTIC'] = '1' if o['elastic'] == 'on' else '0'
        env.update(share_env(o))
        if spec and o['verify'] == 'compact':   # sps-table.json budgets compact verification
            env.update(SGLANG_RAGGED_VERIFY_MODE='compact', SPS_TABLE=str(SPS_TABLE))
    model = os.environ.get('XTIER_MODEL_PATH') or os.environ.get('M')
    if model:   # run-ds-xtier.sh aborts on an unset M; forward the caller's choice
        env['M'] = model
    return env, [str(RUN)]


def tags(o):
    """Extra parts of the server's label, reflecting the active card options."""
    out = []
    if o['packs'] == 'on':
        if o['elastic'] == 'on':
            out.append('elastic')
        if o['spec'] == 'on' and o['verify'] == 'compact':
            out.append('compact verify')
        out += share_tags(o)
    return out + interleave_tags(o)
