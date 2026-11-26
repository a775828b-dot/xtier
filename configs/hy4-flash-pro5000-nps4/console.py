"""XTier console entry (xtier/console): Hunyuan-4-Flash NVFP4 on one RTX PRO 5000, NPS4 host.

Target workload: 1M KV pool, 1M context, 4 concurrent requests. 41 MoE layers x 512 routed experts
(top-16 on 1536 latent dim, NVFP4) live on the CPU engine + elastic GPU hot-expert cache (up to 288
slots/layer with NVFP4 KV); the four FP8 Engram tables (64.7 GB) stay on NVMe through XTier's
RowStore with preloaded per-row BF16 scales and L1-prioritized row cache; 12 DSA + 30 SWA MLA
layers share a hybrid 1M/128K KV pool (7.95 GB in NVFP4 vs 32.32 GB baseline); bundled 5-layer
DFly2 speculative drafter runs zero-patch via CustomSpecAlgo with fused FP8 draft LM head and
confidence-head verify truncation (lossless, skipping low-survival tail MoE routes).
"""

from pathlib import Path

from xtier.console.core import (common_options, elastic_option, interleave_args, interleave_option, interleave_tags,
                                kv_offload_args, kv_offload_env, kv_offload_option, kv_offload_tags,
                                miss_env, miss_option, share_env, share_option, share_tags)

HERE = Path(__file__).resolve().parent
POOL = 1048576
# Base resident hot-expert slots per (kv, chunk): sized so 1M-context cold prefill has >= 22 GiB headroom (0 OOM retries).
BASE_SLOTS = {
    ('nvfp4', '49152'): 152, ('nvfp4', '32768'): 176, ('nvfp4', '16384'): 208,
    ('fp8', '49152'): 120, ('fp8', '32768'): 144, ('fp8', '16384'): 184,
}
ELASTIC_MAX = {'nvfp4': '288', 'fp8': '256'}


def _common():
    options = common_options('DFly2', contexts=('1048576', '524288'))
    context = next(o for o in options if o['key'] == 'context')
    context.update(label='上下文 · KV 池 1M', hint=(
        '单个请求的上限（默认 1M）；4 并发共用 1M 混合 SWA/DSA KV 池（30 层滑窗 128K + 12 层全量 1M）'))
    return options


MODEL = {
    'id': 'hy4flash', 'order': 35,
    'name': 'Hunyuan 4 Flash NVFP4', 'code': 'HY4-FLASH', 'port': 18192, 'served': 'Hy4-flash-NVFP4',
    'spec': 'DFly2',
    'arch': '42 层 · 512 专家 top-16 · 5 层原生 DFly2 · 1M 混合 NVFP4 KV (7.95 GB) · Engram 64 GB NVMe · 4 并发',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 2400,
    'bench': {'kind': 'text', 'dec': '92–152 / 225.8 (4并发)', 'pre': '3630 / 3565 / 3454',
              'title': '1M KV 池 / 4 并发实测（原生 5 层 DFly2 投机 + 置信度截断 0.20 + NVFP4 混合 KV + 234 层 FP8/W8A8 + 48K 分块）：'
                       'decode = bs=1 实测区间（decode_speed 92.1 / 代码散文稳态 124–152 / 照抄 306 tok/s，876K 缓存 decode 88–111 tok/s）'
                       '/ bs=4 稳态 225.8（decode_speed 186.8–190.4）；prefill = 100K / 4×128K / 950K'},
    'options': _common() + [
        {'key': 'truncate', 'label': 'DFly2 置信度截断', 'default': 'on', 'switch': True,
         'choices': [['on', '开 · 阈值 0.20（推荐，无损）'], ['off', '关 · 全量 6 行验证']],
         'hint': '用 DFly2 内置 confidence_head 跳过低存活率尾部草稿的 MoE 专家计算并封顶验证长度（照抄 0 错、PPL 一致）；'
                 'bs=1 步时降 ~14%，bs=4 提速 +23–26%，高置信照抄自动保持满长 6 行（306 tok/s）',
         'off_when': [{'packs': 'off'}, {'spec': 'off'}]},
        {'key': 'kv', 'label': 'KV 缓存精度', 'default': 'nvfp4',
         'choices': [['nvfp4', 'NVFP4 · 7.95 GB（推荐，至多 288 槽）'], ['fp8', 'FP8 · 11.64 GB（至多 256 槽）']],
         'hint': '混合 SWA(128K) + DSA(1M) KV 池；NVFP4 仅占 7.95 GB（较基线省 24.4 GB），decode 经 L2 解包走 FlashInfer 无额外开销',
         'off_when': [{'packs': 'off'}]},
        {'key': 'engram', 'label': 'Engram 行缓存', 'default': '32',
         'choices': [['32', '32 GiB · L1 全表常驻零抖动'], ['16', '16 GiB'], ['4', '4 GiB · 省内存']],
         'hint': '64.7 GB 四层 Engram 放 NVMe，权重优先分给第 1 层（仅 1 层可掩蔽）；32 GiB 时 L1 占 16.3 GiB 零冲突零停顿'},
        elastic_option('on', 'decode 时将 prefill 预留显存借给热专家槽（NVFP4 至多 288 槽 / FP8 至多 256 槽），prefill 前自动归还'),
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '49152',
         'choices': [['49152', '48K · 推荐（152/120 基础槽）'], ['32768', '32K（176/144 基础槽）'], ['16384', '16K（208/184 基础槽）']],
         'hint': '长提示 prefill 分块大小，基础热专家槽按 KV 精度与分块自动留足 1M 上下文显存余量；decode 靠弹性槽自动补满',
         'off_when': [{'packs': 'off'}]},
        miss_option(),
        share_option('on', '读长提示时由 CPU 分担部分专家计算（默认开，长 prefill 稳定提速）'),
        interleave_option([['64', '64 步 / 块'], ['32', '32 步 / 块']], '64',
                          '一个请求在读长提示时，每读完一块先让其他并发请求 decode 这么多步'),
        kv_offload_option(),
    ],
    'notes': [
        {'when': {'packs': 'off'}, 'text': '基线：BF16 稠密层、全量 1M FP8 KV（32.3 GB）、48 槽、16K 分块，仅保留 SM120 / ModelOpt / Engram 必需钩子'},
        {'when': {'packs': 'on', 'elastic': 'off'}, 'warn': True,
         'text': '弹性热专家已关：decode 仅用基础槽数，热专家命中率会下降（开着可自动扩到 288 / 256 槽）'},
    ],
}


def _pool(o):
    return max(int(o['context']), POOL)


def build(o, port):
    base = o['packs'] == 'off'
    kv = 'fp8' if base else o['kv']
    chunk = '16384' if base else o['chunk']
    slots = '48' if base else str(BASE_SLOTS[(kv, chunk)])
    spec = o['spec'] == 'on'
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'DFLY': '1' if spec else '0',
           'CONTEXT': o['context'], 'KV_POOL': str(_pool(o)), 'BATCH': chunk, 'XTIER_CACHE_SLOTS': slots,
           'XTIER_ENGRAM_CACHE_GIB': o['engram'], 'MAX_RUNNING_REQUESTS': '4',
           'XTIER_HY4_KV_FP4': '1' if kv == 'nvfp4' else '0'}
    if base:
        env.update(XTIER_HY4_HYBRID_SWA='0', XTIER_HY4_INDEXPOOL_FP8='0',
                   XTIER_FP8_WEIGHTS='0', XTIER_ENGRAM_PREFETCH='0',
                   XTIER_CACHE_ELASTIC='0', XTIER_DRAFT_LMHEAD_FP8='0',
                   XTIER_DSPARK_TRUNCATE='0')
    else:
        if o['elastic'] == 'on':
            env.update(XTIER_CACHE_ELASTIC='1', XTIER_CACHE_ELASTIC_MAX_SLOTS=ELASTIC_MAX[kv])
        else:
            env['XTIER_CACHE_ELASTIC'] = '0'
        if spec:
            env['XTIER_DRAFT_LMHEAD_FP8'] = '1'
            env['XTIER_DSPARK_TRUNCATE'] = '0.20' if o.get('truncate', 'on') == 'on' else '0'
        else:
            env['XTIER_DSPARK_TRUNCATE'] = '0'
    env.update(miss_env(o))
    env.update(share_env(o))
    env.update(kv_offload_env(o))
    return env, [str(HERE / 'run-hy4-xtier.sh'), *interleave_args(o), *kv_offload_args(o)]


def tags(o):
    out = ['1M KV', '4 并发']
    if o['spec'] == 'on':
        out.append('DFly2')
        if o['packs'] == 'on' and o.get('truncate', 'on') == 'on':
            out.append('置信度截断')
    if o['packs'] == 'on':
        out.append('NVFP4 KV' if o['kv'] == 'nvfp4' else 'FP8 KV')
        if o['elastic'] == 'on':
            out.append('弹性热专家')
    if miss_env(o):
        out.append(f"未命中 {o['missgpu']}% GPU 代算")
    out += share_tags(o)
    return out + interleave_tags(o) + kv_offload_tags(o)
