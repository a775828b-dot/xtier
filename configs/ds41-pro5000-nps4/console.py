"""XTier console entry (xtier/console): DeepSeek V4.1 Flash 1017 on one RTX PRO 5000, NPS4 host. Option defaults = the
README "+ 加速包" row: 48K prefill chunks with 35 hot slots per layer (the elastic expert cache gives decode the slots
a chunk's forward needs back, up to 68), the CPU's share of prefill, Engram rows on NVMe with a 4 GiB row cache, DSpark;
a KV pool of 1M tokens for the server's four running requests (POOL).

48K chunks, not 22K (2026-11-11, docs/reports/2026-11-11-prefill-chunk.md): every chunk uploads each layer's experts
that are not in the hot slots, about 2.5 s a chunk whatever its size, so a larger chunk spreads that over more tokens
(cold prefill +28-35%, decode unchanged). The forward of a larger chunk needs more GPU memory, taken from the hot
slots: CHUNK_SLOTS."""

from pathlib import Path

from xtier.console.core import (common_options, elastic_option, interleave_args, interleave_option, interleave_tags,
                                miss_env, miss_option, share_env, share_option, share_tags)

HERE = Path(__file__).resolve().parent
# the cold-expert store (XTIER_COLD): four PCIe 5.0 drives (the manifest is in the first)
COLD_PATHS = [f'/data/fast/cold{i}/xtier-cold' for i in (1, 2, 3, 4)]
# The hot slots a layer can hold beside a prefill chunk of this size (0.7 GiB a slot over the layers; the elastic cache
# learnt 13-20 GiB for a 48K forward, 10.6 for 32K, 9-10 for 22K). Measured at 48K + 35: a 520K prompt, two requests
# filling the 512K pool and two agent_bench rounds without an out-of-memory error, 69.0 of 71.7 GiB at the peak;
# 32K + 43 and 22K + 50 read 520K prompts too. More slots than this are started as this many.
CHUNK_SLOTS = {'49152': 35, '32768': 43, '22528': 50, '20480': 50, '16384': 50}
# Tokens of the KV pool of all requests together (KV_POOL, the run script's --max-total-tokens), with the packs. The
# server runs 4 requests at a time since 2026-11-12 (an agent and its sub-agents), and in a pool of one context what
# the sub-agents read evicts the main session's cached prompt. 1,673 bytes a token: 0.86 GiB for the 512K tokens above
# a context, one hot slot per layer less while decoding (27.4 against 27.3-27.8 ms a step: no difference), the GPU
# memory peak of a 520K prompt 70,776 against 70,596 MiB (docs/reports/2026-11-12-ds-four-requests.md)
POOL = 1048576


def _common():
    """The options every model has; the context's label and hint say what the KV pool is on this card."""
    options = common_options('DSpark')
    context = next(o for o in options if o['key'] == 'context')
    context.update(label='上下文 · KV 池 1M', hint=(
        '单个请求的上限。所有请求共用的 KV 池固定 1M token（基线配置和上下文一样大）：池子比上下文大，主会话的缓存就不会被子 agent '
        '的提示挤掉。约 37 万 token 的会话在 3 个 8 万 token 的请求之后回来，1M 池缓存全在、首字 1.8 秒；512K 池被挤掉 12 万 token、'
        '要重读 20 秒（11-12 实测）。比 512K 的池多占 0.86 GiB 显存，decode 时每层少 1 个热专家槽，速度没有可测的差别'))
    return options


MODEL = {
    'id': 'ds41', 'order': 30,
    'name': 'DeepSeek V4.1 Flash 1017', 'code': 'DSV4.1-1017', 'port': 18190, 'served': 'DeepSeek-V4.1-Flash-1017',
    'spec': 'DSpark', 'arch': '40 层 · 384 专家 · Engram 189 GB 行放 NVMe · MXFP4',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 2400,
    # README agent speed (production's agent_bench, two rounds, 2026-11-12, v0.1.5: these defaults, compact
    # verification, expert calls of up to 16 tokens): decode of the long responses (>= 300 tokens),
    # cold prefill of the 64K / 500K sessions (mean of the rounds)
    'bench': {'kind': 'agent', 'dec': '156.9', 'pre': '5822 / 6674',
              'title': 'agent 速度（生产的 agent_bench，两轮）：decode = 长回复（≥ 300 token），prefill = 64K / 500K 会话的冷 prefill'},
    'options': _common() + [
        # compact since 2026-11-12, with 4 running requests: a step's time goes by the rows verified in all (a tier of
        # 6 rows each), so with several requests it pays to verify fewer tokens a request
        {'key': 'verify', 'label': 'DSpark 验证', 'default': 'compact',
         'choices': [['compact', '变长（compact）· 实测 · 本机成本表'], ['static', '固定（static）']],
         'hint': '变长：按本机实测的成本表（sps-table.json，测到 4 个请求）决定每个请求验证几个草稿 token，再补满到图的档位。每步时间按验证的总行数分档，'
                 '请求多的时候少验更快：4 个请求同时合计 254 对固定的 212 tok/s，2 个请求和单请求两者差不多（11-12 实测）。照抄 0 错。'
                 '换机器要用 tools/sps_profile.py 重测这张表，没有重测时选固定',
         'off_when': [{'spec': 'off'}]},
        {'key': 'engram', 'label': 'Engram 行缓存', 'default': '4',
         'choices': [['4', '4 GiB · 实测'], ['8', '8 GiB'], ['16', '16 GiB']]},
        elastic_option('on', 'decode 时多 13–15 槽 / 层（至多 68），agent 每步 −4%（10-26）；超过约 66 槽 GPU 一侧成瓶颈'),
        {'key': 'slots', 'label': '热专家槽 / 层', 'default': '35',
         'choices': [['35', '35 · 实测 · 配 48K 分块'], ['43', '43 · 配 32K 分块'], ['47', '47'], ['50', '50 · 配 22K 及以下分块']],
         'hint': '读长提示时每层常备的槽数；decode 时弹性热专家把分块用的显存还回来，补到 68 槽',
         'off_when': [{'packs': 'off'}]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '49152',
         'choices': [['49152', '48K · 实测'], ['32768', '32K'], ['22528', '22K'], ['20480', '20K'], ['16384', '16K']],
         'hint': '每读一块都要把各层不在热专家槽里的专家传上显卡，约 2.5 秒，和块的大小无关；块越大，这段时间摊给越多 token：'
                 '48K 比 22K 冷 prefill 快约 30%，decode 不变（11-11 实测）。一块的 forward 要的显存从热专家槽里让：'
                 '48K 配 35 槽、32K 配 43 槽、22K 配 50 槽', 'off_when': [{'packs': 'off'}]},
        {'key': 'cold', 'label': '冷专家卸载', 'default': 'off',
         'choices': [['off', '关'], ['10', '10% · 27 GiB 放硬盘'], ['20', '20% · 54 GiB 放硬盘'], ['30', '30% · 81 GiB 放硬盘']],
         'hint': '内存只放 70–90% 的专家，其余放在 4 块 NVMe 上、用到时读回（模拟内存不够；11-01 实测见 docs/reports）',
         'unavailable': lambda: {} if all(Path(p).is_dir() for p in COLD_PATHS)
         else {c: '本机没有冷专家盘目录' for c in ('10', '20', '30')}},
        miss_option([{'cold': c} for c in ('10', '20', '30')]),   # not on layers with cold experts
        # 2026-11-11: the cold prefill of a 128K prompt +3% at 48K chunks, of 64K / 520K prompts +10% / +7% at 22K
        # chunks, decode unchanged; on by default since then. Not on layers with cold experts
        share_option('on', '读长提示时 CPU 是空的：路由少的专家块不传上显卡，交给 CPU 同时算。48K 分块下冷 prefill 约快 3%，'
                           '22K 分块下快 7–10%（11-11 实测），decode 不变。CPU 算的那部分用 BF16 激活（GPU 是 MXFP8），'
                           '输出和关着时有微小差别', [{'cold': c} for c in ('10', '20', '30')]),
        # 2026-11-11 at 48K chunks, a 227K-token cold prompt beside a decoding request (158 tok/s alone), 72 steps
        # measured: the decoding request keeps 20% of its speed and waits 7.4 s at the longest, the prompt's first
        # token comes after 44.5 s against 37.4 s alone (484K tokens: 96.0 against 74.3 s). 48 steps (the default of
        # 22K chunks) left 15% at 48K chunks. 24 and 224 from the measured cost of a step between chunks
        interleave_option([['72', '72 步 / 块 · 实测 · decode 约 32 tok/s，长提示首字晚 19–29%'],
                           ['24', '24 步 / 块 · 估 decode 约 12 tok/s，长提示首字晚约 6–10%'],
                           ['224', '224 步 / 块 · 估 decode 约 70 tok/s，长提示首字晚约 60–90%']], '72',
                          '一个请求在读长提示时，每读完一块（48K token，约 7.5 秒）先让另一个请求 decode 这么多步；'
                          '没有请求在 decode 时不生效。关：另一个请求要等整段读完（227K token 要 37 秒）。'
                          '步数按 48K 分块定的：分块调小时同样的步数给另一个请求的份额更大'),
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '基线：Triton 块 FP8、不开 Engram 提前查表与草稿词表头 FP8；22K 分块、50 槽'},
              *[{'when': {'packs': 'on', 'chunk': c, 'slots': s}, 'warn': True,
                 'text': f'{int(c) // 1024}K 分块的显存只够 {n} 槽：按 {n} 槽启动'}
                for c, n in CHUNK_SLOTS.items() for s in ('43', '47', '50') if int(s) > n],
              *[{'when': {'packs': 'on', 'elastic': 'off', 'chunk': c}, 'warn': True,
                 'text': f'弹性热专家关着：decode 只有 {CHUNK_SLOTS[c]} 槽（开着补到 68），会变慢；要关弹性请用 22K 分块配 50 槽'}
                for c in ('49152', '32768')],
              {'when': {'cold': '10'}, 'warn': True,
               'text': '冷专家 10%：27 GiB 专家放 4 块 NVMe；实测 decode 约 −5%，长 prefill 约 −12%，增量 prefill 首字约 1.1 s（不卸载 0.8 s）'},
              {'when': {'cold': '20'}, 'warn': True,
               'text': '冷专家 20%：54 GiB 专家放 4 块 NVMe；实测 decode 约 −13%，长 prefill 约 −13%，增量 prefill 首字约 1.1 s（不卸载 0.8 s）'},
              {'when': {'cold': '30'}, 'warn': True,
               'text': '冷专家 30%：81 GiB 专家放 4 块 NVMe；实测 decode 约 −21%，长 prefill 约 −13%，增量 prefill 首字约 1.3–1.7 s（不卸载 0.8 s）'}],
}


def _pool(o):
    """Tokens of the KV pool: POOL with the packs (a context at least), one context for the baseline."""
    context = int(o['context'])
    return context if o['packs'] == 'off' else max(context, POOL)


def build(o, port):
    """(environment additions, argv) for validated options o."""
    base = o['packs'] == 'off'
    chunk = '22528' if base else o['chunk']   # the baseline: production's chunks and slots, no elastic cache
    slots = '50' if base else str(min(int(o['slots']), CHUNK_SLOTS[chunk]))
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'DSPARK': '1' if o['spec'] == 'on' else '0',
           'CONTEXT': o['context'], 'KV_POOL': str(_pool(o)), 'BATCH': chunk, 'XTIER_CACHE_SLOTS': slots,
           'XTIER_ENGRAM_CACHE_GIB': o['engram']}
    if base:   # README "基线": Triton block FP8 without tuned configs, no prefetch / FP8 draft head
        env.update(XTIER_FP8_CONFIGS='0', FP8_GEMM='auto', XTIER_MXFP8_B12X_ROWS='', XTIER_ENGRAM_PREFETCH='0',
                   XTIER_DRAFT_LMHEAD_FP8='0')
    elif o['elastic'] == 'on':   # past ~66 slots the GPU side of a layer is the slower one (planner, 10-26)
        env.update(XTIER_CACHE_ELASTIC='1', XTIER_CACHE_ELASTIC_MAX_SLOTS='68')
    if o['spec'] == 'on' and o['verify'] == 'compact':   # budgeted by sps-table.json (run-ds-xtier.sh)
        env['SGLANG_RAGGED_VERIFY_MODE'] = 'compact'
    if o['cold'] != 'off':   # the store holds every expert: written at the first start, reused after
        env.update(XTIER_COLD='1', XTIER_COLD_PATHS=':'.join(COLD_PATHS), XTIER_COLD_FRACTION=f"0.{o['cold']}")
    else:   # neither on layers with cold experts
        env.update(miss_env(o))
        env.update(share_env(o))
    return env, [str(HERE / 'run-ds-xtier.sh'), *interleave_args(o)]


def tags(o):
    """Extra parts of the server's label."""
    out = ['弹性热专家'] if o['packs'] == 'on' and o['elastic'] == 'on' else []
    if o['spec'] == 'on' and o['verify'] == 'compact':
        out.append('变长验证')
    if o['cold'] != 'off':
        out.append(f"冷专家 {o['cold']}%")
    elif miss_env(o):
        out.append(f"未命中 {o['missgpu']}% GPU 代算")
    if o['cold'] == 'off':
        out += share_tags(o)
    return out + interleave_tags(o)
