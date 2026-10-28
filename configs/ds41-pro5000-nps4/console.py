"""XTier console entry (xtier/console): DeepSeek V4.1 Flash 1017 on one RTX PRO 5000, NPS4 host. Option defaults = the
README "+ 加速包" row: 50 hot slots per layer, Engram rows on NVMe with a 4 GiB row cache, 22K chunks, DSpark."""

from pathlib import Path

from xtier.console.core import common_options, elastic_option, miss_env, miss_option

HERE = Path(__file__).resolve().parent
# the cold-expert store (XTIER_COLD): Optane x2, 990 PRO, FX700, in this order (the manifest is in the first)
COLD_PATHS = ['/data/fast/p5800x1/xtier-cold', '/data/fast/p5800x2/xtier-cold', '/data/fast/nvme1/xtier-cold',
              '/data/fast/ds41/xtier-cold']

MODEL = {
    'id': 'ds41', 'order': 30,
    'name': 'DeepSeek V4.1 Flash 1017', 'code': 'DSV4.1-1017', 'port': 18190, 'served': 'DeepSeek-V4.1-Flash-1017',
    'spec': 'DSpark', 'arch': '40 层 · 384 专家 · Engram 189 GB 行放 NVMe · MXFP4',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 2400,
    # README agent speed (production's agent_bench, two rounds, 2026-10-26 with the elastic expert cache): decode of the
    # long responses (>= 300 tokens), cold prefill of the 64K / 500K sessions (mean of the rounds)
    'bench': {'kind': 'agent', 'dec': '141.3', 'pre': '4420 / 4806',
              'title': 'agent 速度（生产的 agent_bench，两轮）：decode = 长回复（≥ 300 token），prefill = 64K / 500K 会话的冷 prefill'},
    'options': common_options('DSpark') + [
        {'key': 'engram', 'label': 'Engram 行缓存', 'default': '4',
         'choices': [['4', '4 GiB · 实测'], ['8', '8 GiB'], ['16', '16 GiB']]},
        elastic_option('on', 'decode 时多 13–15 槽 / 层（至多 68），agent 每步 −4%（10-26）；超过约 66 槽 GPU 一侧成瓶颈'),
        {'key': 'slots', 'label': '热专家槽 / 层', 'default': '50',
         'choices': [['50', '50 · 实测'], ['47', '47'], ['43', '43']]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '22528',
         'choices': [['22528', '22K'], ['20480', '20K'], ['16384', '16K']]},
        {'key': 'cold', 'label': '冷专家卸载', 'default': 'off',
         'choices': [['off', '关'], ['10', '10% · 29 GB 放硬盘'], ['20', '20% · 58 GB 放硬盘'], ['30', '30% · 87 GB 放硬盘']],
         'hint': '内存只放 70–90% 的专家，其余放在 4 块 NVMe 上、用到时读回（模拟内存不够；10-26 实测见 docs/reports）',
         'unavailable': lambda: {} if all(Path(p).is_dir() for p in COLD_PATHS)
         else {c: '本机没有冷专家盘目录' for c in ('10', '20', '30')}},
        miss_option([{'cold': c} for c in ('10', '20', '30')]),   # not on layers with cold experts
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '官方 main 基线：Triton 块 FP8、不开 Engram 提前查表与草稿词表头 FP8'},
              {'when': {'cold': '10'}, 'warn': True,
               'text': '冷专家 10%：29 GB 专家放 4 块 NVMe；实测 decode 每步约 −7%，长 prefill 约 −2%，增量 prefill 持平'},
              {'when': {'cold': '20'}, 'warn': True,
               'text': '冷专家 20%：58 GB 专家放 4 块 NVMe；实测 decode 约 −19%，长 prefill 约 −3%，增量 prefill 约 −14%'},
              {'when': {'cold': '30'}, 'warn': True,
               'text': '冷专家 30%：87 GB 专家放 4 块 NVMe；实测 decode 约 −31%，长 prefill 约 −10%，增量 prefill 约 −27%'}],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'DSPARK': '1' if o['spec'] == 'on' else '0',
           'CONTEXT': o['context'], 'BATCH': o['chunk'], 'XTIER_CACHE_SLOTS': o['slots'],
           'XTIER_ENGRAM_CACHE_GIB': o['engram']}
    if o['packs'] == 'off':   # README "官方 main": Triton block FP8 without tuned configs, no prefetch / FP8 draft head
        env.update(XTIER_FP8_CONFIGS='0', FP8_GEMM='auto', XTIER_MXFP8_B12X_ROWS='', XTIER_ENGRAM_PREFETCH='0',
                   XTIER_DRAFT_LMHEAD_FP8='0')
    elif o['elastic'] == 'on':   # past ~66 slots the GPU side of a layer is the slower one (planner, 10-26)
        env.update(XTIER_CACHE_ELASTIC='1', XTIER_CACHE_ELASTIC_MAX_SLOTS='68')
    if o['cold'] != 'off':   # the store holds every expert: written at the first start, reused after
        env.update(XTIER_COLD='1', XTIER_COLD_PATHS=':'.join(COLD_PATHS), XTIER_COLD_FRACTION=f"0.{o['cold']}")
    else:
        env.update(miss_env(o))
    return env, [str(HERE / 'run-ds-xtier.sh')]


def tags(o):
    """Extra parts of the server's label."""
    out = ['弹性热专家'] if o['packs'] == 'on' and o['elastic'] == 'on' else []
    if o['cold'] != 'off':
        out.append(f"冷专家 {o['cold']}%")
    elif miss_env(o):
        out.append(f"未命中 {o['missgpu']}% GPU 代算")
    return out
