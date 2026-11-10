"""XTier console entry (xtier/console): DeepSeek V4.1 Flash 1017 on one RTX PRO 5000, NPS4 host. Option defaults = the
README "+ 加速包" row: 50 hot slots per layer, Engram rows on NVMe with a 4 GiB row cache, 22K chunks, DSpark."""

from pathlib import Path

from xtier.console.core import (common_options, elastic_option, interleave_args, interleave_option, interleave_tags,
                                miss_env, miss_option)

HERE = Path(__file__).resolve().parent
# the cold-expert store (XTIER_COLD): four PCIe 5.0 drives (the manifest is in the first)
COLD_PATHS = [f'/data/fast/cold{i}/xtier-cold' for i in (1, 2, 3, 4)]

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
        {'key': 'verify', 'label': 'DSpark 验证', 'default': 'static',
         'choices': [['static', '固定（static）'], ['compact', '变长（compact）· 本机实测成本表']],
         'hint': '变长：按本机实测的成本表（sps-table.json）决定每个请求验证几个草稿 token；单请求和固定一样，两个请求同时约快 17%'
                 '（11-06 实测，照抄 0 错）。换机器要用 tools/sps_profile.py 重测这张表',
         'off_when': [{'spec': 'off'}]},
        {'key': 'engram', 'label': 'Engram 行缓存', 'default': '4',
         'choices': [['4', '4 GiB · 实测'], ['8', '8 GiB'], ['16', '16 GiB']]},
        elastic_option('on', 'decode 时多 13–15 槽 / 层（至多 68），agent 每步 −4%（10-26）；超过约 66 槽 GPU 一侧成瓶颈'),
        {'key': 'slots', 'label': '热专家槽 / 层', 'default': '50',
         'choices': [['50', '50 · 实测'], ['47', '47'], ['43', '43']]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '22528',
         'choices': [['22528', '22K'], ['20480', '20K'], ['16384', '16K']]},
        {'key': 'cold', 'label': '冷专家卸载', 'default': 'off',
         'choices': [['off', '关'], ['10', '10% · 27 GiB 放硬盘'], ['20', '20% · 54 GiB 放硬盘'], ['30', '30% · 81 GiB 放硬盘']],
         'hint': '内存只放 70–90% 的专家，其余放在 4 块 NVMe 上、用到时读回（模拟内存不够；11-01 实测见 docs/reports）',
         'unavailable': lambda: {} if all(Path(p).is_dir() for p in COLD_PATHS)
         else {c: '本机没有冷专家盘目录' for c in ('10', '20', '30')}},
        miss_option([{'cold': c} for c in ('10', '20', '30')]),   # not on layers with cold experts
        # 2026-11-10, a 214K-token cold prompt beside a decoding request (143 tok/s alone): the decoding request's
        # speed while the prompt is read, and the prompt's time to its first token against 45.4 s without
        interleave_option([['48', '48 步 / 块 · 实测 · decode 约 32 tok/s，长提示首字晚 27%'],
                           ['16', '16 步 / 块 · decode 约 14 tok/s，长提示首字晚 8%'],
                           ['150', '150 步 / 块 · decode 约 67 tok/s，长提示首字晚 86%']], '48',
                          '一个请求在读长提示时，每读完一块（22K token，约 4.6 秒）先让另一个请求 decode 这么多步；'
                          '没有请求在 decode 时不生效。关：另一个请求要等整段读完（214K token 实测停 40 秒）'),
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '基线：Triton 块 FP8、不开 Engram 提前查表与草稿词表头 FP8'},
              {'when': {'cold': '10'}, 'warn': True,
               'text': '冷专家 10%：27 GiB 专家放 4 块 NVMe；实测 decode 约 −5%，长 prefill 约 −12%，增量 prefill 首字约 1.1 s（不卸载 0.8 s）'},
              {'when': {'cold': '20'}, 'warn': True,
               'text': '冷专家 20%：54 GiB 专家放 4 块 NVMe；实测 decode 约 −13%，长 prefill 约 −13%，增量 prefill 首字约 1.1 s（不卸载 0.8 s）'},
              {'when': {'cold': '30'}, 'warn': True,
               'text': '冷专家 30%：81 GiB 专家放 4 块 NVMe；实测 decode 约 −21%，长 prefill 约 −13%，增量 prefill 首字约 1.3–1.7 s（不卸载 0.8 s）'}],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'DSPARK': '1' if o['spec'] == 'on' else '0',
           'CONTEXT': o['context'], 'BATCH': o['chunk'], 'XTIER_CACHE_SLOTS': o['slots'],
           'XTIER_ENGRAM_CACHE_GIB': o['engram']}
    if o['packs'] == 'off':   # README "基线": Triton block FP8 without tuned configs, no prefetch / FP8 draft head
        env.update(XTIER_FP8_CONFIGS='0', FP8_GEMM='auto', XTIER_MXFP8_B12X_ROWS='', XTIER_ENGRAM_PREFETCH='0',
                   XTIER_DRAFT_LMHEAD_FP8='0')
    elif o['elastic'] == 'on':   # past ~66 slots the GPU side of a layer is the slower one (planner, 10-26)
        env.update(XTIER_CACHE_ELASTIC='1', XTIER_CACHE_ELASTIC_MAX_SLOTS='68')
    if o['spec'] == 'on' and o['verify'] == 'compact':   # budgeted by sps-table.json (run-ds-xtier.sh)
        env['SGLANG_RAGGED_VERIFY_MODE'] = 'compact'
    if o['cold'] != 'off':   # the store holds every expert: written at the first start, reused after
        env.update(XTIER_COLD='1', XTIER_COLD_PATHS=':'.join(COLD_PATHS), XTIER_COLD_FRACTION=f"0.{o['cold']}")
    else:
        env.update(miss_env(o))
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
    return out + interleave_tags(o)
