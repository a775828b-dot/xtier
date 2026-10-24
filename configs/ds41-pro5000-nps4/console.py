"""XTier console entry (xtier/console): DeepSeek V4.1 Flash 1017 on one RTX PRO 5000, NPS4 host. Option defaults = the
README "+ 加速包" row: 50 hot slots per layer, Engram rows on NVMe with a 4 GiB row cache, 22K chunks, DSpark."""

from pathlib import Path

from xtier.console.core import common_options

HERE = Path(__file__).resolve().parent

MODEL = {
    'id': 'ds41', 'order': 30,
    'name': 'DeepSeek V4.1 Flash 1017', 'code': 'DSV4.1-1017', 'port': 18190, 'served': 'DeepSeek-V4.1-Flash-1017',
    'spec': 'DSpark', 'arch': '40 层 · 384 专家 · Engram 189 GB 行放 NVMe · MXFP4',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 2400,
    # README agent speed (production's agent_bench, two rounds): decode of the long responses (>= 300 tokens),
    # cold prefill of the 64K / 500K sessions (mean of the rounds)
    'bench': {'kind': 'agent', 'dec': '133.2', 'pre': '4109 / 4791',
              'title': 'agent 速度（生产的 agent_bench，两轮）：decode = 长回复（≥ 300 token），prefill = 64K / 500K 会话的冷 prefill'},
    'options': common_options('DSpark') + [
        {'key': 'engram', 'label': 'Engram 行缓存', 'default': '4',
         'choices': [['4', '4 GiB · 实测'], ['8', '8 GiB'], ['16', '16 GiB']]},
        {'key': 'slots', 'label': '热专家槽 / 层', 'default': '50',
         'choices': [['50', '50 · 实测'], ['47', '47'], ['43', '43']]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '22528',
         'choices': [['22528', '22K'], ['20480', '20K'], ['16384', '16K']]},
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '官方 main 基线：Triton 块 FP8、不开 Engram 提前查表与草稿词表头 FP8'}],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'DSPARK': '1' if o['spec'] == 'on' else '0',
           'CONTEXT': o['context'], 'BATCH': o['chunk'], 'XTIER_CACHE_SLOTS': o['slots'],
           'XTIER_ENGRAM_CACHE_GIB': o['engram']}
    if o['packs'] == 'off':   # README "官方 main": Triton block FP8 without tuned configs, no prefetch / FP8 draft head
        env.update(XTIER_FP8_CONFIGS='0', FP8_GEMM='auto', XTIER_MXFP8_B12X_ROWS='', XTIER_ENGRAM_PREFETCH='0',
                   XTIER_DRAFT_LMHEAD_FP8='0')
    return env, [str(HERE / 'run-ds-xtier.sh')]
