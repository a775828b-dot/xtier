"""XTier console entry (xtier/console): MiniMax-M3.1-Flash on one RTX PRO 5000, NPS4 host, on SGLang's MiniMax M3.1
branch (PR #45925; venv env-m31, see run-minimax-xtier.sh). The context is fixed at 512K (an NVFP4 KV pool of 19 GiB
next to 8 hot slots per layer); option defaults = the measured configuration."""

from pathlib import Path

from xtier.console.core import common_options

HERE = Path(__file__).resolve().parent

MODEL = {
    'id': 'minimax31', 'order': 40,
    'name': 'MiniMax M3.1 Flash', 'code': 'MINIMAX-M3.1-F', 'port': 18195, 'served': 'MiniMax-M3.1-Flash',
    'spec': 'DSpark', 'arch': '60 层 · 128 专家 top-4 + 共享专家 · MSA 稀疏注意力 · NVFP4 专家 / MXFP8',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 3600,
    # README agent speed (production's agent_bench, 64k session, two rounds): decode of the long responses
    # (>= 300 tokens, both rounds), the 64K cold prefill (mean of the rounds)
    'bench': {'kind': 'agent', 'dec': '51.6', 'pre': '1866',
              'title': 'agent 速度（生产的 agent_bench 64k 会话，两轮）：decode = 长回复（≥ 300 token），prefill = 64K 会话的冷 prefill'},
    'options': [o for o in common_options('DSpark') if o['key'] != 'context'] + [   # 512K, no context option
        {'key': 'slots', 'label': '热专家槽 / 层', 'default': '8', 'choices': [['8', '8 · 实测']]},
        {'key': 'gamma', 'label': 'DSpark 草稿长度', 'default': '5',
         'choices': [['5', '5 · 实测'], ['7', '7 · 检查点'], ['3', '3']]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '16384', 'choices': [['16384', '16K'], ['8192', '8K']]},
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '官方 main 基线：只保留在 SM120 上跑起来必需的修正，草稿长度用检查点的 7'}],
}


def tags(o):
    """The fixed context in job labels (there is no context option)."""
    return ['512K']


def build(o, port):
    """(environment additions, argv) for validated options o."""
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'CONTEXT': '524288', 'BATCH': o['chunk'],
           'XTIER_CACHE_SLOTS': o['slots'], 'DSPARK': '1' if o['spec'] == 'on' else '0', 'GAMMA': o['gamma']}
    if o['packs'] == 'off':   # README "官方 main": the SM120 fixes only (packs/minimax_m3), the checkpoint's draft length
        env.update(XTIER_DRAFT_LMHEAD_FP8='0', XTIER_MXFP8_B12X_ROWS='', GAMMA='7', XTIER_DSPARK_TRUNCATE='0')
    return env, [str(HERE / 'run-minimax-xtier.sh')]
