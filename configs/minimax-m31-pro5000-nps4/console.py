"""XTier console entry (xtier/console): MiniMax-M3.1-Flash on one RTX PRO 5000, NPS4 host, on SGLang's MiniMax M3.1
branch (PR #45925; venv env-m31, see run-minimax-xtier.sh). Option defaults = the measured configuration: 16 hot
slots per layer fit a 128K KV pool (FP8 KV); a 256K pool leaves room for about 8."""

from pathlib import Path

from xtier.console.core import common_options

HERE = Path(__file__).resolve().parent

MODEL = {
    'id': 'minimax31', 'order': 40,
    'name': 'MiniMax M3.1 Flash', 'code': 'MINIMAX-M3.1-F', 'port': 18195, 'served': 'MiniMax-M3.1-Flash',
    'spec': 'DSpark', 'arch': '60 层 · 128 专家 top-4 + 共享专家 · MSA 稀疏注意力 · NVFP4 专家 / MXFP8',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 3600,
    # README agent speed (production's agent_bench, 64k session only: the 128k-write history does not fit the 128K
    # context; four rounds): decode of the long responses (>= 300 tokens), the 64K cold prefill (medians)
    'bench': {'kind': 'agent', 'dec': '54.3', 'pre': '1606',
              'title': 'agent 速度（生产的 agent_bench 64k 会话，四轮中位数）：decode = 长回复（≥ 300 token），prefill = 64K 会话的冷 prefill（128K 池放不下 128k-write 会话）'},
    'options': common_options('DSpark', contexts=('131072', '262144')) + [
        {'key': 'slots', 'label': '热专家槽 / 层', 'default': '16',
         'choices': [['16', '16 · 128K'], ['8', '8 · 256K'], ['12', '12']]},
        {'key': 'gamma', 'label': 'DSpark 草稿长度', 'default': '5',
         'choices': [['5', '5 · 实测'], ['7', '7 · 检查点'], ['3', '3']]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '16384', 'choices': [['16384', '16K'], ['8192', '8K']]},
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '官方 main 基线：只保留在 SM120 上跑起来必需的修正，草稿长度用检查点的 7'}],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'CONTEXT': o['context'], 'BATCH': o['chunk'],
           'XTIER_CACHE_SLOTS': o['slots'], 'DSPARK': '1' if o['spec'] == 'on' else '0', 'GAMMA': o['gamma']}
    if o['packs'] == 'off':   # README "官方 main": the SM120 fixes only (packs/minimax_m3), the checkpoint's draft length
        env.update(XTIER_DRAFT_LMHEAD_FP8='0', XTIER_MXFP8_B12X_ROWS='', GAMMA='7')
    return env, [str(HERE / 'run-minimax-xtier.sh')]
