"""XTier console entry (xtier/console): DeepSeek V4.1 Flash 1017 the smallest way, as on an RTX 5090 (32 GB) with
128 GB of host memory and 4 NVMe drives, simulated on the RTX PRO 5000 host (run-ds-mini.sh): the GPU memory above
32000 MiB is held by tools/vram_limit.py and the server's systemd scope gets MemoryMax=128G without swap, so the run,
load included, stays under both or fails. 70 % of every layer's experts live in the cold store on the 4 drives (the
DS card's store, reused), 4 resident hot slots per layer and the elastic cache for decode, a 128K context, DSpark.
Option defaults = the measured configuration (results: docs/reports/2026-10-27-ds41-minimal.md)."""

from pathlib import Path

from xtier.console.core import common_options, elastic_option

HERE = Path(__file__).resolve().parent
# the cold-expert store of the DS card (XTIER_COLD): Optane x2, 990 PRO, FX700, in this order (the manifest is in the first)
COLD_PATHS = ['/data/fast/p5800x1/xtier-cold', '/data/fast/p5800x2/xtier-cold', '/data/fast/nvme1/xtier-cold',
              '/data/fast/ds41/xtier-cold']

MODEL = {
    'id': 'ds41mini', 'order': 31,
    'name': 'DeepSeek V4.1 Flash 1017 · 最小化', 'code': 'DSV4.1-MINI', 'port': 18191, 'served': 'DeepSeek-V4.1-Flash-1017',
    'spec': 'DSpark', 'arch': '40 层 · 384 专家 · 70% 专家放 4 块 NVMe（冷专家）· MXFP4 · 128K',
    'hardware': '模拟 RTX 5090 32 GB + 128 GB 内存 + 4 NVMe（显存 / 内存硬上限）', 'ready_timeout': 2400,
    # tp-sim/mini_speed.py on two starts (10-27): cold prefill of 32K / 128K prompts after a flushed prefix cache (the
    # second round of each start), decode of 512 tokens after them (every round, 32K / 64K / 128K averaged)
    'bench': {'kind': 'text', 'dec': '29.8', 'pre': '1186 / 1155',
              'title': '文本测速（32K / 64K / 128K 冷 prefill 后各 decode 512 token，两次启动）：decode = 平均，prefill = 32K / 128K'},
    'options': [o for o in common_options('DSpark') if o['key'] != 'context'] + [   # 128K, no context option
        {'key': 'cold', 'label': '冷专家卸载', 'default': '70', 'choices': [['70', '70% · 实测']],
         'hint': '内存里每层只放 30% 的专家（约 81 GiB），其余从 4 块 NVMe 读回；内存与显存上限固定（模拟 5090 + 128 GB）'},
        {'key': 'slots', 'label': '常驻热专家槽 / 层', 'default': '4', 'choices': [['4', '4 · 实测']]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '8192', 'choices': [['8192', '8K · 实测'], ['4096', '4K']]},
        elastic_option('on', 'decode 时把 prefill 才用的显存借给热专家（32 GB 里常驻只放 4 槽），prefill 前还回'),
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '官方 main 基线：Triton 块 FP8、不开 Engram 提前查表与草稿词表头 FP8'}],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'DSPARK': '1' if o['spec'] == 'on' else '0',
           'CONTEXT': '131072', 'BATCH': o['chunk'], 'XTIER_CACHE_SLOTS': o['slots'], 'XTIER_ENGRAM_CACHE_GIB': '2',
           'MEM': '0.95', 'XTIER_COLD': '1', 'XTIER_COLD_PATHS': ':'.join(COLD_PATHS),
           'XTIER_COLD_FRACTION': f"0.{o['cold']}", 'XTIER_SIM_VRAM_MIB': '32000', 'XTIER_SIM_HOST_MAX': '128G'}
    if o['packs'] == 'off':   # README "官方 main": Triton block FP8 without tuned configs, no prefetch / FP8 draft head
        env.update(XTIER_FP8_CONFIGS='0', FP8_GEMM='auto', XTIER_MXFP8_B12X_ROWS='', XTIER_ENGRAM_PREFETCH='0',
                   XTIER_DRAFT_LMHEAD_FP8='0')
    elif o['elastic'] == 'on':
        env.update(XTIER_CACHE_ELASTIC='1', XTIER_CACHE_ELASTIC_MAX_SLOTS='30')
    return env, [str(HERE / 'run-ds-mini.sh')]


def tags(o):
    """Extra parts of the server's label."""
    return ['128K', '模拟 5090 32G / 内存 128G', f"冷专家 {o['cold']}%"] + (
        ['弹性热专家'] if o['packs'] == 'on' and o['elastic'] == 'on' else [])
