"""XTier console entry (xtier/console): Kimi K3.1 Pro (2.8T: 92 MoE layers of 896 routed MXFP4 experts, top-16) on one
RTX PRO 5000, NPS4 host, the experiment of branch kimi-k31 on SGLang's kimi-k3.1-support branch (venv env-k31, see
run-kimi-xtier.sh): 78 % of every layer's experts only in the cold store on six drives, DFlash2 block 5, a 96K context
and the route tail skip at R = 9 (XTIER_TAIL_SKIP, changes the output: KL 0.024 to no skip, between Q5_K_M and Q4_K_M
of llama.cpp's Llama 3 8B table). Option defaults = the measured configuration (docs/reports/2026-11-05-kimi-k31.md)."""

from pathlib import Path

from xtier.console.core import common_options

HERE = Path(__file__).resolve().parent

MODEL = {
    'id': 'kimi31', 'order': 35,
    'name': 'Kimi K3.1 Pro · 实验', 'code': 'KIMI-K3.1-PRO', 'port': 18200, 'served': 'Kimi-K3.1-Pro',
    'spec': 'DFlash2', 'arch': '92 层 · 896 专家 top-16 · 78% 专家放 6 块 NVMe（冷专家）· MXFP4 · 96K',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4 · 4×9100 PRO + 2×P5800X', 'ready_timeout': 2400,
    # kimi_code_bench.py (11-05): three code prompts with thinking off, greedy, after three warm-ups; decode = their
    # tokens / their decode time; prefill = a 12K prompt
    'bench': {'kind': 'text', 'dec': '10.3', 'pre': '91',
              'title': '代码题测速（三道代码题，关思考，贪心）：decode = 三题合计，prefill = 12K 提示词'},
    # no packs option (the kimi_k3 pack is what runs DSA on SM120) and no context option (96K)
    'options': [o for o in common_options('DFlash2') if o['key'] in ('spec', 'threads')] + [
        {'key': 'tail', 'label': '尾部跳过', 'default': '9',
         'choices': [['9', 'R = 9 · 实测'], ['8', 'R = 8 · 试用'], ['10', 'R = 10'], ['11', 'R = 11'], ['0', '关']],
         'hint': '不在内存、每个 token 都排在第 R 名以后的专家不读不算，会改变输出。代码题 decode：R = 9 10.3、'
                 'R = 10 9.4、R = 11 8.3、关 4.9 tok/s，R = 8 没实测（估约 11）；精度：不跳过好比 Q8，R = 9 在 Q5_K_M '
                 '和 Q4_K_M 之间，R = 8 估计约等于 Q4_K_M'},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '16384',
         'choices': [['16384', '16K · 固定热专家槽 1 个'], ['8192', '8K · 固定热专家槽 4 个']],
         'hint': 'prefill 每层要把 896 个专家分批上传、反量化，开销按专家数算，块越大分摊越多。16K 一块临时要约 10 G 显存，'
                 '所以固定热专家槽从 4 个降到 1 个（decode 时弹性热专家会把空出的显存用上）'},
    ],
}


def tags(o):
    """The fixed context and the store in job labels."""
    return ['96K', f"{int(o['chunk']) // 1024}K 分块", '冷专家 78%', f"尾部跳过 R = {o['tail']}" if o['tail'] != '0' else '不跳过']


def build(o, port):
    """(environment additions, argv) for validated options o."""
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'DFLASH': '1' if o['spec'] == 'on' else '0',
           'XTIER_TAIL_SKIP': o['tail'], 'BATCH': o['chunk'], 'CACHE_SLOTS': '1' if o['chunk'] == '16384' else '4'}
    return env, [str(HERE / 'run-kimi-xtier.sh')]
