"""XTier console entry (xtier/console): Qwen3.8 Flash Next on one RTX PRO 5000, NPS4 host. Option defaults = the README
"+ 加速包" row (2026-10-24: NVFP4 KV, 24K chunks, MTP) with the planner's layout (2026-10-25): the memory of 26 resident
layers + 22 x 155 hot slots laid out at startup (XTIER_CACHE_PLAN=auto: 29 resident + 18 x 64-88; agent_bench cold
prefill +2.4-2.9%, decode unchanged, docs/reports/2026-10-24-planner.md)."""

from pathlib import Path

from xtier.console.core import common_options

HERE = Path(__file__).resolve().parent
PLE_OPTANE = '/data/fast/p5800x1/qwen38-flash-next-ple'   # the PLE table's rows on an Optane drive (XTIER_PLE_ROWS=nvme)
YARN = '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'
RESIDENT = '0,1,2,5,6,7,8,10,11,12,13,14,17,18,19,20,21,22,26,28,29,33,34,37,38'   # production's 25 resident layers
# r26: layer 27 too, next by hot-slot coverage, on the memory NVFP4 KV frees (64K / 512K prefill +1.4% / +0.5%);
# auto: the planner lays the memory of r26 out at startup (run-qwen-xtier.sh LAYOUT=auto, planner-probe.json)
LAYOUTS = {'r26': RESIDENT + ',27', 'prod': RESIDENT, 'auto': RESIDENT + ',27'}

MODEL = {
    'id': 'qwen38', 'order': 10,
    'name': 'Qwen3.8 Flash Next', 'code': 'QWEN-3.8-FN', 'port': 18180, 'served': 'Qwen3.8-Flash-Next',
    'spec': 'MTP', 'arch': '48 层 · 512 专家 top-10 · QSA + GDN 混合注意力 · NVFP4 W4A16',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 1800,
    # README "+ 加速包" row, text speed: decode / cold prefill tok/s at 64K / 512K
    # README agent speed (production's agent_bench, two rounds): decode of the long responses (>= 300 tokens),
    # cold prefill of the 64K / 500K sessions (mean of the rounds)
    'bench': {'kind': 'agent', 'dec': '182.3', 'pre': '5427 / 5052',
              'title': 'agent 速度（生产的 agent_bench，两轮）：decode = 长回复（≥ 300 token），prefill = 64K / 500K 会话的冷 prefill'},
    'options': common_options('MTP') + [
        {'key': 'kv', 'label': 'KV 缓存', 'default': 'nvfp4', 'choices': [['nvfp4', 'NVFP4 · 与生产同参'], ['fp8', 'FP8']],
         'off_when': [{'packs': 'off'}]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '24576',
         'choices': [['24576', '24K · 实测'], ['32768', '32K'], ['16384', '16K']],
         'hint': '32K 在 64K 冷 prefill 慢 3%；FP8 KV 时显存只够 24K', 'off_when': [{'packs': 'off'}]},
        {'key': 'layout', 'label': '热专家布局', 'default': 'auto',
         'choices': [['auto', '自动规划 · 29 层常驻 + 18×64–88 槽 · 实测'], ['r26', '26 层常驻 + 22×155 槽'],
                     ['prod', '25 层常驻 + 23×155 槽 · 生产布局']],
         'hint': '自动规划：启动时由规划器在 26 层常驻那份显存内重排（agent 冷 prefill +2.4–2.9%，decode 持平）；FP8 KV 时用生产布局',
         'off_when': [{'packs': 'off'}, {'kv': 'fp8'}]},
        {'key': 'ple', 'label': 'PLE 表 (48 GiB)', 'default': 'pinned',
         'choices': [['pinned', '锁页内存'], ['nvme', '傲腾 NVMe · 省 48 GiB 内存']]},
    ],
    'notes': [
        {'when': {'packs': 'off'}, 'text': '官方 main 基线：FP8 KV、16K 分块、不开 hc_mix / PLE / QSA 分段'},
        {'when': {'packs': 'on', 'kv': 'fp8'}, 'text': 'FP8 KV：用生产布局（25 层常驻）', 'warn': True},
        {'when': {'packs': 'on', 'kv': 'fp8', 'chunk': '32768'}, 'text': '分块按 24K 启动', 'warn': True},
        {'when': {'ple': 'nvme'}, 'text': 'PLE 行从傲腾读取，主机内存省 48 GiB'},
    ],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    spec, base, ctx = o['spec'] == 'on', o['packs'] == 'off', o['context']
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port)}
    kv, chunk = ('fp8_e4m3', '16384') if base else ('nvfp4' if o['kv'] == 'nvfp4' else 'fp8_e4m3', o['chunk'])
    layout = 'prod' if kv == 'fp8_e4m3' else o['layout']
    env.update(LAYOUT='auto' if layout == 'auto' else 'prod', SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN='1',
               XTIER_CACHE_SLOTS='155', XTIER_CACHE_FULL=LAYOUTS[layout])
    if base:   # README "官方 main": no pack kernels; SGLang's QSA cannot read NVFP4
        env.update(XTIER_HC_MIX_OVERLAP='0', XTIER_PLE_BLOCK='0', XTIER_QSA_KEYS_PER_CTA='0', XTIER_QSA_FP4='0')
    elif kv == 'fp8_e4m3' and chunk == '32768':
        chunk = '24576'   # 32K chunks need the memory NVFP4 KV frees
    if o['ple'] == 'nvme':
        env.update(XTIER_PLE_ROWS='nvme', XTIER_PLE_MODEL=PLE_OPTANE)
    args = ['--context-length', ctx, '--max-total-tokens', ctx, '--kv-cache-dtype', kv, '--chunked-prefill-size', chunk,
            '--mem-fraction-static', '0.90', '--max-mamba-cache-size', '16', '--mamba-ssm-dtype', 'bfloat16',
            '--json-model-override-args', YARN]
    if spec:
        return env, [str(HERE / 'run-qwen-mtp.sh'), *args, '--speculative-draft-kv-cache-dtype', 'fp8_e4m3']
    return env, [str(HERE / 'run-qwen-xtier.sh'), *args, '--cuda-graph-max-bs-decode', '2']


def tags(o):
    """Extra parts of the server's label."""
    if o['packs'] != 'on':
        return []
    if o['kv'] != 'nvfp4':
        return ['FP8 KV']
    return ['NVFP4 KV', '自动布局'] if o['layout'] == 'auto' else ['NVFP4 KV']
