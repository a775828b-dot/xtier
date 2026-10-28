"""XTier console entry (xtier/console): GLM-5.3 Flash NVFP4 on one RTX PRO 5000, NPS4 host. Option defaults = the
README "+ 加速包" row: 58 hot slots, FP8 attention / dense weights, 16K chunks, EAGLE with the speculative pack
(hot-vocab rejection sampling, verify truncation), the elastic expert cache. 58, not 60 (2026-10-26): at 60 the cold
16K prefill chunks ran at the edge of the GPU memory (the allocator retried 18 times in two agent_bench rounds, one
cold 255K-token prompt failed); the elastic cache gives decode the two slots back."""

from pathlib import Path

from xtier.console.core import common_options, elastic_option, miss_env, miss_option

HERE = Path(__file__).resolve().parent
GLM_REF = '/data/fast/workspaces/gh-ref/lsglang_glm5.3'   # production's adaptive draft steps and draft token map

MODEL = {
    'id': 'glm53', 'order': 20,
    'name': 'GLM-5.3 Flash', 'code': 'GLM-5.3-F', 'port': 18185, 'served': 'GLM-5.3-Flash-NVFP4',
    'spec': 'EAGLE', 'arch': '42 层 · 288 专家 · DSA 稀疏注意力 · NVFP4 专家 + FP8 稠密层',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 2700,
    # README agent speed (production's agent_bench, two rounds, 2026-10-26: 58 slots with the elastic expert cache):
    # decode of the long responses (>= 300 tokens), cold prefill of the 64K / 500K sessions (mean of the rounds)
    'bench': {'kind': 'agent', 'dec': '97.6', 'pre': '2682 / 2638',
              'title': 'agent 速度（生产的 agent_bench，两轮）：decode = 长回复（≥ 300 token），prefill = 64K / 500K 会话的冷 prefill'},
    'options': common_options('EAGLE') + [
        {'key': 'rs', 'label': '无损拒绝采样', 'default': 'on',
         'choices': [['on', '开 · 热词表 + 置信度截断'], ['off', '关']],
         'hint': '需要加速包和投机解码；官方 main 不允许热词表配拒绝采样',
         'off_when': [{'packs': 'off'}, {'spec': 'off'}]},
        {'key': 'slots', 'label': '热专家槽 / 层', 'default': '58',
         'choices': [['58', '58 · 实测'], ['60', '60 · 冷 prefill 显存贴边'], ['55', '55']],
         'hint': '60 槽时 16K 分块的冷 prefill 贴着显存上限（agent_bench 两轮分配器重试 18 次，一次冷提示服务退出）；'
                 '58 槽重试 1 次、每步持平，decode 时弹性槽补回',
         'off_when': [{'packs': 'off'}]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '16384', 'choices': [['16384', '16K'], ['8192', '8K']]},
        elastic_option('on', 'decode 时每层多约 12–14 槽（prefill 余量约 6.6 GiB），agent 每步 −1.5%（10-26）'),
        miss_option(),
    ],
    'notes': [{'when': {'packs': 'off'}, 'text': '官方 main 基线：BF16 权重、55 槽、不开拒绝采样'}],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    spec, base, ctx = o['spec'] == 'on', o['packs'] == 'off', o['context']
    slots = '55' if base else o['slots']   # BF16 weights: 60 slots leave too little for prefill
    env = {'XTIER_THREADS': o['threads'], 'PORT': str(port), 'XTIER_CACHE_SLOTS': slots, 'MAX_TOTAL_TOKENS': ctx}
    if base:
        env['XTIER_FP8_WEIGHTS'] = '0'
    elif o['elastic'] == 'on':
        env['XTIER_CACHE_ELASTIC'] = '1'
    env.update(miss_env(o))
    args = ['--context-length', ctx, '--chunked-prefill-size', o['chunk'], '--mamba-ssm-dtype', 'bfloat16',
            '--mem-fraction-static', '0.92', '--max-mamba-cache-size', '10', '--cuda-graph-max-bs-decode', '2',
            '--dsa-decode-backend', 'tilelang', '--dsa-prefill-backend', 'tilelang', '--kv-cache-dtype', 'fp8_e4m3']
    if spec:
        env['XTIER_DRAFT'] = '1'
        args += ['--speculative-algorithm', 'EAGLE', '--speculative-num-steps', '3', '--speculative-eagle-topk', '1',
                 '--speculative-num-draft-tokens', '4', '--speculative-adaptive', '--speculative-adaptive-config',
                 f'{GLM_REF}/launch/adaptive-steps-1to5.json', '--speculative-token-map',
                 f'{GLM_REF}/lkqmoe/draft-token-map-49152.pt']
        if o['rs'] == 'on' and not base:
            args.append('--speculative-use-rejection-sampling')
    return env, [str(HERE / 'run-glm-xtier.sh'), *args]


def tags(o):
    """Extra parts of the server's label."""
    on = o['packs'] == 'on'
    return (['拒绝采样'] if on and o['spec'] == 'on' and o['rs'] == 'on' else []) + (
        ['弹性热专家'] if on and o['elastic'] == 'on' else []) + (
        [f"未命中 {o['missgpu']}% GPU 代算"] if miss_env(o) else [])
