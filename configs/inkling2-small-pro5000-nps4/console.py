"""XTier console entry (xtier/console): Inkling-2-Small (594B total / 17B active, 48 MoE layers: 40 NVFP4 + 8 MXFP8,
40 local SWA + 8 global attention, native 5-layer DSpark MTP, multimodal ViT) on one RTX PRO 5000 72 GB, NPS4 host.

48 routed MoE layers (384 experts, H=4608, I=2304) take 350.3 GiB in native precision (256.3 GiB for 40 NVFP4 layers +
94.0 GiB for 8 MXFP8 layers). Plan A+ (default) offloads only the coldest 2% tail (~7 GiB, 8/384 experts/layer, guided
by the bundled multi-domain seed profile) to 4 PCIe 5.0 NVMe drives: 376/384 experts (343 GiB) in RAM (~12-15 GiB avail
headroom >= 8 GiB, 2 min fast-load), 0.2-0.7 cold decode reads/step, and 41.5 GiB free VRAM for elastic hot slots
(base 1, reserve 0.8G -> 45 slots/layer at 70.65/73.4 GiB) and single-chunk 64K prefill (BATCH=65664). Plan B2
(cold=off) places only L47 (1 MXFP8 layer, 11.75 GiB) full-GPU with zero CPU RAM so the remaining 47 layers fit in
384 GiB host RAM (~13-20 GiB avail) without NVMe offload, leaving 31.5 GiB VRAM for 48K prefill and 35 hot slots."""

from pathlib import Path

from xtier.console.core import (common_options, elastic_option, interleave_args, interleave_option, interleave_tags,
                                miss_env, miss_option, share_env, share_option, share_tags)

HERE = Path(__file__).resolve().parent
COLD_PATHS = [f'/data/fast/cold{i}/xtier-cold' for i in (1, 2, 3, 4)]
SEED_PROFILE = HERE / 'inkling2-seed-profile.json'
SWA_FOR_CHUNK = {'16384': '0.05', '32768': '0.08', '49152': '0.10', '65664': '0.14'}


def _common():
    options = common_options('DSpark', contexts=('524288', '262144', '131072'))
    context = next(o for o in options if o['key'] == 'context')
    context.update(label='上下文 · KV 池', hint=(
        '目标 48 层启用 NVFP4 KV（72 B/head，3.56× 压缩，512K 仅约 6.75 GiB）+ MTP 5 层 BF16 KV（约 2.80 GiB）；'
        '省出 ~17.25 GiB 显存全部用于弹性热专家（45 槽/层）与 64K 单块 Prefill'))
    return options


MODEL = {
    'id': 'inkling2', 'order': 40,
    'name': 'Inkling-2 Small', 'code': 'INKLING2-S', 'port': 18195, 'served': 'Inkling-2-Small',
    'spec': 'DSpark', 'arch': '48 层（40 SWA + 8 全局）· 384 专家 top-8 · 40 NVFP4 + 8 MXFP8 · 多模态',
    'hardware': 'RTX PRO 5000 · 2×EPYC 9334 NPS4', 'ready_timeout': 3600, 'guard_gib': 8,
    'bench': {'kind': 'agent', 'dec': '66.2', 'pre': '3835 / 1593',
              'title': '方案 A+（2% 冷专家，45 热槽，12 GiB 可用内存）：64K/500K 文本冷 prefill 3835 / 1593 tok/s，'
                       '短测速 1 路 74.7 / 2 路 78.4，64K 稳态 decode 66.7（42.0 ms/step）；'
                       'agent_bench 两轮 64K–500K 长回复 66.2 tok/s（全部 66.3，接受长度 4.02）'},
    'options': _common() + [
        {'key': 'multimodal', 'label': '多模态视觉塔', 'default': 'on', 'switch': True,
         'choices': [['on', '开 · 文本 + 图像'], ['off', '关 · 纯文本']]},
        {'key': 'cold', 'label': '冷专家卸载', 'default': '02',
         'choices': [['02', '2% · 约 7 GiB 放 NVMe（极限压榨推荐，方案 A+，内存留 ~12 GiB）'],
                     ['08', '8% · 约 28 GiB 放 4 块 NVMe（稳妥内存 ~30 GiB，方案 A）'],
                     ['15', '15% · 约 53 GiB 放硬盘'],
                     ['off', '关 · 仅 L47 全驻 GPU 零内存（方案 B2，48K 分块 + 35 热槽）']],
         'hint': '方案 A+（2% 冷专家，376/384 在内存）：仅 0.2–0.7 次/步读盘，2 分钟快载，45 热槽，64K 单块 3835 tok/s；'
                 '方案 B2（关冷专家）：仅将 1 个 MXFP8 层 L47（11.75 GiB）全驻 GPU 换出内存，无 NVMe 读盘，48K 分块 2846 tok/s + 35 热槽'},
        elastic_option('on', 'prefill 时仅留 1 基础槽释放 40+ GiB 显存，decode 时压榨至 70.6/73.4 GiB（方案 A+ 45 槽 / 方案 B2 35 槽）'),
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '65664',
         'choices': [['65664', '64K 单块 · 实测推荐（3835 tok/s）'], ['49152', '48K'], ['32768', '32K'], ['16384', '16K']],
         'hint': '方案 A+ 配合弹性底座（base=1）支持 65664 单块吃下完整 64K 提示；方案 B2 自动使用 48K（49152）',
         'off_when': [{'packs': 'off'}, {'cold': 'off'}]},
        {'key': 'truncate', 'label': 'DSpark 在线截断', 'default': '0.20',
         'choices': [['0.20', '0.20 · 实测推荐（无损 +45~61%）'], ['0.15', '0.15 · 保守'], ['0', '关 · 固定验满 8 步']],
         'hint': '通过 OnlineAcceptanceEstimator 按接受概率跳过低收益草稿验证（仅加速不改采样分布，照抄与 PPL/KL 全项无损）',
         'off_when': [{'packs': 'off'}, {'spec': 'off'}]},
        miss_option([{'cold': c} for c in ('02', '08', '15')]),
        share_option('on', '读长提示时由 CPU 分担路由较少的专家块（方案 B2 关冷专家时生效）',
                     [{'cold': c} for c in ('02', '08', '15')]),
        interleave_option([['64', '64 步 / 块'], ['32', '32 步 / 块']], '64',
                          '一个请求在读长提示时，每读完一块先让其他并发请求 decode 这么多步'),
    ],
    'notes': [
        {'when': {'packs': 'off'}, 'text': '基线：W4A16/W8A16 + BF16 KV + BF16 稠密层 + 16K 分块 + 关弹性槽/Recall/截断'},
        {'when': {'cold': 'off'},
         'text': '方案 B2：仅 L47 全驻 GPU（11.75 GiB 显存、0 内存占用）使剩余 47 层放入 384 GiB 内存（留 ~13 GiB），自动 48K 分块 + 35 弹性热槽'},
    ],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    base = o['packs'] == 'off'
    plan_b = o['cold'] == 'off'
    interleaving = bool(interleave_args(o))
    chunk = '16384' if base else ('49152' if plan_b else o['chunk'])
    swa_ratio = '0.10' if (base or plan_b) else SWA_FOR_CHUNK.get(chunk, '0.14')
    env = {
        'XTIER_THREADS': o['threads'], 'PORT': str(port),
        'DSPARK': '1' if o['spec'] == 'on' else '0',
        'CONTEXT': o['context'], 'KV_POOL': o['context'],
        'BATCH': chunk, 'SWA_RATIO': swa_ratio, 'MEM': '0.86',
        'MULTIMODAL': '1' if o['multimodal'] == 'on' else '0',
        'MAX_RUNNING_REQUESTS': '2',
    }
    if SEED_PROFILE.exists():
        env['XTIER_CACHE_PROFILE'] = str(SEED_PROFILE)
    if base:
        env.update(
            XTIER_CACHE_SLOTS='16', XTIER_CACHE_ELASTIC='0',
            XTIER_PREFILL_FP4='0', XTIER_PREFILL_NVFP4='0', XTIER_PREFILL_MXFP8='0',
            XTIER_INKLING2_KV_FP4='0', XTIER_FP8_WEIGHTS='0',
            XTIER_RECALL='0', XTIER_DSPARK_TRUNCATE='0',
            XTIER_DRAFT_LMHEAD_FP8='0', XTIER_INKLING2_MARKOV_PAD2='0',
        )
    else:
        env.update(
            XTIER_CACHE_SLOTS='32',
            XTIER_CACHE_ELASTIC='1' if o['elastic'] == 'on' else '0',
            XTIER_CACHE_ELASTIC_BASE='0.03125',
            XTIER_CACHE_ELASTIC_MAX_SLOTS='96',
            XTIER_CACHE_ELASTIC_RESERVE_GIB='0.8',
            XTIER_CACHE_ELASTIC_CRITICAL_GIB='0.5',
            XTIER_CACHE_ELASTIC_IDLE_CHUNK='1024' if interleaving else '4',
            XTIER_DSPARK_TRUNCATE=o['truncate'] if o['spec'] == 'on' else '0',
        )
    if not plan_b:
        env.update(
            XTIER_COLD='1', XTIER_FAST_LOAD='1',
            XTIER_COLD_PATHS=':'.join(COLD_PATHS),
            XTIER_COLD_FRACTION=f"0.{o['cold']}",
        )
    else:
        env.update(
            XTIER_COLD='0', XTIER_COLD_PATHS='', XTIER_FAST_LOAD='0',
            XTIER_CACHE_FULL='47', XTIER_PREFILL_FP4_PRESWIZZLE='0',
        )
        env.update(miss_env(o))
        env.update(share_env(o))
    return env, [str(HERE / 'run-inkling2-xtier.sh'), *interleave_args(o)]


def tags(o):
    out = ['多模态' if o['multimodal'] == 'on' else '纯文本']
    if o['cold'] != 'off':
        out.append(f"冷专家 {int(o['cold'])}%")
    else:
        out.append('方案B2全驻L47')
        if miss_env(o):
            out.append(f"未命中 {o['missgpu']}% GPU 代算")
    if o['packs'] == 'on' and o['elastic'] == 'on':
        out.append('弹性热专家')
    if o['packs'] == 'on' and o['spec'] == 'on' and o['truncate'] != '0':
        out.append(f"在线截断 {o['truncate']}")
    if o['cold'] == 'off':
        out += share_tags(o)
    return out + interleave_tags(o)
