"""CPU engine check on one real MoE layer: output digests (bitwise) and per-call timings, for the XTier engine or the
lkqmoe engine it was ported from (one library per process: its workers pin the CPUs).

  python tests/engine_ab.py --preset qwen38 --impl xtier --lib-dir build/engine --threads 112 --tag new --out DIR
  python tests/engine_ab.py --preset qwen38 --impl lkqmoe --library .../liblkqmoe.so --python .../python --tag old --out DIR

Digest = sha256 of every case's float32 output; the cases are seeded, so equal digests mean equal outputs. Set the
production kernel switches the same way on both sides (XTIER_* for xtier, LKQMOE_* for lkqmoe; see PRESETS).
"""
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path

# name: model dir, expert key, (gate, up, down) names, H, I, E, top-k, SwiGLU limit, layer, verify width, format,
#       production kernel switches (without prefix)
PRESETS = {
    'glm53': ('/data/fast/nvme1/models/huggingface/LibertAIDAI/GLM-5.3-Flash-NVFP4', 'model.language_model.layers.{layer}.mlp.experts.{e}.',
              ('gate_proj', 'up_proj', 'down_proj'), 4096, 2048, 288, 8, 10.0, 20, 4, 'nvfp4', {'DYNAMIC': '2', 'DOWN_BF16': '1', 'SPIN_COUNT': '1048576'}),
    'qwen38': ('/data/fast/nvme2/models/huggingface/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4',
               'model.language_model.layers.{layer}.mlp.experts.{e}.', ('gate_proj', 'up_proj', 'down_proj'), 2560, 640, 512, 10, 0.0, 36, 4,
               'nvfp4', {'SPIN_COUNT': '1048576'}),
    'ds41_1017': ('/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017', 'layers.{layer}.ffn.experts.{e}.', ('w1', 'w3', 'w2'),
                  5120, 2304, 384, 6, 10.0, 20, 6, 'mxfp4', {'DYNAMIC': '2', 'DOWN_BF16': '1', 'DOWN_GRAIN': '24', 'SPIN_COUNT': '1048576', 'THP': '0'}),
}
P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), required=True)
P.add_argument('--impl', choices=('xtier', 'lkqmoe'), required=True)
P.add_argument('--lib-dir', help='xtier: directory with libxtier_engine*.so')
P.add_argument('--library', help='lkqmoe: liblkqmoe.so')
P.add_argument('--python', help='lkqmoe: its python directory')
P.add_argument('--threads', type=int, default=112)
P.add_argument('--tag', required=True)
P.add_argument('--out', required=True)
P.add_argument('--rounds', type=int, default=3)
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
prefix = 'XTIER_' if A.impl == 'xtier' else 'LKQMOE_'
for name, value in SWITCHES.items():
    os.environ.setdefault(prefix + name, value)
import torch
from safetensors import safe_open

torch.set_num_threads(8)
G = 32 if FMT == 'mxfp4' else 16


def load():
    root = Path(MODEL)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    w13 = torch.empty((E, 2 * I, H // 2), dtype=torch.uint8); w2 = torch.empty((E, H, I // 2), dtype=torch.uint8)
    s13 = torch.empty((E, 2 * I, H // G), dtype=torch.uint8); s2 = torch.empty((E, H, I // G), dtype=torch.uint8)
    g13 = torch.ones((E, 2), dtype=torch.float32); g2 = torch.ones((E,), dtype=torch.float32)
    scale_suffix = '.scale' if FMT == 'mxfp4' else '.weight_scale'
    grouped = {}
    for e in range(E):
        p = KEY.format(layer=LAYER, e=e)
        for role, n in zip(('gate', 'up', 'down'), NAMES):
            grouped.setdefault(index[p + n + '.weight'], []).append((e, p, n, role))
    for file, items in grouped.items():
        with safe_open(root / file, framework='pt', device='cpu') as f:
            for e, p, n, role in items:
                w, s, g, rows = {'gate': (w13[e], s13[e], g13[e, 0], slice(0, I)), 'up': (w13[e], s13[e], g13[e, 1], slice(I, 2 * I)),
                                 'down': (w2[e], s2[e], g2[e], slice(None))}[role]
                w[rows].copy_(f.get_tensor(p + n + '.weight').view(torch.uint8))
                s[rows].copy_(f.get_tensor(p + n + scale_suffix).view(torch.uint8))
                if FMT == 'nvfp4':
                    g.copy_(f.get_tensor(p + n + '.weight_scale_2'))
    return w13, w2, s13, s2, g13, g2


def cases():
    g = torch.Generator().manual_seed(20261016)
    out = []
    for m, kind, reps in ((1, 'indep', 256), (2, 'indep', 128), (4, 'indep', 128), (VM, 'verify', 128), (8, 'indep', 128),
                          (8, 'verify2', 128), (64, 'indep', 8), (512, 'indep', 3), (2048, 'indep', 2)):
        xs, idss, rws = [], [], []
        for _ in range(min(reps, 16)):
            x = (torch.randn(m, H, generator=g) * (8 if kind == 'verify2' else 1)).bfloat16()
            ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
            if kind.startswith('verify'):
                ids[1:, :K * 2 // 3] = ids[0, :K * 2 // 3]
            xs.append(x); idss.append(ids); rws.append(torch.softmax(torch.randn(m, K, generator=g), -1))
        out.append((f'm{m}-{kind}', m, xs, idss, rws, reps))
    return out


weights = load()
pointers = [t.data_ptr() for t in weights]
if A.impl == 'xtier':
    os.environ['XTIER_ENGINE_LIB'] = str(Path(A.lib_dir).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from xtier.engine import CpuPlan, isa
    plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT)
    kernels = isa()
else:
    sys.path.insert(0, A.python)
    from lkqmoe import NativePlan
    plan = NativePlan(E, A.threads, pointers, A.library, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT)
    kernels = 'lkqmoe'
res = dict(tag=A.tag, impl=A.impl, kernels=kernels, preset=A.preset, threads=A.threads, nodes=plan.nodes(),
           switches={prefix + k: os.environ[prefix + k] for k in SWITCHES}, digest={}, us={})
suite = cases()
for name, m, xs, idss, rws, reps in suite:   # digests (and warm-up)
    h = hashlib.sha256()
    y = torch.empty(m, H)
    for x, ids, rw in zip(xs, idss, rws):
        plan.forward(m, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), y.data_ptr())
        h.update(y.numpy().tobytes())
    res['digest'][name] = h.hexdigest()
for r in range(A.rounds):   # timings: mean per call, median over rounds
    for name, m, xs, idss, rws, reps in suite:
        y = torch.empty(m, H)
        t0 = time.perf_counter()
        for i in range(reps):
            j = i % len(xs)
            plan.forward(m, K, xs[j].data_ptr(), idss[j].data_ptr(), rws[j].data_ptr(), y.data_ptr())
        res['us'].setdefault(name, []).append((time.perf_counter() - t0) * 1e6 / reps)
res['us'] = {k: round(statistics.median(v), 1) for k, v in res['us'].items()}
plan.close()
Path(A.out).mkdir(parents=True, exist_ok=True)
Path(A.out, f'{A.preset}-{A.tag}-t{A.threads}.json').write_text(json.dumps(res, indent=1))
print(json.dumps(dict(tag=A.tag, kernels=kernels, preset=A.preset, threads=A.threads, nodes=res['nodes'], us=res['us'])), flush=True)
