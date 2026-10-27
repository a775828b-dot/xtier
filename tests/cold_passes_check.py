"""CPU calls with cold experts that route to more experts than the free RAM slots hold (DeepSeek V4.1 Flash 1017
layer 20, 30% of the experts on the Optane drives): CpuPlan.forward runs them in passes. Compared with the full plan:
calls of up to 8 tokens bit for bit (one pass), larger ones within float32 summation-order error.
    python tests/cold_passes_check.py DIR1 DIR2      (store directories, one per drive; a test store is written there)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(XTIER_COLD='1', XTIER_COLD_PATHS=':'.join(sys.argv[1:3]), XTIER_COLD_FRACTION='0.3',
                  XTIER_COLD_KEY='passes-check', XTIER_COLD_LOG='0')
for name, value in {'DYNAMIC': '2', 'DOWN_BF16': '1', 'DOWN_GRAIN': '24', 'SPIN_COUNT': '1048576', 'THP': '0'}.items():
    os.environ.setdefault('XTIER_' + name, value)
import json
import torch
from safetensors import safe_open

from xtier.cache import cold
from xtier.engine import CpuPlan

MODEL, KEY, NAMES = '/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017', 'layers.{layer}.ffn.experts.{e}.', ('w1', 'w3', 'w2')
H, I, E, K, LIMIT, LAYER, G = 5120, 2304, 384, 6, 10.0, 20, 32
root = Path(MODEL)
index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
w13 = torch.empty((E, 2 * I, H // 2), dtype=torch.uint8); w2 = torch.empty((E, H, I // 2), dtype=torch.uint8)
s13 = torch.empty((E, 2 * I, H // G), dtype=torch.uint8); s2 = torch.empty((E, H, I // G), dtype=torch.uint8)
g13, g2 = torch.ones((E, 2)), torch.ones((E,))
grouped = {}
for e in range(E):
    p = KEY.format(layer=LAYER, e=e)
    for role, n in zip(('gate', 'up', 'down'), NAMES):
        grouped.setdefault(index[p + n + '.weight'], []).append((e, p, n, role))
for file, items in grouped.items():
    with safe_open(root / file, framework='pt', device='cpu') as f:
        for e, p, n, role in items:
            w, s, rows = {'gate': (w13[e], s13[e], slice(0, I)), 'up': (w13[e], s13[e], slice(I, 2 * I)),
                          'down': (w2[e], s2[e], slice(None))}[role]
            w[rows].copy_(f.get_tensor(p + n + '.weight').view(torch.uint8))
            s[rows].copy_(f.get_tensor(p + n + '.scale').view(torch.uint8))
inputs = [w13, w2, s13, s2, g13, g2]
pointers = [t.data_ptr() for t in inputs]
full = CpuPlan(E, 112, pointers, H, I, LIMIT, 'mxfp4')
lp = cold.LayerPlan(LAYER, E, 50)
plan = CpuPlan(E, 112, pointers, H, I, LIMIT, 'mxfp4', resident=lp.resident)
cold.attach(plan, lp, inputs, 'mxfp4')
print(f'{plan.slots} slots, {plan.dynamic} free, {len(lp.stored)} records', flush=True)
g = torch.Generator().manual_seed(1026)
worst = 0.0
for m in (1, 8, 64, 340, 1024, 1024):
    x = torch.randn(m, H, generator=g).bfloat16()
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
    rw = torch.rand(m, K, generator=g)
    y0, y1 = torch.zeros(m, H), torch.zeros(m, H)
    for pl, y in ((full, y0), (plan, y1)):
        pl.forward(m, K, x.view(torch.int16).data_ptr(), ids.data_ptr(), rw.data_ptr(), y.data_ptr())
    diff = (y0 - y1).abs().max().item() / max(y0.abs().max().item(), 1e-30)
    same = torch.equal(y0, y1)
    print(f'm {m}: identical {same}, max relative difference {diff:.2e}, experts read so far {int(plan.cold_stats()[1])}',
          flush=True)
    # one pass (m <= 8 without XTIER_COLD_OVERLAP) is bitwise the full plan; passes and the overlap sum in FP32 parts
    assert (same if m <= 8 and os.environ.get('XTIER_COLD_OVERLAP') == '0' else diff < 1e-5), (m, diff)
    worst = max(worst, diff)
print(f'cold passes ok: worst relative difference {worst:.2e}')
