"""XTIER_TAIL_SKIP=R (csrc/engine/engine.cpp tail_skip, xtier/cache/tail.py) on DeepSeek V4.1 Flash 1017 layer 20 with
30% of its experts in a cold store, R = 4 of its top-6. A call of at most 8 tokens (XTIER_TAIL_SKIP_TOKENS) leaves out
the experts not in RAM that every token of the call routes at rank R or lower (by weight, 0 the largest, ties by
position): checked bit for bit against the plan with every expert in RAM given the same routes with those experts taken
out (XTIER_COLD_OVERLAP=0: one pass), and the experts left out against the plan's count (cold_stats). A larger call
leaves out nothing. XTIER_TAIL_SKIP_EVAL's prompt mask (tail.prefill_mask) against the same rule per token.
    python tests/tail_skip_check.py DIR [DIR...]      (store directories, one per drive; a test store is written there)
"""
import os
import sys
from pathlib import Path

R = 4
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(XTIER_COLD='1', XTIER_COLD_PATHS=':'.join(sys.argv[1:]), XTIER_COLD_FRACTION='0.3',
                  XTIER_COLD_KEY='tail-skip-check', XTIER_COLD_LOG='0', XTIER_COLD_OVERLAP='0', XTIER_TAIL_SKIP=str(R))
for name, value in {'DYNAMIC': '2', 'DOWN_BF16': '1', 'DOWN_GRAIN': '24', 'SPIN_COUNT': '1048576', 'THP': '0'}.items():
    os.environ.setdefault('XTIER_' + name, value)
import json
import torch
from safetensors import safe_open

from xtier.cache import cold, tail
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
print(f'{plan.slots} slots, {len(lp.stored)} records, R = {R}', flush=True)


def ranks(rw):
    """Per route, its rank among its token's routes by weight (0 the largest; ties: the earlier route first)."""
    m, k = rw.shape
    out = torch.empty((m, k), dtype=torch.int64)
    for t in range(m):
        for r, q in enumerate(sorted(range(k), key=lambda q: (-float(rw[t, q]), q))):
            out[t, q] = r
    return out


def expected(ids, rw, in_ram):
    """The routes tail_skip leaves (the others -1) and the experts it leaves out."""
    rank, lowest = ranks(rw), {}
    for (t, q), e in zip(((t, q) for t in range(ids.shape[0]) for q in range(ids.shape[1])), ids.flatten().tolist()):
        lowest[e] = min(lowest.get(e, K), int(rank[t, q]))
    out = {e for e, r in lowest.items() if not in_ram[e] and r >= R}
    kept = ids.clone()
    for e in out:
        kept[ids == e] = -1
    return kept, out


def check(name, ids, rw, skip=True):
    """One call of the cold plan against the full plan given the routes the rule leaves."""
    m = ids.shape[0]
    in_ram = [s >= 0 for s in plan.slot_table()]   # the residency the call starts from
    kept, out = expected(ids, rw, in_ram) if skip else (ids, set())
    x = torch.randn(m, H, generator=g).bfloat16()
    y0, y1 = torch.zeros(m, H), torch.zeros(m, H)
    before = plan.cold_stats()[6]
    full.forward(m, K, x.view(torch.int16).data_ptr(), kept.contiguous().data_ptr(), rw.data_ptr(), y0.data_ptr())
    plan.forward(m, K, x.view(torch.int16).data_ptr(), ids.data_ptr(), rw.data_ptr(), y1.data_ptr())
    left = int(plan.cold_stats()[6] - before)
    diff = (y0 - y1).abs().max().item() / max(y0.abs().max().item(), 1e-30)
    same = torch.equal(y0, y1)
    print(f'{name}: m {m}, left out {left} (expected {len(out)}), identical {same}, max relative difference {diff:.2e}',
          flush=True)
    assert left == len(out), (name, left, sorted(out))
    assert same if m <= 8 else diff < 1e-5, (name, diff)   # more than 8 tokens: passes, summed in FP32 parts
    return left


g = torch.Generator().manual_seed(1106)
total = 0
for m in (1, 1, 2, 3, 4, 8, 8, 8):   # random routes: about 30% of a token's two smallest routes go to a cold expert
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
    total += check('random', ids, torch.rand(m, K, generator=g))
assert total > 0, 'no call left an expert out: the check did not exercise the skip'

# two tokens share two cold experts: c1 at ranks 4 and 0 stays (one token ranks it high), c2 at rank 5 in both goes
in_ram = [s >= 0 for s in plan.slot_table()]
hot = [e for e in range(E) if in_ram[e]][:8]
c1, c2 = [e for e in range(E) if not in_ram[e]][:2]
ids = torch.tensor([[hot[0], hot[1], hot[2], hot[3], c1, c2], [c1, hot[4], hot[5], hot[6], c2, hot[7]]],
                   dtype=torch.int32)
rw = torch.tensor([[.9, .8, .7, .6, .5, .4], [.95, .5, .4, .3, .2, .25]])
kept, out = expected(ids, rw, in_ram)
assert out == {c2}, out
check('shared', ids, rw)

# more tokens than XTIER_TAIL_SKIP_TOKENS: nothing left out
m = 16
ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
check('16 tokens', ids, torch.rand(m, K, generator=g), skip=False)

# XTIER_TAIL_SKIP_EVAL's prompt mask: per token (no call-wide minimum), the weights of those routes set to 0
m = 64
ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
rw = torch.rand(m, K, generator=g)
in_ram = torch.tensor([s >= 0 for s in plan.slot_table()])
masked = tail.prefill_mask(plan, ids, rw, R)
want = rw.masked_fill(~in_ram[ids.long()] & (ranks(rw) >= R), 0.0)
assert torch.equal(masked, want), 'prefill_mask differs from the rule'
print(f'prefill mask: {int((masked == 0).sum())} of {m * K} routes set to 0, as the rule says')
print('tail skip ok')
