"""XTIER_TAIL_SKIP=K/N (csrc/engine/engine.cpp tail_skip, xtier/cache/tail.py) on DeepSeek V4.1 Flash 1017 layer 20 at
4/6 (keep the 4 largest routes of its top-6). A decode-path call (xtier_forward_decode) of at most 8 tokens
(XTIER_TAIL_SKIP_TOKENS) leaves out the experts that no token of the call has among its 4 largest routes (by weight,
ties by position), in RAM or not: checked bit for bit against xtier_forward given the same routes with those experts
taken out, and the experts left out against the plan's counts (tail_stats). Routes given as -1 (the GPU's) count for
the ranks only. A plain forward (prefill) of at most 8 tokens and a decode-path call of 16 leave out nothing. With store
directories, also a plan with 30% of its experts in a cold store (XTIER_COLD_OVERLAP=0: one pass): the same output, and
the experts not in RAM that it leaves out are not read (cold_stats). XTIER_TAIL_SKIP_EVAL's prompt mask
(tail.prefill_mask) against the same rule per token, for the experts outside a layer's GPU cache. Last, the setting:
malformed values refused by the engine and tail.setting alike, 0 off, a total other than the top-k refused (tail.check)
and leaving nothing out.
    python tests/tail_skip_check.py [DIR ...]      (store directories, one per drive; a test store is written there)
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

R, K = 4, 6   # keep, top-k
DIRS = sys.argv[1:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(XTIER_TAIL_SKIP=f'{R}/{K}')
if DIRS:
    os.environ.update(XTIER_COLD='1', XTIER_COLD_PATHS=':'.join(DIRS), XTIER_COLD_FRACTION='0.3',
                      XTIER_COLD_KEY='tail-skip-check', XTIER_COLD_LOG='0', XTIER_COLD_OVERLAP='0')
for name, value in {'DYNAMIC': '2', 'DOWN_BF16': '1', 'DOWN_GRAIN': '24', 'SPIN_COUNT': '1048576', 'THP': '0'}.items():
    os.environ.setdefault('XTIER_' + name, value)
import json
import torch
from safetensors import safe_open

from xtier.cache import tail
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
plans = [('RAM', full)]
if DIRS:
    from xtier.cache import cold
    lp = cold.LayerPlan(LAYER, E, 50)
    cold_plan = CpuPlan(E, 112, pointers, H, I, LIMIT, 'mxfp4', resident=lp.resident)
    cold.attach(cold_plan, lp, inputs, 'mxfp4')
    plans.append(('cold', cold_plan))
    print(f'cold plan: {cold_plan.slots} slots, {len(lp.stored)} records', flush=True)
print(f'XTIER_TAIL_SKIP={R}/{K}', flush=True)


def ranks(rw):
    """Per route, its rank among its token's routes by weight (0 the largest; ties: the earlier route first)."""
    m, k = rw.shape
    out = torch.empty((m, k), dtype=torch.int64)
    for t in range(m):
        for r, q in enumerate(sorted(range(k), key=lambda q: (-float(rw[t, q]), q))):
            out[t, q] = r
    return out


def expected(ids, rw):
    """The routes tail_skip leaves (the others -1) and the experts it leaves out (routes -1 count for the ranks only)."""
    rank, lowest = ranks(rw), {}
    for (t, q), e in zip(((t, q) for t in range(ids.shape[0]) for q in range(ids.shape[1])), ids.flatten().tolist()):
        if e >= 0:
            lowest[e] = min(lowest.get(e, K), int(rank[t, q]))
    out = {e for e, r in lowest.items() if r >= R}
    kept = ids.clone()
    for e in out:
        kept[ids == e] = -1
    return kept, out


def check(name, plan, ids, rw, decode=True):
    """One call of `plan` (decode path or a plain forward) against the RAM plan's plain forward of the routes the rule
    leaves; the experts left out against tail_stats, and on a cold plan the reads against cold_stats."""
    m = ids.shape[0]
    skip = decode and m <= 8
    kept, out = expected(ids, rw) if skip else (ids, set())
    x = torch.randn(m, H, generator=g).bfloat16()
    y0, y1 = torch.zeros(m, H), torch.zeros(m, H)
    in_ram = [s >= 0 for s in plan.slot_table()]   # the residency the call starts from
    reads = len({e for e in kept.flatten().tolist() if e >= 0 and not in_ram[e]})
    saved = len({e for e in out if not in_ram[e]})
    t0, c0 = plan.tail_stats(), plan.cold_stats()
    full.forward(m, K, x.view(torch.int16).data_ptr(), kept.contiguous().data_ptr(), rw.data_ptr(), y0.data_ptr())
    call = plan.forward_decode if decode else plan.forward
    call(m, K, x.view(torch.int16).data_ptr(), ids.contiguous().data_ptr(), rw.data_ptr(), y1.data_ptr())
    t1, c1 = plan.tail_stats(), plan.cold_stats()
    left, looked = int(t1[2] - t0[2]), int(t1[0] - t0[0])
    diff = (y0 - y1).abs().max().item() / max(y0.abs().max().item(), 1e-30)
    same = torch.equal(y0, y1)
    line = f'{name} ({"decode" if decode else "forward"}): m {m}, left out {left} (expected {len(out)})'
    if plan is not full:
        line += f', read {int(c1[1] - c0[1])} (expected {reads}), reads saved {int(c1[6] - c0[6])} (expected {saved})'
    print(f'{line}, identical {same}, max relative difference {diff:.2e}', flush=True)
    assert left == len(out) and looked == int(skip), (name, left, looked, sorted(out))
    if plan is not full:
        assert int(c1[1] - c0[1]) == reads and int(c1[6] - c0[6]) == saved, (name, c0, c1, reads, saved)
    assert same if m <= 8 else diff < 1e-5, (name, diff)
    return left


g = torch.Generator().manual_seed(1106)
for label, plan in plans:
    total = 0
    for m in (1, 1, 2, 3, 4, 6, 8, 8):   # random routes: a token's two smallest routes are left out unless shared
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
        total += check(f'{label} random', plan, ids, torch.rand(m, K, generator=g))
    assert total > 0, 'no call left an expert out: the check did not exercise the skip'

    # GPU routes (-1) keep their rank: token 0's last two CPU routes are at ranks 4 and 5 (left out, a zero row), token
    # 1's at 0 and 1 (kept)
    a, b, c, d = torch.randperm(E, generator=g)[:4].tolist()
    ids = torch.tensor([[-1, -1, -1, -1, a, b], [c, d, -1, -1, -1, -1]], dtype=torch.int32)
    rw = torch.tensor([[.9, .8, .7, .6, .5, .4], [.9, .8, .7, .6, .5, .4]])
    assert expected(ids, rw)[1] == {a, b}
    check(f'{label} GPU routes', plan, ids, rw)

    # two tokens share two experts: c1 at ranks 4 and 0 stays (one token ranks it high), c2 at rank 5 in both goes, as
    # does e[7] (rank 4 in the one token that routes to it)
    e = torch.randperm(E, generator=g)[:10].tolist()
    c1, c2 = e[8], e[9]
    ids = torch.tensor([[e[0], e[1], e[2], e[3], c1, c2], [c1, e[4], e[5], e[6], c2, e[7]]], dtype=torch.int32)
    rw = torch.tensor([[.9, .8, .7, .6, .5, .4], [.95, .5, .4, .3, .2, .25]])
    assert expected(ids, rw)[1] == {c2, e[7]}
    check(f'{label} shared', plan, ids, rw)

    # a plain forward (a prefill's last chunk) of at most 8 tokens and a decode-path call of more than
    # XTIER_TAIL_SKIP_TOKENS: nothing left out
    for m, decode in ((4, False), (16, True)):
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
        check(f'{label} {m} tokens', plan, ids, torch.rand(m, K, generator=g), decode=decode)

# XTIER_TAIL_SKIP_EVAL's prompt mask: per token (no call-wide minimum), the weights of those routes set to 0, for the
# experts outside the layer's GPU cache (all of them without one, none in a fully cached layer)
m = 64
ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
rw = torch.rand(m, K, generator=g)
table = torch.full((E,), -1, dtype=torch.int32)
table[torch.randperm(E, generator=g)[:50]] = torch.arange(50, dtype=torch.int32)   # 50 experts in GPU slots
low = ranks(rw) >= R
for pool, want in ((SimpleNamespace(table=table, full=False), rw.masked_fill((table[ids.long()] < 0) & low, 0.0)),
                   (None, rw.masked_fill(low, 0.0)),
                   (SimpleNamespace(table=table, full=True), rw)):
    masked = tail.prefill_mask(pool, ids, rw, R)
    assert torch.equal(masked, want), 'prefill_mask differs from the rule'
    print(f'prefill mask ({"no cache" if pool is None else "full cache" if pool.full else "50 GPU slots"}): '
          f'{int((masked == 0).sum())} of {m * K} routes set to 0, as the rule says')

# the setting: keep/total, digits only; the engine (at a plan's creation) and tail.setting refuse the same values; 0 is
# off; a total other than the call's top-k leaves nothing out, and tail.check (a layer's first call) refuses it
ones2, ones1 = torch.ones((10, 2)), torch.ones((10,))


def small():   # ten experts of the real shapes, shards not filled
    return CpuPlan(10, 112, [None] * 4 + [ones2.data_ptr(), ones1.data_ptr()], H, I, LIMIT, 'mxfp4')


for bad in ('4', '6/6', '0/6', '7/6', '4/x', '4/6/1', ' 4/6', '4/6 ', '4 /6', '+4/6', '-1/6', '4/'):
    os.environ['XTIER_TAIL_SKIP'] = bad
    try:
        small().close()
        raise AssertionError(f'the engine accepted XTIER_TAIL_SKIP={bad!r}')
    except RuntimeError as e:
        assert 'keep/total' in str(e), e
    try:
        tail.setting()
        raise AssertionError(f'tail.setting accepted {bad!r}')
    except ValueError:
        pass
os.environ['XTIER_TAIL_SKIP'] = '0'
small().close()
assert tail.setting() == (0, 0) and tail.rank() == 0
os.environ['XTIER_TAIL_SKIP'] = '4/8'
assert tail.setting() == (4, 8)
p = small()
ids = torch.randperm(10, generator=g)[:K].int().view(1, K)
x, y = torch.zeros(1, H).bfloat16(), torch.zeros(1, H)
p.forward_decode(1, K, x.view(torch.int16).data_ptr(), ids.data_ptr(), torch.rand(1, K, generator=g).data_ptr(),
                 y.data_ptr())
assert p.tail_stats()[0] == 0, '4/8 looked at a top-6 call'
p.close()
try:
    tail.check(K)
    raise AssertionError('tail.check accepted 4/8 for a top-6 layer')
except ValueError as e:
    print(f'4/8 on a top-6 layer refused: {e}')
os.environ['XTIER_TAIL_SKIP'] = f'{R}/{K}'
tail.check(K)
print('setting: 12 malformed values refused by both, 0 off, a wrong total refused')
print('tail skip ok')
