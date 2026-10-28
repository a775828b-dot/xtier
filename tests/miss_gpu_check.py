"""Missed experts on the GPU (XTIER_DECODE_MISS_GPU) on one real MoE layer: the decode of a partly cached layer whose
GPU part also takes a share of the call's missed experts, copied from the engine's pinned shards. Eager and replayed
from a CUDA graph, with and without skipped tokens (active flags):
- the experts taken are the share of the missed ones with the most routes (then the lowest id), at most the cap;
- bitwise: the output equals that of a layer which caches the taken experts (same W4A16 kernels, same route order),
  share 1 (every missed expert, cap 128) equals the fully cached layer;
- against the CPU engine alone: relative L2 error as in cache_check.py.

  python tests/miss_gpu_check.py --preset glm53 --threads 112
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, SWIGLU, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), required=True)
P.add_argument('--threads', type=int, default=112)
P.add_argument('--slots', type=int, default=0, help='cached experts (default E / 4)')
P.add_argument('--shares', default='1,0.5,0.3')
P.add_argument('--tolerance', type=float, default=0.02, help='largest accepted relative L2 error against the CPU engine')
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
ALPHA, BETA = SWIGLU.get(A.preset, (1.0, 0.0))
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)
os.environ['XTIER_CACHE_ADAPT'] = '0'
os.environ['XTIER_DECODE_MISS_GPU'] = '1'
os.environ['XTIER_DECODE_MISS_GPU_MAX'] = '128'   # every missed expert of a call can be taken

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xtier.backend.prefill import Prefill  # noqa: E402
from xtier.cache import pool as pool_mod  # noqa: E402
from xtier.cache.pool import LayerPool  # noqa: E402
from xtier.engine import CpuPlan, Mailbox  # noqa: E402

weights = load(A.preset)
pointers = [t.data_ptr() for t in weights]
plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT, swiglu_alpha=ALPHA,
               swiglu_beta=BETA)
plan.pool = None
prefill = Prefill(E, pointers, native=plan, hidden=H, inter=I, swiglu_limit=LIMIT, swiglu_alpha=ALPHA, swiglu_beta=BETA)
del weights, pointers
box = Mailbox(plan)
S = A.slots or E // 4
partial = LayerPool(prefill, S, plan, box, LAYER)          # the layer under test
ref = LayerPool(prefill, S + 8 * K, plan, box, LAYER)      # caches the experts partial's GPU part takes
full = LayerPool(prefill, E, plan, box, LAYER)
stream = torch.cuda.Stream()
cached = list(range(0, E, 2))[:S]
partial.fill(cached, stream)
full.fill([], stream)
ref.fill(cached, stream)
assert partial.miss is not None and full.miss is None
ref.miss = None
g = torch.Generator().manual_seed(20261028)
print(f'{A.preset}: H {H} I {I} E {E} top-{K}, {S} cached, {len(prefill.shards)} NUMA shards, '
      f'{pool_mod._miss[(H, I)][0].numel() / 2**20:.0f} MiB staging', flush=True)


def case(m):
    x = torch.randn(m, H, generator=g).bfloat16()
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
    rw = torch.softmax(torch.randn(m, K, generator=g), -1)
    act = torch.ones(m, dtype=torch.int32)
    if m > 2:
        act[1] = 0   # a skipped token: no routes on either side
    cpu = torch.empty(m, H)
    ids_c = torch.where(act[:, None] != 0, ids, -1).int().contiguous()
    plan.forward(m, K, x.data_ptr(), ids_c.data_ptr(), rw.data_ptr(), cpu.data_ptr())
    return x, ids, rw, act, cpu


def expected(ids, act, share):
    """The experts the GPU part should take, in their staging order."""
    held = set(cached)
    count = {}
    for t in range(ids.shape[0]):
        if act[t]:
            for e in ids[t].tolist():
                if e not in held:
                    count[e] = count.get(e, 0) + 1
    take = min(128, int(np.float32(len(count)) * np.float32(share) + np.float32(0.5)))
    return sorted(count, key=lambda e: (-count[e], e))[:take]


def run(pool, x, ids, rw, act, out):
    pool.decode(torch.cuda.current_stream().cuda_stream, x.shape[0], K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(),
                out.data_ptr(), act.data_ptr())


def rel(a, b):
    return float((a - b).norm() / b.norm())


def check(share, m, x, ids, rw, act, cpu, out):
    """The taken experts, out against the CPU engine and, bitwise, against the layer caching them."""
    taken = [e for e in pool_mod._miss['e'].tolist() if e >= 0]
    want = expected(ids, act, share)
    if share == 1:
        other = full
    else:
        ref.fill(cached + taken, stream)
        ref.miss = None
        other = ref
    out2 = torch.empty_like(out)
    run(other, x.cuda(), ids.cuda(), rw.cuda(), act.cuda(), out2)
    torch.cuda.synchronize()
    same = torch.equal(out, out2)
    err = rel(out.cpu(), cpu)
    return taken == want, same, err, len(taken)


bad = 0
for share in [float(s) for s in A.shares.split(',')]:
    pool_mod._miss['share'] = share   # read at launch: graphs captured below keep it
    for m in range(1, 9):
        row = [f'share {share:g} m={m}']
        x, ids, rw, act, cpu = case(m)
        xd, idsd, rwd, actd = x.cuda(), ids.cuda(), rw.cuda(), act.cuda()
        out = torch.empty(m, H, device='cuda')
        run(partial, xd, idsd, rwd, actd, out)
        torch.cuda.synchronize()
        results = [check(share, m, x, ids, rw, act, cpu, out)]
        graph, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(s):
            run(partial, xd, idsd, rwd, actd, out)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=s):
                run(partial, xd, idsd, rwd, actd, out)
        for _ in range(2):
            x2, ids2, rw2, act2, cpu2 = case(m)
            xd.copy_(x2); idsd.copy_(ids2); rwd.copy_(rw2); actd.copy_(act2)
            graph.replay()
            torch.cuda.synchronize()
            results.append(check(share, m, x2, ids2, rw2, act2, cpu2, out))
        del graph
        ok = all(sel and same and err <= A.tolerance for sel, same, err, _ in results)
        bad += not ok
        row.append(f'taken {"/".join(str(r[3]) for r in results)}  selection {"ok" if all(r[0] for r in results) else "WRONG"}'
                   f'  bitwise {"ok" if all(r[1] for r in results) else "DIFFERENT"}  vs CPU {max(r[2] for r in results):.2e}'
                   f'{"" if ok else "  FAIL"}')
        print('  '.join(row), flush=True)
box.close()
plan.close()
print('MISS-GPU-CHECK', 'PASS' if not bad else f'FAIL ({bad} cases)')
