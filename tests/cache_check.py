"""Expert cache check on one real MoE layer: the decode of a partly cached layer (GPU part for the cached experts, CPU
engine for the rest, through the mailbox) and of a fully cached one (GPU only), eager and replayed from a CUDA graph,
against the CPU engine alone. The GPU part is W4A16 (FP16 / BF16 operands, FP32 accumulation, BF16 intermediate), so
the comparison is a relative L2 error, not bitwise; a CPU-only call through the same path must stay bitwise.

  python tests/cache_check.py --preset qwen38 --threads 112 --slots 256
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), required=True)
P.add_argument('--threads', type=int, default=112)
P.add_argument('--slots', type=int, default=256)
P.add_argument('--tolerance', type=float, default=0.02, help='largest accepted relative L2 error of a GPU-part call')
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)
os.environ['XTIER_CACHE_ADAPT'] = '0'

import torch  # noqa: E402

from xtier.backend.prefill import Prefill  # noqa: E402
from xtier.cache.pool import LayerPool  # noqa: E402
from xtier.engine import CpuPlan, Mailbox  # noqa: E402

weights = load(A.preset)
pointers = [t.data_ptr() for t in weights]
plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT)
plan.pool = None
prefill = Prefill(E, pointers, native=plan, hidden=H, inter=I, swiglu_limit=LIMIT)
del weights, pointers
box = Mailbox(plan)
partial = LayerPool(prefill, A.slots, plan, box, LAYER)
full = LayerPool(prefill, E, plan, box, LAYER)
none = LayerPool(prefill, 1, plan, box, LAYER)   # one slot, filled with an expert no route uses
stream = torch.cuda.Stream()
cached = list(range(0, E, 2))[:A.slots]
partial.fill(cached, stream)
full.fill([], stream)
g = torch.Generator().manual_seed(20261023)


def case(m, avoid=None):
    x = torch.randn(m, H, generator=g).bfloat16()
    pool = [e for e in range(E) if e != avoid]
    ids = torch.stack([torch.tensor(pool)[torch.randperm(len(pool), generator=g)[:K]] for _ in range(m)]).int()
    rw = torch.softmax(torch.randn(m, K, generator=g), -1)
    ref = torch.empty(m, H)
    plan.forward(m, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), ref.data_ptr())
    return x, ids, rw, ref


def run(pool, x, ids, rw, out):
    pool.decode(torch.cuda.current_stream().cuda_stream, x.shape[0], K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(),
                out.data_ptr())


def rel(a, b):
    return float((a - b).norm() / b.norm())


bad = 0
none.fill([E - 1], stream)
for m in range(1, 9):
    row = [f'm={m}']
    for name, pool in (('partial', partial), ('full', full), ('cpu-only', none)):
        x, ids, rw, ref = case(m, avoid=E - 1 if pool is none else None)
        xd, idsd, rwd = x.cuda(), ids.cuda(), rw.cuda()
        out = torch.empty(m, H, device='cuda')
        run(pool, xd, idsd, rwd, out)
        torch.cuda.synchronize()
        errs = [rel(out.cpu(), ref)]
        graph, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(s):
            run(pool, xd, idsd, rwd, out)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=s):
                run(pool, xd, idsd, rwd, out)
        for _ in range(2):
            x2, ids2, rw2, ref2 = case(m, avoid=E - 1 if pool is none else None)
            xd.copy_(x2); idsd.copy_(ids2); rwd.copy_(rw2)
            graph.replay()
            torch.cuda.synchronize()
            errs.append(rel(out.cpu(), ref2))
        del graph
        ok = all(e == 0.0 for e in errs) if pool is none else max(errs) <= A.tolerance
        bad += not ok
        row.append(f'{name} {max(errs):.2e}{"" if ok else " FAIL"}')
    print('  '.join(row), flush=True)
box.close()
plan.close()
print('CACHE-CHECK', 'PASS' if not bad else f'FAIL ({bad} cases)')
