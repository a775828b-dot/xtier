"""Hybrid prefill of a partly cached layer (xtier/cache/prefill.py, hybrid_prefill: cached routes on the GPU, the others
on the CPU engine at the same time) against the CPU engine alone on one real MoE layer: relative L2 error (the GPU
part is W4A16) and time per call, next to the CPU-only time.

  python tests/hybrid_prefill_check.py --preset qwen38 --slots 155 --lengths 100,500,1000,2000
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, SWIGLU, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), required=True)
P.add_argument('--threads', type=int, default=112)
P.add_argument('--slots', type=int, default=155)
P.add_argument('--lengths', default='100,500,1000,2000')
P.add_argument('--tolerance', type=float, default=0.02)
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
ALPHA, BETA = SWIGLU.get(A.preset, (1.0, 0.0))
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)
os.environ['XTIER_CACHE_ADAPT'] = '0'

import torch  # noqa: E402

from xtier.backend.prefill import Prefill  # noqa: E402
from xtier.cache.pool import LayerPool  # noqa: E402
from xtier.cache.prefill import hybrid_prefill  # noqa: E402
from xtier.engine import CpuPlan, Mailbox  # noqa: E402

weights = load(A.preset)
pointers = [t.data_ptr() for t in weights]
plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT, swiglu_alpha=ALPHA,
               swiglu_beta=BETA)
plan.pool = None
prefill = Prefill(E, pointers, native=plan, hidden=H, inter=I, swiglu_limit=LIMIT, swiglu_alpha=ALPHA, swiglu_beta=BETA)
del weights, pointers
box = Mailbox(plan)
pool = LayerPool(prefill, A.slots, plan, box, LAYER)
g = torch.Generator().manual_seed(20261023)
# the hottest experts of a skewed routing distribution are cached, as the profile would place them
popularity = torch.rand(E, generator=g) ** 4
pool.fill(torch.argsort(popularity, descending=True)[:A.slots].tolist(), torch.cuda.Stream())


def cpu_only(x, ids, rw):
    m = x.shape[0]
    out = torch.empty(m, H)
    for s0 in range(0, m, 1024):
        n = min(1024, m - s0)
        plan.forward(n, K, x[s0:].data_ptr(), ids[s0:].data_ptr(), rw[s0:].data_ptr(), out[s0:].data_ptr())
    return out


bad = 0
for m in map(int, A.lengths.split(',')):
    x = torch.randn(m, H, generator=g).bfloat16()
    ids = torch.multinomial(popularity.expand(m, E), K, generator=g).int()
    rw = torch.softmax(torch.randn(m, K, generator=g), -1)
    t = time.perf_counter()
    ref = cpu_only(x, ids, rw)
    t_cpu = (time.perf_counter() - t) * 1e3
    xd, idsd, rwd = x.cuda(), ids.cuda(), rw.cuda()
    hybrid_prefill(pool, plan, xd, idsd, rwd)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(3):
        got = hybrid_prefill(pool, plan, xd, idsd, rwd)
    torch.cuda.synchronize()
    t_hyb = (time.perf_counter() - t) / 3 * 1e3
    hit = float((pool.table[idsd.long()] >= 0).float().mean())
    err = float((got.cpu() - ref).norm() / ref.norm())
    ok = err <= A.tolerance
    bad += not ok
    print(f'm={m}: hit {hit:.2f}, CPU only {t_cpu:.1f} ms, hybrid {t_hyb:.1f} ms, rel L2 {err:.2e}{"" if ok else " FAIL"}',
          flush=True)
box.close()
plan.close()
print('HYBRID-PREFILL-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
