"""The FP4 GPU prefill (xtier/backend/prefill.py _run_fp4: DeepSeek V4.1's prefill on SM120) run again after a call that
failed half-way, as the elastic cache's out-of-memory retry does (xtier/cache/elastic.py _forward), on one real layer with
an elastic hot pool. A call notes which experts the pool holds once the routes are known; uploads it starts before that
must not go by what an earlier, failed call noted (2026-11-10 audit R3: a fault, or the wrong experts without a word).

  A  the pool's mapped slots hold experts 0.., a call fails in the fused MoE, they then hold 100.. (the adaptive
     policy's swaps), the call again                                          -> equal to the pool-less reference
  B  7 extra slots hold experts 0..6, a call fails, those slots are released as the elastic cache does (experts out
     of the table, memory unmapped), the call again                           -> no fault, equal to the reference

  python tests/prefill_fp4_retry_check.py [--preset ds41_1017] [--threads 112]
"""
import argparse
import ctypes as C
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), default='ds41_1017')
P.add_argument('--threads', type=int, default=112)
P.add_argument('--tokens', type=int, default=7500)
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)
SLOTS, EXTRA = 50, 7
for name, value in dict(PREFILL_FP4='1', PREFILL_UNPACK='1', PREFILL_EXPERT_CHUNK='8', PREFILL_ROUTE_ACCUM='1',
                        PROBE_DISPATCH_CPU='95', CACHE_ELASTIC='1', CACHE_ELASTIC_MAX_SLOTS='68').items():
    os.environ['XTIER_' + name] = value

import torch  # noqa: E402

if torch.cuda.get_device_capability()[0] < 10:
    print('PREFILL-FP4-RETRY-CHECK SKIP (the FP4 prefill needs SM100 or newer)')
    sys.exit(0)

import flashinfer.fused_moe as fused_moe  # noqa: E402

from xtier.backend.prefill import Prefill  # noqa: E402
from xtier.cache.pool import LayerPool  # noqa: E402
from xtier.engine import CpuPlan, Mailbox  # noqa: E402

weights = load(A.preset)
pointers = [t.data_ptr() for t in weights]


def prefill(with_pool):
    plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT)
    plan.pool = None
    pre = Prefill(E, pointers, native=plan, hidden=H, inter=I, swiglu_limit=LIMIT)
    if not with_pool:
        return pre, None, None
    box = Mailbox(plan)
    pool = LayerPool(pre, SLOTS, plan, box, LAYER)
    plan.pool = pool
    return pre, pool, box.lib


reference, _, _ = prefill(False)
pre, pool, lib = prefill(True)
if not reference._fp4_ready(torch.bfloat16):
    print(f'PREFILL-FP4-RETRY-CHECK SKIP ({A.preset}: not an MXFP4 layer the FP4 prefill takes)')
    sys.exit(0)
BASE = pool.base   # the slots mapped for good (the elastic pool's address space reaches pool.S)
assert pool.S >= BASE + EXTRA and EXTRA <= 8, (pool.base, pool.S)
stream, copy = torch.cuda.current_stream(), torch.cuda.Stream()
g = torch.Generator().manual_seed(20261110)
m = A.tokens
x = torch.randn(m, H, generator=g).bfloat16().cuda()
ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int().cuda()
rw = torch.softmax(torch.randn(m, K, generator=g), -1).cuda()
bad = 0


def check(name, ok, detail=''):
    global bad
    print(f'{name}: {"ok" if ok else "FAIL"} {detail}'.rstrip(), flush=True)
    bad += not ok


def call(p):
    out = torch.empty_like(x)
    p.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), rw.data_ptr(), m, K, stream.cuda_stream)
    torch.cuda.synchronize()
    return out


def err(got):
    return float((got.float() - ref.float()).norm() / ref.float().norm())


def failing_call(nth):
    """A call whose nth fused MoE call raises out of memory, as an allocation in there would."""
    original, seen = fused_moe.cutlass_fused_moe, []

    def fused(*args, **kwargs):
        seen.append(True)
        if len(seen) == nth:
            raise torch.OutOfMemoryError('CUDA out of memory (injected by the test)')
        return original(*args, **kwargs)

    fused_moe.cutlass_fused_moe = fused
    try:
        call(pre)
        return False
    except torch.OutOfMemoryError:
        return True
    finally:
        fused_moe.cutlass_fused_moe = original
        torch.cuda.synchronize()


def compare(name):
    e = err(call(pre))
    check(name, e < 1e-3, f'rel L2 {e:.1e}')


ref = call(reference)
pool.fill(list(range(BASE)), copy)
compare(f'A: good call, experts 0..{BASE - 1} in the pool')
check('A: a call fails in its first fused MoE call', failing_call(1))
pool.fill(list(range(100, 100 + BASE)), copy)
compare('A: the call again after the pool changed')

for a in pool.arena:   # extra slots get memory, as the elastic cache's growth maps it
    st = a[0].numel()
    assert lib.xtier_mailbox_vmm_map(C.c_void_p(a.data_ptr()), BASE * st, EXTRA * st) == 0
pool.active = BASE + EXTRA
hot = list(range(100, 100 + BASE)) + list(range(EXTRA))   # 0..6: in the first chunks, uploaded before the routes
pool.fill(hot, copy)
compare('B: good call, experts 0..6 in the extra slots')
check('B: a call fails in its 10th fused MoE call', failing_call(10))
table = pool.table.clone()
table[hot[BASE:]] = -1
pool.table.copy_(table)
torch.cuda.synchronize()
for a in pool.arena:   # Governor.release: the experts leave the table, then the extra memory goes
    assert lib.xtier_mailbox_vmm_unmap(C.c_void_p(a.data_ptr()), BASE * a[0].numel()) == 0
pool.active = BASE
torch.cuda.empty_cache()
compare('B: the call again after the extra slots were released')   # before the fix: faults here
print('PREFILL-FP4-RETRY-CHECK', 'PASS' if not bad else f'FAIL ({bad})', flush=True)
os._exit(1 if bad else 0)   # the engine's threads and the unmapped arena: no orderly teardown needed
