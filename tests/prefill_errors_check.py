"""Errors in the prefills that run on the GPU and the CPU at once (xtier/cache/prefill.py hybrid_prefill, the expert split
of xtier/backend/method.py), on a synthetic layer: an error of the CPU part reaches the caller instead of an unfinished
output; an error of the GPU part (out of memory: the elastic cache retries the forward) reaches it only after the CPU
part ended, and the same plan then takes the next call.

  python tests/prefill_errors_check.py
"""
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['XTIER_CACHE_ADAPT'] = '0'
os.environ.setdefault('XTIER_MAIN_CPUS', 'auto')

import torch  # noqa: E402

from xtier.backend.method import XTierMoEMethod  # noqa: E402
from xtier.cache import prefill as cache_prefill  # noqa: E402
from xtier.engine import placement  # noqa: E402
from xtier.planner import probe  # noqa: E402

E, H, I, K, THREADS = 64, 2560, 640, 8, 16
M = 300
placement.apply(THREADS)
layer = probe.Layer(E, H, I, K, 'nvfp4', THREADS)
pool = layer.pool(E // 2, list(range(E // 2)))
g = torch.Generator().manual_seed(20261031)
# small inputs: random FP4 weights give activations far above a real layer, over FP16 in the W4A16 kernels
x = (torch.randn(M, H, generator=g) * 0.01).bfloat16().cuda()
ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).int().cuda()
weights = torch.softmax(torch.randn(M, K, generator=g), -1).cuda()
bad = 0


def check(name, ok, detail=''):
    global bad
    print(f'{name}: {"ok" if ok else "FAIL"} {detail}'.rstrip(), flush=True)
    bad += not ok


class SlowPlan:
    """The real plan, the CPU part taking a while, or failing."""

    def __init__(self, fail=False):
        self.fail, self.ended = fail, False

    def forward(self, *args):
        time.sleep(0.5)
        if self.fail:
            raise RuntimeError('CPU part failed')
        out = layer.plan.forward(*args)
        self.ended = True
        return out


def raised(fn):
    try:
        fn()
    except RuntimeError as e:
        return str(e)
    return None


class FailingKernel:
    def __getitem__(self, grid):
        raise RuntimeError('out of memory in the GPU part')


good = cache_prefill.hybrid_prefill(pool, layer.plan, x, ids, weights)
check('hybrid prefill, no errors', bool(torch.isfinite(good).all()))

check('hybrid prefill, CPU part fails',
      raised(lambda: cache_prefill.hybrid_prefill(pool, SlowPlan(fail=True), x, ids, weights)) == 'CPU part failed')

plan = SlowPlan()
kernel, cache_prefill._gate_up_tiles = cache_prefill._gate_up_tiles, FailingKernel()
message = raised(lambda: cache_prefill.hybrid_prefill(pool, plan, x, ids, weights))
ended = plan.ended
cache_prefill._gate_up_tiles = kernel
check('hybrid prefill, GPU part fails', message == 'out of memory in the GPU part', str(message))
check('hybrid prefill, the CPU part ended before the error came out', ended)
again = cache_prefill.hybrid_prefill(pool, layer.plan, x, ids, weights)
torch.cuda.synchronize()
check('hybrid prefill, the plan takes the retry', float((again - good).norm() / good.norm()) < 1e-5)

split = types.SimpleNamespace(hidden=H, experts=E, _split_experts=lambda m: E // 2, prefill=layer.prefill,
                              plan=SlowPlan(fail=True))
check('split prefill, CPU part fails',
      raised(lambda: XTierMoEMethod._split_prefill(split, x, ids, weights, M, K)) == 'CPU part failed')
split.plan = layer.plan
XTierMoEMethod._split_cost = dict(cpu_ms=1.0, gpu_ms=1.0)
out = XTierMoEMethod._split_prefill(split, x, ids, weights, M, K)
torch.cuda.synchronize()
check('split prefill, no errors', bool(torch.isfinite(out).all()), str(split.last_split))
print('PREFILL-ERRORS-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
sys.exit(1 if bad else 0)
