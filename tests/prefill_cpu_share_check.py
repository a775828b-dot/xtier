"""The FP4 GPU prefill's CPU share (xtier/backend/prefill.py XTIER_PREFILL_CPU_SHARE, _run_fp4 on SM120) on one real
layer: a call uploads the expert chunks that pay and the CPU engine computes the ones with few tokens at the same time.
Routes are drawn unevenly, as a prompt's are: a share of the experts get no token, many a few.

  A  the share on with nothing left to the CPU (XTIER_PREFILL_CPU_SHARE_TARGET=0): only the chunks without a token are
     not uploaded                                               -> bitwise the GPU-only output
  B  the share balanced (the default)                           -> chunks on the CPU; within the precision difference of
     the two paths (the CPU's experts take BF16 activations, the GPU's MXFP8) of the GPU-only output
  C  everything the uploads had not started on goes to the CPU  -> the same bound; closer to the engine's own output
  D  a call fails in a fused MoE call while its CPU part runs, the call again -> as B

  python tests/prefill_cpu_share_check.py [--preset ds41_1017] [--threads 112] [--tokens 12000]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), default='ds41_1017')
P.add_argument('--threads', type=int, default=112)
P.add_argument('--tokens', type=int, default=12000)
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)
for name, value in dict(PREFILL_FP4='1', PREFILL_UNPACK='1', PREFILL_EXPERT_CHUNK='8', PREFILL_ROUTE_ACCUM='1',
                        PREFILL_CPU_SHARE='0').items():
    os.environ['XTIER_' + name] = value

import torch  # noqa: E402

if torch.cuda.get_device_capability()[0] < 10:
    print('PREFILL-CPU-SHARE-CHECK SKIP (the FP4 prefill needs SM100 or newer)')
    sys.exit(0)

import flashinfer.fused_moe as fused_moe  # noqa: E402

from xtier.backend import prefill as prefill_module  # noqa: E402
from xtier.backend.prefill import Prefill  # noqa: E402
from xtier.engine import CpuPlan  # noqa: E402

weights = load(A.preset)
pointers = [t.data_ptr() for t in weights]
plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT)
plan.pool = None
pre = Prefill(E, pointers, native=plan, hidden=H, inter=I, swiglu_limit=LIMIT)
if not pre._fp4_ready(torch.bfloat16):
    print(f'PREFILL-CPU-SHARE-CHECK SKIP ({A.preset}: not an MXFP4 layer the FP4 prefill takes)')
    sys.exit(0)
CH = pre.chunk
stream = torch.cuda.current_stream()
g = torch.Generator().manual_seed(20261111)
m = A.tokens
x = torch.randn(m, H, generator=g).bfloat16().cuda()
# uneven routes: five whole chunks and a fifth of the other experts get no token, the others' shares fall off steeply
# (shuffled over the chunks)
share = torch.zeros(E)
alive = torch.ones(E, dtype=torch.bool)
for dead in (3, 17, 30, 41, 46):
    alive[dead * CH:(dead + 1) * CH] = False
alive[torch.randperm(E, generator=g)[:E // 5]] = False
live = alive.nonzero().squeeze(1)
share[live] = torch.arange(1, len(live) + 1, dtype=torch.float32).pow(-1.3)[torch.randperm(len(live), generator=g)]
ids = torch.multinomial(share.expand(m, E), K, generator=g).int().cuda()
rw = torch.softmax(torch.randn(m, K, generator=g), -1).cuda()
per_chunk = torch.bincount(ids.long().flatten() // CH, minlength=E // CH)
print(f'{m} tokens x top-{K} over {E} experts in chunks of {CH}: {int((per_chunk == 0).sum())} of {E // CH} chunks without '
      f'a route, median {int(per_chunk.median())} routes a chunk, largest {int(per_chunk.max())}', flush=True)
bad = 0


def check(name, ok, detail=''):
    global bad
    print(f'{name}: {"ok" if ok else "FAIL"} {detail}'.rstrip(), flush=True)
    bad += not ok


def call(share_on, target='1'):
    os.environ['XTIER_PREFILL_CPU_SHARE'] = '1' if share_on else '0'
    os.environ['XTIER_PREFILL_CPU_SHARE_TARGET'] = target
    seen = dict(prefill_module._share_seen)
    out = torch.empty_like(x)
    pre.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), rw.data_ptr(), m, K, stream.cuda_stream)
    torch.cuda.synchronize()
    now = prefill_module._share_seen
    return out, {k: now[k] - seen[k] for k in ('chunks', 'skipped', 'cpu_chunks', 'cpu_routes')}


def err(got, ref):
    return float((got.float() - ref.float()).norm() / ref.float().norm())


def cpu_only():
    """The engine alone on every route (FP32 out)."""
    xh, ih, wh = x.cpu().contiguous(), ids.cpu().contiguous(), rw.cpu().contiguous()
    out = torch.empty((m, H), dtype=torch.float32)
    for s in range(0, m, 2048):
        n = min(2048, m - s)
        plan.forward(n, K, xh.data_ptr() + s * H * 2, ih.data_ptr() + s * K * 4, wh.data_ptr() + s * K * 4,
                     out.data_ptr() + s * H * 4)
    return out.cuda()


prefill_module._share_seen['logged'] = float('inf')   # no log lines here: the counters are read
ref, _ = call(False)
engine = cpu_only()
paths = err(ref, engine)   # the two paths' precision difference on this layer, every route on either
print(f'GPU-only against the engine alone: rel L2 {paths:.1e}', flush=True)

out, d = call(True, '0')   # the chunks whose upload started before the routes were known stay on the GPU
same = torch.equal(out, ref)
check('A: no chunk on the CPU, the chunks without a route not uploaded', d['cpu_chunks'] == 0 and d['skipped'] > 0
      and same, f"({d['skipped']} of {d['chunks']} chunks not uploaded, bitwise {same})")
for _ in range(3):   # the costs settle over a few calls
    out, d = call(True)
e = err(out, ref)
check('B: balanced', d['cpu_chunks'] > 0 and e < 1.5 * paths + 1e-3,
      f"({d['cpu_chunks']} chunks with {d['cpu_routes']} routes on the CPU, {d['skipped']} of {d['chunks']} not uploaded; "
      f'rel L2 to GPU-only {e:.1e}, to the engine {err(out, engine):.1e})')
out_c, d_c = call(True, '1e9')
e = err(out_c, ref)
check('C: every chunk the uploads had not started on is the CPU\'s', d_c['cpu_chunks'] > d['cpu_chunks']
      and e < 1.5 * paths + 1e-3 and err(out_c, engine) < err(out, engine),
      f"({d_c['cpu_chunks']} chunks, {d_c['cpu_routes']} routes; rel L2 to GPU-only {e:.1e}, to the engine "
      f'{err(out_c, engine):.1e})')

original, seen = fused_moe.cutlass_fused_moe, []


def failing(*args, **kwargs):
    seen.append(True)
    if len(seen) == 3:
        raise torch.OutOfMemoryError('CUDA out of memory (injected by the test)')
    return original(*args, **kwargs)


fused_moe.cutlass_fused_moe = failing
try:
    call(True)
    failed = False
except torch.OutOfMemoryError:
    failed = True
finally:
    fused_moe.cutlass_fused_moe = original
    torch.cuda.synchronize()
check('D: a call fails in its third fused MoE call', failed)
out, d = call(True)
e = err(out, ref)
check('D: the call again', d['cpu_chunks'] > 0 and e < 1.5 * paths + 1e-3, f'(rel L2 to GPU-only {e:.1e})')
print('PREFILL-CPU-SHARE-CHECK', 'PASS' if not bad else f'FAIL ({bad})', flush=True)
os._exit(1 if bad else 0)   # the engine's threads: no orderly teardown needed
