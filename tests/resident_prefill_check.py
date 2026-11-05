"""Prefill of a fully cached layer (xtier/cache/prefill.py, packed slots in place) against the GPU prefill through its
staging (device copies from the slots) and BF16 unpack (backend/prefill.py) on one real layer: relative L2 error and time
per call. The two compute differently: the cached path's activation is 16-bit between gate and down (W4A16, as decode:
BF16 with MXFP4, FP16 with NVFP4), the staged path's FP32, read by the tensor cores in XTIER_DOWN_PRECISION. Here the
staged path's down runs in BF16, the closest it gets: 1e-4 to 2e-4 (its default tf32x3: 2.5e-3).

  python tests/resident_prefill_check.py --preset qwen38 --lengths 100,1000,4096,8192
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, SITU, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), required=True)
P.add_argument('--threads', type=int, default=16)
P.add_argument('--lengths', default='100,1000,4096,8192')
P.add_argument('--tolerance', type=float, default=3e-4)
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
for name, value in dict(PREFILL_EXPERT_CHUNK='32', PREFILL_ROUTE_ACCUM='1', PREFILL_UNPACK='1',
                        GPU_PREFILL_CHUNK='8192', DOWN_PRECISION='bf16').items():
    os.environ.setdefault('XTIER_' + name, value)

import torch  # noqa: E402

from xtier.backend.prefill import Prefill  # noqa: E402
from xtier.cache.prefill import resident_prefill  # noqa: E402
from xtier.engine import CpuPlan  # noqa: E402

weights = load(A.preset)
pointers = [t.data_ptr() for t in weights]
SITU_BETA, SITU_LIN = SITU.get(A.preset, (0.0, 0.0))
plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT, situ_beta=SITU_BETA,
               situ_linear=SITU_LIN)
plan.pool = None
prefill = Prefill(E, pointers, native=plan, hidden=H, inter=I, swiglu_limit=LIMIT, situ_beta=SITU_BETA,
                  situ_linear=SITU_LIN)
arena = [torch.empty(s, dtype=torch.uint8, device='cuda') for s in prefill.shapes[:4]]
stream = torch.cuda.current_stream()
prefill._upload_chunk(arena, 0, E, stream.cuda_stream)
torch.cuda.synchronize()
del weights, pointers


def from_arena(slot, e0, e1, s, up_first=False):   # the staged path of a fully cached layer: device copies
    with torch.cuda.stream(torch.cuda.ExternalStream(s)):
        for dst, src in zip(slot, arena):
            dst[:e1 - e0].copy_(src[e0:e1], non_blocking=True)


prefill._upload_chunk = from_arena
g = torch.Generator().manual_seed(20261023)


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) / reps * 1e3


bad = 0
for m in map(int, A.lengths.split(',')):
    x = torch.randn(m, H, generator=g).bfloat16().cuda()
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int().cuda()
    rw = torch.softmax(torch.randn(m, K, generator=g), -1).cuda()

    def staged():
        out = torch.empty_like(x)
        prefill.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), rw.data_ptr(), m, K, stream.cuda_stream)
        return out

    ref, t_staged = timed(staged)
    got, t_packed = timed(lambda: resident_prefill(prefill, arena, x, ids, rw))
    err = float((got.float() - ref.float()).norm() / ref.float().norm())
    ok = err <= A.tolerance
    bad += not ok
    print(f'm={m}: staged {t_staged:.2f} ms (device copies + unpack), packed in place {t_packed:.2f} ms, rel L2 {err:.2e}'
          f'{"" if ok else " FAIL"}', flush=True)
prefill.close()
plan.close()
print('RESIDENT-PREFILL-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
sys.exit(0 if not bad else 1)
