"""Experiment: does other GPU work between mailbox calls (as attention between MoE layers in a decode step) add the
~28 us per call a serving model pays over the back-to-back probe? The engine's own time is the same in both (trace).

Per call in one CUDA graph of 48 calls on one synthetic Qwen3.8 layer (20 GPU + 6 CPU experts, m = 4):
  back-to-back            the probe's setting
  + GPU work between      a 64 MiB copy between calls (evicts L2, ~0.1 ms of HBM traffic), its own time subtracted
  CPU-only, same two settings
"""
import os
import sys

sys.path.insert(0, '/data/fast/workspaces/xtier')
os.environ.setdefault('XTIER_SPIN_COUNT', '1048576')
os.environ['XTIER_CACHE_ADAPT'] = '0'
os.environ.setdefault('XTIER_MAIN_CPUS', 'auto')

import torch  # noqa: E402

from xtier.backend import kernels  # noqa: E402
from xtier.engine import placement  # noqa: E402
from xtier.planner import probe  # noqa: E402

E, H, I, K, m, S = 512, 2560, 640, 10, 4, 155
placement.apply(112)
layer = probe.Layer(E, H, I, K, 'nvfp4', 112)
part = layer.pool(S, list(range(S)))
rng = torch.Generator().manual_seed(7)
src = torch.empty(64 << 20, dtype=torch.uint8, device='cuda')
dst = torch.empty_like(src)


def pick(lo, hi, n):
    return [lo + int(i) for i in torch.randperm(hi - lo, generator=rng)[:n]]


ids_part = []
for _ in range(48):
    g, c = pick(0, S, 20), pick(S, E, 6)
    ids_part.append(probe.routes([e for p in zip(g, c) for e in p] + g[len(c):], m, K).cuda())
ids_cpu = [probe.routes(pick(S, E, 6), m, K).cuda() for _ in range(48)]
x = (torch.randn(m, H, device='cuda') * 0.5).bfloat16()
rw = torch.softmax(torch.randn(m, K, device='cuda'), -1)
out = torch.empty(m, H, device='cuda', dtype=torch.float32)


def call_part(st, t):
    part.decode(st, m, K, x.data_ptr(), t.data_ptr(), rw.data_ptr(), out.data_ptr())


def call_cpu(st, t):
    slot = layer.box.slot(st, m, K)
    kernels.pack(slot, m, K, x.data_ptr(), t.data_ptr(), rw.data_ptr(), H, 0)
    layer.box.notify(st, m, K)
    kernels.unpack(slot, m, out.data_ptr(), H)


def timed(fn, ids, gap):
    s = torch.cuda.Stream()

    def body():
        for t in ids:
            if gap:
                dst.copy_(src)
            if fn is not None:
                fn(torch.cuda.current_stream().cuda_stream, t)
    with torch.cuda.stream(s):
        body()
    s.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        body()
    best = None
    with torch.cuda.stream(s):
        g.replay()
        s.synchronize()
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(s)
            for _ in range(20):
                g.replay()
            b.record(s)
            b.synchronize()
            us = a.elapsed_time(b) * 1e3 / (20 * len(ids))
            best = us if best is None else min(best, us)
    return best


gap_only = timed(None, ids_part, True)
for name, fn, ids in (('partial (20 GPU + 6 CPU)', call_part, ids_part), ('CPU only (6 experts)', call_cpu, ids_cpu)):
    b2b = timed(fn, ids, False)
    with_gap = timed(fn, ids, True) - gap_only
    print(f'{name}: back-to-back {b2b:6.1f} us, with GPU work between {with_gap:6.1f} us ({with_gap - b2b:+.1f})',
          flush=True)
print(f'(the 64 MiB copy alone: {gap_only:.1f} us)')
layer.close()
