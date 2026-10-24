"""Experiment: does the mailbox dispatcher's per-call latency grow with the number of registered slots?

The dispatcher thread scans every registered slot of every layer for a request (csrc/mailbox Dispatcher). A server has
one set of slots (decode shapes) per layer that uses the mailbox, so with many partly cached layers each request waits
for a longer scan. The planner's probe registers one layer's slots only. Here: the probe's partial and CPU-only calls
on one synthetic Qwen3.8 layer, before and after registering EXTRA idle slots (other streams / m / k, never signalled).

  python planner_slots_exp.py [EXTRA ...]
"""
import os
import sys

sys.path.insert(0, '/data/fast/workspaces/xtier')
sys.path.insert(0, '/data/fast/workspaces/xtier/tests')
os.environ.setdefault('XTIER_SPIN_COUNT', '1048576')
os.environ['XTIER_CACHE_ADAPT'] = '0'
os.environ.setdefault('XTIER_MAIN_CPUS', 'auto')

import torch  # noqa: E402

from xtier.backend import kernels  # noqa: E402
from xtier.engine import placement  # noqa: E402
from xtier.planner import probe  # noqa: E402

E, H, I, K, m = 512, 2560, 640, 10, 4
placement.apply(112)
layer = probe.Layer(E, H, I, K, 'nvfp4', 112)
S = 256
part = layer.pool(S, list(range(S)))
rng = torch.Generator().manual_seed(3)


def pick(lo, hi, n):
    return [lo + int(i) for i in torch.randperm(hi - lo, generator=rng)[:n]]


def measure(tag):
    sets = []
    for _ in range(32):
        g, c = pick(0, S, 20), pick(S, E, 6)
        mixed = [e for pair in zip(g, c) for e in pair] + g[len(c):]
        sets.append(probe.routes(mixed, m, K))
    us_part = probe._graph_time(lambda st, x, t, rw, o: part.decode(st, m, K, x.data_ptr(), t.data_ptr(), rw.data_ptr(),
                                                                    o.data_ptr()), sets, m, H, 10)

    def plain(st, x, t, rw, o):
        slot = layer.box.slot(st, m, K)
        kernels.pack(slot, m, K, x.data_ptr(), t.data_ptr(), rw.data_ptr(), H, 0)
        layer.box.notify(st, m, K)
        kernels.unpack(slot, m, o.data_ptr(), H)
    us_cpu = probe._graph_time(plain, [probe.routes(pick(0, E, 6), m, K) for _ in range(32)], m, H, 10)
    print(f'{tag}: partial (20 GPU + 6 CPU experts) {us_part:6.1f} us, CPU only (6 experts) {us_cpu:6.1f} us', flush=True)


measure('slots of one layer')
streams = [torch.cuda.Stream() for _ in range(16)]
registered = 0
targets = [int(a) for a in sys.argv[1:]] or [64, 256, 512]
for target in targets:
    for st in streams:
        for kk in range(1, K + 1):
            for mm in range(1, 9):
                for routes in (False, True):
                    if registered >= target:
                        break
                    if kk == K and mm == m:
                        continue
                    layer.box.slot(st.cuda_stream, mm, kk, with_routes=routes)
                    registered += 1
    measure(f'+{registered} idle slots')
layer.close()
