"""Experiment: where do the ~28 us per mailbox call that a serving model pays over the one-layer probe come from?

Partly cached calls (20 GPU + 6 CPU experts, m = 4, Qwen3.8 shape) in one CUDA graph of 48 calls:
  A  every call on the same layer (the probe's setting)
  B  the calls cycle over L separate layers (own CPU plan, mailbox, cache), as a decode step does
  C  B while a Python thread busy-spins on the process's main CPUs (SGLang's scheduler polls; its CPUs are the SMT
     siblings of expert workers)
  D  B with the adaptive residency on (policy thread, usage statistics, swaps), as in serving

  python planner_layers_exp.py [L]
"""
import os
import sys
import threading
import time

sys.path.insert(0, '/data/fast/workspaces/xtier')
os.environ.setdefault('XTIER_SPIN_COUNT', '1048576')
os.environ.setdefault('XTIER_MAIN_CPUS', 'auto')
mode_adapt = '--adapt' in sys.argv
os.environ['XTIER_CACHE_ADAPT'] = '1' if mode_adapt else '0'
args = [a for a in sys.argv[1:] if not a.startswith('--')]

import torch  # noqa: E402

from xtier.engine import placement  # noqa: E402
from xtier.planner import probe  # noqa: E402

E, H, I, K, m, S = 512, 2560, 640, 10, 4, 155
L = int(args[0]) if args else 8
placement.apply(112)
t0 = time.time()
layers = [probe.Layer(E, H, I, K, 'nvfp4', 112) for _ in range(L)]
pools = [ly.pool(S, list(range(S))) for ly in layers]
print(f'{L} layers built in {time.time() - t0:.0f} s', flush=True)
if mode_adapt:
    from xtier.cache import pool as cpool
    cpool.fill_all()   # attaches the partly cached pools to the adaptive policy (filled already: only the policy)
rng = torch.Generator().manual_seed(5)


def pick(lo, hi, n):
    return [lo + int(i) for i in torch.randperm(hi - lo, generator=rng)[:n]]


def sets(n):
    out = []
    for _ in range(n):
        g, c = pick(0, S, 20), pick(S, E, 6)
        out.append(probe.routes([e for p in zip(g, c) for e in p] + g[len(c):], m, K))
    return out


def run(which, tag):
    """which(i) -> pool of call i; one graph of 48 calls."""
    ids = [t.cuda() for t in sets(48)]
    x = (torch.randn(m, H, device='cuda') * 0.5).bfloat16()
    rw = torch.softmax(torch.randn(m, K, device='cuda'), -1)
    out = torch.empty(m, H, device='cuda', dtype=torch.float32)
    s = torch.cuda.Stream()

    def body():
        for i, t in enumerate(ids):
            which(i).decode(torch.cuda.current_stream().cuda_stream, m, K, x.data_ptr(), t.data_ptr(), rw.data_ptr(),
                            out.data_ptr())
    with torch.cuda.stream(s):
        body()
    s.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        body()
    if mode_adapt:
        from xtier.cache import pool as cpool
        cpool.maybe_start()
    best = None
    with torch.cuda.stream(s):
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(s)
            for _ in range(20):
                g.replay()
            b.record(s)
            b.synchronize()
            us = a.elapsed_time(b) * 1e3 / (20 * len(ids))
            best = us if best is None else min(best, us)
    print(f'{tag}: {best:6.1f} us per call', flush=True)
    return best


a = run(lambda i: pools[0], 'A one layer')
b = run(lambda i: pools[i % L], f'B {L} layers in turn')
import subprocess
main = sorted(os.sched_getaffinity(0))
print(f'main CPUs of this process: {main}', flush=True)
results = {}
for nspin in (1, 2, 4):
    procs = [subprocess.Popen(['taskset', '-c', str(main[i % len(main)]), sys.executable, '-c', 'while True: pass'])
             for i in range(nspin)]
    time.sleep(1)
    results[nspin] = run(lambda i: pools[i % L], f'C {L} layers + {nspin} spinning processes on main CPUs '
                                                  f'{[main[i % len(main)] for i in range(nspin)]}')
    for pr in procs:
        pr.kill()
    time.sleep(1)
print(f'B - A {b - a:+.1f} us; C - B ' + ', '.join(f'{n} spinning: {v - b:+.1f} us' for n, v in results.items()),
      flush=True)
for ly in layers:
    ly.close()
