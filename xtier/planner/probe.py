"""Planner probe: the decode cost of one MoE layer of a given shape on this host, measured on XTier's own paths.

Synthetic weights of the model's shape and format (random FP4 codes, unit scales: the kernels' speed does not depend
on the values) go to the CPU engine and the expert cache exactly as a model layer would. Each measurement replays a
CUDA graph of R decode calls of m tokens, every call with other routes, so the weights come from DRAM / HBM and not
from the caches, as in a real decode step where the other layers' weights pass in between. Measured, per call:

- gpu(n):  a fully cached layer (S = E, the GPU alone) whose m tokens use n distinct experts;
- cpu(n, r): a layer without cache (plain mailbox path: pack, CPU engine, unpack) with n distinct experts and r
  routes (token-expert pairs; the engine reads an expert once and computes each of its tokens, which costs about as
  much as the read on large experts);
- part(g, c): a partly cached layer, g distinct experts in its slots and c on the CPU (both sides run at once);
- h2d: host-to-device bandwidth of the GPU prefill's upload (pinned engine shards to device memory).

The fit (xtier/planner/cost.py) turns them into per-call costs: gpu(n) = g0 + g1 n, cpu(n, r) = c0 + c1 n + c2 r
and part(g, c) = max(gpu(g) + p0, cpu(c, r_c) + p1).
"""
import json
import os
import socket
import time

import torch

FORMAT_GROUP = {'nvfp4': 16, 'mxfp4': 32}


def expert_bytes(H, I, fmt):
    """Bytes of one expert in the engine / cache layout: packed FP4 gate+up and down, one scale byte per group."""
    g = FORMAT_GROUP[fmt]
    return 2 * I * H // 2 + H * I // 2 + 2 * I * H // g + H * I // g


def synthetic(E, H, I, fmt, seed=20261024):
    """(w13, w2, s13, s2, g13, g2) host tensors in the engine's layout: random FP4 codes, scales of 1.0."""
    g = FORMAT_GROUP[fmt]
    gen = torch.Generator().manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=gen)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=gen)
    one = 0x38 if fmt == 'nvfp4' else 127   # E4M3 1.0 / UE8M0 2^0
    s13 = torch.full((E, 2 * I, H // g), one, dtype=torch.uint8)
    s2 = torch.full((E, H, I // g), one, dtype=torch.uint8)
    return w13, w2, s13, s2, torch.ones((E, 2), dtype=torch.float32), torch.ones((E,), dtype=torch.float32)


def routes(experts, m, k, r=None):
    """[m, k] int32 routes over exactly the given n distinct experts: token after token, min(k, n) routes each, the
    experts taken in turn (so a token's routes are distinct and all n are used once r >= n); after r routes (default:
    all m min(k, n)) the rest are -1, routes the engine and the cache skip."""
    n = len(experts)
    per = min(k, n)
    r = m * per if r is None else min(r, m * per)
    out = [[-1] * k for _ in range(m)]
    i = 0
    for t in range(m):
        for j in range(per):
            if i < r:
                out[t][j] = experts[i % n]
                i += 1
    return torch.tensor(out, dtype=torch.int32)


class Layer:
    """One synthetic MoE layer on the CPU engine with its prefill (upload) and mailbox."""

    def __init__(self, E, H, I, k, fmt, threads, limit=0.0, alpha=1.0, beta=0.0):
        from xtier.backend.prefill import Prefill
        from xtier.engine import CpuPlan, Mailbox
        self.E, self.H, self.I, self.k, self.fmt = E, H, I, k, fmt
        weights = synthetic(E, H, I, fmt)
        pointers = [t.data_ptr() for t in weights]
        self.plan = CpuPlan(E, threads, pointers, hidden=H, inter=I, swiglu_limit=limit, scale_format=fmt,
                            swiglu_alpha=alpha, swiglu_beta=beta)
        self.plan.pool = None
        self.prefill = Prefill(E, pointers, native=self.plan, hidden=H, inter=I, swiglu_limit=limit,
                               swiglu_alpha=alpha, swiglu_beta=beta)
        del weights, pointers
        self.box = Mailbox(self.plan)
        self.pools = []

    def pool(self, slots, experts):
        from xtier.cache.pool import LayerPool
        p = LayerPool(self.prefill, slots, self.plan, self.box, 1000 + len(self.pools))
        p.fill(experts, torch.cuda.Stream())
        self.pools.append(p)
        return p

    def close(self):
        torch.cuda.synchronize()
        self.box.close()
        self.prefill.close()
        self.plan.close()


def _graph_time(call, id_sets, m, H, reps):
    """Microseconds per call: one CUDA graph of len(id_sets) calls (each with its own routes), replayed reps times."""
    x = (torch.randn(m, H, device='cuda') * 0.5).bfloat16()
    rw = torch.softmax(torch.randn(m, id_sets[0].shape[1], device='cuda'), -1)
    out = torch.empty(m, H, device='cuda', dtype=torch.float32)
    ids = [t.cuda() for t in id_sets]
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for t in ids:   # eager first: creates the mailbox slots, compiles the kernels
            call(torch.cuda.current_stream().cuda_stream, x, t, rw, out)
    s.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for t in ids:
            call(torch.cuda.current_stream().cuda_stream, x, t, rw, out)
    with torch.cuda.stream(s):
        g.replay()
        s.synchronize()
        best = None
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(s)
            for _ in range(reps):
                g.replay()
            b.record(s)
            b.synchronize()
            us = a.elapsed_time(b) * 1e3 / (reps * len(ids))
            best = us if best is None else min(best, us)
    del g
    return best


def measure(E, H, I, k, fmt, threads, m=4, calls=32, reps=10, slots=None, limit=0.0, alpha=1.0, beta=0.0,
            seed=1, log=print):
    """Probe one layer shape; returns the JSON-ready result (raw points and the fit)."""
    from xtier.planner import cost
    os.environ['XTIER_CACHE_ADAPT'] = '0'
    os.environ.setdefault('XTIER_MAIN_CPUS', 'auto')
    from xtier.backend import kernels
    from xtier.engine import placement
    placement.apply(threads)   # as in a server: the dispatcher and this thread off the expert workers' CPUs
    t0 = time.time()
    layer = Layer(E, H, I, k, fmt, threads, limit, alpha, beta)
    S = slots or E // 2
    full = layer.pool(E, [])                 # every expert cached: the GPU alone
    part = layer.pool(S, list(range(S)))     # experts [0, S) cached (pools exist before any graph capture)
    log(f'probe: layer E {E} H {H} I {I} top-{k} {fmt}, {threads} threads, dispatcher CPU '
        f'{os.environ.get("XTIER_PROBE_DISPATCH_CPU")}, built in {time.time() - t0:.1f} s')
    rng = torch.Generator().manual_seed(seed)
    nmax = min(E, m * k)
    grid = sorted({n for n in (1, 2, k // 2, k, (k + nmax) // 2, nmax) + tuple(range(k, nmax + 1, max(1, (nmax - k) // 6)))
                   if n >= 1})

    def pick(lo, hi, n):
        return [lo + int(i) for i in torch.randperm(hi - lo, generator=rng)[:n]]

    out = dict(version=2, host=socket.gethostname(), gpu=torch.cuda.get_device_name(), threads=threads, m=m,
               calls=calls, model=dict(E=E, H=H, I=I, k=k, format=fmt, expert_bytes=expert_bytes(H, I, fmt)),
               engine_isa=None, gpu_full=[], cpu_only=[], partial=[])
    from xtier.engine import isa
    out['engine_isa'] = isa()

    # GPU alone: every expert cached (the GPU part reads each used expert once, whatever its token count)
    for n in grid:
        us = _graph_time(lambda st, x, t, rw, o: full.decode(st, m, k, x.data_ptr(), t.data_ptr(), rw.data_ptr(),
                                                             o.data_ptr()),
                         [routes(pick(0, E, n), m, k) for _ in range(calls)], m, H, reps)
        out['gpu_full'].append([n, us])
        log(f'probe: gpu   n {n:3d}: {us:7.1f} us')

    # CPU alone: the plain mailbox path of an uncached layer, n distinct experts and r routes (token-expert pairs:
    # the engine reads an expert once and computes each of its tokens)
    def plain(st, x, t, rw, o):
        slot = layer.box.slot(st, m, k)
        kernels.pack(slot, m, k, x.data_ptr(), t.data_ptr(), rw.data_ptr(), H, 0)
        layer.box.notify(st, m, k)
        kernels.unpack(slot, m, o.data_ptr(), H)
    for n in grid:
        top = m * min(k, n)
        for r in sorted({n, (n + top) // 2, top}):
            us = _graph_time(plain, [routes(pick(0, E, n), m, k, r) for _ in range(calls)], m, H, reps)
            out['cpu_only'].append([n, r, us])
            log(f'probe: cpu   n {n:3d} r {r:3d}: {us:7.1f} us')

    # partly cached: S slots hold experts [0, S); g of the call's experts from them, c from the others (the GPU and
    # the CPU work at once); the CPU part's routes are counted
    for g in sorted({0, 1, nmax // 4, nmax // 2, (3 * nmax) // 4, nmax - k // 2, nmax}):
        for c in sorted({0, 1, max(1, nmax // 8), nmax // 4, nmax // 2}):
            if g + c < 1 or g + c > nmax or g > S or c > E - S:
                continue
            sets = []
            for _ in range(calls):
                gpu_e, cpu_e = pick(0, S, g), pick(S, E, c)
                mixed = [e for pair in zip(gpu_e, cpu_e) for e in pair] + gpu_e[len(cpu_e):] + cpu_e[len(gpu_e):]
                sets.append(routes(mixed, m, k))
            rc = sum(int((t >= S).sum()) for t in sets) / len(sets)
            us = _graph_time(lambda st, x, t, rw, o: part.decode(st, m, k, x.data_ptr(), t.data_ptr(), rw.data_ptr(),
                                                                 o.data_ptr()), sets, m, H, reps)
            out['partial'].append([g, c, rc, us])
            log(f'probe: part  g {g:3d} c {c:3d} (CPU routes {rc:4.1f}): {us:7.1f} us')

    # prefill upload: all experts, in the GPU prefill's chunks
    chunk = max(1, min(E, 64))
    slot = [torch.empty((chunk,) + tuple(s[1:]), dtype=torch.uint8, device='cuda') for s in layer.prefill.shapes[:4]]
    s = torch.cuda.Stream()
    layer.prefill._upload_chunk(slot, 0, chunk, s.cuda_stream)   # registers the shards (pinned) first
    s.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(s)
    for e0 in range(0, E - E % chunk, chunk):
        layer.prefill._upload_chunk(slot, e0, e0 + chunk, s.cuda_stream)
    b.record(s)
    b.synchronize()
    moved = (E - E % chunk) * expert_bytes(H, I, fmt)
    out['h2d_gbps'] = moved / (a.elapsed_time(b) * 1e-3) / 1e9
    log(f'probe: upload {moved / 2**30:.2f} GiB at {out["h2d_gbps"]:.1f} GB/s')
    layer.close()
    out['fit'] = cost.fit(out)
    log('probe: fit ' + json.dumps(out['fit']))
    return out
