"""Cold experts in a GPU prefill (xtier/cache/cold.py PrefillCold, backend/prefill.py): one real layer (DeepSeek V4.1
Flash 1017 layer 20, MXFP4, the production prefill switches: FP4 tensor cores, 8-expert chunks) with (1 - X) of its
experts in RAM slots and the others in a store on the given drives, against the same layer with every expert in RAM.
The cold layer is built twice (two plans, as two consecutive layers: the first one's calls read the second one's
cold experts ahead) and both run each call in turn. Per call the virtual order puts the RAM slots first; the cold
experts are read into the staging while those upload (async), before the first upload (sync: all reads waited for
at once), or by the previous layer's call (the second layer, XTIER_COLD_READ_AHEAD=1). The stagings are filled with
0xFF before the first layer's call of every mode (the second layer's staging is then written only by reads), so a read
that is missing, or an unread expert that is computed, shows (NaN). Between calls
CPU forwards over random routes move experts in and out of RAM. Reported per call and layer: the relative L2 error
against the full layer (the route sums run in another order: the BF16 chunk outputs group the routes differently),
the time of both, the experts read and how long the uploads waited. After every call the layer's global scales must be
the per-expert ones again (_globals(): the hot pool and the hybrid prefill index them by expert; the call's virtual
order stays inside it — MXFP4's are all ones, so the check is on the object, not the values).

  python tests/cold_prefill_check.py --store DIR1 DIR2 --x 0.3 --lengths 1024,3000,8192
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--store', nargs='+', required=True, help='one directory per drive')
P.add_argument('--x', type=float, default=0.3)
P.add_argument('--threads', type=int, default=112)
P.add_argument('--gpu-slots', type=int, default=50)
P.add_argument('--lengths', default='1024,3000,8192')
P.add_argument('--subset', type=int, default=300, help='calls that route to only this many experts (0: none)')
P.add_argument('--tolerance', type=float, default=1e-2)
P.add_argument('--profile', default='/data/fast/workspaces/ds41-1017/deploy/lkqstrata-ds41-seed.json')
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS['ds41_1017']
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)
for name, value in dict(PREFILL_FP4='1', PREFILL_UNPACK='1', PREFILL_EXPERT_CHUNK='8', PREFILL_ROUTE_ACCUM='1',
                        COLD='1', COLD_PATHS=':'.join(A.store), COLD_FRACTION=str(A.x), COLD_KEY='cold-prefill-check',
                        CACHE_PROFILE=A.profile, COLD_LOG='0').items():
    os.environ['XTIER_' + name] = value

import torch  # noqa: E402

from xtier.backend.prefill import Prefill  # noqa: E402
from xtier.cache import cold  # noqa: E402
from xtier.engine import CpuPlan  # noqa: E402

weights = load('ds41_1017')
pointers = [t.data_ptr() for t in weights]
full = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT)
full.pool = None
reference = Prefill(E, pointers, native=full, hidden=H, inter=I, swiglu_limit=LIMIT)
layers = []   # (plan, PrefillCold, Prefill), in layer order
for _ in range(2):
    lp = cold.LayerPlan(LAYER, E, A.gpu_slots)
    plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT, resident=lp.resident)
    plan.pool = None
    pc = cold.attach(plan, lp, weights, FMT)
    prefill = Prefill(E, pointers, native=plan, hidden=H, inter=I, swiglu_limit=LIMIT)
    prefill.cold = pc
    layers.append((plan, pc, prefill))
del weights, pointers
print(f'layer {LAYER} twice: {plan.slots} of {E} experts in RAM, {pc.K} cold', flush=True)

g = torch.Generator().manual_seed(20261026)
stream = torch.cuda.current_stream()


def call(p, x, ids, rw, fill=True):
    """fill: both stagings to 0xFF first (not before a layer whose cold experts were read ahead into one)."""
    for staging in cold._staging if fill else ():
        if staging is not None:
            staging.fill_(255)
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = torch.empty_like(x)
    p.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), rw.data_ptr(), x.shape[0], K, stream.cuda_stream)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1e3


def shake(plan, calls=30):
    """CPU forwards of 8 tokens over random routes: experts move in and out of RAM."""
    for _ in range(calls):
        xs = torch.randn(8, H, generator=g).bfloat16()
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(8)]).int()
        rw = torch.softmax(torch.randn(8, K, generator=g), -1)
        out = torch.empty(8, H, dtype=torch.float32)
        plan.forward(8, K, xs.data_ptr(), ids.data_ptr(), rw.data_ptr(), out.data_ptr())


def sync_prepare(pc, prepare):   # all reads waited for before the first upload
    def run(globals_, ids, *rest):
        out = prepare(globals_, ids, *rest)
        pc.wait(0, pc.K)
        return out
    return run


originals = [pc.prepare for _, pc, _ in layers]
bad = 0
cases = [(int(m), 0) for m in A.lengths.split(',')] + ([(int(A.lengths.split(',')[0]), A.subset)] if A.subset else [])
for c, (m, subset) in enumerate(cases):
    x = torch.randn(m, H, generator=g).bfloat16().cuda()
    pool = torch.randperm(E, generator=g)[:subset] if subset else torch.arange(E)
    ids = torch.stack([pool[torch.randperm(len(pool), generator=g)[:K]] for _ in range(m)]).int().cuda()
    rw = torch.softmax(torch.randn(m, K, generator=g), -1).cuda()
    if c == 0:   # first-call work (registration, kernels, the stagings) outside the times
        call(reference, x, ids, rw)
        for _, _, prefill in layers:
            call(prefill, x, ids, rw)
    ref, t_ref = call(reference, x, ids, rw)
    for mode in ('sync', 'async', 'ahead'):
        for (plan, pc, _), prepare in zip(layers, originals):
            shake(plan)
            pc.prepare = sync_prepare(pc, prepare) if mode == 'sync' else prepare
        os.environ['XTIER_COLD_READ_AHEAD'] = '1' if mode == 'ahead' else '0'
        line = []
        for i, (plan, pc, prefill) in enumerate(layers):
            reads, waited = pc.reads, pc.read_s
            got, t = call(prefill, x, ids, rw, fill=i == 0)
            err = float((got.float() - ref.float()).norm() / ref.float().norm())
            ok = err <= A.tolerance   # NaN fails
            leaked = prefill._globals() is not prefill.device_globals
            bad += (not ok) + leaked
            line.append(f'layer {i + 1}: {t:.1f} ms, read {pc.reads - reads}, waited {(pc.read_s - waited) * 1e3:.1f} ms, '
                        f'rel L2 {err:.2e}{"" if ok else " FAIL"}{" GLOBALS IN VIRTUAL ORDER" if leaked else ""}')
        print(f'm={m}{f" ({subset} experts routed)" if subset else ""} {mode} (all in RAM {t_ref:.1f} ms): '
              + '; '.join(line), flush=True)
for (_, pc, prefill), prepare in zip(layers, originals):
    pc.prepare = prepare
    pc.finish()
    if pc.ahead is not None:
        pc.store.native.finish(pc.ahead[2])
    prefill.close()
reference.close()
for plan, _, _ in layers:
    plan.close()
full.close()
print('COLD-PREFILL-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
