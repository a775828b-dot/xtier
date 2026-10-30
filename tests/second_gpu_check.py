"""Second GPU as a hot expert pool (cache/second.py) on one real MoE layer of two GPUs:
  tests/second_gpu_check.py [--preset qwen38] [--threads 30] [--out DIR]
Runs the same decode calls in four processes, with 120 experts on GPUs, and compares:
  gpu0: all 120 on the first GPU; gpu2: all 120 on the second (the first holds one expert the routes never use);
  split: 40 + 80; cpu: none of them on a GPU.
gpu2 must equal gpu0 bit for bit (same kernels, same order); split differs only by the order of the FP32 sum; cpu by the
kernels' accuracy. CHECK_REPS=N repeats every call and reports repeats that differ from the first: a GPU's own instability
shows as differences that follow the GPU when CUDA_VISIBLE_DEVICES swaps them (seen on a rented RTX 4090, at 8 tokens)."""
import argparse
import ctypes as C
import json
import os
import random
import subprocess
import sys
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('--preset', default='qwen38')
P.add_argument('--threads', type=int, default=30)
P.add_argument('--out', default='/tmp/second_gpu_check')
P.add_argument('--run', default='')   # internal: first GPU's slots,second GPU's slots
A = P.parse_args()
SHAPES = [(1, 3), (2, 3), (4, 2), (6, 2), (8, 2)]   # tokens, calls
HOT = 120

if not A.run:
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = dict(gpu0=f'{HOT + 1},0', gpu2=f'1,{HOT}', split=f'41,{HOT - 40}', cpu='1,0')
    for name, run in cases.items():
        print(f'--- {name}: {run}', flush=True)
        subprocess.run([sys.executable, __file__, '--preset', A.preset, '--threads', str(A.threads), '--out',
                        str(out / f'{name}.pt'), '--run', run], check=True)
    import torch
    r = {k: torch.load(out / f'{k}.pt') for k in cases}
    ok = True
    for i, ref in enumerate(r['gpu0']):
        same = ref.numpy().tobytes() == r['gpu2'][i].numpy().tobytes()
        rel = lambda u: float((u - ref).norm() / ref.norm())
        split, cpu = rel(r['split'][i]), rel(r['cpu'][i])
        print(f'call {i:2d} m={ref.shape[0]}: gpu2 == gpu0 bit for bit: {same}; split rel L2 {split:.2e}; cpu rel L2 {cpu:.2e}')
        ok &= same and split < 1e-5 and cpu < 1e-2
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

first, second = [int(v) for v in A.run.split(',')]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, SWIGLU, load
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
ALPHA, BETA = SWIGLU.get(A.preset, (1.0, 0.0))
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)
order = list(range(E))
random.Random(1).shuffle(order)
unused, hot, cold = order[0], order[1:HOT + 1], order[HOT + 1:]
# the first GPU's slots come first in the profile; `unused` is never routed
profile = f'{A.out}.profile.json'
with open(profile, 'w') as f:
    json.dump(dict(experts=E, layer_ids=[LAYER], layers=[[unused] + hot + cold]), f)
os.environ.update(XTIER_CACHE_PROFILE=profile, XTIER_CACHE_ADAPT='1', XTIER_GPU2_SLOTS=str(second),
                  XTIER_GPU2_ADAPT='0', XTIER_GPU2_BALANCE='0')   # fixed residency: the same path every time
import torch
from xtier.backend.prefill import Prefill
from xtier.cache import pool as pool_module
from xtier.cache.pool import LayerPool
from xtier.engine import CpuPlan, Mailbox
weights = load(A.preset)
pointers = [t.data_ptr() for t in weights]
plan = CpuPlan(E, A.threads, pointers, hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT, swiglu_alpha=ALPHA, swiglu_beta=BETA)
plan.pool = None
prefill = Prefill(E, pointers, native=plan, hidden=H, inter=I, swiglu_limit=LIMIT, swiglu_alpha=ALPHA, swiglu_beta=BETA)
box = Mailbox(plan)
pool = LayerPool(prefill, first, plan, box, LAYER)
pool_module.fill_all()
assert (pool.second is not None) == (second > 0)
g = torch.Generator().manual_seed(7)
stream = torch.cuda.Stream()
results, differ = [], []
with torch.cuda.stream(stream):
    for m, calls in SHAPES:
        for _ in range(calls):
            ids = torch.empty(m, K, dtype=torch.int32)
            for t in range(m):   # half hot, half cold, in mixed order
                pick = [hot[i] for i in torch.randperm(HOT, generator=g)[:K - K // 2].tolist()]
                pick += [cold[i] for i in torch.randperm(len(cold), generator=g)[:K // 2].tolist()]
                ids[t] = torch.tensor(pick, dtype=torch.int32)[torch.randperm(K, generator=g)]
            x = torch.randn(m, H, generator=g).bfloat16().cuda()
            w = torch.softmax(torch.randn(m, K, generator=g), -1).cuda()
            ids = ids.cuda()
            out = torch.empty(m, H, dtype=torch.float32, device='cuda')
            kept = None
            for rep in range(max(2, int(os.environ.get('CHECK_REPS', '2')))):   # a shape's first call captures the graph
                out.zero_()
                pool.decode(stream.cuda_stream, m, K, x.data_ptr(), ids.data_ptr(), w.data_ptr(), out.data_ptr())
                torch.cuda.synchronize()
                if rep == 1:
                    kept = out.clone()
                elif rep > 1 and not torch.equal(out, kept):
                    differ.append((m, (out != kept).nonzero()[:, 0].unique().tolist()))
            results.append(kept.cpu())
torch.save(results, A.out)
stats = (C.c_double * 3)()
box.lib.xtier_mailbox_pool_stats(box.bridge, C.cast(stats, C.c_void_p))
print(f'calls {stats[0]:.0f}, distinct experts {stats[1]:.0f}, served by GPUs {stats[2]:.0f}; repeats that differ from the '
      f'first: {len(differ)}, tokens {sorted({m for m, _ in differ})}, rows {sorted({r for _, d in differ for r in d})}', flush=True)
os._exit(0)   # the bridges live until the process exits
