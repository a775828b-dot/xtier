"""Decode path check on one real MoE layer: XTier's GPU side (packing kernel, mailbox handshake, unpacking kernel,
xtier/backend/kernels.py) against the CPU engine called directly, bitwise, eager and replayed from a CUDA graph.

  python tests/mailbox_check.py --preset qwen38 --threads 112
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), required=True)
P.add_argument('--threads', type=int, default=112)
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)

import torch  # noqa: E402

from xtier.backend import kernels  # noqa: E402
from xtier.engine import CpuPlan, Mailbox  # noqa: E402

weights = load(A.preset)
plan = CpuPlan(E, A.threads, [t.data_ptr() for t in weights], hidden=H, inter=I, swiglu_limit=LIMIT, scale_format=FMT)
del weights
box = Mailbox(plan)
g = torch.Generator().manual_seed(20261023)


def case(m):
    x = torch.randn(m, H, generator=g).bfloat16()
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
    rw = torch.softmax(torch.randn(m, K, generator=g), -1)
    ref = torch.empty(m, H)
    plan.forward(m, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), ref.data_ptr())
    return x, ids, rw, ref


def gpu_decode(x, ids, rw, out):
    stream = torch.cuda.current_stream().cuda_stream
    m = x.shape[0]
    slot = box.slot(stream, m, K)
    kernels.pack(slot, m, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), H)
    box.notify(stream, m, K)
    kernels.unpack(slot, m, out.data_ptr(), H)


bad = 0
for m in range(1, 9):
    x, ids, rw, ref = case(m)
    xd, idsd, rwd = x.cuda(), ids.cuda(), rw.cuda()
    out = torch.empty(m, H, device='cuda')
    gpu_decode(xd, idsd, rwd, out)
    torch.cuda.synchronize()
    eager = torch.equal(out.cpu(), ref)
    # graph: capture on a side stream with static inputs, replay with new inputs
    graph, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(s):
        gpu_decode(xd, idsd, rwd, out)   # warm-up of the capture stream's slot
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=s):
            gpu_decode(xd, idsd, rwd, out)
    replays = []
    for _ in range(3):
        x2, ids2, rw2, ref2 = case(m)
        xd.copy_(x2); idsd.copy_(ids2); rwd.copy_(rw2)
        graph.replay()
        torch.cuda.synchronize()
        replays.append(torch.equal(out.cpu(), ref2))
    ok = eager and all(replays)
    bad += not ok
    print(f'm={m}: eager {"=" if eager else "DIFFERENT"}, graph replays {"".join("=" if r else "X" for r in replays)}',
          flush=True)
    del graph
box.close()
plan.close()
print('MAILBOX-CHECK', 'PASS' if not bad else f'FAIL ({bad} sizes)')
