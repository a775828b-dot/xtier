"""The CPU engine's GPT-OSS SwiGLU (xtier_set_swiglu: gate * sigmoid(alpha * gate) * (up + beta) after the clamp,
MiniMax M3.1's routed experts) against an FP32 reference on one real layer: NVFP4 decoded in torch, one token at a time.
The same plan switched back to alpha 1 / beta 0 must match plain clamped SiLU. Token counts cover the engine's decode
and prefill paths.

  python tests/swiglu_oai_check.py --preset minimax_m31 --lengths 1,3,8,64,512
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, SWIGLU, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), default='minimax_m31')
P.add_argument('--threads', type=int, default=112)
P.add_argument('--lengths', default='1,3,8,64,512')
P.add_argument('--tolerance', type=float, default=5e-3)
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
ALPHA, BETA = SWIGLU.get(A.preset, (1.0, 0.0))
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)

import torch  # noqa: E402

from xtier.engine import CpuPlan  # noqa: E402

assert FMT == 'nvfp4', 'the reference decodes NVFP4 only'
w13, w2, s13, s2, g13, g2 = load(A.preset)
plan = CpuPlan(E, A.threads, [t.data_ptr() for t in (w13, w2, s13, s2, g13, g2)], hidden=H, inter=I,
               swiglu_limit=LIMIT, scale_format=FMT, swiglu_alpha=ALPHA, swiglu_beta=BETA)
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def dequant(w, s, g):
    """[R, C/2] packed E2M1 (low nibble first), [R, C/16] E4M3 scales, global multiplier -> FP32 [R, C]."""
    v = torch.stack([FP4[(w & 0xF).long()], FP4[(w >> 4).long()]], -1).reshape(w.shape[0], -1)
    return v * s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1) * float(g)


cache = {}


def expert(e):
    if e not in cache:
        cache[e] = (dequant(w13[e, :I], s13[e, :I], g13[e, 0]), dequant(w13[e, I:], s13[e, I:], g13[e, 1]),
                    dequant(w2[e], s2[e], g2[e]))
    return cache[e]


def reference(x, ids, rw, alpha, beta):
    out = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        xt = x[t].float()
        for j in range(K):
            wg, wu, wd = expert(int(ids[t, j]))
            gate, up = wg @ xt, wu @ xt
            if LIMIT > 0:
                up, gate = up.clamp(-LIMIT, LIMIT), gate.clamp(max=LIMIT)
            out[t] += float(rw[t, j]) * (wd @ (gate * torch.sigmoid(alpha * gate) * (up + beta)))
    return out


g = torch.Generator().manual_seed(20261024)
bad = 0
for alpha, beta, label in ((ALPHA, BETA, f'GPT-OSS alpha {ALPHA} beta {BETA}'), (1.0, 0.0, 'SiLU (alpha 1, beta 0)')):
    if plan.lib.xtier_set_swiglu(plan.plan, alpha, beta):
        raise RuntimeError(plan.lib.xtier_last_error().decode())
    for m in map(int, A.lengths.split(',')):
        x = (torch.randn(m, H, generator=g) * 0.5).bfloat16()
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
        rw = torch.softmax(torch.randn(m, K, generator=g), -1)
        got = torch.empty(m, H)
        plan.forward(m, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), got.data_ptr())
        rows = list(range(m)) if m <= 64 else sorted(torch.randperm(m, generator=g)[:48].tolist())
        ref = reference(x[rows], ids[rows], rw[rows], alpha, beta)
        err = float((got[rows] - ref).norm() / ref.norm())
        ok = err <= A.tolerance
        bad += not ok
        print(f'{label}, m={m}: rel L2 {err:.2e} over {len(rows)} tokens{"" if ok else " FAIL"}', flush=True)
plan.close()
print('SWIGLU-OAI-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
