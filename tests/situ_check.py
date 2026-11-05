"""The CPU engine's SiTU-GLU (xtier_set_situ: beta * tanh(gate / beta) * sigmoid(gate) * up, up soft-clipped to
linear * tanh(up / linear); Kimi K3 / K3.1's routed experts) against an FP32 reference on one real layer: MXFP4 decoded
in torch. A second plan without SiTU must match plain SiLU. Token counts cover the engine's decode and prefill paths;
top-k 16 and 896 experts are the engine's new limits.

  python tests/situ_check.py --preset kimi_k31 --lengths 1,3,8,64,512
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, SITU, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(SITU), default='kimi_k31')
P.add_argument('--threads', type=int, default=112)
P.add_argument('--lengths', default='1,3,8,64,512')
P.add_argument('--tolerance', type=float, default=5e-3)
A = P.parse_args()
MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[A.preset]
BETA, LINEAR = SITU[A.preset]
for name, value in SWITCHES.items():
    os.environ.setdefault('XTIER_' + name, value)

import torch  # noqa: E402

from xtier.engine import CpuPlan  # noqa: E402

assert FMT == 'mxfp4', 'the reference decodes MXFP4 only'
w13, w2, s13, s2, g13, g2 = load(A.preset)
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def dequant(w, s):
    """[R, C/2] packed E2M1 (low nibble first), [R, C/32] UE8M0 scales -> FP32 [R, C]."""
    v = torch.stack([FP4[(w & 0xF).long()], FP4[(w >> 4).long()]], -1).reshape(w.shape[0], -1)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)


def act(gate, up, situ):
    if situ:
        up = LINEAR * torch.tanh(up / LINEAR) if LINEAR > 0 else up
        return BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate) * up
    return gate * torch.sigmoid(gate) * up


def reference(x, ids, rw, situ):
    """Grouped by expert: each expert decoded once per call, not kept."""
    out = torch.zeros(x.shape[0], H)
    xs = x.float()
    for e in torch.unique(ids).tolist():
        t, j = (ids == e).nonzero(as_tuple=True)
        wg, wu, wd = dequant(w13[e, :I], s13[e, :I]), dequant(w13[e, I:], s13[e, I:]), dequant(w2[e], s2[e])
        xe = xs[t]
        y = act(xe @ wg.T, xe @ wu.T, situ) @ wd.T
        out.index_add_(0, t, y * rw[t, j][:, None])
    return out


g = torch.Generator().manual_seed(20261105)
bad = 0
for situ, label in ((True, f'SiTU beta {BETA} linear {LINEAR}'), (False, 'SiLU')):
    plan = CpuPlan(E, A.threads, [t.data_ptr() for t in (w13, w2, s13, s2, g13, g2)], hidden=H, inter=I,
                   swiglu_limit=LIMIT, scale_format=FMT, situ_beta=BETA if situ else 0.0,
                   situ_linear=LINEAR if situ else 0.0)
    for m in map(int, A.lengths.split(',')):
        x = (torch.randn(m, H, generator=g) * 0.5).bfloat16()
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).int()
        rw = torch.softmax(torch.randn(m, K, generator=g), -1)
        got = torch.empty(m, H)
        plan.forward(m, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), got.data_ptr())
        rows = list(range(m)) if m <= 64 else sorted(torch.randperm(m, generator=g)[:48].tolist())
        ref = reference(x[rows], ids[rows], rw[rows], situ)
        err = float((got[rows] - ref).norm() / ref.norm())
        ok = err <= A.tolerance and bool(torch.isfinite(got).all())
        bad += not ok
        print(f'{label}, m={m}, k={K}, E={E}: rel L2 {err:.2e} over {len(rows)} tokens{"" if ok else " FAIL"}',
              flush=True)
    plan.close()
print('SITU-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
sys.exit(0 if not bad else 1)
