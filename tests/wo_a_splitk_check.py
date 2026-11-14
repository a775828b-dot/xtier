"""split-K GEMV against a float32 reference, on shapes that cannot come from a checkpoint.

    <SGLang venv>/bin/python tests/wo_a_splitk_check.py

The kernel reduces one activation row, so only T == 1 may reach it; einsum() has to send T > 1 down
the tl.dot path. Both are asserted here, as is the rule that the FP32 scratch buffer is keyed by
output shape rather than by number of elements -- two shapes sharing a numel used to share a buffer
and therefore each other's strides.

Needs a GPU but no weights: the operands are random, with the same per-128x128 block scales the
checkpoint uses.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402

os.environ['XTIER_DS_WO_A_FASTDEC'] = '0'  # keep the e4m3 decoder out of this comparison

from xtier.packs.deepseek_v4 import wo_a  # noqa: E402

DEVICE = 'cuda'
TOL = 2e-2  # bf16 output, fp32 accumulation: 2.3e-3 observed
CASES = [
    (1, 4, 64, 1024),
    (1, 4, 128, 2048),
    (1, 4, 64, 3072),
    (1, 8, 64, 4096),
    (1, 2, 64, 8192),
]


def make(t, g, r, d, fp8, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    o = torch.randn(t, g, d, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    if fp8:
        w = (torch.randn(g, r, d, device=DEVICE, generator=gen) * 0.5).to(torch.float8_e4m3fn)
        scale = (torch.rand(g * r // 128, d // 128, device=DEVICE, generator=gen) * 0.02 + 0.01).float()
    else:
        w = (torch.randn(g, r, d, device=DEVICE, generator=gen) * 0.02).to(torch.bfloat16)
        scale = None
    return o, w, scale


def reference(o, w, scale):
    wf = w.float()
    if scale is not None:
        g, r, d = w.shape
        wf = wf * scale.repeat_interleave(g * r // scale.shape[0], 0).repeat_interleave(
            d // scale.shape[1], 1).view(g, r, d)
    return torch.einsum('tgd,grd->tgr', o.float(), wf)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


failures = []


def check(name, got, want):
    err = rel(got, want)
    ok = err < TOL
    print('  %-34s rel=%-10.4g %s' % (name, err, 'ok' if ok else 'FAIL'))
    if not ok:
        failures.append(name)


print('split-K GEMV vs float32 reference (T == 1):')
for fp8 in (True, False):
    for t, g, r, d in CASES:
        o, w, scale = make(t, g, r, d, fp8, seed=d)
        wo_a._scratch_f32.clear()
        split = max(1, min(8, -(-4 * wo_a._sm_count() // max(triton.cdiv(r, wo_a.GB_R) * g, 1))))
        check('%s fp8=%d split=%d' % ('G%d R%d D%d' % (g, r, d), fp8, split),
              wo_a._gemv(o, w, scale), reference(o, w, scale))

print()
print('T > 1 must stay on the tl.dot path (the kernel has no T dimension):')
for t in (2, 3):
    o, w, scale = make(t, 4, 64, 1024, True, seed=t)
    for flag in ('1', '0'):
        os.environ['XTIER_DS_WO_A_SPLIT_K'] = flag
        check('T=%d SPLIT_K=%s' % (t, flag), wo_a.einsum(o, w, scale), reference(o, w, scale))
os.environ['XTIER_DS_WO_A_SPLIT_K'] = '1'
try:
    wo_a._gemv(*make(2, 4, 64, 1024, True, seed=2))
    print('  _gemv(T=2) raised nothing                             FAIL')
    failures.append('_gemv T=2 guard')
except NotImplementedError:
    print('  _gemv(T=2) raises NotImplementedError                 ok')

print()
print('scratch buffer is keyed by shape, not by numel:')
wo_a._scratch_f32.clear()
o1, w1, s1 = make(1, 8, 64, 1024, True, seed=11)   # 512 elements, (1, 8, 64)
o2, w2, s2 = make(1, 4, 128, 2048, True, seed=12)  # 512 elements, (1, 4, 128)
check('(1,8,64) first', wo_a._gemv(o1, w1, s1), reference(o1, w1, s1))
check('(1,4,128) after (1,8,64)', wo_a._gemv(o2, w2, s2), reference(o2, w2, s2))
buf = wo_a._scratch_f32[(o2.device, 1, 4, 128)]
print('  buffer shape %s stride %s %s'
      % (tuple(buf.shape), buf.stride(),
         'ok' if tuple(buf.shape) == (1, 4, 128) else 'FAIL'))
if tuple(buf.shape) != (1, 4, 128):
    failures.append('scratch shape')

print()
print('PASS' if not failures else 'FAILED: %s' % ', '.join(failures))
sys.exit(1 if failures else 0)
