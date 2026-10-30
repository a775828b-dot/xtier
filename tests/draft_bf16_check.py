"""BF16 draft experts as NVFP4 (xtier/backend/formats.py Bf16AsNvFp4, nvfp4_encode): random matrices dequantize within
their block scale of the input; one real GLM-5.3 NVFP4 layer (tests/layers.py), dequantized to BF16 the way the
engine decodes it (E2M1 of the low nibble first, E4M3 block scale per 16, FP32 global per expert and projection) and
quantized again, comes back as it was, up to the blocks without an E2M1 6 (their scale comes out smaller).

  python tests/draft_bf16_check.py [--preset glm53] [--experts 16]
"""
import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import load  # noqa: E402

import torch  # noqa: E402

from xtier.backend.formats import Bf16AsNvFp4, nvfp4_encode  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', default='glm53')
P.add_argument('--experts', type=int, default=16, help='experts of the real layer (memory: ~0.3 GB each)')
A = P.parse_args()
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def decode(q, s, g):
    """[..., K] FP32 of E2M1 pairs q [..., K/2] (low nibble first), E4M3 scale bytes s [..., K/16], global g."""
    codes = torch.stack([q & 15, q >> 4], -1).flatten(-2).long()
    scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return E2M1[codes] * scale * g


# 1. random matrices: every value within one block scale of its input (the widest E2M1 gap, 4 to 6, is 2 scales)
torch.manual_seed(0)
for rows, k in ((64, 4096), (512, 2048)):
    w = (torch.randn(rows, k) * torch.rand(rows, 1) * 0.05).to(torch.bfloat16)
    q, s, g = nvfp4_encode(w)
    back = decode(q, s, g)
    bound = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1) * g
    worst = ((back - w.float()).abs() / bound.clamp_min(1e-30)).max().item()
    rel = ((back - w.float()).norm() / w.float().norm()).item()
    assert worst <= 1.0 + 1e-6, worst
    print(f'random {rows} x {k}: worst error {worst:.3f} block scales, relative RMS {rel:.4f}')

# 2. one real layer: its NVFP4 dequantized to BF16 and quantized again through the format adapter
w13, w2, s13, s2, g13, g2 = (t[:A.experts] for t in load(A.preset))
E, I2, _ = w13.shape
ref13 = torch.stack([torch.cat([decode(w13[e, j * I2 // 2:(j + 1) * I2 // 2], s13[e, j * I2 // 2:(j + 1) * I2 // 2],
                                       g13[e, j]) for j in range(2)]) for e in range(E)])
ref2 = torch.stack([decode(w2[e], s2[e], g2[e]) for e in range(E)])
layer = types.SimpleNamespace(w13_weight=types.SimpleNamespace(data=ref13.to(torch.bfloat16)),
                              w2_weight=types.SimpleNamespace(data=ref2.to(torch.bfloat16)))
q13, q2, t13, t2, h13, h2 = Bf16AsNvFp4().engine_inputs(layer, up_first=False)
assert q13.shape == w13.shape and q2.shape == w2.shape and t13.shape == s13.shape and t2.shape == s2.shape
back13 = torch.stack([torch.cat([decode(q13[e, j * I2 // 2:(j + 1) * I2 // 2], t13[e, j * I2 // 2:(j + 1) * I2 // 2],
                                        h13[e, j]) for j in range(2)]) for e in range(E)])
back2 = torch.stack([decode(q2[e], t2[e], h2[e]) for e in range(E)])
for name, ref, back, qa, qb in (('w13', ref13, back13, w13, q13), ('w2', ref2, back2, w2, q2)):
    rel = ((back - ref).norm() / ref.norm()).item()
    same = (qa == qb).float().mean().item()
    print(f'{A.preset} {name}: relative RMS {rel:.5f} against the checkpoint\'s values, {same:.1%} of the bytes the same')
    assert rel < 0.02, rel
print('draft bf16 ok')
