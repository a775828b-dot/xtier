"""FP8 weight storage (xtier/packs/glm5_next/fp8_weights.py) on GLM-5.3's BF16 linear shapes with random weights:
the W8A16 decode kernel (M <= 16) against an FP64 product of the dequantized weight, bitwise repeatable and the same
inside a CUDA graph; the W8A8 prefill and the dequant fallback against the same reference.

  python tests/fp8_weights_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from xtier.packs.glm5_next import fp8_weights as f  # noqa: E402

SHAPES = [(4096, 8192), (16384, 1536), (4096, 16384), (4096, 1536), (4096, 12288), (4096, 4096), (4096, 2048),
          (24576, 4096)]
g = torch.Generator(device='cuda').manual_seed(20261023)
bad = 0


def rel(a, b):
    return float((a.double() - b).norm() / b.norm())


for n, k in SHAPES:
    w = (torch.randn(n, k, generator=g, device='cuda') * 0.02).bfloat16()
    q, s = f.quantize(w)
    ref_w = f.dequant(q, s).double()
    cnt = torch.zeros((n + 31) // 32, dtype=torch.int32, device='cuda')
    worst = 0.0
    for m in (1, 4, 5, 16):
        x = torch.randn(m, k, generator=g, device='cuda').bfloat16()
        y = f.gemm_w8(x, q, s, None, cnt)
        again = f.gemm_w8(x, q, s, None, cnt)
        worst = max(worst, rel(y, x.double() @ ref_w.T))
        if not torch.equal(y, again):
            bad += 1
            print(f'  N={n} K={k} M={m}: two runs differ')
    x = torch.randn(5, k, generator=g, device='cuda').bfloat16()
    static = x.clone()
    out = f.gemm_w8(static, q, s, None, cnt)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = f.gemm_w8(static, q, s, None, cnt)
    replay_ok = True
    for _ in range(3):
        x = torch.randn(5, k, generator=g, device='cuda').bfloat16()
        static.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        replay_ok &= torch.equal(out, f.gemm_w8(x, q, s, None, cnt))
    del graph
    xp = torch.randn(512, k, generator=g, device='cuda').bfloat16()
    ref = xp.double() @ ref_w.T
    e_w8a8 = rel(f.gemm_w8a8(xp, q, s), ref)
    e_deq = rel(torch.nn.functional.linear(xp, f.dequant(q, s)), ref)
    ok = worst < 5e-3 and replay_ok and e_w8a8 < 5e-2 and e_deq < 5e-3
    bad += not ok
    print(f'N={n:5d} K={k:5d}: decode rel {worst:.2e}, graph replays {"bitwise" if replay_ok else "DIFFERENT"}, '
          f'W8A8 prefill rel {e_w8a8:.2e}, dequant prefill rel {e_deq:.2e} -> {"ok" if ok else "FAIL"}', flush=True)
print('FP8-WEIGHTS-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
