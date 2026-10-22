"""Overlapped hyper-connection mix (xtier/packs/qwen4_exp/hc_mix.py) against SGLang's fused_hc_mix at Qwen3.8 shapes:
repeated calls (the accumulator pair alternates), every row count up to 16, a CUDA graph replayed several times, and
the time per call. The atomics' summation order varies, so the comparison is a relative L2 error.

  python tests/hc_mix_check.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from sglang.kernels.ops.gemm.hc_mix import fused_hc_mix as sglang_hc_mix  # noqa: E402

from xtier.packs.qwen4_exp.hc_mix import fused_hc_mix  # noqa: E402

HC, HS, LOWRANK = 4, 2560, 320
torch.manual_seed(20261023)
w_down = (torch.randn(LOWRANK, HC * HS, device='cuda') * 0.02).bfloat16()
w_up = (torch.randn(HC * HS, LOWRANK, device='cuda') * 0.05).bfloat16()


def ours(x):
    return fused_hc_mix(sglang_hc_mix, x, w_down, w_up, HC, HS)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


bad = 0
for rows in (1, 2, 3, 4, 5, 8, 12, 16):
    errs = []
    for _ in range(3):   # consecutive calls alternate the accumulator halves
        x = torch.randn(rows, HC * HS, device='cuda').bfloat16()
        errs.append(rel(ours(x), sglang_hc_mix(x, w_down, w_up, HC, HS)))
    x = torch.randn(rows, HC * HS, device='cuda').bfloat16()
    out = ours(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ours(x)
    for _ in range(4):
        x.copy_(torch.randn(rows, HC * HS, device='cuda').bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        errs.append(rel(out, sglang_hc_mix(x, w_down, w_up, HC, HS)))
    times = []
    for fn in (lambda: sglang_hc_mix(x, w_down, w_up, HC, HS), lambda: ours(x)):
        fn()
        g = torch.cuda.CUDAGraph()   # kernel time without launch overhead, as in the decode graphs
        with torch.cuda.graph(g):
            for _ in range(100):
                fn()
        g.replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) / 500 * 1e6)
    ok = max(errs) < 1e-2
    bad += not ok
    print(f'rows {rows:2d}: max rel L2 {max(errs):.2e}{"" if ok else " FAIL"}, SGLang {times[0]:.1f} us, '
          f'overlap {times[1]:.1f} us per call (CUDA graph)', flush=True)
print('HC-MIX-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
