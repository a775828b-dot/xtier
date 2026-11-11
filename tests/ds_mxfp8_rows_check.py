"""FlashInfer's b12x small-M MXFP8 GEMM against its CUTLASS kernel at the row counts the DeepSeek pack sends to b12x
(packs/deepseek_v4: _b12x_rows, _mxfp8_linear; XTIER_MXFP8_B12X_ROWS, default 1-24: every width a decode graph of up
to four requests can have): the same FP8 weights, scales and activations at DeepSeek V4.1 Flash's projection shapes, so
the two can differ only in the order of the FP32 sums (on the RTX PRO 5000 the outputs are bitwise equal, at 25 to 32
rows too). Also checks how the setting is read (counts and ranges), that the hook picks b12x for exactly these row
counts, and prints the GPU time of a call of both, replayed from a CUDA graph as a server does (activation quantize +
GEMM). Here the weights stay in the GPU's cache; in a server, where every layer reads its own, the 1280 -> 32768
projection at 5 rows takes 159 us on CUTLASS and 39 us on b12x. SM120 only: elsewhere the pack leaves every call on
SGLang's backend, and the test is skipped.

  python tests/ds_mxfp8_rows_check.py [--rows 5,6,12]
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    print('SKIP: needs an SM120 GPU (the pack sends MXFP8 calls to b12x there only)')
    sys.exit(0)

from flashinfer import block_scale_interleave  # noqa: E402
from sglang.srt.layers.quantization.fp8_utils import (  # noqa: E402
    flashinfer_mxfp8_blockscaled_linear as linear,
    flashinfer_mxfp8_quantize as quantize,
)

from xtier.packs import deepseek_v4 as pack  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--rows', default=','.join(str(r) for r in sorted(pack._b12x_rows())))
A = P.parse_args()
rows_list = [int(r) for r in A.rows.split(',') if r.strip()]
torch.manual_seed(20261111)
ok = True

# the setting: counts and ranges, comma separated; empty: no row count
saved = os.environ.get('XTIER_MXFP8_B12X_ROWS')
for text, want in (('1,4-6, 12', {1, 4, 5, 6, 12}), ('1-24', set(range(1, 25))), ('6', {6}), ('', set())):
    os.environ['XTIER_MXFP8_B12X_ROWS'] = text
    got = pack._b12x_rows.__wrapped__()
    if got != want:
        ok = False
        print(f'FAIL: XTIER_MXFP8_B12X_ROWS={text!r} read as {sorted(got)}')
if saved is None:
    del os.environ['XTIER_MXFP8_B12X_ROWS']
else:
    os.environ['XTIER_MXFP8_B12X_ROWS'] = saved

# the hook: b12x for the listed row counts, SGLang's backend for the others and for a caller that named another one
seen = []


def fake(input, *args, **kwargs):
    seen.append(kwargs.get('backend'))


for rows in sorted(set(rows_list) | {2, 3, 7, 8, 16, 25, 32, 4096}):
    pack._mxfp8_linear(fake, torch.empty(rows, 64), backend='cutlass')
    want = 'b12x' if rows in pack._b12x_rows() else 'cutlass'
    if seen[-1] != want:
        ok = False
        print(f'FAIL: {rows} rows went to {seen[-1]}, expected {want}')
pack._mxfp8_linear(fake, torch.empty(6, 64), backend='cute-dsl')
if seen[-1] != 'cute-dsl':
    ok = False
    print(f'FAIL: a cute-dsl call was rerouted to {seen[-1]}')
print(f'hook: b12x for rows {sorted(pack._b12x_rows())}, other row counts and backends untouched')

# out features, in features: wq_a, wq_b, wo_b, the shared expert's gate/up and down
SHAPES = [(1280, 5120), (32768, 1280), (5120, 8192), (4608, 5120), (5120, 2304)]
for n, k in SHAPES:
    w = (torch.randn(n, k, device='cuda') * 0.02).bfloat16()
    qw, sf = quantize(w, is_sf_swizzled_layout=False, alignment=32)
    scale = block_scale_interleave(sf.view(n, k // 32).contiguous()).contiguous()
    for rows in rows_list:
        x = torch.randn(rows, k, device='cuda').bfloat16()
        ref = linear(x, qw, scale, backend='cutlass').float()
        got = linear(x, qw, scale, backend='b12x').float()
        err = ((got - ref).norm() / ref.norm()).item()
        # against the BF16 weights the quantized product is some percent off (FP8 E4M3 on both sides); the two kernels
        # see the same quantized operands and may differ by a BF16 rounding step where the FP32 sums differ
        plain = (x.float() @ w.float().T)
        qerr = ((ref - plain).norm() / plain.norm()).item()
        times, replayed = {}, True
        for name in ('cutlass', 'b12x'):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = linear(x, qw, scale, backend=name)
            graph.replay()
            torch.cuda.synchronize()
            replayed &= torch.equal(out.float(), ref if name == 'cutlass' else got)
            t0 = time.perf_counter()
            for _ in range(200):
                graph.replay()
            torch.cuda.synchronize()
            times[name] = 1e6 * (time.perf_counter() - t0) / 200
            del graph
        good = err < 5e-3 and qerr < 0.1 and got.shape == (rows, n) and replayed
        ok &= good
        print(f'{k:5d} -> {n:5d}, {rows:2d} rows: b12x against CUTLASS relative L2 {err:.1e}'
              f'{" (bitwise equal)" if torch.equal(got, ref) else ""}, quantization {qerr:.1e}; '
              f'graph replay {times["cutlass"]:5.0f} us on CUTLASS, {times["b12x"]:5.0f} us on b12x'
              + ('' if good else '  FAIL'))
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
