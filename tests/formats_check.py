"""Checkpoint format adapters (xtier/backend/formats.py) on one real layer: parameters laid out as SGLang's MoE method
creates and loads them (NVFP4: uint8 weights, E4M3 block scales, FP32 scale per projection; DeepSeek MXFP4: int8
weights, block scales as FP32 powers of two) must give back the CPU engine's inputs byte for byte, in [gate; up] order
and in the [up; gate] order some GPU runners load.

  python tests/formats_check.py --preset ds41_1017
"""
import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layers import PRESETS, load  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--preset', choices=sorted(PRESETS), required=True)
A = P.parse_args()
FMT = PRESETS[A.preset][10]

import torch  # noqa: E402

from xtier.backend.formats import Fp8Fp4Experts, ModelOptNvFp4  # noqa: E402

w13, w2, s13, s2, g13, g2 = load(A.preset)


def swap(t):
    half = t.shape[1] // 2
    return torch.cat([t[:, half:], t[:, :half]], dim=1)


def param(t):
    return types.SimpleNamespace(data=t)


bad = 0
for up_first in (False, True):
    order = swap if up_first else (lambda t: t)
    if FMT == 'nvfp4':
        layer = types.SimpleNamespace(w13_weight=param(order(w13)), w2_weight=param(w2),
                                      w13_weight_scale=param(order(s13).view(torch.float8_e4m3fn)),
                                      w2_weight_scale=param(s2.view(torch.float8_e4m3fn)),
                                      w13_weight_scale_2=param(g13), w2_weight_scale_2=param(g2))
        got = ModelOptNvFp4().engine_inputs(layer, up_first)
    else:
        def as_float(s):   # what SGLang's loader stores for F8_E8M0 checkpoint scales
            return s.view(torch.float8_e8m0fnu).to(torch.float32)
        layer = types.SimpleNamespace(w13_weight=param(order(w13).view(torch.int8)), w2_weight=param(w2.view(torch.int8)),
                                      w13_weight_scale_inv=param(as_float(order(s13))),
                                      w2_weight_scale_inv=param(as_float(s2)))
        got = Fp8Fp4Experts().engine_inputs(layer, up_first)
    for name, a, b in zip(('w13', 'w2', 's13', 's2', 'g13', 'g2'), got, (w13, w2, s13, s2, g13, g2)):
        same = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        bad += not same
        if not same:
            print(f'up_first={up_first}: {name} differs ({a.dtype} {tuple(a.shape)} vs {b.dtype} {tuple(b.shape)})')
    print(f'{A.preset} ({FMT}), loaded {"[up; gate]" if up_first else "[gate; up]"}: '
          f'{"engine inputs byte for byte" if not bad else "DIFFERENT"}', flush=True)
print('FORMATS-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
