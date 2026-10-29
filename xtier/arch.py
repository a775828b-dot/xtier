"""What this GPU gives XTier's own Triton kernels, and software decoding where it gives less.

- FP4 (E2M1) to FP16: the cvt.rn.f16x2.e2m1x2 instruction needs SM100 or newer (SM120 included); RTX 30 / 40 series
  (SM86 / SM89) build the FP16 bits with integer operations (XTIER_FP4_DECODE=hw|soft|auto).
- FP8 (E4M3) to FP32: Triton has float8e4nv from SM89 on; RTX 30 series (SM86) build the FP32 bits with integer
  operations (XTIER_FP8_DECODE=hw|soft|auto). The other way, ``e4m3_bits`` encodes FP32 in software as the conversion
  Triton emits from SM89 on (round to nearest even, saturating), for kernels that store FP8 on those GPUs.
The software paths give exactly the values of the conversions (tests/fp4_decode_check.py, tests/fp8_decode_check.py),
so a kernel computes the same results with either. The flags are fixed when this module is first imported, in the
process that runs the kernels.
"""
import os

import torch
import triton
import triton.language as tl


def _hardware(variable, capability):
    mode = os.environ.get(variable, 'auto').strip().lower()
    if mode in ('hw', 'soft'):
        return mode == 'hw'
    return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) >= capability


HW_FP4 = tl.constexpr(_hardware('XTIER_FP4_DECODE', (10, 0)))
HW_FP8 = tl.constexpr(_hardware('XTIER_FP8_DECODE', (8, 9)))


@triton.jit
def e2m1_f16(c):
    """E2M1 code (low 4 bits; sign in bit 3) -> FP16, the bits cvt.rn.f16x2.e2m1x2 gives: magnitudes 0.5..6 have
    exponent field (m >> 1) + 14 and mantissa bit (m & 1), i.e. (m << 9) + 0x3800; 0 and 0.5 (subnormal in E2M1) are
    m * 0x3800; the sign moves from bit 3 to bit 15 (code 8 is -0.0)."""
    m = c & 7
    bits = tl.where(m >= 2, (m << 9) + 0x3800, m * 0x3800) | ((c & 8) << 12)
    return bits.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def e4m3_f32(b):
    """E4M3 byte (uint8 storage of float8_e4m3fn) -> FP32. In software: exponent field e, mantissa m; e > 0 is
    2^(e-7) * (1 + m/8), the FP32 bits ((e + 120) << 23) | (m << 20); e = 0 is m * 2^-9 (subnormal); 0x7F / 0xFF are
    NaN; bit 7 is the sign (0x80 is -0.0)."""
    if HW_FP8:
        return b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        v = b.to(tl.int32)
        e = (v >> 3) & 15
        m = v & 7
        mag = tl.where(e == 0, m.to(tl.float32) * 0.001953125,
                       (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True))
        mag = tl.where((v & 0x7F) == 0x7F, float('nan'), mag)
        return tl.where((v & 0x80) != 0, -mag, mag)


@triton.jit
def e4m3_bits(x):
    """FP32 -> E4M3 (fn) byte as cvt.rn.satfinite.e4m3x2.f32 gives it: round to nearest even, saturating to +-448
    (infinities too), NaN to 0x7F with its sign. From 2^-6 up the FP32 bits are rounded at mantissa bit 20 (a carry
    moves into the exponent); below, the value is |x| * 2^9 subnormal steps rounded to the nearest even (8 steps are
    the smallest normal, code 0x08)."""
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = bits & 0x7FFFFFFF
    r = a + 0x7FFFF + ((a >> 20) & 1)
    normal = (((r >> 23) - 120) << 3) | ((r >> 20) & 7)
    t = a.to(tl.float32, bitcast=True) * 512.0
    lo = tl.floor(t)
    steps = lo.to(tl.int32)
    half = t - lo
    steps += ((half > 0.5) | ((half == 0.5) & ((steps & 1) == 1))).to(tl.int32)
    code = tl.where(a < 0x3C800000, steps, tl.minimum(normal, 0x7E))
    code = tl.where(a > 0x7F800000, 0x7F, code)
    return (code | sign).to(tl.uint8)
