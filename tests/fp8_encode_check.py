"""FP8 (E4M3) encoding in software (xtier.arch.e4m3_bits, for kernels that store FP8 on RTX 30 series) against the
conversion Triton emits on GPUs with float8e4nv (SM89 and newer): every FP32 value whose top 20 bits differ (the FP32
rounding of every halfway point and of its neighbours), random values over the range, and the edge cases (+-0, the
subnormal steps and their halves, 448, 464, 480, infinities, NaN), bit for bit.

  python tests/fp8_encode_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from xtier import arch  # noqa: E402


@triton.jit
def encode(X, SOFT, HARD, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, mask=i < N)
    tl.store(SOFT + i, arch.e4m3_bits(x), mask=i < N)
    tl.store(HARD + i, x.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=i < N)


if __name__ == '__main__':
    dev = torch.device('cuda')
    # the top 20 bits (sign, exponent, 11 mantissa bits) of every FP32 value, with low bits 0, ...7FF, 800, 801, FFF:
    # each E4M3 halfway point and the values on both sides of it
    top = torch.arange(1 << 20, dtype=torch.int64) << 12
    grid = torch.cat([top, top | 0x7FF, top | 0x800, top | 0x801, top | 0xFFF]).to(torch.int32).view(torch.float32)
    g = torch.Generator().manual_seed(20261029)
    rand = (torch.randn(1 << 22, generator=g) * torch.logspace(-6, 3, 1 << 22)).to(torch.float32)
    edges = torch.tensor([0.0, -0.0, 2.0 ** -9, 2.0 ** -10, 3 * 2.0 ** -10, 2.0 ** -6, 2.0 ** -6 - 2.0 ** -10, 448.0,
                          449.0, 463.9, 464.0, 464.1, 479.0, 480.0, 1e30, float('inf'), float('-inf'), float('nan')])
    x = torch.cat([grid, rand, edges, -edges]).to(dev)
    n = x.numel()
    soft = torch.empty(n, dtype=torch.uint8, device=dev)
    hard = torch.empty(n, dtype=torch.uint8, device=dev)
    encode[(triton.cdiv(n, 1024),)](x, soft, hard, n, 1024)
    torch.cuda.synchronize()
    nan = torch.isnan(x)
    bad = (soft != hard) & ~nan
    nan_ok = bool(((soft[nan] & 0x7F) == 0x7F).all() and ((hard[nan] & 0x7F) == 0x7F).all())
    print(f'{n} FP32 values: {int(bad.sum())} encodings differ from float8e4nv; NaN -> NaN both: {nan_ok}')
    if bad.any():
        i = bad.nonzero()[:8, 0]
        for v, a, b in zip(x[i].tolist(), soft[i].tolist(), hard[i].tolist()):
            print(f'  {v!r}: software 0x{a:02x}, hardware 0x{b:02x}')
    ok = not bad.any() and nan_ok
    print('FP8-ENCODE-CHECK', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)
