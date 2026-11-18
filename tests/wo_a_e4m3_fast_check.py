"""_e4m3_fast against xtier.arch.e4m3_f32 over all 256 E4M3 codes, bit for bit.

    <SGLang venv>/bin/python tests/wo_a_e4m3_fast_check.py

The two decode the same values except for the sign bit of the NaN that 0xFF encodes: arch.e4m3_f32 negates
the NaN it builds, _e4m3_fast returns an unsigned one. Every numeric code is bit-identical. On a device with
a hardware conversion (HW_FP8) the reference side is the hardware path, so the same one code is expected.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from xtier.arch import HW_FP8, e4m3_f32  # noqa: E402
from xtier.packs.deepseek_v4.wo_a import _e4m3_fast  # noqa: E402

NAN_SIGN_ONLY = 0xFF


@triton.jit
def _decode(src, fast_ptr, ref_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(src + i)
    tl.store(fast_ptr + i, _e4m3_fast(v))
    tl.store(ref_ptr + i, e4m3_f32(v))


def bits(x):
    return int(x) & 0xFFFFFFFF


def is_nan(u):
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x007FFFFF) != 0


def main():
    if not torch.cuda.is_available():
        raise SystemExit('needs a GPU')
    codes = torch.arange(256, dtype=torch.uint8, device='cuda')
    fast = torch.empty(256, dtype=torch.float32, device='cuda')
    ref = torch.empty(256, dtype=torch.float32, device='cuda')
    _decode[(1,)](codes, fast, ref, BLOCK=256)
    torch.cuda.synchronize()
    fi, ri = fast.view(torch.int32).cpu(), ref.view(torch.int32).cpu()
    print(f'HW_FP8={bool(HW_FP8)} device={torch.cuda.get_device_name(0)}')
    bad = []
    for c in range(256):
        f, r = bits(fi[c]), bits(ri[c])
        if f == r:
            continue
        if is_nan(f) and is_nan(r):
            print(f'  {hex(c)}: NaN on both sides, sign differs only ({f:#010x} vs {r:#010x})')
            if c == NAN_SIGN_ONLY:
                continue
        bad.append(hex(c))
    print('codes not bit-identical: ' + (', '.join(bad) if bad else 'none'))
    if bad:
        raise SystemExit('FAIL')
    print('PASS')


if __name__ == '__main__':
    main()
