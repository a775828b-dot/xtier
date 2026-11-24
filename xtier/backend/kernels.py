"""GPU side of the mailbox: one kernel stores a call's inputs straight into the mapped pinned slot, one kernel
reads the CPU output back after the handshake. Bytes move unchanged (bit copies, no conversion).
Ported from lkqmoe 0.5.6 ``zerocopy.py``.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


class Pointer:
    """A raw device address as a Triton pointer argument."""

    def __init__(self, address, dtype):
        self.address = address
        self.dtype = dtype

    def data_ptr(self):
        return self.address


@triton.jit
def _pack(X, IDS, ROUTE, ACT, SX, SIDS, SROUTE, NX: tl.constexpr, NK: tl.constexpr, K: tl.constexpr,
          HAS_ACT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(SX + i, tl.load(X + i, i < NX), i < NX)
    if tl.program_id(0) == 0:
        j = tl.arange(0, 256)  # m*k <= 144 routes (method.py SLOT_ROUTES)
        ids = tl.load(IDS + j, j < NK)
        if HAS_ACT:  # skipped tokens (method.active_flags): -1 on every route, the engine returns a zero row
            ids = tl.where(tl.load(ACT + j // K, j < NK, other=1) != 0, ids, -1)
        tl.store(SIDS + j, ids, j < NK)
        tl.store(SROUTE + j, tl.load(ROUTE + j, j < NK), j < NK)


@triton.jit
def _unpack(SOUT, OUT, SCALE, N, B: tl.constexpr, OUT_BF16: tl.constexpr = False, HAS_SCALE: tl.constexpr = False):
    i = tl.program_id(0) * B + tl.arange(0, B)
    # .cv: always re-read host memory written by the CPU after the mailbox handshake.
    c = tl.load(SOUT + i, i < N, other=0.0, cache_modifier='.cv')
    if OUT_BF16:
        b = c.to(tl.bfloat16)
        if HAS_SCALE:
            b = (b.to(tl.float32) * SCALE).to(tl.bfloat16)
        tl.store(OUT + i, b, i < N)
    else:
        tl.store(OUT + i, c, i < N)


def pack(slot, m, k, x, ids, routing, H, act=0):
    """x: BF16 [m, H], ids: int32 [m, k], routing: float32 [m, k], act: int32 [m] token flags or 0 (device addresses)
    -> the slot's inputs."""
    sx, sids, sroute = slot[:3]
    _pack[(triton.cdiv(m * H, BLOCK),)](Pointer(x, torch.int16), Pointer(ids, torch.int32), Pointer(routing, torch.int32),
                                        Pointer(act or ids, torch.int32), Pointer(sx, torch.int16),
                                        Pointer(sids, torch.int32), Pointer(sroute, torch.int32), m * H, m * k, k,
                                        bool(act), BLOCK)


def unpack(slot, m, out, H, out_bf16=False, scale=1.0):
    """The slot's float32 output [m, H] -> out (device address; FP32, or scaled BF16 when out_bf16)."""
    n = m * H
    if out_bf16:
        _unpack[(triton.cdiv(n, BLOCK),)](Pointer(slot[3], torch.float32), Pointer(out, torch.bfloat16), float(scale),
                                          n, BLOCK, OUT_BF16=True, HAS_SCALE=(scale != 1.0))
    else:
        _unpack[(triton.cdiv(n, BLOCK),)](Pointer(slot[3], torch.int32), Pointer(out, torch.int32), 1.0,
                                          n, BLOCK, OUT_BF16=False, HAS_SCALE=False)
