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
def _pack(X, IDS, ROUTE, SX, SIDS, SROUTE, NX: tl.constexpr, NK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(SX + i, tl.load(X + i, i < NX), i < NX)
    if tl.program_id(0) == 0:
        j = tl.arange(0, 128)  # m*k <= 8*10 routes
        tl.store(SIDS + j, tl.load(IDS + j, j < NK), j < NK)
        tl.store(SROUTE + j, tl.load(ROUTE + j, j < NK), j < NK)


@triton.jit
def _unpack(SOUT, OUT, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    # .cv: always re-read host memory written by the CPU after the mailbox handshake.
    tl.store(OUT + i, tl.load(SOUT + i, i < N, cache_modifier='.cv'), i < N)


def pack(slot, m, k, x, ids, routing, H):
    """x: BF16 [m, H], ids: int32 [m, k], routing: float32 [m, k] (device addresses) -> the slot's inputs."""
    sx, sids, sroute = slot[:3]
    _pack[(triton.cdiv(m * H, BLOCK),)](Pointer(x, torch.int16), Pointer(ids, torch.int32), Pointer(routing, torch.int32),
                                        Pointer(sx, torch.int16), Pointer(sids, torch.int32), Pointer(sroute, torch.int32),
                                        m * H, m * k, BLOCK)


def unpack(slot, m, out, H):
    """The slot's float32 output [m, H] -> out (device address)."""
    _unpack[(triton.cdiv(m * H, BLOCK),)](Pointer(slot[3], torch.int32), Pointer(out, torch.int32), m * H, BLOCK)
