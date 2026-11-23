"""Grouped GPU prefill of a MoE layer whose experts live in host memory (the CPU engine's NUMA shards).
Ported from lkqmoe 0.5.6 (commit 93edcc1).

Compressed weights and FP8 block scales are unchanged. Gate/up use exact BF16
operands with FP32 accumulation; activation, down and route reduction stay FP32.
Calls are synchronous, serialized, and release temporary device weights before
returning to attention. Retaining the workspace is an explicit experimental opt-in.
"""
import ctypes as C
import gc
import logging
import math
import os
import threading
import time
import weakref
import torch
import triton as tr
import triton.language as tl

from xtier.arch import e4m3_f32


class Pointer:
    def __init__(self, address, dtype):
        self.address = address
        self.dtype = dtype

    def data_ptr(self):
        return self.address


@tr.jit
def _copy_ids(S, D, N, B: tl.constexpr):   # N at run time: prompts end in any length, each would compile a kernel
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(D + i, tl.load(S + i, i < N, 0), i < N)


@tr.jit
def _weight(W, S, e, row, col, ROWS: tl.constexpr, COLS: tl.constexpr, MX: tl.constexpr = False):
    packed = tl.load(W + e * ROWS * (COLS // 2) + row * (COLS // 2) + col // 2)
    q = (packed >> ((col % 2) * 4)) & 15
    a = q & 7
    v = tl.where(a < 4, a * 0.5, tl.where(a == 4, 2., tl.where(a == 5, 3., tl.where(a == 6, 4., 6.))))
    v = tl.where(q >= 8, -v, v)
    s = tl.load(S + e * ROWS * (COLS // (32 if MX else 16)) + row * (COLS // (32 if MX else 16)) + col // (32 if MX else 16))
    if MX:
        # UE8M0 byte 0 represents 2^-127 (a float32 subnormal).
        scale = tl.where(s == 0, 2.0 ** -127, (s.to(tl.uint32) << 23).to(tl.float32, bitcast=True))
    else:
        scale = e4m3_f32(s)
    return v * scale


@tr.jit
def _unpack_all(W, S, D, ROWS: tl.constexpr, COLS: tl.constexpr, B: tl.constexpr, MX: tl.constexpr = False):
    e = tl.program_id(0).to(tl.int64)  # E*rows*cols exceeds 2**31 for DeepSeek V4.1 layers
    i = tl.program_id(1)*B + tl.arange(0,B)
    value = _weight(W,S,e,i//COLS,i%COLS,ROWS,COLS,MX)
    tl.store(D+e*ROWS*COLS+i,value)


@tr.jit
def _gate(X, W, S, G, ORDER, TILE_E, TILE_START, TILE_END, MID, E0, R0,
          K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, UNPACKED: tl.constexpr,
          HID: tl.constexpr, INT: tl.constexpr, LIMIT: tl.constexpr, MX: tl.constexpr = False,
          ALPHA: tl.constexpr = 1.0, BETA: tl.constexpr = 0.0):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)  # expert offsets exceed 2**31 for DeepSeek V4.1
    ew = e - E0  # index inside the staged expert chunk; global scales use e
    r = tl.load(TILE_START + tile) + tl.arange(0, BM)
    end = tl.load(TILE_END + tile)
    route = tl.load(ORDER + r, r < end, 0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for base in range(0, HID, BK):
        c = base + kk
        x = tl.load(X + (route // K)[:, None] * HID + c[None, :], (r < end)[:, None], 0)
        if UNPACKED:
            w0 = tl.load(W + ew*(2*INT)*HID + n[None,:]*HID + c[:,None])
            w1 = tl.load(W + ew*(2*INT)*HID + (n+INT)[None,:]*HID + c[:,None])
        else:
            w0 = _weight(W, S, ew, n[None, :], c[:, None], 2*INT, HID, MX).to(tl.bfloat16)
            w1 = _weight(W, S, ew, (n + INT)[None, :], c[:, None], 2*INT, HID, MX).to(tl.bfloat16)
        gate = tl.dot(x, w0, gate)
        up = tl.dot(x, w1, up)
    gate *= tl.load(G + e * 2)
    up *= tl.load(G + e * 2 + 1)
    if LIMIT > 0:
        # Clamped SwiGLU (DeepSeek V4): up to [-L,L], gate to (-inf,L].
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        gate = tl.minimum(gate, LIMIT)
    if ALPHA != 1.0 or BETA != 0.0:   # GPT-OSS SwiGLU (MiniMax M3)
        value = gate / (1. + tl.exp(-ALPHA * gate)) * (up + BETA)
    else:
        value = gate / (1. + tl.exp(-gate)) * up
    tl.store(MID + (r - R0)[:, None] * INT + n[None, :], value, (r < end)[:, None])


@tr.jit
def _down(MID, W, S, G, ORDER, TILE_E, TILE_START, TILE_END, ROUTE_OUT, E0, R0,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr, UNPACKED: tl.constexpr,
          HID: tl.constexpr, INT: tl.constexpr, BY_ROW: tl.constexpr = False, MX: tl.constexpr = False):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)  # expert offsets exceed 2**31 for DeepSeek V4.1
    ew = e - E0  # index inside the staged expert chunk; global scales use e
    r = tl.load(TILE_START + tile) + tl.arange(0, BM)
    end = tl.load(TILE_END + tile)
    route = tl.load(ORDER + r, r < end, 0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(0, INT, BK):
        c = base + kk
        x = tl.load(MID + (r - R0)[:, None] * INT + c[None, :], (r < end)[:, None], 0)
        if UNPACKED:
            w = tl.load(W + ew*HID*INT + n[None,:]*INT + c[:,None]).to(tl.float32)
        else:
            w = _weight(W, S, ew, n[None, :], c[:, None], HID, INT, MX)
        if PRECISION == 'bf16':
            acc = tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16), acc)
        elif PRECISION == 'split-bf16':
            hi = x.to(tl.bfloat16)
            rem = x - hi.to(tl.float32)
            lo = rem.to(tl.bfloat16)
            tail = (rem - lo.to(tl.float32)).to(tl.bfloat16)
            wb = w.to(tl.bfloat16)
            acc = tl.dot(tail, wb, acc)
            acc = tl.dot(lo, wb, acc)
            acc = tl.dot(hi, wb, acc)
        else:
            acc = tl.dot(x, w, acc, input_precision=PRECISION)
    acc *= tl.load(G + e)
    if BY_ROW:  # chunk-local rows (XTIER_PREFILL_ROUTE_ACCUM); summed into the token by _accum
        tl.store(ROUTE_OUT + (r - R0)[:, None] * HID + n[None, :], acc, (r < end)[:, None])
    else:
        tl.store(ROUTE_OUT + route[:, None] * HID + n[None, :], acc, (r < end)[:, None])


@tr.jit
def _reduce(R, WEIGHTS, OUT, K: tl.constexpr, B: tl.constexpr, HID: tl.constexpr):
    t = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0., tl.float32)
    for k in range(K):
        value = tl.load(R + (t * K + k) * HID + n, n < HID, 0)
        weight = tl.load(WEIGHTS + t * K + k)
        acc += value * weight
    tl.store(OUT + t * HID + n, acc, n < HID)


@tr.jit
def _accum(R, ORDER, INV, WEIGHTS, ACC, R0, ROWS, K: tl.constexpr, B: tl.constexpr, HID: tl.constexpr):
    """Add one expert chunk's routes (rows R0..R0+ROWS of the sorted order, stored chunk-locally in R)
    into the FP32 per-token accumulator. The token's lowest route slot inside the chunk adds all of its
    in-chunk slots in slot order, so each token is written by one program: no atomics, deterministic."""
    i = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    route = tl.load(ORDER + R0 + i)
    t = route // K
    s = route % K
    lead = True
    for q in tl.static_range(K):
        p = tl.load(INV + t * K + q)
        lead = lead & ((q >= s) | (p < R0) | (p >= R0 + ROWS))
    if lead:
        acc = tl.load(ACC + t.to(tl.int64) * HID + n, n < HID, 0)
        for q in tl.static_range(K):
            p = tl.load(INV + t * K + q)
            inside = (p >= R0) & (p < R0 + ROWS)
            value = tl.load(R + (p - R0).to(tl.int64) * HID + n, (n < HID) & inside, 0)
            weight = tl.where(inside, tl.load(WEIGHTS + t * K + q), 0.)
            acc += value * weight
        tl.store(ACC + t.to(tl.int64) * HID + n, acc, n < HID)


@tr.jit
def _emit(ACC, OUT, B: tl.constexpr, HID: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * B + tl.arange(0, B)
    tl.store(OUT + t * HID + n, tl.load(ACC + t * HID + n, n < HID, 0), n < HID)


@tr.jit
def _add_bf16_rows(ACC, PART, ROWS, B: tl.constexpr, HID: tl.constexpr):
    """Deterministic non-atomic accumulation of unique chunk rows `PART` (BF16) into `ACC` (FP32)."""
    i = tl.program_id(0).to(tl.int64)
    row = tl.load(ROWS + i).to(tl.int64)
    n = tl.program_id(1) * B + tl.arange(0, B)
    mask = n < HID
    acc = tl.load(ACC + row * HID + n, mask, 0.0)
    val = tl.load(PART + i * HID + n, mask, 0.0).to(tl.float32)
    tl.store(ACC + row * HID + n, acc + val, mask)


@tr.jit
def _reduce_band(PARTS, PERM, OFF, ACC, CPU_OUT, CPU_IDX, OUT,
                 IS_FIRST: tl.constexpr, IS_LAST: tl.constexpr, HAS_CPU: tl.constexpr,
                 B: tl.constexpr, HID: tl.constexpr):
    """Chunk-ordered FP32 reduction of a band of per-(chunk, token) BF16 outputs.
    Preserves exact IEEE-754 left-to-right addition order across chunks (bitwise identical to _add_bf16_rows + _emit)."""
    t = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * B + tl.arange(0, B)
    mask = n < HID
    s0 = tl.load(OFF + t)
    s1 = tl.load(OFF + t + 1)
    if IS_FIRST:
        acc = tl.zeros((B,), dtype=tl.float32)
    else:
        if not IS_LAST and s0 == s1:
            return
        acc = tl.load(ACC + t * HID + n, mask, 0.0)
    for idx in range(s0, s1):
        p = tl.load(PERM + idx).to(tl.int64)
        acc += tl.load(PARTS + p * HID + n, mask, 0.0).to(tl.float32)
    if IS_LAST:
        if HAS_CPU:
            ci = tl.load(CPU_IDX + t)
            if ci >= 0:
                acc += tl.load(CPU_OUT + ci.to(tl.int64) * HID + n, mask, 0.0)
        tl.store(OUT + t * HID + n, acc.to(tl.bfloat16), mask)
    else:
        tl.store(ACC + t * HID + n, acc, mask)


@tr.jit
def _gather_fp8_and_ids(XQ_ALL, XSF_LIN, IDS_ALL, W_ALL, ROWS, XQ_OUT, XSF_OUT, IDS_OUT, W_OUT,
                        HAS_SKIP: tl.constexpr, DUMMY_ID, B_H: tl.constexpr, B_SF: tl.constexpr,
                        HID: tl.constexpr, K: tl.constexpr):
    """Fused gather of pre-quantized MXFP8 rows, linear UE8M0 scales, expert ids and routing weights."""
    i = tl.program_id(0).to(tl.int64)
    row = tl.load(ROWS + i).to(tl.int64)
    p = tl.program_id(1)
    if p == 0:
        ks = tl.arange(0, 16)
        km = ks < K
        idv = tl.load(IDS_ALL + row * K + ks, km, 0)
        wv = tl.load(W_ALL + row * K + ks, km, 0.0)
        if HAS_SKIP:
            sk = idv < 0
            idv = tl.where(sk, DUMMY_ID, idv)
            wv = tl.where(sk, 0.0, wv)
        tl.store(IDS_OUT + i * K + ks, idv, km)
        tl.store(W_OUT + i * K + ks, wv, km)
        sfs = tl.arange(0, B_SF)
        sfm = sfs < (HID // 32)
        tl.store(XSF_OUT + i * (HID // 32) + sfs, tl.load(XSF_LIN + row * (HID // 32) + sfs, sfm, 0), sfm)
    c = p * B_H + tl.arange(0, B_H)
    cm = c < HID
    tl.store(XQ_OUT + i * HID + c, tl.load(XQ_ALL + row * HID + c, cm, 0), cm)

@tr.jit
def _mark_chunk_tokens(IDS, TOUCH, FLAGS, M, E, CH: tl.constexpr, NC: tl.constexpr, K: tl.constexpr, BM: tl.constexpr):
    """Mark touch[chunk, token] = 1 in an L2-resident uint8 bitmap and flag out-of-range / skipped (-1) ids."""
    t = tl.program_id(0) * BM + tl.arange(0, BM)
    tm = t < M
    t64 = t.to(tl.int64)
    bad = tl.zeros((BM,), dtype=tl.int32)
    sk = tl.zeros((BM,), dtype=tl.int32)
    for q in tl.static_range(K):
        idv = tl.load(IDS + t64 * K + q, mask=tm, other=0)
        bad = bad | tl.where(tm & ((idv < -1) | (idv >= E)), 1, 0)
        sk = sk | tl.where(tm & (idv == -1), 1, 0)
        c = tl.where(idv >= 0, idv // CH, NC).to(tl.int64)
        tl.store(TOUCH + c * M + t64, 1, mask=tm)
    if tl.sum(bad, 0) > 0:
        tl.atomic_or(FLAGS, 1)
    if tl.sum(sk, 0) > 0:
        tl.atomic_or(FLAGS + 1, 1)


@tr.jit
def _compact_chunk_rows(TOUCH, OFFSETS, ROWS_GPU, M, BT: tl.constexpr):
    """Compact each non-skipped chunk's touched tokens into ascending int32 rows_gpu[offsets[j]:offsets[j+1]]."""
    j = tl.program_id(0).to(tl.int64)
    w0 = tl.load(OFFSETS + j)
    w1 = tl.load(OFFSETS + j + 1)
    if w0 == w1:
        return
    base = TOUCH + j * M
    for t0 in range(0, M, BT):
        cols = t0 + tl.arange(0, BT)
        cm = cols < M
        fl = tl.load(base + cols, mask=cm, other=0).to(tl.int32)
        ps = tl.cumsum(fl, 0) - fl
        tl.store(ROWS_GPU + w0 + ps, cols, mask=fl != 0)
        w0 += tl.sum(fl, 0)


@tr.jit
def _hist_rows(ROWS, RC, N, BLOCK: tl.constexpr):
    """Zero-sync histogram of int32 token indices in ROWS[0..N) into RC."""
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = p < N
    r = tl.load(ROWS + p, mask=m, other=0)
    tl.atomic_add(RC + r, 1, mask=m)


@tr.jit
def _copy_pool_chunk8(A0, A1, S0, S1, PACK0: tl.int64, PACK1: tl.int64, I_HALF_Q, W2_Q, BLOCK: tl.constexpr):
    """Fused 64-bit D2D copy of up to 8 pool-resident experts into staging slot ([gate; up] -> [up; gate] + down)."""
    i = tl.program_id(0)
    p = tl.program_id(1)
    shift = (i & 3).to(tl.int64) * 16
    raw = tl.where(i < 4, (PACK0 >> shift) & 0x3FFF, (PACK1 >> shift) & 0x3FFF)
    if raw == 0x3FFF:
        return
    r = raw.to(tl.int64)
    i64 = i.to(tl.int64)
    off = p.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    w13_q = 2 * I_HALF_Q
    if p * BLOCK < w13_q:
        m0 = off < I_HALF_Q
        tl.store(S0 + i64 * w13_q + off, tl.load(A0 + r * w13_q + I_HALF_Q + off, mask=m0, other=0), mask=m0)
        m1 = (off >= I_HALF_Q) & (off < w13_q)
        tl.store(S0 + i64 * w13_q + off, tl.load(A0 + r * w13_q + (off - I_HALF_Q), mask=m1, other=0), mask=m1)
    if p * BLOCK < W2_Q:
        m2 = off < W2_Q
        tl.store(S1 + i64 * W2_Q + off, tl.load(A1 + r * W2_Q + off, mask=m2, other=0), mask=m2)



class _PersistentWorker:
    """Single daemon worker thread reused across prefill calls; exposes .is_alive() and .join()."""

    def __init__(self, name):
        self._start_ev = threading.Event()
        self._done_ev = threading.Event()
        self._done_ev.set()
        self._fn = None
        self._busy = False
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            self._start_ev.wait()
            self._start_ev.clear()
            fn, self._fn = self._fn, None
            try:
                if fn is not None:
                    fn()
            finally:
                self._busy = False
                self._done_ev.set()

    def launch(self, fn):
        self._done_ev.wait()
        self._done_ev.clear()
        self._fn = fn
        self._busy = True
        self._start_ev.set()
        return self

    def is_alive(self):
        return self._busy

    def join(self):
        self._done_ev.wait()


_lock = threading.Lock()
_slots = None      # packed staging slots, each [w13, w2, s13, s2] for one expert chunk
_slot_res = None   # per-slot (residency_slice, pre_swizzled) metadata for _upload_chunk_fp4_dma
_slot_free_ev = None  # per-slot compute completion event when release_workspace=0
_unpacked = None   # BF16 weights of the chunk being computed (XTIER_PREFILL_UNPACK=1)
_pending = None    # (plan, upload-done event, slot) of a cross-layer prefetch
_copy_stream = None
_d2h_stream = None
_share_host = {}   # XTIER_PREFILL_CPU_SHARE: (hidden, top-k) -> pinned host buffers of the CPU part
_share_cost = {}   # (hidden, intermediate, experts per chunk) -> the measured costs its split balances
_share_worker = None   # the thread of the last CPU part (joined before the next one: a call that raised left it running)
_share_seen = dict(calls=0, chunks=0, skipped=0, cpu_chunks=0, routes=0, cpu_routes=0, gpu_s=0.0, cpu_s=0.0, logged=0.0)
logger = logging.getLogger(__name__)
_first_plan = None
_more_chunks = [False]
_last_plan = None


def set_more_chunks(flag: bool) -> None:
    _more_chunks[0] = bool(flag)
_unpack_stream = None  # XTIER_PREFILL_OVERLAP: FP4->BF16 unpack stream
_unpacked2 = None      # XTIER_PREFILL_OVERLAP: two alternating BF16 weight buffers


def _pending_chunks(pending):
    """[(chunk, upload-done event, slot), ...] of a cross-layer prefetch. Formats: (plan, event, slot)
    from _run; (plan, event, slot, (event, slot)) or (plan, event, slot, [(chunk, event, slot), ...])
    from _run_overlap (XTIER_PREFILL_PREFETCH_CHUNKS)."""
    chunks = [(0, pending[1], pending[2])]
    extra = pending[3] if len(pending) > 3 else None
    if isinstance(extra, list):
        chunks += extra
    elif extra is not None:
        chunks.append((1, extra[0], extra[1]))
    return chunks


def drain_prefetch():
    """Release any in-flight source ownership before model teardown or tests."""
    global _pending
    with _lock:
        if _pending is not None:
            for _, done, _ in _pending_chunks(_pending):
                done.synchronize()
            _pending = None


class _DeviceArray:
    """A raw device address as a torch tensor (no copy) through __cuda_array_interface__."""
    def __init__(self, address, shape, typestr):
        self.__cuda_array_interface__ = dict(shape=tuple(shape), typestr=typestr, data=(int(address), False),
                                             version=3, strides=None)


def _device_tensor(address, shape, dtype):
    if dtype == torch.bfloat16:
        return torch.as_tensor(_DeviceArray(address, shape, '<i2'), device='cuda').view(torch.bfloat16)
    typestr = {torch.int32: '<i4', torch.float32: '<f4'}[dtype]
    return torch.as_tensor(_DeviceArray(address, shape, typestr), device='cuda')


class _Fp4Prefetch:
    """Owner of a cross-layer prefetch from _run_fp4: the slots hold the CUTLASS layout ([up; gate],
    interleaved scales), so the BF16 paths (which compare the owner with `is self`) never reuse them."""
    def __init__(self, plan):
        self.plan = plan


_cudart = None


def _runtime():
    global _cudart
    if _cudart is None:
        from xtier.tables.lookup import _runtime as cuda_runtime   # the library of torch's CUDA, 13 or 12
        lib = cuda_runtime()
        lib.cudaHostRegister.argtypes = [C.c_void_p, C.c_size_t, C.c_uint]
        lib.cudaHostUnregister.argtypes = [C.c_void_p]
        lib.cudaHostGetDevicePointer.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint]
        lib.cudaMemcpy2DAsync.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                                          C.c_size_t, C.c_size_t, C.c_int, C.c_void_p]
        _cudart = lib
    return _cudart


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f'{what} failed with CUDA error {rc}')


def _shard_layout(H, I, nodes=4, group=16):
    # Placement of each node shard inside the original tensors: (staging index, first
    # row of node 0, rows per node, total rows, bytes per row). Order matches xtier_shard.
    IN, HN = I // nodes, H // nodes
    return [(0, 0, IN, 2 * I, H // 2), (0, I, IN, 2 * I, H // 2), (2, 0, IN, 2 * I, H // group),
            (2, I, IN, 2 * I, H // group), (1, 0, HN, H, I // 2), (3, 0, HN, H, I // group)]


def cpu_share():
    """XTIER_PREFILL_CPU_SHARE=1 (off by default; the FP4 prefill path, _run_fp4): a prefill call uploads only the expert
    chunks that pay and the CPU engine, idle during a prefill, computes the others at the same time. A chunk's upload
    costs the same whether one token or five thousand route into it (DeepSeek V4.1: 8 experts, 150 MB, about 2.9 ms on
    PCIe 5.0 x16), and a 22K-token call spreads unevenly: a fifth of the experts get no token, half of them 2% of the
    routes (2026-11-10, recorded prefill routes). Chunks without a token are not uploaded; the chunks with the fewest
    tokens go to the CPU until its predicted time reaches the GPU's (XTIER_PREFILL_CPU_SHARE_TARGET x it, default 1), both
    from costs measured call by call. The CPU's experts take BF16 activations where the GPU's take MXFP8: the output
    is not bitwise the GPU-only one. Not on layers with cold experts.

    Measured on DeepSeek V4.1 (2026-11-11, docs/reports/2026-11-11-prefill-chunk.md): cold prefill +7-10% at 22K-token
    prefill chunks (4 in 10 expert chunks not uploaded, 6% of the routes on the CPU), +3% at 48K-token chunks, nothing
    at 64K. The CPU computes a route about 15 times slower than the GPU, so the gain is the uploads left out, and a
    larger chunk routes more tokens to every expert: less is left for the CPU. The console's DeepSeek card sets it.

    Only this path has it. On NVFP4 layers (GLM-5.3, Qwen3.8) the uploads overlap the GEMMs and cost no time: splitting
    their large prefill calls by experts between the GPU and the CPU engine was measured slower (the cold prefill of a
    128K prompt -33% on GLM-5.3, -6..11% on Qwen3.8) and was not merged."""
    return os.environ.get('XTIER_PREFILL_CPU_SHARE', '0') == '1'


def _share_buffers(H, k, n):
    """Pinned host memory for n rows of the CPU part (x, ids, weights in; FP32 out), kept between calls."""
    b = _share_host.get((H, k))
    if b is None or b['cap'] < n:
        cap = -(-max(n, 4096) // 4096) * 4096
        b = _share_host[(H, k)] = dict(
            cap=cap, x=torch.empty((cap, H), dtype=torch.bfloat16, pin_memory=True),
            ids=torch.empty((cap, k), dtype=torch.int32, pin_memory=True),
            w=torch.empty((cap, k), dtype=torch.float32, pin_memory=True),
            out=torch.empty((cap, H), dtype=torch.float32, pin_memory=True))
    return b


class Prefill:
    """Grouped GPU prefill of one MoE layer whose weights live in host memory.

    Weights are uploaded in expert chunks (XTIER_PREFILL_EXPERT_CHUNK experts, default
    all): two packed staging slots alternate so the copy engine uploads chunk j+1 while
    chunk j is unpacked and computed. With one chunk this is the original single-staging
    path (Qwen3.8); DeepSeek V4.1's 7.6 GB layers use chunks to bound device memory.
    Results do not depend on the chunk size: every route is computed by the same kernels
    with the same operands and reduced in the same order (with XTIER_PREFILL_ROUTE_ACCUM=1
    the FP32 route sum runs in chunk order instead, see __init__).
    """

    def __init__(self, experts, pointers, native=None, hidden=2560, inter=640, swiglu_limit=0.0, swiglu_alpha=1.0,
                 swiglu_beta=0.0):
        self.mxfp4 = getattr(native, 'scale_format', 'nvfp4') == 'mxfp4'
        if self.mxfp4 and os.environ.get('XTIER_PREFILL_FROM_SHARDS', '1') != '1':
            raise ValueError('MXFP4 requires native scale shards')
        self.scale_group = 32 if self.mxfp4 else 16
        self.experts = experts
        self.hidden, self.inter, self.swiglu_limit = int(hidden), int(inter), float(swiglu_limit)
        # GPT-OSS SwiGLU (MiniMax M3): gate * sigmoid(alpha * gate) * (up + beta); 1 / 0 = plain SiLU
        self.swiglu_alpha, self.swiglu_beta = float(swiglu_alpha), float(swiglu_beta)
        H, I = self.hidden, self.inter
        self.layout = _shard_layout(H, I, group=self.scale_group)
        # Experts per uploaded chunk. Default: the whole layer while its unpacked BF16
        # weights fit in 6 GiB (Qwen3.8: ~5 GB, unchanged single staging); otherwise
        # ~1.6 GiB of BF16 per chunk (DeepSeek V4.1: 24 of 384 experts), which also lets
        # the upload of one chunk overlap the computation of the previous one.
        chunk = int(os.environ.get('XTIER_PREFILL_EXPERT_CHUNK', '0') or 0)
        per_expert = 3 * H * I * 2
        if chunk <= 0:
            chunk = experts if experts * per_expert <= 6 << 30 else max(1, (1638 << 20) // per_expert)
        self.chunk = min(chunk, experts)
        # XTIER_PREFILL_PREFETCH=1: while this layer computes, the copy engine uploads
        # the next layer's first weight chunk into a free staging slot. The successor is
        # learned from the observed call order (a mismatch only costs a normal upload).
        # no cross-layer prefetch with cold experts (xtier/cache/cold.py): a layer's virtual expert order exists only
        # during its own call
        self.learned_ring = os.environ.get('XTIER_PREFILL_PREFETCH') == '1' and os.environ.get('XTIER_COLD', '0') != '1'
        self.prefetch_enabled = self.learned_ring or os.environ.get('XTIER_EXPERIMENTAL_PREFILL_PREFETCH') == '1'
        # Keeping all unpacked expert weights alive across layers adds ~6 GiB
        # to the next attention call. Return them to the allocator by default.
        self.release_workspace = os.environ.get('XTIER_PREFILL_RELEASE_WORKSPACE', '1') == '1'
        if self.release_workspace and self.prefetch_enabled and not self.learned_ring:
            raise ValueError('Workspace release cannot retain a next-layer prefetch')
        self.workspace_releases = self.workspace_gc_collections = 0
        self.next_prefill = None
        self.order = 0  # creation (layer) order, set by the standalone adapter
        self.prefetch_hits = self.prefetch_submissions = self.prefetch_uploads = 0
        self.chunk_uploads = self.chunks_skipped = 0
        if self.prefetch_enabled and os.environ.get('XTIER_PREFILL_UNPACK') != '1':
            raise ValueError('Prefetch requires unpacked weights')
        # XTIER_PREFILL_ROUTE_ACCUM=1: instead of one FP32 row per route for the whole call
        # ([m*k, H], 4.0 GB for DeepSeek V4.1 at 32K tokens), each expert chunk's routes go to a
        # chunk-sized buffer and are added into an FP32 [m, H] accumulator right away. Same kernels
        # and operands; only the order of the FP32 additions changes (chunk order, then slot order),
        # which is bitwise identical with one chunk. Meant for several chunks.
        self.route_accum = os.environ.get('XTIER_PREFILL_ROUTE_ACCUM') == '1'
        self.down_tile128 = os.environ.get('XTIER_EXPERIMENTAL_PREFILL_TILE128') == '1'
        if self.down_tile128 and os.environ.get('XTIER_PREFILL_UNPACK') != '1':
            raise ValueError('Experimental down tile requires validated unpacked BF16 weights')
        if self.down_tile128 and os.environ.get('XTIER_DOWN_PRECISION','tf32x3') != 'tf32x3':
            raise ValueError('Experimental down tile requires the original tf32x3 precision')
        shapes = [(experts, 2 * I, H // 2), (experts, H, I // 2),
                  (experts, 2 * I, H // self.scale_group), (experts, H, I // self.scale_group), (experts, 2), (experts,)]
        self.shapes = shapes
        self.slot_shapes = [(self.chunk,) + s[1:] for s in shapes[:4]]
        self.device_globals = None
        self.weights = []
        self.shards = None
        self.native = native
        self.cold = None          # cold experts (cold.PrefillCold): this call's virtual expert order
        self._cold_globals = None
        self.registered = False
        if native is not None and os.environ.get('XTIER_PREFILL_FROM_SHARDS', '1') == '1':
            # Upload straight from the NUMA-bound CPU shards instead of keeping a second
            # full pinned copy of every CPU layer (~1.4 GB each). Bytes are identical.
            shards = native.shards()
            if shards is not None:
                # Pinned lazily (first upload): pages must first be moved to their NUMA node,
                # and pinned pages cannot migrate.
                self.shards = shards
                self.layout = _shard_layout(H, I, len(shards), self.scale_group)
                for i in (4, 5):
                    w = torch.empty(shapes[i], dtype=torch.float32, pin_memory=True)
                    C.memmove(w.data_ptr(), pointers[i], w.numel() * w.element_size())
                    self.weights.append(w)
                return
        for i, (shape, ptr) in enumerate(zip(shapes, pointers)):
            w = torch.empty(shape, dtype=torch.uint8 if i < 4 else torch.float32, pin_memory=True)
            C.memmove(w.data_ptr(), ptr, w.numel() * w.element_size())
            self.weights.append(w)

    def ensure_registered(self):
        if self.shards is None or self.registered:
            return
        if self.native is not None and os.environ.get('XTIER_LOCALIZE', '1') != '0':
            self.native.localize()
        cudart = _runtime(); registered = []
        try:
            for row in self.shards:
                for address, size in row:
                    _check(cudart.cudaHostRegister(address, size, 0), 'cudaHostRegister')
                    registered.append(address)
        except BaseException:
            for address in registered: cudart.cudaHostUnregister(address)
            raise
        self.registered = True

    def close(self):
        self._fp4_swizzled_scales = None
        if self.shards is not None:
            if self.registered:
                torch.cuda.synchronize()
                for row in self.shards:
                    for address, _ in row: _runtime().cudaHostUnregister(address)
            self.shards = None

    def _globals(self):
        # Per-projection global scales stay on the device (a few KiB per layer), so a
        # prefetch into the shared staging never races with this layer's kernels.
        # Per expert: the hot pool and the hybrid / fully cached prefill index them by expert.
        if self.device_globals is None:
            self.device_globals = tuple(w.cuda() for w in self.weights[-2:])
        return self.device_globals

    def _call_globals(self):
        # The global scales this call's prefill kernels index: with cold experts in the
        # call's virtual order (run), which no other caller may see.
        return self._cold_globals if self._cold_globals is not None else self._globals()

    def _upload_chunk(self, slot, e0, e1, stream, up_first=False, virtual=True, indices=None):
        """Copy experts [e0, e1) into a staging slot on `stream` (bytes unchanged). up_first: W13 and
        its scales as [up; gate] (FlashInfer CUTLASS) instead of the engine's [gate; up]. With cold experts
        (virtual) e0..e1 are this call's virtual experts: the RAM slots first, then the cold staging; without
        (virtual=False: the hot pool's fill) they are RAM slots. `indices`: optional subset of {0,1,2,3}."""
        self.ensure_registered()
        n = e1 - e0
        I = self.inter
        if self.shards is None:
            with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
                for i, (dst, src) in enumerate(zip(slot, self.weights[:4])):
                    if indices is not None and i not in indices:
                        continue
                    if up_first and i in (0, 2):  # per expert: contiguous halves (async from pinned memory)
                        for e in range(n):
                            dst[e, :I].copy_(src[e0 + e, I:], non_blocking=True)
                            dst[e, I:].copy_(src[e0 + e, :I], non_blocking=True)
                    else:
                        dst[:n].copy_(src[e0:e1], non_blocking=True)
            return
        layout = self.layout
        if up_first:  # gate rows (first 0) go to I.., up rows (first I) to 0..
            layout = [(index, (I - first) if index in (0, 2) else first, rows, total, width)
                      for index, first, rows, total, width in self.layout]
        cudart = _runtime()
        parts = [(self.shards, e0, e1, 0)]   # (shards, first and end row block, first expert of the staging)
        if virtual and self.cold is not None and self.cold.vexpert is not None:
            C = self.cold.C
            parts = [(self.shards, e0, min(e1, C), 0), (self.cold.shards, max(e0, C) - C, e1 - C, max(C - e0, 0))]
        for shards, s0, s1, d0 in parts:
            if s1 <= s0:
                continue
            if shards is not self.shards:   # the cold staging: read from NVMe while the RAM slots upload
                self.cold.wait(s0, s1)
            for node, row in enumerate(shards):
                for (address, size), (index, first, rows, total, width) in zip(row, layout):
                    if indices is not None and index not in indices:
                        continue
                    dst = slot[index].data_ptr() + d0 * total * width + (first + node * rows) * width
                    _check(cudart.cudaMemcpy2DAsync(dst, total * width, address + s0 * rows * width, rows * width,
                                                    rows * width, s1 - s0, 1, stream), 'cudaMemcpy2DAsync')

    def run(self, x, out, ids, routing, m, k, stream, output_dtype=torch.bfloat16, cold_table=None):
        global _first_plan, _last_plan
        if self.learned_ring:
            first = _first_plan() if _first_plan is not None else None
            if first is None or self.order < first.order:
                _first_plan = weakref.ref(self)
            previous = _last_plan() if _last_plan is not None else None
            # Forward links only (layer order): no prefetch from the last layer back to
            # the first unless another chunk of the same chunked prefill follows immediately.
            if previous is not None and previous is not self and previous.order < self.order:
                previous.next_prefill = weakref.ref(self)
            _last_plan = weakref.ref(self)
        vids = None
        if self.cold is not None:   # this call's virtual order, its cold experts read from NVMe
            self._cold_globals = None
            with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
                ids_t = _device_tensor(ids, (m, k), torch.int32)
                self._cold_globals = self.cold.prepare(self._globals(), ids_t, cold_table)
                vids = torch.where(ids_t >= 0, self.cold.vmap[ids_t.clamp(min=0).long()].int(), ids_t)
            self._cold_ids = vids   # alive until the next call (the allocator reuses it in stream order)
            ids = vids.data_ptr()
        retired = self._run(x, out, ids, routing, m, k, stream, output_dtype)
        if retired:
            self.workspace_releases += 1
            # A first Triton specialization can leave cyclic compilation frames
            # holding the previous global workspace list. Run GC only when weak
            # references prove the buffers survived the completed call frame.
            if any(ref() is not None for ref in retired):
                gc.collect()
                self.workspace_gc_collections += 1

    def _run(self, x, out, ids, routing, m, k, stream, output_dtype=torch.bfloat16):
        global _slots, _unpacked, _pending, _copy_stream
        if m < 1 or not 1 <= k <= 16:
            raise ValueError('Invalid prefill shape')
        H, I, E, CH = self.hidden, self.inter, self.experts, self.chunk
        n_chunks = (E + CH - 1) // CH
        n_slots = 1 if n_chunks == 1 else 2
        unpacked = os.environ.get('XTIER_PREFILL_UNPACK', '0') == '1'
        if os.environ.get('XTIER_PREFILL_FP4') == '1' and self._fp4_ready(output_dtype):
            return self._run_fp4(x, out, ids, routing, m, k, stream)
        if (os.environ.get('XTIER_PREFILL_OVERLAP') == '1' and unpacked and n_chunks > 1
                and self.route_accum and not self.down_tile128 and output_dtype == torch.bfloat16):
            # (hybrid CPU/GPU calls pass FP32 output and use few chunks: keep the chunk-skipping path)
            return self._run_overlap(x, out, ids, routing, m, k, stream, output_dtype)
        with _lock, torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            compute = torch.cuda.current_stream()
            events=[]
            def stamp(name):
                if os.environ.get('XTIER_PREFILL_PROFILE')=='1':
                    event=torch.cuda.Event(enable_timing=True);event.record();events.append((name,event))
            stamp('start')
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('GPU prefill capture is not supported; use native CPU decode')
            if _copy_stream is None:
                _copy_stream = torch.cuda.Stream()
            ready = {}  # chunk index -> (slot, upload-done event)
            if _pending is not None:
                target, done, slot = _pending[:3]
                for _, more, _ in _pending_chunks(_pending)[1:]:
                    # further prefetched chunks of XTIER_PREFILL_OVERLAP: not used on this path
                    more.synchronize()
                _pending = None
                if self.prefetch_enabled and target is self and _slots is not None \
                        and len(_slots) == n_slots and _slots[slot] is not None \
                        and [tuple(t.shape) for t in _slots[slot]] == self.slot_shapes:
                    ready[0] = (slot, done)
                else:
                    # CPU rendezvous also retains pinned source ownership until DMA
                    # completes. Out-of-order calls wait and upload their own weights.
                    done.synchronize()
            if _slots is None or len(_slots) != n_slots:
                _slots = [None] * n_slots
            for j in range(n_slots):
                if _slots[j] is None or [tuple(t.shape) for t in _slots[j]] != self.slot_shapes:
                    _slots[j] = [torch.empty(s, dtype=torch.uint8, device='cuda') for s in self.slot_shapes]
            if ready:
                self.prefetch_hits += 1
            else:
                self.prefetch_uploads += 1
            g13, g2 = self._call_globals()
            freed = [None] * n_slots  # event: slot contents no longer needed by compute

            def upload(j, slot):
                e0, e1 = j * CH, min(E, (j + 1) * CH)
                _copy_stream.wait_stream(compute) if freed[slot] is None else _copy_stream.wait_event(freed[slot])
                self._upload_chunk(_slots[slot], e0, e1, _copy_stream.cuda_stream)
                done = torch.cuda.Event(); done.record(_copy_stream)
                ready[j] = (slot, done)
                self.chunk_uploads += 1

            # Routing metadata (small host rendezvous). One chunk: token blocks of
            # XTIER_GPU_PREFILL_CHUNK as before. Several chunks: one block, so each weight
            # chunk is uploaded once per call.
            block = int(os.environ.get('XTIER_GPU_PREFILL_CHUNK', '4096'))
            if block not in (4096, 8192, 16384):raise ValueError('Unsupported GPU prefill chunk')
            if n_chunks > 1:
                block = m
            state = dict(current=None)
            # Tile shapes (defaults are the validated Qwen3.8 kernels): routes per tile,
            # gate BN/BK, down BN/BK, and optional warps/stages as "w,s".
            def env_int(name, default):
                return int(os.environ.get(name, default))
            tm, gbn, gbk = env_int('XTIER_PREFILL_TILE_M', 32), env_int('XTIER_GATE_BN', 64), env_int('XTIER_GATE_BK', 64)
            dbn, dbk = env_int('XTIER_DOWN_BN', 64), env_int('XTIER_DOWN_BK', 32)
            def launch_opts(name):
                v = os.environ.get(name)
                return {} if not v else dict(zip(('num_warps', 'num_stages'), map(int, v.split(','))))
            gate_opts, down_opts = launch_opts('XTIER_GATE_LAUNCH'), launch_opts('XTIER_DOWN_LAUNCH')

            def acquire(j, active):
                """Weights of chunk j ready for compute; also queue the next active chunk."""
                global _unpacked
                if state['current'] == j:
                    return state['weights']
                if j not in ready:
                    upload(j, j % n_slots if n_slots > 1 else 0)
                slot, done = ready.pop(j)
                compute.wait_event(done)
                w = _slots[slot]
                position = active.index(j)
                if position + 1 < len(active) and active[position + 1] not in ready:
                    nxt = active[position + 1]
                    nslot = (slot + 1) % n_slots
                    if n_slots > 1:
                        upload(nxt, nslot)
                if unpacked:
                    shape = (CH, 2 * I, H)
                    if _unpacked is None or tuple(_unpacked[0].shape) != shape:
                        _unpacked = [torch.empty(shape, dtype=torch.bfloat16, device='cuda'),
                                     torch.empty((CH, H, I), dtype=torch.bfloat16, device='cuda')]
                    count = min(E, (j + 1) * CH) - j * CH
                    _unpack_all[(count, 2 * I * H // 1024)](w[0], w[2], _unpacked[0], 2 * I, H, 1024, MX=self.mxfp4)
                    _unpack_all[(count, H * I // 1024)](w[1], w[3], _unpacked[1], H, I, 1024, MX=self.mxfp4)
                    free = torch.cuda.Event(); free.record(compute); freed[slot] = free
                    weights = (_unpacked[0], _unpacked[1], w[2], w[3], slot)
                    stamp('unpack')
                else:
                    weights = (w[0], w[1], w[2], w[3], slot)
                state.update(current=j, weights=weights)
                return weights

            stamp('upload')
            for start in range(0, m, block):
                size = min(block, m - start)
                di = torch.empty(size * k, dtype=torch.int32, device='cuda')
                _copy_ids[(tr.cdiv(size*k, 256),)](Pointer(ids+start*k*4, torch.int32), di, size*k, 256)
                # The small host rendezvous validates routes and builds bounded
                # tile metadata. No assumptions about balanced expert routing.
                host_ids = di.cpu()
                # -1 marks a route computed elsewhere (hybrid CPU/GPU prefill): it sorts
                # first, gets no tile, and its routing weight must be 0.
                skipped = int((host_ids == -1).sum())
                if int(host_ids.min()) < -1 or int(host_ids.max()) >= E:
                    raise ValueError('Invalid expert id')
                counts = torch.bincount(host_ids[host_ids >= 0] if skipped else host_ids, minlength=E).tolist()
                order = torch.argsort(di, stable=True).to(torch.int32)
                tiles = []; offset = skipped; spans = []  # per chunk: (tile begin, tile end, row begin, rows)
                for j in range(n_chunks):
                    t0, r0 = len(tiles), offset
                    for e in range(j * CH, min(E, (j + 1) * CH)):
                        count = counts[e]
                        for q in range(0, count, tm):
                            tiles.append((e, offset + q, offset + count))
                        offset += count
                    spans.append((t0, len(tiles), r0, offset - r0))
                meta = torch.tensor(tiles, dtype=torch.int32).T.contiguous().cuda()
                active = [j for j in range(n_chunks) if spans[j][3] > 0]
                self.chunks_skipped += n_chunks - len(active)
                rows_max = max([s[3] for s in spans] + [1])
                mid = torch.empty((rows_max, I), dtype=torch.float32, device='cuda')
                accum = self.route_accum
                if accum:
                    inv = torch.empty_like(order)
                    inv[order.long()] = torch.arange(size * k, dtype=torch.int32, device='cuda')
                    acc = torch.zeros((size, H), dtype=torch.float32, device='cuda')
                    routes = torch.empty((rows_max, H), dtype=torch.float32, device='cuda')
                    weights_ptr = Pointer(routing + start * k * 4, torch.float32)
                else:
                    routes = (torch.zeros if skipped else torch.empty)((size*k, H), dtype=torch.float32, device='cuda')
                stamp('route')
                for j in active:
                    t0, t1, r0, rows = spans[j]
                    w13, w2, s13, s2, slot = acquire(j, active)
                    e0 = j * CH
                    _gate[(t1 - t0, I // gbn)](Pointer(x+start*H*2, torch.bfloat16), w13, s13, g13,
                                             order, meta[0, t0:t1], meta[1, t0:t1], meta[2, t0:t1], mid, e0, r0,
                                             k, tm, gbn, gbk, unpacked, H, I, self.swiglu_limit, MX=self.mxfp4,
                                             ALPHA=self.swiglu_alpha, BETA=self.swiglu_beta, **gate_opts)
                    stamp('gate')
                    if self.down_tile128:
                        _down[(t1 - t0, H // 128)](mid, w2, s2, g2, order, meta[0, t0:t1], meta[1, t0:t1],
                                                   meta[2, t0:t1], routes, e0, r0,
                                                   32, 128, 32, os.environ.get('XTIER_DOWN_PRECISION','tf32x3'),
                                                   unpacked, H, I, accum, MX=self.mxfp4, num_warps=8, num_stages=3)
                    else:
                        _down[(t1 - t0, H // dbn)](mid, w2, s2, g2, order, meta[0, t0:t1], meta[1, t0:t1],
                                                  meta[2, t0:t1], routes, e0, r0,
                                                  tm, dbn, dbk, os.environ.get('XTIER_DOWN_PRECISION','tf32x3'),
                                                  unpacked, H, I, accum, MX=self.mxfp4, **down_opts)
                    if not unpacked:
                        free = torch.cuda.Event(); free.record(compute); freed[slot] = free
                    stamp('down')
                    if accum:
                        _accum[(rows, tr.cdiv(H, 256))](routes, order, inv, weights_ptr, acc, r0, rows, k, 256, H)
                        stamp('accum')
                dst = Pointer(out+start*H*(2 if output_dtype==torch.bfloat16 else 4), output_dtype)
                if accum:
                    _emit[(size, tr.cdiv(H, 256))](acc, dst, 256, H)
                else:
                    _reduce[(size, tr.cdiv(H, 256))](routes, Pointer(routing+start*k*4, torch.float32), dst,
                                        k, 256, H)
                stamp('reduce')
            # Cross-layer prefetch: the next layer's first chunk into a slot whose
            # contents this call no longer needs (the only slot, once unpacked).
            target = self.next_prefill() if self.prefetch_enabled and self.next_prefill else None
            if target is not None and unpacked and target.slot_shapes == self.slot_shapes:
                last = state['weights'][4] if state['current'] is not None else 0
                slot = (last + 1) % n_slots
                try:
                    _copy_stream.wait_stream(compute) if freed[slot] is None else _copy_stream.wait_event(freed[slot])
                    target._upload_chunk(_slots[slot], 0, min(target.experts, target.chunk), _copy_stream.cuda_stream)
                    for dst in _slots[slot]:
                        dst.record_stream(_copy_stream)
                    done = torch.cuda.Event(); done.record(_copy_stream)
                    _pending = (target, done, slot)
                    self.prefetch_submissions += 1
                except BaseException:
                    _copy_stream.synchronize()
                    raise
            # Ensures callers can free pointer-only inputs immediately and the
            # next layer/stream can safely overwrite the shared staging buffers.
            torch.cuda.current_stream().synchronize()
            self.last_profile=[dict(phase=events[i][0],ms=events[i-1][1].elapsed_time(events[i][1])) for i in range(1,len(events))]
            if self.release_workspace:
                # Kernels have completed. Return temporary weights to the CUDA
                # allocator so the next attention layer can reuse this memory.
                # No empty_cache(), arithmetic changes or CPU weight eviction.
                # A pending next-layer prefetch keeps only its compressed staging
                # slot; the unpacked workspace is always released.
                keep = _pending[2] if _pending is not None else None
                retired = []
                for j in range(len(_slots)):
                    if j != keep and _slots[j] is not None:
                        retired += [weakref.ref(t) for t in _slots[j]]
                        _slots[j] = None
                if keep is None:
                    _slots = None
                retired += [weakref.ref(t) for t in (_unpacked or [])]
                _unpacked = None
                return tuple(retired)

    def _fp4_ready(self, output_dtype):
        """XTIER_PREFILL_FP4 applies to MXFP4 layers split into several equal expert chunks whose
        global scales are all 1 (CUTLASS takes one fc1 global per expert); BF16 output only; GPUs with FP4 tensor
        cores (SM100 and newer)."""
        ok = getattr(self, '_fp4_ok', None)
        if ok is None:
            g13, g2 = self.weights[-2], self.weights[-1]
            ok = (torch.cuda.get_device_capability()[0] >= 10 and self.mxfp4
                  and self.experts % self.chunk == 0 and self.experts // self.chunk > 1
                  and bool((g13 == 1).all()) and bool((g2 == 1).all()))
            self._fp4_ok = ok
        return ok and output_dtype == torch.bfloat16

    def _ensure_fp4_scales(self, compute):
        """Lazily build a pinned host copy of all experts' [up; gate] CUTLASS-interleaved UE8M0 scales when experts
        are static in RAM (cold is None). Eliminates all 3,840 per-chunk GPU scale-swizzle kernels on both streams."""
        if self.cold is not None or os.environ.get('XTIER_PREFILL_FP4_PRESWIZZLE', '1') == '0':
            return None
        swiz = getattr(self, '_fp4_swizzled_scales', None)
        if swiz is not None:
            return swiz
        from flashinfer import block_scale_interleave
        self.ensure_registered()
        CH, E = self.chunk, self.experts
        s13_h = torch.empty(self.shapes[2], dtype=torch.uint8, pin_memory=True)
        s2_h = torch.empty(self.shapes[3], dtype=torch.uint8, pin_memory=True)
        dummy = torch.empty((1,), dtype=torch.uint8, device='cuda')
        s13_d = torch.empty(self.slot_shapes[2], dtype=torch.uint8, device='cuda')
        s2_d = torch.empty(self.slot_shapes[3], dtype=torch.uint8, device='cuda')
        tmp_slot = [dummy, dummy, s13_d, s2_d]
        raw = compute.cuda_stream
        for e0 in range(0, E, CH):
            e1 = min(E, e0 + CH)
            n = e1 - e0
            self._upload_chunk(tmp_slot, e0, e1, raw, up_first=True, virtual=False, indices=(2, 3))
            s13_h[e0:e1].copy_(block_scale_interleave(s13_d[:n]).reshape_as(s13_d[:n]), non_blocking=True)
            s2_h[e0:e1].copy_(block_scale_interleave(s2_d[:n]).reshape_as(s2_d[:n]), non_blocking=True)
        compute.synchronize()
        self._fp4_swizzled_scales = (s13_h, s2_h)
        return self._fp4_swizzled_scales

    def _upload_chunk_fp4_dma(self, slot, e0, e1, stream):
        """Pure H2D DMA upload on _copy_stream (zero SM kernels on _copy_stream so Attention/GEMM never blocks PCIe
        Copy Engine DMA). Uploads pre-swizzled scales when available and skips pool-resident expert weights."""
        swiz = getattr(self, '_fp4_swizzled_scales', None) if self.cold is None else None
        pre_swiz = swiz is not None
        w_indices = (0, 1) if pre_swiz else None
        n = e1 - e0
        if pre_swiz:
            cudart = _runtime()
            for idx, sh in ((2, swiz[0]), (3, swiz[1])):
                nb = n * sh[0].numel()
                _check(cudart.cudaMemcpy2DAsync(slot[idx].data_ptr(), nb, sh.data_ptr() + e0 * sh[0].numel(),
                                                nb, nb, 1, 1, stream), 'cudaMemcpy2DAsync')
        res = getattr(self, '_fp4_residency', None)
        pool = getattr(self.native, 'pool', None) if res is not None else None
        if pool is None:
            self._upload_chunk(slot, e0, e1, stream, up_first=True, indices=w_indices)
            return (None, pre_swiz)
        rslice = tuple(int(res[e]) for e in range(e0, e1))
        e = e0
        while e < e1:
            if rslice[e - e0] >= 0:
                e += 1
            else:
                f = e + 1
                while f < e1 and rslice[f - e0] < 0:
                    f += 1
                self._upload_chunk([t[e - e0:] for t in slot], e, f, stream, up_first=True, indices=w_indices)
                e = f
        return (rslice, pre_swiz)

    def _prepare_chunk_fp4(self, slot, e0, e1, rinfo, compute):
        """On compute stream right before cutlass_fused_moe: fill pool-resident experts from the live pool arena
        (falling back to H2D if evicted since prefetch); swizzle scales only on cold fallback."""
        rslice, pre_swiz = rinfo if rinfo is not None else (None, False)
        if rslice is not None:
            cur_res = getattr(self, '_fp4_residency', None)
            pool = getattr(self.native, 'pool', None)
            raw = compute.cuda_stream
            w_indices = (0, 1) if pre_swiz else None
            n = len(rslice)
            i_half_b = self.inter * (self.hidden // 2)
            w2_b = self.hidden * (self.inter // 2)
            if (pre_swiz and n <= 8 and pool is not None and cur_res is not None
                    and (i_half_b & 7) == 0 and (w2_b & 7) == 0
                    and pool.arena[0].dtype == torch.uint8 and pool.arena[0].is_contiguous()
                    and pool.arena[1].dtype == torch.uint8 and pool.arena[1].is_contiguous()
                    and slot[0].dtype == torch.uint8 and slot[1].dtype == torch.uint8):
                pack = [0x3FFF] * 8
                n_hit = 0
                for i, r_prev in enumerate(rslice):
                    if r_prev >= 0:
                        r_now = int(cur_res[e0 + i])
                        if 0 <= r_now < 0x3FFF:
                            pack[i] = r_now
                            n_hit += 1
                        else:
                            self._upload_chunk([t[i:] for t in slot], e0 + i, e0 + i + 1, raw, up_first=True,
                                               indices=w_indices)
                if n_hit > 0:
                    p0 = (1 << 62) | pack[0] | (pack[1] << 16) | (pack[2] << 32) | (pack[3] << 48)
                    p1 = (1 << 62) | pack[4] | (pack[5] << 16) | (pack[6] << 32) | (pack[7] << 48)
                    i_half_q = i_half_b >> 3
                    w2_q = w2_b >> 3
                    grid_q = tr.cdiv(max(2 * i_half_q, w2_q), 1024)
                    with torch.cuda.stream(compute):
                        _copy_pool_chunk8[(n, grid_q)](
                            pool.arena[0].view(torch.int64), pool.arena[1].view(torch.int64),
                            slot[0].view(torch.int64), slot[1].view(torch.int64),
                            p0, p1, i_half_q, w2_q, 1024, num_warps=4)
                    self.pool_reused = getattr(self, 'pool_reused', 0) + n_hit
            else:
                for i, r_prev in enumerate(rslice):
                    if r_prev >= 0:
                        r_now = cur_res[e0 + i] if (cur_res is not None and pool is not None) else -1
                        if r_now >= 0:
                            self._copy_from_pool(slot, i, pool, r_now, raw, weights_only=pre_swiz)
                            self.pool_reused = getattr(self, 'pool_reused', 0) + 1
                        else:
                            self._upload_chunk([t[i:] for t in slot], e0 + i, e0 + i + 1, raw, up_first=True,
                                               indices=w_indices)
        if not pre_swiz:
            from flashinfer import block_scale_interleave
            n = e1 - e0
            for i in (2, 3):
                scales = slot[i][:n]
                scales.copy_(block_scale_interleave(scales).reshape_as(scales))

    def _copy_from_pool(self, slot, i, pool, r, stream, weights_only=False):
        """Expert in pool slot r -> staging expert i, [gate; up] -> [up; gate] (bytes unchanged)."""
        I = self.inter
        a = pool.arena
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            for t in ((0,) if weights_only else (0, 2)):
                slot[t][i, :I].copy_(a[t][r, I:], non_blocking=True)
                slot[t][i, I:].copy_(a[t][r, :I], non_blocking=True)
            for t in ((1,) if weights_only else (1, 3)):
                slot[t][i].copy_(a[t][r], non_blocking=True)

    def _share_plan(self, tokens, issued, skip):
        """XTIER_PREFILL_CPU_SHARE: split this call's expert chunks by its routes. tokens[j]: the tokens routed into
        chunk j; issued: the chunks whose upload has started (they stay on the GPU). Adds to `skip` the chunks that
        are not uploaded and returns the ones among them the CPU computes (the others have no token)."""
        key = (self.hidden, self.inter, self.chunk)
        cost = _share_cost.get(key)
        if cost is None:   # PCIe 5.0 x16; refined from the calls
            per_expert = 3 * self.hidden * self.inter * 9 // 16
            up = self.chunk * per_expert / 52e9 * 1e3
            cost = _share_cost[key] = dict(up_ms=up, up0=up, gpu_us=0.25, cpu_us=12.0)
        target = float(os.environ.get('XTIER_PREFILL_CPU_SHARE_TARGET', '1') or 1)
        # The GPU part of a call is as long as the uploads it still has to wait for: the chunks issued before the
        # routes were known (the slot ring, the previous layer's prefetch) arrive while the attention computes, the
        # others one after the other on the copy engine during the call (2026-11-10, DeepSeek V4.1: 36 of 48 chunks,
        # 94 ms of a layer's 219). Its arithmetic over the uploaded experts only bounds it from below.
        free = sorted((int(c), j) for j, c in enumerate(tokens) if j not in issued)
        waits = len(free)
        gpu_routes = sum(int(c) for c in tokens)
        cpu, mine = 0.0, []
        for count, j in free:
            if count == 0:
                skip.add(j)
                waits -= 1
                continue
            c_ms = cost['cpu_us'] * 1e-3 * count
            g_ms = max(cost['up_ms'] * (waits - 1), cost['gpu_us'] * 1e-3 * (gpu_routes - count))
            if cpu + c_ms > target * g_ms:
                break   # the chunks after it hold more tokens each
            cpu, waits, gpu_routes = cpu + c_ms, waits - 1, gpu_routes - count
            mine.append(j)
            skip.add(j)
        return mine

    def _share_start(self, mine, touch, x_t, ids_t, w_t, k, n_chunks):
        """The CPU part of a call: the rows with a route into the chunks `mine`, staged in pinned host memory, and the
        engine on them in a thread (routes into other chunks get id -1). Returns (thread, rows, n, buffers, result)."""
        global _share_worker, _d2h_stream
        H, CH = self.hidden, self.chunk
        if _d2h_stream is None:
            _d2h_stream = torch.cuda.Stream()
        compute = torch.cuda.current_stream()
        mine_idx = torch.tensor(mine, dtype=torch.long, device='cuda')
        mask = torch.zeros(n_chunks + 1, dtype=torch.bool, device='cuda')
        mask[mine_idx] = True
        routes = mask[torch.where(ids_t >= 0, torch.div(ids_t, CH, rounding_mode='floor'), n_chunks).long()]
        rows = touch.index_select(0, mine_idx).any(0).nonzero().squeeze(1)
        # by expert: the engine reads an expert's weights once for up to 8 of a call's tokens, and a rarely used
        # expert's tokens lie far apart in the prompt (in prompt order: 13 us a route, three tokens a read)
        first = torch.where(routes, ids_t, self.experts).index_select(0, rows).min(1).values
        rows = rows[torch.argsort(first, stable=True)]
        n = int(rows.shape[0])
        buf = _share_buffers(H, k, n)
        gx = x_t.index_select(0, rows)
        gi = torch.where(routes, ids_t, -1).index_select(0, rows)
        gw = torch.where(routes, w_t, 0.0).index_select(0, rows)
        gx_ev = torch.cuda.Event()
        gx_ev.record(compute)
        with torch.cuda.stream(_d2h_stream):
            _d2h_stream.wait_event(gx_ev)
            buf['x'][:n].copy_(gx, non_blocking=True)
            buf['ids'][:n].copy_(gi, non_blocking=True)
            buf['w'][:n].copy_(gw, non_blocking=True)
            staged = torch.cuda.Event()
            staged.record(_d2h_stream)
        gx.record_stream(_d2h_stream)
        gi.record_stream(_d2h_stream)
        gw.record_stream(_d2h_stream)
        del gx, gi, gw
        result = {}
        forward, batch = self.native.forward, 2048   # the engine's largest call

        def work():
            try:
                staged.synchronize()
                t0 = time.perf_counter()
                for s in range(0, n, batch):
                    forward(min(batch, n - s), k, buf['x'].data_ptr() + s * H * 2, buf['ids'].data_ptr() + s * k * 4,
                            buf['w'].data_ptr() + s * k * 4, buf['out'].data_ptr() + s * H * 4)
                result['s'] = time.perf_counter() - t0
            except BaseException as e:   # re-raised by the caller after the join
                result['error'] = e

        if _share_worker is None:
            _share_worker = _PersistentWorker('xtier-prefill-cpu')
        worker = _share_worker.launch(work)
        return worker, rows, n, buf, result

    def _share_learn(self, tokens, skip, mine, waits, gpu_s, result):
        """The costs of the next split from this call's times (waits: the chunks uploaded during the call); a log
        line now and then."""
        cost = _share_cost[(self.hidden, self.inter, self.chunk)]
        if waits >= 4:   # the upload time of a chunk, within a factor of four of the link's
            up = min(max(gpu_s * 1e3 / waits, 0.25 * cost['up0']), 4 * cost['up0'])
            cost['up_ms'] = 0.8 * cost['up_ms'] + 0.2 * up
        gpu_routes = sum(int(tokens[j]) for j in range(len(tokens)) if j not in skip)
        if gpu_routes >= 4096 and gpu_s > 0:
            g_per = gpu_s * 1e6 / gpu_routes
            cost['gpu_us'] = 0.8 * cost['gpu_us'] + 0.2 * min(max(g_per, 0.15), 1.0)
        cpu_routes = sum(int(tokens[j]) for j in mine)
        if mine and cpu_routes >= 64 and 's' in result:
            per = result['s'] * 1e6 / cpu_routes
            cost['cpu_us'] = 0.8 * cost['cpu_us'] + 0.2 * min(max(per, 0.25 * cost['cpu_us']), 4 * cost['cpu_us'])
        s = _share_seen
        s['calls'] += 1
        s['chunks'] += len(tokens)
        s['skipped'] += len(skip)
        s['cpu_chunks'] += len(mine)
        s['routes'] += sum(int(c) for c in tokens)
        s['cpu_routes'] += cpu_routes
        s['gpu_s'] += gpu_s
        s['cpu_s'] += result.get('s', 0.0)
        if time.monotonic() - s['logged'] > 30 and s['calls'] >= 20:
            s['logged'] = time.monotonic()
            logger.info('xtier prefill CPU share: %d calls, %.0f%% of the expert chunks not uploaded (%.0f%% on the CPU '
                        'with %.1f%% of the routes), GPU part %.0f ms / CPU part %.0f ms a call; costs: upload %.2f ms a '
                        'chunk, CPU %.1f us a route', s['calls'], 100 * s['skipped'] / s['chunks'],
                        100 * s['cpu_chunks'] / s['chunks'], 100 * s['cpu_routes'] / max(s['routes'], 1),
                        1e3 * s['gpu_s'] / s['calls'], 1e3 * s['cpu_s'] / s['calls'], cost['up_ms'], cost['cpu_us'])
            s.update(calls=0, chunks=0, skipped=0, cpu_chunks=0, routes=0, cpu_routes=0, gpu_s=0.0, cpu_s=0.0)

    def _run_fp4(self, x, out, ids, routing, m, k, stream):
        """XTIER_PREFILL_FP4=1 (SM100+/SM120, MXFP4, BF16 out, unit global scales): prefill through
        FlashInfer's CUTLASS fused MoE (MXFP4 weights x MXFP8 activations, block-32 UE8M0, the
        official reference's activation precision), as sglang's DSv4 path runs GPU-resident experts.

        - pure-DMA staging on _copy_stream with pre-swizzled pinned UE8M0 scales (zero SM kernels on _copy_stream,
          zero weight scale swizzle kernels on compute stream), and m-adaptive slot count (default 24 for m >= 16384,
          12 for smaller m so CPU share still skips sparse chunks), allocated on `compute` stream to share PyTorch's
          primary allocator pool;
        - per expert chunk, one fused call (XTIER_PREFILL_FP4_ROWS default 8192) writing into a bounded BF16 band
          buffer (XTIER_PREFILL_FP4_BAND_ROWS default 65536, <=0.63 GiB), reduced per band in exact ascending chunk
          order via _reduce_band (bitwise identical to per-chunk FP32 _add_bf16_rows + _emit).
        """
        global _slots, _slot_res, _slot_free_ev, _pending, _copy_stream
        from flashinfer import block_scale_interleave, mxfp8_quantize
        from flashinfer.fused_moe import cutlass_fused_moe, cutlass_fused_moe_workspace_size
        from flashinfer.fused_moe.core import ActivationType
        H, E, CH = self.hidden, self.experts, self.chunk
        n_chunks = E // CH
        env_slots = os.environ.get('XTIER_PREFILL_FP4_SLOTS')
        if env_slots:
            raw_slots = int(env_slots)
        elif (_pending is not None and isinstance(_pending[0], _Fp4Prefetch)
              and _pending[0].plan is self and _slots is not None):
            raw_slots = len(_slots)
        elif m < 2048:
            raw_slots = 8
        else:
            slot_bytes = sum(math.prod(s) for s in self.slot_shapes)
            held_bytes = (len(_slots) * slot_bytes) if (_slots is not None and _slots and _slots[0] is not None) else 0
            avail_gib = (torch.cuda.mem_get_info()[0] + held_bytes) / (1 << 30)
            raw_slots = 24 if avail_gib >= 9.0 else (20 if avail_gib >= 7.5 else 12)
        n_slots = min(n_chunks, max(2, raw_slots))
        with _lock, torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            compute = torch.cuda.current_stream()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('GPU prefill capture is not supported; use native CPU decode')
            try:
                self._fp4_residency = None
                if _copy_stream is None:
                    _copy_stream = torch.cuda.Stream()
                ready = {}  # chunk -> (slot, upload-done event)
                if _pending is not None:
                    owner, chunks = _pending[0], _pending_chunks(_pending)
                    _pending = None
                    if (isinstance(owner, _Fp4Prefetch) and owner.plan is self and _slots is not None
                            and _slot_res is not None and len(_slots) == n_slots
                            and all(c < n_chunks and _slots[s] is not None
                                    and [tuple(t.shape) for t in _slots[s]] == self.slot_shapes for c, _, s in chunks)):
                        for c, done, s in chunks:
                            ready[c] = (s, done)
                    else:
                        for _, done, _ in chunks:
                            done.synchronize()
                target = self.next_prefill() if self.prefetch_enabled and self.next_prefill else None
                if target is None and self.prefetch_enabled and self.learned_ring:
                    from xtier.cache import elastic
                    if elastic.is_chunking() or _more_chunks[0]:
                        first = _first_plan() if _first_plan is not None else None
                        if first is not None and first is not self:
                            target = first
                if target is not None and (target.slot_shapes != self.slot_shapes or target.chunk != CH
                                           or not target._fp4_ready(torch.bfloat16)):
                    target = None
                if target is not None:
                    target._fp4_residency = None
                # Ensure pre-swizzled pinned scales exist for self and target before issuing DMA uploads.
                self._ensure_fp4_scales(compute)
                if target is not None:
                    target._ensure_fp4_scales(compute)
                if _slots is None or len(_slots) != n_slots:
                    if _slot_free_ev:
                        for ev in _slot_free_ev:
                            if ev is not None:
                                compute.wait_event(ev)
                    _slots = [None] * n_slots
                    _slot_res = [None] * n_slots
                    _slot_free_ev = [None] * n_slots
                newly_allocated = False
                for j in range(n_slots):
                    if _slots[j] is None or [tuple(t.shape) for t in _slots[j]] != self.slot_shapes:
                        if _slot_free_ev[j] is not None:
                            compute.wait_event(_slot_free_ev[j])
                            _slot_free_ev[j] = None
                        _slots[j] = [torch.empty(s, dtype=torch.uint8, device='cuda') for s in self.slot_shapes]
                        _slot_res[j] = None
                        newly_allocated = True
                if newly_allocated:
                    alloc_ev = torch.cuda.Event()
                    alloc_ev.record(compute)
                else:
                    alloc_ev = None
                n_cross = min(n_slots, target.experts // target.chunk) if target is not None else 0
                cross = []  # [(next-layer chunk, upload-done event, slot), ...]
                deferred_cross_slots = []
                defer_cross = [False]
                if ready:
                    self.prefetch_hits += 1
                else:
                    self.prefetch_uploads += 1
                freed = list(_slot_free_ev)
                state = dict(next=0, cross=0)
                issued = set(ready)
                skip = set()
                if _share_worker is not None and _share_worker.is_alive():
                    _share_worker.join()

                def wait_slot_safe(slot):
                    if freed[slot] is not None:
                        _copy_stream.wait_event(freed[slot])
                    elif alloc_ev is not None:
                        _copy_stream.wait_event(alloc_ev)

                def upload(j, slot):
                    wait_slot_safe(slot)
                    _slot_res[slot] = self._upload_chunk_fp4_dma(
                        _slots[slot], j * CH, (j + 1) * CH, _copy_stream.cuda_stream)
                    done = torch.cuda.Event(); done.record(_copy_stream)
                    ready[j] = (slot, done); issued.add(j)
                    self.chunk_uploads += 1

                def issue_cross(slot):
                    c = state['cross']; state['cross'] += 1
                    wait_slot_safe(slot)
                    _slot_res[slot] = target._upload_chunk_fp4_dma(
                        _slots[slot], c * CH, (c + 1) * CH, _copy_stream.cuda_stream)
                    done = torch.cuda.Event(); done.record(_copy_stream)
                    cross.append((c, done, slot))

                def refill(slot):
                    while state['next'] < n_chunks and (state['next'] in issued or state['next'] in skip):
                        state['next'] += 1
                    if state['next'] < n_chunks:
                        upload(state['next'], slot); state['next'] += 1
                    elif state['cross'] < n_cross:
                        if defer_cross[0]:
                            if len(deferred_cross_slots) < n_cross:
                                deferred_cross_slots.append(slot)
                        else:
                            issue_cross(slot)

                # Start pure-DMA weight uploads before the host rendezvous on the routes.
                used = {s for s, _ in ready.values()}
                for slot in range(n_slots):
                    if slot not in used:
                        refill(slot)

                x_t = _device_tensor(x, (m, H), torch.bfloat16)
                ids_t = _device_tensor(ids, (m, k), torch.int32)
                w_t = _device_tensor(routing, (m, k), torch.float32)
                if k <= 16:
                    touch = torch.zeros((n_chunks + 1, m), dtype=torch.uint8, device='cuda')
                    flags = torch.zeros(2, dtype=torch.int32, device='cuda')
                    _mark_chunk_tokens[(tr.cdiv(m, 256),)](ids_t, touch, flags, m, E, CH, n_chunks, k, 256, num_warps=4)
                    counts = touch[:n_chunks].sum(dim=1, dtype=torch.int64)
                    pools = [getattr(p.native, 'pool', None) if p is not None else None for p in (self, target)]
                    tables = [p.table.long() for p in pools if p is not None]
                    host = torch.cat([counts, flags.long()] + tables).cpu().tolist()
                    if host[n_chunks]:
                        raise ValueError('Invalid expert id')
                    rows = host[n_chunks + 2:]
                    for p, pool in zip((self, target), pools):
                        if pool is not None:
                            p._fp4_residency, rows = rows[:E], rows[E:]
                            if p.cold is not None and p.cold.vexpert is not None:
                                p._fp4_residency = [p._fp4_residency[e] for e in p.cold.vexpert]
                    has_skipped = host[n_chunks + 1] > 0
                    self.chunks_skipped += sum(1 for c in host[:n_chunks] if c == 0)
                    share = None
                    if cpu_share() and self.cold is None and self.native is not None and n_chunks > 1:
                        waits = n_chunks - len(issued)
                        mine = self._share_plan(host[:n_chunks], issued, skip)
                        share = (self._share_start(mine, touch, x_t, ids_t, w_t, k, n_chunks) if mine else None, mine,
                                 waits - len(skip), time.perf_counter())
                        defer_cross[0] = share[0] is not None
                    offsets = [0]
                    for j, c in enumerate(host[:n_chunks]):
                        offsets.append(offsets[-1] + (0 if j in skip else int(c)))
                    total_gpu = offsets[-1]
                    rows_gpu = torch.empty(total_gpu, dtype=torch.int32, device='cuda')
                    if total_gpu > 0:
                        off_gpu = torch.tensor(offsets, dtype=torch.int32, device='cuda')
                        _compact_chunk_rows[(n_chunks,)](touch, off_gpu, rows_gpu, m, 1024, num_warps=4)
                    del touch, flags
                else:
                    ids64 = ids_t.long()
                    token = torch.arange(m, device='cuda')
                    chunk_of = torch.where(ids64 >= 0, ids64 // CH, n_chunks)
                    keys = torch.unique(chunk_of * m + token[:, None])
                    counts = torch.bincount(keys // m, minlength=n_chunks + 1)
                    pools = [getattr(p.native, 'pool', None) if p is not None else None for p in (self, target)]
                    tables = [p.table.long() for p in pools if p is not None]
                    host = torch.cat([counts[:n_chunks], ((ids64 < -1) | (ids64 >= E)).sum()[None],
                                      (ids64 == -1).sum()[None]] + tables).cpu().tolist()
                    if host[n_chunks]:
                        raise ValueError('Invalid expert id')
                    rows = host[n_chunks + 2:]
                    for p, pool in zip((self, target), pools):
                        if pool is not None:
                            p._fp4_residency, rows = rows[:E], rows[E:]
                            if p.cold is not None and p.cold.vexpert is not None:
                                p._fp4_residency = [p._fp4_residency[e] for e in p.cold.vexpert]
                    has_skipped = host[n_chunks + 1] > 0
                    rows_all = keys % m
                    self.chunks_skipped += sum(1 for c in host[:n_chunks] if c == 0)
                    share = None
                    if skip:
                        keep_c = torch.ones(n_chunks + 1, dtype=torch.bool, device='cuda')
                        keep_c[list(skip)] = False
                        keep_c[n_chunks] = False
                        rows_gpu = rows_all[keep_c[keys // m]].to(torch.int32)
                    else:
                        rows_gpu = (rows_all[:int(sum(host[:n_chunks]))] if has_skipped else rows_all).to(torch.int32)
                    offsets = [0]
                    for j, c in enumerate(host[:n_chunks]):
                        offsets.append(offsets[-1] + (0 if j in skip else int(c)))
                    total_gpu = offsets[-1]

                ones = torch.ones(CH, dtype=torch.float32, device='cuda')
                limit = (torch.full((CH,), self.swiglu_limit, dtype=torch.float32, device='cuda')
                         if self.swiglu_limit > 0 else None)
                cap = max(256, int(os.environ.get('XTIER_PREFILL_FP4_ROWS', '8192')))
                calls = []
                for j in range(n_chunks):
                    if j in skip:
                        continue
                    for a in range(offsets[j], offsets[j + 1], cap):
                        calls.append((j, a, min(a + cap, offsets[j + 1])))
                self.fp4_calls = getattr(self, 'fp4_calls', 0) + len(calls)
                by_chunk = {}
                for j, a, b in calls:
                    by_chunk.setdefault(j, []).append((a, b))
                max_nr = max((b - a for _, a, b in calls), default=0)
                ws_cap = min(cap, max(256, ((max_nr + 255) // 256) * 256))
                if max_nr > 0:
                    ws_key = (ws_cap, H, self.inter, E, k, n_chunks)
                    if getattr(self, '_fp4_ws_key', None) != ws_key:
                        self._fp4_ws_bytes = cutlass_fused_moe_workspace_size(
                            ws_cap, H, self.inter, E, k, x_dtype=torch.float8_e4m3fn, weight_dtype=torch.int64,
                            use_mxfp8_act_scaling=True, ep_size=n_chunks, ep_rank=0)
                        self._fp4_ws_key = ws_key
                    workspace = torch.empty(self._fp4_ws_bytes, dtype=torch.uint8, device='cuda')
                else:
                    workspace = None
                fast_glue = max_nr > 0 and H % 32 == 0 and os.environ.get('XTIER_PREFILL_FP4_FAST_GLUE', '1') != '0'
                if fast_glue:
                    xq_all, xsf_all = mxfp8_quantize(x_t, is_sf_swizzled_layout=False, alignment=32)
                    if xq_all.shape == (m, H) and xsf_all.numel() == m * (H // 32):
                        xsf_all = xsf_all.view(m, H // 32).view(torch.uint8)
                        xq_buf = torch.empty((max_nr, H), dtype=torch.float8_e4m3fn, device='cuda')
                        xsf_buf = torch.empty((max_nr, H // 32), dtype=torch.uint8, device='cuda')
                        ids_buf = torch.empty((max_nr, k), dtype=torch.int32, device='cuda')
                        w_buf = torch.empty((max_nr, k), dtype=torch.float32, device='cuda')
                        b_sf = tr.next_power_of_2(max(16, H // 32))
                    else:
                        fast_glue = False

                single_reduce = fast_glue or total_gpu == 0
                if single_reduce:
                    band_cap = max(max_nr, int(os.environ.get('XTIER_PREFILL_FP4_BAND_ROWS', '65536')))
                    bands = []  # [(r_start, r_end, last_chunk_j), ...]
                    if total_gpu > 0:
                        b_start = 0
                        last_j = -1
                        for j in range(n_chunks):
                            if j in skip or offsets[j + 1] == offsets[j]:
                                continue
                            c_end = offsets[j + 1]
                            if c_end - b_start > band_cap and offsets[j] > b_start:
                                bands.append((b_start, offsets[j], last_j))
                                b_start = offsets[j]
                            last_j = j
                        if total_gpu > b_start:
                            bands.append((b_start, total_gpu, last_j))
                    max_band = max((b1 - b0 for b0, b1, _ in bands), default=1)
                    parts_buf = torch.empty((max_band, H), dtype=torch.bfloat16, device='cuda')
                    acc = torch.empty((m, H), dtype=torch.float32, device='cuda') if len(bands) > 1 else None
                    roff = torch.zeros(m + 1, dtype=torch.int32, device='cuda')
                    rc = torch.empty(m, dtype=torch.int32, device='cuda')
                    band_by_last_j = {lj: (idx, b0, b1) for idx, (b0, b1, lj) in enumerate(bands)}
                    cur_band_b0 = bands[0][0] if bands else 0
                    next_band_idx = 0
                else:
                    acc = torch.zeros((m, H), dtype=torch.float32, device='cuda')

                cpu_dev, cpu_idx, cpu_alloc_ev, gpu_start_ev, gpu_end_ev = None, None, None, None, None
                if share is not None:
                    gpu_start_ev = torch.cuda.Event(enable_timing=True)
                    gpu_start_ev.record(compute)
                    if share[0] is not None:
                        _, rows_c_pre, n_c_pre, _, _ = share[0]
                        cpu_dev = torch.empty((n_c_pre, H), dtype=torch.float32, device='cuda')
                        if single_reduce:
                            cpu_idx = torch.full((m,), -1, dtype=torch.int32, device='cuda')
                            cpu_idx[rows_c_pre] = torch.arange(n_c_pre, dtype=torch.int32, device='cuda')
                        cpu_alloc_ev = torch.cuda.Event()
                        cpu_alloc_ev.record(compute)

                for j in range(n_chunks):
                    if j in skip:
                        continue
                    slot, done = ready.pop(j)
                    slices_j = by_chunk.get(j)
                    if slices_j:
                        compute.wait_event(done)
                        self._prepare_chunk_fp4(_slots[slot], j * CH, (j + 1) * CH, _slot_res[slot], compute)
                        w = _slots[slot]
                        for r0, r1 in slices_j:
                            nr = r1 - r0
                            rows = rows_gpu[r0:r1]
                            if fast_glue:
                                xq = xq_buf[:nr]
                                xsf_lin = xsf_buf[:nr]
                                ids_j = ids_buf[:nr]
                                w_j = w_buf[:nr]
                                part = parts_buf[r0 - cur_band_b0 : r1 - cur_band_b0]
                                _gather_fp8_and_ids[(nr, tr.cdiv(H, 512))](
                                    xq_all.view(torch.uint8), xsf_all, ids_t, w_t, rows,
                                    xq.view(torch.uint8), xsf_lin, ids_j, w_j,
                                    has_skipped, ((j + 1) % n_chunks) * CH, 512, b_sf, H, k, num_warps=4)
                                xsf = block_scale_interleave(xsf_lin)
                            else:
                                rows64 = rows.long()
                                ids_j, w_j = ids_t.index_select(0, rows64), w_t.index_select(0, rows64)
                                if has_skipped:
                                    skip_mask = ids_j < 0
                                    ids_j = torch.where(skip_mask, ((j + 1) % n_chunks) * CH, ids_j)
                                    w_j = torch.where(skip_mask, 0.0, w_j)
                                xq, xsf = mxfp8_quantize(x_t.index_select(0, rows64), is_sf_swizzled_layout=True, alignment=32)
                                part = torch.empty((nr, H), dtype=torch.bfloat16, device='cuda')
                            cutlass_fused_moe(input=xq, token_selected_experts=ids_j, token_final_scales=w_j,
                                              fc1_expert_weights=w[0].view(torch.int64), fc2_expert_weights=w[1].view(torch.int64),
                                              output_dtype=torch.bfloat16,
                                              quant_scales=[w[2].view(torch.int32), ones, w[3].view(torch.int32), ones],
                                              input_sf=xsf, swiglu_limit=limit, use_w4_group_scaling=False,
                                              use_mxfp8_act_scaling=True, activation_type=ActivationType.Swiglu,
                                              ep_size=n_chunks, ep_rank=j, tune_max_num_tokens=ws_cap, output=part,
                                              workspace_buffer=workspace)
                            if not single_reduce:
                                acc.index_add_(0, rows.long(), part.float())
                        w = None
                        ev = torch.cuda.Event(); ev.record(compute); freed[slot] = ev
                    else:
                        freed[slot] = done
                    refill(slot)
                    if single_reduce and j in band_by_last_j:
                        b_idx, b0, b1 = band_by_last_j[j]
                        is_last_band = (b_idx == len(bands) - 1)
                        if not is_last_band:
                            b_rows = rows_gpu[b0:b1]
                            nb = b1 - b0
                            perm = torch.argsort(b_rows, stable=True).to(torch.int32)
                            rc.zero_()
                            if nb > 0:
                                _hist_rows[(tr.cdiv(nb, 512),)](b_rows, rc, nb, 512, num_warps=4)
                            torch.cumsum(rc, 0, dtype=torch.int32, out=roff[1:])
                            _reduce_band[(m, tr.cdiv(H, 256))](
                                parts_buf, perm, roff, acc, parts_buf, roff, Pointer(out, torch.bfloat16),
                                b_idx == 0, False, False, 256, H, num_warps=8)
                            next_band_idx = b_idx + 1
                            cur_band_b0 = bands[next_band_idx][0]

                share_result = {}
                if share is not None:
                    started, mine, waits, _ = share
                    gpu_end_ev = torch.cuda.Event(enable_timing=True)
                    gpu_end_ev.record(compute)
                    if started is not None:
                        worker, rows_c, n_c, buf, share_result = started
                        worker.join()
                        if 'error' in share_result:
                            raise share_result['error']
                        with torch.cuda.stream(_copy_stream):
                            _copy_stream.wait_event(cpu_alloc_ev)
                            cpu_dev.copy_(buf['out'][:n_c], non_blocking=True)
                            cpu_ready_ev = torch.cuda.Event()
                            cpu_ready_ev.record(_copy_stream)
                        cpu_dev.record_stream(_copy_stream)
                        if deferred_cross_slots:
                            for s_idx in deferred_cross_slots:
                                if state['cross'] < n_cross:
                                    issue_cross(s_idx)
                        compute.wait_event(cpu_ready_ev)
                        if not single_reduce:
                            acc.index_add_(0, rows_c, cpu_dev)

                if single_reduce:
                    has_cpu = cpu_dev is not None
                    if bands:
                        b_idx = len(bands) - 1
                        b0, b1, _ = bands[b_idx]
                        b_rows = rows_gpu[b0:b1]
                        nb = b1 - b0
                        perm = torch.argsort(b_rows, stable=True).to(torch.int32)
                        rc.zero_()
                        if nb > 0:
                            _hist_rows[(tr.cdiv(nb, 512),)](b_rows, rc, nb, 512, num_warps=4)
                        torch.cumsum(rc, 0, dtype=torch.int32, out=roff[1:])
                        _reduce_band[(m, tr.cdiv(H, 256))](
                            parts_buf, perm, roff, acc if acc is not None else parts_buf,
                            cpu_dev if has_cpu else parts_buf, cpu_idx if has_cpu else roff,
                            Pointer(out, torch.bfloat16),
                            b_idx == 0, True, has_cpu, 256, H, num_warps=8)
                    else:
                        _reduce_band[(m, tr.cdiv(H, 256))](
                            parts_buf, roff, roff, parts_buf,
                            cpu_dev if has_cpu else parts_buf, cpu_idx if has_cpu else roff,
                            Pointer(out, torch.bfloat16),
                            True, True, has_cpu, 256, H, num_warps=8)
                else:
                    _emit[(m, tr.cdiv(H, 256))](acc, Pointer(out, torch.bfloat16), 256, H)

                if cross:
                    cross.sort(key=lambda t: t[0])
                    _pending = (_Fp4Prefetch(target), cross[0][1], cross[0][2], cross[1:])
                    self.prefetch_submissions += 1
                torch.cuda.current_stream().synchronize()
                if share is not None:
                    gpu_s = gpu_start_ev.elapsed_time(gpu_end_ev) * 1e-3
                    self._share_learn(host[:n_chunks], skip, mine, share[2], gpu_s, share_result)
                for p in (self, target):
                    if p is not None:
                        p._fp4_residency = None
                self.last_profile = []
                if self.release_workspace:
                    keep = {s for _, _, s in _pending_chunks(_pending)} if _pending is not None else set()
                    retired = []
                    for j in range(len(_slots)):
                        if j not in keep and _slots[j] is not None:
                            retired += [weakref.ref(t) for t in _slots[j]]
                            _slots[j] = None
                            _slot_res[j] = None
                            _slot_free_ev[j] = None
                        else:
                            _slot_free_ev[j] = freed[j]
                    if not keep:
                        _slots = None
                        _slot_res = None
                        _slot_free_ev = None
                    return tuple(retired)
                _slot_free_ev = list(freed)
            except BaseException:
                torch.cuda.synchronize()
                _pending = None
                _slots = None
                _slot_res = None
                _slot_free_ev = None
                for p in (self, locals().get('target')):
                    if p is not None:
                        p._fp4_residency = None
                raise

    def _run_overlap(self, x, out, ids, routing, m, k, stream, output_dtype=torch.bfloat16):
        """XTIER_PREFILL_OVERLAP=1 (unpacked weights, several expert chunks, route accumulation).

        Same kernels, tiles and FP32 accumulation order as _run, so outputs are bitwise identical;
        only the scheduling differs:
        - FP4->BF16 unpack runs on its own stream into two alternating BF16 buffers, so chunk j+1
          is unpacked while the gate/down kernels of chunk j run (unpack is bandwidth bound, the
          GEMMs are compute bound);
        - N packed staging slots (XTIER_PREFILL_PREFETCH_CHUNKS, default 2) form a ring: as soon as
          chunk j is unpacked, its slot takes the upload of chunk j+N, and in the last N chunks of the
          layer it takes chunk j+N-n_chunks of the next layer (cross-layer prefetch), which then
          uploads during this layer's tail and the next attention layer (N=16 at GLM-5.3 shapes:
          16 x 216 MiB, 256 of 288 experts);
        - the first upload/unpack is issued before the host rendezvous on the routes, so the GPU is
          busy while the host builds tile metadata (built with numpy, same order as the loop);
        - every chunk is treated as active (an empty chunk is unpacked but gets no tiles).
        """
        global _slots, _unpacked, _pending, _copy_stream, _unpack_stream, _unpacked2
        import numpy as np
        H, I, E, CH = self.hidden, self.inter, self.experts, self.chunk
        n_chunks = (E + CH - 1) // CH
        n_slots = min(n_chunks, max(2, int(os.environ.get('XTIER_PREFILL_PREFETCH_CHUNKS', '2'))))
        with _lock, torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            compute = torch.cuda.current_stream()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('GPU prefill capture is not supported; use native CPU decode')
            if _copy_stream is None:
                _copy_stream = torch.cuda.Stream()
            if _unpack_stream is None:
                _unpack_stream = torch.cuda.Stream()
            ready = {}  # chunk -> (slot, upload-done event)
            if _pending is not None:
                target, chunks = _pending[0], _pending_chunks(_pending)
                _pending = None
                if self.prefetch_enabled and target is self and _slots is not None and len(_slots) == n_slots \
                        and all(c < n_chunks and _slots[s] is not None
                                and [tuple(t.shape) for t in _slots[s]] == self.slot_shapes for c, _, s in chunks):
                    for c, done, s in chunks:
                        ready[c] = (s, done)
                else:
                    for _, done, _ in chunks:
                        done.synchronize()
            if _slots is None or len(_slots) != n_slots:
                _slots = [None] * n_slots
            for j in range(n_slots):
                if _slots[j] is None or [tuple(t.shape) for t in _slots[j]] != self.slot_shapes:
                    _slots[j] = [torch.empty(s, dtype=torch.uint8, device='cuda') for s in self.slot_shapes]
            # cross-layer prefetch target (next layer of the learned ring, same staging shapes)
            target = self.next_prefill() if self.prefetch_enabled and self.next_prefill else None
            if target is not None and target.slot_shapes != self.slot_shapes:
                target = None
            n_cross = min(n_slots, (target.experts + target.chunk - 1) // target.chunk) if target is not None else 0
            cross = []  # [(next-layer chunk, upload-done event, slot), ...]
            if ready:
                self.prefetch_hits += 1
            else:
                self.prefetch_uploads += 1
            shape13, shape2 = (CH, 2 * I, H), (CH, H, I)
            if _unpacked2 is None or tuple(_unpacked2[0][0].shape) != shape13:
                _unpacked2 = [[torch.empty(shape13, dtype=torch.bfloat16, device='cuda'),
                               torch.empty(shape2, dtype=torch.bfloat16, device='cuda')] for _ in range(2)]
            g13, g2 = self._call_globals()
            slot_free = [None] * n_slots   # event on the unpack stream: slot contents consumed
            buf_free = [None, None]    # event on the compute stream: BF16 buffer no longer read
            unpacked_done = {}

            def upload(j, slot):
                e0, e1 = j * CH, min(E, (j + 1) * CH)
                if slot_free[slot] is None:
                    _copy_stream.wait_stream(compute)
                else:
                    _copy_stream.wait_event(slot_free[slot])
                self._upload_chunk(_slots[slot], e0, e1, _copy_stream.cuda_stream)
                done = torch.cuda.Event(); done.record(_copy_stream)
                ready[j] = (slot, done)
                self.chunk_uploads += 1

            def unpack(j):
                b = j % 2
                slot, done = ready.pop(j)
                _unpack_stream.wait_event(done)
                if buf_free[b] is None:
                    _unpack_stream.wait_stream(compute)
                else:
                    _unpack_stream.wait_event(buf_free[b])
                w = _slots[slot]; dst = _unpacked2[b]
                count = min(E, (j + 1) * CH) - j * CH
                with torch.cuda.stream(_unpack_stream):
                    _unpack_all[(count, 2 * I * H // 1024)](w[0], w[2], dst[0], 2 * I, H, 1024, MX=self.mxfp4)
                    _unpack_all[(count, H * I // 1024)](w[1], w[3], dst[1], H, I, 1024, MX=self.mxfp4)
                    ev = torch.cuda.Event(); ev.record(_unpack_stream)
                slot_free[slot] = ev
                unpacked_done[j] = ev
                if j + n_slots < n_chunks:
                    upload(j + n_slots, slot)
                elif j + n_slots - n_chunks < n_cross:
                    # Slot is free for the rest of this layer: next layer's chunk. Only the unpack
                    # reads staging slots (the GEMMs get the BF16 buffers), so the unpack event suffices.
                    c = j + n_slots - n_chunks
                    _copy_stream.wait_event(ev)
                    target._upload_chunk(_slots[slot], c * target.chunk,
                                         min(target.experts, (c + 1) * target.chunk), _copy_stream.cuda_stream)
                    done = torch.cuda.Event(); done.record(_copy_stream)
                    cross.append((c, done, slot))

            # Start the weight pipeline before the host rendezvous on the routes.
            used = {s for s, _ in ready.values()}
            free = [s for s in range(n_slots) if s not in used]
            for j in range(min(n_slots, n_chunks)):
                if j not in ready:
                    upload(j, free.pop(0))
            unpack(0)

            def env_int(name, default):
                return int(os.environ.get(name, default))
            tm, gbn, gbk = env_int('XTIER_PREFILL_TILE_M', 32), env_int('XTIER_GATE_BN', 64), env_int('XTIER_GATE_BK', 64)
            dbn, dbk = env_int('XTIER_DOWN_BN', 64), env_int('XTIER_DOWN_BK', 32)
            def launch_opts(name):
                v = os.environ.get(name)
                return {} if not v else dict(zip(('num_warps', 'num_stages'), map(int, v.split(','))))
            gate_opts, down_opts = launch_opts('XTIER_GATE_LAUNCH'), launch_opts('XTIER_DOWN_LAUNCH')
            precision = os.environ.get('XTIER_DOWN_PRECISION', 'tf32x3')

            size = m
            di = torch.empty(size * k, dtype=torch.int32, device='cuda')
            _copy_ids[(tr.cdiv(size * k, 256),)](Pointer(ids, torch.int32), di, size * k, 256)
            order = torch.argsort(di, stable=True).to(torch.int32)
            inv = torch.empty_like(order)
            inv[order.long()] = torch.arange(size * k, dtype=torch.int32, device='cuda')
            host_ids = di.cpu().numpy()
            skipped = int((host_ids == -1).sum())
            if int(host_ids.min()) < -1 or int(host_ids.max()) >= E:
                raise ValueError('Invalid expert id')
            counts = np.bincount(host_ids[host_ids >= 0], minlength=E).astype(np.int64)
            ends = skipped + np.cumsum(counts)          # row end of each expert
            starts = ends - counts
            ntile = (counts + tm - 1) // tm
            tile_e = np.repeat(np.arange(E, dtype=np.int64), ntile)
            first = np.cumsum(ntile) - ntile                # first tile index of each expert
            q = np.arange(tile_e.size, dtype=np.int64) - first[tile_e]
            meta_np = np.ascontiguousarray(np.stack([tile_e, starts[tile_e] + q * tm, ends[tile_e]]).astype(np.int32))
            meta = torch.from_numpy(meta_np).cuda()
            tile_bounds = np.concatenate([[0], np.cumsum(ntile)])
            spans = []
            for j in range(n_chunks):
                e0, e1 = j * CH, min(E, (j + 1) * CH)
                spans.append((int(tile_bounds[e0]), int(tile_bounds[e1]), int(starts[e0]),
                              int(ends[e1 - 1] - starts[e0])))
            self.chunks_skipped += sum(1 for s in spans if s[3] == 0)
            rows_max = max([s[3] for s in spans] + [1])
            mid = torch.empty((rows_max, I), dtype=torch.float32, device='cuda')
            acc = torch.zeros((size, H), dtype=torch.float32, device='cuda')
            routes = torch.empty((rows_max, H), dtype=torch.float32, device='cuda')
            weights_ptr = Pointer(routing, torch.float32)
            for j in range(n_chunks):
                if j + 1 < n_chunks:
                    unpack(j + 1)
                t0, t1, r0, rows = spans[j]
                b = j % 2
                compute.wait_event(unpacked_done.pop(j))
                if rows:
                    w13, w2 = _unpacked2[b]
                    s13, s2 = _slots[0][2], _slots[0][3]  # not read with unpacked weights
                    _gate[(t1 - t0, I // gbn)](Pointer(x, torch.bfloat16), w13, s13, g13,
                                             order, meta[0, t0:t1], meta[1, t0:t1], meta[2, t0:t1], mid, j * CH, r0,
                                             k, tm, gbn, gbk, True, H, I, self.swiglu_limit, MX=self.mxfp4,
                                             ALPHA=self.swiglu_alpha, BETA=self.swiglu_beta, **gate_opts)
                    _down[(t1 - t0, H // dbn)](mid, w2, s2, g2, order, meta[0, t0:t1], meta[1, t0:t1],
                                              meta[2, t0:t1], routes, j * CH, r0,
                                              tm, dbn, dbk, precision, True, H, I, True, MX=self.mxfp4, **down_opts)
                ev = torch.cuda.Event(); ev.record(compute); buf_free[b] = ev
                if rows:
                    _accum[(rows, tr.cdiv(H, 256))](routes, order, inv, weights_ptr, acc, r0, rows, k, 256, H)
            _emit[(size, tr.cdiv(H, 256))](acc, Pointer(out, output_dtype), 256, H)
            # Cross-layer prefetch: the next layer's first n_cross chunks were queued on the copy
            # stream as this layer's last slots were unpacked; they upload while the next
            # attention layer computes.
            if cross:
                cross.sort(key=lambda t: t[0])
                _pending = (target, cross[0][1], cross[0][2], cross[1:])
                self.prefetch_submissions += 1
            compute.wait_stream(_unpack_stream)
            torch.cuda.current_stream().synchronize()
            self.last_profile = []
            if self.release_workspace:
                keep = {s for _, _, s in _pending_chunks(_pending)} if _pending is not None else set()
                retired = []
                for j in range(len(_slots)):
                    if j not in keep and _slots[j] is not None:
                        retired += [weakref.ref(t) for t in _slots[j]]
                        _slots[j] = None
                if not keep:
                    _slots = None
                for pair in (_unpacked2 or []):
                    retired += [weakref.ref(t) for t in pair]
                _unpacked2 = None
                return tuple(retired)
