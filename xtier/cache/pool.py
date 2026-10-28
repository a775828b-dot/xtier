"""XTier cache: GPU slots for the hot experts of the XTier MoE layers.

Port of lkqmoe 0.5.6 LKQstrata (python/lkqmoe/pool.py, after Strata, github.com/Niko1221/Strata, MIT: ranking profile
at start, decayed route counts, periodic gain-ordered swaps, eviction at once / admission after the copy lands), with
the slot count per layer instead of one for all: the global tables keep a row stride of E, so the planner can give a
layer a share of the budget or every expert. A layer holding all its experts (S = E) needs no CPU part: its decode
runs on the GPU alone (no mailbox round trip) and its prefill reads the slots instead of uploading from host memory.

Decode of a partly cached layer, per call (mailbox slot with routes, CUDA-graph capturable):
1. pack: stages x / routes into the mailbox slot, and one read of the layer's residency table decides every route:
   a cached expert gets id -1 for the CPU and its slot for the GPU (both halves from the same read, so they agree);
   the original routes go to the slot too (statistics and the adaptive policy in the mailbox dispatcher).
2. mailbox request: the CPU starts on the other routes (the engine skips id -1 routes).
3. GPU part while the CPU computes: W4A16 gate/up and down over the used slots, then the per-token sum of the GPU
   routes in route order (FP32).
4. mailbox wait, then unpack adds the GPU part to the CPU output.
The residency table is device memory read at replay time: changing it needs no recapture.

Adaptive residency (XTIER_CACHE_ADAPT=1, default): the mailbox dispatcher's policy thread decides swaps from the routes
it sees and posts them to a mapped command ring; the first cached layer of every decode window runs a commit kernel
that unpublishes replaced experts at once and publishes new ones once their copy has landed (csrc/mailbox).
Prompt warm-up (XTIER_CACHE_PROMPT=1, default off; record_prompt): the decode windows alone leave a new request with
the previous one's experts for its first tens of windows; the prefill's routes, counted on the GPU, reach the policy
once per prompt and start a few large rounds at the first decode window.
Elastic slots (XTIER_CACHE_ELASTIC=1, default off; cache/elastic.py): a partly cached layer reserves address space for
S slots and maps `base` of them for good; the memory a prefill needs only while it runs backs the others (`active`
usable) between prefills.
Missed experts on the GPU (XTIER_DECODE_MISS_GPU=share, default 0: off): after the packing, the GPU takes round(share
x n) of a call's n missed experts (at most XTIER_DECODE_MISS_GPU_MAX=8, the most routed first; _miss_select): their
routes get -1 for the CPU like cached ones, their weights are copied from the engine's pinned NUMA shards into staging
slots (_miss_copy: read over PCIe by the SMs, since copy-engine copies in a CUDA graph cannot follow experts chosen on
the GPU) and computed with the cached ones while the CPU computes the others. Pays off only where PCIe moves an expert
about as fast as the CPU computes one (few cores, no AVX-512, memory bandwidth to spare). Not on layers with cold
experts (their RAM slots change during a forward). The cache log's hit counts these experts too.

Numerics: GPU routes are W4A16 (hardware FP4->FP16 / UE8M0 x BF16 for MXFP4, FP32 accumulation, BF16 intermediate),
CPU routes unchanged. Without cached experts the result equals the plain mailbox path except for the FP32 addition of
a zero GPU part.
"""
import ctypes as C
import json
import logging
import math
import os
import time

import torch
import triton
import triton.language as tl

from xtier.backend.kernels import Pointer

logger = logging.getLogger(__name__)
MAX_M = 16             # token tile of the GPU kernels (decode slots hold <= 8 tokens)
MAX_LAYERS = 128       # rows of the global residency table
RING = 4096            # command ring entries (csrc/mailbox POOL_RING)
MAX_ROUTES = 128       # m*k per call (8 tokens x top-k <= 10)
MISS_CHUNK = 8192      # words (8 bytes) per program of _miss_copy
BLOCK = 1024
CONFIG = dict(gbn=32, dbn=64, bk=256, gw=4, gs=2, dw=4, ds=2)


def _tile_k(n):
    """K-tile of a GEMM whose reduction dimension is n: the largest of 256 / 128 / 64 that divides it
    (GLM-5.3 4096 / 2048: 256; Qwen3.8 2560 / 640: 256 / 128; DeepSeek V4.1 5120 / 2304: 256 / 256)."""
    for bk in (256, 128, 64):
        if n % bk == 0:
            return bk
    return 0


_shared = {}
_glob = {}             # table [MAX_LAYERS, E], owner [MAX_LAYERS, E], state [2], ring / seqs addresses


@triton.jit
def _pack(X, IDS, ROUTE, ACT, TABLE, SX, SIDS, SROUTE, SORIG, USED, RANK, RSLOT, SLOT_LIST, SLOT_EXP, OWNER,
          NX, NK, K: tl.constexpr, S, HAS_ACT: tl.constexpr, SP: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(SX + i, tl.load(X + i, i < NX), i < NX)
    if tl.program_id(0) == 0:
        j = tl.arange(0, 128)
        mk = j < NK
        ids = tl.load(IDS + j, mk, other=-1)
        if HAS_ACT:  # skipped tokens (truncated verify): -1 on both halves
            ids = tl.where(tl.load(ACT + j // K, mk, other=1) != 0, ids, -1)
        slot = tl.load(TABLE + ids, mk & (ids >= 0), other=-1)
        tl.store(SIDS + j, tl.where(slot >= 0, -1, ids), mk)
        tl.store(SORIG + j, ids, mk)
        tl.store(SROUTE + j, tl.load(ROUTE + j, mk), mk)
        # used slots -> compact ranks (static shapes): SLOT_LIST[rank] = slot, RSLOT[route] = rank or -1
        s = tl.arange(0, SP)
        tl.store(USED + s, 0, mask=s < S)
        tl.store(SLOT_LIST + j, -1)  # j covers all MAX_ROUTES (=128) entries
        tl.debug_barrier()
        tl.store(USED + slot, 1, mask=mk & (slot >= 0))
        tl.debug_barrier()
        used = tl.load(USED + s, mask=s < S, other=0)
        rank = tl.cumsum(used, 0) - 1
        tl.store(RANK + s, rank, mask=s < S)
        tl.store(SLOT_LIST + rank, s, mask=(s < S) & (used > 0))
        tl.store(SLOT_EXP + rank, tl.load(OWNER + s, mask=(s < S) & (used > 0), other=0), mask=(s < S) & (used > 0))
        tl.debug_barrier()
        r = tl.load(RANK + slot, mask=mk & (slot >= 0), other=-1)
        tl.store(RSLOT + j, tl.where(slot >= 0, r, -1), mk)


@triton.jit
def _commit(RING_P, SEQS, STATE, TABLE, OWNER, E, NR: tl.constexpr):
    """Window start: apply posted swaps. New experts whose copy has landed enter the table; then replaced
    experts of newly posted swaps leave it (and `applied` lets the copy stream overwrite their slots).
    Publish before evict: a landed expert may already be the victim of a newer swap (posted after it landed,
    before this window), and must end up out of the table. A swap is always evicted in an earlier window
    than it is published (its copy waits for `applied`). Table and owner rows both have stride E."""
    posted = tl.atomic_add(SEQS, 0, sem='acquire', scope='sys')
    landed = tl.atomic_add(SEQS + 32, 0, sem='acquire', scope='sys')
    done = tl.load(STATE)
    pub = tl.load(STATE + 1)
    upto = tl.minimum(landed, done)
    for s in range(pub + 1, upto + 1):
        i = (s % NR) * 4
        layer = tl.load(RING_P + i, volatile=True)
        ein = tl.load(RING_P + i + 1, volatile=True)
        slot = tl.load(RING_P + i + 3, volatile=True)
        if ein >= 0:   # -1: a slot the elastic cache released (cache/elastic.py), nothing to publish
            tl.store(TABLE + layer * E + ein, slot)
        tl.store(OWNER + layer * E + slot, ein)
    tl.store(STATE + 1, upto)
    for s in range(done + 1, posted + 1):
        i = (s % NR) * 4
        layer = tl.load(RING_P + i, volatile=True)
        out = tl.load(RING_P + i + 2, volatile=True)
        if out >= 0:
            tl.store(TABLE + layer * E + out, -1)
    tl.store(STATE, posted)
    tl.atomic_xchg(SEQS + 16, posted, sem='release', scope='sys')


@triton.jit
def _halves(Q, S, slot, rows, k0, QE, QR, SE, SR, BN: tl.constexpr, BK: tl.constexpr, MX: tl.constexpr = False):
    """Two FP16 tiles [BN, BK/2] of a pool slot: columns k0+2j (low nibbles) and k0+2j+1 (high nibbles).
    QE/QR: slot and row strides of the packed weights; SE/SR: of the linear E4M3 block scales."""
    pb = tl.load(Q + slot * QE + rows[:, None] * QR + (k0 // 2 + tl.arange(0, BK // 2))[None, :]).to(tl.uint32)
    r = tl.inline_asm_elementwise(
        asm="{ .reg .b8 t; cvt.u8.u32 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
        constraints="=r,r", args=[pb], dtype=tl.uint32, is_pure=True, pack=1)
    lo = (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (r >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    GROUP: tl.constexpr = 32 if MX else 16
    c = k0 // GROUP + tl.arange(0, BK // GROUP)
    sb = tl.load(S + slot * SE + rows[:, None] * SR + c[None, :])
    sc = (tl.where(sb == 0, 2.0 ** -127, (sb.to(tl.uint32) << 23).to(tl.float32, bitcast=True)) if MX else sb.to(tl.float8e4nv, bitcast=True).to(tl.float32).to(tl.float16))
    sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // GROUP, GROUP // 2)), (BN, BK // 2))
    if MX:
        return (lo.to(tl.float32) * sc).to(tl.bfloat16), (hi.to(tl.float32) * sc).to(tl.bfloat16)
    else:
        return lo * sc, hi * sc


@triton.jit
def _gate_up(X, SLOT_LIST, SLOT_EXP, Q13, S13, G13, MID, M, H: tl.constexpr, I: tl.constexpr, LIMIT,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MX: tl.constexpr = False,
             ALPHA: tl.constexpr = 1.0, BETA: tl.constexpr = 0.0, STAGED: tl.constexpr = False):
    p = tl.program_id(0)
    slot = tl.load(SLOT_LIST + p)
    if slot < 0:
        return
    row = p
    if STAGED:   # missed experts the GPU took (_miss_select): SLOT_LIST holds the ranks, Q13 / S13 the staging slots
        row = slot
        slot = p
    slot = slot.to(tl.int64)
    e = tl.load(SLOT_EXP + p).to(tl.int64)
    tok = tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        ke = k0 + 2 * tl.arange(0, BK // 2)
        xe = tl.load(X + tok[:, None] * H + ke[None, :], mask=tok[:, None] < M, other=0.0).to(tl.bfloat16 if MX else tl.float16)
        xo = tl.load(X + tok[:, None] * H + ke[None, :] + 1, mask=tok[:, None] < M, other=0.0).to(tl.bfloat16 if MX else tl.float16)
        gl, gh = _halves(Q13, S13, slot, n, k0, 2 * I * (H // 2), H // 2, 2 * I * (H // (32 if MX else 16)), H // (32 if MX else 16), BN, BK, MX)
        ul, uh = _halves(Q13, S13, slot, n + I, k0, 2 * I * (H // 2), H // 2, 2 * I * (H // (32 if MX else 16)), H // (32 if MX else 16), BN, BK, MX)
        accg = tl.dot(xe, tl.trans(gl), accg)
        accg = tl.dot(xo, tl.trans(gh), accg)
        accu = tl.dot(xe, tl.trans(ul), accu)
        accu = tl.dot(xo, tl.trans(uh), accu)
    g = accg * tl.load(G13 + e * 2)
    u = accu * tl.load(G13 + e * 2 + 1)
    if LIMIT > 0:
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        g = tl.minimum(g, LIMIT)
    if ALPHA != 1.0 or BETA != 0.0:   # GPT-OSS SwiGLU (MiniMax M3)
        val = g / (1.0 + tl.exp(-ALPHA * g)) * (u + BETA)
    else:
        val = g / (1.0 + tl.exp(-g)) * u
    tl.store(MID + (row * BM + tok[:, None]) * I + n[None, :], val.to(tl.bfloat16), mask=tok[:, None] < M)


@triton.jit
def _down(MID, SLOT_LIST, SLOT_EXP, Q2, S2, G2, PART, M, H: tl.constexpr, I: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MX: tl.constexpr = False, STAGED: tl.constexpr = False):
    p = tl.program_id(0)
    slot = tl.load(SLOT_LIST + p)
    if slot < 0:
        return
    row = p
    if STAGED:   # as in _gate_up
        row = slot
        slot = p
    slot = slot.to(tl.int64)
    e = tl.load(SLOT_EXP + p).to(tl.int64)
    tok = tl.arange(0, BM)
    h = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        ke = k0 + 2 * tl.arange(0, BK // 2)
        ae = tl.load(MID + (row * BM + tok[:, None]) * I + ke[None, :], mask=tok[:, None] < M, other=0.0).to(tl.bfloat16 if MX else tl.float16)
        ao = tl.load(MID + (row * BM + tok[:, None]) * I + ke[None, :] + 1, mask=tok[:, None] < M, other=0.0).to(tl.bfloat16 if MX else tl.float16)
        wl, wh = _halves(Q2, S2, slot, h, k0, H * (I // 2), I // 2, H * (I // (32 if MX else 16)), I // (32 if MX else 16), BN, BK, MX)
        acc = tl.dot(ae, tl.trans(wl), acc)
        acc = tl.dot(ao, tl.trans(wh), acc)
    acc = acc * tl.load(G2 + e)
    tl.store(PART + (row * BM + tok[:, None]) * H + h[None, :], acc, mask=tok[:, None] < M)


@triton.jit
def _combine(PART, RSLOT, W, GOUT, H: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, B: tl.constexpr):
    """GPU part of token t: sum over its GPU routes in route order (FP32); 0 when none."""
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), dtype=tl.float32)
    for q in tl.static_range(K):
        r = tl.load(RSLOT + t * K + q)
        w = tl.load(W + t * K + q)
        acc += tl.where(r >= 0, w, 0.0) * tl.load(PART + (tl.maximum(r, 0) * BM + t) * H + h,
                                                  mask=(h < H) & (r >= 0), other=0.0)
    tl.store(GOUT + t * H + h, acc, mask=h < H)


@triton.jit
def _unpack(SOUT, GOUT, OUT, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    # .cv: always re-read host memory written by the CPU after the mailbox handshake.
    c = tl.load(SOUT + i, i < N, cache_modifier='.cv')
    tl.store(OUT + i, c + tl.load(GOUT + i, i < N), i < N)


@triton.jit
def _miss_select(SIDS, RSLOT, MISS_E, MISS_R, NK, SHARE, CAP, NP: tl.constexpr, CP: tl.constexpr):
    """XTIER_DECODE_MISS_GPU, after _pack: the GPU takes round(SHARE x n) of the call's n missed experts (at most CAP;
    the most routed first, then the lowest id). Their routes get -1 for the CPU and ranks after the used slots' for
    the GPU part; MISS_E / MISS_R[q]: expert and rank of staging slot q, -1 past the last."""
    j = tl.arange(0, NP)
    mk = j < NK
    ids = tl.load(SIDS + j, mk, other=-1)   # the CPU's routes (-1: cached or skipped)
    rs = tl.load(RSLOT + j, mk, other=-1)
    used = tl.max(rs, 0) + 1                # ranks 0..used-1 are the cached experts'
    v = ids >= 0
    same = (ids[:, None] == ids[None, :]) & v[:, None] & v[None, :]
    first = v & (tl.sum((same & (j[None, :] < j[:, None])).to(tl.int32), 1) == 0)   # one route per missed expert
    count = tl.sum(same.to(tl.int32), 1)
    take = tl.minimum((tl.sum(first.to(tl.int32), 0) * SHARE + 0.5).to(tl.int32), CAP)
    ahead = first[None, :] & ((count[None, :] > count[:, None]) |
                              ((count[None, :] == count[:, None]) & (ids[None, :] < ids[:, None])))
    order = tl.sum(ahead.to(tl.int32), 1)   # of a first route: its expert's place
    place = tl.sum(tl.where(same & first[None, :], order[None, :], 0), 1)   # of every route
    gpu = v & (place < take)
    tl.store(SIDS + j, tl.where(gpu, -1, ids), mk)
    tl.store(RSLOT + j, tl.where(gpu, used + place, rs), mk)
    q = tl.arange(0, CP)
    tl.store(MISS_E + q, -1)
    tl.store(MISS_R + q, -1)
    tl.debug_barrier()
    taken = first & (order < take)
    tl.store(MISS_E + order, ids, mask=taken)
    tl.store(MISS_R + order, used + order, mask=taken)


@triton.jit
def _miss_copy(SRC, DST, PIECES, MISS_E, NP, CHUNK: tl.constexpr, B: tl.constexpr):
    """The missed experts the GPU took into the staging slots, in 8-byte words. Program (q * NP + i, c): words
    [c * CHUNK, (c + 1) * CHUNK) of piece i (one NUMA node's rows of one tensor: contiguous per expert in its pinned
    shard) of expert MISS_E[q]. PIECES[i]: source offset from SRC, words per expert, offset in staging slot 0 from
    DST, words per staging slot."""
    q = tl.program_id(0) // NP
    i = tl.program_id(0) % NP
    e = tl.load(MISS_E + q)
    if e < 0:
        return
    n = tl.load(PIECES + i * 4 + 1)
    start = tl.program_id(1) * CHUNK
    if start >= n:
        return
    src = SRC + tl.load(PIECES + i * 4) + e.to(tl.int64) * n
    dst = DST + tl.load(PIECES + i * 4 + 2) + q * tl.load(PIECES + i * 4 + 3)
    for w in range(0, CHUNK, B):
        o = start + w + tl.arange(0, B)
        tl.store(dst + o, tl.load(src + o, o < n), o < n)


def _scratch(H, I, dev):
    """Buffers shared by every cached layer (calls run one after another on the decode stream)."""
    key = (H, I)
    if key not in _shared:
        _shared[key] = dict(
            mid=torch.empty((MAX_ROUTES, MAX_M, I), dtype=torch.bfloat16, device=dev),
            part=torch.empty((MAX_ROUTES, MAX_M, H), dtype=torch.float32, device=dev),
            gout=torch.empty((8, H), dtype=torch.float32, device=dev),
            rslot=torch.empty(MAX_ROUTES, dtype=torch.int32, device=dev),
            slot_list=torch.empty(MAX_ROUTES, dtype=torch.int32, device=dev),
            slot_exp=torch.empty(MAX_ROUTES, dtype=torch.int32, device=dev),
            # staging of the GPU-only path (full residency): written by _pack, never read
            sx=torch.empty((8, H), dtype=torch.int16, device=dev),
            sids=torch.empty(MAX_ROUTES, dtype=torch.int32, device=dev),
            sroute=torch.empty(MAX_ROUTES, dtype=torch.int32, device=dev),
            sorig=torch.empty(MAX_ROUTES, dtype=torch.int32, device=dev),
        )
    return _shared[key]


_miss = {}   # XTIER_DECODE_MISS_GPU: share, cap, expert / rank lists, staging slots per (H, I); shared by every layer


def _miss_stage(pool):
    """XTIER_DECODE_MISS_GPU on a partly cached layer: the staging slots shared by the layers of its shape, allocated
    with the layer's slots (before the KV cache is sized); None when off, every expert is cached, the layer has cold
    experts or no engine shards."""
    share = float(os.environ.get('XTIER_DECODE_MISS_GPU', '0') or 0)
    if share <= 0 or pool.full:
        return None
    prefill = pool.prefill
    if prefill.shards is None or getattr(pool.plan, 'slots', pool.E) < pool.E:
        if not _miss.get('warned'):
            _miss['warned'] = True
            logger.warning('xtier cache: XTIER_DECODE_MISS_GPU is off on layers with cold experts or without engine '
                           'shards (XTIER_PREFILL_FROM_SHARDS=0)')
        return None
    cap = int(os.environ.get('XTIER_DECODE_MISS_GPU_MAX', '8') or 8)
    if not share <= 1 or not 1 <= cap <= MAX_ROUTES:
        raise ValueError(f'xtier cache: XTIER_DECODE_MISS_GPU={share} (0..1), XTIER_DECODE_MISS_GPU_MAX={cap} '
                         f'(1..{MAX_ROUTES})')
    dev = pool.arena[0].device
    if 'e' not in _miss:
        cp = triton.next_power_of_2(cap)
        _miss.update(share=share, cap=cap, cp=cp, e=torch.full((cp,), -1, dtype=torch.int32, device=dev),
                     r=torch.full((cp,), -1, dtype=torch.int32, device=dev))
    key = (pool.H, pool.I)
    if key not in _miss:   # in the pool's slot layout: w13, w2, s13, s2 one after the other
        shapes = [(cap,) + tuple(s[1:]) for s in prefill.shapes[:4]]
        sizes = [math.prod(s) for s in shapes]
        buf = torch.empty(sum(sizes), dtype=torch.uint8, device=dev)
        offsets = [sum(sizes[:i]) for i in range(4)]
        _miss[key] = (buf, [buf[o:o + n].view(s) for o, n, s in zip(offsets, sizes, shapes)], offsets)
        logger.info('xtier cache: decode: %g of a call\'s missed experts (at most %d) on the GPU, copied from the '
                    'pinned shards over PCIe (%.0f MiB staging for H %d, I %d)', share, cap, buf.numel() / 2**20,
                    pool.H, pool.I)
    return _miss[key]


def _miss_layer(pool):
    """XTIER_DECODE_MISS_GPU at the fill of a partly cached layer (outside capture, its shards pinned): the copy table
    of its shard pieces (_miss_copy) with the staging slots, or None."""
    staged = _miss_stage(pool)
    if staged is None:
        return None
    buf, stage, offsets = staged
    prefill = pool.prefill
    dev = pool.arena[0].device
    from xtier.backend.prefill import _check, _runtime
    prefill.ensure_registered()
    cudart = _runtime()
    pieces = []   # device address, bytes per expert, offset in staging slot 0, bytes per staging slot
    for node, shard in enumerate(prefill.shards):
        for (address, _), (index, first, rows, total, width) in zip(shard, prefill.layout):
            d = C.c_void_p()
            _check(cudart.cudaHostGetDevicePointer(C.byref(d), C.c_void_p(address), 0), 'cudaHostGetDevicePointer')
            pieces.append((d.value, rows * width, offsets[index] + (first + node * rows) * width, total * width))
    src = min(p[0] for p in pieces)
    table = [(a - src, n, o, s) for a, n, o, s in pieces]
    if any(x % 8 for p in table for x in p):
        raise ValueError('xtier cache: shard pieces not in 8-byte words')
    return dict(src=src, dst=buf.data_ptr(), stage=stage, n=len(table), words=max(p[1] for p in table) // 8,
                pieces=torch.tensor([[x // 8 for x in p] for p in table], dtype=torch.int64, device=dev))


class Profile:
    """XTIER_CACHE_PROFILE: {"experts": E, "layer_ids": [model layer of each row], "layers": [[expert ids, most used
    first], ...]} (the LKQstrata profile format). A layer takes the row of its model layer id."""
    _data = None

    @classmethod
    def ranking(cls, layer_id, experts):
        if cls._data is None:
            path = os.environ.get('XTIER_CACHE_PROFILE')
            cls._data = {}
            if path:
                with open(path) as f:
                    cls._data = json.load(f)
        d = cls._data
        if not d or int(d.get('experts', -1)) != experts:
            return []
        ids = d.get('layer_ids')
        if ids is None or layer_id not in ids:
            return []
        return [int(e) for e in d['layers'][ids.index(layer_id)]]


class LayerPool:
    """The cache of one XTier MoE layer: S slots in the engine's shard layout and its residency table row."""
    layers = []

    def __init__(self, prefill, slots, plan, mailbox, layer_id):
        self.prefill, self.S, self.plan, self.mailbox = prefill, int(slots), plan, mailbox
        self.layer_id = layer_id
        H, I, E = prefill.hidden, prefill.inter, prefill.experts
        self.H, self.I, self.E, self.limit = H, I, E, float(prefill.swiglu_limit)
        self.alpha, self.beta = prefill.swiglu_alpha, prefill.swiglu_beta
        self.full = self.S >= E
        self.S = min(self.S, E)
        self.bk_h, self.bk_i = _tile_k(H), _tile_k(I)   # gate/up reduce over H, down over I
        if not self.bk_h or not self.bk_i or I % CONFIG['gbn'] or H % CONFIG['dbn']:
            raise ValueError(f'xtier cache: unsupported shape H={H} I={I}')
        dev = torch.device('cuda', torch.cuda.current_device())
        # same per-expert shapes as the prefill staging slots: w13, w2, s13, s2 (bytes, unchanged)
        from xtier.cache import elastic
        self.base, top = elastic.layout(self.S, E) if not self.full else (self.S, self.S)
        if top > self.base:   # address space for `top` slots, `base` mapped (the rest backs extra ones between prefills)
            self.base = elastic.fit_base(mailbox.lib, self.base, [tuple(s[1:]) for s in prefill.shapes[:4]])
            self.S = top
            self.arena = [elastic.arena(mailbox.lib, self.S, tuple(s[1:]), self.base, dev) for s in prefill.shapes[:4]]
        else:
            self.S = self.base
            self.arena = [torch.empty((self.S,) + tuple(s[1:]), dtype=torch.uint8, device=dev)
                          for s in prefill.shapes[:4]]
        self.active = self.base
        if self.full:   # never swapped: a private table, filled in expert order (slot e = expert e)
            self.index = None
            self.table = torch.full((E,), -1, dtype=torch.int32, device=dev)
            self.owner = torch.zeros((E,), dtype=torch.int32, device=dev)
        else:   # a row of the global tables; the row is also the layer id of the command ring (0 counts windows)
            if not _glob:
                _glob['table'] = torch.full((MAX_LAYERS, E), -1, dtype=torch.int32, device=dev)   # expert -> slot
                _glob['owner'] = torch.zeros((MAX_LAYERS, E), dtype=torch.int32, device=dev)  # slot -> expert
                _glob['state'] = torch.zeros(2, dtype=torch.int32, device=dev)   # commands evicted / published
                _glob['E'], _glob['rows'] = E, 0
                ptr = (C.c_void_p * 2)()   # the mapped command ring must exist before decode graphs are captured
                if mailbox.lib.xtier_mailbox_pool_ring(C.cast(ptr, C.c_void_p)) < 0:
                    raise RuntimeError(mailbox.error())
                _glob['ring'], _glob['seqs'] = ptr[0], ptr[1]
            if _glob['E'] != E:
                raise ValueError('xtier cache: every partly cached layer needs the same expert count')
            self.index = _glob['rows']
            if self.index >= MAX_LAYERS:
                raise ValueError('xtier cache: too many layers')
            _glob['rows'] += 1
            self.table = _glob['table'][self.index]
            self.owner = _glob['owner'][self.index]
        self.used = torch.empty(max(self.S, 1), dtype=torch.int32, device=dev)
        self.rank = torch.empty(max(self.S, 1), dtype=torch.int32, device=dev)
        self.resident = [-1] * self.S
        self.filled = False
        self.scratch = _scratch(H, I, dev)
        self.globals = None
        self.miss = None   # XTIER_DECODE_MISS_GPU: _miss_layer at the fill (the staging slots now)
        _miss_stage(self)
        self._pool_on = False
        LayerPool.layers.append(self)

    def bytes(self):
        return sum(a[:self.active].numel() for a in self.arena)

    def fill(self, experts, stream):
        """Upload `experts` into slots 0.. (copy engine, from the pinned NUMA shards) and publish them.
        Call outside graph capture, with no decode in flight (startup)."""
        experts = list(range(self.E)) if self.full else [e for e in experts if 0 <= e < self.E][:self.active]
        upload = type(self.prefill)._upload_chunk   # from host memory (a fully cached layer's prefill reads the slots)
        ram = None
        if getattr(self.plan, 'slots', self.E) < self.E:   # cold experts: from the experts' RAM slots, only those in RAM
            ram = self.plan.slot_table()
            experts = [e for e in experts if ram[e] >= 0]
        with torch.cuda.stream(stream):
            if self.full:
                upload(self.prefill, self.arena, 0, self.E, stream.cuda_stream)
            for j, e in enumerate(experts if not self.full else ()):
                s = e if ram is None else ram[e]
                upload(self.prefill, [a[j:j + 1] for a in self.arena], s, s + 1, stream.cuda_stream, virtual=False)
            table = torch.full((self.E,), -1, dtype=torch.int32)
            owner = torch.zeros((self.E,), dtype=torch.int32)
            for j, e in enumerate(experts):
                table[e] = j; owner[j] = e
            self.table.copy_(table, non_blocking=False); self.owner.copy_(owner, non_blocking=False)
        stream.synchronize()
        self.resident = experts + [-1] * (self.S - len(experts))
        self.filled = True
        self.miss = _miss_layer(self)
        return len(experts)

    def _kernels(self, m, k, x, routing, gout):
        """GPU part: gate/up and down over the used slots, per-token sum of the GPU routes into gout."""
        if self.globals is None:
            self.globals = self.prefill._globals()
        g13, g2 = self.globals
        H, I, cfg, sc = self.H, self.I, CONFIG, self.scratch
        nk = m * k
        xb = Pointer(x, torch.bfloat16)
        _gate_up[(nk, I // cfg['gbn'])](xb, sc['slot_list'], sc['slot_exp'], self.arena[0], self.arena[2], g13,
                                        sc['mid'], m, H, I, self.limit, MAX_M, cfg['gbn'], self.bk_h,
                                        MX=self.prefill.mxfp4, ALPHA=self.alpha, BETA=self.beta,
                                        num_warps=cfg['gw'], num_stages=cfg['gs'])
        _down[(nk, H // cfg['dbn'])](sc['mid'], sc['slot_list'], sc['slot_exp'], self.arena[1], self.arena[3], g2,
                                     sc['part'], m, H, I, MAX_M, cfg['dbn'], self.bk_i,
                                     MX=self.prefill.mxfp4, num_warps=cfg['dw'], num_stages=cfg['ds'])
        if self.miss is not None:   # the missed experts _miss_select gave the GPU: into the staging slots, then as above
            mi, cap, stage = self.miss, _miss['cap'], self.miss['stage']
            _miss_copy[(cap * mi['n'], triton.cdiv(mi['words'], MISS_CHUNK))](
                Pointer(mi['src'], torch.int64), Pointer(mi['dst'], torch.int64), mi['pieces'], _miss['e'], mi['n'],
                MISS_CHUNK, 1024)
            _gate_up[(cap, I // cfg['gbn'])](xb, _miss['r'], _miss['e'], stage[0], stage[2], g13, sc['mid'], m, H, I,
                                             self.limit, MAX_M, cfg['gbn'], self.bk_h, MX=self.prefill.mxfp4,
                                             ALPHA=self.alpha, BETA=self.beta, STAGED=True, num_warps=cfg['gw'],
                                             num_stages=cfg['gs'])
            _down[(cap, H // cfg['dbn'])](sc['mid'], _miss['r'], _miss['e'], stage[1], stage[3], g2, sc['part'], m, H,
                                          I, MAX_M, cfg['dbn'], self.bk_i, MX=self.prefill.mxfp4, STAGED=True,
                                          num_warps=cfg['dw'], num_stages=cfg['ds'])
        _combine[(m, triton.cdiv(H, 1024))](sc['part'], sc['rslot'], Pointer(routing, torch.float32), gout,
                                            H, k, MAX_M, 1024)

    def _route(self, m, k, x, ids, routing, act, staging):
        H, sc = self.H, self.scratch
        nk = m * k
        if nk > MAX_ROUTES or m > 8:
            raise ValueError('xtier cache: decode slot too large')
        sx, sids, sroute, sorig = staging
        S = max(self.S, 1)
        if torch.cuda.is_current_stream_capturing():
            _glob['captured'] = True   # the adaptive policy starts only after the decode graphs exist
        if self.index == 0 and _glob.get('adapt'):
            # once per decode window (anywhere in the window is safe: a slot is overwritten only after this
            # kernel acknowledged its eviction): apply the posted swaps
            _commit[(1,)](Pointer(_glob['ring'], torch.int32), Pointer(_glob['seqs'], torch.int32), _glob['state'],
                          _glob['table'], _glob['owner'], self.E, RING)
        _pack[(triton.cdiv(m * H, BLOCK),)](
            Pointer(x, torch.int16), Pointer(ids, torch.int32), Pointer(routing, torch.int32),
            Pointer(act or ids, torch.int32), self.table, Pointer(sx, torch.int16), Pointer(sids, torch.int32),
            Pointer(sroute, torch.int32), Pointer(sorig, torch.int32), self.used, self.rank, sc['rslot'],
            sc['slot_list'], sc['slot_exp'], self.owner, m * H, nk, k, self.S, bool(act),
            triton.next_power_of_2(S), BLOCK)

    def decode(self, stream, m, k, x, ids, routing, out, act=0):
        """One decode call of up to 8 tokens on the current stream (device addresses; out: FP32 [m, H])."""
        sc = self.scratch
        if self.full:   # every expert on the GPU: no CPU part
            self._route(m, k, x, ids, routing, act, (sc['sx'].data_ptr(), sc['sids'].data_ptr(),
                                                    sc['sroute'].data_ptr(), sc['sorig'].data_ptr()))
            self._kernels(m, k, x, routing, Pointer(out, torch.float32))
            return
        if not self._pool_on:
            if self.mailbox.lib.xtier_mailbox_pool_enable(self.mailbox.bridge, 1) < 0:
                raise RuntimeError(self.mailbox.error())
            self._pool_on = True
        sx, sids, sroute, sout, sorig = self.mailbox.slot(stream, m, k, with_routes=True)
        self._route(m, k, x, ids, routing, act, (sx, sids, sroute, sorig))
        if self.miss is not None:   # before the CPU reads the routes
            _miss_select[(1,)](Pointer(sids, torch.int32), sc['rslot'], _miss['e'], _miss['r'], m * k, _miss['share'],
                               _miss['cap'], triton.next_power_of_2(m * k), _miss['cp'])
        self.mailbox.signal(stream, m, k, 1)
        self._kernels(m, k, x, routing, sc['gout'])
        self.mailbox.signal(stream, m, k, 2)
        _unpack[(triton.cdiv(m * self.H, BLOCK),)](Pointer(sout, torch.float32), sc['gout'], Pointer(out, torch.float32),
                                                   m * self.H, BLOCK)


def started():
    """The adaptive policy runs (its swaps fill the elastic slots)."""
    return bool(_glob.get('started'))


def commit_now():
    """The commit kernel on the current stream, outside a decode window (cache/elastic.py before releasing slots)."""
    E = _glob['E']
    _commit[(1,)](Pointer(_glob['ring'], torch.int32), Pointer(_glob['seqs'], torch.int32), _glob['state'],
                  _glob['table'], _glob['owner'], E, RING)


def maybe_start():
    """At any MoE call outside capture: start the adaptive policy once the decode graphs have been captured, so its
    copies never overlap a capture."""
    a = _glob.get('adapt')
    if not a or _glob.get('started') or not _glob.get('captured') or torch.cuda.is_current_stream_capturing():
        return
    lib, every, budget, decay, gain = a
    if lib.xtier_mailbox_pool_start(every, budget, C.c_float(decay), C.c_float(gain)) < 0:
        raise RuntimeError(lib.xtier_mailbox_error().decode())
    _glob['started'] = True
    logger.info('xtier cache: adaptive residency started: every %d windows, up to %d swaps per window, decay %s, '
                'min gain %s', every, budget, decay, gain)


def record_prompt(pool, ids):
    """Prefill of a partly cached layer (XTIER_CACHE_PROMPT=1): count its routes on the GPU (int32 [m, k] on the
    current stream; -1 lands in a spare column). The last partly cached layer of a forward hands the counts to the
    policy through pinned memory and a host function on the stream (no synchronization) and clears them; with
    DeepSeek V4's decoder replay the layers past 20 run on a prompt's last chunk only, so a long prompt hands over
    once, with every chunk's counts of the earlier layers."""
    p = _glob.get('prompt')
    if p is None or pool.index is None:
        return
    flat = ids.reshape(-1)
    n = flat.numel()
    if p['ones'] is None or p['ones'].numel() < n:
        p['ones'] = torch.ones(max(n, 1 << 17), dtype=torch.float32, device=ids.device)
    row = p['counts'][pool.index]
    row.index_add_(0, torch.where(flat >= 0, flat, p['stride'] - 1), p['ones'][:n])
    if pool.index == p['rows'] - 1:
        p['host'].copy_(p['counts'], non_blocking=True)
        lib = p['lib']
        if lib.xtier_mailbox_pool_prompt(C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                         C.c_void_p(p['host'].data_ptr()), p['rows'], p['stride']) < 0:
            raise RuntimeError(lib.xtier_mailbox_error().decode())
        p['counts'].zero_()


def fill_all():
    """Outside capture, after the shards are pinned: load the slots of every layer not filled yet (profile order, or
    every expert). The partly cached layers of the first call get the adaptive policy (a speculative draft, loaded
    later, is fully cached)."""
    todo = [p for p in LayerPool.layers if not p.filled]
    if not todo:
        return
    t0 = time.time()
    stream = torch.cuda.Stream()
    n = 0
    for pool in todo:
        n += pool.fill(Profile.ranking(pool.layer_id, pool.E), stream)
    partial = [p for p in todo if not p.full]
    if partial and _glob.get('prepared'):
        logger.warning('xtier cache: %d partly cached layers added after the adaptive policy started keep their '
                       'initial residency', len(partial))
        partial = []
    if partial:
        _glob['prepared'] = True
    if os.environ.get('XTIER_CACHE_ADAPT', '1') != '0' and partial:
        every, budget = int(os.environ.get('XTIER_CACHE_EVERY', '4')), int(os.environ.get('XTIER_CACHE_SWAPS', '16'))
        decay, gain = float(os.environ.get('XTIER_CACHE_DECAY', '0.7')), float(os.environ.get('XTIER_CACHE_MIN_GAIN', '1.5'))
        lib = partial[0].mailbox.lib
        for pool in partial:
            arena = (C.c_void_p * 4)(*[a.data_ptr() for a in pool.arena])
            res = (C.c_int32 * pool.S)(*pool.resident)
            if lib.xtier_mailbox_pool_attach(pool.mailbox.bridge, pool.index, pool.S, pool.active, pool.E,
                                             C.cast(arena, C.c_void_p), C.cast(res, C.c_void_p)) < 0:
                raise RuntimeError(pool.mailbox.error())
        _glob['adapt'] = (lib, every, budget, decay, gain)
        stretch = [p for p in partial if p.S > p.base]
        if stretch:
            from xtier.cache import elastic
            elastic.register(stretch, lib, commit_now)
        if os.environ.get('XTIER_CACHE_PROMPT', '0') == '1':
            keep = float(os.environ.get('XTIER_CACHE_PROMPT_KEEP', '0.7'))
            windows = float(os.environ.get('XTIER_CACHE_PROMPT_WINDOWS', '10'))
            rounds = int(os.environ.get('XTIER_CACHE_PROMPT_ROUNDS', '4'))
            swaps = int(os.environ.get('XTIER_CACHE_PROMPT_SWAPS', '256'))
            if lib.xtier_mailbox_pool_warm(C.c_float(keep), C.c_float(windows), rounds, swaps) < 0:
                raise RuntimeError(lib.xtier_mailbox_error().decode())
            rows, stride = max(p.index for p in partial) + 1, max(p.E for p in partial) + 1
            dev = partial[0].arena[0].device
            _glob['prompt'] = dict(lib=lib, rows=rows, stride=stride, ones=None,
                                   counts=torch.zeros((rows, stride), dtype=torch.float32, device=dev),
                                   host=torch.zeros((rows, stride), dtype=torch.float32, pin_memory=True))
            logger.info('xtier cache: prompt warm-up on: a prompt\'s routes as %s decode windows (old counts x %s), '
                        'then %d rounds of up to %d swaps', windows, keep, rounds, swaps)
    gib = sum(p.bytes() for p in todo) / 2**30
    full = [p.layer_id for p in todo if p.full]
    logger.info('xtier cache: %d layers (%d with every expert: %s), %.1f GiB, %d experts loaded from %s in %.1f s',
                len(todo), len(full), full, gib, n, os.environ.get('XTIER_CACHE_PROFILE') or '(no profile)',
                time.time() - t0)
