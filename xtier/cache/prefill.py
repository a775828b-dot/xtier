"""Prefill of a layer whose experts are all in the cache, on the packed slots in place: no staging copy and no BF16
unpack of the whole layer (5 GB of writes per Qwen3.8 layer).

Routes are sorted by expert and cut into tiles of up to BM routes of one expert (as in backend/prefill.py); each tile
runs W4A16 gate/up and down with the cache's decode arithmetic (hardware FP4 -> FP16 conversion of two nibbles at a
time, block scale applied in FP16, FP32 accumulation, BF16 intermediate; cache/pool.py ``_halves``), then the
per-token sum of the routes in route order (FP32). The software FP4 decode inside the prefill GEMM
(``UNPACKED=False`` there) reaches about 90 GB/s; this one runs at the bandwidth of the decode kernels.
No kernel argument depends on the number of tokens at compile time, so a new prompt length compiles nothing.
"""
import os

import torch
import triton
import triton.language as tl

from xtier.backend.prefill import _reduce
from xtier.cache.pool import _halves, _tile_k


@triton.jit
def _gate_up_tiles(X, ORDER, TILE_E, TILE_S, TILE_START, TILE_END, Q13, S13, G13, MID, K: tl.constexpr, H: tl.constexpr,
                   I: tl.constexpr, LIMIT, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MX: tl.constexpr,
                   ALPHA: tl.constexpr = 1.0, BETA: tl.constexpr = 0.0):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)
    s = tl.load(TILE_S + tile).to(tl.int64)   # slot of the tile's expert (expert e for a fully cached layer)
    r = tl.load(TILE_START + tile) + tl.arange(0, BM)
    rm = r < tl.load(TILE_END + tile)
    tok = (tl.load(ORDER + r, rm, 0) // K).to(tl.int64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    G: tl.constexpr = 32 if MX else 16
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        ke = k0 + 2 * tl.arange(0, BK // 2)
        xe = tl.load(X + tok[:, None] * H + ke[None, :], mask=rm[:, None], other=0.0).to(tl.bfloat16 if MX else tl.float16)
        xo = tl.load(X + tok[:, None] * H + ke[None, :] + 1, mask=rm[:, None], other=0.0).to(tl.bfloat16 if MX else tl.float16)
        gl, gh = _halves(Q13, S13, s, n, k0, 2 * I * (H // 2), H // 2, 2 * I * (H // G), H // G, BN, BK, MX)
        ul, uh = _halves(Q13, S13, s, n + I, k0, 2 * I * (H // 2), H // 2, 2 * I * (H // G), H // G, BN, BK, MX)
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
    tl.store(MID + r[:, None].to(tl.int64) * I + n[None, :], val.to(tl.bfloat16), mask=rm[:, None])


@triton.jit
def _down_tiles(MID, ORDER, TILE_E, TILE_S, TILE_START, TILE_END, Q2, S2, G2, ROUTES, H: tl.constexpr, I: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MX: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)
    s = tl.load(TILE_S + tile).to(tl.int64)   # slot of the tile's expert (expert e for a fully cached layer)
    r = tl.load(TILE_START + tile) + tl.arange(0, BM)
    rm = r < tl.load(TILE_END + tile)
    route = tl.load(ORDER + r, rm, 0).to(tl.int64)
    h = tl.program_id(1) * BN + tl.arange(0, BN)
    G: tl.constexpr = 32 if MX else 16
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        ke = k0 + 2 * tl.arange(0, BK // 2)
        ae = tl.load(MID + r[:, None].to(tl.int64) * I + ke[None, :], mask=rm[:, None], other=0.0).to(tl.bfloat16 if MX else tl.float16)
        ao = tl.load(MID + r[:, None].to(tl.int64) * I + ke[None, :] + 1, mask=rm[:, None], other=0.0).to(tl.bfloat16 if MX else tl.float16)
        wl, wh = _halves(Q2, S2, s, h, k0, H * (I // 2), I // 2, H * (I // G), I // G, BN, BK, MX)
        acc = tl.dot(ae, tl.trans(wl), acc)
        acc = tl.dot(ao, tl.trans(wh), acc)
    acc = acc * tl.load(G2 + e)
    tl.store(ROUTES + route[:, None] * H + h[None, :], acc, mask=rm[:, None])


def _env_int(name, default):
    return int(os.environ.get(name, default))


def resident_prefill(prefill, arena, x, ids, weights):
    """x: BF16 [m, H], ids: int32 [m, k], weights: FP32 [m, k] (device tensors, current stream) -> BF16 [m, H].
    arena: the layer's slots [w13, w2, s13, s2], slot e = expert e."""
    H, I, E = prefill.hidden, prefill.inter, prefill.experts
    g13, g2 = prefill._globals()
    fixed_bm = _env_int('XTIER_CACHE_PREFILL_TILE_M', 0)
    bkh, bki = _tile_k(H), _tile_k(I)
    block = _env_int('XTIER_GPU_PREFILL_CHUNK', 8192)
    m, k = ids.shape
    out = torch.empty_like(x)
    for start in range(0, m, block):
        size = min(block, m - start)
        # tiles of 32 routes while experts get few routes, 64 once they average 48 or more (Qwen3.8: from ~2.5K
        # tokens): fewer FP4 decodes per weight and better MMA use; 128 spills registers
        bm = fixed_bm or (64 if size * k >= 48 * E else 32)
        # launch shapes per regime (Qwen3.8 layer, 2026-10-23 sweep; outputs bitwise unchanged): few routes per expert
        # favour wide gate tiles, many routes deeper gate pipelines and wide down tiles
        small = bm == 32
        gbn = _env_int('XTIER_CACHE_PREFILL_GATE_BN', 64 if small else 32)
        gw = _env_int('XTIER_CACHE_PREFILL_GATE_WARPS', 8 if small else 4)
        gs = _env_int('XTIER_CACHE_PREFILL_GATE_STAGES', 2 if small else 3)
        dbn = _env_int('XTIER_CACHE_PREFILL_DOWN_BN', 64 if small else 128)
        dw = _env_int('XTIER_CACHE_PREFILL_DOWN_WARPS', 4 if small else 8)
        ds = _env_int('XTIER_CACHE_PREFILL_DOWN_STAGES', 2)
        di = ids[start:start + size].reshape(-1)
        counts = torch.bincount(di, minlength=E)
        host = torch.cat([counts, di.min()[None], di.max()[None]]).cpu().tolist()
        if host[E] < 0 or host[E + 1] >= E:
            raise ValueError('xtier: invalid expert id in a fully cached prefill')
        order = torch.argsort(di, stable=True).to(torch.int32)
        tiles, offset = [], 0
        for e in range(E):
            c = host[e]
            for q in range(0, c, bm):
                tiles.append((e, offset + q, offset + c))
            offset += c
        meta = torch.tensor(tiles, dtype=torch.int32).T.contiguous().to(x.device, non_blocking=True)
        mid = torch.empty((size * k, I), dtype=torch.bfloat16, device=x.device)
        routes = torch.empty((size * k, H), dtype=torch.float32, device=x.device)
        nt = len(tiles)
        _gate_up_tiles[(nt, I // gbn)](x[start:start + size], order, meta[0], meta[0], meta[1], meta[2], arena[0], arena[2], g13,
                                       mid, k, H, I, prefill.swiglu_limit, bm, gbn, bkh, MX=prefill.mxfp4,
                                       ALPHA=prefill.swiglu_alpha, BETA=prefill.swiglu_beta,
                                       num_warps=gw, num_stages=gs)
        _down_tiles[(nt, H // dbn)](mid, order, meta[0], meta[0], meta[1], meta[2], arena[1], arena[3], g2, routes, H, I, bm,
                                    dbn, bki, MX=prefill.mxfp4, num_warps=dw, num_stages=ds)
        # M of _reduce is unused by the kernel body: a fixed value keeps one compiled variant for every length
        _reduce[(size, triton.cdiv(H, 256))](routes, weights[start:start + size], out[start:start + size], 1, k,
                                             256, H)
    return out


def hybrid_prefill(pool, plan, x, ids, weights, cpu_batch=1024):
    """Prefill of a partly cached layer: routes to cached experts on the GPU (the tile kernels on the slots, W4A16),
    the others at the same time on the CPU engine (FP32); the two partial sums are added in FP32.
    x: BF16 [m, H], ids: int32 [m, k], weights: FP32 [m, k] (device tensors, current stream) -> FP32 [m, H].
    The residency is read once at the start: swaps are applied only by decode windows, which do not run meanwhile."""
    import threading

    H, I, S = pool.H, pool.I, pool.S
    prefill = pool.prefill
    g13, g2 = prefill._globals()
    m, k = ids.shape
    slot = pool.table[ids.long()]                       # [m, k]: slot of each route, -1 = on the CPU
    on_gpu = slot >= 0
    # CPU part: inputs staged in pinned memory, GPU routes as id -1 (the engine skips them)
    x_h = torch.empty((m, H), dtype=torch.bfloat16, pin_memory=True)
    ids_h = torch.empty((m, k), dtype=torch.int32, pin_memory=True)
    w_h = torch.empty((m, k), dtype=torch.float32, pin_memory=True)
    out_h = torch.empty((m, H), dtype=torch.float32, pin_memory=True)
    x_h.copy_(x, non_blocking=True)
    ids_h.copy_(torch.where(on_gpu, -1, ids).to(torch.int32), non_blocking=True)
    w_h.copy_(weights, non_blocking=True)
    # GPU part: routes sorted by slot (CPU routes last, no tile), one host rendezvous for the tile metadata
    key = torch.where(on_gpu, slot, S).reshape(-1)
    order = torch.argsort(key, stable=True).to(torch.int32)
    counts = torch.bincount(key, minlength=S + 1)[:S]
    host = torch.cat([counts, pool.owner[:S].to(counts.dtype)]).cpu().tolist()   # also completes the staging copies
    errors = []

    def cpu_part():
        try:
            for s0 in range(0, m, cpu_batch):
                n = min(cpu_batch, m - s0)
                plan.forward(n, k, x_h.data_ptr() + s0 * H * 2, ids_h.data_ptr() + s0 * k * 4,
                             w_h.data_ptr() + s0 * k * 4, out_h.data_ptr() + s0 * H * 4)
        except BaseException as e:   # re-raised by the caller after the join
            errors.append(e)

    worker = threading.Thread(target=cpu_part, name='xtier-hybrid-prefill')
    worker.start()
    counts, owner = host[:S], host[S:]
    bm = 64 if sum(counts) >= 48 * max(1, sum(1 for c in counts if c)) else 32
    small = bm == 32
    gbn, gw, gs = (64, 8, 2) if small else (32, 4, 3)
    dbn, dw = (64, 4) if small else (128, 8)
    tiles, offset = [], 0
    for s in range(S):
        c = counts[s]
        for q in range(0, c, bm):
            tiles.append((owner[s], s, offset + q, offset + c))
        offset += c
    out_g = torch.zeros((m, H), dtype=torch.float32, device=x.device)
    if tiles:
        meta = torch.tensor(tiles, dtype=torch.int32).T.contiguous().to(x.device, non_blocking=True)
        mid = torch.empty((max(offset, 1), I), dtype=torch.bfloat16, device=x.device)
        routes = torch.zeros((m * k, H), dtype=torch.float32, device=x.device)   # CPU routes stay 0 for the sum
        nt = len(tiles)
        _gate_up_tiles[(nt, I // gbn)](x, order, meta[0], meta[1], meta[2], meta[3], pool.arena[0], pool.arena[2], g13,
                                       mid, k, H, I, prefill.swiglu_limit, bm, gbn, _tile_k(H), MX=prefill.mxfp4,
                                       ALPHA=prefill.swiglu_alpha, BETA=prefill.swiglu_beta,
                                       num_warps=gw, num_stages=gs)
        _down_tiles[(nt, H // dbn)](mid, order, meta[0], meta[1], meta[2], meta[3], pool.arena[1], pool.arena[3], g2,
                                    routes, H, I, bm, dbn, _tile_k(I), MX=prefill.mxfp4, num_warps=dw, num_stages=2)
        _reduce[(m, triton.cdiv(H, 256))](routes, torch.where(on_gpu, weights, 0.0), out_g, 1, k, 256, H)
    worker.join()
    if errors:
        raise errors[0]
    return out_g + out_h.to(x.device, non_blocking=True)
