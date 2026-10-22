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
def _gate_up_tiles(X, ORDER, TILE_E, TILE_START, TILE_END, Q13, S13, G13, MID, K: tl.constexpr, H: tl.constexpr,
                   I: tl.constexpr, LIMIT, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MX: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)
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
        gl, gh = _halves(Q13, S13, e, n, k0, 2 * I * (H // 2), H // 2, 2 * I * (H // G), H // G, BN, BK, MX)
        ul, uh = _halves(Q13, S13, e, n + I, k0, 2 * I * (H // 2), H // 2, 2 * I * (H // G), H // G, BN, BK, MX)
        accg = tl.dot(xe, tl.trans(gl), accg)
        accg = tl.dot(xo, tl.trans(gh), accg)
        accu = tl.dot(xe, tl.trans(ul), accu)
        accu = tl.dot(xo, tl.trans(uh), accu)
    g = accg * tl.load(G13 + e * 2)
    u = accu * tl.load(G13 + e * 2 + 1)
    if LIMIT > 0:
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        g = tl.minimum(g, LIMIT)
    val = g / (1.0 + tl.exp(-g)) * u
    tl.store(MID + r[:, None].to(tl.int64) * I + n[None, :], val.to(tl.bfloat16), mask=rm[:, None])


@triton.jit
def _down_tiles(MID, ORDER, TILE_E, TILE_START, TILE_END, Q2, S2, G2, ROUTES, H: tl.constexpr, I: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MX: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)
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
        wl, wh = _halves(Q2, S2, e, h, k0, H * (I // 2), I // 2, H * (I // G), I // G, BN, BK, MX)
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
    gbn, dbn = _env_int('XTIER_CACHE_PREFILL_GATE_BN', 32), _env_int('XTIER_CACHE_PREFILL_DOWN_BN', 64)
    gw, dw = _env_int('XTIER_CACHE_PREFILL_GATE_WARPS', 4), _env_int('XTIER_CACHE_PREFILL_DOWN_WARPS', 4)
    bkh, bki = _tile_k(H), _tile_k(I)
    block = _env_int('XTIER_GPU_PREFILL_CHUNK', 8192)
    m, k = ids.shape
    out = torch.empty_like(x)
    for start in range(0, m, block):
        size = min(block, m - start)
        # tiles of 32 routes while experts get few routes, 64 once they average 48 or more (Qwen3.8: from ~2.5K
        # tokens): fewer FP4 decodes per weight and better MMA use; 128 spills registers
        bm = fixed_bm or (64 if size * k >= 48 * E else 32)
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
        _gate_up_tiles[(nt, I // gbn)](x[start:start + size], order, meta[0], meta[1], meta[2], arena[0], arena[2], g13,
                                       mid, k, H, I, prefill.swiglu_limit, bm, gbn, bkh, MX=prefill.mxfp4,
                                       num_warps=gw, num_stages=2)
        _down_tiles[(nt, H // dbn)](mid, order, meta[0], meta[1], meta[2], arena[1], arena[3], g2, routes, H, I, bm,
                                    dbn, bki, MX=prefill.mxfp4, num_warps=dw, num_stages=2)
        # M of _reduce is unused by the kernel body: a fixed value keeps one compiled variant for every length
        _reduce[(size, triton.cdiv(H, 256))](routes, weights[start:start + size], out[start:start + size], 1, k,
                                             256, H)
    return out
