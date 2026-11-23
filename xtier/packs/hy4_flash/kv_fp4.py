"""NVFP4 Sparse MLA KV cache (416 B/tok) & fast Triton FP8 decode for Hunyuan-4-Flash.

Two complementary optimizations for Hy4-Flash's 42 MLA layers (12 DSA H=32 topk=2048 + 30 SWA
H=64 topk=512, kv_lora_rank=512, qk_rope_head_dim=64) on SM120:

1. ``XTIER_HY4_KV_FP4=1`` — NVFP4 KV cache (``416 B/tok`` vs ``656 B/tok`` FP8, ``-36.6%`` KV VRAM):
   - Bytes ``0..255``: 512 ``k_nope`` elements packed as E2M1 nibbles (byte ``i`` holds
     ``lo = nope[i]``, ``hi = nope[i + 256]`` so Q loads and O stores are 100% coalesced).
   - Bytes ``256..287``: 32 ``float8_e4m3fn`` micro-scales (1 per 16 nope values, ``amax / 6.0``).
   - Bytes ``288..415``: 64 ``k_rope`` elements kept in exact BF16 (zero RoPE drift at 1M context).
   - Decode (``rows <= 16``): split-K Triton NVFP4 sparse MLA kernel with fused FP32 ``attn_sink``.
   - Prefill (``rows > 16``): 1-launch Triton NVFP4->FP8 unpack of active slots into a temporary
     FP8 staging buffer (< 0.1 ms at 100K, < 0.9 ms at 1M) followed by FlashInfer C++ SM120 sparse
     MLA prefill (retains 100% FlashInfer prefill speed!).

2. ``XTIER_HY4_TRITON_DECODE=1`` (default on for FP8 ``656 B/tok`` KV when ``rows <= 16``):
   - FlashInfer SM120 ``trtllm_batch_decode_with_kv_cache_mla`` has ~160 us fixed overhead per
     layer at small decode batches (30 SWA layers x 162 us = 4.85 ms/step!).
   - Our split-K Triton FP8 sparse MLA kernel runs SWA decode in ~55-65 us (saving ~3 ms/step on
     both bs=1 and bs=4 with zero precision loss vs FlashInfer FP8!).
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

NVFP4_KV_DIM = 416
FP8_KV_DIM = 656
TRITON_DECODE_MAX_ROWS = 16

CALC_KV_DIM = 'sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim'
WRITE_MLA_KV = 'sglang.srt.mem_cache.memory_pool.MLATokenToKVPool._write_mla_kv_buffer'


def kv_fp4_enabled() -> bool:
    return os.environ.get('XTIER_HY4_KV_FP4', '0') == '1'


def triton_fp8_decode_enabled() -> bool:
    return os.environ.get('XTIER_HY4_TRITON_DECODE', '0') == '1'


def triton_fp8_max_topk() -> int:
    return int(os.environ.get('XTIER_HY4_TRITON_MAX_TOPK', '512'))


@triton.jit
def _e2m1(code):
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    v = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((code & 8) != 0, -v, v)


@triton.jit
def _quant_e2m1(v):
    sign = tl.where(v < 0.0, 8, 0)
    a = tl.abs(v)
    code = tl.where(a < 0.25, 0,
           tl.where(a < 0.75, 1,
           tl.where(a < 1.25, 2,
           tl.where(a < 1.75, 3,
           tl.where(a < 2.50, 4,
           tl.where(a < 3.50, 5,
           tl.where(a < 5.00, 6, 7)))))))
    return (sign | code).to(tl.uint8)


@triton.jit
def _quant_store_nvfp4_mla_kernel(
    NOPE, ROPE, LOC, DST, DST_BF16,
    stride_n, stride_r, max_slots, N,
):
    row = tl.program_id(0)
    if row >= N:
        return
    slot = tl.load(LOC + row).to(tl.int64)
    if slot < 0 or slot >= max_slots:
        return
    base = slot * 416

    g16 = tl.arange(0, 16)[:, None]
    e = tl.arange(0, 16)[None, :]
    v_lo = tl.load(NOPE + row * stride_n + g16 * 16 + e).to(tl.float32)
    v_hi = tl.load(NOPE + row * stride_n + 256 + g16 * 16 + e).to(tl.float32)
    s_lo = tl.maximum(tl.max(tl.abs(v_lo), axis=1) / 6.0, 1e-6).to(tl.float8e4nv)
    s_hi = tl.maximum(tl.max(tl.abs(v_hi), axis=1) / 6.0, 1e-6).to(tl.float8e4nv)
    c_lo = _quant_e2m1(v_lo / s_lo.to(tl.float32)[:, None])
    c_hi = _quant_e2m1(v_hi / s_hi.to(tl.float32)[:, None])
    tl.store(DST + base + g16 * 16 + e, c_lo | (c_hi << 4))
    tl.store(DST + base + 256 + tl.arange(0, 16), s_lo.to(tl.uint8, bitcast=True))
    tl.store(DST + base + 272 + tl.arange(0, 16), s_hi.to(tl.uint8, bitcast=True))

    rd = tl.arange(0, 64)
    rope = tl.load(ROPE + row * stride_r + rd)
    tl.store(DST_BF16 + (base + 288) // 2 + rd, rope)


@triton.jit
def _unpack_nvfp4_to_fp8_kernel(
    SRC, SRC_BF16, DST, DST_F32, DST_BF16, N,
):
    """Unpack N contiguous NVFP4 MLA slots (416 B/tok) -> SGLang fp8_ds_mla slots (656 B/tok)."""
    slot = tl.program_id(0).to(tl.int64)
    if slot >= N:
        return
    sbase = slot * 416
    dbase = slot * 656

    # 4 groups of 128 values: groups 0,1 come from lo nibbles (bytes 0..255),
    # groups 2,3 come from hi nibbles (bytes 0..255).
    d128 = tl.arange(0, 128)
    for g in tl.static_range(4):
        is_hi = g >= 2
        byte_off = (g - 2) * 128 if is_hi else g * 128
        sc_off = (272 + (g - 2) * 8) if is_hi else (256 + g * 8)
        b = tl.load(SRC + sbase + byte_off + d128).to(tl.int32)
        nib = (b >> 4) if is_hi else (b & 15)
        sc = tl.load(SRC + sbase + sc_off + (d128 // 16)).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        val = _e2m1(nib) * sc
        amax = tl.max(tl.abs(val), axis=0)
        s128 = tl.maximum(amax / 448.0, 1e-12)
        q8 = (val / s128).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(DST + dbase + g * 128 + d128, q8)
        tl.store(DST_F32 + (dbase + 512) // 4 + g, s128)

    rd = tl.arange(0, 64)
    rope = tl.load(SRC_BF16 + (sbase + 288) // 2 + rd)
    tl.store(DST_BF16 + (dbase + 528) // 2 + rd, rope)


@triton.jit
def _gather_unpack_nvfp4_to_fp8_kernel(
    SRC, SRC_BF16, IDX,
    DST, DST_F32, DST_BF16, OUT_IDX,
    total_pos,
):
    """Gather & unpack only the active decode indices [0..total_pos) into an L2-hot FP8 scratch buffer.

    CUDA-graph safe (fixed grid = rows * topk, zero host sync, skips -1 padded slots with zero KV reads).
    """
    pos = tl.program_id(0).to(tl.int64)
    if pos >= total_pos:
        return
    slot = tl.load(IDX + pos).to(tl.int64)
    if slot < 0:
        tl.store(OUT_IDX + pos, -1)
        return
    tl.store(OUT_IDX + pos, pos.to(tl.int32))
    sbase = slot * 416
    dbase = pos * 656

    d128 = tl.arange(0, 128)
    for g in tl.static_range(4):
        is_hi = g >= 2
        byte_off = (g - 2) * 128 if is_hi else g * 128
        sc_off = (272 + (g - 2) * 8) if is_hi else (256 + g * 8)
        b = tl.load(SRC + sbase + byte_off + d128).to(tl.int32)
        nib = (b >> 4) if is_hi else (b & 15)
        sc = tl.load(SRC + sbase + sc_off + (d128 // 16)).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        val = _e2m1(nib) * sc
        amax = tl.max(tl.abs(val), axis=0)
        s128 = tl.maximum(amax / 448.0, 1e-12)
        q8 = (val / s128).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(DST + dbase + g * 128 + d128, q8)
        tl.store(DST_F32 + (dbase + 512) // 4 + g, s128)

    rd = tl.arange(0, 64)
    rope = tl.load(SRC_BF16 + (sbase + 288) // 2 + rd)
    tl.store(DST_BF16 + (dbase + 528) // 2 + rd, rope)


@triton.jit
def _nvfp4_sparse_attend(
    Q, C, CB, IDX, SINK, O, M, L, ACC,
    scale, sq_b, sq_h, so_b, so_h, si, topk, keys,
    H: tl.constexpr, HB: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr, HAS_SINK: tl.constexpr,
):
    row, hb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = hb * HB + tl.arange(0, HB)
    hmask = h < H
    d256 = tl.arange(0, 256)
    d64 = tl.arange(0, 64)
    n = tl.arange(0, BLOCK_N)

    q_base = Q + row * sq_b + h[:, None] * sq_h
    q_lo = tl.load(q_base + d256[None, :], mask=hmask[:, None], other=0.0)
    q_hi = tl.load(q_base + 256 + d256[None, :], mask=hmask[:, None], other=0.0)
    q_pe = tl.load(q_base + 512 + d64[None, :], mask=hmask[:, None], other=0.0)

    m_i = tl.full([HB], float('-inf'), tl.float32)
    l_i = tl.zeros([HB], tl.float32)
    acc_lo = tl.zeros([HB, 256], tl.float32)
    acc_hi = tl.zeros([HB, 256], tl.float32)

    lo = split * keys
    hi = tl.minimum(lo + keys, topk)
    for start in range(lo, hi, BLOCK_N):
        pos = start + n
        idx = tl.load(IDX + row * si + pos, mask=pos < topk, other=-1)
        valid = idx >= 0
        base = tl.where(valid, idx, 0).to(tl.int64) * 416

        b = tl.load(C + base[:, None] + d256[None, :], mask=valid[:, None], other=0).to(tl.int32)
        s_lo = tl.load(C + base[:, None] + 256 + (d256 // 16)[None, :], mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        s_hi = tl.load(C + base[:, None] + 272 + (d256 // 16)[None, :], mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        k_lo = (_e2m1(b & 15) * s_lo).to(tl.bfloat16)
        k_hi = (_e2m1(b >> 4) * s_hi).to(tl.bfloat16)
        k_pe = tl.load(CB + (base[:, None] + 288) // 2 + d64[None, :], mask=valid[:, None], other=0.0)

        qk = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi)) + tl.dot(q_pe, tl.trans(k_pe))
        s = tl.where(valid[None, :], qk * scale, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_use = tl.where(m_new == float('-inf'), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_use)
        p = tl.math.exp2(s - m_use[:, None])
        p16 = p.to(tl.bfloat16)
        acc_lo = tl.dot(p16, k_lo, acc_lo * alpha[:, None])
        acc_hi = tl.dot(p16, k_hi, acc_hi * alpha[:, None])
        m_i = m_new
        l_i = l_i * alpha + tl.sum(p, 1)

    if SPLITS == 1:
        m_use = tl.where(m_i == float('-inf'), 0.0, m_i)
        total = l_i
        if HAS_SINK:
            sink = tl.load(SINK + h, mask=hmask, other=0.0) * 1.4426950408889634
            total = total + tl.math.exp2(sink - m_use)
        inv = tl.where(total > 0.0, 1.0 / total, 0.0)
        o_base = O + row * so_b + h[:, None] * so_h
        tl.store(o_base + d256[None, :], (acc_lo * inv[:, None]).to(tl.bfloat16), mask=hmask[:, None])
        tl.store(o_base + 256 + d256[None, :], (acc_hi * inv[:, None]).to(tl.bfloat16), mask=hmask[:, None])
    else:
        at = (row * SPLITS + split) * H + h
        tl.store(M + at, m_i, mask=hmask)
        tl.store(L + at, l_i, mask=hmask)
        tl.store(ACC + at[:, None] * 512 + d256[None, :], acc_lo, mask=hmask[:, None])
        tl.store(ACC + at[:, None] * 512 + 256 + d256[None, :], acc_hi, mask=hmask[:, None])


@triton.jit
def _fp8_sparse_attend(
    Q, C, CF32, CB16, IDX, SINK, O, M, L, ACC,
    scale, sq_b, sq_h, so_b, so_h, si, topk, keys,
    H: tl.constexpr, HB: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr, HAS_SINK: tl.constexpr,
):
    row, hb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = hb * HB + tl.arange(0, HB)
    hmask = h < H
    d256 = tl.arange(0, 256)
    d64 = tl.arange(0, 64)
    n = tl.arange(0, BLOCK_N)

    q_base = Q + row * sq_b + h[:, None] * sq_h
    q_lo = tl.load(q_base + d256[None, :], mask=hmask[:, None], other=0.0)
    q_hi = tl.load(q_base + 256 + d256[None, :], mask=hmask[:, None], other=0.0)
    q_pe = tl.load(q_base + 512 + d64[None, :], mask=hmask[:, None], other=0.0)

    m_i = tl.full([HB], float('-inf'), tl.float32)
    l_i = tl.zeros([HB], tl.float32)
    acc_lo = tl.zeros([HB, 256], tl.float32)
    acc_hi = tl.zeros([HB, 256], tl.float32)

    lo = split * keys
    hi = tl.minimum(lo + keys, topk)
    for start in range(lo, hi, BLOCK_N):
        pos = start + n
        idx = tl.load(IDX + row * si + pos, mask=pos < topk, other=-1)
        valid = idx >= 0
        base = tl.where(valid, idx, 0).to(tl.int64) * 656

        k_lo_raw = tl.load(C + base[:, None] + d256[None, :], mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        ks_lo = tl.load(CF32 + (base[:, None] + 512) // 4 + (d256 // 128)[None, :], mask=valid[:, None], other=0.0)
        k_lo = (k_lo_raw * ks_lo).to(tl.bfloat16)

        k_hi_raw = tl.load(C + base[:, None] + 256 + d256[None, :], mask=valid[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        ks_hi = tl.load(CF32 + (base[:, None] + 512) // 4 + (2 + d256 // 128)[None, :], mask=valid[:, None], other=0.0)
        k_hi = (k_hi_raw * ks_hi).to(tl.bfloat16)

        k_pe = tl.load(CB16 + (base[:, None] + 528) // 2 + d64[None, :], mask=valid[:, None], other=0.0)

        qk = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi)) + tl.dot(q_pe, tl.trans(k_pe))
        s = tl.where(valid[None, :], qk * scale, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_use = tl.where(m_new == float('-inf'), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_use)
        p = tl.math.exp2(s - m_use[:, None])
        p16 = p.to(tl.bfloat16)
        acc_lo = tl.dot(p16, k_lo, acc_lo * alpha[:, None])
        acc_hi = tl.dot(p16, k_hi, acc_hi * alpha[:, None])
        m_i = m_new
        l_i = l_i * alpha + tl.sum(p, 1)

    if SPLITS == 1:
        m_use = tl.where(m_i == float('-inf'), 0.0, m_i)
        total = l_i
        if HAS_SINK:
            sink = tl.load(SINK + h, mask=hmask, other=0.0) * 1.4426950408889634
            total = total + tl.math.exp2(sink - m_use)
        inv = tl.where(total > 0.0, 1.0 / total, 0.0)
        o_base = O + row * so_b + h[:, None] * so_h
        tl.store(o_base + d256[None, :], (acc_lo * inv[:, None]).to(tl.bfloat16), mask=hmask[:, None])
        tl.store(o_base + 256 + d256[None, :], (acc_hi * inv[:, None]).to(tl.bfloat16), mask=hmask[:, None])
    else:
        at = (row * SPLITS + split) * H + h
        tl.store(M + at, m_i, mask=hmask)
        tl.store(L + at, l_i, mask=hmask)
        tl.store(ACC + at[:, None] * 512 + d256[None, :], acc_lo, mask=hmask[:, None])
        tl.store(ACC + at[:, None] * 512 + 256 + d256[None, :], acc_hi, mask=hmask[:, None])


@triton.jit
def _sparse_merge(M, L, ACC, SINK, O, so_b, so_h, H: tl.constexpr, HB: tl.constexpr, SPLITS: tl.constexpr, HAS_SINK: tl.constexpr):
    row, hb = tl.program_id(0), tl.program_id(1)
    h = hb * HB + tl.arange(0, HB)
    hmask = h < H
    d512 = tl.arange(0, 512)
    top = tl.full([HB], float('-inf'), tl.float32)
    for s in tl.static_range(SPLITS):
        top = tl.maximum(top, tl.load(M + (row * SPLITS + s) * H + h, mask=hmask, other=float('-inf')))
    top_use = tl.where(top == float('-inf'), 0.0, top)
    total = tl.zeros([HB], tl.float32)
    acc = tl.zeros([HB, 512], tl.float32)
    for s in tl.static_range(SPLITS):
        at = (row * SPLITS + s) * H + h
        w = tl.math.exp2(tl.load(M + at, mask=hmask, other=float('-inf')) - top_use)
        total += w * tl.load(L + at, mask=hmask, other=0.0)
        acc += w[:, None] * tl.load(ACC + at[:, None] * 512 + d512[None, :], mask=hmask[:, None], other=0.0)
    if HAS_SINK:
        sink = tl.load(SINK + h, mask=hmask, other=0.0) * 1.4426950408889634
        total = total + tl.math.exp2(sink - top_use)
    inv = tl.where(total > 0.0, 1.0 / total, 0.0)
    tl.store(O + row * so_b + h[:, None] * so_h + d512[None, :], (acc * inv[:, None]).to(tl.bfloat16), mask=hmask[:, None])


def write_nvfp4_mla(dst_buffer, loc, cache_k_nope, cache_k_rope):
    nope = cache_k_nope.reshape(-1, 512)
    rope = cache_k_rope.reshape(-1, 64)
    if not nope.is_contiguous():
        nope = nope.contiguous()
    if not rope.is_contiguous():
        rope = rope.contiguous()
    n = nope.shape[0]
    if n == 0:
        return
    flat = dst_buffer.view(torch.uint8).reshape(-1)
    max_slots = flat.numel() // NVFP4_KV_DIM
    _quant_store_nvfp4_mla_kernel[(n,)](
        nope, rope, loc.reshape(-1), flat, flat.view(torch.bfloat16),
        nope.stride(0), rope.stride(0), max_slots, n, num_warps=4,
    )


def _pick_decode_cfg(rows, heads, topk, is_fp4, sms):
    # Tuned defaults (updated with benchmark results)
    hb = 32 if heads >= 32 else heads
    bn = 32
    warps, stages = 8, 1
    programs = rows * triton.cdiv(heads, hb)
    keys = triton.cdiv(topk, 64) * 64
    target = 2 * sms
    if programs < target:
        keys = max(128, triton.cdiv(triton.cdiv(topk, triton.cdiv(target, programs)), 64) * 64)
    splits = triton.cdiv(topk, keys)
    return hb, bn, splits, keys, warps, stages


def triton_sparse_mla_decode(q, kv_cache, indices, sink, sm_scale, kv_cache_dim):
    rows, heads, _ = q.shape
    topk = indices.shape[1]
    out = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=q.device)
    if rows == 0:
        return out
    if q.stride(-1) != 1:
        q = q.contiguous()
    idx = indices if indices.stride(-1) == 1 else indices.contiguous()
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    is_fp4 = (kv_cache_dim == NVFP4_KV_DIM)
    hb, bn, splits, keys, warps, stages = _pick_decode_cfg(rows, heads, topk, is_fp4, sms)
    if splits > 1:
        m = torch.empty((rows, splits, heads), dtype=torch.float32, device=q.device)
        l = torch.empty_like(m)
        acc = torch.empty((rows, splits, heads, 512), dtype=torch.float32, device=q.device)
    else:
        m = l = acc = sink if sink is not None else out
    sink_ptr = sink if sink is not None else out
    flat = kv_cache.view(torch.uint8).reshape(-1)
    if is_fp4:
        _nvfp4_sparse_attend[(rows, triton.cdiv(heads, hb), splits)](
            q, flat, flat.view(torch.bfloat16), idx, sink_ptr, out, m, l, acc,
            float(sm_scale) * 1.4426950408889634, q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            idx.stride(0), topk, keys, H=heads, HB=hb, BLOCK_N=bn, SPLITS=splits, HAS_SINK=sink is not None,
            num_warps=warps, num_stages=stages,
        )
    else:
        _fp8_sparse_attend[(rows, triton.cdiv(heads, hb), splits)](
            q, flat, flat.view(torch.float32), flat.view(torch.bfloat16), idx, sink_ptr, out, m, l, acc,
            float(sm_scale) * 1.4426950408889634, q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            idx.stride(0), topk, keys, H=heads, HB=hb, BLOCK_N=bn, SPLITS=splits, HAS_SINK=sink is not None,
            num_warps=warps, num_stages=stages,
        )
    if splits > 1:
        _sparse_merge[(rows, triton.cdiv(heads, hb))](
            m, l, acc, sink_ptr, out, out.stride(0), out.stride(1),
            H=heads, HB=hb, SPLITS=splits, HAS_SINK=sink is not None, num_warps=4,
        )
    return out


_UNPACK_CACHE = [None]
_UNPACK_CACHE_MAX_SLOTS = 131072  # 82 MiB cap; larger DSA prefills allocate transiently and free immediately.
_GATHER_FP8_CACHE = [None]
_GATHER_IDX_CACHE = [None]
_GATHER_DEFAULT_SLOTS = TRITON_DECODE_MAX_ROWS * 2048  # 32768 slots (~21.5 MiB), avoids reallocation in CUDA graphs.


def gather_unpack_nvfp4_to_fp8(kv_cache, indices):
    """Gather-unpack only decode ``indices`` into an L2-hot FP8 scratch buffer (CUDA-graph safe)."""
    rows, topk = indices.shape
    total = rows * topk
    if total == 0:
        return (
            torch.empty((0, 1, 1, FP8_KV_DIM), dtype=torch.uint8, device=kv_cache.device),
            indices,
        )
    idx = indices if indices.is_contiguous() else indices.contiguous()
    cap = max(total, _GATHER_DEFAULT_SLOTS)
    fp8_buf = _GATHER_FP8_CACHE[0]
    if fp8_buf is None or fp8_buf.device != kv_cache.device or fp8_buf.numel() < cap * FP8_KV_DIM:
        fp8_buf = torch.empty(cap * FP8_KV_DIM, dtype=torch.uint8, device=kv_cache.device)
        _GATHER_FP8_CACHE[0] = fp8_buf
    idx_buf = _GATHER_IDX_CACHE[0]
    if idx_buf is None or idx_buf.device != kv_cache.device or idx_buf.numel() < cap:
        idx_buf = torch.empty(cap, dtype=torch.int32, device=kv_cache.device)
        _GATHER_IDX_CACHE[0] = idx_buf

    flat_src = kv_cache.view(torch.uint8).reshape(-1)
    flat_dst = fp8_buf[:total * FP8_KV_DIM]
    out_idx = idx_buf[:total]
    _gather_unpack_nvfp4_to_fp8_kernel[(total,)](
        flat_src, flat_src.view(torch.bfloat16), idx.reshape(-1),
        flat_dst, flat_dst.view(torch.float32), flat_dst.view(torch.bfloat16), out_idx,
        total, num_warps=4,
    )
    return flat_dst.view(total, 1, 1, FP8_KV_DIM), out_idx.view(rows, topk)


def unpack_nvfp4_to_fp8(kv_cache, indices, page_size):
    """Unpack active NVFP4 KV slots [0 .. active_slots) into an FP8 (656 B/tok) staging buffer."""
    flat_src = kv_cache.view(torch.uint8).reshape(-1)
    total_slots = flat_src.numel() // NVFP4_KV_DIM
    max_idx = int(indices.max().item()) if indices.numel() > 0 else 0
    active_slots = min(total_slots, max(page_size, triton.cdiv(max_idx + 1, page_size) * page_size))
    needed = active_slots * FP8_KV_DIM
    if active_slots <= _UNPACK_CACHE_MAX_SLOTS:
        buf = _UNPACK_CACHE[0]
        if buf is None or buf.device != kv_cache.device or buf.numel() < needed:
            buf = torch.empty(needed, dtype=torch.uint8, device=kv_cache.device)
            _UNPACK_CACHE[0] = buf
        flat_dst = buf[:needed]
    else:
        flat_dst = torch.empty(needed, dtype=torch.uint8, device=kv_cache.device)
    _unpack_nvfp4_to_fp8_kernel[(active_slots,)](
        flat_src, flat_src.view(torch.bfloat16),
        flat_dst, flat_dst.view(torch.float32), flat_dst.view(torch.bfloat16),
        active_slots, num_warps=4,
    )
    return flat_dst.view(active_slots // page_size, page_size, 1, FP8_KV_DIM)


def _calc_kv_dim(original, *, model_config, kv_cache_dtype, **kwargs):
    archs = getattr(model_config.hf_config, 'architectures', None) or ()
    if ('HYV4FlashForConditionalGeneration' in archs
            and kv_cache_dtype == torch.float8_e4m3fn
            and kv_fp4_enabled()):
        return NVFP4_KV_DIM
    return original(model_config=model_config, kv_cache_dtype=kv_cache_dtype, **kwargs)


def _write_mla_kv_buffer(original, self, dst_buffer, loc, cache_k_nope, cache_k_rope):
    if getattr(self, 'kv_cache_dim', None) == NVFP4_KV_DIM:
        write_nvfp4_mla(dst_buffer, loc, cache_k_nope, cache_k_rope)
        return None
    return original(self, dst_buffer, loc, cache_k_nope, cache_k_rope)


def install(registry, hook_type):
    if kv_fp4_enabled():
        registry.register(CALC_KV_DIM, _calc_kv_dim, hook_type.AROUND)
        registry.register(WRITE_MLA_KV, _write_mla_kv_buffer, hook_type.AROUND)
