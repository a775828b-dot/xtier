"""Native FP8 tensor-core IndexPool logits for Hunyuan-4-Flash on SM89+ (``XTIER_HY4_INDEXPOOL_FP8=1``).

Upstream ``hy4_flash_indexpool.py`` casts both ``q_fp8`` and pooled ``k_fp8`` from ``float8_e4m3fn``
to ``bfloat16`` before ``tl.dot(q, tl.trans(k))``. Every E4M3 value is exact in BF16, so on SM89+
with native FP8 tensor cores (``HW_FP8``) keeping both operands in ``float8e4nv`` computes the same
FP32 inner products at 2x tensor-core throughput and half the register/SMEM footprint.
"""

import os

import torch
import triton
import triton.language as tl

from xtier.arch import HW_FP8

TARGET = 'sglang.kernels.ops.attention.dsa.hy4_flash_indexpool.indexpool_logits'


@triton.jit
def _logits_fp8_kernel(
    q_ptr,  # [R, H, 128] uint8 view of float8_e4m3fn
    w_ptr,  # [R, H] fp32
    pages_ptr,  # [B, NP] int32
    row_batch_ptr,  # [R] int64
    pos_ptr,  # [R] int64
    buf_ptr,
    buf_page_stride,
    out_ptr,  # [R, G] fp32
    pages_stride,
    num_pages,
    G,
    H: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    groups = gb * BLOCK_G + tl.arange(0, BLOCK_G)
    pos = tl.load(pos_ptr + r)
    upper = tl.where(pos >= 0, pos // RATIO, 0)
    in_range = groups < G
    if gb * BLOCK_G >= upper:
        tl.store(out_ptr + r * G + groups, -float('inf'), mask=in_range)
        return
    live = groups < upper
    b = tl.load(row_batch_ptr + r)
    tok = groups * RATIO
    page_col = tl.minimum(tok // 64, num_pages - 1)
    page = tl.load(pages_ptr + b * pages_stride + page_col, mask=live, other=0).to(tl.int64)
    slot = page * 64 + tok % 64
    kpage = slot // PAGE
    krow = slot % PAGE
    d = tl.arange(0, 128)
    kbase = buf_ptr + kpage * buf_page_stride
    k = tl.load(kbase[:, None] + krow[:, None] * 128 + d[None, :], mask=live[:, None], other=0).to(
        tl.float8e4nv, bitcast=True
    )
    ks = tl.load((kbase + PAGE * 128 + krow * 4).to(tl.pointer_type(tl.float32)), mask=live, other=0.0)
    hs = tl.arange(0, H)
    q = tl.load(q_ptr + (r * H + hs[:, None]) * 128 + d[None, :]).to(tl.float8e4nv, bitcast=True)
    w = tl.load(w_ptr + r * H + hs)
    s = tl.dot(q, tl.trans(k))
    s = tl.maximum(s, 0.0) * w[:, None]
    logits = tl.sum(s, axis=0) * ks
    logits = tl.where(live, logits, -float('inf'))
    tl.store(out_ptr + r * G + groups, logits, mask=in_range)


@triton.jit
def _logits_rows_fp8_kernel(
    q_ptr,  # [R, H, 128] uint8 view of float8_e4m3fn
    w_ptr,  # [R, H] fp32
    pages_ptr,  # [B, NP] int32
    row_batch_ptr,  # [R] int64
    pos_ptr,  # [R] int64
    buf_ptr,
    buf_page_stride,
    out_ptr,  # [R, G] fp32
    pages_stride,
    num_pages,
    R,
    G,
    H: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    rb = tl.program_id(0)
    gb = tl.program_id(1)
    rows = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < R
    last = tl.minimum(rb * BLOCK_R + BLOCK_R, R) - 1
    groups = gb * BLOCK_G + tl.arange(0, BLOCK_G)
    in_range = groups < G
    pos = tl.load(pos_ptr + rows, mask=row_mask, other=-1)
    upper = tl.where(pos >= 0, pos // RATIO, 0)
    out_offs = rows[:, None].to(tl.int64) * G + groups[None, :]
    out_mask = row_mask[:, None] & in_range[None, :]
    if gb * BLOCK_G >= tl.max(upper, axis=0):
        tl.store(out_ptr + out_offs, -float('inf'), mask=out_mask)
        return
    b_first = tl.load(row_batch_ptr + rb * BLOCK_R)
    b_last = tl.load(row_batch_ptr + last)
    d = tl.arange(0, 128)
    hs = tl.arange(0, H)
    if b_first == b_last:
        live_g = groups < tl.max(upper, axis=0)
        tok = groups * RATIO
        page_col = tl.minimum(tok // 64, num_pages - 1)
        page = tl.load(pages_ptr + b_first * pages_stride + page_col, mask=live_g, other=0).to(tl.int64)
        slot = page * 64 + tok % 64
        kpage = slot // PAGE
        krow = slot % PAGE
        kbase = buf_ptr + kpage * buf_page_stride
        k = tl.load(kbase[:, None] + krow[:, None] * 128 + d[None, :], mask=live_g[:, None], other=0).to(
            tl.float8e4nv, bitcast=True
        )
        ks = tl.load((kbase + PAGE * 128 + krow * 4).to(tl.pointer_type(tl.float32)), mask=live_g, other=0.0)
        qr = (rows[:, None] * H + hs[None, :]).reshape(BLOCK_R * H)
        qm = tl.broadcast_to(row_mask[:, None], (BLOCK_R, H)).reshape(BLOCK_R * H)
        q = tl.load(q_ptr + qr[:, None] * 128 + d[None, :], mask=qm[:, None], other=0).to(
            tl.float8e4nv, bitcast=True
        )
        w = tl.load(w_ptr + qr, mask=qm, other=0.0)
        s = tl.dot(q, tl.trans(k))
        s = tl.maximum(s, 0.0) * w[:, None]
        logits = tl.sum(s.reshape(BLOCK_R, H, BLOCK_G), axis=1) * ks[None, :]
        logits = tl.where(groups[None, :] < upper[:, None], logits, -float('inf'))
        tl.store(out_ptr + out_offs, logits, mask=out_mask)
    else:
        for i in tl.static_range(BLOCK_R):
            r = rb * BLOCK_R + i
            if r < R:
                p = tl.load(pos_ptr + r)
                up = tl.where(p >= 0, p // RATIO, 0)
                live = groups < up
                b = tl.load(row_batch_ptr + r)
                tok = groups * RATIO
                page = tl.load(
                    pages_ptr + b * pages_stride + tl.minimum(tok // 64, num_pages - 1), mask=live, other=0
                ).to(tl.int64)
                slot = page * 64 + tok % 64
                kbase = buf_ptr + (slot // PAGE) * buf_page_stride
                krow = slot % PAGE
                k = tl.load(kbase[:, None] + krow[:, None] * 128 + d[None, :], mask=live[:, None], other=0).to(
                    tl.float8e4nv, bitcast=True
                )
                ks = tl.load((kbase + PAGE * 128 + krow * 4).to(tl.pointer_type(tl.float32)), mask=live, other=0.0)
                q = tl.load(q_ptr + (r * H + hs[:, None]) * 128 + d[None, :]).to(tl.float8e4nv, bitcast=True)
                w = tl.load(w_ptr + r * H + hs)
                s = tl.dot(q, tl.trans(k))
                s = tl.maximum(s, 0.0) * w[:, None]
                logits = tl.sum(s, axis=0) * ks
                logits = tl.where(live, logits, -float('inf'))
                tl.store(out_ptr + r.to(tl.int64) * G + groups, logits, mask=in_range)


def _indexpool_logits(original, q_fp8, weights, pages, row_to_batch, positions, index_buf, page_size, num_groups):
    rows, heads, _ = q_fp8.shape
    if not HW_FP8 or heads < 16 or q_fp8.dtype != torch.float8_e4m3fn or rows == 0 or num_groups == 0:
        return original(q_fp8, weights, pages, row_to_batch, positions, index_buf, page_size, num_groups)
    out = torch.empty((rows, num_groups), dtype=torch.float32, device=q_fp8.device)
    q_u8 = q_fp8.contiguous().view(torch.uint8)
    if rows >= 64:
        block_r, block_g, warps = 4, 256, 4
        _logits_rows_fp8_kernel[(triton.cdiv(rows, block_r), triton.cdiv(num_groups, block_g))](
            q_u8, weights, pages, row_to_batch, positions, index_buf, index_buf.stride(0), out,
            pages.stride(0), pages.shape[1], rows, num_groups,
            H=heads, RATIO=4, PAGE=page_size, BLOCK_R=block_r, BLOCK_G=block_g, num_warps=warps,
        )
        return out
    block_g = 128
    _logits_fp8_kernel[(rows, triton.cdiv(num_groups, block_g))](
        q_u8, weights, pages, row_to_batch, positions, index_buf, index_buf.stride(0), out,
        pages.stride(0), pages.shape[1], num_groups,
        H=heads, RATIO=4, PAGE=page_size, BLOCK_G=block_g, num_warps=4,
    )
    return out


def install(registry, hook_type):
    if HW_FP8 and os.environ.get('XTIER_HY4_INDEXPOOL_FP8', '1') != '0':
        registry.register(TARGET, _indexpool_logits, hook_type.AROUND)
