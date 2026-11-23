"""Native FP8 tensor-core IndexPool logits + length-aware decode top-k for Hunyuan-4-Flash on SM89+
(``XTIER_HY4_INDEXPOOL_FP8=1``).

1. Upstream ``hy4_flash_indexpool.py`` casts both ``q_fp8`` and pooled ``k_fp8`` from ``float8_e4m3fn``
   to ``bfloat16`` before ``tl.dot(q, tl.trans(k))``. Every E4M3 value is exact in BF16, so on SM89+
   with native FP8 tensor cores (``HW_FP8``) keeping both operands in ``float8e4nv`` computes the same
   FP32 inner products at 2x tensor-core throughput and half the register/SMEM footprint.
2. Length-aware decode top-k (mirrors upstream SGLang PR #52400): during decode and DFly target
   verify, CUDA graphs size the IndexPool logits width to the full context length (262,144 groups at
   1M context). Instead of filling the padded tail with ``-inf`` and running full-width ``top_k``,
   score only each row's active groups ``[0, upper)``, record ``lengths[r] = min(upper, G)``, and call
   ``flashinfer.topk.top_k_ragged_transform(..., dsa_graph_safe=True)``.
"""

import os

import torch
import triton
import triton.language as tl

from xtier.arch import HW_FP8

TARGET = 'sglang.kernels.ops.attention.dsa.hy4_flash_indexpool.indexpool_logits'
SELECT_TARGET = 'sglang.srt.models.hunyuan_v4_flash.FlashIndexPool._select'
_DECODE_PROGRAMS_PER_ROW = 64


@triton.jit
def _logits_fp8_block(
    q_ptr,
    w_ptr,
    pages_ptr,
    row_batch_ptr,
    buf_ptr,
    buf_page_stride,
    out_ptr,
    pages_stride,
    num_pages,
    G,
    r,
    gb,
    upper,
    H: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    groups = gb * BLOCK_G + tl.arange(0, BLOCK_G)
    in_range = groups < G
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
    tl.store(out_ptr + r.to(tl.int64) * G + groups, logits, mask=in_range)


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
    len_ptr,  # [R] int32 scored groups per row (when WRITE_LENGTHS)
    H: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    WRITE_LENGTHS: tl.constexpr,
):
    r = tl.program_id(0)
    pos = tl.load(pos_ptr + r)
    upper = tl.where(pos >= 0, pos // RATIO, 0)
    if WRITE_LENGTHS:
        if tl.program_id(1) == 0:
            tl.store(len_ptr + r, tl.minimum(upper, G).to(tl.int32))
        n_blocks = tl.cdiv(tl.minimum(upper, G), BLOCK_G)
        for gb in tl.range(tl.program_id(1), n_blocks, tl.num_programs(1)):
            _logits_fp8_block(
                q_ptr, w_ptr, pages_ptr, row_batch_ptr, buf_ptr, buf_page_stride, out_ptr,
                pages_stride, num_pages, G, r, gb, upper, H, RATIO, PAGE, BLOCK_G,
            )
    else:
        gb = tl.program_id(1)
        if gb * BLOCK_G >= upper:
            groups = gb * BLOCK_G + tl.arange(0, BLOCK_G)
            tl.store(out_ptr + r.to(tl.int64) * G + groups, -float('inf'), mask=groups < G)
            return
        _logits_fp8_block(
            q_ptr, w_ptr, pages_ptr, row_batch_ptr, buf_ptr, buf_page_stride, out_ptr,
            pages_stride, num_pages, G, r, gb, upper, H, RATIO, PAGE, BLOCK_G,
        )


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
    len_ptr,
    H: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_G: tl.constexpr,
    WRITE_LENGTHS: tl.constexpr,
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
    if WRITE_LENGTHS:
        if gb == 0:
            tl.store(len_ptr + rows, tl.minimum(upper, G).to(tl.int32), mask=row_mask)
    if gb * BLOCK_G >= tl.max(upper, axis=0):
        if not WRITE_LENGTHS:
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


def indexpool_logits_fp8(q_fp8, weights, pages, row_to_batch, positions, index_buf, page_size, num_groups, lengths=None):
    rows, heads, _ = q_fp8.shape
    out = torch.empty((rows, num_groups), dtype=torch.float32, device=q_fp8.device)
    write_lengths = lengths is not None
    if rows == 0 or num_groups == 0:
        if write_lengths:
            lengths.zero_()
        return out
    q_u8 = q_fp8.contiguous().view(torch.uint8)
    len_arg = lengths if write_lengths else positions
    if rows >= 64:
        block_r, block_g, warps = 4, 256, 4
        _logits_rows_fp8_kernel[(triton.cdiv(rows, block_r), triton.cdiv(num_groups, block_g))](
            q_u8, weights, pages, row_to_batch, positions, index_buf, index_buf.stride(0), out,
            pages.stride(0), pages.shape[1], rows, num_groups, len_arg,
            H=heads, RATIO=4, PAGE=page_size, BLOCK_R=block_r, BLOCK_G=block_g,
            WRITE_LENGTHS=write_lengths, num_warps=warps,
        )
        return out
    block_g = 128
    group_blocks = triton.cdiv(num_groups, block_g)
    if write_lengths:
        group_blocks = min(group_blocks, _DECODE_PROGRAMS_PER_ROW)
    _logits_fp8_kernel[(rows, group_blocks)](
        q_u8, weights, pages, row_to_batch, positions, index_buf, index_buf.stride(0), out,
        pages.stride(0), pages.shape[1], num_groups, len_arg,
        H=heads, RATIO=4, PAGE=page_size, BLOCK_G=block_g,
        WRITE_LENGTHS=write_lengths, num_warps=4,
    )
    return out


def _indexpool_logits(original, q_fp8, weights, pages, row_to_batch, positions, index_buf, page_size, num_groups, lengths=None):
    rows, heads, _ = q_fp8.shape
    if not HW_FP8 or heads < 16 or q_fp8.dtype != torch.float8_e4m3fn:
        if lengths is not None:
            return original(q_fp8, weights, pages, row_to_batch, positions, index_buf, page_size, num_groups, lengths=lengths)
        return original(q_fp8, weights, pages, row_to_batch, positions, index_buf, page_size, num_groups)
    return indexpool_logits_fp8(
        q_fp8, weights, pages, row_to_batch, positions, index_buf, page_size, num_groups, lengths=lengths
    )


def _indexpool_select_ragged(original, self, layout, q_fp8, weights, decode):
    if not decode or not HW_FP8 or q_fp8.dtype != torch.float8_e4m3fn or q_fp8.shape[1] < 16:
        return original(self, layout, q_fp8, weights, decode=decode)
    positions = layout.positions
    rows = positions.shape[0]
    num_groups = triton.cdiv(layout.max_tokens // self.ratio, 128) * 128
    count = min(self.group_budget, num_groups)
    if rows == 0 or count <= 0:
        return original(self, layout, q_fp8, weights, decode=decode)

    from flashinfer.topk import top_k_ragged_transform
    from sglang.kernels.ops.attention.dsa import hy4_flash_indexpool as indexpool_kernels
    from sglang.srt.models.hunyuan_v4_flash import _ragged_offsets

    pages = layout.metadata.get_page_table_64()
    row_to_batch = layout.row_to_batch.contiguous()
    buf, page_size = self._index_buffer()
    lengths = torch.empty((rows,), dtype=torch.int32, device=positions.device)
    logits = indexpool_logits_fp8(
        q_fp8, weights, pages, row_to_batch, positions, buf, page_size, num_groups, lengths=lengths
    )
    picked = top_k_ragged_transform(
        logits,
        torch.zeros_like(lengths),
        lengths,
        count,
        dsa_graph_safe=True,
    )
    # Upstream indexpool_expand filters by (val > -inf) & (group >= 0); top_k_ragged_transform marks
    # empty slots with group = -1, so zero values preserve exact masking with zero extra kernel.
    return indexpool_kernels.indexpool_expand(
        picked,
        torch.zeros_like(picked, dtype=torch.float32),
        positions,
        pages,
        row_to_batch,
        _ragged_offsets(layout),
        self.index_topk,
    )


def install(registry, hook_type):
    if HW_FP8 and os.environ.get('XTIER_HY4_INDEXPOOL_FP8', '1') != '0':
        registry.register(TARGET, _indexpool_logits, hook_type.AROUND)
        registry.register(SELECT_TARGET, _indexpool_select_ragged, hook_type.AROUND)
