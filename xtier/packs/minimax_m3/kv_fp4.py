"""MiniMax M3.1 sparse attention reading an NVFP4 KV cache on SM120 (XTIER_MINIMAX_KV_FP4, default on).

The KV cache is SGLang's own MSA-v1 NVFP4 pool (``MiniMaxM3Nvfp4KVPool``, what the native SM100 stack uses): main K/V
and the single-head index K packed as E2M1 pairs with one E4M3 scale per 16 values and a unit global scale, in
128-token page rows ``[page, head, slot in page, byte]``. On SM120 the native stack's MSA-v1 kernels do not run, so the
generic stack's Triton sparse attention reads it here: the same top-k blocks and online softmax, with every K/V tile
decoded from its nibbles and scales in registers (the even and odd head dimensions as two half tiles). A top-k block
of 128 tokens is exactly one page (block size = page size), so a tile is one page row.

Each layer's payload views (main K, main V, index K) carry their scale bytes (``attach_scales``), and the sparse ops of
SGLang's generic MiniMax path take the NVFP4 code here when handed such a view: decode reads the cache in registers;
a prefill copies its requests' contexts to BF16 (each token decoded once per layer and chunk) and runs SGLang's own
prefill kernels on the copies, with the in-register prefill kernels as the fallback for copies beyond
XTIER_MINIMAX_FP4_PREFILL_GB. The stores are SGLang's (its fused quantize+store kernel, enabled on SM120 by
``enable_fused_store``).
"""

import logging
import os
import sys

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
PAGE = 128
OPS = 'sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse'
PREFILL_GB = float(os.environ.get('XTIER_MINIMAX_FP4_PREFILL_GB', '1.5'))


@triton.jit
def _e2m1(code):
    """E2M1 nibble (bits 0..3) -> its value as FP32: exponent bits 2..1, mantissa bit 0, sign bit 3."""
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    v = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((code & 8) != 0, -v, v)


@triton.jit
def _fp4_tile(pay, sc, page, head, rows, rmask, offs_i, s_pp, s_ph, s_po, c_pp, c_ph, c_po):
    """Rows `rows` (slots within one page) of one head: (even, odd) halves [rows, D/2] as BF16 (E2M1 x E4M3 scale per
    16 values; byte i holds values 2i, 2i+1 and uses scale i // 8)."""
    base = page * s_pp + head * s_ph
    b = tl.load(pay + base + rows[:, None] * s_po + offs_i[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    s = tl.load(sc + page * c_pp + head * c_ph + rows[:, None] * c_po + (offs_i // 8)[None, :], mask=rmask[:, None],
                other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return (_e2m1(b & 15) * s).to(tl.bfloat16), (_e2m1(b >> 4) * s).to(tl.bfloat16)


@triton.jit
def _fp4_sparse_decode_kernel(
        q_ptr, k_pay, k_sc, v_pay, v_sc, req_to_token_ptr, idx_ptr, o_ptr, lse_ptr, seq_lens, slot_ids,
        batch_size, gqa_group_size, max_topk, max_kv_len, sm_scale, kv_scale,
        stride_q_b, stride_q_h, stride_q_d,
        s_pp, s_ph, s_po, c_pp, c_ph, c_po,
        stride_r2t_b, stride_ti_h, stride_ti_b, stride_ti_t,
        stride_o_c, stride_o_b, stride_o_h, stride_o_d, stride_l_c, stride_l_b, stride_l_h,
        HALF_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_T: tl.constexpr,
        NUM_TOPK_CHUNKS: tl.constexpr):
    """Split-K over the top-k blocks as SGLang's _gqa_share_sparse_decode_kernel_impl (no sink, no HiSparse, block =
    page): partial outputs and log-sum-exps per chunk, merged by its _merge_topk_attn_out_kernel."""
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    seq_len = tl.minimum(tl.load(seq_lens + pid_b), max_kv_len)
    sid = tl.load(slot_ids + pid_b).to(tl.int64)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b
    topk_idx = tl.load(idx_base + off_t * stride_ti_t, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    chunk_end_topk = tl.minimum(chunk_start_topk + chunk_size_topk, real_topk)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    offs_i = tl.arange(0, HALF_D)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    hmask = off_h < gqa_group_size
    q_base = q_ptr + pid_b * stride_q_b + (pid_h + off_h)[:, None] * stride_q_h
    q_even = tl.load(q_base + (2 * offs_i)[None, :] * stride_q_d, mask=hmask[:, None], other=0.0)
    q_odd = tl.load(q_base + (2 * offs_i + 1)[None, :] * stride_q_d, mask=hmask[:, None], other=0.0)
    m_i = tl.full((BLOCK_SIZE_H,), float('-inf'), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float('-inf'), dtype=tl.float32)
    acc_e = tl.zeros((BLOCK_SIZE_H, HALF_D), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, HALF_D), dtype=tl.float32)
    cur_idx_ptr = idx_base + chunk_start_topk * stride_ti_t
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        c = tl.load(cur_idx_ptr).to(tl.int32) * BLOCK_SIZE_N
        cur_idx_ptr = cur_idx_ptr + stride_ti_t
        pos_mask = c + off_n < seq_len
        base_slot = tl.load(req_to_token_ptr + sid * stride_r2t_b + c, mask=c < seq_len, other=0).to(tl.int64)
        page = base_slot // BLOCK_SIZE_N
        rows = base_slot % BLOCK_SIZE_N + off_n
        k_e, k_o = _fp4_tile(k_pay, k_sc, page, pid_kh, rows, pos_mask, offs_i, s_pp, s_ph, s_po, c_pp, c_ph, c_po)
        v_e, v_o = _fp4_tile(v_pay, v_sc, page, pid_kh, rows, pos_mask, offs_i, s_pp, s_ph, s_po, c_pp, c_ph, c_po)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
        qk += tl.where(off_n[None, :] < seq_len - c, 0, float('-inf'))
        qk += (tl.dot(q_even, tl.trans(k_e)) + tl.dot(q_odd, tl.trans(k_o))) * (sm_scale * kv_scale)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        pb = p.to(tl.bfloat16)
        acc_e = acc_e * alpha[:, None] + tl.dot(pb, v_e) * kv_scale
        acc_o = acc_o * alpha[:, None] + tl.dot(pb, v_o) * kv_scale
        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
    scale = tl.where(lse_i > float('-inf'), tl.exp(m_i - lse_i), tl.zeros_like(lse_i))
    o_base = o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + (pid_h + off_h)[:, None] * stride_o_h
    tl.store(o_base + (2 * offs_i)[None, :] * stride_o_d, (acc_e * scale[:, None]).to(o_ptr.dtype.element_ty),
             mask=hmask[:, None])
    tl.store(o_base + (2 * offs_i + 1)[None, :] * stride_o_d, (acc_o * scale[:, None]).to(o_ptr.dtype.element_ty),
             mask=hmask[:, None])
    tl.store(lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h, lse_i, mask=hmask)


def fp4_sparse_decode(q, k_pay, k_sc, v_pay, v_sc, req_to_token, slot_ids, topk_idx, seq_lens, block_size,
                      sm_scale=None, kv_scale=1.0, num_warps=4, num_stages=2):
    """Main-head sparse decode over an NVFP4 page cache: q [b, qh, d] BF16; k_pay / v_pay [pages, kh, 128, d/2] uint8,
    k_sc / v_sc [pages, kh, 128, d/16] float8_e4m3fn (or its bytes); topk_idx [kh, b, topk] block ids (-1 padded). Returns [b, qh, d]."""
    from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import _merge_topk_attn_out_kernel

    assert block_size == PAGE and k_pay.shape[2] == PAGE and k_pay.stride(3) == 1 and k_sc.stride(3) == 1
    assert k_pay.stride() == v_pay.stride() and k_sc.stride() == v_sc.stride()
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_pay.shape[1]
    gqa_group_size = num_q_heads // num_kv_heads
    max_topk = topk_idx.shape[2]
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    target = max(1, min(max_topk, 256 // max(1, batch_size * num_kv_heads)))
    chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(chunks, batch_size, num_q_heads, head_dim, dtype=q.dtype, device=q.device)
    lse_partial = torch.empty(chunks, batch_size, num_q_heads, dtype=torch.float32, device=q.device)
    k_sc8, v_sc8 = (t.view(torch.uint8) if t.dtype != torch.uint8 else t for t in (k_sc, v_sc))
    _fp4_sparse_decode_kernel[(batch_size * chunks, num_kv_heads)](
        q, k_pay, k_sc8, v_pay, v_sc8, req_to_token, topk_idx, o_partial, lse_partial, seq_lens, slot_ids,
        batch_size, gqa_group_size, max_topk, req_to_token.shape[1], sm_scale, kv_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_pay.stride(0), k_pay.stride(1), k_pay.stride(2), k_sc8.stride(0), k_sc8.stride(1), k_sc8.stride(2),
        req_to_token.stride(0), topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        HALF_D=head_dim // 2, BLOCK_SIZE_H=max(16, triton.next_power_of_2(gqa_group_size)), BLOCK_SIZE_N=block_size,
        BLOCK_SIZE_T=triton.next_power_of_2(max_topk), NUM_TOPK_CHUNKS=chunks, num_warps=num_warps,
        num_stages=num_stages)
    _merge_topk_attn_out_kernel[(batch_size, num_q_heads)](
        o_partial, lse_partial, head_dim, o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
        o_partial.stride(3), lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        NUM_TOPK_CHUNKS=chunks)
    return o_partial[0].contiguous()


@triton.jit
def _fp4_rows(pay, sc, slots, rmask, offs_i, s_pp, s_po, c_pp, c_po):
    """Index K rows of arbitrary slots (pool [page, slot in page, byte]): (even, odd) halves [n, D/2] as BF16."""
    page = slots // 128
    row = slots % 128
    b = tl.load(pay + page[:, None] * s_pp + row[:, None] * s_po + offs_i[None, :], mask=rmask[:, None],
                other=0).to(tl.int32)
    s = tl.load(sc + page[:, None] * c_pp + row[:, None] * c_po + (offs_i // 8)[None, :], mask=rmask[:, None],
                other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return (_e2m1(b & 15) * s).to(tl.bfloat16), (_e2m1(b >> 4) * s).to(tl.bfloat16)


@triton.jit
def _fp4_decode_score_kernel(
        q_ptr, k_pay, k_sc, req_to_token_ptr, score_ptr, seq_lens, slot_ids,
        max_slots, batch_size, gqa_group_size, block_size: tl.constexpr, topk: tl.constexpr, sm_scale, k_scale,
        init_blocks, local_blocks, stride_q_b, stride_q_h, stride_q_d, s_pp, s_po, c_pp, c_po, stride_r2t_b,
        stride_s_h, stride_s_b, stride_s_n,
        HALF_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, NUM_KV_CHUNKS: tl.constexpr, SCORE_TYPE: tl.constexpr,
        SKIP_TRIVIAL_TOPK_SCORE: tl.constexpr):
    """SGLang's _decode_score_kernel (score only, one index head, BLOCK_SIZE_N = block_size) over an NVFP4 index K."""
    tl.static_assert(SCORE_TYPE == 'max' or SCORE_TYPE == 'lse')
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = (seq_len + block_size - 1) // block_size
    if SKIP_TRIVIAL_TOPK_SCORE:
        if num_blocks <= topk:
            return
    chunk_size_blocks = tl.cdiv(num_blocks, NUM_KV_CHUNKS)
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    chunk_start = chunk_start_block * block_size
    chunk_end = tl.minimum(chunk_end_block * block_size, seq_len)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    off_h = tl.arange(0, BLOCK_SIZE_H)
    hmask = off_h < gqa_group_size
    offs_i = tl.arange(0, HALF_D)
    q_base = q_ptr + pid_b * stride_q_b + (pid_h + off_h)[:, None] * stride_q_h
    q_even = tl.load(q_base + (2 * offs_i)[None, :] * stride_q_d, mask=hmask[:, None], other=0.0)
    q_odd = tl.load(q_base + (2 * offs_i + 1)[None, :] * stride_q_d, mask=hmask[:, None], other=0.0)
    off_n = tl.arange(0, block_size)
    local_start = tl.maximum(0, num_blocks - local_blocks)
    r2t_base = req_to_token_ptr + sid * stride_r2t_b
    s_base = score_ptr + pid_b * stride_s_b + (pid_h + off_h) * stride_s_h
    for i in range(chunk_start, chunk_end, block_size):
        pos_mask = i + off_n < seq_len
        slots = tl.load(r2t_base + i + off_n, mask=pos_mask, other=0).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        k_e, k_o = _fp4_rows(k_pay, k_sc, slots, pos_mask, offs_i, s_pp, s_po, c_pp, c_po)
        qk = tl.zeros((BLOCK_SIZE_H, block_size), dtype=tl.float32)
        qk += tl.where(off_n[None, :] < chunk_end - i, 0, float('-inf'))
        qk += (tl.dot(q_even, tl.trans(k_e)) + tl.dot(q_odd, tl.trans(k_o))) * (sm_scale_log2e * k_scale)
        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == 'max':
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float('-inf'), score)
        blk = i // block_size
        score = tl.where((blk >= local_start) & (blk < num_blocks), 1e29, tl.where(blk < init_blocks, 1e30, score))
        tl.store(s_base + blk * stride_s_n, score.to(score_ptr.dtype.element_ty), mask=hmask)


class _ScoreShim:
    """Stands in for _decode_score_kernel inside SGLang's flash_decode_with_topk_idx: same launch arguments, the NVFP4
    index K (payload, scale bytes) taken from here instead of the stand-in cache."""

    def __init__(self, pay, sc):
        self.pay, self.sc = pay, sc

    def __getitem__(self, grid):
        def launch(q, _k, req_to_token, score, seq_lens, slot_ids, max_slots, batch_size, gqa_group_size, head_dim,
                   block_size, topk, sm_scale, k_scale, init_blocks, local_blocks, sq0, sq1, sq2, _sk0, _sk1, _sk2,
                   sr0, ss0, ss1, ss2, NUM_KV_CHUNKS, SCORE_TYPE, SKIP_TRIVIAL_TOPK_SCORE, IS_FP8=False, **_):
            pay, sc = self.pay, self.sc
            _fp4_decode_score_kernel[grid](
                q, pay, sc, req_to_token, score, seq_lens, slot_ids, pay.shape[0] * pay.shape[1], batch_size,
                gqa_group_size, block_size, topk, sm_scale, k_scale, init_blocks, local_blocks, sq0, sq1, sq2,
                pay.stride(0), pay.stride(1), sc.stride(0), sc.stride(1), sr0, ss0, ss1, ss2,
                HALF_D=head_dim // 2, BLOCK_SIZE_H=max(16, triton.next_power_of_2(gqa_group_size)),
                NUM_KV_CHUNKS=NUM_KV_CHUNKS, SCORE_TYPE=SCORE_TYPE, SKIP_TRIVIAL_TOPK_SCORE=SKIP_TRIVIAL_TOPK_SCORE,
                num_warps=4, num_stages=2)
        return launch


def fp4_decode_topk_idx(q, idx_pay, idx_sc, run=None, **kwargs):
    """SGLang's flash_decode_with_topk_idx (``run``; score only: disable_index_value) on an NVFP4 index K
    [pages, 128, d/2] with scale bytes [pages, 128, d/16]: its own chunking and top-k, the score kernel replaced for
    the call."""
    if run is None:
        from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import \
            flash_decode_with_topk_idx as run

    assert kwargs.get('disable_index_value') and kwargs.get('sink') is None
    mod = sys.modules[run.__module__]
    sc8 = idx_sc.view(torch.uint8) if idx_sc.dtype != torch.uint8 else idx_sc
    stand_in = torch.empty((1, 1, 1), dtype=q.dtype, device=q.device).expand(
        idx_pay.shape[0] * idx_pay.shape[1], 1, idx_pay.shape[2] * 2)
    real = mod._decode_score_kernel
    mod._decode_score_kernel = _ScoreShim(idx_pay, sc8)
    try:
        return run(q=q, k_cache=stand_in, **kwargs)
    finally:
        mod._decode_score_kernel = real



@triton.jit
def _fp4_rows_h(pay, sc, slots, head, rmask, offs_i, s_pp, s_ph, s_po, c_pp, c_ph, c_po):
    """Main K/V rows of arbitrary slots for one head (pool [page, head, slot in page, byte]): (even, odd) halves."""
    page = slots // 128
    row = slots % 128
    b = tl.load(pay + page[:, None] * s_pp + head * s_ph + row[:, None] * s_po + offs_i[None, :], mask=rmask[:, None],
                other=0).to(tl.int32)
    s = tl.load(sc + page[:, None] * c_pp + head * c_ph + row[:, None] * c_po + (offs_i // 8)[None, :],
                mask=rmask[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return (_e2m1(b & 15) * s).to(tl.bfloat16), (_e2m1(b >> 4) * s).to(tl.bfloat16)


@triton.jit
def _tile3(base, n0, n1, n2, stride0, stride1, stride2, off0, BLOCK0: tl.constexpr, BLOCK1: tl.constexpr,
           BLOCK2: tl.constexpr):
    """Pointers and mask of a [BLOCK0, BLOCK1, BLOCK2] tile (SGLang's tile_ptrs_3d)."""
    i0 = off0 + tl.arange(0, BLOCK0).to(tl.int64)
    i1 = tl.arange(0, BLOCK1).to(tl.int64)
    i2 = tl.arange(0, BLOCK2).to(tl.int64)
    ptrs = base + i0[:, None, None] * stride0 + i1[None, :, None] * stride1 + i2[None, None, :] * stride2
    mask = (i0[:, None, None] < n0) & (i1[None, :, None] < n1) & (i2[None, None, :] < n2)
    return ptrs, mask


@triton.jit
def _fp4_sparse_fwd_kernel(
        q_ptr, k_pay, k_sc, v_pay, v_sc, t_ptr, o_ptr, req_to_token_ptr, cu_seqlens_q, cu_seqblocks_q, seq_lens,
        prefix_lens, slot_ids, max_slots, gqa_group_size, max_topk, num_q_loop, sm_scale, kv_scale,
        stride_qn, stride_qh, stride_qd, s_pp, s_ph, s_po, c_pp, c_ph, c_po, stride_th, stride_tn, stride_tk,
        stride_on, stride_oh, stride_od, stride_r2t_b,
        HALF_D: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
        BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_QH: tl.constexpr):
    """SGLang's _gqa_share_sparse_fwd_kernel (no sink, no HiSparse map, no sub-tiling) over an NVFP4 main cache."""
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    off_n = tl.arange(0, BLOCK_SIZE_K)
    offs_i = tl.arange(0, HALF_D)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
        qe_ptrs, q_mask = _tile3(q_ptr + q_start * stride_qn + pid_h * stride_qh, q_len, gqa_group_size, HALF_D,
                                 stride_qn, stride_qh, 2 * stride_qd, pid_q_j * BLOCK_SIZE_Q, BLOCK_SIZE_Q,
                                 BLOCK_SIZE_H, HALF_D)
        q_e = tl.reshape(tl.load(qe_ptrs, mask=q_mask, other=0.0), BLOCK_SIZE_QH, HALF_D)
        q_o = tl.reshape(tl.load(qe_ptrs + stride_qd, mask=q_mask, other=0.0), BLOCK_SIZE_QH, HALF_D)
        off_q_k = (tl.arange(0, BLOCK_SIZE_Q)[:, None] + pid_q_j * BLOCK_SIZE_Q + prefix_len
                   - tl.arange(0, BLOCK_SIZE_K)[None, :])
        m_i = tl.full((BLOCK_SIZE_QH,), float('-inf'), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_QH,), float('-inf'), dtype=tl.float32)
        acc_e = tl.zeros((BLOCK_SIZE_QH, HALF_D), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, HALF_D), dtype=tl.float32)
        for i in range(real_topk):
            c = tl.load(t_ptr_j).to(tl.int32) * BLOCK_SIZE_K
            t_ptr_j = t_ptr_j + stride_tk
            pos = c + off_n
            pos_mask = pos < seq_len
            slots = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos, mask=pos_mask, other=0).to(tl.int64)
            slots = (slots + max_slots) % max_slots
            k_e, k_o = _fp4_rows_h(k_pay, k_sc, slots, pid_kh, pos_mask, offs_i, s_pp, s_ph, s_po, c_pp, c_ph, c_po)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where(off_q_k[:, None, :] >= c, 0, float('-inf'))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += (tl.dot(q_e, tl.trans(k_e)) + tl.dot(q_o, tl.trans(k_o))) * (sm_scale_log2e * kv_scale)
            qk += tl.where(pos_mask[None, :], 0, float('-inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp2(m_i - m_ij)
            v_e, v_o = _fp4_rows_h(v_pay, v_sc, slots, pid_kh, pos_mask, offs_i, s_pp, s_ph, s_po, c_pp, c_ph, c_po)
            pb = p.to(tl.bfloat16)
            acc_e = acc_e * alpha[:, None] + tl.dot(pb, v_e) * kv_scale
            acc_o = acc_o * alpha[:, None] + tl.dot(pb, v_o) * kv_scale
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        fin = tl.exp2(m_i - lse_i)[:, None]
        oe_ptrs, o_mask = _tile3(o_ptr + q_start * stride_on + pid_h * stride_oh, q_len, gqa_group_size, HALF_D,
                                 stride_on, stride_oh, 2 * stride_od, pid_q_j * BLOCK_SIZE_Q, BLOCK_SIZE_Q,
                                 BLOCK_SIZE_H, HALF_D)
        tl.store(oe_ptrs, tl.reshape(acc_e * fin, BLOCK_SIZE_Q, BLOCK_SIZE_H, HALF_D).to(o_ptr.dtype.element_ty),
                 mask=o_mask)
        tl.store(oe_ptrs + stride_od, tl.reshape(acc_o * fin, BLOCK_SIZE_Q, BLOCK_SIZE_H, HALF_D).to(
            o_ptr.dtype.element_ty), mask=o_mask)


def fp4_prefill_gqa_share_sparse(q, k_cache, v_cache, sink, req_to_token, slot_ids, topk_idx, block_size_q,
                                 block_size_k, cu_seqlens, seq_lens, prefix_lens, max_seqlen_q, sm_scale=None,
                                 use_tma=True, cu_seqblocks_q=None, max_seqblock_q=None, q_scale=None, k_scale=None,
                                 v_scale=None, loc_mapping=None, *, k_sc=None, v_sc=None, kv_scale=1.0,
                                 num_warps=4, num_stages=2):
    """flash_prefill_with_gqa_share_sparse with the main K/V an NVFP4 page cache: k_cache / v_cache the payload
    [pages, kh, 128, d/2], k_sc / v_sc the scale bytes [pages, kh, 128, d/16]."""
    from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks

    assert sink is None and loc_mapping is None and block_size_k == PAGE
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    gqa_group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    sm_scale = sm_scale * (1.0 if q_scale is None else q_scale)
    if cu_seqblocks_q is None or max_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, _, _, _, _ = get_cu_seqblocks(cu_seqlens, max_seqlen_q, block_size_q,
                                                                      block_size_k)
    o = torch.empty(total_q, num_q_heads, head_dim, device=q.device, dtype=q.dtype)
    num_q_loop = max_seqblock_q // 131072 + 1
    bq = triton.next_power_of_2(block_size_q)
    bh = triton.next_power_of_2(max(16 // bq, triton.next_power_of_2(gqa_group_size)))
    k_sc8, v_sc8 = (t.view(torch.uint8) if t.dtype != torch.uint8 else t for t in (k_sc, v_sc))
    grid = (triton.cdiv(triton.cdiv(max_seqlen_q, block_size_q), num_q_loop), num_kv_heads, batch_size)
    _fp4_sparse_fwd_kernel[grid](
        q, k_cache, k_sc8, v_cache, v_sc8, topk_idx, o, req_to_token, cu_seqlens, cu_seqblocks_q, seq_lens,
        prefix_lens, slot_ids, k_cache.shape[0] * PAGE, gqa_group_size, topk_idx.shape[-1], num_q_loop, sm_scale,
        kv_scale, q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k_sc8.stride(0), k_sc8.stride(1), k_sc8.stride(2), topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        o.stride(0), o.stride(1), o.stride(2), req_to_token.stride(0),
        HALF_D=head_dim // 2, BLOCK_SIZE_Q=bq, BLOCK_SIZE_K=block_size_k, BLOCK_SIZE_H=bh,
        BLOCK_SIZE_T=triton.next_power_of_2(topk_idx.shape[-1]), BLOCK_SIZE_QH=bq * bh, num_warps=num_warps,
        num_stages=num_stages)
    return o


@triton.jit
def _fp4_index_score_kernel(
        q_ptr, k_pay, k_sc, score_ptr, req_to_token_ptr, cu_seqlens, seq_lens, prefix_lens, slot_ids, max_slots,
        num_heads, gqa_group_size, sm_scale, k_scale, stride_q_n, stride_q_h, stride_q_d, s_pp, s_po, c_pp, c_po,
        stride_s_h, stride_s_q, stride_s_k, stride_r2t_b,
        HALF_D: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr, block_size: tl.constexpr):
    """Causal max index score per query and KV block over an NVFP4 index K (SGLang's _index_block_score_only_kernel,
    the score half of _flash_attn_fwd_with_block_score_kernel with DISABLE_INDEX_VALUE)."""
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    q_row = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_row < q_len
    offs_i = tl.arange(0, HALF_D)
    q_base = q_ptr + (seq_start + q_row)[:, None] * stride_q_n + pid_h * stride_q_h
    q_e = tl.load(q_base + (2 * offs_i)[None, :] * stride_q_d, mask=q_mask[:, None], other=0.0)
    q_o = tl.load(q_base + (2 * offs_i + 1)[None, :] * stride_q_d, mask=q_mask[:, None], other=0.0)
    off_q = q_row + prefix_len
    off_k = tl.arange(0, block_size)
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, block_size):
        blk = i // block_size
        pos = i + off_k
        pos_mask = pos < seq_len
        slots = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos, mask=pos_mask, other=0).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        k_e, k_o = _fp4_rows(k_pay, k_sc, slots, pos_mask, offs_i, s_pp, s_po, c_pp, c_po)
        qk = (tl.dot(q_e, tl.trans(k_e)) + tl.dot(q_o, tl.trans(k_o))) * (sm_scale_log2e * k_scale)
        qk = tl.where((off_q[:, None] >= pos[None, :]) & pos_mask[None, :], qk, float('-inf'))
        score = tl.max(qk, axis=1)
        tl.store(score_ptr + pid_h * stride_s_h + (seq_start + q_row) * stride_s_q + blk * stride_s_k, score,
                 mask=q_mask)


class _PrefillScoreShim:
    """Stands in for _flash_attn_fwd_with_block_score_kernel inside SGLang's flash_prefill_with_topk_index (score
    only, disable_index_value): the scores from the NVFP4 index K held here."""

    BLOCK_SIZE_Q = 64

    def __init__(self, pay, sc):
        self.pay, self.sc = pay, sc

    def __getitem__(self, grid):
        def launch(q, _k, _v, sink, o, score, req_to_token, cu_seqlens, seq_lens, prefix_lens, slot_ids, max_slots,
                   num_heads, gqa_group_size, qk_head_dim, v_head_dim, block_size_k, sm_scale, k_scale, v_scale,
                   _gumbel, _one, sq0, sq1, sq2, *rest, SCORE_TYPE='max', DISABLE_INDEX_VALUE=True, IS_FP8=False, **_):
            assert DISABLE_INDEX_VALUE and SCORE_TYPE == 'max' and sink is None and o is None
            ss0, ss1, ss2, sr0 = rest[-4:]
            pay, sc = self.pay, self.sc
            g = grid({'BLOCK_SIZE_Q': self.BLOCK_SIZE_Q}) if callable(grid) else grid
            _fp4_index_score_kernel[g](
                q, pay, sc, score, req_to_token, cu_seqlens, seq_lens, prefix_lens, slot_ids,
                pay.shape[0] * pay.shape[1], num_heads, gqa_group_size, sm_scale, k_scale, sq0, sq1, sq2,
                pay.stride(0), pay.stride(1), sc.stride(0), sc.stride(1), ss0, ss1, ss2, sr0,
                HALF_D=qk_head_dim // 2, BLOCK_SIZE_Q=self.BLOCK_SIZE_Q, block_size=block_size_k, num_warps=4,
                num_stages=2)
        return launch


def fp4_prefill_topk_index(q, idx_pay, idx_sc, run=None, **kwargs):
    """SGLang's flash_prefill_with_topk_index (``run``; score only: disable_index_value) on an NVFP4 index K
    [pages, 128, d/2] with scale bytes [pages, 128, d/16]: its own score buffer and top-k, the score kernel replaced
    for the call."""
    if run is None:
        from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import \
            flash_prefill_with_topk_index as run

    assert kwargs.get('disable_index_value') and kwargs.get('sink') is None
    mod = sys.modules[run.__module__]
    sc8 = idx_sc.view(torch.uint8) if idx_sc.dtype != torch.uint8 else idx_sc
    stand_in = torch.empty((1, 1, 1), dtype=q.dtype, device=q.device).expand(
        idx_pay.shape[0] * idx_pay.shape[1], 1, idx_pay.shape[2] * 2)
    real = mod._flash_attn_fwd_with_block_score_kernel
    mod._flash_attn_fwd_with_block_score_kernel = _PrefillScoreShim(idx_pay, sc8)
    try:
        return run(q=q, k_cache=stand_in, **kwargs)
    finally:
        mod._flash_attn_fwd_with_block_score_kernel = real


@triton.jit
def _dequant_kernel(pay, sc, out, req_to_token, slot_ids, seq_lens, max_slots, rows_per_req, s_pp, s_ph, s_po, c_pp,
                    c_ph, c_po, stride_r2t_b, stride_oh, HALF_D: tl.constexpr, HEADS: tl.constexpr):
    """Request b's positions [blk * 128, blk * 128 + 128) of an NVFP4 page cache (main: [page, head, slot, byte]; index:
    head stride 0) -> BF16 rows b * rows_per_req + position of out [rows, HEADS, 2 * HALF_D]."""
    b, blk = tl.program_id(0), tl.program_id(1)
    pos = blk * 128 + tl.arange(0, 128)
    pmask = pos < tl.load(seq_lens + b)
    sid = (tl.load(slot_ids + b).to(tl.int64) + max_slots) % max_slots
    slots = tl.load(req_to_token + sid * stride_r2t_b + pos, mask=pmask, other=0).to(tl.int64)
    slots = (slots + max_slots) % max_slots
    offs_i = tl.arange(0, HALF_D)
    rows = (b * rows_per_req + pos).to(tl.int64)
    for h in tl.static_range(HEADS):
        e, o = _fp4_rows_h(pay, sc, slots, h, pmask, offs_i, s_pp, s_ph, s_po, c_pp, c_ph, c_po)
        vals = tl.reshape(tl.join(e, o), 128, 2 * HALF_D)   # interleave: value 2i, 2i + 1
        tl.store(out + rows[:, None] * (HEADS * 2 * HALF_D) + h * stride_oh + tl.arange(0, 2 * HALF_D)[None, :], vals,
                 mask=pmask[:, None])


def dequantize_context(pay, sc, req_to_token, slot_ids, seq_lens, rows_per_req, out=None):
    """BF16 [b * rows_per_req, heads, d] of each request's first seq_lens[b] positions (rows_per_req >= their
    maximum, a multiple of 128): pay / sc [pages, heads, 128, d / 2 | d / 16] (main) or [pages, 128, ...] (index)."""
    sc8 = sc.view(torch.uint8) if sc.dtype != torch.uint8 else sc
    if pay.dim() == 3:   # index: one head, no head dimension
        pay, sc8 = pay.unsqueeze(1), sc8.unsqueeze(1)
    pages, heads, _, half = pay.shape
    b = slot_ids.shape[0]
    if out is None:
        out = torch.empty(b * rows_per_req, heads, 2 * half, dtype=torch.bfloat16, device=pay.device)
    _dequant_kernel[(b, rows_per_req // PAGE)](
        pay, sc8, out, req_to_token, slot_ids, seq_lens, pages * PAGE, rows_per_req, pay.stride(0), pay.stride(1),
        pay.stride(2), sc8.stride(0), sc8.stride(1), sc8.stride(2), req_to_token.stride(0), 2 * half,
        HALF_D=half, HEADS=heads, num_warps=4)
    return out


def attach_scales(pool):
    """Each layer's payload views of SGLang's NVFP4 pool (main K, main V, index K: the views its accessors return)
    carry their scale bytes; the sparse ops below take such a view to the NVFP4 kernels."""
    main, index = pool.main_pool, pool.index_k_pool
    for pays, scales in ((main.k_buffer, main.k_scale_buffer), (main.v_buffer, main.v_scale_buffer),
                         (index.k_buffer, index.k_scale_buffer)):
        for pay, sc in zip(pays, scales):
            pay._xtier_fp4 = sc
    logger.info('xtier: MiniMax M3.1 KV cache in NVFP4 (SGLang MSA-v1 pool, XTier Triton sparse attention): '
                '%d tokens, %.2f GiB main K/V, %.2f GiB index K', pool.size, main.slab.nbytes / 2 ** 30,
                index.slab.nbytes / 2 ** 30)


def enable_fused_store(pool):
    """SGLang's fused NVFP4 quantize + store (nvfp4_store_kv_index: main K/V and index K in one launch) on SM120 too,
    where the pool takes its eager path (FlashInfer's quantizer per buffer, then the scatters: about ten launches per
    layer); SGLang enables it on SM100 only, for want of a bit-exactness check elsewhere. On SM120 it is byte-identical
    to the eager path (xtier-dev/nvfp4_store_check.py: 1 to 4096 tokens, strided views) and takes 6 instead of 260 us
    per layer call. XTIER_MINIMAX_FP4_FUSED_STORE=0 keeps the eager path."""
    from sglang.srt.environ import envs

    main, index = pool.main_pool, pool.index_k_pool
    if (os.environ.get('XTIER_MINIMAX_FP4_FUSED_STORE', '1') != '0' and not pool.use_nvfp4_fused_quantize_store
            and envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.get() and main.head_dim == index.INDEX_HEAD_DIM
            and main.page_size == index.page_size and torch.equal(main.global_scale, index.global_scale)):
        pool.use_nvfp4_fused_quantize_store = True
        logger.info('xtier: MiniMax M3.1 NVFP4 KV store: fused quantize+store on SM120 (byte-identical)')


def _scales(t):
    return getattr(t, '_xtier_fp4', None)


def _unit(kwargs, *names):
    for name in names:
        if kwargs.get(name) not in (None, 1.0):
            raise ValueError(f'xtier: the NVFP4 KV cache has unit outer scales, got {name}={kwargs[name]}')


def _decode_topk(original, *args, **kwargs):
    sc = _scales(kwargs.get('k_cache'))
    if sc is None:
        return original(*args, **kwargs)
    return fp4_decode_topk_idx(kwargs.pop('q'), kwargs.pop('k_cache'), sc, run=original, **kwargs)


def _decode_main(original, *args, **kwargs):
    k_sc = _scales(kwargs.get('k_cache'))
    if k_sc is None:
        return original(*args, **kwargs)
    _unit(kwargs, 'k_scale', 'v_scale')
    if kwargs.get('sink') is not None or kwargs.get('hisparse_slots') is not None:
        raise NotImplementedError('xtier: the NVFP4 KV sparse decode has no attention sink or HiSparse')
    q, v = kwargs['q'], kwargs['v_cache']
    sm_scale = kwargs.get('sm_scale')
    sm_scale = (q.shape[-1] ** -0.5 if sm_scale is None else sm_scale) * (kwargs.get('q_scale') or 1.0)
    return fp4_sparse_decode(q, kwargs['k_cache'], k_sc, v, _scales(v), kwargs['req_to_token'], kwargs['slot_ids'],
                             kwargs['topk_idx'], kwargs['seq_lens'], kwargs['block_size'], sm_scale=sm_scale)


def _prefill_topk(original, *args, **kwargs):
    sc = _scales(kwargs.get('k_cache'))
    if sc is None:
        return original(*args, **kwargs)
    return fp4_prefill_topk_index(kwargs.pop('q'), kwargs.pop('k_cache'), sc, run=original, **kwargs)


def _prefill_main(original, *args, **kwargs):
    k_sc = _scales(kwargs.get('k_cache'))
    if k_sc is None:
        return original(*args, **kwargs)
    _unit(kwargs, 'k_scale', 'v_scale')
    return fp4_prefill_gqa_share_sparse(**kwargs, k_sc=k_sc, v_sc=_scales(kwargs['v_cache']))


def _sparse_prefill(original, q, k_cache, v_cache, sink, idx_q, idx_k_cache, idx_v_cache, idx_sink, req_to_token,
                    slot_ids, cu_seqlens, seq_lens, prefix_lens, max_seqlen_q, max_seqlen_k, *args, **kwargs):
    """A prefill on the NVFP4 cache runs SGLang's own kernels on BF16 copies of its requests' contexts (each request's
    positions on consecutive rows, a request-local req_to_token): the FP4 kernels decode every K/V block again for
    each query (block_size_q 1), a copy decodes each token once per layer and chunk. The products of E2M1 values and
    E4M3 scales are exact in BF16, so the result is the FP4 kernels'. Beyond XTIER_MINIMAX_FP4_PREFILL_GB of copies
    (default 1.5: one 512K request takes 1.15 GB) the FP4 kernels serve it."""
    k_sc = _scales(k_cache)
    b = slot_ids.shape[0]
    rows = -(-int(max_seqlen_k) // PAGE) * PAGE
    if k_sc is not None and idx_v_cache is None and kwargs.get('loc_mapping') is None:
        width = 2 * k_cache.shape[1] * 2 * k_cache.shape[3] + 2 * idx_k_cache.shape[2]   # K, V and index K values
        if b * rows * width * 2 <= PREFILL_GB * 2 ** 30:
            dev = k_cache.device
            local = torch.arange(b, dtype=torch.int32, device=dev)
            r2t = local[:, None] * rows + torch.arange(rows, dtype=torch.int32, device=dev)[None, :]
            k, v, ik = (dequantize_context(c, _scales(c), req_to_token, slot_ids, seq_lens, rows)
                        for c in (k_cache, v_cache, idx_k_cache))
            return original(q, k, v, sink, idx_q, ik, None, idx_sink, r2t, local, cu_seqlens, seq_lens, prefix_lens,
                            max_seqlen_q, max_seqlen_k, *args, **kwargs)
    return original(q, k_cache, v_cache, sink, idx_q, idx_k_cache, idx_v_cache, idx_sink, req_to_token, slot_ids,
                    cu_seqlens, seq_lens, prefix_lens, max_seqlen_q, max_seqlen_k, *args, **kwargs)


def install(registry, hook_type):
    """The sparse ops of SGLang's generic MiniMax path (as its minimax_sparse module calls them: the four kernels by
    keyword, the prefill entry positionally)."""
    for name, hook in (('flash_decode_with_topk_idx', _decode_topk), ('flash_decode_with_gqa_share_sparse', _decode_main),
                       ('flash_prefill_with_topk_index', _prefill_topk),
                       ('flash_prefill_with_gqa_share_sparse', _prefill_main), ('minimax_sparse_prefill', _sparse_prefill)):
        registry.register(f'{OPS}.{name}', hook, hook_type.AROUND)
