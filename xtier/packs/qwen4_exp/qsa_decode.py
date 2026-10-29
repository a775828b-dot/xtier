"""QSA decode / verify attention below SM90 (RTX 30 / 40 series): split-key Triton kernels in place of flash_attn's
varlen kernel.

Off SM100 / SM120 (trtllm-gen) SGLang's QwenSparseAttnBackend packs each query row's selected keys
(qwen_sparse_kv_extraction_compact_triton) and runs flash_attn_varlen_func on them, one query per row. Below SM90 that
is FA4's SM80 kernel (FA2 is not installed), whose head_dim 256 tiles need 128 KB of shared memory per block where
SM86 / SM89 have 99 KB. SGLang's own Triton decode (its ROCm branch, sparse_gqa_packed_decode_triton) rounds the
scaled query to BF16 and runs one program per row and KV head. Here, on the same packed rows: the softmax scale is
applied to the FP32 scores, P is BF16 for the PV product as in the flash kernels, accumulation is FP32; a row's keys
are split over programs and the partial results merged in FP32, so the values differ from one pass only in summation
order. The hook (on SGLang's resolver of the varlen function) gives these kernels on GPUs below SM90 and leaves every
other GPU to SGLang.
"""
import functools

import torch
import triton
import triton.language as tl

KEYS_PER_SPLIT = 128   # keys of a row per program (top-2048 rows: 16 programs per row and KV head)
BLOCK_N = 32           # keys per iteration: K and V tiles of 16 KB each at head_dim 256, two stages fit in 99 KB
LOG2E = 1.4426950408889634


@triton.jit
def _split(Q, K, V, CU_Q, CU_K, M, L, ACC, scale, SQ_M, SQ_H, SK_N, SK_H, SV_N, SV_H,
           HEADS: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, DIM: tl.constexpr, SPLITS: tl.constexpr,
           KEYS: tl.constexpr, BLOCK_N: tl.constexpr):
    """One row, KV head and key range: the running max M, sum L and unnormalized output ACC of its query heads."""
    row, kvh, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query = tl.load(CU_Q + row).to(tl.int64)
    k0 = tl.load(CU_K + row).to(tl.int64)
    count = tl.load(CU_K + row + 1).to(tl.int64) - k0
    h = tl.arange(0, BLOCK_M)
    d = tl.arange(0, DIM)
    heads = kvh * GROUP + h
    hmask = h < GROUP
    q = tl.load(Q + query * SQ_M + heads[:, None] * SQ_H + d[None, :], mask=hmask[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, DIM], tl.float32)
    lo = split * KEYS
    hi = tl.minimum(lo + KEYS, count)
    n = tl.arange(0, BLOCK_N)
    for start in range(lo, hi, BLOCK_N):
        key = start + n
        valid = key < hi
        k = tl.load(K + (k0 + key)[None, :] * SK_N + kvh * SK_H + d[:, None], mask=valid[None, :], other=0.0)
        v = tl.load(V + (k0 + key)[:, None] * SV_N + kvh * SV_H + d[None, :], mask=valid[:, None], other=0.0)
        s = tl.where(valid[None, :], tl.dot(q, k) * scale, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    at = (row * SPLITS + split) * HEADS + heads
    tl.store(M + at, m_i, mask=hmask)
    tl.store(L + at, l_i, mask=hmask)
    tl.store(ACC + at[:, None] * DIM + d[None, :], acc, mask=hmask[:, None])


@triton.jit
def _merge(M, L, ACC, OUT, CU_Q, SO_M, SO_H, HEADS: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr,
           DIM: tl.constexpr, SPLITS: tl.constexpr):
    """A row's key ranges merged in FP32: weights 2^(m - max m), output = sum(w * ACC) / sum(w * L)."""
    row, kvh = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, BLOCK_M)
    d = tl.arange(0, DIM)
    heads = kvh * GROUP + h
    hmask = h < GROUP
    top = tl.full([BLOCK_M], float('-inf'), tl.float32)
    for s in tl.static_range(SPLITS):
        top = tl.maximum(top, tl.load(M + (row * SPLITS + s) * HEADS + heads, mask=hmask, other=float('-inf')))
    top = tl.where(top == float('-inf'), 0.0, top)   # a row without keys (graph padding): every weight is 0
    total = tl.zeros([BLOCK_M], tl.float32)
    out = tl.zeros([BLOCK_M, DIM], tl.float32)
    for s in tl.static_range(SPLITS):
        at = (row * SPLITS + s) * HEADS + heads
        w = tl.math.exp2(tl.load(M + at, mask=hmask, other=float('-inf')) - top)
        total += w * tl.load(L + at, mask=hmask, other=0.0)
        out += w[:, None] * tl.load(ACC + at[:, None] * DIM + d[None, :], mask=hmask[:, None], other=0.0)
    out = tl.where(total[:, None] > 0, out / total[:, None], 0.0)
    query = tl.load(CU_Q + row).to(tl.int64)
    tl.store(OUT + query * SO_M + heads[:, None] * SO_H + d[None, :], out.to(OUT.dtype.element_ty),
             mask=hmask[:, None])


def varlen_decode(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal=True, **_):
    """flash_attn_varlen_func for QSA's decode-shaped calls: one query per row, which sees all of the row's packed keys
    [cu_seqlens_k[r], cu_seqlens_k[r + 1]). q [rows, heads, dim]; k, v [keys, KV heads, dim]."""
    if max_seqlen_q != 1:
        raise NotImplementedError('the SM8x QSA decode takes one query per row')
    q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
    rows = cu_seqlens_q.shape[0] - 1
    heads, dim = q.shape[1], q.shape[2]
    kv_heads = k.shape[1]
    group = heads // kv_heads
    out = torch.empty_like(q)
    if rows <= 0:
        return out
    splits = triton.cdiv(max(int(max_seqlen_k), 1), KEYS_PER_SPLIT)
    block_m = max(16, triton.next_power_of_2(group))
    m = torch.empty((rows, splits, heads), dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    acc = torch.empty((rows, splits, heads, dim), dtype=torch.float32, device=q.device)
    _split[(rows, kv_heads, splits)](q, k, v, cu_seqlens_q, cu_seqlens_k, m, l, acc, softmax_scale * LOG2E,
                                     q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                                     HEADS=heads, GROUP=group, BLOCK_M=block_m, DIM=dim, SPLITS=splits,
                                     KEYS=KEYS_PER_SPLIT, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
    _merge[(rows, kv_heads)](m, l, acc, out, cu_seqlens_q, out.stride(0), out.stride(1), HEADS=heads, GROUP=group,
                             BLOCK_M=block_m, DIM=dim, SPLITS=splits, num_warps=4)
    return out


@functools.lru_cache(maxsize=1)
def _below_sm90():
    return torch.cuda.get_device_capability()[0] < 9


def resolve(original):
    """AROUND hook of qwen_sparse_attn_backend._resolve_flash_attn_varlen_func."""
    return varlen_decode if _below_sm90() else original()
