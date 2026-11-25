"""Target NVFP4 KV cache for Thinking Machines Inkling-2 (XTIER_INKLING2_KV_FP4=1).

Compresses Inkling-2's 48-layer target KV cache (8 global + 40 SWA, 8 KV heads, head_dim=128) from BF16
(256 B/head = 4,096 B/tok/layer) to packed NVFP4 E2M1 + per-16 E4M3 scales (72 B/head = 1,152 B/tok/layer,
3.556x smaller — saving ~17.25 GiB at 512K context). The 5-layer MTP draft KV cache stays in exact BF16
(only ~2.80 GiB at 512K, preserving draft RoPE and block-anchored bidirectional windows).

Target K is L2-clipped with no RoPE and target V is RMS-normalized (RMS=1.0), making both exceptionally
well-conditioned for block-16 NVFP4 quantization.

Packed uint8 layout per (slot, kv_head) [72 bytes]:
  - Bytes  0..63: 64 packed E2M1 nibbles; byte i holds lo = x[i] (group i//16) and hi = x[i+64] (group 4+i//16).
  - Bytes 64..67: 4 float8_e4m3fn scales for x[0..63]  (1 per 16 values, amax / 6.0, clamped >= 1e-6).
  - Bytes 68..71: 4 float8_e4m3fn scales for x[64..127] (1 per 16 values, amax / 6.0, clamped >= 1e-6).

Execution paths:
  - Store (MHATokenToKVPool.set_kv_buffer): 1-launch CUDA-graph-safe Triton quant+store for K and V.
  - Decode & DSpark verify (rel_bias_partials): split-KV Triton kernel decoding K/V in registers as split-half
    64-dim tiles (zero tl.interleave, pure tl.dot, CUDA-graph safe, 3.56x less KV bandwidth).
  - Prefill / extend (FlashAttentionBackend.flash_attn_with_kvcache): 1-launch Triton unpack of only the active
    pages referenced by page_table/cache_seqlens (bounded by sliding window for SWA layers) into a compact BF16
    staging buffer, then dispatches straight to native FA4 SM120.
"""

import logging
import os
import threading

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

NVFP4_KV_DIM = 72
BF16_KV_BYTES_PER_HEAD = 256  # head_dim=128 * 2 bytes
MAX_SPLITS = 64

HYBRID_SWA_CONFIG_INIT = (
    'sglang.srt.model_executor.pool_configurator.HybridSWAPoolConfigurator.__init__'
)
BUILD_HYBRID_SWA_POOL = (
    'sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_hybrid_swa_kv_pool'
)
SWA_POOL_INIT = 'sglang.srt.mem_cache.swa_memory_pool.SWAKVPool.__init__'
MHA_SET_KV = 'sglang.srt.mem_cache.memory_pool.MHATokenToKVPool.set_kv_buffer'
FA_GET_PAGED = 'sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.get_paged_mha_kv_cache'
FA_WITH_KVCACHE = (
    'sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.flash_attn_with_kvcache'
)

_TLS = threading.local()

# Reusable BF16 prefill unpack scratch (capped at 1024 pages = 131,072 tokens = 512 MiB K+V; larger contexts
# allocate transiently for that single FA4 call and free immediately).
_UNPACK_CACHE_MAX_PAGES = 1024
_UNPACK_K_CACHE = [None]
_UNPACK_V_CACHE = [None]


def kv_fp4_enabled() -> bool:
    return os.environ.get('XTIER_INKLING2_KV_FP4', '0') == '1'


def _is_inkling2_model(model_config) -> bool:
    hf_cfg = getattr(model_config, 'hf_config', None)
    archs = getattr(hf_cfg, 'architectures', None) or ()
    return any('Inkling2' in str(a) for a in archs)


@triton.jit
def _e2m1(code):
    # Pure ALU bit-construction of E2M1 -> IEEE-754 FP32 (zero SFU exp2).
    c = code.to(tl.uint32)
    e = (c >> 1) & 3
    m = c & 1
    mag = tl.where(e == 0, tl.where(m != 0, 0x3F000000, 0), ((e + 126) << 23) | (m << 22))
    return (mag | ((c & 8) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def _quant_e2m1(v):
    sign = tl.where(v < 0.0, 8, 0)
    a = tl.abs(v)
    code = tl.where(
        a < 0.25, 0,
        tl.where(
            a < 0.75, 1,
            tl.where(
                a < 1.25, 2,
                tl.where(
                    a < 1.75, 3,
                    tl.where(
                        a < 2.50, 4,
                        tl.where(
                            a < 3.50, 5,
                            tl.where(a < 5.00, 6, 7),
                        ),
                    ),
                ),
            ),
        ),
    )
    return (sign | code).to(tl.uint8)


@triton.jit
def _quant_store_nvfp4_mha_kernel(
    K_SRC, V_SRC, LOC, K_DST, V_DST,
    sk0, sk1, sv0, sv1, dk0, dk1, dv0, dv1,
    max_slots, N,
):
    """Quantize BF16/FP16 K and V [N, H_kv, 128] -> packed NVFP4 [slots, H_kv, 72] uint8 in 1 launch."""
    row = tl.program_id(0)
    h = tl.program_id(1)
    if row >= N:
        return
    slot = tl.load(LOC + row).to(tl.int64)
    if slot < 0 or slot >= max_slots:
        return

    g4 = tl.arange(0, 4)[:, None]
    e16 = tl.arange(0, 16)[None, :]
    off64 = g4 * 16 + e16
    s4 = tl.arange(0, 4)

    # Key
    k_in = K_SRC + row.to(tl.int64) * sk0 + h.to(tl.int64) * sk1
    k_lo = tl.load(k_in + off64).to(tl.float32)
    k_hi = tl.load(k_in + 64 + off64).to(tl.float32)
    sk_lo = tl.maximum(tl.max(tl.abs(k_lo), axis=1) / 6.0, 1e-6).to(tl.float8e4nv)
    sk_hi = tl.maximum(tl.max(tl.abs(k_hi), axis=1) / 6.0, 1e-6).to(tl.float8e4nv)
    ck_lo = _quant_e2m1(k_lo / sk_lo.to(tl.float32)[:, None])
    ck_hi = _quant_e2m1(k_hi / sk_hi.to(tl.float32)[:, None])
    k_out = K_DST + slot * dk0 + h.to(tl.int64) * dk1
    tl.store(k_out + off64, ck_lo | (ck_hi << 4))
    tl.store(k_out + 64 + s4, sk_lo.to(tl.uint8, bitcast=True))
    tl.store(k_out + 68 + s4, sk_hi.to(tl.uint8, bitcast=True))

    # Value
    v_in = V_SRC + row.to(tl.int64) * sv0 + h.to(tl.int64) * sv1
    v_lo = tl.load(v_in + off64).to(tl.float32)
    v_hi = tl.load(v_in + 64 + off64).to(tl.float32)
    sv_lo = tl.maximum(tl.max(tl.abs(v_lo), axis=1) / 6.0, 1e-6).to(tl.float8e4nv)
    sv_hi = tl.maximum(tl.max(tl.abs(v_hi), axis=1) / 6.0, 1e-6).to(tl.float8e4nv)
    cv_lo = _quant_e2m1(v_lo / sv_lo.to(tl.float32)[:, None])
    cv_hi = _quant_e2m1(v_hi / sv_hi.to(tl.float32)[:, None])
    v_out = V_DST + slot * dv0 + h.to(tl.int64) * dv1
    tl.store(v_out + off64, cv_lo | (cv_hi << 4))
    tl.store(v_out + 64 + s4, sv_lo.to(tl.uint8, bitcast=True))
    tl.store(v_out + 68 + s4, sv_hi.to(tl.uint8, bitcast=True))


def write_nvfp4_mha(k_dst: torch.Tensor, v_dst: torch.Tensor, loc: torch.Tensor,
                    cache_k: torch.Tensor, cache_v: torch.Tensor) -> None:
    """Quantize & store cache_k, cache_v [N, H_kv, 128] into NVFP4 buffers [slots, H_kv, 72] uint8."""
    n = loc.shape[0]
    if n == 0:
        return
    h_kv = k_dst.shape[1]
    max_slots = k_dst.shape[0]
    k_src = cache_k if cache_k.stride(-1) == 1 else cache_k.contiguous()
    v_src = cache_v if cache_v.stride(-1) == 1 else cache_v.contiguous()
    _quant_store_nvfp4_mha_kernel[(n, h_kv)](
        k_src, v_src, loc, k_dst, v_dst,
        k_src.stride(0), k_src.stride(1),
        v_src.stride(0), v_src.stride(1),
        k_dst.stride(0), k_dst.stride(1),
        v_dst.stride(0), v_dst.stride(1),
        max_slots, n,
        num_warps=2,
    )


@triton.jit
def _decode_partials_nvfp4_kernel(
    Q,
    K,
    V,
    PageTable,
    SeqLens,
    Bias,
    OPartial,
    MPartial,
    LPartial,
    scale,
    batch,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SP: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    HEADS_K: tl.constexpr,
    QLEN: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    PAGE: tl.constexpr,
    EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Split-KV paged decode/verify partials reading NVFP4 (72 B/head) K/V directly in registers."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0) // HEADS_K
    hk = tl.program_id(0) % HEADS_K
    split = tl.program_id(1)
    splits = tl.num_programs(1)
    seqlen = tl.load(SeqLens + b)
    first = 0
    if WINDOW >= 0:
        first = tl.maximum(seqlen - QLEN - WINDOW, 0)
    tiles = tl.cdiv(seqlen - first // BLOCK_N * BLOCK_N, BLOCK_N)
    per_split = tl.cdiv(tiles, splits)
    tile_lo = first // BLOCK_N + split * per_split
    tile_hi = tl.minimum(tile_lo + per_split, first // BLOCK_N + tiles)

    rows = tl.arange(0, ROWS)
    head_mask = rows < QLEN * GROUP
    token = b * QLEN + rows // GROUP
    qh = hk * GROUP + rows % GROUP
    q_pos = seqlen - QLEN + rows // GROUP

    d64 = tl.arange(0, 64)
    g_idx = d64 // 16
    q_base = Q + token[:, None] * SQ0 + qh[:, None] * SQ1
    q_lo = tl.load(q_base + d64[None, :], head_mask[:, None], other=0)
    q_hi = tl.load(q_base + (64 + d64)[None, :], head_mask[:, None], other=0)

    m = tl.full((ROWS,), float('-inf'), tl.float32)
    l = tl.zeros((ROWS,), tl.float32)
    acc_lo = tl.zeros((ROWS, 64), tl.float32)
    acc_hi = tl.zeros((ROWS, 64), tl.float32)
    cols = tl.arange(0, BLOCK_N)

    # Fast path for global-attention far tiles (dist > EXTENT): skip causal/valid/bias loads & masking.
    if WINDOW < 0 and PAGE % BLOCK_N == 0:
        near_start = tl.maximum(tile_lo, (seqlen - QLEN - EXTENT - BLOCK_N + 1) // BLOCK_N)
        far_hi = tl.minimum(tile_hi, near_start)
    else:
        far_hi = tile_lo

    for tile in range(tile_lo, far_hi):
        page = tl.load(PageTable + b * SP + (tile * BLOCK_N) // PAGE).to(tl.int64)
        slot = (tile * BLOCK_N) % PAGE + cols
        k_base = page * SK0 + slot[:, None] * SK1 + hk * SK2
        v_base = page * SV0 + slot[:, None] * SV1 + hk * SV2

        kb = tl.load(K + k_base + d64[None, :], eviction_policy='evict_first').to(tl.int32)
        ks_lo = tl.load(
            K + k_base + (64 + g_idx)[None, :], eviction_policy='evict_first'
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        ks_hi = tl.load(
            K + k_base + (68 + g_idx)[None, :], eviction_policy='evict_first'
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        k_lo = (_e2m1(kb & 15) * ks_lo).to(q_lo.dtype)
        k_hi = (_e2m1(kb >> 4) * ks_hi).to(q_hi.dtype)

        s = (tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))) * scale
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp2((m - m_new) * 1.4426950408889634)
        p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
        l = l * alpha + tl.sum(p, 1)

        vb = tl.load(V + v_base + d64[None, :], eviction_policy='evict_first').to(tl.int32)
        vs_lo = tl.load(
            V + v_base + (64 + g_idx)[None, :], eviction_policy='evict_first'
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        vs_hi = tl.load(
            V + v_base + (68 + g_idx)[None, :], eviction_policy='evict_first'
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        v_lo = (_e2m1(vb & 15) * vs_lo).to(q_lo.dtype)
        v_hi = (_e2m1(vb >> 4) * vs_hi).to(q_hi.dtype)
        p_cast = p.to(q_lo.dtype)
        acc_lo = acc_lo * alpha[:, None] + tl.dot(p_cast, v_lo)
        acc_hi = acc_hi * alpha[:, None] + tl.dot(p_cast, v_hi)
        m = m_new

    for tile in range(far_hi, tile_hi):
        pos = tile * BLOCK_N + cols
        valid = (pos < seqlen) & (pos >= first)
        if PAGE % BLOCK_N == 0:
            page = tl.load(PageTable + b * SP + (tile * BLOCK_N) // PAGE).to(tl.int64)
            slot = (tile * BLOCK_N) % PAGE + cols
            k_base = page * SK0 + slot[:, None] * SK1 + hk * SK2
            v_base = page * SV0 + slot[:, None] * SV1 + hk * SV2
        else:
            page = tl.load(PageTable + b * SP + pos // PAGE, valid, other=0).to(tl.int64)
            slot = pos % PAGE
            k_base = page[:, None] * SK0 + slot[:, None] * SK1 + hk * SK2
            v_base = page[:, None] * SV0 + slot[:, None] * SV1 + hk * SV2

        kb = tl.load(K + k_base + d64[None, :], valid[:, None], other=0).to(tl.int32)
        ks_lo = tl.load(
            K + k_base + (64 + g_idx)[None, :], valid[:, None], other=0
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        ks_hi = tl.load(
            K + k_base + (68 + g_idx)[None, :], valid[:, None], other=0
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        k_lo = (_e2m1(kb & 15) * ks_lo).to(q_lo.dtype)
        k_hi = (_e2m1(kb >> 4) * ks_hi).to(q_hi.dtype)

        s = (tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))) * scale
        dist = q_pos[:, None] - pos[None, :]
        keep = valid[None, :] & (dist >= 0)
        if WINDOW >= 0:
            keep &= dist <= WINDOW
        bias = tl.load(
            Bias + token[:, None] * SB0 + qh[:, None] * SB1 + dist,
            head_mask[:, None] & keep & (dist < EXTENT),
            other=0,
        ).to(tl.float32)
        s = tl.where(keep, s + bias, float('-inf'))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_ref = tl.where(m_new == float('-inf'), 0.0, m_new)
        alpha = tl.exp2((m - m_ref) * 1.4426950408889634)
        p = tl.exp2((s - m_ref[:, None]) * 1.4426950408889634)
        l = l * alpha + tl.sum(p, 1)

        vb = tl.load(V + v_base + d64[None, :], valid[:, None], other=0).to(tl.int32)
        vs_lo = tl.load(
            V + v_base + (64 + g_idx)[None, :], valid[:, None], other=0
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        vs_hi = tl.load(
            V + v_base + (68 + g_idx)[None, :], valid[:, None], other=0
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        v_lo = (_e2m1(vb & 15) * vs_lo).to(q_lo.dtype)
        v_hi = (_e2m1(vb >> 4) * vs_hi).to(q_hi.dtype)
        p_cast = p.to(q_lo.dtype)
        acc_lo = acc_lo * alpha[:, None] + tl.dot(p_cast, v_lo)
        acc_hi = acc_hi * alpha[:, None] + tl.dot(p_cast, v_hi)
        m = m_new

    stat = (split * batch * QLEN + token) * (HEADS_K * GROUP) + qh
    tl.store(MPartial + stat, m, head_mask)
    tl.store(LPartial + stat, l, head_mask)
    tl.store(OPartial + stat[:, None] * 128 + d64[None, :], acc_lo, head_mask[:, None])
    tl.store(OPartial + stat[:, None] * 128 + (64 + d64)[None, :], acc_hi, head_mask[:, None])


def decode_partials_nvfp4(
    q,
    k_cache,
    v_cache,
    page_table,
    seqused_k,
    rel_bias,
    softmax_scale,
    *,
    window_size_left=None,
):
    """Drop-in NVFP4 replacement for inkling2_decode_attention_sm90.decode_partials."""
    tokens, heads_q, head_dim = q.shape
    assert head_dim == 128
    batch = seqused_k.shape[0]
    qlen = tokens // max(batch, 1)
    assert qlen * batch == tokens
    _, page_size, heads_k, kv_dim = k_cache.shape
    assert kv_dim == NVFP4_KV_DIM
    group = heads_q // heads_k
    span = (
        window_size_left + qlen
        if window_size_left is not None
        else page_table.shape[1] * page_size
    )
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    mult = 4 if window_size_left is None and span >= 65536 else 2
    max_splits = max(1, min(mult * sms // max(1, batch * heads_k), MAX_SPLITS))
    block_n = 128 if qlen == 1 and span >= 128 * max_splits else (64 if span >= 64 * max_splits else 32)
    splits = min(triton.cdiv(span, block_n), max_splits)
    o = torch.empty((splits, tokens, 1, heads_q, head_dim), dtype=torch.float32, device=q.device)
    m = torch.empty((splits, tokens, 1, heads_q), dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    _decode_partials_nvfp4_kernel[(batch * heads_k, splits)](
        q,
        k_cache,
        v_cache,
        page_table,
        seqused_k,
        rel_bias,
        o,
        m,
        l,
        softmax_scale,
        batch,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        page_table.stride(0),
        rel_bias.stride(0),
        rel_bias.stride(1),
        heads_k,
        qlen,
        group,
        max(16, triton.next_power_of_2(qlen * group)),
        page_size,
        rel_bias.shape[-1],
        -1 if window_size_left is None else window_size_left,
        block_n,
        num_warps=4,
        num_stages=2,
        launch_pdl=True,
    )
    return o, m, l


@triton.jit
def _unpack_pages_nvfp4_to_bf16_kernel(
    K_SRC, V_SRC, PAGES, K_DST, V_DST,
    sk0, sk1, sk2, sv0, sv1, sv2,
    dk0, dk1, dk2, dv0, dv1, dv2,
    PAGE: tl.constexpr, HEADS_K: tl.constexpr,
):
    """Unpack U active NVFP4 pages (U, PAGE, H_kv, 72) uint8 -> (U, PAGE, H_kv, 128) BF16."""
    u = tl.program_id(0).to(tl.int64)
    slot = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2).to(tl.int64)
    page = tl.load(PAGES + u).to(tl.int64)

    d64 = tl.arange(0, 64)
    g_idx = d64 // 16

    k_in = K_SRC + page * sk0 + slot * sk1 + h * sk2
    kb = tl.load(k_in + d64).to(tl.int32)
    ks_lo = tl.load(k_in + 64 + g_idx).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    ks_hi = tl.load(k_in + 68 + g_idx).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    k_out = K_DST + u * dk0 + slot * dk1 + h * dk2
    tl.store(k_out + d64, (_e2m1(kb & 15) * ks_lo).to(tl.bfloat16))
    tl.store(k_out + 64 + d64, (_e2m1(kb >> 4) * ks_hi).to(tl.bfloat16))

    v_in = V_SRC + page * sv0 + slot * sv1 + h * sv2
    vb = tl.load(v_in + d64).to(tl.int32)
    vs_lo = tl.load(v_in + 64 + g_idx).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    vs_hi = tl.load(v_in + 68 + g_idx).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    v_out = V_DST + u * dv0 + slot * dv1 + h * dv2
    tl.store(v_out + d64, (_e2m1(vb & 15) * vs_lo).to(tl.bfloat16))
    tl.store(v_out + 64 + d64, (_e2m1(vb >> 4) * vs_hi).to(tl.bfloat16))


def unpack_active_pages_to_bf16(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q=None,
    q=None,
    window_size=(-1, -1),
):
    """Unpack only the active pages needed by this FA4 extend call into a compact BF16 cache + remapped page_table."""
    num_pages, page_size, heads_k, kv_dim = k_cache.shape
    assert kv_dim == NVFP4_KV_DIM
    dev = k_cache.device
    batch = cache_seqlens.shape[0]
    if batch == 0:
        empty = torch.empty((1, page_size, heads_k, 128), dtype=torch.bfloat16, device=dev)
        return empty, empty, page_table

    seqlens = cache_seqlens.to(torch.int32)
    end_p = (seqlens + page_size - 1) // page_size
    max_end_p = int(end_p.max().item()) if end_p.numel() > 0 else 0
    max_end_p = min(max_end_p, page_table.shape[1])
    if max_end_p <= 0:
        empty = torch.zeros((1, page_size, heads_k, 128), dtype=torch.bfloat16, device=dev)
        return empty, empty, torch.zeros((batch, 1), dtype=torch.int32, device=dev)

    pt_slice = page_table[:, :max_end_p]
    col = torch.arange(max_end_p, device=dev, dtype=torch.int32)[None, :]
    valid_p = col < end_p[:, None]

    w_left = window_size[0] if window_size is not None else -1
    if w_left >= 0:
        if cu_seqlens_q is not None:
            q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
        elif q is not None and q.ndim == 4:
            q_lens = torch.full((batch,), q.shape[1], dtype=torch.int32, device=dev)
        else:
            q_lens = None
        if q_lens is not None:
            start_tok = (seqlens - q_lens - w_left).clamp_min(0)
            start_p = start_tok // page_size
            valid_p = valid_p & (col >= start_p[:, None])

    safe_pt = torch.where(valid_p, pt_slice.clamp(0, num_pages - 1), 0)
    uniq, inv = torch.unique(safe_pt, return_inverse=True)
    new_pt = inv.to(torch.int32)
    u = int(uniq.shape[0])

    shape = (u, page_size, heads_k, 128)
    needed = u * page_size * heads_k * 128
    if u <= _UNPACK_CACHE_MAX_PAGES:
        kb = _UNPACK_K_CACHE[0]
        vb = _UNPACK_V_CACHE[0]
        if kb is None or kb.device != dev or kb.numel() < needed:
            cap = max(u, 64) * page_size * heads_k * 128
            kb = torch.empty(cap, dtype=torch.bfloat16, device=dev)
            vb = torch.empty(cap, dtype=torch.bfloat16, device=dev)
            _UNPACK_K_CACHE[0] = kb
            _UNPACK_V_CACHE[0] = vb
        k_tmp = kb[:needed].view(shape)
        v_tmp = vb[:needed].view(shape)
    else:
        # Global Attention long-context unpack (>128K): drop cached SWA shear workspace (~7.5 GiB)
        # and small unpack cache (~0.5 GiB) so the transient BF16 K/V buffer fits comfortably.
        _UNPACK_K_CACHE[0] = None
        _UNPACK_V_CACHE[0] = None
        if w_left < 0:
            try:
                from sglang.kernels.ops.attention.flash_attn.cute.interface import clear_shear_bias_workspace
                clear_shear_bias_workspace()
            except Exception:
                pass
        k_tmp = torch.empty(shape, dtype=torch.bfloat16, device=dev)
        v_tmp = torch.empty(shape, dtype=torch.bfloat16, device=dev)

    _unpack_pages_nvfp4_to_bf16_kernel[(u, page_size, heads_k)](
        k_cache, v_cache, uniq.to(torch.int32), k_tmp, v_tmp,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        k_tmp.stride(0), k_tmp.stride(1), k_tmp.stride(2),
        v_tmp.stride(0), v_tmp.stride(1), v_tmp.stride(2),
        page_size, heads_k,
        num_warps=2,
    )
    return k_tmp, v_tmp, new_pt


# ---------------------------------------------------------------------------
# SGLang hooks (zero SGLang source modifications)
# ---------------------------------------------------------------------------

def _hybrid_swa_config_init(result, self, kvc):
    if (
        not getattr(kvc, 'is_draft_worker', False)
        and _is_inkling2_model(kvc.model_config)
        and kvc.kv_cache_dtype == torch.bfloat16
        and getattr(kvc.model_config, 'head_dim', 0) == 128
    ):
        self._full_per_token = (self._full_per_token * NVFP4_KV_DIM) // BF16_KV_BYTES_PER_HEAD
        self._swa_per_token = (self._swa_per_token * NVFP4_KV_DIM) // BF16_KV_BYTES_PER_HEAD
        self._recompute_cell_size()
        logger.info(
            'xtier: Inkling-2 NVFP4 target KV sizing enabled '
            '(full_per_tok=%d B, swa_per_tok=%d B, draft_cell=%d B, total_cell=%.1f B)',
            self._full_per_token, self._swa_per_token,
            getattr(self, '_draft_cell_size', 0), self._cell_size,
        )
    return result


def _build_hybrid_swa_kv_pool(original, self, *args, **kwargs):
    active = (
        not getattr(self, 'is_draft_worker', False)
        and _is_inkling2_model(self.model_config)
        and self.kv_cache_dtype == torch.bfloat16
        and getattr(self.model_config, 'head_dim', 0) == 128
    )
    if not active:
        return original(self, *args, **kwargs)
    prev = getattr(_TLS, 'alloc_nvfp4_target', False)
    _TLS.alloc_nvfp4_target = True
    try:
        return original(self, *args, **kwargs)
    finally:
        _TLS.alloc_nvfp4_target = prev


def _swa_pool_init(original, self, *args, **kwargs):
    if not getattr(_TLS, 'alloc_nvfp4_target', False):
        return original(self, *args, **kwargs)
    orig_dtype = kwargs.get('dtype', torch.bfloat16)
    orig_head_dim = kwargs.get('head_dim', 128)
    kwargs['dtype'] = torch.uint8
    kwargs['head_dim'] = NVFP4_KV_DIM
    if 'v_head_dim' in kwargs and kwargs['v_head_dim'] is not None:
        kwargs['v_head_dim'] = NVFP4_KV_DIM
    if 'swa_head_dim' in kwargs and kwargs['swa_head_dim'] is not None:
        kwargs['swa_head_dim'] = NVFP4_KV_DIM
    if 'swa_v_head_dim' in kwargs and kwargs['swa_v_head_dim'] is not None:
        kwargs['swa_v_head_dim'] = NVFP4_KV_DIM
    res = original(self, *args, **kwargs)
    # Restore logical dtype/head_dim on the composite SWAKVPool while keeping inner MHA pools as uint8 (72 B/head).
    self.dtype = orig_dtype
    self.head_dim = orig_head_dim
    self.full_kv_pool._xtier_nvfp4_mha = True
    self.swa_kv_pool._xtier_nvfp4_mha = True
    logger.info(
        'xtier: allocated Inkling-2 target NVFP4 SWAKVPool '
        '(full=%d tok x %dL, swa=%d tok x %dL, total=%.2f GiB)',
        self.size, self.full_layer_nums, self.size_swa, self.swa_layer_nums,
        (sum(self.get_kv_size_bytes()) if isinstance(self.get_kv_size_bytes(), tuple) else self.get_kv_size_bytes()) / (1 << 30),
    )
    return res


def _mha_set_kv_buffer(original, self, layer, loc_info, cache_k, cache_v,
                       k_scale=None, v_scale=None, layer_id_override=None, dcp_kv_mask=None):
    if not getattr(self, '_xtier_nvfp4_mha', False):
        return original(self, layer, loc_info, cache_k, cache_v,
                        k_scale=k_scale, v_scale=v_scale,
                        layer_id_override=layer_id_override, dcp_kv_mask=dcp_kv_mask)
    if dcp_kv_mask is not None:
        raise RuntimeError('dcp_kv_mask is not supported for Inkling-2 NVFP4 KV cache.')
    from sglang.srt.mem_cache.memory_pool import maybe_detect_oob, unwrap_write_loc

    loc, _, _ = unwrap_write_loc(loc_info)
    self._check_physical_write_loc(loc_info, 'set_kv_buffer (MHA NVFP4)')
    maybe_detect_oob(loc, 0, self.size + self.page_size, 'set_kv_buffer (MHA NVFP4)')
    layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
    layer_idx = layer_id - self.start_layer
    write_nvfp4_mha(self.k_buffer[layer_idx], self.v_buffer[layer_idx], loc, cache_k, cache_v)
    return None


def _fa_get_paged_mha_kv_cache(original, self, layer, *, head_group_num=1):
    key_cache, value_cache = self.kv_layers.read(layer.layer_id).kv()
    if key_cache.dtype == torch.uint8 and key_cache.shape[-1] == NVFP4_KV_DIM:
        from sglang.srt.layers.attention.flashattention_backend import paged_kv_view

        key_cache = paged_kv_view(key_cache, self.page_size, layer.tp_k_head_num, NVFP4_KV_DIM)
        value_cache = paged_kv_view(value_cache, self.page_size, layer.tp_v_head_num, NVFP4_KV_DIM)
        if head_group_num != 1:
            assert key_cache.is_contiguous() and value_cache.is_contiguous()
            key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num // head_group_num, NVFP4_KV_DIM)
            value_cache = value_cache.view(-1, self.page_size, layer.tp_v_head_num // head_group_num, NVFP4_KV_DIM)
        return key_cache, value_cache
    return original(self, layer, head_group_num=head_group_num)


def _fa_with_kvcache(original, self, *args, ver, **kwargs):
    k_cache = kwargs.get('k_cache')
    if k_cache is not None and k_cache.dtype == torch.uint8 and k_cache.shape[-1] == NVFP4_KV_DIM:
        v_cache = kwargs['v_cache']
        page_table = kwargs['page_table']
        cache_seqlens = kwargs['cache_seqlens']
        k_tmp, v_tmp, new_pt = unpack_active_pages_to_bf16(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q=kwargs.get('cu_seqlens_q'),
            q=kwargs.get('q'),
            window_size=kwargs.get('window_size', (-1, -1)),
        )
        new_kw = dict(kwargs)
        new_kw['k_cache'] = k_tmp
        new_kw['v_cache'] = v_tmp
        new_kw['page_table'] = new_pt
        return original(self, *args, ver=ver, **new_kw)
    return original(self, *args, ver=ver, **kwargs)


def install(registry, hook_type):
    if not kv_fp4_enabled():
        return
    registry.register(HYBRID_SWA_CONFIG_INIT, _hybrid_swa_config_init, hook_type.AFTER)
    registry.register(BUILD_HYBRID_SWA_POOL, _build_hybrid_swa_kv_pool, hook_type.AROUND)
    registry.register(SWA_POOL_INIT, _swa_pool_init, hook_type.AROUND)
    registry.register(MHA_SET_KV, _mha_set_kv_buffer, hook_type.AROUND)
    registry.register(FA_GET_PAGED, _fa_get_paged_mha_kv_cache, hook_type.AROUND)
    registry.register(FA_WITH_KVCACHE, _fa_with_kvcache, hook_type.AROUND)
    logger.info('xtier: registered Inkling-2 target NVFP4 KV cache hooks')
