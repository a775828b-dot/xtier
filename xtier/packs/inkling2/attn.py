"""SM120 compatibility hooks for Inkling-2 attention and vision encoder.

On SM120 (RTX PRO 5000 / RTX 5090):
1. ``fa4_sm120`` supports ``rel_bias`` with ``tile_m=64, tile_n=128`` and requires ``out.dtype == q.dtype`` (BF16).
   Upstream ``Inkling2Attention`` sets ``out_dtype=torch.float32``, ``verify_wave_splits=True``, and enters the
   SM100-only presheared ``tile_m=128`` verify path unless compact verify is enabled.
2. ``sglang.kernels.ops.attention.inkling2_attn.rel_bias_partials`` dispatches non-SM90 GPUs to SM100-only
   ``flash_fwd_sm100_bias_decode.paged_rel_bias_decode`` (``tcgen05``). SGLang's Triton
   ``inkling2_decode_attention_sm90.decode_partials`` runs natively on SM120 for both decode (``qlen=1``) and DSpark
   verify (``qlen=block_size``).
3. ``Inkling2VisionAttention.forward`` checks ``cap[0] in (9, 10)`` before calling ``flash_attn_varlen_func``, even
   though ``fa4_sm120`` handles non-causal varlen BF16 vision attention natively.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ATTN_CLS = 'sglang.srt.models.inkling_common.inkling2_attn.Inkling2Attention'
REL_BIAS_PARTIALS = 'sglang.kernels.ops.attention.inkling2_attn.rel_bias_partials'
VISION_ATTN = 'sglang.srt.models.inkling2_vision.Inkling2VisionAttention'


def _is_sm120_device(device=None):
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 12


def _attn_init(result, self, *args, **kwargs):
    if not _is_sm120_device():
        return result
    from sglang.srt.models.inkling_common import inkling2_attn as mod

    self.attn.out_dtype = None
    if self.attn_padded is not None:
        self.attn_padded.out_dtype = None
        self.attn_padded.num_splits = 0
    self.verify_wave_splits = False
    # Keep every verify batch (up to bs=32 x 8 tokens) on the SM120-safe compact Triton decode_partials path,
    # avoiding SM100's tile_m=128 presheared rel_bias layout (fa4_sm120 uses tile_m=64).
    if mod.VERIFY_REL_BIAS_MAX_TOKENS < 256:
        mod.VERIFY_REL_BIAS_MAX_TOKENS = 256
    return result


def _rel_bias_partials(original, q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, out, *,
                       window_size_left, verify):
    if k_cache.dtype == torch.uint8 and k_cache.shape[-1] == 72:
        from xtier.packs.inkling2.kv_fp4 import decode_partials_nvfp4

        return decode_partials_nvfp4(
            q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale,
            window_size_left=window_size_left,
        )
    if _is_sm120_device(q.device):
        from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import (
            decode_partials,
        )

        return decode_partials(
            q,
            k_cache,
            v_cache,
            page_table,
            seqused_k,
            rel_bias,
            softmax_scale,
            window_size_left=window_size_left,
        )
    return original(
        q,
        k_cache,
        v_cache,
        page_table,
        seqused_k,
        rel_bias,
        softmax_scale,
        out,
        window_size_left=window_size_left,
        verify=verify,
    )


def _vision_attn_forward(original, self, hidden, positions, lengths, cu_seqlens):
    if not (hidden.is_cuda and _is_sm120_device(hidden.device)):
        return original(self, hidden, positions, lengths, cu_seqlens)
    from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_varlen_func
    from sglang.srt.models.inkling2_vision import _rotary

    shape = (-1, self.num_heads, self.head_dim)
    q, k, v = (
        projection(hidden).reshape(shape)
        for projection in (self.q_proj, self.k_proj, self.v_proj)
    )
    q, k = _rotary(q, k, positions)
    max_length = max(257, sum(lengths) - len(lengths) + 1)
    output, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_length,
        max_seqlen_k=max_length,
        softmax_scale=self.head_dim**-0.5,
        causal=False,
        deterministic=True,
    )
    return self.out_proj(output.reshape(hidden.shape))


def _linear_slice(layer, x, start, end):
    sc = getattr(layer, '_xtier_fp8_scale', None)
    b = getattr(layer, 'bias', None)
    b_sl = None if b is None else b[start:end]
    if sc is not None:
        from xtier.packs.glm5_next import fp8_weights
        w_sl = layer.weight[start:end].contiguous()
        s_sl = sc[start:end].contiguous()
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if getattr(layer, '_xtier_w8a8', False) and x2.shape[0] > fp8_weights.MAX_M:
            y = fp8_weights.gemm_w8a8(x2, w_sl, s_sl, b_sl)
        else:
            y = torch.nn.functional.linear(x2, fp8_weights.dequant(w_sl, s_sl), b_sl)
        return y.view(*x.shape[:-1], end - start)
    return torch.nn.functional.linear(x, layer.weight[start:end], b_sl)


def _project_sources(original, self, sources):
    # On Global Attention prefill with 3 distinct AttentionSources (q, k, v), upstream SGLang stacks
    # (3*M, H) into qkvr -> allocates 3x output (~5.3 GiB at 64K) and discards 2/3 of the GEMM.
    # For prefill (M > 32, outside CUDA graph capture), project q_proj on sources.q and only the
    # k/v slices on sources.k/sources.v: bitwise identical, 2.6x less FLOPs, -3.3 GiB peak VRAM.
    if (
        len({id(sources.q), id(sources.k), id(sources.v)}) == 3
        and sources.q.shape[0] > 32
        and not torch.cuda.is_current_stream_capturing()
        and hasattr(self, 'qkvr')
        and hasattr(self, 'split_sizes')
        and len(self.split_sizes) >= 6
        and getattr(self.qkvr, 'weight', None) is not None
    ):
        s0, s1, s2 = int(self.split_sizes[0]), int(self.split_sizes[1]), int(self.split_sizes[2])
        q_proj = self._project_qkvr(sources.q)
        k_out = _linear_slice(self.qkvr, sources.k, s0, s0 + s1)
        v_out = _linear_slice(self.qkvr, sources.v, s0 + s1, s0 + s1 + s2)
        q, _, _, r, gate, sinks = q_proj
        return q, k_out, v_out, r, gate, sinks
    return original(self, sources)



import triton
import triton.language as tl

REL_LOGITS_PROJ = 'sglang.srt.models.inkling_common.attn.RelLogitsProj'
FA4_FWD = 'sglang.kernels.ops.attention.flash_attn.cute.interface._flash_attn_fwd'


@triton.jit
def _fused_rel_proj_shear_kernel(
    R, PROJ, BIAS,
    CU_Q, CU_K, SEQUSED_Q, SEQUSED_K,
    TOTAL_Q, NUM_SEQS,
    R_STRIDE_T, R_STRIDE_H,
    B_STRIDE_T, B_STRIDE_H,
    WINDOW_LEFT,
    IS_LOCAL: tl.constexpr,
    SINGLE_SEQ: tl.constexpr,
    HAS_SEQ_Q: tl.constexpr,
    HAS_SEQ_K: tl.constexpr,
    ATTN_TILE_M: tl.constexpr,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    D_REL: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    REL_PADDED: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused r[t, h, :D_REL] @ proj[:D_REL, rel_dist] written directly in FA4 SM120 sheared bias layout.
    Eliminates the intermediate [total_q, H, rel_extent] rel_logits tensor (6.45 GiB at 64K) and its
    full GDDR7 write + read round-trip; bitwise identical to einsum + ShearingBias."""
    t = tl.program_id(0)
    c_blk = tl.program_id(1)
    c0 = c_blk * BLOCK_C
    cs = c0 + tl.arange(0, BLOCK_C)

    if SINGLE_SEQ:
        b = 0
        seq_start = tl.load(CU_Q)
        q_end = tl.load(CU_Q + 1)
    else:
        b = 0
        for i in range(0, NUM_SEQS):
            q1 = tl.load(CU_Q + i + 1)
            if t >= q1:
                b = i + 1
        seq_start = tl.load(CU_Q + b)
        q_end = tl.load(CU_Q + b + 1)

    seqlen_q = tl.load(SEQUSED_Q + b) if HAS_SEQ_Q else (q_end - seq_start)
    seqlen_k = tl.load(SEQUSED_K + b) if HAS_SEQ_K else (tl.load(CU_K + b + 1) - tl.load(CU_K + b))

    local_q = t - seq_start
    if local_q >= seqlen_q:
        return

    n_idx_right = local_q + 1 + seqlen_k - seqlen_q
    if IS_LOCAL:
        n_idx_left = tl.maximum(local_q + seqlen_k - seqlen_q - WINDOW_LEFT, 0)
    else:
        n_idx_left = 0
    num_bias_vals = n_idx_right - tl.maximum(n_idx_left, n_idx_right - REL_EXTENT)

    attn_m_block = local_q // ATTN_TILE_M
    m_idx_max = (attn_m_block + 1) * ATTN_TILE_M
    n_idx_blk_right = m_idx_max + seqlen_k - seqlen_q
    attn_n_block_max = tl.minimum((seqlen_k + 127) // 128, (n_idx_blk_right + 127) // 128)

    n_block_for_rel0 = (n_idx_right - 1) // 128
    bias_block_idx_right = 1 + tl.maximum(REL_PADDED // 128 - (attn_n_block_max - n_block_for_rel0), 0)
    bias_idx_right = (bias_block_idx_right - 1) * 128 + ((n_idx_right - 1) % 128) + 1
    bias_idx_left = tl.maximum(0, bias_idx_right - num_bias_vals)

    hs = tl.arange(0, H_PAD)
    h_mask = hs < H
    c_mask = cs < REL_PADDED

    tile_end = c0 + BLOCK_C
    left_val = -float('inf') if IS_LOCAL else 0.0
    if c0 >= bias_idx_right:
        out_val = tl.full((H_PAD, BLOCK_C), -float('inf'), dtype=tl.bfloat16)
    elif tile_end <= bias_idx_left:
        out_val = tl.full((H_PAD, BLOCK_C), left_val, dtype=tl.bfloat16)
    else:
        ds = tl.arange(0, D_REL)
        r_ptrs = R + t.to(tl.int64) * R_STRIDE_T + hs[:, None].to(tl.int64) * R_STRIDE_H + ds[None, :]
        r_tile = tl.load(r_ptrs, mask=h_mask[:, None], other=0.0)
        pb = bias_idx_right - 1 - cs
        pb_valid = (pb >= 0) & (pb < num_bias_vals) & c_mask
        pb_clamped = tl.where(pb_valid, pb, 0)
        p_ptrs = PROJ + ds[:, None] * REL_EXTENT + pb_clamped[None, :]
        p_tile = tl.load(p_ptrs, mask=pb_valid[None, :], other=0.0)
        acc = tl.dot(r_tile, p_tile, out_dtype=tl.float32).to(tl.bfloat16)
        right_mask = cs >= bias_idx_right
        pad_fill = tl.where(right_mask, -float('inf'), left_val).to(tl.bfloat16)
        out_val = tl.where(pb_valid[None, :], acc, pad_fill[None, :])

    b_ptrs = BIAS + t.to(tl.int64) * B_STRIDE_T + hs[:, None].to(tl.int64) * B_STRIDE_H + cs[None, :]
    tl.store(b_ptrs, out_val, mask=h_mask[:, None] & c_mask[None, :])


def _fused_shear_enabled() -> bool:
    return os.environ.get('XTIER_INKLING2_FUSED_SHEAR', '1') != '0'


def _rel_logits_proj_forward(original, self, r_out, log_scaling_tau=None):
    if (
        _fused_shear_enabled()
        and r_out.is_cuda
        and r_out.dtype == torch.bfloat16
        and self.proj.dtype == torch.bfloat16
        and r_out.ndim == 3
        and r_out.shape[0] >= 256
        and r_out.shape[1] <= 64
        and r_out.shape[2] == 16
        and self.rel_extent % 128 == 0
        and not torch.cuda.is_current_stream_capturing()
        and _is_sm120_device(r_out.device)
    ):
        from sglang.srt.environ import envs
        from sglang.srt.models.inkling_common.attn import _apply_log_scaling_tau, is_batch_invariant
        if envs.SGLANG_OPT_USE_INKLING_SHEARED_BIAS.get() and not (
            is_batch_invariant is not None and is_batch_invariant()
        ):
            r_eff = r_out
            post_tau = None
            if log_scaling_tau is not None:
                if getattr(self, '_prescale_tau', True):
                    r_eff = _apply_log_scaling_tau(r_out, log_scaling_tau.view(-1, 1, 1))
                else:
                    post_tau = log_scaling_tau
            if r_eff.stride(-1) != 1:
                r_eff = r_eff.contiguous()
            proj_c = self.proj if self.proj.is_contiguous() else self.proj.contiguous()
            stub = torch.empty((1,), dtype=r_eff.dtype, device=r_eff.device).expand(
                r_eff.shape[0], r_eff.shape[1], self.rel_extent
            )
            stub._xtier_fused_rel = (r_eff, proj_c, post_tau)
            return stub
    return original(self, r_out, log_scaling_tau)


def _materialize_rel(r_eff, proj_c, post_tau):
    out = torch.einsum('thd,de->the', r_eff, proj_c)
    if post_tau is not None:
        from sglang.srt.models.inkling_common.attn import _apply_log_scaling_tau
        out = _apply_log_scaling_tau(out, post_tau.view(-1, 1, 1))
    return out


def _fa4_fwd_fused_shear(original, *args, **kwargs):
    rel_bias = kwargs.get('rel_bias', None)
    payload = getattr(rel_bias, '_xtier_fused_rel', None) if rel_bias is not None else None
    if payload is None:
        return original(*args, **kwargs)
    r_eff, proj_c, post_tau = payload
    cu_q = kwargs.get('cu_seqlens_q', None)
    cu_k = kwargs.get('cu_seqlens_k', None)
    seq_q = kwargs.get('seqused_q', None)
    seq_k = kwargs.get('seqused_k', None)
    causal = kwargs.get('causal', False)
    win_l = kwargs.get('window_size_left', None)
    win_r = kwargs.get('window_size_right', None)
    pack_gqa = kwargs.get('pack_gqa', None)
    mask_mod = kwargs.get('mask_mod', None)
    tile_mn = kwargs.get('tile_mn', None)
    presheared = kwargs.get('rel_bias_presheared', False)
    q = args[0] if len(args) > 0 else kwargs.get('q', None)
    k = args[1] if len(args) > 1 else kwargs.get('k', None)
    actual_pack_gqa = (q.shape[-2] // k.shape[-2] > 1) if (pack_gqa is None and q is not None and k is not None) else bool(pack_gqa)
    if win_l is not None and win_l < 0:
        win_l = None
    if win_r is not None and win_r < 0:
        win_r = None
    can_fuse = (
        post_tau is None
        and not presheared
        and mask_mod is None
        and not actual_pack_gqa
        and causal
        and win_r in (None, 0)
        and cu_q is not None
        and (seq_k is not None or cu_k is not None)
        and (tile_mn is None or tile_mn == (64, 128))
        and not torch.cuda.is_current_stream_capturing()
    )
    if not can_fuse:
        kwargs['rel_bias'] = _materialize_rel(r_eff, proj_c, post_tau)
        return original(*args, **kwargs)

    from sglang.kernels.ops.attention.flash_attn.cute.interface import _round_up_to_tile, _shear_bias_empty
    total_q, h, d_rel = r_eff.shape
    rel_extent = proj_c.shape[1]
    rel_padded = rel_extent + 256
    tile_m = 64
    bias = _shear_bias_empty(
        (_round_up_to_tile(total_q, tile_m), h, rel_padded),
        r_eff.dtype,
        r_eff.device,
    )
    num_seqs = int(cu_q.shape[0]) - 1
    h_pad = triton.next_power_of_2(max(16, h))
    block_c = 256
    _fused_rel_proj_shear_kernel[(total_q, triton.cdiv(rel_padded, block_c))](
        r_eff, proj_c, bias,
        cu_q, cu_k if cu_k is not None else cu_q,
        seq_q if seq_q is not None else cu_q,
        seq_k if seq_k is not None else cu_q,
        total_q, num_seqs,
        r_eff.stride(0), r_eff.stride(1),
        bias.stride(0), bias.stride(1),
        int(win_l) if win_l is not None else 0,
        win_l is not None,
        num_seqs == 1,
        seq_q is not None,
        seq_k is not None,
        tile_m,
        h, h_pad, d_rel, rel_extent, rel_padded, block_c,
        num_warps=4,
    )
    kwargs['rel_bias'] = bias
    kwargs['rel_bias_presheared'] = True
    return original(*args, **kwargs)


def install(registry, hook_type):
    # Enable compact verify rel_bias kernel on SM120 before Inkling2Attention.forward inspects the env flag.
    os.environ.setdefault('SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL', '1')
    registry.register(f'{ATTN_CLS}.__init__', _attn_init, hook_type.AFTER)
    registry.register(f'{ATTN_CLS}._project_sources', _project_sources, hook_type.AROUND)
    registry.register(f'{REL_LOGITS_PROJ}.forward', _rel_logits_proj_forward, hook_type.AROUND)
    registry.register(FA4_FWD, _fa4_fwd_fused_shear, hook_type.AROUND)
    registry.register(REL_BIAS_PARTIALS, _rel_bias_partials, hook_type.AROUND)
    registry.register(f'{VISION_ATTN}.forward', _vision_attn_forward, hook_type.AROUND)
    logger.info('xtier: Inkling-2 SM120 attention & vision hooks registered')
