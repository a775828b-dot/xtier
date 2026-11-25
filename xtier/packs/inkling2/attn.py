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


def install(registry, hook_type):
    # Enable compact verify rel_bias kernel on SM120 before Inkling2Attention.forward inspects the env flag.
    os.environ.setdefault('SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL', '1')
    registry.register(f'{ATTN_CLS}.__init__', _attn_init, hook_type.AFTER)
    registry.register(f'{ATTN_CLS}._project_sources', _project_sources, hook_type.AROUND)
    registry.register(REL_BIAS_PARTIALS, _rel_bias_partials, hook_type.AROUND)
    registry.register(f'{VISION_ATTN}.forward', _vision_attn_forward, hook_type.AROUND)
    logger.info('xtier: Inkling-2 SM120 attention & vision hooks registered')
