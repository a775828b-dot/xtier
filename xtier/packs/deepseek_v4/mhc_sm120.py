"""Lossless SM120 prefill hooks for DeepSeek V4.1:
1. ``XTIER_DS_MHC_POST_TRITON``: bitwise-exact Triton ``deepseek_v4_mhc.post`` for ``T > 384`` (60.1 ms + 22.5 GiB
   temporary broadcast spike -> 4.36 ms, ``torch.equal == True``).
2. ``XTIER_DS_MHC_BF16X3``: prepares SGLang's 3-term BF16 exact 24-bit mantissa mHC weights (``split_bf16_hc_weight``)
   on SM120 without turning on ``SGLANG_OPT_DEEPGEMM_HC_PRENORM`` (which would call SM90/SM100-only
   ``tf32_hc_prenorm_gemm`` on unfused boundary layers), speeding fused mHC pre-norm from 5.55 ms to 1.94 ms per
   sublayer (-152 ms per 48K chunk).
3. ``XTIER_DS_MXFP8_AUTOTUNE``: keeps ``b12x`` pinned for decode / small turns (``< 512`` rows) while routing large
   prefill MXFP8 dense linears (``>= 512`` rows) in ``_b12x_or_cutlass_mxfp8_linear`` to autotuned ``cutlass`` and
   extending ``DeepseekV4ForCausalLM.autotune_prefill_kernels`` to ``is_flashinfer_cutlass()`` (1,156 ms on ``b12x`` ->
   815 ms on tuned ``cutlass`` per 40 layers at M=49,152, ``torch.equal == True``).
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MHC_POST = 'sglang.srt.models.deepseek_v4_mhc.post'
MHC_MIX_STATS = 'sglang.srt.models.deepseek_v4_mhc._mix_stats_impl'
REFRESH_MHC = 'sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayerBase.refresh_mhc_norm_weight_cache'
AUTOTUNE_PREFILL = 'sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.autotune_prefill_kernels'
B12X_OR_CUTLASS = 'sglang.srt.layers.quantization.fp8_utils._b12x_or_cutlass_mxfp8_linear'


@triton.jit
def _post_torch_exact(p, x, c0, r0, c1, r1, c2, r2, c3, r3):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 px, m0, m1, m2, m3, s0, s1, s2;
            mul.rn.f32 px, $1, $2;
            mul.rn.f32 m0, $3, $4;
            mul.rn.f32 m1, $5, $6;
            mul.rn.f32 m2, $7, $8;
            mul.rn.f32 m3, $9, $10;
            add.rn.f32 s0, m0, m1;
            add.rn.f32 s1, s0, m2;
            add.rn.f32 s2, s1, m3;
            add.rn.f32 $0, px, s2;
        }
        """,
        constraints='=f,f,f,f,f,f,f,f,f,f,f',
        args=[p, x, c0, r0, c1, r1, c2, r2, c3, r3],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mhc_post_exact_kernel(
    X,
    RES,
    POST,
    COMB,
    OUT,
    M,
    H: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_h * BH + tl.arange(0, BH)
    rmask = rows < M
    cmask = cols < H
    mask = rmask[:, None] & cmask[None, :]

    rows64 = rows.to(tl.int64)
    x = tl.load(X + rows64[:, None] * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(RES + (rows64[:, None] * 4 + 0) * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(RES + (rows64[:, None] * 4 + 1) * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(RES + (rows64[:, None] * 4 + 2) * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(RES + (rows64[:, None] * 4 + 3) * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)

    for i in tl.static_range(4):
        p = tl.load(POST + rows64 * 4 + i, mask=rmask, other=0.0)[:, None]
        c0 = tl.load(COMB + rows64 * 16 + 0 * 4 + i, mask=rmask, other=0.0)[:, None]
        c1 = tl.load(COMB + rows64 * 16 + 1 * 4 + i, mask=rmask, other=0.0)[:, None]
        c2 = tl.load(COMB + rows64 * 16 + 2 * 4 + i, mask=rmask, other=0.0)[:, None]
        c3 = tl.load(COMB + rows64 * 16 + 3 * 4 + i, mask=rmask, other=0.0)[:, None]
        u = _post_torch_exact(p, x, c0, r0, c1, r1, c2, r2, c3, r3).to(tl.bfloat16)
        tl.store(OUT + (rows64[:, None] * 4 + i) * H + cols[None, :], u, mask=mask)


def mhc_post_exact(x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    m, h = x.shape
    out = torch.empty_like(residual)
    bm, bh = 32, 128
    _mhc_post_exact_kernel[(triton.cdiv(m, bm), triton.cdiv(h, bh))](
        x, residual, post_mix, comb, out, m, H=h, BM=bm, BH=bh, num_warps=4, num_stages=2
    )
    return out


def _post_hook(original, cfg, x, residual, post_mix, comb):
    if (
        torch.version.hip is None
        and x.is_cuda
        and x.dim() == 2
        and x.shape[0] > 384
        and cfg.mult == 4
        and x.shape[1] % 128 == 0
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and post_mix.dtype == torch.float32
        and comb.dtype == torch.float32
        and residual.shape == (x.shape[0], 4, x.shape[1])
        and post_mix.shape == (x.shape[0], 4)
        and comb.shape == (x.shape[0], 4, 4)
        and x.is_contiguous()
        and residual.is_contiguous()
        and post_mix.is_contiguous()
        and comb.is_contiguous()
    ):
        return mhc_post_exact(x, residual, post_mix, comb)
    return original(cfg, x, residual, post_mix, comb)


def _refresh_mhc_hook(original, self, *args, **kwargs):
    ret = original(self, *args, **kwargs)
    if (
        self.hc_pre_from_prev_sublayer
        and self._hc_attn_bf16_parts is None
        and self.hc_attn_fn.shape == (24, 20480)
        and getattr(self.config, 'model_type', None) == 'deepseek_v41'
    ):
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        if not is_batch_invariant_mode_enabled():
            from sglang.kernels.ops.layernorm.mhc import split_bf16_hc_weight

            self._hc_attn_bf16_parts = split_bf16_hc_weight(self.hc_attn_fn.data)
            self._hc_ffn_bf16_parts = split_bf16_hc_weight(self.hc_ffn_fn.data)
            self._init_hyper_connections()
            self.__dict__.pop('_can_fuse_attn_mhc', None)
            self.__dict__.pop('_can_fuse_ffn_mhc', None)
    return ret


def _mix_stats_hook(original, hc, x):
    bf16_parts = hc.bf16_parts
    if bf16_parts is not None and x.is_cuda and x.dtype == torch.bfloat16:
        x_flat = x.flatten(1)
        if 4096 <= x_flat.shape[0] <= 65536 and x_flat.is_contiguous():
            from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

            if not is_batch_invariant_mode_enabled():
                from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn_bf16x3

                cfg = hc.cfg
                return hc_mix_stats_sinkhorn_bf16x3(
                    x_flat,
                    bf16_parts,
                    hc.scale,
                    hc.base,
                    cfg.sinkhorn_iters,
                    cfg.rms_eps,
                    cfg.eps,
                    decode_grid=False,
                )
    return original(hc, x)


def _b12x_or_cutlass_hook(original, input, weight, *args, **kwargs):
    rows = input.numel() // input.shape[-1]
    if rows >= int(os.environ.get('XTIER_DS_MXFP8_CUTLASS_MIN_M', '768')):
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_blockscaled_linear

        pin = kwargs.pop('pin_tactic', is_batch_invariant_mode_enabled())
        return flashinfer_mxfp8_blockscaled_linear(
            input, weight, *args, backend='cutlass', pin_tactic=pin, **kwargs
        )
    return original(input, weight, *args, **kwargs)


def _autotune_prefill_cutlass_hook(original, self, num_tokens: int, *, dtype: torch.dtype) -> int:
    tuned = original(self, num_tokens, dtype=dtype)
    if tuned or getattr(self.config, 'model_type', None) != 'deepseek_v41':
        return tuned
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    seen = set()
    for layer in self.model.modules():
        method = getattr(layer, 'quant_method', None)
        if not isinstance(method, Fp8LinearMethod):
            continue
        if not (method.use_mxfp8 or method.block_fp8_as_mxfp8):
            continue
        if method.block_fp8_as_mxfp8 and not getattr(layer, 'block_fp8_mxfp8_ready', False):
            continue
        backend = method.mxfp8_dense_backend
        if backend is None or not backend.is_flashinfer_cutlass():
            continue
        weight = layer.weight
        if weight.numel() == 0:
            continue
        scale = layer.weight_scale_inv_swizzled
        key = (weight.shape, weight.stride(), weight.dtype, scale.shape, scale.stride(), scale.dtype)
        if key in seen:
            continue
        seen.add(key)
        x = torch.zeros((num_tokens, weight.shape[1]), dtype=dtype, device=weight.device)
        method.apply(layer, x)
        del x
    if seen:
        logger.info('xtier: FlashInfer CUTLASS prefill autotune: %d MXFP8 weight layouts at M=%d', len(seen), num_tokens)
    return len(seen)


def install(registry, hook_type):
    cap = torch.cuda.get_device_capability()
    if os.environ.get('XTIER_DS_MHC_POST_TRITON', '1') != '0' and cap[0] >= 8:
        registry.register(MHC_POST, _post_hook, hook_type.AROUND)
    if os.environ.get('XTIER_DS_MHC_BF16X3', '1') != '0' and cap == (12, 0):
        registry.register(REFRESH_MHC, _refresh_mhc_hook, hook_type.AROUND)
        registry.register(MHC_MIX_STATS, _mix_stats_hook, hook_type.AROUND)
    if os.environ.get('XTIER_DS_MXFP8_AUTOTUNE', '1') != '0' and cap[0] >= 10:
        registry.register(AUTOTUNE_PREFILL, _autotune_prefill_cutlass_hook, hook_type.AROUND)
        if cap == (12, 0):
            registry.register(B12X_OR_CUTLASS, _b12x_or_cutlass_hook, hook_type.AROUND)
